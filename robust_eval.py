"""Robust decoding evaluation for the C-CO12 CEBRA+LABEL runners.

For every trained arm (``seed_*/<arm>/cebra.pt`` + ``decoder.pt``) found in the
given run directories, the VALID inputs are attacked under several threat
models and the decoding R2 under attack is reported. The result is the
train-arm x attack matrix (mean +- std over seeds).

Threat models (all L-inf boxes, one perturbation per input window):
    clean          no attack (sanity check against metrics.json)
    constant       |delta| <= eps                          (ordinary PGD)
    noise          |delta| <= coef * sigma_j(local level)  (Acorn-noise-eps)
    gain           x * (1 + g),       |g| <= gain_eps       (Acorn-structured-attack)
    baseline       x + b * s_j,       |b| <= baseline_eps
    gain_baseline  x * (1 + g) + b * s_j

The attacks and the noise model are implemented here (mirroring the two forks),
so checkpoints from ANY fork can be evaluated with the same code. The fork given
by --cebra-dir is only used for its model registry. The encoder and the decoder
run in eval mode (no dropout) during the attack.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import json
import sys
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
CEBRA_DIR = ROOT / 'Acorn-noise-eps'
OUT_ROOT = ROOT / 'ROBUST_EVAL'
ATTACKS = ('clean', 'constant', 'noise', 'gain', 'baseline', 'gain_baseline')
OBJECTIVES = ('decoder_mse', 'embedding')
NOISE_MODELS = ('poisson_gaussian', 'poisson', 'gaussian')
BATCH_SIZE = 512


# ----------------------------------------------------------------------
# Noise model / scales (identical to Acorn-noise-eps and Acorn-structured-attack)
# ----------------------------------------------------------------------
def fit_noise_model(neural, noise_model, relative_floor):
    """Per-neuron ``var_j(level) = a_j * level + b_j`` from first differences."""
    x = neural.double()
    d2 = (x[1:] - x[:-1])**2 / 2
    m = (x[1:] + x[:-1]) / 2
    mean_d2 = d2.mean(dim=0)
    if noise_model == 'gaussian':
        a, b = torch.zeros_like(mean_d2), mean_d2
    elif noise_model == 'poisson':
        a, b = torch.ones_like(mean_d2), torch.zeros_like(mean_d2)
    else:
        mean_m = m.mean(dim=0)
        var_m = ((m - mean_m)**2).mean(dim=0)
        cov = ((m - mean_m) * (d2 - mean_d2)).mean(dim=0)
        a = torch.where(var_m > 1e-12, cov / var_m.clamp(min=1e-12),
                        torch.zeros_like(var_m)).clamp(min=0)
        b = (mean_d2 - a * mean_m).clamp(min=0)
    noise_std = mean_d2.sqrt()
    active = noise_std[noise_std > 0]
    reference = float(active.median()) if active.numel() > 0 else 1.0
    return a.float(), b.float(), max(relative_floor * reference, 1e-8)


def local_level(windows, width):
    """Moving average over the time axis of ``(batch, neurons, time)`` windows."""
    width = min(width, windows.size(-1))
    if width % 2 == 0:
        width -= 1
    if width <= 1:
        return windows
    return torch.nn.functional.avg_pool1d(windows, kernel_size=width, stride=1,
                                          padding=width // 2,
                                          count_include_pad=False)


def neuron_scale(neural, relative_floor):
    """Per-neuron std, floored at ``relative_floor`` x the median std."""
    std = neural.float().std(dim=0, unbiased=False)
    active = std[std > 0]
    reference = float(active.median()) if active.numel() > 0 else 1.0
    return std.clamp(min=max(relative_floor * reference, 1e-8))


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
class TwoLayerMLP(nn.Module):
    """Same layout as the runners' decoder, so ``decoder.pt`` loads directly."""

    def __init__(self, dim, out, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return self.net(x)


def load_cebra_fork(path):
    path = path.expanduser().resolve()
    if not (path / 'cebra' / '__init__.py').is_file():
        raise FileNotFoundError(f'CEBRA checkout not found: {path}. Use --cebra-dir /path/to/fork')
    for name in list(sys.modules):
        if name == 'cebra' or name.startswith('cebra.'):
            del sys.modules[name]
    sys.path.insert(0, str(path))
    import cebra
    imported = Path(cebra.__file__).resolve()
    if path not in imported.parents:
        raise RuntimeError(f'Wrong import: {imported}; expected under {path}')
    print('Using fork (model registry only):', imported, flush=True)
    return cebra


def load_encoder(cebra, path, device):
    """Rebuild the encoder from a sklearn-backend checkpoint without cebra.CEBRA(**args).

    CEBRA.load would call the constructor with the fork-specific adv_* arguments,
    which fails across forks; only the architecture and the weights are needed.
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)  # own trusted files
    args, state = checkpoint['args'], checkpoint['state']
    if state.get('num_sessions_') is not None:
        raise ValueError(f'{path}: multi-session checkpoints are not supported.')
    model = cebra.models.init(args['model_architecture'],
                              num_neurons=state['n_features_in_'],
                              num_units=args['num_hidden_units'],
                              num_output=args['output_dimension'])
    model.load_state_dict(checkpoint['state_dict']['model'])
    return model.to(device).eval()


def load_decoder(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    decoder = TwoLayerMLP(checkpoint['input_dim'], checkpoint['output_dim'],
                          checkpoint['hidden'], checkpoint['dropout'])
    decoder.load_state_dict(checkpoint['state_dict'])
    return decoder.to(device).eval()


def encode(encoder, windows):
    """``(batch, neurons, window)`` -> ``(batch, latent)``."""
    return encoder(windows).squeeze(-1)


# ----------------------------------------------------------------------
# Attacks
# ----------------------------------------------------------------------
def pgd_box(per_sample_loss, apply, bounds, steps, alpha_ratio, restarts):
    """Sign-gradient PGD over variables constrained to ``|p| <= bound`` (element-wise).

    ``bounds`` are full-shape tensors, one per attack variable; ``apply`` maps the
    variables to the adversarial input. Keeps the per-sample worst restart.
    """
    best_x = best_loss = None
    for _ in range(restarts):
        params = [torch.empty_like(b).uniform_(-1.0, 1.0) * b for b in bounds]
        for _ in range(steps):
            for p in params:
                p.requires_grad_(True)
            loss = per_sample_loss(apply(*params))
            grads = torch.autograd.grad(loss.sum(), params)
            with torch.no_grad():
                params = [torch.max(torch.min(p + alpha_ratio * b * g.sign(), b), -b)
                          for p, g, b in zip(params, grads, bounds)]
        with torch.no_grad():
            x = apply(*params)
            loss = per_sample_loss(x)
            if best_x is None:
                best_x, best_loss = x, loss
            else:
                better = loss > best_loss
                best_x[better] = x[better]
                best_loss = torch.where(better, loss, best_loss)
    return best_x.detach()


def attack_windows(kind, budget, x0, per_sample_loss, stats, args):
    """Adversarial version of the clean windows ``x0`` for one threat model."""
    if kind == 'clean':
        return x0
    num, neurons, _ = x0.shape
    run = lambda apply, bounds: pgd_box(per_sample_loss, apply, bounds, args.steps,
                                        args.alpha_ratio, args.restarts)
    scale = stats['scale'].view(1, -1, 1).expand(num, neurons, 1).contiguous()
    per_neuron = lambda value: torch.full((num, neurons, 1), value, device=x0.device)
    if kind == 'constant':
        return run(lambda d: x0 + d, [torch.full_like(x0, budget)])
    if kind == 'noise':
        level = local_level(x0, args.noise_smoothing).clamp(min=0)
        sigma = (stats['noise_a'].view(1, -1, 1) * level +
                 stats['noise_b'].view(1, -1, 1)).sqrt().clamp(min=stats['noise_floor'])
        return run(lambda d: x0 + d, [budget * sigma])
    if kind == 'gain':
        return run(lambda g: x0 * (1 + g), [per_neuron(budget)])
    if kind == 'baseline':
        return run(lambda b: x0 + b, [budget * scale])
    if kind == 'gain_baseline':
        gain_eps, baseline_eps = budget
        return run(lambda g, b: x0 * (1 + g) + b,
                   [per_neuron(gain_eps), baseline_eps * scale])
    raise ValueError(kind)


def threat_list(args):
    """(attack, budget) pairs to evaluate."""
    pairs = []
    for kind in args.attacks:
        if kind == 'clean':
            pairs.append((kind, None))
        elif kind == 'constant':
            pairs += [(kind, e) for e in args.epsilon]
        elif kind == 'noise':
            pairs += [(kind, c) for c in args.noise_coef]
        elif kind == 'gain':
            pairs += [(kind, g) for g in args.gain_epsilon]
        elif kind == 'baseline':
            pairs += [(kind, b) for b in args.baseline_epsilon]
        elif kind == 'gain_baseline':
            pairs += [(kind, (g, b)) for g, b in zip(args.gain_epsilon, args.baseline_epsilon)]
    return pairs


def threat_name(kind, budget):
    if budget is None:
        return kind
    if isinstance(budget, tuple):
        return f'{kind}@{budget[0]:g}/{budget[1]:g}'
    return f'{kind}@{budget:g}'


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def r2_per_output(y_true, y_pred):
    ss_res = ((y_true - y_pred)**2).sum(dim=0)
    ss_tot = ((y_true - y_true.mean(dim=0))**2).sum(dim=0)
    return (1 - ss_res / ss_tot).tolist()


def evaluate(encoder, decoder, x_pad, y, window, kind, budget, stats, args):
    """Decode all valid time points with every window attacked independently."""
    device = x_pad.device
    offsets = torch.arange(window, device=device)
    y_std = stats['y_std']
    preds, abs_delta, rel_delta, shift = [], [], [], []
    for start in range(0, len(y), args.batch_size):
        index = torch.arange(start, min(start + args.batch_size, len(y)), device=device)
        x0 = x_pad[index[:, None] + offsets].transpose(1, 2).contiguous()
        target = y[index]
        with torch.no_grad():
            z0 = encode(encoder, x0)
        if args.objective == 'decoder_mse':
            # Label-std normalised MSE, so every output counts like it does in mean R2.
            per_sample_loss = lambda x: (((decoder(encode(encoder, x)) - target) / y_std)**2).mean(dim=1)
        else:
            per_sample_loss = lambda x: ((encode(encoder, x) - z0)**2).sum(dim=1)
        x_adv = attack_windows(kind, budget, x0, per_sample_loss, stats, args)
        with torch.no_grad():
            z = encode(encoder, x_adv)
            preds.append(decoder(z))
            delta = (x_adv - x0).abs()
            abs_delta.append(delta.mean(dim=(1, 2)))
            rel_delta.append((delta / stats['scale'].view(1, -1, 1)).mean(dim=(1, 2)))
            shift.append((z - z0).norm(dim=1))
    pred = torch.cat(preds)
    per_output = r2_per_output(y, pred)
    return dict(valid_mean_r2=float(np.mean(per_output)), valid_r2_per_output=per_output,
                mean_abs_delta=float(torch.cat(abs_delta).mean()),
                mean_abs_delta_over_std=float(torch.cat(rel_delta).mean()),
                mean_embedding_shift=float(torch.cat(shift).mean()))


def load_data(data_dir, session):
    path = data_dir.expanduser() / f'{session}.npz'
    if not path.is_file():
        raise FileNotFoundError(f'Dataset not found: {path}')
    with np.load(path, allow_pickle=False) as data:
        arrays = [np.asarray(data[k], dtype=np.float32)
                  for k in ('train_data', 'valid_data', 'train_label', 'valid_label')]
    x_train, x_valid, y_train, y_valid = arrays
    if y_train.ndim == 1:
        y_train, y_valid = y_train[:, None], y_valid[:, None]
    return path, x_train, x_valid, y_train, y_valid


def find_arms(run_dirs, labels):
    """[(label, seed, arm, folder)] for every finished arm in the run directories."""
    found = []
    for run_dir, label in zip(run_dirs, labels):
        for ckpt in sorted(run_dir.glob('seed_*/*/cebra.pt')):
            folder = ckpt.parent
            if not (folder / 'decoder.pt').is_file():
                print(f'SKIP (no decoder.pt): {folder}', flush=True)
                continue
            seed = int(folder.parent.name.split('_', 1)[1])
            found.append((label, seed, folder.name, folder))
    if not found:
        raise FileNotFoundError('No seed_*/<arm>/cebra.pt + decoder.pt found in the run directories.')
    return found


def run_labels(run_dirs):
    labels = [d.parent.name for d in run_dirs]
    if len(set(labels)) < len(labels):
        labels = [f'{d.parent.name}/{d.name}' for d in run_dirs]
    return labels


def summarize(rows, threats, out):
    summary = {}
    keys = sorted({(r['run'], r['arm']) for r in rows})
    names = [threat_name(k, b) for k, b in threats]
    widths = [max(len(n), 20) + 2 for n in names]
    label_width = max(len(f'{run}:{arm}') for run, arm in keys) + 2
    print('\nVALID mean R2 under attack (mean +- std over seeds)', flush=True)
    print(f"{'run:arm':<{label_width}}" + ''.join(f'{n:>{w}}' for n, w in zip(names, widths)), flush=True)
    for run, arm in keys:
        line, item = f'{run + ":" + arm:<{label_width}}', {}
        for name, width in zip(names, widths):
            scores = np.array([r['valid_mean_r2'] for r in rows
                               if r['run'] == run and r['arm'] == arm and r['threat'] == name])
            std = float(scores.std(ddof=1)) if len(scores) > 1 else None
            item[name] = dict(mean=float(scores.mean()), sample_std=std, n_seeds=len(scores))
            cell = f'{scores.mean():.4f}' + (f' +-{std:.3f}' if std is not None else '')
            line += f'{cell:>{width}}'
        summary[f'{run}:{arm}'] = item
        print(line, flush=True)
    save_json(out / 'summary.json', summary)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dirs', nargs='+', type=Path,
                        help='Runner output folders (the ones holding run_config.json and seed_*/).')
    parser.add_argument('--cebra-dir', type=Path, default=CEBRA_DIR)
    parser.add_argument('--data-dir', type=Path, default=None, help='Default: from run_config.json.')
    parser.add_argument('--session', default=None, help='Default: from run_config.json.')
    parser.add_argument('--out-root', type=Path, default=OUT_ROOT)
    parser.add_argument('--attacks', nargs='+', choices=ATTACKS, default=list(ATTACKS))
    parser.add_argument('--objective', choices=OBJECTIVES, default='decoder_mse',
                        help='decoder_mse: maximise the (label-std normalised) decoding error; '
                             'embedding: maximise the embedding displacement (decoder-free).')
    parser.add_argument('--epsilon', nargs='+', type=float, default=[5.0], help='constant budgets')
    parser.add_argument('--noise-coef', nargs='+', type=float, default=[1.0], help='noise budgets (noise std units)')
    parser.add_argument('--noise-model', choices=NOISE_MODELS, default='poisson_gaussian')
    parser.add_argument('--noise-smoothing', type=int, default=5)
    parser.add_argument('--noise-floor', type=float, default=0.1)
    parser.add_argument('--gain-epsilon', nargs='+', type=float, default=[0.1])
    parser.add_argument('--baseline-epsilon', nargs='+', type=float, default=[0.5],
                        help='in units of the per-neuron train std; gain_baseline zips it with --gain-epsilon')
    parser.add_argument('--baseline-floor', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--alpha-ratio', type=float, default=None, help='Step / budget. Default: 2.5 / steps.')
    parser.add_argument('--restarts', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--attack-seed', type=int, default=0)
    parser.add_argument('--device', default='cuda_if_available')
    args = parser.parse_args()
    if args.alpha_ratio is None:
        args.alpha_ratio = 2.5 / args.steps
    if min(args.steps, args.restarts, args.batch_size) < 1 or args.alpha_ratio <= 0:
        parser.error('steps, restarts, batch-size and alpha-ratio must be positive.')
    budgets = args.epsilon + args.noise_coef + args.gain_epsilon + args.baseline_epsilon
    if not np.isfinite(budgets).all() or min(budgets) < 0:
        parser.error('Budgets must be finite and nonnegative.')
    if 'gain_baseline' in args.attacks and len(args.gain_epsilon) != len(args.baseline_epsilon):
        parser.error('gain_baseline zips --gain-epsilon with --baseline-epsilon; give them the same length.')
    for run_dir in args.run_dirs:
        if not run_dir.is_dir():
            parser.error(f'Not a directory: {run_dir}')
    return args


def main():
    args = parse_args()
    run_dirs = [d.expanduser().resolve() for d in args.run_dirs]
    config_path = run_dirs[0] / 'run_config.json'
    run_config = json.loads(config_path.read_text())['args'] if config_path.is_file() else {}
    session = args.session or run_config.get('session')
    data_dir = args.data_dir or (Path(run_config['data_dir']) if 'data_dir' in run_config else None)
    if session is None or data_dir is None:
        raise ValueError('Give --session and --data-dir (no run_config.json found).')
    for run_dir in run_dirs[1:]:
        other = run_dir / 'run_config.json'
        if other.is_file() and json.loads(other.read_text())['args'].get('session') != session:
            raise ValueError(f'{run_dir} was trained on another session than {session}.')

    if args.device == 'cuda_if_available':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    cebra = load_cebra_fork(args.cebra_dir)
    npz_path, x_train, x_valid, y_train, y_valid = load_data(data_dir, session)
    print('DATA:', npz_path, '| shapes:', x_train.shape, x_valid.shape, y_train.shape, y_valid.shape, flush=True)

    x_train_t = torch.from_numpy(x_train)
    noise_a, noise_b, noise_floor = fit_noise_model(x_train_t, args.noise_model, args.noise_floor)
    stats = dict(noise_a=noise_a.to(device), noise_b=noise_b.to(device), noise_floor=noise_floor,
                 scale=neuron_scale(x_train_t, args.baseline_floor).to(device),
                 y_std=torch.from_numpy(y_train.std(axis=0)).clamp(min=1e-8).to(device))
    y = torch.from_numpy(y_valid).to(device)
    threats = threat_list(args)
    arms = find_arms(run_dirs, run_labels(run_dirs))
    print(f'{len(arms)} trained arms x {len(threats)} threats:', [threat_name(*t) for t in threats], flush=True)
    print(f'objective={args.objective} steps={args.steps} alpha_ratio={args.alpha_ratio:g} '
          f'restarts={args.restarts} | train std (median over neurons)={float(stats["scale"].median()):.4f}',
          flush=True)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out = args.out_root.expanduser() / f'{session}_{args.objective}_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'eval_config.json', dict(
        args={k: (str(v) if isinstance(v, Path) else [str(p) for p in v] if k == 'run_dirs' else v)
              for k, v in vars(args).items()},
        data=str(npz_path), threats=[threat_name(*t) for t in threats],
        arms=[dict(run=r, seed=s, arm=a, folder=str(f)) for r, s, a, f in arms]))
    print('OUTPUT:', out, flush=True)

    rows = []
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['run', 'seed', 'arm', 'threat', 'valid_mean_r2', 'mean_abs_delta',
                         'mean_abs_delta_over_std', 'mean_embedding_shift'] +
                        [f'valid_r2_output_{i}' for i in range(y.shape[1])])
        for run, seed, arm, folder in arms:
            encoder = load_encoder(cebra, folder / 'cebra.pt', device)
            decoder = load_decoder(folder / 'decoder.pt', device)
            for p in list(encoder.parameters()) + list(decoder.parameters()):
                p.requires_grad_(False)
            offset = encoder.get_offset()
            window = offset.left + offset.right
            x_pad = torch.from_numpy(np.pad(x_valid, ((offset.left, offset.right - 1), (0, 0)),
                                            mode='edge')).to(device)
            reference = None
            metrics_path = folder / 'metrics.json'
            if metrics_path.is_file():
                reference = json.loads(metrics_path.read_text()).get('valid_mean_r2')
            print(f'\n{run} | seed {seed} | {arm}', flush=True)
            for kind, budget in threats:
                torch.manual_seed(args.attack_seed)
                result = evaluate(encoder, decoder, x_pad, y, window, kind, budget, stats, args)
                name = threat_name(kind, budget)
                row = dict(run=run, seed=seed, arm=arm, threat=name, **result)
                rows.append(row)
                writer.writerow([run, seed, arm, name, result['valid_mean_r2'], result['mean_abs_delta'],
                                 result['mean_abs_delta_over_std'], result['mean_embedding_shift']] +
                                result['valid_r2_per_output'])
                handle.flush()
                note = ''
                if kind == 'clean' and reference is not None:
                    note = f'  (metrics.json: {reference:.6f}, diff {abs(result["valid_mean_r2"] - reference):.1e})'
                print(f'  {name:<24} R2={result["valid_mean_r2"]:.4f}  mean|delta|={result["mean_abs_delta"]:.4f}'
                      f'  (/std={result["mean_abs_delta_over_std"]:.3f})'
                      f'  emb shift={result["mean_embedding_shift"]:.4f}{note}', flush=True)
            del encoder, decoder, x_pad
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    save_json(out / 'results.json', rows)
    summarize(rows, threats, out)
    print('Saved:', out, flush=True)


if __name__ == '__main__':
    main()

"""Margin-cap sweep on the 10 random Perich sessions: does a larger epsilon give a larger R2?

Trains ONLY the margin arm of Acorn-margin-eps with adv_margin_max_std = 4 and 6 (the cap of
the per-sample budget, in units of the median neuron std), with exactly the settings of the
previous ACORN_MARGIN_VS_FIXED_RANDOM10 runs. fixed_eps_3 and margin with cap 2 are NOT
re-run: their summary.csv (--previous-summary) is read and put in the same table.

Needs only the Acorn-margin-eps fork next to this file (or --cebra-dir).
Evaluation: clean-input VALID R2 over all label columns, as in the previous runs.

Output: <out>/summary.txt (send this one), summary.csv, results.json and the checkpoints.
Re-running with --out <same folder> resumes (finished runs are skipped).
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
from datetime import datetime
import argparse
import csv
import gc
import inspect
import json
import random
import sys
import time
import numpy as np
import torch
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path('/data/hossein/mm_project/perich_data_valid_final_raw')
OUT_ROOT = ROOT / 'ACORN_MARGIN_CAP_SWEEP'
PREVIOUS_SUMMARY = ROOT / 'ACORN_MARGIN_VS_FIXED_RANDOM10' / 'selection_seed_42_n10' / 'summary.csv'
SESSIONS = ('C-CO11', 'C-CO21', 'C-CO22', 'C-CO25', 'C-CO35',
            'C-CO38', 'C-CO41', 'M-CO19', 'M-CO4', 'M-RT2')
CAPS = (4.0, 6.0)
SEEDS = (42,)

# Same settings as the previous fixed_eps_3 / margin runs.
ITERATIONS = 3000
BATCH_SIZE = 2048
ARCHITECTURE = 'offset36-model-more-dropout'
LATENT_DIM = 64
ENCODER_HIDDEN = 64
TEMPERATURE = 0.4
ENCODER_LR = 3e-4
ATTACK_STEPS = 10
MARGIN_REFINE_STEPS = 3
DECODER_EPOCHS = 2500
DECODER_HIDDEN = 64
DECODER_DROPOUT = 0.4
DECODER_LR = 1e-3
DECODER_SEED_OFFSET = 10000
PREDICT_BATCH_SIZE = 8192


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_fork(path):
    path = path.expanduser().resolve()
    if not (path / 'cebra' / '__init__.py').is_file():
        raise FileNotFoundError(f'Fork not found: {path}. Place Acorn-margin-eps beside this script or use --cebra-dir.')
    for name in list(sys.modules):
        if name == 'cebra' or name.startswith('cebra.'):
            del sys.modules[name]
    sys.path.insert(0, str(path))
    import cebra
    actual = Path(cebra.__file__).resolve()
    if path not in actual.parents:
        raise RuntimeError(f'Wrong CEBRA imported: {actual}; expected under {path}')
    missing = {'adv_epsilon_mode', 'adv_margin_max_std'} - set(inspect.signature(cebra.CEBRA.__init__).parameters)
    if missing:
        raise RuntimeError(f'{path} is not Acorn-margin-eps (missing {sorted(missing)}).')
    print('Using fork:', actual, flush=True)
    return cebra


def load_session(data_dir, session):
    path = data_dir.expanduser() / f'{session}.npz'
    keys = ('train_data', 'valid_data', 'train_label', 'valid_label')
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in keys if k not in data.files]
        if missing:
            raise KeyError(f'{path}: missing {missing}')
        xtr, xva, ytr, yva = [np.array(data[k], dtype=np.float32, order='C', copy=True) for k in keys]
    if ytr.ndim == 1:
        ytr, yva = ytr[:, None], yva[:, None]
    for key, arr in zip(keys, (xtr, xva, ytr, yva)):
        if arr.ndim != 2 or min(arr.shape) < 1 or not np.isfinite(arr).all():
            raise ValueError(f'{path}: {key} must be a finite, nonempty 2D array')
    if len(xtr) != len(ytr) or len(xva) != len(yva) or xtr.shape[1] != xva.shape[1]:
        raise ValueError(f'{path}: inconsistent shapes')
    return xtr, xva, ytr, yva


def arm_name(max_std):
    return f'margin_std{max_std:g}'


def encoder_config(max_std, args):
    # Identical to the previous margin arm except adv_margin_max_std.
    return dict(batch_size=BATCH_SIZE, temperature=TEMPERATURE, temperature_mode='constant', distance='cosine',
                model_architecture=ARCHITECTURE, time_offsets=1, max_iterations=args.iterations,
                output_dimension=LATENT_DIM, num_hidden_units=ENCODER_HIDDEN, learning_rate=ENCODER_LR,
                pad_before_transform=True, hybrid=False, training_mode='adversarial', attack_norm='linf',
                adv_epsilon=3.0, adv_alpha=0.6, adv_steps=ATTACK_STEPS,  # epsilon/alpha unused in margin mode
                adv_epsilon_mode='margin', adv_margin_epsilon_max=None, adv_margin_max_std=max_std,
                adv_margin_refine_steps=MARGIN_REFINE_STEPS, adv_margin_eval_mode=True,
                device=args.device, verbose=True)


def build_decoder(in_dim, out_dim, device):
    # Same layout as the previous runs.
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, DECODER_HIDDEN),
        torch.nn.LayerNorm(DECODER_HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Dropout(DECODER_DROPOUT),
        torch.nn.Linear(DECODER_HIDDEN, out_dim),
    ).to(device)


def train_decoder(ztr, ytr, seed, device, epochs):
    seed_all(seed)
    decoder = build_decoder(ztr.shape[1], ytr.shape[1], device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR)
    z = torch.as_tensor(ztr, dtype=torch.float32, device=device)
    y = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    decoder.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(decoder(z), y)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite decoder MSE at epoch {epoch}')
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 500 == 0 or epoch == epochs:
            print(f'  decoder {epoch}/{epochs}: train MSE={float(loss):.6f}', flush=True)
    return decoder.eval()


def predict(decoder, z, device):
    with torch.no_grad():
        return np.concatenate([decoder(torch.as_tensor(z[s:s + PREDICT_BATCH_SIZE], dtype=torch.float32,
                                                       device=device)).cpu().numpy()
                               for s in range(0, len(z), PREDICT_BATCH_SIZE)])


def score(y, pred):
    per_output = np.asarray(r2_score(y, pred, multioutput='raw_values'), dtype=float)
    return float(per_output.mean()), per_output.tolist()


def margin_log_summary(log, max_std):
    """Per-sample budget statistics: last batch (as in the previous runs) and last / first 5% of steps."""
    n = len(log.get('adv_eps_mean', []))
    if n == 0:
        return {}
    part = max(1, n // 20)
    stats = {}
    for key in ('adv_eps_mean', 'adv_eps_median', 'adv_frac_chance', 'adv_frac_capped'):
        values = log[key]
        stats[f'{key}_last'] = float(values[-1])
        stats[f'{key}_end'] = float(np.mean(values[-part:]))
        stats[f'{key}_start'] = float(np.mean(values[:part]))
    cap = float(log['adv_eps_cap'][-1])
    stats['adv_eps_cap_last'] = cap
    stats['median_std'] = cap / max_std
    return stats


def run_one(cebra, session, max_std, seed, data, folder, args, device):
    xtr, xva, ytr, yva = data
    folder.mkdir(parents=True, exist_ok=True)
    config = encoder_config(max_std, args)
    print(f'\n{"=" * 80}\nTRAIN {session} | {arm_name(max_std)} | seed={seed}\n{"=" * 80}', flush=True)
    seed_all(seed)
    started = time.perf_counter()
    model = cebra.CEBRA(**config)
    model.fit(xtr, ytr)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    encoder_seconds = time.perf_counter() - started
    margin_stats = margin_log_summary(model.solver_.log, max_std)
    model.save(str(folder / 'cebra.pt'), backend='sklearn')
    ztr = np.ascontiguousarray(model.transform(xtr), dtype=np.float32)
    zva = np.ascontiguousarray(model.transform(xva), dtype=np.float32)
    for name, z, y in (('train', ztr, ytr), ('valid', zva, yva)):
        if z.shape != (len(y), LATENT_DIM) or not np.isfinite(z).all():
            raise RuntimeError(f'{name} embeddings invalid: {z.shape}')
    del model
    cleanup()
    decoder = train_decoder(ztr, ytr, seed + DECODER_SEED_OFFSET, device, args.decoder_epochs)
    r2_train, train_per_output = score(ytr, predict(decoder, ztr, device))
    r2_valid, valid_per_output = score(yva, predict(decoder, zva, device))
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in decoder.state_dict().items()},
                'input_dim': LATENT_DIM, 'output_dim': int(ytr.shape[1]), 'hidden': DECODER_HIDDEN,
                'dropout': DECODER_DROPOUT, 'layout': 'Linear-LayerNorm-ReLU-Dropout-Linear',
                'seed': seed + DECODER_SEED_OFFSET}, folder / 'decoder.pt')
    row = dict(session=session, arm=arm_name(max_std), max_std=max_std, seed=seed, neurons=int(xtr.shape[1]),
               train_samples=len(xtr), valid_samples=len(xva), r2_train=r2_train, r2_valid=r2_valid,
               train_per_output=train_per_output, valid_per_output=valid_per_output,
               encoder_seconds=encoder_seconds, config=config, **margin_stats)
    save_json(folder / 'metrics.json', row)  # written last: marks the run as finished
    print(f'{session} {arm_name(max_std)}: TRAIN R2={r2_train:.4f} | VALID R2={r2_valid:.4f} | mean eps (last batch) '
          f'{margin_stats.get("adv_eps_mean_last", float("nan")):.3f} = '
          f'{margin_stats.get("adv_eps_mean_last", float("nan")) / margin_stats.get("median_std", float("nan")):.2f} x std'
          f' | at cap {margin_stats.get("adv_frac_capped_last", float("nan")):.1%} | {encoder_seconds / 60:.1f} min',
          flush=True)
    del decoder
    cleanup()
    return row


def read_previous(path):
    """{(session, arm, seed): row} from the previous fixed_eps_3 / margin summary.csv (margin -> margin_std2)."""
    if path is None or not path.is_file():
        return {}
    rows = {}
    with path.open(newline='') as handle:
        for r in csv.DictReader(handle):
            arm = 'margin_std2' if r['arm'] == 'margin' else r['arm']
            out = dict(session=r['session'], arm=arm, seed=int(r['seed']), r2_valid=float(r['r2_valid']),
                       r2_train=float(r['r2_train']))
            if arm == 'margin_std2' and r.get('adv_eps_mean_last'):
                cap = float(r['adv_eps_cap_last'])
                out.update(adv_eps_mean_last=float(r['adv_eps_mean_last']), adv_eps_cap_last=cap,
                           adv_frac_capped_last=float(r['adv_frac_capped_last']),
                           adv_frac_chance_last=float(r['adv_frac_chance_last']), median_std=cap / 2.0)
            rows[(out['session'], arm, out['seed'])] = out
    return rows


def paired(rows, sessions, seeds, arm, reference):
    diffs = [rows[(s, arm, k)]['r2_valid'] - rows[(s, reference, k)]['r2_valid']
             for s in sessions for k in seeds if (s, arm, k) in rows and (s, reference, k) in rows]
    if not diffs:
        return None
    text = (f'{np.mean(diffs):+.4f} mean, {np.median(diffs):+.4f} median, '
            f'better in {sum(d > 0 for d in diffs)}/{len(diffs)}')
    if len(diffs) >= 6:
        try:
            from scipy.stats import wilcoxon
            text += f', Wilcoxon p={wilcoxon(diffs).pvalue:.3g}'
        except Exception:
            pass
    return text


def write_summary(out, new_rows, previous, args):
    rows = dict(previous)
    rows.update({(r['session'], r['arm'], r['seed']): r for r in new_rows})
    arms = ['fixed_eps_3', 'margin_std2'] + [arm_name(c) for c in args.caps]
    arms = [a for a in arms if any(k[1] == a for k in rows)]
    margin_arms = [a for a in arms if a.startswith('margin')]
    sessions = [s for s in args.sessions if any(k[0] == s for k in rows)]
    lines = [f'MARGIN CAP SWEEP | {len(sessions)} sessions | seeds={args.seeds} | iterations={args.iterations} | '
             'clean-input VALID R2 (all label columns)',
             f'previous results (fixed_eps_3, margin_std2): '
             f'{args.previous_summary if previous else "NOT FOUND (pass --previous-summary)"}', '',
             'Per session: VALID R2 [margin arms: mean per-sample epsilon of the last batch in x median neuron std,'
             ' % of samples at the cap]']
    width = 28
    lines.append(f'{"session":<9}' + ''.join(f'{a:>{width}}' for a in arms))
    for seed in args.seeds:
        for session in sessions:
            line = f'{session:<9}'
            for arm in arms:
                r = rows.get((session, arm, seed))
                if r is None:
                    cell = '-'
                elif arm.startswith('margin') and 'adv_eps_mean_last' in r:
                    cell = (f'{r["r2_valid"]:.3f} [{r["adv_eps_mean_last"] / r["median_std"]:.2f}x,'
                            f' {r["adv_frac_capped_last"]:.0%}]')
                else:
                    cell = f'{r["r2_valid"]:.3f}'
                line += f'{cell:>{width}}'
            lines.append(line + (f'   (seed {seed})' if len(args.seeds) > 1 else ''))
    line = f'{"MEAN":<9}'
    for arm in arms:
        values = [rows[(s, arm, k)]['r2_valid'] for s in sessions for k in args.seeds if (s, arm, k) in rows]
        eps = [rows[(s, arm, k)]['adv_eps_mean_last'] / rows[(s, arm, k)]['median_std'] for s in sessions
               for k in args.seeds if (s, arm, k) in rows and 'adv_eps_mean_last' in rows[(s, arm, k)]]
        cell = f'{np.mean(values):.4f}' + (f' [{np.mean(eps):.2f}x]' if eps else '') if values else '-'
        line += f'{cell:>{width}}'
    lines.append(line)

    lines += ['', 'PAIRED DIFFERENCES in VALID R2 (same session and seed)']
    for arm in margin_arms:
        for reference in ('fixed_eps_3', 'margin_std2'):
            if arm != reference and reference in arms:
                result = paired(rows, sessions, args.seeds, arm, reference)
                if result:
                    lines.append(f'  {arm:<13} - {reference:<12}: {result}')

    if len(margin_arms) >= 2:
        caps = sorted(margin_arms, key=lambda a: float(a.replace('margin_std', '')))
        complete = [(s, k) for s in sessions for k in args.seeds if all((s, a, k) in rows for a in caps)]
        if complete:
            increasing = sum(all(rows[(s, caps[i + 1], k)]['r2_valid'] > rows[(s, caps[i], k)]['r2_valid']
                                 for i in range(len(caps) - 1)) for s, k in complete)
            best = {a: sum(max(caps, key=lambda c: rows[(s, c, k)]['r2_valid']) == a for s, k in complete)
                    for a in caps}
            means = {a: np.mean([rows[(s, a, k)]['r2_valid'] for s, k in complete]) for a in caps}
            lines += ['', f'TREND over the cap ({" < ".join(caps)}), {len(complete)} complete sessions:',
                      f'  mean VALID R2: ' + ', '.join(f'{a}={means[a]:.4f}' for a in caps),
                      f'  R2 increases at every larger cap in {increasing}/{len(complete)} sessions',
                      f'  best cap per session: ' + ', '.join(f'{a}: {n}' for a, n in best.items())]
            eps = [(rows[(s, a, k)]['adv_eps_mean_last'] / rows[(s, a, k)]['median_std'], rows[(s, a, k)]['r2_valid'])
                   for s, k in complete for a in caps if 'adv_eps_mean_last' in rows[(s, a, k)]]
            if len(eps) >= 4:
                within = []
                for s, k in complete:
                    pts = [(rows[(s, a, k)]['adv_eps_mean_last'] / rows[(s, a, k)]['median_std'],
                            rows[(s, a, k)]['r2_valid']) for a in caps if 'adv_eps_mean_last' in rows[(s, a, k)]]
                    if len(pts) >= 2 and np.std([p[0] for p in pts]) > 0:
                        within.append(np.corrcoef(*zip(*pts))[0, 1])
                if within:
                    lines.append(f'  within-session correlation (mean epsilon x std vs R2): mean r = '
                                 f'{np.mean(within):+.3f} over {len(within)} sessions')
    text = '\n'.join(lines)
    (out / 'summary.txt').write_text(text + '\n', encoding='utf-8')
    fields = ['session', 'arm', 'seed', 'max_std', 'neurons', 'r2_train', 'r2_valid', 'encoder_seconds',
              'adv_eps_mean_last', 'adv_eps_median_last', 'adv_eps_cap_last', 'adv_frac_chance_last',
              'adv_frac_capped_last', 'adv_eps_mean_end', 'adv_frac_capped_end', 'median_std']
    with (out / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(new_rows)
    save_json(out / 'results.json', new_rows)
    print('\n' + text, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sessions', nargs='+', default=list(SESSIONS))
    parser.add_argument('--caps', nargs='+', type=float, default=list(CAPS),
                        help='adv_margin_max_std values to train (the previous run used 2).')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--cebra-dir', type=Path, default=ROOT / 'Acorn-margin-eps')
    parser.add_argument('--previous-summary', type=Path, default=PREVIOUS_SUMMARY,
                        help='summary.csv of the previous fixed_eps_3 / margin run.')
    parser.add_argument('--out', type=Path, default=None, help='Run folder; give an existing one to resume.')
    parser.add_argument('--iterations', type=int, default=ITERATIONS)
    parser.add_argument('--decoder-epochs', type=int, default=DECODER_EPOCHS)
    parser.add_argument('--device', default='cuda_if_available')
    args = parser.parse_args()
    if min(args.caps) <= 0 or min(args.iterations, args.decoder_epochs) < 1:
        parser.error('Caps, iterations and decoder epochs must be positive.')
    args.sessions = list(dict.fromkeys(args.sessions))
    args.caps = sorted(set(args.caps))
    args.seeds = list(dict.fromkeys(args.seeds))
    return args


def main():
    args = parse_args()
    if args.device == 'cuda_if_available':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    missing = [s for s in args.sessions if not (args.data_dir.expanduser() / f'{s}.npz').is_file()]
    if missing:
        raise FileNotFoundError(f'NPZ not found in {args.data_dir} for sessions: {missing}')
    cebra = load_fork(args.cebra_dir)
    for cap in args.caps:
        cebra.CEBRA(**encoder_config(cap, args))  # fail early on a constructor mismatch
    previous = read_previous(args.previous_summary)
    print(f'Previous results: {len(previous)} rows from {args.previous_summary}' if previous else
          f'Previous results not found at {args.previous_summary}; the table will only show the new arms.', flush=True)
    out = (args.out or OUT_ROOT / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}').expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'config.json', dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                        encoder_configs={arm_name(c): encoder_config(c, args) for c in args.caps},
                                        evaluation='clean-input VALID R2, all label columns'))
    print('OUTPUT:', out, flush=True)
    new_rows = []
    for session in args.sessions:
        data = None
        for cap in args.caps:
            for seed in args.seeds:
                folder = out / session / f'seed_{seed}' / arm_name(cap)
                if (folder / 'metrics.json').is_file():
                    new_rows.append(json.loads((folder / 'metrics.json').read_text()))
                    print(f'skip (done): {session} {arm_name(cap)} seed {seed}', flush=True)
                    continue
                if data is None:
                    data = load_session(args.data_dir, session)
                    print(f'\nSESSION {session}: train {data[0].shape}, valid {data[1].shape}', flush=True)
                new_rows.append(run_one(cebra, session, cap, seed, data, folder, args, device))
                write_summary(out, new_rows, previous, args)
        del data
        cleanup()
    write_summary(out, new_rows, previous, args)
    print('\nDone. Send this file:', out / 'summary.txt', flush=True)


if __name__ == '__main__':
    main()

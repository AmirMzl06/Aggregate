 
"""Data-derived epsilon without tuning: eps = sqrt(N / d) x median neuron std, on the 10 random Perich sessions.

Rationale: a perturbation that is not aligned with the behavioral signal is averaged over the N
neurons by the population readout, so its effect shrinks like sqrt(d / N), with d the dimension
of the signal. Asking the perturbation to be as large as the signal at the readout (SNR = 1)
gives k = sqrt(N / d) in units of the neuron std. Nothing is tuned:
    auto_pr      d = participation ratio of the neural covariance (train data), PR = (sum l)^2 / sum l^2
    auto_dlabel  d = number of behavior label columns
N counts the active neurons (std > 0); the std is the median over active neurons (as in the forks).

Part 1  per session: N, median std, PR, the predicted epsilons (no training; --preview-only stops here)
Part 2  training: the ORIGINAL PGD (constant mode of Acorn-margin-eps) with that epsilon, alpha = eps / 5,
        every other setting identical to the previous fixed_eps_3 runs
Part 3  clean-input VALID R2 compared with fixed_eps_3 (and margin_std2/4/6 when their summary.csv
        files are found), paired over sessions

Output: <out>/summary.txt holds all three parts (send this one), plus part1_epsilons.csv,
summary.csv, results.json and the checkpoints. Re-running with --out <same folder> resumes.
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
import math
import random
import sys
import time
import numpy as np
import torch
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path('/data/hossein/mm_project/perich_data_valid_final_raw')
OUT_ROOT = ROOT / 'ACORN_AUTO_EPS'
PREVIOUS_SUMMARY = ROOT / 'ACORN_MARGIN_VS_FIXED_RANDOM10' / 'selection_seed_42_n10' / 'summary.csv'
SWEEP_SUMMARY = ROOT / 'ACORN_MARGIN_CAP_SWEEP' / 'caps4_6_seed42' / 'summary.csv'
SESSIONS = ('C-CO11', 'C-CO21', 'C-CO22', 'C-CO25', 'C-CO35',
            'C-CO38', 'C-CO41', 'M-CO19', 'M-CO4', 'M-RT2')
ARMS = ('auto_pr', 'auto_dlabel')
SEEDS = (42,)
FIXED_EPSILON = 3.0  # the hand-set baseline the automatic rule is compared with

# Same settings as the previous fixed_eps_3 runs.
ITERATIONS = 3000
BATCH_SIZE = 2048
ARCHITECTURE = 'offset36-model-more-dropout'
LATENT_DIM = 64
ENCODER_HIDDEN = 64
TEMPERATURE = 0.4
ENCODER_LR = 3e-4
ATTACK_STEPS = 10
ALPHA_RATIO = 0.2  # alpha = eps / 5, as fixed_eps_3 (3 / 5 = 0.6)
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
    if 'adv_epsilon_mode' not in inspect.signature(cebra.CEBRA.__init__).parameters:
        raise RuntimeError(f'{path} is not Acorn-margin-eps (no adv_epsilon_mode).')
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


# ----------------------------------------------------------------------
# Part 1: the data-derived epsilon
# ----------------------------------------------------------------------
def participation_ratio(matrix):
    """(sum l)^2 / sum l^2 of the eigenvalues l of a symmetric PSD matrix."""
    eigenvalues = np.clip(np.linalg.eigvalsh(matrix), 0.0, None)
    return float(eigenvalues.sum()**2 / (eigenvalues**2).sum())


def session_epsilons(xtr, ytr):
    """Statistics of the TRAIN split and the epsilon of every automatic arm."""
    std = torch.from_numpy(xtr).float().std(dim=0, unbiased=False)  # as in Acorn-margin-eps
    active_mask = (std > 0).numpy()
    active = std[std > 0]
    median_std = float(active.median()) if active.numel() > 0 else 1.0
    n_active = int(active_mask.sum())
    x = xtr[:, active_mask].astype(np.float64)
    pr_cov = participation_ratio(np.cov(x, rowvar=False))
    pr_corr = participation_ratio(np.corrcoef(x, rowvar=False))
    n_labels = int(ytr.shape[1])
    k = dict(auto_pr=math.sqrt(n_active / pr_cov), auto_dlabel=math.sqrt(n_active / n_labels))
    return dict(neurons=int(xtr.shape[1]), active_neurons=n_active, median_std=median_std, pr_cov=pr_cov,
                pr_corr=pr_corr, labels=n_labels, k=k, epsilon={a: k[a] * median_std for a in k},
                fixed_k=FIXED_EPSILON / median_std)


def part1_lines(stats, sessions):
    lines = ['PART 1: data-derived epsilon (TRAIN split only, no tuning)',
             '  k = sqrt(N_active / d) in units of the median neuron std; epsilon = k x median std.',
             '  auto_pr: d = participation ratio of the covariance; auto_dlabel: d = number of label columns.',
             '  (PR of the correlation matrix is listed for reference only.)',
             f'{"session":<9}{"N":>5}{"N_act":>7}{"med_std":>9}{"PR_cov":>8}{"PR_corr":>9}{"labels":>7}'
             f'{"k_pr":>7}{"eps_pr":>8}{"k_dlab":>8}{"eps_dlab":>9}{"k(eps=3)":>10}']
    for s in sessions:
        st = stats[s]
        lines.append(f'{s:<9}{st["neurons"]:>5}{st["active_neurons"]:>7}{st["median_std"]:>9.3f}{st["pr_cov"]:>8.2f}'
                     f'{st["pr_corr"]:>9.2f}{st["labels"]:>7}{st["k"]["auto_pr"]:>7.2f}'
                     f'{st["epsilon"]["auto_pr"]:>8.3f}{st["k"]["auto_dlabel"]:>8.2f}'
                     f'{st["epsilon"]["auto_dlabel"]:>9.3f}{st["fixed_k"]:>10.2f}')
    for arm in ARMS:
        ks = [stats[s]['k'][arm] for s in sessions]
        eps = [stats[s]['epsilon'][arm] for s in sessions]
        lines.append(f'  {arm:<12} k range {min(ks):.2f}-{max(ks):.2f} (median {np.median(ks):.2f}); '
                     f'epsilon range {min(eps):.3f}-{max(eps):.3f}')
    fixed = [stats[s]['fixed_k'] for s in sessions]
    lines.append(f'  fixed_eps_3  k range {min(fixed):.2f}-{max(fixed):.2f} (median {np.median(fixed):.2f})')
    return lines


# ----------------------------------------------------------------------
# Part 2: training
# ----------------------------------------------------------------------
def encoder_config(epsilon, args):
    # Identical to the previous fixed_eps_3 arm except the value of epsilon (and alpha = epsilon / 5).
    return dict(batch_size=BATCH_SIZE, temperature=TEMPERATURE, temperature_mode='constant', distance='cosine',
                model_architecture=ARCHITECTURE, time_offsets=1, max_iterations=args.iterations,
                output_dimension=LATENT_DIM, num_hidden_units=ENCODER_HIDDEN, learning_rate=ENCODER_LR,
                pad_before_transform=True, hybrid=False, training_mode='adversarial', attack_norm='linf',
                adv_epsilon=float(epsilon), adv_alpha=float(ALPHA_RATIO * epsilon), adv_steps=ATTACK_STEPS,
                adv_epsilon_mode='constant', device=args.device, verbose=True)


def build_decoder(in_dim, out_dim, device):
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


def run_one(cebra, session, arm, seed, data, st, folder, args, device):
    xtr, xva, ytr, yva = data
    folder.mkdir(parents=True, exist_ok=True)
    epsilon = st['epsilon'][arm]
    config = encoder_config(epsilon, args)
    print(f'\n{"=" * 80}\nTRAIN {session} | {arm} | epsilon={epsilon:.4f} (k={st["k"][arm]:.2f} x std) | '
          f'seed={seed}\n{"=" * 80}', flush=True)
    seed_all(seed)
    started = time.perf_counter()
    model = cebra.CEBRA(**config)
    model.fit(xtr, ytr)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    encoder_seconds = time.perf_counter() - started
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
    row = dict(session=session, arm=arm, seed=seed, epsilon=epsilon, k=st['k'][arm], median_std=st['median_std'],
               d=st['pr_cov'] if arm == 'auto_pr' else st['labels'], neurons=st['neurons'],
               active_neurons=st['active_neurons'], r2_train=r2_train, r2_valid=r2_valid,
               train_per_output=train_per_output, valid_per_output=valid_per_output,
               encoder_seconds=encoder_seconds, config=config)
    save_json(folder / 'metrics.json', row)  # written last: marks the run as finished
    print(f'{session} {arm}: TRAIN R2={r2_train:.4f} | VALID R2={r2_valid:.4f} | {encoder_seconds / 60:.1f} min',
          flush=True)
    del decoder
    cleanup()
    return row


# ----------------------------------------------------------------------
# Part 3: comparison
# ----------------------------------------------------------------------
def read_summary(path, rename):
    """{(session, arm, seed): r2_valid} from a previous summary.csv, arms renamed by ``rename``."""
    if path is None or not path.is_file():
        return {}
    rows = {}
    with path.open(newline='') as handle:
        for r in csv.DictReader(handle):
            arm = rename.get(r['arm'], r['arm'])
            rows[(r['session'], arm, int(r['seed']))] = float(r['r2_valid'])
    return rows


def paired_stats(diffs):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    out = dict(n=n, mean=float(diffs.mean()), median=float(np.median(diffs)), better=int((diffs > 0).sum()))
    if n >= 2:
        sem = diffs.std(ddof=1) / math.sqrt(n)
        try:
            from scipy.stats import t
            q = float(t.ppf(0.975, n - 1))
        except Exception:
            q = 1.96
        out.update(ci_low=out['mean'] - q * sem, ci_high=out['mean'] + q * sem)
    if n >= 6:
        try:
            from scipy.stats import wilcoxon
            out['wilcoxon_p'] = float(wilcoxon(diffs).pvalue)
        except Exception:
            pass
    return out


def part2_lines(new_rows):
    lines = ['PART 2: training runs (original PGD, alpha = epsilon / 5, same settings as fixed_eps_3)',
             f'{"session":<9}{"arm":<13}{"seed":>5}{"epsilon":>9}{"k":>6}{"R2 train":>10}{"R2 valid":>10}{"min":>7}']
    for r in new_rows:
        lines.append(f'{r["session"]:<9}{r["arm"]:<13}{r["seed"]:>5}{r["epsilon"]:>9.3f}{r["k"]:>6.2f}'
                     f'{r["r2_train"]:>10.4f}{r["r2_valid"]:>10.4f}{r["encoder_seconds"] / 60:>7.1f}')
    return lines


def part3_lines(new_rows, previous, args):
    rows = dict(previous)
    rows.update({(r['session'], r['arm'], r['seed']): r['r2_valid'] for r in new_rows})
    candidates = ['fixed_eps_3'] + list(args.arms) + ['margin_std2', 'margin_std4', 'margin_std6']
    arms = [a for a in candidates if any(k[1] == a for k in rows)]
    sessions = [s for s in args.sessions if any(k[0] == s for k in rows)]
    lines = ['PART 3: clean-input VALID R2 (all label columns), compared with the hand-set fixed_eps_3',
             f'  previous fixed_eps_3 / margin_std2: {args.previous_summary if args.previous_summary.is_file() else "NOT FOUND"}',
             f'  margin_std4 / margin_std6 sweep:     {args.sweep_summary if args.sweep_summary.is_file() else "NOT FOUND"}',
             f'{"session":<9}' + ''.join(f'{a:>14}' for a in arms)]
    for seed in args.seeds:
        for s in sessions:
            cells = [rows.get((s, a, seed)) for a in arms]
            lines.append(f'{s:<9}' + ''.join(f'{(f"{c:.4f}" if c is not None else "-"):>14}' for c in cells)
                         + (f'   (seed {seed})' if len(args.seeds) > 1 else ''))
    means = []
    for a in arms:
        values = [rows[(s, a, k)] for s in sessions for k in args.seeds if (s, a, k) in rows]
        means.append(f'{np.mean(values):.4f}' if values else '-')
    lines.append(f'{"MEAN":<9}' + ''.join(f'{m:>14}' for m in means))
    if 'fixed_eps_3' in arms:
        lines += ['', '  Paired difference vs fixed_eps_3 (same session and seed): mean [95% CI], median, '
                      'better in, Wilcoxon p']
        for a in arms:
            if a == 'fixed_eps_3':
                continue
            diffs = [rows[(s, a, k)] - rows[(s, 'fixed_eps_3', k)] for s in sessions for k in args.seeds
                     if (s, a, k) in rows and (s, 'fixed_eps_3', k) in rows]
            if not diffs:
                continue
            p = paired_stats(diffs)
            ci = f' [{p["ci_low"]:+.4f}, {p["ci_high"]:+.4f}]' if 'ci_low' in p else ''
            wp = f', p={p["wilcoxon_p"]:.3g}' if 'wilcoxon_p' in p else ''
            lines.append(f'  {a:<13} {p["mean"]:+.4f}{ci}, median {p["median"]:+.4f}, '
                         f'better in {p["better"]}/{p["n"]}{wp}')
        lines += ['  Reading: if the 95% CI of an automatic arm contains 0 and is narrow, its R2 matches the hand-set',
                  '  epsilon without tuning; a CI entirely above 0 means it is better, entirely below 0 worse.']
    return lines


def write_summary(out, stats, new_rows, previous, args):
    sessions = [s for s in args.sessions if s in stats]
    header = [f'ACORN automatic epsilon | {len(sessions)} sessions | seeds={args.seeds} | iterations={args.iterations} '
              f'| arms={list(args.arms)}', '']
    lines = header + part1_lines(stats, sessions)
    if new_rows:
        lines += [''] + part2_lines(new_rows) + [''] + part3_lines(new_rows, previous, args)
    text = '\n'.join(lines)
    (out / 'summary.txt').write_text(text + '\n', encoding='utf-8')
    with (out / 'part1_epsilons.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['session', 'neurons', 'active_neurons', 'median_std', 'pr_cov', 'pr_corr', 'labels',
                         'k_auto_pr', 'eps_auto_pr', 'k_auto_dlabel', 'eps_auto_dlabel', 'k_fixed_eps_3'])
        for s in sessions:
            st = stats[s]
            writer.writerow([s, st['neurons'], st['active_neurons'], st['median_std'], st['pr_cov'], st['pr_corr'],
                             st['labels'], st['k']['auto_pr'], st['epsilon']['auto_pr'], st['k']['auto_dlabel'],
                             st['epsilon']['auto_dlabel'], st['fixed_k']])
    fields = ['session', 'arm', 'seed', 'epsilon', 'k', 'd', 'median_std', 'neurons', 'active_neurons',
              'r2_train', 'r2_valid', 'encoder_seconds']
    with (out / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(new_rows)
    save_json(out / 'results.json', dict(part1=stats, runs=new_rows))
    print('\n' + text, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sessions', nargs='+', default=list(SESSIONS))
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--cebra-dir', type=Path, default=ROOT / 'Acorn-margin-eps')
    parser.add_argument('--previous-summary', type=Path, default=PREVIOUS_SUMMARY,
                        help='summary.csv of the previous fixed_eps_3 / margin run.')
    parser.add_argument('--sweep-summary', type=Path, default=SWEEP_SUMMARY,
                        help='summary.csv of the margin cap sweep (margin_std4 / margin_std6).')
    parser.add_argument('--out', type=Path, default=None, help='Run folder; give an existing one to resume.')
    parser.add_argument('--iterations', type=int, default=ITERATIONS)
    parser.add_argument('--decoder-epochs', type=int, default=DECODER_EPOCHS)
    parser.add_argument('--preview-only', action='store_true', help='Part 1 only (no training).')
    parser.add_argument('--device', default='cuda_if_available')
    args = parser.parse_args()
    if min(args.iterations, args.decoder_epochs) < 1:
        parser.error('Iterations and decoder epochs must be positive.')
    args.sessions = list(dict.fromkeys(args.sessions))
    args.arms = list(dict.fromkeys(args.arms))
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
    out = (args.out or OUT_ROOT / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}').expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    print('OUTPUT:', out, flush=True)

    print('\nPART 1: data-derived epsilon per session', flush=True)
    stats = {s: session_epsilons(*load_session(args.data_dir, s)[0::2]) for s in args.sessions}
    save_json(out / 'config.json', dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                        part1=stats, evaluation='clean-input VALID R2, all label columns'))
    if args.preview_only:
        write_summary(out, stats, [], {}, args)
        print('\nPreview only. Send this file:', out / 'summary.txt', flush=True)
        return

    cebra = load_fork(args.cebra_dir)
    cebra.CEBRA(**encoder_config(1.0, args))  # fail early on a constructor mismatch
    previous = read_summary(args.previous_summary, {'margin': 'margin_std2'})
    previous.update(read_summary(args.sweep_summary, {}))
    print(f'Previous results: {len(previous)} rows', flush=True)

    print('\nPART 2: training', flush=True)
    new_rows = []
    for session in args.sessions:
        data = None
        for arm in args.arms:
            for seed in args.seeds:
                folder = out / session / f'seed_{seed}' / arm
                if (folder / 'metrics.json').is_file():
                    new_rows.append(json.loads((folder / 'metrics.json').read_text()))
                    print(f'skip (done): {session} {arm} seed {seed}', flush=True)
                    continue
                if data is None:
                    data = load_session(args.data_dir, session)
                    print(f'\nSESSION {session}: train {data[0].shape}, valid {data[1].shape}', flush=True)
                new_rows.append(run_one(cebra, session, arm, seed, data, stats[session], folder, args, device))
                write_summary(out, stats, new_rows, previous, args)
        del data
        cleanup()
    write_summary(out, stats, new_rows, previous, args)
    print('\nDone. Send this file:', out / 'summary.txt', flush=True)


if __name__ == '__main__':
    main()

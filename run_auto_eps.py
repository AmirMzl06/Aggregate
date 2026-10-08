"""Automatic epsilon vs hand-set epsilon on randomly chosen Perich sessions.

To run 50 held-out sessions on four visible GPUs:
    python -u run_auto_eps_parallel.py --num-sessions 50 --gpus 0,1,2,3
One worker process is assigned to each GPU; each trains both arms on its sessions.
Without --gpus this program uses the original single-process execution.

For every session two arms are trained from scratch with the ORIGINAL PGD (constant mode of
Acorn-margin-eps; alpha = epsilon / 5; every other setting as in the previous fixed_eps_3 runs):
    fixed_eps  the hand-set epsilon (--fixed-epsilon, default 3)
    auto_eps   epsilon = sqrt(N / d) x median neuron std, with N the active neurons and d the number
               of behavior label columns; nothing is tuned. Rationale: a perturbation not aligned with
               the behavioral signal is averaged over the N neurons by the readout, so its effect
               shrinks like sqrt(d / N); asking it to be as large as the signal (SNR = 1) gives
               k = sqrt(N / d) in units of the neuron std.

Sessions: --num-sessions picks that many at random (--selection-seed) among the NPZ files of
--data-dir, leaving out the sessions the rule was developed on (--include-used keeps them);
--sessions gives an explicit list instead. No previous result is read.

Output: <out>/summary.txt (send this one), with
    Part 1  per session: N, labels, median std, k and epsilon of both arms
    Part 2  every training run: R2 train / valid
    Part 3  paired comparison of clean-input VALID R2 and the equivalence test
plus sessions.txt, summary.csv, results.json and the checkpoints. Re-running with
--out <same folder> resumes (finished runs and the session list are reused).
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
from datetime import datetime
import argparse
import csv
import gc
import hashlib
import inspect
import json
import math
import random
import subprocess
import sys
import time
import numpy as np
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path('/data/hossein/mm_project/perich_data_valid_final_raw')
OUT_ROOT = ROOT / 'ACORN_AUTO_EPS'
ARMS = ('fixed_eps', 'auto_eps')
# Sessions the automatic rule was developed on (left out unless --include-used).
USED_SESSIONS = ('C-CO12', 'C-CO11', 'C-CO21', 'C-CO22', 'C-CO25', 'C-CO35',
                 'C-CO38', 'C-CO41', 'M-CO19', 'M-CO4', 'M-RT2')
NPZ_KEYS = ('train_data', 'valid_data', 'train_label', 'valid_label')

# Same settings as the previous fixed_eps_3 runs.
ITERATIONS = 3000
BATCH_SIZE = 2048
ARCHITECTURE = 'offset36-model-more-dropout'
LATENT_DIM = 64
ENCODER_HIDDEN = 64
TEMPERATURE = 0.4
ENCODER_LR = 3e-4
ATTACK_STEPS = 10
ALPHA_RATIO = 0.2  # alpha = epsilon / 5 (fixed_eps_3: 3 / 5 = 0.6)
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
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in NPZ_KEYS if k not in data.files]
        if missing:
            raise KeyError(f'{path}: missing {missing}')
        xtr, xva, ytr, yva = [np.array(data[k], dtype=np.float32, order='C', copy=True) for k in NPZ_KEYS]
    if ytr.ndim == 1:
        ytr, yva = ytr[:, None], yva[:, None]
    for key, arr in zip(NPZ_KEYS, (xtr, xva, ytr, yva)):
        if arr.ndim != 2 or min(arr.shape) < 1 or not np.isfinite(arr).all():
            raise ValueError(f'{path}: {key} must be a finite, nonempty 2D array')
    if (len(xtr) != len(ytr) or len(xva) != len(yva) or xtr.shape[1] != xva.shape[1]
            or ytr.shape[1] != yva.shape[1] or min(len(xtr), len(xva)) < 2):
        raise ValueError(f'{path}: inconsistent shapes')
    return xtr, xva, ytr, yva


def choose_sessions(args, out):
    """Explicit --sessions, the list saved in a resumed run, or a random draw from --data-dir."""
    saved = out / 'sessions.txt'
    if args.sessions:
        sessions = list(dict.fromkeys(args.sessions))
        if saved.is_file() and sessions != saved.read_text().split():
            raise RuntimeError(f'{saved} has a different session list. Use a new --out path.')
    elif saved.is_file():
        sessions = [s for s in saved.read_text().split() if s]
        print(f'Resuming with the {len(sessions)} sessions of {saved}', flush=True)
    else:
        candidates = []
        for path in sorted(args.data_dir.expanduser().glob('*.npz')):
            if not args.include_used and path.stem in USED_SESSIONS:
                continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    if all(k in data.files for k in NPZ_KEYS):
                        candidates.append(path.stem)
            except Exception:
                continue
        if args.num_sessions > len(candidates):
            raise ValueError(f'--num-sessions {args.num_sessions} but only {len(candidates)} eligible sessions in '
                             f'{args.data_dir} (used sessions {"kept" if args.include_used else "left out"}).')
        sessions = sorted(random.Random(args.selection_seed).sample(candidates, args.num_sessions))
        print(f'Selected {len(sessions)} of {len(candidates)} eligible sessions (selection seed '
              f'{args.selection_seed}): {sessions}', flush=True)
    saved.write_text('\n'.join(sessions) + '\n')
    return sessions


# ----------------------------------------------------------------------
# Part 1: epsilon of each arm
# ----------------------------------------------------------------------
def session_epsilons(xtr, ytr, fixed_epsilon):
    std = torch.from_numpy(xtr).float().std(dim=0, unbiased=False)  # as in Acorn-margin-eps
    active = std[std > 0]
    if active.numel() == 0:
        raise ValueError('No active neurons in the training split.')
    median_std = float(active.median())
    n_active, n_labels = int(active.numel()), int(ytr.shape[1])
    k_auto = math.sqrt(n_active / n_labels)
    return dict(neurons=int(xtr.shape[1]), active_neurons=n_active, labels=n_labels, median_std=median_std,
                train_samples=int(len(xtr)),
                epsilon=dict(fixed_eps=float(fixed_epsilon), auto_eps=k_auto * median_std),
                k=dict(fixed_eps=fixed_epsilon / median_std, auto_eps=k_auto))


def part1_lines(stats, sessions, args):
    lines = ['PART 1: epsilon of each arm (TRAIN split only)',
             f'  fixed_eps: epsilon = {args.fixed_epsilon:g} (hand-set);  auto_eps: epsilon = sqrt(N_active / labels)'
             ' x median neuron std (no tuning). k = epsilon / median std.',
             f'{"session":<12}{"N":>6}{"N_act":>7}{"labels":>7}{"n_train":>9}{"med_std":>9}'
             f'{"k_fixed":>9}{"k_auto":>8}{"eps_auto":>10}']
    for s in sessions:
        st = stats[s]
        lines.append(f'{s:<12}{st["neurons"]:>6}{st["active_neurons"]:>7}{st["labels"]:>7}{st["train_samples"]:>9}'
                     f'{st["median_std"]:>9.3f}{st["k"]["fixed_eps"]:>9.2f}{st["k"]["auto_eps"]:>8.2f}'
                     f'{st["epsilon"]["auto_eps"]:>10.3f}')
    for arm in ARMS:
        ks = [stats[s]['k'][arm] for s in sessions]
        lines.append(f'  {arm:<10} k range {min(ks):.2f}-{max(ks):.2f} (median {np.median(ks):.2f})')
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
                adv_epsilon=float(epsilon), adv_alpha=float(epsilon / 5.0), adv_steps=ATTACK_STEPS,
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
               neurons=st['neurons'], active_neurons=st['active_neurons'], labels=st['labels'],
               r2_train=r2_train, r2_valid=r2_valid, train_per_output=train_per_output,
               valid_per_output=valid_per_output, encoder_seconds=encoder_seconds, config=config)
    save_json(folder / 'metrics.json', row)  # written last: marks the run as finished
    print(f'{session} {arm}: TRAIN R2={r2_train:.4f} | VALID R2={r2_valid:.4f} | {encoder_seconds / 60:.1f} min',
          flush=True)
    del decoder
    cleanup()
    return row


# ----------------------------------------------------------------------
# Part 3: comparison
# ----------------------------------------------------------------------
def paired_stats(diffs, margin):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    out = dict(n=n, mean=float(diffs.mean()), median=float(np.median(diffs)), better=int((diffs > 0).sum()))
    if n < 2:
        return out
    sem = float(diffs.std(ddof=1) / math.sqrt(n))
    try:
        from scipy.stats import t
        q, cdf = float(t.ppf(0.975, n - 1)), (lambda x: float(t.cdf(x, n - 1)))
    except Exception:
        q, cdf = 1.96, (lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2))))
    out.update(sem=sem, ci_low=out['mean'] - q * sem, ci_high=out['mean'] + q * sem)
    if sem > 0:
        # TOST: two one-sided t-tests against -margin and +margin.
        out['tost_p'] = max(1 - cdf((out['mean'] + margin) / sem), cdf((out['mean'] - margin) / sem))
    if n >= 6:
        try:
            from scipy.stats import wilcoxon
            out['wilcoxon_p'] = float(wilcoxon(diffs).pvalue)
        except Exception:
            pass
    return out


def part2_lines(rows):
    lines = ['PART 2: training runs (original PGD, alpha = epsilon / 5)',
             f'{"session":<12}{"arm":<11}{"seed":>5}{"epsilon":>9}{"k":>7}{"R2 train":>10}{"R2 valid":>10}{"min":>7}']
    for r in rows:
        lines.append(f'{r["session"]:<12}{r["arm"]:<11}{r["seed"]:>5}{r["epsilon"]:>9.3f}{r["k"]:>7.2f}'
                     f'{r["r2_train"]:>10.4f}{r["r2_valid"]:>10.4f}{r["encoder_seconds"] / 60:>7.1f}')
    return lines


def part3_lines(rows, sessions, args):
    r2 = {(r['session'], r['arm'], r['seed']): r['r2_valid'] for r in rows}
    pairs = [(s, k) for s in sessions for k in args.seeds if (s, 'fixed_eps', k) in r2 and (s, 'auto_eps', k) in r2]
    lines = [f'PART 3: clean-input VALID R2 (all label columns), auto_eps vs fixed_eps (epsilon = {args.fixed_epsilon:g})',
             f'{"session":<12}{"seed":>5}{"fixed_eps":>11}{"auto_eps":>10}{"diff":>9}']
    for s, k in pairs:
        lines.append(f'{s:<12}{k:>5}{r2[(s, "fixed_eps", k)]:>11.4f}{r2[(s, "auto_eps", k)]:>10.4f}'
                     f'{r2[(s, "auto_eps", k)] - r2[(s, "fixed_eps", k)]:>+9.4f}')
    if not pairs:
        return lines + ['  (no session has both arms finished yet)']
    fixed = [r2[(s, 'fixed_eps', k)] for s, k in pairs]
    auto = [r2[(s, 'auto_eps', k)] for s, k in pairs]
    lines.append(f'{"MEAN":<12}{"":>5}{np.mean(fixed):>11.4f}{np.mean(auto):>10.4f}{np.mean(auto) - np.mean(fixed):>+9.4f}')
    p = paired_stats(np.subtract(auto, fixed), args.equivalence_margin)
    lines += ['', f'  auto_eps - fixed_eps over {p["n"]} paired runs: mean {p["mean"]:+.4f}, median {p["median"]:+.4f}, '
                  f'auto better in {p["better"]}/{p["n"]}']
    if 'ci_low' in p:
        lines.append(f'  95% CI [{p["ci_low"]:+.4f}, {p["ci_high"]:+.4f}]'
                     + (f' | Wilcoxon p={p["wilcoxon_p"]:.3g}' if 'wilcoxon_p' in p else '')
                     + (f' | TOST p={p["tost_p"]:.3g} (margin +-{args.equivalence_margin:g})' if 'tost_p' in p else ''))
        m = args.equivalence_margin
        if p['ci_low'] > 0:
            verdict = 'auto_eps is BETTER than fixed_eps (95% CI above 0)'
        elif -m < p['ci_low'] and p['ci_high'] < m:
            verdict = (f'EQUIVALENT: the whole 95% CI lies within +-{m:g} (auto_eps matches the hand-set epsilon)'
                       + (f'; it is slightly lower, by less than {m:g}' if p['ci_high'] < 0 else ''))
        elif p['ci_high'] < 0:
            verdict = f'auto_eps is WORSE than fixed_eps (95% CI below 0 and reaching beyond -{m:g})'
        else:
            verdict = f'INCONCLUSIVE: the 95% CI is wider than +-{m:g}; more sessions needed'
        lines.append(f'  VERDICT: {verdict}')
    return lines


def write_summary(out, stats, sessions, rows, args):
    done = [s for s in sessions if s in stats]
    lines = [f'ACORN automatic epsilon | {len(done)} sessions | seeds={args.seeds} | iterations={args.iterations} | '
             f'fixed epsilon={args.fixed_epsilon:g}', ''] + part1_lines(stats, done, args)
    if rows:
        lines += [''] + part2_lines(rows) + [''] + part3_lines(rows, done, args)
    text = '\n'.join(lines)
    (out / 'summary.txt').write_text(text + '\n', encoding='utf-8')
    fields = ['session', 'arm', 'seed', 'epsilon', 'k', 'median_std', 'neurons', 'active_neurons', 'labels',
              'r2_train', 'r2_valid', 'encoder_seconds']
    with (out / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    save_json(out / 'results.json', dict(sessions=sessions, part1=stats, runs=rows))
    print('\n' + text, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--num-sessions', type=int, default=20, help='How many sessions to draw at random.')
    parser.add_argument('--selection-seed', type=int, default=0, help='Seed of the random session draw.')
    parser.add_argument('--include-used', action='store_true',
                        help='Also draw from the 11 sessions the rule was developed on.')
    parser.add_argument('--sessions', nargs='+', default=None, help='Explicit session list (overrides the draw).')
    parser.add_argument('--fixed-epsilon', type=float, default=3.0, help='Hand-set epsilon of the fixed_eps arm.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--equivalence-margin', type=float, default=0.01)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--cebra-dir', type=Path, default=ROOT / 'Acorn-margin-eps')
    parser.add_argument('--out', type=Path, default=None, help='Run folder; give an existing one to resume.')
    parser.add_argument('--iterations', type=int, default=ITERATIONS)
    parser.add_argument('--decoder-epochs', type=int, default=DECODER_EPOCHS)
    parser.add_argument('--device', default='cuda_if_available')
    parser.add_argument('--gpus', default=None,
                        help='GPU IDs visible to this process, e.g. 0,1,2,3. One worker per GPU.')
    args = parser.parse_args()
    if args.num_sessions < 1 or min(args.iterations, args.decoder_epochs) < 1:
        parser.error('--num-sessions, --iterations and --decoder-epochs must be positive.')
    if args.fixed_epsilon <= 0 or args.equivalence_margin <= 0:
        parser.error('--fixed-epsilon and --equivalence-margin must be positive.')
    args.seeds = list(dict.fromkeys(args.seeds))
    return args


def run_parallel(args, out, sessions):
    """Run disjoint sets of sessions in separate processes, then merge their reports."""
    gpus = [g.strip() for g in args.gpus.split(',')]
    if not gpus or any(not g.isdecimal() for g in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError('--gpus must contain distinct numeric IDs, for example 0,1,2,3')
    if args.device == 'cpu' or not torch.cuda.is_available():
        raise RuntimeError('--gpus requires GPUs visible inside this container (and a CUDA PyTorch build).')
    if torch.cuda.device_count() < len(gpus):
        raise RuntimeError(f'Only {torch.cuda.device_count()} GPUs visible; requested {len(gpus)}. '
                           'Allocate four GPUs to this job before passing --gpus 0,1,2,3.')
    # Put both training arms for a session on the same GPU and keep assignment stable on resume.
    assignments = {g: sessions[i::len(gpus)] for i, g in enumerate(gpus)}
    manifest = dict(sessions=sessions, gpus=gpus, selection_seed=args.selection_seed,
                    include_used=args.include_used, seeds=args.seeds,
                    fixed_epsilon=args.fixed_epsilon, equivalence_margin=args.equivalence_margin,
                    iterations=args.iterations, decoder_epochs=args.decoder_epochs,
                    data_dir=str(args.data_dir.expanduser().resolve()),
                    cebra_dir=str(args.cebra_dir.expanduser().resolve()),
                    fork_sha256={name: hashlib.sha256((args.cebra_dir.expanduser().resolve() / name).read_bytes()).hexdigest()
                                 for name in ('cebra/integrations/sklearn/cebra.py', 'cebra/solver/base.py')})
    manifest_path = out / 'parallel_manifest.json'
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError(f'{out} has different sessions/config/fork/GPU assignment. '
                               'Use a fresh --out, or rerun with exactly the original options.')
    else:
        save_json(manifest_path, manifest)

    children = {}
    cpu_threads = max(1, (os.cpu_count() or len(gpus)) // len(gpus))
    try:
        for gpu, subset in assignments.items():
            if not subset:
                continue
            worker_out = out / 'workers' / f'gpu_{gpu}'
            worker_out.mkdir(parents=True, exist_ok=True)
            log_path = out / f'gpu_{gpu}.log'
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()),
                   '--sessions', *subset, '--out', str(worker_out),
                   '--data-dir', str(args.data_dir), '--cebra-dir', str(args.cebra_dir),
                   '--fixed-epsilon', str(args.fixed_epsilon),
                   '--equivalence-margin', str(args.equivalence_margin),
                   '--iterations', str(args.iterations), '--decoder-epochs', str(args.decoder_epochs),
                   '--seeds', *map(str, args.seeds), '--device', 'cuda']
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS=str(cpu_threads), MKL_NUM_THREADS=str(cpu_threads))
            log = log_path.open('a', encoding='utf-8')
            try:
                process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            children[gpu] = (process, log, worker_out)
            print(f'GPU {gpu}: {len(subset)} sessions, PID={process.pid}, log={log_path}', flush=True)
        status = {}
        while len(status) < len(children):
            for gpu, (process, log, _) in children.items():
                if gpu in status:
                    continue
                exit_code = process.poll()
                if exit_code is not None:
                    status[gpu] = exit_code
                    log.close()
                    print(f'GPU {gpu}: exit={exit_code}', flush=True)
            if len(status) < len(children):
                time.sleep(5)
    except KeyboardInterrupt:
        for process, log, _ in children.values():
            if process.poll() is None:
                process.terminate()
        for process, log, _ in children.values():
            process.wait()
            if not log.closed:
                log.close()
        print('Interrupted; rerun with the same --out and options to resume.', flush=True)
        raise
    except Exception:
        for process, _, _ in children.values():
            if process.poll() is None:
                process.terminate()
        for process, log, _ in children.values():
            process.wait()
            if not log.closed:
                log.close()
        raise
    # The workers never write in the same directory. The parent alone creates the joint table.
    stats, rows = {}, []
    for _, _, worker_out in children.values():
        path = worker_out / 'results.json'
        if path.is_file():
            result = json.loads(path.read_text(encoding='utf-8'))
            stats.update(result['part1'])
            rows.extend(result['runs'])
    rows.sort(key=lambda r: (sessions.index(r['session']), r['seed'], ARMS.index(r['arm'])))
    if stats:
        write_summary(out, stats, sessions, rows, args)
    failures = [gpu for gpu, exit_code in status.items() if exit_code != 0]
    if failures:
        raise RuntimeError(f'Workers failed on GPUs {failures}; inspect gpu_<id>.log in {out}. '
                           f'Completed arms are saved; rerun with --out {out} to resume.')
    print(f'All GPUs finished. Joint report: {out / "summary.txt"}', flush=True)


def main():
    args = parse_args()
    global torch
    import torch
    if args.device == 'cuda_if_available':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    out = (args.out or OUT_ROOT / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}').expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    print('OUTPUT:', out, flush=True)
    sessions = choose_sessions(args, out)
    missing = [s for s in sessions if not (args.data_dir.expanduser() / f'{s}.npz').is_file()]
    if missing:
        raise FileNotFoundError(f'NPZ not found in {args.data_dir} for sessions: {missing}')
    if args.gpus:
        run_parallel(args, out, sessions)
        return
    cebra = load_fork(args.cebra_dir)
    cebra.CEBRA(**encoder_config(1.0, args))  # fail early on a constructor mismatch
    save_json(out / 'config.json', dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                        sessions=sessions, evaluation='clean-input VALID R2, all label columns'))

    stats, rows = {}, []
    for session in sessions:
        data = load_session(args.data_dir, session)
        stats[session] = session_epsilons(data[0], data[2], args.fixed_epsilon)
        print(f'\nSESSION {session}: train {data[0].shape}, valid {data[1].shape}, epsilon fixed='
              f'{stats[session]["epsilon"]["fixed_eps"]:.3f} auto={stats[session]["epsilon"]["auto_eps"]:.3f}',
              flush=True)
        for seed in args.seeds:
            for arm in ARMS:
                folder = out / session / f'seed_{seed}' / arm
                if (folder / 'metrics.json').is_file():
                    rows.append(json.loads((folder / 'metrics.json').read_text()))
                    print(f'skip (done): {session} {arm} seed {seed}', flush=True)
                    continue
                rows.append(run_one(cebra, session, arm, seed, data, stats[session], folder, args, device))
                write_summary(out, stats, sessions, rows, args)
        del data
        cleanup()
    write_summary(out, stats, sessions, rows, args)
    print('\nDone. Send this file:', out / 'summary.txt', flush=True)


if __name__ == '__main__':
    main()

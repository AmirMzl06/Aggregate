#!/usr/bin/env python3
"""Perich Exp14: prospective frozen-v5 validation on 46 new sessions, 2 seeds.

Put NEXT TO the unmodified Exp13 `dprime_exp9.py` and `Acorn-dprime/`.
No training/evaluation algorithms are implemented here: this is a selection,
pre-registration and orchestration wrapper for the VERIFIED Exp13 runner.

Intended commands, inside the RunAI container with /data mounted:
  python -u perich_exp14_46_runner.py --stage prepare
  python -u perich_exp14_46_runner.py --stage plan --batch-index 0
  python -u perich_exp14_46_runner.py --stage run --batch-index 0 --gpu-workers 4
  python -u perich_exp14_46_runner.py --stage run --batch-index 1 --gpu-workers 4
  ... (indices 0,1,2,3,4 at batch-size 10)
  python -u perich_exp14_46_runner.py --stage status

Notes:
* 46 sessions are drawn once, without replacement, using selection seed 2026
  from the 92 previously unused Perich sessions. Session exclusions are the
  19 documented historical Perich sessions (Exp5-13).
* Only training seeds 42 and 43 are used. Supervised has seven arms; time-only
  has six. Total: 46 * 2 * 13 = 1196 model runs (644 sup + 552 time).
* `prepare` writes the frozen selection, the stock runner's PREDICTIONS files,
  and frozen kappa-based reference-margin predictions BEFORE any new training.
* Output directories are separate from Exp11-13; existing checkpoint files
  are never overwritten and --resume is always enabled.
* `--stage run` refuses to start before signed pre-registration is complete.
* Uses the original Exp13 `dprime_exp9.py` via --suite custom, with v2 evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path('/data/hossein/mm_project/perich_data_valid_final_raw')
FORK = ROOT / 'Acorn-dprime'
ORIGINAL = ROOT / 'dprime_exp9.py'
SELECTION = ROOT / 'DPRIME_EXP14_PERICH46_SELECTION.json'
LOCK = ROOT / 'DPRIME_EXP14_PERICH46_PREREG_LOCK.json'

N_SELECT = 46
DRAW_SEED = 2026
TRAIN_SEEDS = (42, 43)
WINDOW = 36
MIN_TRAIN_BINS = 2000
MIN_CHANNELS = 10

# 19 DIFFERENT Perich sessions documented in completed Exp5--Exp13 runs.
EXCLUDED = frozenset({
    'C-CO12', 'C-CO13', 'C-RT13', 'C-RT6', 'M-CO9', 'M-RT3',
    'T-RT1', 'T-RT4', 'J-CO0',
    'C-CO25', 'C-CO26', 'T-RT2', 'C-CO2', 'T-CO1', 'C-CO39',
    'C-CO21', 'J-CO1', 'M-CO7', 'C-RT12',
})

ARMS = {
    'supervised': (
        'CLEAN', 'PGD_L2_EPS6', 'MADRY_SPDIFF_D8',
        'MADRY_SPDIFF_AUTOBEH', 'MADRY_SP_AUTOBEH_POOL',
        'MADRY_SP_D8', 'MADRY_SPDIFF_GAUSS',
    ),
    'time_only': (
        'CLEAN', 'PGD_L2_EPS6', 'MADRY_SPDIFF_D8',
        'MADRY_SPDIFF_AUTOSIG', 'MADRY_SP_AUTOSIG_POOL',
        'MADRY_SPDIFF_GAUSS',
    ),
}
OUTPUTS = {
    'supervised': ROOT / 'DPRIME_EXP14_PERICH46_SUPERVISED',
    'time_only': ROOT / 'DPRIME_EXP14_PERICH46_TIMEONLY',
}

# Prospectively frozen values provided after Exp13 by the supervisor:
# per-arm median kappa = eps_half_refined/reference_eps_beh on the 10 Perich
# sessions in the Exp11--13 validation cohort. Never re-fit on Exp14.
FROZEN_K = {
    'supervised': {
        'CLEAN': 0.080, 'PGD_L2_EPS6': 0.124,
        'MADRY_SPDIFF_D8': 0.151,
        'MADRY_SPDIFF_AUTOBEH': 0.150,
        'MADRY_SP_AUTOBEH_POOL': 0.137,
        'MADRY_SP_D8': 0.070,
        'MADRY_SPDIFF_GAUSS': 0.103,
    },
    'time_only': {
        'CLEAN': 0.035, 'PGD_L2_EPS6': 0.116,
        'MADRY_SPDIFF_D8': 0.139,
        'MADRY_SPDIFF_AUTOSIG': 0.133,
        'MADRY_SP_AUTOSIG_POOL': 0.140,
        'MADRY_SPDIFF_GAUSS': 0.106,
    },
}

# Exact 8 source hashes from the actual Exp13_bundle.zip v5; the fork is frozen.
FROZEN_EXP13_RUNNER_SHA256 = '08500859dd8e8b44998254a9b844a5ae2db71c86b9e46b4b1eef120f5cdc389e'
FROZEN_FORK_SHA256 = {
    'cebra/distributions/base.py': '75c49d4d0941685ce2d41e106cea549f7cff3e5a28c29aacdd3168f9c07def90',
    'cebra/dprime/training.py': 'c7756e1479bf48417f0f4e6bb5f80f05fc5650a080181bd48405892b86a15913',
    'cebra/dprime/evaluation.py': '12f1b2f52b28cd1da7db77a37151375696a4610232467d437f10c23e374a413f',
    'cebra/dprime/noise.py': '4de3e1278bbc823ddb3d39561eae1b4986ed92225e336296994d07f7ed901402',
    'cebra/dprime/attacks.py': 'c7540bc19d4b832790a764699d3e99c4563e5fe0dca43371be6c6219dc884b20',
    'cebra/dprime/calibration.py': '245156612bec860b62e0b56eef8ba63d10f19df247d2cc7d174eed0ca749f69c',
    'cebra/integrations/sklearn/cebra.py': '41503cb7abffcce90d3d6042be3b894c1e248d2deaf86362c267a6924634af02',
    'cebra/solver/base.py': 'a6b191f44fdb449fc06fdcd3572b2d16abcb8019b4560fd1f30d48d02132c1fc',
}



def fail(message: str) -> None:
    raise SystemExit('ERROR: ' + message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    fd, temp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def imported_runner():
    if not ORIGINAL.is_file():
        fail(f'Missing official Exp13 runner at {ORIGINAL}')
    spec = importlib.util.spec_from_file_location('dprime_exp13_original', ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_environment(data_needed: bool = True):
    if not ORIGINAL.is_file():
        fail('Put this file NEXT TO dprime_exp9.py')
    if sha256(ORIGINAL) != FROZEN_EXP13_RUNNER_SHA256:
        fail('dprime_exp9.py is NOT the unmodified Exp13 runner from Exp13_bundle.zip')
    for rel, expected in FROZEN_FORK_SHA256.items():
        f = FORK / rel
        if not f.is_file():
            fail(f'Missing frozen fork file {f}')
        actual = sha256(f)
        if actual != expected:
            fail(f'FROZEN FORK MISMATCH: {rel}\nExpected {expected}\nActual   {actual}')
    if data_needed and not DATA.is_dir():
        fail(f'Data not mounted: {DATA}. Execute inside interactive/RunAI with /data mounted.')


def pick_sessions(runner):
    import numpy as np
    if SELECTION.exists():
        return read_selection()
    available = sorted(p.stem for p in DATA.glob('*.npz'))
    if len(available) != 111 or len(set(available)) != 111:
        fail(f'Expected 111 unique Perich NPZ files; found {len(available)}')
    if not EXCLUDED.issubset(available):
        fail('Previously used sessions not found: ' + repr(sorted(EXCLUDED - set(available))))
    candidates = sorted(set(available) - EXCLUDED)
    if len(candidates) != 92:
        fail(f'Expected exactly 92 unused sessions; found {len(candidates)}')
    eligible, rejected = [], []
    for session in candidates:
        path = DATA / (session + '.npz')
        try:
            xtrain, xvalid, ytrain, yvalid = runner.load_data(path)
            n, channels = xtrain.shape
            if n < MIN_TRAIN_BINS or channels < MIN_CHANNELS:
                rejected.append({'session': session,
                                 'reason': f'{n} train bins, {channels} neurons'})
                continue
            eligible.append(session)
        except Exception as e:
            rejected.append({'session': session, 'reason': f'{type(e).__name__}: {e}'})
    if len(eligible) < N_SELECT:
        fail(f'Only {len(eligible)}/92 sessions usable; need {N_SELECT}; rejected={rejected}')
    # One random draw without replacement. Never redrawn once saved.
    chosen = [str(s) for s in np.random.default_rng(DRAW_SEED).choice(
        sorted(eligible), size=N_SELECT, replace=False).tolist()]
    info = []
    for session in chosen:
        path = DATA / (session + '.npz')
        with np.load(path, allow_pickle=False) as f:
            n_train, c = f['train_data'].shape
            n_valid = f['valid_data'].shape[0]
            label_dims = f['train_label'].shape[-1] if f['train_label'].ndim > 1 else 1
        info.append({'session': session, 'n_train': int(n_train), 'n_valid': int(n_valid),
                     'neurons': int(c), 'label_dims': int(label_dims),
                     'data_sha256': sha256(path)})
    selection = {
        'schema': 'frozen-exp14-perich46-v1', 'created_utc': stamp(),
        'data_dir': str(DATA), 'selection_seed': DRAW_SEED,
        'requested_n': N_SELECT, 'training_seeds': list(TRAIN_SEEDS),
        'n_available': len(available), 'n_candidates': len(candidates),
        'n_eligible': len(eligible), 'excluded': sorted(EXCLUDED),
        'eligible': eligible, 'rejected': rejected,
        'chosen': chosen, 'chosen_info': info,
        'arms': {k: list(v) for k, v in ARMS.items()},
        'frozen_kappa': FROZEN_K,
        'fork_sha256': FROZEN_FORK_SHA256,
        'original_runner_sha256': sha256(ORIGINAL),
    }
    atomic_json(SELECTION, selection)
    print(f'[SELECT] {len(available)} total -> {len(candidates)} unused -> '
          f'{len(eligible)} eligible -> {len(chosen)} FROZEN random sessions', flush=True)
    for i, name in enumerate(chosen, 1):
        print(f'  {i:02d}. {name}', flush=True)
    print(f'[SELECT] Frozen manifest: {SELECTION}', flush=True)
    if rejected:
        print(f'[SELECT] Skipped {len(rejected)} ineligible sessions; see selection manifest.')
    return selection


def read_selection():
    if not SELECTION.is_file():
        fail(f'Selection missing; run --stage prepare first: {SELECTION}')
    s = json.loads(SELECTION.read_text(encoding='utf-8'))
    chosen = s.get('chosen', [])
    if (s.get('schema') != 'frozen-exp14-perich46-v1' or len(chosen) != N_SELECT
            or len(set(chosen)) != N_SELECT or set(chosen) & EXCLUDED
            or s.get('training_seeds') != list(TRAIN_SEEDS)
            or s.get('selection_seed') != DRAW_SEED
            or s.get('excluded') != sorted(EXCLUDED)
            or s.get('arms') != {k: list(v) for k, v in ARMS.items()}
            or s.get('frozen_kappa') != FROZEN_K
            or s.get('fork_sha256') != FROZEN_FORK_SHA256):
        fail('Frozen session manifest differs from intended Exp14 protocol')
    if s.get('original_runner_sha256') != sha256(ORIGINAL):
        fail('Official dprime_exp9.py changed after selection; restore Exp13 runner')
    return s


def verify_data(s):
    for item in s['chosen_info']:
        p = DATA / (item['session'] + '.npz')
        if not p.is_file() or sha256(p) != item['data_sha256']:
            fail(f'Data changed/missing after selection: {item["session"]}')


def plan_args(runner, mode, sessions):
    cli = ['--suite', 'custom', '--mode', mode, '--arms', *ARMS[mode],
           '--sessions', *sessions, '--seeds', *map(str, TRAIN_SEEDS),
           '--output-root', str(OUTPUTS[mode]), '--data-dir', str(DATA),
           '--fork-dir', str(FORK), '--max-iterations', '3000',
           '--eval-tag', 'v2', '--eval-refine', '4', '--eval-kappa',
           '--no-eval-margins']
    args = runner.build_parser().parse_args(cli)
    return args, runner.resolve_plan(args)


def predictions_for_mode(runner, mode, chosen):
    args, plan = plan_args(runner, mode, chosen)
    runner.write_predictions(args, plan)  # off-the-shelf Exp13 prediction routine
    path = OUTPUTS[mode] / 'PREDICTIONS.csv'
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    expected = {(s, arm) for s in chosen for arm in ARMS[mode]}
    actual = [(r['session'], r['arm']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        fail(f'Incomplete/non-unique pre-registration for {mode}: '
             f'found {len(actual)} expected {len(expected)}')
    return path


def frozen_half_predictions(runner, chosen, mode):
    # Same reference geometry and eps_beh estimator used by fork v5's evaluator.
    runner.load_fork(FORK, needs_eval_options=True)
    from cebra.dprime import calibration, evaluation
    rows = []
    for session in chosen:
        x, _, y, _ = runner.load_data(DATA / f'{session}.npz')
        model = evaluation.common_metrics(x, window_len=WINDOW, labels=None,
                                          residuals='whole')['reference']
        yhat, info = calibration.behaviour_prediction(x, y)
        eps_beh = float(calibration.eps_behaviour(
            x, yhat, model.spatial_cov, model.temporal_cov))
        if not (math.isfinite(eps_beh) and eps_beh > 0):
            fail(f'Non-finite/non-positive reference eps_beh for {session}: {eps_beh}')
        for arm in ARMS[mode]:
            k = FROZEN_K[mode][arm]
            rows.append({'session': session, 'mode': mode, 'arm': arm,
                         'eps_beh_reference': f'{eps_beh:.12g}',
                         'kappa_frozen': f'{k:.6f}',
                         'pred_eps_half_reference': f'{eps_beh * k:.12g}',
                         'prediction_rule': 'fixed_Exp13_median_kappa_times_eps_beh_ref',
                         'behaviour_r2': f'{info["r2"]:.12g}',
                         'registered_utc': stamp()})
        print(f'[FROZEN KAPPA] {mode:10s} {session:8s} '
              f'eps_beh_ref={eps_beh:.4f}', flush=True)
    output = OUTPUTS[mode] / 'FROZEN_REF_PREDICTIONS.csv'
    if output.exists():
        # Never silently replace predictions which may have been logged before training.
        with output.open(newline='', encoding='utf-8') as f:
            old = list(csv.DictReader(f))
        cols = ('session', 'mode', 'arm', 'eps_beh_reference',
                'kappa_frozen', 'pred_eps_half_reference')
        if [{c: row[c] for c in cols} for row in old] != [
                {c: row[c] for c in cols} for row in rows]:
            fail(f'Frozen prospective predictions changed: {output}')
    else:
        with io.StringIO() as out:
            fields = list(rows[0])
            w = csv.DictWriter(out, fields)
            w.writeheader()
            w.writerows(rows)
            runner.atomic_write_text(output, out.getvalue())
    return output


def report_paths():
    result = {}
    for mode in ARMS:
        result[mode] = [OUTPUTS[mode] / 'PREDICTIONS.csv',
                        OUTPUTS[mode] / 'PREDICTIONS.json',
                        OUTPUTS[mode] / 'FROZEN_REF_PREDICTIONS.csv']
    return result


def check_no_training_yet():
    for mode, output in OUTPUTS.items():
        if output.exists() and (list(output.glob('*/seed*/models/*.pt')) or
                                list(output.glob('*/seed*/records*/*.json')) or
                                list(output.glob('*/seed*/records*/*/*.json'))):
            fail(f'Existing training/evaluation results in {output} but no valid '
                 'preregistration lock. Inspect before continuing.')


def create_or_check_lock():
    paths = [p for seq in report_paths().values() for p in seq]
    if any(not p.is_file() for p in paths):
        fail('One or more prospective predictions files are missing')
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    expected = {'schema': 'frozen-exp14-perich46-lock-v1',
                'selection_sha256': sha256(SELECTION), 'files_sha256': hashes}
    if LOCK.exists():
        previous = json.loads(LOCK.read_text(encoding='utf-8'))
        if {k: previous.get(k) for k in expected} != expected:
            fail('Pre-registration lock/hash mismatch! Do NOT train or overwrite files.')
    else:
        atomic_json(LOCK, {'created_utc': stamp(), **expected})
    print(f'[LOCK] All prospective predictions frozen: {LOCK}', flush=True)


def preregister(runner, s):
    if LOCK.exists():
        create_or_check_lock()
        return
    check_no_training_yet()
    verify_data(s)
    for mode in ('supervised', 'time_only'):
        print(f'========== PRE-REGISTER {mode.upper()} ==========', flush=True)
        predictions_for_mode(runner, mode, s['chosen'])
        frozen_half_predictions(runner, s['chosen'], mode)
    create_or_check_lock()


def commands(s, gpu_workers, batch_index, batch_size, only_mode):
    chosen = s['chosen']
    if batch_index is not None:
        start = batch_index * batch_size
        chosen = chosen[start:start + batch_size]
        if not chosen:
            fail(f'Batch {batch_index} out of bounds (46 sessions, batch size {batch_size})')
    modes = list(ARMS) if only_mode == 'both' else [only_mode]
    result = []
    for mode in modes:
        cmd = [sys.executable, '-u', str(ORIGINAL), '--suite', 'custom',
               '--mode', mode, '--arms', *ARMS[mode], '--sessions', *chosen,
               '--seeds', '42', '43', '--max-iterations', '3000',
               '--data-dir', str(DATA), '--fork-dir', str(FORK),
               '--output-root', str(OUTPUTS[mode]), '--gpu-workers', str(gpu_workers),
               '--eval-tag', 'v2', '--eval-refine', '4', '--eval-kappa',
               '--no-eval-margins', '--resume']
        result.append((mode, cmd))
    return chosen, result


def status():
    print('========== EXP14 CURRENT STATUS ==========', flush=True)
    for mode, arms in ARMS.items():
        path = OUTPUTS[mode] / 'summary_v2.csv'
        total = N_SELECT * len(TRAIN_SEEDS) * len(arms)
        if not path.is_file():
            print(f'{mode:11s} 0/{total} (no summary)', flush=True)
            continue
        with path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        done = [r for r in rows if r.get('status') == 'success']
        print(f'{mode:11s} successful={len(done)}/{total} total_rows={len(rows)} '
              f'errors={sum(r.get("status") != "success" for r in rows)}', flush=True)
    print('Pre-registration lock:', LOCK.exists(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', choices=('prepare', 'plan', 'run', 'all', 'status'),
                   required=True)
    p.add_argument('--gpu-workers', type=int, default=4)
    p.add_argument('--batch-index', type=int, default=None,
                   help='Zero-based batch number; omit to run ALL 46 selected sessions')
    p.add_argument('--batch-size', type=int, default=10)
    p.add_argument('--only-mode', choices=('both', 'supervised', 'time_only'), default='both')
    a = p.parse_args()
    if a.gpu_workers < 1 or a.batch_size < 1 or (
            a.batch_index is not None and a.batch_index < 0):
        fail('GPU workers and batch size must be positive; batch index must be >= 0')
    if a.stage == 'status':
        status()
        return
    check_environment(data_needed=True)
    runner = imported_runner()
    if a.stage in ('prepare', 'all'):
        s = pick_sessions(runner)
        preregister(runner, s)
        if a.stage == 'prepare':
            print('[READY] 46 frozen sessions, 2 seeds, 13 arms; 1196 jobs total.')
            print('[NEXT] --stage plan --batch-index 0, then --stage run --batch-index 0')
            return
    else:
        s = read_selection()
    chosen, cmds = commands(s, a.gpu_workers, a.batch_index, a.batch_size,
                            a.only_mode)
    print('========== FROZEN EXP14 RUN PLAN ==========', flush=True)
    print('selected in this call:', len(chosen), chosen, flush=True)
    print('training seeds:', list(TRAIN_SEEDS), flush=True)
    print('total jobs in this call:', len(chosen) * len(TRAIN_SEEDS) *
          sum(len(ARMS[m]) for m, _ in cmds), flush=True)
    for mode, cmd in cmds:
        print(f'[{mode}] {shlex.join(cmd)}', flush=True)
    if a.stage == 'plan':
        return
    verify_data(s)
    create_or_check_lock()  # requires all predictions were saved prior to training
    for mode, cmd in cmds:
        print(f'========== START {mode.upper()} ==========', flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
    print('========== EXP14 REQUESTED JOBS COMPLETED ==========', flush=True)
    status()


if __name__ == '__main__':
    main()

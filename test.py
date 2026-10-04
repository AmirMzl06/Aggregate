"""Calibrate a constant ACORN epsilon, then compare it with a fixed epsilon grid.

Default experiment:
  * Session: C-CO12, using the NPZ train/valid split as provided.
  * CEBRA fork: ``CEBRA_DIR`` imported from ``utils.constants``.
  * Arms: clean, fixed eps={0.1, 0.5, 1, 3, 5, 7}, and
    {0.5, 1, 1.5} times the data-calibrated epsilon.
  * CEBRA and the decoder use ALL label columns.
  * The calibration uses only train_data/train_label. valid_data is untouched.
  * Every trained embedding gets the same full-batch two-layer MLP decoder.

The calibrated epsilon is the largest constant L-infinity budget for which a
PGD attack against an independent raw-window ridge decoder reduces its mean R2
by no more than ``tau`` (an absolute R2 drop).
"""

from pathlib import Path
from datetime import datetime, timezone
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
from torch import nn
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent

# The original ACORN fork used by this project.
from utils.constants import CEBRA_DIR

CEBRA_DIR = Path(CEBRA_DIR).expanduser().resolve()
PERICH_DATA_DIR = Path('/data/hossein/mm_project/perich_data_valid_final_raw/')
SESSION = 'C-CO12'
OUT_ROOT = ROOT / 'ACORN_CALIBRATED_EPS_GRID_CCO12'

# CEBRA + attack.
LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 3000
TEMPERATURE = 0.4
MODEL_ARCH = 'offset36-model-more-dropout'
TIME_OFFSETS = 1
LEARNING_RATE = 3e-4
DEVICE = 'cuda_if_available'
ADV_STEPS = 10
ATTACK_NORM = 'linf'
TRAIN_ALPHA_RATIO = 0.2  # adv_alpha = epsilon / 5

# Comparison arms.
FIXED_EPSILONS = (0.1, 0.5, 1.0, 3.0, 5.0, 7.0)
CALIBRATED_MULTIPLIERS = (0.5, 1.0, 1.5)
SEEDS = (42,)
SAVE_CHECKPOINTS = True

# Independent reference decoder and epsilon calibration.
WINDOW_LEFT = 18
WINDOW_RIGHT = 18
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
RIDGE_FIT_FRACTION = 0.70
RIDGE_SELECTION_FRACTION = 0.10
CALIBRATION_TAU = 0.05
CALIBRATION_STEPS = 20
CALIBRATION_RESTARTS = 2
CALIBRATION_BISECTION_ITERS = 10
CALIBRATION_MAX_EPSILON = 20.0
CALIBRATION_BATCH_SIZE = 1024
CALIBRATION_ATTACK_SEED = 20261004

# Same decoder as the previous ACORN comparison runners.
MLP_EPOCHS = 2500
MLP_HIDDEN = 64
MLP_DROP = 0.4
MLP_LR = 1e-3
EVAL_BATCH_SIZE = 8192


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


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_cebra_fork():
    if not (CEBRA_DIR / 'cebra' / '__init__.py').is_file():
        raise FileNotFoundError(
            f'Original ACORN/CEBRA checkout not found at {CEBRA_DIR}. '
            'Check utils.constants.CEBRA_DIR.'
        )
    for name in list(sys.modules):
        if name == 'cebra' or name.startswith('cebra.'):
            del sys.modules[name]
    while str(CEBRA_DIR) in sys.path:
        sys.path.remove(str(CEBRA_DIR))
    sys.path.insert(0, str(CEBRA_DIR))
    import cebra
    imported = Path(cebra.__file__).resolve()
    if CEBRA_DIR not in imported.parents:
        raise RuntimeError(f'Wrong CEBRA imported: {imported}; expected under {CEBRA_DIR}')
    params = inspect.signature(cebra.CEBRA.__init__).parameters
    required = {'training_mode', 'adv_epsilon', 'adv_alpha', 'adv_steps', 'attack_norm'}
    missing = sorted(required - set(params))
    if missing:
        raise RuntimeError(f'The ACORN fork is missing constructor arguments: {missing}')
    print('Using original ACORN fork:', imported, flush=True)
    return cebra, params


def load_data(data_dir, session):
    path = data_dir.expanduser() / f'{session}.npz'
    if not path.is_file():
        raise FileNotFoundError(f'Dataset not found: {path}')
    with np.load(path, allow_pickle=False) as data:
        required = ('train_data', 'valid_data', 'train_label', 'valid_label')
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f'{path} is missing arrays: {missing}')
        arrays = [np.asarray(data[key], dtype=np.float32) for key in required]
    x_train, x_valid, y_train, y_valid = arrays
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_valid.ndim == 1:
        y_valid = y_valid[:, None]
    arrays = (x_train, x_valid, y_train, y_valid)
    for name, value in zip(('X_train', 'X_valid', 'Y_train', 'Y_valid'), arrays):
        if value.ndim != 2 or min(value.shape) == 0:
            raise ValueError(f'{name} must be a nonempty 2D array; got {value.shape}')
        if not np.isfinite(value).all():
            raise ValueError(f'{name} contains NaN or infinity.')
    if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
        raise ValueError('Feature and label lengths do not match.')
    if x_train.shape[1] != x_valid.shape[1]:
        raise ValueError('Train and validation neuron counts differ.')
    if y_train.shape[1] != y_valid.shape[1]:
        raise ValueError('Train and validation label dimensions differ.')
    return path, tuple(np.ascontiguousarray(a) for a in arrays)


# -----------------------------------------------------------------------------
# Independent raw-window ridge decoder used only to calibrate epsilon.
# -----------------------------------------------------------------------------
class RidgeDecoder:
    def __init__(self, mean, std, weight, bias):
        self.mean = mean
        self.std = std
        self.weight = weight
        self.bias = bias

    def __call__(self, windows):
        features = ((windows - self.mean.view(1, -1, 1)) /
                    self.std.view(1, -1, 1)).flatten(1)
        return features @ self.weight + self.bias


def make_padded(neural, left, right):
    padded = np.pad(neural, ((left, right - 1), (0, 0)), mode='edge')
    return torch.from_numpy(np.ascontiguousarray(padded))


def window_batches(padded, num_samples, window, device, batch_size):
    offsets = torch.arange(window, device=device)
    for start in range(0, num_samples, batch_size):
        index = torch.arange(start, min(start + batch_size, num_samples), device=device)
        windows = padded[index[:, None] + offsets].transpose(1, 2).contiguous()
        yield index, windows


def torch_r2_per_output(y_true, y_pred):
    residual = ((y_true - y_pred) ** 2).sum(dim=0)
    total = ((y_true - y_true.mean(dim=0)) ** 2).sum(dim=0)
    return 1 - residual / total.clamp(min=1e-12)


def ridge_predictions(decoder, neural, left, right, device, batch_size):
    padded = make_padded(neural, left, right).to(device)
    window = left + right
    result = []
    with torch.inference_mode():
        for _, windows in window_batches(padded, len(neural), window, device, batch_size):
            result.append(decoder(windows))
    return torch.cat(result)


def fit_reference_ridge(x_fit, y_fit, x_select, y_select, left, right,
                        device, batch_size):
    window = left + right
    mean = torch.from_numpy(x_fit.mean(axis=0)).to(device)
    std = torch.from_numpy(x_fit.std(axis=0)).clamp(min=1e-6).to(device)
    padded = make_padded(x_fit, left, right).to(device)
    y = torch.from_numpy(y_fit).double().to(device)
    dim = x_fit.shape[1] * window
    gram = torch.zeros(dim, dim, dtype=torch.float64, device=device)
    cross = torch.zeros(dim, y.shape[1], dtype=torch.float64, device=device)
    feature_sum = torch.zeros(dim, dtype=torch.float64, device=device)
    for index, windows in window_batches(padded, len(x_fit), window, device, batch_size):
        features = ((windows - mean.view(1, -1, 1)) /
                    std.view(1, -1, 1)).flatten(1).double()
        gram += features.T @ features
        cross += features.T @ y[index]
        feature_sum += features.sum(dim=0)
    count = len(x_fit)
    feature_mean = feature_sum / count
    label_mean = y.mean(dim=0)
    gram -= count * torch.outer(feature_mean, feature_mean)
    cross -= count * torch.outer(feature_mean, label_mean)
    ridge_scale = float(torch.diagonal(gram).mean().clamp(min=1e-12))
    identity = torch.eye(dim, dtype=torch.float64, device=device)
    select_y = torch.from_numpy(y_select).to(device)
    best = None
    print('Selecting ridge penalty on its own TRAIN subsection:', flush=True)
    for penalty in RIDGE_GRID:
        weight = torch.linalg.solve(gram + penalty * ridge_scale * identity, cross)
        bias = label_mean - feature_mean @ weight
        decoder = RidgeDecoder(mean, std, weight.float(), bias.float())
        prediction = ridge_predictions(decoder, x_select, left, right, device, batch_size)
        per_output = torch_r2_per_output(select_y, prediction)
        score = float(per_output.mean())
        print(f'  penalty={penalty:g}: selection mean R2={score:.6f}', flush=True)
        if best is None or score > best['mean_r2']:
            best = dict(mean_r2=score, per_output=per_output.tolist(),
                        penalty=penalty, decoder=decoder)
    return best


def reference_pgd(decoder, x0, target, y_std, epsilon, steps,
                  alpha_ratio, restarts, seed):
    """Per-sample worst constant L-infinity perturbation found by PGD.

    The clean point, every random start, and every PGD iterate compete for the
    best loss. This avoids returning a weaker final iterate.
    """
    def loss_for(value):
        return (((decoder(value) - target) / y_std) ** 2).mean(dim=1)

    with torch.no_grad():
        best_x = x0.clone()
        best_loss = loss_for(x0)
    if epsilon == 0:
        return best_x

    for restart in range(restarts):
        generator = torch.Generator(device=x0.device)
        generator.manual_seed(seed + restart)
        delta = (torch.rand(x0.shape, device=x0.device, generator=generator) * 2 - 1) * epsilon
        with torch.no_grad():
            candidate_loss = loss_for(x0 + delta)
            better = candidate_loss > best_loss
            best_x[better] = (x0 + delta)[better]
            best_loss = torch.where(better, candidate_loss, best_loss)
        for _ in range(steps):
            delta.requires_grad_(True)
            loss = loss_for(x0 + delta)
            gradient = torch.autograd.grad(loss.sum(), delta)[0]
            with torch.no_grad():
                delta = (delta + alpha_ratio * epsilon * gradient.sign()).clamp(
                    min=-epsilon, max=epsilon)
                candidate = x0 + delta
                candidate_loss = loss_for(candidate)
                better = candidate_loss > best_loss
                best_x[better] = candidate[better]
                best_loss = torch.where(better, candidate_loss, best_loss)
    return best_x.detach()


def evaluate_reference_attack(decoder, x_calibrate, y_calibrate, left, right,
                              epsilon, args, device):
    padded = make_padded(x_calibrate, left, right).to(device)
    y = torch.from_numpy(y_calibrate).to(device)
    y_std = y.std(dim=0, unbiased=False).clamp(min=1e-8)
    window = left + right
    predictions = []
    abs_delta = []
    # Reset per epsilon so the comparison uses matching random starts.
    batch_number = 0
    for index, x0 in window_batches(
            padded, len(x_calibrate), window, device, args.calibration_batch_size):
        target = y[index]
        attacked = reference_pgd(
            decoder, x0, target, y_std, epsilon,
            args.calibration_steps, args.calibration_alpha_ratio,
            args.calibration_restarts,
            args.calibration_attack_seed + batch_number * 10_000,
        )
        with torch.no_grad():
            predictions.append(decoder(attacked))
            abs_delta.append((attacked - x0).abs().mean(dim=(1, 2)))
        batch_number += 1
    prediction = torch.cat(predictions)
    per_output = torch_r2_per_output(y, prediction)
    return dict(
        mean_r2=float(per_output.mean()),
        r2_per_output=per_output.tolist(),
        mean_abs_delta=float(torch.cat(abs_delta).mean()),
    )


def calibrate_epsilon(decoder, x_calibrate, y_calibrate, left, right, args, device):
    clean_prediction = ridge_predictions(
        decoder, x_calibrate, left, right, device, args.calibration_batch_size)
    y = torch.from_numpy(y_calibrate).to(device)
    clean_per_output = torch_r2_per_output(y, clean_prediction)
    clean_mean = float(clean_per_output.mean())
    trace = {0.0: dict(epsilon=0.0, mean_r2=clean_mean,
                       r2_per_output=clean_per_output.tolist(),
                       damage=0.0, mean_abs_delta=0.0)}

    def evaluate(epsilon):
        epsilon = float(epsilon)
        key = round(epsilon, 12)
        if key not in trace:
            result = evaluate_reference_attack(
                decoder, x_calibrate, y_calibrate, left, right,
                epsilon, args, device)
            result.update(epsilon=epsilon, damage=max(0.0, clean_mean - result['mean_r2']))
            trace[key] = result
            print(f"  eps={epsilon:.8g}: R2={result['mean_r2']:.6f}, "
                  f"drop={result['damage']:.6f}, mean|delta|={result['mean_abs_delta']:.6f}",
                  flush=True)
        return trace[key]

    start = args.calibration_start
    if start is None:
        active_std = x_calibrate.std(axis=0)
        active_std = active_std[active_std > 0]
        median_std = float(np.median(active_std)) if len(active_std) else 1.0
        start = max(0.5 * median_std, 1e-6)
    start = min(start, args.calibration_max_epsilon)
    print(f'Calibrating epsilon: clean R2={clean_mean:.6f}, tau={args.tau:g}, '
          f'start={start:.6g}', flush=True)

    lo, hi = 0.0, start
    while evaluate(hi)['damage'] <= args.tau and hi < args.calibration_max_epsilon:
        lo = hi
        hi = min(2 * hi, args.calibration_max_epsilon)
        if hi == lo:
            break
    if evaluate(hi)['damage'] <= args.tau:
        selected = hi
        status = 'reached calibration maximum before exceeding tau'
    else:
        for _ in range(args.calibration_bisection_iters):
            middle = (lo + hi) / 2
            if evaluate(middle)['damage'] <= args.tau:
                lo = middle
            else:
                hi = middle
        selected = lo
        status = 'ok'

    selected_result = evaluate(selected)
    return dict(
        epsilon=selected,
        status=status,
        tau=args.tau,
        clean_mean_r2=clean_mean,
        clean_r2_per_output=clean_per_output.tolist(),
        selected_result=selected_result,
        trace=sorted(trace.values(), key=lambda item: item['epsilon']),
    )


def temporal_calibration_splits(x, y, window):
    """Create fit/penalty-selection/calibration blocks with window-sized gaps."""
    n = len(x)
    fit_end = int(n * RIDGE_FIT_FRACTION)
    select_start = fit_end + window
    select_end = int(n * (RIDGE_FIT_FRACTION + RIDGE_SELECTION_FRACTION))
    calibrate_start = select_end + window
    if fit_end < window or select_end - select_start < window or n - calibrate_start < window:
        raise ValueError('Training split is too short for ridge fit/selection/calibration blocks.')
    return dict(
        fit=(x[:fit_end], y[:fit_end]),
        selection=(x[select_start:select_end], y[select_start:select_end]),
        calibration=(x[calibrate_start:], y[calibrate_start:]),
        indices=dict(fit=[0, fit_end], selection=[select_start, select_end],
                     calibration=[calibrate_start, n], discarded_gap=window),
    )


# -----------------------------------------------------------------------------
# CEBRA and downstream decoder.
# -----------------------------------------------------------------------------
class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, MLP_HIDDEN),
            nn.LayerNorm(MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(MLP_DROP),
            nn.Linear(MLP_HIDDEN, output_dim),
        )

    def forward(self, value):
        return self.net(value)


def train_decoder(z_train, y_train, seed, device):
    seed_all(seed)
    decoder = TwoLayerMLP(z_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=MLP_LR)
    criterion = nn.MSELoss()
    z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    losses = []
    decoder.train()
    for epoch in range(MLP_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(decoder(z), y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite decoder loss at epoch {epoch + 1}.')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if (epoch + 1) % 500 == 0 or epoch + 1 == MLP_EPOCHS:
            print(f'Decoder {epoch + 1}/{MLP_EPOCHS}: train MSE={losses[-1]:.6f}', flush=True)
    return decoder, losses


def decoder_predictions(decoder, embeddings):
    decoder.eval()
    device = next(decoder.parameters()).device
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), EVAL_BATCH_SIZE):
            value = torch.as_tensor(
                embeddings[start:start + EVAL_BATCH_SIZE],
                dtype=torch.float32, device=device)
            outputs.append(decoder(value).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def numpy_r2(labels, predictions):
    per_output = np.atleast_1d(
        r2_score(labels, predictions, multioutput='raw_values'))
    return float(per_output.mean()), per_output.tolist()


def epsilon_token(value):
    return f'{value:.8g}'.replace('-', 'm').replace('.', 'p')


def build_arm_specs(calibrated_epsilon, fixed_epsilons, multipliers):
    specs = [dict(arm='clean', epsilon=None, source='clean', aliases=[])]

    def add(arm, epsilon, source):
        for spec in specs[1:]:
            if math.isclose(spec['epsilon'], epsilon, rel_tol=1e-10, abs_tol=1e-12):
                spec['aliases'].append(dict(arm=arm, source=source))
                return
        specs.append(dict(arm=arm, epsilon=float(epsilon), source=source, aliases=[]))

    for epsilon in fixed_epsilons:
        add(f'fixed_eps_{epsilon_token(epsilon)}', epsilon, 'fixed_grid')
    for multiplier in multipliers:
        epsilon = calibrated_epsilon * multiplier
        add(f'cal_x{epsilon_token(multiplier)}_eps_{epsilon_token(epsilon)}', epsilon,
            f'{multiplier:g} x calibrated epsilon')
    return specs


def encoder_config(spec, args, constructor_parameters):
    config = dict(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=TIME_OFFSETS,
        max_iterations=args.max_iter,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        learning_rate=LEARNING_RATE,
        temperature_mode='constant',
        training_mode='standard' if spec['epsilon'] is None else 'adversarial',
        device=args.device,
        verbose=True,
    )
    if spec['epsilon'] is not None:
        epsilon = spec['epsilon']
        config.update(
            adv_epsilon=epsilon,
            adv_alpha=epsilon * args.train_alpha_ratio,
            adv_steps=args.adv_steps,
            attack_norm=ATTACK_NORM,
        )
        optional = dict(attack_target='reference', adv_restarts=1, adv_best_iterate=False)
        config.update({key: value for key, value in optional.items()
                       if key in constructor_parameters})
    unknown = sorted(set(config) - set(constructor_parameters))
    if unknown:
        raise RuntimeError(f'CEBRA constructor does not accept: {unknown}')
    return config


def run_arm(cebra, constructor_parameters, spec, seed, arrays, out, args):
    x_train, x_valid, y_train, y_valid = arrays
    folder = out / f'seed_{seed}' / spec['arm']
    folder.mkdir(parents=True, exist_ok=False)
    config = encoder_config(spec, args, constructor_parameters)
    encoder = decoder = z_train = z_valid = None
    started = time.perf_counter()
    try:
        print('\n' + '=' * 100, flush=True)
        print(f"SEED {seed} | {spec['arm']} | epsilon={spec['epsilon']} | {spec['source']}",
              flush=True)
        if spec['aliases']:
            print('Equivalent requested arms (deduplicated):', spec['aliases'], flush=True)
        print(json.dumps(config, indent=2), flush=True)
        seed_all(seed)
        encoder = cebra.CEBRA(**config)
        encoder.fit(x_train, y_train)  # All label columns.
        z_train = np.asarray(encoder.transform(x_train), dtype=np.float32)
        z_valid = np.asarray(encoder.transform(x_valid), dtype=np.float32)
        for name, z, x in [('train', z_train, x_train), ('valid', z_valid, x_valid)]:
            if z.shape != (len(x), LATENT_DIM) or not np.isfinite(z).all():
                raise ValueError(f'{name} embedding is invalid: {z.shape}')
        encoder_seconds = time.perf_counter() - started
        decoder_seed = seed + 100_000
        decoder, losses = train_decoder(
            z_train, y_train, decoder_seed, torch.device(encoder.device_))
        train_prediction = decoder_predictions(decoder, z_train)
        valid_prediction = decoder_predictions(decoder, z_valid)
        train_mean, train_per_output = numpy_r2(y_train, train_prediction)
        valid_mean, valid_per_output = numpy_r2(y_valid, valid_prediction)
        result = dict(
            session=args.session, seed=seed, arm=spec['arm'],
            source=spec['source'], aliases=spec['aliases'], epsilon=spec['epsilon'],
            adv_alpha=None if spec['epsilon'] is None else
                      spec['epsilon'] * args.train_alpha_ratio,
            train_mean_r2=train_mean, valid_mean_r2=valid_mean,
            train_r2_per_output=train_per_output,
            valid_r2_per_output=valid_per_output,
            encoder_seconds=encoder_seconds,
            total_seconds=time.perf_counter() - started,
            encoder_config=config,
        )
        save_json(folder / 'metrics.json', result)
        np.save(folder / 'decoder_train_mse.npy', np.asarray(losses, dtype=np.float32))
        np.savez_compressed(folder / 'validation_predictions.npz',
                            y_true=y_valid, y_pred=valid_prediction)
        if not args.no_checkpoints:
            encoder.save(str(folder / 'cebra.pt'), backend='sklearn')
            torch.save(dict(
                state_dict={key: value.detach().cpu()
                            for key, value in decoder.state_dict().items()},
                input_dim=int(z_train.shape[1]), output_dim=int(y_train.shape[1]),
                hidden=MLP_HIDDEN, dropout=MLP_DROP, seed=decoder_seed,
            ), folder / 'decoder.pt')
        print(f'TRAIN mean R2={train_mean:.6f}', flush=True)
        print(f'VALID mean R2={valid_mean:.6f}', flush=True)
        print('VALID per-output R2:', valid_per_output, flush=True)
        return result
    except Exception as exc:
        save_json(folder / 'failure.json',
                  dict(seed=seed, spec=spec, config=config, error=repr(exc)))
        raise
    finally:
        del encoder, decoder, z_train, z_valid
        cleanup()


def summarize(rows, specs, out):
    output_count = len(rows[0]['valid_r2_per_output'])
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['session', 'seed', 'arm', 'source', 'epsilon', 'adv_alpha',
                         'train_mean_r2', 'valid_mean_r2'] +
                        [f'valid_r2_output_{i}' for i in range(output_count)])
        for row in rows:
            writer.writerow([
                row['session'], row['seed'], row['arm'], row['source'], row['epsilon'],
                row['adv_alpha'], row['train_mean_r2'], row['valid_mean_r2'],
            ] + row['valid_r2_per_output'])
    by_arm = {}
    for spec in specs:
        group = [row for row in rows if row['arm'] == spec['arm']]
        if not group:
            continue
        scores = np.asarray([row['valid_mean_r2'] for row in group])
        per_output = np.asarray([row['valid_r2_per_output'] for row in group])
        by_arm[spec['arm']] = dict(
            source=spec['source'], aliases=spec['aliases'], epsilon=spec['epsilon'],
            mean=float(scores.mean()),
            sample_std=float(scores.std(ddof=1)) if len(scores) > 1 else None,
            per_seed={str(row['seed']): row['valid_mean_r2'] for row in group},
            mean_r2_per_output=per_output.mean(axis=0).tolist(),
        )
    save_json(out / 'summary.json', dict(by_arm=by_arm))
    print('\nFINAL CLEAN-INPUT VALIDATION R2 (all label columns)', flush=True)
    for arm, item in sorted(by_arm.items(), key=lambda pair: pair[1]['mean'], reverse=True):
        print(f"{arm:<38} eps={str(item['epsilon']):<12} R2={item['mean']:.6f} "
              f"std={item['sample_std']} seeds={item['per_seed']}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', default=SESSION)
    parser.add_argument('--data-dir', type=Path, default=PERICH_DATA_DIR)
    parser.add_argument('--out-root', type=Path, default=OUT_ROOT)
    parser.add_argument('--fixed-epsilons', nargs='+', type=float,
                        default=list(FIXED_EPSILONS))
    parser.add_argument('--calibrated-multipliers', nargs='+', type=float,
                        default=list(CALIBRATED_MULTIPLIERS))
    parser.add_argument('--tau', type=float, default=CALIBRATION_TAU,
                        help='Allowed absolute mean-R2 drop of the independent ridge decoder.')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--max-iter', type=int, default=MAX_ITER)
    parser.add_argument('--decoder-epochs', type=int, default=MLP_EPOCHS)
    parser.add_argument('--adv-steps', type=int, default=ADV_STEPS)
    parser.add_argument('--train-alpha-ratio', type=float, default=TRAIN_ALPHA_RATIO)
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--calibration-steps', type=int, default=CALIBRATION_STEPS)
    parser.add_argument('--calibration-restarts', type=int, default=CALIBRATION_RESTARTS)
    parser.add_argument('--calibration-bisection-iters', type=int,
                        default=CALIBRATION_BISECTION_ITERS)
    parser.add_argument('--calibration-max-epsilon', type=float,
                        default=CALIBRATION_MAX_EPSILON)
    parser.add_argument('--calibration-start', type=float, default=None)
    parser.add_argument('--calibration-alpha-ratio', type=float, default=None)
    parser.add_argument('--calibration-batch-size', type=int,
                        default=CALIBRATION_BATCH_SIZE)
    parser.add_argument('--calibration-attack-seed', type=int,
                        default=CALIBRATION_ATTACK_SEED)
    parser.add_argument('--calibration-only', action='store_true')
    parser.add_argument('--no-checkpoints', action='store_true',
                        default=not SAVE_CHECKPOINTS)
    args = parser.parse_args()
    if args.calibration_alpha_ratio is None:
        args.calibration_alpha_ratio = 2.5 / args.calibration_steps
    positive = (args.fixed_epsilons + args.calibrated_multipliers +
                [args.train_alpha_ratio, args.calibration_max_epsilon,
                 args.calibration_alpha_ratio])
    if not np.isfinite(positive).all() or min(positive) <= 0:
        parser.error('Epsilons, multipliers, maxima and alpha ratios must be finite and positive.')
    integer_positive = [args.max_iter, args.decoder_epochs, args.adv_steps,
                        args.calibration_steps, args.calibration_restarts,
                        args.calibration_bisection_iters, args.calibration_batch_size]
    if min(integer_positive) < 1:
        parser.error('Iteration, step, restart and batch values must be positive.')
    if not 0 < args.tau < 1:
        parser.error('--tau must be in (0, 1).')
    if args.calibration_start is not None and args.calibration_start <= 0:
        parser.error('--calibration-start must be positive.')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('Seeds must be unique nonnegative integers.')
    return args


def main():
    global MLP_EPOCHS
    args = parse_args()
    MLP_EPOCHS = args.decoder_epochs
    if args.device == 'cuda_if_available':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    npz_path, arrays = load_data(args.data_dir, args.session)
    x_train, x_valid, y_train, y_valid = arrays
    window = WINDOW_LEFT + WINDOW_RIGHT
    split = temporal_calibration_splits(x_train, y_train, window)
    print('DATA:', npz_path, flush=True)
    print('Shapes:', [value.shape for value in arrays], flush=True)
    print('Calibration train subsections:', split['indices'], flush=True)
    print('VALID split is reserved for final CEBRA decoding only.', flush=True)

    ridge = fit_reference_ridge(
        *split['fit'], *split['selection'], WINDOW_LEFT, WINDOW_RIGHT,
        device, args.calibration_batch_size)
    print(f"Chosen ridge penalty={ridge['penalty']:g}; "
          f"selection R2={ridge['mean_r2']:.6f}", flush=True)
    calibration = calibrate_epsilon(
        ridge['decoder'], *split['calibration'], WINDOW_LEFT, WINDOW_RIGHT,
        args, device)
    calibrated_epsilon = calibration['epsilon']
    print(f"CALIBRATED EPSILON={calibrated_epsilon:.8g} "
          f"[{calibration['status']}]", flush=True)
    print('Requested calibrated arms:',
          {f'{multiplier:g}x': multiplier * calibrated_epsilon
           for multiplier in args.calibrated_multipliers}, flush=True)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out = args.out_root.expanduser() / f'{args.session}_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'calibration.json', dict(
        session=args.session, data=str(npz_path), split=split['indices'],
        reference_ridge={key: value for key, value in ridge.items()
                         if key != 'decoder'},
        calibration=calibration,
    ))
    if args.calibration_only:
        print('Saved calibration:', out / 'calibration.json', flush=True)
        return

    cebra, constructor_parameters = load_cebra_fork()
    specs = build_arm_specs(
        calibrated_epsilon, args.fixed_epsilons, args.calibrated_multipliers)
    configs = {spec['arm']: encoder_config(spec, args, constructor_parameters)
               for spec in specs}
    # Constructor preflight: fail before the expensive grid starts.
    for config in configs.values():
        cebra.CEBRA(**config)
    save_json(out / 'run_config.json', dict(
        args={key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()},
        fork=str(Path(cebra.__file__).resolve()), data_path=str(npz_path),
        session=args.session, data_shapes=[list(value.shape) for value in arrays],
        labels='all columns', calibrated_epsilon=calibrated_epsilon,
        specs=specs, configs=configs,
        decoder=dict(epochs=MLP_EPOCHS, hidden=MLP_HIDDEN,
                     dropout=MLP_DROP, learning_rate=MLP_LR,
                     batch_mode='full'),
    ))
    print('OUTPUT:', out, flush=True)
    print('Unique training arms:', flush=True)
    for spec in specs:
        print(' ', spec, flush=True)

    rows = []
    for seed in args.seeds:
        for spec in specs:
            rows.append(run_arm(
                cebra, constructor_parameters, spec, seed, arrays, out, args))
            save_json(out / 'results_so_far.json', rows)
            summarize(rows, specs, out)
    print('Saved all results:', out, flush=True)


if __name__ == '__main__':
    main()

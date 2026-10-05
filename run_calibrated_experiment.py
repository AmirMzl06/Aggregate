"""ACORN calibrated-budget experiment: calibrate -> train -> robust evaluation, in one script.

Needs only the two forks next to this file (or given with --noise-fork / --structured-fork):
    Acorn-noise-eps          clean, constant and noise arms
    Acorn-structured-attack  gain, baseline and gain_baseline arms

Stages:
  1. CALIBRATE  A ridge decoder on raw input windows is fitted on the first 80% of
                TRAIN; on the last 20% of TRAIN, the largest budget of every threat
                model whose perturbation lowers the ridge R2 by <= tau is found by
                bisection (oracle 'random' by default, 'adversarial' optional).
  2. TRAIN      Every arm x seed: supervised CEBRA (all label columns) trained with
                its calibrated budget, then the same full-batch MLP decoder as the
                previous runners. Checkpoints go to <out>/seed_<s>/<arm>/.
  3. EVALUATE   Every trained arm is perturbed on VALID under every threat model at
                0.25 / 0.5 / 1 x its calibrated budget (--eval-scales), both by
                PGD on encoder + decoder (worst case, eval mode) and by one uniform
                random draw inside the same budget (--no-eval-random to skip).

Output: <out>/summary.txt (send this one), plus calibration.json, results.csv,
summary.json and the checkpoints. Re-running with --out <same folder> resumes:
trained arms are skipped and finished evaluations (seed_*/<arm>/robust_eval.json)
are reused, so an old run folder only gets the missing evaluations.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import gc
import json
import math
import random
import sys
import time
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
PERICH_DATA_DIR = Path('/data/hossein/mm_project/perich_data_valid_final_raw/')
SESSION = 'C-CO12'
OUT_ROOT = ROOT / 'ACORN_CALIBRATED_EXPERIMENT'

# CEBRA (same as the previous C-CO12 runners).
LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 3000
TEMPERATURE = 0.4
MODEL_ARCH = 'offset36-model-more-dropout'
TIME_OFFSETS = 1
LEARNING_RATE = 3e-4
ADV_STEPS = 10
ALPHA_RATIO = 0.2  # training PGD step = 0.2 x budget (= eps / 5 of the runners)
SEEDS = (42, 43, 44)

# Decoder (same as the previous runners).
MLP_EPOCHS = 2500
MLP_HIDDEN = 64
MLP_DROP = 0.4
MLP_LR = 1e-3
EVAL_BATCH_SIZE = 8192

# Calibration.
WINDOW_LEFT = 18
WINDOW_RIGHT = 18
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
HELD_OUT_FRACTION = 0.2
NOISE_MODEL = 'poisson_gaussian'
NOISE_SMOOTHING = 5
NOISE_FLOOR = 0.1
BASELINE_FLOOR = 0.1

THREATS = ('clean', 'constant', 'noise', 'gain', 'baseline', 'gain_baseline')
ARMS = ('clean', 'constant', 'noise', 'gain', 'baseline', 'gain_baseline')
NOISE_FORK_ARMS = ('clean', 'constant', 'noise')
ATTACK_BATCH_SIZE = 512


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
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


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def load_cebra_fork(path):
    """Import ``cebra`` from ``path``, dropping any previously imported fork."""
    path = path.expanduser().resolve()
    if not (path / 'cebra' / '__init__.py').is_file():
        raise FileNotFoundError(f'CEBRA fork not found: {path}')
    for name in list(sys.modules):
        if name == 'cebra' or name.startswith('cebra.'):
            del sys.modules[name]
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != path]
    sys.path.insert(0, str(path))
    import cebra
    imported = Path(cebra.__file__).resolve()
    if path not in imported.parents:
        raise RuntimeError(f'Wrong import: {imported}; expected under {path}')
    print('Using fork:', imported, flush=True)
    return cebra


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
    for name, value in zip(('X_train', 'X_valid', 'Y_train', 'Y_valid'), (x_train, x_valid, y_train, y_valid)):
        if value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all():
            raise ValueError(f'{name} must be a finite nonempty 2D array; got {value.shape}.')
    if len(x_train) != len(y_train) or len(x_valid) != len(y_valid) or x_train.shape[1] != x_valid.shape[1]:
        raise ValueError('Inconsistent train/valid shapes.')
    return path, tuple(np.ascontiguousarray(a) for a in (x_train, x_valid, y_train, y_valid))


# ----------------------------------------------------------------------
# Threat models (identical to the forks, robust_eval.py and calibrate_budget.py)
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
    width = min(width, windows.size(-1))
    if width % 2 == 0:
        width -= 1
    if width <= 1:
        return windows
    return torch.nn.functional.avg_pool1d(windows, kernel_size=width, stride=1,
                                          padding=width // 2, count_include_pad=False)


def neuron_scale(neural, relative_floor):
    std = neural.float().std(dim=0, unbiased=False)
    active = std[std > 0]
    reference = float(active.median()) if active.numel() > 0 else 1.0
    return std.clamp(min=max(relative_floor * reference, 1e-8))


def threat_stats(x_train, device):
    """Noise model and per-neuron scale on the whole TRAIN split (as the forks do in fit)."""
    x = torch.from_numpy(x_train)
    a, b, floor = fit_noise_model(x, NOISE_MODEL, NOISE_FLOOR)
    return dict(noise_a=a.to(device), noise_b=b.to(device), noise_floor=floor,
                scale=neuron_scale(x, BASELINE_FLOOR).to(device))


def pgd_box(per_sample_loss, apply, bounds, steps, alpha_ratio, restarts):
    """Sign-gradient PGD with ``|p| <= bound`` per variable; ``steps=0`` = one random draw."""
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


def attack_windows(kind, budget, x0, per_sample_loss, stats, steps, alpha_ratio, restarts):
    """Perturbed version of the clean windows ``x0`` (batch, neurons, time) for one threat."""
    if kind == 'clean':
        return x0
    num, neurons, _ = x0.shape
    run = lambda apply, bounds: pgd_box(per_sample_loss, apply, bounds, steps, alpha_ratio,
                                        restarts if steps else 1)
    scale = stats['scale'].view(1, -1, 1).expand(num, neurons, 1).contiguous()
    per_neuron = lambda value: torch.full((num, neurons, 1), value, device=x0.device)
    if kind == 'constant':
        return run(lambda d: x0 + d, [torch.full_like(x0, budget)])
    if kind == 'noise':
        level = local_level(x0, NOISE_SMOOTHING).clamp(min=0)
        sigma = (stats['noise_a'].view(1, -1, 1) * level +
                 stats['noise_b'].view(1, -1, 1)).sqrt().clamp(min=stats['noise_floor'])
        return run(lambda d: x0 + d, [budget * sigma])
    if kind == 'gain':
        return run(lambda g: x0 * (1 + g), [per_neuron(budget)])
    if kind == 'baseline':
        return run(lambda b: x0 + b, [budget * scale])
    if kind == 'gain_baseline':
        gain_eps, baseline_eps = budget
        return run(lambda g, b: x0 * (1 + g) + b, [per_neuron(gain_eps), baseline_eps * scale])
    raise ValueError(kind)


def padded_series(neural, left, right, device):
    """Edge padding exactly like cebra.CEBRA.transform."""
    return torch.from_numpy(np.pad(neural, ((left, right - 1), (0, 0)), mode='edge')).to(device)


def window_batches(padded, num, window, batch_size):
    offsets = torch.arange(window, device=padded.device)
    for start in range(0, num, batch_size):
        index = torch.arange(start, min(start + batch_size, num), device=padded.device)
        yield index, padded[index[:, None] + offsets].transpose(1, 2).contiguous()


def r2_per_output(y_true, y_pred):
    ss_res = ((y_true - y_pred)**2).sum(dim=0)
    ss_tot = ((y_true - y_true.mean(dim=0))**2).sum(dim=0)
    return 1 - ss_res / ss_tot.clamp(min=1e-12)


def format_budget(budget):
    if budget is None:
        return '-'
    if isinstance(budget, (tuple, list)):
        return f'{budget[0]:.4g}/{budget[1]:.4g}'
    return f'{budget:.4g}'


# ----------------------------------------------------------------------
# Stage 1: calibration
# ----------------------------------------------------------------------
class RidgeDecoder:
    def __init__(self, mean, std, weight, bias):
        self.mean, self.std, self.weight, self.bias = mean, std, weight, bias

    def __call__(self, windows):
        z = ((windows - self.mean.view(1, -1, 1)) / self.std.view(1, -1, 1)).flatten(1)
        return z @ self.weight + self.bias


def fit_ridge(x_fit, y_fit, x_held, y_held, device):
    window = WINDOW_LEFT + WINDOW_RIGHT
    mean = torch.from_numpy(x_fit.mean(axis=0)).to(device)
    std = torch.from_numpy(x_fit.std(axis=0)).clamp(min=1e-6).to(device)
    padded = padded_series(x_fit, WINDOW_LEFT, WINDOW_RIGHT, device)
    y = torch.from_numpy(y_fit).double().to(device)
    dim = x_fit.shape[1] * window
    gram = torch.zeros(dim, dim, dtype=torch.float64, device=device)
    cross = torch.zeros(dim, y.shape[1], dtype=torch.float64, device=device)
    z_sum = torch.zeros(dim, dtype=torch.float64, device=device)
    for index, windows in window_batches(padded, len(x_fit), window, 1024):
        z = ((windows - mean.view(1, -1, 1)) / std.view(1, -1, 1)).flatten(1).double()
        gram += z.T @ z
        cross += z.T @ y[index]
        z_sum += z.sum(dim=0)
    n = len(x_fit)
    z_mean, y_mean = z_sum / n, y.mean(dim=0)
    gram -= n * torch.outer(z_mean, z_mean)
    cross -= n * torch.outer(z_mean, y_mean)
    scale = float(torch.diagonal(gram).mean())
    held = padded_series(x_held, WINDOW_LEFT, WINDOW_RIGHT, device)
    y_held_t = torch.from_numpy(y_held).to(device)
    best = None
    for penalty in RIDGE_GRID:
        weight = torch.linalg.solve(gram + penalty * scale * torch.eye(dim, dtype=gram.dtype, device=device), cross)
        decoder = RidgeDecoder(mean, std, weight.float(), (y_mean - z_mean @ weight).float())
        with torch.no_grad():
            pred = torch.cat([decoder(w) for _, w in window_batches(held, len(y_held), window, 1024)])
        r2 = float(r2_per_output(y_held_t, pred).mean())
        print(f'  ridge penalty {penalty:g}: held-out R2={r2:.4f}', flush=True)
        if best is None or r2 > best[0]:
            best = (r2, penalty, decoder)
    return best


def bisect_budget(damage, start, tau, iters, upper=None, max_doublings=16):
    """Largest budget with ``damage(budget) <= tau`` (damage assumed nondecreasing)."""
    lo = hi = None
    if damage(start) <= tau:
        lo = start
        for _ in range(max_doublings):
            if upper is not None and lo >= upper:
                return upper, 'reached upper bound'
            candidate = lo * 2 if upper is None else min(lo * 2, upper)
            if damage(candidate) > tau:
                hi = candidate
                break
            lo = candidate
        else:
            return lo, f'damage stayed <= tau up to {lo:g}; budget capped'
    else:
        hi = start
        for _ in range(max_doublings):
            candidate = hi / 2
            if damage(candidate) <= tau:
                lo = candidate
                break
            hi = candidate
        else:
            return 0.0, f'damage > tau even at {hi:g}'
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if damage(mid) <= tau:
            lo = mid
        else:
            hi = mid
    return lo, 'ok'


def calibrate(x_train, y_train, stats, args, device):
    """Calibrated budget per threat model; see the module docstring."""
    split = int(round(len(x_train) * (1 - HELD_OUT_FRACTION)))
    x_fit, y_fit, x_held, y_held = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
    print(f'Reference ridge decoder: fit {x_fit.shape}, held-out {x_held.shape} (VALID not used)', flush=True)
    r2_clean, penalty, ridge = fit_ridge(x_fit, y_fit, x_held, y_held, device)
    print(f'Reference ridge: penalty={penalty:g}, held-out clean R2={r2_clean:.4f}', flush=True)
    if r2_clean < 0.3:
        print('WARNING: reference decoder is weak (R2 < 0.3); budgets are not meaningful.', flush=True)
    window = WINDOW_LEFT + WINDOW_RIGHT
    held = padded_series(x_held, WINDOW_LEFT, WINDOW_RIGHT, device)
    y = torch.from_numpy(y_held).to(device)
    y_std = y.std(dim=0).clamp(min=1e-8)
    steps = args.calibration_steps if args.oracle == 'adversarial' else 0

    def damage_and_size(kind, budget):
        torch.manual_seed(args.attack_seed)
        preds, rel = [], []
        for index, x0 in window_batches(held, len(y), window, ATTACK_BATCH_SIZE):
            target = y[index]
            loss = lambda x: (((ridge(x) - target) / y_std)**2).mean(dim=1)
            x_adv = attack_windows(kind, budget, x0, loss, stats, steps, 2.5 / args.calibration_steps, 1)
            with torch.no_grad():
                preds.append(ridge(x_adv))
                rel.append(((x_adv - x0).abs() / stats['scale'].view(1, -1, 1)).mean(dim=(1, 2)))
        r2 = float(r2_per_output(y, torch.cat(preds)).mean())
        return r2_clean - r2, float(torch.cat(rel).mean())

    def cached_damage(trace, kind, to_budget):
        """damage(value) with memoisation in ``trace`` (value -> (damage, size))."""
        def damage(value):
            key = round(value, 12)
            if key not in trace:
                trace[key] = damage_and_size(kind, to_budget(value))
            return trace[key][0]
        return damage

    budgets, details = {}, {}
    starts = dict(constant=0.5 * float(stats['scale'].median()), noise=1.0, gain=0.1, baseline=0.5)
    for kind in ('constant', 'noise', 'gain', 'baseline'):
        trace = {}
        damage = cached_damage(trace, kind, lambda b: b)
        budget, status = bisect_budget(damage, starts[kind], args.tau, 8)
        budgets[kind] = budget
        details[kind] = dict(budget=budget, status=status, trace=[
            dict(budget=b, damage=d, mean_abs_delta_over_std=r) for b, (d, r) in sorted(trace.items())])
        print(f'  {kind:<14} budget={budget:.4g}  [{status}]  ({len(trace)} evaluations)', flush=True)
    trace = {}
    pair = lambda s: (s * budgets['gain'], s * budgets['baseline'])
    damage = cached_damage(trace, 'gain_baseline', pair)
    scale, status = bisect_budget(damage, 1.0, args.tau, 8, upper=1.0)
    budgets['gain_baseline'] = pair(scale)
    details['gain_baseline'] = dict(budget=pair(scale), scale=scale, status=status, trace=[
        dict(scale=s, damage=d, mean_abs_delta_over_std=r) for s, (d, r) in sorted(trace.items())])
    print(f'  {"gain_baseline":<14} budget={format_budget(budgets["gain_baseline"])} (scale {scale:.3f})  [{status}]',
          flush=True)
    return budgets, dict(oracle=args.oracle, tau=args.tau, ridge_penalty=penalty,
                         ridge_held_out_r2=r2_clean, fit_samples=len(x_fit),
                         held_out_samples=len(x_held), details=details)


# ----------------------------------------------------------------------
# Stage 2: training
# ----------------------------------------------------------------------
class TwoLayerMLP(nn.Module):
    """Same layout as the runners' decoder."""

    def __init__(self, dim, out, hidden=MLP_HIDDEN, dropout=MLP_DROP):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


def train_decoder(z_train, y_train, seed, device, epochs):
    seed_all(seed)
    decoder = TwoLayerMLP(z_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=MLP_LR)
    z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    decoder.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(decoder(z), y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite decoder loss at epoch {epoch + 1}.')
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 500 == 0 or epoch + 1 == epochs:
            print(f'  decoder {epoch + 1}/{epochs}: train MSE={float(loss):.4f}', flush=True)
    return decoder.eval()


def predict(decoder, embeddings):
    device = next(decoder.parameters()).device
    with torch.inference_mode():
        return np.concatenate([decoder(torch.as_tensor(embeddings[s:s + EVAL_BATCH_SIZE], device=device)).cpu().numpy()
                               for s in range(0, len(embeddings), EVAL_BATCH_SIZE)])


def arm_config(arm, budgets, args):
    """CEBRA constructor arguments of one arm (fork-specific adversarial arguments)."""
    config = dict(batch_size=BATCH_SIZE, temperature=TEMPERATURE, model_architecture=MODEL_ARCH,
                  time_offsets=TIME_OFFSETS, max_iterations=args.max_iter, output_dimension=LATENT_DIM,
                  num_hidden_units=HIDDEN, learning_rate=LEARNING_RATE, temperature_mode='constant',
                  device=args.device, verbose=True,
                  training_mode='standard' if arm == 'clean' else 'adversarial')
    if arm == 'clean':
        return config
    config.update(adv_steps=args.adv_steps, attack_norm='linf')
    if arm == 'constant':
        config.update(adv_epsilon=budgets['constant'], adv_alpha=ALPHA_RATIO * budgets['constant'],
                      adv_epsilon_mode='constant')
    elif arm == 'noise':
        config.update(adv_epsilon_mode='noise', adv_epsilon_coef=budgets['noise'],
                      adv_noise_model=NOISE_MODEL, adv_noise_smoothing=NOISE_SMOOTHING,
                      adv_noise_floor=NOISE_FLOOR, adv_alpha_ratio=ALPHA_RATIO)
    else:
        gain, baseline = dict(gain=(budgets['gain'], 0.0), baseline=(0.0, budgets['baseline']),
                              gain_baseline=budgets['gain_baseline'])[arm]
        config.update(adv_attack_type='structured', adv_gain_epsilon=gain, adv_baseline_epsilon=baseline,
                      adv_baseline_scale='std', adv_baseline_floor=BASELINE_FLOOR,
                      adv_structured_shared=False, adv_alpha_ratio=ALPHA_RATIO)
    return config


def train_arm(cebra, arm, seed, config, arrays, folder, args):
    x_train, x_valid, y_train, y_valid = arrays
    folder.mkdir(parents=True, exist_ok=True)
    print('\n' + '=' * 90 + f'\nTRAIN | seed {seed} | arm {arm}\n' + json.dumps(config, indent=2), flush=True)
    started = time.perf_counter()
    seed_all(seed)
    encoder = cebra.CEBRA(**config)
    encoder.fit(x_train, y_train)
    z_train = np.asarray(encoder.transform(x_train), dtype=np.float32)
    z_valid = np.asarray(encoder.transform(x_valid), dtype=np.float32)
    for name, z, x in (('train', z_train, x_train), ('valid', z_valid, x_valid)):
        if z.shape != (len(x), LATENT_DIM) or not np.isfinite(z).all():
            raise ValueError(f'{name}: invalid embedding {z.shape}')
    decoder = train_decoder(z_train, y_train, seed + 100_000, torch.device(encoder.device_), args.decoder_epochs)
    valid_pred = predict(decoder, z_valid)
    valid_r2 = r2_per_output(torch.from_numpy(y_valid), torch.from_numpy(valid_pred)).tolist()
    train_r2 = r2_per_output(torch.from_numpy(y_train), torch.from_numpy(predict(decoder, z_train))).tolist()
    encoder.save(str(folder / 'cebra.pt'), backend='sklearn')
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in decoder.state_dict().items()},
                    input_dim=LATENT_DIM, output_dim=int(y_train.shape[1]), hidden=MLP_HIDDEN,
                    dropout=MLP_DROP, seed=seed + 100_000), folder / 'decoder.pt')
    metrics = dict(seed=seed, arm=arm, train_mean_r2=float(np.mean(train_r2)), valid_mean_r2=float(np.mean(valid_r2)),
                   valid_r2_per_output=valid_r2, seconds=time.perf_counter() - started, encoder_config=config)
    save_json(folder / 'metrics.json', metrics)  # written last: marks the arm as finished
    print(f'{arm} seed {seed}: TRAIN R2={metrics["train_mean_r2"]:.4f} | VALID R2={metrics["valid_mean_r2"]:.4f}',
          flush=True)
    del encoder, decoder
    cleanup()


# ----------------------------------------------------------------------
# Stage 3: robust evaluation
# ----------------------------------------------------------------------
def load_encoder(cebra, path, device):
    """Rebuild the encoder from its weights (fork-independent, unlike cebra.CEBRA.load)."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)  # own trusted files
    args, state = checkpoint['args'], checkpoint['state']
    model = cebra.models.init(args['model_architecture'], num_neurons=state['n_features_in_'],
                              num_units=args['num_hidden_units'], num_output=args['output_dimension'])
    model.load_state_dict(checkpoint['state_dict']['model'])
    return model.to(device).eval()


def load_decoder(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    decoder = TwoLayerMLP(checkpoint['input_dim'], checkpoint['output_dim'], checkpoint['hidden'], checkpoint['dropout'])
    decoder.load_state_dict(checkpoint['state_dict'])
    return decoder.to(device).eval()


def eval_specs(budgets, args):
    """Every evaluation of one model: clean, then each threat x scale x mode.

    ``pgd``: worst case found by PGD on encoder + decoder. ``random``: one uniform
    random draw inside the same budget (no optimisation), the counterpart of the
    random calibration oracle.
    """
    specs = [dict(key='clean', mode='none', threat='clean', scale=0.0, budget=None)]
    modes = ('pgd', 'random') if args.eval_random else ('pgd',)
    for mode in modes:
        for scale in args.eval_scales:
            for kind in THREATS[1:]:
                budget = budgets[kind]
                budget = tuple(scale * b for b in budget) if isinstance(budget, tuple) else scale * budget
                specs.append(dict(key=f'{mode}:{kind}@{scale:g}', mode=mode, threat=kind, scale=scale,
                                  budget=budget))
    return specs


def evaluate_arm(cebra, folder, x_valid, y_valid, y_train_std, specs, stats, budgets, args, device):
    """Results of every spec for one trained model, cached in <folder>/robust_eval.json (resume)."""
    cache_path = folder / 'robust_eval.json'
    settings = dict(eval_steps=args.eval_steps, eval_restarts=args.eval_restarts,
                    attack_seed=args.attack_seed, budgets={k: format_budget(v) for k, v in budgets.items()})
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    results = cache.get('results', {}) if cache.get('settings') == settings else {}
    todo = [s for s in specs if s['key'] not in results]
    if not todo:
        return results
    encoder = load_encoder(cebra, folder / 'cebra.pt', device)
    decoder = load_decoder(folder / 'decoder.pt', device)
    for p in list(encoder.parameters()) + list(decoder.parameters()):
        p.requires_grad_(False)
    offset = encoder.get_offset()
    window = offset.left + offset.right
    padded = padded_series(x_valid, offset.left, offset.right, device)
    y = torch.from_numpy(y_valid).to(device)
    encode = lambda x: encoder(x).squeeze(-1)
    for spec in todo:
        steps, restarts = (args.eval_steps, args.eval_restarts) if spec['mode'] == 'pgd' else (0, 1)
        torch.manual_seed(args.attack_seed)
        preds, rel = [], []
        for index, x0 in window_batches(padded, len(y), window, ATTACK_BATCH_SIZE):
            target = y[index]
            loss = lambda x: (((decoder(encode(x)) - target) / y_train_std)**2).mean(dim=1)
            x_adv = attack_windows(spec['threat'], spec['budget'], x0, loss, stats, steps,
                                   2.5 / args.eval_steps, restarts)
            with torch.no_grad():
                preds.append(decoder(encode(x_adv)))
                rel.append(((x_adv - x0).abs() / stats['scale'].view(1, -1, 1)).mean(dim=(1, 2)))
        per_output = r2_per_output(y, torch.cat(preds)).tolist()
        results[spec['key']] = dict(valid_mean_r2=float(np.mean(per_output)), valid_r2_per_output=per_output,
                                    mean_abs_delta_over_std=float(torch.cat(rel).mean()))
        save_json(cache_path, dict(settings=settings, results=results))
    del encoder, decoder, padded
    cleanup()
    return results


def write_summary(out, rows, arms, budgets, calibration, train_r2, data_path, args):
    lines = [f'ACORN calibrated experiment | {args.session} | data {data_path}',
             f'oracle={calibration["oracle"]} tau={calibration["tau"]:g} | reference ridge held-out R2='
             f'{calibration["ridge_held_out_r2"]:.4f} | seeds={args.seeds} | max_iter={args.max_iter} | '
             f'eval on VALID: PGD {args.eval_steps} steps x {args.eval_restarts} restarts; '
             f'scales={args.eval_scales}; random={args.eval_random}',
             '', 'CALIBRATED BUDGETS (1.0 x; used for training AND evaluation)']
    for kind in ('constant', 'noise', 'gain', 'baseline', 'gain_baseline'):
        lines.append(f'  {kind:<14} {format_budget(budgets[kind]):<20} [{calibration["details"][kind]["status"]}]')
    modes = ('pgd', 'random') if args.eval_random else ('pgd',)
    scales = sorted(args.eval_scales, reverse=True)
    titles = dict(pgd='PGD (worst case)', random='RANDOM (one uniform draw, no optimisation)')
    width, names = 18, list(THREATS)
    summary = {}
    for mode in modes:
        for scale in scales:
            table = f'{mode}@{scale:g}'
            keys = {k: 'clean' if k == 'clean' else f'{mode}:{k}@{scale:g}' for k in names}
            sizes = {k: np.mean([r['mean_abs_delta_over_std'] for r in rows if r['key'] == keys[k]])
                     for k in names[1:]}
            lines += ['', f'{titles[mode]} at {scale:g} x calibrated budget: VALID mean R2 (mean +- std over seeds)',
                      '  mean |delta| / neuron std: ' + ', '.join(f'{k}={v:.3f}' for k, v in sizes.items()),
                      f'{"arm":<16}' + ''.join(f'{n:>{width}}' for n in names)]
            summary[table] = {}
            for arm in arms:
                line, summary[table][arm] = f'{arm:<16}', {}
                for name in names:
                    scores = np.array([r['valid_mean_r2'] for r in rows if r['arm'] == arm and r['key'] == keys[name]])
                    if len(scores) == 0:
                        line += f'{"-":>{width}}'
                        continue
                    std = float(scores.std(ddof=1)) if len(scores) > 1 else None
                    summary[table][arm][name] = dict(mean=float(scores.mean()), sample_std=std, n_seeds=len(scores))
                    cell = f'{scores.mean():.4f}' + (f' +-{std:.3f}' if std is not None else '')
                    line += f'{cell:>{width}}'
                lines.append(line)
    lines += ['', 'Sanity: clean R2 recomputed vs. training metrics (max abs diff): '
              f'{max((abs(r["valid_mean_r2"] - train_r2[(r["arm"], r["seed"])]) for r in rows if r["key"] == "clean"), default=float("nan")):.2e}']
    text = '\n'.join(lines)
    (out / 'summary.txt').write_text(text + '\n', encoding='utf-8')
    save_json(out / 'summary.json', dict(budgets={k: v for k, v in budgets.items()}, by_table=summary))
    print('\n' + text, flush=True)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session', default=SESSION)
    parser.add_argument('--data-dir', type=Path, default=PERICH_DATA_DIR)
    parser.add_argument('--noise-fork', type=Path, default=ROOT / 'Acorn-noise-eps')
    parser.add_argument('--structured-fork', type=Path, default=ROOT / 'Acorn-structured-attack')
    parser.add_argument('--out', type=Path, default=None,
                        help='Run folder. Default: a new folder in ACORN_CALIBRATED_EXPERIMENT/. '
                             'Give an existing one to resume.')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--oracle', choices=('random', 'adversarial'), default='random')
    parser.add_argument('--tau', type=float, default=0.05, help='Allowed drop of the reference ridge R2.')
    parser.add_argument('--max-iter', type=int, default=MAX_ITER)
    parser.add_argument('--decoder-epochs', type=int, default=MLP_EPOCHS)
    parser.add_argument('--adv-steps', type=int, default=ADV_STEPS)
    parser.add_argument('--calibration-steps', type=int, default=20)
    parser.add_argument('--eval-steps', type=int, default=50)
    parser.add_argument('--eval-restarts', type=int, default=2)
    parser.add_argument('--eval-scales', nargs='+', type=float, default=[0.25, 0.5, 1.0],
                        help='Evaluate every threat at these multiples of its calibrated budget.')
    parser.add_argument('--no-eval-random', dest='eval_random', action='store_false',
                        help='Skip the random-perturbation evaluation (on by default).')
    parser.add_argument('--attack-seed', type=int, default=0)
    parser.add_argument('--device', default='cuda_if_available')
    args = parser.parse_args()
    if not 0 < args.tau < 1:
        parser.error('--tau must be in (0, 1).')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('Seeds must be unique and nonnegative.')
    if min(args.max_iter, args.decoder_epochs, args.adv_steps, args.calibration_steps,
           args.eval_steps, args.eval_restarts) < 1:
        parser.error('Iterations, epochs, steps and restarts must be positive.')
    if 1.0 not in args.eval_scales:
        args.eval_scales.append(1.0)  # the main table is at the calibrated budget
    if min(args.eval_scales) <= 0 or len(set(args.eval_scales)) != len(args.eval_scales):
        parser.error('--eval-scales must be unique and positive.')
    return args


def main():
    args = parse_args()
    if args.device == 'cuda_if_available':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    forks = {'noise': args.noise_fork, 'structured': args.structured_fork}
    needed = {('noise' if arm in NOISE_FORK_ARMS else 'structured') for arm in args.arms}
    for name in needed:
        if not (forks[name].expanduser() / 'cebra' / '__init__.py').is_file():
            raise FileNotFoundError(f'{name} fork not found: {forks[name]} (use --{name}-fork).')

    data_path, arrays = load_data(args.data_dir, args.session)
    x_train, x_valid, y_train, y_valid = arrays
    print('DATA:', data_path, '| shapes:', [a.shape for a in arrays], flush=True)
    stats = threat_stats(x_train, device)

    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        args.out = OUT_ROOT / f'{args.session}_{args.oracle}_tau{args.tau:g}_{stamp}'
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    print('OUTPUT:', out, flush=True)

    # Stage 1 (reused when resuming, so every arm of a run shares the same budgets).
    calibration_path = out / 'calibration.json'
    if calibration_path.is_file():
        saved = json.loads(calibration_path.read_text())
        budgets = {k: tuple(v) if isinstance(v, list) else v for k, v in saved['budgets'].items()}
        calibration = saved['calibration']
        print('\nSTAGE 1: reusing calibration from', calibration_path, flush=True)
    else:
        print(f'\nSTAGE 1: CALIBRATE (oracle={args.oracle}, tau={args.tau:g})', flush=True)
        budgets, calibration = calibrate(x_train, y_train, stats, args, device)
        save_json(calibration_path, dict(budgets=budgets, calibration=calibration, session=args.session,
                                         data=str(data_path)))
    print('Budgets:', {k: format_budget(v) for k, v in budgets.items()}, flush=True)
    save_json(out / 'run_config.json', dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        session=args.session, data_dir=str(args.data_dir), budgets=budgets,
        configs={arm: arm_config(arm, budgets, args) for arm in args.arms}))

    # Stage 2: one fork at a time.
    print('\nSTAGE 2: TRAIN', flush=True)
    for fork_name in ('noise', 'structured'):
        arms = [a for a in args.arms if (a in NOISE_FORK_ARMS) == (fork_name == 'noise')]
        todo = [(arm, seed) for seed in args.seeds for arm in arms
                if not (out / f'seed_{seed}' / arm / 'metrics.json').is_file()]
        for arm in arms:
            done = [s for s in args.seeds if (out / f'seed_{s}' / arm / 'metrics.json').is_file()]
            if done:
                print(f'  {arm}: already trained for seeds {done} (skipped)', flush=True)
        if not todo:
            continue
        cebra = load_cebra_fork(forks[fork_name])
        for arm, seed in todo:
            config = arm_config(arm, budgets, args)
            cebra.CEBRA(**config)  # fail early on a constructor mismatch
            train_arm(cebra, arm, seed, config, arrays, out / f'seed_{seed}' / arm, args)

    # Stage 3 (per-model results are cached, so an interrupted evaluation resumes).
    specs = eval_specs(budgets, args)
    print(f'\nSTAGE 3: ROBUST EVALUATION on VALID: {len(specs)} evaluations per model '
          f'(PGD {args.eval_steps} steps x {args.eval_restarts} restarts; scales {args.eval_scales}; '
          f'random={args.eval_random})', flush=True)
    cebra = load_cebra_fork(forks['noise' if 'noise' in needed else 'structured'])  # model registry only
    y_train_std = torch.from_numpy(y_train.std(axis=0)).clamp(min=1e-8).to(device)
    rows, train_r2 = [], {}
    main_keys = ['clean'] + [f'pgd:{k}@1' for k in THREATS[1:]]
    for seed in args.seeds:
        for arm in args.arms:
            folder = out / f'seed_{seed}' / arm
            metrics = json.loads((folder / 'metrics.json').read_text())
            train_r2[(arm, seed)] = metrics['valid_mean_r2']
            results = evaluate_arm(cebra, folder, x_valid, y_valid, y_train_std, specs, stats, budgets, args, device)
            print(f'  seed {seed} {arm:<14} ' + '  '.join(f'{k}={results[k]["valid_mean_r2"]:.4f}'
                                                         for k in main_keys if k in results), flush=True)
            for spec in specs:
                rows.append(dict(seed=seed, arm=arm, key=spec['key'], mode=spec['mode'], threat=spec['threat'],
                                 scale=spec['scale'], budget=format_budget(spec['budget']), **results[spec['key']]))
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['seed', 'arm', 'attack', 'threat', 'scale', 'budget', 'valid_mean_r2',
                         'mean_abs_delta_over_std'] + [f'valid_r2_output_{i}' for i in range(y_valid.shape[1])])
        for r in rows:
            writer.writerow([r['seed'], r['arm'], r['mode'], r['threat'], r['scale'], r['budget'], r['valid_mean_r2'],
                             r['mean_abs_delta_over_std']] + r['valid_r2_per_output'])
    write_summary(out, rows, args.arms, budgets, calibration, train_r2, data_path, args)
    print('\nDone. Send this file:', out / 'summary.txt', flush=True)


if __name__ == '__main__':
    main()

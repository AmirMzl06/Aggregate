"""Data-driven attack budgets: calibrate every threat model against a reference decoder.

Idea 3 of ACORN. In vision the budget is "what a human cannot see"; here the oracle
is a REFERENCE DECODER that is independent of CEBRA: ridge regression from the raw
input window (the same window the CEBRA encoder sees) to the behavior labels. For
every threat model the script finds, by bisection, the LARGEST budget whose attack
lowers the reference decoder's R2 by at most --tau (absolute):

    budget* = max { b : R2_ref(clean) - R2_ref(attack_b) <= tau }

so all threat models are calibrated to the same semantic damage and can be
compared fairly.

Oracles:
    adversarial  PGD against the reference decoder (strict; default)
    random       one uniform random draw inside the budget (lenient)
    both         calibrate with both and report both

Threat models (identical definitions to robust_eval.py and the forks):
    constant       |delta| <= eps
    noise          |delta| <= coef * sigma_j(local level)   (Acorn-noise-eps)
    gain           x * (1 + g),  |g| <= gain_eps            (Acorn-structured-attack)
    baseline       x + b * s_j,  |b| <= baseline_eps
    gain_baseline  calibrated as a scale s in (0, 1] on (gain*, baseline*)

No leakage: the reference decoder is fitted on the first part of the TRAIN split
and the budgets are calibrated on the held-out last part of the TRAIN split. The
VALID split is never used. Only numpy + torch are needed (no CEBRA).
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import math
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PERICH_DATA_DIR = Path('/data/hossein/mm_project/perich_data_valid_final_raw/')
SESSION = 'C-CO12'
OUT_ROOT = ROOT / 'CALIBRATION'
THREATS = ('constant', 'noise', 'gain', 'baseline', 'gain_baseline')
ORACLES = ('adversarial', 'random')
NOISE_MODELS = ('poisson_gaussian', 'poisson', 'gaussian')
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
BATCH_SIZE = 1024


# ----------------------------------------------------------------------
# Noise model / scales (identical to robust_eval.py and the forks)
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
# Attacks (identical to robust_eval.py)
# ----------------------------------------------------------------------
def pgd_box(per_sample_loss, apply, bounds, steps, alpha_ratio, restarts):
    """Sign-gradient PGD over variables constrained to ``|p| <= bound`` (element-wise).

    With ``steps=0`` this is one uniform random draw inside the box.
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


def attack_windows(kind, budget, x0, per_sample_loss, stats, steps, args):
    """Perturbed version of the clean windows ``x0`` for one threat model."""
    num, neurons, _ = x0.shape
    run = lambda apply, bounds: pgd_box(per_sample_loss, apply, bounds, steps,
                                        args.alpha_ratio, args.restarts if steps else 1)
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


# ----------------------------------------------------------------------
# Reference decoder: ridge regression on the raw input window
# ----------------------------------------------------------------------
class RidgeDecoder:
    """``y = flatten((x - mean_j) / std_j) @ W + bias`` on ``(batch, neurons, window)``."""

    def __init__(self, mean, std, weight, bias):
        self.mean, self.std, self.weight, self.bias = mean, std, weight, bias

    def __call__(self, windows):
        z = ((windows - self.mean.view(1, -1, 1)) / self.std.view(1, -1, 1)).flatten(1)
        return z @ self.weight + self.bias


def make_windows(neural, left, right):
    """Edge-padded windows exactly like cebra.CEBRA.transform: ``(time, neurons, window)`` generator input."""
    padded = np.pad(neural, ((left, right - 1), (0, 0)), mode='edge')
    return torch.from_numpy(padded)


def batches(padded, num, window, device, batch_size):
    offsets = torch.arange(window, device=device)
    for start in range(0, num, batch_size):
        index = torch.arange(start, min(start + batch_size, num), device=device)
        yield index, padded[index[:, None] + offsets].transpose(1, 2).contiguous()


def fit_ridge(x_fit, y_fit, x_held, y_held, left, right, device, batch_size):
    """Fit ridge on the fit part, pick the penalty on the held-out part."""
    window = left + right
    mean = torch.from_numpy(x_fit.mean(axis=0)).to(device)
    std = torch.from_numpy(x_fit.std(axis=0)).clamp(min=1e-6).to(device)
    padded = make_windows(x_fit, left, right).to(device)
    y = torch.from_numpy(y_fit).double().to(device)
    dim = x_fit.shape[1] * window
    gram = torch.zeros(dim, dim, dtype=torch.float64, device=device)
    cross = torch.zeros(dim, y.shape[1], dtype=torch.float64, device=device)
    z_sum = torch.zeros(dim, dtype=torch.float64, device=device)
    for index, windows in batches(padded, len(x_fit), window, device, batch_size):
        z = ((windows - mean.view(1, -1, 1)) / std.view(1, -1, 1)).flatten(1).double()
        gram += z.T @ z
        cross += z.T @ y[index]
        z_sum += z.sum(dim=0)
    n = len(x_fit)
    z_mean, y_mean = z_sum / n, y.mean(dim=0)
    gram -= n * torch.outer(z_mean, z_mean)
    cross -= n * torch.outer(z_mean, y_mean)
    scale = float(torch.diagonal(gram).mean())

    held = make_windows(x_held, left, right).to(device)
    y_held_t = torch.from_numpy(y_held).to(device)
    best = None
    for penalty in RIDGE_GRID:
        weight = torch.linalg.solve(gram + penalty * scale * torch.eye(dim, dtype=gram.dtype, device=device), cross)
        bias = y_mean - z_mean @ weight
        decoder = RidgeDecoder(mean, std, weight.float(), bias.float())
        r2 = mean_r2(decoder, held, y_held_t, window, device, batch_size)
        print(f'  ridge penalty {penalty:g}: held-out R2={r2:.4f}', flush=True)
        if best is None or r2 > best[0]:
            best = (r2, penalty, decoder)
    return best


def r2_per_output(y_true, y_pred):
    ss_res = ((y_true - y_pred)**2).sum(dim=0)
    ss_tot = ((y_true - y_true.mean(dim=0))**2).sum(dim=0)
    return 1 - ss_res / ss_tot


def mean_r2(decoder, padded, y, window, device, batch_size):
    with torch.no_grad():
        pred = torch.cat([decoder(w) for _, w in batches(padded, len(y), window, device, batch_size)])
    return float(r2_per_output(y, pred).mean())


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------
def attacked_r2(decoder, padded, y, window, kind, budget, stats, steps, args, device):
    """Reference-decoder R2 with every window perturbed; also the mean |delta| / std."""
    y_std = y.std(dim=0).clamp(min=1e-8)
    preds, rel_delta = [], []
    for index, x0 in batches(padded, len(y), window, device, args.batch_size):
        target = y[index]
        per_sample_loss = lambda x: (((decoder(x) - target) / y_std)**2).mean(dim=1)
        x_adv = attack_windows(kind, budget, x0, per_sample_loss, stats, steps, args)
        with torch.no_grad():
            preds.append(decoder(x_adv))
            rel_delta.append(((x_adv - x0).abs() / stats['scale'].view(1, -1, 1)).mean(dim=(1, 2)))
    return float(r2_per_output(y, torch.cat(preds)).mean()), float(torch.cat(rel_delta).mean())


def bisect_budget(damage, start, tau, iters, upper=None, max_doublings=16):
    """Largest budget with ``damage(budget) <= tau`` (damage assumed nondecreasing).

    Brackets by doubling/halving from ``start``, then bisects geometrically.
    Returns ``(budget, status)``.
    """
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


def calibrate_threat(kind, oracle, decoder, padded, y, window, stats, args, device, known):
    """Calibrate one threat model; returns a result dict with the full search trace."""
    steps = args.steps if oracle == 'adversarial' else 0
    r2_clean = known['r2_clean']
    trace = {}

    def evaluate(budget):
        key = round(budget, 12) if not isinstance(budget, tuple) else tuple(round(b, 12) for b in budget)
        if key not in trace:
            torch.manual_seed(args.attack_seed)  # same random start for every budget
            r2, rel = attacked_r2(decoder, padded, y, window, kind, budget, stats, steps, args, device)
            trace[key] = dict(budget=budget, r2=r2, damage=r2_clean - r2, mean_abs_delta_over_std=rel)
            print(f'    {kind:<14} budget={budget!s:<28} R2={r2:.4f}  dR2={r2_clean - r2:.4f}'
                  f'  mean|delta|/std={rel:.3f}', flush=True)
        return trace[key]

    if kind == 'gain_baseline':
        gain, baseline = known['gain'], known['baseline']
        scale, status = bisect_budget(lambda s: evaluate((s * gain, s * baseline))['damage'],
                                      1.0, args.tau, args.iters, upper=1.0)
        budget = (scale * gain, scale * baseline)
        result = dict(scale=scale, gain_epsilon=budget[0], baseline_epsilon=budget[1])
    else:
        start = dict(constant=0.5 * known['median_std'], noise=1.0, gain=0.1, baseline=0.5)[kind]
        budget, status = bisect_budget(lambda b: evaluate(b)['damage'], start, args.tau, args.iters)
        result = dict(budget=budget)
    final = evaluate(budget) if (budget if not isinstance(budget, tuple) else budget[0]) > 0 else None
    result.update(status=status, oracle=oracle,
                  damage_at_budget=final['damage'] if final else None,
                  mean_abs_delta_over_std=final['mean_abs_delta_over_std'] if final else None,
                  trace=sorted(trace.values(), key=lambda t: t['damage']))
    return result


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def load_train(data_dir, session):
    path = data_dir.expanduser() / f'{session}.npz'
    if not path.is_file():
        raise FileNotFoundError(f'Dataset not found: {path}')
    with np.load(path, allow_pickle=False) as data:
        x = np.ascontiguousarray(np.asarray(data['train_data'], dtype=np.float32))
        y = np.asarray(data['train_label'], dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or len(x) != len(y) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(f'Invalid train arrays: {x.shape}, {y.shape}')
    return path, x, np.ascontiguousarray(y)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session', default=SESSION)
    parser.add_argument('--data-dir', type=Path, default=PERICH_DATA_DIR)
    parser.add_argument('--out-root', type=Path, default=OUT_ROOT)
    parser.add_argument('--threats', nargs='+', choices=THREATS, default=list(THREATS))
    parser.add_argument('--oracle', choices=ORACLES + ('both',), default='adversarial')
    parser.add_argument('--tau', type=float, default=0.05,
                        help='Allowed drop of the reference decoder R2 (absolute).')
    parser.add_argument('--held-out-fraction', type=float, default=0.2,
                        help='Last part of TRAIN used to pick the ridge penalty and to calibrate.')
    parser.add_argument('--window-left', type=int, default=18, help='Encoder offset (offset36: 18).')
    parser.add_argument('--window-right', type=int, default=18, help='Encoder offset (offset36: 18).')
    parser.add_argument('--noise-model', choices=NOISE_MODELS, default='poisson_gaussian')
    parser.add_argument('--noise-smoothing', type=int, default=5)
    parser.add_argument('--noise-floor', type=float, default=0.1)
    parser.add_argument('--baseline-floor', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--alpha-ratio', type=float, default=None, help='Step / budget. Default: 2.5 / steps.')
    parser.add_argument('--restarts', type=int, default=1)
    parser.add_argument('--iters', type=int, default=8, help='Bisection iterations after bracketing.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--attack-seed', type=int, default=0)
    parser.add_argument('--device', default='cuda_if_available')
    args = parser.parse_args()
    if args.alpha_ratio is None:
        args.alpha_ratio = 2.5 / args.steps
    if not 0 < args.tau < 1:
        parser.error('--tau must be in (0, 1).')
    if not 0 < args.held_out_fraction < 1:
        parser.error('--held-out-fraction must be in (0, 1).')
    if min(args.steps, args.restarts, args.iters, args.batch_size, args.window_left, args.window_right) < 1:
        parser.error('steps, restarts, iters, batch-size and window offsets must be positive.')
    if 'gain_baseline' in args.threats and not {'gain', 'baseline'} <= set(args.threats):
        parser.error('gain_baseline is calibrated on top of gain and baseline; include both.')
    return args


def main():
    args = parse_args()
    if args.device == 'cuda_if_available':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    npz_path, x_train, y_train = load_train(args.data_dir, args.session)
    split = int(round(len(x_train) * (1 - args.held_out_fraction)))
    window = args.window_left + args.window_right
    if min(split, len(x_train) - split) < window:
        raise ValueError('Not enough time points for the fit / held-out parts.')
    x_fit, y_fit = x_train[:split], y_train[:split]
    x_held, y_held = x_train[split:], y_train[split:]
    print('DATA:', npz_path, '| train', x_train.shape, '-> fit', x_fit.shape, '+ held-out', x_held.shape,
          '(VALID split not used)', flush=True)

    print('Reference decoder (ridge on raw windows):', flush=True)
    r2_clean, penalty, decoder = fit_ridge(x_fit, y_fit, x_held, y_held, args.window_left,
                                           args.window_right, device, args.batch_size)
    print(f'Reference decoder: penalty={penalty:g}, held-out clean R2={r2_clean:.4f}', flush=True)
    if r2_clean < 0.3:
        print('WARNING: the reference decoder is weak (R2 < 0.3); calibrated budgets are not meaningful.',
              flush=True)

    # Statistics on the WHOLE train split, exactly as the forks and robust_eval.py compute them.
    x_train_t = torch.from_numpy(x_train)
    noise_a, noise_b, noise_floor = fit_noise_model(x_train_t, args.noise_model, args.noise_floor)
    stats = dict(noise_a=noise_a.to(device), noise_b=noise_b.to(device), noise_floor=noise_floor,
                 scale=neuron_scale(x_train_t, args.baseline_floor).to(device))
    padded = make_windows(x_held, args.window_left, args.window_right).to(device)
    y = torch.from_numpy(y_held).to(device)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out = args.out_root.expanduser() / f'{args.session}_tau{args.tau:g}_{args.oracle}_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    oracles = ORACLES if args.oracle == 'both' else (args.oracle,)
    results = {}
    for oracle in oracles:
        print(f'\nORACLE: {oracle} | tau={args.tau:g}', flush=True)
        known = dict(r2_clean=r2_clean, median_std=float(stats['scale'].median()))
        results[oracle] = {}
        for kind in args.threats:
            result = calibrate_threat(kind, oracle, decoder, padded, y, window, stats, args, device, known)
            results[oracle][kind] = result
            if kind in ('gain', 'baseline'):
                known[kind] = result['budget']
            save_json(out / 'calibration.json', dict(results=results))

    summary = {}
    print(f'\nCALIBRATED BUDGETS (reference ridge R2={r2_clean:.4f}, tau={args.tau:g})', flush=True)
    for oracle, by_threat in results.items():
        print(f'[{oracle}]', flush=True)
        summary[oracle] = {}
        for kind, r in by_threat.items():
            if kind == 'gain_baseline':
                value = dict(gain_epsilon=r['gain_epsilon'], baseline_epsilon=r['baseline_epsilon'], scale=r['scale'])
                text = f"gain={r['gain_epsilon']:.4g}, baseline={r['baseline_epsilon']:.4g} (scale {r['scale']:.3f})"
            else:
                value, text = r['budget'], f"{r['budget']:.4g}"
            summary[oracle][kind] = value
            rel = r['mean_abs_delta_over_std']
            print(f'  {kind:<14} {text:<45} mean|delta|/std={rel if rel is None else round(rel, 3)}'
                  f'  [{r["status"]}]', flush=True)
        flags = []
        by = summary[oracle]
        if 'constant' in by:
            flags.append(f"--epsilon {by['constant']:.4g} --alpha {by['constant'] / 5:.4g}")
        if 'noise' in by:
            flags.append(f"--noise-coef {by['noise']:.4g}")
        if 'gain' in by:
            flags.append(f"--gain-epsilon {by['gain']:.4g}")
        if 'baseline' in by:
            flags.append(f"--baseline-epsilon {by['baseline']:.4g}")
        print('  runner / robust_eval.py flags:', ' '.join(flags), flush=True)
        summary[oracle]['flags'] = ' '.join(flags)

    save_json(out / 'calibration.json', dict(
        session=args.session, data=str(npz_path), tau=args.tau,
        reference_decoder=dict(kind='ridge on raw windows', penalty=penalty, held_out_clean_r2=r2_clean,
                               fit_samples=len(x_fit), held_out_samples=len(x_held), window=window),
        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        summary=summary, results=results))
    print('Saved:', out / 'calibration.json', flush=True)


if __name__ == '__main__':
    main()

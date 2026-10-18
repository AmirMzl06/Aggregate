"""C-CO12: clean vs PGD vs spectral-gap adversarial CEBRA (Acorn-SVD fork).

Run from anywhere; place this file BESIDE the Acorn-SVD-v4/ fork:
    python -u run_acorn_svd_v4.py --data /path/to/C-CO12.npz

It also works from inside the fork, as Acorn-SVD-v4/run_acorn_svd.py.
By default V4 outputs go to the separate ACORN_SVD_RESULTS_V4/ directory.
Use --plan-only to inspect the number of training jobs WITHOUT loading data or CEBRA.

Arms (--arms):
    clean             CEBRA without adversarial updates.
    pgd               Previous project's PGD on the reference (L2, eps=3), bit-exact.
    svd_gap           Spectral-gap subspace attack, per-window budget capped at eps=3.
    svd_gap_uncapped  Same, budget limited only by the spectral gap.
    svd_tangent       On-manifold: tangent space of the top-k structure, half-gap budget.
    svd_split         Both: complement (eps=3) + tangent (half-gap), joint certificate.
    svd_tangent_budget Ablation: half-gap budget without the projection.
    svd_subspace      Subspace projection with the fixed budget eps=3.
    svd_subspace_cert Same, final perturbation projected onto ||D||_2 <= gamma * gap.
    svd_cert_every    Same, projected after every PGD step (uses more of the budget).
    svd_cert_radial   Ablation: final perturbation shrunk as a whole (the earlier certificate).
    svd_split_half    svd_split with half of the complement budget (trade-off knob).
    svd_split_every   svd_split, jointly certified after every PGD step.
    svd_tangent_fixed Ablation: tangent projection with the fixed budget eps.
    svd_gap_budget    Ablation: gap budget without the projection.

For every (seed, arm) the runner:
    - fits the encoder and trains the same MLP decoder on clean train embeddings;
    - reports validation R2;
    - unless --no-eval-attack, reports R2 under held-out attacks on the validation
      windows (plain PGD, spectral gap, tangent, and complement vs tangent at the same
      radius; --eval-epsilon, --eval-steps, --eval-restarts);
    - reports R2 on corrupted recordings: Poisson resampling, gain drift, random and
      nested 5-50% unit loss, and loss of the 20% most active, variable or important units.

--epsilon-unit window expresses all budgets as fractions of the median norm of
centred training windows, so that sessions are comparable. --scale-only prints
that norm, the budgets in both units and the spectral gaps of the session, then
exits without training. Results are saved after every arm in a new, unique run
directory; no checkpoints are stored.
"""

import argparse
import csv
import gc
import inspect
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch import nn

# Prevent the local fork import from generating __pycache__ files.
sys.dont_write_bytecode = True

# Resolve the fork from either supported layout:
#     Aggregate/run_acorn_svd_v4.py + Aggregate/Acorn-SVD-v4/   (recommended)
#     Aggregate/Acorn-SVD-v4/run_acorn_svd.py                   (also supported)
# Do not silently fall back to the OLD Acorn-SVD/ fork.
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "cebra" / "__init__.py").is_file():
    ROOT = SCRIPT_DIR
elif (SCRIPT_DIR / "Acorn-SVD-v4" / "cebra" / "__init__.py").is_file():
    ROOT = SCRIPT_DIR / "Acorn-SVD-v4"
else:
    raise FileNotFoundError(
        "Cannot find the V4 fork. Put run_acorn_svd_v4.py BESIDE Acorn-SVD-v4/, "
        "or run Acorn-SVD-v4/run_acorn_svd.py from inside the fork. "
        f"Script directory: {SCRIPT_DIR}"
    )

DEFAULT_OUTPUT = ROOT.parent / "ACORN_SVD_RESULTS_V4"
DEFAULT_DATA = Path("/data/hossein/mm_project/perich_data_valid_final_raw/C-CO12.npz")

EPSILON = 3.0  # Training budget (cap for the spectral arms); --train-epsilon overrides.
EPS_STEPS = 10
ALPHA_FRACTION = 1 / 5  # Step size as a fraction of the budget: 0.6 for eps=3.
ATTACK_NORM = "l2"
UNCAPPED_EPSILON = 1e3  # Effectively no cap: the spectral gap sets the budget.

# Spectral-gap settings, shared by all spectral arms and the evaluation attack.
GAMMA = 0.9
K_MIN, K_MAX = 1, 8
CENTER = True
PRESERVE_MEAN = False
DIAGNOSTICS_EVERY = 50

LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 3000
TEMPERATURE = 0.4
ARCH = "offset36-model-more-dropout"
DEVICE = "cuda_if_available"

# Decoder: identical to the previous runner.
MLP_EPOCHS = 2500
MLP_HIDDEN = 64
MLP_DROP = 0.4
MLP_LR = 1e-3
EVAL_BATCH_SIZE = 8192

# Held-out attack on the frozen encoder + decoder (prediction MSE).
EVAL_EPSILON = 3.0  # Fixed across runs so that robustness is comparable (--eval-epsilon).
EVAL_STEPS = 20  # --eval-steps; the step size is 2.5 * eps / steps.
EVAL_RESTARTS = 1  # --eval-restarts; the worst case over restarts is kept per window.
EVAL_ATTACK_BATCH = 2048
EVAL_SEED = 1234
WINDOW_SAMPLES = 4096  # Windows used to measure the typical window norm.
WINDOW_LENGTH = 36  # receptive field of offset36-model-more-dropout

# Recording corruptions of the validation data (Poisson resampling, lost units,
# per-neuron gain drift), averaged over a few random draws.
NATURAL_DRAWS = 5
NATURAL_SEED = 4321
NESTED_CURVE = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)  # nested random unit loss per draw
STRUCTURED_FRACTION = 0.2  # most active / most variable / most important units

ARMS = {
    "clean": dict(training_mode="standard"),
    "pgd": dict(adv_spectral_mode="none"),
    "svd_gap": dict(adv_spectral_mode="gap"),
    "svd_gap_uncapped": dict(adv_spectral_mode="gap",
                             adv_epsilon=UNCAPPED_EPSILON,
                             adv_alpha=UNCAPPED_EPSILON / 5),
    "svd_tangent": dict(adv_spectral_mode="tangent"),
    "svd_split": dict(adv_spectral_mode="split"),
    "svd_tangent_budget": dict(adv_spectral_mode="tangent_budget"),
    "svd_subspace": dict(adv_spectral_mode="subspace"),
    "svd_subspace_cert": dict(adv_spectral_mode="subspace_cert"),
    "svd_cert_every": dict(adv_spectral_mode="subspace_cert", adv_spectral_certify="every"),
    "svd_cert_radial": dict(adv_spectral_mode="subspace_cert", adv_spectral_certify="radial"),
    "svd_split_half": dict(adv_spectral_mode="split", adv_spectral_split_complement=0.5),
    "svd_split_every": dict(adv_spectral_mode="split", adv_spectral_certify="every"),
    "svd_tangent_fixed": dict(adv_spectral_mode="tangent_fixed"),
    "svd_gap_budget": dict(adv_spectral_mode="gap_budget"),
}
DEFAULT_ARMS = ("clean", "pgd", "svd_subspace_cert", "svd_cert_every", "svd_tangent", "svd_split",
                "svd_split_half")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_fork():
    if not (ROOT / "cebra" / "__init__.py").is_file():
        raise FileNotFoundError(f"Expected the Acorn-SVD fork next to this script: {ROOT}")
    # Avoid accidentally using an installed/unrelated version of CEBRA.
    for key in list(sys.modules):
        if key == "cebra" or key.startswith("cebra."):
            del sys.modules[key]
    sys.path.insert(0, str(ROOT))
    import cebra
    import cebra.solver.spectral_gap
    if not Path(cebra.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError(f"Incorrect CEBRA package imported: {cebra.__file__}")
    params = inspect.signature(cebra.CEBRA.__init__).parameters
    required = {"adv_spectral_mode", "adv_spectral_gamma", "adv_diagnostics_every"}
    missing = required - params.keys()
    if missing:
        raise RuntimeError(f"Acorn-SVD fork lacks expected parameters: {sorted(missing)}")
    return cebra


def load_data(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Session file not found: {path}")
    with np.load(path, allow_pickle=False) as f:
        x_train, x_valid, y_train, y_valid = (
            np.asarray(f[key], dtype=np.float32)
            for key in ("train_data", "valid_data", "train_label", "valid_label"))
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_valid.ndim == 1:
        y_valid = y_valid[:, None]
    for key, arr in (("train_data", x_train), ("valid_data", x_valid),
                     ("train_label", y_train), ("valid_label", y_valid)):
        if arr.ndim != 2 or not all(arr.shape) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid {key} shape or NaN/Inf: {arr.shape}")
    if x_train.shape[1] != x_valid.shape[1] or y_train.shape[1] != y_valid.shape[1]:
        raise ValueError("Train/validation feature or label dimensions differ")
    if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
        raise ValueError("Number of labels and neural samples must match")
    if min(len(x_train), len(x_valid)) < 36:
        raise ValueError("offset36 encoder requires >=36 time bins per split")
    return tuple(np.ascontiguousarray(a) for a in (x_train, x_valid, y_train, y_valid))


def encoder_config(arm: str, iterations: int, epsilon: float = EPSILON, gamma: float = GAMMA,
                   k_min: int = K_MIN, k_max: int = K_MAX) -> dict:
    config = dict(
        model_architecture=ARCH,
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        time_offsets=1,
        max_iterations=iterations,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        learning_rate=3e-4,
        temperature_mode="constant",
        training_mode="adversarial",
        adv_epsilon=epsilon,
        adv_alpha=epsilon * ALPHA_FRACTION,
        adv_steps=EPS_STEPS,
        attack_norm=ATTACK_NORM,
        adv_spectral_gamma=gamma,
        adv_spectral_k_min=k_min,
        adv_spectral_k_max=k_max,
        adv_spectral_center=CENTER,
        adv_spectral_preserve_mean=PRESERVE_MEAN,
        adv_attack_eval_mode=False,  # Same attack protocol as the previous project.
        adv_diagnostics_every=DIAGNOSTICS_EVERY,
        device=DEVICE,
        verbose=True,  # CEBRA training progress and iteration/loss logs.
    )
    config.update(ARMS[arm])
    return config


class TwoLayerMLP(nn.Module):

    def __init__(self, dim, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, MLP_HIDDEN), nn.LayerNorm(MLP_HIDDEN),
            nn.ReLU(), nn.Dropout(MLP_DROP), nn.Linear(MLP_HIDDEN, out),
        )

    def forward(self, x):
        return self.net(x)


def train_decoder(z_train, y_train, seed, device):
    seed_all(seed)  # Same decoder seed in all arms.
    decoder = TwoLayerMLP(z_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=MLP_LR)
    criterion = nn.MSELoss()
    z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    decoder.train()
    print(f"[DECODER] Start {MLP_EPOCHS} epochs; all {y_train.shape[1]} label columns", flush=True)
    decoder_start = time.perf_counter()
    for epoch in range(MLP_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(decoder(z), y)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Nonfinite decoder loss at epoch {epoch + 1}")
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 250 == 0 or epoch + 1 == MLP_EPOCHS:
            print(f"[DECODER] epoch {epoch + 1}/{MLP_EPOCHS} | train_MSE={loss.item():.6f} "
                  f"| elapsed={time.perf_counter() - decoder_start:.1f}s", flush=True)
    return decoder


def predict_decoder(decoder, embeddings):
    decoder.eval()  # Disable Dropout for R2 evaluation.
    device = next(decoder.parameters()).device
    preds = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), EVAL_BATCH_SIZE):
            z = torch.as_tensor(embeddings[start:start + EVAL_BATCH_SIZE],
                                dtype=torch.float32, device=device)
            preds.append(decoder(z).cpu().numpy())
    return np.concatenate(preds, axis=0)


def mean_r2(y_true, y_pred):
    per_label = np.atleast_1d(r2_score(y_true, y_pred, multioutput="raw_values"))
    if not np.isfinite(per_label).all():
        raise ValueError(f"Nonfinite R2 score: {per_label}")
    return float(per_label.mean()), [float(v) for v in per_label]


def validation_windows(x_valid, y_valid, offset):
    """All validation windows that lie fully inside the recording, as in training."""
    length = offset.left + offset.right
    windows = torch.as_tensor(x_valid).unfold(0, length, 1)  # (M, neurons, length)
    labels = y_valid[offset.left:offset.left + len(windows)]
    return windows.contiguous(), labels


def natural_corruptions(x, rng):
    """Recording perturbations of validation spike counts: noise, lost units, gain drift."""
    yield "poisson", rng.poisson(np.clip(x, 0, None)).astype(np.float32)
    for fraction in (0.1, 0.3):
        keep = np.ones(x.shape[1], dtype=np.float32)
        keep[rng.choice(x.shape[1], int(round(fraction * x.shape[1])), replace=False)] = 0
        yield f"dropout{int(fraction * 100)}", x * keep
    yield "gain30", x * np.exp(rng.normal(0, 0.3, x.shape[1])).astype(np.float32)


def nested_unit_loss(x, rng):
    """Random unit loss with nested sets: each draw removes a growing prefix of one order."""
    order = rng.permutation(x.shape[1])
    for fraction in NESTED_CURVE:
        keep = np.ones(x.shape[1], dtype=np.float32)
        keep[order[:int(round(fraction * x.shape[1]))]] = 0
        yield f"nested{int(fraction * 100)}", x * keep


def _decode_r2(encoder, decoder, x, y):
    z = np.asarray(encoder.transform(x), dtype=np.float32)
    return mean_r2(y, predict_decoder(decoder, z))


def structured_unit_loss(encoder, decoder, x, y):
    """Remove the most active, most variable or most decoding-relevant units.

    Decoding relevance is the drop of validation R2 when a single unit is
    removed (leave-one-unit-out), measured for every model separately.
    """
    count = int(round(STRUCTURED_FRACTION * x.shape[1]))
    base = _decode_r2(encoder, decoder, x, y)[0]
    importance = np.empty(x.shape[1])
    for unit in range(x.shape[1]):
        keep = np.ones(x.shape[1], dtype=np.float32)
        keep[unit] = 0
        importance[unit] = base - _decode_r2(encoder, decoder, x * keep, y)[0]
    out = {}
    tag = int(STRUCTURED_FRACTION * 100)
    for name, score in (("top", x.mean(0)), ("var", x.var(0)), ("imp", importance)):
        keep = np.ones(x.shape[1], dtype=np.float32)
        keep[np.argsort(-score)[:count]] = 0
        r2, per_label = _decode_r2(encoder, decoder, x * keep, y)
        out[f"r2_drop{name}{tag}"] = r2
        out[f"r2_drop{name}{tag}_per_label"] = per_label
    out["unit_importance_max"] = float(importance.max())
    out["unit_importance_top_share"] = float(
        np.sort(np.clip(importance, 0, None))[::-1][:count].sum() /
        max(np.clip(importance, 0, None).sum(), 1e-12))
    return out


def natural_robustness(encoder, decoder, x_valid, y_valid):
    """Validation R2 of the frozen encoder + decoder on corrupted recordings."""
    rng = np.random.default_rng(NATURAL_SEED)
    # A separate stream keeps the draws of the original corruptions unchanged.
    curve_rng = np.random.default_rng(NATURAL_SEED + 1)
    scores, labels = {}, {}
    for _ in range(NATURAL_DRAWS):
        for name, corrupted in list(natural_corruptions(x_valid, rng)) + list(
                nested_unit_loss(x_valid, curve_rng)):
            r2, per_label = _decode_r2(encoder, decoder, corrupted, y_valid)
            scores.setdefault(name, []).append(r2)
            labels.setdefault(name, []).append(per_label)
    out = {}
    for name, values in scores.items():
        out[f"r2_{name}"] = float(np.mean(values))
        out[f"r2_{name}_draw_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        out[f"r2_{name}_per_label"] = [float(v) for v in np.mean(labels[name], axis=0)]
    out.update(structured_unit_loss(encoder, decoder, x_valid, y_valid))
    return out


def attack_eval_configs(spectral_gap, epsilon=EVAL_EPSILON, steps=EVAL_STEPS):
    shared = dict(epsilon=epsilon, alpha=2.5 * epsilon / steps, steps=steps, norm="l2")
    spectral = dict(gamma=GAMMA, k_min=K_MIN, k_max=K_MAX, center=CENTER)
    return {
        "pgd": spectral_gap.AttackConfig(mode="none", **shared),
        "svd_gap": spectral_gap.AttackConfig(mode="gap", **spectral, **shared),
        "svd_tangent": spectral_gap.AttackConfig(mode="tangent", **spectral, **shared),
        # Equal radius, different directions: off-manifold vs on-manifold.
        "complement": spectral_gap.AttackConfig(mode="subspace", **spectral, **shared),
        "tangent_fixed": spectral_gap.AttackConfig(mode="tangent_fixed", **spectral, **shared),
    }


def robustness(cebra, encoder, decoder, x_valid, y_valid, z_valid, epsilon=EVAL_EPSILON,
               steps=EVAL_STEPS, restarts=EVAL_RESTARTS):
    """Validation R2 of the frozen encoder + decoder under held-out attacks."""
    spectral_gap = cebra.solver.spectral_gap
    model = encoder.solver_.model
    device = next(model.parameters()).device
    model.eval()
    decoder.eval()
    windows, labels = validation_windows(x_valid, y_valid, model.get_offset())
    offset = model.get_offset()
    # The per-window model must reproduce the transform used for clean R2.
    with torch.no_grad():
        probe = model(windows[:256].to(device)).cpu().numpy()
    mismatch = float(np.abs(probe - z_valid[offset.left:offset.left + len(probe)]).max())
    if mismatch > 1e-3:
        raise RuntimeError(f"Window embedding does not match transform (max diff {mismatch})")

    out = {"window_mismatch": mismatch, "num_windows": int(len(windows))}
    configs = attack_eval_configs(spectral_gap, epsilon, steps)
    for name, config in [("clean", None)] + list(configs.items()):
        seed_all(EVAL_SEED)
        preds, budgets = [], []
        for start in range(0, len(windows), EVAL_ATTACK_BATCH):
            batch = windows[start:start + EVAL_ATTACK_BATCH].to(device)
            target = torch.as_tensor(labels[start:start + len(batch)], device=device)
            if config is None:
                with torch.no_grad():
                    preds.append(decoder(model(batch)).cpu().numpy())
                continue
            frame = spectral_gap.spectral_frame(batch, config) if config.spectral else None
            if frame is not None:
                budgets.append(frame.epsilon.cpu())

            def loss_fn(reference, target=target):
                return (decoder(model(reference)) - target).square().mean()

            worst, worst_error = None, None
            for _ in range(restarts):  # keep the strongest restart per window
                attacked = spectral_gap.attack(batch, loss_fn, config, frame)
                with torch.no_grad():
                    prediction = decoder(model(attacked))
                error = (prediction - target).square().sum(dim=1)
                if worst is None:
                    worst, worst_error = prediction, error
                else:
                    better = error > worst_error
                    worst = torch.where(better[:, None], prediction, worst)
                    worst_error = torch.where(better, error, worst_error)
            preds.append(worst.cpu().numpy())
        r2, per_label = mean_r2(labels, np.concatenate(preds))
        out[f"r2_{name}"] = r2
        out[f"r2_{name}_per_label"] = per_label
        if budgets:
            out[f"eval_eps_{name}_mean"] = float(torch.cat(budgets).mean())
    return out


def attack_statistics(solver) -> dict:
    """Training-time attack statistics averaged over iterations."""
    log = solver.log
    stats = {}
    for key in ("adv_loss", "adv_delta_norm", "adv_delta_norm_p10", "adv_delta_norm_p50",
                "adv_delta_norm_p90", "adv_eps_mean", "adv_eps_min", "adv_gap_limited",
                "adv_k_mean", "adv_gap_mean", "adv_tangent_norm", "adv_complement_norm",
                "adv_tangent_eps_mean", "adv_cert_active", "adv_cert_retained"):
        if log.get(key):
            stats[f"train_{key}"] = float(np.mean(log[key]))
    diagnostics = log.get("adv_diagnostics", [])
    for key in ("sv_err_max", "left_subspace_err_max", "right_subspace_err_max",
                "spectral_drift_mean", "delta_over_gap_max", "gap_kept_min",
                "collapse_fraction", "mode_exchange_fraction", "tangent_energy_fraction",
                "tangent_chance_fraction"):
        values = [d[key] for d in diagnostics if np.isfinite(d.get(key, np.nan))]
        if values:
            reduce = np.max if key.endswith("_max") else np.min if key.endswith("_min") else np.mean
            stats[f"diag_{key}"] = float(reduce(values))
    return stats


def sample_windows(x, length=WINDOW_LENGTH, samples=WINDOW_SAMPLES):
    """Fixed random sample of training windows, shape (windows, time, units)."""
    rng = np.random.default_rng(0)
    count = min(samples, len(x) - length + 1)
    starts = rng.choice(len(x) - length + 1, count, replace=False)
    return np.stack([x[start:start + length] for start in starts])


def window_norm_median(windows):
    """Median Frobenius norm of temporally centred training windows (the attack's scale)."""
    centred = windows - windows.mean(axis=1, keepdims=True)
    return float(np.median(np.linalg.norm(centred.reshape(len(windows), -1), axis=1)))


def window_gaps(spectral_gap, windows, gamma, k_min, k_max):
    """Quantiles of the selected spectral gap g_k and of k over the sampled windows."""
    config = spectral_gap.AttackConfig(epsilon=1.0, alpha=1.0, steps=1, norm="l2", mode="gap",
                                       gamma=gamma, k_min=k_min, k_max=k_max, center=CENTER)
    frame = spectral_gap.spectral_frame(torch.from_numpy(windows).mT.contiguous(), config)
    gap, k = frame.gap.double().numpy(), frame.k.double().numpy()
    stats = {f"gap_p{q}": float(np.quantile(gap, q / 100)) for q in (10, 50, 90)}
    stats["k_median"] = float(np.median(k))
    return stats


def run_model(cebra, arm, seed, arrays, iterations, eval_attack, epsilon=EPSILON,
              spectral=None, evaluation=None, meta=None) -> dict:
    x_train, x_valid, y_train, y_valid = arrays
    encoder = decoder = z_train = z_valid = None
    arm_start = time.perf_counter()
    try:
        seed_all(seed)  # Same initialization and sampling seed in each arm.
        print(f"\n{'=' * 75}", flush=True)
        print(f"[START] {arm} | seed={seed} | eps={epsilon} | {iterations} CEBRA iterations",
              flush=True)
        encoder = cebra.CEBRA(**encoder_config(arm, iterations, epsilon, **(spectral or {})))
        encoder.fit(x_train, y_train)
        encoder_seconds = time.perf_counter() - arm_start
        print(f"[ENCODER] {arm}: {encoder_seconds / 60:.1f} min "
              f"({encoder_seconds / iterations:.3f} s/iteration)", flush=True)

        z_train = np.asarray(encoder.transform(x_train), dtype=np.float32)
        z_valid = np.asarray(encoder.transform(x_valid), dtype=np.float32)
        if (len(z_train) != len(x_train) or len(z_valid) != len(x_valid)
                or not np.isfinite(z_train).all() or not np.isfinite(z_valid).all()):
            raise ValueError(f"Bad embeddings or alignment for arm={arm}")

        decoder_start = time.perf_counter()
        decoder = train_decoder(z_train, y_train, seed + 100_000,
                                torch.device(encoder.device_))
        decoder_seconds = time.perf_counter() - decoder_start
        predictions = predict_decoder(decoder, z_valid)
        if predictions.shape != y_valid.shape or not np.isfinite(predictions).all():
            raise ValueError(f"Bad decoder prediction for arm={arm}")
        r2, per_label = mean_r2(y_valid, predictions)
        row = {
            "model": arm, "seed": seed, "epsilon": float(encoder.adv_epsilon),
            **(meta or {}),
            "gamma": float(encoder.adv_spectral_gamma),
            "k_range": [encoder.adv_spectral_k_min, encoder.adv_spectral_k_max],
            "spectral_mode": encoder.adv_spectral_mode
            if encoder.training_mode == "adversarial" else "clean",
            "valid_mean_r2": r2, "valid_r2_per_label": per_label,
            "encoder_seconds": float(encoder_seconds),
            "seconds_per_iteration": float(encoder_seconds / iterations),
            "decoder_seconds": float(decoder_seconds),
        }
        print(f"[RESULT] {arm} seed={seed}: validation mean R2 = {r2:.6f} | per label = {per_label}",
              flush=True)
        if encoder.training_mode == "adversarial":
            row.update(attack_statistics(encoder.solver_))
        row.update(natural_robustness(encoder, decoder, x_valid, y_valid))
        if eval_attack:
            eval_start = time.perf_counter()
            row.update(robustness(cebra, encoder, decoder, x_valid, y_valid, z_valid,
                                  **(evaluation or {})))
            row["eval_attack_seconds"] = float(time.perf_counter() - eval_start)
            print(f"[ROBUST] {arm} seed={seed}: R2 clean windows = {row['r2_clean']:.4f} | "
                  f"PGD = {row['r2_pgd']:.4f} | SVD-gap = {row['r2_svd_gap']:.4f} | "
                  f"SVD-tangent = {row['r2_svd_tangent']:.4f} | complement = "
                  f"{row['r2_complement']:.4f} | tangent (eps) = {row['r2_tangent_fixed']:.4f}",
                  flush=True)
        row["total_seconds"] = float(time.perf_counter() - arm_start)
        return row
    finally:
        del encoder, decoder, z_train, z_valid
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


SCALAR_COLUMNS = (
    "model", "seed", "spectral_mode", "epsilon", "epsilon_nominal", "epsilon_unit",
    "window_norm_median", "valid_mean_r2", "r2_clean", "r2_pgd", "r2_svd_gap",
    "eval_eps_svd_gap_mean", "r2_svd_tangent", "r2_complement", "r2_tangent_fixed",
    "r2_poisson", "r2_dropout10", "r2_dropout30", "r2_gain30", "r2_nested5", "r2_nested10",
    "r2_nested20", "r2_nested30", "r2_nested40", "r2_nested50", "r2_droptop20", "r2_dropvar20",
    "r2_dropimp20", "unit_importance_top_share", "train_adv_eps_mean", "train_adv_gap_limited", "train_adv_cert_active",
    "train_adv_cert_retained", "train_adv_k_mean", "train_adv_delta_norm",
    "train_adv_delta_norm_p10", "train_adv_delta_norm_p90", "train_adv_loss",
    "train_adv_tangent_norm", "train_adv_complement_norm", "diag_sv_err_max",
    "diag_left_subspace_err_max", "diag_right_subspace_err_max", "diag_spectral_drift_mean",
    "diag_delta_over_gap_max", "diag_gap_kept_min", "diag_collapse_fraction",
    "diag_mode_exchange_fraction", "diag_tangent_energy_fraction",
    "diag_tangent_chance_fraction", "seconds_per_iteration", "encoder_seconds",
    "decoder_seconds", "total_seconds",
)


def save_results(out: Path, results: list):
    """Persist all completed arms atomically, even if a later arm is slow/fails."""
    json_path, temp_json = out / "results_so_far.json", out / ".results_so_far.json.tmp"
    temp_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp_json, json_path)
    num_labels = len(results[0]["valid_r2_per_label"])
    csv_path, temp_csv = out / "results_so_far.csv", out / ".results_so_far.csv.tmp"
    with temp_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(SCALAR_COLUMNS) + [f"r2_label_{i}" for i in range(num_labels)])
        for row in results:
            writer.writerow([row.get(key, "") for key in SCALAR_COLUMNS] +
                            row["valid_r2_per_label"])
    os.replace(temp_csv, csv_path)


def arm_label(row: dict) -> str:
    if row["model"] == "clean":
        return "clean"
    unit = "w" if row.get("epsilon_unit") == "window" else ""
    return f"{row['model']}@{row.get('epsilon_nominal', row['epsilon']):g}{unit}"


def summarize(results: list) -> list:
    summary = []
    for arm in dict.fromkeys(arm_label(row) for row in results):
        rows = [row for row in results if arm_label(row) == arm]
        entry = {"model": arm, "num_seeds": len(rows)}
        for key in ("valid_mean_r2", "r2_pgd", "r2_svd_gap", "r2_svd_tangent", "r2_complement",
                    "r2_tangent_fixed", "r2_poisson", "r2_dropout30", "r2_nested30",
                    "r2_nested50", "r2_droptop20", "r2_dropimp20", "r2_gain30",
                    "train_adv_delta_norm", "train_adv_cert_active", "seconds_per_iteration"):
            values = [row[key] for row in rows if key in row]
            if values:
                entry[f"{key}_mean"] = float(np.mean(values))
                entry[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(entry)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=list(ARMS))
    parser.add_argument("--iterations", type=int, default=MAX_ITER)
    parser.add_argument("--train-epsilon", type=float, nargs="+", default=[EPSILON],
                        help="Training budget(s) of the pgd/svd arms (step = budget/5); several "
                             "values give a budget sweep.")
    parser.add_argument("--eval-epsilon", type=float, default=EVAL_EPSILON,
                        help="Radius of the held-out attacks (the same for every arm).")
    parser.add_argument("--eval-steps", type=int, default=EVAL_STEPS)
    parser.add_argument("--eval-restarts", type=int, default=EVAL_RESTARTS,
                        help="Random restarts of each held-out attack; the worst is kept.")
    parser.add_argument("--epsilon-unit", choices=("absolute", "window"), default="absolute",
                        help="'window': --train-epsilon and --eval-epsilon are fractions of the "
                             "median norm of centred training windows, comparable across sessions.")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                        help="Training gap fraction; evaluation attacks keep gamma=0.9.")
    parser.add_argument("--k-min", type=int, default=K_MIN)
    parser.add_argument("--k-max", type=int, default=K_MAX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT,
                        help="Results directory (default: sibling ACORN_SVD_RESULTS_V4; "
                             "old results are never overwritten).")
    parser.add_argument("--plan-only", action="store_true",
                        help="List every planned training job and exit; no model imports, "
                             "data access, or GPU usage.")
    parser.add_argument("--no-eval-attack", action="store_true",
                        help="Skip the held-out attacked-R2 evaluation.")
    parser.add_argument("--scale-only", action="store_true",
                        help="Print the window norm, both budget units and the spectral gaps, "
                             "then exit without training.")
    args = parser.parse_args()

    # Fail fast on configuration mistakes before importing the model or using GPU.
    if args.iterations < 1 or args.eval_steps < 1 or args.eval_restarts < 1:
        parser.error("--iterations, --eval-steps and --eval-restarts must be >= 1")
    if not args.train_epsilon or any(not np.isfinite(v) or v <= 0 for v in args.train_epsilon):
        parser.error("--train-epsilon must contain positive finite values")
    if not np.isfinite(args.eval_epsilon) or args.eval_epsilon <= 0:
        parser.error("--eval-epsilon must be a positive finite value")
    if not np.isfinite(args.gamma) or not 0 < args.gamma < 1:
        parser.error("--gamma must be between 0 and 1")
    if args.k_min < 1 or args.k_max < args.k_min:
        parser.error("Require 1 <= --k-min <= --k-max")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds contains duplicates")
    if len(set(args.arms)) != len(args.arms):
        parser.error("--arms contains duplicates")
    if len(set(args.train_epsilon)) != len(args.train_epsilon):
        parser.error("--train-epsilon contains duplicates")

    # List exactly the work the user requested; one dataset per invocation.
    planned_jobs = [(seed, nominal, arm)
                    for seed in args.seeds
                    for i, nominal in enumerate(args.train_epsilon)
                    for arm in args.arms
                    if i == 0 or arm not in ("clean", "svd_gap_uncapped")]
    print(f"[PLAN] V4 fork: {ROOT}", flush=True)
    print(f"[PLAN] Dataset: {args.data} | output root: {args.out}", flush=True)
    print(f"[PLAN] {len(planned_jobs)} encoder training jobs; "
          f"{len(args.seeds)} seeds x arms={args.arms} x "
          f"train epsilons={args.train_epsilon} ({args.epsilon_unit})", flush=True)
    for index, (seed, nominal, arm) in enumerate(planned_jobs, 1):
        print(f"[PLAN] {index:>3}/{len(planned_jobs)}  "
              f"seed={seed}  arm={arm}  train_epsilon={nominal:g} {args.epsilon_unit}",
              flush=True)
    if args.epsilon_unit == "window":
        print("[WARNING] --epsilon-unit window treats 1.0 as ONE ENTIRE median centred "
              "window Frobenius norm; 3.0 = THREE times that norm, not 3%.", flush=True)
    if args.plan_only:
        print("[PLAN ONLY] No data loaded; no training or evaluation performed.", flush=True)
        return

    spectral = dict(gamma=args.gamma, k_min=args.k_min, k_max=args.k_max)
    cebra = load_fork()
    print(f"[FORK] {cebra.__file__}", flush=True)
    arrays = load_data(args.data)
    print(f"[DATA] Train X={arrays[0].shape}, Y={arrays[2].shape}", flush=True)
    print(f"[DATA] Valid X={arrays[1].shape}, Y={arrays[3].shape}", flush=True)
    windows = sample_windows(arrays[0])
    norm = window_norm_median(windows)
    gaps = window_gaps(cebra.solver.spectral_gap, windows, **spectral)
    scale = norm if args.epsilon_unit == "window" else 1.0
    train_eps = [nominal * scale for nominal in args.train_epsilon]
    evaluation = dict(epsilon=args.eval_epsilon * scale, steps=args.eval_steps,
                      restarts=args.eval_restarts)
    print(f"[SCALE] median centred window norm = {norm:.3f} | training eps = "
          f"{[round(e, 4) for e in train_eps]} | evaluation eps = {evaluation['epsilon']:.4f} "
          f"({args.eval_steps} steps, {args.eval_restarts} restart(s))", flush=True)
    fractions = [float(f"{e / norm:.4g}") for e in train_eps]
    print(f"[SCALE] as window fractions: training eps = {fractions} | evaluation eps = "
          f"{evaluation['epsilon'] / norm:.4g}", flush=True)
    print(f"[SCALE] gap g_k (k in [{args.k_min}, {args.k_max}]) p10/p50/p90 = "
          f"{gaps['gap_p10']:.3f} / {gaps['gap_p50']:.3f} / {gaps['gap_p90']:.3f} | median k = "
          f"{gaps['k_median']:g} | gamma * median gap = {args.gamma * gaps['gap_p50']:.3f}",
          flush=True)
    if args.scale_only:
        return

    # mkdtemp uses atomic creation; runs cannot overwrite one another.
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(tempfile.mkdtemp(prefix=f"{args.data.stem}_{stamp}_", dir=args.out))
    print(f"[START] New experiment: {args.data.stem} | output: {out}", flush=True)

    config = {
        "data_path": str(args.data), "fork": str(ROOT), "cebra_import": str(cebra.__file__),
        "runner_file": str(Path(__file__).resolve()), "runner_version": "v4-sibling-ready",
        "planned_jobs": len(planned_jobs),
        "arms": args.arms, "seeds": args.seeds, "iterations": args.iterations,
        "window_norm_median": norm, "window_gap": gaps, "epsilon_unit": args.epsilon_unit,
        "evaluation": evaluation,
        "encoder_configs": {f"{arm}@{nominal:g}": encoder_config(arm, args.iterations, eps,
                                                                 spectral["gamma"],
                                                                 spectral["k_min"],
                                                                 spectral["k_max"])
                            for nominal, eps in zip(args.train_epsilon, train_eps)
                            for arm in args.arms},
        "decoder": {"architecture": "Linear -> LayerNorm -> ReLU -> Dropout -> Linear",
                    "epochs": MLP_EPOCHS, "hidden": MLP_HIDDEN, "dropout": MLP_DROP,
                    "learning_rate": MLP_LR},
        "eval_attack": None if args.no_eval_attack else {
            name: vars(c) for name, c in attack_eval_configs(
                cebra.solver.spectral_gap, evaluation["epsilon"], args.eval_steps).items()},
        "shapes": {"x_train": arrays[0].shape, "x_valid": arrays[1].shape,
                   "y_train": arrays[2].shape, "y_valid": arrays[3].shape},
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Clean training and the uncapped arm do not depend on the training budget.
    eps_by_nominal = dict(zip(args.train_epsilon, train_eps))
    jobs = [(seed, nominal, eps_by_nominal[nominal], arm)
            for seed, nominal, arm in planned_jobs]
    results = []
    try:
        for seed, nominal, eps, arm in jobs:  # Seed-major: every arm of a seed finishes first.
            meta = dict(epsilon_nominal=nominal, epsilon_unit=args.epsilon_unit,
                        window_norm_median=norm, eval_epsilon=evaluation["epsilon"])
            results.append(run_model(cebra, arm, seed, arrays, args.iterations,
                                     not args.no_eval_attack, eps, spectral, evaluation, meta))
            save_results(out, results)  # Save right after each completed arm.
            print(f"[SAVED] {len(results)}/{len(jobs)} results -> {out / 'results_so_far.csv'}",
                  flush=True)
    except BaseException as exc:
        print(f"[INTERRUPTED] {type(exc).__name__}: {exc}", flush=True)
        print(f"[RECOVERY] Completed arms' results (if any) remain in: {out}", flush=True)
        raise

    os.replace(out / "results_so_far.csv", out / "results.csv")
    os.replace(out / "results_so_far.json", out / "results.json")
    summary = summarize(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 75, flush=True)
    print("SUMMARY (mean ± std over seeds)", flush=True)
    for entry in summary:
        text = " | ".join(
            f"{key[:-5]}={entry[key]:.4f}±{entry[key[:-5] + '_std']:.4f}"
            for key in entry if key.endswith("_mean"))
        print(f"{entry['model']:22s} n={entry['num_seeds']} | {text}", flush=True)
    print(f"[COMPLETE] Results saved in: {out}", flush=True)


if __name__ == "__main__":
    main()




# """C-CO12: clean vs PGD vs spectral-gap adversarial CEBRA (Acorn-SVD fork).

# Place this runner next to the Acorn-SVD/ directory; it imports the sibling fork:
#     python -u run_acorn_svd.py --data /path/to/C-CO12.npz

# Arms (--arms):
#     clean             CEBRA without adversarial updates.
#     pgd               Previous project's PGD on the reference (L2, eps=3), bit-exact.
#     svd_gap           Spectral-gap subspace attack, per-window budget capped at eps=3.
#     svd_gap_uncapped  Same, budget limited only by the spectral gap.
#     svd_tangent       On-manifold: tangent space of the top-k structure, half-gap budget.
#     svd_split         Both: complement + tangent, shared total eps and joint certificate.
#     svd_tangent_budget Ablation: half-gap budget without the projection.
#     svd_subspace      Subspace projection with the fixed budget eps=3.
#     svd_subspace_cert Same, final perturbation certified: ||D||_2 <= gamma * gap.
#     svd_gap_budget    Ablation: gap budget without the projection.

# For every (seed, arm): fit the encoder, train the same MLP decoder on clean
# train embeddings, report validation R2 and, unless --no-eval-attack, R2 under
# plain-PGD and spectral-gap attacks on the validation windows, and R2 on
# corrupted recordings (Poisson resampling, 5-50% lost units, loss of the 20% most
# active units, gain drift). Results are saved
# after every arm in a new, unique run directory; no checkpoints are stored.
# """

# import argparse
# import csv
# import gc
# import inspect
# import json
# import os
# import random
# import sys
# import tempfile
# import time
# from datetime import datetime, timezone
# from pathlib import Path

# import numpy as np
# import torch
# from sklearn.metrics import r2_score
# from torch import nn

# # Prevent the local fork import from generating __pycache__ files.
# sys.dont_write_bytecode = True

# ROOT = Path(__file__).resolve().parent
# FORK_DIR = ROOT / "Acorn-SVD"
# DEFAULT_DATA = Path("/data/hossein/mm_project/perich_data_valid_final_raw/C-CO12.npz")

# EPSILON = 3.0  # Training budget (cap for the spectral arms); --train-epsilon overrides.
# EPS_STEPS = 10
# ALPHA_FRACTION = 1 / 5  # Step size as a fraction of the budget: 0.6 for eps=3.
# ATTACK_NORM = "l2"
# UNCAPPED_EPSILON = 1e3  # Effectively no cap: the spectral gap sets the budget.

# # Spectral-gap settings, shared by all spectral arms and the evaluation attack.
# GAMMA = 0.9
# K_MIN, K_MAX = 1, 8
# CENTER = True
# PRESERVE_MEAN = False
# DIAGNOSTICS_EVERY = 50

# LATENT_DIM = 64
# HIDDEN = 64
# BATCH_SIZE = 2048
# MAX_ITER = 3000
# TEMPERATURE = 0.4
# ARCH = "offset36-model-more-dropout"
# DEVICE = "cuda_if_available"

# # Decoder: identical to the previous runner.
# MLP_EPOCHS = 2500
# MLP_HIDDEN = 64
# MLP_DROP = 0.4
# MLP_LR = 1e-3
# EVAL_BATCH_SIZE = 8192

# # Held-out attack on the frozen encoder + decoder (prediction MSE).
# EVAL_EPSILON = 3.0  # Fixed across runs so that robustness is comparable.
# EVAL_STEPS = 20
# EVAL_ALPHA = 2.5 * EVAL_EPSILON / EVAL_STEPS
# EVAL_ATTACK_BATCH = 2048
# EVAL_SEED = 1234

# # Recording corruptions of the validation data (Poisson resampling, lost units,
# # per-neuron gain drift), averaged over a few random draws.
# NATURAL_DRAWS = 5
# NATURAL_SEED = 4321
# DROPOUT_CURVE = (0.05, 0.2, 0.4, 0.5)  # in addition to 10% and 30%

# ARMS = {
#     "clean": dict(training_mode="standard"),
#     "pgd": dict(adv_spectral_mode="none"),
#     "svd_gap": dict(adv_spectral_mode="gap"),
#     "svd_gap_uncapped": dict(adv_spectral_mode="gap",
#                              adv_epsilon=UNCAPPED_EPSILON,
#                              adv_alpha=UNCAPPED_EPSILON / 5),
#     "svd_tangent": dict(adv_spectral_mode="tangent"),
#     "svd_split": dict(adv_spectral_mode="split"),
#     "svd_tangent_budget": dict(adv_spectral_mode="tangent_budget"),
#     "svd_subspace": dict(adv_spectral_mode="subspace"),
#     "svd_subspace_cert": dict(adv_spectral_mode="subspace_cert"),
#     "svd_gap_budget": dict(adv_spectral_mode="gap_budget"),
# }
# DEFAULT_ARMS = ("clean", "pgd", "svd_subspace_cert", "svd_subspace", "svd_tangent", "svd_split")


# def seed_all(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True


# def load_fork():
#     if not (FORK_DIR / "cebra" / "__init__.py").is_file():
#         raise FileNotFoundError(f"Expected Acorn-SVD/ next to this script: {FORK_DIR}")
#     # Avoid accidentally using an installed/unrelated version of CEBRA.
#     for key in list(sys.modules):
#         if key == "cebra" or key.startswith("cebra."):
#             del sys.modules[key]
#     sys.path.insert(0, str(FORK_DIR.resolve()))
#     import cebra
#     import cebra.solver.spectral_gap
#     if not Path(cebra.__file__).resolve().is_relative_to(FORK_DIR.resolve()):
#         raise RuntimeError(f"Incorrect CEBRA package imported: {cebra.__file__}")
#     params = inspect.signature(cebra.CEBRA.__init__).parameters
#     required = {"adv_spectral_mode", "adv_spectral_gamma", "adv_diagnostics_every"}
#     missing = required - params.keys()
#     if missing:
#         raise RuntimeError(f"Acorn-SVD fork lacks expected parameters: {sorted(missing)}")
#     modes = set(cebra.solver.spectral_gap.ATTACK_MODES)
#     if "subspace_cert" not in modes:
#         raise RuntimeError("The sibling Acorn-SVD/ fork is outdated: 'subspace_cert' is missing")
#     return cebra


# def load_data(path: Path):
#     if not path.is_file():
#         raise FileNotFoundError(f"Session file not found: {path}")
#     with np.load(path, allow_pickle=False) as f:
#         x_train, x_valid, y_train, y_valid = (
#             np.asarray(f[key], dtype=np.float32)
#             for key in ("train_data", "valid_data", "train_label", "valid_label"))
#     if y_train.ndim == 1:
#         y_train = y_train[:, None]
#     if y_valid.ndim == 1:
#         y_valid = y_valid[:, None]
#     for key, arr in (("train_data", x_train), ("valid_data", x_valid),
#                      ("train_label", y_train), ("valid_label", y_valid)):
#         if arr.ndim != 2 or not all(arr.shape) or not np.isfinite(arr).all():
#             raise ValueError(f"Invalid {key} shape or NaN/Inf: {arr.shape}")
#     if x_train.shape[1] != x_valid.shape[1] or y_train.shape[1] != y_valid.shape[1]:
#         raise ValueError("Train/validation feature or label dimensions differ")
#     if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
#         raise ValueError("Number of labels and neural samples must match")
#     if min(len(x_train), len(x_valid)) < 36:
#         raise ValueError("offset36 encoder requires >=36 time bins per split")
#     return tuple(np.ascontiguousarray(a) for a in (x_train, x_valid, y_train, y_valid))


# def encoder_config(arm: str, iterations: int, epsilon: float = EPSILON, gamma: float = GAMMA,
#                    k_min: int = K_MIN, k_max: int = K_MAX) -> dict:
#     config = dict(
#         model_architecture=ARCH,
#         batch_size=BATCH_SIZE,
#         temperature=TEMPERATURE,
#         time_offsets=1,
#         max_iterations=iterations,
#         output_dimension=LATENT_DIM,
#         num_hidden_units=HIDDEN,
#         learning_rate=3e-4,
#         temperature_mode="constant",
#         training_mode="adversarial",
#         adv_epsilon=epsilon,
#         adv_alpha=epsilon * ALPHA_FRACTION,
#         adv_steps=EPS_STEPS,
#         attack_norm=ATTACK_NORM,
#         adv_spectral_gamma=gamma,
#         adv_spectral_k_min=k_min,
#         adv_spectral_k_max=k_max,
#         adv_spectral_center=CENTER,
#         adv_spectral_preserve_mean=PRESERVE_MEAN,
#         adv_attack_eval_mode=False,  # Same attack protocol as the previous project.
#         adv_diagnostics_every=DIAGNOSTICS_EVERY,
#         device=DEVICE,
#         verbose=True,  # CEBRA training progress and iteration/loss logs.
#     )
#     config.update(ARMS[arm])
#     return config


# class TwoLayerMLP(nn.Module):

#     def __init__(self, dim, out):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, MLP_HIDDEN), nn.LayerNorm(MLP_HIDDEN),
#             nn.ReLU(), nn.Dropout(MLP_DROP), nn.Linear(MLP_HIDDEN, out),
#         )

#     def forward(self, x):
#         return self.net(x)


# def train_decoder(z_train, y_train, seed, device):
#     seed_all(seed)  # Same decoder seed in all arms.
#     decoder = TwoLayerMLP(z_train.shape[1], y_train.shape[1]).to(device)
#     optimizer = torch.optim.Adam(decoder.parameters(), lr=MLP_LR)
#     criterion = nn.MSELoss()
#     z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
#     y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
#     decoder.train()
#     print(f"[DECODER] Start {MLP_EPOCHS} epochs; all {y_train.shape[1]} label columns", flush=True)
#     decoder_start = time.perf_counter()
#     for epoch in range(MLP_EPOCHS):
#         optimizer.zero_grad(set_to_none=True)
#         loss = criterion(decoder(z), y)
#         if not torch.isfinite(loss).item():
#             raise FloatingPointError(f"Nonfinite decoder loss at epoch {epoch + 1}")
#         loss.backward()
#         optimizer.step()
#         if (epoch + 1) % 250 == 0 or epoch + 1 == MLP_EPOCHS:
#             print(f"[DECODER] epoch {epoch + 1}/{MLP_EPOCHS} | train_MSE={loss.item():.6f} "
#                   f"| elapsed={time.perf_counter() - decoder_start:.1f}s", flush=True)
#     return decoder


# def predict_decoder(decoder, embeddings):
#     decoder.eval()  # Disable Dropout for R2 evaluation.
#     device = next(decoder.parameters()).device
#     preds = []
#     with torch.inference_mode():
#         for start in range(0, len(embeddings), EVAL_BATCH_SIZE):
#             z = torch.as_tensor(embeddings[start:start + EVAL_BATCH_SIZE],
#                                 dtype=torch.float32, device=device)
#             preds.append(decoder(z).cpu().numpy())
#     return np.concatenate(preds, axis=0)


# def mean_r2(y_true, y_pred):
#     per_label = np.atleast_1d(r2_score(y_true, y_pred, multioutput="raw_values"))
#     if not np.isfinite(per_label).all():
#         raise ValueError(f"Nonfinite R2 score: {per_label}")
#     return float(per_label.mean()), [float(v) for v in per_label]


# def validation_windows(x_valid, y_valid, offset):
#     """All validation windows that lie fully inside the recording, as in training."""
#     length = offset.left + offset.right
#     windows = torch.as_tensor(x_valid).unfold(0, length, 1)  # (M, neurons, length)
#     labels = y_valid[offset.left:offset.left + len(windows)]
#     return windows.contiguous(), labels


# def natural_corruptions(x, rng):
#     """Recording perturbations of validation spike counts: noise, lost units, gain drift."""
#     yield "poisson", rng.poisson(np.clip(x, 0, None)).astype(np.float32)
#     for fraction in (0.1, 0.3):
#         keep = np.ones(x.shape[1], dtype=np.float32)
#         keep[rng.choice(x.shape[1], int(round(fraction * x.shape[1])), replace=False)] = 0
#         yield f"dropout{int(fraction * 100)}", x * keep
#     yield "gain30", x * np.exp(rng.normal(0, 0.3, x.shape[1])).astype(np.float32)


# def unit_loss_curve(x, rng):
#     """More unit-loss severities, and loss of the most active units (structured)."""
#     for fraction in DROPOUT_CURVE:
#         keep = np.ones(x.shape[1], dtype=np.float32)
#         keep[rng.choice(x.shape[1], int(round(fraction * x.shape[1])), replace=False)] = 0
#         yield f"dropout{int(fraction * 100)}", x * keep
#     keep = np.ones(x.shape[1], dtype=np.float32)
#     keep[np.argsort(-x.mean(0))[:int(round(0.2 * x.shape[1]))]] = 0
#     yield "droptop20", x * keep


# def natural_robustness(encoder, decoder, x_valid, y_valid):
#     """Validation R2 of the frozen encoder + decoder on corrupted recordings."""
#     rng = np.random.default_rng(NATURAL_SEED)
#     # A separate stream keeps the draws of the original corruptions unchanged.
#     curve_rng = np.random.default_rng(NATURAL_SEED + 1)
#     scores = {}
#     for _ in range(NATURAL_DRAWS):
#         for name, corrupted in list(natural_corruptions(x_valid, rng)) + list(
#                 unit_loss_curve(x_valid, curve_rng)):
#             z = np.asarray(encoder.transform(corrupted), dtype=np.float32)
#             scores.setdefault(name, []).append(mean_r2(y_valid, predict_decoder(decoder, z))[0])
#     return {f"r2_{name}": float(np.mean(values)) for name, values in scores.items()}


# def attack_eval_configs(spectral_gap):
#     shared = dict(epsilon=EVAL_EPSILON, alpha=EVAL_ALPHA, steps=EVAL_STEPS, norm="l2")
#     return {
#         "pgd": spectral_gap.AttackConfig(mode="none", **shared),
#         "svd_gap": spectral_gap.AttackConfig(mode="gap", gamma=GAMMA, k_min=K_MIN, k_max=K_MAX,
#                                              center=CENTER, **shared),
#         "svd_tangent": spectral_gap.AttackConfig(mode="tangent", gamma=GAMMA, k_min=K_MIN,
#                                                  k_max=K_MAX, center=CENTER, **shared),
#     }


# def robustness(cebra, encoder, decoder, x_valid, y_valid, z_valid):
#     """Validation R2 of the frozen encoder + decoder under held-out attacks."""
#     spectral_gap = cebra.solver.spectral_gap
#     model = encoder.solver_.model
#     device = next(model.parameters()).device
#     model.eval()
#     decoder.eval()
#     windows, labels = validation_windows(x_valid, y_valid, model.get_offset())
#     offset = model.get_offset()
#     # The per-window model must reproduce the transform used for clean R2.
#     with torch.no_grad():
#         probe = model(windows[:256].to(device)).cpu().numpy()
#     mismatch = float(np.abs(probe - z_valid[offset.left:offset.left + len(probe)]).max())
#     if mismatch > 1e-3:
#         raise RuntimeError(f"Window embedding does not match transform (max diff {mismatch})")

#     out = {"window_mismatch": mismatch, "num_windows": int(len(windows))}
#     for name, config in [("clean", None)] + list(attack_eval_configs(spectral_gap).items()):
#         seed_all(EVAL_SEED)
#         preds, budgets = [], []
#         for start in range(0, len(windows), EVAL_ATTACK_BATCH):
#             batch = windows[start:start + EVAL_ATTACK_BATCH].to(device)
#             target = torch.as_tensor(labels[start:start + len(batch)], device=device)
#             if config is not None:
#                 frame = spectral_gap.spectral_frame(batch, config) if config.spectral else None
#                 if frame is not None:
#                     budgets.append(frame.epsilon.cpu())

#                 def loss_fn(reference, target=target):
#                     return (decoder(model(reference)) - target).square().mean()

#                 batch = spectral_gap.attack(batch, loss_fn, config, frame)
#             with torch.no_grad():
#                 preds.append(decoder(model(batch)).cpu().numpy())
#         r2, per_label = mean_r2(labels, np.concatenate(preds))
#         out[f"r2_{name}"] = r2
#         out[f"r2_{name}_per_label"] = per_label
#         if budgets:
#             out[f"eval_eps_{name}_mean"] = float(torch.cat(budgets).mean())
#     return out


# def attack_statistics(solver) -> dict:
#     """Training-time attack statistics averaged over iterations."""
#     log = solver.log
#     stats = {}
#     for key in ("adv_loss", "adv_delta_norm", "adv_eps_mean", "adv_eps_min",
#                 "adv_gap_limited", "adv_k_mean", "adv_gap_mean", "adv_tangent_norm",
#                 "adv_complement_norm", "adv_tangent_eps_mean"):
#         if log.get(key):
#             stats[f"train_{key}"] = float(np.mean(log[key]))
#     diagnostics = log.get("adv_diagnostics", [])
#     for key in ("sv_err_max", "left_subspace_err_max", "right_subspace_err_max",
#                 "spectral_drift_mean", "delta_over_gap_max", "gap_kept_min",
#                 "collapse_fraction", "mode_exchange_fraction", "tangent_energy_fraction",
#                 "tangent_chance_fraction"):
#         values = [d[key] for d in diagnostics if np.isfinite(d.get(key, np.nan))]
#         if values:
#             reduce = np.max if key.endswith("_max") else np.min if key.endswith("_min") else np.mean
#             stats[f"diag_{key}"] = float(reduce(values))
#     return stats


# def run_model(cebra, arm, seed, arrays, iterations, eval_attack, epsilon=EPSILON,
#               spectral=None) -> dict:
#     x_train, x_valid, y_train, y_valid = arrays
#     encoder = decoder = z_train = z_valid = None
#     arm_start = time.perf_counter()
#     try:
#         seed_all(seed)  # Same initialization and sampling seed in each arm.
#         print(f"\n{'=' * 75}", flush=True)
#         print(f"[START] {arm} | seed={seed} | eps={epsilon} | {iterations} CEBRA iterations",
#               flush=True)
#         encoder = cebra.CEBRA(**encoder_config(arm, iterations, epsilon, **(spectral or {})))
#         encoder.fit(x_train, y_train)
#         encoder_seconds = time.perf_counter() - arm_start
#         print(f"[ENCODER] {arm}: {encoder_seconds / 60:.1f} min "
#               f"({encoder_seconds / iterations:.3f} s/iteration)", flush=True)

#         z_train = np.asarray(encoder.transform(x_train), dtype=np.float32)
#         z_valid = np.asarray(encoder.transform(x_valid), dtype=np.float32)
#         if (len(z_train) != len(x_train) or len(z_valid) != len(x_valid)
#                 or not np.isfinite(z_train).all() or not np.isfinite(z_valid).all()):
#             raise ValueError(f"Bad embeddings or alignment for arm={arm}")

#         decoder_start = time.perf_counter()
#         decoder = train_decoder(z_train, y_train, seed + 100_000,
#                                 torch.device(encoder.device_))
#         decoder_seconds = time.perf_counter() - decoder_start
#         predictions = predict_decoder(decoder, z_valid)
#         if predictions.shape != y_valid.shape or not np.isfinite(predictions).all():
#             raise ValueError(f"Bad decoder prediction for arm={arm}")
#         r2, per_label = mean_r2(y_valid, predictions)
#         row = {
#             "model": arm, "seed": seed, "epsilon": float(encoder.adv_epsilon),
#             "gamma": float(encoder.adv_spectral_gamma),
#             "k_range": [encoder.adv_spectral_k_min, encoder.adv_spectral_k_max],
#             "spectral_mode": encoder.adv_spectral_mode
#             if encoder.training_mode == "adversarial" else "clean",
#             "valid_mean_r2": r2, "valid_r2_per_label": per_label,
#             "encoder_seconds": float(encoder_seconds),
#             "seconds_per_iteration": float(encoder_seconds / iterations),
#             "decoder_seconds": float(decoder_seconds),
#         }
#         print(f"[RESULT] {arm} seed={seed}: validation mean R2 = {r2:.6f} | per label = {per_label}",
#               flush=True)
#         if encoder.training_mode == "adversarial":
#             row.update(attack_statistics(encoder.solver_))
#         row.update(natural_robustness(encoder, decoder, x_valid, y_valid))
#         if eval_attack:
#             eval_start = time.perf_counter()
#             row.update(robustness(cebra, encoder, decoder, x_valid, y_valid, z_valid))
#             row["eval_attack_seconds"] = float(time.perf_counter() - eval_start)
#             print(f"[ROBUST] {arm} seed={seed}: R2 clean windows = {row['r2_clean']:.4f} | "
#                   f"PGD = {row['r2_pgd']:.4f} | SVD-gap = {row['r2_svd_gap']:.4f} | "
#                   f"SVD-tangent = {row['r2_svd_tangent']:.4f}", flush=True)
#         row["total_seconds"] = float(time.perf_counter() - arm_start)
#         return row
#     finally:
#         del encoder, decoder, z_train, z_valid
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()


# SCALAR_COLUMNS = (
#     "model", "seed", "spectral_mode", "epsilon", "valid_mean_r2", "r2_clean", "r2_pgd",
#     "r2_svd_gap", "eval_eps_svd_gap_mean", "r2_svd_tangent", "r2_poisson", "r2_dropout10", "r2_dropout30",
#     "r2_gain30", "train_adv_eps_mean", "train_adv_gap_limited",
#     "train_adv_k_mean", "train_adv_delta_norm", "train_adv_loss", "diag_sv_err_max",
#     "diag_left_subspace_err_max", "diag_right_subspace_err_max", "diag_spectral_drift_mean",
#     "diag_delta_over_gap_max", "diag_gap_kept_min", "diag_collapse_fraction",
#     "diag_mode_exchange_fraction", "diag_tangent_energy_fraction",
#     "diag_tangent_chance_fraction", "train_adv_tangent_norm", "train_adv_complement_norm",
#     "r2_dropout5", "r2_dropout20", "r2_dropout40", "r2_dropout50", "r2_droptop20",
#     "seconds_per_iteration", "encoder_seconds", "decoder_seconds",
#     "total_seconds",
# )


# def save_results(out: Path, results: list):
#     """Persist all completed arms atomically, even if a later arm is slow/fails."""
#     json_path, temp_json = out / "results_so_far.json", out / ".results_so_far.json.tmp"
#     temp_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
#     os.replace(temp_json, json_path)
#     num_labels = len(results[0]["valid_r2_per_label"])
#     csv_path, temp_csv = out / "results_so_far.csv", out / ".results_so_far.csv.tmp"
#     with temp_csv.open("w", newline="", encoding="utf-8") as handle:
#         writer = csv.writer(handle)
#         writer.writerow(list(SCALAR_COLUMNS) + [f"r2_label_{i}" for i in range(num_labels)])
#         for row in results:
#             writer.writerow([row.get(key, "") for key in SCALAR_COLUMNS] +
#                             row["valid_r2_per_label"])
#     os.replace(temp_csv, csv_path)


# def arm_label(row: dict) -> str:
#     return row["model"] if row["model"] == "clean" else f"{row['model']}@{row['epsilon']:g}"


# def summarize(results: list) -> list:
#     summary = []
#     for arm in dict.fromkeys(arm_label(row) for row in results):
#         rows = [row for row in results if arm_label(row) == arm]
#         entry = {"model": arm, "num_seeds": len(rows)}
#         for key in ("valid_mean_r2", "r2_pgd", "r2_svd_gap", "r2_svd_tangent", "r2_poisson",
#                     "r2_dropout10", "r2_dropout30", "r2_dropout50", "r2_droptop20", "r2_gain30",
#                     "seconds_per_iteration"):
#             values = [row[key] for row in rows if key in row]
#             if values:
#                 entry[f"{key}_mean"] = float(np.mean(values))
#                 entry[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
#         summary.append(entry)
#     return summary


# def main():
#     parser = argparse.ArgumentParser(description=__doc__,
#                                      formatter_class=argparse.RawDescriptionHelpFormatter)
#     parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
#     parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
#     parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=list(ARMS))
#     parser.add_argument("--iterations", type=int, default=MAX_ITER)
#     parser.add_argument("--train-epsilon", type=float, nargs="+", default=[EPSILON],
#                         help="Training budget(s) of the pgd/svd arms (step = budget/5); several "
#                              "values give a budget sweep. The evaluation attacks always use eps=3.")
#     parser.add_argument("--gamma", type=float, default=GAMMA,
#                         help="Training gap fraction; evaluation attacks keep gamma=0.9.")
#     parser.add_argument("--k-min", type=int, default=K_MIN)
#     parser.add_argument("--k-max", type=int, default=K_MAX)
#     parser.add_argument("--out", type=Path, default=ROOT / "ACORN_SVD_RESULTS")
#     parser.add_argument("--no-eval-attack", action="store_true",
#                         help="Skip the held-out attacked-R2 evaluation.")
#     args = parser.parse_args()

#     # mkdtemp uses atomic creation; runs cannot overwrite one another.
#     args.out.mkdir(parents=True, exist_ok=True)
#     stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
#     out = Path(tempfile.mkdtemp(prefix=f"{args.data.stem}_{stamp}_", dir=args.out))
#     print(f"[START] New experiment: {args.data.stem} | output: {out}", flush=True)
#     print(f"[START] Arms: {args.arms} | seeds: {args.seeds} | eps: {args.train_epsilon}",
#           flush=True)
#     spectral = dict(gamma=args.gamma, k_min=args.k_min, k_max=args.k_max)
#     cebra = load_fork()
#     print(f"[FORK] {cebra.__file__}", flush=True)
#     arrays = load_data(args.data)
#     print(f"[DATA] Train X={arrays[0].shape}, Y={arrays[2].shape}", flush=True)
#     print(f"[DATA] Valid X={arrays[1].shape}, Y={arrays[3].shape}", flush=True)

#     config = {
#         "data_path": str(args.data), "fork": str(FORK_DIR.resolve()), "cebra_import": str(cebra.__file__),
#         "arms": args.arms, "seeds": args.seeds, "iterations": args.iterations,
#         "train_epsilons": args.train_epsilon, "corruption_draws": NATURAL_DRAWS,
#         "dropout_fractions": sorted(set(DROPOUT_CURVE + (0.10, 0.30))),
#         "encoder_configs": {f"{arm}@{eps:g}": encoder_config(arm, args.iterations, eps,
#                                                              spectral["gamma"],
#                                                              spectral["k_min"],
#                                                              spectral["k_max"])
#                             for i, eps in enumerate(args.train_epsilon) for arm in args.arms
#                             if i == 0 or arm not in ("clean", "svd_gap_uncapped")},
#         "decoder": {"architecture": "Linear -> LayerNorm -> ReLU -> Dropout -> Linear",
#                     "epochs": MLP_EPOCHS, "hidden": MLP_HIDDEN, "dropout": MLP_DROP,
#                     "learning_rate": MLP_LR},
#         "eval_attack": None if args.no_eval_attack else {
#             name: vars(c) for name, c in attack_eval_configs(cebra.solver.spectral_gap).items()},
#         "shapes": {"x_train": arrays[0].shape, "x_valid": arrays[1].shape,
#                    "y_train": arrays[2].shape, "y_valid": arrays[3].shape},
#     }
#     (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

#     # Clean training and the uncapped arm do not depend on the training budget.
#     jobs = [(seed, eps, arm) for seed in args.seeds for i, eps in enumerate(args.train_epsilon)
#             for arm in args.arms if i == 0 or arm not in ("clean", "svd_gap_uncapped")]
#     results = []
#     try:
#         for seed, eps, arm in jobs:  # Seed-major: every arm of a seed finishes first.
#             results.append(run_model(cebra, arm, seed, arrays, args.iterations,
#                                      not args.no_eval_attack, eps, spectral))
#             save_results(out, results)  # Save right after each completed arm.
#             print(f"[SAVED] {len(results)}/{len(jobs)} results -> {out / 'results_so_far.csv'}",
#                   flush=True)
#     except BaseException as exc:
#         print(f"[INTERRUPTED] {type(exc).__name__}: {exc}", flush=True)
#         print(f"[RECOVERY] Completed arms' results (if any) remain in: {out}", flush=True)
#         raise

#     os.replace(out / "results_so_far.csv", out / "results.csv")
#     os.replace(out / "results_so_far.json", out / "results.json")
#     summary = summarize(results)
#     (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
#     print("\n" + "=" * 75, flush=True)
#     print("SUMMARY (mean ± std over seeds)", flush=True)
#     for entry in summary:
#         text = " | ".join(
#             f"{key[:-5]}={entry[key]:.4f}±{entry[key[:-5] + '_std']:.4f}"
#             for key in entry if key.endswith("_mean"))
#         print(f"{entry['model']:22s} n={entry['num_seeds']} | {text}", flush=True)
#     print(f"[COMPLETE] Results saved in: {out}", flush=True)


# if __name__ == "__main__":
#     main()




# # """C-CO12: clean vs PGD vs spectral-gap adversarial CEBRA (Acorn-SVD fork).

# # Place this runner beside the Acorn-SVD/ folder; it imports the sibling fork:
# #     python -u run_acorn_svd.py --data /path/to/C-CO12.npz

# # Arms (--arms):
# #     clean             CEBRA without adversarial updates.
# #     pgd               Previous project's PGD on the reference (L2, eps=3), bit-exact.
# #     svd_gap           Spectral-gap subspace attack, per-window budget capped at eps=3.
# #     svd_gap_uncapped  Same, budget limited only by the spectral gap.
# #     svd_tangent       On-manifold: tangent space of the top-k structure, half-gap budget.
# #     svd_split         Both: complement (eps=3) + tangent (half-gap), joint certificate.
# #     svd_tangent_budget Ablation: half-gap budget without the projection.
# #     svd_subspace      Ablation: subspace projection with the fixed budget eps=3.
# #     svd_gap_budget    Ablation: gap budget without the projection.

# # For every (seed, arm): fit the encoder, train the same MLP decoder on clean
# # train embeddings, report validation R2 and, unless --no-eval-attack, R2 under
# # plain-PGD and spectral-gap attacks on the validation windows, and R2 on
# # corrupted recordings (Poisson resampling, 10%/30% lost units, gain drift). Results are saved
# # after every arm in a new, unique run directory; no checkpoints are stored.
# # """

# # import argparse
# # import csv
# # import gc
# # import inspect
# # import json
# # import os
# # import random
# # import sys
# # import tempfile
# # import time
# # from datetime import datetime, timezone
# # from pathlib import Path

# # import numpy as np
# # import torch
# # from sklearn.metrics import r2_score
# # from torch import nn

# # # Prevent the local fork import from generating __pycache__ files.
# # sys.dont_write_bytecode = True

# # ROOT = Path(__file__).resolve().parent
# # FORK_DIR = ROOT / "Acorn-SVD"
# # DEFAULT_DATA = Path("/data/hossein/mm_project/perich_data_valid_final_raw/C-CO12.npz")

# # EPSILON = 3.0  # Training budget (cap for the spectral arms); --train-epsilon overrides.
# # EPS_STEPS = 10
# # ALPHA_FRACTION = 1 / 5  # Step size as a fraction of the budget: 0.6 for eps=3.
# # ATTACK_NORM = "l2"
# # UNCAPPED_EPSILON = 1e3  # Effectively no cap: the spectral gap sets the budget.

# # # Spectral-gap settings, shared by all spectral arms and the evaluation attack.
# # GAMMA = 0.9
# # K_MIN, K_MAX = 1, 8
# # CENTER = True
# # PRESERVE_MEAN = False
# # DIAGNOSTICS_EVERY = 50

# # LATENT_DIM = 64
# # HIDDEN = 64
# # BATCH_SIZE = 2048
# # MAX_ITER = 3000
# # TEMPERATURE = 0.4
# # ARCH = "offset36-model-more-dropout"
# # DEVICE = "cuda_if_available"

# # # Decoder: identical to the previous runner.
# # MLP_EPOCHS = 2500
# # MLP_HIDDEN = 64
# # MLP_DROP = 0.4
# # MLP_LR = 1e-3
# # EVAL_BATCH_SIZE = 8192

# # # Held-out attack on the frozen encoder + decoder (prediction MSE).
# # EVAL_EPSILON = 3.0  # Fixed across runs so that robustness is comparable.
# # EVAL_STEPS = 20
# # EVAL_ALPHA = 2.5 * EVAL_EPSILON / EVAL_STEPS
# # EVAL_ATTACK_BATCH = 2048
# # EVAL_SEED = 1234

# # # Recording corruptions of the validation data (Poisson resampling, lost units,
# # # per-neuron gain drift), averaged over a few random draws.
# # NATURAL_DRAWS = 5
# # NATURAL_SEED = 4321

# # ARMS = {
# #     "clean": dict(training_mode="standard"),
# #     "pgd": dict(adv_spectral_mode="none"),
# #     "svd_gap": dict(adv_spectral_mode="gap"),
# #     "svd_gap_uncapped": dict(adv_spectral_mode="gap",
# #                              adv_epsilon=UNCAPPED_EPSILON,
# #                              adv_alpha=UNCAPPED_EPSILON / 5),
# #     "svd_tangent": dict(adv_spectral_mode="tangent"),
# #     "svd_split": dict(adv_spectral_mode="split"),
# #     "svd_tangent_budget": dict(adv_spectral_mode="tangent_budget"),
# #     "svd_subspace": dict(adv_spectral_mode="subspace"),
# #     "svd_gap_budget": dict(adv_spectral_mode="gap_budget"),
# # }
# # DEFAULT_ARMS = ("clean", "pgd", "svd_gap", "svd_tangent", "svd_split", "svd_gap_budget",
# #                 "svd_tangent_budget", "svd_subspace")


# # def seed_all(seed: int) -> None:
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(seed)
# #     torch.backends.cudnn.benchmark = False
# #     torch.backends.cudnn.deterministic = True


# # def load_fork():
# #     if not (FORK_DIR / "cebra" / "__init__.py").is_file():
# #         raise FileNotFoundError(f"Expected Acorn-SVD/ next to this script: {FORK_DIR}")
# #     # Avoid accidentally using an installed/unrelated version of CEBRA.
# #     for key in list(sys.modules):
# #         if key == "cebra" or key.startswith("cebra."):
# #             del sys.modules[key]
# #     sys.path.insert(0, str(FORK_DIR.resolve()))
# #     import cebra
# #     import cebra.solver.spectral_gap
# #     if not Path(cebra.__file__).resolve().is_relative_to(FORK_DIR.resolve()):
# #         raise RuntimeError(f"Incorrect CEBRA package imported: {cebra.__file__}")
# #     params = inspect.signature(cebra.CEBRA.__init__).parameters
# #     required = {"adv_spectral_mode", "adv_spectral_gamma", "adv_diagnostics_every"}
# #     missing = required - params.keys()
# #     if missing:
# #         raise RuntimeError(f"Acorn-SVD fork lacks expected parameters: {sorted(missing)}")
# #     return cebra


# # def load_data(path: Path):
# #     if not path.is_file():
# #         raise FileNotFoundError(f"Session file not found: {path}")
# #     with np.load(path, allow_pickle=False) as f:
# #         x_train, x_valid, y_train, y_valid = (
# #             np.asarray(f[key], dtype=np.float32)
# #             for key in ("train_data", "valid_data", "train_label", "valid_label"))
# #     if y_train.ndim == 1:
# #         y_train = y_train[:, None]
# #     if y_valid.ndim == 1:
# #         y_valid = y_valid[:, None]
# #     for key, arr in (("train_data", x_train), ("valid_data", x_valid),
# #                      ("train_label", y_train), ("valid_label", y_valid)):
# #         if arr.ndim != 2 or not all(arr.shape) or not np.isfinite(arr).all():
# #             raise ValueError(f"Invalid {key} shape or NaN/Inf: {arr.shape}")
# #     if x_train.shape[1] != x_valid.shape[1] or y_train.shape[1] != y_valid.shape[1]:
# #         raise ValueError("Train/validation feature or label dimensions differ")
# #     if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
# #         raise ValueError("Number of labels and neural samples must match")
# #     if min(len(x_train), len(x_valid)) < 36:
# #         raise ValueError("offset36 encoder requires >=36 time bins per split")
# #     return tuple(np.ascontiguousarray(a) for a in (x_train, x_valid, y_train, y_valid))


# # def encoder_config(arm: str, iterations: int, epsilon: float = EPSILON, gamma: float = GAMMA,
# #                    k_min: int = K_MIN, k_max: int = K_MAX) -> dict:
# #     config = dict(
# #         model_architecture=ARCH,
# #         batch_size=BATCH_SIZE,
# #         temperature=TEMPERATURE,
# #         time_offsets=1,
# #         max_iterations=iterations,
# #         output_dimension=LATENT_DIM,
# #         num_hidden_units=HIDDEN,
# #         learning_rate=3e-4,
# #         temperature_mode="constant",
# #         training_mode="adversarial",
# #         adv_epsilon=epsilon,
# #         adv_alpha=epsilon * ALPHA_FRACTION,
# #         adv_steps=EPS_STEPS,
# #         attack_norm=ATTACK_NORM,
# #         adv_spectral_gamma=gamma,
# #         adv_spectral_k_min=k_min,
# #         adv_spectral_k_max=k_max,
# #         adv_spectral_center=CENTER,
# #         adv_spectral_preserve_mean=PRESERVE_MEAN,
# #         adv_attack_eval_mode=False,  # Same attack protocol as the previous project.
# #         adv_diagnostics_every=DIAGNOSTICS_EVERY,
# #         device=DEVICE,
# #         verbose=True,  # CEBRA training progress and iteration/loss logs.
# #     )
# #     config.update(ARMS[arm])
# #     return config


# # class TwoLayerMLP(nn.Module):

# #     def __init__(self, dim, out):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(dim, MLP_HIDDEN), nn.LayerNorm(MLP_HIDDEN),
# #             nn.ReLU(), nn.Dropout(MLP_DROP), nn.Linear(MLP_HIDDEN, out),
# #         )

# #     def forward(self, x):
# #         return self.net(x)


# # def train_decoder(z_train, y_train, seed, device):
# #     seed_all(seed)  # Same decoder seed in all arms.
# #     decoder = TwoLayerMLP(z_train.shape[1], y_train.shape[1]).to(device)
# #     optimizer = torch.optim.Adam(decoder.parameters(), lr=MLP_LR)
# #     criterion = nn.MSELoss()
# #     z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
# #     y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
# #     decoder.train()
# #     print(f"[DECODER] Start {MLP_EPOCHS} epochs; all {y_train.shape[1]} label columns", flush=True)
# #     decoder_start = time.perf_counter()
# #     for epoch in range(MLP_EPOCHS):
# #         optimizer.zero_grad(set_to_none=True)
# #         loss = criterion(decoder(z), y)
# #         if not torch.isfinite(loss).item():
# #             raise FloatingPointError(f"Nonfinite decoder loss at epoch {epoch + 1}")
# #         loss.backward()
# #         optimizer.step()
# #         if (epoch + 1) % 250 == 0 or epoch + 1 == MLP_EPOCHS:
# #             print(f"[DECODER] epoch {epoch + 1}/{MLP_EPOCHS} | train_MSE={loss.item():.6f} "
# #                   f"| elapsed={time.perf_counter() - decoder_start:.1f}s", flush=True)
# #     return decoder


# # def predict_decoder(decoder, embeddings):
# #     decoder.eval()  # Disable Dropout for R2 evaluation.
# #     device = next(decoder.parameters()).device
# #     preds = []
# #     with torch.inference_mode():
# #         for start in range(0, len(embeddings), EVAL_BATCH_SIZE):
# #             z = torch.as_tensor(embeddings[start:start + EVAL_BATCH_SIZE],
# #                                 dtype=torch.float32, device=device)
# #             preds.append(decoder(z).cpu().numpy())
# #     return np.concatenate(preds, axis=0)


# # def mean_r2(y_true, y_pred):
# #     per_label = np.atleast_1d(r2_score(y_true, y_pred, multioutput="raw_values"))
# #     if not np.isfinite(per_label).all():
# #         raise ValueError(f"Nonfinite R2 score: {per_label}")
# #     return float(per_label.mean()), [float(v) for v in per_label]


# # def validation_windows(x_valid, y_valid, offset):
# #     """All validation windows that lie fully inside the recording, as in training."""
# #     length = offset.left + offset.right
# #     windows = torch.as_tensor(x_valid).unfold(0, length, 1)  # (M, neurons, length)
# #     labels = y_valid[offset.left:offset.left + len(windows)]
# #     return windows.contiguous(), labels


# # def natural_corruptions(x, rng):
# #     """Recording perturbations of validation spike counts: noise, lost units, gain drift."""
# #     yield "poisson", rng.poisson(np.clip(x, 0, None)).astype(np.float32)
# #     for fraction in (0.1, 0.3):
# #         keep = np.ones(x.shape[1], dtype=np.float32)
# #         keep[rng.choice(x.shape[1], int(round(fraction * x.shape[1])), replace=False)] = 0
# #         yield f"dropout{int(fraction * 100)}", x * keep
# #     yield "gain30", x * np.exp(rng.normal(0, 0.3, x.shape[1])).astype(np.float32)


# # def natural_robustness(encoder, decoder, x_valid, y_valid):
# #     """Validation R2 of the frozen encoder + decoder on corrupted recordings."""
# #     rng = np.random.default_rng(NATURAL_SEED)
# #     scores = {}
# #     for _ in range(NATURAL_DRAWS):
# #         for name, corrupted in natural_corruptions(x_valid, rng):
# #             z = np.asarray(encoder.transform(corrupted), dtype=np.float32)
# #             scores.setdefault(name, []).append(mean_r2(y_valid, predict_decoder(decoder, z))[0])
# #     return {f"r2_{name}": float(np.mean(values)) for name, values in scores.items()}


# # def attack_eval_configs(spectral_gap):
# #     shared = dict(epsilon=EVAL_EPSILON, alpha=EVAL_ALPHA, steps=EVAL_STEPS, norm="l2")
# #     return {
# #         "pgd": spectral_gap.AttackConfig(mode="none", **shared),
# #         "svd_gap": spectral_gap.AttackConfig(mode="gap", gamma=GAMMA, k_min=K_MIN, k_max=K_MAX,
# #                                              center=CENTER, **shared),
# #         "svd_tangent": spectral_gap.AttackConfig(mode="tangent", gamma=GAMMA, k_min=K_MIN,
# #                                                  k_max=K_MAX, center=CENTER, **shared),
# #     }


# # def robustness(cebra, encoder, decoder, x_valid, y_valid, z_valid):
# #     """Validation R2 of the frozen encoder + decoder under held-out attacks."""
# #     spectral_gap = cebra.solver.spectral_gap
# #     model = encoder.solver_.model
# #     device = next(model.parameters()).device
# #     model.eval()
# #     decoder.eval()
# #     windows, labels = validation_windows(x_valid, y_valid, model.get_offset())
# #     offset = model.get_offset()
# #     # The per-window model must reproduce the transform used for clean R2.
# #     with torch.no_grad():
# #         probe = model(windows[:256].to(device)).cpu().numpy()
# #     mismatch = float(np.abs(probe - z_valid[offset.left:offset.left + len(probe)]).max())
# #     if mismatch > 1e-3:
# #         raise RuntimeError(f"Window embedding does not match transform (max diff {mismatch})")

# #     out = {"window_mismatch": mismatch, "num_windows": int(len(windows))}
# #     for name, config in [("clean", None)] + list(attack_eval_configs(spectral_gap).items()):
# #         seed_all(EVAL_SEED)
# #         preds, budgets = [], []
# #         for start in range(0, len(windows), EVAL_ATTACK_BATCH):
# #             batch = windows[start:start + EVAL_ATTACK_BATCH].to(device)
# #             target = torch.as_tensor(labels[start:start + len(batch)], device=device)
# #             if config is not None:
# #                 frame = spectral_gap.spectral_frame(batch, config) if config.spectral else None
# #                 if frame is not None:
# #                     budgets.append(frame.epsilon.cpu())

# #                 def loss_fn(reference, target=target):
# #                     return (decoder(model(reference)) - target).square().mean()

# #                 batch = spectral_gap.attack(batch, loss_fn, config, frame)
# #             with torch.no_grad():
# #                 preds.append(decoder(model(batch)).cpu().numpy())
# #         r2, per_label = mean_r2(labels, np.concatenate(preds))
# #         out[f"r2_{name}"] = r2
# #         out[f"r2_{name}_per_label"] = per_label
# #         if budgets:
# #             out[f"eval_eps_{name}_mean"] = float(torch.cat(budgets).mean())
# #     return out


# # def attack_statistics(solver) -> dict:
# #     """Training-time attack statistics averaged over iterations."""
# #     log = solver.log
# #     stats = {}
# #     for key in ("adv_loss", "adv_delta_norm", "adv_eps_mean", "adv_eps_min",
# #                 "adv_gap_limited", "adv_k_mean", "adv_gap_mean", "adv_tangent_norm",
# #                 "adv_complement_norm", "adv_tangent_eps_mean"):
# #         if log.get(key):
# #             stats[f"train_{key}"] = float(np.mean(log[key]))
# #     diagnostics = log.get("adv_diagnostics", [])
# #     for key in ("sv_err_max", "left_subspace_err_max", "right_subspace_err_max",
# #                 "spectral_drift_mean", "delta_over_gap_max", "gap_kept_min",
# #                 "collapse_fraction", "mode_exchange_fraction"):
# #         values = [d[key] for d in diagnostics if np.isfinite(d.get(key, np.nan))]
# #         if values:
# #             reduce = np.max if key.endswith("_max") else np.min if key.endswith("_min") else np.mean
# #             stats[f"diag_{key}"] = float(reduce(values))
# #     return stats


# # def run_model(cebra, arm, seed, arrays, iterations, eval_attack, epsilon=EPSILON,
# #               spectral=None) -> dict:
# #     x_train, x_valid, y_train, y_valid = arrays
# #     encoder = decoder = z_train = z_valid = None
# #     arm_start = time.perf_counter()
# #     try:
# #         seed_all(seed)  # Same initialization and sampling seed in each arm.
# #         print(f"\n{'=' * 75}", flush=True)
# #         print(f"[START] {arm} | seed={seed} | eps={epsilon} | {iterations} CEBRA iterations",
# #               flush=True)
# #         encoder = cebra.CEBRA(**encoder_config(arm, iterations, epsilon, **(spectral or {})))
# #         encoder.fit(x_train, y_train)
# #         encoder_seconds = time.perf_counter() - arm_start
# #         print(f"[ENCODER] {arm}: {encoder_seconds / 60:.1f} min "
# #               f"({encoder_seconds / iterations:.3f} s/iteration)", flush=True)

# #         z_train = np.asarray(encoder.transform(x_train), dtype=np.float32)
# #         z_valid = np.asarray(encoder.transform(x_valid), dtype=np.float32)
# #         if (len(z_train) != len(x_train) or len(z_valid) != len(x_valid)
# #                 or not np.isfinite(z_train).all() or not np.isfinite(z_valid).all()):
# #             raise ValueError(f"Bad embeddings or alignment for arm={arm}")

# #         decoder_start = time.perf_counter()
# #         decoder = train_decoder(z_train, y_train, seed + 100_000,
# #                                 torch.device(encoder.device_))
# #         decoder_seconds = time.perf_counter() - decoder_start
# #         predictions = predict_decoder(decoder, z_valid)
# #         if predictions.shape != y_valid.shape or not np.isfinite(predictions).all():
# #             raise ValueError(f"Bad decoder prediction for arm={arm}")
# #         r2, per_label = mean_r2(y_valid, predictions)
# #         row = {
# #             "model": arm, "seed": seed, "epsilon": float(encoder.adv_epsilon),
# #             "gamma": float(encoder.adv_spectral_gamma),
# #             "k_range": [encoder.adv_spectral_k_min, encoder.adv_spectral_k_max],
# #             "spectral_mode": encoder.adv_spectral_mode
# #             if encoder.training_mode == "adversarial" else "clean",
# #             "valid_mean_r2": r2, "valid_r2_per_label": per_label,
# #             "encoder_seconds": float(encoder_seconds),
# #             "seconds_per_iteration": float(encoder_seconds / iterations),
# #             "decoder_seconds": float(decoder_seconds),
# #         }
# #         print(f"[RESULT] {arm} seed={seed}: validation mean R2 = {r2:.6f} | per label = {per_label}",
# #               flush=True)
# #         if encoder.training_mode == "adversarial":
# #             row.update(attack_statistics(encoder.solver_))
# #         row.update(natural_robustness(encoder, decoder, x_valid, y_valid))
# #         if eval_attack:
# #             eval_start = time.perf_counter()
# #             row.update(robustness(cebra, encoder, decoder, x_valid, y_valid, z_valid))
# #             row["eval_attack_seconds"] = float(time.perf_counter() - eval_start)
# #             print(f"[ROBUST] {arm} seed={seed}: R2 clean windows = {row['r2_clean']:.4f} | "
# #                   f"PGD = {row['r2_pgd']:.4f} | SVD-gap = {row['r2_svd_gap']:.4f} | "
# #                   f"SVD-tangent = {row['r2_svd_tangent']:.4f}", flush=True)
# #         row["total_seconds"] = float(time.perf_counter() - arm_start)
# #         return row
# #     finally:
# #         del encoder, decoder, z_train, z_valid
# #         gc.collect()
# #         if torch.cuda.is_available():
# #             torch.cuda.empty_cache()


# # SCALAR_COLUMNS = (
# #     "model", "seed", "spectral_mode", "epsilon", "valid_mean_r2", "r2_clean", "r2_pgd",
# #     "r2_svd_gap", "eval_eps_svd_gap_mean", "r2_svd_tangent", "r2_poisson", "r2_dropout10", "r2_dropout30",
# #     "r2_gain30", "train_adv_eps_mean", "train_adv_gap_limited",
# #     "train_adv_k_mean", "train_adv_delta_norm", "train_adv_loss", "diag_sv_err_max",
# #     "diag_left_subspace_err_max", "diag_right_subspace_err_max", "diag_spectral_drift_mean",
# #     "diag_delta_over_gap_max", "diag_gap_kept_min", "diag_collapse_fraction",
# #     "diag_mode_exchange_fraction", "train_adv_tangent_norm", "train_adv_complement_norm",
# #     "seconds_per_iteration", "encoder_seconds", "decoder_seconds",
# #     "total_seconds",
# # )


# # def save_results(out: Path, results: list):
# #     """Persist all completed arms atomically, even if a later arm is slow/fails."""
# #     json_path, temp_json = out / "results_so_far.json", out / ".results_so_far.json.tmp"
# #     temp_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
# #     os.replace(temp_json, json_path)
# #     num_labels = len(results[0]["valid_r2_per_label"])
# #     csv_path, temp_csv = out / "results_so_far.csv", out / ".results_so_far.csv.tmp"
# #     with temp_csv.open("w", newline="", encoding="utf-8") as handle:
# #         writer = csv.writer(handle)
# #         writer.writerow(list(SCALAR_COLUMNS) + [f"r2_label_{i}" for i in range(num_labels)])
# #         for row in results:
# #             writer.writerow([row.get(key, "") for key in SCALAR_COLUMNS] +
# #                             row["valid_r2_per_label"])
# #     os.replace(temp_csv, csv_path)


# # def summarize(results: list) -> list:
# #     summary = []
# #     for arm in dict.fromkeys(row["model"] for row in results):
# #         rows = [row for row in results if row["model"] == arm]
# #         entry = {"model": arm, "num_seeds": len(rows)}
# #         for key in ("valid_mean_r2", "r2_pgd", "r2_svd_gap", "r2_svd_tangent", "r2_poisson",
# #                     "r2_dropout10",
# #                     "r2_dropout30", "r2_gain30", "seconds_per_iteration"):
# #             values = [row[key] for row in rows if key in row]
# #             if values:
# #                 entry[f"{key}_mean"] = float(np.mean(values))
# #                 entry[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
# #         summary.append(entry)
# #     return summary


# # def main():
# #     parser = argparse.ArgumentParser(description=__doc__,
# #                                      formatter_class=argparse.RawDescriptionHelpFormatter)
# #     parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
# #     parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
# #     parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=list(ARMS))
# #     parser.add_argument("--iterations", type=int, default=MAX_ITER)
# #     parser.add_argument("--train-epsilon", type=float, default=EPSILON,
# #                         help="Training budget of the pgd/svd arms (step = budget/5). "
# #                              "The evaluation attacks always use eps=3.")
# #     parser.add_argument("--gamma", type=float, default=GAMMA,
# #                         help="Training gap fraction; evaluation attacks keep gamma=0.9.")
# #     parser.add_argument("--k-min", type=int, default=K_MIN)
# #     parser.add_argument("--k-max", type=int, default=K_MAX)
# #     parser.add_argument("--out", type=Path, default=ROOT / "ACORN_SVD_RESULTS")
# #     parser.add_argument("--no-eval-attack", action="store_true",
# #                         help="Skip the held-out attacked-R2 evaluation.")
# #     args = parser.parse_args()

# #     # mkdtemp uses atomic creation; runs cannot overwrite one another.
# #     args.out.mkdir(parents=True, exist_ok=True)
# #     stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
# #     out = Path(tempfile.mkdtemp(prefix=f"{args.data.stem}_{stamp}_", dir=args.out))
# #     print(f"[START] New experiment: {args.data.stem} | output: {out}", flush=True)
# #     print(f"[START] Arms: {args.arms} | seeds: {args.seeds} | eps: {args.train_epsilon}",
# #           flush=True)
# #     spectral = dict(gamma=args.gamma, k_min=args.k_min, k_max=args.k_max)
# #     cebra = load_fork()
# #     print(f"[FORK] {cebra.__file__}", flush=True)
# #     arrays = load_data(args.data)
# #     print(f"[DATA] Train X={arrays[0].shape}, Y={arrays[2].shape}", flush=True)
# #     print(f"[DATA] Valid X={arrays[1].shape}, Y={arrays[3].shape}", flush=True)

# #     config = {
# #         "data_path": str(args.data), "fork": str(FORK_DIR.resolve()), "cebra_import": str(cebra.__file__),
# #         "arms": args.arms, "seeds": args.seeds, "iterations": args.iterations,
# #         "encoder_configs": {arm: encoder_config(arm, args.iterations, args.train_epsilon,
# #                                                 spectral["gamma"], spectral["k_min"],
# #                                                 spectral["k_max"])
# #                             for arm in args.arms},
# #         "decoder": {"architecture": "Linear -> LayerNorm -> ReLU -> Dropout -> Linear",
# #                     "epochs": MLP_EPOCHS, "hidden": MLP_HIDDEN, "dropout": MLP_DROP,
# #                     "learning_rate": MLP_LR},
# #         "eval_attack": None if args.no_eval_attack else {
# #             name: vars(c) for name, c in attack_eval_configs(cebra.solver.spectral_gap).items()},
# #         "shapes": {"x_train": arrays[0].shape, "x_valid": arrays[1].shape,
# #                    "y_train": arrays[2].shape, "y_valid": arrays[3].shape},
# #     }
# #     (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

# #     results = []
# #     try:
# #         for seed in args.seeds:  # Seed-major: every arm of a seed finishes first.
# #             for arm in args.arms:
# #                 results.append(run_model(cebra, arm, seed, arrays, args.iterations,
# #                                          not args.no_eval_attack, args.train_epsilon, spectral))
# #                 save_results(out, results)  # Save right after each completed arm.
# #                 print(f"[SAVED] {len(results)}/{len(args.arms) * len(args.seeds)} results -> "
# #                       f"{out / 'results_so_far.csv'}", flush=True)
# #     except BaseException as exc:
# #         print(f"[INTERRUPTED] {type(exc).__name__}: {exc}", flush=True)
# #         print(f"[RECOVERY] Completed arms' results (if any) remain in: {out}", flush=True)
# #         raise

# #     os.replace(out / "results_so_far.csv", out / "results.csv")
# #     os.replace(out / "results_so_far.json", out / "results.json")
# #     summary = summarize(results)
# #     (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
# #     print("\n" + "=" * 75, flush=True)
# #     print("SUMMARY (mean ± std over seeds)", flush=True)
# #     for entry in summary:
# #         text = " | ".join(
# #             f"{key[:-5]}={entry[key]:.4f}±{entry[key[:-5] + '_std']:.4f}"
# #             for key in entry if key.endswith("_mean"))
# #         print(f"{entry['model']:18s} n={entry['num_seeds']} | {text}", flush=True)
# #     print(f"[COMPLETE] Results saved in: {out}", flush=True)


# # if __name__ == "__main__":
# #     main()


# # # """C-CO12: clean vs PGD vs spectral-gap adversarial CEBRA (Acorn-SVD fork).

# # # Place this runner beside the Acorn-SVD/ directory; run from anywhere:
# # #     python -u run_acorn_svd.py --data /path/to/C-CO12.npz

# # # Arms (--arms):
# # #     clean             CEBRA without adversarial updates.
# # #     pgd               Previous project's PGD on the reference (L2, eps=3), bit-exact.
# # #     svd_gap           Spectral-gap subspace attack, per-window budget capped at eps=3.
# # #     svd_gap_uncapped  Same, budget limited only by the spectral gap.
# # #     svd_tangent       On-manifold: tangent space of the top-k structure, half-gap budget.
# # #     svd_subspace      Ablation: subspace projection with the fixed budget eps=3.
# # #     svd_gap_budget    Ablation: gap budget without the projection.

# # # For every (seed, arm): fit the encoder, train the same MLP decoder on clean
# # # train embeddings, report validation R2 and, unless --no-eval-attack, R2 under
# # # plain-PGD and spectral-gap attacks on the validation windows, and R2 on
# # # corrupted recordings (Poisson resampling, 10%/30% lost units, gain drift). Results are saved
# # # after every arm in a new, unique run directory; no checkpoints are stored.
# # # """

# # # import argparse
# # # import csv
# # # import gc
# # # import inspect
# # # import json
# # # import os
# # # import random
# # # import sys
# # # import tempfile
# # # import time
# # # from datetime import datetime, timezone
# # # from pathlib import Path

# # # import numpy as np
# # # import torch
# # # from sklearn.metrics import r2_score
# # # from torch import nn

# # # # Prevent the local fork import from generating __pycache__ files.
# # # sys.dont_write_bytecode = True

# # # ROOT = Path(__file__).resolve().parent
# # # FORK_DIR = ROOT / "Acorn-SVD"
# # # DEFAULT_DATA = Path("/data/hossein/mm_project/perich_data_valid_final_raw/C-CO12.npz")

# # # EPSILON = 3.0  # Training budget (cap for the spectral arms); --train-epsilon overrides.
# # # EPS_STEPS = 10
# # # ALPHA_FRACTION = 1 / 5  # Step size as a fraction of the budget: 0.6 for eps=3.
# # # ATTACK_NORM = "l2"
# # # UNCAPPED_EPSILON = 1e3  # Effectively no cap: the spectral gap sets the budget.

# # # # Spectral-gap settings, shared by all spectral arms and the evaluation attack.
# # # GAMMA = 0.9
# # # K_MIN, K_MAX = 1, 8
# # # CENTER = True
# # # PRESERVE_MEAN = False
# # # DIAGNOSTICS_EVERY = 50

# # # LATENT_DIM = 64
# # # HIDDEN = 64
# # # BATCH_SIZE = 2048
# # # MAX_ITER = 3000
# # # TEMPERATURE = 0.4
# # # ARCH = "offset36-model-more-dropout"
# # # DEVICE = "cuda_if_available"

# # # # Decoder: identical to the previous runner.
# # # MLP_EPOCHS = 2500
# # # MLP_HIDDEN = 64
# # # MLP_DROP = 0.4
# # # MLP_LR = 1e-3
# # # EVAL_BATCH_SIZE = 8192

# # # # Held-out attack on the frozen encoder + decoder (prediction MSE).
# # # EVAL_EPSILON = 3.0  # Fixed across runs so that robustness is comparable.
# # # EVAL_STEPS = 20
# # # EVAL_ALPHA = 2.5 * EVAL_EPSILON / EVAL_STEPS
# # # EVAL_ATTACK_BATCH = 2048
# # # EVAL_SEED = 1234

# # # # Recording corruptions of the validation data (Poisson resampling, lost units,
# # # # per-neuron gain drift), averaged over a few random draws.
# # # NATURAL_DRAWS = 5
# # # NATURAL_SEED = 4321

# # # ARMS = {
# # #     "clean": dict(training_mode="standard"),
# # #     "pgd": dict(adv_spectral_mode="none"),
# # #     "svd_gap": dict(adv_spectral_mode="gap"),
# # #     "svd_gap_uncapped": dict(adv_spectral_mode="gap",
# # #                              adv_epsilon=UNCAPPED_EPSILON,
# # #                              adv_alpha=UNCAPPED_EPSILON / 5),
# # #     "svd_tangent": dict(adv_spectral_mode="tangent"),
# # #     "svd_subspace": dict(adv_spectral_mode="subspace"),
# # #     "svd_gap_budget": dict(adv_spectral_mode="gap_budget"),
# # # }
# # # DEFAULT_ARMS = ("clean", "pgd", "svd_gap", "svd_tangent", "svd_gap_budget", "svd_subspace")


# # # def seed_all(seed: int) -> None:
# # #     random.seed(seed)
# # #     np.random.seed(seed)
# # #     torch.manual_seed(seed)
# # #     if torch.cuda.is_available():
# # #         torch.cuda.manual_seed_all(seed)
# # #     torch.backends.cudnn.benchmark = False
# # #     torch.backends.cudnn.deterministic = True


# # # def load_fork():
# # #     if not (FORK_DIR / "cebra" / "__init__.py").is_file():
# # #         raise FileNotFoundError(f"Expected the Acorn-SVD/ directory beside this runner: {FORK_DIR}")
# # #     # Avoid accidentally using an installed/unrelated version of CEBRA.
# # #     for key in list(sys.modules):
# # #         if key == "cebra" or key.startswith("cebra."):
# # #             del sys.modules[key]
# # #     sys.path.insert(0, str(FORK_DIR.resolve()))
# # #     import cebra
# # #     import cebra.solver.spectral_gap
# # #     if not Path(cebra.__file__).resolve().is_relative_to(FORK_DIR.resolve()):
# # #         raise RuntimeError(f"Incorrect CEBRA package imported: {cebra.__file__}")
# # #     params = inspect.signature(cebra.CEBRA.__init__).parameters
# # #     required = {"adv_spectral_mode", "adv_spectral_gamma", "adv_diagnostics_every"}
# # #     missing = required - params.keys()
# # #     if missing:
# # #         raise RuntimeError(f"Acorn-SVD fork lacks expected parameters: {sorted(missing)}")
# # #     return cebra


# # # def load_data(path: Path):
# # #     if not path.is_file():
# # #         raise FileNotFoundError(f"Session file not found: {path}")
# # #     with np.load(path, allow_pickle=False) as f:
# # #         x_train, x_valid, y_train, y_valid = (
# # #             np.asarray(f[key], dtype=np.float32)
# # #             for key in ("train_data", "valid_data", "train_label", "valid_label"))
# # #     if y_train.ndim == 1:
# # #         y_train = y_train[:, None]
# # #     if y_valid.ndim == 1:
# # #         y_valid = y_valid[:, None]
# # #     for key, arr in (("train_data", x_train), ("valid_data", x_valid),
# # #                      ("train_label", y_train), ("valid_label", y_valid)):
# # #         if arr.ndim != 2 or not all(arr.shape) or not np.isfinite(arr).all():
# # #             raise ValueError(f"Invalid {key} shape or NaN/Inf: {arr.shape}")
# # #     if x_train.shape[1] != x_valid.shape[1] or y_train.shape[1] != y_valid.shape[1]:
# # #         raise ValueError("Train/validation feature or label dimensions differ")
# # #     if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
# # #         raise ValueError("Number of labels and neural samples must match")
# # #     if min(len(x_train), len(x_valid)) < 36:
# # #         raise ValueError("offset36 encoder requires >=36 time bins per split")
# # #     return tuple(np.ascontiguousarray(a) for a in (x_train, x_valid, y_train, y_valid))


# # # def encoder_config(arm: str, iterations: int, epsilon: float = EPSILON) -> dict:
# # #     config = dict(
# # #         model_architecture=ARCH,
# # #         batch_size=BATCH_SIZE,
# # #         temperature=TEMPERATURE,
# # #         time_offsets=1,
# # #         max_iterations=iterations,
# # #         output_dimension=LATENT_DIM,
# # #         num_hidden_units=HIDDEN,
# # #         learning_rate=3e-4,
# # #         temperature_mode="constant",
# # #         training_mode="adversarial",
# # #         adv_epsilon=epsilon,
# # #         adv_alpha=epsilon * ALPHA_FRACTION,
# # #         adv_steps=EPS_STEPS,
# # #         attack_norm=ATTACK_NORM,
# # #         adv_spectral_gamma=GAMMA,
# # #         adv_spectral_k_min=K_MIN,
# # #         adv_spectral_k_max=K_MAX,
# # #         adv_spectral_center=CENTER,
# # #         adv_spectral_preserve_mean=PRESERVE_MEAN,
# # #         adv_attack_eval_mode=False,  # Same attack protocol as the previous project.
# # #         adv_diagnostics_every=DIAGNOSTICS_EVERY,
# # #         device=DEVICE,
# # #         verbose=True,  # CEBRA training progress and iteration/loss logs.
# # #     )
# # #     config.update(ARMS[arm])
# # #     return config


# # # class TwoLayerMLP(nn.Module):

# # #     def __init__(self, dim, out):
# # #         super().__init__()
# # #         self.net = nn.Sequential(
# # #             nn.Linear(dim, MLP_HIDDEN), nn.LayerNorm(MLP_HIDDEN),
# # #             nn.ReLU(), nn.Dropout(MLP_DROP), nn.Linear(MLP_HIDDEN, out),
# # #         )

# # #     def forward(self, x):
# # #         return self.net(x)


# # # def train_decoder(z_train, y_train, seed, device):
# # #     seed_all(seed)  # Same decoder seed in all arms.
# # #     decoder = TwoLayerMLP(z_train.shape[1], y_train.shape[1]).to(device)
# # #     optimizer = torch.optim.Adam(decoder.parameters(), lr=MLP_LR)
# # #     criterion = nn.MSELoss()
# # #     z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
# # #     y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
# # #     decoder.train()
# # #     print(f"[DECODER] Start {MLP_EPOCHS} epochs; all {y_train.shape[1]} label columns", flush=True)
# # #     decoder_start = time.perf_counter()
# # #     for epoch in range(MLP_EPOCHS):
# # #         optimizer.zero_grad(set_to_none=True)
# # #         loss = criterion(decoder(z), y)
# # #         if not torch.isfinite(loss).item():
# # #             raise FloatingPointError(f"Nonfinite decoder loss at epoch {epoch + 1}")
# # #         loss.backward()
# # #         optimizer.step()
# # #         if (epoch + 1) % 250 == 0 or epoch + 1 == MLP_EPOCHS:
# # #             print(f"[DECODER] epoch {epoch + 1}/{MLP_EPOCHS} | train_MSE={loss.item():.6f} "
# # #                   f"| elapsed={time.perf_counter() - decoder_start:.1f}s", flush=True)
# # #     return decoder


# # # def predict_decoder(decoder, embeddings):
# # #     decoder.eval()  # Disable Dropout for R2 evaluation.
# # #     device = next(decoder.parameters()).device
# # #     preds = []
# # #     with torch.inference_mode():
# # #         for start in range(0, len(embeddings), EVAL_BATCH_SIZE):
# # #             z = torch.as_tensor(embeddings[start:start + EVAL_BATCH_SIZE],
# # #                                 dtype=torch.float32, device=device)
# # #             preds.append(decoder(z).cpu().numpy())
# # #     return np.concatenate(preds, axis=0)


# # # def mean_r2(y_true, y_pred):
# # #     per_label = np.atleast_1d(r2_score(y_true, y_pred, multioutput="raw_values"))
# # #     if not np.isfinite(per_label).all():
# # #         raise ValueError(f"Nonfinite R2 score: {per_label}")
# # #     return float(per_label.mean()), [float(v) for v in per_label]


# # # def validation_windows(x_valid, y_valid, offset):
# # #     """All validation windows that lie fully inside the recording, as in training."""
# # #     length = offset.left + offset.right
# # #     windows = torch.as_tensor(x_valid).unfold(0, length, 1)  # (M, neurons, length)
# # #     labels = y_valid[offset.left:offset.left + len(windows)]
# # #     return windows.contiguous(), labels


# # # def natural_corruptions(x, rng):
# # #     """Recording perturbations of validation spike counts: noise, lost units, gain drift."""
# # #     yield "poisson", rng.poisson(np.clip(x, 0, None)).astype(np.float32)
# # #     for fraction in (0.1, 0.3):
# # #         keep = np.ones(x.shape[1], dtype=np.float32)
# # #         keep[rng.choice(x.shape[1], int(round(fraction * x.shape[1])), replace=False)] = 0
# # #         yield f"dropout{int(fraction * 100)}", x * keep
# # #     yield "gain30", x * np.exp(rng.normal(0, 0.3, x.shape[1])).astype(np.float32)


# # # def natural_robustness(encoder, decoder, x_valid, y_valid):
# # #     """Validation R2 of the frozen encoder + decoder on corrupted recordings."""
# # #     rng = np.random.default_rng(NATURAL_SEED)
# # #     scores = {}
# # #     for _ in range(NATURAL_DRAWS):
# # #         for name, corrupted in natural_corruptions(x_valid, rng):
# # #             z = np.asarray(encoder.transform(corrupted), dtype=np.float32)
# # #             scores.setdefault(name, []).append(mean_r2(y_valid, predict_decoder(decoder, z))[0])
# # #     return {f"r2_{name}": float(np.mean(values)) for name, values in scores.items()}


# # # def attack_eval_configs(spectral_gap):
# # #     shared = dict(epsilon=EVAL_EPSILON, alpha=EVAL_ALPHA, steps=EVAL_STEPS, norm="l2")
# # #     return {
# # #         "pgd": spectral_gap.AttackConfig(mode="none", **shared),
# # #         "svd_gap": spectral_gap.AttackConfig(mode="gap", gamma=GAMMA, k_min=K_MIN, k_max=K_MAX,
# # #                                              center=CENTER, **shared),
# # #         "svd_tangent": spectral_gap.AttackConfig(mode="tangent", gamma=GAMMA, k_min=K_MIN,
# # #                                                  k_max=K_MAX, center=CENTER, **shared),
# # #     }


# # # def robustness(cebra, encoder, decoder, x_valid, y_valid, z_valid):
# # #     """Validation R2 of the frozen encoder + decoder under held-out attacks."""
# # #     spectral_gap = cebra.solver.spectral_gap
# # #     model = encoder.solver_.model
# # #     device = next(model.parameters()).device
# # #     model.eval()
# # #     decoder.eval()
# # #     windows, labels = validation_windows(x_valid, y_valid, model.get_offset())
# # #     offset = model.get_offset()
# # #     # The per-window model must reproduce the transform used for clean R2.
# # #     with torch.no_grad():
# # #         probe = model(windows[:256].to(device)).cpu().numpy()
# # #     mismatch = float(np.abs(probe - z_valid[offset.left:offset.left + len(probe)]).max())
# # #     if mismatch > 1e-3:
# # #         raise RuntimeError(f"Window embedding does not match transform (max diff {mismatch})")

# # #     out = {"window_mismatch": mismatch, "num_windows": int(len(windows))}
# # #     for name, config in [("clean", None)] + list(attack_eval_configs(spectral_gap).items()):
# # #         seed_all(EVAL_SEED)
# # #         preds, budgets = [], []
# # #         for start in range(0, len(windows), EVAL_ATTACK_BATCH):
# # #             batch = windows[start:start + EVAL_ATTACK_BATCH].to(device)
# # #             target = torch.as_tensor(labels[start:start + len(batch)], device=device)
# # #             if config is not None:
# # #                 frame = spectral_gap.spectral_frame(batch, config) if config.spectral else None
# # #                 if frame is not None:
# # #                     budgets.append(frame.epsilon.cpu())

# # #                 def loss_fn(reference, target=target):
# # #                     return (decoder(model(reference)) - target).square().mean()

# # #                 batch = spectral_gap.attack(batch, loss_fn, config, frame)
# # #             with torch.no_grad():
# # #                 preds.append(decoder(model(batch)).cpu().numpy())
# # #         r2, per_label = mean_r2(labels, np.concatenate(preds))
# # #         out[f"r2_{name}"] = r2
# # #         out[f"r2_{name}_per_label"] = per_label
# # #         if budgets:
# # #             out[f"eval_eps_{name}_mean"] = float(torch.cat(budgets).mean())
# # #     return out


# # # def attack_statistics(solver) -> dict:
# # #     """Training-time attack statistics averaged over iterations."""
# # #     log = solver.log
# # #     stats = {}
# # #     for key in ("adv_loss", "adv_delta_norm", "adv_eps_mean", "adv_eps_min",
# # #                 "adv_gap_limited", "adv_k_mean", "adv_gap_mean"):
# # #         if log.get(key):
# # #             stats[f"train_{key}"] = float(np.mean(log[key]))
# # #     diagnostics = log.get("adv_diagnostics", [])
# # #     for key in ("sv_err_max", "left_subspace_err_max", "right_subspace_err_max",
# # #                 "spectral_drift_mean", "delta_over_gap_max", "gap_kept_min",
# # #                 "collapse_fraction"):
# # #         values = [d[key] for d in diagnostics if np.isfinite(d.get(key, np.nan))]
# # #         if values:
# # #             reduce = np.max if key.endswith("_max") else np.min if key.endswith("_min") else np.mean
# # #             stats[f"diag_{key}"] = float(reduce(values))
# # #     return stats


# # # def run_model(cebra, arm, seed, arrays, iterations, eval_attack, epsilon=EPSILON) -> dict:
# # #     x_train, x_valid, y_train, y_valid = arrays
# # #     encoder = decoder = z_train = z_valid = None
# # #     arm_start = time.perf_counter()
# # #     try:
# # #         seed_all(seed)  # Same initialization and sampling seed in each arm.
# # #         print(f"\n{'=' * 75}", flush=True)
# # #         print(f"[START] {arm} | seed={seed} | eps={epsilon} | {iterations} CEBRA iterations",
# # #               flush=True)
# # #         encoder = cebra.CEBRA(**encoder_config(arm, iterations, epsilon))
# # #         encoder.fit(x_train, y_train)
# # #         encoder_seconds = time.perf_counter() - arm_start
# # #         print(f"[ENCODER] {arm}: {encoder_seconds / 60:.1f} min "
# # #               f"({encoder_seconds / iterations:.3f} s/iteration)", flush=True)

# # #         z_train = np.asarray(encoder.transform(x_train), dtype=np.float32)
# # #         z_valid = np.asarray(encoder.transform(x_valid), dtype=np.float32)
# # #         if (len(z_train) != len(x_train) or len(z_valid) != len(x_valid)
# # #                 or not np.isfinite(z_train).all() or not np.isfinite(z_valid).all()):
# # #             raise ValueError(f"Bad embeddings or alignment for arm={arm}")

# # #         decoder_start = time.perf_counter()
# # #         decoder = train_decoder(z_train, y_train, seed + 100_000,
# # #                                 torch.device(encoder.device_))
# # #         decoder_seconds = time.perf_counter() - decoder_start
# # #         predictions = predict_decoder(decoder, z_valid)
# # #         if predictions.shape != y_valid.shape or not np.isfinite(predictions).all():
# # #             raise ValueError(f"Bad decoder prediction for arm={arm}")
# # #         r2, per_label = mean_r2(y_valid, predictions)
# # #         row = {
# # #             "model": arm, "seed": seed, "epsilon": float(encoder.adv_epsilon),
# # #             "spectral_mode": encoder.adv_spectral_mode
# # #             if encoder.training_mode == "adversarial" else "clean",
# # #             "valid_mean_r2": r2, "valid_r2_per_label": per_label,
# # #             "encoder_seconds": float(encoder_seconds),
# # #             "seconds_per_iteration": float(encoder_seconds / iterations),
# # #             "decoder_seconds": float(decoder_seconds),
# # #         }
# # #         print(f"[RESULT] {arm} seed={seed}: validation mean R2 = {r2:.6f} | per label = {per_label}",
# # #               flush=True)
# # #         if encoder.training_mode == "adversarial":
# # #             row.update(attack_statistics(encoder.solver_))
# # #         row.update(natural_robustness(encoder, decoder, x_valid, y_valid))
# # #         if eval_attack:
# # #             eval_start = time.perf_counter()
# # #             row.update(robustness(cebra, encoder, decoder, x_valid, y_valid, z_valid))
# # #             row["eval_attack_seconds"] = float(time.perf_counter() - eval_start)
# # #             print(f"[ROBUST] {arm} seed={seed}: R2 clean windows = {row['r2_clean']:.4f} | "
# # #                   f"PGD = {row['r2_pgd']:.4f} | SVD-gap = {row['r2_svd_gap']:.4f} | "
# # #                   f"SVD-tangent = {row['r2_svd_tangent']:.4f}", flush=True)
# # #         row["total_seconds"] = float(time.perf_counter() - arm_start)
# # #         return row
# # #     finally:
# # #         del encoder, decoder, z_train, z_valid
# # #         gc.collect()
# # #         if torch.cuda.is_available():
# # #             torch.cuda.empty_cache()


# # # SCALAR_COLUMNS = (
# # #     "model", "seed", "spectral_mode", "epsilon", "valid_mean_r2", "r2_clean", "r2_pgd",
# # #     "r2_svd_gap", "eval_eps_svd_gap_mean", "r2_svd_tangent", "r2_poisson", "r2_dropout10", "r2_dropout30",
# # #     "r2_gain30", "train_adv_eps_mean", "train_adv_gap_limited",
# # #     "train_adv_k_mean", "train_adv_delta_norm", "train_adv_loss", "diag_sv_err_max",
# # #     "diag_left_subspace_err_max", "diag_right_subspace_err_max", "diag_spectral_drift_mean",
# # #     "diag_delta_over_gap_max", "diag_gap_kept_min", "diag_collapse_fraction", "seconds_per_iteration", "encoder_seconds", "decoder_seconds",
# # #     "total_seconds",
# # # )


# # # def save_results(out: Path, results: list):
# # #     """Persist all completed arms atomically, even if a later arm is slow/fails."""
# # #     json_path, temp_json = out / "results_so_far.json", out / ".results_so_far.json.tmp"
# # #     temp_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
# # #     os.replace(temp_json, json_path)
# # #     num_labels = len(results[0]["valid_r2_per_label"])
# # #     csv_path, temp_csv = out / "results_so_far.csv", out / ".results_so_far.csv.tmp"
# # #     with temp_csv.open("w", newline="", encoding="utf-8") as handle:
# # #         writer = csv.writer(handle)
# # #         writer.writerow(list(SCALAR_COLUMNS) + [f"r2_label_{i}" for i in range(num_labels)])
# # #         for row in results:
# # #             writer.writerow([row.get(key, "") for key in SCALAR_COLUMNS] +
# # #                             row["valid_r2_per_label"])
# # #     os.replace(temp_csv, csv_path)


# # # def summarize(results: list) -> list:
# # #     summary = []
# # #     for arm in dict.fromkeys(row["model"] for row in results):
# # #         rows = [row for row in results if row["model"] == arm]
# # #         entry = {"model": arm, "num_seeds": len(rows)}
# # #         for key in ("valid_mean_r2", "r2_pgd", "r2_svd_gap", "r2_svd_tangent", "r2_poisson",
# # #                     "r2_dropout10",
# # #                     "r2_dropout30", "r2_gain30", "seconds_per_iteration"):
# # #             values = [row[key] for row in rows if key in row]
# # #             if values:
# # #                 entry[f"{key}_mean"] = float(np.mean(values))
# # #                 entry[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
# # #         summary.append(entry)
# # #     return summary


# # # def main():
# # #     parser = argparse.ArgumentParser(description=__doc__,
# # #                                      formatter_class=argparse.RawDescriptionHelpFormatter)
# # #     parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
# # #     parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
# # #     parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=list(ARMS))
# # #     parser.add_argument("--iterations", type=int, default=MAX_ITER)
# # #     parser.add_argument("--train-epsilon", type=float, default=EPSILON,
# # #                         help="Training budget of the pgd/svd arms (step = budget/5). "
# # #                              "The evaluation attacks always use eps=3.")
# # #     parser.add_argument("--out", type=Path, default=ROOT / "ACORN_SVD_RESULTS")
# # #     parser.add_argument("--no-eval-attack", action="store_true",
# # #                         help="Skip the held-out attacked-R2 evaluation.")
# # #     args = parser.parse_args()

# # #     # mkdtemp uses atomic creation; runs cannot overwrite one another.
# # #     args.out.mkdir(parents=True, exist_ok=True)
# # #     stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
# # #     out = Path(tempfile.mkdtemp(prefix=f"{args.data.stem}_{stamp}_", dir=args.out))
# # #     print(f"[START] New experiment: {args.data.stem} | output: {out}", flush=True)
# # #     print(f"[START] Arms: {args.arms} | seeds: {args.seeds} | eps: {args.train_epsilon}",
# # #           flush=True)
# # #     cebra = load_fork()
# # #     print(f"[FORK] {cebra.__file__}", flush=True)
# # #     arrays = load_data(args.data)
# # #     print(f"[DATA] Train X={arrays[0].shape}, Y={arrays[2].shape}", flush=True)
# # #     print(f"[DATA] Valid X={arrays[1].shape}, Y={arrays[3].shape}", flush=True)

# # #     config = {
# # #         "data_path": str(args.data), "fork": str(FORK_DIR.resolve()), "cebra_import": str(cebra.__file__),
# # #         "arms": args.arms, "seeds": args.seeds, "iterations": args.iterations,
# # #         "encoder_configs": {arm: encoder_config(arm, args.iterations, args.train_epsilon)
# # #                             for arm in args.arms},
# # #         "decoder": {"architecture": "Linear -> LayerNorm -> ReLU -> Dropout -> Linear",
# # #                     "epochs": MLP_EPOCHS, "hidden": MLP_HIDDEN, "dropout": MLP_DROP,
# # #                     "learning_rate": MLP_LR},
# # #         "eval_attack": None if args.no_eval_attack else {
# # #             name: vars(c) for name, c in attack_eval_configs(cebra.solver.spectral_gap).items()},
# # #         "shapes": {"x_train": arrays[0].shape, "x_valid": arrays[1].shape,
# # #                    "y_train": arrays[2].shape, "y_valid": arrays[3].shape},
# # #     }
# # #     (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

# # #     results = []
# # #     try:
# # #         for seed in args.seeds:  # Seed-major: every arm of a seed finishes first.
# # #             for arm in args.arms:
# # #                 results.append(run_model(cebra, arm, seed, arrays, args.iterations,
# # #                                          not args.no_eval_attack, args.train_epsilon))
# # #                 save_results(out, results)  # Save right after each completed arm.
# # #                 print(f"[SAVED] {len(results)}/{len(args.arms) * len(args.seeds)} results -> "
# # #                       f"{out / 'results_so_far.csv'}", flush=True)
# # #     except BaseException as exc:
# # #         print(f"[INTERRUPTED] {type(exc).__name__}: {exc}", flush=True)
# # #         print(f"[RECOVERY] Completed arms' results (if any) remain in: {out}", flush=True)
# # #         raise

# # #     os.replace(out / "results_so_far.csv", out / "results.csv")
# # #     os.replace(out / "results_so_far.json", out / "results.json")
# # #     summary = summarize(results)
# # #     (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
# # #     print("\n" + "=" * 75, flush=True)
# # #     print("SUMMARY (mean ± std over seeds)", flush=True)
# # #     for entry in summary:
# # #         text = " | ".join(
# # #             f"{key[:-5]}={entry[key]:.4f}±{entry[key[:-5] + '_std']:.4f}"
# # #             for key in entry if key.endswith("_mean"))
# # #         print(f"{entry['model']:18s} n={entry['num_seeds']} | {text}", flush=True)
# # #     print(f"[COMPLETE] Results saved in: {out}", flush=True)


# # # if __name__ == "__main__":
# # #     main()



# # # #"""C-CO12: clean vs PGD vs spectral-gap adversarial CEBRA (Acorn-SVD fork).

# # # # Place this script next to the Acorn-SVD/ directory; run from anywhere:
# # # #     python -u run_acorn_svd.py --data /path/to/C-CO12.npz

# # # # Arms (--arms):
# # # #     clean             CEBRA without adversarial updates.
# # # #     pgd               Previous project's PGD on the reference (L2, eps=3), bit-exact.
# # # #     svd_gap           Spectral-gap subspace attack, per-window budget capped at eps=3.
# # # #     svd_gap_uncapped  Same, budget limited only by the spectral gap.
# # # #     svd_subspace      Ablation: subspace projection with the fixed budget eps=3.
# # # #     svd_gap_budget    Ablation: gap budget without the projection.

# # # # For every (seed, arm): fit the encoder, train the same MLP decoder on clean
# # # # train embeddings, report validation R2 and, unless --no-eval-attack, R2 under
# # # # plain-PGD and spectral-gap attacks on the validation windows. Results are saved
# # # # after every arm in a new, unique run directory; no checkpoints are stored.
# # # # """

# # # # import argparse
# # # # import csv
# # # # import gc
# # # # import inspect
# # # # import json
# # # # import os
# # # # import random
# # # # import sys
# # # # import tempfile
# # # # import time
# # # # from datetime import datetime, timezone
# # # # from pathlib import Path

# # # # import numpy as np
# # # # import torch
# # # # from sklearn.metrics import r2_score
# # # # from torch import nn

# # # # # Prevent the local fork import from generating __pycache__ files.
# # # # sys.dont_write_bytecode = True

# # # # ROOT = Path(__file__).resolve().parent
# # # # FORK_DIR = ROOT / "Acorn-SVD"
# # # # DEFAULT_DATA = Path("/data/hossein/mm_project/perich_data_valid_final_raw/C-CO12.npz")

# # # # EPSILON = 3.0
# # # # EPS_STEPS = 10
# # # # EPS_ALPHA = EPSILON / 5  # 0.6
# # # # ATTACK_NORM = "l2"
# # # # UNCAPPED_EPSILON = 1e3  # Effectively no cap: the spectral gap sets the budget.

# # # # # Spectral-gap settings, shared by all spectral arms and the evaluation attack.
# # # # GAMMA = 0.9
# # # # K_MIN, K_MAX = 1, 8
# # # # CENTER = True
# # # # PRESERVE_MEAN = False
# # # # DIAGNOSTICS_EVERY = 50

# # # # LATENT_DIM = 64
# # # # HIDDEN = 64
# # # # BATCH_SIZE = 2048
# # # # MAX_ITER = 3000
# # # # TEMPERATURE = 0.4
# # # # ARCH = "offset36-model-more-dropout"
# # # # DEVICE = "cuda_if_available"

# # # # # Decoder: identical to the previous runner.
# # # # MLP_EPOCHS = 2500
# # # # MLP_HIDDEN = 64
# # # # MLP_DROP = 0.4
# # # # MLP_LR = 1e-3
# # # # EVAL_BATCH_SIZE = 8192

# # # # # Held-out attack on the frozen encoder + decoder (prediction MSE).
# # # # EVAL_STEPS = 20
# # # # EVAL_ALPHA = 2.5 * EPSILON / EVAL_STEPS
# # # # EVAL_ATTACK_BATCH = 2048
# # # # EVAL_SEED = 1234

# # # # ARMS = {
# # # #     "clean": dict(training_mode="standard"),
# # # #     "pgd": dict(adv_spectral_mode="none"),
# # # #     "svd_gap": dict(adv_spectral_mode="gap"),
# # # #     "svd_gap_uncapped": dict(adv_spectral_mode="gap",
# # # #                              adv_epsilon=UNCAPPED_EPSILON,
# # # #                              adv_alpha=UNCAPPED_EPSILON / 5),
# # # #     "svd_subspace": dict(adv_spectral_mode="subspace"),
# # # #     "svd_gap_budget": dict(adv_spectral_mode="gap_budget"),
# # # # }
# # # # DEFAULT_ARMS = ("clean", "pgd", "svd_gap", "svd_gap_uncapped")


# # # # def seed_all(seed: int) -> None:
# # # #     random.seed(seed)
# # # #     np.random.seed(seed)
# # # #     torch.manual_seed(seed)
# # # #     if torch.cuda.is_available():
# # # #         torch.cuda.manual_seed_all(seed)
# # # #     torch.backends.cudnn.benchmark = False
# # # #     torch.backends.cudnn.deterministic = True


# # # # def load_fork():
# # # #     if not (FORK_DIR / "cebra" / "__init__.py").is_file():
# # # #         raise FileNotFoundError(
# # # #             f"Expected the Acorn-SVD fork next to this script: {FORK_DIR}"
# # # #         )
# # # #     # Avoid accidentally using an installed/unrelated version of CEBRA.
# # # #     for key in list(sys.modules):
# # # #         if key == "cebra" or key.startswith("cebra."):
# # # #             del sys.modules[key]
# # # #     sys.path.insert(0, str(FORK_DIR.resolve()))
# # # #     import cebra
# # # #     import cebra.solver.spectral_gap
# # # #     if not Path(cebra.__file__).resolve().is_relative_to(FORK_DIR.resolve()):
# # # #         raise RuntimeError(f"Incorrect CEBRA package imported: {cebra.__file__}")
# # # #     params = inspect.signature(cebra.CEBRA.__init__).parameters
# # # #     required = {"adv_spectral_mode", "adv_spectral_gamma", "adv_diagnostics_every"}
# # # #     missing = required - params.keys()
# # # #     if missing:
# # # #         raise RuntimeError(f"Acorn-SVD fork lacks expected parameters: {sorted(missing)}")
# # # #     return cebra


# # # # def load_data(path: Path):
# # # #     if not path.is_file():
# # # #         raise FileNotFoundError(f"Session file not found: {path}")
# # # #     with np.load(path, allow_pickle=False) as f:
# # # #         x_train, x_valid, y_train, y_valid = (
# # # #             np.asarray(f[key], dtype=np.float32)
# # # #             for key in ("train_data", "valid_data", "train_label", "valid_label"))
# # # #     if y_train.ndim == 1:
# # # #         y_train = y_train[:, None]
# # # #     if y_valid.ndim == 1:
# # # #         y_valid = y_valid[:, None]
# # # #     for key, arr in (("train_data", x_train), ("valid_data", x_valid),
# # # #                      ("train_label", y_train), ("valid_label", y_valid)):
# # # #         if arr.ndim != 2 or not all(arr.shape) or not np.isfinite(arr).all():
# # # #             raise ValueError(f"Invalid {key} shape or NaN/Inf: {arr.shape}")
# # # #     if x_train.shape[1] != x_valid.shape[1] or y_train.shape[1] != y_valid.shape[1]:
# # # #         raise ValueError("Train/validation feature or label dimensions differ")
# # # #     if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
# # # #         raise ValueError("Number of labels and neural samples must match")
# # # #     if min(len(x_train), len(x_valid)) < 36:
# # # #         raise ValueError("offset36 encoder requires >=36 time bins per split")
# # # #     return tuple(np.ascontiguousarray(a) for a in (x_train, x_valid, y_train, y_valid))


# # # # def encoder_config(arm: str, iterations: int) -> dict:
# # # #     config = dict(
# # # #         model_architecture=ARCH,
# # # #         batch_size=BATCH_SIZE,
# # # #         temperature=TEMPERATURE,
# # # #         time_offsets=1,
# # # #         max_iterations=iterations,
# # # #         output_dimension=LATENT_DIM,
# # # #         num_hidden_units=HIDDEN,
# # # #         learning_rate=3e-4,
# # # #         temperature_mode="constant",
# # # #         training_mode="adversarial",
# # # #         adv_epsilon=EPSILON,
# # # #         adv_alpha=EPS_ALPHA,
# # # #         adv_steps=EPS_STEPS,
# # # #         attack_norm=ATTACK_NORM,
# # # #         adv_spectral_gamma=GAMMA,
# # # #         adv_spectral_k_min=K_MIN,
# # # #         adv_spectral_k_max=K_MAX,
# # # #         adv_spectral_center=CENTER,
# # # #         adv_spectral_preserve_mean=PRESERVE_MEAN,
# # # #         adv_attack_eval_mode=False,  # Same attack protocol as the previous project.
# # # #         adv_diagnostics_every=DIAGNOSTICS_EVERY,
# # # #         device=DEVICE,
# # # #         verbose=True,  # CEBRA training progress and iteration/loss logs.
# # # #     )
# # # #     config.update(ARMS[arm])
# # # #     return config


# # # # class TwoLayerMLP(nn.Module):

# # # #     def __init__(self, dim, out):
# # # #         super().__init__()
# # # #         self.net = nn.Sequential(
# # # #             nn.Linear(dim, MLP_HIDDEN), nn.LayerNorm(MLP_HIDDEN),
# # # #             nn.ReLU(), nn.Dropout(MLP_DROP), nn.Linear(MLP_HIDDEN, out),
# # # #         )

# # # #     def forward(self, x):
# # # #         return self.net(x)


# # # # def train_decoder(z_train, y_train, seed, device):
# # # #     seed_all(seed)  # Same decoder seed in all arms.
# # # #     decoder = TwoLayerMLP(z_train.shape[1], y_train.shape[1]).to(device)
# # # #     optimizer = torch.optim.Adam(decoder.parameters(), lr=MLP_LR)
# # # #     criterion = nn.MSELoss()
# # # #     z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
# # # #     y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
# # # #     decoder.train()
# # # #     print(f"[DECODER] Start {MLP_EPOCHS} epochs; all {y_train.shape[1]} label columns", flush=True)
# # # #     decoder_start = time.perf_counter()
# # # #     for epoch in range(MLP_EPOCHS):
# # # #         optimizer.zero_grad(set_to_none=True)
# # # #         loss = criterion(decoder(z), y)
# # # #         if not torch.isfinite(loss).item():
# # # #             raise FloatingPointError(f"Nonfinite decoder loss at epoch {epoch + 1}")
# # # #         loss.backward()
# # # #         optimizer.step()
# # # #         if (epoch + 1) % 250 == 0 or epoch + 1 == MLP_EPOCHS:
# # # #             print(f"[DECODER] epoch {epoch + 1}/{MLP_EPOCHS} | train_MSE={loss.item():.6f} "
# # # #                   f"| elapsed={time.perf_counter() - decoder_start:.1f}s", flush=True)
# # # #     return decoder


# # # # def predict_decoder(decoder, embeddings):
# # # #     decoder.eval()  # Disable Dropout for R2 evaluation.
# # # #     device = next(decoder.parameters()).device
# # # #     preds = []
# # # #     with torch.inference_mode():
# # # #         for start in range(0, len(embeddings), EVAL_BATCH_SIZE):
# # # #             z = torch.as_tensor(embeddings[start:start + EVAL_BATCH_SIZE],
# # # #                                 dtype=torch.float32, device=device)
# # # #             preds.append(decoder(z).cpu().numpy())
# # # #     return np.concatenate(preds, axis=0)


# # # # def mean_r2(y_true, y_pred):
# # # #     per_label = np.atleast_1d(r2_score(y_true, y_pred, multioutput="raw_values"))
# # # #     if not np.isfinite(per_label).all():
# # # #         raise ValueError(f"Nonfinite R2 score: {per_label}")
# # # #     return float(per_label.mean()), [float(v) for v in per_label]


# # # # def validation_windows(x_valid, y_valid, offset):
# # # #     """All validation windows that lie fully inside the recording, as in training."""
# # # #     length = offset.left + offset.right
# # # #     windows = torch.as_tensor(x_valid).unfold(0, length, 1)  # (M, neurons, length)
# # # #     labels = y_valid[offset.left:offset.left + len(windows)]
# # # #     return windows.contiguous(), labels


# # # # def attack_eval_configs(spectral_gap):
# # # #     shared = dict(epsilon=EPSILON, alpha=EVAL_ALPHA, steps=EVAL_STEPS, norm="l2")
# # # #     return {
# # # #         "pgd": spectral_gap.AttackConfig(mode="none", **shared),
# # # #         "svd_gap": spectral_gap.AttackConfig(mode="gap", gamma=GAMMA, k_min=K_MIN, k_max=K_MAX,
# # # #                                              center=CENTER, **shared),
# # # #     }


# # # # def robustness(cebra, encoder, decoder, x_valid, y_valid, z_valid):
# # # #     """Validation R2 of the frozen encoder + decoder under held-out attacks."""
# # # #     spectral_gap = cebra.solver.spectral_gap
# # # #     model = encoder.solver_.model
# # # #     device = next(model.parameters()).device
# # # #     model.eval()
# # # #     decoder.eval()
# # # #     windows, labels = validation_windows(x_valid, y_valid, model.get_offset())
# # # #     offset = model.get_offset()
# # # #     # The per-window model must reproduce the transform used for clean R2.
# # # #     with torch.no_grad():
# # # #         probe = model(windows[:256].to(device)).cpu().numpy()
# # # #     mismatch = float(np.abs(probe - z_valid[offset.left:offset.left + len(probe)]).max())
# # # #     if mismatch > 1e-3:
# # # #         raise RuntimeError(f"Window embedding does not match transform (max diff {mismatch})")

# # # #     out = {"window_mismatch": mismatch, "num_windows": int(len(windows))}
# # # #     for name, config in [("clean", None)] + list(attack_eval_configs(spectral_gap).items()):
# # # #         seed_all(EVAL_SEED)
# # # #         preds, budgets = [], []
# # # #         for start in range(0, len(windows), EVAL_ATTACK_BATCH):
# # # #             batch = windows[start:start + EVAL_ATTACK_BATCH].to(device)
# # # #             target = torch.as_tensor(labels[start:start + len(batch)], device=device)
# # # #             if config is not None:
# # # #                 frame = spectral_gap.spectral_frame(batch, config) if config.spectral else None
# # # #                 if frame is not None:
# # # #                     budgets.append(frame.epsilon.cpu())

# # # #                 def loss_fn(reference, target=target):
# # # #                     return (decoder(model(reference)) - target).square().mean()

# # # #                 batch = spectral_gap.attack(batch, loss_fn, config, frame)
# # # #             with torch.no_grad():
# # # #                 preds.append(decoder(model(batch)).cpu().numpy())
# # # #         r2, per_label = mean_r2(labels, np.concatenate(preds))
# # # #         out[f"r2_{name}"] = r2
# # # #         out[f"r2_{name}_per_label"] = per_label
# # # #         if budgets:
# # # #             out[f"eval_eps_{name}_mean"] = float(torch.cat(budgets).mean())
# # # #     return out


# # # # def attack_statistics(solver) -> dict:
# # # #     """Training-time attack statistics averaged over iterations."""
# # # #     log = solver.log
# # # #     stats = {}
# # # #     for key in ("adv_loss", "adv_delta_norm", "adv_eps_mean", "adv_eps_min",
# # # #                 "adv_gap_limited", "adv_k_mean", "adv_gap_mean"):
# # # #         if log.get(key):
# # # #             stats[f"train_{key}"] = float(np.mean(log[key]))
# # # #     diagnostics = log.get("adv_diagnostics", [])
# # # #     for key in ("sv_err_max", "left_subspace_err_max", "right_subspace_err_max",
# # # #                 "spectral_drift_mean", "delta_over_gap_max"):
# # # #         values = [d[key] for d in diagnostics if np.isfinite(d.get(key, np.nan))]
# # # #         if values:
# # # #             stats[f"diag_{key}"] = float(np.max(values) if key.endswith("_max") else np.mean(values))
# # # #     return stats


# # # # def run_model(cebra, arm, seed, arrays, iterations, eval_attack) -> dict:
# # # #     x_train, x_valid, y_train, y_valid = arrays
# # # #     encoder = decoder = z_train = z_valid = None
# # # #     arm_start = time.perf_counter()
# # # #     try:
# # # #         seed_all(seed)  # Same initialization and sampling seed in each arm.
# # # #         print(f"\n{'=' * 75}", flush=True)
# # # #         print(f"[START] {arm} | seed={seed} | eps={EPSILON} | {iterations} CEBRA iterations",
# # # #               flush=True)
# # # #         encoder = cebra.CEBRA(**encoder_config(arm, iterations))
# # # #         encoder.fit(x_train, y_train)
# # # #         encoder_seconds = time.perf_counter() - arm_start
# # # #         print(f"[ENCODER] {arm}: {encoder_seconds / 60:.1f} min "
# # # #               f"({encoder_seconds / iterations:.3f} s/iteration)", flush=True)

# # # #         z_train = np.asarray(encoder.transform(x_train), dtype=np.float32)
# # # #         z_valid = np.asarray(encoder.transform(x_valid), dtype=np.float32)
# # # #         if (len(z_train) != len(x_train) or len(z_valid) != len(x_valid)
# # # #                 or not np.isfinite(z_train).all() or not np.isfinite(z_valid).all()):
# # # #             raise ValueError(f"Bad embeddings or alignment for arm={arm}")

# # # #         decoder_start = time.perf_counter()
# # # #         decoder = train_decoder(z_train, y_train, seed + 100_000,
# # # #                                 torch.device(encoder.device_))
# # # #         decoder_seconds = time.perf_counter() - decoder_start
# # # #         predictions = predict_decoder(decoder, z_valid)
# # # #         if predictions.shape != y_valid.shape or not np.isfinite(predictions).all():
# # # #             raise ValueError(f"Bad decoder prediction for arm={arm}")
# # # #         r2, per_label = mean_r2(y_valid, predictions)
# # # #         row = {
# # # #             "model": arm, "seed": seed, "epsilon": EPSILON,
# # # #             "spectral_mode": encoder.adv_spectral_mode
# # # #             if encoder.training_mode == "adversarial" else "clean",
# # # #             "valid_mean_r2": r2, "valid_r2_per_label": per_label,
# # # #             "encoder_seconds": float(encoder_seconds),
# # # #             "seconds_per_iteration": float(encoder_seconds / iterations),
# # # #             "decoder_seconds": float(decoder_seconds),
# # # #         }
# # # #         print(f"[RESULT] {arm} seed={seed}: validation mean R2 = {r2:.6f} | per label = {per_label}",
# # # #               flush=True)
# # # #         if encoder.training_mode == "adversarial":
# # # #             row.update(attack_statistics(encoder.solver_))
# # # #         if eval_attack:
# # # #             eval_start = time.perf_counter()
# # # #             row.update(robustness(cebra, encoder, decoder, x_valid, y_valid, z_valid))
# # # #             row["eval_attack_seconds"] = float(time.perf_counter() - eval_start)
# # # #             print(f"[ROBUST] {arm} seed={seed}: R2 clean windows = {row['r2_clean']:.4f} | "
# # # #                   f"PGD = {row['r2_pgd']:.4f} | SVD-gap = {row['r2_svd_gap']:.4f}", flush=True)
# # # #         row["total_seconds"] = float(time.perf_counter() - arm_start)
# # # #         return row
# # # #     finally:
# # # #         del encoder, decoder, z_train, z_valid
# # # #         gc.collect()
# # # #         if torch.cuda.is_available():
# # # #             torch.cuda.empty_cache()


# # # # SCALAR_COLUMNS = (
# # # #     "model", "seed", "spectral_mode", "epsilon", "valid_mean_r2", "r2_clean", "r2_pgd",
# # # #     "r2_svd_gap", "eval_eps_svd_gap_mean", "train_adv_eps_mean", "train_adv_gap_limited",
# # # #     "train_adv_k_mean", "train_adv_delta_norm", "train_adv_loss", "diag_sv_err_max",
# # # #     "diag_left_subspace_err_max", "diag_right_subspace_err_max", "diag_spectral_drift_mean",
# # # #     "diag_delta_over_gap_max", "seconds_per_iteration", "encoder_seconds", "decoder_seconds",
# # # #     "total_seconds",
# # # # )


# # # # def save_results(out: Path, results: list):
# # # #     """Persist all completed arms atomically, even if a later arm is slow/fails."""
# # # #     json_path, temp_json = out / "results_so_far.json", out / ".results_so_far.json.tmp"
# # # #     temp_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
# # # #     os.replace(temp_json, json_path)
# # # #     num_labels = len(results[0]["valid_r2_per_label"])
# # # #     csv_path, temp_csv = out / "results_so_far.csv", out / ".results_so_far.csv.tmp"
# # # #     with temp_csv.open("w", newline="", encoding="utf-8") as handle:
# # # #         writer = csv.writer(handle)
# # # #         writer.writerow(list(SCALAR_COLUMNS) + [f"r2_label_{i}" for i in range(num_labels)])
# # # #         for row in results:
# # # #             writer.writerow([row.get(key, "") for key in SCALAR_COLUMNS] +
# # # #                             row["valid_r2_per_label"])
# # # #     os.replace(temp_csv, csv_path)


# # # # def summarize(results: list) -> list:
# # # #     summary = []
# # # #     for arm in dict.fromkeys(row["model"] for row in results):
# # # #         rows = [row for row in results if row["model"] == arm]
# # # #         entry = {"model": arm, "num_seeds": len(rows)}
# # # #         for key in ("valid_mean_r2", "r2_pgd", "r2_svd_gap", "seconds_per_iteration"):
# # # #             values = [row[key] for row in rows if key in row]
# # # #             if values:
# # # #                 entry[f"{key}_mean"] = float(np.mean(values))
# # # #                 entry[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
# # # #         summary.append(entry)
# # # #     return summary


# # # # def main():
# # # #     parser = argparse.ArgumentParser(description=__doc__,
# # # #                                      formatter_class=argparse.RawDescriptionHelpFormatter)
# # # #     parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
# # # #     parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
# # # #     parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=list(ARMS))
# # # #     parser.add_argument("--iterations", type=int, default=MAX_ITER)
# # # #     parser.add_argument("--out", type=Path, default=ROOT / "ACORN_SVD_RESULTS")
# # # #     parser.add_argument("--no-eval-attack", action="store_true",
# # # #                         help="Skip the held-out attacked-R2 evaluation.")
# # # #     args = parser.parse_args()

# # # #     # mkdtemp uses atomic creation; runs cannot overwrite one another.
# # # #     args.out.mkdir(parents=True, exist_ok=True)
# # # #     stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
# # # #     out = Path(tempfile.mkdtemp(prefix=f"{args.data.stem}_{stamp}_", dir=args.out))
# # # #     print(f"[START] New experiment: {args.data.stem} | output: {out}", flush=True)
# # # #     print(f"[START] Arms: {args.arms} | seeds: {args.seeds} | eps: {EPSILON}", flush=True)
# # # #     cebra = load_fork()
# # # #     print(f"[FORK] {cebra.__file__}", flush=True)
# # # #     arrays = load_data(args.data)
# # # #     print(f"[DATA] Train X={arrays[0].shape}, Y={arrays[2].shape}", flush=True)
# # # #     print(f"[DATA] Valid X={arrays[1].shape}, Y={arrays[3].shape}", flush=True)

# # # #     config = {
# # # #         "data_path": str(args.data), "fork": str(FORK_DIR), "cebra_import": str(cebra.__file__),
# # # #         "arms": args.arms, "seeds": args.seeds, "iterations": args.iterations,
# # # #         "encoder_configs": {arm: encoder_config(arm, args.iterations) for arm in args.arms},
# # # #         "decoder": {"architecture": "Linear -> LayerNorm -> ReLU -> Dropout -> Linear",
# # # #                     "epochs": MLP_EPOCHS, "hidden": MLP_HIDDEN, "dropout": MLP_DROP,
# # # #                     "learning_rate": MLP_LR},
# # # #         "eval_attack": None if args.no_eval_attack else {
# # # #             name: vars(c) for name, c in attack_eval_configs(cebra.solver.spectral_gap).items()},
# # # #         "shapes": {"x_train": arrays[0].shape, "x_valid": arrays[1].shape,
# # # #                    "y_train": arrays[2].shape, "y_valid": arrays[3].shape},
# # # #     }
# # # #     (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

# # # #     results = []
# # # #     try:
# # # #         for seed in args.seeds:  # Seed-major: every arm of a seed finishes first.
# # # #             for arm in args.arms:
# # # #                 results.append(run_model(cebra, arm, seed, arrays, args.iterations,
# # # #                                          not args.no_eval_attack))
# # # #                 save_results(out, results)  # Save right after each completed arm.
# # # #                 print(f"[SAVED] {len(results)}/{len(args.arms) * len(args.seeds)} results -> "
# # # #                       f"{out / 'results_so_far.csv'}", flush=True)
# # # #     except BaseException as exc:
# # # #         print(f"[INTERRUPTED] {type(exc).__name__}: {exc}", flush=True)
# # # #         print(f"[RECOVERY] Completed arms' results (if any) remain in: {out}", flush=True)
# # # #         raise

# # # #     os.replace(out / "results_so_far.csv", out / "results.csv")
# # # #     os.replace(out / "results_so_far.json", out / "results.json")
# # # #     summary = summarize(results)
# # # #     (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
# # # #     print("\n" + "=" * 75, flush=True)
# # # #     print("SUMMARY (mean ± std over seeds)", flush=True)
# # # #     for entry in summary:
# # # #         text = " | ".join(
# # # #             f"{key[:-5]}={entry[key]:.4f}±{entry[key[:-5] + '_std']:.4f}"
# # # #             for key in entry if key.endswith("_mean"))
# # # #         print(f"{entry['model']:18s} n={entry['num_seeds']} | {text}", flush=True)
# # # #     print(f"[COMPLETE] Results saved in: {out}", flush=True)


# # # # if __name__ == "__main__":
# # # #     main()

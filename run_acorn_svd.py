"""C-CO12: clean vs PGD vs spectral-gap adversarial CEBRA (Acorn-SVD fork).

Place this runner NEXT TO the Acorn-SVD/ directory:
    Aggregate/run_acorn_svd.py
    Aggregate/Acorn-SVD/cebra/
Run from anywhere, e.g. python -u run_acorn_svd.py --data /path/to/C-CO12.npz

Multi-GPU: --gpu-workers N assigns one independent experiment per GPU.
Plan safely with --plan-only. --run-dir + --resume preserve completed runs.

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
    svd_cert_radial   Same, final perturbation shrunk as a whole (Weyl radius, as in v3).
    svd_cert_collapse Same, final perturbation scaled to the exact collapse radius.
    svd_split_half    svd_split with half of the complement budget (trade-off knob).
    svd_split_every   svd_split, jointly certified after every PGD step.
    svd_tangent_fixed Ablation: tangent projection with the fixed budget eps.
    svd_gap_budget    Ablation: gap budget without the projection.
    pgd_match_split   Control: unrestricted PGD with svd_split's per-window perturbation norm.
    pgd_match_radial  Control: unrestricted PGD with svd_cert_radial's per-window norm.

For every (seed, arm) the runner:
    - fits the encoder and trains the same MLP decoder on clean train embeddings;
    - reports validation R2;
    - unless --no-eval-attack, reports R2 under held-out attacks on the validation
      windows (plain PGD, spectral gap, tangent, and complement vs tangent at the same
      radius; --eval-epsilon, --eval-steps, --eval-restarts; plain PGD also at the
      radii of --eval-radii);
    - reports R2 on corrupted recordings: Poisson resampling, gain drift, random and
      nested 5-50% unit loss, and loss of the 20% most active, variable or important units.

--epsilon-unit window expresses all budgets as fractions of the median norm of
centred training windows, so that sessions are comparable. --scale-only prints
that norm, the budgets in both units and the spectral gaps of the session, then
exits without training. Results are saved after every arm in a new, unique run
directory; no checkpoints are stored. Default output is a separate
ACORN_SVD_RESULTS_V5/ directory alongside Acorn-SVD/.
"""

import argparse
import hashlib
import queue
import subprocess
import threading
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

SCRIPT_DIR = Path(__file__).resolve().parent
# User-required original layout, with a secondary standalone fork option for tests.
if (SCRIPT_DIR / "Acorn-SVD" / "cebra" / "__init__.py").is_file():
    ROOT = SCRIPT_DIR / "Acorn-SVD"
elif (SCRIPT_DIR / "cebra" / "__init__.py").is_file():
    ROOT = SCRIPT_DIR
else:
    raise FileNotFoundError(
        "Expected Acorn-SVD/cebra/ alongside run_acorn_svd.py. "
        f"Runner directory: {SCRIPT_DIR}"
    )
# Reject older forks even in plan-only mode, without loading data or touching CUDA.
engine = ROOT / "cebra" / "solver" / "spectral_gap.py"
integration = ROOT / "cebra" / "integrations" / "sklearn" / "cebra.py"
if not engine.is_file() or not integration.is_file() or not all(
    key in engine.read_text(encoding="utf-8")
    for key in ("certify_collapse", "adv_top_unit_energy", "adv_spectral_energy_match")
) or "adv_spectral_energy_match" not in integration.read_text(encoding="utf-8"):
    raise RuntimeError(f"This is not the updated V5 spectral fork: {ROOT}. "
                       "Install the full updated Acorn-SVD/ before using this runner.")
DEFAULT_OUTPUT = ROOT.parent / "ACORN_SVD_RESULTS_V5"
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
    "svd_cert_collapse": dict(adv_spectral_mode="subspace_cert", adv_spectral_certify="collapse"),
    # Same per-window energy as the spectral arm, unrestricted direction.
    "pgd_match_split": dict(adv_spectral_mode="split", adv_spectral_energy_match=True),
    "pgd_match_radial": dict(adv_spectral_mode="subspace_cert", adv_spectral_certify="radial",
                             adv_spectral_energy_match=True),
}
DEFAULT_ARMS = ("clean", "pgd", "svd_cert_radial", "svd_cert_collapse", "svd_split",
                "pgd_match_split", "pgd_match_radial")


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
    required = {"adv_spectral_mode", "adv_spectral_gamma", "adv_diagnostics_every",
                "adv_spectral_certify", "adv_spectral_energy_match"}
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
               steps=EVAL_STEPS, restarts=EVAL_RESTARTS, radii=()):
    """Validation R2 of the frozen encoder + decoder under held-out attacks.

    ``radii`` holds ``(label, radius)`` pairs at which plain PGD is also run
    (``r2_pgd_eps<label>``); each attack restarts from the same seed, so extra
    radii do not change the other results.
    """
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
    for label, radius in radii:
        configs[f"pgd_eps{label}"] = spectral_gap.AttackConfig(
            mode="none", epsilon=radius, alpha=2.5 * radius / steps, steps=steps, norm="l2")
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
                "adv_tangent_eps_mean", "adv_cert_active", "adv_cert_retained",
                "adv_cert_weyl_gain", "adv_top_unit_energy", "adv_top_unit_signal"):
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
            curve = "".join(f" | PGD@{key[10:]} = {row[key]:.4f}" for key in radius_columns([row]))
            print(f"[ROBUST] {arm} seed={seed}: R2 clean windows = {row['r2_clean']:.4f} | "
                  f"PGD = {row['r2_pgd']:.4f} | SVD-gap = {row['r2_svd_gap']:.4f} | "
                  f"SVD-tangent = {row['r2_svd_tangent']:.4f} | complement = "
                  f"{row['r2_complement']:.4f} | tangent (eps) = {row['r2_tangent_fixed']:.4f}"
                  f"{curve}", flush=True)
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
    "r2_dropimp20", "unit_importance_top_share", "train_adv_eps_mean", "train_adv_gap_limited",
    "train_adv_cert_active", "train_adv_cert_retained", "train_adv_cert_weyl_gain",
    "train_adv_top_unit_energy", "train_adv_top_unit_signal", "train_adv_k_mean",
    "train_adv_delta_norm",
    "train_adv_delta_norm_p10", "train_adv_delta_norm_p90", "train_adv_loss",
    "train_adv_tangent_norm", "train_adv_complement_norm", "diag_sv_err_max",
    "diag_left_subspace_err_max", "diag_right_subspace_err_max", "diag_spectral_drift_mean",
    "diag_delta_over_gap_max", "diag_gap_kept_min", "diag_collapse_fraction",
    "diag_mode_exchange_fraction", "diag_tangent_energy_fraction",
    "diag_tangent_chance_fraction", "seconds_per_iteration", "encoder_seconds",
    "decoder_seconds", "total_seconds",
)


def radius_columns(results: list) -> list:
    """Names of the PGD results at the extra evaluation radii (--eval-radii)."""
    return sorted({key for row in results for key in row
                   if key.startswith("r2_pgd_eps") and not key.endswith("_per_label")})


def save_results(out: Path, results: list):
    """Persist all completed arms atomically, even if a later arm is slow/fails."""
    json_path, temp_json = out / "results_so_far.json", out / ".results_so_far.json.tmp"
    temp_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp_json, json_path)
    num_labels = len(results[0]["valid_r2_per_label"])
    csv_path, temp_csv = out / "results_so_far.csv", out / ".results_so_far.csv.tmp"
    with temp_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        columns = list(SCALAR_COLUMNS) + radius_columns(results)
        writer.writerow(columns + [f"r2_label_{i}" for i in range(num_labels)])
        for row in results:
            writer.writerow([row.get(key, "") for key in columns] + row["valid_r2_per_label"])
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
                    "train_adv_delta_norm", "train_adv_cert_active", "train_adv_cert_retained",
                    "train_adv_cert_weyl_gain", "train_adv_top_unit_energy",
                    "train_adv_top_unit_signal", "r2_dropvar20",
                    "seconds_per_iteration",
                    *radius_columns(rows)):
            values = [row[key] for row in rows if key in row]
            if values:
                entry[f"{key}_mean"] = float(np.mean(values))
                entry[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(entry)
    return summary


def _atomic_json(path: Path, value) -> None:
    """One worker owns one per-task JSON; commit its contents atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _task_filename(index: int, task: dict) -> str:
    return f"{index:03d}_seed{task['seed']}_{task['arm']}_eps{task['nominal']:g}".replace(".", "p")


def _task_result(out: Path, index: int, task: dict) -> Path:
    return out / "jobs" / (_task_filename(index, task) + ".json")


def _read_finished(out: Path, tasks: list[dict]) -> tuple[list[dict], list[int]]:
    """Read only final committed results, in stable plan order."""
    rows, missing = [], []
    for i, task in enumerate(tasks, 1):
        p = _task_result(out, i, task)
        if not p.is_file():
            missing.append(i)
            continue
        record = json.loads(p.read_text(encoding="utf-8"))
        if record.get("task") != task or record.get("status") != "success":
            raise RuntimeError(f"Invalid/mismatched task result: {p}")
        rows.append(record["result"])
    return rows, missing


def _gpu_tokens(workers: int) -> list[str]:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPUs visible. The GPU training job requires RunAI --gpu N.")
    visible_count = torch.cuda.device_count()
    if visible_count < workers:
        raise RuntimeError(f"--gpu-workers {workers} requested, but only {visible_count} CUDA GPUs visible")
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visibility and visibility not in ("-1", "NoDevFiles"):
        tokens = [v.strip() for v in visibility.split(",") if v.strip()]
        if len(tokens) < workers:
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visibility!r} exposes fewer than {workers} tokens")
        return tokens[:workers]
    return [str(i) for i in range(workers)]


def _worker_main(run_dir: Path, task_index: int, cpu_threads: int) -> int:
    """Called in a fresh subprocess with exactly one visible CUDA device."""
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A GPU worker must see exactly ONE CUDA device")
    torch.set_num_threads(cpu_threads)
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    task = cfg["tasks"][task_index - 1]
    result_path = _task_result(run_dir, task_index, task)
    if result_path.exists():
        raise RuntimeError(f"Refusing to overwrite completed task: {result_path}")
    print(f"[WORKER] GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} | "
          f"{task_index}/{len(cfg['tasks'])} | {task} | pid={os.getpid()}", flush=True)
    cebra = load_fork()
    arrays = load_data(Path(cfg["data_path"]))
    meta = {"epsilon_nominal": task["nominal"], "epsilon_unit": cfg["epsilon_unit"],
            "window_norm_median": cfg["window_norm_median"],
            "eval_epsilon": cfg["evaluation"]["epsilon"]}
    # Validate fork and runner fingerprints again inside each isolated process.
    for rel, expected_hash in cfg["fork_sha256"].items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"Fork file changed during experiment: {rel}")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != cfg["runner_sha256"]:
        raise RuntimeError("Runner changed during experiment; refusing mixed-version results")
    row = run_model(cebra, task["arm"], task["seed"], arrays, cfg["iterations"],
                    not cfg["no_eval_attack"], task["epsilon"], cfg["spectral"],
                    cfg["evaluation"], meta)
    _atomic_json(result_path, {"status": "success", "task": task, "result": row})
    print(f"[WORKER] COMMITTED {result_path}", flush=True)
    return 0


def _parallel_execute(out: Path, tasks: list[dict], workers: int, cpu_threads: int) -> int:
    rows, missing = _read_finished(out, tasks)
    if rows:
        save_results(out, rows)
    if not missing:
        print(f"[RESUME] All {len(tasks)} tasks already completed", flush=True)
        return 0
    tokens = _gpu_tokens(workers)
    print(f"[GPU] {workers} parallel processes, {len(missing)}/{len(tasks)} pending tasks; "
          f"tokens={tokens}; cpu-threads/process={cpu_threads}", flush=True)
    todo = queue.Queue()
    for i in missing:
        todo.put(i)
    logdir = out / "logs"
    logdir.mkdir(exist_ok=True)
    mutex = threading.Lock()
    failures = []

    def gpu_loop(slot: int, token: str):
        while True:
            try:
                index = todo.get_nowait()
            except queue.Empty:
                return
            task = tasks[index - 1]
            fname = _task_filename(index, task)
            log_file = logdir / f"{fname}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = token
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[key] = str(cpu_threads)
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--_worker", "--_task-index", str(index),
                       "--_run-dir", str(out), "--cpu-threads", str(cpu_threads)]
            with mutex:
                print(f"[GPU {slot}] START {index}/{len(tasks)} {task['arm']} "
                      f"seed={task['seed']} eps={task['nominal']} | {log_file}", flush=True)
            try:
                with log_file.open("w", encoding="utf-8") as f:
                    status = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT,
                                            check=False).returncode
            except Exception as exc:
                status = -1
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(f"\nCOORDINATOR EXCEPTION: {type(exc).__name__}: {exc}\n")
            with mutex:
                if status == 0 and _task_result(out, index, task).is_file():
                    print(f"[GPU {slot}] DONE {index}/{len(tasks)} {task['arm']} "
                          f"seed={task['seed']} eps={task['nominal']}", flush=True)
                else:
                    failures.append(index)
                    print(f"[GPU {slot}] FAILED task {index} exit={status}; log={log_file}", flush=True)
                # Even if the coordinator is interrupted, per-task JSONs survive.
                finished, _ = _read_finished(out, tasks)
                if finished:
                    save_results(out, finished)
                print(f"[PROGRESS] {len(finished)}/{len(tasks)} committed", flush=True)
            todo.task_done()

    threads = [threading.Thread(target=gpu_loop, args=(slot, token), daemon=False)
               for slot, token in enumerate(tokens)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return len(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=list(ARMS))
    parser.add_argument("--iterations", type=int, default=MAX_ITER)
    parser.add_argument("--train-epsilon", type=float, nargs="+", default=[EPSILON])
    parser.add_argument("--eval-epsilon", type=float, default=EVAL_EPSILON)
    parser.add_argument("--eval-steps", type=int, default=EVAL_STEPS)
    parser.add_argument("--eval-restarts", type=int, default=EVAL_RESTARTS)
    parser.add_argument("--eval-radii", type=float, nargs="*", default=[],
                        help="Additional PGD robustness radii, e.g. --eval-radii 1 6; main radius 3 by default")
    parser.add_argument("--epsilon-unit", choices=("absolute", "window"), default="absolute")
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--k-min", type=int, default=K_MIN)
    parser.add_argument("--k-max", type=int, default=K_MAX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT,
                        help="Persistent V5 output root; distinct from 72-run V4 results")
    parser.add_argument("--run-dir", type=Path, help="Fixed subfolder of --out, for resumable experiments")
    parser.add_argument("--resume", action="store_true", help="Skip fully committed tasks after preemption")
    parser.add_argument("--gpu-workers", type=int, default=1,
                        help="Number of independent GPU subprocesses (request same count in RunAI)")
    parser.add_argument("--cpu-threads", type=int, default=8,
                        help="CPU threads for each GPU subprocess")
    parser.add_argument("--plan-only", action="store_true", help="Show plan; NO data, GPU or fork import")
    parser.add_argument("--scale-only", action="store_true", help="Show dataset/window scale; NO training")
    parser.add_argument("--no-eval-attack", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_task-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        if args._run_dir is None or args._task_index < 1:
            parser.error("Internal worker requires --_run-dir and --_task-index")
        return _worker_main(args._run_dir, args._task_index, args.cpu_threads)

    if args.gpu_workers < 1 or args.cpu_threads < 1:
        parser.error("--gpu-workers and --cpu-threads must be >= 1")
    if args.resume and args.run_dir is None:
        parser.error("--resume requires --run-dir")
    if args.iterations < 1 or args.eval_steps < 1 or args.eval_restarts < 1:
        parser.error("iterations, eval-steps and eval-restarts must be positive")
    if not args.train_epsilon or any(not np.isfinite(v) or v <= 0 for v in args.train_epsilon):
        parser.error("train-epsilon must contain positive finite numbers")
    if not np.isfinite(args.eval_epsilon) or args.eval_epsilon <= 0:
        parser.error("eval-epsilon must be positive and finite")
    if (any(not np.isfinite(v) or v <= 0 for v in args.eval_radii)
            or len(set(args.eval_radii)) != len(args.eval_radii)
            or args.eval_epsilon in args.eval_radii):
        parser.error("eval-radii must be distinct positive finite radii, not repeating eval-epsilon")
    if not 0 < args.gamma < 1:
        parser.error("Require 0 < gamma < 1")
    if args.k_min < 1 or args.k_max < args.k_min:
        parser.error("Require 1 <= k-min <= k-max")
    for name, values in (("seeds", args.seeds), ("arms", args.arms),
                         ("train-epsilon", args.train_epsilon)):
        if len(values) != len(set(values)):
            parser.error(f"Duplicate values in --{name}")

    # Exactly one session for each invocation. Clean is not repeated for the second epsilon.
    planned_jobs = [(seed, nominal, arm)
                    for seed in args.seeds
                    for i, nominal in enumerate(args.train_epsilon)
                    for arm in args.arms
                    if i == 0 or arm not in ("clean", "svd_gap_uncapped")]
    print(f"[PLAN] V5 fork: {ROOT}", flush=True)
    print(f"[PLAN] Dataset: {args.data} | output root: {args.out}", flush=True)
    print(f"[PLAN] GPU workers: {args.gpu_workers} (one independent encoder per GPU)", flush=True)
    print(f"[PLAN] {len(planned_jobs)} encoder training jobs; {len(args.seeds)} seeds x "
          f"arms={args.arms} x train epsilons={args.train_epsilon} ({args.epsilon_unit})", flush=True)
    for index, (seed, nominal, arm) in enumerate(planned_jobs, 1):
        print(f"[PLAN] {index:>3}/{len(planned_jobs)} seed={seed} arm={arm} "
              f"train_epsilon={nominal:g} {args.epsilon_unit}", flush=True)
    if args.epsilon_unit == "window":
        print("[WARNING] window epsilon=3 is 3x the median window norm, NOT absolute epsilon 3.", flush=True)
    if args.plan_only:
        print("[PLAN ONLY] No data loaded; no training or evaluation performed.", flush=True)
        return 0
    if not args.scale_only:
        tokens = _gpu_tokens(args.gpu_workers)
        print(f"[GPU PREFLIGHT] {len(tokens)} GPUs available: {tokens}", flush=True)

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
    unit = "w" if args.epsilon_unit == "window" else ""
    evaluation = dict(epsilon=args.eval_epsilon * scale, steps=args.eval_steps,
                      restarts=args.eval_restarts,
                      radii=[[f"{radius:g}{unit}", radius * scale] for radius in args.eval_radii])
    print(f"[SCALE] median centred window norm = {norm:.3f} | training eps = "
          f"{[round(e, 4) for e in train_eps]} | evaluation eps = {evaluation['epsilon']:.4f} "
          f"({args.eval_steps} steps, {args.eval_restarts} restart(s))", flush=True)
    print(f"[SCALE] extra PGD evaluation radii = "
          f"{[round(r, 4) for _, r in evaluation['radii']]}", flush=True)
    print(f"[SCALE] as window fractions: training eps = {[round(e/norm,5) for e in train_eps]} "
          f"| evaluation eps = {evaluation['epsilon'] / norm:.5f}", flush=True)
    print(f"[SCALE] gap g_k p10/p50/p90 = {gaps['gap_p10']:.3f} / "
          f"{gaps['gap_p50']:.3f} / {gaps['gap_p90']:.3f} | k median = {gaps['k_median']:g}",
          flush=True)
    if args.scale_only:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    if args.run_dir is not None:
        out = args.run_dir.resolve()
        if not out.is_relative_to(args.out.resolve()):
            raise RuntimeError("--run-dir MUST be a subdirectory of --out")
        if args.resume:
            if not (out / "run_config.json").is_file():
                raise FileNotFoundError(f"Cannot resume missing run: {out}")
        else:
            out.mkdir(parents=True, exist_ok=False)  # cannot overwrite results
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = Path(tempfile.mkdtemp(prefix=f"{args.data.stem}_{stamp}_", dir=args.out))
    print(f"[START] Experiment: {args.data.stem} | output: {out}", flush=True)

    config = {
        "data_path": str(args.data.resolve()), "fork": str(ROOT.resolve()),
        "cebra_import": str(Path(cebra.__file__).resolve()),
        "runner_file": str(Path(__file__).resolve()),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runner_version": "v5-multigpu-independent-v1",
        "planned_jobs": len(planned_jobs),
        "tasks": [{"seed": int(seed), "nominal": float(nominal),
                   "epsilon": float(nominal * scale), "arm": arm}
                  for seed, nominal, arm in planned_jobs],
        "no_eval_attack": bool(args.no_eval_attack), "spectral": spectral,
        "fork_sha256": {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
                        for rel in ("cebra/solver/spectral_gap.py", "cebra/integrations/sklearn/cebra.py")},
        "arms": args.arms, "seeds": args.seeds, "iterations": args.iterations,
        "window_norm_median": norm, "window_gap": gaps, "epsilon_unit": args.epsilon_unit,
        "evaluation": evaluation,
        "encoder_configs": {f"{arm}@{nominal:g}": encoder_config(arm, args.iterations, eps,
                                                                 spectral["gamma"], spectral["k_min"],
                                                                 spectral["k_max"])
                            for nominal, eps in zip(args.train_epsilon, train_eps)
                            for arm in args.arms},
        "decoder": {"architecture": "Linear -> LayerNorm -> ReLU -> Dropout -> Linear",
                    "epochs": MLP_EPOCHS, "hidden": MLP_HIDDEN,
                    "dropout": MLP_DROP, "learning_rate": MLP_LR},
        "eval_attack": None if args.no_eval_attack else {
            name: vars(c) for name, c in attack_eval_configs(
                cebra.solver.spectral_gap, evaluation["epsilon"], args.eval_steps).items()},
        "shapes": {"x_train": arrays[0].shape, "x_valid": arrays[1].shape,
                   "y_train": arrays[2].shape, "y_valid": arrays[3].shape},
    }
    config = json.loads(json.dumps(config))
    config_path = out / "run_config.json"
    if args.resume:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise RuntimeError("Resume configuration differs: fork, runner, dataset, tasks, "
                               "or evaluation changed. Use a NEW --run-dir.")
        print(f"[RESUME] Matched original plan, fork fingerprint and runner SHA: {out}", flush=True)
    else:
        _atomic_json(config_path, config)
    for folder in ("jobs", "logs"):
        (out / folder).mkdir(exist_ok=True)

    failure_count = _parallel_execute(out, config["tasks"], args.gpu_workers, args.cpu_threads)
    results, missing = _read_finished(out, config["tasks"])
    if missing or failure_count:
        print(f"[PARTIAL] {len(results)}/{len(config['tasks'])} saved, "
              f"{len(missing)} incomplete and {failure_count} failed. Resume with --run-dir "
              f"{out} --resume; inspect {out / 'logs'}", flush=True)
        raise RuntimeError("Some independent GPU workers failed")
    if not (out / "results.json").is_file():
        os.replace(out / "results_so_far.csv", out / "results.csv")
        os.replace(out / "results_so_far.json", out / "results.json")
    summary = summarize(results)
    _atomic_json(out / "summary.json", summary)
    print("\n" + "=" * 75, flush=True)
    print("SUMMARY (mean ± std over seeds)", flush=True)
    for entry in summary:
        msg = " | ".join(
            f"{key[:-5]}={entry[key]:.4f}±{entry[key[:-5] + '_std']:.4f}"
            for key in entry if key.endswith("_mean"))
        print(f"{entry['model']:22s} n={entry['num_seeds']} | {msg}", flush=True)
    print(f"[COMPLETE] Results saved in: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

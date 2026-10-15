"""C-CO12: clean vs PGD vs spectral-gap adversarial CEBRA (Acorn-SVD fork).

Run from anywhere; the script imports the fork from its own directory:
    python -u run_acorn_svd.py --data /path/to/C-CO12.npz

Arms (--arms):
    clean             CEBRA without adversarial updates.
    pgd               Previous project's PGD on the reference (L2, eps=3), bit-exact.
    svd_gap           Spectral-gap subspace attack, per-window budget capped at eps=3.
    svd_gap_uncapped  Same, budget limited only by the spectral gap.
    svd_subspace      Ablation: subspace projection with the fixed budget eps=3.
    svd_gap_budget    Ablation: gap budget without the projection.

For every (seed, arm): fit the encoder, train the same MLP decoder on clean
train embeddings, report validation R2 and, unless --no-eval-attack, R2 under
plain-PGD and spectral-gap attacks on the validation windows. Results are saved
after every arm in a new, unique run directory; no checkpoints are stored.
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

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path("/data/hossein/mm_project/perich_data_valid_final_raw/C-CO12.npz")

EPSILON = 3.0
EPS_STEPS = 10
EPS_ALPHA = EPSILON / 5  # 0.6
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
EVAL_STEPS = 20
EVAL_ALPHA = 2.5 * EPSILON / EVAL_STEPS
EVAL_ATTACK_BATCH = 2048
EVAL_SEED = 1234

ARMS = {
    "clean": dict(training_mode="standard"),
    "pgd": dict(adv_spectral_mode="none"),
    "svd_gap": dict(adv_spectral_mode="gap"),
    "svd_gap_uncapped": dict(adv_spectral_mode="gap",
                             adv_epsilon=UNCAPPED_EPSILON,
                             adv_alpha=UNCAPPED_EPSILON / 5),
    "svd_subspace": dict(adv_spectral_mode="subspace"),
    "svd_gap_budget": dict(adv_spectral_mode="gap_budget"),
}
DEFAULT_ARMS = ("clean", "pgd", "svd_gap", "svd_gap_uncapped")


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


def encoder_config(arm: str, iterations: int) -> dict:
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
        adv_epsilon=EPSILON,
        adv_alpha=EPS_ALPHA,
        adv_steps=EPS_STEPS,
        attack_norm=ATTACK_NORM,
        adv_spectral_gamma=GAMMA,
        adv_spectral_k_min=K_MIN,
        adv_spectral_k_max=K_MAX,
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


def attack_eval_configs(spectral_gap):
    shared = dict(epsilon=EPSILON, alpha=EVAL_ALPHA, steps=EVAL_STEPS, norm="l2")
    return {
        "pgd": spectral_gap.AttackConfig(mode="none", **shared),
        "svd_gap": spectral_gap.AttackConfig(mode="gap", gamma=GAMMA, k_min=K_MIN, k_max=K_MAX,
                                             center=CENTER, **shared),
    }


def robustness(cebra, encoder, decoder, x_valid, y_valid, z_valid):
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
    for name, config in [("clean", None)] + list(attack_eval_configs(spectral_gap).items()):
        seed_all(EVAL_SEED)
        preds, budgets = [], []
        for start in range(0, len(windows), EVAL_ATTACK_BATCH):
            batch = windows[start:start + EVAL_ATTACK_BATCH].to(device)
            target = torch.as_tensor(labels[start:start + len(batch)], device=device)
            if config is not None:
                frame = spectral_gap.spectral_frame(batch, config) if config.spectral else None
                if frame is not None:
                    budgets.append(frame.epsilon.cpu())

                def loss_fn(reference, target=target):
                    return (decoder(model(reference)) - target).square().mean()

                batch = spectral_gap.attack(batch, loss_fn, config, frame)
            with torch.no_grad():
                preds.append(decoder(model(batch)).cpu().numpy())
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
    for key in ("adv_loss", "adv_delta_norm", "adv_eps_mean", "adv_eps_min",
                "adv_gap_limited", "adv_k_mean", "adv_gap_mean"):
        if log.get(key):
            stats[f"train_{key}"] = float(np.mean(log[key]))
    diagnostics = log.get("adv_diagnostics", [])
    for key in ("sv_err_max", "left_subspace_err_max", "right_subspace_err_max",
                "spectral_drift_mean", "delta_over_gap_max"):
        values = [d[key] for d in diagnostics if np.isfinite(d.get(key, np.nan))]
        if values:
            stats[f"diag_{key}"] = float(np.max(values) if key.endswith("_max") else np.mean(values))
    return stats


def run_model(cebra, arm, seed, arrays, iterations, eval_attack) -> dict:
    x_train, x_valid, y_train, y_valid = arrays
    encoder = decoder = z_train = z_valid = None
    arm_start = time.perf_counter()
    try:
        seed_all(seed)  # Same initialization and sampling seed in each arm.
        print(f"\n{'=' * 75}", flush=True)
        print(f"[START] {arm} | seed={seed} | eps={EPSILON} | {iterations} CEBRA iterations",
              flush=True)
        encoder = cebra.CEBRA(**encoder_config(arm, iterations))
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
            "model": arm, "seed": seed, "epsilon": EPSILON,
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
        if eval_attack:
            eval_start = time.perf_counter()
            row.update(robustness(cebra, encoder, decoder, x_valid, y_valid, z_valid))
            row["eval_attack_seconds"] = float(time.perf_counter() - eval_start)
            print(f"[ROBUST] {arm} seed={seed}: R2 clean windows = {row['r2_clean']:.4f} | "
                  f"PGD = {row['r2_pgd']:.4f} | SVD-gap = {row['r2_svd_gap']:.4f}", flush=True)
        row["total_seconds"] = float(time.perf_counter() - arm_start)
        return row
    finally:
        del encoder, decoder, z_train, z_valid
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


SCALAR_COLUMNS = (
    "model", "seed", "spectral_mode", "epsilon", "valid_mean_r2", "r2_clean", "r2_pgd",
    "r2_svd_gap", "eval_eps_svd_gap_mean", "train_adv_eps_mean", "train_adv_gap_limited",
    "train_adv_k_mean", "train_adv_delta_norm", "train_adv_loss", "diag_sv_err_max",
    "diag_left_subspace_err_max", "diag_right_subspace_err_max", "diag_spectral_drift_mean",
    "diag_delta_over_gap_max", "seconds_per_iteration", "encoder_seconds", "decoder_seconds",
    "total_seconds",
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


def summarize(results: list) -> list:
    summary = []
    for arm in dict.fromkeys(row["model"] for row in results):
        rows = [row for row in results if row["model"] == arm]
        entry = {"model": arm, "num_seeds": len(rows)}
        for key in ("valid_mean_r2", "r2_pgd", "r2_svd_gap", "seconds_per_iteration"):
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
    parser.add_argument("--out", type=Path, default=ROOT / "ACORN_SVD_RESULTS")
    parser.add_argument("--no-eval-attack", action="store_true",
                        help="Skip the held-out attacked-R2 evaluation.")
    args = parser.parse_args()

    # mkdtemp uses atomic creation; runs cannot overwrite one another.
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(tempfile.mkdtemp(prefix=f"{args.data.stem}_{stamp}_", dir=args.out))
    print(f"[START] New experiment: {args.data.stem} | output: {out}", flush=True)
    print(f"[START] Arms: {args.arms} | seeds: {args.seeds} | eps: {EPSILON}", flush=True)
    cebra = load_fork()
    print(f"[FORK] {cebra.__file__}", flush=True)
    arrays = load_data(args.data)
    print(f"[DATA] Train X={arrays[0].shape}, Y={arrays[2].shape}", flush=True)
    print(f"[DATA] Valid X={arrays[1].shape}, Y={arrays[3].shape}", flush=True)

    config = {
        "data_path": str(args.data), "fork": str(ROOT), "cebra_import": str(cebra.__file__),
        "arms": args.arms, "seeds": args.seeds, "iterations": args.iterations,
        "encoder_configs": {arm: encoder_config(arm, args.iterations) for arm in args.arms},
        "decoder": {"architecture": "Linear -> LayerNorm -> ReLU -> Dropout -> Linear",
                    "epochs": MLP_EPOCHS, "hidden": MLP_HIDDEN, "dropout": MLP_DROP,
                    "learning_rate": MLP_LR},
        "eval_attack": None if args.no_eval_attack else {
            name: vars(c) for name, c in attack_eval_configs(cebra.solver.spectral_gap).items()},
        "shapes": {"x_train": arrays[0].shape, "x_valid": arrays[1].shape,
                   "y_train": arrays[2].shape, "y_valid": arrays[3].shape},
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = []
    try:
        for seed in args.seeds:  # Seed-major: every arm of a seed finishes first.
            for arm in args.arms:
                results.append(run_model(cebra, arm, seed, arrays, args.iterations,
                                         not args.no_eval_attack))
                save_results(out, results)  # Save right after each completed arm.
                print(f"[SAVED] {len(results)}/{len(args.arms) * len(args.seeds)} results -> "
                      f"{out / 'results_so_far.csv'}", flush=True)
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
        print(f"{entry['model']:18s} n={entry['num_seeds']} | {text}", flush=True)
    print(f"[COMPLETE] Results saved in: {out}", flush=True)


if __name__ == "__main__":
    main()

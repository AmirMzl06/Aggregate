"""C-CO12: PGD epsilon=3 vs SVD Gate vs SVD Guided.

Place this script next to the CEBRA-SVD fork directory and run:
    python -u run_CCO12_CEBRA_SVD_verbose_results_v3.py

Show all CEBRA progress, and save R2 metrics as soon as each model finishes.
Results go to a new, unique run directory; no checkpoints or predictions are saved.
"""

from pathlib import Path
import csv
import gc
import json
import inspect
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
import random
import sys

import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score

# Prevent the local fork import from generating __pycache__ files.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
CEBRA_DIR = ROOT / "CEBRA-SVD"
DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
SESSION = "C-CO12"
DATA_PATH = DATA_DIR / f"{SESSION}.npz"
# New name, separate from all prior run output folders.
OUT_ROOT = ROOT / "CCO12_SVD_VERBOSE_RESULTS_V3"

SEED = 42
EPSILON = 3.0
EPS_STEPS = 10
EPS_ALPHA = EPSILON / 5  # 0.6; relevant in continuous mode.
ATTACK_NORM = "l2"
INPUT_DOMAIN = "counts"  # Raw nonnegative integer spike counts.

# SVD parameters: the same in both spectral arms.
SVD_TAU = 0.05
SVD_WEIGHT = 1.0
SVD_CENTER = True

LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 3000
TEMPERATURE = 0.4
ARCH = "offset36-model-more-dropout"
DEVICE = "cuda_if_available"

# EXACT decoder architecture and hyperparameters from the supplied idea-1 runner.
MLP_EPOCHS = 2500
MLP_HIDDEN = 64
MLP_DROP = 0.4
MLP_LR = 1e-3
EVAL_BATCH_SIZE = 8192


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_fork():
    if not (CEBRA_DIR / "cebra" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"Expected fork directory: {CEBRA_DIR}\n"
            "Put this runner next to the folder named CEBRA-SVD."
        )
    # Avoid accidentally using an installed/unrelated version of CEBRA.
    for key in list(sys.modules):
        if key == "cebra" or key.startswith("cebra."):
            del sys.modules[key]
    sys.path.insert(0, str(CEBRA_DIR.resolve()))
    import cebra
    if not Path(cebra.__file__).resolve().is_relative_to(CEBRA_DIR.resolve()):
        raise RuntimeError(f"Incorrect CEBRA package imported: {cebra.__file__}")
    params = inspect.signature(cebra.CEBRA.__init__).parameters
    required = {"adv_svd_mode", "adv_svd_tau", "adv_svd_weight", "adv_input_domain"}
    missing = required - params.keys()
    if missing:
        raise RuntimeError(f"CEBRA-SVD fork lacks expected parameters: {sorted(missing)}")
    return cebra


def load_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Session file not found: {DATA_PATH}")
    with np.load(DATA_PATH, allow_pickle=False) as f:
        x_train, x_valid, y_train, y_valid = (
            np.asarray(f[key], dtype=np.float32)
            for key in ("train_data", "valid_data", "train_label", "valid_label")
        )

    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_valid.ndim == 1:
        y_valid = y_valid[:, None]

    for key, arr in (('train_data', x_train), ('valid_data', x_valid),
                     ('train_label', y_train), ('valid_label', y_valid)):
        if arr.ndim != 2 or not all(arr.shape) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid {key} shape or NaN/Inf: {arr.shape}")
    if x_train.shape[1] != x_valid.shape[1] or y_train.shape[1] != y_valid.shape[1]:
        raise ValueError("Train/validation feature or label dimensions differ")
    if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
        raise ValueError("Number of labels and neural samples must match")
    if min(len(x_train), len(x_valid)) < 36:
        raise ValueError("offset36 encoder requires >=36 time bins per split")
    if INPUT_DOMAIN == "counts":
        for name, arr in (("train_data", x_train), ("valid_data", x_valid)):
            if np.any(arr < 0) or not np.array_equal(arr, np.rint(arr)):
                raise ValueError(
                    f"{name} contains non-integer or negative values. "
                    "INPUT_DOMAIN='counts' needs raw binned spike counts. "
                    "If the data were normalized/smoothed, use 'continuous' consistently "
                    "for all three methods instead."
                )
    return tuple(np.ascontiguousarray(a) for a in (x_train, x_valid, y_train, y_valid))


def encoder_config(svd_mode: str) -> dict:
    return dict(
        model_architecture=ARCH,
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        time_offsets=1,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        learning_rate=3e-4,
        temperature_mode="constant",
        training_mode="adversarial",
        adv_epsilon=EPSILON,
        adv_alpha=EPS_ALPHA,
        adv_steps=EPS_STEPS,
        attack_norm=ATTACK_NORM,
        adv_input_domain=INPUT_DOMAIN,
        adv_svd_mode=svd_mode,
        adv_svd_tau=SVD_TAU,
        adv_svd_weight=SVD_WEIGHT,
        adv_svd_center=SVD_CENTER,
        adv_epsilon_factors=(1.0,),
        adv_best_iterate=False,  # Plain PGD baseline; gate/guided keep best feasible internally.
        adv_rate_tau=None,
        device=DEVICE,
        verbose=True,  # CEBRA training progress and iteration/loss logs.
    )


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
    seed_all(seed)  # Same decoder seed in all three arms.
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
            print(
                f"[DECODER] epoch {epoch + 1}/{MLP_EPOCHS} "
                f"| train_MSE={loss.item():.6f} "
                f"| elapsed={time.perf_counter() - decoder_start:.1f}s",
                flush=True,
            )
    return decoder


def predict_decoder(decoder, embeddings):
    decoder.eval()  # Disable Dropout for R2 evaluation.
    device = next(decoder.parameters()).device
    preds = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), EVAL_BATCH_SIZE):
            z = torch.as_tensor(
                embeddings[start:start + EVAL_BATCH_SIZE],
                dtype=torch.float32, device=device,
            )
            preds.append(decoder(z).cpu().numpy())
    return np.concatenate(preds, axis=0)


LABELS = {"none": "PGD_EPS3", "gate": "SVD_GATE_EPS3", "guided": "SVD_GUIDED_EPS3"}
MODES = ("none", "gate", "guided")


def run_model(cebra, mode: str, arrays) -> dict:
    x_train, x_valid, y_train, y_valid = arrays
    encoder = decoder = z_train = z_valid = None
    arm_start = time.perf_counter()
    label = LABELS[mode]
    try:
        seed_all(SEED)  # Same initialization and sampling seed in each arm.
        print(f"\n{'=' * 75}", flush=True)
        print(f"[START] {label} | seed={SEED} | eps={EPSILON} | {MAX_ITER} CEBRA iterations", flush=True)
        print(f"[ENCODER] {label}: training started", flush=True)
        encoder = cebra.CEBRA(**encoder_config(mode))
        encoder.fit(x_train, y_train)
        encoder_seconds = time.perf_counter() - arm_start
        print(f"[ENCODER] {label}: completed in {encoder_seconds / 3600:.2f} hours", flush=True)

        print(f"[EMBEDDING] {label}: transforming train / validation", flush=True)
        z_train = np.asarray(encoder.transform(x_train), dtype=np.float32)
        z_valid = np.asarray(encoder.transform(x_valid), dtype=np.float32)
        if (len(z_train) != len(x_train) or len(z_valid) != len(x_valid)
                or not np.isfinite(z_train).all() or not np.isfinite(z_valid).all()):
            raise ValueError(f"Bad embeddings or alignment for mode={mode}")

        print(f"[DECODER] {label}: training started", flush=True)
        decoder_start = time.perf_counter()
        decoder = train_decoder(z_train, y_train, SEED + 100_000,
                                torch.device(encoder.device_))
        decoder_seconds = time.perf_counter() - decoder_start
        print(f"[DECODER] {label}: completed in {decoder_seconds / 60:.1f} minutes", flush=True)

        predictions = predict_decoder(decoder, z_valid)
        if predictions.shape != y_valid.shape or not np.isfinite(predictions).all():
            raise ValueError(f"Bad decoder prediction for mode={mode}")
        per_label = np.atleast_1d(r2_score(y_valid, predictions, multioutput="raw_values"))
        if not np.isfinite(per_label).all():
            raise ValueError(f"Nonfinite R2 score for mode={mode}: {per_label}")
        row = {
            "model": label, "svd_mode": mode, "seed": SEED,
            "epsilon": EPSILON, "valid_mean_r2": float(per_label.mean()),
            "valid_r2_per_label": [float(v) for v in per_label],
            "encoder_seconds": float(encoder_seconds),
            "decoder_seconds": float(decoder_seconds),
            "total_seconds": float(time.perf_counter() - arm_start),
        }
        print(f"[RESULT] {label}: validation mean R2 = {row['valid_mean_r2']:.6f}", flush=True)
        print(f"[RESULT] {label}: R2 per label = {row['valid_r2_per_label']}", flush=True)
        return row
    finally:
        del encoder, decoder, z_train, z_valid
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_results(out: Path, results: list[dict]):
    """Persist all completed arms atomically, even if a later arm is slow/fails."""
    json_path = out / "results_so_far.json"
    temp_json = out / ".results_so_far.json.tmp"
    temp_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temp_json, json_path)

    csv_path = out / "results_so_far.csv"
    temp_csv = out / ".results_so_far.csv.tmp"
    num_labels = len(results[0]["valid_r2_per_label"])
    with temp_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "svd_mode", "seed", "epsilon", "valid_mean_r2"]
            + [f"r2_label_{i}" for i in range(num_labels)]
            + ["encoder_seconds", "decoder_seconds", "total_seconds"]
        )
        for row in results:
            writer.writerow([
                row["model"], row["svd_mode"], row["seed"], row["epsilon"],
                row["valid_mean_r2"], *row["valid_r2_per_label"],
                row["encoder_seconds"], row["decoder_seconds"], row["total_seconds"],
            ])
    os.replace(temp_csv, csv_path)


def main():
    # mkdtemp uses atomic creation; runs cannot overwrite one another.
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(tempfile.mkdtemp(prefix=f"CCO12_seed{SEED}_{stamp}_", dir=OUT_ROOT))
    print(f"[START] New experiment: {SESSION} | output: {out}", flush=True)
    print(f"[START] Modes: {MODES} | epsilon: {EPSILON} | seed: {SEED}", flush=True)
    cebra = load_fork()
    print(f"[FORK] {cebra.__file__}", flush=True)
    arrays = load_data()
    print(f"[DATA] Train X={arrays[0].shape}, Y={arrays[2].shape}", flush=True)
    print(f"[DATA] Valid X={arrays[1].shape}, Y={arrays[3].shape}", flush=True)

    config = {
        "session": SESSION, "data_path": str(DATA_PATH),
        "fork": str(CEBRA_DIR), "cebra_import": str(cebra.__file__),
        "modes": list(MODES), "seed": SEED,
        "encoder_configs": {mode: encoder_config(mode) for mode in MODES},
        "decoder": {
            "architecture": "Linear -> LayerNorm -> ReLU -> Dropout -> Linear",
            "epochs": MLP_EPOCHS, "hidden": MLP_HIDDEN,
            "dropout": MLP_DROP, "learning_rate": MLP_LR,
        },
        "shapes": {"x_train": arrays[0].shape, "x_valid": arrays[1].shape,
                   "y_train": arrays[2].shape, "y_valid": arrays[3].shape},
    }
    (out / "run_config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False), encoding="utf-8"
    )

    results = []
    try:
        for mode in MODES:
            row = run_model(cebra, mode, arrays)
            results.append(row)
            save_results(out, results)  # Save right after each completed decoder.
            print(f"[SAVED] {len(results)}/{len(MODES)} results -> {out / 'results_so_far.csv'}", flush=True)
    except BaseException as exc:
        print(f"[INTERRUPTED] {type(exc).__name__}: {exc}", flush=True)
        print(f"[RECOVERY] Completed models' results (if any) remain in: {out}", flush=True)
        raise

    # Canonical final files; partial files remain available from earlier completions.
    shutil.copyfile(out / "results_so_far.csv", out / "results.csv")
    shutil.copyfile(out / "results_so_far.json", out / "results.json")
    print("\n" + "=" * 75, flush=True)
    print("FINAL VALIDATION R2 (all label columns)", flush=True)
    for row in results:
        print(f"{row['model']}: R2 = {row['valid_mean_r2']:.6f} | "
              f"per_label = {row['valid_r2_per_label']}", flush=True)
    print(f"[COMPLETE] Results saved in: {out}", flush=True)


if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """Acorn-dprime Exp3: 12 independent models per seed on 4 CUDA GPUs.

# Run this script once per seed (42, 43, 44), using a DIFFERENT persistent
# --output-dir for each seed: total 12 x 3 = 36 models.

# Models: CLEAN, PGD_L2_EPS3, MADRY_D{0,1,3,8},
# TRADES_D{0,1,3,8}, TRADES_ISO_D{1,3}.

# D0 means epsilon=1e-6, not literally zero. TRADES_D0 thus probes
# (dropout) consistency, not an exact no-attack control.

# Only neural x_train is used in CEBRA.fit; behavioral labels are used
# for the downstream MLP decoder and validation R2 only.

# Every completed model writes its own atomic record + dprime report,
# then refreshes results.csv/results.txt on a mounted persistent home PVC.
# Resume reuses successful records with the same manifest; never reuse
# this output directory with a changed fork/script/seed.
# """
# from __future__ import annotations

# import argparse
# import csv
# import json
# from datetime import datetime, timezone
# import hashlib
# import contextlib
# from dataclasses import dataclass, field
# import gc
# import inspect
# import os
# from pathlib import Path
# import random
# import queue
# import subprocess
# import tempfile
# import threading
# import time
# import traceback
# import sys
# from typing import Any, Dict, List, Optional, Tuple

# sys.dont_write_bytecode = True

# import numpy as np
# import torch
# from torch import nn
# from sklearn.metrics import r2_score

# ROOT = Path(__file__).resolve().parent
# DEFAULT_FORK_DIR = ROOT / "Acorn-dprime"
# DEFAULT_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
# DEFAULT_SESSION = "C-CO12"
# DEFAULT_RESULTS_DIR = ROOT / "DPRIME_CCO12_EXP3_SEED42"

# SEED = 42
# PGD_EPS = 3.0
# ATTACK_STEPS = 10
# PGD_ALPHA = PGD_EPS / 5.0

# LATENT_DIM = 64
# HIDDEN = 64
# BATCH_SIZE = 2048
# MAX_ITER = 3000
# TEMPERATURE = 0.4
# ARCH = "offset36-model-more-dropout"
# DEVICE = "cuda_if_available"

# MLP_EPOCHS = 2500
# MLP_HIDDEN = 64
# MLP_DROP = 0.4
# MLP_LR = 1e-3
# EVAL_BATCH_SIZE = 8192


# @dataclass(frozen=True)
# class Experiment:
#     name: str
#     overrides: Dict[str, Any] = field(default_factory=dict)
#     core: bool = False


# def experiments() -> List[Experiment]:
#     """Exp3: 12 arms/seed to disentangle TRADES, dropout consistency,
#     attack budget, and covariance-aware versus isotropic d-prime geometry.

#     The nearly-zero control uses d-prime epsilon=1e-6 (not exactly zero).
#     All d-prime arms share APGD, 1 restart, no guards, no warmup/curriculum.
#     The clean baseline uses standard CEBRA (time-only). 
#     """
#     common = dict(
#         adv_epsilon_mode="dprime",
#         adv_dprime_layer="stationary",
#         adv_dprime_residuals="whole",
#         adv_dprime_spatial="ledoit_wolf",
#         adv_dprime_temporal="welch",
#         adv_dprime_optimizer="apgd",
#         adv_dprime_restarts=1,
#         adv_dprime_instance_adaptive=False,
#         adv_dprime_guards=(),
#         adv_dprime_guard_mode="off",
#         adv_dprime_warmup=0.0,
#         adv_dprime_curriculum=0.0,
#         adv_dprime_trades_beta=6.0,
#     )

#     def arm(name: str, **kw: Any) -> Experiment:
#         return Experiment(name=name, overrides={**common, **kw}, core=True)

#     result = [
#         Experiment("CLEAN", {"training_mode": "standard"}, True),
#         Experiment("PGD_L2_EPS3", {
#             "training_mode": "adversarial",
#             "adv_epsilon_mode": "constant",
#             "attack_norm": "l2",
#         }, True),
#     ]
#     for objective in ("madry", "trades"):
#         for tag, budget in (("0", 1e-6), ("1", 1.0), ("3", 3.0), ("8", 8.0)):
#             result.append(arm(f"{objective.upper()}_D{tag}",
#                               adv_dprime_objective=objective,
#                               adv_dprime_epsilon=budget))
#     for tag, budget in (("1", 1.0), ("3", 3.0)):
#         result.append(arm(f"TRADES_ISO_D{tag}",
#                           adv_dprime_objective="trades",
#                           adv_dprime_epsilon=budget,
#                           adv_dprime_spatial="isotropic",
#                           adv_dprime_temporal="white"))
#     return result


# def seed_all(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True


# def load_fork(fork_dir: Path):
#     package = fork_dir / "cebra" / "__init__.py"
#     if not package.is_file():
#         raise FileNotFoundError(
#             f"Expected fork at {fork_dir}. Place this runner next to Acorn-dprime "
#             "or set --fork-dir PATH."
#         )
#     for key in list(sys.modules):
#         if key == "cebra" or key.startswith("cebra."):
#             del sys.modules[key]
#     sys.path.insert(0, str(fork_dir.resolve()))
#     import cebra

#     actual = Path(cebra.__file__).resolve()
#     if not actual.is_relative_to(fork_dir.resolve()):
#         raise RuntimeError(f"Wrong CEBRA imported: {actual}")
#     params = inspect.signature(cebra.CEBRA.__init__).parameters
#     required = {
#         "training_mode", "adv_epsilon", "adv_alpha", "adv_steps", "attack_norm",
#         "adv_epsilon_mode", "adv_dprime_epsilon", "adv_dprime_layer",
#         "adv_dprime_optimizer", "adv_dprime_instance_adaptive", "adv_dprime_temporal",
#         "adv_dprime_spatial", "adv_dprime_guards", "adv_dprime_objective",
#     }
#     missing = sorted(required - set(params))
#     if missing:
#         raise RuntimeError(f"This fork lacks Acorn-dprime parameters: {missing}")
#     from cebra.dprime.noise import SPATIAL_ESTIMATORS
#     if "isotropic" not in SPATIAL_ESTIMATORS:
#         raise RuntimeError("This is an OLD Acorn-dprime fork; Exp3 needs isotropic covariance."
#                            " Extract the newly supplied Acorn-dprime ZIP.")
#     return cebra


# def load_data(data_path: Path):
#     if not data_path.is_file():
#         raise FileNotFoundError(f"No session file at: {data_path}")
#     with np.load(data_path, allow_pickle=False) as f:
#         x_train, x_valid, y_train, y_valid = (
#             np.asarray(f[k], dtype=np.float32)
#             for k in ("train_data", "valid_data", "train_label", "valid_label")
#         )

#     if y_train.ndim == 1:
#         y_train = y_train[:, None]
#     if y_valid.ndim == 1:
#         y_valid = y_valid[:, None]
#     for key, arr in (("train_data", x_train), ("valid_data", x_valid),
#                      ("train_label", y_train), ("valid_label", y_valid)):
#         if arr.ndim != 2 or 0 in arr.shape or not np.isfinite(arr).all():
#             raise ValueError(f"Invalid shape / nonfinite {key}: {arr.shape}")
#     if (x_train.shape[1] != x_valid.shape[1]
#             or y_train.shape[1] != y_valid.shape[1]):
#         raise ValueError("Train/valid feature or label dimensions differ")
#     if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
#         raise ValueError("Neural data and decoder labels have different lengths")
#     if min(len(x_train), len(x_valid)) < 36:
#         raise ValueError("The offset36 model needs at least 36 bins in each split")
#     for name, x in (("train_data", x_train), ("valid_data", x_valid)):
#         if np.any(x < 0) or not np.array_equal(x, np.rint(x)):
#             raise ValueError(f"{name} must be RAW nonnegative integer spike counts")
#     return tuple(np.ascontiguousarray(a) for a in (x_train, x_valid, y_train, y_valid))


# def encoder_config(exp: Experiment, iterations: int, batch_size: int,
#                    device: str, seed: int, pgd_eps: float) -> Dict[str, Any]:
#     overrides = dict(exp.overrides)
#     return {
#         "model_architecture": ARCH,
#         "batch_size": batch_size,
#         "temperature": TEMPERATURE,
#         "time_offsets": 1,
#         "max_iterations": iterations,
#         "output_dimension": LATENT_DIM,
#         "num_hidden_units": HIDDEN,
#         "learning_rate": 3e-4,
#         "temperature_mode": "constant",
#         "training_mode": "adversarial",
#         "adv_epsilon": pgd_eps,
#         "adv_alpha": pgd_eps / 5.0,
#         "adv_steps": ATTACK_STEPS,
#         "attack_norm": "l2",   # A placeholder ignored by d' training.
#         "adv_dprime_seed": seed,
#         "device": device,
#         "verbose": False,
#         **overrides,
#     }


# class TwoLayerMLP(nn.Module):
#     def __init__(self, dim: int, out: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, MLP_HIDDEN),
#             nn.LayerNorm(MLP_HIDDEN),
#             nn.ReLU(),
#             nn.Dropout(MLP_DROP),
#             nn.Linear(MLP_HIDDEN, out),
#         )

#     def forward(self, x):
#         return self.net(x)


# def train_decoder(z_train, y_train, seed: int, device, epochs: int):
#     seed_all(seed)
#     decoder = TwoLayerMLP(z_train.shape[1], y_train.shape[1]).to(device)
#     opt = torch.optim.Adam(decoder.parameters(), lr=MLP_LR)
#     criterion = nn.MSELoss()
#     z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
#     y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
#     decoder.train()
#     for epoch in range(epochs):
#         opt.zero_grad(set_to_none=True)
#         loss = criterion(decoder(z), y)
#         if not torch.isfinite(loss).item():
#             raise FloatingPointError(f"Nonfinite decoder loss at epoch {epoch + 1}")
#         loss.backward()
#         opt.step()
#     return decoder


# def predict_decoder(decoder, embeddings):
#     decoder.eval()
#     device = next(decoder.parameters()).device
#     preds = []
#     with torch.inference_mode():
#         for first in range(0, len(embeddings), EVAL_BATCH_SIZE):
#             z = torch.as_tensor(embeddings[first:first + EVAL_BATCH_SIZE],
#                                 dtype=torch.float32, device=device)
#             preds.append(decoder(z).cpu().numpy())
#     return np.concatenate(preds, axis=0)


# def epsilon_summary(report: Optional[Dict[str, Any]]) -> str:
#     if not report:
#         return "no-dprime-report"
#     e = report.get("epsilon", {})
#     nn = report.get("eps_nn", {})
#     t = report.get("training", {})
#     dist = t.get("epsilon_per_reference", {}) or {}
#     return (f"eps_dist={dist} "
#             f"eps_star={e.get('eps_star')} "
#             f"source={e.get('source')} "
#             f"eps_NN={nn.get('value')} "
#             f"NN_binding={t.get('eps_nn_binding_rate')} "
#             f"mean_dprime={t.get('mean_dprime')} "
#             f"final_epsilon={t.get('final_epsilon')} "
#             f"warmup_steps={t.get('warmup_steps')} "
#             f"curriculum_steps={t.get('curriculum_steps')}")


# def run_one(cebra, exp: Experiment, arrays, args) -> Tuple[float, Optional[Dict[str, Any]]]:
#     x_train, x_valid, y_train, y_valid = arrays
#     encoder = decoder = z_train = z_valid = None
#     report = None
#     try:
#         seed_all(args.seed)
#         encoder = cebra.CEBRA(**encoder_config(exp, args.max_iterations, args.batch_size,
#                                                args.device, args.seed, args.epsilon))
#         # TIME-ONLY training: y_train is deliberately NOT passed here.
#         encoder.fit(x_train)
#         report = getattr(encoder, "dprime_report_", None)
#         z_train = np.asarray(encoder.transform(x_train), dtype=np.float32)
#         z_valid = np.asarray(encoder.transform(x_valid), dtype=np.float32)
#         if (len(z_train) != len(x_train) or len(z_valid) != len(x_valid)
#                 or not np.isfinite(z_train).all() or not np.isfinite(z_valid).all()):
#             raise ValueError(f"Bad embedding shape / alignment for {exp.name}")
#         decoder = train_decoder(z_train, y_train, args.seed + 100_000,
#                                 torch.device(encoder.device_), args.mlp_epochs)
#         predictions = predict_decoder(decoder, z_valid)
#         if predictions.shape != y_valid.shape or not np.isfinite(predictions).all():
#             raise ValueError(f"Bad decoder predictions for {exp.name}")
#         value = float(np.atleast_1d(r2_score(y_valid, predictions,
#                                              multioutput="raw_values")).mean())
#         if not np.isfinite(value):
#             raise ValueError("R2 is nonfinite; check validation label variability")
#         return value, report
#     finally:
#         del encoder, decoder, z_train, z_valid
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()


# def build_parser() -> argparse.ArgumentParser:
#     p = argparse.ArgumentParser(description=__doc__,
#                                 formatter_class=argparse.RawDescriptionHelpFormatter)
#     p.add_argument("--fork-dir", type=Path, default=DEFAULT_FORK_DIR)
#     p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
#     p.add_argument("--session", default=DEFAULT_SESSION)
#     p.add_argument("--suite", choices=("all", "core"), default="all")
#     p.add_argument("--only", nargs="+", help="Run named arms; use --list to inspect names")
#     p.add_argument("--list", action="store_true", help="List all experiments and exit")
#     p.add_argument("--dry-run", action="store_true", help="List selected arms without fitting")
#     p.add_argument("--seed", type=int, default=SEED)
#     p.add_argument("--epsilon", type=float, default=PGD_EPS,
#                    help="Original input-space PGD L2 baseline epsilon (default 3)")
#     p.add_argument("--max-iterations", type=int, default=MAX_ITER)
#     p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
#     p.add_argument("--mlp-epochs", type=int, default=MLP_EPOCHS)
#     p.add_argument("--device", default=DEVICE)
#     p.add_argument("--report-eps", action="store_true",
#                    help="Print dprime calibration diagnostics to stderr after each run")
#     p.add_argument("--gpu-workers", type=int, default=4,
#                    help="Number of independently allocated GPUs to use (default 4)")
#     p.add_argument("--cpu-threads", type=int, default=2,
#                    help="CPU threads per GPU subprocess (default 2)")
#     p.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR,
#                    help="PERSISTENT results directory. MUST reside on a shared/host volume")
#     p.add_argument("--resume", action="store_true",
#                    help="Skip successful experiment records, retry interrupted/failed runs")
#     p.add_argument("--continue-on-error", action="store_true",
#                    help="Kept for CLI compatibility; the 4-GPU suite always records each error and continues")
#     # Internal worker arguments; not intended for direct use.
#     p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
#     p.add_argument("--_experiment", default=None, help=argparse.SUPPRESS)
#     p.add_argument("--_gpu_slot", type=int, default=-1, help=argparse.SUPPRESS)
#     return p


# def now_utc() -> str:
#     return datetime.now(timezone.utc).isoformat(timespec="seconds")


# def atomic_write_text(path: Path, content: str) -> None:
#     """Commit one entire file; readers see either old or new version."""
#     path.parent.mkdir(parents=True, exist_ok=True)
#     fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
#     try:
#         with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
#             f.write(content)
#             f.flush()
#             os.fsync(f.fileno())
#         os.replace(temp_path, path)
#     finally:
#         if os.path.exists(temp_path):
#             os.unlink(temp_path)


# def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
#     atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2,
#                                        allow_nan=False, sort_keys=True) + "\n")


# def record_path(root: Path, name: str) -> Path:
#     if not name.replace("_", "").isalnum():
#         raise ValueError(f"Unsafe experiment name: {name!r}")
#     return root / "records" / f"{name}.json"


# def read_record(root: Path, name: str):
#     path = record_path(root, name)
#     if not path.is_file():
#         return None
#     with path.open("r", encoding="utf-8") as f:
#         obj = json.load(f)
#     if obj.get("experiment") != name:
#         raise RuntimeError(f"Mismatched record in {path}")
#     return obj


# def write_snapshots(root: Path, selected: List[Experiment]) -> None:
#     """Consolidate atomic per-arm records into CSV and readable text."""
#     import io
#     rows = []
#     summaries = []
#     for exp in selected:
#         record = read_record(root, exp.name)
#         if record is None:
#             continue
#         rows.append({
#             "experiment": exp.name,
#             "status": record.get("status", "unknown"),
#             "r2": record.get("r2", ""),
#             "gpu_slot": record.get("gpu_slot", ""),
#             "cuda_visible_devices": record.get("cuda_visible_devices", ""),
#             "elapsed_seconds": record.get("elapsed_seconds", ""),
#             "finished_utc": record.get("finished_utc", ""),
#             "epsilon_summary": record.get("epsilon_summary", ""),
#             "error": record.get("error", ""),
#         })
#         if record.get("status") == "success":
#             summaries.append(f"{exp.name}: R2 = {record['r2']:.6f}"
#                              f" | GPU slot {record.get('gpu_slot')}"
#                              f" | elapsed {record.get('elapsed_seconds',0):.1f}s")
#         else:
#             summaries.append(f"{exp.name}: ERROR: {record.get('error')}")
#     stream = io.StringIO()
#     fields = ["experiment", "status", "r2", "gpu_slot", "cuda_visible_devices",
#               "elapsed_seconds", "finished_utc", "epsilon_summary", "error"]
#     writer = csv.DictWriter(stream, fieldnames=fields)
#     writer.writeheader()
#     writer.writerows(rows)
#     atomic_write_text(root / "results.csv", stream.getvalue())
#     atomic_write_text(root / "results.txt", "\n".join(summaries) + ("\n" if summaries else ""))


# def selected_experiments(args):
#     all_exp = experiments()
#     names = {e.name for e in all_exp}
#     if args.only:
#         unknown = sorted(set(args.only) - names)
#         if unknown:
#             raise ValueError(f"Unknown experiments: {unknown}; inspect --list")
#         return [e for e in all_exp if e.name in set(args.only)]
#     if args.suite == "core":
#         return [e for e in all_exp if e.core]
#     return all_exp


# def config_manifest(args, selected: List[Experiment]) -> dict:
#     """Prevent accidentally mixing incomparable R2 values across restarts."""
#     data_path = (args.data_dir / f"{args.session}.npz").resolve()
#     return json.loads(json.dumps({
#         "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
#         "fork_dir": str(args.fork_dir.resolve()),
#         "fork_fingerprints": {
#             rel: hashlib.sha256((args.fork_dir / rel).read_bytes()).hexdigest()
#             for rel in ("cebra/distributions/base.py", "cebra/dprime/training.py",
#                         "cebra/dprime/noise.py", "cebra/dprime/attacks.py",
#                         "cebra/integrations/sklearn/cebra.py")
#         },
#         "data_path": str(data_path),
#         "session": args.session,
#         "seed": args.seed,
#         "epsilon": args.epsilon,
#         "batch_size": args.batch_size,
#         "max_iterations": args.max_iterations,
#         "mlp_epochs": args.mlp_epochs,
#         "architecture": ARCH,
#         "latent_dim": LATENT_DIM,
#         "hidden": HIDDEN,
#         "temperature": TEMPERATURE,
#         "attack_steps": ATTACK_STEPS,
#         "experiments": [{"name": exp.name, "overrides": exp.overrides}
#                         for exp in selected],
#     }, sort_keys=True))


# def gpu_tokens(workers: int) -> list[str]:
#     if workers < 1:
#         raise ValueError("--gpu-workers must be at least 1")
#     if not torch.cuda.is_available():
#         raise RuntimeError("No CUDA GPU available in this RunAI container")
#     count = torch.cuda.device_count()
#     if count < workers:
#         raise RuntimeError(f"Requested {workers} GPU workers but only {count} CUDA devices visible. "
#                            "Submit RunAI job with --gpu 4 and check GPU allocation.")
#     visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
#     if visibility and visibility != "-1":
#         tokens = [s.strip() for s in visibility.split(",") if s.strip()]
#         if len(tokens) < workers:
#             raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visibility!r} has fewer than {workers} entries")
#         return tokens[:workers]
#     return [str(i) for i in range(workers)]


# def worker_run(args, exp: Experiment) -> int:
#     """Executed INSIDE subprocess already masked to one physical GPU."""
#     start = time.monotonic()
#     if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
#         raise RuntimeError("Worker must see exactly ONE GPU; check CUDA_VISIBLE_DEVICES")
#     torch.set_num_threads(args.cpu_threads)
#     root = args.output_dir.resolve()
#     base = {
#         "experiment": exp.name,
#         "status": "error",
#         "r2": None,
#         "gpu_slot": args._gpu_slot,
#         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
#         "started_utc": now_utc(),
#         "finished_utc": None,
#         "elapsed_seconds": None,
#         "epsilon_summary": "",
#         "error": "",
#     }
#     try:
#         cebra = load_fork(args.fork_dir)
#         arrays = load_data(args.data_dir / f"{args.session}.npz")
#         value, report = run_one(cebra, exp, arrays, args)
#         base["r2"] = float(value)
#         if exp.overrides.get("adv_epsilon_mode") == "dprime":
#             base["epsilon_summary"] = epsilon_summary(report)
#             if report is None:
#                 raise RuntimeError("dprime_report_ is missing: cannot verify calibration")
#             report_dest = root / "reports" / f"{exp.name}.json"
#             atomic_write_json(report_dest, report)
#         base["status"] = "success"  # Only after durable calibration report is saved.
#         print(f"{exp.name}: R2 = {value:.6f}", flush=True)
#         if args.report_eps:
#             print(f"{exp.name} calibration: {base['epsilon_summary']}", flush=True)
#     except BaseException as exc:
#         base["status"] = "error"
#         base["error"] = f"{type(exc).__name__}: {exc}"
#         traceback.print_exc()
#     finally:
#         base["elapsed_seconds"] = round(time.monotonic() - start, 3)
#         base["finished_utc"] = now_utc()
#         atomic_write_json(record_path(root, exp.name), base)
#     return 0 if base["status"] == "success" else 1


# def orchestrate(args, selected: List[Experiment]) -> int:
#     root = args.output_dir.resolve()
#     if root.exists() and not root.is_dir():
#         raise ValueError(f"--output-dir is not a directory: {root}")
#     root.mkdir(parents=True, exist_ok=True)
#     (root / "records").mkdir(exist_ok=True)
#     (root / "logs").mkdir(exist_ok=True)
#     (root / "reports").mkdir(exist_ok=True)
#     manifest_path = root / "run_config.json"
#     manifest = config_manifest(args, selected)
#     if manifest_path.exists():
#         old = json.loads(manifest_path.read_text(encoding="utf-8"))
#         if old != manifest:
#             raise RuntimeError("Result directory contains DIFFERENT config; use a new "
#                                "--output-dir (or restore exact previous script/config).")
#         if not args.resume and any(record_path(root, e.name).exists() for e in selected):
#             raise RuntimeError("Completed records exist. Use --resume or a new --output-dir; "
#                                "never overwrite old experimental results unintentionally.")
#     else:
#         if any((root / "records").glob("*.json")):
#             raise RuntimeError("Records exist without config manifest; choose a new output dir")
#         atomic_write_json(manifest_path, manifest)

#     pending = [e for e in selected if not (args.resume and
#                (rec := read_record(root, e.name)) is not None and rec.get("status") == "success")]
#     write_snapshots(root, selected)
#     if not pending:
#         print(f"All {len(selected)} results already present in {root}", flush=True)
#         return 0
#     tokens = gpu_tokens(args.gpu_workers)
#     todo = queue.Queue()
#     for e in pending:
#         todo.put(e)
#     print(f"Starting {len(pending)} experiments on {args.gpu_workers} GPUs. "
#           f"Durable output: {root}", flush=True)
#     for e in selected:
#         if e not in pending:
#             print(f"SKIP {e.name}: previously successful", flush=True)

#     # Only the coordinator (this process) writes consolidated CSV/text files.
#     # Every child writes its own JSON record atomically before it terminates.
#     update_lock = threading.Lock()
#     failures: list[str] = []

#     def gpu_loop(slot: int, cuda_token: str) -> None:
#         while True:
#             try:
#                 exp = todo.get_nowait()
#             except queue.Empty:
#                 return
#             filename = root / "logs" / f"{exp.name}.log"
#             cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
#                    "--_worker", "--_experiment", exp.name,
#                    "--_gpu_slot", str(slot),
#                    "--fork-dir", str(args.fork_dir.resolve()),
#                    "--data-dir", str(args.data_dir.resolve()),
#                    "--session", args.session, "--seed", str(args.seed),
#                    "--epsilon", str(args.epsilon),
#                    "--max-iterations", str(args.max_iterations),
#                    "--batch-size", str(args.batch_size),
#                    "--mlp-epochs", str(args.mlp_epochs),
#                    "--device", args.device,
#                    "--cpu-threads", str(args.cpu_threads),
#                    "--output-dir", str(root)]
#             if args.report_eps:
#                 cmd.append("--report-eps")
#             env = dict(os.environ)
#             env["CUDA_VISIBLE_DEVICES"] = cuda_token
#             for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
#                         "NUMEXPR_NUM_THREADS"):
#                 env[key] = str(args.cpu_threads)
#             with update_lock:
#                 print(f"[GPU {slot}] START {exp.name}", flush=True)
#             try:
#                 with filename.open("w", encoding="utf-8") as log:
#                     completed = subprocess.run(cmd, env=env, stdout=log,
#                                                stderr=subprocess.STDOUT, check=False)
#                 record = read_record(root, exp.name)
#                 if record is None:
#                     # Includes unexpected termination, e.g. a killed worker.
#                     record = dict(experiment=exp.name, status="error", r2=None,
#                                   gpu_slot=slot, cuda_visible_devices=cuda_token,
#                                   started_utc="", finished_utc=now_utc(),
#                                   elapsed_seconds=None, epsilon_summary="",
#                                   error=f"Worker exited with code {completed.returncode}; see {filename}")
#                     atomic_write_json(record_path(root, exp.name), record)
#                 if completed.returncode != 0 and record["status"] == "success":
#                     record["status"] = "error"
#                     record["error"] = f"Worker returned nonzero {completed.returncode}"
#                     atomic_write_json(record_path(root, exp.name), record)
#             except Exception as exc:
#                 record = dict(experiment=exp.name, status="error", r2=None,
#                               gpu_slot=slot, cuda_visible_devices=cuda_token,
#                               started_utc="", finished_utc=now_utc(),
#                               elapsed_seconds=None, epsilon_summary="",
#                               error=f"Coordinator exception: {type(exc).__name__}: {exc}")
#                 atomic_write_json(record_path(root, exp.name), record)
#             with update_lock:
#                 write_snapshots(root, selected)
#                 if record["status"] == "success":
#                     print(f"[GPU {slot}] DONE {exp.name}: R2 = {record['r2']:.6f} "
#                           f"({record.get('elapsed_seconds')}s)", flush=True)
#                 else:
#                     failures.append(exp.name)
#                     print(f"[GPU {slot}] FAILED {exp.name}: {record.get('error')} "
#                           f"[log={filename}]", flush=True)
#             todo.task_done()

#     threads = [threading.Thread(target=gpu_loop, args=(i, tokens[i]), daemon=False)
#                for i in range(args.gpu_workers)]
#     for thread in threads:
#         thread.start()
#     for thread in threads:
#         thread.join()
#     write_snapshots(root, selected)
#     print(f"Finished: {len(selected) - len(failures)} successful/previously completed "
#           f"out of {len(selected)}; {len(failures)} failed. "
#           f"Results: {root / 'results.csv'}", flush=True)
#     return 1 if failures else 0


# def main():
#     args = build_parser().parse_args()
#     selected = selected_experiments(args)
#     if args.list:
#         for e in experiments():
#             print(f"{e.name} [{'core' if e.core else 'all'}]")
#         return 0
#     if args.dry_run:
#         for e in selected:
#             print(e.name)
#         return 0
#     if args.max_iterations < 1 or args.mlp_epochs < 1 or args.batch_size < 1:
#         raise ValueError("Iterations, MLP epochs and batch size must be >= 1")
#     if not np.isfinite(args.epsilon) or args.epsilon < 0:
#         raise ValueError("PGD epsilon must be finite and nonnegative")
#     if args.cpu_threads < 1:
#         raise ValueError("--cpu-threads must be at least 1")
#     if args.device not in ("cuda_if_available", "cuda", "cuda:0"):
#         raise ValueError("With isolated one-GPU workers, --device must be cuda_if_available, cuda, or cuda:0")
#     if args._worker:
#         candidates = [exp for exp in experiments() if exp.name == args._experiment]
#         if len(candidates) != 1:
#             raise ValueError(f"Unknown internal worker name {args._experiment!r}")
#         return worker_run(args, candidates[0])
#     return orchestrate(args, selected)


# if __name__ == "__main__":
#     raise SystemExit(main())


# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-
# # """C-CO12: 4-GPU parallel Acorn-dprime comparison with crash-safe results.

# # Place next to Acorn-dprime/ and execute on a node/pod ALLOCATED FOUR GPUs:
# #     python -u dprime_4gpu.py --gpu-workers 4

# # 24 DISTINCT experiments: original raw PGD L2/Linf eps=3 and 22 dprime arms.
# # Each worker is a SEPARATE subprocess that sees exactly one GPU, so the four
# # independent experiments run concurrently. This is NOT multi-GPU data-parallel
# # training of a single encoder. All results are persisted after EACH experiment
# # under --output-dir (default: DPRIME_CCO12_4GPU_RESULTS beside this script).

# # If a workload is interrupted, restart the SAME script using --resume and
# # --output-dir pointing to the old, persistent directory. Completed records
# # are skipped; failed/incomplete ones are retried. Runs with different settings
# # must use different output directories.

# # IMPORTANT: Acorn-dprime uses whitened L2 in dprime mode, regardless of
# # attack_norm. Raw baselines support L2 and Linf; dprime-Linf is NOT implemented.
# # Data-only CEBRA.fit(x_train); y_train goes only to the downstream decoder.
# # """

# # from __future__ import annotations

# # import argparse
# # import csv
# # import json
# # from datetime import datetime, timezone
# # import hashlib
# # import contextlib
# # from dataclasses import dataclass, field
# # import gc
# # import inspect
# # import os
# # from pathlib import Path
# # import random
# # import queue
# # import subprocess
# # import tempfile
# # import threading
# # import time
# # import traceback
# # import sys
# # from typing import Any, Dict, List, Optional, Tuple

# # sys.dont_write_bytecode = True

# # import numpy as np
# # import torch
# # from torch import nn
# # from sklearn.metrics import r2_score

# # ROOT = Path(__file__).resolve().parent
# # DEFAULT_FORK_DIR = ROOT / "Acorn-dprime"
# # DEFAULT_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
# # DEFAULT_SESSION = "C-CO12"
# # DEFAULT_RESULTS_DIR = ROOT / "DPRIME_CCO12_4GPU_RESULTS"

# # SEED = 42
# # PGD_EPS = 3.0
# # ATTACK_STEPS = 10
# # PGD_ALPHA = PGD_EPS / 5.0

# # LATENT_DIM = 64
# # HIDDEN = 64
# # BATCH_SIZE = 2048
# # MAX_ITER = 3000
# # TEMPERATURE = 0.4
# # ARCH = "offset36-model-more-dropout"
# # DEVICE = "cuda_if_available"

# # MLP_EPOCHS = 2500
# # MLP_HIDDEN = 64
# # MLP_DROP = 0.4
# # MLP_LR = 1e-3
# # EVAL_BATCH_SIZE = 8192


# # @dataclass(frozen=True)
# # class Experiment:
# #     name: str
# #     overrides: Dict[str, Any] = field(default_factory=dict)
# #     core: bool = False


# # def experiments() -> List[Experiment]:
# #     """One-factor-at-a-time comparisons and a few faithful combined methods.

# #     Fixed d' epsilon=3 is NOT equivalent to raw L2/Linf epsilon=3: the units
# #     and feasible sets differ. Auto-epsilon runs include detector calibration.
# #     """
# #     base = [
# #         Experiment("PGD_L2_EPS3", {"adv_epsilon_mode": "constant", "attack_norm": "l2"}, True),
# #         Experiment("PGD_LINF_EPS3", {"adv_epsilon_mode": "constant", "attack_norm": "linf"}, True),
# #     ]

# #     # d' defaults: stationary, APGD, Madry, auto epsilon, instance-adaptive,
# #     # Ledoit-Wolf + Welch, guards off, time-only calibration, 1 restart.
# #     d = dict(
# #         adv_epsilon_mode="dprime",
# #         adv_dprime_layer="stationary",
# #         adv_dprime_epsilon="auto",
# #         adv_dprime_jnd=1.0,
# #         adv_dprime_residuals="whole",  # No behavior labels anywhere in CEBRA.
# #         adv_dprime_spatial="ledoit_wolf",
# #         adv_dprime_temporal="welch",
# #         adv_dprime_optimizer="apgd",
# #         adv_dprime_restarts=1,
# #         adv_dprime_objective="madry",
# #         adv_dprime_instance_adaptive=True,
# #         adv_dprime_guards=(),
# #         adv_dprime_guard_mode="off",
# #         adv_dprime_warmup=0.10,
# #         adv_dprime_curriculum=0.50,
# #     )

# #     def arm(name: str, core: bool = False, **kw: Any) -> Experiment:
# #         return Experiment(name, {**d, **kw}, core)

# #     return base + [
# #         # Algorithm and step adaptation, with the same fixed whitened L2 budget.
# #         arm("DPRIME_STATIONARY_PGD_FIXED3", True,
# #             adv_dprime_epsilon=3.0, adv_dprime_optimizer="pgd",
# #             adv_dprime_instance_adaptive=False),
# #         arm("DPRIME_STATIONARY_APGD_FIXED3", True,
# #             adv_dprime_epsilon=3.0, adv_dprime_instance_adaptive=False),
# #         arm("DPRIME_APGD_RESTARTS3_FIXED3",
# #             adv_dprime_epsilon=3.0, adv_dprime_instance_adaptive=False,
# #             adv_dprime_restarts=3),
# #         # The full 3-criterion (JND/NN/detector) epsilon protocol.
# #         arm("DPRIME_AUTO_GLOBAL", True, adv_dprime_instance_adaptive=False),
# #         arm("DPRIME_AUTO_INSTANCE", True),
# #         arm("DPRIME_AUTO_INSTANCE_NO_CURRICULUM", adv_dprime_curriculum=0.0),
# #         # TRADES is a DIFFERENT training objective than the original Madry AT.
# #         arm("DPRIME_TRADES_AUTO_INSTANCE", adv_dprime_objective="trades"),
# #         # Spatial and temporal ablations of stationary NoiseModel.
# #         arm("DPRIME_SPATIAL_OAS", adv_dprime_spatial="oas"),
# #         arm("DPRIME_SPATIAL_TYLER", adv_dprime_spatial="tyler"),
# #         arm("DPRIME_SPATIAL_EMPIRICAL", adv_dprime_spatial="empirical"),
# #         arm("DPRIME_TEMPORAL_AR4", adv_dprime_temporal="ar", adv_dprime_ar_order=4),
# #         arm("DPRIME_TEMPORAL_WHITE", adv_dprime_temporal="white"),
# #         arm("DPRIME_TEMPORAL_VANROSSUM", True,
# #             adv_dprime_temporal="vanrossum", adv_dprime_vanrossum_tau="auto"),
# #         # Nonstationary, trial-specific time-frequency coloring (still whitened L2).
# #         arm("DPRIME_TRIAL_MASK", True, adv_dprime_layer="trial_mask"),
# #         arm("DPRIME_TRIAL_MASK_PGD", adv_dprime_layer="trial_mask",
# #             adv_dprime_optimizer="pgd"),
# #         # Layer 3, evaluated separately from the basic metric for ablations.
# #         arm("DPRIME_GUARD_RIEMANN", adv_dprime_guards=("riemann",),
# #             adv_dprime_guard_mode="backtrack"),
# #         arm("DPRIME_GUARD_ARTIFACT", adv_dprime_guards=("artifact",),
# #             adv_dprime_guard_mode="backtrack"),
# #         arm("DPRIME_GUARD_BAND_POWER", adv_dprime_guards=("band_power",),
# #             adv_dprime_guard_mode="backtrack"),
# #         arm("DPRIME_GUARD_APERIODIC", adv_dprime_guards=("aperiodic",),
# #             adv_dprime_guard_mode="backtrack"),
# #         arm("DPRIME_GUARDS_ALL", adv_dprime_guards=("band_power", "aperiodic", "riemann", "artifact"),
# #             adv_dprime_guard_mode="backtrack"),
# #         arm("DPRIME_GUARDS_FINAL_ONLY", adv_dprime_guards=("riemann",),
# #             adv_dprime_guard_mode="final"),
# #         arm("DPRIME_GUARD_SOFT_RIEMANN", adv_dprime_guards=("riemann",),
# #             adv_dprime_guard_mode="backtrack", adv_dprime_soft_riemann=1.0),
# #     ]


# # def seed_all(seed: int) -> None:
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(seed)
# #     torch.backends.cudnn.benchmark = False
# #     torch.backends.cudnn.deterministic = True


# # def load_fork(fork_dir: Path):
# #     package = fork_dir / "cebra" / "__init__.py"
# #     if not package.is_file():
# #         raise FileNotFoundError(
# #             f"Expected fork at {fork_dir}. Place this runner next to Acorn-dprime "
# #             "or set --fork-dir PATH."
# #         )
# #     for key in list(sys.modules):
# #         if key == "cebra" or key.startswith("cebra."):
# #             del sys.modules[key]
# #     sys.path.insert(0, str(fork_dir.resolve()))
# #     import cebra

# #     actual = Path(cebra.__file__).resolve()
# #     if not actual.is_relative_to(fork_dir.resolve()):
# #         raise RuntimeError(f"Wrong CEBRA imported: {actual}")
# #     params = inspect.signature(cebra.CEBRA.__init__).parameters
# #     required = {
# #         "training_mode", "adv_epsilon", "adv_alpha", "adv_steps", "attack_norm",
# #         "adv_epsilon_mode", "adv_dprime_epsilon", "adv_dprime_layer",
# #         "adv_dprime_optimizer", "adv_dprime_instance_adaptive", "adv_dprime_temporal",
# #         "adv_dprime_spatial", "adv_dprime_guards", "adv_dprime_objective",
# #     }
# #     missing = sorted(required - set(params))
# #     if missing:
# #         raise RuntimeError(f"This fork lacks Acorn-dprime parameters: {missing}")
# #     return cebra


# # def load_data(data_path: Path):
# #     if not data_path.is_file():
# #         raise FileNotFoundError(f"No session file at: {data_path}")
# #     with np.load(data_path, allow_pickle=False) as f:
# #         x_train, x_valid, y_train, y_valid = (
# #             np.asarray(f[k], dtype=np.float32)
# #             for k in ("train_data", "valid_data", "train_label", "valid_label")
# #         )

# #     if y_train.ndim == 1:
# #         y_train = y_train[:, None]
# #     if y_valid.ndim == 1:
# #         y_valid = y_valid[:, None]
# #     for key, arr in (("train_data", x_train), ("valid_data", x_valid),
# #                      ("train_label", y_train), ("valid_label", y_valid)):
# #         if arr.ndim != 2 or 0 in arr.shape or not np.isfinite(arr).all():
# #             raise ValueError(f"Invalid shape / nonfinite {key}: {arr.shape}")
# #     if (x_train.shape[1] != x_valid.shape[1]
# #             or y_train.shape[1] != y_valid.shape[1]):
# #         raise ValueError("Train/valid feature or label dimensions differ")
# #     if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
# #         raise ValueError("Neural data and decoder labels have different lengths")
# #     if min(len(x_train), len(x_valid)) < 36:
# #         raise ValueError("The offset36 model needs at least 36 bins in each split")
# #     for name, x in (("train_data", x_train), ("valid_data", x_valid)):
# #         if np.any(x < 0) or not np.array_equal(x, np.rint(x)):
# #             raise ValueError(f"{name} must be RAW nonnegative integer spike counts")
# #     return tuple(np.ascontiguousarray(a) for a in (x_train, x_valid, y_train, y_valid))


# # def encoder_config(exp: Experiment, iterations: int, batch_size: int,
# #                    device: str, seed: int, pgd_eps: float) -> Dict[str, Any]:
# #     overrides = dict(exp.overrides)
# #     return {
# #         "model_architecture": ARCH,
# #         "batch_size": batch_size,
# #         "temperature": TEMPERATURE,
# #         "time_offsets": 1,
# #         "max_iterations": iterations,
# #         "output_dimension": LATENT_DIM,
# #         "num_hidden_units": HIDDEN,
# #         "learning_rate": 3e-4,
# #         "temperature_mode": "constant",
# #         "training_mode": "adversarial",
# #         "adv_epsilon": pgd_eps,
# #         "adv_alpha": pgd_eps / 5.0,
# #         "adv_steps": ATTACK_STEPS,
# #         "attack_norm": "l2",   # A placeholder ignored by d' training.
# #         "adv_dprime_seed": seed,
# #         "device": device,
# #         "verbose": False,
# #         **overrides,
# #     }


# # class TwoLayerMLP(nn.Module):
# #     def __init__(self, dim: int, out: int):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(dim, MLP_HIDDEN),
# #             nn.LayerNorm(MLP_HIDDEN),
# #             nn.ReLU(),
# #             nn.Dropout(MLP_DROP),
# #             nn.Linear(MLP_HIDDEN, out),
# #         )

# #     def forward(self, x):
# #         return self.net(x)


# # def train_decoder(z_train, y_train, seed: int, device, epochs: int):
# #     seed_all(seed)
# #     decoder = TwoLayerMLP(z_train.shape[1], y_train.shape[1]).to(device)
# #     opt = torch.optim.Adam(decoder.parameters(), lr=MLP_LR)
# #     criterion = nn.MSELoss()
# #     z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
# #     y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
# #     decoder.train()
# #     for epoch in range(epochs):
# #         opt.zero_grad(set_to_none=True)
# #         loss = criterion(decoder(z), y)
# #         if not torch.isfinite(loss).item():
# #             raise FloatingPointError(f"Nonfinite decoder loss at epoch {epoch + 1}")
# #         loss.backward()
# #         opt.step()
# #     return decoder


# # def predict_decoder(decoder, embeddings):
# #     decoder.eval()
# #     device = next(decoder.parameters()).device
# #     preds = []
# #     with torch.inference_mode():
# #         for first in range(0, len(embeddings), EVAL_BATCH_SIZE):
# #             z = torch.as_tensor(embeddings[first:first + EVAL_BATCH_SIZE],
# #                                 dtype=torch.float32, device=device)
# #             preds.append(decoder(z).cpu().numpy())
# #     return np.concatenate(preds, axis=0)


# # def epsilon_summary(report: Optional[Dict[str, Any]]) -> str:
# #     if not report:
# #         return "no-dprime-report"
# #     e = report.get("epsilon", {})
# #     nn = report.get("eps_nn", {})
# #     t = report.get("training", {})
# #     return (f"eps_star={e.get('eps_star')} "
# #             f"eps_NN={nn.get('value')} "
# #             f"mean_dprime={t.get('mean_dprime')} "
# #             f"NN_binding={t.get('eps_nn_binding_rate')} "
# #             f"guard_accept={t.get('accept_rate')}")


# # def run_one(cebra, exp: Experiment, arrays, args) -> Tuple[float, Optional[Dict[str, Any]]]:
# #     x_train, x_valid, y_train, y_valid = arrays
# #     encoder = decoder = z_train = z_valid = None
# #     report = None
# #     try:
# #         seed_all(args.seed)
# #         encoder = cebra.CEBRA(**encoder_config(exp, args.max_iterations, args.batch_size,
# #                                                args.device, args.seed, args.epsilon))
# #         # TIME-ONLY training: y_train is deliberately NOT passed here.
# #         encoder.fit(x_train)
# #         report = getattr(encoder, "dprime_report_", None)
# #         z_train = np.asarray(encoder.transform(x_train), dtype=np.float32)
# #         z_valid = np.asarray(encoder.transform(x_valid), dtype=np.float32)
# #         if (len(z_train) != len(x_train) or len(z_valid) != len(x_valid)
# #                 or not np.isfinite(z_train).all() or not np.isfinite(z_valid).all()):
# #             raise ValueError(f"Bad embedding shape / alignment for {exp.name}")
# #         decoder = train_decoder(z_train, y_train, args.seed + 100_000,
# #                                 torch.device(encoder.device_), args.mlp_epochs)
# #         predictions = predict_decoder(decoder, z_valid)
# #         if predictions.shape != y_valid.shape or not np.isfinite(predictions).all():
# #             raise ValueError(f"Bad decoder predictions for {exp.name}")
# #         value = float(np.atleast_1d(r2_score(y_valid, predictions,
# #                                              multioutput="raw_values")).mean())
# #         if not np.isfinite(value):
# #             raise ValueError("R2 is nonfinite; check validation label variability")
# #         return value, report
# #     finally:
# #         del encoder, decoder, z_train, z_valid
# #         gc.collect()
# #         if torch.cuda.is_available():
# #             torch.cuda.empty_cache()


# # def build_parser() -> argparse.ArgumentParser:
# #     p = argparse.ArgumentParser(description=__doc__,
# #                                 formatter_class=argparse.RawDescriptionHelpFormatter)
# #     p.add_argument("--fork-dir", type=Path, default=DEFAULT_FORK_DIR)
# #     p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
# #     p.add_argument("--session", default=DEFAULT_SESSION)
# #     p.add_argument("--suite", choices=("all", "core"), default="all")
# #     p.add_argument("--only", nargs="+", help="Run named arms; use --list to inspect names")
# #     p.add_argument("--list", action="store_true", help="List all experiments and exit")
# #     p.add_argument("--dry-run", action="store_true", help="List selected arms without fitting")
# #     p.add_argument("--seed", type=int, default=SEED)
# #     p.add_argument("--epsilon", type=float, default=PGD_EPS,
# #                    help="Epsilon for the ORIGINAL L2/Linf PGD only (also sets alpha=epsilon/5)")
# #     p.add_argument("--max-iterations", type=int, default=MAX_ITER)
# #     p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
# #     p.add_argument("--mlp-epochs", type=int, default=MLP_EPOCHS)
# #     p.add_argument("--device", default=DEVICE)
# #     p.add_argument("--report-eps", action="store_true",
# #                    help="Print dprime calibration diagnostics to stderr after each run")
# #     p.add_argument("--gpu-workers", type=int, default=4,
# #                    help="Number of independently allocated GPUs to use (default 4)")
# #     p.add_argument("--cpu-threads", type=int, default=2,
# #                    help="CPU threads per GPU subprocess (default 2)")
# #     p.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR,
# #                    help="PERSISTENT results directory. MUST reside on a shared/host volume")
# #     p.add_argument("--resume", action="store_true",
# #                    help="Skip successful experiment records, retry interrupted/failed runs")
# #     p.add_argument("--continue-on-error", action="store_true",
# #                    help="Kept for CLI compatibility; the 4-GPU suite always records each error and continues")
# #     # Internal worker arguments; not intended for direct use.
# #     p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
# #     p.add_argument("--_experiment", default=None, help=argparse.SUPPRESS)
# #     p.add_argument("--_gpu_slot", type=int, default=-1, help=argparse.SUPPRESS)
# #     return p


# # def now_utc() -> str:
# #     return datetime.now(timezone.utc).isoformat(timespec="seconds")


# # def atomic_write_text(path: Path, content: str) -> None:
# #     """Commit one entire file; readers see either old or new version."""
# #     path.parent.mkdir(parents=True, exist_ok=True)
# #     fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
# #     try:
# #         with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
# #             f.write(content)
# #             f.flush()
# #             os.fsync(f.fileno())
# #         os.replace(temp_path, path)
# #     finally:
# #         if os.path.exists(temp_path):
# #             os.unlink(temp_path)


# # def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
# #     atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2,
# #                                        allow_nan=False, sort_keys=True) + "\n")


# # def record_path(root: Path, name: str) -> Path:
# #     if not name.replace("_", "").isalnum():
# #         raise ValueError(f"Unsafe experiment name: {name!r}")
# #     return root / "records" / f"{name}.json"


# # def read_record(root: Path, name: str):
# #     path = record_path(root, name)
# #     if not path.is_file():
# #         return None
# #     with path.open("r", encoding="utf-8") as f:
# #         obj = json.load(f)
# #     if obj.get("experiment") != name:
# #         raise RuntimeError(f"Mismatched record in {path}")
# #     return obj


# # def write_snapshots(root: Path, selected: List[Experiment]) -> None:
# #     """Consolidate atomic per-arm records into CSV and readable text."""
# #     import io
# #     rows = []
# #     summaries = []
# #     for exp in selected:
# #         record = read_record(root, exp.name)
# #         if record is None:
# #             continue
# #         rows.append({
# #             "experiment": exp.name,
# #             "status": record.get("status", "unknown"),
# #             "r2": record.get("r2", ""),
# #             "gpu_slot": record.get("gpu_slot", ""),
# #             "cuda_visible_devices": record.get("cuda_visible_devices", ""),
# #             "elapsed_seconds": record.get("elapsed_seconds", ""),
# #             "finished_utc": record.get("finished_utc", ""),
# #             "epsilon_summary": record.get("epsilon_summary", ""),
# #             "error": record.get("error", ""),
# #         })
# #         if record.get("status") == "success":
# #             summaries.append(f"{exp.name}: R2 = {record['r2']:.6f}"
# #                              f" | GPU slot {record.get('gpu_slot')}"
# #                              f" | elapsed {record.get('elapsed_seconds',0):.1f}s")
# #         else:
# #             summaries.append(f"{exp.name}: ERROR: {record.get('error')}")
# #     stream = io.StringIO()
# #     fields = ["experiment", "status", "r2", "gpu_slot", "cuda_visible_devices",
# #               "elapsed_seconds", "finished_utc", "epsilon_summary", "error"]
# #     writer = csv.DictWriter(stream, fieldnames=fields)
# #     writer.writeheader()
# #     writer.writerows(rows)
# #     atomic_write_text(root / "results.csv", stream.getvalue())
# #     atomic_write_text(root / "results.txt", "\n".join(summaries) + ("\n" if summaries else ""))


# # def selected_experiments(args):
# #     all_exp = experiments()
# #     names = {e.name for e in all_exp}
# #     if args.only:
# #         unknown = sorted(set(args.only) - names)
# #         if unknown:
# #             raise ValueError(f"Unknown experiments: {unknown}; inspect --list")
# #         return [e for e in all_exp if e.name in set(args.only)]
# #     if args.suite == "core":
# #         return [e for e in all_exp if e.core]
# #     return all_exp


# # def config_manifest(args, selected: List[Experiment]) -> dict:
# #     """Prevent accidentally mixing incomparable R2 values across restarts."""
# #     data_path = (args.data_dir / f"{args.session}.npz").resolve()
# #     return json.loads(json.dumps({
# #         "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
# #         "fork_dir": str(args.fork_dir.resolve()),
# #         "data_path": str(data_path),
# #         "session": args.session,
# #         "seed": args.seed,
# #         "epsilon": args.epsilon,
# #         "batch_size": args.batch_size,
# #         "max_iterations": args.max_iterations,
# #         "mlp_epochs": args.mlp_epochs,
# #         "architecture": ARCH,
# #         "latent_dim": LATENT_DIM,
# #         "hidden": HIDDEN,
# #         "temperature": TEMPERATURE,
# #         "attack_steps": ATTACK_STEPS,
# #         "experiments": [{"name": exp.name, "overrides": exp.overrides}
# #                         for exp in selected],
# #     }, sort_keys=True))


# # def gpu_tokens(workers: int) -> list[str]:
# #     if workers < 1:
# #         raise ValueError("--gpu-workers must be at least 1")
# #     if not torch.cuda.is_available():
# #         raise RuntimeError("No CUDA GPU available in this RunAI container")
# #     count = torch.cuda.device_count()
# #     if count < workers:
# #         raise RuntimeError(f"Requested {workers} GPU workers but only {count} CUDA devices visible. "
# #                            "Submit RunAI job with --gpu 4 and check GPU allocation.")
# #     visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
# #     if visibility and visibility != "-1":
# #         tokens = [s.strip() for s in visibility.split(",") if s.strip()]
# #         if len(tokens) < workers:
# #             raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visibility!r} has fewer than {workers} entries")
# #         return tokens[:workers]
# #     return [str(i) for i in range(workers)]


# # def worker_run(args, exp: Experiment) -> int:
# #     """Executed INSIDE subprocess already masked to one physical GPU."""
# #     start = time.monotonic()
# #     if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
# #         raise RuntimeError("Worker must see exactly ONE GPU; check CUDA_VISIBLE_DEVICES")
# #     torch.set_num_threads(args.cpu_threads)
# #     root = args.output_dir.resolve()
# #     base = {
# #         "experiment": exp.name,
# #         "status": "error",
# #         "r2": None,
# #         "gpu_slot": args._gpu_slot,
# #         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
# #         "started_utc": now_utc(),
# #         "finished_utc": None,
# #         "elapsed_seconds": None,
# #         "epsilon_summary": "",
# #         "error": "",
# #     }
# #     try:
# #         cebra = load_fork(args.fork_dir)
# #         arrays = load_data(args.data_dir / f"{args.session}.npz")
# #         value, report = run_one(cebra, exp, arrays, args)
# #         base["status"] = "success"
# #         base["r2"] = float(value)
# #         if exp.overrides.get("adv_epsilon_mode") == "dprime":
# #             base["epsilon_summary"] = epsilon_summary(report)
# #         print(f"{exp.name}: R2 = {value:.6f}", flush=True)
# #         if args.report_eps:
# #             print(f"{exp.name} calibration: {base['epsilon_summary']}", flush=True)
# #     except BaseException as exc:
# #         base["error"] = f"{type(exc).__name__}: {exc}"
# #         traceback.print_exc()
# #     finally:
# #         base["elapsed_seconds"] = round(time.monotonic() - start, 3)
# #         base["finished_utc"] = now_utc()
# #         atomic_write_json(record_path(root, exp.name), base)
# #     return 0 if base["status"] == "success" else 1


# # def orchestrate(args, selected: List[Experiment]) -> int:
# #     root = args.output_dir.resolve()
# #     if root.exists() and not root.is_dir():
# #         raise ValueError(f"--output-dir is not a directory: {root}")
# #     root.mkdir(parents=True, exist_ok=True)
# #     (root / "records").mkdir(exist_ok=True)
# #     (root / "logs").mkdir(exist_ok=True)
# #     manifest_path = root / "run_config.json"
# #     manifest = config_manifest(args, selected)
# #     if manifest_path.exists():
# #         old = json.loads(manifest_path.read_text(encoding="utf-8"))
# #         if old != manifest:
# #             raise RuntimeError("Result directory contains DIFFERENT config; use a new "
# #                                "--output-dir (or restore exact previous script/config).")
# #         if not args.resume and any(record_path(root, e.name).exists() for e in selected):
# #             raise RuntimeError("Completed records exist. Use --resume or a new --output-dir; "
# #                                "never overwrite old experimental results unintentionally.")
# #     else:
# #         if any((root / "records").glob("*.json")):
# #             raise RuntimeError("Records exist without config manifest; choose a new output dir")
# #         atomic_write_json(manifest_path, manifest)

# #     pending = [e for e in selected if not (args.resume and
# #                (rec := read_record(root, e.name)) is not None and rec.get("status") == "success")]
# #     write_snapshots(root, selected)
# #     if not pending:
# #         print(f"All {len(selected)} results already present in {root}", flush=True)
# #         return 0
# #     tokens = gpu_tokens(args.gpu_workers)
# #     todo = queue.Queue()
# #     for e in pending:
# #         todo.put(e)
# #     print(f"Starting {len(pending)} experiments on {args.gpu_workers} GPUs. "
# #           f"Durable output: {root}", flush=True)
# #     for e in selected:
# #         if e not in pending:
# #             print(f"SKIP {e.name}: previously successful", flush=True)

# #     # Only the coordinator (this process) writes consolidated CSV/text files.
# #     # Every child writes its own JSON record atomically before it terminates.
# #     update_lock = threading.Lock()
# #     failures: list[str] = []

# #     def gpu_loop(slot: int, cuda_token: str) -> None:
# #         while True:
# #             try:
# #                 exp = todo.get_nowait()
# #             except queue.Empty:
# #                 return
# #             filename = root / "logs" / f"{exp.name}.log"
# #             cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
# #                    "--_worker", "--_experiment", exp.name,
# #                    "--_gpu_slot", str(slot),
# #                    "--fork-dir", str(args.fork_dir.resolve()),
# #                    "--data-dir", str(args.data_dir.resolve()),
# #                    "--session", args.session, "--seed", str(args.seed),
# #                    "--epsilon", str(args.epsilon),
# #                    "--max-iterations", str(args.max_iterations),
# #                    "--batch-size", str(args.batch_size),
# #                    "--mlp-epochs", str(args.mlp_epochs),
# #                    "--device", args.device,
# #                    "--cpu-threads", str(args.cpu_threads),
# #                    "--output-dir", str(root)]
# #             if args.report_eps:
# #                 cmd.append("--report-eps")
# #             env = dict(os.environ)
# #             env["CUDA_VISIBLE_DEVICES"] = cuda_token
# #             for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
# #                         "NUMEXPR_NUM_THREADS"):
# #                 env[key] = str(args.cpu_threads)
# #             with update_lock:
# #                 print(f"[GPU {slot}] START {exp.name}", flush=True)
# #             try:
# #                 with filename.open("w", encoding="utf-8") as log:
# #                     completed = subprocess.run(cmd, env=env, stdout=log,
# #                                                stderr=subprocess.STDOUT, check=False)
# #                 record = read_record(root, exp.name)
# #                 if record is None:
# #                     # Includes unexpected termination, e.g. a killed worker.
# #                     record = dict(experiment=exp.name, status="error", r2=None,
# #                                   gpu_slot=slot, cuda_visible_devices=cuda_token,
# #                                   started_utc="", finished_utc=now_utc(),
# #                                   elapsed_seconds=None, epsilon_summary="",
# #                                   error=f"Worker exited with code {completed.returncode}; see {filename}")
# #                     atomic_write_json(record_path(root, exp.name), record)
# #                 if completed.returncode != 0 and record["status"] == "success":
# #                     record["status"] = "error"
# #                     record["error"] = f"Worker returned nonzero {completed.returncode}"
# #                     atomic_write_json(record_path(root, exp.name), record)
# #             except Exception as exc:
# #                 record = dict(experiment=exp.name, status="error", r2=None,
# #                               gpu_slot=slot, cuda_visible_devices=cuda_token,
# #                               started_utc="", finished_utc=now_utc(),
# #                               elapsed_seconds=None, epsilon_summary="",
# #                               error=f"Coordinator exception: {type(exc).__name__}: {exc}")
# #                 atomic_write_json(record_path(root, exp.name), record)
# #             with update_lock:
# #                 write_snapshots(root, selected)
# #                 if record["status"] == "success":
# #                     print(f"[GPU {slot}] DONE {exp.name}: R2 = {record['r2']:.6f} "
# #                           f"({record.get('elapsed_seconds')}s)", flush=True)
# #                 else:
# #                     failures.append(exp.name)
# #                     print(f"[GPU {slot}] FAILED {exp.name}: {record.get('error')} "
# #                           f"[log={filename}]", flush=True)
# #             todo.task_done()

# #     threads = [threading.Thread(target=gpu_loop, args=(i, tokens[i]), daemon=False)
# #                for i in range(args.gpu_workers)]
# #     for thread in threads:
# #         thread.start()
# #     for thread in threads:
# #         thread.join()
# #     write_snapshots(root, selected)
# #     print(f"Finished: {len(selected) - len(failures)} successful/previously completed "
# #           f"out of {len(selected)}; {len(failures)} failed. "
# #           f"Results: {root / 'results.csv'}", flush=True)
# #     return 1 if failures else 0


# # def main():
# #     args = build_parser().parse_args()
# #     selected = selected_experiments(args)
# #     if args.list:
# #         for e in experiments():
# #             print(f"{e.name} [{'core' if e.core else 'all'}]")
# #         return 0
# #     if args.dry_run:
# #         for e in selected:
# #             print(e.name)
# #         return 0
# #     if args.max_iterations < 1 or args.mlp_epochs < 1 or args.batch_size < 1:
# #         raise ValueError("Iterations, MLP epochs and batch size must be >= 1")
# #     if not np.isfinite(args.epsilon) or args.epsilon < 0:
# #         raise ValueError("PGD epsilon must be finite and nonnegative")
# #     if args.cpu_threads < 1:
# #         raise ValueError("--cpu-threads must be at least 1")
# #     if args.device not in ("cuda_if_available", "cuda", "cuda:0"):
# #         raise ValueError("With isolated one-GPU workers, --device must be cuda_if_available, cuda, or cuda:0")
# #     if args._worker:
# #         candidates = [exp for exp in experiments() if exp.name == args._experiment]
# #         if len(candidates) != 1:
# #             raise ValueError(f"Unknown internal worker name {args._experiment!r}")
# #         return worker_run(args, candidates[0])
# #     return orchestrate(args, selected)


# # if __name__ == "__main__":
# #     raise SystemExit(main())

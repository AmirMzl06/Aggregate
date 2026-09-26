"""Four-arm CEBRA-vs-MSE comparison on Perich C-CO12.

Arms
----
1) cebra_all_labels   : Offset36 encoder trained with supervised CEBRA/InfoNCE
                        using ALL behavioral labels for positive sampling.
2) cebra_label0_only  : same encoder/objective, but the dataset is treated as
                        if ONLY y[:, 0] exists.
3) mse_all_labels     : same Offset36 encoder trained end-to-end with direct MSE
                        to ALL behavioral labels.
4) mse_label0_only    : same MSE setup, but the dataset is treated as if ONLY
                        y[:, 0] exists.

Primary evaluation
------------------
After encoder training, every arm gets a NEW decoder with the SAME architecture
and training protocol. The encoder is frozen. This keeps the final R2 comparison
focused on representation quality rather than giving the MSE arms an advantage
from their jointly-trained regression head.

The MSE arms also report the R2 of their own jointly-trained head as a secondary
metric.

Expected local layout
---------------------
<project>/
    cebra_vs_mse.py
    run_cebra_vs_mse_4arms_cco12.py
    CEBRA-original/
        cebra/
            __init__.py

The CEBRA fork is selected and verified inside cebra_vs_mse.py.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import csv
import gc
import json
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

from cebra_vs_mse import (
    CompareConfig,
    build_paired_models,
    train_infonce,
    train_mse,
)


# =============================================================================
# CONFIG
# =============================================================================
ROOT = Path(__file__).resolve().parent
PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
SESSION = "C-CO12"
NPZ_PATH = PERICH_DATA_DIR / f"{SESSION}.npz"
OUT_ROOT = ROOT / "CEBRA_VS_MSE_4ARMS_CCO12"

# Encoder settings: kept identical across all four arms.
LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 3000
TEMPERATURE = 0.4
ARCH = "offset36-model-more-dropout"
TIME_OFFSETS = 1
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.0
DEVICE = "cuda_if_available"

# Same fresh frozen-encoder decoder for ALL four arms.
# Here MLP_STEPS means optimizer updates, not full-dataset epochs.
MLP_STEPS = 2500
MLP_BATCH_SIZE = 2048
MLP_HIDDEN = 64
MLP_DROP = 0.4
MLP_LR = 1e-3
EVAL_BATCH_SIZE = 8192

SEEDS = (42,)  # e.g. (42, 43, 44)
SAVE_CHECKPOINTS = True


# =============================================================================
# REPRODUCIBILITY / UTILS
# =============================================================================
def resolve_device() -> str:
    if DEVICE == "cuda_if_available":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_data():
    if not NPZ_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {NPZ_PATH}")

    with np.load(NPZ_PATH, allow_pickle=False) as data:
        arrays = [
            np.asarray(data[key], dtype=np.float32)
            for key in ("train_data", "valid_data", "train_label", "valid_label")
        ]

    x_train, x_valid, y_train, y_valid = arrays

    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_valid.ndim == 1:
        y_valid = y_valid[:, None]

    for name, values in zip(
        ("X_train", "X_valid", "Y_train", "Y_valid"),
        (x_train, x_valid, y_train, y_valid),
    ):
        if values.ndim != 2 or min(values.shape) == 0:
            raise ValueError(f"{name} must be a nonempty 2D array; got {values.shape}.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN/Inf.")

    if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
        raise ValueError("Feature and label lengths do not match.")
    if x_train.shape[1] != x_valid.shape[1]:
        raise ValueError("Train/valid neuron dimensions do not match.")
    if y_train.shape[1] != y_valid.shape[1]:
        raise ValueError("Train/valid label dimensions do not match.")
    if len(x_train) < 36 or len(x_valid) < 36:
        raise ValueError("offset36 requires at least 36 timepoints per split.")

    return tuple(np.ascontiguousarray(a) for a in (x_train, x_valid, y_train, y_valid))


def standardize_from_train(y_train: np.ndarray, y_valid: np.ndarray):
    """Use training-label statistics only; no validation leakage."""
    mu = y_train.mean(axis=0, keepdims=True)
    sd = y_train.std(axis=0, keepdims=True)
    sd = np.maximum(sd, 1e-8)
    return (
        np.ascontiguousarray((y_train - mu) / sd, dtype=np.float32),
        np.ascontiguousarray((y_valid - mu) / sd, dtype=np.float32),
        mu.astype(np.float32),
        sd.astype(np.float32),
    )


def make_cfg(seed: int, device: str) -> CompareConfig:
    return CompareConfig(
        model_name=ARCH,
        hidden_dim=HIDDEN,
        embedding_dim=LATENT_DIM,
        normalize_embedding=True,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        steps=MAX_ITER,
        temperature=TEMPERATURE,
        time_offset=TIME_OFFSETS,
        conditional="time_delta",
        mse_head_hidden=MLP_HIDDEN,
        mse_head_dropout=MLP_DROP,
        seed=seed,
        device=device,
    )


# =============================================================================
# BATCHED ENCODING
# =============================================================================
@torch.no_grad()
def encode_batched(encoder: nn.Module, X: np.ndarray, batch_size: int) -> np.ndarray:
    """Encode every time bin with Offset36 while keeping sequence length T.

    This uses the same replicate-padding convention as cebra_vs_mse.py but
    evaluates windows in batches so a long Perich session does not need one
    huge convolutional forward pass.
    """
    encoder.eval()
    device = next(encoder.parameters()).device

    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    if x.ndim != 2:
        raise ValueError(f"Expected X=(T,N), got {tuple(x.shape)}")

    offset = encoder.get_offset()
    left = int(offset.left)
    right = int(offset.right)
    window = left + right

    # (T,N) -> (1,N,T), then pad so unfold gives exactly T windows.
    x = x.T.unsqueeze(0)
    x = F.pad(x, (left, right - 1), mode="replicate")
    windows = x.unfold(dimension=2, size=window, step=1)  # (1,N,T,window)

    T = X.shape[0]
    out = []
    for start in range(0, T, batch_size):
        stop = min(start + batch_size, T)
        xb = windows[0, :, start:stop, :].permute(1, 0, 2).contiguous()  # (B,N,36)
        z = encoder(xb)
        if z.ndim != 2:
            raise RuntimeError(f"Expected encoder output (B,D), got {tuple(z.shape)}")
        out.append(z.detach().cpu())

    return torch.cat(out, dim=0).numpy().astype(np.float32, copy=False)


@torch.no_grad()
def predict_joint_mse_batched(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    """Prediction from the MSE arm's own head; secondary metric only."""
    model.eval()
    device = next(model.parameters()).device
    encoder = model.encoder

    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    offset = encoder.get_offset()
    left = int(offset.left)
    right = int(offset.right)
    window = left + right

    x = x.T.unsqueeze(0)
    x = F.pad(x, (left, right - 1), mode="replicate")
    windows = x.unfold(2, window, 1)

    T = X.shape[0]
    pred = []
    for start in range(0, T, batch_size):
        stop = min(start + batch_size, T)
        xb = windows[0, :, start:stop, :].permute(1, 0, 2).contiguous()
        yhat, _ = model.forward_windows(xb)
        pred.append(yhat.detach().cpu())

    return torch.cat(pred, dim=0).numpy().astype(np.float32, copy=False)


# =============================================================================
# IDENTICAL FROZEN-ENCODER DECODER FOR ALL ARMS
# =============================================================================
class TwoLayerMLP(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, MLP_HIDDEN),
            nn.LayerNorm(MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(MLP_DROP),
            nn.Linear(MLP_HIDDEN, out),
        )

    def forward(self, x):
        return self.net(x)


def fit_probe(
    z_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    device: str,
) -> TwoLayerMLP:
    """Train the same fresh MLP on frozen embeddings for every arm."""
    seed_all(seed)

    z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    if y.ndim == 1:
        y = y[:, None]

    model = TwoLayerMLP(z.shape[1], y.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    mse = nn.MSELoss()

    g = torch.Generator(device=device)
    g.manual_seed(seed + 100_000)

    model.train()
    for step in range(MLP_STEPS):
        idx = torch.randint(
            0,
            z.shape[0],
            (min(MLP_BATCH_SIZE, z.shape[0]),),
            generator=g,
            device=device,
        )
        pred = model(z[idx])
        loss = mse(pred, y[idx])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == MLP_STEPS - 1:
            print(f"    [probe] step={step:4d}/{MLP_STEPS} mse={loss.item():.6f}", flush=True)

    return model


@torch.no_grad()
def predict_probe(model: nn.Module, z: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    zt = torch.as_tensor(z, dtype=torch.float32)
    out = []
    for start in range(0, len(zt), batch_size):
        stop = min(start + batch_size, len(zt))
        out.append(model(zt[start:stop].to(device)).cpu())
    return torch.cat(out, dim=0).numpy().astype(np.float32, copy=False)


def r2_vector(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    return np.asarray(
        [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])],
        dtype=np.float64,
    )


# =============================================================================
# ONE ARM
# =============================================================================
def run_arm(
    *,
    objective: str,          # "infonce" or "mse"
    label_mode: str,         # "all" or "label0"
    seed: int,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    Y_train_all: np.ndarray,
    Y_valid_all: np.ndarray,
    out_dir: Path,
):
    if objective not in {"infonce", "mse"}:
        raise ValueError(objective)
    if label_mode not in {"all", "label0"}:
        raise ValueError(label_mode)

    device = resolve_device()
    seed_all(seed)
    cleanup()

    # "label0" really behaves as a one-label dataset from this point onward.
    if label_mode == "all":
        y_train_raw = Y_train_all
        y_valid_raw = Y_valid_all
    else:
        y_train_raw = Y_train_all[:, :1]
        y_valid_raw = Y_valid_all[:, :1]

    y_train, y_valid, y_mu, y_sd = standardize_from_train(y_train_raw, y_valid_raw)
    cfg = make_cfg(seed, device)

    print("\n" + "=" * 88, flush=True)
    print(
        f"ARM objective={objective} label_mode={label_mode} seed={seed} "
        f"targets={y_train.shape[1]} device={device}",
        flush=True,
    )
    print("=" * 88, flush=True)

    # build_paired_models guarantees the InfoNCE and MSE backbones begin from
    # the same initialization for a given seed. Calling it again with the same
    # seed also makes all-label and label0 arms start from the same backbone.
    info_encoder, mse_model = build_paired_models(
        num_neurons=X_train.shape[1],
        target_dim=y_train.shape[1],
        cfg=cfg,
    )

    t0 = time.time()
    if objective == "infonce":
        del mse_model
        cleanup()
        encoder = info_encoder
        history = train_infonce(encoder, X_train, y_train, cfg, verbose_every=200)
        joint_pred = None
    else:
        del info_encoder
        cleanup()
        history = train_mse(mse_model, X_train, y_train, cfg, verbose_every=200)
        encoder = mse_model.encoder
        joint_pred = predict_joint_mse_batched(mse_model, X_valid, EVAL_BATCH_SIZE)

    encoder_train_seconds = time.time() - t0

    # ------------------------------------------------------------------
    # PRIMARY FAIR EVALUATION:
    # freeze encoder -> same fresh decoder for every arm.
    # ------------------------------------------------------------------
    print("  Encoding train/valid for the common frozen probe...", flush=True)
    z_train = encode_batched(encoder, X_train, EVAL_BATCH_SIZE)
    z_valid = encode_batched(encoder, X_valid, EVAL_BATCH_SIZE)

    probe = fit_probe(z_train, y_train, seed=seed, device=device)
    pred_probe = predict_probe(probe, z_valid, EVAL_BATCH_SIZE, device)

    probe_r2 = r2_vector(y_valid, pred_probe)
    probe_mean_r2 = float(probe_r2.mean())

    if joint_pred is not None:
        joint_r2 = r2_vector(y_valid, joint_pred)
        joint_mean_r2 = float(joint_r2.mean())
    else:
        joint_r2 = None
        joint_mean_r2 = None

    arm_name = f"{objective}_{'all_labels' if label_mode == 'all' else 'label0_only'}"
    arm_dir = out_dir / f"seed_{seed}" / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)

    np.save(arm_dir / "encoder_loss.npy", np.asarray(history, dtype=np.float32))
    np.save(arm_dir / "probe_r2_per_dim.npy", probe_r2)
    np.save(arm_dir / "label_mean.npy", y_mu)
    np.save(arm_dir / "label_std.npy", y_sd)

    if joint_r2 is not None:
        np.save(arm_dir / "joint_head_r2_per_dim.npy", joint_r2)

    if SAVE_CHECKPOINTS:
        torch.save(encoder.state_dict(), arm_dir / "encoder.pt")
        torch.save(probe.state_dict(), arm_dir / "probe.pt")
        if objective == "mse":
            torch.save(mse_model.state_dict(), arm_dir / "mse_joint_model.pt")

    result = {
        "seed": seed,
        "arm": arm_name,
        "objective": objective,
        "label_mode": label_mode,
        "num_targets": int(y_train.shape[1]),
        "probe_mean_r2": probe_mean_r2,
        "probe_label0_r2": float(probe_r2[0]),
        "probe_r2_per_dim": [float(v) for v in probe_r2],
        "joint_mean_r2": joint_mean_r2,
        "joint_label0_r2": None if joint_r2 is None else float(joint_r2[0]),
        "joint_r2_per_dim": None if joint_r2 is None else [float(v) for v in joint_r2],
        "encoder_train_seconds": float(encoder_train_seconds),
    }

    with open(arm_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"  PRIMARY frozen-probe R2 per dim: {probe_r2}", flush=True)
    print(f"  PRIMARY frozen-probe mean R2:    {probe_mean_r2:.6f}", flush=True)
    if joint_r2 is not None:
        print(f"  secondary MSE joint-head R2:     {joint_r2}", flush=True)

    # Free GPU memory before the next arm.
    del probe, encoder, z_train, z_valid
    if objective == "mse":
        del mse_model
    cleanup()

    return result


# =============================================================================
# MAIN
# =============================================================================
def main():
    X_train, X_valid, Y_train, Y_valid = load_data()

    print(f"Dataset: {NPZ_PATH}", flush=True)
    print(f"X_train: {X_train.shape} | X_valid: {X_valid.shape}", flush=True)
    print(f"Y_train: {Y_train.shape} | Y_valid: {Y_valid.shape}", flush=True)
    print(f"Label 0 only means y[:, 0:1], with every other label completely removed.", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    config_dump = {
        "session": SESSION,
        "npz_path": str(NPZ_PATH),
        "latent_dim": LATENT_DIM,
        "hidden": HIDDEN,
        "batch_size": BATCH_SIZE,
        "max_iter": MAX_ITER,
        "temperature": TEMPERATURE,
        "arch": ARCH,
        "time_offsets": TIME_OFFSETS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "mlp_steps": MLP_STEPS,
        "mlp_batch_size": MLP_BATCH_SIZE,
        "mlp_hidden": MLP_HIDDEN,
        "mlp_drop": MLP_DROP,
        "mlp_lr": MLP_LR,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "seeds": list(SEEDS),
        "primary_metric": "fresh frozen-encoder probe R2",
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dump, f, indent=2)

    all_results = []
    for seed in SEEDS:
        # Order chosen so objective changes slowly; it has no effect on initialization
        # because each arm is explicitly re-seeded and rebuilt from scratch.
        for objective, label_mode in (
            ("infonce", "all"),
            ("infonce", "label0"),
            ("mse", "all"),
            ("mse", "label0"),
        ):
            all_results.append(
                run_arm(
                    objective=objective,
                    label_mode=label_mode,
                    seed=seed,
                    X_train=X_train,
                    X_valid=X_valid,
                    Y_train_all=Y_train,
                    Y_valid_all=Y_valid,
                    out_dir=out_dir,
                )
            )

    # Long-form summary CSV.
    csv_path = out_dir / "summary.csv"
    fields = [
        "seed",
        "arm",
        "objective",
        "label_mode",
        "num_targets",
        "probe_mean_r2",
        "probe_label0_r2",
        "joint_mean_r2",
        "joint_label0_r2",
        "encoder_train_seconds",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: row.get(k) for k in fields})

    with open(out_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "#" * 88, flush=True)
    print("FINAL SUMMARY (primary = SAME fresh frozen-encoder MLP probe)", flush=True)
    print("#" * 88, flush=True)
    for r in all_results:
        print(
            f"seed={r['seed']:>3} | {r['arm']:<24} | "
            f"mean R2={r['probe_mean_r2']:+.5f} | "
            f"label0 R2={r['probe_label0_r2']:+.5f}",
            flush=True,
        )

    print(f"\nSaved to: {out_dir}", flush=True)
    print(f"Summary:  {csv_path}", flush=True)


if __name__ == "__main__":
    main()

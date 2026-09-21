"""
Label-ablation experiment: behavior-conditioned CEBRA vs a simple MSE encoder.

For one Perich session, run four experiments:
1) CEBRA trained with all labels -> same post-hoc decoder -> R2 all labels.
2) CEBRA trained with first N_SUBSET_LABELS -> same post-hoc decoder -> R2 subset labels.
3) Simple MSE encoder + same decoder trained end-to-end with all labels.
4) Simple MSE encoder + same decoder trained end-to-end with first N_SUBSET_LABELS.

Outputs:
- model/decoder checkpoints
- r2_summary.csv
- r2_per_label_long.csv
"""

from __future__ import annotations

import csv
import gc
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# PATHS / ORIGINAL CEBRA IMPORT
# =============================================================================

ROOT = Path(__file__).resolve().parent

# User said the fork folder is CEBRA-orginal. Fallback supports the common spelling.
_CEBRA_CANDIDATES = [ROOT / "CEBRA-orginal", ROOT / "CEBRA-original"]
CEBRA_DIR = next((p for p in _CEBRA_CANDIDATES if p.exists()), _CEBRA_CANDIDATES[0])

if not CEBRA_DIR.exists():
    raise FileNotFoundError(
        "Original CEBRA checkout not found. Tried:\n"
        + "\n".join(str(p) for p in _CEBRA_CANDIDATES)
    )

for module_name in list(sys.modules):
    if module_name == "cebra" or module_name.startswith("cebra."):
        del sys.modules[module_name]

for p in [str(x) for x in _CEBRA_CANDIDATES]:
    while p in sys.path:
        sys.path.remove(p)

sys.path.insert(0, str(CEBRA_DIR))

import cebra
from cebra import CEBRA

print("\nUsing ORIGINAL CEBRA:")
print(cebra.__file__)


# =============================================================================
# CONFIG
# =============================================================================

PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
TARGET_SESSION = "C-CO12"  # change this only to test another session
N_SUBSET_LABELS = 2
SEED = 42

# CEBRA
LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
CEBRA_MAX_ITER = 3000
TEMPERATURE = 0.4
MODEL_ARCH = "offset36-model-more-dropout"
DEVICE = "cuda_if_available"
OFFSET = 1
CONDITIONAL = "time_delta"

# Shared decoder: EXACT SAME architecture in all conditions.
DECODER_HIDDEN_DIM = 64
DECODER_DROPOUT = 0.4
DECODER_EPOCHS = 2500
DECODER_BATCH_SIZE = 256
DECODER_LR = 1e-3
DECODER_WEIGHT_DECAY = 1e-4
DECODER_PRINT_EVERY = 500

# Simple MSE encoder.
MSE_ENCODER_HIDDEN = 128
MSE_ENCODER_DROPOUT = 0.2
MSE_EPOCHS = 2500
MSE_BATCH_SIZE = 256
MSE_LR = 1e-3
MSE_WEIGHT_DECAY = 1e-4
MSE_PRINT_EVERY = 500

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = ROOT / f"LabelAblation_CEBRA_vs_MSE_{TARGET_SESSION}"
MODELS_DIR = OUT_DIR / "models"
CSV_SUMMARY = OUT_DIR / "r2_summary.csv"
CSV_LONG = OUT_DIR / "r2_per_label_long.csv"


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def seed_all(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_target_session():
    path = PERICH_DATA_DIR / f"{TARGET_SESSION}.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    d = np.load(path, allow_pickle=True)
    X_train = np.asarray(d["train_data"], dtype=np.float32)
    X_test = np.asarray(d["valid_data"], dtype=np.float32)
    Y_train = np.asarray(d["train_label"], dtype=np.float32)
    Y_test = np.asarray(d["valid_label"], dtype=np.float32)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(f"Expected 2D neural arrays, got {X_train.shape=} {X_test.shape=}")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train/test neuron count differs.")

    if Y_train.ndim == 1:
        Y_train = Y_train[:, None]
    if Y_test.ndim == 1:
        Y_test = Y_test[:, None]
    if Y_train.ndim != 2 or Y_test.ndim != 2:
        raise ValueError(f"Expected 2D labels, got {Y_train.shape=} {Y_test.shape=}")
    if Y_train.shape[1] != Y_test.shape[1]:
        raise ValueError("Train/test label count differs.")

    if len(X_train) != len(Y_train) or len(X_test) != len(Y_test):
        raise ValueError("Neural/label length mismatch.")

    for name, arr in (("X_train", X_train), ("X_test", X_test),
                      ("Y_train", Y_train), ("Y_test", Y_test)):
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{name} contains NaN/Inf.")

    if Y_train.shape[1] < N_SUBSET_LABELS:
        raise ValueError(
            f"Dataset has {Y_train.shape[1]} labels but N_SUBSET_LABELS={N_SUBSET_LABELS}."
        )

    print("\n" + "=" * 90)
    print("DATA")
    print("=" * 90)
    print("session :", TARGET_SESSION)
    print("X_train :", X_train.shape)
    print("X_test  :", X_test.shape)
    print("Y_train :", Y_train.shape)
    print("Y_test  :", Y_test.shape)
    print("subset labels:", list(range(N_SUBSET_LABELS)))

    return X_train, X_test, Y_train, Y_test


def align_embedding_and_labels(Z: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(Z), len(Y))
    return Z[:n], Y[:n]


def compute_r2(Y_true: np.ndarray, Y_pred: np.ndarray):
    r2_each = np.asarray(
        r2_score(Y_true, Y_pred, multioutput="raw_values"),
        dtype=float,
    )
    return r2_each, float(np.mean(r2_each))


# =============================================================================
# SAME DECODER FOR EVERY CONDITION
# =============================================================================

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.net(x)


def train_shared_decoder(display_name, Z_train, Z_test, Y_train, Y_test, save_path):
    Z_train, Ytr = align_embedding_and_labels(Z_train, Y_train)
    Z_test, Yte = align_embedding_and_labels(Z_test, Y_test)

    seed_all(SEED)
    decoder = TwoLayerMLP(
        input_dim=Z_train.shape[1],
        hidden_dim=DECODER_HIDDEN_DIM,
        output_dim=Ytr.shape[1],
        dropout_rate=DECODER_DROPOUT,
    ).to(TORCH_DEVICE)

    ds = TensorDataset(
        torch.from_numpy(Z_train.astype(np.float32)),
        torch.from_numpy(Ytr.astype(np.float32)),
    )
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        ds,
        batch_size=DECODER_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=g,
    )

    opt = torch.optim.Adam(
        decoder.parameters(),
        lr=DECODER_LR,
        weight_decay=DECODER_WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss()

    print(f"\nDecoder {display_name}: {DECODER_EPOCHS} epochs")
    for epoch in range(1, DECODER_EPOCHS + 1):
        decoder.train()
        for xb, yb in loader:
            xb = xb.to(TORCH_DEVICE)
            yb = yb.to(TORCH_DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(decoder(xb), yb)
            loss.backward()
            opt.step()
        if epoch == 1 or epoch % DECODER_PRINT_EVERY == 0 or epoch == DECODER_EPOCHS:
            print(f"{display_name}: decoder epoch {epoch}/{DECODER_EPOCHS}")

    decoder.eval()
    with torch.no_grad():
        pred = decoder(torch.from_numpy(Z_test.astype(np.float32)).to(TORCH_DEVICE)).cpu().numpy()

    r2_each, mean_r2 = compute_r2(Yte, pred)
    torch.save(decoder.state_dict(), save_path)

    print(f"{display_name} R2 per label:", r2_each)
    print(f"{display_name} Mean R2 = {mean_r2:.6f}")

    del decoder, opt
    cleanup()
    return r2_each, mean_r2


# =============================================================================
# CEBRA CONDITIONS
# =============================================================================

def build_cebra():
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=OFFSET,
        conditional=CONDITIONAL,
        max_iterations=CEBRA_MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        device=DEVICE,
        verbose=True,
    )


def run_cebra(condition_name, X_train, X_test, Y_train_cond, Y_test_cond):
    print("\n" + "#" * 100)
    print(f"CEBRA -- {condition_name}")
    print("#" * 100)
    print("labels used during CEBRA training:", Y_train_cond.shape[1])

    seed_all(SEED)
    model = build_cebra()
    model.fit(X_train, Y_train_cond)

    model_path = MODELS_DIR / f"cebra_{condition_name}.pt"
    model.save(str(model_path))

    Z_train = np.asarray(model.transform(X_train.astype(np.float32)), dtype=np.float32)
    Z_test = np.asarray(model.transform(X_test.astype(np.float32)), dtype=np.float32)

    r2_each, mean_r2 = train_shared_decoder(
        f"CEBRA {condition_name}",
        Z_train,
        Z_test,
        Y_train_cond,
        Y_test_cond,
        MODELS_DIR / f"cebra_{condition_name}_decoder.pt",
    )

    del model
    cleanup()
    return {
        "method": "CEBRA",
        "condition": condition_name,
        "n_train_labels": int(Y_train_cond.shape[1]),
        "r2_each": r2_each,
        "mean_r2": mean_r2,
    }


# =============================================================================
# SIMPLE MSE ENCODER + SAME DECODER
# =============================================================================

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, latent_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.net(x)


class MSEEncoderWithDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = SimpleEncoder(
            input_dim=input_dim,
            hidden_dim=MSE_ENCODER_HIDDEN,
            latent_dim=LATENT_DIM,
            dropout_rate=MSE_ENCODER_DROPOUT,
        )
        # SAME decoder class/architecture as CEBRA uses.
        self.decoder = TwoLayerMLP(
            input_dim=LATENT_DIM,
            hidden_dim=DECODER_HIDDEN_DIM,
            output_dim=output_dim,
            dropout_rate=DECODER_DROPOUT,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def run_mse(condition_name, X_train, X_test, Y_train_cond, Y_test_cond):
    print("\n" + "#" * 100)
    print(f"SIMPLE MSE ENCODER -- {condition_name}")
    print("#" * 100)
    print("labels used during MSE training:", Y_train_cond.shape[1])

    seed_all(SEED)
    model = MSEEncoderWithDecoder(
        input_dim=X_train.shape[1],
        output_dim=Y_train_cond.shape[1],
    ).to(TORCH_DEVICE)

    ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(Y_train_cond.astype(np.float32)),
    )
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        ds,
        batch_size=MSE_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=g,
    )

    opt = torch.optim.Adam(
        model.parameters(),
        lr=MSE_LR,
        weight_decay=MSE_WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss()

    for epoch in range(1, MSE_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(TORCH_DEVICE)
            yb = yb.to(TORCH_DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        if epoch == 1 or epoch % MSE_PRINT_EVERY == 0 or epoch == MSE_EPOCHS:
            print(f"MSE {condition_name}: epoch {epoch}/{MSE_EPOCHS}")

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_test.astype(np.float32)).to(TORCH_DEVICE)).cpu().numpy()

    r2_each, mean_r2 = compute_r2(Y_test_cond, pred)

    torch.save(model.encoder.state_dict(), MODELS_DIR / f"mse_{condition_name}_encoder.pt")
    torch.save(model.decoder.state_dict(), MODELS_DIR / f"mse_{condition_name}_decoder.pt")

    print(f"MSE {condition_name} R2 per label:", r2_each)
    print(f"MSE {condition_name} Mean R2 = {mean_r2:.6f}")

    del model, opt
    cleanup()
    return {
        "method": "MSE_ENCODER",
        "condition": condition_name,
        "n_train_labels": int(Y_train_cond.shape[1]),
        "r2_each": r2_each,
        "mean_r2": mean_r2,
    }


# =============================================================================
# RESULTS
# =============================================================================

def save_results(results):
    max_labels = max(len(r["r2_each"]) for r in results)

    with CSV_SUMMARY.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["session", "method", "condition", "n_train_labels", "mean_r2"]
            + [f"r2_label_{i}" for i in range(max_labels)]
        )
        for r in results:
            vals = r["r2_each"].tolist() + [""] * (max_labels - len(r["r2_each"]))
            w.writerow([
                TARGET_SESSION,
                r["method"],
                r["condition"],
                r["n_train_labels"],
                r["mean_r2"],
            ] + vals)

    with CSV_LONG.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "method", "condition", "n_train_labels", "label_index", "r2"])
        for r in results:
            for i, value in enumerate(r["r2_each"]):
                w.writerow([
                    TARGET_SESSION,
                    r["method"],
                    r["condition"],
                    r["n_train_labels"],
                    i,
                    float(value),
                ])

    print("\nsaved:", CSV_SUMMARY)
    print("saved:", CSV_LONG)


def print_final(results):
    print("\n" + "=" * 115)
    print("FINAL TEST R2")
    print("=" * 115)
    for r in results:
        print(
            f"{r['method']:<12} | {r['condition']:<12} | "
            f"n_labels={r['n_train_labels']:<3} | mean={r['mean_r2']:.6f} | "
            f"per-label={np.array2string(r['r2_each'], precision=4)}"
        )
    print("=" * 115)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ensure_dirs()
    seed_all(SEED)

    X_train, X_test, Y_train, Y_test = load_target_session()

    Y_train_2 = Y_train[:, :N_SUBSET_LABELS].copy().astype(np.float32)
    Y_test_2 = Y_test[:, :N_SUBSET_LABELS].copy().astype(np.float32)
    Y_train_all = Y_train.copy().astype(np.float32)
    Y_test_all = Y_test.copy().astype(np.float32)

    print("\nConditions:")
    print(f"first_{N_SUBSET_LABELS}: {N_SUBSET_LABELS} labels")
    print(f"all_labels: {Y_train.shape[1]} labels")

    results = []

    # CEBRA with first 2 labels.
    results.append(run_cebra(
        f"first_{N_SUBSET_LABELS}",
        X_train,
        X_test,
        Y_train_2,
        Y_test_2,
    ))

    # CEBRA with all labels.
    results.append(run_cebra(
        "all_labels",
        X_train,
        X_test,
        Y_train_all,
        Y_test_all,
    ))

    # Simple MSE encoder with first 2 labels.
    results.append(run_mse(
        f"first_{N_SUBSET_LABELS}",
        X_train,
        X_test,
        Y_train_2,
        Y_test_2,
    ))

    # Simple MSE encoder with all labels.
    results.append(run_mse(
        "all_labels",
        X_train,
        X_test,
        Y_train_all,
        Y_test_all,
    ))

    save_results(results)
    print_final(results)


if __name__ == "__main__":
    main()

import os
import gc
import sys
import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt

from utils.constants import CEBRA_DIR
from utils.min_distance import min_l2_distance

sys.path.insert(0, str(CEBRA_DIR))

import cebra
import cebra.attribution
from cebra import CEBRA


# =====================================================
# Config
# =====================================================
DATA_DIR = "./data/seed_model_training_data/mat"
SESSION_FILE = "t5.2022.05.18.mat"
MAT_PATH = os.path.join(DATA_DIR, SESSION_FILE)

TRIAL_ID = 0

# Per the dataset README: only area 6v carries the recommended signal for this
# style of analysis. In the 256-channel layout, area 6v = first 128 columns.
AREA_6V_CHANNELS = 128

IMG_DIR = "./image"
os.makedirs(IMG_DIR, exist_ok=True)

BATCH_SIZE = 256
MAX_ITER = 20000
OUTPUT_DIM = 16

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# Load data
# =====================================================
print(f"Loading session: {MAT_PATH}")
data = sio.loadmat(MAT_PATH)
present_keys = [k for k in data.keys() if not k.startswith("__")]
print("Available keys:", present_keys)


# =====================================================
# Generic cell-array helpers
# (scipy loadmat keeps MATLAB "S x 1" or "1 x S" cell arrays as object ndarrays)
# =====================================================
def cell_len(mat_cell):
    arr = np.asarray(mat_cell)
    return int(max(arr.shape))


def get_cell(mat_cell, idx):
    arr = np.asarray(mat_cell)
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            return arr[0, idx]
        elif arr.shape[1] == 1:
            return arr[idx, 0]
    return arr.flatten()[idx]


# =====================================================
# Helpers (shared)
# =====================================================
def normalize_trial(X):
    """Legacy per-trial z-scoring (fallback path only)."""
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    return ((X - mu) / sigma).astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)


def reduce_attr_map(arr):
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    else:
        arr = np.asarray(arr)

    arr = np.abs(arr)

    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    elif arr.ndim == 2:
        pass
    elif arr.ndim == 1:
        arr = arr[None, :]
    else:
        raise ValueError(f"Unsupported attribution shape: {arr.shape}")

    return arr.astype(np.float32)


def save_heatmap(arr, path, title, feature_boundary=None, feature_labels=None):
    plt.figure(figsize=(10, 6))
    plt.imshow(arr, aspect="auto", cmap="viridis")
    plt.colorbar(label="absolute attribution")

    if feature_boundary is not None:
        plt.axvline(feature_boundary, color="white", linestyle="--", linewidth=1)
        if feature_labels is not None:
            ymax = arr.shape[0]
            plt.text(feature_boundary / 2, -0.6, feature_labels[0],
                      ha="center", va="bottom", fontsize=8, color="black")
            plt.text(feature_boundary + (arr.shape[1] - feature_boundary) / 2, -0.6,
                      feature_labels[1], ha="center", va="bottom", fontsize=8, color="black")

    plt.xlabel("Neural feature / channel (area 6v)")
    plt.ylabel("Latent dimension")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved:", path)


# =====================================================
# Feature extraction: official format (spikePow + tx1, area 6v, blockwise z-score)
# =====================================================
def extract_features_official(data, trial_idx):
    spikePow_trial = np.asarray(get_cell(data["spikePow"], trial_idx), dtype=np.float32)
    tx1_trial = np.asarray(get_cell(data["tx1"], trial_idx), dtype=np.float32)

    spikePow_6v = spikePow_trial[:, :AREA_6V_CHANNELS]
    tx1_6v = tx1_trial[:, :AREA_6V_CHANNELS]

    return np.concatenate([spikePow_6v, tx1_6v], axis=1)


def get_block_id(data, trial_idx):
    return int(np.squeeze(get_cell(data["blockIdx"], trial_idx)))


def compute_blockwise_stats(data, trial_ids_in_block, feature_fn):
    feats = [feature_fn(data, tid) for tid in trial_ids_in_block]
    all_feats = np.concatenate(feats, axis=0)
    mu = all_feats.mean(axis=0, keepdims=True).astype(np.float32)
    sigma = (all_feats.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
    return mu, sigma


def decode_sentence_text(data, trial_idx):
    raw = get_cell(data["sentenceText"], trial_idx)
    raw = np.asarray(raw)
    try:
        chars = [chr(int(c)) for c in raw.flatten() if int(c) != 0]
        return "".join(chars).strip()
    except Exception:
        return str(raw)


# =====================================================
# Build X_trial: auto-detect dataset format
# =====================================================
USE_OFFICIAL_FORMAT = all(k in data for k in ("spikePow", "tx1", "blockIdx"))

if USE_OFFICIAL_FORMAT:
    print("\nDetected official format (spikePow / tx1 / blockIdx).")
    print("-> using area-6v [spikePow | tx1] features with BLOCKWISE z-scoring.")

    n_trials = cell_len(data["spikePow"])
    print("Number of trials in session:", n_trials)

    if TRIAL_ID >= n_trials:
        raise ValueError(f"TRIAL_ID={TRIAL_ID} invalid, session only has {n_trials} trials.")

    block_id = get_block_id(data, TRIAL_ID)
    block_ids_all = np.array([get_block_id(data, i) for i in range(n_trials)])
    trial_ids_in_block = np.where(block_ids_all == block_id)[0]
    print(f"Trial {TRIAL_ID} belongs to block {block_id} "
          f"({len(trial_ids_in_block)} trials share this block; used to compute z-score stats)")

    mu, sigma = compute_blockwise_stats(data, trial_ids_in_block, extract_features_official)

    X_raw = extract_features_official(data, TRIAL_ID)
    X_trial = ((X_raw - mu) / sigma).astype(np.float32)

    print("Selected trial:", TRIAL_ID, "| shape (time bins x features):", X_trial.shape,
          f"(={AREA_6V_CHANNELS} spikePow-6v + {AREA_6V_CHANNELS} tx1-6v)")

    if "sentenceText" in data:
        try:
            print("Sentence:", decode_sentence_text(data, TRIAL_ID))
        except Exception as e:
            print("Could not decode sentence text:", e)

    FEATURE_BOUNDARY = AREA_6V_CHANNELS
    FEATURE_LABELS = ("spikePow (6v)", "tx1 (6v)")

else:
    print("\nspikePow/tx1/blockIdx NOT found in this file.")
    print("-> falling back to legacy 'tx_feats' format (per-trial z-scoring; "
          "no block info available, so blockwise normalization from the README cannot be applied here).")

    tx_feats = data["tx_feats"]
    n_trials = tx_feats.shape[1]
    print("Number of trials in session:", n_trials)

    if TRIAL_ID >= n_trials:
        raise ValueError(f"TRIAL_ID={TRIAL_ID} invalid, session only has {n_trials} trials.")

    X_raw = tx_feats[0, TRIAL_ID].astype(np.float32)
    X_trial, mu, sigma = normalize_trial(X_raw)
    print("Selected trial:", TRIAL_ID, "| shape (time bins x features):", X_trial.shape)

    if "sentences" in data:
        try:
            print("Sentence label:", data["sentences"][TRIAL_ID])
        except Exception as e:
            print("Could not read sentence label:", e)

    FEATURE_BOUNDARY = None
    FEATURE_LABELS = None


# =====================================================
# Model training helpers
# =====================================================
def train_cebra_standard(X):
    """
    Plain (non-adversarial) CEBRA baseline, single-trial, self-supervised.
    NOTE: training_mode/adv_* kwargs are intentionally omitted so the fork's
    default (clean/self-supervised) path runs. If your CEBRA_DIR fork needs an
    explicit flag to disable adversarial training, set it here.
    """
    train_batch_size = min(BATCH_SIZE, len(X))

    model = CEBRA(
        batch_size=train_batch_size,
        temperature=0.4,
        model_architecture="offset10-model",
        time_offsets=10,
        max_iterations=MAX_ITER,
        output_dimension=OUTPUT_DIM,
        verbose=True,
        num_hidden_units=32,
        device="cuda_if_available",
    )

    labels = np.arange(len(X), dtype=np.float32)
    model.fit(X.astype(np.float32), labels)
    return model


def train_acorn(X):
    """Adversarially-robust CEBRA (ACORN), single-trial."""
    x_torch = torch.tensor(X, dtype=torch.float32)
    eps = float(min_l2_distance(x_torch)) / 2.0
    eps = max(eps, 1e-6)
    print("Training ACORN | epsilon:", eps)

    train_batch_size = min(BATCH_SIZE, len(X))

    model = CEBRA(
        batch_size=train_batch_size,
        temperature=0.4,
        model_architecture="offset10-model",
        time_offsets=10,
        max_iterations=MAX_ITER,
        output_dimension=OUTPUT_DIM,
        verbose=True,
        training_mode="adversarial",
        adv_alpha=eps / 5,
        adv_epsilon=eps,
        adv_steps=10,
        attack_norm="linf",
        num_hidden_units=32,
        device="cuda_if_available",
    )

    labels = np.arange(len(X), dtype=np.float32)
    model.fit(X.astype(np.float32), labels)
    return model, eps


def compute_attribution(model, X, model_name):
    encoder = model.solver_.model.to(device)
    if hasattr(encoder, "split_outputs"):
        encoder.split_outputs = False
    encoder.eval()

    x_tensor = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)

    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=encoder,
        input_data=x_tensor,
        output_dimension=OUTPUT_DIM,
    )

    result = method.compute_attribution_map(batch_size=min(128, len(X)))
    print(f"[{model_name}] attribution keys:", result.keys())

    if "jf-inv-svd" in result:
        jfinv_key = "jf-inv-svd"
    elif "jf-inv-lsq" in result:
        jfinv_key = "jf-inv-lsq"
    elif "jf-inv" in result:
        jfinv_key = "jf-inv"
    else:
        raise KeyError(f"No inverse attribution key found. Available: {list(result.keys())}")

    jf = reduce_attr_map(result["jf"])
    jfinv = reduce_attr_map(result[jfinv_key])

    del encoder, x_tensor, method, result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return jf, jfinv


def cleanup(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# =====================================================
# 1) Standard CEBRA
# =====================================================
print("\n" + "=" * 90)
print(f"Training standard CEBRA | session={SESSION_FILE} | trial={TRIAL_ID}")
print("=" * 90)

cebra_model = train_cebra_standard(X_trial)
cebra_jf, cebra_jfinv = compute_attribution(cebra_model, X_trial, "CEBRA")

save_heatmap(
    cebra_jf,
    os.path.join(IMG_DIR, f"CEBRA_trial{TRIAL_ID}_jf.png"),
    f"CEBRA - Jacobian attribution (trial {TRIAL_ID})",
    feature_boundary=FEATURE_BOUNDARY,
    feature_labels=FEATURE_LABELS,
)
save_heatmap(
    cebra_jfinv,
    os.path.join(IMG_DIR, f"CEBRA_trial{TRIAL_ID}_jfinv.png"),
    f"CEBRA - Inverse Jacobian attribution (trial {TRIAL_ID})",
    feature_boundary=FEATURE_BOUNDARY,
    feature_labels=FEATURE_LABELS,
)

cleanup(cebra_model)


# =====================================================
# 2) ACORN (adversarial CEBRA)
# =====================================================
print("\n" + "=" * 90)
print(f"Training ACORN | session={SESSION_FILE} | trial={TRIAL_ID}")
print("=" * 90)

acorn_model, eps = train_acorn(X_trial)
acorn_jf, acorn_jfinv = compute_attribution(acorn_model, X_trial, "ACORN")

save_heatmap(
    acorn_jf,
    os.path.join(IMG_DIR, f"ACORN_trial{TRIAL_ID}_jf.png"),
    f"ACORN (eps={eps:.4f}) - Jacobian attribution (trial {TRIAL_ID})",
    feature_boundary=FEATURE_BOUNDARY,
    feature_labels=FEATURE_LABELS,
)
save_heatmap(
    acorn_jfinv,
    os.path.join(IMG_DIR, f"ACORN_trial{TRIAL_ID}_jfinv.png"),
    f"ACORN (eps={eps:.4f}) - Inverse Jacobian attribution (trial {TRIAL_ID})",
    feature_boundary=FEATURE_BOUNDARY,
    feature_labels=FEATURE_LABELS,
)

cleanup(acorn_model)


print("\nDONE")
print("session:", SESSION_FILE, "| trial:", TRIAL_ID)
print("format used:", "official (spikePow+tx1, area6v, blockwise z-score)" if USE_OFFICIAL_FORMAT else "legacy tx_feats (per-trial z-score)")
print("saved heatmaps to:", IMG_DIR)

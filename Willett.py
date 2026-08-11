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

IMG_DIR = "./image"
os.makedirs(IMG_DIR, exist_ok=True)

BATCH_SIZE = 256
MAX_ITER = 5000
OUTPUT_DIM = 16

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# Load data (same loader logic as your prep script)
# =====================================================
print(f"Loading session: {MAT_PATH}")
data = sio.loadmat(MAT_PATH)
print("Available keys:", [k for k in data.keys() if not k.startswith("__")])

tx_feats = data["tx_feats"]
n_trials = tx_feats.shape[1]
print("Number of trials in session:", n_trials)

if TRIAL_ID >= n_trials:
    raise ValueError(
        f"TRIAL_ID={TRIAL_ID} is invalid. This session has only {n_trials} trials."
    )

X_trial = tx_feats[0, TRIAL_ID].astype(np.float32)
print("Selected trial:", TRIAL_ID, "| shape (time bins x features):", X_trial.shape)

if "sentences" in data:
    try:
        print("Sentence label:", data["sentences"][TRIAL_ID])
    except Exception as e:
        print("Could not read sentence label:", e)


# =====================================================
# Helpers
# =====================================================
def normalize_trial(X):
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


def save_heatmap(arr, path, title):
    plt.figure(figsize=(10, 6))
    plt.imshow(arr, aspect="auto", cmap="viridis")
    plt.colorbar(label="absolute attribution")
    plt.xlabel("Neural feature / channel")
    plt.ylabel("Latent dimension")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved:", path)


def train_cebra_standard(X):
    """
    Plain (non-adversarial) CEBRA baseline, single-trial, self-supervised.
    NOTE: I deliberately did NOT pass training_mode / adv_* kwargs here so the
    fork's own default (clean/self-supervised) training path is used. If your
    CEBRA_DIR fork requires an explicit flag (e.g. training_mode="self-supervised"
    or training_mode="clean") to disable adversarial training, set it here.
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
# Preprocess
# =====================================================
X_trial, mu, sigma = normalize_trial(X_trial)


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
)
save_heatmap(
    cebra_jfinv,
    os.path.join(IMG_DIR, f"CEBRA_trial{TRIAL_ID}_jfinv.png"),
    f"CEBRA - Inverse Jacobian attribution (trial {TRIAL_ID})",
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
)
save_heatmap(
    acorn_jfinv,
    os.path.join(IMG_DIR, f"ACORN_trial{TRIAL_ID}_jfinv.png"),
    f"ACORN (eps={eps:.4f}) - Inverse Jacobian attribution (trial {TRIAL_ID})",
)

cleanup(acorn_model)


print("\nDONE")
print("session:", SESSION_FILE, "| trial:", TRIAL_ID)
print("saved heatmaps to:", IMG_DIR)

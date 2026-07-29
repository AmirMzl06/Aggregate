import os
import gc
import random
import numpy as np
import torch
import pandas as pd

from sklearn.metrics import roc_auc_score
from utils.min_distance import min_l2_distance
from utils.constants import CEBRA_DIR

import sys
sys.path.insert(0, str(CEBRA_DIR))
import cebra
from cebra import CEBRA

# ============================================================
# 1) Synthetic Data Config & Generation (Torus Manifold)
# ============================================================
T = 100_000
D_LATENT = 2        # 2D Latent variables (Z1, Z2) representing Torus angles
N_NEURONS = 100     # Projected high-dimensional space
OUTPUT_DIM = D_LATENT

BATCH_SIZE = 2048
MAX_ITER = 2500
ATTR_BATCH_SIZE = 128

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = 88
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def generate_torus_sparse_data(T=T, N_neurons=N_NEURONS, noise_level=0.5, seed=RANDOM_SEED):
    """
    Generates synthetic neural data by mapping a 2D Torus manifold
    to a high-dimensional space using a sparse orthogonal matrix.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Generate Latents (Z1, Z2) -> Continuous trajectories on a 2D Torus
    # We use random walks wrapped around a circle to create smooth toroidal trajectories
    t = np.linspace(0, 50 * np.pi, T)
    
    # Add cumulative Gaussian noise to simulate realistic continuous exploration
    theta = np.mod(t + 0.2 * rng.standard_normal(T).cumsum(), 2 * np.pi)
    phi = np.mod(t * 0.73 + 0.2 * rng.standard_normal(T).cumsum(), 2 * np.pi)
    
    # Z inputs: we use sin/cos to avoid the 2*pi discontinuity in the latent space
    z1 = np.sin(theta)
    z2 = np.sin(phi)
    
    latent = np.vstack([z1, z2]).T.astype(np.float32)  # Shape: (T, 2)
    
    # 2. Generate Sparse Orthogonal Projection Matrix (W)
    # W has shape (2, N_neurons). Half for Z1, half for Z2.
    # The dot product of the two rows is strictly 0 (Orthogonal).
    W = np.zeros((D_LATENT, N_neurons), dtype=np.float32)
    
    # Assign random weights to the non-zero connections (Sparse)
    weights = rng.uniform(0.5, 1.5, size=N_neurons) * rng.choice([-1, 1], size=N_neurons)
    
    for i in range(N_neurons):
        if i < N_neurons // 2:
            W[0, i] = weights[i]
        else:
            W[1, i] = weights[i]
            
    # 3. Project to High-Dimensional Space and Add Gaussian Noise
    # X = Z * W + Noise
    X = latent @ W + noise_level * rng.standard_normal((T, N_neurons)).astype(np.float32)
    
    # 4. Create Ground Truth Mask
    gt_bool = (W != 0)
    gt_attr = gt_bool.astype(np.float32)
    
    return X, latent, gt_attr, gt_bool


# ============================================================
# 2) Utils
# ============================================================
def cleanup_cuda(*objs):
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def reduce_attr_map(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return np.abs(arr).mean(axis=0)
    if arr.ndim == 2:
        return np.abs(arr)
    if arr.ndim == 1:
        return np.abs(arr)[None, :]
    raise ValueError(f"Unsupported attribution shape: {arr.shape}")


def compute_auroc(attr_map_2d, gt_bool):
    y_true = gt_bool.ravel().astype(int)
    y_score = np.asarray(attr_map_2d, dtype=np.float64).ravel()

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            f"Shape mismatch: y_true has {y_true.shape[0]} elements, "
            f"y_score has {y_score.shape[0]} elements."
        )

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(roc_auc_score(y_true, y_score))


# ============================================================
# 3) Data Generation
# ============================================================
print("Generating synthetic Geometric Manifold (Torus) dataset...")
x_np, y_np, gt_attr, gt_attr_bool = generate_torus_sparse_data(T=T, seed=RANDOM_SEED)

print("X (Neural Space) shape:", x_np.shape)              # (T, 100)
print("Z (Latent Manifold) shape:", y_np.shape)           # (T, 2)
print("Ground Truth Mask shape:", gt_attr_bool.shape)     # (2, 100)

split_idx = int(0.8 * len(x_np))
train_data = x_np[:split_idx].astype(np.float32)
train_continuous_label = y_np[:split_idx].astype(np.float32)

# Dynamic epsilon calculation
adv_epsilon = float(min_l2_distance(train_data)) / 2.0
adv_epsilon = max(adv_epsilon, 1e-6)

# ============================================================
# 4) Train + Attribution extraction
# ============================================================
rows = []
all_results = {}

for adv in [False, True]:
    cleanup_cuda()

    model_name = "ACORN" if adv else "CEBRA"
    training_mode = "adversarial" if adv else "clean"

    print("\n" + "=" * 70)
    print(f"Training {model_name} ({training_mode} mode)")
    print("=" * 70)

    model = CEBRA(
        batch_size=BATCH_SIZE,
        temperature=0.4,
        model_architecture="offset36-model-more-dropout",
        time_offsets=4,
        max_iterations=MAX_ITER,
        output_dimension=OUTPUT_DIM,
        verbose=True,
        training_mode=training_mode,
        adv_alpha=adv_epsilon / 5,
        adv_epsilon=adv_epsilon,
        adv_steps=10,
        attack_norm="linf",
        num_hidden_units=32,
    )

    model.fit(train_data, train_continuous_label)

    save_path = os.path.join(OUT_DIR, f"{model_name}_torus.pth")
    model.save(save_path)
    print("Saved model to:", save_path)

    trained_model = model.solver_.model.to(device)
    input_tensor = torch.from_numpy(train_data).float().to(device).requires_grad_(True)
    output_dim = int(getattr(trained_model, "num_output", OUTPUT_DIM))

    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=trained_model,
        input_data=input_tensor,
        output_dimension=output_dim,
    )

    result = method.compute_attribution_map(batch_size=min(128, len(train_data)))
    print("Attribution keys:", list(result.keys()))

    # Reduce maps to 2D
    jc_map = reduce_attr_map(result["jf"])                      # (2, 100)
    jc_inv_map = reduce_attr_map(result["jf-inv-svd"])          # (100, 2)
    jc_invconv_map = reduce_attr_map(result["jf-convabs-inv-svd"])  # (100, 2)

    # Note: Transpose inverse matrices to align with the (2, 100) GT mask
    auc_jc = compute_auroc(jc_map, gt_attr_bool)
    auc_jc_inv = compute_auroc(jc_inv_map.T, gt_attr_bool)
    auc_jc_invconv = compute_auroc(jc_invconv_map.T, gt_attr_bool)

    print(f"** {model_name} jc AUROC:        {auc_jc:.4f} **")
    print(f"** {model_name} jc_inv AUROC:    {auc_jc_inv:.4f} **")
    print(f"** {model_name} jc_invconv AUROC:{auc_jc_invconv:.4f} **")

    all_results[model_name] = {
        "auc_jc": auc_jc,
        "auc_jc_inv": auc_jc_inv,
        "auc_jc_invconv": auc_jc_invconv,
    }

    rows.extend([
        {"model": model_name, "metric": "jc", "auroc": auc_jc},
        {"model": model_name, "metric": "jc_inv", "auroc": auc_jc_inv},
        {"model": model_name, "metric": "jc_invconv", "auroc": auc_jc_invconv},
    ])

    cleanup_cuda(method, trained_model, input_tensor, model)

# ============================================================
# 5) Summary
# ============================================================
print("\n" + "=" * 80)
print(" SUMMARY OF EXPERIMENT RESULTS (TORUS MANIFOLD) ".center(80, "="))
print("=" * 80)
for model_name, res in all_results.items():
    print(
        f" Model: {model_name:<6} | "
        f"jc={res['auc_jc']:.4f} | "
        f"jc_inv={res['auc_jc_inv']:.4f} | "
        f"jc_invconv={res['auc_jc_invconv']:.4f}"
    )
print("=" * 80)

# ============================================================
# 6) Save CSV
# ============================================================
results_df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "torus_auroc_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"Saved AUROC results to: {csv_path}")

print("Done.")

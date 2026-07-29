import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from sklearn.metrics import roc_auc_score
from utils.min_distance import min_l2_distance
from utils.constants import CEBRA_DIR

import sys
sys.path.insert(0, str(CEBRA_DIR))
import cebra
from cebra import CEBRA

# ============================================================
# 1) Synthetic Data Config & Generation (Poisson Place Cell)
# ============================================================
T = 100_000
D_LATENT = 2       # Z1 and Z2 (Position in 2D space)
N_NEURONS = 3      # Total observed neurons
OUTPUT_DIM = D_LATENT  # 2

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


def generate_poisson_place_cells(T=T, N_neurons=N_NEURONS, max_firing_rate=15.0, seed=RANDOM_SEED):
    """
    Generates synthetic neural data using a Poisson Place Cell model.
    Half of the neurons are tuned to Z1, the other half to Z2.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Generate Latents (Z1, Z2) -> Simulated continuous trajectories
    t = np.linspace(0, 50 * np.pi, T)
    z1 = np.sin(t) + 0.1 * rng.standard_normal(T)
    z2 = np.cos(t * 0.7) + 0.1 * rng.standard_normal(T)
    
    # Min-Max Normalize to [-1, 1]
    z1 = 2 * (z1 - z1.min()) / (z1.max() - z1.min()) - 1
    z2 = 2 * (z2 - z2.min()) / (z2.max() - z2.min()) - 1
    
    latent = np.vstack([z1, z2]).T.astype(np.float32)  # Shape: (T, 2)
    
    # 2. Setup Neurons and Ground Truth Map
    x = np.zeros((T, N_neurons), dtype=np.float32)
    gt_bool = np.zeros((2, N_neurons), dtype=bool)
    
    centers = rng.uniform(-0.8, 0.8, size=N_neurons)
    sigma = 0.2  # Tuning curve width
    
    # 3. Generate Spikes
    for i in range(N_neurons):
        if i < N_neurons // 2:
            # First half tuned to Z1
            distance_sq = (latent[:, 0] - centers[i])**2
            gt_bool[0, i] = True
        else:
            # Second half tuned to Z2
            distance_sq = (latent[:, 1] - centers[i])**2
            gt_bool[1, i] = True
            
        firing_rate = max_firing_rate * np.exp(-distance_sq / (2 * sigma**2))
        spikes = rng.poisson(firing_rate)
        
        # Add background continuous noise for stability with CEBRA
        signal = spikes + rng.normal(0, 0.5, size=T)
        x[:, i] = np.clip(signal, 0, None)
        
    gt_attr = gt_bool.astype(np.float32)
    return x, latent, gt_attr, gt_bool


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
    """
    Convert attribution output to 2D map [output_dim, input_dim]
    if it has sample dimension.
    """
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
            f"y_score has {y_score.shape[0]} elements, "
            f"attr_map shape={attr_map_2d.shape}, gt shape={gt_bool.shape}"
        )

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(roc_auc_score(y_true, y_score))


# ============================================================
# 3) Data Generation
# ============================================================
print("Generating synthetic Poisson Place Cell dataset...")
x_np, y_np, gt_attr, gt_attr_bool = generate_poisson_place_cells(T=T, seed=RANDOM_SEED)

print("x shape:", x_np.shape)             # (T, 50)
print("y shape:", y_np.shape)             # (T, 2)
print("gt_attr_bool shape:", gt_attr_bool.shape)  # (2, 50)

split_idx = int(0.8 * len(x_np))
train_data = x_np[:split_idx].astype(np.float32)
train_continuous_label = y_np[:split_idx].astype(np.float32)

# For later use in CEBRA setup
adv_epsilon = float(min_l2_distance(train_data)) / 2.0
adv_epsilon = max(adv_epsilon, 1e-6)

# ============================================================
# 4) Train + Attribution
# ============================================================
rows = []
all_results = {}

for adv in [False, True]:
    cleanup_cuda()

    model_name = "ACORN" if adv else "CEBRA"
    training_mode = "adversarial" if adv else "clean"

    print("\n" + "=" * 70)
    print(f"Training {model_name} ({training_mode})")
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

    save_path = os.path.join(OUT_DIR, f"{model_name}_synthetic.pth")
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

    # Reduce to 2D maps
    jc_map = reduce_attr_map(result["jf"])                      # expected shape: (2, 50)
    jc_inv_map = reduce_attr_map(result["jf-inv-svd"])          # expected shape: (50, 2)
    jc_invconv_map = reduce_attr_map(result["jf-convabs-inv-svd"])  # expected shape: (50, 2)

    # AUROC scores
    auc_jc = compute_auroc(jc_map, gt_attr_bool)
    auc_jc_inv = compute_auroc(jc_inv_map.T, gt_attr_bool)
    auc_jc_invconv = compute_auroc(jc_invconv_map.T, gt_attr_bool)

    print(f"** {model_name} jc AUROC:        {auc_jc:.4f} **")
    print(f"** {model_name} jc_inv AUROC:    {auc_jc_inv:.4f} **")
    print(f"** {model_name} jc_invconv AUROC:{auc_jc_invconv:.4f} **")

    all_results[model_name] = {
        "jc": jc_map,
        "jc_inv": jc_inv_map,
        "jc_invconv": jc_invconv_map,
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
print(" SUMMARY OF EXPERIMENT RESULTS ".center(80, "="))
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
csv_path = os.path.join(OUT_DIR, "synthetic_auroc_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"Saved AUROC results to: {csv_path}")

print("Done.")

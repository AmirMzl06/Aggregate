import os
import sys
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from utils.min_distance import min_l2_distance

from utils.constants import CEBRA_DIR, DATA_DIR
from utils.dataset_loader import DatasetLoader

sys.path.insert(0, str(CEBRA_DIR))
import cebra
from cebra import CEBRA

# -----------------------------
# 1. Synthetic Data Config & Generation Functions
# -----------------------------
T = 100_000  
D1 = 24
D2 = 24
D_LATENT = D1 + D2
N1 = 24
N2 = 24
D_OBS = N1 + N2 
N_MLP_LAYERS = 4
SIGMA_EPS = 0.03

def make_mlp(in_dim, out_dim, n_layers=4, seed=0):
    torch.manual_seed(seed)
    layers = []
    d_in = in_dim
    hidden = in_dim * 10

    for i in range(n_layers - 1):
        d_h = in_dim * 30 if i < n_layers - 2 else hidden
        lin = nn.Linear(d_in, d_h)
        nn.init.orthogonal_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.GELU()]
        d_in = d_h

    lin = nn.Linear(d_in, out_dim)
    nn.init.orthogonal_(lin.weight)
    nn.init.zeros_(lin.bias)
    layers.append(lin)

    mlp = nn.Sequential(*layers)
    for p in mlp.parameters():
        p.requires_grad_(False)
    return mlp.eval()

def brownian_motion_box(T, d, sigma=0.03, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((T, d), dtype=np.float32)
    x[0] = rng.uniform(-1.0, 1.0, size=d).astype(np.float32)

    for t in range(T - 1):
        step = rng.normal(loc=0.0, scale=sigma, size=d).astype(np.float32)
        x[t + 1] = np.clip(x[t] + step, -1.0, 1.0)
    return x

def make_binary_ground_truth(D1, D2, N1, N2):
    gt = np.zeros((D1 + D2, N1 + N2), dtype=bool)
    gt[:D1, :] = True
    gt[D1:, N1:] = True
    return gt

def generate_synthetic_data(T=T, seed=42):
    z1 = brownian_motion_box(T, D1, sigma=SIGMA_EPS, seed=seed)
    z2 = brownian_motion_box(T, D2, sigma=SIGMA_EPS, seed=seed + 1)

    g1 = make_mlp(D1, N1, n_layers=N_MLP_LAYERS, seed=seed + 10)
    g2 = make_mlp(D1 + D2, N2, n_layers=N_MLP_LAYERS, seed=seed + 20)

    z1_t = torch.tensor(z1, dtype=torch.float32)
    z2_t = torch.tensor(z2, dtype=torch.float32)

    with torch.no_grad():
        x1 = g1(z1_t).cpu().numpy()
        x2 = g2(torch.cat([z1_t, z2_t], dim=1)).cpu().numpy()

    x = np.concatenate([x1, x2], axis=1).astype(np.float32)
    latent = np.concatenate([z1, z2], axis=1).astype(np.float32)

    gt_bool = make_binary_ground_truth(D1, D2, N1, N2)
    gt_attr = gt_bool.astype(np.float32)

    return x, latent, gt_attr, gt_bool

# -----------------------------
# Config (Directories & Device)
# -----------------------------
target_file = "Synthetic_Data_Experiment"
out_dir = "outputs"
img_dir = "images"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Helpers
# -----------------------------
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

# -----------------------------
# Data Generation & Train/Valid Split
# -----------------------------
print("Generating synthetic dataset matching CEBRA configurations...")
x_np, y_np, gt_attr, gt_attr_bool = generate_synthetic_data(T=T, seed=42)

print("x shape (Neurons):", x_np.shape)
print("y shape (True Latents):", y_np.shape)
print("gt_attr_bool shape:", gt_attr_bool.shape)

split_idx = int(0.8 * len(x_np))
train_data = x_np[:split_idx].astype(np.float32)
train_continuous_label = y_np[:split_idx].astype(np.float32)

results = {}
auroc_results = {}

# -----------------------------
# Train CEBRA / ACORN & Compute Attribution
# -----------------------------
for adv in [False, True]:
    cleanup_cuda()

    model_name = "ACORN" if adv else "CEBRA"
    print(f"\n==================== Training {model_name} ====================")
    
    adv_epsilon = min_l2_distance(train_data) / 2
    model = CEBRA(
        batch_size=2048,
        temperature=0.4,
        model_architecture="offset36-model-more-dropout",
        time_offsets=4,
        max_iterations=2500,
        output_dimension=48,
        verbose=True,
        training_mode="adversarial" if adv else "clean",
        adv_alpha=adv_epsilon / 5,
        adv_epsilon=adv_epsilon,
        adv_steps=10,
        attack_norm="linf",
        num_hidden_units=32
    )

    model.fit(train_data, train_continuous_label)

    save_path = os.path.join(out_dir, f"{model_name}_{target_file}.pth")
    model.save(save_path)
    print("Saved model to:", save_path)

    # -----------------------------
    # Attribution Map Calculation
    # -----------------------------
    trained_model = model.solver_.model.to(device)

    N_ATTR = min(512, len(train_data))
    attr_idx = np.random.choice(len(train_data), N_ATTR, replace=False)
    attr_data = train_data[attr_idx]

    input_tensor = torch.from_numpy(attr_data).float().to(device).requires_grad_(True)
    output_dim = int(getattr(trained_model, "num_output", 48))
    
    method = cebra.attribution.init(
        name="jacobian-based",
        model=trained_model,
        input_data=input_tensor,
        output_dimension=output_dim,
    )
    result = method.compute_attribution_map()
    results[model_name] = result

    jf = abs(result["jf"]).mean(0) 
    
    y_true = gt_attr_bool.T.ravel().astype(int)
    y_score = jf.ravel()
    
    if len(np.unique(y_true)) >= 2:
        current_auroc = roc_auc_score(y_true, y_score)
    else:
        current_auroc = float("nan")
        
    auroc_results[model_name] = current_auroc
    print(f"** {model_name} Attribution Map AUROC: {current_auroc:.4f} **")

    cleanup_cuda(method, trained_model, input_tensor, attr_data, model)

# -----------------------------
# Print AUROC Summary
# -----------------------------
print("\n" + "=" * 55)
print(" SUMMARY OF EXPERIMENT RESULTS ".center(55, "="))
print("=" * 55)
for name in auroc_results.keys():
    print(f" Model: {name:<6} | Attribution AUROC: {auroc_results[name]:.4f}")
print("=" * 55)

# -----------------------------
# Plot and Save Jacobians
# -----------------------------
# fig, axes = plt.subplots(1, 2, figsize=(15, 8)) 
# model_names = ["CEBRA", "ACORN"]
# ims = []

# for ax, name in zip(axes, model_names):
#     result = results[name]
#     jf = abs(result["jf"]).mean(0)
#     jf = jf / jf.sum()

#     n_rows, n_cols = jf.shape

#     im = ax.matshow(
#         jf,
#         aspect="auto", 
#     )
#     ims.append(im)

#     ax.set_title(f"{name}\nAUROC={auroc_results[name]:.3f}", pad=20) 
#     ax.set_xlabel(f"Latent Dimension ({n_cols})")
#     ax.set_ylabel(f"Neuron ({n_rows})")

# fig.subplots_adjust(right=0.85, top=0.85) 

# cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
# fig.colorbar(ims[0], cax=cbar_ax)

# safe_target = target_file.replace(".", "_")
# fig_path = os.path.join(
#     img_dir,
#     f"{safe_target}_CEBRA_vs_ACORN.png",
# )

# plt.savefig(
#     fig_path,
#     dpi=300,
#     bbox_inches="tight",
# )

# plt.show()

print("Saved figure to:", fig_path)
print("Attribution AUROCs:", auroc_results)

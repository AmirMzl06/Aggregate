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
# 1) Synthetic Data Config & Generation
# ============================================================
T = 100_000
D1 = 2
D2 = 2
D_LATENT = D1 + D2

N1 = 3
N2 = 3
D_OBS = N1 + N2

N_MLP_LAYERS = 4
SIGMA_EPS = 0.03

OUTPUT_DIM = 4
BATCH_SIZE = 2048
MAX_ITER = 2500
adv_epsilon_default = 0.5

ATTR_BATCH_SIZE = 128

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


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
    """
    Ground truth:
    - first D1 latents connect to all neurons
    - last D2 latents connect only to x2 block (last N2 neurons)
    shape = [D_LATENT, D_OBS]
    """
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
print("Generating synthetic dataset...")
x_np, y_np, gt_attr, gt_attr_bool = generate_synthetic_data(T=T, seed=42)

print("x shape:", x_np.shape)  # (T, 6)
print("y shape:", y_np.shape)  # (T, 4)
print("gt_attr_bool shape:", gt_attr_bool.shape)  # (4, 6)

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

    # Use full training data for attribution; batch_size smaller to avoid errors
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
    jc_map = reduce_attr_map(result["jf"])
    jc_inv_map = reduce_attr_map(result["jf-inv-svd"])
    jc_invconv_map = reduce_attr_map(result["jf-convabs-inv-svd"])

    # AUROC scores
    auc_jc = compute_auroc(jc_map, gt_attr_bool)
    auc_jc_inv = compute_auroc(jc_inv_map, gt_attr_bool)
    auc_jc_invconv = compute_auroc(jc_invconv_map, gt_attr_bool)

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

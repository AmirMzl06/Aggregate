# Lorenzo
# import os
# import gc
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import pandas as pd

# from sklearn.metrics import roc_auc_score
# from utils.min_distance import min_l2_distance
# from utils.constants import CEBRA_DIR

# import sys
# sys.path.insert(0, str(CEBRA_DIR))
# import cebra
# from cebra import CEBRA


# # ============================================================
# # 1) Synthetic Data Config & Generation
# # ============================================================
# T = 100_000
# D1 = 3  # Lorenz system latents
# D2 = 3  # Lorenz system latents
# D_LATENT = D1 + D2  # 6

# N1 = 25
# N2 = 25
# D_OBS = N1 + N2     # 4

# N_MLP_LAYERS = 4
# SIGMA_EPS = 0.03

# # Output dimension must match latent dimension
# OUTPUT_DIM = D_LATENT  # 6
# BATCH_SIZE = 2048
# MAX_ITER = 2500
# adv_epsilon_default = 0.5

# ATTR_BATCH_SIZE = 128

# OUT_DIR = "outputs"
# os.makedirs(OUT_DIR, exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RANDOM_SEED = 88
# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)
# random.seed(RANDOM_SEED)


# def make_mlp(in_dim, out_dim, n_layers=4, seed=0):
#     torch.manual_seed(seed)
#     layers = []
#     d_in = in_dim
#     hidden = in_dim * 10

#     for i in range(n_layers - 1):
#         d_h = in_dim * 30 if i < n_layers - 2 else hidden
#         lin = nn.Linear(d_in, d_h)
#         nn.init.orthogonal_(lin.weight)
#         nn.init.zeros_(lin.bias)
#         layers += [lin, nn.GELU()]
#         d_in = d_h

#     lin = nn.Linear(d_in, out_dim)
#     nn.init.orthogonal_(lin.weight)
#     nn.init.zeros_(lin.bias)
#     layers.append(lin)

#     mlp = nn.Sequential(*layers)
#     for p in mlp.parameters():
#         p.requires_grad_(False)
#     return mlp.eval()


# def brownian_motion_box(T, d, sigma=0.03, seed=0):
#     rng = np.random.default_rng(seed)
#     x = np.zeros((T, d), dtype=np.float32)
#     x[0] = rng.uniform(-1.0, 1.0, size=d).astype(np.float32)

#     for t in range(T - 1):
#         step = rng.normal(loc=0.0, scale=sigma, size=d).astype(np.float32)
#         x[t + 1] = np.clip(x[t] + step, -1.0, 1.0)
#     return x


# def lorenz_system(T, dt=0.01, seed=0):
#     """
#     Generates T steps of the Lorenz attractor.
#     Standard parameters: sigma=10, rho=28, beta=8/3
#     """
#     rng = np.random.default_rng(seed)

#     sigma = 10.0
#     rho = 28.0
#     beta = 8.0 / 3.0

#     xyz = np.zeros((T, 3), dtype=np.float32)

#     # seed-dependent initial condition so z1 and z2 are different
#     xyz[0] = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)

#     for t in range(T - 1):
#         x, y, z = xyz[t]
#         dx = sigma * (y - x) * dt
#         dy = (x * (rho - z) - y) * dt
#         dz = (x * y - beta * z) * dt
#         xyz[t + 1] = [x + dx, y + dy, z + dz]

#     # Min-Max Normalize to [-1, 1]
#     xyz_min = xyz.min(axis=0, keepdims=True)
#     xyz_max = xyz.max(axis=0, keepdims=True)
#     xyz_norm = 2.0 * (xyz - xyz_min) / (xyz_max - xyz_min + 1e-8) - 1.0

#     return xyz_norm.astype(np.float32)


# def make_binary_ground_truth(D1, D2, N1, N2):
#     """
#     Ground truth:
#     - first D1 latents connect to all neurons
#     - last D2 latents connect only to x2 block (last N2 neurons)
#     shape = [D_LATENT, D_OBS] -> [6, 4]
#     """
#     gt = np.zeros((D1 + D2, N1 + N2), dtype=bool)
#     gt[:D1, :] = True
#     gt[D1:, N1:] = True
#     return gt


# def generate_synthetic_data(T=T, seed=42):
#     # z1 uses Lorenz System (3 dimensions)
#     z1 = lorenz_system(T, dt=0.01, seed=seed)

#     # z2 also uses Lorenz System (3 dimensions)
#     z2 = lorenz_system(T, dt=0.01, seed=seed + 1)

#     g1 = make_mlp(D1, N1, n_layers=N_MLP_LAYERS, seed=seed + 10)
#     g2 = make_mlp(D1 + D2, N2, n_layers=N_MLP_LAYERS, seed=seed + 20)

#     z1_t = torch.tensor(z1, dtype=torch.float32)
#     z2_t = torch.tensor(z2, dtype=torch.float32)

#     with torch.no_grad():
#         x1 = g1(z1_t).cpu().numpy()
#         x2 = g2(torch.cat([z1_t, z2_t], dim=1)).cpu().numpy()

#     x = np.concatenate([x1, x2], axis=1).astype(np.float32)
#     latent = np.concatenate([z1, z2], axis=1).astype(np.float32)

#     gt_bool = make_binary_ground_truth(D1, D2, N1, N2)
#     gt_attr = gt_bool.astype(np.float32)

#     return x, latent, gt_attr, gt_bool


# # ============================================================
# # 2) Utils
# # ============================================================
# def cleanup_cuda(*objs):
#     for obj in objs:
#         try:
#             del obj
#         except Exception:
#             pass
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()


# def reduce_attr_map(arr):
#     """
#     Convert attribution output to 2D map [output_dim, input_dim]
#     if it has sample dimension.
#     """
#     arr = np.asarray(arr)
#     if arr.ndim == 3:
#         return np.abs(arr).mean(axis=0)
#     if arr.ndim == 2:
#         return np.abs(arr)
#     if arr.ndim == 1:
#         return np.abs(arr)[None, :]
#     raise ValueError(f"Unsupported attribution shape: {arr.shape}")


# def compute_auroc(attr_map_2d, gt_bool):
#     y_true = gt_bool.ravel().astype(int)
#     y_score = np.asarray(attr_map_2d, dtype=np.float64).ravel()

#     if y_true.shape[0] != y_score.shape[0]:
#         raise ValueError(
#             f"Shape mismatch: y_true has {y_true.shape[0]} elements, "
#             f"y_score has {y_score.shape[0]} elements, "
#             f"attr_map shape={attr_map_2d.shape}, gt shape={gt_bool.shape}"
#         )

#     if len(np.unique(y_true)) < 2:
#         return float("nan")

#     return float(roc_auc_score(y_true, y_score))


# # ============================================================
# # 3) Data Generation
# # ============================================================
# print("Generating synthetic dataset...")
# x_np, y_np, gt_attr, gt_attr_bool = generate_synthetic_data(T=T, seed=42)

# print("x shape:", x_np.shape)             # (T, 4)
# print("y shape:", y_np.shape)             # (T, 6)
# print("gt_attr_bool shape:", gt_attr_bool.shape)  # (6, 4)

# split_idx = int(0.8 * len(x_np))
# train_data = x_np[:split_idx].astype(np.float32)
# train_continuous_label = y_np[:split_idx].astype(np.float32)

# # For later use in CEBRA setup
# adv_epsilon = float(min_l2_distance(train_data)) / 2.0
# adv_epsilon = max(adv_epsilon, 1e-6)

# # ============================================================
# # 4) Train + Attribution
# # ============================================================
# rows = []
# all_results = {}

# for adv in [False, True]:
#     cleanup_cuda()

#     model_name = "ACORN" if adv else "CEBRA"
#     training_mode = "adversarial" if adv else "clean"

#     print("\n" + "=" * 70)
#     print(f"Training {model_name} ({training_mode})")
#     print("=" * 70)

#     model = CEBRA(
#         batch_size=BATCH_SIZE,
#         temperature=0.4,
#         model_architecture="offset36-model-more-dropout",
#         time_offsets=4,
#         max_iterations=MAX_ITER,
#         output_dimension=OUTPUT_DIM,
#         verbose=True,
#         training_mode=training_mode,
#         adv_alpha=adv_epsilon / 5,
#         adv_epsilon=adv_epsilon,
#         adv_steps=10,
#         attack_norm="linf",
#         num_hidden_units=32,
#     )

#     model.fit(train_data, train_continuous_label)

#     save_path = os.path.join(OUT_DIR, f"{model_name}_synthetic.pth")
#     model.save(save_path)
#     print("Saved model to:", save_path)

#     trained_model = model.solver_.model.to(device)

#     input_tensor = torch.from_numpy(train_data).float().to(device).requires_grad_(True)
#     output_dim = int(getattr(trained_model, "num_output", OUTPUT_DIM))

#     method = cebra.attribution.init(
#         name="jacobian-based-batched",
#         model=trained_model,
#         input_data=input_tensor,
#         output_dimension=output_dim,
#     )

#     result = method.compute_attribution_map(batch_size=min(128, len(train_data)))
#     print("Attribution keys:", list(result.keys()))

#     # Reduce to 2D maps
#     jc_map = reduce_attr_map(result["jf"])                      # expected shape: (6, 4)
#     jc_inv_map = reduce_attr_map(result["jf-inv-svd"])          # expected shape: (4, 6)
#     jc_invconv_map = reduce_attr_map(result["jf-convabs-inv-svd"])  # expected shape: (4, 6)

#     # AUROC scores
#     auc_jc = compute_auroc(jc_map, gt_attr_bool)
#     auc_jc_inv = compute_auroc(jc_inv_map.T, gt_attr_bool)
#     auc_jc_invconv = compute_auroc(jc_invconv_map.T, gt_attr_bool)

#     print(f"** {model_name} jc AUROC:        {auc_jc:.4f} **")
#     print(f"** {model_name} jc_inv AUROC:    {auc_jc_inv:.4f} **")
#     print(f"** {model_name} jc_invconv AUROC:{auc_jc_invconv:.4f} **")

#     all_results[model_name] = {
#         "jc": jc_map,
#         "jc_inv": jc_inv_map,
#         "jc_invconv": jc_invconv_map,
#         "auc_jc": auc_jc,
#         "auc_jc_inv": auc_jc_inv,
#         "auc_jc_invconv": auc_jc_invconv,
#     }

#     rows.extend([
#         {"model": model_name, "metric": "jc", "auroc": auc_jc},
#         {"model": model_name, "metric": "jc_inv", "auroc": auc_jc_inv},
#         {"model": model_name, "metric": "jc_invconv", "auroc": auc_jc_invconv},
#     ])

#     cleanup_cuda(method, trained_model, input_tensor, model)

# # ============================================================
# # 5) Summary
# # ============================================================
# print("\n" + "=" * 80)
# print(" SUMMARY OF EXPERIMENT RESULTS ".center(80, "="))
# print("=" * 80)
# for model_name, res in all_results.items():
#     print(
#         f" Model: {model_name:<6} | "
#         f"jc={res['auc_jc']:.4f} | "
#         f"jc_inv={res['auc_jc_inv']:.4f} | "
#         f"jc_invconv={res['auc_jc_invconv']:.4f}"
#     )
# print("=" * 80)

# # ============================================================
# # 6) Save CSV
# # ============================================================
# results_df = pd.DataFrame(rows)
# csv_path = os.path.join(OUT_DIR, "synthetic_auroc_results.csv")
# results_df.to_csv(csv_path, index=False)
# print(f"Saved AUROC results to: {csv_path}")

# print("Done.")

#Brownian
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
# 1) Global Config
# ============================================================
T = 100_000
N_MLP_LAYERS = 4
SIGMA_EPS_DEFAULT = 0.03

BATCH_SIZE = 2048
MAX_ITER = 15000
ATTR_BATCH_SIZE = 128

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [121]

# One dataset only
DATASET_CFG = {
    "name": "FIG5_SINGLE",
    "D1": 3,
    "D2": 3,
    "N1": 25,
    "N2": 25,
    "sigma_eps": SIGMA_EPS_DEFAULT,
}


# ============================================================
# 2) Reproducibility
# ============================================================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ============================================================
# 3) Synthetic Data Generation
# ============================================================
class ScaledTanh(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x):
        return self.scale * torch.tanh(x)


def make_mlp(in_dim, out_dim, n_layers=4, seed=0):
    """
    3 hidden layers with GELU, then one output layer + scaled tanh.
    IMPORTANT: move the model to the same device as the inputs.
    """
    torch.manual_seed(seed)

    layers = []
    d_in = in_dim
    hidden = max(64, 8 * max(in_dim, out_dim))

    for _ in range(n_layers - 1):
        lin = nn.Linear(d_in, hidden)
        nn.init.orthogonal_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.GELU()]
        d_in = hidden

    lin = nn.Linear(d_in, out_dim)
    nn.init.orthogonal_(lin.weight)
    nn.init.zeros_(lin.bias)
    layers += [lin, ScaledTanh(scale=1.0)]

    mlp = nn.Sequential(*layers).to(device).eval()
    for p in mlp.parameters():
        p.requires_grad_(False)
    return mlp


def brownian_motion_box(T, d, sigma=0.03, seed=0):
    """
    Brownian motion in [-1, 1]^d.
    Uses rejection sampling so the latent always stays in the box.
    """
    rng = np.random.default_rng(seed)
    z = np.empty((T, d), dtype=np.float32)
    z[0] = rng.uniform(-1.0, 1.0, size=d).astype(np.float32)

    for t in range(1, T):
        prev = z[t - 1].copy()
        nxt = prev + rng.normal(0.0, sigma, size=d).astype(np.float32)

        mask = (nxt < -1.0) | (nxt > 1.0)
        while np.any(mask):
            nxt[mask] = prev[mask] + rng.normal(0.0, sigma, size=mask.sum()).astype(np.float32)
            mask = (nxt < -1.0) | (nxt > 1.0)

        z[t] = nxt

    return z


def make_binary_ground_truth(D1, D2, N1, N2):
    """
    Ground truth map:
      rows = latent variables [z1, z2]
      cols = observed neurons [x1, x2]

    z1 -> x1 and x2
    z2 -> x2 only
    """
    gt = np.zeros((D1 + D2, N1 + N2), dtype=bool)
    gt[:D1, :] = True
    gt[D1:, N1:] = True
    return gt


def generate_synthetic_data(cfg, seed=42):
    D1 = int(cfg["D1"])
    D2 = int(cfg["D2"])
    N1 = int(cfg["N1"])
    N2 = int(cfg["N2"])
    sigma_eps = float(cfg.get("sigma_eps", SIGMA_EPS_DEFAULT))

    z1 = brownian_motion_box(T, D1, sigma=sigma_eps, seed=seed)
    z2 = brownian_motion_box(T, D2, sigma=sigma_eps, seed=seed + 1)

    # mixing functions (new random mixing per seed)
    g1 = make_mlp(D1, N1, n_layers=N_MLP_LAYERS, seed=seed + 10)
    g2 = make_mlp(D1 + D2, N2, n_layers=N_MLP_LAYERS, seed=seed + 20)

    z1_t = torch.tensor(z1, dtype=torch.float32, device=device)
    z2_t = torch.tensor(z2, dtype=torch.float32, device=device)

    with torch.no_grad():
        x1 = g1(z1_t).cpu().numpy()
        x2 = g2(torch.cat([z1_t, z2_t], dim=1)).cpu().numpy()

    x = np.concatenate([x1, x2], axis=1).astype(np.float32)
    latent = np.concatenate([z1, z2], axis=1).astype(np.float32)

    gt_bool = make_binary_ground_truth(D1, D2, N1, N2)
    gt_attr = gt_bool.astype(np.float32)

    return x, latent, gt_attr, gt_bool


# ============================================================
# 4) Utils
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
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    else:
        arr = np.asarray(arr)

    if arr.ndim == 3:
        return np.abs(arr).mean(axis=0)
    if arr.ndim == 2:
        return np.abs(arr)
    if arr.ndim == 1:
        return np.abs(arr)[None, :]
    raise ValueError(f"Unsupported attribution shape: {arr.shape}")


def align_attr_to_gt(attr_map_2d, gt_bool):
    if attr_map_2d.shape == gt_bool.shape:
        return attr_map_2d
    if attr_map_2d.T.shape == gt_bool.shape:
        return attr_map_2d.T
    raise ValueError(
        f"Cannot align attribution map shape {attr_map_2d.shape} "
        f"to ground truth shape {gt_bool.shape}"
    )


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
# 5) Training + Attribution
# ============================================================
def run_one_model(
    cfg,
    seed,
    train_data,
    train_continuous_label,
    gt_attr_bool,
    training_mode,
    adv_epsilon,
):
    D1 = int(cfg["D1"])
    D2 = int(cfg["D2"])
    N1 = int(cfg["N1"])
    N2 = int(cfg["N2"])
    D_LATENT = D1 + D2

    model_name = "ACORN" if training_mode == "adversarial" else "CEBRA"

    set_all_seeds(seed)

    model = CEBRA(
        batch_size=BATCH_SIZE,
        temperature=0.4,
        model_architecture="offset36-model-more-dropout",
        time_offsets=4,
        max_iterations=MAX_ITER,
        output_dimension=D_LATENT,
        verbose=True,
        training_mode=training_mode,
        adv_alpha=adv_epsilon / 5,
        adv_epsilon=adv_epsilon,
        adv_steps=10,
        attack_norm="linf",   # keep your own setting
        num_hidden_units=32,
    )

    model.fit(train_data, train_continuous_label)

    save_path = os.path.join(OUT_DIR, f"{cfg['name']}_seed{seed}_{model_name}.pth")
    try:
        model.save(save_path)
        print("Saved model to:", save_path)
    except Exception as e:
        print("Could not save model:", e)

    trained_model = model.solver_.model.to(device)

    input_tensor = torch.from_numpy(train_data).float().to(device).requires_grad_(True)

    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=trained_model,
        input_data=input_tensor,
        output_dimension=D_LATENT,
    )

    result = method.compute_attribution_map(batch_size=min(ATTR_BATCH_SIZE, len(train_data)))
    print("Attribution keys:", list(result.keys()))

    jac_raw = reduce_attr_map(result["jf"])
    jac_inv_raw = reduce_attr_map(result["jf-inv-svd"])

    jac_map = align_attr_to_gt(jac_raw, gt_attr_bool)
    jac_inv_map = align_attr_to_gt(jac_inv_raw, gt_attr_bool)

    auc_jac = compute_auroc(jac_map, gt_attr_bool)
    auc_jac_inv = compute_auroc(jac_inv_map, gt_attr_bool)

    print(f"** {cfg['name']} | seed={seed} | {model_name} jac AUROC:     {auc_jac:.4f} **")
    print(f"** {cfg['name']} | seed={seed} | {model_name} jac_inv AUROC: {auc_jac_inv:.4f} **")

    cleanup_cuda(method, trained_model, input_tensor, model)

    return {
        "setup": cfg["name"],
        "seed": seed,
        "D1": D1,
        "D2": D2,
        "N1": N1,
        "N2": N2,
        "D_LATENT": D_LATENT,
        "D_OBS": N1 + N2,
        "model": model_name,
        "training_mode": training_mode,
        "jac_auc": auc_jac,
        "jac_inv_auc": auc_jac_inv,
    }


# ============================================================
# 6) Main
# ============================================================
all_rows = []

print("\n" + "#" * 90)
print(f"SETUP: {DATASET_CFG['name']} | D1={DATASET_CFG['D1']} D2={DATASET_CFG['D2']} | N1={DATASET_CFG['N1']} N2={DATASET_CFG['N2']} | sigma={DATASET_CFG['sigma_eps']}")
print("#" * 90)

set_all_seeds(42)
x_np, y_np, gt_attr, gt_attr_bool = generate_synthetic_data(DATASET_CFG, seed=42)

print("x shape:", x_np.shape)
print("y shape:", y_np.shape)
print("gt_attr_bool shape:", gt_attr_bool.shape)

split_idx = int(0.8 * len(x_np))
train_data = x_np[:split_idx].astype(np.float32)
train_continuous_label = y_np[:split_idx].astype(np.float32)

adv_epsilon = float(min_l2_distance(train_data)) / 2.0
adv_epsilon = max(adv_epsilon, 1e-6)
print("adv_epsilon:", adv_epsilon)

for seed in SEEDS:
    for training_mode in ["clean", "adversarial"]:
        cleanup_cuda()

        print("\n" + "=" * 70)
        print(f"Training {DATASET_CFG['name']} | seed={seed} | mode={training_mode}")
        print("=" * 70)

        row = run_one_model(
            cfg=DATASET_CFG,
            seed=seed,
            train_data=train_data,
            train_continuous_label=train_continuous_label,
            gt_attr_bool=gt_attr_bool,
            training_mode=training_mode,
            adv_epsilon=adv_epsilon,
        )
        all_rows.append(row)

results_df = pd.DataFrame(all_rows)

detailed_csv = os.path.join(OUT_DIR, "synthetic_auroc_detailed.csv")
results_df.to_csv(detailed_csv, index=False)
print(f"\nSaved detailed results to: {detailed_csv}")

summary_df = (
    results_df
    .groupby(["setup", "model", "training_mode", "D1", "D2", "N1", "N2", "D_LATENT", "D_OBS"], as_index=False)
    .agg(
        jac_mean=("jac_auc", "mean"),
        jac_std=("jac_auc", "std"),
        jac_inv_mean=("jac_inv_auc", "mean"),
        jac_inv_std=("jac_inv_auc", "std"),
        n_runs=("seed", "count"),
    )
)

summary_csv = os.path.join(OUT_DIR, "synthetic_auroc_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"Saved summary results to: {summary_csv}")

print("\n" + "=" * 120)
print(" SUMMARY ".center(120, "="))
print("=" * 120)
print(summary_df.to_string(index=False))
print("=" * 120)

print("Done.")

# import os
# import gc
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import pandas as pd

# from sklearn.metrics import roc_auc_score
# from utils.min_distance import min_l2_distance
# from utils.constants import CEBRA_DIR

# import sys
# sys.path.insert(0, str(CEBRA_DIR))
# import cebra
# from cebra import CEBRA


# # ============================================================
# # 1) Synthetic Data Config & Generation
# # ============================================================
# T = 100_000
# D1 = 3
# D2 = 3
# D_LATENT = D1 + D2

# N1 = 2
# N2 = 2
# D_OBS = N1 + N2

# N_MLP_LAYERS = 4
# SIGMA_EPS = 0.03

# OUTPUT_DIM = D_LATENT
# BATCH_SIZE = 2048
# MAX_ITER = 2500
# adv_epsilon_default = 0.5

# ATTR_BATCH_SIZE = 128

# OUT_DIR = "outputs"
# os.makedirs(OUT_DIR, exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)
# random.seed(RANDOM_SEED)


# def make_mlp(in_dim, out_dim, n_layers=4, seed=0):
#     torch.manual_seed(seed)
#     layers = []
#     d_in = in_dim
#     hidden = in_dim * 10

#     for i in range(n_layers - 1):
#         d_h = in_dim * 30 if i < n_layers - 2 else hidden
#         lin = nn.Linear(d_in, d_h)
#         nn.init.orthogonal_(lin.weight)
#         nn.init.zeros_(lin.bias)
#         layers += [lin, nn.GELU()]
#         d_in = d_h

#     lin = nn.Linear(d_in, out_dim)
#     nn.init.orthogonal_(lin.weight)
#     nn.init.zeros_(lin.bias)
#     layers.append(lin)

#     mlp = nn.Sequential(*layers)
#     for p in mlp.parameters():
#         p.requires_grad_(False)
#     return mlp.eval()


# def brownian_motion_box(T, d, sigma=0.03, seed=0):
#     rng = np.random.default_rng(seed)
#     x = np.zeros((T, d), dtype=np.float32)
#     x[0] = rng.uniform(-1.0, 1.0, size=d).astype(np.float32)

#     for t in range(T - 1):
#         step = rng.normal(loc=0.0, scale=sigma, size=d).astype(np.float32)
#         x[t + 1] = np.clip(x[t] + step, -1.0, 1.0)
#     return x


# def make_binary_ground_truth(D1, D2, N1, N2):
#     """
#     Ground truth:
#     - first D1 latents connect to all neurons
#     - last D2 latents connect only to x2 block (last N2 neurons)
#     shape = [D_LATENT, D_OBS]
#     """
#     gt = np.zeros((D1 + D2, N1 + N2), dtype=bool)
#     gt[:D1, :] = True
#     gt[D1:, N1:] = True
#     return gt


# def generate_synthetic_data(T=T, seed=42):
#     z1 = brownian_motion_box(T, D1, sigma=SIGMA_EPS, seed=seed)
#     z2 = brownian_motion_box(T, D2, sigma=SIGMA_EPS, seed=seed + 1)

#     g1 = make_mlp(D1, N1, n_layers=N_MLP_LAYERS, seed=seed + 10)
#     g2 = make_mlp(D1 + D2, N2, n_layers=N_MLP_LAYERS, seed=seed + 20)

#     z1_t = torch.tensor(z1, dtype=torch.float32)
#     z2_t = torch.tensor(z2, dtype=torch.float32)

#     with torch.no_grad():
#         x1 = g1(z1_t).cpu().numpy()
#         x2 = g2(torch.cat([z1_t, z2_t], dim=1)).cpu().numpy()

#     x = np.concatenate([x1, x2], axis=1).astype(np.float32)
#     latent = np.concatenate([z1, z2], axis=1).astype(np.float32)

#     gt_bool = make_binary_ground_truth(D1, D2, N1, N2)
#     gt_attr = gt_bool.astype(np.float32)

#     return x, latent, gt_attr, gt_bool


# # ============================================================
# # 2) Utils
# # ============================================================
# def cleanup_cuda(*objs):
#     for obj in objs:
#         try:
#             del obj
#         except Exception:
#             pass
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()


# def reduce_attr_map(arr):
#     """
#     Convert attribution output to 2D map [output_dim, input_dim]
#     if it has sample dimension.
#     """
#     arr = np.asarray(arr)
#     if arr.ndim == 3:
#         return np.abs(arr).mean(axis=0)
#     if arr.ndim == 2:
#         return np.abs(arr)
#     if arr.ndim == 1:
#         return np.abs(arr)[None, :]
#     raise ValueError(f"Unsupported attribution shape: {arr.shape}")


# def compute_auroc(attr_map_2d, gt_bool):
#     y_true = gt_bool.ravel().astype(int)
#     y_score = np.asarray(attr_map_2d, dtype=np.float64).ravel()

#     if y_true.shape[0] != y_score.shape[0]:
#         raise ValueError(
#             f"Shape mismatch: y_true has {y_true.shape[0]} elements, "
#             f"y_score has {y_score.shape[0]} elements, "
#             f"attr_map shape={attr_map_2d.shape}, gt shape={gt_bool.shape}"
#         )

#     if len(np.unique(y_true)) < 2:
#         return float("nan")

#     return float(roc_auc_score(y_true, y_score))


# # ============================================================
# # 3) Data Generation
# # ============================================================
# print("Generating synthetic dataset...")
# x_np, y_np, gt_attr, gt_attr_bool = generate_synthetic_data(T=T, seed=42)

# print("x shape:", x_np.shape)  # (T, 6)
# print("y shape:", y_np.shape)  # (T, 4)
# print("gt_attr_bool shape:", gt_attr_bool.shape)  # (4, 6)

# split_idx = int(0.8 * len(x_np))
# train_data = x_np[:split_idx].astype(np.float32)
# train_continuous_label = y_np[:split_idx].astype(np.float32)

# # For later use in CEBRA setup
# adv_epsilon = float(min_l2_distance(train_data)) / 2.0
# adv_epsilon = max(adv_epsilon, 1e-6)

# # ============================================================
# # 4) Train + Attribution
# # ============================================================
# rows = []
# all_results = {}

# for adv in [False, True]:
#     cleanup_cuda()

#     model_name = "ACORN" if adv else "CEBRA"
#     training_mode = "adversarial" if adv else "clean"

#     print("\n" + "=" * 70)
#     print(f"Training {model_name} ({training_mode})")
#     print("=" * 70)

#     model = CEBRA(
#         batch_size=BATCH_SIZE,
#         temperature=0.4,
#         model_architecture="offset36-model-more-dropout",
#         time_offsets=4,
#         max_iterations=MAX_ITER,
#         output_dimension=OUTPUT_DIM,
#         verbose=True,
#         training_mode=training_mode,
#         adv_alpha=adv_epsilon / 5,
#         adv_epsilon=adv_epsilon,
#         adv_steps=10,
#         attack_norm="linf",
#         num_hidden_units=32,
#     )

#     model.fit(train_data, train_continuous_label)

#     save_path = os.path.join(OUT_DIR, f"{model_name}_synthetic.pth")
#     model.save(save_path)
#     print("Saved model to:", save_path)

#     trained_model = model.solver_.model.to(device)

#     # Use full training data for attribution; batch_size smaller to avoid errors
#     input_tensor = torch.from_numpy(train_data).float().to(device).requires_grad_(True)

#     output_dim = int(getattr(trained_model, "num_output", OUTPUT_DIM))

#     method = cebra.attribution.init(
#         name="jacobian-based-batched",
#         model=trained_model,
#         input_data=input_tensor,
#         output_dimension=output_dim,
#     )

#     result = method.compute_attribution_map(batch_size=min(128, len(train_data)))
#     print("Attribution keys:", list(result.keys()))

#     # Reduce to 2D maps
#     jc_map = reduce_attr_map(result["jf"])
#     jc_inv_map = reduce_attr_map(result["jf-inv-svd"])
#     jc_invconv_map = reduce_attr_map(result["jf-convabs-inv-svd"])

#     # AUROC scores
#     auc_jc = compute_auroc(jc_map, gt_attr_bool)
#     auc_jc_inv = compute_auroc(jc_inv_map, gt_attr_bool)
#     auc_jc_invconv = compute_auroc(jc_invconv_map, gt_attr_bool)

#     print(f"** {model_name} jc AUROC:        {auc_jc:.4f} **")
#     print(f"** {model_name} jc_inv AUROC:    {auc_jc_inv:.4f} **")
#     print(f"** {model_name} jc_invconv AUROC:{auc_jc_invconv:.4f} **")

#     all_results[model_name] = {
#         "jc": jc_map,
#         "jc_inv": jc_inv_map,
#         "jc_invconv": jc_invconv_map,
#         "auc_jc": auc_jc,
#         "auc_jc_inv": auc_jc_inv,
#         "auc_jc_invconv": auc_jc_invconv,
#     }

#     rows.extend([
#         {"model": model_name, "metric": "jc", "auroc": auc_jc},
#         {"model": model_name, "metric": "jc_inv", "auroc": auc_jc_inv},
#         {"model": model_name, "metric": "jc_invconv", "auroc": auc_jc_invconv},
#     ])

#     cleanup_cuda(method, trained_model, input_tensor, model)

# # ============================================================
# # 5) Summary
# # ============================================================
# print("\n" + "=" * 80)
# print(" SUMMARY OF EXPERIMENT RESULTS ".center(80, "="))
# print("=" * 80)
# for model_name, res in all_results.items():
#     print(
#         f" Model: {model_name:<6} | "
#         f"jc={res['auc_jc']:.4f} | "
#         f"jc_inv={res['auc_jc_inv']:.4f} | "
#         f"jc_invconv={res['auc_jc_invconv']:.4f}"
#     )
# print("=" * 80)

# # ============================================================
# # 6) Save CSV
# # ============================================================
# results_df = pd.DataFrame(rows)
# csv_path = os.path.join(OUT_DIR, "synthetic_auroc_results.csv")
# results_df.to_csv(csv_path, index=False)
# print(f"Saved AUROC results to: {csv_path}")

# print("Done.")

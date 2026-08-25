import os
import gc
import sys
import json
import random
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LinearRegression

from utils.constants import CEBRA_DIR


# # ============================================================
# # 0) Import ONLY the custom CEBRA fork
# # ============================================================
# for name in list(sys.modules):
#     if name == "cebra" or name.startswith("cebra."):
#         del sys.modules[name]

sys.path.insert(0, str(CEBRA_DIR))
# importlib.invalidate_caches()

import cebra
from cebra import CEBRA

print("Using CEBRA from:", cebra.__file__)
print("CEBRA version:", getattr(cebra, "__version__", "unknown"))


# ============================================================
# 1) Experiment mode
# ============================================================
# "stability":
#   - one fixed data-generating process
#   - many CEBRA training seeds
#   - isolates optimization/training variance
#
# "paper_final":
#   - 10 different synthetic datasets / mixing functions
#   - one fixed training seed per dataset
#   - mirrors the paper's idea of averaging over 10 different datasets
#
EXPERIMENT_MODE = "stability"

STABILITY_DATA_SEED = 2025
STABILITY_TRAIN_SEEDS = [38, 226, 1, 36, 989, 26, 84, 66, 27, 81, 49]

FINAL_DATA_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
FINAL_TRAIN_SEEDS = [38]

if EXPERIMENT_MODE == "stability":
    DATA_SEEDS = [STABILITY_DATA_SEED]
    TRAIN_SEEDS = STABILITY_TRAIN_SEEDS
elif EXPERIMENT_MODE == "paper_final":
    DATA_SEEDS = FINAL_DATA_SEEDS
    TRAIN_SEEDS = FINAL_TRAIN_SEEDS
else:
    raise ValueError("EXPERIMENT_MODE must be 'stability' or 'paper_final'.")


# ============================================================
# 2) Global config
# ============================================================
T = 100_000

D1 = 3
D2 = 3

N1 = 25
N2 = 25

D_LATENT = D1 + D2
D_OBS = N1 + N2
OUTPUT_DIM = D_LATENT

# Brownian motion
BROWNIAN_SIGMA = 0.03

# Injective mixing
# g(z) = tanh(gain * W z), with W full-column-rank.
# This is injective because W is injective and tanh is strictly monotone.
MIX_GAIN = 1.5
MIN_ABS_MIX_WEIGHT = 0.004

# Train/test
TRAIN_FRAC = 0.80

# Paper reports batch size 5000 and 20k training steps.
BATCH_SIZE = 5000
MAX_ITER = 20_000

# Keep your known-working architecture.
# If your fork also has a no-dropout equivalent and you later want a
# strict architecture ablation, test it separately; do not mix it into
# the main benchmark without reporting it.
MODEL_ARCHITECTURE = "offset36-model-more-dropout"
TIME_OFFSETS = 4
TEMPERATURE = 0.4
NUM_HIDDEN_UNITS = 64

# PGD: use the scale already defined as sensible defaults in your base.py.
# Inputs are standardized before training, so epsilon has a consistent meaning.
ADV_EPSILON = 0.05
ADV_ALPHA = 0.01
ADV_STEPS = 10
ATTACK_NORM = "linf"

# Attribution: use a CONTIGUOUS held-out segment.
ATTR_POINTS = 10_000
ATTR_MARGIN = 128
ATTR_BATCH_SIZE = 128

# Linear latent recovery sanity-check
R2_TRAIN_POINTS = 20_000
R2_TEST_POINTS = 10_000

# Output
ROOT_DIR = Path("Fake_dataset_stable")
OUT_DIR = ROOT_DIR / "outputs"
IMG_DIR = ROOT_DIR / "images"
DATA_DIR = ROOT_DIR / "datasets"

for p in [ROOT_DIR, OUT_DIR, IMG_DIR, DATA_DIR]:
    p.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_CFG = {
    "name": "FIG5_INJECTIVE",
    "T": T,
    "D1": D1,
    "D2": D2,
    "N1": N1,
    "N2": N2,
    "sigma": BROWNIAN_SIGMA,
    "mix_gain": MIX_GAIN,
}


# ============================================================
# 3) Reproducibility
# ============================================================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Good for a reproducibility-focused synthetic benchmark.
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ============================================================
# 4) Brownian latent process
# ============================================================
def brownian_motion_box(
    n_samples: int,
    dim: int,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """
    Brownian motion constrained to [-1, 1]^dim.

    z_0 ~ Uniform([-1,1]^dim)
    z_t ~ truncated Normal(z_{t-1}, sigma^2 I)

    Rejection sampling is used so accepted steps are inside the box.
    """
    rng = np.random.default_rng(seed)

    z = np.empty((n_samples, dim), dtype=np.float32)
    z[0] = rng.uniform(-1.0, 1.0, size=dim).astype(np.float32)

    for t in range(1, n_samples):
        prev = z[t - 1]

        nxt = prev + rng.normal(
            loc=0.0,
            scale=sigma,
            size=dim,
        ).astype(np.float32)

        bad = (nxt < -1.0) | (nxt > 1.0)

        while np.any(bad):
            nxt[bad] = (
                prev[bad]
                + rng.normal(
                    loc=0.0,
                    scale=sigma,
                    size=int(bad.sum()),
                ).astype(np.float32)
            )
            bad = (nxt < -1.0) | (nxt > 1.0)

        z[t] = nxt

    return z


# ============================================================
# 5) Guaranteed injective mixing functions
# ============================================================
def make_dense_orthogonal_matrix(
    out_dim: int,
    in_dim: int,
    seed: int,
    min_abs_weight: float = 0.004,
    max_tries: int = 20_000,
) -> torch.Tensor:
    """
    Create W in R^{out_dim x in_dim} with:
      - orthonormal columns -> rank(W) = in_dim and condition number = 1
      - dense entries, avoiding near-zero direct connections

    Requires out_dim >= in_dim.
    """
    if out_dim < in_dim:
        raise ValueError(
            f"Injective linear map requires out_dim >= in_dim, got {out_dim} < {in_dim}"
        )

    for attempt in range(max_tries):
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed + attempt))

        A = torch.randn(out_dim, in_dim, generator=g, dtype=torch.float32)

        # Reduced QR: Q has orthonormal columns.
        Q, _ = torch.linalg.qr(A, mode="reduced")

        if float(Q.abs().min()) >= float(min_abs_weight):
            return Q.contiguous()

    raise RuntimeError(
        f"Could not generate a dense orthogonal matrix after {max_tries} attempts. "
        f"Try decreasing min_abs_weight={min_abs_weight}."
    )


class InjectiveMixer(nn.Module):
    """
    Smooth, nonlinear, guaranteed-injective map:
        x = tanh(gain * W z)

    W has full column rank and tanh is strictly monotonic.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        seed: int,
        gain: float = 1.5,
    ):
        super().__init__()

        W = make_dense_orthogonal_matrix(
            out_dim=out_dim,
            in_dim=in_dim,
            seed=seed,
            min_abs_weight=MIN_ABS_MIX_WEIGHT,
        )

        self.register_buffer("W", W)
        self.gain = float(gain)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pre = self.gain * (z @ self.W.T)
        return torch.tanh(pre)


# ============================================================
# 6) Ground-truth structure
# ============================================================
def make_binary_ground_truth() -> np.ndarray:
    """
    Rows: latent dimensions [z1(3), z2(3)]
    Cols: observed variables [x1(25), x2(25)]

    z1 -> x1 and x2
    z2 -> x2 only
    """
    gt = np.zeros((D_LATENT, D_OBS), dtype=bool)

    # z1 affects x1 and x2
    gt[:D1, :] = True

    # z2 affects x2 only
    gt[D1:, N1:] = True

    return gt


# ============================================================
# 7) Generate ONE synthetic dataset
# ============================================================
def generate_synthetic_data(data_seed: int):
    """
    Figure-5-style data:
      z1: 3 latent dims
      z2: 3 latent dims
      g1: z1       -> 25 observed variables
      g2: [z1,z2]  -> 25 observed variables
      x = [x1,x2]  -> 50 observed variables
    """
    # All six latent coordinates follow the same Brownian rule.
    latent = brownian_motion_box(
        n_samples=T,
        dim=D_LATENT,
        sigma=BROWNIAN_SIGMA,
        seed=data_seed,
    )

    z1 = latent[:, :D1]
    z2 = latent[:, D1:]

    g1 = InjectiveMixer(
        in_dim=D1,
        out_dim=N1,
        seed=data_seed + 10_000,
        gain=MIX_GAIN,
    ).to(device).eval()

    g2 = InjectiveMixer(
        in_dim=D_LATENT,
        out_dim=N2,
        seed=data_seed + 20_000,
        gain=MIX_GAIN,
    ).to(device).eval()

    z_t = torch.from_numpy(latent).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        x1 = g1(z_t[:, :D1])
        x2 = g2(z_t)
        x_raw = torch.cat([x1, x2], dim=1).cpu().numpy().astype(np.float32)

    gt_bool = make_binary_ground_truth()

    return {
        "x_raw": x_raw,
        "latent": latent.astype(np.float32),
        "z1": z1.astype(np.float32),
        "z2": z2.astype(np.float32),
        "gt_bool": gt_bool,
        "g1": g1,
        "g2": g2,
    }


# ============================================================
# 8) Train-only standardization
# ============================================================
def split_and_standardize(x_raw: np.ndarray, latent: np.ndarray):
    split_idx = int(TRAIN_FRAC * len(x_raw))

    train_x_raw = x_raw[:split_idx]
    test_x_raw = x_raw[split_idx:]

    train_latent = latent[:split_idx]
    test_latent = latent[split_idx:]

    # IMPORTANT: statistics are estimated from training data only.
    mu = train_x_raw.mean(axis=0, keepdims=True).astype(np.float32)
    sd = train_x_raw.std(axis=0, keepdims=True).astype(np.float32)

    # Avoid division by numerical zero.
    sd = np.maximum(sd, 1e-6).astype(np.float32)

    train_x = ((train_x_raw - mu) / sd).astype(np.float32)
    test_x = ((test_x_raw - mu) / sd).astype(np.float32)

    return {
        "split_idx": split_idx,
        "train_x": train_x,
        "test_x": test_x,
        "train_latent": train_latent.astype(np.float32),
        "test_latent": test_latent.astype(np.float32),
        "mu": mu,
        "sd": sd,
    }


# ============================================================
# 9) Oracle generator-Jacobian check
# ============================================================
def compute_oracle_generator_map(
    latent_np: np.ndarray,
    g1: nn.Module,
    g2: nn.Module,
    feature_sd: np.ndarray,
    n_points: int = 128,
) -> np.ndarray:
    """
    Compute |J_g| averaged over points.

    Because training inputs are standardized feature-wise, each observed
    derivative is divided by the corresponding feature standard deviation.
    """
    idx = np.linspace(
        0,
        len(latent_np) - 1,
        min(n_points, len(latent_np)),
    ).astype(int)

    z = torch.tensor(
        latent_np[idx],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    x1 = g1(z[:, :D1])
    x2 = g2(z)
    x = torch.cat([x1, x2], dim=1)

    # Feature standardization: x_std_j = (x_j - mu_j) / sd_j
    sd_t = torch.tensor(
        feature_sd.reshape(-1),
        dtype=torch.float32,
        device=device,
    )
    x_std = x / sd_t[None, :]

    oracle = torch.zeros(
        D_LATENT,
        D_OBS,
        dtype=torch.float32,
        device=device,
    )

    # Each output feature is sample-wise independent, so grad(sum_j x[:,j], z)
    # returns the per-sample derivative for that feature.
    for feature_idx in range(D_OBS):
        grad = torch.autograd.grad(
            x_std[:, feature_idx].sum(),
            z,
            retain_graph=True,
            create_graph=False,
        )[0]

        oracle[:, feature_idx] = grad.abs().mean(dim=0)

    return oracle.detach().cpu().numpy().astype(np.float32)


# ============================================================
# 10) Metrics / attribution utils
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
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def reduce_attr_map(arr):
    """
    Return [latent_dim, observed_dim] after taking absolute values and
    averaging across samples if a sample dimension exists.
    """
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    else:
        arr = np.asarray(arr)

    arr = np.abs(arr)

    if arr.ndim == 3:
        return arr.mean(axis=0).astype(np.float32)

    if arr.ndim == 2:
        return arr.astype(np.float32)

    if arr.ndim == 1:
        return arr[None, :].astype(np.float32)

    raise ValueError(f"Unsupported attribution shape: {arr.shape}")


def align_attr_to_gt(attr_map: np.ndarray, gt_bool: np.ndarray) -> np.ndarray:
    if attr_map.shape == gt_bool.shape:
        return attr_map

    if attr_map.T.shape == gt_bool.shape:
        return attr_map.T

    raise ValueError(
        f"Cannot align attribution shape {attr_map.shape} "
        f"to GT shape {gt_bool.shape}"
    )


def compute_binary_scores(attr_map: np.ndarray, gt_bool: np.ndarray):
    y_true = gt_bool.ravel().astype(np.int64)
    y_score = np.asarray(attr_map, dtype=np.float64).ravel()

    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))

    return auroc, auprc


def save_heatmap(mat, path, title):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(mat, aspect="auto", cmap="cividis")
    ax.set_title(title)
    ax.set_xlabel("Observed feature")
    ax.set_ylabel("Latent dimension")
    fig.colorbar(im, ax=ax, shrink=0.90)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def get_contiguous_attr_segment(test_x: np.ndarray) -> np.ndarray:
    start = min(ATTR_MARGIN, max(0, len(test_x) - ATTR_POINTS))

    end = min(
        start + ATTR_POINTS,
        len(test_x),
    )

    x = test_x[start:end]

    if len(x) < 100:
        raise ValueError(
            f"Attribution segment is too short: {len(x)} samples."
        )

    return x.astype(np.float32)


# ============================================================
# 11) CEBRA model
# ============================================================
def build_model(adversarial: bool):
    """
    We DO NOT modify base.py.

    We only pass PGD hyperparameters to your existing implementation.
    """
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCHITECTURE,
        time_offsets=TIME_OFFSETS,
        max_iterations=MAX_ITER,
        output_dimension=OUTPUT_DIM,
        verbose=True,

        # Keep this benchmark on the ordinary Solver path.
        # In your custom base.py, hybrid adversarial code has separate behavior.
        hybrid=False,

        training_mode="adversarial" if adversarial else "clean",

        # IMPORTANT: no hard-coded epsilon=5.
        adv_epsilon=ADV_EPSILON if adversarial else 0.0,
        adv_alpha=ADV_ALPHA if adversarial else 0.0,
        adv_steps=ADV_STEPS if adversarial else 0,
        attack_norm=ATTACK_NORM,

        num_hidden_units=NUM_HIDDEN_UNITS,
        device="cuda_if_available",
    )


# ============================================================
# 12) Latent recovery sanity-check
# ============================================================
def latent_linear_r2(
    model,
    train_x,
    train_latent,
    test_x,
    test_latent,
):
    """
    Check whether the learned representation contains the six ground-truth
    latents linearly. This helps separate representation failure from
    attribution failure.
    """
    try:
        # Use contiguous temporal windows. Do NOT subsample with linspace
        # before passing data through an offset/temporal encoder.
        n_train = min(R2_TRAIN_POINTS, len(train_x))
        n_test = min(R2_TEST_POINTS, len(test_x))

        train_x_r2 = train_x[:n_train]
        train_z_r2 = train_latent[:n_train]

        test_x_r2 = test_x[:n_test]
        test_z_r2 = test_latent[:n_test]

        emb_train = model.transform(train_x_r2)
        emb_test = model.transform(test_x_r2)

        emb_train = np.asarray(emb_train)
        emb_test = np.asarray(emb_test)

        reg = LinearRegression()
        reg.fit(emb_train, train_z_r2)

        return float(
            reg.score(
                emb_test,
                test_z_r2,
            )
        )

    except Exception as exc:
        print("WARNING: Could not compute latent R2:", repr(exc))
        return float("nan")


# ============================================================
# 13) Train + attribute one run
# ============================================================
def train_and_score_one_run(
    data_seed: int,
    train_seed: int,
    train_x: np.ndarray,
    train_latent: np.ndarray,
    test_x: np.ndarray,
    test_latent: np.ndarray,
    gt_bool: np.ndarray,
    adversarial: bool,
):
    # Critical: this seed changes ONLY model/training randomness.
    set_all_seeds(train_seed)

    model_name = "ACORN" if adversarial else "CEBRA"

    print("\n" + "=" * 90)
    print(
        f"data_seed={data_seed} | train_seed={train_seed} | "
        f"model={model_name}"
    )
    print("=" * 90)

    model = build_model(adversarial=adversarial)

    # Two continuous auxiliary blocks, preserving your original setup.
    z1_train = train_latent[:, :D1]
    z2_train = train_latent[:, D1:]

    model.fit(
        train_x.astype(np.float32),
        z1_train.astype(np.float32),
        z2_train.astype(np.float32),
    )

    run_tag = (
        f"{DATASET_CFG['name']}_"
        f"dseed{data_seed}_"
        f"tseed{train_seed}_"
        f"{model_name}"
    )

    save_path = OUT_DIR / f"{run_tag}.pth"

    try:
        model.save(str(save_path))
        print("Saved model:", save_path)
    except Exception as exc:
        print("WARNING: Could not save model:", repr(exc))

    # --------------------------------------------------------
    # Representation sanity check
    # --------------------------------------------------------
    r2 = latent_linear_r2(
        model=model,
        train_x=train_x,
        train_latent=train_latent,
        test_x=test_x,
        test_latent=test_latent,
    )
    print(f"Latent linear R2: {r2:.4f}")

    # --------------------------------------------------------
    # Attribution on CONTIGUOUS held-out data
    # --------------------------------------------------------
    trained_model = model.solver_.model.to(device)

    if hasattr(trained_model, "split_outputs"):
        trained_model.split_outputs = False

    trained_model.eval()

    attr_x = get_contiguous_attr_segment(test_x)

    input_tensor = torch.from_numpy(attr_x).to(
        device=device,
        dtype=torch.float32,
    )
    input_tensor.requires_grad_(True)

    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=trained_model,
        input_data=input_tensor,
        output_dimension=int(
            getattr(trained_model, "num_output", OUTPUT_DIM)
        ),
    )

    attr_batch_size = min(ATTR_BATCH_SIZE, len(attr_x))
    result = method.compute_attribution_map(
        batch_size=attr_batch_size
    )

    print("Attribution keys:", list(result.keys()))

    if "jf" not in result:
        raise KeyError(
            f"'jf' not found. Available keys: {list(result.keys())}"
        )

    if "jf-inv-svd" in result:
        inv_key = "jf-inv-svd"
    elif "jf-inv" in result:
        inv_key = "jf-inv"
    else:
        raise KeyError(
            f"No inverse-Jacobian attribution found. "
            f"Available keys: {list(result.keys())}"
        )

    jf_raw = reduce_attr_map(result["jf"])
    jfinv_raw = reduce_attr_map(result[inv_key])

    jf_map = align_attr_to_gt(jf_raw, gt_bool)
    jfinv_map = align_attr_to_gt(jfinv_raw, gt_bool)

    auroc_jf, auprc_jf = compute_binary_scores(
        jf_map,
        gt_bool,
    )

    auroc_jfinv, auprc_jfinv = compute_binary_scores(
        jfinv_map,
        gt_bool,
    )

    print(
        f"JF     | AUROC={auroc_jf:.4f} | AUPRC={auprc_jf:.4f}"
    )
    print(
        f"JF-INV | AUROC={auroc_jfinv:.4f} | AUPRC={auprc_jfinv:.4f}"
    )

    np.savez_compressed(
        OUT_DIR / f"{run_tag}_attrs.npz",
        jf=jf_map.astype(np.float32),
        jfinv=jfinv_map.astype(np.float32),
        gt=gt_bool.astype(np.uint8),
        latent_r2=np.array([r2], dtype=np.float32),
        auroc_jf=np.array([auroc_jf], dtype=np.float32),
        auprc_jf=np.array([auprc_jf], dtype=np.float32),
        auroc_jfinv=np.array([auroc_jfinv], dtype=np.float32),
        auprc_jfinv=np.array([auprc_jfinv], dtype=np.float32),
    )

    save_heatmap(
        jf_map,
        IMG_DIR / f"{run_tag}_JF.png",
        f"{run_tag} | JF",
    )

    save_heatmap(
        jfinv_map,
        IMG_DIR / f"{run_tag}_JF_INV.png",
        f"{run_tag} | JF-INV",
    )

    cleanup_cuda(
        method,
        trained_model,
        input_tensor,
        model,
    )

    return {
        "setup": DATASET_CFG["name"],
        "experiment_mode": EXPERIMENT_MODE,
        "data_seed": data_seed,
        "train_seed": train_seed,
        "model": model_name,
        "training_mode": (
            "adversarial"
            if adversarial
            else "clean"
        ),
        "adv_epsilon": (
            ADV_EPSILON
            if adversarial
            else 0.0
        ),
        "adv_alpha": (
            ADV_ALPHA
            if adversarial
            else 0.0
        ),
        "adv_steps": (
            ADV_STEPS
            if adversarial
            else 0
        ),
        "latent_r2": r2,
        "auroc_jf": auroc_jf,
        "auprc_jf": auprc_jf,
        "auroc_jfinv": auroc_jfinv,
        "auprc_jfinv": auprc_jfinv,
    }


# ============================================================
# 14) Paired summary
# ============================================================
def make_paired_summary(results_df: pd.DataFrame):
    metrics = [
        "latent_r2",
        "auroc_jf",
        "auprc_jf",
        "auroc_jfinv",
        "auprc_jfinv",
    ]

    rows = []

    for metric in metrics:
        piv = results_df.pivot_table(
            index=["data_seed", "train_seed"],
            columns="model",
            values=metric,
            aggfunc="first",
        )

        if "CEBRA" not in piv.columns or "ACORN" not in piv.columns:
            continue

        piv = piv.dropna(
            subset=["CEBRA", "ACORN"]
        )

        if len(piv) == 0:
            continue

        diff = (
            piv["ACORN"].to_numpy()
            - piv["CEBRA"].to_numpy()
        )

        rows.append({
            "metric": metric,
            "n_pairs": len(diff),
            "CEBRA_mean": float(piv["CEBRA"].mean()),
            "CEBRA_std": float(piv["CEBRA"].std(ddof=1)) if len(piv) > 1 else 0.0,
            "ACORN_mean": float(piv["ACORN"].mean()),
            "ACORN_std": float(piv["ACORN"].std(ddof=1)) if len(piv) > 1 else 0.0,
            "ACORN_minus_CEBRA_mean": float(diff.mean()),
            "ACORN_minus_CEBRA_std": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
            "ACORN_better_fraction": float((diff > 0).mean()),
        })

    return pd.DataFrame(rows)


# ============================================================
# 15) Main
# ============================================================
def main():
    print("\n" + "#" * 100)
    print("STABLE SYNTHETIC CEBRA / PGD BENCHMARK")
    print("#" * 100)
    print("Experiment mode:", EXPERIMENT_MODE)
    print("Data seeds:", DATA_SEEDS)
    print("Training seeds:", TRAIN_SEEDS)
    print("Device:", device)
    print(
        f"PGD: eps={ADV_EPSILON}, alpha={ADV_ALPHA}, "
        f"steps={ADV_STEPS}, norm={ATTACK_NORM}"
    )

    config_to_save = {
        "experiment_mode": EXPERIMENT_MODE,
        "data_seeds": DATA_SEEDS,
        "train_seeds": TRAIN_SEEDS,
        "T": T,
        "D1": D1,
        "D2": D2,
        "N1": N1,
        "N2": N2,
        "brownian_sigma": BROWNIAN_SIGMA,
        "mix_gain": MIX_GAIN,
        "batch_size": BATCH_SIZE,
        "max_iterations": MAX_ITER,
        "model_architecture": MODEL_ARCHITECTURE,
        "temperature": TEMPERATURE,
        "num_hidden_units": NUM_HIDDEN_UNITS,
        "adv_epsilon": ADV_EPSILON,
        "adv_alpha": ADV_ALPHA,
        "adv_steps": ADV_STEPS,
        "attack_norm": ATTACK_NORM,
        "attr_points": ATTR_POINTS,
    }

    with open(
        ROOT_DIR / "experiment_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config_to_save,
            f,
            indent=2,
        )

    all_rows = []

    for data_seed in DATA_SEEDS:
        print("\n" + "#" * 100)
        print(f"GENERATING DATASET | data_seed={data_seed}")
        print("#" * 100)

        # ----------------------------------------------------
        # Generate data ONCE for this data seed.
        # ----------------------------------------------------
        set_all_seeds(data_seed)

        data = generate_synthetic_data(
            data_seed=data_seed
        )

        split = split_and_standardize(
            x_raw=data["x_raw"],
            latent=data["latent"],
        )

        gt_bool = data["gt_bool"]

        # ----------------------------------------------------
        # Verify the ground-truth generator Jacobian.
        # ----------------------------------------------------
        oracle_map = compute_oracle_generator_map(
            latent_np=data["latent"],
            g1=data["g1"],
            g2=data["g2"],
            feature_sd=split["sd"],
            n_points=128,
        )

        oracle_auroc, oracle_auprc = compute_binary_scores(
            oracle_map,
            gt_bool,
        )

        print(
            f"Oracle generator map | "
            f"AUROC={oracle_auroc:.6f} | "
            f"AUPRC={oracle_auprc:.6f}"
        )

        if oracle_auroc < 0.999:
            print(
                "WARNING: Oracle AUROC is below 0.999. "
                "Inspect the mixing function before interpreting CEBRA results."
            )

        save_heatmap(
            gt_bool.astype(np.float32),
            IMG_DIR / f"dseed{data_seed}_GT.png",
            f"Ground truth | data_seed={data_seed}",
        )

        save_heatmap(
            oracle_map,
            IMG_DIR / f"dseed{data_seed}_ORACLE_JG.png",
            f"Oracle |J_g| | data_seed={data_seed}",
        )

        np.savez_compressed(
            DATA_DIR / f"synthetic_dseed{data_seed}.npz",
            x_raw=data["x_raw"],
            latent=data["latent"],
            train_x=split["train_x"],
            test_x=split["test_x"],
            train_latent=split["train_latent"],
            test_latent=split["test_latent"],
            mean=split["mu"],
            std=split["sd"],
            gt=gt_bool.astype(np.uint8),
            oracle_map=oracle_map.astype(np.float32),
            oracle_auroc=np.array(
                [oracle_auroc],
                dtype=np.float32,
            ),
            oracle_auprc=np.array(
                [oracle_auprc],
                dtype=np.float32,
            ),
        )

        # ----------------------------------------------------
        # Train clean / adversarial on EXACTLY THE SAME DATA.
        # ----------------------------------------------------
        for train_seed in TRAIN_SEEDS:
            for adversarial in [False, True]:
                cleanup_cuda()

                row = train_and_score_one_run(
                    data_seed=data_seed,
                    train_seed=train_seed,
                    train_x=split["train_x"],
                    train_latent=split["train_latent"],
                    test_x=split["test_x"],
                    test_latent=split["test_latent"],
                    gt_bool=gt_bool,
                    adversarial=adversarial,
                )

                row["oracle_auroc"] = oracle_auroc
                row["oracle_auprc"] = oracle_auprc

                all_rows.append(row)

                # Save incrementally so a long run can be resumed/analyzed
                # even if a later seed fails.
                pd.DataFrame(all_rows).to_csv(
                    OUT_DIR / "results_detailed_partial.csv",
                    index=False,
                )

        cleanup_cuda(
            data["g1"],
            data["g2"],
        )

    # ========================================================
    # Final result tables
    # ========================================================
    results_df = pd.DataFrame(all_rows)

    detailed_path = OUT_DIR / "results_detailed.csv"
    results_df.to_csv(
        detailed_path,
        index=False,
    )

    summary_df = (
        results_df
        .groupby(
            ["model", "training_mode"],
            as_index=False,
        )
        .agg(
            n_runs=("train_seed", "count"),

            latent_r2_mean=("latent_r2", "mean"),
            latent_r2_std=("latent_r2", "std"),

            auroc_jf_mean=("auroc_jf", "mean"),
            auroc_jf_std=("auroc_jf", "std"),

            auprc_jf_mean=("auprc_jf", "mean"),
            auprc_jf_std=("auprc_jf", "std"),

            auroc_jfinv_mean=("auroc_jfinv", "mean"),
            auroc_jfinv_std=("auroc_jfinv", "std"),

            auprc_jfinv_mean=("auprc_jfinv", "mean"),
            auprc_jfinv_std=("auprc_jfinv", "std"),
        )
    )

    summary_path = OUT_DIR / "results_summary.csv"
    summary_df.to_csv(
        summary_path,
        index=False,
    )

    paired_df = make_paired_summary(
        results_df
    )

    paired_path = OUT_DIR / "results_paired_comparison.csv"
    paired_df.to_csv(
        paired_path,
        index=False,
    )

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 120)
    print("PAIRED ACORN - CEBRA")
    print("=" * 120)
    print(paired_df.to_string(index=False))

    print("\nSaved:")
    print("  ", detailed_path)
    print("  ", summary_path)
    print("  ", paired_path)

    print("\nDONE.")

if __name__ == "__main__":
    main()

###########################
###########################
###########################
###########################
###########################
###########################
###########################
###########################
######################################################
#################################################################################
############################################################################################################
#######################################################################################################################################
##################################################################################################################################################################

# # Lorenzo
# # import os
# # import gc
# # import random
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import pandas as pd

# # from sklearn.metrics import roc_auc_score
# # from utils.min_distance import min_l2_distance
# # from utils.constants import CEBRA_DIR

# # import sys
# # sys.path.insert(0, str(CEBRA_DIR))
# # import cebra
# # from cebra import CEBRA


# # # ============================================================
# # # 1) Synthetic Data Config & Generation
# # # ============================================================
# # T = 100_000
# # D1 = 3  # Lorenz system latents
# # D2 = 3  # Lorenz system latents
# # D_LATENT = D1 + D2  # 6

# # N1 = 25
# # N2 = 25
# # D_OBS = N1 + N2     # 4

# # N_MLP_LAYERS = 4
# # SIGMA_EPS = 0.03

# # # Output dimension must match latent dimension
# # OUTPUT_DIM = D_LATENT  # 6
# # BATCH_SIZE = 2048
# # MAX_ITER = 2500
# # adv_epsilon_default = 0.5

# # ATTR_BATCH_SIZE = 128

# # OUT_DIR = "outputs"
# # os.makedirs(OUT_DIR, exist_ok=True)

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # RANDOM_SEED = 88
# # np.random.seed(RANDOM_SEED)
# # torch.manual_seed(RANDOM_SEED)
# # random.seed(RANDOM_SEED)


# # def make_mlp(in_dim, out_dim, n_layers=4, seed=0):
# #     torch.manual_seed(seed)
# #     layers = []
# #     d_in = in_dim
# #     hidden = in_dim * 10

# #     for i in range(n_layers - 1):
# #         d_h = in_dim * 30 if i < n_layers - 2 else hidden
# #         lin = nn.Linear(d_in, d_h)
# #         nn.init.orthogonal_(lin.weight)
# #         nn.init.zeros_(lin.bias)
# #         layers += [lin, nn.GELU()]
# #         d_in = d_h

# #     lin = nn.Linear(d_in, out_dim)
# #     nn.init.orthogonal_(lin.weight)
# #     nn.init.zeros_(lin.bias)
# #     layers.append(lin)

# #     mlp = nn.Sequential(*layers)
# #     for p in mlp.parameters():
# #         p.requires_grad_(False)
# #     return mlp.eval()


# # def brownian_motion_box(T, d, sigma=0.03, seed=0):
# #     rng = np.random.default_rng(seed)
# #     x = np.zeros((T, d), dtype=np.float32)
# #     x[0] = rng.uniform(-1.0, 1.0, size=d).astype(np.float32)

# #     for t in range(T - 1):
# #         step = rng.normal(loc=0.0, scale=sigma, size=d).astype(np.float32)
# #         x[t + 1] = np.clip(x[t] + step, -1.0, 1.0)
# #     return x


# # def lorenz_system(T, dt=0.01, seed=0):
# #     """
# #     Generates T steps of the Lorenz attractor.
# #     Standard parameters: sigma=10, rho=28, beta=8/3
# #     """
# #     rng = np.random.default_rng(seed)

# #     sigma = 10.0
# #     rho = 28.0
# #     beta = 8.0 / 3.0

# #     xyz = np.zeros((T, 3), dtype=np.float32)

# #     # seed-dependent initial condition so z1 and z2 are different
# #     xyz[0] = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)

# #     for t in range(T - 1):
# #         x, y, z = xyz[t]
# #         dx = sigma * (y - x) * dt
# #         dy = (x * (rho - z) - y) * dt
# #         dz = (x * y - beta * z) * dt
# #         xyz[t + 1] = [x + dx, y + dy, z + dz]

# #     # Min-Max Normalize to [-1, 1]
# #     xyz_min = xyz.min(axis=0, keepdims=True)
# #     xyz_max = xyz.max(axis=0, keepdims=True)
# #     xyz_norm = 2.0 * (xyz - xyz_min) / (xyz_max - xyz_min + 1e-8) - 1.0

# #     return xyz_norm.astype(np.float32)


# # def make_binary_ground_truth(D1, D2, N1, N2):
# #     """
# #     Ground truth:
# #     - first D1 latents connect to all neurons
# #     - last D2 latents connect only to x2 block (last N2 neurons)
# #     shape = [D_LATENT, D_OBS] -> [6, 4]
# #     """
# #     gt = np.zeros((D1 + D2, N1 + N2), dtype=bool)
# #     gt[:D1, :] = True
# #     gt[D1:, N1:] = True
# #     return gt


# # def generate_synthetic_data(T=T, seed=42):
# #     # z1 uses Lorenz System (3 dimensions)
# #     z1 = lorenz_system(T, dt=0.01, seed=seed)

# #     # z2 also uses Lorenz System (3 dimensions)
# #     z2 = lorenz_system(T, dt=0.01, seed=seed + 1)

# #     g1 = make_mlp(D1, N1, n_layers=N_MLP_LAYERS, seed=seed + 10)
# #     g2 = make_mlp(D1 + D2, N2, n_layers=N_MLP_LAYERS, seed=seed + 20)

# #     z1_t = torch.tensor(z1, dtype=torch.float32)
# #     z2_t = torch.tensor(z2, dtype=torch.float32)

# #     with torch.no_grad():
# #         x1 = g1(z1_t).cpu().numpy()
# #         x2 = g2(torch.cat([z1_t, z2_t], dim=1)).cpu().numpy()

# #     x = np.concatenate([x1, x2], axis=1).astype(np.float32)
# #     latent = np.concatenate([z1, z2], axis=1).astype(np.float32)

# #     gt_bool = make_binary_ground_truth(D1, D2, N1, N2)
# #     gt_attr = gt_bool.astype(np.float32)

# #     return x, latent, gt_attr, gt_bool


# # # ============================================================
# # # 2) Utils
# # # ============================================================
# # def cleanup_cuda(*objs):
# #     for obj in objs:
# #         try:
# #             del obj
# #         except Exception:
# #             pass
# #     gc.collect()
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()
# #         torch.cuda.ipc_collect()


# # def reduce_attr_map(arr):
# #     """
# #     Convert attribution output to 2D map [output_dim, input_dim]
# #     if it has sample dimension.
# #     """
# #     arr = np.asarray(arr)
# #     if arr.ndim == 3:
# #         return np.abs(arr).mean(axis=0)
# #     if arr.ndim == 2:
# #         return np.abs(arr)
# #     if arr.ndim == 1:
# #         return np.abs(arr)[None, :]
# #     raise ValueError(f"Unsupported attribution shape: {arr.shape}")


# # def compute_auroc(attr_map_2d, gt_bool):
# #     y_true = gt_bool.ravel().astype(int)
# #     y_score = np.asarray(attr_map_2d, dtype=np.float64).ravel()

# #     if y_true.shape[0] != y_score.shape[0]:
# #         raise ValueError(
# #             f"Shape mismatch: y_true has {y_true.shape[0]} elements, "
# #             f"y_score has {y_score.shape[0]} elements, "
# #             f"attr_map shape={attr_map_2d.shape}, gt shape={gt_bool.shape}"
# #         )

# #     if len(np.unique(y_true)) < 2:
# #         return float("nan")

# #     return float(roc_auc_score(y_true, y_score))


# # # ============================================================
# # # 3) Data Generation
# # # ============================================================
# # print("Generating synthetic dataset...")
# # x_np, y_np, gt_attr, gt_attr_bool = generate_synthetic_data(T=T, seed=42)

# # print("x shape:", x_np.shape)             # (T, 4)
# # print("y shape:", y_np.shape)             # (T, 6)
# # print("gt_attr_bool shape:", gt_attr_bool.shape)  # (6, 4)

# # split_idx = int(0.8 * len(x_np))
# # train_data = x_np[:split_idx].astype(np.float32)
# # train_continuous_label = y_np[:split_idx].astype(np.float32)

# # # For later use in CEBRA setup
# # adv_epsilon = float(min_l2_distance(train_data)) / 2.0
# # adv_epsilon = max(adv_epsilon, 1e-6)

# # # ============================================================
# # # 4) Train + Attribution
# # # ============================================================
# # rows = []
# # all_results = {}

# # for adv in [False, True]:
# #     cleanup_cuda()

# #     model_name = "ACORN" if adv else "CEBRA"
# #     training_mode = "adversarial" if adv else "clean"

# #     print("\n" + "=" * 70)
# #     print(f"Training {model_name} ({training_mode})")
# #     print("=" * 70)

# #     model = CEBRA(
# #         batch_size=BATCH_SIZE,
# #         temperature=0.4,
# #         model_architecture="offset36-model-more-dropout",
# #         time_offsets=4,
# #         max_iterations=MAX_ITER,
# #         output_dimension=OUTPUT_DIM,
# #         verbose=True,
# #         training_mode=training_mode,
# #         adv_alpha=adv_epsilon / 5,
# #         adv_epsilon=adv_epsilon,
# #         adv_steps=10,
# #         attack_norm="linf",
# #         num_hidden_units=32,
# #     )

# #     model.fit(train_data, train_continuous_label)

# #     save_path = os.path.join(OUT_DIR, f"{model_name}_synthetic.pth")
# #     model.save(save_path)
# #     print("Saved model to:", save_path)

# #     trained_model = model.solver_.model.to(device)

# #     input_tensor = torch.from_numpy(train_data).float().to(device).requires_grad_(True)
# #     output_dim = int(getattr(trained_model, "num_output", OUTPUT_DIM))

# #     method = cebra.attribution.init(
# #         name="jacobian-based-batched",
# #         model=trained_model,
# #         input_data=input_tensor,
# #         output_dimension=output_dim,
# #     )

# #     result = method.compute_attribution_map(batch_size=min(128, len(train_data)))
# #     print("Attribution keys:", list(result.keys()))

# #     # Reduce to 2D maps
# #     jc_map = reduce_attr_map(result["jf"])                      # expected shape: (6, 4)
# #     jc_inv_map = reduce_attr_map(result["jf-inv-svd"])          # expected shape: (4, 6)
# #     jc_invconv_map = reduce_attr_map(result["jf-convabs-inv-svd"])  # expected shape: (4, 6)

# #     # AUROC scores
# #     auc_jc = compute_auroc(jc_map, gt_attr_bool)
# #     auc_jc_inv = compute_auroc(jc_inv_map.T, gt_attr_bool)
# #     auc_jc_invconv = compute_auroc(jc_invconv_map.T, gt_attr_bool)

# #     print(f"** {model_name} jc AUROC:        {auc_jc:.4f} **")
# #     print(f"** {model_name} jc_inv AUROC:    {auc_jc_inv:.4f} **")
# #     print(f"** {model_name} jc_invconv AUROC:{auc_jc_invconv:.4f} **")

# #     all_results[model_name] = {
# #         "jc": jc_map,
# #         "jc_inv": jc_inv_map,
# #         "jc_invconv": jc_invconv_map,
# #         "auc_jc": auc_jc,
# #         "auc_jc_inv": auc_jc_inv,
# #         "auc_jc_invconv": auc_jc_invconv,
# #     }

# #     rows.extend([
# #         {"model": model_name, "metric": "jc", "auroc": auc_jc},
# #         {"model": model_name, "metric": "jc_inv", "auroc": auc_jc_inv},
# #         {"model": model_name, "metric": "jc_invconv", "auroc": auc_jc_invconv},
# #     ])

# #     cleanup_cuda(method, trained_model, input_tensor, model)

# # # ============================================================
# # # 5) Summary
# # # ============================================================
# # print("\n" + "=" * 80)
# # print(" SUMMARY OF EXPERIMENT RESULTS ".center(80, "="))
# # print("=" * 80)
# # for model_name, res in all_results.items():
# #     print(
# #         f" Model: {model_name:<6} | "
# #         f"jc={res['auc_jc']:.4f} | "
# #         f"jc_inv={res['auc_jc_inv']:.4f} | "
# #         f"jc_invconv={res['auc_jc_invconv']:.4f}"
# #     )
# # print("=" * 80)

# # # ============================================================
# # # 6) Save CSV
# # # ============================================================
# # results_df = pd.DataFrame(rows)
# # csv_path = os.path.join(OUT_DIR, "synthetic_auroc_results.csv")
# # results_df.to_csv(csv_path, index=False)
# # print(f"Saved AUROC results to: {csv_path}")

# # print("Done.")

# #Brownian
# import os
# import gc
# import random
# import copy
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt

# from sklearn.metrics import roc_auc_score, average_precision_score

# from utils.min_distance import min_l2_distance
# from utils.constants import CEBRA_DIR

# import sys
# if "cebra" in sys.modules:
#     del sys.modules["cebra"]
# sys.path.insert(0, str(CEBRA_DIR))

# import cebra
# from cebra import CEBRA


# # ============================================================
# # 1) Global Config
# # ============================================================
# T = 100_000

# # Synthetic latent blocks
# D1 = 3
# D2 = 3

# # Observed neurons/features
# N1 = 25
# N2 = 25

# # Generator
# N_MLP_LAYERS = 4
# SIGMA_EPS_DEFAULT = 0.03

# # Training
# BATCH_SIZE = 2048
# MAX_ITER = 25000
# ATTR_BATCH_SIZE = 128

# # Model / run
# OUTPUT_DIM = D1 + D2
# OUT_DIR = "outputs"
# IMG_DIR = "images"
# os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(IMG_DIR, exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Keep one seed by default; add more if you want averaging
# SEEDS = [38,226,1,36,989,26,84,66,27,81,49]

# DATASET_CFG = {
#     "name": "FIG5_SINGLE",
#     "D1": D1,
#     "D2": D2,
#     "N1": N1,
#     "N2": N2,
#     "sigma_eps": SIGMA_EPS_DEFAULT,
# }


# # ============================================================
# # 2) Reproducibility
# # ============================================================
# def set_all_seeds(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     try:
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#     except Exception:
#         pass


# # ============================================================
# # 3) Synthetic Data Generation
# # ============================================================
# class ScaledTanh(nn.Module):
#     def __init__(self, scale=1.0):
#         super().__init__()
#         self.scale = float(scale)

#     def forward(self, x):
#         return self.scale * torch.tanh(x)


# def make_mlp(in_dim, out_dim, n_layers=4, seed=0):
#     """
#     Small random mixing network.
#     """
#     torch.manual_seed(seed)

#     layers = []
#     d_in = in_dim
#     hidden = max(64, 8 * max(in_dim, out_dim))

#     for _ in range(n_layers - 1):
#         lin = nn.Linear(d_in, hidden)
#         nn.init.orthogonal_(lin.weight)
#         nn.init.zeros_(lin.bias)
#         layers += [lin, nn.GELU()]
#         d_in = hidden

#     lin = nn.Linear(d_in, out_dim)
#     nn.init.orthogonal_(lin.weight)
#     nn.init.zeros_(lin.bias)
#     layers += [lin, ScaledTanh(scale=1.0)]

#     mlp = nn.Sequential(*layers).to(device).eval()
#     for p in mlp.parameters():
#         p.requires_grad_(False)
#     return mlp


# def brownian_motion_box(T, d, sigma=0.03, seed=0):
#     """
#     Brownian motion in [-1, 1]^d with rejection to keep it inside the box.
#     """
#     rng = np.random.default_rng(seed)
#     z = np.empty((T, d), dtype=np.float32)
#     z[0] = rng.uniform(-1.0, 1.0, size=d).astype(np.float32)

#     for t in range(1, T):
#         prev = z[t - 1].copy()
#         nxt = prev + rng.normal(0.0, sigma, size=d).astype(np.float32)

#         mask = (nxt < -1.0) | (nxt > 1.0)
#         while np.any(mask):
#             nxt[mask] = prev[mask] + rng.normal(0.0, sigma, size=mask.sum()).astype(np.float32)
#             mask = (nxt < -1.0) | (nxt > 1.0)

#         z[t] = nxt

#     return z


# def make_binary_ground_truth(D1, D2, N1, N2):
#     """
#     Ground truth attribution map:
#       rows = latent dimensions [z1, z2]
#       cols = observed neurons/features [x1, x2]

#     z1 -> x1 and x2
#     z2 -> x2 only
#     """
#     gt = np.zeros((D1 + D2, N1 + N2), dtype=bool)
#     gt[:D1, :] = True
#     gt[D1:, N1:] = True
#     return gt


# def generate_synthetic_data(cfg, seed=42):
#     D1 = int(cfg["D1"])
#     D2 = int(cfg["D2"])
#     N1 = int(cfg["N1"])
#     N2 = int(cfg["N2"])
#     sigma_eps = float(cfg.get("sigma_eps", SIGMA_EPS_DEFAULT))

#     z1 = brownian_motion_box(T, D1, sigma=sigma_eps, seed=seed)
#     z2 = brownian_motion_box(T, D2, sigma=sigma_eps, seed=seed + 1)

#     g1 = make_mlp(D1, N1, n_layers=N_MLP_LAYERS, seed=seed + 10)
#     g2 = make_mlp(D1 + D2, N2, n_layers=N_MLP_LAYERS, seed=seed + 20)

#     z1_t = torch.tensor(z1, dtype=torch.float32, device=device)
#     z2_t = torch.tensor(z2, dtype=torch.float32, device=device)

#     with torch.no_grad():
#         x1 = g1(z1_t).cpu().numpy()
#         x2 = g2(torch.cat([z1_t, z2_t], dim=1)).cpu().numpy()

#     x = np.concatenate([x1, x2], axis=1).astype(np.float32)
#     latent = np.concatenate([z1, z2], axis=1).astype(np.float32)

#     gt_bool = make_binary_ground_truth(D1, D2, N1, N2)
#     gt_attr = gt_bool.astype(np.float32)

#     return x, latent, gt_attr, gt_bool


# # ============================================================
# # 4) Utils
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
#     Works for torch.Tensor or numpy.ndarray.
#     """
#     if torch.is_tensor(arr):
#         arr = arr.detach().cpu().numpy()
#     else:
#         arr = np.asarray(arr)

#     arr = np.abs(arr)

#     if arr.ndim == 3:
#         # [samples, latent, features] -> average over samples
#         return arr.mean(axis=0).astype(np.float32)
#     if arr.ndim == 2:
#         return arr.astype(np.float32)
#     if arr.ndim == 1:
#         return np.abs(arr)[None, :].astype(np.float32)

#     raise ValueError(f"Unsupported attribution shape: {arr.shape}")


# def align_attr_to_gt(attr_map_2d, gt_bool):
#     if attr_map_2d.shape == gt_bool.shape:
#         return attr_map_2d
#     if attr_map_2d.T.shape == gt_bool.shape:
#         return attr_map_2d.T
#     raise ValueError(
#         f"Cannot align attribution map shape {attr_map_2d.shape} to ground truth shape {gt_bool.shape}"
#     )


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
#         return float("nan"), float("nan")

#     auroc = float(roc_auc_score(y_true, y_score))
#     auprc = float(average_precision_score(y_true, y_score))
#     return auroc, auprc


# def infer_adv_epsilon(train_x_np: np.ndarray) -> float:
#     try:
#         x_t = torch.tensor(train_x_np, dtype=torch.float32)
#         eps = float(min_l2_distance(x_t)) / 2.0
#         return max(eps, 1e-6)
#     except Exception:
#         return max(float(np.std(train_x_np)) * 0.05, 1e-6)


# def save_heatmap(mat, path, title):
#     fig, ax = plt.subplots(figsize=(10, 5))
#     im = ax.imshow(mat, aspect="auto", cmap="cividis")
#     ax.set_title(title)
#     ax.set_xlabel("Observed feature")
#     ax.set_ylabel("Latent dimension")
#     fig.colorbar(im, ax=ax, shrink=0.9)
#     fig.tight_layout()
#     fig.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close(fig)


# def subsample_for_attribution(x_np, max_points=10000):
#     if len(x_np) <= max_points:
#         return x_np
#     idx = np.linspace(0, len(x_np) - 1, max_points).astype(int)
#     return x_np[idx]


# # ============================================================
# # 5) Model / Attribution
# # ============================================================
# def build_model(adv: bool, adv_epsilon: float):
#     return CEBRA(
#         batch_size=BATCH_SIZE,
#         temperature=0.4,
#         model_architecture="offset36-model-more-dropout",
#         time_offsets=4,
#         max_iterations=MAX_ITER,
#         output_dimension=OUTPUT_DIM,
#         verbose=True,
#         training_mode="adversarial" if adv else "clean",
#         adv_alpha=(5 / 5.0) if adv else 0.0,
#         adv_epsilon=5 if adv else 0.0,
#         adv_steps=10 if adv else 0,
#         attack_norm="linf",
#         num_hidden_units=32,
#         device="cuda_if_available",
#     )


# def train_and_score_one_run(
#     cfg,
#     seed,
#     train_x_np,
#     z1_train_np,
#     z2_train_np,
#     gt_bool,
#     adv: bool,
# ):
#     set_all_seeds(seed)

#     model_name = "ACORN" if adv else "CEBRA"
#     adv_epsilon = infer_adv_epsilon(train_x_np) if adv else 0.0

#     model = build_model(adv=adv, adv_epsilon=adv_epsilon)
#     model.fit(
#         train_x_np.astype(np.float32),
#         z1_train_np.astype(np.float32),
#         z2_train_np.astype(np.float32),
#     )

#     save_path = os.path.join(OUT_DIR, f"{cfg['name']}_seed{seed}_{model_name}.pth")
#     try:
#         model.save(save_path)
#         print("Saved model to:", save_path)
#     except Exception as e:
#         print("Could not save model:", e)

#     trained_model = model.solver_.model.to(device)
#     if hasattr(trained_model, "split_outputs"):
#         trained_model.split_outputs = False
#     trained_model.eval()

#     attr_x = subsample_for_attribution(train_x_np, max_points=10000)
#     input_tensor = torch.from_numpy(attr_x.astype(np.float32)).to(device)
#     input_tensor.requires_grad_(True)

#     method = cebra.attribution.init(
#         name="jacobian-based-batched",
#         model=trained_model,
#         input_data=input_tensor,
#         output_dimension=int(getattr(trained_model, "num_output", OUTPUT_DIM)),
#     )

#     batch_size = min(ATTR_BATCH_SIZE, len(attr_x))
#     result = method.compute_attribution_map(batch_size=batch_size)
#     print("Attribution keys:", list(result.keys()))

#     jf_key = "jf"
#     if "jf-inv-svd" in result:
#         jfinv_key = "jf-inv-svd"
#     elif "jf-inv" in result:
#         jfinv_key = "jf-inv"
#     else:
#         raise KeyError(f"No inverse attribution key found. Available: {list(result.keys())}")

#     jf_raw = reduce_attr_map(result[jf_key])
#     jfinv_raw = reduce_attr_map(result[jfinv_key])

#     jf_map = align_attr_to_gt(jf_raw, gt_bool)
#     jfinv_map = align_attr_to_gt(jfinv_raw, gt_bool)

#     auroc_jf, auprc_jf = compute_auroc(jf_map, gt_bool)
#     auroc_jfinv, auprc_jfinv = compute_auroc(jfinv_map, gt_bool)

#     print(f"[{cfg['name']}] seed={seed} | {model_name} | JF     AUROC: {auroc_jf:.4f} | AUPRC: {auprc_jf:.4f}")
#     print(f"[{cfg['name']}] seed={seed} | {model_name} | JF-INV AUROC: {auroc_jfinv:.4f} | AUPRC: {auprc_jfinv:.4f}")

#     # Save results / plots
#     run_tag = f"{cfg['name']}_seed{seed}_{model_name}"
#     np.savez_compressed(
#         os.path.join(OUT_DIR, f"{run_tag}_attrs.npz"),
#         jf=jf_map.astype(np.float32),
#         jfinv=jfinv_map.astype(np.float32),
#         gt=gt_bool.astype(np.uint8),
#         auroc_jf=np.array([auroc_jf], dtype=np.float32),
#         auprc_jf=np.array([auprc_jf], dtype=np.float32),
#         auroc_jfinv=np.array([auroc_jfinv], dtype=np.float32),
#         auprc_jfinv=np.array([auprc_jfinv], dtype=np.float32),
#     )

#     save_heatmap(jf_map, os.path.join(IMG_DIR, f"{run_tag}_JF.png"), f"{run_tag} | JF")
#     save_heatmap(jfinv_map, os.path.join(IMG_DIR, f"{run_tag}_JF_INV.png"), f"{run_tag} | JF-INV")
#     save_heatmap(gt_bool.astype(np.float32), os.path.join(IMG_DIR, f"{run_tag}_GT.png"), f"{run_tag} | GT")

#     cleanup_cuda(method, trained_model, input_tensor, model)

#     return {
#         "setup": cfg["name"],
#         "seed": seed,
#         "D1": int(cfg["D1"]),
#         "D2": int(cfg["D2"]),
#         "N1": int(cfg["N1"]),
#         "N2": int(cfg["N2"]),
#         "D_LATENT": int(cfg["D1"] + cfg["D2"]),
#         "D_OBS": int(cfg["N1"] + cfg["N2"]),
#         "model": model_name,
#         "training_mode": "adversarial" if adv else "clean",
#         "auroc_jf": auroc_jf,
#         "auprc_jf": auprc_jf,
#         "auroc_jfinv": auroc_jfinv,
#         "auprc_jfinv": auprc_jfinv,
#     }


# # ============================================================
# # 6) Main
# # ============================================================
# def main():
#     all_rows = []

#     print("\n" + "#" * 90)
#     print(
#         f"SETUP: {DATASET_CFG['name']} | "
#         f"D1={DATASET_CFG['D1']} D2={DATASET_CFG['D2']} | "
#         f"N1={DATASET_CFG['N1']} N2={DATASET_CFG['N2']} | "
#         f"sigma={DATASET_CFG['sigma_eps']}"
#     )
#     print("#" * 90)

#     # Generate one synthetic dataset per seed for reproducibility
#     for seed in SEEDS:
#         set_all_seeds(seed)
#         x_np, latent_np, gt_attr, gt_bool = generate_synthetic_data(DATASET_CFG, seed=seed)

#         print("x shape:", x_np.shape)
#         print("latent shape:", latent_np.shape)
#         print("gt_attr shape:", gt_attr.shape)

#         split_idx = int(0.8 * len(x_np))
#         train_x = x_np[:split_idx].astype(np.float32)
#         train_latent = latent_np[:split_idx].astype(np.float32)

#         z1_train = train_latent[:, :DATASET_CFG["D1"]]
#         z2_train = train_latent[:, DATASET_CFG["D1"]:]

#         # Save one copy of the synthetic ground truth maps for inspection
#         save_heatmap(gt_bool.astype(np.float32), os.path.join(IMG_DIR, f"{DATASET_CFG['name']}_seed{seed}_GT_ONLY.png"), "GT only")

#         adv_epsilon = infer_adv_epsilon(train_x)
#         print("adv_epsilon:", adv_epsilon)

#         for adv in [False, True]:
#             cleanup_cuda()
#             mode_name = "adversarial" if adv else "clean"
#             print("\n" + "=" * 70)
#             print(f"Training {DATASET_CFG['name']} | seed={seed} | mode={mode_name}")
#             print("=" * 70)

#             row = train_and_score_one_run(
#                 cfg=DATASET_CFG,
#                 seed=seed,
#                 train_x_np=train_x,
#                 z1_train_np=z1_train,
#                 z2_train_np=z2_train,
#                 gt_bool=gt_bool,
#                 adv=adv,
#             )
#             all_rows.append(row)

#     results_df = pd.DataFrame(all_rows)

#     detailed_csv = os.path.join(OUT_DIR, "synthetic_auroc_detailed.csv")
#     results_df.to_csv(detailed_csv, index=False)
#     print(f"\nSaved detailed results to: {detailed_csv}")

#     summary_df = (
#         results_df
#         .groupby(["setup", "model", "training_mode", "D1", "D2", "N1", "N2", "D_LATENT", "D_OBS"], as_index=False)
#         .agg(
#             auroc_jf_mean=("auroc_jf", "mean"),
#             auroc_jf_std=("auroc_jf", "std"),
#             auprc_jf_mean=("auprc_jf", "mean"),
#             auprc_jf_std=("auprc_jf", "std"),
#             auroc_jfinv_mean=("auroc_jfinv", "mean"),
#             auroc_jfinv_std=("auroc_jfinv", "std"),
#             auprc_jfinv_mean=("auprc_jfinv", "mean"),
#             auprc_jfinv_std=("auprc_jfinv", "std"),
#             n_runs=("seed", "count"),
#         )
#     )

#     summary_csv = os.path.join(OUT_DIR, "synthetic_auroc_summary.csv")
#     summary_df.to_csv(summary_csv, index=False)
#         # ============================================================
#     # Save seed-wise + mean ± std results
#     # ============================================================

#     fake_result_dir = "Fake_dataset_result"
#     os.makedirs(fake_result_dir, exist_ok=True)

#     seed_rows = []

#     # individual seed results
#     for _, row in results_df.iterrows():
#         seed_rows.append({
#             "type": "seed",
#             "setup": row["setup"],
#             "model": row["model"],
#             "training_mode": row["training_mode"],
#             "seed": row["seed"],
#             "AUROC_JF": row["auroc_jf"],
#             "AUROC_JFINV": row["auroc_jfinv"],
#             "AUPRC_JF": row["auprc_jf"],
#             "AUPRC_JFINV": row["auprc_jfinv"],
#         })


#     # mean ± std (paper format)
#     for _, row in summary_df.iterrows():
#         seed_rows.append({
#             "type": "mean_std",
#             "setup": row["setup"],
#             "model": row["model"],
#             "training_mode": row["training_mode"],
#             "seed": "all",

#             "AUROC_JF":
#                 f"{row['auroc_jf_mean']:.4f} ± {row['auroc_jf_std']:.4f}",

#             "AUROC_JFINV":
#                 f"{row['auroc_jfinv_mean']:.4f} ± {row['auroc_jfinv_std']:.4f}",

#             "AUPRC_JF":
#                 f"{row['auprc_jf_mean']:.4f} ± {row['auprc_jf_std']:.4f}",

#             "AUPRC_JFINV":
#                 f"{row['auprc_jfinv_mean']:.4f} ± {row['auprc_jfinv_std']:.4f}",
#         })


#     seed_summary_df = pd.DataFrame(seed_rows)

#     seed_summary_csv = os.path.join(
#         fake_result_dir,
#         "Fake_dataset_all_seeds_results.csv"
#     )

#     seed_summary_df.to_csv(seed_summary_csv, index=False)

#     print(f"Saved seed results to: {seed_summary_csv}")
#     print(f"Saved summary results to: {summary_csv}")

#     print("\n" + "=" * 120)
#     print(" SUMMARY ".center(120, "="))
#     print("=" * 120)
#     print(summary_df.to_string(index=False))
#     print("=" * 120)

#     print("Done.")


# if __name__ == "__main__":
#     main()




# #$#$#$#$#$#$#$#$#

# #$#$#$#$#$#$#$#$#

# #$#$#$#$#$#$#$#$#
# # import os
# # import gc
# # import random
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import pandas as pd

# # from sklearn.metrics import roc_auc_score
# # from utils.min_distance import min_l2_distance
# # from utils.constants import CEBRA_DIR

# # import sys
# # sys.path.insert(0, str(CEBRA_DIR))
# # import cebra
# # from cebra import CEBRA


# # # ============================================================
# # # 1) Global Config
# # # ============================================================
# # T = 100_000
# # N_MLP_LAYERS = 4
# # SIGMA_EPS_DEFAULT = 0.03

# # BATCH_SIZE = 2048
# # MAX_ITER = 15000
# # ATTR_BATCH_SIZE = 128

# # OUT_DIR = "outputs"
# # os.makedirs(OUT_DIR, exist_ok=True)

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # SEEDS = [38]

# # # One dataset only
# # DATASET_CFG = {
# #     "name": "FIG5_SINGLE",
# #     "D1": 3,
# #     "D2": 3,
# #     "N1": 25,
# #     "N2": 25,
# #     "sigma_eps": SIGMA_EPS_DEFAULT,
# # }


# # # ============================================================
# # # 2) Reproducibility
# # # ============================================================
# # def set_all_seeds(seed: int) -> None:
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed(seed)
# #         torch.cuda.manual_seed_all(seed)
# #     try:
# #         torch.backends.cudnn.deterministic = True
# #         torch.backends.cudnn.benchmark = False
# #     except Exception:
# #         pass


# # # ============================================================
# # # 3) Synthetic Data Generation
# # # ============================================================
# # class ScaledTanh(nn.Module):
# #     def __init__(self, scale=1.0):
# #         super().__init__()
# #         self.scale = float(scale)

# #     def forward(self, x):
# #         return self.scale * torch.tanh(x)


# # def make_mlp(in_dim, out_dim, n_layers=4, seed=0):
# #     """
# #     3 hidden layers with GELU, then one output layer + scaled tanh.
# #     IMPORTANT: move the model to the same device as the inputs.
# #     """
# #     torch.manual_seed(seed)

# #     layers = []
# #     d_in = in_dim
# #     hidden = max(64, 8 * max(in_dim, out_dim))

# #     for _ in range(n_layers - 1):
# #         lin = nn.Linear(d_in, hidden)
# #         nn.init.orthogonal_(lin.weight)
# #         nn.init.zeros_(lin.bias)
# #         layers += [lin, nn.GELU()]
# #         d_in = hidden

# #     lin = nn.Linear(d_in, out_dim)
# #     nn.init.orthogonal_(lin.weight)
# #     nn.init.zeros_(lin.bias)
# #     layers += [lin, ScaledTanh(scale=1.0)]

# #     mlp = nn.Sequential(*layers).to(device).eval()
# #     for p in mlp.parameters():
# #         p.requires_grad_(False)
# #     return mlp


# # def brownian_motion_box(T, d, sigma=0.03, seed=0):
# #     """
# #     Brownian motion in [-1, 1]^d.
# #     Uses rejection sampling so the latent always stays in the box.
# #     """
# #     rng = np.random.default_rng(seed)
# #     z = np.empty((T, d), dtype=np.float32)
# #     z[0] = rng.uniform(-1.0, 1.0, size=d).astype(np.float32)

# #     for t in range(1, T):
# #         prev = z[t - 1].copy()
# #         nxt = prev + rng.normal(0.0, sigma, size=d).astype(np.float32)

# #         mask = (nxt < -1.0) | (nxt > 1.0)
# #         while np.any(mask):
# #             nxt[mask] = prev[mask] + rng.normal(0.0, sigma, size=mask.sum()).astype(np.float32)
# #             mask = (nxt < -1.0) | (nxt > 1.0)

# #         z[t] = nxt

# #     return z


# # def make_binary_ground_truth(D1, D2, N1, N2):
# #     """
# #     Ground truth map:
# #       rows = latent variables [z1, z2]
# #       cols = observed neurons [x1, x2]

# #     z1 -> x1 and x2
# #     z2 -> x2 only
# #     """
# #     gt = np.zeros((D1 + D2, N1 + N2), dtype=bool)
# #     gt[:D1, :] = True
# #     gt[D1:, N1:] = True
# #     return gt


# # def generate_synthetic_data(cfg, seed=42):
# #     D1 = int(cfg["D1"])
# #     D2 = int(cfg["D2"])
# #     N1 = int(cfg["N1"])
# #     N2 = int(cfg["N2"])
# #     sigma_eps = float(cfg.get("sigma_eps", SIGMA_EPS_DEFAULT))

# #     z1 = brownian_motion_box(T, D1, sigma=sigma_eps, seed=seed)
# #     z2 = brownian_motion_box(T, D2, sigma=sigma_eps, seed=seed + 1)

# #     # mixing functions (new random mixing per seed)
# #     g1 = make_mlp(D1, N1, n_layers=N_MLP_LAYERS, seed=seed + 10)
# #     g2 = make_mlp(D1 + D2, N2, n_layers=N_MLP_LAYERS, seed=seed + 20)

# #     z1_t = torch.tensor(z1, dtype=torch.float32, device=device)
# #     z2_t = torch.tensor(z2, dtype=torch.float32, device=device)

# #     with torch.no_grad():
# #         x1 = g1(z1_t).cpu().numpy()
# #         x2 = g2(torch.cat([z1_t, z2_t], dim=1)).cpu().numpy()

# #     x = np.concatenate([x1, x2], axis=1).astype(np.float32)
# #     latent = np.concatenate([z1, z2], axis=1).astype(np.float32)

# #     gt_bool = make_binary_ground_truth(D1, D2, N1, N2)
# #     gt_attr = gt_bool.astype(np.float32)

# #     return x, latent, gt_attr, gt_bool


# # # ============================================================
# # # 4) Utils
# # # ============================================================
# # def cleanup_cuda(*objs):
# #     for obj in objs:
# #         try:
# #             del obj
# #         except Exception:
# #             pass
# #     gc.collect()
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()
# #         torch.cuda.ipc_collect()


# # def reduce_attr_map(arr):
# #     """
# #     Convert attribution output to 2D map [output_dim, input_dim]
# #     if it has sample dimension.
# #     """
# #     if torch.is_tensor(arr):
# #         arr = arr.detach().cpu().numpy()
# #     else:
# #         arr = np.asarray(arr)

# #     if arr.ndim == 3:
# #         return np.abs(arr).mean(axis=0)
# #     if arr.ndim == 2:
# #         return np.abs(arr)
# #     if arr.ndim == 1:
# #         return np.abs(arr)[None, :]
# #     raise ValueError(f"Unsupported attribution shape: {arr.shape}")


# # def align_attr_to_gt(attr_map_2d, gt_bool):
# #     if attr_map_2d.shape == gt_bool.shape:
# #         return attr_map_2d
# #     if attr_map_2d.T.shape == gt_bool.shape:
# #         return attr_map_2d.T
# #     raise ValueError(
# #         f"Cannot align attribution map shape {attr_map_2d.shape} "
# #         f"to ground truth shape {gt_bool.shape}"
# #     )


# # def compute_auroc(attr_map_2d, gt_bool):
# #     y_true = gt_bool.ravel().astype(int)
# #     y_score = np.asarray(attr_map_2d, dtype=np.float64).ravel()

# #     if y_true.shape[0] != y_score.shape[0]:
# #         raise ValueError(
# #             f"Shape mismatch: y_true has {y_true.shape[0]} elements, "
# #             f"y_score has {y_score.shape[0]} elements, "
# #             f"attr_map shape={attr_map_2d.shape}, gt shape={gt_bool.shape}"
# #         )

# #     if len(np.unique(y_true)) < 2:
# #         return float("nan")

# #     return float(roc_auc_score(y_true, y_score))


# # # ============================================================
# # # 5) Training + Attribution
# # # ============================================================
# # def run_one_model(
# #     cfg,
# #     seed,
# #     train_data,
# #     train_continuous_label,
# #     gt_attr_bool,
# #     training_mode,
# #     adv_epsilon,
# # ):
# #     D1 = int(cfg["D1"])
# #     D2 = int(cfg["D2"])
# #     N1 = int(cfg["N1"])
# #     N2 = int(cfg["N2"])
# #     D_LATENT = D1 + D2

# #     model_name = "ACORN" if training_mode == "adversarial" else "CEBRA"

# #     set_all_seeds(seed)

# #     model = CEBRA(
# #         batch_size=BATCH_SIZE,
# #         temperature=0.4,
# #         model_architecture="offset36-model-more-dropout",
# #         time_offsets=4,
# #         max_iterations=MAX_ITER,
# #         output_dimension=D_LATENT,
# #         verbose=True,
# #         training_mode=training_mode,
# #         adv_alpha=0.1 / 5,
# #         adv_epsilon=0.1,
# #         adv_steps=10,
# #         attack_norm="linf",   # keep your own setting
# #         num_hidden_units=32,
# #     )

# #     model.fit(train_data, train_continuous_label)

# #     save_path = os.path.join(OUT_DIR, f"{cfg['name']}_seed{seed}_{model_name}.pth")
# #     try:
# #         model.save(save_path)
# #         print("Saved model to:", save_path)
# #     except Exception as e:
# #         print("Could not save model:", e)

# #     trained_model = model.solver_.model.to(device)

# #     input_tensor = torch.from_numpy(train_data).float().to(device).requires_grad_(True)

# #     method = cebra.attribution.init(
# #         name="jacobian-based-batched",
# #         model=trained_model,
# #         input_data=input_tensor,
# #         output_dimension=D_LATENT,
# #     )

# #     result = method.compute_attribution_map(batch_size=min(ATTR_BATCH_SIZE, len(train_data)))
# #     print("Attribution keys:", list(result.keys()))

# #     jac_raw = reduce_attr_map(result["jf"])
# #     jac_inv_raw = reduce_attr_map(result["jf-inv-svd"])

# #     jac_map = align_attr_to_gt(jac_raw, gt_attr_bool)
# #     jac_inv_map = align_attr_to_gt(jac_inv_raw, gt_attr_bool)

# #     auc_jac = compute_auroc(jac_map, gt_attr_bool)
# #     auc_jac_inv = compute_auroc(jac_inv_map, gt_attr_bool)

# #     print(f"** {cfg['name']} | seed={seed} | {model_name} jac AUROC:     {auc_jac:.4f} **")
# #     print(f"** {cfg['name']} | seed={seed} | {model_name} jac_inv AUROC: {auc_jac_inv:.4f} **")

# #     cleanup_cuda(method, trained_model, input_tensor, model)

# #     return {
# #         "setup": cfg["name"],
# #         "seed": seed,
# #         "D1": D1,
# #         "D2": D2,
# #         "N1": N1,
# #         "N2": N2,
# #         "D_LATENT": D_LATENT,
# #         "D_OBS": N1 + N2,
# #         "model": model_name,
# #         "training_mode": training_mode,
# #         "jac_auc": auc_jac,
# #         "jac_inv_auc": auc_jac_inv,
# #     }


# # # ============================================================
# # # 6) Main
# # # ============================================================
# # all_rows = []

# # print("\n" + "#" * 90)
# # print(f"SETUP: {DATASET_CFG['name']} | D1={DATASET_CFG['D1']} D2={DATASET_CFG['D2']} | N1={DATASET_CFG['N1']} N2={DATASET_CFG['N2']} | sigma={DATASET_CFG['sigma_eps']}")
# # print("#" * 90)

# # set_all_seeds(42)
# # x_np, y_np, gt_attr, gt_attr_bool = generate_synthetic_data(DATASET_CFG, seed=42)

# # print("x shape:", x_np.shape)
# # print("y shape:", y_np.shape)
# # print("gt_attr_bool shape:", gt_attr_bool.shape)

# # split_idx = int(0.8 * len(x_np))
# # train_data = x_np[:split_idx].astype(np.float32)
# # train_continuous_label = y_np[:split_idx].astype(np.float32)

# # adv_epsilon = float(min_l2_distance(train_data)) / 2.0
# # adv_epsilon = max(adv_epsilon, 1e-6)
# # print("adv_epsilon:", adv_epsilon)

# # for seed in SEEDS:
# #     for training_mode in ["clean", "adversarial"]:
# #         cleanup_cuda()

# #         print("\n" + "=" * 70)
# #         print(f"Training {DATASET_CFG['name']} | seed={seed} | mode={training_mode}")
# #         print("=" * 70)

# #         row = run_one_model(
# #             cfg=DATASET_CFG,
# #             seed=seed,
# #             train_data=train_data,
# #             train_continuous_label=train_continuous_label,
# #             gt_attr_bool=gt_attr_bool,
# #             training_mode=training_mode,
# #             adv_epsilon=adv_epsilon,
# #         )
# #         all_rows.append(row)

# # results_df = pd.DataFrame(all_rows)

# # detailed_csv = os.path.join(OUT_DIR, "synthetic_auroc_detailed.csv")
# # results_df.to_csv(detailed_csv, index=False)
# # print(f"\nSaved detailed results to: {detailed_csv}")

# # summary_df = (
# #     results_df
# #     .groupby(["setup", "model", "training_mode", "D1", "D2", "N1", "N2", "D_LATENT", "D_OBS"], as_index=False)
# #     .agg(
# #         jac_mean=("jac_auc", "mean"),
# #         jac_std=("jac_auc", "std"),
# #         jac_inv_mean=("jac_inv_auc", "mean"),
# #         jac_inv_std=("jac_inv_auc", "std"),
# #         n_runs=("seed", "count"),
# #     )
# # )

# # summary_csv = os.path.join(OUT_DIR, "synthetic_auroc_summary.csv")
# # summary_df.to_csv(summary_csv, index=False)
# # print(f"Saved summary results to: {summary_csv}")

# # print("\n" + "=" * 120)
# # print(" SUMMARY ".center(120, "="))
# # print("=" * 120)
# # print(summary_df.to_string(index=False))
# # print("=" * 120)

# # print("Done.")

# # import os
# # import gc
# # import random
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import pandas as pd

# # from sklearn.metrics import roc_auc_score
# # from utils.min_distance import min_l2_distance
# # from utils.constants import CEBRA_DIR

# # import sys
# # sys.path.insert(0, str(CEBRA_DIR))
# # import cebra
# # from cebra import CEBRA


# # # ============================================================
# # # 1) Synthetic Data Config & Generation
# # # ============================================================
# # T = 100_000
# # D1 = 3
# # D2 = 3
# # D_LATENT = D1 + D2

# # N1 = 2
# # N2 = 2
# # D_OBS = N1 + N2

# # N_MLP_LAYERS = 4
# # SIGMA_EPS = 0.03

# # OUTPUT_DIM = D_LATENT
# # BATCH_SIZE = 2048
# # MAX_ITER = 2500
# # adv_epsilon_default = 0.5

# # ATTR_BATCH_SIZE = 128

# # OUT_DIR = "outputs"
# # os.makedirs(OUT_DIR, exist_ok=True)

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # RANDOM_SEED = 42
# # np.random.seed(RANDOM_SEED)
# # torch.manual_seed(RANDOM_SEED)
# # random.seed(RANDOM_SEED)


# # def make_mlp(in_dim, out_dim, n_layers=4, seed=0):
# #     torch.manual_seed(seed)
# #     layers = []
# #     d_in = in_dim
# #     hidden = in_dim * 10

# #     for i in range(n_layers - 1):
# #         d_h = in_dim * 30 if i < n_layers - 2 else hidden
# #         lin = nn.Linear(d_in, d_h)
# #         nn.init.orthogonal_(lin.weight)
# #         nn.init.zeros_(lin.bias)
# #         layers += [lin, nn.GELU()]
# #         d_in = d_h

# #     lin = nn.Linear(d_in, out_dim)
# #     nn.init.orthogonal_(lin.weight)
# #     nn.init.zeros_(lin.bias)
# #     layers.append(lin)

# #     mlp = nn.Sequential(*layers)
# #     for p in mlp.parameters():
# #         p.requires_grad_(False)
# #     return mlp.eval()


# # def brownian_motion_box(T, d, sigma=0.03, seed=0):
# #     rng = np.random.default_rng(seed)
# #     x = np.zeros((T, d), dtype=np.float32)
# #     x[0] = rng.uniform(-1.0, 1.0, size=d).astype(np.float32)

# #     for t in range(T - 1):
# #         step = rng.normal(loc=0.0, scale=sigma, size=d).astype(np.float32)
# #         x[t + 1] = np.clip(x[t] + step, -1.0, 1.0)
# #     return x


# # def make_binary_ground_truth(D1, D2, N1, N2):
# #     """
# #     Ground truth:
# #     - first D1 latents connect to all neurons
# #     - last D2 latents connect only to x2 block (last N2 neurons)
# #     shape = [D_LATENT, D_OBS]
# #     """
# #     gt = np.zeros((D1 + D2, N1 + N2), dtype=bool)
# #     gt[:D1, :] = True
# #     gt[D1:, N1:] = True
# #     return gt


# # def generate_synthetic_data(T=T, seed=42):
# #     z1 = brownian_motion_box(T, D1, sigma=SIGMA_EPS, seed=seed)
# #     z2 = brownian_motion_box(T, D2, sigma=SIGMA_EPS, seed=seed + 1)

# #     g1 = make_mlp(D1, N1, n_layers=N_MLP_LAYERS, seed=seed + 10)
# #     g2 = make_mlp(D1 + D2, N2, n_layers=N_MLP_LAYERS, seed=seed + 20)

# #     z1_t = torch.tensor(z1, dtype=torch.float32)
# #     z2_t = torch.tensor(z2, dtype=torch.float32)

# #     with torch.no_grad():
# #         x1 = g1(z1_t).cpu().numpy()
# #         x2 = g2(torch.cat([z1_t, z2_t], dim=1)).cpu().numpy()

# #     x = np.concatenate([x1, x2], axis=1).astype(np.float32)
# #     latent = np.concatenate([z1, z2], axis=1).astype(np.float32)

# #     gt_bool = make_binary_ground_truth(D1, D2, N1, N2)
# #     gt_attr = gt_bool.astype(np.float32)

# #     return x, latent, gt_attr, gt_bool


# # # ============================================================
# # # 2) Utils
# # # ============================================================
# # def cleanup_cuda(*objs):
# #     for obj in objs:
# #         try:
# #             del obj
# #         except Exception:
# #             pass
# #     gc.collect()
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()
# #         torch.cuda.ipc_collect()


# # def reduce_attr_map(arr):
# #     """
# #     Convert attribution output to 2D map [output_dim, input_dim]
# #     if it has sample dimension.
# #     """
# #     arr = np.asarray(arr)
# #     if arr.ndim == 3:
# #         return np.abs(arr).mean(axis=0)
# #     if arr.ndim == 2:
# #         return np.abs(arr)
# #     if arr.ndim == 1:
# #         return np.abs(arr)[None, :]
# #     raise ValueError(f"Unsupported attribution shape: {arr.shape}")


# # def compute_auroc(attr_map_2d, gt_bool):
# #     y_true = gt_bool.ravel().astype(int)
# #     y_score = np.asarray(attr_map_2d, dtype=np.float64).ravel()

# #     if y_true.shape[0] != y_score.shape[0]:
# #         raise ValueError(
# #             f"Shape mismatch: y_true has {y_true.shape[0]} elements, "
# #             f"y_score has {y_score.shape[0]} elements, "
# #             f"attr_map shape={attr_map_2d.shape}, gt shape={gt_bool.shape}"
# #         )

# #     if len(np.unique(y_true)) < 2:
# #         return float("nan")

# #     return float(roc_auc_score(y_true, y_score))


# # # ============================================================
# # # 3) Data Generation
# # # ============================================================
# # print("Generating synthetic dataset...")
# # x_np, y_np, gt_attr, gt_attr_bool = generate_synthetic_data(T=T, seed=42)

# # print("x shape:", x_np.shape)  # (T, 6)
# # print("y shape:", y_np.shape)  # (T, 4)
# # print("gt_attr_bool shape:", gt_attr_bool.shape)  # (4, 6)

# # split_idx = int(0.8 * len(x_np))
# # train_data = x_np[:split_idx].astype(np.float32)
# # train_continuous_label = y_np[:split_idx].astype(np.float32)

# # # For later use in CEBRA setup
# # adv_epsilon = float(min_l2_distance(train_data)) / 2.0
# # adv_epsilon = max(adv_epsilon, 1e-6)

# # # ============================================================
# # # 4) Train + Attribution
# # # ============================================================
# # rows = []
# # all_results = {}

# # for adv in [False, True]:
# #     cleanup_cuda()

# #     model_name = "ACORN" if adv else "CEBRA"
# #     training_mode = "adversarial" if adv else "clean"

# #     print("\n" + "=" * 70)
# #     print(f"Training {model_name} ({training_mode})")
# #     print("=" * 70)

# #     model = CEBRA(
# #         batch_size=BATCH_SIZE,
# #         temperature=0.4,
# #         model_architecture="offset36-model-more-dropout",
# #         time_offsets=4,
# #         max_iterations=MAX_ITER,
# #         output_dimension=OUTPUT_DIM,
# #         verbose=True,
# #         training_mode=training_mode,
# #         adv_alpha=adv_epsilon / 5,
# #         adv_epsilon=adv_epsilon,
# #         adv_steps=10,
# #         attack_norm="linf",
# #         num_hidden_units=32,
# #     )

# #     model.fit(train_data, train_continuous_label)

# #     save_path = os.path.join(OUT_DIR, f"{model_name}_synthetic.pth")
# #     model.save(save_path)
# #     print("Saved model to:", save_path)

# #     trained_model = model.solver_.model.to(device)

# #     # Use full training data for attribution; batch_size smaller to avoid errors
# #     input_tensor = torch.from_numpy(train_data).float().to(device).requires_grad_(True)

# #     output_dim = int(getattr(trained_model, "num_output", OUTPUT_DIM))

# #     method = cebra.attribution.init(
# #         name="jacobian-based-batched",
# #         model=trained_model,
# #         input_data=input_tensor,
# #         output_dimension=output_dim,
# #     )

# #     result = method.compute_attribution_map(batch_size=min(128, len(train_data)))
# #     print("Attribution keys:", list(result.keys()))

# #     # Reduce to 2D maps
# #     jc_map = reduce_attr_map(result["jf"])
# #     jc_inv_map = reduce_attr_map(result["jf-inv-svd"])
# #     jc_invconv_map = reduce_attr_map(result["jf-convabs-inv-svd"])

# #     # AUROC scores
# #     auc_jc = compute_auroc(jc_map, gt_attr_bool)
# #     auc_jc_inv = compute_auroc(jc_inv_map, gt_attr_bool)
# #     auc_jc_invconv = compute_auroc(jc_invconv_map, gt_attr_bool)

# #     print(f"** {model_name} jc AUROC:        {auc_jc:.4f} **")
# #     print(f"** {model_name} jc_inv AUROC:    {auc_jc_inv:.4f} **")
# #     print(f"** {model_name} jc_invconv AUROC:{auc_jc_invconv:.4f} **")

# #     all_results[model_name] = {
# #         "jc": jc_map,
# #         "jc_inv": jc_inv_map,
# #         "jc_invconv": jc_invconv_map,
# #         "auc_jc": auc_jc,
# #         "auc_jc_inv": auc_jc_inv,
# #         "auc_jc_invconv": auc_jc_invconv,
# #     }

# #     rows.extend([
# #         {"model": model_name, "metric": "jc", "auroc": auc_jc},
# #         {"model": model_name, "metric": "jc_inv", "auroc": auc_jc_inv},
# #         {"model": model_name, "metric": "jc_invconv", "auroc": auc_jc_invconv},
# #     ])

# #     cleanup_cuda(method, trained_model, input_tensor, model)

# # # ============================================================
# # # 5) Summary
# # # ============================================================
# # print("\n" + "=" * 80)
# # print(" SUMMARY OF EXPERIMENT RESULTS ".center(80, "="))
# # print("=" * 80)
# # for model_name, res in all_results.items():
# #     print(
# #         f" Model: {model_name:<6} | "
# #         f"jc={res['auc_jc']:.4f} | "
# #         f"jc_inv={res['auc_jc_inv']:.4f} | "
# #         f"jc_invconv={res['auc_jc_invconv']:.4f}"
# #     )
# # print("=" * 80)

# # # ============================================================
# # # 6) Save CSV
# # # ============================================================
# # results_df = pd.DataFrame(rows)
# # csv_path = os.path.join(OUT_DIR, "synthetic_auroc_results.csv")
# # results_df.to_csv(csv_path, index=False)
# # print(f"Saved AUROC results to: {csv_path}")

# # print("Done.")

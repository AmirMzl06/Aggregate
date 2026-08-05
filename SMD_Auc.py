import sys
import os
import re
import gc
import json
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors

if "cebra" in sys.modules:
    del sys.modules["cebra"]

from utils.constants import CEBRA_DIR

sys.path.insert(0, str(CEBRA_DIR))
import cebra
import cebra.attribution
from cebra import CEBRA

from utils.min_distance import min_l2_distance


# ============================================================
# Config
# ============================================================
DATA_ROOT = "/data/SMD"
OUTPUT_ROOT = "./outputs_smd_feature_eval"
IMAGES_ROOT = "./images_smd_feature_eval"

OUTPUT_DIM = 16
BATCH_SIZE = 1024
ATTR_BATCH_SIZE = 128
MAX_ITERATIONS = 2500

RANDOM_SEED = 42
SAVE_MODELS = True
SAVE_SUMMARY_CSV = True
SAVE_INTERVAL_CSV = True
SAVE_PLOTS = False

DEVICE = "cuda_if_available"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(IMAGES_ROOT, exist_ok=True)


# ============================================================
# IO helpers
# ============================================================
def list_txt_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.endswith(".txt")])


def load_matrix(path: str) -> np.ndarray:
    try:
        x = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except Exception:
        x = np.loadtxt(path, dtype=np.float32)

    if x.ndim == 1:
        x = x[:, None]
    return x.astype(np.float32)


def load_binary_labels(path: str) -> np.ndarray:
    try:
        y = np.loadtxt(path, dtype=np.float32)
    except Exception:
        y = np.loadtxt(path, delimiter=",", dtype=np.float32)
    y = np.asarray(y).reshape(-1)
    return (y > 0).astype(np.int64)


def discover_machines(data_root: str):
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")
    test_label_dir = os.path.join(data_root, "test_label")
    interp_dir = os.path.join(data_root, "interpretation_label")

    train_files = set(list_txt_files(train_dir))
    test_files = set(list_txt_files(test_dir))
    tl_files = set(list_txt_files(test_label_dir))
    itp_files = set(list_txt_files(interp_dir))

    common = sorted(train_files & test_files & tl_files & itp_files)
    machines = []

    for fn in common:
        name = os.path.splitext(fn)[0]
        machines.append(
            {
                "name": name,
                "train_path": os.path.join(train_dir, fn),
                "test_path": os.path.join(test_dir, fn),
                "test_label_path": os.path.join(test_label_dir, fn),
                "interpretation_path": os.path.join(interp_dir, fn),
            }
        )
    return machines


# ============================================================
# interpretation_label parsing
# ============================================================
def parse_interpretation_label(path: str) -> List[Tuple[int, int, List[int]]]:
    """
    Example:
        15849-16368:1,9,10,12,13,14,15
    """
    out = []
    if not os.path.isfile(path):
        return out

    line_re = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*:\s*([0-9,\s]+)\s*$")
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            m = line_re.match(raw)
            if not m:
                continue
            s = int(m.group(1))
            e = int(m.group(2))
            feats = [int(x) for x in re.split(r"\s*,\s*", m.group(3).strip()) if x != ""]
            out.append((s, e, feats))
    return out


def normalize_feature_indices(raw_indices: List[int], n_features: int) -> List[int]:
    if len(raw_indices) == 0:
        return []

    idx = sorted(set(int(x) for x in raw_indices))

    if 0 in idx:
        out = idx
    elif min(idx) >= 1 and max(idx) <= n_features:
        out = [x - 1 for x in idx]
    elif min(idx) >= 0 and max(idx) < n_features:
        out = idx
    else:
        cand = [x - 1 for x in idx if 1 <= x <= n_features]
        out = cand if len(cand) > 0 else [x for x in idx if 0 <= x < n_features]

    out = sorted(set([i for i in out if 0 <= i < n_features]))
    return out


def gt_binary_vector(raw_indices: List[int], n_features: int) -> np.ndarray:
    gt = np.zeros(n_features, dtype=np.int64)
    idx = normalize_feature_indices(raw_indices, n_features)
    if len(idx) > 0:
        gt[idx] = 1
    return gt


def align_interval_to_test_labels(
    start: int,
    end: int,
    test_label: np.ndarray,
) -> Tuple[int, int, float, int]:
    """
    Returns:
        aligned_start, aligned_end, anomaly_coverage, chosen_offset
    """
    candidates = []
    for offset in (0, 1):
        s = start - offset
        e = end - offset
        if s < 0 or e >= len(test_label) or s > e:
            continue
        seg = test_label[s : e + 1]
        coverage = float(seg.mean()) if len(seg) > 0 else 0.0
        candidates.append((coverage, s, e, offset))

    if len(candidates) == 0:
        raise ValueError(f"Cannot align interval [{start}, {end}] with test_label length={len(test_label)}")

    candidates.sort(key=lambda x: (x[0], -x[3]), reverse=True)
    coverage, s, e, offset = candidates[0]
    return s, e, coverage, offset


# ============================================================
# CEBRA helpers
# ============================================================
def time_labels(n_steps: int) -> np.ndarray:
    if n_steps <= 1:
        return np.zeros((n_steps, 1), dtype=np.float32)
    return np.linspace(0.0, 1.0, n_steps, dtype=np.float32).reshape(-1, 1)


def infer_adv_epsilon(train_x: np.ndarray, sample_size: int = 2048) -> float:
    """
    Conservative epsilon estimate from local neighbor distance.
    Uses your min_l2_distance utility on a subset if possible.
    """
    n = len(train_x)
    if n < 2:
        return 1e-6

    ss = min(sample_size, n)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(n, size=ss, replace=False)
    xs = train_x[idx].astype(np.float32)

    try:
        xs_t = torch.from_numpy(xs)
        eps = float(min_l2_distance(xs_t)) / 2.0
        return max(eps, 1e-6)
    except Exception:
        if len(xs) < 2:
            return max(float(np.std(train_x)) * 0.05, 1e-6)

        nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
        nn.fit(xs)
        dists, _ = nn.kneighbors(xs, return_distance=True)
        nn_dist = dists[:, 1]
        eps = 0.5 * float(np.median(nn_dist))
        return max(eps, 1e-6)


def build_cebra_model(
    adv: bool,
    adv_epsilon: float,
    output_dim: int = OUTPUT_DIM,
    batch_size: int = BATCH_SIZE,
    max_iterations: int = MAX_ITERATIONS,
):
    return CEBRA(
        batch_size=batch_size,
        temperature=0.4,
        model_architecture="offset10-model",
        time_offsets=10,
        max_iterations=max_iterations,
        output_dimension=output_dim,
        verbose=True,
        training_mode="adversarial" if adv else "clean",
        adv_alpha=(adv_epsilon / 5.0) if adv else 0.0,
        adv_epsilon=adv_epsilon if adv else 0.0,
        adv_steps=10 if adv else 0,
        attack_norm="linf",
        num_hidden_units=32,
        device=DEVICE,
    )


def train_cebra(train_x: np.ndarray, adv: bool):
    adv_epsilon = infer_adv_epsilon(train_x) if adv else 0.0
    model = build_cebra_model(
        adv=adv,
        adv_epsilon=adv_epsilon,
        output_dim=OUTPUT_DIM,
        batch_size=BATCH_SIZE,
        max_iterations=MAX_ITERATIONS,
    )
    train_t = time_labels(len(train_x))
    model.fit(train_x.astype(np.float32), train_t.astype(np.float32))
    return model, adv_epsilon


def get_trained_model(cebra_model):
    trained = cebra_model.solver_.model
    trained = trained.to("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(trained, "split_outputs"):
        trained.split_outputs = False
    trained.eval()
    return trained


def reduce_attr_to_matrix(attr_tensor, n_features: int) -> np.ndarray:
    if torch.is_tensor(attr_tensor):
        attr = attr_tensor.detach().cpu()
    else:
        attr = torch.as_tensor(attr_tensor).cpu()

    attr = torch.abs(attr)

    if attr.ndim == 3:
        attr_2d = attr.mean(dim=0)
    elif attr.ndim == 2:
        attr_2d = attr
    elif attr.ndim == 1:
        attr_2d = attr[None, :]
    else:
        raise ValueError(f"Unsupported attribution shape: {tuple(attr.shape)}")

    if attr_2d.shape[1] == n_features:
        mat = attr_2d
    elif attr_2d.shape[0] == n_features:
        mat = attr_2d.T
    else:
        raise ValueError(
            f"Cannot identify feature axis. Reduced attribution shape={tuple(attr_2d.shape)}, n_features={n_features}"
        )

    return mat.numpy().astype(np.float32)


def score_vector_from_attr(attr_tensor, n_features: int) -> np.ndarray:
    mat = reduce_attr_to_matrix(attr_tensor, n_features)
    return np.abs(mat).mean(axis=0).astype(np.float32)


def compute_attr_scores(cebra_model, x_np: np.ndarray, n_features: int, attr_batch_size: int) -> Dict[str, np.ndarray]:
    trained_model = get_trained_model(cebra_model)
    device = next(trained_model.parameters()).device

    x_t = torch.from_numpy(x_np.astype(np.float32)).to(device)
    x_t.requires_grad_(True)

    output_dim = int(getattr(trained_model, "num_output", OUTPUT_DIM))

    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=trained_model,
        input_data=x_t,
        output_dimension=output_dim,
    )

    result = method.compute_attribution_map(batch_size=min(attr_batch_size, len(x_np)))

    out = {}
    if "jf" in result:
        out["jf"] = score_vector_from_attr(result["jf"], n_features)

    if "jf-inv-svd" in result:
        out["jf-inv-svd"] = score_vector_from_attr(result["jf-inv-svd"], n_features)
    elif "jf-inv" in result:
        out["jf-inv"] = score_vector_from_attr(result["jf-inv"], n_features)

    return out


# ============================================================
# Metrics
# ============================================================
def evaluate_feature_prediction(scores: np.ndarray, gt: np.ndarray, k: Optional[int] = None) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt, dtype=np.int64).reshape(-1)

    assert scores.shape[0] == gt.shape[0], (scores.shape, gt.shape)

    pos = int(gt.sum())
    n = len(gt)

    if k is None:
        k = pos if pos > 0 else 1
    k = int(max(1, min(k, n)))

    order = np.argsort(scores)[::-1]
    topk = order[:k]

    pred = np.zeros_like(gt)
    pred[topk] = 1

    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2.0 * precision * recall / max(precision + recall, 1e-12)) if (precision + recall) > 0 else 0.0
    iou = tp / max(tp + fp + fn, 1)

    roc_auc = np.nan
    pr_auc = np.nan
    if len(np.unique(gt)) > 1:
        try:
            roc_auc = float(roc_auc_score(gt, scores))
        except Exception:
            roc_auc = np.nan
    if pos > 0:
        try:
            pr_auc = float(average_precision_score(gt, scores))
        except Exception:
            pr_auc = np.nan

    return {
        "k": float(k),
        "gt_size": float(pos),
        "precision_at_k": float(precision),
        "recall_at_k": float(recall),
        "f1_at_k": float(f1),
        "iou_at_k": float(iou),
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else np.nan,
        "pr_auc": float(pr_auc) if not np.isnan(pr_auc) else np.nan,
    }


def format_mean_std(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "nan"
    return f"{x.mean():.4f} ± {x.std(ddof=0):.4f}"


# ============================================================
# Optional plot
# ============================================================
def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if np.isclose(mx, mn):
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def save_heatmap(mat: np.ndarray, save_path: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(minmax_normalize(mat), aspect="auto", cmap="cividis")
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, shrink=0.9, label="Normalized |importance|")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main per machine
# ============================================================
def evaluate_one_machine(machine: Dict[str, str]) -> List[Dict]:
    name = machine["name"]
    print("\n" + "=" * 120)
    print(f"Machine: {name}")
    print("=" * 120)

    x_train = load_matrix(machine["train_path"])
    x_test = load_matrix(machine["test_path"])
    test_label = load_binary_labels(machine["test_label_path"])
    interp = parse_interpretation_label(machine["interpretation_path"])

    n_features = x_train.shape[1]
    if x_test.shape[1] != n_features:
        raise ValueError(f"[{name}] train/test feature mismatch: {x_train.shape} vs {x_test.shape}")

    if len(interp) == 0:
        warnings.warn(f"[{name}] No interpretation labels found. Skipping.")
        return []

    eval_intervals = []
    for (s_raw, e_raw, feat_raw) in interp:
        s, e, coverage, offset = align_interval_to_test_labels(s_raw, e_raw, test_label)
        gt = gt_binary_vector(feat_raw, n_features)
        gt_idx = np.where(gt == 1)[0].tolist()

        if len(gt_idx) == 0:
            continue

        eval_intervals.append(
            {
                "start_raw": s_raw,
                "end_raw": e_raw,
                "start": s,
                "end": e,
                "coverage": coverage,
                "offset": offset,
                "gt": gt,
                "gt_idx": gt_idx,
            }
        )

    if len(eval_intervals) == 0:
        warnings.warn(f"[{name}] No valid intervals after parsing alignment.")
        return []

    machine_out = os.path.join(OUTPUT_ROOT, name)
    machine_img = os.path.join(IMAGES_ROOT, name)
    os.makedirs(machine_out, exist_ok=True)
    os.makedirs(machine_img, exist_ok=True)

    all_rows = []

    for adv in [False, True]:
        model_name = "ACORN" if adv else "CEBRA"
        print(f"\n--- Training {model_name} ---")

        cebra_model, adv_epsilon = train_cebra(x_train, adv=adv)

        if SAVE_MODELS:
            model_path = os.path.join(machine_out, f"{name}_{model_name}.pth")
            cebra_model.save(model_path)
            print(f"Saved model -> {model_path}")

        per_interval_rows = []
        store_scores = {}
        store_gts = {}

        for interval_id, item in enumerate(eval_intervals):
            s, e = item["start"], item["end"]
            gt = item["gt"]
            gt_size = int(gt.sum())

            x_seg = x_test[s : e + 1].astype(np.float32)
            if len(x_seg) < 2:
                continue

            try:
                score_dict = compute_attr_scores(
                    cebra_model=cebra_model,
                    x_np=x_seg,
                    n_features=n_features,
                    attr_batch_size=ATTR_BATCH_SIZE,
                )
            except Exception as ex:
                warnings.warn(f"[{name} | {model_name}] Attribution failed on interval {s}-{e}: {ex}")
                continue

            for attr_type, scores in score_dict.items():
                metrics = evaluate_feature_prediction(scores=scores, gt=gt, k=gt_size)

                row = {
                    "dataset": name,
                    "model": model_name,
                    "attr_type": attr_type,
                    "interval_id": int(interval_id),
                    "interval_start": int(s),
                    "interval_end": int(e),
                    "raw_start": int(item["start_raw"]),
                    "raw_end": int(item["end_raw"]),
                    "coverage": float(item["coverage"]),
                    "offset_used": int(item["offset"]),
                    "gt_size": int(gt_size),
                    "n_features": int(n_features),
                    "adv": bool(adv),
                    "adv_epsilon": float(adv_epsilon),
                    **metrics,
                    "gt_indices": json.dumps([int(x) for x in item["gt_idx"]]),
                    "top_pred_indices": json.dumps([int(x) for x in np.argsort(scores)[::-1][:gt_size].tolist()]),
                }
                per_interval_rows.append(row)

                if attr_type not in store_scores:
                    store_scores[attr_type] = []
                    store_gts[attr_type] = []

                store_scores[attr_type].append(np.asarray(scores, dtype=np.float32))
                store_gts[attr_type].append(np.asarray(gt, dtype=np.int64))

        if len(per_interval_rows) == 0:
            warnings.warn(f"[{name} | {model_name}] No interval-level results.")
            cleanup_cuda(cebra_model)
            continue

        df_interval = pd.DataFrame(per_interval_rows)
        if SAVE_INTERVAL_CSV:
            df_interval.to_csv(os.path.join(machine_out, f"{name}_{model_name}_per_interval.csv"), index=False)

        summary_rows = []
        for attr_type in sorted(df_interval["attr_type"].unique().tolist()):
            sub = df_interval[df_interval["attr_type"] == attr_type].copy()

            summary = {
                "dataset": name,
                "model": model_name,
                "attr_type": attr_type,
                "n_intervals": int(len(sub)),
                "macro_precision_at_k": float(sub["precision_at_k"].mean()),
                "macro_recall_at_k": float(sub["recall_at_k"].mean()),
                "macro_f1_at_k": float(sub["f1_at_k"].mean()),
                "macro_iou_at_k": float(sub["iou_at_k"].mean()),
                "macro_roc_auc": float(sub["roc_auc"].mean()),
                "macro_pr_auc": float(sub["pr_auc"].mean()),
                "mean_gt_size": float(sub["gt_size"].mean()),
                "mean_k": float(sub["k"].mean()),
                "adv": bool(adv),
                "adv_epsilon": float(adv_epsilon),
            }

            if len(store_scores.get(attr_type, [])) > 0:
                flat_scores = np.concatenate(store_scores[attr_type], axis=0)
                flat_gt = np.concatenate(store_gts[attr_type], axis=0)
                micro = evaluate_feature_prediction(scores=flat_scores, gt=flat_gt, k=int(flat_gt.sum()))
                summary["micro_precision_at_k"] = micro["precision_at_k"]
                summary["micro_recall_at_k"] = micro["recall_at_k"]
                summary["micro_f1_at_k"] = micro["f1_at_k"]
                summary["micro_iou_at_k"] = micro["iou_at_k"]
                summary["micro_roc_auc"] = micro["roc_auc"]
                summary["micro_pr_auc"] = micro["pr_auc"]
            else:
                summary["micro_precision_at_k"] = np.nan
                summary["micro_recall_at_k"] = np.nan
                summary["micro_f1_at_k"] = np.nan
                summary["micro_iou_at_k"] = np.nan
                summary["micro_roc_auc"] = np.nan
                summary["micro_pr_auc"] = np.nan

            summary_rows.append(summary)

        df_summary = pd.DataFrame(summary_rows)
        if SAVE_SUMMARY_CSV:
            df_summary.to_csv(os.path.join(machine_out, f"{name}_{model_name}_summary.csv"), index=False)

        # Print after each dataset/model
        print(f"\n[{name} | {model_name}] Summary")
        for _, row in df_summary.iterrows():
            print(
                f"  {row['attr_type']:>9} | "
                f"macro ROC-AUC={row['macro_roc_auc']:.4f} | "
                f"macro PR-AUC={row['macro_pr_auc']:.4f} | "
                f"micro ROC-AUC={row['micro_roc_auc']:.4f} | "
                f"micro PR-AUC={row['micro_pr_auc']:.4f} | "
                f"P@k={row['macro_precision_at_k']:.4f} | "
                f"R@k={row['macro_recall_at_k']:.4f} | "
                f"F1={row['macro_f1_at_k']:.4f}"
            )

        all_rows.extend(summary_rows)

        if SAVE_PLOTS:
            try:
                first_item = eval_intervals[0]
                x_seg = x_test[first_item["start"] : first_item["end"] + 1].astype(np.float32)
                score_dict = compute_attr_scores(
                    cebra_model=cebra_model,
                    x_np=x_seg,
                    n_features=n_features,
                    attr_batch_size=ATTR_BATCH_SIZE,
                )
                for attr_type, scores in score_dict.items():
                    fig_path = os.path.join(machine_img, f"{name}_{model_name}_{attr_type}_feature_scores.png")
                    save_heatmap(scores[None, :], fig_path, f"{name} | {model_name} | {attr_type}")
            except Exception as ex:
                warnings.warn(f"[{name} | {model_name}] plot failed: {ex}")

        cleanup_cuda(cebra_model)

    return all_rows


# ============================================================
# Global summary
# ============================================================
def build_global_summary(all_rows: List[Dict]):
    if len(all_rows) == 0:
        print("No results to summarize.")
        return

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(OUTPUT_ROOT, "all_machines_summary.csv"), index=False)

    global_summary = (
        df_all.groupby(["model", "attr_type"], as_index=False)
        .agg(
            n_machines=("dataset", "nunique"),
            n_rows=("dataset", "size"),
            macro_precision_at_k=("macro_precision_at_k", "mean"),
            macro_recall_at_k=("macro_recall_at_k", "mean"),
            macro_f1_at_k=("macro_f1_at_k", "mean"),
            macro_iou_at_k=("macro_iou_at_k", "mean"),
            macro_roc_auc=("macro_roc_auc", "mean"),
            macro_pr_auc=("macro_pr_auc", "mean"),
            micro_precision_at_k=("micro_precision_at_k", "mean"),
            micro_recall_at_k=("micro_recall_at_k", "mean"),
            micro_f1_at_k=("micro_f1_at_k", "mean"),
            micro_iou_at_k=("micro_iou_at_k", "mean"),
            micro_roc_auc=("micro_roc_auc", "mean"),
            micro_pr_auc=("micro_pr_auc", "mean"),
        )
    )

    global_summary.to_csv(os.path.join(OUTPUT_ROOT, "global_summary.csv"), index=False)

    print("\n" + "=" * 120)
    print("GLOBAL SUMMARY")
    print("=" * 120)
    print(global_summary.to_string(index=False))


# ============================================================
# Main
# ============================================================
def main():
    machines = discover_machines(DATA_ROOT)
    if len(machines) == 0:
        raise FileNotFoundError(
            f"No matched machine files found under {DATA_ROOT}. "
            f"Expected identical .txt filenames in train/test/test_label/interpretation_label."
        )

    all_rows = []
    for machine in machines:
        try:
            rows = evaluate_one_machine(machine)
            all_rows.extend(rows)
        except Exception as ex:
            warnings.warn(f"[{machine['name']}] Failed: {ex}")
            continue

    build_global_summary(all_rows)


if __name__ == "__main__":
    main()

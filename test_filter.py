import os
import gc
import json
import numpy as np
import pandas as pd
import torch
import scipy.io as sio
import matplotlib.pyplot as plt

from utils.constants import CEBRA_DIR
from utils.min_distance import min_l2_distance

import sys
sys.path.insert(0, str(CEBRA_DIR))

import cebra
import cebra.attribution
from cebra import CEBRA


# =====================================================
# Config
# =====================================================
DATA_PATH = "./data/spk/M021519_spk.mat"
BHV_PATH = "./data/behav/M021519_trialtype.csv"

OUT_DIR = "./outputs"
IMG_DIR = "./image"
PER_TRIAL_DIR = os.path.join(OUT_DIR, "per_trial")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(PER_TRIAL_DIR, exist_ok=True)

BATCH_SIZE = 256
MAX_ITER = 5000
OUTPUT_DIM = 16

PRE_MS = 500
POST_MS = 1000
BIN_MS = 10

EVENT_NAME = "stim_on"
TOP_K = 10
RANDOM_SEED = 42

USE_ONLY_2AFC = True
USE_ONLY_COMPLETED = True
USE_ONLY_CORRECT_BY_PROBABILITY = True
MIN_DELTA_P = None  
MAX_TRIALS = None

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# Load data
# =====================================================
mat = sio.loadmat(DATA_PATH, simplify_cells=True, squeeze_me=True)
unit = mat["unit"]
t_evt = mat["t_evt"]

print("neurons:", len(unit))
print("events:", t_evt.keys())

bhv = None
if os.path.exists(BHV_PATH):
    try:
        bhv = pd.read_csv(BHV_PATH)
        print("behavior file loaded:", BHV_PATH)
        print("bhv shape:", bhv.shape)
    except Exception as e:
        print("Could not load behavior file:", e)
        bhv = None


# =====================================================
# Helpers
# =====================================================
def make_trial_matrix(unit, event_time, pre_ms, post_ms, bin_ms):
    n_neurons = len(unit)
    n_bins = int((pre_ms + post_ms) / bin_ms)
    X = np.zeros((n_bins, n_neurons), dtype=np.float32)

    start = event_time - pre_ms / 1000.0
    end = event_time + post_ms / 1000.0

    for n in range(n_neurons):
        spikes = unit[n]["timestamps"]
        spikes = np.asarray(spikes, dtype=np.float32).reshape(-1)

        spikes = spikes[(spikes >= start) & (spikes <= end)]
        spikes_ms = (spikes - event_time) * 1000.0
        bins = ((spikes_ms + pre_ms) / bin_ms).astype(int)
        bins = bins[(bins >= 0) & (bins < n_bins)]

        for b in bins:
            X[b, n] += 1.0

    return X


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
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    else:
        arr = np.asarray(arr)

    if arr.ndim == 3:
        arr = np.abs(arr).mean(axis=0)
    else:
        arr = np.abs(arr)

    plt.figure(figsize=(10, 6))
    plt.imshow(arr, aspect="auto", cmap="viridis")
    plt.colorbar(label="absolute attribution")
    plt.xlabel("Neuron")
    plt.ylabel("Latent dimension")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def get_event_times_and_candidate_trials():
    event_times = np.asarray(t_evt[EVENT_NAME], dtype=np.float32).reshape(-1)
    event_times = event_times[np.isfinite(event_times)]

    candidate_trial_ids = np.arange(len(event_times), dtype=int)

    if bhv is not None:
        candidate_trial_ids = candidate_trial_ids[candidate_trial_ids < len(bhv)]

        if USE_ONLY_2AFC and "task" in bhv.columns:
            candidate_trial_ids = candidate_trial_ids[
                bhv.iloc[candidate_trial_ids]["task"].astype(str).to_numpy() == "2AFC"
            ]

        if USE_ONLY_COMPLETED and "brk" in bhv.columns:
            candidate_trial_ids = candidate_trial_ids[
                bhv.iloc[candidate_trial_ids]["brk"].to_numpy() == 0
            ]

        if USE_ONLY_CORRECT_BY_PROBABILITY and {
            "probaL_2AFC", "probaR_2AFC", "chosenproba_2AFC"
        }.issubset(bhv.columns):
            sub = bhv.iloc[candidate_trial_ids].copy()

            probaL = pd.to_numeric(sub["probaL_2AFC"], errors="coerce").to_numpy(dtype=float)
            probaR = pd.to_numeric(sub["probaR_2AFC"], errors="coerce").to_numpy(dtype=float)
            chosen = pd.to_numeric(sub["chosenproba_2AFC"], errors="coerce").to_numpy(dtype=float)
            best = np.maximum(probaL, probaR)

            mask = np.isfinite(probaL) & np.isfinite(probaR) & np.isfinite(chosen) & np.isfinite(best)
            mask &= np.isclose(chosen, best, equal_nan=False)

            if MIN_DELTA_P is not None:
                delta = np.abs(probaL - probaR)
                mask &= delta >= float(MIN_DELTA_P)

            candidate_trial_ids = candidate_trial_ids[mask]

        print("filtered candidate trials:", len(candidate_trial_ids))
    else:
        print("using all valid stim_on trials.")

    if MAX_TRIALS is not None:
        candidate_trial_ids = candidate_trial_ids[:MAX_TRIALS]
        print("capped candidate trials:", len(candidate_trial_ids))

    return event_times, candidate_trial_ids


def train_acorn_on_trial(X_trial):
    x_torch = torch.tensor(X_trial, dtype=torch.float32)
    eps = float(min_l2_distance(x_torch)) / 2.0
    eps = max(eps, 1e-6)

    print("Training ACORN | epsilon:", eps)

    train_batch_size = min(BATCH_SIZE, len(X_trial))

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

    labels = np.arange(len(X_trial), dtype=np.float32)
    model.fit(X_trial.astype(np.float32), labels)

    return model, eps


def compute_trial_attribution(model, X_trial, trial_idx):
    encoder = model.solver_.model.to(device)
    if hasattr(encoder, "split_outputs"):
        encoder.split_outputs = False
    encoder.eval()

    x_tensor = torch.tensor(
        X_trial,
        dtype=torch.float32,
        device=device,
        requires_grad=True
    )

    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=encoder,
        input_data=x_tensor,
        output_dimension=OUTPUT_DIM,
    )

    result = method.compute_attribution_map(batch_size=min(128, len(X_trial)))
    print(f"[trial {trial_idx}] attribution keys:", result.keys())

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


def plot_topk_counts(counts, unit):
    counts = np.asarray(counts, dtype=int)
    order = np.argsort(counts)[::-1]

    labels = []
    for idx in order:
        area = unit[idx]["area"]
        ch = unit[idx]["ch"]
        labels.append(f"{idx}\nch:{ch}\n{area}")

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar(np.arange(len(order)), counts[order])
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel(f"Top-{TOP_K} frequency")
    ax.set_xlabel("Neuron index / channel / area")
    ax.set_title(f"ACORN top-{TOP_K} frequency across filtered trials")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "ACORN_top10_frequency.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# Select ALL filtered trials
# =====================================================
stim_times, candidate_trial_ids = get_event_times_and_candidate_trials()

if len(candidate_trial_ids) == 0:
    raise RuntimeError("No candidate trials found.")

selected_trial_ids = np.sort(candidate_trial_ids)
print("selected trial IDs count:", len(selected_trial_ids))
print("selected trial IDs preview:", selected_trial_ids[:20].tolist())


# =====================================================
# Main loop
# =====================================================
n_neurons = len(unit)
topk_counts = np.zeros(n_neurons, dtype=np.int32)

trial_rows = []
selected_meta_rows = []

for step, trial_idx in enumerate(selected_trial_ids, start=1):
    print("\n" + "=" * 90)
    print(f"Processing trial {step}/{len(selected_trial_ids)} | trial_id = {trial_idx}")
    print("=" * 90)

    event_time = float(stim_times[trial_idx])

    X_trial = make_trial_matrix(
        unit=unit,
        event_time=event_time,
        pre_ms=PRE_MS,
        post_ms=POST_MS,
        bin_ms=BIN_MS,
    )

    X_trial, mu, sigma = normalize_trial(X_trial)

    acorn_model, eps = train_acorn_on_trial(X_trial)

    jf_map, jfinv_map = compute_trial_attribution(acorn_model, X_trial, trial_idx)

    neuron_scores = np.abs(jf_map).mean(axis=0)
    top10 = np.argsort(neuron_scores)[::-1][:TOP_K]

    topk_counts[top10] += 1

    # save per-trial results
    np.save(os.path.join(PER_TRIAL_DIR, f"trial_{trial_idx}_jf.npy"), jf_map)
    np.save(os.path.join(PER_TRIAL_DIR, f"trial_{trial_idx}_jfinv.npy"), jfinv_map)
    np.save(os.path.join(PER_TRIAL_DIR, f"trial_{trial_idx}_neuron_scores.npy"), neuron_scores)
    np.save(os.path.join(PER_TRIAL_DIR, f"trial_{trial_idx}_top10.npy"), top10)

    trial_rows.append({
        "trial_id": int(trial_idx),
        "event_time": float(event_time),
        "top10_neurons": json.dumps([int(x) for x in top10.tolist()]),
        "top10_scores": json.dumps([float(neuron_scores[x]) for x in top10.tolist()]),
        "eps": float(eps),
    })

    if bhv is not None and trial_idx < len(bhv):
        row = bhv.iloc[int(trial_idx)]
        meta = {
            "trial_id": int(trial_idx),
            "task": row.get("task", None),
            "brk": row.get("brk", None),
            "rew": row.get("rew", None),
            "resp_loc": row.get("resp_loc", None),
            "flavorL_2AFC": row.get("flavorL_2AFC", None),
            "flavorR_2AFC": row.get("flavorR_2AFC", None),
            "probaL_2AFC": row.get("probaL_2AFC", None),
            "probaR_2AFC": row.get("probaR_2AFC", None),
            "chosenflavor_2AFC": row.get("chosenflavor_2AFC", None),
            "chosenproba_2AFC": row.get("chosenproba_2AFC", None),
            "unchosenproba_2AFC": row.get("unchosenproba_2AFC", None),
            "chosenside_2AFC": row.get("chosenside_2AFC", None),
        }
        selected_meta_rows.append(meta)

    print(f"trial {trial_idx} top10 neurons:", top10.tolist())

    del acorn_model, X_trial, jf_map, jfinv_map, neuron_scores, top10
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# =====================================================
# Save summaries
# =====================================================
summary_rows = []
for n in range(n_neurons):
    summary_rows.append({
        "neuron_index": int(n),
        "session": unit[n]["session"],
        "channel": unit[n]["ch"],
        "cluster_id": unit[n]["clust_id"],
        "area": unit[n]["area"],
        "num_spikes": int(len(unit[n]["timestamps"])),
        "top10_frequency": int(topk_counts[n]),
        "top10_frequency_ratio": float(topk_counts[n] / max(len(selected_trial_ids), 1)),
    })

summary_df = pd.DataFrame(summary_rows).sort_values(
    by=["top10_frequency", "num_spikes"],
    ascending=[False, False]
)

summary_csv = os.path.join(OUT_DIR, "ACORN_neuron_frequency_summary.csv")
summary_df.to_csv(summary_csv, index=False)

trial_df = pd.DataFrame(trial_rows)
trial_csv = os.path.join(OUT_DIR, "ACORN_selected_trials_top10.csv")
trial_df.to_csv(trial_csv, index=False)

if len(selected_meta_rows) > 0:
    meta_df = pd.DataFrame(selected_meta_rows)
    meta_csv = os.path.join(OUT_DIR, "ACORN_selected_trials_behavior.csv")
    meta_df.to_csv(meta_csv, index=False)
    print("saved behavior labels to:", meta_csv)

plot_topk_counts(topk_counts, unit)

print("\nDONE")
print("selected trials used:", len(selected_trial_ids))
print("saved trial top10 csv to:", trial_csv)
print("saved neuron frequency summary to:", summary_csv)
print("saved frequency plot to:", os.path.join(IMG_DIR, "ACORN_top10_frequency.png"))

print("\nTop neurons by frequency:")
print(summary_df.head(15).to_string(index=False))



# import os
# import gc
# import json
# import numpy as np
# import pandas as pd
# import torch
# import scipy.io as sio
# import matplotlib.pyplot as plt

# from utils.constants import CEBRA_DIR
# from utils.min_distance import min_l2_distance

# import sys
# sys.path.insert(0, str(CEBRA_DIR))

# import cebra
# import cebra.attribution
# from cebra import CEBRA


# # =====================================================
# # Config
# # =====================================================
# DATA_PATH = "./data/spk/M021519_spk.mat"
# BHV_PATH = "./data/behav/M021519_trialtype.csv"

# OUT_DIR = "./outputs"
# IMG_DIR = "./image"
# PER_TRIAL_DIR = os.path.join(OUT_DIR, "per_trial")

# os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(IMG_DIR, exist_ok=True)
# os.makedirs(PER_TRIAL_DIR, exist_ok=True)

# BATCH_SIZE = 256
# MAX_ITER = 2500
# OUTPUT_DIM = 16

# PRE_MS = 500
# POST_MS = 1000
# BIN_MS = 10

# EVENT_NAME = "stim_on"
# N_RANDOM_TRIALS = 20
# TOP_K = 10
# RANDOM_SEED = 42

# # Filters:
# USE_ONLY_2AFC = True
# USE_ONLY_COMPLETED = True
# USE_ONLY_CORRECT_BY_PROBABILITY = True
# MIN_DELTA_P = None  # set e.g. 20.0 to keep only trials with |P_L - P_R| >= 20

# torch.manual_seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # =====================================================
# # Load data
# # =====================================================
# mat = sio.loadmat(DATA_PATH, simplify_cells=True, squeeze_me=True)
# unit = mat["unit"]
# t_evt = mat["t_evt"]

# print("neurons:", len(unit))
# print("events:", t_evt.keys())

# bhv = None
# if os.path.exists(BHV_PATH):
#     try:
#         bhv = pd.read_csv(BHV_PATH)
#         print("behavior file loaded:", BHV_PATH)
#         print("bhv shape:", bhv.shape)
#     except Exception as e:
#         print("Could not load behavior file:", e)
#         bhv = None


# # =====================================================
# # Helpers
# # =====================================================
# def make_trial_matrix(unit, event_time, pre_ms, post_ms, bin_ms):
#     """
#     Returns:
#         X_trial: [time_bins, neurons]
#     """
#     n_neurons = len(unit)
#     n_bins = int((pre_ms + post_ms) / bin_ms)
#     X = np.zeros((n_bins, n_neurons), dtype=np.float32)

#     start = event_time - pre_ms / 1000.0
#     end = event_time + post_ms / 1000.0

#     for n in range(n_neurons):
#         spikes = unit[n]["timestamps"]
#         spikes = np.asarray(spikes, dtype=np.float32).reshape(-1)

#         spikes = spikes[(spikes >= start) & (spikes <= end)]
#         spikes_ms = (spikes - event_time) * 1000.0
#         bins = ((spikes_ms + pre_ms) / bin_ms).astype(int)
#         bins = bins[(bins >= 0) & (bins < n_bins)]

#         for b in bins:
#             X[b, n] += 1.0

#     return X


# def normalize_trial(X):
#     """
#     z-score each trial neuron-wise.
#     X: [time_bins, neurons]
#     """
#     mu = X.mean(axis=0, keepdims=True)
#     sigma = X.std(axis=0, keepdims=True) + 1e-8
#     return ((X - mu) / sigma).astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)


# def reduce_attr_map(arr):
#     """
#     Convert attribution output to [latent_dim, neurons].
#     """
#     if torch.is_tensor(arr):
#         arr = arr.detach().cpu().numpy()
#     else:
#         arr = np.asarray(arr)

#     arr = np.abs(arr)

#     if arr.ndim == 3:
#         arr = arr.mean(axis=0)
#     elif arr.ndim == 2:
#         pass
#     elif arr.ndim == 1:
#         arr = arr[None, :]
#     else:
#         raise ValueError(f"Unsupported attribution shape: {arr.shape}")

#     return arr.astype(np.float32)


# def save_heatmap(arr, path, title):
#     if torch.is_tensor(arr):
#         arr = arr.detach().cpu().numpy()
#     else:
#         arr = np.asarray(arr)

#     if arr.ndim == 3:
#         arr = np.abs(arr).mean(axis=0)
#     else:
#         arr = np.abs(arr)

#     plt.figure(figsize=(10, 6))
#     plt.imshow(arr, aspect="auto", cmap="viridis")
#     plt.colorbar(label="absolute attribution")
#     plt.xlabel("Neuron")
#     plt.ylabel("Latent dimension")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()


# def get_event_times_and_candidate_trials():
#     """
#     Returns:
#         event_times: all valid stim_on times
#         candidate_trial_ids: indices that can be sampled
#     """
#     event_times = np.asarray(t_evt[EVENT_NAME], dtype=np.float32).reshape(-1)
#     event_times = event_times[np.isfinite(event_times)]

#     candidate_trial_ids = np.arange(len(event_times), dtype=int)

#     if bhv is not None:
#         candidate_trial_ids = candidate_trial_ids[candidate_trial_ids < len(bhv)]

#         if USE_ONLY_2AFC and "task" in bhv.columns:
#             candidate_trial_ids = candidate_trial_ids[
#                 bhv.iloc[candidate_trial_ids]["task"].astype(str).to_numpy() == "2AFC"
#             ]

#         if USE_ONLY_COMPLETED and "brk" in bhv.columns:
#             candidate_trial_ids = candidate_trial_ids[
#                 bhv.iloc[candidate_trial_ids]["brk"].to_numpy() == 0
#             ]

#         if USE_ONLY_CORRECT_BY_PROBABILITY and {
#             "probaL_2AFC", "probaR_2AFC", "chosenproba_2AFC"
#         }.issubset(bhv.columns):
#             sub = bhv.iloc[candidate_trial_ids].copy()

#             probaL = pd.to_numeric(sub["probaL_2AFC"], errors="coerce").to_numpy(dtype=float)
#             probaR = pd.to_numeric(sub["probaR_2AFC"], errors="coerce").to_numpy(dtype=float)
#             chosen = pd.to_numeric(sub["chosenproba_2AFC"], errors="coerce").to_numpy(dtype=float)

#             best = np.maximum(probaL, probaR)

#             mask = np.isfinite(probaL) & np.isfinite(probaR) & np.isfinite(chosen) & np.isfinite(best)
#             mask &= np.isclose(chosen, best, equal_nan=False)

#             if MIN_DELTA_P is not None:
#                 delta = np.abs(probaL - probaR)
#                 mask &= delta >= float(MIN_DELTA_P)

#             candidate_trial_ids = candidate_trial_ids[mask]

#         print("filtered candidate trials:", len(candidate_trial_ids))
#     else:
#         print("using all valid stim_on trials.")

#     return event_times, candidate_trial_ids


# def train_acorn_on_trial(X_trial):
#     """
#     Train one ACORN model on one trial.
#     """
#     x_torch = torch.tensor(X_trial, dtype=torch.float32)
#     eps = float(min_l2_distance(x_torch)) / 2.0
#     eps = max(eps, 1e-6)

#     print("Training ACORN | epsilon:", eps)

#     train_batch_size = min(BATCH_SIZE, len(X_trial))

#     model = CEBRA(
#         batch_size=train_batch_size,
#         temperature=0.4,
#         model_architecture="offset10-model",
#         time_offsets=10,
#         max_iterations=MAX_ITER,
#         output_dimension=OUTPUT_DIM,
#         verbose=True,
#         training_mode="adversarial",
#         adv_alpha=eps / 5,
#         adv_epsilon=eps,
#         adv_steps=10,
#         attack_norm="linf",
#         num_hidden_units=32,
#         device="cuda_if_available",
#     )

#     labels = np.arange(len(X_trial), dtype=np.float32)
#     model.fit(X_trial.astype(np.float32), labels)

#     return model, eps


# def compute_trial_attribution(model, X_trial, trial_idx):
#     """
#     Compute JF and JF-inv for one trial.
#     """
#     encoder = model.solver_.model.to(device)
#     if hasattr(encoder, "split_outputs"):
#         encoder.split_outputs = False
#     encoder.eval()

#     x_tensor = torch.tensor(
#         X_trial,
#         dtype=torch.float32,
#         device=device,
#         requires_grad=True
#     )

#     method = cebra.attribution.init(
#         name="jacobian-based-batched",
#         model=encoder,
#         input_data=x_tensor,
#         output_dimension=OUTPUT_DIM,
#     )

#     result = method.compute_attribution_map(batch_size=min(128, len(X_trial)))
#     print(f"[trial {trial_idx}] attribution keys:", result.keys())

#     jf_key = "jf"
#     if "jf-inv-svd" in result:
#         jfinv_key = "jf-inv-svd"
#     elif "jf-inv-lsq" in result:
#         jfinv_key = "jf-inv-lsq"
#     elif "jf-inv" in result:
#         jfinv_key = "jf-inv"
#     else:
#         raise KeyError(f"No inverse attribution key found. Available: {list(result.keys())}")

#     jf = reduce_attr_map(result[jf_key])
#     jfinv = reduce_attr_map(result[jfinv_key])

#     del encoder, x_tensor, method, result
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()

#     return jf, jfinv


# def plot_topk_counts(counts, unit):
#     counts = np.asarray(counts, dtype=int)
#     order = np.argsort(counts)[::-1]

#     labels = []
#     for idx in order:
#         area = unit[idx]["area"]
#         ch = unit[idx]["ch"]
#         labels.append(f"{idx}\nch:{ch}\n{area}")

#     fig, ax = plt.subplots(figsize=(18, 7))
#     ax.bar(np.arange(len(order)), counts[order])
#     ax.set_xticks(np.arange(len(order)))
#     ax.set_xticklabels(labels, rotation=0, fontsize=8)
#     ax.set_ylabel(f"Top-{TOP_K} frequency")
#     ax.set_xlabel("Neuron index / channel / area")
#     ax.set_title(f"ACORN top-{TOP_K} frequency across selected trials")
#     ax.grid(axis="y", alpha=0.3)
#     fig.tight_layout()
#     fig.savefig(os.path.join(IMG_DIR, "ACORN_top10_frequency.png"), dpi=300, bbox_inches="tight")
#     plt.close(fig)


# # =====================================================
# # Select trials
# # =====================================================
# stim_times, candidate_trial_ids = get_event_times_and_candidate_trials()

# if len(candidate_trial_ids) == 0:
#     raise RuntimeError("No candidate trials found.")

# rng = np.random.default_rng(RANDOM_SEED)
# n_select = min(N_RANDOM_TRIALS, len(candidate_trial_ids))
# selected_trial_ids = rng.choice(candidate_trial_ids, size=n_select, replace=False)
# selected_trial_ids = np.sort(selected_trial_ids)

# print("selected trial IDs:", selected_trial_ids.tolist())


# # =====================================================
# # Main loop
# # =====================================================
# n_neurons = len(unit)
# topk_counts = np.zeros(n_neurons, dtype=np.int32)

# trial_rows = []
# selected_meta_rows = []

# for step, trial_idx in enumerate(selected_trial_ids, start=1):
#     print("\n" + "=" * 90)
#     print(f"Processing random trial {step}/{len(selected_trial_ids)} | trial_id = {trial_idx}")
#     print("=" * 90)

#     event_time = float(stim_times[trial_idx])

#     X_trial = make_trial_matrix(
#         unit=unit,
#         event_time=event_time,
#         pre_ms=PRE_MS,
#         post_ms=POST_MS,
#         bin_ms=BIN_MS,
#     )

#     X_trial, mu, sigma = normalize_trial(X_trial)

#     acorn_model, eps = train_acorn_on_trial(X_trial)

#     jf_map, jfinv_map = compute_trial_attribution(acorn_model, X_trial, trial_idx)

#     # use ONLY Jacobian for top-k counting
#     neuron_scores = np.abs(jf_map).mean(axis=0)
#     top10 = np.argsort(neuron_scores)[::-1][:TOP_K]

#     topk_counts[top10] += 1

#     # save per-trial results
#     np.save(os.path.join(PER_TRIAL_DIR, f"trial_{trial_idx}_jf.npy"), jf_map)
#     np.save(os.path.join(PER_TRIAL_DIR, f"trial_{trial_idx}_jfinv.npy"), jfinv_map)
#     np.save(os.path.join(PER_TRIAL_DIR, f"trial_{trial_idx}_neuron_scores.npy"), neuron_scores)
#     np.save(os.path.join(PER_TRIAL_DIR, f"trial_{trial_idx}_top10.npy"), top10)

#     trial_rows.append({
#         "trial_id": int(trial_idx),
#         "event_time": float(event_time),
#         "top10_neurons": json.dumps([int(x) for x in top10.tolist()]),
#         "top10_scores": json.dumps([float(neuron_scores[x]) for x in top10.tolist()]),
#         "eps": float(eps),
#     })

#     if bhv is not None and trial_idx < len(bhv):
#         row = bhv.iloc[int(trial_idx)]
#         meta = {
#             "trial_id": int(trial_idx),
#             "task": row.get("task", None),
#             "brk": row.get("brk", None),
#             "rew": row.get("rew", None),
#             "resp_loc": row.get("resp_loc", None),
#             "flavorL_2AFC": row.get("flavorL_2AFC", None),
#             "flavorR_2AFC": row.get("flavorR_2AFC", None),
#             "probaL_2AFC": row.get("probaL_2AFC", None),
#             "probaR_2AFC": row.get("probaR_2AFC", None),
#             "chosenflavor_2AFC": row.get("chosenflavor_2AFC", None),
#             "chosenproba_2AFC": row.get("chosenproba_2AFC", None),
#             "unchosenproba_2AFC": row.get("unchosenproba_2AFC", None),
#             "chosenside_2AFC": row.get("chosenside_2AFC", None),
#         }
#         selected_meta_rows.append(meta)

#     print(f"trial {trial_idx} top10 neurons:", top10.tolist())
#     del acorn_model, X_trial, jf_map, jfinv_map, neuron_scores, top10
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()



# # =====================================================
# # Save summaries
# # =====================================================
# summary_rows = []
# for n in range(n_neurons):
#     summary_rows.append({
#         "neuron_index": int(n),
#         "session": unit[n]["session"],
#         "channel": unit[n]["ch"],
#         "cluster_id": unit[n]["clust_id"],
#         "area": unit[n]["area"],
#         "num_spikes": int(len(unit[n]["timestamps"])),
#         "top10_frequency": int(topk_counts[n]),
#         "top10_frequency_ratio": float(topk_counts[n] / max(len(selected_trial_ids), 1)),
#     })

# summary_df = pd.DataFrame(summary_rows).sort_values(
#     by=["top10_frequency", "num_spikes"],
#     ascending=[False, False]
# )

# summary_csv = os.path.join(OUT_DIR, "ACORN_neuron_frequency_summary.csv")
# summary_df.to_csv(summary_csv, index=False)

# trial_df = pd.DataFrame(trial_rows)
# trial_csv = os.path.join(OUT_DIR, "ACORN_selected_trials_top10.csv")
# trial_df.to_csv(trial_csv, index=False)

# if len(selected_meta_rows) > 0:
#     meta_df = pd.DataFrame(selected_meta_rows)
#     meta_csv = os.path.join(OUT_DIR, "ACORN_selected_trials_behavior.csv")
#     meta_df.to_csv(meta_csv, index=False)
#     print("saved behavior labels to:", meta_csv)

# plot_topk_counts(topk_counts, unit)

# print("\nDONE")
# print("selected trials used:", len(selected_trial_ids))
# print("saved trial top10 csv to:", trial_csv)
# print("saved neuron frequency summary to:", summary_csv)
# print("saved frequency plot to:", os.path.join(IMG_DIR, "ACORN_top10_frequency.png"))

# print("\nTop neurons by frequency:")
# print(summary_df.head(15).to_string(index=False))

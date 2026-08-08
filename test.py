import os
import gc
import numpy as np
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
OUT_DIR = "./outputs"
IMG_DIR = "./image"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

BATCH_SIZE = 256
MAX_ITER = 2500
OUTPUT_DIM = 16

PRE_MS = 500
POST_MS = 1000
BIN_MS = 10

EVENT_NAME = "stim_on"   # alignment event
TRIAL_LIMIT = None       # e.g. 20 for debugging, None = all valid trials
N_RANDOM_TRIALS = 20 

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# Load data
# =====================================================
mat = sio.loadmat(DATA_PATH, simplify_cells=True, squeeze_me=True)
unit = mat["unit"]
t_evt = mat["t_evt"]

print("neurons:", len(unit))
print("events:", t_evt.keys())


# =====================================================
# Helpers
# =====================================================
def make_trial_matrix(unit, event_time, pre_ms, post_ms, bin_ms):
    """
    Returns:
        X_trial: [time_bins, neurons]
    """
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


def build_all_trials(unit, event_times, pre_ms, post_ms, bin_ms, trial_limit=None):
    """
    Returns:
        trials: [n_trials, time_bins, neurons]
        used_event_times: valid event times used
    """
    event_times = np.asarray(event_times, dtype=np.float32).reshape(-1)
    event_times = event_times[np.isfinite(event_times)]

    if trial_limit is not None:
        event_times = event_times[:trial_limit]

    trials = []
    used = []

    for evt in event_times:
        X_trial = make_trial_matrix(unit, evt, pre_ms, post_ms, bin_ms)
        trials.append(X_trial)
        used.append(evt)

    trials = np.stack(trials, axis=0).astype(np.float32)
    return trials, np.asarray(used, dtype=np.float32)


def normalize_trial(X):
    """
    z-score each trial neuron-wise.
    X: [time_bins, neurons]
    """
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    return ((X - mu) / sigma).astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)


def save_heatmap(arr, path, title):
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    else:
        arr = np.asarray(arr)

    # Average over sample dimension if present
    if arr.ndim == 3:
        arr = np.abs(arr).mean(axis=0)
    else:
        arr = np.abs(arr)

    plt.figure(figsize=(10, 6))
    plt.imshow(arr, aspect="auto")
    plt.colorbar(label="absolute attribution")
    plt.xlabel("Neuron")
    plt.ylabel("Latent dimension")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def reduce_attr_map(arr):
    """
    Convert attribution output to [latent_dim, neurons].
    """
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


def train_cebra_on_trial(X_trial, adv=False):
    """
    Train one model on one trial.
    """
    mode = "adversarial" if adv else "clean"

    x_torch = torch.tensor(X_trial, dtype=torch.float32)
    eps = float(min_l2_distance(x_torch)) / 2.0
    eps = max(eps, 1e-6)

    print("\nTraining", mode, "epsilon:", eps)

    model = CEBRA(
        batch_size=BATCH_SIZE,
        temperature=0.4,
        model_architecture="offset10-model",
        time_offsets=10,
        max_iterations=MAX_ITER,
        output_dimension=OUTPUT_DIM,
        verbose=True,
        training_mode="adversarial" if adv else "clean",
        adv_alpha=eps / 5 if adv else 0,
        adv_epsilon=eps if adv else 0,
        adv_steps=10 if adv else 0,
        attack_norm="linf",
        num_hidden_units=32,
        device="cuda_if_available",
    )

    labels = np.arange(len(X_trial), dtype=np.float32)
    model.fit(X_trial.astype(np.float32), labels)

    return model, eps


def compute_trial_attribution(model, X_trial, tag, trial_idx):
    """
    Compute JF and JF-inv for one trial.
    """
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
    print(f"[{tag} | trial {trial_idx}] attribution keys:", result.keys())

    jf_key = "jf"
    if "jf-inv-svd" in result:
        jfinv_key = "jf-inv-svd"
    elif "jf-inv-lsq" in result:
        jfinv_key = "jf-inv-lsq"
    elif "jf-inv" in result:
        jfinv_key = "jf-inv"
    else:
        raise KeyError(f"No inverse attribution key found. Available: {list(result.keys())}")

    jf = reduce_attr_map(result[jf_key])
    jfinv = reduce_attr_map(result[jfinv_key])

    # save per-trial raw arrays
    np.save(os.path.join(OUT_DIR, f"{tag}_trial{trial_idx}_jf.npy"), jf)
    np.save(os.path.join(OUT_DIR, f"{tag}_trial{trial_idx}_jfinv.npy"), jfinv)

    # save per-trial plots
    save_heatmap(jf, os.path.join(IMG_DIR, f"{tag}_trial{trial_idx}_jf.png"), f"{tag} | trial {trial_idx} | JF")
    save_heatmap(jfinv, os.path.join(IMG_DIR, f"{tag}_trial{trial_idx}_jfinv.png"), f"{tag} | trial {trial_idx} | JF-INV")

    del encoder, x_tensor, method, result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return jf, jfinv


# =====================================================
# Build trials
# =====================================================
stim_times = np.asarray(t_evt[EVENT_NAME], dtype=np.float32).reshape(-1)
stim_times = stim_times[np.isfinite(stim_times)]

if TRIAL_LIMIT is not None:
    stim_times = stim_times[:TRIAL_LIMIT]

print("number of valid trials:", len(stim_times))

all_trials, used_event_times = build_all_trials(
    unit=unit,
    event_times=stim_times,
    pre_ms=PRE_MS,
    post_ms=POST_MS,
    bin_ms=BIN_MS,
    trial_limit=TRIAL_LIMIT,
)

print("all_trials shape:", all_trials.shape)   # [n_trials, time_bins, neurons]


# =====================================================
# =====================================================
total_trials = all_trials.shape[0]
if total_trials > N_RANDOM_TRIALS:
    random_indices = np.random.choice(total_trials, size=N_RANDOM_TRIALS, replace=False)
    random_indices.sort() 
else:
    random_indices = np.arange(total_trials)

print(f"\n--- Randomly selected {len(random_indices)} trials out of {total_trials} ---")


# =====================================================
# Loop over trials:
# train separately on each trial -> compute JF/JF-inv -> average later
# =====================================================
cebra_jf_sum = None
cebra_jfinv_sum = None
acorn_jf_sum = None
acorn_jfinv_sum = None

n_trials_used = 0

for step, trial_idx in enumerate(random_indices):
    print("\n" + "=" * 80)
    print(f"Processing randomly selected trial {step + 1}/{len(random_indices)} (Original Trial ID: {trial_idx})")
    print("=" * 80)

    X_trial = all_trials[trial_idx]
    X_trial, mu, sigma = normalize_trial(X_trial)

    # -----------------------
    # CEBRA
    # -----------------------
    cebra_model, _ = train_cebra_on_trial(X_trial, adv=False)
    cebra_jf, cebra_jfinv = compute_trial_attribution(
        cebra_model, X_trial, "CEBRA", trial_idx
    )

    del cebra_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # -----------------------
    # ACORN
    # -----------------------
    acorn_model, _ = train_cebra_on_trial(X_trial, adv=True)
    acorn_jf, acorn_jfinv = compute_trial_attribution(
        acorn_model, X_trial, "ACORN", trial_idx
    )

    del acorn_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # -----------------------
    # accumulate means
    # -----------------------
    if cebra_jf_sum is None:
        cebra_jf_sum = np.zeros_like(cebra_jf, dtype=np.float64)
        cebra_jfinv_sum = np.zeros_like(cebra_jfinv, dtype=np.float64)
        acorn_jf_sum = np.zeros_like(acorn_jf, dtype=np.float64)
        acorn_jfinv_sum = np.zeros_like(acorn_jfinv, dtype=np.float64)

    cebra_jf_sum += cebra_jf
    cebra_jfinv_sum += cebra_jfinv
    acorn_jf_sum += acorn_jf
    acorn_jfinv_sum += acorn_jfinv

    n_trials_used += 1

    # free trial arrays
    del X_trial, cebra_jf, cebra_jfinv, acorn_jf, acorn_jfinv
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# =====================================================
# Final mean maps
# =====================================================
cebra_jf_mean = (cebra_jf_sum / max(n_trials_used, 1)).astype(np.float32)
cebra_jfinv_mean = (cebra_jfinv_sum / max(n_trials_used, 1)).astype(np.float32)

acorn_jf_mean = (acorn_jf_sum / max(n_trials_used, 1)).astype(np.float32)
acorn_jfinv_mean = (acorn_jfinv_sum / max(n_trials_used, 1)).astype(np.float32)

np.save(os.path.join(OUT_DIR, "CEBRA_mean_jf.npy"), cebra_jf_mean)
np.save(os.path.join(OUT_DIR, "CEBRA_mean_jfinv.npy"), cebra_jfinv_mean)
np.save(os.path.join(OUT_DIR, "ACORN_mean_jf.npy"), acorn_jf_mean)
np.save(os.path.join(OUT_DIR, "ACORN_mean_jfinv.npy"), acorn_jfinv_mean)

save_heatmap(cebra_jf_mean, os.path.join(IMG_DIR, "CEBRA_jf_mean.png"), "CEBRA | mean JF over 20 random trials")
save_heatmap(cebra_jfinv_mean, os.path.join(IMG_DIR, "CEBRA_jfinv_mean.png"), "CEBRA | mean JF-INV over 20 random trials")
save_heatmap(acorn_jf_mean, os.path.join(IMG_DIR, "ACORN_jf_mean.png"), "ACORN | mean JF over 20 random trials")
save_heatmap(acorn_jfinv_mean, os.path.join(IMG_DIR, "ACORN_jfinv_mean.png"), "ACORN | mean JF-INV over 20 random trials")

print("\nDONE")
print("trials used:", n_trials_used)
print("saved plots in:", IMG_DIR)
print("saved mean arrays in:", OUT_DIR)

# import os
# import gc
# import numpy as np
# import torch
# import scipy.io as sio
# import matplotlib.pyplot as plt
# from utils.constants import CEBRA_DIR
# from utils.min_distance import min_l2_distance
# import sys
# sys.path.insert(0, str(CEBRA_DIR))
# import cebra
# from cebra import CEBRA

# DATA_PATH = "./data/spk/M021519_spk.mat"
# OUT_DIR = "./outputs"
# IMG_DIR = "./image"
# os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(IMG_DIR, exist_ok=True)
# BATCH_SIZE = 256
# MAX_ITER = 10000
# OUTPUT_DIM = 16
# TRIAL_ID = 126
# PRE_MS = 500
# POST_MS = 1000
# BIN_MS = 10

# mat = sio.loadmat(DATA_PATH, simplify_cells=True, squeeze_me=True)
# unit = mat["unit"]
# t_evt = mat["t_evt"]
# print("neurons:", len(unit))
# print("events:", t_evt.keys())

# def make_trial_matrix(unit, event_time, pre_ms, post_ms, bin_ms):
#     n_neurons = len(unit)
#     n_bins = int((pre_ms + post_ms) / bin_ms)
#     X = np.zeros((n_bins, n_neurons), dtype=np.float32)
#     start = event_time - pre_ms / 1000
#     end = event_time + post_ms / 1000
#     for n in range(n_neurons):
#         spikes = unit[n]["timestamps"]
#         spikes = spikes[(spikes >= start) & (spikes <= end)]
#         spikes_ms = (spikes - event_time) * 1000
#         bins = ((spikes_ms + pre_ms) / bin_ms).astype(int)
#         bins = bins[(bins >= 0) & (bins < n_bins)]
#         for b in bins:
#             X[b, n] += 1
#     return X

# stim_times = np.asarray(t_evt["stim_on"])
# trial_time = stim_times[TRIAL_ID]
# X = make_trial_matrix(unit, trial_time, PRE_MS, POST_MS, BIN_MS)
# print("X:", X.shape)
# X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
# X = X.astype(np.float32)

# def train_cebra(X, adv=False):
#     mode = "adversarial" if adv else "clean"
#     eps = float(min_l2_distance(X)) / 2
#     eps = max(eps, 1e-6)
#     print("\nTraining", mode, "epsilon:", eps)
#     model = CEBRA(
#         batch_size=BATCH_SIZE,
#         temperature=0.4,
#         model_architecture="offset10-model",
#         time_offsets=10,
#         max_iterations=MAX_ITER,
#         output_dimension=OUTPUT_DIM,
#         verbose=True,
#         training_mode="adversarial" if adv else "clean",
#         adv_alpha=eps / 5 if adv else 0,
#         adv_epsilon=eps if adv else 0,
#         adv_steps=10 if adv else 0,
#         attack_norm="linf",
#         num_hidden_units=32,
#     )
#     labels = np.arange(len(X)).astype(np.float32)
#     model.fit(X, labels)
#     return model, eps

# def get_attribution(model, name):
#     encoder = model.solver_.model.to("cuda")
#     x_tensor = torch.tensor(X, dtype=torch.float32, device="cuda", requires_grad=True)
#     method = cebra.attribution.init(name="jacobian-based-batched", model=encoder, input_data=x_tensor, output_dimension=OUTPUT_DIM)
#     result = method.compute_attribution_map(batch_size=128)
#     print(result.keys())
#     jf = result["jf"]
#     jf_inv = result["jf-inv-svd"]
#     torch.save(jf, f"{OUT_DIR}/M021519_trial{TRIAL_ID}_{name}_jf.pt")
#     torch.save(jf_inv, f"{OUT_DIR}/M021519_trial{TRIAL_ID}_{name}_jf_inv.pt")
#     save_heatmap(jf, name + "_jacobian")
#     save_heatmap(jf_inv, name + "_inverse_jacobian")
#     del encoder, x_tensor
#     gc.collect()
#     torch.cuda.empty_cache()

# def save_heatmap(tensor, name):
#     if torch.is_tensor(tensor):
#         arr = tensor.detach().cpu().numpy()
#     else:
#         arr = np.asarray(tensor)
#     if arr.ndim == 3:
#         arr = np.abs(arr).mean(axis=0)
#     else:
#         arr = np.abs(arr)
#     plt.figure(figsize=(10,6))
#     plt.imshow(arr, aspect="auto")
#     plt.colorbar(label="absolute attribution")
#     plt.xlabel("Neuron")
#     plt.ylabel("Latent dimension")
#     plt.title(name)
#     plt.tight_layout()
#     plt.savefig(f"{IMG_DIR}/{name}.png", dpi=300)
#     plt.close()

# cebra_model, eps = train_cebra(X, adv=False)
# get_attribution(cebra_model, "CEBRA")
# del cebra_model
# gc.collect()
# torch.cuda.empty_cache()

# acorn_model, eps = train_cebra(X, adv=True)
# get_attribution(acorn_model, "ACORN")
# print("\nDONE")

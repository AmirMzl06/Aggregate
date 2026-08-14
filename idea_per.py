import os
import gc
import sys
import numpy as np
import pandas as pd
import torch
import scipy.io as sio
import matplotlib.pyplot as plt

from utils.constants import CEBRA_DIR
from utils.min_distance import min_l2_distance
sys.path.insert(0, str(CEBRA_DIR))
import cebra
import cebra.attribution
from cebra import CEBRA

# =================== CONFIG ===================
DATA_PATH = "./data/spk/X021920_spk.mat"
BHV_PATH = "./data/behav/X021920_trialtype.csv"
SESSION_PREFIX = os.path.basename(DATA_PATH).replace("_spk.mat", "")
OUT_DIR = "./outputs_per_trial"
IMG_DIR = "./image_per_trial"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

BATCH_SIZE = 256
MAX_ITER = 5000
OUTPUT_DIM = 16
PRE_MS = 500
POST_MS = 1000
BIN_MS = 10
SEQ_LEN = int((PRE_MS + POST_MS) / BIN_MS)
EPS_SAMPLE_SIZE = 2000
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================== LOAD ===================
mat = sio.loadmat(DATA_PATH, simplify_cells=True, squeeze_me=True)
unit = mat["unit"]
t_evt = mat["t_evt"]
bhv = pd.read_csv(BHV_PATH)
stim_times = np.asarray(t_evt["stim_on"], dtype=np.float32).reshape(-1)
print(f"neurons: {len(unit)} | device: {device}")

def cleanup(*objects):
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def make_trial_matrix(event_time):
    n_neurons = len(unit)
    n_bins = SEQ_LEN
    X = np.zeros((n_bins, n_neurons), dtype=np.float32)
    start = event_time - PRE_MS / 1000.0
    end = event_time + POST_MS / 1000.0
    for n in range(n_neurons):
        spikes = np.asarray(unit[n]["timestamps"], dtype=np.float32).reshape(-1)
        spikes = spikes[(spikes >= start) & (spikes <= end)]
        spikes_ms = (spikes - event_time) * 1000.0
        bins = ((spikes_ms + PRE_MS) / BIN_MS).astype(int)
        bins = bins[(bins >= 0) & (bins < n_bins)]
        for b in bins:
            X[b, n] += 1.0
    return X

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

# =================== FILTER ===================
valid_trial_ids = []
for tid in range(min(len(stim_times), len(bhv))):
    row = bhv.iloc[tid]
    if str(row.get("task", "")).strip().lower() != "2afc":
        continue
    brk = pd.to_numeric(row.get("brk", np.nan), errors="coerce")
    if not np.isfinite(brk) or brk != 0:
        continue
    valid_trial_ids.append(tid)

valid_trial_ids = np.asarray(valid_trial_ids, dtype=int)
print(f"valid 2AFC trials: {len(valid_trial_ids)}")

# Reference trial = first valid trial
REF_TID = int(valid_trial_ids[0])
X_ref_raw = make_trial_matrix(float(stim_times[REF_TID]))
mu_ref = X_ref_raw.mean(axis=0, keepdims=True)
sigma_ref = X_ref_raw.std(axis=0, keepdims=True) + 1e-8
X_ref_norm = ((X_ref_raw - mu_ref) / sigma_ref).astype(np.float32)

# =================== TRAIN FUNCTION ===================
def train_model_on_trial(X_norm, adv=False):
    name = "ACORN" if adv else "CEBRA"
    sample_size = min(EPS_SAMPLE_SIZE, len(X_norm))
    sample_idx = rng.choice(len(X_norm), size=sample_size, replace=False)
    eps = float(min_l2_distance(X_norm[sample_idx])) / 2.0
    eps = max(eps, 1e-6)
    
    model = CEBRA(
        batch_size=min(BATCH_SIZE, len(X_norm)),
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
    )
    time_lbl = np.arange(len(X_norm), dtype=np.float32).reshape(-1, 1)
    trial_lbl = np.zeros(len(X_norm), dtype=np.int64)
    model.fit(X_norm.astype(np.float32), time_lbl, trial_lbl)
    return model

# =================== JACOBIAN FUNCTION ===================
def get_jacobian_tensors(model, X_input):
    encoder = model.solver_.model.to(device)
    if hasattr(encoder, "split_outputs"):
        encoder.split_outputs = False
    encoder.eval()
    
    x_tensor = torch.tensor(X_input, dtype=torch.float32, device=device, requires_grad=True)
    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=encoder,
        input_data=x_tensor,
        output_dimension=OUTPUT_DIM,
    )
    result = method.compute_attribution_map(batch_size=min(128, len(X_input)))
    
    jf = result["jf"].cpu()
    jf_inv = result.get("jf-inv-svd", result.get("jf-inv-lsq")).cpu()
    
    cleanup(encoder, x_tensor, method, result)
    return jf, jf_inv

# ============================================================
# EXPERIMENT 1: SINGLE-TRIAL
# Train & Jacobian both on REF trial
# ============================================================
print("\n" + "="*70)
print("EXPERIMENT 1: SINGLE-TRIAL")
print(f"Train & Jacobian on trial {REF_TID}")
print("="*70)

cebra_single = train_model_on_trial(X_ref_norm, adv=False)
acorn_single = train_model_on_trial(X_ref_norm, adv=True)

jf_cs, jfinv_cs = get_jacobian_tensors(cebra_single, X_ref_norm)
jf_as, jfinv_as = get_jacobian_tensors(acorn_single, X_ref_norm)

# Save
torch.save(jf_cs, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_ref{REF_TID}_CEBRA_single_jf.pt"))
torch.save(jfinv_cs, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_ref{REF_TID}_CEBRA_single_jf_inv.pt"))
torch.save(jf_as, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_ref{REF_TID}_ACORN_single_jf.pt"))
torch.save(jfinv_as, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_ref{REF_TID}_ACORN_single_jf_inv.pt"))

save_heatmap(jf_cs, os.path.join(IMG_DIR, "CEBRA_single_jacobian.png"), "CEBRA Single-Trial Jacobian")
save_heatmap(jfinv_cs, os.path.join(IMG_DIR, "CEBRA_single_inv_jacobian.png"), "CEBRA Single-Trial Inv Jacobian")
save_heatmap(jf_as, os.path.join(IMG_DIR, "ACORN_single_jacobian.png"), "ACORN Single-Trial Jacobian")
save_heatmap(jfinv_as, os.path.join(IMG_DIR, "ACORN_single_inv_jacobian.png"), "ACORN Single-Trial Inv Jacobian")

cleanup(cebra_single, acorn_single)
print("Single-trial done.")

# ============================================================
# EXPERIMENT 2: PER-TRIAL MODELS
# Each trial -> its own CEBRA + ACORN
# Jacobian computed on REF trial (with that model's norm stats)
# ============================================================
print("\n" + "="*70)
print("EXPERIMENT 2: PER-TRIAL MODELS")
print(f"Training {len(valid_trial_ids)} CEBRA + {len(valid_trial_ids)} ACORN models...")
print("WARNING: This will take a very long time!")
print("="*70)

cebra_jf_list = []
cebra_jfinv_list = []
acorn_jf_list = []
acorn_jfinv_list = []

for idx, tid in enumerate(valid_trial_ids):
    print(f"\n>>> {idx+1}/{len(valid_trial_ids)} | Trial ID={tid}")
    
    # Build & normalize THIS trial
    X_i_raw = make_trial_matrix(float(stim_times[tid]))
    mu_i = X_i_raw.mean(axis=0, keepdims=True)
    sigma_i = X_i_raw.std(axis=0, keepdims=True) + 1e-8
    X_i_norm = ((X_i_raw - mu_i) / sigma_i).astype(np.float32)
    
    # REF trial normalized with THIS model's stats
    X_ref_for_model = ((X_ref_raw - mu_i) / sigma_i).astype(np.float32)
    
    # Train
    cebra_i = train_model_on_trial(X_i_norm, adv=False)
    acorn_i = train_model_on_trial(X_i_norm, adv=True)
    
    # Jacobians on REF trial
    jf_c, jfinv_c = get_jacobian_tensors(cebra_i, X_ref_for_model)
    jf_a, jfinv_a = get_jacobian_tensors(acorn_i, X_ref_for_model)
    
    cebra_jf_list.append(jf_c)
    cebra_jfinv_list.append(jfinv_c)
    acorn_jf_list.append(jf_a)
    acorn_jfinv_list.append(jfinv_a)
    
    # Save per-trial
    suffix = f"trial{tid}_on_ref"
    torch.save(jf_c, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_ref{REF_TID}_CEBRA_{suffix}_jf.pt"))
    torch.save(jfinv_c, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_ref{REF_TID}_CEBRA_{suffix}_jf_inv.pt"))
    torch.save(jf_a, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_ref{REF_TID}_ACORN_{suffix}_jf.pt"))
    torch.save(jfinv_a, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_ref{REF_TID}_ACORN_{suffix}_jf_inv.pt"))
    
    # Plot only for REF model (tid == REF_TID)
    if tid == REF_TID:
        save_heatmap(jf_c, os.path.join(IMG_DIR, "CEBRA_trialREF_on_ref_jacobian.png"), "CEBRA TrialREF-on-REF Jacobian")
        save_heatmap(jfinv_c, os.path.join(IMG_DIR, "CEBRA_trialREF_on_ref_inv_jacobian.png"), "CEBRA TrialREF-on-REF Inv Jacobian")
        save_heatmap(jf_a, os.path.join(IMG_DIR, "ACORN_trialREF_on_ref_jacobian.png"), "ACORN TrialREF-on-REF Jacobian")
        save_heatmap(jfinv_a, os.path.join(IMG_DIR, "ACORN_trialREF_on_ref_inv_jacobian.png"), "ACORN TrialREF-on-REF Inv Jacobian")
    
    cleanup(cebra_i, acorn_i)

# =================== AVERAGE JACOBIANS ===================
print("\nComputing averages...")

cebra_jf_avg = torch.stack(cebra_jf_list).mean(dim=0)
cebra_jfinv_avg = torch.stack(cebra_jfinv_list).mean(dim=0)
acorn_jf_avg = torch.stack(acorn_jf_list).mean(dim=0)
acorn_jfinv_avg = torch.stack(acorn_jfinv_list).mean(dim=0)

torch.save(cebra_jf_avg, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_ref{REF_TID}_CEBRA_avg_jf.pt"))
torch.save(cebra_jfinv_avg, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_ref{REF_TID}_CEBRA_avg_jf_inv.pt"))
torch.save(acorn_jf_avg, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_ref{REF_TID}_ACORN_avg_jf.pt"))
torch.save(acorn_jfinv_avg, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_ref{REF_TID}_ACORN_avg_jf_inv.pt"))

save_heatmap(cebra_jf_avg, os.path.join(IMG_DIR, "CEBRA_avg_jacobian.png"), "CEBRA Avg Jacobian (all per-trial models)")
save_heatmap(cebra_jfinv_avg, os.path.join(IMG_DIR, "CEBRA_avg_inv_jacobian.png"), "CEBRA Avg Inv Jacobian")
save_heatmap(acorn_jf_avg, os.path.join(IMG_DIR, "ACORN_avg_jacobian.png"), "ACORN Avg Jacobian (all per-trial models)")
save_heatmap(acorn_jfinv_avg, os.path.join(IMG_DIR, "ACORN_avg_inv_jacobian.png"), "ACORN Avg Inv Jacobian")

# =================== SUMMARY ===================
print("\n" + "="*70)
print("ALL DONE")
print(f"Ref trial: {REF_TID} | Total per-trial models: {len(valid_trial_ids)}")
print("\nOutputs:")
print("  [Exp1] *_single_jf.pt / *_single_jf_inv.pt")
print("  [Exp2] *_trial{ID}_on_ref_jf.pt / *_jf_inv.pt  (per trial)")
print("  [Exp2] *_avg_jf.pt / *_avg_jf_inv.pt  (mean over all)")
print("="*70)

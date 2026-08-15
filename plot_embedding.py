import os
import gc
import sys
import numpy as np
import pandas as pd
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.constants import CEBRA_DIR
from utils.min_distance import min_l2_distance
sys.path.insert(0, str(CEBRA_DIR))
import cebra
import cebra.attribution
from cebra import CEBRA
from scipy.ndimage import gaussian_filter1d

DATA_PATH = "./data/spk/X021920_spk.mat"
BHV_PATH = "./data/behav/X021920_trialtype.csv"
SESSION_PREFIX = os.path.basename(DATA_PATH).replace("_spk.mat", "")
OUT_DIR = "./outputs"
IMG_DIR = "./image_smooth"
EMB_DIR = "./image_emb"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)
BATCH_SIZE = 256
MAX_ITER = 10000
OUTPUT_DIM = 16
PRE_MS = 500
POST_MS = 1000
BIN_MS = 10
SEQ_LEN = int((PRE_MS + POST_MS) / BIN_MS)
EPS_SAMPLE_SIZE = 2000
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

mat = sio.loadmat(DATA_PATH, simplify_cells=True, squeeze_me=True)
unit = mat["unit"]
t_evt = mat["t_evt"]
bhv = pd.read_csv(BHV_PATH)
stim_times = np.asarray(t_evt["stim_on"], dtype=np.float32).reshape(-1)
print(f"neurons: {len(unit)} | behavior trials: {len(bhv)} | device: {DEVICE}")

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

ATTR_TRIAL_IDX = valid_trial_ids[0]
event_time_attr = float(stim_times[ATTR_TRIAL_IDX])
X_single = make_trial_matrix(event_time_attr)
print("Single trial matrix:", X_single.shape)
mu_single = X_single.mean(axis=0, keepdims=True)
sigma_single = X_single.std(axis=0, keepdims=True) + 1e-8
X_single_norm = ((X_single - mu_single) / sigma_single).astype(np.float32)

all_trials_raw = []
for tid in valid_trial_ids:
    all_trials_raw.append(make_trial_matrix(float(stim_times[tid])))
all_concat = np.concatenate(all_trials_raw, axis=0)
mu_all = all_concat.mean(axis=0, keepdims=True)
sigma_all = all_concat.std(axis=0, keepdims=True) + 1e-8

def normalize_multi(X):
    X = X.astype(np.float32).copy()
    sigma_bins = 100.0 / BIN_MS
    for n in range(X.shape[1]):
        X[:, n] = gaussian_filter1d(X[:, n], sigma=sigma_bins, mode="reflect")
    return X

X_multi_parts = []
time_multi_parts = []
trial_multi_parts = []
for i, raw in enumerate(all_trials_raw):
    X_t = normalize_multi(raw)
    X_multi_parts.append(X_t)
    time_multi_parts.append(np.arange(len(X_t), dtype=np.float32))
    trial_multi_parts.append(np.full(len(X_t), i, dtype=np.int64))
X_multi = np.concatenate(X_multi_parts, axis=0)
time_labels_multi = np.concatenate(time_multi_parts).reshape(-1, 1)
trial_labels_multi = np.concatenate(trial_multi_parts)
print("Multi-trial concatenated:", X_multi.shape)
X_attr_multi = normalize_multi(X_single)

def train_cebra(X, time_lbl, trial_lbl, adv=False):
    name = "ACORN" if adv else "CEBRA"
    sample_size = min(EPS_SAMPLE_SIZE, len(X))
    sample_idx = rng.choice(len(X), size=sample_size, replace=False)
    eps = float(min_l2_distance(X[sample_idx])) / 2.0
    eps = max(eps, 1e-6)
    print(f"\nTraining {name} | samples={len(X)} | eps={eps:.5f}")
    model = CEBRA(
        batch_size=min(BATCH_SIZE, len(X)),
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
    model.fit(X.astype(np.float32), time_lbl, trial_lbl)
    return model

print("\n" + "=" * 70)
print("EXPERIMENT 1: SINGLE TRIAL")
print("=" * 70)
time_single = np.arange(len(X_single_norm), dtype=np.float32).reshape(-1, 1)
trial_single = np.zeros(len(X_single_norm), dtype=np.int64)
cebra_single = train_cebra(X_single_norm, time_single, trial_single, adv=False)
acorn_single = train_cebra(X_single_norm, time_single, trial_single, adv=True)

print("\n" + "=" * 70)
print("EXPERIMENT 2: MULTI TRIAL")
print("=" * 70)
cebra_multi = train_cebra(X_multi, time_labels_multi, trial_labels_multi, adv=False)
acorn_multi = train_cebra(X_multi, time_labels_multi, trial_labels_multi, adv=True)

def compute_jacobian(model, name, X_ref, suffix):
    encoder = model.solver_.model.to(DEVICE)
    if hasattr(encoder, "split_outputs"):
        encoder.split_outputs = False
    encoder.eval()
    x_tensor = torch.tensor(X_ref, dtype=torch.float32, device=DEVICE, requires_grad=True)
    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=encoder,
        input_data=x_tensor,
        output_dimension=OUTPUT_DIM
    )
    result = method.compute_attribution_map(batch_size=min(128, len(X_ref)))
    jf = result["jf"]
    jf_inv = result.get("jf-inv-svd", result.get("jf-inv-lsq"))
    torch.save(jf, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_trial{ATTR_TRIAL_IDX}_{name}_{suffix}_jf.pt"))
    torch.save(jf_inv, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_trial{ATTR_TRIAL_IDX}_{name}_{suffix}_jf_inv.pt"))
    save_heatmap(jf, os.path.join(IMG_DIR, f"{name}_{suffix}_jacobian.png"), f"{name} {suffix} Jacobian")
    save_heatmap(jf_inv, os.path.join(IMG_DIR, f"{name}_{suffix}_inv_jacobian.png"), f"{name} {suffix} Inverse Jacobian")
    cleanup(encoder, x_tensor, method, result)

print("\nComputing Jacobians...")
compute_jacobian(cebra_single, "CEBRA", X_single_norm, "single")
compute_jacobian(acorn_single, "ACORN", X_single_norm, "single")
compute_jacobian(cebra_multi, "CEBRA", X_attr_multi, "multi")
compute_jacobian(acorn_multi, "ACORN", X_attr_multi, "multi")

def get_embedding(model, X):
    embedding = model.transform(X.astype(np.float32))
    if torch.is_tensor(embedding):
        embedding = embedding.detach().cpu().numpy()
    return np.asarray(embedding, dtype=np.float32)

def normalized_time(n):
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    return np.linspace(0.0, 1.0, n, dtype=np.float32)

def plot_native_cebra_embedding(embedding, name, suffix):
    labels = normalized_time(len(embedding))
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    cebra.plot_embedding(
        embedding,
        embedding_labels=labels,
        cmap="viridis",
        idx_order=(0, 1, 2),
        markersize=1.5,
        alpha=0.6,
        title=f"{name} | {suffix} | Latents 1-3",
        ax=ax
    )
    path = os.path.join(EMB_DIR, f"{name}_{suffix}_native_latents123.png")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

def plot_pca_embedding(embedding, name, suffix):
    pca3 = PCA(n_components=3)
    emb_pca3 = pca3.fit_transform(embedding)
    explained = pca3.explained_variance_ratio_
    labels = normalized_time(len(embedding))
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(emb_pca3[:, 0], emb_pca3[:, 1], emb_pca3[:, 2], c=labels, cmap="viridis", s=3, alpha=0.6)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({explained[2]*100:.1f}%)")
    ax.set_title(f"{name} | {suffix} | PCA 3D")
    fig.colorbar(scatter, ax=ax, shrink=0.7, label="Normalized time")
    fig.tight_layout()
    path3 = os.path.join(EMB_DIR, f"{name}_{suffix}_PCA3D.png")
    fig.savefig(path3, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path3)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(emb_pca3[:, 0], emb_pca3[:, 1], c=labels, cmap="viridis", s=4, alpha=0.65)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title(f"{name} | {suffix} | PCA 2D")
    fig.colorbar(scatter, ax=ax, label="Normalized time")
    fig.tight_layout()
    path2 = os.path.join(EMB_DIR, f"{name}_{suffix}_PCA2D.png")
    fig.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path2)
    np.savez(os.path.join(EMB_DIR, f"{name}_{suffix}_PCA.npz"),
             explained_variance_ratio=explained,
             components=pca3.components_,
             transformed=emb_pca3)

def visualize_model(model, X, name, suffix):
    print(f"\nEmbedding visualization: {name} | {suffix}")
    embedding = get_embedding(model, X)
    print("Embedding shape:", embedding.shape)
    np.save(os.path.join(EMB_DIR, f"{name}_{suffix}_embedding.npy"), embedding)
    plot_native_cebra_embedding(embedding, name, suffix)
    plot_pca_embedding(embedding, name, suffix)
    return embedding

print("\n" + "=" * 70)
print("EMBEDDING VISUALIZATION")
print("=" * 70)
embeddings = {}
embeddings["CEBRA_single"] = visualize_model(cebra_single, X_single_norm, "CEBRA", "single")
embeddings["ACORN_single"] = visualize_model(acorn_single, X_single_norm, "ACORN", "single")
embeddings["CEBRA_multi"] = visualize_model(cebra_multi, X_multi, "CEBRA", "multi")
embeddings["ACORN_multi"] = visualize_model(acorn_multi, X_multi, "ACORN", "multi")

print("\nCreating combined PCA comparison...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()
for ax, (key, emb) in zip(axes, embeddings.items()):
    pca = PCA(n_components=2)
    z = pca.fit_transform(emb)
    color = normalized_time(len(z))
    scatter = ax.scatter(z[:, 0], z[:, 1], c=color, cmap="viridis", s=3, alpha=0.6)
    explained = pca.explained_variance_ratio_
    ax.set_title(f"{key} | PC1={explained[0]*100:.1f}% PC2={explained[1]*100:.1f}%")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
fig.colorbar(scatter, ax=axes, shrink=0.85, label="Normalized time")
fig.suptitle(f"{SESSION_PREFIX} | CEBRA / ACORN Embedding Comparison", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
combined_path = os.path.join(EMB_DIR, f"{SESSION_PREFIX}_all_models_PCA2D.png")
fig.savefig(combined_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved:", combined_path)

print("\n" + "=" * 70)
print("ALL DONE")
print("=" * 70)
print("Embedding images saved to:", EMB_DIR)
print("\nGenerated for each model:")
print("  - native_latents123.png")
print("  - PCA3D.png")
print("  - PCA2D.png")
print("  - embedding.npy")
print("\nGenerated additionally:")
print(f"  - {SESSION_PREFIX}_all_models_PCA2D.png")
cleanup(cebra_single, acorn_single, cebra_multi, acorn_multi)
print("\nDONE!")

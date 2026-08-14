import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import sys

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
TRIAL_IDX = 0
PRE_MS = 500
POST_MS = 1000
BIN_MS = 10
SEQ_LEN = int((PRE_MS + POST_MS) / BIN_MS)
OUTPUT_DIM = 16
MAX_ITER = 5000
RANDOM_SEED = 42

# Predictor
HIDDEN_DIM = 64
EPOCHS = 5000
LR = 1e-3
WD = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUT_DIR = "./outputs"
IMG_DIR = "./image"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# =================== LOAD ===================
mat = sio.loadmat(DATA_PATH, simplify_cells=True, squeeze_me=True)
unit = mat["unit"]
t_evt = mat["t_evt"]
stim_times = np.asarray(t_evt["stim_on"], dtype=np.float32).reshape(-1)
bhv = pd.read_csv(BHV_PATH)

valid_ids = []
for tid in range(min(len(stim_times), len(bhv))):
    row = bhv.iloc[tid]
    if str(row.get("task", "")).strip().lower() != "2afc":
        continue
    brk = pd.to_numeric(row.get("brk", np.nan), errors="coerce")
    if not np.isfinite(brk) or brk != 0:
        continue
    valid_ids.append(tid)

trial_id = valid_ids[TRIAL_IDX]
event_time = float(stim_times[trial_id])
print(f"Trial: {trial_id} | event_time: {event_time:.3f}s")

# =================== TRIAL MATRIX ===================
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

X_trial = make_trial_matrix(event_time)
n_neurons = X_trial.shape[1]
print(f"X_trial: {X_trial.shape}")

# Normalize per neuron (z-score across time)
mu = X_trial.mean(axis=0, keepdims=True)
sigma = X_trial.std(axis=0, keepdims=True) + 1e-8
X_norm = ((X_trial - mu) / sigma).astype(np.float32)

# =================== TRAIN CEBRA & ACORN ===================
def train_cebra_model(X, adv=False):
    name = "ACORN" if adv else "CEBRA"
    x_torch = torch.tensor(X, dtype=torch.float32)
    eps = float(min_l2_distance(x_torch)) / 2.0
    eps = max(eps, 1e-6)
    print(f"\nTraining {name} | eps={eps:.5f}")

    model = CEBRA(
        batch_size=min(256, len(X)),
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
    time_labels = np.arange(len(X), dtype=np.float32).reshape(-1, 1)
    trial_labels = np.zeros(len(X), dtype=np.int64)
    model.fit(X.astype(np.float32), time_labels, trial_labels)
    return model

cebra_model = train_cebra_model(X_norm, adv=False)
acorn_model = train_cebra_model(X_norm, adv=True)

# =================== JACOBIAN & INVERSE JACOBIAN ===================
def compute_and_save_attribution(model, name, X_ref, trial_id):
    encoder = model.solver_.model.to(device)
    if hasattr(encoder, "split_outputs"):
        encoder.split_outputs = False
    encoder.eval()

    x_tensor = torch.tensor(X_ref, dtype=torch.float32, device=device, requires_grad=True)

    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=encoder,
        input_data=x_tensor,
        output_dimension=OUTPUT_DIM,
    )
    result = method.compute_attribution_map(batch_size=min(128, len(X_ref)))

    jf = result["jf"]
    if "jf-inv-svd" in result:
        jf_inv = result["jf-inv-svd"]
    elif "jf-inv-lsq" in result:
        jf_inv = result["jf-inv-lsq"]
    else:
        jf_inv = result["jf-inv"]

    # Save tensors
    torch.save(jf, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_trial{trial_id}_{name}_jf.pt"))
    torch.save(jf_inv, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_trial{trial_id}_{name}_jf_inv.pt"))

    # Save heatmaps
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

    save_heatmap(jf, os.path.join(IMG_DIR, f"{name}_jacobian_trial{trial_id}.png"), f"{name} Jacobian")
    save_heatmap(jf_inv, os.path.join(IMG_DIR, f"{name}_inv_jacobian_trial{trial_id}.png"), f"{name} Inverse Jacobian")

    # Cleanup
    del encoder, x_tensor, method, result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nComputing CEBRA attribution...")
compute_and_save_attribution(cebra_model, "CEBRA", X_norm, trial_id)

print("Computing ACORN attribution...")
compute_and_save_attribution(acorn_model, "ACORN", X_norm, trial_id)

# =================== GET EMBEDDINGS ===================
Z_cebra = np.asarray(cebra_model.transform(X_norm))
Z_acorn = np.asarray(acorn_model.transform(X_norm))
print(f"\nZ_cebra: {Z_cebra.shape} | Z_acorn: {Z_acorn.shape}")

# =================== NEXT-STEP: Z[t] -> X[t+1] ===================
def prepare_nextstep_data(Z):
    X_input = Z[:-1]           # (149, 16)
    Y_target = X_norm[1:]      # (149, n_neurons)
    n_samples = len(X_input)
    split_idx = int(0.8 * n_samples)
    return {
        "X_train": X_input[:split_idx],
        "Y_train": Y_target[:split_idx],
        "X_test": X_input[split_idx:],
        "Y_test": Y_target[split_idx:],
        "split": split_idx,
    }

data_cebra = prepare_nextstep_data(Z_cebra)
data_acorn = prepare_nextstep_data(Z_acorn)

print(f"\nNext-step Z[t] -> X[t+1]")
print(f"  Train: bins 0-{data_cebra['split']-1} ({len(data_cebra['X_train'])} samples)")
print(f"  Test:  bins {data_cebra['split']}-{len(X_norm)-2} ({len(data_cebra['X_test'])} samples)")

# =================== MLP DECODER ===================
class NextStepMLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hid_dim, hid_dim // 2),
            nn.LayerNorm(hid_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid_dim // 2, out_dim)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    def forward(self, x):
        return self.net(x)

def train_and_evaluate(data, tag):
    Xtr = torch.tensor(data["X_train"], dtype=torch.float32, device=device)
    Ytr = torch.tensor(data["Y_train"], dtype=torch.float32, device=device)
    Xte = torch.tensor(data["X_test"], dtype=torch.float32, device=device)
    Yte = torch.tensor(data["Y_test"], dtype=torch.float32, device=device)

    decoder = NextStepMLP(OUTPUT_DIM, HIDDEN_DIM, n_neurons).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        decoder.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i+32]
            optimizer.zero_grad()
            loss = loss_fn(decoder(Xtr[idx]), Ytr[idx])
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 1000 == 0:
            decoder.eval()
            with torch.no_grad():
                print(f"  [{tag}] Epoch {epoch+1} | Train MSE: {loss_fn(decoder(Xtr), Ytr).item():.6f}")

    decoder.eval()
    with torch.no_grad():
        pred = decoder(Xte).cpu().numpy()

    mse = mean_squared_error(data["Y_test"], pred)
    r2 = r2_score(data["Y_test"], pred, multioutput='raw_values')
    r2_avg = r2_score(data["Y_test"], pred)

    print(f"\n{'='*50}")
    print(f"[{tag}] TEST | MSE: {mse:.6f} | Avg R²: {r2_avg:.4f}")
    print(f"  R² range: [{r2.min():.3f}, {r2.max():.3f}]")
    print(f"{'='*50}")

    # Plot top 16 neurons
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    for i in range(min(16, n_neurons)):
        ax = axes[i]
        ax.plot(data["Y_test"][:, i], 'b-', label='True', alpha=0.7)
        ax.plot(pred[:, i], 'r--', label='Pred', alpha=0.7)
        ax.set_title(f"N{i} | R²={r2[i]:.3f}")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    plt.suptitle(f"{tag} | Z[t] -> X[t+1] | Trial {trial_id} | Avg R²={r2_avg:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"nextstep_{tag}_trial{trial_id}.png"), dpi=300)
    plt.close()

    # Attribution: which Z dims matter?
    Xte_grad = torch.tensor(data["X_test"], dtype=torch.float32, device=device, requires_grad=True)
    out = decoder(Xte_grad)
    out.mean().backward()
    attr = Xte_grad.grad.cpu().numpy()
    mean_attr = np.abs(attr).mean(axis=0)

    plt.figure(figsize=(10, 4))
    plt.bar(range(OUTPUT_DIM), mean_attr, color='steelblue')
    plt.xlabel(f"{tag} Latent Dimension (Z[t])")
    plt.ylabel("Attribution for X[t+1]")
    plt.title(f"Which {tag} dims predict next neural activity? (Trial {trial_id})")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"attribution_{tag}_trial{trial_id}.png"), dpi=300)
    plt.close()

    print(f"[{tag}] Top predictive dims: {np.argsort(mean_attr)[::-1][:5].tolist()}")

    # Save model
    torch.save(decoder.state_dict(), os.path.join(OUT_DIR, f"nextstep_decoder_{tag}_trial{trial_id}.pt"))

    del decoder, Xtr, Ytr, Xte, Yte
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return r2_avg

# Run for both
r2_cebra = train_and_evaluate(data_cebra, "CEBRA")
r2_acorn = train_and_evaluate(data_acorn, "ACORN")

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"  CEBRA next-step R²: {r2_cebra:.4f}")
print(f"  ACORN next-step R²: {r2_acorn:.4f}")
print(f"{'='*60}")

del cebra_model, acorn_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("\nDONE!")

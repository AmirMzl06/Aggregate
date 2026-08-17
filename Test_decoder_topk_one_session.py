### 2AFC — TopK neuron-attribution pipeline ###
# Trains 2 FULL models (CEBRA clean, ACORN) on all neurons, uses their
# Jacobian / inverse-Jacobian attribution (computed on held-out TEST trials)
# to pick the top-K neurons, then retrains 8 REDUCED models
# (4 neuron subsets x {clean, adversarial}) + a GRU decoder for each.
# Total: 2 full + 8 reduced = 10 CEBRA models, each with its own GRU decoder.

import os
import gc
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.ndimage import gaussian_filter1d

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
OUT_DIR = "./outputs"
IMG_DIR = "./image"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

BATCH_SIZE = 256
MAX_ITER = 10000
OUTPUT_DIM = 16
PRE_MS = 500
POST_MS = 1000
BIN_MS = 10
SEQ_LEN = int((PRE_MS + POST_MS) / BIN_MS)
TEST_SIZE = 0.20
MAX_TRIALS = None
EPS_SAMPLE_SIZE = 2000
RANDOM_SEED = 42
SIDE_TO_LABEL = {"left": 0, "right": 1}
LABEL_NAMES = ["left", "right"]

# --- TopK config ---
# n = number of neurons to keep in each reduced subset.
# None -> defaults to round(sqrt(total_neurons)), same heuristic you used before.
TOPK_N = None
# cap on how many (concatenated, out-of-sample) bins we run the Jacobian
# attribution on, purely so it stays fast; None = use all test bins
ATTR_SAMPLE_SIZE = 3000

# GRU Decoder hyperparameters
DECODER_HIDDEN_DIM = 64
DECODER_NUM_LAYERS = 2
DECODER_DROPOUT = 0.4
DECODER_LR = 1e-3
DECODER_WEIGHT_DECAY = 1e-4
DECODER_EPOCHS = 5000
DECODER_BATCH_SIZE = 32
PRINT_EVERY = 500

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================== LOAD DATA ===================
mat = sio.loadmat(DATA_PATH, simplify_cells=True, squeeze_me=True)
unit = mat["unit"]
t_evt = mat["t_evt"]
bhv = pd.read_csv(BHV_PATH)
stim_times = np.asarray(t_evt["stim_on"], dtype=np.float32).reshape(-1)
TOTAL_NEURONS = len(unit)
print(f"neurons: {TOTAL_NEURONS}")
print(f"behavior trials: {len(bhv)}")
print(f"device: {device}")

K_NEURONS = TOPK_N if TOPK_N is not None else int(round(np.sqrt(TOTAL_NEURONS)))
print(f"TopK neurons per subset (n): {K_NEURONS} / {TOTAL_NEURONS}")


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


# =================== TRIAL / NEURAL HELPERS ===================
def make_trial_matrix(event_time):
    n_neurons = TOTAL_NEURONS
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


def normalize(X):
    # Gaussian smoothing only (z-score left disabled, matching your latest choice)
    X = X.astype(np.float32).copy()
    sigma_bins = 100.0 / BIN_MS  # 100ms smoothing kernel in bin units
    for n in range(X.shape[1]):
        X[:, n] = gaussian_filter1d(X[:, n], sigma=sigma_bins, mode="reflect")
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
    plt.imshow(arr, aspect="auto")
    plt.colorbar(label="absolute attribution")
    plt.xlabel("Neuron")
    plt.ylabel("Latent dimension")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def build_labels(trial_ids):
    """time/trial labels only depend on #trials and #bins, never on which
    neurons are kept, so this is computed once and reused for every subset."""
    time_parts, trial_parts = [], []
    for i in range(len(trial_ids)):
        time_parts.append(np.arange(SEQ_LEN, dtype=np.float32))
        trial_parts.append(np.full(SEQ_LEN, i, dtype=np.int64))
    time_labels = np.concatenate(time_parts).reshape(-1, 1)
    trial_labels = np.concatenate(trial_parts)
    return time_labels, trial_labels


def build_encoder_X(trial_ids, neuron_indices=None):
    parts = []
    for tid in trial_ids:
        raw = make_trial_matrix(float(stim_times[tid]))
        if neuron_indices is not None:
            raw = raw[:, neuron_indices]
        parts.append(normalize(raw))
    return np.concatenate(parts, axis=0).astype(np.float32)


def build_embeddings(trial_ids, model, neuron_indices=None):
    """Returns (n_trials, seq_len, output_dim) -- no pooling, for the GRU decoder."""
    features = []
    for tid in trial_ids:
        raw = make_trial_matrix(float(stim_times[tid]))
        if neuron_indices is not None:
            raw = raw[:, neuron_indices]
        X_t = normalize(raw)
        emb = np.asarray(model.transform(X_t))
        features.append(emb)
    return np.stack(features).astype(np.float32)


# =================== FILTER TRIALS ===================
valid_trial_ids = []
y_all = []
n_candidates = min(len(stim_times), len(bhv))
for tid in range(n_candidates):
    row = bhv.iloc[tid]
    if str(row.get("task", "")).strip().lower() != "2afc":
        continue
    brk = pd.to_numeric(row.get("brk", np.nan), errors="coerce")
    if not np.isfinite(brk) or brk != 0:
        continue
    side = str(row.get("chosenside_2AFC", "")).strip().lower()
    if side not in SIDE_TO_LABEL:
        continue
    valid_trial_ids.append(tid)
    y_all.append(SIDE_TO_LABEL[side])

valid_trial_ids = np.asarray(valid_trial_ids, dtype=int)
y_all = np.asarray(y_all, dtype=np.int64)
if MAX_TRIALS is not None:
    valid_trial_ids = valid_trial_ids[:MAX_TRIALS]
    y_all = y_all[:MAX_TRIALS]
print(f"valid 2AFC trials: {len(valid_trial_ids)} (left={np.sum(y_all == 0)}, right={np.sum(y_all == 1)})")

(train_ids, test_ids, y_train, y_test) = train_test_split(
    valid_trial_ids, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_all
)
print(f"train={len(train_ids)} | test={len(test_ids)}")

train_time_labels, train_trial_labels = build_labels(train_ids)
CHANCE = max(np.mean(y_test == 0), np.mean(y_test == 1))


# =================== CEBRA TRAINING ===================
def train_model(X, time_labels, trial_labels, adv=False):
    name = "ACORN" if adv else "CEBRA"
    sample_size = min(EPS_SAMPLE_SIZE, len(X))
    sample_idx = rng.choice(len(X), size=sample_size, replace=False)
    eps = float(min_l2_distance(X[sample_idx])) / 2.0
    eps = max(eps, 1e-6)
    print(f"training {name} | eps={eps:.5f} | X={X.shape}")
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
    )
    model.fit(X, time_labels, trial_labels)
    return model


# =================== ATTRIBUTION ===================
def get_attribution(model, name, X_ref, save=True):
    encoder = model.solver_.model.to(device)
    if hasattr(encoder, "split_outputs"):
        encoder.split_outputs = False
    x_tensor = torch.tensor(X_ref, dtype=torch.float32,
                             device=next(encoder.parameters()).device, requires_grad=True)
    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=encoder,
        input_data=x_tensor,
        output_dimension=OUTPUT_DIM,
    )
    result = method.compute_attribution_map(batch_size=min(128, len(X_ref)))
    jf = torch.as_tensor(result["jf"]).detach().cpu()
    jf_inv = torch.as_tensor(result.get("jf-inv-svd", result.get("jf-inv-lsq"))).detach().cpu()

    if save:
        torch.save(jf, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_{name}_jf.pt"))
        torch.save(jf_inv, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_{name}_jf_inv.pt"))
        save_heatmap(jf, os.path.join(IMG_DIR, f"{name}_jacobian.png"), f"{name} Jacobian")
        save_heatmap(jf_inv, os.path.join(IMG_DIR, f"{name}_inverse_jacobian.png"), f"{name} inverse Jacobian")

    cleanup(encoder, x_tensor, method, result)
    return jf, jf_inv


def get_per_neuron_score(attr_tensor, total_neurons):
    """One score per neuron, robust to (samples, latent, neurons) /
    (samples, neurons, latent) / (latent, neurons) / (neurons, latent)."""
    attr = torch.abs(attr_tensor)
    if attr.ndim == 3:
        attr_2d = attr.mean(dim=0)
    elif attr.ndim == 2:
        attr_2d = attr
    else:
        raise ValueError(f"Unsupported attribution shape: {tuple(attr.shape)}")

    if attr_2d.shape[0] == total_neurons:
        scores = attr_2d.mean(dim=1)
    elif attr_2d.shape[1] == total_neurons:
        scores = attr_2d.mean(dim=0)
    else:
        raise ValueError(
            f"Cannot identify neuron axis. shape={tuple(attr_2d.shape)}, total_neurons={total_neurons}"
        )
    return scores.cpu().numpy()


def top_k_indices(scores, k):
    return np.argsort(scores)[::-1][:k]


# =================== GRU DECODER ===================
class GRUDecoder(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=2,
                 num_layers=2, dropout_rate=0.4, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        gru_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_out_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(gru_out_dim, output_dim)
        )
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        _, hidden = self.gru(x)
        if self.gru.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            last_hidden = hidden[-1]
        return self.classifier(last_hidden)


def train_decoder(X_train_feats, y_train, X_test_feats, y_test, tag):
    mu = X_train_feats.mean(axis=(0, 1), keepdims=True)
    sigma = X_train_feats.std(axis=(0, 1), keepdims=True) + 1e-8
    Xtr = (X_train_feats - mu) / sigma
    Xte = (X_test_feats - mu) / sigma

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(y_train, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)

    decoder = GRUDecoder(
        input_dim=Xtr.shape[2], hidden_dim=DECODER_HIDDEN_DIM, output_dim=2,
        num_layers=DECODER_NUM_LAYERS, dropout_rate=DECODER_DROPOUT, bidirectional=False,
    ).to(device)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR, weight_decay=DECODER_WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(DECODER_EPOCHS):
        decoder.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for start in range(0, len(Xtr_t), DECODER_BATCH_SIZE):
            idx = perm[start:start + DECODER_BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(decoder(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            optimizer.step()

        if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0:
            decoder.eval()
            with torch.no_grad():
                train_acc = accuracy_score(y_train, decoder(Xtr_t).argmax(dim=1).cpu().numpy())
            print(f"  [{tag}] epoch {epoch+1}/{DECODER_EPOCHS} | train_acc={train_acc:.4f}")

    decoder.eval()
    with torch.no_grad():
        pred = decoder(Xte_t).argmax(dim=1).cpu().numpy()
    accuracy = accuracy_score(y_test, pred)
    print(f"{tag} | test accuracy={accuracy:.4f} | chance={CHANCE:.4f}")

    torch.save(decoder.state_dict(), os.path.join(OUT_DIR, f"decoder_{tag}_gru.pt"))

    cm = np.zeros((2, 2), dtype=int)
    for true, p in zip(y_test, pred):
        cm[true, p] += 1
    pd.DataFrame(cm, index=["true_left", "true_right"], columns=["pred_left", "pred_right"]).to_csv(
        os.path.join(OUT_DIR, f"decoder_{tag}_confusion.csv")
    )

    cleanup(decoder, optimizer, Xtr_t, ytr_t, Xte_t)
    return accuracy


# =====================================================================
# STEP 1 — train the 2 FULL models (all neurons)
# =====================================================================
X_train_full = build_encoder_X(train_ids, neuron_indices=None)
print(f"encoder training data (full): {X_train_full.shape}")

full_models = {}
full_results = {}
for adv, name in [(False, "CEBRA"), (True, "ACORN")]:
    cleanup()
    print(f"\n==================== Training FULL {name} ====================")
    model = train_model(X_train_full, train_time_labels, train_trial_labels, adv=adv)
    full_models[name] = model

    Xtr_emb = build_embeddings(train_ids, model, neuron_indices=None)
    Xte_emb = build_embeddings(test_ids, model, neuron_indices=None)
    acc = train_decoder(Xtr_emb, y_train, Xte_emb, y_test, tag=f"FULL__{name}")
    full_results[name] = {"accuracy": acc, "num_neurons": TOTAL_NEURONS}
    cleanup(Xtr_emb, Xte_emb)

# =====================================================================
# STEP 2 — attribution on held-out TEST trials (out-of-sample, robust:
# uses ALL test bins, not a single trial), then top-K neuron selection
# =====================================================================
test_raw_parts = [normalize(make_trial_matrix(float(stim_times[tid]))) for tid in test_ids]
X_attr_ref = np.concatenate(test_raw_parts, axis=0).astype(np.float32)
if ATTR_SAMPLE_SIZE is not None and len(X_attr_ref) > ATTR_SAMPLE_SIZE:
    attr_idx = rng.choice(len(X_attr_ref), size=ATTR_SAMPLE_SIZE, replace=False)
    X_attr_ref = X_attr_ref[attr_idx]
print(f"\nattribution reference (out-of-sample test bins): {X_attr_ref.shape}")

topk_indices = {}  # e.g. "CEBRA_topJf" -> neuron index array
for name in ["CEBRA", "ACORN"]:
    jf, jf_inv = get_attribution(full_models[name], name, X_attr_ref, save=True)
    jf_scores = get_per_neuron_score(jf, TOTAL_NEURONS)
    jfinv_scores = get_per_neuron_score(jf_inv, TOTAL_NEURONS)

    topk_indices[f"{name}_topJf"] = top_k_indices(jf_scores, K_NEURONS)
    topk_indices[f"{name}_topJfinv"] = top_k_indices(jfinv_scores, K_NEURONS)
    print(f"[{name}] top-{K_NEURONS} Jf neurons:    {topk_indices[f'{name}_topJf'].tolist()}")
    print(f"[{name}] top-{K_NEURONS} Jf-inv neurons: {topk_indices[f'{name}_topJfinv'].tolist()}")

cleanup(*full_models.values())

# =====================================================================
# STEP 3 — retrain on each of the 4 neuron subsets x {clean, adversarial}
# = 8 reduced models, each with its own GRU decoder
# =====================================================================
reduced_results = {}
for reduced_name, neuron_idx in topk_indices.items():
    for adv, mode_name in [(False, "CEBRA"), (True, "ACORN")]:
        tag = f"{reduced_name}__{mode_name}"
        cleanup()
        print(f"\n==================== Reduced: {tag} ({len(neuron_idx)} neurons) ====================")

        X_train_reduced = build_encoder_X(train_ids, neuron_indices=neuron_idx)
        model = train_model(X_train_reduced, train_time_labels, train_trial_labels, adv=adv)

        Xtr_emb = build_embeddings(train_ids, model, neuron_indices=neuron_idx)
        Xte_emb = build_embeddings(test_ids, model, neuron_indices=neuron_idx)
        acc = train_decoder(Xtr_emb, y_train, Xte_emb, y_test, tag=tag)

        reduced_results[tag] = {"accuracy": acc, "num_neurons": len(neuron_idx)}
        cleanup(model, Xtr_emb, Xte_emb)

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "#" * 80)
print(f" SUMMARY FOR {SESSION_PREFIX} ".center(80, "#"))
print("#" * 80)
print(f"chance level (test) = {CHANCE:.4f}  (n_test={len(test_ids)})\n")

rows = []
for name in ["CEBRA", "ACORN"]:
    info = full_results[name]
    print(f"FULL {name:>6} | neurons={info['num_neurons']:<4d} | accuracy={info['accuracy']:.4f}")
    rows.append({"setting": "full", "model": name, "neurons": info["num_neurons"],
                 "accuracy": info["accuracy"], "chance": CHANCE})

for tag, info in reduced_results.items():
    reduced_name, model_name = tag.split("__")
    print(f"{tag:<28} | neurons={info['num_neurons']:<4d} | accuracy={info['accuracy']:.4f}")
    rows.append({"setting": reduced_name, "model": model_name, "neurons": info["num_neurons"],
                 "accuracy": info["accuracy"], "chance": CHANCE})

summary_df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, f"{SESSION_PREFIX}_topk_results.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\nSaved summary CSV to: {csv_path}")
print("\nDONE")

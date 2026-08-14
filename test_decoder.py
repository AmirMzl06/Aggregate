### 2AFC ###
# GRU
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

# GRU Decoder hyperparameters
DECODER_HIDDEN_DIM = 64
DECODER_NUM_LAYERS = 2
DECODER_DROPOUT = 0.4
DECODER_LR = 1e-3
DECODER_WEIGHT_DECAY = 1e-4
DECODER_EPOCHS = 10000
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
print(f"neurons: {len(unit)}")
print(f"behavior trials: {len(bhv)}")
print(f"device: {device}")

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
    plt.imshow(arr, aspect="auto")
    plt.colorbar(label="absolute attribution")
    plt.xlabel("Neuron")
    plt.ylabel("Latent dimension")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

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

# =================== BUILD TRAIN DATA FOR CEBRA ===================
train_raw_parts = []
for tid in train_ids:
    train_raw_parts.append(make_trial_matrix(float(stim_times[tid])))
train_raw_concat = np.concatenate(train_raw_parts, axis=0)
TRAIN_MU = train_raw_concat.mean(axis=0, keepdims=True).astype(np.float32)
TRAIN_SIGMA = (train_raw_concat.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)

def normalize(X):
    return ((X - TRAIN_MU) / TRAIN_SIGMA).astype(np.float32)

X_parts = []
time_parts = []
trial_parts = []
for i, raw_X in enumerate(train_raw_parts):
    X_t = normalize(raw_X)
    X_parts.append(X_t)
    time_parts.append(np.arange(len(X_t), dtype=np.float32))
    trial_parts.append(np.full(len(X_t), i, dtype=np.int64))
X_train = np.concatenate(X_parts, axis=0)
time_labels = np.concatenate(time_parts).reshape(-1, 1)
trial_labels = np.concatenate(trial_parts)
print(f"encoder training data: {X_train.shape}")

# =================== TRAIN CEBRA / ACORN ===================
def train_model(X, adv=False):
    name = "ACORN" if adv else "CEBRA"
    sample_size = min(EPS_SAMPLE_SIZE, len(X))
    sample_idx = rng.choice(len(X), size=sample_size, replace=False)
    eps = float(min_l2_distance(X[sample_idx])) / 2.0
    eps = max(eps, 1e-6)
    print(f"training {name} | eps={eps:.5f}")
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

cebra_model = train_model(X_train, adv=False)
acorn_model = train_model(X_train, adv=True)

# =================== ATTRIBUTION ===================
ATTR_TRIAL_ID = int(train_ids[0])
X_attr = normalize(train_raw_parts[0])

def get_attribution(model, name, X_ref):
    encoder = model.solver_.model.to(device)
    if hasattr(encoder, "split_outputs"):
        encoder.split_outputs = False
    x_tensor = torch.tensor(X_ref, dtype=torch.float32, device=next(encoder.parameters()).device, requires_grad=True)
    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=encoder,
        input_data=x_tensor,
        output_dimension=OUTPUT_DIM
    )
    result = method.compute_attribution_map(batch_size=128)
    jf = result["jf"]
    jf_inv = result.get("jf-inv-svd", result.get("jf-inv-lsq"))
    
    torch.save(jf, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_trial{ATTR_TRIAL_ID}_{name}_jf.pt"))
    torch.save(jf_inv, os.path.join(OUT_DIR, f"{SESSION_PREFIX}_trial{ATTR_TRIAL_ID}_{name}_jf_inv.pt"))
    save_heatmap(jf, os.path.join(IMG_DIR, f"{name}_jacobian.png"), f"{name} Jacobian")
    save_heatmap(jf_inv, os.path.join(IMG_DIR, f"{name}_inverse_jacobian.png"), f"{name} inverse Jacobian")
    cleanup(encoder, x_tensor, method, result)

get_attribution(cebra_model, "CEBRA", X_attr)
get_attribution(acorn_model, "ACORN", X_attr)

# =================== BUILD SEQUENCE-LEVEL EMBEDDINGS ===================
def build_embeddings(trial_ids, model):
    """Returns (n_trials, seq_len, output_dim) — NO mean pooling"""
    features = []
    for tid in trial_ids:
        raw_X = make_trial_matrix(float(stim_times[tid]))
        X_t = normalize(raw_X)
        emb = np.asarray(model.transform(X_t))
        features.append(emb)
    return np.stack(features).astype(np.float32)

print("\nbuilding CEBRA embeddings...")
X_train_cebra = build_embeddings(train_ids, cebra_model)
X_test_cebra = build_embeddings(test_ids, cebra_model)
print(f"  CEBRA train shape: {X_train_cebra.shape}")

print("building ACORN embeddings...")
X_train_acorn = build_embeddings(train_ids, acorn_model)
X_test_acorn = build_embeddings(test_ids, acorn_model)
print(f"  ACORN train shape: {X_train_acorn.shape}")

# =================== GRU DECODER ===================
class GRUDecoder(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=2,
                 num_layers=2, dropout_rate=0.4, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        gru_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gru_out_dim, output_dim)
        )
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
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

# =================== TRAIN DECODER ===================
def train_decoder(X_train_feats, y_train, X_test_feats, y_test, tag):
    mu = X_train_feats.mean(axis=(0, 1), keepdims=True)
    sigma = X_train_feats.std(axis=(0, 1), keepdims=True) + 1e-8
    Xtr = (X_train_feats - mu) / sigma
    Xte = (X_test_feats - mu) / sigma

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(y_train, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)

    decoder = GRUDecoder(
        input_dim=Xtr.shape[2],
        hidden_dim=DECODER_HIDDEN_DIM,
        output_dim=2,
        num_layers=DECODER_NUM_LAYERS,
        dropout_rate=DECODER_DROPOUT,
        bidirectional=False,
    ).to(device)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR, weight_decay=DECODER_WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(DECODER_EPOCHS):
        decoder.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for start in range(0, len(Xtr_t), DECODER_BATCH_SIZE):
            idx = perm[start:start + DECODER_BATCH_SIZE]
            optimizer.zero_grad()
            logits = decoder(Xtr_t[idx])
            loss = loss_fn(logits, ytr_t[idx])
            loss.backward()
            optimizer.step()

        if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0:
            decoder.eval()
            with torch.no_grad():
                train_pred = decoder(Xtr_t).argmax(dim=1).cpu().numpy()
                train_acc = accuracy_score(y_train, train_pred)
            print(f"  [{tag}] epoch {epoch+1}/{DECODER_EPOCHS} | train_acc={train_acc:.4f}")

    decoder.eval()
    with torch.no_grad():
        pred = decoder(Xte_t).argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, pred)
    chance = max(np.mean(y_test == 0), np.mean(y_test == 1))
    print(f"\n{tag} | test accuracy={accuracy:.4f} | chance={chance:.4f}")

    torch.save(decoder.state_dict(), os.path.join(OUT_DIR, f"decoder_{tag}_gru.pt"))

    cm = np.zeros((2, 2), dtype=int)
    for true, p in zip(y_test, pred):
        cm[true, p] += 1
    pd.DataFrame(cm, index=["true_left", "true_right"], columns=["pred_left", "pred_right"]).to_csv(
        os.path.join(OUT_DIR, f"decoder_{tag}_confusion.csv")
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{tag} (GRU) | accuracy={accuracy:.3f}")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"decoder_{tag}_confusion.png"), dpi=300)
    plt.close()

    cleanup(decoder, optimizer, Xtr_t, ytr_t, Xte_t)
    return accuracy

# =================== RUN ===================
cebra_acc = train_decoder(X_train_cebra, y_train, X_test_cebra, y_test, "CEBRA")
acorn_acc = train_decoder(X_train_acorn, y_train, X_test_acorn, y_test, "ACORN")

comparison = pd.DataFrame({
    "model": ["CEBRA", "ACORN"],
    "test_accuracy": [cebra_acc, acorn_acc]
})
comparison.to_csv(os.path.join(OUT_DIR, f"{SESSION_PREFIX}_CEBRA_vs_ACORN_gru_accuracy.csv"), index=False)

print("\n" + "="*60)
print("DONE")
print(f"Session: {SESSION_PREFIX}")
print(f"Attribution trial: {ATTR_TRIAL_ID}")
print(f"Train trials: {len(train_ids)} | Test trials: {len(test_ids)}")
print(f"CEBRA (GRU) accuracy: {cebra_acc:.4f}")
print(f"ACORN (GRU) accuracy: {acorn_acc:.4f}")
print("="*60)

cleanup(cebra_model, acorn_model)


# Two Layer MLP
# import os
# import gc
# import sys
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import scipy.io as sio
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from utils.constants import CEBRA_DIR
# from utils.min_distance import min_l2_distance
# sys.path.insert(0, str(CEBRA_DIR))
# import cebra
# import cebra.attribution
# from cebra import CEBRA

# DATA_PATH = "./data/spk/M021519_spk.mat"
# BHV_PATH = "./data/behav/M021519_trialtype.csv"
# OUT_DIR = "./outputs"
# IMG_DIR = "./image"
# os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(IMG_DIR, exist_ok=True)

# BATCH_SIZE = 256
# MAX_ITER = 10000
# OUTPUT_DIM = 16
# PRE_MS = 500
# POST_MS = 1000
# BIN_MS = 10
# TEST_SIZE = 0.20
# MAX_TRIALS = None
# EPS_SAMPLE_SIZE = 2000
# RANDOM_SEED = 42
# SIDE_TO_LABEL = {"left": 0, "right": 1}
# LABEL_NAMES = ["left", "right"]
# DECODER_HIDDEN_DIM = 48
# DECODER_DROPOUT = 0.4
# DECODER_LR = 1e-3
# DECODER_WEIGHT_DECAY = 1e-4
# DECODER_EPOCHS = 10000
# DECODER_BATCH_SIZE = 32

# torch.manual_seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
# rng = np.random.default_rng(RANDOM_SEED)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mat = sio.loadmat(DATA_PATH, simplify_cells=True, squeeze_me=True)
# unit = mat["unit"]
# t_evt = mat["t_evt"]
# bhv = pd.read_csv(BHV_PATH)
# stim_times = np.asarray(t_evt["stim_on"], dtype=np.float32).reshape(-1)
# print(f"neurons: {len(unit)}")
# print(f"behavior trials: {len(bhv)}")
# print(f"device: {device}")

# def cleanup(*objects):
#     for obj in objects:
#         try:
#             del obj
#         except Exception:
#             pass
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()

# def make_trial_matrix(event_time):
#     n_neurons = len(unit)
#     n_bins = int((PRE_MS + POST_MS) / BIN_MS)
#     X = np.zeros((n_bins, n_neurons), dtype=np.float32)
#     start = event_time - PRE_MS / 1000.0
#     end = event_time + POST_MS / 1000.0
#     for n in range(n_neurons):
#         spikes = np.asarray(unit[n]["timestamps"], dtype=np.float32).reshape(-1)
#         spikes = spikes[(spikes >= start) & (spikes <= end)]
#         spikes_ms = (spikes - event_time) * 1000.0
#         bins = ((spikes_ms + PRE_MS) / BIN_MS).astype(int)
#         bins = bins[(bins >= 0) & (bins < n_bins)]
#         for b in bins:
#             X[b, n] += 1.0
#     return X

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
#     plt.imshow(arr, aspect="auto")
#     plt.colorbar(label="absolute attribution")
#     plt.xlabel("Neuron")
#     plt.ylabel("Latent dimension")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()

# valid_trial_ids = []
# y_all = []
# n_candidates = min(len(stim_times), len(bhv))
# for tid in range(n_candidates):
#     row = bhv.iloc[tid]
#     if str(row.get("task", "")).strip().lower() != "2afc":
#         continue
#     brk = pd.to_numeric(row.get("brk", np.nan), errors="coerce")
#     if not np.isfinite(brk) or brk != 0:
#         continue
#     side = str(row.get("chosenside_2AFC", "")).strip().lower()
#     if side not in SIDE_TO_LABEL:
#         continue
#     valid_trial_ids.append(tid)
#     y_all.append(SIDE_TO_LABEL[side])
# valid_trial_ids = np.asarray(valid_trial_ids, dtype=int)
# y_all = np.asarray(y_all, dtype=np.int64)
# if MAX_TRIALS is not None:
#     valid_trial_ids = valid_trial_ids[:MAX_TRIALS]
#     y_all = y_all[:MAX_TRIALS]
# print(f"valid 2AFC trials: {len(valid_trial_ids)} (left={np.sum(y_all == 0)}, right={np.sum(y_all == 1)})")

# (train_ids, test_ids, y_train, y_test) = train_test_split(
#     valid_trial_ids, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_all
# )
# print(f"train={len(train_ids)} | test={len(test_ids)}")

# train_raw_parts = []
# for tid in train_ids:
#     train_raw_parts.append(make_trial_matrix(float(stim_times[tid])))
# train_raw_concat = np.concatenate(train_raw_parts, axis=0)
# TRAIN_MU = train_raw_concat.mean(axis=0, keepdims=True).astype(np.float32)
# TRAIN_SIGMA = (train_raw_concat.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)

# def normalize(X):
#     return ((X - TRAIN_MU) / TRAIN_SIGMA).astype(np.float32)

# X_parts = []
# time_parts = []
# trial_parts = []
# for i, raw_X in enumerate(train_raw_parts):
#     X_t = normalize(raw_X)
#     X_parts.append(X_t)
#     time_parts.append(np.arange(len(X_t), dtype=np.float32))
#     trial_parts.append(np.full(len(X_t), i, dtype=np.int64))
# X_train = np.concatenate(X_parts, axis=0)
# time_labels = np.concatenate(time_parts).reshape(-1, 1)
# trial_labels = np.concatenate(trial_parts)
# print(f"encoder training data: {X_train.shape}")

# def train_model(X, adv=False):
#     name = "ACORN" if adv else "CEBRA"
#     sample_size = min(EPS_SAMPLE_SIZE, len(X))
#     sample_idx = rng.choice(len(X), size=sample_size, replace=False)
#     eps = float(min_l2_distance(X[sample_idx])) / 2.0
#     eps = max(eps, 1e-6)
#     # eps = 5
#     print(f"training {name} | eps={eps:.5f}")
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
#         num_hidden_units=32, #32,
#     )
#     model.fit(X, time_labels, trial_labels)
#     return model

# cebra_model = train_model(X_train, adv=False)
# acorn_model = train_model(X_train, adv=True)

# ATTR_TRIAL_ID = int(train_ids[0])
# X_attr = normalize(train_raw_parts[0])

# def get_attribution(model, name, X_ref):
#     encoder = model.solver_.model.to(device)
#     if hasattr(encoder, "split_outputs"):
#         encoder.split_outputs = False
#     x_tensor = torch.tensor(X_ref, dtype=torch.float32, device=next(encoder.parameters()).device, requires_grad=True)
#     method = cebra.attribution.init(
#         name="jacobian-based-batched",
#         model=encoder,
#         input_data=x_tensor,
#         output_dimension=OUTPUT_DIM
#     )
#     result = method.compute_attribution_map(batch_size=128)
#     jf = result["jf"]
#     if "jf-inv-svd" in result:
#         jf_inv = result["jf-inv-svd"]
#     else:
#         jf_inv = result["jf-inv-lsq"]
#     torch.save(jf, os.path.join(OUT_DIR, f"M021519_trial{ATTR_TRIAL_ID}_{name}_jf.pt"))
#     torch.save(jf_inv, os.path.join(OUT_DIR, f"M021519_trial{ATTR_TRIAL_ID}_{name}_jf_inv.pt"))
#     save_heatmap(jf, os.path.join(IMG_DIR, f"{name}_jacobian.png"), f"{name} Jacobian")
#     save_heatmap(jf_inv, os.path.join(IMG_DIR, f"{name}_inverse_jacobian.png"), f"{name} inverse Jacobian")
#     cleanup(encoder, x_tensor, method, result)

# get_attribution(cebra_model, "CEBRA", X_attr)
# get_attribution(acorn_model, "ACORN", X_attr)

# def build_embeddings(trial_ids, model):
#     features = []
#     for tid in trial_ids:
#         raw_X = make_trial_matrix(float(stim_times[tid]))
#         X_t = normalize(raw_X)
#         emb = np.asarray(model.transform(X_t))
#         features.append(emb.mean(axis=0))
#     return np.stack(features).astype(np.float32)

# print("building CEBRA embeddings...")
# X_train_cebra = build_embeddings(train_ids, cebra_model)
# X_test_cebra = build_embeddings(test_ids, cebra_model)
# print("building ACORN embeddings...")
# X_train_acorn = build_embeddings(train_ids, acorn_model)
# X_test_acorn = build_embeddings(test_ids, acorn_model)

# class TwoLayerMLP(nn.Module):
#     def __init__(self, input_dim=16, hidden_dim=64, output_dim=2, dropout_rate=0.4):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim, output_dim)
#         )
#         for layer in self.net:
#             if isinstance(layer, nn.Linear):
#                 nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
#                 nn.init.constant_(layer.bias, 0)
#     def forward(self, x):
#         return self.net(x)

# def train_decoder(X_train_feats, y_train, X_test_feats, y_test, tag):
#     mu = X_train_feats.mean(axis=0, keepdims=True)
#     sigma = X_train_feats.std(axis=0, keepdims=True) + 1e-8
#     Xtr = (X_train_feats - mu) / sigma
#     Xte = (X_test_feats - mu) / sigma
#     Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
#     ytr_t = torch.tensor(y_train, dtype=torch.long, device=device)
#     Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
#     decoder = TwoLayerMLP(
#         input_dim=Xtr.shape[1],
#         hidden_dim=DECODER_HIDDEN_DIM,
#         output_dim=2,
#         dropout_rate=DECODER_DROPOUT
#     ).to(device)
#     optimizer = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR, weight_decay=DECODER_WEIGHT_DECAY)
#     loss_fn = nn.CrossEntropyLoss()
#     for epoch in range(DECODER_EPOCHS):
#         print(f"epoch = {epoch}")
#         decoder.train()
#         perm = torch.randperm(len(Xtr_t), device=device)
#         for start in range(0, len(Xtr_t), DECODER_BATCH_SIZE):
#             idx = perm[start:start + DECODER_BATCH_SIZE]
#             optimizer.zero_grad()
#             logits = decoder(Xtr_t[idx])
#             loss = loss_fn(logits, ytr_t[idx])
#             loss.backward()
#             optimizer.step()
#     decoder.eval()
#     with torch.no_grad():
#         pred = decoder(Xte_t).argmax(dim=1).cpu().numpy()
#     accuracy = accuracy_score(y_test, pred)
#     chance = max(np.mean(y_test == 0), np.mean(y_test == 1))
#     cm = np.zeros((2, 2), dtype=int)
#     for true, p in zip(y_test, pred):
#         cm[true, p] += 1
#     print(f"{tag} | test accuracy={accuracy:.4f} | chance={chance:.4f}")
#     torch.save(decoder.state_dict(), os.path.join(OUT_DIR, f"decoder_{tag}_side_classifier.pt"))
#     pd.DataFrame(cm, index=["true_left", "true_right"], columns=["pred_left", "pred_right"]).to_csv(
#         os.path.join(OUT_DIR, f"decoder_{tag}_confusion.csv")
#     )
#     fig, ax = plt.subplots(figsize=(5, 5))
#     im = ax.imshow(cm, cmap="Blues")
#     for i in range(2):
#         for j in range(2):
#             ax.text(j, i, str(cm[i, j]), ha="center", va="center")
#     ax.set_xticks([0, 1])
#     ax.set_xticklabels(LABEL_NAMES)
#     ax.set_yticks([0, 1])
#     ax.set_yticklabels(LABEL_NAMES)
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("True")
#     ax.set_title(f"{tag} | accuracy={accuracy:.3f}")
#     plt.colorbar(im)
#     plt.tight_layout()
#     plt.savefig(os.path.join(IMG_DIR, f"decoder_{tag}_confusion.png"), dpi=300)
#     plt.close()
#     cleanup(decoder, optimizer, Xtr_t, ytr_t, Xte_t)
#     return accuracy


# cebra_acc = train_decoder(X_train_cebra, y_train, X_test_cebra, y_test, "CEBRA")
# acorn_acc = train_decoder(X_train_acorn, y_train, X_test_acorn, y_test, "ACORN")

# comparison = pd.DataFrame({
#     "model": ["CEBRA", "ACORN"],
#     "test_accuracy": [cebra_acc, acorn_acc]
# })

# comparison.to_csv(os.path.join(OUT_DIR, "CEBRA_vs_ACORN_decoder_accuracy.csv"), index=False)

# print("\nDONE")
# print(f"Attribution trial: {ATTR_TRIAL_ID}")
# print(f"Train trials: {len(train_ids)} | Test trials: {len(test_ids)}")
# print(f"CEBRA accuracy: {cebra_acc:.4f}")
# print(f"ACORN accuracy: {acorn_acc:.4f}")

# cleanup(cebra_model, acorn_model)

##########
## 1FC ###
# import os
# import gc
# import sys
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import scipy.io as sio
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from utils.constants import CEBRA_DIR
# from utils.min_distance import min_l2_distance
# sys.path.insert(0, str(CEBRA_DIR))
# import cebra
# import cebra.attribution
# from cebra import CEBRA

# DATA_PATH = "./data/spk/M021519_spk.mat"
# BHV_PATH = "./data/behav/M021519_trialtype.csv"
# OUT_DIR = "./outputs"
# IMG_DIR = "./image"
# os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(IMG_DIR, exist_ok=True)

# BATCH_SIZE = 256
# MAX_ITER = 10000
# OUTPUT_DIM = 16
# PRE_MS = 500
# POST_MS = 1000
# BIN_MS = 10
# TEST_SIZE = 0.20
# MAX_TRIALS = None
# EPS_SAMPLE_SIZE = 2000
# RANDOM_SEED = 42
# SIDE_TO_LABEL = {"left": 0, "right": 1}
# LABEL_NAMES = ["left", "right"]
# DECODER_HIDDEN_DIM = 64
# DECODER_DROPOUT = 0.4
# DECODER_LR = 1e-3
# DECODER_WEIGHT_DECAY = 1e-4
# DECODER_EPOCHS = 50000
# DECODER_BATCH_SIZE = 32

# torch.manual_seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
# rng = np.random.default_rng(RANDOM_SEED)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mat = sio.loadmat(DATA_PATH, simplify_cells=True, squeeze_me=True)
# unit = mat["unit"]
# t_evt = mat["t_evt"]
# bhv = pd.read_csv(BHV_PATH)
# stim_times = np.asarray(t_evt["stim_on"], dtype=np.float32).reshape(-1)
# print(f"neurons: {len(unit)}")
# print(f"behavior trials: {len(bhv)}")
# print(f"device: {device}")

# def cleanup(*objects):
#     for obj in objects:
#         try:
#             del obj
#         except Exception:
#             pass
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()

# def make_trial_matrix(event_time):
#     n_neurons = len(unit)
#     n_bins = int((PRE_MS + POST_MS) / BIN_MS)
#     X = np.zeros((n_bins, n_neurons), dtype=np.float32)
#     start = event_time - PRE_MS / 1000.0
#     end = event_time + POST_MS / 1000.0
#     for n in range(n_neurons):
#         spikes = np.asarray(unit[n]["timestamps"], dtype=np.float32).reshape(-1)
#         spikes = spikes[(spikes >= start) & (spikes <= end)]
#         spikes_ms = (spikes - event_time) * 1000.0
#         bins = ((spikes_ms + PRE_MS) / BIN_MS).astype(int)
#         bins = bins[(bins >= 0) & (bins < n_bins)]
#         for b in bins:
#             X[b, n] += 1.0
#     return X

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
#     plt.imshow(arr, aspect="auto")
#     plt.colorbar(label="absolute attribution")
#     plt.xlabel("Neuron")
#     plt.ylabel("Latent dimension")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()

# valid_trial_ids = []
# y_all = []
# n_candidates = min(len(stim_times), len(bhv))
# for tid in range(n_candidates):
#     row = bhv.iloc[tid]
#     if str(row.get("task", "")).strip() != "1FC":
#         continue
#     brk = pd.to_numeric(row.get("brk", np.nan), errors="coerce")
#     if not np.isfinite(brk) or brk != 0:
#         continue
#     side = str(row.get("side_1FC", "")).strip().lower()
#     if side not in SIDE_TO_LABEL:
#         continue
#     valid_trial_ids.append(tid)
#     y_all.append(SIDE_TO_LABEL[side])
# valid_trial_ids = np.asarray(valid_trial_ids, dtype=int)
# y_all = np.asarray(y_all, dtype=np.int64)
# if MAX_TRIALS is not None:
#     valid_trial_ids = valid_trial_ids[:MAX_TRIALS]
#     y_all = y_all[:MAX_TRIALS]
# print(f"valid 1FC trials: {len(valid_trial_ids)} (left={np.sum(y_all == 0)}, right={np.sum(y_all == 1)})")

# (train_ids, test_ids, y_train, y_test) = train_test_split(
#     valid_trial_ids, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_all
# )
# print(f"train={len(train_ids)} | test={len(test_ids)}")

# train_raw_parts = []
# for tid in train_ids:
#     train_raw_parts.append(make_trial_matrix(float(stim_times[tid])))
# train_raw_concat = np.concatenate(train_raw_parts, axis=0)
# TRAIN_MU = train_raw_concat.mean(axis=0, keepdims=True).astype(np.float32)
# TRAIN_SIGMA = (train_raw_concat.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)

# def normalize(X):
#     return ((X - TRAIN_MU) / TRAIN_SIGMA).astype(np.float32)

# X_parts = []
# time_parts = []
# trial_parts = []
# for i, raw_X in enumerate(train_raw_parts):
#     X_t = normalize(raw_X)
#     X_parts.append(X_t)
#     time_parts.append(np.arange(len(X_t), dtype=np.float32))
#     trial_parts.append(np.full(len(X_t), i, dtype=np.int64))
# X_train = np.concatenate(X_parts, axis=0)
# time_labels = np.concatenate(time_parts).reshape(-1, 1)
# trial_labels = np.concatenate(trial_parts)
# print(f"encoder training data: {X_train.shape}")

# def train_model(X, adv=False):
#     name = "ACORN" if adv else "CEBRA"
#     sample_size = min(EPS_SAMPLE_SIZE, len(X))
#     sample_idx = rng.choice(len(X), size=sample_size, replace=False)
#     eps = float(min_l2_distance(X[sample_idx])) / 2.0
#     eps = max(eps, 1e-6)
#     print(f"training {name} | eps={eps:.5f}")
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
#     model.fit(X, time_labels, trial_labels)
#     return model

# cebra_model = train_model(X_train, adv=False)
# acorn_model = train_model(X_train, adv=True)

# ATTR_TRIAL_ID = int(train_ids[0])
# X_attr = normalize(train_raw_parts[0])

# def get_attribution(model, name, X_ref):
#     encoder = model.solver_.model.to(device)
#     if hasattr(encoder, "split_outputs"):
#         encoder.split_outputs = False
#     x_tensor = torch.tensor(X_ref, dtype=torch.float32, device=next(encoder.parameters()).device, requires_grad=True)
#     method = cebra.attribution.init(
#         name="jacobian-based-batched",
#         model=encoder,
#         input_data=x_tensor,
#         output_dimension=OUTPUT_DIM
#     )
#     result = method.compute_attribution_map(batch_size=128)
#     jf = result["jf"]
#     if "jf-inv-svd" in result:
#         jf_inv = result["jf-inv-svd"]
#     else:
#         jf_inv = result["jf-inv-lsq"]
#     torch.save(jf, os.path.join(OUT_DIR, f"M021519_trial{ATTR_TRIAL_ID}_{name}_jf.pt"))
#     torch.save(jf_inv, os.path.join(OUT_DIR, f"M021519_trial{ATTR_TRIAL_ID}_{name}_jf_inv.pt"))
#     save_heatmap(jf, os.path.join(IMG_DIR, f"{name}_jacobian.png"), f"{name} Jacobian")
#     save_heatmap(jf_inv, os.path.join(IMG_DIR, f"{name}_inverse_jacobian.png"), f"{name} inverse Jacobian")
#     cleanup(encoder, x_tensor, method, result)

# get_attribution(cebra_model, "CEBRA", X_attr)
# get_attribution(acorn_model, "ACORN", X_attr)

# def build_embeddings(trial_ids, model):
#     features = []
#     for tid in trial_ids:
#         raw_X = make_trial_matrix(float(stim_times[tid]))
#         X_t = normalize(raw_X)
#         emb = np.asarray(model.transform(X_t))
#         features.append(emb.mean(axis=0))
#     return np.stack(features).astype(np.float32)

# print("building CEBRA embeddings...")
# X_train_cebra = build_embeddings(train_ids, cebra_model)
# X_test_cebra = build_embeddings(test_ids, cebra_model)
# print("building ACORN embeddings...")
# X_train_acorn = build_embeddings(train_ids, acorn_model)
# X_test_acorn = build_embeddings(test_ids, acorn_model)

# class TwoLayerMLP(nn.Module):
#     def __init__(self, input_dim=16, hidden_dim=64, output_dim=2, dropout_rate=0.4):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim, output_dim)
#         )
#         for layer in self.net:
#             if isinstance(layer, nn.Linear):
#                 nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
#                 nn.init.constant_(layer.bias, 0)
#     def forward(self, x):
#         return self.net(x)

# def train_decoder(X_train_feats, y_train, X_test_feats, y_test, tag):
#     mu = X_train_feats.mean(axis=0, keepdims=True)
#     sigma = X_train_feats.std(axis=0, keepdims=True) + 1e-8
#     Xtr = (X_train_feats - mu) / sigma
#     Xte = (X_test_feats - mu) / sigma
#     Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
#     ytr_t = torch.tensor(y_train, dtype=torch.long, device=device)
#     Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
#     decoder = TwoLayerMLP(
#         input_dim=Xtr.shape[1],
#         hidden_dim=DECODER_HIDDEN_DIM,
#         output_dim=2,
#         dropout_rate=DECODER_DROPOUT
#     ).to(device)
#     optimizer = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR, weight_decay=DECODER_WEIGHT_DECAY)
#     loss_fn = nn.CrossEntropyLoss()
#     for epoch in range(DECODER_EPOCHS):
#         decoder.train()
#         perm = torch.randperm(len(Xtr_t), device=device)
#         for start in range(0, len(Xtr_t), DECODER_BATCH_SIZE):
#             idx = perm[start:start + DECODER_BATCH_SIZE]
#             optimizer.zero_grad()
#             logits = decoder(Xtr_t[idx])
#             loss = loss_fn(logits, ytr_t[idx])
#             loss.backward()
#             optimizer.step()
#     decoder.eval()
#     with torch.no_grad():
#         pred = decoder(Xte_t).argmax(dim=1).cpu().numpy()
#     accuracy = accuracy_score(y_test, pred)
#     chance = max(np.mean(y_test == 0), np.mean(y_test == 1))
#     cm = np.zeros((2, 2), dtype=int)
#     for true, p in zip(y_test, pred):
#         cm[true, p] += 1
#     print(f"{tag} | test accuracy={accuracy:.4f} | chance={chance:.4f}")
#     torch.save(decoder.state_dict(), os.path.join(OUT_DIR, f"decoder_{tag}_side_classifier.pt"))
#     pd.DataFrame(cm, index=["true_left", "true_right"], columns=["pred_left", "pred_right"]).to_csv(
#         os.path.join(OUT_DIR, f"decoder_{tag}_confusion.csv")
#     )
#     fig, ax = plt.subplots(figsize=(5, 5))
#     im = ax.imshow(cm, cmap="Blues")
#     for i in range(2):
#         for j in range(2):
#             ax.text(j, i, str(cm[i, j]), ha="center", va="center")
#     ax.set_xticks([0, 1])
#     ax.set_xticklabels(LABEL_NAMES)
#     ax.set_yticks([0, 1])
#     ax.set_yticklabels(LABEL_NAMES)
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("True")
#     ax.set_title(f"{tag} | accuracy={accuracy:.3f}")
#     plt.colorbar(im)
#     plt.tight_layout()
#     plt.savefig(os.path.join(IMG_DIR, f"decoder_{tag}_confusion.png"), dpi=300)
#     plt.close()
#     cleanup(decoder, optimizer, Xtr_t, ytr_t, Xte_t)
#     return accuracy


# cebra_acc = train_decoder(X_train_cebra, y_train, X_test_cebra, y_test, "CEBRA")
# acorn_acc = train_decoder(X_train_acorn, y_train, X_test_acorn, y_test, "ACORN")

# comparison = pd.DataFrame({
#     "model": ["CEBRA", "ACORN"],
#     "test_accuracy": [cebra_acc, acorn_acc]
# })

# comparison.to_csv(os.path.join(OUT_DIR, "CEBRA_vs_ACORN_decoder_accuracy.csv"), index=False)

# print("\nDONE")
# print(f"Attribution trial: {ATTR_TRIAL_ID}")
# print(f"Train trials: {len(train_ids)} | Test trials: {len(test_ids)}")
# print(f"CEBRA accuracy: {cebra_acc:.4f}")
# print(f"ACORN accuracy: {acorn_acc:.4f}")

# cleanup(cebra_model, acorn_model)

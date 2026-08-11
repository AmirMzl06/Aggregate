import os
import gc
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

import sys
sys.path.insert(0, str(CEBRA_DIR))

import cebra
from cebra import CEBRA


# =====================================================
# Config
# =====================================================
DATA_PATH = "./data/spk/M021519_spk.mat"
BHV_PATH = "./data/behav/M021519_trialtype.csv"

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

TEST_SIZE = 0.2          # 20% of trials held out, NEVER seen by the encoder or the decoder during training
MAX_TRIALS = None        # cap the number of usable trials for a quick test run, or None to use all

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- decoder config (binary classification: side_1FC left/right) ---
SIDE_TO_LABEL = {"left": 0, "right": 1}
LABEL_NAMES = ["left", "right"]

DECODER_HIDDEN_DIM = 64
DECODER_DROPOUT = 0.4
DECODER_LR = 1e-3
DECODER_WEIGHT_DECAY = 1e-4
DECODER_EPOCHS = 300
DECODER_BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# Load data
# =====================================================
mat = sio.loadmat(DATA_PATH, simplify_cells=True, squeeze_me=True)
unit = mat["unit"]
t_evt = mat["t_evt"]
print("neurons:", len(unit))
print("events:", t_evt.keys())

bhv = pd.read_csv(BHV_PATH)
print("behavior file loaded:", BHV_PATH, "| shape:", bhv.shape)

stim_times = np.asarray(t_evt["stim_on"])


# =====================================================
# Helpers
# =====================================================
def make_trial_matrix(unit, event_time, pre_ms, post_ms, bin_ms):
    n_neurons = len(unit)
    n_bins = int((pre_ms + post_ms) / bin_ms)
    X = np.zeros((n_bins, n_neurons), dtype=np.float32)

    start = event_time - pre_ms / 1000
    end = event_time + post_ms / 1000

    for n in range(n_neurons):
        spikes = unit[n]["timestamps"]
        spikes = spikes[(spikes >= start) & (spikes <= end)]
        spikes_ms = (spikes - event_time) * 1000
        bins = ((spikes_ms + pre_ms) / bin_ms).astype(int)
        bins = bins[(bins >= 0) & (bins < n_bins)]
        for b in bins:
            X[b, n] += 1
    return X


def normalize_trial(X):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    return ((X - mu) / sigma).astype(np.float32)


def save_heatmap(tensor, name):
    if torch.is_tensor(tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    if arr.ndim == 3:
        arr = np.abs(arr).mean(axis=0)
    else:
        arr = np.abs(arr)
    plt.figure(figsize=(10, 6))
    plt.imshow(arr, aspect="auto")
    plt.colorbar(label="absolute attribution")
    plt.xlabel("Neuron")
    plt.ylabel("Latent dimension")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/{name}.png", dpi=300)
    plt.close()
    print("saved:", f"{IMG_DIR}/{name}.png")


# =====================================================
# 1) Build the trial-level train/test split FIRST.
#    Everything downstream (encoder fitting AND decoder training) only
#    ever touches its own side of this split.
# =====================================================
valid_trial_ids = []
y_all = []

n_candidates = min(len(stim_times), len(bhv))
for tid in range(n_candidates):
    row = bhv.iloc[tid]
    if str(row.get("task", "")).strip() != "1FC":
        continue
    if pd.to_numeric(row.get("brk", 1), errors="coerce") != 0:
        continue
    side = str(row.get("side_1FC", "")).strip().lower()
    if side not in SIDE_TO_LABEL:
        continue
    valid_trial_ids.append(tid)
    y_all.append(SIDE_TO_LABEL[side])

valid_trial_ids = np.array(valid_trial_ids)
y_all = np.array(y_all)

print(f"\nFound {len(valid_trial_ids)} completed 1FC trials with a valid side_1FC label.")
print(f"  class balance -> left: {(y_all == 0).sum()}, right: {(y_all == 1).sum()}")

if MAX_TRIALS is not None:
    valid_trial_ids = valid_trial_ids[:MAX_TRIALS]
    y_all = y_all[:MAX_TRIALS]

train_ids, test_ids, y_train, y_test = train_test_split(
    valid_trial_ids, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_all
)
print(f"train trials: {len(train_ids)} | test trials: {len(test_ids)}  (test set untouched until final eval)")


def build_trial_matrices(trial_ids):
    """Per-trial spike matrix + per-trial z-score (identical recipe for every trial)."""
    X_list, label_list = [], []
    for tid in trial_ids:
        event_time = float(stim_times[tid])
        X_t = make_trial_matrix(unit, event_time, PRE_MS, POST_MS, BIN_MS)
        X_t = normalize_trial(X_t)
        X_list.append(X_t)
        label_list.append(np.arange(len(X_t)).astype(np.float32))
    return X_list, label_list


print("\nBuilding per-trial matrices for the TRAINING trials only...")
X_train_list, labels_train_list = build_trial_matrices(train_ids)


# =====================================================
# 2) Fit CEBRA / ACORN on the TRAIN trials as a *multisession* fit.
#    Passing a *list* of per-trial arrays (instead of one concatenated
#    array) makes CEBRA sample positive/negative pairs only *within*
#    each trial -- never across trial boundaries. This is what actually
#    fixes the single-trial generalization problem.
#
#    NOTE: multisession fit/transform is standard CEBRA behavior. If your
#    CEBRA_DIR fork changed this API (e.g. .fit expects something other
#    than a plain list, or .transform needs a different session-id kwarg),
#    check that before trusting the numbers below.
# =====================================================
def train_cebra_multisession(X_list, labels_list, adv=False):
    mode = "adversarial" if adv else "clean"

    X_concat_for_eps = np.concatenate(X_list, axis=0)
    eps = float(min_l2_distance(X_concat_for_eps)) / 2
    eps = max(eps, 1e-6)
    print(f"\nTraining {mode} (multisession, {len(X_list)} trials) | epsilon: {eps:.5f}")

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

    model.fit(X_list, labels_list)
    return model, eps


cebra_model, cebra_eps = train_cebra_multisession(X_train_list, labels_train_list, adv=False)
acorn_model, acorn_eps = train_cebra_multisession(X_train_list, labels_train_list, adv=True)


# =====================================================
# 3) Attribution on a representative TRAINING trial (never a test trial)
# =====================================================
ATTR_TRIAL_ID = int(train_ids[0])
X_attr = X_train_list[0]
print(f"\nUsing training trial {ATTR_TRIAL_ID} for Jacobian attribution.")


def get_attribution(model, name, X_ref):
    encoder = model.solver_.model.to(device)
    enc_device = next(encoder.parameters()).device
    x_tensor = torch.tensor(X_ref, dtype=torch.float32, device=enc_device, requires_grad=True)

    method = cebra.attribution.init(
        name="jacobian-based-batched", model=encoder, input_data=x_tensor, output_dimension=OUTPUT_DIM
    )
    result = method.compute_attribution_map(batch_size=128)
    print(f"[{name}] attribution keys:", result.keys())

    jf = result["jf"]
    jf_inv = result["jf-inv-svd"]

    torch.save(jf, f"{OUT_DIR}/M021519_trial{ATTR_TRIAL_ID}_{name}_jf.pt")
    torch.save(jf_inv, f"{OUT_DIR}/M021519_trial{ATTR_TRIAL_ID}_{name}_jf_inv.pt")

    save_heatmap(jf, name + "_jacobian")
    save_heatmap(jf_inv, name + "_inverse_jacobian")

    del encoder, x_tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


get_attribution(cebra_model, "CEBRA", X_attr)
get_attribution(acorn_model, "ACORN", X_attr)


# =====================================================
# 4) Decoder features: transform TRAIN and TEST trials through the
#    already-fixed (frozen) encoders. Neither encoder was ever fit on
#    the test trials, so this is a genuine held-out evaluation.
# =====================================================
def transform_embedding(model, X_t):
    try:
        emb = model.transform(X_t, session_id=0)
    except TypeError:
        emb = model.transform(X_t)
    return np.asarray(emb)


def build_decoder_features(trial_ids):
    feats = []
    for tid in trial_ids:
        event_time = float(stim_times[tid])
        X_t = make_trial_matrix(unit, event_time, PRE_MS, POST_MS, BIN_MS)
        X_t = normalize_trial(X_t)

        cebra_pooled = transform_embedding(cebra_model, X_t).mean(axis=0)
        acorn_pooled = transform_embedding(acorn_model, X_t).mean(axis=0)

        feats.append(np.concatenate([cebra_pooled, acorn_pooled]))
    return np.stack(feats).astype(np.float32)


print("\nEmbedding TRAIN trials for the decoder...")
X_train_feats = build_decoder_features(train_ids)
print("Embedding TEST trials for the decoder...")
X_test_feats = build_decoder_features(test_ids)
print("X_train_feats:", X_train_feats.shape, "| X_test_feats:", X_test_feats.shape)


# =====================================================
# 5) Decoder (TwoLayerMLP) as a binary classifier -> accuracy
# =====================================================
class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=2, dropout_rate=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.net(x)


def train_decoder_classifier(X_train_feats, y_train, X_test_feats, y_test):
    feat_mu = X_train_feats.mean(axis=0, keepdims=True)
    feat_sigma = X_train_feats.std(axis=0, keepdims=True) + 1e-8
    Xtr = (X_train_feats - feat_mu) / feat_sigma
    Xte = (X_test_feats - feat_mu) / feat_sigma

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(y_train, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)

    decoder = TwoLayerMLP(
        input_dim=Xtr.shape[1], hidden_dim=DECODER_HIDDEN_DIM, output_dim=2, dropout_rate=DECODER_DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR, weight_decay=DECODER_WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    n_train = Xtr_t.shape[0]
    print(f"\nTraining decoder classifier | train: {n_train} | test: {Xte_t.shape[0]}")

    for epoch in range(DECODER_EPOCHS):
        decoder.train()
        perm = torch.randperm(n_train)
        epoch_loss, correct = 0.0, 0

        for i in range(0, n_train, DECODER_BATCH_SIZE):
            idx = perm[i:i + DECODER_BATCH_SIZE]
            xb, yb = Xtr_t[idx], ytr_t[idx]

            optimizer.zero_grad()
            logits = decoder(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            correct += (logits.argmax(dim=1) == yb).sum().item()

        epoch_loss /= n_train
        train_acc = correct / n_train
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  [decoder] epoch {epoch + 1}/{DECODER_EPOCHS} | train loss {epoch_loss:.4f} | train acc {train_acc:.3f}")

    decoder.eval()
    with torch.no_grad():
        test_logits = decoder(Xte_t).cpu().numpy()
    y_pred = test_logits.argmax(axis=1)

    test_acc = accuracy_score(y_test, y_pred)
    chance = max((y_test == 0).mean(), (y_test == 1).mean())
    print(f"\nDecoder TEST accuracy: {test_acc:.4f}  (chance level = {chance:.3f})")

    torch.save(decoder.state_dict(), os.path.join(OUT_DIR, "decoder_side_classifier_state_dict.pt"))

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_test, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"side_1FC decode | test acc={test_acc:.3f}")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "decoder_side_confusion.png"), dpi=300)
    plt.close()
    print("saved:", os.path.join(IMG_DIR, "decoder_side_confusion.png"))

    return decoder, test_acc


decoder, test_acc = train_decoder_classifier(X_train_feats, y_train, X_test_feats, y_test)


del cebra_model, acorn_model, decoder
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nDONE")
print(f"Attribution trial: {ATTR_TRIAL_ID}")
print(f"Encoder trained on {len(train_ids)} trials, evaluated decoder on {len(test_ids)} held-out trials")
print(f"Decoder TEST accuracy: {test_acc:.4f}")

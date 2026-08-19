import os
import gc
import sys
import glob
import random
import numpy as np
import pandas as pd

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TRITON_DISABLE"] = "1"

import torch
import torch.nn as nn
import scipy.io as sio
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.constants import CEBRA_DIR
from utils.min_distance import min_l2_distance
sys.path.insert(0, str(CEBRA_DIR))
import cebra
import cebra.models.layers as cebra_layers
from cebra import CEBRA

SPK_DIR = "./data/spk"
BHV_DIR = "./data/behav"
OUT_DIR = "./outputs_multi_session"
os.makedirs(OUT_DIR, exist_ok=True)

PREPROCESS_MODE = "smooth"
SMOOTH_SIGMA_MS = 100.0
MAX_SESSIONS = None
TASK_NAME = "2afc"
REQUIRE_BRK_ZERO = True
PRE_MS = 500
POST_MS = 1000
BIN_MS = 10
SEQ_LEN = int((PRE_MS + POST_MS) / BIN_MS)
HIDDEN_DIM = 256
OUTPUT_DIM = 16
NUM_SHARED_BLOCKS = 16
DROPOUT = 0.1
OFFSET_LEFT = 18
OFFSET_RIGHT = 18
BATCH_SIZE = 1024
TOTAL_STEPS = 10000
NUM_SESSIONS_PER_ITER = 1
TEMPERATURE = 0.4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
ADV_STEPS = 10
ATTACK_NORM = "linf"
RUN_CEBRA = True
RUN_ACORN = True
EPS_SAMPLE_SIZE = 2000
MIN_TRIALS_PER_SESSION = 10
TEST_SIZE = 0.20
SIDE_TO_LABEL = {"left": 0, "right": 1}
EXTRACT_BATCH_SIZE = 512
EVAL_BATCH_SIZE = 1024
DECODER_HIDDEN_DIM = 64
DECODER_NUM_LAYERS = 2
DECODER_DROPOUT = 0.4
DECODER_LR = 1e-3
DECODER_WEIGHT_DECAY = 1e-4
DECODER_EPOCHS = 5000
DECODER_BATCH_SIZE = 256
PRINT_EVERY = 5
RANDOM_SEED = 42

N_COMPARE_SESSIONS = 10
SINGLE_CEBRA_BATCH_SIZE = 256
SINGLE_CEBRA_MAX_ITER = 10000

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)
print("PREPROCESS_MODE:", PREPROCESS_MODE)
if PREPROCESS_MODE not in {"raw", "normalize", "smooth"}:
    raise ValueError("PREPROCESS_MODE must be one of: 'raw', 'normalize', 'smooth'")
if SMOOTH_SIGMA_MS <= 0:
    raise ValueError("SMOOTH_SIGMA_MS must be > 0.")
SMOOTH_SIGMA_BINS = SMOOTH_SIGMA_MS / BIN_MS
print(f"Gaussian sigma: {SMOOTH_SIGMA_MS:.1f} ms = {SMOOTH_SIGMA_BINS:.2f} bins")

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

def preprocess_trial(X):
    X = np.asarray(X, dtype=np.float32).copy()
    if PREPROCESS_MODE == "raw":
        return X
    if PREPROCESS_MODE == "normalize":
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True) + 1e-8
        return ((X - mu) / sigma).astype(np.float32)
    if PREPROCESS_MODE == "smooth":
        for neuron_idx in range(X.shape[1]):
            X[:, neuron_idx] = gaussian_filter1d(X[:, neuron_idx], sigma=SMOOTH_SIGMA_BINS, mode="reflect")
        return X.astype(np.float32)

class SessionLayer(nn.Module):
    def __init__(self, num_units, kernel=2):
        super().__init__()
        self.num_units = num_units
        self.kernel = kernel
        self.session_dict = nn.ModuleDict()
    def add_session(self, session_name, num_neurons):
        self.session_dict.add_module(session_name, nn.Conv1d(num_neurons, self.num_units, self.kernel))
    def forward(self, x, session_name):
        if session_name not in self.session_dict:
            raise KeyError(f"Unknown session '{session_name}'. Registered: {list(self.session_dict.keys())}")
        return self.session_dict[session_name](x)

class Offset36Multi(nn.Module):
    def _make_layers(self, num_units, dropout, n):
        return [cebra_layers._Skip(nn.Dropout1d(p=dropout), nn.Conv1d(num_units, num_units, 3), nn.GELU()) for _ in range(n)]
    def __init__(self, num_units, num_outputs, normalize_output=True):
        super().__init__()
        self.num_units = num_units
        self.num_output = num_outputs
        self.session_layer = SessionLayer(num_units, 2)
        self.first_net = nn.Sequential(nn.Dropout1d(p=0.1), nn.GELU(), *self._make_layers(num_units, DROPOUT, NUM_SHARED_BLOCKS))
        self.last_layer_multi = SessionLayer(num_outputs, 3)
        last_layers = []
        if normalize_output:
            last_layers.append(cebra_layers._Norm())
        last_layers.append(cebra_layers.Squeeze())
        self.last_net = nn.Sequential(*last_layers)
    def add_session(self, session_name, num_neurons):
        self.session_layer.add_session(session_name, num_neurons)
        self.last_layer_multi.add_session(session_name, self.num_units)
    def forward(self, x, session_name):
        x = self.session_layer(x, session_name)
        x = self.first_net(x)
        x = self.last_layer_multi(x, session_name)
        x = self.last_net(x)
        return x
    def get_offset(self):
        return cebra.data.Offset(OFFSET_LEFT, OFFSET_RIGHT)

def make_trial_matrix(unit, event_time):
    n_neurons = len(unit)
    X = np.zeros((SEQ_LEN, n_neurons), dtype=np.float32)
    start = event_time - PRE_MS / 1000.0
    end = event_time + POST_MS / 1000.0
    for neuron_idx in range(n_neurons):
        spikes = np.asarray(unit[neuron_idx]["timestamps"], dtype=np.float32).reshape(-1)
        spikes = spikes[(spikes >= start) & (spikes <= end)]
        spikes_ms = (spikes - event_time) * 1000.0
        bins = ((spikes_ms + PRE_MS) / BIN_MS).astype(int)
        bins = bins[(bins >= 0) & (bins < SEQ_LEN)]
        for b in bins:
            X[b, neuron_idx] += 1.0
    return X

def load_session(spk_path):
    session_name = os.path.basename(spk_path).replace("_spk.mat", "")
    bhv_path = os.path.join(BHV_DIR, session_name + "_trialtype.csv")
    if not os.path.exists(bhv_path):
        print(f"[SKIP] {session_name}: behavior file not found.")
        return None
    print("\n" + "=" * 80)
    print("LOADING SESSION:", session_name)
    print("=" * 80)
    mat = sio.loadmat(spk_path, simplify_cells=True, squeeze_me=True)
    unit = mat["unit"]
    t_evt = mat["t_evt"]
    stim_times = np.asarray(t_evt["stim_on"], dtype=np.float32).reshape(-1)
    bhv = pd.read_csv(bhv_path)
    print("Neurons:", len(unit))
    print("Behavior rows:", len(bhv))
    return {"name": session_name, "unit": unit, "stim_times": stim_times, "bhv": bhv}

def get_valid_trials_with_labels(session):
    stim_times = session["stim_times"]
    bhv = session["bhv"]
    valid_ids = []
    labels = []
    n_candidates = min(len(stim_times), len(bhv))
    for tid in range(n_candidates):
        row = bhv.iloc[tid]
        task = str(row.get("task", "")).strip().lower()
        if task != TASK_NAME:
            continue
        if REQUIRE_BRK_ZERO:
            brk = pd.to_numeric(row.get("brk", np.nan), errors="coerce")
            if not np.isfinite(brk) or brk != 0:
                continue
        side = str(row.get("chosenside_2AFC", "")).strip().lower()
        if side not in SIDE_TO_LABEL:
            continue
        valid_ids.append(tid)
        labels.append(SIDE_TO_LABEL[side])
    return np.asarray(valid_ids, dtype=np.int64), np.asarray(labels, dtype=np.int64)

def prepare_session(session):
    name = session["name"]
    unit = session["unit"]
    stim_times = session["stim_times"]
    valid_ids, y_all = get_valid_trials_with_labels(session)
    if len(valid_ids) < MIN_TRIALS_PER_SESSION:
        print(f"[SKIP] {name}: only {len(valid_ids)} valid labeled 2AFC trials.")
        return None
    try:
        train_ids, test_ids, y_train, y_test = train_test_split(
            valid_ids, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_all
        )
    except ValueError:
        train_ids, test_ids, y_train, y_test = train_test_split(
            valid_ids, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
    print(f"{name}: total={len(valid_ids)} | train={len(train_ids)} | test={len(test_ids)}")
    train_trials = [preprocess_trial(make_trial_matrix(unit, float(stim_times[tid]))) for tid in train_ids]
    test_trials = [preprocess_trial(make_trial_matrix(unit, float(stim_times[tid]))) for tid in test_ids]
    neural = np.concatenate(train_trials, axis=0)
    continuous = np.concatenate([np.arange(len(X), dtype=np.float32) for X in train_trials]).reshape(-1, 1)
    discrete = np.concatenate([np.full(len(X), i, dtype=np.int64) for i, X in enumerate(train_trials)])
    sample_n = min(EPS_SAMPLE_SIZE, len(neural))
    sample_idx = rng.choice(len(neural), size=sample_n, replace=False)
    eps = float(min_l2_distance(neural[sample_idx])) / 2.0
    eps = max(eps, 1e-6)
    print(f"{name}: neural={neural.shape} | continuous={continuous.shape} | discrete={discrete.shape} | eps={eps:.5f}")
    return {
        "name": name,
        "unit": unit,
        "stim_times": stim_times,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "y_train": y_train,
        "y_test": y_test,
        "train_trials": train_trials,
        "test_trials": test_trials,
        "neural": neural.astype(np.float32),
        "continuous": continuous.astype(np.float32),
        "discrete": discrete.astype(np.int64),
        "n_neurons": neural.shape[1],
        "eps": eps,
    }

def create_dataset(session_data, model):
    return cebra.data.TensorDataset(neural=session_data["neural"], continuous=session_data["continuous"], discrete=session_data["discrete"], offset=model.get_offset(), device=DEVICE)

def create_loader(session_data, model):
    dataset = create_dataset(session_data, model)
    loader = cebra.data.single_session.MixedDataLoader(dataset=dataset, time_offset=OFFSET_RIGHT, num_steps=TOTAL_STEPS, batch_size=BATCH_SIZE, conditional="time_delta")
    return dataset, iter(loader)

def choose_sessions(step, n_sessions, sessions_per_iter):
    first = (step * sessions_per_iter) % n_sessions
    return [(first + i) % n_sessions for i in range(sessions_per_iter)]

def normalize_l2(x):
    norm = x.norm(dim=-1, keepdim=True) + 1e-12
    return x / norm

def project_l2_ball(x_adv, x_ref, epsilon):
    delta = x_adv - x_ref
    norm = delta.norm(dim=-1, keepdim=True) + 1e-12
    factor = torch.clamp(epsilon / norm, max=1.0)
    return x_ref + delta * factor

def train_multisession_model(sessions, model_name, session_eps, adversarial=False):
    print("\n" + "=" * 90)
    print(f"TRAINING {model_name} | sessions={len(sessions)}")
    print("=" * 90)
    model = Offset36Multi(num_units=HIDDEN_DIM, num_outputs=OUTPUT_DIM, normalize_output=True)
    for session in sessions:
        model.add_session(session["name"], session["n_neurons"])
        print(f"Registered {session['name']} | neurons={session['n_neurons']} | eps={session_eps[session['name']]:.5f}")
    model = model.to(DEVICE)
    loaders = []
    for session in sessions:
        dataset, loader = create_loader(session, model)
        session["dataset"] = dataset
        session["loader"] = loader
        loaders.append(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY)
    criterion = cebra.models.FixedCosineInfoNCE(temperature=TEMPERATURE).to(DEVICE)
    progress = tqdm(range(TOTAL_STEPS), desc=model_name)
    for step in progress:
        selected = choose_sessions(step, len(sessions), NUM_SESSIONS_PER_ITER)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batches = []
        session_names = []
        total_loss = torch.tensor(0.0, device=DEVICE)
        for idx in selected:
            session = sessions[idx]
            name = session["name"]
            batch = next(session["loader"])
            batch.to(DEVICE)
            reference = model(batch.reference, name)
            positive = model(batch.positive, name)
            negative = model(batch.negative, name)
            loss, _, _ = criterion(reference, positive, negative)
            total_loss = total_loss + loss / len(selected)
            batches.append(batch)
            session_names.append(name)
        total_loss.backward()
        optimizer.step()
        if adversarial:
            optimizer.zero_grad(set_to_none=True)
            adv_batches = []
            for batch, name in zip(batches, session_names):
                eps = session_eps[name]
                alpha = eps / 5.0
                x_ref = batch.reference.detach()
                noise = normalize_l2(torch.randn_like(x_ref))
                radius = torch.rand((x_ref.shape[0], 1, 1), device=DEVICE) * eps
                x_adv = (x_ref + noise * radius).detach()
                x_adv.requires_grad_(True)
                for _ in range(ADV_STEPS):
                    r_adv = model(x_adv, name)
                    p = model(batch.positive, name)
                    n = model(batch.negative, name)
                    adv_loss, _, _ = criterion(r_adv, p, n)
                    grad_x = torch.autograd.grad(adv_loss, x_adv, retain_graph=False, create_graph=False)[0]
                    with torch.no_grad():
                        x_adv = x_adv + alpha * normalize_l2(grad_x)
                        x_adv = project_l2_ball(x_adv, x_ref, eps)
                    x_adv.requires_grad_(True)
                adv_batches.append((x_adv.detach(), batch, name))
            total_adv_loss = torch.tensor(0.0, device=DEVICE)
            for x_adv, batch, name in adv_batches:
                r_adv = model(x_adv, name)
                p = model(batch.positive, name)
                n = model(batch.negative, name)
                adv_loss, _, _ = criterion(r_adv, p, n)
                total_adv_loss = total_adv_loss + adv_loss / len(adv_batches)
            total_adv_loss.backward()
            optimizer.step()
        progress.set_postfix(loss=f"{float(total_loss.detach().cpu()):.4f}")
    return model

def transform_batch(model, session_name, X_batch_np):
    x = torch.tensor(X_batch_np, dtype=torch.float32, device=DEVICE)
    x = x.permute(0, 2, 1)
    with torch.no_grad():
        out = model(x, session_name)
    return out.permute(0, 2, 1).cpu().numpy()

def extract_session_embeddings(model, session, trials):
    model.eval()
    embeddings = []
    for start in range(0, len(trials), EXTRACT_BATCH_SIZE):
        chunk = trials[start:start + EXTRACT_BATCH_SIZE]
        X_batch = np.stack(chunk).astype(np.float32)
        emb = transform_batch(model, session["name"], X_batch)
        embeddings.append(emb)
    return np.concatenate(embeddings, axis=0).astype(np.float32)

def build_pooled_split(model, sessions, split):
    X_parts, y_parts = [], []
    session_slices = {}
    cursor = 0
    for session in tqdm(sessions, desc=f"embed[{split}]"):
        if split == "train":
            trials, y = session["train_trials"], session["y_train"]
        else:
            trials, y = session["test_trials"], session["y_test"]
        if len(trials) == 0:
            continue
        emb = extract_session_embeddings(model, session, trials)
        X_parts.append(emb)
        y_parts.append(y)
        n = len(trials)
        session_slices[session["name"]] = (cursor, cursor + n)
        cursor += n
    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int64)
    return X, y, session_slices

class GRUDecoder(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=2, num_layers=2, dropout_rate=0.4, bidirectional=False):
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
            nn.Linear(gru_out_dim, output_dim),
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

def batched_predict(decoder, X_tensor, batch_size):
    decoder.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            chunk = X_tensor[start:start + batch_size]
            logits = decoder(chunk)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)

def train_decoder(X_train_feats, y_train, X_test_feats, y_test, tag):
    mu = X_train_feats.mean(axis=(0, 1), keepdims=True)
    sigma = X_train_feats.std(axis=(0, 1), keepdims=True) + 1e-8
    Xtr = (X_train_feats - mu) / sigma
    Xte = (X_test_feats - mu) / sigma
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
    decoder = GRUDecoder(
        input_dim=Xtr.shape[2],
        hidden_dim=DECODER_HIDDEN_DIM,
        output_dim=2,
        num_layers=DECODER_NUM_LAYERS,
        dropout_rate=DECODER_DROPOUT,
        bidirectional=False,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR, weight_decay=DECODER_WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    n_train = len(Xtr_t)
    for epoch in range(DECODER_EPOCHS):
        decoder.train()
        perm = torch.randperm(n_train, device=DEVICE)
        for start in range(0, n_train, DECODER_BATCH_SIZE):
            idx = perm[start:start + DECODER_BATCH_SIZE]
            optimizer.zero_grad()
            logits = decoder(Xtr_t[idx])
            loss = loss_fn(logits, ytr_t[idx])
            loss.backward()
            optimizer.step()
        if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0:
            train_pred = batched_predict(decoder, Xtr_t, EVAL_BATCH_SIZE)
            train_acc = accuracy_score(y_train, train_pred)
            print(f"  [{tag}] epoch {epoch + 1}/{DECODER_EPOCHS} | train_acc={train_acc:.4f}")
    pred = batched_predict(decoder, Xte_t, EVAL_BATCH_SIZE)
    accuracy = accuracy_score(y_test, pred)
    chance = max(np.mean(y_test == 0), np.mean(y_test == 1))
    print(f"\n{tag} | test accuracy={accuracy:.4f} | chance={chance:.4f}")
    cleanup(Xtr_t, ytr_t, Xte_t, optimizer)
    return accuracy, decoder, mu, sigma

def evaluate_decoder_on_subset(decoder, mu, sigma, X_pooled, y_pooled, start, end):
    X_sub = X_pooled[start:end]
    y_sub = y_pooled[start:end]
    X_sub_norm = ((X_sub - mu) / sigma).astype(np.float32)
    X_sub_t = torch.tensor(X_sub_norm, dtype=torch.float32, device=DEVICE)
    pred = batched_predict(decoder, X_sub_t, EVAL_BATCH_SIZE)
    cleanup(X_sub_t)
    return accuracy_score(y_sub, pred)

def train_cebra_single_session(session, adv):
    X_list = session["train_trials"]
    time_parts = [np.arange(len(X), dtype=np.float32) for X in X_list]
    trial_parts = [np.full(len(X), i, dtype=np.int64) for i, X in enumerate(X_list)]
    X_concat = np.concatenate(X_list, axis=0)
    time_labels = np.concatenate(time_parts).reshape(-1, 1)
    trial_labels = np.concatenate(trial_parts)
    sample_n = min(EPS_SAMPLE_SIZE, len(X_concat))
    sample_idx = rng.choice(len(X_concat), size=sample_n, replace=False)
    eps = float(min_l2_distance(X_concat[sample_idx])) / 2.0
    eps = max(eps, 1e-6)
    model = CEBRA(
        batch_size=min(SINGLE_CEBRA_BATCH_SIZE, len(X_concat)),
        temperature=0.4,
        model_architecture="offset10-model",
        time_offsets=10,
        max_iterations=SINGLE_CEBRA_MAX_ITER,
        output_dimension=OUTPUT_DIM,
        verbose=False,
        training_mode="adversarial" if adv else "clean",
        adv_alpha=eps / 5 if adv else 0,
        adv_epsilon=eps if adv else 0,
        adv_steps=10 if adv else 0,
        attack_norm="linf",
        num_hidden_units=32,
    )
    model.fit(X_concat, time_labels, trial_labels)
    return model, eps

def build_single_session_embeddings(model, trials):
    features = [np.asarray(model.transform(X)) for X in trials]
    return np.stack(features).astype(np.float32)

spk_files = sorted(glob.glob(os.path.join(SPK_DIR, "X*_spk.mat")))
if len(spk_files) == 0:
    raise RuntimeError(f"No X*_spk.mat files found in {SPK_DIR}")
random.seed(RANDOM_SEED)
if MAX_SESSIONS is not None and MAX_SESSIONS < len(spk_files):
    selected_spk = random.sample(spk_files, MAX_SESSIONS)
else:
    selected_spk = spk_files
selected_spk = sorted(selected_spk)
print("\nSELECTED SESSIONS:")
for path in selected_spk:
    print(" ", os.path.basename(path))

sessions = []
for spk_path in selected_spk:
    session = load_session(spk_path)
    if session is None:
        continue
    prepared = prepare_session(session)
    if prepared is not None:
        sessions.append(prepared)
if len(sessions) == 0:
    raise RuntimeError("No usable sessions.")
print("\nUsable sessions:", len(sessions))

session_eps = {s["name"]: s["eps"] for s in sessions}
session_lookup = {s["name"]: s for s in sessions}

cebra_decoder = acorn_decoder = None
cebra_mu = cebra_sigma = acorn_mu = acorn_sigma = None
X_test_cebra = y_test_cebra = cebra_test_slices = None
X_test_acorn = y_test_acorn = acorn_test_slices = None
cebra_pooled_acc = acorn_pooled_acc = None

if RUN_CEBRA:
    cebra_model = train_multisession_model(sessions, model_name="CEBRA", session_eps=session_eps, adversarial=False)
    X_train_cebra, y_train_cebra, _ = build_pooled_split(cebra_model, sessions, "train")
    X_test_cebra, y_test_cebra, cebra_test_slices = build_pooled_split(cebra_model, sessions, "test")
    print(f"CEBRA embeddings | train={X_train_cebra.shape} | test={X_test_cebra.shape}")
    cleanup(cebra_model)
    cebra_pooled_acc, cebra_decoder, cebra_mu, cebra_sigma = train_decoder(
        X_train_cebra, y_train_cebra, X_test_cebra, y_test_cebra, "CEBRA"
    )
    cleanup(X_train_cebra, y_train_cebra)

if RUN_ACORN:
    acorn_model = train_multisession_model(sessions, model_name="ACORN", session_eps=session_eps, adversarial=True)
    X_train_acorn, y_train_acorn, _ = build_pooled_split(acorn_model, sessions, "train")
    X_test_acorn, y_test_acorn, acorn_test_slices = build_pooled_split(acorn_model, sessions, "test")
    print(f"ACORN embeddings | train={X_train_acorn.shape} | test={X_test_acorn.shape}")
    cleanup(acorn_model)
    acorn_pooled_acc, acorn_decoder, acorn_mu, acorn_sigma = train_decoder(
        X_train_acorn, y_train_acorn, X_test_acorn, y_test_acorn, "ACORN"
    )
    cleanup(X_train_acorn, y_train_acorn)

comparison = pd.DataFrame({"model": ["CEBRA", "ACORN"], "test_accuracy": [cebra_pooled_acc, acorn_pooled_acc]})
comparison_path = os.path.join(OUT_DIR, "CEBRA_vs_ACORN_multisession_gru_accuracy.csv")
comparison.to_csv(comparison_path, index=False)

compare_rng = random.Random(RANDOM_SEED)
n_pick = min(N_COMPARE_SESSIONS, len(sessions))
compare_session_names = compare_rng.sample([s["name"] for s in sessions], n_pick)
print("\n" + "=" * 90)
print(f"SESSIONS SELECTED FOR MULTI vs SINGLE COMPARISON ({n_pick}):")
for nm in compare_session_names:
    print(" ", nm)
print("=" * 90)

rows = []

print("\n" + "=" * 90)
print("PART A: per-session accuracy of the MULTI-session model + pooled decoder")
print("=" * 90)
for name in compare_session_names:
    if RUN_CEBRA and cebra_decoder is not None:
        start, end = cebra_test_slices[name]
        acc = evaluate_decoder_on_subset(cebra_decoder, cebra_mu, cebra_sigma, X_test_cebra, y_test_cebra, start, end)
        print(f"  [multi][CEBRA] {name}: acc={acc:.4f} (n_test={end - start})")
        rows.append({"session": name, "model": "CEBRA", "pipeline": "multi", "accuracy": acc})
    if RUN_ACORN and acorn_decoder is not None:
        start, end = acorn_test_slices[name]
        acc = evaluate_decoder_on_subset(acorn_decoder, acorn_mu, acorn_sigma, X_test_acorn, y_test_acorn, start, end)
        print(f"  [multi][ACORN] {name}: acc={acc:.4f} (n_test={end - start})")
        rows.append({"session": name, "model": "ACORN", "pipeline": "multi", "accuracy": acc})

cleanup(cebra_decoder, acorn_decoder, X_test_cebra, y_test_cebra, X_test_acorn, y_test_acorn)

print("\n" + "=" * 90)
print("PART B: fresh SINGLE-session CEBRA/ACORN + decoder, same train/test split per session")
print("\n" + "=" * 90)
print("PART B: fresh SINGLE-session Offset36Multi (same architecture) + decoder, same train/test split per session")
print("=" * 90)
for name in compare_session_names:
    session = session_lookup[name]
    print(f"\n--- session {name} ---")

    if RUN_CEBRA:
        model_s = train_multisession_model([session], model_name=f"CEBRA[{name}]", session_eps=session_eps, adversarial=False)
        X_tr = extract_session_embeddings(model_s, session, session["train_trials"])
        X_te = extract_session_embeddings(model_s, session, session["test_trials"])
        cleanup(model_s)
        acc, dec_s, mu_s, sigma_s = train_decoder(X_tr, session["y_train"], X_te, session["y_test"], f"CEBRA[{name}]")
        print(f"  [single][CEBRA] {name}: acc={acc:.4f}")
        rows.append({"session": name, "model": "CEBRA", "pipeline": "single", "accuracy": acc})
        cleanup(dec_s, X_tr, X_te)

    if RUN_ACORN:
        model_s = train_multisession_model([session], model_name=f"ACORN[{name}]", session_eps=session_eps, adversarial=True)
        X_tr = extract_session_embeddings(model_s, session, session["train_trials"])
        X_te = extract_session_embeddings(model_s, session, session["test_trials"])
        cleanup(model_s)
        acc, dec_s, mu_s, sigma_s = train_decoder(X_tr, session["y_train"], X_te, session["y_test"], f"ACORN[{name}]")
        print(f"  [single][ACORN] {name}: acc={acc:.4f}")
        rows.append({"session": name, "model": "ACORN", "pipeline": "single", "accuracy": acc})
        cleanup(dec_s, X_tr, X_te)
# print("=" * 90)
# for name in compare_session_names:
#     session = session_lookup[name]
#     print(f"\n--- session {name} ---")

#     if RUN_CEBRA:
#         model_s, eps_s = train_cebra_single_session(session, adv=False)
#         X_tr = build_single_session_embeddings(model_s, session["train_trials"])
#         X_te = build_single_session_embeddings(model_s, session["test_trials"])
#         cleanup(model_s)
#         acc, dec_s, mu_s, sigma_s = train_decoder(X_tr, session["y_train"], X_te, session["y_test"], f"CEBRA[{name}]")
#         print(f"  [single][CEBRA] {name}: acc={acc:.4f} | eps={eps_s:.5f}")
#         rows.append({"session": name, "model": "CEBRA", "pipeline": "single", "accuracy": acc})
#         cleanup(dec_s, X_tr, X_te)

#     if RUN_ACORN:
#         model_s, eps_s = train_cebra_single_session(session, adv=True)
#         X_tr = build_single_session_embeddings(model_s, session["train_trials"])
#         X_te = build_single_session_embeddings(model_s, session["test_trials"])
#         cleanup(model_s)
#         acc, dec_s, mu_s, sigma_s = train_decoder(X_tr, session["y_train"], X_te, session["y_test"], f"ACORN[{name}]")
#         print(f"  [single][ACORN] {name}: acc={acc:.4f} | eps={eps_s:.5f}")
#         rows.append({"session": name, "model": "ACORN", "pipeline": "single", "accuracy": acc})
#         cleanup(dec_s, X_tr, X_te)

compare_df = pd.DataFrame(rows)
compare_path = os.path.join(OUT_DIR, "session_comparison_multi_vs_single.csv")
compare_df.to_csv(compare_path, index=False)

print("\n" + "=" * 90)
print("ALL DONE")
print("=" * 90)
print("Sessions used (encoder training pool):", len(sessions))
print("Preprocessing:", PREPROCESS_MODE)
if cebra_pooled_acc is not None:
    print(f"CEBRA pooled (all sessions) accuracy: {cebra_pooled_acc:.4f}")
if acorn_pooled_acc is not None:
    print(f"ACORN pooled (all sessions) accuracy: {acorn_pooled_acc:.4f}")
print("Saved:", comparison_path)
print("Saved:", compare_path)
print("\nMulti vs single, per session:")
pivot = compare_df.pivot_table(index="session", columns=["model", "pipeline"], values="accuracy")
print(pivot.to_string())
print("=" * 90)
cleanup(sessions)


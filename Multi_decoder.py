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
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.constants import CEBRA_DIR
from utils.min_distance import min_l2_distance
sys.path.insert(0, str(CEBRA_DIR))
import cebra
import cebra.attribution
import cebra.models.layers as cebra_layers
from cebra import CEBRA

SPK_DIR = "./data/spk"
BHV_DIR = "./data/behav"
OUT_DIR = "./outputs_multi_session"
IMG_DIR = "./image_multi_session"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

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
ATTACK_NORM = "l2"
RUN_CEBRA = True
RUN_ACORN = True
ATTR_BATCH_SIZE = 256
SAVE_HEATMAPS = True
TOP_K = 10
EPS_SAMPLE_SIZE = 2000
MIN_TRIALS_PER_SESSION = 10
TEST_SIZE = 0.20
SIDE_TO_LABEL = {"left": 0, "right": 1}
LABEL_NAMES = ["left", "right"]
DECODER_HIDDEN_DIM = 64
DECODER_NUM_LAYERS = 2
DECODER_DROPOUT = 0.4
DECODER_LR = 1e-3
DECODER_WEIGHT_DECAY = 1e-4
DECODER_EPOCHS = 5000
DECODER_BATCH_SIZE = 32
PRINT_EVERY = 500
RANDOM_SEED = 42

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

class FixedSessionModel(nn.Module):
    def __init__(self, model, session_name):
        super().__init__()
        self.model = model
        self.session_name = session_name
        self.num_output = model.num_output
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)
        out = self.model(x, self.session_name)
        if out.dim() == 3:
            out = out.squeeze(0).permute(1, 0)
        return out
    def get_offset(self):
        return self.model.get_offset()

def reduce_attribution(attr):
    if torch.is_tensor(attr):
        attr = attr.detach().cpu().numpy()
    attr = np.abs(np.asarray(attr))
    if attr.ndim == 3:
        attr = attr.mean(axis=0)
    elif attr.ndim == 1:
        attr = attr[None, :]
    elif attr.ndim != 2:
        raise ValueError(f"Unexpected attribution shape: {attr.shape}")
    return attr.astype(np.float32)

def compute_trial_jacobian(model, session_name, trial_X, session_name_for_file, trial_id, model_name):
    fixed_model = FixedSessionModel(model, session_name).to(DEVICE)
    fixed_model.eval()
    x_tensor = torch.tensor(trial_X, dtype=torch.float32, device=DEVICE, requires_grad=True)
    method = cebra.attribution.init(name="jacobian-based-batched", model=fixed_model, input_data=x_tensor, output_dimension=OUTPUT_DIM)
    result = method.compute_attribution_map(batch_size=min(ATTR_BATCH_SIZE, len(trial_X)))
    jf = result["jf"]
    if "jf-inv-svd" in result:
        jf_inv = result["jf-inv-svd"]
    elif "jf-inv-lsq" in result:
        jf_inv = result["jf-inv-lsq"]
    elif "jf-inv" in result:
        jf_inv = result["jf-inv"]
    else:
        raise KeyError(f"No inverse Jacobian. Available keys: {list(result.keys())}")
    jf_matrix = reduce_attribution(jf)
    jf_inv_matrix = reduce_attribution(jf_inv)
    jf_score = np.abs(jf_matrix).mean(axis=0)
    jf_inv_score = np.abs(jf_inv_matrix).mean(axis=0)
    file_prefix = f"{session_name_for_file}_trial{trial_id}_{model_name}"
    torch.save(jf, os.path.join(OUT_DIR, file_prefix + "_jf.pt"))
    torch.save(jf_inv, os.path.join(OUT_DIR, file_prefix + "_jf_inv.pt"))
    np.savez_compressed(os.path.join(OUT_DIR, file_prefix + "_scores.npz"), jf=jf_matrix, jf_inv=jf_inv_matrix, jf_score=jf_score, jf_inv_score=jf_inv_score)
    if SAVE_HEATMAPS:
        plt.figure(figsize=(10, 6))
        plt.imshow(jf_matrix, aspect="auto", cmap="viridis")
        plt.colorbar(label="|Jf|")
        plt.xlabel("Neuron")
        plt.ylabel("Latent dimension")
        plt.title(f"{session_name_for_file} | Trial {trial_id} | {model_name} | Jf")
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, file_prefix + "_jf.png"), dpi=300, bbox_inches="tight")
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.imshow(jf_inv_matrix, aspect="auto", cmap="viridis")
        plt.colorbar(label="|Jf-inv|")
        plt.xlabel("Neuron")
        plt.ylabel("Latent dimension")
        plt.title(f"{session_name_for_file} | Trial {trial_id} | {model_name} | Jf-inv")
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, file_prefix + "_jf_inv.png"), dpi=300, bbox_inches="tight")
        plt.close()
    cleanup(fixed_model, x_tensor, method, result)
    return {"jf_score": jf_score, "jf_inv_score": jf_inv_score}

def select_attribution_trials(sessions, seed):
    rng_local = random.Random(seed)
    selected = {}
    for session in sessions:
        idx = rng_local.randrange(len(session["train_trials"]))
        selected[session["name"]] = idx
    return selected

def compute_all_attributions(model, sessions, model_name, attribution_trial_idx):
    rows = []
    print("\n" + "=" * 90)
    print(f"COMPUTING {model_name} JACOBIANS (1 random train trial per session)")
    print("=" * 90)
    model.eval()
    for session in sessions:
        session_name = session["name"]
        local_idx = attribution_trial_idx[session_name]
        trial_X = session["train_trials"][local_idx]
        original_trial_id = int(session["train_ids"][local_idx])
        print(f"\nSession: {session_name} | neurons={session['n_neurons']} | selected trial_id={original_trial_id} (local idx {local_idx})")
        scores = compute_trial_jacobian(model=model, session_name=session_name, trial_X=trial_X, session_name_for_file=session_name, trial_id=original_trial_id, model_name=model_name)
        top_k = min(TOP_K, session["n_neurons"])
        top_jf = np.argsort(scores["jf_score"])[::-1][:top_k]
        top_jfi = np.argsort(scores["jf_inv_score"])[::-1][:top_k]
        rows.append({"session": session_name, "trial_id": original_trial_id, "n_neurons": session["n_neurons"], "model": model_name, "top_jf_neurons": top_jf.tolist(), "top_jf_scores": scores["jf_score"][top_jf].tolist(), "top_jf_inv_neurons": top_jfi.tolist(), "top_jf_inv_scores": scores["jf_inv_score"][top_jfi].tolist()})
    return pd.DataFrame(rows)

def extract_session_embeddings(model, session, trials):
    fixed_model = FixedSessionModel(model, session["name"]).to(DEVICE)
    fixed_model.eval()
    embeddings = []
    with torch.no_grad():
        for X_t in trials:
            x_tensor = torch.tensor(X_t, dtype=torch.float32, device=DEVICE)
            emb = fixed_model(x_tensor)
            embeddings.append(emb.detach().cpu().numpy())
    cleanup(fixed_model)
    return np.stack(embeddings).astype(np.float32)

def build_pooled_split(model, sessions, split):
    X_parts, y_parts = [], []
    for session in sessions:
        if split == "train":
            trials, y = session["train_trials"], session["y_train"]
        else:
            trials, y = session["test_trials"], session["y_test"]
        if len(trials) == 0:
            continue
        emb = extract_session_embeddings(model, session, trials)
        X_parts.append(emb)
        y_parts.append(y)
    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int64)
    return X, y

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
    for epoch in range(DECODER_EPOCHS):
        decoder.train()
        perm = torch.randperm(len(Xtr_t), device=DEVICE)
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
            print(f"  [{tag}] epoch {epoch + 1}/{DECODER_EPOCHS} | train_acc={train_acc:.4f}")
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
    ax.set_title(f"{tag} (GRU, multi-session) | accuracy={accuracy:.3f}")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"decoder_{tag}_confusion.png"), dpi=300)
    plt.close()
    cleanup(decoder, optimizer, Xtr_t, ytr_t, Xte_t)
    return accuracy

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
attribution_trial_idx = select_attribution_trials(sessions, RANDOM_SEED)
print("\nSelected attribution trials:")
for session in sessions:
    local_idx = attribution_trial_idx[session["name"]]
    print(f"  {session['name']}: local_idx={local_idx} trial_id={int(session['train_ids'][local_idx])}")

X_train_cebra = X_test_cebra = y_train_cebra = y_test_cebra = None
X_train_acorn = X_test_acorn = y_train_acorn = y_test_acorn = None

if RUN_CEBRA:
    cebra_model = train_multisession_model(sessions, model_name="CEBRA", session_eps=session_eps, adversarial=False)
    cebra_path = os.path.join(OUT_DIR, "CEBRA_multisession_shared.pt")
    torch.save(cebra_model.state_dict(), cebra_path)
    print("Saved CEBRA:", cebra_path)
    cebra_summary = compute_all_attributions(cebra_model, sessions, "CEBRA", attribution_trial_idx)
    cebra_summary.to_csv(os.path.join(OUT_DIR, "CEBRA_trial_jacobian_summary.csv"), index=False)
    X_train_cebra, y_train_cebra = build_pooled_split(cebra_model, sessions, "train")
    X_test_cebra, y_test_cebra = build_pooled_split(cebra_model, sessions, "test")
    print(f"CEBRA embeddings | train={X_train_cebra.shape} | test={X_test_cebra.shape}")
    cleanup(cebra_model)

if RUN_ACORN:
    acorn_model = train_multisession_model(sessions, model_name="ACORN", session_eps=session_eps, adversarial=True)
    acorn_path = os.path.join(OUT_DIR, "ACORN_multisession_shared.pt")
    torch.save(acorn_model.state_dict(), acorn_path)
    print("Saved ACORN:", acorn_path)
    acorn_summary = compute_all_attributions(acorn_model, sessions, "ACORN", attribution_trial_idx)
    acorn_summary.to_csv(os.path.join(OUT_DIR, "ACORN_trial_jacobian_summary.csv"), index=False)
    X_train_acorn, y_train_acorn = build_pooled_split(acorn_model, sessions, "train")
    X_test_acorn, y_test_acorn = build_pooled_split(acorn_model, sessions, "test")
    print(f"ACORN embeddings | train={X_train_acorn.shape} | test={X_test_acorn.shape}")
    cleanup(acorn_model)

cebra_acc = None
acorn_acc = None
if RUN_CEBRA:
    cebra_acc = train_decoder(X_train_cebra, y_train_cebra, X_test_cebra, y_test_cebra, "CEBRA")
if RUN_ACORN:
    acorn_acc = train_decoder(X_train_acorn, y_train_acorn, X_test_acorn, y_test_acorn, "ACORN")

comparison = pd.DataFrame({"model": ["CEBRA", "ACORN"], "test_accuracy": [cebra_acc, acorn_acc]})
comparison.to_csv(os.path.join(OUT_DIR, "CEBRA_vs_ACORN_multisession_gru_accuracy.csv"), index=False)

print("\n" + "=" * 90)
print("ALL DONE")
print("=" * 90)
print("Sessions used:", len(sessions))
print("Preprocessing:", PREPROCESS_MODE)
print("Gaussian sigma:", f"{SMOOTH_SIGMA_MS:.1f} ms" if PREPROCESS_MODE == "smooth" else "N/A")
print("Outputs:", OUT_DIR)
print("Images:", IMG_DIR)
if cebra_acc is not None:
    print(f"CEBRA (GRU, multi-session) accuracy: {cebra_acc:.4f}")
if acorn_acc is not None:
    print(f"ACORN (GRU, multi-session) accuracy: {acorn_acc:.4f}")
print("=" * 90)
cleanup(sessions)

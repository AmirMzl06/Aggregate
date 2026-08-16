import os
import gc
import sys
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
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
MAX_SESSIONS = 4
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
TOTAL_STEPS = 3000
NUM_SESSIONS_PER_ITER = 1
TEMPERATURE = 0.4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
ADV_EPSILON = 0.2
ADV_ALPHA = ADV_EPSILON / 5.0
ADV_STEPS = 10
RUN_CEBRA = True
RUN_ACORN = True
ATTACK_NORM = "l2"
ATTR_BATCH_SIZE = 256
SAVE_HEATMAPS = True
TOP_K = 10
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
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

def get_valid_trial_ids(session):
    stim_times = session["stim_times"]
    bhv = session["bhv"]
    valid_ids = []
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
        valid_ids.append(tid)
    return np.asarray(valid_ids, dtype=np.int64)

def prepare_session(session):
    name = session["name"]
    unit = session["unit"]
    stim_times = session["stim_times"]
    valid_ids = get_valid_trial_ids(session)
    if len(valid_ids) == 0:
        print(f"[SKIP] {name}: no valid trials.")
        return None
    print(f"{name}: {len(valid_ids)} valid 2AFC trials")
    raw_trials = []
    processed_trials = []
    for tid in valid_ids:
        raw_X = make_trial_matrix(unit, float(stim_times[tid]))
        processed_X = preprocess_trial(raw_X)
        raw_trials.append(raw_X)
        processed_trials.append(processed_X)
    neural = np.concatenate(processed_trials, axis=0)
    continuous = np.concatenate([np.arange(len(X), dtype=np.float32) for X in processed_trials]).reshape(-1, 1)
    discrete = np.concatenate([np.full(len(X), trial_idx, dtype=np.int64) for trial_idx, X in enumerate(processed_trials)])
    print(f"{name}: neural={neural.shape} | continuous={continuous.shape} | discrete={discrete.shape}")
    return {"name": name, "unit": unit, "stim_times": stim_times, "valid_ids": valid_ids, "raw_trials": raw_trials, "trials": processed_trials, "neural": neural.astype(np.float32), "continuous": continuous.astype(np.float32), "discrete": discrete.astype(np.int64), "n_neurons": neural.shape[1]}

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

def train_multisession_model(sessions, model_name, adversarial=False):
    print("\n" + "=" * 90)
    print(f"TRAINING {model_name} | sessions={len(sessions)}")
    print("=" * 90)
    model = Offset36Multi(num_units=HIDDEN_DIM, num_outputs=OUTPUT_DIM, normalize_output=True)
    for session in sessions:
        model.add_session(session["name"], session["n_neurons"])
        print(f"Registered {session['name']} | neurons={session['n_neurons']}")
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
                x_ref = batch.reference.detach()
                noise = normalize_l2(torch.randn_like(x_ref))
                radius = torch.rand((x_ref.shape[0], 1, 1), device=DEVICE) * ADV_EPSILON
                x_adv = (x_ref + noise * radius).detach()
                x_adv.requires_grad_(True)
                for _ in range(ADV_STEPS):
                    r_adv = model(x_adv, name)
                    p = model(batch.positive, name)
                    n = model(batch.negative, name)
                    adv_loss, _, _ = criterion(r_adv, p, n)
                    grad_x = torch.autograd.grad(adv_loss, x_adv, retain_graph=False, create_graph=False)[0]
                    with torch.no_grad():
                        x_adv = x_adv + ADV_ALPHA * normalize_l2(grad_x)
                        x_adv = project_l2_ball(x_adv, x_ref, ADV_EPSILON)
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
        idx = rng_local.randrange(len(session["trials"]))
        selected[session["name"]] = idx
    return selected

def compute_all_attributions(model, sessions, model_name, attribution_trial_idx):
    rows = []
    print("\n" + "=" * 90)
    print(f"COMPUTING {model_name} JACOBIANS (1 random trial per session)")
    print("=" * 90)
    model.eval()
    for session in sessions:
        session_name = session["name"]
        local_idx = attribution_trial_idx[session_name]
        trial_X = session["trials"][local_idx]
        original_trial_id = int(session["valid_ids"][local_idx])
        print(f"\nSession: {session_name} | neurons={session['n_neurons']} | selected trial_id={original_trial_id} (local idx {local_idx})")
        scores = compute_trial_jacobian(model=model, session_name=session_name, trial_X=trial_X, session_name_for_file=session_name, trial_id=original_trial_id, model_name=model_name)
        top_k = min(TOP_K, session["n_neurons"])
        top_jf = np.argsort(scores["jf_score"])[::-1][:top_k]
        top_jfi = np.argsort(scores["jf_inv_score"])[::-1][:top_k]
        rows.append({"session": session_name, "trial_id": original_trial_id, "n_neurons": session["n_neurons"], "model": model_name, "top_jf_neurons": top_jf.tolist(), "top_jf_scores": scores["jf_score"][top_jf].tolist(), "top_jf_inv_neurons": top_jfi.tolist(), "top_jf_inv_scores": scores["jf_inv_score"][top_jfi].tolist()})
    return pd.DataFrame(rows)

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

attribution_trial_idx = select_attribution_trials(sessions, RANDOM_SEED)
print("\nSelected attribution trials:")
for session in sessions:
    local_idx = attribution_trial_idx[session["name"]]
    print(f"  {session['name']}: local_idx={local_idx} trial_id={int(session['valid_ids'][local_idx])}")

if RUN_CEBRA:
    cebra_model = train_multisession_model(sessions, model_name="CEBRA", adversarial=False)
    cebra_path = os.path.join(OUT_DIR, "CEBRA_multisession_shared.pt")
    torch.save(cebra_model.state_dict(), cebra_path)
    print("Saved CEBRA:", cebra_path)
    cebra_summary = compute_all_attributions(cebra_model, sessions, "CEBRA", attribution_trial_idx)
    cebra_summary.to_csv(os.path.join(OUT_DIR, "CEBRA_trial_jacobian_summary.csv"), index=False)
    cleanup(cebra_model)

if RUN_ACORN:
    acorn_model = train_multisession_model(sessions, model_name="ACORN", adversarial=True)
    acorn_path = os.path.join(OUT_DIR, "ACORN_multisession_shared.pt")
    torch.save(acorn_model.state_dict(), acorn_path)
    print("Saved ACORN:", acorn_path)
    acorn_summary = compute_all_attributions(acorn_model, sessions, "ACORN", attribution_trial_idx)
    acorn_summary.to_csv(os.path.join(OUT_DIR, "ACORN_trial_jacobian_summary.csv"), index=False)
    cleanup(acorn_model)

print("\n" + "=" * 90)
print("ALL DONE")
print("=" * 90)
print("Sessions used:", len(sessions))
print("Preprocessing:", PREPROCESS_MODE)
print("Gaussian sigma:", f"{SMOOTH_SIGMA_MS:.1f} ms" if PREPROCESS_MODE == "smooth" else "N/A")
print("Outputs:", OUT_DIR)
print("Images:", IMG_DIR)
print("=" * 90)
cleanup(sessions)

# import os
# import gc
# import sys
# import glob
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import scipy.io as sio
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from scipy.ndimage import gaussian_filter1d
# from utils.constants import CEBRA_DIR
# from utils.min_distance import min_l2_distance
# sys.path.insert(0, str(CEBRA_DIR))
# import cebra
# import cebra.attribution
# import cebra.models.layers as cebra_layers
# from cebra import CEBRA

# SPK_DIR = "./data/spk"
# BHV_DIR = "./data/behav"
# OUT_DIR = "./outputs_multi_session"
# IMG_DIR = "./image_multi_session"
# os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(IMG_DIR, exist_ok=True)

# PREPROCESS_MODE = "smooth"
# SMOOTH_SIGMA_MS = 100.0
# MAX_SESSIONS = 4
# TASK_NAME = "2afc"
# REQUIRE_BRK_ZERO = True
# PRE_MS = 500
# POST_MS = 1000
# BIN_MS = 10
# SEQ_LEN = int((PRE_MS + POST_MS) / BIN_MS)
# HIDDEN_DIM = 256
# OUTPUT_DIM = 16
# NUM_SHARED_BLOCKS = 16
# DROPOUT = 0.1
# OFFSET_LEFT = 18
# OFFSET_RIGHT = 18
# BATCH_SIZE = 1024
# TOTAL_STEPS = 3
# NUM_SESSIONS_PER_ITER = 1
# TEMPERATURE = 0.4
# LEARNING_RATE = 1e-4
# WEIGHT_DECAY = 0.01
# ADV_EPSILON = 0.2
# ADV_ALPHA = ADV_EPSILON / 5.0
# ADV_STEPS = 10
# RUN_CEBRA = True
# RUN_ACORN = True
# ATTACK_NORM = "l2"
# ATTR_BATCH_SIZE = 256
# SAVE_HEATMAPS = True
# TOP_K = 10
# RANDOM_SEED = 42
# torch.manual_seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
# random.seed(RANDOM_SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(RANDOM_SEED)
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("DEVICE:", DEVICE)
# print("PREPROCESS_MODE:", PREPROCESS_MODE)
# if PREPROCESS_MODE not in {"raw", "normalize", "smooth"}:
#     raise ValueError("PREPROCESS_MODE must be one of: 'raw', 'normalize', 'smooth'")
# if SMOOTH_SIGMA_MS <= 0:
#     raise ValueError("SMOOTH_SIGMA_MS must be > 0.")
# SMOOTH_SIGMA_BINS = SMOOTH_SIGMA_MS / BIN_MS
# print(f"Gaussian sigma: {SMOOTH_SIGMA_MS:.1f} ms = {SMOOTH_SIGMA_BINS:.2f} bins")

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

# def preprocess_trial(X):
#     X = np.asarray(X, dtype=np.float32).copy()
#     if PREPROCESS_MODE == "raw":
#         return X
#     if PREPROCESS_MODE == "normalize":
#         mu = X.mean(axis=0, keepdims=True)
#         sigma = X.std(axis=0, keepdims=True) + 1e-8
#         return ((X - mu) / sigma).astype(np.float32)
#     if PREPROCESS_MODE == "smooth":
#         for neuron_idx in range(X.shape[1]):
#             X[:, neuron_idx] = gaussian_filter1d(X[:, neuron_idx], sigma=SMOOTH_SIGMA_BINS, mode="reflect")
#         return X.astype(np.float32)

# class SessionLayer(nn.Module):
#     def __init__(self, num_units, kernel=2):
#         super().__init__()
#         self.num_units = num_units
#         self.kernel = kernel
#         self.session_dict = nn.ModuleDict()
#     def add_session(self, session_name, num_neurons):
#         self.session_dict.add_module(session_name, nn.Conv1d(num_neurons, self.num_units, self.kernel))
#     def forward(self, x, session_name):
#         if session_name not in self.session_dict:
#             raise KeyError(f"Unknown session '{session_name}'. Registered: {list(self.session_dict.keys())}")
#         return self.session_dict[session_name](x)

# class Offset36Multi(nn.Module):
#     def _make_layers(self, num_units, dropout, n):
#         return [cebra_layers._Skip(nn.Dropout1d(p=dropout), nn.Conv1d(num_units, num_units, 3), nn.GELU()) for _ in range(n)]
#     def __init__(self, num_units, num_outputs, normalize_output=True):
#         super().__init__()
#         self.num_units = num_units
#         self.num_output = num_outputs
#         self.session_layer = SessionLayer(num_units, 2)
#         self.first_net = nn.Sequential(nn.Dropout1d(p=0.1), nn.GELU(), *self._make_layers(num_units, DROPOUT, NUM_SHARED_BLOCKS))
#         self.last_layer_multi = SessionLayer(num_outputs, 3)
#         last_layers = []
#         if normalize_output:
#             last_layers.append(cebra_layers._Norm())
#         last_layers.append(cebra_layers.Squeeze())
#         self.last_net = nn.Sequential(*last_layers)
#     def add_session(self, session_name, num_neurons):
#         self.session_layer.add_session(session_name, num_neurons)
#         self.last_layer_multi.add_session(session_name, self.num_units)
#     def forward(self, x, session_name):
#         x = self.session_layer(x, session_name)
#         x = self.first_net(x)
#         x = self.last_layer_multi(x, session_name)
#         x = self.last_net(x)
#         return x
#     def get_offset(self):
#         return cebra.data.Offset(OFFSET_LEFT, OFFSET_RIGHT)

# def make_trial_matrix(unit, event_time):
#     n_neurons = len(unit)
#     X = np.zeros((SEQ_LEN, n_neurons), dtype=np.float32)
#     start = event_time - PRE_MS / 1000.0
#     end = event_time + POST_MS / 1000.0
#     for neuron_idx in range(n_neurons):
#         spikes = np.asarray(unit[neuron_idx]["timestamps"], dtype=np.float32).reshape(-1)
#         spikes = spikes[(spikes >= start) & (spikes <= end)]
#         spikes_ms = (spikes - event_time) * 1000.0
#         bins = ((spikes_ms + PRE_MS) / BIN_MS).astype(int)
#         bins = bins[(bins >= 0) & (bins < SEQ_LEN)]
#         for b in bins:
#             X[b, neuron_idx] += 1.0
#     return X

# def load_session(spk_path):
#     session_name = os.path.basename(spk_path).replace("_spk.mat", "")
#     bhv_path = os.path.join(BHV_DIR, session_name + "_trialtype.csv")
#     if not os.path.exists(bhv_path):
#         print(f"[SKIP] {session_name}: behavior file not found.")
#         return None
#     print("\n" + "=" * 80)
#     print("LOADING SESSION:", session_name)
#     print("=" * 80)
#     mat = sio.loadmat(spk_path, simplify_cells=True, squeeze_me=True)
#     unit = mat["unit"]
#     t_evt = mat["t_evt"]
#     stim_times = np.asarray(t_evt["stim_on"], dtype=np.float32).reshape(-1)
#     bhv = pd.read_csv(bhv_path)
#     print("Neurons:", len(unit))
#     print("Behavior rows:", len(bhv))
#     return {"name": session_name, "unit": unit, "stim_times": stim_times, "bhv": bhv}

# def get_valid_trial_ids(session):
#     stim_times = session["stim_times"]
#     bhv = session["bhv"]
#     valid_ids = []
#     n_candidates = min(len(stim_times), len(bhv))
#     for tid in range(n_candidates):
#         row = bhv.iloc[tid]
#         task = str(row.get("task", "")).strip().lower()
#         if task != TASK_NAME:
#             continue
#         if REQUIRE_BRK_ZERO:
#             brk = pd.to_numeric(row.get("brk", np.nan), errors="coerce")
#             if not np.isfinite(brk) or brk != 0:
#                 continue
#         valid_ids.append(tid)
#     return np.asarray(valid_ids, dtype=np.int64)

# def prepare_session(session):
#     name = session["name"]
#     unit = session["unit"]
#     stim_times = session["stim_times"]
#     valid_ids = get_valid_trial_ids(session)
#     if len(valid_ids) == 0:
#         print(f"[SKIP] {name}: no valid trials.")
#         return None
#     print(f"{name}: {len(valid_ids)} valid 2AFC trials")
#     raw_trials = []
#     processed_trials = []
#     for tid in valid_ids:
#         raw_X = make_trial_matrix(unit, float(stim_times[tid]))
#         processed_X = preprocess_trial(raw_X)
#         raw_trials.append(raw_X)
#         processed_trials.append(processed_X)
#     neural = np.concatenate(processed_trials, axis=0)
#     continuous = np.concatenate([np.arange(len(X), dtype=np.float32) for X in processed_trials]).reshape(-1, 1)
#     discrete = np.concatenate([np.full(len(X), trial_idx, dtype=np.int64) for trial_idx, X in enumerate(processed_trials)])
#     print(f"{name}: neural={neural.shape} | continuous={continuous.shape} | discrete={discrete.shape}")
#     return {"name": name, "unit": unit, "stim_times": stim_times, "valid_ids": valid_ids, "raw_trials": raw_trials, "trials": processed_trials, "neural": neural.astype(np.float32), "continuous": continuous.astype(np.float32), "discrete": discrete.astype(np.int64), "n_neurons": neural.shape[1]}

# def create_dataset(session_data, model):
#     return cebra.data.TensorDataset(neural=session_data["neural"], continuous=session_data["continuous"], discrete=session_data["discrete"], offset=model.get_offset(), device=DEVICE)

# def create_loader(session_data, model):
#     dataset = create_dataset(session_data, model)
#     loader = cebra.data.single_session.MixedDataLoader(dataset=dataset, time_offset=OFFSET_RIGHT, num_steps=TOTAL_STEPS, batch_size=BATCH_SIZE, conditional="time_delta")
#     return dataset, iter(loader)

# def choose_sessions(step, n_sessions, sessions_per_iter):
#     first = (step * sessions_per_iter) % n_sessions
#     return [(first + i) % n_sessions for i in range(sessions_per_iter)]

# def normalize_l2(x):
#     norm = x.norm(dim=-1, keepdim=True) + 1e-12
#     return x / norm

# def project_l2_ball(x_adv, x_ref, epsilon):
#     delta = x_adv - x_ref
#     norm = delta.norm(dim=-1, keepdim=True) + 1e-12
#     factor = torch.clamp(epsilon / norm, max=1.0)
#     return x_ref + delta * factor

# def train_multisession_model(sessions, model_name, adversarial=False):
#     print("\n" + "=" * 90)
#     print(f"TRAINING {model_name} | sessions={len(sessions)}")
#     print("=" * 90)
#     # model = Offset36Multi(num_units=HIDDEN_DIM, num_outputs=OUTPUT_DIM, normalize_output=True).to(DEVICE)
#     # for session in sessions:
#     #     model.add_session(session["name"], session["n_neurons"])
#     #     print(f"Registered {session['name']} | neurons={session['n_neurons']}")
#     model = Offset36Multi(num_units=HIDDEN_DIM,num_outputs=OUTPUT_DIM,normalize_output=True)
    
#     for session in sessions:
#         model.add_session(session["name"],session["n_neurons"])
#         print(f"Registered {session['name']} | neurons={session['n_neurons']}")
        
#     # Move all shared + session-specific layers to GPU
#     model = model.to(DEVICE)
#     loaders = []
#     for session in sessions:
#         dataset, loader = create_loader(session, model)
#         session["dataset"] = dataset
#         session["loader"] = loader
#         loaders.append(loader)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY)
#     criterion = cebra.models.FixedCosineInfoNCE(temperature=TEMPERATURE).to(DEVICE)
#     progress = tqdm(range(TOTAL_STEPS), desc=model_name)
#     for step in progress:
#         selected = choose_sessions(step, len(sessions), NUM_SESSIONS_PER_ITER)
#         model.train()
#         optimizer.zero_grad(set_to_none=True)
#         batches = []
#         session_names = []
#         total_loss = torch.tensor(0.0, device=DEVICE)
#         for idx in selected:
#             session = sessions[idx]
#             name = session["name"]
#             batch = next(session["loader"])
#             batch.to(DEVICE)
#             reference = model(batch.reference, name)
#             positive = model(batch.positive, name)
#             negative = model(batch.negative, name)
#             loss, _, _ = criterion(reference, positive, negative)
#             total_loss = total_loss + loss / len(selected)
#             batches.append(batch)
#             session_names.append(name)
#         total_loss.backward()
#         optimizer.step()
#         if adversarial:
#             optimizer.zero_grad(set_to_none=True)
#             adv_batches = []
#             for batch, name in zip(batches, session_names):
#                 x_ref = batch.reference.detach()
#                 noise = normalize_l2(torch.randn_like(x_ref))
#                 radius = torch.rand((x_ref.shape[0], 1, 1), device=DEVICE) * ADV_EPSILON
#                 x_adv = (x_ref + noise * radius).detach()
#                 x_adv.requires_grad_(True)
#                 for _ in range(ADV_STEPS):
#                     r_adv = model(x_adv, name)
#                     p = model(batch.positive, name)
#                     n = model(batch.negative, name)
#                     adv_loss, _, _ = criterion(r_adv, p, n)
#                     grad_x = torch.autograd.grad(adv_loss, x_adv, retain_graph=False, create_graph=False)[0]
#                     with torch.no_grad():
#                         x_adv = x_adv + ADV_ALPHA * normalize_l2(grad_x)
#                         x_adv = project_l2_ball(x_adv, x_ref, ADV_EPSILON)
#                     x_adv.requires_grad_(True)
#                 adv_batches.append((x_adv.detach(), batch, name))
#             total_adv_loss = torch.tensor(0.0, device=DEVICE)
#             for x_adv, batch, name in adv_batches:
#                 r_adv = model(x_adv, name)
#                 p = model(batch.positive, name)
#                 n = model(batch.negative, name)
#                 adv_loss, _, _ = criterion(r_adv, p, n)
#                 total_adv_loss = total_adv_loss + adv_loss / len(adv_batches)
#             total_adv_loss.backward()
#             optimizer.step()
#         progress.set_postfix(loss=f"{float(total_loss.detach().cpu()):.4f}")
#     return model

# # class FixedSessionModel(nn.Module):
# #     def __init__(self, model, session_name):
# #         super().__init__()
# #         self.model = model
# #         self.session_name = session_name
# #         self.num_output = model.num_output
# #     def forward(self, x):
# #         return self.model(x, self.session_name)
# #     def get_offset(self):
# #         return self.model.get_offset()
# class FixedSessionModel(nn.Module):
#     def __init__(self, model, session_name):
#         super().__init__()
#         self.model = model
#         self.session_name = session_name
#         self.num_output = model.num_output

#     def forward(self, x):
#         if x.dim() == 2:
#             x = x.unsqueeze(0)       
#         x = x.permute(0, 2, 1)
#         out = self.model(x, self.session_name)
#         if out.dim() == 3:
#             out = out.squeeze(0).permute(1, 0)   
#         return out

#     def get_offset(self):
#         return self.model.get_offset()

# def reduce_attribution(attr):
#     if torch.is_tensor(attr):
#         attr = attr.detach().cpu().numpy()
#     attr = np.abs(np.asarray(attr))
#     if attr.ndim == 3:
#         attr = attr.mean(axis=0)
#     elif attr.ndim == 1:
#         attr = attr[None, :]
#     elif attr.ndim != 2:
#         raise ValueError(f"Unexpected attribution shape: {attr.shape}")
#     return attr.astype(np.float32)

# def compute_trial_jacobian(model, session_name, trial_X, session_name_for_file, trial_id, model_name):
#     fixed_model = FixedSessionModel(model, session_name).to(DEVICE)
#     fixed_model.eval()
#     x_tensor = torch.tensor(trial_X, dtype=torch.float32, device=DEVICE, requires_grad=True)
#     method = cebra.attribution.init(name="jacobian-based-batched", model=fixed_model, input_data=x_tensor, output_dimension=OUTPUT_DIM)
#     result = method.compute_attribution_map(batch_size=min(ATTR_BATCH_SIZE, len(trial_X)))
#     jf = result["jf"]
#     if "jf-inv-svd" in result:
#         jf_inv = result["jf-inv-svd"]
#     elif "jf-inv-lsq" in result:
#         jf_inv = result["jf-inv-lsq"]
#     elif "jf-inv" in result:
#         jf_inv = result["jf-inv"]
#     else:
#         raise KeyError(f"No inverse Jacobian. Available keys: {list(result.keys())}")
#     jf_matrix = reduce_attribution(jf)
#     jf_inv_matrix = reduce_attribution(jf_inv)
#     jf_score = np.abs(jf_matrix).mean(axis=0)
#     jf_inv_score = np.abs(jf_inv_matrix).mean(axis=0)
#     file_prefix = f"{session_name_for_file}_trial{trial_id}_{model_name}"
#     torch.save(jf, os.path.join(OUT_DIR, file_prefix + "_jf.pt"))
#     torch.save(jf_inv, os.path.join(OUT_DIR, file_prefix + "_jf_inv.pt"))
#     np.savez_compressed(os.path.join(OUT_DIR, file_prefix + "_scores.npz"), jf=jf_matrix, jf_inv=jf_inv_matrix, jf_score=jf_score, jf_inv_score=jf_inv_score)
#     if SAVE_HEATMAPS:
#         plt.figure(figsize=(10, 6))
#         plt.imshow(jf_matrix, aspect="auto", cmap="viridis")
#         plt.colorbar(label="|Jf|")
#         plt.xlabel("Neuron")
#         plt.ylabel("Latent dimension")
#         plt.title(f"{session_name_for_file} | Trial {trial_id} | {model_name} | Jf")
#         plt.tight_layout()
#         plt.savefig(os.path.join(IMG_DIR, file_prefix + "_jf.png"), dpi=300, bbox_inches="tight")
#         plt.close()
#         plt.figure(figsize=(10, 6))
#         plt.imshow(jf_inv_matrix, aspect="auto", cmap="viridis")
#         plt.colorbar(label="|Jf-inv|")
#         plt.xlabel("Neuron")
#         plt.ylabel("Latent dimension")
#         plt.title(f"{session_name_for_file} | Trial {trial_id} | {model_name} | Jf-inv")
#         plt.tight_layout()
#         plt.savefig(os.path.join(IMG_DIR, file_prefix + "_jf_inv.png"), dpi=300, bbox_inches="tight")
#         plt.close()
#     cleanup(fixed_model, x_tensor, method, result)
#     return {"jf_score": jf_score, "jf_inv_score": jf_inv_score}

# def compute_all_attributions(model, sessions, model_name):
#     rows = []
#     print("\n" + "=" * 90)
#     print(f"COMPUTING {model_name} JACOBIANS")
#     print("=" * 90)
#     model.eval()
#     for session in sessions:
#         session_name = session["name"]
#         trials = session["trials"]
#         valid_ids = session["valid_ids"]
#         print(f"\nSession: {session_name} | neurons={session['n_neurons']} | trials={len(trials)}")
#         for local_idx, trial_X in enumerate(tqdm(trials, desc=session_name)):
#             original_trial_id = int(valid_ids[local_idx])
#             scores = compute_trial_jacobian(model=model, session_name=session_name, trial_X=trial_X, session_name_for_file=session_name, trial_id=original_trial_id, model_name=model_name)
#             top_k = min(TOP_K, session["n_neurons"])
#             top_jf = np.argsort(scores["jf_score"])[::-1][:top_k]
#             top_jfi = np.argsort(scores["jf_inv_score"])[::-1][:top_k]
#             rows.append({"session": session_name, "trial_id": original_trial_id, "n_neurons": session["n_neurons"], "model": model_name, "top_jf_neurons": top_jf.tolist(), "top_jf_scores": scores["jf_score"][top_jf].tolist(), "top_jf_inv_neurons": top_jfi.tolist(), "top_jf_inv_scores": scores["jf_inv_score"][top_jfi].tolist()})
#     return pd.DataFrame(rows)

# spk_files = sorted(glob.glob(os.path.join(SPK_DIR, "X*_spk.mat")))
# if len(spk_files) == 0:
#     raise RuntimeError(f"No X*_spk.mat files found in {SPK_DIR}")
# random.seed(RANDOM_SEED)
# if MAX_SESSIONS is not None and MAX_SESSIONS < len(spk_files):
#     selected_spk = random.sample(spk_files, MAX_SESSIONS)
# else:
#     selected_spk = spk_files
# selected_spk = sorted(selected_spk)
# print("\nSELECTED SESSIONS:")
# for path in selected_spk:
#     print(" ", os.path.basename(path))

# sessions = []
# for spk_path in selected_spk:
#     session = load_session(spk_path)
#     if session is None:
#         continue
#     prepared = prepare_session(session)
#     if prepared is not None:
#         sessions.append(prepared)
# if len(sessions) == 0:
#     raise RuntimeError("No usable sessions.")
# print("\nUsable sessions:", len(sessions))

# if RUN_CEBRA:
#     cebra_model = train_multisession_model(sessions, model_name="CEBRA", adversarial=False)
#     cebra_path = os.path.join(OUT_DIR, "CEBRA_multisession_shared.pt")
#     torch.save(cebra_model.state_dict(), cebra_path)
#     print("Saved CEBRA:", cebra_path)
#     cebra_summary = compute_all_attributions(cebra_model, sessions, "CEBRA")
#     cebra_summary.to_csv(os.path.join(OUT_DIR, "CEBRA_trial_jacobian_summary.csv"), index=False)
#     cleanup(cebra_model)

# if RUN_ACORN:
#     acorn_model = train_multisession_model(sessions, model_name="ACORN", adversarial=True)
#     acorn_path = os.path.join(OUT_DIR, "ACORN_multisession_shared.pt")
#     torch.save(acorn_model.state_dict(), acorn_path)
#     print("Saved ACORN:", acorn_path)
#     acorn_summary = compute_all_attributions(acorn_model, sessions, "ACORN")
#     acorn_summary.to_csv(os.path.join(OUT_DIR, "ACORN_trial_jacobian_summary.csv"), index=False)
#     cleanup(acorn_model)

# print("\n" + "=" * 90)
# print("ALL DONE")
# print("=" * 90)
# print("Sessions used:", len(sessions))
# print("Preprocessing:", PREPROCESS_MODE)
# print("Gaussian sigma:", f"{SMOOTH_SIGMA_MS:.1f} ms" if PREPROCESS_MODE == "smooth" else "N/A")
# print("Outputs:", OUT_DIR)
# print("Images:", IMG_DIR)
# print("=" * 90)
# cleanup(sessions)




# # import os
# # import gc
# # import sys
# # import glob
# # import random
# # import numpy as np
# # import pandas as pd
# # import torch
# # import torch.nn as nn
# # import scipy.io as sio
# # import matplotlib.pyplot as plt
# # from tqdm import tqdm
# # from utils.constants import CEBRA_DIR
# # from utils.min_distance import min_l2_distance
# # sys.path.insert(0, str(CEBRA_DIR))
# # import cebra
# # import cebra.attribution
# # import cebra.models.layers as cebra_layers
# # from cebra import CEBRA

# # SPK_DIR = "./data/spk"
# # BHV_DIR = "./data/behav"
# # OUT_DIR = "./outputs_multi_session"
# # IMG_DIR = "./image_multi_session"
# # os.makedirs(OUT_DIR, exist_ok=True)
# # os.makedirs(IMG_DIR, exist_ok=True)

# # MAX_SESSIONS = 3
# # TASK_NAME = "2afc"
# # REQUIRE_BRK_ZERO = True
# # PRE_MS = 500
# # POST_MS = 1000
# # BIN_MS = 10
# # SEQ_LEN = int((PRE_MS + POST_MS) / BIN_MS)
# # USE_ZSCORE = True
# # GAUSS_IN = False
# # GAUSS_SIGMA = 2.0
# # GAUSS_KERNEL = 20
# # HIDDEN_DIM = 256
# # OUTPUT_DIM = 16
# # NUM_SHARED_BLOCKS = 16
# # DROPOUT = 0.1
# # OFFSET_LEFT = 18
# # OFFSET_RIGHT = 18
# # BATCH_SIZE = 1024
# # TOTAL_STEPS = 60000
# # NUM_SESSIONS_PER_ITER = 1
# # TEMPERATURE = 0.4
# # LEARNING_RATE = 1e-4
# # WEIGHT_DECAY = 0.01
# # ADV_EPSILON = 0.2
# # ADV_ALPHA = ADV_EPSILON / 5.0
# # ADV_STEPS = 10
# # ATTACK_NORM = "l2"
# # RUN_CEBRA = True
# # RUN_ACORN = True
# # ATTR_BATCH_SIZE = 128
# # SAVE_HEATMAPS = True
# # RANDOM_SEED = 42

# # torch.manual_seed(RANDOM_SEED)
# # np.random.seed(RANDOM_SEED)
# # random.seed(RANDOM_SEED)
# # if torch.cuda.is_available():
# #     torch.cuda.manual_seed_all(RANDOM_SEED)
# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print("DEVICE:", DEVICE)

# # def cleanup(*objects):
# #     for obj in objects:
# #         try:
# #             del obj
# #         except Exception:
# #             pass
# #     gc.collect()
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()
# #         torch.cuda.ipc_collect()

# # class SessionLayer(nn.Module):
# #     def __init__(self, num_units, kernel=2):
# #         super().__init__()
# #         self.num_units = num_units
# #         self.kernel = kernel
# #         self.session_dict = nn.ModuleDict()
# #     def add_session(self, session_name, num_neurons):
# #         self.session_dict.add_module(session_name, nn.Conv1d(num_neurons, self.num_units, self.kernel))
# #     def forward(self, x, session_name):
# #         if session_name not in self.session_dict:
# #             raise KeyError(f"Unknown session '{session_name}'. Registered sessions: {list(self.session_dict.keys())}")
# #         return self.session_dict[session_name](x)

# # class Offset36Multi(nn.Module):
# #     def _make_layers(self, num_units, dropout, n):
# #         return [cebra_layers._Skip(nn.Dropout1d(p=dropout), nn.Conv1d(num_units, num_units, 3), nn.GELU()) for _ in range(n)]
# #     def __init__(self, num_units, num_outputs, normalize=True, gauss_in=False):
# #         super().__init__()
# #         self.num_units = num_units
# #         self.num_output = num_outputs
# #         self.gauss_in = gauss_in
# #         if self.gauss_in:
# #             self.smoothers = nn.ModuleDict()
# #         self.session_layer = SessionLayer(num_units=num_units, kernel=2)
# #         shared_layers = [nn.Dropout1d(p=0.1), nn.GELU(), *self._make_layers(num_units, DROPOUT, NUM_SHARED_BLOCKS)]
# #         self.first_net = nn.Sequential(*shared_layers)
# #         self.last_layer_multi = SessionLayer(num_outputs, kernel=3)
# #         last_layers = []
# #         if normalize:
# #             last_layers.append(cebra_layers._Norm())
# #         last_layers.append(cebra_layers.Squeeze())
# #         self.last_net = nn.Sequential(*last_layers)
# #     def add_session(self, session_name, num_neurons):
# #         if self.gauss_in:
# #             self.smoothers[session_name] = nn.AvgPool1d(kernel_size=GAUSS_KERNEL, stride=1, padding=GAUSS_KERNEL // 2)
# #         self.session_layer.add_session(session_name, num_neurons)
# #         self.last_layer_multi.add_session(session_name, self.num_units)
# #     def forward(self, x, session_name):
# #         if self.gauss_in:
# #             x = self.smoothers[session_name](x)
# #         x = self.session_layer(x, session_name)
# #         x = self.first_net(x)
# #         x = self.last_layer_multi(x, session_name)
# #         x = self.last_net(x)
# #         return x
# #     def get_offset(self):
# #         return cebra.data.Offset(OFFSET_LEFT, OFFSET_RIGHT)

# # def make_trial_matrix(unit, event_time):
# #     n_neurons = len(unit)
# #     X = np.zeros((SEQ_LEN, n_neurons), dtype=np.float32)
# #     start = event_time - PRE_MS / 1000.0
# #     end = event_time + POST_MS / 1000.0
# #     for neuron_idx in range(n_neurons):
# #         spikes = np.asarray(unit[neuron_idx]["timestamps"], dtype=np.float32).reshape(-1)
# #         spikes = spikes[(spikes >= start) & (spikes <= end)]
# #         spikes_ms = (spikes - event_time) * 1000.0
# #         bins = ((spikes_ms + PRE_MS) / BIN_MS).astype(int)
# #         bins = bins[(bins >= 0) & (bins < SEQ_LEN)]
# #         for b in bins:
# #             X[b, neuron_idx] += 1.0
# #     return X

# # def load_session(spk_path):
# #     session_name = os.path.basename(spk_path).replace("_spk.mat", "")
# #     bhv_path = os.path.join(BHV_DIR, session_name + "_trialtype.csv")
# #     if not os.path.exists(bhv_path):
# #         print(f"[SKIP] {session_name}: behavior file not found.")
# #         return None
# #     print("\n" + "=" * 80)
# #     print("LOADING:", session_name)
# #     print("=" * 80)
# #     mat = sio.loadmat(spk_path, simplify_cells=True, squeeze_me=True)
# #     unit = mat["unit"]
# #     t_evt = mat["t_evt"]
# #     stim_times = np.asarray(t_evt["stim_on"], dtype=np.float32).reshape(-1)
# #     bhv = pd.read_csv(bhv_path)
# #     print("neurons:", len(unit))
# #     print("behavior rows:", len(bhv))
# #     return {"name": session_name, "unit": unit, "stim_times": stim_times, "bhv": bhv}

# # def get_valid_trial_ids(session):
# #     stim_times = session["stim_times"]
# #     bhv = session["bhv"]
# #     valid_ids = []
# #     n_candidates = min(len(stim_times), len(bhv))
# #     for tid in range(n_candidates):
# #         row = bhv.iloc[tid]
# #         task = str(row.get("task", "")).strip().lower()
# #         if task != TASK_NAME:
# #             continue
# #         if REQUIRE_BRK_ZERO:
# #             brk = pd.to_numeric(row.get("brk", np.nan), errors="coerce")
# #             if not np.isfinite(brk) or brk != 0:
# #                 continue
# #         valid_ids.append(tid)
# #     return np.asarray(valid_ids, dtype=np.int64)

# # def prepare_session(session):
# #     name = session["name"]
# #     unit = session["unit"]
# #     stim_times = session["stim_times"]
# #     valid_ids = get_valid_trial_ids(session)
# #     if len(valid_ids) == 0:
# #         print(f"[SKIP] {name}: no valid 2AFC trials.")
# #         return None
# #     print(f"{name}: {len(valid_ids)} valid trials")
# #     raw_trials = []
# #     for tid in valid_ids:
# #         X = make_trial_matrix(unit, float(stim_times[tid]))
# #         raw_trials.append(X)
# #     raw_concat = np.concatenate(raw_trials, axis=0)
# #     if USE_ZSCORE:
# #         mu = raw_concat.mean(axis=0, keepdims=True)
# #         sigma = raw_concat.std(axis=0, keepdims=True) + 1e-8
# #     else:
# #         mu = np.zeros((1, raw_concat.shape[1]), dtype=np.float32)
# #         sigma = np.ones((1, raw_concat.shape[1]), dtype=np.float32)
# #     trials = []
# #     for X in raw_trials:
# #         Xn = ((X - mu) / sigma).astype(np.float32)
# #         trials.append(Xn)
# #     neural = np.concatenate(trials, axis=0)
# #     n_total = len(neural)
# #     continuous = np.concatenate([np.arange(len(X), dtype=np.float32) for X in trials]).reshape(-1, 1)
# #     discrete = np.concatenate([np.full(len(X), trial_idx, dtype=np.int64) for trial_idx, X in enumerate(trials)])
# #     print(f"{name}: neural shape = {neural.shape}")
# #     print(f"{name}: continuous shape = {continuous.shape}")
# #     print(f"{name}: discrete shape = {discrete.shape}")
# #     return {"name": name, "valid_ids": valid_ids, "unit": unit, "trials": trials, "neural": neural, "continuous": continuous, "discrete": discrete, "mu": mu, "sigma": sigma, "n_neurons": neural.shape[1]}

# # def create_dataset(session_data, model):
# #     dataset = cebra.data.datasets.TensorDataset(
# #         neural=session_data["neural"],
# #         continuous=session_data["continuous"],
# #         discrete=session_data["discrete"],
# #         offset=model.get_offset(),
# #         device=DEVICE
# #     )
# #     return dataset

# # def create_loader(session_data, model):
# #     dataset = create_dataset(session_data, model)
# #     loader = cebra.data.single_session.MixedDataLoader(
# #         dataset=dataset,
# #         time_offset=OFFSET_RIGHT,
# #         num_steps=TOTAL_STEPS,
# #         batch_size=BATCH_SIZE,
# #         conditional="time_delta"
# #     )
# #     return dataset, iter(loader)

# # def choose_sessions(step, n_sessions, sessions_per_iter):
# #     first = (step * sessions_per_iter) % n_sessions
# #     return [(first + i) % n_sessions for i in range(sessions_per_iter)]

# # def normalize_l2(x):
# #     norm = x.norm(dim=-1, keepdim=True) + 1e-12
# #     return x / norm

# # def project_l2_ball(x_adv, x_ref, epsilon):
# #     delta = x_adv - x_ref
# #     norm = delta.norm(dim=-1, keepdim=True) + 1e-12
# #     factor = torch.clamp(epsilon / norm, max=1.0)
# #     return x_ref + delta * factor

# # def train_multisession_model(session_data, model_name, adversarial=False):
# #     print("\n" + "=" * 90)
# #     print(f"TRAINING {model_name}")
# #     print(f"Number of sessions: {len(session_data)}")
# #     print("=" * 90)
# #     model = Offset36Multi(num_units=HIDDEN_DIM, num_outputs=OUTPUT_DIM, normalize=True, gauss_in=GAUSS_IN).to(DEVICE)
# #     for data in session_data:
# #         model.add_session(data["name"], data["n_neurons"])
# #         print(f"Registered: {data['name']} | neurons={data['n_neurons']}")
# #     loaders = []
# #     for data in session_data:
# #         dataset, loader = create_loader(data, model)
# #         data["dataset"] = dataset
# #         data["loader"] = loader
# #         loaders.append(loader)
# #     optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY)
# #     criterion = cebra.models.FixedCosineInfoNCE(temperature=TEMPERATURE).to(DEVICE)
# #     progress = tqdm(range(TOTAL_STEPS), desc=model_name)
# #     for step in progress:
# #         selected = choose_sessions(step, len(session_data), NUM_SESSIONS_PER_ITER)
# #         model.train()
# #         optimizer.zero_grad(set_to_none=True)
# #         batches = []
# #         session_names = []
# #         total_loss = torch.tensor(0.0, device=DEVICE)
# #         for idx in selected:
# #             data = session_data[idx]
# #             name = data["name"]
# #             batch = next(data["loader"])
# #             batch.to(DEVICE)
# #             reference = model(batch.reference, name)
# #             positive = model(batch.positive, name)
# #             negative = model(batch.negative, name)
# #             loss, _, _ = criterion(reference, positive, negative)
# #             total_loss = total_loss + loss / len(selected)
# #             batches.append(batch)
# #             session_names.append(name)
# #         total_loss.backward()
# #         optimizer.step()
# #         if adversarial:
# #             optimizer.zero_grad(set_to_none=True)
# #             adversarial_batches = []
# #             for batch, name in zip(batches, session_names):
# #                 x_ref = batch.reference.detach()
# #                 x_adv = x_ref + normalize_l2(torch.randn_like(x_ref)) * (torch.rand((x_ref.shape[0], 1, 1), device=DEVICE) * ADV_EPSILON)
# #                 x_adv = x_adv.detach()
# #                 x_adv.requires_grad_(True)
# #                 for _ in range(ADV_STEPS):
# #                     r_adv = model(x_adv, name)
# #                     p = model(batch.positive, name)
# #                     n = model(batch.negative, name)
# #                     adv_loss, _, _ = criterion(r_adv, p, n)
# #                     grad_x = torch.autograd.grad(adv_loss, x_adv, retain_graph=False, create_graph=False)[0]
# #                     with torch.no_grad():
# #                         grad_dir = normalize_l2(grad_x)
# #                         x_adv = x_adv + ADV_ALPHA * grad_dir
# #                         x_adv = project_l2_ball(x_adv, x_ref, ADV_EPSILON)
# #                     x_adv.requires_grad_(True)
# #                 adversarial_batches.append((x_adv.detach(), batch, name))
# #             total_adv_loss = torch.tensor(0.0, device=DEVICE)
# #             for x_adv, batch, name in adversarial_batches:
# #                 r_adv = model(x_adv, name)
# #                 p = model(batch.positive, name)
# #                 n = model(batch.negative, name)
# #                 adv_loss, _, _ = criterion(r_adv, p, n)
# #                 total_adv_loss = total_adv_loss + adv_loss / len(adversarial_batches)
# #             total_adv_loss.backward()
# #             optimizer.step()
# #         progress.set_postfix(loss=f"{float(total_loss):.4f}")
# #     return model

# # class FixedSessionModel(nn.Module):
# #     def __init__(self, model, session_name):
# #         super().__init__()
# #         self.model = model
# #         self.session_name = session_name
# #         self.num_output = model.num_output
# #     def forward(self, x):
# #         return self.model(x, self.session_name)
# #     def get_offset(self):
# #         return self.model.get_offset()

# # def reduce_attribution(attr):
# #     if torch.is_tensor(attr):
# #         attr = attr.detach().cpu().numpy()
# #     attr = np.abs(np.asarray(attr))
# #     if attr.ndim == 3:
# #         attr = attr.mean(axis=0)
# #     elif attr.ndim == 1:
# #         attr = attr[None, :]
# #     elif attr.ndim != 2:
# #         raise ValueError(f"Unexpected attribution shape: {attr.shape}")
# #     return attr.astype(np.float32)

# # def compute_trial_jacobian(model, session_name, trial_X, session_prefix, trial_id, model_name):
# #     fixed_model = FixedSessionModel(model, session_name).to(DEVICE)
# #     fixed_model.eval()
# #     x_tensor = torch.tensor(trial_X, dtype=torch.float32, device=DEVICE, requires_grad=True)
# #     method = cebra.attribution.init(name="jacobian-based-batched", model=fixed_model, input_data=x_tensor, output_dimension=OUTPUT_DIM)
# #     result = method.compute_attribution_map(batch_size=min(ATTR_BATCH_SIZE, len(trial_X)))
# #     jf = result["jf"]
# #     if "jf-inv-svd" in result:
# #         jf_inv = result["jf-inv-svd"]
# #     elif "jf-inv-lsq" in result:
# #         jf_inv = result["jf-inv-lsq"]
# #     elif "jf-inv" in result:
# #         jf_inv = result["jf-inv"]
# #     else:
# #         raise KeyError(f"No inverse Jacobian. Available keys: {list(result.keys())}")
# #     jf_matrix = reduce_attribution(jf)
# #     jf_inv_matrix = reduce_attribution(jf_inv)
# #     jf_score = np.abs(jf_matrix).mean(axis=0)
# #     jf_inv_score = np.abs(jf_inv_matrix).mean(axis=0)
# #     prefix = f"{session_prefix}_trial{trial_id}_{model_name}"
# #     torch.save(jf, os.path.join(OUT_DIR, prefix + "_jf.pt"))
# #     torch.save(jf_inv, os.path.join(OUT_DIR, prefix + "_jf_inv.pt"))
# #     np.savez_compressed(os.path.join(OUT_DIR, prefix + "_scores.npz"), jf=jf_matrix, jf_inv=jf_inv_matrix, jf_score=jf_score, jf_inv_score=jf_inv_score)
# #     if SAVE_HEATMAPS:
# #         plt.figure(figsize=(10, 6))
# #         plt.imshow(jf_matrix, aspect="auto", cmap="viridis")
# #         plt.colorbar(label="|Jf|")
# #         plt.xlabel("Neuron")
# #         plt.ylabel("Latent dimension")
# #         plt.title(f"{session_prefix} | Trial {trial_id} | {model_name} | Jf")
# #         plt.tight_layout()
# #         plt.savefig(os.path.join(IMG_DIR, prefix + "_jf.png"), dpi=300, bbox_inches="tight")
# #         plt.close()
# #         plt.figure(figsize=(10, 6))
# #         plt.imshow(jf_inv_matrix, aspect="auto", cmap="viridis")
# #         plt.colorbar(label="|Jf-inv|")
# #         plt.xlabel("Neuron")
# #         plt.ylabel("Latent dimension")
# #         plt.title(f"{session_prefix} | Trial {trial_id} | {model_name} | Jf-inv")
# #         plt.tight_layout()
# #         plt.savefig(os.path.join(IMG_DIR, prefix + "_jf_inv.png"), dpi=300, bbox_inches="tight")
# #         plt.close()
# #     cleanup(fixed_model, x_tensor, method, result)
# #     return {"jf_score": jf_score, "jf_inv_score": jf_inv_score}

# # def compute_all_attributions(model, sessions, model_name):
# #     rows = []
# #     print("\n" + "=" * 90)
# #     print(f"COMPUTING {model_name} JACOBIANS")
# #     print("=" * 90)
# #     model.eval()
# #     for session_data in sessions:
# #         session_name = session_data["name"]
# #         trials = session_data["trials"]
# #         valid_ids = session_data["valid_ids"]
# #         print(f"\n{session_name}: {len(trials)} trials | {session_data['n_neurons']} neurons")
# #         for local_idx, trial_X in enumerate(tqdm(trials, desc=session_name)):
# #             trial_id = int(valid_ids[local_idx])
# #             scores = compute_trial_jacobian(model=model, session_name=session_name, trial_X=trial_X, session_prefix=session_name, trial_id=trial_id, model_name=model_name)
# #             top_k = min(10, session_data["n_neurons"])
# #             top_jf = np.argsort(scores["jf_score"])[::-1][:top_k]
# #             top_jfi = np.argsort(scores["jf_inv_score"])[::-1][:top_k]
# #             rows.append({"session": session_name, "trial_id": trial_id, "n_neurons": session_data["n_neurons"], "model": model_name, "top_jf": top_jf.tolist(), "top_jf_scores": scores["jf_score"][top_jf].tolist(), "top_jf_inv": top_jfi.tolist(), "top_jf_inv_scores": scores["jf_inv_score"][top_jfi].tolist()})
# #         gc.collect()
# #         if torch.cuda.is_available():
# #             torch.cuda.empty_cache()
# #     return pd.DataFrame(rows)

# # spk_files = sorted(glob.glob(os.path.join(SPK_DIR, "X*_spk.mat")))
# # if len(spk_files) == 0:
# #     raise RuntimeError(f"No X*_spk.mat files found in {SPK_DIR}")
# # random.seed(RANDOM_SEED)
# # if MAX_SESSIONS is not None and MAX_SESSIONS < len(spk_files):
# #     selected_spk = random.sample(spk_files, MAX_SESSIONS)
# # else:
# #     selected_spk = spk_files
# # selected_spk = sorted(selected_spk)
# # print("\nSELECTED SESSIONS:")
# # for path in selected_spk:
# #     print(" ", os.path.basename(path))

# # sessions = []
# # for spk_path in selected_spk:
# #     session = load_session(spk_path)
# #     if session is None:
# #         continue
# #     prepared = prepare_session(session)
# #     if prepared is not None:
# #         sessions.append(prepared)
# # if len(sessions) == 0:
# #     raise RuntimeError("No usable sessions.")
# # print("\nUsable sessions:", len(sessions))

# # cebra_model = None
# # cebra_summary = None
# # if RUN_CEBRA:
# #     cebra_model = train_multisession_model(sessions, model_name="CEBRA", adversarial=False)
# #     cebra_model_path = os.path.join(OUT_DIR, "CEBRA_multisession_shared.pt")
# #     torch.save(cebra_model.state_dict(), cebra_model_path)
# #     print("\nSaved:", cebra_model_path)
# #     cebra_summary = compute_all_attributions(cebra_model, sessions, "CEBRA")
# #     cebra_summary.to_csv(os.path.join(OUT_DIR, "CEBRA_trial_jacobian_summary.csv"), index=False)
# #     cleanup(cebra_model)
# #     cebra_model = None

# # acorn_model = None
# # acorn_summary = None
# # if RUN_ACORN:
# #     acorn_model = train_multisession_model(sessions, model_name="ACORN", adversarial=True)
# #     acorn_model_path = os.path.join(OUT_DIR, "ACORN_multisession_shared.pt")
# #     torch.save(acorn_model.state_dict(), acorn_model_path)
# #     print("\nSaved:", acorn_model_path)
# #     acorn_summary = compute_all_attributions(acorn_model, sessions, "ACORN")
# #     acorn_summary.to_csv(os.path.join(OUT_DIR, "ACORN_trial_jacobian_summary.csv"), index=False)
# #     cleanup(acorn_model)
# #     acorn_model = None

# # print("\n" + "=" * 90)
# # print("ALL DONE")
# # print("=" * 90)
# # print(f"Sessions used: {len(sessions)}")
# # print(f"Output: {OUT_DIR}")
# # print(f"Images: {IMG_DIR}")
# # print("\nEach valid trial has:")
# # print("  *_CEBRA_jf.pt")
# # print("  *_CEBRA_jf_inv.pt")
# # print("  *_ACORN_jf.pt")
# # print("  *_ACORN_jf_inv.pt")
# # print("\nThe *_scores.npz files contain:")
# # print("  jf")
# # print("  jf_inv")
# # print("  jf_score")
# # print("  jf_inv_score")
# # print("=" * 90)
# # cleanup(*sessions)

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
from cebra import CEBRA

DATA_PATH = "./data/spk/M021519_spk.mat"
OUT_DIR = "./outputs"
IMG_DIR = "./image"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
BATCH_SIZE = 256
MAX_ITER = 10000
OUTPUT_DIM = 16
TRIAL_ID = 
PRE_MS = 500
POST_MS = 1000
BIN_MS = 10

mat = sio.loadmat(DATA_PATH, simplify_cells=True, squeeze_me=True)
unit = mat["unit"]
t_evt = mat["t_evt"]
print("neurons:", len(unit))
print("events:", t_evt.keys())

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

stim_times = np.asarray(t_evt["stim_on"])
trial_time = stim_times[TRIAL_ID]
X = make_trial_matrix(unit, trial_time, PRE_MS, POST_MS, BIN_MS)
print("X:", X.shape)
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
X = X.astype(np.float32)

def train_cebra(X, adv=False):
    mode = "adversarial" if adv else "clean"
    eps = float(min_l2_distance(X)) / 2
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
    )
    labels = np.arange(len(X)).astype(np.float32)
    model.fit(X, labels)
    return model, eps

def get_attribution(model, name):
    encoder = model.solver_.model.to("cuda")
    x_tensor = torch.tensor(X, dtype=torch.float32, device="cuda", requires_grad=True)
    method = cebra.attribution.init(name="jacobian-based-batched", model=encoder, input_data=x_tensor, output_dimension=OUTPUT_DIM)
    result = method.compute_attribution_map(batch_size=128)
    print(result.keys())
    jf = result["jf"]
    jf_inv = result["jf-inv-svd"]
    torch.save(jf, f"{OUT_DIR}/M021519_trial{TRIAL_ID}_{name}_jf.pt")
    torch.save(jf_inv, f"{OUT_DIR}/M021519_trial{TRIAL_ID}_{name}_jf_inv.pt")
    save_heatmap(jf, name + "_jacobian")
    save_heatmap(jf_inv, name + "_inverse_jacobian")
    del encoder, x_tensor
    gc.collect()
    torch.cuda.empty_cache()

def save_heatmap(tensor, name):
    if torch.is_tensor(tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    if arr.ndim == 3:
        arr = np.abs(arr).mean(axis=0)
    else:
        arr = np.abs(arr)
    plt.figure(figsize=(10,6))
    plt.imshow(arr, aspect="auto")
    plt.colorbar(label="absolute attribution")
    plt.xlabel("Neuron")
    plt.ylabel("Latent dimension")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/{name}.png", dpi=300)
    plt.close()

cebra_model, eps = train_cebra(X, adv=False)
get_attribution(cebra_model, "CEBRA")
del cebra_model
gc.collect()
torch.cuda.empty_cache()

acorn_model, eps = train_cebra(X, adv=True)
get_attribution(acorn_model, "ACORN")
print("\nDONE")

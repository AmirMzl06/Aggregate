import os
import sys
import gc
import math
import numbers
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.constants import CEBRA_DIR
from utils.min_distance import min_l2_distance

for module_name in list(sys.modules):
    if module_name == "cebra" or module_name.startswith("cebra."):
        del sys.modules[module_name]
sys.path.insert(0, str(CEBRA_DIR))
import cebra
import cebra.attribution
from cebra import CEBRA
print("\nUsing CEBRA from:")
print(cebra.__file__)

NWB_PATH = "data/Area2_Bump/sub-Han_desc-train_behavior+ecephys.nwb"
OUT = "Area2_Bump_Jacobian"
os.makedirs(OUT, exist_ok=True)

BIN_MS = 50.0
BIN_SEC = BIN_MS / 1000.0
SMOOTH_SD_MS = 100.0
SMOOTH_SIGMA_BINS = SMOOTH_SD_MS / BIN_MS
SMOOTH_KERNEL_SIZE = 17
SEED = 42
LATENT_DIM = 128
NUM_HIDDEN_UNITS = 128
BATCH_SIZE = 512
MAX_ITER = 3000
TEMPERATURE = 0.4
TIME_OFFSETS = 4
MODEL_ARCH = "offset36-model-more-dropout"
DEVICE = "cuda_if_available"
ADV_STEPS = 10
ATTACK_NORM = "linf"
ATTR_N_CHUNKS = 16
ATTR_CHUNK_LEN = 128
ATTR_BATCH_SIZE = 16

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_all(SEED)

class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=1):
        super().__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        kernel = 1.0
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing="ij")
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2.0
            kernel *= (1.0 / (std * math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * ((mgrid - mean) / std) ** 2))
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        self.register_buffer("weight", kernel)
        self.groups = channels
        if dim == 1:
            self.conv = F.conv1d
        else:
            raise RuntimeError("This script only uses 1D smoothing.")
    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.conv(x, weight=self.weight, groups=self.groups, padding="same")
        x = torch.permute(x, (0, 2, 1))
        return x

def get_recording_interval(f):
    g = f["processing/behavior/hand_vel"]
    n_samples = g["data"].shape[0]
    if "starting_time" in g:
        st = g["starting_time"]
        t_start = float(st[()])
        if "rate" not in st.attrs:
            raise RuntimeError("hand_vel starting_time has no rate.")
        rate = float(st.attrs["rate"])
        t_stop = t_start + n_samples / rate
    elif "timestamps" in g:
        timestamps = g["timestamps"]
        t_start = float(timestamps[0])
        t_stop = float(timestamps[-1])
        rate = None
    else:
        raise RuntimeError("Could not determine recording timeline.")
    print("\n" + "=" * 80)
    print("RECORDING INTERVAL")
    print("=" * 80)
    print("t_start:", t_start)
    print("t_stop :", t_stop)
    print("duration:", (t_stop - t_start) / 60.0, "min")
    if rate is not None:
        print("reference rate:", rate, "Hz")
    return t_start, t_stop

def build_spike_counts():
    print("\n" + "=" * 80)
    print("BUILDING AREA2 SPIKE COUNTS")
    print("=" * 80)
    with h5py.File(NWB_PATH, "r") as f:
        t_start, t_stop = get_recording_interval(f)
        units = f["units"]
        unit_ids = np.asarray(units["id"][:], dtype=np.int64)
        heldout = np.asarray(units["heldout"][:], dtype=bool)
        n_units = len(unit_ids)
        print("\nUnits:", n_units)
        print("Heldout flags:", int(heldout.sum()))
        print("Using ALL units:", n_units)
        edges = np.arange(t_start, t_stop + BIN_SEC, BIN_SEC, dtype=np.float64)
        n_bins = len(edges) - 1
        print("\nBin width:", BIN_MS, "ms")
        print("Time bins:", n_bins)
        X = np.zeros((n_bins, n_units), dtype=np.float32)
        spike_times = units["spike_times"]
        spike_index = np.asarray(units["spike_times_index"][:], dtype=np.int64)
        print("\nBinning spikes...")
        for neuron_idx in range(n_units):
            end_idx = int(spike_index[neuron_idx])
            start_idx = 0 if neuron_idx == 0 else int(spike_index[neuron_idx - 1])
            spikes = np.asarray(spike_times[start_idx:end_idx], dtype=np.float64)
            left = np.searchsorted(spikes, t_start, side="left")
            right = np.searchsorted(spikes, t_stop, side="right")
            spikes = spikes[left:right]
            bin_idx = ((spikes - t_start) / BIN_SEC).astype(np.int64)
            valid = (bin_idx >= 0) & (bin_idx < n_bins)
            counts = np.bincount(bin_idx[valid], minlength=n_bins)
            X[:, neuron_idx] = counts[:n_bins].astype(np.float32)
        print("\nRAW X")
        print("shape:", X.shape)
        print("min :", float(X.min()))
        print("max :", float(X.max()))
        print("mean:", float(X.mean()))
        print("\nNeuron input-column mapping:")
        for idx, unit_id in enumerate(unit_ids):
            print(f"{idx:2d} -> unit {unit_id}")
    return X, unit_ids

def smooth_spike_counts(X):
    print("\n" + "=" * 80)
    print("GAUSSIAN SMOOTHING")
    print("=" * 80)
    print("Input shape:", X.shape)
    print("Bin width:", BIN_MS, "ms")
    print("Gaussian SD:", SMOOTH_SD_MS, "ms")
    print("Gaussian sigma:", SMOOTH_SIGMA_BINS, "bins")
    print("Kernel size:", SMOOTH_KERNEL_SIZE)
    n_neurons = X.shape[1]
    x_tensor = torch.from_numpy(X.astype(np.float32, copy=False)).unsqueeze(0)
    smoother = GaussianSmoothing(channels=n_neurons, kernel_size=SMOOTH_KERNEL_SIZE, sigma=SMOOTH_SIGMA_BINS, dim=1)
    smoother.eval()
    with torch.no_grad():
        x_smooth = smoother(x_tensor)
    X_smooth = x_smooth.squeeze(0).cpu().numpy().astype(np.float32)
    print("\nSmoothed X")
    print("shape:", X_smooth.shape)
    print("min :", float(X_smooth.min()))
    print("max :", float(X_smooth.max()))
    print("mean:", float(X_smooth.mean()))
    print("\n*** NO Z-SCORE ***")
    print("*** NO NORMALIZATION ***")
    return X_smooth

def compute_adv_epsilon(X):
    print("\n" + "=" * 80)
    print("COMPUTING ACORN EPSILON")
    print("=" * 80)
    x_tensor = torch.from_numpy(X).float()
    adv_epsilon = float(min_l2_distance(x_tensor)) / 2.0
    adv_epsilon = max(adv_epsilon, 1e-6)
    adv_alpha = adv_epsilon / 5.0
    print("epsilon:", adv_epsilon)
    print("alpha  :", adv_alpha)
    print("steps  :", ADV_STEPS)
    print("norm   :", ATTACK_NORM)
    return adv_epsilon

def build_model(adversarial=False, adv_epsilon=0.0):
    if adversarial:
        adv_alpha = adv_epsilon / 5.0
    else:
        adv_alpha = 0.0
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=TIME_OFFSETS,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=NUM_HIDDEN_UNITS,
        training_mode="adversarial" if adversarial else "clean",
        adv_alpha=adv_alpha if adversarial else 0.0,
        adv_epsilon=adv_epsilon if adversarial else 0.0,
        adv_steps=ADV_STEPS if adversarial else 0,
        attack_norm=ATTACK_NORM,
        device=DEVICE,
        verbose=True,
    )

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def orient_forward_jacobian(arr, n_neurons, latent_dim):
    a = np.abs(to_numpy(arr))
    a = np.squeeze(a)
    latent_axes = [i for i, size in enumerate(a.shape) if size == latent_dim]
    neuron_axes = [i for i, size in enumerate(a.shape) if size == n_neurons]
    if not latent_axes or not neuron_axes:
        raise RuntimeError(f"Cannot orient forward Jacobian. Raw shape={a.shape}; latent={latent_dim}; neurons={n_neurons}")
    latent_axis = latent_axes[-1]
    neuron_axis = neuron_axes[-1]
    if latent_axis == neuron_axis:
        raise RuntimeError(f"Ambiguous JF shape: {a.shape}")
    a = np.moveaxis(a, (latent_axis, neuron_axis), (-2, -1))
    if a.ndim > 2:
        a = a.mean(axis=tuple(range(a.ndim - 2)))
    if a.shape == (n_neurons, latent_dim):
        a = a.T
    expected = (latent_dim, n_neurons)
    if a.shape != expected:
        raise RuntimeError(f"JF final shape={a.shape}; expected={expected}")
    return a.astype(np.float32)

def compute_forward_jacobian(model, X, model_name):
    print("\n" + "=" * 80)
    print(f"FORWARD JACOBIAN: {model_name}")
    print("=" * 80)
    net = model.solver_.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    net.eval()
    if hasattr(net, "split_outputs"):
        net.split_outputs = False
    n_time, n_neurons = X.shape
    max_start = n_time - ATTR_CHUNK_LEN - 1
    if max_start <= 0:
        raise RuntimeError("Not enough samples for attribution.")
    starts = np.linspace(0, max_start, ATTR_N_CHUNKS, dtype=int)
    jf_sum = np.zeros((LATENT_DIM, n_neurons), dtype=np.float64)
    total_weight = 0
    print("Attribution chunks:", ATTR_N_CHUNKS)
    print("Chunk length:", ATTR_CHUNK_LEN)
    print("Attribution batch size:", ATTR_BATCH_SIZE)
    for chunk_index, start in enumerate(starts):
        stop = start + ATTR_CHUNK_LEN
        chunk = X[start:stop].astype(np.float32, copy=True)
        inp = torch.from_numpy(chunk).to(device)
        inp.requires_grad_(True)
        method = cebra.attribution.init(name="jacobian-based-batched", model=net, input_data=inp, output_dimension=LATENT_DIM)
        result = method.compute_attribution_map(batch_size=ATTR_BATCH_SIZE)
        if "jf" not in result:
            raise RuntimeError(f"No forward Jacobian in attribution result. Keys={list(result.keys())}")
        jf_raw = result["jf"]
        if chunk_index == 0:
            print("\nAttribution keys:", list(result.keys()))
            print("RAW JF shape:", to_numpy(jf_raw).shape)
        jf_chunk = orient_forward_jacobian(jf_raw, n_neurons=n_neurons, latent_dim=LATENT_DIM)
        weight = len(chunk)
        jf_sum += jf_chunk * weight
        total_weight += weight
        print(f"chunk {chunk_index + 1:02d}/{ATTR_N_CHUNKS} done")
        del method, result, jf_raw, jf_chunk, inp, chunk
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    jf = (jf_sum / total_weight).astype(np.float32)
    print("\nFINAL JF")
    print("shape:", jf.shape)
    print("meaning:", "latent × neuron")
    print("JF = mean absolute |dz/dx|")
    return jf

def train_and_attribute(X, adversarial=False, adv_epsilon=0.0):
    model_name = "ACORN" if adversarial else "CEBRA CLEAN"
    print("\n")
    print("#" * 90)
    print(f"TRAINING {model_name}")
    print("#" * 90)
    seed_all(SEED)
    model = build_model(adversarial=adversarial, adv_epsilon=adv_epsilon)
    print("Input shape:", X.shape)
    print("Latent dimension:", LATENT_DIM)
    print("Hidden units:", NUM_HIDDEN_UNITS)
    print("Iterations:", MAX_ITER)
    if adversarial:
        print("epsilon:", adv_epsilon)
        print("alpha:", adv_epsilon / 5.0)
        print("steps:", ADV_STEPS)
    model.fit(X.astype(np.float32, copy=False))
    jf = compute_forward_jacobian(model, X, model_name)
    return jf, model

def save_forward_plot(clean_jf, acorn_jf):
    vmax = max(float(np.nanmax(clean_jf)), float(np.nanmax(acorn_jf)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(27, 10), constrained_layout=True)
    im = axes[0].imshow(clean_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[0].set_title("CEBRA CLEAN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$", fontsize=17)
    axes[0].set_xlabel("Neuron / input column", fontsize=13)
    axes[0].set_ylabel("Latent dimension", fontsize=13)
    axes[1].imshow(acorn_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[1].set_title("ACORN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$", fontsize=17)
    axes[1].set_xlabel("Neuron / input column", fontsize=13)
    axes[1].set_ylabel("Latent dimension", fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.85, label="Mean absolute forward Jacobian")
    path = os.path.join(OUT, "JF_CLEAN_vs_ACORN.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved:")
    print(path)

def print_top_neurons(jf, unit_ids, name, top_k=10):
    scores = jf.mean(axis=0)
    order = np.argsort(scores)[::-1]
    print("\n" + "=" * 80)
    print(f"TOP {top_k} NEURONS — {name}")
    print("=" * 80)
    for rank, idx in enumerate(order[:top_k], start=1):
        print(f"{rank:2d}. input={idx:2d}  unit_id={unit_ids[idx]:5d}  score={scores[idx]:.12f}")

def main():
    print("\n" + "=" * 90)
    print("AREA2 BUMP")
    print("CEBRA CLEAN vs ACORN")
    print("FORWARD JACOBIAN dz/dx ONLY")
    print("=" * 90)
    print("NWB:", NWB_PATH)
    print("Bin:", BIN_MS, "ms")
    print("Gaussian SD:", SMOOTH_SD_MS, "ms")
    print("Latent:", LATENT_DIM)
    print("No normalization")
    print("No decoder")
    X_counts, unit_ids = build_spike_counts()
    X = smooth_spike_counts(X_counts)
    del X_counts
    gc.collect()
    adv_epsilon = compute_adv_epsilon(X)
    clean_jf, clean_model = train_and_attribute(X, adversarial=False, adv_epsilon=0.0)
    del clean_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    acorn_jf, acorn_model = train_and_attribute(X, adversarial=True, adv_epsilon=adv_epsilon)
    del acorn_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n" + "=" * 90)
    print("FINAL JACOBIANS")
    print("=" * 90)
    print("CLEAN JF:", clean_jf.shape)
    print("ACORN JF:", acorn_jf.shape)
    print("Expected:", (LATENT_DIM, len(unit_ids)))
    print_top_neurons(clean_jf, unit_ids, "CEBRA CLEAN")
    print_top_neurons(acorn_jf, unit_ids, "ACORN")
    save_forward_plot(clean_jf, acorn_jf)
    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print("JF definition: dz/dx")
    print("rows    = 128 latent dimensions")
    print("columns = 65 neurons")
    print("No decoder was trained.")
    print("No normalization was applied.")

if __name__ == "__main__":
    main()

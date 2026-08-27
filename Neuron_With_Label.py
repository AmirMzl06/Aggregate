import os
import sys
import gc
import random
import numpy as np
import pandas as pd
import h5py
import torch
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

SESSION_ID = 1119946360
NWB_PATH = f"data/AllenVBN/ecephys_sessions/ecephys_session_{SESSION_ID}.nwb"
UNITS_CSV = "data/units.csv"
OUT = f"AllenVBN_Jacobian_Plots_{SESSION_ID}"
os.makedirs(OUT, exist_ok=True)

BIN_MS = 50.0
BIN_SEC = BIN_MS / 1000.0
PRESENCE_RATIO_MIN = 0.90
ISI_VIOLATIONS_MAX = 0.50
AMPLITUDE_CUTOFF_MAX = 0.10

SEED = 42
LATENT_DIM = 64
BATCH_SIZE = 512
MAX_ITER = 3000
TEMPERATURE = 0.4
TIME_OFFSETS = 4
MODEL_ARCH = "offset36-model-more-dropout"
NUM_HIDDEN_UNITS = 64
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

def load_qc_metadata():
    print("\n" + "=" * 80)
    print("LOADING QC UNITS")
    print("=" * 80)
    units = pd.read_csv(UNITS_CSV)
    session_units = units[units["ecephys_session_id"] == SESSION_ID].copy()
    print("Total units:", len(session_units))
    qc = session_units[
        (session_units["presence_ratio"] >= PRESENCE_RATIO_MIN) &
        (session_units["isi_violations"] <= ISI_VIOLATIONS_MAX) &
        (session_units["amplitude_cutoff"] <= AMPLITUDE_CUTOFF_MAX)
    ].copy()
    print("QC-pass units:", len(qc))
    return qc

def get_active_block():
    print("\n" + "=" * 80)
    print("FINDING ACTIVE BLOCK")
    print("=" * 80)
    table = "intervals/Natural_Images_Lum_Matched_set_ophys_H_2019_presentations"
    with h5py.File(NWB_PATH, "r") as f:
        g = f[table]
        active = np.asarray(g["active"][:]).astype(bool)
        start_times = np.asarray(g["start_time"][:], dtype=np.float64)
        stop_times = np.asarray(g["stop_time"][:], dtype=np.float64)
        stimulus_block = np.asarray(g["stimulus_block"][:])
        mask = active & (stimulus_block == 0)
        t_start = float(start_times[mask].min())
        t_stop = float(stop_times[mask].max())
        n_presentations = int(mask.sum())
    print("Active presentations:", n_presentations)
    print("Start:", t_start, "sec")
    print("Stop:", t_stop, "sec")
    print("Duration:", round((t_stop - t_start) / 60.0, 2), "minutes")
    return t_start, t_stop

def build_continuous_activity(qc_metadata, t_start, t_stop):
    print("\n" + "=" * 80)
    print("BUILDING RAW SPIKE-COUNT MATRIX")
    print("=" * 80)
    qc_id_set = set(qc_metadata["unit_id"].astype(np.int64).tolist())
    edges = np.arange(t_start, t_stop + BIN_SEC, BIN_SEC, dtype=np.float64)
    n_bins = len(edges) - 1
    print("Bin size:", BIN_MS, "ms")
    print("Time bins:", n_bins)
    with h5py.File(NWB_PATH, "r") as f:
        ug = f["units"]
        nwb_unit_ids = np.asarray(ug["id"][:]).astype(np.int64)
        unit_positions = [i for i, unit_id in enumerate(nwb_unit_ids) if int(unit_id) in qc_id_set]
        unit_ids = nwb_unit_ids[unit_positions]
        print("QC units found:", len(unit_ids))
        if len(unit_ids) != len(qc_metadata):
            raise RuntimeError(f"Expected {len(qc_metadata)} QC units, found {len(unit_ids)}")
        n_units = len(unit_ids)
        X = np.zeros((n_bins, n_units), dtype=np.float32)
        spike_times = ug["spike_times"]
        spike_index = np.asarray(ug["spike_times_index"][:]).astype(np.int64)
        print("\nBinning spikes...")
        for output_column, nwb_position in enumerate(unit_positions):
            end_index = spike_index[nwb_position]
            start_index = 0 if nwb_position == 0 else spike_index[nwb_position - 1]
            spikes = np.asarray(spike_times[start_index:end_index], dtype=np.float64)
            left = np.searchsorted(spikes, t_start, side="left")
            right = np.searchsorted(spikes, t_stop, side="right")
            spikes = spikes[left:right]
            bin_idx = ((spikes - t_start) / BIN_SEC).astype(np.int64)
            valid = (bin_idx >= 0) & (bin_idx < n_bins)
            counts = np.bincount(bin_idx[valid], minlength=n_bins)
            X[:, output_column] = counts.astype(np.float32)
            if (output_column + 1) % 100 == 0 or (output_column + 1) == n_units:
                print(f"  {output_column + 1}/{n_units}")
    print("\nRAW X shape:", X.shape)
    print("Meaning: time × neuron")
    print("RAW X min:", float(X.min()))
    print("RAW X max:", float(X.max()))
    print("RAW X mean:", float(X.mean()))
    print("\n*** NO NORMALIZATION APPLIED ***")
    return X

def compute_adv_epsilon(train_x_np):
    print("\n" + "=" * 80)
    print("COMPUTING ADV EPSILON")
    print("=" * 80)
    train_tensor = torch.from_numpy(train_x_np).float()
    adv_epsilon = float(min_l2_distance(train_tensor)) / 2.0
    adv_epsilon = max(adv_epsilon, 1e-6)
    print("ADV epsilon:", adv_epsilon)
    print("ADV alpha:", adv_epsilon / 5.0)
    print("ADV steps:", ADV_STEPS)
    return adv_epsilon

def build_model(adversarial=False, adv_epsilon=0.0):
    adv_alpha = adv_epsilon / 5.0 if adversarial else 0.0
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=TIME_OFFSETS,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=NUM_HIDDEN_UNITS,
        training_mode="adversarial" if adversarial else "clean",
        adv_epsilon=adv_epsilon if adversarial else 0.0,
        adv_alpha=adv_alpha if adversarial else 0.0,
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
    latent_axes = [i for i, s in enumerate(a.shape) if s == latent_dim]
    neuron_axes = [i for i, s in enumerate(a.shape) if s == n_neurons]
    if not latent_axes or not neuron_axes:
        raise RuntimeError(f"Cannot orient JF. raw={a.shape}, latent={latent_dim}, neurons={n_neurons}")
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

def orient_inverse_jacobian(arr, n_neurons, latent_dim):
    a = np.abs(to_numpy(arr))
    a = np.squeeze(a)
    latent_axes = [i for i, s in enumerate(a.shape) if s == latent_dim]
    neuron_axes = [i for i, s in enumerate(a.shape) if s == n_neurons]
    if not latent_axes or not neuron_axes:
        raise RuntimeError(f"Cannot orient JFINV. raw={a.shape}, latent={latent_dim}, neurons={n_neurons}")
    latent_axis = latent_axes[-1]
    neuron_axis = neuron_axes[-1]
    if latent_axis == neuron_axis:
        raise RuntimeError(f"Ambiguous JFINV shape: {a.shape}")
    a = np.moveaxis(a, (neuron_axis, latent_axis), (-2, -1))
    if a.ndim > 2:
        a = a.mean(axis=tuple(range(a.ndim - 2)))
    if a.shape == (latent_dim, n_neurons):
        a = a.T
    expected = (n_neurons, latent_dim)
    if a.shape != expected:
        raise RuntimeError(f"JFINV final shape={a.shape}; expected={expected}")
    return a.astype(np.float32)

def compute_attribution(model, X, model_name):
    print("\n" + "=" * 80)
    print(f"ATTRIBUTION: {model_name}")
    print("=" * 80)
    net = model.solver_.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    net.eval()
    if hasattr(net, "split_outputs"):
        net.split_outputs = False
    n_time, n_neurons = X.shape
    if n_time < ATTR_CHUNK_LEN + 10:
        raise RuntimeError("Not enough time bins for attribution.")
    max_start = n_time - ATTR_CHUNK_LEN - 1
    starts = np.linspace(0, max_start, ATTR_N_CHUNKS, dtype=int)
    jf_sum = np.zeros((LATENT_DIM, n_neurons), dtype=np.float64)
    jfinv_sum = np.zeros((n_neurons, LATENT_DIM), dtype=np.float64)
    total_weight = 0
    print("Attribution chunks:", ATTR_N_CHUNKS)
    print("Chunk length:", ATTR_CHUNK_LEN)
    for chunk_index, start in enumerate(starts):
        stop = start + ATTR_CHUNK_LEN
        chunk = X[start:stop].astype(np.float32, copy=True)
        inp = torch.from_numpy(chunk).to(device)
        inp.requires_grad_(True)
        method = cebra.attribution.init(
            name="jacobian-based-batched",
            model=net,
            input_data=inp,
            output_dimension=LATENT_DIM,
        )
        result = method.compute_attribution_map(batch_size=ATTR_BATCH_SIZE)
        if "jf" not in result:
            raise RuntimeError(f"No jf. Keys={list(result.keys())}")
        jf_raw = result["jf"]
        if "jf-inv-svd" in result:
            jfinv_raw = result["jf-inv-svd"]
            inverse_key = "jf-inv-svd"
        elif "jf-inv" in result:
            jfinv_raw = result["jf-inv"]
            inverse_key = "jf-inv"
        else:
            raise RuntimeError(f"No inverse Jacobian. Keys={list(result.keys())}")
        if chunk_index == 0:
            print("\nAttribution keys:", list(result.keys()))
            print("RAW JF:", to_numpy(jf_raw).shape)
            print(f"RAW {inverse_key}:", to_numpy(jfinv_raw).shape)
        jf_chunk = orient_forward_jacobian(jf_raw, n_neurons, LATENT_DIM)
        jfinv_chunk = orient_inverse_jacobian(jfinv_raw, n_neurons, LATENT_DIM)
        weight = len(chunk)
        jf_sum += jf_chunk * weight
        jfinv_sum += jfinv_chunk * weight
        total_weight += weight
        print(f"chunk {chunk_index + 1:02d}/{ATTR_N_CHUNKS} done")
        del method, result, jf_raw, jfinv_raw, jf_chunk, jfinv_chunk, inp, chunk
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    jf = (jf_sum / total_weight).astype(np.float32)
    jfinv = (jfinv_sum / total_weight).astype(np.float32)
    print("\nFINAL ATTRIBUTION:")
    print("JF dz/dx:", jf.shape, "= latent × neuron")
    print("JFINV dx/dz:", jfinv.shape, "= neuron × latent")
    return jf, jfinv

def train_and_attribute(X, adversarial=False, adv_epsilon=0.0):
    model_name = "ACORN" if adversarial else "CEBRA CLEAN"
    print("\n" + "#" * 80)
    print(f"TRAINING {model_name}")
    print("#" * 80)
    seed_all(SEED)
    model = build_model(adversarial=adversarial, adv_epsilon=adv_epsilon)
    print("Input:", X.shape)
    if adversarial:
        print("epsilon:", adv_epsilon)
        print("alpha:", adv_epsilon / 5.0)
        print("steps:", ADV_STEPS)
    model.fit(X.astype(np.float32, copy=False))
    jf, jfinv = compute_attribution(model, X, model_name)
    return jf, jfinv, model

def save_forward_plot(clean_jf, acorn_jf):
    vmax = max(float(np.nanmax(clean_jf)), float(np.nanmax(acorn_jf)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(24, 7), constrained_layout=True)
    im = axes[0].imshow(clean_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[0].set_title("CEBRA CLEAN\n" r"$|J_F|=|\partial z/\partial x|$", fontsize=15)
    axes[0].set_xlabel("QC neuron")
    axes[0].set_ylabel("Latent dimension")
    axes[1].imshow(acorn_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[1].set_title("ACORN\n" r"$|J_F|=|\partial z/\partial x|$", fontsize=15)
    axes[1].set_xlabel("QC neuron")
    axes[1].set_ylabel("Latent dimension")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Mean absolute Jacobian")
    path = os.path.join(OUT, "JF_CLEAN_vs_ACORN.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved:", path)

def save_inverse_plot(clean_inv, acorn_inv):
    clean_plot = clean_inv.T
    acorn_plot = acorn_inv.T
    vmax = max(float(np.nanmax(clean_plot)), float(np.nanmax(acorn_plot)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(24, 7), constrained_layout=True)
    im = axes[0].imshow(clean_plot, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[0].set_title("CEBRA CLEAN\n" r"$|J_F^{-1}|\approx|\partial x/\partial z|$", fontsize=15)
    axes[0].set_xlabel("QC neuron")
    axes[0].set_ylabel("Latent dimension")
    axes[1].imshow(acorn_plot, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[1].set_title("ACORN\n" r"$|J_F^{-1}|\approx|\partial x/\partial z|$", fontsize=15)
    axes[1].set_xlabel("QC neuron")
    axes[1].set_ylabel("Latent dimension")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Mean absolute inverse Jacobian")
    path = os.path.join(OUT, "JFINV_CLEAN_vs_ACORN.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

def main():
    print("\n" + "=" * 80)
    print("ALLEN VBN")
    print("RAW SPIKE COUNTS")
    print("CEBRA CLEAN vs ACORN")
    print("=" * 80)
    qc_metadata = load_qc_metadata()
    t_start, t_stop = get_active_block()
    X = build_continuous_activity(qc_metadata, t_start, t_stop)
    train_x_np = X.astype(np.float32, copy=False)
    n_time, n_units = train_x_np.shape
    print("\nTRAIN DATA:")
    print("shape:", train_x_np.shape)
    print("time bins:", n_time)
    print("neurons:", n_units)
    print("\nNO NORMALIZATION.")
    adv_epsilon = compute_adv_epsilon(train_x_np)
    clean_jf, clean_inv, clean_model = train_and_attribute(train_x_np, adversarial=False, adv_epsilon=0.0)
    del clean_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    acorn_jf, acorn_inv, acorn_model = train_and_attribute(train_x_np, adversarial=True, adv_epsilon=adv_epsilon)
    del acorn_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n" + "=" * 80)
    print("FINAL SHAPES")
    print("=" * 80)
    print("CLEAN JF:", clean_jf.shape)
    print("ACORN JF:", acorn_jf.shape)
    print("CLEAN JFINV:", clean_inv.shape)
    print("ACORN JFINV:", acorn_inv.shape)
    assert clean_jf.shape == (LATENT_DIM, n_units)
    assert acorn_jf.shape == (LATENT_DIM, n_units)
    assert clean_inv.shape == (n_units, LATENT_DIM)
    assert acorn_inv.shape == (n_units, LATENT_DIM)
    save_forward_plot(clean_jf, acorn_jf)
    save_inverse_plot(clean_inv, acorn_inv)
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print("\nONLY THESE TWO PNGs:")
    print(os.path.join(OUT, "JF_CLEAN_vs_ACORN.png"))
    print(os.path.join(OUT, "JFINV_CLEAN_vs_ACORN.png"))

if __name__ == "__main__":
    main()

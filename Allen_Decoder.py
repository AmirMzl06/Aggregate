import os
import sys
import gc
import math
import numbers
import random
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
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

SESSION_ID = 1105543760
NWB_PATH = f"data/AllenVBN/ecephys_sessions/ecephys_session_{SESSION_ID}.nwb"
UNITS_CSV = "data/units.csv"
OUT = f"AllenVBN_9Class_Decoder_{SESSION_ID}"
os.makedirs(OUT, exist_ok=True)

BIN_MS = 50.0
BIN_SEC = BIN_MS / 1000.0
SMOOTH_SD_MS = 100.0
SMOOTH_SIGMA_BINS = SMOOTH_SD_MS / BIN_MS
SMOOTH_KERNEL_SIZE = 17

PRESENCE_RATIO_MIN = 0.90
ISI_VIOLATIONS_MAX = 0.50
AMPLITUDE_CUTOFF_MAX = 0.10

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
N_DECODER_FOLDS = 5
DECODER_MAX_ITER = 5000

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
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError("Only 1D, 2D and 3D Gaussian smoothing supported.")
    def forward(self, input):
        input = torch.permute(input, (0, 2, 1))
        input = self.conv(input, weight=self.weight, groups=self.groups, padding="same")
        input = torch.permute(input, (0, 2, 1))
        return input

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
    print("FINDING ACTIVE NATURAL-IMAGE BLOCK")
    print("=" * 80)
    with h5py.File(NWB_PATH, "r") as f:
        interval_names = list(f["intervals"].keys())
        candidates = [name for name in interval_names if "Natural_Images" in name and "presentations" in name]
        if len(candidates) == 0:
            raise RuntimeError("No Natural Images presentation table found.\nAvailable tables: " + str(interval_names))
        if len(candidates) > 1:
            print("\nWARNING: multiple Natural Images tables found:")
            for name in candidates:
                print(" ", name)
        table_name = candidates[0]
        print("Using table:")
        print(table_name)
        g = f["intervals"][table_name]
        active = np.asarray(g["active"][:]).astype(bool)
        start_times = np.asarray(g["start_time"][:], dtype=np.float64)
        stop_times = np.asarray(g["stop_time"][:], dtype=np.float64)
        mask = active.copy()
        if "stimulus_block" in g:
            stimulus_block = np.asarray(g["stimulus_block"][:])
            if np.any(active & (stimulus_block == 0)):
                mask = active & (stimulus_block == 0)
                print("Using active stimulus_block == 0")
        if mask.sum() == 0:
            raise RuntimeError("No active Natural Images presentations found.")
        t_start = float(start_times[mask].min())
        t_stop = float(stop_times[mask].max())
        n_presentations = int(mask.sum())
    print("Active presentations:", n_presentations)
    print("Start:", t_start, "sec")
    print("Stop:", t_stop, "sec")
    print("Duration:", round((t_stop - t_start) / 60.0, 2), "minutes")
    return t_start, t_stop, table_name

def build_spike_counts(qc_metadata, t_start, t_stop):
    print("\n" + "=" * 80)
    print("BUILDING RAW 50 ms SPIKE COUNTS")
    print("=" * 80)
    qc_id_set = set(qc_metadata["unit_id"].astype(np.int64).tolist())
    edges = np.arange(t_start, t_stop + BIN_SEC, BIN_SEC, dtype=np.float64)
    n_bins = len(edges) - 1
    print("Bin width:", BIN_MS, "ms")
    print("Time bins:", n_bins)
    with h5py.File(NWB_PATH, "r") as f:
        units_group = f["units"]
        nwb_unit_ids = np.asarray(units_group["id"][:]).astype(np.int64)
        unit_positions = [i for i, unit_id in enumerate(nwb_unit_ids) if int(unit_id) in qc_id_set]
        unit_ids = nwb_unit_ids[unit_positions]
        print("QC units found in NWB:", len(unit_ids))
        if len(unit_ids) != len(qc_metadata):
            raise RuntimeError(f"Expected {len(qc_metadata)} QC units but found {len(unit_ids)}")
        n_units = len(unit_ids)
        X = np.zeros((n_bins, n_units), dtype=np.float32)
        spike_times = units_group["spike_times"]
        spike_index = np.asarray(units_group["spike_times_index"][:]).astype(np.int64)
        print("\nBinning spikes...")
        for output_column, nwb_position in enumerate(unit_positions):
            end_index = spike_index[nwb_position]
            start_index = 0 if nwb_position == 0 else spike_index[nwb_position - 1]
            spikes = np.asarray(spike_times[start_index:end_index], dtype=np.float64)
            left = np.searchsorted(spikes, t_start, side="left")
            right = np.searchsorted(spikes, t_stop, side="right")
            spikes = spikes[left:right]
            bin_index = ((spikes - t_start) / BIN_SEC).astype(np.int64)
            valid = (bin_index >= 0) & (bin_index < n_bins)
            counts = np.bincount(bin_index[valid], minlength=n_bins)
            X[:, output_column] = counts.astype(np.float32)
            if (output_column + 1) % 100 == 0 or (output_column + 1) == n_units:
                print(f"  {output_column + 1}/{n_units}")
    print("\nRAW X shape:", X.shape)
    print("RAW X min:", float(X.min()))
    print("RAW X max:", float(X.max()))
    print("RAW X mean:", float(X.mean()))
    return X

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
    print("\nSmoothed X shape:", X_smooth.shape)
    print("Smoothed min:", float(X_smooth.min()))
    print("Smoothed max:", float(X_smooth.max()))
    print("Smoothed mean:", float(X_smooth.mean()))
    print("\n*** GAUSSIAN SMOOTHING ONLY ***")
    print("*** NO Z-SCORE ***")
    print("*** NO NORMALIZATION ***")
    return X_smooth

def decode_string_array(arr):
    return np.asarray([x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr])

def build_9class_labels(t_start, t_stop, n_bins, table_name):
    print("\n" + "=" * 80)
    print("BUILDING 9-CLASS LABELS")
    print("8 NATURAL IMAGES + GRAY")
    print("=" * 80)
    bin_centers = t_start + (np.arange(n_bins, dtype=np.float64) + 0.5) * BIN_SEC
    label_text = np.full(n_bins, "GRAY", dtype=object)
    with h5py.File(NWB_PATH, "r") as f:
        g = f["intervals"][table_name]
        active = np.asarray(g["active"][:]).astype(bool)
        start_times = np.asarray(g["start_time"][:], dtype=np.float64)
        stop_times = np.asarray(g["stop_time"][:], dtype=np.float64)
        image_names = decode_string_array(np.asarray(g["image_name"][:]))
        selected = active.copy()
        if "stimulus_block" in g:
            stimulus_block = np.asarray(g["stimulus_block"][:])
            if np.any(active & (stimulus_block == 0)):
                selected = active & (stimulus_block == 0)
        if "omitted" in g:
            omitted = np.asarray(g["omitted"][:]).astype(bool)
        else:
            omitted = np.zeros(len(start_times), dtype=bool)
        valid_image_presentations = selected & (~omitted)
        image_classes = sorted(np.unique(image_names[valid_image_presentations]).tolist())
        if len(image_classes) != 8:
            raise RuntimeError(f"Expected exactly 8 natural-image classes, found {len(image_classes)}: {image_classes}")
        for row_index in np.where(valid_image_presentations)[0]:
            start = start_times[row_index]
            stop = stop_times[row_index]
            image_name = image_names[row_index]
            left = np.searchsorted(bin_centers, start, side="left")
            right = np.searchsorted(bin_centers, stop, side="left")
            left = max(left, 0)
            right = min(right, n_bins)
            if right > left:
                label_text[left:right] = image_name
    class_names = image_classes + ["GRAY"]
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    y = np.asarray([class_to_id[name] for name in label_text], dtype=np.int64)
    print("\nClass mapping:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")
    print("\nLabel counts across ALL 50 ms bins:")
    counts = np.bincount(y, minlength=len(class_names))
    for idx, name in enumerate(class_names):
        percentage = 100.0 * counts[idx] / len(y)
        print(f"  {name:12s}: {counts[idx]:6d} bins ({percentage:6.2f}%)")
    print("\nTotal labeled bins:", len(y))
    print("Unlabeled bins: 0")
    return y, class_names, bin_centers

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
        verbose=True,
        training_mode="adversarial" if adversarial else "clean",
        adv_alpha=adv_alpha if adversarial else 0.0,
        adv_epsilon=adv_epsilon if adversarial else 0.0,
        adv_steps=ADV_STEPS if adversarial else 0,
        attack_norm=ATTACK_NORM,
        device=DEVICE,
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

def orient_inverse_jacobian(arr, n_neurons, latent_dim):
    a = np.abs(to_numpy(arr))
    a = np.squeeze(a)
    latent_axes = [i for i, size in enumerate(a.shape) if size == latent_dim]
    neuron_axes = [i for i, size in enumerate(a.shape) if size == n_neurons]
    if not latent_axes or not neuron_axes:
        raise RuntimeError(f"Cannot orient inverse Jacobian. Raw shape={a.shape}; latent={latent_dim}; neurons={n_neurons}")
    latent_axis = latent_axes[-1]
    neuron_axis = neuron_axes[-1]
    if latent_axis == neuron_axis:
        raise RuntimeError(f"Ambiguous inverse shape: {a.shape}")
    a = np.moveaxis(a, (neuron_axis, latent_axis), (-2, -1))
    if a.ndim > 2:
        a = a.mean(axis=tuple(range(a.ndim - 2)))
    if a.shape == (latent_dim, n_neurons):
        a = a.T
    expected = (n_neurons, latent_dim)
    if a.shape != expected:
        raise RuntimeError(f"Inverse final shape={a.shape}; expected={expected}")
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
    max_start = n_time - ATTR_CHUNK_LEN - 1
    if max_start <= 0:
        raise RuntimeError("Not enough samples for attribution.")
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
            raise RuntimeError(f"No JF found. Keys={list(result.keys())}")
        jf_raw = result["jf"]
        if "jf-inv-svd" in result:
            jfinv_raw = result["jf-inv-svd"]
            inverse_key = "jf-inv-svd"
        elif "jf-inv" in result:
            jfinv_raw = result["jf-inv"]
            inverse_key = "jf-inv"
        elif "jf-inv-lsq" in result:
            jfinv_raw = result["jf-inv-lsq"]
            inverse_key = "jf-inv-lsq"
        else:
            raise RuntimeError(f"No inverse Jacobian. Keys={list(result.keys())}")
        if chunk_index == 0:
            print("\nAttribution keys:", list(result.keys()))
            print("RAW JF shape:", to_numpy(jf_raw).shape)
            print(f"RAW {inverse_key} shape:", to_numpy(jfinv_raw).shape)
        jf_chunk = orient_forward_jacobian(jf_raw, n_neurons=n_neurons, latent_dim=LATENT_DIM)
        jfinv_chunk = orient_inverse_jacobian(jfinv_raw, n_neurons=n_neurons, latent_dim=LATENT_DIM)
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
    print("Input shape:", X.shape)
    print("Latent dimension:", LATENT_DIM)
    print("Hidden units:", NUM_HIDDEN_UNITS)
    if adversarial:
        print("epsilon:", adv_epsilon)
        print("alpha:", adv_epsilon / 5.0)
        print("steps:", ADV_STEPS)
    model.fit(X.astype(np.float32, copy=False))
    jf, jfinv = compute_attribution(model, X, model_name)
    return jf, jfinv, model

def get_model_offset(model):
    net = model.solver_.model
    if not hasattr(net, "get_offset"):
        return 0, 0
    offset = net.get_offset()
    left = int(getattr(offset, "left", 0))
    right = int(getattr(offset, "right", 0))
    return left, right

def transform_and_align(model, X, y, model_name):
    print("\n" + "=" * 80)
    print(f"TRANSFORMING: {model_name}")
    print("=" * 80)
    X32 = X.astype(np.float32, copy=False)
    z_raw = model.transform(X32)
    Z = to_numpy(z_raw).astype(np.float32)
    if Z.ndim != 2:
        raise RuntimeError(f"Unexpected embedding shape for {model_name}: {Z.shape}")
    n_input = len(X)
    n_embed = len(Z)
    left_offset, right_offset = get_model_offset(model)
    print("Input bins:", n_input)
    print("Embedding bins:", n_embed)
    print("Embedding dimension:", Z.shape[1])
    print("Model offset:", f"left={left_offset}, right={right_offset}")
    if n_embed == n_input:
        time_indices = np.arange(n_input, dtype=np.int64)
        y_aligned = y.copy()
        print("Alignment: one embedding per original 50 ms bin.")
    else:
        start = left_offset
        stop = start + n_embed
        if start < 0 or stop > n_input:
            removed = n_input - n_embed
            start = removed // 2
            stop = start + n_embed
        if start < 0 or stop > n_input:
            raise RuntimeError(f"Cannot align {model_name} embedding length {n_embed} to input length {n_input}.")
        time_indices = np.arange(start, stop, dtype=np.int64)
        y_aligned = y[time_indices]
        print(f"Alignment: valid temporal crop [{start}:{stop}]")
    if len(y_aligned) != len(Z):
        raise RuntimeError(f"Alignment mismatch for {model_name}: Z={len(Z)}, y={len(y_aligned)}")
    return Z, y_aligned, time_indices, left_offset, right_offset

def run_temporal_decoder_cv(Z, y, class_names, guard_bins, model_name):
    print("\n" + "=" * 100)
    print(f"9-CLASS TEMPORAL DECODER: {model_name}")
    print("=" * 100)
    n_samples = len(y)
    n_classes = len(class_names)
    if n_classes != 9:
        raise RuntimeError(f"Expected 9 decoder classes, got {n_classes}.")
    if len(Z) != n_samples:
        raise RuntimeError(f"Z/y mismatch: {len(Z)} vs {n_samples}")
    counts = np.bincount(y, minlength=n_classes)
    majority_accuracy = counts.max() / counts.sum()
    print("Samples:", n_samples)
    print("Features:", Z.shape[1])
    print("Classes:", n_classes)
    print("Folds:", N_DECODER_FOLDS)
    print("Guard bins:", guard_bins)
    print("Guard duration:", f"{guard_bins * BIN_MS:.1f} ms per side")
    print("Uniform 9-class chance:", f"{100.0 / n_classes:.2f}%")
    print("Majority-class raw-accuracy baseline:", f"{100.0 * majority_accuracy:.2f}%")
    all_indices = np.arange(n_samples, dtype=np.int64)
    folds = np.array_split(all_indices, N_DECODER_FOLDS)
    all_predictions = np.full(n_samples, -1, dtype=np.int64)
    fold_rows = []
    for fold_number, test_idx in enumerate(folds, start=1):
        if len(test_idx) == 0:
            continue
        test_start = int(test_idx[0])
        test_stop = int(test_idx[-1]) + 1
        train_mask = np.ones(n_samples, dtype=bool)
        guard_start = max(0, test_start - guard_bins)
        guard_stop = min(n_samples, test_stop + guard_bins)
        train_mask[guard_start:guard_stop] = False
        train_idx = np.where(train_mask)[0]
        train_classes = np.unique(y[train_idx])
        test_classes = np.unique(y[test_idx])
        if len(train_classes) != n_classes:
            raise RuntimeError(f"Fold {fold_number}: training set does not contain all 9 classes. Present={train_classes.tolist()}")
        if len(test_classes) != n_classes:
            print(f"WARNING Fold {fold_number}: test block contains {len(test_classes)}/9 classes.")
        decoder = LogisticRegression(max_iter=DECODER_MAX_ITER, class_weight="balanced", solver="lbfgs", random_state=SEED)
        decoder.fit(Z[train_idx], y[train_idx])
        pred = decoder.predict(Z[test_idx]).astype(np.int64)
        all_predictions[test_idx] = pred
        acc = accuracy_score(y[test_idx], pred)
        bacc = balanced_accuracy_score(y[test_idx], pred)
        macro_f1 = f1_score(y[test_idx], pred, average="macro", labels=np.arange(n_classes), zero_division=0)
        fold_rows.append({"fold": fold_number, "n_train": len(train_idx), "n_test": len(test_idx), "accuracy": acc, "balanced_accuracy": bacc, "macro_f1": macro_f1})
        print(f"Fold {fold_number}: train={len(train_idx):6d}  test={len(test_idx):6d}  ACC={acc:.4f}  BACC={bacc:.4f}  MacroF1={macro_f1:.4f}")
    if np.any(all_predictions < 0):
        missing = int(np.sum(all_predictions < 0))
        raise RuntimeError(f"Decoder predictions missing for {missing} samples.")
    overall_acc = accuracy_score(y, all_predictions)
    overall_bacc = balanced_accuracy_score(y, all_predictions)
    overall_macro_f1 = f1_score(y, all_predictions, average="macro", labels=np.arange(n_classes), zero_division=0)
    cm = confusion_matrix(y, all_predictions, labels=np.arange(n_classes))
    recalls = np.divide(np.diag(cm), cm.sum(axis=1), out=np.zeros(n_classes, dtype=np.float64), where=cm.sum(axis=1) != 0)
    print("\n" + "-" * 100)
    print(f"FINAL {model_name} DECODER RESULTS")
    print("-" * 100)
    print(f"Accuracy          : {overall_acc:.6f}")
    print(f"Balanced Accuracy : {overall_bacc:.6f}")
    print(f"Macro F1          : {overall_macro_f1:.6f}")
    print("\nPer-class recall:")
    for idx, name in enumerate(class_names):
        print(f"  {name:12s}: {recalls[idx]:.6f}")
    print("\nConfusion matrix (ROWS=true, COLS=predicted):")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df.to_string())
    fold_df = pd.DataFrame(fold_rows)
    print("\nFold mean ± std:")
    for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
        mean_value = fold_df[metric].mean()
        std_value = fold_df[metric].std(ddof=0)
        print(f"  {metric:20s}: {mean_value:.6f} ± {std_value:.6f}")
    return {"accuracy": overall_acc, "balanced_accuracy": overall_bacc, "macro_f1": overall_macro_f1}

def save_forward_plot(clean_jf, acorn_jf):
    vmax = max(float(np.nanmax(clean_jf)), float(np.nanmax(acorn_jf)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(27, 10), constrained_layout=True)
    im = axes[0].imshow(clean_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[0].set_title("CEBRA CLEAN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$", fontsize=17)
    axes[0].set_xlabel("QC neuron", fontsize=13)
    axes[0].set_ylabel("Latent dimension", fontsize=13)
    axes[1].imshow(acorn_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[1].set_title("ACORN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$", fontsize=17)
    axes[1].set_xlabel("QC neuron", fontsize=13)
    axes[1].set_ylabel("Latent dimension", fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.85, label="Mean absolute forward Jacobian")
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
    fig, axes = plt.subplots(1, 2, figsize=(27, 10), constrained_layout=True)
    im = axes[0].imshow(clean_plot, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[0].set_title("CEBRA CLEAN\n" r"$\mathrm{Mean}\ |\partial x/\partial z|$", fontsize=17)
    axes[0].set_xlabel("QC neuron", fontsize=13)
    axes[0].set_ylabel("Latent dimension", fontsize=13)
    axes[1].imshow(acorn_plot, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[1].set_title("ACORN\n" r"$\mathrm{Mean}\ |\partial x/\partial z|$", fontsize=17)
    axes[1].set_xlabel("QC neuron", fontsize=13)
    axes[1].set_ylabel("Latent dimension", fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.85, label="Mean absolute inverse Jacobian")
    path = os.path.join(OUT, "JFINV_CLEAN_vs_ACORN.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

def main():
    print("\n" + "=" * 100)
    print("ALLEN VBN: CEBRA CLEAN vs ACORN + 9-CLASS DECODER")
    print("=" * 100)
    print("Session:", SESSION_ID)
    print("50 ms bins")
    print("Gaussian SD = 100 ms")
    print("NO Z-SCORE")
    print("NO NORMALIZATION")
    print("Decoder = 8 images + GRAY")
    print("Decoder = temporally blocked 5-fold linear probe")
    print("NO RAW-X DECODER")
    print("SAVED FILES = ONLY TWO JACOBIAN PNGs")

    qc_metadata = load_qc_metadata()
    t_start, t_stop, table_name = get_active_block()
    X_counts = build_spike_counts(qc_metadata, t_start, t_stop)
    X_smooth = smooth_spike_counts(X_counts)
    del X_counts
    gc.collect()
    train_x_np = X_smooth.astype(np.float32, copy=False)
    n_time, n_units = train_x_np.shape
    print("\n" + "=" * 80)
    print("FINAL MODEL INPUT")
    print("=" * 80)
    print("shape:", train_x_np.shape)
    print("time bins:", n_time)
    print("neurons:", n_units)
    print("bin:", BIN_MS, "ms")
    print("Gaussian SD:", SMOOTH_SD_MS, "ms")
    print("sigma:", SMOOTH_SIGMA_BINS, "bins")
    print("kernel:", SMOOTH_KERNEL_SIZE)
    print("\n*** NO NORMALIZATION ***")

    y_full, class_names, bin_centers = build_9class_labels(t_start, t_stop, n_time, table_name)
    assert len(y_full) == n_time
    assert len(bin_centers) == n_time
    assert len(class_names) == 9

    adv_epsilon = compute_adv_epsilon(train_x_np)

    clean_jf, clean_inv, clean_model = train_and_attribute(train_x_np, adversarial=False, adv_epsilon=0.0)
    Z_clean, y_clean, clean_time_indices, clean_left_offset, clean_right_offset = transform_and_align(clean_model, train_x_np, y_full, "CEBRA CLEAN")
    clean_guard_bins = max(clean_left_offset, clean_right_offset) + SMOOTH_KERNEL_SIZE // 2
    del clean_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    acorn_jf, acorn_inv, acorn_model = train_and_attribute(train_x_np, adversarial=True, adv_epsilon=adv_epsilon)
    Z_acorn, y_acorn, acorn_time_indices, acorn_left_offset, acorn_right_offset = transform_and_align(acorn_model, train_x_np, y_full, "ACORN")
    acorn_guard_bins = max(acorn_left_offset, acorn_right_offset) + SMOOTH_KERNEL_SIZE // 2
    del acorn_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not np.array_equal(clean_time_indices, acorn_time_indices):
        raise RuntimeError("CLEAN and ACORN embeddings are not aligned to the same time bins.")
    if not np.array_equal(y_clean, y_acorn):
        raise RuntimeError("CLEAN and ACORN decoder labels differ.")
    if len(Z_clean) != len(Z_acorn):
        raise RuntimeError("CLEAN and ACORN embedding lengths differ.")
    guard_bins = max(clean_guard_bins, acorn_guard_bins)
    print("\n" + "=" * 80)
    print("DECODER ALIGNMENT CHECK")
    print("=" * 80)
    print("CLEAN Z:", Z_clean.shape)
    print("ACORN Z:", Z_acorn.shape)
    print("labels:", y_clean.shape)
    print("shared guard bins:", guard_bins)
    print("shared guard duration:", guard_bins * BIN_MS, "ms")

    clean_decoder_result = run_temporal_decoder_cv(Z_clean, y_clean, class_names, guard_bins=guard_bins, model_name="CEBRA CLEAN")
    acorn_decoder_result = run_temporal_decoder_cv(Z_acorn, y_acorn, class_names, guard_bins=guard_bins, model_name="ACORN")

    print("\n" + "=" * 100)
    print("FINAL 9-CLASS DECODER COMPARISON")
    print("=" * 100)
    comparison = pd.DataFrame([
        {"model": "CEBRA CLEAN", **clean_decoder_result},
        {"model": "ACORN", **acorn_decoder_result},
    ])
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n" + "=" * 80)
    print("FINAL JACOBIAN SHAPES")
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

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print("\nONLY THESE FILES WERE SAVED BY THIS SCRIPT:")
    print(os.path.join(OUT, "JF_CLEAN_vs_ACORN.png"))
    print(os.path.join(OUT, "JFINV_CLEAN_vs_ACORN.png"))
    print("\nDecoder results were PRINTED ONLY; not saved.")

if __name__ == "__main__":
    main()

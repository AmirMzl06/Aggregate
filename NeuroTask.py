### TopK
import os
import sys
import gc
import math
import numbers
import random
import warnings
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
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
OUT = "Area2_Bump_TopK"
os.makedirs(OUT, exist_ok=True)

BIN_MS = 50.0
BIN_SEC = BIN_MS / 1000.0
SMOOTH_SD_MS = 100.0
SMOOTH_SIGMA_BINS = SMOOTH_SD_MS / BIN_MS
SMOOTH_KERNEL_SIZE = 17
TRAIN_FRAC = 0.80
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
DECODER_HIDDEN = 64
DECODER_DROPOUT = 0.4
DECODER_EPOCHS = 2500
DECODER_LR = 1e-3
DECODER_WEIGHT_DECAY = 2e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch device:", device)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_all(SEED)

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        if dim != 1:
            raise RuntimeError("Only 1D smoothing is used.")
        self.conv = F.conv1d
    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.conv(x, weight=self.weight, groups=self.groups, padding="same")
        x = torch.permute(x, (0, 2, 1))
        return x

def load_area2():
    print("\n" + "=" * 100)
    print("LOADING AREA2 BUMP")
    print("=" * 100)
    with h5py.File(NWB_PATH, "r") as f:
        vel_group = f["processing/behavior/hand_vel"]
        vel_ds = vel_group["data"]
        n_behavior_samples = vel_ds.shape[0]
        if "starting_time" in vel_group:
            st = vel_group["starting_time"]
            t_start = float(st[()])
            behavior_rate = float(st.attrs["rate"])
        else:
            raise RuntimeError("hand_vel starting_time not found.")
        print("Behavior raw shape:", vel_ds.shape)
        print("Behavior rate:", behavior_rate, "Hz")
        samples_per_bin = int(round(behavior_rate * BIN_SEC))
        print("Samples per 50-ms bin:", samples_per_bin)
        n_bins = n_behavior_samples // samples_per_bin
        usable_samples = n_bins * samples_per_bin
        t_stop = t_start + n_bins * BIN_SEC
        print("Complete bins:", n_bins)
        print("t_start:", t_start)
        print("t_stop :", t_stop)
        print("Duration:", (t_stop - t_start) / 60.0, "minutes")
        vel_raw = np.asarray(vel_ds[:usable_samples], dtype=np.float32)
        if vel_raw.ndim != 2 or vel_raw.shape[1] != 2:
            raise RuntimeError(f"Expected hand_vel shape (T, 2), got {vel_raw.shape}")
        vel_reshaped = vel_raw.reshape(n_bins, samples_per_bin, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            Y = np.nanmean(vel_reshaped, axis=1).astype(np.float32)
        print("\nHand velocity Y:")
        print("shape:", Y.shape)
        print("NaN count:", int(np.isnan(Y).sum()))
        units = f["units"]
        unit_ids = np.asarray(units["id"][:], dtype=np.int64)
        heldout = np.asarray(units["heldout"][:], dtype=bool)
        n_units = len(unit_ids)
        print("\nUnits:")
        print("Total:", n_units)
        print("Heldout flagged:", int(heldout.sum()))
        print("Using ALL units:", n_units)
        spike_times = units["spike_times"]
        spike_index = np.asarray(units["spike_times_index"][:], dtype=np.int64)
        X_counts = np.zeros((n_bins, n_units), dtype=np.float32)
        print("\nBinning spikes...")
        for neuron_idx in range(n_units):
            end_idx = int(spike_index[neuron_idx])
            start_idx = 0 if neuron_idx == 0 else int(spike_index[neuron_idx - 1])
            spikes = np.asarray(spike_times[start_idx:end_idx], dtype=np.float64)
            left = np.searchsorted(spikes, t_start, side="left")
            right = np.searchsorted(spikes, t_stop, side="left")
            spikes = spikes[left:right]
            bin_idx = ((spikes - t_start) / BIN_SEC).astype(np.int64)
            valid = (bin_idx >= 0) & (bin_idx < n_bins)
            counts = np.bincount(bin_idx[valid], minlength=n_bins)
            X_counts[:, neuron_idx] = counts[:n_bins].astype(np.float32)
    print("\nRAW X:")
    print("shape:", X_counts.shape)
    print("min :", float(X_counts.min()))
    print("max :", float(X_counts.max()))
    print("mean:", float(X_counts.mean()))
    assert X_counts.shape[0] == Y.shape[0]
    return X_counts, Y, unit_ids

def smooth_neural(X):
    print("\n" + "=" * 100)
    print("GAUSSIAN SMOOTHING")
    print("=" * 100)
    print("Input:", X.shape)
    print("Bin:", BIN_MS, "ms")
    print("Gaussian SD:", SMOOTH_SD_MS, "ms")
    print("Sigma:", SMOOTH_SIGMA_BINS, "bins")
    print("Kernel size:", SMOOTH_KERNEL_SIZE)
    n_neurons = X.shape[1]
    x_tensor = torch.from_numpy(X.astype(np.float32, copy=False)).unsqueeze(0)
    smoother = GaussianSmoothing(channels=n_neurons, kernel_size=SMOOTH_KERNEL_SIZE, sigma=SMOOTH_SIGMA_BINS, dim=1)
    smoother.eval()
    with torch.no_grad():
        X_smooth = smoother(x_tensor).squeeze(0).cpu().numpy().astype(np.float32)
    print("\nSmoothed X:")
    print("shape:", X_smooth.shape)
    print("min :", float(X_smooth.min()))
    print("max :", float(X_smooth.max()))
    print("mean:", float(X_smooth.mean()))
    print("\n*** NO Z-SCORE ***")
    print("*** NO NORMALIZATION ***")
    return X_smooth

def compute_adv_epsilon(train_x_np):
    print("\n" + "=" * 100)
    print("COMPUTING ACORN EPSILON")
    print("=" * 100)
    x_tensor = torch.from_numpy(train_x_np).float()
    adv_epsilon = float(min_l2_distance(x_tensor)) / 2.0
    adv_epsilon = max(adv_epsilon, 1e-6)
    print("epsilon:", adv_epsilon)
    print("alpha:", adv_epsilon / 5.0)
    print("steps:", ADV_STEPS)
    print("norm:", ATTACK_NORM)
    return adv_epsilon

def build_cebra(adversarial=False, adv_epsilon=0.0):
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

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=2, dropout_rate=0.4):
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

def get_embeddings(model, X):
    x_t = torch.from_numpy(X.astype(np.float32, copy=False)).float()
    z = model.transform(x_t)
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    z = np.asarray(z, dtype=np.float32)
    return z

def train_decoder(z_train_np, y_train_np, z_test_np, y_test_np, model_name):
    print("\n" + "=" * 100)
    print(f"TRAINING DECODER — {model_name}")
    print("=" * 100)
    print("\nFinite-value check:")
    print("NaN z_train:", int(np.isnan(z_train_np).sum()))
    print("NaN y_train:", int(np.isnan(y_train_np).sum()))
    print("NaN z_test :", int(np.isnan(z_test_np).sum()))
    print("NaN y_test :", int(np.isnan(y_test_np).sum()))
    train_mask = np.isfinite(z_train_np).all(axis=1) & np.isfinite(y_train_np).all(axis=1)
    test_mask = np.isfinite(z_test_np).all(axis=1) & np.isfinite(y_test_np).all(axis=1)
    print("Decoder train bins before:", len(z_train_np))
    print("Decoder train bins valid :", int(train_mask.sum()))
    print("Decoder test bins before :", len(z_test_np))
    print("Decoder test bins valid  :", int(test_mask.sum()))
    z_train_np = z_train_np[train_mask]
    y_train_np = y_train_np[train_mask]
    z_test_np = z_test_np[test_mask]
    y_test_np = y_test_np[test_mask]
    assert len(z_train_np) > 0
    assert len(z_test_np) > 0
    assert np.isfinite(z_train_np).all()
    assert np.isfinite(y_train_np).all()
    assert np.isfinite(z_test_np).all()
    assert np.isfinite(y_test_np).all()
    print("\nTrain embedding:", z_train_np.shape)
    print("Train target:", y_train_np.shape)
    print("Test embedding:", z_test_np.shape)
    print("Test target:", y_test_np.shape)
    seed_all(SEED)
    decoder = TwoLayerMLP(
        input_dim=z_train_np.shape[1],
        hidden_dim=DECODER_HIDDEN,
        output_dim=2,
        dropout_rate=DECODER_DROPOUT,
    ).to(device)
    z_train = torch.from_numpy(z_train_np).float().to(device)
    y_train = torch.from_numpy(y_train_np).float().to(device)
    z_test = torch.from_numpy(z_test_np).float().to(device)
    y_test = torch.from_numpy(y_test_np).float().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR, weight_decay=DECODER_WEIGHT_DECAY)
    for epoch in range(DECODER_EPOCHS):
        decoder.train()
        optimizer.zero_grad()
        prediction = decoder(z_train)
        loss = criterion(prediction, y_train)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 500 == 0:
            print(f"{model_name} | Epoch {epoch + 1:5d}/{DECODER_EPOCHS} | MSE={loss.item():.8f}")
    decoder.eval()
    with torch.no_grad():
        test_pred = decoder(z_test).cpu().numpy()
        test_true = y_test.cpu().numpy()
    mse = float(np.mean((test_true - test_pred) ** 2))
    r2_vx = float(r2_score(test_true[:, 0], test_pred[:, 0]))
    r2_vy = float(r2_score(test_true[:, 1], test_pred[:, 1]))
    mean_r2 = float((r2_vx + r2_vy) / 2.0)
    print("\n" + "-" * 75)
    print(f"{model_name} TEST RESULTS")
    print("-" * 75)
    print("MSE     :", f"{mse:.8f}")
    print("R2 vx   :", f"{r2_vx:.8f}")
    print("R2 vy   :", f"{r2_vy:.8f}")
    print("Mean R2 :", f"{mean_r2:.8f}")
    del z_train, y_train, z_test, y_test, decoder, optimizer
    cleanup()
    return {"mse": mse, "r2_vx": r2_vx, "r2_vy": r2_vy, "mean_r2": mean_r2}

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
        raise RuntimeError(f"Cannot orient forward Jacobian. Raw={a.shape}; latent={latent_dim}; neurons={n_neurons}")
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
        raise RuntimeError(f"Cannot orient inverse Jacobian. Raw={a.shape}; latent={latent_dim}; neurons={n_neurons}")
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
        raise RuntimeError(f"JFINV final shape={a.shape}; expected={expected}")
    return a.astype(np.float32)

def compute_attribution(model, X, model_name):
    print("\n" + "=" * 100)
    print(f"ATTRIBUTION — {model_name}")
    print("=" * 100)
    net = model.solver_.model
    attr_device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(attr_device)
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
    print("Attribution input:", X.shape)
    print("Chunks:", ATTR_N_CHUNKS)
    print("Chunk length:", ATTR_CHUNK_LEN)
    for chunk_index, start in enumerate(starts):
        stop = start + ATTR_CHUNK_LEN
        chunk = X[start:stop].astype(np.float32, copy=True)
        inp = torch.from_numpy(chunk).to(attr_device)
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
            print("Attribution keys:", list(result.keys()))
            print("RAW JF:", to_numpy(jf_raw).shape)
            print(f"RAW {inverse_key}:", to_numpy(jfinv_raw).shape)
        jf_chunk = orient_forward_jacobian(jf_raw, n_neurons=n_neurons, latent_dim=LATENT_DIM)
        jfinv_chunk = orient_inverse_jacobian(jfinv_raw, n_neurons=n_neurons, latent_dim=LATENT_DIM)
        weight = len(chunk)
        jf_sum += jf_chunk * weight
        jfinv_sum += jfinv_chunk * weight
        total_weight += weight
        print(f"chunk {chunk_index + 1:02d}/{ATTR_N_CHUNKS} done")
        del method, result, jf_raw, jfinv_raw, jf_chunk, jfinv_chunk, inp, chunk
        cleanup()
    jf = (jf_sum / total_weight).astype(np.float32)
    jfinv = (jfinv_sum / total_weight).astype(np.float32)
    print("\nFINAL ATTRIBUTION")
    print("JF:", jf.shape, "= latent × neuron")
    print("JFINV:", jfinv.shape, "= neuron × latent")
    return jf, jfinv

def train_full_model(X_train, adversarial=False):
    model_name = "ACORN" if adversarial else "CEBRA CLEAN"
    print("\n")
    print("#" * 100)
    print(f"TRAINING FULL {model_name}")
    print("#" * 100)
    print("Input:", X_train.shape)
    seed_all(SEED)
    if adversarial:
        adv_epsilon = compute_adv_epsilon(X_train)
    else:
        adv_epsilon = 0.0
    model = build_cebra(adversarial=adversarial, adv_epsilon=adv_epsilon)
    model.fit(X_train.astype(np.float32, copy=False))
    jf, jfinv = compute_attribution(model, X_train, model_name)
    return model, jf, jfinv

def get_topk(jf, jfinv, unit_ids, model_name):
    n_neurons = jf.shape[1]
    assert jfinv.shape[0] == n_neurons
    k = int(np.sqrt(n_neurons))
    k = max(k, 1)
    jf_scores = np.mean(jf, axis=0)
    jfinv_scores = np.mean(jfinv, axis=1)
    top_jf = np.argsort(jf_scores)[::-1][:k]
    top_jfinv = np.argsort(jfinv_scores)[::-1][:k]
    print("\n" + "=" * 100)
    print(f"TOP-K SELECTION — {model_name}")
    print("=" * 100)
    print("N neurons:", n_neurons)
    print("K:", k)
    print("\nTop-JF indices:")
    print(top_jf.tolist())
    print("Top-JF unit IDs:")
    print(unit_ids[top_jf].tolist())
    print("\nTop-JFINV indices:")
    print(top_jfinv.tolist())
    print("Top-JFINV unit IDs:")
    print(unit_ids[top_jfinv].tolist())
    print("\nTOP JF:")
    for rank, idx in enumerate(top_jf, start=1):
        print(f"{rank:2d}. index={idx:2d}  unit_id={unit_ids[idx]:5d}  score={jf_scores[idx]:.12f}")
    print("\nTOP JFINV:")
    for rank, idx in enumerate(top_jfinv, start=1):
        print(f"{rank:2d}. index={idx:2d}  unit_id={unit_ids[idx]:5d}  score={jfinv_scores[idx]:.12f}")
    return top_jf, top_jfinv

def save_forward_plot(clean_jf, acorn_jf):
    vmax = max(float(np.nanmax(clean_jf)), float(np.nanmax(acorn_jf)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(27, 10), constrained_layout=True)
    im = axes[0].imshow(clean_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[0].set_title("CEBRA CLEAN — FULL 65 neurons\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$", fontsize=17)
    axes[0].set_xlabel("Neuron / input column")
    axes[0].set_ylabel("Latent dimension")
    axes[1].imshow(acorn_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[1].set_title("ACORN — FULL 65 neurons\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$", fontsize=17)
    axes[1].set_xlabel("Neuron / input column")
    axes[1].set_ylabel("Latent dimension")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Mean absolute forward Jacobian")
    path = os.path.join(OUT, "JF_CLEAN_vs_ACORN.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved:", path)

def save_inverse_plot(clean_jfinv, acorn_jfinv):
    clean_plot = clean_jfinv.T
    acorn_plot = acorn_jfinv.T
    vmax = max(float(np.nanmax(clean_plot)), float(np.nanmax(acorn_plot)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(27, 10), constrained_layout=True)
    im = axes[0].imshow(clean_plot, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[0].set_title("CEBRA CLEAN — FULL 65 neurons\n" r"$\mathrm{Mean}\ |\partial x/\partial z|$", fontsize=17)
    axes[0].set_xlabel("Neuron / input column")
    axes[0].set_ylabel("Latent dimension")
    axes[1].imshow(acorn_plot, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[1].set_title("ACORN — FULL 65 neurons\n" r"$\mathrm{Mean}\ |\partial x/\partial z|$", fontsize=17)
    axes[1].set_xlabel("Neuron / input column")
    axes[1].set_ylabel("Latent dimension")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Mean absolute inverse Jacobian")
    path = os.path.join(OUT, "JFINV_CLEAN_vs_ACORN.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

def run_reduced_condition(selector_name, neuron_indices, X_train, X_test, Y_train, Y_test, adversarial=False):
    training_name = "ACORN" if adversarial else "CEBRA"
    tag = f"{selector_name}__{training_name}"
    print("\n")
    print("#" * 105)
    print(f"REDUCED CONDITION — {tag}")
    print("#" * 105)
    print("Selected neuron indices:", neuron_indices.tolist())
    X_train_reduced = X_train[:, neuron_indices].astype(np.float32)
    X_test_reduced = X_test[:, neuron_indices].astype(np.float32)
    print("Train reduced:", X_train_reduced.shape)
    print("Test reduced :", X_test_reduced.shape)
    if adversarial:
        adv_epsilon = compute_adv_epsilon(X_train_reduced)
    else:
        adv_epsilon = 0.0
    seed_all(SEED)
    model = build_cebra(adversarial=adversarial, adv_epsilon=adv_epsilon)
    model.fit(X_train_reduced)
    z_train = get_embeddings(model, X_train_reduced)
    z_test = get_embeddings(model, X_test_reduced)
    print("z_train:", z_train.shape)
    print("z_test :", z_test.shape)
    result = train_decoder(
        z_train_np=z_train,
        y_train_np=Y_train,
        z_test_np=z_test,
        y_test_np=Y_test,
        model_name=tag,
    )
    result["n_neurons"] = len(neuron_indices)
    result["selector"] = selector_name
    result["training"] = training_name
    del model, z_train, z_test, X_train_reduced, X_test_reduced
    cleanup()
    return result

def main():
    print("\n" + "=" * 110)
    print("AREA2 BUMP — TOP-K JACOBIAN EXPERIMENT")
    print("=" * 110)
    print("Full neurons: 65 expected")
    print("Target: hand velocity [vx, vy]")
    print("Split: first 80% train / last 20% test")
    print("CEBRA: time-contrastive / label-free")
    print("Decoder:" + f" TwoLayerMLP / {DECODER_EPOCHS} epochs")
    print("No normalization.")
    print("Only full-model JF/JFINV PNGs will be saved.")
    X_counts, Y, unit_ids = load_area2()
    X = smooth_neural(X_counts)
    del X_counts
    cleanup()
    print("\n" + "=" * 100)
    print("FULL DATA")
    print("=" * 100)
    print("X:", X.shape)
    print("Y:", Y.shape)
    print("Unit IDs:", unit_ids.shape)
    split_idx = int(TRAIN_FRAC * len(X))
    X_train = X[:split_idx].astype(np.float32)
    X_test = X[split_idx:].astype(np.float32)
    Y_train = Y[:split_idx].astype(np.float32)
    Y_test = Y[split_idx:].astype(np.float32)
    print("\n" + "=" * 100)
    print("TEMPORAL SPLIT")
    print("=" * 100)
    print("Split index:", split_idx)
    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("X_test :", X_test.shape)
    print("Y_test :", Y_test.shape)
    clean_model, clean_jf, clean_jfinv = train_full_model(X_train, adversarial=False)
    clean_top_jf, clean_top_jfinv = get_topk(clean_jf, clean_jfinv, unit_ids, "CEBRA CLEAN")
    acorn_model, acorn_jf, acorn_jfinv = train_full_model(X_train, adversarial=True)
    acorn_top_jf, acorn_top_jfinv = get_topk(acorn_jf, acorn_jfinv, unit_ids, "ACORN")
    save_forward_plot(clean_jf, acorn_jf)
    save_inverse_plot(clean_jfinv, acorn_jfinv)
    del clean_model, acorn_model
    cleanup()
    reduced_sets = {
        "CLEAN_topJF": clean_top_jf,
        "ACORN_topJF": acorn_top_jf,
        "CLEAN_topJFINV": clean_top_jfinv,
        "ACORN_topJFINV": acorn_top_jfinv,
    }
    print("\n")
    print("=" * 110)
    print("FOUR REDUCED NEURON SETS")
    print("=" * 110)
    for selector_name, idxs in reduced_sets.items():
        print(f"\n{selector_name}")
        print("indices :", idxs.tolist())
        print("unit IDs:", unit_ids[idxs].tolist())
    reduced_results = {}
    condition_number = 0
    for selector_name, neuron_indices in reduced_sets.items():
        for adversarial in (False, True):
            condition_number += 1
            training_name = "ACORN" if adversarial else "CEBRA"
            tag = f"{selector_name}__{training_name}"
            print("\n")
            print("=" * 110)
            print(f"CONDITION {condition_number}/8")
            print(tag)
            print("=" * 110)
            result = run_reduced_condition(
                selector_name=selector_name,
                neuron_indices=neuron_indices,
                X_train=X_train,
                X_test=X_test,
                Y_train=Y_train,
                Y_test=Y_test,
                adversarial=adversarial,
            )
            reduced_results[tag] = result
    print("\n")
    print("#" * 115)
    print("FINAL RESULTS — 8 REDUCED CONDITIONS")
    print("#" * 115)
    print(f"{'CONDITION':40s} {'N':>4s} {'MSE':>12s} {'R2 vx':>12s} {'R2 vy':>12s} {'Mean R2':>12s}")
    print("-" * 100)
    for tag, result in reduced_results.items():
        print(f"{tag:40s} {result['n_neurons']:4d} {result['mse']:12.6f} {result['r2_vx']:12.6f} {result['r2_vy']:12.6f} {result['mean_r2']:12.6f}")
    best_tag = max(reduced_results, key=lambda name: reduced_results[name]["mean_r2"])
    best = reduced_results[best_tag]
    print("\n" + "=" * 100)
    print("BEST REDUCED CONDITION")
    print("=" * 100)
    print("Condition:", best_tag)
    print("MSE:", best["mse"])
    print("R2 vx:", best["r2_vx"])
    print("R2 vy:", best["r2_vy"])
    print("Mean R2:", best["mean_r2"])
    print("\n" + "=" * 100)
    print("SAVED FILES")
    print("=" * 100)
    print(os.path.join(OUT, "JF_CLEAN_vs_ACORN.png"))
    print(os.path.join(OUT, "JFINV_CLEAN_vs_ACORN.png"))
    print("\nNo reduced Jacobians saved.")
    print("No models saved.")
    print("No CSV saved.")
    print("No decoder files saved.")

if __name__ == "__main__":
    main()



### With TwoLayerMlp 
# import os
# import sys
# import gc
# import math
# import numbers
# import random
# import h5py
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.metrics import r2_score
# from utils.constants import CEBRA_DIR
# from utils.min_distance import min_l2_distance

# for module_name in list(sys.modules):
#     if module_name == "cebra" or module_name.startswith("cebra."):
#         del sys.modules[module_name]
# sys.path.insert(0, str(CEBRA_DIR))
# import cebra
# from cebra import CEBRA
# print("\nUsing CEBRA from:")
# print(cebra.__file__)

# NWB_PATH = "data/Area2_Bump/sub-Han_desc-train_behavior+ecephys.nwb"
# BIN_MS = 50.0
# BIN_SEC = BIN_MS / 1000.0
# SMOOTH_SD_MS = 100.0
# SMOOTH_SIGMA_BINS = SMOOTH_SD_MS / BIN_MS
# SMOOTH_KERNEL_SIZE = 17
# SEED = 42
# LATENT_DIM = 128
# NUM_HIDDEN_UNITS = 128
# BATCH_SIZE = 512
# MAX_ITER = 3000
# TEMPERATURE = 0.4
# TIME_OFFSETS = 4
# MODEL_ARCH = "offset36-model-more-dropout"
# DEVICE = "cuda_if_available"
# ADV_STEPS = 10
# ATTACK_NORM = "linf"
# DECODER_HIDDEN = 64
# DECODER_DROPOUT = 0.4
# DECODER_EPOCHS = 2500
# DECODER_LR = 1e-3
# DECODER_WEIGHT_DECAY = 2e-4
# TRAIN_FRAC = 0.80

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def seed_all(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
# seed_all(SEED)

# class GaussianSmoothing(nn.Module):
#     def __init__(self, channels, kernel_size, sigma, dim=1):
#         super().__init__()
#         if isinstance(kernel_size, numbers.Number):
#             kernel_size = [kernel_size] * dim
#         if isinstance(sigma, numbers.Number):
#             sigma = [sigma] * dim
#         kernel = 1.0
#         meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing="ij")
#         for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
#             mean = (size - 1) / 2.0
#             kernel *= (1.0 / (std * math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * ((mgrid - mean) / std) ** 2))
#         kernel = kernel / torch.sum(kernel)
#         kernel = kernel.view(1, 1, *kernel.size())
#         kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
#         self.register_buffer("weight", kernel)
#         self.groups = channels
#         self.conv = F.conv1d
#     def forward(self, x):
#         x = torch.permute(x, (0, 2, 1))
#         x = self.conv(x, weight=self.weight, groups=self.groups, padding="same")
#         x = torch.permute(x, (0, 2, 1))
#         return x

# def load_area2():
#     print("\n" + "=" * 90)
#     print("LOADING AREA2 BUMP")
#     print("=" * 90)
#     with h5py.File(NWB_PATH, "r") as f:
#         vel_group = f["processing/behavior/hand_vel"]
#         vel_ds = vel_group["data"]
#         n_samples = vel_ds.shape[0]
#         starting_time = float(vel_group["starting_time"][()])
#         behavior_rate = float(vel_group["starting_time"].attrs["rate"])
#         print("Behavior rate:", behavior_rate, "Hz")
#         print("Raw hand_vel shape:", vel_ds.shape)
#         samples_per_bin = int(round(behavior_rate * BIN_SEC))
#         print("Samples / 50-ms bin:", samples_per_bin)
#         n_bins = n_samples // samples_per_bin
#         usable_samples = n_bins * samples_per_bin
#         t_start = starting_time
#         t_stop = t_start + n_bins * BIN_SEC
#         print("Complete 50-ms bins:", n_bins)
#         print("t_start:", t_start)
#         print("t_stop :", t_stop)
#         print("Duration:", (t_stop - t_start) / 60, "min")
#         vel_raw = np.asarray(vel_ds[:usable_samples], dtype=np.float32)
#         if vel_raw.shape[1] != 2:
#             raise RuntimeError(f"Expected hand_vel with 2 dims, got {vel_raw.shape}")
#         Y = vel_raw.reshape(n_bins, samples_per_bin, 2).mean(axis=1).astype(np.float32)
#         print("\nY hand velocity:", Y.shape)
#         print("vx range:", float(Y[:, 0].min()), float(Y[:, 0].max()))
#         print("vy range:", float(Y[:, 1].min()), float(Y[:, 1].max()))
#         units = f["units"]
#         unit_ids = np.asarray(units["id"][:], dtype=np.int64)
#         n_units = len(unit_ids)
#         spike_times = units["spike_times"]
#         spike_index = np.asarray(units["spike_times_index"][:], dtype=np.int64)
#         print("\nUnits:", n_units)
#         edges = t_start + np.arange(n_bins + 1, dtype=np.float64) * BIN_SEC
#         X_counts = np.zeros((n_bins, n_units), dtype=np.float32)
#         print("\nBinning spikes...")
#         for neuron_idx in range(n_units):
#             end_idx = int(spike_index[neuron_idx])
#             start_idx = 0 if neuron_idx == 0 else int(spike_index[neuron_idx - 1])
#             spikes = np.asarray(spike_times[start_idx:end_idx], dtype=np.float64)
#             left = np.searchsorted(spikes, t_start, side="left")
#             right = np.searchsorted(spikes, t_stop, side="left")
#             spikes = spikes[left:right]
#             counts, _ = np.histogram(spikes, bins=edges)
#             X_counts[:, neuron_idx] = counts.astype(np.float32)
#     print("\nRAW X:")
#     print("shape:", X_counts.shape)
#     print("min :", float(X_counts.min()))
#     print("max :", float(X_counts.max()))
#     print("mean:", float(X_counts.mean()))
#     assert X_counts.shape[0] == Y.shape[0]
#     return X_counts, Y, unit_ids

# def smooth_neural(X):
#     print("\n" + "=" * 90)
#     print("GAUSSIAN SMOOTHING")
#     print("=" * 90)
#     print("Gaussian SD:", SMOOTH_SD_MS, "ms")
#     print("Sigma:", SMOOTH_SIGMA_BINS, "bins")
#     print("Kernel:", SMOOTH_KERNEL_SIZE)
#     n_neurons = X.shape[1]
#     x_tensor = torch.from_numpy(X).float().unsqueeze(0)
#     smoother = GaussianSmoothing(channels=n_neurons, kernel_size=SMOOTH_KERNEL_SIZE, sigma=SMOOTH_SIGMA_BINS, dim=1)
#     smoother.eval()
#     with torch.no_grad():
#         X_smooth = smoother(x_tensor).squeeze(0).cpu().numpy().astype(np.float32)
#     print("Smoothed X:", X_smooth.shape)
#     print("min :", float(X_smooth.min()))
#     print("max :", float(X_smooth.max()))
#     print("mean:", float(X_smooth.mean()))
#     print("\nNO Z-SCORE")
#     print("NO NORMALIZATION")
#     return X_smooth

# def compute_adv_epsilon(train_x_np):
#     print("\n" + "=" * 90)
#     print("ACORN EPSILON")
#     print("=" * 90)
#     train_tensor = torch.from_numpy(train_x_np).float()
#     adv_epsilon = float(min_l2_distance(train_tensor)) / 2.0
#     adv_epsilon = max(adv_epsilon, 1e-6)
#     print("epsilon:", adv_epsilon)
#     print("alpha:", adv_epsilon / 5.0)
#     return adv_epsilon

# def build_cebra(adversarial, adv_epsilon):
#     return CEBRA(
#         batch_size=BATCH_SIZE,
#         temperature=TEMPERATURE,
#         model_architecture=MODEL_ARCH,
#         time_offsets=TIME_OFFSETS,
#         max_iterations=MAX_ITER,
#         output_dimension=LATENT_DIM,
#         num_hidden_units=NUM_HIDDEN_UNITS,
#         training_mode="adversarial" if adversarial else "clean",
#         adv_alpha=adv_epsilon / 5.0 if adversarial else 0.0,
#         adv_epsilon=adv_epsilon if adversarial else 0.0,
#         adv_steps=ADV_STEPS if adversarial else 0,
#         attack_norm=ATTACK_NORM,
#         device=DEVICE,
#         verbose=True,
#     )

# class TwoLayerMLP(nn.Module):
#     def __init__(self, input_dim=128, hidden_dim=64, output_dim=2, dropout_rate=0.4):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim, output_dim),
#         )
#         self._initialize_weights()
#     def _initialize_weights(self):
#         for layer in self.net:
#             if isinstance(layer, nn.Linear):
#                 nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
#                 if layer.bias is not None:
#                     nn.init.constant_(layer.bias, 0)
#     def forward(self, x):
#         return self.net(x)

# def get_embeddings(cebra_model, X):
#     x_t = torch.from_numpy(X).float()
#     z = cebra_model.transform(x_t)
#     if isinstance(z, torch.Tensor):
#         z = z.detach().cpu().numpy()
#     return np.asarray(z, dtype=np.float32)

# def train_decoder(z_train_np, y_train_np, z_test_np, y_test_np, model_name):
#     print("\n" + "=" * 90)
#     print(f"TRAINING DECODER — {model_name}")
#     print("=" * 90)
#     print("Train embedding:", z_train_np.shape)
#     print("Train target:", y_train_np.shape)
#     print("Test embedding:", z_test_np.shape)
#     print("Test target:", y_test_np.shape)
#     seed_all(SEED)
#     decoder = TwoLayerMLP(
#         input_dim=z_train_np.shape[1],
#         hidden_dim=DECODER_HIDDEN,
#         output_dim=2,
#         dropout_rate=DECODER_DROPOUT,
#     ).to(device)
#     print("\nChecking finite values...")
#     print("NaN z_train:", int(np.isnan(z_train_np).sum()))
#     print("NaN y_train:", int(np.isnan(y_train_np).sum()))
#     print("NaN z_test :", int(np.isnan(z_test_np).sum()))
#     print("NaN y_test :", int(np.isnan(y_test_np).sum()))
#     train_mask = np.isfinite(z_train_np).all(axis=1) & np.isfinite(y_train_np).all(axis=1)
#     test_mask = np.isfinite(z_test_np).all(axis=1) & np.isfinite(y_test_np).all(axis=1)
#     print("Decoder train bins before:", len(z_train_np))
#     print("Decoder train bins valid :", int(train_mask.sum()))
#     print("Decoder test bins before :", len(z_test_np))
#     print("Decoder test bins valid  :", int(test_mask.sum()))
#     z_train_np = z_train_np[train_mask]
#     y_train_np = y_train_np[train_mask]
#     z_test_np = z_test_np[test_mask]
#     y_test_np = y_test_np[test_mask]
#     z_train = torch.from_numpy(z_train_np).float().to(device)
#     y_train = torch.from_numpy(y_train_np).float().to(device)
#     z_test = torch.from_numpy(z_test_np).float().to(device)
#     y_test = torch.from_numpy(y_test_np).float().to(device)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR, weight_decay=DECODER_WEIGHT_DECAY)
#     for epoch in range(DECODER_EPOCHS):
#         decoder.train()
#         optimizer.zero_grad()
#         pred = decoder(z_train)
#         loss = criterion(pred, y_train)
#         loss.backward()
#         optimizer.step()
#         if epoch == 0 or (epoch + 1) % 1000 == 0:
#             print(f"{model_name} | Epoch {epoch + 1:5d}/{DECODER_EPOCHS} | MSE={loss.item():.8f}")
#     decoder.eval()
#     with torch.no_grad():
#         test_pred = decoder(z_test).cpu().numpy()
#         test_true = y_test.cpu().numpy()
#     mse = float(np.mean((test_true - test_pred) ** 2))
#     r2_vx = float(r2_score(test_true[:, 0], test_pred[:, 0]))
#     r2_vy = float(r2_score(test_true[:, 1], test_pred[:, 1]))
#     mean_r2 = (r2_vx + r2_vy) / 2.0
#     print("\n" + "-" * 70)
#     print(f"{model_name} TEST RESULTS")
#     print("-" * 70)
#     print("MSE     :", f"{mse:.8f}")
#     print("R2 vx   :", f"{r2_vx:.8f}")
#     print("R2 vy   :", f"{r2_vy:.8f}")
#     print("Mean R2 :", f"{mean_r2:.8f}")
#     del z_train, y_train, z_test, y_test, optimizer
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     return {"mse": mse, "r2_vx": r2_vx, "r2_vy": r2_vy, "mean_r2": mean_r2}

# def run_one(X_train, X_test, Y_train, Y_test, adversarial, adv_epsilon):
#     name = "ACORN" if adversarial else "CEBRA CLEAN"
#     print("\n")
#     print("#" * 90)
#     print(f"TRAINING {name}")
#     print("#" * 90)
#     seed_all(SEED)
#     model = build_cebra(adversarial=adversarial, adv_epsilon=adv_epsilon)
#     model.fit(X_train.astype(np.float32, copy=False))
#     print(f"\n{name} training done.")
#     z_train = get_embeddings(model, X_train)
#     z_test = get_embeddings(model, X_test)
#     print("z_train:", z_train.shape)
#     print("z_test :", z_test.shape)
#     result = train_decoder(z_train_np=z_train, y_train_np=Y_train, z_test_np=z_test, y_test_np=Y_test, model_name=name)
#     del z_train, z_test, model
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     return result

# def main():
#     print("\n" + "=" * 90)
#     print("AREA2 BUMP — HAND VELOCITY DECODER")
#     print("=" * 90)
#     print("Target = [vx, vy]")
#     print("Split = first 80% train / last 20% test")
#     print("No shuffle")
#     print("No validation")
#     print("No early stopping")
#     print("Decoder epochs:", DECODER_EPOCHS)
#     print("No normalization / scaling")
#     X_counts, Y, unit_ids = load_area2()
#     X = smooth_neural(X_counts)
#     del X_counts
#     gc.collect()
#     split_idx = int(TRAIN_FRAC * len(X))
#     X_train = X[:split_idx].astype(np.float32)
#     X_test = X[split_idx:].astype(np.float32)
#     Y_train = Y[:split_idx].astype(np.float32)
#     Y_test = Y[split_idx:].astype(np.float32)
#     print("\n" + "=" * 90)
#     print("TEMPORAL SPLIT")
#     print("=" * 90)
#     print("Full X:", X.shape)
#     print("Full Y:", Y.shape)
#     print("Split index:", split_idx)
#     print("X train:", X_train.shape)
#     print("Y train:", Y_train.shape)
#     print("X test :", X_test.shape)
#     print("Y test :", Y_test.shape)
#     print("\nTrain fraction:", len(X_train) / len(X))
#     print("Test fraction :", len(X_test) / len(X))
#     adv_epsilon = compute_adv_epsilon(X_train)
#     clean_result = run_one(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, adversarial=False, adv_epsilon=0.0)
#     acorn_result = run_one(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, adversarial=True, adv_epsilon=adv_epsilon)
#     print("\n")
#     print("=" * 90)
#     print("FINAL DECODER RESULTS")
#     print("=" * 90)
#     print(f"{'MODEL':15s} {'MSE':>12s} {'R2 vx':>12s} {'R2 vy':>12s} {'Mean R2':>12s}")
#     print("-" * 70)
#     for name, result in [("CEBRA CLEAN", clean_result), ("ACORN", acorn_result)]:
#         print(f"{name:15s} {result['mse']:12.6f} {result['r2_vx']:12.6f} {result['r2_vy']:12.6f} {result['mean_r2']:12.6f}")

# if __name__ == "__main__":
#     main()



### Without Decoder
# import os
# import sys
# import gc
# import math
# import numbers
# import random
# import h5py
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from utils.constants import CEBRA_DIR
# from utils.min_distance import min_l2_distance

# for module_name in list(sys.modules):
#     if module_name == "cebra" or module_name.startswith("cebra."):
#         del sys.modules[module_name]
# sys.path.insert(0, str(CEBRA_DIR))
# import cebra
# import cebra.attribution
# from cebra import CEBRA
# print("\nUsing CEBRA from:")
# print(cebra.__file__)

# NWB_PATH = "data/Area2_Bump/sub-Han_desc-train_behavior+ecephys.nwb"
# OUT = "Area2_Bump_Jacobian"
# os.makedirs(OUT, exist_ok=True)

# BIN_MS = 50.0
# BIN_SEC = BIN_MS / 1000.0
# SMOOTH_SD_MS = 100.0
# SMOOTH_SIGMA_BINS = SMOOTH_SD_MS / BIN_MS
# SMOOTH_KERNEL_SIZE = 17
# SEED = 42
# LATENT_DIM = 128
# NUM_HIDDEN_UNITS = 128
# BATCH_SIZE = 512
# MAX_ITER = 3000
# TEMPERATURE = 0.4
# TIME_OFFSETS = 4
# MODEL_ARCH = "offset36-model-more-dropout"
# DEVICE = "cuda_if_available"
# ADV_STEPS = 10
# ATTACK_NORM = "linf"
# ATTR_N_CHUNKS = 16
# ATTR_CHUNK_LEN = 128
# ATTR_BATCH_SIZE = 16

# def seed_all(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
# seed_all(SEED)

# class GaussianSmoothing(nn.Module):
#     def __init__(self, channels, kernel_size, sigma, dim=1):
#         super().__init__()
#         if isinstance(kernel_size, numbers.Number):
#             kernel_size = [kernel_size] * dim
#         if isinstance(sigma, numbers.Number):
#             sigma = [sigma] * dim
#         kernel = 1.0
#         meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing="ij")
#         for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
#             mean = (size - 1) / 2.0
#             kernel *= (1.0 / (std * math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * ((mgrid - mean) / std) ** 2))
#         kernel = kernel / torch.sum(kernel)
#         kernel = kernel.view(1, 1, *kernel.size())
#         kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
#         self.register_buffer("weight", kernel)
#         self.groups = channels
#         if dim == 1:
#             self.conv = F.conv1d
#         else:
#             raise RuntimeError("This script only uses 1D smoothing.")
#     def forward(self, x):
#         x = torch.permute(x, (0, 2, 1))
#         x = self.conv(x, weight=self.weight, groups=self.groups, padding="same")
#         x = torch.permute(x, (0, 2, 1))
#         return x

# def get_recording_interval(f):
#     g = f["processing/behavior/hand_vel"]
#     n_samples = g["data"].shape[0]
#     if "starting_time" in g:
#         st = g["starting_time"]
#         t_start = float(st[()])
#         if "rate" not in st.attrs:
#             raise RuntimeError("hand_vel starting_time has no rate.")
#         rate = float(st.attrs["rate"])
#         t_stop = t_start + n_samples / rate
#     elif "timestamps" in g:
#         timestamps = g["timestamps"]
#         t_start = float(timestamps[0])
#         t_stop = float(timestamps[-1])
#         rate = None
#     else:
#         raise RuntimeError("Could not determine recording timeline.")
#     print("\n" + "=" * 80)
#     print("RECORDING INTERVAL")
#     print("=" * 80)
#     print("t_start:", t_start)
#     print("t_stop :", t_stop)
#     print("duration:", (t_stop - t_start) / 60.0, "min")
#     if rate is not None:
#         print("reference rate:", rate, "Hz")
#     return t_start, t_stop

# def build_spike_counts():
#     print("\n" + "=" * 80)
#     print("BUILDING AREA2 SPIKE COUNTS")
#     print("=" * 80)
#     with h5py.File(NWB_PATH, "r") as f:
#         t_start, t_stop = get_recording_interval(f)
#         units = f["units"]
#         unit_ids = np.asarray(units["id"][:], dtype=np.int64)
#         heldout = np.asarray(units["heldout"][:], dtype=bool)
#         n_units = len(unit_ids)
#         print("\nUnits:", n_units)
#         print("Heldout flags:", int(heldout.sum()))
#         print("Using ALL units:", n_units)
#         edges = np.arange(t_start, t_stop + BIN_SEC, BIN_SEC, dtype=np.float64)
#         n_bins = len(edges) - 1
#         print("\nBin width:", BIN_MS, "ms")
#         print("Time bins:", n_bins)
#         X = np.zeros((n_bins, n_units), dtype=np.float32)
#         spike_times = units["spike_times"]
#         spike_index = np.asarray(units["spike_times_index"][:], dtype=np.int64)
#         print("\nBinning spikes...")
#         for neuron_idx in range(n_units):
#             end_idx = int(spike_index[neuron_idx])
#             start_idx = 0 if neuron_idx == 0 else int(spike_index[neuron_idx - 1])
#             spikes = np.asarray(spike_times[start_idx:end_idx], dtype=np.float64)
#             left = np.searchsorted(spikes, t_start, side="left")
#             right = np.searchsorted(spikes, t_stop, side="right")
#             spikes = spikes[left:right]
#             bin_idx = ((spikes - t_start) / BIN_SEC).astype(np.int64)
#             valid = (bin_idx >= 0) & (bin_idx < n_bins)
#             counts = np.bincount(bin_idx[valid], minlength=n_bins)
#             X[:, neuron_idx] = counts[:n_bins].astype(np.float32)
#         print("\nRAW X")
#         print("shape:", X.shape)
#         print("min :", float(X.min()))
#         print("max :", float(X.max()))
#         print("mean:", float(X.mean()))
#         print("\nNeuron input-column mapping:")
#         for idx, unit_id in enumerate(unit_ids):
#             print(f"{idx:2d} -> unit {unit_id}")
#     return X, unit_ids

# def smooth_spike_counts(X):
#     print("\n" + "=" * 80)
#     print("GAUSSIAN SMOOTHING")
#     print("=" * 80)
#     print("Input shape:", X.shape)
#     print("Bin width:", BIN_MS, "ms")
#     print("Gaussian SD:", SMOOTH_SD_MS, "ms")
#     print("Gaussian sigma:", SMOOTH_SIGMA_BINS, "bins")
#     print("Kernel size:", SMOOTH_KERNEL_SIZE)
#     n_neurons = X.shape[1]
#     x_tensor = torch.from_numpy(X.astype(np.float32, copy=False)).unsqueeze(0)
#     smoother = GaussianSmoothing(channels=n_neurons, kernel_size=SMOOTH_KERNEL_SIZE, sigma=SMOOTH_SIGMA_BINS, dim=1)
#     smoother.eval()
#     with torch.no_grad():
#         x_smooth = smoother(x_tensor)
#     X_smooth = x_smooth.squeeze(0).cpu().numpy().astype(np.float32)
#     print("\nSmoothed X")
#     print("shape:", X_smooth.shape)
#     print("min :", float(X_smooth.min()))
#     print("max :", float(X_smooth.max()))
#     print("mean:", float(X_smooth.mean()))
#     print("\n*** NO Z-SCORE ***")
#     print("*** NO NORMALIZATION ***")
#     return X_smooth

# def compute_adv_epsilon(X):
#     print("\n" + "=" * 80)
#     print("COMPUTING ACORN EPSILON")
#     print("=" * 80)
#     x_tensor = torch.from_numpy(X).float()
#     adv_epsilon = float(min_l2_distance(x_tensor)) / 2.0
#     adv_epsilon = max(adv_epsilon, 1e-6)
#     adv_alpha = adv_epsilon / 5.0
#     print("epsilon:", adv_epsilon)
#     print("alpha  :", adv_alpha)
#     print("steps  :", ADV_STEPS)
#     print("norm   :", ATTACK_NORM)
#     return adv_epsilon

# def build_model(adversarial=False, adv_epsilon=0.0):
#     if adversarial:
#         adv_alpha = adv_epsilon / 5.0
#     else:
#         adv_alpha = 0.0
#     return CEBRA(
#         batch_size=BATCH_SIZE,
#         temperature=TEMPERATURE,
#         model_architecture=MODEL_ARCH,
#         time_offsets=TIME_OFFSETS,
#         max_iterations=MAX_ITER,
#         output_dimension=LATENT_DIM,
#         num_hidden_units=NUM_HIDDEN_UNITS,
#         training_mode="adversarial" if adversarial else "clean",
#         adv_alpha=adv_alpha if adversarial else 0.0,
#         adv_epsilon=adv_epsilon if adversarial else 0.0,
#         adv_steps=ADV_STEPS if adversarial else 0,
#         attack_norm=ATTACK_NORM,
#         device=DEVICE,
#         verbose=True,
#     )

# def to_numpy(x):
#     if isinstance(x, torch.Tensor):
#         return x.detach().cpu().numpy()
#     return np.asarray(x)

# def orient_forward_jacobian(arr, n_neurons, latent_dim):
#     a = np.abs(to_numpy(arr))
#     a = np.squeeze(a)
#     latent_axes = [i for i, size in enumerate(a.shape) if size == latent_dim]
#     neuron_axes = [i for i, size in enumerate(a.shape) if size == n_neurons]
#     if not latent_axes or not neuron_axes:
#         raise RuntimeError(f"Cannot orient forward Jacobian. Raw shape={a.shape}; latent={latent_dim}; neurons={n_neurons}")
#     latent_axis = latent_axes[-1]
#     neuron_axis = neuron_axes[-1]
#     if latent_axis == neuron_axis:
#         raise RuntimeError(f"Ambiguous JF shape: {a.shape}")
#     a = np.moveaxis(a, (latent_axis, neuron_axis), (-2, -1))
#     if a.ndim > 2:
#         a = a.mean(axis=tuple(range(a.ndim - 2)))
#     if a.shape == (n_neurons, latent_dim):
#         a = a.T
#     expected = (latent_dim, n_neurons)
#     if a.shape != expected:
#         raise RuntimeError(f"JF final shape={a.shape}; expected={expected}")
#     return a.astype(np.float32)

# def compute_forward_jacobian(model, X, model_name):
#     print("\n" + "=" * 80)
#     print(f"FORWARD JACOBIAN: {model_name}")
#     print("=" * 80)
#     net = model.solver_.model
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     net = net.to(device)
#     net.eval()
#     if hasattr(net, "split_outputs"):
#         net.split_outputs = False
#     n_time, n_neurons = X.shape
#     max_start = n_time - ATTR_CHUNK_LEN - 1
#     if max_start <= 0:
#         raise RuntimeError("Not enough samples for attribution.")
#     starts = np.linspace(0, max_start, ATTR_N_CHUNKS, dtype=int)
#     jf_sum = np.zeros((LATENT_DIM, n_neurons), dtype=np.float64)
#     total_weight = 0
#     print("Attribution chunks:", ATTR_N_CHUNKS)
#     print("Chunk length:", ATTR_CHUNK_LEN)
#     print("Attribution batch size:", ATTR_BATCH_SIZE)
#     for chunk_index, start in enumerate(starts):
#         stop = start + ATTR_CHUNK_LEN
#         chunk = X[start:stop].astype(np.float32, copy=True)
#         inp = torch.from_numpy(chunk).to(device)
#         inp.requires_grad_(True)
#         method = cebra.attribution.init(name="jacobian-based-batched", model=net, input_data=inp, output_dimension=LATENT_DIM)
#         result = method.compute_attribution_map(batch_size=ATTR_BATCH_SIZE)
#         if "jf" not in result:
#             raise RuntimeError(f"No forward Jacobian in attribution result. Keys={list(result.keys())}")
#         jf_raw = result["jf"]
#         if chunk_index == 0:
#             print("\nAttribution keys:", list(result.keys()))
#             print("RAW JF shape:", to_numpy(jf_raw).shape)
#         jf_chunk = orient_forward_jacobian(jf_raw, n_neurons=n_neurons, latent_dim=LATENT_DIM)
#         weight = len(chunk)
#         jf_sum += jf_chunk * weight
#         total_weight += weight
#         print(f"chunk {chunk_index + 1:02d}/{ATTR_N_CHUNKS} done")
#         del method, result, jf_raw, jf_chunk, inp, chunk
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#     jf = (jf_sum / total_weight).astype(np.float32)
#     print("\nFINAL JF")
#     print("shape:", jf.shape)
#     print("meaning:", "latent × neuron")
#     print("JF = mean absolute |dz/dx|")
#     return jf

# def train_and_attribute(X, adversarial=False, adv_epsilon=0.0):
#     model_name = "ACORN" if adversarial else "CEBRA CLEAN"
#     print("\n")
#     print("#" * 90)
#     print(f"TRAINING {model_name}")
#     print("#" * 90)
#     seed_all(SEED)
#     model = build_model(adversarial=adversarial, adv_epsilon=adv_epsilon)
#     print("Input shape:", X.shape)
#     print("Latent dimension:", LATENT_DIM)
#     print("Hidden units:", NUM_HIDDEN_UNITS)
#     print("Iterations:", MAX_ITER)
#     if adversarial:
#         print("epsilon:", adv_epsilon)
#         print("alpha:", adv_epsilon / 5.0)
#         print("steps:", ADV_STEPS)
#     model.fit(X.astype(np.float32, copy=False))
#     jf = compute_forward_jacobian(model, X, model_name)
#     return jf, model

# def save_forward_plot(clean_jf, acorn_jf):
#     vmax = max(float(np.nanmax(clean_jf)), float(np.nanmax(acorn_jf)))
#     if not np.isfinite(vmax) or vmax <= 0:
#         vmax = 1.0
#     fig, axes = plt.subplots(1, 2, figsize=(27, 10), constrained_layout=True)
#     im = axes[0].imshow(clean_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
#     axes[0].set_title("CEBRA CLEAN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$", fontsize=17)
#     axes[0].set_xlabel("Neuron / input column", fontsize=13)
#     axes[0].set_ylabel("Latent dimension", fontsize=13)
#     axes[1].imshow(acorn_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
#     axes[1].set_title("ACORN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$", fontsize=17)
#     axes[1].set_xlabel("Neuron / input column", fontsize=13)
#     axes[1].set_ylabel("Latent dimension", fontsize=13)
#     fig.colorbar(im, ax=axes, shrink=0.85, label="Mean absolute forward Jacobian")
#     path = os.path.join(OUT, "JF_CLEAN_vs_ACORN.png")
#     fig.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     print("\nSaved:")
#     print(path)

# def print_top_neurons(jf, unit_ids, name, top_k=10):
#     scores = jf.mean(axis=0)
#     order = np.argsort(scores)[::-1]
#     print("\n" + "=" * 80)
#     print(f"TOP {top_k} NEURONS — {name}")
#     print("=" * 80)
#     for rank, idx in enumerate(order[:top_k], start=1):
#         print(f"{rank:2d}. input={idx:2d}  unit_id={unit_ids[idx]:5d}  score={scores[idx]:.12f}")

# def main():
#     print("\n" + "=" * 90)
#     print("AREA2 BUMP")
#     print("CEBRA CLEAN vs ACORN")
#     print("FORWARD JACOBIAN dz/dx ONLY")
#     print("=" * 90)
#     print("NWB:", NWB_PATH)
#     print("Bin:", BIN_MS, "ms")
#     print("Gaussian SD:", SMOOTH_SD_MS, "ms")
#     print("Latent:", LATENT_DIM)
#     print("No normalization")
#     print("No decoder")
#     X_counts, unit_ids = build_spike_counts()
#     X = smooth_spike_counts(X_counts)
#     del X_counts
#     gc.collect()
#     adv_epsilon = compute_adv_epsilon(X)
#     clean_jf, clean_model = train_and_attribute(X, adversarial=False, adv_epsilon=0.0)
#     del clean_model
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     acorn_jf, acorn_model = train_and_attribute(X, adversarial=True, adv_epsilon=adv_epsilon)
#     del acorn_model
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     print("\n" + "=" * 90)
#     print("FINAL JACOBIANS")
#     print("=" * 90)
#     print("CLEAN JF:", clean_jf.shape)
#     print("ACORN JF:", acorn_jf.shape)
#     print("Expected:", (LATENT_DIM, len(unit_ids)))
#     print_top_neurons(clean_jf, unit_ids, "CEBRA CLEAN")
#     print_top_neurons(acorn_jf, unit_ids, "ACORN")
#     save_forward_plot(clean_jf, acorn_jf)
#     print("\n" + "=" * 90)
#     print("DONE")
#     print("=" * 90)
#     print("JF definition: dz/dx")
#     print("rows    = 128 latent dimensions")
#     print("columns = 65 neurons")
#     print("No decoder was trained.")
#     print("No normalization was applied.")

# if __name__ == "__main__":
#     main()

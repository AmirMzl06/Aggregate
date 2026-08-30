import os
import sys
import gc
import math
import numbers
import random
import warnings
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from utils.constants import CEBRA_DIR
from utils.min_distance import min_l2_distance

for module_name in list(sys.modules):
    if module_name == "cebra" or module_name.startswith("cebra."):
        del sys.modules[module_name]
sys.path.insert(0, str(CEBRA_DIR))
import cebra
from cebra import CEBRA
print("\nUsing CEBRA from:")
print(cebra.__file__)

NWB_PATH = "data/Area2_Bump/sub-Han_desc-train_behavior+ecephys.nwb"
OUT = "Area2_Bump_RandomBaseline"
os.makedirs(OUT, exist_ok=True)

BIN_MS = 50.0
BIN_SEC = BIN_MS / 1000.0
SMOOTH_SD_MS = 100.0
SMOOTH_SIGMA_BINS = SMOOTH_SD_MS / BIN_MS
SMOOTH_KERNEL_SIZE = 17
TRAIN_FRAC = 0.80
SEED = 42
N_RANDOM = 30

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

DECODER_HIDDEN = 64
DECODER_DROPOUT = 0.4
DECODER_EPOCHS = 6000
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
        starting_time_ds = vel_group["starting_time"]
        t_start = float(starting_time_ds[()])
        behavior_rate = float(starting_time_ds.attrs["rate"])
        print("Behavior raw shape:", vel_ds.shape)
        print("Behavior rate:", behavior_rate, "Hz")
        samples_per_bin = int(round(behavior_rate * BIN_SEC))
        n_bins = n_behavior_samples // samples_per_bin
        usable_samples = n_bins * samples_per_bin
        t_stop = t_start + n_bins * BIN_SEC
        print("Samples per bin:", samples_per_bin)
        print("Number of bins:", n_bins)
        print("Duration:", (t_stop - t_start) / 60.0, "minutes")
        vel_raw = np.asarray(vel_ds[:usable_samples], dtype=np.float32)
        if vel_raw.ndim != 2 or vel_raw.shape[1] != 2:
            raise RuntimeError(f"Expected hand_vel=(T,2), got {vel_raw.shape}")
        vel_reshaped = vel_raw.reshape(n_bins, samples_per_bin, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            Y = np.nanmean(vel_reshaped, axis=1).astype(np.float32)
        print("\nBehavior target:")
        print("Y shape:", Y.shape)
        print("Y NaNs:", int(np.isnan(Y).sum()))
        units = f["units"]
        unit_ids = np.asarray(units["id"][:], dtype=np.int64)
        n_units = len(unit_ids)
        print("\nNumber of neurons:", n_units)
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
    print("Gaussian SD:", SMOOTH_SD_MS, "ms")
    print("Sigma:", SMOOTH_SIGMA_BINS, "bins")
    print("Kernel:", SMOOTH_KERNEL_SIZE)
    n_neurons = X.shape[1]
    x_tensor = torch.from_numpy(X.astype(np.float32, copy=False)).float().unsqueeze(0)
    smoother = GaussianSmoothing(channels=n_neurons, kernel_size=SMOOTH_KERNEL_SIZE, sigma=SMOOTH_SIGMA_BINS, dim=1)
    smoother.eval()
    with torch.no_grad():
        X_smooth = smoother(x_tensor).squeeze(0).cpu().numpy().astype(np.float32)
    print("\nSmoothed X:")
    print("shape:", X_smooth.shape)
    print("min :", float(X_smooth.min()))
    print("max :", float(X_smooth.max()))
    print("mean:", float(X_smooth.mean()))
    print("\nNO Z-SCORE")
    print("NO NORMALIZATION")
    return X_smooth

def compute_adv_epsilon(train_x_np):
    train_tensor = torch.from_numpy(train_x_np).float()
    adv_epsilon = float(min_l2_distance(train_tensor)) / 2.0
    adv_epsilon = max(adv_epsilon, 1e-6)
    print("ACORN epsilon:", adv_epsilon)
    print("ACORN alpha:", adv_epsilon / 5.0)
    return adv_epsilon

def build_cebra(adversarial, adv_epsilon):
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=TIME_OFFSETS,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=NUM_HIDDEN_UNITS,
        training_mode="adversarial" if adversarial else "clean",
        adv_alpha=adv_epsilon / 5.0 if adversarial else 0.0,
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
    return np.asarray(z, dtype=np.float32)

def train_decoder(z_train_np, y_train_np, z_test_np, y_test_np, model_name):
    print("\n" + "=" * 100)
    print(f"TRAINING DECODER — {model_name}")
    print("=" * 100)
    train_mask = np.isfinite(z_train_np).all(axis=1) & np.isfinite(y_train_np).all(axis=1)
    test_mask = np.isfinite(z_test_np).all(axis=1) & np.isfinite(y_test_np).all(axis=1)
    print("Train bins:", len(train_mask), "->", int(train_mask.sum()))
    print("Test bins :", len(test_mask), "->", int(test_mask.sum()))
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
        pred = decoder(z_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 1000 == 0:
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
    print(f"{model_name} TEST")
    print("-" * 75)
    print("MSE     :", f"{mse:.8f}")
    print("R2 vx   :", f"{r2_vx:.8f}")
    print("R2 vy   :", f"{r2_vy:.8f}")
    print("Mean R2 :", f"{mean_r2:.8f}")
    del decoder, optimizer, z_train, y_train, z_test, y_test
    cleanup()
    return {"mse": mse, "r2_vx": r2_vx, "r2_vy": r2_vy, "mean_r2": mean_r2}

def run_random_subset(random_id, neuron_indices, X_train, X_test, Y_train, Y_test, unit_ids):
    neuron_indices = np.asarray(neuron_indices, dtype=np.int64)
    selected_unit_ids = unit_ids[neuron_indices]
    print("\n")
    print("#" * 110)
    print(f"RANDOM SUBSET {random_id}/{N_RANDOM}")
    print("#" * 110)
    print("Neuron indices:", neuron_indices.tolist())
    print("Unit IDs:", selected_unit_ids.tolist())
    X_train_reduced = X_train[:, neuron_indices].astype(np.float32)
    X_test_reduced = X_test[:, neuron_indices].astype(np.float32)
    print("Reduced train:", X_train_reduced.shape)
    print("Reduced test :", X_test_reduced.shape)
    result = {"random_id": random_id, "n_neurons": len(neuron_indices), "neuron_indices": neuron_indices.tolist(), "unit_ids": selected_unit_ids.tolist()}
    for adversarial in (False, True):
        model_name = "ACORN" if adversarial else "CLEAN"
        print("\n" + "=" * 100)
        print(f"RANDOM {random_id} — {model_name}")
        print("=" * 100)
        if adversarial:
            adv_epsilon = compute_adv_epsilon(X_train_reduced)
        else:
            adv_epsilon = 0.0
        seed_all(SEED)
        model = build_cebra(adversarial=adversarial, adv_epsilon=adv_epsilon)
        model.fit(X_train_reduced.astype(np.float32, copy=False))
        z_train = get_embeddings(model, X_train_reduced)
        z_test = get_embeddings(model, X_test_reduced)
        print("z_train:", z_train.shape)
        print("z_test :", z_test.shape)
        decoder_result = train_decoder(
            z_train_np=z_train,
            y_train_np=Y_train,
            z_test_np=z_test,
            y_test_np=Y_test,
            model_name=f"Random{random_id:02d}_{model_name}",
        )
        prefix = "acorn" if adversarial else "clean"
        result[f"{prefix}_mse"] = decoder_result["mse"]
        result[f"{prefix}_r2_vx"] = decoder_result["r2_vx"]
        result[f"{prefix}_r2_vy"] = decoder_result["r2_vy"]
        result[f"{prefix}_mean_r2"] = decoder_result["mean_r2"]
        del model, z_train, z_test
        cleanup()
    del X_train_reduced, X_test_reduced
    cleanup()
    return result

def main():
    print("\n" + "=" * 110)
    print("AREA2 BUMP — RANDOM TOP-K BASELINE")
    print("=" * 110)
    print("Random subsets:", N_RANDOM)
    print("K = floor(sqrt(number of neurons))")
    print("Each subset: CLEAN + ACORN + MLP decoder")
    print("Decoder epochs:", DECODER_EPOCHS)
    print("No normalization.")
    X_counts, Y, unit_ids = load_area2()
    X = smooth_neural(X_counts)
    del X_counts
    cleanup()
    print("\nFULL DATA")
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
    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("X_test :", X_test.shape)
    print("Y_test :", Y_test.shape)
    n_neurons = X_train.shape[1]
    k = int(np.sqrt(n_neurons)) - 3
    k = max(1, min(k, n_neurons))
    print("\n" + "=" * 100)
    print("RANDOM BASELINE CONFIG")
    print("=" * 100)
    print("Total neurons:", n_neurons)
    print("K:", k)
    print("Random subsets:", N_RANDOM)
    subset_rng = np.random.default_rng(SEED)
    used_subsets = set()
    random_results = []
    random_id = 1
    while random_id <= N_RANDOM:
        neuron_indices = subset_rng.choice(n_neurons, size=k, replace=False)
        neuron_indices = np.sort(neuron_indices)
        subset_key = tuple(neuron_indices.tolist())
        if subset_key in used_subsets:
            continue
        used_subsets.add(subset_key)
        result = run_random_subset(
            random_id=random_id,
            neuron_indices=neuron_indices,
            X_train=X_train,
            X_test=X_test,
            Y_train=Y_train,
            Y_test=Y_test,
            unit_ids=unit_ids,
        )
        random_results.append(result)
        random_id += 1
    rows = []
    for result in random_results:
        rows.append({
            "random_id": result["random_id"],
            "n_neurons": result["n_neurons"],
            "neuron_indices": ",".join(map(str, result["neuron_indices"])),
            "unit_ids": ",".join(map(str, result["unit_ids"])),
            "clean_mse": result["clean_mse"],
            "clean_r2_vx": result["clean_r2_vx"],
            "clean_r2_vy": result["clean_r2_vy"],
            "clean_mean_r2": result["clean_mean_r2"],
            "acorn_mse": result["acorn_mse"],
            "acorn_r2_vx": result["acorn_r2_vx"],
            "acorn_r2_vy": result["acorn_r2_vy"],
            "acorn_mean_r2": result["acorn_mean_r2"],
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT, "Random_Subsets_30_K8_CLEAN_ACORN.csv")
    df.to_csv(csv_path, index=False, float_format="%.8f")
    print("\n")
    print("#" * 110)
    print("30 RANDOM SUBSET RESULTS")
    print("#" * 110)
    print(f"{'ID':>4s} {'N':>4s} {'CLEAN Mean R2':>16s} {'ACORN Mean R2':>16s}")
    print("-" * 50)
    for result in random_results:
        print(f"{result['random_id']:4d} {result['n_neurons']:4d} {result['clean_mean_r2']:16.6f} {result['acorn_mean_r2']:16.6f}")
    clean_values = df["clean_mean_r2"].to_numpy()
    acorn_values = df["acorn_mean_r2"].to_numpy()
    print("\n" + "=" * 100)
    print("RANDOM BASELINE SUMMARY")
    print("=" * 100)
    print("\nCLEAN:")
    print("Mean:", float(clean_values.mean()))
    print("Std :", float(clean_values.std(ddof=1)))
    print("Min :", float(clean_values.min()))
    print("Max :", float(clean_values.max()))
    print("\nACORN:")
    print("Mean:", float(acorn_values.mean()))
    print("Std :", float(acorn_values.std(ddof=1)))
    print("Min :", float(acorn_values.min()))
    print("Max :", float(acorn_values.max()))
    print("\n" + "=" * 100)
    print("CSV SAVED")
    print("=" * 100)
    print(csv_path)
    print("\nRows:", len(df))
    print("Expected rows:", N_RANDOM)

if __name__ == "__main__":
    main()

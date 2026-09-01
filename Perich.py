import os
import sys
import random
import gc
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score
from utils.constants import CEBRA_DIR
for module_name in list(sys.modules):
    if module_name == "cebra" or module_name.startswith("cebra."):
        del sys.modules[module_name]
sys.path.insert(0, str(CEBRA_DIR))
import cebra
import cebra.attribution
from cebra import CEBRA
print("\nUsing CEBRA:")
print(cebra.__file__)

DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw"
DATASET = "C-CO"
DAY = 10
NPZ_PATH = os.path.join(DATA_DIR, f"{DATASET}{DAY}.npz")
OUT = "Perich_CLEAN_ACORN_Jacobian"
os.makedirs(OUT, exist_ok=True)

SEED = 42
LATENT_DIM = 64
HIDDEN = 512
BATCH_SIZE = 1024 * 2
MAX_ITER = 3000
TEMPERATURE = 0.4
TIME_OFFSETS = 4
MODEL_ARCH = "offset36-model-more-dropout"
ADV_EPS = 0.5
ADV_STEPS = 10
DEVICE = "cuda_if_available"

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_all(SEED)

def load_data():
    print("=" * 90)
    print("LOADING PERICH")
    print("=" * 90)
    print(NPZ_PATH)
    data = np.load(NPZ_PATH, allow_pickle=True)
    X_train = data["train_data"].astype(np.float32)
    X_test = data["valid_data"].astype(np.float32)
    Y_train = data["train_label"].astype(np.float32)
    Y_test = data["valid_label"].astype(np.float32)
    print("Labels:", Y_train.shape, Y_test.shape)
    Y_train = Y_train[:, 2:4]
    Y_test = Y_test[:, 2:4]
    print("X train:", X_train.shape)
    print("X test :", X_test.shape)
    print("Y train:", Y_train.shape)
    print("Y test :", Y_test.shape)
    return X_train, X_test, Y_train, Y_test

def build_model(adversarial=False):
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=TIME_OFFSETS,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        training_mode="adversarial" if adversarial else "clean",
        adv_alpha=ADV_EPS / 5 if adversarial else 0.0,
        adv_epsilon=ADV_EPS if adversarial else 0.0,
        adv_steps=ADV_STEPS if adversarial else 0,
        attack_norm="linf",
        conditional="time_delta",
        device=DEVICE,
        verbose=True,
    )

class SimpleGRUDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

def train_decoder(model, X, Y, epochs=2000):
    model.train()
    X = torch.tensor(X, dtype=torch.float32, device="cuda")
    Y = torch.tensor(Y, dtype=torch.float32, device="cuda")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for e in tqdm(range(epochs)):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()
        if e % 200 == 0:
            print("epoch", e, "loss", float(loss.detach()))

def evaluate(model, X, Y, name):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32, device="cuda")).cpu().numpy()
    print("\n" + "=" * 80)
    print(name)
    scores = []
    for i, n in enumerate(["vx", "vy"]):
        r = r2_score(Y[:, i], pred[:, i])
        scores.append(r)
        print(n, "R2:", r)
    print("Mean R2:", np.mean(scores))

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def get_inverse(result):
    return result["jf-inv-svd"]

def orient_jacobian(arr):
    arr = np.abs(to_numpy(arr)).squeeze()
    print("Raw jacobian:", arr.shape)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    print("After time averaging:", arr.shape)
    if arr.shape == (88, 64):
        arr = arr.T
    print("Final:", arr.shape)
    return arr.astype(np.float32)

def compute_jacobian(model, X):
    print("\nComputing Jacobian")
    net = model.solver_.model
    net.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    n = min(len(X), 2048)
    x = torch.tensor(X[:n], dtype=torch.float32, device=device, requires_grad=True)
    attr = cebra.attribution.init(
        name="jacobian-based-batched",
        model=net,
        input_data=x,
        output_dimension=LATENT_DIM
    )
    result = attr.compute_attribution_map(batch_size=32)
    print("Attribution keys:")
    print(result.keys())
    jf = orient_jacobian(result["jf"])
    jinv = orient_jacobian(result["jf-inv-svd"])
    return jf, jinv

def save_plot(clean, acorn, filename, title):
    print("Plot shapes:", clean.shape, acorn.shape)
    fig, ax = plt.subplots(1, 2, figsize=(22, 8))
    vmax = max(clean.max(), acorn.max())
    im = ax[0].imshow(clean, aspect="auto", vmin=0, vmax=vmax)
    ax[0].set_title("CLEAN")
    ax[1].imshow(acorn, aspect="auto", vmin=0, vmax=vmax)
    ax[1].set_title("ACORN")
    fig.colorbar(im, ax=ax)
    fig.suptitle(title)
    path = os.path.join(OUT, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", path)

def main():
    X_train, X_test, Y_train, Y_test = load_data()
    results = {}
    for name, adv in [("CLEAN", False), ("ACORN", True)]:
        print("\n")
        print("#" * 90)
        print("TRAINING", name)
        print("#" * 90)
        model = build_model(adv)
        model.fit(X_train, Y_train)
        Z_train = model.transform(X_train)
        Z_test = model.transform(X_test)
        decoder = SimpleGRUDecoder(LATENT_DIM, 128, 2).cuda()
        train_decoder(decoder, Z_train, Y_train)
        evaluate(decoder, Z_test, Y_test, name + " TEST")
        jf, jinv = compute_jacobian(model, X_train)
        results[name] = (jf, jinv)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    save_plot(results["CLEAN"][0], results["ACORN"][0], "JF_CLEAN_vs_ACORN.png", "Forward Jacobian |dz/dx|")
    save_plot(results["CLEAN"][1], results["ACORN"][1], "JFINV_CLEAN_vs_ACORN.png", "Inverse Jacobian")
    print("\nDONE")

if __name__ == "__main__":
    main()


# ##GRU
# import os
# import sys
# import gc
# import random
# import numpy as np
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score, mean_squared_error
# from utils.constants import CEBRA_DIR
# from gru_decoder_monkey import MonkeyDecoder

# for module_name in list(sys.modules):
#     if module_name == "cebra" or module_name.startswith("cebra."):
#         del sys.modules[module_name]
# sys.path.insert(0, str(CEBRA_DIR))
# import cebra
# import cebra.attribution
# from cebra import CEBRA

# print("\nUsing CEBRA from:")
# print(cebra.__file__)

# PERICH_DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw/"
# DATASET_NAME = "C-CO"
# DAY = 0
# SESSION_NAME = f"{DATASET_NAME}{DAY}"
# NPZ_PATH = os.path.join(PERICH_DATA_DIR, f"{SESSION_NAME}.npz")
# OUT = f"Perich_{SESSION_NAME}_TopK_JF_MonkeyDecoder"
# os.makedirs(OUT, exist_ok=True)

# SEED = 42
# LATENT_DIM = 64
# NUM_HIDDEN_UNITS = 512
# BATCH_SIZE = 512
# MAX_ITER = 5000
# TEMPERATURE = 0.4
# TIME_OFFSETS = 4
# MODEL_ARCH = "offset36-model-more-dropout"
# DEVICE = "cuda_if_available"

# ADV_EPSILON = 0.5
# ADV_ALPHA = ADV_EPSILON / 5.0
# ADV_STEPS = 10
# ATTACK_NORM = "linf"

# ATTR_N_CHUNKS = 16
# ATTR_CHUNK_LEN = 128
# ATTR_BATCH_SIZE = 16

# DECODER_HIDDEN = 512
# DECODER_LAYERS = 2
# DECODER_DROPOUT = 0.4
# DECODER_BIDIRECTIONAL = False
# DECODER_STEPS = 2500
# DECODER_ADV = False

# def seed_all(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# seed_all(SEED)

# def cleanup():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

# def load_perich_session():
#     print("\n" + "=" * 100)
#     print("LOADING PERICH")
#     print("=" * 100)
#     print("Session:", SESSION_NAME)
#     print("File   :", NPZ_PATH)
#     if not os.path.exists(NPZ_PATH):
#         raise FileNotFoundError(f"Could not find:\n{NPZ_PATH}")
#     loaded = np.load(NPZ_PATH, allow_pickle=True)
#     print("\nKeys:")
#     print(loaded.files)
#     required = ["train_data", "valid_data", "train_label", "valid_label"]
#     for key in required:
#         if key not in loaded.files:
#             raise RuntimeError(f"Missing key '{key}'. Available={loaded.files}")
#     X_train = loaded["train_data"].astype(np.float32, copy=False)
#     X_test = loaded["valid_data"].astype(np.float32, copy=False)
#     labels_train = loaded["train_label"].astype(np.float32, copy=False)
#     labels_test = loaded["valid_label"].astype(np.float32, copy=False)
#     Y_train = labels_train[:, 2:4]
#     Y_test = labels_test[:, 2:4]
#     if X_train.shape[0] != Y_train.shape[0]:
#         raise RuntimeError("X_train / Y_train length mismatch.")
#     if X_test.shape[0] != Y_test.shape[0]:
#         raise RuntimeError("X_test / Y_test length mismatch.")
#     if X_train.shape[1] != X_test.shape[1]:
#         raise RuntimeError("Train/test neuron dimension mismatch.")
#     if not np.isfinite(X_train).all():
#         raise RuntimeError("X_train contains NaN/Inf.")
#     if not np.isfinite(X_test).all():
#         raise RuntimeError("X_test contains NaN/Inf.")
#     if not np.isfinite(Y_train).all():
#         raise RuntimeError("Y_train contains NaN/Inf.")
#     if not np.isfinite(Y_test).all():
#         raise RuntimeError("Y_test contains NaN/Inf.")
#     print("\nShapes")
#     print("X_train:", X_train.shape)
#     print("X_test :", X_test.shape)
#     print("Y_train:", Y_train.shape)
#     print("Y_test :", Y_test.shape)
#     print("N neurons:", X_train.shape[1])
#     print("\nX_train stats")
#     print("min :", float(X_train.min()))
#     print("max :", float(X_train.max()))
#     print("mean:", float(X_train.mean()))
#     print("std :", float(X_train.std()))
#     print("\n*** NO NORMALIZATION IN THIS SCRIPT ***")
#     print("*** NO EXTRA SMOOTHING IN THIS SCRIPT ***")
#     print("*** CEBRA TRAINING IS LABEL-FREE ***")
#     print("*** DECODER TARGET = vx, vy ***")
#     return X_train, X_test, Y_train, Y_test

# def build_model(adversarial=False):
#     return CEBRA(
#         batch_size=BATCH_SIZE,
#         temperature=TEMPERATURE,
#         model_architecture=MODEL_ARCH,
#         time_offsets=TIME_OFFSETS,
#         max_iterations=MAX_ITER,
#         output_dimension=LATENT_DIM,
#         num_hidden_units=NUM_HIDDEN_UNITS,
#         training_mode="adversarial" if adversarial else "clean",
#         adv_alpha=ADV_ALPHA if adversarial else 0.0,
#         adv_epsilon=ADV_EPSILON if adversarial else 0.0,
#         adv_steps=ADV_STEPS if adversarial else 0,
#         attack_norm=ATTACK_NORM,
#         device=DEVICE,
#         verbose=True,
#     )

# def train_representation(X_train, adversarial=False, label=""):
#     seed_all(SEED)
#     cleanup()
#     model_name = "ACORN" if adversarial else "CEBRA CLEAN"
#     print("\n")
#     print("#" * 110)
#     print(f"TRAINING {label} — {model_name}")
#     print("#" * 110)
#     print("X_train:", X_train.shape)
#     print("N neurons:", X_train.shape[1])
#     print("Latent:", LATENT_DIM)
#     print("Hidden:", NUM_HIDDEN_UNITS)
#     print("Iterations:", MAX_ITER)
#     print("Time offsets:", TIME_OFFSETS)
#     if adversarial:
#         print("epsilon:", ADV_EPSILON)
#         print("alpha:", ADV_ALPHA)
#         print("steps:", ADV_STEPS)
#         print("norm:", ATTACK_NORM)
#     model = build_model(adversarial=adversarial)
#     model.fit(X_train.astype(np.float32, copy=False))
#     return model

# def build_decoder():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     decoder = MonkeyDecoder(
#         LATENT_DIM,
#         DECODER_HIDDEN,
#         DECODER_LAYERS,
#         DECODER_DROPOUT,
#         DECODER_BIDIRECTIONAL,
#         2,
#         n_train_steps=DECODER_STEPS,
#         device=device,
#     ).to(device)
#     return decoder, device

# def train_decoder(model, X_train, X_test, Y_train, Y_test, condition_name):
#     print("\n" + "=" * 100)
#     print(f"MONKEY GRU DECODER — {condition_name}")
#     print("=" * 100)
#     Z_train = model.transform(X_train.astype(np.float32, copy=False))
#     Z_test = model.transform(X_test.astype(np.float32, copy=False))
#     Z_train = np.asarray(Z_train, dtype=np.float32)
#     Z_test = np.asarray(Z_test, dtype=np.float32)
#     print("Raw embedding train:", Z_train.shape)
#     print("Raw embedding test :", Z_test.shape)
#     train_mask = np.isfinite(Z_train).all(axis=1) & np.isfinite(Y_train).all(axis=1)
#     test_mask = np.isfinite(Z_test).all(axis=1) & np.isfinite(Y_test).all(axis=1)
#     Z_train = Z_train[train_mask]
#     y_train = Y_train[train_mask]
#     Z_test = Z_test[test_mask]
#     y_test = Y_test[test_mask]
#     print("Decoder train:", Z_train.shape)
#     print("Decoder test :", Z_test.shape)
#     if Z_train.ndim != 2:
#         raise RuntimeError(f"Expected Z_train = time x latent, got {Z_train.shape}")
#     if Z_train.shape[1] != LATENT_DIM:
#         raise RuntimeError(f"Expected latent dimension {LATENT_DIM}, got {Z_train.shape[1]}")
#     xtr = torch.from_numpy(Z_train).float()
#     ytr = torch.from_numpy(y_train).float()
#     xte = torch.from_numpy(Z_test).float()
#     yte = torch.from_numpy(y_test).float()
#     seed_all(SEED)
#     decoder, device = build_decoder()
#     print("\nDecoder config")
#     print("hidden:", DECODER_HIDDEN)
#     print("layers:", DECODER_LAYERS)
#     print("dropout:", DECODER_DROPOUT)
#     print("bidirectional:", DECODER_BIDIRECTIONAL)
#     print("steps:", DECODER_STEPS)
#     print("decoder adversarial:", DECODER_ADV)
#     print("\nTraining MonkeyDecoder...")
#     decoder.fit(xtr, ytr, seed=SEED, adv=DECODER_ADV)
#     professor_mean_r2 = decoder.score(xte, yte, device)
#     decoder.eval()
#     with torch.no_grad():
#         prediction = decoder(xte.to(device))
#         prediction = prediction.float().cpu().numpy()
#     true_values = yte.float().cpu().numpy()
#     if prediction.shape != true_values.shape:
#         raise RuntimeError(f"Decoder prediction shape mismatch: pred={prediction.shape}, true={true_values.shape}")
#     mse = mean_squared_error(true_values, prediction)
#     r2_vx = r2_score(true_values[:, 0], prediction[:, 0])
#     r2_vy = r2_score(true_values[:, 1], prediction[:, 1])
#     direct_mean_r2 = (r2_vx + r2_vy) / 2.0
#     print("\nRESULT")
#     print("MSE             :", mse)
#     print("R2 vx           :", r2_vx)
#     print("R2 vy           :", r2_vy)
#     print("Mean R2 direct  :", direct_mean_r2)
#     print("Mean R2 score() :", professor_mean_r2)
#     mean_r2 = float(professor_mean_r2)
#     del decoder, xtr, ytr, xte, yte
#     cleanup()
#     return {
#         "condition": condition_name,
#         "n_neurons": int(X_train.shape[1]),
#         "mse": float(mse),
#         "r2_vx": float(r2_vx),
#         "r2_vy": float(r2_vy),
#         "mean_r2": mean_r2,
#         "direct_mean_r2": float(direct_mean_r2),
#     }

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
#         raise RuntimeError(f"Cannot orient JF. shape={a.shape}; latent={latent_dim}; neurons={n_neurons}")
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
#     print("\n" + "=" * 100)
#     print(f"FORWARD JACOBIAN — {model_name}")
#     print("=" * 100)
#     net = model.solver_.model
#     try:
#         device = next(net.parameters()).device
#     except StopIteration:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net.eval()
#     if hasattr(net, "split_outputs"):
#         net.split_outputs = False
#     n_time = X.shape[0]
#     n_neurons = X.shape[1]
#     max_start = n_time - ATTR_CHUNK_LEN - 1
#     if max_start <= 0:
#         raise RuntimeError("Not enough samples for attribution.")
#     starts = np.linspace(0, max_start, ATTR_N_CHUNKS, dtype=int)
#     jf_sum = np.zeros((LATENT_DIM, n_neurons), dtype=np.float64)
#     total_weight = 0
#     print("Attribution chunks:", ATTR_N_CHUNKS)
#     print("Chunk length:", ATTR_CHUNK_LEN)
#     print("Attribution batch:", ATTR_BATCH_SIZE)
#     for chunk_index, start in enumerate(starts):
#         stop = start + ATTR_CHUNK_LEN
#         chunk = X[start:stop].astype(np.float32, copy=True)
#         inp = torch.from_numpy(chunk).float().to(device).detach().requires_grad_(True)
#         method = cebra.attribution.init(
#             name="jacobian-based-batched",
#             model=net,
#             input_data=inp,
#             output_dimension=LATENT_DIM
#         )
#         with torch.enable_grad():
#             result = method.compute_attribution_map(batch_size=ATTR_BATCH_SIZE)
#         if "jf" not in result:
#             raise RuntimeError(f"No forward Jacobian. Keys={list(result.keys())}")
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
#         cleanup()
#     jf = (jf_sum / total_weight).astype(np.float32)
#     print("\nFINAL JF")
#     print("shape:", jf.shape)
#     print("meaning:", "latent x neuron")
#     print("score:", "mean absolute |dz/dx|")
#     return jf

# def select_topk(jf, k, selector_name):
#     scores = jf.mean(axis=0)
#     order = np.argsort(scores)[::-1]
#     topk = order[:k]
#     print("\n" + "=" * 100)
#     print(f"TOP-{k} — {selector_name}")
#     print("=" * 100)
#     for rank, idx in enumerate(topk, start=1):
#         print(f"{rank:2d}. neuron={idx:3d} score={scores[idx]:.12f}")
#     return topk.astype(int), scores.astype(np.float32)

# def save_forward_plot(clean_jf, acorn_jf):
#     vmax = max(float(np.nanmax(clean_jf)), float(np.nanmax(acorn_jf)))
#     if not np.isfinite(vmax) or vmax <= 0:
#         vmax = 1.0
#     fig, axes = plt.subplots(1, 2, figsize=(22, 9), constrained_layout=True)
#     im = axes[0].imshow(clean_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
#     axes[0].set_title("CEBRA CLEAN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$")
#     axes[0].set_xlabel("Neuron")
#     axes[0].set_ylabel("Latent dimension")
#     axes[1].imshow(acorn_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
#     axes[1].set_title("ACORN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$")
#     axes[1].set_xlabel("Neuron")
#     axes[1].set_ylabel("Latent dimension")
#     fig.colorbar(im, ax=axes, shrink=0.85, label="Mean absolute forward Jacobian")
#     path = os.path.join(OUT, "JF_CLEAN_vs_ACORN.png")
#     fig.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     print("\nSaved:", path)

# def print_results(results):
#     print("\n")
#     print("=" * 120)
#     print("FINAL RESULTS")
#     print("=" * 120)
#     print(f"{'CONDITION':45s}{'N':>7s}{'MSE':>14s}{'R2 vx':>14s}{'R2 vy':>14s}{'Mean R2':>14s}")
#     print("-" * 120)
#     for row in results:
#         print(f"{row['condition']:45s}{row['n_neurons']:7d}{row['mse']:14.6f}{row['r2_vx']:14.6f}{row['r2_vy']:14.6f}{row['mean_r2']:14.6f}")

# def main():
#     print("\n" + "=" * 110)
#     print("PERICH — FORWARD JACOBIAN TOP-K")
#     print("MONKEY GRU DECODER")
#     print("=" * 110)
#     print("Session:", SESSION_NAME)
#     print("Latent:", LATENT_DIM)
#     print("CEBRA hidden:", NUM_HIDDEN_UNITS)
#     print("CEBRA iterations:", MAX_ITER)
#     print("ACORN epsilon:", ADV_EPSILON)
#     print("ACORN alpha:", ADV_ALPHA)
#     print("ACORN steps:", ADV_STEPS)
#     print("Attack norm:", ATTACK_NORM)
#     print("Decoder:", "MonkeyDecoder")
#     print("Decoder hidden:", DECODER_HIDDEN)
#     print("Decoder layers:", DECODER_LAYERS)
#     print("Decoder steps:", DECODER_STEPS)
#     print("No normalization")
#     print("No Jacobian regularizer")
#     print("No inverse Jacobian")

#     X_train, X_test, Y_train, Y_test = load_perich_session()
#     N = X_train.shape[1]
#     K = int(np.floor(np.sqrt(N)))
#     print("\n" + "=" * 100)
#     print("TOP-K")
#     print("=" * 100)
#     print("N =", N)
#     print("K = floor(sqrt(N)) =", K)

#     results = []

#     clean_model = train_representation(X_train, adversarial=False, label="FULL")
#     clean_result = train_decoder(clean_model, X_train, X_test, Y_train, Y_test, "FULL CLEAN")
#     results.append(clean_result)
#     clean_jf = compute_forward_jacobian(clean_model, X_train, "FULL CLEAN")
#     np.save(os.path.join(OUT, "FULL_CLEAN_JF.npy"), clean_jf)

#     acorn_model = train_representation(X_train, adversarial=True, label="FULL")
#     acorn_result = train_decoder(acorn_model, X_train, X_test, Y_train, Y_test, "FULL ACORN")
#     results.append(acorn_result)
#     acorn_jf = compute_forward_jacobian(acorn_model, X_train, "FULL ACORN")
#     np.save(os.path.join(OUT, "FULL_ACORN_JF.npy"), acorn_jf)

#     save_forward_plot(clean_jf, acorn_jf)

#     clean_topk, clean_scores = select_topk(clean_jf, K, "CLEAN Forward Jacobian")
#     acorn_topk, acorn_scores = select_topk(acorn_jf, K, "ACORN Forward Jacobian")

#     np.save(os.path.join(OUT, "CLEAN_topJF_indices.npy"), clean_topk)
#     np.save(os.path.join(OUT, "ACORN_topJF_indices.npy"), acorn_topk)
#     np.save(os.path.join(OUT, "CLEAN_JF_scores.npy"), clean_scores)
#     np.save(os.path.join(OUT, "ACORN_JF_scores.npy"), acorn_scores)

#     del clean_model, acorn_model
#     cleanup()

#     reduced_sets = {"CLEAN_topJF": clean_topk, "ACORN_topJF": acorn_topk}

#     for selector_name, selected in reduced_sets.items():
#         print("\n")
#         print("=" * 110)
#         print(f"REDUCED SET: {selector_name}")
#         print("=" * 110)
#         print("Selected neurons:", selected.tolist())
#         X_train_reduced = X_train[:, selected]
#         X_test_reduced = X_test[:, selected]
#         print("X_train reduced:", X_train_reduced.shape)
#         print("X_test reduced :", X_test_reduced.shape)

#         clean_reduced = train_representation(X_train_reduced, adversarial=False, label=selector_name)
#         result = train_decoder(clean_reduced, X_train_reduced, X_test_reduced, Y_train, Y_test, f"{selector_name}__CEBRA")
#         results.append(result)
#         del clean_reduced
#         cleanup()

#         acorn_reduced = train_representation(X_train_reduced, adversarial=True, label=selector_name)
#         result = train_decoder(acorn_reduced, X_train_reduced, X_test_reduced, Y_train, Y_test, f"{selector_name}__ACORN")
#         results.append(result)
#         del acorn_reduced
#         cleanup()

#     print_results(results)
#     df = pd.DataFrame(results)
#     csv_path = os.path.join(OUT, "Perich_TopK_JF_MonkeyDecoder_Results.csv")
#     df.to_csv(csv_path, index=False)

#     print("\n")
#     print("=" * 90)
#     print("MEAN R2 SUMMARY")
#     print("=" * 90)
#     print(f"{'CHOSEN TOP-K BY':35s}{'RETRAINED MODEL':20s}{'MEAN R2':>12s}")
#     print("-" * 70)
#     print(f"{'All neurons':35s}{'CEBRA':20s}{results[0]['mean_r2']:12.6f}")
#     print(f"{'All neurons':35s}{'ACORN':20s}{results[1]['mean_r2']:12.6f}")
#     for row in results[2:]:
#         condition = row["condition"]
#         selector, retrained = condition.split("__")
#         print(f"{selector:35s}{retrained:20s}{row['mean_r2']:12.6f}")

#     print("\nSaved CSV:")
#     print(csv_path)
#     print("\nOutput folder:")
#     print(OUT)
#     print("\nExperiment conditions:")
#     print("1. FULL CLEAN")
#     print("2. FULL ACORN")
#     print("3. CLEAN_topJF__CEBRA")
#     print("4. CLEAN_topJF__ACORN")
#     print("5. ACORN_topJF__CEBRA")
#     print("6. ACORN_topJF__ACORN")
#     print("\nNo JFINV.")
#     print("No Jacobian regularization.")
#     print("Done.")

# if __name__ == "__main__":
#     main()

# ## Decoder + topK
# # import os
# # import sys
# # import gc
# # import random
# # import numpy as np
# # import pandas as pd
# # import torch
# # import torch.nn as nn
# # import matplotlib.pyplot as plt
# # from sklearn.metrics import r2_score, mean_squared_error
# # from utils.constants import CEBRA_DIR

# # for module_name in list(sys.modules):
# #     if module_name == "cebra" or module_name.startswith("cebra."):
# #         del sys.modules[module_name]
# # sys.path.insert(0, str(CEBRA_DIR))
# # import cebra
# # import cebra.attribution
# # from cebra import CEBRA
# # print("\nUsing CEBRA from:")
# # print(cebra.__file__)

# # PERICH_DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw/"
# # DATASET_NAME = "C-CO"
# # DAY = 0
# # SESSION_NAME = f"{DATASET_NAME}{DAY}"
# # NPZ_PATH = os.path.join(PERICH_DATA_DIR, f"{SESSION_NAME}.npz")
# # OUT = f"Perich_{SESSION_NAME}_TopK_JF"
# # os.makedirs(OUT, exist_ok=True)

# # SEED = 42
# # LATENT_DIM = 64
# # NUM_HIDDEN_UNITS = 512
# # BATCH_SIZE = 512
# # MAX_ITER = 5000
# # TEMPERATURE = 0.4
# # TIME_OFFSETS = 4
# # MODEL_ARCH = "offset36-model-more-dropout"
# # DEVICE = "cuda_if_available"
# # ADV_EPSILON = 0.5
# # ADV_ALPHA = ADV_EPSILON / 5.0
# # ADV_STEPS = 10
# # ATTACK_NORM = "linf"
# # ATTR_N_CHUNKS = 16
# # ATTR_CHUNK_LEN = 128
# # ATTR_BATCH_SIZE = 16
# # DECODER_HIDDEN = 64
# # DECODER_DROPOUT = 0.4
# # DECODER_LR = 1e-3
# # DECODER_WEIGHT_DECAY = 2e-4
# # DECODER_EPOCHS = 6000

# # def seed_all(seed):
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(seed)
# # seed_all(SEED)

# # def cleanup():
# #     gc.collect()
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()

# # def load_perich_session():
# #     print("\n" + "=" * 100)
# #     print("LOADING PERICH")
# #     print("=" * 100)
# #     print("Session:", SESSION_NAME)
# #     print("File   :", NPZ_PATH)
# #     if not os.path.exists(NPZ_PATH):
# #         raise FileNotFoundError(f"Could not find:\n{NPZ_PATH}")
# #     loaded = np.load(NPZ_PATH, allow_pickle=True)
# #     print("Keys:", loaded.files)
# #     required = ["train_data", "valid_data", "train_label", "valid_label"]
# #     for key in required:
# #         if key not in loaded.files:
# #             raise RuntimeError(f"Missing key '{key}'. Available={loaded.files}")
# #     X_train = loaded["train_data"].astype(np.float32, copy=False)
# #     X_test = loaded["valid_data"].astype(np.float32, copy=False)
# #     labels_train = loaded["train_label"].astype(np.float32, copy=False)
# #     labels_test = loaded["valid_label"].astype(np.float32, copy=False)
# #     Y_train = labels_train[:, 2:4]
# #     Y_test = labels_test[:, 2:4]
# #     if X_train.shape[0] != Y_train.shape[0]:
# #         raise RuntimeError("X_train / Y_train length mismatch.")
# #     if X_test.shape[0] != Y_test.shape[0]:
# #         raise RuntimeError("X_test / Y_test length mismatch.")
# #     if X_train.shape[1] != X_test.shape[1]:
# #         raise RuntimeError("Train/test neuron dimension mismatch.")
# #     if not np.isfinite(X_train).all():
# #         raise RuntimeError("X_train contains NaN/Inf.")
# #     if not np.isfinite(X_test).all():
# #         raise RuntimeError("X_test contains NaN/Inf.")
# #     print("\nShapes")
# #     print("X_train:", X_train.shape)
# #     print("X_test :", X_test.shape)
# #     print("Y_train:", Y_train.shape)
# #     print("Y_test :", Y_test.shape)
# #     print("N neurons:", X_train.shape[1])
# #     print("\n*** NO NORMALIZATION ***")
# #     print("*** NO EXTRA SMOOTHING ***")
# #     print("*** CEBRA TRAINING IS LABEL-FREE ***")
# #     print("*** DECODER TARGET = vx, vy ***")
# #     return X_train, X_test, Y_train, Y_test

# # def build_model(adversarial=False):
# #     return CEBRA(
# #         batch_size=BATCH_SIZE,
# #         temperature=TEMPERATURE,
# #         model_architecture=MODEL_ARCH,
# #         time_offsets=TIME_OFFSETS,
# #         max_iterations=MAX_ITER,
# #         output_dimension=LATENT_DIM,
# #         num_hidden_units=NUM_HIDDEN_UNITS,
# #         training_mode="adversarial" if adversarial else "clean",
# #         adv_alpha=ADV_ALPHA if adversarial else 0.0,
# #         adv_epsilon=ADV_EPSILON if adversarial else 0.0,
# #         adv_steps=ADV_STEPS if adversarial else 0,
# #         attack_norm=ATTACK_NORM,
# #         device=DEVICE,
# #         verbose=True,
# #     )

# # def train_representation(X_train, adversarial=False, label=""):
# #     seed_all(SEED)
# #     cleanup()
# #     model_name = "ACORN" if adversarial else "CEBRA CLEAN"
# #     print("\n")
# #     print("#" * 110)
# #     print(f"TRAINING {label} — {model_name}")
# #     print("#" * 110)
# #     print("X_train:", X_train.shape)
# #     print("N neurons:", X_train.shape[1])
# #     print("Latent:", LATENT_DIM)
# #     print("Hidden:", NUM_HIDDEN_UNITS)
# #     print("Iterations:", MAX_ITER)
# #     if adversarial:
# #         print("epsilon:", ADV_EPSILON)
# #         print("alpha:", ADV_ALPHA)
# #         print("steps:", ADV_STEPS)
# #         print("norm:", ATTACK_NORM)
# #     model = build_model(adversarial=adversarial)
# #     model.fit(X_train.astype(np.float32, copy=False))
# #     return model

# # class TwoLayerMLP(nn.Module):
# #     def __init__(self, input_dim=LATENT_DIM, hidden_dim=DECODER_HIDDEN, output_dim=2, dropout_rate=DECODER_DROPOUT):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(input_dim, hidden_dim),
# #             nn.LayerNorm(hidden_dim),
# #             nn.ReLU(),
# #             nn.Dropout(dropout_rate),
# #             nn.Linear(hidden_dim, output_dim)
# #         )
# #         self._initialize()
# #     def _initialize(self):
# #         for module in self.modules():
# #             if isinstance(module, nn.Linear):
# #                 nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
# #                 if module.bias is not None:
# #                     nn.init.zeros_(module.bias)
# #     def forward(self, x):
# #         return self.net(x)

# # def train_decoder(model, X_train, X_test, Y_train, Y_test, condition_name):
# #     print("\n" + "=" * 100)
# #     print(f"DECODER — {condition_name}")
# #     print("=" * 100)
# #     Z_train = model.transform(X_train.astype(np.float32, copy=False))
# #     Z_test = model.transform(X_test.astype(np.float32, copy=False))
# #     Z_train = np.asarray(Z_train, dtype=np.float32)
# #     Z_test = np.asarray(Z_test, dtype=np.float32)
# #     train_mask = np.isfinite(Z_train).all(axis=1) & np.isfinite(Y_train).all(axis=1)
# #     test_mask = np.isfinite(Z_test).all(axis=1) & np.isfinite(Y_test).all(axis=1)
# #     Z_train = Z_train[train_mask]
# #     y_train = Y_train[train_mask]
# #     Z_test = Z_test[test_mask]
# #     y_test = Y_test[test_mask]
# #     print("Decoder train:", Z_train.shape)
# #     print("Decoder test :", Z_test.shape)
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     xtr = torch.from_numpy(Z_train).float().to(device)
# #     ytr = torch.from_numpy(y_train).float().to(device)
# #     xte = torch.from_numpy(Z_test).float().to(device)
# #     seed_all(SEED)
# #     decoder = TwoLayerMLP(
# #         input_dim=LATENT_DIM,
# #         hidden_dim=DECODER_HIDDEN,
# #         output_dim=2,
# #         dropout_rate=DECODER_DROPOUT,
# #     ).to(device)
# #     optimizer = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR, weight_decay=DECODER_WEIGHT_DECAY)
# #     criterion = nn.MSELoss()
# #     decoder.train()
# #     for epoch in range(DECODER_EPOCHS):
# #         optimizer.zero_grad()
# #         pred = decoder(xtr)
# #         loss = criterion(pred, ytr)
# #         loss.backward()
# #         optimizer.step()
# #         if (epoch + 1) % 1000 == 0 or epoch == 0:
# #             print(f"epoch {epoch + 1:5d}/{DECODER_EPOCHS} loss={loss.item():.6f}")
# #     decoder.eval()
# #     with torch.no_grad():
# #         prediction = decoder(xte).cpu().numpy()
# #     mse = mean_squared_error(y_test, prediction)
# #     r2_vx = r2_score(y_test[:, 0], prediction[:, 0])
# #     r2_vy = r2_score(y_test[:, 1], prediction[:, 1])
# #     mean_r2 = (r2_vx + r2_vy) / 2.0
# #     print("\nRESULT")
# #     print("MSE    :", mse)
# #     print("R2 vx  :", r2_vx)
# #     print("R2 vy  :", r2_vy)
# #     print("Mean R2:", mean_r2)
# #     del decoder, xtr, ytr, xte
# #     cleanup()
# #     return {"condition": condition_name, "n_neurons": X_train.shape[1], "mse": float(mse), "r2_vx": float(r2_vx), "r2_vy": float(r2_vy), "mean_r2": float(mean_r2)}

# # def to_numpy(x):
# #     if isinstance(x, torch.Tensor):
# #         return x.detach().cpu().numpy()
# #     return np.asarray(x)

# # def orient_forward_jacobian(arr, n_neurons, latent_dim):
# #     a = np.abs(to_numpy(arr))
# #     a = np.squeeze(a)
# #     latent_axes = [i for i, size in enumerate(a.shape) if size == latent_dim]
# #     neuron_axes = [i for i, size in enumerate(a.shape) if size == n_neurons]
# #     if not latent_axes or not neuron_axes:
# #         raise RuntimeError(f"Cannot orient JF. shape={a.shape}; latent={latent_dim}; neurons={n_neurons}")
# #     latent_axis = latent_axes[-1]
# #     neuron_axis = neuron_axes[-1]
# #     if latent_axis == neuron_axis:
# #         raise RuntimeError(f"Ambiguous JF shape: {a.shape}")
# #     a = np.moveaxis(a, (latent_axis, neuron_axis), (-2, -1))
# #     if a.ndim > 2:
# #         a = a.mean(axis=tuple(range(a.ndim - 2)))
# #     if a.shape == (n_neurons, latent_dim):
# #         a = a.T
# #     expected = (latent_dim, n_neurons)
# #     if a.shape != expected:
# #         raise RuntimeError(f"JF final shape={a.shape}; expected={expected}")
# #     return a.astype(np.float32)

# # def compute_forward_jacobian(model, X, model_name):
# #     print("\n" + "=" * 100)
# #     print(f"FORWARD JACOBIAN — {model_name}")
# #     print("=" * 100)
# #     net = model.solver_.model
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     net = net.to(device)
# #     net.eval()
# #     if hasattr(net, "split_outputs"):
# #         net.split_outputs = False
# #     n_time = X.shape[0]
# #     n_neurons = X.shape[1]
# #     max_start = n_time - ATTR_CHUNK_LEN - 1
# #     if max_start <= 0:
# #         raise RuntimeError("Not enough samples for attribution.")
# #     starts = np.linspace(0, max_start, ATTR_N_CHUNKS, dtype=int)
# #     jf_sum = np.zeros((LATENT_DIM, n_neurons), dtype=np.float64)
# #     total_weight = 0
# #     for chunk_index, start in enumerate(starts):
# #         stop = start + ATTR_CHUNK_LEN
# #         chunk = X[start:stop].astype(np.float32, copy=True)
# #         inp = torch.from_numpy(chunk).to(device)
# #         inp.requires_grad_(True)
# #         method = cebra.attribution.init(
# #             name="jacobian-based-batched",
# #             model=net,
# #             input_data=inp,
# #             output_dimension=LATENT_DIM
# #         )
# #         with torch.enable_grad():
# #             result = method.compute_attribution_map(batch_size=ATTR_BATCH_SIZE)
# #         if "jf" not in result:
# #             raise RuntimeError(f"No JF. Keys={list(result.keys())}")
# #         jf_raw = result["jf"]
# #         if chunk_index == 0:
# #             print("Attribution keys:", list(result.keys()))
# #             print("RAW JF:", to_numpy(jf_raw).shape)
# #         jf_chunk = orient_forward_jacobian(jf_raw, n_neurons=n_neurons, latent_dim=LATENT_DIM)
# #         weight = len(chunk)
# #         jf_sum += jf_chunk * weight
# #         total_weight += weight
# #         print(f"chunk {chunk_index + 1:02d}/{ATTR_N_CHUNKS} done")
# #         del method, result, jf_raw, jf_chunk, inp, chunk
# #         cleanup()
# #     jf = (jf_sum / total_weight).astype(np.float32)
# #     print("Final JF:", jf.shape)
# #     return jf

# # def select_topk(jf, k, selector_name):
# #     scores = jf.mean(axis=0)
# #     order = np.argsort(scores)[::-1]
# #     topk = order[:k]
# #     print("\n" + "=" * 100)
# #     print(f"TOP-{k} — {selector_name}")
# #     print("=" * 100)
# #     for rank, idx in enumerate(topk, start=1):
# #         print(f"{rank:2d}. neuron={idx:3d} score={scores[idx]:.12f}")
# #     return topk.astype(int), scores

# # def save_forward_plot(clean_jf, acorn_jf):
# #     vmax = max(float(np.nanmax(clean_jf)), float(np.nanmax(acorn_jf)))
# #     if not np.isfinite(vmax) or vmax <= 0:
# #         vmax = 1.0
# #     fig, axes = plt.subplots(1, 2, figsize=(22, 9), constrained_layout=True)
# #     im = axes[0].imshow(clean_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
# #     axes[0].set_title("CEBRA CLEAN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$")
# #     axes[0].set_xlabel("Neuron")
# #     axes[0].set_ylabel("Latent dimension")
# #     axes[1].imshow(acorn_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
# #     axes[1].set_title("ACORN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$")
# #     axes[1].set_xlabel("Neuron")
# #     axes[1].set_ylabel("Latent dimension")
# #     fig.colorbar(im, ax=axes, shrink=0.85, label="Mean absolute forward Jacobian")
# #     path = os.path.join(OUT, "JF_CLEAN_vs_ACORN.png")
# #     fig.savefig(path, dpi=300, bbox_inches="tight")
# #     plt.close(fig)
# #     print("Saved:", path)

# # def print_results(results):
# #     print("\n")
# #     print("=" * 110)
# #     print("FINAL RESULTS")
# #     print("=" * 110)
# #     print(f"{'CONDITION':45s}{'N':>7s}{'MSE':>14s}{'R2 vx':>14s}{'R2 vy':>14s}{'Mean R2':>14s}")
# #     print("-" * 110)
# #     for row in results:
# #         print(f"{row['condition']:45s}{row['n_neurons']:7d}{row['mse']:14.6f}{row['r2_vx']:14.6f}{row['r2_vy']:14.6f}{row['mean_r2']:14.6f}")

# # def main():
# #     print("\n" + "=" * 110)
# #     print("PERICH — FORWARD JACOBIAN TOP-K")
# #     print("=" * 110)
# #     print("Session:", SESSION_NAME)
# #     print("Latent:", LATENT_DIM)
# #     print("Hidden:", NUM_HIDDEN_UNITS)
# #     print("Iterations:", MAX_ITER)
# #     print("ACORN epsilon:", ADV_EPSILON)
# #     print("ACORN alpha:", ADV_ALPHA)
# #     print("Attack:", ATTACK_NORM)
# #     print("No normalization")
# #     print("No Jacobian regularizer")
# #     X_train, X_test, Y_train, Y_test = load_perich_session()
# #     N = X_train.shape[1]
# #     K = int(np.floor(np.sqrt(N)))
# #     print("\n" + "=" * 100)
# #     print("TOP-K")
# #     print("=" * 100)
# #     print("N =", N)
# #     print("K = floor(sqrt(N)) =", K)
# #     results = []
# #     clean_model = train_representation(X_train, adversarial=False, label="FULL")
# #     clean_result = train_decoder(clean_model, X_train, X_test, Y_train, Y_test, "FULL CLEAN")
# #     results.append(clean_result)
# #     clean_jf = compute_forward_jacobian(clean_model, X_train, "FULL CLEAN")
# #     np.save(os.path.join(OUT, "FULL_CLEAN_JF.npy"), clean_jf)
# #     acorn_model = train_representation(X_train, adversarial=True, label="FULL")
# #     acorn_result = train_decoder(acorn_model, X_train, X_test, Y_train, Y_test, "FULL ACORN")
# #     results.append(acorn_result)
# #     acorn_jf = compute_forward_jacobian(acorn_model, X_train, "FULL ACORN")
# #     np.save(os.path.join(OUT, "FULL_ACORN_JF.npy"), acorn_jf)
# #     save_forward_plot(clean_jf, acorn_jf)
# #     clean_topk, clean_scores = select_topk(clean_jf, K, "CLEAN Forward Jacobian")
# #     acorn_topk, acorn_scores = select_topk(acorn_jf, K, "ACORN Forward Jacobian")
# #     np.save(os.path.join(OUT, "CLEAN_topJF_indices.npy"), clean_topk)
# #     np.save(os.path.join(OUT, "ACORN_topJF_indices.npy"), acorn_topk)
# #     del clean_model, acorn_model
# #     cleanup()
# #     reduced_sets = {"CLEAN_topJF": clean_topk, "ACORN_topJF": acorn_topk}
# #     for selector_name, selected in reduced_sets.items():
# #         print("\n")
# #         print("=" * 110)
# #         print(f"REDUCED SET: {selector_name}")
# #         print("=" * 110)
# #         print("Selected neurons:", selected.tolist())
# #         X_train_reduced = X_train[:, selected]
# #         X_test_reduced = X_test[:, selected]
# #         print("X_train reduced:", X_train_reduced.shape)
# #         clean_reduced = train_representation(X_train_reduced, adversarial=False, label=selector_name)
# #         result = train_decoder(clean_reduced, X_train_reduced, X_test_reduced, Y_train, Y_test, f"{selector_name}__CEBRA")
# #         results.append(result)
# #         del clean_reduced
# #         cleanup()
# #         acorn_reduced = train_representation(X_train_reduced, adversarial=True, label=selector_name)
# #         result = train_decoder(acorn_reduced, X_train_reduced, X_test_reduced, Y_train, Y_test, f"{selector_name}__ACORN")
# #         results.append(result)
# #         del acorn_reduced
# #         cleanup()
# #     print_results(results)
# #     df = pd.DataFrame(results)
# #     csv_path = os.path.join(OUT, "Perich_TopK_JF_Results.csv")
# #     df.to_csv(csv_path, index=False)
# #     print("\n")
# #     print("=" * 90)
# #     print("MEAN R2 SUMMARY")
# #     print("=" * 90)
# #     print(f"{'CHOSEN TOP-K BY':35s}{'RETRAINED MODEL':20s}{'MEAN R2':>12s}")
# #     print("-" * 70)
# #     print(f"{'All neurons':35s}{'CEBRA':20s}{results[0]['mean_r2']:12.6f}")
# #     print(f"{'All neurons':35s}{'ACORN':20s}{results[1]['mean_r2']:12.6f}")
# #     for row in results[2:]:
# #         condition = row["condition"]
# #         if "__" in condition:
# #             selector, retrained = condition.split("__")
# #         else:
# #             selector = condition
# #             retrained = ""
# #         print(f"{selector:35s}{retrained:20s}{row['mean_r2']:12.6f}")
# #     print("\nSaved CSV:")
# #     print(csv_path)
# #     print("\nOutput:")
# #     print(OUT)
# #     print("\nDONE.")

# # if __name__ == "__main__":
# #     main()

# ## CEBRA + jacobian
# # import os
# # import sys
# # import gc
# # import random
# # import numpy as np
# # import torch
# # import matplotlib.pyplot as plt
# # from utils.constants import CEBRA_DIR
# # from utils.min_distance import min_l2_distance

# # for module_name in list(sys.modules):
# #     if module_name == "cebra" or module_name.startswith("cebra."):
# #         del sys.modules[module_name]
# # sys.path.insert(0, str(CEBRA_DIR))
# # import cebra
# # import cebra.attribution
# # from cebra import CEBRA
# # print("\nUsing CEBRA from:")
# # print(cebra.__file__)

# # PERICH_DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw/"
# # DATASET_NAME = "C-CO"
# # DAY = 0
# # NPZ_PATH = os.path.join(PERICH_DATA_DIR, f"{DATASET_NAME}{DAY}.npz")
# # OUT = f"Perich_{DATASET_NAME}{DAY}_Jacobian"
# # os.makedirs(OUT, exist_ok=True)

# # SEED = 42
# # LATENT_DIM = 64
# # NUM_HIDDEN_UNITS = 512
# # BATCH_SIZE = 512
# # MAX_ITER = 5000
# # TEMPERATURE = 0.4
# # TIME_OFFSETS = 4
# # MODEL_ARCH = "offset36-model-more-dropout"
# # DEVICE = "cuda_if_available"
# # ADV_STEPS = 10
# # ATTACK_NORM = "linf"
# # ATTR_N_CHUNKS = 16
# # ATTR_CHUNK_LEN = 128
# # ATTR_BATCH_SIZE = 16

# # def seed_all(seed):
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(seed)
# # seed_all(SEED)

# # def load_perich_session():
# #     print("\n" + "=" * 80)
# #     print("LOADING PERICH SESSION")
# #     print("=" * 80)
# #     print("Session:", f"{DATASET_NAME}{DAY}")
# #     print("File   :", NPZ_PATH)
# #     if not os.path.exists(NPZ_PATH):
# #         raise FileNotFoundError(f"\nCould not find Perich session:\n{NPZ_PATH}\n")
# #     loaded = np.load(NPZ_PATH, allow_pickle=True)
# #     print("\nKeys:")
# #     print(loaded.files)
# #     required = ["train_data", "valid_data", "train_label", "valid_label"]
# #     for key in required:
# #         if key not in loaded.files:
# #             raise RuntimeError(f"Missing '{key}' in {NPZ_PATH}. Available={loaded.files}")
# #     X_train = loaded["train_data"].astype(np.float32, copy=False)
# #     X_test = loaded["valid_data"].astype(np.float32, copy=False)
# #     Y_train = loaded["train_label"].astype(np.float32, copy=False)
# #     Y_test = loaded["valid_label"].astype(np.float32, copy=False)
# #     print("\nPERICH DATA")
# #     print("X_train:", X_train.shape)
# #     print("X_test :", X_test.shape)
# #     print("Y_train:", Y_train.shape)
# #     print("Y_test :", Y_test.shape)
# #     print("\nNumber neurons:", X_train.shape[1])
# #     print("\nX_train stats")
# #     print("min :", float(X_train.min()))
# #     print("max :", float(X_train.max()))
# #     print("mean:", float(X_train.mean()))
# #     print("std :", float(X_train.std()))
# #     if not np.isfinite(X_train).all():
# #         raise RuntimeError("X_train contains NaN or Inf.")
# #     if not np.isfinite(X_test).all():
# #         raise RuntimeError("X_test contains NaN or Inf.")
# #     if X_train.shape[1] != X_test.shape[1]:
# #         raise RuntimeError("Different neuron count between train/test.")
# #     unit_ids = np.arange(X_train.shape[1])
# #     print("\n*** USING PREPARED PERICH TRAIN/VALID SPLIT ***")
# #     print("*** NO EXTRA SMOOTHING ***")
# #     print("*** NO Z-SCORE ***")
# #     print("*** NO NORMALIZATION ***")
# #     print("*** LABELS ARE NOT USED FOR CEBRA TRAINING ***")
# #     return X_train, X_test, Y_train, Y_test, unit_ids

# # def compute_adv_epsilon(X):
# #     print("\n" + "=" * 80)
# #     print("COMPUTING ACORN EPSILON")
# #     print("=" * 80)
# #     x_tensor = torch.from_numpy(X).float()
# #     min_distance = float(min_l2_distance(x_tensor))
# #     adv_epsilon = min_distance / 2.0
# #     adv_epsilon = max(adv_epsilon, 1e-6)
# #     adv_epsilon = 0.5
# #     adv_alpha = adv_epsilon / 5.0
# #     print("min L2 distance:", min_distance)
# #     print("epsilon:", adv_epsilon)
# #     print("alpha  :", adv_alpha)
# #     print("steps  :", ADV_STEPS)
# #     print("norm   :", ATTACK_NORM)
# #     return adv_epsilon

# # def build_model(adversarial=False, adv_epsilon=0.0):
# #     if adversarial:
# #         adv_alpha = adv_epsilon / 5.0
# #     else:
# #         adv_alpha = 0.0
# #     return CEBRA(
# #         batch_size=BATCH_SIZE,
# #         temperature=TEMPERATURE,
# #         model_architecture=MODEL_ARCH,
# #         time_offsets=TIME_OFFSETS,
# #         max_iterations=MAX_ITER,
# #         output_dimension=LATENT_DIM,
# #         num_hidden_units=NUM_HIDDEN_UNITS,
# #         training_mode="adversarial" if adversarial else "clean",
# #         adv_alpha=adv_alpha if adversarial else 0.0,
# #         adv_epsilon=adv_epsilon if adversarial else 0.0,
# #         adv_steps=ADV_STEPS if adversarial else 0,
# #         attack_norm=ATTACK_NORM,
# #         device=DEVICE,
# #         verbose=True,
# #     )

# # def to_numpy(x):
# #     if isinstance(x, torch.Tensor):
# #         return x.detach().cpu().numpy()
# #     return np.asarray(x)

# # def orient_forward_jacobian(arr, n_neurons, latent_dim):
# #     a = np.abs(to_numpy(arr))
# #     a = np.squeeze(a)
# #     latent_axes = [i for i, size in enumerate(a.shape) if size == latent_dim]
# #     neuron_axes = [i for i, size in enumerate(a.shape) if size == n_neurons]
# #     if not latent_axes or not neuron_axes:
# #         raise RuntimeError(f"Cannot orient forward Jacobian. Raw shape={a.shape}; latent={latent_dim}; neurons={n_neurons}")
# #     latent_axis = latent_axes[-1]
# #     neuron_axis = neuron_axes[-1]
# #     if latent_axis == neuron_axis:
# #         raise RuntimeError(f"Ambiguous JF shape: {a.shape}")
# #     a = np.moveaxis(a, (latent_axis, neuron_axis), (-2, -1))
# #     if a.ndim > 2:
# #         a = a.mean(axis=tuple(range(a.ndim - 2)))
# #     if a.shape == (n_neurons, latent_dim):
# #         a = a.T
# #     expected = (latent_dim, n_neurons)
# #     if a.shape != expected:
# #         raise RuntimeError(f"JF final shape={a.shape}; expected={expected}")
# #     return a.astype(np.float32)

# # def orient_inverse_jacobian(arr, n_neurons, latent_dim):
# #     a = np.abs(to_numpy(arr))
# #     a = np.squeeze(a)
# #     neuron_axes = [i for i, size in enumerate(a.shape) if size == n_neurons]
# #     latent_axes = [i for i, size in enumerate(a.shape) if size == latent_dim]
# #     if not neuron_axes or not latent_axes:
# #         raise RuntimeError(f"Cannot orient inverse Jacobian. Raw shape={a.shape}; neurons={n_neurons}; latent={latent_dim}")
# #     neuron_axis = neuron_axes[-1]
# #     latent_axis = latent_axes[-1]
# #     if neuron_axis == latent_axis:
# #         raise RuntimeError(f"Ambiguous JFINV shape: {a.shape}")
# #     a = np.moveaxis(a, (neuron_axis, latent_axis), (-2, -1))
# #     if a.ndim > 2:
# #         a = a.mean(axis=tuple(range(a.ndim - 2)))
# #     if a.shape == (latent_dim, n_neurons):
# #         a = a.T
# #     expected = (n_neurons, latent_dim)
# #     if a.shape != expected:
# #         raise RuntimeError(f"JFINV final shape={a.shape}; expected={expected}")
# #     return a.astype(np.float32)

# # def get_inverse_raw(result):
# #     candidates = ["jf-inv-svd", "jf-inv", "jf-inv-lsq"]
# #     for key in candidates:
# #         if key in result:
# #             return result[key], key
# #     raise RuntimeError(f"No inverse Jacobian found. Available keys={list(result.keys())}")

# # def compute_jacobians(model, X, model_name):
# #     print("\n" + "=" * 80)
# #     print(f"JACOBIAN ATTRIBUTION: {model_name}")
# #     print("=" * 80)
# #     net = model.solver_.model
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     net = net.to(device)
# #     net.eval()
# #     if hasattr(net, "split_outputs"):
# #         net.split_outputs = False
# #     n_time = X.shape[0]
# #     n_neurons = X.shape[1]
# #     max_start = n_time - ATTR_CHUNK_LEN - 1
# #     if max_start <= 0:
# #         raise RuntimeError("Not enough samples for attribution.")
# #     starts = np.linspace(0, max_start, ATTR_N_CHUNKS, dtype=int)
# #     jf_sum = np.zeros((LATENT_DIM, n_neurons), dtype=np.float64)
# #     jfinv_sum = np.zeros((n_neurons, LATENT_DIM), dtype=np.float64)
# #     total_weight = 0
# #     print("Attribution chunks:", ATTR_N_CHUNKS)
# #     print("Chunk length:", ATTR_CHUNK_LEN)
# #     print("Attribution batch size:", ATTR_BATCH_SIZE)
# #     inverse_key_used = None
# #     for chunk_index, start in enumerate(starts):
# #         stop = start + ATTR_CHUNK_LEN
# #         chunk = X[start:stop].astype(np.float32, copy=True)
# #         inp = torch.from_numpy(chunk).to(device)
# #         inp.requires_grad_(True)
# #         method = cebra.attribution.init(
# #             name="jacobian-based-batched",
# #             model=net,
# #             input_data=inp,
# #             output_dimension=LATENT_DIM
# #         )
# #         with torch.enable_grad():
# #             result = method.compute_attribution_map(batch_size=ATTR_BATCH_SIZE)
# #         if "jf" not in result:
# #             raise RuntimeError(f"No forward Jacobian in attribution result. Keys={list(result.keys())}")
# #         jf_raw = result["jf"]
# #         jf_chunk = orient_forward_jacobian(jf_raw, n_neurons=n_neurons, latent_dim=LATENT_DIM)
# #         jfinv_raw, inverse_key = get_inverse_raw(result)
# #         inverse_key_used = inverse_key
# #         jfinv_chunk = orient_inverse_jacobian(jfinv_raw, n_neurons=n_neurons, latent_dim=LATENT_DIM)
# #         if chunk_index == 0:
# #             print("\nAttribution keys:", list(result.keys()))
# #             print("RAW JF shape:", to_numpy(jf_raw).shape)
# #             print("Inverse key:", inverse_key)
# #             print("RAW JFINV shape:", to_numpy(jfinv_raw).shape)
# #         weight = len(chunk)
# #         jf_sum += jf_chunk * weight
# #         jfinv_sum += jfinv_chunk * weight
# #         total_weight += weight
# #         print(f"chunk {chunk_index + 1:02d}/{ATTR_N_CHUNKS} done")
# #         del method, result, jf_raw, jf_chunk, jfinv_raw, jfinv_chunk, inp, chunk
# #         gc.collect()
# #         if torch.cuda.is_available():
# #             torch.cuda.empty_cache()
# #     jf = (jf_sum / total_weight).astype(np.float32)
# #     jfinv = (jfinv_sum / total_weight).astype(np.float32)
# #     print("\nFINAL JF")
# #     print("shape:", jf.shape)
# #     print("meaning:", "latent × neuron")
# #     print("JF = mean absolute |dz/dx|")
# #     print("\nFINAL JFINV")
# #     print("shape:", jfinv.shape)
# #     print("meaning:", "neuron × latent")
# #     print("inverse method:", inverse_key_used)
# #     return jf, jfinv

# # def train_and_attribute(X, adversarial=False, adv_epsilon=0.0):
# #     model_name = "ACORN" if adversarial else "CEBRA CLEAN"
# #     print("\n")
# #     print("#" * 90)
# #     print(f"TRAINING {model_name}")
# #     print("#" * 90)
# #     seed_all(SEED)
# #     model = build_model(adversarial=adversarial, adv_epsilon=adv_epsilon)
# #     print("Input shape:", X.shape)
# #     print("Latent dimension:", LATENT_DIM)
# #     print("Hidden units:", NUM_HIDDEN_UNITS)
# #     print("Iterations:", MAX_ITER)
# #     print("Time offsets:", TIME_OFFSETS)
# #     if adversarial:
# #         print("epsilon:", adv_epsilon)
# #         print("alpha:", adv_epsilon / 5.0)
# #         print("steps:", ADV_STEPS)
# #         print("norm:", ATTACK_NORM)
# #     model.fit(X.astype(np.float32, copy=False))
# #     jf, jfinv = compute_jacobians(model, X, model_name)
# #     return jf, jfinv, model

# # def save_forward_plot(clean_jf, acorn_jf):
# #     vmax = max(float(np.nanmax(clean_jf)), float(np.nanmax(acorn_jf)))
# #     if not np.isfinite(vmax) or vmax <= 0:
# #         vmax = 1.0
# #     fig, axes = plt.subplots(1, 2, figsize=(27, 10), constrained_layout=True)
# #     im = axes[0].imshow(clean_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
# #     axes[0].set_title("CEBRA CLEAN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$", fontsize=17)
# #     axes[0].set_xlabel("Neuron / input column", fontsize=13)
# #     axes[0].set_ylabel("Latent dimension", fontsize=13)
# #     axes[1].imshow(acorn_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
# #     axes[1].set_title("ACORN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$", fontsize=17)
# #     axes[1].set_xlabel("Neuron / input column", fontsize=13)
# #     axes[1].set_ylabel("Latent dimension", fontsize=13)
# #     fig.colorbar(im, ax=axes, shrink=0.85, label="Mean absolute forward Jacobian")
# #     path = os.path.join(OUT, "JF_CLEAN_vs_ACORN.png")
# #     fig.savefig(path, dpi=300, bbox_inches="tight")
# #     plt.close(fig)
# #     print("\nSaved:")
# #     print(path)

# # def save_inverse_plot(clean_inv, acorn_inv):
# #     vmax = max(float(np.nanmax(clean_inv)), float(np.nanmax(acorn_inv)))
# #     if not np.isfinite(vmax) or vmax <= 0:
# #         vmax = 1.0
# #     fig, axes = plt.subplots(1, 2, figsize=(27, 10), constrained_layout=True)
# #     im = axes[0].imshow(clean_inv, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
# #     axes[0].set_title("CEBRA CLEAN\n" "Inverse Jacobian", fontsize=17)
# #     axes[0].set_xlabel("Latent dimension", fontsize=13)
# #     axes[0].set_ylabel("Neuron / input column", fontsize=13)
# #     axes[1].imshow(acorn_inv, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
# #     axes[1].set_title("ACORN\n" "Inverse Jacobian", fontsize=17)
# #     axes[1].set_xlabel("Latent dimension", fontsize=13)
# #     axes[1].set_ylabel("Neuron / input column", fontsize=13)
# #     fig.colorbar(im, ax=axes, shrink=0.85, label="Mean absolute inverse Jacobian")
# #     path = os.path.join(OUT, "JFINV_CLEAN_vs_ACORN.png")
# #     fig.savefig(path, dpi=300, bbox_inches="tight")
# #     plt.close(fig)
# #     print("\nSaved:")
# #     print(path)

# # def print_top_forward_neurons(jf, unit_ids, name, top_k=10):
# #     scores = jf.mean(axis=0)
# #     order = np.argsort(scores)[::-1]
# #     print("\n" + "=" * 80)
# #     print(f"TOP {top_k} FORWARD NEURONS — {name}")
# #     print("=" * 80)
# #     for rank, idx in enumerate(order[:top_k], start=1):
# #         print(f"{rank:2d}. input={idx:3d}  unit_id={unit_ids[idx]:3d}  score={scores[idx]:.12f}")

# # def print_top_inverse_neurons(jfinv, unit_ids, name, top_k=10):
# #     scores = jfinv.mean(axis=1)
# #     order = np.argsort(scores)[::-1]
# #     print("\n" + "=" * 80)
# #     print(f"TOP {top_k} INVERSE NEURONS — {name}")
# #     print("=" * 80)
# #     for rank, idx in enumerate(order[:top_k], start=1):
# #         print(f"{rank:2d}. input={idx:3d}  unit_id={unit_ids[idx]:3d}  score={scores[idx]:.12f}")

# # def main():
# #     print("\n" + "=" * 90)
# #     print("PERICH SINGLE SESSION")
# #     print("CEBRA CLEAN vs ACORN")
# #     print("FORWARD + INVERSE JACOBIAN")
# #     print("=" * 90)
# #     print("Session:", f"{DATASET_NAME}{DAY}")
# #     print("NPZ:", NPZ_PATH)
# #     print("Latent:", LATENT_DIM)
# #     print("No normalization")
# #     print("No decoder")
# #     print("No Jacobian regularizer")
# #     X_train, X_test, Y_train, Y_test, unit_ids = load_perich_session()
# #     print("\nTraining only uses:")
# #     print("X_train:", X_train.shape)
# #     print("Validation data is NOT used for CEBRA/ACORN training.")
# #     adv_epsilon = compute_adv_epsilon(X_train)
# #     clean_jf, clean_inv, clean_model = train_and_attribute(X_train, adversarial=False, adv_epsilon=0.0)
# #     del clean_model
# #     gc.collect()
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()
# #     acorn_jf, acorn_inv, acorn_model = train_and_attribute(X_train, adversarial=True, adv_epsilon=adv_epsilon)
# #     del acorn_model
# #     gc.collect()
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()
# #     print("\n" + "=" * 90)
# #     print("FINAL JACOBIANS")
# #     print("=" * 90)
# #     print("CLEAN JF:", clean_jf.shape)
# #     print("ACORN JF:", acorn_jf.shape)
# #     print("CLEAN JFINV:", clean_inv.shape)
# #     print("ACORN JFINV:", acorn_inv.shape)
# #     print("Expected JF:", (LATENT_DIM, len(unit_ids)))
# #     print("Expected JFINV:", (len(unit_ids), LATENT_DIM))
# #     print_top_forward_neurons(clean_jf, unit_ids, "CEBRA CLEAN")
# #     print_top_forward_neurons(acorn_jf, unit_ids, "ACORN")
# #     print_top_inverse_neurons(clean_inv, unit_ids, "CEBRA CLEAN")
# #     print_top_inverse_neurons(acorn_inv, unit_ids, "ACORN")
# #     np.save(os.path.join(OUT, "CLEAN_JF.npy"), clean_jf)
# #     np.save(os.path.join(OUT, "ACORN_JF.npy"), acorn_jf)
# #     np.save(os.path.join(OUT, "CLEAN_JFINV.npy"), clean_inv)
# #     np.save(os.path.join(OUT, "ACORN_JFINV.npy"), acorn_inv)
# #     save_forward_plot(clean_jf, acorn_jf)
# #     save_inverse_plot(clean_inv, acorn_inv)
# #     print("\n" + "=" * 90)
# #     print("DONE")
# #     print("=" * 90)
# #     print("Session:", f"{DATASET_NAME}{DAY}")
# #     print("Number neurons:", len(unit_ids))
# #     print("epsilon:", adv_epsilon)
# #     print("alpha:", adv_epsilon / 5.0)
# #     print("attack:", ATTACK_NORM)
# #     print("steps:", ADV_STEPS)
# #     print("\nOutput folder:")
# #     print(OUT)
# #     print("\nSaved:")
# #     print(os.path.join(OUT, "JF_CLEAN_vs_ACORN.png"))
# #     print(os.path.join(OUT, "JFINV_CLEAN_vs_ACORN.png"))
# #     print("\nJF definition:")
# #     print("dz/dx")
# #     print("JF rows    = 128 latent dimensions")
# #     print("JF columns = neurons")
# #     print("JFINV rows = neurons")
# #     print("JFINV cols = 128 latent dimensions")
# #     print("\nNo decoder was trained.")
# #     print("No normalization was applied.")
# #     print("No Jacobian regularization was applied.")

# # if __name__ == "__main__":
# #     main()

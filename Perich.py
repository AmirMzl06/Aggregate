import os
import sys
import gc
import random
import numpy as np
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

PERICH_DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw/"
DATASET_NAME = "C-CO"
DAY = 0
NPZ_PATH = os.path.join(PERICH_DATA_DIR, f"{DATASET_NAME}{DAY}.npz")
OUT = f"Perich_{DATASET_NAME}{DAY}_Jacobian"
os.makedirs(OUT, exist_ok=True)

SEED = 42
LATENT_DIM = 64
NUM_HIDDEN_UNITS = 512
BATCH_SIZE = 512
MAX_ITER = 5000
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

def load_perich_session():
    print("\n" + "=" * 80)
    print("LOADING PERICH SESSION")
    print("=" * 80)
    print("Session:", f"{DATASET_NAME}{DAY}")
    print("File   :", NPZ_PATH)
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"\nCould not find Perich session:\n{NPZ_PATH}\n")
    loaded = np.load(NPZ_PATH, allow_pickle=True)
    print("\nKeys:")
    print(loaded.files)
    required = ["train_data", "valid_data", "train_label", "valid_label"]
    for key in required:
        if key not in loaded.files:
            raise RuntimeError(f"Missing '{key}' in {NPZ_PATH}. Available={loaded.files}")
    X_train = loaded["train_data"].astype(np.float32, copy=False)
    X_test = loaded["valid_data"].astype(np.float32, copy=False)
    Y_train = loaded["train_label"].astype(np.float32, copy=False)
    Y_test = loaded["valid_label"].astype(np.float32, copy=False)
    print("\nPERICH DATA")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("Y_train:", Y_train.shape)
    print("Y_test :", Y_test.shape)
    print("\nNumber neurons:", X_train.shape[1])
    print("\nX_train stats")
    print("min :", float(X_train.min()))
    print("max :", float(X_train.max()))
    print("mean:", float(X_train.mean()))
    print("std :", float(X_train.std()))
    if not np.isfinite(X_train).all():
        raise RuntimeError("X_train contains NaN or Inf.")
    if not np.isfinite(X_test).all():
        raise RuntimeError("X_test contains NaN or Inf.")
    if X_train.shape[1] != X_test.shape[1]:
        raise RuntimeError("Different neuron count between train/test.")
    unit_ids = np.arange(X_train.shape[1])
    print("\n*** USING PREPARED PERICH TRAIN/VALID SPLIT ***")
    print("*** NO EXTRA SMOOTHING ***")
    print("*** NO Z-SCORE ***")
    print("*** NO NORMALIZATION ***")
    print("*** LABELS ARE NOT USED FOR CEBRA TRAINING ***")
    return X_train, X_test, Y_train, Y_test, unit_ids

def compute_adv_epsilon(X):
    print("\n" + "=" * 80)
    print("COMPUTING ACORN EPSILON")
    print("=" * 80)
    x_tensor = torch.from_numpy(X).float()
    min_distance = float(min_l2_distance(x_tensor))
    adv_epsilon = min_distance / 2.0
    adv_epsilon = max(adv_epsilon, 1e-6)
    adv_epsilon = 0.5
    adv_alpha = adv_epsilon / 5.0
    print("min L2 distance:", min_distance)
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

def orient_inverse_jacobian(arr, n_neurons, latent_dim):
    a = np.abs(to_numpy(arr))
    a = np.squeeze(a)
    neuron_axes = [i for i, size in enumerate(a.shape) if size == n_neurons]
    latent_axes = [i for i, size in enumerate(a.shape) if size == latent_dim]
    if not neuron_axes or not latent_axes:
        raise RuntimeError(f"Cannot orient inverse Jacobian. Raw shape={a.shape}; neurons={n_neurons}; latent={latent_dim}")
    neuron_axis = neuron_axes[-1]
    latent_axis = latent_axes[-1]
    if neuron_axis == latent_axis:
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

def get_inverse_raw(result):
    candidates = ["jf-inv-svd", "jf-inv", "jf-inv-lsq"]
    for key in candidates:
        if key in result:
            return result[key], key
    raise RuntimeError(f"No inverse Jacobian found. Available keys={list(result.keys())}")

def compute_jacobians(model, X, model_name):
    print("\n" + "=" * 80)
    print(f"JACOBIAN ATTRIBUTION: {model_name}")
    print("=" * 80)
    net = model.solver_.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    net.eval()
    if hasattr(net, "split_outputs"):
        net.split_outputs = False
    n_time = X.shape[0]
    n_neurons = X.shape[1]
    max_start = n_time - ATTR_CHUNK_LEN - 1
    if max_start <= 0:
        raise RuntimeError("Not enough samples for attribution.")
    starts = np.linspace(0, max_start, ATTR_N_CHUNKS, dtype=int)
    jf_sum = np.zeros((LATENT_DIM, n_neurons), dtype=np.float64)
    jfinv_sum = np.zeros((n_neurons, LATENT_DIM), dtype=np.float64)
    total_weight = 0
    print("Attribution chunks:", ATTR_N_CHUNKS)
    print("Chunk length:", ATTR_CHUNK_LEN)
    print("Attribution batch size:", ATTR_BATCH_SIZE)
    inverse_key_used = None
    for chunk_index, start in enumerate(starts):
        stop = start + ATTR_CHUNK_LEN
        chunk = X[start:stop].astype(np.float32, copy=True)
        inp = torch.from_numpy(chunk).to(device)
        inp.requires_grad_(True)
        method = cebra.attribution.init(
            name="jacobian-based-batched",
            model=net,
            input_data=inp,
            output_dimension=LATENT_DIM
        )
        with torch.enable_grad():
            result = method.compute_attribution_map(batch_size=ATTR_BATCH_SIZE)
        if "jf" not in result:
            raise RuntimeError(f"No forward Jacobian in attribution result. Keys={list(result.keys())}")
        jf_raw = result["jf"]
        jf_chunk = orient_forward_jacobian(jf_raw, n_neurons=n_neurons, latent_dim=LATENT_DIM)
        jfinv_raw, inverse_key = get_inverse_raw(result)
        inverse_key_used = inverse_key
        jfinv_chunk = orient_inverse_jacobian(jfinv_raw, n_neurons=n_neurons, latent_dim=LATENT_DIM)
        if chunk_index == 0:
            print("\nAttribution keys:", list(result.keys()))
            print("RAW JF shape:", to_numpy(jf_raw).shape)
            print("Inverse key:", inverse_key)
            print("RAW JFINV shape:", to_numpy(jfinv_raw).shape)
        weight = len(chunk)
        jf_sum += jf_chunk * weight
        jfinv_sum += jfinv_chunk * weight
        total_weight += weight
        print(f"chunk {chunk_index + 1:02d}/{ATTR_N_CHUNKS} done")
        del method, result, jf_raw, jf_chunk, jfinv_raw, jfinv_chunk, inp, chunk
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    jf = (jf_sum / total_weight).astype(np.float32)
    jfinv = (jfinv_sum / total_weight).astype(np.float32)
    print("\nFINAL JF")
    print("shape:", jf.shape)
    print("meaning:", "latent × neuron")
    print("JF = mean absolute |dz/dx|")
    print("\nFINAL JFINV")
    print("shape:", jfinv.shape)
    print("meaning:", "neuron × latent")
    print("inverse method:", inverse_key_used)
    return jf, jfinv

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
    print("Time offsets:", TIME_OFFSETS)
    if adversarial:
        print("epsilon:", adv_epsilon)
        print("alpha:", adv_epsilon / 5.0)
        print("steps:", ADV_STEPS)
        print("norm:", ATTACK_NORM)
    model.fit(X.astype(np.float32, copy=False))
    jf, jfinv = compute_jacobians(model, X, model_name)
    return jf, jfinv, model

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

def save_inverse_plot(clean_inv, acorn_inv):
    vmax = max(float(np.nanmax(clean_inv)), float(np.nanmax(acorn_inv)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(27, 10), constrained_layout=True)
    im = axes[0].imshow(clean_inv, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[0].set_title("CEBRA CLEAN\n" "Inverse Jacobian", fontsize=17)
    axes[0].set_xlabel("Latent dimension", fontsize=13)
    axes[0].set_ylabel("Neuron / input column", fontsize=13)
    axes[1].imshow(acorn_inv, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[1].set_title("ACORN\n" "Inverse Jacobian", fontsize=17)
    axes[1].set_xlabel("Latent dimension", fontsize=13)
    axes[1].set_ylabel("Neuron / input column", fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.85, label="Mean absolute inverse Jacobian")
    path = os.path.join(OUT, "JFINV_CLEAN_vs_ACORN.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved:")
    print(path)

def print_top_forward_neurons(jf, unit_ids, name, top_k=10):
    scores = jf.mean(axis=0)
    order = np.argsort(scores)[::-1]
    print("\n" + "=" * 80)
    print(f"TOP {top_k} FORWARD NEURONS — {name}")
    print("=" * 80)
    for rank, idx in enumerate(order[:top_k], start=1):
        print(f"{rank:2d}. input={idx:3d}  unit_id={unit_ids[idx]:3d}  score={scores[idx]:.12f}")

def print_top_inverse_neurons(jfinv, unit_ids, name, top_k=10):
    scores = jfinv.mean(axis=1)
    order = np.argsort(scores)[::-1]
    print("\n" + "=" * 80)
    print(f"TOP {top_k} INVERSE NEURONS — {name}")
    print("=" * 80)
    for rank, idx in enumerate(order[:top_k], start=1):
        print(f"{rank:2d}. input={idx:3d}  unit_id={unit_ids[idx]:3d}  score={scores[idx]:.12f}")

def main():
    print("\n" + "=" * 90)
    print("PERICH SINGLE SESSION")
    print("CEBRA CLEAN vs ACORN")
    print("FORWARD + INVERSE JACOBIAN")
    print("=" * 90)
    print("Session:", f"{DATASET_NAME}{DAY}")
    print("NPZ:", NPZ_PATH)
    print("Latent:", LATENT_DIM)
    print("No normalization")
    print("No decoder")
    print("No Jacobian regularizer")
    X_train, X_test, Y_train, Y_test, unit_ids = load_perich_session()
    print("\nTraining only uses:")
    print("X_train:", X_train.shape)
    print("Validation data is NOT used for CEBRA/ACORN training.")
    adv_epsilon = compute_adv_epsilon(X_train)
    clean_jf, clean_inv, clean_model = train_and_attribute(X_train, adversarial=False, adv_epsilon=0.0)
    del clean_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    acorn_jf, acorn_inv, acorn_model = train_and_attribute(X_train, adversarial=True, adv_epsilon=adv_epsilon)
    del acorn_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n" + "=" * 90)
    print("FINAL JACOBIANS")
    print("=" * 90)
    print("CLEAN JF:", clean_jf.shape)
    print("ACORN JF:", acorn_jf.shape)
    print("CLEAN JFINV:", clean_inv.shape)
    print("ACORN JFINV:", acorn_inv.shape)
    print("Expected JF:", (LATENT_DIM, len(unit_ids)))
    print("Expected JFINV:", (len(unit_ids), LATENT_DIM))
    print_top_forward_neurons(clean_jf, unit_ids, "CEBRA CLEAN")
    print_top_forward_neurons(acorn_jf, unit_ids, "ACORN")
    print_top_inverse_neurons(clean_inv, unit_ids, "CEBRA CLEAN")
    print_top_inverse_neurons(acorn_inv, unit_ids, "ACORN")
    np.save(os.path.join(OUT, "CLEAN_JF.npy"), clean_jf)
    np.save(os.path.join(OUT, "ACORN_JF.npy"), acorn_jf)
    np.save(os.path.join(OUT, "CLEAN_JFINV.npy"), clean_inv)
    np.save(os.path.join(OUT, "ACORN_JFINV.npy"), acorn_inv)
    save_forward_plot(clean_jf, acorn_jf)
    save_inverse_plot(clean_inv, acorn_inv)
    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print("Session:", f"{DATASET_NAME}{DAY}")
    print("Number neurons:", len(unit_ids))
    print("epsilon:", adv_epsilon)
    print("alpha:", adv_epsilon / 5.0)
    print("attack:", ATTACK_NORM)
    print("steps:", ADV_STEPS)
    print("\nOutput folder:")
    print(OUT)
    print("\nSaved:")
    print(os.path.join(OUT, "JF_CLEAN_vs_ACORN.png"))
    print(os.path.join(OUT, "JFINV_CLEAN_vs_ACORN.png"))
    print("\nJF definition:")
    print("dz/dx")
    print("JF rows    = 128 latent dimensions")
    print("JF columns = neurons")
    print("JFINV rows = neurons")
    print("JFINV cols = 128 latent dimensions")
    print("\nNo decoder was trained.")
    print("No normalization was applied.")
    print("No Jacobian regularization was applied.")

if __name__ == "__main__":
    main()

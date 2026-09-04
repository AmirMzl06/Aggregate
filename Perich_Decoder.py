import os
import sys
import gc
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utils.constants import CEBRA_DIR
from utils.min_distance import min_l2_distance
from gru_decoder_monkey import MonkeyDecoder
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(CEBRA_DIR))
import cebra
from cebra import CEBRA
import cebra.attribution

print("\nUsing CEBRA:")
print(cebra.__file__)

PERICH_DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw/"
DATASET_NAME = "C-CO"
DAY = 0
SESSION = f"{DATASET_NAME}{DAY}"
NPZ_PATH = os.path.join(PERICH_DATA_DIR, f"{SESSION}.npz")
OUT = f"ACORN_CLEAN_LABEL_{SESSION}"
os.makedirs(OUT, exist_ok=True)

SEED = 42
LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 5000
TEMPERATURE = 0.4
OFFSET = 1
MODEL_ARCH = "offset36-model-more-dropout"

ADV_STEPS = 10
ATTACK_NORM = "linf"

DECODER_HIDDEN = 512
DECODER_LAYERS = 2
DECODER_DROPOUT = 0.4
DECODER_STEPS = 2500

ATTR_CHUNKS = 16
ATTR_LEN = 128
ATTR_BATCH = 16
DEVICE = "cuda_if_available"

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

seed_all(SEED)

def load_perich():
    print("\n" + "=" * 90)
    print("LOADING PERICH")
    print("=" * 90)
    print("File:", NPZ_PATH)
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(NPZ_PATH)
    data = np.load(NPZ_PATH, allow_pickle=True)
    X_train = data["train_data"].astype(np.float32)
    X_test = data["valid_data"].astype(np.float32)
    Y_train = data["train_label"].astype(np.float32)
    Y_test = data["valid_label"].astype(np.float32)
    print("\nRAW")
    print("X train:", X_train.shape)
    print("X test :", X_test.shape)
    print("Y train:", Y_train.shape)
    print("Y test :", Y_test.shape)
    if not np.isfinite(X_train).all():
        raise RuntimeError("X_train contains NaN or Inf.")
    if not np.isfinite(X_test).all():
        raise RuntimeError("X_test contains NaN or Inf.")
    if not np.isfinite(Y_train).all():
        raise RuntimeError("Y_train contains NaN or Inf.")
    if not np.isfinite(Y_test).all():
        raise RuntimeError("Y_test contains NaN or Inf.")
    return (X_train.astype(np.float32), X_test.astype(np.float32), Y_train.astype(np.float32), Y_test.astype(np.float32))

def compute_adv_epsilon(X):
    print("\n" + "=" * 90)
    print("COMPUTING ACORN EPSILON")
    print("=" * 90)
    dist = float(min_l2_distance(torch.from_numpy(X).float()))
    eps = max(dist / 2.0, 1e-6)
    eps = 0.5
    print("min L2 distance:", dist)
    print("epsilon        :", eps)
    return eps

def build_acorn(eps):
    print("\nBuilding ACORN")
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=OFFSET,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        training_mode="adversarial",
        adv_alpha=eps / 5.0,
        adv_epsilon=eps,
        adv_steps=ADV_STEPS,
        attack_norm=ATTACK_NORM,
        device=DEVICE,
        verbose=True
    )

def build_clean():
    print("\nBuilding CLEAN")
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=OFFSET,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        training_mode="clean",
        device=DEVICE,
        verbose=True
    )

def train_acorn_label(X_train, Y_train, eps):
    print("\n" + "=" * 100)
    print("TRAINING ACORN + LABEL")
    print("=" * 100)
    model = build_acorn(eps)
    model.fit(X_train, Y_train)
    return model

def train_clean_label(X_train, Y_train):
    print("\n" + "=" * 100)
    print("TRAINING CLEAN + LABEL")
    print("=" * 100)
    model = build_clean()
    model.fit(X_train, Y_train)
    return model

def train_decoder(Z, Y, tag):
    print("\n" + "=" * 90)
    print("TRAIN DECODER:", tag)
    print("=" * 90)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = MonkeyDecoder(
        LATENT_DIM,
        DECODER_HIDDEN,
        DECODER_LAYERS,
        DECODER_DROPOUT,
        False,
        Y.shape[1],
        n_train_steps=DECODER_STEPS
    ).to(device)
    print(decoder)
    Z_tensor = torch.tensor(Z, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device=device)
    decoder.fit(Z_tensor, Y_tensor)
    return decoder

def evaluate(decoder, Z, Y, name):
    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)
    device = next(decoder.parameters()).device
    decoder.eval()
    with torch.no_grad():
        pred = decoder(torch.tensor(Z, dtype=torch.float32, device=device))
        pred = pred.detach().cpu().numpy()
    r2s = []
    for i in range(Y.shape[1]):
        r2 = float(r2_score(Y[:, i], pred[:, i]))
        r2s.append(r2)
        print(f"{name} dim {i} R2: {r2:.8f}")
    mean_r2 = float(np.mean(r2s))
    print(f"{name} Mean R2: {mean_r2:.8f}")
    return mean_r2, r2s

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def orient_jacobian(arr, n_neurons):
    a = np.abs(to_numpy(arr))
    a = np.squeeze(a)
    if a.ndim < 2:
        raise RuntimeError(f"Unexpected Jacobian shape: {a.shape}")
    if a.shape[-2:] == (LATENT_DIM, n_neurons):
        if a.ndim > 2:
            a = a.mean(axis=tuple(range(a.ndim - 2)))
        result = a
    elif a.shape[-2:] == (n_neurons, LATENT_DIM):
        if a.ndim > 2:
            a = a.mean(axis=tuple(range(a.ndim - 2)))
        result = a.T
    else:
        latent_axes = [i for i, size in enumerate(a.shape) if size == LATENT_DIM]
        neuron_axes = [i for i, size in enumerate(a.shape) if size == n_neurons]
        pairs = [(la, na) for la in latent_axes for na in neuron_axes if la != na]
        if not pairs:
            raise RuntimeError(f"Could not orient Jacobian.\nRaw shape = {a.shape}\nLATENT_DIM = {LATENT_DIM}\nneurons = {n_neurons}")
        latent_axis, neuron_axis = pairs[0]
        a = np.moveaxis(a, (latent_axis, neuron_axis), (-2, -1))
        if a.ndim > 2:
            a = a.mean(axis=tuple(range(a.ndim - 2)))
        if a.shape == (n_neurons, LATENT_DIM):
            a = a.T
        result = a
    expected = (LATENT_DIM, n_neurons)
    if result.shape != expected:
        raise RuntimeError(f"Final Jacobian shape = {result.shape}, expected = {expected}")
    return result.astype(np.float32)

def compute_jacobian(model, X, tag):
    print("\n" + "=" * 100)
    print("COMPUTING JACOBIAN:", tag)
    print("=" * 100)
    net = model.solver_.model
    device = next(net.parameters()).device
    net.eval()
    n_time = X.shape[0]
    n_neurons = X.shape[1]
    if n_time <= ATTR_LEN + 1:
        raise RuntimeError(f"Not enough time points for ATTR_LEN={ATTR_LEN}. Got n_time={n_time}.")
    starts = np.linspace(0, n_time - ATTR_LEN - 1, ATTR_CHUNKS, dtype=int)
    jf_sum = np.zeros((LATENT_DIM, n_neurons), dtype=np.float64)
    total = 0
    for chunk_id, start in enumerate(starts):
        chunk = np.asarray(X[start:start + ATTR_LEN], dtype=np.float32)
        inp = torch.tensor(chunk, dtype=torch.float32, device=device, requires_grad=True)
        method = cebra.attribution.init(
            name="jacobian-based-batched",
            model=net,
            input_data=inp,
            output_dimension=LATENT_DIM
        )
        with torch.enable_grad():
            result = method.compute_attribution_map(batch_size=ATTR_BATCH)
        if "jf" not in result:
            raise RuntimeError(f"No 'jf' key in attribution result.\nAvailable keys: {list(result.keys())}")
        jf_chunk = orient_jacobian(result["jf"], n_neurons)
        weight = len(chunk)
        jf_sum += jf_chunk * weight
        total += weight
        print(f"chunk {chunk_id + 1:02d}/{len(starts):02d} done")
        del method, result, jf_chunk, inp
        cleanup()
    jf = (jf_sum / float(total)).astype(np.float32)
    print("Final JF shape:", jf.shape)
    npy_path = os.path.join(OUT, f"{tag}_JF.npy")
    np.save(npy_path, jf)
    plt.figure(figsize=(12, 8))
    plt.imshow(jf, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Mean absolute forward Jacobian")
    plt.xlabel("Neuron / input column")
    plt.ylabel("Latent dimension")
    plt.title(f"{tag} Forward Jacobian")
    png_path = os.path.join(OUT, f"{tag}_JF.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Jacobian NPY:", npy_path)
    print("Saved Jacobian PNG:", png_path)
    return jf

def save_decoder(decoder, tag):
    path = os.path.join(OUT, f"{tag}_decoder.pt")
    torch.save(decoder.state_dict(), path)
    print("Saved decoder:", path)

def main():
    print("\n" + "#" * 110)
    print("ACORN + CLEAN LABEL COMPARISON")
    print("#" * 110)
    X_train, X_test, Y_train, Y_test = load_perich()
    n_neurons = X_train.shape[1]
    print("\nTotal neurons:", n_neurons)
    print("Target dimensions:", Y_train.shape[1])

    eps = compute_adv_epsilon(X_train)

    acorn_model = train_acorn_label(X_train, Y_train, eps)
    Z_train_acorn = np.asarray(acorn_model.transform(X_train), dtype=np.float32)
    Z_test_acorn = np.asarray(acorn_model.transform(X_test), dtype=np.float32)
    print("\nACORN embeddings")
    print("Z train:", Z_train_acorn.shape)
    print("Z test :", Z_test_acorn.shape)
    acorn_decoder = train_decoder(Z_train_acorn, Y_train, tag="ACORN_LABEL")
    acorn_train_r2, acorn_train_r2_dims = evaluate(acorn_decoder, Z_train_acorn, Y_train, "ACORN TRAIN")
    acorn_test_r2, acorn_test_r2_dims = evaluate(acorn_decoder, Z_test_acorn, Y_test, "ACORN TEST")
    acorn_jf = compute_jacobian(acorn_model, X_train, tag="ACORN_LABEL")
    save_decoder(acorn_decoder, "ACORN_LABEL")

    cleanup()

    clean_model = train_clean_label(X_train, Y_train)
    Z_train_clean = np.asarray(clean_model.transform(X_train), dtype=np.float32)
    Z_test_clean = np.asarray(clean_model.transform(X_test), dtype=np.float32)
    print("\nCLEAN embeddings")
    print("Z train:", Z_train_clean.shape)
    print("Z test :", Z_test_clean.shape)
    clean_decoder = train_decoder(Z_train_clean, Y_train, tag="CLEAN_LABEL")
    clean_train_r2, clean_train_r2_dims = evaluate(clean_decoder, Z_train_clean, Y_train, "CLEAN TRAIN")
    clean_test_r2, clean_test_r2_dims = evaluate(clean_decoder, Z_test_clean, Y_test, "CLEAN TEST")
    clean_jf = compute_jacobian(clean_model, X_train, tag="CLEAN_LABEL")
    save_decoder(clean_decoder, "CLEAN_LABEL")

    print("\n" + "#" * 110)
    print("FINAL COMPARISON")
    print("#" * 110)
    print(f"CLEAN | neurons={n_neurons:3d} | train mean R2={clean_train_r2:.8f} | test mean R2={clean_test_r2:.8f}")
    print(f"ACORN | neurons={n_neurons:3d} | train mean R2={acorn_train_r2:.8f} | test mean R2={acorn_test_r2:.8f}")
    print("\n" + "-" * 100)
    print("TEST R2 COMPARISON")
    print("-" * 100)
    print(f"CLEAN TEST R2 : {clean_test_r2:.8f}")
    print(f"ACORN TEST R2 : {acorn_test_r2:.8f}")
    print("\n" + "-" * 100)
    print("JACOBIAN FILES")
    print("-" * 100)
    print("CLEAN:", os.path.join(OUT, "CLEAN_LABEL_JF.png"))
    print("ACORN:", os.path.join(OUT, "ACORN_LABEL_JF.png"))
    print("\n" + "=" * 110)
    print("DONE")
    print("=" * 110)
    print("Output directory:", OUT)

    del clean_decoder, acorn_decoder, clean_model, acorn_model
    del Z_train_clean, Z_test_clean, Z_train_acorn, Z_test_acorn
    cleanup()

if __name__ == "__main__":
    main()






# import os
# import sys
# import gc
# import random
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score
# from utils.constants import CEBRA_DIR
# from utils.min_distance import min_l2_distance

# from gru_decoder_monkey import MonkeyDecoder

# sys.path.insert(0, str(CEBRA_DIR))
# import cebra
# from cebra import CEBRA
# import cebra.attribution

# print("\nUsing CEBRA:")
# print(cebra.__file__)

# PERICH_DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw/"
# DATASET_NAME = "C-CO"
# DAY = 0
# SESSION = f"{DATASET_NAME}{DAY}"
# NPZ_PATH = os.path.join(PERICH_DATA_DIR, f"{SESSION}.npz")
# OUT = f"Teacher_test_{SESSION}"
# os.makedirs(OUT, exist_ok=True)

# SEED = 42

# def seed_all(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# seed_all(SEED)

# LATENT_DIM = 64
# HIDDEN = 64
# BATCH_SIZE = 2048
# MAX_ITER = 5000
# TEMPERATURE = 0.4
# OFFSET = 1
# MODEL_ARCH = "offset36-model-more-dropout"

# ADV_EPS = None
# ADV_STEPS = 10
# ATTACK_NORM = "linf"

# DECODER_HIDDEN = 512
# DECODER_LAYERS = 2
# DECODER_DROPOUT = 0.4
# DECODER_STEPS = 2500

# ATTR_CHUNKS = 16
# ATTR_LEN = 128
# ATTR_BATCH = 16

# def load_perich():
#     print("\n" + "=" * 90)
#     print("LOADING PERICH")
#     print("=" * 90)
#     print("File:", NPZ_PATH)
#     data = np.load(NPZ_PATH, allow_pickle=True)
#     train_data = data["train_data"].astype(np.float32)
#     valid_data = data["valid_data"].astype(np.float32)
#     train_label = data["train_label"].astype(np.float32)
#     valid_label = data["valid_label"].astype(np.float32)
#     Y_train = train_label
#     Y_test = valid_label
#     print("\nRAW")
#     print("X train:", train_data.shape)
#     print("X test :", valid_data.shape)
#     print("Y train:", Y_train.shape)
#     print("Y test :", Y_test.shape)
#     mean = train_data.mean(0)
#     std = train_data.std(0) + 1e-3
#     train_data = (train_data - mean) / std
#     test_mean = valid_data.mean(0)
#     test_std = valid_data.std(0) + 1e-3
#     valid_data = (valid_data - test_mean) / test_std
#     print("\nAFTER NORMALIZATION")
#     print("train mean:", float(train_data.mean()), "std:", float(train_data.std()))
#     print("test mean:", float(valid_data.mean()), "std:", float(valid_data.std()))
#     return (
#         train_data.astype(np.float32),
#         valid_data.astype(np.float32),
#         Y_train.astype(np.float32),
#         Y_test.astype(np.float32)
#     )

# def compute_adv_epsilon(X):
#     print("\n" + "=" * 90)
#     print("COMPUTING ADV EPSILON")
#     print("=" * 90)
#     x_tensor = torch.from_numpy(X).float()
#     min_dist = float(min_l2_distance(x_tensor))
#     adv_eps = min_dist / 2.0
#     adv_eps = max(adv_eps, 1e-6)
#     adv_alpha = adv_eps / 5.0
#     print("min L2 distance:", min_dist)
#     print("epsilon:", adv_eps)
#     print("alpha:", adv_alpha)
#     return adv_eps
    
# def build_cebra(adversarial=False):
#     global ADV_EPS
#     print("\nBuilding CEBRA")
#     print("mode:", "ACORN" if adversarial else "CLEAN")
#     if adversarial:
#         return CEBRA(
#             batch_size=BATCH_SIZE,
#             temperature=TEMPERATURE,
#             model_architecture=MODEL_ARCH,
#             time_offsets=OFFSET,
#             max_iterations=MAX_ITER,
#             output_dimension=LATENT_DIM,
#             num_hidden_units=HIDDEN,
#             training_mode="adversarial",
#             adv_alpha=ADV_EPS / 5,
#             adv_epsilon=ADV_EPS,
#             adv_steps=ADV_STEPS,
#             attack_norm=ATTACK_NORM,
#             device="cuda_if_available",
#             verbose=True
#         )
#     else:
#         return CEBRA(
#             batch_size=BATCH_SIZE,
#             temperature=TEMPERATURE,
#             model_architecture=MODEL_ARCH,
#             time_offsets=OFFSET,
#             max_iterations=MAX_ITER,
#             output_dimension=LATENT_DIM,
#             num_hidden_units=HIDDEN,
#             training_mode="clean",
#             device="cuda_if_available",
#             verbose=True
#         )
# def train_cebra(X_train, adversarial=False):
#     model = build_cebra(adversarial=adversarial)
#     name = "ACORN" if adversarial else "CLEAN"
#     print("\n" + "=" * 90)
#     print("TRAINING", name)
#     print("=" * 90)
#     model.fit(X_train.astype(np.float32))
#     return model

# def train_decoder(Z_train, Y_train, tag):
#     print("\n" + "=" * 90)
#     print("TRAIN DECODER:", tag)
#     print("=" * 90)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.tensor(Z_train, dtype=torch.float32, device=device)
#     y = torch.tensor(Y_train, dtype=torch.float32, device=device)
#     decoder = MonkeyDecoder(
#         LATENT_DIM,
#         DECODER_HIDDEN,
#         DECODER_LAYERS,
#         DECODER_DROPOUT,
#         False,
#         Y_train.shape[1],
#         n_train_steps=DECODER_STEPS
#     ).to(device)
#     print(decoder)
#     decoder.fit(x, y)
#     return decoder

# def evaluate_decoder(decoder, Z_test, Y_test, tag):
#     print("\n" + "=" * 90)
#     print(tag)
#     print("=" * 90)
#     device = next(decoder.parameters()).device
#     decoder.eval()
#     with torch.no_grad():
#         pred = decoder(torch.tensor(Z_test, dtype=torch.float32, device=device))
#         pred = pred.cpu().numpy()
#     r2s = []
#     for i in range(Y_test.shape[1]):
#         r2 = r2_score(Y_test[:, i], pred[:, i])
#         r2s.append(r2)
#         print("dim", i, "R2:", r2)
#     mean_r2 = float(np.mean(r2s))
#     print("Mean R2:", mean_r2)
#     return mean_r2

# def seed_all(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
# seed_all(SEED)

# def load_perich():
#     print("\n" + "=" * 90)
#     print("LOADING PERICH")
#     print("=" * 90)
#     print("File:", NPZ_PATH)
#     data = np.load(NPZ_PATH, allow_pickle=True)
#     train_data = data["train_data"].astype(np.float32)
#     valid_data = data["valid_data"].astype(np.float32)
#     train_label = data["train_label"].astype(np.float32)
#     valid_label = data["valid_label"].astype(np.float32)
#     Y_train = train_label
#     Y_test = valid_label
#     print("\nRAW")
#     print("X train:", train_data.shape)
#     print("X test :", valid_data.shape)
#     print("Y train:", Y_train.shape)
#     print("Y test :", Y_test.shape)
#     mean = train_data.mean(0)
#     std = train_data.std(0) + 1e-3
#     train_data = (train_data - mean) / std
#     test_mean = valid_data.mean(0)
#     test_std = valid_data.std(0) + 1e-3
#     valid_data = (valid_data - test_mean) / test_std
#     print("\nAFTER NORMALIZATION")
#     print("train mean:", float(train_data.mean()), "std:", float(train_data.std()))
#     print("test mean:", float(valid_data.mean()), "std:", float(valid_data.std()))
#     return (train_data.astype(np.float32), valid_data.astype(np.float32), Y_train.astype(np.float32), Y_test.astype(np.float32))

# def build_cebra(adversarial=False):
#     print("\nBuilding CEBRA")
#     print("mode:", "ACORN" if adversarial else "CLEAN")
#     if adversarial:
#         return CEBRA(
#             batch_size=BATCH_SIZE,
#             temperature=TEMPERATURE,
#             model_architecture=MODEL_ARCH,
#             time_offsets=OFFSET,
#             max_iterations=MAX_ITER,
#             output_dimension=LATENT_DIM,
#             num_hidden_units=HIDDEN,
#             training_mode="adversarial",
#             adv_alpha=ADV_EPS / 5,
#             adv_epsilon=ADV_EPS,
#             adv_steps=ADV_STEPS,
#             attack_norm=ATTACK_NORM,
#             device="cuda_if_available",
#             verbose=True
#         )
#     else:
#         return CEBRA(
#             batch_size=BATCH_SIZE,
#             temperature=TEMPERATURE,
#             model_architecture=MODEL_ARCH,
#             time_offsets=OFFSET,
#             max_iterations=MAX_ITER,
#             output_dimension=LATENT_DIM,
#             num_hidden_units=HIDDEN,
#             training_mode="clean",
#             device="cuda_if_available",
#             verbose=True
#         )

# def train_cebra(X_train, adversarial=False):
#     model = build_cebra(adversarial=adversarial)
#     name = "ACORN" if adversarial else "CLEAN"
#     print("\n" + "=" * 90)
#     print("TRAINING", name)
#     print("=" * 90)
#     model.fit(X_train.astype(np.float32))
#     return model

# def train_decoder(Z_train, Y_train, tag):
#     print("\n" + "=" * 90)
#     print("TRAIN DECODER:", tag)
#     print("=" * 90)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.tensor(Z_train, dtype=torch.float32, device=device)
#     y = torch.tensor(Y_train, dtype=torch.float32, device=device)
#     decoder = MonkeyDecoder(
#         LATENT_DIM,
#         DECODER_HIDDEN,
#         DECODER_LAYERS,
#         DECODER_DROPOUT,
#         False,
#         Y_train.shape[1],
#         n_train_steps=DECODER_STEPS
#     ).to(device)
#     print(decoder)
#     decoder.fit(x, y)
#     return decoder

# def evaluate_decoder(decoder, Z_test, Y_test, tag):
#     print("\n" + "=" * 90)
#     print(tag)
#     print("=" * 90)
#     device = next(decoder.parameters()).device
#     decoder.eval()
#     with torch.no_grad():
#         pred = decoder(torch.tensor(Z_test, dtype=torch.float32, device=device))
#         pred = pred.cpu().numpy()
#     r2s = []
#     for i in range(Y_test.shape[1]):
#         r2 = r2_score(Y_test[:, i], pred[:, i])
#         r2s.append(r2)
#         print("dim", i, "R2:", r2)
#     mean_r2 = float(np.mean(r2s))
#     print("Mean R2:", mean_r2)
#     return mean_r2

# def compute_jacobian(model, X, name):
#     print("\n" + "=" * 90)
#     print("JACOBIAN:", name)
#     print("=" * 90)
#     net = model.solver_.model
#     device = next(net.parameters()).device
#     net.eval()
#     n_neurons = X.shape[1]
#     starts = np.linspace(0, len(X) - ATTR_LEN - 1, ATTR_CHUNKS, dtype=int)
#     jf_sum = np.zeros((LATENT_DIM, n_neurons), dtype=np.float64)
#     total = 0
#     for i, start in enumerate(starts):
#         chunk = X[start:start + ATTR_LEN]
#         inp = torch.tensor(chunk, dtype=torch.float32, device=device, requires_grad=True)
#         method = cebra.attribution.init(
#             name="jacobian-based-batched",
#             model=net,
#             input_data=inp,
#             output_dimension=LATENT_DIM
#         )
#         with torch.enable_grad():
#             result = method.compute_attribution_map(batch_size=ATTR_BATCH)
#         jf = result["jf"]
#         jf = np.abs(np.asarray(jf))
#         jf = np.squeeze(jf)
#         if jf.shape != (LATENT_DIM, n_neurons):
#             jf = np.mean(jf, axis=tuple(range(jf.ndim - 2)))
#         jf_sum += jf * len(chunk)
#         total += len(chunk)
#         print("chunk", i + 1, "/", ATTR_CHUNKS)
#     jf = (jf_sum / total).astype(np.float32)
#     print("Final JF:", jf.shape)
#     np.save(os.path.join(OUT, f"{name}_JF.npy"), jf)
#     plt.figure(figsize=(12, 8))
#     plt.imshow(jf, aspect="auto")
#     plt.colorbar()
#     plt.title(f"{name} Forward Jacobian")
#     path = os.path.join(OUT, f"{name}_JF.png")
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print("saved:", path)
#     return jf

# def main():
#     global ADV_EPS
#     print("\n" + "=" * 100)
#     print("PERICH TEACHER SETUP TEST")
#     print("=" * 100)
#     X_train, X_test, Y_train, Y_test = load_perich()
#     ADV_EPS = compute_adv_epsilon(X_train)
#     clean_model = train_cebra(X_train, adversarial=False)
#     Z_train = clean_model.transform(X_train)
#     Z_test = clean_model.transform(X_test)
#     clean_decoder = train_decoder(Z_train, Y_train, "CLEAN")
#     clean_r2 = evaluate_decoder(clean_decoder, Z_test, Y_test, "CLEAN TEST")
#     compute_jacobian(clean_model, X_train, "CLEAN")
#     torch.save(clean_decoder.state_dict(), os.path.join(OUT, "clean_decoder.pt"))
#     del clean_model, clean_decoder
#     gc.collect()
#     torch.cuda.empty_cache()
#     acorn_model = train_cebra(X_train, adversarial=True)
#     Z_train = acorn_model.transform(X_train)
#     Z_test = acorn_model.transform(X_test)
#     acorn_decoder = train_decoder(Z_train, Y_train, "ACORN")
#     acorn_r2 = evaluate_decoder(acorn_decoder, Z_test, Y_test, "ACORN TEST")
#     compute_jacobian(acorn_model, X_train, "ACORN")
#     torch.save(acorn_decoder.state_dict(), os.path.join(OUT, "acorn_decoder.pt"))
#     print("\n" + "=" * 100)
#     print("FINAL")
#     print("=" * 100)
#     print("CLEAN R2:", clean_r2)
#     print("ACORN R2:", acorn_r2)
#     np.save(os.path.join(OUT, "summary.npy"), np.array([clean_r2, acorn_r2]))
#     print("\nDONE")
#     print("Output:", OUT)

# if __name__ == "__main__":
#     main()

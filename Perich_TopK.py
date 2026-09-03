import os
import sys
import gc
import csv
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utils.constants import CEBRA_DIR
from utils.min_distance import min_l2_distance
from gru_decoder_monkey import MonkeyDecoder

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
OUT = f"Teacher_test_{SESSION}_TOPK_6MODELS"
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
TOPK_N = None

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
    train_data = data["train_data"].astype(np.float32)
    valid_data = data["valid_data"].astype(np.float32)
    train_label = data["train_label"].astype(np.float32)
    valid_label = data["valid_label"].astype(np.float32)
    Y_train = train_label
    Y_test = valid_label
    print("\nRAW")
    print("X train:", train_data.shape)
    print("X test :", valid_data.shape)
    print("Y train:", Y_train.shape)
    print("Y test :", Y_test.shape)
    mean = train_data.mean(0)
    std = train_data.std(0) + 1e-3
    train_data = (train_data - mean) / std
    test_mean = valid_data.mean(0)
    test_std = valid_data.std(0) + 1e-3
    valid_data = (valid_data - test_mean) / test_std
    print("\nAFTER NORMALIZATION")
    print("train mean:", float(train_data.mean()), "std:", float(train_data.std()))
    print("test mean :", float(valid_data.mean()), "std:", float(valid_data.std()))
    if not np.isfinite(train_data).all():
        raise RuntimeError("X_train contains NaN or Inf.")
    if not np.isfinite(valid_data).all():
        raise RuntimeError("X_test contains NaN or Inf.")
    return (train_data.astype(np.float32), valid_data.astype(np.float32), Y_train.astype(np.float32), Y_test.astype(np.float32))

def compute_adv_epsilon(X, tag):
    print("\n" + "=" * 90)
    print("COMPUTING ADV EPSILON:", tag)
    print("=" * 90)
    x_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32)).float()
    min_dist = float(min_l2_distance(x_tensor))
    eps = max(min_dist / 2.0, 1e-6)
    alpha = eps / 5.0
    print("min L2 distance:", min_dist)
    print("epsilon        :", eps)
    print("alpha          :", alpha)
    return eps

def build_cebra(adversarial=False, adv_eps=None):
    print("\nBuilding CEBRA")
    print("mode:", "ACORN" if adversarial else "CLEAN")
    common = dict(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=OFFSET,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        device="cuda_if_available",
        verbose=True,
    )
    if adversarial:
        if adv_eps is None:
            raise ValueError("ACORN requires adv_eps.")
        return CEBRA(
            **common,
            training_mode="adversarial",
            adv_alpha=adv_eps / 5.0,
            adv_epsilon=adv_eps,
            adv_steps=ADV_STEPS,
            attack_norm=ATTACK_NORM,
        )
    return CEBRA(**common, training_mode="clean")

def train_cebra(X_train, tag, adversarial=False, adv_eps=None):
    print("\n" + "=" * 100)
    print("TRAINING CEBRA MODEL:", tag)
    print("=" * 100)
    print("X_train:", X_train.shape)
    model = build_cebra(adversarial=adversarial, adv_eps=adv_eps)
    model.fit(np.asarray(X_train, dtype=np.float32))
    return model

def train_decoder(Z_train, Y_train, tag):
    print("\n" + "=" * 90)
    print("TRAIN DECODER:", tag)
    print("=" * 90)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(Z_train, dtype=torch.float32, device=device)
    y = torch.tensor(Y_train, dtype=torch.float32, device=device)
    decoder = MonkeyDecoder(
        LATENT_DIM,
        DECODER_HIDDEN,
        DECODER_LAYERS,
        DECODER_DROPOUT,
        False,
        Y_train.shape[1],
        n_train_steps=DECODER_STEPS,
    ).to(device)
    print(decoder)
    decoder.fit(x, y)
    return decoder

def evaluate_decoder(decoder, Z, Y, tag):
    print("\n" + "=" * 90)
    print(tag)
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
        print(f"dim {i} R2: {r2:.8f}")
    mean_r2 = float(np.mean(r2s))
    print("Mean R2:", mean_r2)
    return mean_r2, r2s

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def orient_forward_jacobian(arr, n_neurons):
    a = np.abs(to_numpy(arr))
    a = np.squeeze(a)
    if a.ndim < 2:
        raise RuntimeError(f"Unexpected Jacobian shape: {a.shape}")
    if a.shape[-2:] == (LATENT_DIM, n_neurons):
        if a.ndim > 2:
            a = a.mean(axis=tuple(range(a.ndim - 2)))
        return a.astype(np.float32)
    if a.shape[-2:] == (n_neurons, LATENT_DIM):
        if a.ndim > 2:
            a = a.mean(axis=tuple(range(a.ndim - 2)))
        return a.T.astype(np.float32)
    latent_axes = [i for i, size in enumerate(a.shape) if size == LATENT_DIM]
    neuron_axes = [i for i, size in enumerate(a.shape) if size == n_neurons]
    pairs = [(la, na) for la in latent_axes for na in neuron_axes if la != na]
    if not pairs:
        raise RuntimeError(f"Could not orient Jacobian. raw shape={a.shape}, latent={LATENT_DIM}, neurons={n_neurons}")
    latent_axis, neuron_axis = max(pairs, key=lambda p: p[0] + p[1])
    a = np.moveaxis(a, (latent_axis, neuron_axis), (-2, -1))
    if a.ndim > 2:
        a = a.mean(axis=tuple(range(a.ndim - 2)))
    if a.shape == (n_neurons, LATENT_DIM):
        a = a.T
    expected = (LATENT_DIM, n_neurons)
    if a.shape != expected:
        raise RuntimeError(f"Final Jacobian shape={a.shape}, expected={expected}")
    return a.astype(np.float32)

def compute_jacobian(model, X, tag):
    print("\n" + "=" * 100)
    print("JACOBIAN:", tag)
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
    for i, start in enumerate(starts):
        chunk = np.asarray(X[start:start + ATTR_LEN], dtype=np.float32)
        inp = torch.tensor(chunk, dtype=torch.float32, device=device, requires_grad=True)
        method = cebra.attribution.init(
            name="jacobian-based-batched",
            model=net,
            input_data=inp,
            output_dimension=LATENT_DIM,
        )
        with torch.enable_grad():
            result = method.compute_attribution_map(batch_size=ATTR_BATCH)
        if "jf" not in result:
            raise RuntimeError(f"No 'jf' key in attribution result. Available keys={list(result.keys())}")
        jf_chunk = orient_forward_jacobian(result["jf"], n_neurons)
        weight = len(chunk)
        jf_sum += jf_chunk * weight
        total += weight
        print(f"chunk {i + 1:02d}/{len(starts):02d} done")
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
    print("Saved:", npy_path)
    print("Saved:", png_path)
    return jf

def get_topk_neurons(jf, k, selector_tag):
    scores = np.mean(np.abs(jf), axis=0)
    order = np.argsort(scores)[::-1]
    selected = order[:k].astype(np.int64)
    print("\n" + "=" * 90)
    print(f"TOP-{k} SELECTION FROM {selector_tag}")
    print("=" * 90)
    for rank, idx in enumerate(selected, start=1):
        print(f"{rank:02d}. neuron={int(idx):03d} score={float(scores[idx]):.12f}")
    np.save(os.path.join(OUT, f"{selector_tag}_top{k}_neurons.npy"), selected)
    np.save(os.path.join(OUT, f"{selector_tag}_all_neuron_scores.npy"), scores.astype(np.float32))
    return selected, scores

def save_decoder(decoder, tag):
    path = os.path.join(OUT, f"{tag}_decoder.pt")
    torch.save(decoder.state_dict(), path)
    print("Saved decoder:", path)

def run_experiment(tag, X_train, X_test, Y_train, Y_test, adversarial, compute_jf=True):
    print("\n" + "#" * 110)
    print("RUN:", tag)
    print("#" * 110)
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    adv_eps = None
    if adversarial:
        adv_eps = compute_adv_epsilon(X_train, tag=tag)
    model = train_cebra(X_train=X_train, tag=tag, adversarial=adversarial, adv_eps=adv_eps)
    Z_train = np.asarray(model.transform(X_train), dtype=np.float32)
    Z_test = np.asarray(model.transform(X_test), dtype=np.float32)
    print("\nEmbeddings")
    print("Z_train:", Z_train.shape)
    print("Z_test :", Z_test.shape)
    decoder = train_decoder(Z_train, Y_train, tag=tag)
    train_mean_r2, train_r2_dims = evaluate_decoder(decoder, Z_train, Y_train, tag=f"{tag} TRAIN")
    test_mean_r2, test_r2_dims = evaluate_decoder(decoder, Z_test, Y_test, tag=f"{tag} TEST")
    save_decoder(decoder, tag)
    jf = None
    if compute_jf:
        jf = compute_jacobian(model, X_train, tag)
    result = {
        "tag": tag,
        "model": "ACORN" if adversarial else "CLEAN",
        "n_neurons": int(X_train.shape[1]),
        "adv_epsilon": float(adv_eps) if adv_eps is not None else np.nan,
        "train_mean_r2": float(train_mean_r2),
        "test_mean_r2": float(test_mean_r2),
        "train_r2_dims": train_r2_dims,
        "test_r2_dims": test_r2_dims,
    }
    del decoder, model, Z_train, Z_test
    cleanup()
    return result, jf

def save_summary(results, n_targets):
    csv_path = os.path.join(OUT, f"{SESSION}_topk_6models_summary.csv")
    fieldnames = ["tag", "model", "n_neurons", "adv_epsilon", "train_mean_r2", "test_mean_r2"]
    fieldnames += [f"train_r2_dim{i}" for i in range(n_targets)]
    fieldnames += [f"test_r2_dim{i}" for i in range(n_targets)]
    rows = []
    for r in results:
        row = {
            "tag": r["tag"],
            "model": r["model"],
            "n_neurons": r["n_neurons"],
            "adv_epsilon": r["adv_epsilon"],
            "train_mean_r2": r["train_mean_r2"],
            "test_mean_r2": r["test_mean_r2"],
        }
        for i, value in enumerate(r["train_r2_dims"]):
            row[f"train_r2_dim{i}"] = value
        for i, value in enumerate(r["test_r2_dims"]):
            row[f"test_r2_dim{i}"] = value
        rows.append(row)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.save(os.path.join(OUT, f"{SESSION}_test_mean_r2.npy"),
            np.asarray([r["test_mean_r2"] for r in results], dtype=np.float32))
    print("\nSaved summary:", csv_path)

def main():
    print("\n" + "=" * 110)
    print("PERICH TOP-K: 2 FULL MODELS + 4 REDUCED MODELS")
    print("=" * 110)
    X_train, X_test, Y_train, Y_test = load_perich()
    n_neurons = X_train.shape[1]
    n_targets = Y_train.shape[1]
    if TOPK_N is None:
        k = int(np.floor(np.sqrt(n_neurons)))
    else:
        k = int(TOPK_N)
    if k < 1 or k > n_neurons:
        raise ValueError(f"Invalid K={k} for n_neurons={n_neurons}")
    print("\nTotal neurons:", n_neurons)
    print("Top-K        :", k)
    all_results = []
    full_clean_result, clean_jf = run_experiment(
        tag="FULL_CLEAN",
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test,
        adversarial=False,
        compute_jf=True,
    )
    all_results.append(full_clean_result)
    full_acorn_result, acorn_jf = run_experiment(
        tag="FULL_ACORN",
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test,
        adversarial=True,
        compute_jf=True,
    )
    all_results.append(full_acorn_result)
    clean_topk, clean_scores = get_topk_neurons(clean_jf, k=k, selector_tag="FULL_CLEAN_JF")
    acorn_topk, acorn_scores = get_topk_neurons(acorn_jf, k=k, selector_tag="FULL_ACORN_JF")
    selection_csv = os.path.join(OUT, f"{SESSION}_topk_selections.csv")
    with open(selection_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["selector", "rank", "neuron_index", "score"])
        for rank, idx in enumerate(clean_topk, start=1):
            writer.writerow(["FULL_CLEAN_JF", rank, int(idx), float(clean_scores[idx])])
        for rank, idx in enumerate(acorn_topk, start=1):
            writer.writerow(["FULL_ACORN_JF", rank, int(idx), float(acorn_scores[idx])])
    print("Saved selections:", selection_csv)
    X_train_clean_topk = X_train[:, clean_topk].astype(np.float32)
    X_test_clean_topk = X_test[:, clean_topk].astype(np.float32)
    X_train_acorn_topk = X_train[:, acorn_topk].astype(np.float32)
    X_test_acorn_topk = X_test[:, acorn_topk].astype(np.float32)

    result, _ = run_experiment(
        tag=f"TOP{k}_BY_CLEANJF__CLEAN",
        X_train=X_train_clean_topk,
        X_test=X_test_clean_topk,
        Y_train=Y_train,
        Y_test=Y_test,
        adversarial=False,
        compute_jf=True,
    )
    all_results.append(result)

    result, _ = run_experiment(
        tag=f"TOP{k}_BY_CLEANJF__ACORN",
        X_train=X_train_clean_topk,
        X_test=X_test_clean_topk,
        Y_train=Y_train,
        Y_test=Y_test,
        adversarial=True,
        compute_jf=True,
    )
    all_results.append(result)

    result, _ = run_experiment(
        tag=f"TOP{k}_BY_ACORNJF__CLEAN",
        X_train=X_train_acorn_topk,
        X_test=X_test_acorn_topk,
        Y_train=Y_train,
        Y_test=Y_test,
        adversarial=False,
        compute_jf=True,
    )
    all_results.append(result)

    result, _ = run_experiment(
        tag=f"TOP{k}_BY_ACORNJF__ACORN",
        X_train=X_train_acorn_topk,
        X_test=X_test_acorn_topk,
        Y_train=Y_train,
        Y_test=Y_test,
        adversarial=True,
        compute_jf=True,
    )
    all_results.append(result)

    print("\n" + "#" * 110)
    print("FINAL TEST MEAN R2 -- ALL 6 MODELS")
    print("#" * 110)
    for r in all_results:
        print(f"{r['tag']:34s} | neurons={r['n_neurons']:3d} | test mean R2={r['test_mean_r2']:.8f}")
    save_summary(all_results, n_targets=n_targets)
    print("\n" + "=" * 110)
    print("DONE")
    print("=" * 110)
    print("Output directory:", OUT)

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
 
 
# # =====================================================================
# # NEW: top-K neuron-attribution extension
# # Nothing above this line was changed. Everything below is additive.
# # =====================================================================
 
# def get_topk_neurons(jf, k):
#     """jf: (LATENT_DIM, n_neurons) forward-Jacobian matrix. Returns the
#     indices of the k neurons with the highest mean |dz/dx| across latent
#     dims (same scoring convention as the Mihili top-K script)."""
#     scores = jf.mean(axis=0)
#     order = np.argsort(scores)[::-1]
#     return order[:k].astype(int)
 
 
# def build_cebra_with_eps(adversarial, eps):
#     """Same as build_cebra(), but takes an explicit epsilon instead of
#     reading the global ADV_EPS. Needed when retraining on a reduced neuron
#     subset: the L2 geometry (and therefore the right epsilon) is different
#     from the full neuron set the global ADV_EPS was computed on."""
#     if not adversarial:
#         return build_cebra(adversarial=False)
#     return CEBRA(
#         batch_size=BATCH_SIZE,
#         temperature=TEMPERATURE,
#         model_architecture=MODEL_ARCH,
#         time_offsets=OFFSET,
#         max_iterations=MAX_ITER,
#         output_dimension=LATENT_DIM,
#         num_hidden_units=HIDDEN,
#         training_mode="adversarial",
#         adv_alpha=eps / 5,
#         adv_epsilon=eps,
#         adv_steps=ADV_STEPS,
#         attack_norm=ATTACK_NORM,
#         device="cuda_if_available",
#         verbose=True,
#     )
 
 
# def compute_jacobian_plot_only(model, X, name):
#     """Same computation as compute_jacobian(), but saves ONLY the heatmap
#     PNG -- no .npy score file, no other artifacts."""
#     print("\n" + "=" * 90)
#     print("JACOBIAN (reduced):", name)
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
 
#     plt.figure(figsize=(12, 8))
#     plt.imshow(jf, aspect="auto")
#     plt.colorbar()
#     plt.title(f"{name} Forward Jacobian (reduced)")
#     path = os.path.join(OUT, f"{name}_JF.png")
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print("saved:", path)
 
 
# def run_topk_reduced_jacobians(clean_jf, acorn_jf, X_train, n_neurons):
#     """The 4 reduced cases: {CLEAN_topJF, ACORN_topJF} neuron subsets x
#     {CEBRA, ACORN} retraining. For each, only the Jacobian heatmap PNG is
#     saved -- no decoder, no .npy, no checkpoints."""
#     print("\n" + "#" * 100)
#     print("TOP-K REDUCED JACOBIANS (4 cases)")
#     print("#" * 100)
 
#     k = int(np.floor(np.sqrt(n_neurons)))
#     print("N neurons:", n_neurons, "| K = floor(sqrt(N)) =", k)
 
#     clean_topk = get_topk_neurons(clean_jf, k)
#     acorn_topk = get_topk_neurons(acorn_jf, k)
#     print("CLEAN top-K neurons:", clean_topk.tolist())
#     print("ACORN top-K neurons:", acorn_topk.tolist())
 
#     neuron_sets = {
#         "CLEAN_topJF": clean_topk,
#         "ACORN_topJF": acorn_topk,
#     }
 
#     for selector_name, idxs in neuron_sets.items():
#         X_reduced = X_train[:, idxs]
#         print(f"\n--- neuron set: {selector_name} | kept {len(idxs)}/{n_neurons} neurons ---")
 
#         for adversarial, mode_name in [(False, "CEBRA"), (True, "ACORN")]:
#             tag = f"{selector_name}__{mode_name}"
#             print(f"\n=== retraining {tag} ===")
 
#             if adversarial:
#                 eps = compute_adv_epsilon(X_reduced)
#                 model = build_cebra_with_eps(adversarial=True, eps=eps)
#             else:
#                 model = build_cebra_with_eps(adversarial=False, eps=None)
 
#             model.fit(X_reduced.astype(np.float32))
#             compute_jacobian_plot_only(model, X_reduced, tag)
 
#             del model
#             gc.collect()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
 
#     print("\nDONE: top-K reduced Jacobians (4/4 cases)")
 
 
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
#     clean_jf = compute_jacobian(clean_model, X_train, "CLEAN")
#     torch.save(clean_decoder.state_dict(), os.path.join(OUT, "clean_decoder.pt"))
#     del clean_model, clean_decoder
#     gc.collect()
#     torch.cuda.empty_cache()
 
#     acorn_model = train_cebra(X_train, adversarial=True)
#     Z_train = acorn_model.transform(X_train)
#     Z_test = acorn_model.transform(X_test)
#     acorn_decoder = train_decoder(Z_train, Y_train, "ACORN")
#     acorn_r2 = evaluate_decoder(acorn_decoder, Z_test, Y_test, "ACORN TEST")
#     acorn_jf = compute_jacobian(acorn_model, X_train, "ACORN")
#     torch.save(acorn_decoder.state_dict(), os.path.join(OUT, "acorn_decoder.pt"))
#     del acorn_model, acorn_decoder
#     gc.collect()
#     torch.cuda.empty_cache()
 
#     # ---- NEW: the 4 top-K reduced-neuron Jacobians ----
#     run_topk_reduced_jacobians(clean_jf, acorn_jf, X_train, X_train.shape[1])
 
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

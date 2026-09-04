### CEBRA Behavior ###
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
OUT = f"ACORN_LABEL_{SESSION}"
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

seed_all(SEED)

def load_perich():
    print("\nLoading data")
    data = np.load(NPZ_PATH, allow_pickle=True)
    X_train = data["train_data"].astype(np.float32)
    X_test = data["valid_data"].astype(np.float32)
    Y_train = data["train_label"].astype(np.float32)
    Y_test = data["valid_label"].astype(np.float32)
    print("X train:", X_train.shape)
    print("X test :", X_test.shape)
    print("Y train:", Y_train.shape)
    print("Y test :", Y_test.shape)
    return (X_train.astype(np.float32), X_test.astype(np.float32), Y_train.astype(np.float32), Y_test.astype(np.float32))

def compute_adv_epsilon(X):
    print("\nComputing epsilon")
    dist = float(min_l2_distance(torch.from_numpy(X).float()))
    eps = max(dist / 2.0, 1e-6)
    eps = 0.1
    print("min L2 distance:", dist)
    print("epsilon:", eps)
    return eps

def build_acorn(eps):
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

def train_acorn_label(X_train, Y_train, eps):
    print("\nTraining ACORN + LABEL")
    model = build_acorn(eps)
    model.fit(X_train, Y_train)
    return model

def train_decoder(Z, Y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = MonkeyDecoder(LATENT_DIM, DECODER_HIDDEN, DECODER_LAYERS, DECODER_DROPOUT, False, Y.shape[1], n_train_steps=DECODER_STEPS).to(device)
    decoder.fit(torch.tensor(Z, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device))
    return decoder

def evaluate(decoder, Z, Y, name):
    device = next(decoder.parameters()).device
    decoder.eval()
    with torch.no_grad():
        pred = decoder(torch.tensor(Z, dtype=torch.float32, device=device))
        pred = pred.cpu().numpy()
    r2s = []
    for i in range(Y.shape[1]):
        r2 = r2_score(Y[:, i], pred[:, i])
        r2s.append(r2)
        print(name, "dim", i, "R2:", r2)
    print(name, "Mean R2:", np.mean(r2s))
    return float(np.mean(r2s))

def compute_jacobian(model, X):
    print("\nComputing Jacobian")
    net = model.solver_.model
    device = next(net.parameters()).device
    net.eval()
    n_neurons = X.shape[1]
    starts = np.linspace(0, len(X) - ATTR_LEN - 1, ATTR_CHUNKS, dtype=int)
    jf_sum = np.zeros((LATENT_DIM, n_neurons))
    total = 0
    for start in starts:
        chunk = X[start:start + ATTR_LEN]
        inp = torch.tensor(chunk, dtype=torch.float32, device=device, requires_grad=True)
        method = cebra.attribution.init(name="jacobian-based-batched", model=net, input_data=inp, output_dimension=LATENT_DIM)
        with torch.enable_grad():
            result = method.compute_attribution_map(batch_size=ATTR_BATCH)
        jf = np.abs(np.asarray(result["jf"]))
        jf = np.squeeze(jf)
        if jf.shape != (LATENT_DIM, n_neurons):
            jf = np.mean(jf, axis=tuple(range(jf.ndim - 2)))
        jf_sum += jf * len(chunk)
        total += len(chunk)
    jf = (jf_sum / total).astype(np.float32)
    np.save(os.path.join(OUT, "ACORN_LABEL_JF.npy"), jf)
    plt.figure(figsize=(12, 8))
    plt.imshow(jf, aspect="auto")
    plt.colorbar()
    plt.title("ACORN + LABEL Forward Jacobian")
    path = os.path.join(OUT, "ACORN_LABEL_JF.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Jacobian:", path)
    return jf

def main():
    X_train, X_test, Y_train, Y_test = load_perich()
    eps = compute_adv_epsilon(X_train)
    model = train_acorn_label(X_train, Y_train, eps)
    Z_train = model.transform(X_train)
    Z_test = model.transform(X_test)
    decoder = train_decoder(Z_train, Y_train)
    train_r2 = evaluate(decoder, Z_train, Y_train, "TRAIN")
    test_r2 = evaluate(decoder, Z_test, Y_test, "TEST")
    compute_jacobian(model, X_train)
    torch.save(decoder.state_dict(), os.path.join(OUT, "decoder.pt"))
    print("\n====================")
    print("FINAL")
    print("====================")
    print("TRAIN R2:", train_r2)
    print("TEST R2:", test_r2)
    print("Output:", OUT)

if __name__ == "__main__":
    main()

### CEBRA Time ###
# import os
# import sys
# import gc
# import random
# import numpy as np
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score
# from utils.constants import CEBRA_DIR
# from utils.min_distance import min_l2_distance
# from gru_decoder_monkey import MonkeyDecoder

# for module_name in list(sys.modules):
#     if module_name == "cebra" or module_name.startswith("cebra."):
#         del sys.modules[module_name]
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
# OUT = f"ACORN_seed_stability_{SESSION}"
# os.makedirs(OUT, exist_ok=True)

# SEEDS = [0, 1, 2, 3, 4]
# LATENT_DIM = 64
# HIDDEN = 64
# BATCH_SIZE = 2048
# MAX_ITER = 5000
# TEMPERATURE = 0.4
# OFFSET = 1
# MODEL_ARCH = "offset36-model-more-dropout"
# ADV_STEPS = 10
# ATTACK_NORM = "linf"
# DECODER_HIDDEN = 512
# DECODER_LAYERS = 2
# DECODER_DROPOUT = 0.4
# DECODER_STEPS = 2500
# ATTR_CHUNKS = 16
# ATTR_LEN = 128
# ATTR_BATCH = 16
# TOP_N = 10
# DEVICE = "cuda_if_available"

# def seed_all(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# def cleanup(*objects):
#     for obj in objects:
#         try:
#             del obj
#         except Exception:
#             pass
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         try:
#             torch.cuda.ipc_collect()
#         except Exception:
#             pass

# def load_perich():
#     print("\n" + "=" * 100)
#     print("LOADING PERICH")
#     print("=" * 100)
#     print("File:", NPZ_PATH)
#     if not os.path.exists(NPZ_PATH):
#         raise FileNotFoundError(f"Could not find:\n{NPZ_PATH}")
#     data = np.load(NPZ_PATH, allow_pickle=True)
#     train_data = data["train_data"].astype(np.float32, copy=False)
#     valid_data = data["valid_data"].astype(np.float32, copy=False)
#     train_label = data["train_label"].astype(np.float32, copy=False)
#     valid_label = data["valid_label"].astype(np.float32, copy=False)
#     Y_train = train_label
#     Y_test = valid_label
#     print("\nRAW")
#     print("X train:", train_data.shape)
#     print("X test :", valid_data.shape)
#     print("Y train:", Y_train.shape)
#     print("Y test :", Y_test.shape)
#     train_mean = train_data.mean(0)
#     train_std = train_data.std(0) + 1e-3
#     train_data = (train_data - train_mean) / train_std
#     test_mean = valid_data.mean(0)
#     test_std = valid_data.std(0) + 1e-3
#     valid_data = (valid_data - test_mean) / test_std
#     print("\nAFTER NORMALIZATION")
#     print("train mean:", float(train_data.mean()), "std:", float(train_data.std()))
#     print("test mean :", float(valid_data.mean()), "std:", float(valid_data.std()))
#     if not np.isfinite(train_data).all():
#         raise RuntimeError("train_data contains NaN/Inf.")
#     if not np.isfinite(valid_data).all():
#         raise RuntimeError("valid_data contains NaN/Inf.")
#     if not np.isfinite(Y_train).all():
#         raise RuntimeError("Y_train contains NaN/Inf.")
#     if not np.isfinite(Y_test).all():
#         raise RuntimeError("Y_test contains NaN/Inf.")
#     return (
#         train_data.astype(np.float32, copy=False),
#         valid_data.astype(np.float32, copy=False),
#         Y_train.astype(np.float32, copy=False),
#         Y_test.astype(np.float32, copy=False),
#     )

# def compute_adv_epsilon(X):
#     print("\n" + "=" * 100)
#     print("COMPUTING ADV EPSILON FROM DATA")
#     print("=" * 100)
#     x_tensor = torch.from_numpy(X).float()
#     min_dist = float(min_l2_distance(x_tensor))
#     adv_eps = min_dist / 2.0
#     adv_eps = max(adv_eps, 1e-6)
#     adv_alpha = adv_eps / 5.0
#     print("min L2 distance:", min_dist)
#     print("adv epsilon    :", adv_eps)
#     print("adv alpha      :", adv_alpha)
#     cleanup(x_tensor)
#     return adv_eps

# def build_acorn(adv_eps):
#     return CEBRA(
#         batch_size=BATCH_SIZE,
#         temperature=TEMPERATURE,
#         model_architecture=MODEL_ARCH,
#         time_offsets=OFFSET,
#         max_iterations=MAX_ITER,
#         output_dimension=LATENT_DIM,
#         num_hidden_units=HIDDEN,
#         training_mode="adversarial",
#         adv_alpha=adv_eps / 5.0,
#         adv_epsilon=adv_eps,
#         adv_steps=ADV_STEPS,
#         attack_norm=ATTACK_NORM,
#         device=DEVICE,
#         verbose=True,
#     )

# def train_acorn(X_train, adv_eps, seed):
#     print("\n" + "=" * 100)
#     print(f"TRAINING ACORN | seed={seed}")
#     print("=" * 100)
#     seed_all(seed)
#     model = build_acorn(adv_eps)
#     model.fit(X_train.astype(np.float32, copy=False))
#     return model

# def train_decoder(Z_train, Y_train, seed):
#     print("\n" + "=" * 100)
#     print(f"TRAIN DECODER | seed={seed}")
#     print("=" * 100)
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
#     try:
#         decoder.fit(x, y, seed=seed)
#     except TypeError:
#         decoder.fit(x, y)
#     return decoder

# def evaluate_decoder(decoder, Z, Y, tag):
#     print("\n" + "=" * 100)
#     print(tag)
#     print("=" * 100)
#     device = next(decoder.parameters()).device
#     decoder.eval()
#     with torch.no_grad():
#         pred = decoder(torch.tensor(Z, dtype=torch.float32, device=device))
#         pred = pred.cpu().numpy()
#     r2s = []
#     for i in range(Y.shape[1]):
#         r2 = r2_score(Y[:, i], pred[:, i])
#         r2s.append(float(r2))
#         print(f"dim {i} R2: {r2}")
#     mean_r2 = float(np.mean(r2s))
#     print("Mean R2:", mean_r2)
#     return {"mean_r2": mean_r2, "per_dim_r2": r2s}

# def to_numpy(x):
#     if isinstance(x, torch.Tensor):
#         return x.detach().cpu().numpy()
#     return np.asarray(x)

# def orient_forward_jacobian(arr, n_neurons):
#     a = np.abs(to_numpy(arr))
#     a = np.squeeze(a)
#     latent_axes = [i for i, size in enumerate(a.shape) if size == LATENT_DIM]
#     neuron_axes = [i for i, size in enumerate(a.shape) if size == n_neurons]
#     if not latent_axes:
#         raise RuntimeError(f"Could not find latent axis in attribution output. shape={a.shape}")
#     if not neuron_axes:
#         raise RuntimeError(f"Could not find neuron axis in attribution output. shape={a.shape}")
#     latent_axis = latent_axes[-1]
#     neuron_axis = neuron_axes[-1]
#     if latent_axis == neuron_axis:
#         raise RuntimeError(f"Ambiguous Jacobian shape: {a.shape}")
#     a = np.moveaxis(a, (latent_axis, neuron_axis), (-2, -1))
#     if a.ndim > 2:
#         a = a.mean(axis=tuple(range(a.ndim - 2)))
#     if a.shape == (n_neurons, LATENT_DIM):
#         a = a.T
#     expected = (LATENT_DIM, n_neurons)
#     if a.shape != expected:
#         raise RuntimeError(f"Final Jacobian shape={a.shape}, expected={expected}")
#     return a.astype(np.float32)

# def compute_jacobian(model, X, seed):
#     print("\n" + "=" * 100)
#     print(f"FORWARD JACOBIAN | seed={seed}")
#     print("=" * 100)
#     net = model.solver_.model
#     device = next(net.parameters()).device
#     net.eval()
#     if hasattr(net, "split_outputs"):
#         net.split_outputs = False
#     n_time = X.shape[0]
#     n_neurons = X.shape[1]
#     max_start = n_time - ATTR_LEN - 1
#     if max_start <= 0:
#         raise RuntimeError("Not enough samples for attribution.")
#     starts = np.linspace(0, max_start, ATTR_CHUNKS, dtype=int)
#     jf_sum = np.zeros((LATENT_DIM, n_neurons), dtype=np.float64)
#     total_weight = 0
#     print("n_time       :", n_time)
#     print("n_neurons    :", n_neurons)
#     print("chunks       :", ATTR_CHUNKS)
#     print("chunk length :", ATTR_LEN)
#     print("attr batch   :", ATTR_BATCH)
#     for chunk_id, start in enumerate(starts, start=1):
#         stop = start + ATTR_LEN
#         chunk = X[start:stop].astype(np.float32, copy=True)
#         inp = torch.tensor(chunk, dtype=torch.float32, device=device, requires_grad=True)
#         method = cebra.attribution.init(
#             name="jacobian-based-batched",
#             model=net,
#             input_data=inp,
#             output_dimension=LATENT_DIM
#         )
#         with torch.enable_grad():
#             result = method.compute_attribution_map(batch_size=ATTR_BATCH)
#         if "jf" not in result:
#             raise RuntimeError(f"'jf' not found in attribution output. Keys={list(result.keys())}")
#         jf_chunk = orient_forward_jacobian(result["jf"], n_neurons)
#         weight = len(chunk)
#         jf_sum += jf_chunk * weight
#         total_weight += weight
#         print(f"chunk {chunk_id:02d}/{len(starts):02d} done")
#         cleanup(method, result, inp, chunk, jf_chunk)
#     jf = (jf_sum / total_weight).astype(np.float32)
#     print("Final JF shape:", jf.shape)
#     return jf

# def save_jacobian_plot(jf, seed):
#     path = os.path.join(OUT, f"seed_{seed}_JF.png")
#     plt.figure(figsize=(12, 8))
#     plt.imshow(jf, aspect="auto", interpolation="nearest")
#     plt.colorbar(label="Mean absolute forward Jacobian")
#     plt.title(f"ACORN Forward Jacobian | seed={seed}")
#     plt.xlabel("Neuron")
#     plt.ylabel("Latent dimension")
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print("Saved plot:", path)
#     return path

# def get_top_neurons_from_jf(jf, top_n=10):
#     scores = np.mean(np.abs(jf), axis=0)
#     order = np.argsort(scores)[::-1]
#     top_idx = order[:top_n]
#     top_scores = scores[top_idx]
#     return top_idx.astype(int), top_scores.astype(np.float32)

# def run_one_seed(seed, X_train, X_test, Y_train, Y_test, adv_eps):
#     print("\n" + "#" * 120)
#     print(f"RUNNING SEED = {seed}")
#     print("#" * 120)
#     seed_all(seed)
#     model = train_acorn(X_train, adv_eps=adv_eps, seed=seed)
#     Z_train = np.asarray(model.transform(X_train.astype(np.float32, copy=False)), dtype=np.float32)
#     Z_test = np.asarray(model.transform(X_test.astype(np.float32, copy=False)), dtype=np.float32)
#     print("\nEmbeddings")
#     print("Z_train:", Z_train.shape)
#     print("Z_test :", Z_test.shape)
#     decoder = train_decoder(Z_train, Y_train, seed=seed)
#     train_eval = evaluate_decoder(decoder, Z_train, Y_train, f"ACORN TRAIN | seed={seed}")
#     test_eval = evaluate_decoder(decoder, Z_test, Y_test, f"ACORN TEST | seed={seed}")
#     jf = compute_jacobian(model, X_train, seed=seed)
#     plot_path = save_jacobian_plot(jf, seed)
#     jf_npy_path = os.path.join(OUT, f"seed_{seed}_JF.npy")
#     np.save(jf_npy_path, jf)
#     print("Saved array:", jf_npy_path)
#     top_idx, top_scores = get_top_neurons_from_jf(jf, top_n=TOP_N)
#     print("\n" + "=" * 100)
#     print(f"TOP-{TOP_N} NEURONS | seed={seed}")
#     print("=" * 100)
#     for rank, (idx, score) in enumerate(zip(top_idx, top_scores), start=1):
#         print(f"{rank:2d}. neuron={idx:3d} score={score:.12f}")
#     decoder_path = os.path.join(OUT, f"seed_{seed}_decoder.pt")
#     torch.save(decoder.state_dict(), decoder_path)
#     print("Saved decoder:", decoder_path)
#     result_row = {
#         "seed": seed,
#         "adv_epsilon": float(adv_eps),
#         "train_mean_r2": float(train_eval["mean_r2"]),
#         "test_mean_r2": float(test_eval["mean_r2"]),
#         "jacobian_png": plot_path,
#         "jacobian_npy": jf_npy_path,
#         "decoder_path": decoder_path,
#         "top10_neurons": ",".join(map(str, top_idx.tolist())),
#         "top10_scores": ",".join([f"{x:.12f}" for x in top_scores.tolist()]),
#     }
#     for i, r2 in enumerate(train_eval["per_dim_r2"]):
#         result_row[f"train_r2_dim_{i}"] = float(r2)
#     for i, r2 in enumerate(test_eval["per_dim_r2"]):
#         result_row[f"test_r2_dim_{i}"] = float(r2)
#     top_rows = []
#     for rank, (idx, score) in enumerate(zip(top_idx, top_scores), start=1):
#         top_rows.append({
#             "seed": seed,
#             "rank": rank,
#             "neuron_index": int(idx),
#             "score": float(score),
#         })
#     cleanup(model, decoder, Z_train, Z_test, jf)
#     return result_row, top_rows

# def main():
#     print("\n" + "=" * 120)
#     print("ACORN SEED STABILITY TEST")
#     print("=" * 120)
#     print("Session          :", SESSION)
#     print("Seeds            :", SEEDS)
#     print("Latent dim       :", LATENT_DIM)
#     print("Hidden           :", HIDDEN)
#     print("Batch size       :", BATCH_SIZE)
#     print("Max iter         :", MAX_ITER)
#     print("Temperature      :", TEMPERATURE)
#     print("Offset           :", OFFSET)
#     print("Model arch       :", MODEL_ARCH)
#     print("Attack norm      :", ATTACK_NORM)
#     print("Attack steps     :", ADV_STEPS)
#     print("Decoder hidden   :", DECODER_HIDDEN)
#     print("Decoder layers   :", DECODER_LAYERS)
#     print("Decoder dropout  :", DECODER_DROPOUT)
#     print("Decoder steps    :", DECODER_STEPS)
#     print("Attribution      :", f"{ATTR_CHUNKS} chunks x len {ATTR_LEN}")
#     print("Top neurons      :", TOP_N)
#     print("Output dir       :", OUT)
#     X_train, X_test, Y_train, Y_test = load_perich()
#     adv_eps = compute_adv_epsilon(X_train)
#     results = []
#     top_neuron_rows = []
#     for seed in SEEDS:
#         seed_result, seed_top_rows = run_one_seed(
#             seed=seed,
#             X_train=X_train,
#             X_test=X_test,
#             Y_train=Y_train,
#             Y_test=Y_test,
#             adv_eps=adv_eps
#         )
#         results.append(seed_result)
#         top_neuron_rows.extend(seed_top_rows)
#     results_df = pd.DataFrame(results)
#     results_csv = os.path.join(OUT, "acorn_seed_results.csv")
#     results_df.to_csv(results_csv, index=False)
#     top_df = pd.DataFrame(top_neuron_rows)
#     top_csv = os.path.join(OUT, "acorn_seed_top10_neurons.csv")
#     top_df.to_csv(top_csv, index=False)
#     numeric_cols = ["train_mean_r2", "test_mean_r2"]
#     for i in range(Y_train.shape[1]):
#         numeric_cols.append(f"train_r2_dim_{i}")
#     for i in range(Y_test.shape[1]):
#         numeric_cols.append(f"test_r2_dim_{i}")
#     summary_rows = []
#     for col in numeric_cols:
#         values = results_df[col].values.astype(float)
#         summary_rows.append({
#             "metric": col,
#             "mean": float(np.mean(values)),
#             "std": float(np.std(values)),
#             "min": float(np.min(values)),
#             "max": float(np.max(values)),
#         })
#     summary_df = pd.DataFrame(summary_rows)
#     summary_csv = os.path.join(OUT, "acorn_seed_summary_stats.csv")
#     summary_df.to_csv(summary_csv, index=False)
#     print("\n" + "#" * 120)
#     print("FINAL PER-SEED RESULTS")
#     print("#" * 120)
#     print(results_df.to_string(index=False))
#     print("\n" + "#" * 120)
#     print("FINAL SUMMARY STATS")
#     print("#" * 120)
#     print(summary_df.to_string(index=False))
#     print("\n" + "#" * 120)
#     print("TOP-10 NEURONS ACROSS SEEDS")
#     print("#" * 120)
#     print(top_df.to_string(index=False))
#     print("\nSaved files:")
#     print(results_csv)
#     print(top_csv)
#     print(summary_csv)
#     print("\nDONE.")
#     print("Output directory:", OUT)

# if __name__ == "__main__":
#     main()

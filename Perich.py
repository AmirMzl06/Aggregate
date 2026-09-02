## check
import os
import sys
import gc
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from utils.constants import CEBRA_DIR

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
SESSION_NAME = f"{DATASET_NAME}{DAY}"
NPZ_PATH = os.path.join(PERICH_DATA_DIR, f"{SESSION_NAME}.npz")
OUT = f"Perich_{SESSION_NAME}_CEBRA_Time_Jacobian_Check"
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

ADV_EPSILON = 0.5
ADV_ALPHA = ADV_EPSILON / 5.0
ADV_STEPS = 10
ATTACK_NORM = "linf"

ATTR_N_CHUNKS = 16
ATTR_CHUNK_LEN = 128
ATTR_BATCH_SIZE = 16

DECODER_HIDDEN = 128
DECODER_STEPS = 2500

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

def load_perich_session():
    print("\n" + "=" * 100)
    print("LOADING PERICH")
    print("=" * 100)
    print("Session:", SESSION_NAME)
    print("File   :", NPZ_PATH)
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"Could not find:\n{NPZ_PATH}")
    loaded = np.load(NPZ_PATH, allow_pickle=True)
    print("\nKeys:")
    print(loaded.files)
    required = ["train_data", "valid_data", "train_label", "valid_label"]
    for key in required:
        if key not in loaded.files:
            raise RuntimeError(f"Missing key '{key}'. Available={loaded.files}")
    X_train = loaded["train_data"].astype(np.float32, copy=False)
    X_test = loaded["valid_data"].astype(np.float32, copy=False)
    labels_train = loaded["train_label"].astype(np.float32, copy=False)
    labels_test = loaded["valid_label"].astype(np.float32, copy=False)
    Y_train = labels_train[:, 2:4]
    Y_test = labels_test[:, 2:4]
    if X_train.shape[0] != Y_train.shape[0]:
        raise RuntimeError("X_train / Y_train length mismatch.")
    if X_test.shape[0] != Y_test.shape[0]:
        raise RuntimeError("X_test / Y_test length mismatch.")
    if X_train.shape[1] != X_test.shape[1]:
        raise RuntimeError("Train/test neuron dimension mismatch.")
    if not np.isfinite(X_train).all():
        raise RuntimeError("X_train contains NaN/Inf.")
    if not np.isfinite(X_test).all():
        raise RuntimeError("X_test contains NaN/Inf.")
    if not np.isfinite(Y_train).all():
        raise RuntimeError("Y_train contains NaN/Inf.")
    if not np.isfinite(Y_test).all():
        raise RuntimeError("Y_test contains NaN/Inf.")
    print("\nShapes")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("Y_train:", Y_train.shape)
    print("Y_test :", Y_test.shape)
    print("N neurons:", X_train.shape[1])
    print("\nX_train stats")
    print("min :", float(X_train.min()))
    print("max :", float(X_train.max()))
    print("mean:", float(X_train.mean()))
    print("std :", float(X_train.std()))
    print("\n*** NO NORMALIZATION IN THIS SCRIPT ***")
    print("*** NO EXTRA SMOOTHING IN THIS SCRIPT ***")
    print("*** CEBRA TRAINING IS LABEL-FREE ***")
    print("*** DECODER TARGET = vx, vy ***")
    return X_train, X_test, Y_train, Y_test

def build_model(adversarial=False):
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=TIME_OFFSETS,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=NUM_HIDDEN_UNITS,
        training_mode="adversarial" if adversarial else "clean",
        adv_alpha=ADV_ALPHA if adversarial else 0.0,
        adv_epsilon=ADV_EPSILON if adversarial else 0.0,
        adv_steps=ADV_STEPS if adversarial else 0,
        attack_norm=ATTACK_NORM,
        device=DEVICE,
        verbose=True,
    )

def train_representation(X_train, adversarial=False, label=""):
    seed_all(SEED)
    cleanup()
    model_name = "ACORN" if adversarial else "CEBRA CLEAN"
    print("\n")
    print("#" * 110)
    print(f"TRAINING {label} — {model_name}")
    print("#" * 110)
    print("X_train:", X_train.shape)
    print("N neurons:", X_train.shape[1])
    print("Latent:", LATENT_DIM)
    print("Hidden:", NUM_HIDDEN_UNITS)
    print("Iterations:", MAX_ITER)
    print("Time offsets:", TIME_OFFSETS)
    if adversarial:
        print("epsilon:", ADV_EPSILON)
        print("alpha:", ADV_ALPHA)
        print("steps:", ADV_STEPS)
        print("norm:", ATTACK_NORM)
    model = build_model(adversarial=adversarial)
    model.fit(X_train.astype(np.float32, copy=False))
    return model

class TwoLayerMLP(torch.nn.Module):
    def __init__(self, input_dim=LATENT_DIM, hidden_dim=DECODER_HIDDEN, output_dim=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def build_decoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = TwoLayerMLP(
        input_dim=LATENT_DIM,
        hidden_dim=DECODER_HIDDEN,
        output_dim=2
    ).to(device)
    return decoder, device

def train_decoder(model, X_train, X_test, Y_train, Y_test, condition_name):
    print("\n" + "=" * 100)
    print(f"TWO LAYER MLP DECODER — {condition_name}")
    print("=" * 100)

    Z_train = np.asarray(model.transform(X_train.astype(np.float32, copy=False)), dtype=np.float32)
    Z_test = np.asarray(model.transform(X_test.astype(np.float32, copy=False)), dtype=np.float32)

    print("Embedding train:", Z_train.shape)
    print("Embedding test :", Z_test.shape)

    train_mask = np.isfinite(Z_train).all(axis=1) & np.isfinite(Y_train).all(axis=1)
    test_mask = np.isfinite(Z_test).all(axis=1) & np.isfinite(Y_test).all(axis=1)

    Z_train = Z_train[train_mask]
    y_train = Y_train[train_mask]
    Z_test = Z_test[test_mask]
    y_test = Y_test[test_mask]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xtr = torch.from_numpy(Z_train).float().to(device)
    ytr = torch.from_numpy(y_train).float().to(device)
    xte = torch.from_numpy(Z_test).float().to(device)
    yte = torch.from_numpy(y_test).float().to(device)

    seed_all(SEED)

    decoder = TwoLayerMLP(
        input_dim=LATENT_DIM,
        hidden_dim=DECODER_HIDDEN,
        output_dim=2
    ).to(device)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    print("\nTraining MLP decoder...")
    decoder.train()

    for epoch in range(DECODER_STEPS):
        optimizer.zero_grad()
        pred = decoder(xtr)
        loss = loss_fn(pred, ytr)
        loss.backward()
        optimizer.step()

        if epoch == 0 or (epoch + 1) % 1000 == 0:
            print(f"epoch {epoch+1}/{DECODER_STEPS} loss={loss.item():.6f}")

    decoder.eval()

    with torch.no_grad():
        prediction = decoder(xte).cpu().numpy()

    true_values = yte.cpu().numpy()

    mse = mean_squared_error(true_values, prediction)
    r2_vx = r2_score(true_values[:,0], prediction[:,0])
    r2_vy = r2_score(true_values[:,1], prediction[:,1])
    mean_r2 = (r2_vx + r2_vy) / 2.0

    print("\nRESULT")
    print("MSE:", mse)
    print("R2 vx:", r2_vx)
    print("R2 vy:", r2_vy)
    print("Mean R2:", mean_r2)

    del decoder, xtr, ytr, xte, yte
    cleanup()

    return {
        "condition": condition_name,
        "n_neurons": int(X_train.shape[1]),
        "mse": float(mse),
        "r2_vx": float(r2_vx),
        "r2_vy": float(r2_vy),
        "mean_r2": float(mean_r2),
        "direct_mean_r2": float(mean_r2),
    }

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
        raise RuntimeError(f"Cannot orient JF. shape={a.shape}; latent={latent_dim}; neurons={n_neurons}")
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
    print("\n" + "=" * 100)
    print(f"FORWARD JACOBIAN — {model_name}")
    print("=" * 100)
    net = model.solver_.model
    try:
        device = next(net.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    total_weight = 0
    print("Attribution chunks:", ATTR_N_CHUNKS)
    print("Chunk length:", ATTR_CHUNK_LEN)
    print("Attribution batch:", ATTR_BATCH_SIZE)
    for chunk_index, start in enumerate(starts):
        stop = start + ATTR_CHUNK_LEN
        chunk = X[start:stop].astype(np.float32, copy=True)
        inp = torch.from_numpy(chunk).float().to(device).detach().requires_grad_(True)
        method = cebra.attribution.init(
            name="jacobian-based-batched",
            model=net,
            input_data=inp,
            output_dimension=LATENT_DIM
        )
        with torch.enable_grad():
            result = method.compute_attribution_map(batch_size=ATTR_BATCH_SIZE)
        if "jf" not in result:
            raise RuntimeError(f"No forward Jacobian. Keys={list(result.keys())}")
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
        cleanup()
    jf = (jf_sum / total_weight).astype(np.float32)
    print("\nFINAL JF")
    print("shape:", jf.shape)
    print("meaning:", "latent x neuron")
    print("score:", "mean absolute |dz/dx|")
    return jf

def select_topk(jf, k, selector_name):
    scores = jf.mean(axis=0)
    order = np.argsort(scores)[::-1]
    topk = order[:k]
    print("\n" + "=" * 100)
    print(f"TOP-{k} — {selector_name}")
    print("=" * 100)
    for rank, idx in enumerate(topk, start=1):
        print(f"{rank:2d}. neuron={idx:3d} score={scores[idx]:.12f}")
    return topk.astype(int), scores.astype(np.float32)

def save_forward_plot(clean_jf, acorn_jf):
    vmax = max(float(np.nanmax(clean_jf)), float(np.nanmax(acorn_jf)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(22, 9), constrained_layout=True)
    im = axes[0].imshow(clean_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[0].set_title("CEBRA CLEAN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$")
    axes[0].set_xlabel("Neuron")
    axes[0].set_ylabel("Latent dimension")
    axes[1].imshow(acorn_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    axes[1].set_title("ACORN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$")
    axes[1].set_xlabel("Neuron")
    axes[1].set_ylabel("Latent dimension")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Mean absolute forward Jacobian")
    path = os.path.join(OUT, "JF_CLEAN_vs_ACORN.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved:", path)

def print_results(results):
    print("\n")
    print("=" * 120)
    print("FINAL RESULTS")
    print("=" * 120)
    print(f"{'CONDITION':45s}{'N':>7s}{'MSE':>14s}{'R2 vx':>14s}{'R2 vy':>14s}{'Mean R2':>14s}")
    print("-" * 120)
    for row in results:
        print(f"{row['condition']:45s}{row['n_neurons']:7d}{row['mse']:14.6f}{row['r2_vx']:14.6f}{row['r2_vy']:14.6f}{row['mean_r2']:14.6f}")

def main():
    print("\n" + "=" * 110)
    print("PERICH — FORWARD JACOBIAN TOP-K")
    print("MONKEY GRU DECODER")
    print("=" * 110)
    print("Session:", SESSION_NAME)
    print("Latent:", LATENT_DIM)
    print("CEBRA hidden:", NUM_HIDDEN_UNITS)
    print("CEBRA iterations:", MAX_ITER)
    print("ACORN epsilon:", ADV_EPSILON)
    print("ACORN alpha:", ADV_ALPHA)
    print("ACORN steps:", ADV_STEPS)
    print("Attack norm:", ATTACK_NORM)
    print("Decoder:", "TwoLayerMLP")
    print("Decoder hidden:", DECODER_HIDDEN)
    print("Decoder layers:", DECODER_LAYERS)
    print("Decoder steps:", DECODER_STEPS)
    print("No normalization")
    print("No Jacobian regularizer")
    print("No inverse Jacobian")

    X_train, X_test, Y_train, Y_test = load_perich_session()
    N = X_train.shape[1]
    K = int(np.floor(np.sqrt(N)))
    print("\n" + "=" * 100)
    print("TOP-K")
    print("=" * 100)
    print("N =", N)
    print("K = floor(sqrt(N)) =", K)

    results = []

    clean_model = train_representation(X_train, adversarial=False, label="FULL")
    clean_result = train_decoder(clean_model, X_train, X_test, Y_train, Y_test, "FULL CLEAN")
    results.append(clean_result)
    clean_jf = compute_forward_jacobian(clean_model, X_train, "FULL CLEAN")
    np.save(os.path.join(OUT, "FULL_CLEAN_JF.npy"), clean_jf)

    acorn_model = train_representation(X_train, adversarial=True, label="FULL")
    acorn_result = train_decoder(acorn_model, X_train, X_test, Y_train, Y_test, "FULL ACORN")
    results.append(acorn_result)
    acorn_jf = compute_forward_jacobian(acorn_model, X_train, "FULL ACORN")
    np.save(os.path.join(OUT, "FULL_ACORN_JF.npy"), acorn_jf)

    save_forward_plot(clean_jf, acorn_jf)

    print("\nTOP-K REMOVED: only FULL CLEAN and FULL ACORN are evaluated.")

    print_results(results)
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUT, "Perich_TopK_JF_MonkeyDecoder_Results.csv")
    df.to_csv(csv_path, index=False)

    print("\n")
    print("=" * 90)
    print("MEAN R2 SUMMARY")
    print("=" * 90)
    print(f"{'CHOSEN TOP-K BY':35s}{'RETRAINED MODEL':20s}{'MEAN R2':>12s}")
    print("-" * 70)
    print(f"{'All neurons':35s}{'CEBRA':20s}{results[0]['mean_r2']:12.6f}")
    print(f"{'All neurons':35s}{'ACORN':20s}{results[1]['mean_r2']:12.6f}")


    print("\nSaved CSV:")
    print(csv_path)
    print("\nOutput folder:")
    print(OUT)
    print("\nExperiment conditions:")
    print("1. FULL CLEAN")
    print("2. FULL ACORN")
    print("3. CLEAN_topJF__CEBRA")
    print("4. CLEAN_topJF__ACORN")
    print("5. ACORN_topJF__CEBRA")
    print("6. ACORN_topJF__ACORN")
    print("\nNo JFINV.")
    print("No Jacobian regularization.")
    print("Done.")

if __name__ == "__main__":
    main()

# # import os
# # import sys
# # import gc
# # import random
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import matplotlib.pyplot as plt
# # import pandas as pd
# # from tqdm import tqdm
# # from sklearn.metrics import r2_score
# # from utils.constants import CEBRA_DIR
# # from utils.min_distance import min_l2_distance

# # for module_name in list(sys.modules):
# #     if module_name == "cebra" or module_name.startswith("cebra."):
# #         del sys.modules[module_name]
# # sys.path.insert(0, str(CEBRA_DIR))
# # import cebra
# # import cebra.attribution
# # from cebra import CEBRA
# # print("\nUsing CEBRA:")
# # print(cebra.__file__)

# # DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw"
# # DATASET = "C-CO"
# # DAY = 0
# # NPZ_PATH = os.path.join(DATA_DIR, f"{DATASET}{DAY}.npz")
# # SEED = 42

# # LATENT_DIM = 64
# # HIDDEN = 512
# # BATCH_SIZE = 512
# # MAX_ITER = 5000
# # TEMPERATURE = 0.4
# # TIME_OFFSETS = 4
# # MODEL_ARCH = "offset36-model-more-dropout"

# # ADV_EPS = 0.5
# # ADV_STEPS = 10
# # ADV_NORM = "linf"

# # TOPK_N = None
# # ATTR_CHUNKS = 16
# # ATTR_CHUNK_LEN = 256
# # ATTR_BATCH = 64

# # DECODER_HIDDEN = 128
# # DECODER_EPOCHS = 2000
# # DECODER_LR = 1e-3

# # OUT_DIR = "./outputs_perich_topk"
# # IMG_DIR = "./image_perich_topk"
# # os.makedirs(OUT_DIR, exist_ok=True)
# # os.makedirs(IMG_DIR, exist_ok=True)

# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print("\nDEVICE:", DEVICE)

# # def seed_all(seed):
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(seed)

# # seed_all(SEED)
# # rng = np.random.default_rng(SEED)

# # def cleanup(*objects):
# #     for obj in objects:
# #         try:
# #             del obj
# #         except Exception:
# #             pass
# #     gc.collect()
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()
# #         torch.cuda.ipc_collect()

# # def load_data():
# #     print("\n" + "=" * 90)
# #     print("LOADING PERICH")
# #     print("=" * 90)
# #     print("Path:", NPZ_PATH)
# #     data = np.load(NPZ_PATH, allow_pickle=True)
# #     print("Available keys:", data.files)
# #     X_train = data["train_data"].astype(np.float32)
# #     X_test = data["valid_data"].astype(np.float32)
# #     Y_train = data["train_label"].astype(np.float32)
# #     Y_test = data["valid_label"].astype(np.float32)
# #     print("\nOriginal labels:")
# #     print("Y_train:", Y_train.shape)
# #     print("Y_test :", Y_test.shape)
# #     Y_train = Y_train[:, 2:4]
# #     Y_test = Y_test[:, 2:4]
# #     print("\nFinal data:")
# #     print("X_train:", X_train.shape)
# #     print("X_test :", X_test.shape)
# #     print("Y_train:", Y_train.shape)
# #     print("Y_test :", Y_test.shape)
# #     return X_train, X_test, Y_train, Y_test

# # def diagnostic(X_train, X_test, Y_train, Y_test):
# #     print("\n" + "=" * 90)
# #     print("DATA DIAGNOSTIC")
# #     print("=" * 90)
# #     for name, x in [("X_train", X_train), ("X_test", X_test), ("Y_train", Y_train), ("Y_test", Y_test)]:
# #         print("\n", name)
# #         print("shape:", x.shape)
# #         print("mean :", float(x.mean()))
# #         print("std  :", float(x.std()))
# #         print("min  :", float(x.min()))
# #         print("max  :", float(x.max()))
# #         print("nan  :", int(np.isnan(x).sum()))

# # def train_cebra(X_train, Y_train):
# #     print("\n" + "=" * 90)
# #     print("TRAINING CLEAN CEBRA")
# #     print("=" * 90)
# #     model = CEBRA(
# #         batch_size=BATCH_SIZE,
# #         temperature=TEMPERATURE,
# #         model_architecture=MODEL_ARCH,
# #         time_offsets=TIME_OFFSETS,
# #         max_iterations=MAX_ITER,
# #         output_dimension=LATENT_DIM,
# #         num_hidden_units=HIDDEN,
# #         training_mode="clean",
# #         conditional="time_delta",
# #         device="cuda_if_available",
# #         verbose=True,
# #     )
# #     model.fit(X_train, Y_train)
# #     return model

# # def train_acorn(X_train, Y_train):
# #     print("\n" + "=" * 90)
# #     print("TRAINING ACORN")
# #     print("=" * 90)
# #     eps = ADV_EPS
# #     print("adv epsilon:", eps)
# #     print("adv norm   :", ADV_NORM)
# #     print("adv steps  :", ADV_STEPS)
# #     model = CEBRA(
# #         batch_size=BATCH_SIZE,
# #         temperature=TEMPERATURE,
# #         model_architecture=MODEL_ARCH,
# #         time_offsets=TIME_OFFSETS,
# #         max_iterations=MAX_ITER,
# #         output_dimension=LATENT_DIM,
# #         num_hidden_units=HIDDEN,
# #         training_mode="adversarial",
# #         conditional="time_delta",
# #         adv_alpha=eps / 5.0,
# #         adv_epsilon=eps,
# #         adv_steps=ADV_STEPS,
# #         attack_norm=ADV_NORM,
# #         device="cuda_if_available",
# #         verbose=True,
# #     )
# #     model.fit(X_train, Y_train)
# #     return model

# # def embedding_check(Z_train, Z_test, tag):
# #     print("\n" + "=" * 90)
# #     print(f"EMBEDDING CHECK: {tag}")
# #     print("=" * 90)
# #     print("Z_train:", Z_train.shape, "| mean:", float(Z_train.mean()), "| std:", float(Z_train.std()))
# #     print("Z_test :", Z_test.shape, "| mean:", float(Z_test.mean()), "| std:", float(Z_test.std()))

# # class SimpleGRUDecoder(nn.Module):
# #     def __init__(self, input_dim, hidden_dim=128, output_dim=2):
# #         super().__init__()
# #         self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
# #         self.fc = nn.Linear(hidden_dim, output_dim)
# #     def forward(self, x):
# #         if x.ndim == 2:
# #             x = x.unsqueeze(1)
# #         out, _ = self.gru(x)
# #         out = out[:, -1, :]
# #         return self.fc(out)

# # def train_decoder(X, Y, epochs=DECODER_EPOCHS, tag=""):
# #     model = SimpleGRUDecoder(input_dim=LATENT_DIM, hidden_dim=DECODER_HIDDEN, output_dim=2).to(DEVICE)
# #     X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
# #     Y_t = torch.tensor(Y, dtype=torch.float32, device=DEVICE)
# #     optimizer = torch.optim.Adam(model.parameters(), lr=DECODER_LR)
# #     loss_fn = nn.MSELoss()
# #     model.train()
# #     for epoch in tqdm(range(epochs), desc=f"Decoder {tag}"):
# #         optimizer.zero_grad()
# #         pred = model(X_t)
# #         loss = loss_fn(pred, Y_t)
# #         loss.backward()
# #         optimizer.step()
# #     return model

# # def evaluate(model, X, Y, name):
# #     model.eval()
# #     X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
# #     with torch.no_grad():
# #         pred = model(X_t).cpu().numpy()
# #     scores = []
# #     for i, target_name in enumerate(["vx", "vy"]):
# #         r2 = r2_score(Y[:, i], pred[:, i])
# #         scores.append(float(r2))
# #         print(f"{name} | {target_name} R2: {r2:.6f}")
# #     mean_r2 = float(np.mean(scores))
# #     print(f"{name} | Mean R2: {mean_r2:.6f}")
# #     return mean_r2

# # def save_heatmap(arr, path, title):
# #     arr = np.asarray(arr)
# #     plt.figure(figsize=(12, 6))
# #     plt.imshow(np.abs(arr), aspect="auto", cmap="viridis")
# #     plt.colorbar(label="absolute attribution")
# #     plt.xlabel("Neuron")
# #     plt.ylabel("Latent dimension")
# #     plt.title(title)
# #     plt.tight_layout()
# #     plt.savefig(path, dpi=300, bbox_inches="tight")
# #     plt.close()
# #     print("saved:", path)

# # def orient_attribution(arr, total_neurons):
# #     if torch.is_tensor(arr):
# #         arr = arr.detach().cpu().numpy()
# #     arr = np.abs(np.asarray(arr))
# #     if arr.ndim == 3:
# #         if arr.shape[-1] == total_neurons:
# #             arr = arr.mean(axis=0)
# #         elif arr.shape[1] == total_neurons:
# #             arr = arr.mean(axis=0).T
# #         else:
# #             raise ValueError(f"Cannot find neuron axis in attribution shape {arr.shape}; neurons={total_neurons}")
# #     elif arr.ndim == 2:
# #         if arr.shape[1] == total_neurons:
# #             pass
# #         elif arr.shape[0] == total_neurons:
# #             arr = arr.T
# #         else:
# #             raise ValueError(f"Cannot find neuron axis in shape {arr.shape}; neurons={total_neurons}")
# #     elif arr.ndim == 1:
# #         if arr.shape[0] != total_neurons:
# #             raise ValueError(f"1D attribution has length {arr.shape[0]}, but neurons={total_neurons}")
# #         arr = arr[None, :]
# #     else:
# #         raise ValueError(f"Unsupported attribution shape: {arr.shape}")
# #     return arr.astype(np.float32)

# # def run_attribution(model, X_train, output_dim, tag, total_neurons):
# #     print("\n" + "=" * 90)
# #     print(f"ATTRIBUTION CHUNKED: {tag}")
# #     print("=" * 90)
# #     encoder = model.solver_.model
# #     device = next(encoder.parameters()).device
# #     encoder.eval()
# #     if hasattr(encoder, "split_outputs"):
# #         encoder.split_outputs = False
# #     N = len(X_train)
# #     if N <= ATTR_CHUNK_LEN:
# #         starts = np.array([0], dtype=int)
# #     else:
# #         starts = np.linspace(0, N - ATTR_CHUNK_LEN, ATTR_CHUNKS).astype(int)
# #         starts = np.unique(starts)
# #     jf_list = []
# #     jfinv_list = []
# #     for i, start in enumerate(starts):
# #         print(f"chunk {i + 1}/{len(starts)} | start={start}")
# #         chunk = X_train[start:start + ATTR_CHUNK_LEN]
# #         x_tensor = torch.tensor(chunk, dtype=torch.float32, device=device, requires_grad=True)
# #         method = cebra.attribution.init(
# #             name="jacobian-based-batched",
# #             model=encoder,
# #             input_data=x_tensor,
# #             output_dimension=output_dim
# #         )
# #         result = method.compute_attribution_map(batch_size=min(ATTR_BATCH, len(chunk)))
# #         jf_raw = result["jf"]
# #         if "jf-inv-svd" in result:
# #             jinv_raw = result["jf-inv-svd"]
# #         elif "jf-inv-lsq" in result:
# #             jinv_raw = result["jf-inv-lsq"]
# #         elif "jf-inv" in result:
# #             jinv_raw = result["jf-inv"]
# #         else:
# #             raise KeyError(f"No inverse Jacobian found. Available keys: {list(result.keys())}")
# #         jf = orient_attribution(jf_raw, total_neurons)
# #         jinv = orient_attribution(jinv_raw, total_neurons)
# #         print("JF:", jf.shape, "| JFINV:", jinv.shape)
# #         jf_list.append(jf)
# #         jfinv_list.append(jinv)
# #         del x_tensor, method, result
# #         if torch.cuda.is_available():
# #             torch.cuda.empty_cache()
# #     JF_final = np.mean(np.stack(jf_list), axis=0)
# #     JFINV_final = np.mean(np.stack(jfinv_list), axis=0)
# #     print("\nFINAL JF:", JF_final.shape)
# #     print("FINAL JFINV:", JFINV_final.shape)
# #     np.save(os.path.join(OUT_DIR, f"{tag}_JF.npy"), JF_final)
# #     np.save(os.path.join(OUT_DIR, f"{tag}_JFINV.npy"), JFINV_final)
# #     save_heatmap(JF_final, os.path.join(IMG_DIR, f"{tag}_JF.png"), f"{tag} Forward Jacobian")
# #     save_heatmap(JFINV_final, os.path.join(IMG_DIR, f"{tag}_JFINV.png"), f"{tag} Inverse Jacobian")
# #     return JF_final, JFINV_final

# # def select_topk_neurons(attribution, k=None, tag=""):
# #     scores = np.mean(np.abs(attribution), axis=0)
# #     if k is None:
# #         k = int(np.sqrt(attribution.shape[1]))
# #     idx = np.argsort(scores)[::-1][:k]
# #     print("\n" + "-" * 70)
# #     print("TOP-K SELECTION")
# #     print("source:", tag)
# #     print("K:", k)
# #     print("neurons:", idx.tolist())
# #     print("scores:", scores[idx].tolist())
# #     return idx, scores

# # def train_reduced_experiment(selection_name, selected_neurons, mode, X_train, X_test, Y_train, Y_test):
# #     tag = f"{selection_name}__{mode}"
# #     print("\n" + "#" * 90)
# #     print(f"REDUCED EXPERIMENT: {tag}")
# #     print("Selected neurons:", len(selected_neurons))
# #     print("#" * 90)
# #     Xtr = X_train[:, selected_neurons].copy()
# #     Xte = X_test[:, selected_neurons].copy()
# #     print("Reduced X_train:", Xtr.shape)
# #     print("Reduced X_test :", Xte.shape)
# #     if mode == "CLEAN":
# #         reduced_model = train_cebra(Xtr, Y_train)
# #     elif mode == "ACORN":
# #         reduced_model = train_acorn(Xtr, Y_train)
# #     else:
# #         raise ValueError(f"Unknown mode: {mode}")
# #     Ztr = np.asarray(reduced_model.transform(Xtr), dtype=np.float32)
# #     Zte = np.asarray(reduced_model.transform(Xte), dtype=np.float32)
# #     embedding_check(Ztr, Zte, tag)
# #     decoder = train_decoder(Ztr, Y_train, epochs=DECODER_EPOCHS, tag=tag)
# #     train_r2 = evaluate(decoder, Ztr, Y_train, tag + " TRAIN")
# #     test_r2 = evaluate(decoder, Zte, Y_test, tag + " TEST")
# #     try:
# #         reduced_model.save(os.path.join(OUT_DIR, f"cebra_{tag}"))
# #     except Exception as e:
# #         print("Could not save CEBRA model with .save():", e)
# #     torch.save(decoder.state_dict(), os.path.join(OUT_DIR, f"decoder_{tag}.pt"))
# #     np.save(os.path.join(OUT_DIR, f"{tag}_neurons.npy"), np.asarray(selected_neurons, dtype=np.int64))
# #     cleanup(decoder, reduced_model, Ztr, Zte)
# #     return {"selection": selection_name, "model": mode, "neurons": len(selected_neurons), "train_r2": train_r2, "test_r2": test_r2}

# # def run_full_model(name, train_fn, X_train, X_test, Y_train, Y_test):
# #     print("\n" + "=" * 100)
# #     print(f"FULL MODEL: {name}")
# #     print("=" * 100)
# #     model = train_fn(X_train, Y_train)
# #     Ztr = np.asarray(model.transform(X_train), dtype=np.float32)
# #     Zte = np.asarray(model.transform(X_test), dtype=np.float32)
# #     embedding_check(Ztr, Zte, f"FULL__{name}")
# #     decoder = train_decoder(Ztr, Y_train, epochs=DECODER_EPOCHS, tag=f"FULL__{name}")
# #     train_r2 = evaluate(decoder, Ztr, Y_train, f"FULL__{name} TRAIN")
# #     test_r2 = evaluate(decoder, Zte, Y_test, f"FULL__{name} TEST")
# #     jf, jfinv = run_attribution(model, X_train, LATENT_DIM, f"{name}_FULL", total_neurons=X_train.shape[1])
# #     top_jf, jf_scores = select_topk_neurons(jf, k=TOPK_N, tag=f"{name} JF")
# #     top_jfinv, jfinv_scores = select_topk_neurons(jfinv, k=TOPK_N, tag=f"{name} JFINV")
# #     np.save(os.path.join(OUT_DIR, f"{name}_JF_neuron_scores.npy"), jf_scores)
# #     np.save(os.path.join(OUT_DIR, f"{name}_JFINV_neuron_scores.npy"), jfinv_scores)
# #     try:
# #         model.save(os.path.join(OUT_DIR, f"cebra_FULL_{name}"))
# #     except Exception as e:
# #         print("Could not save full CEBRA model:", e)
# #     torch.save(decoder.state_dict(), os.path.join(OUT_DIR, f"decoder_FULL__{name}.pt"))
# #     full_result = {"setting": "full", "selection": "all", "model": name, "neurons": X_train.shape[1], "train_r2": train_r2, "test_r2": test_r2}
# #     selections = {f"{name}_topJf": top_jf, f"{name}_topJfinv": top_jfinv}
# #     cleanup(decoder, model, Ztr, Zte)
# #     return full_result, selections

# # def main():
# #     print("\n")
# #     print("=" * 100)
# #     print("PERICH TOP-K JACOBIAN PIPELINE")
# #     print("FULL CLEAN + FULL ACORN")
# #     print("JF + JFINV TOP-K")
# #     print("REAL REDUCED CEBRA RETRAINING")
# #     print("=" * 100)
# #     X_train, X_test, Y_train, Y_test = load_data()
# #     diagnostic(X_train, X_test, Y_train, Y_test)
# #     total_neurons = X_train.shape[1]
# #     K = TOPK_N if TOPK_N is not None else int(np.sqrt(total_neurons))
# #     print("\nTOTAL NEURONS:", total_neurons)
# #     print("TOP-K:", K)
# #     clean_full_result, clean_selections = run_full_model(
# #         name="CLEAN",
# #         train_fn=train_cebra,
# #         X_train=X_train,
# #         X_test=X_test,
# #         Y_train=Y_train,
# #         Y_test=Y_test
# #     )
# #     acorn_full_result, acorn_selections = run_full_model(
# #         name="ACORN",
# #         train_fn=train_acorn,
# #         X_train=X_train,
# #         X_test=X_test,
# #         Y_train=Y_train,
# #         Y_test=Y_test
# #     )
# #     topk_sets = {}
# #     topk_sets.update(clean_selections)
# #     topk_sets.update(acorn_selections)
# #     reduced_results = []
# #     for selection_name, neurons in topk_sets.items():
# #         result = train_reduced_experiment(
# #             selection_name=selection_name,
# #             selected_neurons=neurons,
# #             mode="CLEAN",
# #             X_train=X_train,
# #             X_test=X_test,
# #             Y_train=Y_train,
# #             Y_test=Y_test
# #         )
# #         reduced_results.append(result)
# #         result = train_reduced_experiment(
# #             selection_name=selection_name,
# #             selected_neurons=neurons,
# #             mode="ACORN",
# #             X_train=X_train,
# #             X_test=X_test,
# #             Y_train=Y_train,
# #             Y_test=Y_test
# #         )
# #         reduced_results.append(result)
# #     print("\n" + "#" * 100)
# #     print("FINAL RESULTS")
# #     print("#" * 100)
# #     all_results = [clean_full_result, acorn_full_result]
# #     all_results.extend(reduced_results)
# #     for r in all_results:
# #         print(f"{r['selection']:25s} | {r['model']:6s} | neurons={r['neurons']:4d} | train R2={r['train_r2']:.4f} | test R2={r['test_r2']:.4f}")
# #     results_df = pd.DataFrame(all_results)
# #     csv_path = os.path.join(OUT_DIR, f"{DATASET}{DAY}_topk_results.csv")
# #     results_df.to_csv(csv_path, index=False)
# #     print("\nSaved results:", csv_path)
# #     selection_rows = []
# #     for selection_name, neurons in topk_sets.items():
# #         for rank, neuron_idx in enumerate(neurons, start=1):
# #             selection_rows.append({"selection": selection_name, "rank": rank, "neuron": int(neuron_idx)})
# #     selection_df = pd.DataFrame(selection_rows)
# #     selection_csv = os.path.join(OUT_DIR, f"{DATASET}{DAY}_topk_neuron_selections.csv")
# #     selection_df.to_csv(selection_csv, index=False)
# #     print("Saved neuron selections:", selection_csv)
# #     print("\nDONE")

# # if __name__ == "__main__":
# #     main()


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

# # ## Decoder + topK
# # # import os
# # # import sys
# # # import gc
# # # import random
# # # import numpy as np
# # # import pandas as pd
# # # import torch
# # # import torch.nn as nn
# # # import matplotlib.pyplot as plt
# # # from sklearn.metrics import r2_score, mean_squared_error
# # # from utils.constants import CEBRA_DIR

# # # for module_name in list(sys.modules):
# # #     if module_name == "cebra" or module_name.startswith("cebra."):
# # #         del sys.modules[module_name]
# # # sys.path.insert(0, str(CEBRA_DIR))
# # # import cebra
# # # import cebra.attribution
# # # from cebra import CEBRA
# # # print("\nUsing CEBRA from:")
# # # print(cebra.__file__)

# # # PERICH_DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw/"
# # # DATASET_NAME = "C-CO"
# # # DAY = 0
# # # SESSION_NAME = f"{DATASET_NAME}{DAY}"
# # # NPZ_PATH = os.path.join(PERICH_DATA_DIR, f"{SESSION_NAME}.npz")
# # # OUT = f"Perich_{SESSION_NAME}_TopK_JF"
# # # os.makedirs(OUT, exist_ok=True)

# # # SEED = 42
# # # LATENT_DIM = 64
# # # NUM_HIDDEN_UNITS = 512
# # # BATCH_SIZE = 512
# # # MAX_ITER = 5000
# # # TEMPERATURE = 0.4
# # # TIME_OFFSETS = 4
# # # MODEL_ARCH = "offset36-model-more-dropout"
# # # DEVICE = "cuda_if_available"
# # # ADV_EPSILON = 0.5
# # # ADV_ALPHA = ADV_EPSILON / 5.0
# # # ADV_STEPS = 10
# # # ATTACK_NORM = "linf"
# # # ATTR_N_CHUNKS = 16
# # # ATTR_CHUNK_LEN = 128
# # # ATTR_BATCH_SIZE = 16
# # # DECODER_HIDDEN = 64
# # # DECODER_DROPOUT = 0.4
# # # DECODER_LR = 1e-3
# # # DECODER_WEIGHT_DECAY = 2e-4
# # # DECODER_EPOCHS = 6000

# # # def seed_all(seed):
# # #     random.seed(seed)
# # #     np.random.seed(seed)
# # #     torch.manual_seed(seed)
# # #     if torch.cuda.is_available():
# # #         torch.cuda.manual_seed_all(seed)
# # # seed_all(SEED)

# # # def cleanup():
# # #     gc.collect()
# # #     if torch.cuda.is_available():
# # #         torch.cuda.empty_cache()

# # # def load_perich_session():
# # #     print("\n" + "=" * 100)
# # #     print("LOADING PERICH")
# # #     print("=" * 100)
# # #     print("Session:", SESSION_NAME)
# # #     print("File   :", NPZ_PATH)
# # #     if not os.path.exists(NPZ_PATH):
# # #         raise FileNotFoundError(f"Could not find:\n{NPZ_PATH}")
# # #     loaded = np.load(NPZ_PATH, allow_pickle=True)
# # #     print("Keys:", loaded.files)
# # #     required = ["train_data", "valid_data", "train_label", "valid_label"]
# # #     for key in required:
# # #         if key not in loaded.files:
# # #             raise RuntimeError(f"Missing key '{key}'. Available={loaded.files}")
# # #     X_train = loaded["train_data"].astype(np.float32, copy=False)
# # #     X_test = loaded["valid_data"].astype(np.float32, copy=False)
# # #     labels_train = loaded["train_label"].astype(np.float32, copy=False)
# # #     labels_test = loaded["valid_label"].astype(np.float32, copy=False)
# # #     Y_train = labels_train[:, 2:4]
# # #     Y_test = labels_test[:, 2:4]
# # #     if X_train.shape[0] != Y_train.shape[0]:
# # #         raise RuntimeError("X_train / Y_train length mismatch.")
# # #     if X_test.shape[0] != Y_test.shape[0]:
# # #         raise RuntimeError("X_test / Y_test length mismatch.")
# # #     if X_train.shape[1] != X_test.shape[1]:
# # #         raise RuntimeError("Train/test neuron dimension mismatch.")
# # #     if not np.isfinite(X_train).all():
# # #         raise RuntimeError("X_train contains NaN/Inf.")
# # #     if not np.isfinite(X_test).all():
# # #         raise RuntimeError("X_test contains NaN/Inf.")
# # #     print("\nShapes")
# # #     print("X_train:", X_train.shape)
# # #     print("X_test :", X_test.shape)
# # #     print("Y_train:", Y_train.shape)
# # #     print("Y_test :", Y_test.shape)
# # #     print("N neurons:", X_train.shape[1])
# # #     print("\n*** NO NORMALIZATION ***")
# # #     print("*** NO EXTRA SMOOTHING ***")
# # #     print("*** CEBRA TRAINING IS LABEL-FREE ***")
# # #     print("*** DECODER TARGET = vx, vy ***")
# # #     return X_train, X_test, Y_train, Y_test

# # # def build_model(adversarial=False):
# # #     return CEBRA(
# # #         batch_size=BATCH_SIZE,
# # #         temperature=TEMPERATURE,
# # #         model_architecture=MODEL_ARCH,
# # #         time_offsets=TIME_OFFSETS,
# # #         max_iterations=MAX_ITER,
# # #         output_dimension=LATENT_DIM,
# # #         num_hidden_units=NUM_HIDDEN_UNITS,
# # #         training_mode="adversarial" if adversarial else "clean",
# # #         adv_alpha=ADV_ALPHA if adversarial else 0.0,
# # #         adv_epsilon=ADV_EPSILON if adversarial else 0.0,
# # #         adv_steps=ADV_STEPS if adversarial else 0,
# # #         attack_norm=ATTACK_NORM,
# # #         device=DEVICE,
# # #         verbose=True,
# # #     )

# # # def train_representation(X_train, adversarial=False, label=""):
# # #     seed_all(SEED)
# # #     cleanup()
# # #     model_name = "ACORN" if adversarial else "CEBRA CLEAN"
# # #     print("\n")
# # #     print("#" * 110)
# # #     print(f"TRAINING {label} — {model_name}")
# # #     print("#" * 110)
# # #     print("X_train:", X_train.shape)
# # #     print("N neurons:", X_train.shape[1])
# # #     print("Latent:", LATENT_DIM)
# # #     print("Hidden:", NUM_HIDDEN_UNITS)
# # #     print("Iterations:", MAX_ITER)
# # #     if adversarial:
# # #         print("epsilon:", ADV_EPSILON)
# # #         print("alpha:", ADV_ALPHA)
# # #         print("steps:", ADV_STEPS)
# # #         print("norm:", ATTACK_NORM)
# # #     model = build_model(adversarial=adversarial)
# # #     model.fit(X_train.astype(np.float32, copy=False))
# # #     return model

# # # class TwoLayerMLP(nn.Module):
# # #     def __init__(self, input_dim=LATENT_DIM, hidden_dim=DECODER_HIDDEN, output_dim=2, dropout_rate=DECODER_DROPOUT):
# # #         super().__init__()
# # #         self.net = nn.Sequential(
# # #             nn.Linear(input_dim, hidden_dim),
# # #             nn.LayerNorm(hidden_dim),
# # #             nn.ReLU(),
# # #             nn.Dropout(dropout_rate),
# # #             nn.Linear(hidden_dim, output_dim)
# # #         )
# # #         self._initialize()
# # #     def _initialize(self):
# # #         for module in self.modules():
# # #             if isinstance(module, nn.Linear):
# # #                 nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
# # #                 if module.bias is not None:
# # #                     nn.init.zeros_(module.bias)
# # #     def forward(self, x):
# # #         return self.net(x)

# # # def train_decoder(model, X_train, X_test, Y_train, Y_test, condition_name):
# # #     print("\n" + "=" * 100)
# # #     print(f"DECODER — {condition_name}")
# # #     print("=" * 100)
# # #     Z_train = model.transform(X_train.astype(np.float32, copy=False))
# # #     Z_test = model.transform(X_test.astype(np.float32, copy=False))
# # #     Z_train = np.asarray(Z_train, dtype=np.float32)
# # #     Z_test = np.asarray(Z_test, dtype=np.float32)
# # #     train_mask = np.isfinite(Z_train).all(axis=1) & np.isfinite(Y_train).all(axis=1)
# # #     test_mask = np.isfinite(Z_test).all(axis=1) & np.isfinite(Y_test).all(axis=1)
# # #     Z_train = Z_train[train_mask]
# # #     y_train = Y_train[train_mask]
# # #     Z_test = Z_test[test_mask]
# # #     y_test = Y_test[test_mask]
# # #     print("Decoder train:", Z_train.shape)
# # #     print("Decoder test :", Z_test.shape)
# # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # #     xtr = torch.from_numpy(Z_train).float().to(device)
# # #     ytr = torch.from_numpy(y_train).float().to(device)
# # #     xte = torch.from_numpy(Z_test).float().to(device)
# # #     seed_all(SEED)
# # #     decoder = TwoLayerMLP(
# # #         input_dim=LATENT_DIM,
# # #         hidden_dim=DECODER_HIDDEN,
# # #         output_dim=2,
# # #         dropout_rate=DECODER_DROPOUT,
# # #     ).to(device)
# # #     optimizer = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR, weight_decay=DECODER_WEIGHT_DECAY)
# # #     criterion = nn.MSELoss()
# # #     decoder.train()
# # #     for epoch in range(DECODER_EPOCHS):
# # #         optimizer.zero_grad()
# # #         pred = decoder(xtr)
# # #         loss = criterion(pred, ytr)
# # #         loss.backward()
# # #         optimizer.step()
# # #         if (epoch + 1) % 1000 == 0 or epoch == 0:
# # #             print(f"epoch {epoch + 1:5d}/{DECODER_EPOCHS} loss={loss.item():.6f}")
# # #     decoder.eval()
# # #     with torch.no_grad():
# # #         prediction = decoder(xte).cpu().numpy()
# # #     mse = mean_squared_error(y_test, prediction)
# # #     r2_vx = r2_score(y_test[:, 0], prediction[:, 0])
# # #     r2_vy = r2_score(y_test[:, 1], prediction[:, 1])
# # #     mean_r2 = (r2_vx + r2_vy) / 2.0
# # #     print("\nRESULT")
# # #     print("MSE    :", mse)
# # #     print("R2 vx  :", r2_vx)
# # #     print("R2 vy  :", r2_vy)
# # #     print("Mean R2:", mean_r2)
# # #     del decoder, xtr, ytr, xte
# # #     cleanup()
# # #     return {"condition": condition_name, "n_neurons": X_train.shape[1], "mse": float(mse), "r2_vx": float(r2_vx), "r2_vy": float(r2_vy), "mean_r2": float(mean_r2)}

# # # def to_numpy(x):
# # #     if isinstance(x, torch.Tensor):
# # #         return x.detach().cpu().numpy()
# # #     return np.asarray(x)

# # # def orient_forward_jacobian(arr, n_neurons, latent_dim):
# # #     a = np.abs(to_numpy(arr))
# # #     a = np.squeeze(a)
# # #     latent_axes = [i for i, size in enumerate(a.shape) if size == latent_dim]
# # #     neuron_axes = [i for i, size in enumerate(a.shape) if size == n_neurons]
# # #     if not latent_axes or not neuron_axes:
# # #         raise RuntimeError(f"Cannot orient JF. shape={a.shape}; latent={latent_dim}; neurons={n_neurons}")
# # #     latent_axis = latent_axes[-1]
# # #     neuron_axis = neuron_axes[-1]
# # #     if latent_axis == neuron_axis:
# # #         raise RuntimeError(f"Ambiguous JF shape: {a.shape}")
# # #     a = np.moveaxis(a, (latent_axis, neuron_axis), (-2, -1))
# # #     if a.ndim > 2:
# # #         a = a.mean(axis=tuple(range(a.ndim - 2)))
# # #     if a.shape == (n_neurons, latent_dim):
# # #         a = a.T
# # #     expected = (latent_dim, n_neurons)
# # #     if a.shape != expected:
# # #         raise RuntimeError(f"JF final shape={a.shape}; expected={expected}")
# # #     return a.astype(np.float32)

# # # def compute_forward_jacobian(model, X, model_name):
# # #     print("\n" + "=" * 100)
# # #     print(f"FORWARD JACOBIAN — {model_name}")
# # #     print("=" * 100)
# # #     net = model.solver_.model
# # #     device = "cuda" if torch.cuda.is_available() else "cpu"
# # #     net = net.to(device)
# # #     net.eval()
# # #     if hasattr(net, "split_outputs"):
# # #         net.split_outputs = False
# # #     n_time = X.shape[0]
# # #     n_neurons = X.shape[1]
# # #     max_start = n_time - ATTR_CHUNK_LEN - 1
# # #     if max_start <= 0:
# # #         raise RuntimeError("Not enough samples for attribution.")
# # #     starts = np.linspace(0, max_start, ATTR_N_CHUNKS, dtype=int)
# # #     jf_sum = np.zeros((LATENT_DIM, n_neurons), dtype=np.float64)
# # #     total_weight = 0
# # #     for chunk_index, start in enumerate(starts):
# # #         stop = start + ATTR_CHUNK_LEN
# # #         chunk = X[start:stop].astype(np.float32, copy=True)
# # #         inp = torch.from_numpy(chunk).to(device)
# # #         inp.requires_grad_(True)
# # #         method = cebra.attribution.init(
# # #             name="jacobian-based-batched",
# # #             model=net,
# # #             input_data=inp,
# # #             output_dimension=LATENT_DIM
# # #         )
# # #         with torch.enable_grad():
# # #             result = method.compute_attribution_map(batch_size=ATTR_BATCH_SIZE)
# # #         if "jf" not in result:
# # #             raise RuntimeError(f"No JF. Keys={list(result.keys())}")
# # #         jf_raw = result["jf"]
# # #         if chunk_index == 0:
# # #             print("Attribution keys:", list(result.keys()))
# # #             print("RAW JF:", to_numpy(jf_raw).shape)
# # #         jf_chunk = orient_forward_jacobian(jf_raw, n_neurons=n_neurons, latent_dim=LATENT_DIM)
# # #         weight = len(chunk)
# # #         jf_sum += jf_chunk * weight
# # #         total_weight += weight
# # #         print(f"chunk {chunk_index + 1:02d}/{ATTR_N_CHUNKS} done")
# # #         del method, result, jf_raw, jf_chunk, inp, chunk
# # #         cleanup()
# # #     jf = (jf_sum / total_weight).astype(np.float32)
# # #     print("Final JF:", jf.shape)
# # #     return jf

# # # def select_topk(jf, k, selector_name):
# # #     scores = jf.mean(axis=0)
# # #     order = np.argsort(scores)[::-1]
# # #     topk = order[:k]
# # #     print("\n" + "=" * 100)
# # #     print(f"TOP-{k} — {selector_name}")
# # #     print("=" * 100)
# # #     for rank, idx in enumerate(topk, start=1):
# # #         print(f"{rank:2d}. neuron={idx:3d} score={scores[idx]:.12f}")
# # #     return topk.astype(int), scores

# # # def save_forward_plot(clean_jf, acorn_jf):
# # #     vmax = max(float(np.nanmax(clean_jf)), float(np.nanmax(acorn_jf)))
# # #     if not np.isfinite(vmax) or vmax <= 0:
# # #         vmax = 1.0
# # #     fig, axes = plt.subplots(1, 2, figsize=(22, 9), constrained_layout=True)
# # #     im = axes[0].imshow(clean_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
# # #     axes[0].set_title("CEBRA CLEAN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$")
# # #     axes[0].set_xlabel("Neuron")
# # #     axes[0].set_ylabel("Latent dimension")
# # #     axes[1].imshow(acorn_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
# # #     axes[1].set_title("ACORN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$")
# # #     axes[1].set_xlabel("Neuron")
# # #     axes[1].set_ylabel("Latent dimension")
# # #     fig.colorbar(im, ax=axes, shrink=0.85, label="Mean absolute forward Jacobian")
# # #     path = os.path.join(OUT, "JF_CLEAN_vs_ACORN.png")
# # #     fig.savefig(path, dpi=300, bbox_inches="tight")
# # #     plt.close(fig)
# # #     print("Saved:", path)

# # # def print_results(results):
# # #     print("\n")
# # #     print("=" * 110)
# # #     print("FINAL RESULTS")
# # #     print("=" * 110)
# # #     print(f"{'CONDITION':45s}{'N':>7s}{'MSE':>14s}{'R2 vx':>14s}{'R2 vy':>14s}{'Mean R2':>14s}")
# # #     print("-" * 110)
# # #     for row in results:
# # #         print(f"{row['condition']:45s}{row['n_neurons']:7d}{row['mse']:14.6f}{row['r2_vx']:14.6f}{row['r2_vy']:14.6f}{row['mean_r2']:14.6f}")

# # # def main():
# # #     print("\n" + "=" * 110)
# # #     print("PERICH — FORWARD JACOBIAN TOP-K")
# # #     print("=" * 110)
# # #     print("Session:", SESSION_NAME)
# # #     print("Latent:", LATENT_DIM)
# # #     print("Hidden:", NUM_HIDDEN_UNITS)
# # #     print("Iterations:", MAX_ITER)
# # #     print("ACORN epsilon:", ADV_EPSILON)
# # #     print("ACORN alpha:", ADV_ALPHA)
# # #     print("Attack:", ATTACK_NORM)
# # #     print("No normalization")
# # #     print("No Jacobian regularizer")
# # #     X_train, X_test, Y_train, Y_test = load_perich_session()
# # #     N = X_train.shape[1]
# # #     K = int(np.floor(np.sqrt(N)))
# # #     print("\n" + "=" * 100)
# # #     print("TOP-K")
# # #     print("=" * 100)
# # #     print("N =", N)
# # #     print("K = floor(sqrt(N)) =", K)
# # #     results = []
# # #     clean_model = train_representation(X_train, adversarial=False, label="FULL")
# # #     clean_result = train_decoder(clean_model, X_train, X_test, Y_train, Y_test, "FULL CLEAN")
# # #     results.append(clean_result)
# # #     clean_jf = compute_forward_jacobian(clean_model, X_train, "FULL CLEAN")
# # #     np.save(os.path.join(OUT, "FULL_CLEAN_JF.npy"), clean_jf)
# # #     acorn_model = train_representation(X_train, adversarial=True, label="FULL")
# # #     acorn_result = train_decoder(acorn_model, X_train, X_test, Y_train, Y_test, "FULL ACORN")
# # #     results.append(acorn_result)
# # #     acorn_jf = compute_forward_jacobian(acorn_model, X_train, "FULL ACORN")
# # #     np.save(os.path.join(OUT, "FULL_ACORN_JF.npy"), acorn_jf)
# # #     save_forward_plot(clean_jf, acorn_jf)
# # #     clean_topk, clean_scores = select_topk(clean_jf, K, "CLEAN Forward Jacobian")
# # #     acorn_topk, acorn_scores = select_topk(acorn_jf, K, "ACORN Forward Jacobian")
# # #     np.save(os.path.join(OUT, "CLEAN_topJF_indices.npy"), clean_topk)
# # #     np.save(os.path.join(OUT, "ACORN_topJF_indices.npy"), acorn_topk)
# # #     del clean_model, acorn_model
# # #     cleanup()
# # #     reduced_sets = {"CLEAN_topJF": clean_topk, "ACORN_topJF": acorn_topk}
# # #     for selector_name, selected in reduced_sets.items():
# # #         print("\n")
# # #         print("=" * 110)
# # #         print(f"REDUCED SET: {selector_name}")
# # #         print("=" * 110)
# # #         print("Selected neurons:", selected.tolist())
# # #         X_train_reduced = X_train[:, selected]
# # #         X_test_reduced = X_test[:, selected]
# # #         print("X_train reduced:", X_train_reduced.shape)
# # #         clean_reduced = train_representation(X_train_reduced, adversarial=False, label=selector_name)
# # #         result = train_decoder(clean_reduced, X_train_reduced, X_test_reduced, Y_train, Y_test, f"{selector_name}__CEBRA")
# # #         results.append(result)
# # #         del clean_reduced
# # #         cleanup()
# # #         acorn_reduced = train_representation(X_train_reduced, adversarial=True, label=selector_name)
# # #         result = train_decoder(acorn_reduced, X_train_reduced, X_test_reduced, Y_train, Y_test, f"{selector_name}__ACORN")
# # #         results.append(result)
# # #         del acorn_reduced
# # #         cleanup()
# # #     print_results(results)
# # #     df = pd.DataFrame(results)
# # #     csv_path = os.path.join(OUT, "Perich_TopK_JF_Results.csv")
# # #     df.to_csv(csv_path, index=False)
# # #     print("\n")
# # #     print("=" * 90)
# # #     print("MEAN R2 SUMMARY")
# # #     print("=" * 90)
# # #     print(f"{'CHOSEN TOP-K BY':35s}{'RETRAINED MODEL':20s}{'MEAN R2':>12s}")
# # #     print("-" * 70)
# # #     print(f"{'All neurons':35s}{'CEBRA':20s}{results[0]['mean_r2']:12.6f}")
# # #     print(f"{'All neurons':35s}{'ACORN':20s}{results[1]['mean_r2']:12.6f}")
# # #     for row in results[2:]:
# # #         condition = row["condition"]
# # #         if "__" in condition:
# # #             selector, retrained = condition.split("__")
# # #         else:
# # #             selector = condition
# # #             retrained = ""
# # #         print(f"{selector:35s}{retrained:20s}{row['mean_r2']:12.6f}")
# # #     print("\nSaved CSV:")
# # #     print(csv_path)
# # #     print("\nOutput:")
# # #     print(OUT)
# # #     print("\nDONE.")

# # # if __name__ == "__main__":
# # #     main()

# # ## CEBRA + jacobian
# # # import os
# # # import sys
# # # import gc
# # # import random
# # # import numpy as np
# # # import torch
# # # import matplotlib.pyplot as plt
# # # from utils.constants import CEBRA_DIR
# # # from utils.min_distance import min_l2_distance

# # # for module_name in list(sys.modules):
# # #     if module_name == "cebra" or module_name.startswith("cebra."):
# # #         del sys.modules[module_name]
# # # sys.path.insert(0, str(CEBRA_DIR))
# # # import cebra
# # # import cebra.attribution
# # # from cebra import CEBRA
# # # print("\nUsing CEBRA from:")
# # # print(cebra.__file__)

# # # PERICH_DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw/"
# # # DATASET_NAME = "C-CO"
# # # DAY = 0
# # # NPZ_PATH = os.path.join(PERICH_DATA_DIR, f"{DATASET_NAME}{DAY}.npz")
# # # OUT = f"Perich_{DATASET_NAME}{DAY}_Jacobian"
# # # os.makedirs(OUT, exist_ok=True)

# # # SEED = 42
# # # LATENT_DIM = 64
# # # NUM_HIDDEN_UNITS = 512
# # # BATCH_SIZE = 512
# # # MAX_ITER = 5000
# # # TEMPERATURE = 0.4
# # # TIME_OFFSETS = 4
# # # MODEL_ARCH = "offset36-model-more-dropout"
# # # DEVICE = "cuda_if_available"
# # # ADV_STEPS = 10
# # # ATTACK_NORM = "linf"
# # # ATTR_N_CHUNKS = 16
# # # ATTR_CHUNK_LEN = 128
# # # ATTR_BATCH_SIZE = 16

# # # def seed_all(seed):
# # #     random.seed(seed)
# # #     np.random.seed(seed)
# # #     torch.manual_seed(seed)
# # #     if torch.cuda.is_available():
# # #         torch.cuda.manual_seed_all(seed)
# # # seed_all(SEED)

# # # def load_perich_session():
# # #     print("\n" + "=" * 80)
# # #     print("LOADING PERICH SESSION")
# # #     print("=" * 80)
# # #     print("Session:", f"{DATASET_NAME}{DAY}")
# # #     print("File   :", NPZ_PATH)
# # #     if not os.path.exists(NPZ_PATH):
# # #         raise FileNotFoundError(f"\nCould not find Perich session:\n{NPZ_PATH}\n")
# # #     loaded = np.load(NPZ_PATH, allow_pickle=True)
# # #     print("\nKeys:")
# # #     print(loaded.files)
# # #     required = ["train_data", "valid_data", "train_label", "valid_label"]
# # #     for key in required:
# # #         if key not in loaded.files:
# # #             raise RuntimeError(f"Missing '{key}' in {NPZ_PATH}. Available={loaded.files}")
# # #     X_train = loaded["train_data"].astype(np.float32, copy=False)
# # #     X_test = loaded["valid_data"].astype(np.float32, copy=False)
# # #     Y_train = loaded["train_label"].astype(np.float32, copy=False)
# # #     Y_test = loaded["valid_label"].astype(np.float32, copy=False)
# # #     print("\nPERICH DATA")
# # #     print("X_train:", X_train.shape)
# # #     print("X_test :", X_test.shape)
# # #     print("Y_train:", Y_train.shape)
# # #     print("Y_test :", Y_test.shape)
# # #     print("\nNumber neurons:", X_train.shape[1])
# # #     print("\nX_train stats")
# # #     print("min :", float(X_train.min()))
# # #     print("max :", float(X_train.max()))
# # #     print("mean:", float(X_train.mean()))
# # #     print("std :", float(X_train.std()))
# # #     if not np.isfinite(X_train).all():
# # #         raise RuntimeError("X_train contains NaN or Inf.")
# # #     if not np.isfinite(X_test).all():
# # #         raise RuntimeError("X_test contains NaN or Inf.")
# # #     if X_train.shape[1] != X_test.shape[1]:
# # #         raise RuntimeError("Different neuron count between train/test.")
# # #     unit_ids = np.arange(X_train.shape[1])
# # #     print("\n*** USING PREPARED PERICH TRAIN/VALID SPLIT ***")
# # #     print("*** NO EXTRA SMOOTHING ***")
# # #     print("*** NO Z-SCORE ***")
# # #     print("*** NO NORMALIZATION ***")
# # #     print("*** LABELS ARE NOT USED FOR CEBRA TRAINING ***")
# # #     return X_train, X_test, Y_train, Y_test, unit_ids

# # # def compute_adv_epsilon(X):
# # #     print("\n" + "=" * 80)
# # #     print("COMPUTING ACORN EPSILON")
# # #     print("=" * 80)
# # #     x_tensor = torch.from_numpy(X).float()
# # #     min_distance = float(min_l2_distance(x_tensor))
# # #     adv_epsilon = min_distance / 2.0
# # #     adv_epsilon = max(adv_epsilon, 1e-6)
# # #     adv_epsilon = 0.5
# # #     adv_alpha = adv_epsilon / 5.0
# # #     print("min L2 distance:", min_distance)
# # #     print("epsilon:", adv_epsilon)
# # #     print("alpha  :", adv_alpha)
# # #     print("steps  :", ADV_STEPS)
# # #     print("norm   :", ATTACK_NORM)
# # #     return adv_epsilon

# # # def build_model(adversarial=False, adv_epsilon=0.0):
# # #     if adversarial:
# # #         adv_alpha = adv_epsilon / 5.0
# # #     else:
# # #         adv_alpha = 0.0
# # #     return CEBRA(
# # #         batch_size=BATCH_SIZE,
# # #         temperature=TEMPERATURE,
# # #         model_architecture=MODEL_ARCH,
# # #         time_offsets=TIME_OFFSETS,
# # #         max_iterations=MAX_ITER,
# # #         output_dimension=LATENT_DIM,
# # #         num_hidden_units=NUM_HIDDEN_UNITS,
# # #         training_mode="adversarial" if adversarial else "clean",
# # #         adv_alpha=adv_alpha if adversarial else 0.0,
# # #         adv_epsilon=adv_epsilon if adversarial else 0.0,
# # #         adv_steps=ADV_STEPS if adversarial else 0,
# # #         attack_norm=ATTACK_NORM,
# # #         device=DEVICE,
# # #         verbose=True,
# # #     )

# # # def to_numpy(x):
# # #     if isinstance(x, torch.Tensor):
# # #         return x.detach().cpu().numpy()
# # #     return np.asarray(x)

# # # def orient_forward_jacobian(arr, n_neurons, latent_dim):
# # #     a = np.abs(to_numpy(arr))
# # #     a = np.squeeze(a)
# # #     latent_axes = [i for i, size in enumerate(a.shape) if size == latent_dim]
# # #     neuron_axes = [i for i, size in enumerate(a.shape) if size == n_neurons]
# # #     if not latent_axes or not neuron_axes:
# # #         raise RuntimeError(f"Cannot orient forward Jacobian. Raw shape={a.shape}; latent={latent_dim}; neurons={n_neurons}")
# # #     latent_axis = latent_axes[-1]
# # #     neuron_axis = neuron_axes[-1]
# # #     if latent_axis == neuron_axis:
# # #         raise RuntimeError(f"Ambiguous JF shape: {a.shape}")
# # #     a = np.moveaxis(a, (latent_axis, neuron_axis), (-2, -1))
# # #     if a.ndim > 2:
# # #         a = a.mean(axis=tuple(range(a.ndim - 2)))
# # #     if a.shape == (n_neurons, latent_dim):
# # #         a = a.T
# # #     expected = (latent_dim, n_neurons)
# # #     if a.shape != expected:
# # #         raise RuntimeError(f"JF final shape={a.shape}; expected={expected}")
# # #     return a.astype(np.float32)

# # # def orient_inverse_jacobian(arr, n_neurons, latent_dim):
# # #     a = np.abs(to_numpy(arr))
# # #     a = np.squeeze(a)
# # #     neuron_axes = [i for i, size in enumerate(a.shape) if size == n_neurons]
# # #     latent_axes = [i for i, size in enumerate(a.shape) if size == latent_dim]
# # #     if not neuron_axes or not latent_axes:
# # #         raise RuntimeError(f"Cannot orient inverse Jacobian. Raw shape={a.shape}; neurons={n_neurons}; latent={latent_dim}")
# # #     neuron_axis = neuron_axes[-1]
# # #     latent_axis = latent_axes[-1]
# # #     if neuron_axis == latent_axis:
# # #         raise RuntimeError(f"Ambiguous JFINV shape: {a.shape}")
# # #     a = np.moveaxis(a, (neuron_axis, latent_axis), (-2, -1))
# # #     if a.ndim > 2:
# # #         a = a.mean(axis=tuple(range(a.ndim - 2)))
# # #     if a.shape == (latent_dim, n_neurons):
# # #         a = a.T
# # #     expected = (n_neurons, latent_dim)
# # #     if a.shape != expected:
# # #         raise RuntimeError(f"JFINV final shape={a.shape}; expected={expected}")
# # #     return a.astype(np.float32)

# # # def get_inverse_raw(result):
# # #     candidates = ["jf-inv-svd", "jf-inv", "jf-inv-lsq"]
# # #     for key in candidates:
# # #         if key in result:
# # #             return result[key], key
# # #     raise RuntimeError(f"No inverse Jacobian found. Available keys={list(result.keys())}")

# # # def compute_jacobians(model, X, model_name):
# # #     print("\n" + "=" * 80)
# # #     print(f"JACOBIAN ATTRIBUTION: {model_name}")
# # #     print("=" * 80)
# # #     net = model.solver_.model
# # #     device = "cuda" if torch.cuda.is_available() else "cpu"
# # #     net = net.to(device)
# # #     net.eval()
# # #     if hasattr(net, "split_outputs"):
# # #         net.split_outputs = False
# # #     n_time = X.shape[0]
# # #     n_neurons = X.shape[1]
# # #     max_start = n_time - ATTR_CHUNK_LEN - 1
# # #     if max_start <= 0:
# # #         raise RuntimeError("Not enough samples for attribution.")
# # #     starts = np.linspace(0, max_start, ATTR_N_CHUNKS, dtype=int)
# # #     jf_sum = np.zeros((LATENT_DIM, n_neurons), dtype=np.float64)
# # #     jfinv_sum = np.zeros((n_neurons, LATENT_DIM), dtype=np.float64)
# # #     total_weight = 0
# # #     print("Attribution chunks:", ATTR_N_CHUNKS)
# # #     print("Chunk length:", ATTR_CHUNK_LEN)
# # #     print("Attribution batch size:", ATTR_BATCH_SIZE)
# # #     inverse_key_used = None
# # #     for chunk_index, start in enumerate(starts):
# # #         stop = start + ATTR_CHUNK_LEN
# # #         chunk = X[start:stop].astype(np.float32, copy=True)
# # #         inp = torch.from_numpy(chunk).to(device)
# # #         inp.requires_grad_(True)
# # #         method = cebra.attribution.init(
# # #             name="jacobian-based-batched",
# # #             model=net,
# # #             input_data=inp,
# # #             output_dimension=LATENT_DIM
# # #         )
# # #         with torch.enable_grad():
# # #             result = method.compute_attribution_map(batch_size=ATTR_BATCH_SIZE)
# # #         if "jf" not in result:
# # #             raise RuntimeError(f"No forward Jacobian in attribution result. Keys={list(result.keys())}")
# # #         jf_raw = result["jf"]
# # #         jf_chunk = orient_forward_jacobian(jf_raw, n_neurons=n_neurons, latent_dim=LATENT_DIM)
# # #         jfinv_raw, inverse_key = get_inverse_raw(result)
# # #         inverse_key_used = inverse_key
# # #         jfinv_chunk = orient_inverse_jacobian(jfinv_raw, n_neurons=n_neurons, latent_dim=LATENT_DIM)
# # #         if chunk_index == 0:
# # #             print("\nAttribution keys:", list(result.keys()))
# # #             print("RAW JF shape:", to_numpy(jf_raw).shape)
# # #             print("Inverse key:", inverse_key)
# # #             print("RAW JFINV shape:", to_numpy(jfinv_raw).shape)
# # #         weight = len(chunk)
# # #         jf_sum += jf_chunk * weight
# # #         jfinv_sum += jfinv_chunk * weight
# # #         total_weight += weight
# # #         print(f"chunk {chunk_index + 1:02d}/{ATTR_N_CHUNKS} done")
# # #         del method, result, jf_raw, jf_chunk, jfinv_raw, jfinv_chunk, inp, chunk
# # #         gc.collect()
# # #         if torch.cuda.is_available():
# # #             torch.cuda.empty_cache()
# # #     jf = (jf_sum / total_weight).astype(np.float32)
# # #     jfinv = (jfinv_sum / total_weight).astype(np.float32)
# # #     print("\nFINAL JF")
# # #     print("shape:", jf.shape)
# # #     print("meaning:", "latent × neuron")
# # #     print("JF = mean absolute |dz/dx|")
# # #     print("\nFINAL JFINV")
# # #     print("shape:", jfinv.shape)
# # #     print("meaning:", "neuron × latent")
# # #     print("inverse method:", inverse_key_used)
# # #     return jf, jfinv

# # # def train_and_attribute(X, adversarial=False, adv_epsilon=0.0):
# # #     model_name = "ACORN" if adversarial else "CEBRA CLEAN"
# # #     print("\n")
# # #     print("#" * 90)
# # #     print(f"TRAINING {model_name}")
# # #     print("#" * 90)
# # #     seed_all(SEED)
# # #     model = build_model(adversarial=adversarial, adv_epsilon=adv_epsilon)
# # #     print("Input shape:", X.shape)
# # #     print("Latent dimension:", LATENT_DIM)
# # #     print("Hidden units:", NUM_HIDDEN_UNITS)
# # #     print("Iterations:", MAX_ITER)
# # #     print("Time offsets:", TIME_OFFSETS)
# # #     if adversarial:
# # #         print("epsilon:", adv_epsilon)
# # #         print("alpha:", adv_epsilon / 5.0)
# # #         print("steps:", ADV_STEPS)
# # #         print("norm:", ATTACK_NORM)
# # #     model.fit(X.astype(np.float32, copy=False))
# # #     jf, jfinv = compute_jacobians(model, X, model_name)
# # #     return jf, jfinv, model

# # # def save_forward_plot(clean_jf, acorn_jf):
# # #     vmax = max(float(np.nanmax(clean_jf)), float(np.nanmax(acorn_jf)))
# # #     if not np.isfinite(vmax) or vmax <= 0:
# # #         vmax = 1.0
# # #     fig, axes = plt.subplots(1, 2, figsize=(27, 10), constrained_layout=True)
# # #     im = axes[0].imshow(clean_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
# # #     axes[0].set_title("CEBRA CLEAN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$", fontsize=17)
# # #     axes[0].set_xlabel("Neuron / input column", fontsize=13)
# # #     axes[0].set_ylabel("Latent dimension", fontsize=13)
# # #     axes[1].imshow(acorn_jf, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
# # #     axes[1].set_title("ACORN\n" r"$\mathrm{Mean}\ |\partial z/\partial x|$", fontsize=17)
# # #     axes[1].set_xlabel("Neuron / input column", fontsize=13)
# # #     axes[1].set_ylabel("Latent dimension", fontsize=13)
# # #     fig.colorbar(im, ax=axes, shrink=0.85, label="Mean absolute forward Jacobian")
# # #     path = os.path.join(OUT, "JF_CLEAN_vs_ACORN.png")
# # #     fig.savefig(path, dpi=300, bbox_inches="tight")
# # #     plt.close(fig)
# # #     print("\nSaved:")
# # #     print(path)

# # # def save_inverse_plot(clean_inv, acorn_inv):
# # #     vmax = max(float(np.nanmax(clean_inv)), float(np.nanmax(acorn_inv)))
# # #     if not np.isfinite(vmax) or vmax <= 0:
# # #         vmax = 1.0
# # #     fig, axes = plt.subplots(1, 2, figsize=(27, 10), constrained_layout=True)
# # #     im = axes[0].imshow(clean_inv, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
# # #     axes[0].set_title("CEBRA CLEAN\n" "Inverse Jacobian", fontsize=17)
# # #     axes[0].set_xlabel("Latent dimension", fontsize=13)
# # #     axes[0].set_ylabel("Neuron / input column", fontsize=13)
# # #     axes[1].imshow(acorn_inv, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
# # #     axes[1].set_title("ACORN\n" "Inverse Jacobian", fontsize=17)
# # #     axes[1].set_xlabel("Latent dimension", fontsize=13)
# # #     axes[1].set_ylabel("Neuron / input column", fontsize=13)
# # #     fig.colorbar(im, ax=axes, shrink=0.85, label="Mean absolute inverse Jacobian")
# # #     path = os.path.join(OUT, "JFINV_CLEAN_vs_ACORN.png")
# # #     fig.savefig(path, dpi=300, bbox_inches="tight")
# # #     plt.close(fig)
# # #     print("\nSaved:")
# # #     print(path)

# # # def print_top_forward_neurons(jf, unit_ids, name, top_k=10):
# # #     scores = jf.mean(axis=0)
# # #     order = np.argsort(scores)[::-1]
# # #     print("\n" + "=" * 80)
# # #     print(f"TOP {top_k} FORWARD NEURONS — {name}")
# # #     print("=" * 80)
# # #     for rank, idx in enumerate(order[:top_k], start=1):
# # #         print(f"{rank:2d}. input={idx:3d}  unit_id={unit_ids[idx]:3d}  score={scores[idx]:.12f}")

# # # def print_top_inverse_neurons(jfinv, unit_ids, name, top_k=10):
# # #     scores = jfinv.mean(axis=1)
# # #     order = np.argsort(scores)[::-1]
# # #     print("\n" + "=" * 80)
# # #     print(f"TOP {top_k} INVERSE NEURONS — {name}")
# # #     print("=" * 80)
# # #     for rank, idx in enumerate(order[:top_k], start=1):
# # #         print(f"{rank:2d}. input={idx:3d}  unit_id={unit_ids[idx]:3d}  score={scores[idx]:.12f}")

# # # def main():
# # #     print("\n" + "=" * 90)
# # #     print("PERICH SINGLE SESSION")
# # #     print("CEBRA CLEAN vs ACORN")
# # #     print("FORWARD + INVERSE JACOBIAN")
# # #     print("=" * 90)
# # #     print("Session:", f"{DATASET_NAME}{DAY}")
# # #     print("NPZ:", NPZ_PATH)
# # #     print("Latent:", LATENT_DIM)
# # #     print("No normalization")
# # #     print("No decoder")
# # #     print("No Jacobian regularizer")
# # #     X_train, X_test, Y_train, Y_test, unit_ids = load_perich_session()
# # #     print("\nTraining only uses:")
# # #     print("X_train:", X_train.shape)
# # #     print("Validation data is NOT used for CEBRA/ACORN training.")
# # #     adv_epsilon = compute_adv_epsilon(X_train)
# # #     clean_jf, clean_inv, clean_model = train_and_attribute(X_train, adversarial=False, adv_epsilon=0.0)
# # #     del clean_model
# # #     gc.collect()
# # #     if torch.cuda.is_available():
# # #         torch.cuda.empty_cache()
# # #     acorn_jf, acorn_inv, acorn_model = train_and_attribute(X_train, adversarial=True, adv_epsilon=adv_epsilon)
# # #     del acorn_model
# # #     gc.collect()
# # #     if torch.cuda.is_available():
# # #         torch.cuda.empty_cache()
# # #     print("\n" + "=" * 90)
# # #     print("FINAL JACOBIANS")
# # #     print("=" * 90)
# # #     print("CLEAN JF:", clean_jf.shape)
# # #     print("ACORN JF:", acorn_jf.shape)
# # #     print("CLEAN JFINV:", clean_inv.shape)
# # #     print("ACORN JFINV:", acorn_inv.shape)
# # #     print("Expected JF:", (LATENT_DIM, len(unit_ids)))
# # #     print("Expected JFINV:", (len(unit_ids), LATENT_DIM))
# # #     print_top_forward_neurons(clean_jf, unit_ids, "CEBRA CLEAN")
# # #     print_top_forward_neurons(acorn_jf, unit_ids, "ACORN")
# # #     print_top_inverse_neurons(clean_inv, unit_ids, "CEBRA CLEAN")
# # #     print_top_inverse_neurons(acorn_inv, unit_ids, "ACORN")
# # #     np.save(os.path.join(OUT, "CLEAN_JF.npy"), clean_jf)
# # #     np.save(os.path.join(OUT, "ACORN_JF.npy"), acorn_jf)
# # #     np.save(os.path.join(OUT, "CLEAN_JFINV.npy"), clean_inv)
# # #     np.save(os.path.join(OUT, "ACORN_JFINV.npy"), acorn_inv)
# # #     save_forward_plot(clean_jf, acorn_jf)
# # #     save_inverse_plot(clean_inv, acorn_inv)
# # #     print("\n" + "=" * 90)
# # #     print("DONE")
# # #     print("=" * 90)
# # #     print("Session:", f"{DATASET_NAME}{DAY}")
# # #     print("Number neurons:", len(unit_ids))
# # #     print("epsilon:", adv_epsilon)
# # #     print("alpha:", adv_epsilon / 5.0)
# # #     print("attack:", ATTACK_NORM)
# # #     print("steps:", ADV_STEPS)
# # #     print("\nOutput folder:")
# # #     print(OUT)
# # #     print("\nSaved:")
# # #     print(os.path.join(OUT, "JF_CLEAN_vs_ACORN.png"))
# # #     print(os.path.join(OUT, "JFINV_CLEAN_vs_ACORN.png"))
# # #     print("\nJF definition:")
# # #     print("dz/dx")
# # #     print("JF rows    = 128 latent dimensions")
# # #     print("JF columns = neurons")
# # #     print("JFINV rows = neurons")
# # #     print("JFINV cols = 128 latent dimensions")
# # #     print("\nNo decoder was trained.")
# # #     print("No normalization was applied.")
# # #     print("No Jacobian regularization was applied.")

# # # if __name__ == "__main__":
# # #     main()

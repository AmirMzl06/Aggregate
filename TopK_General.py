import os
import sys
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utils.constants import CEBRA_DIR
from utils.min_distance import min_l2_distance
 
sys.path.insert(0, str(CEBRA_DIR))
import cebra
from cebra import CEBRA
import cebra.attribution
 
print("\nUsing CEBRA:")
print(cebra.__file__)
 
PERICH_DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw/"
DATASET_NAME = "C-CO"
DAY = 8
SESSION = f"{DATASET_NAME}{DAY}"
NPZ_PATH = os.path.join(PERICH_DATA_DIR, f"{SESSION}.npz")
OUT = f"TopK_Percentage_{SESSION}"
os.makedirs(OUT, exist_ok=True)
 
SEED = 42
LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 3000
TEMPERATURE = 0.4
OFFSET = 1
MODEL_ARCH = "offset36-model-more-dropout"
 
ADV_STEPS = 10
ATTACK_NORM = "linf"
 
ATTR_CHUNKS = 16
ATTR_LEN = 128
ATTR_BATCH = 16
DEVICE = "cuda_if_available"
 
PERCENTAGES = [1, 5, 15, 30, 50, 75, 90]
 
MLP_HIDDEN_DIM = 64
MLP_DROPOUT = 0.4
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_EPOCHS = 2000
MLP_BATCH_SIZE = 256
MLP_PRINT_EVERY = 200
 
 
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
    eps = 0.1
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
 
 
# =====================================================================
# top-K by percentage helpers
# =====================================================================
def top_k_indices(scores, k):
    return np.argsort(scores)[::-1][:k]
 
 
def reduce_neurons(X, idx):
    return X[:, idx].astype(np.float32)
 
 
def percentage_to_k(pct, n_neurons):
    return max(1, round(pct / 100.0 * n_neurons))
 
 
# =====================================================================
# Decoder: TwoLayerMLP (as given), with a manual train loop since it has
# no built-in .fit() the way MonkeyDecoder did.
# =====================================================================
class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=6, dropout_rate=0.4):
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
 
 
def train_mlp_decoder(Z_train, Y_train, tag):
    print("\n" + "=" * 90)
    print("TRAIN MLP DECODER:", tag)
    print("=" * 90)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Zt = torch.tensor(Z_train, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y_train, dtype=torch.float32, device=device)
 
    decoder = TwoLayerMLP(
        input_dim=Z_train.shape[1],
        hidden_dim=MLP_HIDDEN_DIM,
        output_dim=Y_train.shape[1],
        dropout_rate=MLP_DROPOUT,
    ).to(device)
 
    optimizer = torch.optim.Adam(decoder.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
 
    n = len(Zt)
    for epoch in range(MLP_EPOCHS):
        decoder.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for start in range(0, n, MLP_BATCH_SIZE):
            idx = perm[start:start + MLP_BATCH_SIZE]
            optimizer.zero_grad()
            pred = decoder(Zt[idx])
            loss = loss_fn(pred, Yt[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n
        if (epoch + 1) % MLP_PRINT_EVERY == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{MLP_EPOCHS} | train MSE: {epoch_loss:.6f}")
 
    del Zt, Yt, optimizer
    cleanup()
    return decoder
 
 
def evaluate_mlp_decoder(decoder, Z, Y, name):
    device = next(decoder.parameters()).device
    decoder.eval()
    with torch.no_grad():
        Zt = torch.tensor(Z, dtype=torch.float32, device=device)
        pred = decoder(Zt).cpu().numpy()
    r2s = []
    for i in range(Y.shape[1]):
        r2 = float(r2_score(Y[:, i], pred[:, i]))
        r2s.append(r2)
        print(f"{name} dim {i} R2: {r2:.6f}")
    mean_r2 = float(np.mean(r2s))
    print(f"{name} Mean R2: {mean_r2:.6f}")
    return mean_r2
 
 
def run_reduced_case(X_train_red, X_test_red, Y_train, Y_test, tag):
    """Trains ONLY a plain (clean) CEBRA on the reduced neuron set, then a
    fresh TwoLayerMLP decoder on top of it."""
    model = build_clean()
    model.fit(X_train_red.astype(np.float32), Y_train)
    Z_train = np.asarray(model.transform(X_train_red), dtype=np.float32)
    Z_test = np.asarray(model.transform(X_test_red), dtype=np.float32)
 
    decoder = train_mlp_decoder(Z_train, Y_train, tag)
    r2 = evaluate_mlp_decoder(decoder, Z_test, Y_test, tag)
 
    del model, decoder, Z_train, Z_test
    cleanup()
    return r2
 
 
def main():
    print("\n" + "#" * 110)
    print("TOP-K PERCENTAGE SWEEP: CLEAN vs ACORN JACOBIAN RANKING")
    print("#" * 110)
 
    X_train, X_test, Y_train, Y_test = load_perich()
    n_neurons = X_train.shape[1]
    print("\nTotal neurons:", n_neurons)
    print("Target dimensions:", Y_train.shape[1])
 
    eps = compute_adv_epsilon(X_train)
 
    # ---------------- full ACORN: jacobian only, no decoder ----------------
    acorn_model = train_acorn_label(X_train, Y_train, eps)
    acorn_jf = compute_jacobian(acorn_model, X_train, tag="ACORN_LABEL_full")
    del acorn_model
    cleanup()
 
    # ---------------- full CLEAN: jacobian only, no decoder ----------------
    clean_model = train_clean_label(X_train, Y_train)
    clean_jf = compute_jacobian(clean_model, X_train, tag="CLEAN_LABEL_full")
    del clean_model
    cleanup()
 
    # ---------------- per-neuron scores from each model's own JF ----------------
    clean_score = clean_jf.mean(axis=0)  # (n_neurons,)
    acorn_score = acorn_jf.mean(axis=0)  # (n_neurons,)
 
    # ---------------- percentage sweep: retrain ONLY clean CEBRA ----------------
    clean_curve = []   # R2 when retraining clean CEBRA on CLEAN-ranked neurons
    acorn_curve = []   # R2 when retraining clean CEBRA on ACORN-ranked neurons
    k_list = []
 
    for pct in PERCENTAGES:
        k = percentage_to_k(pct, n_neurons)
        k_list.append(k)
        print("\n" + "=" * 110)
        print(f"PERCENTAGE {pct}% -> k={k} neurons (out of {n_neurons})")
        print("=" * 110)
 
        clean_idx = top_k_indices(clean_score, k)
        acorn_idx = top_k_indices(acorn_score, k)
 
        X_train_c = reduce_neurons(X_train, clean_idx)
        X_test_c = reduce_neurons(X_test, clean_idx)
        r2_c = run_reduced_case(X_train_c, X_test_c, Y_train, Y_test, tag=f"CLEANranked_{pct}pct")
        clean_curve.append(r2_c)
 
        X_train_a = reduce_neurons(X_train, acorn_idx)
        X_test_a = reduce_neurons(X_test, acorn_idx)
        r2_a = run_reduced_case(X_train_a, X_test_a, Y_train, Y_test, tag=f"ACORNranked_{pct}pct")
        acorn_curve.append(r2_a)
 
        print(f"\npct={pct}% | k={k} | CLEAN-ranked R2={r2_c:.4f} | ACORN-ranked R2={r2_a:.4f}")
 
    # ---------------- comparison curve plot ----------------
    plt.figure(figsize=(10, 7))
    plt.plot(PERCENTAGES, clean_curve, marker="o", color="blue", label="Neurons ranked by CLEAN Jacobian")
    plt.plot(PERCENTAGES, acorn_curve, marker="o", color="red", label="Neurons ranked by ACORN Jacobian")
    plt.xlabel("Top-% neurons kept")
    plt.ylabel("Decoder Mean R2")
    plt.title("Plain CEBRA retrained on reduced neuron sets\n(neurons selected via CLEAN vs ACORN Jacobian)")
    plt.legend()
    plt.grid(alpha=0.3)
    curve_path = os.path.join(OUT, "R2_vs_percentage_curve.png")
    plt.savefig(curve_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("\nSaved:", curve_path)
 
    # ---------------- summary CSV ----------------
    summary_df = pd.DataFrame({
        "percentage": PERCENTAGES,
        "k_neurons": k_list,
        "r2_clean_ranked": clean_curve,
        "r2_acorn_ranked": acorn_curve,
    })
    csv_path = os.path.join(OUT, "topk_percentage_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)
 
    print("\n" + "=" * 110)
    print("DONE")
    print("=" * 110)
    print("Output directory:", OUT)
    print(summary_df.to_string(index=False))
 
 
if __name__ == "__main__":
    main()

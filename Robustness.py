## CLEAN vs ACORN robustness on noisy test data
## Fixed offset=1
## Fixed epsilon=0.3

import os
import sys
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from utils.constants import CEBRA_DIR

sys.path.insert(0, str(CEBRA_DIR))
import cebra
from cebra import CEBRA

print("\nUsing CEBRA:")
print(cebra.__file__)

PERICH_DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw/"
DATASET_NAME = "C-CO"
DAY = 12
SESSION = f"{DATASET_NAME}{DAY}"
NPZ_PATH = os.path.join(PERICH_DATA_DIR, f"{SESSION}.npz")
OUT = f"Robustness_{SESSION}_CLEAN_vs_ACORN"
os.makedirs(OUT, exist_ok=True)

SEED = 42
LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 3000
TEMPERATURE = 0.4
OFFSET = 1
MODEL_ARCH = "offset36-model-more-dropout"
ADV_EPSILON = 0.3
ADV_STEPS = 10
ATTACK_NORM = "linf"
DEVICE = "cuda_if_available"
NOISE_STD = 0.1
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
    print("\n" + "=" * 100)
    print("LOADING PERICH")
    print("=" * 100)
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
    return (X_train, X_test, Y_train, Y_test)

def add_test_noise(X_test, noise_std):
    rng = np.random.default_rng(SEED + 1)
    noise = rng.normal(loc=0.0, scale=noise_std, size=X_test.shape).astype(np.float32)
    X_test_noisy = (X_test + noise).astype(np.float32)
    return X_test_noisy

def build_clean():
    print("\nBuilding CLEAN")
    print("offset =", OFFSET)
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
        verbose=True,
    )

def build_acorn():
    print("\nBuilding ACORN")
    print("offset =", OFFSET)
    print("eps    =", ADV_EPSILON)
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=OFFSET,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        training_mode="adversarial",
        adv_alpha=ADV_EPSILON / 5.0,
        adv_epsilon=ADV_EPSILON,
        adv_steps=ADV_STEPS,
        attack_norm=ATTACK_NORM,
        device=DEVICE,
        verbose=True,
    )

def train_clean(X_train, Y_train):
    print("\n" + "=" * 110)
    print("TRAINING CLEAN")
    print("=" * 110)
    model = build_clean()
    model.fit(X_train, Y_train)
    return model

def train_acorn(X_train, Y_train):
    print("\n" + "=" * 110)
    print(f"TRAINING ACORN | " f"offset={OFFSET} | " f"eps={ADV_EPSILON:.4f}")
    print("=" * 110)
    model = build_acorn()
    model.fit(X_train, Y_train)
    return model

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
    print("\n" + "=" * 100)
    print("TRAINING MLP DECODER:", tag)
    print("=" * 100)
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
            print(f"  epoch " f"{epoch + 1}/{MLP_EPOCHS}" f" | train MSE: " f"{epoch_loss:.6f}")
    del Zt
    del Yt
    del optimizer
    cleanup()
    return decoder

def evaluate_decoder(decoder, Z, Y, name):
    device = next(decoder.parameters()).device
    decoder.eval()
    with torch.no_grad():
        Zt = torch.tensor(Z, dtype=torch.float32, device=device)
        pred = decoder(Zt).cpu().numpy()
    r2s = []
    for i in range(Y.shape[1]):
        r2 = float(r2_score(Y[:, i], pred[:, i]))
        r2s.append(r2)
        print(f"{name} " f"dim {i} R2: " f"{r2:.6f}")
    mean_r2 = float(np.mean(r2s))
    print(f"{name} " f"Mean R2: " f"{mean_r2:.6f}")
    return mean_r2

def compute_robustness_metrics(clean_r2, noisy_r2):
    absolute_drop = clean_r2 - noisy_r2
    relative_drop = absolute_drop / max(abs(clean_r2), 1e-12)
    retention = noisy_r2 / max(clean_r2, 1e-12)
    return (absolute_drop, relative_drop, retention)

def main():
    print("\n" + "#" * 120)
    print("CLEAN vs ACORN ROBUSTNESS TEST")
    print(f"OFFSET = {OFFSET}")
    print(f"ACORN EPSILON = {ADV_EPSILON:.4f}")
    print(f"NOISE STD = {NOISE_STD:.4f}")
    print("#" * 120)

    (X_train, X_test, Y_train, Y_test) = load_perich()

    print("\n" + "=" * 100)
    print("ADDING NOISE TO TEST DATA ONLY")
    print("=" * 100)
    print("Noise distribution: Gaussian")
    print("Noise std:", NOISE_STD)
    X_test_noisy = add_test_noise(X_test, NOISE_STD)
    print("Clean test shape:", X_test.shape)
    print("Noisy test shape:", X_test_noisy.shape)

    clean_model = train_clean(X_train, Y_train)
    Z_train_clean = np.asarray(clean_model.transform(X_train), dtype=np.float32)
    Z_test_clean = np.asarray(clean_model.transform(X_test), dtype=np.float32)
    Z_test_noisy_clean = np.asarray(clean_model.transform(X_test_noisy), dtype=np.float32)

    clean_decoder = train_mlp_decoder(Z_train_clean, Y_train, tag="CLEAN")

    print("\n" + "=" * 100)
    print("CLEAN MODEL | CLEAN TEST")
    print("=" * 100)
    clean_r2_before = evaluate_decoder(clean_decoder, Z_test_clean, Y_test, name="CLEAN_TEST")

    print("\n" + "=" * 100)
    print("CLEAN MODEL | NOISY TEST")
    print("=" * 100)
    clean_r2_after = evaluate_decoder(clean_decoder, Z_test_noisy_clean, Y_test, name="CLEAN_NOISY_TEST")

    (clean_drop, clean_relative_drop, clean_retention) = compute_robustness_metrics(clean_r2_before, clean_r2_after)

    print("\nCLEAN ROBUSTNESS")
    print("Before noise R2 :", f"{clean_r2_before:.6f}")
    print("After noise R2  :", f"{clean_r2_after:.6f}")
    print("Absolute drop   :", f"{clean_drop:.6f}")
    print("Relative drop   :", f"{clean_relative_drop * 100:.2f}%")
    print("R2 retention    :", f"{clean_retention * 100:.2f}%")

    del clean_model
    del clean_decoder
    del Z_train_clean
    del Z_test_clean
    del Z_test_noisy_clean
    cleanup()

    acorn_model = train_acorn(X_train, Y_train)
    Z_train_acorn = np.asarray(acorn_model.transform(X_train), dtype=np.float32)
    Z_test_clean_acorn = np.asarray(acorn_model.transform(X_test), dtype=np.float32)
    Z_test_noisy_acorn = np.asarray(acorn_model.transform(X_test_noisy), dtype=np.float32)

    acorn_decoder = train_mlp_decoder(Z_train_acorn, Y_train, tag="ACORN")

    print("\n" + "=" * 100)
    print("ACORN MODEL | CLEAN TEST")
    print("=" * 100)
    acorn_r2_before = evaluate_decoder(acorn_decoder, Z_test_clean_acorn, Y_test, name="ACORN_TEST")

    print("\n" + "=" * 100)
    print("ACORN MODEL | NOISY TEST")
    print("=" * 100)
    acorn_r2_after = evaluate_decoder(acorn_decoder, Z_test_noisy_acorn, Y_test, name="ACORN_NOISY_TEST")

    (acorn_drop, acorn_relative_drop, acorn_retention) = compute_robustness_metrics(acorn_r2_before, acorn_r2_after)

    print("\nACORN ROBUSTNESS")
    print("Before noise R2 :", f"{acorn_r2_before:.6f}")
    print("After noise R2  :", f"{acorn_r2_after:.6f}")
    print("Absolute drop   :", f"{acorn_drop:.6f}")
    print("Relative drop   :", f"{acorn_relative_drop * 100:.2f}%")
    print("R2 retention    :", f"{acorn_retention * 100:.2f}%")

    results = pd.DataFrame(
        [
            {
                "model": "CLEAN",
                "offset": OFFSET,
                "epsilon": 0.0,
                "noise_std": NOISE_STD,
                "R2_before_noise": clean_r2_before,
                "R2_after_noise": clean_r2_after,
                "R2_absolute_drop": clean_drop,
                "R2_relative_drop_percent": clean_relative_drop * 100.0,
                "R2_retention_percent": clean_retention * 100.0,
            },
            {
                "model": "ACORN",
                "offset": OFFSET,
                "epsilon": ADV_EPSILON,
                "noise_std": NOISE_STD,
                "R2_before_noise": acorn_r2_before,
                "R2_after_noise": acorn_r2_after,
                "R2_absolute_drop": acorn_drop,
                "R2_relative_drop_percent": acorn_relative_drop * 100.0,
                "R2_retention_percent": acorn_retention * 100.0,
            },
        ]
    )

    csv_path = os.path.join(OUT, "robustness_summary.csv")
    results.to_csv(csv_path, index=False)

    print("\n" + "#" * 120)
    print("FINAL ROBUSTNESS COMPARISON")
    print("#" * 120)
    print(results.to_string(index=False))
    print("\nSaved:")
    print(csv_path)
    print("\n" + "=" * 110)

    if acorn_relative_drop < clean_relative_drop:
        print("RESULT: ACORN is MORE ROBUST " "to this test-time noise.")
    elif acorn_relative_drop > clean_relative_drop:
        print("RESULT: CLEAN is MORE ROBUST " "to this test-time noise.")
    else:
        print("RESULT: CLEAN and ACORN have " "the same relative robustness.")

    print("=" * 110)
    print("\nOutput directory:", OUT)

if __name__ == "__main__":
    main()

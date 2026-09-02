import os
import sys
import gc
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utils.constants import CEBRA_DIR
from utils.gru_decoder_monkey import MonkeyDecoder

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
OUT = f"Teacher_test_{SESSION}"
os.makedirs(OUT, exist_ok=True)

SEED = 42

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_all(SEED)

LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 5000
TEMPERATURE = 0.4
OFFSET = 1
MODEL_ARCH = "offset36-model-more-dropout"

ADV_EPS = 0.5
ADV_STEPS = 10
ATTACK_NORM = "l2"

DECODER_HIDDEN = 512
DECODER_LAYERS = 2
DECODER_DROPOUT = 0.4
DECODER_STEPS = 2500

ATTR_CHUNKS = 16
ATTR_LEN = 128
ATTR_BATCH = 16

def load_perich():
    print("\n" + "=" * 90)
    print("LOADING PERICH")
    print("=" * 90)
    print("File:", NPZ_PATH)
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
    print("test mean:", float(valid_data.mean()), "std:", float(valid_data.std()))
    return (
        train_data.astype(np.float32),
        valid_data.astype(np.float32),
        Y_train.astype(np.float32),
        Y_test.astype(np.float32)
    )

def build_cebra(adversarial=False):
    print("\nBuilding CEBRA")
    print("mode:", "ACORN" if adversarial else "CLEAN")
    if adversarial:
        return CEBRA(
            batch_size=BATCH_SIZE,
            temperature=TEMPERATURE,
            model_architecture=MODEL_ARCH,
            time_offsets=OFFSET,
            max_iterations=MAX_ITER,
            output_dimension=LATENT_DIM,
            num_hidden_units=HIDDEN,
            training_mode="adversarial",
            adv_alpha=ADV_EPS / 5,
            adv_epsilon=ADV_EPS,
            adv_steps=ADV_STEPS,
            attack_norm=ATTACK_NORM,
            device="cuda_if_available",
            verbose=True
        )
    else:
        return CEBRA(
            batch_size=BATCH_SIZE,
            temperature=TEMPERATURE,
            model_architecture=MODEL_ARCH,
            time_offsets=OFFSET,
            max_iterations=MAX_ITER,
            output_dimension=LATENT_DIM,
            num_hidden_units=HIDDEN,
            training_mode="clean",
            device="cuda_if_available",
            verbose=True
        )
def train_cebra(X_train, adversarial=False):
    model = build_cebra(adversarial=adversarial)
    name = "ACORN" if adversarial else "CLEAN"
    print("\n" + "=" * 90)
    print("TRAINING", name)
    print("=" * 90)
    model.fit(X_train.astype(np.float32))
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
        n_train_steps=DECODER_STEPS
    ).to(device)
    print(decoder)
    decoder.fit(x, y)
    return decoder

def evaluate_decoder(decoder, Z_test, Y_test, tag):
    print("\n" + "=" * 90)
    print(tag)
    print("=" * 90)
    device = next(decoder.parameters()).device
    decoder.eval()
    with torch.no_grad():
        pred = decoder(torch.tensor(Z_test, dtype=torch.float32, device=device))
        pred = pred.cpu().numpy()
    r2s = []
    for i in range(Y_test.shape[1]):
        r2 = r2_score(Y_test[:, i], pred[:, i])
        r2s.append(r2)
        print("dim", i, "R2:", r2)
    mean_r2 = float(np.mean(r2s))
    print("Mean R2:", mean_r2)
    return mean_r2
import os, sys, gc, random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utils.constants import CEBRA_DIR
from utils.gru_decoder_monkey import MonkeyDecoder

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
OUT = f"Teacher_test_{SESSION}"
os.makedirs(OUT, exist_ok=True)

SEED = 42
LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 5000
TEMPERATURE = 0.4
OFFSET = 1
MODEL_ARCH = "offset36-model-more-dropout"
ADV_EPS = 0.5
ADV_STEPS = 10
ATTACK_NORM = "l2"
DECODER_HIDDEN = 512
DECODER_LAYERS = 2
DECODER_DROPOUT = 0.4
DECODER_STEPS = 2500
ATTR_CHUNKS = 16
ATTR_LEN = 128
ATTR_BATCH = 16

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_all(SEED)

def load_perich():
    print("\n" + "=" * 90)
    print("LOADING PERICH")
    print("=" * 90)
    print("File:", NPZ_PATH)
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
    print("test mean:", float(valid_data.mean()), "std:", float(valid_data.std()))
    return (train_data.astype(np.float32), valid_data.astype(np.float32), Y_train.astype(np.float32), Y_test.astype(np.float32))

def build_cebra(adversarial=False):
    print("\nBuilding CEBRA")
    print("mode:", "ACORN" if adversarial else "CLEAN")
    if adversarial:
        return CEBRA(
            batch_size=BATCH_SIZE,
            temperature=TEMPERATURE,
            model_architecture=MODEL_ARCH,
            time_offsets=OFFSET,
            max_iterations=MAX_ITER,
            output_dimension=LATENT_DIM,
            num_hidden_units=HIDDEN,
            training_mode="adversarial",
            adv_alpha=ADV_EPS / 5,
            adv_epsilon=ADV_EPS,
            adv_steps=ADV_STEPS,
            attack_norm=ATTACK_NORM,
            device="cuda_if_available",
            verbose=True
        )
    else:
        return CEBRA(
            batch_size=BATCH_SIZE,
            temperature=TEMPERATURE,
            model_architecture=MODEL_ARCH,
            time_offsets=OFFSET,
            max_iterations=MAX_ITER,
            output_dimension=LATENT_DIM,
            num_hidden_units=HIDDEN,
            training_mode="clean",
            device="cuda_if_available",
            verbose=True
        )

def train_cebra(X_train, adversarial=False):
    model = build_cebra(adversarial=adversarial)
    name = "ACORN" if adversarial else "CLEAN"
    print("\n" + "=" * 90)
    print("TRAINING", name)
    print("=" * 90)
    model.fit(X_train.astype(np.float32))
    return model

def train_decoder(Z_train, Y_train, tag):
    print("\n" + "=" * 90)
    print("TRAIN DECODER:", tag)
    print("=" * 90)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(Z_train, dtype=torch.float32, device=device)
    y = torch.tensor(Y_train, dtype=torch.float32, device=device)
    decoder = MonkeyDecoder(
        input_dim=LATENT_DIM,
        hidden_dim=DECODER_HIDDEN,
        output_dim=Y_train.shape[1],
        dropout=DECODER_DROPOUT,
        bidirectional=False,
        layers=DECODER_LAYERS,
        n_train_steps=DECODER_STEPS
    ).to(device)
    print(decoder)
    decoder.fit(x, y)
    return decoder

def evaluate_decoder(decoder, Z_test, Y_test, tag):
    print("\n" + "=" * 90)
    print(tag)
    print("=" * 90)
    device = next(decoder.parameters()).device
    decoder.eval()
    with torch.no_grad():
        pred = decoder(torch.tensor(Z_test, dtype=torch.float32, device=device))
        pred = pred.cpu().numpy()
    r2s = []
    for i in range(Y_test.shape[1]):
        r2 = r2_score(Y_test[:, i], pred[:, i])
        r2s.append(r2)
        print("dim", i, "R2:", r2)
    mean_r2 = float(np.mean(r2s))
    print("Mean R2:", mean_r2)
    return mean_r2

def compute_jacobian(model, X, name):
    print("\n" + "=" * 90)
    print("JACOBIAN:", name)
    print("=" * 90)
    net = model.solver_.model
    device = next(net.parameters()).device
    net.eval()
    n_neurons = X.shape[1]
    starts = np.linspace(0, len(X) - ATTR_LEN - 1, ATTR_CHUNKS, dtype=int)
    jf_sum = np.zeros((LATENT_DIM, n_neurons), dtype=np.float64)
    total = 0
    for i, start in enumerate(starts):
        chunk = X[start:start + ATTR_LEN]
        inp = torch.tensor(chunk, dtype=torch.float32, device=device, requires_grad=True)
        method = cebra.attribution.init(
            name="jacobian-based-batched",
            model=net,
            input_data=inp,
            output_dimension=LATENT_DIM
        )
        with torch.enable_grad():
            result = method.compute_attribution_map(batch_size=ATTR_BATCH)
        jf = result["jf"]
        jf = np.abs(np.asarray(jf))
        jf = np.squeeze(jf)
        if jf.shape != (LATENT_DIM, n_neurons):
            jf = np.mean(jf, axis=tuple(range(jf.ndim - 2)))
        jf_sum += jf * len(chunk)
        total += len(chunk)
        print("chunk", i + 1, "/", ATTR_CHUNKS)
    jf = (jf_sum / total).astype(np.float32)
    print("Final JF:", jf.shape)
    np.save(os.path.join(OUT, f"{name}_JF.npy"), jf)
    plt.figure(figsize=(12, 8))
    plt.imshow(jf, aspect="auto")
    plt.colorbar()
    plt.title(f"{name} Forward Jacobian")
    path = os.path.join(OUT, f"{name}_JF.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved:", path)
    return jf

def main():
    print("\n" + "=" * 100)
    print("PERICH TEACHER SETUP TEST")
    print("=" * 100)
    X_train, X_test, Y_train, Y_test = load_perich()
    clean_model = train_cebra(X_train, adversarial=False)
    Z_train = clean_model.transform(X_train)
    Z_test = clean_model.transform(X_test)
    clean_decoder = train_decoder(Z_train, Y_train, "CLEAN")
    clean_r2 = evaluate_decoder(clean_decoder, Z_test, Y_test, "CLEAN TEST")
    compute_jacobian(clean_model, X_train, "CLEAN")
    torch.save(clean_decoder.state_dict(), os.path.join(OUT, "clean_decoder.pt"))
    del clean_model, clean_decoder
    gc.collect()
    torch.cuda.empty_cache()
    acorn_model = train_cebra(X_train, adversarial=True)
    Z_train = acorn_model.transform(X_train)
    Z_test = acorn_model.transform(X_test)
    acorn_decoder = train_decoder(Z_train, Y_train, "ACORN")
    acorn_r2 = evaluate_decoder(acorn_decoder, Z_test, Y_test, "ACORN TEST")
    compute_jacobian(acorn_model, X_train, "ACORN")
    torch.save(acorn_decoder.state_dict(), os.path.join(OUT, "acorn_decoder.pt"))
    print("\n" + "=" * 100)
    print("FINAL")
    print("=" * 100)
    print("CLEAN R2:", clean_r2)
    print("ACORN R2:", acorn_r2)
    np.save(os.path.join(OUT, "summary.npy"), np.array([clean_r2, acorn_r2]))
    print("\nDONE")
    print("Output:", OUT)

if __name__ == "__main__":
    main()

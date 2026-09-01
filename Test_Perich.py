import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import r2_score
from utils.constants import CEBRA_DIR

for module_name in list(sys.modules):
    if module_name == "cebra" or module_name.startswith("cebra."):
        del sys.modules[module_name]
sys.path.insert(0, str(CEBRA_DIR))
import cebra
from cebra import CEBRA

print("\nUsing CEBRA:")
print(cebra.__file__)

DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw"
DATASET = "C-CO"
DAY = 10
NPZ_PATH = os.path.join(DATA_DIR, f"{DATASET}{DAY}.npz")
SEED = 42
LATENT_DIM = 64
HIDDEN = 512
BATCH_SIZE = 1024 * 2
MAX_ITER = 5000
TEMPERATURE = 0.4
TIME_OFFSETS = 4
MODEL_ARCH = "offset36-model-more-dropout"
DEVICE = "cuda_if_available"

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_all(SEED)

def load_data():
    print("\n")
    print("=" * 90)
    print("LOADING PERICH")
    print("=" * 90)
    print(NPZ_PATH)
    data = np.load(NPZ_PATH, allow_pickle=True)
    print(data.files)
    X_train = data["train_data"].astype(np.float32)
    X_test = data["valid_data"].astype(np.float32)
    Y_train = data["train_label"].astype(np.float32)
    Y_test = data["valid_label"].astype(np.float32)
    print("\nOriginal labels")
    print("train:", Y_train.shape)
    print("test :", Y_test.shape)
    Y_train = Y_train[:, 2:4]
    Y_test = Y_test[:, 2:4]
    print("\nFinal")
    print("X train:", X_train.shape)
    print("X test :", X_test.shape)
    print("Y train:", Y_train.shape)
    print("Y test :", Y_test.shape)
    return X_train, X_test, Y_train, Y_test

def diagnostic(X_train, X_test, Y_train, Y_test):
    print("\n")
    print("=" * 90)
    print("DATA DIAGNOSTIC")
    print("=" * 90)
    for name, x in [("X_train", X_train), ("X_test", X_test), ("Y_train", Y_train), ("Y_test", Y_test)]:
        print("\n", name)
        print("shape:", x.shape)
        print("mean :", float(x.mean()))
        print("std  :", float(x.std()))
        print("min  :", float(x.min()))
        print("max  :", float(x.max()))
        print("nan :", np.isnan(x).sum())

def train_cebra(X_train, Y_train):
    print("\n")
    print("=" * 90)
    print("TRAINING CLEAN CEBRA (SUPERVISED)")
    print("=" * 90)
    model = CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=TIME_OFFSETS,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        training_mode="clean",
        conditional="time_delta",
        device=DEVICE,
        verbose=True
    )
    model.fit(X_train, Y_train)
    return model

def embedding_check(Z_train, Z_test):
    print("\n")
    print("=" * 90)
    print("EMBEDDING")
    print("=" * 90)
    for n, z in [("Z_train", Z_train), ("Z_test", Z_test)]:
        print("\n", n)
        print(z.shape)
        print("mean:", z.mean())
        print("std :", z.std())

class SimpleGRUDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

def train_decoder(model, X, Y, epochs=2000):
    model.train()
    X = torch.tensor(X, dtype=torch.float32, device="cuda")
    Y = torch.tensor(Y, dtype=torch.float32, device="cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for e in tqdm(range(epochs)):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()
        if e % 200 == 0:
            print("epoch", e, "loss", float(loss.detach()))

def evaluate(model, X, Y, name):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32, device="cuda")
        pred = model(X).cpu().numpy()
    print("\n")
    print("=" * 80)
    print(name)
    print("=" * 80)
    scores = []
    for i, n in enumerate(["vx", "vy"]):
        r2 = r2_score(Y[:, i], pred[:, i])
        scores.append(r2)
        print(n, "R2:", r2)
    print("Mean R2:", np.mean(scores))
    print("\nExamples")
    for i in range(5):
        print("true:", Y[i], "pred:", pred[i])

def main():
    print("\n")
    print("=" * 90)
    print("PERICH SUPERVISED CEBRA + GRU")
    print("=" * 90)
    X_train, X_test, Y_train, Y_test = load_data()
    diagnostic(X_train, X_test, Y_train, Y_test)
    model = train_cebra(X_train, Y_train)
    Z_train = model.transform(X_train)
    Z_test = model.transform(X_test)
    embedding_check(Z_train, Z_test)
    decoder = SimpleGRUDecoder(LATENT_DIM, 128, 2).cuda()
    print("\nTRAINING DECODER")
    train_decoder(decoder, Z_train, Y_train, epochs=2000)
    evaluate(decoder, Z_train, Y_train, "TRAIN")
    evaluate(decoder, Z_test, Y_test, "TEST")
    print("\nDONE")

if __name__ == "__main__":
    main()

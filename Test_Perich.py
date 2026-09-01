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
DAY = 0
NPZ_PATH = os.path.join(DATA_DIR, f"{DATASET}{DAY}.npz")
LATENT_DIM = 64
HIDDEN = 512
BATCH_SIZE = 512
MAX_ITER = 5000
TEMPERATURE = 0.4
TIME_OFFSETS = 4
MODEL_ARCH = "offset36-model-more-dropout"
DEVICE = "cuda_if_available"
SEED = 42

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_all(SEED)

def load_data():
    print("\n" + "="*80)
    print("LOADING PERICH")
    print("="*80)
    print(NPZ_PATH)
    data = np.load(NPZ_PATH, allow_pickle=True)
    print(data.files)
    X_train = data["train_data"].astype(np.float32)
    X_test = data["valid_data"].astype(np.float32)
    Y_train = data["train_label"].astype(np.float32)
    Y_test = data["valid_label"].astype(np.float32)
    Y_train = Y_train[:,2:4]
    Y_test = Y_test[:,2:4]
    print("\nShapes")
    print("X train:", X_train.shape)
    print("X test :", X_test.shape)
    print("Y train:", Y_train.shape)
    print("Y test :", Y_test.shape)
    print("\nStats")
    print("X mean:", X_train.mean())
    print("X std :", X_train.std())
    return X_train, X_test, Y_train, Y_test

class SimpleGRUDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

def train_decoder(model, X, Y, epochs=1000, lr=1e-3):
    model.train()
    X = torch.tensor(X, dtype=torch.float32, device="cuda")
    Y = torch.tensor(Y, dtype=torch.float32, device="cuda")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for e in tqdm(range(epochs)):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()
        if e % 100 == 0:
            print("epoch", e, "loss", float(loss))

def evaluate_decoder(model, X, Y):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32, device="cuda")
        pred = model(X).cpu().numpy()
    r2x = r2_score(Y[:,0], pred[:,0])
    r2y = r2_score(Y[:,1], pred[:,1])
    print("\nRESULT")
    print("="*60)
    print("R2 vx:", r2x)
    print("R2 vy:", r2y)
    print("Mean R2:", (r2x+r2y)/2)

def train_cebra(X_train):
    print("\nTRAINING CLEAN CEBRA")
    print("="*80)
    model = CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=TIME_OFFSETS,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        training_mode="clean",
        device=DEVICE,
        verbose=True
    )
    model.fit(X_train)
    return model

def main():
    print("\n")
    print("="*90)
    print("PERICH CLEAN CEBRA + SIMPLE GRU DECODER")
    print("="*90)
    X_train, X_test, Y_train, Y_test = load_data()
    cebra_model = train_cebra(X_train)
    print("\nEmbedding")
    Z_train = cebra_model.transform(X_train)
    Z_test = cebra_model.transform(X_test)
    print("Z train:", Z_train.shape)
    print("Z test:", Z_test.shape)
    decoder = SimpleGRUDecoder(input_dim=LATENT_DIM, hidden_dim=128, output_dim=2).cuda()
    print("\nTraining decoder")
    train_decoder(decoder, Z_train, Y_train, epochs=2000)
    evaluate_decoder(decoder, Z_test, Y_test)
    print("\nDONE")

if __name__ == "__main__":
    main()

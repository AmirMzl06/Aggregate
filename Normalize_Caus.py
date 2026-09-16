import sys
import gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

if 'cebra' in sys.modules:
    del sys.modules['cebra']

from utils.constants import CEBRA_DIR

sys.path.insert(0, str(CEBRA_DIR))
import cebra
from cebra import CEBRA

print('Using:', cebra.__file__)

DATA_DIR = Path('/data/hossein/mm_project/perich_data_valid_final_raw/')
SESSION = 'C-CO12'
EPSILON = 0.2
LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 3000
TEMP = 0.4
ARCH = 'offset36-model-more-dropout'
DECODER_EPOCHS = 2500
DEVICE = 'cuda_if_available'

d = np.load(DATA_DIR / f'{SESSION}.npz', allow_pickle=True)
Xtr_raw = d['train_data'].astype(np.float32)
Xte_raw = d['valid_data'].astype(np.float32)
Ytr = d['train_label'].astype(np.float32)
Yte = d['valid_label'].astype(np.float32)

mean = Xtr_raw.mean(axis=0)
std = Xtr_raw.std(axis=0) + 1e-8
Xtr_norm = (Xtr_raw - mean) / std
Xte_norm = (Xte_raw - mean) / std

def build_acorn():
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMP,
        model_architecture=ARCH,
        time_offsets=1,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        training_mode='adversarial',
        adv_epsilon=EPSILON,
        adv_alpha=EPSILON / 5,
        adv_steps=10,
        attack_norm='l2',
        device=DEVICE,
        verbose=True,
    )

class Decoder(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, out)
        )
    def forward(self, x):
        return self.net(x)

def train_decoder(z, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Decoder(z.shape[1], y.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    z = torch.tensor(z, device=device)
    y = torch.tensor(y, device=device)
    for _ in range(DECODER_EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(z), y)
        loss.backward()
        opt.step()
    return model

def evaluate(model, z, y):
    device = next(model.parameters()).device
    with torch.no_grad():
        pred = model(torch.tensor(z, device=device)).cpu().numpy()
    return float(np.mean([r2_score(y[:, i], pred[:, i]) for i in range(y.shape[1])]))

print('\\n================ RAW ACORN ================')
acorn_raw = build_acorn()
acorn_raw.fit(Xtr_raw, Ytr)
Ztr_raw = acorn_raw.transform(Xtr_raw)
Zte_raw = acorn_raw.transform(Xte_raw)
decoder_raw = train_decoder(Ztr_raw, Ytr)
r2_raw = evaluate(decoder_raw, Zte_raw, Yte)

print('\\n================ NORMALIZED ACORN ================')
acorn_norm = build_acorn()
acorn_norm.fit(Xtr_norm, Ytr)
Ztr_norm = acorn_norm.transform(Xtr_norm)
Zte_norm = acorn_norm.transform(Xte_norm)
decoder_norm = train_decoder(Ztr_norm, Ytr)
r2_norm = evaluate(decoder_norm, Zte_norm, Yte)

print('\\n================ FINAL R2 ================')
print('ACORN RAW        :', r2_raw)
print('ACORN Z-SCORE    :', r2_norm)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

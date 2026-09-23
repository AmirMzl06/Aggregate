# CEBRA time-only + two-label decoder experiment
# Generated runner
# CEBRA receives NO labels. Decoder receives first two behavior labels.

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
import sys
import importlib


LATENT = 48
DECODER_EPOCHS = 10000


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Decoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def mean_r2(y, p):
    return float(np.mean([r2_score(y[:, i], p[:, i]) for i in range(2)]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cebra-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-iterations", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_all(args.seed)

    sys.path.insert(0, args.cebra_dir)
    importlib.invalidate_caches()
    from cebra import CEBRA

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with np.load(args.data) as f:
        x_train = f["train_data"].astype(np.float32)
        y_train = f["train_label"].astype(np.float32)[:, :2]
        x_valid = f["valid_data"].astype(np.float32)
        y_valid = f["valid_label"].astype(np.float32)[:, :2]

    print("train:", x_train.shape, y_train.shape)
    print("valid:", x_valid.shape, y_valid.shape)

    model = CEBRA(
        model_architecture="offset36-model-more-dropout",
        batch_size=2048,
        temperature=0.4,
        time_offsets=4,
        max_iterations=args.max_iterations,
        output_dimension=LATENT,
        num_hidden_units=32,
        conditional="time",
        device=device,
        verbose=True,
    )

    # IMPORTANT: no labels passed here
    model.fit(x_train)

    z_train = np.asarray(model.transform(x_train))
    z_valid = np.asarray(model.transform(x_valid))

    dec = Decoder(z_train.shape[1]).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3, weight_decay=2e-4)

    xt = torch.tensor(z_train).float().to(device)
    yt = torch.tensor(y_train[:len(z_train)]).float().to(device)
    xv = torch.tensor(z_valid).float().to(device)

    best = -999
    best_state = None

    for e in range(DECODER_EPOCHS):
        dec.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(dec(xt), yt)
        loss.backward()
        opt.step()

        if (e + 1) % 100 == 0:
            dec.eval()
            with torch.no_grad():
                pred = dec(xv).cpu().numpy()
            score = mean_r2(y_valid[:len(pred)], pred)
            if score > best:
                best = score
                best_state = {k:v.cpu().clone() for k,v in dec.state_dict().items()}

    dec.load_state_dict(best_state)
    print("FINAL VALID R2:", best)


if __name__ == "__main__":
    main()

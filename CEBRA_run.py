# CEBRA time-only experiment
# CEBRA receives NO behavior labels.
# Decoder receives up to the first two behavior labels.

import argparse
import json
import random
from pathlib import Path
import sys
import importlib

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score


LATENT = 128
DECODER_EPOCHS = 2500


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_labels(labels):
    """Return labels as [time, targets], using at most two targets."""
    labels = np.asarray(labels, dtype=np.float32)

    if labels.ndim == 1:
        labels = labels[:, None]
    elif labels.ndim == 2:
        labels = labels[:, :]
    else:
        raise ValueError(
            f"Labels must be 1D or 2D, got shape {labels.shape}"
        )

    if not np.isfinite(labels).all():
        raise ValueError("Labels contain NaN or Inf")

    return np.ascontiguousarray(labels)


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def calculate_r2(y_true, y_pred):
    """Mean R² over however many target columns exist."""
    per_target = r2_score(
        y_true,
        y_pred,
        multioutput="raw_values",
    )

    per_target = np.atleast_1d(per_target).astype(float)

    return {
        "mean": float(np.mean(per_target)),
        "per_target": per_target.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cebra-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", default="./cebra_C-RT9")
    parser.add_argument("--max-iterations", type=int, default=2500)
    parser.add_argument("--decoder-epochs", type=int, default=DECODER_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive")

    if args.decoder_epochs < 1:
        parser.error("--decoder-epochs must be positive")

    seed_all(args.seed)

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, args.cebra_dir)
    importlib.invalidate_caches()

    from cebra import CEBRA

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device:", device, flush=True)

    with np.load(args.data, allow_pickle=False) as f:
        x_train = np.asarray(
            f["train_data"],
            dtype=np.float32,
        )
        y_train = prepare_labels(f["train_label"])

        x_valid = np.asarray(
            f["valid_data"],
            dtype=np.float32,
        )
        y_valid = prepare_labels(f["valid_label"])

    if x_train.ndim != 2 or x_valid.ndim != 2:
        raise ValueError(
            f"Neural data must be 2D; "
            f"train={x_train.shape}, valid={x_valid.shape}"
        )

    if len(x_train) != len(y_train):
        raise ValueError(
            f"Train length mismatch: {x_train.shape} vs {y_train.shape}"
        )

    if len(x_valid) != len(y_valid):
        raise ValueError(
            f"Validation length mismatch: "
            f"{x_valid.shape} vs {y_valid.shape}"
        )

    if x_train.shape[1] != x_valid.shape[1]:
        raise ValueError(
            f"Neuron count mismatch: "
            f"train={x_train.shape[1]}, valid={x_valid.shape[1]}"
        )

    if y_train.shape[1] != y_valid.shape[1]:
        raise ValueError(
            f"Target count mismatch: "
            f"train={y_train.shape[1]}, valid={y_valid.shape[1]}"
        )

    if not np.isfinite(x_train).all():
        raise ValueError("Train neural data contains NaN or Inf")

    if not np.isfinite(x_valid).all():
        raise ValueError("Validation neural data contains NaN or Inf")

    print("train:", x_train.shape, y_train.shape, flush=True)
    print("valid:", x_valid.shape, y_valid.shape, flush=True)
    print("number of decoder targets:", y_train.shape[1], flush=True)

    model = CEBRA(
        model_architecture="offset36-model-more-dropout",
        batch_size=2048,
        temperature=0.4,
        time_offsets=4,
        max_iterations=args.max_iterations,
        output_dimension=LATENT,
        num_hidden_units=128,
        conditional="time",
        device=device,
        verbose=True,
    )

    # CEBRA receives neural activity only. No behavior labels are passed.
    model.fit(x_train)

    # Save immediately so a later decoder failure does not waste CEBRA training.
    cebra_path = output_dir / "cebra_model.pt"
    model.save(str(cebra_path))

    print(f"CEBRA saved to: {cebra_path}", flush=True)

    z_train = np.asarray(
        model.transform(x_train),
        dtype=np.float32,
    )
    z_valid = np.asarray(
        model.transform(x_valid),
        dtype=np.float32,
    )

    # Align labels in case this CEBRA version shortens transform output.
    y_train = y_train[:len(z_train)]
    y_valid = y_valid[:len(z_valid)]

    if len(z_train) != len(y_train):
        raise ValueError("Could not align train embedding and labels")

    if len(z_valid) != len(y_valid):
        raise ValueError("Could not align validation embedding and labels")

    print("z_train:", z_train.shape, flush=True)
    print("z_valid:", z_valid.shape, flush=True)

    decoder = Decoder(
        input_dim=z_train.shape[1],
        output_dim=y_train.shape[1],
    ).to(device)

    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=1e-3,
        weight_decay=2e-4,
    )

    xt = torch.from_numpy(z_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    xv = torch.from_numpy(z_valid).to(device)

    best_r2 = -float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(args.decoder_epochs):
        decoder.train()
        optimizer.zero_grad(set_to_none=True)

        prediction = decoder(xt)
        loss = nn.functional.mse_loss(prediction, yt)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Nonfinite decoder loss at epoch {epoch + 1}"
            )

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            decoder.eval()

            with torch.no_grad():
                valid_prediction = decoder(xv).cpu().numpy()

            result = calculate_r2(
                y_valid,
                valid_prediction,
            )

            print(
                f"decoder epoch {epoch + 1:5d} | "
                f"train MSE {loss.item():.6f} | "
                f"valid R2 {result['mean']:.6f}",
                flush=True,
            )

            if result["mean"] > best_r2:
                best_r2 = result["mean"]
                best_epoch = epoch + 1

                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in decoder.state_dict().items()
                }

    if best_state is None:
        raise RuntimeError("No valid decoder checkpoint was selected")

    decoder.load_state_dict(best_state)
    decoder.eval()

    with torch.no_grad():
        final_prediction = decoder(xv).cpu().numpy()

    final_result = calculate_r2(
        y_valid,
        final_prediction,
    )

    torch.save(
        {
            "state_dict": best_state,
            "input_dimension": int(z_train.shape[1]),
            "output_dimension": int(y_train.shape[1]),
            "best_epoch": best_epoch,
            "seed": args.seed,
        },
        output_dir / "decoder_best.pt",
    )

    np.savez_compressed(
        output_dir / "valid_predictions.npz",
        truth=y_valid,
        prediction=final_prediction,
        embedding=z_valid,
    )

    report = {
        "data": str(Path(args.data).resolve()),
        "cebra_model": str(cebra_path.resolve()),
        "seed": args.seed,
        "max_iterations": args.max_iterations,
        "decoder_epochs": args.decoder_epochs,
        "best_decoder_epoch": best_epoch,
        "targets": int(y_train.shape[1]),
        "valid_r2": final_result,
    }

    with (output_dir / "results.json").open("w") as f:
        json.dump(report, f, indent=2)

    print()
    print("FINAL VALID R2:", final_result["mean"])
    print("PER-TARGET R2:", final_result["per_target"])
    print("BEST DECODER EPOCH:", best_epoch)
    print("Results saved to:", output_dir / "results.json")


if __name__ == "__main__":
    main()

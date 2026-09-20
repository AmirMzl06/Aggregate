"""Train the four-bin POSITION puzzle, then a separate behavioral decoder.

Place next to puzzle_positions.py. Run:
    python -u puzzle_positions_test.py --epochs 1000
    python -u puzzle_positions_test.py --variant random

Encoder fit uses only NPZ train_data. Decoder uses train embeddings/train_label.
Both puzzle and decoder validation use valid_data/valid_label, with NO new 80/20
split. These are validation results, not an untouched test set. The encoder is
frozen for decoding. PCA is fit on training embeddings only.

The new head predicts four 4-class position distributions. CE reference=ln(4),
not ln(24). Both exact puzzle accuracy (chance 4.17%) and per-bin position
accuracy (chance 25%) use a one-to-one assignment. No tiles/gaps/selected masks.
Only four raw time bins are used per encoder input. Default pad=False aligns
natural windows [t,t+1,t+2,t+3] to behavior label t+2.

Effective model configuration is printed and saved before fit, including the
actual epoch count after command-line overrides. CLI --epochs overrides the
PUZZLE_EPOCHS constant. Every run starts from scratch; --variant random uses
zero puzzle epochs and otherwise the same encoder/decoder settings.

Dependencies: torch, numpy, scikit-learn, matplotlib, local puzzle_positions.py.
"""
from datetime import datetime, timezone
from pathlib import Path
import csv
import gc
import inspect
import json
import random
import time

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")  # Cluster/headless rendering.
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from puzzle_positions import puzzle, PERMUTATIONS


ROOT = Path(__file__).resolve().parent
SESSION = "C-CO16"
PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
NPZ_PATH = PERICH_DATA_DIR / f"{SESSION}.npz"
OUT_ROOT = ROOT / f"PUZZLE_POSITIONS_{SESSION}_RESULTS"
SEED = 42

# The new architecture requires exactly four time bins per window.
WINDOW_SIZE = 4
LATENT_DIM = 64
ENCODER_HIDDEN = 64
HEAD_HIDDEN = 128
PUZZLE_BATCH_SIZE = 2048
PUZZLE_EPOCHS = 1000
PUZZLE_LR = 3e-4
PUZZLE_DROPOUT = 0.1
PERMUTATIONS_PER_WINDOW = 1  # Fresh labels every visit.
PUZZLE_EVAL_WINDOWS = 1024
PUZZLE_EVAL_BATCH_SIZE = 1024  # SHUFFLED windows per forward pass.
PUZZLE_EVAL_PERMUTATIONS = 24
PUZZLE_EVAL_SEED = SEED + 200000
DEVICE = "cuda_if_available"

# True: one embedding per input bin, with edge padding per split.
# False: only interior windows; labels are aligned using returned time indices.
PAD_TRANSFORM = False
TRANSFORM_BATCH_SIZE = 2048

# Same full-batch MLP layout/hyperparameters as the earlier comparison scripts.
DECODER_EPOCHS = 2500
DECODER_HIDDEN = 64
DECODER_DROPOUT = 0.4
DECODER_LR = 1e-3
PREDICT_BATCH_SIZE = 8192

# Both PCA plots contain train and validation panels in the SAME fitted basis.
PCA_COLOR_LABEL = 0  # Zero-based behavior-label column; no assumed physical meaning.
PLOT_MAX_POINTS = 10000  # Per split, for rendering only; PCA uses ALL train embeddings.
SAVE_EMBEDDINGS = True


def save_json(path, values):
    path.write_text(json.dumps(values, indent=2, allow_nan=False), encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_data(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = [np.asarray(data[key], dtype=np.float32) for key in
                  ("train_data", "valid_data", "train_label", "valid_label")]
    x_train, x_valid, y_train, y_valid = arrays
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_valid.ndim == 1:
        y_valid = y_valid[:, None]
    arrays = (x_train, x_valid, y_train, y_valid)
    for name, x in zip(("train_data", "valid_data", "train_label", "valid_label"), arrays):
        if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
            raise ValueError(f"{name} must be a finite nonempty 2D array; got {x.shape}.")
    if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
        raise ValueError("Feature/label time lengths do not match.")
    if x_train.shape[1] != x_valid.shape[1] or y_train.shape[1] != y_valid.shape[1]:
        raise ValueError("Train/validation channel or label dimensions do not match.")
    if min(len(x_train), len(x_valid)) < WINDOW_SIZE + 1:
        raise ValueError("Each split needs at least WINDOW_SIZE+1 bins for R2 with valid windows.")
    if not PAD_TRANSFORM and len(x_train) - WINDOW_SIZE + 1 < 3:
        raise ValueError("Without padding, training needs at least three valid windows for 3D PCA.")
    if not 0 <= PCA_COLOR_LABEL < y_train.shape[1]:
        raise ValueError(f"PCA_COLOR_LABEL must be in [0, {y_train.shape[1] - 1}].")
    return tuple(np.ascontiguousarray(a) for a in arrays)


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, DECODER_HIDDEN), nn.LayerNorm(DECODER_HIDDEN),
            nn.ReLU(), nn.Dropout(DECODER_DROPOUT),
            nn.Linear(DECODER_HIDDEN, output_dim),
        )

    def forward(self, z):
        return self.net(z)


def train_decoder(z_train, y_train, device):
    seed_all(SEED + 100000)
    model = Decoder(z_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=DECODER_LR)
    z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    losses = []
    model.train()
    for epoch in range(DECODER_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(z), y)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Nonfinite decoder loss at epoch {epoch + 1}.")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if (epoch + 1) % 250 == 0 or epoch + 1 == DECODER_EPOCHS:
            print(f"Decoder {epoch + 1}/{DECODER_EPOCHS}: train MSE={losses[-1]:.6f}", flush=True)
    return model, losses


def predict_decoder(model, z):
    model.eval()  # Dropout must be off for R2 evaluation.
    device = next(model.parameters()).device
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(z), PREDICT_BATCH_SIZE):
            inputs = torch.as_tensor(z[start:start + PREDICT_BATCH_SIZE],
                                     dtype=torch.float32, device=device)
            predictions.append(model(inputs).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def score_r2(y, predictions):
    if y.shape != predictions.shape or len(y) < 2 or not np.isfinite(predictions).all():
        raise ValueError("Invalid predictions for R2.")
    # Explicitly record constant targets; sklearn maps their undefined R2 to 0/1.
    values = np.atleast_1d(r2_score(y, predictions, multioutput="raw_values", force_finite=True))
    constant = np.flatnonzero(np.all(y == y[0], axis=0)).tolist()
    return dict(mean_r2=float(values.mean()), per_output_r2=values.tolist(),
                constant_target_columns=constant)


def plot_r2(scores, out):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    positions = np.arange(len(scores["validation"]["per_output_r2"]))
    for offset, (split, values) in zip((-0.18, 0.18), scores.items()):
        ax.bar(positions + offset, values["per_output_r2"], width=0.36,
               label=f"{split}: mean R2={values['mean_r2']:.4f}")
    ax.set_xticks(positions, [str(i) for i in positions])
    ax.set_xlabel("Behavior label column")
    ax.set_ylabel("Decoder R2")
    ax.set_title(f"{SESSION} | puzzle embeddings, full-dimensional MLP decoder")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend()
    fig.savefig(out / "decoder_R2.png", dpi=250)
    plt.close(fig)


def save_pca_plots(z_train, z_valid, y_train, y_valid, train_indices, valid_indices, out):
    """Train-only PCA fit; validation is transformed with exactly the same basis."""
    pca = PCA(n_components=3, svd_solver="randomized", random_state=SEED)
    pca.fit(z_train)
    points = (pca.transform(z_train), pca.transform(z_valid))
    ratio = pca.explained_variance_ratio_
    if not np.isfinite(ratio).all() or not all(np.isfinite(p).all() for p in points):
        raise ValueError("PCA is nonfinite; check for collapsed or invalid embeddings.")
    np.savez_compressed(out / "pca_coordinates.npz", train=points[0], validation=points[1],
                        components=pca.components_, mean=pca.mean_,
                        explained_variance_ratio=ratio, explained_variance=pca.explained_variance_,
                        train_indices=train_indices, valid_indices=valid_indices)
    colors = (y_train[:, PCA_COLOR_LABEL], y_valid[:, PCA_COLOR_LABEL])
    low = min(float(c.min()) for c in colors)
    high = max(float(c.max()) for c in colors)
    if low == high:
        low, high = low - 0.5, high + 0.5
    norm = Normalize(vmin=low, vmax=high)
    rng = np.random.default_rng(SEED)
    selected = [np.sort(rng.choice(len(p), size=min(len(p), PLOT_MAX_POINTS), replace=False))
                for p in points]
    # Save the plotted subset too, so each figure can be reproduced.
    np.savez_compressed(out / "pca_plot_indices.npz", train=selected[0], validation=selected[1])
    axes_names = [f"PC{i + 1} ({100 * ratio[i]:.1f}% train variance)" for i in range(3)]
    limits = []
    for i in range(3):
        lo = min(float(p[:, i].min()) for p in points)
        hi = max(float(p[:, i].max()) for p in points)
        margin = max((hi - lo) * 0.03, 1e-5)
        limits.append((lo - margin, hi + margin))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, p, c, selected_rows, split in zip(axes, points, colors, selected, ("Train", "Validation")):
        scatter = ax.scatter(p[selected_rows, 0], p[selected_rows, 1], c=c[selected_rows],
                             cmap="viridis", norm=norm, s=5, alpha=0.75, linewidths=0)
        ax.set(xlabel=axes_names[0], ylabel=axes_names[1], xlim=limits[0], ylim=limits[1],
               title=f"{split} | {len(selected_rows):,} plotted / {len(p):,} points")
    fig.colorbar(scatter, ax=list(axes), label=f"Behavior label column {PCA_COLOR_LABEL}", shrink=0.85)
    fig.suptitle(f"{SESSION} | puzzle embeddings: 2D PCA (fit on train only)")
    fig.savefig(out / "embeddings_PCA_2D.png", dpi=250)
    plt.close(fig)
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    axes = [fig.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
    for ax, p, c, selected_rows, split in zip(axes, points, colors, selected, ("Train", "Validation")):
        scatter = ax.scatter(p[selected_rows, 0], p[selected_rows, 1], p[selected_rows, 2],
                             c=c[selected_rows], cmap="viridis", norm=norm, s=5,
                             alpha=0.75, linewidths=0, depthshade=False)
        ax.set(xlabel=axes_names[0], ylabel=axes_names[1], zlabel=axes_names[2],
               xlim=limits[0], ylim=limits[1], zlim=limits[2],
               title=f"{split} | {len(selected_rows):,} plotted / {len(p):,} points")
        ax.view_init(elev=22, azim=45)
    fig.colorbar(scatter, ax=axes, label=f"Behavior label column {PCA_COLOR_LABEL}", shrink=0.65, pad=0.08)
    fig.suptitle(f"{SESSION} | puzzle embeddings: 3D PCA (fit on train only)")
    fig.savefig(out / "embeddings_PCA_3D.png", dpi=250)
    plt.close(fig)
    return dict(fit_split="train", n_fit_samples=len(z_train),
                explained_variance_ratio=ratio.tolist(), color_label_column=PCA_COLOR_LABEL,
                plot_max_points_per_split=PLOT_MAX_POINTS)



def save_reorder_examples(model, x, out):
    """Save actual input/output examples, with no fabricated network predictions."""
    rng = np.random.default_rng(SEED + 300000)
    starts = np.sort(rng.choice(len(x) - 3, size=min(8, len(x) - 3), replace=False))
    original = np.stack([x[t:t + 4].T for t in starts])  # (examples, neurons, 4)
    ids = rng.integers(0, 24, size=len(starts))
    ids[0] = 13  # First example: shuffled=[x2,x0,x3,x1], correct positions=[2,0,3,1].
    targets = PERMUTATIONS[ids]
    shuffled = np.take_along_axis(original, targets[:, None, :], axis=2)
    ordered, positions = model.reorder(shuffled, return_positions=True)
    # The output MUST be an exact column permutation, regardless of accuracy.
    np.testing.assert_array_equal(np.sort(positions, axis=1), np.tile(np.arange(4), (len(starts), 1)))
    expected_from_prediction = np.take_along_axis(shuffled, np.argsort(positions, axis=1)[:, None, :], axis=2)
    np.testing.assert_array_equal(ordered, expected_from_prediction)
    np.savez_compressed(out / "reorder_examples.npz", source_split="valid_data",
                        window_starts=starts, original=original, shuffled=shuffled,
                        predicted_ordered=ordered, true_positions=targets,
                        predicted_positions=positions, permutation_ids=ids)
    print("Reorder example 0, zero-based: true positions=", targets[0].tolist(),
          "predicted=", positions[0].tolist(), flush=True)


def main(variant="positions"):
    if LATENT_DIM < 3 or DECODER_EPOCHS < 1 or PLOT_MAX_POINTS < 1:
        raise ValueError("Need LATENT_DIM>=3, DECODER_EPOCHS>=1 and PLOT_MAX_POINTS>=1.")
    actual = Path(inspect.getfile(puzzle)).resolve()
    if actual != (ROOT / "puzzle_positions.py").resolve():
        raise RuntimeError(f"Expected local {ROOT / 'puzzle_positions.py'}, imported {actual} instead.")
    x_train, x_valid, y_train_full, y_valid_full = load_data(NPZ_PATH)
    seed_all(SEED)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = OUT_ROOT / variant / stamp
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    print("Data:", NPZ_PATH, "| shapes:", x_train.shape, x_valid.shape, flush=True)
    print("Outputs:", out, flush=True)
    model = puzzle(
        window_size=WINDOW_SIZE, output_dimension=LATENT_DIM,
        num_hidden_units=ENCODER_HIDDEN, head_hidden_units=HEAD_HIDDEN,
        batch_size=PUZZLE_BATCH_SIZE, max_epochs=0 if variant == "random" else PUZZLE_EPOCHS,
        learning_rate=PUZZLE_LR, dropout=PUZZLE_DROPOUT, normalize=True,
        permutations_per_window=PERMUTATIONS_PER_WINDOW,
        device=DEVICE, random_state=SEED, verbose=True,
    )
    print("Architecture source:", actual, flush=True)
    print("Effective encoder configuration:\n" + json.dumps(model.get_params(), indent=2), flush=True)
    save_json(out / "run_config.json", dict(
        session=SESSION, variant=variant, data_path=str(NPZ_PATH), puzzle_module=str(actual),
        evaluation=dict(source="valid_data", max_windows=PUZZLE_EVAL_WINDOWS,
                        permutations_per_window=PUZZLE_EVAL_PERMUTATIONS, seed=PUZZLE_EVAL_SEED,
                        uniform_position_ce=float(np.log(4))),
        encoder=model.get_params(), pad_transform=PAD_TRANSFORM,
        label_alignment="transform return_indices, same input time bin",
        decoder=dict(epochs=DECODER_EPOCHS, hidden=DECODER_HIDDEN,
                     dropout=DECODER_DROPOUT, lr=DECODER_LR, seed=SEED + 100000,
                     batch_mode="full", input="original latent, not PCA"),
        pca=dict(n_components=3, fit_split="train", color_label=PCA_COLOR_LABEL),
        data_shapes=dict(train=list(x_train.shape), validation=list(x_valid.shape)),
        versions=dict(torch=str(torch.__version__), numpy=np.__version__),
    ))
    model.fit(x_train)  # No behavior labels or validation inputs enter encoder fit.
    model.save(out / "puzzle_model.pt")
    save_json(out / "puzzle_history.json", model.history_)
    puzzle_scores = {}
    for split, data in (("train", x_train), ("validation", x_valid)):
        metrics, details = model.evaluate_puzzle(
            data, max_windows=PUZZLE_EVAL_WINDOWS, permutations_per_window=PUZZLE_EVAL_PERMUTATIONS,
            batch_size=PUZZLE_EVAL_BATCH_SIZE, random_state=PUZZLE_EVAL_SEED,
            return_details=True)
        metrics["source"] = "train_data" if split == "train" else "valid_data"
        puzzle_scores[split] = metrics
        save_json(out / f"puzzle_accuracy_{split}.json", metrics)
        np.savez_compressed(out / f"puzzle_predictions_{split}.npz", **details)
        print(f"\nPUZZLE {split.upper()} (fixed weights): "
              f"exact accuracy={metrics['exact_accuracy_percent']:.2f}% (chance 4.17%) | "
              f"bin accuracy={metrics['position_accuracy_percent']:.2f}% (chance 25%) | "
              f"position CE={metrics['position_cross_entropy']:.5f} (uniform 1.38629) | "
              f"windows={metrics['n_windows']}", flush=True)
    save_reorder_examples(model, x_valid, out)
    if variant == "random":
        print("Random control: puzzle head is also untrained; judge this control by decoder R2.", flush=True)
    z_train, train_indices = model.transform(x_train, pad=PAD_TRANSFORM,
                                             batch_size=TRANSFORM_BATCH_SIZE, return_indices=True)
    z_valid, valid_indices = model.transform(x_valid, pad=PAD_TRANSFORM,
                                             batch_size=TRANSFORM_BATCH_SIZE, return_indices=True)
    for z, indices in ((z_train, train_indices), (z_valid, valid_indices)):
        if z.shape != (len(indices), LATENT_DIM) or not np.isfinite(z).all():
            raise ValueError("Invalid puzzle embeddings.")
    y_train = y_train_full[train_indices]
    y_valid = y_valid_full[valid_indices]
    if SAVE_EMBEDDINGS:
        np.savez_compressed(out / "embeddings.npz", Z_train=z_train, Z_valid=z_valid,
                            Y_train=y_train, Y_valid=y_valid,
                            train_indices=train_indices, valid_indices=valid_indices)
    model_epochs, model_steps = model.max_epochs, model.n_steps_
    device = model.device_
    encoder_seconds = time.perf_counter() - started
    del model, x_train, x_valid, y_train_full, y_valid_full
    cleanup()

    decoder, losses = train_decoder(z_train, y_train, device)
    train_pred = predict_decoder(decoder, z_train)
    valid_pred = predict_decoder(decoder, z_valid)
    scores = dict(train=score_r2(y_train, train_pred), validation=score_r2(y_valid, valid_pred))
    save_json(out / "R2.json", scores)
    with (out / "R2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "mean_r2"] + [f"r2_label_{i}" for i in range(y_train.shape[1])])
        for split, values in scores.items():
            writer.writerow([split, values["mean_r2"], *values["per_output_r2"]])
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in decoder.state_dict().items()},
                    input_dim=LATENT_DIM, output_dim=y_train.shape[1], hidden=DECODER_HIDDEN,
                    dropout=DECODER_DROPOUT, seed=SEED + 100000), out / "decoder.pt")
    np.save(out / "decoder_train_mse.npy", np.asarray(losses))
    np.savez_compressed(out / "decoder_predictions.npz", train_true=y_train, train_pred=train_pred,
                        valid_true=y_valid, valid_pred=valid_pred,
                        train_indices=train_indices, valid_indices=valid_indices)
    plot_r2(scores, out)
    pca_info = save_pca_plots(z_train, z_valid, y_train, y_valid, train_indices, valid_indices, out)
    save_json(out / "pca_info.json", pca_info)
    save_json(out / "timing.json", dict(encoder_and_transform_seconds=encoder_seconds,
                                        total_seconds=time.perf_counter() - started))
    for split, values in scores.items():
        print(f"\n{split.upper()} mean R2: {values['mean_r2']:.6f}", flush=True)
        print("Per-output R2:", values["per_output_r2"], flush=True)
        if values["constant_target_columns"]:
            print("Constant target columns (sklearn finite R2 convention):",
                  values["constant_target_columns"], flush=True)
    save_json(out / "summary.json", dict(variant=variant, encoder_epochs=model_epochs, optimizer_steps=model_steps,
                                          puzzle=puzzle_scores, decoder=scores))
    print("\nSaved all results:", out, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("positions", "random"), default="positions")
    parser.add_argument("--epochs", type=int, default=PUZZLE_EPOCHS)
    parser.add_argument("--lr", type=float, default=PUZZLE_LR)
    parser.add_argument("--batch-size", type=int, default=PUZZLE_BATCH_SIZE)
    parser.add_argument("--encoder-hidden", type=int, default=ENCODER_HIDDEN)
    parser.add_argument("--head-hidden", type=int, default=HEAD_HIDDEN)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--dropout", type=float, default=PUZZLE_DROPOUT)
    parser.add_argument("--permutations-per-window", type=int, default=PERMUTATIONS_PER_WINDOW)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--dataset", type=Path, default=NPZ_PATH)
    parser.add_argument("--decoder-epochs", type=int, default=DECODER_EPOCHS)
    args = parser.parse_args()
    PUZZLE_EPOCHS, PUZZLE_LR, PUZZLE_BATCH_SIZE = args.epochs, args.lr, args.batch_size
    ENCODER_HIDDEN, HEAD_HIDDEN, LATENT_DIM = args.encoder_hidden, args.head_hidden, args.latent_dim
    PUZZLE_DROPOUT, PERMUTATIONS_PER_WINDOW = args.dropout, args.permutations_per_window
    SEED, DEVICE = args.seed, args.device
    PUZZLE_EVAL_SEED = SEED + 200000
    NPZ_PATH, DECODER_EPOCHS = args.dataset, args.decoder_epochs
    SESSION = NPZ_PATH.stem
    OUT_ROOT = ROOT / f"PUZZLE_POSITIONS_{SESSION}_RESULTS"
    main(args.variant)

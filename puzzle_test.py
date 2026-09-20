"""Train puzzle + a separate behavioral decoder on Perich C-CO16.

Place next to the previously supplied puzzle.py and run:
    python -u train_puzzle_cco16_eval.py

Dependencies: torch, numpy, scikit-learn, matplotlib, and local puzzle.py.
The NPZ must contain train_data, valid_data, train_label, valid_label.
Each feature split is treated as a continuous (time, neurons) recording.
Decoder R2 is measured on valid_data: this is VALIDATION, not held-out test R2.
The encoder sees only train_data; the decoder sees only training embeddings
and training labels. No checkpoint selection or early stopping uses valid_data.
PCA is for visualization only; the decoder uses the full latent dimension.

Final puzzle evaluation uses test_data if present, otherwise valid_data (named
VALIDATION explicitly). No behavioral labels are needed for puzzle evaluation.
Evaluate an existing checkpoint WITHOUT training the encoder or decoder:
    python -u train_puzzle_cco16_eval.py --puzzle-checkpoint /path/to/puzzle_model.pt
Use --eval-split test to require test_data, or --eval-split validation for valid_data.
The checkpoint's window_size and delta always control evaluation sampling.
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from puzzle import puzzle, _draw_plan, _make_puzzles, PERMUTATIONS


ROOT = Path(__file__).resolve().parent
SESSION = "C-CO16"
PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
NPZ_PATH = PERICH_DATA_DIR / f"{SESSION}.npz"
OUT_ROOT = ROOT / f"PUZZLE_{SESSION}_RESULTS"
SEED = 42

# Puzzle: full WINDOW_SIZE-bin windows; four shuffled slots at fixed spacing delta.
WINDOW_SIZE = 4
DELTA = 1
LATENT_DIM = 64
ENCODER_HIDDEN = 64
HEAD_HIDDEN = 128
PUZZLE_BATCH_SIZE = 2048
PUZZLE_EPOCHS = 15000
PUZZLE_LR = 3e-4
PUZZLE_DROPOUT = 0.1
PERMUTATIONS_PER_WINDOW = 1  # 24 processes all permutations; much higher memory/cost.
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

# Final fixed-weight puzzle accuracy. These settings do not affect training.
PUZZLE_EVAL_SPLIT = "auto"  # Prefer test_data; otherwise explicitly report validation.
PUZZLE_EVAL_WINDOWS = 1024  # Random distinct valid window starts; use all if fewer.
PUZZLE_EVAL_PERMUTATIONS = 24  # All classes per sampled quartet, for balanced evaluation.
PUZZLE_EVAL_BATCH_SIZE = 256  # Maximum PUZZLES per forward pass (not base windows).
PUZZLE_EVAL_SEED = SEED + 200000


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


def load_puzzle_eval_data(path, split, window_size, n_features):
    if split not in ("auto", "test", "validation"):
        raise ValueError("eval split must be auto, test, or validation.")
    with np.load(path, allow_pickle=False) as data:
        if split == "auto":
            split = "test" if "test_data" in data.files else "validation"
        key = "test_data" if split == "test" else "valid_data"
        if key not in data.files:
            raise KeyError(f"Requested {split} puzzle evaluation, but {key!r} is missing from {path}.")
        x = np.asarray(data[key], dtype=np.float32)
    if (x.ndim != 2 or x.shape[0] < window_size or x.shape[1] != n_features
            or not np.isfinite(x).all()):
        raise ValueError(f"{key}: expected finite (T,{n_features}) with T>={window_size}; got {x.shape}.")
    print(f"Puzzle evaluation source: {key} -> {split.upper()}", flush=True)
    return np.ascontiguousarray(x), split, key


def make_eval_plan(n_time, window_size, deltas):
    """Plan generated once: independent of evaluation batch size or training RNG."""
    for name, value in (("PUZZLE_EVAL_WINDOWS", PUZZLE_EVAL_WINDOWS),
                        ("PUZZLE_EVAL_BATCH_SIZE", PUZZLE_EVAL_BATCH_SIZE),
                        ("PUZZLE_EVAL_PERMUTATIONS", PUZZLE_EVAL_PERMUTATIONS)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if PUZZLE_EVAL_PERMUTATIONS > 24:
        raise ValueError("PUZZLE_EVAL_PERMUTATIONS must not exceed 24.")
    if n_time < window_size:
        raise ValueError("Evaluation data are shorter than one full window.")
    rng = np.random.default_rng(PUZZLE_EVAL_SEED)
    n_available = n_time - window_size + 1
    count = min(PUZZLE_EVAL_WINDOWS, n_available)
    starts = np.sort(rng.choice(n_available, count, replace=False))
    # Same helper/table/convention as training. Sample one valid quartet per window.
    slots, labels = _draw_plan(count, window_size, tuple(deltas),
                               PUZZLE_EVAL_PERMUTATIONS, rng)
    return np.repeat(starts, PUZZLE_EVAL_PERMUTATIONS), slots, labels, count


def summarize_puzzle_predictions(targets, predictions, nll):
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if (targets.ndim != 1 or len(targets) == 0 or predictions.shape != targets.shape
            or np.any((targets < 0) | (targets >= 24))
            or np.any((predictions < 0) | (predictions >= 24))):
        raise ValueError("Invalid puzzle class IDs.")
    confusion = np.zeros((24, 24), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    counts = confusion.sum(axis=1)
    per_class = [float(confusion[i, i] / n) if n else None for i, n in enumerate(counts)]
    identity = int(np.flatnonzero(np.all(PERMUTATIONS == np.arange(4), axis=1))[0])
    correct = targets == predictions
    return dict(
        accuracy=float(correct.mean()), accuracy_percent=float(100 * correct.mean()),
        cross_entropy=float(nll / len(targets)), n_puzzles=len(targets),
        chance_accuracy_percent=100 / 24,
        identity_class_id=identity,
        identity_accuracy=(float(correct[targets == identity].mean())
                           if np.any(targets == identity) else None),
        nonidentity_accuracy=(float(correct[targets != identity].mean())
                              if np.any(targets != identity) else None),
        class_counts=counts.tolist(), per_class_accuracy=per_class,
        confusion_matrix=confusion.tolist(),
    )


def evaluate_puzzle(model, x, split, data_key, out):
    """Evaluate encoder+head on shuffled held-out windows; never update weights."""
    width = model.window_size
    deltas = model.delta if isinstance(model.delta, (tuple, list)) else (model.delta,)
    starts, slots, labels, n_windows = make_eval_plan(len(x), width, deltas)
    predictions = np.empty(len(labels), dtype=np.int64)
    total_nll = 0.0
    encoder_mode, head_mode = model.encoder_.training, model.head_.training
    model.encoder_.eval()
    model.head_.eval()
    try:
        with torch.inference_mode():
            for begin in range(0, len(labels), PUZZLE_EVAL_BATCH_SIZE):
                end = min(begin + PUZZLE_EVAL_BATCH_SIZE, len(labels))
                indices = starts[begin:end, None] + np.arange(width)[None, :]
                windows = np.ascontiguousarray(x[indices].transpose(0, 2, 1))
                windows = torch.from_numpy(windows).to(model.device_)
                # Base windows are already repeated by the fixed plan: count=1 here.
                inputs, mask, target = _make_puzzles(windows, slots[begin:end], labels[begin:end], 1)
                z = model.encoder_(inputs)
                logits = model.head_(torch.cat((z, mask), dim=1))
                if logits.shape != (end - begin, 24) or not torch.isfinite(logits).all().item():
                    raise ValueError("Puzzle head returned invalid logits.")
                total_nll += nn.functional.cross_entropy(logits, target, reduction="sum").item()
                predictions[begin:end] = logits.argmax(dim=1).cpu().numpy()
    finally:
        model.encoder_.train(encoder_mode)
        model.head_.train(head_mode)
    metrics = summarize_puzzle_predictions(labels, predictions, total_nll)
    metrics.update(split=split, data_key=data_key, window_size=width, deltas=list(deltas),
                   n_windows=n_windows, n_available_windows=len(x) - width + 1,
                   permutations_per_window=PUZZLE_EVAL_PERMUTATIONS,
                   evaluation_seed=PUZZLE_EVAL_SEED, padded=False,
                   sampling="distinct window starts; one random quartet per window; windows may overlap",
                   definition="fixed-weight encoder+head, eval mode; exact 24-class permutation accuracy")
    save_json(out / f"puzzle_accuracy_{split}.json", metrics)
    np.savez_compressed(out / f"puzzle_predictions_{split}.npz",
                        window_starts=starts, selected_relative_bins=slots,
                        selected_absolute_bins=starts[:, None] + slots,
                        permutation_table=PERMUTATIONS,
                        true_class=labels, predicted_class=predictions)
    print_puzzle_accuracy(metrics)
    print("Saved puzzle evaluation:", out, flush=True)
    return metrics


def print_puzzle_accuracy(metrics):
    print(f"\nPUZZLE {metrics['split'].upper()} ACCURACY: {metrics['accuracy_percent']:.2f}% "
          f"| CE={metrics['cross_entropy']:.5f} | chance=4.17%", flush=True)
    print(f"Windows={metrics['n_windows']}, puzzles={metrics['n_puzzles']}, "
          f"window_size={metrics['window_size']}, delta={metrics['deltas']}", flush=True)


def evaluate_saved_puzzle(checkpoint, split):
    """Read only a saved puzzle model and evaluation features; no training."""
    model = puzzle.load(checkpoint, device=DEVICE)
    x, split, key = load_puzzle_eval_data(NPZ_PATH, split, model.window_size, model.n_features_in_)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = OUT_ROOT / f"puzzle_eval_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "evaluation_config.json", dict(checkpoint=str(Path(checkpoint).resolve()),
                                                   data_path=str(NPZ_PATH), model=model.get_params()))
    evaluate_puzzle(model, x, split, key, out)


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


def main(puzzle_checkpoint=None, eval_split=PUZZLE_EVAL_SPLIT):
    if puzzle_checkpoint is not None:
        evaluate_saved_puzzle(puzzle_checkpoint, eval_split)
        return
    if LATENT_DIM < 3 or DECODER_EPOCHS < 1 or PLOT_MAX_POINTS < 1:
        raise ValueError("Need LATENT_DIM>=3, DECODER_EPOCHS>=1 and PLOT_MAX_POINTS>=1.")
    actual = Path(inspect.getfile(puzzle)).resolve()
    if actual != (ROOT / "puzzle.py").resolve():
        raise RuntimeError(f"Expected local {ROOT / 'puzzle.py'}, imported {actual} instead.")
    x_train, x_valid, y_train_full, y_valid_full = load_data(NPZ_PATH)
    # Validate the requested held-out split before a potentially long training run.
    x_eval, actual_split, eval_key = load_puzzle_eval_data(
        NPZ_PATH, eval_split, WINDOW_SIZE, x_train.shape[1])
    seed_all(SEED)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = OUT_ROOT / stamp
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    print("Data:", NPZ_PATH, "| shapes:", x_train.shape, x_valid.shape, flush=True)
    print("Outputs:", out, flush=True)
    model = puzzle(
        window_size=WINDOW_SIZE, delta=DELTA, output_dimension=LATENT_DIM,
        num_hidden_units=ENCODER_HIDDEN, head_hidden_units=HEAD_HIDDEN,
        batch_size=PUZZLE_BATCH_SIZE, max_epochs=PUZZLE_EPOCHS,
        learning_rate=PUZZLE_LR, dropout=PUZZLE_DROPOUT, normalize=True,
        permutations_per_window=PERMUTATIONS_PER_WINDOW,
        device=DEVICE, random_state=SEED, verbose=True,
    )
    save_json(out / "run_config.json", dict(
        session=SESSION, data_path=str(NPZ_PATH), puzzle_module=str(actual),
        encoder=model.get_params(), pad_transform=PAD_TRANSFORM,
        label_alignment="transform return_indices, same input time bin",
        decoder=dict(epochs=DECODER_EPOCHS, hidden=DECODER_HIDDEN,
                     dropout=DECODER_DROPOUT, lr=DECODER_LR, seed=SEED + 100000,
                     batch_mode="full", input="original latent, not PCA"),
        pca=dict(n_components=3, fit_split="train", color_label=PCA_COLOR_LABEL),
        puzzle_evaluation=dict(split=actual_split, data_key=eval_key,
                               max_windows=PUZZLE_EVAL_WINDOWS,
                               permutations_per_window=PUZZLE_EVAL_PERMUTATIONS,
                               seed=PUZZLE_EVAL_SEED),
        data_shapes=dict(train=list(x_train.shape), validation=list(x_valid.shape)),
        versions=dict(torch=str(torch.__version__), numpy=np.__version__),
    ))
    model.fit(x_train)  # No behavior labels or validation inputs enter encoder fit.
    model.save(out / "puzzle_model.pt")
    save_json(out / "puzzle_history.json", model.history_)
    # End-of-encoder-training evaluation, saved before downstream decoder/PCA work.
    puzzle_metrics = evaluate_puzzle(model, x_eval, actual_split, eval_key, out)
    del x_eval
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
    print_puzzle_accuracy(puzzle_metrics)  # Show puzzle accuracy alongside final R2.
    print("\nSaved all results:", out, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puzzle-checkpoint", type=Path, default=None,
                        help="Only evaluate this saved puzzle_model.pt; skip all training and decoding.")
    parser.add_argument("--eval-split", choices=("auto", "test", "validation"),
                        default=PUZZLE_EVAL_SPLIT)
    options = parser.parse_args()
    main(puzzle_checkpoint=options.puzzle_checkpoint, eval_split=options.eval_split)





#####################################################
# """Train puzzle + a separate behavioral decoder on Perich C-CO16.

# Place next to the previously supplied puzzle.py and run:
#     python -u train_puzzle_cco16.py

# Dependencies: torch, numpy, scikit-learn, matplotlib, and local puzzle.py.
# The NPZ must contain train_data, valid_data, train_label, valid_label.
# Each feature split is treated as a continuous (time, neurons) recording.
# Decoder R2 is measured on valid_data: this is VALIDATION, not held-out test R2.
# The encoder sees only train_data; the decoder sees only training embeddings
# and training labels. No checkpoint selection or early stopping uses valid_data.
# PCA is for visualization only; the decoder uses the full latent dimension.
# """
# from datetime import datetime, timezone
# from pathlib import Path
# import csv
# import gc
# import inspect
# import json
# import random
# import time

# import numpy as np
# import torch
# from torch import nn
# import matplotlib
# matplotlib.use("Agg")  # Cluster/headless rendering.
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# from sklearn.decomposition import PCA
# from sklearn.metrics import r2_score

# from puzzle import puzzle


# ROOT = Path(__file__).resolve().parent
# SESSION = "C-CO16"
# PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
# NPZ_PATH = PERICH_DATA_DIR / f"{SESSION}.npz"
# OUT_ROOT = ROOT / f"PUZZLE_{SESSION}_RESULTS"
# SEED = 42

# # Puzzle: full 10-bin windows, four shuffled slots with fixed spacing delta.
# WINDOW_SIZE = 20
# DELTA = 1
# LATENT_DIM = 64
# ENCODER_HIDDEN = 64
# HEAD_HIDDEN = 128
# PUZZLE_BATCH_SIZE = 256
# PUZZLE_EPOCHS = 15000
# PUZZLE_LR = 3e-4
# PUZZLE_DROPOUT = 0.0
# PERMUTATIONS_PER_WINDOW = 4  # 24 processes all permutations; much higher memory/cost.
# DEVICE = "cuda_if_available"

# # True: one embedding per input bin, with edge padding per split.
# # False: only interior windows; labels are aligned using returned time indices.
# PAD_TRANSFORM = True
# TRANSFORM_BATCH_SIZE = 2048

# # Same full-batch MLP layout/hyperparameters as the earlier comparison scripts.
# DECODER_EPOCHS = 2500
# DECODER_HIDDEN = 64
# DECODER_DROPOUT = 0.4
# DECODER_LR = 1e-3
# PREDICT_BATCH_SIZE = 8192

# # Both PCA plots contain train and validation panels in the SAME fitted basis.
# PCA_COLOR_LABEL = 0  # Zero-based behavior-label column; no assumed physical meaning.
# PLOT_MAX_POINTS = 10000  # Per split, for rendering only; PCA uses ALL train embeddings.
# SAVE_EMBEDDINGS = True


# def save_json(path, values):
#     path.write_text(json.dumps(values, indent=2, allow_nan=False), encoding="utf-8")


# def seed_all(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True


# def cleanup():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


# def load_data(path):
#     with np.load(path, allow_pickle=False) as data:
#         arrays = [np.asarray(data[key], dtype=np.float32) for key in
#                   ("train_data", "valid_data", "train_label", "valid_label")]
#     x_train, x_valid, y_train, y_valid = arrays
#     if y_train.ndim == 1:
#         y_train = y_train[:, None]
#     if y_valid.ndim == 1:
#         y_valid = y_valid[:, None]
#     arrays = (x_train, x_valid, y_train, y_valid)
#     for name, x in zip(("train_data", "valid_data", "train_label", "valid_label"), arrays):
#         if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
#             raise ValueError(f"{name} must be a finite nonempty 2D array; got {x.shape}.")
#     if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
#         raise ValueError("Feature/label time lengths do not match.")
#     if x_train.shape[1] != x_valid.shape[1] or y_train.shape[1] != y_valid.shape[1]:
#         raise ValueError("Train/validation channel or label dimensions do not match.")
#     if min(len(x_train), len(x_valid)) < WINDOW_SIZE + 1:
#         raise ValueError("Each split needs at least WINDOW_SIZE+1 bins for R2 with valid windows.")
#     if not PAD_TRANSFORM and len(x_train) - WINDOW_SIZE + 1 < 3:
#         raise ValueError("Without padding, training needs at least three valid windows for 3D PCA.")
#     if not 0 <= PCA_COLOR_LABEL < y_train.shape[1]:
#         raise ValueError(f"PCA_COLOR_LABEL must be in [0, {y_train.shape[1] - 1}].")
#     return tuple(np.ascontiguousarray(a) for a in arrays)


# class Decoder(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, DECODER_HIDDEN), nn.LayerNorm(DECODER_HIDDEN),
#             nn.ReLU(), nn.Dropout(DECODER_DROPOUT),
#             nn.Linear(DECODER_HIDDEN, output_dim),
#         )

#     def forward(self, z):
#         return self.net(z)


# def train_decoder(z_train, y_train, device):
#     seed_all(SEED + 100000)
#     model = Decoder(z_train.shape[1], y_train.shape[1]).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=DECODER_LR)
#     z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
#     y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
#     losses = []
#     model.train()
#     for epoch in range(DECODER_EPOCHS):
#         optimizer.zero_grad(set_to_none=True)
#         loss = nn.functional.mse_loss(model(z), y)
#         if not torch.isfinite(loss).item():
#             raise FloatingPointError(f"Nonfinite decoder loss at epoch {epoch + 1}.")
#         loss.backward()
#         optimizer.step()
#         losses.append(float(loss.detach().item()))
#         if (epoch + 1) % 250 == 0 or epoch + 1 == DECODER_EPOCHS:
#             print(f"Decoder {epoch + 1}/{DECODER_EPOCHS}: train MSE={losses[-1]:.6f}", flush=True)
#     return model, losses


# def predict_decoder(model, z):
#     model.eval()  # Dropout must be off for R2 evaluation.
#     device = next(model.parameters()).device
#     predictions = []
#     with torch.inference_mode():
#         for start in range(0, len(z), PREDICT_BATCH_SIZE):
#             inputs = torch.as_tensor(z[start:start + PREDICT_BATCH_SIZE],
#                                      dtype=torch.float32, device=device)
#             predictions.append(model(inputs).cpu().numpy())
#     return np.concatenate(predictions, axis=0)


# def score_r2(y, predictions):
#     if y.shape != predictions.shape or len(y) < 2 or not np.isfinite(predictions).all():
#         raise ValueError("Invalid predictions for R2.")
#     # Explicitly record constant targets; sklearn maps their undefined R2 to 0/1.
#     values = np.atleast_1d(r2_score(y, predictions, multioutput="raw_values", force_finite=True))
#     constant = np.flatnonzero(np.all(y == y[0], axis=0)).tolist()
#     return dict(mean_r2=float(values.mean()), per_output_r2=values.tolist(),
#                 constant_target_columns=constant)


# def plot_r2(scores, out):
#     fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
#     positions = np.arange(len(scores["validation"]["per_output_r2"]))
#     for offset, (split, values) in zip((-0.18, 0.18), scores.items()):
#         ax.bar(positions + offset, values["per_output_r2"], width=0.36,
#                label=f"{split}: mean R2={values['mean_r2']:.4f}")
#     ax.set_xticks(positions, [str(i) for i in positions])
#     ax.set_xlabel("Behavior label column")
#     ax.set_ylabel("Decoder R2")
#     ax.set_title(f"{SESSION} | puzzle embeddings, full-dimensional MLP decoder")
#     ax.axhline(0, color="black", linewidth=0.6)
#     ax.legend()
#     fig.savefig(out / "decoder_R2.png", dpi=250)
#     plt.close(fig)


# def save_pca_plots(z_train, z_valid, y_train, y_valid, train_indices, valid_indices, out):
#     """Train-only PCA fit; validation is transformed with exactly the same basis."""
#     pca = PCA(n_components=3, svd_solver="randomized", random_state=SEED)
#     pca.fit(z_train)
#     points = (pca.transform(z_train), pca.transform(z_valid))
#     ratio = pca.explained_variance_ratio_
#     if not np.isfinite(ratio).all() or not all(np.isfinite(p).all() for p in points):
#         raise ValueError("PCA is nonfinite; check for collapsed or invalid embeddings.")
#     np.savez_compressed(out / "pca_coordinates.npz", train=points[0], validation=points[1],
#                         components=pca.components_, mean=pca.mean_,
#                         explained_variance_ratio=ratio, explained_variance=pca.explained_variance_,
#                         train_indices=train_indices, valid_indices=valid_indices)
#     colors = (y_train[:, PCA_COLOR_LABEL], y_valid[:, PCA_COLOR_LABEL])
#     low = min(float(c.min()) for c in colors)
#     high = max(float(c.max()) for c in colors)
#     if low == high:
#         low, high = low - 0.5, high + 0.5
#     norm = Normalize(vmin=low, vmax=high)
#     rng = np.random.default_rng(SEED)
#     selected = [np.sort(rng.choice(len(p), size=min(len(p), PLOT_MAX_POINTS), replace=False))
#                 for p in points]
#     # Save the plotted subset too, so each figure can be reproduced.
#     np.savez_compressed(out / "pca_plot_indices.npz", train=selected[0], validation=selected[1])
#     axes_names = [f"PC{i + 1} ({100 * ratio[i]:.1f}% train variance)" for i in range(3)]
#     limits = []
#     for i in range(3):
#         lo = min(float(p[:, i].min()) for p in points)
#         hi = max(float(p[:, i].max()) for p in points)
#         margin = max((hi - lo) * 0.03, 1e-5)
#         limits.append((lo - margin, hi + margin))
#     fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True,
#                              constrained_layout=True)
#     for ax, p, c, selected_rows, split in zip(axes, points, colors, selected, ("Train", "Validation")):
#         scatter = ax.scatter(p[selected_rows, 0], p[selected_rows, 1], c=c[selected_rows],
#                              cmap="viridis", norm=norm, s=5, alpha=0.75, linewidths=0)
#         ax.set(xlabel=axes_names[0], ylabel=axes_names[1], xlim=limits[0], ylim=limits[1],
#                title=f"{split} | {len(selected_rows):,} plotted / {len(p):,} points")
#     fig.colorbar(scatter, ax=list(axes), label=f"Behavior label column {PCA_COLOR_LABEL}", shrink=0.85)
#     fig.suptitle(f"{SESSION} | puzzle embeddings: 2D PCA (fit on train only)")
#     fig.savefig(out / "embeddings_PCA_2D.png", dpi=250)
#     plt.close(fig)
#     fig = plt.figure(figsize=(14, 6), constrained_layout=True)
#     axes = [fig.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
#     for ax, p, c, selected_rows, split in zip(axes, points, colors, selected, ("Train", "Validation")):
#         scatter = ax.scatter(p[selected_rows, 0], p[selected_rows, 1], p[selected_rows, 2],
#                              c=c[selected_rows], cmap="viridis", norm=norm, s=5,
#                              alpha=0.75, linewidths=0, depthshade=False)
#         ax.set(xlabel=axes_names[0], ylabel=axes_names[1], zlabel=axes_names[2],
#                xlim=limits[0], ylim=limits[1], zlim=limits[2],
#                title=f"{split} | {len(selected_rows):,} plotted / {len(p):,} points")
#         ax.view_init(elev=22, azim=45)
#     fig.colorbar(scatter, ax=axes, label=f"Behavior label column {PCA_COLOR_LABEL}", shrink=0.65, pad=0.08)
#     fig.suptitle(f"{SESSION} | puzzle embeddings: 3D PCA (fit on train only)")
#     fig.savefig(out / "embeddings_PCA_3D.png", dpi=250)
#     plt.close(fig)
#     return dict(fit_split="train", n_fit_samples=len(z_train),
#                 explained_variance_ratio=ratio.tolist(), color_label_column=PCA_COLOR_LABEL,
#                 plot_max_points_per_split=PLOT_MAX_POINTS)


# def main():
#     if LATENT_DIM < 3 or DECODER_EPOCHS < 1 or PLOT_MAX_POINTS < 1:
#         raise ValueError("Need LATENT_DIM>=3, DECODER_EPOCHS>=1 and PLOT_MAX_POINTS>=1.")
#     actual = Path(inspect.getfile(puzzle)).resolve()
#     if actual != (ROOT / "puzzle.py").resolve():
#         raise RuntimeError(f"Expected local {ROOT / 'puzzle.py'}, imported {actual} instead.")
#     x_train, x_valid, y_train_full, y_valid_full = load_data(NPZ_PATH)
#     seed_all(SEED)
#     stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
#     out = OUT_ROOT / stamp
#     out.mkdir(parents=True, exist_ok=False)
#     started = time.perf_counter()
#     print("Data:", NPZ_PATH, "| shapes:", x_train.shape, x_valid.shape, flush=True)
#     print("Outputs:", out, flush=True)
#     model = puzzle(
#         window_size=WINDOW_SIZE, delta=DELTA, output_dimension=LATENT_DIM,
#         num_hidden_units=ENCODER_HIDDEN, head_hidden_units=HEAD_HIDDEN,
#         batch_size=PUZZLE_BATCH_SIZE, max_epochs=PUZZLE_EPOCHS,
#         learning_rate=PUZZLE_LR, dropout=PUZZLE_DROPOUT, normalize=True,
#         permutations_per_window=PERMUTATIONS_PER_WINDOW,
#         device=DEVICE, random_state=SEED, verbose=True,
#     )
#     save_json(out / "run_config.json", dict(
#         session=SESSION, data_path=str(NPZ_PATH), puzzle_module=str(actual),
#         encoder=model.get_params(), pad_transform=PAD_TRANSFORM,
#         label_alignment="transform return_indices, same input time bin",
#         decoder=dict(epochs=DECODER_EPOCHS, hidden=DECODER_HIDDEN,
#                      dropout=DECODER_DROPOUT, lr=DECODER_LR, seed=SEED + 100000,
#                      batch_mode="full", input="original latent, not PCA"),
#         pca=dict(n_components=3, fit_split="train", color_label=PCA_COLOR_LABEL),
#         data_shapes=dict(train=list(x_train.shape), validation=list(x_valid.shape)),
#         versions=dict(torch=str(torch.__version__), numpy=np.__version__),
#     ))
#     model.fit(x_train)  # No behavior labels or validation inputs enter encoder fit.
#     model.save(out / "puzzle_model.pt")
#     save_json(out / "puzzle_history.json", model.history_)
#     z_train, train_indices = model.transform(x_train, pad=PAD_TRANSFORM,
#                                              batch_size=TRANSFORM_BATCH_SIZE, return_indices=True)
#     z_valid, valid_indices = model.transform(x_valid, pad=PAD_TRANSFORM,
#                                              batch_size=TRANSFORM_BATCH_SIZE, return_indices=True)
#     for z, indices in ((z_train, train_indices), (z_valid, valid_indices)):
#         if z.shape != (len(indices), LATENT_DIM) or not np.isfinite(z).all():
#             raise ValueError("Invalid puzzle embeddings.")
#     y_train = y_train_full[train_indices]
#     y_valid = y_valid_full[valid_indices]
#     if SAVE_EMBEDDINGS:
#         np.savez_compressed(out / "embeddings.npz", Z_train=z_train, Z_valid=z_valid,
#                             Y_train=y_train, Y_valid=y_valid,
#                             train_indices=train_indices, valid_indices=valid_indices)
#     device = model.device_
#     encoder_seconds = time.perf_counter() - started
#     del model, x_train, x_valid, y_train_full, y_valid_full
#     cleanup()

#     decoder, losses = train_decoder(z_train, y_train, device)
#     train_pred = predict_decoder(decoder, z_train)
#     valid_pred = predict_decoder(decoder, z_valid)
#     scores = dict(train=score_r2(y_train, train_pred), validation=score_r2(y_valid, valid_pred))
#     save_json(out / "R2.json", scores)
#     with (out / "R2.csv").open("w", newline="", encoding="utf-8") as handle:
#         writer = csv.writer(handle)
#         writer.writerow(["split", "mean_r2"] + [f"r2_label_{i}" for i in range(y_train.shape[1])])
#         for split, values in scores.items():
#             writer.writerow([split, values["mean_r2"], *values["per_output_r2"]])
#     torch.save(dict(state_dict={k: v.detach().cpu() for k, v in decoder.state_dict().items()},
#                     input_dim=LATENT_DIM, output_dim=y_train.shape[1], hidden=DECODER_HIDDEN,
#                     dropout=DECODER_DROPOUT, seed=SEED + 100000), out / "decoder.pt")
#     np.save(out / "decoder_train_mse.npy", np.asarray(losses))
#     np.savez_compressed(out / "decoder_predictions.npz", train_true=y_train, train_pred=train_pred,
#                         valid_true=y_valid, valid_pred=valid_pred,
#                         train_indices=train_indices, valid_indices=valid_indices)
#     plot_r2(scores, out)
#     pca_info = save_pca_plots(z_train, z_valid, y_train, y_valid, train_indices, valid_indices, out)
#     save_json(out / "pca_info.json", pca_info)
#     save_json(out / "timing.json", dict(encoder_and_transform_seconds=encoder_seconds,
#                                         total_seconds=time.perf_counter() - started))
#     for split, values in scores.items():
#         print(f"\n{split.upper()} mean R2: {values['mean_r2']:.6f}", flush=True)
#         print("Per-output R2:", values["per_output_r2"], flush=True)
#         if values["constant_target_columns"]:
#             print("Constant target columns (sklearn finite R2 convention):",
#                   values["constant_target_columns"], flush=True)
#     print("\nSaved all results:", out, flush=True)


# if __name__ == "__main__":
#     main()

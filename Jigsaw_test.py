"""Ablation run for JigsawCEBRA on one Perich session: does the pretext help?

This script is deliberately NOT a "tune the pretext" sweep. It is built to
answer one question your own tile_normalize sweep could not answer, because it
was missing the controls that make the answer identifiable:

    Does learning temporal order add anything to a behavior embedding,
    over and above (a) an untrained encoder, (b) the same model with the
    pretext switched off, and (c) decoding the raw neural window directly?

Every number is therefore reported next to its own null:

    R2                  vs  RAW WINDOW ridge/MLP R2   (the real ceiling)
                        vs  random-encoder arm        (did training do anything)
                        vs  lambda_puzzle=0 arm       (did the PUZZLE do anything)
    puzzle pair %       vs  mean-sort / norm-sort baselines (the drift shortcut)
    puzzle exact %      vs  1/K! chance, with a 95% CI and a z-test, so a
                            "4.69% vs 4.17% chance" reading is called what it
                            is: noise.

It also runs a LEARNABILITY LADDER. If the model cannot beat chance on the
easiest possible version of the pretext (n_tiles=2, i.e. "which of these two
snippets came first", chance 50%), then no conclusion about "puzzle learning
does not build good embeddings" is available from the data -- the pretext was
simply never learned, and every arm is a random encoder wearing a hat.

Data contract (same .npz layout as the earlier scripts):
    train_data (T1,N), valid_data (T2,N), train_label (T1,M), valid_label (T2,M)
    test_data (T3,N) optional; used for the puzzle evaluation when present.

Usage
    python run_jigsaw_perich.py --quick              # ~minutes, sanity check
    python run_jigsaw_perich.py                      # the full table
    python run_jigsaw_perich.py --arms proposed infonce_only random_encoder
    python run_jigsaw_perich.py --session C-CO16 --epochs 60

Reading the result: the pretext is worth keeping only if `proposed` beats BOTH
`infonce_only` and `random_encoder` on valid R2, on the same split, with the
same seed. If `raw_window` beats all of them, the encoder is subtracting
information and the honest report is that it does.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import gc
import json
import math
import random
import time

import numpy as np
import torch
from torch import nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jigsaw_cebra import JigsawCEBRA, _ridge_r2  # _ridge_r2: chronological-split ridge

ROOT = Path(__file__).resolve().parent
PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
SEED = 42

# -- encoder: identical across every arm, so the arm is the only difference -- #
WINDOW_SIZE = 10
N_TILES = 4
TILE_GAP = (1, 8)
BEHAVIOR_DIM = 32
PUZZLE_DIM = 16
ENCODER_HIDDEN = 64
HEAD_HIDDEN = 64
BATCH_SIZE = 512
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
DROPOUT = 0.0
TEMPERATURE = 1.0
LAMBDA_INFONCE = 1.0
LAMBDA_PUZZLE = 0.3
LAMBDA_DECORRELATION = 1.0
TILE_NORM = "mean"
NEURON_DROPOUT = 0.1
GAIN_JITTER = 0.1
DEVICE = "cuda_if_available"

# -- downstream decoder: the same full-batch MLP the earlier scripts used ---- #
DECODER_EPOCHS = 2500
DECODER_HIDDEN = 64
DECODER_DROPOUT = 0.4
DECODER_LR = 1e-3
PREDICT_BATCH_SIZE = 8192
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)

# -- puzzle evaluation ------------------------------------------------------- #
PUZZLE_EVAL_SPANS = 2048
PUZZLE_EVAL_BATCH = 256
PUZZLE_EVAL_SEED = SEED + 200000

MAX_RAW_FEATURES = 20000  # guard on the flattened-window baseline


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
ARMS = {
    # The proposal: split latent, InfoNCE on z_behavior, jigsaw on z_puzzle.
    "proposed": {},
    # Did the PUZZLE do anything? Same code path, pretext weight zero.
    "infonce_only": dict(lambda_puzzle=0.0),
    # The failure mode on purpose: pretext alone, nothing pulling toward behavior.
    "puzzle_only": dict(lambda_infonce=0.0, lambda_decorrelation=0.0),
    # Did TRAINING do anything? Same architecture, zero optimizer steps.
    "random_encoder": dict(max_epochs=0),
    # Is order even decodable from behavior features? Head trains, trunk is
    # shielded from its gradient, so R2 cannot be harmed by the pretext.
    "puzzle_probe_only": dict(puzzle_grad_scale=0.0),
    # Leave the level/drift shortcut wide open: puzzle accuracy should jump and
    # track the mean-sort baseline. That is the shortcut, caught in the act.
    "shortcut_open": dict(tile_norm="none", separate_puzzle_view=False,
                          neuron_dropout=0.0, gain_jitter=0.0),
    # MobileNetV2-style depthwise-separable trunk (the advisor's suggestion).
    "mobilenet_trunk": dict(trunk_block="separable"),
    # LEARNABILITY LADDER: easiest possible pretext, chance is 50% not 4.17%.
    "ladder_2_tiles": dict(n_tiles=2),
    "ladder_3_tiles": dict(n_tiles=3),
    # Is the pretext only hard because the tiles are close together in time?
    "wide_gaps": dict(tile_gap=(8, 40)),
    # No cross-block decorrelation: does the penalty matter at all?
    "no_decorrelation": dict(lambda_decorrelation=0.0),
}
DEFAULT_ARMS = ("proposed", "infonce_only", "random_encoder", "puzzle_only",
                "shortcut_open", "mobilenet_trunk", "ladder_2_tiles")


# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def json_safe(value):
    """NaN/Inf -> null, so the JSON stays strictly valid and loadable anywhere.

    This matters: the random_encoder arm has no training history, so several
    fields are legitimately NaN, and json.dumps(allow_nan=False) would crash the
    whole run at the very last step.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def save_json(path, values):
    path.write_text(json.dumps(json_safe(values), indent=2), encoding="utf-8")


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def normal_tail(z):
    """One-sided upper-tail probability of the standard normal."""
    return 0.5 * math.erfc(abs(z) / math.sqrt(2.0))


def proportion_versus_chance(successes, trials, chance):
    """Wilson 95% interval plus a one-sided z-test against a known chance rate.

    Why this is here: with 1024 spans the standard error of a 4.17%-chance
    accuracy is about 0.6 points, so 4.69% is well inside the noise band. Any
    claim of the form "the model is slightly above chance" needs this column
    before it means anything.
    """
    trials = int(trials)
    if trials <= 0:
        return dict(estimate=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
                    z=float("nan"), p_one_sided=float("nan"), above_chance=False, n=0)
    phat = successes / trials
    z = 1.959963985
    denominator = 1.0 + z * z / trials
    center = (phat + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denominator
    standard_error = math.sqrt(max(chance * (1 - chance) / trials, 1e-300))
    statistic = (phat - chance) / standard_error
    return dict(estimate=100 * phat, ci_low=100 * (center - spread), ci_high=100 * (center + spread),
                chance=100 * chance, z=statistic, p_one_sided=normal_tail(statistic) if statistic > 0 else 0.5,
                above_chance=bool(statistic > 1.96), n=trials)


def load_session(path):
    with np.load(path, allow_pickle=False) as data:
        available = set(data.files)
        required = ("train_data", "valid_data", "train_label", "valid_label")
        missing = [key for key in required if key not in available]
        if missing:
            raise KeyError(f"{path} is missing {missing}.")
        arrays = {key: np.ascontiguousarray(np.asarray(data[key], dtype=np.float32))
                  for key in required}
        test = np.ascontiguousarray(np.asarray(data["test_data"], dtype=np.float32)) \
            if "test_data" in available else None
    for key in ("train_label", "valid_label"):
        if arrays[key].ndim == 1:
            arrays[key] = arrays[key][:, None]
    for key, value in arrays.items():
        if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
            raise ValueError(f"{key}: expected a finite nonempty 2D array; got {value.shape}.")
    if len(arrays["train_data"]) != len(arrays["train_label"]) \
            or len(arrays["valid_data"]) != len(arrays["valid_label"]):
        raise ValueError("Feature and label lengths disagree.")
    if arrays["train_data"].shape[1] != arrays["valid_data"].shape[1]:
        raise ValueError("Train and validation neuron counts disagree.")
    if test is not None and test.shape[1] != arrays["train_data"].shape[1]:
        raise ValueError("test_data neuron count disagrees with train_data.")
    return arrays["train_data"], arrays["valid_data"], arrays["train_label"], \
        arrays["valid_label"], test


# --------------------------------------------------------------------------- #
# downstream decoding
# --------------------------------------------------------------------------- #
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


def r2_raw(y_true, y_hat):
    residual = np.square(y_true - y_hat).sum(0)
    total = np.square(y_true - y_true.mean(0, keepdims=True)).sum(0)
    return 1.0 - residual / np.maximum(total, 1e-12)


def mlp_r2(z_train, y_train, z_valid, y_valid, device, epochs, tag=""):
    """Full-batch MLP decoder, identical in shape to the earlier scripts.

    Standardization uses TRAIN statistics only. The decoder is reseeded per arm
    so decoder initialization is not a confound between arms.
    """
    seed_all(SEED + 100000)
    mean = z_train.mean(0, keepdims=True)
    scale = z_train.std(0, keepdims=True) + 1e-8
    model = Decoder(z_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=DECODER_LR)
    features = torch.as_tensor((z_train - mean) / scale, dtype=torch.float32, device=device)
    targets = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    losses = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(features), targets)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Nonfinite decoder loss at epoch {epoch + 1}.")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if (epoch + 1) % 500 == 0 or epoch + 1 == epochs:
            print(f"      decoder{tag} {epoch + 1}/{epochs}: train MSE={losses[-1]:.6f}", flush=True)

    def predict(z):
        model.eval()
        out = []
        standardized = (z - mean) / scale
        with torch.inference_mode():
            for start in range(0, len(standardized), PREDICT_BATCH_SIZE):
                block = torch.as_tensor(standardized[start:start + PREDICT_BATCH_SIZE],
                                        dtype=torch.float32, device=device)
                out.append(model(block).cpu().numpy())
        return np.concatenate(out, axis=0)

    train_r2 = r2_raw(y_train, predict(z_train))
    valid_r2 = r2_raw(y_valid, predict(z_valid))
    del model, features, targets
    cleanup()
    return dict(train_r2=float(train_r2.mean()), valid_r2=float(valid_r2.mean()),
                train_r2_per_dimension=train_r2.tolist(),
                valid_r2_per_dimension=valid_r2.tolist(),
                final_train_mse=losses[-1] if losses else float("nan"))


def safe_ridge(x_train, y_train, x_test, y_test):
    """Closed-form ridge, in float64 unless the design matrix is too big for it.

    Embedding blocks are tiny, so they always run in float64. The flattened raw
    window can be 40k x 2k, which is 640 MB per copy in float64; that case falls
    back to float32, which is fine because the features are standardized and the
    smallest alpha is 1e-3.
    """
    dtype = np.float64 if (x_train.size + x_test.size) * 8 < 1.5e9 else np.float32
    return _ridge_r2(x_train.astype(dtype), y_train.astype(dtype),
                     x_test.astype(dtype), y_test.astype(dtype), RIDGE_ALPHAS)


def sliding_windows(x, window_size):
    """(T,N) -> (T-W+1, W*N) flattened windows, matching transform(pad=False)."""
    strides = np.lib.stride_tricks.sliding_window_view(x, window_size, axis=0)
    return np.ascontiguousarray(strides.reshape(len(strides), -1))


def raw_window_baseline(x_train, y_train, x_valid, y_valid, window_size, device,
                        decoder_epochs, run_mlp=True):
    """Decode behavior straight from the raw window. This is the number every
    embedding has to beat before it is worth anything.
    """
    left = window_size // 2
    right = window_size - left - 1
    features_train = sliding_windows(x_train, window_size)
    features_valid = sliding_windows(x_valid, window_size)
    targets_train = y_train[left:len(x_train) - right]
    targets_valid = y_valid[left:len(x_valid) - right]
    result = dict(n_features=int(features_train.shape[1]), window_size=window_size)
    if features_train.shape[1] > MAX_RAW_FEATURES:
        print(f"  raw baseline: {features_train.shape[1]} features > {MAX_RAW_FEATURES}; "
              f"falling back to the single centre bin.", flush=True)
        features_train = x_train[left:len(x_train) - right]
        features_valid = x_valid[left:len(x_valid) - right]
        result["n_features"] = int(features_train.shape[1])
        result["window_size"] = 1
    ridge = safe_ridge(features_train, targets_train, features_valid, targets_valid)
    result["ridge_valid_r2"] = ridge["r2"]
    result["ridge_alpha"] = ridge["alpha"]
    result["participation_ratio"] = ridge["participation_ratio"]
    if run_mlp:
        mlp = mlp_r2(features_train, targets_train, features_valid, targets_valid,
                     device, decoder_epochs, tag=" [raw]")
        result["mlp_train_r2"] = mlp["train_r2"]
        result["mlp_valid_r2"] = mlp["valid_r2"]
    print(f"  RAW WINDOW baseline: ridge valid R2={result['ridge_valid_r2']:.4f}"
          + (f", MLP valid R2={result['mlp_valid_r2']:.4f}" if run_mlp else "")
          + f"  ({result['n_features']} features)", flush=True)
    return result


# --------------------------------------------------------------------------- #
# one arm
# --------------------------------------------------------------------------- #
def build_model(override, epochs, device):
    settings = dict(
        window_size=WINDOW_SIZE, n_tiles=N_TILES, tile_gap=TILE_GAP,
        behavior_dim=BEHAVIOR_DIM, puzzle_dim=PUZZLE_DIM,
        num_hidden_units=ENCODER_HIDDEN, head_hidden_units=HEAD_HIDDEN,
        dropout=DROPOUT, temperature=TEMPERATURE, lambda_infonce=LAMBDA_INFONCE,
        lambda_puzzle=LAMBDA_PUZZLE, lambda_decorrelation=LAMBDA_DECORRELATION,
        tile_norm=TILE_NORM, neuron_dropout=NEURON_DROPOUT, gain_jitter=GAIN_JITTER,
        batch_size=BATCH_SIZE, max_epochs=epochs, learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY, device=device, random_state=SEED, verbose=True,
        log_every=max(1, epochs // 20) if epochs else 1,
    )
    settings.update(override)
    return JigsawCEBRA(**settings)


def puzzle_report(model, x, name):
    metrics = model.evaluate_puzzle(x, max_spans=PUZZLE_EVAL_SPANS, batch_size=PUZZLE_EVAL_BATCH,
                                    random_state=PUZZLE_EVAL_SEED, verbose=False)
    spans = metrics["n_spans"]
    pairs = spans * model.n_tiles * (model.n_tiles - 1)
    exact_test = proportion_versus_chance(
        metrics["exact_accuracy_percent"] / 100 * spans, spans,
        metrics["chance_exact_accuracy_percent"] / 100)
    pair_test = proportion_versus_chance(
        metrics["pair_accuracy_percent"] / 100 * pairs, pairs, 0.5)
    metrics["exact_significance"] = exact_test
    metrics["pair_significance"] = pair_test
    verdict = "ABOVE CHANCE" if (exact_test["above_chance"] or pair_test["above_chance"]) \
        else "at chance -> the pretext was NOT learned on this split"
    print(f"  puzzle[{name}]: exact={metrics['exact_accuracy_percent']:.2f}% "
          f"[{exact_test['ci_low']:.2f}, {exact_test['ci_high']:.2f}] "
          f"vs chance {metrics['chance_exact_accuracy_percent']:.2f}% (z={exact_test['z']:.2f}) | "
          f"pair={metrics['pair_accuracy_percent']:.2f}% vs 50% (z={pair_test['z']:.2f}) | "
          f"mean-sort shortcut {metrics['baseline_mean_sort_pair_percent']:.1f}% | {verdict}",
          flush=True)
    return metrics


def run_arm(name, override, data, out_root, device, epochs, decoder_epochs):
    x_train, x_valid, y_train, y_valid, x_eval = data
    seed_all(SEED)
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n===== arm: {name}  {override or '(defaults)'} =====", flush=True)

    model = build_model(override, epochs, device)
    save_json(out / "config.json", dict(arm=name, override=override, params=model.get_params()))
    started = time.perf_counter()
    model.fit(x_train, X_valid=x_valid, validate_every=max(1, epochs // 10) if epochs else 1)
    train_seconds = time.perf_counter() - started
    model.save(out / "encoder.pt")
    save_json(out / "history.json", model.history_)
    save_json(out / "validation_history.json", model.validation_history_)

    # Puzzle: on the training split (how much was memorized) and on held-out data.
    puzzle_train = puzzle_report(model, x_train, "train")
    puzzle_valid = puzzle_report(model, x_valid, "valid")
    puzzle_test = puzzle_report(model, x_eval, "test") if x_eval is not None else None
    save_json(out / "puzzle.json", dict(train=puzzle_train, valid=puzzle_valid, test=puzzle_test))

    # Embeddings. pad=False plus return_indices keeps labels exactly aligned and
    # avoids the edge-padding artifact at the two ends of the recording.
    z_train, index_train = model.transform(x_train, block="behavior", pad=False,
                                           batch_size=2048, return_indices=True)
    z_valid, index_valid = model.transform(x_valid, block="behavior", pad=False,
                                           batch_size=2048, return_indices=True)
    targets_train, targets_valid = y_train[index_train], y_valid[index_valid]
    for name_, z in (("train", z_train), ("valid", z_valid)):
        if not np.isfinite(z).all():
            raise FloatingPointError(f"Nonfinite {name_} embedding in arm {name}.")
    np.savez_compressed(out / "embeddings.npz", Z_train=z_train, Z_valid=z_valid,
                        Y_train=targets_train, Y_valid=targets_valid,
                        train_indices=index_train, valid_indices=index_valid)

    ridge = safe_ridge(z_train, targets_train, z_valid, targets_valid)
    mlp = mlp_r2(z_train, targets_train, z_valid, targets_valid, device, decoder_epochs)
    # The nuisance block: order information is supposed to have moved HERE.
    z_puzzle_train = model.transform(x_train, block="puzzle", pad=False, batch_size=2048)
    z_puzzle_valid = model.transform(x_valid, block="puzzle", pad=False, batch_size=2048)
    ridge_puzzle = safe_ridge(z_puzzle_train, targets_train, z_puzzle_valid, targets_valid)
    scores = dict(ridge_behavior=ridge, ridge_puzzle=ridge_puzzle, mlp_behavior=mlp)
    save_json(out / "R2.json", scores)

    row = dict(
        arm=name, override=json.dumps(override), out_dir=str(out),
        epochs=epochs if "max_epochs" not in override else override["max_epochs"],
        n_tiles=model.n_tiles, train_seconds=train_seconds,
        r2_ridge_valid=ridge["r2"], r2_ridge_train_pr=ridge["participation_ratio"],
        r2_mlp_train=mlp["train_r2"], r2_mlp_valid=mlp["valid_r2"],
        r2_puzzle_block_valid=ridge_puzzle["r2"],
        puzzle_train_exact=puzzle_train["exact_accuracy_percent"],
        puzzle_valid_exact=puzzle_valid["exact_accuracy_percent"],
        puzzle_valid_exact_ci_low=puzzle_valid["exact_significance"]["ci_low"],
        puzzle_valid_exact_ci_high=puzzle_valid["exact_significance"]["ci_high"],
        puzzle_valid_exact_z=puzzle_valid["exact_significance"]["z"],
        puzzle_chance_exact=puzzle_valid["chance_exact_accuracy_percent"],
        puzzle_train_pair=puzzle_train["pair_accuracy_percent"],
        puzzle_valid_pair=puzzle_valid["pair_accuracy_percent"],
        puzzle_valid_pair_z=puzzle_valid["pair_significance"]["z"],
        shortcut_mean_sort_pair=puzzle_valid["baseline_mean_sort_pair_percent"],
        shortcut_norm_sort_pair=puzzle_valid["baseline_norm_sort_pair_percent"],
        puzzle_learned=bool(puzzle_valid["pair_significance"]["above_chance"]
                            or puzzle_valid["exact_significance"]["above_chance"]),
        infonce_final=model.history_[-1]["contrastive"] if model.history_ else float("nan"),
        infonce_accuracy_final=(100 * model.history_[-1]["contrastive_accuracy"]
                                if model.history_ else float("nan")),
    )
    print(f"[{name}] valid R2: ridge={row['r2_ridge_valid']:.4f}  MLP={row['r2_mlp_valid']:.4f} "
          f"(train {row['r2_mlp_train']:.4f}) | puzzle block R2={row['r2_puzzle_block_valid']:.4f} "
          f"| puzzle learned: {row['puzzle_learned']} | {train_seconds:.0f}s", flush=True)
    del model, z_train, z_valid, z_puzzle_train, z_puzzle_valid
    cleanup()
    return row


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def plot_summary(summary, baseline, out_root, session):
    names = [row["arm"] for row in summary]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(max(11, 1.6 * len(names) + 5), 5),
                             constrained_layout=True)

    axes[0].bar(x - 0.19, [row["r2_mlp_train"] for row in summary], width=0.38, label="train (MLP)")
    axes[0].bar(x + 0.19, [row["r2_mlp_valid"] for row in summary], width=0.38, label="valid (MLP)")
    axes[0].plot(x, [row["r2_ridge_valid"] for row in summary], "k.", markersize=9,
                 label="valid (ridge)")
    if baseline is not None:
        level = baseline.get("mlp_valid_r2", baseline["ridge_valid_r2"])
        axes[0].axhline(level, color="crimson", linestyle="--", linewidth=1.4,
                        label=f"raw window ({level:.3f})")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, names, rotation=30, ha="right")
    axes[0].set_ylabel("Behavior decoding R2")
    axes[0].set_title("What actually matters: R2 vs the raw-window ceiling")
    axes[0].legend(fontsize=8)

    axes[1].bar(x - 0.19, [row["puzzle_train_pair"] for row in summary], width=0.38, label="train")
    axes[1].bar(x + 0.19, [row["puzzle_valid_pair"] for row in summary], width=0.38, label="valid")
    axes[1].plot(x, [row["shortcut_mean_sort_pair"] for row in summary], "v", color="darkorange",
                 markersize=8, label="mean-sort shortcut")
    axes[1].axhline(50, color="black", linestyle="--", linewidth=1, label="chance (50%)")
    axes[1].set_xticks(x, names, rotation=30, ha="right")
    axes[1].set_ylabel("Pairwise order accuracy (%)")
    axes[1].set_ylim(40, 100)
    axes[1].set_title("Was the pretext learned at all?")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"{session} | JigsawCEBRA ablation (seed {SEED})")
    fig.savefig(out_root / "ablation.png", dpi=220)
    plt.close(fig)


def print_table(summary, baseline):
    header = (f"{'arm':<20}{'R2 valid':>10}{'R2 train':>10}{'R2 ridge':>10}"
              f"{'pair %':>9}{'shortcut':>10}{'exact %':>9}{'chance':>8}{'z':>7}{'learned':>9}")
    print("\n" + "=" * len(header))
    print("SUMMARY (sorted by validation R2)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in summary:
        print(f"{row['arm']:<20}{row['r2_mlp_valid']:>10.4f}{row['r2_mlp_train']:>10.4f}"
              f"{row['r2_ridge_valid']:>10.4f}{row['puzzle_valid_pair']:>9.2f}"
              f"{row['shortcut_mean_sort_pair']:>10.2f}{row['puzzle_valid_exact']:>9.2f}"
              f"{row['puzzle_chance_exact']:>8.2f}{row['puzzle_valid_pair_z']:>7.2f}"
              f"{str(row['puzzle_learned']):>9}")
    print("-" * len(header))
    if baseline is not None:
        print(f"{'RAW WINDOW':<20}{baseline.get('mlp_valid_r2', float('nan')):>10.4f}"
              f"{'':>10}{baseline['ridge_valid_r2']:>10.4f}"
              f"   <- no encoder at all; every arm must beat this to be useful")

    best = summary[0]
    by_name = {row["arm"]: row for row in summary}
    print("\nHOW TO READ THIS")
    if not any(row["puzzle_learned"] for row in summary):
        print("  * NO arm beat chance on held-out order. Nothing here supports any claim "
              "about what puzzle learning does to an embedding: the pretext was never "
              "learned out of sample, so every encoder is effectively a random encoder.")
    if "proposed" in by_name and "infonce_only" in by_name:
        delta = by_name["proposed"]["r2_mlp_valid"] - by_name["infonce_only"]["r2_mlp_valid"]
        print(f"  * proposed - infonce_only = {delta:+.4f} R2  <- the PUZZLE's contribution.")
    if "proposed" in by_name and "random_encoder" in by_name:
        delta = by_name["proposed"]["r2_mlp_valid"] - by_name["random_encoder"]["r2_mlp_valid"]
        print(f"  * proposed - random_encoder = {delta:+.4f} R2  <- TRAINING's contribution. "
              f"If this is ~0, the decoder is doing all the work.")
    if baseline is not None:
        level = baseline.get("mlp_valid_r2", baseline["ridge_valid_r2"])
        print(f"  * best arm ({best['arm']}) - raw window = "
              f"{best['r2_mlp_valid'] - level:+.4f} R2  <- the encoder's contribution.")
    print("  * If 'pair %' sits on 'shortcut', the puzzle is being solved by level drift.")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default="C-CO16")
    parser.add_argument("--data-dir", default=str(PERICH_DATA_DIR))
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=list(DEFAULT_ARMS))
    parser.add_argument("--all-arms", action="store_true", help="run every arm in ARMS")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--decoder-epochs", type=int, default=None)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--no-raw-baseline", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Few epochs, small decoder. Run this once before the full table.")
    options = parser.parse_args()

    epochs = options.epochs if options.epochs is not None else (3 if options.quick else EPOCHS)
    decoder_epochs = options.decoder_epochs if options.decoder_epochs is not None \
        else (200 if options.quick else DECODER_EPOCHS)
    arms = sorted(ARMS) if options.all_arms else options.arms

    path = Path(options.data_dir) / f"{options.session}.npz"
    x_train, x_valid, y_train, y_valid, x_test = load_session(path)
    x_eval = x_test if x_test is not None else x_valid
    device_name = options.device
    if device_name == "cuda_if_available":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / f"JIGSAW_CEBRA_{options.session}_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Session {options.session}: train={x_train.shape} valid={x_valid.shape} "
          f"labels={y_train.shape[1]}D | puzzle-eval split="
          f"{'test_data' if x_test is not None else 'valid_data'} | device={device_name} | "
          f"epochs={epochs} | arms={arms}\nOutput: {out_root}", flush=True)

    seed_all(SEED)
    baseline = None
    if not options.no_raw_baseline:
        baseline = raw_window_baseline(x_train, y_train, x_valid, y_valid, WINDOW_SIZE,
                                       device_name, decoder_epochs)
        save_json(out_root / "raw_window_baseline.json", baseline)

    data = (x_train, x_valid, y_train, y_valid, x_eval)
    summary = []
    for arm in arms:
        summary.append(run_arm(arm, ARMS[arm], data, out_root, device_name, epochs, decoder_epochs))
        summary_sorted = sorted(summary, key=lambda row: row["r2_mlp_valid"], reverse=True)
        save_json(out_root / "summary.json",
                  dict(session=options.session, seed=SEED, epochs=epochs,
                       raw_window_baseline=baseline, arms=summary_sorted))
    summary.sort(key=lambda row: row["r2_mlp_valid"], reverse=True)

    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    plot_summary(summary, baseline, out_root, options.session)
    print_table(summary, baseline)
    print(f"\nSaved: {out_root}", flush=True)


if __name__ == "__main__":
    main()

# """Sweep NeuralJigsaw's tile_normalize modes on one session and compare them
# on BOTH puzzle-solving accuracy and downstream behavioral R^2 -- the two
# numbers that turned out to disagree for the earlier architectures, so this
# script always reports both, side by side, for every mode, rather than
# picking a "winner" from puzzle accuracy alone.

# Mirrors the data loading / decoder / R2 conventions of the earlier
# window-mode comparison script (train_data/valid_data/train_label/valid_label
# in one .npz, optional test_data for puzzle evaluation). The only thing that
# changes between runs is NeuralJigsaw's tile_normalize; every other
# hyperparameter is held fixed so the comparison is apples-to-apples.

# The encoder checkpoint for each mode is selected by a CHEAP ridge probe on
# held-out validation embeddings (monitor_fn), evaluated periodically during
# fit -- i.e. by the metric you actually care about, not by puzzle loss.

# Usage:
#     python sweep_tile_normalize.py                      # off, center, zscore
#     python sweep_tile_normalize.py --modes center zscore
#     python sweep_tile_normalize.py --eval-split test
#     python sweep_tile_normalize.py --quick               # fast smoke test
# """
# from datetime import datetime, timezone
# from pathlib import Path
# import argparse
# import csv
# import gc
# import json
# import random
# import time

# import numpy as np
# import torch
# from torch import nn
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Ridge
# from sklearn.metrics import r2_score

# from Neural_Jigsaw import NeuralJigsaw


# ROOT = Path(__file__).resolve().parent
# SESSION = "C-CO16"
# PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
# NPZ_PATH = PERICH_DATA_DIR / f"{SESSION}.npz"
# OUT_ROOT = ROOT / f"NEURAL_JIGSAW_{SESSION}_SWEEP"
# SEED = 42

# MODES = ("off", "center", "zscore")  # every tile_normalize value NeuralJigsaw offers

# # --- encoder hyperparameters: identical across modes, so tile_normalize is ---
# # --- the only thing that differs between runs.                            ---
# WINDOW_SIZE = 8
# TILE_GAP = (1, 4)
# LATENT_DIM = 64
# ENCODER_HIDDEN = 64
# HEAD_HIDDEN = 128
# PUZZLE_BATCH_SIZE = 256
# PUZZLE_EPOCHS = 1
# PUZZLE_LR = 3e-4
# PUZZLE_WEIGHT_DECAY = 1e-4
# PUZZLE_DROPOUT = 0.1
# AUGMENT_GAIN_JITTER = 0.1
# AUGMENT_NOISE_STD = 0.05
# PERMUTATIONS_PER_WINDOW = 1
# DEVICE = "cuda"

# # --- monitor_fn: closed-form ridge probe on held-out validation embeddings, ---
# # --- called periodically during fit() so the CHECKPOINT is picked by       ---
# # --- downstream R2, not by puzzle CE/accuracy (see the 10-vs-20k paradox). ---
# MONITOR_EVERY = max(1, PUZZLE_EPOCHS // 40)   # ~40 probes over the whole run
# MONITOR_MAX_SAMPLES = 4000                     # subsample -- speed only
# MONITOR_RIDGE_ALPHA = 1.0

# # --- transform() alignment ---
# PAD_TRANSFORM = False
# TRANSFORM_BATCH_SIZE = 2048

# # --- final downstream decoder: same full-batch MLP as the earlier scripts ---
# DECODER_EPOCHS = 2500
# DECODER_HIDDEN = 64
# DECODER_DROPOUT = 0.4
# DECODER_LR = 1e-3
# PREDICT_BATCH_SIZE = 8192

# # --- fixed-weight puzzle evaluation (reported, but NOT used to pick a mode) ---
# PUZZLE_EVAL_SPLIT = "auto"       # prefer test_data; otherwise validation
# PUZZLE_EVAL_WINDOWS = 1024
# PUZZLE_EVAL_PERMUTATIONS = 24
# PUZZLE_EVAL_BATCH_SIZE = 256
# PUZZLE_EVAL_SEED = SEED + 200000
# # min_gap is set at runtime to each model's training_span (de-correlated windows)


# def training_span(window_size, tile_gap):
#     return 4 * window_size + 3 * tile_gap[1]


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
#     span = training_span(WINDOW_SIZE, TILE_GAP)
#     if min(len(x_train), len(x_valid)) < span + 1:
#         raise ValueError(f"Each split needs at least training_span+1={span + 1} bins; "
#                          f"got train={len(x_train)}, valid={len(x_valid)}.")
#     return tuple(np.ascontiguousarray(a) for a in arrays)


# def load_puzzle_eval_data(path, split, min_length, n_features):
#     if split not in ("auto", "test", "validation"):
#         raise ValueError("eval split must be auto, test, or validation.")
#     with np.load(path, allow_pickle=False) as data:
#         if split == "auto":
#             split = "test" if "test_data" in data.files else "validation"
#         key = "test_data" if split == "test" else "valid_data"
#         if key not in data.files:
#             raise KeyError(f"Requested {split} puzzle evaluation, but {key!r} is missing from {path}.")
#         x = np.asarray(data[key], dtype=np.float32)
#     if (x.ndim != 2 or x.shape[0] < min_length or x.shape[1] != n_features
#             or not np.isfinite(x).all()):
#         raise ValueError(f"{key}: expected finite (T,{n_features}) with T>={min_length}; got {x.shape}.")
#     print(f"Puzzle evaluation source: {key} -> {split.upper()}", flush=True)
#     return np.ascontiguousarray(x), split, key


# def make_monitor(x_valid, y_valid, rng_seed):
#     """Cheap closed-form ridge probe: model.transform -> ridge.fit on half of
#     a held-out subsample -> R2 on the other half. Meant to be called every
#     few hundred epochs from inside fit(), NOT to replace the real decoder
#     used for the final reported R2 (that one is a proper trained MLP, below).
#     """
#     rng = np.random.default_rng(rng_seed)

#     def monitor(model):
#         z, indices = model.transform(x_valid, pad=PAD_TRANSFORM,
#                                       batch_size=TRANSFORM_BATCH_SIZE, return_indices=True)
#         y = y_valid[indices]
#         if len(z) > MONITOR_MAX_SAMPLES:
#             pick = rng.choice(len(z), size=MONITOR_MAX_SAMPLES, replace=False)
#             z, y = z[pick], y[pick]
#         half = len(z) // 2
#         if half < 2:
#             return float("-inf")
#         probe = Ridge(alpha=MONITOR_RIDGE_ALPHA)
#         probe.fit(z[:half], y[:half])
#         prediction = probe.predict(z[half:])
#         return float(r2_score(y[half:], prediction, multioutput="uniform_average", force_finite=True))

#     return monitor


# def run_puzzle_eval(model, x, split, key, out, min_gap):
#     metrics, details = model.evaluate_puzzle(
#         x, max_windows=PUZZLE_EVAL_WINDOWS, permutations_per_window=PUZZLE_EVAL_PERMUTATIONS,
#         batch_size=PUZZLE_EVAL_BATCH_SIZE, random_state=PUZZLE_EVAL_SEED, min_gap=min_gap,
#         return_details=True)
#     metrics = dict(metrics, split=split, data_key=key)
#     save_json(out / f"puzzle_accuracy_{split}.json", metrics)
#     np.savez_compressed(out / f"puzzle_predictions_{split}.npz", **details)
#     print(f"  PUZZLE {split.upper()} ACCURACY: {metrics['accuracy_percent']:.2f}% "
#           f"| CE={metrics['cross_entropy']:.5f} | chance={metrics['chance_accuracy_percent']:.2f}% "
#           f"| min_gap={min_gap}", flush=True)
#     return metrics


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


# def train_decoder(z_train, y_train, device, epochs):
#     seed_all(SEED + 100000)
#     model = Decoder(z_train.shape[1], y_train.shape[1]).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=DECODER_LR)
#     z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
#     y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
#     losses = []
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad(set_to_none=True)
#         loss = nn.functional.mse_loss(model(z), y)
#         if not torch.isfinite(loss).item():
#             raise FloatingPointError(f"Nonfinite decoder loss at epoch {epoch + 1}.")
#         loss.backward()
#         optimizer.step()
#         losses.append(float(loss.detach().item()))
#         if (epoch + 1) % 250 == 0 or epoch + 1 == epochs:
#             print(f"    decoder {epoch + 1}/{epochs}: train MSE={losses[-1]:.6f}", flush=True)
#     return model, losses


# def predict_decoder(model, z):
#     model.eval()
#     device = next(model.parameters()).device
#     predictions = []
#     with torch.inference_mode():
#         for start in range(0, len(z), PREDICT_BATCH_SIZE):
#             inputs = torch.as_tensor(z[start:start + PREDICT_BATCH_SIZE],
#                                       dtype=torch.float32, device=device)
#             predictions.append(model(inputs).cpu().numpy())
#     return np.concatenate(predictions, axis=0)


# def score_r2(y, predictions):
#     if y.shape != predictions.shape or len(y) < 2 or not np.isfinite(predictions).all():
#         raise ValueError("Invalid predictions for R2.")
#     values = np.atleast_1d(r2_score(y, predictions, multioutput="raw_values", force_finite=True))
#     constant = np.flatnonzero(np.all(y == y[0], axis=0)).tolist()
#     return dict(mean_r2=float(values.mean()), per_output_r2=values.tolist(),
#                 constant_target_columns=constant)


# def run_mode(mode, x_train, x_valid, y_train, y_valid, x_eval, eval_split, eval_key,
#              out_root, device, epochs, monitor_every, decoder_epochs):
#     seed_all(SEED)
#     stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
#     out = out_root / f"{mode}_{stamp}"
#     out.mkdir(parents=True, exist_ok=False)
#     print(f"\n===== tile_normalize={mode} -> {out} =====", flush=True)

#     model = NeuralJigsaw(
#         window_size=WINDOW_SIZE, tile_gap=TILE_GAP, output_dimension=LATENT_DIM,
#         num_hidden_units=ENCODER_HIDDEN, head_hidden_units=HEAD_HIDDEN,
#         batch_size=PUZZLE_BATCH_SIZE, max_epochs=epochs, learning_rate=PUZZLE_LR,
#         weight_decay=PUZZLE_WEIGHT_DECAY, dropout=PUZZLE_DROPOUT, normalize=True,
#         tile_normalize=mode, augment_gain_jitter=AUGMENT_GAIN_JITTER,
#         augment_noise_std=AUGMENT_NOISE_STD, permutations_per_window=PERMUTATIONS_PER_WINDOW,
#         device=device, random_state=SEED, verbose=True,
#     )
#     save_json(out / "run_config.json", dict(session=SESSION, mode=mode, encoder=model.get_params(),
#                                             monitor_every=monitor_every, decoder_epochs=decoder_epochs))

#     monitor = make_monitor(x_valid, y_valid, rng_seed=SEED + 500000)
#     started = time.perf_counter()
#     model.fit(x_train, monitor_fn=monitor, monitor_every=monitor_every, select_best=True)
#     model.save(out / "encoder.pt")
#     save_json(out / "history.json", model.history_)
#     save_json(out / "checkpoint_selection.json",
#               dict(best_epoch=model.best_epoch_, best_monitor_r2=model.best_score_,
#                    last_epoch=epochs, monitor_every=monitor_every))

#     min_gap = model.training_span
#     train_puzzle = run_puzzle_eval(model, x_train, "train", "train_data", out, min_gap)
#     eval_puzzle = run_puzzle_eval(model, x_eval, eval_split, eval_key, out, min_gap)

#     z_train, idx_train = model.transform(x_train, pad=PAD_TRANSFORM,
#                                          batch_size=TRANSFORM_BATCH_SIZE, return_indices=True)
#     z_valid, idx_valid = model.transform(x_valid, pad=PAD_TRANSFORM,
#                                          batch_size=TRANSFORM_BATCH_SIZE, return_indices=True)
#     for z, indices in ((z_train, idx_train), (z_valid, idx_valid)):
#         if z.shape != (len(indices), LATENT_DIM) or not np.isfinite(z).all():
#             raise ValueError("Invalid NeuralJigsaw embeddings.")
#     yt, yv = y_train[idx_train], y_valid[idx_valid]
#     np.savez_compressed(out / "embeddings.npz", Z_train=z_train, Z_valid=z_valid,
#                         Y_train=yt, Y_valid=yv, train_indices=idx_train, valid_indices=idx_valid)

#     decoder, losses = train_decoder(z_train, yt, device, decoder_epochs)
#     train_pred = predict_decoder(decoder, z_train)
#     valid_pred = predict_decoder(decoder, z_valid)
#     scores = dict(train=score_r2(yt, train_pred), validation=score_r2(yv, valid_pred))
#     save_json(out / "R2.json", scores)
#     np.save(out / "decoder_train_mse.npy", np.asarray(losses))
#     torch.save(dict(state_dict={k: v.detach().cpu() for k, v in decoder.state_dict().items()},
#                     input_dim=LATENT_DIM, output_dim=yt.shape[1], hidden=DECODER_HIDDEN,
#                     dropout=DECODER_DROPOUT), out / "decoder.pt")

#     elapsed = time.perf_counter() - started
#     save_json(out / "timing.json", dict(total_seconds=elapsed))
#     print(f"[{mode}] puzzle {eval_split}={eval_puzzle['accuracy_percent']:.2f}% "
#           f"(train={train_puzzle['accuracy_percent']:.2f}%, chance={eval_puzzle['chance_accuracy_percent']:.2f}%) "
#           f"| R2 train={scores['train']['mean_r2']:.4f} valid={scores['validation']['mean_r2']:.4f} "
#           f"| best_epoch={model.best_epoch_}/{epochs} ({elapsed:.0f}s)", flush=True)

#     return dict(mode=mode, out_dir=str(out), best_epoch=model.best_epoch_,
#                 best_monitor_r2=model.best_score_,
#                 puzzle_train_accuracy_percent=train_puzzle["accuracy_percent"],
#                 puzzle_train_cross_entropy=train_puzzle["cross_entropy"],
#                 puzzle_eval_split=eval_split,
#                 puzzle_eval_accuracy_percent=eval_puzzle["accuracy_percent"],
#                 puzzle_eval_cross_entropy=eval_puzzle["cross_entropy"],
#                 chance_accuracy_percent=eval_puzzle["chance_accuracy_percent"],
#                 r2_train=scores["train"]["mean_r2"], r2_valid=scores["validation"]["mean_r2"])


# def plot_comparison(summary, out_root):
#     order = [row["mode"] for row in summary]
#     by_mode = {row["mode"]: row for row in summary}
#     modes = [m for m in MODES if m in by_mode]  # canonical left-to-right order for the figure
#     x = np.arange(len(modes))
#     fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

#     axes[0].bar(x - 0.18, [by_mode[m]["puzzle_train_accuracy_percent"] for m in modes], width=0.36, label="train")
#     axes[0].bar(x + 0.18, [by_mode[m]["puzzle_eval_accuracy_percent"] for m in modes], width=0.36,
#                label=by_mode[modes[0]]["puzzle_eval_split"])
#     axes[0].axhline(by_mode[modes[0]]["chance_accuracy_percent"], color="black", linestyle="--",
#                     linewidth=1, label="chance")
#     axes[0].set_xticks(x, modes)
#     axes[0].set_ylabel("Puzzle accuracy (%)")
#     axes[0].set_title("Puzzle-solving accuracy")
#     axes[0].legend()

#     axes[1].bar(x - 0.18, [by_mode[m]["r2_train"] for m in modes], width=0.36, label="train")
#     axes[1].bar(x + 0.18, [by_mode[m]["r2_valid"] for m in modes], width=0.36, label="validation")
#     axes[1].axhline(0, color="black", linewidth=0.8)
#     axes[1].set_xticks(x, modes)
#     axes[1].set_ylabel("Decoder R2")
#     axes[1].set_title("Downstream behavior R2  <- what actually matters")
#     axes[1].legend()

#     fig.suptitle(f"{SESSION} | NeuralJigsaw tile_normalize comparison "
#                 f"(best valid R2: {order[0]})")
#     fig.savefig(out_root / "tile_normalize_comparison.png", dpi=250)
#     plt.close(fig)


# def main(modes, eval_split, epochs, monitor_every, decoder_epochs):
#     if epochs < 1 or LATENT_DIM < 1:
#         raise ValueError("Bad hyperparameters.")
#     x_train, x_valid, y_train, y_valid = load_data(NPZ_PATH)
#     span = training_span(WINDOW_SIZE, TILE_GAP)
#     x_eval, actual_split, eval_key = load_puzzle_eval_data(NPZ_PATH, eval_split, span, x_train.shape[1])
#     OUT_ROOT.mkdir(parents=True, exist_ok=True)
#     device_name = "cuda" if (DEVICE == "cuda_if_available" and torch.cuda.is_available()) else DEVICE
#     print(f"Data: {NPZ_PATH} | shapes: train={x_train.shape} valid={x_valid.shape} "
#           f"| device={device_name} | modes={list(modes)}", flush=True)

#     summary = []
#     for mode in modes:
#         row = run_mode(mode, x_train, x_valid, y_train, y_valid, x_eval, actual_split, eval_key,
#                        OUT_ROOT, DEVICE, epochs, monitor_every, decoder_epochs)
#         summary.append(row)
#         cleanup()

#     summary.sort(key=lambda row: row["r2_valid"], reverse=True)
#     save_json(OUT_ROOT / "sweep_summary.json", summary)
#     with (OUT_ROOT / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
#         writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
#         writer.writeheader()
#         writer.writerows(summary)
#     plot_comparison(summary, OUT_ROOT)

#     print("\n===== SUMMARY (best validation R2 first) =====", flush=True)
#     for row in summary:
#         print(f"{row['mode']:8s} | puzzle {row['puzzle_eval_split']}={row['puzzle_eval_accuracy_percent']:6.2f}% "
#               f"(chance {row['chance_accuracy_percent']:.2f}%) | R2 train={row['r2_train']:.4f} "
#               f"valid={row['r2_valid']:.4f} | best_epoch={row['best_epoch']}/{epochs}", flush=True)
#     print(f"\nBest by validation R2: {summary[0]['mode']}", flush=True)
#     print(f"(Best by puzzle accuracy alone might be a DIFFERENT mode -- that is exactly the "
#           f"discrepancy this script is meant to expose. Trust the R2 column.)", flush=True)
#     print(f"Saved: {OUT_ROOT}", flush=True)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description=__doc__,
#                                      formatter_class=argparse.RawDescriptionHelpFormatter)
#     parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
#     parser.add_argument("--eval-split", choices=("auto", "test", "validation"), default=PUZZLE_EVAL_SPLIT)
#     parser.add_argument("--quick", action="store_true",
#                         help="Fast smoke test: few encoder/decoder epochs, frequent monitoring. "
#                              "Run this once before committing to the full sweep.")
#     parser.add_argument("--epochs", type=int, default=None, help="Override PUZZLE_EPOCHS.")
#     options = parser.parse_args()
#     if options.quick:
#         run_epochs = options.epochs or 200
#         run_monitor_every = 10
#         run_decoder_epochs = 200
#     else:
#         run_epochs = options.epochs or PUZZLE_EPOCHS
#         run_monitor_every = max(1, run_epochs // 40)
#         run_decoder_epochs = DECODER_EPOCHS
#     main(options.modes, options.eval_split, run_epochs, run_monitor_every, run_decoder_epochs)

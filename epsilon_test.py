#!/usr/bin/env python3
"""Eight independent adversarial CEBRA + MLP runs; exactly ONE result CSV.

Place this file beside the extracted CEBRA-adaptive-epsilon directory and run:
    python -u compare_adaptive_epsilon_CCO12.py

Optional overrides:
    --session C-CO16 --data /path/session.npz --fork /path/CEBRA-adaptive-epsilon
    --out /path/results.csv --eval-split test --device cuda --overwrite

Input: train_data/train_label and valid_data/valid_label (or test_data/test_label).
All hyperparameters below are shared across the eight runs, except norm and mode.
Only the CSV is written: no checkpoints, embeddings, plots, or log files.
The CSV is flushed after each run, preserving completed rows if interrupted.
No validation/test labels are used for encoder/decoder fitting or model selection.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import csv
import gc
import inspect
import os
from pathlib import Path
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parent
SESSION = "C-CO12"
PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
FORK_DIR = ROOT / "CEBRA-adaptive-epsilon"
SEED = 42
DEVICE = "cuda_if_available"

# CEBRA settings: edit these to match your existing encoder experiment.
MODEL_ARCHITECTURE = "offset10-model"
LATENT_DIM = 64
ENCODER_HIDDEN = 64
CEBRA_BATCH_SIZE = 512
CEBRA_ITERATIONS = 3000
CEBRA_LR = 3e-4
CEBRA_TEMPERATURE = 1.0
CEBRA_CONDITIONAL = "time_delta"
TIME_OFFSETS = 10
PAD_TRANSFORM = True  # One embedding/label per original bin, per split.
TRANSFORM_BATCH_SIZE = 2048

# Same decoder as the supplied script: full-batch Adam, final epoch, no scaling.
DECODER_EPOCHS = 2500
DECODER_HIDDEN = 64
DECODER_DROPOUT = 0.4
DECODER_LR = 1e-3
PREDICT_BATCH_SIZE = 8192

MODES = ("fixed", "positive", "gain", "hybrid")
NORMS = ("linf", "l2")
# These numeric defaults are NOT equivalent attack strengths across norms.
EPSILON = {"linf": 0.2, "l2": 0.2}
ALPHA = {"linf": 0.04, "l2": 0.04}
PGD_STEPS = 10
WARMUP_STEPS = 100  # Included within CEBRA_ITERATIONS, identical in all 8 runs.
POSITIVE_QUANTILE = 0.10
POSITIVE_SCALE = 0.5
POSITIVE_CAP_SCALE = 1.0
CALIBRATION_BATCHES = 32
EPSILON_MIN = 1e-6
# None: gain cap = 4 * initial epsilon; hybrid cap = POSITIVE_CAP_SCALE * Q.
EPSILON_MAX = {"linf": None, "l2": None}
GAIN_TARGET = 0.05
GAIN_EMA_DECAY = 0.9
GAIN_RATE = 0.05
GAIN_INTERVAL = 10
GAIN_DEADBAND = 0.10


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


def load_data(path, split="valid"):
    keys = ("train_data", f"{split}_data", "train_label", f"{split}_label")
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in keys if key not in data]
        if missing:
            raise KeyError(f"Missing NPZ keys {missing}; available keys: {data.files}")
        arrays = [np.asarray(data[key], dtype=np.float32) for key in keys]
    x_train, x_eval, y_train, y_eval = arrays
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_eval.ndim == 1:
        y_eval = y_eval[:, None]
    arrays = (x_train, x_eval, y_train, y_eval)
    for name, value in zip(keys, arrays):
        if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite nonempty 2D array; got {value.shape}.")
    if len(x_train) != len(y_train) or len(x_eval) != len(y_eval):
        raise ValueError("Feature/label time lengths do not match.")
    if x_train.shape[1] != x_eval.shape[1] or y_train.shape[1] != y_eval.shape[1]:
        raise ValueError("Train/evaluation channel or label dimensions do not match.")
    if min(len(x_train), len(x_eval)) < 2:
        raise ValueError("R2 requires at least two samples per split.")
    return tuple(np.ascontiguousarray(value) for value in arrays)


def load_cebra_fork(path):
    path = path.expanduser().resolve()
    if not (path / "cebra" / "__init__.py").is_file():
        raise FileNotFoundError(f"CEBRA package not found in {path}; set FORK_DIR or --fork.")
    sys.path.insert(0, str(path))
    import cebra
    origin = Path(cebra.__file__).resolve()
    if path not in origin.parents:
        raise ImportError(f"Wrong CEBRA imported: {origin}; expected {path}.")
    parameters = inspect.signature(cebra.CEBRA.__init__).parameters
    if "adv_epsilon_mode" not in parameters:
        raise ImportError("This fork lacks adaptive epsilon. Use the supplied CEBRA-adaptive-epsilon package.")
    print(f"CEBRA: {origin}", flush=True)
    return cebra


def seed_loader_generators(loader, seed):
    """This fork seeds private samplers from entropy; explicitly reseed them.

    Called after model/loader construction, before calibration or any training.
    Traversal is stable so all eight runs start from the same sampler states.
    """
    seen = set()
    counter = 0

    def visit(obj):
        nonlocal counter
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.Generator):
            obj.manual_seed(seed + counter)
            counter += 1
        elif isinstance(obj, dict):
            for key in sorted(obj, key=str):
                visit(obj[key])
        elif isinstance(obj, (tuple, list)):
            for value in obj:
                visit(value)
        elif obj is loader or type(obj).__module__.startswith("cebra.distributions"):
            for key, value in sorted(vars(obj).items()):
                if key != "dataset":
                    visit(value)

    visit(loader)


def seeded_cebra_class(cebra, seed):
    class SeededCEBRA(cebra.CEBRA):
        def _prepare_fit(self, *args, **kwargs):
            state = super()._prepare_fit(*args, **kwargs)
            seed_loader_generators(state[2], seed + 200000)
            return state
    return SeededCEBRA


def transform_in_chunks(model, x):
    """Preserve the full transform's temporal context across chunk boundaries."""
    offset = model.offset_
    left, right = offset.left, offset.right - 1
    first = 0 if PAD_TRANSFORM else left
    stop = len(x) if PAD_TRANSFORM else len(x) - right
    if len(x) < len(offset) or stop - first < 2:
        raise ValueError("Split is too short for the encoder receptive field and R2.")
    batch_size = max(TRANSFORM_BATCH_SIZE, len(offset))
    chunks = []
    for start in range(first, stop, batch_size):
        end = min(start + batch_size, stop)
        lo, hi = max(0, start - left), min(len(x), end + right)
        # Without padding, keep >=2 output bins: this fork squeezes the time
        # dimension when a convolutional transform produces exactly one bin.
        minimum_input = len(offset) + (not PAD_TRANSFORM)
        lo = max(0, min(lo, hi - minimum_input))
        z = model.transform(x[lo:hi])
        origin = lo if PAD_TRANSFORM else lo + left
        chunks.append(z[start - origin:end - origin])
    z = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
    if len(z) != stop - first or not np.isfinite(z).all():
        raise ValueError("Invalid or misaligned embeddings from transform.")
    return z, np.arange(first, stop)


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


def train_decoder(z_train, y_train, device, epochs, seed):
    seed_all(seed + 100000)
    model = Decoder(z_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=DECODER_LR)
    z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    model.train()
    last_loss = float("nan")
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(z), y)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Nonfinite decoder loss at epoch {epoch + 1}.")
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().item())
        if (epoch + 1) % 250 == 0 or epoch + 1 == epochs:
            print(f"Decoder {epoch + 1}/{epochs}: train MSE={last_loss:.6f}", flush=True)
    return model, last_loss


def predict_decoder(model, z):
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(z), PREDICT_BATCH_SIZE):
            values = torch.as_tensor(z[start:start + PREDICT_BATCH_SIZE],
                                     dtype=torch.float32, device=device)
            predictions.append(model(values).cpu().numpy())
    result = np.concatenate(predictions)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite decoder predictions.")
    return result


def scores(y, prediction):
    # Preserve undefined R2 for constant targets rather than replacing it with 0/1.
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.atleast_1d(r2_score(y, prediction, multioutput="raw_values", force_finite=False))
    return raw, float(raw.mean())


def run_one(cebra_type, arrays, args, norm, mode, device):
    x_train, x_eval, y_train, y_eval = arrays
    seed_all(args.seed)
    model = cebra_type(
        model_architecture=MODEL_ARCHITECTURE, device=device,
        batch_size=args.encoder_batch_size, learning_rate=CEBRA_LR,
        temperature=CEBRA_TEMPERATURE, temperature_mode="constant",
        output_dimension=LATENT_DIM, num_hidden_units=ENCODER_HIDDEN,
        max_iterations=args.iterations, conditional=CEBRA_CONDITIONAL,
        time_offsets=TIME_OFFSETS, distance="cosine", verbose=args.verbose,
        pad_before_transform=PAD_TRANSFORM,
        training_mode="adversarial", attack_norm=norm,
        adv_epsilon=EPSILON[norm], adv_alpha=ALPHA[norm], adv_steps=PGD_STEPS,
        adv_epsilon_mode=mode, adv_warmup_steps=args.warmup,
        adv_positive_quantile=POSITIVE_QUANTILE,
        adv_positive_scale=POSITIVE_SCALE, adv_positive_cap_scale=POSITIVE_CAP_SCALE,
        adv_calibration_batches=CALIBRATION_BATCHES,
        adv_epsilon_min=EPSILON_MIN, adv_epsilon_max=EPSILON_MAX[norm],
        adv_gain_target=GAIN_TARGET, adv_gain_ema_decay=GAIN_EMA_DECAY,
        adv_gain_rate=GAIN_RATE, adv_gain_interval=GAIN_INTERVAL,
        adv_gain_deadband=GAIN_DEADBAND,
    )
    started = time.perf_counter()

    def progress(step, solver):
        if (step + 1) % 250 == 0 or step + 1 == args.iterations:
            print(f"CEBRA {step + 1}/{args.iterations}: "
                  f"epsilon={solver.epsilon_state['epsilon']:.6g}", flush=True)

    # The callback only prints. No logdir, save(), or checkpoint path is supplied.
    model.fit(x_train, y_train, callback=progress, callback_frequency=1)
    encoder_seconds = time.perf_counter() - started
    z_train, train_indices = transform_in_chunks(model, x_train)
    z_eval, eval_indices = transform_in_chunks(model, x_eval)
    y_train, y_eval = y_train[train_indices], y_eval[eval_indices]
    log, state = model.solver_.log, model.solver_.epsilon_state
    attacked = np.asarray(log["adv_warmup"]) == 0
    used = np.asarray(log["adv_epsilon"])[attacked]
    gain = np.asarray(log["adv_gain"])[attacked]
    calibration = state["calibration"] or {}
    details = dict(
        epsilon_first=float(used[0]), epsilon_last=float(used[-1]),
        epsilon_mean=float(used.mean()), epsilon_min_used=float(used.min()),
        epsilon_max_used=float(used.max()), epsilon_next=state["epsilon"],
        epsilon_cap=state["maximum"], positive_distance_quantile=calibration.get("distance"),
        gain_mean=float(gain.mean()), gain_last=float(gain[-1]),
        gain_ema_last=state["gain_ema"], encoder_seconds=encoder_seconds,
        n_train_decoded=len(y_train), n_eval_decoded=len(y_eval),
    )
    del model
    cleanup()
    started = time.perf_counter()
    decoder, last_mse = train_decoder(z_train, y_train, device, args.decoder_epochs, args.seed)
    details["decoder_seconds"] = time.perf_counter() - started
    details["decoder_last_train_mse"] = last_mse
    train_raw, details["r2_train_mean"] = scores(y_train, predict_decoder(decoder, z_train))
    eval_raw, details["r2_eval_mean"] = scores(y_eval, predict_decoder(decoder, z_eval))
    details.update({f"r2_train_y{i}": float(value) for i, value in enumerate(train_raw)})
    details.update({f"r2_eval_y{i}": float(value) for i, value in enumerate(eval_raw)})
    details["constant_eval_targets"] = int(np.sum(np.var(y_eval.astype(np.float64), axis=0) == 0))
    return details


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default=SESSION)
    parser.add_argument("--data", type=Path, help="Default: PERICH_DATA_DIR / SESSION.npz")
    parser.add_argument("--fork", type=Path, default=FORK_DIR)
    parser.add_argument("--out", type=Path, help="Default: beside this script, adaptive_epsilon_SESSION_8models.csv")
    parser.add_argument("--eval-split", choices=("valid", "test"), default="valid")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--iterations", type=int, default=CEBRA_ITERATIONS)
    parser.add_argument("--decoder-epochs", type=int, default=DECODER_EPOCHS)
    parser.add_argument("--encoder-batch-size", type=int, default=CEBRA_BATCH_SIZE)
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--verbose", action="store_true", help="Show CEBRA tqdm in addition to periodic stdout.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing result CSV.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.iterations <= args.warmup or args.warmup < 0:
        raise ValueError("Require 0 <= warmup < iterations so all eight runs actually use PGD.")
    if min(args.decoder_epochs, args.encoder_batch_size, args.threads) < 1:
        raise ValueError("Epochs, batch size, and thread count must be positive.")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "cuda_if_available" else args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    torch.set_num_threads(args.threads)
    data_path = (args.data or PERICH_DATA_DIR / f"{args.session}.npz").expanduser().resolve()
    output = (args.out or ROOT / f"adaptive_epsilon_{args.session}_8models.csv").expanduser().resolve()
    if output.suffix.lower() != ".csv":
        raise ValueError("--out must name a .csv file.")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; choose --out or explicitly use --overwrite.")
    arrays = load_data(data_path, args.eval_split)
    cebra = load_cebra_fork(args.fork)
    cebra_type = seeded_cebra_class(cebra, args.seed)
    n_targets = arrays[2].shape[1]
    evaluation = "validation" if args.eval_split == "valid" else "test"
    common = dict(
        session=args.session, data_path=str(data_path), evaluation_split=evaluation,
        seed=args.seed, decoder_seed=args.seed + 100000, device=device,
        architecture=MODEL_ARCHITECTURE, latent_dim=LATENT_DIM,
        encoder_hidden=ENCODER_HIDDEN, encoder_batch_size=args.encoder_batch_size,
        encoder_iterations=args.iterations, encoder_lr=CEBRA_LR,
        temperature=CEBRA_TEMPERATURE, conditional=CEBRA_CONDITIONAL,
        time_offsets=TIME_OFFSETS, pad_transform=PAD_TRANSFORM,
        decoder_epochs=args.decoder_epochs, decoder_hidden=DECODER_HIDDEN,
        decoder_dropout=DECODER_DROPOUT, decoder_lr=DECODER_LR,
        decoder_training="full_batch_final_epoch", warmup_steps=args.warmup,
        pgd_steps=PGD_STEPS, positive_q=POSITIVE_QUANTILE,
        positive_scale=POSITIVE_SCALE, positive_cap_scale=POSITIVE_CAP_SCALE,
        calibration_batches=CALIBRATION_BATCHES, gain_target=GAIN_TARGET,
        gain_ema_decay=GAIN_EMA_DECAY, gain_rate=GAIN_RATE,
        gain_interval=GAIN_INTERVAL, gain_deadband=GAIN_DEADBAND,
        epsilon_floor=EPSILON_MIN, input_channels=arrays[0].shape[1],
        n_train=len(arrays[0]), n_eval=len(arrays[1]), n_targets=n_targets,
    )
    results_fields = [
        "r2_eval_mean", "delta_r2_eval_vs_fixed", "r2_train_mean",
        *[f"r2_eval_y{i}" for i in range(n_targets)],
        *[f"r2_train_y{i}" for i in range(n_targets)],
        "constant_eval_targets", "epsilon_first", "epsilon_last", "epsilon_mean",
        "epsilon_min_used", "epsilon_max_used", "epsilon_next", "epsilon_cap",
        "positive_distance_quantile", "gain_mean", "gain_last", "gain_ema_last",
        "decoder_last_train_mse", "n_train_decoded", "n_eval_decoded",
        "encoder_seconds", "decoder_seconds", "total_seconds", "error",
    ]
    fields = ["run", "norm", "mode", "status", *results_fields,
              "epsilon_config", "alpha_config", "epsilon_max_config", "started_utc", *common]
    output.parent.mkdir(parents=True, exist_ok=True)
    failures, baseline = 0, {}
    print(f"Session={args.session}; split={evaluation}; device={device}; 8 encoders + 8 decoders", flush=True)
    print(f"Only output file: {output}", flush=True)
    with output.open("w" if args.overwrite else "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        handle.flush()
        for number, (norm, mode) in enumerate(((n, m) for n in NORMS for m in MODES), 1):
            row = dict(common, run=number, norm=norm, mode=mode, status="error",
                       epsilon_config=EPSILON[norm], alpha_config=ALPHA[norm],
                       epsilon_max_config=EPSILON_MAX[norm],
                       started_utc=datetime.now(timezone.utc).isoformat())
            print(f"\n[{number}/8] {norm} / {mode}", flush=True)
            started = time.perf_counter()
            try:
                row.update(run_one(cebra_type, arrays, args, norm, mode, device))
                row["status"] = "ok"
                if mode == "fixed":
                    baseline[norm] = row["r2_eval_mean"]
                if norm in baseline:
                    row["delta_r2_eval_vs_fixed"] = row["r2_eval_mean"] - baseline[norm]
                print(f"R2 {evaluation} = {row['r2_eval_mean']:.6f}", flush=True)
            except Exception as error:
                failures += 1
                row["error"] = f"{type(error).__name__}: {error}"
                print(f"FAILED: {row['error']}", flush=True)
            finally:
                row["total_seconds"] = time.perf_counter() - started
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
                cleanup()
    print(f"\nFinished: {8 - failures}/8 successful; results: {output}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

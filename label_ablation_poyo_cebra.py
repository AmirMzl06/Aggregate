"""Label ablation: POYO independently first, then unchanged CEBRA + original MLP.

Run: python label_ablation_poyo_cebra.py --bin-size-ms 10 --subset-labels 1
10 ms must be the bin width of your processed NPZ; see README_FA.md.
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from poyo_binned import (POYOConfig, EventStore, resolve_bin_seconds, infer_segments,
                         to_counts, train_poyo)

ROOT = Path(__file__).resolve().parent
PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
TARGET_SESSION = "C-CO12"
N_SUBSET_LABELS = 1
SEED = 42
CEBRA = None
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = None

# The following CEBRA and MLP settings/functions are copied from your upload.
LATENT_DIM = 64

HIDDEN = 64

BATCH_SIZE = 2048

CEBRA_MAX_ITER = 3000

TEMPERATURE = 0.4

MODEL_ARCH = "offset36-model-more-dropout"

DEVICE = "cuda_if_available"

OFFSET = 1

CONDITIONAL = "time_delta"

DECODER_HIDDEN_DIM = 64

DECODER_DROPOUT = 0.4

DECODER_EPOCHS = 2500

DECODER_BATCH_SIZE = 256

DECODER_LR = 1e-3

DECODER_WEIGHT_DECAY = 1e-4

DECODER_PRINT_EVERY = 500

def seed_all(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def compute_r2(Y_true: np.ndarray, Y_pred: np.ndarray):
    r2_each = np.asarray(
        r2_score(Y_true, Y_pred, multioutput="raw_values"),
        dtype=float,
    )
    return r2_each, float(np.mean(r2_each))

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.net(x)

def train_shared_decoder(display_name, Z_train, Z_test, Y_train, Y_test, save_path):
    Z_train, Ytr = align_embedding_and_labels(Z_train, Y_train)
    Z_test, Yte = align_embedding_and_labels(Z_test, Y_test)

    seed_all(SEED)
    decoder = TwoLayerMLP(
        input_dim=Z_train.shape[1],
        hidden_dim=DECODER_HIDDEN_DIM,
        output_dim=Ytr.shape[1],
        dropout_rate=DECODER_DROPOUT,
    ).to(TORCH_DEVICE)

    ds = TensorDataset(
        torch.from_numpy(Z_train.astype(np.float32)),
        torch.from_numpy(Ytr.astype(np.float32)),
    )
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        ds,
        batch_size=DECODER_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=g,
    )

    opt = torch.optim.Adam(
        decoder.parameters(),
        lr=DECODER_LR,
        weight_decay=DECODER_WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss()

    print(f"\nDecoder {display_name}: {DECODER_EPOCHS} epochs")
    for epoch in range(1, DECODER_EPOCHS + 1):
        decoder.train()
        for xb, yb in loader:
            xb = xb.to(TORCH_DEVICE)
            yb = yb.to(TORCH_DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(decoder(xb), yb)
            loss.backward()
            opt.step()
        if epoch == 1 or epoch % DECODER_PRINT_EVERY == 0 or epoch == DECODER_EPOCHS:
            print(f"{display_name}: decoder epoch {epoch}/{DECODER_EPOCHS}")

    decoder.eval()
    with torch.no_grad():
        pred = decoder(torch.from_numpy(Z_test.astype(np.float32)).to(TORCH_DEVICE)).cpu().numpy()

    r2_each, mean_r2 = compute_r2(Yte, pred)
    torch.save(decoder.state_dict(), save_path)

    print(f"{display_name} R2 per label:", r2_each)
    print(f"{display_name} Mean R2 = {mean_r2:.6f}")

    del decoder, opt
    cleanup()
    return r2_each, mean_r2

def build_cebra():
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=OFFSET,
        conditional=CONDITIONAL,
        max_iterations=CEBRA_MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        device=DEVICE,
        verbose=True,
    )

def run_cebra(condition_name, X_train, X_test, Y_train_cond, Y_test_cond):
    print("\n" + "#" * 100)
    print(f"CEBRA -- {condition_name}")
    print("#" * 100)
    print("labels used during CEBRA training:", Y_train_cond.shape[1])

    seed_all(SEED)
    model = build_cebra()
    model.fit(X_train, Y_train_cond)

    model_path = MODELS_DIR / f"cebra_{condition_name}.pt"
    model.save(str(model_path))

    Z_train = np.asarray(model.transform(X_train.astype(np.float32)), dtype=np.float32)
    Z_test = np.asarray(model.transform(X_test.astype(np.float32)), dtype=np.float32)

    r2_each, mean_r2 = train_shared_decoder(
        f"CEBRA {condition_name}",
        Z_train,
        Z_test,
        Y_train_cond,
        Y_test_cond,
        MODELS_DIR / f"cebra_{condition_name}_decoder.pt",
    )

    del model
    cleanup()
    return {
        "method": "CEBRA",
        "condition": condition_name,
        "n_train_labels": int(Y_train_cond.shape[1]),
        "r2_each": r2_each,
        "mean_r2": mean_r2,
    }


def align_embedding_and_labels(Z, Y):
    # Same rows as the original pipeline, but never silently truncate misalignment.
    if Z.ndim != 2 or Y.ndim != 2 or len(Z) != len(Y):
        raise ValueError(f"CEBRA embedding/label mismatch: {Z.shape} vs {Y.shape}")
    return Z, Y


def load_original_cebra(explicit_dir=None):
    global CEBRA
    candidates = ([Path(explicit_dir)] if explicit_dir else
                  [ROOT / "CEBRA-orginal", ROOT / "CEBRA-original",
                   ROOT.parent / "CEBRA-orginal", ROOT.parent / "CEBRA-original"])
    chosen = next((p.resolve() for p in candidates if (p / "cebra" / "__init__.py").is_file()), None)
    if chosen is None:
        raise FileNotFoundError("Original CEBRA checkout not found. Use --cebra-dir /path/to/CEBRA-orginal")
    for name in list(sys.modules):
        if name == "cebra" or name.startswith("cebra."):
            del sys.modules[name]
    sys.path.insert(0, str(chosen))
    import cebra
    CEBRA = cebra.CEBRA
    print("Using ORIGINAL CEBRA:", cebra.__file__, flush=True)
    return str(chosen)


def load_data(path):
    with np.load(path, allow_pickle=False) as f:
        keys = ("train_data", "valid_data", "train_label", "valid_label")
        arrays = [np.asarray(f[k], dtype=np.float32) for k in keys]
        meta_keys = ("bin_size_s", "bin_size_seconds", "bin_size_ms", "dt",
                     "sampling_rate_hz", "sampling_rate", "train_trial_ids",
                     "train_trial_id", "valid_trial_ids", "valid_trial_id",
                     "train_timestamps", "valid_timestamps")
        metadata = {k: f[k] for k in meta_keys if k in f.files}
    xtr, xva, ytr, yva = arrays
    if ytr.ndim == 1:
        ytr = ytr[:, None]
    if yva.ndim == 1:
        yva = yva[:, None]
    for name, arr in zip(keys, (xtr, xva, ytr, yva)):
        if arr.ndim != 2 or min(arr.shape) < 1 or not np.isfinite(arr).all():
            raise ValueError(f"{name}: expected finite nonempty 2D array, got {arr.shape}")
    if len(xtr) != len(ytr) or len(xva) != len(yva):
        raise ValueError("Neural and label row counts differ")
    if xtr.shape[1] != xva.shape[1] or ytr.shape[1] != yva.shape[1]:
        raise ValueError("Train/validation channel counts differ")
    return xtr, xva, ytr, yva, metadata


def serialize(value):
    if isinstance(value, np.ndarray):
        return [serialize(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    return value


def save_results(results, out, session, n_subset):
    """Save after EVERY condition, including per-label matched comparisons."""
    n_labels = max(len(r["r2_each"]) for r in results)
    with (out / "r2_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session", "method", "condition", "n_train_labels", "mean_r2_valid",
                         "mean_r2_common_first_n", "n_common_labels", "evaluation_split"]
                        + [f"r2_label_{i}" for i in range(n_labels)])
        for r in results:
            values = r["r2_each"].tolist()
            writer.writerow([session, r["method"], r["condition"], r["n_train_labels"],
                             r["mean_r2"], float(np.mean(r["r2_each"][:n_subset])),
                             n_subset, "valid_data"] + values + [""] * (n_labels-len(values)))
    with (out / "r2_per_label_long.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session", "method", "condition", "n_train_labels", "label_index", "r2"])
        for r in results:
            for i, value in enumerate(r["r2_each"]):
                writer.writerow([session, r["method"], r["condition"], r["n_train_labels"], i, value])
    with (out / "r2_common_labels.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "label_index", "trained_first_n", "trained_all", "all_minus_first_n"])
        for method in dict.fromkeys(r["method"] for r in results):
            records = {r["condition"]: r for r in results if r["method"] == method}
            if "all_labels" in records and f"first_{n_subset}" in records:
                a = records[f"first_{n_subset}"]["r2_each"]
                b = records["all_labels"]["r2_each"]
                for i in range(n_subset):
                    writer.writerow([method, i, a[i], b[i], b[i] - a[i]])
    (out / "results.json").write_text(json.dumps(serialize(results), indent=2, allow_nan=False))


def main():
    global SEED, TARGET_SESSION, N_SUBSET_LABELS, TORCH_DEVICE, DEVICE
    global MODELS_DIR, CEBRA_MAX_ITER, DECODER_EPOCHS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=TARGET_SESSION)
    parser.add_argument("--data-dir", type=Path, default=PERICH_DATA_DIR)
    parser.add_argument("--npz", type=Path, help="Explicit session NPZ, overrides --data-dir")
    parser.add_argument("--subset-labels", type=int, default=N_SUBSET_LABELS)
    parser.add_argument("--bin-size-ms", type=float, default=None,
                        help="Width of YOUR NPZ bins. MP reference: 10 ms; no silent default")
    parser.add_argument("--input-kind", choices=("counts", "rates_hz", "poisson_rate"), default="counts")
    parser.add_argument("--spike-placement", choices=("center", "uniform"), default="center")
    parser.add_argument("--cebra-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--poyo-steps", type=int, default=3000)
    parser.add_argument("--poyo-batch-size", type=int, default=16)
    parser.add_argument("--poyo-context-bins", type=int, default=100)
    parser.add_argument("--poyo-lr", type=float, default=3e-4)
    parser.add_argument("--poyo-dim", type=int, default=64)
    parser.add_argument("--poyo-depth", type=int, default=6)
    parser.add_argument("--poyo-standardize-targets", action="store_true",
                        help="POYO only: train-label zscore; inverse-transform predictions before R2")
    parser.add_argument("--cebra-iterations", type=int, default=CEBRA_MAX_ITER)
    parser.add_argument("--decoder-epochs", type=int, default=DECODER_EPOCHS)
    parser.add_argument("--poyo-only", action="store_true", help="Run only the first two conditions")
    parser.add_argument("--check-data", action="store_true", help="Validate time/count conversion without training")
    args = parser.parse_args()
    if args.subset_labels < 1 or args.seed < 0 or args.cebra_iterations < 1 or args.decoder_epochs < 1:
        parser.error("subset-labels, iterations, epochs must be positive; seed must be nonnegative")
    SEED, TARGET_SESSION, N_SUBSET_LABELS = args.seed, args.session, args.subset_labels
    CEBRA_MAX_ITER, DECODER_EPOCHS = args.cebra_iterations, args.decoder_epochs
    DEVICE = args.device
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if DEVICE == "cuda_if_available" else DEVICE
    TORCH_DEVICE = torch.device(dev)
    path = args.npz or args.data_dir / f"{TARGET_SESSION}.npz"
    xtr, xva, ytr, yva, metadata = load_data(path)
    if N_SUBSET_LABELS > ytr.shape[1]:
        parser.error(f"Only {ytr.shape[1]} labels in this NPZ")
    dt, time_source = resolve_bin_seconds(metadata, args.bin_size_ms)
    segtr, srctr = infer_segments(len(xtr), metadata, "train", dt)
    segva, srcva = infer_segments(len(xva), metadata, "valid", dt)
    print(f"DATA: {path}\ntrain={xtr.shape}, valid={xva.shape}, labels={ytr.shape[1]}\n"
          f"dt={dt:g}s, fs={1/dt:g}Hz, source={time_source}\n"
          f"POYO train boundaries: {srctr} ({len(segtr)} segments)\n"
          f"POYO valid boundaries: {srcva} ({len(segva)} segments)\n"
          f"Input conversion={args.input_kind}; synthetic within-bin placement={args.spike_placement}", flush=True)
    ctr = to_counts(xtr, dt, args.input_kind, SEED + 1000)
    cva = to_counts(xva, dt, args.input_kind, SEED + 2000)
    train_store = EventStore(ctr, dt, segtr, args.spike_placement, SEED + 3000)
    valid_store = EventStore(cva, dt, segva, args.spike_placement, SEED + 4000)
    print(f"Synthetic event totals: train={len(train_store.units):,}, valid={len(valid_store.units):,}", flush=True)
    if args.check_data:
        print("Data checks passed; no model trained.")
        return
    config = POYOConfig(context_bins=args.poyo_context_bins, steps=args.poyo_steps,
                        batch_size=args.poyo_batch_size, eval_batch_size=args.poyo_batch_size,
                        learning_rate=args.poyo_lr, dim=args.poyo_dim, depth=args.poyo_depth,
                        standardize_targets=args.poyo_standardize_targets, seed=SEED)
    config.validate()
    # Validate import before spending time on POYO; CEBRA TRAINING is still last.
    cebra_path = None if args.poyo_only else load_original_cebra(args.cebra_dir)
    if cebra_path and (len(segtr) > 1 or len(segva) > 1):
        print("CEBRA receives the original arrays unchanged; its original pipeline does not use trial metadata.", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = args.out_dir or ROOT / f"LabelAblation_POYO_vs_CEBRA_{TARGET_SESSION}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        raise FileExistsError(f"Existing experiment in {out}; choose a new output directory")
    MODELS_DIR = out / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    cebra_params = None if CEBRA is None else build_cebra().get_params()
    experiment = dict(session=TARGET_SESSION, npz=str(path), seed=SEED,
                      n_subset_labels=N_SUBSET_LABELS, n_all_labels=ytr.shape[1],
                      bin_size_s=dt, bin_size_source=time_source, input_kind=args.input_kind,
                      spike_placement=args.spike_placement, timing_is_reconstructed=True,
                      poyo=asdict(config), cebra_dir=cebra_path, cebra_params=cebra_params,
                      cebra_decoder=dict(hidden=DECODER_HIDDEN_DIM, dropout=DECODER_DROPOUT,
                         epochs=DECODER_EPOCHS, batch_size=DECODER_BATCH_SIZE,
                         lr=DECODER_LR, weight_decay=DECODER_WEIGHT_DECAY),
                      poyo_train_segments=segtr, poyo_valid_segments=segva,
                      split="original train_data / valid_data; no new 80/20 split",
                      order=["POYO first_n", "POYO all", "CEBRA+MLP first_n", "CEBRA+MLP all"])
    (out / "config.json").write_text(json.dumps(serialize(experiment), indent=2, default=str))
    conditions = [(f"first_{N_SUBSET_LABELS}", ytr[:, :N_SUBSET_LABELS], yva[:, :N_SUBSET_LABELS]),
                  ("all_labels", ytr, yva)]
    results = []
    for condition, yt, yv in conditions:
        print(f"\nPOYO independent -- {condition}", flush=True)
        pt, pv = train_poyo(train_store, valid_store, yt, yv, TARGET_SESSION,
                            config, TORCH_DEVICE, MODELS_DIR / f"poyo_{condition}")
        each, avg = compute_r2(yv, pv)
        row = dict(method="POYO", condition=condition, n_train_labels=yt.shape[1],
                   r2_each=each, mean_r2=avg, train_mean_r2=compute_r2(yt, pt)[1])
        results.append(row)
        print(f"POYO {condition}: VALIDATION R2={avg:.6f}; per-label={each}", flush=True)
        save_results(results, out, TARGET_SESSION, N_SUBSET_LABELS)
        del pt, pv
        cleanup()
    del train_store, valid_store, ctr, cva
    cleanup()
    if not args.poyo_only:
        for condition, yt, yv in conditions:
            results.append(run_cebra(condition, xtr, xva, yt, yv))
            save_results(results, out, TARGET_SESSION, N_SUBSET_LABELS)
    print("\nFINAL VALIDATION R2 (original valid_data split)")
    for r in results:
        common = np.mean(r["r2_each"][:N_SUBSET_LABELS])
        print(f"{r['method']:<8} {r['condition']:<12}: mean={r['mean_r2']:.6f}, "
              f"common-first-{N_SUBSET_LABELS}={common:.6f}, per-label={r['r2_each']}")
    print("Saved:", out)


if __name__ == "__main__":
    main()

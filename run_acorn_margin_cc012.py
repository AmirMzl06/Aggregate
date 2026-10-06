"""Compare fixed-epsilon ACORN with margin ACORN on an existing Perich NPZ split.

Place this script beside the Acorn-margin-eps/ checkout and run:
    python -u run_acorn_margin_cc012.py
    python -u run_acorn_margin_cc012.py --seeds 42 43

Both encoders are supervised by ALL train_label columns. The behavioral MLP
also predicts ALL columns. Validation is used only for the final evaluation.
The uploaded fork is used unchanged, including its attack search behavior.
"""

import argparse
import csv
import gc
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime

import numpy as np
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parent
CEBRA_DIR = ROOT / "Acorn-margin-eps"
DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw")
ARMS = ("fixed_eps_3", "margin")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default="C-CO12")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--npz", type=Path, help="Optional exact NPZ path")
    p.add_argument("--cebra-dir", type=Path, default=CEBRA_DIR)
    p.add_argument("--out-dir", type=Path, default=ROOT / "ACORN_MARGIN_VS_FIXED_CCO12")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--architecture", default="offset36-model-more-dropout")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--encoder-hidden", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--encoder-lr", type=float, default=3e-4)
    p.add_argument("--attack-norm", choices=["linf", "l2"], default="linf")
    p.add_argument("--attack-steps", type=int, default=10)
    p.add_argument("--fixed-alpha", type=float, default=0.6, help="Fixed arm: 3/5")
    p.add_argument("--margin-max-std", type=float, default=2.0)
    p.add_argument("--margin-refine-steps", type=int, default=3)
    p.add_argument("--decoder-epochs", type=int, default=2500)
    p.add_argument("--decoder-hidden", type=int, default=64)
    p.add_argument("--decoder-dropout", type=float, default=0.4)
    p.add_argument("--decoder-lr", type=float, default=1e-3)
    p.add_argument("--predict-batch-size", type=int, default=8192)
    p.add_argument("--device", default="cuda_if_available")
    args = p.parse_args()
    positive = ("iterations", "batch_size", "latent_dim", "encoder_hidden",
                "temperature", "encoder_lr", "attack_steps", "fixed_alpha",
                "margin_max_std", "decoder_epochs", "decoder_hidden",
                "decoder_lr", "predict_batch_size")
    for key in positive:
        if getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if args.margin_refine_steps < 0 or not 0 <= args.decoder_dropout < 1:
        p.error("Invalid margin-refine-steps or decoder-dropout")
    if any(s < 0 or s >= 2**32 - 10000 for s in args.seeds):
        p.error("Seeds must be in [0, 2**32 - 10000)")
    args.seeds = list(dict.fromkeys(args.seeds))
    return args


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if hasattr(value, "detach"):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load_data(path):
    keys = ("train_data", "valid_data", "train_label", "valid_label")
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in keys if k not in data.files]
        if missing:
            raise KeyError(f"{path}: missing {missing}; available keys: {data.files}")
        arrays = [np.array(data[k], dtype=np.float32, order="C", copy=True)
                  for k in keys]
    xtr, xva, ytr, yva = arrays
    if ytr.ndim == 1:
        ytr = ytr[:, None]
    if yva.ndim == 1:
        yva = yva[:, None]
    for key, arr in zip(keys, (xtr, xva, ytr, yva)):
        if arr.ndim != 2 or min(arr.shape) < 1 or not np.isfinite(arr).all():
            raise ValueError(f"{key} must be a finite, nonempty 2D numeric array")
    if len(xtr) != len(ytr) or len(xva) != len(yva):
        raise ValueError("Neural data and labels have different lengths")
    if xtr.shape[1] != xva.shape[1] or ytr.shape[1] != yva.shape[1]:
        raise ValueError("Train/validation feature dimensions do not match")
    if min(len(xtr), len(xva)) < 2:
        raise ValueError("R2 needs at least two samples per split")
    return xtr, xva, ytr, yva


def load_fork(path):
    path = path.expanduser().resolve()
    if not (path / "cebra" / "integrations" / "sklearn" / "cebra.py").is_file():
        raise FileNotFoundError(f"Fork not found: {path}. Place Acorn-margin-eps beside this script.")
    for key in list(sys.modules):
        if key == "cebra" or key.startswith("cebra."):
            del sys.modules[key]
    sys.path.insert(0, str(path))
    importlib.invalidate_caches()
    cebra = importlib.import_module("cebra")
    actual = Path(cebra.__file__).resolve()
    if path not in actual.parents:
        raise RuntimeError(f"Wrong CEBRA imported: {actual}")
    required = {"training_mode", "adv_epsilon_mode", "adv_margin_epsilon_max",
                "adv_margin_max_std", "adv_margin_refine_steps", "adv_margin_eval_mode"}
    missing = required - set(inspect.signature(cebra.CEBRA.__init__).parameters)
    if missing:
        raise RuntimeError(f"This checkout lacks margin arguments: {sorted(missing)}")
    print(f"Using CEBRA: {actual}", flush=True)
    return cebra


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def encoder_config(args, arm):
    # Both arms use the same fork and its original reference-only attack.
    # In margin mode, the fork ignores adv_epsilon and adv_alpha.
    return dict(
        batch_size=args.batch_size,
        temperature=args.temperature,
        temperature_mode="constant",
        distance="cosine",
        model_architecture=args.architecture,
        time_offsets=1,
        max_iterations=args.iterations,
        output_dimension=args.latent_dim,
        num_hidden_units=args.encoder_hidden,
        learning_rate=args.encoder_lr,
        pad_before_transform=True,
        hybrid=False,
        training_mode="adversarial",
        attack_norm=args.attack_norm,
        adv_epsilon=3.0,
        adv_alpha=args.fixed_alpha,
        adv_steps=args.attack_steps,
        adv_epsilon_mode="constant" if arm == "fixed_eps_3" else "margin",
        adv_margin_epsilon_max=None,
        adv_margin_max_std=args.margin_max_std,
        adv_margin_refine_steps=args.margin_refine_steps,
        adv_margin_eval_mode=True,
        device=args.device,
        verbose=True,
    )


def build_decoder(in_dim, out_dim, args, device):
    # Identical layout to the earlier full-batch MLP comparison.
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, args.decoder_hidden),
        torch.nn.LayerNorm(args.decoder_hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.decoder_dropout),
        torch.nn.Linear(args.decoder_hidden, out_dim),
    ).to(device)


def train_decoder(ztr, ytr, args, seed, device):
    # Reset independently of how many random draws the encoder attack consumed.
    seed_all(seed)
    decoder = build_decoder(ztr.shape[1], ytr.shape[1], args, device)
    opt = torch.optim.Adam(decoder.parameters(), lr=args.decoder_lr)
    z = torch.as_tensor(ztr, dtype=torch.float32, device=device)
    y = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    losses = []
    decoder.train()
    for epoch in range(1, args.decoder_epochs + 1):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(decoder(z), y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite decoder MSE at epoch {epoch}")
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 500 == 0 or epoch == args.decoder_epochs:
            print(f"Decoder {epoch}/{args.decoder_epochs}: train MSE={losses[-1]:.6f}", flush=True)
    decoder.eval()
    return decoder, losses


def predict(decoder, z, device, batch_size):
    decoder.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            batch = torch.as_tensor(z[start:start + batch_size], dtype=torch.float32, device=device)
            chunks.append(decoder(batch).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def score(y, pred):
    if y.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError("Predictions have invalid shape or nonfinite values")
    per = np.asarray(r2_score(y, pred, multioutput="raw_values"), dtype=float)
    return {"mean": float(per.mean()), "per_output": per.tolist()}


def save_summary(out, rows):
    write_json(out / "results.json", rows)
    fields = ["arm", "seed", "r2_train", "r2_valid", "encoder_seconds",
              "decoder_seconds", "adv_eps_mean_last", "adv_eps_cap_last",
              "adv_frac_chance_last", "adv_frac_capped_last"]
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    aggregates = {}
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        if group:
            vals = np.array([r["r2_valid"] for r in group])
            aggregates[arm] = {
                "r2_valid_mean": float(vals.mean()),
                "r2_valid_std": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                "seeds": {str(r["seed"]): r["r2_valid"] for r in group},
            }
    write_json(out / "aggregate.json", aggregates)
    return aggregates


def main():
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    global torch
    import torch

    # Keep both training and decoding on the explicitly selected device.
    if args.device == "cuda_if_available":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    cebra = load_fork(args.cebra_dir)
    npz = (args.npz or args.data_dir / f"{args.session}.npz").expanduser().resolve()
    xtr, xva, ytr, yva = load_data(npz)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = args.out_dir.expanduser().resolve() / f"{args.session}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    fork = args.cebra_dir.expanduser().resolve()
    hashes = {name: hashlib.sha256((fork / name).read_bytes()).hexdigest()
              for name in ("cebra/solver/base.py", "cebra/integrations/sklearn/cebra.py")}
    write_json(out / "config.json", {
        "args": vars(args), "npz": npz, "fork": fork, "fork_sha256": hashes,
        "torch_version": torch.__version__, "cebra_version": getattr(cebra, "__version__", None),
        "data_shapes": {k: list(v.shape) for k, v in zip(
            ("train_data", "valid_data", "train_label", "valid_label"), (xtr, xva, ytr, yva))},
        "encoder_configs": {arm: encoder_config(args, arm) for arm in ARMS},
        "evaluation": "Clean-input R2, all label columns, uniform mean; original NPZ split",
        "preprocessing": "No extra data/label normalization; transform pads each split separately",
        "attack_note": "Fork unchanged: constant uses random start/train mode; margin uses zero start/eval search",
    })
    print(f"DATA: {npz}\ntrain={xtr.shape}, valid={xva.shape}, labels={ytr.shape[1]}", flush=True)
    print("Using NPZ train/valid splits directly; no 80/20 resplit. ALL labels are used.", flush=True)
    print(f"Both arms: adversarial, norm={args.attack_norm}, steps={args.attack_steps}, seeds={args.seeds}")
    print(f"Outputs: {out}", flush=True)
    rows = []

    for seed in args.seeds:
        for arm in ARMS:
            run = out / f"seed_{seed}" / arm
            run.mkdir(parents=True)
            config = encoder_config(args, arm)
            write_json(run / "config.json", {"seed": seed, "decoder_seed": seed + 10000, "cebra": config})
            print(f"\n{'=' * 80}\nTRAIN {arm} | seed={seed}\n{'=' * 80}", flush=True)
            seed_all(seed)
            start = time.perf_counter()
            model = cebra.CEBRA(**config)
            model.fit(xtr, ytr)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            encoder_seconds = time.perf_counter() - start
            model.save(str(run / "cebra_model.pt"))
            log = json_safe(model.solver_.log)
            write_json(run / "encoder_log.json", log)
            ztr = np.ascontiguousarray(model.transform(xtr), dtype=np.float32)
            zva = np.ascontiguousarray(model.transform(xva), dtype=np.float32)
            for name, z, y in (("train", ztr, ytr), ("valid", zva, yva)):
                if z.shape != (len(y), args.latent_dim) or not np.isfinite(z).all():
                    raise RuntimeError(f"{name} embeddings invalid or misaligned: {z.shape}, labels={y.shape}")
            np.savez_compressed(run / "embeddings.npz", train=ztr, valid=zva)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            start = time.perf_counter()
            decoder, decoder_losses = train_decoder(ztr, ytr, args, seed + 10000, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            decoder_seconds = time.perf_counter() - start
            ptr = predict(decoder, ztr, device, args.predict_batch_size)
            pva = predict(decoder, zva, device, args.predict_batch_size)
            train_r2, valid_r2 = score(ytr, ptr), score(yva, pva)
            torch.save({
                "state_dict": {k: v.detach().cpu() for k, v in decoder.state_dict().items()},
                "input_dim": ztr.shape[1], "output_dim": ytr.shape[1],
                "hidden": args.decoder_hidden, "dropout": args.decoder_dropout,
                "layout": "Linear-LayerNorm-ReLU-Dropout-Linear", "seed": seed + 10000,
            }, run / "decoder.pt")
            np.save(run / "decoder_train_mse.npy", np.asarray(decoder_losses))
            np.savez_compressed(run / "predictions.npz", train_pred=ptr, valid_pred=pva,
                                train_label=ytr, valid_label=yva)
            row = dict(arm=arm, seed=seed, r2_train=train_r2["mean"],
                       r2_valid=valid_r2["mean"], train_per_output=train_r2["per_output"],
                       valid_per_output=valid_r2["per_output"], encoder_seconds=encoder_seconds,
                       decoder_seconds=decoder_seconds)
            for key in ("adv_eps_mean", "adv_eps_median", "adv_eps_cap",
                        "adv_frac_chance", "adv_frac_capped"):
                values = log.get(key, [])
                row[key + "_last"] = values[-1] if values else None
            write_json(run / "metrics.json", row)
            rows.append(row)
            save_summary(out, rows)
            print(f"{arm}: TRAIN mean R2={train_r2['mean']:.6f} | VALID mean R2={valid_r2['mean']:.6f}")
            print("TRAIN per-output R2:", train_r2["per_output"])
            print("VALID per-output R2:", valid_r2["per_output"], flush=True)
            if arm == "margin":
                print("Last-batch margin diagnostics:", {k: v for k, v in row.items() if k.endswith("_last")}, flush=True)
            del decoder, ztr, zva, ptr, pva
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregates = save_summary(out, rows)
    print("\nFINAL CLEAN-INPUT VALIDATION R2 (ALL LABEL COLUMNS)")
    for arm, agg in sorted(aggregates.items(), key=lambda item: item[1]["r2_valid_mean"], reverse=True):
        print(f"{arm:<20s} R2={agg['r2_valid_mean']:.6f} std={agg['r2_valid_std']} seeds={agg['seeds']}")
    delta = aggregates["margin"]["r2_valid_mean"] - aggregates["fixed_eps_3"]["r2_valid_mean"]
    print(f"Margin minus fixed epsilon=3: {delta:+.6f} R2")
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()

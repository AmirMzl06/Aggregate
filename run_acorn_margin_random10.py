"""Run fixed-epsilon and margin ACORN on 10 reproducibly sampled Perich sessions.

Requirements:
  - run_acorn_margin_cc012.py beside this file
  - Acorn-margin-eps/ checkout beside this file (or pass --cebra-dir)

Default run:
    python -u run_acorn_margin_random10.py

The random session list is deterministic and saved before training. C-CO12 is
excluded by default because it was already evaluated. Finished arms are skipped
when the same command is rerun, so an interrupted suite resumes safely.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

import run_acorn_margin_cc012 as core


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw")
ARMS = core.ARMS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--cebra-dir", type=Path, default=ROOT / "Acorn-margin-eps")
    p.add_argument("--out-dir", type=Path, default=ROOT / "ACORN_MARGIN_VS_FIXED_RANDOM10")
    p.add_argument("--num-sessions", type=int, default=10)
    p.add_argument("--selection-seed", type=int, default=42)
    p.add_argument("--exclude", nargs="*", default=["C-CO12"],
                   help="Session stems excluded before random sampling")
    p.add_argument("--seed", type=int, default=42, help="Encoder/decoder experiment seed")
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--architecture", default="offset36-model-more-dropout")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--encoder-hidden", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--encoder-lr", type=float, default=3e-4)
    p.add_argument("--attack-norm", choices=["linf", "l2"], default="linf")
    p.add_argument("--attack-steps", type=int, default=10)
    p.add_argument("--fixed-alpha", type=float, default=0.6)
    p.add_argument("--margin-max-std", type=float, default=2.0)
    p.add_argument("--margin-refine-steps", type=int, default=3)
    p.add_argument("--decoder-epochs", type=int, default=2500)
    p.add_argument("--decoder-hidden", type=int, default=64)
    p.add_argument("--decoder-dropout", type=float, default=0.4)
    p.add_argument("--decoder-lr", type=float, default=1e-3)
    p.add_argument("--predict-batch-size", type=int, default=8192)
    p.add_argument("--device", default="cuda_if_available")
    p.add_argument("--fail-fast", action="store_true",
                   help="Stop at the first failed arm; default records the error and continues")
    args = p.parse_args()
    positive = ("num_sessions", "iterations", "batch_size", "latent_dim",
                "encoder_hidden", "temperature", "encoder_lr", "attack_steps",
                "fixed_alpha", "margin_max_std", "decoder_epochs", "decoder_hidden",
                "decoder_lr", "predict_batch_size")
    for key in positive:
        if getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if args.margin_refine_steps < 0 or not 0 <= args.decoder_dropout < 1:
        p.error("Invalid margin-refine-steps or decoder-dropout")
    if min(args.seed, args.selection_seed) < 0 or max(args.seed, args.selection_seed) >= 2**32 - 10000:
        p.error("Seeds must be in [0, 2**32 - 10000)")
    return args


def resolve_fork(path):
    path = path.expanduser().resolve()
    marker = Path("cebra/integrations/sklearn/cebra.py")
    if (path / marker).is_file():
        return path
    # Be forgiving about Acorn/ACORN casing, which is significant on Linux.
    matches = [p.resolve() for p in ROOT.iterdir()
               if p.is_dir() and p.name.lower() == "acorn-margin-eps" and (p / marker).is_file()]
    if len(matches) == 1:
        print(f"Requested fork was absent; using detected checkout: {matches[0]}", flush=True)
        return matches[0]
    raise FileNotFoundError(
        f"Fork not found: {path}. Pass --cebra-dir /exact/path/to/Acorn-margin-eps")


def discover_and_sample(args):
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Perich data directory not found: {data_dir}")
    excluded = {name[:-4] if name.endswith(".npz") else name for name in args.exclude}
    all_files = sorted((p.resolve() for p in data_dir.glob("*.npz")), key=lambda p: p.name)
    candidates = [p for p in all_files if p.stem not in excluded]
    if len(candidates) < args.num_sessions:
        raise ValueError(
            f"Found {len(all_files)} NPZ files, but only {len(candidates)} remain after exclusions; "
            f"cannot select {args.num_sessions}.")
    selected = random.Random(args.selection_seed).sample(candidates, args.num_sessions)
    return all_files, selected


def signature(args, selected, fork):
    return {
        "selected_sessions": [p.stem for p in selected],
        "selected_npz": [str(p) for p in selected],
        "selection_seed": args.selection_seed,
        "experiment_seed": args.seed,
        "excluded_sessions": list(args.exclude),
        "fork": str(fork),
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "architecture": args.architecture,
        "latent_dim": args.latent_dim,
        "encoder_hidden": args.encoder_hidden,
        "temperature": args.temperature,
        "encoder_lr": args.encoder_lr,
        "attack_norm": args.attack_norm,
        "attack_steps": args.attack_steps,
        "fixed_epsilon": 3.0,
        "fixed_alpha": args.fixed_alpha,
        "margin_max_std": args.margin_max_std,
        "margin_refine_steps": args.margin_refine_steps,
        "decoder_epochs": args.decoder_epochs,
        "decoder_hidden": args.decoder_hidden,
        "decoder_dropout": args.decoder_dropout,
        "decoder_lr": args.decoder_lr,
    }


def configure_core_args(args):
    # core.encoder_config expects these common names.
    return argparse.Namespace(
        batch_size=args.batch_size, temperature=args.temperature,
        architecture=args.architecture, iterations=args.iterations,
        latent_dim=args.latent_dim, encoder_hidden=args.encoder_hidden,
        encoder_lr=args.encoder_lr, attack_norm=args.attack_norm,
        attack_steps=args.attack_steps, fixed_alpha=args.fixed_alpha,
        margin_max_std=args.margin_max_std,
        margin_refine_steps=args.margin_refine_steps,
        decoder_epochs=args.decoder_epochs, decoder_hidden=args.decoder_hidden,
        decoder_dropout=args.decoder_dropout, decoder_lr=args.decoder_lr,
        predict_batch_size=args.predict_batch_size, device=args.device,
    )


def read_completed_rows(out):
    rows = []
    for metrics in sorted(out.glob("sessions/*/*/metrics.json")):
        try:
            row = json.loads(metrics.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("completed") is True:
            rows.append(row)
    return rows


def save_suite_summary(out):
    rows = read_completed_rows(out)
    core.write_json(out / "results.json", rows)
    fields = [
        "session", "arm", "seed", "neurons", "labels", "train_samples", "valid_samples",
        "r2_train", "r2_valid", "encoder_seconds", "decoder_seconds",
        "adv_eps_mean_last", "adv_eps_median_last", "adv_eps_cap_last",
        "adv_frac_chance_last", "adv_frac_capped_last",
    ]
    import csv
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["session"], r["arm"])))

    by_arm = {}
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        values = np.asarray([r["r2_valid"] for r in group], dtype=float)
        by_arm[arm] = {
            "n_completed": len(group),
            "r2_valid_mean_across_sessions": float(values.mean()) if len(values) else None,
            "r2_valid_std_across_sessions": float(values.std(ddof=1)) if len(values) > 1 else None,
            "sessions": {r["session"]: r["r2_valid"] for r in group},
        }
    paired = []
    for session in sorted({r["session"] for r in rows}):
        session_rows = {r["arm"]: r for r in rows if r["session"] == session}
        if all(arm in session_rows for arm in ARMS):
            paired.append({
                "session": session,
                "fixed_eps_3": session_rows["fixed_eps_3"]["r2_valid"],
                "margin": session_rows["margin"]["r2_valid"],
                "delta_margin_minus_fixed": (
                    session_rows["margin"]["r2_valid"] - session_rows["fixed_eps_3"]["r2_valid"]),
            })
    deltas = np.asarray([r["delta_margin_minus_fixed"] for r in paired], dtype=float)
    aggregate = {
        "by_arm": by_arm,
        "paired_sessions": paired,
        "paired_delta_mean": float(deltas.mean()) if len(deltas) else None,
        "paired_delta_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else None,
        "margin_wins": int((deltas > 0).sum()),
        "fixed_wins": int((deltas < 0).sum()),
        "ties": int((deltas == 0).sum()),
    }
    core.write_json(out / "aggregate.json", aggregate)
    return rows, aggregate


def clean_cuda(torch):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_arm(cebra, torch, device, args, cargs, session, npz, arm, out):
    run = out / "sessions" / session / arm
    metrics_path = run / "metrics.json"
    if metrics_path.is_file():
        try:
            old = json.loads(metrics_path.read_text())
            if old.get("completed") is True:
                print(f"SKIP completed: {session} / {arm} | VALID R2={old['r2_valid']:.6f}", flush=True)
                return
        except (OSError, json.JSONDecodeError):
            pass
    run.mkdir(parents=True, exist_ok=True)
    error_path = run / "error.json"
    if error_path.exists():
        error_path.unlink()
    xtr = xva = ytr = yva = model = decoder = None
    try:
        xtr, xva, ytr, yva = core.load_data(npz)
        config = core.encoder_config(cargs, arm)
        core.write_json(run / "config.json", {
            "session": session, "npz": npz, "seed": args.seed,
            "decoder_seed": args.seed + 10000, "cebra": config,
            "data_shapes": {"train_data": xtr.shape, "valid_data": xva.shape,
                            "train_label": ytr.shape, "valid_label": yva.shape},
        })
        print(f"\n{'=' * 92}\n{session} | {arm} | train={xtr.shape}, valid={xva.shape}, labels={ytr.shape[1]}\n{'=' * 92}",
              flush=True)
        core.seed_all(args.seed)
        start = time.perf_counter()
        model = cebra.CEBRA(**config)
        model.fit(xtr, ytr)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encoder_seconds = time.perf_counter() - start
        model.save(str(run / "cebra_model.pt"))
        log = core.json_safe(model.solver_.log)
        core.write_json(run / "encoder_log.json", log)
        ztr = np.ascontiguousarray(model.transform(xtr), dtype=np.float32)
        zva = np.ascontiguousarray(model.transform(xva), dtype=np.float32)
        for split, z, y in (("train", ztr, ytr), ("valid", zva, yva)):
            if z.shape != (len(y), args.latent_dim) or not np.isfinite(z).all():
                raise RuntimeError(f"{split} embeddings invalid/misaligned: z={z.shape}, labels={y.shape}")
        np.savez_compressed(run / "embeddings.npz", train=ztr, valid=zva)
        del model
        model = None
        clean_cuda(torch)

        start = time.perf_counter()
        decoder, losses = core.train_decoder(ztr, ytr, cargs, args.seed + 10000, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        decoder_seconds = time.perf_counter() - start
        ptr = core.predict(decoder, ztr, device, args.predict_batch_size)
        pva = core.predict(decoder, zva, device, args.predict_batch_size)
        r2_train, r2_valid = core.score(ytr, ptr), core.score(yva, pva)
        torch.save({
            "state_dict": {k: v.detach().cpu() for k, v in decoder.state_dict().items()},
            "input_dim": ztr.shape[1], "output_dim": ytr.shape[1],
            "hidden": args.decoder_hidden, "dropout": args.decoder_dropout,
            "layout": "Linear-LayerNorm-ReLU-Dropout-Linear", "seed": args.seed + 10000,
        }, run / "decoder.pt")
        np.save(run / "decoder_train_mse.npy", np.asarray(losses))
        np.savez_compressed(run / "predictions.npz", train_pred=ptr, valid_pred=pva,
                            train_label=ytr, valid_label=yva)
        row = {
            "completed": True, "session": session, "arm": arm, "seed": args.seed,
            "neurons": xtr.shape[1], "labels": ytr.shape[1],
            "train_samples": len(xtr), "valid_samples": len(xva),
            "r2_train": r2_train["mean"], "r2_valid": r2_valid["mean"],
            "train_per_output": r2_train["per_output"],
            "valid_per_output": r2_valid["per_output"],
            "encoder_seconds": encoder_seconds, "decoder_seconds": decoder_seconds,
        }
        for key in ("adv_eps_mean", "adv_eps_median", "adv_eps_cap",
                    "adv_frac_chance", "adv_frac_capped"):
            values = log.get(key, [])
            row[key + "_last"] = values[-1] if values else None
        core.write_json(metrics_path, row)
        print(f"RESULT {session} / {arm}: TRAIN R2={r2_train['mean']:.6f} | VALID R2={r2_valid['mean']:.6f}")
        print("VALID per-output R2:", r2_valid["per_output"], flush=True)
    except Exception as exc:
        core.write_json(error_path, {
            "completed": False, "session": session, "arm": arm,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        print(f"FAILED {session} / {arm}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if args.fail_fast:
            raise
    finally:
        del model, decoder, xtr, xva, ytr, yva
        clean_cuda(torch)


def print_summary(selected, rows, agg, out):
    found = {(r["session"], r["arm"]): r for r in rows}
    print("\n" + "=" * 100)
    print("FINAL CLEAN-INPUT VALIDATION R2 (ALL LABEL COLUMNS; HIGHER IS BETTER)")
    print("=" * 100)
    print(f"{'session':<14s}{'fixed_eps_3':>16s}{'margin':>16s}{'margin-fixed':>16s}")
    for npz in selected:
        f = found.get((npz.stem, "fixed_eps_3"))
        m = found.get((npz.stem, "margin"))
        fv = f["r2_valid"] if f else None
        mv = m["r2_valid"] if m else None
        delta = mv - fv if fv is not None and mv is not None else None
        fmt = lambda v: f"{v:.6f}" if v is not None else "PENDING/FAILED"
        print(f"{npz.stem:<14s}{fmt(fv):>16s}{fmt(mv):>16s}{fmt(delta):>16s}")
    print("-" * 100)
    for arm in ARMS:
        a = agg["by_arm"][arm]
        print(f"{arm}: mean R2={a['r2_valid_mean_across_sessions']} "
              f"std={a['r2_valid_std_across_sessions']} n={a['n_completed']}")
    print(f"Paired margin-fixed delta: mean={agg['paired_delta_mean']} std={agg['paired_delta_std']} | "
          f"wins margin/fixed/ties={agg['margin_wins']}/{agg['fixed_wins']}/{agg['ties']}")
    print(f"Saved: {out}", flush=True)


def main():
    args = parse_args()
    import torch
    core.torch = torch
    args.cebra_dir = resolve_fork(args.cebra_dir)
    all_files, selected = discover_and_sample(args)
    out = (args.out_dir.expanduser().resolve() /
           f"selection_seed_{args.selection_seed}_n{args.num_sessions}")
    out.mkdir(parents=True, exist_ok=True)
    sig = signature(args, selected, args.cebra_dir)
    manifest = out / "manifest.json"
    if manifest.is_file():
        old = json.loads(manifest.read_text())
        if old.get("signature") != core.json_safe(sig):
            raise RuntimeError(
                f"Existing run at {out} has different settings. Use a different --out-dir, "
                "or restore the original command.")
    else:
        fork_hashes = {name: hashlib.sha256((args.cebra_dir / name).read_bytes()).hexdigest()
                       for name in ("cebra/solver/base.py", "cebra/integrations/sklearn/cebra.py")}
        core.write_json(manifest, {
            "signature": sig, "npz_files_found": len(all_files),
            "all_npz_names": [p.name for p in all_files],
            "fork_sha256": fork_hashes,
            "evaluation": "Original NPZ train/valid split; clean-input R2; all label columns",
        })
    if args.device == "cuda_if_available":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    cargs = configure_core_args(args)
    device = torch.device(args.device)
    cebra = core.load_fork(args.cebra_dir)
    print(f"Found {len(all_files)} NPZ files in {args.data_dir.expanduser().resolve()}")
    print(f"Excluded before sampling: {args.exclude}")
    print(f"Random 10 selection (seed={args.selection_seed}): {[p.stem for p in selected]}")
    print("Each selected NPZ uses its original train_data/valid_data split and ALL label columns.")
    print(f"Outputs/resume directory: {out}", flush=True)

    for npz in selected:
        for arm in ARMS:
            run_arm(cebra, torch, device, args, cargs, npz.stem, npz, arm, out)
            rows, agg = save_suite_summary(out)
            print_summary(selected, rows, agg, out)


if __name__ == "__main__":
    main()

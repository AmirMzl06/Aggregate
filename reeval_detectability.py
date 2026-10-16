"""Exp6a: detectability of the decoder attacks at equal damage, on the SAVED models (no training).

Exp5 showed that noise-aware training is more robust to noise-shaped attacks and less robust to
isotropic ones (9/9 sessions each way). Which threat matters is the note's question of
detectability: "ε_det: the largest ε at which a detector trained to separate clean from
adversarial trials (plus a clean-only anomaly detector such as an autoencoder) stays <= 0.55
AUC". For every saved encoder + decoder this script runs
``cebra.dprime.evaluation.evaluate_detectability``: the decoder attacks of both shared metrics
(reference = Ledoit-Wolf x Welch, isotropic = sigma-scaled L2) at budgets 0.5 ... 16, scored by
the decoding R² (damage), a blind autoencoder AUC, a supervised CNN AUC and the realised sizes,
and interpolated at the budget that halves the clean R² (equal damage).

Layout expected (as written by the Exp5 runner): <exp-dir>/<SESSION>/seed<SEED>/models/
<ARM>_encoder.pt and <ARM>_decoder.pt; results go to .../seed<SEED>/detectability/<ARM>.json and a
summary CSV to <exp-dir>/detectability_seed<SEED>.csv.

Run one process per GPU, each with its own shard, e.g. on two GPUs:
    CUDA_VISIBLE_DEVICES=0 python -u reeval_detectability.py --shard 0/2 &
    CUDA_VISIBLE_DEVICES=1 python -u reeval_detectability.py --shard 1/2 &
    wait; python -u reeval_detectability.py --summarize-only
Re-running skips finished models (delete a JSON to recompute it).
"""

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SESSION_PATTERN = re.compile(r"^[A-Z]-(CO|RT)\d+$")
METRICS = ("reference", "isotropic")
SUMMARY_KEYS = ("epsilon", "r2", "ae_auc", "cnn_auc", "dprime_reference", "dprime_isotropic",
                "raw_l2")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fork-dir", type=Path, default=ROOT / "Acorn-dprime")
    p.add_argument("--data-dir", type=Path,
                   default=Path("/data/hossein/mm_project/perich_data_valid_final_raw/"))
    p.add_argument("--exp-dir", type=Path, default=ROOT / "DPRIME_EXP5")
    p.add_argument("--seed", type=int, default=42, help="Training seed of the saved models")
    p.add_argument("--sessions", nargs="+", default=None)
    p.add_argument("--arms", nargs="+", default=None)
    p.add_argument("--shard", default="0/1", help="i/k: this process does every k-th model")
    p.add_argument("--n-windows", type=int, default=2048)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--detector-steps", type=int, default=300)
    p.add_argument("--summarize-only", action="store_true")
    return p.parse_args()


def load_fork(fork_dir: Path):
    if not (fork_dir / "cebra" / "__init__.py").is_file():
        raise FileNotFoundError(f"No Acorn-dprime fork at {fork_dir}")
    for key in list(sys.modules):
        if key == "cebra" or key.startswith("cebra."):
            del sys.modules[key]
    sys.path.insert(0, str(fork_dir.resolve()))
    import cebra
    from cebra.dprime import evaluation
    if not str(Path(cebra.__file__).resolve()).startswith(str(fork_dir.resolve()) + os.sep):
        raise RuntimeError(f"Wrong CEBRA imported: {cebra.__file__}")
    if not hasattr(evaluation, "evaluate_detectability"):
        raise RuntimeError("This fork has no evaluate_detectability; use the new Acorn-dprime.zip")
    return cebra, evaluation


def find_jobs(args):
    """(session, arm, encoder path, decoder path, output path) of every saved model."""
    jobs = []
    for encoder_path in sorted(args.exp_dir.rglob("*_encoder.pt")):
        if encoder_path.parent.name != "models" or encoder_path.parent.parent.name != f"seed{args.seed}":
            continue
        session = next((part for part in encoder_path.parts if SESSION_PATTERN.match(part)), None)
        if session is None or (args.sessions and session not in args.sessions):
            continue
        arm = encoder_path.name[:-len("_encoder.pt")]
        if args.arms and arm not in args.arms:
            continue
        decoder_path = encoder_path.with_name(f"{arm}_decoder.pt")
        if not decoder_path.is_file():
            print(f"SKIP {session} {arm}: no decoder file", flush=True)
            continue
        output = encoder_path.parent.parent / "detectability" / f"{arm}.json"
        jobs.append((session, arm, encoder_path, decoder_path, output))
    return jobs


def load_data(data_dir: Path, session: str):
    with np.load(data_dir / f"{session}.npz", allow_pickle=False) as f:
        x_train, x_valid, y_valid = (np.asarray(f[k], dtype=np.float32)
                                     for k in ("train_data", "valid_data", "valid_label"))
    if y_valid.ndim == 1:
        y_valid = y_valid[:, None]
    return x_train, x_valid, y_valid


class TwoLayerMLP(nn.Module):
    """The runner's decoder: Linear - LayerNorm - ReLU - Dropout - Linear."""

    def __init__(self, dim: int, hidden: int, out: int, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


def load_decoder(path: Path, device) -> nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    hidden, dim = state["net.0.weight"].shape
    out = state["net.4.weight"].shape[0]
    decoder = TwoLayerMLP(dim, hidden, out)
    decoder.load_state_dict(state)
    return decoder.to(device).eval()


def load_encoder(cebra, path: Path):
    try:
        return cebra.CEBRA.load(str(path))
    except Exception:  # older torch / custom globals: the file is our own, so unpickle fully
        return cebra.CEBRA.load(str(path), weights_only=False)


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    os.replace(temp, path)


def summarize(args, jobs) -> None:
    rows = []
    for session, arm, _, _, output in jobs:
        if not output.is_file():
            continue
        report = json.loads(output.read_text())
        row = {"session": session, "arm": arm, "clean_r2": report.get("clean_r2")}
        for metric in METRICS:
            half = (report.get(metric) or {}).get("at_half_r2") or {}
            for key in SUMMARY_KEYS:
                row[f"{metric}_half_{key}"] = half.get(key)
            for point in (report.get(metric) or {}).get("curve", []):
                if point["epsilon"] == 4.0:
                    for key in ("r2", "ae_auc", "cnn_auc", "dprime_reference"):
                        row[f"{metric}_eps4_{key}"] = point[key]
        rows.append(row)
    if not rows:
        print("No finished models to summarize.", flush=True)
        return
    path = args.exp_dir / f"detectability_seed{args.seed}.csv"
    fields = sorted({key for row in rows for key in row},
                    key=lambda k: (k not in ("session", "arm", "clean_r2"), k))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary of {len(rows)} models: {path}", flush=True)
    # Arm means over sessions at equal damage (half the clean R²).
    print(f"{'arm':<22}" + "".join(f"{m[:3]}_{k:<10}" for m in METRICS
                                   for k in ("ae_auc", "cnn_auc", "dref")), flush=True)
    for arm in sorted({row["arm"] for row in rows}):
        values = []
        for metric in METRICS:
            for key in ("ae_auc", "cnn_auc", "dprime_reference"):
                data = [row[f"{metric}_half_{key}"] for row in rows
                        if row["arm"] == arm and row[f"{metric}_half_{key}"] is not None]
                values.append(f"{np.mean(data):<14.3f}" if data else f"{'-':<14}")
        print(f"{arm:<22}" + "".join(values), flush=True)


def main():
    args = parse_args()
    jobs = find_jobs(args)
    if not jobs:
        raise SystemExit(f"No saved models under {args.exp_dir} (seed {args.seed}).")
    if args.summarize_only:
        summarize(args, jobs)
        return
    shard, shards = (int(v) for v in args.shard.split("/"))
    mine = jobs[shard::shards]
    print(f"{len(mine)} of {len(jobs)} models in shard {shard}/{shards}", flush=True)
    cebra, evaluation = load_fork(args.fork_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cache = {}
    failures = 0
    for session, arm, encoder_path, decoder_path, output in mine:
        if output.is_file():
            print(f"SKIP {session} {arm}: done", flush=True)
            continue
        start = time.monotonic()
        try:
            if session not in data_cache:
                data_cache.clear()
                data_cache[session] = load_data(args.data_dir, session)
            x_train, x_valid, y_valid = data_cache[session]
            encoder = load_encoder(cebra, encoder_path)
            decoder = load_decoder(decoder_path, torch.device(encoder.device_))
            report = evaluation.evaluate_detectability(encoder, decoder, x_train, x_valid,
                                                       y_valid, n_windows=args.n_windows,
                                                       steps=args.steps,
                                                       detector_steps=args.detector_steps,
                                                       seed=args.seed + 300_000)
            report.update(session=session, arm=arm, training_seed=args.seed)
            atomic_write_json(output, report)
            ref = report["reference"]["at_half_r2"] or {}
            iso = report["isotropic"]["at_half_r2"] or {}
            print(f"DONE {session} {arm} ({time.monotonic() - start:.0f}s): clean R2 "
                  f"{report['clean_r2']:.3f} | at half R2: ref AE {ref.get('ae_auc')} "
                  f"CNN {ref.get('cnn_auc')} | iso AE {iso.get('ae_auc')} "
                  f"CNN {iso.get('cnn_auc')}", flush=True)
        except Exception:
            failures += 1
            print(f"FAILED {session} {arm}", flush=True)
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    summarize(args, jobs)
    if failures:
        raise SystemExit(f"{failures} models failed")


if __name__ == "__main__":
    main()

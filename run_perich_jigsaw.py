"""Perich temporal Jigsaw experiment runner -- pair with perich_jigsaw_model.py.

Install: python -m pip install numpy scipy scikit-learn torch
Inspect: python run_perich_jigsaw.py --list
Preview: python run_perich_jigsaw.py --suite core --dry-run

Example (68th NPZ in lexicographic order, ONE-based index):
 python run_perich_jigsaw.py --session-index 68 --suite core --steps 6000 \
     --architectures residual mobile_v2 mobile_v3 mobile_level --seeds 0 1 2 \
     --controls random frozen --out-dir ./perich_context_results

Four training/noise combinations, with *Gaussian* noise only:
 python run_perich_jigsaw.py --session-index 68 --arms context \
     --training-modes frozen joint --noise-sites input embedding --noise-sd 0.2

The four NPZ keys from the supplied Perich runner are supported:
 train_data [time,neurons], train_label [time,targets], valid_data, valid_label.
Also supports optional test_data/test_label. If present, valid is for selection
and test is final. Otherwise train is internally split and the original valid
is treated as final held-out data. Never tune using those final results.

Optional per-bin *_trial_ids / *_trial_id / *_sequence_ids and *_time_index
preserve sequence boundaries. Optional *_lengths stores lengths of trials.
Without boundaries, each provided split is ASSUMED chronologically contiguous;
internal selection uses a purged chronological block, NOT a claimed trial split.
Use --require-trial-ids to refuse that assumption. Already shuffled/concatenated
rows require original metadata; no program can recover missing chronology.

Default downstream evaluation: frozen encoder + ridge and MLP, uniform-average
R2 on the first two behavioral columns (same targets as the supplied runner).
Positive --behavior-lag predicts y[t+lag] from neural data up to t. It must be
chosen using development data, not by looking at final held-out R2.

Every trained arm gets a matched random encoder by default. Frozen-label
controls randomize ALL order/lag labels, not just the position head. For hybrid
arms, --controls no_order also leaves only the auxiliary task. Pure Jigsaw arms
have no reconstruction, contrastive or supervised encoder loss. 'joint' is a
separately labeled supervised comparison (Jigsaw + behavior MSE, never CTC).

Teacher coverage: short/gapped/contiguous/curriculum, jitter/time roll, linear or
relational head, hard mining, head reset, drift-based span selection, count-match,
raw/mean/zscore views, signed lag/time-axis, order ramp, full combination, label
controls, Mobile V1/V2/V3 and level statistics. These are reimplementations.
The unavailable mobile_jigsaw.py cannot be reproduced exactly. Reconstruction
anchor/decay/forecast/reconstruction-level arms are intentionally absent: this
experiment tests temporal learning without a reconstruction objective.

Paper-inspired separate alternatives: channel-time inconsistency and ensemble
discrimination (PopT), signed channel shift, CPC-style future prediction, and a
SCARF-style temporal-channel corruption contrastive control. These are NOT full
paper replications; the last uses same-sequence whole-channel windows, not the
original tabular feature-marginal sampler. The population Transformer uses
session-local neuron IDs, not PopT's anatomical coordinates/BrainBERT or POYO's
individual spike tokens. Multi-session pooling with invented neuron alignment
is deliberately unsupported. Every session is independently fitted.

STEPS are optimizer updates, not epochs. Resume interrupted runs with --resume
and the identical training configuration. Checkpoints contain model, optimizer,
Gaussian RNGs and sampler state; only load checkpoints you trust. --stop-after
is a resumable interruption aid, useful for cluster time limits and tests.
--select-checkpoint last (default) is fixed-budget; 'best' selects by internal
pretext validation loss, not final decoding R2. Probe selection always uses the
internal validation split. Metrics include per-target R2, no fabricated CIs.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from perich_jigsaw_model import (ARCHITECTURES, ContextJigsaw, JigsawConfig,
                                 SpanSampler, assignment_ranks, effective_dimension, order_metrics)

VERSION = "1.0.0"
REFERENCES = {
    "Skip-Clip": "https://arxiv.org/abs/1910.12770",
    "PopT": "https://arxiv.org/abs/2406.03044",
    "NDT2": "https://papers.neurips.cc/paper_files/paper/2023/file/fe51de4e7baf52e743b679e3bdba7905-Paper-Conference.pdf",
    "POYO": "https://arxiv.org/abs/2310.16046",
    "consistent_ensembles": "https://proceedings.mlr.press/v197/jude23a.html",
    "CPC": "https://arxiv.org/abs/1807.03748",
    "SCARF": "https://arxiv.org/abs/2106.15147",
}

# Each entry changes only the named mechanism. All reconstruction weights are
# absent, not silently inherited from the old JigsawNet defaults.
ARMS = {
    "jigsaw": dict(context_mode="none"),
    "context": {},
    "context_pair": dict(lambda_order=0),
    "context_position": dict(lambda_pair=0),
    "context_relational": dict(head="relational"),
    "context_linear": dict(head="linear"),
    "context_lag": dict(lambda_lag=1),
    "context_time_axis": dict(lambda_order=0, lambda_pair=0, lambda_lag=1, head="linear"),
    "context_multiscale": dict(scales=(1, 2)),
    "context_shuffled": dict(context_mode="shuffled"),
    "context_reversed": dict(context_mode="reversed"),
    "context_full": dict(scales=(1, 2), head="relational", lambda_lag=1, hard_fraction=.5),
    "order_default": dict(context_mode="none", window=10, gap_min=1, gap_max=8, view="mean"),
    "order_short_span": dict(context_mode="none", window=6, gap_min=1, gap_max=3, view="mean"),
    "order_gapped": dict(context_mode="none", window=6, gap_min=3, gap_max=5, view="mean"),
    "order_contiguous": dict(context_mode="none", window=6, gap_min=0, gap_max=0, view="mean"),
    "order_curriculum": dict(context_mode="none", gap_curriculum=.5),
    "order_jittered": dict(context_mode="none", time_roll=2),
    "order_linear_head": dict(context_mode="none", head="linear"),
    "order_reset": dict(context_mode="none", reset_every=400),
    "order_hard": dict(context_mode="none", hard_fraction=.25),
    "order_relational": dict(context_mode="none", head="relational"),
    "order_decorrelated": dict(context_mode="none", span_selection="decorrelated"),
    # The teacher's 'adversarial' span selector renamed: no gradient/noise attack.
    "order_reverse_drift": dict(context_mode="none", span_selection="reverse_drift"),
    "order_shortcut": dict(context_mode="none", span_selection="shortcut"),
    "order_count_match": dict(context_mode="none", view="count_match"),
    "order_raw": dict(context_mode="none", view="raw"),
    "order_zscore": dict(context_mode="none", view="zscore"),
    "order_lag": dict(context_mode="none", lambda_lag=1),
    "order_time_axis": dict(context_mode="none", lambda_order=0, lambda_pair=0, lambda_lag=1, head="linear"),
    "order_lag_only": dict(context_mode="none", lambda_order=0, lambda_pair=0, lambda_lag=1),
    "order_ramp": dict(context_mode="none", order_final_scale=3),
    "order_full": dict(context_mode="none", window=6, gap_min=3, gap_max=5, view="mean",
                       time_roll=2, lambda_lag=1, hard_fraction=.25, reset_every=400),
    "tiles_2": dict(tiles=2), "tiles_3": dict(tiles=3), "tiles_6": dict(tiles=6),
    "wide_gaps": dict(gap_min=8, gap_max=40), "tight_gaps": dict(gap_min=1, gap_max=2),
    "dim_16": dict(dimension=16), "dim_32": dict(dimension=32),
    "dim_128": dict(dimension=128), "dim_256": dict(dimension=256),
    "normalized": dict(l2_normalize=True),
    "mobile_v2_t6": dict(architecture="mobile_v2", expansion=6),
    "mobile_stem_mix": dict(architecture="mobile_v2", stem="mix"),
    "mobile_alpha_035": dict(architecture="mobile_v2", width_multiplier=.35),
    "mobile_alpha_050": dict(architecture="mobile_v2", width_multiplier=.5),
    "mobile_alpha_140": dict(architecture="mobile_v2", width_multiplier=1.4),
    "mobile_batchnorm": dict(architecture="mobile_v2", norm="batch"),
    "mobile_hardswish": dict(architecture="mobile_v2", activation="hardswish"),
    "channel": dict(lambda_order=0, lambda_pair=0, auxiliary="channel", lambda_aux=1),
    "channel_shift": dict(lambda_order=0, lambda_pair=0, auxiliary="shift", lambda_aux=1),
    "ensemble": dict(context_mode="none", lambda_order=0, lambda_pair=0, auxiliary="ensemble", lambda_aux=1),
    "cpc": dict(context_mode="none", lambda_order=0, lambda_pair=0, auxiliary="cpc", lambda_aux=1),
    "scarf": dict(context_mode="none", lambda_order=0, lambda_pair=0, auxiliary="scarf", lambda_aux=1),
    "context_channel": dict(auxiliary="channel", lambda_aux=.5),
    "context_shift": dict(auxiliary="shift", lambda_aux=.5),
}
SUITES = {
    "quick": ["jigsaw", "context"],
    "core": ["jigsaw", "context", "context_lag", "context_multiscale", "context_full"],
    "context": [k for k in ARMS if k.startswith("context") and k not in ("context_channel", "context_shift")],
    "teacher": [k for k in ARMS if k.startswith("order_")],
    "capacity": [k for k in ARMS if k.startswith(("mobile_", "dim_", "tiles_"))] + ["normalized", "wide_gaps", "tight_gaps"],
    "alternatives": ["channel", "channel_shift", "ensemble", "cpc", "scarf", "context_channel", "context_shift"],
    "all": list(ARMS),
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def plain(obj):
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, np.generic):
        return plain(obj.item())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def torch_save(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


@dataclass
class SequenceData:
    x: np.ndarray
    y: np.ndarray
    name: str
    trial: str | None = None


def split_arrays(handle, prefix, target_columns, require_ids):
    keys = (prefix+"_data", prefix+"_label")
    if any(k not in handle for k in keys):
        raise KeyError(f"Need {keys}; available: {list(handle.keys())}")
    x, y = (np.asarray(handle[k], np.float32) for k in keys)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim == 2 and len(x) != len(y) and x.shape[1] == len(y):
        x = x.T
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError(f"{prefix}: need matching [time,features] arrays; got {x.shape}, {y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{prefix}: nonfinite data/labels; clean or segment them explicitly")
    if min(target_columns) < 0 or max(target_columns) >= y.shape[1]:
        raise ValueError(f"{prefix}: target columns {target_columns} invalid for {y.shape[1]} labels")
    ids, times = None, None
    for suffix in ("trial_ids", "trial_id", "sequence_ids"):
        key = prefix+"_"+suffix
        if key in handle:
            ids = np.asarray(handle[key]).reshape(-1)
            break
    if ids is None and prefix+"_lengths" in handle:
        lengths = np.asarray(handle[prefix+"_lengths"], np.int64).reshape(-1)
        if np.any(lengths <= 0) or lengths.sum() != len(x):
            raise ValueError(f"{prefix}_lengths must be positive and sum to {len(x)}")
        ids = np.repeat(np.arange(len(lengths)), lengths)
    if ids is not None and len(ids) != len(x):
        raise ValueError(f"{prefix}: per-bin trial IDs length mismatch")
    if prefix+"_time_index" in handle:
        times = np.asarray(handle[prefix+"_time_index"]).reshape(-1)
        if len(times) != len(x):
            raise ValueError(f"{prefix}: time_index length mismatch")
    if require_ids and ids is None:
        raise ValueError(f"{prefix}: --require-trial-ids needs trial_ids or lengths")
    boundary = np.zeros(max(0, len(x)-1), bool)
    if ids is not None:
        boundary |= ids[1:] != ids[:-1]
    if times is not None:
        boundary |= np.diff(times) != 1  # integer bin indices, NOT seconds
    cuts = np.r_[0, np.flatnonzero(boundary)+1, len(x)]
    sequences = [SequenceData(np.ascontiguousarray(x[a:b]), np.ascontiguousarray(y[a:b, target_columns]),
                              f"{prefix}:{a}:{b}", str(ids[a]) if ids is not None else None)
                 for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    return sequences, dict(has_trial_ids=ids is not None, has_time_index=times is not None,
                           n_sequences=len(sequences), bins=len(x))


def load_session(path, args, purge):
    with np.load(path, allow_pickle=False) as data:
        train, train_meta = split_arrays(data, "train", args.target_columns, args.require_trial_ids)
        original_valid, valid_meta = split_arrays(data, "valid", args.target_columns, args.require_trial_ids)
        if "test_data" in data or "test_label" in data:
            final, final_meta = split_arrays(data, "test", args.target_columns, args.require_trial_ids)
            fit, val = train, original_valid
            protocol = "explicit_train_valid_test"
        else:
            final, final_meta = original_valid, valid_meta
            trial_names = list(dict.fromkeys(s.trial for s in train)) if train_meta["has_trial_ids"] else []
            if len(trial_names) > 1:
                # Keep every disjoint segment of the SAME trial on the SAME side.
                nval = max(1, min(len(trial_names)-1, math.ceil(args.val_fraction*len(trial_names))))
                held = set(trial_names[-nval:])
                fit, val = [s for s in train if s.trial not in held], [s for s in train if s.trial in held]
                protocol = "held_out_training_trials__original_valid_is_final"
            elif len(train) > 1 and not trial_names:
                nval = max(1, min(len(train)-1, math.ceil(args.val_fraction*len(train))))
                fit, val = train[:-nval], train[-nval:]
                protocol = "held_out_training_sequences__original_valid_is_final"
            elif len(train) > 1:
                raise ValueError("Only one trial, with disconnected segments: supply an explicit valid/test split")
            else:
                seq = train[0]
                cut = int(len(seq.x)*(1-args.val_fraction))
                a, b = cut-purge, cut+purge
                if a < 2 or len(seq.x)-b < 2:
                    raise ValueError("Not enough training bins for purged selection; supply trials or smaller spans")
                fit = [SequenceData(seq.x[:a], seq.y[:a], f"fit:0:{a}")]
                val = [SequenceData(seq.x[b:], seq.y[b:], f"val:{b}:{len(seq.x)}")]
                protocol = "purged_chronological_blocks__original_valid_is_final"
    channels = fit[0].x.shape[1]
    if any(s.x.shape[1] != channels for split in (fit, val, final) for s in split):
        raise ValueError("Neuron dimension differs between sequences/splits")
    # All preprocessing and channel selection fit on FIT only, including when valid exists.
    joined = np.concatenate([s.x for s in fit])
    std = joined.std(0)
    alive = std > 1e-6
    if not alive.any():
        raise ValueError("No varying neurons in fit split")
    offset = joined[:, alive].mean(0) if args.preprocess == "zscore" else np.zeros(alive.sum(), np.float32)
    scale = std[alive] if args.preprocess in ("zscore", "scale") else np.ones(alive.sum(), np.float32)
    scale = np.maximum(scale, 1e-4)
    for split in (fit, val, final):
        for s in split:
            s.x = np.ascontiguousarray((s.x[:, alive]-offset)/scale, dtype=np.float32)
    if not train_meta["has_trial_ids"]:
        print("NOTE: no train trial IDs: assuming each uninterrupted block is chronological; "
              "trial boundaries and earlier NPZ preprocessing cannot be verified.", flush=True)
    metadata = dict(protocol=protocol, train=train_meta, original_valid=valid_meta, final=final_meta,
                    fit_sequences=[s.name for s in fit], val_sequences=[s.name for s in val],
                    final_sequences=[s.name for s in final], purge_bins=purge,
                    original_neurons=channels, kept_neurons=int(alive.sum()),
                    target_columns=args.target_columns, behavior_lag=args.behavior_lag,
                    bin_ms=args.bin_ms, final_data_used_for_selection=False)
    return fit, val, final, metadata, dict(alive=alive, offset=offset, scale=scale)


def row_index(sequences, warmup, lag, maximum=None, seed=0):
    rows = np.asarray([(s, t) for s, seq in enumerate(sequences)
                       for t in range(warmup-1, len(seq.x)-lag)], np.int64).reshape(-1, 2)
    if len(rows) < 2:
        raise ValueError(f"Too few decode samples: need sequences > warmup {warmup} + lag {lag}")
    if maximum and len(rows) > maximum:
        keep = np.random.default_rng(seed).choice(len(rows), maximum, replace=False)
        rows = rows[np.sort(keep)]
    return rows


def windows_at(sequences, rows, cfg, lag, device):
    windows, contexts, y = [], [], []
    for sid, t in rows:
        seq = sequences[int(sid)]
        start = int(t)+1-cfg.window
        end_context = start-cfg.gap_min
        windows.append(seq.x[start:t+1])
        contexts.append(seq.x[end_context-cfg.context:end_context])
        y.append(seq.y[t+lag])
    tensors = [torch.from_numpy(np.ascontiguousarray(np.stack(v), dtype=np.float32)).to(device)
               for v in (contexts, windows, y)]
    return tensors


@torch.no_grad()
def extract(model, sequences, rows, cfg, args, raw=False):
    model.eval()
    features, labels = [], []
    for start in range(0, len(rows), args.eval_batch_size):
        c, x, y = windows_at(sequences, rows[start:start+args.eval_batch_size], cfg, args.behavior_lag, args.device)
        if raw:
            z = x.flatten(1) if cfg.context_mode == "none" else torch.cat((c.flatten(1), x.flatten(1)), 1)
        else:
            z = model.embed(c, x)
        features.append(z.cpu().numpy())
        labels.append(y.cpu().numpy())
    return np.concatenate(features), np.concatenate(labels)


def r2(y, prediction):
    y, prediction = np.asarray(y, np.float64), np.asarray(prediction, np.float64)
    total = ((y-y.mean(0))**2).sum(0)
    values = 1-((y-prediction)**2).sum(0)/np.maximum(total, 1e-12)
    # Constant targets have no defined R2; do not fabricate a finite score.
    values[total < 1e-12] = np.nan
    return dict(mean=float(np.nanmean(values)) if np.isfinite(values).any() else float("nan"), per_target=values.tolist())


class BehaviorDecoder(nn.Module):
    def __init__(self, dim, targets, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, targets))

    def forward(self, x):
        return self.net(x)


def probes(features, args, seed, directory):
    (xt, yt), (xv, yv), (xe, ye) = features
    scaler = StandardScaler().fit(xt)
    xt, xv, xe = [scaler.transform(x).astype(np.float32) for x in (xt, xv, xe)]
    best, best_score, best_alpha = None, -math.inf, None
    for alpha in args.ridge_alphas:
        estimator = Ridge(alpha=alpha, solver="lsqr").fit(xt, yt)
        score = r2(yv, estimator.predict(xv))["mean"]
        if np.isfinite(score) and score > best_score:
            best, best_score, best_alpha = estimator, score, alpha
    if best is None:
        raise ValueError("All ridge validation scores undefined; inspect behavioral targets")
    pred = best.predict(xe)
    result = dict(ridge=dict(final=r2(ye, pred), validation=best_score, alpha=best_alpha))
    # Numeric artifacts loadable with allow_pickle=False; also sufficient for deployment.
    np.savez_compressed(directory/"ridge.npz", x_mean=scaler.mean_, x_scale=scaler.scale_,
                        coef=best.coef_, intercept=best.intercept_, alpha=best_alpha)
    predictions = {"truth": ye, "ridge": pred}
    if args.probe_steps > 0:
        seed_all(seed+741)
        ym, ys = yt.mean(0), np.maximum(yt.std(0), 1e-4)
        tx = torch.from_numpy(xt).to(args.device)
        ty = torch.from_numpy((yt-ym)/ys).to(args.device)
        vx = torch.from_numpy(xv).to(args.device)
        net = BehaviorDecoder(xt.shape[1], yt.shape[1], args.probe_hidden, args.probe_dropout).to(args.device)
        opt = torch.optim.AdamW(net.parameters(), lr=args.probe_lr, weight_decay=args.weight_decay)
        best_state, best_value, best_step = None, -math.inf, 0
        generator = np.random.default_rng(seed+741)
        for step in range(1, args.probe_steps+1):
            net.train()
            ids = torch.from_numpy(generator.integers(0, len(tx), min(args.batch_size, len(tx)))).to(args.device)
            opt.zero_grad(set_to_none=True)
            F.mse_loss(net(tx[ids]), ty[ids]).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            opt.step()
            if step % args.probe_eval_every == 0 or step == args.probe_steps:
                net.eval()
                with torch.no_grad():
                    pv = net(vx).cpu().numpy()*ys+ym
                value = r2(yv, pv)["mean"]
                if np.isfinite(value) and value > best_value:
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    best_value, best_step = value, step
        if best_state is None:
            raise ValueError("MLP validation R2 is undefined")
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            predictions["mlp"] = np.concatenate([net(torch.from_numpy(xe[i:i+args.eval_batch_size]).to(args.device)).cpu().numpy()
                                                  for i in range(0, len(xe), args.eval_batch_size)])*ys+ym
        result["mlp"] = dict(final=r2(ye, predictions["mlp"]), validation=best_value, best_step=best_step)
        torch_save(directory/"probe_mlp.pt", dict(state=best_state, x_mean=scaler.mean_, x_scale=scaler.scale_,
                                                y_mean=ym, y_scale=ys, hidden=args.probe_hidden,
                                                input_dimension=xt.shape[1], target_dimension=yt.shape[1],
                                                dropout=args.probe_dropout))
    np.savez_compressed(directory/"predictions.npz", **predictions)
    return result


@torch.no_grad()
def evaluate_pretext(model, sequences, cfg, args, seed=991):
    model.eval()
    sampler = SpanSampler([s.x for s in sequences], cfg, seed)
    values = []
    for _ in range(args.eval_batches):
        batch = sampler.sample(args.eval_batch_size, device=args.device)
        loss, metrics = model.loss(batch, progress=1, augment=False)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite pretext validation loss")
        out = model(batch)
        metrics.update(order_metrics(out, batch, cfg))
        if cfg.lambda_lag:
            target = model.lag_targets(batch["offsets"])
            metrics["lag_accuracy"] = float((out["lag"].argmax(-1) == target).float().mean())
        values.append(metrics)
    return {key: float(np.mean([v[key] for v in values])) for key in values[0]}


def shortcut_arrays(batch, use_context):
    tiles = batch["tiles"].cpu().numpy()
    context = batch["context"].cpu().numpy()
    rank = batch["rank"].cpu().numpy()
    k = tiles.shape[1]
    i, j = np.triu_indices(k, 1)
    mean, var = tiles.mean(2), tiles.var(2)
    features = {}
    for name, value in (("mean", mean), ("variance", var)):
        a, b = value[:, i], value[:, j]
        features[name] = np.concatenate((a-b, a+b), -1).reshape(-1, 2*value.shape[-1])
    target = (rank[:, i] < rank[:, j]).reshape(-1)
    # Exhaustive minimum-boundary-distance ordering; no true rank consulted.
    orders = np.asarray(list(__import__("itertools").permutations(range(k))))
    cost = np.zeros((len(tiles), len(orders)))
    for p in range(k-1):
        delta = tiles[:, orders[:, p], -1] - tiles[:, orders[:, p+1], 0]
        cost += np.mean(delta**2, -1)
    if use_context:
        cost += np.mean((context[:, None, -1] - tiles[:, orders[:, 0], 0])**2, -1)
    perm = orders[cost.argmin(1)]
    pred = np.argsort(perm, 1)
    boundary = dict(pair=float(np.mean((pred[:, i] < pred[:, j]) == (rank[:, i] < rank[:, j]))),
                    exact=float(np.mean(np.all(pred == rank, 1))))
    return features, target, boundary


@torch.no_grad()
def diagnostics(model, fit, val, final, cfg, args):
    model.eval()
    # Evaluation uses UNIFORM spans, including for drift-selected training arms.
    # This avoids reporting success only on the sampler's chosen distribution.
    eval_cfg = replace(cfg, span_selection="random", gap_curriculum=0)
    sampled = [SpanSampler([s.x for s in split], eval_cfg, args.seed_diagnostics).sample(
        args.diagnostic_samples, device=args.device, scale=1) for split in (fit, val, final)]
    baseline_data = [shortcut_arrays(batch, cfg.context_mode != "none") for batch in sampled]
    result = dict(chance_pair=.5, chance_exact=1/math.factorial(cfg.tiles),
                  sampling="uniform spans at base scale; examples overlap and are not independent trials",
                  mean_variance_baselines_use="globally preprocessed input, before optional per-tile centering",
                  original_order_metrics={})
    for label, batch in zip(("fit", "validation", "final"), sampled):
        out = model(batch)
        metrics = order_metrics(out, batch, cfg)
        if cfg.lambda_lag:
            target = model.lag_targets(batch["offsets"])
            metrics["lag_accuracy"] = float((out["lag"].argmax(-1) == target).float().mean())
        if cfg.context_mode not in ("none", "shuffled") and (cfg.lambda_order or cfg.lambda_pair):
            # Same tiles, trained weights, changed context: tests actual reliance on context.
            metrics["shuffled_context"] = order_metrics(model(batch, context_override="shuffled"), batch, cfg)
            metrics["removed_context"] = order_metrics(model(batch, context_override="none"), batch, cfg)
        if cfg.auxiliary in ("channel", "shift"):
            z = model.embed(batch["context"], batch["corrupt"])
            logits = model.channel_head(z)
            if cfg.auxiliary == "channel":
                y = batch["corruption_mask"].cpu().numpy().reshape(-1)
                p = logits.sigmoid().cpu().numpy().reshape(-1)
                pred = p >= .5
                metrics["channel_balanced_accuracy"] = float(.5*((pred[y == 1]).mean() + (~pred[y == 0]).mean()))
                metrics["channel_auroc"] = float(roc_auc_score(y, p))
                metrics["channel_positive_fraction"] = float(y.mean())
            else:
                pred = logits.reshape(len(z), cfg.channels, 3).argmax(-1)
                truth = batch["shift_label"]
                recalls = [float((pred[truth == i] == i).float().mean()) for i in range(3) if (truth == i).any()]
                metrics["shift_macro_recall"] = float(np.mean(recalls))
        if cfg.auxiliary == "ensemble":
            z1, z2 = model.local(batch["ensemble_a"]), model.local(batch["ensemble_b"])
            p = model.ensemble_head(torch.cat((z1, z2, z1*z2), -1)).squeeze(-1).sigmoid().cpu().numpy()
            metrics["ensemble_auroc"] = float(roc_auc_score(batch["ensemble_label"].cpu().numpy(), p))
        result["original_order_metrics"][label] = metrics
    for name in ("mean", "variance"):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=500, random_state=cfg.seed))
        clf.fit(baseline_data[0][0][name], baseline_data[0][1])
        result[name+"_only_pair_classifier"] = {label: float(clf.score(d[0][name], d[1]))
                                                for label, d in zip(("fit", "validation", "final"), baseline_data)}
    result["boundary_matching"] = {label: d[2] for label, d in zip(("fit", "validation", "final"), baseline_data)}
    if cfg.lambda_lag:
        count = torch.bincount(model.lag_targets(sampled[0]["offsets"]).flatten(), minlength=cfg.lag_classes)
        majority = int(count.argmax())
        result["lag_majority_class_from_fit"] = majority
        result["lag_majority_accuracy"] = {label: float((model.lag_targets(b["offsets"]) == majority).float().mean())
                                            for label, b in zip(("fit", "validation", "final"), sampled)}
    return result


STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Signal {signum}: finishing this update and saving a resumable checkpoint", flush=True)


def rng_state(sampler, supervised_rng):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                sampler=sampler.rng.bit_generator.state, supervised=supervised_rng.bit_generator.state)


def restore_rng(state, sampler, supervised_rng):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    sampler.rng.bit_generator.state = state["sampler"]
    supervised_rng.bit_generator.state = state["supervised"]


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def train_model(model, fit, val, rows_fit, cfg, args, mode, control, directory, signature):
    sampler = SpanSampler([s.x for s in fit], cfg, cfg.seed+101)
    supervised_rng = np.random.default_rng(cfg.seed+301)
    decoder, ymean, yscale = None, None, None
    if mode == "joint":
        decoder = BehaviorDecoder(cfg.dimension, fit[0].y.shape[1], args.probe_hidden, args.probe_dropout).to(args.device)
        yfit = np.stack([fit[s].y[t+args.behavior_lag] for s, t in rows_fit])
        ymean = torch.from_numpy(yfit.mean(0)).to(args.device)
        yscale = torch.from_numpy(np.maximum(yfit.std(0), 1e-4)).to(args.device)
    params = list(model.parameters()) + ([] if decoder is None else list(decoder.parameters()))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    checkpoint = directory/"last.pt"
    history, start, best_value, best_state, best_decoder, best_step = [], 0, math.inf, None, None, 0
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=args.device, weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Checkpoint configuration or input file changed; use a new --out-dir")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if decoder is not None:
            decoder.load_state_dict(state["decoder"])
        history, start = state["history"], state["step"]
        best_value, best_state, best_decoder, best_step = state["best_value"], state["best_model"], state["best_decoder"], state["best_step"]
        restore_rng(state["rng"], sampler, supervised_rng)
        print(f"  resumed optimizer update {start}/{args.steps}", flush=True)

    def save(step):
        torch_save(checkpoint, dict(signature=signature, config=cfg.to_dict(), model=model.state_dict(),
                                   decoder=decoder.state_dict() if decoder is not None else None,
                                   optimizer=optimizer.state_dict(), step=step, history=history,
                                   best_value=best_value, best_model=best_state, best_decoder=best_decoder,
                                   best_step=best_step, rng=rng_state(sampler, supervised_rng),
                                   y_mean=ymean, y_scale=yscale))
        write_json(directory/"history.json", history)

    steps = 0 if control == "random" else args.steps
    step = start
    started = time.monotonic()
    for step in range(start+1, steps+1):
        if cfg.reset_every and step > 1 and (step-1) % cfg.reset_every == 0:
            model.reset_heads(optimizer)
        model.train()
        if decoder is not None:
            decoder.train()
        progress = (step-1)/max(1, args.steps-1)
        warm = min(1, step/max(1, args.warmup_steps))
        lr = args.lr*warm*(.1+.9*.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = sampler.sample(args.batch_size, progress=progress, device=args.device)
        optimizer.zero_grad(set_to_none=True)
        loss, values = model.loss(batch, progress=progress)
        if decoder is not None:
            sample_rows = rows_fit[supervised_rng.integers(0, len(rows_fit), args.batch_size)]
            c, x, y = windows_at(fit, sample_rows, cfg, args.behavior_lag, args.device)
            mse = F.mse_loss(decoder(model.embed(c, x, augment=True)), (y-ymean)/yscale)
            loss = loss + args.supervised_weight*mse
            values["behavior_mse"] = float(mse.detach())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at update {step}")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(params, args.grad_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite gradient at update {step}")
        optimizer.step()
        if step % args.log_every == 0 or step == 1 or step == steps:
            elapsed = time.monotonic()-started
            record = dict(step=step, lr=lr, gradient_norm=float(norm), **values)
            validation = evaluate_pretext(model, val, cfg, args)
            record["validation"] = validation
            value = validation["loss"]
            if decoder is not None:
                model.eval()
                decoder.eval()
                vr = row_index(val, args.common_warmup, args.behavior_lag, args.eval_batch_size*2, seed=301)
                with torch.no_grad():
                    c, x, y = windows_at(val, vr, cfg, args.behavior_lag, args.device)
                    vl = F.mse_loss(decoder(model.embed(c, x)), (y-ymean)/yscale)
                record["val_behavior_mse"] = float(vl)
                value += args.supervised_weight*float(vl)
            if value < best_value:
                best_value, best_step, best_state = value, step, cpu_state(model)
                best_decoder = cpu_state(decoder) if decoder is not None else None
            history.append(record)
            print(f"  {step:6d}/{steps} loss={values['loss']:.4f} val={validation['loss']:.4f} "
                  f"updates/s={(step-start)/max(elapsed, .001):.2f}", flush=True)
        stop = STOP_REQUESTED or (args.stop_after is not None and step >= args.stop_after and step < steps)
        if step % args.checkpoint_every == 0 or step == steps or stop:
            save(step)
        if stop:
            return None, None, step, False
    if steps == 0:
        save(0)
    selected = step
    if args.select_checkpoint == "best" and best_state is not None:
        model.load_state_dict(best_state)
        if decoder is not None:
            decoder.load_state_dict(best_decoder)
        selected = best_step
    torch_save(directory/"encoder.pt", dict(config=cfg.to_dict(), model=cpu_state(model), selected_step=selected,
                                           preprocessing_file="preprocessing.npz", version=VERSION))
    joint = None
    if decoder is not None:
        joint = dict(decoder=decoder, mean=ymean, scale=yscale)
        torch_save(directory/"joint_decoder.pt", dict(state=cpu_state(decoder), y_mean=ymean.cpu(), y_scale=yscale.cpu(),
                                                      hidden=args.probe_hidden, dropout=args.probe_dropout))
    return model, joint, selected, True


def build_jobs(args):
    names = args.arms if args.arms else SUITES[args.suite]
    jobs, seen = [], set()
    for arm in names:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm}; use --list")
        for architecture in args.architectures:
            # Architecture-specific ablations run once, not once per redundant outer architecture.
            if "architecture" in ARMS[arm] and architecture != args.architectures[0]:
                continue
            for seed in args.seeds:
                for mode in args.training_modes:
                    for site in args.noise_sites:
                        cfg = JigsawConfig(channels=2, window=args.window, context=args.context, tiles=args.tiles,
                                           gap_min=args.gap[0], gap_max=args.gap[1], architecture=architecture,
                                           width=args.width, dimension=args.dimension, depth=args.depth,
                                           dropout=args.dropout, seed=seed, noise_site=site,
                                           noise_sd=args.noise_sd if site != "none" else 0)
                        cfg = replace(cfg, **ARMS[arm])
                        cfg.validate()
                        controls = ["trained"]
                        for control in args.controls:
                            if control == "random":
                                controls.append("supervised_only" if mode == "joint" else "random")
                            elif control == "frozen" and cfg.lambda_order+cfg.lambda_pair+cfg.lambda_lag > 0:
                                controls.append("frozen")
                            elif control == "fresh" and cfg.lambda_order+cfg.lambda_pair+cfg.lambda_lag > 0:
                                controls.append("fresh")
                            elif control == "no_order" and cfg.lambda_aux:
                                controls.append("no_order")
                        for control in controls:
                            settings = cfg
                            if control in ("frozen", "fresh"):
                                settings = replace(cfg, label_control=control)
                            elif control in ("supervised_only", "no_order"):
                                settings = replace(cfg, lambda_order=0, lambda_pair=0, lambda_lag=0,
                                                   lambda_aux=0 if control == "supervised_only" else cfg.lambda_aux)
                            key = (arm, settings.architecture, seed, mode, site, control)
                            if key not in seen:
                                seen.add(key)
                                jobs.append(dict(arm=arm, cfg=settings, mode=mode, control=control))
    return jobs


def run_one(path, data, job, args, source_hashes, data_hash):
    fit, val, final, metadata, preprocessing = data
    cfg = replace(job["cfg"], channels=fit[0].x.shape[1]).validate()
    for label, split in zip(("fit", "validation", "final"), (fit, val, final)):
        if max(len(s.x) for s in split) < cfg.max_span:
            raise ValueError(f"{path.name}/{job['arm']}: no {label} sequence long enough for {cfg.max_span} bins")
    if cfg.view == "count_match" and args.preprocess == "zscore":
        raise ValueError("order_count_match requires --preprocess scale (no centering)")
    site = f"{cfg.noise_site}_{cfg.noise_sd:g}"
    directory = args.out_dir/path.stem/cfg.architecture/job["arm"]/f"seed_{cfg.seed}"/site/job["mode"]/job["control"]
    manifest = dict(version=VERSION, config=cfg.to_dict(), arm=job["arm"], mode=job["mode"], control=job["control"],
                    source_sha256=source_hashes, data_file=path.name, data_sha256=data_hash, data=metadata,
                    settings={k: v for k, v in vars(args).items() if k not in
                              ("out_dir", "resume", "stop_after", "dry_run", "list", "data_dir", "files",
                               "session_index", "sessions", "arms", "suite", "architectures", "seeds",
                               "noise_sites", "training_modes", "controls")},
                    references=REFERENCES, torch_version=torch.__version__, numpy_version=np.__version__)
    signature = hashlib.sha256(json.dumps(plain(manifest), sort_keys=True).encode()).hexdigest()
    result_path = directory/"metrics.json"
    if directory.exists() and any(directory.iterdir()):
        if not args.resume:
            raise FileExistsError(f"Existing run: {directory}. Use --resume or a new --out-dir")
        old = json.loads((directory/"manifest.json").read_text())
        if old["signature"] != signature:
            raise ValueError(f"Configuration/data/code changed for {directory}; use a new --out-dir")
        if result_path.exists():
            print(f"SKIP completed {directory}", flush=True)
            return json.loads(result_path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"manifest.json", dict(signature=signature, **manifest))
    np.savez_compressed(directory/"preprocessing.npz", **preprocessing)
    rows = [row_index(s, args.common_warmup, args.behavior_lag,
                      args.max_probe_samples if i == 0 else None, cfg.seed+551)
            for i, s in enumerate((fit, val, final))]
    seed_all(cfg.seed)
    model = ContextJigsaw(cfg).to(args.device)
    print(f"RUN {path.name} {cfg.architecture}/{job['arm']}/{job['control']} "
          f"seed={cfg.seed} {job['mode']} {site} params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    started = time.monotonic()
    model, joint, selected, complete = train_model(model, fit, val, rows[0], cfg, args, job["mode"], job["control"], directory, signature)
    if not complete:
        print(f"Saved interruption at update {selected}; resume with the same command plus --resume", flush=True)
        raise SystemExit(130)
    model.eval()
    features = [extract(model, s, r, cfg, args) for s, r in zip((fit, val, final), rows)]
    representation = dict(participation_ratio=effective_dimension(features[0][0]),
                          feature_std_mean=float(features[0][0].std(0).mean()),
                          dimension=cfg.dimension)
    probe_dir = directory/"embedding_probe"
    probe_dir.mkdir(exist_ok=True)
    scores = probes(features, args, cfg.seed, probe_dir)
    raw_scores = None
    if not args.skip_raw_baseline:
        # Cache ONLY a truly identical input geometry, preprocessing, sample set and probe settings.
        raw_key = dict(session=path.name, data_sha256=data_hash, window=cfg.window, context=cfg.context,
                       gap=cfg.gap_min, use_context=cfg.context_mode != "none", seed=cfg.seed,
                       warmup=args.common_warmup, settings=manifest["settings"], preprocessing=args.preprocess,
                       source=source_hashes)
        cache_id = hashlib.sha256(json.dumps(plain(raw_key), sort_keys=True).encode()).hexdigest()[:16]
        raw_dir = args.out_dir/path.stem/"raw_baselines"/cache_id
        raw_file = raw_dir/"metrics.json"
        if raw_file.exists():
            raw_scores = json.loads(raw_file.read_text())
        else:
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_features = [extract(model, s, r, cfg, args, raw=True) for s, r in zip((fit, val, final), rows)]
            raw_scores = probes(raw_features, args, cfg.seed, raw_dir)
            write_json(raw_file, raw_scores)
            write_json(raw_dir/"manifest.json", raw_key)
    joint_scores = None
    if joint is not None:
        joint["decoder"].eval()
        joint_scores, joint_predictions = {}, {}
        with torch.no_grad():
            for label, (z, y) in zip(("fit", "validation", "final"), features):
                prediction = np.concatenate([(joint["decoder"](torch.from_numpy(z[i:i+args.eval_batch_size]).to(args.device))
                                              *joint["scale"]+joint["mean"]).cpu().numpy()
                                             for i in range(0, len(z), args.eval_batch_size)])
                joint_scores[label] = r2(y, prediction)
                if label == "final":
                    joint_predictions = dict(truth=y, joint=prediction)
        np.savez_compressed(directory/"joint_predictions.npz", **joint_predictions)
    diagnostic = diagnostics(model, fit, val, final, cfg, args)
    np.savez_compressed(directory/"evaluation_rows.npz", fit=rows[0], validation=rows[1], final=rows[2])
    result = dict(session=path.name, architecture=cfg.architecture, arm=job["arm"], control=job["control"],
                  seed=cfg.seed, mode=job["mode"], noise_site=cfg.noise_site, noise_sd=cfg.noise_sd,
                  selected_step=selected, optimizer_budget=args.steps, scores=scores, raw_baseline=raw_scores,
                  joint=joint_scores, representation=representation, pretext=diagnostic,
                  n_fit=len(rows[0]), n_validation=len(rows[1]), n_final=len(rows[2]),
                  elapsed_seconds=time.monotonic()-started, signature=signature,
                  final_results_are_for_reporting_only=True, directory=str(directory.resolve()))
    write_json(result_path, result)
    print(f"DONE ridge R2={scores['ridge']['final']['mean']:.4f}" +
          (f" MLP R2={scores['mlp']['final']['mean']:.4f}" if "mlp" in scores else "") +
          f" effective_dim={representation['participation_ratio']:.2f}", flush=True)
    return result


def write_summary(out_dir):
    results = []
    for path in sorted(out_dir.glob("*/*/*/seed_*/*/*/*/metrics.json")):
        result = json.loads(path.read_text())
        if "arm" not in result:
            continue
        row = {k: result[k] for k in ("session", "architecture", "arm", "control", "seed", "mode", "noise_site", "noise_sd", "selected_step", "n_final")}
        for probe in ("ridge", "mlp"):
            if probe in result["scores"]:
                row[probe+"_final_r2"] = result["scores"][probe]["final"]["mean"]
                row[probe+"_validation_r2"] = result["scores"][probe]["validation"]
            if result["raw_baseline"] and probe in result["raw_baseline"]:
                row["raw_"+probe+"_final_r2"] = result["raw_baseline"][probe]["final"]["mean"]
        if result["joint"]:
            row["joint_final_r2"] = result["joint"]["final"]["mean"]
        row["participation_ratio"] = result["representation"]["participation_ratio"]
        row["signature"] = result["signature"]
        results.append(row)
    if results:
        matching = ("session", "architecture", "arm", "seed", "mode", "noise_site", "noise_sd")
        groups = {}
        for row in results:
            groups.setdefault(tuple(row[k] for k in matching), {})[row["control"]] = row
        for group in groups.values():
            trained = group.get("trained")
            if trained is None:
                continue
            for name in ("random", "frozen", "fresh", "no_order", "supervised_only"):
                if name not in group:
                    continue
                for score in ("ridge_final_r2", "mlp_final_r2", "joint_final_r2"):
                    a, b = trained.get(score), group[name].get(score)
                    if a is not None and b is not None:
                        trained[score+"_delta_vs_"+name] = a-b
        fields = list(dict.fromkeys(k for r in results for k in r))
        temporary = out_dir/"summary.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        temporary.replace(out_dir/"summary.csv")
        # Variability across seeds, not a bin-wise confidence interval. Sessions
        # remain separate; correlated bins are not counted as independent trials.
        group_keys = tuple(k for k in matching if k != "seed") + ("control",)
        grouped = {}
        for row in results:
            grouped.setdefault(tuple(row[k] for k in group_keys), []).append(row)
        aggregated = []
        scores = [k for k in fields if "r2" in k]
        for key, rows in grouped.items():
            row = dict(zip(group_keys, key), n_seeds=len(rows))
            for score in scores:
                values = [r[score] for r in rows if r.get(score) is not None]
                if values:
                    row[score+"_mean"] = float(np.mean(values))
                    row[score+"_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            aggregated.append(row)
        fields = list(dict.fromkeys(k for r in aggregated for k in r))
        with (out_dir/"aggregate.csv.tmp").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(aggregated)
        (out_dir/"aggregate.csv.tmp").replace(out_dir/"aggregate.csv")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("/data/hossein/mm_project/perich_data_valid_final_raw"))
    p.add_argument("--files", type=Path, nargs="+", help="explicit NPZ paths; overrides data-dir/index")
    p.add_argument("--session-index", type=int, default=1, help="ONE-based first index among sorted *.npz")
    p.add_argument("--sessions", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=Path("./perich_context_results"))
    p.add_argument("--suite", choices=SUITES, default="quick")
    p.add_argument("--arms", nargs="+", choices=sorted(ARMS))
    p.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=["residual"])
    p.add_argument("--controls", nargs="*", choices=("random", "frozen", "fresh", "no_order"), default=["random", "frozen"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--training-modes", nargs="+", choices=("frozen", "joint"), default=["frozen"])
    p.add_argument("--noise-sites", nargs="+", choices=("none", "input", "embedding"), default=["none"])
    p.add_argument("--noise-sd", type=float, default=0, help="absolute SD: scaled input units or latent units")
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--context", type=int, default=24)
    p.add_argument("--tiles", type=int, default=4)
    p.add_argument("--gap", type=int, nargs=2, default=[3, 5], metavar=("MIN", "MAX"))
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--dimension", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--preprocess", choices=("zscore", "scale", "none"), default="zscore")
    p.add_argument("--target-columns", nargs="+", type=int, default=[0, 1])
    p.add_argument("--behavior-lag", type=int, default=0, help="nonnegative bins, y[t+lag] from x[:t+1]")
    p.add_argument("--bin-ms", type=float, default=None, help="only if known; no assumed physical bin size")
    p.add_argument("--require-trial-ids", action="store_true")
    p.add_argument("--val-fraction", type=float, default=.15)
    p.add_argument("--steps", type=int, default=6000, help="optimizer updates, NOT epochs")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--supervised-weight", type=float, default=1, help="joint mode behavior MSE coefficient")
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--select-checkpoint", choices=("last", "best"), default="last")
    p.add_argument("--eval-batches", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--diagnostic-samples", type=int, default=512)
    p.add_argument("--seed-diagnostics", type=int, default=78901)
    p.add_argument("--probe-steps", type=int, default=1500, help="0 skips the MLP; ridge always runs")
    p.add_argument("--probe-hidden", type=int, default=64)
    p.add_argument("--probe-dropout", type=float, default=.4)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--probe-eval-every", type=int, default=50)
    p.add_argument("--max-probe-samples", type=int, default=20000, help="cap fit rows; 0 uses all")
    p.add_argument("--ridge-alphas", nargs="+", type=float, default=[.001, .01, .1, 1, 10, 100, 1000])
    p.add_argument("--skip-raw-baseline", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="print the exact job matrix without opening data or training")
    p.add_argument("--list", action="store_true", help="print suites, arms and architectures")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        print(json.dumps(dict(architectures=ARCHITECTURES, suites=SUITES, arms=ARMS, references=REFERENCES), indent=2))
        return
    for key in ("session_index", "sessions", "steps", "batch_size", "eval_batch_size", "diagnostic_samples",
                "eval_batches", "log_every", "checkpoint_every", "probe_eval_every", "threads"):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.behavior_lag < 0 or not 0 < args.val_fraction < .5 or args.noise_sd < 0:
        raise ValueError("Require behavior-lag >=0, 0 < val-fraction <.5, noise-sd >=0")
    if args.batch_size < 2 or args.eval_batch_size < 2 or args.diagnostic_samples < 8:
        raise ValueError("Batch sizes must be >=2 and diagnostic-samples >=8")
    if args.lr <= 0 or args.probe_lr <= 0 or args.grad_clip <= 0 or args.supervised_weight <= 0:
        raise ValueError("Learning rates, grad-clip and supervised-weight must be positive")
    if args.weight_decay < 0 or args.warmup_steps < 0 or args.probe_hidden < 1 or not 0 <= args.probe_dropout < 1:
        raise ValueError("Invalid weight decay, warmup or probe settings")
    if args.bin_ms is not None and args.bin_ms <= 0:
        raise ValueError("--bin-ms must be positive")
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("--stop-after must be positive")
    if args.probe_steps < 0 or args.max_probe_samples < 0 or min(args.ridge_alphas) <= 0 or min(args.seeds) < 0:
        raise ValueError("Invalid probe steps/sample cap/ridge alpha/seed")
    if args.noise_sd and args.noise_sites == ["none"]:
        raise ValueError("--noise-sd >0 needs --noise-sites input and/or embedding")
    jobs = build_jobs(args)
    if any(j["cfg"].view == "count_match" for j in jobs) and args.preprocess == "zscore":
        raise ValueError("This suite includes order_count_match; use --preprocess scale for the whole matched suite")
    args.common_warmup = max(j["cfg"].decode_span for j in jobs)
    purge = max(j["cfg"].max_span for j in jobs)
    print(f"{len(jobs)} runs/session; common decode warmup={args.common_warmup} bins; maximum puzzle={purge} bins", flush=True)
    if args.bin_ms:
        print(f"Base context={args.context*args.bin_ms:g} ms; tile={args.window*args.bin_ms:g} ms", flush=True)
    if args.dry_run:
        for j in jobs:
            c = j["cfg"]
            print(f"{j['arm']:24} {c.architecture:24} seed={c.seed} {j['mode']:6} "
                  f"{j['control']:16} noise={c.noise_site}:{c.noise_sd:g} span<={c.max_span}")
        return
    if args.files:
        files = [p.resolve() for p in args.files]
    else:
        all_files = sorted(args.data_dir.glob("*.npz"))
        first = args.session_index-1
        if first+args.sessions > len(all_files):
            raise ValueError(f"Requested indices {args.session_index}..{first+args.sessions}; found only {len(all_files)} NPZ files")
        files = all_files[first:first+args.sessions]
    if len({p.stem for p in files}) != len(files):
        raise ValueError("Session filenames must have distinct stems")
    torch.set_num_threads(args.threads)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu or install a CUDA PyTorch build")
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: sha256(Path(__file__).with_name(name))
                     for name in ("run_perich_jigsaw.py", "perich_jigsaw_model.py")}
    for path in files:
        print(f"SESSION {path}", flush=True)
        data = load_session(path, args, purge)
        data_hash = sha256(path)
        for job in jobs:
            run_one(path, data, job, args, source_hashes, data_hash)
            write_summary(args.out_dir)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"Results: {args.out_dir/'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()

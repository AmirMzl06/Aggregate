"""Binned-data adapter and supervised training for the OFFICIAL POYO class.

No POYO architecture is reimplemented here. Spike times inside a bin are
synthetic; event multiplicities are preserved for counts / unsmoothed rates.
The train and validation event stores are always separate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import importlib.metadata
import json
import random

import numpy as np
import torch
from torch import nn

TORCH_BRAIN_COMMIT = "ca3cfb691e01764577c3e62a7d5092539f79519a"


@dataclass
class POYOConfig:
    context_bins: int = 100
    steps: int = 3000
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dim: int = 64
    depth: int = 6
    dim_head: int = 64
    cross_heads: int = 2
    self_heads: int = 8
    latent_time_steps: int = 8
    latents_per_step: int = 16
    ffn_dropout: float = 0.2
    lin_dropout: float = 0.4
    atn_dropout: float = 0.2
    standardize_targets: bool = False
    eval_batch_size: int = 16
    print_every: int = 100
    max_tokens_per_window: int = 50000
    seed: int = 42

    def validate(self):
        for key in ("context_bins", "steps", "batch_size", "dim", "depth", "dim_head",
                    "cross_heads", "self_heads", "latent_time_steps", "latents_per_step",
                    "eval_batch_size", "print_every", "max_tokens_per_window"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.dim_head % 4:
            raise ValueError("dim_head must be divisible by 4 for rotary embeddings")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        for key in ("ffn_dropout", "lin_dropout", "atn_dropout"):
            if not 0 <= getattr(self, key) < 1:
                raise ValueError(f"{key} must be in [0, 1)")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_bin_seconds(metadata, explicit_ms=None):
    """Use explicit width or unambiguous metadata. Never infer NPZ rate from its name."""
    candidates = []
    for key, factor in (("bin_size_s", 1.0), ("bin_size_seconds", 1.0),
                        ("bin_size_ms", 0.001), ("dt", 1.0),
                        ("sampling_rate_hz", None), ("sampling_rate", None)):
        if key in metadata:
            value = np.asarray(metadata[key])
            if value.size != 1:
                raise ValueError(f"{key} must be a scalar")
            value = float(value.item())
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid time metadata: {key}={value}")
            candidates.append((key, 1 / value if factor is None else value * factor))
    if explicit_ms is not None:
        dt = float(explicit_ms) / 1000
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("--bin-size-ms must be positive")
        if any(not np.isclose(dt, v, rtol=1e-5, atol=1e-9) for _, v in candidates):
            raise ValueError(f"--bin-size-ms contradicts NPZ timing metadata: {candidates}")
        return dt, "explicit --bin-size-ms"
    if candidates:
        dt = candidates[0][1]
        if any(not np.isclose(dt, v, rtol=1e-5, atol=1e-9) for _, v in candidates):
            raise ValueError(f"Conflicting timing metadata: {candidates}")
        return dt, ", ".join(k for k, _ in candidates)
    raise ValueError(
        "NPZ has no recognized bin-width metadata. Pass --bin-size-ms using the width "
        "of THIS processed NPZ. POYO Appendix C.1 reports 10 ms (100 Hz) for MP data; "
        "use --bin-size-ms 10 only if your NPZ has not been rebinned/downsampled."
    )


def infer_segments(length, metadata, split, dt):
    """Respect optional trial IDs / timestamp discontinuities, without moving rows."""
    cuts = np.zeros(max(length - 1, 0), dtype=bool)
    sources = []
    for key in (f"{split}_trial_ids", f"{split}_trial_id"):
        if key in metadata:
            ids = np.asarray(metadata[key]).reshape(-1)
            if len(ids) != length:
                raise ValueError(f"{key}: expected one trial ID per row")
            cuts |= ids[1:] != ids[:-1]
            sources.append(key)
    key = f"{split}_timestamps"
    if key in metadata:
        t = np.asarray(metadata[key], dtype=np.float64).reshape(-1)
        if len(t) != length or not np.isfinite(t).all():
            raise ValueError(f"Invalid {key}; expected finite seconds, one per row")
        diffs = np.diff(t)
        cuts |= ~np.isclose(diffs, dt, rtol=1e-4, atol=1e-8)
        sources.append(key)
        if length > 1 and np.all(cuts):
            raise ValueError(f"Every row is a boundary: check {key} units and bin size")
    bounds = np.r_[0, np.flatnonzero(cuts) + 1, length]
    segments = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    return segments, (", ".join(sources) if sources else "assumed continuous within split")


def to_counts(x, dt, input_kind="counts", seed=42):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("POYO event conversion needs finite nonnegative (time, neurons) data")
    if input_kind == "counts":
        expected = x
    elif input_kind in ("rates_hz", "poisson_rate"):
        expected = x * dt
    else:
        raise ValueError(f"Unknown input kind: {input_kind}")
    if expected.max() > 1000000 or expected.sum() > 20000000:
        raise ValueError("More than 20M expected events or >1M/bin; check count/rate units")
    if input_kind == "poisson_rate":
        # Explicit surrogate experiment, NOT recovery of measured events.
        return np.random.default_rng(seed).poisson(expected).astype(np.int64)
    if not np.allclose(expected, np.rint(expected), rtol=0, atol=1e-4):
        raise ValueError(
            f"Input is not integer spike counts after conversion from {input_kind}. "
            "Do not threshold or round smoothed/normalized data into events. Supply raw "
            "counts, unsmoothed Hz rates via --input-kind rates_hz, or explicitly choose "
            "--input-kind poisson_rate for a SYNTHETIC Poisson-rate experiment."
        )
    return np.rint(expected).astype(np.int64)


class EventStore:
    """Fixed synthetic timestamps, reused across both label conditions."""
    def __init__(self, counts, dt, segments=None, placement="center", seed=42):
        counts = np.asarray(counts)
        if counts.ndim != 2 or min(counts.shape) < 1:
            raise ValueError("counts must be a nonempty (time, neurons) matrix")
        if not np.isfinite(counts).all() or (counts < 0).any() or np.any(counts != np.rint(counts)):
            raise ValueError("counts must contain nonnegative integers")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive")
        counts = counts.astype(np.int64)
        self.n_bins, self.n_units = counts.shape
        self.dt = float(dt)
        self.placement = placement
        self.segments = segments or [(0, self.n_bins)]
        if (self.segments[0][0] != 0 or self.segments[-1][1] != self.n_bins
                or any(a >= b for a, b in self.segments)
                or any(b != c for (_, b), (c, _) in zip(self.segments[:-1], self.segments[1:]))):
            raise ValueError("segments must partition every row in order")
        row_totals = counts.sum(axis=1)
        self.prefix = np.r_[0, np.cumsum(row_totals)]
        if self.prefix[-1] > 20000000:
            raise ValueError("More than 20M events; check input units")
        bins, units = np.nonzero(counts)
        reps = counts[bins, units]
        self.event_bins = np.repeat(bins, reps)
        self.units = np.repeat(units, reps).astype(np.int64)
        if placement == "center":
            frac = np.full(len(self.units), 0.5)
        elif placement == "uniform":
            frac = np.random.default_rng(seed).uniform(1e-6, 1 - 1e-6, len(self.units))
            # Keep within-bin timestamps sorted; event counts remain unchanged.
            order = np.lexsort((frac, self.event_bins))
            self.event_bins, self.units, frac = self.event_bins[order], self.units[order], frac[order]
        else:
            raise ValueError("placement must be center or uniform")
        self.fractions = frac

    def events(self, start, end):
        lo, hi = self.prefix[start], self.prefix[end]
        times = ((self.event_bins[lo:hi] - start) + self.fractions[lo:hi]) * self.dt
        return self.units[lo:hi], times.astype(np.float32)

    def windows(self, context, stride=1):
        windows = []
        for a, b in self.segments:
            if b - a <= context:
                windows.append((a, b))
            else:
                starts = list(range(a, b - context + 1, stride))
                if starts[-1] != b - context:
                    starts.append(b - context)
                windows.extend((s, s + context) for s in starts)
        return np.asarray(windows, dtype=np.int64)


def make_model(config, output_dim, n_units, session, dt):
    config.validate()
    try:
        from torch_brain.models.poyo import POYO
        from torch_brain.registry import ModalitySpec, DataType
    except ImportError as exc:
        raise ImportError("Install the pinned requirements-poyo.txt; do not pip install 'poyo'.") from exc
    spec = ModalitySpec(id=0, dim=int(output_dim), type=DataType.CONTINUOUS,
                        timestamp_key="behavior.timestamps", value_key="behavior.values",
                        loss_fn=nn.MSELoss())
    duration = float(config.context_bins * dt)
    seed_all(config.seed)
    model = POYO(sequence_length=duration, readout_spec=spec,
                 latent_step=float(duration / config.latent_time_steps),
                 num_latents_per_step=config.latents_per_step, dim=config.dim,
                 depth=config.depth, dim_head=config.dim_head,
                 cross_heads=config.cross_heads, self_heads=config.self_heads,
                 ffn_dropout=config.ffn_dropout, lin_dropout=config.lin_dropout,
                 atn_dropout=config.atn_dropout)
    # Same unit/session initialization even when output_dim changes RNG consumption.
    seed_all(config.seed + 1)
    unit_ids = [f"{session}/unit_{i}" for i in range(n_units)]
    model.unit_emb.initialize_vocab(unit_ids)
    model.session_emb.initialize_vocab([session])
    # Official SDPA path; avoid xformers/CUDA version coupling.
    for module in model.modules():
        if hasattr(module, "use_xformers"):
            module.use_xformers = False
    return model, unit_ids


class TokenBatcher:
    """Official token conventions: spike=0, start=1, end=2; True masks are valid."""
    def __init__(self, model, unit_ids, session, config, dt):
        from torch_brain.utils import create_linspace_latent_tokens
        self.unit_map = np.asarray(model.unit_emb.tokenizer(unit_ids), dtype=np.int64)
        self.session_index = int(model.session_emb.tokenizer(session))
        self.config, self.dt = config, dt
        self.latent_index, self.latent_times = create_linspace_latent_tokens(
            0., model.sequence_length, model.latent_step, config.latents_per_step)

    def __call__(self, store, windows, device, targets=None):
        B, C = len(windows), store.n_units
        entries = []
        for start, end in windows:
            units, times = store.events(start, end)
            n = len(units) + 2 * C
            if n > self.config.max_tokens_per_window:
                raise ValueError(f"{n} tokens/window exceeds limit; reduce context or check input units")
            uid = np.r_[np.repeat(self.unit_map, 2), self.unit_map[units]]
            tt = np.r_[np.tile([1, 2], C), np.zeros(len(units), dtype=np.int64)]
            ts = np.r_[np.tile([0., (end - start) * self.dt], C), times]
            entries.append((uid, tt, ts))
        max_in = ((max(len(e[0]) for e in entries) + 7) // 8) * 8
        max_out = int(max(b - a for a, b in windows))
        arrays = dict(
            input_unit_index=np.zeros((B, max_in), dtype=np.int64),
            input_timestamps=np.zeros((B, max_in), dtype=np.float32),
            input_token_type=np.zeros((B, max_in), dtype=np.int64),
            input_mask=np.zeros((B, max_in), dtype=bool),
            latent_index=np.tile(self.latent_index, (B, 1)).astype(np.int64),
            latent_timestamps=np.tile(self.latent_times, (B, 1)).astype(np.float32),
            output_session_index=np.full((B, max_out), self.session_index, dtype=np.int64),
            output_timestamps=np.zeros((B, max_out), dtype=np.float32),
            output_mask=np.zeros((B, max_out), dtype=bool),
        )
        yy = None if targets is None else np.zeros((B, max_out, targets.shape[1]), dtype=np.float32)
        for i, ((a, b), (units, types, times)) in enumerate(zip(windows, entries)):
            n, length = len(units), int(b - a)
            arrays["input_unit_index"][i, :n] = units
            arrays["input_token_type"][i, :n] = types
            arrays["input_timestamps"][i, :n] = times
            arrays["input_mask"][i, :n] = True
            arrays["output_timestamps"][i, :length] = (np.arange(length) + 0.5) * self.dt
            arrays["output_mask"][i, :length] = True
            if yy is not None:
                yy[i, :length] = targets[a:b]
        result = {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}
        return result, None if yy is None else torch.as_tensor(yy, device=device)


@torch.inference_mode()
def predict(model, store, batcher, config, device):
    model.eval()
    windows = store.windows(config.context_bins, stride=max(1, config.context_bins // 2))
    pred = np.zeros((store.n_bins, model.readout_spec.dim), dtype=np.float64)
    votes = np.zeros(store.n_bins, dtype=np.int64)
    for begin in range(0, len(windows), config.eval_batch_size):
        chunk = windows[begin:begin + config.eval_batch_size]
        inputs, _ = batcher(store, chunk, device)
        output = model(**inputs).cpu().numpy()
        for i, (a, b) in enumerate(chunk):
            pred[a:b] += output[i, :b-a]
            votes[a:b] += 1
    if not np.all(votes > 0) or not np.isfinite(pred).all():
        raise RuntimeError("Evaluation stitching left missing/nonfinite predictions")
    return (pred / votes[:, None]).astype(np.float32)


def train_poyo(train_store, valid_store, y_train, y_valid, session, config, device, out_dir):
    """Train from scratch, fixed step budget, no validation-based model selection."""
    config.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(y_train) != train_store.n_bins or len(y_valid) != valid_store.n_bins:
        raise ValueError("POYO label / neural row count mismatch")
    y_train = np.asarray(y_train, dtype=np.float32)
    y_valid = np.asarray(y_valid, dtype=np.float32)
    if y_train.ndim != 2 or y_valid.ndim != 2 or y_train.shape[1] != y_valid.shape[1]:
        raise ValueError("Expected matching 2D label matrices")
    mean = y_train.mean(0) if config.standardize_targets else np.zeros(y_train.shape[1], np.float32)
    scale = y_train.std(0) if config.standardize_targets else np.ones(y_train.shape[1], np.float32)
    scale = np.where(scale < 1e-6, 1., scale).astype(np.float32)
    targets = (y_train - mean) / scale
    model, unit_ids = make_model(config, y_train.shape[1], train_store.n_units, session, train_store.dt)
    model.to(device)
    batcher = TokenBatcher(model, unit_ids, session, config, train_store.dt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    candidates = train_store.windows(config.context_bins)
    rng = np.random.default_rng(config.seed + 2)
    seed_all(config.seed + 3)
    print(f"POYO official: parameters={sum(p.numel() for p in model.parameters()):,}, "
          f"outputs={y_train.shape[1]}, context={config.context_bins} bins "
          f"({model.sequence_length:.6g}s), steps={config.steps}, "
          f"input events={len(train_store.units):,}, target_standardization={config.standardize_targets}", flush=True)
    history = []
    model.train()
    for step in range(1, config.steps + 1):
        windows = candidates[rng.integers(len(candidates), size=config.batch_size)]
        inputs, truth = batcher(train_store, windows, device, targets)
        optimizer.zero_grad(set_to_none=True)
        output = model(**inputs)
        mask = inputs["output_mask"]
        loss = nn.functional.mse_loss(output[mask], truth[mask])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite POYO loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append(float(loss.detach()))
        if step == 1 or step % config.print_every == 0 or step == config.steps:
            print(f"POYO {step}/{config.steps}: train MSE={history[-1]:.6f}", flush=True)
    pred_train = predict(model, train_store, batcher, config, device) * scale + mean
    pred_valid = predict(model, valid_store, batcher, config, device) * scale + mean
    provenance = {"torch_brain_version": importlib.metadata.version("torch_brain"),
                  "expected_commit": TORCH_BRAIN_COMMIT, "torch_version": str(torch.__version__),
                  "model_class": f"{type(model).__module__}.{type(model).__name__}",
                  "time_source": "synthetic within-bin event timestamps", "pretrained": False,
                  "optimizer": "AdamW", "causal": False}
    try:
        provenance["installed_direct_url"] = json.loads(
            importlib.metadata.distribution("torch_brain").read_text("direct_url.json") or "null")
    except (ValueError, TypeError):
        pass
    torch.save({"state_dict": model.state_dict(), "config": asdict(config),
                "unit_ids": unit_ids, "session": session, "bin_size_s": train_store.dt,
                "n_outputs": y_train.shape[1], "target_mean": torch.from_numpy(mean),
                "target_scale": torch.from_numpy(scale), "provenance": provenance}, out_dir / "poyo.pt")
    np.savez_compressed(out_dir / "predictions.npz", Y_train=y_train, Y_valid=y_valid,
                        pred_train=pred_train, pred_valid=pred_valid)
    (out_dir / "training.json").write_text(json.dumps(
        {"config": asdict(config), "provenance": provenance, "loss": history}, indent=2))
    return pred_train, pred_valid

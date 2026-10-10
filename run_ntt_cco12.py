#!/usr/bin/env python3
"""C-CO12 Neural Transformation Tournament experiments (two-file distribution).

Place ntt_architectures.py beside this file. Dependencies:
    python -m pip install torch numpy scipy scikit-learn matplotlib

Recommended first run (four objectives, paired initialization, 3 seeds):
    python run_ntt_cco12.py --architectures tcn --seeds 42 43 44

All seven backbones (substantially more compute):
    python run_ntt_cco12.py --architectures all --seeds 42 43 44

Quick wiring test, explicitly synthetic (NOT a Perich performance result):
    python run_ntt_cco12.py --smoke-test --device cpu

Optional ablations:
    --arms rank rank_shuffled consistency reg_only rank_only rank_consistency
    --decode-spaces h z
    --transform count_jitter       # only if arrays are RAW integer counts
    --include-cebra supervised time
    --causal-window               # 36 bins ending at label timestamp
    --resume /path/to/existing/run # same experimental arguments required
    --self-test                    # architecture/transform/gradient tests only

Protocol:
* Original NPZ train/valid split and all label columns are retained.
* Default windows match reference code: [t-18, ..., t+17], with edge padding.
  This is an OFFLINE centered-window protocol, not online/causal decoding.
* Encoder pretraining never receives behavior labels. Last 15% of TRAIN is
  reserved for pretext diagnostics, separated by a window-sized embargo.
* Input normalization is fitted on the pretraining subsection only.
* At each saved step a NEW downstream decoder is trained from scratch on all
  train embeddings, including at step 0. Random encoder == zero encoder steps;
  the decoder is always trained. Decoder initialization is paired across steps.
* Primary decoder exactly matches the supplied script: Linear(d,64), LayerNorm,
  ReLU, Dropout(.4), Linear(64,Y); Adam(lr=.001), raw-label MSE, full-batch,
  2500 epochs by default, no early stopping and no validation selection.
* The same exact encoder initialization and pretraining minibatches/transforms
  are reused for all objective arms within each architecture/seed.
* Primary outcome: clean valid mean R2 and per-output R2, plus paired delta vs
  that architecture/seed's step-0 encoder. Also save MSE, MAE, Pearson, a ridge
  probe, collapse diagnostics, held-out pretext ranks, negative controls,
  finite-difference geometry, permutation shortcuts, learning curves and CIs.
* Validation checkpoints are descriptive. No automatic best-valid selection.
  Repeatedly inspecting this split makes it a development set, not a fresh test.
* Trial IDs are honored if train_trial_id/valid_trial_id exist (keys configurable).
  Without IDs the provided row order is assumed continuous. The runner cannot
  detect trial boundaries or repair leakage already present in the NPZ split.
* Transform strength is in BINS; bin_ms is optional reporting metadata only.
  bin_permute is the default and works with counts, rates or signed features.
  It preserves per-neuron bin-value multisets inside fixed coarse blocks. It
  is NOT a claim of sub-bin spike jitter or monotonic behavioral damage.
* Optional CEBRA-supervised uses labels and is marked as a supervised reference;
  CEBRA-time uses no behavior labels. Neither is an identical-architecture test.

Outputs:
  run_config.json, dataset_audit.json, results.csv, summary.json,
  controls.csv, controls_summary.json, comparisons.json, learning_curves.png;
  each checkpoint: metrics, predictions, decoder loss and state, encoder state;
  each arm: training.csv and latest_training.pt (resumable incl. RNG/optimizer).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback
import warnings

import numpy as np
import torch
from torch import nn
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from ntt_architectures import (
    ARCHITECTURES, OBJECTIVES, TRANSFORMS, EncoderConfig, TransformConfig,
    TournamentEncoder, make_tournament, tournament_loss, parameter_count,
    ranking_diagnostics, representation_diagnostics,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path('/data/hossein/mm_project/perich_data_valid_final_raw/')
MLP_HIDDEN, MLP_DROP, MLP_LR = 64, 0.4, 1e-3


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def load_own_checkpoint(path):
    # Load only checkpoints produced by this runner; pickle is unsafe for
    # untrusted files. Explicit weights_only accommodates PyTorch >=2.6.
    return torch.load(path, map_location='cpu', weights_only=False)


def state_on_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


@contextlib.contextmanager
def isolated_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def segment_bounds(ids, n):
    if ids is None:
        ids = np.zeros(n, dtype=np.int64)
    if np.asarray(ids).shape != (n,):
        raise ValueError('Trial/segment IDs must be a length-T vector.')
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    lower, upper = np.empty(n, np.int64), np.empty(n, np.int64)
    for lo, hi in zip(starts, ends):
        lower[lo:hi], upper[lo:hi] = lo, hi - 1
    return lower, upper, list(zip(starts.tolist(), ends.tolist()))


class WindowSource:
    """Windows are built lazily on CPU; no giant T x N x W materialization."""
    def __init__(self, data, left, right, ids=None):
        self.data = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
        self.left, self.right, self.window = left, right, left + right
        self.lower, self.upper, self.segments = segment_bounds(ids, len(data))
        self.lower_t, self.upper_t = torch.from_numpy(self.lower), torch.from_numpy(self.upper)
        self.offsets = torch.arange(-left, right)
        self.interior = np.flatnonzero((np.arange(len(data)) - left >= self.lower) &
                                       (np.arange(len(data)) + right - 1 <= self.upper))

    def get(self, centers):
        centers = torch.as_tensor(centers, dtype=torch.long, device='cpu')
        index = centers[:, None] + self.offsets[None, :]
        index = torch.maximum(index, self.lower_t[centers, None])
        index = torch.minimum(index, self.upper_t[centers, None])
        return self.data[index].transpose(1, 2).contiguous()

    def __len__(self):
        return len(self.data)


def load_data(args):
    path = args.data_dir.expanduser() / f'{args.session}.npz'
    if not path.is_file():
        raise FileNotFoundError(f'Dataset not found: {path}')
    names = ('train_data', 'valid_data', 'train_label', 'valid_label')
    with np.load(path, allow_pickle=False) as archive:
        for name in names:
            if name not in archive:
                raise KeyError(f'{path} is missing {name}.')
        arrays = [np.asarray(archive[name], dtype=np.float32) for name in names]
        train_ids = np.asarray(archive[args.train_segments_key]) if args.train_segments_key in archive else None
        valid_ids = np.asarray(archive[args.valid_segments_key]) if args.valid_segments_key in archive else None
        keys = list(archive.files)
    for i in (2, 3):
        if arrays[i].ndim == 1:
            arrays[i] = arrays[i][:, None]
    for name, value in zip(names, arrays):
        if value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all():
            raise ValueError(f'{name} must be a finite nonempty 2D array: {value.shape}')
    xt, xv, yt, yv = arrays
    if len(xt) != len(yt) or len(xv) != len(yv):
        raise ValueError('Feature/label lengths do not match.')
    if xt.shape[1] != xv.shape[1] or yt.shape[1] != yv.shape[1]:
        raise ValueError('Train/valid dimensions do not match.')
    if len(xt) < 8 * args.window or len(xv) < 2:
        raise ValueError('Insufficient data for train-only holdout and windows.')
    if train_ids is None or valid_ids is None:
        warnings.warn('Trial IDs absent for at least one split: continuity of NPZ row order is assumed. '
                      'Provide segment ID keys if rows concatenate trials.')
    left = args.window - 1 if args.causal_window else args.window // 2
    right = args.window - left
    train = WindowSource(xt, left, right, train_ids)
    valid = WindowSource(xv, left, right, valid_ids)
    cut = int(len(xt) * (1 - args.pretext_holdout))
    # Place the internal cut at a trial boundary when possible.
    boundaries = np.array([lo for lo, _ in train.segments if lo > args.window * 2 and lo < len(xt) - args.window * 2])
    if len(boundaries):
        cut = int(boundaries[np.argmin(abs(boundaries - cut))])
    fit_end, hold_start = cut - args.window, cut + args.window
    fit_centers = train.interior[train.interior + right <= fit_end]
    held_centers = train.interior[train.interior - left >= hold_start]
    if min(len(fit_centers), len(held_centers)) < 8:
        raise ValueError('Too few unpadded windows for pretraining/holdout; reduce window or supply longer segments.')
    if args.standardize:
        mean, std = xt[:fit_end].mean(0), xt[:fit_end].std(0).clip(1e-6)
    else:
        mean, std = np.zeros(xt.shape[1], np.float32), np.ones(xt.shape[1], np.float32)
    values = xt.reshape(-1)
    audit_sample = values[::max(1, len(values) // 100000)]
    count_like = bool((audit_sample >= 0).all() and
                      np.allclose(audit_sample, np.rint(audit_sample), atol=1e-5, rtol=0))
    if args.transform == 'count_jitter':
        # Validate the WHOLE array rather than trusting the sample above.
        for start in range(0, len(xt), 8192):
            chunk = xt[start:start + 8192]
            if (chunk < 0).any() or not np.allclose(chunk, np.rint(chunk), atol=1e-5, rtol=0):
                raise ValueError('count_jitter requires integer counts. Use bin_permute for rates/normalized data.')
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    for ids in (train_ids, valid_ids):
        if ids is not None:
            digest.update(str(ids.dtype).encode())
            digest.update(np.ascontiguousarray(ids).tobytes())
    audit = dict(path=str(path.resolve()), keys=keys, shapes=[list(a.shape) for a in arrays],
                 sha256_loaded_arrays=digest.hexdigest(), count_like_sample=count_like,
                 value_min=float(xt.min()), value_max=float(xt.max()),
                 n_constant_neurons=int((xt[:fit_end].std(0) < 1e-6).sum()),
                 constant_label_columns_train=np.flatnonzero(yt.std(0) == 0).tolist(),
                 constant_label_columns_valid=np.flatnonzero(yv.std(0) == 0).tolist(),
                 window_offsets=[-left, right - 1], bin_ms=args.bin_ms,
                 online_causal=args.causal_window,
                 train_segment_count=len(train.segments), valid_segment_count=len(valid.segments),
                 train_ids_available=train_ids is not None, valid_ids_available=valid_ids is not None,
                 encoder_fit_rows=[0, fit_end], pretext_holdout_rows=[hold_start, len(xt)],
                 embargo_rows=[fit_end, hold_start],
                 n_ssl_centers=len(fit_centers), n_heldout_centers=len(held_centers),
                 valid_interior_count=len(valid.interior),
                 warning='Original NPZ split integrity and temporal row order cannot be verified here.')
    return dict(path=path, xt=xt, xv=xv, yt=yt, yv=yv, train=train, valid=valid,
                fit_centers=fit_centers, held_centers=held_centers, mean=mean, std=std,
                fit_end=fit_end, hold_start=hold_start, audit=audit)


class TwoLayerMLP(nn.Module):
    """Exact downstream architecture from the supplied CEBRA/ACORN runner."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, MLP_HIDDEN), nn.LayerNorm(MLP_HIDDEN),
                                 nn.ReLU(), nn.Dropout(MLP_DROP), nn.Linear(MLP_HIDDEN, output_dim))

    def forward(self, value):
        return self.net(value)


def train_decoder(z_train, y_train, seed, device, epochs, verbose=False):
    seed_all(seed)
    decoder = TwoLayerMLP(z_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=MLP_LR)
    criterion = nn.MSELoss()
    z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    losses = []
    decoder.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(decoder(z), y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite decoder loss at epoch {epoch + 1}.')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if verbose and ((epoch + 1) % 500 == 0 or epoch + 1 == epochs):
            print(f'  decoder {epoch + 1}/{epochs}: MSE={losses[-1]:.6f}', flush=True)
    return decoder, losses


@torch.inference_mode()
def decoder_predictions(decoder, embeddings, batch_size):
    decoder.eval()
    device = next(decoder.parameters()).device
    return np.concatenate([decoder(torch.as_tensor(embeddings[start:start + batch_size],
                                                   dtype=torch.float32, device=device)).cpu().numpy()
                           for start in range(0, len(embeddings), batch_size)])


def regression_metrics(y, prediction, train_label_variance):
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Predictions contain NaN or infinity.')
    r2 = np.atleast_1d(r2_score(y, prediction, multioutput='raw_values'))
    err = prediction.astype(np.float64) - y
    yc = y.astype(np.float64) - y.mean(0)
    pc = prediction.astype(np.float64) - prediction.mean(0)
    denom = np.sqrt((yc ** 2).sum(0) * (pc ** 2).sum(0))
    corr = np.divide((yc * pc).sum(0), denom, out=np.full(y.shape[1], np.nan), where=denom > 1e-12)
    mse, mae = (err ** 2).mean(0), abs(err).mean(0)
    return clean_json(dict(mean_r2=float(r2.mean()), r2_per_output=r2.tolist(),
                           mse=float(mse.mean()), mse_per_output=mse.tolist(),
                           mae=float(mae.mean()), mae_per_output=mae.tolist(),
                           nmse_train_variance=float((mse / np.maximum(train_label_variance, 1e-12)).mean()),
                           pearson_per_output=corr.tolist(),
                           r2_constant_output_policy='sklearn force_finite=True'))


def auxiliary_ridge(z_train, z_valid, data):
    fit, held = data['fit_centers'], data['held_centers']
    scaler = StandardScaler().fit(z_train[fit])
    xfit, xheld = scaler.transform(z_train[fit]), scaler.transform(z_train[held])
    best_score, best_alpha = -np.inf, None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        model = Ridge(alpha=alpha).fit(xfit, data['yt'][fit])
        score = r2_score(data['yt'][held], model.predict(xheld), multioutput='uniform_average')
        if score > best_score:
            best_score, best_alpha = score, alpha
    scaler = StandardScaler().fit(z_train)
    model = Ridge(alpha=best_alpha).fit(scaler.transform(z_train), data['yt'])
    prediction = model.predict(scaler.transform(z_valid))
    return dict(alpha=best_alpha, train_only_selection_r2=best_score,
                valid=regression_metrics(data['yv'], prediction, data['yt'].var(0)))


@torch.inference_mode()
def extract_embeddings(model, source, spaces, device, batch_size):
    model.eval()
    parts = {space: [] for space in spaces}
    for start in range(0, len(source), batch_size):
        result = model(source.get(np.arange(start, min(start + batch_size, len(source)))).to(device))
        for space in spaces:
            parts[space].append(result[space].cpu().numpy())
    return {space: np.ascontiguousarray(np.concatenate(values), dtype=np.float32)
            for space, values in parts.items()}


def block_bootstrap_delta(y, prediction, reference, segments, block_size, repeats, seed):
    """Non-overlapping block bootstrap, paired predictions; respects segment edges.

    Conditional on these fitted models and this session. Not a population-level
    significance test. Sufficient statistics avoid O(repeats * T) copies.
    """
    if repeats <= 0:
        return None
    stats = []
    y = y.astype(np.float64)
    for lo, hi in segments:
        for start in range(lo, hi, block_size):
            stop = min(hi, start + block_size)
            a = y[start:stop]
            stats.append((len(a), a.sum(0), (a * a).sum(0),
                          ((prediction[start:stop] - a) ** 2).sum(0),
                          ((reference[start:stop] - a) ** 2).sum(0)))
    if len(stats) < 5:
        return dict(status='fewer_than_5_blocks', n_blocks=len(stats), ci95=None)
    count = np.array([s[0] for s in stats])
    sy, sy2, se, se0 = [np.stack([s[i] for s in stats]) for i in range(1, 5)]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        indices = rng.integers(0, len(stats), len(stats))
        total = sy2[indices].sum(0) - sy[indices].sum(0) ** 2 / count[indices].sum()
        usable = total > 1e-12
        if usable.any():
            estimates.append(((se0[indices].sum(0) - se[indices].sum(0))[usable] / total[usable]).mean())
    return dict(status='ok' if estimates else 'constant_labels', n_blocks=len(stats),
                block_size_bins=block_size, repeats=len(estimates),
                ci95=np.quantile(estimates, [.025, .975]).tolist() if estimates else None,
                interpretation='paired validation blocks, conditional on fitted models; no cross-session inference')


def statistical_features(windows):
    """Per-neuron means, std, adjacent covariance and first-difference energy."""
    x = windows.float()
    mean = x.mean(-1)
    centered = x - mean.unsqueeze(-1)
    variance = centered.square().mean(-1)
    lag = (centered[..., 1:] * centered[..., :-1]).mean(-1)
    rough = (x[..., 1:] - x[..., :-1]).square().mean(-1)
    return torch.cat((mean, variance.add(1e-8).sqrt(), lag, rough), -1)


def build_transform(args, levels=None):
    return TransformConfig(args.transform, tuple(args.levels if levels is None else levels),
                           args.block_size, args.max_events).validate()


@torch.inference_mode()
def encode_views(model, views, device, batch_size):
    v, b, n, t = views.shape
    flat = views.reshape(v * b, n, t)
    parts = {'h': [], 'z': []}
    model.eval()
    for start in range(0, len(flat), batch_size):
        result = model(flat[start:start + batch_size].to(device))
        for key in parts:
            parts[key].append(result[key].cpu())
    return {key: torch.cat(values).reshape(v, b, -1) for key, values in parts.items()}


def fixed_centers(centers, count, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(centers, min(count, len(centers)), replace=False))


@torch.inference_mode()
def pretext_diagnostics(model, data, args, seed, device):
    centers = fixed_centers(data['held_centers'], args.diagnostic_samples, seed + 701)
    windows = data['train'].get(centers)
    result = {}
    for name, levels, shared in [
        ('heldout_seen_strengths', args.levels, False),
        ('heldout_unseen_strengths', args.eval_levels, False),
        ('shared_neuron_time_transform', args.levels, True),
    ]:
        if shared and args.transform == 'count_jitter':
            continue
        generator = torch.Generator().manual_seed(seed + 711)
        views, audit = make_tournament(windows, build_transform(args, levels), generator, shared)
        encoded = encode_views(model, views, device, args.encoder_eval_batch)
        entry = dict(levels=levels, samples=len(centers),
                     transform_audit={k: v.mean(1).tolist() if torch.is_tensor(v) else v
                                      for k, v in audit.items()})
        for space, values in encoded.items():
            distances = (values[1:] - values[0]).square().sum(-1).T
            entry[space + '_ranking'] = ranking_diagnostics(distances)
        raw_distances = (views[1:] - views[0]).square().mean((2, 3)).T
        entry['raw_l2_ranking'] = ranking_diagnostics(raw_distances)
        # Features standardized using clean TRAIN windows, never valid data.
        stats = torch.stack([statistical_features(view) for view in views])
        denom = stats[0].std(0, unbiased=False).clamp_min(1e-5)
        entry['statistical_distance_ranking'] = ranking_diagnostics(
            (((stats[1:] - stats[0]) / denom).square().mean(-1)).T)
        power = torch.fft.rfft(views, dim=-1, norm='ortho').abs().square().log1p()
        entry['power_spectrum_distance_ranking'] = ranking_diagnostics(
            (power[1:] - power[0]).square().mean((2, 3)).T)
        entry['strict_full_order_random_chance'] = 1 / math.factorial(len(levels))
        result[name] = entry
    return result


def fit_shortcut_audit(data, args, seed):
    """Label-free in neuroscience terms: predict synthetic severity, not behavior."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    centers_fit = fixed_centers(data['fit_centers'], args.diagnostic_samples, seed + 951)
    centers_held = fixed_centers(data['held_centers'], args.diagnostic_samples, seed + 952)
    parts = []
    for centers in (centers_fit, centers_held):
        windows = data['train'].get(centers)
        views, _ = make_tournament(windows, build_transform(args), torch.Generator().manual_seed(seed + 953))
        features = torch.cat([statistical_features(view) for view in views[1:]]).numpy()
        labels = np.repeat(np.arange(len(args.levels)), len(centers))
        parts.append((features, labels, len(centers)))
    x, labels, _ = parts[0]
    held_x, held_y, count = parts[1]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, random_state=seed))
    model.fit(x, labels)
    probabilities = model.predict_proba(held_x)
    expectation = probabilities @ np.arange(len(args.levels))
    ranking = ranking_diagnostics(torch.from_numpy(expectation.reshape(len(args.levels), count).T))
    return dict(features='per-neuron mean/std/lag1 covariance/difference energy',
                heldout_accuracy=float((probabilities.argmax(1) == held_y).mean()),
                class_chance=1 / len(args.levels), heldout_ranking=ranking,
                interpretation='A successful nuisance probe demonstrates a shortcut is available; '
                               'a failed probe does not establish shortcut freedom.')


@torch.inference_mode()
def corruption_sensitivity(model, decoder, data, args, seed, space, device):
    if args.robustness_samples <= 0:
        return None
    candidates = data['valid'].interior
    if not len(candidates):
        return dict(status='no_unpadded_validation_windows')
    centers = fixed_centers(candidates, args.robustness_samples, seed + 812)
    views, _ = make_tournament(data['valid'].get(centers), build_transform(args),
                               torch.Generator().manual_seed(seed + 813))
    features = encode_views(model, views, device, args.encoder_eval_batch)[space].numpy()
    rows = []
    for level, z in zip([0.0, *args.levels], features):
        prediction = decoder_predictions(decoder, z, args.decoder_eval_batch)
        rows.append(dict(level=level, **regression_metrics(data['yv'][centers], prediction,
                                                          data['yt'].var(0))))
    return dict(samples=len(centers), values=rows,
                interpretation='Sensitivity to corruption using ORIGINAL targets; '
                               'not proof these targets remain correct for transformed activity.')


def evaluate_features(z_train, z_valid, data, args, device, folder, metadata,
                      reference_folder=None, model=None, space='h'):
    """Each decoder repeat is an independent, paired full-batch supervised probe."""
    results = []
    seed = metadata['seed']
    label_variance = data['yt'].var(0)
    if z_train.shape[1] != z_valid.shape[1] or not np.isfinite(z_train).all() or not np.isfinite(z_valid).all():
        raise ValueError('Invalid embeddings.')
    for repeat in range(args.decoder_repeats):
        destination = folder / f'decoder_{repeat}'
        metrics_file = destination / 'metrics.json'
        if metrics_file.is_file():
            results.append(json.loads(metrics_file.read_text()))
            continue
        destination.mkdir(parents=True, exist_ok=True)
        decoder_seed = seed + 100_000 + repeat * 10_000
        started = time.perf_counter()
        with isolated_rng():
            decoder, losses = train_decoder(z_train, data['yt'], decoder_seed, device,
                                             args.decoder_epochs, args.verbose_decoder)
            prediction_train = decoder_predictions(decoder, z_train, args.decoder_eval_batch)
            prediction_valid = decoder_predictions(decoder, z_valid, args.decoder_eval_batch)
            train_metrics = regression_metrics(data['yt'], prediction_train, label_variance)
            valid_metrics = regression_metrics(data['yv'], prediction_valid, label_variance)
            interior = data['valid'].interior
            interior_metrics = regression_metrics(data['yv'][interior], prediction_valid[interior], label_variance) if len(interior) > 1 else None
            diag_centers = fixed_centers(np.arange(len(z_train)), args.diagnostic_samples, seed + 621)
            geometry = representation_diagnostics(torch.from_numpy(z_train[diag_centers]))
            row = dict(**metadata, feature_space=space, decoder_repeat=repeat,
                       decoder_seed=decoder_seed, decoder_epochs=args.decoder_epochs,
                       feature_dim=z_train.shape[1], train=train_metrics, valid=valid_metrics,
                       valid_unpadded=interior_metrics, generalization_gap=train_metrics['mean_r2'] - valid_metrics['mean_r2'],
                       geometry=geometry, seconds=time.perf_counter() - started,
                       prediction_path=str((destination / 'validation_predictions.npz').resolve()),
                       delta_vs_init=None, delta_vs_init_per_output=None, paired_block_bootstrap=None)
            if reference_folder is not None:
                refdir = reference_folder / f'decoder_{repeat}'
                refrow = json.loads((refdir / 'metrics.json').read_text())
                with np.load(refdir / 'validation_predictions.npz') as archive:
                    refprediction = archive['y_pred']
                    if not np.array_equal(archive['y_true'], data['yv']):
                        raise ValueError('Reference predictions do not match current validation labels.')
                row['delta_vs_init'] = valid_metrics['mean_r2'] - refrow['valid']['mean_r2']
                row['delta_vs_init_per_output'] = (np.array(valid_metrics['r2_per_output']) -
                                                   refrow['valid']['r2_per_output']).tolist()
                row['paired_block_bootstrap'] = block_bootstrap_delta(
                    data['yv'], prediction_valid, refprediction, data['valid'].segments,
                    args.bootstrap_block, args.bootstrap_repeats, seed + 411 + repeat)
            if args.ridge_probe and repeat == 0:
                row['ridge_probe'] = auxiliary_ridge(z_train, z_valid, data)
            if model is not None and metadata['iteration'] in (0, args.max_iter):
                row['corruption_sensitivity'] = corruption_sensitivity(
                    model, decoder, data, args, seed, space, device)
            np.save(destination / 'decoder_train_mse.npy', np.asarray(losses, np.float32))
            np.savez_compressed(destination / 'validation_predictions.npz',
                                y_true=data['yv'], y_pred=prediction_valid,
                                centers=np.arange(len(prediction_valid)))
            if args.save_checkpoints:
                save_torch(destination / 'decoder.pt', dict(
                    state_dict=state_on_cpu(decoder), input_dim=z_train.shape[1],
                    output_dim=data['yt'].shape[1], hidden=MLP_HIDDEN,
                    dropout=MLP_DROP, learning_rate=MLP_LR, seed=decoder_seed))
            save_json(metrics_file, row)  # Written LAST: completion marker.
            results.append(clean_json(row))
            print(f"  {metadata['architecture']} | {metadata['arm']} | step={metadata['iteration']} "
                  f"| {space} | seed={seed} | decoder={repeat}: valid R2={valid_metrics['mean_r2']:.6f} "
                  f"delta0={row['delta_vs_init']}", flush=True)
            del decoder
    return results


def evaluate_checkpoint(model, data, args, device, root, architecture, seed, arm, step, init_sha,
                        encoder_seconds=0.0):
    folder = root / architecture / f'seed_{seed}' / arm / f'step_{step:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    complete = [folder / space / f'decoder_{r}' / 'metrics.json'
                for space in args.decode_spaces for r in range(args.decoder_repeats)]
    if all(path.is_file() for path in complete) and (folder / 'pretext.json').is_file():
        return [json.loads(path.read_text()) for path in complete]
    with isolated_rng():
        model.eval()
        pretrained_sha = state_digest(model.state_dict())
        z_train = extract_embeddings(model, data['train'], args.decode_spaces, device, args.encoder_eval_batch)
        z_valid = extract_embeddings(model, data['valid'], args.decode_spaces, device, args.encoder_eval_batch)
        save_json(folder / 'pretext.json', pretext_diagnostics(model, data, args, seed, device))
        metadata = dict(session=args.session, seed=seed, architecture=architecture, arm=arm,
                        iteration=step, label_use='encoder:none; decoder:all columns',
                        init_sha256=init_sha, encoder_sha256=pretrained_sha,
                        parameter_count=parameter_count(model), encoder_seconds=encoder_seconds,
                        encoder_config=model.config_dict(), synthetic=args.smoke_test,
                        pretext_path=str((folder / 'pretext.json').resolve()))
        rows = []
        for space in args.decode_spaces:
            reference = None if step == 0 else root / architecture / f'seed_{seed}' / 'init' / 'step_000000' / space
            rows.extend(evaluate_features(z_train[space], z_valid[space], data, args, device,
                                           folder / space, metadata, reference, model, space))
        if args.save_checkpoints:
            save_torch(folder / 'encoder.pt', dict(state_dict=state_on_cpu(model),
                                                   config=model.config_dict(), step=step,
                                                   init_sha256=init_sha))
        if args.save_embeddings:
            np.savez_compressed(folder / 'embeddings.npz',
                                **{f'train_{key}': value for key, value in z_train.items()},
                                **{f'valid_{key}': value for key, value in z_valid.items()})
        if state_digest(model.state_dict()) != pretrained_sha:
            raise RuntimeError('Evaluation unexpectedly mutated encoder weights/buffers.')
        del z_train, z_valid
    model.train()
    cleanup()
    return rows


def train_arm(config, initial_state, data, args, device, root, seed, arm, init_sha):
    """No labels or validation arrays are read inside the optimization step."""
    folder = root / config.architecture / f'seed_{seed}' / arm
    folder.mkdir(parents=True, exist_ok=True)
    model = TournamentEncoder(config).to(device)
    model.load_state_dict(initial_state)
    if state_digest(model.state_dict()) != init_sha:
        raise RuntimeError('Objective arms did not start from identical weights.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    seed_all(seed + 200_000)
    # Separate generators keep minibatches/augmentations paired despite shuffled
    # ordering, decoder fitting, checkpoint frequency or evaluation RNG usage.
    sample_gen = torch.Generator().manual_seed(seed + 300_000)
    transform_gen = torch.Generator().manual_seed(seed + 400_000)
    ordering_gen = torch.Generator().manual_seed(seed + 500_000)
    candidates = torch.as_tensor(data['fit_centers'])
    transform_config = build_transform(args)
    latest = folder / 'latest_training.pt'
    start_step, previous_seconds = 0, 0.0
    if latest.is_file():
        checkpoint = load_own_checkpoint(latest)
        if checkpoint['init_sha256'] != init_sha:
            raise RuntimeError('Resume checkpoint has a different initialization.')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        sample_gen.set_state(checkpoint['sample_rng'])
        transform_gen.set_state(checkpoint['transform_rng'])
        ordering_gen.set_state(checkpoint['ordering_rng'])
        restore_rng(checkpoint['rng'])
        start_step, previous_seconds = checkpoint['step'], checkpoint['encoder_seconds']
        print(f'Resuming {config.architecture}/{seed}/{arm} at step {start_step}', flush=True)
    log_path = folder / 'training.csv'
    # A crash may leave log lines after the last saved training state.
    if log_path.is_file():
        with log_path.open(newline='') as handle:
            old = list(csv.DictReader(handle))
        old = [r for r in old if int(r['step']) <= start_step]
        if old:
            with log_path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, list(old[0]))
                writer.writeheader()
                writer.writerows(old)
        else:
            log_path.unlink()
    elapsed = previous_seconds
    rows = []
    # If the previous process stopped during evaluation, finish that checkpoint.
    if start_step in args.eval_steps and start_step > 0:
        rows.extend(evaluate_checkpoint(model, data, args, device, root, config.architecture,
                                         seed, arm, start_step, init_sha, elapsed))
    for step in range(start_step + 1, args.max_iter + 1):
        started = time.perf_counter()
        model.train()
        indices = torch.randint(len(candidates), (args.batch_size,), generator=sample_gen)
        windows = data['train'].get(candidates[indices])
        views, transform_audit = make_tournament(windows, transform_config, transform_gen)
        v, b, n, t = views.shape
        result = model(views.reshape(v * b, n, t).to(device))
        h, z = (result[key].reshape(v, b, -1) for key in ('h', 'z'))
        ordering = None
        if arm == 'rank_shuffled':
            ordering = torch.rand(b, v - 1, generator=ordering_gen).argsort(1).to(device)
        loss, log = tournament_loss(h, z, args.levels, objective=arm, margin=args.margin,
                                    variance_weight=args.variance_weight,
                                    covariance_weight=args.covariance_weight,
                                    consistency_weight=args.consistency_weight, ordering=ordering)
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite encoder loss at step {step}.')
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        elapsed += time.perf_counter() - started
        if step == 1 or step % args.log_every == 0 or step in args.eval_steps:
            distance = (z[1:].detach() - z[0].detach()).square().sum(-1).T
            ranking = ranking_diagnostics(distance)
            log.update(step=step, grad_norm=float(grad_norm),
                       pair_accuracy=ranking['pair_accuracy'], tie_fraction=ranking['tie_fraction'],
                       full_order_accuracy=ranking['full_order_accuracy'],
                       mean_std_clean_h=float(h[0].detach().std(0, unbiased=False).mean()),
                       max_count_error=transform_audit['max_count_error'], encoder_seconds=elapsed)
            new_file = not log_path.exists()
            with log_path.open('a', newline='') as handle:
                writer = csv.DictWriter(handle, list(log))
                if new_file:
                    writer.writeheader()
                writer.writerow(log)
            print(f'{config.architecture} seed={seed} {arm} step={step}/{args.max_iter} '
                  f'loss={log["loss"]:.4f} pair={log["pair_accuracy"]:.3f}', flush=True)
        if step % args.save_every == 0 or step in args.eval_steps or step == args.max_iter:
            save_torch(latest, dict(model=state_on_cpu(model), optimizer=optimizer.state_dict(),
                                    step=step, config=asdict(config), init_sha256=init_sha,
                                    sample_rng=sample_gen.get_state(), transform_rng=transform_gen.get_state(),
                                    ordering_rng=ordering_gen.get_state(), rng=rng_state(), encoder_seconds=elapsed))
        # Do not retain the preceding minibatch computation graph during decoders.
        del result, h, z, loss, views, windows
        if step in args.eval_steps:
            rows.extend(evaluate_checkpoint(model, data, args, device, root, config.architecture,
                                             seed, arm, step, init_sha, elapsed))
            summarize(root, args)
    del model, optimizer
    cleanup()
    return rows


def normalize_windows(windows, data):
    return ((windows - torch.from_numpy(data['mean']).view(1, -1, 1)) /
            torch.from_numpy(data['std']).view(1, -1, 1))


def baseline_features(kind, data, args):
    if kind == 'raw':
        return ((data['xt'] - data['mean']) / data['std'],
                (data['xv'] - data['mean']) / data['std'])
    if kind in ('statistics', 'raw_window'):
        results = []
        for source in (data['train'], data['valid']):
            parts = []
            for start in range(0, len(source), args.encoder_eval_batch):
                centers = np.arange(start, min(start + args.encoder_eval_batch, len(source)))
                windows = normalize_windows(source.get(centers), data)
                features = statistical_features(windows) if kind == 'statistics' else windows.flatten(1)
                parts.append(features.numpy())
            results.append(np.concatenate(parts))
        return tuple(results)
    if kind != 'pca_window':
        raise ValueError(f'Unknown baseline {kind}')
    centers = data['fit_centers']
    components = min(args.latent_dim, len(centers) - 1, data['xt'].shape[1] * args.window)
    batch_size = max(args.encoder_eval_batch, components * 2)
    pca = IncrementalPCA(n_components=components, batch_size=batch_size)
    start = 0
    while start < len(centers):
        stop = min(start + batch_size, len(centers))
        if len(centers) - stop < components:
            stop = len(centers)
        features = normalize_windows(data['train'].get(centers[start:stop]), data).flatten(1).numpy()
        pca.partial_fit(features)
        start = stop
    results = []
    for source in (data['train'], data['valid']):
        parts = []
        for start in range(0, len(source), args.encoder_eval_batch):
            centers_batch = np.arange(start, min(start + args.encoder_eval_batch, len(source)))
            features = normalize_windows(source.get(centers_batch), data).flatten(1).numpy()
            projection = pca.transform(features).astype(np.float32)
            # A fixed latent dimension gives the same downstream parameter count.
            parts.append(np.pad(projection, ((0, 0), (0, args.latent_dim - components))))
        results.append(np.concatenate(parts))
    return tuple(results)


def run_baselines(data, args, device, root):
    for seed in args.seeds:
        folder = root / 'baseline_mean' / f'seed_{seed}' / 'train_mean' / 'step_000000' / 'raw' / 'decoder_0'
        if not (folder / 'metrics.json').exists():
            folder.mkdir(parents=True, exist_ok=True)
            pred_train = np.broadcast_to(data['yt'].mean(0), data['yt'].shape).copy()
            pred_valid = np.broadcast_to(data['yt'].mean(0), data['yv'].shape).copy()
            row = dict(session=args.session, seed=seed, architecture='baseline_mean', arm='train_mean',
                        iteration=0, feature_space='raw', decoder_repeat=0, decoder_seed=None,
                        decoder_epochs=0, label_use='train-label mean, no encoder/decoder optimization',
                        train=regression_metrics(data['yt'], pred_train, data['yt'].var(0)),
                        valid=regression_metrics(data['yv'], pred_valid, data['yt'].var(0)),
                        synthetic=args.smoke_test, delta_vs_init=None,
                        prediction_path=str((folder / 'validation_predictions.npz').resolve()))
            np.savez_compressed(folder / 'validation_predictions.npz', y_true=data['yv'], y_pred=pred_valid)
            save_json(folder / 'metrics.json', row)
    for kind in args.baselines:
        complete = [root / ('baseline_' + kind) / f'seed_{seed}' / kind / 'step_000000' / 'raw' /
                    f'decoder_{repeat}' / 'metrics.json'
                    for seed in args.seeds for repeat in range(args.decoder_repeats)]
        if all(path.is_file() for path in complete):
            continue
        print(f'Building baseline {kind}', flush=True)
        with isolated_rng():
            seed_all(123)
            z_train, z_valid = baseline_features(kind, data, args)
            for seed in args.seeds:
                folder = root / ('baseline_' + kind) / f'seed_{seed}' / kind / 'step_000000' / 'raw'
                metadata = dict(session=args.session, seed=seed, architecture='baseline_' + kind,
                                arm=kind, iteration=0, parameter_count=0,
                                label_use='encoder:none; decoder:all columns', synthetic=args.smoke_test)
                evaluate_features(z_train, z_valid, data, args, device, folder, metadata, space='raw')
                summarize(root, args)
        del z_train, z_valid
        cleanup()


def run_label_shuffle_control(model, data, args, device, root, architecture, seed):
    """Same random-init representation, fully shuffled TRAIN labels; valid untouched."""
    folder = root / architecture / f'seed_{seed}' / 'decoder_label_shuffle' / 'step_000000' / 'h'
    if all((folder / f'decoder_{r}' / 'metrics.json').exists() for r in range(args.decoder_repeats)):
        return
    with isolated_rng():
        train = extract_embeddings(model, data['train'], ['h'], device, args.encoder_eval_batch)['h']
        valid = extract_embeddings(model, data['valid'], ['h'], device, args.encoder_eval_batch)['h']
        shuffled = dict(data)
        shuffled['yt'] = data['yt'][np.random.default_rng(seed + 531).permutation(len(data['yt']))]
        metadata = dict(session=args.session, seed=seed, architecture=architecture,
                        arm='decoder_label_shuffle', iteration=0, parameter_count=parameter_count(model),
                        label_use='encoder:none; decoder:randomly permuted train labels',
                        synthetic=args.smoke_test,
                        note='Train R2 is against shuffled train targets; valid R2 uses true labels.')
        evaluate_features(train, valid, shuffled, args, device, folder, metadata, space='h')


def mean_seed_summary(rows, value_fn):
    """Decoder repeats are averaged WITHIN seed, not counted as independent seeds."""
    by_seed = {}
    for row in rows:
        value = value_fn(row)
        if value is not None and np.isfinite(value):
            by_seed.setdefault(str(row['seed']), []).append(value)
    seed_means = {seed: float(np.mean(values)) for seed, values in by_seed.items()}
    x = np.array(list(seed_means.values()))
    if not len(x):
        return None
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else None
    ci = None
    if len(x) > 1:
        from scipy.stats import t
        half = float(t.ppf(.975, len(x) - 1) * sd / np.sqrt(len(x)))
        ci = [mean - half, mean + half]
    return dict(mean=mean, seed_std=sd, ci95_t_across_seeds=ci,
                n_seeds=len(x), per_seed=seed_means,
                interpretation='seed variability within one session; small-n CI, no multiple-comparison correction')


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    temporary.replace(path)


def summarize(root, args, plots=False):
    rows = []
    for path in sorted(root.glob('*/seed_*/*/step_*/*/decoder_*/metrics.json')):
        row = json.loads(path.read_text())
        rows.append(row)
    if not rows:
        return
    flat, groups = [], {}
    for row in rows:
        entry = {key: row.get(key) for key in (
            'session', 'seed', 'architecture', 'arm', 'iteration', 'feature_space', 'feature_dim',
            'parameter_count', 'decoder_seed', 'decoder_repeat', 'decoder_epochs', 'label_use',
            'delta_vs_init', 'synthetic', 'encoder_seconds')}
        entry.update(train_mean_r2=row['train']['mean_r2'], valid_mean_r2=row['valid']['mean_r2'],
                     valid_mse=row['valid']['mse'], valid_mae=row['valid']['mae'],
                     train_valid_gap=row['train']['mean_r2'] - row['valid']['mean_r2'])
        for i, value in enumerate(row['valid']['r2_per_output']):
            entry[f'valid_r2_output_{i}'] = value
        for key in ('mean_std', 'effective_rank', 'participation_ratio', 'inactive_fraction', 'mean_pair_cosine'):
            entry[key] = row.get('geometry', {}).get(key)
        if row.get('paired_block_bootstrap'):
            ci = row['paired_block_bootstrap'].get('ci95')
            entry['paired_delta_ci_low'], entry['paired_delta_ci_high'] = ci or (None, None)
        if row.get('ridge_probe'):
            entry['ridge_valid_mean_r2'] = row['ridge_probe']['valid']['mean_r2']
        if row.get('valid_unpadded'):
            entry['valid_unpadded_mean_r2'] = row['valid_unpadded']['mean_r2']
        if row.get('pretext_path') and Path(row['pretext_path']).exists():
            pretext = json.loads(Path(row['pretext_path']).read_text())
            for name in ('heldout_seen_strengths', 'heldout_unseen_strengths'):
                rank = pretext[name]['z_ranking']
                entry[name + '_pair_accuracy'] = rank['pair_accuracy']
                entry[name + '_tie_fraction'] = rank['tie_fraction']
                entry[name + '_full_order_accuracy'] = rank['full_order_accuracy']
        flat.append(entry)
        key = (row['architecture'], row['arm'], row['iteration'], row['feature_space'])
        groups.setdefault(key, []).append(row)
    write_csv(root / 'results.csv', flat)
    group_summary = []
    for (architecture, arm, step, space), items in sorted(groups.items()):
        group_summary.append(dict(architecture=architecture, arm=arm, iteration=step, feature_space=space,
                                  valid_r2=mean_seed_summary(items, lambda r: r['valid']['mean_r2']),
                                  delta_vs_init=mean_seed_summary(items, lambda r: r.get('delta_vs_init'))))
    save_json(root / 'summary.json', dict(groups=group_summary,
        checkpoint_selection='None: all prespecified steps are reported.',
        primary='clean validation mean R2 from the supplied full-batch MLP decoder',
        note='Paired delta, not pretext accuracy, answers whether training improves over random init.'))
    index = {(r['architecture'], r['seed'], r['iteration'], r['feature_space'],
              r['decoder_repeat'], r['arm']): r for r in rows}
    comparisons = []
    for row in rows:
        if row['arm'] not in ('rank', 'rank_consistency', 'rank_only') or row['iteration'] == 0:
            continue
        for control in ('init', 'rank_shuffled', 'consistency', 'reg_only'):
            reference = index.get((row['architecture'], row['seed'], 0 if control == 'init' else row['iteration'],
                                   row['feature_space'], row['decoder_repeat'], control))
            if reference:
                comparisons.append(dict(architecture=row['architecture'], seed=row['seed'],
                                         iteration=row['iteration'], feature_space=row['feature_space'],
                                         decoder_repeat=row['decoder_repeat'], arm=row['arm'], control=control,
                                         delta_r2=row['valid']['mean_r2'] - reference['valid']['mean_r2']))
    write_csv(root / 'controls.csv', comparisons)
    comp_groups = {}
    for row in comparisons:
        key = (row['architecture'], row['arm'], row['control'], row['iteration'], row['feature_space'])
        comp_groups.setdefault(key, []).append(row)
    save_json(root / 'controls_summary.json', [dict(
        architecture=k[0], arm=k[1], control=k[2], iteration=k[3], feature_space=k[4],
        delta_r2=mean_seed_summary(v, lambda r: r['delta_r2'])) for k, v in sorted(comp_groups.items())])
    # Match each architecture/seed to independently fit classical baselines.
    baseline_comparisons = []
    for row in rows:
        if row['arm'] != 'rank' or row['iteration'] != args.max_iter:
            continue
        for baseline in args.baselines:
            candidates = [r for r in rows if r['architecture'] == 'baseline_' + baseline and
                          r['seed'] == row['seed'] and r['decoder_repeat'] == row['decoder_repeat']]
            if candidates:
                baseline_comparisons.append(dict(architecture=row['architecture'], seed=row['seed'],
                    feature_space=row['feature_space'], decoder_repeat=row['decoder_repeat'],
                    baseline=baseline, delta_r2=row['valid']['mean_r2'] - candidates[0]['valid']['mean_r2']))
    save_json(root / 'comparisons.json', baseline_comparisons)
    if plots:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for architecture in args.architectures:
                for arm in args.arms:
                    selected = [g for g in group_summary if g['architecture'] == architecture and
                                g['arm'] == arm and g['feature_space'] == args.decode_spaces[0]]
                    initial = [g for g in group_summary if g['architecture'] == architecture and
                               g['arm'] == 'init' and g['feature_space'] == args.decode_spaces[0]]
                    selected = sorted(initial + selected, key=lambda g: g['iteration'])
                    if not selected:
                        continue
                    x = [g['iteration'] for g in selected]
                    y = [g['valid_r2']['mean'] for g in selected]
                    error = [g['valid_r2']['seed_std'] or 0 for g in selected]
                    axes[0].errorbar(x, y, yerr=error, marker='o', capsize=2, label=f'{architecture}/{arm}')
                    delta = [g['delta_vs_init']['mean'] if g['delta_vs_init'] else 0.0 for g in selected]
                    axes[1].plot(x, delta, marker='o', label=f'{architecture}/{arm}')
            axes[0].set(title='Clean validation R2 (error bars: seed SD)', ylabel='Mean R2')
            axes[1].set(title='Paired gain over same encoder at step 0', ylabel='Delta R2')
            axes[1].axhline(0, color='black', linewidth=.8)
            for ax in axes:
                ax.set_xlabel('Encoder optimization steps')
                ax.grid(alpha=.2)
            axes[0].legend(fontsize=7, loc='best')
            fig.tight_layout()
            fig.savefig(root / 'learning_curves.png', dpi=160)
            plt.close(fig)
        except ImportError:
            warnings.warn('matplotlib not installed; numeric results are still saved.')
    return group_summary


def run_cebra(data, args, device, root):
    if not args.include_cebra:
        return
    fork = args.cebra_dir
    if fork is None:
        try:
            from utils.constants import CEBRA_DIR
            fork = Path(CEBRA_DIR)
        except ImportError as exc:
            raise RuntimeError('Set --cebra-dir to the supplied ACORN/CEBRA checkout, '
                               'or run beside utils.constants.') from exc
    fork = Path(fork).expanduser().resolve()
    if not (fork / 'cebra' / '__init__.py').is_file():
        raise FileNotFoundError(f'Invalid CEBRA checkout: {fork}')
    for key in list(sys.modules):
        if key == 'cebra' or key.startswith('cebra.'):
            del sys.modules[key]
    sys.path.insert(0, str(fork))
    cebra = importlib.import_module('cebra')
    if fork not in Path(cebra.__file__).resolve().parents:
        raise RuntimeError('Wrong CEBRA module imported.')
    parameters = inspect.signature(cebra.CEBRA.__init__).parameters
    for mode in args.include_cebra:
        for seed in args.seeds:
            architecture = 'cebra_' + mode
            folder = root / architecture / f'seed_{seed}' / mode / f'step_{args.max_iter:06d}' / 'embedding'
            if all((folder / f'decoder_{r}' / 'metrics.json').exists() for r in range(args.decoder_repeats)):
                continue
            with isolated_rng():
                seed_all(seed)
                config = dict(batch_size=2048, temperature=.4,
                              model_architecture='offset36-model-more-dropout', time_offsets=1,
                              max_iterations=args.max_iter, output_dimension=args.latent_dim,
                              num_hidden_units=64, learning_rate=3e-4, temperature_mode='constant',
                              device=str(device), verbose=True)
                if 'training_mode' in parameters:
                    config['training_mode'] = 'standard'
                model = cebra.CEBRA(**config)
                if mode == 'supervised':
                    model.fit(data['xt'], data['yt'])  # Exact reference's all-column supervision.
                    label_use = 'encoder:all train labels; decoder:all train labels'
                else:
                    model.fit(data['xt'][:data['fit_end']])
                    label_use = 'encoder:none; decoder:all train labels'
                z_train = np.asarray(model.transform(data['xt']), np.float32)
                z_valid = np.asarray(model.transform(data['xv']), np.float32)
                if z_train.shape != (len(data['xt']), args.latent_dim) or z_valid.shape != (len(data['xv']), args.latent_dim):
                    raise ValueError('Unexpected CEBRA transform alignment/dimension.')
                metadata = dict(session=args.session, seed=seed, architecture=architecture,
                                arm=mode, iteration=args.max_iter, label_use=label_use,
                                encoder_config=config, synthetic=args.smoke_test,
                                comparison_note='Different architecture; supervised mode has extra label access. '
                                                'Fork window handling follows CEBRA, not WindowSource trial clamps.')
                evaluate_features(z_train, z_valid, data, args, device, folder, metadata, space='embedding')
                if args.save_checkpoints:
                    try:
                        model.save(str(folder / 'cebra.pt'), backend='sklearn')
                    except TypeError:
                        model.save(str(folder / 'cebra.pt'))
                summarize(root, args)
                del model
            cleanup()


def self_test():
    """Fast mathematical/wiring checks; no claimed decoding performance."""
    torch.set_num_threads(1)
    seed_all(7)
    x = torch.poisson(torch.full((6, 7, 36), .35))
    original = x.clone()
    for mode in TRANSFORMS:
        cfg = TransformConfig(mode=mode)
        a, audit = make_tournament(x, cfg, torch.Generator().manual_seed(51))
        b, _ = make_tournament(x, cfg, torch.Generator().manual_seed(51))
        assert torch.equal(a, b) and torch.equal(x, original)
        assert a.shape == (4, 6, 7, 36)
        assert audit['max_count_error'] < 1e-5
        if mode != 'neuron_shift':
            assert torch.equal(a.reshape(4, 6, 7, 3, 12).sum(-1),
                               x.reshape(6, 7, 3, 12).sum(-1).expand(4, -1, -1, -1))
        if mode == 'bin_permute':
            assert torch.equal(a.sort(-1).values, x.sort(-1).values.expand_as(a))
    try:
        make_tournament(x + .2, TransformConfig(mode='count_jitter'))
        raise AssertionError('Fractional counts were accepted.')
    except ValueError:
        pass
    for architecture in ARCHITECTURES:
        model = TournamentEncoder(EncoderConfig(7, width=16, latent_dim=8, projection_dim=8,
                                                  depth=1, architecture=architecture))
        result = model(a.reshape(-1, 7, 36))
        assert result['h'].shape == (24, 8)
        assert torch.allclose(result['z'].norm(dim=-1), torch.ones(24), atol=1e-5)
        for objective in OBJECTIVES:
            model.zero_grad(set_to_none=True)
            result = model(a.reshape(-1, 7, 36))
            h, z = [result[k].reshape(4, 6, 8) for k in ('h', 'z')]
            ordering = torch.rand(6, 3).argsort(1) if objective == 'rank_shuffled' else None
            loss, _ = tournament_loss(h, z, (1., 2., 4.), objective=objective, ordering=ordering)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        print(f'PASS architecture/gradient: {architecture}', flush=True)
    collapsed = ranking_diagnostics(torch.ones(6, 3))
    assert collapsed['pair_accuracy'] == 0 and collapsed['tie_fraction'] == 1
    perfect = ranking_diagnostics(torch.tensor([[0.1, 0.2, 0.4]]).repeat(6, 1))
    assert perfect['pair_accuracy'] == perfect['full_order_accuracy'] == 1
    assert representation_diagnostics(torch.ones(6, 8))['effective_rank'] == 0
    raw = np.arange(90, dtype=np.float32).reshape(30, 3)
    source = WindowSource(raw, 3, 3)
    expected = np.stack([np.pad(raw, ((3, 2), (0, 0)), mode='edge')[i:i + 6].T for i in range(30)])
    assert np.array_equal(source.get(np.arange(30)).numpy(), expected)
    segmented = WindowSource(raw, 3, 3, np.repeat([0, 1], 15))
    assert torch.equal(segmented.get([14])[0, :, -1], torch.from_numpy(raw[14]))
    assert torch.equal(segmented.get([15])[0, :, 0], torch.from_numpy(raw[15]))
    seed_all(91)
    state = torch.get_rng_state().clone()
    with isolated_rng():
        seed_all(8)
        torch.randn(100)
    assert torch.equal(state, torch.get_rng_state())
    print('PASS transforms, count preservation, ties/collapse, window alignment, trial clamps and RNG isolation.', flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session', default='C-CO12')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--out-root', type=Path, default=ROOT / 'NTT_CCO12_RESULTS')
    parser.add_argument('--resume', type=Path, help='Existing output folder; use identical experiment arguments.')
    parser.add_argument('--architectures', nargs='+', default=['tcn'], help='Names in ntt_architectures.ARCHITECTURES, or all.')
    parser.add_argument('--arms', nargs='+', choices=OBJECTIVES,
                        default=['rank', 'rank_shuffled', 'consistency', 'reg_only'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--max-iter', type=int, default=3000)
    parser.add_argument('--eval-steps', type=int, nargs='+', default=[0, 300, 1000, 3000])
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Anchor windows; actual encoder batch is (1+levels) times this.')
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--window', type=int, default=36)
    parser.add_argument('--causal-window', action='store_true')
    parser.add_argument('--bin-ms', type=float, default=None, help='Reporting metadata; do not guess if unknown.')
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--projection-dim', type=int, default=64)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--encoder-dropout', type=float, default=0.0,
                        help='Zero default avoids dropout noise dominating small perturbation rankings.')
    parser.add_argument('--pooling', choices=['center', 'mean', 'center_mean', 'attention'], default='center_mean')
    parser.add_argument('--standardize', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--transform', choices=TRANSFORMS, default='bin_permute')
    parser.add_argument('--levels', type=float, nargs='+', default=[1.0, 2.0, 4.0])
    parser.add_argument('--eval-levels', type=float, nargs='+', default=[1.5, 3.0, 5.0])
    parser.add_argument('--block-size', type=int, default=12)
    parser.add_argument('--max-events', type=int, default=2_000_000)
    parser.add_argument('--margin', type=float, default=.15)
    parser.add_argument('--variance-weight', type=float, default=1.0)
    parser.add_argument('--covariance-weight', type=float, default=.01)
    parser.add_argument('--consistency-weight', type=float, default=1.0)
    parser.add_argument('--pretext-holdout', type=float, default=.15)
    parser.add_argument('--train-segments-key', default='train_trial_id')
    parser.add_argument('--valid-segments-key', default='valid_trial_id')
    parser.add_argument('--decode-spaces', nargs='+', choices=['h', 'z'], default=['h'])
    parser.add_argument('--decoder-epochs', type=int, default=2500)
    parser.add_argument('--decoder-repeats', type=int, default=1)
    parser.add_argument('--decoder-eval-batch', type=int, default=8192)
    parser.add_argument('--encoder-eval-batch', type=int, default=256)
    parser.add_argument('--baselines', nargs='*', choices=['raw', 'pca_window', 'statistics', 'raw_window'],
                        default=['raw', 'pca_window', 'statistics'])
    parser.add_argument('--ridge-probe', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--label-shuffle-control', action='store_true')
    parser.add_argument('--diagnostic-samples', type=int, default=512)
    parser.add_argument('--robustness-samples', type=int, default=256,
                        help='Sensitivity at step0/final; 0 disables. Original labels may change meaning under corruption.')
    parser.add_argument('--bootstrap-repeats', type=int, default=300)
    parser.add_argument('--bootstrap-block', type=int, default=256)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=250)
    parser.add_argument('--save-checkpoints', action=argparse.BooleanOptionalAction, default=True,
                        help='Per-step encoder/decoder files; latest training state is always saved for resume.')
    parser.add_argument('--save-embeddings', action='store_true')
    parser.add_argument('--device', default='cuda_if_available')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--strict-determinism', action='store_true', help='May reject nondeterministic CUDA kernels.')
    parser.add_argument('--verbose-decoder', action='store_true')
    parser.add_argument('--include-cebra', nargs='*', choices=['supervised', 'time'], default=[])
    parser.add_argument('--cebra-dir', type=Path)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    if args.architectures == ['all']:
        args.architectures = list(ARCHITECTURES)
    if any(a not in ARCHITECTURES for a in args.architectures):
        parser.error(f'Architecture choices: {ARCHITECTURES}')
    if args.smoke_test:
        args.session = 'SYNTHETIC'
        args.data_dir = args.out_root.expanduser() / 'synthetic_data'
        args.seeds, args.max_iter, args.eval_steps = [42], 3, [0, 1, 3]
        args.width, args.depth, args.latent_dim, args.projection_dim = 16, 1, 8, 8
        args.batch_size, args.decoder_epochs = 8, 12
        args.diagnostic_samples, args.robustness_samples = 32, 16
        args.bootstrap_repeats, args.bootstrap_block = 20, 64
        args.log_every, args.save_every = 1, 1
        args.include_cebra = []
    if args.max_iter < 0 or any(step < 0 for step in args.eval_steps):
        parser.error('Iterations must be nonnegative; 0 is supported.')
    args.eval_steps = sorted(set([0, args.max_iter] + [v for v in args.eval_steps if v <= args.max_iter]))
    positive = [args.batch_size, args.width, args.latent_dim, args.projection_dim,
                args.depth, args.heads, args.decoder_epochs, args.decoder_repeats,
                args.decoder_eval_batch, args.encoder_eval_batch, args.log_every,
                args.save_every, args.threads, args.learning_rate, args.grad_clip,
                args.margin, args.max_events]
    if not np.isfinite(positive).all() or min(positive) < 1e-12:
        parser.error('Dimensions, epochs, batch sizes, intervals, lr, margin and grad clip must be positive.')
    if args.batch_size < 2 or args.window < 4 or args.diagnostic_samples < 8:
        parser.error('Need batch-size>=2, window>=4 and diagnostic-samples>=8.')
    if not .05 <= args.pretext_holdout <= .4:
        parser.error('pretext-holdout must be between .05 and .4.')
    if not 0 <= args.encoder_dropout < 1:
        parser.error('encoder-dropout must be in [0,1).')
    if args.bootstrap_block < args.window or args.bootstrap_repeats < 0 or args.robustness_samples < 0:
        parser.error('bootstrap-block must cover at least a window; repeat/sample counts must be nonnegative.')
    if args.bin_ms is not None and (not np.isfinite(args.bin_ms) or args.bin_ms <= 0):
        parser.error('bin-ms must be finite and positive.')
    if any(v < 0 or not np.isfinite(v) for v in [args.variance_weight, args.covariance_weight,
                                               args.consistency_weight, args.weight_decay]):
        parser.error('Loss weights and weight decay must be finite and nonnegative.')
    for name in ('architectures', 'arms', 'seeds', 'decode_spaces', 'baselines'):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f'Duplicated {name}.')
    if min(args.seeds) < 0 or max(args.seeds) >= 2**31 - 600_000:
        parser.error('Seeds must fit a nonnegative 31-bit integer minus RNG offsets.')
    try:
        build_transform(args)
        build_transform(args, args.eval_levels)
    except ValueError as exc:
        parser.error(str(exc))
    if args.transform == 'neuron_shift' and max(args.levels + args.eval_levels) >= args.window / 2:
        parser.error('Shift levels must be smaller than window/2.')
    if args.include_cebra and args.max_iter < 1:
        parser.error('CEBRA reference does not support the zero-iteration protocol.')
    return args


def make_synthetic_data(args):
    args.data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(614)
    total, neurons = 1500, 12
    t = np.arange(total)
    latent = np.stack((np.sin(t / 13), np.cos(t / 17), np.sin(t / 31), np.cos(t / 37)), -1)
    weights = rng.normal(0, .35, (4, neurons))
    rates = np.exp(latent @ weights - .7)
    counts = rng.poisson(rates).astype(np.float32)
    labels = (latent + rng.normal(0, .03, latent.shape)).astype(np.float32)
    np.savez_compressed(args.data_dir / 'SYNTHETIC.npz',
                        train_data=counts[:1100], valid_data=counts[1100:],
                        train_label=labels[:1100], valid_label=labels[1100:],
                        train_trial_id=np.repeat(np.arange(11), 100),
                        valid_trial_id=np.repeat(np.arange(4), 100))


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return
    torch.set_num_threads(args.threads)
    if args.strict_determinism:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
    if args.device == 'cuda_if_available':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Use --device cpu for smoke tests.')
    if args.smoke_test:
        make_synthetic_data(args)
        print('SYNTHETIC WIRING TEST ONLY: no Perich performance claim.', flush=True)
    data = load_data(args)
    if args.transform == 'bin_permute':
        print('Transform: whole-bin local permutation, not sub-bin spike jitter.', flush=True)
    if not args.causal_window:
        print(f'OFFLINE windows: {data["audit"]["window_offsets"]}; same 36-bin convention as reference by default.', flush=True)
    print('Shapes:', [data[k].shape for k in ('xt', 'xv', 'yt', 'yv')], flush=True)
    print('Decoder: full batch, hidden=64, LayerNorm/ReLU/dropout=.4, Adam=.001, '
          f'{args.decoder_epochs} epochs, all label columns.', flush=True)
    config_args = clean_json(vars(args))
    signature_args = {k: v for k, v in config_args.items() if k not in (
        'resume', 'out_root', 'verbose_decoder', 'log_every', 'save_every', 'save_embeddings',
        'save_checkpoints', 'self_test')}
    module_path = ROOT / 'ntt_architectures.py'
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (Path(__file__).resolve(), module_path)}
    fingerprint = hashlib.sha256(json.dumps(dict(args=signature_args, source=source_hashes,
        dataset=data['audit']['sha256_loaded_arrays']), sort_keys=True).encode()).hexdigest()
    if args.resume:
        root = args.resume.expanduser().resolve()
        if not (root / 'run_config.json').is_file():
            raise FileNotFoundError('Resume directory lacks run_config.json.')
        previous = json.loads((root / 'run_config.json').read_text())
        if previous['fingerprint'] != fingerprint:
            raise ValueError('Resume arguments, dataset, or code differ. Reuse the original configuration '
                             'or start a new run; results from different protocols cannot be mixed.')
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        root = args.out_root.expanduser().resolve() / f'{args.session}_{stamp}'
        root.mkdir(parents=True, exist_ok=False)
        save_json(root / 'run_config.json', dict(
            fingerprint=fingerprint, args=config_args, source_sha256=source_hashes,
            dataset_sha256=data['audit']['sha256_loaded_arrays'],
            decoder=dict(hidden=64, dropout=.4, lr=.001, epochs=args.decoder_epochs,
                         batch_mode='full', label_scaling='none', labels='all columns'),
            encoder_label_use='none', zero_iteration='encoder untrained; decoder fully trained',
            primary='paired clean-valid R2 gain over same initialization',
            selection='No best-validation checkpoint or early stopping',
            environment=dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                             numpy=np.__version__, cuda=torch.version.cuda, device=str(device)),
        ))
        # Reproducibility snapshots travel with remote experiment results.
        for source in (Path(__file__).resolve(), module_path):
            (root / source.name).write_bytes(source.read_bytes())
        save_json(root / 'dataset_audit.json', data['audit'])
        np.savez_compressed(root / 'normalization.npz', mean=data['mean'], std=data['std'])
    lock = root / 'RUNNING.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError('Run is locked. If its previous process has stopped, remove RUNNING.lock before resuming.') from exc
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    eval_count = len(args.architectures) * len(args.seeds) * len(args.decode_spaces) * args.decoder_repeats * (
        1 + len(args.arms) * len([v for v in args.eval_steps if v > 0]))
    print('OUTPUT:', root, flush=True)
    print(f'{eval_count} encoder-checkpoint decoder fits, plus baselines; steps={args.eval_steps}.', flush=True)
    if args.max_iter > 0 and device.type == 'cpu' and not args.smoke_test:
        warnings.warn('Full-batch decoder sweeps may be slow on CPU; no performance results have been assumed.')
    try:
        shortcut_file = root / 'shortcut_audit.json'
        if not shortcut_file.exists():
            save_json(shortcut_file, fit_shortcut_audit(data, args, args.seeds[0]))
        run_baselines(data, args, device, root)
        for architecture in args.architectures:
            for seed in args.seeds:
                seed_all(seed)
                config = EncoderConfig(
                    n_neurons=data['xt'].shape[1], window=args.window, center=data['train'].left,
                    architecture=architecture, width=args.width, latent_dim=args.latent_dim,
                    projection_dim=args.projection_dim, depth=args.depth, heads=args.heads,
                    dropout=args.encoder_dropout, pooling=args.pooling).validate()
                initial = TournamentEncoder(config, data['mean'], data['std']).to(device)
                state = state_on_cpu(initial)
                initial_sha = state_digest(state)
                evaluate_checkpoint(initial, data, args, device, root, architecture, seed, 'init', 0, initial_sha)
                if args.label_shuffle_control:
                    run_label_shuffle_control(initial, data, args, device, root, architecture, seed)
                del initial
                summarize(root, args)
                for arm in args.arms:
                    if args.max_iter > 0:
                        train_arm(config, state, data, args, device, root, seed, arm, initial_sha)
                cleanup()
        run_cebra(data, args, device, root)
        summary = summarize(root, args, plots=True)
        save_json(root / 'completed.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
                                                synthetic=args.smoke_test))
        print('\nFINAL CLEAN-INPUT RESULTS (prespecified checkpoints, no best-step selection)', flush=True)
        for group in summary:
            if group['iteration'] == args.max_iter and group['arm'] in args.arms:
                delta = group['delta_vs_init']
                print(f"{group['architecture']}/{group['arm']}/{group['feature_space']}: "
                      f"R2={group['valid_r2']['mean']:.6f}, "
                      f"delta0={None if delta is None else delta['mean']}", flush=True)
        print('Saved:', root, flush=True)
    except BaseException as exc:
        save_json(root / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
                                             time=datetime.now(timezone.utc).isoformat()))
        try:
            summarize(root, args)
        except Exception:
            pass
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()




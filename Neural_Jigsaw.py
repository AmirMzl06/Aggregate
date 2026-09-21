"""Jigsaw-CEBRA: temporal order as a NUISANCE-CARVING side task, not a representation.

Requires NumPy and PyTorch >= 2.0 (nn.Dropout1d needs torch >= 1.12; set
dropout=0.0 if you are stuck on something older). Inputs are (time, neurons)
float arrays.

    from jigsaw_cebra import JigsawCEBRA
    model = JigsawCEBRA(window_size=10, n_tiles=4, tile_gap=(1, 8),
                        behavior_dim=32, puzzle_dim=16, max_epochs=30)
    model.fit(X_train, X_valid=X_valid)
    Z = model.transform(X_valid, block="behavior")      # use THIS for decoding
    print(model.evaluate_puzzle(X_valid))
    print(model.evaluate_decoding(X_train, Y_train, X_valid, Y_valid))

WHY THIS DIFFERS FROM A PLAIN TEMPORAL JIGSAW
---------------------------------------------
A pure temporal-jigsaw objective tends to make downstream decoding WORSE than a
random encoder. Three mechanisms, and the structural answer to each:

1. Objective conflict. Time-contrastive learning wants the latent INVARIANT to
   small temporal shifts; a jigsaw head wants it EQUIVARIANT to exact temporal
   position. On a continuous neural trajectory these are the same information,
   so the two objectives fight over one latent. Answer: the latent is SPLIT into
   z_behavior and z_puzzle, produced by two projectors on one shared trunk. Only
   z_puzzle feeds the puzzle head; only z_behavior feeds the contrastive loss and
   is the default output of transform. A cross-block decorrelation penalty pushes
   order/phase nuisance OUT of z_behavior. The jigsaw stops being the
   representation objective and becomes a place to PUT nuisance.
2. Shortcuts. Tile order is recoverable from slow drift in overall level, so the
   head learns a nonstationarity detector that does not transfer. Answer: the
   puzzle branch sees per-tile normalized tiles (tile_norm), independent
   per-tile augmentation (neuron dropout / gain jitter / noise), random gaps,
   and an optional shortcut filter that down-weights quartets whose order the
   trivial "sort by mean activity" baseline already gets right. evaluate_puzzle
   always reports that baseline next to the model.
3. Train/test input mismatch. Heads that classify a permutation from a SHUFFLED
   input train the encoder almost exclusively on non-physical inputs, while
   transform only ever sees natural windows. Answer: tiles are always intact and
   chronological; the head is permutation-EQUIVARIANT, so shuffling is provably
   unnecessary (shuffle_tiles=True is kept only as a parity check, and the
   self-test proves the loss is invariant to it).

THE HEAD
--------
Each tile's z_puzzle plus a mean-pooled set context (DeepSets, permutation
invariant) becomes a token. Two low-capacity relational readouts share it:
  - "assign": token . position_embedding -> (K,K) logits, log-domain Sinkhorn
    normalization, NLL of the true slot->position assignment. Soft bijection.
  - "rank":   token -> one scalar time score, Bradley-Terry / RankNet pairwise
    logistic loss over all tile pairs. Gives an interpretable 1-D time axis.
Both are equivariant in the slot index and agnostic to n_tiles, so K=6 or 8 is a
strictly harder pretext at no code change. Neither can memorize a 24-way label.

THE TRUNK
---------
trunk_block="residual" is a plain valid-convolution residual stack.
trunk_block="separable" is a MobileNetV2-style inverted residual: 1x1 expand,
GELU, DEPTHWISE 3-tap temporal convolution, GELU, 1x1 linear projection. On
population recordings that factorization is the useful part of MobileNet: each
channel gets its own temporal filter and mixing across neurons happens only in
the pointwise layers (the same decomposition EEGNet uses). It is NOT a
robustness fix by itself -- a shortcut that is linearly available stays
available -- but it is cheaper per parameter and a fair ablation to run. Both
blocks consume exactly two bins, so the receptive field stays window_size.

DIAGNOSTIC CONTROLS BUILT IN (all reachable from the constructor)
  lambda_puzzle=0.0     -> pure time-contrastive CEBRA-style baseline
  lambda_infonce=0.0    -> pure jigsaw (reproduces the failure mode on purpose)
  max_epochs=0          -> frozen random-encoder control
  puzzle_grad_scale=0.0 -> puzzle head trained on a trunk it cannot influence;
                           a probe for "is order even decodable from behavior
                           features", with zero risk to R2
  puzzle_warmup_fraction / lambda_puzzle -> the InfoNCE head start and the
                           side-task weight; anneal these before anything else
Report puzzle accuracy AND the mean-sort baseline AND decoding R2 together.
InfoNCE accuracy is logged every epoch: if it is near 100% the positive pair is
solvable from window identity and the embedding will carry no behavior. Tiles
are separated by at least one discarded bin, so anchor and positive windows can
never overlap - the usual cause of that failure.

Time alignment. transform runs the trunk on natural windows with NO augmentation
and NO tile normalization. pad=False aligns the window starting at j with label
j + window_size//2; pad=True edge-pads per sequence and returns one row per
input bin. Contexts are noncausal and centered.

Split recordings BEFORE fit. A single 2D array is one continuous recording; pass
a list of trial arrays to avoid crossing trials. fit always resets weights. One
epoch visits every valid span start once in random order with fresh gaps.
Each fit/evaluate sequence needs at least training_span bins; transform needs
window_size (or 1 with pad=True).

Save/load stores weights and config, not optimizer state, so runs cannot resume.
Puzzle accuracy is not evidence of behavioral usefulness: the only claim this
module supports is the one you get from evaluate_decoding against the
lambda_puzzle=0 and max_epochs=0 arms on the SAME split.

Inspired by Noroozi & Favaro (2016), arXiv:1603.09246, and by the multiobjective
block structure of xCEBRA. Sinkhorn assignment follows Mena et al. (2018),
arXiv:1802.08665.

Self-test:  python jigsaw_cebra.py --self-test
Ablation demo on synthetic data with a known latent:  python jigsaw_cebra.py --demo
"""
from itertools import permutations
import copy
import math
import numbers
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

_EPS = 1e-8

TILE_NORMS = ("none", "mean", "zscore", "global_mean")
PUZZLE_HEADS = ("assign", "rank", "both")
TRUNK_BLOCKS = ("residual", "separable")


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _integer(name, value, minimum=1, maximum=None):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}; got {value!r}.")
    return int(value)


def _real(name, value, minimum, maximum=None, strict_min=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number; got {value!r}.")
    value = float(value)
    if (not math.isfinite(value) or value < minimum
            or (strict_min and value == minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"Invalid {name}: {value}.")
    return value


def _boolean(name, value):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool; got {value!r}.")
    return value


def _sequences(X, min_length, n_features=None):
    """Accept one (T,C) array, or a list of (T_i,C) arrays, without joining trials."""
    def array(value):
        if torch.is_tensor(value):
            value = value.detach().to(device="cpu", dtype=torch.float32).numpy()
        return np.asarray(value, dtype=np.float32)

    is_list = (isinstance(X, (list, tuple)) and len(X) > 0 and array(X[0]).ndim == 2)
    values = list(X) if is_list else [X]
    result = []
    for index, value in enumerate(values):
        x = array(value)
        if x.ndim != 2 or x.shape[0] < min_length or x.shape[1] < 1:
            raise ValueError(
                f"Sequence {index}: expected (T,C) with T >= {min_length}, C >= 1; got {x.shape}.")
        if not np.isfinite(x).all():
            raise ValueError(f"Sequence {index} contains NaN or infinite values.")
        if n_features is None:
            n_features = x.shape[1]
        if x.shape[1] != n_features:
            raise ValueError(f"Sequence {index}: expected {n_features} channels, got {x.shape[1]}.")
        # Own the storage: training must never modify the caller's data.
        result.append(np.array(x, dtype=np.float32, order="C", copy=True))
    return result, is_list


def _permutation_table(n_tiles):
    """All permutations for small K, else None (greedy assignment fallback)."""
    return np.asarray(list(permutations(range(n_tiles))), dtype=np.int64) if n_tiles <= 7 else None


# --------------------------------------------------------------------------- #
# modules
# --------------------------------------------------------------------------- #
class _GradScale(torch.autograd.Function):
    """Identity forward, gradient multiplied by `scale` on the way back."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class _Residual(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout1d(dropout), nn.Conv1d(width, width, 3), nn.GELU())

    def forward(self, x):
        return x[..., 1:-1] + self.net(x)


class _Separable(nn.Module):
    """MobileNetV2-style inverted residual with a VALID depthwise 3-tap conv.

    1x1 expand -> GELU -> depthwise temporal conv -> GELU -> 1x1 linear project.
    Temporal filtering is per channel, cross-neuron mixing happens only in the
    pointwise layers. Consumes exactly two bins, like _Residual, so the trunk's
    receptive-field arithmetic is unchanged.
    """

    def __init__(self, width, dropout, expansion=4):
        super().__init__()
        hidden = max(width, width * expansion)
        self.net = nn.Sequential(
            nn.Dropout1d(dropout),
            nn.Conv1d(width, hidden, 1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, groups=hidden), nn.GELU(),
            nn.Conv1d(hidden, width, 1),
        )

    def forward(self, x):
        return x[..., 1:-1] + self.net(x)


class _Trunk(nn.Module):
    """Valid-convolution stack with receptive field exactly window_size."""

    def __init__(self, channels, window_size, width, dropout, block="residual"):
        super().__init__()
        if block not in TRUNK_BLOCKS:
            raise ValueError(f"trunk_block must be one of {TRUNK_BLOCKS}.")
        # Sum(kernel - 1) = W - 1, hence W input bins -> exactly one output bin.
        first_kernel = 2 if window_size % 2 == 0 else 3
        blocks = (window_size - first_kernel - 2) // 2
        make = _Residual if block == "residual" else _Separable
        self.layers = nn.Sequential(
            nn.Conv1d(channels, width, first_kernel), nn.Dropout1d(dropout), nn.GELU(),
            *[make(width, dropout) for _ in range(blocks)],
            nn.Conv1d(width, width, 3), nn.GELU(),
        )

    def forward(self, x):
        h = self.layers(x)
        if h.shape[-1] != 1:
            raise ValueError("Trunk expects exactly window_size input bins.")
        return h.squeeze(-1)


class _Encoder(nn.Module):
    """Shared trunk, two projectors: z_behavior and z_puzzle, normalized per block."""

    def __init__(self, channels, window_size, width, behavior_dim, puzzle_dim,
                 dropout, normalize, trunk_block="residual"):
        super().__init__()
        self.trunk = _Trunk(channels, window_size, width, dropout, trunk_block)
        self.project_behavior = nn.Linear(width, behavior_dim)
        self.project_puzzle = nn.Linear(width, puzzle_dim)
        self.normalize = normalize

    def _maybe_normalize(self, z):
        return F.normalize(z, p=2, dim=-1, eps=_EPS) if self.normalize else z

    def features(self, x):
        return self.trunk(x)

    def behavior(self, features):
        return self._maybe_normalize(self.project_behavior(features))

    def puzzle(self, features):
        return self._maybe_normalize(self.project_puzzle(features))

    def forward(self, x):
        features = self.features(x)
        return self.behavior(features), self.puzzle(features)


class _PuzzleHead(nn.Module):
    """Permutation-equivariant relational head over K tile latents.

    Token i = MLP([z_i, mean_j z_j]). The context is mean-pooled, hence
    permutation invariant, so permuting the tile axis permutes exactly the rows
    of `logits` and the entries of `scores`, and nothing else.
    """

    def __init__(self, dimension, hidden, n_tiles):
        super().__init__()
        self.hidden = hidden
        self.token = nn.Sequential(nn.Linear(2 * dimension, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden))
        self.position = nn.Parameter(torch.randn(n_tiles, hidden) / math.sqrt(hidden))
        self.score = nn.Linear(hidden, 1)

    def forward(self, z):
        if z.ndim != 3:
            raise ValueError("Puzzle head expects (batch, n_tiles, dimension).")
        context = z.mean(dim=1, keepdim=True).expand_as(z)
        tokens = self.token(torch.cat((z, context), dim=2))
        logits = tokens @ self.position.t() / math.sqrt(self.hidden)
        return logits, self.score(tokens).squeeze(-1)


# --------------------------------------------------------------------------- #
# tile construction, augmentation, normalization
# --------------------------------------------------------------------------- #
def _gather_tiles(data, starts, gaps, window_size):
    """data (T,N) -> (B,K,N,W). gaps counts DISCARDED bins between tiles."""
    zero = torch.zeros((starts.shape[0], 1), dtype=torch.long, device=data.device)
    offsets = torch.cat((zero, torch.cumsum(gaps + window_size, dim=1)), dim=1)
    index = (starts[:, None] + offsets)[:, :, None] + torch.arange(window_size, device=data.device)
    return data[index].permute(0, 1, 3, 2).contiguous()


def _augment(tiles, neuron_dropout, gain_jitter, noise, generator):
    """Independent per (sample, tile): whole-neuron dropout, gain, additive noise."""
    shape = tiles.shape[:3]
    out = tiles
    if neuron_dropout > 0:
        keep = torch.rand(shape, device=tiles.device, generator=generator) >= neuron_dropout
        out = out * keep.to(tiles.dtype)[..., None]
    if gain_jitter > 0:
        gain = 1 + (2 * torch.rand(shape, device=tiles.device, generator=generator) - 1) * gain_jitter
        out = out * gain[..., None]
    if noise > 0:
        out = out + noise * torch.randn(tiles.shape, device=tiles.device, generator=generator)
    return out


def _tile_normalize(tiles, mode):
    """Remove the per-tile level so tile order is not recoverable from drift."""
    if mode == "none":
        return tiles
    if mode == "global_mean":
        return tiles - tiles.mean(dim=(2, 3), keepdim=True)
    centered = tiles - tiles.mean(dim=3, keepdim=True)
    if mode == "mean":
        return centered
    if mode == "zscore":
        return centered / (tiles.std(dim=3, keepdim=True, unbiased=False) + 1e-5)
    raise ValueError(f"Unknown tile_norm {mode!r}; expected one of {TILE_NORMS}.")


# --------------------------------------------------------------------------- #
# losses and metrics
# --------------------------------------------------------------------------- #
def _log_sinkhorn(logits, iterations):
    """Log-domain doubly-stochastic normalization; rows end as log-probabilities."""
    log_alpha = logits
    for _ in range(iterations):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)


def _weighted_mean(per_sample, weight):
    return (per_sample * weight).sum() / weight.sum().clamp_min(_EPS)


def _assignment_loss(logits, positions, iterations, weight):
    log_p = _log_sinkhorn(logits, iterations)
    nll = -log_p.gather(2, positions[:, :, None]).squeeze(-1)
    return _weighted_mean(nll.mean(dim=1), weight)


def _rank_loss(scores, positions, weight):
    """Bradley-Terry over all tile pairs: later tiles must score higher."""
    difference = scores[:, :, None] - scores[:, None, :]
    sign = torch.sign(positions[:, :, None] - positions[:, None, :]).to(difference.dtype)
    mask = (sign != 0).to(difference.dtype)
    per_sample = (F.softplus(-sign * difference) * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)
    return _weighted_mean(per_sample, weight)


def _ranks(values):
    """(B,K) values -> (B,K) integer ranks, i.e. predicted chronological position."""
    return values.argsort(dim=1).argsort(dim=1)


def _best_assignment(logits, table):
    """Highest-scoring bijective slot -> position map. Exact for n_tiles <= 7."""
    n_tiles = logits.shape[1]
    if table is not None:
        candidates = torch.as_tensor(table, dtype=torch.long, device=logits.device)
        rows = torch.arange(n_tiles, device=logits.device).expand(len(candidates), n_tiles)
        scores = logits[:, rows, candidates].sum(-1)
        return candidates[scores.argmax(dim=1)]
    # Greedy fallback for large K: repeatedly take the best remaining (slot, position).
    work = logits.clone()
    result = torch.zeros(logits.shape[:2], dtype=torch.long, device=logits.device)
    for _ in range(n_tiles):
        flat = work.reshape(len(work), -1).argmax(dim=1)
        slot, position = flat // n_tiles, flat % n_tiles
        result[torch.arange(len(work), device=logits.device), slot] = position
        work[torch.arange(len(work), device=logits.device), slot, :] = -math.inf
        work[torch.arange(len(work), device=logits.device), :, position] = -math.inf
    return result


def _order_metrics(predicted, positions):
    """Exact-permutation and pairwise-order accuracy for integer position maps."""
    exact = (predicted == positions).all(dim=1).to(torch.float32)
    truth = torch.sign(positions[:, :, None] - positions[:, None, :])
    guess = torch.sign(predicted[:, :, None] - predicted[:, None, :])
    mask = truth != 0
    pair = ((guess == truth) & mask).sum((1, 2)).to(torch.float32) / mask.sum((1, 2)).clamp_min(1)
    return exact, pair


def _infonce(z, temperature, generator):
    """Time-contrastive InfoNCE on z_behavior (B,K,D), in-batch negatives.

    Positive pair = two DIFFERENT tiles of the same span, so anchor and positive
    windows are separated by at least one discarded bin and can never overlap.
    Negatives = every tile of every OTHER span in the batch.
    """
    batch, n_tiles, dimension = z.shape
    if batch < 2:
        raise ValueError("InfoNCE needs at least two spans per batch.")
    device = z.device
    index = torch.arange(batch, device=device)
    anchor = torch.randint(n_tiles, (batch,), device=device, generator=generator)
    step = torch.randint(n_tiles - 1, (batch,), device=device, generator=generator)
    positive = (anchor + 1 + step) % n_tiles
    reference = z[index, anchor]
    pair = z[index, positive]
    pool = z.reshape(batch * n_tiles, dimension)
    logits = reference @ pool.t() / temperature
    same_span = (torch.arange(batch * n_tiles, device=device) // n_tiles)[None, :] == index[:, None]
    logits = logits.masked_fill(same_span, -float("inf"))
    positive_logit = (reference * pair).sum(-1, keepdim=True) / temperature
    full = torch.cat((positive_logit, logits), dim=1)
    loss = (-positive_logit.squeeze(1) + torch.logsumexp(full, dim=1)).mean()
    accuracy = (full.argmax(dim=1) == 0).to(torch.float32).mean()
    return loss, accuracy


def _decorrelation(a, b):
    """Mean squared cross-correlation between the two latent blocks."""
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    a = a / (a.std(0, unbiased=False, keepdim=True) + 1e-5)
    b = b / (b.std(0, unbiased=False, keepdim=True) + 1e-5)
    return ((a.t() @ b) / a.shape[0]).pow(2).mean()


def _participation_ratio(features):
    """(sum s)^2 / sum s^2 over covariance eigenvalues: effective dimensionality."""
    centered = features - features.mean(0, keepdims=True)
    eigenvalues = np.linalg.eigvalsh(np.cov(centered, rowvar=False) + 1e-12 * np.eye(centered.shape[1]))
    eigenvalues = np.clip(eigenvalues, 0, None)
    total = eigenvalues.sum()
    return float(total ** 2 / np.square(eigenvalues).sum()) if total > 0 else 0.0


def _ridge_r2(train_features, train_targets, test_features, test_targets, alphas):
    """Closed-form ridge with a CHRONOLOGICAL internal split for alpha selection."""
    mean = train_features.mean(0, keepdims=True)
    scale = train_features.std(0, keepdims=True) + 1e-8
    a_train = (train_features - mean) / scale
    a_test = (test_features - mean) / scale
    cut = max(1, int(0.8 * len(a_train)))
    inner_x, inner_y = a_train[:cut], train_targets[:cut]
    hold_x, hold_y = a_train[cut:], train_targets[cut:]
    if len(hold_x) < 2:
        inner_x, inner_y, hold_x, hold_y = a_train, train_targets, a_train, train_targets

    def solve(x, y, alpha):
        offset = y.mean(0, keepdims=True)
        gram = x.T @ x + alpha * np.eye(x.shape[1])
        return np.linalg.solve(gram, x.T @ (y - offset)), offset

    def r2(y_true, y_hat):
        residual = np.square(y_true - y_hat).sum(0)
        total = np.square(y_true - y_true.mean(0, keepdims=True)).sum(0)
        return 1.0 - residual / np.maximum(total, 1e-12)

    scored = [(float(np.mean(r2(hold_y, hold_x @ w + b))), alpha)
              for alpha in alphas for w, b in [solve(inner_x, inner_y, alpha)]]
    best_alpha = max(scored)[1]
    weights, offset = solve(a_train, train_targets, best_alpha)
    per_dimension = r2(test_targets, a_test @ weights + offset)
    return dict(r2=float(np.mean(per_dimension)), r2_per_dimension=per_dimension.tolist(),
                alpha=float(best_alpha), n_train=int(len(a_train)), n_test=int(len(a_test)),
                participation_ratio=_participation_ratio(train_features))


# --------------------------------------------------------------------------- #
# estimator
# --------------------------------------------------------------------------- #
class JigsawCEBRA:
    """Shared trunk, split latent, time-contrastive z_behavior + jigsaw z_puzzle.

    window_size is the receptive field used for decoding. n_tiles intact,
    chronological, nonoverlapping tiles of that length are cut from one span,
    separated by tile_gap=(minimum, maximum) DISCARDED bins resampled at every
    visit. Larger n_tiles is a strictly harder pretext at no extra code.

    transform(block="behavior") is what you decode. block="all" returns
    [z_behavior, z_puzzle] as two separately L2-normalized blocks, mirroring
    xCEBRA's multiobjective layout.
    """

    def __init__(self, window_size=10, n_tiles=4, tile_gap=(1, 8),
                 behavior_dim=32, puzzle_dim=16, num_hidden_units=64,
                 head_hidden_units=64, dropout=0.0, normalize=True,
                 trunk_block="residual",
                 temperature=1.0, lambda_infonce=1.0, lambda_puzzle=0.3,
                 puzzle_warmup_fraction=0.1, lambda_decorrelation=1.0,
                 puzzle_grad_scale=1.0, puzzle_head="both", rank_weight=0.5,
                 sinkhorn_iterations=5, tile_norm="mean", neuron_dropout=0.1,
                 gain_jitter=0.1, noise_std=0.0, shortcut_reject=0.0,
                 separate_puzzle_view=None, shuffle_tiles=False,
                 batch_size=512, max_epochs=30, learning_rate=3e-4,
                 weight_decay=0.0, device="cuda_if_available", random_state=42,
                 verbose=True, log_every=1):
        self.window_size = _integer("window_size", window_size, 4)
        self.n_tiles = _integer("n_tiles", n_tiles, 2, 12)
        gaps = tile_gap if isinstance(tile_gap, (tuple, list)) else (tile_gap, tile_gap)
        if len(gaps) != 2:
            raise ValueError("tile_gap must be an integer or a (minimum, maximum) pair.")
        self.tile_gap = tuple(_integer("tile_gap", value, 0) for value in gaps)
        if self.tile_gap[0] > self.tile_gap[1]:
            raise ValueError("tile_gap minimum exceeds maximum.")
        self.behavior_dim = _integer("behavior_dim", behavior_dim)
        self.puzzle_dim = _integer("puzzle_dim", puzzle_dim)
        self.num_hidden_units = _integer("num_hidden_units", num_hidden_units)
        self.head_hidden_units = _integer("head_hidden_units", head_hidden_units)
        self.dropout = _real("dropout", dropout, 0, 1)
        self.normalize = _boolean("normalize", normalize)
        if trunk_block not in TRUNK_BLOCKS:
            raise ValueError(f"trunk_block must be one of {TRUNK_BLOCKS}.")
        self.trunk_block = trunk_block
        self.temperature = _real("temperature", temperature, 0, strict_min=True)
        self.lambda_infonce = _real("lambda_infonce", lambda_infonce, 0)
        self.lambda_puzzle = _real("lambda_puzzle", lambda_puzzle, 0)
        self.puzzle_warmup_fraction = _real("puzzle_warmup_fraction", puzzle_warmup_fraction, 0, 1)
        self.lambda_decorrelation = _real("lambda_decorrelation", lambda_decorrelation, 0)
        self.puzzle_grad_scale = _real("puzzle_grad_scale", puzzle_grad_scale, 0)
        if puzzle_head not in PUZZLE_HEADS:
            raise ValueError(f"puzzle_head must be one of {PUZZLE_HEADS}.")
        self.puzzle_head = puzzle_head
        self.rank_weight = _real("rank_weight", rank_weight, 0, 1)
        self.sinkhorn_iterations = _integer("sinkhorn_iterations", sinkhorn_iterations, 0)
        if tile_norm not in TILE_NORMS:
            raise ValueError(f"tile_norm must be one of {TILE_NORMS}.")
        self.tile_norm = tile_norm
        self.neuron_dropout = _real("neuron_dropout", neuron_dropout, 0, 1)
        self.gain_jitter = _real("gain_jitter", gain_jitter, 0)
        self.noise_std = _real("noise_std", noise_std, 0)
        self.shortcut_reject = _real("shortcut_reject", shortcut_reject, 0, 1)
        if separate_puzzle_view is None:
            separate_puzzle_view = tile_norm != "none"
        self.separate_puzzle_view = _boolean("separate_puzzle_view", separate_puzzle_view)
        if self.tile_norm != "none" and not self.separate_puzzle_view:
            raise ValueError("tile_norm != 'none' requires separate_puzzle_view=True, otherwise "
                             "transform would see a different input distribution than training.")
        self.shuffle_tiles = _boolean("shuffle_tiles", shuffle_tiles)
        self.batch_size = _integer("batch_size", batch_size, 2)
        self.max_epochs = _integer("max_epochs", max_epochs, 0)
        self.learning_rate = _real("learning_rate", learning_rate, 0, strict_min=True)
        self.weight_decay = _real("weight_decay", weight_decay, 0)
        self.device = str(device)
        self.random_state = _integer("random_state", random_state, 0)
        self.verbose = _boolean("verbose", verbose)
        self.log_every = _integer("log_every", log_every)

    # -- configuration ----------------------------------------------------- #
    @property
    def training_span(self):
        """Raw bins needed to cut one puzzle; the receptive field stays window_size."""
        return self.n_tiles * self.window_size + (self.n_tiles - 1) * self.tile_gap[1]

    @property
    def output_dimension(self):
        return self.behavior_dim + self.puzzle_dim

    def _block_dimension(self, block):
        if block == "behavior":
            return self.behavior_dim
        if block == "puzzle":
            return self.puzzle_dim
        if block == "all":
            return self.output_dimension
        raise ValueError("block must be 'behavior', 'puzzle' or 'all'.")

    def get_params(self):
        names = ("window_size", "n_tiles", "tile_gap", "behavior_dim", "puzzle_dim",
                 "num_hidden_units", "head_hidden_units", "dropout", "normalize",
                 "trunk_block",
                 "temperature", "lambda_infonce", "lambda_puzzle", "puzzle_warmup_fraction",
                 "lambda_decorrelation", "puzzle_grad_scale", "puzzle_head", "rank_weight",
                 "sinkhorn_iterations", "tile_norm", "neuron_dropout", "gain_jitter",
                 "noise_std", "shortcut_reject", "separate_puzzle_view", "shuffle_tiles",
                 "batch_size", "max_epochs", "learning_rate", "weight_decay", "device",
                 "random_state", "verbose", "log_every")
        return {name: getattr(self, name) for name in names}

    def _build(self, channels):
        name = self.device
        if name == "cuda_if_available":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_ = torch.device(name)
        self.n_features_in_ = int(channels)
        self.permutation_table_ = _permutation_table(self.n_tiles)
        self.encoder_ = _Encoder(channels, self.window_size, self.num_hidden_units,
                                 self.behavior_dim, self.puzzle_dim, self.dropout,
                                 self.normalize, self.trunk_block).to(self.device_)
        self.head_ = _PuzzleHead(self.puzzle_dim, self.head_hidden_units,
                                 self.n_tiles).to(self.device_)
        self._generator = torch.Generator(device=self.device_).manual_seed(self.random_state)

    def _check_fitted(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit(X_train) first; max_epochs=0 gives a random-encoder control.")

    def _modules(self):
        return self.encoder_, self.head_

    # -- span bookkeeping -------------------------------------------------- #
    def _spans(self, X, n_features=None):
        """Concatenate sequences and list every span start that stays inside one."""
        sequences, _ = _sequences(X, self.training_span, n_features)
        lengths = [len(sequence) for sequence in sequences]
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        starts = np.concatenate([offsets[index] + np.arange(length - self.training_span + 1)
                                 for index, length in enumerate(lengths)])
        return np.concatenate(sequences, axis=0), starts.astype(np.int64), offsets

    def _draw_gaps(self, batch, rng):
        low, high = self.tile_gap
        return rng.integers(low, high + 1, size=(batch, self.n_tiles - 1)).astype(np.int64)

    def _tiles(self, data, starts, gaps):
        return _gather_tiles(data,
                             torch.as_tensor(starts, dtype=torch.long, device=data.device),
                             torch.as_tensor(gaps, dtype=torch.long, device=data.device),
                             self.window_size)

    def _puzzle_terms(self, logits, scores, positions, weight):
        zero = logits.new_zeros(())
        assignment = (_assignment_loss(logits, positions, self.sinkhorn_iterations, weight)
                      if self.puzzle_head in ("assign", "both") else zero)
        rank = (_rank_loss(scores, positions, weight)
                if self.puzzle_head in ("rank", "both") else zero)
        if self.puzzle_head == "both":
            combined = self.rank_weight * rank + (1 - self.rank_weight) * assignment
        else:
            combined = rank if self.puzzle_head == "rank" else assignment
        return combined, assignment, rank

    def _predicted_positions(self, logits, scores):
        if self.puzzle_head == "rank":
            return _ranks(scores)
        return _best_assignment(logits, self.permutation_table_)

    # -- training ---------------------------------------------------------- #
    def _training_step(self, tiles_raw, chronological):
        batch = tiles_raw.shape[0]
        noise = self.noise_std * self.data_scale_
        view_a = _augment(tiles_raw, self.neuron_dropout, self.gain_jitter, noise, self._generator)
        features_a = self.encoder_.features(
            view_a.reshape(-1, self.n_features_in_, self.window_size))
        z_behavior = self.encoder_.behavior(features_a).reshape(batch, self.n_tiles, self.behavior_dim)
        z_puzzle_a = self.encoder_.puzzle(features_a).reshape(batch, self.n_tiles, self.puzzle_dim)

        if self.separate_puzzle_view:
            view_b = _tile_normalize(
                _augment(tiles_raw, self.neuron_dropout, self.gain_jitter, noise, self._generator),
                self.tile_norm)
            features_b = self.encoder_.features(
                view_b.reshape(-1, self.n_features_in_, self.window_size))
        else:
            features_b = features_a
        z_puzzle_b = self.encoder_.puzzle(
            _GradScale.apply(features_b, self.puzzle_grad_scale)
        ).reshape(batch, self.n_tiles, self.puzzle_dim)

        weight = torch.ones(batch, device=tiles_raw.device)
        if self.shortcut_reject > 0:
            trivial = (_ranks(tiles_raw.mean(dim=(2, 3))) == chronological).all(dim=1)
            drawn = torch.rand(batch, device=tiles_raw.device, generator=self._generator)
            weight = weight.masked_fill(trivial & (drawn < self.shortcut_reject), 0.0)

        positions = chronological
        if self.shuffle_tiles:
            noise_keys = torch.rand((batch, self.n_tiles), device=tiles_raw.device,
                                    generator=self._generator)
            positions = noise_keys.argsort(dim=1)
            z_puzzle_b = z_puzzle_b.gather(
                1, positions[:, :, None].expand(-1, -1, self.puzzle_dim))

        logits, scores = self.head_(z_puzzle_b)
        puzzle, assignment, rank = self._puzzle_terms(logits, scores, positions, weight)
        contrastive, contrastive_accuracy = _infonce(z_behavior, self.temperature, self._generator)
        decorrelation = _decorrelation(z_behavior.reshape(-1, self.behavior_dim),
                                       z_puzzle_a.reshape(-1, self.puzzle_dim))
        with torch.no_grad():
            exact, pair = _order_metrics(self._predicted_positions(logits.detach(), scores.detach()),
                                         positions)
        return dict(puzzle=puzzle, assignment=assignment, rank=rank, contrastive=contrastive,
                    decorrelation=decorrelation, contrastive_accuracy=contrastive_accuracy,
                    exact=exact.mean(), pair=pair.mean(), kept=weight.mean())

    def fit(self, X, X_valid=None, *, validate_every=1, early_stopping_patience=None):
        """Train both objectives. X_valid is used for MONITORING only (and, if
        early_stopping_patience is set, for restoring the best puzzle checkpoint).

        Model selection should ultimately use evaluate_decoding, not puzzle loss.
        """
        validate_every = _integer("validate_every", validate_every)
        if early_stopping_patience is not None:
            early_stopping_patience = _integer("early_stopping_patience", early_stopping_patience)
        data, starts, _ = self._spans(X)
        self.is_fitted_ = False
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self._build(data.shape[1])
        scale = float(np.std(data))
        self.data_scale_ = scale if scale > 0 else 1.0
        rng = np.random.default_rng(self.random_state)
        tensor = torch.from_numpy(data).to(self.device_)
        chronological = torch.arange(self.n_tiles, device=self.device_).expand(self.batch_size, -1)
        self.n_spans_ = int(len(starts))
        if self.n_spans_ < 2 and self.max_epochs > 0:
            raise ValueError(
                f"Only {self.n_spans_} valid span start(s): every training batch would hold "
                f"fewer than two spans and InfoNCE would have no negatives. Each sequence needs "
                f"at least training_span+1={self.training_span + 1} bins, or reduce window_size / "
                f"n_tiles / tile_gap.")
        self.history_, self.validation_history_, self.n_steps_ = [], [], 0
        steps_per_epoch = max(1, math.ceil(self.n_spans_ / self.batch_size))
        warmup = int(self.puzzle_warmup_fraction * max(1, self.max_epochs) * steps_per_epoch)
        optimizer = torch.optim.Adam(
            list(self.encoder_.parameters()) + list(self.head_.parameters()),
            lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.verbose:
            print(f"JigsawCEBRA: {self.n_spans_} spans, window={self.window_size}, "
                  f"n_tiles={self.n_tiles}, tile_gap={self.tile_gap}, "
                  f"training_span={self.training_span}, latent={self.behavior_dim}+{self.puzzle_dim}, "
                  f"trunk={self.trunk_block}, head={self.puzzle_head}, tile_norm={self.tile_norm}, "
                  f"lambda_infonce={self.lambda_infonce}, lambda_puzzle={self.lambda_puzzle}, "
                  f"device={self.device_}", flush=True)
            if self.max_epochs == 0:
                print("Random-encoder control: zero optimizer steps.", flush=True)

        best = dict(value=math.inf, epoch=0, state=None, waited=0)
        for epoch in range(self.max_epochs):
            self.encoder_.train()
            self.head_.train()
            order = rng.permutation(self.n_spans_)
            totals, seen = {}, 0
            for begin in range(0, self.n_spans_, self.batch_size):
                chunk = starts[order[begin:begin + self.batch_size]]
                if len(chunk) < 2:  # InfoNCE needs at least two distinct spans.
                    continue
                tiles = self._tiles(tensor, chunk, self._draw_gaps(len(chunk), rng))
                optimizer.zero_grad(set_to_none=True)
                parts = self._training_step(tiles, chronological[:len(chunk)])
                weight_puzzle = self.lambda_puzzle * (1.0 if warmup == 0
                                                      else min(1.0, (self.n_steps_ + 1) / warmup))
                loss = (self.lambda_infonce * parts["contrastive"]
                        + weight_puzzle * parts["puzzle"]
                        + self.lambda_decorrelation * parts["decorrelation"])
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Nonfinite loss at optimizer step {self.n_steps_}.")
                loss.backward()
                optimizer.step()
                self.n_steps_ += 1
                seen += len(chunk)
                totals["loss"] = totals.get("loss", 0.0) + loss.item() * len(chunk)
                totals["lambda_puzzle"] = weight_puzzle
                for key, value in parts.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(chunk)
            row = {key: (value / max(seen, 1) if key != "lambda_puzzle" else value)
                   for key, value in totals.items()}
            row.update(epoch=epoch + 1, steps=self.n_steps_, n_spans=seen)
            self.history_.append(row)
            if self.verbose and ((epoch + 1) % self.log_every == 0 or epoch + 1 == self.max_epochs):
                print(f"Epoch {epoch + 1}/{self.max_epochs}: loss={row['loss']:.5f} "
                      f"| infonce={row['contrastive']:.4f} (acc {100 * row['contrastive_accuracy']:.1f}%) "
                      f"| puzzle={row['puzzle']:.4f} (pair {100 * row['pair']:.1f}%, "
                      f"exact {100 * row['exact']:.1f}%) "
                      f"| decorr={row['decorrelation']:.5f} | lam={row['lambda_puzzle']:.3f}", flush=True)

            if X_valid is not None and (epoch + 1) % validate_every == 0:
                self.is_fitted_ = True  # allow the evaluator to run mid-training
                metrics = self.evaluate_puzzle(X_valid, max_spans=512, verbose=False)
                self.is_fitted_ = False
                metrics["epoch"] = epoch + 1
                self.validation_history_.append(metrics)
                monitored = (metrics["rank_loss"] if self.puzzle_head == "rank"
                             else metrics["assignment_cross_entropy"])
                if self.verbose:
                    print(f"           valid: monitored={monitored:.4f}, "
                          f"pair={metrics['pair_accuracy_percent']:.1f}% "
                          f"(mean-sort baseline {metrics['baseline_mean_sort_pair_percent']:.1f}%), "
                          f"exact={metrics['exact_accuracy_percent']:.1f}%", flush=True)
                if early_stopping_patience is not None:
                    if monitored < best["value"] - 1e-6:
                        best.update(value=monitored, epoch=epoch + 1, waited=0, state=(
                            copy.deepcopy(self.encoder_.state_dict()),
                            copy.deepcopy(self.head_.state_dict())))
                    else:
                        best["waited"] += 1
                        if best["waited"] >= early_stopping_patience:
                            if best["state"] is not None:
                                self.encoder_.load_state_dict(best["state"][0])
                                self.head_.load_state_dict(best["state"][1])
                            if self.verbose:
                                print(f"Early stop at epoch {epoch + 1}; restored epoch "
                                      f"{best['epoch']}.", flush=True)
                            break
        self.best_epoch_ = best["epoch"] if best["state"] is not None else self.max_epochs
        self.encoder_.eval()
        self.head_.eval()
        self.is_fitted_ = True
        return self

    # -- inference --------------------------------------------------------- #
    def transform(self, X, *, block="behavior", pad=True, batch_size=None, return_indices=False):
        """Natural, unshuffled, unnormalized, unaugmented windows. No puzzle head.

        block='behavior' (default) is the representation to decode. 'all' returns
        [z_behavior, z_puzzle]; 'puzzle' returns the nuisance block alone, which
        is useful to confirm that order information really moved there.
        """
        self._check_fitted()
        _boolean("pad", pad)
        dimension = self._block_dimension(block)
        size = self.batch_size if batch_size is None else _integer("batch_size", batch_size)
        sequences, is_list = _sequences(X, 1 if pad else self.window_size, self.n_features_in_)
        left = self.window_size // 2
        right = self.window_size - left - 1
        result, indices = [], []
        modes = self.encoder_.training, self.head_.training
        self.encoder_.eval()
        self.head_.eval()
        try:
            with torch.inference_mode():
                for sequence in sequences:
                    if pad:
                        centers = np.arange(len(sequence), dtype=np.int64)
                        sequence = np.pad(sequence, ((left, right), (0, 0)), mode="edge")
                    else:
                        centers = np.arange(left, len(sequence) - right, dtype=np.int64)
                    embedding = np.empty((len(centers), dimension), dtype=np.float32)
                    for start in range(0, len(centers), size):
                        end = min(start + size, len(centers))
                        locations = np.arange(start, end)[:, None] + np.arange(self.window_size)
                        windows = np.ascontiguousarray(sequence[locations].transpose(0, 2, 1))
                        features = self.encoder_.features(
                            torch.from_numpy(windows).to(self.device_))
                        if block == "behavior":
                            output = self.encoder_.behavior(features)
                        elif block == "puzzle":
                            output = self.encoder_.puzzle(features)
                        else:
                            output = torch.cat((self.encoder_.behavior(features),
                                                self.encoder_.puzzle(features)), dim=1)
                        embedding[start:end] = output.cpu().numpy()
                    result.append(embedding)
                    indices.append(centers)
        finally:
            self.encoder_.train(modes[0])
            self.head_.train(modes[1])
        values = result if is_list else result[0]
        times = indices if is_list else indices[0]
        return (values, times) if return_indices else values

    def fit_transform(self, X, **transform_kwargs):
        return self.fit(X).transform(X, **transform_kwargs)

    # -- evaluation -------------------------------------------------------- #
    def evaluate_puzzle(self, X, *, max_spans=1024, batch_size=256, random_state=200042,
                        verbose=None, return_details=False):
        """Fixed-weight puzzle metrics next to the trivial shortcut baselines.

        Uses tile_norm (as in training) but NO augmentation. Overlapping spans
        are correlated observations, not independent replicates. If the model's
        pair accuracy is not clearly above baseline_mean_sort_pair_percent, the
        pretext is being solved by level drift and nothing has been learned.
        """
        self._check_fitted()
        maximum = _integer("max_spans", max_spans)
        size = _integer("batch_size", batch_size)
        seed = _integer("random_state", random_state, 0)
        data, starts, _ = self._spans(X, self.n_features_in_)
        rng = np.random.default_rng(seed)
        count = min(maximum, len(starts))
        chosen = np.sort(rng.choice(len(starts), size=count, replace=False))
        selected = starts[chosen]
        gaps = self._draw_gaps(count, rng)
        tensor = torch.from_numpy(data).to(self.device_)
        chronological = torch.arange(self.n_tiles, device=self.device_)
        collected = {key: [] for key in ("assignment", "rank", "exact", "pair", "mean_exact",
                                         "mean_pair", "norm_exact", "norm_pair", "score_exact",
                                         "score_pair")}
        predictions = []
        modes = self.encoder_.training, self.head_.training
        self.encoder_.eval()
        self.head_.eval()
        try:
            with torch.inference_mode():
                for begin in range(0, count, size):
                    end = min(begin + size, count)
                    tiles = self._tiles(tensor, selected[begin:end], gaps[begin:end])
                    batch = tiles.shape[0]
                    positions = chronological.expand(batch, -1)
                    view = _tile_normalize(tiles, self.tile_norm)
                    features = self.encoder_.features(
                        view.reshape(-1, self.n_features_in_, self.window_size))
                    z = self.encoder_.puzzle(features).reshape(batch, self.n_tiles, self.puzzle_dim)
                    logits, scores = self.head_(z)
                    if not torch.isfinite(logits).all().item():
                        raise FloatingPointError("Nonfinite puzzle logits during evaluation.")
                    ones = torch.ones(batch, device=tiles.device)
                    _, assignment, rank = self._puzzle_terms(logits, scores, positions, ones)
                    predicted = self._predicted_positions(logits, scores)
                    exact, pair = _order_metrics(predicted, positions)
                    score_exact, score_pair = _order_metrics(_ranks(scores), positions)
                    level = _ranks(tiles.mean(dim=(2, 3)))
                    norm = _ranks(tiles.flatten(2).norm(dim=2))
                    mean_exact, mean_pair = _order_metrics(level, positions)
                    norm_exact, norm_pair = _order_metrics(norm, positions)
                    collected["assignment"].append(float(assignment) * batch)
                    collected["rank"].append(float(rank) * batch)
                    for key, value in (("exact", exact), ("pair", pair),
                                       ("mean_exact", mean_exact), ("mean_pair", mean_pair),
                                       ("norm_exact", norm_exact), ("norm_pair", norm_pair),
                                       ("score_exact", score_exact), ("score_pair", score_pair)):
                        collected[key].append(float(value.sum()))
                    predictions.append(predicted.cpu().numpy())
        finally:
            self.encoder_.train(modes[0])
            self.head_.train(modes[1])
        total = float(count)
        summed = {key: sum(values) / total for key, values in collected.items()}
        metrics = dict(
            objective=f"{self.puzzle_head} head, {self.n_tiles} tiles, relational and equivariant",
            window_size=self.window_size, n_tiles=self.n_tiles, tile_gap=list(self.tile_gap),
            tile_norm=self.tile_norm, training_span=self.training_span,
            assignment_cross_entropy=summed["assignment"], rank_loss=summed["rank"],
            uniform_assignment_cross_entropy=float(np.log(self.n_tiles)),
            uniform_rank_loss=float(np.log(2)),
            exact_accuracy_percent=100 * summed["exact"], pair_accuracy_percent=100 * summed["pair"],
            rank_score_exact_percent=100 * summed["score_exact"],
            rank_score_pair_percent=100 * summed["score_pair"],
            baseline_mean_sort_exact_percent=100 * summed["mean_exact"],
            baseline_mean_sort_pair_percent=100 * summed["mean_pair"],
            baseline_norm_sort_exact_percent=100 * summed["norm_exact"],
            baseline_norm_sort_pair_percent=100 * summed["norm_pair"],
            chance_exact_accuracy_percent=100 / math.factorial(self.n_tiles),
            chance_pair_accuracy_percent=50.0,
            n_spans=count, available_spans=int(len(starts)), random_state=seed,
            sampling="Distinct span starts, fresh gaps, no augmentation, no padding.",
        )
        if verbose or (verbose is None and self.verbose):
            print(f"Puzzle: pair={metrics['pair_accuracy_percent']:.1f}% vs mean-sort "
                  f"{metrics['baseline_mean_sort_pair_percent']:.1f}% | exact="
                  f"{metrics['exact_accuracy_percent']:.1f}% vs chance "
                  f"{metrics['chance_exact_accuracy_percent']:.2f}%", flush=True)
        if not return_details:
            return metrics
        return metrics, dict(span_starts=selected, gaps=gaps,
                             predicted_positions=np.concatenate(predictions, axis=0))

    def evaluate_decoding(self, X_train, y_train, X_test, y_test, *,
                          blocks=("behavior", "puzzle", "all"),
                          alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)):
        """Frozen-encoder ridge decoding R2 per latent block. THE metric that matters.

        y must be aligned row-for-row with X (pad=True embedding, one row per bin).
        Ridge alpha is picked on a chronological 80/20 split of the training part,
        never on the test part. Compare against lambda_puzzle=0 and max_epochs=0
        runs on the SAME split before claiming the puzzle helped.
        """
        self._check_fitted()

        def stack(values):
            if isinstance(values, (list, tuple)):
                return np.concatenate([np.asarray(v, dtype=np.float64) for v in values], axis=0)
            return np.asarray(values, dtype=np.float64)

        targets_train, targets_test = stack(y_train), stack(y_test)
        if targets_train.ndim == 1:
            targets_train = targets_train[:, None]
        if targets_test.ndim == 1:
            targets_test = targets_test[:, None]
        result = {}
        for block in blocks:
            features_train = stack(self.transform(X_train, block=block, pad=True))
            features_test = stack(self.transform(X_test, block=block, pad=True))
            if len(features_train) != len(targets_train) or len(features_test) != len(targets_test):
                raise ValueError(f"Label/embedding length mismatch: "
                                 f"{len(features_train)} vs {len(targets_train)} (train), "
                                 f"{len(features_test)} vs {len(targets_test)} (test).")
            result[block] = _ridge_r2(features_train.astype(np.float64), targets_train,
                                      features_test.astype(np.float64), targets_test, alphas)
        return result

    # -- persistence ------------------------------------------------------- #
    def save(self, path):
        """Weights and config only; no data and no optimizer state."""
        self._check_fitted()
        torch.save(dict(
            model_type="jigsaw_cebra", format_version=1, params=self.get_params(),
            n_features=self.n_features_in_, data_scale=self.data_scale_,
            encoder={k: v.detach().cpu() for k, v in self.encoder_.state_dict().items()},
            head={k: v.detach().cpu() for k, v in self.head_.state_dict().items()},
            history=self.history_, validation_history=self.validation_history_,
            n_steps=self.n_steps_, n_spans=self.n_spans_,
        ), Path(path))

    @classmethod
    def load(cls, path, device="cuda_if_available"):
        try:
            data = torch.load(Path(path), map_location="cpu", weights_only=True)
        except TypeError:  # torch < 1.13 has no weights_only argument
            data = torch.load(Path(path), map_location="cpu")
        if data.get("model_type") != "jigsaw_cebra" or data.get("format_version") != 1:
            raise ValueError("Expected a jigsaw_cebra checkpoint.")
        model = cls(**dict(data["params"], device=device))
        model._build(data["n_features"])
        model.encoder_.load_state_dict(data["encoder"])
        model.head_.load_state_dict(data["head"])
        model.data_scale_ = data["data_scale"]
        model.history_ = data["history"]
        model.validation_history_ = data["validation_history"]
        model.n_steps_ = data["n_steps"]
        model.n_spans_ = data["n_spans"]
        model.encoder_.eval()
        model.head_.eval()
        model.is_fitted_ = True
        return model


Jigsaw = JigsawCEBRA
puzzle = JigsawCEBRA


# --------------------------------------------------------------------------- #
# self test
# --------------------------------------------------------------------------- #
def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _self_test(device="cpu"):
    """Verify the pieces that are easy to get silently wrong. Run this first."""
    torch.manual_seed(0)
    dev = torch.device(device)

    # 1. Tile arithmetic: gaps are DISCARDED bins, tiles never overlap.
    ramp = torch.arange(200, dtype=torch.float32, device=dev)[:, None].repeat(1, 3)
    starts = torch.tensor([0, 10], device=dev)
    gaps = torch.tensor([[1, 2, 3], [0, 0, 0]], device=dev)
    tiles = _gather_tiles(ramp, starts, gaps, 4)
    _check(tuple(tiles.shape) == (2, 4, 3, 4), f"tile shape {tuple(tiles.shape)}")
    _check(tiles[0, :, 0, 0].tolist() == [0, 5, 11, 18], "tile starts with gaps (1,2,3)")
    _check(tiles[1, :, 0, 0].tolist() == [10, 14, 18, 22], "tile starts with zero gaps")
    _check(tiles[0, :, 0, -1].tolist() == [3, 8, 14, 21], "tile ends")
    ends = tiles[:, :-1, 0, -1]
    _check(bool((tiles[:, 1:, 0, 0] > ends).all()), "tiles must not overlap")

    # 2. Trunk receptive field is exactly window_size, for both block types.
    for block in TRUNK_BLOCKS:
        for window in (4, 5, 9, 10, 16, 21):
            trunk = _Trunk(3, window, 8, 0.0, block).to(dev).eval()
            _check(tuple(trunk(torch.randn(2, 3, window, device=dev)).shape) == (2, 8),
                   f"trunk({block}) output for window_size={window}")
            try:
                trunk(torch.randn(2, 3, window + 1, device=dev))
            except ValueError:
                pass
            else:
                raise AssertionError("trunk accepted the wrong number of bins")
        grad_input = torch.randn(1, 3, 10, device=dev, requires_grad=True)
        _Trunk(3, 10, 8, 0.0, block).to(dev).eval()(grad_input).sum().backward()
        _check(bool((grad_input.grad.abs().sum(dim=(0, 1)) > 0).all()),
               f"every bin in the window must reach the output ({block})")
    # The depthwise block must not mix channels inside its temporal convolution.
    depthwise = [m for m in _Separable(8, 0.0).modules()
                 if isinstance(m, nn.Conv1d) and m.kernel_size == (3,)]
    _check(len(depthwise) == 1 and depthwise[0].groups == depthwise[0].in_channels,
           "the separable block's 3-tap convolution must be depthwise")
    _check(sum(p.numel() for p in _Separable(64, 0.0).parameters())
           < 4 * sum(p.numel() for p in _Residual(64, 0.0).parameters()),
           "the separable block should stay in the same parameter class as the plain one")

    # 3. Head equivariance -> shuffling tiles cannot change the loss.
    batch, n_tiles, dimension = 6, 4, 5
    head = _PuzzleHead(dimension, 16, n_tiles).to(dev).double().eval()
    z = torch.randn(batch, n_tiles, dimension, device=dev, dtype=torch.float64)
    order = torch.stack([torch.randperm(n_tiles, device=dev) for _ in range(batch)])
    logits, scores = head(z)
    shuffled_logits, shuffled_scores = head(z.gather(1, order[:, :, None].expand(-1, -1, dimension)))
    _check(torch.allclose(shuffled_logits,
                          logits.gather(1, order[:, :, None].expand(-1, -1, n_tiles)), atol=1e-10),
           "puzzle head is not permutation-equivariant")
    _check(torch.allclose(shuffled_scores, scores.gather(1, order), atol=1e-10),
           "puzzle scores are not permutation-equivariant")
    chronological = torch.arange(n_tiles, device=dev).expand(batch, -1)
    ones = torch.ones(batch, device=dev, dtype=torch.float64)
    _check(abs(float(_assignment_loss(logits, chronological, 5, ones))
               - float(_assignment_loss(shuffled_logits, order, 5, ones))) < 1e-10,
           "assignment loss changed under shuffling")
    _check(abs(float(_rank_loss(scores, chronological, ones))
               - float(_rank_loss(shuffled_scores, order, ones))) < 1e-10,
           "rank loss changed under shuffling")

    # 4. Sinkhorn: rows are exact probabilities, columns converge to uniform.
    mild = torch.randn(200, 5, 5, device=dev, dtype=torch.float64)
    log_p = _log_sinkhorn(mild, 50)
    _check(torch.allclose(log_p.exp().sum(2), torch.ones(200, 5, device=dev, dtype=torch.float64),
                          atol=1e-9), "sinkhorn rows are not probabilities")
    _check(float((log_p.exp().sum(1) - 1).abs().max()) < 1e-6, "sinkhorn columns far from uniform")
    sharp = mild * 4
    deviations = [float((_log_sinkhorn(sharp, it).exp().sum(1) - 1).abs().max())
                  for it in (0, 1, 5, 20, 100)]
    _check(all(a > b for a, b in zip(deviations, deviations[1:])),
           f"more sinkhorn iterations must balance the columns further: {deviations}")
    # At the default 5 iterations with sharp logits the matrix is only PARTIALLY
    # balanced: the assignment loss is a soft bijection prior, not a constraint.
    _check(torch.allclose(_log_sinkhorn(sharp, 0), sharp - torch.logsumexp(sharp, 2, keepdim=True)),
           "sinkhorn_iterations=0 must reduce to a per-slot softmax over positions")
    _check(bool(torch.isfinite(_log_sinkhorn(torch.zeros(2, 4, 4, device=dev), 5)).all()),
           "sinkhorn produced nonfinite values")

    # 5. Assignment decoding is bijective and exact.
    table = _permutation_table(4)
    truth = torch.tensor([[2, 0, 3, 1], [0, 1, 2, 3]], device=dev)
    onehot = F.one_hot(truth, 4).float() * 12 + torch.randn(2, 4, 4, device=dev) * 0.1
    _check(torch.equal(_best_assignment(onehot, table), truth), "exact assignment failed")
    _check(torch.equal(_best_assignment(onehot, None), truth), "greedy assignment failed")
    random_logits = torch.randn(32, 4, 4, device=dev)
    decoded = _best_assignment(random_logits, table)
    _check(bool((decoded.sort(dim=1).values == torch.arange(4, device=dev)).all()),
           "assignment is not a permutation")
    exact_score = random_logits.gather(2, decoded[:, :, None]).sum((1, 2))
    greedy_score = random_logits.gather(2, _best_assignment(random_logits, None)[:, :, None]).sum((1, 2))
    _check(bool((exact_score >= greedy_score - 1e-6).all()), "greedy beat the exact solver")

    # 6. Ranks and order metrics.
    _check(_ranks(torch.tensor([[3.0, 1.0, 2.0]], device=dev)).tolist() == [[2, 0, 1]], "_ranks")
    exact, pair = _order_metrics(chronological, chronological)
    _check(float(exact.mean()) == 1.0 and float(pair.mean()) == 1.0, "perfect order metrics")
    reverse = torch.arange(n_tiles - 1, -1, -1, device=dev).expand(batch, -1)
    exact, pair = _order_metrics(reverse, chronological)
    _check(float(exact.mean()) == 0.0 and float(pair.mean()) == 0.0, "reversed order metrics")

    # 7. InfoNCE separates spans and flags leakage through its accuracy.
    directions = F.normalize(torch.randn(8, 12, device=dev), dim=1)
    easy = directions[:, None, :].repeat(1, 4, 1)
    loss, accuracy = _infonce(easy, 0.1, torch.Generator(device=dev).manual_seed(0))
    collapse = math.log(1 + (8 - 1) * 4)  # every embedding identical: the chance level.
    _check(float(accuracy) == 1.0 and float(loss) < 0.2 * collapse,
           f"InfoNCE on separable spans: {float(loss)} (collapse costs {collapse:.3f})")
    flat = F.normalize(torch.randn(1, 12, device=dev), dim=1).expand(8, 12)
    collapsed_loss, _ = _infonce(flat[:, None, :].repeat(1, 4, 1), 0.1,
                                 torch.Generator(device=dev).manual_seed(0))
    _check(abs(float(collapsed_loss) - collapse) < 1e-4,
           f"a collapsed embedding must cost exactly log(1+(B-1)K)={collapse:.4f}")
    try:
        _infonce(easy[:1], 1.0, torch.Generator(device=dev).manual_seed(0))
    except ValueError:
        pass
    else:
        raise AssertionError("InfoNCE accepted a single-span batch")

    # 8. Decorrelation penalty responds in the right direction.
    a = torch.randn(4096, 8, device=dev)
    b = torch.randn(4096, 8, device=dev)
    _check(float(_decorrelation(a, a)) > 10 * float(_decorrelation(a, b)),
           "decorrelation does not separate identical from independent blocks")

    # 9. Tile normalization removes a per-tile level offset exactly.
    raw = torch.randn(3, 4, 6, 10, device=dev)
    offset = torch.randn(3, 4, 1, 1, device=dev) * 5
    for mode in ("mean", "zscore", "global_mean"):
        shifted = _tile_normalize(raw + offset, mode)
        _check(torch.allclose(shifted, _tile_normalize(raw, mode), atol=1e-4),
               f"tile_norm={mode} did not remove the level shortcut")
    _check(torch.equal(_tile_normalize(raw, "none"), raw), "tile_norm='none' must be identity")
    ramp = raw + torch.arange(4, device=dev).reshape(1, 4, 1, 1) * 3.0
    slots = torch.arange(4, device=dev)
    _check(bool((_ranks(ramp.mean(dim=(2, 3))) == slots).all()),
           "mean-sort must solve a level ramp perfectly: that is the shortcut the baseline measures")
    _check(not bool((_ranks(_tile_normalize(ramp, "mean").mean(dim=(2, 3))) == slots).all()),
           "tile normalization must destroy the level-ramp shortcut")

    # 10. Augmentation endpoints.
    generator = torch.Generator(device=dev).manual_seed(0)
    _check(float(_augment(raw, 1.0, 0.0, 0.0, generator).abs().max()) == 0.0,
           "neuron_dropout=1 must zero the input")
    _check(torch.equal(_augment(raw, 0.0, 0.0, 0.0, generator), raw),
           "augmentation with all strengths zero must be identity")
    dropped = _augment(raw, 0.5, 0.0, 0.0, generator) == 0
    _check(bool((dropped.all(dim=3) | (~dropped).all(dim=3)).all()),
           "neuron dropout must drop whole neurons, not single bins")

    # 11. Ridge recovers a linear map.
    features = np.random.default_rng(0).normal(size=(600, 6))
    mapping = np.random.default_rng(1).normal(size=(6, 2))
    result = _ridge_r2(features[:400], features[:400] @ mapping,
                       features[400:], features[400:] @ mapping, (1e-3, 1e-2, 1e-1))
    _check(result["r2"] > 0.99, f"ridge R2 on a linear problem was {result['r2']:.4f}")
    _check(4.0 < result["participation_ratio"] <= 6.01,
           f"participation ratio {result['participation_ratio']:.2f}")

    # 12. Spans never cross a trial boundary.
    model = JigsawCEBRA(window_size=8, n_tiles=3, tile_gap=(1, 4), max_epochs=0,
                        device=device, verbose=False)
    trials = [np.random.default_rng(s).normal(size=(120, 5)).astype(np.float32) for s in (0, 1)]
    data, starts, offsets = model._spans(trials)
    span = model.training_span
    _check(len(starts) == sum(len(trial) - span + 1 for trial in trials), "span count")
    for index in range(len(trials)):
        inside = starts[(starts >= offsets[index]) & (starts < offsets[index + 1])]
        _check(int(inside.max()) + span <= offsets[index + 1], "a span crossed a trial boundary")

    # 13. puzzle_grad_scale=0 cuts the pretext gradient out of the shared trunk.
    blocked = JigsawCEBRA(window_size=8, n_tiles=3, tile_gap=(1, 2), behavior_dim=6, puzzle_dim=4,
                          num_hidden_units=8, head_hidden_units=8, puzzle_grad_scale=0.0,
                          device=device, verbose=False)
    blocked._build(5)
    blocked.data_scale_ = 1.0
    sample = torch.randn(4, 3, 5, 8, device=blocked.device_)
    order = torch.arange(3, device=blocked.device_).expand(4, -1)
    blocked._training_step(sample, order)["puzzle"].backward()
    trunk_grad = sum(float(p.grad.abs().sum()) for p in blocked.encoder_.trunk.parameters()
                     if p.grad is not None)
    head_grad = sum(float(p.grad.abs().sum()) for p in blocked.head_.parameters()
                    if p.grad is not None)
    _check(trunk_grad == 0.0, f"trunk received puzzle gradient despite scale 0 ({trunk_grad})")
    _check(head_grad > 0.0, "puzzle head received no gradient")
    _check(float(blocked.encoder_.project_puzzle.weight.grad.abs().sum()) > 0,
           "puzzle projector received no gradient")

    # 14. End to end: fit, transform, evaluate, save, load.
    rng = np.random.default_rng(3)
    train = rng.normal(size=(700, 5)).astype(np.float32)
    test = rng.normal(size=(300, 5)).astype(np.float32)
    fitted = JigsawCEBRA(window_size=8, n_tiles=3, tile_gap=(1, 3), behavior_dim=6, puzzle_dim=4,
                         num_hidden_units=12, head_hidden_units=12, batch_size=64, max_epochs=2,
                         device=device, verbose=False, random_state=7)
    fitted.fit(train, X_valid=test, validate_every=1)
    _check(len(fitted.history_) == 2 and len(fitted.validation_history_) == 2, "history lengths")
    for block, width in (("behavior", 6), ("puzzle", 4), ("all", 10)):
        padded = fitted.transform(train, block=block, pad=True)
        _check(padded.shape == (len(train), width), f"padded transform shape for {block}")
        cropped = fitted.transform(train, block=block, pad=False)
        _check(cropped.shape == (len(train) - 7, width), f"unpadded transform shape for {block}")
    if fitted.normalize:
        both = fitted.transform(test, block="all")
        _check(np.allclose(np.linalg.norm(both[:, :6], axis=1), 1, atol=1e-4)
               and np.allclose(np.linalg.norm(both[:, 6:], axis=1), 1, atol=1e-4),
               "blocks in 'all' must be normalized separately")
    _check(np.array_equal(fitted.transform(test), fitted.transform(test)),
           "transform is not deterministic")
    as_list, times = fitted.transform([train, test], return_indices=True)
    _check(isinstance(as_list, list) and len(as_list) == 2 and len(times[1]) == len(test),
           "list input must return a list")
    metrics = fitted.evaluate_puzzle(test, max_spans=64, verbose=False)
    for key in ("pair_accuracy_percent", "exact_accuracy_percent",
                "baseline_mean_sort_pair_percent", "assignment_cross_entropy"):
        _check(key in metrics and np.isfinite(metrics[key]), f"missing metric {key}")
    _check(0 <= metrics["pair_accuracy_percent"] <= 100, "pair accuracy out of range")
    scores = fitted.evaluate_decoding(train, rng.normal(size=(700, 2)), test,
                                      rng.normal(size=(300, 2)), blocks=("behavior", "all"))
    _check(set(scores) == {"behavior", "all"} and np.isfinite(scores["behavior"]["r2"]),
           "evaluate_decoding output")
    try:
        fitted.evaluate_decoding(train, rng.normal(size=(699, 2)), test,
                                 rng.normal(size=(300, 2)), blocks=("behavior",))
    except ValueError:
        pass
    else:
        raise AssertionError("evaluate_decoding accepted misaligned labels")
    path = Path("_jigsaw_cebra_selftest.pt")
    try:
        fitted.save(path)
        restored = JigsawCEBRA.load(path, device=device)
        _check(np.allclose(fitted.transform(test), restored.transform(test), atol=1e-6),
               "save/load changed the embedding")
        _check(restored.get_params() == fitted.get_params(), "save/load changed the parameters")
    finally:
        path.unlink(missing_ok=True)

    # 15. Controls run: random encoder, and each objective switched off.
    control = JigsawCEBRA(window_size=8, n_tiles=3, tile_gap=(1, 3), num_hidden_units=8,
                          max_epochs=0, device=device, verbose=False).fit(train)
    _check(control.n_steps_ == 0 and control.transform(test).shape[0] == len(test),
           "max_epochs=0 control")
    for override in ({"lambda_puzzle": 0.0}, {"lambda_infonce": 0.0},
                     {"lambda_decorrelation": 0.0}, {"puzzle_head": "rank"},
                     {"puzzle_head": "assign"}, {"shuffle_tiles": True},
                     {"shortcut_reject": 1.0}, {"tile_norm": "none", "separate_puzzle_view": False},
                     {"normalize": False}, {"n_tiles": 8}, {"n_tiles": 2},
                     {"trunk_block": "separable"}, {"puzzle_grad_scale": 0.0}):
        JigsawCEBRA(window_size=8, tile_gap=(1, 2), behavior_dim=6, puzzle_dim=4,
                    num_hidden_units=8, head_hidden_units=8, batch_size=32, max_epochs=1,
                    device=device, verbose=False, **override).fit(train).transform(test)

    # 16. Configuration guards.
    for bad in ({"tile_norm": "mean", "separate_puzzle_view": False}, {"window_size": 3},
                {"tile_gap": (5, 1)}, {"puzzle_head": "sort"}, {"rank_weight": 1.5},
                {"n_tiles": 1}, {"temperature": 0.0}, {"trunk_block": "mobilenet"}):
        try:
            JigsawCEBRA(**bad)
        except ValueError:
            continue
        raise AssertionError(f"constructor accepted {bad}")
    try:
        fitted.transform(np.full((200, 5), np.nan, dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("transform accepted nonfinite input")
    # A sequence shorter than the receptive field is only an error without padding;
    # with pad=True the edges are extended and every bin still gets a row.
    short = np.zeros((4, 5), dtype=np.float32)
    try:
        fitted.transform(short, pad=False)
    except ValueError:
        pass
    else:
        raise AssertionError("transform accepted a sequence shorter than window_size")
    _check(fitted.transform(short, pad=True).shape == (4, fitted.behavior_dim),
           "pad=True must return one row per bin even for a very short sequence")
    try:
        fitted.transform(np.zeros((200, 9), dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("transform accepted the wrong neuron count")
    # A recording too short to hold two spans must fail loudly, not divide by zero.
    try:
        JigsawCEBRA(window_size=8, n_tiles=3, tile_gap=(1, 3), max_epochs=1,
                    device=device, verbose=False).fit(np.zeros((30, 5), dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("fit accepted a recording with fewer than two spans")

    print("All self tests passed.")
    return True


# --------------------------------------------------------------------------- #
# synthetic demo: the ablation table you should reproduce on Perich
# --------------------------------------------------------------------------- #
def _synthetic(n_samples=8000, n_neurons=60, seed=0):
    """Ring latent + slow multiplicative drift (the level shortcut, on purpose)."""
    rng = np.random.default_rng(seed)
    time = np.arange(n_samples)
    speed = 0.05 + 0.03 * np.sin(2 * np.pi * time / 1700)
    angle = np.cumsum(speed)
    centers = rng.uniform(0, 2 * np.pi, n_neurons)
    rate = np.exp(2.0 * np.cos(angle[:, None] - centers[None, :]))
    drift = np.cumsum(rng.normal(0, 1, n_samples))
    drift = drift / (np.abs(drift).max() + 1e-12)
    gain = rng.uniform(0.5, 1.5, n_neurons)
    neural = rate * (1 + 0.6 * drift[:, None] * gain[None, :]) + rng.normal(0, 0.4, rate.shape)
    targets = np.stack([np.cos(angle), np.sin(angle), speed], axis=1)
    return neural.astype(np.float32), targets.astype(np.float32)


def _demo(device="cuda_if_available", max_epochs=10, seed=0, n_samples=8000):
    """Run the arms that decide whether the pretext helps. Read the R2 column."""
    neural, targets = _synthetic(n_samples=n_samples, seed=seed)
    cut = int(0.7 * len(neural))
    train, test = neural[:cut], neural[cut:]
    y_train, y_test = targets[:cut], targets[cut:]
    shared = dict(window_size=10, n_tiles=4, tile_gap=(1, 8), behavior_dim=16, puzzle_dim=8,
                  num_hidden_units=64, head_hidden_units=64, batch_size=256, learning_rate=1e-3,
                  max_epochs=max_epochs, device=device, verbose=False, random_state=seed)
    arms = {
        "jigsaw + infonce (proposed)": {},
        "infonce only (lambda_puzzle=0)": dict(lambda_puzzle=0.0),
        "puzzle only (lambda_infonce=0)": dict(lambda_infonce=0.0, lambda_decorrelation=0.0),
        "no decorrelation": dict(lambda_decorrelation=0.0),
        "mobilenet-style trunk": dict(trunk_block="separable"),
        "easiest pretext (n_tiles=2)": dict(n_tiles=2),
        "shortcut left open": dict(tile_norm="none", separate_puzzle_view=False,
                                   neuron_dropout=0.0, gain_jitter=0.0),
        "random encoder (max_epochs=0)": dict(max_epochs=0),
    }
    print(f"Synthetic ring latent: train {train.shape}, test {test.shape}, "
          f"targets cos/sin/speed. Higher R2 is better.\n")
    rows = []
    for name, override in arms.items():
        model = JigsawCEBRA(**dict(shared, **override)).fit(train)
        puzzle = model.evaluate_puzzle(test, max_spans=512, verbose=False)
        decoding = model.evaluate_decoding(train, y_train, test, y_test,
                                           blocks=("behavior", "puzzle", "all"))
        rows.append((name, decoding["behavior"]["r2"], decoding["puzzle"]["r2"],
                     decoding["all"]["r2"], puzzle["pair_accuracy_percent"],
                     puzzle["baseline_mean_sort_pair_percent"],
                     decoding["behavior"]["participation_ratio"]))
    header = (f"{'arm':<32}{'R2 behav':>10}{'R2 puzz':>9}{'R2 all':>9}"
              f"{'pair %':>9}{'shortcut %':>12}{'PR':>7}")
    print(header)
    print("-" * len(header))
    for name, behavior, puzzle_r2, everything, pair, baseline, ratio in rows:
        print(f"{name:<32}{behavior:>10.3f}{puzzle_r2:>9.3f}{everything:>9.3f}"
              f"{pair:>9.1f}{baseline:>12.1f}{ratio:>7.2f}")
    print("\nRead it like this: the pretext is only worth keeping if 'R2 behav' for the "
          "proposed arm beats BOTH 'infonce only' and 'random encoder'. If 'pair %' is not "
          "clearly above 'shortcut %', the puzzle is being solved by drift, not by structure.")
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true", help="run correctness checks")
    parser.add_argument("--demo", action="store_true", help="run the synthetic ablation table")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda_if_available")
    parser.add_argument("--epochs", type=int, default=10, help="epochs per demo arm")
    parser.add_argument("--samples", type=int, default=8000, help="synthetic sequence length")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    if not (arguments.self_test or arguments.demo):
        parser.print_help()
        return
    if arguments.self_test:
        _self_test(device=arguments.device)
    if arguments.demo:
        _demo(device=arguments.device, max_epochs=arguments.epochs, seed=arguments.seed,
              n_samples=arguments.samples)


if __name__ == "__main__":
    main()
   
# """Shortcut-robust temporal jigsaw pretext task for neural population data.

# Companion / replacement for the earlier `puzzle` module. Keeps the tiles-mode
# contract (shared-weight, context-free per-tile encoding, no cross-tile
# convolution, 24-class permutation head) but changes two things that most
# directly explain "good train puzzle accuracy, bad test puzzle accuracy, and
# a HEAVILY puzzle-trained encoder gives worse downstream R^2 than a barely
# trained one":

# 1. Per-tile statistic shortcut. Firing-rate style signals drift slowly
#    (state, adaptation, electrode drift) and are smooth bin-to-bin, so a
#    tile's own mean level is, by itself, a weak but real cue to its rank
#    among the four tiles -- even if nothing about within-tile *dynamics* is
#    learned. This is the neural-data analogue of the "low-level statistics"
#    shortcut in Noroozi & Favaro (2016): they had to explicitly normalize
#    each patch's mean/std before their siamese encoder (their Table 5:
#    removing just that normalization cost ~9 points on the downstream task,
#    almost as much as removing the inter-tile gaps). It plausibly explains
#    your symptoms directly: a level-based shortcut is easy to fit on the
#    *particular* drift trajectory seen during training (-> good train
#    accuracy), does not transported to a held-out trajectory (-> bad test
#    accuracy), and, if the encoder spends its capacity on "what is this
#    tile's baseline" instead of "what is this tile's local geometry", the
#    resulting embedding can be actively worse for behavior decoding than a
#    random encoder's generic features (-> R^2 goes down as puzzle accuracy
#    goes up). `tile_normalize` removes each tile's own per-channel mean (and
#    optionally its std) before encoding, and `augment_*` adds fresh random
#    gain/offset noise on top, only while training, so a residual amplitude
#    correlate cannot be memorized either.

# 2. Selecting a checkpoint by pretext loss/accuracy. Puzzle accuracy is not
#    a proxy for representation quality -- your own experiment is a direct
#    demonstration of that. `fit` accepts an optional
#    `monitor_fn(model) -> float`, evaluated every `monitor_every` epochs;
#    whichever epoch scores best is what `fit` returns (the raw last-epoch
#    weights stay reachable through `last_encoder_state_` / `last_head_state_`
#    for comparison). The intended `monitor_fn` is a closure that embeds a
#    small held-out behavioral set with `model.transform` and fits a quick
#    ridge/linear decoder to get an R^2 -- i.e. select the encoder on the
#    metric you actually care about, not on the pretext task.

# Everything else (window/tile geometry, evaluate_puzzle semantics, fit/
# transform/save/load contract) intentionally mirrors the earlier `puzzle`
# module so the two are easy to compare head-to-head.

# Requires NumPy and PyTorch >= 2.0. Inputs are (time, neurons) float arrays,
# or a list of such arrays (one per trial/recording -- never join trials).

#     from neural_jigsaw import NeuralJigsaw

#     def r2_monitor(model):
#         # X_val, y_val: held-out neural data / behavior, not used in fit()
#         z = model.transform(X_val, pad=False)
#         return quick_ridge_r2(z, y_val)          # your own linear probe

#     model = NeuralJigsaw(window_size=8, tile_gap=(1, 4), tile_normalize="center",
#                           augment_gain_jitter=0.1, augment_noise_std=0.05)
#     model.fit(X_train, monitor_fn=r2_monitor, monitor_every=5)
#     print(model.best_epoch_, model.best_score_)
#     z_valid = model.transform(X_valid, pad=False)
#     metrics = model.evaluate_puzzle(X_valid, min_gap=window_size)  # de-correlated

# Quick real-PyTorch smoke test: python neural_jigsaw.py --self-test
# """
# from copy import deepcopy
# from itertools import permutations
# import math
# import numbers
# from pathlib import Path

# import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.utils.data import DataLoader, Dataset

# # label -> source ranks in ascending destination-slot order (same convention
# # as the earlier `puzzle` module, so labels/checkpoints reason about
# # identically).
# PERMUTATIONS = np.asarray(list(permutations(range(4))), dtype=np.int64)


# # --------------------------------------------------------------------------
# # small shared helpers (kept close to the earlier `puzzle` module)
# # --------------------------------------------------------------------------
# def _integer(name, value, minimum=1):
#     if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
#         raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}.")
#     return int(value)


# def _real(name, value, minimum, maximum=None, strict_min=False):
#     if isinstance(value, bool) or not isinstance(value, numbers.Real):
#         raise ValueError(f"{name} must be a real number.")
#     value = float(value)
#     if (not math.isfinite(value) or value < minimum
#             or (strict_min and value == minimum)
#             or (maximum is not None and value >= maximum)):
#         raise ValueError(f"Invalid {name}: {value}.")
#     return value


# def _sequences(X, min_length, n_features=None):
#     """Accept one (T,C) array, or a list of (T_i,C) arrays, without joining trials."""
#     def array(value):
#         if torch.is_tensor(value):
#             value = value.detach().to(device="cpu", dtype=torch.float32).numpy()
#         return np.asarray(value, dtype=np.float32)

#     is_list = (isinstance(X, (list, tuple)) and len(X) > 0
#                and array(X[0]).ndim == 2)
#     values = list(X) if is_list else [X]
#     result = []
#     for index, value in enumerate(values):
#         x = array(value)
#         if x.ndim != 2 or x.shape[0] < min_length or x.shape[1] < 1:
#             raise ValueError(f"Sequence {index}: expected (T,C), T >= {min_length}, C >= 1; got {x.shape}.")
#         if not np.isfinite(x).all():
#             raise ValueError(f"Sequence {index} contains NaN or infinite values.")
#         if n_features is None:
#             n_features = x.shape[1]
#         if x.shape[1] != n_features:
#             raise ValueError(f"Sequence {index}: expected {n_features} channels, got {x.shape[1]}.")
#         result.append(np.array(x, dtype=np.float32, order="C", copy=True))
#     return result, is_list


# class _Windows(Dataset):
#     def __init__(self, sequences, window_size):
#         self.sequences = sequences
#         self.window_size = window_size
#         self.ends = np.cumsum([len(x) - window_size + 1 for x in sequences])

#     def __len__(self):
#         return int(self.ends[-1])

#     def __getitem__(self, index):
#         seq = int(np.searchsorted(self.ends, index, side="right"))
#         start = int(index - (self.ends[seq - 1] if seq else 0))
#         x = self.sequences[seq][start:start + self.window_size]
#         return torch.from_numpy(x.T.copy())  # (channels, time)

#     def locate(self, index):
#         """index -> (sequence_id, start), for de-correlated evaluation sampling."""
#         seq = int(np.searchsorted(self.ends, index, side="right"))
#         start = int(index - (self.ends[seq - 1] if seq else 0))
#         return seq, start


# class _Residual(nn.Module):
#     def __init__(self, width, dropout):
#         super().__init__()
#         self.net = nn.Sequential(nn.Dropout1d(dropout), nn.Conv1d(width, width, 3), nn.GELU())

#     def forward(self, x):
#         return x[..., 1:-1] + self.net(x)


# class _Encoder(nn.Module):
#     """Valid-convolution residual net; receptive field is exactly window_size."""
#     def __init__(self, channels, window_size, width, output_dimension, dropout, normalize):
#         super().__init__()
#         first_kernel = 2 if window_size % 2 == 0 else 3
#         blocks = (window_size - first_kernel - 2) // 2
#         self.layers = nn.Sequential(
#             nn.Conv1d(channels, width, first_kernel),
#             nn.Dropout1d(dropout), nn.GELU(),
#             *[_Residual(width, dropout) for _ in range(blocks)],
#             nn.Conv1d(width, output_dimension, 3),
#         )
#         self.normalize = normalize

#     def forward(self, x):
#         z = self.layers(x)
#         if z.shape[-1] != 1:
#             raise ValueError("Encoder expects exactly window_size input bins.")
#         z = z.squeeze(-1)
#         return F.normalize(z, p=2, dim=1, eps=1e-8) if self.normalize else z


# # --------------------------------------------------------------------------
# # tile geometry, shortcut-removal normalization, and training-time jitter
# # --------------------------------------------------------------------------
# def _draw_tile_plan(batch, window_size, tile_gap, count, rng):
#     """Four intact, nonoverlapping tiles; gaps/positions NEVER enter the head."""
#     gaps = rng.integers(tile_gap[0], tile_gap[1] + 1, size=(batch, 3))
#     starts = np.concatenate((np.zeros((batch, 1), dtype=np.int64),
#                              np.cumsum(window_size + gaps, axis=1)), axis=1)
#     if count == 24:
#         labels = np.broadcast_to(np.arange(24), (batch, 24)).copy()
#     elif count == 1:
#         labels = rng.integers(0, 24, size=(batch, 1))
#     else:
#         labels = np.argsort(rng.random((batch, 24)), axis=1)[:, :count]
#     return starts, labels.reshape(-1)


# def _extract_tiles(windows, starts, window_size):
#     """(B,C,span) -> (B,4,C,W), with no convolution across tile boundaries."""
#     starts = torch.as_tensor(starts, dtype=torch.long, device=windows.device)
#     offsets = torch.arange(window_size, device=windows.device)
#     indices = (starts[:, :, None] + offsets).reshape(len(windows), 1, -1)
#     tiles = windows.gather(2, indices.expand(-1, windows.shape[1], -1))
#     return tiles.reshape(len(windows), windows.shape[1], 4, window_size).transpose(1, 2)


# def _normalize_tiles(tiles, mode):
#     """Remove each tile's OWN per-channel level (and optionally scale).

#     tiles: (..., C, W). Uses only that tile's W bins -- never statistics
#     from other tiles or from the full recording -- so the label (which
#     tile came first/second/...) cannot be read off from a tile's absolute
#     baseline. "center" (default) subtracts the per-channel mean only, which
#     is the safer choice for small W (a handful of bins gives a very noisy
#     std estimate); "zscore" additionally divides by the per-channel std;
#     "off" reproduces the earlier module's behaviour (no normalization).
#     """
#     if mode == "off":
#         return tiles
#     mean = tiles.mean(dim=-1, keepdim=True)
#     centered = tiles - mean
#     if mode == "center":
#         return centered
#     if mode == "zscore":
#         std = centered.std(dim=-1, unbiased=False, keepdim=True)
#         return centered / (std + 1e-6)
#     raise ValueError("tile_normalize must be 'off', 'center', or 'zscore'.")


# def _augment_tiles(tiles, gain_jitter, noise_std, channel_std, generator):
#     """Training-only jitter: one random gain per tile (all channels, so the
#     within-tile *relative* activity across neurons is preserved) plus
#     independent per-channel additive noise. Both are fresh every visit, so
#     they cannot themselves be memorized as a shortcut; their only effect is
#     to make any residual reliance on absolute amplitude a losing strategy.
#     noise_std is a fraction of each channel's own std in the TRAINING data
#     (channel_std), so one dimensionless number works regardless of firing-
#     rate units.
#     """
#     if gain_jitter > 0:
#         gain = 1.0 + (2 * torch.rand(tiles.shape[:2] + (1, 1), generator=generator,
#                                      device=tiles.device) - 1.0) * gain_jitter
#         tiles = tiles * gain
#     if noise_std > 0:
#         scale = (noise_std * channel_std).reshape(1, 1, -1, 1)
#         tiles = tiles + torch.randn(tiles.shape, generator=generator, device=tiles.device) * scale
#     return tiles


# def _arrange_features(features, labels, count):
#     """Shared-encoder features (B,4,D) -> shuffled (B*count,4*D)."""
#     features = features.repeat_interleave(count, dim=0)
#     target = torch.as_tensor(labels, dtype=torch.long, device=features.device)
#     table = torch.as_tensor(PERMUTATIONS, dtype=torch.long, device=features.device)
#     order = table[target, :, None].expand(-1, -1, features.shape[2])
#     return features.gather(1, order).reshape(len(features), -1), target


# def _min_gap_sample(dataset, n, min_gap, rng):
#     """Pick up to n window starts, keeping picks from the same sequence at
#     least min_gap bins apart, to reduce the pseudo-replication that comes
#     from evaluating on heavily overlapping sliding windows. Falls back to
#     plain sampling (like before) when min_gap is None.
#     """
#     if min_gap is None:
#         return np.sort(rng.choice(len(dataset), size=min(n, len(dataset)), replace=False))
#     order = rng.permutation(len(dataset))
#     accepted, accepted_starts = [], {}
#     for index in order:
#         seq, start = dataset.locate(int(index))
#         starts = accepted_starts.setdefault(seq, [])
#         if all(abs(start - other) >= min_gap for other in starts):
#             starts.append(start)
#             accepted.append(int(index))
#             if len(accepted) == n:
#                 break
#     return np.sort(np.asarray(accepted, dtype=np.int64))


# # --------------------------------------------------------------------------
# # estimator
# # --------------------------------------------------------------------------
# class NeuralJigsaw:
#     """Shared-encoder temporal-tiles jigsaw estimator with checkpoint selection.

#     Architecturally this is the same context-free design as tiles-mode in
#     the earlier module (encoder RF == window_size, 4 independent tiles,
#     24-class head over the concatenated, permuted latents). The additions
#     are `tile_normalize` / `augment_*` (see module docstring, point 1) and
#     `monitor_fn`-based checkpoint selection in `fit` (point 2). No behavior
#     labels enter fit() itself -- monitor_fn is the only place they may.
#     """

#     def __init__(self, window_size=8, output_dimension=64, num_hidden_units=64,
#                  head_hidden_units=128, batch_size=256, max_epochs=200,
#                  learning_rate=3e-4, weight_decay=1e-4, dropout=0.1, normalize=True,
#                  tile_normalize="center", augment_gain_jitter=0.1, augment_noise_std=0.05,
#                  permutations_per_window=1, tile_gap=(1, 4), device="cuda_if_available",
#                  random_state=42, verbose=True):
#         self.window_size = _integer("window_size", window_size, 4)
#         self.output_dimension = _integer("output_dimension", output_dimension)
#         self.num_hidden_units = _integer("num_hidden_units", num_hidden_units)
#         self.head_hidden_units = _integer("head_hidden_units", head_hidden_units)
#         self.batch_size = _integer("batch_size", batch_size)
#         self.max_epochs = _integer("max_epochs", max_epochs, 0)
#         self.learning_rate = _real("learning_rate", learning_rate, 0, strict_min=True)
#         self.weight_decay = _real("weight_decay", weight_decay, 0)
#         self.dropout = _real("dropout", dropout, 0, 1)
#         if not isinstance(normalize, bool):
#             raise ValueError("normalize must be bool (latent L2 normalization).")
#         self.normalize = normalize
#         if tile_normalize not in ("off", "center", "zscore"):
#             raise ValueError("tile_normalize must be 'off', 'center', or 'zscore'.")
#         self.tile_normalize = tile_normalize
#         self.augment_gain_jitter = _real("augment_gain_jitter", augment_gain_jitter, 0, 1)
#         self.augment_noise_std = _real("augment_noise_std", augment_noise_std, 0)
#         self.permutations_per_window = _integer("permutations_per_window", permutations_per_window)
#         if self.permutations_per_window > 24:
#             raise ValueError("permutations_per_window must be between 1 and 24.")
#         gaps = tile_gap if isinstance(tile_gap, (tuple, list)) else (tile_gap, tile_gap)
#         if len(gaps) != 2:
#             raise ValueError("tile_gap must be an integer or (minimum, maximum) pair.")
#         self.tile_gap = tuple(_integer("tile_gap", g, 0) for g in gaps)
#         if self.tile_gap[0] > self.tile_gap[1]:
#             raise ValueError("tile_gap minimum exceeds maximum.")
#         self.device = str(device)
#         self.random_state = _integer("random_state", random_state, 0)
#         self.verbose = bool(verbose)

#     @property
#     def training_span(self):
#         return 4 * self.window_size + 3 * self.tile_gap[1]

#     def get_params(self):
#         names = ("window_size", "output_dimension", "num_hidden_units", "head_hidden_units",
#                  "batch_size", "max_epochs", "learning_rate", "weight_decay", "dropout",
#                  "normalize", "tile_normalize", "augment_gain_jitter", "augment_noise_std",
#                  "permutations_per_window", "tile_gap", "device", "random_state", "verbose")
#         return {name: getattr(self, name) for name in names}

#     def _build(self, channels):
#         name = self.device
#         if name == "cuda_if_available":
#             name = "cuda" if torch.cuda.is_available() else "cpu"
#         self.device_ = torch.device(name)
#         self.n_features_in_ = int(channels)
#         self.encoder_ = _Encoder(channels, self.window_size, self.num_hidden_units,
#                                  self.output_dimension, self.dropout, self.normalize).to(self.device_)
#         self.head_ = nn.Sequential(
#             nn.Linear(4 * self.output_dimension, self.head_hidden_units),
#             nn.GELU(), nn.Linear(self.head_hidden_units, 24),
#         ).to(self.device_)

#     def _tile_logits(self, windows, starts, labels, count, *, training):
#         tiles = _extract_tiles(windows, starts, self.window_size)
#         tiles = _normalize_tiles(tiles, self.tile_normalize)
#         if training:
#             tiles = _augment_tiles(tiles, self.augment_gain_jitter, self.augment_noise_std,
#                                    self._channel_std_, self._augment_generator_)
#         z = self.encoder_(tiles.reshape(-1, self.n_features_in_, self.window_size))
#         z = z.reshape(len(windows), 4, self.output_dimension)
#         features, target = _arrange_features(z, labels, count)
#         return self.head_(features), target

#     def fit(self, X, *, monitor_fn=None, monitor_every=1, select_best=True):
#         """Fit on X. If monitor_fn is given, it is called as monitor_fn(self)
#         every `monitor_every` epochs (higher is assumed better -- e.g. a
#         downstream R^2). The BEST-scoring epoch's weights are what the
#         estimator holds after fit() returns (when select_best=True); the
#         final epoch's weights remain available via last_encoder_state_ /
#         last_head_state_ regardless, so you can compare the two directly.
#         """
#         sequences, _ = _sequences(X, self.training_span)
#         self.is_fitted_ = False
#         torch.manual_seed(self.random_state)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(self.random_state)
#         rng = np.random.default_rng(self.random_state)
#         generator = torch.Generator().manual_seed(self.random_state)
#         # self._augment_generator_ = torch.Generator().manual_seed(self.random_state + 1)
#         self._augment_generator_ = None
#         self._build(sequences[0].shape[1])
#         self._augment_generator_ = torch.Generator(device=self.device_).manual_seed(self.random_state + 1)
       
#         # Per-channel std of the TRAINING data only, used to scale augmentation
#         # noise in physically meaningful (relative) units; never touches test data.
#         concatenated = np.concatenate(sequences, axis=0)
#         self._channel_std_ = torch.as_tensor(concatenated.std(axis=0), dtype=torch.float32,
#                                              device=self.device_).clamp_min(1e-6)
#         dataset = _Windows(sequences, self.training_span)
#         self.n_windows_ = len(dataset)
#         loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
#                             drop_last=False, num_workers=0, generator=generator,
#                             pin_memory=self.device_.type == "cuda")
#         parameters = list(self.encoder_.parameters()) + list(self.head_.parameters())
#         optimizer = torch.optim.Adam(parameters, lr=self.learning_rate,
#                                       weight_decay=self.weight_decay)
#         self.history_ = []
#         self.n_steps_ = 0
#         self.best_score_, self.best_epoch_ = -math.inf, None
#         self.best_encoder_state_ = self.best_head_state_ = None
#         if self.verbose:
#             print(f"NeuralJigsaw: {self.n_windows_} valid starts, window={self.window_size}, "
#                   f"training_span={self.training_span}, tile_gap={self.tile_gap}, "
#                   f"tile_normalize={self.tile_normalize}, device={self.device_}", flush=True)
#         for epoch in range(self.max_epochs):
#             self.encoder_.train()
#             self.head_.train()
#             total_loss, correct, seen = 0.0, 0, 0
#             for windows in loader:
#                 windows = windows.to(self.device_, non_blocking=True)
#                 optimizer.zero_grad(set_to_none=True)
#                 starts, labels = _draw_tile_plan(len(windows), self.window_size,
#                                                  self.tile_gap, self.permutations_per_window, rng)
#                 logits, target = self._tile_logits(windows, starts, labels,
#                                                    self.permutations_per_window, training=True)
#                 loss = F.cross_entropy(logits, target)
#                 if not torch.isfinite(loss).item():
#                     raise FloatingPointError(f"Nonfinite puzzle loss at step {self.n_steps_}.")
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item() * len(target)
#                 correct += (logits.detach().argmax(dim=1) == target).sum().item()
#                 seen += len(target)
#                 self.n_steps_ += 1
#             self.encoder_.eval()
#             self.head_.eval()
#             self.is_fitted_ = True  # transform()/monitor_fn may run mid-fit from here on
#             row = dict(epoch=epoch + 1, loss=total_loss / seen, accuracy=correct / seen,
#                        n_puzzles=seen, steps=self.n_steps_, monitor_score=None)
#             do_monitor = monitor_fn is not None and ((epoch + 1) % monitor_every == 0
#                                                       or epoch + 1 == self.max_epochs)
#             if do_monitor:
#                 score = float(monitor_fn(self))
#                 row["monitor_score"] = score
#                 if score > self.best_score_:
#                     self.best_score_, self.best_epoch_ = score, epoch + 1
#                     self.best_encoder_state_ = deepcopy(self.encoder_.state_dict())
#                     self.best_head_state_ = deepcopy(self.head_.state_dict())
#             self.history_.append(row)
#             if self.verbose:
#                 suffix = f", monitor={row['monitor_score']:.4f}" if do_monitor else ""
#                 print(f"Epoch {epoch + 1}/{self.max_epochs}: loss={row['loss']:.5f}, "
#                       f"permutation accuracy={100 * row['accuracy']:.2f}%{suffix}", flush=True)
#         self.last_encoder_state_ = deepcopy(self.encoder_.state_dict())
#         self.last_head_state_ = deepcopy(self.head_.state_dict())
#         if select_best and self.best_encoder_state_ is not None:
#             self.encoder_.load_state_dict(self.best_encoder_state_)
#             self.head_.load_state_dict(self.best_head_state_)
#             if self.verbose:
#                 print(f"Restored best checkpoint: epoch {self.best_epoch_}, "
#                       f"monitor={self.best_score_:.4f}", flush=True)
#         self.is_fitted_ = True
#         return self

#     def use_checkpoint(self, which):
#         """Switch the live weights between 'best' and 'last' after fit()."""
#         self._check_fitted()
#         if which == "best":
#             if self.best_encoder_state_ is None:
#                 raise RuntimeError("No monitor_fn was used during fit(); there is no 'best' checkpoint.")
#             self.encoder_.load_state_dict(self.best_encoder_state_)
#             self.head_.load_state_dict(self.best_head_state_)
#         elif which == "last":
#             self.encoder_.load_state_dict(self.last_encoder_state_)
#             self.head_.load_state_dict(self.last_head_state_)
#         else:
#             raise ValueError("which must be 'best' or 'last'.")
#         return self

#     def _check_fitted(self):
#         if not getattr(self, "is_fitted_", False):
#             raise RuntimeError("Call fit(X_train) before transform/evaluate_puzzle/save.")

#     def evaluate_puzzle(self, X, *, max_windows=1024, permutations_per_window=24,
#                         batch_size=128, random_state=200042, min_gap=None, return_details=False):
#         """Fixed-weight CE/accuracy. Pass min_gap (e.g. window_size, or the
#         full training_span for a stricter test) to keep sampled windows at
#         least that many bins apart within a sequence -- overlapping windows
#         share almost all of their underlying bins, so without this the
#         reported accuracy is optimistic about how independent the "test"
#         observations really are.
#         """
#         self._check_fitted()
#         maximum = _integer("max_windows", max_windows)
#         size = _integer("batch_size", batch_size)
#         count = _integer("permutations_per_window", permutations_per_window)
#         if count > 24:
#             raise ValueError("permutations_per_window must not exceed 24.")
#         seed = _integer("random_state", random_state, 0)
#         sequences, _ = _sequences(X, self.training_span, self.n_features_in_)
#         dataset = _Windows(sequences, self.training_span)
#         rng = np.random.default_rng(seed)
#         chosen = _min_gap_sample(dataset, maximum, min_gap, rng)
#         n = len(chosen)
#         starts, labels = _draw_tile_plan(n, self.window_size, self.tile_gap, count, rng)
#         predictions = np.empty(n * count, dtype=np.int64)
#         losses = np.empty(n * count, dtype=np.float32)
#         modes = self.encoder_.training, self.head_.training
#         self.encoder_.eval()
#         self.head_.eval()
#         try:
#             with torch.inference_mode():
#                 for begin in range(0, n, size):
#                     end = min(begin + size, n)
#                     windows = torch.stack([dataset[int(i)] for i in chosen[begin:end]]).to(self.device_)
#                     label_batch = labels[begin * count:end * count]
#                     logits, target = self._tile_logits(windows, starts[begin:end], label_batch,
#                                                        count, training=False)
#                     if logits.shape != ((end - begin) * count, 24) or not torch.isfinite(logits).all().item():
#                         raise FloatingPointError("Invalid puzzle evaluation logits.")
#                     losses[begin * count:end * count] = F.cross_entropy(logits, target, reduction="none").cpu().numpy()
#                     predictions[begin * count:end * count] = logits.argmax(dim=1).cpu().numpy()
#         finally:
#             self.encoder_.train(modes[0])
#             self.head_.train(modes[1])
#         correct = labels == predictions
#         metrics = dict(
#             window_size=self.window_size, training_span=self.training_span,
#             tile_gap=list(self.tile_gap), tile_normalize=self.tile_normalize,
#             accuracy=float(correct.mean()), accuracy_percent=float(100 * correct.mean()),
#             cross_entropy=float(losses.mean(dtype=np.float64)), chance_accuracy_percent=100 / 24,
#             n_windows=n, available_starts=len(dataset), n_puzzles=len(labels),
#             permutations_per_window=count, random_state=seed, min_gap=min_gap,
#             sampling=("Distinct starts, no minimum spacing enforced (pseudo-replication likely)."
#                      if min_gap is None else
#                      f"Starts within a sequence kept >= {min_gap} bins apart."),
#         )
#         if not return_details:
#             return metrics
#         sequence_ids = np.searchsorted(dataset.ends, chosen, side="right")
#         beginnings = np.r_[0, dataset.ends[:-1]]
#         window_starts = chosen - beginnings[sequence_ids]
#         details = dict(sequence_ids=sequence_ids, window_starts=window_starts,
#                        true_class=labels.reshape(n, count), predicted_class=predictions.reshape(n, count),
#                        cross_entropy=losses.reshape(n, count))
#         return metrics, details

#     def transform(self, X, *, pad=True, batch_size=None, return_indices=False):
#         """Chronological-window embeddings; NEVER runs the puzzle head. Each
#         window is put through the SAME tile_normalize step used in training
#         (that is the point: the encoder was trained to be blind to a
#         window's absolute level, so it must be queried the same way, or
#         train/inference inputs would come from different distributions).
#         """
#         self._check_fitted()
#         if not isinstance(pad, bool):
#             raise ValueError("pad must be bool.")
#         size = self.batch_size if batch_size is None else _integer("batch_size", batch_size)
#         sequences, is_list = _sequences(X, 1 if pad else self.window_size, self.n_features_in_)
#         left = self.window_size // 2
#         right = self.window_size - left - 1
#         result, indices = [], []
#         was_training = self.encoder_.training
#         self.encoder_.eval()
#         try:
#             with torch.inference_mode():
#                 for x in sequences:
#                     if pad:
#                         centers = np.arange(len(x), dtype=np.int64)
#                         x = np.pad(x, ((left, right), (0, 0)), mode="edge")
#                     else:
#                         centers = np.arange(left, len(x) - right, dtype=np.int64)
#                     embeddings = np.empty((len(centers), self.output_dimension), dtype=np.float32)
#                     for start in range(0, len(centers), size):
#                         end = min(start + size, len(centers))
#                         locations = np.arange(start, end)[:, None] + np.arange(self.window_size)
#                         windows = np.ascontiguousarray(x[locations].transpose(0, 2, 1))
#                         tensor = torch.from_numpy(windows).to(self.device_)
#                         tensor = _normalize_tiles(tensor, self.tile_normalize)
#                         embeddings[start:end] = self.encoder_(tensor).cpu().numpy()
#                     result.append(embeddings)
#                     indices.append(centers)
#         finally:
#             self.encoder_.train(was_training)
#         values = result if is_list else result[0]
#         times = indices if is_list else indices[0]
#         return (values, times) if return_indices else values

#     def fit_transform(self, X, **kwargs):
#         transform_kwargs = {k: v for k, v in kwargs.items()
#                             if k in ("pad", "batch_size", "return_indices")}
#         fit_kwargs = {k: v for k, v in kwargs.items() if k not in transform_kwargs}
#         return self.fit(X, **fit_kwargs).transform(X, **transform_kwargs)

#     def save(self, path):
#         self._check_fitted()
#         payload = dict(
#             format_version=1, model_type="neural_jigsaw", params=self.get_params(),
#             n_features=self.n_features_in_,
#             encoder={k: v.detach().cpu() for k, v in self.encoder_.state_dict().items()},
#             head={k: v.detach().cpu() for k, v in self.head_.state_dict().items()},
#             last_encoder=self.last_encoder_state_, last_head=self.last_head_state_,
#             best_epoch=self.best_epoch_, best_score=self.best_score_,
#             history=self.history_, n_steps=self.n_steps_, n_windows=self.n_windows_,
#         )
#         torch.save(payload, Path(path))

#     @classmethod
#     def load(cls, path, device="cuda_if_available"):
#         data = torch.load(Path(path), map_location="cpu", weights_only=True)
#         if data.get("model_type") != "neural_jigsaw":
#             raise ValueError("Not a NeuralJigsaw checkpoint.")
#         model = cls(**dict(data["params"], device=device))
#         model._build(data["n_features"])
#         model.encoder_.load_state_dict(data["encoder"])
#         model.head_.load_state_dict(data["head"])
#         model.last_encoder_state_ = data["last_encoder"]
#         model.last_head_state_ = data["last_head"]
#         model.best_epoch_, model.best_score_ = data["best_epoch"], data["best_score"]
#         model.best_encoder_state_ = model.best_head_state_ = None
#         model.history_, model.n_steps_, model.n_windows_ = data["history"], data["n_steps"], data["n_windows"]
#         model.encoder_.eval()
#         model.head_.eval()
#         model.is_fitted_ = True
#         return model


# def _self_test():
#     """Small real-PyTorch CPU test: python neural_jigsaw.py --self-test."""
#     import tempfile
#     previous_threads = torch.get_num_threads()
#     torch.set_num_threads(1)
#     try:
#         # 1) Per-tile normalization: zeroes each tile's own per-channel mean,
#         #    and does so using ONLY that tile's bins (verified against a
#         #    hand-computed reference), never other tiles' statistics.
#         tiles = torch.randn(3, 4, 5, 6) * 10 + 100  # (B,4,C,W), large offset
#         centered = _normalize_tiles(tiles, "center")
#         torch.testing.assert_close(centered.mean(dim=-1), torch.zeros(3, 4, 5), atol=1e-4, rtol=0)
#         reference = tiles[0, 1] - tiles[0, 1].mean(dim=-1, keepdim=True)
#         torch.testing.assert_close(centered[0, 1], reference)
#         zscored = _normalize_tiles(tiles, "zscore")
#         torch.testing.assert_close(zscored.std(dim=-1, unbiased=False), torch.ones(3, 4, 5), atol=1e-3, rtol=0)
#         torch.testing.assert_close(_normalize_tiles(tiles, "off"), tiles)

#         # 2) A tile's absolute level is destroyed by design: two tiles that
#         #    differ only by a constant offset are identical after centering.
#         shifted = tiles.clone()
#         shifted[0, 2] = tiles[0, 2] + 37.0
#         torch.testing.assert_close(_normalize_tiles(tiles, "center")[0, 2],
#                                    _normalize_tiles(shifted, "center")[0, 2], atol=1e-4, rtol=1e-4)

#         # 3) Augmentation changes values but keeps shape/finiteness, and a
#         #    zero-jitter/zero-noise call is a no-op.
#         gen = torch.Generator().manual_seed(0)
#         channel_std = torch.ones(5)
#         augmented = _augment_tiles(tiles, 0.2, 0.1, channel_std, gen)
#         assert augmented.shape == tiles.shape and torch.isfinite(augmented).all()
#         assert not torch.equal(augmented, tiles)
#         torch.testing.assert_close(_augment_tiles(tiles, 0.0, 0.0, channel_std, gen), tiles)

#         # 4) Tiny real fit + monitor-based checkpoint selection on synthetic
#         #    data with an injected pure drift shortcut. A model that ignores
#         #    the shortcut (tile_normalize="center") should not be able to
#         #    solve the puzzle purely from tile means; a raw model given the
#         #    same drift with normalization off can still see the mean cue.
#         rng = np.random.default_rng(3)
#         T, C = 400, 6
#         drift = np.linspace(0, 5, T)[:, None]
#         trials = [(rng.normal(scale=0.3, size=(T, C)) + drift).astype(np.float32)]

#         scores = iter([0.1, 0.2, 0.15, 0.4, 0.3])  # deliberately non-monotonic
#         seen_scores = []

#         def fake_monitor(model):
#             value = next(scores)
#             seen_scores.append(value)
#             return value

#         params = dict(window_size=4, output_dimension=6, num_hidden_units=8,
#                       head_hidden_units=16, batch_size=16, device="cpu", verbose=False,
#                       tile_gap=(1, 3), random_state=11, augment_gain_jitter=0.0,
#                       augment_noise_std=0.0)
#         model = NeuralJigsaw(**params, tile_normalize="center", max_epochs=5).fit(
#             trials, monitor_fn=fake_monitor, monitor_every=1)
#         assert model.history_[-1]["steps"] == model.n_steps_ and model.n_steps_ > 0
#         assert seen_scores == [0.1, 0.2, 0.15, 0.4, 0.3]
#         assert model.best_epoch_ == 4 and abs(model.best_score_ - 0.4) < 1e-9
#         assert any(not torch.equal(a, b) for a, b in
#                    zip(model.encoder_.state_dict().values(), model.last_encoder_state_.values()))
#         model.use_checkpoint("last")
#         assert all(torch.equal(a, b) for a, b in
#                    zip(model.encoder_.state_dict().values(), model.last_encoder_state_.values()))
#         model.use_checkpoint("best")
#         assert all(torch.equal(a, b) for a, b in
#                    zip(model.encoder_.state_dict().values(), model.best_encoder_state_.values()))

#         # 5) transform shapes/alignment, and evaluate_puzzle with a min_gap
#         #    actually enforces spacing between the sampled windows.
#         padded = model.transform(trials, pad=True)
#         valid, indices = model.transform(trials, pad=False, return_indices=True)
#         assert padded[0].shape == (T, params["output_dimension"])
#         assert valid[0].shape == (T - model.window_size + 1, params["output_dimension"])
#         np.testing.assert_array_equal(indices[0], np.arange(len(valid[0])) + model.window_size // 2)

#         metrics, details = model.evaluate_puzzle(trials, max_windows=8, min_gap=20,
#                                                  permutations_per_window=4, return_details=True)
#         starts = np.sort(details["window_starts"])
#         assert np.all(np.diff(starts) >= 20)
#         assert metrics["accuracy"] >= 0.0 and math.isfinite(metrics["cross_entropy"])

#         # 6) save/load round-trips weights, history and best/last bookkeeping.
#         with tempfile.TemporaryDirectory() as tmp:
#             path = Path(tmp) / "neural_jigsaw.pt"
#             model.save(path)
#             loaded = NeuralJigsaw.load(path, device="cpu")
#             np.testing.assert_allclose(loaded.transform(trials[0]), padded[0], rtol=1e-5, atol=1e-6)
#             assert loaded.best_epoch_ == model.best_epoch_
#             assert loaded.get_params() == model.get_params()

#         print("PASS: tile normalization removes absolute level (incl. against a constant "
#               "offset), augmentation is a real but shape-preserving perturbation, monitor-"
#               "based best-checkpoint selection picks the right non-final epoch and is "
#               "restorable, transform alignment, min_gap de-correlated evaluation, save/load.")
#     finally:
#         torch.set_num_threads(previous_threads)


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("--self-test", action="store_true", help="Run the small CPU test.")
#     args = parser.parse_args()
#     if args.self_test:
#         _self_test()
#     else:
#         parser.print_help()

"""JigsawNet: a self-supervised neural encoder trained ONLY with cross-entropy.

No contrastive learning. No InfoNCE. No positive/negative pairs. No temperature.
No dependency on, and no objective borrowed from, time-contrastive embedding
methods. Every loss in this file is torch cross_entropy or
binary_cross_entropy_with_logits on a task with a hard, known-correct label.

    from jigsaw_net import JigsawNet
    model = JigsawNet(window_size=10, n_tiles=4, max_epochs=60)
    model.fit(X_train, X_valid=X_valid)
    Z = model.transform(X_valid)                    # decode THIS
    print(model.evaluate_pretext(X_valid))
    print(model.evaluate_decoding(X_train, Y_train, X_valid, Y_valid))

THE THREE TASKS (all cross-entropy)
-----------------------------------
1. ORDER (the jigsaw). K intact, chronological, nonoverlapping tiles are cut
   from one span. Two readouts share a permutation-equivariant DeepSets token:
     - position CE : K-way softmax per tile, "which slot in time is this?".
       Chance 1/K. Scales to any K, unlike a K!-way label which explodes and is
       memorizable (24 classes is a lookup table; 4 classes x 4 tiles is not).
     - pair BCE    : for every ordered pair, "did i come before j?". Chance 50%,
       which is a far higher-powered test than 1/24 = 4.17%.
   The position head is DECODED WITH AN ASSIGNMENT, not an argmax -- see below.
2. RECONSTRUCT (the anti-collapse anchor, ON by default). From each tile's
   embedding, predict the quantized activity of EVERY bin of that same tile.
   The input is augmented and the target is clean, so it is a denoising task.
3. FORECAST (OFF by default, lambda_forecast=0). From each tile's embedding,
   predict the QUANTIZED activity of the bin immediately AFTER that tile, per
   neuron. The target bin is always inside the discarded gap, so it never leaks.

WHY THE DEFAULT IS ORDER + RECONSTRUCT, WITH FORECAST OFF. Task 2 was added
because a pure order task gives the encoder no reason to keep absolute firing
rate, and rate is usually the dominant behavior-coding feature -- that is how
pretext training ended up BELOW a frozen random encoder. Reconstruction fixed
that decisively: on Perich C-CO0 (two seeds) order+reconstruct reached MLP R2
0.848 / 0.835 against a frozen-encoder control at 0.573 / 0.571, and the order
term itself contributed +0.057 / +0.040 on top of reconstruction alone.

Task 3 was the FIRST attempt at the same anchoring job and it is strictly worse:
adding the forecast term to the winning model cost -0.132 / -0.139 R2 on both
seeds. One predicted bin is a weak constraint, and predicting across a random
gap makes the encoder model the gap process rather than the window. It is kept
as an ablation (lambda_forecast > 0) because "predict the next bin" is the
obvious thing a reader will ask about, and the answer should be measured.

READING THE ORDER HEAD. argmax over the K-way position logits is NOT a
permutation: it can put two tiles in the same slot, and an UNTRAINED head puts
all of them in one slot because the class biases swamp the tile-dependent part
of the logits. Scoring those ties as errors moves the chance level of pair
accuracy from 50% to (1 - 1/K)/2 = 37.5% at K=4, and to 0% for the degenerate
case -- so a frozen random encoder scores 0.00% and looks catastrophically
anti-correlated when it is simply undefined. This module therefore decodes with
the best-scoring permutation (`_assign`), gives ties half credit, and reports
`argmax_tie_rate_percent` so a degenerate readout is visible.

NO IDENTITY LABELS. The tiles are handed to the order head in a RANDOM order and
the label is that permutation, never [0,1,...,K-1] (shuffle_tiles=True). With a
constant identity label the head only has to be order-dependent -- through a
non-symmetric aggregation, a positional term, anything -- and "always answer
[0,1,2,3]" becomes a free 100% that no metric in this file would catch. The
DeepSets head is permutation-equivariant by construction, but that is a property
asserted in a self-test; shuffling removes the whole failure class for the price
of one gather. Set shuffle_tiles=False only to reproduce older runs.

NO L2 NORMALIZATION BY DEFAULT. normalize=True projects the embedding onto a
hypersphere and throws away magnitude. If ridge on the raw window beats your
embedding, this is the first thing to check. Left available as an ablation only.

EPOCHS. Spans overlap by construction, so the effective sample size is far below
the span count and a large budget WILL memorize the pretext task: held-out
position CE ends up many times log(K). Do not read that as "stop training". On
Perich the best decoding R2 came from the largest budget tried (10000 epochs)
with the pretext task thoroughly overfit -- the order head memorizing says
nothing about whether the trunk learned useful features. Sweep max_epochs and
pick the peak by DECODING R2, never by pretext accuracy or pretext CE.

SHORTCUTS. Tile order is recoverable from slow drift in overall level, so the
order head can become a nonstationarity detector that does not transfer. The
order branch therefore sees per-tile normalized tiles (tile_norm) plus
independent per-tile augmentation and random gaps. The forecast branch sees the
RAW view, because it is supposed to see level. evaluate_pretext always reports
the trivial "sort by mean activity" baseline next to the model.

TRUNK. trunk_block="residual" is a valid-convolution residual stack with
receptive field exactly window_size. trunk_block="separable" is a
MobileNetV2-style inverted residual (1x1 expand, GELU, DEPTHWISE 3-tap temporal
conv, GELU, 1x1 project): each neuron gets its own temporal filter and mixing
across neurons happens only in the pointwise layers, the decomposition EEGNet
uses. It is not a shortcut fix -- a linearly available shortcut stays available
-- but it is cheap and a fair ablation. Both blocks consume exactly two bins.

CONTROLS (all from the constructor)
  max_epochs=0        -> frozen random-encoder control. Your embedding must beat
                         this or training is destroying information.
  lambda_order=0.0    -> reconstruct only (the anchor's own contribution)
  lambda_forecast>0   -> add the next-bin term back (measured harmful, see above)
  lambda_reconstruct=0 -> the pre-anchor model (reproduces the collapse)
  order_grad_scale=0  -> order head trains on a trunk it cannot influence; asks
                         "is order decodable?" with zero risk to R2
  shuffle_tiles=False -> identity labels, for reproducing older runs only
  normalize=True      -> the sphere ablation
Always report pretext accuracy AND the mean-sort baseline AND decoding R2 AND
ridge on the raw window. Pretext accuracy alone proves nothing.

ALIGNMENT. transform uses natural windows, no augmentation, no tile
normalization, no masking. pad=False aligns the window starting at j with label
j + window_size//2. pad=True edge-pads and returns one row per input bin.

Split recordings BEFORE fit: a 2D array is one continuous recording; pass a list
of trial arrays to avoid crossing trials. fit always resets weights.

Self-test:  python jigsaw_net.py --self-test
Demo:       python jigsaw_net.py --demo
"""
import argparse
import copy
import itertools
import math
import numbers
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

_EPS = 1e-8
TILE_NORMS = ("none", "mean", "zscore", "global_mean")
TRUNK_BLOCKS = ("residual", "separable")


# --------------------------------------------------------------------------- #
# validation
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
    if (not math.isfinite(value) or value < minimum or (strict_min and value == minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"Invalid {name}: {value}.")
    return value


def _boolean(name, value):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool; got {value!r}.")
    return value


def _sequences(X, minimum_length, n_features=None):
    is_list = isinstance(X, (list, tuple))
    items = list(X) if is_list else [X]
    if not items:
        raise ValueError("X is empty.")
    out = []
    for i, item in enumerate(items):
        array = np.ascontiguousarray(np.asarray(item, dtype=np.float32))
        if array.ndim != 2:
            raise ValueError(f"Sequence {i} must be 2D (time, neurons); got {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError(f"Sequence {i} contains NaN or Inf.")
        if len(array) < minimum_length:
            raise ValueError(f"Sequence {i} has {len(array)} bins; needs >= {minimum_length}.")
        if n_features is not None and array.shape[1] != n_features:
            raise ValueError(f"Sequence {i} has {array.shape[1]} neurons; expected {n_features}.")
        out.append(array)
    if n_features is None and len({a.shape[1] for a in out}) != 1:
        raise ValueError("All sequences must have the same number of neurons.")
    return out, is_list


# --------------------------------------------------------------------------- #
# modules
# --------------------------------------------------------------------------- #
class _GradScale(torch.autograd.Function):
    """Straight-through forward, scaled gradient backward."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class _Residual(nn.Module):
    """Valid 3-tap residual block; consumes exactly two bins."""

    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(width, width, 3), nn.GELU(),
            nn.Conv1d(width, width, 1),
        )

    def forward(self, x):
        return x[..., 1:-1] + self.net(x)


class _Separable(nn.Module):
    """MobileNetV2-style inverted residual with a VALID depthwise 3-tap conv.

    1x1 expand -> GELU -> depthwise temporal conv -> GELU -> 1x1 linear project.
    Consumes exactly two bins, like _Residual, so trunk arithmetic is unchanged.
    """

    def __init__(self, width, dropout, expansion=4):
        super().__init__()
        hidden = width * expansion
        self.net = nn.Sequential(
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(width, hidden, 1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, groups=hidden), nn.GELU(),
            nn.Conv1d(hidden, width, 1),
        )

    def forward(self, x):
        return x[..., 1:-1] + self.net(x)


class _Trunk(nn.Module):
    """Valid-convolution stack whose receptive field is exactly window_size."""

    def __init__(self, channels, window_size, width, dropout, block="residual"):
        super().__init__()
        if block not in TRUNK_BLOCKS:
            raise ValueError(f"trunk_block must be one of {TRUNK_BLOCKS}.")
        first_kernel = 2 if window_size % 2 == 0 else 3
        blocks = (window_size - first_kernel - 2) // 2
        make = _Residual if block == "residual" else _Separable
        self.layers = nn.Sequential(
            nn.Conv1d(channels, width, first_kernel),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(), nn.GELU(),
            *[make(width, dropout) for _ in range(blocks)],
            nn.Conv1d(width, width, 3), nn.GELU(),
        )

    def forward(self, x):
        h = self.layers(x)
        if h.shape[-1] != 1:
            raise ValueError("Trunk expects exactly window_size input bins.")
        return h.squeeze(-1)


class _Encoder(nn.Module):
    def __init__(self, channels, window_size, width, output_dimension, dropout,
                 normalize, trunk_block):
        super().__init__()
        self.trunk = _Trunk(channels, window_size, width, dropout, trunk_block)
        self.project = nn.Linear(width, output_dimension)
        self.normalize = normalize

    def forward(self, x):
        z = self.project(self.trunk(x))
        return F.normalize(z, dim=-1) if self.normalize else z


class _OrderHead(nn.Module):
    """Permutation-equivariant DeepSets head. Two cross-entropy readouts.

    position : K-way logits per tile, "which time slot is this tile?"
    pair     : one scalar per tile; their difference is the before/after logit.
    """

    def __init__(self, dimension, hidden, n_tiles):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2 * dimension, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU())
        self.position = nn.Linear(hidden, n_tiles)
        self.score = nn.Linear(hidden, 1)

    def forward(self, z):
        context = z.mean(dim=1, keepdim=True).expand_as(z)
        tokens = self.body(torch.cat((z, context), dim=-1))
        return self.position(tokens), self.score(tokens).squeeze(-1)


class _ForecastHead(nn.Module):
    """Per-neuron cross-entropy over quantized activity of the next bin."""

    def __init__(self, dimension, hidden, channels, levels):
        super().__init__()
        self.channels, self.levels = channels, levels
        self.net = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
                                 nn.Linear(hidden, channels * levels))

    def forward(self, z):
        return self.net(z).reshape(-1, self.channels, self.levels)


class _ReconstructHead(nn.Module):
    """Per-neuron, per-bin cross-entropy over the quantized tile the embedding came from.

    The anti-collapse anchor. The order task only needs a low-dimensional
    "where in time am I" code and will happily throw everything else away --
    measured as a participation ratio of 3.4 out of 64 and ridge R2 of 0.005.
    Forecasting one bin is too weak a constraint to stop that. Requiring the
    embedding to reproduce EVERY bin of its own window forces it to stay an
    information-preserving bottleneck, and the raw window is exactly what the
    decoding ceiling is computed from, so this term targets the measured gap
    rather than a proxy for it. Still nothing but cross-entropy.
    """

    def __init__(self, dimension, hidden, channels, window_size, levels):
        super().__init__()
        self.channels, self.window_size, self.levels = channels, window_size, levels
        self.net = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
                                 nn.Linear(hidden, channels * window_size * levels))

    def forward(self, z):
        return self.net(z).reshape(-1, self.channels, self.window_size, self.levels)


# --------------------------------------------------------------------------- #
# functional pieces
# --------------------------------------------------------------------------- #
def _gather_tiles(data, starts, gaps, window_size):
    """(B,K,N,W) tiles plus the index of the bin right after each tile."""
    offsets = torch.cat((torch.zeros_like(gaps[:, :1]),
                         torch.cumsum(gaps + window_size, dim=1)), dim=1)
    tile_starts = starts[:, None] + offsets
    index = tile_starts[:, :, None] + torch.arange(window_size, device=data.device)
    return data[index].permute(0, 1, 3, 2), tile_starts + window_size


def _augment(tiles, neuron_dropout, gain_jitter, generator):
    if neuron_dropout > 0:
        keep = (torch.rand(tiles.shape[:3], device=tiles.device, generator=generator)
                >= neuron_dropout).to(tiles.dtype)
        tiles = tiles * keep[..., None]
    if gain_jitter > 0:
        gain = 1.0 + gain_jitter * (2 * torch.rand(tiles.shape[:3], device=tiles.device,
                                                   generator=generator) - 1)
        tiles = tiles * gain[..., None]
    return tiles


def _tile_normalize(tiles, mode):
    if mode == "none":
        return tiles
    if mode == "mean":
        return tiles - tiles.mean(dim=3, keepdim=True)
    if mode == "global_mean":
        return tiles - tiles.mean(dim=(2, 3), keepdim=True)
    if mode == "zscore":
        return (tiles - tiles.mean(dim=3, keepdim=True)) / (tiles.std(dim=3, keepdim=True) + 1e-5)
    raise ValueError(f"tile_norm must be one of {TILE_NORMS}.")


def _pair_targets(positions):
    """Upper-triangular before/after labels and the mask of valid pairs."""
    difference = positions[:, :, None] - positions[:, None, :]
    mask = torch.triu(torch.ones_like(difference, dtype=torch.bool), diagonal=1)
    return (difference < 0).to(torch.float32), mask


def _order_losses(position_logits, scores, positions):
    """Both readouts, pure cross-entropy."""
    n_tiles = position_logits.shape[1]
    position_loss = F.cross_entropy(position_logits.reshape(-1, n_tiles), positions.reshape(-1))
    target, mask = _pair_targets(positions)
    difference = scores[:, :, None] - scores[:, None, :]
    pair_loss = F.binary_cross_entropy_with_logits(difference[mask], target[mask])
    return position_loss, pair_loss


def _ranks(values):
    return values.argsort(dim=1).argsort(dim=1)


def _shuffle_tiles(tiles, enabled, generator):
    """Present the tiles in a RANDOM order and return the matching labels.

    Until now the tiles were always handed to the head in chronological order
    and the label was always the identity permutation. That is only safe if the
    head is perfectly permutation-equivariant: the instant any order-dependence
    leaks in -- an aggregation that is not symmetric, a positional term, a
    non-deterministic reduction -- "always answer [0,1,...,K-1]" becomes a free
    100%, and nothing in the metric would say so. Shuffling removes the whole
    failure class instead of relying on a property that is asserted once in a
    self-test, and it costs one gather.

    Returns (presented tiles, labels) where labels[b, j] is the CHRONOLOGICAL
    index of whichever tile is sitting in presented slot j -- which is exactly
    what the K-way position head is asked to predict.
    """
    batch, n_tiles = tiles.shape[:2]
    if not enabled:
        return tiles, torch.arange(n_tiles, device=tiles.device).expand(batch, -1)
    keys = torch.rand(batch, n_tiles, device=tiles.device, generator=generator)
    labels = keys.argsort(dim=1)
    index = labels[:, :, None, None].expand(-1, -1, tiles.shape[2], tiles.shape[3])
    return torch.gather(tiles, 1, index), labels


def _assign(position_logits):
    """Decode the K-way position head into an actual PERMUTATION.

    argmax is not a permutation: it can drop two tiles into the same slot, and
    an untrained head drops ALL of them into one slot because the class biases
    dominate the tile-dependent part of the logits. Reading the head correctly
    means taking the permutation with the highest total log-probability, which
    is what a jigsaw solver outputs. K! is tiny here, so brute force is exact;
    above K=6 fall back to ranking the tiles by their own argmax slot.
    """
    n_tiles = position_logits.shape[1]
    if n_tiles > 6:
        return _ranks(position_logits.argmax(-1).to(torch.float32))
    table = torch.tensor(list(itertools.permutations(range(n_tiles))),
                         device=position_logits.device)                     # (P, K)
    log_probability = F.log_softmax(position_logits.to(torch.float32), dim=-1)
    slots = table.transpose(0, 1)                                           # (K, P)
    total = sum(log_probability[:, tile, slots[tile]] for tile in range(n_tiles))
    return table[total.argmax(dim=1)]                                       # (B, K)


def _order_metrics(predicted, positions, *, tie_credit=0.5):
    """Exact accuracy, pair accuracy, and the TIE RATE.

    Ties are not a detail. If a pair of tiles is predicted into the same slot
    there is no implied order, and scoring that as an error silently drags the
    chance level of pair accuracy from 50% down to (1 - 1/K)/2 -- 37.5% at K=4
    -- and all the way to 0% for a head whose argmax is constant across tiles.
    Half credit puts chance back at exactly 50% for every predictor, because
    (1 - 1/K)/2 + (1/2)(1/K) = 1/2, and the tie rate is returned so that a
    degenerate readout shows up as a tie rate of 100% instead of masquerading
    as a below-chance score.
    """
    exact = (predicted == positions).all(dim=1).to(torch.float32)
    truth = torch.sign(positions[:, :, None] - positions[:, None, :])
    guess = torch.sign(predicted[:, :, None] - predicted[:, None, :])
    mask = truth != 0
    denominator = mask.sum((1, 2)).clamp_min(1).to(torch.float32)
    hits = ((guess == truth) & mask).sum((1, 2)).to(torch.float32)
    ties = ((guess == 0) & mask).sum((1, 2)).to(torch.float32)
    return exact, (hits + tie_credit * ties) / denominator, ties / denominator


def _participation_ratio(features):
    centered = features - features.mean(0, keepdims=True)
    values = np.linalg.eigvalsh(np.cov(centered, rowvar=False)
                                + 1e-12 * np.eye(centered.shape[1]))
    values = np.clip(values, 0, None)
    total = values.sum()
    return float(total ** 2 / np.square(values).sum()) if total > 0 else 0.0


def _ridge_r2(train_features, train_targets, test_features, test_targets, alphas):
    """Closed-form ridge with a CHRONOLOGICAL internal split for alpha selection."""
    mean = train_features.mean(0, keepdims=True)
    scale = train_features.std(0, keepdims=True) + _EPS
    a_train = (train_features - mean) / scale
    a_test = (test_features - mean) / scale
    cut = max(1, int(0.8 * len(a_train)))
    inner_x, inner_y, hold_x, hold_y = a_train[:cut], train_targets[:cut], \
        a_train[cut:], train_targets[cut:]
    if len(hold_x) < 2:
        inner_x, inner_y, hold_x, hold_y = a_train, train_targets, a_train, train_targets

    def solve(x, y, alpha):
        offset = y.mean(0, keepdims=True)
        return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - offset)), offset

    def r2(y_true, y_hat):
        residual = np.square(y_true - y_hat).sum(0)
        total = np.square(y_true - y_true.mean(0, keepdims=True)).sum(0)
        return 1.0 - residual / np.maximum(total, 1e-12)

    scored = [(float(np.mean(r2(hold_y, hold_x @ w + b))), alpha)
              for alpha in alphas for w, b in [solve(inner_x, inner_y, alpha)]]
    best = max(scored)[1]
    weights, offset = solve(a_train, train_targets, best)
    per_dimension = r2(test_targets, a_test @ weights + offset)
    return dict(r2=float(np.mean(per_dimension)), r2_per_dimension=per_dimension.tolist(),
                alpha=float(best), participation_ratio=_participation_ratio(train_features))


# --------------------------------------------------------------------------- #
# estimator
# --------------------------------------------------------------------------- #
class JigsawNet:
    """Cross-entropy-only self-supervised encoder: temporal order + forecasting."""

    _MODEL_TYPE = "jigsaw_net"
    _PARAM_NAMES = ("window_size", "n_tiles", "tile_gap", "output_dimension",
                    "num_hidden_units", "head_hidden_units", "dropout", "normalize",
                    "trunk_block", "lambda_order", "lambda_pair", "lambda_forecast",
                    "lambda_reconstruct", "forecast_levels", "order_grad_scale",
                    "tile_norm", "shuffle_tiles", "neuron_dropout",
                    "gain_jitter", "batch_size", "max_epochs", "learning_rate",
                    "weight_decay", "device", "random_state", "verbose", "log_every")

    def __init__(self, window_size=10, n_tiles=4, tile_gap=(1, 8),
                 output_dimension=64, num_hidden_units=64, head_hidden_units=64,
                 dropout=0.0, normalize=False, trunk_block="residual",
                 lambda_order=1.0, lambda_pair=0.5, lambda_forecast=0.0,
                 lambda_reconstruct=1.0,
                 forecast_levels=8, order_grad_scale=1.0,
                 tile_norm="mean", shuffle_tiles=True,
                 neuron_dropout=0.1, gain_jitter=0.1,
                 batch_size=512, max_epochs=60, learning_rate=1e-3, weight_decay=0.0,
                 device="cuda_if_available", random_state=42, verbose=True, log_every=1):
        self.window_size = _integer("window_size", window_size, 4)
        self.n_tiles = _integer("n_tiles", n_tiles, 2, 8)
        if not (isinstance(tile_gap, (tuple, list)) and len(tile_gap) == 2):
            raise ValueError("tile_gap must be a (minimum, maximum) pair.")
        low = _integer("tile_gap[0]", tile_gap[0], 1)
        high = _integer("tile_gap[1]", tile_gap[1], low)
        self.tile_gap = (low, high)
        self.output_dimension = _integer("output_dimension", output_dimension, 2)
        self.num_hidden_units = _integer("num_hidden_units", num_hidden_units, 2)
        self.head_hidden_units = _integer("head_hidden_units", head_hidden_units, 2)
        self.dropout = _real("dropout", dropout, 0, 1)
        self.normalize = _boolean("normalize", normalize)
        if trunk_block not in TRUNK_BLOCKS:
            raise ValueError(f"trunk_block must be one of {TRUNK_BLOCKS}.")
        self.trunk_block = trunk_block
        self.lambda_order = _real("lambda_order", lambda_order, 0)
        self.lambda_pair = _real("lambda_pair", lambda_pair, 0)
        self.lambda_forecast = _real("lambda_forecast", lambda_forecast, 0)
        self.lambda_reconstruct = _real("lambda_reconstruct", lambda_reconstruct, 0)
        self.forecast_levels = _integer("forecast_levels", forecast_levels, 2, 64)
        self.order_grad_scale = _real("order_grad_scale", order_grad_scale, 0)
        if tile_norm not in TILE_NORMS:
            raise ValueError(f"tile_norm must be one of {TILE_NORMS}.")
        self.tile_norm = tile_norm
        self.shuffle_tiles = _boolean("shuffle_tiles", shuffle_tiles)
        self.neuron_dropout = _real("neuron_dropout", neuron_dropout, 0, 1)
        self.gain_jitter = _real("gain_jitter", gain_jitter, 0)
        self.batch_size = _integer("batch_size", batch_size, 2)
        self.max_epochs = _integer("max_epochs", max_epochs, 0)
        self.learning_rate = _real("learning_rate", learning_rate, 0, strict_min=True)
        self.weight_decay = _real("weight_decay", weight_decay, 0)
        self.device = device
        self.random_state = _integer("random_state", random_state, 0)
        self.verbose = _boolean("verbose", verbose)
        self.log_every = _integer("log_every", log_every, 1)
        if self.lambda_order == 0 and self.lambda_pair == 0 and self.lambda_forecast == 0 \
                and self.lambda_reconstruct == 0 and self.max_epochs > 0:
            raise ValueError("All loss weights are zero; set max_epochs=0 for a random encoder.")
        self.is_fitted_ = False

    # -- properties -------------------------------------------------------- #
    @property
    def training_span(self):
        """Bins consumed by one training example (+1 for the forecast target)."""
        return self.n_tiles * self.window_size + (self.n_tiles - 1) * self.tile_gap[1] + 1

    def get_params(self):
        return {name: getattr(self, name) for name in self._PARAM_NAMES}

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Call fit before using this model.")

    # -- setup ------------------------------------------------------------- #
    def _resolve_device(self):
        name = self.device
        if name == "cuda_if_available":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(name)

    def _make_encoder(self, channels):
        """Override to swap the trunk. Must map (B, channels, window_size) -> (B, D)."""
        return _Encoder(channels, self.window_size, self.num_hidden_units,
                        self.output_dimension, self.dropout, self.normalize,
                        self.trunk_block)

    def _build(self, channels):
        torch.manual_seed(self.random_state)
        self.device_ = self._resolve_device()
        self.n_features_in_ = channels
        self.encoder_ = self._make_encoder(channels).to(self.device_)
        self.order_head_ = _OrderHead(self.output_dimension, self.head_hidden_units,
                                      self.n_tiles).to(self.device_)
        self.forecast_head_ = _ForecastHead(self.output_dimension, self.head_hidden_units,
                                            channels, self.forecast_levels).to(self.device_)
        self.reconstruct_head_ = _ReconstructHead(
            self.output_dimension, self.head_hidden_units, channels, self.window_size,
            self.forecast_levels).to(self.device_)
        self._generator = torch.Generator(device=self.device_)
        self._generator.manual_seed(self.random_state + 1)

    def _quantize_edges(self, data):
        """Per-neuron quantile edges for the forecast target.

        Quantiles, not equal-width bins: spike counts are heavily skewed, and
        equal-width bins would put almost every target in class 0, making the
        cross-entropy trivially minimized by predicting the mode.
        """
        probabilities = np.linspace(0, 1, self.forecast_levels + 1)[1:-1]
        edges = np.quantile(data, probabilities, axis=0).astype(np.float32)
        edges = np.maximum.accumulate(edges, axis=0)  # keep monotone under ties
        return torch.from_numpy(np.ascontiguousarray(edges)).to(self.device_)

    def _spans(self, X, n_features=None):
        sequences, _ = _sequences(X, self.training_span, n_features)
        lengths = [len(s) for s in sequences]
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        starts = np.concatenate([offsets[i] + np.arange(n - self.training_span + 1)
                                 for i, n in enumerate(lengths)]) if lengths else np.empty(0, int)
        data = torch.from_numpy(np.concatenate(sequences, axis=0)).to(self.device_)
        return data, starts.astype(np.int64), sequences[0].shape[1]

    def _draw(self, count):
        low, high = self.tile_gap
        return torch.randint(low, high + 1, (count, self.n_tiles - 1),
                             device=self.device_, generator=self._generator)

    # -- one step ---------------------------------------------------------- #
    def _step(self, data, starts):
        batch = len(starts)
        tiles, next_index = _gather_tiles(data, starts, self._draw(batch), self.window_size)
        parts = {}
        total = torch.zeros((), device=self.device_)

        # FORECAST / RECONSTRUCT share one forward on the RAW view, so the trunk
        # must represent absolute level. Skip it entirely when both are off
        # (order_only), otherwise that arm pays for a forward nothing reads.
        needs_raw = self.lambda_forecast > 0 or self.lambda_reconstruct > 0
        if needs_raw:
            raw = _augment(tiles, self.neuron_dropout, self.gain_jitter, self._generator)
            z_raw = self.encoder_(raw.reshape(-1, self.n_features_in_, self.window_size))
        if self.lambda_forecast > 0:
            target = self._bucketize(data[next_index.reshape(-1)])
            logits = self.forecast_head_(z_raw)
            forecast = F.cross_entropy(logits.reshape(-1, self.forecast_levels),
                                       target.reshape(-1))
            with torch.no_grad():
                parts["forecast_accuracy"] = (logits.argmax(-1) == target).to(
                    torch.float32).mean()
            parts["forecast"] = forecast
            total = total + self.lambda_forecast * forecast
        else:
            parts["forecast"] = torch.zeros((), device=self.device_)
            parts["forecast_accuracy"] = torch.zeros((), device=self.device_)

        # RECONSTRUCT: same raw embedding must reproduce every bin of its own tile.
        # Input is augmented, target is the CLEAN tile, so this is a denoising task.
        if self.lambda_reconstruct > 0:
            flat = tiles.permute(0, 1, 3, 2).reshape(-1, self.n_features_in_)
            target = self._bucketize(flat).reshape(-1, self.window_size,
                                                   self.n_features_in_).permute(0, 2, 1)
            logits = self.reconstruct_head_(z_raw)
            reconstruct = F.cross_entropy(logits.reshape(-1, self.forecast_levels),
                                          target.reshape(-1))
            with torch.no_grad():
                parts["reconstruct_accuracy"] = (logits.argmax(-1) == target).to(
                    torch.float32).mean()
            parts["reconstruct"] = reconstruct
            total = total + self.lambda_reconstruct * reconstruct
        else:
            parts["reconstruct"] = torch.zeros((), device=self.device_)
            parts["reconstruct_accuracy"] = torch.zeros((), device=self.device_)

        # ORDER: normalized view, so level drift cannot give the answer away.
        if self.lambda_order > 0 or self.lambda_pair > 0:
            view = _tile_normalize(
                _augment(tiles, self.neuron_dropout, self.gain_jitter, self._generator),
                self.tile_norm)
            # Present them in a RANDOM order; the label is that permutation, not
            # the identity. See _shuffle_tiles for why this is not cosmetic.
            view, labels = _shuffle_tiles(view, self.shuffle_tiles, self._generator)
            z_order = self.encoder_(view.reshape(-1, self.n_features_in_, self.window_size))
            z_order = _GradScale.apply(z_order, self.order_grad_scale)
            position_logits, scores = self.order_head_(
                z_order.reshape(batch, self.n_tiles, self.output_dimension))
            position_loss, pair_loss = _order_losses(position_logits, scores, labels)
            parts["position"], parts["pair"] = position_loss, pair_loss
            total = total + self.lambda_order * position_loss + self.lambda_pair * pair_loss
            with torch.no_grad():
                predicted = _ranks(scores) if self.lambda_order == 0 \
                    else _assign(position_logits)
                exact, pair_accuracy, tie_rate = _order_metrics(predicted, labels)
                parts["exact"], parts["pair_accuracy"] = exact.mean(), pair_accuracy.mean()
                parts["tie_rate"] = tie_rate.mean()
        else:
            for key in ("position", "pair", "exact", "pair_accuracy", "tie_rate"):
                parts[key] = torch.zeros((), device=self.device_)
        parts["total"] = total
        return parts

    def _bucketize(self, values):
        """(M,N) activity -> (M,N) integer class per neuron, using stored edges."""
        out = torch.zeros(values.shape, dtype=torch.long, device=values.device)
        for level in range(self.forecast_levels - 1):
            out = out + (values > self.forecast_edges_[level][None, :]).to(torch.long)
        return out

    # -- fit --------------------------------------------------------------- #
    def fit(self, X, X_valid=None, *, validate_every=1):
        validate_every = _integer("validate_every", validate_every)
        first, _ = _sequences(X, self.training_span)
        self._build(first[0].shape[1])
        data, starts, channels = self._spans(X, first[0].shape[1])
        self.forecast_edges_ = self._quantize_edges(
            np.concatenate(first, axis=0)) if self.forecast_levels > 1 else None
        self.n_spans_ = int(len(starts))
        if self.n_spans_ < 2 and self.max_epochs > 0:
            raise ValueError(
                f"Only {self.n_spans_} valid span start(s). Each sequence needs at least "
                f"training_span+1={self.training_span + 1} bins, or reduce window_size / "
                f"n_tiles / tile_gap.")
        self.is_fitted_ = True
        self.history_, self.validation_history_ = [], []
        if self.max_epochs == 0:
            if self.verbose:
                print(f"JigsawNet: max_epochs=0 -> frozen random encoder "
                      f"({self.n_spans_} spans, {channels} neurons).", flush=True)
            return self

        parameters = list(self.encoder_.parameters()) + list(self.order_head_.parameters()) \
            + list(self.forecast_head_.parameters()) + list(self.reconstruct_head_.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        if self.verbose:
            print(f"JigsawNet: {self.n_spans_} spans, window={self.window_size}, "
                  f"K={self.n_tiles}, dim={self.output_dimension}, trunk={self.trunk_block}, "
                  f"normalize={self.normalize}, tile_norm={self.tile_norm}, "
                  f"lambda(order,pair,forecast,reconstruct)="
                  f"({self.lambda_order},{self.lambda_pair},{self.lambda_forecast},"
                  f"{self.lambda_reconstruct}), device={self.device_}", flush=True)
            if self.max_epochs * self.n_spans_ > 2_000_000:
                print(f"      NOTE: {self.max_epochs} epochs x {self.n_spans_} overlapping "
                      f"spans. Expect the held-out position CE to end up far above log(K) -- "
                      f"the ORDER HEAD will memorize. That is a statement about the head, not "
                      f"about the embedding: on Perich the best decoding R2 was reached at the "
                      f"LARGEST budget tried, with the pretext task thoroughly overfit. Sweep "
                      f"max_epochs and pick by R2, never by pretext CE.", flush=True)

        index = torch.from_numpy(starts).to(self.device_)
        for epoch in range(self.max_epochs):
            order = torch.randperm(self.n_spans_, device=self.device_, generator=self._generator)
            totals, seen = {}, 0
            self.encoder_.train(); self.order_head_.train(); self.forecast_head_.train()
            self.reconstruct_head_.train()
            for begin in range(0, self.n_spans_, self.batch_size):
                chunk = index[order[begin:begin + self.batch_size]]
                if len(chunk) < 2:
                    continue
                parts = self._step(data, chunk)
                if not torch.isfinite(parts["total"]):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}.")
                optimizer.zero_grad(set_to_none=True)
                parts["total"].backward()
                nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
                for key, value in parts.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(chunk)
                seen += len(chunk)
            row = {"epoch": epoch + 1, **{k: v / max(seen, 1) for k, v in totals.items()}}
            self.history_.append(row)
            if self.verbose and ((epoch + 1) % self.log_every == 0 or epoch + 1 == self.max_epochs):
                print(f"  epoch {epoch + 1}/{self.max_epochs} loss={row['total']:.4f} "
                      f"| position CE={row['position']:.4f} pair BCE={row['pair']:.4f} "
                      f"forecast CE={row['forecast']:.4f} "
                      f"reconstruct CE={row['reconstruct']:.4f} "
                      f"| train exact={100 * row['exact']:.1f}% "
                      f"pair={100 * row['pair_accuracy']:.1f}% "
                      f"ties={100 * row['tie_rate']:.1f}% "
                      f"forecast acc={100 * row['forecast_accuracy']:.1f}% "
                      f"recon acc={100 * row['reconstruct_accuracy']:.1f}%", flush=True)
            if X_valid is not None and ((epoch + 1) % validate_every == 0
                                        or epoch + 1 == self.max_epochs):
                metrics = self.evaluate_pretext(X_valid, max_spans=512, verbose=False)
                metrics["epoch"] = epoch + 1
                self.validation_history_.append(metrics)
                if self.verbose:
                    print(f"      valid: exact={metrics['exact_accuracy_percent']:.2f}% "
                          f"(chance {metrics['chance_exact_percent']:.2f}%) "
                          f"pair={metrics['pair_accuracy_percent']:.2f}% "
                          f"| mean-sort baseline pair={metrics['baseline_mean_sort_pair_percent']:.2f}%",
                          flush=True)
        return self

    # -- inference --------------------------------------------------------- #
    def transform(self, X, *, pad=True, batch_size=None, return_indices=False):
        """Natural windows. No augmentation, no tile normalization, no heads."""
        self._check_fitted()
        _boolean("pad", pad)
        size = self.batch_size if batch_size is None else _integer("batch_size", batch_size)
        sequences, is_list = _sequences(X, 1 if pad else self.window_size, self.n_features_in_)
        left = self.window_size // 2
        right = self.window_size - left - 1
        values, times = [], []
        self.encoder_.eval()
        with torch.inference_mode():
            for sequence in sequences:
                if pad:
                    centers = np.arange(len(sequence), dtype=np.int64)
                    sequence = np.pad(sequence, ((left, right), (0, 0)), mode="edge")
                else:
                    centers = np.arange(left, len(sequence) - right, dtype=np.int64)
                out = np.empty((len(centers), self.output_dimension), dtype=np.float32)
                for begin in range(0, len(centers), size):
                    end = min(begin + size, len(centers))
                    locations = np.arange(begin, end)[:, None] + np.arange(self.window_size)
                    windows = np.ascontiguousarray(sequence[locations].transpose(0, 2, 1))
                    out[begin:end] = self.encoder_(
                        torch.from_numpy(windows).to(self.device_)).cpu().numpy()
                values.append(out)
                times.append(centers)
        self.encoder_.train()
        result = values if is_list else values[0]
        index = times if is_list else times[0]
        return (result, index) if return_indices else result

    def fit_transform(self, X, **kwargs):
        return self.fit(X).transform(X, **kwargs)

    # -- evaluation -------------------------------------------------------- #
    def evaluate_pretext(self, X, *, max_spans=1024, batch_size=256, repeats=8,
                         random_state=200042, verbose=None):
        """Held-out order accuracy next to the trivial shortcut baselines.

        `repeats` re-draws the gaps and the presentation permutation for every
        span start and averages. This is a MEASUREMENT-noise fix, not a sample-
        size fix: on a short validation split there may be only ~550 usable
        span starts, which puts the standard error on pair accuracy near 2%, and
        at that resolution a 49% arm and a 53% arm are the same arm. Averaging
        repeated draws shrinks the noise in the estimate; it does NOT create new
        independent spans, so every confidence interval and z-test below still
        uses n = number of span starts. Reported as n_spans (for the tests) and
        n_draws (what the averages were computed from).
        """
        self._check_fitted()
        verbose = self.verbose if verbose is None else verbose
        repeats = _integer("repeats", repeats, minimum=1)
        data, starts, _ = self._spans(X, self.n_features_in_)
        if len(starts) == 0:
            raise ValueError("No valid spans in X.")
        rng = np.random.default_rng(random_state)
        chosen = starts if len(starts) <= max_spans else rng.choice(starts, max_spans, False)
        chosen = torch.from_numpy(np.sort(chosen)).to(self.device_)
        generator = torch.Generator(device=self.device_).manual_seed(random_state)
        low, high = self.tile_gap
        totals = {k: 0.0 for k in ("exact", "pair", "tie", "argmax_pair", "argmax_tie",
                                   "rank_exact", "rank_pair", "mean_exact", "mean_pair",
                                   "norm_exact", "norm_pair", "position_ce")}
        count = 0
        self.encoder_.eval(); self.order_head_.eval()
        with torch.inference_mode():
          for _ in range(repeats):
            for begin in range(0, len(chosen), batch_size):
                block = chosen[begin:begin + batch_size]
                gaps = torch.randint(low, high + 1, (len(block), self.n_tiles - 1),
                                     device=self.device_, generator=generator)
                tiles, _ = _gather_tiles(data, block, gaps, self.window_size)
                view = _tile_normalize(tiles, self.tile_norm)
                # Shuffle AND carry the same permutation into the baselines, so
                # the shortcut baselines are scored on exactly what the model saw.
                view, positions = _shuffle_tiles(view, self.shuffle_tiles, generator)
                tiles = torch.gather(
                    tiles, 1, positions[:, :, None, None].expand(-1, -1, *tiles.shape[2:])
                ) if self.shuffle_tiles else tiles
                z = self.encoder_(view.reshape(-1, self.n_features_in_, self.window_size))
                position_logits, scores = self.order_head_(
                    z.reshape(len(block), self.n_tiles, self.output_dimension))
                # Headline decode: a real permutation, so chance is 1/K! and 50% exactly.
                predicted = _assign(position_logits) if self.lambda_order > 0 else _ranks(scores)
                exact, pair, tie = _order_metrics(predicted, positions)
                # Kept for comparison: the naive argmax, which is NOT a permutation.
                _, argmax_pair, argmax_tie = _order_metrics(position_logits.argmax(-1), positions)
                rank_exact, rank_pair, _ = _order_metrics(_ranks(scores), positions)
                mean_exact, mean_pair, _ = _order_metrics(_ranks(tiles.mean(dim=(2, 3))), positions)
                norm_exact, norm_pair, _ = _order_metrics(
                    _ranks(tiles.reshape(len(block), self.n_tiles, -1).norm(dim=2)), positions)
                weight = len(block)
                for key, value in (("exact", exact), ("pair", pair), ("tie", tie),
                                   ("argmax_pair", argmax_pair), ("argmax_tie", argmax_tie),
                                   ("rank_exact", rank_exact), ("rank_pair", rank_pair),
                                   ("mean_exact", mean_exact), ("mean_pair", mean_pair),
                                   ("norm_exact", norm_exact), ("norm_pair", norm_pair)):
                    totals[key] += float(value.mean()) * weight
                totals["position_ce"] += float(F.cross_entropy(
                    position_logits.reshape(-1, self.n_tiles), positions.reshape(-1))) * weight
                count += weight
        self.encoder_.train(); self.order_head_.train()
        scale = 100.0 / max(count, 1)
        result = dict(
            exact_accuracy_percent=totals["exact"] * scale,
            pair_accuracy_percent=totals["pair"] * scale,
            tie_rate_percent=totals["tie"] * scale,
            argmax_pair_accuracy_percent=totals["argmax_pair"] * scale,
            argmax_tie_rate_percent=totals["argmax_tie"] * scale,
            rank_exact_accuracy_percent=totals["rank_exact"] * scale,
            rank_pair_accuracy_percent=totals["rank_pair"] * scale,
            baseline_mean_sort_exact_percent=totals["mean_exact"] * scale,
            baseline_mean_sort_pair_percent=totals["mean_pair"] * scale,
            baseline_norm_sort_exact_percent=totals["norm_exact"] * scale,
            baseline_norm_sort_pair_percent=totals["norm_pair"] * scale,
            position_cross_entropy=totals["position_ce"] / max(count, 1),
            uniform_cross_entropy=math.log(self.n_tiles),
            decode="assignment" if self.lambda_order > 0 else "score_rank",
            chance_exact_percent=100.0 / math.factorial(self.n_tiles),
            chance_pair_percent=50.0,
            # n_spans is the INDEPENDENT sample size -- span starts, not draws.
            # Every CI and z-test downstream must use this one. n_draws only says
            # how much averaging went into the point estimates.
            n_spans=int(len(chosen)), n_draws=int(count), repeats=int(repeats))
        if verbose:
            print(f"  pretext: exact={result['exact_accuracy_percent']:.2f}% "
                  f"(chance {result['chance_exact_percent']:.2f}%) "
                  f"pair={result['pair_accuracy_percent']:.2f}% (chance 50%) "
                  f"| decode={result['decode']} "
                  f"argmax pair={result['argmax_pair_accuracy_percent']:.2f}% "
                  f"ties={result['argmax_tie_rate_percent']:.1f}% "
                  f"| mean-sort pair={result['baseline_mean_sort_pair_percent']:.2f}%", flush=True)
        return result

    def evaluate_decoding(self, X_train, y_train, X_test, y_test,
                          *, alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)):
        """Ridge R2 from the embedding AND from the raw window, side by side.

        The raw-window number is not decoration. If the embedding does not beat
        it, the encoder is removing information and that is the finding.
        """
        self._check_fitted()
        out = {}
        z_train, index_train = self.transform(X_train, pad=False, return_indices=True)
        z_test, index_test = self.transform(X_test, pad=False, return_indices=True)
        target_train = np.asarray(y_train, dtype=np.float64)[index_train]
        target_test = np.asarray(y_test, dtype=np.float64)[index_test]
        out["embedding"] = _ridge_r2(z_train.astype(np.float64), target_train,
                                     z_test.astype(np.float64), target_test, alphas)
        raw_train = np.lib.stride_tricks.sliding_window_view(
            np.asarray(X_train, dtype=np.float64), self.window_size, axis=0)
        raw_test = np.lib.stride_tricks.sliding_window_view(
            np.asarray(X_test, dtype=np.float64), self.window_size, axis=0)
        out["raw_window"] = _ridge_r2(raw_train.reshape(len(raw_train), -1), target_train,
                                      raw_test.reshape(len(raw_test), -1), target_test, alphas)
        out["embedding_minus_raw"] = out["embedding"]["r2"] - out["raw_window"]["r2"]
        return out

    # -- persistence ------------------------------------------------------- #
    def save(self, path):
        self._check_fitted()
        torch.save(dict(model_type=self._MODEL_TYPE, format_version=1, params=self.get_params(),
                        n_features_in=self.n_features_in_,
                        encoder=self.encoder_.state_dict(),
                        order_head=self.order_head_.state_dict(),
                        forecast_head=self.forecast_head_.state_dict(),
                        reconstruct_head=self.reconstruct_head_.state_dict(),
                        forecast_edges=None if self.forecast_edges_ is None
                        else self.forecast_edges_.cpu(),
                        history=self.history_, validation_history=self.validation_history_),
                   Path(path))
        return self

    @classmethod
    def load(cls, path, device="cuda_if_available"):
        try:
            data = torch.load(Path(path), map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(Path(path), map_location="cpu")
        if data.get("model_type") != cls._MODEL_TYPE or data.get("format_version") != 1:
            raise ValueError("Expected a jigsaw_net checkpoint.")
        params = dict(data["params"]); params["device"] = device
        model = cls(**params)
        model._build(data["n_features_in"])
        model.encoder_.load_state_dict(data["encoder"])
        model.order_head_.load_state_dict(data["order_head"])
        model.forecast_head_.load_state_dict(data["forecast_head"])
        if "reconstruct_head" in data:
            model.reconstruct_head_.load_state_dict(data["reconstruct_head"])
        edges = data.get("forecast_edges")
        model.forecast_edges_ = None if edges is None else edges.to(model.device_)
        model.history_ = data["history"]
        model.validation_history_ = data["validation_history"]
        model.is_fitted_ = True
        return model


Jigsaw = JigsawNet


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"  ok  {message}")


def _synthetic(n=6000, channels=24, seed=0):
    """Latent circle -> Poisson-ish rates. Behavior is linearly decodable."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 60 * np.pi, n)
    latent = np.stack([np.sin(t), np.cos(t)], axis=1)
    weights = rng.normal(size=(2, channels))
    rates = np.exp(0.8 * latent @ weights + 0.4 * rng.normal(size=(n, channels)) - 0.5)
    return rng.poisson(rates).astype(np.float32), latent.astype(np.float32)


def _self_test():
    print("1. trunk receptive field")
    for window in (4, 5, 9, 10, 16, 21):
        for block in TRUNK_BLOCKS:
            trunk = _Trunk(7, window, 8, 0.0, block)
            out = trunk(torch.zeros(3, 7, window))
            _check(out.shape == (3, 8), f"{block} trunk, window={window} -> one bin, {out.shape}")
    depthwise = [m for m in _Separable(8, 0.0).modules()
                 if isinstance(m, nn.Conv1d) and m.kernel_size == (3,)]
    _check(len(depthwise) == 1 and depthwise[0].groups == depthwise[0].in_channels,
           "the separable block's 3-tap convolution is depthwise")

    print("2. tiles never overlap and the forecast target sits in the gap")
    ramp = torch.arange(300.0)[:, None].repeat(1, 3)
    tiles, nxt = _gather_tiles(ramp, torch.tensor([0, 50]), torch.tensor([[1, 2, 3], [1, 1, 1]]), 4)
    _check(tiles.shape == (2, 4, 3, 4), f"shape (B,K,N,W)={tuple(tiles.shape)}")
    _check(tiles[0, :, 0, 0].tolist() == [0, 5, 11, 18], "gapped tile starts [0,5,11,18]")
    _check((tiles[:, 1:, 0, 0] > tiles[:, :-1, 0, -1]).all(), "tiles are disjoint")
    _check((nxt[:, :-1] < tiles[:, 1:, 0, 0]).all(),
           "every forecast target bin lies strictly inside a discarded gap")

    print("3. losses are cross-entropy and hit the right chance levels")
    batch, K = 64, 4
    positions = torch.arange(K).expand(batch, -1)
    position_loss, pair_loss = _order_losses(torch.zeros(batch, K, K), torch.zeros(batch, K),
                                             positions)
    _check(abs(float(position_loss) - math.log(K)) < 1e-6,
           f"uniform position logits cost exactly log(K)={math.log(K):.4f}")
    _check(abs(float(pair_loss) - math.log(2)) < 1e-6,
           f"tied pair scores cost exactly log(2)={math.log(2):.4f}")
    perfect = F.one_hot(positions, K).float() * 20
    position_loss, _ = _order_losses(perfect, -positions.float(), positions)
    _check(float(position_loss) < 1e-6, "a correct confident prediction costs ~0")

    print("3b. the order metric has the chance level it claims to have")
    big = 200000
    truth = torch.arange(K).expand(big, -1)
    constant = torch.randint(0, K, (big, 1)).expand(-1, K)
    _, pair_half, tie = _order_metrics(constant, truth)
    _check(abs(float(pair_half.mean()) - 0.5) < 1e-6 and float(tie.mean()) == 1.0,
           "a head that puts every tile in ONE slot scores exactly 50% with 100% ties "
           "(scoring ties as errors would call this 0.00% and look anti-correlated)")
    _, pair_zero, _ = _order_metrics(constant, truth, tie_credit=0.0)
    _check(float(pair_zero.mean()) == 0.0,
           "  ...and the tie_credit=0 convention is what produced the 0.00% in run 1")
    iid = torch.randint(0, K, (big, K))
    _, pair_iid, tie_iid = _order_metrics(iid, truth)
    _, pair_iid_zero, _ = _order_metrics(iid, truth, tie_credit=0.0)
    _check(abs(float(pair_iid.mean()) - 0.5) < 0.005,
           f"an i.i.d. random argmax scores {100 * float(pair_iid.mean()):.2f}% with half credit")
    _check(abs(float(pair_iid_zero.mean()) - (1 - 1 / K) / 2) < 0.005,
           f"  ...but only {100 * float(pair_iid_zero.mean()):.2f}% with tie_credit=0, i.e. the "
           f"true chance level of the old metric was (1-1/K)/2={100 * (1 - 1 / K) / 2:.1f}%, not 50%")
    permutation = torch.argsort(torch.rand(big, K), dim=1)
    _, pair_perm, tie_perm = _order_metrics(permutation, truth)
    _check(abs(float(pair_perm.mean()) - 0.5) < 0.005 and float(tie_perm.mean()) == 0.0,
           "a permutation decode has no ties and sits at 50% either way")

    print("3c. _assign returns a permutation and maximizes total log-probability")
    logits = torch.randn(256, K, K) * 3
    assigned = _assign(logits)
    _check(all(sorted(row.tolist()) == list(range(K)) for row in assigned),
           "every _assign output is a genuine permutation of the K slots")
    log_probability = F.log_softmax(logits, dim=-1)
    chosen_score = log_probability.gather(2, assigned[:, :, None]).squeeze(-1).sum(1)
    every = torch.tensor(list(itertools.permutations(range(K))))
    brute = torch.stack([log_probability.gather(
        2, p[None, :, None].expand(256, -1, -1)).squeeze(-1).sum(1) for p in every])
    _check(torch.allclose(chosen_score, brute.max(dim=0).values, atol=1e-5),
           "_assign finds the highest-scoring permutation (checked against all K! by hand)")
    confident = F.one_hot(torch.arange(K), K).float()[None].expand(64, -1, -1) * 20
    _check((_assign(confident) == torch.arange(K)).all(),
           "a confident correct head decodes to the identity permutation")
    degenerate = torch.zeros(64, K, K); degenerate[:, :, 1] = 10.0
    _check(all(sorted(row.tolist()) == list(range(K)) for row in _assign(degenerate)),
           "even when every tile votes for the SAME slot, _assign still returns a permutation")

    print("4. the head is permutation equivariant")
    head = _OrderHead(6, 16, K)
    z = torch.randn(8, K, 6)
    shuffle = torch.stack([torch.randperm(K) for _ in range(8)])
    logits_a, scores_a = head(z)
    logits_b, scores_b = head(z.gather(1, shuffle[:, :, None].expand(-1, -1, 6)))
    _check(torch.allclose(logits_b, logits_a.gather(1, shuffle[:, :, None].expand(-1, -1, K)),
                          atol=1e-5), "position logits permute with the tiles")
    _check(torch.allclose(scores_b, scores_a.gather(1, shuffle), atol=1e-5),
           "pair scores permute with the tiles")

    print("4b. shuffling presents a real permutation and keeps tile<->label paired")
    gen = torch.Generator().manual_seed(11)
    marks = torch.arange(6 * K, dtype=torch.float32).reshape(6, K, 1, 1).expand(6, K, 3, 5)
    shown, labels = _shuffle_tiles(marks.contiguous(), True, gen)
    _check(all(sorted(row.tolist()) == list(range(K)) for row in labels),
           "every label row is a permutation of 0..K-1, not a repeated draw")
    # labels[b, j] must be the CHRONOLOGICAL index of the tile now sitting in slot j.
    # Each tile is stamped with its own chronological index, so this is checkable.
    stamped = shown[:, :, 0, 0] - (torch.arange(6)[:, None] * K)
    _check(torch.equal(stamped.long(), labels),
           "the tile in presented slot j really is the tile whose label is labels[:, j]")
    _check(not torch.equal(labels, torch.arange(K).expand(6, -1)),
           "the shuffled label is NOT the identity permutation (that was the hazard)")
    off_shown, off_labels = _shuffle_tiles(marks.contiguous(), False, gen)
    _check(torch.equal(off_labels, torch.arange(K).expand(6, -1))
           and torch.equal(off_shown, marks),
           "shuffle_tiles=False is an exact no-op, so old runs stay reproducible")
    # The real payoff: a head that ignores its input and always answers the
    # identity scores at chance once the labels are shuffled, and 100% without.
    cheat = torch.zeros(6, K, K)
    cheat[:, torch.arange(K), torch.arange(K)] = 10.0
    _check(float(_order_metrics(_assign(cheat), off_labels)[0].mean()) == 1.0,
           "with identity labels a constant 'answer [0,1,...,K-1]' head scores 100%")
    _check(float(_order_metrics(_assign(cheat), labels)[0].mean()) < 0.5,
           "with shuffled labels the same cheating head collapses to chance")

    print("5. tile normalization kills the level shortcut")
    raw = torch.randn(4, K, 5, 8)
    offset = torch.randn(4, K, 1, 1) * 5
    for mode in ("mean", "zscore", "global_mean"):
        _check(torch.allclose(_tile_normalize(raw + offset, mode), _tile_normalize(raw, mode),
                              atol=1e-3), f"tile_norm={mode} removes per-tile level")
    ramped = raw + torch.arange(K).float()[None, :, None, None] * 4
    _check((_ranks(ramped.mean(dim=(2, 3))) == torch.arange(K)).all(),
           "mean-sort solves the puzzle perfectly under a level ramp (the shortcut)")
    _check(not (_ranks(_tile_normalize(ramped, "mean").mean(dim=(2, 3))) == torch.arange(K)).all(),
           "after normalization the ramp no longer gives the order away")

    print("6. quantization is balanced, not mode-collapsed")
    data, _ = _synthetic(3000, 12, seed=3)
    model = JigsawNet(window_size=8, n_tiles=3, tile_gap=(1, 3), output_dimension=8,
                      num_hidden_units=12, head_hidden_units=12, max_epochs=0,
                      device="cpu", verbose=False).fit(data)
    model.forecast_edges_ = model._quantize_edges(data)
    classes = model._bucketize(torch.from_numpy(data))
    share = torch.bincount(classes.reshape(-1), minlength=8).float() / classes.numel()
    _check(int(classes.max()) < model.forecast_levels and int(classes.min()) >= 0,
           "every target lands inside [0, forecast_levels)")
    _check(float(share.max()) < 0.75,
           f"quantile edges avoid mode collapse (largest class share {float(share.max()):.2f}) "
           "-> the forecast CE is not minimized by always predicting one class")

    print("6b. the reconstruction target is the tile itself, bin for bin")
    window, tiles_k, neurons = model.window_size, model.n_tiles, 12
    tiles = torch.from_numpy(data[:5 * tiles_k * window].reshape(5, tiles_k, window, neurons)
                             ).permute(0, 1, 3, 2).contiguous()          # (B,K,N,W)
    flat = tiles.permute(0, 1, 3, 2).reshape(-1, neurons)
    target = model._bucketize(flat).reshape(-1, window, neurons).permute(0, 2, 1)
    _check(target.shape == (5 * tiles_k, neurons, window),
           f"target is (B*K, N, W)={tuple(target.shape)}, matching the head's logit layout")
    direct = model._bucketize(tiles[0, 0].T).T                           # (N, W), the honest way
    _check(torch.equal(target[0], direct),
           "tile (0,0) round-trips: the reshape/permute chain preserves the (neuron, bin) pairing")
    head = _ReconstructHead(8, 12, neurons, window, model.forecast_levels)
    logits = head(torch.randn(5 * tiles_k, 8))
    _check(logits.shape == (5 * tiles_k, neurons, window, model.forecast_levels),
           f"head logits {tuple(logits.shape)} line up with the target elementwise")
    _check(logits.reshape(-1, model.forecast_levels).shape[0] == target.reshape(-1).shape[0],
           "flattening logits and target for cross_entropy keeps them aligned")

    print("7. end to end: training must not destroy decodable structure")
    data, latent = _synthetic(7000, 24, seed=1)
    cut = 5000
    alphas = (1e-2, 1e-1, 1.0, 10.0, 100.0)
    shared = dict(window_size=8, n_tiles=3, tile_gap=(1, 3), output_dimension=16,
                  num_hidden_units=32, head_hidden_units=32, batch_size=256,
                  learning_rate=3e-3, device="cpu", verbose=False, random_state=0)
    trained = JigsawNet(max_epochs=12, **shared).fit(data[:cut])
    random_encoder = JigsawNet(max_epochs=0, **shared).fit(data[:cut])
    scores = {}
    for name, fitted in (("trained", trained), ("random", random_encoder)):
        z_train, i_train = fitted.transform(data[:cut], pad=False, return_indices=True)
        z_test, i_test = fitted.transform(data[cut:], pad=False, return_indices=True)
        scores[name] = _ridge_r2(z_train.astype(np.float64), latent[:cut][i_train].astype(np.float64),
                                 z_test.astype(np.float64), latent[cut:][i_test].astype(np.float64),
                                 alphas)["r2"]
    print(f"      trained R2={scores['trained']:.4f}  random-encoder R2={scores['random']:.4f}")
    _check(scores["trained"] > 0.5, f"trained embedding decodes the latent (R2={scores['trained']:.3f})")
    _check(scores["trained"] > scores["random"] - 0.02,
           "training does not fall behind the random-encoder control")
    pretext = trained.evaluate_pretext(data[cut:], max_spans=400, repeats=4, verbose=False)
    print(f"      held-out exact={pretext['exact_accuracy_percent']:.1f}% "
          f"(chance {pretext['chance_exact_percent']:.1f}%) "
          f"pair={pretext['pair_accuracy_percent']:.1f}% "
          f"(shortcut {pretext['baseline_mean_sort_pair_percent']:.1f}%)")
    # Threshold deliberately modest. Labels are shuffled now, so this is a real
    # test of the head rather than a test that it can output a constant, and it
    # runs for 12 epochs on 7000 synthetic bins. What must hold is that it beats
    # chance AND beats the level shortcut; a large margin is not the claim.
    _check(pretext["pair_accuracy_percent"] > 53
           and pretext["pair_accuracy_percent"] > pretext["baseline_mean_sort_pair_percent"],
           "the order task is learned out of sample, above chance and above the shortcut")

    print("8. normalize=True is the ablation that costs R2")
    spherical = JigsawNet(max_epochs=12, normalize=True, **shared).fit(data[:cut])
    z_train, i_train = spherical.transform(data[:cut], pad=False, return_indices=True)
    z_test, i_test = spherical.transform(data[cut:], pad=False, return_indices=True)
    sphere_r2 = _ridge_r2(z_train.astype(np.float64), latent[:cut][i_train].astype(np.float64),
                          z_test.astype(np.float64), latent[cut:][i_test].astype(np.float64),
                          alphas)["r2"]
    print(f"      normalize=True R2={sphere_r2:.4f} vs normalize=False R2={scores['trained']:.4f}")
    _check(np.isfinite(sphere_r2), "the spherical ablation runs")

    print("9. transform is deterministic, aligned and augmentation-free")
    a = trained.transform(data[cut:], pad=False)
    b = trained.transform(data[cut:], pad=False)
    _check(np.allclose(a, b), "transform is deterministic (no augmentation leaks in)")
    _check(len(a) == len(data[cut:]) - trained.window_size + 1,
           f"pad=False drops exactly window_size-1={trained.window_size - 1} bins")
    _check(len(trained.transform(data[cut:], pad=True)) == len(data[cut:]),
           "pad=True returns one row per input bin")
    _, index = trained.transform(data[cut:], pad=False, return_indices=True)
    _check(index[0] == trained.window_size // 2, "pad=False starts at window_size//2")

    print("10. controls run and bad configs are rejected")
    for override in ({"lambda_order": 0.0, "lambda_pair": 0.0}, {"lambda_forecast": 1.0},
                     {"lambda_reconstruct": 0.0}, {"shuffle_tiles": False},
                     {"lambda_forecast": 1.0, "lambda_order": 0.0,
                      "lambda_pair": 0.0, "lambda_reconstruct": 0.0},
                     {"order_grad_scale": 0.0}, {"trunk_block": "separable"},
                     {"tile_norm": "none"}, {"n_tiles": 2}):
        settings = dict(shared, max_epochs=2); settings.update(override)
        fitted = JigsawNet(**settings).fit(data[:1500])
        _check(np.isfinite(fitted.transform(data[:400], pad=False)).all(),
               f"control {override} trains and transforms cleanly")
    for bad in ({"window_size": 3}, {"n_tiles": 1}, {"tile_gap": (0, 3)},
                {"trunk_block": "mobilenet"}, {"tile_norm": "l2"},
                {"lambda_reconstruct": -1.0},
                {"lambda_order": 0.0, "lambda_pair": 0.0, "lambda_forecast": 0.0,
                 "lambda_reconstruct": 0.0}):
        try:
            JigsawNet(**dict(shared, **bad))
            raise AssertionError(f"{bad} should have been rejected")
        except ValueError:
            pass
    _check(True, "invalid constructor arguments all raise ValueError")
    try:
        JigsawNet(window_size=8, n_tiles=3, tile_gap=(1, 3), max_epochs=1, device="cpu",
                  verbose=False).fit(np.zeros((31, 5), dtype=np.float32))
        raise AssertionError("a one-span recording should have been rejected")
    except ValueError:
        _check(True, "a recording with fewer than two spans raises instead of dividing by zero")

    print("11. save / load round trip")
    path = Path("_jigsaw_net_selftest.pt")
    try:
        trained.save(path)
        restored = JigsawNet.load(path, device="cpu")
        _check(np.allclose(restored.transform(data[cut:cut + 300], pad=False),
                           trained.transform(data[cut:cut + 300], pad=False), atol=1e-6),
               "a reloaded model reproduces its embedding exactly")
    finally:
        path.unlink(missing_ok=True)

    print("\nAll self-tests passed. No contrastive loss exists in this module.")


def _demo():
    data, latent = _synthetic(9000, 32, seed=7)
    cut = 6500
    shared = dict(window_size=10, n_tiles=4, tile_gap=(1, 6), output_dimension=32,
                  num_hidden_units=48, head_hidden_units=48, batch_size=256,
                  learning_rate=2e-3, device="cuda_if_available", verbose=False, random_state=0)
    arms = {
        "proposed (order+recon)": dict(max_epochs=25),
        "+ forecast (ablation)": dict(max_epochs=25, lambda_forecast=1.0),
        "no reconstruct anchor": dict(max_epochs=25, lambda_reconstruct=0.0),
        "order only": dict(max_epochs=25, lambda_reconstruct=0.0),
        "forecast only": dict(max_epochs=25, lambda_order=0.0, lambda_pair=0.0,
                              lambda_forecast=1.0, lambda_reconstruct=0.0),
        "reconstruct only": dict(max_epochs=25, lambda_order=0.0, lambda_pair=0.0),
        "random encoder": dict(max_epochs=0),
        "normalize=True (sphere)": dict(max_epochs=25, normalize=True),
        "separable trunk": dict(max_epochs=25, trunk_block="separable"),
    }
    print(f"{'arm':<26}{'R2':>9}{'raw R2':>9}{'delta':>9}{'p.ratio':>9}"
          f"{'pair %':>9}{'ties %':>8}{'shortcut':>10}")
    print("-" * 89)
    for name, override in arms.items():
        model = JigsawNet(**dict(shared, **override)).fit(data[:cut])
        decoding = model.evaluate_decoding(data[:cut], latent[:cut], data[cut:], latent[cut:])
        pretext = model.evaluate_pretext(data[cut:], max_spans=400, verbose=False)
        print(f"{name:<26}{decoding['embedding']['r2']:>9.4f}"
              f"{decoding['raw_window']['r2']:>9.4f}{decoding['embedding_minus_raw']:>9.4f}"
              f"{decoding['embedding']['participation_ratio']:>9.2f}"
              f"{pretext['pair_accuracy_percent']:>9.2f}"
              f"{pretext['argmax_tie_rate_percent']:>8.1f}"
              f"{pretext['baseline_mean_sort_pair_percent']:>10.2f}")
    print("-" * 89)
    print("delta < 0 means the encoder is removing information the raw window already had.")
    print("p.ratio is the participation ratio of the embedding: if it collapses, so does R2.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--demo", action="store_true")
    options = parser.parse_args()
    if options.self_test:
        _self_test()
    elif options.demo:
        _demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


# """JigsawNet: a self-supervised neural encoder trained ONLY with cross-entropy.

# No contrastive learning. No InfoNCE. No positive/negative pairs. No temperature.
# No dependency on, and no objective borrowed from, time-contrastive embedding
# methods. Every loss in this file is torch cross_entropy or
# binary_cross_entropy_with_logits on a task with a hard, known-correct label.

#     from jigsaw_net import JigsawNet
#     model = JigsawNet(window_size=10, n_tiles=4, max_epochs=60)
#     model.fit(X_train, X_valid=X_valid)
#     Z = model.transform(X_valid)                    # decode THIS
#     print(model.evaluate_pretext(X_valid))
#     print(model.evaluate_decoding(X_train, Y_train, X_valid, Y_valid))

# THE THREE TASKS (all cross-entropy)
# -----------------------------------
# 1. ORDER (the jigsaw). K intact, chronological, nonoverlapping tiles are cut
#    from one span. Two readouts share a permutation-equivariant DeepSets token:
#      - position CE : K-way softmax per tile, "which slot in time is this?".
#        Chance 1/K. Scales to any K, unlike a K!-way label which explodes and is
#        memorizable (24 classes is a lookup table; 4 classes x 4 tiles is not).
#      - pair BCE    : for every ordered pair, "did i come before j?". Chance 50%,
#        which is a far higher-powered test than 1/24 = 4.17%.
#    The position head is DECODED WITH AN ASSIGNMENT, not an argmax -- see below.
# 2. FORECAST (why this one exists). From each tile's embedding, predict the
#    QUANTIZED activity of the bin immediately AFTER that tile, per neuron, with
#    cross-entropy over count bins. The target bin is always inside the discarded
#    gap, so it never leaks into another tile.
# 3. RECONSTRUCT (the anti-collapse anchor). From each tile's embedding, predict
#    the quantized activity of EVERY bin of that same tile. Input is augmented,
#    target is clean, so it is a denoising task. lambda_reconstruct=0 removes it.

# Task 2 is load-bearing. A pure order task gives the encoder no reason to keep
# absolute firing rate, and absolute rate is usually the dominant behavior-coding
# feature; that is how pretext training ends up BELOW a random encoder. Forecasting
# quantized counts cannot be solved without representing level, so it pins rate
# information into the trunk while the order task carves temporal structure.

# Task 3 exists because task 2 turned out to be too weak a constraint on real
# data: the order objective drove the embedding's participation ratio to 3.4 out
# of 64 dimensions and ridge R2 to 0.005. One predicted bin does not stop a
# 64-dimensional code from collapsing; all W bins does. The decoding ceiling is
# ridge on the raw window, so requiring the embedding to reproduce the raw window
# attacks the measured gap head on instead of a proxy for it.

# READING THE ORDER HEAD. argmax over the K-way position logits is NOT a
# permutation: it can put two tiles in the same slot, and an UNTRAINED head puts
# all of them in one slot because the class biases swamp the tile-dependent part
# of the logits. Scoring those ties as errors moves the chance level of pair
# accuracy from 50% to (1 - 1/K)/2 = 37.5% at K=4, and to 0% for the degenerate
# case -- so a frozen random encoder scores 0.00% and looks catastrophically
# anti-correlated when it is simply undefined. This module therefore decodes with
# the best-scoring permutation (`_assign`), gives ties half credit, and reports
# `argmax_tie_rate_percent` so a degenerate readout is visible.

# NO L2 NORMALIZATION BY DEFAULT. normalize=True projects the embedding onto a
# hypersphere and throws away magnitude. If ridge on the raw window beats your
# embedding, this is the first thing to check. Left available as an ablation only.

# EPOCHS. Spans overlap by construction, so the effective sample size is far below
# the span count and a large epoch budget memorizes the pretext task. Sweep
# max_epochs (10 / 30 / 100 / 300 / 1000) before concluding anything about an arm;
# fit prints a warning when max_epochs x n_spans gets large.

# SHORTCUTS. Tile order is recoverable from slow drift in overall level, so the
# order head can become a nonstationarity detector that does not transfer. The
# order branch therefore sees per-tile normalized tiles (tile_norm) plus
# independent per-tile augmentation and random gaps. The forecast branch sees the
# RAW view, because it is supposed to see level. evaluate_pretext always reports
# the trivial "sort by mean activity" baseline next to the model.

# TRUNK. trunk_block="residual" is a valid-convolution residual stack with
# receptive field exactly window_size. trunk_block="separable" is a
# MobileNetV2-style inverted residual (1x1 expand, GELU, DEPTHWISE 3-tap temporal
# conv, GELU, 1x1 project): each neuron gets its own temporal filter and mixing
# across neurons happens only in the pointwise layers, the decomposition EEGNet
# uses. It is not a shortcut fix -- a linearly available shortcut stays available
# -- but it is cheap and a fair ablation. Both blocks consume exactly two bins.

# CONTROLS (all from the constructor)
#   max_epochs=0        -> frozen random-encoder control. Your embedding must beat
#                          this or training is destroying information.
#   lambda_order=0.0    -> forecast + reconstruct only
#   lambda_forecast=0.0 -> drop the next-bin term
#   lambda_reconstruct=0 -> the pre-anchor model (reproduces the collapse)
#   order_grad_scale=0  -> order head trains on a trunk it cannot influence; asks
#                          "is order decodable?" with zero risk to R2
#   normalize=True      -> the sphere ablation
# Always report pretext accuracy AND the mean-sort baseline AND decoding R2 AND
# ridge on the raw window. Pretext accuracy alone proves nothing.

# ALIGNMENT. transform uses natural windows, no augmentation, no tile
# normalization, no masking. pad=False aligns the window starting at j with label
# j + window_size//2. pad=True edge-pads and returns one row per input bin.

# Split recordings BEFORE fit: a 2D array is one continuous recording; pass a list
# of trial arrays to avoid crossing trials. fit always resets weights.

# Self-test:  python jigsaw_net.py --self-test
# Demo:       python jigsaw_net.py --demo
# """
# import argparse
# import copy
# import itertools
# import math
# import numbers
# from pathlib import Path

# import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional as F

# _EPS = 1e-8
# TILE_NORMS = ("none", "mean", "zscore", "global_mean")
# TRUNK_BLOCKS = ("residual", "separable")


# # --------------------------------------------------------------------------- #
# # validation
# # --------------------------------------------------------------------------- #
# def _integer(name, value, minimum=1, maximum=None):
#     if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
#         raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}.")
#     if maximum is not None and value > maximum:
#         raise ValueError(f"{name} must be <= {maximum}; got {value!r}.")
#     return int(value)


# def _real(name, value, minimum, maximum=None, strict_min=False):
#     if isinstance(value, bool) or not isinstance(value, numbers.Real):
#         raise ValueError(f"{name} must be a real number; got {value!r}.")
#     value = float(value)
#     if (not math.isfinite(value) or value < minimum or (strict_min and value == minimum)
#             or (maximum is not None and value > maximum)):
#         raise ValueError(f"Invalid {name}: {value}.")
#     return value


# def _boolean(name, value):
#     if not isinstance(value, bool):
#         raise ValueError(f"{name} must be a bool; got {value!r}.")
#     return value


# def _sequences(X, minimum_length, n_features=None):
#     is_list = isinstance(X, (list, tuple))
#     items = list(X) if is_list else [X]
#     if not items:
#         raise ValueError("X is empty.")
#     out = []
#     for i, item in enumerate(items):
#         array = np.ascontiguousarray(np.asarray(item, dtype=np.float32))
#         if array.ndim != 2:
#             raise ValueError(f"Sequence {i} must be 2D (time, neurons); got {array.shape}.")
#         if not np.isfinite(array).all():
#             raise ValueError(f"Sequence {i} contains NaN or Inf.")
#         if len(array) < minimum_length:
#             raise ValueError(f"Sequence {i} has {len(array)} bins; needs >= {minimum_length}.")
#         if n_features is not None and array.shape[1] != n_features:
#             raise ValueError(f"Sequence {i} has {array.shape[1]} neurons; expected {n_features}.")
#         out.append(array)
#     if n_features is None and len({a.shape[1] for a in out}) != 1:
#         raise ValueError("All sequences must have the same number of neurons.")
#     return out, is_list


# # --------------------------------------------------------------------------- #
# # modules
# # --------------------------------------------------------------------------- #
# class _GradScale(torch.autograd.Function):
#     """Straight-through forward, scaled gradient backward."""

#     @staticmethod
#     def forward(ctx, x, scale):
#         ctx.scale = float(scale)
#         return x

#     @staticmethod
#     def backward(ctx, grad):
#         return grad * ctx.scale, None


# class _Residual(nn.Module):
#     """Valid 3-tap residual block; consumes exactly two bins."""

#     def __init__(self, width, dropout):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
#             nn.Conv1d(width, width, 3), nn.GELU(),
#             nn.Conv1d(width, width, 1),
#         )

#     def forward(self, x):
#         return x[..., 1:-1] + self.net(x)


# class _Separable(nn.Module):
#     """MobileNetV2-style inverted residual with a VALID depthwise 3-tap conv.

#     1x1 expand -> GELU -> depthwise temporal conv -> GELU -> 1x1 linear project.
#     Consumes exactly two bins, like _Residual, so trunk arithmetic is unchanged.
#     """

#     def __init__(self, width, dropout, expansion=4):
#         super().__init__()
#         hidden = width * expansion
#         self.net = nn.Sequential(
#             nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
#             nn.Conv1d(width, hidden, 1), nn.GELU(),
#             nn.Conv1d(hidden, hidden, 3, groups=hidden), nn.GELU(),
#             nn.Conv1d(hidden, width, 1),
#         )

#     def forward(self, x):
#         return x[..., 1:-1] + self.net(x)


# class _Trunk(nn.Module):
#     """Valid-convolution stack whose receptive field is exactly window_size."""

#     def __init__(self, channels, window_size, width, dropout, block="residual"):
#         super().__init__()
#         if block not in TRUNK_BLOCKS:
#             raise ValueError(f"trunk_block must be one of {TRUNK_BLOCKS}.")
#         first_kernel = 2 if window_size % 2 == 0 else 3
#         blocks = (window_size - first_kernel - 2) // 2
#         make = _Residual if block == "residual" else _Separable
#         self.layers = nn.Sequential(
#             nn.Conv1d(channels, width, first_kernel),
#             nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(), nn.GELU(),
#             *[make(width, dropout) for _ in range(blocks)],
#             nn.Conv1d(width, width, 3), nn.GELU(),
#         )

#     def forward(self, x):
#         h = self.layers(x)
#         if h.shape[-1] != 1:
#             raise ValueError("Trunk expects exactly window_size input bins.")
#         return h.squeeze(-1)


# class _Encoder(nn.Module):
#     def __init__(self, channels, window_size, width, output_dimension, dropout,
#                  normalize, trunk_block):
#         super().__init__()
#         self.trunk = _Trunk(channels, window_size, width, dropout, trunk_block)
#         self.project = nn.Linear(width, output_dimension)
#         self.normalize = normalize

#     def forward(self, x):
#         z = self.project(self.trunk(x))
#         return F.normalize(z, dim=-1) if self.normalize else z


# class _OrderHead(nn.Module):
#     """Permutation-equivariant DeepSets head. Two cross-entropy readouts.

#     position : K-way logits per tile, "which time slot is this tile?"
#     pair     : one scalar per tile; their difference is the before/after logit.
#     """

#     def __init__(self, dimension, hidden, n_tiles):
#         super().__init__()
#         self.body = nn.Sequential(nn.Linear(2 * dimension, hidden), nn.GELU(),
#                                   nn.Linear(hidden, hidden), nn.GELU())
#         self.position = nn.Linear(hidden, n_tiles)
#         self.score = nn.Linear(hidden, 1)

#     def forward(self, z):
#         context = z.mean(dim=1, keepdim=True).expand_as(z)
#         tokens = self.body(torch.cat((z, context), dim=-1))
#         return self.position(tokens), self.score(tokens).squeeze(-1)


# class _ForecastHead(nn.Module):
#     """Per-neuron cross-entropy over quantized activity of the next bin."""

#     def __init__(self, dimension, hidden, channels, levels):
#         super().__init__()
#         self.channels, self.levels = channels, levels
#         self.net = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
#                                  nn.Linear(hidden, channels * levels))

#     def forward(self, z):
#         return self.net(z).reshape(-1, self.channels, self.levels)


# class _ReconstructHead(nn.Module):
#     """Per-neuron, per-bin cross-entropy over the quantized tile the embedding came from.

#     The anti-collapse anchor. The order task only needs a low-dimensional
#     "where in time am I" code and will happily throw everything else away --
#     measured as a participation ratio of 3.4 out of 64 and ridge R2 of 0.005.
#     Forecasting one bin is too weak a constraint to stop that. Requiring the
#     embedding to reproduce EVERY bin of its own window forces it to stay an
#     information-preserving bottleneck, and the raw window is exactly what the
#     decoding ceiling is computed from, so this term targets the measured gap
#     rather than a proxy for it. Still nothing but cross-entropy.
#     """

#     def __init__(self, dimension, hidden, channels, window_size, levels):
#         super().__init__()
#         self.channels, self.window_size, self.levels = channels, window_size, levels
#         self.net = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
#                                  nn.Linear(hidden, channels * window_size * levels))

#     def forward(self, z):
#         return self.net(z).reshape(-1, self.channels, self.window_size, self.levels)


# # --------------------------------------------------------------------------- #
# # functional pieces
# # --------------------------------------------------------------------------- #
# def _gather_tiles(data, starts, gaps, window_size):
#     """(B,K,N,W) tiles plus the index of the bin right after each tile."""
#     offsets = torch.cat((torch.zeros_like(gaps[:, :1]),
#                          torch.cumsum(gaps + window_size, dim=1)), dim=1)
#     tile_starts = starts[:, None] + offsets
#     index = tile_starts[:, :, None] + torch.arange(window_size, device=data.device)
#     return data[index].permute(0, 1, 3, 2), tile_starts + window_size


# def _augment(tiles, neuron_dropout, gain_jitter, generator):
#     if neuron_dropout > 0:
#         keep = (torch.rand(tiles.shape[:3], device=tiles.device, generator=generator)
#                 >= neuron_dropout).to(tiles.dtype)
#         tiles = tiles * keep[..., None]
#     if gain_jitter > 0:
#         gain = 1.0 + gain_jitter * (2 * torch.rand(tiles.shape[:3], device=tiles.device,
#                                                    generator=generator) - 1)
#         tiles = tiles * gain[..., None]
#     return tiles


# def _tile_normalize(tiles, mode):
#     if mode == "none":
#         return tiles
#     if mode == "mean":
#         return tiles - tiles.mean(dim=3, keepdim=True)
#     if mode == "global_mean":
#         return tiles - tiles.mean(dim=(2, 3), keepdim=True)
#     if mode == "zscore":
#         return (tiles - tiles.mean(dim=3, keepdim=True)) / (tiles.std(dim=3, keepdim=True) + 1e-5)
#     raise ValueError(f"tile_norm must be one of {TILE_NORMS}.")


# def _pair_targets(positions):
#     """Upper-triangular before/after labels and the mask of valid pairs."""
#     difference = positions[:, :, None] - positions[:, None, :]
#     mask = torch.triu(torch.ones_like(difference, dtype=torch.bool), diagonal=1)
#     return (difference < 0).to(torch.float32), mask


# def _order_losses(position_logits, scores, positions):
#     """Both readouts, pure cross-entropy."""
#     n_tiles = position_logits.shape[1]
#     position_loss = F.cross_entropy(position_logits.reshape(-1, n_tiles), positions.reshape(-1))
#     target, mask = _pair_targets(positions)
#     difference = scores[:, :, None] - scores[:, None, :]
#     pair_loss = F.binary_cross_entropy_with_logits(difference[mask], target[mask])
#     return position_loss, pair_loss


# def _ranks(values):
#     return values.argsort(dim=1).argsort(dim=1)


# def _assign(position_logits):
#     """Decode the K-way position head into an actual PERMUTATION.

#     argmax is not a permutation: it can drop two tiles into the same slot, and
#     an untrained head drops ALL of them into one slot because the class biases
#     dominate the tile-dependent part of the logits. Reading the head correctly
#     means taking the permutation with the highest total log-probability, which
#     is what a jigsaw solver outputs. K! is tiny here, so brute force is exact;
#     above K=6 fall back to ranking the tiles by their own argmax slot.
#     """
#     n_tiles = position_logits.shape[1]
#     if n_tiles > 6:
#         return _ranks(position_logits.argmax(-1).to(torch.float32))
#     table = torch.tensor(list(itertools.permutations(range(n_tiles))),
#                          device=position_logits.device)                     # (P, K)
#     log_probability = F.log_softmax(position_logits.to(torch.float32), dim=-1)
#     slots = table.transpose(0, 1)                                           # (K, P)
#     total = sum(log_probability[:, tile, slots[tile]] for tile in range(n_tiles))
#     return table[total.argmax(dim=1)]                                       # (B, K)


# def _order_metrics(predicted, positions, *, tie_credit=0.5):
#     """Exact accuracy, pair accuracy, and the TIE RATE.

#     Ties are not a detail. If a pair of tiles is predicted into the same slot
#     there is no implied order, and scoring that as an error silently drags the
#     chance level of pair accuracy from 50% down to (1 - 1/K)/2 -- 37.5% at K=4
#     -- and all the way to 0% for a head whose argmax is constant across tiles.
#     Half credit puts chance back at exactly 50% for every predictor, because
#     (1 - 1/K)/2 + (1/2)(1/K) = 1/2, and the tie rate is returned so that a
#     degenerate readout shows up as a tie rate of 100% instead of masquerading
#     as a below-chance score.
#     """
#     exact = (predicted == positions).all(dim=1).to(torch.float32)
#     truth = torch.sign(positions[:, :, None] - positions[:, None, :])
#     guess = torch.sign(predicted[:, :, None] - predicted[:, None, :])
#     mask = truth != 0
#     denominator = mask.sum((1, 2)).clamp_min(1).to(torch.float32)
#     hits = ((guess == truth) & mask).sum((1, 2)).to(torch.float32)
#     ties = ((guess == 0) & mask).sum((1, 2)).to(torch.float32)
#     return exact, (hits + tie_credit * ties) / denominator, ties / denominator


# def _participation_ratio(features):
#     centered = features - features.mean(0, keepdims=True)
#     values = np.linalg.eigvalsh(np.cov(centered, rowvar=False)
#                                 + 1e-12 * np.eye(centered.shape[1]))
#     values = np.clip(values, 0, None)
#     total = values.sum()
#     return float(total ** 2 / np.square(values).sum()) if total > 0 else 0.0


# def _ridge_r2(train_features, train_targets, test_features, test_targets, alphas):
#     """Closed-form ridge with a CHRONOLOGICAL internal split for alpha selection."""
#     mean = train_features.mean(0, keepdims=True)
#     scale = train_features.std(0, keepdims=True) + _EPS
#     a_train = (train_features - mean) / scale
#     a_test = (test_features - mean) / scale
#     cut = max(1, int(0.8 * len(a_train)))
#     inner_x, inner_y, hold_x, hold_y = a_train[:cut], train_targets[:cut], \
#         a_train[cut:], train_targets[cut:]
#     if len(hold_x) < 2:
#         inner_x, inner_y, hold_x, hold_y = a_train, train_targets, a_train, train_targets

#     def solve(x, y, alpha):
#         offset = y.mean(0, keepdims=True)
#         return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - offset)), offset

#     def r2(y_true, y_hat):
#         residual = np.square(y_true - y_hat).sum(0)
#         total = np.square(y_true - y_true.mean(0, keepdims=True)).sum(0)
#         return 1.0 - residual / np.maximum(total, 1e-12)

#     scored = [(float(np.mean(r2(hold_y, hold_x @ w + b))), alpha)
#               for alpha in alphas for w, b in [solve(inner_x, inner_y, alpha)]]
#     best = max(scored)[1]
#     weights, offset = solve(a_train, train_targets, best)
#     per_dimension = r2(test_targets, a_test @ weights + offset)
#     return dict(r2=float(np.mean(per_dimension)), r2_per_dimension=per_dimension.tolist(),
#                 alpha=float(best), participation_ratio=_participation_ratio(train_features))


# # --------------------------------------------------------------------------- #
# # estimator
# # --------------------------------------------------------------------------- #
# class JigsawNet:
#     """Cross-entropy-only self-supervised encoder: temporal order + forecasting."""

#     _MODEL_TYPE = "jigsaw_net"
#     _PARAM_NAMES = ("window_size", "n_tiles", "tile_gap", "output_dimension",
#                     "num_hidden_units", "head_hidden_units", "dropout", "normalize",
#                     "trunk_block", "lambda_order", "lambda_pair", "lambda_forecast",
#                     "lambda_reconstruct", "forecast_levels", "order_grad_scale",
#                     "tile_norm", "neuron_dropout",
#                     "gain_jitter", "batch_size", "max_epochs", "learning_rate",
#                     "weight_decay", "device", "random_state", "verbose", "log_every")

#     def __init__(self, window_size=10, n_tiles=4, tile_gap=(1, 8),
#                  output_dimension=64, num_hidden_units=64, head_hidden_units=64,
#                  dropout=0.0, normalize=False, trunk_block="residual",
#                  lambda_order=1.0, lambda_pair=0.5, lambda_forecast=1.0,
#                  lambda_reconstruct=1.0,
#                  forecast_levels=8, order_grad_scale=1.0,
#                  tile_norm="mean", neuron_dropout=0.1, gain_jitter=0.1,
#                  batch_size=512, max_epochs=60, learning_rate=1e-3, weight_decay=0.0,
#                  device="cuda_if_available", random_state=42, verbose=True, log_every=1):
#         self.window_size = _integer("window_size", window_size, 4)
#         self.n_tiles = _integer("n_tiles", n_tiles, 2, 8)
#         if not (isinstance(tile_gap, (tuple, list)) and len(tile_gap) == 2):
#             raise ValueError("tile_gap must be a (minimum, maximum) pair.")
#         low = _integer("tile_gap[0]", tile_gap[0], 1)
#         high = _integer("tile_gap[1]", tile_gap[1], low)
#         self.tile_gap = (low, high)
#         self.output_dimension = _integer("output_dimension", output_dimension, 2)
#         self.num_hidden_units = _integer("num_hidden_units", num_hidden_units, 2)
#         self.head_hidden_units = _integer("head_hidden_units", head_hidden_units, 2)
#         self.dropout = _real("dropout", dropout, 0, 1)
#         self.normalize = _boolean("normalize", normalize)
#         if trunk_block not in TRUNK_BLOCKS:
#             raise ValueError(f"trunk_block must be one of {TRUNK_BLOCKS}.")
#         self.trunk_block = trunk_block
#         self.lambda_order = _real("lambda_order", lambda_order, 0)
#         self.lambda_pair = _real("lambda_pair", lambda_pair, 0)
#         self.lambda_forecast = _real("lambda_forecast", lambda_forecast, 0)
#         self.lambda_reconstruct = _real("lambda_reconstruct", lambda_reconstruct, 0)
#         self.forecast_levels = _integer("forecast_levels", forecast_levels, 2, 64)
#         self.order_grad_scale = _real("order_grad_scale", order_grad_scale, 0)
#         if tile_norm not in TILE_NORMS:
#             raise ValueError(f"tile_norm must be one of {TILE_NORMS}.")
#         self.tile_norm = tile_norm
#         self.neuron_dropout = _real("neuron_dropout", neuron_dropout, 0, 1)
#         self.gain_jitter = _real("gain_jitter", gain_jitter, 0)
#         self.batch_size = _integer("batch_size", batch_size, 2)
#         self.max_epochs = _integer("max_epochs", max_epochs, 0)
#         self.learning_rate = _real("learning_rate", learning_rate, 0, strict_min=True)
#         self.weight_decay = _real("weight_decay", weight_decay, 0)
#         self.device = device
#         self.random_state = _integer("random_state", random_state, 0)
#         self.verbose = _boolean("verbose", verbose)
#         self.log_every = _integer("log_every", log_every, 1)
#         if self.lambda_order == 0 and self.lambda_pair == 0 and self.lambda_forecast == 0 \
#                 and self.lambda_reconstruct == 0 and self.max_epochs > 0:
#             raise ValueError("All loss weights are zero; set max_epochs=0 for a random encoder.")
#         self.is_fitted_ = False

#     # -- properties -------------------------------------------------------- #
#     @property
#     def training_span(self):
#         """Bins consumed by one training example (+1 for the forecast target)."""
#         return self.n_tiles * self.window_size + (self.n_tiles - 1) * self.tile_gap[1] + 1

#     def get_params(self):
#         return {name: getattr(self, name) for name in self._PARAM_NAMES}

#     def _check_fitted(self):
#         if not self.is_fitted_:
#             raise RuntimeError("Call fit before using this model.")

#     # -- setup ------------------------------------------------------------- #
#     def _resolve_device(self):
#         name = self.device
#         if name == "cuda_if_available":
#             name = "cuda" if torch.cuda.is_available() else "cpu"
#         return torch.device(name)

#     def _make_encoder(self, channels):
#         """Override to swap the trunk. Must map (B, channels, window_size) -> (B, D)."""
#         return _Encoder(channels, self.window_size, self.num_hidden_units,
#                         self.output_dimension, self.dropout, self.normalize,
#                         self.trunk_block)

#     def _build(self, channels):
#         torch.manual_seed(self.random_state)
#         self.device_ = self._resolve_device()
#         self.n_features_in_ = channels
#         self.encoder_ = self._make_encoder(channels).to(self.device_)
#         self.order_head_ = _OrderHead(self.output_dimension, self.head_hidden_units,
#                                       self.n_tiles).to(self.device_)
#         self.forecast_head_ = _ForecastHead(self.output_dimension, self.head_hidden_units,
#                                             channels, self.forecast_levels).to(self.device_)
#         self.reconstruct_head_ = _ReconstructHead(
#             self.output_dimension, self.head_hidden_units, channels, self.window_size,
#             self.forecast_levels).to(self.device_)
#         self._generator = torch.Generator(device=self.device_)
#         self._generator.manual_seed(self.random_state + 1)

#     def _quantize_edges(self, data):
#         """Per-neuron quantile edges for the forecast target.

#         Quantiles, not equal-width bins: spike counts are heavily skewed, and
#         equal-width bins would put almost every target in class 0, making the
#         cross-entropy trivially minimized by predicting the mode.
#         """
#         probabilities = np.linspace(0, 1, self.forecast_levels + 1)[1:-1]
#         edges = np.quantile(data, probabilities, axis=0).astype(np.float32)
#         edges = np.maximum.accumulate(edges, axis=0)  # keep monotone under ties
#         return torch.from_numpy(np.ascontiguousarray(edges)).to(self.device_)

#     def _spans(self, X, n_features=None):
#         sequences, _ = _sequences(X, self.training_span, n_features)
#         lengths = [len(s) for s in sequences]
#         offsets = np.concatenate([[0], np.cumsum(lengths)])
#         starts = np.concatenate([offsets[i] + np.arange(n - self.training_span + 1)
#                                  for i, n in enumerate(lengths)]) if lengths else np.empty(0, int)
#         data = torch.from_numpy(np.concatenate(sequences, axis=0)).to(self.device_)
#         return data, starts.astype(np.int64), sequences[0].shape[1]

#     def _draw(self, count):
#         low, high = self.tile_gap
#         return torch.randint(low, high + 1, (count, self.n_tiles - 1),
#                              device=self.device_, generator=self._generator)

#     # -- one step ---------------------------------------------------------- #
#     def _step(self, data, starts):
#         batch = len(starts)
#         tiles, next_index = _gather_tiles(data, starts, self._draw(batch), self.window_size)
#         positions = torch.arange(self.n_tiles, device=self.device_).expand(batch, -1)
#         parts = {}
#         total = torch.zeros((), device=self.device_)

#         # FORECAST: raw view, so the trunk must represent absolute level.
#         raw = _augment(tiles, self.neuron_dropout, self.gain_jitter, self._generator)
#         z_raw = self.encoder_(raw.reshape(-1, self.n_features_in_, self.window_size))
#         if self.lambda_forecast > 0:
#             target = self._bucketize(data[next_index.reshape(-1)])
#             logits = self.forecast_head_(z_raw)
#             forecast = F.cross_entropy(logits.reshape(-1, self.forecast_levels),
#                                        target.reshape(-1))
#             with torch.no_grad():
#                 parts["forecast_accuracy"] = (logits.argmax(-1) == target).to(
#                     torch.float32).mean()
#             parts["forecast"] = forecast
#             total = total + self.lambda_forecast * forecast
#         else:
#             parts["forecast"] = torch.zeros((), device=self.device_)
#             parts["forecast_accuracy"] = torch.zeros((), device=self.device_)

#         # RECONSTRUCT: same raw embedding must reproduce every bin of its own tile.
#         # Input is augmented, target is the CLEAN tile, so this is a denoising task.
#         if self.lambda_reconstruct > 0:
#             flat = tiles.permute(0, 1, 3, 2).reshape(-1, self.n_features_in_)
#             target = self._bucketize(flat).reshape(-1, self.window_size,
#                                                    self.n_features_in_).permute(0, 2, 1)
#             logits = self.reconstruct_head_(z_raw)
#             reconstruct = F.cross_entropy(logits.reshape(-1, self.forecast_levels),
#                                           target.reshape(-1))
#             with torch.no_grad():
#                 parts["reconstruct_accuracy"] = (logits.argmax(-1) == target).to(
#                     torch.float32).mean()
#             parts["reconstruct"] = reconstruct
#             total = total + self.lambda_reconstruct * reconstruct
#         else:
#             parts["reconstruct"] = torch.zeros((), device=self.device_)
#             parts["reconstruct_accuracy"] = torch.zeros((), device=self.device_)

#         # ORDER: normalized view, so level drift cannot give the answer away.
#         if self.lambda_order > 0 or self.lambda_pair > 0:
#             view = _tile_normalize(
#                 _augment(tiles, self.neuron_dropout, self.gain_jitter, self._generator),
#                 self.tile_norm)
#             z_order = self.encoder_(view.reshape(-1, self.n_features_in_, self.window_size))
#             z_order = _GradScale.apply(z_order, self.order_grad_scale)
#             position_logits, scores = self.order_head_(
#                 z_order.reshape(batch, self.n_tiles, self.output_dimension))
#             position_loss, pair_loss = _order_losses(position_logits, scores, positions)
#             parts["position"], parts["pair"] = position_loss, pair_loss
#             total = total + self.lambda_order * position_loss + self.lambda_pair * pair_loss
#             with torch.no_grad():
#                 predicted = _ranks(scores) if self.lambda_order == 0 \
#                     else _assign(position_logits)
#                 exact, pair_accuracy, tie_rate = _order_metrics(predicted, positions)
#                 parts["exact"], parts["pair_accuracy"] = exact.mean(), pair_accuracy.mean()
#                 parts["tie_rate"] = tie_rate.mean()
#         else:
#             for key in ("position", "pair", "exact", "pair_accuracy", "tie_rate"):
#                 parts[key] = torch.zeros((), device=self.device_)
#         parts["total"] = total
#         return parts

#     def _bucketize(self, values):
#         """(M,N) activity -> (M,N) integer class per neuron, using stored edges."""
#         out = torch.zeros(values.shape, dtype=torch.long, device=values.device)
#         for level in range(self.forecast_levels - 1):
#             out = out + (values > self.forecast_edges_[level][None, :]).to(torch.long)
#         return out

#     # -- fit --------------------------------------------------------------- #
#     def fit(self, X, X_valid=None, *, validate_every=1):
#         validate_every = _integer("validate_every", validate_every)
#         first, _ = _sequences(X, self.training_span)
#         self._build(first[0].shape[1])
#         data, starts, channels = self._spans(X, first[0].shape[1])
#         self.forecast_edges_ = self._quantize_edges(
#             np.concatenate(first, axis=0)) if self.forecast_levels > 1 else None
#         self.n_spans_ = int(len(starts))
#         if self.n_spans_ < 2 and self.max_epochs > 0:
#             raise ValueError(
#                 f"Only {self.n_spans_} valid span start(s). Each sequence needs at least "
#                 f"training_span+1={self.training_span + 1} bins, or reduce window_size / "
#                 f"n_tiles / tile_gap.")
#         self.is_fitted_ = True
#         self.history_, self.validation_history_ = [], []
#         if self.max_epochs == 0:
#             if self.verbose:
#                 print(f"JigsawNet: max_epochs=0 -> frozen random encoder "
#                       f"({self.n_spans_} spans, {channels} neurons).", flush=True)
#             return self

#         parameters = list(self.encoder_.parameters()) + list(self.order_head_.parameters()) \
#             + list(self.forecast_head_.parameters()) + list(self.reconstruct_head_.parameters())
#         optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate,
#                                       weight_decay=self.weight_decay)
#         if self.verbose:
#             print(f"JigsawNet: {self.n_spans_} spans, window={self.window_size}, "
#                   f"K={self.n_tiles}, dim={self.output_dimension}, trunk={self.trunk_block}, "
#                   f"normalize={self.normalize}, tile_norm={self.tile_norm}, "
#                   f"lambda(order,pair,forecast,reconstruct)="
#                   f"({self.lambda_order},{self.lambda_pair},{self.lambda_forecast},"
#                   f"{self.lambda_reconstruct}), device={self.device_}", flush=True)
#             if self.max_epochs * self.n_spans_ > 2_000_000:
#                 print(f"      WARNING: {self.max_epochs} epochs x {self.n_spans_} overlapping "
#                       f"spans. Spans overlap by construction, so the effective sample size is "
#                       f"far below the span count and this budget memorizes the pretext task. "
#                       f"Sweep max_epochs before trusting any arm.", flush=True)

#         index = torch.from_numpy(starts).to(self.device_)
#         for epoch in range(self.max_epochs):
#             order = torch.randperm(self.n_spans_, device=self.device_, generator=self._generator)
#             totals, seen = {}, 0
#             self.encoder_.train(); self.order_head_.train(); self.forecast_head_.train()
#             self.reconstruct_head_.train()
#             for begin in range(0, self.n_spans_, self.batch_size):
#                 chunk = index[order[begin:begin + self.batch_size]]
#                 if len(chunk) < 2:
#                     continue
#                 parts = self._step(data, chunk)
#                 if not torch.isfinite(parts["total"]):
#                     raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}.")
#                 optimizer.zero_grad(set_to_none=True)
#                 parts["total"].backward()
#                 nn.utils.clip_grad_norm_(parameters, 5.0)
#                 optimizer.step()
#                 for key, value in parts.items():
#                     totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(chunk)
#                 seen += len(chunk)
#             row = {"epoch": epoch + 1, **{k: v / max(seen, 1) for k, v in totals.items()}}
#             self.history_.append(row)
#             if self.verbose and ((epoch + 1) % self.log_every == 0 or epoch + 1 == self.max_epochs):
#                 print(f"  epoch {epoch + 1}/{self.max_epochs} loss={row['total']:.4f} "
#                       f"| position CE={row['position']:.4f} pair BCE={row['pair']:.4f} "
#                       f"forecast CE={row['forecast']:.4f} "
#                       f"reconstruct CE={row['reconstruct']:.4f} "
#                       f"| train exact={100 * row['exact']:.1f}% "
#                       f"pair={100 * row['pair_accuracy']:.1f}% "
#                       f"ties={100 * row['tie_rate']:.1f}% "
#                       f"forecast acc={100 * row['forecast_accuracy']:.1f}% "
#                       f"recon acc={100 * row['reconstruct_accuracy']:.1f}%", flush=True)
#             if X_valid is not None and ((epoch + 1) % validate_every == 0
#                                         or epoch + 1 == self.max_epochs):
#                 metrics = self.evaluate_pretext(X_valid, max_spans=512, verbose=False)
#                 metrics["epoch"] = epoch + 1
#                 self.validation_history_.append(metrics)
#                 if self.verbose:
#                     print(f"      valid: exact={metrics['exact_accuracy_percent']:.2f}% "
#                           f"(chance {metrics['chance_exact_percent']:.2f}%) "
#                           f"pair={metrics['pair_accuracy_percent']:.2f}% "
#                           f"| mean-sort baseline pair={metrics['baseline_mean_sort_pair_percent']:.2f}%",
#                           flush=True)
#         return self

#     # -- inference --------------------------------------------------------- #
#     def transform(self, X, *, pad=True, batch_size=None, return_indices=False):
#         """Natural windows. No augmentation, no tile normalization, no heads."""
#         self._check_fitted()
#         _boolean("pad", pad)
#         size = self.batch_size if batch_size is None else _integer("batch_size", batch_size)
#         sequences, is_list = _sequences(X, 1 if pad else self.window_size, self.n_features_in_)
#         left = self.window_size // 2
#         right = self.window_size - left - 1
#         values, times = [], []
#         self.encoder_.eval()
#         with torch.inference_mode():
#             for sequence in sequences:
#                 if pad:
#                     centers = np.arange(len(sequence), dtype=np.int64)
#                     sequence = np.pad(sequence, ((left, right), (0, 0)), mode="edge")
#                 else:
#                     centers = np.arange(left, len(sequence) - right, dtype=np.int64)
#                 out = np.empty((len(centers), self.output_dimension), dtype=np.float32)
#                 for begin in range(0, len(centers), size):
#                     end = min(begin + size, len(centers))
#                     locations = np.arange(begin, end)[:, None] + np.arange(self.window_size)
#                     windows = np.ascontiguousarray(sequence[locations].transpose(0, 2, 1))
#                     out[begin:end] = self.encoder_(
#                         torch.from_numpy(windows).to(self.device_)).cpu().numpy()
#                 values.append(out)
#                 times.append(centers)
#         self.encoder_.train()
#         result = values if is_list else values[0]
#         index = times if is_list else times[0]
#         return (result, index) if return_indices else result

#     def fit_transform(self, X, **kwargs):
#         return self.fit(X).transform(X, **kwargs)

#     # -- evaluation -------------------------------------------------------- #
#     def evaluate_pretext(self, X, *, max_spans=1024, batch_size=256, random_state=200042,
#                          verbose=None):
#         """Held-out order accuracy next to the trivial shortcut baselines."""
#         self._check_fitted()
#         verbose = self.verbose if verbose is None else verbose
#         data, starts, _ = self._spans(X, self.n_features_in_)
#         if len(starts) == 0:
#             raise ValueError("No valid spans in X.")
#         rng = np.random.default_rng(random_state)
#         chosen = starts if len(starts) <= max_spans else rng.choice(starts, max_spans, False)
#         chosen = torch.from_numpy(np.sort(chosen)).to(self.device_)
#         generator = torch.Generator(device=self.device_).manual_seed(random_state)
#         low, high = self.tile_gap
#         totals = {k: 0.0 for k in ("exact", "pair", "tie", "argmax_pair", "argmax_tie",
#                                    "rank_exact", "rank_pair", "mean_exact", "mean_pair",
#                                    "norm_exact", "norm_pair", "position_ce")}
#         count = 0
#         self.encoder_.eval(); self.order_head_.eval()
#         with torch.inference_mode():
#             for begin in range(0, len(chosen), batch_size):
#                 block = chosen[begin:begin + batch_size]
#                 gaps = torch.randint(low, high + 1, (len(block), self.n_tiles - 1),
#                                      device=self.device_, generator=generator)
#                 tiles, _ = _gather_tiles(data, block, gaps, self.window_size)
#                 positions = torch.arange(self.n_tiles, device=self.device_).expand(len(block), -1)
#                 view = _tile_normalize(tiles, self.tile_norm)
#                 z = self.encoder_(view.reshape(-1, self.n_features_in_, self.window_size))
#                 position_logits, scores = self.order_head_(
#                     z.reshape(len(block), self.n_tiles, self.output_dimension))
#                 # Headline decode: a real permutation, so chance is 1/K! and 50% exactly.
#                 predicted = _assign(position_logits) if self.lambda_order > 0 else _ranks(scores)
#                 exact, pair, tie = _order_metrics(predicted, positions)
#                 # Kept for comparison: the naive argmax, which is NOT a permutation.
#                 _, argmax_pair, argmax_tie = _order_metrics(position_logits.argmax(-1), positions)
#                 rank_exact, rank_pair, _ = _order_metrics(_ranks(scores), positions)
#                 mean_exact, mean_pair, _ = _order_metrics(_ranks(tiles.mean(dim=(2, 3))), positions)
#                 norm_exact, norm_pair, _ = _order_metrics(
#                     _ranks(tiles.reshape(len(block), self.n_tiles, -1).norm(dim=2)), positions)
#                 weight = len(block)
#                 for key, value in (("exact", exact), ("pair", pair), ("tie", tie),
#                                    ("argmax_pair", argmax_pair), ("argmax_tie", argmax_tie),
#                                    ("rank_exact", rank_exact), ("rank_pair", rank_pair),
#                                    ("mean_exact", mean_exact), ("mean_pair", mean_pair),
#                                    ("norm_exact", norm_exact), ("norm_pair", norm_pair)):
#                     totals[key] += float(value.mean()) * weight
#                 totals["position_ce"] += float(F.cross_entropy(
#                     position_logits.reshape(-1, self.n_tiles), positions.reshape(-1))) * weight
#                 count += weight
#         self.encoder_.train(); self.order_head_.train()
#         scale = 100.0 / max(count, 1)
#         result = dict(
#             exact_accuracy_percent=totals["exact"] * scale,
#             pair_accuracy_percent=totals["pair"] * scale,
#             tie_rate_percent=totals["tie"] * scale,
#             argmax_pair_accuracy_percent=totals["argmax_pair"] * scale,
#             argmax_tie_rate_percent=totals["argmax_tie"] * scale,
#             rank_exact_accuracy_percent=totals["rank_exact"] * scale,
#             rank_pair_accuracy_percent=totals["rank_pair"] * scale,
#             baseline_mean_sort_exact_percent=totals["mean_exact"] * scale,
#             baseline_mean_sort_pair_percent=totals["mean_pair"] * scale,
#             baseline_norm_sort_exact_percent=totals["norm_exact"] * scale,
#             baseline_norm_sort_pair_percent=totals["norm_pair"] * scale,
#             position_cross_entropy=totals["position_ce"] / max(count, 1),
#             uniform_cross_entropy=math.log(self.n_tiles),
#             decode="assignment" if self.lambda_order > 0 else "score_rank",
#             chance_exact_percent=100.0 / math.factorial(self.n_tiles),
#             chance_pair_percent=50.0, n_spans=int(count))
#         if verbose:
#             print(f"  pretext: exact={result['exact_accuracy_percent']:.2f}% "
#                   f"(chance {result['chance_exact_percent']:.2f}%) "
#                   f"pair={result['pair_accuracy_percent']:.2f}% (chance 50%) "
#                   f"| decode={result['decode']} "
#                   f"argmax pair={result['argmax_pair_accuracy_percent']:.2f}% "
#                   f"ties={result['argmax_tie_rate_percent']:.1f}% "
#                   f"| mean-sort pair={result['baseline_mean_sort_pair_percent']:.2f}%", flush=True)
#         return result

#     def evaluate_decoding(self, X_train, y_train, X_test, y_test,
#                           *, alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)):
#         """Ridge R2 from the embedding AND from the raw window, side by side.

#         The raw-window number is not decoration. If the embedding does not beat
#         it, the encoder is removing information and that is the finding.
#         """
#         self._check_fitted()
#         out = {}
#         z_train, index_train = self.transform(X_train, pad=False, return_indices=True)
#         z_test, index_test = self.transform(X_test, pad=False, return_indices=True)
#         target_train = np.asarray(y_train, dtype=np.float64)[index_train]
#         target_test = np.asarray(y_test, dtype=np.float64)[index_test]
#         out["embedding"] = _ridge_r2(z_train.astype(np.float64), target_train,
#                                      z_test.astype(np.float64), target_test, alphas)
#         raw_train = np.lib.stride_tricks.sliding_window_view(
#             np.asarray(X_train, dtype=np.float64), self.window_size, axis=0)
#         raw_test = np.lib.stride_tricks.sliding_window_view(
#             np.asarray(X_test, dtype=np.float64), self.window_size, axis=0)
#         out["raw_window"] = _ridge_r2(raw_train.reshape(len(raw_train), -1), target_train,
#                                       raw_test.reshape(len(raw_test), -1), target_test, alphas)
#         out["embedding_minus_raw"] = out["embedding"]["r2"] - out["raw_window"]["r2"]
#         return out

#     # -- persistence ------------------------------------------------------- #
#     def save(self, path):
#         self._check_fitted()
#         torch.save(dict(model_type=self._MODEL_TYPE, format_version=1, params=self.get_params(),
#                         n_features_in=self.n_features_in_,
#                         encoder=self.encoder_.state_dict(),
#                         order_head=self.order_head_.state_dict(),
#                         forecast_head=self.forecast_head_.state_dict(),
#                         reconstruct_head=self.reconstruct_head_.state_dict(),
#                         forecast_edges=None if self.forecast_edges_ is None
#                         else self.forecast_edges_.cpu(),
#                         history=self.history_, validation_history=self.validation_history_),
#                    Path(path))
#         return self

#     @classmethod
#     def load(cls, path, device="cuda_if_available"):
#         try:
#             data = torch.load(Path(path), map_location="cpu", weights_only=False)
#         except TypeError:
#             data = torch.load(Path(path), map_location="cpu")
#         if data.get("model_type") != cls._MODEL_TYPE or data.get("format_version") != 1:
#             raise ValueError("Expected a jigsaw_net checkpoint.")
#         params = dict(data["params"]); params["device"] = device
#         model = cls(**params)
#         model._build(data["n_features_in"])
#         model.encoder_.load_state_dict(data["encoder"])
#         model.order_head_.load_state_dict(data["order_head"])
#         model.forecast_head_.load_state_dict(data["forecast_head"])
#         if "reconstruct_head" in data:
#             model.reconstruct_head_.load_state_dict(data["reconstruct_head"])
#         edges = data.get("forecast_edges")
#         model.forecast_edges_ = None if edges is None else edges.to(model.device_)
#         model.history_ = data["history"]
#         model.validation_history_ = data["validation_history"]
#         model.is_fitted_ = True
#         return model


# Jigsaw = JigsawNet


# # --------------------------------------------------------------------------- #
# # self-test
# # --------------------------------------------------------------------------- #
# def _check(condition, message):
#     if not condition:
#         raise AssertionError(message)
#     print(f"  ok  {message}")


# def _synthetic(n=6000, channels=24, seed=0):
#     """Latent circle -> Poisson-ish rates. Behavior is linearly decodable."""
#     rng = np.random.default_rng(seed)
#     t = np.linspace(0, 60 * np.pi, n)
#     latent = np.stack([np.sin(t), np.cos(t)], axis=1)
#     weights = rng.normal(size=(2, channels))
#     rates = np.exp(0.8 * latent @ weights + 0.4 * rng.normal(size=(n, channels)) - 0.5)
#     return rng.poisson(rates).astype(np.float32), latent.astype(np.float32)


# def _self_test():
#     print("1. trunk receptive field")
#     for window in (4, 5, 9, 10, 16, 21):
#         for block in TRUNK_BLOCKS:
#             trunk = _Trunk(7, window, 8, 0.0, block)
#             out = trunk(torch.zeros(3, 7, window))
#             _check(out.shape == (3, 8), f"{block} trunk, window={window} -> one bin, {out.shape}")
#     depthwise = [m for m in _Separable(8, 0.0).modules()
#                  if isinstance(m, nn.Conv1d) and m.kernel_size == (3,)]
#     _check(len(depthwise) == 1 and depthwise[0].groups == depthwise[0].in_channels,
#            "the separable block's 3-tap convolution is depthwise")

#     print("2. tiles never overlap and the forecast target sits in the gap")
#     ramp = torch.arange(300.0)[:, None].repeat(1, 3)
#     tiles, nxt = _gather_tiles(ramp, torch.tensor([0, 50]), torch.tensor([[1, 2, 3], [1, 1, 1]]), 4)
#     _check(tiles.shape == (2, 4, 3, 4), f"shape (B,K,N,W)={tuple(tiles.shape)}")
#     _check(tiles[0, :, 0, 0].tolist() == [0, 5, 11, 18], "gapped tile starts [0,5,11,18]")
#     _check((tiles[:, 1:, 0, 0] > tiles[:, :-1, 0, -1]).all(), "tiles are disjoint")
#     _check((nxt[:, :-1] < tiles[:, 1:, 0, 0]).all(),
#            "every forecast target bin lies strictly inside a discarded gap")

#     print("3. losses are cross-entropy and hit the right chance levels")
#     batch, K = 64, 4
#     positions = torch.arange(K).expand(batch, -1)
#     position_loss, pair_loss = _order_losses(torch.zeros(batch, K, K), torch.zeros(batch, K),
#                                              positions)
#     _check(abs(float(position_loss) - math.log(K)) < 1e-6,
#            f"uniform position logits cost exactly log(K)={math.log(K):.4f}")
#     _check(abs(float(pair_loss) - math.log(2)) < 1e-6,
#            f"tied pair scores cost exactly log(2)={math.log(2):.4f}")
#     perfect = F.one_hot(positions, K).float() * 20
#     position_loss, _ = _order_losses(perfect, -positions.float(), positions)
#     _check(float(position_loss) < 1e-6, "a correct confident prediction costs ~0")

#     print("3b. the order metric has the chance level it claims to have")
#     big = 200000
#     truth = torch.arange(K).expand(big, -1)
#     constant = torch.randint(0, K, (big, 1)).expand(-1, K)
#     _, pair_half, tie = _order_metrics(constant, truth)
#     _check(abs(float(pair_half.mean()) - 0.5) < 1e-6 and float(tie.mean()) == 1.0,
#            "a head that puts every tile in ONE slot scores exactly 50% with 100% ties "
#            "(scoring ties as errors would call this 0.00% and look anti-correlated)")
#     _, pair_zero, _ = _order_metrics(constant, truth, tie_credit=0.0)
#     _check(float(pair_zero.mean()) == 0.0,
#            "  ...and the tie_credit=0 convention is what produced the 0.00% in run 1")
#     iid = torch.randint(0, K, (big, K))
#     _, pair_iid, tie_iid = _order_metrics(iid, truth)
#     _, pair_iid_zero, _ = _order_metrics(iid, truth, tie_credit=0.0)
#     _check(abs(float(pair_iid.mean()) - 0.5) < 0.005,
#            f"an i.i.d. random argmax scores {100 * float(pair_iid.mean()):.2f}% with half credit")
#     _check(abs(float(pair_iid_zero.mean()) - (1 - 1 / K) / 2) < 0.005,
#            f"  ...but only {100 * float(pair_iid_zero.mean()):.2f}% with tie_credit=0, i.e. the "
#            f"true chance level of the old metric was (1-1/K)/2={100 * (1 - 1 / K) / 2:.1f}%, not 50%")
#     permutation = torch.argsort(torch.rand(big, K), dim=1)
#     _, pair_perm, tie_perm = _order_metrics(permutation, truth)
#     _check(abs(float(pair_perm.mean()) - 0.5) < 0.005 and float(tie_perm.mean()) == 0.0,
#            "a permutation decode has no ties and sits at 50% either way")

#     print("3c. _assign returns a permutation and maximizes total log-probability")
#     logits = torch.randn(256, K, K) * 3
#     assigned = _assign(logits)
#     _check(all(sorted(row.tolist()) == list(range(K)) for row in assigned),
#            "every _assign output is a genuine permutation of the K slots")
#     log_probability = F.log_softmax(logits, dim=-1)
#     chosen_score = log_probability.gather(2, assigned[:, :, None]).squeeze(-1).sum(1)
#     every = torch.tensor(list(itertools.permutations(range(K))))
#     brute = torch.stack([log_probability.gather(
#         2, p[None, :, None].expand(256, -1, -1)).squeeze(-1).sum(1) for p in every])
#     _check(torch.allclose(chosen_score, brute.max(dim=0).values, atol=1e-5),
#            "_assign finds the highest-scoring permutation (checked against all K! by hand)")
#     confident = F.one_hot(torch.arange(K), K).float()[None].expand(64, -1, -1) * 20
#     _check((_assign(confident) == torch.arange(K)).all(),
#            "a confident correct head decodes to the identity permutation")
#     degenerate = torch.zeros(64, K, K); degenerate[:, :, 1] = 10.0
#     _check(all(sorted(row.tolist()) == list(range(K)) for row in _assign(degenerate)),
#            "even when every tile votes for the SAME slot, _assign still returns a permutation")

#     print("4. the head is permutation equivariant")
#     head = _OrderHead(6, 16, K)
#     z = torch.randn(8, K, 6)
#     shuffle = torch.stack([torch.randperm(K) for _ in range(8)])
#     logits_a, scores_a = head(z)
#     logits_b, scores_b = head(z.gather(1, shuffle[:, :, None].expand(-1, -1, 6)))
#     _check(torch.allclose(logits_b, logits_a.gather(1, shuffle[:, :, None].expand(-1, -1, K)),
#                           atol=1e-5), "position logits permute with the tiles")
#     _check(torch.allclose(scores_b, scores_a.gather(1, shuffle), atol=1e-5),
#            "pair scores permute with the tiles")

#     print("5. tile normalization kills the level shortcut")
#     raw = torch.randn(4, K, 5, 8)
#     offset = torch.randn(4, K, 1, 1) * 5
#     for mode in ("mean", "zscore", "global_mean"):
#         _check(torch.allclose(_tile_normalize(raw + offset, mode), _tile_normalize(raw, mode),
#                               atol=1e-3), f"tile_norm={mode} removes per-tile level")
#     ramped = raw + torch.arange(K).float()[None, :, None, None] * 4
#     _check((_ranks(ramped.mean(dim=(2, 3))) == torch.arange(K)).all(),
#            "mean-sort solves the puzzle perfectly under a level ramp (the shortcut)")
#     _check(not (_ranks(_tile_normalize(ramped, "mean").mean(dim=(2, 3))) == torch.arange(K)).all(),
#            "after normalization the ramp no longer gives the order away")

#     print("6. quantization is balanced, not mode-collapsed")
#     data, _ = _synthetic(3000, 12, seed=3)
#     model = JigsawNet(window_size=8, n_tiles=3, tile_gap=(1, 3), output_dimension=8,
#                       num_hidden_units=12, head_hidden_units=12, max_epochs=0,
#                       device="cpu", verbose=False).fit(data)
#     model.forecast_edges_ = model._quantize_edges(data)
#     classes = model._bucketize(torch.from_numpy(data))
#     share = torch.bincount(classes.reshape(-1), minlength=8).float() / classes.numel()
#     _check(int(classes.max()) < model.forecast_levels and int(classes.min()) >= 0,
#            "every target lands inside [0, forecast_levels)")
#     _check(float(share.max()) < 0.75,
#            f"quantile edges avoid mode collapse (largest class share {float(share.max()):.2f}) "
#            "-> the forecast CE is not minimized by always predicting one class")

#     print("6b. the reconstruction target is the tile itself, bin for bin")
#     window, tiles_k, neurons = model.window_size, model.n_tiles, 12
#     tiles = torch.from_numpy(data[:5 * tiles_k * window].reshape(5, tiles_k, window, neurons)
#                              ).permute(0, 1, 3, 2).contiguous()          # (B,K,N,W)
#     flat = tiles.permute(0, 1, 3, 2).reshape(-1, neurons)
#     target = model._bucketize(flat).reshape(-1, window, neurons).permute(0, 2, 1)
#     _check(target.shape == (5 * tiles_k, neurons, window),
#            f"target is (B*K, N, W)={tuple(target.shape)}, matching the head's logit layout")
#     direct = model._bucketize(tiles[0, 0].T).T                           # (N, W), the honest way
#     _check(torch.equal(target[0], direct),
#            "tile (0,0) round-trips: the reshape/permute chain preserves the (neuron, bin) pairing")
#     head = _ReconstructHead(8, 12, neurons, window, model.forecast_levels)
#     logits = head(torch.randn(5 * tiles_k, 8))
#     _check(logits.shape == (5 * tiles_k, neurons, window, model.forecast_levels),
#            f"head logits {tuple(logits.shape)} line up with the target elementwise")
#     _check(logits.reshape(-1, model.forecast_levels).shape[0] == target.reshape(-1).shape[0],
#            "flattening logits and target for cross_entropy keeps them aligned")

#     print("7. end to end: training must not destroy decodable structure")
#     data, latent = _synthetic(7000, 24, seed=1)
#     cut = 5000
#     alphas = (1e-2, 1e-1, 1.0, 10.0, 100.0)
#     shared = dict(window_size=8, n_tiles=3, tile_gap=(1, 3), output_dimension=16,
#                   num_hidden_units=32, head_hidden_units=32, batch_size=256,
#                   learning_rate=3e-3, device="cpu", verbose=False, random_state=0)
#     trained = JigsawNet(max_epochs=12, **shared).fit(data[:cut])
#     random_encoder = JigsawNet(max_epochs=0, **shared).fit(data[:cut])
#     scores = {}
#     for name, fitted in (("trained", trained), ("random", random_encoder)):
#         z_train, i_train = fitted.transform(data[:cut], pad=False, return_indices=True)
#         z_test, i_test = fitted.transform(data[cut:], pad=False, return_indices=True)
#         scores[name] = _ridge_r2(z_train.astype(np.float64), latent[:cut][i_train].astype(np.float64),
#                                  z_test.astype(np.float64), latent[cut:][i_test].astype(np.float64),
#                                  alphas)["r2"]
#     print(f"      trained R2={scores['trained']:.4f}  random-encoder R2={scores['random']:.4f}")
#     _check(scores["trained"] > 0.5, f"trained embedding decodes the latent (R2={scores['trained']:.3f})")
#     _check(scores["trained"] > scores["random"] - 0.02,
#            "training does not fall behind the random-encoder control")
#     pretext = trained.evaluate_pretext(data[cut:], max_spans=400, verbose=False)
#     print(f"      held-out exact={pretext['exact_accuracy_percent']:.1f}% "
#           f"(chance {pretext['chance_exact_percent']:.1f}%) "
#           f"pair={pretext['pair_accuracy_percent']:.1f}%")
#     _check(pretext["pair_accuracy_percent"] > 55, "the order task is learned out of sample")

#     print("8. normalize=True is the ablation that costs R2")
#     spherical = JigsawNet(max_epochs=12, normalize=True, **shared).fit(data[:cut])
#     z_train, i_train = spherical.transform(data[:cut], pad=False, return_indices=True)
#     z_test, i_test = spherical.transform(data[cut:], pad=False, return_indices=True)
#     sphere_r2 = _ridge_r2(z_train.astype(np.float64), latent[:cut][i_train].astype(np.float64),
#                           z_test.astype(np.float64), latent[cut:][i_test].astype(np.float64),
#                           alphas)["r2"]
#     print(f"      normalize=True R2={sphere_r2:.4f} vs normalize=False R2={scores['trained']:.4f}")
#     _check(np.isfinite(sphere_r2), "the spherical ablation runs")

#     print("9. transform is deterministic, aligned and augmentation-free")
#     a = trained.transform(data[cut:], pad=False)
#     b = trained.transform(data[cut:], pad=False)
#     _check(np.allclose(a, b), "transform is deterministic (no augmentation leaks in)")
#     _check(len(a) == len(data[cut:]) - trained.window_size + 1,
#            f"pad=False drops exactly window_size-1={trained.window_size - 1} bins")
#     _check(len(trained.transform(data[cut:], pad=True)) == len(data[cut:]),
#            "pad=True returns one row per input bin")
#     _, index = trained.transform(data[cut:], pad=False, return_indices=True)
#     _check(index[0] == trained.window_size // 2, "pad=False starts at window_size//2")

#     print("10. controls run and bad configs are rejected")
#     for override in ({"lambda_order": 0.0, "lambda_pair": 0.0}, {"lambda_forecast": 0.0},
#                      {"lambda_reconstruct": 0.0}, {"lambda_forecast": 0.0, "lambda_order": 0.0,
#                                                    "lambda_pair": 0.0},
#                      {"order_grad_scale": 0.0}, {"trunk_block": "separable"},
#                      {"tile_norm": "none"}, {"n_tiles": 2}):
#         settings = dict(shared, max_epochs=2); settings.update(override)
#         fitted = JigsawNet(**settings).fit(data[:1500])
#         _check(np.isfinite(fitted.transform(data[:400], pad=False)).all(),
#                f"control {override} trains and transforms cleanly")
#     for bad in ({"window_size": 3}, {"n_tiles": 1}, {"tile_gap": (0, 3)},
#                 {"trunk_block": "mobilenet"}, {"tile_norm": "l2"},
#                 {"lambda_reconstruct": -1.0},
#                 {"lambda_order": 0.0, "lambda_pair": 0.0, "lambda_forecast": 0.0,
#                  "lambda_reconstruct": 0.0}):
#         try:
#             JigsawNet(**dict(shared, **bad))
#             raise AssertionError(f"{bad} should have been rejected")
#         except ValueError:
#             pass
#     _check(True, "invalid constructor arguments all raise ValueError")
#     try:
#         JigsawNet(window_size=8, n_tiles=3, tile_gap=(1, 3), max_epochs=1, device="cpu",
#                   verbose=False).fit(np.zeros((31, 5), dtype=np.float32))
#         raise AssertionError("a one-span recording should have been rejected")
#     except ValueError:
#         _check(True, "a recording with fewer than two spans raises instead of dividing by zero")

#     print("11. save / load round trip")
#     path = Path("_jigsaw_net_selftest.pt")
#     try:
#         trained.save(path)
#         restored = JigsawNet.load(path, device="cpu")
#         _check(np.allclose(restored.transform(data[cut:cut + 300], pad=False),
#                            trained.transform(data[cut:cut + 300], pad=False), atol=1e-6),
#                "a reloaded model reproduces its embedding exactly")
#     finally:
#         path.unlink(missing_ok=True)

#     print("\nAll self-tests passed. No contrastive loss exists in this module.")


# def _demo():
#     data, latent = _synthetic(9000, 32, seed=7)
#     cut = 6500
#     shared = dict(window_size=10, n_tiles=4, tile_gap=(1, 6), output_dimension=32,
#                   num_hidden_units=48, head_hidden_units=48, batch_size=256,
#                   learning_rate=2e-3, device="cuda_if_available", verbose=False, random_state=0)
#     arms = {
#         "proposed (all three)": dict(max_epochs=25),
#         "no reconstruct anchor": dict(max_epochs=25, lambda_reconstruct=0.0),
#         "order only": dict(max_epochs=25, lambda_forecast=0.0, lambda_reconstruct=0.0),
#         "forecast only": dict(max_epochs=25, lambda_order=0.0, lambda_pair=0.0,
#                               lambda_reconstruct=0.0),
#         "reconstruct only": dict(max_epochs=25, lambda_order=0.0, lambda_pair=0.0,
#                                  lambda_forecast=0.0),
#         "random encoder": dict(max_epochs=0),
#         "normalize=True (sphere)": dict(max_epochs=25, normalize=True),
#         "separable trunk": dict(max_epochs=25, trunk_block="separable"),
#     }
#     print(f"{'arm':<26}{'R2':>9}{'raw R2':>9}{'delta':>9}{'p.ratio':>9}"
#           f"{'pair %':>9}{'ties %':>8}{'shortcut':>10}")
#     print("-" * 89)
#     for name, override in arms.items():
#         model = JigsawNet(**dict(shared, **override)).fit(data[:cut])
#         decoding = model.evaluate_decoding(data[:cut], latent[:cut], data[cut:], latent[cut:])
#         pretext = model.evaluate_pretext(data[cut:], max_spans=400, verbose=False)
#         print(f"{name:<26}{decoding['embedding']['r2']:>9.4f}"
#               f"{decoding['raw_window']['r2']:>9.4f}{decoding['embedding_minus_raw']:>9.4f}"
#               f"{decoding['embedding']['participation_ratio']:>9.2f}"
#               f"{pretext['pair_accuracy_percent']:>9.2f}"
#               f"{pretext['argmax_tie_rate_percent']:>8.1f}"
#               f"{pretext['baseline_mean_sort_pair_percent']:>10.2f}")
#     print("-" * 89)
#     print("delta < 0 means the encoder is removing information the raw window already had.")
#     print("p.ratio is the participation ratio of the embedding: if it collapses, so does R2.")


# def main():
#     parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
#     parser.add_argument("--self-test", action="store_true")
#     parser.add_argument("--demo", action="store_true")
#     options = parser.parse_args()
#     if options.self_test:
#         _self_test()
#     elif options.demo:
#         _demo()
#     else:
#         parser.print_help()


# if __name__ == "__main__":
#     main()

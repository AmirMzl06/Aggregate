"""Make the ORDER term earn its keep.

Read this before running anything, because it is the reason the file exists.

WHAT THE LOGS ACTUALLY SHOWED (J-CO0, seed 8, 6000 epochs)

    ctrl_fresh_labels      0.5423     <- RANDOM labels. Best arm in the table.
    audited_proposed       0.5334     <- real jigsaw labels
    relational_head        0.5296
    ctrl_frozen_labels     0.5109     <- arbitrary but FIXED chronology
    relational_counts      0.4777
    ctrl_tcl               0.4628
    tiles_2_counts         0.4546     <- the only arm that LEARNED the pretext
    order_count_matched    0.4504
    order_raw_view         0.3233     (participation ratio 4.19 -- collapsed)
    random_encoder         0.3199

A pretext task whose labels are drawn fresh from a uniform distribution every
single step scored HIGHER than the real one. That is not a small effect being
swamped by noise; it is the absence of an effect. The user's own reading --
"the jigsaw has hardly any effect, reconstruct is what produces the result" --
is what the table says, and this file starts from accepting it.

The honest version of the number: `fresh_random` is a degenerate control. Its
position CE sat at exactly 1.3863 = log 4 and its pair BCE at exactly 0.6931 =
log 2 for all 6000 epochs, which means the head collapsed to a constant and the
gradient reaching the encoder from the order branch was ~0. So it is a control
for "the order branch is switched off", not for "the order branch regularizes".
The control that actually trains the head on a wrong-but-learnable target is
`frozen_random`, at 0.5109. Against it:

    ORDER effect = 0.5334 - 0.5109 = +0.0225   on one seed.

Small, and inside the seed-to-seed spread seen in earlier runs.

WHY IT IS DEAD -- four mechanisms, each with its own switch below

1.  THE TASK IS UNDECIDABLE AT THE SPAN LENGTH USED.  training_span =
    K*W + (K-1)*gap_max + 1 = 4*10 + 3*8 + 1 = 65 bins. The first and last
    tiles are >= 51 bins apart, well past the behavioural autocorrelation time
    of a centre-out reach. Nothing in the data says which of two far-apart
    tiles came first, except slow drift in the firing level -- which is exactly
    the shortcut we delete. So the task is either cheatable or impossible.
    The evidence is in the table: every K=4 arm sits at chance, while K=2
    (`tiles_2_counts`, span 21 bins) reaches 67.97% against a 50.00% shortcut.
    -> `gap_curriculum`, and the `order_short_span` arm.

2.  THE HEAD MEMORIZES, SO THE TERM SWITCHES ITSELF OFF.  Position CE was
    0.062 at epoch 600 and 0.0026 at epoch 6000, with train exact 99.9%. Spans
    overlap at stride 1, the head has 2 hidden layers, and it simply learns the
    3752 training spans. Once the loss is ~0 its gradient is ~0, so for 80% of
    training the order term contributes literally nothing.
    -> `order_head_kind="linear"`, `order_head_reset_every`, `order_hard_fraction`.

3.  THE ORDER BRANCH IS TRAINED ON A VIEW THAT NEVER OCCURS AT TEST TIME.
    `tile_norm="mean"` subtracts each tile's own per-neuron mean. The decoder,
    and `transform`, see the raw window. So the order gradient shapes the
    encoder's behaviour on inputs it is never asked about, and whatever
    transfers to the raw view does so by accident. Worse, mean subtraction
    removes the firing LEVEL -- and level is the dominant behaviour code, which
    is why `order_raw_view` (level restored, shortcut restored) collapses to a
    participation ratio of 4.19: given the shortcut, the encoder throws
    everything else away.
    -> `span_selection`: kill the shortcut by DATA SELECTION, not by deleting
       the level, so the order branch and the decoder see the same raw view.

    AND THE SHORTCUT WAS NEVER BEING MEASURED CORRECTLY, which matters for how
    the old numbers get read. `baseline_mean_sort_pair_percent` applies one
    fixed rule -- "later tiles fire more" -- and it has printed ~50% on every
    run, which looked like proof that no level shortcut existed. It is not.
    Population drift goes up as often as it goes down, so a FIXED rule averages
    to chance even when the level determines the order perfectly inside every
    single span. The quantity a network can actually exploit is the UNSIGNED
    one: pick the direction per span, then sort. Measured on drifting Poisson
    counts (6000 bins, 20 neurons, span 65):

        fixed "later = higher"            50.3%   <- what the runner printed
        same statistic, sign per span     84.8%   <- what is actually available

    So a 50% mean-sort baseline never licensed the conclusion that the model
    was not using level. `baseline_level_oracle_pair_percent` is reported from
    now on, and it is the number to quote. It is an upper bound (it peeks at
    one bit of the label to choose the sign) and it is only meaningful for
    K >= 3 -- at K=2 there is a single pair, so choosing its sign is the whole
    task and the bound is trivially 100%.

    Selection then has to beat that unsigned shortcut, and gaps alone cannot:
    varying the gaps within 1..8 moves the tiles by ~20 bins inside a 65-bin
    span, so a span whose drift is monotone stays monotone. Only 27.7% of spans
    had ANY gap draw reaching exactly balanced. Moving the span START as well
    is what creates the diversity (same synthetic data, oracle-sign shortcut):

        candidates P:          1       4       8      16      24
        gaps only           84.8%   79.3%   77.5%   75.8%   75.1%
        gaps + start jitter 84.8%   71.5%   65.7%   60.9%   59.3%

    Hence `selection_pool` AND `selection_jitter`. Candidate 0 is always the
    unjittered natural span, so selection can never be worse than random.

4.  A K-WAY SLOT LABEL CARRIES ALMOST NO INFORMATION.  log 4 = 1.39 nats per
    tile, and the label is invariant to how far apart the tiles actually are.
    Two tiles 11 bins apart and two tiles 51 bins apart get the same label.
    -> `lambda_lag`: classify the SIGNED, QUANTIZED BIN OFFSET instead. Same
       cross-entropy, several times the information, and it cannot be solved by
       a monotone statistic because it needs the magnitude too.

WHAT THIS FILE IS NOT
  No InfoNCE, no contrastive loss, no negatives, no temperature, no CEBRA --
  the only losses anywhere in it are `F.cross_entropy` and
  `F.binary_cross_entropy_with_logits`, same as `jigsaw_net.py`.

HOW IT PLUGS IN
  Everything is a subclass of `JigsawNet` / `MobileJigsaw`, so fit / transform /
  save / load / `evaluate_decoding` and the whole runner keep working unchanged.
  Every new knob defaults to a no-op: `OrderJigsaw()` with no arguments is
  bit-for-bit `JigsawNet()`. That is asserted in `_smoke_test`, and it is what
  makes the `order_default` arm a real control rather than a hope.

  In run_jigsaw.py:
      from jigsaw_order import register
      register(MODELS, ARMS)

THE MEASUREMENT BUG IN THE PREVIOUS AUDIT FILE, FIXED HERE
  Taking non-overlapping spans (`starts[::training_span]`) is the right instinct
  and the wrong implementation: on a 955-bin validation split it left
  n_spans = 14 and a +-26.19% confidence interval, which made every pretext
  number in that run unreadable. This file keeps ALL overlapping spans for the
  point estimate and puts a MOVING-BLOCK BOOTSTRAP (block = training_span)
  around it, which is the standard fix for exactly this situation: the estimate
  stays precise, the interval honestly reflects the autocorrelation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from jigsaw_net import (JigsawNet, _GradScale, _assign, _augment, _gather_tiles,
                        _integer, _order_metrics, _pair_targets, _ranks, _real,
                        _shuffle_tiles, _tile_normalize)
from mobile_jigsaw import MobileJigsaw

ORDER_VIEWS = ("tile_norm", "count_match", "raw")
ORDER_HEADS = ("deepsets", "relational", "linear")
LABEL_CONTROLS = ("true", "fresh_random", "frozen_random")
SPAN_SELECTIONS = ("random", "decorrelated", "adversarial", "shortcut")

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# 1. killing the level shortcut without deleting the level
# --------------------------------------------------------------------------- #
def concordance(statistic):
    """Fraction of chronological pairs that a per-tile statistic gets right.

    `statistic` is (B, K) with the tiles in CHRONOLOGICAL order, so the correct
    answer is "increasing". Returns (B,) in [0, 1]: 1.0 means sorting the tiles
    by this statistic reproduces the true order exactly, 0.5 means the statistic
    is uninformative, 0.0 means it is exactly backwards.

    This is the SIGNED quantity, and it is the one
    `baseline_mean_sort_pair_percent` has always reported. Read
    `shortcut_strength` before drawing any conclusion from it.
    """
    difference = statistic[:, None, :] - statistic[:, :, None]      # [i, j] = s_j - s_i
    mask = torch.triu(torch.ones_like(difference, dtype=torch.bool), diagonal=1)
    wins = ((difference > 0).to(torch.float32) + 0.5 * (difference == 0).to(torch.float32))
    return (wins * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)


def shortcut_strength(statistic):
    """max(c, 1-c): the level shortcut as a network can actually use it.

    A fixed rule -- "later tiles fire more" -- scores 50% whenever the drift
    rises as often as it falls, which is why every run so far printed a
    reassuring mean-sort baseline of ~50% while the level may have been handing
    over the whole permutation. Choosing the direction per span costs one bit
    and is plainly within a network's reach, and on drifting Poisson counts it
    takes the same statistic from 50.3% to 84.8%.

    Upper bound, not an achievable baseline: it reads the sign off the label.
    Trustworthy for K >= 3, where one bit buys C(K,2) >= 3 pair decisions; at
    K=2 the single pair IS the sign, so the bound is 100% and says nothing.
    """
    c = concordance(statistic)
    return torch.maximum(c, 1.0 - c)


def select_span(data, starts, valid_starts, gaps, jitter, window_size, mode):
    """Pick, per span, the candidate whose LEVEL says least about chronology.

    Candidates vary in two ways: the gap draw (which bins become tiles) and an
    offset applied to the span start. Both are needed. Gaps alone move the tiles
    by about 20 bins inside a 65-bin span, so a span whose population drift is
    monotone throughout stays monotone no matter how the gaps fall -- measured,
    only 27.7% of spans had any gap draw that balanced. Adding start jitter took
    the oracle-sign shortcut from 77.5% to 65.7% at P=8.

        "decorrelated" : minimize |concordance - 0.5|. Note this is the UNSIGNED
                         criterion -- it removes "later fires more" AND "later
                         fires less" together, which is the point, since a
                         network is free to use either. The order branch can
                         then be fed RAW tiles, the same view `transform` and
                         the decoder see, so the level survives as a behaviour
                         code while stopping short of solving the puzzle.

        "adversarial"  : minimize concordance, i.e. spans where the level order
                         is actively backwards. Mainly an EVALUATION probe: a
                         model trained on decorrelated spans that still beats
                         chance here cannot be sorting by level. As a training
                         mode it is weaker than it looks, because a consistent
                         "later fires less" rule would solve it.

        "shortcut"     : maximize |concordance - 0.5|, the positive control.
                         Training here should give high pretext accuracy and a
                         collapsed embedding -- the `order_raw_view` failure
                         (participation ratio 4.19) on purpose, as the top of
                         the sweep.

    `jitter` is (P, B) integer offsets in the INDEX SPACE of `valid_starts`, not
    in bins. That is deliberate: `valid_starts` already excludes every start
    that would run off the end of a sequence, so offsetting inside it cannot
    produce an out-of-range span or straddle two concatenated sessions. Row 0
    must be all zeros so the natural span is always a candidate.

    Returns (starts, gaps). P=1 or mode="random" returns the inputs untouched,
    so `selection_pool=1` is an exact no-op.
    """
    pool = gaps.shape[0]
    if pool == 1 or mode == "random":
        return starts, gaps[0]
    with torch.no_grad():
        if jitter is None:
            candidate_starts = starts[None].expand(pool, -1)
        else:
            position = torch.searchsorted(valid_starts, starts)
            position = (position[None, :] + jitter).clamp_(0, len(valid_starts) - 1)
            candidate_starts = valid_starts[position]
        scores = torch.stack([
            concordance(_gather_tiles(data, candidate_starts[p], gaps[p],
                                      window_size)[0].sum(dim=(2, 3)))
            for p in range(pool)], dim=0)                            # (P, B)
        if mode == "decorrelated":
            cost = (scores - 0.5).abs()
        elif mode == "adversarial":
            cost = scores
        else:
            cost = -(scores - 0.5).abs()
        choice = cost.argmin(dim=0)                                  # (B,)
    rows = torch.arange(len(starts), device=starts.device)
    return candidate_starts[choice, rows], gaps[choice, rows]


def count_match(tiles, generator, *, cap=24):
    """Equalize every neuron's TOTAL spike count across the K tiles exactly.

    Why this is not the same as subtracting the mean. For integer counts the
    removed mean stays recoverable from the lattice and the sparsity pattern:
    a small MLP reads the per-tile level back out of mean-subtracted Poisson
    counts at R^2 ~ 0.95 (0.83 after z-scoring, 0.00 for Gaussian data). So
    `tile_norm="mean"` does not remove the shortcut from a network that is
    willing to look for it -- it only removes it from sort-by-mean, which is
    the baseline we happen to print.

    This removes it for real: per (span, tile, neuron) we keep a uniformly
    random subset of the individual spikes, of size c* = min over tiles of the
    total. After it, every neuron fires the same number of spikes in every tile
    and the only thing left to tell the tiles apart is WHEN inside the window
    those spikes happen -- which is the temporal structure the jigsaw is
    supposed to be about.

    Cost: it also throws away real signal, and the J-CO0 run showed that cost
    (R2 0.4504 vs 0.5334). Kept as an ablation and as the honest "shortcut is
    provably gone" arm, not as the default; `span_selection="decorrelated"`
    reaches the same conclusion without deleting anything.

    `cap` bounds the per-bin count considered, to bound the (B,K,N,W,cap)
    working tensor. Counts above it are clamped FOR THE ORDER VIEW ONLY.
    """
    counts = tiles.round().clamp_(min=0, max=float(cap))
    batch, n_tiles, channels, window = counts.shape
    ceiling = int(counts.max().item()) if counts.numel() else 0
    if ceiling == 0:
        return counts
    totals = counts.sum(-1)                                          # (B,K,N)
    target = totals.min(dim=1).values                                # (B,N)
    slot = torch.arange(ceiling, device=tiles.device)
    valid = slot < counts.unsqueeze(-1)                              # (B,K,N,W,ceiling)
    keys = torch.rand(valid.shape, device=tiles.device, generator=generator)
    keys = keys.masked_fill(~valid, 2.0)
    flat = keys.reshape(batch, n_tiles, channels, window * ceiling)
    ordered = flat.sort(dim=-1).values
    wanted = target[:, None, :].expand(batch, n_tiles, channels)
    index = (wanted.to(torch.long) - 1).clamp_min(0).unsqueeze(-1)
    threshold = ordered.gather(-1, index).squeeze(-1)                # (B,K,N)
    # target == 0 means "keep nothing": a threshold below every uniform draw.
    threshold = torch.where(wanted > 0, threshold, torch.full_like(threshold, -1.0))
    keep = valid & (keys <= threshold[..., None, None])
    return keep.sum(-1).to(tiles.dtype)


def remove_slow_drift(X, window_bins, *, causal=False):
    """Optional PREPROCESSING: divide out a slow multiplicative gain.

    The level shortcut exists because population firing drifts on a timescale
    far longer than one span, so "which tile fires more" answers "which tile is
    later" without any temporal structure being learned. Dividing each neuron by
    its own running mean removes that drift at the source, before any tiling,
    and leaves the fast within-window structure intact. Applies equally to
    train and test, so it is a change of representation, not a leak -- but fit
    the window on TRAIN and reuse it, and never let `window_bins` approach the
    behavioural timescale or it will eat the signal too.

    Not wired into the model on purpose: it changes the data every arm sees,
    including the ceiling, so it belongs in the runner's loader if you want it.
    """
    x = np.asarray(X, dtype=np.float64)
    window = max(1, int(window_bins))
    padded = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)], axis=0)
    if causal:
        lo = np.maximum(np.arange(len(x)) - window + 1, 0)
        hi = np.arange(len(x)) + 1
    else:
        half = window // 2
        lo = np.maximum(np.arange(len(x)) - half, 0)
        hi = np.minimum(np.arange(len(x)) + half + 1, len(x))
    local = (padded[hi] - padded[lo]) / (hi - lo)[:, None]
    return (x / (local + _EPS) * x.mean(0, keepdims=True)).astype(np.float32)


# --------------------------------------------------------------------------- #
# 2. heads that cannot memorize their way out of the job
# --------------------------------------------------------------------------- #
class _LinearOrderHead(nn.Module):
    """The lowest-capacity head that can express the task at all.

        position logits = W z_k + b          (per tile, no cross-tile context)
        pair logit(i,j) = w^T (z_i - z_j)    (one direction, exactly antisymmetric)

    Two reasons this is the interesting head and not a crippled one.

    It cannot memorize. A 64x4 matrix plus a 64-vector has 324 parameters
    against 3752 training spans, so there is no lookup table available; if the
    order loss goes down, the order information went into the EMBEDDING, which
    is the only thing we actually care about. The DeepSets head has ~12k
    parameters and drove its own loss to 0.0026 while the embedding learned
    nothing -- that is the failure this replaces.

    It states a strong, quotable property. `pair logit = w^T (z_i - z_j)` says
    time is a single LINEAR DIRECTION in embedding space; driving that BCE down
    means the encoder has laid the trajectory out along an axis, which is
    exactly the geometry a linear decoder wants. Permutation equivariance is
    automatic here because nothing mixes tiles.
    """

    def __init__(self, dimension, n_tiles):
        super().__init__()
        self.position = nn.Linear(dimension, n_tiles)
        self.score = nn.Linear(dimension, 1, bias=False)

    def forward(self, z):
        return self.position(z), self.score(z).squeeze(-1)


class _RelationalOrderHead(nn.Module):
    """Self-attention across tiles, with NO positional encoding.

    The DeepSets head sees each tile next to the MEAN of the set, so the only
    relation it can express is "how does this tile differ from the average".
    "B is what A turns into" is not in that family. Attention is, and dropping
    the positional encoding keeps the whole head permutation-equivariant, so
    the shuffled-label protocol still holds. This is the capacity-UP arm, the
    opposite bet from `_LinearOrderHead`; running both says whether the order
    term is limited by what the head can express or by what it can memorize.
    """

    def __init__(self, dimension, hidden, n_tiles, *, layers=2, heads=4):
        super().__init__()
        heads = max(1, math.gcd(hidden, heads))
        self.input = nn.Linear(dimension, hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads,
                                           dim_feedforward=2 * hidden,
                                           dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, num_layers=layers)
        self.position = nn.Linear(hidden, n_tiles)
        self.score = nn.Linear(hidden, 1)

    def forward(self, z):
        tokens = self.body(self.input(z))
        return self.position(tokens), self.score(tokens).squeeze(-1)


class _LagHead(nn.Module):
    """Classify the SIGNED, QUANTIZED bin offset between two tiles.

    The slot label is worth log K = 1.39 nats and is blind to distance: two
    tiles 11 bins apart and two tiles 51 bins apart carry the same target. The
    lag label is worth log(lag_classes) nats and is not. It also cannot be
    solved by any monotone statistic on its own, because getting the class right
    needs the MAGNITUDE of the separation, not just its sign -- so the level
    drift that hands over the permutation for free does not hand this over.

    Reads `z_i - z_j`, so it is a pure relation and adds nothing per-tile.
    Still one `F.cross_entropy`.
    """

    def __init__(self, dimension, hidden, classes, *, linear=False):
        super().__init__()
        self.classes = classes
        self.net = nn.Linear(dimension, classes) if linear else nn.Sequential(
            nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, classes))

    def forward(self, difference):
        return self.net(difference)


def lag_edges(window_size, n_tiles, gap_low, gap_high, classes, device):
    """Geometric magnitude bands, mirrored around zero. Even class count.

    Reachable |offset| runs from window_size+gap_low (neighbours, tightest gap)
    to (K-1)*window_size + (K-1)*gap_high (the outermost pair, widest gaps).
    Geometric rather than linear because the distribution of |offset| is heavily
    weighted toward the small end and equal-width bands would leave the top
    classes nearly empty -- the same argument that made the forecast targets
    per-neuron quantiles instead of equal-width bins.
    """
    bands = classes // 2
    low = float(window_size + gap_low)
    high = float((n_tiles - 1) * (window_size + gap_high))
    inner = torch.logspace(math.log10(low), math.log10(max(high, low + 1.0)),
                           bands + 1, device=device)[1:-1]
    return inner


def lag_targets(tile_starts, labels, edges, classes):
    """(rows,) class index for every upper-triangular pair of presented slots.

    `tile_starts` is (B, K) in CHRONOLOGICAL order; `labels[b, s]` is the
    chronological index of whichever tile is sitting in presented slot s, which
    is exactly what `_shuffle_tiles` returns. Gathering by `labels` puts the
    starts back in PRESENTED order so the target lines up with the embeddings.

    Orientation follows `_pair_targets` in jigsaw_net.py: entry [i, j] is about
    x_i - x_j, so the feature fed to the head must be `z_i - z_j` with the same
    indexing. Getting this backwards would train the head on mirrored labels and
    still look like it was learning, so it is worth stating twice.
    """
    presented = torch.gather(tile_starts, 1, labels)
    delta = presented[:, :, None] - presented[:, None, :]            # [i, j] = t_i - t_j
    mask = torch.triu(torch.ones_like(delta, dtype=torch.bool), diagonal=1)
    magnitude = torch.bucketize(delta.abs()[mask].to(torch.float32), edges)
    bands = classes // 2
    later = (delta[mask] > 0)                                        # slot i came AFTER slot j
    return torch.where(later, bands + magnitude,
                       bands - 1 - magnitude).clamp(0, classes - 1), mask


# --------------------------------------------------------------------------- #
# 3. the model
# --------------------------------------------------------------------------- #
class OrderJigsaw(JigsawNet):
    """JigsawNet with an order term built to survive its own controls.

    Every knob below defaults to the value that reproduces `JigsawNet` exactly,
    so `order_default` is a genuine no-op control and any difference from
    `proposed` is a bug, not a finding. Turn them on one at a time.

    order_view            "tile_norm" (inherit self.tile_norm) | "count_match" | "raw"
    order_head_kind       "deepsets" | "relational" | "linear"
    label_control         "true" | "fresh_random" | "frozen_random"
    span_selection        "random" | "decorrelated" | "adversarial" | "shortcut"
    selection_pool        candidate spans to choose between; 1 forces "random"
    selection_jitter      how far the candidate starts may move, in span-start
                          index units (0 = vary the gaps only)
    lambda_lag            weight on the signed-offset CE (0 = off)
    lag_classes           even; number of signed offset bands
    order_hard_fraction   keep only this fraction of the batch, hardest first
    order_head_reset_every  re-initialize the order head every N optimizer steps
    gap_curriculum        fraction of training over which gap_max grows from gap_min
    anchor_final_scale    lambda_reconstruct multiplier at the end of training
    order_final_scale     lambda_order / lambda_pair / lambda_lag multiplier at the end
    count_match_cap       per-bin count ceiling for the count_match view
    frozen_block          span-start block size for the frozen_random control
    """

    _MODEL_TYPE = "order_jigsaw"
    _PARAM_NAMES = JigsawNet._PARAM_NAMES + (
        "order_view", "order_head_kind", "label_control", "span_selection",
        "selection_pool", "selection_jitter", "lambda_lag", "lag_classes",
        "order_hard_fraction", "order_head_reset_every", "gap_curriculum",
        "anchor_final_scale", "order_final_scale", "count_match_cap", "frozen_block")

    def __init__(self, *, order_view="tile_norm", order_head_kind="deepsets",
                 label_control="true", span_selection="random", selection_pool=1,
                 selection_jitter=0, lambda_lag=0.0, lag_classes=6,
                 order_hard_fraction=1.0, order_head_reset_every=0,
                 gap_curriculum=0.0, anchor_final_scale=1.0, order_final_scale=1.0,
                 count_match_cap=24, frozen_block=0, **kwargs):
        super().__init__(**kwargs)
        for name, value, allowed in (("order_view", order_view, ORDER_VIEWS),
                                     ("order_head_kind", order_head_kind, ORDER_HEADS),
                                     ("label_control", label_control, LABEL_CONTROLS),
                                     ("span_selection", span_selection, SPAN_SELECTIONS)):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}.")
        self.order_view = order_view
        self.order_head_kind = order_head_kind
        self.label_control = label_control
        self.span_selection = span_selection
        self.selection_pool = _integer("selection_pool", selection_pool, 1, 64)
        self.selection_jitter = _integer("selection_jitter", selection_jitter, 0)
        self.lambda_lag = _real("lambda_lag", lambda_lag, 0)
        self.lag_classes = _integer("lag_classes", lag_classes, 2, 32)
        if self.lag_classes % 2:
            raise ValueError("lag_classes must be EVEN: the bands are mirrored "
                             "around zero, half negative and half positive.")
        self.order_hard_fraction = _real("order_hard_fraction", order_hard_fraction,
                                         0, 1, strict_min=True)
        self.order_head_reset_every = _integer("order_head_reset_every",
                                               order_head_reset_every, 0)
        self.gap_curriculum = _real("gap_curriculum", gap_curriculum, 0, 1)
        self.anchor_final_scale = _real("anchor_final_scale", anchor_final_scale, 0)
        self.order_final_scale = _real("order_final_scale", order_final_scale, 0)
        self.count_match_cap = _integer("count_match_cap", count_match_cap, 1, 256)
        self.frozen_block = _integer("frozen_block", frozen_block, 0)
        if self.span_selection != "random" and self.selection_pool == 1:
            raise ValueError(f"span_selection={self.span_selection!r} needs "
                             "selection_pool > 1; with one candidate there is "
                             "nothing to select between.")
        if self.lambda_lag > 0 and self.n_tiles < 2:
            raise ValueError("lambda_lag needs at least 2 tiles.")

    # -- construction ------------------------------------------------------ #
    def _build(self, channels):
        super()._build(channels)
        dimension, hidden = self.output_dimension, self.head_hidden_units
        if self.order_head_kind == "relational":
            self.order_head_ = _RelationalOrderHead(dimension, hidden,
                                                    self.n_tiles).to(self.device_)
        elif self.order_head_kind == "linear":
            self.order_head_ = _LinearOrderHead(dimension, self.n_tiles).to(self.device_)
        # The lag head is attached AS A SUBMODULE of the order head, not as a
        # separate estimator attribute. JigsawNet.fit builds its AdamW from an
        # explicit list -- encoder + order + forecast + reconstruct -- so a
        # free-standing head would receive gradients, never be stepped, and stay
        # at its initialization for the whole run while its loss sat in the
        # total. Nothing would raise; the arm would just quietly do nothing.
        # Hanging it here also means save/load and .train()/.eval() pick it up
        # for free, since it rides inside order_head_'s state_dict.
        self.order_head_.lag_head = _LagHead(
            dimension, hidden, self.lag_classes,
            linear=self.order_head_kind == "linear").to(self.device_)
        self.lag_head_ = self.order_head_.lag_head
        self.lag_edges_ = lag_edges(self.window_size, self.n_tiles, self.tile_gap[0],
                                    self.tile_gap[1], self.lag_classes, self.device_)
        # One arbitrary-but-FIXED chronology per block of span starts. Redrawing
        # it every step (fresh_random) makes the target unlearnable, the head
        # collapses to uniform, and the gradient reaching the encoder is ~0 --
        # which is why that control measured "no order branch" rather than
        # "order branch with useless labels". Frozen labels stay learnable, so
        # the head still trains and still pushes on the encoder; the ONLY thing
        # removed is that the labels mean anything about time.
        table = torch.Generator(device=self.device_)
        table.manual_seed(self.random_state + 977)
        self._frozen_table = torch.rand(4096, self.n_tiles, device=self.device_,
                                        generator=table).argsort(dim=1)
        # Cleared here, and `fit` calls `_build` BEFORE its own `_spans` call, so
        # the first cache write always comes from the training array. See
        # `_spans` for why letting validation win would be silently wrong.
        self._valid_starts_ = None
        self._steps_ = 0

    def _reset_order_head(self):
        """Re-initialize the order head IN PLACE, keeping optimizer references.

        Rebuilding the module object would orphan the parameters the optimizer
        already holds, so the new head would never be updated -- a silent, total
        failure. Copying fresh values into the existing tensors avoids that.
        Adam's moment estimates survive the reset, which softens it slightly;
        that is a known and accepted approximation here.

        Normalization SCALES are restored to 1, not zeroed. A LayerNorm weight
        is 1-D like a bias, and zeroing it inside the relational head's
        transformer would kill the signal path permanently -- the head would
        never recover and the arm would look like proof that resetting is a bad
        idea, when it would only be proof of this bug.
        """
        seed = self.random_state + 7919 + self._steps_
        fresh = torch.Generator(device="cpu")
        fresh.manual_seed(seed % (2 ** 31 - 1))
        with torch.no_grad():
            for name, parameter in self.order_head_.named_parameters():
                if parameter.dim() >= 2:
                    bound = 1.0 / math.sqrt(max(parameter.shape[1], 1))
                    sample = torch.empty(parameter.shape, device="cpu")
                    sample.uniform_(-bound, bound, generator=fresh)
                    parameter.copy_(sample.to(parameter.device))
                elif name.endswith("bias"):
                    parameter.zero_()
                else:
                    parameter.fill_(1.0)

    # -- schedules --------------------------------------------------------- #
    def _progress(self):
        """Fraction of the optimizer steps consumed so far, in [0, 1]."""
        per_epoch = max(1, math.ceil(getattr(self, "n_spans_", 1) / self.batch_size))
        total = max(1, self.max_epochs * per_epoch)
        return min(1.0, max(0, self._steps_ - 1) / max(1, total - 1))

    def _draw(self, count):
        """Gap draw, with the curriculum and the candidate pool for selection.

        The curriculum grows the gap CEILING from gap_min to gap_max over the
        first `gap_curriculum` of training. `training_span` still uses gap_max,
        so the span budget, the number of spans and the validation protocol are
        all unchanged -- only which tiles are drawn moves. Early training then
        sees the regime we know is decidable (K=2 at a 21-bin span reached 68%),
        and the task is widened only once something has been learned.

        Returns (P, count, K-1); P=1 unless a selection mode is active.
        """
        low, high = self.tile_gap
        if self.gap_curriculum > 0 and high > low:
            fraction = min(1.0, self._progress() / max(self.gap_curriculum, _EPS))
            high = int(round(low + fraction * (high - low)))
        pool = 1 if self.span_selection == "random" else self.selection_pool
        return torch.randint(low, high + 1, (pool, count, self.n_tiles - 1),
                             device=self.device_, generator=self._generator)

    def _jitter(self, pool, count, valid_starts, generator=None):
        """(P, count) start offsets in valid-start index units, row 0 all zeros.

        Row 0 being zero is what makes selection monotone: the natural span is
        always one of the candidates, so the chosen span can never score worse
        on the selection criterion than no selection at all.

        `valid_starts` is passed in rather than read off the instance because
        training and evaluation walk DIFFERENT start lists -- see `_spans`.
        """
        if (self.selection_jitter == 0 or pool == 1 or valid_starts is None
                or self.span_selection == "random"):
            return None
        offsets = torch.randint(-self.selection_jitter, self.selection_jitter + 1,
                                (pool, count), device=self.device_,
                                generator=self._generator if generator is None else generator)
        offsets[0].zero_()
        return offsets

    def _spans(self, X, n_features=None):
        """Cache the TRAINING span starts, once, for start-jitter selection.

        Only the first call made from inside `fit` is cached. `evaluate_pretext`
        also calls `_spans`, on the VALIDATION array, and letting that overwrite
        the cache would have `_step` jitter training starts through validation
        indices -- silently training on the wrong spans, or indexing out of
        range. `_build` clears the cache and runs before `fit`'s own call, so the
        ordering is what makes this safe.
        """
        data, starts, channels = super()._spans(X, n_features)
        if getattr(self, "_fitting_", False) and getattr(self, "_valid_starts_", None) is None:
            self._valid_starts_ = torch.from_numpy(
                np.ascontiguousarray(starts)).to(self.device_)
        return data, starts, channels

    def _control_labels(self, truth, starts):
        if self.label_control == "true":
            return truth
        if self.label_control == "fresh_random":
            keys = torch.rand(truth.shape, device=truth.device, generator=self._generator)
            return keys.argsort(dim=1)
        block = self.frozen_block if self.frozen_block > 0 else self.training_span
        row = (starts // max(block, 1)) % self._frozen_table.shape[0]
        return self._frozen_table[row]

    def _order_view(self, tiles, generator=None):
        if self.order_view == "raw":
            return tiles
        if self.order_view == "count_match":
            return count_match(tiles, generator if generator is not None
                               else self._generator, cap=self.count_match_cap)
        return _tile_normalize(tiles, self.tile_norm)

    # -- one step ---------------------------------------------------------- #
    def _step(self, data, starts):
        """Same structure as JigsawNet._step. Only the ORDER branch differs."""
        batch = len(starts)
        self._steps_ += 1
        if self.order_head_reset_every and self._steps_ % self.order_head_reset_every == 0:
            self._reset_order_head()
        progress = self._progress()
        anchor_scale = 1.0 + (self.anchor_final_scale - 1.0) * progress
        order_scale = 1.0 + (self.order_final_scale - 1.0) * progress

        candidates = self._draw(batch)
        jitter = self._jitter(candidates.shape[0], batch, self._valid_starts_)
        starts, gaps = select_span(data, starts, self._valid_starts_, candidates,
                                   jitter, self.window_size, self.span_selection)
        tiles, next_index = _gather_tiles(data, starts, gaps, self.window_size)
        tile_starts = next_index - self.window_size
        parts, total = {}, torch.zeros((), device=self.device_)

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
            total = total + anchor_scale * self.lambda_reconstruct * reconstruct
        else:
            parts["reconstruct"] = torch.zeros((), device=self.device_)
            parts["reconstruct_accuracy"] = torch.zeros((), device=self.device_)

        if self.lambda_order > 0 or self.lambda_pair > 0 or self.lambda_lag > 0:
            view = self._order_view(
                _augment(tiles, self.neuron_dropout, self.gain_jitter, self._generator))
            view, labels = _shuffle_tiles(view, self.shuffle_tiles, self._generator)
            labels = self._control_labels(labels, starts)
            z_order = self.encoder_(view.reshape(-1, self.n_features_in_, self.window_size))
            z_order = _GradScale.apply(z_order, self.order_grad_scale)
            z_order = z_order.reshape(batch, self.n_tiles, self.output_dimension)
            position_logits, scores = self.order_head_(z_order)

            # Per-SPAN losses, so hard-example mining has something to rank by.
            position_each = F.cross_entropy(
                position_logits.reshape(-1, self.n_tiles), labels.reshape(-1),
                reduction="none").reshape(batch, self.n_tiles).mean(1)
            target, mask = _pair_targets(labels)
            difference = scores[:, :, None] - scores[:, None, :]
            pair_all = F.binary_cross_entropy_with_logits(difference, target,
                                                          reduction="none")
            pair_each = (pair_all * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)

            if self.lambda_lag > 0:
                lag_target, lag_mask = lag_targets(tile_starts, labels,
                                                   self.lag_edges_, self.lag_classes)
                pair_z = (z_order[:, :, None, :] - z_order[:, None, :, :])[lag_mask]
                lag_logits = self.lag_head_(pair_z)
                lag_each = F.cross_entropy(lag_logits, lag_target, reduction="none")
                lag_each = lag_each.reshape(batch, -1).mean(1)
                with torch.no_grad():
                    parts["lag_accuracy"] = (lag_logits.argmax(-1) == lag_target).to(
                        torch.float32).mean()
            else:
                lag_each = torch.zeros(batch, device=self.device_)
                parts["lag_accuracy"] = torch.zeros((), device=self.device_)

            combined = (self.lambda_order * position_each + self.lambda_pair * pair_each
                        + self.lambda_lag * lag_each)
            if self.order_hard_fraction < 1.0:
                # The order loss reaches ~0 on the easy majority within a few
                # hundred epochs and then contributes no gradient at all. Keeping
                # only the hardest spans means the term stays alive for the whole
                # budget instead of switching itself off at epoch 600.
                keep = max(2, int(round(self.order_hard_fraction * batch)))
                picked = combined.detach().topk(keep).indices
                order_term = combined[picked].mean()
            else:
                order_term = combined.mean()
            total = total + order_scale * order_term

            parts["position"] = position_each.mean().detach()
            parts["pair"] = pair_each.mean().detach()
            parts["lag"] = lag_each.mean().detach()
            with torch.no_grad():
                predicted = _ranks(scores) if self.lambda_order == 0 \
                    else _assign(position_logits)
                exact, pair_accuracy, tie_rate = _order_metrics(predicted, labels)
                parts["exact"], parts["pair_accuracy"] = exact.mean(), pair_accuracy.mean()
                parts["tie_rate"] = tie_rate.mean()
                parts["shortcut"] = shortcut_strength(tiles.sum(dim=(2, 3))).mean()
        else:
            for key in ("position", "pair", "lag", "exact", "pair_accuracy",
                        "tie_rate", "lag_accuracy", "shortcut"):
                parts[key] = torch.zeros((), device=self.device_)
        parts["total"] = total
        return parts

    # -- fit --------------------------------------------------------------- #
    def fit(self, X, X_valid=None, *, validate_every=1):
        if self.order_view == "count_match":
            sample = X[0] if isinstance(X, (list, tuple)) else X
            values = np.asarray(sample, dtype=np.float64)
            if not np.allclose(values, np.round(values), atol=1e-6) or values.min() < -1e-6:
                raise ValueError(
                    "order_view='count_match' needs raw non-negative INTEGER spike "
                    "counts: it works by dropping individual spikes. This data is "
                    "not integer (smoothed, z-scored or rate-converted), so the "
                    "subsampling is meaningless. Use span_selection='decorrelated' "
                    "instead -- it blocks the same shortcut without touching the data.")
        # JigsawNet.fit calls evaluate_pretext once per epoch for monitoring and
        # throws the interval away. Running 1000 bootstrap replicates for a
        # number nobody reads would add minutes over a 6000-epoch run, so the
        # per-epoch calls skip it and only the runner's final call pays for it.
        self._fitting_ = True
        try:
            return super().fit(X, X_valid, validate_every=validate_every)
        finally:
            self._fitting_ = False

    # -- evaluation -------------------------------------------------------- #
    def evaluate_pretext(self, X, *, max_spans=1024, batch_size=256, repeats=8,
                         random_state=200042, verbose=None, n_boot=1000):
        """Held-out order accuracy with a MOVING-BLOCK BOOTSTRAP interval.

        Spans overlap at stride 1, so the ~900 validation spans are nowhere near
        900 independent observations and a Wilson interval on n=900 is far too
        narrow. The previous audit build fixed that by keeping only
        `starts[::training_span]`, which on a 955-bin split left n = 14 and a
        +-26% interval -- correct in spirit, useless in practice.

        A moving-block bootstrap gets both: the point estimate still uses every
        span, and the interval resamples contiguous blocks of length
        `training_span`, so it inherits the autocorrelation instead of ignoring
        it. `n_effective` reports the implied independent sample size.

        Also reported, and more informative than the model's own accuracy:
        `shortcut_pair_percent` under the SAME span selection the model trained
        with. If selection worked, this sits at 50%; anything the model scores
        above it is not level.
        """
        self._check_fitted()
        verbose = self.verbose if verbose is None else verbose
        repeats = _integer("repeats", repeats, minimum=1)
        data, starts, _ = self._spans(X, self.n_features_in_)
        if len(starts) == 0:
            raise ValueError("No valid spans in X.")
        rng = np.random.default_rng(random_state)
        chosen = starts if len(starts) <= max_spans else np.sort(
            rng.choice(starts, max_spans, replace=False))
        chosen = torch.from_numpy(np.ascontiguousarray(chosen)).to(self.device_)
        # Start-jitter walks in VALID-START INDEX space, so it needs the full
        # start list of THIS array -- not the cached training one, which would
        # index a different (and possibly shorter) sequence.
        local_starts = torch.from_numpy(np.ascontiguousarray(starts)).to(self.device_)
        generator = torch.Generator(device=self.device_).manual_seed(random_state)
        low, high = self.tile_gap
        pool = 1 if self.span_selection == "random" else self.selection_pool
        keys = ("exact", "pair", "tie", "argmax_pair", "argmax_tie", "rank_exact",
                "rank_pair", "mean_exact", "mean_pair", "norm_exact", "norm_pair",
                "level_oracle", "lag_accuracy")
        per_span = {k: torch.zeros(len(chosen), device=self.device_) for k in keys}
        position_ce = 0.0
        self.encoder_.eval(); self.order_head_.eval(); self.lag_head_.eval()
        with torch.inference_mode():
          for _ in range(repeats):
            for begin in range(0, len(chosen), batch_size):
                block = chosen[begin:begin + batch_size]
                stop = begin + len(block)
                candidates = torch.randint(low, high + 1,
                                           (pool, len(block), self.n_tiles - 1),
                                           device=self.device_, generator=generator)
                jitter = self._jitter(pool, len(block), local_starts, generator)
                block, gaps = select_span(data, block, local_starts, candidates,
                                          jitter, self.window_size, self.span_selection)
                tiles, next_index = _gather_tiles(data, block, gaps, self.window_size)
                tile_starts = next_index - self.window_size
                view = self._order_view(tiles, generator)
                view, positions = _shuffle_tiles(view, self.shuffle_tiles, generator)
                tiles = torch.gather(
                    tiles, 1, positions[:, :, None, None].expand(-1, -1, *tiles.shape[2:])
                ) if self.shuffle_tiles else tiles
                z = self.encoder_(view.reshape(-1, self.n_features_in_, self.window_size))
                z = z.reshape(len(block), self.n_tiles, self.output_dimension)
                position_logits, scores = self.order_head_(z)
                predicted = _assign(position_logits) if self.lambda_order > 0 \
                    else _ranks(scores)
                exact, pair, tie = _order_metrics(predicted, positions)
                _, argmax_pair, argmax_tie = _order_metrics(position_logits.argmax(-1),
                                                            positions)
                rank_exact, rank_pair, _ = _order_metrics(_ranks(scores), positions)
                mean_exact, mean_pair, _ = _order_metrics(_ranks(tiles.mean(dim=(2, 3))),
                                                          positions)
                norm_exact, norm_pair, _ = _order_metrics(
                    _ranks(tiles.reshape(len(block), self.n_tiles, -1).norm(dim=2)),
                    positions)
                if self.lambda_lag > 0:
                    lag_target, lag_mask = lag_targets(tile_starts, positions,
                                                       self.lag_edges_, self.lag_classes)
                    pair_z = (z[:, :, None, :] - z[:, None, :, :])[lag_mask]
                    hit = (self.lag_head_(pair_z).argmax(-1) == lag_target)
                    lag_hit = hit.reshape(len(block), -1).to(torch.float32).mean(1)
                else:
                    lag_hit = torch.zeros(len(block), device=self.device_)
                for key, value in (("exact", exact), ("pair", pair), ("tie", tie),
                                   ("argmax_pair", argmax_pair), ("argmax_tie", argmax_tie),
                                   ("rank_exact", rank_exact), ("rank_pair", rank_pair),
                                   ("mean_exact", mean_exact), ("mean_pair", mean_pair),
                                   ("norm_exact", norm_exact), ("norm_pair", norm_pair),
                                   ("level_oracle", torch.maximum(mean_pair, 1 - mean_pair)),
                                   ("lag_accuracy", lag_hit)):
                    per_span[key][begin:stop] += value.to(torch.float32) / repeats
                position_ce += float(F.cross_entropy(
                    position_logits.reshape(-1, self.n_tiles),
                    positions.reshape(-1))) * len(block)
        self.encoder_.train(); self.order_head_.train(); self.lag_head_.train()
        mean = {k: float(v.mean()) for k, v in per_span.items()}
        pair_values = per_span["pair"].cpu().numpy()
        boot = 0 if getattr(self, "_fitting_", False) else n_boot
        low_ci, high_ci, spread, n_effective = block_bootstrap(
            pair_values, self.training_span, n_boot=boot, random_state=random_state)
        result = dict(
            exact_accuracy_percent=100 * mean["exact"],
            pair_accuracy_percent=100 * mean["pair"],
            pair_ci95_percent=[100 * low_ci, 100 * high_ci],
            pair_block_resolution_percent=100 * spread,
            n_effective=n_effective,
            tie_rate_percent=100 * mean["tie"],
            argmax_pair_accuracy_percent=100 * mean["argmax_pair"],
            argmax_tie_rate_percent=100 * mean["argmax_tie"],
            rank_exact_accuracy_percent=100 * mean["rank_exact"],
            rank_pair_accuracy_percent=100 * mean["rank_pair"],
            lag_accuracy_percent=100 * mean["lag_accuracy"],
            chance_lag_percent=100.0 / self.lag_classes,
            baseline_mean_sort_exact_percent=100 * mean["mean_exact"],
            baseline_mean_sort_pair_percent=100 * mean["mean_pair"],
            baseline_norm_sort_exact_percent=100 * mean["norm_exact"],
            baseline_norm_sort_pair_percent=100 * mean["norm_pair"],
            # THE baseline that matters. baseline_mean_sort applies one FIXED
            # rule ("later fires more") and therefore averages to ~50% whenever
            # drift rises as often as it falls -- even when the level fully
            # determines the order inside every single span. An encoder only has
            # to pick the sign PER SPAN to get max(c, 1-c), which on drifting
            # Poisson counts is 84.8% where the fixed rule reads 50.3%. Every
            # "shortcut 50.00%" line in the earlier runs was uninformative.
            baseline_level_oracle_pair_percent=100 * mean["level_oracle"],
            position_cross_entropy=position_ce / max(repeats * len(chosen), 1),
            uniform_cross_entropy=math.log(self.n_tiles),
            decode="assignment" if self.lambda_order > 0 else "score_rank",
            order_view=self.order_view, span_selection=self.span_selection,
            label_control=self.label_control,
            chance_exact_percent=100.0 / math.factorial(self.n_tiles),
            chance_pair_percent=50.0,
            n_spans=int(len(chosen)), n_draws=int(repeats * len(chosen)),
            repeats=int(repeats))
        if verbose:
            print(f"  pretext: exact={result['exact_accuracy_percent']:.2f}% "
                  f"(chance {result['chance_exact_percent']:.2f}%) "
                  f"pair={result['pair_accuracy_percent']:.2f}% "
                  f"[{result['pair_ci95_percent'][0]:.2f}, "
                  f"{result['pair_ci95_percent'][1]:.2f}] block-bootstrap "
                  f"n_eff={n_effective} | level-oracle="
                  f"{result['baseline_level_oracle_pair_percent']:.2f}% "
                  f"(fixed-rule {result['baseline_mean_sort_pair_percent']:.2f}%)",
                  flush=True)
        return result


def block_bootstrap(values, block, *, n_boot=2000, random_state=0):
    """Moving-block bootstrap 95% interval for the mean of a serial sequence.

    `values` must be in span-start order. Blocks of `block` consecutive entries
    are resampled with replacement until the replicate is as long as the
    original, which preserves within-block dependence -- the reason a Wilson
    interval on overlapping spans is far too optimistic.

    Returns (low, high, half_width, n_effective) with n_effective = n / block,
    the number of genuinely independent spans the interval corresponds to.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    n = len(values)
    block = int(max(1, min(block, n)))
    if n < 2:
        return float(values.mean() if n else 0.0), float(values.mean() if n else 0.0), 0.0, n
    point = float(values.mean())
    if n_boot <= 0:                      # monitoring call: point estimate only
        return point, point, 0.0, int(max(1, round(n / block)))
    rng = np.random.default_rng(random_state)
    n_blocks = int(math.ceil(n / block))
    offsets = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    index = (offsets[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = values[index].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high), float((high - low) / 2), int(max(1, round(n / block)))


# --------------------------------------------------------------------------- #
# 4. MobileNet variant that can still see the firing level
# --------------------------------------------------------------------------- #
class _StatsPassthroughEncoder(nn.Module):
    """Re-inject per-group log1p(mean) and log1p(std) beside the trunk output.

    GroupNorm immediately after a bias-free convolution makes the trunk exactly
    scale-invariant: f(c*x) = f(x) to within 1e-5. That is a fine property for
    images and a disastrous one here, because the population gain IS the
    behaviour code -- the reconstruction anchor exists precisely to keep it.
    Two scalars per group restore it without giving up the normalization.
    """

    def __init__(self, trunk, channels, output_dimension, normalize, groups=8):
        super().__init__()
        self.trunk = trunk
        self.groups = max(1, math.gcd(channels, groups))
        self.channels = channels
        self.project = nn.Linear(trunk.out_features + 2 * self.groups, output_dimension)
        self.normalize = normalize

    def forward(self, x):
        h = self.trunk(x)
        grouped = x.reshape(x.shape[0], self.groups, -1)
        stats = torch.cat((torch.log1p(grouped.mean(-1).clamp_min(0)),
                           torch.log1p(grouped.std(-1))), dim=-1)
        z = self.project(torch.cat((h, stats), dim=-1))
        return F.normalize(z, dim=-1) if self.normalize else z


class LevelAwareMobileJigsaw(MobileJigsaw, OrderJigsaw):
    """MobileNet trunk + the level passthrough + every OrderJigsaw mechanism."""

    _MODEL_TYPE = "level_aware_mobile_jigsaw"
    _PARAM_NAMES = tuple(dict.fromkeys(MobileJigsaw._PARAM_NAMES
                                       + OrderJigsaw._PARAM_NAMES))

    def _make_encoder(self, channels):
        trunk = MobileJigsaw._make_encoder(self, channels).trunk
        return _StatsPassthroughEncoder(trunk, channels, self.output_dimension,
                                        self.normalize)


# --------------------------------------------------------------------------- #
# 5. arms
# --------------------------------------------------------------------------- #
#  The point of this table is that the ORDER EFFECT is measured against a
#  control that differs in ONE thing. `proposed - reconstruct_only` does not
#  qualify any more: those two arms differ in the order loss AND the extra
#  encoder forward AND the normalized view AND the augmentation draw. The
#  matched pair is `X` vs `X_anchor` (same everything, order weights zeroed)
#  and `X` vs `X_frozen` (same everything, labels are an arbitrary fixed
#  chronology). If a mechanism only beats `_anchor` and not `_frozen`, it is a
#  regularizer, not temporal-order learning, and the paper has to say so.
_FULL = dict(_model="order", tile_norm="none", span_selection="decorrelated",
             selection_pool=6, selection_jitter=12, order_head_kind="linear",
             lambda_lag=1.0, order_hard_fraction=0.25, order_head_reset_every=400,
             gap_curriculum=0.5, window_size=6, tile_gap=(1, 3))

ORDER_ARMS = {
    # -- the no-op control. MUST equal `proposed` to ~1e-6. ----------------- #
    "order_default": dict(_model="order"),

    # -- mechanism 1: make the task decidable ------------------------------ #
    # span 4*6+3*3+1 = 34 bins instead of 65, inside one autocorrelation time.
    "order_short_span": dict(_model="order", window_size=6, tile_gap=(1, 3)),
    "order_curriculum": dict(_model="order", gap_curriculum=0.5),

    # -- mechanism 2: stop the head memorizing ----------------------------- #
    "order_linear_head": dict(_model="order", order_head_kind="linear"),
    "order_reset": dict(_model="order", order_head_reset_every=400),
    "order_hard": dict(_model="order", order_hard_fraction=0.25),
    "order_relational": dict(_model="order", order_head_kind="relational"),

    # -- mechanism 3: same view as the decoder, shortcut killed by sampling - #
    # selection_jitter is NOT optional here. Gap choice alone can only
    # decorrelate a span whose drift reverses inside it; on the mirror only
    # 27.7% of spans had any balanced gap draw at all, so pool-without-jitter
    # leaves the level shortcut almost untouched. Moving the START is what
    # gives the sampler a genuinely different stretch of drift to choose from.
    "order_decorrelated": dict(_model="order", tile_norm="none",
                               span_selection="decorrelated", selection_pool=6,
                               selection_jitter=12),
    "order_adversarial": dict(_model="order", tile_norm="none",
                              span_selection="adversarial", selection_pool=6,
                              selection_jitter=12),
    # Positive control: pick the MOST level-predictable span every time. If the
    # encoder is exploiting drift, this arm should post the highest pretext
    # accuracy in the table and gain the least R2 -- that pattern is the proof
    # that the shortcut is what the other arms are avoiding.
    "order_shortcut": dict(_model="order", tile_norm="none",
                           span_selection="shortcut", selection_pool=6,
                           selection_jitter=12),
    "order_count_match": dict(_model="order", order_view="count_match"),
    "order_raw": dict(_model="order", order_view="raw"),

    # -- mechanism 4: a label worth more than log K ------------------------ #
    "order_lag": dict(_model="order", lambda_lag=1.0),
    "order_lag_only": dict(_model="order", lambda_order=0.0, lambda_pair=0.0,
                           lambda_lag=1.0),

    # -- schedules --------------------------------------------------------- #
    "order_anchor_decay": dict(_model="order", anchor_final_scale=0.25),
    "order_ramp": dict(_model="order", order_final_scale=3.0),

    # -- the proposal, and its two matched controls ------------------------ #
    "order_full": dict(_FULL),
    "order_full_anchor": dict(_FULL, lambda_order=0.0, lambda_pair=0.0, lambda_lag=0.0),
    "order_full_frozen": dict(_FULL, label_control="frozen_random"),
    "order_full_fresh": dict(_FULL, label_control="fresh_random"),
    "order_full_random": dict(_FULL, max_epochs=0),

    # -- controls on the stock settings, for comparison with the old table -- #
    "ctrl_frozen_labels": dict(_model="order", label_control="frozen_random"),
    "ctrl_fresh_labels": dict(_model="order", label_control="fresh_random"),

    # -- MobileNet with the level restored --------------------------------- #
    "mobile_level": dict(_model="mobile_level", version="v2", expansion=3),
    "mobile_level_full": dict(_FULL, _model="mobile_level", version="v2", expansion=3),
    "mobile_level_random": dict(_model="mobile_level", max_epochs=0),
}

# The pairs the runner should contrast. (arm, control, what it proves)
ORDER_CONTRASTS = (
    ("order_full", "order_full_anchor", "ORDER effect (matched)",
     "the order term against the SAME model with the order weights at zero."),
    ("order_full", "order_full_frozen", "ORDER effect vs FROZEN LABELS",
     "the honest one: same loss, same head, same gradient path, labels that "
     "mean nothing about time. If this is ~0 the term is a regularizer."),
    ("order_default", "ctrl_frozen_labels", "ORDER effect (stock settings)",
     "the same contrast on the old configuration. Measured +0.0225 on J-CO0."),
    ("order_full", "order_full_random", "vs FROZEN ENCODER",
     "training must be worth something before any of the above matters."),
)

DEFAULT_ORDER_ARMS = ("order_full", "order_full_anchor", "order_full_frozen",
                      "order_default", "ctrl_frozen_labels", "order_short_span",
                      "order_decorrelated", "order_linear_head", "order_lag",
                      "random_encoder")


def register(models, arms, contrasts=None):
    """Wire this module into run_jigsaw.py without editing its tables by hand."""
    models["order"] = OrderJigsaw
    models["mobile_level"] = LevelAwareMobileJigsaw
    for name, settings in ORDER_ARMS.items():
        arms[name] = dict(settings)
    if contrasts is not None:
        for row in ORDER_CONTRASTS:
            if row not in contrasts:
                contrasts.append(row)
    return models, arms


# --------------------------------------------------------------------------- #
# 6. self-test
# --------------------------------------------------------------------------- #
def _check(condition, message):
    print(f"  {'ok  ' if condition else 'FAIL'}  {message}")
    if not condition:
        raise AssertionError(message)


def _counts(n=2400, channels=12, seed=0):
    """Poisson counts with a slow multiplicative drift -- the shortcut, on purpose."""
    rng = np.random.default_rng(seed)
    time = np.arange(n)[:, None]
    phase = rng.uniform(0, 2 * np.pi, channels)[None, :]
    rate = 1.5 + 1.2 * np.sin(2 * np.pi * time / 90 + phase)
    drift = 1.0 + 0.8 * np.sin(2 * np.pi * time / 1500)
    return rng.poisson(np.clip(rate * drift, 0.05, None)).astype(np.float32)


def _smoke_test():
    print("jigsaw_order self-test\n")
    torch.manual_seed(0)
    data = _counts()

    print("1. count_match equalizes totals EXACTLY and never invents spikes")
    tiles = torch.from_numpy(data[:4 * 4 * 8].reshape(4, 4, 8, 12).astype(np.float32)
                             ).permute(0, 1, 3, 2).contiguous()
    generator = torch.Generator().manual_seed(3)
    matched = count_match(tiles, generator)
    totals = matched.sum(-1)
    _check(torch.allclose(totals, totals[:, :1].expand_as(totals)),
           "every tile has the same per-neuron total")
    _check(bool((matched <= tiles.round()).all()), "only ever removes spikes")
    _check(torch.equal(totals[:, 0], tiles.round().sum(-1).min(dim=1).values),
           "the common total is the per-neuron minimum over tiles")

    print("\n2. concordance is 1.0 for an increasing statistic, 0.0 reversed, 0.5 flat")
    rising = torch.arange(4.0)[None, :].expand(5, -1)
    _check(float(concordance(rising).mean()) == 1.0, "increasing -> 1.0")
    _check(float(concordance(-rising).mean()) == 0.0, "decreasing -> 0.0")
    _check(float(concordance(torch.zeros(5, 4)).mean()) == 0.5, "constant -> 0.5 (ties)")
    _check(float(shortcut_strength(rising).mean()) == 1.0
           and float(shortcut_strength(-rising).mean()) == 1.0,
           "shortcut_strength is blind to the sign: both directions score 1.0")

    print("\n3. the level shortcut is much larger than the fixed-rule number says,")
    print("   and only start-jitter selection actually moves it")
    device = torch.device("cpu")
    tensor = torch.from_numpy(data).to(device)
    valid = torch.arange(0, len(data) - 200, device=device)
    starts = valid[::7].contiguous()
    pool = torch.randint(1, 9, (8, len(starts), 3), generator=generator)
    plain, _ = _gather_tiles(tensor, starts, pool[0], 10)
    fixed = float(concordance(plain.sum(dim=(2, 3))).mean())
    base = float(shortcut_strength(plain.sum(dim=(2, 3))).mean())
    print(f"       fixed-rule concordance {100*fixed:.1f}%  vs  "
          f"oracle-sign strength {100*base:.1f}%")
    _check(abs(fixed - 0.5) < 0.05 and base > 0.70,
           "a 'shortcut = 50%' reading can hide a ~80% exploitable shortcut")

    # gaps only: the start is pinned, so a monotone stretch of drift stays monotone
    only_gaps, _ = select_span(tensor, starts, valid, pool, None, 10, "decorrelated")
    tiles_g, _ = _gather_tiles(tensor, starts, only_gaps, 10)
    gap_only = float(shortcut_strength(tiles_g.sum(dim=(2, 3))).mean())
    jitter = torch.randint(-20, 21, (8, len(starts)), generator=generator)
    jitter[0].zero_()
    moved, picked = select_span(tensor, starts, valid, pool, jitter, 10, "decorrelated")
    tiles_d, _ = _gather_tiles(tensor, moved, picked, 10)
    after = float(shortcut_strength(tiles_d.sum(dim=(2, 3))).mean())
    bad_starts, bad_gaps = select_span(tensor, starts, valid, pool, jitter, 10, "shortcut")
    tiles_s, _ = _gather_tiles(tensor, bad_starts, bad_gaps, 10)
    worst = float(shortcut_strength(tiles_s.sum(dim=(2, 3))).mean())
    print(f"       strength: random {100*base:.1f}% -> gaps only {100*gap_only:.1f}% "
          f"-> gaps+jitter {100*after:.1f}% -> shortcut arm {100*worst:.1f}%")
    _check(after < gap_only, "moving the START is what buys the decorrelation")
    _check(worst > base, "the positive control really does find worse spans")
    _check(bool((moved >= valid[0]).all() and (moved <= valid[-1]).all()),
           "jitter stays inside the valid-start list, so gathers can never run off")
    _check(torch.equal(select_span(tensor, starts, valid, pool, jitter, 10, "random")[0],
                       starts), "span_selection='random' leaves the starts untouched")

    print("\n4. lag targets are symmetric, in range, and use the magnitude")
    edges = lag_edges(10, 4, 1, 8, 6, device)
    tile_starts = torch.tensor([[0, 15, 30, 45], [0, 11, 22, 80]], dtype=torch.long)
    labels = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    target, mask = lag_targets(tile_starts, labels, edges, 6)
    _check(int(mask.sum()) == 2 * 6, "6 upper-triangular pairs per span")
    _check(bool(((target >= 0) & (target < 6)).all()), "every class in range")
    # Row 0 is presented chronologically: slot i is always EARLIER than slot j
    # for i<j, so every class falls in the negative half (0..2). Row 1 is
    # presented backwards, so every class falls in the positive half (3..5).
    _check(int(target[:6].max()) <= 2 and int(target[6:].min()) >= 3,
           "chronological row is all-negative, reversed row all-positive")
    far = lag_targets(torch.tensor([[0, 11, 22, 200]]), torch.tensor([[0, 1, 2, 3]]),
                      edges, 6)[0]
    near = lag_targets(torch.tensor([[0, 11, 22, 33]]), torch.tensor([[0, 1, 2, 3]]),
                       edges, 6)[0]
    # NOT `far.min() < near.min()`: both spans contain the same tight neighbour
    # pairs, so both reach the extreme band and both minima are 0. What separates
    # them is HOW MANY pairs land there -- 3 for the stretched span, 1 for the
    # compact one. The first form passes for the wrong reason or not at all.
    _check(int((far == 0).sum()) > int((near == 0).sum()),
           f"{int((far == 0).sum())} vs {int((near == 0).sum())} pairs in the far "
           "band -- magnitude matters, not just direction")

    print("\n5. block bootstrap widens with block length and reports n_effective")
    noise = np.repeat(np.random.default_rng(1).normal(0.5, 0.2, 40), 20)
    _, _, tight, n_tight = block_bootstrap(noise, 1, n_boot=400, random_state=2)
    _, _, wide, n_wide = block_bootstrap(noise, 20, n_boot=400, random_state=2)
    print(f"       half-width: block=1 {100*tight:.2f}%  block=20 {100*wide:.2f}%  "
          f"(n_eff {n_tight} -> {n_wide})")
    _check(wide > tight, "a correlated series gets a WIDER interval, as it must")
    _check(n_wide == 40 and n_tight == 800, "n_effective = n / block")

    print("\n6. the linear head is permutation-equivariant and tiny")
    head = _LinearOrderHead(16, 4)
    z = torch.randn(3, 4, 16)
    perm = torch.tensor([2, 0, 3, 1])
    a, b = head(z)
    c, d = head(z[:, perm])
    _check(torch.allclose(a[:, perm], c, atol=1e-6), "position logits follow the tiles")
    _check(torch.allclose(b[:, perm], d, atol=1e-6), "scores follow the tiles")
    _check(sum(p.numel() for p in head.parameters()) < 200,
           f"{sum(p.numel() for p in head.parameters())} parameters -- cannot memorize")

    print("\n7. OrderJigsaw() with no options reproduces JigsawNet() exactly")
    common = dict(window_size=8, n_tiles=3, tile_gap=(1, 3), output_dimension=16,
                  num_hidden_units=16, head_hidden_units=16, batch_size=64,
                  max_epochs=3, device="cpu", random_state=5, verbose=False)
    a = JigsawNet(**common).fit(data)
    b = OrderJigsaw(**common).fit(data)
    gap = np.abs(a.transform(data) - b.transform(data)).max()
    print(f"       max |JigsawNet - OrderJigsaw| = {gap:.2e}")
    _check(gap < 1e-4, "the default is a genuine no-op control")

    print("\n8. every mechanism trains and evaluates")
    for name, extra in (("short_span", dict(window_size=6, tile_gap=(1, 2))),
                        ("linear_head", dict(order_head_kind="linear")),
                        ("relational", dict(order_head_kind="relational")),
                        ("decorrelated", dict(tile_norm="none",
                                              span_selection="decorrelated",
                                              selection_pool=4)),
                        ("decorr+jitter", dict(tile_norm="none",
                                               span_selection="decorrelated",
                                               selection_pool=4,
                                               selection_jitter=15)),
                        ("adversarial", dict(tile_norm="none",
                                             span_selection="adversarial",
                                             selection_pool=4,
                                             selection_jitter=15)),
                        ("shortcut_ctl", dict(tile_norm="none",
                                              span_selection="shortcut",
                                              selection_pool=4,
                                              selection_jitter=15)),
                        ("count_match", dict(order_view="count_match")),
                        ("lag", dict(lambda_lag=1.0)),
                        ("hard", dict(order_hard_fraction=0.25)),
                        ("reset", dict(order_head_reset_every=2)),
                        ("curriculum", dict(gap_curriculum=0.5)),
                        ("schedules", dict(anchor_final_scale=0.25,
                                           order_final_scale=3.0)),
                        ("frozen_labels", dict(label_control="frozen_random")),
                        ("fresh_labels", dict(label_control="fresh_random"))):
        settings = dict(common); settings.update(extra)
        model = OrderJigsaw(**settings).fit(data)
        metrics = model.evaluate_pretext(data, max_spans=160, repeats=2,
                                         n_boot=200, verbose=False)
        embedding = model.transform(data)
        _check(np.isfinite(embedding).all() and math.isfinite(
            metrics["pair_accuracy_percent"]),
            f"{name}: pair={metrics['pair_accuracy_percent']:.1f}% "
            f"+-{metrics['pair_block_resolution_percent']:.1f} "
            f"n_eff={metrics['n_effective']}")

    print("\n9. count_match refuses non-integer data instead of silently lying")
    try:
        OrderJigsaw(order_view="count_match", **common).fit(data / 3.7 + 0.5)
        _check(False, "should have raised")
    except ValueError as error:
        _check("integer" in str(error).lower(), "clear error on non-integer input")

    print("\n10. the head reset changes the head, keeps the tensors, and spares the norms")
    model = OrderJigsaw(order_head_kind="relational", order_head_reset_every=1000, **common)
    model.fit(data)
    named = dict(model.order_head_.named_parameters())
    before = {k: v.detach().clone() for k, v in named.items()}
    identity = {k: v for k, v in named.items()}          # same objects afterwards?
    model._reset_order_head()
    moved = max(float((before[k] - named[k]).abs().max()) for k in named)
    _check(moved > 1e-3, f"parameters moved by {moved:.3f}")
    _check(all(identity[k] is dict(model.order_head_.named_parameters())[k] for k in named),
           "the parameter OBJECTS are unchanged, so the optimizer still owns them")
    norms = [v for k, v in named.items() if v.dim() == 1 and not k.endswith("bias")]
    _check(norms and all(float(v.min()) == 1.0 for v in norms),
           f"{len(norms)} normalization scales restored to 1, not zeroed")
    _check(model.lag_head_ is model.order_head_.lag_head
           and any("lag_head" in k for k in named),
           "the lag head rides inside order_head_, so fit's optimizer trains it")

    print("\n11. the level passthrough breaks MobileNet's exact scale invariance")
    plain = MobileJigsaw(**common, version="v2").fit(data)
    aware = LevelAwareMobileJigsaw(**common, version="v2").fit(data)
    probe = torch.from_numpy(data[:64].T[None].astype(np.float32)).repeat(4, 1, 1)
    probe = probe[:, :, :common["window_size"]]
    for label, model in (("stock ", plain), ("level ", aware)):
        model.encoder_.eval()
        with torch.no_grad():
            one, two = model.encoder_(probe), model.encoder_(2 * probe)
        rel = float((two - one).norm() / one.norm().clamp_min(1e-8))
        print(f"       {label}|f(2x)-f(x)|/|f(x)| = {rel:.5f}")
        if label.strip() == "level":
            _check(rel > 1e-3, "the level-aware trunk can see a doubling")

    print("\n12. remove_slow_drift flattens the drift without touching fast structure")
    flat = remove_slow_drift(data, 400)
    slow = lambda x: np.abs(np.diff(x.mean(1).reshape(-1, 50).mean(1))).mean()
    print(f"       slow-component drift: {slow(data):.4f} -> {slow(flat):.4f}")
    _check(slow(flat) < slow(data), "the slow component shrank")

    print("\nall checks passed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--list", action="store_true")
    options = parser.parse_args()
    if options.list:
        width = max(len(k) for k in ORDER_ARMS)
        for name, settings in ORDER_ARMS.items():
            print(f"  {name:<{width}}  {settings}")
    else:
        _smoke_test()

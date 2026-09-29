"""Pure Jigsaw self-supervision: no reconstruction, forecast, or InfoNCE.

Grouped temporal Jigsaw independently permutes K chunks for each neuron group.
With n_groups=0 each neuron is a group; group_chimera=False uses one shared
permutation with the same heads. Stitch predicts successor slots, spatial
Jigsaw predicts neuron-block identities, and row-phase Jigsaw predicts an
independent cyclic shift for each neuron's row. The shift head is affine.

These objectives may encourage richer embeddings; neither independent labels,
subspaces, nor a linear readout guarantees an N-dimensional representation.
Participation ratio and held-out transfer must measure that hypothesis.

For independent uniform transformations, the nominal JOINT label entropy is
G*log(K!) for temporal permutations, N*log(W) for row phase, and log(G!) for
spatial permutations (natural logarithms). A shared temporal permutation counts
only once. With N=38, K=4, W=10 the combined temporal+shift value is about 208
nats BEFORE masking. This is target entropy, not recoverable information or a
supervision budget guaranteed to reach the encoder. Constant/periodic rows are
excluded from shift CE and accuracy; even distinct rotations do not guarantee
that a canonical phase is identifiable from the data distribution.

The earlier reconstruction target's N*W*log(8) is an alphabet-size upper bound,
not its measured entropy. Historical R2=0.5450 (MLP), 0.3484 (ridge) and
participation ratio 18.76 are reference observations, not a matched baseline
re-run by this file. The previous two-seed Solo run gave solo_full=0.4229,
solo_random=0.3280, solo_space_only=0.3199, and solo_rank8=0.2997 (MLP).

Important implementation contracts:
  - Input transformations are sampled independently of label controls.
    Frozen/fresh controls replace training targets ONLY. Evaluation always
    scores the true transformation, including for frozen and untrained arms.
  - Oracle templates use every valid TRAIN span start, in bounded batches,
    with evaluation-time normalization. They are frozen before evaluation;
    validation data never calibrate them. Buffers travel with the head state.
  - Shift accuracy is pooled over eligible rows; no eligible rows means None.
  - lambda_spread is an optional variance regularizer, not a Jigsaw objective.
  - Eleven default arms; reconstruction/forecast weights must remain zero.

The public fit/transform/evaluate_pretext/save/load API is inherited from
OrderJigsaw. Existing checkpoints with the old MLP shift head need retraining;
this revision changes the head and fixes the meaning of frozen-label controls.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from jigsaw_net import (_GradScale, _OrderHead, _assign, _augment, _boolean,
                        _gather_tiles, _integer, _order_metrics, _pair_targets,
                        _participation_ratio, _ranks, _real)
from jigsaw_order import (ANCHOR_MATCH_KEYS, OrderJigsaw, block_bootstrap,
                          continuity_strength, roll_tiles, select_span,
                          shortcut_strength)

# Two families, and the difference between them is the whole audit.
#   block_* removes a statistic of the WHOLE block, so "neuron 7 fires twice as
#           often as its blockmate neuron 12" survives -- a constant of the
#           session, learnable once and reusable on every span.
#   row_*   removes that statistic PER NEURON, so no neuron's mean rate (and
#           under row_zscore no neuron's gain either) can name the block. What
#           is left is the temporal shape of each row, which is a fact about
#           this span and not about the session.
# The 20260928 run only ever used block_mean, so its 85-96% space accuracy is
# not yet evidence of anything beyond static neuron identity.
SPACE_NORMS = ("none", "block_mean", "block_zscore", "row_mean", "row_zscore")


# --------------------------------------------------------------------------- #
# 1. heads
# --------------------------------------------------------------------------- #
class _GroupOrderHead(nn.Module):
    """K-way slot logits and a before/after score, FOR EACH of G groups.

    subspace=True: group g sees only z[..., g*w:(g+1)*w], through shared
    readout weights. subspace=False: every group sees the full embedding with
    separate output rows. Neither wiring proves a global covariance-rank floor;
    correlated slices, priors and head capacity still matter.

    At G=1 with subspace=True this is `_OrderHead` exactly: same input width,
    same body, same two readouts, same parameter count. `solo_rank1` is
    therefore the classic jigsaw with the anchor removed, not a new model that
    happens to resemble it.
    """

    def __init__(self, dimension, hidden, n_tiles, n_groups, subspace):
        super().__init__()
        self.n_tiles, self.n_groups, self.subspace = n_tiles, n_groups, subspace
        self.width = dimension // n_groups if subspace else dimension
        self.body = nn.Sequential(nn.Linear(2 * self.width, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU())
        outputs = n_tiles if subspace else n_groups * n_tiles
        self.position = nn.Linear(hidden, outputs)
        self.score = nn.Linear(hidden, 1 if subspace else n_groups)

    def slices(self, z):
        """(B, K, D) -> (B, G, K, width). The view every group-wise term uses."""
        batch, n_tiles = z.shape[0], z.shape[1]
        if not self.subspace:
            return z[:, None].expand(-1, self.n_groups, -1, -1)
        return z.reshape(batch, n_tiles, self.n_groups,
                         self.width).permute(0, 2, 1, 3)

    def forward(self, z):
        batch, n_tiles = z.shape[0], z.shape[1]
        if self.subspace:
            parts = self.slices(z)                                # (B,G,K,w)
            context = parts.mean(dim=2, keepdim=True).expand_as(parts)
            tokens = self.body(torch.cat((parts, context), dim=-1))
            return self.position(tokens), self.score(tokens).squeeze(-1)
        context = z.mean(dim=1, keepdim=True).expand_as(z)
        tokens = self.body(torch.cat((z, context), dim=-1))       # (B,K,H)
        position = self.position(tokens).reshape(
            batch, n_tiles, self.n_groups, self.n_tiles).permute(0, 2, 1, 3)
        score = self.score(tokens).permute(0, 2, 1)               # (B,G,K)
        return position, score


class _StitchHead(nn.Module):
    """Bilinear successor scores plus a last-tile class; self-pairs are masked.

    This limits readout capacity relative to an MLP, but does not prove that the
    encoder cannot memorize spans or use shortcuts. Held-out controls test that.
    """

    def __init__(self, width, rank):
        super().__init__()
        self.query = nn.Linear(width, rank, bias=False)
        self.key = nn.Linear(width, rank, bias=False)
        self.last = nn.Linear(width, 1)
        self.scale = rank ** -0.5

    def forward(self, parts):                                     # (B,G,K,w)
        query, key = self.query(parts), self.key(parts)
        logits = torch.einsum("bgir,bgjr->bgij", query, key) * self.scale
        n_tiles = parts.shape[2]
        eye = torch.eye(n_tiles, dtype=torch.bool, device=parts.device)
        logits = logits.masked_fill(eye, float("-inf"))
        return torch.cat((logits, self.last(parts)), dim=-1)      # (B,G,K,K+1)


class _SpaceHead(nn.Module):
    """G-way "which neuron group is sitting in this row slot?" per slot."""

    def __init__(self, dimension, hidden, n_groups):
        super().__init__()
        self.n_groups = n_groups
        self.net = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_groups * n_groups))

    def forward(self, z):                                         # (B,D)
        return self.net(z).reshape(-1, self.n_groups, self.n_groups)


class _ShiftHead(nn.Module):
    """Affine W-way readout per neuron; it does not impose a rank lower bound."""

    def __init__(self, dimension, n_neurons, n_shifts):
        super().__init__()
        self.n_neurons, self.n_shifts = n_neurons, n_shifts
        self.net = nn.Linear(dimension, n_neurons * n_shifts)

    def forward(self, z):                                         # (B,D)
        return self.net(z).reshape(-1, self.n_neurons, self.n_shifts)


# --------------------------------------------------------------------------- #
# 2. functional pieces
# --------------------------------------------------------------------------- #
def successor_targets(labels):
    """(B,G,K) chronological labels -> (B,G,K) successor SLOT, class K = last.

    `labels[b,g,j]` is the chronological index of the chunk presented in slot j,
    so `labels.argsort(-1)[b,g,c]` is the slot holding chunk c -- the inverse
    permutation. The successor of slot j is therefore the slot holding chunk
    labels[b,g,j] + 1, and the tile holding the final chunk gets the extra
    class. Written out rather than looped because it runs every step.
    """
    n_tiles = labels.shape[-1]
    inverse = labels.argsort(dim=-1)
    following = torch.gather(inverse, -1, (labels + 1).clamp(max=n_tiles - 1))
    return torch.where(labels == n_tiles - 1,
                       torch.full_like(following, n_tiles), following)


def place_groups(tiles, labels, group_index):
    """Build the chimera: (B,K,N,W) tiles + (B,G,K) labels -> (B,K,N,W).

    Row n of presented tile j is taken from chronological chunk
    labels[b, group_index[n], j]. With G=1 this reduces to the ordinary tile
    shuffle, element for element, which is what keeps `solo_rank1` a true
    control.
    """
    index = labels[:, group_index, :].permute(0, 2, 1)            # (B,K,N)
    index = index[..., None].expand(-1, -1, -1, tiles.shape[3])
    return torch.gather(tiles, 1, index)


def permute_blocks(window, slots, sigma):
    """Row-permute equal-size neuron blocks. (B,N,W), (G,S), (B,G) -> (B,N,W).

    `slots[g]` lists the channel indices belonging to group g, TRUNCATED to a
    common size S so that every block is the same height. Unequal blocks would
    hand the answer over for free -- a five-row block among four-row blocks
    names itself -- and the N mod G leftover channels are therefore left where
    they are rather than being padded into a block.

    `sigma[b, s]` is the true group placed in row slot s, which is exactly the
    label the space head is asked to produce.
    """
    rows = slots.reshape(-1)                                      # (G*S,)
    source = slots[sigma].reshape(window.shape[0], -1)            # (B,G*S)
    gathered = torch.gather(
        window, 1, source[..., None].expand(-1, -1, window.shape[2]))
    out = window.clone()
    out[:, rows, :] = gathered
    return out


def normalize_blocks(window, slots, mode):
    """Remove a cue the space task would otherwise ride on. (B,N,W) -> (B,N,W).

    Every mode here is computed per BLOCK or per ROW of a block, never over the
    whole window, so the statistic travels with the block wherever the row
    permutation puts it. That is what keeps the normalization from leaking the
    answer: slot s is normalized by the contents of slot s, and those contents
    are whichever group landed there.

    none          nothing removed. The rate cue is fully available.
    block_mean    the block's scalar level. Relative rates WITHIN the block
                  survive, and those are session constants.
    block_zscore  the level and the overall scale, same caveat.
    row_mean      each neuron's own mean over the window. No neuron's rate can
                  name its block any more; gains still can.
    row_zscore    each neuron's mean AND gain. Only the temporal shape of each
                  row is left, which is the one cue that is about this span.
                  Rows that never fire in the window become exactly zero, so a
                  silent neuron stops being an identity beacon.
    """
    if mode == "none":
        return window
    rows = slots.reshape(-1)
    groups, block = slots.shape
    patch = window[:, rows, :].reshape(len(window), groups, block, -1)
    if mode in ("row_mean", "row_zscore"):
        centered = patch - patch.mean(dim=3, keepdim=True)
        if mode == "row_zscore":
            deviation = patch.std(dim=3, keepdim=True)
            centered = torch.where(deviation > 1e-4, centered / (deviation + 1e-5),
                                   torch.zeros_like(centered))
    else:
        flat = patch.reshape(len(window), groups, -1)
        centered = flat - flat.mean(dim=2, keepdim=True)
        if mode == "block_zscore":
            centered = centered / (flat.std(dim=2, keepdim=True) + 1e-5)
        centered = centered.reshape(patch.shape)
    out = window.clone()
    out[:, rows, :] = centered.reshape(len(window), len(rows), -1)
    return out


def shuffle_time(window, generator):
    """Permute the W time bins, one permutation per sample, shared by all rows.

    The control for "does the spatial puzzle need dynamics at all?". Every
    instantaneous population vector is preserved exactly and so is every
    neuron's mean rate, variance and cross-neuron covariance; the only thing
    destroyed is the ORDER of the bins. A model reading static neuron identity
    scores exactly the same with this on. A model reading each block's temporal
    signature does not.
    """
    samples, channels, width = window.shape
    order = torch.rand(samples, width, device=window.device,
                       generator=generator).argsort(dim=1)
    return torch.gather(window, 2, order[:, None, :].expand(-1, channels, -1))


def nearest_template(patch, template):
    """(B,G,F) candidates against (G,F) templates -> (B,G) predicted group.

    The label-free opponent for the spatial puzzle. Cosine and not squared
    distance on purpose: the template is averaged over training windows and
    individual evaluation windows can differ in scale. Under L2 that scale alone shifts the argmin toward
    whichever template happens to be smallest, and the oracle would then lose
    for a bookkeeping reason instead of a real one.
    """
    return (F.normalize(patch, dim=-1)
            @ F.normalize(template, dim=-1).t()).argmax(-1)


def roll_rows(window, shift):
    """(B,N,W) rolled along time by shift[b,n], INDEPENDENTLY for every neuron.

    `out[b,n,t] = window[b, n, (t - shift[b,n]) % W]`, i.e. a right-roll, so
    shift 0 is the identity and the W shifts form a cyclic group. This is the
    jigsaw on the time axis at the finest grain the window allows: a cyclic
    permutation is a permutation, recovering it is a classification, and doing
    it per neuron is what makes the target N-dimensional instead of scalar.

    Rolling and not cropping is deliberate. A crop would change each row's
    CONTENT, and "which W bins am I looking at" is answerable from level alone
    -- the shortcut this project has spent three runs removing. A roll keeps
    every row's multiset of values exactly, so the sorted row carries no
    information about the answer and only POSITION does.
    """
    width = window.shape[2]
    time = torch.arange(width, device=window.device)
    index = (time[None, None, :] - shift[:, :, None]) % width
    return torch.gather(window, 2, index)


def unique_cyclic_rows(window, atol=1e-6):
    """True only for rows with W distinct cyclic rotations, up to atol.

    Test all nonidentity rotations: a nonconstant row can still repeat with
    period 2, 5, etc. Exclusion is invariant under rolling and does not depend
    on either the sampled shift or the training label control.
    """
    width = window.shape[-1]
    unique = torch.ones(window.shape[:-1], dtype=torch.bool, device=window.device)
    if width < 2:
        return torch.zeros_like(unique)
    for shift in range(1, width):
        identical = torch.isclose(window, torch.roll(window, shift, dims=-1),
                                  rtol=0.0, atol=atol).all(dim=-1)
        unique = unique & ~identical
    return unique


def masked_span_loss(losses, eligible):
    """Per-span means, rescaled so empty spans do not dilute batch.mean().

    An all-empty batch returns differentiable zeros. Losses must be finite
    before masking (cross-entropy of finite logits satisfies that contract).
    """
    weight = eligible.to(losses.dtype)
    count = weight.sum(1)
    means = (losses * weight).sum(1) / count.clamp_min(1.0)
    active = (count > 0).to(losses.dtype)
    return means * (len(losses) / active.sum().clamp_min(1.0))


def target_entropy_nats(n_groups, n_tiles, n_neurons, window_size, *,
                        temporal, shared, shift, spatial):
    """Nominal joint entropy of independent uniform transformation labels.

    This is before shift masking and describes transformation truth, not the
    empirical entropy of a finite frozen-label table or recoverable evidence.
    """
    return float(((1 if shared else n_groups) * math.lgamma(n_tiles + 1)
                  if temporal else 0.0)
                 + (n_neurons * math.log(window_size) if shift else 0.0)
                 + (math.lgamma(n_groups + 1) if spatial else 0.0))


def roll_scores(rows, template):
    """(B,N,W) against a per-neuron (N,W) template -> (B,N,W) cosine per shift.

    The label-free opponent for the row-phase puzzle, and the one number that
    decides whether mechanism 6 means anything. The rule it implements is
    "every neuron has a canonical session-average shape; slide the template
    around the circle and report the offset that fits best". That rule needs no
    labels and no training, so if the model only ties it, the model learned a
    session constant and not this span.

    Centered cosine for the same reason as `nearest_template`: individual windows can differ in scale from the
    training-average template, and an uncentered score would be dominated by each row's mean
    rate, which is shift-invariant and therefore pure noise for this question.
    """
    width = rows.shape[-1]
    time = torch.arange(width, device=rows.device)
    index = (time[None, :] - time[:, None]) % width                   # (S,W)
    rolled = torch.gather(
        template[:, None, :].expand(-1, width, -1), 2,
        index[None].expand(len(template), -1, -1))                    # (N,S,W)
    rolled = rolled - rolled.mean(-1, keepdim=True)
    centered = rows - rows.mean(-1, keepdim=True)
    numerator = torch.einsum("bnw,nsw->bns", centered, rolled)
    denominator = (centered.norm(dim=-1)[:, :, None]
                   * rolled.norm(dim=-1)[None].clamp_min(1e-8))
    return numerator / denominator.clamp_min(1e-8)


def group_partition(channels, n_groups, seed):
    """Balanced groups over a RANDOM assignment of neurons.

    Contiguous groups would be groups of neighbouring electrodes, and on array
    recordings neighbouring electrodes share tuning -- the puzzle would then be
    easier for reasons that have nothing to do with the model. Round-robin over
    a seeded permutation keeps the sizes within one of each other while making
    membership arbitrary.
    """
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(channels, generator=generator)
    index = torch.empty(channels, dtype=torch.long)
    index[order] = torch.arange(channels) % n_groups
    return index


# --------------------------------------------------------------------------- #
# 3. the model
# --------------------------------------------------------------------------- #
class SoloJigsaw(OrderJigsaw):
    """A jigsaw that has to stand on its own: no reconstruction, no forecast.

    n_groups         independent puzzles per span. 1 = the classic jigsaw.
                     0 = ONE GROUP PER NEURON, resolved at fit time from the
                     data (G = N). That is the high-rank end of the ladder and
                     requires group_subspace=False.
    group_chimera    True  -> one permutation PER GROUP (the rank mechanism)
                     False -> one permutation shared by all groups, same
                              architecture. This is the control for n_groups.
    group_subspace   group g reads only its own D/G slice of the embedding.
                     Requires output_dimension % n_groups == 0.
    group_seed       which random partition of the neurons to use.
    lambda_stitch    weight on the successor CE (bilinear head, no lookup table)
    stitch_rank      rank of that bilinear form
    lambda_shift     weight on the row-phase puzzle: each neuron's row is rolled
                     around the window by its own amount and the model names all
                     N of them. `window_size` sets the class count. Rows that
                     have repeated cyclic rotations are masked out of the loss.
    lambda_space     weight on the neuron-axis (spatial) jigsaw
    space_norm       which cue to delete from the spatial puzzle. See
                     `normalize_blocks`: the block_* modes leave per-neuron
                     rates intact, the row_* modes do not.
    space_time_shuffle
                     permute the window's time bins before the spatial puzzle.
                     Keeps every static statistic, destroys temporal order.
    lambda_spread    variance floor on the embedding. NOT a puzzle: a
                     degeneracy guard, off by default, with its own arm so its
                     contribution never hides inside a headline number.

    Everything inherited from OrderJigsaw still applies: span selection, the
    gap curriculum, order-view-only augmentation, hard-example mining, head
    resets, and the label controls. `lambda_lag`
    is not implemented in this subclass and is rejected; use OrderJigsaw for it.
    """

    _MODEL_TYPE = "solo_jigsaw"
    _PARAM_NAMES = OrderJigsaw._PARAM_NAMES + (
        "n_groups", "group_chimera", "group_subspace", "group_seed",
        "lambda_stitch", "stitch_rank", "lambda_shift", "lambda_space",
        "space_norm", "space_time_shuffle", "lambda_spread")

    def __init__(self, *, n_groups=1, group_chimera=True, group_subspace=True,
                 group_seed=101, lambda_stitch=0.0, stitch_rank=32,
                 lambda_shift=0.0, lambda_space=0.0, space_norm="none",
                 space_time_shuffle=False, lambda_spread=0.0, **kwargs):
        for forbidden in ("lambda_reconstruct", "lambda_forecast"):
            if float(kwargs.pop(forbidden, 0.0)) != 0.0:
                raise ValueError(
                    f"{forbidden} must be 0 in SoloJigsaw. This file exists to "
                    "measure what the PUZZLE is worth with no anchor holding it "
                    "up; an arm that quietly turns the anchor back on would "
                    "answer a question we already answered (the anchor is worth "
                    "a lot). Use jigsaw_order.OrderJigsaw for anchored arms.")
        if space_norm not in SPACE_NORMS:
            raise ValueError(f"space_norm must be one of {SPACE_NORMS}.")
        # super()'s "all loss weights are zero" guard predates the solo terms
        # and would reject a space-only or stitch-only arm. Build with
        # max_epochs=0 so the guard cannot fire, restore the real budget, then
        # run the SAME guard over the real weight set below.
        epochs = kwargs.pop("max_epochs", 60)
        super().__init__(max_epochs=0, lambda_reconstruct=0.0,
                         lambda_forecast=0.0, **kwargs)
        self.max_epochs = _integer("max_epochs", epochs, 0)
        # 0 is "one group per neuron" and is resolved in _build, where the
        # channel count is known. Everything downstream reads `self.n_groups_`,
        # never `self.n_groups`, so the sentinel cannot leak into a shape.
        self.n_groups = _integer("n_groups", n_groups, 0)
        self.group_chimera = _boolean("group_chimera", group_chimera)
        self.group_subspace = _boolean("group_subspace", group_subspace)
        self.group_seed = _integer("group_seed", group_seed, 0)
        self.lambda_stitch = _real("lambda_stitch", lambda_stitch, 0)
        self.stitch_rank = _integer("stitch_rank", stitch_rank, 1, 512)
        self.lambda_shift = _real("lambda_shift", lambda_shift, 0)
        self.lambda_space = _real("lambda_space", lambda_space, 0)
        self.space_norm = space_norm
        self.space_time_shuffle = _boolean("space_time_shuffle", space_time_shuffle)
        self.lambda_spread = _real("lambda_spread", lambda_spread, 0)
        if self.n_groups == 0 and self.group_subspace:
            raise ValueError(
                "n_groups=0 means one group per neuron, and D/N is not an "
                "integer in general -- 64/38 is not. Set group_subspace=False: "
                "at this width the rank has to be demanded by the TASK rather "
                "than donated by the wiring, which is the stronger claim "
                "anyway.")
        if self.n_groups > 0 and self.group_subspace \
                and self.output_dimension % self.n_groups:
            raise ValueError(
                f"group_subspace splits the embedding into n_groups slices, so "
                f"output_dimension ({self.output_dimension}) must be divisible "
                f"by n_groups ({self.n_groups}). Pick 64/8, 64/4, 60/5 and so "
                f"on, or set group_subspace=False.")
        if self.lambda_space > 0 and self.n_groups == 1:
            raise ValueError("lambda_space needs more than one block: with one "
                             "block there is no spatial permutation to recover.")
        if self.lambda_space > 0 and self.n_groups == 0:
            raise ValueError(
                "lambda_space at one group per neuron is an N-way puzzle over "
                "1-row blocks, which is the 'name this neuron from its own row' "
                "task -- a pure session constant with no span content in it at "
                "all. Use a real block count, or lambda_shift, which asks about "
                "this span.")
        if self.lambda_lag > 0:
            raise ValueError("SoloJigsaw does not implement lag loss; use "
                             "OrderJigsaw for lambda_lag > 0.")
        live = (self.lambda_order + self.lambda_pair + self.lambda_lag
                + self.lambda_stitch + self.lambda_shift + self.lambda_space
                + self.lambda_spread)
        if live == 0 and self.max_epochs > 0:
            raise ValueError("All loss weights are zero; set max_epochs=0 for a "
                             "random encoder.")
        if not self.shuffle_tiles and self.lambda_order + self.lambda_pair > 0:
            raise ValueError(
                "shuffle_tiles=False hands every group the identity permutation, "
                "so the grouped puzzle has a constant answer and the whole rank "
                "argument evaporates. Keep it True.")

    # -- construction ------------------------------------------------------- #
    def _build(self, channels):
        # OrderJigsaw._build gives us the frozen label table, the lag head, the
        # lag edges, the span cache and the step counter. We keep all of it and
        # only swap the order head, so the inherited machinery stays wired.
        super()._build(channels)
        # THE sentinel resolution, and the only place it happens. n_groups=0
        # means "one group per neuron", which cannot be known until the channel
        # count is; from here on the file reads n_groups_ exclusively.
        self.n_groups_ = channels if self.n_groups == 0 else self.n_groups
        if self.n_groups_ > channels:
            raise ValueError(f"n_groups={self.n_groups} exceeds the {channels} "
                             "recorded neurons.")
        self.group_index_ = group_partition(
            channels, self.n_groups_, self.random_state + self.group_seed
        ).to(self.device_)
        sizes = torch.bincount(self.group_index_, minlength=self.n_groups_)
        self.group_sizes_ = sizes.tolist()
        lag_head = self.order_head_.lag_head
        self.order_head_ = _GroupOrderHead(
            self.output_dimension, self.head_hidden_units, self.n_tiles,
            self.n_groups_, self.group_subspace).to(self.device_)
        self.order_head_.lag_head = lag_head
        self.lag_head_ = lag_head
        if self.lambda_stitch > 0:
            self.order_head_.stitch = _StitchHead(
                self.order_head_.width, self.stitch_rank).to(self.device_)
        if self.lambda_shift > 0:
            if self.window_size < 2:
                raise ValueError(
                    "lambda_shift rolls each row around a window of "
                    f"{self.window_size} bin(s); with fewer than 2 there is only "
                    "the identity roll and the label is constant. Raise "
                    "window_size.")
            self.order_head_.shift = _ShiftHead(
                self.output_dimension, channels,
                self.window_size).to(self.device_)
            self.order_head_.register_buffer(
                "shift_profile", torch.zeros(channels, self.window_size,
                                             device=self.device_))
        if self.lambda_space > 0:
            self.order_head_.space = _SpaceHead(
                self.output_dimension, self.head_hidden_units,
                self.n_groups_).to(self.device_)
            # Equal-height blocks. See `permute_blocks` for why the leftover
            # channels are dropped from the spatial puzzle rather than padded.
            block = int(sizes.min())
            if block < 1:
                raise ValueError(
                    f"n_groups={self.n_groups_} over {channels} neurons leaves an "
                    "empty group; the spatial puzzle needs at least one row per "
                    "block.")
            members = [torch.nonzero(self.group_index_ == g, as_tuple=False)
                       .flatten()[:block] for g in range(self.n_groups_)]
            self.space_slots_ = torch.stack(members).to(self.device_)
            self.space_block_ = block
            self.order_head_.register_buffer(
                "group_profile", torch.zeros(self.n_groups_, block,
                                             self.window_size, device=self.device_))
        self.order_head_.register_buffer(
            "oracle_profile_samples", torch.zeros((), dtype=torch.long,
                                                  device=self.device_))
        # (4096, G, K) and (4096, G): the frozen-label control, group-aware.
        # Same contract as the inherited table -- one arbitrary but FIXED answer
        # per block of span starts, so the head can still learn it and the only
        # thing removed is that the answer means anything.
        table = torch.Generator(device=self.device_)
        table.manual_seed(self.random_state + 1861)
        self._frozen_groups = torch.rand(
            4096, self.n_groups_, self.n_tiles, device=self.device_,
            generator=table).argsort(dim=2)
        self._frozen_space = torch.rand(
            4096, self.n_groups_, device=self.device_, generator=table).argsort(dim=1)
        # Uniform over W classes and FIXED per row block, so the frozen-shift
        # arm has exactly the same label marginal as the true one -- otherwise
        # the control would be easier for a reason that has nothing to do with
        # meaning. randint and not argsort: the shifts are independent draws
        # from {0..W-1}, not a permutation of them.
        self._frozen_shift = torch.randint(
            0, self.window_size, (4096, channels), device=self.device_,
            generator=table)

    @property
    def shift_profile_(self):
        return (getattr(self.order_head_, "shift_profile", None)
                if self.oracle_profile_samples_ else None)

    @property
    def group_profile_(self):
        return (getattr(self.order_head_, "group_profile", None)
                if self.oracle_profile_samples_ else None)

    @property
    def oracle_profile_samples_(self):
        return int(self.order_head_.oracle_profile_samples.item())

    def _fit_oracle_profiles(self, data, starts, batch_size=256):
        """One deterministic pass over ALL valid training span starts.

        Use the first tile (the same window sampled by space/shift objectives),
        no dropout/gain jitter, and the same normalization as held-out scoring.
        Accumulate in float64 with sample weighting, not a mean of batch means.
        No model/training RNG is consumed. Buffer writes occur only at the end.
        """
        if self.lambda_space + self.lambda_shift == 0:
            return
        starts = torch.as_tensor(starts, dtype=torch.long, device=self.device_)
        if not len(starts):
            raise ValueError("No training spans available for oracle calibration.")
        shift_sum = (torch.zeros_like(self.order_head_.shift_profile,
                                      dtype=torch.float64)
                     if self.lambda_shift > 0 else None)
        group_sum = (torch.zeros_like(self.order_head_.group_profile,
                                      dtype=torch.float64)
                     if self.lambda_space > 0 else None)
        with torch.no_grad():
            for begin in range(0, len(starts), batch_size):
                block = starts[begin:begin + batch_size]
                gaps = torch.full((len(block), self.n_tiles - 1), self.tile_gap[0],
                                  dtype=torch.long, device=self.device_)
                tiles, _ = _gather_tiles(data, block, gaps, self.window_size)
                window = tiles[:, 0]
                if shift_sum is not None:
                    shift_sum += window.to(torch.float64).sum(dim=0)
                if group_sum is not None:
                    home = normalize_blocks(window, self.space_slots_, self.space_norm)
                    if self.space_time_shuffle:
                        # Exact expectation under the uniform time permutation.
                        home = home.mean(-1, keepdim=True).expand_as(home)
                    group_sum += self._block_patches(home).to(torch.float64).sum(0)
            if shift_sum is not None:
                self.order_head_.shift_profile.copy_(shift_sum / len(starts))
            if group_sum is not None:
                self.order_head_.group_profile.copy_(group_sum / len(starts))
            self.order_head_.oracle_profile_samples.fill_(len(starts))

    def calibrate_oracles(self, X_train, *, batch_size=256):
        """Calibrate templates on training data only, including untrained models."""
        self._check_fitted()
        batch_size = _integer("batch_size", batch_size, 1)
        data, starts, _ = self._spans(X_train, self.n_features_in_)
        self._fit_oracle_profiles(data, starts, batch_size)
        return self

    def fit(self, X, *args, **kwargs):
        super().fit(X, *args, **kwargs)
        if self.lambda_space + self.lambda_shift > 0 and not self.oracle_profile_samples_:
            self.calibrate_oracles(X)
        return self

    # -- labels ------------------------------------------------------------- #
    def _group_labels(self, batch):
        """(B,G,K). Independent permutations unless group_chimera is off."""
        rows = self.n_groups_ if self.group_chimera else 1
        keys = torch.rand(batch, rows, self.n_tiles, device=self.device_,
                          generator=self._generator)
        labels = keys.argsort(dim=2)
        return labels if self.group_chimera else labels.expand(
            -1, self.n_groups_, -1).contiguous()

    def _shift_labels(self, batch, channels, generator=None):
        """(B,N). One independent roll per neuron, uniform over the W bins.

        `generator` is threaded through rather than reaching for
        `self._generator` so that `evaluate_pretext` draws its rolls from its
        own seeded stream: a held-out number that moves when an unrelated
        training draw happens to consume a value is not a held-out number.
        """
        return torch.randint(0, self.window_size, (batch, channels),
                             device=self.device_,
                             generator=self._generator if generator is None
                             else generator)

    def _control(self, truth, starts, table):
        if self.label_control == "true":
            return truth
        if self.label_control == "fresh_random":
            # The shift labels are independent draws, not a permutation, so a
            # fresh argsort would hand the control a DIFFERENT distribution from
            # the real task (a derangement-ish one, never repeating a class
            # within a row). Match the marginal the truth was drawn from, or the
            # control stops being a control.
            if table is getattr(self, "_frozen_shift", None):
                return torch.randint(0, self.window_size, truth.shape,
                                     device=truth.device,
                                     generator=self._generator)
            keys = torch.rand(truth.shape, device=truth.device,
                              generator=self._generator)
            return keys.argsort(dim=-1)
        block = self.frozen_block if self.frozen_block > 0 else self.training_span
        row = (starts // max(block, 1)) % table.shape[0]
        return table[row]

    def _block_patches(self, window):
        """(B,N,W) -> (B,G,S,W): the block sitting in each row slot, intact.

        The 20260928 run measured its label-free baseline on this tensor
        AVERAGED OVER THE ROWS, which was a mistake worth naming. Averaging
        over S destroys exactly the cue the encoder is most likely to be using
        -- "in this block neuron 7 fires twice as often as neuron 12", a
        constant of the session that is visible to the network on every span
        and invisible to a row-averaged template. The baseline therefore sat at
        13-16% against a model at 85-96%, and the gap was an artefact of the
        baseline being blindfolded. Keep the rows; let the opponent see what
        the model sees.
        """
        rows = self.space_slots_.reshape(-1)
        return window[:, rows].reshape(len(window), self.n_groups_,
                                       self.space_block_, -1)

    def space_block_rows(self):
        """Rows per spatial block (S), or 0 when there is no spatial puzzle.

        Reported because every spatial number depends on it: with S=1 there is
        no within-block structure at all, so the rate oracle has a single
        scalar to work with and stops being informative.
        """
        return int(getattr(self, "space_block_", 0))

    def _shift_view(self, window, starts, generator):
        """Return a random input roll, TRUE shift labels, and uniqueness mask.

        `starts` is retained for call compatibility; label controls belong only
        in _step after this view is built, never in input generation/evaluation.
        """
        batch, channels = window.shape[:2]
        shift = self._shift_labels(batch, channels, generator)
        return roll_rows(window, shift), shift, unique_cyclic_rows(window)


    def _space_view(self, window, generator):
        """Time shuffle (optional) -> block permutation -> normalization.

        One function so that training and `evaluate_pretext` cannot drift
        apart, which is the classic way a pretext number stops meaning
        anything. Returns the shown window, the truth `sigma[b, s]` = the group
        placed in row slot s, and the pre-permutation window, which is what the
        label-free template has to be built from if it is to be an opponent
        rather than a formality.
        """
        if self.space_time_shuffle:
            window = shuffle_time(window, generator)
        sigma = torch.rand(len(window), self.n_groups_, device=window.device,
                           generator=generator).argsort(dim=1)
        shown = normalize_blocks(permute_blocks(window, self.space_slots_, sigma),
                                 self.space_slots_, self.space_norm)
        return shown, sigma, window

    def _order_forward(self, view, batch):
        """Encoder over the K tiles, then the grouped head. (z, logits, scores)."""
        z = self.encoder_(view.reshape(-1, self.n_features_in_, self.window_size))
        z = _GradScale.apply(z, self.order_grad_scale)
        z = z.reshape(batch, self.n_tiles, self.output_dimension)
        return (z,) + tuple(self.order_head_(z))

    # -- one step ----------------------------------------------------------- #
    def _step(self, data, starts):
        """Pure cross-entropy, four puzzle terms, zero reconstruction."""
        batch = len(starts)
        if self.lambda_space + self.lambda_shift > 0 and not self.oracle_profile_samples_:
            self._fit_oracle_profiles(data, self._valid_starts_)
        self._steps_ += 1
        if self.order_head_reset_every and self._steps_ % self.order_head_reset_every == 0:
            self._reset_order_head()
        order_scale = 1.0 + (self.order_final_scale - 1.0) * self._progress()
        n_tiles, n_groups = self.n_tiles, self.n_groups_

        candidates = self._draw(batch)
        jitter = self._jitter(candidates.shape[0], batch, self._valid_starts_)
        starts, gaps = select_span(data, starts, self._valid_starts_, candidates,
                                   jitter, self.window_size, self.span_selection)
        tiles, next_index = _gather_tiles(data, starts, gaps, self.window_size)
        parts, total = {}, torch.zeros((), device=self.device_)
        zero = torch.zeros((), device=self.device_)
        # These two exist only so the inherited fit/logging finds its keys. They
        # are structurally zero: see the constructor, which refuses to build a
        # model with either weight above 0.
        for key in ("forecast", "forecast_accuracy", "reconstruct",
                    "reconstruct_accuracy"):
            parts[key] = zero

        view = roll_tiles(tiles, self.order_time_roll, self._generator)
        view = self._order_view(_augment(
            view, min(0.95, self.neuron_dropout * self.order_augment_scale),
            self.gain_jitter * self.order_augment_scale, self._generator))
        truth = self._group_labels(batch)
        view = place_groups(view, truth, self.group_index_)
        labels = self._control(truth, starts, self._frozen_groups)
        # The permutation forward pass has to happen even on the space-only
        # arms -- the summary table, the shortcut audit and the memorization
        # gap are all read off it -- but its BACKWARD does not, because a term
        # multiplied by 0.0 contributes exactly no gradient. Running it under
        # no_grad leaves every reported number identical and drops most of the
        # step cost, which is what makes a space ladder affordable at 6000
        # epochs.
        order_live = (self.lambda_order + self.lambda_pair
                      + self.lambda_stitch + self.lambda_spread) > 0
        if order_live:
            z, position_logits, scores = self._order_forward(view, batch)
        else:
            with torch.no_grad():
                z, position_logits, scores = self._order_forward(view, batch)

        flat_labels = labels.reshape(-1, n_tiles)                    # (B*G, K)
        position_each = F.cross_entropy(
            position_logits.reshape(-1, n_tiles), labels.reshape(-1),
            reduction="none").reshape(batch, n_groups * n_tiles).mean(1)
        target, mask = _pair_targets(flat_labels)
        flat_scores = scores.reshape(-1, n_tiles)
        difference = flat_scores[:, :, None] - flat_scores[:, None, :]
        pair_all = F.binary_cross_entropy_with_logits(difference, target,
                                                      reduction="none")
        pair_each = ((pair_all * mask).sum((1, 2))
                     / mask.sum((1, 2)).clamp_min(1)).reshape(batch, n_groups).mean(1)

        if self.lambda_stitch > 0:
            stitch_logits = self.order_head_.stitch(self.order_head_.slices(z))
            stitch_target = successor_targets(labels)
            stitch_each = F.cross_entropy(
                stitch_logits.reshape(-1, n_tiles + 1), stitch_target.reshape(-1),
                reduction="none").reshape(batch, n_groups * n_tiles).mean(1)
            with torch.no_grad():
                parts["stitch_accuracy"] = (
                    stitch_logits.argmax(-1) == stitch_target).to(torch.float32).mean()
        else:
            stitch_each, parts["stitch_accuracy"] = torch.zeros(
                batch, device=self.device_), zero

        if self.lambda_space > 0:
            window = _augment(tiles[:, :1], self.neuron_dropout, self.gain_jitter,
                              self._generator)[:, 0]                 # (B,N,W)
            window, sigma, home = self._space_view(window, self._generator)
            sigma = self._control(sigma, starts, self._frozen_space)
            space_logits = self.order_head_.space(self.encoder_(window))
            space_each = F.cross_entropy(
                space_logits.reshape(-1, n_groups), sigma.reshape(-1),
                reduction="none").reshape(batch, n_groups).mean(1)
            with torch.no_grad():
                parts["space_accuracy"] = (
                    space_logits.argmax(-1) == sigma).to(torch.float32).mean()
        else:
            space_each, parts["space_accuracy"] = torch.zeros(
                batch, device=self.device_), zero

        if self.lambda_shift > 0:
            # Mechanism 6. One window, every row rolled by its own amount, and
            # the head names all N of them from one D-vector. The two masks
            # below are not bookkeeping: a row that is constant inside the
            # window has NO recoverable phase, so leaving it in would pay the
            # model for guessing the label marginal and would put a fake floor
            # under the reported accuracy.
            window = _augment(tiles[:, :1], self.neuron_dropout, self.gain_jitter,
                              self._generator)[:, 0]                 # (B,N,W)
            rolled, shift, alive = self._shift_view(window, starts, self._generator)
            shift = self._control(shift, starts, self._frozen_shift)
            shift_logits = self.order_head_.shift(self.encoder_(rolled))
            shift_all = F.cross_entropy(
                shift_logits.reshape(-1, self.window_size), shift.reshape(-1),
                reduction="none").reshape(batch, -1)                 # (B,N)
            weight = alive.to(shift_all.dtype)
            shift_each = masked_span_loss(shift_all, alive)
            with torch.no_grad():
                hit = (shift_logits.argmax(-1) == shift).to(torch.float32)
                parts["shift_accuracy"] = ((hit * weight).sum()
                                           / weight.sum().clamp_min(1.0))
                parts["shift_live_rows"] = weight.mean()
                nonconstant = (window.amax(-1) - window.amin(-1)) > 1e-6
                parts["shift_periodic_rows"] = (nonconstant & ~alive).float().mean()
        else:
            shift_each = torch.zeros(batch, device=self.device_)
            parts["shift_accuracy"] = parts["shift_live_rows"] = zero
            parts["shift_periodic_rows"] = zero

        combined = (self.lambda_order * position_each
                    + self.lambda_pair * pair_each
                    + self.lambda_stitch * stitch_each
                    + self.lambda_space * space_each
                    + self.lambda_shift * shift_each)
        if self.order_hard_fraction < 1.0:
            keep = min(batch, max(1, int(round(self.order_hard_fraction * batch))))
            combined = combined[combined.detach().topk(keep).indices]
        total = total + order_scale * combined.mean()

        if self.lambda_spread > 0:
            # A batch statistic, so it sits OUTSIDE the per-span hard-example
            # mining above -- there is no "hardest span" for a variance floor.
            # Hinge at 1 rather than maximizing variance: the job is to stop a
            # dimension from dying, not to inflate the ones that are alive.
            deviation = z.reshape(-1, self.output_dimension).std(dim=0)
            spread = F.relu(1.0 - deviation).pow(2).mean()
            total = total + self.lambda_spread * spread
            parts["spread"] = spread.detach()
        else:
            parts["spread"] = zero

        parts["position"] = position_each.mean().detach()
        parts["pair"] = pair_each.mean().detach()
        parts["stitch"] = stitch_each.mean().detach()
        parts["space"] = space_each.mean().detach()
        parts["shift"] = shift_each.mean().detach()
        parts["lag"] = zero
        parts["lag_accuracy"] = zero
        with torch.no_grad():
            predicted = _assign(position_logits.reshape(-1, n_tiles, n_tiles)) \
                if self.lambda_order > 0 else _ranks(flat_scores)
            exact, pair_accuracy, tie_rate = _order_metrics(predicted, flat_labels)
            parts["exact"] = exact.mean()
            parts["pair_accuracy"] = pair_accuracy.mean()
            parts["tie_rate"] = tie_rate.mean()
            parts["shortcut"] = shortcut_strength(tiles.sum(dim=(2, 3))).mean()
            self._train_exact_ = float(parts["exact"])
            self._train_pair_ = float(parts["pair_accuracy"])
            # None and not 0.0. A switched-off term that reports 0% invites the
            # reader to line it up against a 25% or 12.5% chance row and
            # conclude the head is broken, when in fact it was never asked.
            self._train_stitch_ = (float(parts["stitch_accuracy"])
                                   if self.lambda_stitch > 0 else None)
            self._train_space_ = (float(parts["space_accuracy"])
                                  if self.lambda_space > 0 else None)
            self._train_shift_ = (float(parts["shift_accuracy"])
                                  if self.lambda_shift > 0 and
                                  float(parts["shift_live_rows"]) > 0 else None)
        parts["total"] = total
        self._trace(parts)
        return parts

    def _trace(self, parts):
        """Live line for the terms the inherited epoch log knows nothing about.

        The base logger prints position CE, pair BCE, forecast CE and
        reconstruct CE. In this file the last two are structurally 0.0000 --
        useful as proof, useless as progress -- and the stitch, space and shift
        terms it does not know about are the ones worth watching. Printed on the
        same cadence as the base line, and only then, so the cost of the device
        sync is once per `log_every` epochs. Each accuracy is printed next to
        ITS OWN chance level, because they differ: 1/K for stitch, 1/G for
        space, 1/W for shift, and a reader comparing 12.5% against 25% without
        being told which is which will read a win as a failure.
        """
        live = self.lambda_stitch + self.lambda_space + self.lambda_shift
        if not self.verbose or live == 0:
            return
        per_epoch = max(1, math.ceil(getattr(self, "n_spans_", 1) / self.batch_size))
        if self._steps_ % max(1, per_epoch * self.log_every):
            return
        line = (f"      solo[last batch]: stitch CE={float(parts['stitch']):.4f} "
                f"(acc {100 * float(parts['stitch_accuracy']):.1f}%, "
                f"chance {100 / self.n_tiles:.1f}%) "
                f"space CE={float(parts['space']):.4f} "
                f"(acc {100 * float(parts['space_accuracy']):.1f}%, "
                f"chance {100 / self.n_groups_:.1f}%)")
        if self.lambda_shift > 0:
            line += (f" shift CE={float(parts['shift']):.4f} "
                     f"(acc {100 * float(parts['shift_accuracy']):.1f}%, "
                     f"chance {100 / self.window_size:.1f}%, "
                     f"unique {100 * float(parts['shift_live_rows']):.1f}%, "
                     f"periodic {100 * float(parts['shift_periodic_rows']):.1f}% of rows)")
        print(line + f" spread={float(parts['spread']):.4f}", flush=True)

    # -- evaluation --------------------------------------------------------- #
    def evaluate_pretext(self, X, *, max_spans=1024, batch_size=256, repeats=8,
                         random_state=200042, verbose=None, n_boot=1000,
                         baseline_groups=4):
        """Held-out puzzle accuracy, averaged over the G groups.

        Same key set as `OrderJigsaw.evaluate_pretext` so the runner's tables,
        audits and contrasts keep working, plus the solo terms. Two things are
        done per GROUP rather than per span, and they are the two that decide
        whether a number means anything:

        * accuracy is the mean over groups of that group's own permutation
          accuracy, so at G=1 it is the old number exactly;
        * the label-free baselines -- fixed-rule level sort, level oracle,
          tail-to-head continuity -- are computed on THAT GROUP'S ROWS ONLY
          against THAT GROUP'S permutation. A continuity baseline measured on
          all 38 neurons would be answering a different question from the one
          the group had to answer, and would flatter the model.

        `baseline_groups` caps how many groups the K! chain enumeration runs
        over, because that is the one expensive baseline; the accuracy itself
        always uses every group.
        """
        self._check_fitted()
        verbose = self.verbose if verbose is None else verbose
        repeats = _integer("repeats", repeats, minimum=1)
        max_spans = _integer("max_spans", max_spans, minimum=1)
        batch_size = _integer("batch_size", batch_size, minimum=1)
        baseline_groups = _integer("baseline_groups", baseline_groups, minimum=1)
        n_tiles, n_groups = self.n_tiles, self.n_groups_
        data, starts, _ = self._spans(X, self.n_features_in_)
        if len(starts) == 0:
            raise ValueError("No valid spans in X.")
        rng = np.random.default_rng(random_state)
        chosen = starts if len(starts) <= max_spans else np.sort(
            rng.choice(starts, max_spans, replace=False))
        chosen = torch.from_numpy(np.ascontiguousarray(chosen)).to(self.device_)
        local_starts = torch.from_numpy(np.ascontiguousarray(starts)).to(self.device_)
        generator = torch.Generator(device=self.device_).manual_seed(random_state)
        low, high = self.tile_gap
        pool = 1 if self.span_selection == "random" else self.selection_pool
        shown = min(baseline_groups, n_groups)
        keys = ("exact", "pair", "tie", "mean_exact", "mean_pair", "level_oracle",
                "continuity", "stitch", "space", "space_oracle",
                "space_oracle_rate", "space_oracle_level",
                "shift", "shift_oracle", "shift_live", "shift_periodic",
                "shift_constant", "shift_rows", "shift_ce")
        per_span = {k: torch.zeros(len(chosen), device=self.device_) for k in keys}
        position_ce = 0.0
        encoder_training = self.encoder_.training
        head_training = self.order_head_.training
        self.encoder_.eval()
        self.order_head_.eval()
        with torch.inference_mode():
            for _ in range(repeats):
                for begin in range(0, len(chosen), batch_size):
                    block = chosen[begin:begin + batch_size]
                    stop, size = begin + len(block), len(block)
                    candidates = torch.randint(
                        low, high + 1, (pool, size, n_tiles - 1),
                        device=self.device_, generator=generator)
                    jitter = self._jitter(pool, size, local_starts, generator)
                    block, gaps = select_span(data, block, local_starts, candidates,
                                              jitter, self.window_size,
                                              self.span_selection)
                    tiles, _ = _gather_tiles(data, block, gaps, self.window_size)
                    keys_g = torch.rand(size,
                                        n_groups if self.group_chimera else 1,
                                        n_tiles, device=self.device_,
                                        generator=generator)
                    labels = keys_g.argsort(dim=2)
                    if not self.group_chimera:
                        labels = labels.expand(-1, n_groups, -1).contiguous()
                    # `_order_view` rather than a hand-rolled normalization, so
                    # `count_match` and `raw` arms are evaluated the way they were
                    # trained. Normalizing before the chimera gather is identical
                    # to after it: the statistic is per (span, tile, neuron) and
                    # the gather only permutes the tile index within a neuron.
                    view = place_groups(self._order_view(tiles, generator),
                                        labels, self.group_index_)
                    shown_tiles = place_groups(tiles, labels, self.group_index_)
                    z = self.encoder_(view.reshape(-1, self.n_features_in_,
                                                   self.window_size))
                    z = z.reshape(size, n_tiles, self.output_dimension)
                    position_logits, scores = self.order_head_(z)
                    flat_labels = labels.reshape(-1, n_tiles)
                    predicted = _assign(position_logits.reshape(-1, n_tiles, n_tiles)) \
                        if self.lambda_order > 0 else _ranks(scores.reshape(-1, n_tiles))
                    exact, pair, tie = _order_metrics(predicted, flat_labels)
                    for name, value in (("exact", exact), ("pair", pair), ("tie", tie)):
                        per_span[name][begin:stop] += value.reshape(
                            size, n_groups).mean(1).to(torch.float32) / repeats

                    level = torch.zeros(size, device=self.device_)
                    level_exact = torch.zeros(size, device=self.device_)
                    continuity = torch.zeros(size, device=self.device_)
                    level_oracle = torch.zeros(size, device=self.device_)
                    for group in range(n_groups):
                        rows = torch.nonzero(self.group_index_ == group,
                                             as_tuple=False).flatten()
                        piece = shown_tiles[:, :, rows, :]
                        truth = labels[:, group, :]
                        g_exact, g_pair, _ = _order_metrics(
                            _ranks(piece.mean(dim=(2, 3))), truth)
                        level = level + g_pair / n_groups
                        level_exact = level_exact + g_exact / n_groups
                        level_oracle += torch.maximum(g_pair, 1 - g_pair) / n_groups
                        if group < shown:
                            continuity += continuity_strength(piece, truth) / shown
                    per_span["mean_pair"][begin:stop] += level / repeats
                    per_span["mean_exact"][begin:stop] += level_exact / repeats
                    per_span["level_oracle"][begin:stop] += level_oracle / repeats
                    per_span["continuity"][begin:stop] += continuity / repeats

                    if self.lambda_stitch > 0:
                        logits = self.order_head_.stitch(self.order_head_.slices(z))
                        hit = (logits.argmax(-1) == successor_targets(labels))
                        per_span["stitch"][begin:stop] += hit.reshape(
                            size, -1).to(torch.float32).mean(1) / repeats
                    if self.lambda_space > 0:
                        shown_window, sigma, _ = self._space_view(tiles[:, 0],
                                                                 generator)
                        guess = self.order_head_.space(
                            self.encoder_(shown_window)).argmax(-1)
                        per_span["space"][begin:stop] += (guess == sigma).to(
                            torch.float32).mean(1) / repeats
                        # THE label-free opponent, in three strengths, on the
                        # SAME view the encoder got. Name each block by matching
                        # it to that group's template:
                        #   space_oracle       the whole (S,W) patch. The
                        #                      strongest rule, and the one the
                        #                      model has to beat to be
                        #                      interesting.
                        #   ..._rate           each row's mean over time only.
                        #                      Pure static neuron identity:
                        #                      "this block is the one where the
                        #                      second row outfires the first".
                        #                      A session constant, so if THIS is
                        #                      already high the puzzle is not
                        #                      about the current span at all.
                        #   ..._level          the row-AVERAGED profile. This is
                        #                      the only one the 20260928 run
                        #                      reported, and it is the weakest of
                        #                      the three by construction.
                        # Training-only templates are registered head buffers;
                        # validation must never fill a missing profile.
                        profile = getattr(self, "group_profile_", None)
                        if profile is None:
                            for name in ("space_oracle", "space_oracle_rate",
                                         "space_oracle_level"):
                                per_span[name][begin:stop] = float("nan")
                        else:
                            patch = self._block_patches(shown_window)  # (B,G,S,W)
                            views = (                 # profile is (G,S,W)
                                ("space_oracle",
                                 patch.reshape(size, n_groups, -1),
                                 profile.reshape(n_groups, -1)),
                                ("space_oracle_rate",          # over time -> (.,S)
                                 patch.mean(dim=3), profile.mean(dim=2)),
                                ("space_oracle_level",         # over rows -> (.,W)
                                 patch.mean(dim=2), profile.mean(dim=1)))
                            for name, candidate, template in views:
                                hit = nearest_template(candidate, template) == sigma
                                per_span[name][begin:stop] += hit.to(
                                    torch.float32).mean(1) / repeats
                    if self.lambda_shift > 0:
                        # Held-out row phase. Everything here is masked by
                        # `alive`, INCLUDING the oracle, so the model and the
                        # rule are scored on exactly the same rows -- scoring
                        # the oracle on dead rows would let it bank the
                        # arbitrary argmax of a flat cosine and look stronger
                        # than it is.
                        rolled, shift, alive = self._shift_view(
                            tiles[:, 0], block, generator)
                        weight = alive.to(torch.float32)
                        shift_logits = self.order_head_.shift(self.encoder_(rolled))
                        guess = shift_logits.argmax(-1)
                        per_span["shift"][begin:stop] += (
                            ((guess == shift).to(torch.float32) * weight).sum(1)
                            / repeats)
                        per_span["shift_rows"][begin:stop] += weight.sum(1) / repeats
                        per_span["shift_live"][begin:stop] += weight.mean(1) / repeats
                        ce = F.cross_entropy(shift_logits.reshape(-1, self.window_size),
                                             shift.reshape(-1), reduction="none")
                        per_span["shift_ce"][begin:stop] += (
                            (ce.reshape_as(weight) * weight).sum(1) / repeats)
                        nonconstant = (tiles[:, 0].amax(-1) - tiles[:, 0].amin(-1)) > 1e-6
                        per_span["shift_periodic"][begin:stop] += (
                            (nonconstant & ~alive).float().mean(1) / repeats)
                        per_span["shift_constant"][begin:stop] += (
                            (~nonconstant).float().mean(1) / repeats)
                        # The label-free opponent: match each rolled row against
                        # every cyclic shift of that neuron's own template and
                        # take the best cosine. If this already names the roll,
                        # the puzzle is a template-matching exercise and the
                        # encoder is not needed for it.
                        profile = getattr(self, "shift_profile_", None)
                        if profile is None:
                            per_span["shift_oracle"][begin:stop] = float("nan")
                        else:
                            oracle = roll_scores(rolled, profile).argmax(-1)
                            per_span["shift_oracle"][begin:stop] += (
                                ((oracle == shift).to(torch.float32) * weight).sum(1)
                                / repeats)
                    position_ce += float(F.cross_entropy(
                        position_logits.reshape(-1, n_tiles),
                        labels.reshape(-1))) * size
        self.encoder_.train(encoder_training)
        self.order_head_.train(head_training)
        mean = {k: float(v.mean()) for k, v in per_span.items()}
        # Pool numerators and denominators across rows/spans/repeats. Empty
        # spans contribute no trials; an all-empty evaluation is undefined.
        for key in ("shift", "shift_oracle", "shift_ce"):
            mean[key] = (mean[key] / mean["shift_rows"]
                         if mean["shift_rows"] > 0 else float("nan"))
        temporal_live = self.lambda_order + self.lambda_pair + self.lambda_stitch > 0
        entropy = target_entropy_nats(
            n_groups, n_tiles, self.n_features_in_, self.window_size,
            temporal=temporal_live, shared=not self.group_chimera,
            shift=self.lambda_shift > 0, spatial=self.lambda_space > 0)

        def percent(key, enabled=True):
            """100x, or None when the number would be a lie.

            Covers both "the term is switched off" and "the statistic could not
            be computed" (K > 6 for the chain baselines, a reloaded model with no
            training-time group profile). Reporting 0.0 for those would put a
            number in the table that a reader would compare against chance.
            """
            value = mean[key]
            if not enabled or not math.isfinite(value) or value < 0:
                return None
            return 100 * value

        boot = 0 if getattr(self, "_fitting_", False) else n_boot
        low_ci, high_ci, spread, n_effective = block_bootstrap(
            per_span["pair"].cpu().numpy(), self.training_span, n_boot=boot,
            random_state=random_state)
        result = dict(
            exact_accuracy_percent=100 * mean["exact"],
            pair_accuracy_percent=100 * mean["pair"],
            pair_ci95_percent=[100 * low_ci, 100 * high_ci],
            pair_block_resolution_percent=100 * spread,
            n_effective=n_effective,
            tie_rate_percent=100 * mean["tie"],
            argmax_pair_accuracy_percent=100 * mean["pair"],
            argmax_tie_rate_percent=100 * mean["tie"],
            rank_exact_accuracy_percent=100 * mean["exact"],
            rank_pair_accuracy_percent=100 * mean["pair"],
            lag_accuracy_percent=0.0,
            chance_lag_percent=100.0 / self.lag_classes,
            baseline_mean_sort_exact_percent=100 * mean["mean_exact"],
            baseline_mean_sort_pair_percent=100 * mean["mean_pair"],
            baseline_norm_sort_exact_percent=100 * mean["mean_exact"],
            baseline_norm_sort_pair_percent=100 * mean["mean_pair"],
            baseline_level_oracle_pair_percent=100 * mean["level_oracle"],
            baseline_continuity_pair_percent=percent("continuity"),
            continuity_gap_bins=int(self.tile_gap[0]),
            baseline_groups=int(shown),
            level_baseline_groups=int(n_groups),
            train_pair_accuracy_percent=(None if not hasattr(self, "_train_pair_")
                                         else 100 * self._train_pair_),
            train_exact_accuracy_percent=(None if not hasattr(self, "_train_exact_")
                                          else 100 * self._train_exact_),
            memorization_gap_percent=(None if not hasattr(self, "_train_pair_")
                                      else 100 * (self._train_pair_ - mean["pair"])),
            # -- the solo terms ------------------------------------------- #
            n_groups=int(n_groups),
            group_chimera=bool(self.group_chimera),
            group_subspace=bool(self.group_subspace),
            group_sizes=list(getattr(self, "group_sizes_", [])),
            target_entropy_nats_per_span=entropy,
            # Compatibility alias only: legacy key used nats despite its name.
            label_bits_per_span=entropy,
            target_entropy_scope="uniform transformation truth; before masking",
            temporal_objective_active=bool(temporal_live),
            shift_head_kind="linear" if self.lambda_shift > 0 else None,
            oracle_profile_samples=self.oracle_profile_samples_,
            oracle_profile_source="all_valid_training_span_starts",
            evaluation_labels="true",
            stitch_accuracy_percent=percent("stitch", self.lambda_stitch > 0),
            # The best LABEL-FREE constant rule ("always answer 'I am last'", or
            # always answer some fixed slot) scores 1/K, not 1/(K+1): the self
            # logit is masked, so there are only K live options. Quoting the
            # smaller number would make a useless head look educated.
            chance_stitch_percent=100.0 / n_tiles,
            train_stitch_accuracy_percent=(
                None if getattr(self, "_train_stitch_", None) is None
                else 100 * self._train_stitch_),
            space_accuracy_percent=percent("space", self.lambda_space > 0),
            space_oracle_percent=percent("space_oracle", self.lambda_space > 0),
            space_oracle_rate_percent=percent("space_oracle_rate",
                                              self.lambda_space > 0
                                              and self.space_block_rows() > 1),
            space_oracle_level_percent=percent("space_oracle_level",
                                               self.lambda_space > 0),
            # None at G=1: "chance is 100%" is arithmetically true and reads as
            # a result. There is no spatial permutation of one block.
            chance_space_percent=(100.0 / n_groups if n_groups > 1 else None),
            space_norm=self.space_norm,
            space_time_shuffle=bool(self.space_time_shuffle),
            space_block_rows=self.space_block_rows(),
            train_space_accuracy_percent=(
                None if getattr(self, "_train_space_", None) is None
                else 100 * self._train_space_),
            shift_accuracy_percent=percent("shift", self.lambda_shift > 0),
            shift_oracle_percent=percent("shift_oracle", self.lambda_shift > 0),
            # None at W=1, for the same reason as the space row: there is no
            # phase to recover in a one-bin window, so "chance 100%" would be a
            # true sentence that reads as a finding. The constructor already
            # refuses that configuration; this keeps the table honest anyway.
            chance_shift_percent=(100.0 / self.window_size
                                  if self.window_size > 1 else None),
            # The share of rows that actually carried a phase. A run where this
            # is small is not measuring what the column name says: the accuracy
            # is then an average over a handful of neurons, and its resolution
            # is correspondingly worse than the span count suggests.
            shift_live_rows_percent=percent("shift_live", self.lambda_shift > 0),
            shift_periodic_rows_percent=percent("shift_periodic", self.lambda_shift > 0),
            shift_constant_rows_percent=percent("shift_constant", self.lambda_shift > 0),
            shift_evaluated_rows=(int(round(float(per_span["shift_rows"].sum()) * repeats))
                                  if self.lambda_shift > 0 else 0),
            shift_cross_entropy=(mean["shift_ce"] if self.lambda_shift > 0
                                  and math.isfinite(mean["shift_ce"]) else None),
            shift_uniform_cross_entropy=(math.log(self.window_size)
                                          if self.lambda_shift > 0 else None),
            shift_classes=int(self.window_size),
            train_shift_accuracy_percent=(
                None if getattr(self, "_train_shift_", None) is None
                else 100 * self._train_shift_),
            position_cross_entropy=position_ce / max(repeats * len(chosen), 1),
            uniform_cross_entropy=math.log(n_tiles),
            decode="assignment" if self.lambda_order > 0 else "score_rank",
            order_view=self.order_view, span_selection=self.span_selection,
            label_control=self.label_control,
            chance_exact_percent=100.0 / math.factorial(n_tiles),
            chance_pair_percent=50.0,
            n_spans=int(len(chosen)), n_draws=int(repeats * len(chosen)),
            repeats=int(repeats))
        if verbose:
            def show(key, absent="off"):
                value = result[key]
                return absent if value is None else f"{value:.2f}%"
            print(f"  pretext[G={n_groups}]: "
                  f"exact={result['exact_accuracy_percent']:.2f}% "
                  f"(chance {result['chance_exact_percent']:.2f}%) "
                  f"pair={result['pair_accuracy_percent']:.2f}% "
                  f"[{result['pair_ci95_percent'][0]:.2f}, "
                  f"{result['pair_ci95_percent'][1]:.2f}] n_eff={n_effective} "
                  f"| level-oracle={result['baseline_level_oracle_pair_percent']:.2f}% "
                  f"continuity={show('baseline_continuity_pair_percent', 'n/a')} "
                  f"| stitch={show('stitch_accuracy_percent')} "
                  f"(chance {result['chance_stitch_percent']:.2f}%)", flush=True)
            # The spatial puzzle gets its own line because it needs three
            # baselines next to it to mean anything, and a one-line version
            # would push them off the right-hand side of the log.
            if result["space_accuracy_percent"] is not None:
                print(f"      space[{self.space_norm}"
                      f"{', time-shuffled' if self.space_time_shuffle else ''}, "
                      f"S={result['space_block_rows']}]: "
                      f"{show('space_accuracy_percent')} "
                      f"(chance {show('chance_space_percent', 'n/a')}) "
                      f"vs template oracles -- patch "
                      f"{show('space_oracle_percent', 'n/a')}, rate "
                      f"{show('space_oracle_rate_percent', 'n/a')}, level "
                      f"{show('space_oracle_level_percent', 'n/a')}", flush=True)
            if result["shift_accuracy_percent"] is not None:
                print(f"      shift[W={self.window_size}]: "
                      f"{show('shift_accuracy_percent')} "
                      f"(chance {show('chance_shift_percent', 'n/a')}, train "
                      f"{show('train_shift_accuracy_percent', 'n/a')}) "
                      f"vs roll oracle {show('shift_oracle_percent', 'n/a')} "
                      f"| live rows {show('shift_live_rows_percent', 'n/a')} "
                      f"| {result['target_entropy_nats_per_span']:.0f} nominal label nats/span",
                      flush=True)
        return result


# --------------------------------------------------------------------------- #
# 4. arms
# --------------------------------------------------------------------------- #
#  Read this table as four blocks:
#    the LADDER   -- rank 1, 4, 8, 16, and now N. One variable, and the
#                    prediction the whole file rests on: R2 and participation
#                    ratio rise with the number of independent questions.
#    the CONTROLS -- shared permutations, no subspaces, frozen labels, frozen
#                    encoder. Each removes exactly one thing the ladder claims
#                    is load-bearing.
#    the TERMS    -- stitch, space and shift on their own and left out, so the
#                    headline arm can be decomposed rather than admired.
#    the HIGH-RANK PAIR -- mechanisms 5 and 6, added 20260929 after the
#                    20260928 run killed the spatial story. These are the arms
#                    this round exists to measure.
#
#  Every arm here is a PERMUTATION/POSITION puzzle trained with cross-entropy.
#  There is no reconstruction arm and there will not be one: that family was
#  settled in earlier rounds, its number is quoted in the module docstring as a
#  fixed target, and re-running it spends seeds without answering anything.
_SOLO = dict(_model="solo")
_LADDER = dict(_SOLO, lambda_stitch=1.0, lambda_space=0.5, space_norm="block_mean")


def _space(groups, norm, **extra):
    """A SPACE-ONLY arm: permutation terms off, neuron-axis puzzle on.

    With `lambda_order`, `lambda_pair` and `lambda_stitch` all zero the model
    still runs the K-tile forward pass (the tables need it) but skips its
    backward, so these arms are the cheap ones despite looking like the others.
    """
    return dict(_SOLO, n_groups=groups, lambda_order=0.0, lambda_pair=0.0,
                lambda_space=1.0, space_norm=norm, **extra)


SOLO_ARMS = {
    # -- the ladder ------------------------------------------------------- #
    # G=1 is the classic jigsaw with the anchor deleted: same head, same
    # parameter count, same loss. Historically this is `order_only`, which
    # scored R2 0.005 at participation ratio 3.4. If the rank story is right
    # this arm reproduces that collapse and the rest of the ladder escapes it.
    "solo_rank1": dict(_SOLO, n_groups=1),
    "solo_rank4": dict(_SOLO, n_groups=4),
    "solo_rank8": dict(_SOLO, n_groups=8),
    "solo_rank16": dict(_SOLO, n_groups=16),
    # -- the controls for the ladder --------------------------------------- #
    # THE control. Eight groups, eight subspaces, eight readouts, identical
    # parameter count and identical number of CE terms -- and one single
    # permutation copied to all of them. The ONLY thing removed is the
    # independence of the labels, which is the only thing the rank argument
    # claims to need. If `solo_rank8 - solo_shared8` is zero, the idea is wrong.
    "solo_shared8": dict(_SOLO, n_groups=8, group_chimera=False),
    # Same grouped task with full-embedding readouts instead of disjoint slices.
    "solo_wide8": dict(_SOLO, n_groups=8, group_subspace=False),
    "solo_frozen8": dict(_SOLO, n_groups=8, label_control="frozen_random"),
    "solo_random": dict(_SOLO, n_groups=8, max_epochs=0),
    # -- the full model and its term ablations ------------------------------ #
    "solo_full": dict(_LADDER, n_groups=8),
    "solo_full_frozen": dict(_LADDER, n_groups=8, label_control="frozen_random"),
    "solo_full_shared": dict(_LADDER, n_groups=8, group_chimera=False),
    "solo_no_stitch": dict(_LADDER, n_groups=8, lambda_stitch=0.0),
    "solo_no_space": dict(_LADDER, n_groups=8, lambda_space=0.0),
    "solo_no_subspace": dict(_LADDER, n_groups=8, group_subspace=False),
    # Each term completely alone, with the permutation CE switched off, so the
    # decomposition adds up instead of being asserted.
    "solo_stitch_only": dict(_SOLO, n_groups=8, lambda_order=0.0, lambda_pair=0.0,
                             lambda_stitch=1.0),
    # -- the spatial puzzle, which is the part that actually worked ---------- #
    # 20260928, three seeds. Read against the FROZEN random encoder at R2
    # 0.3199, not against zero:
    #     solo_rank8    0.3186   the grouped TEMPORAL puzzle, alone -> a dead
    #                            heat with an untrained encoder.
    #     solo_no_space 0.2538   temporal puzzle + stitch, no spatial term ->
    #                            BELOW the frozen encoder. Actively harmful.
    #     solo_no_stitch 0.4158  temporal puzzle + spatial term.
    #     solo_full      0.4345  everything.
    # The entire result is the neuron-axis term, so from here it is the
    # experiment and the temporal ladder is the control. The open question is
    # NOT whether it helps -- it plainly does -- but whether it is solvable from
    # static neuron identity, which is a session constant the encoder can
    # memorize once, or from this span's dynamics. That is what space_norm and
    # space_time_shuffle are for, and what the three template oracles measure.
    "solo_space_only":   _space(8, "block_mean"),
    # The rate cue left fully in place. If this MATCHES solo_space_only then
    # block_mean was removing nothing the model was using.
    "solo_space_raw":    _space(8, "none"),
    # Each neuron's own mean over the window removed, then its gain too. Under
    # row_zscore no neuron's firing rate can name its block: what survives is
    # the temporal shape of each row, which is a fact about this span. If the
    # spatial puzzle still transfers here, the claim is about dynamics.
    "solo_space_rowm":   _space(8, "row_mean"),
    "solo_space_rowz":   _space(8, "row_zscore"),
    # Same static statistics, temporal order destroyed. Scores identically to
    # solo_space_only if and only if the task never needed time.
    "solo_space_tshuf":  _space(8, "block_mean", space_time_shuffle=True),
    # The label control for the spatial term on its own, matching what
    # solo_full_frozen did for the whole model (train 88-93%, held out 12.3-12.7%
    # -- exactly chance -- and R2 0.1592, well BELOW the frozen encoder).
    "solo_space_frozen": _space(8, "block_mean", label_control="frozen_random"),
    # The ladder on the axis that works. Not a one-variable contrast and the
    # comment should say so: raising G raises the class count AND shrinks the
    # block to 38//G rows, so a harder question is being asked of less evidence.
    "solo_space_g4":     _space(4, "block_mean"),
    "solo_space_g16":    _space(16, "block_mean"),
    # -- the regularizer, kept separate on purpose -------------------------- #
    # Not a puzzle. Here so that "the embedding did not collapse" can be
    # attributed to the task rather than to a variance floor -- if
    # solo_rank8 and solo_spread land in the same place, the floor is doing
    # nothing and the puzzle earned it.
    "solo_spread": dict(_SOLO, n_groups=8, lambda_spread=1.0),
    "solo_rank1_spread": dict(_SOLO, n_groups=1, lambda_spread=1.0),
    # -- hardened geometry --------------------------------------------------- #
    # Gap floor 3 so tiles never touch, short span so the drift is real. This is
    # the geometry where `order_gapped` was the only arm in the 20260927 run to
    # genuinely learn the pretext (54.67% against a 51.21% continuity baseline,
    # z=20) -- and had the worst R2 of any trained arm, 0.3840. Worth repeating
    # with a rank-8 puzzle: the question is whether learning a HIGH-RANK version
    # of the same task transfers where the rank-1 version did not.
    "solo_gapped8": dict(_LADDER, n_groups=8, window_size=6, tile_gap=(3, 5)),
    "solo_gapped1": dict(_LADDER, n_groups=1, window_size=6, tile_gap=(3, 5),
                         lambda_space=0.0),
    # Hard-example mining plus periodic head resets, the two counters to the
    # gradient-death pattern measured on the anchored runs (order CE 0.0153 at
    # epoch 1000 falling to 0.0006 by 10000, i.e. the term switching itself off).
    "solo_hard8": dict(_LADDER, n_groups=8, order_hard_fraction=0.25,
                       order_head_reset_every=400),
    # -- mechanism 5: ONE GROUP PER NEURON ---------------------------------- #
    # The rank ladder taken to N. Every neuron's row is chopped into the same K
    # chunks, each row gets its OWN permutation, and the head has to place all
    # N of them from one embedding: 38 x log(4!) = 121 nats per span against the
    # 3.18 that the classic jigsaw asks. `group_subspace` is off because D/N is
    # not an integer and a one-row group has no business owning a private slice
    # of the embedding -- which is the point, since the rank then has to be
    # demanded by the QUESTION rather than donated by the wiring.
    "solo_neuron": dict(_SOLO, n_groups=0, group_subspace=False),
    # THE falsifier for mechanism 5, and the only contrast in this block worth
    # reading first. Same N groups, same N readouts, same N cross-entropy terms,
    # same parameter count -- and one permutation copied to every row instead of
    # N independent ones. Everything about the arm is held fixed except the
    # thing the rank argument says is load-bearing. Zero here ends the idea.
    "solo_neuron_shared": dict(_SOLO, n_groups=0, group_subspace=False,
                               group_chimera=False),
    # The floor. Mechanisms 1 and 4 both died against an untrained encoder, so
    # no arm in this file gets called "learned" until it clears its own frozen
    # twin -- same geometry, same head, zero gradient steps.
    "solo_random_neuron": dict(_SOLO, n_groups=0, group_subspace=False,
                               max_epochs=0),
    # Labels that mean nothing, holding the loss, the heads and the gradient
    # path fixed. Separates "the per-neuron puzzle taught it something" from
    # "training on anything at all moves R2".
    "solo_neuron_frozen": dict(_SOLO, n_groups=0, group_subspace=False,
                               label_control="frozen_random"),
    # -- mechanism 6: ROW PHASE --------------------------------------------- #
    # No tiles, no chimera gather, one encoder pass: roll each row around the
    # window by its own amount and name all N rolls. 38 x log(10) = 87 nats per
    # span. A roll preserves each row's multiset exactly, so a sorted row says
    # nothing and only POSITION carries the answer -- the static-identity
    # shortcut that ate the neuron-axis puzzle is unavailable here by
    # construction, not by normalization.
    "solo_shift": dict(_SOLO, lambda_order=0.0, lambda_pair=0.0, lambda_shift=1.0),
    "solo_shift_frozen": dict(_SOLO, lambda_order=0.0, lambda_pair=0.0,
                              lambda_shift=1.0, label_control="frozen_random"),
    "solo_shift_untrained": dict(_SOLO, lambda_order=0.0, lambda_pair=0.0,
                                 lambda_shift=1.0, max_epochs=0),
    # -- 5 and 6 together, which is the arm aimed at the 0.54 bar ------------ #
    # About 208 nats of nominal joint label entropy BEFORE masking. More target
    # entropy need not be identifiable or preserve downstream information.
    "solo_neuron_shift": dict(_SOLO, n_groups=0, group_subspace=False,
                              lambda_shift=1.0),
    # The same arm with an explicit variance floor bolted on. Not a puzzle and
    # not a claim: it is the measurement that says whether whatever gap remains
    # at the end is still collapse (floor helps) or is missing information
    # (floor does nothing). Either answer decides what to build next.
    "solo_neuron_shift_spread": dict(_SOLO, n_groups=0, group_subspace=False,
                                     lambda_shift=1.0, lambda_spread=1.0),
}

# `group_chimera`, the lambdas (`lambda_space`, `lambda_stitch`, `lambda_shift`,
# ...), `space_norm`, `space_time_shuffle` and `lambda_spread` are deliberately
# absent: those are the variables under test, exactly as the order weights are
# absent from ANCHOR_MATCH_KEYS. A `space_norm` entry here (v1 had one) made the
# runner shout "NOT ONE VARIABLE" at the rowz and raw contrasts, which were the
# whole point of that round.
# `n_groups` IS here, because changing it changes the subspace width and the
# block height and therefore the readout capacity -- a rank-1-versus-rank-8
# comparison is not a one-variable contrast and the runner should say so. Note
# that it is the RAW value that is compared, so the `n_groups=0` sentinel
# matches only other per-neuron arms: `solo_neuron` against `solo_neuron_shared`
# is clean (0 vs 0, chimera differs), while `solo_neuron_shift` against
# `solo_full` is correctly flagged, because it is a dose curve and not a
# contrast.
SOLO_MATCH_KEYS = ANCHOR_MATCH_KEYS + ("n_groups", "group_subspace", "group_seed",
                                       "stitch_rank")


SOLO_CONTRASTS = (
    ("solo_rank8", "solo_shared8", "RANK effect (independent puzzles)",
     "identical architecture, identical parameter count, identical number of CE "
     "terms. The only difference is whether the G permutations are independent. "
     "This is the contrast the whole file stands or falls on."),
    ("solo_full", "solo_full_shared", "RANK effect (full model)",
     "the same test with the stitch and space terms switched on."),
    ("solo_rank8", "solo_rank1", "GROUPS effect",
     "rank 8 against the classic jigsaw with the anchor removed. Not a "
     "one-variable contrast -- the subspace width changes too -- so read it "
     "next to the shared-permutation control, not instead of it."),
    ("solo_full", "solo_full_frozen", "LABEL effect",
     "same loss, same heads, same gradient path, labels that mean nothing about "
     "time or identity. On the anchored runs this was ~0 and the frozen arm "
     "often WON, which is what sent us here."),
    ("solo_full", "solo_no_stitch", "STITCH effect",
     "what the bilinear successor CE adds, assessed with held-out transfer."),
    ("solo_full", "solo_no_space", "SPACE effect",
     "what the neuron-axis puzzle adds."),
    ("solo_full", "solo_no_subspace", "SUBSPACE effect",
     "disjoint readout slices against a shared embedding; rank is measured empirically."),
    ("solo_rank8", "solo_spread", "what a variance floor adds on top",
     "if this is large, the puzzle is not preventing collapse on its own and the "
     "honest description of the result includes a regularizer."),
    # -- the spatial round -------------------------------------------------- #
    ("solo_space_only", "solo_random", "SPACE ALONE vs FROZEN ENCODER",
     "the floor that matters. On 20260928 the temporal puzzle failed exactly "
     "this test -- solo_rank8 0.3186 against solo_random 0.3199 -- so an arm "
     "that does not clear it has not earned the word 'learned'."),
    ("solo_space_only", "solo_space_frozen", "SPACE LABEL effect",
     "same loss, same head, same gradient path, block identities that mean "
     "nothing. This is what makes the spatial number evidence rather than a "
     "side effect of training on something."),
    ("solo_space_only", "solo_space_rowz", "what the STATIC RATE cue was worth",
     "row_zscore removes each row's mean/gain. A small effect weakens that "
     "specific cue explanation; it does not establish temporal dynamics."),
    ("solo_space_only", "solo_space_tshuf", "what TEMPORAL ORDER was worth to it",
     "the time bins are permuted, so every static statistic survives and only "
     "the order of the window is destroyed. A null effect does not rule out "
     "other static population cues."),
    ("solo_space_g16", "solo_space_only", "SPACE GROUPS effect",
     "G=16 asks a 16-way question of 2-row blocks where G=8 asks an 8-way "
     "question of 4-row blocks. Two variables, so read it as a dose curve and "
     "not as a contrast."),
    ("solo_rank8", "solo_random", "TEMPORAL PUZZLE vs FROZEN ENCODER",
     "the 20260928 null, kept in the table so it is re-measured rather than "
     "remembered: +0.0 means sorting time-tiles is worth nothing on its own."),
    ("solo_full", "solo_random", "vs FROZEN ENCODER",
     "what training is worth at all. Must be positive or nothing else counts."),
    # -- the high-rank round, 20260929 -------------------------------------- #
    # Read these five FIRST. The three above them are last round's questions,
    # kept only so the 20260928 numbers line up on the same seeds.
    ("solo_neuron", "solo_neuron_shared", "PER-NEURON RANK effect",
     "THE contrast of this round, and the one the file stands or falls on. N "
     "independent row permutations against ONE permutation copied to all N "
     "rows: same head, same parameter count, same number of CE terms, same "
     "everything else. The rank argument says the first asks 38 x log(4!) = 121 "
     "nats per span and the second asks 3.18. If that difference buys nothing, "
     "mechanism 5 is wrong and no amount of tuning will save it."),
    ("solo_neuron", "solo_random_neuron", "PER-NEURON vs FROZEN ENCODER",
     "the floor that killed the temporal ladder (+0.0 at 3 seeds) and the "
     "spatial puzzle (-0.0081 at 2 seeds). Until an arm clears its own "
     "untrained twin, nothing else in its row is worth reading."),
    ("solo_neuron", "solo_neuron_frozen", "PER-NEURON LABEL effect",
     "meaningless labels through the identical gradient path. On the spatial "
     "arms this came back +0.1653, which sounds like a win and is not: it only "
     "said that training on noise is actively destructive."),
    ("solo_shift", "solo_shift_untrained", "ROW PHASE vs FROZEN ENCODER",
     "the same floor test for mechanism 6. Rolling a row preserves its multiset "
     "exactly. Mean-rate shortcuts are excluded, but distributional asymmetry "
     "and boundary artifacts remain possible; inspect the held-out oracle."),
    ("solo_shift", "solo_shift_frozen", "ROW PHASE LABEL effect",
     "same input rolls and eligible rows; only training targets are frozen random."),
    ("solo_neuron_shift", "solo_neuron", "what ROW PHASE adds on top",
     "combined tasks against per-neuron temporal Jigsaw. The nominal label "
     "entropy rises; a positive transfer effect does not prove a rank mechanism."),
    ("solo_neuron_shift", "solo_full", "HIGH-RANK vs last round's best jigsaw",
     "solo_full scored 0.4229 mean (spread 0.1733) on seeds 8 and 10. The bar "
     "for this round is 0.54, so this contrast needs about +0.12 -- anything "
     "less and the rank story is real but too small to matter."),
    ("solo_neuron_shift_spread", "solo_neuron_shift",
     "is the REMAINING gap still collapse?",
     "a variance floor is not a puzzle and earns no credit in the paper, but it "
     "is the cheapest way to find out what the gap is made of. Large and "
     "positive indicates benefit from this regularizer. A null effect alone "
     "does not establish full rank; inspect participation ratio and optimization."),
)

# Eleven default arms, including three max_epochs=0 encoder controls.
# Label entropy is nominal and is not a rank guarantee or recoverable evidence.
# The two historical arms permit same-seed comparisons with the previous round.
#
# There is deliberately NO reconstruction arm. Its number (0.5450 MLP, p.ratio
# 18.76, seeds 8 and 10) is quoted in the module docstring and is the bar these
# arms are judged against; spending a slot to re-derive it would answer nothing.
DEFAULT_SOLO_ARMS = ("solo_neuron_shift", "solo_neuron", "solo_neuron_shared",
                     "solo_random_neuron", "solo_neuron_frozen", "solo_shift",
                     "solo_shift_frozen", "solo_shift_untrained",
                     "solo_neuron_shift_spread", "solo_full", "solo_random")


def register(models, arms, contrasts=None):
    """Wire this module into run_jigsaw.py without editing its tables by hand."""
    models["solo"] = SoloJigsaw
    for name, settings in SOLO_ARMS.items():
        arms[name] = dict(settings)
    if contrasts is not None:
        for row in SOLO_CONTRASTS:
            if row not in contrasts:
                contrasts.append(row)
    missing = [name for name in DEFAULT_SOLO_ARMS if name not in arms]
    if missing:
        # Loud on purpose. On 20260928 `reconstruct_only` sat in the default
        # list, resolved to nothing, and the run finished without the one
        # comparison it existed to make -- with no warning anywhere in the log.
        print(f"  WARNING: default solo arms not registered: {missing}. They "
              f"will be skipped and any contrast that names them will be "
              f"missing from the output.", flush=True)
    return models, arms


# --------------------------------------------------------------------------- #
# 5. self-test
# --------------------------------------------------------------------------- #
def _check(condition, message):
    print(f"  {'ok  ' if condition else 'FAIL'}  {message}")
    if not condition:
        raise AssertionError(message)


def _counts(n=2600, channels=12, seed=0):
    """Drifting Poisson counts: a slow level ramp plus a fast oscillation.

    The ramp is what the level shortcut reads; the oscillation is what an
    honest solution has to read. Both are present so the baselines in
    `evaluate_pretext` have something to detect.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(n)[:, None]
    phase = rng.uniform(0, 2 * np.pi, size=(1, channels))
    rate = (0.6 + 0.8 * time / n
            + 0.5 * np.sin(2 * np.pi * time / 37 + phase))
    return rng.poisson(np.clip(rate, 0.05, None)).astype(np.float32)


def _smoke_test():
    print("jigsaw_solo self-test")
    torch.manual_seed(0)
    counts = _counts()
    channels = counts.shape[1]

    # Deterministic regressions for the corrected phase task.
    rows = torch.tensor([[[0., 0., 0., 0., 0., 0.],
                          [1., 0., 1., 0., 1., 0.],
                          [1., 2., 3., 1., 2., 3.],
                          [1., 0., 0., 0., 0., 0.]]])
    _check(unique_cyclic_rows(rows).tolist() == [[False, False, False, True]],
           "constant and periodic rows are excluded; a unique phase survives")
    _check(torch.equal(unique_cyclic_rows(roll_rows(rows, torch.tensor([[0, 1, 2, 3]]))),
                       unique_cyclic_rows(rows)), "phase mask is rotation-invariant")
    head = _ShiftHead(7, 4, 6)
    x, y = torch.randn(3, 7), torch.randn(3, 7)
    _check(torch.allclose(head(x + y), head(x) + head(y) - head(torch.zeros_like(x)),
                          atol=1e-6), "shift logits are affine in the embedding")
    losses = torch.tensor([[2., 7.], [9., 9.]], requires_grad=True)
    eligible = torch.tensor([[True, False], [False, False]])
    masked = masked_span_loss(losses, eligible).mean()
    masked.backward()
    _check(float(masked.detach()) == 2.0 and losses.grad.tolist() == [[1., 0.], [0., 0.]],
           "empty spans do not dilute phase loss or receive phase gradients")
    empty = torch.randn(2, 3, requires_grad=True)
    masked_span_loss(empty, torch.zeros_like(empty, dtype=torch.bool)).mean().backward()
    _check(torch.equal(empty.grad, torch.zeros_like(empty)),
           "an all-empty phase batch is differentiable with zero gradients")
    shared_entropy = target_entropy_nats(38, 4, 38, 10, temporal=True,
                                        shared=True, shift=False, spatial=False)
    _check(abs(shared_entropy - math.lgamma(5)) < 1e-8,
           "one shared permutation has log(K!) entropy, not N times that")

    # 1. the guarantee
    for weight in ("lambda_reconstruct", "lambda_forecast"):
        try:
            SoloJigsaw(**{weight: 1.0})
            raised = False
        except ValueError:
            raised = True
        _check(raised, f"{weight}>0 is refused by the constructor")

    # 2. divisibility and the other constructor guards
    for kwargs in (dict(n_groups=7, output_dimension=64),
                   dict(n_groups=1, lambda_space=1.0),
                   dict(n_groups=4, lambda_lag=1.0),
                   dict(n_groups=1, lambda_lag=1.0),
                   # The per-neuron sentinel cannot own private subspaces: D/N
                   # is not an integer and a one-row group has nothing to own.
                   dict(n_groups=0),
                   # ...and it cannot be asked to name a block by that block's
                   # own single row, which is a pure session constant.
                   dict(n_groups=0, group_subspace=False, lambda_space=1.0),
                   dict(n_groups=0, group_subspace=False, lambda_lag=1.0),
                   dict(shuffle_tiles=False)):
        try:
            SoloJigsaw(**kwargs)
            raised = False
        except ValueError:
            raised = True
        _check(raised, f"rejected: {kwargs}")
    _check(SoloJigsaw(n_groups=8, lambda_order=0.0, lambda_pair=0.0,
                      lambda_space=1.0).max_epochs == 60,
           "a space-only arm survives the all-zero-weights guard")
    _check(SoloJigsaw(lambda_order=0.0, lambda_pair=0.0,
                      lambda_shift=1.0).max_epochs == 60,
           "a shift-only arm survives the all-zero-weights guard")

    # 3. successor targets, by hand
    labels = torch.tensor([[[2, 0, 1, 3]]])          # slot0 holds chunk 2, ...
    #   chunk 0 sits in slot 1, chunk 1 in slot 2, chunk 2 in slot 0, chunk 3 in 3
    #   slot 0 holds chunk 2 -> successor is the slot holding chunk 3 = slot 3
    #   slot 1 holds chunk 0 -> slot holding chunk 1 = slot 2
    #   slot 2 holds chunk 1 -> slot holding chunk 2 = slot 0
    #   slot 3 holds chunk 3 -> last -> class K = 4
    _check(successor_targets(labels).tolist() == [[[3, 2, 0, 4]]],
           "successor_targets matches a hand-worked permutation")
    big = torch.rand(32, 3, 5).argsort(-1)
    target = successor_targets(big)
    ok = True
    for b in range(32):
        for g in range(3):
            for slot in range(5):
                chunk = int(big[b, g, slot])
                if chunk == 4:
                    ok &= int(target[b, g, slot]) == 5
                else:
                    ok &= int(big[b, g, int(target[b, g, slot])]) == chunk + 1
    _check(ok, "successor_targets is the inverse-permutation shift at K=5")

    # 4. the chimera really is per-group
    tiles = torch.arange(2 * 4 * 6 * 3, dtype=torch.float32).reshape(2, 4, 6, 3)
    index = torch.tensor([0, 0, 0, 1, 1, 1])
    labels = torch.tensor([[[1, 0, 3, 2], [3, 2, 1, 0]],
                           [[0, 1, 2, 3], [2, 3, 0, 1]]])
    placed = place_groups(tiles, labels, index)
    ok = all(torch.equal(placed[b, j, n], tiles[b, int(labels[b, int(index[n]), j]), n])
             for b in range(2) for j in range(4) for n in range(6))
    _check(ok, "place_groups pulls row n of slot j from chunk labels[b,group[n],j]")
    same = labels[:, :1].expand(-1, 2, -1).contiguous()
    shared = place_groups(tiles, same, index)
    direct = torch.gather(tiles, 1, same[:, 0][:, :, None, None].expand(
        -1, -1, 6, 3))
    _check(torch.equal(shared, direct),
           "with one shared permutation the chimera IS the ordinary tile shuffle")

    # 5. block permutation is a permutation, and the leftovers stay put
    window = torch.randn(5, 11, 4)
    slots = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    sigma = torch.rand(5, 3).argsort(1)
    moved = permute_blocks(window, slots, sigma)
    _check(torch.equal(moved[:, 9:], window[:, 9:]),
           "channels outside the equal-height blocks are untouched")
    ok = all(torch.equal(moved[b, slots[s]], window[b, slots[int(sigma[b, s])]])
             for b in range(5) for s in range(3))
    _check(ok, "row slot s receives the block of group sigma[b,s]")
    _check(torch.allclose(moved.sum(1), window.sum(1)),
           "permuting blocks conserves the total activity")
    zeroed = normalize_blocks(moved, slots, "block_mean")
    means = zeroed[:, slots.reshape(-1)].reshape(5, 3, -1).mean(2)
    _check(torch.allclose(means, torch.zeros_like(means), atol=1e-5),
           "block_mean leaves every block at zero mean (the rate cue is gone)")
    # block_mean does NOT remove the cue the model is most likely to be using.
    # Spell it out, because the 20260928 baseline was built as if it did.
    rows = slots.reshape(-1)
    per_row = zeroed[:, rows].reshape(5, 3, 3, -1).mean(3)
    _check(float(per_row.std()) > 1e-3,
           "block_mean leaves per-NEURON rates inside the block intact -- this "
           "is the static-identity shortcut, and it is why the row_* modes exist")
    for mode in ("row_mean", "row_zscore"):
        stripped = normalize_blocks(moved, slots, mode)
        row_means = stripped[:, rows].mean(dim=2)
        _check(torch.allclose(row_means, torch.zeros_like(row_means), atol=1e-5),
               f"{mode} puts EVERY row at zero mean over the window")
        _check(torch.equal(stripped[:, 9:], moved[:, 9:]),
               f"{mode} touches only the slotted channels")
    scaled = normalize_blocks(moved * 7.0, slots, "row_zscore")
    plain = normalize_blocks(moved, slots, "row_zscore")
    _check(torch.allclose(scaled[:, rows], plain[:, rows], atol=1e-3),
           "row_zscore is invariant to a global gain, so no neuron's gain can "
           "name its block either")
    silent = moved.clone()
    silent[:, 0] = 0.0
    _check(float(normalize_blocks(silent, slots, "row_zscore")[:, 0].abs().max()) == 0.0,
           "a row that never fires becomes exactly zero rather than amplified noise")

    # 5b. the time shuffle keeps every static statistic and destroys order
    generator = torch.Generator().manual_seed(5)
    rolled = shuffle_time(window, generator)
    _check(torch.allclose(rolled.sum(2), window.sum(2), atol=1e-5),
           "shuffle_time conserves each neuron's total, so rates are untouched")
    _check(all(torch.allclose(rolled[b].sum(0).sort().values,
                              window[b].sum(0).sort().values, atol=1e-5)
               for b in range(5)),
           "the population vectors are the same multiset, only reordered")
    # One permutation per SAMPLE, shared by every row. Tested on a constant ramp
    # because on real values each row's sort order is its own and proves nothing.
    ramp = torch.arange(4, dtype=torch.float32).expand(5, 11, 4).contiguous()
    mixed = shuffle_time(ramp, torch.Generator().manual_seed(6))
    _check(all(torch.equal(mixed[b, 0], mixed[b, j]) for b in range(5) for j in range(11)),
           "one permutation per sample, shared by every row")
    _check(not torch.equal(mixed[0, 0], mixed[1, 0]),
           "and a different one for the next sample")
    _check(not torch.equal(rolled, window), "and the order really did change")

    # 5c. the template oracle: a perfect template must win
    template = torch.randn(4, 6)
    order = torch.rand(7, 4).argsort(1)
    _check(torch.equal(nearest_template(template[order], template), order),
           "nearest_template recovers the permutation from exact templates")
    _check(torch.equal(nearest_template(3.5 * template[order], template), order),
           "and is invariant to a global scale, which is why it is cosine")

    # 6. balanced random partition
    index = group_partition(38, 8, 7)
    sizes = torch.bincount(index, minlength=8)
    _check(int(sizes.max() - sizes.min()) <= 1 and int(sizes.sum()) == 38,
           f"group_partition balances 38 neurons into 8 groups {sizes.tolist()}")
    _check(not torch.equal(index, group_partition(38, 8, 8)),
           "a different group_seed gives a different partition")

    # 7. the grouped head: shapes, and G=1 equals the classic head
    head = _GroupOrderHead(64, 32, 4, 8, True)
    z = torch.randn(9, 4, 64)
    position, score = head(z)
    _check(tuple(position.shape) == (9, 8, 4, 4) and tuple(score.shape) == (9, 8, 4),
           "subspace head emits (B,G,K,K) and (B,G,K)")
    wide = _GroupOrderHead(64, 32, 4, 8, False)
    position, score = wide(z)
    _check(tuple(position.shape) == (9, 8, 4, 4) and tuple(score.shape) == (9, 8, 4),
           "non-subspace head emits the same shapes")
    _check(sum(p.numel() for p in _GroupOrderHead(64, 32, 4, 1, True).parameters())
           == sum(p.numel() for p in _OrderHead(64, 32, 4).parameters()),
           "at G=1 the grouped head has exactly the classic head's parameters")
    # Check the slice layout without assuming a global rank lower bound.
    slices = head.slices(torch.randn(3, 4, 64))
    _check(tuple(slices.shape) == (3, 8, 4, 8),
           "slices() cuts the embedding into G disjoint blocks of D/G")

    # 8. the stitch head has no per-span capacity
    stitch = _StitchHead(8, 16)
    logits = stitch(torch.randn(6, 8, 4, 8))
    _check(tuple(logits.shape) == (6, 8, 4, 5), "stitch emits K+1 classes per tile")
    _check(bool(torch.isinf(logits[:, :, 0, 0]).all()),
           "a tile is masked out as its own successor")
    _check(sum(p.numel() for p in stitch.parameters()) == 8 * 16 * 2 + 8 + 1,
           "stitch parameter count is independent of the number of spans")

    # 9. end to end, and the thing this file is for: rank follows n_groups
    ratios = {}
    for groups in (1, 8):
        model = SoloJigsaw(n_groups=groups, window_size=6, n_tiles=3,
                           tile_gap=(1, 3), output_dimension=24,
                           num_hidden_units=32, head_hidden_units=32,
                           lambda_stitch=1.0, lambda_space=0.5 if groups > 1 else 0.0,
                           space_norm="block_mean", batch_size=128, max_epochs=6,
                           device="cpu", random_state=3, verbose=False)
        model.fit(counts)
        embedding = model.transform(counts)
        _check(embedding.shape == (len(counts), 24),
               f"G={groups}: transform returns one 24-d vector per bin")
        _check(np.isfinite(embedding).all(), f"G={groups}: embedding is finite")
        metrics = model.evaluate_pretext(counts, max_spans=128, repeats=2,
                                         verbose=False, n_boot=0)
        _check(metrics["n_groups"] == groups and metrics["chance_pair_percent"] == 50.0,
               f"G={groups}: evaluate_pretext reports the group count")
        _check(metrics["stitch_accuracy_percent"] is not None,
               f"G={groups}: stitch accuracy is reported when the term is on")
        # The tile term plus, when the spatial puzzle is on, the block
        # permutation. `label_bits_per_span` is the argument of this whole file
        # expressed as one number, so it is asserted against a formula written
        # out by hand rather than against whatever the code happens to return.
        expected = groups * math.lgamma(4)
        if model.lambda_space > 0:
            expected += math.lgamma(groups + 1)
        _check(abs(metrics["label_bits_per_span"] - expected) < 1e-6,
               f"G={groups}: the label budget scales with the group count")
        _check(all(float(row["reconstruct"]) == 0.0 for row in model.history_),
               f"G={groups}: reconstruct CE is EXACTLY zero on every epoch")
        ratios[groups] = _participation_ratio(embedding)
        print(f"        G={groups}: participation ratio {ratios[groups]:.2f} / 24, "
              f"pair {metrics['pair_accuracy_percent']:.1f}%, "
              f"continuity baseline {metrics['baseline_continuity_pair_percent']}")
    _check(all(math.isfinite(r) for r in ratios.values()),
           "participation ratios are finite; their ordering is a hypothesis, not a test")

    # 9b. a SPACE-ONLY arm: the three oracles, and the fast path must still
    #     deliver gradient to the encoder. This is the arm the next run leans
    #     on, so it gets checked end to end rather than by inspection.
    space = SoloJigsaw(n_groups=4, window_size=6, n_tiles=3, tile_gap=(1, 3),
                       output_dimension=24, num_hidden_units=32,
                       head_hidden_units=32, lambda_order=0.0, lambda_pair=0.0,
                       lambda_space=1.0, space_norm="row_zscore",
                       batch_size=128, max_epochs=6, device="cpu",
                       random_state=3, verbose=False)
    space.fit(counts)
    metrics = space.evaluate_pretext(counts, max_spans=128, repeats=2,
                                     verbose=False, n_boot=0)
    for key in ("space_accuracy_percent", "space_oracle_percent",
                "space_oracle_rate_percent", "space_oracle_level_percent"):
        _check(metrics[key] is not None, f"space-only arm reports {key}")
    _check(metrics["stitch_accuracy_percent"] is None
           and metrics["train_stitch_accuracy_percent"] is None,
           "a switched-off term reports None, not 0.0 against a 25% chance line")
    _check(metrics["space_norm"] == "row_zscore"
           and metrics["space_time_shuffle"] is False
           and metrics["space_block_rows"] == space.space_block_rows() > 1,
           "the spatial configuration is reported alongside its numbers")
    _check(metrics["chance_space_percent"] == 25.0,
           "chance for a 4-block spatial puzzle is 25%")
    # The no_grad fast path is the one change in this round that could silently
    # break training, so it is checked against its own zero-epoch twin: same
    # seed, same init, no steps. If the encoder weights come out identical the
    # spatial term never reached them and every space-only number would be a
    # frozen encoder wearing a hat.
    frozen = SoloJigsaw(n_groups=4, window_size=6, n_tiles=3, tile_gap=(1, 3),
                        output_dimension=24, num_hidden_units=32,
                        head_hidden_units=32, lambda_order=0.0, lambda_pair=0.0,
                        lambda_space=1.0, space_norm="row_zscore",
                        batch_size=128, max_epochs=0, device="cpu",
                        random_state=3, verbose=False)
    frozen.fit(counts)
    _check(frozen.oracle_profile_samples_ > 1 and frozen.group_profile_ is not None,
           "the untrained control also has a training-only spatial oracle")
    moved_weights = [not torch.allclose(a, b) for a, b in
                     zip(space.encoder_.parameters(), frozen.encoder_.parameters())]
    _check(any(moved_weights),
           "the spatial term ALONE moves the encoder -- the no_grad fast path "
           "skips the permutation backward, not the spatial one")
    _check(float(space.history_[-1]["position"]) > 0.0,
           "the permutation head is still measured on a space-only arm")

    # 9c. mechanism 6, by hand: `roll_rows` is a RIGHT roll, it is a bijection,
    #     and it preserves each row's multiset exactly. That last property is
    #     the whole reason this puzzle cannot be solved by static neuron
    #     identity, so it is asserted rather than assumed.
    window = torch.tensor([[[0., 1., 2., 3.],
                            [10., 20., 30., 40.]]])              # (1,2,4)
    rolled = roll_rows(window, torch.tensor([[1, 3]]))
    _check(rolled.tolist() == [[[3., 0., 1., 2.], [20., 30., 40., 10.]]],
           "roll_rows shifts each row RIGHT by its own amount, independently")
    _check(torch.equal(roll_rows(window, torch.zeros(1, 2, dtype=torch.long)),
                       window), "a zero roll is the identity")
    _check(torch.equal(roll_rows(window, torch.tensor([[4, 4]])), window),
           "a roll of W wraps back to the identity")
    big = torch.randn(7, 5, 8)
    shift = torch.randint(0, 8, (7, 5))
    _check(torch.equal(roll_rows(big, shift).sort(dim=2).values,
                       big.sort(dim=2).values),
           "a roll preserves every row's multiset -- a sorted row says NOTHING "
           "about the answer, so only position can carry it")
    _check(torch.equal(roll_rows(roll_rows(big, shift), (8 - shift) % 8), big),
           "rolling by r then by W-r returns the original: it is a bijection")

    # 9d. the label-free opponent has to be able to win when the answer is
    #     actually there, or a low oracle number would be evidence of a broken
    #     baseline rather than of a hard task. Exact template, so it must be
    #     perfect -- and scale-invariant, because it is a centered cosine.
    template = torch.randn(5, 8)
    truth = torch.randint(0, 8, (7, 5))
    exact = roll_rows(template[None].expand(7, -1, -1).contiguous(), truth)
    _check(torch.equal(roll_scores(exact, template).argmax(-1), truth),
           "roll_scores recovers a known roll exactly from its own template")
    scaled = exact * torch.rand(7, 5, 1).clamp_min(0.2) + torch.randn(7, 5, 1)
    _check(torch.equal(roll_scores(scaled, template).argmax(-1), truth),
           "roll_scores is invariant to each row's gain and offset, so the "
           "oracle is not beaten by normalization the encoder also gets")
    _check(roll_scores(exact, template).shape == (7, 5, 8),
           "roll_scores returns one score per (span, neuron, candidate shift)")

    # 9e. mechanisms 5 and 6 end to end. The sentinel must resolve to N, the
    #     head must be the right shape, the dead-row mask must actually exclude
    #     constant rows, and the frozen control must have the SAME label
    #     marginal as the true task -- an argsort control would be a permutation
    #     (no repeats) where the truth is N i.i.d. draws, and the gap would then
    #     be a distribution difference wearing the label "learning".
    both = SoloJigsaw(n_groups=0, group_subspace=False, lambda_shift=1.0,
                      window_size=6, n_tiles=3, tile_gap=(1, 3),
                      output_dimension=24, num_hidden_units=32,
                      head_hidden_units=32, batch_size=128, max_epochs=6,
                      device="cpu", random_state=3, verbose=False)
    both.fit(counts)
    _check(both.oracle_profile_samples_ > both.batch_size,
           "phase oracle uses all training span starts, beyond the first batch")
    _check("shift_profile" in both.order_head_.state_dict()
           and "oracle_profile_samples" in both.order_head_.state_dict(),
           "oracle template and calibration count are persisted head buffers")
    channels = counts.shape[1]
    _check(both.n_groups == 0 and both.n_groups_ == channels,
           f"the n_groups=0 sentinel resolves to N={channels} at fit time and "
           "the raw parameter is left alone for the runner to match on")
    _check(len(both.group_sizes_) == channels
           and set(int(s) for s in both.group_sizes_) == {1},
           "one group per neuron means N groups of exactly one row each")
    _check(both.order_head_.shift(torch.zeros(4, 24)).shape == (4, channels, 6),
           "_ShiftHead emits N x W affine logits from one D-vector")
    _check(both._frozen_shift.shape[1] == channels
           and int(both._frozen_shift.max()) < 6
           and int(both._frozen_shift.min()) >= 0,
           "the frozen shift table is drawn from the same {0..W-1} as the truth")
    duplicates = (both._frozen_shift.shape[1]
                  - torch.tensor([len(set(row.tolist()))
                                  for row in both._frozen_shift[:64]]).float().mean())
    _check(duplicates > 0,
           f"the frozen control REPEATS shift values ({duplicates:.1f} per row "
           "on average), exactly as i.i.d. draws do and a permutation never would")
    flat = torch.zeros(3, channels, 6)                 # every row constant
    flat[:, :2] = torch.randn(3, 2, 6)                 # ...except two of them
    _, _, alive = both._shift_view(
        flat, torch.zeros(3, dtype=torch.long, device=both.device_),
        both._generator)
    _check(alive.shape == (3, channels) and int(alive.sum()) == 6,
           "a row that is constant inside the window has no recoverable phase "
           "and is dropped from BOTH the loss and the accuracy")
    control_before = both.label_control
    seed = 819
    true_view = both._shift_view(flat, torch.zeros(3, dtype=torch.long),
                                torch.Generator(device=both.device_).manual_seed(seed))
    both.label_control = "frozen_random"
    frozen_view = both._shift_view(flat, torch.zeros(3, dtype=torch.long),
                                  torch.Generator(device=both.device_).manual_seed(seed))
    both.label_control = control_before
    _check(all(torch.equal(a, b) for a, b in zip(true_view, frozen_view)),
           "frozen labels cannot change phase inputs, true shifts, or the mask")
    metrics = both.evaluate_pretext(counts, max_spans=128, repeats=2,
                                    verbose=False, n_boot=0)
    for key in ("shift_accuracy_percent", "shift_oracle_percent",
                "shift_live_rows_percent", "train_shift_accuracy_percent"):
        _check(metrics[key] is not None, f"the high-rank arm reports {key}")
    _check(metrics["chance_shift_percent"] == 100.0 / 6
           and metrics["shift_classes"] == 6,
           "chance for a 6-bin row phase is 1/6, quoted next to the number")
    expected = channels * (math.lgamma(4) + math.log(6))
    _check(abs(metrics["label_bits_per_span"] - expected) < 1e-6,
           f"the high-rank pair asks {expected:.0f} nats per span against the "
           f"{math.lgamma(4):.2f} a rank-1 jigsaw asks -- a factor of "
           f"{expected / math.lgamma(4):.0f}, which is the entire idea")
    _check(all(float(row["reconstruct"]) == 0.0 for row in both.history_),
           "no reconstruction anywhere: the CE is EXACTLY zero on every epoch")
    solo_rank1 = SoloJigsaw(window_size=6, n_tiles=3, tile_gap=(1, 3),
                            output_dimension=24, num_hidden_units=32,
                            head_hidden_units=32, batch_size=128, max_epochs=6,
                            device="cpu", random_state=3, verbose=False)
    solo_rank1.fit(counts)
    pair = (_participation_ratio(both.transform(counts)),
            _participation_ratio(solo_rank1.transform(counts)))
    print(f"        participation ratio: per-neuron+phase {pair[0]:.2f} / 24 "
          f"vs classic rank-1 jigsaw {pair[1]:.2f} / 24")
    _check(all(math.isfinite(r) for r in pair),
           "participation ratios are finite; greater rank is not assumed")

    # 10. the arm table
    stray, broke = [], []
    known = set(SoloJigsaw._PARAM_NAMES) | {"_model"}
    for name, settings in SOLO_ARMS.items():
        stray += [f"{name}.{k}" for k in settings if k not in known]
        try:
            SoloJigsaw(**{k: v for k, v in settings.items() if k != "_model"})
        except Exception as error:                            # noqa: BLE001
            broke.append(f"{name}: {error}")
    _check(not stray, f"every arm key is a real parameter{'' if not stray else stray}")
    _check(not broke, f"every arm constructs{'' if not broke else broke}")
    # No reconstruction arm may creep back in. Every entry in this file has to
    # be constructible as a SoloJigsaw, and that class refuses a reconstruction
    # weight in its constructor, so this is belt-and-braces -- but the previous
    # version of this file DID register a second model key, and the assertion
    # is cheaper than noticing it in a 6000-epoch log.
    leaked = sorted({k for s in SOLO_ARMS.values() for k in s
                     if "reconstruct" in k or "forecast" in k}
                    | {s["_model"] for s in SOLO_ARMS.values()} - {"solo"})
    _check(not leaked,
           f"the arm table is jigsaw-family only{'' if not leaked else leaked}")
    named = set(SOLO_ARMS)
    missing = [f"{a}-{b}" for a, b, _, _ in SOLO_CONTRASTS
               if a not in named or b not in named]
    _check(not missing, f"every contrast names real arms{'' if not missing else missing}")
    _check(all(a in named for a in DEFAULT_SOLO_ARMS),
           "DEFAULT_SOLO_ARMS resolves -- every one of them, which is what the "
           "20260928 run could not say about reconstruct_only")
    registered, arm_table = register({}, {}, [])
    _check(set(registered) == {"solo"}
           and not [a for a in DEFAULT_SOLO_ARMS if a not in arm_table],
           "register() wires exactly one model key and every default arm")
    for left, right in (("solo_rank8", "solo_shared8"),
                        ("solo_space_only", "solo_space_rowz"),
                        ("solo_space_only", "solo_space_tshuf"),
                        ("solo_space_only", "solo_space_frozen"),
                        # The three that decide this round.
                        ("solo_neuron", "solo_neuron_shared"),
                        ("solo_neuron", "solo_random_neuron"),
                        ("solo_neuron", "solo_neuron_frozen"),
                        ("solo_shift", "solo_shift_frozen"),
                        ("solo_shift", "solo_shift_untrained"),
                        ("solo_neuron_shift", "solo_neuron"),
                        ("solo_neuron_shift", "solo_neuron_shift_spread")):
        differ = [k for k in SOLO_MATCH_KEYS
                  if SOLO_ARMS[left].get(k) != SOLO_ARMS[right].get(k)]
        _check(not differ,
               f"{left} - {right} is geometry-matched{'' if not differ else differ}")
    print("\nall checks passed.")


if __name__ == "__main__":
    _smoke_test()







# """jigsaw_solo.py -- the jigsaw STANDING ALONE. No reconstruction anywhere.

# WHY THIS FILE EXISTS
# --------------------
# The 20260927 run settled the previous question, and not in the jigsaw's favour.
# Every matched contrast came out negative:

#     order_default - anchor_default      -0.0344     (stock geometry)
#     order_gapped  - anchor_gapped       -0.0402     (continuity blocked, gap 3)
#     order_default - ctrl_frozen_labels  -0.0359     (labels that mean nothing)
#     order_full    - order_full_frozen   -0.0050     (same loss, dead labels)

# and the single best arm in the whole table was `ctrl_frozen_labels` at 0.5617 --
# the arm whose jigsaw labels carry NO information about time. On the 3-seed run
# the same thing happened again: order_lag_frozen 0.5500 and
# order_time_axis_frozen 0.5467 beat their own label-carrying versions (0.5371,
# 0.5404). Whatever the order term was doing, it was not teaching the encoder
# about time; it was injecting gradient noise, and a meaningless label injected it
# just as well. Everything that made those numbers good came from the
# reconstruction anchor.

# So: drop the anchor, and make the PUZZLE carry the load.

# WHY THE PUZZLE COULD NOT CARRY IT BEFORE -- the one-line diagnosis
# -----------------------------------------------------------------
# Sorting K tiles is a RANK-1 problem. A permutation of K items is recovered from
# one scalar per item: give each tile a score, sort the scores, done. So the
# minimal encoder that solves the classic temporal jigsaw perfectly is

#     f(tile) = (an estimate of when this tile happened)   -- one number.

# Every other direction of the embedding is free to be anything, including
# nothing. That is not a training pathology, it is the information geometry of the
# task, and it predicts all three things we measured:

#   * alone, the objective collapses the embedding -- `order_only` scored R2 0.005
#     at participation ratio 3.4 out of 64, which is what "one useful direction
#     plus noise" looks like;
#   * bolted onto a reconstruction anchor it contributes ~nothing, because it only
#     ever pushes on one of the 64 directions and it pushes AGAINST the anchor
#     there;
#   * a frozen random label works as well as the true one, because a rank-1 target
#     is a rank-1 target whatever it means.

# The label budget says the same thing in bits. One span of the classic jigsaw
# carries log(4!) = 3.18 nats of supervision. One span of the reconstruction
# target carries 38 neurons x 10 bins x log(8) levels = 790 nats. A factor of 248.
# The autoencoder was never beating the jigsaw on cleverness; it was beating it on
# supervision, by two and a half orders of magnitude.

# WHAT THIS FILE DOES ABOUT IT
# ----------------------------
# Raise the RANK and the BIT COUNT of the puzzle itself, without adding a
# reconstruction term, without InfoNCE, without contrastive learning, and without
# leaving cross-entropy. Three mechanisms, each a puzzle, each pure CE:

# 1. GROUPED JIGSAW (`n_groups`). Split the neurons into G groups and draw an
#    INDEPENDENT permutation of the K time chunks FOR EACH GROUP. Tile j is then a
#    chimera: group 1's rows come from chunk 3, group 2's from chunk 1, and so on.
#    The head must place all G of them. One scalar can no longer do it -- two
#    groups with different permutations need two different scores from the same
#    embedding -- so the sufficient statistic is G-dimensional and the label budget
#    is G x log(K!) nats per span instead of log(K!). G=1 is EXACTLY the classic
#    jigsaw, same head, same parameter count, which is what makes `solo_rank1` an
#    honest control rather than a different model.

# 2. GROUP SUBSPACES (`group_subspace`). Group g reads only its own D/G slice of
#    the embedding. Now the rank floor is structural, not statistical: if slice g
#    collapses, group g's puzzle is unsolvable and its cross-entropy stays at
#    log K forever. Collapse stops being a free minimum and becomes a penalty.

# 3. STITCH CE (`lambda_stitch`). For each tile, "which of the other tiles is my
#    immediate successor?" -- K+1 classes, the last one meaning "nothing follows
#    me". Scored by a BILINEAR compatibility form, which is the point: the head is
#    a fixed pair of D x r matrices with NO per-span capacity, so it cannot become
#    the lookup table that took `order_default`'s train pair accuracy to 100% while
#    held-out sat at 48.2%. The only way to lower this loss is an embedding in
#    which the end of one chunk and the start of the next are recognisably the same
#    trajectory -- an instantaneous-state code, which is exactly what a velocity
#    decoder wants, and the opposite of the "when am I" code that sorting rewards.

#    and a fourth, on the other axis of the data:

# 4. NEURON-AXIS JIGSAW (`lambda_space`). The original jigsaw puzzle is SPATIAL.
#    Permute which neuron block sits in which row slot and ask the model to name
#    them: G-way CE per slot, log(G!) nats per span. Solving it requires the
#    embedding to carry per-group identity -- which cells are active, not when --
#    and that is high-rank by construction.

# One knob is NOT a puzzle and is labelled as such: `lambda_spread` is a variance
# floor on the embedding (a degeneracy guard, no positives, no negatives, no
# InfoNCE). It defaults to 0 so that nothing in the headline number comes from it
# unless you ask; there is an arm that turns it on so its contribution stays
# attributable.

# WHAT THE 20260928 RUN SAID -- read this before the sections above
# -----------------------------------------------------------------
# Three seeds, 6000 epochs, Perich. R2 (MLP) against the FROZEN RANDOM ENCODER at
# 0.3199, which is the only floor that means anything:

#     solo_full       0.4345    everything on
#     solo_no_stitch  0.4158    - stitch
#     solo_random     0.3199    frozen, untrained
#     solo_rank8      0.3186    the grouped TEMPORAL puzzle, alone
#     solo_no_space   0.2538    temporal + stitch, no spatial term
#     solo_full_frozen 0.1592   everything on, labels meaningless
#     solo_shared8    0.0269    the shared-permutation control

# Two things follow, and they point in opposite directions.

# The good one: the jigsaw now stands alone. 0.4345 MLP / 0.3292 ridge with
# `reconstruct CE=0.0000` printed on every epoch of every arm, against a RAW
# WINDOW ceiling of 0.4997 / 0.4283 -- 87% and 77% of what the decoder gets from
# the unprocessed spikes. That is the first time in this project a number came out
# of a puzzle with no autoencoder underneath it.

# The bad one: mechanism 1, the thing the file was named after, is worth zero.
# `solo_rank8` 0.3186 versus `solo_random` 0.3199 is a dead heat with an UNTRAINED
# encoder, and `solo_no_space` at 0.2538 says the temporal puzzle plus stitch is
# actively worse than not training at all. Every point of the result is
# mechanism 4, the NEURON-AXIS puzzle. `solo_rank8 - solo_shared8` came out
# +0.2916 and seed-consistent, but that contrast only says the labels have to be
# independent; it does not say sorting time is useful, and the frozen-encoder
# comparison says it is not.

# So the spatial term is the finding and the temporal ladder is now a control.

# WHAT IS NOT YET ESTABLISHED ABOUT THE SPATIAL TERM
# --------------------------------------------------
# Held-out block accuracy was 85.2 / 94.0 / 95.8% against 12.5% chance, with train
# accuracy LOWER than held-out (no memorization) and the frozen-label twin at
# exactly chance out of sample. All real. But the label-free baseline it was
# measured against, at 13-16%, was built from a ROW-AVERAGED block template, and
# averaging over the rows deletes the single most likely shortcut: "in this block
# neuron 7 outfires neuron 12", a constant of the session that the encoder can
# learn once and reuse on every span. The baseline was blindfolded, so the 85-96%
# is not yet evidence about dynamics.

# This round fixes that rather than assuming either answer. `space_oracle` is now
# a full (S,W) patch template, `space_oracle_rate` is the pure static-identity
# rule, and two new arms delete the cue at the source: `space_norm="row_zscore"`
# removes every neuron's own mean and gain inside the window, and
# `space_time_shuffle` permutes the time bins so every static statistic survives
# and only the order dies. If `solo_space_only - solo_space_rowz` is near zero the
# claim is about dynamics; if it is large the claim is "the embedding must encode
# which cells these are", which is still a real self-supervised objective and
# still beats the autoencoder if the R2 says so -- just a different sentence in
# the paper.

# THE GUARANTEE
# -------------
# `lambda_reconstruct` and `lambda_forecast` are forced to zero in the constructor
# and raise if you pass anything else. There is no reconstruction head in the loss
# of this file. When you run it you will see `reconstruct CE=0.0000` on every
# epoch line -- that zero is the proof, printed once per log interval, that the
# number at the end came from the puzzle.

# WHAT WOULD FALSIFY THE IDEA
# ---------------------------
# `solo_space_only - solo_random` and `solo_space_only - solo_space_frozen`. The
# first asks whether the spatial puzzle beats an untrained encoder at all -- the
# test the temporal puzzle failed on 20260928. The second asks whether it needs
# the labels to mean anything. Both must be clearly positive, on every seed,
# before any of the rest of this file is worth reading. `solo_space_only -
# ae_matched` is then the number the project is about.

# API is the same as the rest of the project: fit / transform / fit_transform /
# evaluate_pretext / save / load, sklearn-style, no CEBRA, no contrastive loss.
# """

# from __future__ import annotations

# import math

# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch import nn

# from jigsaw_net import (_GradScale, _OrderHead, _assign, _augment, _boolean,
#                         _gather_tiles, _integer, _order_metrics, _pair_targets,
#                         _participation_ratio, _ranks, _real)
# from jigsaw_order import (ANCHOR_MATCH_KEYS, OrderJigsaw, block_bootstrap,
#                           continuity_strength, roll_tiles, select_span,
#                           shortcut_strength)

# # Two families, and the difference between them is the whole audit.
# #   block_* removes a statistic of the WHOLE block, so "neuron 7 fires twice as
# #           often as its blockmate neuron 12" survives -- a constant of the
# #           session, learnable once and reusable on every span.
# #   row_*   removes that statistic PER NEURON, so no neuron's mean rate (and
# #           under row_zscore no neuron's gain either) can name the block. What
# #           is left is the temporal shape of each row, which is a fact about
# #           this span and not about the session.
# # The 20260928 run only ever used block_mean, so its 85-96% space accuracy is
# # not yet evidence of anything beyond static neuron identity.
# SPACE_NORMS = ("none", "block_mean", "block_zscore", "row_mean", "row_zscore")


# # --------------------------------------------------------------------------- #
# # 1. heads
# # --------------------------------------------------------------------------- #
# class _GroupOrderHead(nn.Module):
#     """K-way slot logits and a before/after score, FOR EACH of G groups.

#     Two wirings, and the difference between them is the whole rank argument.

#     subspace=True   group g is shown only z[..., g*w:(g+1)*w]. The readout
#                     weights are shared across groups, so the groups can only
#                     give different answers if their slices hold different
#                     information. That is the structural rank floor: a dead slice
#                     is a group stuck at log K forever.
#     subspace=False  every group sees the whole embedding and gets its own
#                     readout row. Rank is then only encouraged, never enforced --
#                     which is the control that tells you whether the subspaces
#                     are doing the work or the independent labels are.

#     At G=1 with subspace=True this is `_OrderHead` exactly: same input width,
#     same body, same two readouts, same parameter count. `solo_rank1` is
#     therefore the classic jigsaw with the anchor removed, not a new model that
#     happens to resemble it.
#     """

#     def __init__(self, dimension, hidden, n_tiles, n_groups, subspace):
#         super().__init__()
#         self.n_tiles, self.n_groups, self.subspace = n_tiles, n_groups, subspace
#         self.width = dimension // n_groups if subspace else dimension
#         self.body = nn.Sequential(nn.Linear(2 * self.width, hidden), nn.GELU(),
#                                   nn.Linear(hidden, hidden), nn.GELU())
#         outputs = n_tiles if subspace else n_groups * n_tiles
#         self.position = nn.Linear(hidden, outputs)
#         self.score = nn.Linear(hidden, 1 if subspace else n_groups)

#     def slices(self, z):
#         """(B, K, D) -> (B, G, K, width). The view every group-wise term uses."""
#         batch, n_tiles = z.shape[0], z.shape[1]
#         if not self.subspace:
#             return z[:, None].expand(-1, self.n_groups, -1, -1)
#         return z.reshape(batch, n_tiles, self.n_groups,
#                          self.width).permute(0, 2, 1, 3)

#     def forward(self, z):
#         batch, n_tiles = z.shape[0], z.shape[1]
#         if self.subspace:
#             parts = self.slices(z)                                # (B,G,K,w)
#             context = parts.mean(dim=2, keepdim=True).expand_as(parts)
#             tokens = self.body(torch.cat((parts, context), dim=-1))
#             return self.position(tokens), self.score(tokens).squeeze(-1)
#         context = z.mean(dim=1, keepdim=True).expand_as(z)
#         tokens = self.body(torch.cat((z, context), dim=-1))       # (B,K,H)
#         position = self.position(tokens).reshape(
#             batch, n_tiles, self.n_groups, self.n_tiles).permute(0, 2, 1, 3)
#         score = self.score(tokens).permute(0, 2, 1)               # (B,G,K)
#         return position, score


# class _StitchHead(nn.Module):
#     """"Which tile follows me?" as a bilinear form with no per-span capacity.

#     logit(i -> j) = <Q z_i, K z_j> / sqrt(r), plus one extra class per tile for
#     "nothing follows me, I am last". The self-pair is masked out.

#     The parameter count is 2 x width x r plus a vector, INDEPENDENT of how many
#     spans exist. That matters more than it sounds: the reason the old order term
#     stopped teaching the encoder anything was that its MLP head had enough
#     capacity to identify which of the ~900 span starts it was looking at and
#     answer from a table -- train pair accuracy 100%, held-out 48.2%. A bilinear
#     score cannot hold a table. It can only report whether two embeddings look
#     like consecutive pieces of one trajectory, so if this loss falls, something
#     in the embedding got better at representing the trajectory.
#     """

#     def __init__(self, width, rank):
#         super().__init__()
#         self.query = nn.Linear(width, rank, bias=False)
#         self.key = nn.Linear(width, rank, bias=False)
#         self.last = nn.Linear(width, 1)
#         self.scale = rank ** -0.5

#     def forward(self, parts):                                     # (B,G,K,w)
#         query, key = self.query(parts), self.key(parts)
#         logits = torch.einsum("bgir,bgjr->bgij", query, key) * self.scale
#         n_tiles = parts.shape[2]
#         eye = torch.eye(n_tiles, dtype=torch.bool, device=parts.device)
#         logits = logits.masked_fill(eye, float("-inf"))
#         return torch.cat((logits, self.last(parts)), dim=-1)      # (B,G,K,K+1)


# class _SpaceHead(nn.Module):
#     """G-way "which neuron group is sitting in this row slot?" per slot."""

#     def __init__(self, dimension, hidden, n_groups):
#         super().__init__()
#         self.n_groups = n_groups
#         self.net = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
#                                  nn.Linear(hidden, n_groups * n_groups))

#     def forward(self, z):                                         # (B,D)
#         return self.net(z).reshape(-1, self.n_groups, self.n_groups)


# # --------------------------------------------------------------------------- #
# # 2. functional pieces
# # --------------------------------------------------------------------------- #
# def successor_targets(labels):
#     """(B,G,K) chronological labels -> (B,G,K) successor SLOT, class K = last.

#     `labels[b,g,j]` is the chronological index of the chunk presented in slot j,
#     so `labels.argsort(-1)[b,g,c]` is the slot holding chunk c -- the inverse
#     permutation. The successor of slot j is therefore the slot holding chunk
#     labels[b,g,j] + 1, and the tile holding the final chunk gets the extra
#     class. Written out rather than looped because it runs every step.
#     """
#     n_tiles = labels.shape[-1]
#     inverse = labels.argsort(dim=-1)
#     following = torch.gather(inverse, -1, (labels + 1).clamp(max=n_tiles - 1))
#     return torch.where(labels == n_tiles - 1,
#                        torch.full_like(following, n_tiles), following)


# def place_groups(tiles, labels, group_index):
#     """Build the chimera: (B,K,N,W) tiles + (B,G,K) labels -> (B,K,N,W).

#     Row n of presented tile j is taken from chronological chunk
#     labels[b, group_index[n], j]. With G=1 this reduces to the ordinary tile
#     shuffle, element for element, which is what keeps `solo_rank1` a true
#     control.
#     """
#     index = labels[:, group_index, :].permute(0, 2, 1)            # (B,K,N)
#     index = index[..., None].expand(-1, -1, -1, tiles.shape[3])
#     return torch.gather(tiles, 1, index)


# def permute_blocks(window, slots, sigma):
#     """Row-permute equal-size neuron blocks. (B,N,W), (G,S), (B,G) -> (B,N,W).

#     `slots[g]` lists the channel indices belonging to group g, TRUNCATED to a
#     common size S so that every block is the same height. Unequal blocks would
#     hand the answer over for free -- a five-row block among four-row blocks
#     names itself -- and the N mod G leftover channels are therefore left where
#     they are rather than being padded into a block.

#     `sigma[b, s]` is the true group placed in row slot s, which is exactly the
#     label the space head is asked to produce.
#     """
#     rows = slots.reshape(-1)                                      # (G*S,)
#     source = slots[sigma].reshape(window.shape[0], -1)            # (B,G*S)
#     gathered = torch.gather(
#         window, 1, source[..., None].expand(-1, -1, window.shape[2]))
#     out = window.clone()
#     out[:, rows, :] = gathered
#     return out


# def normalize_blocks(window, slots, mode):
#     """Remove a cue the space task would otherwise ride on. (B,N,W) -> (B,N,W).

#     Every mode here is computed per BLOCK or per ROW of a block, never over the
#     whole window, so the statistic travels with the block wherever the row
#     permutation puts it. That is what keeps the normalization from leaking the
#     answer: slot s is normalized by the contents of slot s, and those contents
#     are whichever group landed there.

#     none          nothing removed. The rate cue is fully available.
#     block_mean    the block's scalar level. Relative rates WITHIN the block
#                   survive, and those are session constants.
#     block_zscore  the level and the overall scale, same caveat.
#     row_mean      each neuron's own mean over the window. No neuron's rate can
#                   name its block any more; gains still can.
#     row_zscore    each neuron's mean AND gain. Only the temporal shape of each
#                   row is left, which is the one cue that is about this span.
#                   Rows that never fire in the window become exactly zero, so a
#                   silent neuron stops being an identity beacon.
#     """
#     if mode == "none":
#         return window
#     rows = slots.reshape(-1)
#     groups, block = slots.shape
#     patch = window[:, rows, :].reshape(len(window), groups, block, -1)
#     if mode in ("row_mean", "row_zscore"):
#         centered = patch - patch.mean(dim=3, keepdim=True)
#         if mode == "row_zscore":
#             deviation = patch.std(dim=3, keepdim=True)
#             centered = torch.where(deviation > 1e-4, centered / (deviation + 1e-5),
#                                    torch.zeros_like(centered))
#     else:
#         flat = patch.reshape(len(window), groups, -1)
#         centered = flat - flat.mean(dim=2, keepdim=True)
#         if mode == "block_zscore":
#             centered = centered / (flat.std(dim=2, keepdim=True) + 1e-5)
#         centered = centered.reshape(patch.shape)
#     out = window.clone()
#     out[:, rows, :] = centered.reshape(len(window), len(rows), -1)
#     return out


# def shuffle_time(window, generator):
#     """Permute the W time bins, one permutation per sample, shared by all rows.

#     The control for "does the spatial puzzle need dynamics at all?". Every
#     instantaneous population vector is preserved exactly and so is every
#     neuron's mean rate, variance and cross-neuron covariance; the only thing
#     destroyed is the ORDER of the bins. A model reading static neuron identity
#     scores exactly the same with this on. A model reading each block's temporal
#     signature does not.
#     """
#     samples, channels, width = window.shape
#     order = torch.rand(samples, width, device=window.device,
#                        generator=generator).argsort(dim=1)
#     return torch.gather(window, 2, order[:, None, :].expand(-1, channels, -1))


# def nearest_template(patch, template):
#     """(B,G,F) candidates against (G,F) templates -> (B,G) predicted group.

#     The label-free opponent for the spatial puzzle. Cosine and not squared
#     distance on purpose: the template is accumulated during training, under
#     neuron dropout and gain jitter, so it carries a global scale the evaluation
#     window does not have. Under L2 that scale alone shifts the argmin toward
#     whichever template happens to be smallest, and the oracle would then lose
#     for a bookkeeping reason instead of a real one.
#     """
#     return (F.normalize(patch, dim=-1)
#             @ F.normalize(template, dim=-1).t()).argmax(-1)


# def group_partition(channels, n_groups, seed):
#     """Balanced groups over a RANDOM assignment of neurons.

#     Contiguous groups would be groups of neighbouring electrodes, and on array
#     recordings neighbouring electrodes share tuning -- the puzzle would then be
#     easier for reasons that have nothing to do with the model. Round-robin over
#     a seeded permutation keeps the sizes within one of each other while making
#     membership arbitrary.
#     """
#     generator = torch.Generator().manual_seed(int(seed))
#     order = torch.randperm(channels, generator=generator)
#     index = torch.empty(channels, dtype=torch.long)
#     index[order] = torch.arange(channels) % n_groups
#     return index


# # --------------------------------------------------------------------------- #
# # 3. the model
# # --------------------------------------------------------------------------- #
# class SoloJigsaw(OrderJigsaw):
#     """A jigsaw that has to stand on its own: no reconstruction, no forecast.

#     n_groups         independent puzzles per span. 1 = the classic jigsaw.
#     group_chimera    True  -> one permutation PER GROUP (the rank mechanism)
#                      False -> one permutation shared by all groups, same
#                               architecture. This is the control for n_groups.
#     group_subspace   group g reads only its own D/G slice of the embedding.
#                      Requires output_dimension % n_groups == 0.
#     group_seed       which random partition of the neurons to use.
#     lambda_stitch    weight on the successor CE (bilinear head, no lookup table)
#     stitch_rank      rank of that bilinear form
#     lambda_space     weight on the neuron-axis (spatial) jigsaw
#     space_norm       which cue to delete from the spatial puzzle. See
#                      `normalize_blocks`: the block_* modes leave per-neuron
#                      rates intact, the row_* modes do not.
#     space_time_shuffle
#                      permute the window's time bins before the spatial puzzle.
#                      Keeps every static statistic, destroys temporal order.
#     lambda_spread    variance floor on the embedding. NOT a puzzle: a
#                      degeneracy guard, off by default, with its own arm so its
#                      contribution never hides inside a headline number.

#     Everything inherited from OrderJigsaw still applies: span selection, the
#     gap curriculum, order-view-only augmentation, hard-example mining, head
#     resets, and the label controls. `lambda_lag` is available at n_groups=1
#     only -- with independent per-group permutations a single signed lag between
#     two tiles is not well defined, and the 3-seed run showed the lag readout
#     scoring HIGHER with frozen labels (25.0%) than with true ones (21.9%), so it
#     was not evidence of anything worth generalizing.
#     """

#     _MODEL_TYPE = "solo_jigsaw"
#     _PARAM_NAMES = OrderJigsaw._PARAM_NAMES + (
#         "n_groups", "group_chimera", "group_subspace", "group_seed",
#         "lambda_stitch", "stitch_rank", "lambda_space", "space_norm",
#         "space_time_shuffle", "lambda_spread")

#     def __init__(self, *, n_groups=1, group_chimera=True, group_subspace=True,
#                  group_seed=101, lambda_stitch=0.0, stitch_rank=32,
#                  lambda_space=0.0, space_norm="none", space_time_shuffle=False,
#                  lambda_spread=0.0, **kwargs):
#         for forbidden in ("lambda_reconstruct", "lambda_forecast"):
#             if float(kwargs.pop(forbidden, 0.0)) != 0.0:
#                 raise ValueError(
#                     f"{forbidden} must be 0 in SoloJigsaw. This file exists to "
#                     "measure what the PUZZLE is worth with no anchor holding it "
#                     "up; an arm that quietly turns the anchor back on would "
#                     "answer a question we already answered (the anchor is worth "
#                     "a lot). Use jigsaw_order.OrderJigsaw for anchored arms.")
#         if space_norm not in SPACE_NORMS:
#             raise ValueError(f"space_norm must be one of {SPACE_NORMS}.")
#         # super()'s "all loss weights are zero" guard predates the solo terms
#         # and would reject a space-only or stitch-only arm. Build with
#         # max_epochs=0 so the guard cannot fire, restore the real budget, then
#         # run the SAME guard over the real weight set below.
#         epochs = kwargs.pop("max_epochs", 60)
#         super().__init__(max_epochs=0, lambda_reconstruct=0.0,
#                          lambda_forecast=0.0, **kwargs)
#         self.max_epochs = _integer("max_epochs", epochs, 0)
#         self.n_groups = _integer("n_groups", n_groups, 1)
#         self.group_chimera = _boolean("group_chimera", group_chimera)
#         self.group_subspace = _boolean("group_subspace", group_subspace)
#         self.group_seed = _integer("group_seed", group_seed, 0)
#         self.lambda_stitch = _real("lambda_stitch", lambda_stitch, 0)
#         self.stitch_rank = _integer("stitch_rank", stitch_rank, 1, 512)
#         self.lambda_space = _real("lambda_space", lambda_space, 0)
#         self.space_norm = space_norm
#         self.space_time_shuffle = _boolean("space_time_shuffle", space_time_shuffle)
#         self.lambda_spread = _real("lambda_spread", lambda_spread, 0)
#         if self.group_subspace and self.output_dimension % self.n_groups:
#             raise ValueError(
#                 f"group_subspace splits the embedding into n_groups slices, so "
#                 f"output_dimension ({self.output_dimension}) must be divisible "
#                 f"by n_groups ({self.n_groups}). Pick 64/8, 64/4, 60/5 and so "
#                 f"on, or set group_subspace=False.")
#         if self.lambda_space > 0 and self.n_groups < 2:
#             raise ValueError("lambda_space needs n_groups >= 2: with one block "
#                              "there is no spatial permutation to recover.")
#         if self.lambda_lag > 0 and self.n_groups > 1:
#             raise ValueError(
#                 "lambda_lag is defined for a single shared permutation and "
#                 "n_groups > 1 draws one permutation per group, so 'the signed "
#                 "offset between tile i and tile j' has G different answers. Set "
#                 "n_groups=1 to use the lag term.")
#         live = (self.lambda_order + self.lambda_pair + self.lambda_lag
#                 + self.lambda_stitch + self.lambda_space + self.lambda_spread)
#         if live == 0 and self.max_epochs > 0:
#             raise ValueError("All loss weights are zero; set max_epochs=0 for a "
#                              "random encoder.")
#         if not self.shuffle_tiles and self.lambda_order + self.lambda_pair > 0:
#             raise ValueError(
#                 "shuffle_tiles=False hands every group the identity permutation, "
#                 "so the grouped puzzle has a constant answer and the whole rank "
#                 "argument evaporates. Keep it True.")

#     # -- construction ------------------------------------------------------- #
#     def _build(self, channels):
#         # OrderJigsaw._build gives us the frozen label table, the lag head, the
#         # lag edges, the span cache and the step counter. We keep all of it and
#         # only swap the order head, so the inherited machinery stays wired.
#         super()._build(channels)
#         if self.n_groups > channels:
#             raise ValueError(f"n_groups={self.n_groups} exceeds the {channels} "
#                              "recorded neurons.")
#         self.group_index_ = group_partition(
#             channels, self.n_groups, self.random_state + self.group_seed
#         ).to(self.device_)
#         sizes = torch.bincount(self.group_index_, minlength=self.n_groups)
#         self.group_sizes_ = sizes.tolist()
#         lag_head = self.order_head_.lag_head
#         self.order_head_ = _GroupOrderHead(
#             self.output_dimension, self.head_hidden_units, self.n_tiles,
#             self.n_groups, self.group_subspace).to(self.device_)
#         self.order_head_.lag_head = lag_head
#         self.lag_head_ = lag_head
#         if self.lambda_stitch > 0:
#             self.order_head_.stitch = _StitchHead(
#                 self.order_head_.width, self.stitch_rank).to(self.device_)
#         if self.lambda_space > 0:
#             self.order_head_.space = _SpaceHead(
#                 self.output_dimension, self.head_hidden_units,
#                 self.n_groups).to(self.device_)
#             # Equal-height blocks. See `permute_blocks` for why the leftover
#             # channels are dropped from the spatial puzzle rather than padded.
#             block = int(sizes.min())
#             if block < 1:
#                 raise ValueError(
#                     f"n_groups={self.n_groups} over {channels} neurons leaves an "
#                     "empty group; the spatial puzzle needs at least one row per "
#                     "block.")
#             members = [torch.nonzero(self.group_index_ == g, as_tuple=False)
#                        .flatten()[:block] for g in range(self.n_groups)]
#             self.space_slots_ = torch.stack(members).to(self.device_)
#             self.space_block_ = block
#         # (4096, G, K) and (4096, G): the frozen-label control, group-aware.
#         # Same contract as the inherited table -- one arbitrary but FIXED answer
#         # per block of span starts, so the head can still learn it and the only
#         # thing removed is that the answer means anything.
#         table = torch.Generator(device=self.device_)
#         table.manual_seed(self.random_state + 1861)
#         self._frozen_groups = torch.rand(
#             4096, self.n_groups, self.n_tiles, device=self.device_,
#             generator=table).argsort(dim=2)
#         self._frozen_space = torch.rand(
#             4096, self.n_groups, device=self.device_, generator=table).argsort(dim=1)
#         if self.lambda_space > 0:
#             self.group_profile_ = None       # filled on the first training step

#     # -- labels ------------------------------------------------------------- #
#     def _group_labels(self, batch):
#         """(B,G,K). Independent permutations unless group_chimera is off."""
#         rows = self.n_groups if self.group_chimera else 1
#         keys = torch.rand(batch, rows, self.n_tiles, device=self.device_,
#                           generator=self._generator)
#         labels = keys.argsort(dim=2)
#         return labels if self.group_chimera else labels.expand(
#             -1, self.n_groups, -1).contiguous()

#     def _control(self, truth, starts, table):
#         if self.label_control == "true":
#             return truth
#         if self.label_control == "fresh_random":
#             keys = torch.rand(truth.shape, device=truth.device,
#                               generator=self._generator)
#             return keys.argsort(dim=-1)
#         block = self.frozen_block if self.frozen_block > 0 else self.training_span
#         row = (starts // max(block, 1)) % table.shape[0]
#         return table[row]

#     def _block_patches(self, window):
#         """(B,N,W) -> (B,G,S,W): the block sitting in each row slot, intact.

#         The 20260928 run measured its label-free baseline on this tensor
#         AVERAGED OVER THE ROWS, which was a mistake worth naming. Averaging
#         over S destroys exactly the cue the encoder is most likely to be using
#         -- "in this block neuron 7 fires twice as often as neuron 12", a
#         constant of the session that is visible to the network on every span
#         and invisible to a row-averaged template. The baseline therefore sat at
#         13-16% against a model at 85-96%, and the gap was an artefact of the
#         baseline being blindfolded. Keep the rows; let the opponent see what
#         the model sees.
#         """
#         rows = self.space_slots_.reshape(-1)
#         return window[:, rows].reshape(len(window), self.n_groups,
#                                        self.space_block_, -1)

#     def space_block_rows(self):
#         """Rows per spatial block (S), or 0 when there is no spatial puzzle.

#         Reported because every spatial number depends on it: with S=1 there is
#         no within-block structure at all, so the rate oracle has a single
#         scalar to work with and stops being informative.
#         """
#         return int(getattr(self, "space_block_", 0))

#     def _space_view(self, window, generator):
#         """Time shuffle (optional) -> block permutation -> normalization.

#         One function so that training and `evaluate_pretext` cannot drift
#         apart, which is the classic way a pretext number stops meaning
#         anything. Returns the shown window, the truth `sigma[b, s]` = the group
#         placed in row slot s, and the pre-permutation window, which is what the
#         label-free template has to be built from if it is to be an opponent
#         rather than a formality.
#         """
#         if self.space_time_shuffle:
#             window = shuffle_time(window, generator)
#         sigma = torch.rand(len(window), self.n_groups, device=window.device,
#                            generator=generator).argsort(dim=1)
#         shown = normalize_blocks(permute_blocks(window, self.space_slots_, sigma),
#                                  self.space_slots_, self.space_norm)
#         return shown, sigma, window

#     def _order_forward(self, view, batch):
#         """Encoder over the K tiles, then the grouped head. (z, logits, scores)."""
#         z = self.encoder_(view.reshape(-1, self.n_features_in_, self.window_size))
#         z = _GradScale.apply(z, self.order_grad_scale)
#         z = z.reshape(batch, self.n_tiles, self.output_dimension)
#         return (z,) + tuple(self.order_head_(z))

#     # -- one step ----------------------------------------------------------- #
#     def _step(self, data, starts):
#         """Pure cross-entropy, four puzzle terms, zero reconstruction."""
#         batch = len(starts)
#         self._steps_ += 1
#         if self.order_head_reset_every and self._steps_ % self.order_head_reset_every == 0:
#             self._reset_order_head()
#         order_scale = 1.0 + (self.order_final_scale - 1.0) * self._progress()
#         n_tiles, n_groups = self.n_tiles, self.n_groups

#         candidates = self._draw(batch)
#         jitter = self._jitter(candidates.shape[0], batch, self._valid_starts_)
#         starts, gaps = select_span(data, starts, self._valid_starts_, candidates,
#                                    jitter, self.window_size, self.span_selection)
#         tiles, next_index = _gather_tiles(data, starts, gaps, self.window_size)
#         parts, total = {}, torch.zeros((), device=self.device_)
#         zero = torch.zeros((), device=self.device_)
#         # These two exist only so the inherited fit/logging finds its keys. They
#         # are structurally zero: see the constructor, which refuses to build a
#         # model with either weight above 0.
#         for key in ("forecast", "forecast_accuracy", "reconstruct",
#                     "reconstruct_accuracy"):
#             parts[key] = zero

#         view = roll_tiles(tiles, self.order_time_roll, self._generator)
#         view = self._order_view(_augment(
#             view, min(0.95, self.neuron_dropout * self.order_augment_scale),
#             self.gain_jitter * self.order_augment_scale, self._generator))
#         labels = self._control(self._group_labels(batch), starts,
#                                self._frozen_groups)
#         view = place_groups(view, labels, self.group_index_)
#         # The permutation forward pass has to happen even on the space-only
#         # arms -- the summary table, the shortcut audit and the memorization
#         # gap are all read off it -- but its BACKWARD does not, because a term
#         # multiplied by 0.0 contributes exactly no gradient. Running it under
#         # no_grad leaves every reported number identical and drops most of the
#         # step cost, which is what makes a space ladder affordable at 6000
#         # epochs.
#         order_live = (self.lambda_order + self.lambda_pair
#                       + self.lambda_stitch) > 0
#         if order_live:
#             z, position_logits, scores = self._order_forward(view, batch)
#         else:
#             with torch.no_grad():
#                 z, position_logits, scores = self._order_forward(view, batch)

#         flat_labels = labels.reshape(-1, n_tiles)                    # (B*G, K)
#         position_each = F.cross_entropy(
#             position_logits.reshape(-1, n_tiles), labels.reshape(-1),
#             reduction="none").reshape(batch, n_groups * n_tiles).mean(1)
#         target, mask = _pair_targets(flat_labels)
#         flat_scores = scores.reshape(-1, n_tiles)
#         difference = flat_scores[:, :, None] - flat_scores[:, None, :]
#         pair_all = F.binary_cross_entropy_with_logits(difference, target,
#                                                       reduction="none")
#         pair_each = ((pair_all * mask).sum((1, 2))
#                      / mask.sum((1, 2)).clamp_min(1)).reshape(batch, n_groups).mean(1)

#         if self.lambda_stitch > 0:
#             stitch_logits = self.order_head_.stitch(self.order_head_.slices(z))
#             stitch_target = successor_targets(labels)
#             stitch_each = F.cross_entropy(
#                 stitch_logits.reshape(-1, n_tiles + 1), stitch_target.reshape(-1),
#                 reduction="none").reshape(batch, n_groups * n_tiles).mean(1)
#             with torch.no_grad():
#                 parts["stitch_accuracy"] = (
#                     stitch_logits.argmax(-1) == stitch_target).to(torch.float32).mean()
#         else:
#             stitch_each, parts["stitch_accuracy"] = torch.zeros(
#                 batch, device=self.device_), zero

#         if self.lambda_space > 0:
#             window = _augment(tiles[:, :1], self.neuron_dropout, self.gain_jitter,
#                               self._generator)[:, 0]                 # (B,N,W)
#             window, sigma, home = self._space_view(window, self._generator)
#             if self.group_profile_ is None:
#                 # The per-group template for the label-free baseline in
#                 # `evaluate_pretext`, built from `home`: the same augmentation,
#                 # the same time shuffle and the same normalization the encoder
#                 # is shown, with only the block permutation left out. Taking it
#                 # from the raw window would build a baseline out of cues that
#                 # `space_norm` has already deleted, and the model would then be
#                 # "beating" a rule it was never up against. The rows are KEPT
#                 # (see `_block_patches`): a row-averaged template cannot use
#                 # relative rates within a block, which is the single most
#                 # likely shortcut, so averaging them away would hand the model
#                 # a win it had not earned.
#                 with torch.no_grad():
#                     self.group_profile_ = self._block_patches(
#                         normalize_blocks(home, self.space_slots_,
#                                          self.space_norm)).mean(dim=0)
#             sigma = self._control(sigma, starts, self._frozen_space)
#             space_logits = self.order_head_.space(self.encoder_(window))
#             space_each = F.cross_entropy(
#                 space_logits.reshape(-1, self.n_groups), sigma.reshape(-1),
#                 reduction="none").reshape(batch, self.n_groups).mean(1)
#             with torch.no_grad():
#                 parts["space_accuracy"] = (
#                     space_logits.argmax(-1) == sigma).to(torch.float32).mean()
#         else:
#             space_each, parts["space_accuracy"] = torch.zeros(
#                 batch, device=self.device_), zero

#         combined = (self.lambda_order * position_each
#                     + self.lambda_pair * pair_each
#                     + self.lambda_stitch * stitch_each
#                     + self.lambda_space * space_each)
#         if self.order_hard_fraction < 1.0:
#             keep = max(2, int(round(self.order_hard_fraction * batch)))
#             combined = combined[combined.detach().topk(keep).indices]
#         total = total + order_scale * combined.mean()

#         if self.lambda_spread > 0:
#             # A batch statistic, so it sits OUTSIDE the per-span hard-example
#             # mining above -- there is no "hardest span" for a variance floor.
#             # Hinge at 1 rather than maximizing variance: the job is to stop a
#             # dimension from dying, not to inflate the ones that are alive.
#             deviation = z.reshape(-1, self.output_dimension).std(dim=0)
#             spread = F.relu(1.0 - deviation).pow(2).mean()
#             total = total + self.lambda_spread * spread
#             parts["spread"] = spread.detach()
#         else:
#             parts["spread"] = zero

#         parts["position"] = position_each.mean().detach()
#         parts["pair"] = pair_each.mean().detach()
#         parts["stitch"] = stitch_each.mean().detach()
#         parts["space"] = space_each.mean().detach()
#         parts["lag"] = zero
#         parts["lag_accuracy"] = zero
#         with torch.no_grad():
#             predicted = _assign(position_logits.reshape(-1, n_tiles, n_tiles)) \
#                 if self.lambda_order > 0 else _ranks(flat_scores)
#             exact, pair_accuracy, tie_rate = _order_metrics(predicted, flat_labels)
#             parts["exact"] = exact.mean()
#             parts["pair_accuracy"] = pair_accuracy.mean()
#             parts["tie_rate"] = tie_rate.mean()
#             parts["shortcut"] = shortcut_strength(tiles.sum(dim=(2, 3))).mean()
#             self._train_exact_ = float(parts["exact"])
#             self._train_pair_ = float(parts["pair_accuracy"])
#             # None and not 0.0. A switched-off term that reports 0% invites the
#             # reader to line it up against a 25% or 12.5% chance row and
#             # conclude the head is broken, when in fact it was never asked.
#             self._train_stitch_ = (float(parts["stitch_accuracy"])
#                                    if self.lambda_stitch > 0 else None)
#             self._train_space_ = (float(parts["space_accuracy"])
#                                   if self.lambda_space > 0 else None)
#         parts["total"] = total
#         self._trace(parts)
#         return parts

#     def _trace(self, parts):
#         """Live line for the terms the inherited epoch log knows nothing about.

#         The base logger prints position CE, pair BCE, forecast CE and
#         reconstruct CE. In this file the last two are structurally 0.0000 --
#         useful as proof, useless as progress -- and the stitch and space terms
#         it does not know about are the ones worth watching. Printed on the same
#         cadence as the base line, and only then, so the cost of the device sync
#         is once per `log_every` epochs.
#         """
#         if not self.verbose or self.lambda_stitch + self.lambda_space == 0:
#             return
#         per_epoch = max(1, math.ceil(getattr(self, "n_spans_", 1) / self.batch_size))
#         if self._steps_ % max(1, per_epoch * self.log_every):
#             return
#         print(f"      solo[last batch]: stitch CE={float(parts['stitch']):.4f} "
#               f"(acc {100 * float(parts['stitch_accuracy']):.1f}%, "
#               f"chance {100 / self.n_tiles:.1f}%) "
#               f"space CE={float(parts['space']):.4f} "
#               f"(acc {100 * float(parts['space_accuracy']):.1f}%, "
#               f"chance {100 / self.n_groups:.1f}%) "
#               f"spread={float(parts['spread']):.4f}", flush=True)

#     # -- evaluation --------------------------------------------------------- #
#     def evaluate_pretext(self, X, *, max_spans=1024, batch_size=256, repeats=8,
#                          random_state=200042, verbose=None, n_boot=1000,
#                          baseline_groups=4):
#         """Held-out puzzle accuracy, averaged over the G groups.

#         Same key set as `OrderJigsaw.evaluate_pretext` so the runner's tables,
#         audits and contrasts keep working, plus the solo terms. Two things are
#         done per GROUP rather than per span, and they are the two that decide
#         whether a number means anything:

#         * accuracy is the mean over groups of that group's own permutation
#           accuracy, so at G=1 it is the old number exactly;
#         * the label-free baselines -- fixed-rule level sort, level oracle,
#           tail-to-head continuity -- are computed on THAT GROUP'S ROWS ONLY
#           against THAT GROUP'S permutation. A continuity baseline measured on
#           all 38 neurons would be answering a different question from the one
#           the group had to answer, and would flatter the model.

#         `baseline_groups` caps how many groups the K! chain enumeration runs
#         over, because that is the one expensive baseline; the accuracy itself
#         always uses every group.
#         """
#         self._check_fitted()
#         verbose = self.verbose if verbose is None else verbose
#         repeats = _integer("repeats", repeats, minimum=1)
#         n_tiles, n_groups = self.n_tiles, self.n_groups
#         data, starts, _ = self._spans(X, self.n_features_in_)
#         if len(starts) == 0:
#             raise ValueError("No valid spans in X.")
#         rng = np.random.default_rng(random_state)
#         chosen = starts if len(starts) <= max_spans else np.sort(
#             rng.choice(starts, max_spans, replace=False))
#         chosen = torch.from_numpy(np.ascontiguousarray(chosen)).to(self.device_)
#         local_starts = torch.from_numpy(np.ascontiguousarray(starts)).to(self.device_)
#         generator = torch.Generator(device=self.device_).manual_seed(random_state)
#         low, high = self.tile_gap
#         pool = 1 if self.span_selection == "random" else self.selection_pool
#         shown = min(baseline_groups, n_groups)
#         keys = ("exact", "pair", "tie", "mean_exact", "mean_pair", "level_oracle",
#                 "continuity", "stitch", "space", "space_oracle",
#                 "space_oracle_rate", "space_oracle_level")
#         per_span = {k: torch.zeros(len(chosen), device=self.device_) for k in keys}
#         position_ce = 0.0
#         self.encoder_.eval()
#         self.order_head_.eval()
#         with torch.inference_mode():
#             for _ in range(repeats):
#                 for begin in range(0, len(chosen), batch_size):
#                     block = chosen[begin:begin + batch_size]
#                     stop, size = begin + len(block), len(block)
#                     candidates = torch.randint(
#                         low, high + 1, (pool, size, n_tiles - 1),
#                         device=self.device_, generator=generator)
#                     jitter = self._jitter(pool, size, local_starts, generator)
#                     block, gaps = select_span(data, block, local_starts, candidates,
#                                               jitter, self.window_size,
#                                               self.span_selection)
#                     tiles, _ = _gather_tiles(data, block, gaps, self.window_size)
#                     keys_g = torch.rand(size,
#                                         n_groups if self.group_chimera else 1,
#                                         n_tiles, device=self.device_,
#                                         generator=generator)
#                     labels = keys_g.argsort(dim=2)
#                     if not self.group_chimera:
#                         labels = labels.expand(-1, n_groups, -1).contiguous()
#                     # `_order_view` rather than a hand-rolled normalization, so
#                     # `count_match` and `raw` arms are evaluated the way they were
#                     # trained. Normalizing before the chimera gather is identical
#                     # to after it: the statistic is per (span, tile, neuron) and
#                     # the gather only permutes the tile index within a neuron.
#                     view = place_groups(self._order_view(tiles, generator),
#                                         labels, self.group_index_)
#                     shown_tiles = place_groups(tiles, labels, self.group_index_)
#                     z = self.encoder_(view.reshape(-1, self.n_features_in_,
#                                                    self.window_size))
#                     z = z.reshape(size, n_tiles, self.output_dimension)
#                     position_logits, scores = self.order_head_(z)
#                     flat_labels = labels.reshape(-1, n_tiles)
#                     predicted = _assign(position_logits.reshape(-1, n_tiles, n_tiles)) \
#                         if self.lambda_order > 0 else _ranks(scores.reshape(-1, n_tiles))
#                     exact, pair, tie = _order_metrics(predicted, flat_labels)
#                     for name, value in (("exact", exact), ("pair", pair), ("tie", tie)):
#                         per_span[name][begin:stop] += value.reshape(
#                             size, n_groups).mean(1).to(torch.float32) / repeats

#                     level = torch.zeros(size, device=self.device_)
#                     level_exact = torch.zeros(size, device=self.device_)
#                     continuity = torch.zeros(size, device=self.device_)
#                     for group in range(shown):
#                         rows = torch.nonzero(self.group_index_ == group,
#                                              as_tuple=False).flatten()
#                         piece = shown_tiles[:, :, rows, :]
#                         truth = labels[:, group, :]
#                         g_exact, g_pair, _ = _order_metrics(
#                             _ranks(piece.mean(dim=(2, 3))), truth)
#                         level = level + g_pair / shown
#                         level_exact = level_exact + g_exact / shown
#                         continuity = continuity + continuity_strength(piece, truth) / shown
#                     per_span["mean_pair"][begin:stop] += level / repeats
#                     per_span["mean_exact"][begin:stop] += level_exact / repeats
#                     per_span["level_oracle"][begin:stop] += torch.maximum(
#                         level, 1 - level) / repeats
#                     per_span["continuity"][begin:stop] += continuity / repeats

#                     if self.lambda_stitch > 0:
#                         logits = self.order_head_.stitch(self.order_head_.slices(z))
#                         hit = (logits.argmax(-1) == successor_targets(labels))
#                         per_span["stitch"][begin:stop] += hit.reshape(
#                             size, -1).to(torch.float32).mean(1) / repeats
#                     if self.lambda_space > 0:
#                         shown_window, sigma, _ = self._space_view(tiles[:, 0],
#                                                                  generator)
#                         guess = self.order_head_.space(
#                             self.encoder_(shown_window)).argmax(-1)
#                         per_span["space"][begin:stop] += (guess == sigma).to(
#                             torch.float32).mean(1) / repeats
#                         # THE label-free opponent, in three strengths, on the
#                         # SAME view the encoder got. Name each block by matching
#                         # it to that group's template:
#                         #   space_oracle       the whole (S,W) patch. The
#                         #                      strongest rule, and the one the
#                         #                      model has to beat to be
#                         #                      interesting.
#                         #   ..._rate           each row's mean over time only.
#                         #                      Pure static neuron identity:
#                         #                      "this block is the one where the
#                         #                      second row outfires the first".
#                         #                      A session constant, so if THIS is
#                         #                      already high the puzzle is not
#                         #                      about the current span at all.
#                         #   ..._level          the row-AVERAGED profile. This is
#                         #                      the only one the 20260928 run
#                         #                      reported, and it is the weakest of
#                         #                      the three by construction.
#                         # `group_profile_` is a training-time statistic and is
#                         # not in the state_dict, so a reloaded model reports None
#                         # here rather than a number built from the wrong thing.
#                         profile = getattr(self, "group_profile_", None)
#                         if profile is None:
#                             for name in ("space_oracle", "space_oracle_rate",
#                                          "space_oracle_level"):
#                                 per_span[name][begin:stop] = float("nan")
#                         else:
#                             patch = self._block_patches(shown_window)  # (B,G,S,W)
#                             views = (                 # profile is (G,S,W)
#                                 ("space_oracle",
#                                  patch.reshape(size, n_groups, -1),
#                                  profile.reshape(n_groups, -1)),
#                                 ("space_oracle_rate",          # over time -> (.,S)
#                                  patch.mean(dim=3), profile.mean(dim=2)),
#                                 ("space_oracle_level",         # over rows -> (.,W)
#                                  patch.mean(dim=2), profile.mean(dim=1)))
#                             for name, candidate, template in views:
#                                 hit = nearest_template(candidate, template) == sigma
#                                 per_span[name][begin:stop] += hit.to(
#                                     torch.float32).mean(1) / repeats
#                     position_ce += float(F.cross_entropy(
#                         position_logits.reshape(-1, n_tiles),
#                         labels.reshape(-1))) * size
#         self.encoder_.train()
#         self.order_head_.train()
#         mean = {k: float(v.mean()) for k, v in per_span.items()}

#         def percent(key, enabled=True):
#             """100x, or None when the number would be a lie.

#             Covers both "the term is switched off" and "the statistic could not
#             be computed" (K > 6 for the chain baselines, a reloaded model with no
#             training-time group profile). Reporting 0.0 for those would put a
#             number in the table that a reader would compare against chance.
#             """
#             value = mean[key]
#             if not enabled or not math.isfinite(value) or value < 0:
#                 return None
#             return 100 * value

#         boot = 0 if getattr(self, "_fitting_", False) else n_boot
#         low_ci, high_ci, spread, n_effective = block_bootstrap(
#             per_span["pair"].cpu().numpy(), self.training_span, n_boot=boot,
#             random_state=random_state)
#         result = dict(
#             exact_accuracy_percent=100 * mean["exact"],
#             pair_accuracy_percent=100 * mean["pair"],
#             pair_ci95_percent=[100 * low_ci, 100 * high_ci],
#             pair_block_resolution_percent=100 * spread,
#             n_effective=n_effective,
#             tie_rate_percent=100 * mean["tie"],
#             argmax_pair_accuracy_percent=100 * mean["pair"],
#             argmax_tie_rate_percent=100 * mean["tie"],
#             rank_exact_accuracy_percent=100 * mean["exact"],
#             rank_pair_accuracy_percent=100 * mean["pair"],
#             lag_accuracy_percent=0.0,
#             chance_lag_percent=100.0 / self.lag_classes,
#             baseline_mean_sort_exact_percent=100 * mean["mean_exact"],
#             baseline_mean_sort_pair_percent=100 * mean["mean_pair"],
#             baseline_norm_sort_exact_percent=100 * mean["mean_exact"],
#             baseline_norm_sort_pair_percent=100 * mean["mean_pair"],
#             baseline_level_oracle_pair_percent=100 * mean["level_oracle"],
#             baseline_continuity_pair_percent=percent("continuity"),
#             continuity_gap_bins=int(self.tile_gap[0]),
#             baseline_groups=int(shown),
#             train_pair_accuracy_percent=(None if not hasattr(self, "_train_pair_")
#                                          else 100 * self._train_pair_),
#             train_exact_accuracy_percent=(None if not hasattr(self, "_train_exact_")
#                                           else 100 * self._train_exact_),
#             memorization_gap_percent=(None if not hasattr(self, "_train_pair_")
#                                       else 100 * (self._train_pair_ - mean["pair"])),
#             # -- the solo terms ------------------------------------------- #
#             n_groups=int(n_groups),
#             group_chimera=bool(self.group_chimera),
#             group_subspace=bool(self.group_subspace),
#             group_sizes=list(getattr(self, "group_sizes_", [])),
#             label_bits_per_span=float(n_groups * math.lgamma(n_tiles + 1)),
#             stitch_accuracy_percent=percent("stitch", self.lambda_stitch > 0),
#             # The best LABEL-FREE constant rule ("always answer 'I am last'", or
#             # always answer some fixed slot) scores 1/K, not 1/(K+1): the self
#             # logit is masked, so there are only K live options. Quoting the
#             # smaller number would make a useless head look educated.
#             chance_stitch_percent=100.0 / n_tiles,
#             train_stitch_accuracy_percent=(
#                 None if getattr(self, "_train_stitch_", None) is None
#                 else 100 * self._train_stitch_),
#             space_accuracy_percent=percent("space", self.lambda_space > 0),
#             space_oracle_percent=percent("space_oracle", self.lambda_space > 0),
#             space_oracle_rate_percent=percent("space_oracle_rate",
#                                               self.lambda_space > 0
#                                               and self.space_block_rows() > 1),
#             space_oracle_level_percent=percent("space_oracle_level",
#                                                self.lambda_space > 0),
#             # None at G=1: "chance is 100%" is arithmetically true and reads as
#             # a result. There is no spatial permutation of one block.
#             chance_space_percent=(100.0 / n_groups if n_groups > 1 else None),
#             space_norm=self.space_norm,
#             space_time_shuffle=bool(self.space_time_shuffle),
#             space_block_rows=self.space_block_rows(),
#             train_space_accuracy_percent=(
#                 None if getattr(self, "_train_space_", None) is None
#                 else 100 * self._train_space_),
#             position_cross_entropy=position_ce / max(repeats * len(chosen), 1),
#             uniform_cross_entropy=math.log(n_tiles),
#             decode="assignment" if self.lambda_order > 0 else "score_rank",
#             order_view=self.order_view, span_selection=self.span_selection,
#             label_control=self.label_control,
#             chance_exact_percent=100.0 / math.factorial(n_tiles),
#             chance_pair_percent=50.0,
#             n_spans=int(len(chosen)), n_draws=int(repeats * len(chosen)),
#             repeats=int(repeats))
#         if verbose:
#             def show(key, absent="off"):
#                 value = result[key]
#                 return absent if value is None else f"{value:.2f}%"
#             print(f"  pretext[G={n_groups}]: "
#                   f"exact={result['exact_accuracy_percent']:.2f}% "
#                   f"(chance {result['chance_exact_percent']:.2f}%) "
#                   f"pair={result['pair_accuracy_percent']:.2f}% "
#                   f"[{result['pair_ci95_percent'][0]:.2f}, "
#                   f"{result['pair_ci95_percent'][1]:.2f}] n_eff={n_effective} "
#                   f"| level-oracle={result['baseline_level_oracle_pair_percent']:.2f}% "
#                   f"continuity={show('baseline_continuity_pair_percent', 'n/a')} "
#                   f"| stitch={show('stitch_accuracy_percent')} "
#                   f"(chance {result['chance_stitch_percent']:.2f}%)", flush=True)
#             # The spatial puzzle gets its own line because it needs three
#             # baselines next to it to mean anything, and a one-line version
#             # would push them off the right-hand side of the log.
#             if result["space_accuracy_percent"] is not None:
#                 print(f"      space[{self.space_norm}"
#                       f"{', time-shuffled' if self.space_time_shuffle else ''}, "
#                       f"S={result['space_block_rows']}]: "
#                       f"{show('space_accuracy_percent')} "
#                       f"(chance {show('chance_space_percent', 'n/a')}) "
#                       f"vs template oracles -- patch "
#                       f"{show('space_oracle_percent', 'n/a')}, rate "
#                       f"{show('space_oracle_rate_percent', 'n/a')}, level "
#                       f"{show('space_oracle_level_percent', 'n/a')}", flush=True)
#         return result


# # --------------------------------------------------------------------------- #
# # 4. arms
# # --------------------------------------------------------------------------- #
# #  Read this table as three blocks:
# #    the LADDER   -- rank 1, 4, 8, 16. One variable, and the prediction the
# #                    whole file rests on: R2 and participation ratio rise with G.
# #    the CONTROLS -- shared permutations, no subspaces, frozen labels, frozen
# #                    encoder. Each removes exactly one thing the ladder claims
# #                    is load-bearing.
# #    the TERMS    -- stitch and space on their own and left out, so the headline
# #                    arm can be decomposed rather than admired.
# _SOLO = dict(_model="solo")
# _LADDER = dict(_SOLO, lambda_stitch=1.0, lambda_space=0.5, space_norm="block_mean")


# def _space(groups, norm, **extra):
#     """A SPACE-ONLY arm: permutation terms off, neuron-axis puzzle on.

#     With `lambda_order`, `lambda_pair` and `lambda_stitch` all zero the model
#     still runs the K-tile forward pass (the tables need it) but skips its
#     backward, so these arms are the cheap ones despite looking like the others.
#     """
#     return dict(_SOLO, n_groups=groups, lambda_order=0.0, lambda_pair=0.0,
#                 lambda_space=1.0, space_norm=norm, **extra)


# SOLO_ARMS = {
#     # -- the ladder ------------------------------------------------------- #
#     # G=1 is the classic jigsaw with the anchor deleted: same head, same
#     # parameter count, same loss. Historically this is `order_only`, which
#     # scored R2 0.005 at participation ratio 3.4. If the rank story is right
#     # this arm reproduces that collapse and the rest of the ladder escapes it.
#     "solo_rank1": dict(_SOLO, n_groups=1),
#     "solo_rank4": dict(_SOLO, n_groups=4),
#     "solo_rank8": dict(_SOLO, n_groups=8),
#     "solo_rank16": dict(_SOLO, n_groups=16),
#     # -- the controls for the ladder --------------------------------------- #
#     # THE control. Eight groups, eight subspaces, eight readouts, identical
#     # parameter count and identical number of CE terms -- and one single
#     # permutation copied to all of them. The ONLY thing removed is the
#     # independence of the labels, which is the only thing the rank argument
#     # claims to need. If `solo_rank8 - solo_shared8` is zero, the idea is wrong.
#     "solo_shared8": dict(_SOLO, n_groups=8, group_chimera=False),
#     # Independent labels but no structural subspace: rank is encouraged by the
#     # task and not enforced by the wiring. Separates "the puzzle needs rank"
#     # from "the architecture was handed rank".
#     "solo_wide8": dict(_SOLO, n_groups=8, group_subspace=False),
#     "solo_frozen8": dict(_SOLO, n_groups=8, label_control="frozen_random"),
#     "solo_random": dict(_SOLO, n_groups=8, max_epochs=0),
#     # -- the full model and its term ablations ------------------------------ #
#     "solo_full": dict(_LADDER, n_groups=8),
#     "solo_full_frozen": dict(_LADDER, n_groups=8, label_control="frozen_random"),
#     "solo_full_shared": dict(_LADDER, n_groups=8, group_chimera=False),
#     "solo_no_stitch": dict(_LADDER, n_groups=8, lambda_stitch=0.0),
#     "solo_no_space": dict(_LADDER, n_groups=8, lambda_space=0.0),
#     "solo_no_subspace": dict(_LADDER, n_groups=8, group_subspace=False),
#     # Each term completely alone, with the permutation CE switched off, so the
#     # decomposition adds up instead of being asserted.
#     "solo_stitch_only": dict(_SOLO, n_groups=8, lambda_order=0.0, lambda_pair=0.0,
#                              lambda_stitch=1.0),
#     # -- the spatial puzzle, which is the part that actually worked ---------- #
#     # 20260928, three seeds. Read against the FROZEN random encoder at R2
#     # 0.3199, not against zero:
#     #     solo_rank8    0.3186   the grouped TEMPORAL puzzle, alone -> a dead
#     #                            heat with an untrained encoder.
#     #     solo_no_space 0.2538   temporal puzzle + stitch, no spatial term ->
#     #                            BELOW the frozen encoder. Actively harmful.
#     #     solo_no_stitch 0.4158  temporal puzzle + spatial term.
#     #     solo_full      0.4345  everything.
#     # The entire result is the neuron-axis term, so from here it is the
#     # experiment and the temporal ladder is the control. The open question is
#     # NOT whether it helps -- it plainly does -- but whether it is solvable from
#     # static neuron identity, which is a session constant the encoder can
#     # memorize once, or from this span's dynamics. That is what space_norm and
#     # space_time_shuffle are for, and what the three template oracles measure.
#     "solo_space_only":   _space(8, "block_mean"),
#     # The rate cue left fully in place. If this MATCHES solo_space_only then
#     # block_mean was removing nothing the model was using.
#     "solo_space_raw":    _space(8, "none"),
#     # Each neuron's own mean over the window removed, then its gain too. Under
#     # row_zscore no neuron's firing rate can name its block: what survives is
#     # the temporal shape of each row, which is a fact about this span. If the
#     # spatial puzzle still transfers here, the claim is about dynamics.
#     "solo_space_rowm":   _space(8, "row_mean"),
#     "solo_space_rowz":   _space(8, "row_zscore"),
#     # Same static statistics, temporal order destroyed. Scores identically to
#     # solo_space_only if and only if the task never needed time.
#     "solo_space_tshuf":  _space(8, "block_mean", space_time_shuffle=True),
#     # The label control for the spatial term on its own, matching what
#     # solo_full_frozen did for the whole model (train 88-93%, held out 12.3-12.7%
#     # -- exactly chance -- and R2 0.1592, well BELOW the frozen encoder).
#     "solo_space_frozen": _space(8, "block_mean", label_control="frozen_random"),
#     # The ladder on the axis that works. Not a one-variable contrast and the
#     # comment should say so: raising G raises the class count AND shrinks the
#     # block to 38//G rows, so a harder question is being asked of less evidence.
#     "solo_space_g4":     _space(4, "block_mean"),
#     "solo_space_g16":    _space(16, "block_mean"),
#     # -- the regularizer, kept separate on purpose -------------------------- #
#     # Not a puzzle. Here so that "the embedding did not collapse" can be
#     # attributed to the task rather than to a variance floor -- if
#     # solo_rank8 and solo_spread land in the same place, the floor is doing
#     # nothing and the puzzle earned it.
#     "solo_spread": dict(_SOLO, n_groups=8, lambda_spread=1.0),
#     "solo_rank1_spread": dict(_SOLO, n_groups=1, lambda_spread=1.0),
#     # -- hardened geometry --------------------------------------------------- #
#     # Gap floor 3 so tiles never touch, short span so the drift is real. This is
#     # the geometry where `order_gapped` was the only arm in the 20260927 run to
#     # genuinely learn the pretext (54.67% against a 51.21% continuity baseline,
#     # z=20) -- and had the worst R2 of any trained arm, 0.3840. Worth repeating
#     # with a rank-8 puzzle: the question is whether learning a HIGH-RANK version
#     # of the same task transfers where the rank-1 version did not.
#     "solo_gapped8": dict(_LADDER, n_groups=8, window_size=6, tile_gap=(3, 5)),
#     "solo_gapped1": dict(_LADDER, n_groups=1, window_size=6, tile_gap=(3, 5),
#                          lambda_space=0.0),
#     # Hard-example mining plus periodic head resets, the two counters to the
#     # gradient-death pattern measured on the anchored runs (order CE 0.0153 at
#     # epoch 1000 falling to 0.0006 by 10000, i.e. the term switching itself off).
#     "solo_hard8": dict(_LADDER, n_groups=8, order_hard_fraction=0.25,
#                        order_head_reset_every=400),
# }

# # The autoencoder this project has been leaning on since the start, run through
# # the SAME class family and the same defaults so "the jigsaw alone" has an
# # honest opponent in the same table. It is not a SoloJigsaw -- that class refuses
# # a reconstruction weight on purpose -- so it is registered under its own model
# # key and kept out of SOLO_ARMS, whose every entry must be constructible as a
# # SoloJigsaw. The 20260928 run listed `reconstruct_only` in its defaults and it
# # silently never ran, so the one contrast the project is actually about was
# # missing from the output. This is the fix.
# SOLO_BASELINE_ARMS = {
#     "ae_matched": dict(_model="solo_ae", lambda_order=0.0, lambda_pair=0.0,
#                        lambda_lag=0.0, lambda_reconstruct=1.0),
# }

# # `group_chimera`, the four lambdas, `space_norm`, `space_time_shuffle` and
# # `lambda_spread` are deliberately absent: those are the variables under test,
# # exactly as the order weights are absent from ANCHOR_MATCH_KEYS. A
# # `space_norm` entry here (v1 had one) made the runner shout "NOT ONE VARIABLE"
# # at the rowz and raw contrasts, which are the whole point of this round.
# # `n_groups` IS here, because changing it changes the subspace width and the
# # block height and therefore the readout capacity -- a rank-1-versus-rank-8
# # comparison is not a one-variable contrast and the runner should say so.
# SOLO_MATCH_KEYS = ANCHOR_MATCH_KEYS + ("n_groups", "group_subspace", "group_seed",
#                                        "stitch_rank")


# SOLO_CONTRASTS = (
#     ("solo_rank8", "solo_shared8", "RANK effect (independent puzzles)",
#      "identical architecture, identical parameter count, identical number of CE "
#      "terms. The only difference is whether the G permutations are independent. "
#      "This is the contrast the whole file stands or falls on."),
#     ("solo_full", "solo_full_shared", "RANK effect (full model)",
#      "the same test with the stitch and space terms switched on."),
#     ("solo_rank8", "solo_rank1", "GROUPS effect",
#      "rank 8 against the classic jigsaw with the anchor removed. Not a "
#      "one-variable contrast -- the subspace width changes too -- so read it "
#      "next to the shared-permutation control, not instead of it."),
#     ("solo_full", "solo_full_frozen", "LABEL effect",
#      "same loss, same heads, same gradient path, labels that mean nothing about "
#      "time or identity. On the anchored runs this was ~0 and the frozen arm "
#      "often WON, which is what sent us here."),
#     ("solo_full", "solo_no_stitch", "STITCH effect",
#      "what the successor CE adds. Its head is bilinear, so unlike the slot head "
#      "it has no per-span capacity to memorize with."),
#     ("solo_full", "solo_no_space", "SPACE effect",
#      "what the neuron-axis puzzle adds."),
#     ("solo_full", "solo_no_subspace", "SUBSPACE effect",
#      "structural rank floor against a merely encouraged one."),
#     ("solo_rank8", "solo_spread", "what a variance floor adds on top",
#      "if this is large, the puzzle is not preventing collapse on its own and the "
#      "honest description of the result includes a regularizer."),
#     # -- the spatial round -------------------------------------------------- #
#     ("solo_space_only", "ae_matched", "JIGSAW ALONE vs THE AUTOENCODER",
#      "the number the project is actually about. ae_matched is the arm that "
#      "carried every previous result; the spatial jigsaw never sees a "
#      "reconstruction target. Positive means the puzzle stands on its own."),
#     ("solo_space_only", "solo_random", "SPACE ALONE vs FROZEN ENCODER",
#      "the floor that matters. On 20260928 the temporal puzzle failed exactly "
#      "this test -- solo_rank8 0.3186 against solo_random 0.3199 -- so an arm "
#      "that does not clear it has not earned the word 'learned'."),
#     ("solo_space_only", "solo_space_frozen", "SPACE LABEL effect",
#      "same loss, same head, same gradient path, block identities that mean "
#      "nothing. This is what makes the spatial number evidence rather than a "
#      "side effect of training on something."),
#     ("solo_space_only", "solo_space_rowz", "what the STATIC RATE cue was worth",
#      "row_zscore deletes every neuron's own mean and gain inside the window, so "
#      "no session-constant firing rate can name a block. Near zero means the "
#      "spatial puzzle is about this span's dynamics. Large and positive means it "
#      "was largely 'memorize which cell is which', which is still a real "
#      "objective but a much smaller claim."),
#     ("solo_space_only", "solo_space_tshuf", "what TEMPORAL ORDER was worth to it",
#      "the time bins are permuted, so every static statistic survives and only "
#      "the order of the window is destroyed. Zero here plus zero above would "
#      "mean the task needs neither identity nor time, which would be a bug, not "
#      "a finding."),
#     ("solo_space_g16", "solo_space_only", "SPACE GROUPS effect",
#      "G=16 asks a 16-way question of 2-row blocks where G=8 asks an 8-way "
#      "question of 4-row blocks. Two variables, so read it as a dose curve and "
#      "not as a contrast."),
#     ("solo_rank8", "solo_random", "TEMPORAL PUZZLE vs FROZEN ENCODER",
#      "the 20260928 null, kept in the table so it is re-measured rather than "
#      "remembered: +0.0 means sorting time-tiles is worth nothing on its own."),
#     ("solo_full", "solo_random", "vs FROZEN ENCODER",
#      "what training is worth at all. Must be positive or nothing else counts."),
# )

# # The spatial puzzle is the experiment now and the temporal ladder is the
# # control, so these ten are chosen to answer one question -- is the neuron-axis
# # jigsaw reading dynamics or memorizing neuron identity? -- while keeping
# # solo_full and solo_rank8 in the table so the 20260928 numbers can be lined up
# # against the new ones on the same seeds.
# DEFAULT_SOLO_ARMS = ("solo_space_only", "solo_space_rowz", "solo_space_tshuf",
#                      "solo_space_frozen", "solo_space_g16", "solo_no_stitch",
#                      "solo_rank8", "solo_random", "solo_full", "ae_matched")


# def register(models, arms, contrasts=None):
#     """Wire this module into run_jigsaw.py without editing its tables by hand."""
#     models["solo"] = SoloJigsaw
#     models["solo_ae"] = OrderJigsaw
#     for table in (SOLO_ARMS, SOLO_BASELINE_ARMS):
#         for name, settings in table.items():
#             arms[name] = dict(settings)
#     if contrasts is not None:
#         for row in SOLO_CONTRASTS:
#             if row not in contrasts:
#                 contrasts.append(row)
#     missing = [name for name in DEFAULT_SOLO_ARMS if name not in arms]
#     if missing:
#         # Loud on purpose. On 20260928 `reconstruct_only` sat in the default
#         # list, resolved to nothing, and the run finished without the one
#         # comparison it existed to make -- with no warning anywhere in the log.
#         print(f"  WARNING: default solo arms not registered: {missing}. They "
#               f"will be skipped and any contrast that names them will be "
#               f"missing from the output.", flush=True)
#     return models, arms


# # --------------------------------------------------------------------------- #
# # 5. self-test
# # --------------------------------------------------------------------------- #
# def _check(condition, message):
#     print(f"  {'ok  ' if condition else 'FAIL'}  {message}")
#     if not condition:
#         raise AssertionError(message)


# def _counts(n=2600, channels=12, seed=0):
#     """Drifting Poisson counts: a slow level ramp plus a fast oscillation.

#     The ramp is what the level shortcut reads; the oscillation is what an
#     honest solution has to read. Both are present so the baselines in
#     `evaluate_pretext` have something to detect.
#     """
#     rng = np.random.default_rng(seed)
#     time = np.arange(n)[:, None]
#     phase = rng.uniform(0, 2 * np.pi, size=(1, channels))
#     rate = (0.6 + 0.8 * time / n
#             + 0.5 * np.sin(2 * np.pi * time / 37 + phase))
#     return rng.poisson(np.clip(rate, 0.05, None)).astype(np.float32)


# def _smoke_test():
#     print("jigsaw_solo self-test")
#     torch.manual_seed(0)
#     counts = _counts()
#     channels = counts.shape[1]

#     # 1. the guarantee
#     for weight in ("lambda_reconstruct", "lambda_forecast"):
#         try:
#             SoloJigsaw(**{weight: 1.0})
#             raised = False
#         except ValueError:
#             raised = True
#         _check(raised, f"{weight}>0 is refused by the constructor")

#     # 2. divisibility and the other constructor guards
#     for kwargs in (dict(n_groups=7, output_dimension=64),
#                    dict(n_groups=1, lambda_space=1.0),
#                    dict(n_groups=4, lambda_lag=1.0),
#                    dict(shuffle_tiles=False)):
#         try:
#             SoloJigsaw(**kwargs)
#             raised = False
#         except ValueError:
#             raised = True
#         _check(raised, f"rejected: {kwargs}")
#     _check(SoloJigsaw(n_groups=8, lambda_order=0.0, lambda_pair=0.0,
#                       lambda_space=1.0).max_epochs == 60,
#            "a space-only arm survives the all-zero-weights guard")

#     # 3. successor targets, by hand
#     labels = torch.tensor([[[2, 0, 1, 3]]])          # slot0 holds chunk 2, ...
#     #   chunk 0 sits in slot 1, chunk 1 in slot 2, chunk 2 in slot 0, chunk 3 in 3
#     #   slot 0 holds chunk 2 -> successor is the slot holding chunk 3 = slot 3
#     #   slot 1 holds chunk 0 -> slot holding chunk 1 = slot 2
#     #   slot 2 holds chunk 1 -> slot holding chunk 2 = slot 0
#     #   slot 3 holds chunk 3 -> last -> class K = 4
#     _check(successor_targets(labels).tolist() == [[[3, 2, 0, 4]]],
#            "successor_targets matches a hand-worked permutation")
#     big = torch.rand(32, 3, 5).argsort(-1)
#     target = successor_targets(big)
#     ok = True
#     for b in range(32):
#         for g in range(3):
#             for slot in range(5):
#                 chunk = int(big[b, g, slot])
#                 if chunk == 4:
#                     ok &= int(target[b, g, slot]) == 5
#                 else:
#                     ok &= int(big[b, g, int(target[b, g, slot])]) == chunk + 1
#     _check(ok, "successor_targets is the inverse-permutation shift at K=5")

#     # 4. the chimera really is per-group
#     tiles = torch.arange(2 * 4 * 6 * 3, dtype=torch.float32).reshape(2, 4, 6, 3)
#     index = torch.tensor([0, 0, 0, 1, 1, 1])
#     labels = torch.tensor([[[1, 0, 3, 2], [3, 2, 1, 0]],
#                            [[0, 1, 2, 3], [2, 3, 0, 1]]])
#     placed = place_groups(tiles, labels, index)
#     ok = all(torch.equal(placed[b, j, n], tiles[b, int(labels[b, int(index[n]), j]), n])
#              for b in range(2) for j in range(4) for n in range(6))
#     _check(ok, "place_groups pulls row n of slot j from chunk labels[b,group[n],j]")
#     same = labels[:, :1].expand(-1, 2, -1).contiguous()
#     shared = place_groups(tiles, same, index)
#     direct = torch.gather(tiles, 1, same[:, 0][:, :, None, None].expand(
#         -1, -1, 6, 3))
#     _check(torch.equal(shared, direct),
#            "with one shared permutation the chimera IS the ordinary tile shuffle")

#     # 5. block permutation is a permutation, and the leftovers stay put
#     window = torch.randn(5, 11, 4)
#     slots = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
#     sigma = torch.rand(5, 3).argsort(1)
#     moved = permute_blocks(window, slots, sigma)
#     _check(torch.equal(moved[:, 9:], window[:, 9:]),
#            "channels outside the equal-height blocks are untouched")
#     ok = all(torch.equal(moved[b, slots[s]], window[b, slots[int(sigma[b, s])]])
#              for b in range(5) for s in range(3))
#     _check(ok, "row slot s receives the block of group sigma[b,s]")
#     _check(torch.allclose(moved.sum(1), window.sum(1)),
#            "permuting blocks conserves the total activity")
#     zeroed = normalize_blocks(moved, slots, "block_mean")
#     means = zeroed[:, slots.reshape(-1)].reshape(5, 3, -1).mean(2)
#     _check(torch.allclose(means, torch.zeros_like(means), atol=1e-5),
#            "block_mean leaves every block at zero mean (the rate cue is gone)")
#     # block_mean does NOT remove the cue the model is most likely to be using.
#     # Spell it out, because the 20260928 baseline was built as if it did.
#     rows = slots.reshape(-1)
#     per_row = zeroed[:, rows].reshape(5, 3, 3, -1).mean(3)
#     _check(float(per_row.std()) > 1e-3,
#            "block_mean leaves per-NEURON rates inside the block intact -- this "
#            "is the static-identity shortcut, and it is why the row_* modes exist")
#     for mode in ("row_mean", "row_zscore"):
#         stripped = normalize_blocks(moved, slots, mode)
#         row_means = stripped[:, rows].mean(dim=2)
#         _check(torch.allclose(row_means, torch.zeros_like(row_means), atol=1e-5),
#                f"{mode} puts EVERY row at zero mean over the window")
#         _check(torch.equal(stripped[:, 9:], moved[:, 9:]),
#                f"{mode} touches only the slotted channels")
#     scaled = normalize_blocks(moved * 7.0, slots, "row_zscore")
#     plain = normalize_blocks(moved, slots, "row_zscore")
#     _check(torch.allclose(scaled[:, rows], plain[:, rows], atol=1e-3),
#            "row_zscore is invariant to a global gain, so no neuron's gain can "
#            "name its block either")
#     silent = moved.clone()
#     silent[:, 0] = 0.0
#     _check(float(normalize_blocks(silent, slots, "row_zscore")[:, 0].abs().max()) == 0.0,
#            "a row that never fires becomes exactly zero rather than amplified noise")

#     # 5b. the time shuffle keeps every static statistic and destroys order
#     generator = torch.Generator().manual_seed(5)
#     rolled = shuffle_time(window, generator)
#     _check(torch.allclose(rolled.sum(2), window.sum(2), atol=1e-5),
#            "shuffle_time conserves each neuron's total, so rates are untouched")
#     _check(all(torch.allclose(rolled[b].sum(0).sort().values,
#                               window[b].sum(0).sort().values, atol=1e-5)
#                for b in range(5)),
#            "the population vectors are the same multiset, only reordered")
#     # One permutation per SAMPLE, shared by every row. Tested on a constant ramp
#     # because on real values each row's sort order is its own and proves nothing.
#     ramp = torch.arange(4, dtype=torch.float32).expand(5, 11, 4).contiguous()
#     mixed = shuffle_time(ramp, torch.Generator().manual_seed(6))
#     _check(all(torch.equal(mixed[b, 0], mixed[b, j]) for b in range(5) for j in range(11)),
#            "one permutation per sample, shared by every row")
#     _check(not torch.equal(mixed[0, 0], mixed[1, 0]),
#            "and a different one for the next sample")
#     _check(not torch.equal(rolled, window), "and the order really did change")

#     # 5c. the template oracle: a perfect template must win
#     template = torch.randn(4, 6)
#     order = torch.rand(7, 4).argsort(1)
#     _check(torch.equal(nearest_template(template[order], template), order),
#            "nearest_template recovers the permutation from exact templates")
#     _check(torch.equal(nearest_template(3.5 * template[order], template), order),
#            "and is invariant to a global scale, which is why it is cosine")

#     # 6. balanced random partition
#     index = group_partition(38, 8, 7)
#     sizes = torch.bincount(index, minlength=8)
#     _check(int(sizes.max() - sizes.min()) <= 1 and int(sizes.sum()) == 38,
#            f"group_partition balances 38 neurons into 8 groups {sizes.tolist()}")
#     _check(not torch.equal(index, group_partition(38, 8, 8)),
#            "a different group_seed gives a different partition")

#     # 7. the grouped head: shapes, and G=1 equals the classic head
#     head = _GroupOrderHead(64, 32, 4, 8, True)
#     z = torch.randn(9, 4, 64)
#     position, score = head(z)
#     _check(tuple(position.shape) == (9, 8, 4, 4) and tuple(score.shape) == (9, 8, 4),
#            "subspace head emits (B,G,K,K) and (B,G,K)")
#     wide = _GroupOrderHead(64, 32, 4, 8, False)
#     position, score = wide(z)
#     _check(tuple(position.shape) == (9, 8, 4, 4) and tuple(score.shape) == (9, 8, 4),
#            "non-subspace head emits the same shapes")
#     _check(sum(p.numel() for p in _GroupOrderHead(64, 32, 4, 1, True).parameters())
#            == sum(p.numel() for p in _OrderHead(64, 32, 4).parameters()),
#            "at G=1 the grouped head has exactly the classic head's parameters")
#     # A dead subspace must NOT be recoverable: that is the structural floor.
#     slices = head.slices(torch.randn(3, 4, 64))
#     _check(tuple(slices.shape) == (3, 8, 4, 8),
#            "slices() cuts the embedding into G disjoint blocks of D/G")

#     # 8. the stitch head has no per-span capacity
#     stitch = _StitchHead(8, 16)
#     logits = stitch(torch.randn(6, 8, 4, 8))
#     _check(tuple(logits.shape) == (6, 8, 4, 5), "stitch emits K+1 classes per tile")
#     _check(bool(torch.isinf(logits[:, :, 0, 0]).all()),
#            "a tile is masked out as its own successor")
#     _check(sum(p.numel() for p in stitch.parameters()) == 8 * 16 * 2 + 8 + 1,
#            "stitch parameter count is independent of the number of spans")

#     # 9. end to end, and the thing this file is for: rank follows n_groups
#     ratios = {}
#     for groups in (1, 8):
#         model = SoloJigsaw(n_groups=groups, window_size=6, n_tiles=3,
#                            tile_gap=(1, 3), output_dimension=24,
#                            num_hidden_units=32, head_hidden_units=32,
#                            lambda_stitch=1.0, lambda_space=0.5 if groups > 1 else 0.0,
#                            space_norm="block_mean", batch_size=128, max_epochs=6,
#                            device="cpu", random_state=3, verbose=False)
#         model.fit(counts)
#         embedding = model.transform(counts)
#         _check(embedding.shape == (len(counts), 24),
#                f"G={groups}: transform returns one 24-d vector per bin")
#         _check(np.isfinite(embedding).all(), f"G={groups}: embedding is finite")
#         metrics = model.evaluate_pretext(counts, max_spans=128, repeats=2,
#                                          verbose=False, n_boot=0)
#         _check(metrics["n_groups"] == groups and metrics["chance_pair_percent"] == 50.0,
#                f"G={groups}: evaluate_pretext reports the group count")
#         _check(metrics["stitch_accuracy_percent"] is not None,
#                f"G={groups}: stitch accuracy is reported when the term is on")
#         _check(abs(metrics["label_bits_per_span"]
#                    - groups * math.lgamma(4)) < 1e-6,
#                f"G={groups}: the label budget scales with the group count")
#         _check(all(float(row["reconstruct"]) == 0.0 for row in model.history_),
#                f"G={groups}: reconstruct CE is EXACTLY zero on every epoch")
#         ratios[groups] = _participation_ratio(embedding)
#         print(f"        G={groups}: participation ratio {ratios[groups]:.2f} / 24, "
#               f"pair {metrics['pair_accuracy_percent']:.1f}%, "
#               f"continuity baseline {metrics['baseline_continuity_pair_percent']}")
#     _check(ratios[8] > ratios[1],
#            f"rank rises with the group count ({ratios[1]:.2f} -> {ratios[8]:.2f}); "
#            "six epochs on synthetic counts, so this is a smoke test of the "
#            "MECHANISM, not evidence about Perich")

#     # 9b. a SPACE-ONLY arm: the three oracles, and the fast path must still
#     #     deliver gradient to the encoder. This is the arm the next run leans
#     #     on, so it gets checked end to end rather than by inspection.
#     space = SoloJigsaw(n_groups=4, window_size=6, n_tiles=3, tile_gap=(1, 3),
#                        output_dimension=24, num_hidden_units=32,
#                        head_hidden_units=32, lambda_order=0.0, lambda_pair=0.0,
#                        lambda_space=1.0, space_norm="row_zscore",
#                        batch_size=128, max_epochs=6, device="cpu",
#                        random_state=3, verbose=False)
#     space.fit(counts)
#     metrics = space.evaluate_pretext(counts, max_spans=128, repeats=2,
#                                      verbose=False, n_boot=0)
#     for key in ("space_accuracy_percent", "space_oracle_percent",
#                 "space_oracle_rate_percent", "space_oracle_level_percent"):
#         _check(metrics[key] is not None, f"space-only arm reports {key}")
#     _check(metrics["stitch_accuracy_percent"] is None
#            and metrics["train_stitch_accuracy_percent"] is None,
#            "a switched-off term reports None, not 0.0 against a 25% chance line")
#     _check(metrics["space_norm"] == "row_zscore"
#            and metrics["space_time_shuffle"] is False
#            and metrics["space_block_rows"] == space.space_block_rows() > 1,
#            "the spatial configuration is reported alongside its numbers")
#     _check(metrics["chance_space_percent"] == 25.0,
#            "chance for a 4-block spatial puzzle is 25%")
#     # The no_grad fast path is the one change in this round that could silently
#     # break training, so it is checked against its own zero-epoch twin: same
#     # seed, same init, no steps. If the encoder weights come out identical the
#     # spatial term never reached them and every space-only number would be a
#     # frozen encoder wearing a hat.
#     frozen = SoloJigsaw(n_groups=4, window_size=6, n_tiles=3, tile_gap=(1, 3),
#                         output_dimension=24, num_hidden_units=32,
#                         head_hidden_units=32, lambda_order=0.0, lambda_pair=0.0,
#                         lambda_space=1.0, space_norm="row_zscore",
#                         batch_size=128, max_epochs=0, device="cpu",
#                         random_state=3, verbose=False)
#     frozen.fit(counts)
#     moved_weights = [not torch.allclose(a, b) for a, b in
#                      zip(space.encoder_.parameters(), frozen.encoder_.parameters())]
#     _check(any(moved_weights),
#            "the spatial term ALONE moves the encoder -- the no_grad fast path "
#            "skips the permutation backward, not the spatial one")
#     _check(float(space.history_[-1]["position"]) > 0.0,
#            "the permutation head is still measured on a space-only arm")

#     # 10. the arm table
#     stray, broke = [], []
#     known = set(SoloJigsaw._PARAM_NAMES) | {"_model"}
#     for name, settings in SOLO_ARMS.items():
#         stray += [f"{name}.{k}" for k in settings if k not in known]
#         try:
#             SoloJigsaw(**{k: v for k, v in settings.items() if k != "_model"})
#         except Exception as error:                            # noqa: BLE001
#             broke.append(f"{name}: {error}")
#     _check(not stray, f"every arm key is a real parameter{'' if not stray else stray}")
#     _check(not broke, f"every arm constructs{'' if not broke else broke}")
#     # The baseline arm is an OrderJigsaw, not a SoloJigsaw, and must construct
#     # as one -- otherwise the autoencoder contrast vanishes again.
#     for name, settings in SOLO_BASELINE_ARMS.items():
#         OrderJigsaw(**{k: v for k, v in settings.items() if k != "_model"})
#         _check(settings["_model"] == "solo_ae",
#                f"{name} is dispatched to the baseline model key")
#     named = set(SOLO_ARMS) | set(SOLO_BASELINE_ARMS)
#     missing = [f"{a}-{b}" for a, b, _, _ in SOLO_CONTRASTS
#                if a not in named or b not in named]
#     _check(not missing, f"every contrast names real arms{'' if not missing else missing}")
#     _check(all(a in named for a in DEFAULT_SOLO_ARMS),
#            "DEFAULT_SOLO_ARMS resolves -- every one of them, which is what the "
#            "20260928 run could not say about reconstruct_only")
#     registered, arm_table = register({}, {}, [])
#     _check("solo" in registered and "solo_ae" in registered
#            and not [a for a in DEFAULT_SOLO_ARMS if a not in arm_table],
#            "register() wires both model keys and every default arm")
#     for left, right in (("solo_rank8", "solo_shared8"),
#                         ("solo_space_only", "solo_space_rowz"),
#                         ("solo_space_only", "solo_space_tshuf"),
#                         ("solo_space_only", "solo_space_frozen")):
#         differ = [k for k in SOLO_MATCH_KEYS
#                   if SOLO_ARMS[left].get(k) != SOLO_ARMS[right].get(k)]
#         _check(not differ,
#                f"{left} - {right} is geometry-matched{'' if not differ else differ}")
#     print("\nall checks passed.")


# if __name__ == "__main__":
#     _smoke_test()

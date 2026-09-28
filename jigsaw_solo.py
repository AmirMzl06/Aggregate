"""jigsaw_solo.py -- the jigsaw STANDING ALONE. No reconstruction anywhere.

WHY THIS FILE EXISTS
--------------------
The 20260927 run settled the previous question, and not in the jigsaw's favour.
Every matched contrast came out negative:

    order_default - anchor_default      -0.0344     (stock geometry)
    order_gapped  - anchor_gapped       -0.0402     (continuity blocked, gap 3)
    order_default - ctrl_frozen_labels  -0.0359     (labels that mean nothing)
    order_full    - order_full_frozen   -0.0050     (same loss, dead labels)

and the single best arm in the whole table was `ctrl_frozen_labels` at 0.5617 --
the arm whose jigsaw labels carry NO information about time. On the 3-seed run
the same thing happened again: order_lag_frozen 0.5500 and
order_time_axis_frozen 0.5467 beat their own label-carrying versions (0.5371,
0.5404). Whatever the order term was doing, it was not teaching the encoder
about time; it was injecting gradient noise, and a meaningless label injected it
just as well. Everything that made those numbers good came from the
reconstruction anchor.

So: drop the anchor, and make the PUZZLE carry the load.

WHY THE PUZZLE COULD NOT CARRY IT BEFORE -- the one-line diagnosis
-----------------------------------------------------------------
Sorting K tiles is a RANK-1 problem. A permutation of K items is recovered from
one scalar per item: give each tile a score, sort the scores, done. So the
minimal encoder that solves the classic temporal jigsaw perfectly is

    f(tile) = (an estimate of when this tile happened)   -- one number.

Every other direction of the embedding is free to be anything, including
nothing. That is not a training pathology, it is the information geometry of the
task, and it predicts all three things we measured:

  * alone, the objective collapses the embedding -- `order_only` scored R2 0.005
    at participation ratio 3.4 out of 64, which is what "one useful direction
    plus noise" looks like;
  * bolted onto a reconstruction anchor it contributes ~nothing, because it only
    ever pushes on one of the 64 directions and it pushes AGAINST the anchor
    there;
  * a frozen random label works as well as the true one, because a rank-1 target
    is a rank-1 target whatever it means.

The label budget says the same thing in bits. One span of the classic jigsaw
carries log(4!) = 3.18 nats of supervision. One span of the reconstruction
target carries 38 neurons x 10 bins x log(8) levels = 790 nats. A factor of 248.
The autoencoder was never beating the jigsaw on cleverness; it was beating it on
supervision, by two and a half orders of magnitude.

WHAT THIS FILE DOES ABOUT IT
----------------------------
Raise the RANK and the BIT COUNT of the puzzle itself, without adding a
reconstruction term, without InfoNCE, without contrastive learning, and without
leaving cross-entropy. Three mechanisms, each a puzzle, each pure CE:

1. GROUPED JIGSAW (`n_groups`). Split the neurons into G groups and draw an
   INDEPENDENT permutation of the K time chunks FOR EACH GROUP. Tile j is then a
   chimera: group 1's rows come from chunk 3, group 2's from chunk 1, and so on.
   The head must place all G of them. One scalar can no longer do it -- two
   groups with different permutations need two different scores from the same
   embedding -- so the sufficient statistic is G-dimensional and the label budget
   is G x log(K!) nats per span instead of log(K!). G=1 is EXACTLY the classic
   jigsaw, same head, same parameter count, which is what makes `solo_rank1` an
   honest control rather than a different model.

2. GROUP SUBSPACES (`group_subspace`). Group g reads only its own D/G slice of
   the embedding. Now the rank floor is structural, not statistical: if slice g
   collapses, group g's puzzle is unsolvable and its cross-entropy stays at
   log K forever. Collapse stops being a free minimum and becomes a penalty.

3. STITCH CE (`lambda_stitch`). For each tile, "which of the other tiles is my
   immediate successor?" -- K+1 classes, the last one meaning "nothing follows
   me". Scored by a BILINEAR compatibility form, which is the point: the head is
   a fixed pair of D x r matrices with NO per-span capacity, so it cannot become
   the lookup table that took `order_default`'s train pair accuracy to 100% while
   held-out sat at 48.2%. The only way to lower this loss is an embedding in
   which the end of one chunk and the start of the next are recognisably the same
   trajectory -- an instantaneous-state code, which is exactly what a velocity
   decoder wants, and the opposite of the "when am I" code that sorting rewards.

   and a fourth, on the other axis of the data:

4. NEURON-AXIS JIGSAW (`lambda_space`). The original jigsaw puzzle is SPATIAL.
   Permute which neuron block sits in which row slot and ask the model to name
   them: G-way CE per slot, log(G!) nats per span. Solving it requires the
   embedding to carry per-group identity -- which cells are active, not when --
   and that is high-rank by construction.

One knob is NOT a puzzle and is labelled as such: `lambda_spread` is a variance
floor on the embedding (a degeneracy guard, no positives, no negatives, no
InfoNCE). It defaults to 0 so that nothing in the headline number comes from it
unless you ask; there is an arm that turns it on so its contribution stays
attributable.

THE GUARANTEE
-------------
`lambda_reconstruct` and `lambda_forecast` are forced to zero in the constructor
and raise if you pass anything else. There is no reconstruction head in the loss
of this file. When you run it you will see `reconstruct CE=0.0000` on every
epoch line -- that zero is the proof, printed once per log interval, that the
number at the end came from the puzzle.

WHAT WOULD FALSIFY THE IDEA
---------------------------
`solo_rank8 - solo_shared8`. Same architecture, same head, same G subspaces,
same parameter count, same number of cross-entropy terms; the ONLY difference is
whether the G permutations are drawn independently or are all copies of one
permutation. If independence is worth nothing, that contrast is zero and the
rank story is wrong. Run it before you believe anything else in this file.

API is the same as the rest of the project: fit / transform / fit_transform /
evaluate_pretext / save / load, sklearn-style, no CEBRA, no contrastive loss.
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

SPACE_NORMS = ("none", "block_mean", "block_zscore")


# --------------------------------------------------------------------------- #
# 1. heads
# --------------------------------------------------------------------------- #
class _GroupOrderHead(nn.Module):
    """K-way slot logits and a before/after score, FOR EACH of G groups.

    Two wirings, and the difference between them is the whole rank argument.

    subspace=True   group g is shown only z[..., g*w:(g+1)*w]. The readout
                    weights are shared across groups, so the groups can only
                    give different answers if their slices hold different
                    information. That is the structural rank floor: a dead slice
                    is a group stuck at log K forever.
    subspace=False  every group sees the whole embedding and gets its own
                    readout row. Rank is then only encouraged, never enforced --
                    which is the control that tells you whether the subspaces
                    are doing the work or the independent labels are.

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
    """"Which tile follows me?" as a bilinear form with no per-span capacity.

    logit(i -> j) = <Q z_i, K z_j> / sqrt(r), plus one extra class per tile for
    "nothing follows me, I am last". The self-pair is masked out.

    The parameter count is 2 x width x r plus a vector, INDEPENDENT of how many
    spans exist. That matters more than it sounds: the reason the old order term
    stopped teaching the encoder anything was that its MLP head had enough
    capacity to identify which of the ~900 span starts it was looking at and
    answer from a table -- train pair accuracy 100%, held-out 48.2%. A bilinear
    score cannot hold a table. It can only report whether two embeddings look
    like consecutive pieces of one trajectory, so if this loss falls, something
    in the embedding got better at representing the trajectory.
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
    """Remove the per-block level cue the space task would otherwise ride on."""
    if mode == "none":
        return window
    rows = slots.reshape(-1)
    blocks = window[:, rows, :].reshape(window.shape[0], slots.shape[0], -1)
    centered = blocks - blocks.mean(dim=2, keepdim=True)
    if mode == "block_zscore":
        centered = centered / (blocks.std(dim=2, keepdim=True) + 1e-5)
    out = window.clone()
    out[:, rows, :] = centered.reshape(window.shape[0], len(rows), -1)
    return out


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
    group_chimera    True  -> one permutation PER GROUP (the rank mechanism)
                     False -> one permutation shared by all groups, same
                              architecture. This is the control for n_groups.
    group_subspace   group g reads only its own D/G slice of the embedding.
                     Requires output_dimension % n_groups == 0.
    group_seed       which random partition of the neurons to use.
    lambda_stitch    weight on the successor CE (bilinear head, no lookup table)
    stitch_rank      rank of that bilinear form
    lambda_space     weight on the neuron-axis (spatial) jigsaw
    space_norm       "none" | "block_mean" | "block_zscore" -- removes the
                     per-block firing-rate cue from the spatial puzzle
    lambda_spread    variance floor on the embedding. NOT a puzzle: a
                     degeneracy guard, off by default, with its own arm so its
                     contribution never hides inside a headline number.

    Everything inherited from OrderJigsaw still applies: span selection, the
    gap curriculum, order-view-only augmentation, hard-example mining, head
    resets, and the label controls. `lambda_lag` is available at n_groups=1
    only -- with independent per-group permutations a single signed lag between
    two tiles is not well defined, and the 3-seed run showed the lag readout
    scoring HIGHER with frozen labels (25.0%) than with true ones (21.9%), so it
    was not evidence of anything worth generalizing.
    """

    _MODEL_TYPE = "solo_jigsaw"
    _PARAM_NAMES = OrderJigsaw._PARAM_NAMES + (
        "n_groups", "group_chimera", "group_subspace", "group_seed",
        "lambda_stitch", "stitch_rank", "lambda_space", "space_norm",
        "lambda_spread")

    def __init__(self, *, n_groups=1, group_chimera=True, group_subspace=True,
                 group_seed=101, lambda_stitch=0.0, stitch_rank=32,
                 lambda_space=0.0, space_norm="none", lambda_spread=0.0,
                 **kwargs):
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
        self.n_groups = _integer("n_groups", n_groups, 1)
        self.group_chimera = _boolean("group_chimera", group_chimera)
        self.group_subspace = _boolean("group_subspace", group_subspace)
        self.group_seed = _integer("group_seed", group_seed, 0)
        self.lambda_stitch = _real("lambda_stitch", lambda_stitch, 0)
        self.stitch_rank = _integer("stitch_rank", stitch_rank, 1, 512)
        self.lambda_space = _real("lambda_space", lambda_space, 0)
        self.space_norm = space_norm
        self.lambda_spread = _real("lambda_spread", lambda_spread, 0)
        if self.group_subspace and self.output_dimension % self.n_groups:
            raise ValueError(
                f"group_subspace splits the embedding into n_groups slices, so "
                f"output_dimension ({self.output_dimension}) must be divisible "
                f"by n_groups ({self.n_groups}). Pick 64/8, 64/4, 60/5 and so "
                f"on, or set group_subspace=False.")
        if self.lambda_space > 0 and self.n_groups < 2:
            raise ValueError("lambda_space needs n_groups >= 2: with one block "
                             "there is no spatial permutation to recover.")
        if self.lambda_lag > 0 and self.n_groups > 1:
            raise ValueError(
                "lambda_lag is defined for a single shared permutation and "
                "n_groups > 1 draws one permutation per group, so 'the signed "
                "offset between tile i and tile j' has G different answers. Set "
                "n_groups=1 to use the lag term.")
        live = (self.lambda_order + self.lambda_pair + self.lambda_lag
                + self.lambda_stitch + self.lambda_space + self.lambda_spread)
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
        if self.n_groups > channels:
            raise ValueError(f"n_groups={self.n_groups} exceeds the {channels} "
                             "recorded neurons.")
        self.group_index_ = group_partition(
            channels, self.n_groups, self.random_state + self.group_seed
        ).to(self.device_)
        sizes = torch.bincount(self.group_index_, minlength=self.n_groups)
        self.group_sizes_ = sizes.tolist()
        lag_head = self.order_head_.lag_head
        self.order_head_ = _GroupOrderHead(
            self.output_dimension, self.head_hidden_units, self.n_tiles,
            self.n_groups, self.group_subspace).to(self.device_)
        self.order_head_.lag_head = lag_head
        self.lag_head_ = lag_head
        if self.lambda_stitch > 0:
            self.order_head_.stitch = _StitchHead(
                self.order_head_.width, self.stitch_rank).to(self.device_)
        if self.lambda_space > 0:
            self.order_head_.space = _SpaceHead(
                self.output_dimension, self.head_hidden_units,
                self.n_groups).to(self.device_)
            # Equal-height blocks. See `permute_blocks` for why the leftover
            # channels are dropped from the spatial puzzle rather than padded.
            block = int(sizes.min())
            if block < 1:
                raise ValueError(
                    f"n_groups={self.n_groups} over {channels} neurons leaves an "
                    "empty group; the spatial puzzle needs at least one row per "
                    "block.")
            members = [torch.nonzero(self.group_index_ == g, as_tuple=False)
                       .flatten()[:block] for g in range(self.n_groups)]
            self.space_slots_ = torch.stack(members).to(self.device_)
            self.space_block_ = block
        # (4096, G, K) and (4096, G): the frozen-label control, group-aware.
        # Same contract as the inherited table -- one arbitrary but FIXED answer
        # per block of span starts, so the head can still learn it and the only
        # thing removed is that the answer means anything.
        table = torch.Generator(device=self.device_)
        table.manual_seed(self.random_state + 1861)
        self._frozen_groups = torch.rand(
            4096, self.n_groups, self.n_tiles, device=self.device_,
            generator=table).argsort(dim=2)
        self._frozen_space = torch.rand(
            4096, self.n_groups, device=self.device_, generator=table).argsort(dim=1)
        if self.lambda_space > 0:
            self.group_profile_ = None       # filled on the first training step

    # -- labels ------------------------------------------------------------- #
    def _group_labels(self, batch):
        """(B,G,K). Independent permutations unless group_chimera is off."""
        rows = self.n_groups if self.group_chimera else 1
        keys = torch.rand(batch, rows, self.n_tiles, device=self.device_,
                          generator=self._generator)
        labels = keys.argsort(dim=2)
        return labels if self.group_chimera else labels.expand(
            -1, self.n_groups, -1).contiguous()

    def _control(self, truth, starts, table):
        if self.label_control == "true":
            return truth
        if self.label_control == "fresh_random":
            keys = torch.rand(truth.shape, device=truth.device,
                              generator=self._generator)
            return keys.argsort(dim=-1)
        block = self.frozen_block if self.frozen_block > 0 else self.training_span
        row = (starts // max(block, 1)) % table.shape[0]
        return table[row]

    def _block_stats(self, window):
        """(B,N,W) -> (B,G,W): each row slot's block averaged over its rows.

        The summary a label-free rule gets to use. Averaging over rows and not
        over time is deliberate: a scalar per block would be exactly the cue
        `space_norm="block_mean"` removes, so the baseline would collapse to
        chance for a reason that has nothing to do with how hard the task is.
        """
        rows = self.space_slots_.reshape(-1)
        return window[:, rows].reshape(len(window), self.n_groups,
                                       self.space_block_, -1).mean(dim=2)

    # -- one step ----------------------------------------------------------- #
    def _step(self, data, starts):
        """Pure cross-entropy, four puzzle terms, zero reconstruction."""
        batch = len(starts)
        self._steps_ += 1
        if self.order_head_reset_every and self._steps_ % self.order_head_reset_every == 0:
            self._reset_order_head()
        order_scale = 1.0 + (self.order_final_scale - 1.0) * self._progress()
        n_tiles, n_groups = self.n_tiles, self.n_groups

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
        labels = self._control(self._group_labels(batch), starts,
                               self._frozen_groups)
        view = place_groups(view, labels, self.group_index_)
        z = self.encoder_(view.reshape(-1, self.n_features_in_, self.window_size))
        z = _GradScale.apply(z, self.order_grad_scale)
        z = z.reshape(batch, n_tiles, self.output_dimension)
        position_logits, scores = self.order_head_(z)         # (B,G,K,K) (B,G,K)

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
            sigma = torch.rand(batch, self.n_groups, device=self.device_,
                               generator=self._generator).argsort(dim=1)
            window = normalize_blocks(
                permute_blocks(window, self.space_slots_, sigma),
                self.space_slots_, self.space_norm)
            if self.group_profile_ is None:
                # The per-group template for the label-free baseline in
                # `evaluate_pretext`, measured ON THE NORMALIZED VIEW -- the same
                # thing the encoder is shown. Taking it from the raw window would
                # build a baseline out of a cue that `space_norm` has already
                # deleted, and the model would then be "beating" a rule it was
                # never up against. Averaged over the block's rows but NOT over
                # time, so under block_mean (which removes only the scalar level)
                # the block's temporal shape still makes this a real opponent.
                with torch.no_grad():
                    self.group_profile_ = self._block_stats(
                        normalize_blocks(tiles[:, 0], self.space_slots_,
                                         self.space_norm)).mean(dim=0)
            sigma = self._control(sigma, starts, self._frozen_space)
            space_logits = self.order_head_.space(self.encoder_(window))
            space_each = F.cross_entropy(
                space_logits.reshape(-1, self.n_groups), sigma.reshape(-1),
                reduction="none").reshape(batch, self.n_groups).mean(1)
            with torch.no_grad():
                parts["space_accuracy"] = (
                    space_logits.argmax(-1) == sigma).to(torch.float32).mean()
        else:
            space_each, parts["space_accuracy"] = torch.zeros(
                batch, device=self.device_), zero

        combined = (self.lambda_order * position_each
                    + self.lambda_pair * pair_each
                    + self.lambda_stitch * stitch_each
                    + self.lambda_space * space_each)
        if self.order_hard_fraction < 1.0:
            keep = max(2, int(round(self.order_hard_fraction * batch)))
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
            self._train_stitch_ = float(parts["stitch_accuracy"])
            self._train_space_ = float(parts["space_accuracy"])
        parts["total"] = total
        self._trace(parts)
        return parts

    def _trace(self, parts):
        """Live line for the terms the inherited epoch log knows nothing about.

        The base logger prints position CE, pair BCE, forecast CE and
        reconstruct CE. In this file the last two are structurally 0.0000 --
        useful as proof, useless as progress -- and the stitch and space terms
        it does not know about are the ones worth watching. Printed on the same
        cadence as the base line, and only then, so the cost of the device sync
        is once per `log_every` epochs.
        """
        if not self.verbose or self.lambda_stitch + self.lambda_space == 0:
            return
        per_epoch = max(1, math.ceil(getattr(self, "n_spans_", 1) / self.batch_size))
        if self._steps_ % max(1, per_epoch * self.log_every):
            return
        print(f"      solo[last batch]: stitch CE={float(parts['stitch']):.4f} "
              f"(acc {100 * float(parts['stitch_accuracy']):.1f}%, "
              f"chance {100 / self.n_tiles:.1f}%) "
              f"space CE={float(parts['space']):.4f} "
              f"(acc {100 * float(parts['space_accuracy']):.1f}%, "
              f"chance {100 / self.n_groups:.1f}%) "
              f"spread={float(parts['spread']):.4f}", flush=True)

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
        n_tiles, n_groups = self.n_tiles, self.n_groups
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
                "continuity", "stitch", "space", "space_oracle")
        per_span = {k: torch.zeros(len(chosen), device=self.device_) for k in keys}
        position_ce = 0.0
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
                    for group in range(shown):
                        rows = torch.nonzero(self.group_index_ == group,
                                             as_tuple=False).flatten()
                        piece = shown_tiles[:, :, rows, :]
                        truth = labels[:, group, :]
                        g_exact, g_pair, _ = _order_metrics(
                            _ranks(piece.mean(dim=(2, 3))), truth)
                        level = level + g_pair / shown
                        level_exact = level_exact + g_exact / shown
                        continuity = continuity + continuity_strength(piece, truth) / shown
                    per_span["mean_pair"][begin:stop] += level / repeats
                    per_span["mean_exact"][begin:stop] += level_exact / repeats
                    per_span["level_oracle"][begin:stop] += torch.maximum(
                        level, 1 - level) / repeats
                    per_span["continuity"][begin:stop] += continuity / repeats

                    if self.lambda_stitch > 0:
                        logits = self.order_head_.stitch(self.order_head_.slices(z))
                        hit = (logits.argmax(-1) == successor_targets(labels))
                        per_span["stitch"][begin:stop] += hit.reshape(
                            size, -1).to(torch.float32).mean(1) / repeats
                    if self.lambda_space > 0:
                        sigma = torch.rand(size, n_groups, device=self.device_,
                                           generator=generator).argsort(dim=1)
                        shown_window = normalize_blocks(
                            permute_blocks(tiles[:, 0], self.space_slots_, sigma),
                            self.space_slots_, self.space_norm)
                        guess = self.order_head_.space(
                            self.encoder_(shown_window)).argmax(-1)
                        per_span["space"][begin:stop] += (guess == sigma).to(
                            torch.float32).mean(1) / repeats
                        # The label-free opponent: name each block by matching it
                        # to the per-group template, on the SAME view the encoder
                        # got. If the model cannot beat this, it has learned which
                        # cells fire how much, not which cells they are.
                        # `group_profile_` is a training-time statistic and is not
                        # in the state_dict, so a reloaded model reports None here
                        # rather than a number built from the wrong thing.
                        profile = getattr(self, "group_profile_", None)
                        if profile is None:
                            per_span["space_oracle"][begin:stop] = float("nan")
                        else:
                            stats = self._block_stats(shown_window)   # (B,G,W)
                            distance = (stats[:, :, None, :]
                                        - profile[None, None, :, :]).pow(2).sum(-1)
                            per_span["space_oracle"][begin:stop] += (
                                distance.argmin(-1) == sigma).to(
                                    torch.float32).mean(1) / repeats
                    position_ce += float(F.cross_entropy(
                        position_logits.reshape(-1, n_tiles),
                        labels.reshape(-1))) * size
        self.encoder_.train()
        self.order_head_.train()
        mean = {k: float(v.mean()) for k, v in per_span.items()}

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
            label_bits_per_span=float(n_groups * math.lgamma(n_tiles + 1)),
            stitch_accuracy_percent=percent("stitch", self.lambda_stitch > 0),
            # The best LABEL-FREE constant rule ("always answer 'I am last'", or
            # always answer some fixed slot) scores 1/K, not 1/(K+1): the self
            # logit is masked, so there are only K live options. Quoting the
            # smaller number would make a useless head look educated.
            chance_stitch_percent=100.0 / n_tiles,
            train_stitch_accuracy_percent=(100 * self._train_stitch_
                                           if hasattr(self, "_train_stitch_") else None),
            space_accuracy_percent=percent("space", self.lambda_space > 0),
            space_oracle_percent=percent("space_oracle", self.lambda_space > 0),
            chance_space_percent=100.0 / n_groups,
            train_space_accuracy_percent=(100 * self._train_space_
                                          if hasattr(self, "_train_space_") else None),
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
                  f"(chance {result['chance_stitch_percent']:.2f}%) "
                  f"| space={show('space_accuracy_percent')} "
                  f"(chance {result['chance_space_percent']:.2f}%, rate-oracle "
                  f"{show('space_oracle_percent', 'n/a')})", flush=True)
        return result


# --------------------------------------------------------------------------- #
# 4. arms
# --------------------------------------------------------------------------- #
#  Read this table as three blocks:
#    the LADDER   -- rank 1, 4, 8, 16. One variable, and the prediction the
#                    whole file rests on: R2 and participation ratio rise with G.
#    the CONTROLS -- shared permutations, no subspaces, frozen labels, frozen
#                    encoder. Each removes exactly one thing the ladder claims
#                    is load-bearing.
#    the TERMS    -- stitch and space on their own and left out, so the headline
#                    arm can be decomposed rather than admired.
_SOLO = dict(_model="solo")
_LADDER = dict(_SOLO, lambda_stitch=1.0, lambda_space=0.5, space_norm="block_mean")

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
    # Independent labels but no structural subspace: rank is encouraged by the
    # task and not enforced by the wiring. Separates "the puzzle needs rank"
    # from "the architecture was handed rank".
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
    "solo_space_only": dict(_SOLO, n_groups=8, lambda_order=0.0, lambda_pair=0.0,
                            lambda_space=1.0, space_norm="block_mean"),
    # The spatial puzzle WITHOUT the rate cue removed. If this beats
    # `solo_space_only` the model was reading firing rates, which the
    # space_oracle column in the audit will confirm or deny.
    "solo_space_raw": dict(_SOLO, n_groups=8, lambda_order=0.0, lambda_pair=0.0,
                           lambda_space=1.0, space_norm="none"),
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
}

# `group_chimera`, the four lambdas and `lambda_spread` are deliberately absent:
# those are the variables under test, exactly as the order weights are absent
# from ANCHOR_MATCH_KEYS. `n_groups` IS here, because changing it changes the
# subspace width and therefore the readout capacity -- a rank-1-versus-rank-8
# comparison is not a one-variable contrast and the runner should say so.
SOLO_MATCH_KEYS = ANCHOR_MATCH_KEYS + ("n_groups", "group_subspace", "group_seed",
                                       "space_norm", "stitch_rank")

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
     "what the successor CE adds. Its head is bilinear, so unlike the slot head "
     "it has no per-span capacity to memorize with."),
    ("solo_full", "solo_no_space", "SPACE effect",
     "what the neuron-axis puzzle adds."),
    ("solo_full", "solo_no_subspace", "SUBSPACE effect",
     "structural rank floor against a merely encouraged one."),
    ("solo_rank8", "solo_spread", "what a variance floor adds on top",
     "if this is large, the puzzle is not preventing collapse on its own and the "
     "honest description of the result includes a regularizer."),
    ("solo_full", "reconstruct_only", "JIGSAW ALONE vs THE AUTOENCODER",
     "the number the project is actually about. reconstruct_only is the arm that "
     "carried every previous result; solo_full never sees a reconstruction "
     "target. Positive means the puzzle stands on its own."),
    ("solo_full", "solo_random", "vs FROZEN ENCODER",
     "what training is worth at all. Must be positive or nothing else counts."),
)

DEFAULT_SOLO_ARMS = ("solo_rank1", "solo_rank4", "solo_rank8", "solo_shared8",
                     "solo_full", "solo_full_frozen", "solo_no_stitch",
                     "solo_no_space", "solo_random", "reconstruct_only")


def register(models, arms, contrasts=None):
    """Wire this module into run_jigsaw.py without editing its tables by hand."""
    models["solo"] = SoloJigsaw
    for name, settings in SOLO_ARMS.items():
        arms[name] = dict(settings)
    if contrasts is not None:
        for row in SOLO_CONTRASTS:
            if row not in contrasts:
                contrasts.append(row)
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
    # A dead subspace must NOT be recoverable: that is the structural floor.
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
        _check(abs(metrics["label_bits_per_span"]
                   - groups * math.lgamma(4)) < 1e-6,
               f"G={groups}: the label budget scales with the group count")
        _check(all(float(row["reconstruct"]) == 0.0 for row in model.history_),
               f"G={groups}: reconstruct CE is EXACTLY zero on every epoch")
        ratios[groups] = _participation_ratio(embedding)
        print(f"        G={groups}: participation ratio {ratios[groups]:.2f} / 24, "
              f"pair {metrics['pair_accuracy_percent']:.1f}%, "
              f"continuity baseline {metrics['baseline_continuity_pair_percent']}")
    _check(ratios[8] > ratios[1],
           f"rank rises with the group count ({ratios[1]:.2f} -> {ratios[8]:.2f}); "
           "six epochs on synthetic counts, so this is a smoke test of the "
           "MECHANISM, not evidence about Perich")

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
    named = set(SOLO_ARMS)
    missing = [f"{a}-{b}" for a, b, _, _ in SOLO_CONTRASTS
               if a not in named or (b not in named and b != "reconstruct_only")]
    _check(not missing, f"every contrast names real arms{'' if not missing else missing}")
    _check(all(a in named for a in DEFAULT_SOLO_ARMS if a != "reconstruct_only"),
           "DEFAULT_SOLO_ARMS resolves")
    pair = (SOLO_ARMS["solo_rank8"], SOLO_ARMS["solo_shared8"])
    differ = [k for k in SOLO_MATCH_KEYS if pair[0].get(k) != pair[1].get(k)]
    _check(not differ,
           f"the headline contrast is geometry-matched{'' if not differ else differ}")
    print("\nall checks passed.")


if __name__ == "__main__":
    _smoke_test()

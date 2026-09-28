"""Context-conditioned temporal Jigsaw for binned neural activity (PyTorch >=2.1).

This is a new, standalone implementation, NOT a reproduction of the unavailable
mobile_jigsaw.py. No reconstruction, behavioral targets, CTC, or gradient attack
is used by this module. Gaussian noise is the only stochastic noise augmentation.
Tile reordering / channel time replacement are pretext tasks, not attacks.

Research pointers (ideas, not claims of reproducing these papers):
  Skip-Clip: https://arxiv.org/abs/1910.12770 -- context-conditioned future order.
  PopT: https://arxiv.org/abs/2406.03044 -- channel/ensemble discrimination.
  NDT2: https://papers.neurips.cc/paper_files/paper/2023/file/fe51de4e7baf52e743b679e3bdba7905-Paper-Conference.pdf
  POYO: https://arxiv.org/abs/2310.16046 -- population aggregation, not our loss.
  CPC: https://arxiv.org/abs/1807.03748 -- separate future contrastive control.
  SCARF: https://arxiv.org/abs/2106.15147 -- separate corruption contrastive control.

All durations are BINS. A gap of zero means exactly contiguous, nonoverlapping
tiles. Inference is causal: the last bin of the local window is the decoded bin.
Positions inside a window are allowed; candidate tiles NEVER get their true time,
rank, gap, session/trial ID, or shuffled-slot positional encoding as model input.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURES = ("residual", "mobile_v1", "mobile_v2", "mobile_v3",
                 "mobile_level", "temporal_transformer", "population_transformer")


@dataclass
class JigsawConfig:
    channels: int = 1
    window: int = 8
    context: int = 24
    tiles: int = 4
    gap_min: int = 3
    gap_max: int = 5
    scales: tuple[int, ...] = (1,)
    architecture: str = "residual"
    width: int = 64
    dimension: int = 64
    depth: int = 3
    dropout: float = 0.1
    expansion: int = 3
    width_multiplier: float = 1.0
    stem: str = "neuron"
    norm: str = "layer"
    activation: str = "gelu"
    level_stats: bool = False
    head: str = "deepsets"
    context_mode: str = "true"  # true / none / shuffled / reversed
    view: str = "raw"          # raw / mean / zscore / count_match
    l2_normalize: bool = False
    lambda_order: float = 1.0
    lambda_pair: float = 0.5
    lambda_lag: float = 0.0
    lag_classes: int = 8
    auxiliary: str = "none"    # none / channel / shift / ensemble / cpc / scarf
    lambda_aux: float = 0.0
    corruption_fraction: float = 0.2
    temperature: float = 0.1
    label_control: str = "true" # true / frozen / fresh
    span_selection: str = "random" # random / decorrelated / reverse_drift / shortcut
    selection_pool: int = 6
    selection_jitter: int = 12
    gap_curriculum: float = 0.0
    hard_fraction: float = 1.0
    reset_every: int = 0
    order_final_scale: float = 1.0
    time_roll: int = 0
    noise_site: str = "none"    # none / input / embedding
    noise_sd: float = 0.0
    seed: int = 0

    def validate(self):
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        for k in ("channels", "window", "context", "width", "dimension", "depth"):
            if getattr(self, k) < 1:
                raise ValueError(f"{k} must be positive")
        if not 2 <= self.tiles <= 6:
            raise ValueError("tiles must be 2..6 (exact assignment evaluation)")
        if self.gap_min < 0 or self.gap_max < self.gap_min:
            raise ValueError("Require 0 <= gap_min <= gap_max")
        if not self.scales or min(self.scales) < 1:
            raise ValueError("scales must contain positive integers")
        if self.lag_classes < 2 or self.lag_classes % 2:
            raise ValueError("lag_classes must be even and >=2")
        choices = dict(head=("linear", "deepsets", "relational"),
                       context_mode=("true", "none", "shuffled", "reversed"),
                       view=("raw", "mean", "zscore", "count_match"),
                       auxiliary=("none", "channel", "shift", "ensemble", "cpc", "scarf"),
                       label_control=("true", "frozen", "fresh"),
                       span_selection=("random", "decorrelated", "reverse_drift", "shortcut"),
                       noise_site=("none", "input", "embedding"),
                       stem=("neuron", "mix"), norm=("layer", "batch", "none"),
                       activation=("gelu", "hardswish"))
        for key, allowed in choices.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f"{key} must be one of {allowed}")
        if not 0 < self.hard_fraction <= 1 or not 0 < self.corruption_fraction < 0.5:
            raise ValueError("hard_fraction in (0,1], corruption_fraction in (0,.5)")
        if not 0 <= self.dropout < 1 or not 0 <= self.gap_curriculum <= 1:
            raise ValueError("dropout in [0,1), gap_curriculum in [0,1]")
        if min(self.noise_sd, self.lambda_order, self.lambda_pair, self.lambda_lag,
               self.lambda_aux, self.order_final_scale) < 0 or self.temperature <= 0:
            raise ValueError("Loss/noise coefficients must be nonnegative; temperature >0")
        if self.noise_site == "none" and self.noise_sd:
            raise ValueError("Set --noise-site input or embedding when --noise-sd >0")
        if self.auxiliary == "none" and self.lambda_aux:
            raise ValueError("lambda_aux requires an auxiliary task")
        if self.channels < 2 and self.lambda_aux and self.auxiliary in ("channel", "shift", "ensemble", "scarf"):
            raise ValueError("Channel tasks require at least two varying neurons")
        if self.view == "count_match" and self.context_mode != "none":
            raise ValueError("count_match is a legacy context-free control only")
        if self.selection_pool < 1 or self.reset_every < 0 or self.time_roll < 0 or self.expansion < 1 or self.width_multiplier <= 0:
            raise ValueError("Invalid sampler/reset/roll setting")
        return self

    @property
    def max_span(self):
        return max(self.scales) * (self.context + self.tiles * (self.window + self.gap_max))

    @property
    def decode_span(self):
        return self.context + self.gap_min + self.window

    def to_dict(self):
        return asdict(self)


class ChannelNorm(nn.Module):
    """Normalize feature channels at each time, not across batch/time."""
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def normalization(width, kind):
    return {"layer": ChannelNorm, "batch": nn.BatchNorm1d,
            "none": lambda _: nn.Identity()}[kind](width)


def act(kind):
    return nn.Hardswish() if kind == "hardswish" else nn.GELU()


class SqueezeExcite(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(width, max(4, width//4), 1),
                                 nn.ReLU(), nn.Conv1d(max(4, width//4), width, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, width, cfg):
        super().__init__()
        kind = cfg.architecture
        a = "hardswish" if kind == "mobile_v3" else cfg.activation
        if kind == "residual":
            body = [nn.Conv1d(width, width, 3, padding=1), act(a),
                    nn.Conv1d(width, width, 3, padding=1)]
        elif kind == "mobile_v1":
            body = [nn.Conv1d(width, width, 3, padding=1, groups=width),
                    normalization(width, cfg.norm), act(a), nn.Conv1d(width, width, 1)]
        else:
            h = width * cfg.expansion
            body = [nn.Conv1d(width, h, 1), normalization(h, cfg.norm), act(a),
                    nn.Conv1d(h, h, 3, padding=1, groups=h), normalization(h, cfg.norm), act(a)]
            if kind == "mobile_v3":
                body += [SqueezeExcite(h)]
            body += [nn.Conv1d(h, width, 1)]  # linear bottleneck
        self.body = nn.Sequential(*body)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return x + self.drop(self.body(x))


def sinusoid(length, width, device, dtype):
    t = torch.arange(length, device=device, dtype=dtype)[:, None]
    f = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype) * (-math.log(10000) / width))
    p = torch.zeros(length, width, device=device, dtype=dtype)
    p[:, 0::2] = (t*f).sin()
    p[:, 1::2] = (t*f[:p[:, 1::2].shape[1]]).cos()
    return p


class WindowEncoder(nn.Module):
    """Variable-length [B,T,N] -> [B,D]; time detail preserved by 4-bin pooling."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        w = max(8, round(cfg.width * cfg.width_multiplier))
        self.width = w
        self.stats = cfg.level_stats or cfg.architecture == "mobile_level"
        self.groups = math.gcd(cfg.channels, 8)
        if cfg.architecture == "temporal_transformer":
            self.input = nn.Linear(cfg.channels, w)
            layer = nn.TransformerEncoderLayer(w, math.gcd(w, 4), 3*w, cfg.dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.trunk = nn.TransformerEncoder(layer, cfg.depth, enable_nested_tensor=False)
            out = 2*w
        elif cfg.architecture == "population_transformer":
            self.channel_net = nn.Sequential(nn.Conv1d(1, w//2, 3, padding=1), nn.GELU(),
                                             nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(4*(w//2), w))
            # Session-local neuron identity, NOT anatomical position or shared cross-session identity.
            self.neuron_id = nn.Parameter(torch.randn(1, cfg.channels, w) * .02)
            self.cls = nn.Parameter(torch.randn(1, 1, w) * .02)
            layer = nn.TransformerEncoderLayer(w, math.gcd(w, 4), 3*w, cfg.dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.trunk = nn.TransformerEncoder(layer, cfg.depth, enable_nested_tensor=False)
            out = w
        else:
            stem = []
            if cfg.stem == "neuron" and cfg.architecture != "residual":
                stem += [nn.Conv1d(cfg.channels, cfg.channels, 3, padding=1, groups=cfg.channels), act(cfg.activation)]
            stem += [nn.Conv1d(cfg.channels, w, 3, padding=1), act(cfg.activation)]
            self.trunk = nn.Sequential(*stem, *[ConvBlock(w, cfg) for _ in range(cfg.depth)],
                                       nn.AdaptiveAvgPool1d(4), nn.Flatten())
            out = w * 4
        self.project = nn.Linear(out + (2*self.groups if self.stats else 0), cfg.dimension)

    def channel_tokens(self, x):
        b, t, n = x.shape
        z = self.channel_net(x.transpose(1, 2).reshape(b*n, 1, t)).reshape(b, n, self.width)
        z = z + self.neuron_id
        return self.trunk(torch.cat((self.cls.expand(b, -1, -1), z), 1))

    def forward(self, x):
        if self.cfg.architecture == "temporal_transformer":
            h = self.input(x)
            h = self.trunk(h + sinusoid(len(h[0]), self.width, x.device, x.dtype))
            h = torch.cat((h.mean(1), h[:, -1]), -1)
        elif self.cfg.architecture == "population_transformer":
            h = self.channel_tokens(x)[:, 0]
        else:
            h = self.trunk(x.transpose(1, 2))
        if self.stats:
            g = x.transpose(1, 2).reshape(len(x), self.groups, -1)
            # Signed means: supports standardized/smoothed rates as well as counts.
            h = torch.cat((h, g.mean(-1), g.std(-1, unbiased=False)), -1)
        z = self.project(h)
        return F.normalize(z, dim=-1) if self.cfg.l2_normalize else z


class OrderHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.kind = cfg.head
        d = cfg.dimension
        if self.kind == "linear":
            self.net = nn.Identity()
        elif self.kind == "deepsets":
            self.net = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        else:
            layer = nn.TransformerEncoderLayer(d, math.gcd(d, 4), 2*d, cfg.dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            # No tile-slot positional encoding: permutation equivariant.
            self.net = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.position = nn.Linear(d, cfg.tiles)
        self.score = nn.Linear(d, 1)

    def forward(self, z):
        if self.kind == "deepsets":
            z = torch.cat((z, z.mean(1, keepdim=True).expand_as(z)), -1)
        h = self.net(z)
        return self.position(h), self.score(h).squeeze(-1)


class ContextJigsaw(nn.Module):
    def __init__(self, cfg: JigsawConfig):
        super().__init__()
        self.cfg = cfg.validate()
        self.encoder = WindowEncoder(cfg)
        d = cfg.dimension
        self.fusion = nn.Sequential(nn.Linear(4*d, 2*d), nn.GELU(), nn.Linear(2*d, d))
        self.order_head = OrderHead(cfg)
        self.lag_head = (nn.Linear(d, cfg.lag_classes) if cfg.head == "linear" else
                         nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.lag_classes)))
        self.channel_head = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(),
                                          nn.Linear(2*d, cfg.channels * (3 if cfg.auxiliary == "shift" else 1)))
        self.ensemble_head = nn.Sequential(nn.Linear(3*d, d), nn.GELU(), nn.Linear(d, 1))
        self.future_heads = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(cfg.tiles)])
        self.contrast_project = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        ij = torch.triu_indices(cfg.tiles, cfg.tiles, 1)
        self.register_buffer("pair_i", ij[0], persistent=False)
        self.register_buffer("pair_j", ij[1], persistent=False)
        low = cfg.window + cfg.gap_min
        high = max(low+1, max(cfg.scales)*(cfg.tiles-1)*(cfg.window+cfg.gap_max))
        edges = torch.logspace(math.log10(low), math.log10(high), cfg.lag_classes//2 + 1)[1:-1]
        self.register_buffer("lag_edges", edges)

    def noisy(self, x, site, enabled):
        if enabled and self.cfg.noise_site == site and self.cfg.noise_sd:
            return x + torch.randn_like(x) * self.cfg.noise_sd
        return x

    def view(self, x):
        if self.cfg.view == "mean":
            return x - x.mean(-2, keepdim=True)
        if self.cfg.view == "zscore":
            return (x - x.mean(-2, keepdim=True)) / x.std(-2, keepdim=True, unbiased=False).clamp_min(.1)
        return x

    def local(self, x, augment=False):
        x = self.noisy(self.view(x), "input", augment)
        if augment and self.cfg.time_roll:
            # Legacy ablation only: roll within each window, never across a trial.
            shift = torch.randint(-self.cfg.time_roll, self.cfg.time_roll+1, (len(x), 1, 1), device=x.device)
            idx = (torch.arange(x.shape[1], device=x.device)[None, :, None]-shift) % x.shape[1]
            x = x.gather(1, idx.expand_as(x))
        return self.noisy(self.encoder(x), "embedding", augment)

    def condition(self, z, c):
        if self.cfg.context_mode == "none":
            return z  # same allocated capacity; no untrained fusion at inference
        return z + self.fusion(torch.cat((z, c, z*c, z-c), -1))

    def embed(self, context, window, augment=False):
        """Chronological, causal downstream representation; no pretext head needed."""
        z = self.local(window, augment)
        if self.cfg.context_mode == "none":
            return z
        c = self.local(context, augment)
        # shuffled/reversed contexts are TRAINING controls; decode uses true past.
        return self.condition(z, c)

    def forward(self, batch, augment=False, context_override=None):
        x = batch["tiles"]
        b, k, w, n = x.shape
        z = self.local(x.reshape(b*k, w, n), augment).reshape(b, k, -1)
        context = batch["context"]
        mode = self.cfg.context_mode if context_override is None else context_override
        if mode == "reversed":
            context = context.flip(1)
        c = (self.local(context, augment) if mode != "none" or self.cfg.auxiliary == "cpc"
             else torch.zeros_like(z[:, 0]))
        if mode == "shuffled":
            c = c.roll(1, 0)
        if mode == "none":
            h = z
        else:
            h = self.condition(z, c[:, None].expand_as(z))
        pos, scores = self.order_head(h)
        lag = self.lag_head(h[:, self.pair_i] - h[:, self.pair_j])
        return dict(position=pos, score=scores, lag=lag, z=z, context_z=c)

    def lag_targets(self, offsets):
        delta = offsets[:, self.pair_j] - offsets[:, self.pair_i]
        magnitude = torch.bucketize(delta.abs().contiguous().float(), self.lag_edges)
        half = self.cfg.lag_classes//2
        return torch.where(delta >= 0, half+magnitude, half-1-magnitude)

    def aux_loss(self, batch, out, augment):
        cfg = self.cfg
        b = len(batch["tiles"])
        if cfg.auxiliary in ("channel", "shift"):
            x = batch["corrupt"]
            z = self.embed(batch["context"], x, augment)
            logits = self.channel_head(z)
            if cfg.auxiliary == "shift":
                logits = logits.reshape(b, cfg.channels, 3)
                target = batch["shift_label"].long()
                # Balance unchanged/past/future classes; shifts may be impossible at one edge.
                counts = torch.bincount(target.flatten(), minlength=3).float().clamp_min(1)
                weight = counts.sum() / (3*counts)
                return F.cross_entropy(logits.transpose(1, 2), target, weight=weight, reduction="none").mean(1)
            target = batch["corruption_mask"].float()
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            # Balanced BCE: predicting all channels clean is not a cheap optimum.
            return .5*((loss*target).sum(1)/target.sum(1).clamp_min(1) +
                       (loss*(1-target)).sum(1)/(1-target).sum(1).clamp_min(1))
        if cfg.auxiliary == "ensemble":
            z1 = self.local(batch["ensemble_a"], augment)
            z2 = self.local(batch["ensemble_b"], augment)
            logits = self.ensemble_head(torch.cat((z1, z2, z1*z2), -1)).squeeze(-1)
            return F.binary_cross_entropy_with_logits(logits, batch["ensemble_label"].float(), reduction="none")
        if cfg.auxiliary == "cpc":
            # Exclude nearby same-sequence negatives; still no claim of eliminating all false negatives.
            order = batch["rank"].argsort(1)
            z = out["z"].gather(1, order[:, :, None].expand(-1, -1, cfg.dimension))
            targets = torch.arange(b, device=z.device)
            legal = self.negative_mask(batch)
            if not (legal.sum(1) > 1).any():
                raise ValueError("CPC batch has no valid distant negatives; use longer/more sequences")
            losses = []
            for i, head in enumerate(self.future_heads):
                q = F.normalize(head(out["context_z"]), dim=-1)
                key = F.normalize(z[:, i], dim=-1)
                logits = (q @ key.T / cfg.temperature).masked_fill(~legal, -1e4)
                losses.append(F.cross_entropy(logits, targets, reduction="none"))
            return torch.stack(losses).mean(0)
        if cfg.auxiliary == "scarf":
            z1 = F.normalize(self.contrast_project(self.local(batch["clean"], augment)), dim=-1)
            z2 = F.normalize(self.contrast_project(self.local(batch["corrupt"], augment)), dim=-1)
            legal = self.negative_mask(batch)
            if not (legal.sum(1) > 1).any():
                raise ValueError("Contrastive batch has no valid distant negatives")
            logits = (z1 @ z2.T / cfg.temperature).masked_fill(~legal, -1e4)
            target = torch.arange(b, device=z1.device)
            return .5*(F.cross_entropy(logits, target, reduction="none") +
                       F.cross_entropy(logits.T, target, reduction="none"))
        return out["z"].sum((1, 2))*0

    def negative_mask(self, batch):
        seq, start = batch["sequence"], batch["start"]
        legal = (seq[:, None] != seq[None, :]) | ((start[:, None]-start[None, :]).abs() >= self.cfg.max_span)
        return legal | torch.eye(len(seq), dtype=torch.bool, device=seq.device)

    def loss(self, batch, progress=0.0, augment=True):
        out = self(batch, augment=augment)
        cfg = self.cfg
        labels = batch["target_rank"].long()
        order = F.cross_entropy(out["position"].transpose(1, 2), labels, reduction="none").mean(1)
        earlier = (labels[:, self.pair_i] < labels[:, self.pair_j]).float()
        logits = out["score"][:, self.pair_i]-out["score"][:, self.pair_j]
        pair = F.binary_cross_entropy_with_logits(logits, earlier, reduction="none").mean(1)
        lag = F.cross_entropy(out["lag"].transpose(1, 2), self.lag_targets(batch["target_offsets"]),
                             reduction="none").mean(1)
        order_weight = 1 + (cfg.order_final_scale-1)*progress
        per = order_weight*(cfg.lambda_order*order + cfg.lambda_pair*pair + cfg.lambda_lag*lag)
        aux = self.aux_loss(batch, out, augment) if cfg.lambda_aux else per*0
        per = per + cfg.lambda_aux*aux
        if cfg.hard_fraction < 1 and augment:
            per = per.topk(max(1, math.ceil(len(per)*cfg.hard_fraction))).values
        return per.mean(), {"loss": float(per.detach().mean()), "order_ce": float(order.detach().mean()),
                            "pair_bce": float(pair.detach().mean()), "lag_ce": float(lag.detach().mean()),
                            "aux": float(aux.detach().mean())}

    def reset_heads(self, optimizer):
        """Reset parameters IN PLACE and discard their optimizer moments."""
        for head in (self.order_head, self.lag_head):
            for module in head.modules():
                if isinstance(module, (nn.Linear, nn.LayerNorm)):
                    module.reset_parameters()
                elif isinstance(module, nn.MultiheadAttention):
                    nn.init.xavier_uniform_(module.in_proj_weight)
                    if module.in_proj_bias is not None:
                        nn.init.zeros_(module.in_proj_bias)
            for p in head.parameters():
                optimizer.state.pop(p, None)


class SpanSampler:
    """Samples contiguous spans entirely inside explicitly supplied sequences.

    Input arrays are already scaled using FIT data statistics by the runner.
    Behavioral labels are never passed here. Sampling is uniform over valid start
    locations, so long sequences contribute proportionally. RNG state is resumable.
    """
    def __init__(self, sequences: Sequence[np.ndarray], cfg, seed=0):
        self.sequences = [np.asarray(x, np.float32) for x in sequences]
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        if not self.sequences or max(map(len, self.sequences)) < cfg.max_span:
            raise ValueError(f"Need a contiguous sequence with >= {cfg.max_span} bins for this configuration")

    def _locations(self, count, span):
        counts = np.asarray([max(0, len(x)-span+1) for x in self.sequences])
        cumulative = counts.cumsum()
        ids = self.rng.integers(0, cumulative[-1], count)
        seq = np.searchsorted(cumulative, ids, side="right")
        before = np.r_[0, cumulative[:-1]]
        return seq, ids-before[seq]

    def sample(self, batch_size, progress=1.0, device="cpu", scale=None):
        cfg, rng = self.cfg, self.rng
        scale = int(rng.choice(cfg.scales)) if scale is None else int(scale)
        w, c, k = cfg.window*scale, cfg.context*scale, cfg.tiles
        upper = cfg.gap_max
        if cfg.gap_curriculum:
            upper = round(cfg.gap_min + (cfg.gap_max-cfg.gap_min)*min(1, progress/cfg.gap_curriculum))
        span = c + k*(w+upper*scale)
        pool = cfg.selection_pool if cfg.span_selection != "random" else 1
        count = batch_size*pool
        seq, starts = self._locations(count, span)
        if pool > 1 and cfg.selection_jitter:
            # Candidates are neighboring spans around one random base, as in the teacher's sampler.
            seq = np.repeat(seq[::pool], pool)
            starts = np.repeat(starts[::pool], pool) + rng.integers(-cfg.selection_jitter, cfg.selection_jitter+1, count)
            starts = np.maximum(0, np.minimum(starts, np.array([len(self.sequences[s])-span for s in seq])))
        gaps = rng.integers(cfg.gap_min*scale, upper*scale+1, (count, k))
        offsets = c + np.cumsum(gaps, 1) + np.arange(k)*w
        contexts = np.stack([self.sequences[s][a:a+c] for s, a in zip(seq, starts)])
        tiles = np.stack([np.stack([self.sequences[s][a+o:a+o+w] for o in off])
                          for s, a, off in zip(seq, starts, offsets)])
        if pool > 1:
            levels = tiles.mean((2, 3))
            ii, jj = np.triu_indices(k, 1)
            delta = levels[:, jj]-levels[:, ii]
            concordance = ((delta > 0) + .5*(delta == 0)).mean(1).reshape(batch_size, pool)
            if cfg.span_selection == "decorrelated":
                pick = np.abs(concordance-.5).argmin(1)
            elif cfg.span_selection == "reverse_drift":
                pick = concordance.argmin(1)
            else:
                pick = np.abs(concordance-.5).argmax(1)
            keep = np.arange(batch_size)*pool+pick
            seq, starts, contexts, tiles, offsets = [a[keep] for a in (seq, starts, contexts, tiles, offsets)]
        clean = tiles[:, 0].copy()
        corrupt = clean.copy()
        mask = np.zeros((batch_size, cfg.channels), np.float32)
        shift_label = np.zeros_like(mask, dtype=np.int64)
        donor_windows = []
        if cfg.lambda_aux:
            for b, (sid, start, off) in enumerate(zip(seq, starts, offsets)):
                x = self.sequences[sid]
                here = int(start+off[0])
                # Whole-window replacement from SAME channel, SAME sequence, nonoverlapping time.
                separation = 2*w+cfg.gap_max*scale if cfg.auxiliary == "ensemble" else w
                left = max(0, here-separation+1)
                right_first = here+separation
                right = max(0, len(x)-w-right_first+1)
                if left+right == 0:
                    raise ValueError("Sequence too short for a nonoverlapping corruption donor")
                v = int(rng.integers(left+right))
                donor = v if v < left else right_first+v-left
                donor_windows.append(x[donor:donor+w])
                chosen = rng.choice(cfg.channels, max(1, round(cfg.channels*cfg.corruption_fraction)), replace=False)
                corrupt[b][:, chosen] = x[donor:donor+w][:, chosen]
                mask[b, chosen] = 1
                shift_label[b, chosen] = 1 if donor < here else 2
        permutation = np.argsort(rng.random((batch_size, k)), 1)
        batch = dict(context=contexts, tiles=tiles[np.arange(batch_size)[:, None], permutation],
                     rank=permutation.copy(), offsets=np.take_along_axis(offsets, permutation, 1),
                     sequence=seq, start=starts, clean=clean, corrupt=corrupt,
                     corruption_mask=mask, shift_label=shift_label)
        if cfg.view == "count_match":
            if batch["tiles"].min() < 0:
                raise ValueError("count_match requires nonnegative input; use --preprocess scale")
            # Deterministic total-count equalization; not Poisson resampling or an adversarial attack.
            total = batch["tiles"].sum(2, keepdims=True)
            batch["tiles"] *= total.min(1, keepdims=True)/np.maximum(total, 1e-6)
        labels = permutation.copy()
        if cfg.label_control != "true":
            if cfg.label_control == "fresh":
                chronology = np.argsort(rng.random((batch_size, k)), 1)
            else:
                chronology = np.stack([np.random.default_rng(np.random.SeedSequence(
                    [cfg.seed, int(s), int(a)])).permutation(k) for s, a in zip(seq, starts)])
            labels = np.take_along_axis(chronology, permutation, 1)
        batch["target_rank"] = labels
        batch["target_offsets"] = (batch["offsets"] if cfg.label_control == "true" else labels*(w+(cfg.gap_min+upper)*scale/2))
        if cfg.auxiliary == "ensemble":
            # Two disjoint random channel subsets, always the same masks for positive/negative cases.
            ch = rng.random((batch_size, cfg.channels)) < .5
            ch[:, 0] = True
            if cfg.channels > 1:
                ch[:, -1] = False
            label = rng.integers(0, 2, batch_size)
            next_window = tiles[:, 1]
            far = np.stack(donor_windows)
            batch.update(ensemble_a=clean*ch[:, None, :],
                         ensemble_b=np.where(label[:, None, None].astype(bool), next_window, far)*(~ch[:, None, :]),
                         ensemble_label=label)
        return {key: torch.as_tensor(np.ascontiguousarray(value), device=device) for key, value in batch.items()}


def assignment_ranks(position_logits):
    """Maximum-score bijection; independent per-tile argmax is not a permutation."""
    k = position_logits.shape[1]
    perm = torch.tensor(list(permutations(range(k))), device=position_logits.device)
    logp = position_logits.log_softmax(-1)
    scores = logp[:, torch.arange(k, device=logp.device)[None, :], perm].sum(-1)
    return perm[scores.argmax(1)]


def order_metrics(out, batch, cfg):
    """Only report heads actually trained. Scores are LARGE for EARLY tiles."""
    result = {}
    target = batch["rank"]
    i, j = torch.triu_indices(cfg.tiles, cfg.tiles, 1, device=target.device)
    if cfg.lambda_order:
        pred = assignment_ranks(out["position"])
        result.update(position_exact=float((pred == target).all(1).float().mean()),
                      position_pair=float(((pred[:, i] < pred[:, j]) == (target[:, i] < target[:, j])).float().mean()))
    if cfg.lambda_pair:
        delta = out["score"][:, i]-out["score"][:, j]
        expected = target[:, i] < target[:, j]
        result["score_pair"] = float((((delta > 0) == expected).float()*(delta != 0) + .5*(delta == 0)).mean())
        # Sort descending; argsort(argsort(-score)) is the predicted chronological rank.
        pred = (-out["score"]).argsort(1).argsort(1)
        result["score_exact"] = float((pred == target).all(1).float().mean())
    return result


def effective_dimension(z):
    z = np.asarray(z, np.float64)
    s = np.linalg.eigvalsh(np.atleast_2d(np.cov(z, rowvar=False))).clip(0)
    return float(s.sum()**2 / max(float((s*s).sum()), 1e-20))

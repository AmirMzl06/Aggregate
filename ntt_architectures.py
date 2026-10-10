"""Neural Transformation Tournament: encoders, binned transforms, and objectives.

Requires Python >=3.10 and PyTorch >=2.1. No CEBRA dependency.
All encoders consume (batch, neurons, time) and expose h (representation) and
unit-normalized z (projection). Every view uses exactly the same encoder.

Architectures: mlp, tcn, inception, transformer, neuron_attention,
               rate_temporal, spectral_tcn.
These are alternative experimental backbones, not established new methods.

IMPORTANT: inputs from Perich NPZ files are already binned. ``bin_permute``
moves whole bins and preserves each neuron's values and counts inside each
coarse block; it is NOT sub-bin spike-time jitter. ``count_jitter`` moves
individual integer count events at bin resolution and requires raw counts.
``neuron_shift`` is a circular, per-neuron shift, provided as an artifact-prone
comparison only. Increased nominal strength does not mathematically guarantee
increased biological damage or greater distance for each realized sample.
No behavior labels, trial positions, or corruption strengths enter the encoder.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ARCHITECTURES = (
    "mlp", "tcn", "inception", "transformer", "neuron_attention",
    "rate_temporal", "spectral_tcn",
)
OBJECTIVES = (
    "rank", "rank_shuffled", "consistency", "reg_only", "rank_only",
    "rank_consistency",
)
TRANSFORMS = ("bin_permute", "count_jitter", "neuron_shift")


@dataclass
class EncoderConfig:
    n_neurons: int
    window: int = 36
    center: int = 18
    architecture: str = "tcn"
    width: int = 64
    latent_dim: int = 64
    projection_dim: int = 64
    depth: int = 3
    heads: int = 4
    dropout: float = 0.0
    pooling: str = "center_mean"

    def validate(self):
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        if min(self.n_neurons, self.width, self.latent_dim,
               self.projection_dim, self.depth, self.heads) < 1:
            raise ValueError("All architecture dimensions must be positive.")
        if self.window < 2 or not 0 <= self.center < self.window:
            raise ValueError("Invalid window or anchor position.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if self.pooling not in ("center", "mean", "center_mean", "attention"):
            raise ValueError("Unknown pooling method.")
        if self.architecture in ("transformer", "neuron_attention"):
            if self.width % self.heads:
                raise ValueError("width must be divisible by attention heads.")
        return self


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels at each timestamp; no batch-statistic leakage."""
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TemporalPool(nn.Module):
    def __init__(self, width, center, mode):
        super().__init__()
        self.center, self.mode = center, mode
        self.score = nn.Conv1d(width, 1, 1) if mode == "attention" else None
        self.merge = nn.Linear(2 * width, width) if mode == "center_mean" else nn.Identity()

    def forward(self, x):
        if self.mode == "center":
            return x[:, :, self.center]
        if self.mode == "mean":
            return x.mean(-1)
        if self.mode == "attention":
            return (x * self.score(x).softmax(-1)).sum(-1)
        return self.merge(torch.cat((x[:, :, self.center], x.mean(-1)), dim=-1))


class TemporalResidual(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            ChannelLayerNorm(width), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation),
            ChannelLayerNorm(width), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
        )

    def forward(self, x):
        return x + self.net(x)


class TemporalTrunk(nn.Module):
    def __init__(self, c: EncoderConfig):
        super().__init__()
        self.stem = nn.Conv1d(c.n_neurons, c.width, 1)
        self.blocks = nn.Sequential(*[
            TemporalResidual(c.width, 2 ** i, c.dropout) for i in range(c.depth)
        ])
        self.pool = TemporalPool(c.width, c.center, c.pooling)

    def forward(self, x):
        return self.pool(self.blocks(self.stem(x)))


class InceptionResidual(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        branch_width = max(4, width // 4)
        self.norm = ChannelLayerNorm(width)
        self.branches = nn.ModuleList([
            nn.Conv1d(width, branch_width, k, padding=k // 2) for k in (3, 7, 11)
        ])
        self.pool_branch = nn.Sequential(
            nn.AvgPool1d(3, stride=1, padding=1), nn.Conv1d(width, branch_width, 1))
        self.merge = nn.Sequential(nn.GELU(), nn.Dropout(dropout),
                                   nn.Conv1d(4 * branch_width, width, 1))

    def forward(self, x):
        y = self.norm(x)
        features = [branch(y) for branch in self.branches] + [self.pool_branch(y)]
        return x + self.merge(torch.cat(features, dim=1))


class InceptionTrunk(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.stem = nn.Conv1d(c.n_neurons, c.width, 1)
        self.blocks = nn.Sequential(*[InceptionResidual(c.width, c.dropout)
                                      for _ in range(c.depth)])
        self.pool = TemporalPool(c.width, c.center, c.pooling)

    def forward(self, x):
        return self.pool(self.blocks(self.stem(x)))


def attention_stack(c):
    layer = nn.TransformerEncoderLayer(
        d_model=c.width, nhead=c.heads, dim_feedforward=4 * c.width,
        dropout=c.dropout, activation="gelu", batch_first=True, norm_first=True)
    stack = nn.TransformerEncoder(layer, num_layers=c.depth,
                                  norm=nn.LayerNorm(c.width), enable_nested_tensor=False)
    # PyTorch clones the initial layer. Explicitly initialize each matrix
    # independently rather than starting all layers from identical weights.
    for block in stack.layers:
        for name, parameter in block.named_parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
            elif name.endswith("bias"):
                nn.init.zeros_(parameter)
    return stack


class TransformerTrunk(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.stem = nn.Linear(c.n_neurons, c.width)
        self.position = nn.Parameter(torch.randn(1, c.window, c.width) * 0.02)
        self.layers = attention_stack(c)
        self.pool = TemporalPool(c.width, c.center, c.pooling)

    def forward(self, x):
        y = self.layers(self.stem(x.transpose(1, 2)) + self.position)
        return self.pool(y.transpose(1, 2))


class NeuronAttentionTrunk(nn.Module):
    """A temporal patch per neuron; attention across neurons with fixed identities.

    This model is session-specific. It does NOT claim zero-shot generalization
    to a different number/order of neurons. Attention costs O(N_neurons**2).
    """
    def __init__(self, c):
        super().__init__()
        self.temporal = nn.Sequential(nn.Linear(c.window, c.width), nn.GELU(),
                                      nn.Linear(c.width, c.width))
        self.identity = nn.Parameter(torch.randn(1, c.n_neurons, c.width) * 0.02)
        self.layers = attention_stack(c)
        self.score = nn.Linear(c.width, 1)

    def forward(self, x):
        tokens = self.layers(self.temporal(x) + self.identity)
        return (tokens * self.score(tokens).softmax(dim=1)).sum(dim=1)


class RateTemporalTrunk(nn.Module):
    """Separate window-rate statistics and demeaned temporal structure, then fuse.

    Architectural separation is not a claim of statistical disentanglement.
    """
    def __init__(self, c):
        super().__init__()
        self.temporal = TemporalTrunk(c)
        self.rate = nn.Sequential(nn.Linear(2 * c.n_neurons, c.width), nn.GELU(),
                                  nn.Linear(c.width, c.width))
        self.fuse = nn.Sequential(nn.Linear(2 * c.width, c.width), nn.GELU())

    def forward(self, x):
        mean = x.mean(-1)
        stats = torch.cat((mean, x.var(-1, unbiased=False).add(1e-6).sqrt()), dim=-1)
        return self.fuse(torch.cat((self.temporal(x - mean.unsqueeze(-1)),
                                    self.rate(stats)), dim=-1))


class SpectralTemporalTrunk(nn.Module):
    """Temporal branch + log-power branch. Useful also as a spectrum-shortcut check."""
    def __init__(self, c):
        super().__init__()
        self.temporal = TemporalTrunk(c)
        self.spectral = nn.Sequential(
            nn.Linear(c.n_neurons * (c.window // 2 + 1), c.width), nn.GELU(),
            nn.Linear(c.width, c.width))
        self.fuse = nn.Linear(2 * c.width, c.width)

    def forward(self, x):
        power = torch.fft.rfft(x, dim=-1, norm="ortho").abs().square().log1p()
        return self.fuse(torch.cat((self.temporal(x), self.spectral(power.flatten(1))), -1))


class TournamentEncoder(nn.Module):
    def __init__(self, config: EncoderConfig, mean=None, std=None):
        super().__init__()
        self.config = config.validate()
        c = config
        mean = torch.zeros(c.n_neurons) if mean is None else torch.as_tensor(mean).float()
        std = torch.ones(c.n_neurons) if std is None else torch.as_tensor(std).float()
        if mean.shape != (c.n_neurons,) or std.shape != mean.shape:
            raise ValueError("Invalid input-normalization shapes.")
        self.register_buffer("input_mean", mean.reshape(1, -1, 1).clone())
        self.register_buffer("input_std", std.clamp_min(1e-6).reshape(1, -1, 1).clone())
        if c.architecture == "mlp":
            self.backbone = nn.Sequential(
                nn.Flatten(), nn.Linear(c.n_neurons * c.window, 2 * c.width),
                nn.LayerNorm(2 * c.width), nn.GELU(), nn.Dropout(c.dropout),
                nn.Linear(2 * c.width, c.width), nn.GELU())
        else:
            constructor = dict(tcn=TemporalTrunk, inception=InceptionTrunk,
                               transformer=TransformerTrunk,
                               neuron_attention=NeuronAttentionTrunk,
                               rate_temporal=RateTemporalTrunk,
                               spectral_tcn=SpectralTemporalTrunk)[c.architecture]
            self.backbone = constructor(c)
        self.representation = nn.Sequential(nn.LayerNorm(c.width), nn.GELU(),
                                             nn.Linear(c.width, c.latent_dim))
        self.projector = nn.Sequential(nn.Linear(c.latent_dim, c.width), nn.GELU(),
                                       nn.Linear(c.width, c.projection_dim))

    def forward(self, x: Tensor):
        if x.ndim != 3 or x.shape[1:] != (self.config.n_neurons, self.config.window):
            raise ValueError(f"Expected B,N,T with N={self.config.n_neurons}, "
                             f"T={self.config.window}; got {tuple(x.shape)}")
        h = self.representation(self.backbone((x - self.input_mean) / self.input_std))
        z = F.normalize(self.projector(h), dim=-1, eps=1e-8)
        return {"h": h, "z": z}

    def config_dict(self):
        return asdict(self.config)


@dataclass
class TransformConfig:
    mode: str = "bin_permute"
    levels: tuple[float, ...] = (1.0, 2.0, 4.0)
    block_size: int = 12
    max_events: int = 2_000_000

    def validate(self):
        if self.mode not in TRANSFORMS:
            raise ValueError(f"Unknown transform {self.mode}")
        if len(self.levels) < 2 or any(not 0 < v < float("inf") for v in self.levels):
            raise ValueError("Need at least two finite, positive corruption levels.")
        if any(a >= b for a, b in zip(self.levels, self.levels[1:])):
            raise ValueError("Corruption levels must be strictly increasing.")
        if self.block_size < 2 or self.max_events < 1:
            raise ValueError("Invalid block size or event budget.")
        return self


@torch.no_grad()
def make_tournament(x: Tensor, config: TransformConfig,
                    generator: torch.Generator | None = None,
                    shared_across_neurons: bool = False):
    """Return V,B,N,T views (clean first) and measured perturbation diagnostics.

    Use a generator on the same device as x. Common random numbers couple the
    strength levels, but DO NOT guarantee monotonic realized damage. Default
    runner transforms CPU windows before device transfer. The clean input is
    never changed in place. ``shared_across_neurons`` is an evaluation control
    for bin_permute/neuron_shift, not used as a label in encoder training.
    """
    config.validate()
    if x.ndim != 3 or not torch.isfinite(x).all():
        raise ValueError("Transforms require finite B,N,T input.")
    b, n, t = x.shape
    levels = config.levels
    displacement = torch.zeros(len(levels), b, device=x.device)
    outputs = [x.clone() for _ in levels]
    if config.mode == "bin_permute":
        for start in range(0, t, config.block_size):
            length = min(config.block_size, t - start)
            positions = torch.arange(length, device=x.device, dtype=x.dtype)
            noise = torch.randn(b, 1 if shared_across_neurons else n, length,
                                device=x.device, generator=generator)
            for k, level in enumerate(levels):
                order = (positions + level * noise).argsort(-1).expand(b, n, length)
                outputs[k][:, :, start:start + length] = x[:, :, start:start + length].gather(-1, order)
                displacement[k] += (order - positions).abs().float().mean((1, 2)) * length / t
    elif config.mode == "neuron_shift":
        if max(levels) >= t / 2:
            raise ValueError("Circular shifts must be smaller than half the window.")
        noise = torch.rand(b, 1 if shared_across_neurons else n, 1,
                           device=x.device, generator=generator) * 2 - 1
        positions = torch.arange(t, device=x.device).view(1, 1, t)
        for k, level in enumerate(levels):
            shift = (noise * level).round().long().expand(b, n, 1)
            outputs[k] = x.gather(-1, (positions + shift) % t)
            displacement[k] = shift.abs().float().mean((1, 2))
    else:
        if shared_across_neurons:
            raise ValueError("A shared per-event jitter is not defined across unequal spike trains.")
        if (x < 0).any() or not torch.allclose(x, x.round(), atol=1e-5, rtol=0):
            raise ValueError("count_jitter requires raw, nonnegative integer counts, not rates/z-scores.")
        counts = x.round().long().flatten()
        total = int(counts.sum().item())
        if total > config.max_events:
            raise ValueError(f"{total} events exceeds max_events={config.max_events}; reduce batch size.")
        sources = torch.repeat_interleave(torch.arange(counts.numel(), device=x.device), counts)
        time_index = sources % t
        block_start = (time_index // config.block_size) * config.block_size
        length = (t - block_start).clamp_max(config.block_size)
        offset = time_index - block_start
        noise = torch.randn(total, device=x.device, generator=generator)
        event_batch = sources // (n * t)
        events_per_batch = torch.bincount(event_batch, minlength=b).clamp_min(1)
        for k, level in enumerate(levels):
            shifted = offset + (level * noise).round().long()
            period = (2 * (length - 1)).clamp_min(1)
            reflected = shifted.remainder(period)
            reflected = torch.minimum(reflected, period - reflected)
            reflected = torch.where(length > 1, reflected, torch.zeros_like(reflected))
            target_time = block_start + reflected
            targets = sources - time_index + target_time
            flat = torch.zeros_like(x).flatten()
            flat.scatter_add_(0, targets, torch.ones_like(targets, dtype=x.dtype))
            outputs[k] = flat.reshape_as(x)
            disp = torch.zeros(b, device=x.device)
            disp.scatter_add_(0, event_batch, (target_time - time_index).abs().float())
            displacement[k] = disp / events_per_batch
    views = torch.stack([x, *outputs], dim=0)
    difference = views[1:] - x.unsqueeze(0)
    diagnostics = {
        "nominal_levels": list(levels),
        "mean_abs_change": difference.abs().mean((2, 3)),  # K,B
        "rms_change": difference.square().mean((2, 3)).sqrt(),
        "changed_fraction": (difference != 0).float().mean((2, 3)),
        "mean_displacement_bins": displacement,
        "max_count_error": float((views.sum(-1) - x.sum(-1)).abs().max()),
    }
    return views, diagnostics


def variance_covariance(h: Tensor, target_std: float = 1.0):
    if len(h) < 2:
        raise ValueError("Variance regularization needs at least two samples.")
    centered = h - h.mean(0)
    std = (centered.square().mean(0) + 1e-4).sqrt()
    variance = F.relu(target_std - std).mean()
    cov = centered.T @ centered / (len(h) - 1)
    off_diag = cov - torch.diag_embed(torch.diagonal(cov))
    covariance = off_diag.square().sum() / h.shape[1]
    return variance, covariance


def tournament_loss(h: Tensor, z: Tensor, levels: Sequence[float], objective="rank",
                    margin=0.15, variance_weight=1.0, covariance_weight=0.01,
                    consistency_weight=1.0, ordering=None):
    """h,z have V,B,D shape; view 0 is clean. ``ordering`` is B,K for shuffled control.

    All corrupted pairs participate. Hinge margins are scaled by nominal level
    separation. Unit z removes a trivial global-norm inflation solution.
    Regularizers operate on CLEAN h, preventing easy variation solely from
    corruption severity. rank_shuffled keeps transforms and regularizers fixed.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective {objective}")
    k = len(levels)
    if z.shape[0] != k + 1 or h.shape[:2] != z.shape[:2]:
        raise ValueError("Mismatched tournament shapes.")
    distances = (z[1:] - z[0]).square().sum(-1).transpose(0, 1)  # B,K
    ordered = distances
    if objective == "rank_shuffled":
        if ordering is None or ordering.shape != distances.shape:
            raise ValueError("rank_shuffled requires a fresh B,K random permutation.")
        ordered = distances.gather(1, ordering)
    scale = max(levels[-1] - levels[0], 1e-8)
    terms = [F.relu(margin * (levels[j] - levels[i]) / scale + ordered[:, i] - ordered[:, j])
             for i, j in combinations(range(k), 2)]
    rank = torch.stack(terms).mean()
    consistency = (z[0] - z[1]).square().sum(-1).mean()
    variance, covariance = variance_covariance(h[0])
    zero = h.sum() * 0
    total = rank if objective in ("rank", "rank_shuffled", "rank_only", "rank_consistency") else zero
    if objective in ("consistency", "rank_consistency"):
        total = total + consistency_weight * consistency
    if objective != "rank_only":
        total = total + variance_weight * variance + covariance_weight * covariance
    logs = {"loss": float(total.detach()), "rank_loss": float(rank.detach()),
            "consistency_loss": float(consistency.detach()),
            "variance_loss": float(variance.detach()), "covariance_loss": float(covariance.detach())}
    return total, logs


@torch.no_grad()
def representation_diagnostics(values: Tensor):
    x = values.float()
    centered = x - x.mean(0)
    std = centered.std(0, unbiased=False)
    cov = centered.T @ centered / max(1, len(x) - 1)
    spectrum = torch.linalg.eigvalsh(cov).clamp_min(0)
    if float(spectrum.sum()) <= 1e-12:
        effective_rank = participation = 0.0
    else:
        prob = spectrum / spectrum.sum()
        effective_rank = float((-(prob * prob.clamp_min(1e-12).log()).sum()).exp())
        participation = float(spectrum.sum().square() / spectrum.square().sum().clamp_min(1e-12))
    norms = x.norm(dim=-1)
    pairs = F.normalize(x[:512], dim=-1, eps=1e-8)
    if len(pairs) > 1:
        sims = pairs @ pairs.T
        mean_cos = float((sims.sum() - sims.diag().sum()) / (len(pairs) * (len(pairs) - 1)))
    else:
        mean_cos = 0.0
    return dict(n_samples=len(x), dimension=x.shape[1], mean_std=float(std.mean()),
                min_std=float(std.min()), inactive_fraction=float((std < 1e-3).float().mean()),
                effective_rank=effective_rank, participation_ratio=participation,
                mean_norm=float(norms.mean()), mean_pair_cosine=mean_cos,
                covariance_eigenvalues=spectrum.cpu().tolist())


@torch.no_grad()
def ranking_diagnostics(distances: Tensor):
    """B,K distances in known nominal order; count ties separately, never as success."""
    x = distances.float()
    k = x.shape[1]
    differences = torch.stack([x[:, j] - x[:, i] for i, j in combinations(range(k), 2)], 1)
    tol = 1e-7
    wins, ties = differences > tol, differences.abs() <= tol
    return dict(pair_accuracy=float(wins.float().mean()), tie_fraction=float(ties.float().mean()),
                pair_accuracy_ties_half=float((wins.float() + 0.5 * ties.float()).mean()),
                full_order_accuracy=float(((x[:, 1:] - x[:, :-1]) > tol).all(1).float().mean()),
                mean_distance_by_level=x.mean(0).tolist(),
                mean_gap=float(differences.mean()), random_pair_chance=0.5)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


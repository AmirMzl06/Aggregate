"""MobileJigsaw: the same cross-entropy jigsaw, on a real MobileNet trunk.

Subclasses JigsawNet, so the losses, training loop, transform, evaluation and
checkpointing are LITERALLY THE SAME CODE. Only the trunk differs. That is the
point: any difference in the result is attributable to the architecture and
nothing else -- same seed, same tiles, same order CE, same forecast CE.

Still cross-entropy only. No contrastive loss. No CEBRA.

    from mobile_jigsaw import MobileJigsaw
    model = MobileJigsaw(version="v2", width_multiplier=1.0, stem="per_neuron")
    model.fit(X_train, X_valid=X_valid)
    Z = model.transform(X_valid)
    print(model.summary())                 # per-layer shapes and parameter counts

WHAT "MOBILENET" ACTUALLY MEANS HERE
  version="v1"  Howard et al. 2017. A stack of depthwise-separable blocks:
                depthwise k=3 -> norm -> act -> pointwise 1x1 -> norm -> act.
  version="v2"  Sandler et al. 2018. Inverted residual with a LINEAR
                bottleneck: 1x1 expand -> norm -> act -> depthwise k=3 -> norm
                -> act -> 1x1 project -> norm, NO activation on the projection,
                plus a residual connection.
  version="v3"  Howard et al. 2019. V2 block plus squeeze-excitation channel
                gating and hard-swish.
Two genuine MobileNet knobs are exposed: width_multiplier (alpha) and expansion
(t). Channel counts are rounded to a multiple of 8 by the paper's
_make_divisible rule.

THE HONEST CASE FOR TRYING IT, AND THE PART THAT IS OVERSOLD
  Oversold: "MobileNet is robust." Its design target is FLOPs and parameters on
  ImageNet, not robustness to shortcut learning. A shortcut that is linearly
  available to a plain conv is just as available to a depthwise one. Expect
  tile_norm, not the trunk, to control shortcut exploitation.

  Also oversold in 1D specifically: the famous 1/9 parameter saving is for 2D
  k=3x3. In 1D with k=3 a separable block is only ~1/3 of a full conv, and an
  inverted residual at the paper's t=6 is about 4x MORE parameters than the
  plain conv it replaces (C=64: 50304 vs 12288). So t=6 does not make this
  model smaller, it makes it bigger. Default here is expansion=3, and
  version="v1" is the only variant that is actually leaner than the baseline.
  Read the parameter count printed by summary() before claiming efficiency.

  Genuinely worth testing, three reasons:
  1. LINEAR BOTTLENECK. MobileNetV2's central argument is that ReLU destroys
     information in low-dimensional spaces, so the projection must be linear.
     That is exactly the failure we measured: a narrow embedding losing the
     firing-rate structure that raw-window ridge exploits. V2 is the one
     mainstream architecture designed around this specific concern.
  2. PER-NEURON TEMPORAL FILTERS. With stem="per_neuron" the first depthwise
     conv has groups=n_neurons, so every neuron gets its own temporal kernel
     and no cross-neuron mixing happens until the pointwise layer. This is
     EEGNet's inductive bias and it fits heterogeneous neural timescales.
     stem="mix" is the image-style alternative that mixes neurons immediately.
  3. Parameter efficiency AS REGULARIZATION on short sessions, but only at
     version="v1" or expansion<=2 -- see above.

NORMALIZATION: WHY GroupNorm IS THE DEFAULT AND BatchNorm IS A TRAP HERE
  Canonical MobileNet uses BatchNorm everywhere. In this training loop the
  trunk sees TWO distributions -- the raw augmented view (forecast branch) and
  the tile-normalized view (order branch) -- while transform sees a third: raw
  natural windows, no augmentation. BatchNorm's running statistics would be a
  blend of the first two and then get applied to the third, so the embedding
  you decode is computed with the wrong statistics.
  norm="group" has no running statistics and no train/eval discrepancy.
  norm="batch" is available as an ablation; measure_train_eval_gap() quantifies
  the damage so you can see it rather than take my word for it.

Self-test:  python mobile_jigsaw.py --self-test
Compare:    python mobile_jigsaw.py --compare
"""
import argparse
import math

import numpy as np
import torch
from torch import nn

from jigsaw_net import (JigsawNet, _ridge_r2, _synthetic, _check, _integer, _real,
                        _boolean, _sequences)

MOBILE_VERSIONS = ("v1", "v2", "v3")
STEMS = ("per_neuron", "mix")
NORMS = ("group", "batch", "none")
ACTIVATIONS = ("relu6", "hardswish", "gelu")


# --------------------------------------------------------------------------- #
# MobileNet primitives
# --------------------------------------------------------------------------- #
def _make_divisible(value, divisor=8, minimum=None):
    """MobileNet's channel rounding: nearest multiple of 8, never below 90%."""
    minimum = divisor if minimum is None else minimum
    rounded = max(minimum, int(value + divisor / 2) // divisor * divisor)
    return rounded + divisor if rounded < 0.9 * value else rounded


class _HardSigmoid(nn.Module):
    """ReLU6(x + 3) / 6, MobileNetV3's cheap sigmoid."""

    def forward(self, x):
        return torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


class _HardSwish(nn.Module):
    """x * ReLU6(x + 3) / 6, MobileNetV3's cheap swish."""

    def forward(self, x):
        return x * torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


def _activation(kind):
    if kind == "relu6":
        return nn.ReLU6()
    if kind == "hardswish":
        return _HardSwish()
    if kind == "gelu":
        return nn.GELU()
    raise ValueError(f"activation must be one of {ACTIVATIONS}.")


def _norm_layer(kind, channels, groups=8):
    """GroupNorm by default: no running statistics, so train == eval."""
    if kind == "group":
        divisor = math.gcd(channels, groups)
        return nn.GroupNorm(max(1, divisor), channels)
    if kind == "batch":
        return nn.BatchNorm1d(channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"norm must be one of {NORMS}.")


class _SqueezeExcite(nn.Module):
    """MobileNetV3 channel gating. On neural data this is attention over feature
    maps: it can learn to suppress neurons that only carry nuisance drift."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = _make_divisible(channels / reduction, 8)
        self.gate = nn.Sequential(nn.Conv1d(channels, hidden, 1), nn.ReLU(),
                                  nn.Conv1d(hidden, channels, 1), _HardSigmoid())

    def forward(self, x):
        return x * self.gate(x.mean(dim=-1, keepdim=True))


class _SeparableBlock(nn.Module):
    """MobileNetV1 block. VALID depthwise k=3, so it consumes exactly two bins."""

    def __init__(self, in_channels, out_channels, norm, activation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(in_channels, in_channels, 3, groups=in_channels, bias=False),
            _norm_layer(norm, in_channels), _activation(activation),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            _norm_layer(norm, out_channels), _activation(activation),
        )

    def forward(self, x):
        return self.net(x)


class _InvertedResidual(nn.Module):
    """MobileNetV2/V3 block: expand 1x1, VALID depthwise k=3, project 1x1 LINEAR.

    The projection has no activation on purpose. That is the paper's linear
    bottleneck: a ReLU on a narrow layer collapses part of the manifold, and
    here the manifold we must not collapse is the one carrying firing rate.

    Consumes exactly two bins, like _SeparableBlock, so the receptive-field
    arithmetic is identical across versions and the comparison stays fair.
    The residual branch crops the input to match.
    """

    def __init__(self, in_channels, out_channels, expansion, norm, activation,
                 dropout, squeeze_excite=False):
        super().__init__()
        hidden = _make_divisible(in_channels * expansion, 8)
        self.residual = in_channels == out_channels
        layers = [nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()]
        if hidden != in_channels:
            layers += [nn.Conv1d(in_channels, hidden, 1, bias=False),
                       _norm_layer(norm, hidden), _activation(activation)]
        layers += [nn.Conv1d(hidden, hidden, 3, groups=hidden, bias=False),
                   _norm_layer(norm, hidden), _activation(activation)]
        if squeeze_excite:
            layers.append(_SqueezeExcite(hidden))
        layers += [nn.Conv1d(hidden, out_channels, 1, bias=False),
                   _norm_layer(norm, out_channels)]          # linear: no activation
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out + x[..., 1:-1] if self.residual else out


class _MobileTrunk(nn.Module):
    """MobileNet trunk whose receptive field is exactly window_size.

    stem consumes (kernel - 1) bins, every block consumes 2, and the widths are
    (kernel, blocks) = (2, (W-2)/2) for even W and (3, (W-3)/2) for odd W, so
    the output length is always exactly 1. Verified for W = 4..40.
    """

    def __init__(self, channels, window_size, base_width,
                 version="v2", width_multiplier=1.0, expansion=3, stem="per_neuron",
                 norm="group", activation="relu6", dropout=0.0, head_channels=0,
                 squeeze_excite=None):
        super().__init__()
        if version not in MOBILE_VERSIONS:
            raise ValueError(f"version must be one of {MOBILE_VERSIONS}.")
        if stem not in STEMS:
            raise ValueError(f"stem must be one of {STEMS}.")
        kernel = 2 if window_size % 2 == 0 else 3
        n_blocks = (window_size - kernel) // 2
        if n_blocks < 1:
            raise ValueError(f"window_size={window_size} is too small for a MobileNet trunk; "
                             "need >= 4.")
        squeeze_excite = (version == "v3") if squeeze_excite is None else squeeze_excite
        width = _make_divisible(base_width * width_multiplier, 8)

        # ---- stem ---------------------------------------------------------- #
        if stem == "per_neuron":
            # groups=channels: every NEURON gets its own temporal filter and
            # nothing mixes across neurons until the pointwise below. EEGNet.
            self.stem = nn.Sequential(
                nn.Conv1d(channels, channels, kernel, groups=channels, bias=False),
                _norm_layer(norm, channels), _activation(activation),
                nn.Conv1d(channels, width, 1, bias=False),
                _norm_layer(norm, width), _activation(activation))
        else:
            # image-style: mix neurons immediately with a full conv.
            self.stem = nn.Sequential(
                nn.Conv1d(channels, width, kernel, bias=False),
                _norm_layer(norm, width), _activation(activation))

        # ---- body ---------------------------------------------------------- #
        blocks, in_width, self.plan = [], width, []
        for i in range(n_blocks):
            out_width = _make_divisible(base_width * width_multiplier
                                        * min(2 ** (i // 2), 4), 8)
            if version == "v1":
                blocks.append(_SeparableBlock(in_width, out_width, norm, activation, dropout))
            else:
                blocks.append(_InvertedResidual(in_width, out_width, expansion, norm,
                                                activation, dropout, squeeze_excite))
            self.plan.append((in_width, out_width))
            in_width = out_width
        self.blocks = nn.Sequential(*blocks)

        # ---- last pointwise, MobileNet's final 1x1 before the classifier ---- #
        if head_channels:
            final = _make_divisible(head_channels * width_multiplier, 8)
            self.head = nn.Sequential(nn.Conv1d(in_width, final, 1, bias=False),
                                      _norm_layer(norm, final), _activation(activation))
            in_width = final
        else:
            self.head = nn.Identity()
        self.out_features = in_width
        self.kernel, self.n_blocks = kernel, n_blocks

    def forward(self, x):
        h = self.head(self.blocks(self.stem(x)))
        if h.shape[-1] != 1:
            raise ValueError(f"MobileNet trunk expects exactly window_size bins; "
                             f"got length {h.shape[-1]} at the output.")
        return h.squeeze(-1)


class _MobileEncoder(nn.Module):
    def __init__(self, trunk, output_dimension, normalize):
        super().__init__()
        self.trunk = trunk
        self.project = nn.Linear(trunk.out_features, output_dimension)
        self.normalize = normalize

    def forward(self, x):
        z = self.project(self.trunk(x))
        return nn.functional.normalize(z, dim=-1) if self.normalize else z


# --------------------------------------------------------------------------- #
# estimator
# --------------------------------------------------------------------------- #
class MobileJigsaw(JigsawNet):
    """JigsawNet with a MobileNet trunk. Identical losses, identical everything else."""

    _MODEL_TYPE = "mobile_jigsaw"
    _PARAM_NAMES = JigsawNet._PARAM_NAMES + (
        "version", "width_multiplier", "expansion", "stem", "norm", "activation",
        "head_channels", "squeeze_excite")

    def __init__(self, *, version="v2", width_multiplier=1.0, expansion=3,
                 stem="per_neuron", norm="group", activation="relu6",
                 head_channels=0, squeeze_excite=None, **kwargs):
        super().__init__(**kwargs)
        if version not in MOBILE_VERSIONS:
            raise ValueError(f"version must be one of {MOBILE_VERSIONS}.")
        if stem not in STEMS:
            raise ValueError(f"stem must be one of {STEMS}.")
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}.")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}.")
        if squeeze_excite is not None:
            _boolean("squeeze_excite", squeeze_excite)
        self.version = version
        self.width_multiplier = _real("width_multiplier", width_multiplier, 0, strict_min=True)
        self.expansion = _real("expansion", expansion, 1)
        self.stem = stem
        self.norm = norm
        self.activation = activation
        self.head_channels = _integer("head_channels", head_channels, 0)
        self.squeeze_excite = squeeze_excite
        if self.window_size < 4:
            raise ValueError("window_size must be >= 4 for a MobileNet trunk.")

    def _make_encoder(self, channels):
        uses_se = self.version == "v3" if self.squeeze_excite is None else self.squeeze_excite
        trunk = _MobileTrunk(channels, self.window_size, self.num_hidden_units,
                             version=self.version,
                             width_multiplier=self.width_multiplier,
                             expansion=self.expansion, stem=self.stem, norm=self.norm,
                             activation=self.activation, dropout=self.dropout,
                             head_channels=self.head_channels,
                             squeeze_excite=self.squeeze_excite)
        if self.verbose:
            print(f"MobileJigsaw trunk: {self.version}, alpha={self.width_multiplier}, "
                  f"t={self.expansion}, stem={self.stem}, norm={self.norm}, "
                  f"act={self.activation}, squeeze_excite={uses_se} | "
                  f"stem kernel={trunk.kernel}, {trunk.n_blocks} blocks "
                  f"{[o for _, o in trunk.plan]} -> {trunk.out_features} "
                  f"-> {self.output_dimension} | trunk params "
                  f"{sum(p.numel() for p in trunk.parameters()):,}", flush=True)
        return _MobileEncoder(trunk, self.output_dimension, self.normalize)

    # -- introspection ----------------------------------------------------- #
    def n_parameters(self):
        self._check_fitted()
        return dict(trunk=sum(p.numel() for p in self.encoder_.trunk.parameters()),
                    project=sum(p.numel() for p in self.encoder_.project.parameters()),
                    order_head=sum(p.numel() for p in self.order_head_.parameters()),
                    forecast_head=sum(p.numel() for p in self.forecast_head_.parameters()),
                    reconstruct_head=sum(p.numel()
                                         for p in self.reconstruct_head_.parameters()))

    def summary(self):
        """Per-stage shapes and parameter counts. Run this before claiming
        anything about efficiency."""
        self._check_fitted()
        lines = [f"MobileJigsaw {self.version}  alpha={self.width_multiplier} "
                 f"t={self.expansion} stem={self.stem} norm={self.norm}",
                 f"  input (1, {self.n_features_in_}, {self.window_size})"]
        x = torch.zeros(2, self.n_features_in_, self.window_size, device=self.device_)
        trunk = self.encoder_.trunk
        was_training = self.encoder_.training
        self.encoder_.eval()
        with torch.no_grad():
            for name, module in (("stem", trunk.stem), *((f"block{i}", b) for i, b in
                                                         enumerate(trunk.blocks)),
                                 ("head", trunk.head)):
                x = module(x)
                count = sum(p.numel() for p in module.parameters())
                lines.append(f"  {name:<9} -> ({x.shape[1]:>4}, {x.shape[2]:>2})"
                             f"   {count:>9,} params")
        if was_training:
            self.encoder_.train()
        counts = self.n_parameters()
        lines.append(f"  project   -> ({self.output_dimension},)"
                     f"      {counts['project']:>9,} params")
        lines.append(f"  TRUNK TOTAL {counts['trunk']:,} | heads "
                     f"{counts['order_head'] + counts['forecast_head']:,} "
                     f"(heads are discarded at transform time)")
        return "\n".join(lines)

    def measure_train_eval_gap(self, X, *, n=512):
        """How much does the embedding change between train and eval mode?

        With norm="group" this is ~0 (up to dropout). With norm="batch" it is
        the size of the BatchNorm running-statistics mismatch described in the
        module docstring -- the embedding you decode is not the one the loss
        was computed on. Large values here invalidate the R2 comparison.
        """
        self._check_fitted()
        sequences, _ = _sequences(X, self.window_size, self.n_features_in_)
        sequence = sequences[0]
        count = min(n, len(sequence) - self.window_size + 1)
        locations = np.arange(count)[:, None] + np.arange(self.window_size)
        windows = torch.from_numpy(
            np.ascontiguousarray(sequence[locations].transpose(0, 2, 1))).to(self.device_)
        self.encoder_.eval()
        with torch.no_grad():
            evaluated = self.encoder_(windows)
        self.encoder_.train()
        with torch.no_grad():
            trained = self.encoder_(windows)
        difference = (trained - evaluated).norm(dim=1)
        scale = evaluated.norm(dim=1).clamp_min(1e-8)
        return dict(relative_gap=float((difference / scale).mean()),
                    max_relative_gap=float((difference / scale).max()),
                    norm=self.norm, n=int(count))


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _self_test():
    print("1. MobileNet primitives")
    _check(_make_divisible(32 * 0.35) == 16 and _make_divisible(32 * 1.4) == 48,
           "_make_divisible matches the paper's rounding (alpha=0.35 -> 16, 1.4 -> 48)")
    x = torch.linspace(-8, 8, 33)
    _check(float(_HardSigmoid()(x)[0]) == 0.0 and float(_HardSigmoid()(x)[-1]) == 1.0,
           "hard-sigmoid saturates to exactly 0 and 1")
    _check(abs(float(_HardSwish()(torch.zeros(1)))) < 1e-9,
           "hard-swish(0) == 0")
    _check(float(nn.ReLU6()(torch.tensor([99.0]))) == 6.0, "ReLU6 clamps at 6")

    print("2. every block consumes exactly two bins, so versions stay comparable")
    for block in (_SeparableBlock(12, 16, "group", "relu6", 0.0),
                  _InvertedResidual(12, 16, 3, "group", "relu6", 0.0),
                  _InvertedResidual(12, 12, 3, "group", "hardswish", 0.0, True)):
        out = block(torch.randn(4, 12, 9))
        _check(out.shape[-1] == 7, f"{type(block).__name__} 9 bins -> 7")
    identity = _InvertedResidual(12, 12, 3, "group", "relu6", 0.0)
    _check(identity.residual, "the inverted residual connects when widths match")
    _check(not _InvertedResidual(12, 16, 3, "group", "relu6", 0.0).residual,
           "and skips the connection when they do not")

    print("3. the projection really is linear (V2's whole point)")
    block = _InvertedResidual(8, 8, 3, "none", "relu6", 0.0)
    last = list(block.net)[-2]
    _check(isinstance(last, nn.Conv1d) and last.kernel_size == (1,),
           "the final layer of the block is a 1x1 projection")
    _check(not any(isinstance(m, (nn.ReLU6, _HardSwish, nn.GELU))
                   for m in list(block.net)[list(block.net).index(last):]),
           "NO activation appears after the projection -> linear bottleneck preserved")
    depthwise = [m for m in block.net if isinstance(m, nn.Conv1d) and m.kernel_size == (3,)]
    _check(len(depthwise) == 1 and depthwise[0].groups == depthwise[0].in_channels,
           "the 3-tap convolution is depthwise (groups == channels)")

    print("4. receptive field is exactly window_size, for every version")
    for window in (4, 5, 8, 10, 13, 16, 21):
        for version in MOBILE_VERSIONS:
            for stem in STEMS:
                trunk = _MobileTrunk(9, window, 16, version=version, stem=stem)
                out = trunk(torch.zeros(3, 9, window))
                _check(out.shape == (3, trunk.out_features),
                       f"{version}/{stem} W={window}: {trunk.n_blocks} blocks -> one bin")

    print("5. the per_neuron stem does not mix neurons before the pointwise")
    trunk = _MobileTrunk(6, 8, 16, stem="per_neuron")
    first = list(trunk.stem)[0]
    _check(first.groups == 6, "stem='per_neuron' first conv has groups == n_neurons")
    _check(list(_MobileTrunk(6, 8, 16, stem="mix").stem)[0].groups == 1,
           "stem='mix' first conv mixes all neurons immediately")
    # perturbing one neuron must leave the other neurons' stem channels untouched
    probe = nn.Sequential(*list(trunk.stem)[:1]).eval()
    a = torch.zeros(1, 6, 8)
    b = a.clone(); b[0, 2] = 5.0
    with torch.no_grad():
        delta = (probe(b) - probe(a)).abs().sum(dim=-1)[0]
    touched = (delta > 1e-6).nonzero().flatten().tolist()
    _check(touched == [2], f"perturbing neuron 2 changes only its own filter output {touched}")

    print("6. parameter accounting -- is this actually 'efficient' in 1D?")
    data, latent = _synthetic(4000, 40, seed=5)
    shared = dict(window_size=10, n_tiles=3, tile_gap=(1, 3), output_dimension=32,
                  num_hidden_units=32, head_hidden_units=32, max_epochs=0,
                  device="cpu", verbose=False)
    base = JigsawNet(trunk_block="residual", **shared).fit(data)
    plain = sum(p.numel() for p in base.encoder_.trunk.parameters())
    print(f"      plain residual trunk         {plain:>9,} params")
    for version, t in (("v1", 3), ("v2", 1), ("v2", 3), ("v2", 6), ("v3", 3)):
        model = MobileJigsaw(version=version, expansion=t, **shared).fit(data)
        count = model.n_parameters()["trunk"]
        print(f"      mobilenet {version} t={t}              {count:>9,} params "
              f"({count / plain:.2f}x the plain trunk)")
    _check(MobileJigsaw(version="v1", expansion=3, **shared).fit(data).n_parameters()["trunk"]
           < plain, "v1 IS leaner than the plain trunk")
    _check(MobileJigsaw(version="v2", expansion=6, **shared).fit(data).n_parameters()["trunk"]
           > plain, "v2 at the paper's t=6 is BIGGER than the plain trunk, as predicted")

    print("7. width_multiplier scales the model the way alpha should")
    counts = []
    for alpha in (0.35, 0.5, 1.0, 1.4):
        model = MobileJigsaw(width_multiplier=alpha, **shared).fit(data)
        counts.append(model.n_parameters()["trunk"])
        print(f"      alpha={alpha:<5} {counts[-1]:>9,} params")
    _check(all(b > a for a, b in zip(counts, counts[1:])),
           "parameter count increases monotonically with alpha")

    print("8. GroupNorm has no train/eval gap; BatchNorm does")
    gaps = {}
    for norm in ("group", "batch"):
        model = MobileJigsaw(norm=norm, **dict(shared, max_epochs=3)).fit(data[:3000])
        gaps[norm] = model.measure_train_eval_gap(data[3000:])["relative_gap"]
        print(f"      norm={norm:<6} relative train-vs-eval embedding gap = {gaps[norm]:.4f}")
    _check(gaps["group"] < 1e-5,
           f"norm='group' is exactly consistent between modes ({gaps['group']:.2e})")
    _check(gaps["batch"] > gaps["group"],
           "norm='batch' shows a measurable gap -> why it is not the default here")

    print("9. end to end, and it must beat its own random-encoder control")
    data, latent = _synthetic(7000, 24, seed=1)
    cut, alphas = 5000, (1e-2, 1e-1, 1.0, 10.0, 100.0)
    run = dict(window_size=8, n_tiles=3, tile_gap=(1, 3), output_dimension=32,
               num_hidden_units=32, head_hidden_units=32, batch_size=256,
               learning_rate=3e-3, device="cpu", verbose=False, random_state=0)

    def score(model):
        z_train, i_train = model.transform(data[:cut], pad=False, return_indices=True)
        z_test, i_test = model.transform(data[cut:], pad=False, return_indices=True)
        return _ridge_r2(z_train.astype(np.float64), latent[:cut][i_train].astype(np.float64),
                         z_test.astype(np.float64), latent[cut:][i_test].astype(np.float64),
                         alphas)["r2"]

    trained = MobileJigsaw(max_epochs=12, **run).fit(data[:cut])
    frozen = MobileJigsaw(max_epochs=0, **run).fit(data[:cut])
    r2_trained, r2_frozen = score(trained), score(frozen)
    print(f"      trained R2={r2_trained:.4f}  random-encoder R2={r2_frozen:.4f}")
    _check(r2_trained > 0.4, f"the MobileNet embedding decodes the latent ({r2_trained:.3f})")
    _check(r2_trained > r2_frozen - 0.02, "training does not fall behind its frozen control")
    pretext = trained.evaluate_pretext(data[cut:], max_spans=400, verbose=False)
    print(f"      held-out pair={pretext['pair_accuracy_percent']:.1f}% "
          f"(decode={pretext['decode']}, argmax would give "
          f"{pretext['argmax_pair_accuracy_percent']:.1f}% at "
          f"{pretext['argmax_tie_rate_percent']:.0f}% ties; "
          f"shortcut baseline {pretext['baseline_mean_sort_pair_percent']:.1f}%)")
    _check(pretext["pair_accuracy_percent"] > 55, "the order task is learned out of sample")

    print("10. inherited machinery still works on the subclass")
    _check(trained.summary().count("params") >= 5, "summary() walks every stage")
    a = trained.transform(data[cut:], pad=False)
    _check(np.allclose(a, trained.transform(data[cut:], pad=False)),
           "transform is deterministic")
    _check(len(trained.transform(data[cut:], pad=True)) == len(data[cut:]),
           "pad=True returns one row per bin")
    path = __import__("pathlib").Path("_mobile_jigsaw_selftest.pt")
    try:
        trained.save(path)
        restored = MobileJigsaw.load(path, device="cpu")
        _check(restored.version == trained.version and restored.stem == trained.stem,
               "MobileNet-specific parameters survive the checkpoint")
        _check(np.allclose(restored.transform(data[cut:cut + 300], pad=False),
                           trained.transform(data[cut:cut + 300], pad=False), atol=1e-6),
               "a reloaded model reproduces its embedding exactly")
    finally:
        path.unlink(missing_ok=True)
    try:
        MobileJigsaw(version="v4", **run)
        raise AssertionError("version='v4' should have been rejected")
    except ValueError:
        pass
    for bad in ({"stem": "conv"}, {"norm": "layer"}, {"activation": "swish"},
                {"width_multiplier": 0.0}, {"expansion": 0.5}, {"window_size": 3}):
        try:
            MobileJigsaw(**dict(run, **bad))
            raise AssertionError(f"{bad} should have been rejected")
        except ValueError:
            pass
    _check(True, "invalid MobileNet arguments all raise ValueError")

    print("\nAll self-tests passed. Same cross-entropy losses, MobileNet trunk.")


def _compare():
    """Head to head on synthetic data: same seed, same tiles, same losses."""
    data, latent = _synthetic(9000, 40, seed=7)
    cut = 6500
    shared = dict(window_size=10, n_tiles=4, tile_gap=(1, 6), output_dimension=64,
                  num_hidden_units=48, head_hidden_units=48, batch_size=256,
                  learning_rate=2e-3, device="cuda_if_available", verbose=False,
                  random_state=0)
    arms = [
        ("plain residual", JigsawNet, dict(max_epochs=25)),
        ("plain (frozen)", JigsawNet, dict(max_epochs=0)),
        ("mobilenet v1", MobileJigsaw, dict(max_epochs=25, version="v1")),
        ("mobilenet v2 t=3", MobileJigsaw, dict(max_epochs=25, version="v2", expansion=3)),
        ("mobilenet v2 t=6", MobileJigsaw, dict(max_epochs=25, version="v2", expansion=6)),
        ("mobilenet v3 +SE", MobileJigsaw, dict(max_epochs=25, version="v3")),
        ("v2, stem=mix", MobileJigsaw, dict(max_epochs=25, stem="mix")),
        ("v2, alpha=0.35", MobileJigsaw, dict(max_epochs=25, width_multiplier=0.35)),
        ("v2, batchnorm", MobileJigsaw, dict(max_epochs=25, norm="batch")),
        ("mobilenet (frozen)", MobileJigsaw, dict(max_epochs=0)),
    ]
    print(f"{'arm':<20}{'R2':>9}{'raw R2':>9}{'pair %':>9}{'shortcut':>10}"
          f"{'trunk params':>14}{'tr/ev gap':>11}")
    print("-" * 82)
    for name, factory, override in arms:
        model = factory(**dict(shared, **override)).fit(data[:cut])
        decoding = model.evaluate_decoding(data[:cut], latent[:cut], data[cut:], latent[cut:])
        pretext = model.evaluate_pretext(data[cut:], max_spans=400, verbose=False)
        params = (model.n_parameters()["trunk"] if isinstance(model, MobileJigsaw)
                  else sum(p.numel() for p in model.encoder_.trunk.parameters()))
        gap = (model.measure_train_eval_gap(data[cut:])["relative_gap"]
               if isinstance(model, MobileJigsaw) else float("nan"))
        print(f"{name:<20}{decoding['embedding']['r2']:>9.4f}"
              f"{decoding['raw_window']['r2']:>9.4f}"
              f"{pretext['pair_accuracy_percent']:>9.2f}"
              f"{pretext['baseline_mean_sort_pair_percent']:>10.2f}"
              f"{params:>14,}{gap:>11.4f}")
    print("-" * 82)
    print("raw R2 is the ceiling. Compare each trunk against its OWN frozen control.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--compare", action="store_true")
    options = parser.parse_args()
    if options.self_test:
        _self_test()
    elif options.compare:
        _compare()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

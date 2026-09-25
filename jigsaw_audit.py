"""jigsaw_audit.py -- shortcut controls, a correct rate-removal operator, and new
architectures for JigsawNet. Everything subclasses the existing estimators, so
fit / transform / save / load and run_jigsaw.py keep working unchanged.

    from jigsaw_audit import register
    register(MODELS, ARMS)          # one line in run_jigsaw.py, after ARMS is defined

AuditedJigsaw(JigsawNet)
    order_view      "tile_norm" (legacy) | "count_match" (exact rate removal) | "raw"
    label_control   "true" | "fresh_random" (regularisation control)
                    | "frozen_random" (memorisation control)
    lambda_tcl      > 0 adds time-segment classification (TCL). Cross-entropy only.
    order_head_kind "deepsets" (yours) | "relational" (self-attention across tiles)
    evaluate_pretext uses the configured order view, the fixed score sign,
    NON-overlapping spans and a bootstrap CI. With default arguments training is
    the same computation as JigsawNet with the same seed.

LevelAwareMobileJigsaw(MobileJigsaw)
    The stock trunk is exactly invariant to input scale (GroupNorm after bias-free
    convs). This one re-injects per-window level and spread.

Diagnostics
    count_match                  per-neuron spike-count equalisation across tiles
    shortcut_ranker              linear ranking of rate statistics, fitted on train
    level_leak_probe             does a tile_norm view actually remove level?
    irreversibility_lower_bound  arrow-of-time information in nats (research idea)
    remove_slow_drift            timescale-selective drift removal (preprocessing)
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from jigsaw_net import (JigsawNet, _GradScale, _assign, _augment, _gather_tiles,
                        _order_losses, _order_metrics, _ranks, _shuffle_tiles,
                        _tile_normalize)
from mobile_jigsaw import MobileJigsaw


# --------------------------------------------------------------------------- #
# shortcut removal that is correct for spike counts
# --------------------------------------------------------------------------- #
def count_match(tiles, generator):
    """Exact per-neuron spike-count equalisation across the K tiles of every span.

    tiles: (B, K, N, W) non-negative integer counts (any float dtype).
    For each (span, neuron) the target is c* = min_k sum_t tiles[b, k, n, t]; every
    tile keeps a uniformly random subset of c* of ITS OWN spikes (sampling without
    replacement at the spike level). Afterwards all tiles of a span have identical
    per-neuron counts, so no function of a tile, linear or not, can order tiles by
    per-neuron rate. Only where inside each tile the kept spikes fall survives.
    Mean subtraction cannot do this for counts: the removed mean stays recoverable
    from the lattice / sparsity pattern (probe R^2 = 0.95 on Poisson data).
    Memory is B*K*N*W*max_count; ~110 ms per 256x4x40x10 batch on one CPU core.
    """
    counts = tiles.round().long()
    batch, n_tiles, neurons, width = counts.shape
    cmax = int(counts.max())
    if cmax == 0:
        return tiles
    target = counts.sum(-1).min(dim=1).values                               # (B, N)
    slot = torch.arange(cmax, device=tiles.device)
    valid = slot.view(1, 1, 1, 1, cmax) < counts.unsqueeze(-1)              # (B,K,N,W,C)
    keys = torch.rand(valid.shape, device=tiles.device, generator=generator)
    keys = keys.masked_fill(~valid, 2.0).reshape(batch, n_tiles, neurons, width * cmax)
    # keep the c* smallest keys: one sort gives each row's c*-th smallest key as a
    # threshold (ties have probability 0; invalid slots carry 2.0 and are never kept).
    ordered = keys.sort(-1).values
    need = target[:, None, :, None].expand(-1, n_tiles, -1, 1)
    threshold = ordered.gather(-1, (need - 1).clamp_min(0))
    keep = (keys <= threshold) & (need > 0)
    return keep.reshape(batch, n_tiles, neurons, width, cmax).sum(-1).to(tiles.dtype)


def remove_slow_drift(X, window_bins, *, causal=False):
    """Timescale-selective alternative to tile normalisation.

    Divides each neuron by its moving-average rate over `window_bins` (choose far
    longer than a trial, e.g. 60 s) and rescales to its overall mean: slow drift
    goes, fast task-locked rate modulation stays. Apply to each split separately;
    causal=True uses only past bins. Output is not integer, so it cannot be combined
    with order_view="count_match".
    """
    X = np.asarray(X, dtype=np.float64)
    csum = np.vstack((np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)))
    t = np.arange(len(X))
    lo = np.maximum(0, t - window_bins + 1) if causal else np.maximum(0, t - window_bins // 2)
    hi = t + 1 if causal else np.minimum(len(X), t + window_bins // 2 + 1)
    trend = (csum[hi] - csum[lo]) / (hi - lo)[:, None]
    mean = X.mean(0, keepdims=True)
    return (X * mean / np.maximum(trend, 0.1 * mean + 1e-6)).astype(np.float32)


# --------------------------------------------------------------------------- #
# architectures
# --------------------------------------------------------------------------- #
class RelationalOrderHead(nn.Module):
    """Drop-in replacement for _OrderHead: self-attention across tiles.

    No positional encoding, so it stays permutation-equivariant, but tiles can now
    compare each other pairwise. The paper's fc7 sees all tiles jointly; the DeepSets
    head only sees the set mean and cannot express "B is what A evolves into".
    Pairwise access also re-opens boundary matching across small gaps, so pair it
    with count_match or with gaps longer than the autocorrelation time.
    """

    def __init__(self, dimension, hidden, n_tiles, layers=2, heads=4):
        super().__init__()
        self.inp = nn.Linear(dimension, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=2 * hidden,
                                           dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.position = nn.Linear(hidden, n_tiles)
        self.score = nn.Linear(hidden, 1)

    def forward(self, z):
        tokens = self.body(self.inp(z))
        return self.position(tokens), self.score(tokens).squeeze(-1)


class _StatsPassthroughEncoder(nn.Module):
    """GroupNorm right after a bias-free conv normalises every window by its own
    statistics, so the stock MobileNet trunk satisfies f(c*x) = f(x) to ~1e-5
    (measured): population gain is invisible. This concatenates the per-window,
    per-group level and spread that the first normalisation discards before the
    projection. The trunk itself is untouched."""

    def __init__(self, trunk, channels, output_dimension, normalize, groups=8):
        super().__init__()
        self.trunk, self.normalize = trunk, normalize
        self.groups = math.gcd(channels, groups)
        self.project = nn.Linear(trunk.out_features + 2 * self.groups, output_dimension)

    def forward(self, x):
        grouped = x.reshape(len(x), self.groups, -1)
        stats = torch.cat((torch.log1p(grouped.mean(-1).clamp_min(0)),
                           torch.log1p(grouped.std(-1, correction=0))), dim=-1)
        z = self.project(torch.cat((self.trunk(x), stats), dim=-1))
        return F.normalize(z, dim=-1) if self.normalize else z


class LevelAwareMobileJigsaw(MobileJigsaw):
    """MobileJigsaw whose embedding can see input scale again."""

    _MODEL_TYPE = "level_aware_mobile_jigsaw"

    def _make_encoder(self, channels):
        trunk = super()._make_encoder(channels).trunk
        return _StatsPassthroughEncoder(trunk, channels, self.output_dimension, self.normalize)


# --------------------------------------------------------------------------- #
# estimator with controls
# --------------------------------------------------------------------------- #
class AuditedJigsaw(JigsawNet):
    _MODEL_TYPE = "audited_jigsaw"
    _PARAM_NAMES = JigsawNet._PARAM_NAMES + ("order_view", "label_control", "frozen_block",
                                             "lambda_tcl", "tcl_segments", "order_head_kind")
    ORDER_VIEWS = ("tile_norm", "count_match", "raw")
    LABEL_CONTROLS = ("true", "fresh_random", "frozen_random")
    ORDER_HEADS = ("deepsets", "relational")

    def __init__(self, *, order_view="tile_norm", label_control="true", frozen_block=0,
                 lambda_tcl=0.0, tcl_segments=64, order_head_kind="deepsets", **kwargs):
        super().__init__(**kwargs)
        for name, value, allowed in (("order_view", order_view, self.ORDER_VIEWS),
                                     ("label_control", label_control, self.LABEL_CONTROLS),
                                     ("order_head_kind", order_head_kind, self.ORDER_HEADS)):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}; got {value!r}.")
        self.order_view, self.label_control = order_view, label_control
        self.order_head_kind = order_head_kind
        self.frozen_block = int(frozen_block)
        self.lambda_tcl, self.tcl_segments = float(lambda_tcl), int(tcl_segments)
        self._frozen_table = None

    # -- setup ------------------------------------------------------------- #
    def _build(self, channels):
        super()._build(channels)          # encoder + default heads initialised as in JigsawNet
        if self.order_head_kind == "relational":
            self.order_head_ = RelationalOrderHead(self.output_dimension, self.head_hidden_units,
                                                   self.n_tiles).to(self.device_)
        # Hung on a module that JigsawNet.fit already optimises and saves, so fit()
        # and save()/load() need no change.
        self.reconstruct_head_.tcl = nn.Sequential(
            nn.Linear(self.output_dimension, self.head_hidden_units), nn.GELU(),
            nn.Linear(self.head_hidden_units, self.tcl_segments)).to(self.device_)

    def fit(self, X, X_valid=None, **kwargs):
        if self.order_view == "count_match":
            for array in (X if isinstance(X, (list, tuple)) else [X]):
                array = np.asarray(array)
                if (array < 0).any() or not np.allclose(array, np.round(array)):
                    raise ValueError("order_view='count_match' needs raw non-negative "
                                     "integer spike counts.")
        self._frozen_table = None
        return super().fit(X, X_valid, **kwargs)

    # -- order branch pieces ------------------------------------------------ #
    def _order_input(self, tiles, generator, augment):
        def aug(t):
            return _augment(t, self.neuron_dropout, self.gain_jitter, generator) if augment else t
        if self.order_view == "count_match":
            return aug(count_match(tiles, generator))   # match first, then augment
        if self.order_view == "raw":
            return aug(tiles)
        return _tile_normalize(aug(tiles), self.tile_norm)

    def _control_labels(self, truth, starts):
        """'fresh_random' redraws the target every step: the order branch becomes pure
        gradient noise (tests regularisation). 'frozen_random' fixes one arbitrary
        chronology per block of span starts: only memorising WHERE a span came from
        can fit it (tests memorisation pressure)."""
        if self.label_control == "true":
            return truth
        if self.label_control == "fresh_random":
            return torch.rand(truth.shape, device=truth.device,
                              generator=self._generator).argsort(1)
        if self._frozen_table is None:
            g = torch.Generator(device=self.device_).manual_seed(self.random_state + 7)
            self._frozen_table = torch.rand(65536, self.n_tiles, device=self.device_,
                                            generator=g).argsort(1)
        block = self.frozen_block or self.training_span
        pseudo = self._frozen_table[torch.div(starts, block, rounding_mode="floor") % 65536]
        return pseudo.gather(1, truth)

    # -- one step (same RNG order as JigsawNet._step for default arguments) -- #
    def _step(self, data, starts):
        batch = len(starts)
        tiles, next_index = _gather_tiles(data, starts, self._draw(batch), self.window_size)
        zero = torch.zeros((), device=self.device_)
        parts = {k: zero for k in ("forecast", "forecast_accuracy", "reconstruct",
                                   "reconstruct_accuracy", "tcl", "position", "pair", "exact",
                                   "pair_accuracy", "tie_rate", "pair_accuracy_vs_truth")}
        total = zero
        if self.lambda_forecast > 0 or self.lambda_reconstruct > 0 or self.lambda_tcl > 0:
            raw = _augment(tiles, self.neuron_dropout, self.gain_jitter, self._generator)
            z_raw = self.encoder_(raw.reshape(-1, self.n_features_in_, self.window_size))
            if self.lambda_forecast > 0:
                target = self._bucketize(data[next_index.reshape(-1)])
                logits = self.forecast_head_(z_raw)
                parts["forecast"] = F.cross_entropy(logits.reshape(-1, self.forecast_levels),
                                                    target.reshape(-1))
                with torch.no_grad():
                    parts["forecast_accuracy"] = (logits.argmax(-1) == target).float().mean()
                total = total + self.lambda_forecast * parts["forecast"]
            if self.lambda_reconstruct > 0:
                flat = tiles.permute(0, 1, 3, 2).reshape(-1, self.n_features_in_)
                target = self._bucketize(flat).reshape(
                    -1, self.window_size, self.n_features_in_).permute(0, 2, 1)
                logits = self.reconstruct_head_(z_raw)
                parts["reconstruct"] = F.cross_entropy(
                    logits.reshape(-1, self.forecast_levels), target.reshape(-1))
                with torch.no_grad():
                    parts["reconstruct_accuracy"] = (logits.argmax(-1) == target).float().mean()
                total = total + self.lambda_reconstruct * parts["reconstruct"]
            if self.lambda_tcl > 0:
                tile_starts = (next_index - self.window_size).reshape(-1)
                segment = (tile_starts * self.tcl_segments // data.shape[0]).clamp_max(
                    self.tcl_segments - 1)
                parts["tcl"] = F.cross_entropy(self.reconstruct_head_.tcl(z_raw), segment)
                total = total + self.lambda_tcl * parts["tcl"]

        if self.lambda_order > 0 or self.lambda_pair > 0:
            view, truth = _shuffle_tiles(self._order_input(tiles, self._generator, True),
                                         self.shuffle_tiles, self._generator)
            targets = self._control_labels(truth, starts)
            z = self.encoder_(view.reshape(-1, self.n_features_in_, self.window_size))
            z = _GradScale.apply(z, self.order_grad_scale)
            logits, scores = self.order_head_(z.reshape(batch, self.n_tiles, self.output_dimension))
            parts["position"], parts["pair"] = _order_losses(logits, scores, targets)
            total = total + self.lambda_order * parts["position"] + self.lambda_pair * parts["pair"]
            with torch.no_grad():
                # larger score = EARLIER (target 1 iff i precedes j, logit s_i - s_j)
                predicted = _assign(logits) if self.lambda_order > 0 else _ranks(-scores)
                exact, pair, ties = _order_metrics(predicted, targets)
                parts["exact"], parts["pair_accuracy"], parts["tie_rate"] = \
                    exact.mean(), pair.mean(), ties.mean()
                parts["pair_accuracy_vs_truth"] = _order_metrics(predicted, truth)[1].mean()
        parts["total"] = total
        return parts

    # -- honest pretext evaluation ----------------------------------------- #
    def evaluate_pretext(self, X, *, max_spans=1024, batch_size=256, repeats=8,
                         random_state=200042, verbose=None, n_boot=2000):
        """Same keys as JigsawNet.evaluate_pretext, plus pair_ci95_percent.

        Spans are NON-overlapping (stride = training_span), so n_spans counts
        roughly independent samples; the parent's stride-1 spans share most bins.
        """
        self._check_fitted()
        data, starts, _ = self._spans(X, self.n_features_in_)
        chosen = starts[::self.training_span][:max_spans]
        if len(chosen) < 2:
            raise ValueError("Fewer than two non-overlapping spans in X.")
        chosen = torch.from_numpy(chosen).to(self.device_)
        g = torch.Generator(device=self.device_).manual_seed(random_state)
        low, high = self.tile_gap
        self.encoder_.eval(); self.order_head_.eval()
        with torch.no_grad():
            acc = {k: torch.zeros(len(chosen), device=self.device_)
                   for k in ("exact", "pair", "ce", "mean", "norm")}
            for _ in range(repeats):
                for begin in range(0, len(chosen), batch_size):
                    block = chosen[begin:begin + batch_size]
                    gaps = torch.randint(low, high + 1, (len(block), self.n_tiles - 1),
                                         device=self.device_, generator=g)
                    tiles, _ = _gather_tiles(data, block, gaps, self.window_size)
                    view, truth = _shuffle_tiles(self._order_input(tiles, g, False), True, g)
                    raw = torch.gather(tiles, 1, truth[:, :, None, None].expand(
                        -1, -1, *tiles.shape[2:]))
                    z = self.encoder_(view.reshape(-1, self.n_features_in_, self.window_size))
                    logits, scores = self.order_head_(z.reshape(len(block), self.n_tiles, -1))
                    predicted = _assign(logits) if self.lambda_order > 0 else _ranks(-scores)
                    exact, pair, _ = _order_metrics(predicted, truth)
                    ce = F.cross_entropy(logits.reshape(-1, self.n_tiles), truth.reshape(-1),
                                         reduction="none").reshape(len(block), -1).mean(1)
                    mean = _order_metrics(_ranks(raw.mean(dim=(2, 3))), truth)[1]
                    norm = _order_metrics(_ranks(raw.flatten(2).norm(dim=2)), truth)[1]
                    for key, value in (("exact", exact), ("pair", pair), ("ce", ce),
                                       ("mean", mean), ("norm", norm)):
                        acc[key][begin:begin + len(block)] += value / repeats
        self.encoder_.train(); self.order_head_.train()
        a = {k: v.cpu().numpy() for k, v in acc.items()}
        rng = np.random.default_rng(random_state)
        boot = a["pair"][rng.integers(0, len(a["pair"]), (n_boot, len(a["pair"])))].mean(1)
        mean_sort = 100.0 * float(a["mean"].mean())
        result = dict(
            exact_accuracy_percent=100.0 * float(a["exact"].mean()),
            pair_accuracy_percent=100.0 * float(a["pair"].mean()),
            pair_ci95_percent=[100.0 * float(np.quantile(boot, 0.025)),
                               100.0 * float(np.quantile(boot, 0.975))],
            tie_rate_percent=0.0, argmax_tie_rate_percent=float("nan"),
            baseline_mean_sort_pair_percent=mean_sort,
            baseline_mean_sort_best_direction_percent=max(mean_sort, 100.0 - mean_sort),
            baseline_norm_sort_pair_percent=100.0 * float(a["norm"].mean()),
            position_cross_entropy=float(a["ce"].mean()),
            uniform_cross_entropy=math.log(self.n_tiles),
            decode="assignment" if self.lambda_order > 0 else "score_rank(-scores)",
            order_view=self.order_view,
            chance_exact_percent=100.0 / math.factorial(self.n_tiles), chance_pair_percent=50.0,
            n_spans=int(len(chosen)), n_draws=int(len(chosen) * repeats), repeats=int(repeats))
        if self.verbose if verbose is None else verbose:
            lo, hi = result["pair_ci95_percent"]
            print(f"  pretext[{self.order_view}]: pair={result['pair_accuracy_percent']:.2f}% "
                  f"(95% CI {lo:.1f}-{hi:.1f}, n={len(chosen)} non-overlapping spans) "
                  f"exact={result['exact_accuracy_percent']:.2f}% "
                  f"(chance {result['chance_exact_percent']:.2f}%)", flush=True)
        return result


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def _sample_tiles(model, X, count, seed):
    """Chronological (B, K, N, W) tiles from random span starts of X, and a generator."""
    data, starts, _ = model._spans(X, model.n_features_in_)
    rng = np.random.default_rng(seed)
    index = torch.from_numpy(rng.choice(starts, min(count, len(starts)), replace=False))
    g = torch.Generator(device=model.device_).manual_seed(seed)
    gaps = torch.randint(model.tile_gap[0], model.tile_gap[1] + 1,
                         (len(index), model.n_tiles - 1), device=model.device_, generator=g)
    return _gather_tiles(data, index.to(model.device_), gaps, model.window_size)[0], g


def tile_statistics(tiles):
    """Per-tile rate statistics, (B, K, N, W) -> (B, K, 4N + 1): per-neuron mean,
    variance, fraction of empty bins, within-tile slope, plus population mean."""
    t = torch.linspace(-1.0, 1.0, tiles.shape[-1], device=tiles.device)
    mean = tiles.mean(-1)
    slope = ((tiles - mean[..., None]) * t).mean(-1) / t.pow(2).mean()
    return torch.cat((mean, tiles.var(-1, correction=0), (tiles == 0).float().mean(-1),
                      slope, tiles.mean(dim=(2, 3))[..., None]), dim=-1)


def shortcut_ranker(model, X_train, X_test, *, features=None, n_train=20000, n_test=4000,
                    steps=500, l2=1e-3, seed=0):
    """Held-out pair accuracy of a LINEAR Bradley-Terry ranking of rate statistics,
    fitted on TRAIN spans. The bar the network must clear instead of mean-sort, which
    is 1-D with a fixed sign (it scored 48% on data where rate pattern gives 91%).
    features=lambda t: tile_statistics(_tile_normalize(t, "mean")) measures what
    survives your order view."""
    features = tile_statistics if features is None else features
    f_train = features(_sample_tiles(model, X_train, n_train, seed)[0])
    f_test = features(_sample_tiles(model, X_test, n_test, seed + 1)[0])
    mu, sd = f_train.mean((0, 1)), f_train.std((0, 1)) + 1e-6
    f_train, f_test = (f_train - mu) / sd, (f_test - mu) / sd
    i, j = (v.to(f_train.device) for v in torch.triu_indices(model.n_tiles, model.n_tiles, 1))
    w = torch.zeros(f_train.shape[-1], device=f_train.device, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=1e-2)
    for _ in range(steps):                  # tiles are chronological: i precedes j
        s = f_train @ w                     # larger = earlier, like the model's scores
        loss = F.softplus(s[:, j] - s[:, i]).mean() + l2 * w.pow(2).sum()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        s = f_test @ w
        return 100.0 * float((s[:, i] > s[:, j]).float().mean())


def level_leak_probe(model, X, *, mode=None, n_spans=2000, steps=1500, seed=0):
    """Held-out R^2 for recovering each (tile, neuron) mean count from its
    tile-normalised trace. ~0: the view removes level. ~1: it only hides level from
    LINEAR readouts (synthetic Poisson: 0.95 "mean", 0.83 "zscore"; Gaussian 0.00)."""
    tiles, _ = _sample_tiles(model, X, n_spans, seed)
    view = _tile_normalize(tiles, mode or model.tile_norm)
    x, y = view.reshape(-1, model.window_size), tiles.mean(-1).reshape(-1)
    cut = int(0.8 * len(x))
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(model.window_size, 64), nn.GELU(), nn.Linear(64, 64),
                        nn.GELU(), nn.Linear(64, 1)).to(x.device)
    optimizer = torch.optim.Adam(net.parameters(), 3e-3)
    for _ in range(steps):
        k = torch.randint(0, cut, (1024,), device=x.device)
        loss = (net(x[k]).squeeze(-1) - y[k]).pow(2).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        return 1.0 - float((net(x[cut:]).squeeze(-1) - y[cut:]).pow(2).mean() / y[cut:].var())


def irreversibility_lower_bound(model, X, *, n_spans=4000, seed=0):
    """Donsker-Varadhan lower bound, in nats per tile pair, on KL(forward || reversed)
    at the model's tile/gap scale. The pair logit l = s_i - s_j is antisymmetric and,
    if well trained, estimates log p(A then B) / p(B then A), so
        KL >= E_fwd[l] - log E_fwd[exp(-l)].
    Use held-out data, ideally a K=2 model trained with order_view="count_match" so
    the irreversibility cannot come from rate changes."""
    tiles, g = _sample_tiles(model, X, n_spans, seed)
    view = model._order_input(tiles, g, False) if hasattr(model, "_order_input") \
        else _tile_normalize(tiles, model.tile_norm)
    model.encoder_.eval(); model.order_head_.eval()
    with torch.no_grad():
        z = model.encoder_(view.reshape(-1, model.n_features_in_, model.window_size))
        _, s = model.order_head_(z.reshape(len(view), model.n_tiles, -1))
    model.encoder_.train(); model.order_head_.train()
    i, j = (v.to(s.device) for v in torch.triu_indices(model.n_tiles, model.n_tiles, 1))
    ell = (s[:, i] - s[:, j]).reshape(-1).double()      # chronological: i precedes j
    return float(ell.mean() - (torch.logsumexp(-ell, 0) - math.log(len(ell))))


# --------------------------------------------------------------------------- #
# run_jigsaw.py integration
# --------------------------------------------------------------------------- #
def register(models, arms):
    """Add the models and arms to run_jigsaw's MODELS / ARMS dictionaries."""
    models.update(audited=AuditedJigsaw, mobile_level=LevelAwareMobileJigsaw)
    arms.update({
        # paired reference: same training as "proposed", honest pretext evaluation
        "audited_proposed": dict(_model="audited"),
        # is the ORDER EFFECT about temporal order at all?
        "ctrl_fresh_labels": dict(_model="audited", label_control="fresh_random"),
        "ctrl_frozen_labels": dict(_model="audited", label_control="frozen_random"),
        "ctrl_tcl": dict(_model="audited", lambda_order=0.0, lambda_pair=0.0, lambda_tcl=1.0),
        # rate removal that actually removes rate (spike counts only)
        "order_count_matched": dict(_model="audited", order_view="count_match"),
        "order_raw_view": dict(_model="audited", order_view="raw"),
        "tiles_2_counts": dict(_model="audited", n_tiles=2, order_view="count_match"),
        # relational reasoning between tiles
        "relational_head": dict(_model="audited", order_head_kind="relational"),
        "relational_counts": dict(_model="audited", order_head_kind="relational",
                                  order_view="count_match"),
        # MobileNet that can see population gain. print_table groups by
        # _model == "mobile", so compare these two head to head.
        "mobile_level_v2": dict(_model="mobile_level", version="v2", expansion=3),
        "mobile_level_random": dict(_model="mobile_level", max_epochs=0),
    })


# --------------------------------------------------------------------------- #
def _smoke_test():
    from jigsaw_net import _synthetic
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    tiles = torch.poisson(torch.rand(8, 4, 5, 10) * 2)
    matched = count_match(tiles, g)
    sums = matched.sum(-1)
    assert torch.equal(sums, sums[:, :1].expand_as(sums)), "counts must match across tiles"
    assert (matched <= tiles).all(), "count_match may only delete spikes"
    assert torch.equal(sums[:, 0], tiles.sum(-1).min(1).values), "target is the minimum"
    print("ok  count_match equalises per-neuron counts exactly and only deletes spikes")

    head = RelationalOrderHead(16, 32, 4).eval()
    z = torch.randn(3, 4, 16)
    perm = torch.stack([torch.randperm(4) for _ in range(3)])
    la, sa = head(z)
    lb, sb = head(z.gather(1, perm[:, :, None].expand(-1, -1, 16)))
    assert torch.allclose(sb, sa.gather(1, perm), atol=1e-5)
    assert torch.allclose(lb, la.gather(1, perm[:, :, None].expand(-1, -1, 4)), atol=1e-5)
    print("ok  relational head is permutation-equivariant")

    data, _ = _synthetic(3000, 16, seed=0)
    common = dict(window_size=8, n_tiles=3, tile_gap=(1, 3), output_dimension=16,
                  num_hidden_units=16, head_hidden_units=16, batch_size=128,
                  device="cpu", verbose=False, random_state=0)
    base = JigsawNet(max_epochs=2, **common).fit(data[:2400])
    same = AuditedJigsaw(max_epochs=2, **common).fit(data[:2400])
    gap = np.abs(base.transform(data[2400:], pad=False) - same.transform(data[2400:], pad=False))
    assert gap.max() < 1e-4, gap.max()
    print(f"ok  AuditedJigsaw with defaults reproduces JigsawNet (max |dz| = {gap.max():.1e})")
    for kw in (dict(order_view="count_match"), dict(order_view="raw"),
               dict(label_control="fresh_random"), dict(label_control="frozen_random"),
               dict(lambda_order=0.0, lambda_pair=0.0, lambda_tcl=1.0),
               dict(order_head_kind="relational"), dict(lambda_order=0.0, lambda_pair=1.0)):
        model = AuditedJigsaw(max_epochs=1, **kw, **common).fit(data[:2400])
        pretext = model.evaluate_pretext(data[2400:], repeats=2, verbose=False)
        assert np.isfinite(pretext["pair_accuracy_percent"])
        print(f"ok  {kw} trains and evaluates (pair {pretext['pair_accuracy_percent']:.1f}%)")

    mobile = LevelAwareMobileJigsaw(max_epochs=0, **common).fit(data)
    stock = MobileJigsaw(max_epochs=0, **common).fit(data)
    x = torch.from_numpy(np.stack([data[i:i + 8].T for i in range(0, 800, 8)]))
    for name, m in (("stock", stock), ("level-aware", mobile)):
        m.encoder_.eval()
        with torch.no_grad():
            a, b = m.encoder_(x), m.encoder_(2 * x)
        print(f"    {name:<12} mobile encoder |f(2x)-f(x)|/|f(x)| = "
              f"{float(((b - a).norm(dim=1) / a.norm(dim=1)).mean()):.2e}")
    print("smoke test passed")


if __name__ == "__main__":
    _smoke_test()

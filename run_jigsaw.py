"""Perich runner for JigsawNet. Architecture lives in jigsaw_net.py; this file
only loads data, runs arms, and reports.

    python run_jigsaw.py                          # default arms, one seed
    python run_jigsaw.py --seeds 8 9 10           # pooled, with the spread shown
    python run_jigsaw.py --arms proposed random_encoder dim_128
    python run_jigsaw.py --list
    python run_jigsaw.py --epoch-sweep

THE FOUR NUMBERS THAT DECIDE EVERYTHING
  R2 raw     ridge AND MLP on the raw window. The CEILING. Compare like with
             like: embedding-ridge against raw-ridge, embedding-MLP against
             raw-MLP. The MLP-on-embedding path contains a trained encoder plus
             a trained MLP, so holding it up against raw-ridge flatters it.
  R2 random  the SAME architecture with max_epochs=0, frozen at init. This is
             what your weights are worth before any learning. If a trained arm
             is below it, training made things worse, and pretext accuracy is
             irrelevant at that point.
  pair %     held-out order accuracy. The decode is an ASSIGNMENT (a real
             permutation), so chance is exactly 50% and a degenerate head scores
             50%, not 0%. Reported next to the sort-by-level shortcut baseline.
             Look at n_spans before believing any gap: at n=551 the standard
             error is ~2.1%, so 49% and 53% are the same number.
  p.ratio    participation ratio of the embedding. It tracked ridge R2 almost
             monotonically across every run so far (order_only: 3.4/64 -> 0.005).
             A collapsing p.ratio is the failure, and no pretext number fixes it.

Every arm also prints d.random = R2 - R2(frozen control of the same family),
which is the only honest measure of what training contributed -- but read the
control-stability warning first: a noisy control makes d.random meaningless.

EPOCHS ARE A HYPERPARAMETER, NOT A BUDGET, AND MORE IS NOT OBVIOUSLY WORSE.
Spans overlap at stride 1, so the effective sample size is far below the span
count and the order head WILL memorize: held-out position CE ends up many times
log(K). That is a fact about the head, not about the embedding. The best
decoding R2 observed so far came from the largest budget tried, with the pretext
task thoroughly overfit. Sweep with `--epoch-sweep` and pick the peak by R2.
"""
import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

from jigsaw_net import JigsawNet, _ridge_r2
from mobile_jigsaw import MobileJigsaw

MODELS = {"jigsaw": JigsawNet, "mobile": MobileJigsaw}

PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
OUTPUT_ROOT = Path("/home/mirzaei/sam/result/Aggregate")
SEED = 8

WINDOW_SIZE = 10
N_TILES = 4
TILE_GAP = (1, 8)
OUTPUT_DIMENSION = 64
NUM_HIDDEN_UNITS = 64
HEAD_HIDDEN_UNITS = 64
BATCH_SIZE = 512
EPOCHS = 10000
LEARNING_RATE = 1e-3
TILE_NORM = "mean"
NEURON_DROPOUT = 0.1
GAIN_JITTER = 0.1

DECODER_EPOCHS = 2500
DECODER_HIDDEN = 64
DECODER_DROPOUT = 0.4
DECODER_LR = 1e-3
PRETEXT_SPANS = 2048
PRETEXT_REPEATS = 8
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)

# --------------------------------------------------------------------------- #
# arms. No contrastive arm exists -- the model has no contrastive loss.
#
# The default objective is now ORDER + RECONSTRUCT. The forecast CE is off by
# default because it measured as harmful (-0.13 R2 on both seeds), so it lives
# here as an ablation instead of in the proposal.
# --------------------------------------------------------------------------- #
ARMS = {
    # ---- the proposal: order CE + pair BCE + reconstruction CE ----------- #
    "proposed": {},

    # ---- the three arms the whole claim rests on ------------------------- #
    # Frozen at init. THE reference point. Beat this or nothing else matters.
    "random_encoder": dict(max_epochs=0),
    # proposed MINUS the anchor == the classic jigsaw, nothing keeping firing
    # rate. This is the arm that collapsed to p.ratio 3.4/64 and R2 0.005.
    "order_only": dict(lambda_reconstruct=0.0),
    # proposed MINUS the puzzle == a pure denoising autoencoder in CE form.
    # If THIS is the best arm, the jigsaw is dead weight and the paper should
    # say so rather than hide it. proposed - reconstruct_only is the ORDER
    # EFFECT, and it is the single number the thesis lives or dies on.
    "reconstruct_only": dict(lambda_order=0.0, lambda_pair=0.0),

    # ---- the forecast term, kept only as an ablation --------------------- #
    # Adding it back to the winner cost -0.132 / -0.139 R2 across two seeds.
    # It stays runnable because "why not just predict the next bin?" is the
    # first question a reader asks, and the answer should be measured.
    "with_forecast": dict(lambda_forecast=1.0),
    "forecast_only": dict(lambda_order=0.0, lambda_pair=0.0,
                          lambda_forecast=1.0, lambda_reconstruct=0.0),

    # ---- other controls -------------------------------------------------- #
    # Anchor turned up. If R2 keeps climbing with this, the bottleneck -- not
    # the pretext task -- is what the whole experiment is measuring.
    "reconstruct_heavy": dict(lambda_reconstruct=4.0),
    # Order head trains on a trunk it cannot influence. Asks "is order
    # decodable from a reconstruction-trained trunk?" with zero risk to R2.
    "order_probe_only": dict(order_grad_scale=0.0),
    # Every anti-shortcut guard off. If pretext accuracy JUMPS here, the task
    # was being solved by level drift, not by temporal structure.
    "shortcut_open": dict(tile_norm="none", neuron_dropout=0.0, gain_jitter=0.0),
    # Identity labels, i.e. the pre-2026-09-23 behaviour. Only for reproducing
    # old runs: with a constant label an order-dependent head gets free
    # accuracy, so a JUMP here is a bug signature, not an improvement.
    "identity_labels": dict(shuffle_tiles=False),

    # ---- epochs ---------------------------------------------------------- #
    "epochs_10": dict(max_epochs=10),
    "epochs_30": dict(max_epochs=30),
    "epochs_100": dict(max_epochs=100),
    "epochs_300": dict(max_epochs=300),
    "epochs_1000": dict(max_epochs=1000),
    "epochs_3000": dict(max_epochs=3000),
    "epochs_10000": dict(max_epochs=10000),

    # ---- the bottleneck, which the numpy analysis says costs the most ---- #
    "dim_16": dict(output_dimension=16),
    "dim_32": dict(output_dimension=32),
    "dim_128": dict(output_dimension=128),
    "dim_256": dict(output_dimension=256),

    # ---- ablations ------------------------------------------------------- #
    # The sphere. Expected to LOSE: it deletes magnitude.
    "normalized": dict(normalize=True),
    # A single MobileNetV2-style block dropped into the PLAIN trunk. Cheap
    # sanity check; for the real architecture use the mobile_* arms below.
    "separable_block": dict(trunk_block="separable"),
    # Difficulty ladder. n_tiles=2 is the most sensitive learnability test
    # there is: chance is exactly 50%, so tiny effects are detectable, and it
    # is the ONLY setting where the puzzle has been learned out of sample.
    "tiles_2": dict(n_tiles=2),
    "tiles_3": dict(n_tiles=3),
    "tiles_6": dict(n_tiles=6),
    # Long-range order: are far-apart tiles easier (more drift) or harder?
    "wide_gaps": dict(tile_gap=(8, 40)),
    "tight_gaps": dict(tile_gap=(1, 2)),
    # Coarser / finer reconstruction targets (forecast_levels is shared).
    "levels_4": dict(forecast_levels=4),
    "levels_16": dict(forecast_levels=16),
    "zscore_tiles": dict(tile_norm="zscore"),

    # ---- MOBILENET TRUNKS ------------------------------------------------ #
    # Same losses, same tiles, same seed -- only the trunk changes, so any
    # difference here is attributable to the architecture. "_model" is consumed
    # by build_model and is not a constructor argument.
    "mobile_v1": dict(_model="mobile", version="v1"),
    "mobile_v2": dict(_model="mobile", version="v2", expansion=3),
    # The paper's t=6. In 1D this is ~4x MORE parameters than a plain conv, so
    # it is a capacity arm, not an efficiency arm. Label it honestly.
    "mobile_v2_t6": dict(_model="mobile", version="v2", expansion=6),
    "mobile_v3": dict(_model="mobile", version="v3"),
    # Image-style stem: mixes neurons immediately instead of giving each neuron
    # its own temporal filter. Isolates the EEGNet-style inductive bias.
    "mobile_stem_mix": dict(_model="mobile", stem="mix"),
    # alpha, MobileNet's real efficiency knob. Small alpha = regularization,
    # which is the plausible win on short sessions.
    "mobile_alpha_035": dict(_model="mobile", width_multiplier=0.35),
    "mobile_alpha_050": dict(_model="mobile", width_multiplier=0.5),
    "mobile_alpha_140": dict(_model="mobile", width_multiplier=1.4),
    # Canonical MobileNet normalization. Expected to LOSE: running statistics
    # are collected on the augmented/normalized training views and then applied
    # to the raw windows transform sees. The table prints the measured gap.
    "mobile_batchnorm": dict(_model="mobile", norm="batch"),
    "mobile_hardswish": dict(_model="mobile", activation="hardswish"),
    "mobile_head_256": dict(_model="mobile", head_channels=256),
    # The control that makes the whole comparison readable -- and the one that
    # swung 0.2142 -> 0.0946 across two seeds, so run several seeds before
    # reading d.random for the MobileNet family at all.
    "mobile_random": dict(_model="mobile", max_epochs=0),
}

# Old names kept working, because they appear in previous logs and commands.
ALIASES = {
    # forecast is off by default now, so these two collapsed onto existing arms
    "order_reconstruct": "proposed",
    "no_reconstruct": "order_only",
    "forecast_4": "levels_4",
    "forecast_16": "levels_16",
}

DEFAULT_ARMS = ("proposed", "reconstruct_only", "order_only", "random_encoder",
                "with_forecast", "forecast_only", "shortcut_open",
                "dim_256", "tiles_2",
                "mobile_v1", "mobile_v2", "mobile_v3", "mobile_alpha_035",
                "mobile_stem_mix", "mobile_random")

# The epoch sweep. Run this FIRST on a new session: every other comparison is
# conditional on being in a sane epoch regime.
EPOCH_ARMS = ("epochs_10", "epochs_100", "epochs_1000", "epochs_3000",
              "epochs_10000", "random_encoder")

MOBILE_ARMS = tuple(k for k, v in ARMS.items() if v.get("_model") == "mobile")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(obj):
    """NaN/Inf -> None so one broken arm cannot poison the whole dump."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path, payload):
    path.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False))


def cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def normal_tail(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def versus_chance(successes, trials, chance):
    """Wilson 95% CI plus a one-sided z-test against a KNOWN chance rate.

    The chance rate here is known exactly (1/K! or 1/2), not estimated, so a
    one-sample z-test is the right test and no correction is needed.
    """
    if trials <= 0:
        return dict(rate=float("nan"), low=float("nan"), high=float("nan"),
                    z=float("nan"), p=float("nan"), significant=False, n=0)
    rate = successes / trials
    z_critical = 1.959963985
    denominator = 1 + z_critical ** 2 / trials
    centre = (rate + z_critical ** 2 / (2 * trials)) / denominator
    spread = z_critical * math.sqrt(rate * (1 - rate) / trials
                                    + z_critical ** 2 / (4 * trials ** 2)) / denominator
    standard_error = math.sqrt(max(chance * (1 - chance) / trials, 1e-300))
    z = (rate - chance) / standard_error
    p = normal_tail(z)
    return dict(rate=rate, low=max(0.0, centre - spread), high=min(1.0, centre + spread),
                z=z, p=p, significant=bool(p < 0.05 and rate > chance), n=int(trials))


def detectable_difference(n, chance=0.5):
    """Smallest pair-accuracy gap worth discussing at this sample size.

    1.96 standard errors, in percent. Printed next to the table so a 2-point
    difference between two arms is not read as a finding when the resolution
    of the measurement is 4 points.
    """
    if n <= 0:
        return float("nan")
    return 100.0 * 1.959963985 * math.sqrt(chance * (1 - chance) / n)


def load_session(path):
    """Load the existing NPZ train/validation split without splitting again.

    Keep the original runner's first-two-label selection. The neuron mask is
    fitted on training data only and applied identically to both splits.
    """
    path = Path(path)
    required = ("train_data", "train_label", "valid_data", "valid_label")
    with np.load(path, allow_pickle=False) as handle:
        missing = [key for key in required if key not in handle.files]
        if missing:
            raise KeyError(
                f"{path.name}: missing NPZ keys {missing}; has {sorted(handle.files)}")
        arrays = [np.asarray(handle[key], dtype=np.float32) for key in required]
    splits = []
    for name, spikes, behavior in (("train", arrays[0], arrays[1]),
                                   ("valid", arrays[2], arrays[3])):
        if behavior.ndim == 1:
            behavior = behavior[:, None]
        if spikes.ndim != 2 or behavior.ndim != 2:
            raise ValueError(
                f"{path.name}: {name} must be 2D, got spikes {spikes.shape} "
                f"and behavior {behavior.shape}")
        if len(spikes) != len(behavior) and spikes.shape[1] == len(behavior):
            spikes = spikes.T
        if len(spikes) != len(behavior):
            raise ValueError(
                f"{path.name}: {name} length mismatch, spikes {spikes.shape} "
                f"vs behavior {behavior.shape}")
        if not (np.isfinite(spikes).all() and np.isfinite(behavior).all()):
            raise ValueError(f"{path.name}: nonfinite values in {name}")
        splits.append((spikes, behavior))
    spikes_train, behavior_train = splits[0]
    spikes_valid, behavior_valid = splits[1]
    if spikes_train.shape[1] != spikes_valid.shape[1]:
        raise ValueError(
            f"{path.name}: train has {spikes_train.shape[1]} neurons but valid has "
            f"{spikes_valid.shape[1]}")
        
    behavior_train = behavior_train[:, :2]   ## Only train on 2 firsts label 
    behavior_valid = behavior_valid[:, :2]   ## Only train on 2 firsts label 
    
    # Fitted on TRAIN only: deciding which neurons exist using the validation
    # split would be a (small) leak, and it costs nothing to avoid.
    alive = spikes_train.std(0) > 0
    if not alive.any():
        raise ValueError(f"{path.name}: no varying neurons in train_data")
    return (np.ascontiguousarray(spikes_train[:, alive]),
            np.ascontiguousarray(behavior_train),
            np.ascontiguousarray(spikes_valid[:, alive]),
            np.ascontiguousarray(behavior_valid))


class Decoder(nn.Module):
    def __init__(self, dimension, targets, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, hidden), nn.LayerNorm(hidden),
                                 nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, targets))

    def forward(self, x):
        return self.net(x)


def r2_raw(truth, prediction):
    residual = np.square(truth - prediction).sum(0)
    total = np.square(truth - truth.mean(0, keepdims=True)).sum(0)
    return float(np.mean(1.0 - residual / np.maximum(total, 1e-12)))


def mlp_r2(x_train, y_train, x_test, y_test, *, epochs, hidden, dropout,
           learning_rate, seed, device):
    """Chronological internal split for model selection -- never random.

    A random split lets adjacent, near-identical bins land on both sides, and
    the score is then optimistic by a wide margin on autocorrelated data.
    """
    torch.manual_seed(seed)
    mean, scale = x_train.mean(0, keepdims=True), x_train.std(0, keepdims=True) + 1e-8
    cut = max(1, int(0.9 * len(x_train)))
    tensors = [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
               for a in ((x_train - mean) / scale, y_train, (x_test - mean) / scale, y_test)]
    xt, yt, xv, yv = tensors
    inner_x, inner_y, hold_x, hold_y = xt[:cut], yt[:cut], xt[cut:], yt[cut:]
    if len(hold_x) < 2:
        inner_x, inner_y, hold_x, hold_y = xt, yt, xt, yt
    model = Decoder(xt.shape[1], yt.shape[1], hidden, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state, best_score, best_epoch = None, -float("inf"), 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        nn.functional.mse_loss(model(inner_x), inner_y).backward()
        optimizer.step()
        if (epoch + 1) % 25 == 0 or epoch + 1 == epochs:
            model.eval()
            with torch.no_grad():
                score = r2_raw(hold_y.cpu().numpy(), model(hold_x).cpu().numpy())
            if score > best_score:
                best_score, best_epoch = score, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test = r2_raw(yv.cpu().numpy(), model(xv).cpu().numpy())
        train = r2_raw(yt.cpu().numpy(), model(xt).cpu().numpy())
    del model, xt, yt, xv, yv
    cleanup()
    return dict(r2=test, r2_train=train, r2_internal_holdout=best_score,
                best_epoch=best_epoch)


def safe_ridge(x_train, y_train, x_test, y_test):
    """float64 unless the design matrix would blow past ~1.5 GB per copy."""
    if (x_train.size + x_test.size) * 8 >= 1.5e9:
        cast = np.float32
        print("      (ridge in float32: the raw design matrix is too large for float64)",
              flush=True)
    else:
        cast = np.float64
    return _ridge_r2(x_train.astype(cast), y_train.astype(np.float64),
                     x_test.astype(cast), y_test.astype(np.float64), RIDGE_ALPHAS)


def sliding_windows(spikes, window_size):
    view = np.lib.stride_tricks.sliding_window_view(spikes, window_size, axis=0)
    return view.reshape(len(view), -1), window_size // 2


def raw_window_baseline(spikes_train, behavior_train, spikes_test, behavior_test,
                        window_size, device, seed):
    """The ceiling: ridge and an MLP on the flattened raw window."""
    x_train, offset = sliding_windows(spikes_train, window_size)
    x_test, _ = sliding_windows(spikes_test, window_size)
    y_train = behavior_train[offset:offset + len(x_train)]
    y_test = behavior_test[offset:offset + len(x_test)]
    ridge = safe_ridge(x_train, y_train, x_test, y_test)
    mlp = mlp_r2(x_train, y_train, x_test, y_test, epochs=DECODER_EPOCHS,
                 hidden=DECODER_HIDDEN, dropout=DECODER_DROPOUT,
                 learning_rate=DECODER_LR, seed=seed, device=device)
    return dict(ridge=ridge, mlp=mlp, n_features=int(x_train.shape[1]))


def build_model(override, epochs, device, verbose, seed):
    """Dispatch on the '_model' key so both trunks appear in ONE table."""
    override = dict(override)
    factory = MODELS[override.pop("_model", "jigsaw")]
    settings = dict(window_size=WINDOW_SIZE, n_tiles=N_TILES, tile_gap=TILE_GAP,
                    output_dimension=OUTPUT_DIMENSION, num_hidden_units=NUM_HIDDEN_UNITS,
                    head_hidden_units=HEAD_HIDDEN_UNITS, tile_norm=TILE_NORM,
                    neuron_dropout=NEURON_DROPOUT, gain_jitter=GAIN_JITTER,
                    batch_size=BATCH_SIZE, max_epochs=epochs,
                    learning_rate=LEARNING_RATE, device=device, random_state=seed,
                    verbose=verbose)
    settings.update(override)
    # AFTER the update: the epoch-sweep arms override max_epochs, and logging
    # every epochs//10 of the DEFAULT budget would print once or not at all.
    settings.setdefault("log_every", max(1, settings["max_epochs"] // 10)
                        if settings["max_epochs"] else 1)
    return factory(**settings)


def arm_epochs(name, epochs):
    """The budget this arm will actually train for, for logging/validation."""
    return int(ARMS[name].get("max_epochs", epochs))


def trunk_parameters(model):
    return int(sum(p.numel() for p in model.encoder_.trunk.parameters()))


def pretext_report(model, spikes_valid, repeats):
    metrics = model.evaluate_pretext(spikes_valid, max_spans=PRETEXT_SPANS,
                                     repeats=repeats, verbose=False)
    # n_spans is the number of INDEPENDENT span starts. n_draws is larger
    # (repeats x spans) and only reduces measurement noise in the point
    # estimate; it does not buy statistical power, so every test uses n_spans.
    n = metrics["n_spans"]
    exact_chance = metrics["chance_exact_percent"] / 100.0
    exact = versus_chance(round(metrics["exact_accuracy_percent"] / 100.0 * n), n, exact_chance)
    # The decode is an assignment, i.e. a genuine permutation, so 50% is the
    # true chance rate for pair accuracy. This was NOT true of the old argmax
    # decode: argmax can put two tiles in one slot, those ties were scored as
    # errors, and that moved chance to (1-1/K)/2 = 37.5% for a random head and
    # all the way to 0% for a degenerate constant head. The 0.00% the frozen
    # random_encoder printed on 2026-09-22 was that artifact, not a result.
    pair = versus_chance(round(metrics["pair_accuracy_percent"] / 100.0 * n), n, 0.5)
    metrics["exact_test"] = exact
    metrics["pair_test"] = pair
    metrics["pair_resolution_percent"] = detectable_difference(n)
    # "learned" requires beating chance AND both level-sorting shortcuts, by
    # more than the resolution of the measurement itself.
    shortcut = max(metrics["baseline_mean_sort_pair_percent"],
                   metrics["baseline_norm_sort_pair_percent"])
    metrics["shortcut_pair_percent"] = shortcut
    metrics["learned"] = bool(
        (exact["significant"] or pair["significant"])
        and metrics["pair_accuracy_percent"] > shortcut + metrics["pair_resolution_percent"])
    # Held-out position CE above log(K) means the ORDER HEAD is confidently
    # wrong on unseen spans. Flagged, not condemned -- see print_table.
    metrics["memorizing"] = bool(
        metrics["position_cross_entropy"] > 1.02 * metrics["uniform_cross_entropy"])
    return metrics


def run_arm(name, override, data, device, epochs, verbose, seed, repeats):
    spikes_train, behavior_train, spikes_valid, behavior_valid = data
    started = time.time()
    budget = int(override.get("max_epochs", epochs))
    model = build_model(override, epochs, device, verbose, seed)
    model.fit(spikes_train, X_valid=spikes_valid,
              validate_every=max(1, budget // 6) if budget else 1)

    z_train, index_train = model.transform(spikes_train, pad=False, return_indices=True)
    z_valid, index_valid = model.transform(spikes_valid, pad=False, return_indices=True)
    y_train = behavior_train[index_train]
    y_valid = behavior_valid[index_valid]

    mlp = mlp_r2(z_train, y_train, z_valid, y_valid, epochs=DECODER_EPOCHS,
                 hidden=DECODER_HIDDEN, dropout=DECODER_DROPOUT,
                 learning_rate=DECODER_LR, seed=seed, device=device)
    ridge = safe_ridge(z_train, y_train, z_valid, y_valid)
    pretext = pretext_report(model, spikes_valid, repeats)
    trunk = trunk_parameters(model)
    gap = (model.measure_train_eval_gap(spikes_valid)
           if isinstance(model, MobileJigsaw) else None)
    architecture = getattr(model, "version", None) or model.trunk_block
    del model, z_train, z_valid
    cleanup()
    return dict(arm=name, override=override, architecture=architecture, seed=seed,
                epochs=budget, trunk_parameters=trunk, train_eval_gap=gap,
                mlp=mlp, ridge=ridge, pretext=pretext, seconds=time.time() - started)


# --------------------------------------------------------------------------- #
# pooling across seeds
# --------------------------------------------------------------------------- #
def dig(row, *path, default=float("nan")):
    current = row
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current if isinstance(current, (int, float)) and not isinstance(current, bool) \
        else default


def series(rows, *path):
    values = [dig(r, *path) for r in rows]
    return [v for v in values if isinstance(v, float) and math.isfinite(v)
            or isinstance(v, int)]


def mean_of(rows, *path):
    values = series(rows, *path)
    return sum(values) / len(values) if values else float("nan")


def spread_of(rows, *path):
    """max - min across seeds. Zero with one seed, which is the honest answer:
    a single seed reports no spread because it measured none, not because
    there is none."""
    values = series(rows, *path)
    return (max(values) - min(values)) if len(values) > 1 else 0.0


class Pooled:
    """One arm's results across seeds, with mean/spread accessors."""

    def __init__(self, name, rows):
        self.name = name
        self.rows = rows
        self.n_seeds = len(rows)

    def mean(self, *path):
        return mean_of(self.rows, *path)

    def spread(self, *path):
        return spread_of(self.rows, *path)

    @property
    def override(self):
        return self.rows[0]["override"]

    @property
    def is_mobile(self):
        return self.override.get("_model") == "mobile"

    def first(self, key, default=None):
        return self.rows[0].get(key, default)


def pool(per_seed):
    """{seed: {arm: row}} -> {arm: Pooled} keeping only arms that ran cleanly."""
    names = []
    for results in per_seed.values():
        for name in results:
            if name not in names:
                names.append(name)
    pooled = {}
    for name in names:
        rows = [results[name] for results in per_seed.values()
                if name in results and "error" not in results[name]]
        if rows:
            pooled[name] = Pooled(name, rows)
    return pooled


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def contrast(pooled, left, right, label, meaning):
    """Per-seed difference between two arms, printed with every seed shown.

    Averaging first and differencing after would hide seed disagreement, which
    is exactly what a two-seed study is for. Paired per seed, because both arms
    share the seed's data order and initialization draw.
    """
    a, b = pooled.get(left), pooled.get(right)
    if not a or not b:
        return None
    by_seed = {r["seed"]: r for r in b.rows}
    deltas = [(r["seed"], r["mlp"]["r2"] - by_seed[r["seed"]]["mlp"]["r2"])
              for r in a.rows if r["seed"] in by_seed]
    if not deltas:
        return None
    values = [d for _, d in deltas]
    mean = sum(values) / len(values)
    detail = ", ".join(f"seed {s}: {d:+.4f}" for s, d in deltas)
    agree = all(d > 0 for d in values) or all(d < 0 for d in values)
    print(f"  {label:<18}{mean:+.4f}   [{detail}]"
          + ("" if agree or len(values) < 2 else "   <- SEEDS DISAGREE ON THE SIGN"))
    print(f"                    = {left} - {right}: {meaning}")
    return dict(label=label, left=left, right=right, mean=mean,
                per_seed=dict(deltas), seeds_agree=bool(agree))


def print_table(pooled, baselines, seeds):
    if not pooled:
        print("no arm completed")
        return
    rows = list(pooled.values())
    multi = len(seeds) > 1

    # Each family is compared against ITS OWN frozen control. Comparing a
    # MobileNet arm to a plain-trunk random encoder would confound the
    # architecture with the initialization scale, which is not the question.
    controls = {False: pooled.get("random_encoder"), True: pooled.get("mobile_random")}
    control_r2 = {k: (v.mean("mlp", "r2") if v else None) for k, v in controls.items()}

    header = (f"{'arm':<20}{'arch':>10}{'ep':>6}{'R2 mlp':>9}{'spread':>8}"
              f"{'R2 ridge':>10}{'d.random':>10}{'p.ratio':>9}{'pair%':>8}"
              f"{'shortcut':>10}{'exact%':>8}{'chance':>8}{'z':>7}{'ties%':>7}"
              f"{'posCE':>8}{'learned':>9}{'trunk par':>11}{'tr/ev':>8}")
    print("\n" + "=" * len(header))
    print(f"SUMMARY   ({len(seeds)} seed{'s' if multi else ''}: {list(seeds)})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: -r.mean("mlp", "r2")):
        reference = control_r2[row.is_mobile]
        delta = (f"{row.mean('mlp', 'r2') - reference:+.4f}"
                 if reference is not None else "n/a")
        gap = row.first("train_eval_gap")
        gap_text = f"{gap['relative_gap']:.4f}" if gap else "-"
        flag = "*" if row.rows[0]["pretext"].get("memorizing") else " "
        learned = all(r["pretext"]["learned"] for r in row.rows)
        print(f"{row.name:<20}{str(row.first('architecture', '?')):>10}"
              f"{row.first('epochs', 0):>6}"
              f"{row.mean('mlp', 'r2'):>9.4f}{row.spread('mlp', 'r2'):>8.4f}"
              f"{row.mean('ridge', 'r2'):>10.4f}{delta:>10}"
              f"{row.mean('ridge', 'participation_ratio'):>9.2f}"
              f"{row.mean('pretext', 'pair_accuracy_percent'):>8.2f}"
              f"{row.mean('pretext', 'shortcut_pair_percent'):>10.2f}"
              f"{row.mean('pretext', 'exact_accuracy_percent'):>8.2f}"
              f"{row.mean('pretext', 'chance_exact_percent'):>8.2f}"
              f"{row.mean('pretext', 'exact_test', 'z'):>7.2f}"
              f"{row.mean('pretext', 'argmax_tie_rate_percent'):>7.1f}"
              f"{row.mean('pretext', 'position_cross_entropy'):>7.2f}{flag}"
              f"{str(learned):>9}{row.first('trunk_parameters', 0):>11,}{gap_text:>8}")
    print("-" * len(header))
    raw_mlp = mean_of(baselines, "mlp", "r2")
    raw_ridge = mean_of(baselines, "ridge", "r2")
    print(f"{'RAW WINDOW':<20}{'-':>10}{'-':>6}{raw_mlp:>9.4f}"
          f"{spread_of(baselines, 'mlp', 'r2'):>8.4f}{raw_ridge:>10.4f}{'-':>10}"
          f"{mean_of(baselines, 'ridge', 'participation_ratio'):>9.2f}"
          f"   <- CEILING ({baselines[0]['n_features']} features, no encoder at all)")
    print("=" * len(header))

    best = max(rows, key=lambda r: r.mean("mlp", "r2"))
    uniform = rows[0].rows[0]["pretext"]["uniform_cross_entropy"]
    n_spans = int(rows[0].rows[0]["pretext"].get("n_spans", 0))
    n_draws = int(rows[0].rows[0]["pretext"].get("n_draws", n_spans))
    resolution = detectable_difference(n_spans)

    print("\nHOW TO READ THIS")
    print("  R2 mlp    = MLP decoder on the embedding. spread = max-min across seeds;")
    print("              treat any gap smaller than the spread as noise.")
    print("  d.random  = R2 minus the frozen control OF THE SAME FAMILY")
    print("              (random_encoder for the plain trunk, mobile_random for")
    print("              MobileNet). NEGATIVE means training destroyed information,")
    print("              and pretext accuracy cannot rescue a negative number.")
    print("  p.ratio   = participation ratio of the embedding (out of the output")
    print("              dimension). This is the collapse detector: it has tracked")
    print("              ridge R2 almost monotonically in every run so far, from")
    print("              3.4 -> R2 0.005 (order_only) up to 61.8 -> 0.784 (raw).")
    print("  pair %    = order accuracy under an ASSIGNMENT decode, so it is a real")
    print("              permutation and chance is exactly 50%. Tiles are presented")
    print("              in RANDOM order now, so the label is never the identity")
    print("              permutation and an order-dependent head cannot cheat.")
    print("  shortcut  = the BEST of sort-by-mean and sort-by-norm pair accuracy.")
    print("              The model must beat this, not just beat 50%.")
    print(f"  posCE     = held-out position cross-entropy; uniform = {uniform:.3f}.")
    print("              '*' marks posCE above uniform: the ORDER HEAD is confidently")
    print("              wrong out of sample. That is a statement about the head, NOT")
    print("              about the embedding -- the best R2 so far was produced by a")
    print("              model whose posCE was 4-8. Do not cut epochs because of it;")
    print("              pick the budget from the epoch sweep, by R2.")
    print("  trunk par = trunk parameters. Check this before calling MobileNet")
    print("              'efficient': in 1D, v2 at t=6 is LARGER than a plain conv.")
    print("  tr/ev     = relative embedding change between train and eval mode.")
    print("              Should be ~0. Large values mean BatchNorm running stats")
    print("              are wrong for the windows transform actually sees.")

    print(f"\nPRETEXT RESOLUTION  n_spans={n_spans} independent span starts "
          f"({n_draws} draws after repeats)")
    print(f"  +-{resolution:.2f}% at 95% confidence. Any two pair% values closer than")
    print(f"  {resolution:.2f} points are the same number. Before reading a pretext")
    print("  difference as a result, check it clears this bar.")

    # ---- the honest ceiling comparison ---------------------------------- #
    print("\nCEILING, COMPARED LIKE WITH LIKE")
    emb_ridge, emb_mlp = best.mean("ridge", "r2"), best.mean("mlp", "r2")
    print(f"  best arm: {best.name}  ({best.first('architecture')}, "
          f"{best.first('epochs')} epochs)")
    print(f"  ridge vs ridge : embedding {emb_ridge:.4f}  raw {raw_ridge:.4f}  "
          f"-> {emb_ridge - raw_ridge:+.4f}")
    print(f"  mlp   vs mlp   : embedding {emb_mlp:.4f}  raw {raw_mlp:.4f}  "
          f"-> {emb_mlp - raw_mlp:+.4f}")
    print("  RIDGE vs RIDGE IS THE APPLES-TO-APPLES NUMBER. The MLP-on-embedding")
    print("  path is a trained encoder followed by a trained MLP, so it has more")
    print("  nonlinear capacity than an MLP on raw windows and will flatter the")
    print("  embedding. Quote the ridge line unless you also report the raw MLP's")
    print("  train R2, which overfits hard on a flattened window.")
    raw_over = mean_of(baselines, "mlp", "r2_train") - raw_mlp
    best_over = best.mean("mlp", "r2_train") - emb_mlp
    print(f"  overfit gap (train - test) : raw MLP {raw_over:+.4f}, "
          f"{best.name} {best_over:+.4f}")
    if emb_ridge < raw_ridge and emb_mlp > raw_mlp:
        print("  -> so the defensible claim is 'MATCHES the linear ceiling with 64")
        print("     dims instead of {n} raw features', not 'beats the ceiling'."
              .format(n=baselines[0]["n_features"]))

    # ---- family verdicts and control stability --------------------------- #
    print("\nFAMILY VERDICTS")
    for mobile, label in ((False, "plain trunk"), (True, "mobilenet")):
        family = [r for r in rows if r.is_mobile == mobile]
        reference = control_r2[mobile]
        if not family or reference is None:
            continue
        top = max(family, key=lambda r: r.mean("mlp", "r2"))
        verdict = ("training ADDS value" if top.mean("mlp", "r2") > reference
                   else "!! every trained arm is at or below its RANDOM control")
        print(f"  {label:<12} best={top.name} {top.mean('mlp', 'r2'):.4f} "
              f"vs frozen {reference:.4f}: {verdict}")
    warn_control_stability(controls, control_r2, multi)

    trained = [r for r in rows if r.override.get("max_epochs") != 0]
    plain = [r for r in trained if not r.is_mobile]
    mobile = [r for r in trained if r.is_mobile]
    if plain and mobile:
        a = max(r.mean("mlp", "r2") for r in plain)
        b = max(r.mean("mlp", "r2") for r in mobile)
        print(f"  HEAD TO HEAD  plain {a:.4f}  vs  mobilenet {b:.4f}  -> "
              f"{'mobilenet' if b > a else 'plain trunk'} wins by {abs(b - a):.4f}")
        print("                head-to-head needs no control, so it survives an")
        print("                unstable random encoder. Prefer it to d.random.")

    learners = [r.name for r in rows if all(x["pretext"]["learned"] for x in r.rows)]
    print(f"\n  arms that learned the pretext on EVERY seed: {learners or 'NONE'}")
    if not learners:
        print("    At K=4 no arm has ever cleared this bar. K=2 (tiles_2) has, by a")
        print("    wide margin, so the task is learnable -- K=4 is simply too hard")
        print("    for this much data. That does not stop the order TERM from")
        print("    helping R2; the two questions are separate.")

    print_ablations(pooled)
    print_epoch_sweep(rows)


def warn_control_stability(controls, control_r2, multi):
    """A noisy or collapsed frozen control makes d.random unreadable.

    mobile_random swung 0.2142 -> 0.0946 between two seeds while random_encoder
    held at 0.5731 -> 0.5707. A d.random of +0.65 computed against the low draw
    is a statement about the control, not about the architecture.
    """
    messages = []
    for mobile, label in ((False, "random_encoder"), (True, "mobile_random")):
        control = controls[mobile]
        if control is None:
            continue
        spread = control.spread("mlp", "r2")
        if multi and spread > 0.05:
            messages.append(
                f"  !! {label} moved {spread:.4f} between seeds "
                f"(mean {control.mean('mlp', 'r2'):.4f}). d.random for this family is")
            messages.append(
                "     measuring the control, not the architecture. Use HEAD TO HEAD.")
    both = [control_r2[False], control_r2[True]]
    if all(v is not None for v in both) and abs(both[0] - both[1]) > 0.15:
        messages.append(
            f"  !! the two frozen controls disagree by {abs(both[0] - both[1]):.4f} "
            f"(plain {both[0]:.4f}, mobile {both[1]:.4f}).")
        messages.append(
            "     Their initializations are not comparable, so d.random values from")
        messages.append(
            "     different families must NOT be compared with each other.")
    if not multi and any(controls.values()):
        messages.append("  (one seed only: no control-stability estimate. "
                        "mobile_random has been unstable before -- run --seeds.)")
    for line in messages:
        print(line)


def print_ablations(pooled):
    """The three contrasts the whole design exists to produce."""
    print("\nABLATION CONTRASTS (paired per seed, positive = the term HELPS)")
    found = []
    for left, right, label, meaning in (
        ("proposed", "reconstruct_only", "ORDER effect",
         "what the JIGSAW adds on top of reconstruction. This is the thesis."),
        ("proposed", "order_only", "ANCHOR effect",
         "what reconstruction adds on top of the jigsaw. Expected large."),
        ("with_forecast", "proposed", "FORECAST effect",
         "what the next-bin CE adds. Measured NEGATIVE twice; that is why it is off."),
        ("proposed", "random_encoder", "vs FROZEN",
         "what training is worth at all. Must be positive or nothing else counts."),
        ("proposed", "identity_labels", "SHUFFLE effect",
         "identity labels let an order-dependent head cheat; a big gap here is a bug."),
    ):
        result = contrast(pooled, left, right, label, meaning)
        if result:
            found.append(result)
    if not found:
        print("  (none available -- run proposed, reconstruct_only and order_only "
              "together)")
        return
    order = next((f for f in found if f["label"] == "ORDER effect"), None)
    if order:
        print()
        if order["mean"] > 0.01 and order["seeds_agree"]:
            print(f"  => the jigsaw term contributes {order['mean']:+.4f} R2 over a pure")
            print("     reconstruction autoencoder, consistently across seeds. That is")
            print("     the positive result, and it is worth stating exactly this way:")
            print("     as a contribution ON TOP of an anchor, not as a standalone method.")
        elif order["mean"] <= 0.01:
            print("  => the jigsaw term is NOT contributing over plain reconstruction.")
            print("     Report that. A negative result about the pretext task is still")
            print("     a result; dressing up reconstruction as a jigsaw paper is not.")
        else:
            print("  => seeds disagree on the sign of the order effect. Run more seeds")
            print("     before claiming anything in either direction.")


def print_epoch_sweep(rows):
    # Only arms that differ from `proposed` in max_epochs ALONE belong on this
    # curve: mixing objectives in would make it a function of two variables.
    ladder = [r for r in rows
              if set(r.override) <= {"max_epochs"} and r.override.get("max_epochs") != 0]
    if len({r.first("epochs", 0) for r in ladder}) <= 1:
        return
    print("\nEPOCH SWEEP (same objective, budget is the only difference)")
    print(f"  {'epochs':>8}{'R2 mlp':>9}{'spread':>8}{'R2 ridge':>10}{'p.ratio':>9}"
          f"{'pair%':>8}{'posCE':>8}  arm")
    for row in sorted(ladder, key=lambda r: r.first("epochs", 0)):
        print(f"  {row.first('epochs', 0):>8}{row.mean('mlp', 'r2'):>9.4f}"
              f"{row.spread('mlp', 'r2'):>8.4f}{row.mean('ridge', 'r2'):>10.4f}"
              f"{row.mean('ridge', 'participation_ratio'):>9.2f}"
              f"{row.mean('pretext', 'pair_accuracy_percent'):>8.2f}"
              f"{row.mean('pretext', 'position_cross_entropy'):>8.2f}  {row.name}")
    peak = max(ladder, key=lambda r: r.mean("mlp", "r2"))
    biggest = max(ladder, key=lambda r: r.first("epochs", 0))
    print(f"  peak: {peak.first('epochs')} epochs, R2 {peak.mean('mlp', 'r2'):.4f}. "
          f"Rerun every other arm at that budget.")
    if peak is biggest:
        print("  The peak is at the LARGEST budget tried, so the curve has not turned")
        print("  over yet -- extend the ladder before fixing a budget. Note that posCE")
        print("  is probably far above uniform here; that is the order head memorizing")
        print("  and it has not stopped R2 from climbing.")
    print("  A comparison made in the wrong epoch regime is not a comparison")
    print("  between objectives, it is a comparison between overfits.")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[SEED],
                        help="run every arm once per seed and pool. Two seeds is the "
                             "minimum that can tell a result from an initialization.")
    parser.add_argument("--epoch-sweep", action="store_true",
                        help="run the epoch ladder instead of --arms. Do this FIRST: "
                             "every objective comparison is conditional on the budget.")
    parser.add_argument("--list", action="store_true", help="print the arm table and exit")
    parser.add_argument("--data-dir", type=Path, default=PERICH_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--pretext-repeats", type=int, default=PRETEXT_REPEATS,
                        help="independent gap+permutation draws per span start")
    parser.add_argument("--tag", default="JIGSAWNET")
    parser.add_argument("--quiet", action="store_true")
    options = parser.parse_args()

    if options.list:
        width = max(len(k) for k in ARMS)
        for name, override in ARMS.items():
            mark = "*" if name in DEFAULT_ARMS else ("e" if name in EPOCH_ARMS else " ")
            print(f" {mark} {name:<{width}}  {override or '(defaults)'}")
        print("\n* = in the default set,  e = in --epoch-sweep")
        if ALIASES:
            print("\nrenamed arms (old name -> new name):")
            for old, new in ALIASES.items():
                print(f"    {old:<20} -> {new}")
        return

    if options.epoch_sweep:
        options.arms = list(EPOCH_ARMS)
        print(f"epoch sweep: {options.arms}")

    resolved = []
    for name in options.arms:
        if name in ALIASES:
            print(f"note: '{name}' is now '{ALIASES[name]}' "
                  f"(forecast is off by default, so the two collapsed together).")
            name = ALIASES[name]
        if name not in resolved:
            resolved.append(name)
    unknown = [a for a in resolved if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}\nvalid: {sorted(ARMS)}")
    # Each trunk family needs its OWN frozen control, otherwise d.random
    # confounds the architecture with the initialization scale.
    needed = {"random_encoder"}
    if any(ARMS[a].get("_model") == "mobile" for a in resolved):
        needed.add("mobile_random")
    for control in sorted(needed - set(resolved)):
        resolved.append(control)
        print(f"note: added the {control} control -- results are unreadable without it.")
    options.arms = resolved

    seeds = list(dict.fromkeys(options.seeds))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(p for p in options.data_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz files under {options.data_dir}")
    # files = files[:max(1, options.sessions)]
    files = [files[68]]
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for path in files:
        spikes_train, behavior_train, spikes_valid, behavior_valid = load_session(path)
        data = (spikes_train, behavior_train, spikes_valid, behavior_valid)
        out_dir = options.out_dir / f"{options.tag}_{path.stem}_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "#" * 78)
        print(f"# {path.stem}: {spikes_train.shape[1]} neurons, "
              f"{behavior_train.shape[1]} behavior dims | "
              f"train {len(spikes_train)} / valid {len(spikes_valid)} (NPZ split)")
        print(f"# device={device} epochs={options.epochs} seeds={seeds} -> {out_dir}")
        print("#" * 78, flush=True)

        per_seed, baselines = {}, []
        for seed in seeds:
            seed_all(seed)
            print(f"\n[ceiling | seed {seed}] ridge + MLP on the raw window", flush=True)
            baseline = raw_window_baseline(*data, WINDOW_SIZE, device, seed)
            baselines.append(baseline)
            print(f"  raw window: ridge R2={baseline['ridge']['r2']:.4f}  "
                  f"MLP R2={baseline['mlp']['r2']:.4f}  "
                  f"({baseline['n_features']} features)", flush=True)

            results = {}
            for i, name in enumerate(options.arms, 1):
                print(f"\n[seed {seed} | {i}/{len(options.arms)}] {name}  "
                      f"{ARMS[name] or '(defaults)'}"
                      f"  [{arm_epochs(name, options.epochs)} epochs]", flush=True)
                try:
                    results[name] = run_arm(name, ARMS[name], data, device,
                                            options.epochs, not options.quiet,
                                            seed, options.pretext_repeats)
                    row = results[name]
                    print(f"  -> R2 mlp={row['mlp']['r2']:.4f} "
                          f"ridge={row['ridge']['r2']:.4f} "
                          f"p.ratio={row['ridge'].get('participation_ratio', float('nan')):.2f} "
                          f"| pair={row['pretext']['pair_accuracy_percent']:.2f}% "
                          f"(shortcut {row['pretext']['shortcut_pair_percent']:.2f}%, "
                          f"+-{row['pretext']['pair_resolution_percent']:.2f}) "
                          f"| {row['seconds']:.0f}s", flush=True)
                except Exception as error:  # one bad arm must not kill the sweep
                    traceback.print_exc()
                    results[name] = dict(arm=name, seed=seed,
                                         error=f"{type(error).__name__}: {error}")
                    cleanup()
                per_seed[seed] = results
                save_json(out_dir / "results.json",
                          dict(session=path.stem, stamp=stamp, device=device,
                               epochs=options.epochs, seeds=seeds,
                               baselines={str(s): b for s, b
                                          in zip(seeds[:len(baselines)], baselines)},
                               arms={str(s): r for s, r in per_seed.items()}))

        print_table(pool(per_seed), baselines, seeds)
        print(f"\nwritten to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()




# """Perich runner for JigsawNet. Architecture lives in jigsaw_net.py; this file
# only loads data, runs arms, and reports.

#     python run_jigsaw.py                          # default arms
#     python run_jigsaw.py --arms proposed random_encoder dim_128
#     python run_jigsaw.py --list
#     python run_jigsaw.py --epochs 200 --sessions 3

# THE FOUR NUMBERS THAT DECIDE EVERYTHING
#   R2 raw     ridge on the raw window. The CEILING. Not a baseline you may lose
#              to -- if your embedding is below it, the encoder deleted signal.
#   R2 random  the SAME architecture with max_epochs=0, frozen at init. This is
#              what your weights are worth before any learning. If a trained arm
#              is below it, training made things worse, and pretext accuracy is
#              irrelevant at that point.
#   pair %     held-out order accuracy. The decode is now an ASSIGNMENT (a real
#              permutation), so chance is exactly 50% and a degenerate head scores
#              50%, not 0%. Reported next to the sort-by-mean shortcut baseline.
#   p.ratio    participation ratio of the embedding. In the 2026-09-22 run it
#              tracked ridge R2 almost monotonically (order_only: 3.4/64 -> 0.005).
#              A collapsing p.ratio is the failure, and no pretext number fixes it.

# Every arm also prints delta_random = R2 - R2(random_encoder), which is the only
# honest measure of what training contributed.

# EPOCHS ARE A HYPERPARAMETER, NOT A BUDGET. Spans overlap by construction
# (stride 1), so the effective sample size is far below the span count and a big
# epoch count memorizes the pretext task: the 6000-epoch run reached a held-out
# position cross-entropy of 5.7-36.8 against a uniform 1.386, i.e. confidently
# WRONG out of sample. Run `--arms epochs_10 epochs_30 ... ` (or just
# `--epoch-sweep`) before believing any single-epoch-count table.
# """
# import argparse
# import json
# import math
# import time
# import traceback
# from pathlib import Path

# import numpy as np
# import torch
# from torch import nn

# from jigsaw_net import JigsawNet, _ridge_r2
# from mobile_jigsaw import MobileJigsaw

# MODELS = {"jigsaw": JigsawNet, "mobile": MobileJigsaw}

# PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
# OUTPUT_ROOT = Path("/home/mirzaei/sam/result/Aggregate")
# SEED = 8

# WINDOW_SIZE = 10
# N_TILES = 4
# TILE_GAP = (1, 8)
# OUTPUT_DIMENSION = 64
# NUM_HIDDEN_UNITS = 64
# HEAD_HIDDEN_UNITS = 64
# BATCH_SIZE = 512
# EPOCHS = 60
# LEARNING_RATE = 1e-3
# TILE_NORM = "mean"
# NEURON_DROPOUT = 0.1
# GAIN_JITTER = 0.1

# DECODER_EPOCHS = 2500
# DECODER_HIDDEN = 64
# DECODER_DROPOUT = 0.4
# DECODER_LR = 1e-3
# PRETEXT_SPANS = 2048
# RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)

# # --------------------------------------------------------------------------- #
# # arms. No contrastive arm exists -- the model has no contrastive loss.
# # --------------------------------------------------------------------------- #
# ARMS = {
#     # ---- the proposal ---------------------------------------------------- #
#     "proposed": {},

#     # ---- controls that can falsify it ----------------------------------- #
#     # Frozen at init. THE reference point. Beat this or nothing else matters.
#     "random_encoder": dict(max_epochs=0),
#     # Order CE only: the classic jigsaw. Nothing keeps firing-rate level.
#     # This is the arm that collapsed to p.ratio 3.4/64 and R2 0.005.
#     "order_only": dict(lambda_forecast=0.0, lambda_reconstruct=0.0),
#     # Forecast CE only: is the puzzle contributing anything at all?
#     "forecast_only": dict(lambda_order=0.0, lambda_pair=0.0, lambda_reconstruct=0.0),
#     # The anti-collapse anchor on its own: a pure denoising autoencoder in CE
#     # form, no puzzle at all. If THIS is the best arm, the jigsaw is dead weight
#     # and you should say so in the paper rather than hide it.
#     "reconstruct_only": dict(lambda_order=0.0, lambda_pair=0.0, lambda_forecast=0.0),
#     # The proposal MINUS the anchor == the model that produced the 2026-09-22
#     # table. Its job is to reproduce the collapse, so the anchor's effect is
#     # measured against the right thing and not against a random encoder.
#     "no_reconstruct": dict(lambda_reconstruct=0.0),
#     # Puzzle + anchor, no forecasting: is forecast doing anything the
#     # reconstruction is not already doing? They are both "keep the rate".
#     "order_reconstruct": dict(lambda_forecast=0.0),
#     # Anchor turned up. If R2 keeps climbing with this, the bottleneck -- not
#     # the pretext task -- is what the whole experiment is measuring.
#     "reconstruct_heavy": dict(lambda_reconstruct=4.0),
#     # Order head trains on a trunk it cannot influence. Asks "is order
#     # decodable from a forecast-trained trunk?" with zero risk to R2.
#     "order_probe_only": dict(order_grad_scale=0.0),
#     # Every anti-shortcut guard off. If pretext accuracy JUMPS here, the task
#     # was being solved by level drift, not by temporal structure.
#     "shortcut_open": dict(tile_norm="none", neuron_dropout=0.0, gain_jitter=0.0),

#     # ---- epochs. The 6000-epoch run memorized the pretext task ----------- #
#     # Held-out position CE was 5.7-36.8 against a uniform 1.386: confidently
#     # wrong, which is overfitting, not underfitting. Spans overlap at stride 1,
#     # so ~2400 spans are worth far fewer independent samples than that.
#     "epochs_10": dict(max_epochs=10),
#     "epochs_30": dict(max_epochs=30),
#     "epochs_100": dict(max_epochs=100),
#     "epochs_300": dict(max_epochs=300),
#     "epochs_1000": dict(max_epochs=1000),

#     # ---- the bottleneck, which the numpy analysis says costs the most ---- #
#     "dim_16": dict(output_dimension=16),
#     "dim_32": dict(output_dimension=32),
#     "dim_128": dict(output_dimension=128),
#     "dim_256": dict(output_dimension=256),

#     # ---- ablations ------------------------------------------------------- #
#     # The sphere. Expected to LOSE: it deletes magnitude.
#     "normalized": dict(normalize=True),
#     # A single MobileNetV2-style block dropped into the PLAIN trunk. Cheap
#     # sanity check; for the real architecture use the mobile_* arms below.
#     "separable_block": dict(trunk_block="separable"),
#     # Difficulty ladder. n_tiles=2 is the most sensitive learnability test
#     # there is: chance is exactly 50%, so tiny effects are detectable.
#     "tiles_2": dict(n_tiles=2),
#     "tiles_3": dict(n_tiles=3),
#     "tiles_6": dict(n_tiles=6),
#     # Long-range order: are far-apart tiles easier (more drift) or harder?
#     "wide_gaps": dict(tile_gap=(8, 40)),
#     "tight_gaps": dict(tile_gap=(1, 2)),
#     # Coarser / finer forecast targets.
#     "forecast_4": dict(forecast_levels=4),
#     "forecast_16": dict(forecast_levels=16),
#     "zscore_tiles": dict(tile_norm="zscore"),

#     # ---- MOBILENET TRUNKS ------------------------------------------------ #
#     # Same losses, same tiles, same seed -- only the trunk changes, so any
#     # difference here is attributable to the architecture. "_model" is consumed
#     # by build_model and is not a constructor argument.
#     "mobile_v1": dict(_model="mobile", version="v1"),
#     "mobile_v2": dict(_model="mobile", version="v2", expansion=3),
#     # The paper's t=6. In 1D this is ~4x MORE parameters than a plain conv, so
#     # it is a capacity arm, not an efficiency arm. Label it honestly.
#     "mobile_v2_t6": dict(_model="mobile", version="v2", expansion=6),
#     "mobile_v3": dict(_model="mobile", version="v3"),
#     # Image-style stem: mixes neurons immediately instead of giving each neuron
#     # its own temporal filter. Isolates the EEGNet-style inductive bias.
#     "mobile_stem_mix": dict(_model="mobile", stem="mix"),
#     # alpha, MobileNet's real efficiency knob. Small alpha = regularization,
#     # which is the plausible win on short sessions.
#     "mobile_alpha_035": dict(_model="mobile", width_multiplier=0.35),
#     "mobile_alpha_050": dict(_model="mobile", width_multiplier=0.5),
#     "mobile_alpha_140": dict(_model="mobile", width_multiplier=1.4),
#     # Canonical MobileNet normalization. Expected to LOSE: running statistics
#     # are collected on the augmented/normalized training views and then applied
#     # to the raw windows transform sees. The table prints the measured gap.
#     "mobile_batchnorm": dict(_model="mobile", norm="batch"),
#     "mobile_hardswish": dict(_model="mobile", activation="hardswish"),
#     # MobileNet's final 1x1 before the classifier.
#     "mobile_head_256": dict(_model="mobile", head_channels=256),
#     # The control that makes the whole comparison readable.
#     "mobile_random": dict(_model="mobile", max_epochs=0),
# }

# DEFAULT_ARMS = ("proposed", "no_reconstruct", "random_encoder", "order_only",
#                 "forecast_only", "reconstruct_only", "order_reconstruct",
#                 "shortcut_open", "dim_256", "tiles_2",
#                 "mobile_v1", "mobile_v2", "mobile_v3", "mobile_alpha_035",
#                 "mobile_stem_mix", "mobile_random")

# # The epoch sweep. Run this FIRST on a new session: every other comparison is
# # conditional on being in a sane epoch regime, and the previous table was not.
# EPOCH_ARMS = ("epochs_10", "epochs_30", "epochs_100", "epochs_300",
#               "epochs_1000", "random_encoder")

# # Arms grouped by trunk, so the table can be read per-architecture.
# MOBILE_ARMS = tuple(k for k, v in ARMS.items() if v.get("_model") == "mobile")


# # --------------------------------------------------------------------------- #
# # helpers
# # --------------------------------------------------------------------------- #
# def seed_all(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def json_safe(obj):
#     """NaN/Inf -> None so one broken arm cannot poison the whole dump."""
#     if isinstance(obj, dict):
#         return {str(k): json_safe(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)):
#         return [json_safe(v) for v in obj]
#     if isinstance(obj, (np.integer,)):
#         return int(obj)
#     if isinstance(obj, (np.floating, float)):
#         value = float(obj)
#         return value if math.isfinite(value) else None
#     if isinstance(obj, np.ndarray):
#         return json_safe(obj.tolist())
#     if isinstance(obj, (np.bool_, bool)):
#         return bool(obj)
#     if isinstance(obj, Path):
#         return str(obj)
#     return obj


# def save_json(path, payload):
#     path.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False))


# def cleanup():
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


# def normal_tail(z):
#     return 0.5 * math.erfc(z / math.sqrt(2.0))


# def versus_chance(successes, trials, chance):
#     """Wilson 95% CI plus a one-sided z-test against a KNOWN chance rate.

#     The chance rate here is known exactly (1/K! or 1/2), not estimated, so a
#     one-sample z-test is the right test and no correction is needed.
#     """
#     if trials <= 0:
#         return dict(rate=float("nan"), low=float("nan"), high=float("nan"),
#                     z=float("nan"), p=float("nan"), significant=False, n=0)
#     rate = successes / trials
#     z_critical = 1.959963985
#     denominator = 1 + z_critical ** 2 / trials
#     centre = (rate + z_critical ** 2 / (2 * trials)) / denominator
#     spread = z_critical * math.sqrt(rate * (1 - rate) / trials
#                                     + z_critical ** 2 / (4 * trials ** 2)) / denominator
#     standard_error = math.sqrt(max(chance * (1 - chance) / trials, 1e-300))
#     z = (rate - chance) / standard_error
#     p = normal_tail(z)
#     return dict(rate=rate, low=max(0.0, centre - spread), high=min(1.0, centre + spread),
#                 z=z, p=p, significant=bool(p < 0.05 and rate > chance), n=int(trials))


# def load_session(path):
#     """Load the existing NPZ train/validation split without splitting again.

#     Keep the original runner's first-two-label selection. The neuron mask is
#     fitted on training data only and applied identically to both splits.
#     """
#     path = Path(path)
#     required = ("train_data", "train_label", "valid_data", "valid_label")
#     with np.load(path, allow_pickle=False) as handle:
#         missing = [key for key in required if key not in handle.files]
#         if missing:
#             raise KeyError(
#                 f"{path.name}: missing NPZ keys {missing}; has {sorted(handle.files)}")
#         arrays = [np.asarray(handle[key], dtype=np.float32) for key in required]

#     splits = []
#     for name, spikes, behavior in (("train", arrays[0], arrays[1]),
#                                   ("valid", arrays[2], arrays[3])):
#         if behavior.ndim == 1:
#             behavior = behavior[:, None]
#         if spikes.ndim != 2 or behavior.ndim != 2:
#             raise ValueError(
#                 f"{path.name}: {name} data/labels must be 2D; "
#                 f"got {spikes.shape} and {behavior.shape}")
#         # Preserve support for neural arrays stored as (neurons, time).
#         if len(spikes) != len(behavior) and spikes.shape[1] == len(behavior):
#             spikes = spikes.T
#         if len(spikes) != len(behavior):
#             raise ValueError(
#                 f"{path.name}: {name} data/label length mismatch: "
#                 f"{len(spikes)} vs {len(behavior)}")
#         if len(spikes) == 0 or spikes.shape[1] == 0 or behavior.shape[1] == 0:
#             raise ValueError(f"{path.name}: {name} contains an empty array")
#         if not (np.isfinite(spikes).all() and np.isfinite(behavior).all()):
#             raise ValueError(f"{path.name}: {name} contains NaN or Inf")
#         splits.append((spikes, behavior))

#     spikes_train, behavior_train = splits[0]
#     spikes_valid, behavior_valid = splits[1]
#     if spikes_train.shape[1] != spikes_valid.shape[1]:
#         raise ValueError(f"{path.name}: train/valid neuron counts differ")
#     if behavior_train.shape[1] != behavior_valid.shape[1]:
#         raise ValueError(f"{path.name}: train/valid label counts differ")

#     # Same label selection as the supplied runner; no assumption about names.
#     behavior_train = behavior_train[:, :2]
#     behavior_valid = behavior_valid[:, :2]
#     alive = spikes_train.std(0) > 0
#     if not alive.any():
#         raise ValueError(f"{path.name}: no varying neurons in train_data")
#     return (np.ascontiguousarray(spikes_train[:, alive]),
#             np.ascontiguousarray(behavior_train),
#             np.ascontiguousarray(spikes_valid[:, alive]),
#             np.ascontiguousarray(behavior_valid))


# class Decoder(nn.Module):
#     def __init__(self, dimension, targets, hidden, dropout):
#         super().__init__()
#         self.net = nn.Sequential(nn.Linear(dimension, hidden), nn.LayerNorm(hidden),
#                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, targets))

#     def forward(self, x):
#         return self.net(x)


# def r2_raw(truth, prediction):
#     residual = np.square(truth - prediction).sum(0)
#     total = np.square(truth - truth.mean(0, keepdims=True)).sum(0)
#     return float(np.mean(1.0 - residual / np.maximum(total, 1e-12)))


# def mlp_r2(x_train, y_train, x_test, y_test, *, epochs, hidden, dropout,
#            learning_rate, seed, device):
#     """Chronological internal split for model selection -- never random.

#     A random split lets adjacent, near-identical bins land on both sides, and
#     the score is then optimistic by a wide margin on autocorrelated data.
#     """
#     torch.manual_seed(seed)
#     mean, scale = x_train.mean(0, keepdims=True), x_train.std(0, keepdims=True) + 1e-8
#     cut = max(1, int(0.9 * len(x_train)))
#     tensors = [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
#                for a in ((x_train - mean) / scale, y_train, (x_test - mean) / scale, y_test)]
#     xt, yt, xv, yv = tensors
#     inner_x, inner_y, hold_x, hold_y = xt[:cut], yt[:cut], xt[cut:], yt[cut:]
#     if len(hold_x) < 2:
#         inner_x, inner_y, hold_x, hold_y = xt, yt, xt, yt
#     model = Decoder(xt.shape[1], yt.shape[1], hidden, dropout).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     best_state, best_score, best_epoch = None, -float("inf"), 0
#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad(set_to_none=True)
#         nn.functional.mse_loss(model(inner_x), inner_y).backward()
#         optimizer.step()
#         if (epoch + 1) % 25 == 0 or epoch + 1 == epochs:
#             model.eval()
#             with torch.no_grad():
#                 score = r2_raw(hold_y.cpu().numpy(), model(hold_x).cpu().numpy())
#             if score > best_score:
#                 best_score, best_epoch = score, epoch + 1
#                 best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
#     if best_state is not None:
#         model.load_state_dict(best_state)
#     model.eval()
#     with torch.no_grad():
#         test = r2_raw(yv.cpu().numpy(), model(xv).cpu().numpy())
#         train = r2_raw(yt.cpu().numpy(), model(xt).cpu().numpy())
#     del model, xt, yt, xv, yv
#     cleanup()
#     return dict(r2=test, r2_train=train, r2_internal_holdout=best_score,
#                 best_epoch=best_epoch)


# def safe_ridge(x_train, y_train, x_test, y_test):
#     """float64 unless the design matrix would blow past ~1.5 GB per copy."""
#     if (x_train.size + x_test.size) * 8 >= 1.5e9:
#         cast = np.float32
#         print("      (ridge in float32: the raw design matrix is too large for float64)",
#               flush=True)
#     else:
#         cast = np.float64
#     return _ridge_r2(x_train.astype(cast), y_train.astype(np.float64),
#                      x_test.astype(cast), y_test.astype(np.float64), RIDGE_ALPHAS)


# def sliding_windows(spikes, window_size):
#     view = np.lib.stride_tricks.sliding_window_view(spikes, window_size, axis=0)
#     return view.reshape(len(view), -1), window_size // 2


# def raw_window_baseline(spikes_train, behavior_train, spikes_test, behavior_test,
#                         window_size, device):
#     """The ceiling: ridge and an MLP on the flattened raw window."""
#     x_train, offset = sliding_windows(spikes_train, window_size)
#     x_test, _ = sliding_windows(spikes_test, window_size)
#     y_train = behavior_train[offset:offset + len(x_train)]
#     y_test = behavior_test[offset:offset + len(x_test)]
#     ridge = safe_ridge(x_train, y_train, x_test, y_test)
#     mlp = mlp_r2(x_train, y_train, x_test, y_test, epochs=DECODER_EPOCHS,
#                  hidden=DECODER_HIDDEN, dropout=DECODER_DROPOUT,
#                  learning_rate=DECODER_LR, seed=SEED, device=device)
#     return dict(ridge=ridge, mlp=mlp, n_features=int(x_train.shape[1]))


# def build_model(override, epochs, device, verbose):
#     """Dispatch on the '_model' key so both trunks appear in ONE table."""
#     override = dict(override)
#     factory = MODELS[override.pop("_model", "jigsaw")]
#     settings = dict(window_size=WINDOW_SIZE, n_tiles=N_TILES, tile_gap=TILE_GAP,
#                     output_dimension=OUTPUT_DIMENSION, num_hidden_units=NUM_HIDDEN_UNITS,
#                     head_hidden_units=HEAD_HIDDEN_UNITS, tile_norm=TILE_NORM,
#                     neuron_dropout=NEURON_DROPOUT, gain_jitter=GAIN_JITTER,
#                     batch_size=BATCH_SIZE, max_epochs=epochs,
#                     learning_rate=LEARNING_RATE, device=device, random_state=SEED,
#                     verbose=verbose)
#     settings.update(override)
#     # AFTER the update: the epoch-sweep arms override max_epochs, and logging
#     # every epochs//10 of the DEFAULT budget would print once or not at all.
#     settings.setdefault("log_every", max(1, settings["max_epochs"] // 10)
#                         if settings["max_epochs"] else 1)
#     return factory(**settings)


# def arm_epochs(name, epochs):
#     """The budget this arm will actually train for, for logging/validation."""
#     return int(ARMS[name].get("max_epochs", epochs))


# def trunk_parameters(model):
#     return int(sum(p.numel() for p in model.encoder_.trunk.parameters()))


# def pretext_report(model, spikes_valid):
#     metrics = model.evaluate_pretext(spikes_valid, max_spans=PRETEXT_SPANS, verbose=False)
#     n = metrics["n_spans"]
#     exact_chance = metrics["chance_exact_percent"] / 100.0
#     exact = versus_chance(round(metrics["exact_accuracy_percent"] / 100.0 * n), n, exact_chance)
#     # The decode is an assignment, i.e. a genuine permutation, so 50% is the
#     # true chance rate for pair accuracy. This was NOT true of the old argmax
#     # decode: argmax can put two tiles in one slot, those ties were scored as
#     # errors, and that moved chance to (1-1/K)/2 = 37.5% for a random head and
#     # all the way to 0% for a degenerate constant head. The 0.00% the frozen
#     # random_encoder printed on 2026-09-22 was that artifact, not a result.
#     pair = versus_chance(round(metrics["pair_accuracy_percent"] / 100.0 * n), n, 0.5)
#     metrics["exact_test"] = exact
#     metrics["pair_test"] = pair
#     # "learned" requires beating chance AND both level-sorting shortcuts.
#     shortcut = max(metrics["baseline_mean_sort_pair_percent"],
#                    metrics["baseline_norm_sort_pair_percent"])
#     metrics["shortcut_pair_percent"] = shortcut
#     metrics["learned"] = bool(
#         (exact["significant"] or pair["significant"])
#         and metrics["pair_accuracy_percent"] > shortcut + 1.0)
#     # Held-out position CE above log(K) means the head is confidently WRONG on
#     # unseen spans -- memorization, which more epochs makes worse, not better.
#     # The 2% margin matters: a frozen encoder sits AT uniform by construction,
#     # and without a tolerance sampling noise flags every random control.
#     metrics["memorizing"] = bool(
#         metrics["position_cross_entropy"] > 1.02 * metrics["uniform_cross_entropy"])
#     return metrics


# def run_arm(name, override, data, device, epochs, verbose):
#     spikes_train, behavior_train, spikes_valid, behavior_valid = data
#     started = time.time()
#     budget = int(override.get("max_epochs", epochs))
#     model = build_model(override, epochs, device, verbose)
#     model.fit(spikes_train, X_valid=spikes_valid,
#               validate_every=max(1, budget // 6) if budget else 1)

#     z_train, index_train = model.transform(spikes_train, pad=False, return_indices=True)
#     z_valid, index_valid = model.transform(spikes_valid, pad=False, return_indices=True)
#     y_train = behavior_train[index_train]
#     y_valid = behavior_valid[index_valid]

#     mlp = mlp_r2(z_train, y_train, z_valid, y_valid, epochs=DECODER_EPOCHS,
#                  hidden=DECODER_HIDDEN, dropout=DECODER_DROPOUT,
#                  learning_rate=DECODER_LR, seed=SEED, device=device)
#     ridge = safe_ridge(z_train, y_train, z_valid, y_valid)
#     pretext = pretext_report(model, spikes_valid)
#     trunk = trunk_parameters(model)
#     gap = (model.measure_train_eval_gap(spikes_valid)
#            if isinstance(model, MobileJigsaw) else None)
#     architecture = getattr(model, "version", None) or model.trunk_block
#     del model, z_train, z_valid
#     cleanup()
#     return dict(arm=name, override=override, architecture=architecture,
#                 epochs=budget, trunk_parameters=trunk, train_eval_gap=gap,
#                 mlp=mlp, ridge=ridge, pretext=pretext, seconds=time.time() - started)


# def print_table(results, baseline):
#     rows = [r for r in results.values() if "error" not in r]
#     if not rows:
#         print("no arm completed")
#         return

#     def is_mobile(row):
#         return row["override"].get("_model") == "mobile"

#     # Each family is compared against ITS OWN frozen control. Comparing a
#     # MobileNet arm to a plain-trunk random encoder would confound the
#     # architecture with the initialization scale, which is not the question.
#     controls = {False: results.get("random_encoder", {}).get("mlp", {}).get("r2"),
#                 True: results.get("mobile_random", {}).get("mlp", {}).get("r2")}
#     header = (f"{'arm':<20}{'arch':>10}{'ep':>6}{'R2 mlp':>9}{'R2 ridge':>10}"
#               f"{'d.random':>10}{'p.ratio':>9}{'pair%':>8}{'shortcut':>10}"
#               f"{'exact%':>8}{'chance':>8}{'z':>7}{'ties%':>7}{'posCE':>8}"
#               f"{'learned':>9}{'trunk par':>11}{'tr/ev':>8}")
#     print("\n" + "=" * len(header))
#     print("SUMMARY")
#     print("=" * len(header))
#     print(header)
#     print("-" * len(header))
#     for row in sorted(rows, key=lambda r: -r["mlp"]["r2"]):
#         p = row["pretext"]
#         reference = controls[is_mobile(row)]
#         delta = f"{row['mlp']['r2'] - reference:+.4f}" if reference is not None else "n/a"
#         gap = row.get("train_eval_gap")
#         gap_text = f"{gap['relative_gap']:.4f}" if gap else "-"
#         flag = "*" if p.get("memorizing") else " "
#         print(f"{row['arm']:<20}{str(row.get('architecture', '?')):>10}"
#               f"{row.get('epochs', 0):>6}"
#               f"{row['mlp']['r2']:>9.4f}{row['ridge']['r2']:>10.4f}{delta:>10}"
#               f"{row['ridge'].get('participation_ratio', float('nan')):>9.2f}"
#               f"{p['pair_accuracy_percent']:>8.2f}"
#               f"{p.get('shortcut_pair_percent', p['baseline_mean_sort_pair_percent']):>10.2f}"
#               f"{p['exact_accuracy_percent']:>8.2f}{p['chance_exact_percent']:>8.2f}"
#               f"{p['exact_test']['z']:>7.2f}{p.get('argmax_tie_rate_percent', 0.0):>7.1f}"
#               f"{p['position_cross_entropy']:>7.2f}{flag}{str(p['learned']):>9}"
#               f"{row.get('trunk_parameters', 0):>11,}{gap_text:>8}")
#     print("-" * len(header))
#     print(f"{'RAW WINDOW':<20}{'-':>10}{'-':>6}{baseline['mlp']['r2']:>9.4f}"
#           f"{baseline['ridge']['r2']:>10.4f}{'-':>10}"
#           f"{baseline['ridge'].get('participation_ratio', float('nan')):>9.2f}"
#           f"   <- CEILING ({baseline['n_features']} features, no encoder at all)")
#     print("=" * len(header))

#     best = max(rows, key=lambda r: r["mlp"]["r2"])
#     ceiling = max(baseline["mlp"]["r2"], baseline["ridge"]["r2"])
#     uniform = rows[0]["pretext"]["uniform_cross_entropy"]
#     print("\nHOW TO READ THIS")
#     print("  d.random  = R2 minus the frozen control OF THE SAME FAMILY")
#     print("              (random_encoder for the plain trunk, mobile_random for")
#     print("              MobileNet). NEGATIVE means training destroyed information,")
#     print("              and pretext accuracy cannot rescue a negative number.")
#     print("  p.ratio   = participation ratio of the embedding (out of the output")
#     print("              dimension). This is the collapse detector: on 2026-09-22 it")
#     print("              tracked ridge R2 almost monotonically, from 3.4 -> R2 0.005")
#     print("              (order_only) up to 61.8 -> 0.784 (the raw window itself).")
#     print("              If p.ratio falls, R2 falls, whatever the pretext says.")
#     print("  pair %    = order accuracy under an ASSIGNMENT decode, so it is a real")
#     print("              permutation and chance is exactly 50%. The older argmax")
#     print("              decode was not a permutation, its ties were scored as")
#     print("              errors, and chance was silently 37.5% (or 0% for a")
#     print("              degenerate head). argmax is still reported in results.json")
#     print("              as argmax_pair_accuracy_percent for comparison.")
#     print("  shortcut  = the BEST of sort-by-mean and sort-by-norm pair accuracy.")
#     print("              The model must beat this, not just beat 50%.")
#     print(f"  posCE     = held-out position cross-entropy; uniform = {uniform:.3f}.")
#     print("              A '*' marks posCE ABOVE uniform, i.e. the head is")
#     print("              confidently WRONG out of sample. That is memorization and")
#     print("              more epochs make it worse. Run the epoch sweep.")
#     print("  trunk par = trunk parameters. Check this before calling MobileNet")
#     print("              'efficient': in 1D, v2 at t=6 is LARGER than a plain conv.")
#     print("  tr/ev     = relative embedding change between train and eval mode.")
#     print("              Should be ~0. Large values mean BatchNorm running stats")
#     print("              are wrong for the windows transform actually sees.")
#     print(f"\nbest arm: {best['arm']}  R2={best['mlp']['r2']:.4f}  ({best.get('architecture')})")
#     for mobile, label in ((False, "plain trunk"), (True, "mobilenet")):
#         family = [r for r in rows if is_mobile(r) == mobile]
#         reference = controls[mobile]
#         if not family or reference is None:
#             continue
#         top = max(family, key=lambda r: r["mlp"]["r2"])
#         verdict = ("training ADDS value" if top["mlp"]["r2"] > reference
#                    else "!! every trained arm is at or below its RANDOM control")
#         print(f"  {label:<12} best={top['arm']} {top['mlp']['r2']:.4f} "
#               f"vs frozen {reference:.4f}: {verdict}")
#     trained = [r for r in rows if r["override"].get("max_epochs") != 0]
#     plain = [r for r in trained if not is_mobile(r)]
#     mobile = [r for r in trained if is_mobile(r)]
#     if plain and mobile:
#         a = max(r["mlp"]["r2"] for r in plain)
#         b = max(r["mlp"]["r2"] for r in mobile)
#         print(f"  HEAD TO HEAD  plain {a:.4f}  vs  mobilenet {b:.4f}  -> "
#               f"{'mobilenet' if b > a else 'plain trunk'} wins by {abs(b - a):.4f}")
#     gap = best["mlp"]["r2"] - ceiling
#     print(f"  vs raw-window ceiling ({ceiling:.4f}): {gap:+.4f}"
#           + ("" if gap >= 0 else "  <- the encoder is a lossy bottleneck; try a larger"
#                                  " output_dimension (dim_128 / dim_256) first,"
#                                  " the trunk is not the binding constraint"))
#     learners = [r["arm"] for r in rows if r["pretext"]["learned"]]
#     print(f"  arms that actually learned the pretext: {learners or 'NONE'}")
#     if learners and gap < 0:
#         print("  note: pretext learned but R2 still under the ceiling -> the order task")
#         print("        is solvable yet its solution is not what the decoder needs.")

#     # The anchor A/B. This is the only pair that isolates the reconstruction CE.
#     anchored, bare = results.get("proposed"), results.get("no_reconstruct")
#     if anchored and bare and "error" not in anchored and "error" not in bare:
#         d_r2 = anchored["mlp"]["r2"] - bare["mlp"]["r2"]
#         d_pr = (anchored["ridge"].get("participation_ratio", float("nan"))
#                 - bare["ridge"].get("participation_ratio", float("nan")))
#         print(f"\nRECONSTRUCTION ANCHOR  proposed - no_reconstruct: "
#               f"R2 {d_r2:+.4f}, p.ratio {d_pr:+.2f}")
#         print("  no_reconstruct IS the model that produced the 2026-09-22 table, so")
#         print("  this difference -- not the distance to random_encoder -- is what the")
#         print("  anchor is worth. If it is ~0, drop the anchor and report that.")

#     memorizers = [r["arm"] for r in rows if r["pretext"].get("memorizing")]
#     if memorizers:
#         print(f"\n!! held-out position CE more than 2% ABOVE uniform for: {memorizers}")
#         print("   These arms are confidently wrong on unseen spans. That is")
#         print("   overfitting, not underfitting -- cut epochs, do not add them.")

#     # Epoch sweep. Only arms that differ from `proposed` in max_epochs ALONE
#     # belong on this curve: mixing objectives in would make it a function of
#     # two variables and unreadable as a budget curve.
#     ladder = [r for r in rows
#               if set(r["override"]) <= {"max_epochs"} and r["override"].get("max_epochs") != 0]
#     if len({r.get("epochs", 0) for r in ladder}) > 1:
#         print("\nEPOCH SWEEP (same objective, budget is the only difference)")
#         print(f"  {'epochs':>8}{'R2 mlp':>9}{'p.ratio':>9}{'pair%':>8}{'posCE':>8}  arm")
#         for row in sorted(ladder, key=lambda r: r.get("epochs", 0)):
#             p = row["pretext"]
#             print(f"  {row.get('epochs', 0):>8}{row['mlp']['r2']:>9.4f}"
#                   f"{row['ridge'].get('participation_ratio', float('nan')):>9.2f}"
#                   f"{p['pair_accuracy_percent']:>8.2f}"
#                   f"{p['position_cross_entropy']:>8.2f}  {row['arm']}")
#         peak = max(ladder, key=lambda r: r["mlp"]["r2"])
#         print(f"  peak: {peak.get('epochs')} epochs, R2 {peak['mlp']['r2']:.4f}. "
#               f"Rerun every other arm at that budget.")
#         print("  A comparison made in the wrong epoch regime is not a comparison")
#         print("  between objectives, it is a comparison between overfits.")


# def main():
#     parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
#     parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
#     parser.add_argument("--epoch-sweep", action="store_true",
#                         help="run the epoch ladder instead of --arms. Do this FIRST: "
#                              "every objective comparison is conditional on the budget.")
#     parser.add_argument("--list", action="store_true", help="print the arm table and exit")
#     parser.add_argument("--data-dir", type=Path, default=PERICH_DATA_DIR)
#     parser.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT)
#     parser.add_argument("--sessions", type=int, default=1)
#     parser.add_argument("--epochs", type=int, default=EPOCHS)
#     parser.add_argument("--tag", default="JIGSAWNET")
#     parser.add_argument("--quiet", action="store_true")
#     options = parser.parse_args()

#     if options.list:
#         width = max(len(k) for k in ARMS)
#         for name, override in ARMS.items():
#             mark = "*" if name in DEFAULT_ARMS else ("e" if name in EPOCH_ARMS else " ")
#             print(f" {mark} {name:<{width}}  {override or '(defaults)'}")
#         print("\n* = in the default set,  e = in --epoch-sweep")
#         return

#     if options.epoch_sweep:
#         options.arms = list(EPOCH_ARMS)
#         print(f"epoch sweep: {options.arms}")

#     unknown = [a for a in options.arms if a not in ARMS]
#     if unknown:
#         raise SystemExit(f"unknown arm(s) {unknown}\nvalid: {sorted(ARMS)}")
#     # Each trunk family needs its OWN frozen control, otherwise d.random
#     # confounds the architecture with the initialization scale.
#     selected = list(options.arms)
#     needed = {"random_encoder"}
#     if any(ARMS[a].get("_model") == "mobile" for a in selected):
#         needed.add("mobile_random")
#     for control in sorted(needed - set(selected)):
#         selected.append(control)
#         print(f"note: added the {control} control -- results are unreadable without it.")
#     options.arms = selected

#     seed_all(SEED)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     files = sorted(p for p in options.data_dir.glob("*.npz"))
#     if not files:
#         raise SystemExit(f"no .npz files under {options.data_dir}")
#     files = files[:max(1, options.sessions)]
#     stamp = time.strftime("%Y%m%d_%H%M%S")

#     for path in files:
#         data = load_session(path)
#         spikes_train, behavior_train, spikes_valid, behavior_valid = data
#         out_dir = options.out_dir / f"{options.tag}_{path.stem}_{stamp}"
#         out_dir.mkdir(parents=True, exist_ok=True)
#         print("\n" + "#" * 78)
#         print(f"# {path.stem}: {len(spikes_train) + len(spikes_valid)} bins, "
#               f"{spikes_train.shape[1]} neurons, {behavior_train.shape[1]} behavior dims | "
#               f"train {len(spikes_train)} / valid {len(spikes_valid)} (NPZ split)")
#         print(f"# device={device} epochs={options.epochs} -> {out_dir}")
#         print("#" * 78, flush=True)

#         print("\n[ceiling] ridge + MLP on the raw window", flush=True)
#         baseline = raw_window_baseline(*data, WINDOW_SIZE, device)
#         print(f"  raw window: ridge R2={baseline['ridge']['r2']:.4f}  "
#               f"MLP R2={baseline['mlp']['r2']:.4f}  ({baseline['n_features']} features)",
#               flush=True)

#         results = {}
#         for i, name in enumerate(options.arms, 1):
#             print(f"\n[{i}/{len(options.arms)}] {name}  {ARMS[name] or '(defaults)'}"
#                   f"  [{arm_epochs(name, options.epochs)} epochs]", flush=True)
#             try:
#                 results[name] = run_arm(name, ARMS[name], data, device,
#                                         options.epochs, not options.quiet)
#                 row = results[name]
#                 print(f"  -> R2 mlp={row['mlp']['r2']:.4f} ridge={row['ridge']['r2']:.4f} "
#                       f"p.ratio={row['ridge'].get('participation_ratio', float('nan')):.2f} "
#                       f"| pair={row['pretext']['pair_accuracy_percent']:.2f}% "
#                       f"(shortcut {row['pretext']['shortcut_pair_percent']:.2f}%, "
#                       f"posCE {row['pretext']['position_cross_entropy']:.2f} vs "
#                       f"{row['pretext']['uniform_cross_entropy']:.2f} uniform) "
#                       f"| {row['seconds']:.0f}s", flush=True)
#             except Exception as error:  # one bad arm must not kill the sweep
#                 traceback.print_exc()
#                 results[name] = dict(arm=name, error=f"{type(error).__name__}: {error}")
#                 cleanup()
#             save_json(out_dir / "results.json",
#                       dict(session=path.stem, stamp=stamp, device=device,
#                            epochs=options.epochs, baseline=baseline, arms=results))

#         print_table(results, baseline)
#         print(f"\nwritten to {out_dir / 'results.json'}")


# if __name__ == "__main__":
#     main()

"""Perich runner for JigsawNet. Architecture lives in jigsaw_net.py; this file
only loads data, runs arms, and reports.

    python run_jigsaw.py                          # default arms
    python run_jigsaw.py --arms proposed random_encoder dim_128
    python run_jigsaw.py --list
    python run_jigsaw.py --epochs 200 --sessions 3

THE FOUR NUMBERS THAT DECIDE EVERYTHING
  R2 raw     ridge on the raw window. The CEILING. Not a baseline you may lose
             to -- if your embedding is below it, the encoder deleted signal.
  R2 random  the SAME architecture with max_epochs=0, frozen at init. This is
             what your weights are worth before any learning. If a trained arm
             is below it, training made things worse, and pretext accuracy is
             irrelevant at that point.
  pair %     held-out order accuracy. The decode is now an ASSIGNMENT (a real
             permutation), so chance is exactly 50% and a degenerate head scores
             50%, not 0%. Reported next to the sort-by-mean shortcut baseline.
  p.ratio    participation ratio of the embedding. In the 2026-09-22 run it
             tracked ridge R2 almost monotonically (order_only: 3.4/64 -> 0.005).
             A collapsing p.ratio is the failure, and no pretext number fixes it.

Every arm also prints delta_random = R2 - R2(random_encoder), which is the only
honest measure of what training contributed.

EPOCHS ARE A HYPERPARAMETER, NOT A BUDGET. Spans overlap by construction
(stride 1), so the effective sample size is far below the span count and a big
epoch count memorizes the pretext task: the 6000-epoch run reached a held-out
position cross-entropy of 5.7-36.8 against a uniform 1.386, i.e. confidently
WRONG out of sample. Run `--arms epochs_10 epochs_30 ... ` (or just
`--epoch-sweep`) before believing any single-epoch-count table.
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
SEED = 42
TRAIN_FRACTION = 0.8

WINDOW_SIZE = 10
N_TILES = 4
TILE_GAP = (1, 8)
OUTPUT_DIMENSION = 64
NUM_HIDDEN_UNITS = 64
HEAD_HIDDEN_UNITS = 64
BATCH_SIZE = 512
EPOCHS = 60
LEARNING_RATE = 1e-3
TILE_NORM = "mean"
NEURON_DROPOUT = 0.1
GAIN_JITTER = 0.1

DECODER_EPOCHS = 2500
DECODER_HIDDEN = 64
DECODER_DROPOUT = 0.4
DECODER_LR = 1e-3
PRETEXT_SPANS = 2048
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)

# --------------------------------------------------------------------------- #
# arms. No contrastive arm exists -- the model has no contrastive loss.
# --------------------------------------------------------------------------- #
ARMS = {
    # ---- the proposal ---------------------------------------------------- #
    "proposed": {},

    # ---- controls that can falsify it ----------------------------------- #
    # Frozen at init. THE reference point. Beat this or nothing else matters.
    "random_encoder": dict(max_epochs=0),
    # Order CE only: the classic jigsaw. Nothing keeps firing-rate level.
    # This is the arm that collapsed to p.ratio 3.4/64 and R2 0.005.
    "order_only": dict(lambda_forecast=0.0, lambda_reconstruct=0.0),
    # Forecast CE only: is the puzzle contributing anything at all?
    "forecast_only": dict(lambda_order=0.0, lambda_pair=0.0, lambda_reconstruct=0.0),
    # The anti-collapse anchor on its own: a pure denoising autoencoder in CE
    # form, no puzzle at all. If THIS is the best arm, the jigsaw is dead weight
    # and you should say so in the paper rather than hide it.
    "reconstruct_only": dict(lambda_order=0.0, lambda_pair=0.0, lambda_forecast=0.0),
    # The proposal MINUS the anchor == the model that produced the 2026-09-22
    # table. Its job is to reproduce the collapse, so the anchor's effect is
    # measured against the right thing and not against a random encoder.
    "no_reconstruct": dict(lambda_reconstruct=0.0),
    # Puzzle + anchor, no forecasting: is forecast doing anything the
    # reconstruction is not already doing? They are both "keep the rate".
    "order_reconstruct": dict(lambda_forecast=0.0),
    # Anchor turned up. If R2 keeps climbing with this, the bottleneck -- not
    # the pretext task -- is what the whole experiment is measuring.
    "reconstruct_heavy": dict(lambda_reconstruct=4.0),
    # Order head trains on a trunk it cannot influence. Asks "is order
    # decodable from a forecast-trained trunk?" with zero risk to R2.
    "order_probe_only": dict(order_grad_scale=0.0),
    # Every anti-shortcut guard off. If pretext accuracy JUMPS here, the task
    # was being solved by level drift, not by temporal structure.
    "shortcut_open": dict(tile_norm="none", neuron_dropout=0.0, gain_jitter=0.0),

    # ---- epochs. The 6000-epoch run memorized the pretext task ----------- #
    # Held-out position CE was 5.7-36.8 against a uniform 1.386: confidently
    # wrong, which is overfitting, not underfitting. Spans overlap at stride 1,
    # so ~2400 spans are worth far fewer independent samples than that.
    "epochs_10": dict(max_epochs=10),
    "epochs_30": dict(max_epochs=30),
    "epochs_100": dict(max_epochs=100),
    "epochs_300": dict(max_epochs=300),
    "epochs_1000": dict(max_epochs=1000),

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
    # there is: chance is exactly 50%, so tiny effects are detectable.
    "tiles_2": dict(n_tiles=2),
    "tiles_3": dict(n_tiles=3),
    "tiles_6": dict(n_tiles=6),
    # Long-range order: are far-apart tiles easier (more drift) or harder?
    "wide_gaps": dict(tile_gap=(8, 40)),
    "tight_gaps": dict(tile_gap=(1, 2)),
    # Coarser / finer forecast targets.
    "forecast_4": dict(forecast_levels=4),
    "forecast_16": dict(forecast_levels=16),
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
    # MobileNet's final 1x1 before the classifier.
    "mobile_head_256": dict(_model="mobile", head_channels=256),
    # The control that makes the whole comparison readable.
    "mobile_random": dict(_model="mobile", max_epochs=0),
}

DEFAULT_ARMS = ("proposed", "no_reconstruct", "random_encoder", "order_only",
                "forecast_only", "reconstruct_only", "order_reconstruct",
                "shortcut_open", "dim_256", "tiles_2",
                "mobile_v1", "mobile_v2", "mobile_v3", "mobile_alpha_035",
                "mobile_stem_mix", "mobile_random")

# The epoch sweep. Run this FIRST on a new session: every other comparison is
# conditional on being in a sane epoch regime, and the previous table was not.
EPOCH_ARMS = ("epochs_10", "epochs_30", "epochs_100", "epochs_300",
              "epochs_1000", "random_encoder")

# Arms grouped by trunk, so the table can be read per-architecture.
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


def load_session(path):
    with np.load(path, allow_pickle=True) as handle:
        keys = set(handle.files)
        spikes_key = next((k for k in ("spikes", "neural", "rates", "X", "counts")
                           if k in keys), None)
        behavior_key = next((k for k in ("behavior", "velocity", "vel", "Y", "y", "kin")
                            if k in keys), None)
        if spikes_key is None or behavior_key is None:
            raise KeyError(f"{path.name}: need a spike key and a behavior key; has {sorted(keys)}")
        spikes = np.asarray(handle[spikes_key], dtype=np.float32)
        behavior = np.asarray(handle[behavior_key], dtype=np.float32)
    if spikes.ndim != 2:
        raise ValueError(f"{path.name}: spikes must be 2D, got {spikes.shape}")
    if behavior.ndim == 1:
        behavior = behavior[:, None]
    if spikes.shape[0] != behavior.shape[0] and spikes.shape[1] == behavior.shape[0]:
        spikes = spikes.T
    length = min(len(spikes), len(behavior))
    spikes, behavior = spikes[:length], behavior[:length]
    if behavior.shape[1] > 2:
        behavior = behavior[:, :2]
    keep = np.isfinite(spikes).all(1) & np.isfinite(behavior).all(1)
    if not keep.all():
        first, last = int(np.argmax(keep)), length - int(np.argmax(keep[::-1]))
        spikes, behavior = spikes[first:last], behavior[first:last]
        if not (np.isfinite(spikes).all() and np.isfinite(behavior).all()):
            raise ValueError(f"{path.name}: nonfinite values in the interior, not just the edges")
    alive = spikes.std(0) > 0
    return np.ascontiguousarray(spikes[:, alive]), np.ascontiguousarray(behavior)


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
                        window_size, device):
    """The ceiling: ridge and an MLP on the flattened raw window."""
    x_train, offset = sliding_windows(spikes_train, window_size)
    x_test, _ = sliding_windows(spikes_test, window_size)
    y_train = behavior_train[offset:offset + len(x_train)]
    y_test = behavior_test[offset:offset + len(x_test)]
    ridge = safe_ridge(x_train, y_train, x_test, y_test)
    mlp = mlp_r2(x_train, y_train, x_test, y_test, epochs=DECODER_EPOCHS,
                 hidden=DECODER_HIDDEN, dropout=DECODER_DROPOUT,
                 learning_rate=DECODER_LR, seed=SEED, device=device)
    return dict(ridge=ridge, mlp=mlp, n_features=int(x_train.shape[1]))


def build_model(override, epochs, device, verbose):
    """Dispatch on the '_model' key so both trunks appear in ONE table."""
    override = dict(override)
    factory = MODELS[override.pop("_model", "jigsaw")]
    settings = dict(window_size=WINDOW_SIZE, n_tiles=N_TILES, tile_gap=TILE_GAP,
                    output_dimension=OUTPUT_DIMENSION, num_hidden_units=NUM_HIDDEN_UNITS,
                    head_hidden_units=HEAD_HIDDEN_UNITS, tile_norm=TILE_NORM,
                    neuron_dropout=NEURON_DROPOUT, gain_jitter=GAIN_JITTER,
                    batch_size=BATCH_SIZE, max_epochs=epochs,
                    learning_rate=LEARNING_RATE, device=device, random_state=SEED,
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


def pretext_report(model, spikes_valid):
    metrics = model.evaluate_pretext(spikes_valid, max_spans=PRETEXT_SPANS, verbose=False)
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
    # "learned" requires beating chance AND both level-sorting shortcuts.
    shortcut = max(metrics["baseline_mean_sort_pair_percent"],
                   metrics["baseline_norm_sort_pair_percent"])
    metrics["shortcut_pair_percent"] = shortcut
    metrics["learned"] = bool(
        (exact["significant"] or pair["significant"])
        and metrics["pair_accuracy_percent"] > shortcut + 1.0)
    # Held-out position CE above log(K) means the head is confidently WRONG on
    # unseen spans -- memorization, which more epochs makes worse, not better.
    # The 2% margin matters: a frozen encoder sits AT uniform by construction,
    # and without a tolerance sampling noise flags every random control.
    metrics["memorizing"] = bool(
        metrics["position_cross_entropy"] > 1.02 * metrics["uniform_cross_entropy"])
    return metrics


def run_arm(name, override, data, device, epochs, verbose):
    spikes_train, behavior_train, spikes_valid, behavior_valid = data
    started = time.time()
    budget = int(override.get("max_epochs", epochs))
    model = build_model(override, epochs, device, verbose)
    model.fit(spikes_train, X_valid=spikes_valid,
              validate_every=max(1, budget // 6) if budget else 1)

    z_train, index_train = model.transform(spikes_train, pad=False, return_indices=True)
    z_valid, index_valid = model.transform(spikes_valid, pad=False, return_indices=True)
    y_train = behavior_train[index_train]
    y_valid = behavior_valid[index_valid]

    mlp = mlp_r2(z_train, y_train, z_valid, y_valid, epochs=DECODER_EPOCHS,
                 hidden=DECODER_HIDDEN, dropout=DECODER_DROPOUT,
                 learning_rate=DECODER_LR, seed=SEED, device=device)
    ridge = safe_ridge(z_train, y_train, z_valid, y_valid)
    pretext = pretext_report(model, spikes_valid)
    trunk = trunk_parameters(model)
    gap = (model.measure_train_eval_gap(spikes_valid)
           if isinstance(model, MobileJigsaw) else None)
    architecture = getattr(model, "version", None) or model.trunk_block
    del model, z_train, z_valid
    cleanup()
    return dict(arm=name, override=override, architecture=architecture,
                epochs=budget, trunk_parameters=trunk, train_eval_gap=gap,
                mlp=mlp, ridge=ridge, pretext=pretext, seconds=time.time() - started)


def print_table(results, baseline):
    rows = [r for r in results.values() if "error" not in r]
    if not rows:
        print("no arm completed")
        return

    def is_mobile(row):
        return row["override"].get("_model") == "mobile"

    # Each family is compared against ITS OWN frozen control. Comparing a
    # MobileNet arm to a plain-trunk random encoder would confound the
    # architecture with the initialization scale, which is not the question.
    controls = {False: results.get("random_encoder", {}).get("mlp", {}).get("r2"),
                True: results.get("mobile_random", {}).get("mlp", {}).get("r2")}
    header = (f"{'arm':<20}{'arch':>10}{'ep':>6}{'R2 mlp':>9}{'R2 ridge':>10}"
              f"{'d.random':>10}{'p.ratio':>9}{'pair%':>8}{'shortcut':>10}"
              f"{'exact%':>8}{'chance':>8}{'z':>7}{'ties%':>7}{'posCE':>8}"
              f"{'learned':>9}{'trunk par':>11}{'tr/ev':>8}")
    print("\n" + "=" * len(header))
    print("SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: -r["mlp"]["r2"]):
        p = row["pretext"]
        reference = controls[is_mobile(row)]
        delta = f"{row['mlp']['r2'] - reference:+.4f}" if reference is not None else "n/a"
        gap = row.get("train_eval_gap")
        gap_text = f"{gap['relative_gap']:.4f}" if gap else "-"
        flag = "*" if p.get("memorizing") else " "
        print(f"{row['arm']:<20}{str(row.get('architecture', '?')):>10}"
              f"{row.get('epochs', 0):>6}"
              f"{row['mlp']['r2']:>9.4f}{row['ridge']['r2']:>10.4f}{delta:>10}"
              f"{row['ridge'].get('participation_ratio', float('nan')):>9.2f}"
              f"{p['pair_accuracy_percent']:>8.2f}"
              f"{p.get('shortcut_pair_percent', p['baseline_mean_sort_pair_percent']):>10.2f}"
              f"{p['exact_accuracy_percent']:>8.2f}{p['chance_exact_percent']:>8.2f}"
              f"{p['exact_test']['z']:>7.2f}{p.get('argmax_tie_rate_percent', 0.0):>7.1f}"
              f"{p['position_cross_entropy']:>7.2f}{flag}{str(p['learned']):>9}"
              f"{row.get('trunk_parameters', 0):>11,}{gap_text:>8}")
    print("-" * len(header))
    print(f"{'RAW WINDOW':<20}{'-':>10}{'-':>6}{baseline['mlp']['r2']:>9.4f}"
          f"{baseline['ridge']['r2']:>10.4f}{'-':>10}"
          f"{baseline['ridge'].get('participation_ratio', float('nan')):>9.2f}"
          f"   <- CEILING ({baseline['n_features']} features, no encoder at all)")
    print("=" * len(header))

    best = max(rows, key=lambda r: r["mlp"]["r2"])
    ceiling = max(baseline["mlp"]["r2"], baseline["ridge"]["r2"])
    uniform = rows[0]["pretext"]["uniform_cross_entropy"]
    print("\nHOW TO READ THIS")
    print("  d.random  = R2 minus the frozen control OF THE SAME FAMILY")
    print("              (random_encoder for the plain trunk, mobile_random for")
    print("              MobileNet). NEGATIVE means training destroyed information,")
    print("              and pretext accuracy cannot rescue a negative number.")
    print("  p.ratio   = participation ratio of the embedding (out of the output")
    print("              dimension). This is the collapse detector: on 2026-09-22 it")
    print("              tracked ridge R2 almost monotonically, from 3.4 -> R2 0.005")
    print("              (order_only) up to 61.8 -> 0.784 (the raw window itself).")
    print("              If p.ratio falls, R2 falls, whatever the pretext says.")
    print("  pair %    = order accuracy under an ASSIGNMENT decode, so it is a real")
    print("              permutation and chance is exactly 50%. The older argmax")
    print("              decode was not a permutation, its ties were scored as")
    print("              errors, and chance was silently 37.5% (or 0% for a")
    print("              degenerate head). argmax is still reported in results.json")
    print("              as argmax_pair_accuracy_percent for comparison.")
    print("  shortcut  = the BEST of sort-by-mean and sort-by-norm pair accuracy.")
    print("              The model must beat this, not just beat 50%.")
    print(f"  posCE     = held-out position cross-entropy; uniform = {uniform:.3f}.")
    print("              A '*' marks posCE ABOVE uniform, i.e. the head is")
    print("              confidently WRONG out of sample. That is memorization and")
    print("              more epochs make it worse. Run the epoch sweep.")
    print("  trunk par = trunk parameters. Check this before calling MobileNet")
    print("              'efficient': in 1D, v2 at t=6 is LARGER than a plain conv.")
    print("  tr/ev     = relative embedding change between train and eval mode.")
    print("              Should be ~0. Large values mean BatchNorm running stats")
    print("              are wrong for the windows transform actually sees.")
    print(f"\nbest arm: {best['arm']}  R2={best['mlp']['r2']:.4f}  ({best.get('architecture')})")
    for mobile, label in ((False, "plain trunk"), (True, "mobilenet")):
        family = [r for r in rows if is_mobile(r) == mobile]
        reference = controls[mobile]
        if not family or reference is None:
            continue
        top = max(family, key=lambda r: r["mlp"]["r2"])
        verdict = ("training ADDS value" if top["mlp"]["r2"] > reference
                   else "!! every trained arm is at or below its RANDOM control")
        print(f"  {label:<12} best={top['arm']} {top['mlp']['r2']:.4f} "
              f"vs frozen {reference:.4f}: {verdict}")
    trained = [r for r in rows if r["override"].get("max_epochs") != 0]
    plain = [r for r in trained if not is_mobile(r)]
    mobile = [r for r in trained if is_mobile(r)]
    if plain and mobile:
        a = max(r["mlp"]["r2"] for r in plain)
        b = max(r["mlp"]["r2"] for r in mobile)
        print(f"  HEAD TO HEAD  plain {a:.4f}  vs  mobilenet {b:.4f}  -> "
              f"{'mobilenet' if b > a else 'plain trunk'} wins by {abs(b - a):.4f}")
    gap = best["mlp"]["r2"] - ceiling
    print(f"  vs raw-window ceiling ({ceiling:.4f}): {gap:+.4f}"
          + ("" if gap >= 0 else "  <- the encoder is a lossy bottleneck; try a larger"
                                 " output_dimension (dim_128 / dim_256) first,"
                                 " the trunk is not the binding constraint"))
    learners = [r["arm"] for r in rows if r["pretext"]["learned"]]
    print(f"  arms that actually learned the pretext: {learners or 'NONE'}")
    if learners and gap < 0:
        print("  note: pretext learned but R2 still under the ceiling -> the order task")
        print("        is solvable yet its solution is not what the decoder needs.")

    # The anchor A/B. This is the only pair that isolates the reconstruction CE.
    anchored, bare = results.get("proposed"), results.get("no_reconstruct")
    if anchored and bare and "error" not in anchored and "error" not in bare:
        d_r2 = anchored["mlp"]["r2"] - bare["mlp"]["r2"]
        d_pr = (anchored["ridge"].get("participation_ratio", float("nan"))
                - bare["ridge"].get("participation_ratio", float("nan")))
        print(f"\nRECONSTRUCTION ANCHOR  proposed - no_reconstruct: "
              f"R2 {d_r2:+.4f}, p.ratio {d_pr:+.2f}")
        print("  no_reconstruct IS the model that produced the 2026-09-22 table, so")
        print("  this difference -- not the distance to random_encoder -- is what the")
        print("  anchor is worth. If it is ~0, drop the anchor and report that.")

    memorizers = [r["arm"] for r in rows if r["pretext"].get("memorizing")]
    if memorizers:
        print(f"\n!! held-out position CE more than 2% ABOVE uniform for: {memorizers}")
        print("   These arms are confidently wrong on unseen spans. That is")
        print("   overfitting, not underfitting -- cut epochs, do not add them.")

    # Epoch sweep. Only arms that differ from `proposed` in max_epochs ALONE
    # belong on this curve: mixing objectives in would make it a function of
    # two variables and unreadable as a budget curve.
    ladder = [r for r in rows
              if set(r["override"]) <= {"max_epochs"} and r["override"].get("max_epochs") != 0]
    if len({r.get("epochs", 0) for r in ladder}) > 1:
        print("\nEPOCH SWEEP (same objective, budget is the only difference)")
        print(f"  {'epochs':>8}{'R2 mlp':>9}{'p.ratio':>9}{'pair%':>8}{'posCE':>8}  arm")
        for row in sorted(ladder, key=lambda r: r.get("epochs", 0)):
            p = row["pretext"]
            print(f"  {row.get('epochs', 0):>8}{row['mlp']['r2']:>9.4f}"
                  f"{row['ridge'].get('participation_ratio', float('nan')):>9.2f}"
                  f"{p['pair_accuracy_percent']:>8.2f}"
                  f"{p['position_cross_entropy']:>8.2f}  {row['arm']}")
        peak = max(ladder, key=lambda r: r["mlp"]["r2"])
        print(f"  peak: {peak.get('epochs')} epochs, R2 {peak['mlp']['r2']:.4f}. "
              f"Rerun every other arm at that budget.")
        print("  A comparison made in the wrong epoch regime is not a comparison")
        print("  between objectives, it is a comparison between overfits.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--epoch-sweep", action="store_true",
                        help="run the epoch ladder instead of --arms. Do this FIRST: "
                             "every objective comparison is conditional on the budget.")
    parser.add_argument("--list", action="store_true", help="print the arm table and exit")
    parser.add_argument("--data-dir", type=Path, default=PERICH_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--tag", default="JIGSAWNET")
    parser.add_argument("--quiet", action="store_true")
    options = parser.parse_args()

    if options.list:
        width = max(len(k) for k in ARMS)
        for name, override in ARMS.items():
            mark = "*" if name in DEFAULT_ARMS else ("e" if name in EPOCH_ARMS else " ")
            print(f" {mark} {name:<{width}}  {override or '(defaults)'}")
        print("\n* = in the default set,  e = in --epoch-sweep")
        return

    if options.epoch_sweep:
        options.arms = list(EPOCH_ARMS)
        print(f"epoch sweep: {options.arms}")

    unknown = [a for a in options.arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}\nvalid: {sorted(ARMS)}")
    # Each trunk family needs its OWN frozen control, otherwise d.random
    # confounds the architecture with the initialization scale.
    selected = list(options.arms)
    needed = {"random_encoder"}
    if any(ARMS[a].get("_model") == "mobile" for a in selected):
        needed.add("mobile_random")
    for control in sorted(needed - set(selected)):
        selected.append(control)
        print(f"note: added the {control} control -- results are unreadable without it.")
    options.arms = selected

    seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(p for p in options.data_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz files under {options.data_dir}")
    files = files[:max(1, options.sessions)]
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for path in files:
        spikes, behavior = load_session(path)
        cut = int(TRAIN_FRACTION * len(spikes))
        data = (spikes[:cut], behavior[:cut], spikes[cut:], behavior[cut:])
        out_dir = options.out_dir / f"{options.tag}_{path.stem}_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "#" * 78)
        print(f"# {path.stem}: {len(spikes)} bins, {spikes.shape[1]} neurons, "
              f"{behavior.shape[1]} behavior dims | train {cut} / valid {len(spikes) - cut}")
        print(f"# device={device} epochs={options.epochs} -> {out_dir}")
        print("#" * 78, flush=True)

        print("\n[ceiling] ridge + MLP on the raw window", flush=True)
        baseline = raw_window_baseline(*data, WINDOW_SIZE, device)
        print(f"  raw window: ridge R2={baseline['ridge']['r2']:.4f}  "
              f"MLP R2={baseline['mlp']['r2']:.4f}  ({baseline['n_features']} features)",
              flush=True)

        results = {}
        for i, name in enumerate(options.arms, 1):
            print(f"\n[{i}/{len(options.arms)}] {name}  {ARMS[name] or '(defaults)'}"
                  f"  [{arm_epochs(name, options.epochs)} epochs]", flush=True)
            try:
                results[name] = run_arm(name, ARMS[name], data, device,
                                        options.epochs, not options.quiet)
                row = results[name]
                print(f"  -> R2 mlp={row['mlp']['r2']:.4f} ridge={row['ridge']['r2']:.4f} "
                      f"p.ratio={row['ridge'].get('participation_ratio', float('nan')):.2f} "
                      f"| pair={row['pretext']['pair_accuracy_percent']:.2f}% "
                      f"(shortcut {row['pretext']['shortcut_pair_percent']:.2f}%, "
                      f"posCE {row['pretext']['position_cross_entropy']:.2f} vs "
                      f"{row['pretext']['uniform_cross_entropy']:.2f} uniform) "
                      f"| {row['seconds']:.0f}s", flush=True)
            except Exception as error:  # one bad arm must not kill the sweep
                traceback.print_exc()
                results[name] = dict(arm=name, error=f"{type(error).__name__}: {error}")
                cleanup()
            save_json(out_dir / "results.json",
                      dict(session=path.stem, stamp=stamp, device=device,
                           epochs=options.epochs, baseline=baseline, arms=results))

        print_table(results, baseline)
        print(f"\nwritten to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()



# """Sweep NeuralJigsaw's tile_normalize modes on one session and compare them
# on BOTH puzzle-solving accuracy and downstream behavioral R^2 -- the two
# numbers that turned out to disagree for the earlier architectures, so this
# script always reports both, side by side, for every mode, rather than
# picking a "winner" from puzzle accuracy alone.

# Mirrors the data loading / decoder / R2 conventions of the earlier
# window-mode comparison script (train_data/valid_data/train_label/valid_label
# in one .npz, optional test_data for puzzle evaluation). The only thing that
# changes between runs is NeuralJigsaw's tile_normalize; every other
# hyperparameter is held fixed so the comparison is apples-to-apples.

# The encoder checkpoint for each mode is selected by a CHEAP ridge probe on
# held-out validation embeddings (monitor_fn), evaluated periodically during
# fit -- i.e. by the metric you actually care about, not by puzzle loss.

# Usage:
#     python sweep_tile_normalize.py                      # off, center, zscore
#     python sweep_tile_normalize.py --modes center zscore
#     python sweep_tile_normalize.py --eval-split test
#     python sweep_tile_normalize.py --quick               # fast smoke test
# """
# from datetime import datetime, timezone
# from pathlib import Path
# import argparse
# import csv
# import gc
# import json
# import random
# import time

# import numpy as np
# import torch
# from torch import nn
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Ridge
# from sklearn.metrics import r2_score

# from Neural_Jigsaw import NeuralJigsaw


# ROOT = Path(__file__).resolve().parent
# SESSION = "C-CO16"
# PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
# NPZ_PATH = PERICH_DATA_DIR / f"{SESSION}.npz"
# OUT_ROOT = ROOT / f"NEURAL_JIGSAW_{SESSION}_SWEEP"
# SEED = 42

# MODES = ("off", "center", "zscore")  # every tile_normalize value NeuralJigsaw offers

# # --- encoder hyperparameters: identical across modes, so tile_normalize is ---
# # --- the only thing that differs between runs.                            ---
# WINDOW_SIZE = 8
# TILE_GAP = (1, 4)
# LATENT_DIM = 64
# ENCODER_HIDDEN = 64
# HEAD_HIDDEN = 128
# PUZZLE_BATCH_SIZE = 256
# PUZZLE_EPOCHS = 1
# PUZZLE_LR = 3e-4
# PUZZLE_WEIGHT_DECAY = 1e-4
# PUZZLE_DROPOUT = 0.1
# AUGMENT_GAIN_JITTER = 0.1
# AUGMENT_NOISE_STD = 0.05
# PERMUTATIONS_PER_WINDOW = 1
# DEVICE = "cuda"

# # --- monitor_fn: closed-form ridge probe on held-out validation embeddings, ---
# # --- called periodically during fit() so the CHECKPOINT is picked by       ---
# # --- downstream R2, not by puzzle CE/accuracy (see the 10-vs-20k paradox). ---
# MONITOR_EVERY = max(1, PUZZLE_EPOCHS // 40)   # ~40 probes over the whole run
# MONITOR_MAX_SAMPLES = 4000                     # subsample -- speed only
# MONITOR_RIDGE_ALPHA = 1.0

# # --- transform() alignment ---
# PAD_TRANSFORM = False
# TRANSFORM_BATCH_SIZE = 2048

# # --- final downstream decoder: same full-batch MLP as the earlier scripts ---
# DECODER_EPOCHS = 2500
# DECODER_HIDDEN = 64
# DECODER_DROPOUT = 0.4
# DECODER_LR = 1e-3
# PREDICT_BATCH_SIZE = 8192

# # --- fixed-weight puzzle evaluation (reported, but NOT used to pick a mode) ---
# PUZZLE_EVAL_SPLIT = "auto"       # prefer test_data; otherwise validation
# PUZZLE_EVAL_WINDOWS = 1024
# PUZZLE_EVAL_PERMUTATIONS = 24
# PUZZLE_EVAL_BATCH_SIZE = 256
# PUZZLE_EVAL_SEED = SEED + 200000
# # min_gap is set at runtime to each model's training_span (de-correlated windows)


# def training_span(window_size, tile_gap):
#     return 4 * window_size + 3 * tile_gap[1]


# def save_json(path, values):
#     path.write_text(json.dumps(values, indent=2, allow_nan=False), encoding="utf-8")


# def seed_all(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True


# def cleanup():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


# def load_data(path):
#     with np.load(path, allow_pickle=False) as data:
#         arrays = [np.asarray(data[key], dtype=np.float32) for key in
#                   ("train_data", "valid_data", "train_label", "valid_label")]
#     x_train, x_valid, y_train, y_valid = arrays
#     if y_train.ndim == 1:
#         y_train = y_train[:, None]
#     if y_valid.ndim == 1:
#         y_valid = y_valid[:, None]
#     arrays = (x_train, x_valid, y_train, y_valid)
#     for name, x in zip(("train_data", "valid_data", "train_label", "valid_label"), arrays):
#         if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
#             raise ValueError(f"{name} must be a finite nonempty 2D array; got {x.shape}.")
#     if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
#         raise ValueError("Feature/label time lengths do not match.")
#     if x_train.shape[1] != x_valid.shape[1] or y_train.shape[1] != y_valid.shape[1]:
#         raise ValueError("Train/validation channel or label dimensions do not match.")
#     span = training_span(WINDOW_SIZE, TILE_GAP)
#     if min(len(x_train), len(x_valid)) < span + 1:
#         raise ValueError(f"Each split needs at least training_span+1={span + 1} bins; "
#                          f"got train={len(x_train)}, valid={len(x_valid)}.")
#     return tuple(np.ascontiguousarray(a) for a in arrays)


# def load_puzzle_eval_data(path, split, min_length, n_features):
#     if split not in ("auto", "test", "validation"):
#         raise ValueError("eval split must be auto, test, or validation.")
#     with np.load(path, allow_pickle=False) as data:
#         if split == "auto":
#             split = "test" if "test_data" in data.files else "validation"
#         key = "test_data" if split == "test" else "valid_data"
#         if key not in data.files:
#             raise KeyError(f"Requested {split} puzzle evaluation, but {key!r} is missing from {path}.")
#         x = np.asarray(data[key], dtype=np.float32)
#     if (x.ndim != 2 or x.shape[0] < min_length or x.shape[1] != n_features
#             or not np.isfinite(x).all()):
#         raise ValueError(f"{key}: expected finite (T,{n_features}) with T>={min_length}; got {x.shape}.")
#     print(f"Puzzle evaluation source: {key} -> {split.upper()}", flush=True)
#     return np.ascontiguousarray(x), split, key


# def make_monitor(x_valid, y_valid, rng_seed):
#     """Cheap closed-form ridge probe: model.transform -> ridge.fit on half of
#     a held-out subsample -> R2 on the other half. Meant to be called every
#     few hundred epochs from inside fit(), NOT to replace the real decoder
#     used for the final reported R2 (that one is a proper trained MLP, below).
#     """
#     rng = np.random.default_rng(rng_seed)

#     def monitor(model):
#         z, indices = model.transform(x_valid, pad=PAD_TRANSFORM,
#                                       batch_size=TRANSFORM_BATCH_SIZE, return_indices=True)
#         y = y_valid[indices]
#         if len(z) > MONITOR_MAX_SAMPLES:
#             pick = rng.choice(len(z), size=MONITOR_MAX_SAMPLES, replace=False)
#             z, y = z[pick], y[pick]
#         half = len(z) // 2
#         if half < 2:
#             return float("-inf")
#         probe = Ridge(alpha=MONITOR_RIDGE_ALPHA)
#         probe.fit(z[:half], y[:half])
#         prediction = probe.predict(z[half:])
#         return float(r2_score(y[half:], prediction, multioutput="uniform_average", force_finite=True))

#     return monitor


# def run_puzzle_eval(model, x, split, key, out, min_gap):
#     metrics, details = model.evaluate_puzzle(
#         x, max_windows=PUZZLE_EVAL_WINDOWS, permutations_per_window=PUZZLE_EVAL_PERMUTATIONS,
#         batch_size=PUZZLE_EVAL_BATCH_SIZE, random_state=PUZZLE_EVAL_SEED, min_gap=min_gap,
#         return_details=True)
#     metrics = dict(metrics, split=split, data_key=key)
#     save_json(out / f"puzzle_accuracy_{split}.json", metrics)
#     np.savez_compressed(out / f"puzzle_predictions_{split}.npz", **details)
#     print(f"  PUZZLE {split.upper()} ACCURACY: {metrics['accuracy_percent']:.2f}% "
#           f"| CE={metrics['cross_entropy']:.5f} | chance={metrics['chance_accuracy_percent']:.2f}% "
#           f"| min_gap={min_gap}", flush=True)
#     return metrics


# class Decoder(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, DECODER_HIDDEN), nn.LayerNorm(DECODER_HIDDEN),
#             nn.ReLU(), nn.Dropout(DECODER_DROPOUT),
#             nn.Linear(DECODER_HIDDEN, output_dim),
#         )

#     def forward(self, z):
#         return self.net(z)


# def train_decoder(z_train, y_train, device, epochs):
#     seed_all(SEED + 100000)
#     model = Decoder(z_train.shape[1], y_train.shape[1]).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=DECODER_LR)
#     z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
#     y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
#     losses = []
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad(set_to_none=True)
#         loss = nn.functional.mse_loss(model(z), y)
#         if not torch.isfinite(loss).item():
#             raise FloatingPointError(f"Nonfinite decoder loss at epoch {epoch + 1}.")
#         loss.backward()
#         optimizer.step()
#         losses.append(float(loss.detach().item()))
#         if (epoch + 1) % 250 == 0 or epoch + 1 == epochs:
#             print(f"    decoder {epoch + 1}/{epochs}: train MSE={losses[-1]:.6f}", flush=True)
#     return model, losses


# def predict_decoder(model, z):
#     model.eval()
#     device = next(model.parameters()).device
#     predictions = []
#     with torch.inference_mode():
#         for start in range(0, len(z), PREDICT_BATCH_SIZE):
#             inputs = torch.as_tensor(z[start:start + PREDICT_BATCH_SIZE],
#                                       dtype=torch.float32, device=device)
#             predictions.append(model(inputs).cpu().numpy())
#     return np.concatenate(predictions, axis=0)


# def score_r2(y, predictions):
#     if y.shape != predictions.shape or len(y) < 2 or not np.isfinite(predictions).all():
#         raise ValueError("Invalid predictions for R2.")
#     values = np.atleast_1d(r2_score(y, predictions, multioutput="raw_values", force_finite=True))
#     constant = np.flatnonzero(np.all(y == y[0], axis=0)).tolist()
#     return dict(mean_r2=float(values.mean()), per_output_r2=values.tolist(),
#                 constant_target_columns=constant)


# def run_mode(mode, x_train, x_valid, y_train, y_valid, x_eval, eval_split, eval_key,
#              out_root, device, epochs, monitor_every, decoder_epochs):
#     seed_all(SEED)
#     stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
#     out = out_root / f"{mode}_{stamp}"
#     out.mkdir(parents=True, exist_ok=False)
#     print(f"\n===== tile_normalize={mode} -> {out} =====", flush=True)

#     model = NeuralJigsaw(
#         window_size=WINDOW_SIZE, tile_gap=TILE_GAP, output_dimension=LATENT_DIM,
#         num_hidden_units=ENCODER_HIDDEN, head_hidden_units=HEAD_HIDDEN,
#         batch_size=PUZZLE_BATCH_SIZE, max_epochs=epochs, learning_rate=PUZZLE_LR,
#         weight_decay=PUZZLE_WEIGHT_DECAY, dropout=PUZZLE_DROPOUT, normalize=True,
#         tile_normalize=mode, augment_gain_jitter=AUGMENT_GAIN_JITTER,
#         augment_noise_std=AUGMENT_NOISE_STD, permutations_per_window=PERMUTATIONS_PER_WINDOW,
#         device=device, random_state=SEED, verbose=True,
#     )
#     save_json(out / "run_config.json", dict(session=SESSION, mode=mode, encoder=model.get_params(),
#                                             monitor_every=monitor_every, decoder_epochs=decoder_epochs))

#     monitor = make_monitor(x_valid, y_valid, rng_seed=SEED + 500000)
#     started = time.perf_counter()
#     model.fit(x_train, monitor_fn=monitor, monitor_every=monitor_every, select_best=True)
#     model.save(out / "encoder.pt")
#     save_json(out / "history.json", model.history_)
#     save_json(out / "checkpoint_selection.json",
#               dict(best_epoch=model.best_epoch_, best_monitor_r2=model.best_score_,
#                    last_epoch=epochs, monitor_every=monitor_every))

#     min_gap = model.training_span
#     train_puzzle = run_puzzle_eval(model, x_train, "train", "train_data", out, min_gap)
#     eval_puzzle = run_puzzle_eval(model, x_eval, eval_split, eval_key, out, min_gap)

#     z_train, idx_train = model.transform(x_train, pad=PAD_TRANSFORM,
#                                          batch_size=TRANSFORM_BATCH_SIZE, return_indices=True)
#     z_valid, idx_valid = model.transform(x_valid, pad=PAD_TRANSFORM,
#                                          batch_size=TRANSFORM_BATCH_SIZE, return_indices=True)
#     for z, indices in ((z_train, idx_train), (z_valid, idx_valid)):
#         if z.shape != (len(indices), LATENT_DIM) or not np.isfinite(z).all():
#             raise ValueError("Invalid NeuralJigsaw embeddings.")
#     yt, yv = y_train[idx_train], y_valid[idx_valid]
#     np.savez_compressed(out / "embeddings.npz", Z_train=z_train, Z_valid=z_valid,
#                         Y_train=yt, Y_valid=yv, train_indices=idx_train, valid_indices=idx_valid)

#     decoder, losses = train_decoder(z_train, yt, device, decoder_epochs)
#     train_pred = predict_decoder(decoder, z_train)
#     valid_pred = predict_decoder(decoder, z_valid)
#     scores = dict(train=score_r2(yt, train_pred), validation=score_r2(yv, valid_pred))
#     save_json(out / "R2.json", scores)
#     np.save(out / "decoder_train_mse.npy", np.asarray(losses))
#     torch.save(dict(state_dict={k: v.detach().cpu() for k, v in decoder.state_dict().items()},
#                     input_dim=LATENT_DIM, output_dim=yt.shape[1], hidden=DECODER_HIDDEN,
#                     dropout=DECODER_DROPOUT), out / "decoder.pt")

#     elapsed = time.perf_counter() - started
#     save_json(out / "timing.json", dict(total_seconds=elapsed))
#     print(f"[{mode}] puzzle {eval_split}={eval_puzzle['accuracy_percent']:.2f}% "
#           f"(train={train_puzzle['accuracy_percent']:.2f}%, chance={eval_puzzle['chance_accuracy_percent']:.2f}%) "
#           f"| R2 train={scores['train']['mean_r2']:.4f} valid={scores['validation']['mean_r2']:.4f} "
#           f"| best_epoch={model.best_epoch_}/{epochs} ({elapsed:.0f}s)", flush=True)

#     return dict(mode=mode, out_dir=str(out), best_epoch=model.best_epoch_,
#                 best_monitor_r2=model.best_score_,
#                 puzzle_train_accuracy_percent=train_puzzle["accuracy_percent"],
#                 puzzle_train_cross_entropy=train_puzzle["cross_entropy"],
#                 puzzle_eval_split=eval_split,
#                 puzzle_eval_accuracy_percent=eval_puzzle["accuracy_percent"],
#                 puzzle_eval_cross_entropy=eval_puzzle["cross_entropy"],
#                 chance_accuracy_percent=eval_puzzle["chance_accuracy_percent"],
#                 r2_train=scores["train"]["mean_r2"], r2_valid=scores["validation"]["mean_r2"])


# def plot_comparison(summary, out_root):
#     order = [row["mode"] for row in summary]
#     by_mode = {row["mode"]: row for row in summary}
#     modes = [m for m in MODES if m in by_mode]  # canonical left-to-right order for the figure
#     x = np.arange(len(modes))
#     fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

#     axes[0].bar(x - 0.18, [by_mode[m]["puzzle_train_accuracy_percent"] for m in modes], width=0.36, label="train")
#     axes[0].bar(x + 0.18, [by_mode[m]["puzzle_eval_accuracy_percent"] for m in modes], width=0.36,
#                label=by_mode[modes[0]]["puzzle_eval_split"])
#     axes[0].axhline(by_mode[modes[0]]["chance_accuracy_percent"], color="black", linestyle="--",
#                     linewidth=1, label="chance")
#     axes[0].set_xticks(x, modes)
#     axes[0].set_ylabel("Puzzle accuracy (%)")
#     axes[0].set_title("Puzzle-solving accuracy")
#     axes[0].legend()

#     axes[1].bar(x - 0.18, [by_mode[m]["r2_train"] for m in modes], width=0.36, label="train")
#     axes[1].bar(x + 0.18, [by_mode[m]["r2_valid"] for m in modes], width=0.36, label="validation")
#     axes[1].axhline(0, color="black", linewidth=0.8)
#     axes[1].set_xticks(x, modes)
#     axes[1].set_ylabel("Decoder R2")
#     axes[1].set_title("Downstream behavior R2  <- what actually matters")
#     axes[1].legend()

#     fig.suptitle(f"{SESSION} | NeuralJigsaw tile_normalize comparison "
#                 f"(best valid R2: {order[0]})")
#     fig.savefig(out_root / "tile_normalize_comparison.png", dpi=250)
#     plt.close(fig)


# def main(modes, eval_split, epochs, monitor_every, decoder_epochs):
#     if epochs < 1 or LATENT_DIM < 1:
#         raise ValueError("Bad hyperparameters.")
#     x_train, x_valid, y_train, y_valid = load_data(NPZ_PATH)
#     span = training_span(WINDOW_SIZE, TILE_GAP)
#     x_eval, actual_split, eval_key = load_puzzle_eval_data(NPZ_PATH, eval_split, span, x_train.shape[1])
#     OUT_ROOT.mkdir(parents=True, exist_ok=True)
#     device_name = "cuda" if (DEVICE == "cuda_if_available" and torch.cuda.is_available()) else DEVICE
#     print(f"Data: {NPZ_PATH} | shapes: train={x_train.shape} valid={x_valid.shape} "
#           f"| device={device_name} | modes={list(modes)}", flush=True)

#     summary = []
#     for mode in modes:
#         row = run_mode(mode, x_train, x_valid, y_train, y_valid, x_eval, actual_split, eval_key,
#                        OUT_ROOT, DEVICE, epochs, monitor_every, decoder_epochs)
#         summary.append(row)
#         cleanup()

#     summary.sort(key=lambda row: row["r2_valid"], reverse=True)
#     save_json(OUT_ROOT / "sweep_summary.json", summary)
#     with (OUT_ROOT / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
#         writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
#         writer.writeheader()
#         writer.writerows(summary)
#     plot_comparison(summary, OUT_ROOT)

#     print("\n===== SUMMARY (best validation R2 first) =====", flush=True)
#     for row in summary:
#         print(f"{row['mode']:8s} | puzzle {row['puzzle_eval_split']}={row['puzzle_eval_accuracy_percent']:6.2f}% "
#               f"(chance {row['chance_accuracy_percent']:.2f}%) | R2 train={row['r2_train']:.4f} "
#               f"valid={row['r2_valid']:.4f} | best_epoch={row['best_epoch']}/{epochs}", flush=True)
#     print(f"\nBest by validation R2: {summary[0]['mode']}", flush=True)
#     print(f"(Best by puzzle accuracy alone might be a DIFFERENT mode -- that is exactly the "
#           f"discrepancy this script is meant to expose. Trust the R2 column.)", flush=True)
#     print(f"Saved: {OUT_ROOT}", flush=True)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description=__doc__,
#                                      formatter_class=argparse.RawDescriptionHelpFormatter)
#     parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
#     parser.add_argument("--eval-split", choices=("auto", "test", "validation"), default=PUZZLE_EVAL_SPLIT)
#     parser.add_argument("--quick", action="store_true",
#                         help="Fast smoke test: few encoder/decoder epochs, frequent monitoring. "
#                              "Run this once before committing to the full sweep.")
#     parser.add_argument("--epochs", type=int, default=None, help="Override PUZZLE_EPOCHS.")
#     options = parser.parse_args()
#     if options.quick:
#         run_epochs = options.epochs or 200
#         run_monitor_every = 10
#         run_decoder_epochs = 200
#     else:
#         run_epochs = options.epochs or PUZZLE_EPOCHS
#         run_monitor_every = max(1, run_epochs // 40)
#         run_decoder_epochs = DECODER_EPOCHS
#     main(options.modes, options.eval_split, run_epochs, run_monitor_every, run_decoder_epochs)

"""Retrospective tests A, B and C of a signal-referenced training budget (no training).

Background. The note's threat budget, ε* = min(1, ε_NN, ε_det), measures a perturbation
against the recording's noise; on the Perich sessions it is always 1 JND. A *training* budget
must in addition stay below the size of natural changes of the *signal*: a training perturbation
that can move a window as far apart as two natural signal states asks the encoder to ignore the
signal itself. For a training metric M = Σ_s ⊗ Σ_t (d′ = ‖M^{-1/2} δ‖) the typical d′ distance
between the signal parts of two windows is

    ε_sig(M)² = 2 Σ_{i,j} (Σ_t⁻¹)_{ij} g^sig_{|i−j|},   g_k = tr(Σ_s⁻¹ Γ_k),   Γ_k = Cov(x_{t+k}, x_t).

With temporally white noise every g_k (k ≥ 1) is signal. The signal part of g_0 is estimated by
g_1, because Cov(x) − Cov((x_t − x_{t−1})/√2) = sym Γ_1, which is exactly the part the
difference residual removes. For a white temporal factor this reduces to

    ε_sig(M)² = 2T · tr(Σ_s⁻¹ sym Γ_1).

In the noise metric (difference residuals) this is the note's ε_NN idea with the noise floor
removed: E[d′²(x_a − x_b)] − 2D over pairs of windows (computed here as a check). The
hypothesis is that ε_train / ε_sig(M_train), and not the nominal ε, decides whether adversarial
training is safe.

A  Does the training loss gain fall on one curve of ε_train / ε_sig(M_train) across sessions and
   geometries? Leave-one-session-out isotonic fits are compared with the nominal ε,
   ε / (2D)^{1/4} and the realised perturbation sizes.
B  Do over-budget failures (loss-gain saturation; decoders broken at the smallest evaluation
   budget) happen at large ε_train / ε_sig? The specific cases are AUTO in M-RT3 / J-CO0
   (time-only) and SP8 / REG8 in supervised CEBRA; all runs are also pooled.
C  Does the robustness the decoders reach (reference ε½) follow the session's signal separation
   in the evaluation metric?

The noise models are the fork's own (``cebra/dprime/noise.py`` of --fork-dir, imported on its
own), fitted on the same training split as in training.

Usage (CPU only, about a minute):
    python signal_budget_tests.py \\
        --data-dir /data/hossein/mm_project/perich_data_valid_final_raw \\
        --results-dir /path/to/EXP8_SUPERVISED_AND_TIMEONLY_COMPLETE \\
        --fork-dir ./Acorn-dprime --out-dir ./SIGNAL_BUDGET_TESTS
    python signal_budget_tests.py --fork-dir ./Acorn-dprime --self-test   # synthetic check

--results-dir is the folder that holds DPRIME_EXP5, DPRIME_EXP6 and
DPRIME_SUPERVISED_PERICH_EXP8. Outputs in --out-dir: session_signal.csv / .json (stage 1),
runs.csv, testA_collapse.csv, testA_comparisons.csv, testA_sessions.csv, testA_within_arm.csv,
testB_auto.csv, testB_supervised.csv, testB_supervised_summary.csv, testB_pooled.csv,
testC_margin.csv, fig_testA.png, fig_testB.png, fig_testC.png and SUMMARY.txt (start here).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "8")  # small matrices: many BLAS threads only contend

import numpy as np
import pandas as pd
from scipy import stats

SESSIONS = ("C-CO12", "C-CO13", "C-RT13", "C-RT6", "M-CO9", "M-RT3", "T-RT1", "T-RT4", "J-CO0")
SOURCES = (("DPRIME_EXP5", "time_only"), ("DPRIME_EXP6", "time_only"),
           ("DPRIME_SUPERVISED_PERICH_EXP8", "supervised"))
MODES = ("time_only", "supervised")
SHORT = {"CLEAN": "CLEAN", "PGD_L2_EPS3": "PGD3", "PGD_L2_EPS6": "PGD6", "PGD_L2_EPS12": "PGD12",
         "MADRY_ISO_D8": "ISO8", "MADRY_ISO_D16": "ISO16", "MADRY_SP_D4": "SP4", "MADRY_SP_D8": "SP8",
         "MADRY_SP_D16": "SP16", "MADRY_NA_D4": "NA4", "MADRY_SPDIFF_D8": "DIFF8",
         "MADRY_SP_D8_POOL": "POOL", "MADRY_SP_AUTOBLIND": "AUTO", "MADRY_SPREG_D8": "REG8"}
# d′ metric of each training geometry as fit_noise_model arguments (residuals, spatial, temporal).
METRICS = {
    "sp": ("whole", "ledoit_wolf", "white"),
    "diff": ("difference", "ledoit_wolf", "white"),
    "reg": ("regression", "ledoit_wolf", "white"),
    "iso": ("whole", "isotropic", "white"),
    "ref": ("whole", "ledoit_wolf", "welch"),  # the NA arms, and the evaluation's reference metric
}
# "raw" (PGD arms) is the identity metric in spike counts; "pool" mixes sp and iso.
CLIPPED_HALF = 0.5 + 1e-6  # ε½ at the smallest evaluation budget: broken already at d′ = 0.5
SATURATED_GAIN = 2.0  # training loss gain of the saturated regime (SP16: 2.25–2.29 in Exp5)
A_PREDICTORS = ("x_nominal", "x_blind", "x_ref_size", "x_raw_size")
PALETTE = {"sp": "#2a78d6", "diff": "#1baf7a", "reg": "#eb6834", "iso": "#898781", "ref": "#3d3c39",
           "raw": "#0b0b0b", "pool": "#eb6834"}
MARKERS = {"sp": "o", "diff": "^", "reg": "v", "iso": "s", "ref": "D", "raw": "x", "pool": "s"}
LABELS = {"sp": "SP: LW ⊗ I, whole", "diff": "DIFF: LW ⊗ I, difference", "reg": "REG: LW ⊗ I, regression",
          "iso": "ISO: σ̄²I ⊗ I", "ref": "NA: LW ⊗ Welch", "raw": "PGD: raw L2",
          "pool": "POOL (mixture, not fitted)"}


# ----------------------------------------------------------------------------- inputs
def load_noise_module(fork_dir: Path):
    """The fork's noise.py, imported on its own (it needs numpy, scipy, sklearn and torch only)."""
    path = fork_dir / "cebra" / "dprime" / "noise.py"
    if not path.is_file():
        raise FileNotFoundError(f"No noise.py at {path}; set --fork-dir to the Acorn-dprime fork.")
    spec = importlib.util.spec_from_file_location("acorn_dprime_noise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if "difference" not in module.RESIDUAL_MODES or "isotropic" not in module.SPATIAL_ESTIMATORS:
        raise RuntimeError(f"{path} lacks the difference residuals or the isotropic estimator.")
    return module


def load_session(data_dir: Path, session: str) -> Tuple[np.ndarray, np.ndarray, Path]:
    """Training split of a session, with the runners' checks (raw non-negative integer counts)."""
    path = data_dir / f"{session}.npz"
    with np.load(path, allow_pickle=False) as f:
        x = np.asarray(f["train_data"], dtype=np.float32)
        y = np.asarray(f["train_label"], dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{path}: invalid train_data / train_label {x.shape} {y.shape}")
    if np.any(x < 0) or not np.array_equal(x, np.rint(x)):
        raise ValueError(f"{path}: train_data must be raw non-negative integer spike counts")
    # The trainer sees the float32 tensor; the noise model works in float64.
    return x.astype(np.float64), y.astype(np.float64), path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def note_eps_nn(results_dir: Optional[Path], session: str) -> Optional[float]:
    """The note's ε_NN (10th-percentile whitened NN distance) of the Exp5 SP8 run, for reference."""
    if results_dir is None:
        return None
    for path in sorted((results_dir / "DPRIME_EXP5" / session).glob("seed*/reports/MADRY_SP_D8.json")):
        try:
            return (json.loads(path.read_text()).get("eps_nn") or {}).get("value")
        except (OSError, json.JSONDecodeError):
            continue
    return None


def training_sha256(results_dir: Optional[Path], session: str) -> Optional[str]:
    """SHA256 of the data file used for training, as recorded in run_config.json."""
    if results_dir is None:
        return None
    for source, _ in SOURCES:
        for config in sorted((results_dir / source / session).glob("seed*/run_config.json")):
            try:
                value = json.loads(config.read_text()).get("data_sha256")
            except (OSError, json.JSONDecodeError):
                continue
            if value:
                return value
    return None


# ----------------------------------------------------------------------------- stage 1
def inverse_sqrt(cov: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (cov + cov.T))
    eigenvalues = np.maximum(eigenvalues, 1e-12 * max(float(np.mean(eigenvalues)), 1e-300))
    return (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T


def lag_traces(w: np.ndarray, max_lag: int) -> np.ndarray:
    """``g_k = tr(Σ_s⁻¹ Γ_k)`` = mean_t ⟨w_t, w_{t+k}⟩ of the centred, spatially whitened recording."""
    n = len(w)
    return np.array([np.einsum("ij,ij->", w[:n - k], w[k:]) / (n - k) for k in range(max_lag + 1)])


def signal_separation(g: np.ndarray, temporal_cov: np.ndarray, lag: int) -> float:
    """ε_sig of the metric ``Σ_s ⊗ Σ_t`` from its lag traces ``g``.

    Noise is taken as temporally white, so ``g_k`` (k ≥ ``lag``) is all signal; below ``lag``
    the signal part is set to ``g_lag`` (a slow signal is flat at short lags). ``lag = 1`` is the
    difference-residual decomposition; larger lags guard against noise correlated over a bin.
    """
    window_len = len(temporal_cov)
    gsig = np.array(g[:window_len], dtype=np.float64)
    gsig[:lag] = g[lag]
    lags = np.abs(np.subtract.outer(np.arange(window_len), np.arange(window_len)))
    value = 2.0 * float(np.sum(np.linalg.inv(temporal_cov) * gsig[lags]))
    return math.sqrt(value) if value > 0 else float("nan")


def pair_separation(w: np.ndarray, window_len: int, g: np.ndarray, n_pairs: int,
                    rng: np.random.Generator) -> Tuple[float, float, float]:
    """Empirical check for a white temporal factor: mean d′² of random non-overlapping window
    pairs minus the noise floor ``2T (g_0 − g_1)``. Returns (ε_sig, RMS pair d′, noise floor)."""
    starts = len(w) - window_len + 1
    a = rng.integers(0, starts, size=4 * n_pairs)
    b = rng.integers(0, starts, size=4 * n_pairs)
    keep = np.abs(a - b) >= window_len
    a, b = a[keep][:n_pairs], b[keep][:n_pairs]
    offsets = np.arange(window_len)
    total = 0.0
    for i in range(0, len(a), 256):
        diff = w[a[i:i + 256, None] + offsets] - w[b[i:i + 256, None] + offsets]
        total += float(np.sum(diff**2))
    mean_d2 = total / len(a)
    floor = 2.0 * window_len * (g[0] - g[1])
    excess = mean_d2 - floor
    return (math.sqrt(excess) if excess > 0 else float("nan"), math.sqrt(mean_d2),
            math.sqrt(max(floor, 0.0)))


def session_statistics(noise, x: np.ndarray, y: Optional[np.ndarray], window_len: int,
                       lags: Sequence[int], n_pairs: int, seed: int) -> Tuple[Dict[str, Any],
                                                                              Dict[str, Any]]:
    """ε_sig of every training metric and descriptors of the signal/noise structure."""
    n_time, num_channels = x.shape
    xc = x - x.mean(axis=0, keepdims=True)
    max_lag = max(window_len - 1, max(lags))
    rng = np.random.default_rng(seed)
    dim = num_channels * window_len
    row: Dict[str, Any] = {"n_ch": num_channels, "n_time": n_time, "window_len": window_len,
                           "D": dim, "sqrt_2D": math.sqrt(2 * dim),
                           "eps_blind_theory": (2.0 * dim)**0.25}
    covs = {}
    for name, (residuals, estimator, temporal) in METRICS.items():
        labels = y if residuals == "regression" else None
        if residuals == "regression" and y is None:
            continue
        model, _ = noise.fit_noise_model(x, labels,
                                         label_kind="continuous" if labels is not None else None,
                                         window_len=window_len, residual_mode=residuals,
                                         spatial_estimator=estimator, temporal_method=temporal,
                                         kronecker_channels=0)
        covs[name] = (model.spatial_cov, model.temporal_cov)
    covs["raw"] = (np.eye(num_channels), np.eye(window_len))
    for name, (spatial, temporal) in covs.items():
        w = xc @ inverse_sqrt(spatial)
        g = lag_traces(w, max_lag)
        row[f"g0_{name}"], row[f"g1_{name}"] = float(g[0]), float(g[1])
        for lag in lags:
            row[f"eps_sig_{name}_lag{lag}"] = signal_separation(g, temporal, lag)
        if np.allclose(temporal, np.eye(window_len)):
            pair, rms, floor = pair_separation(w, window_len, g, n_pairs, rng)
            row[f"eps_sig_{name}_pairs"] = pair
            if name == "diff":
                row["pair_rms_dprime_diff"], row["noise_floor_diff"] = rms, floor

    whole, difference = covs["sp"][0], covs["diff"][0]
    row["sigma_bar"] = math.sqrt(float(covs["iso"][0][0, 0]))
    row["noise_sd_median"] = float(np.median(np.sqrt(np.diag(whole))))
    row["noise_sd_median_diff"] = float(np.median(np.sqrt(np.diag(difference))))
    row["kappa_whole"] = float(np.linalg.cond(whole))
    row["kappa_diff"] = float(np.linalg.cond(difference))
    # Signal SNR per mode: generalised eigenvalues of (sym Γ_1, Σ_diff), per bin.
    gamma1 = xc[1:].T @ xc[:-1] / (n_time - 1)
    root = inverse_sqrt(difference)
    psi = np.linalg.eigvalsh(root @ (0.5 * (gamma1 + gamma1.T)) @ root)[::-1]
    positive = psi[psi > 0]
    row["psi_sum"] = float(psi.sum())
    row["psi_top1"], row["psi_top2"], row["psi_top3"] = (float(v) for v in psi[:3])
    row["n_modes_window_detectable"] = int(np.sum(psi * window_len > 1.0))  # T·ψ > 1
    row["psi_participation"] = float(positive.sum()**2 / max(np.sum(positive**2), 1e-300))
    row["signal_fraction_whole"] = row["g1_sp"] / row["g0_sp"]
    return row, {"psi": [float(v) for v in psi]}


def stage1(args, noise) -> pd.DataFrame:
    rows, spectra = [], {}
    for session in args.sessions:
        path = args.data_dir / f"{session}.npz"
        if not path.is_file():
            warnings.warn(f"{session}: no data file at {path}; skipped.")
            continue
        x, y, path = load_session(args.data_dir, session)
        expected = training_sha256(args.results_dir, session)
        actual = file_sha256(path)
        if expected is not None and expected != actual:
            warnings.warn(f"{session}: {path} differs from the data used for training "
                          f"(sha256 {actual[:12]} vs {expected[:12]}).")
        row, extra = session_statistics(noise, x, y, args.window_len, args.lags, args.pairs,
                                        args.seed)
        row["session"] = session
        row["data_sha_matches_training"] = None if expected is None else expected == actual
        row["eps_nn_note_sp"] = note_eps_nn(args.results_dir, session)
        rows.append(row)
        spectra[session] = extra
        print(f"{session}: C={row['n_ch']} n={row['n_time']}  ε_sig sp {row['eps_sig_sp_lag1']:.2f}"
              f"  diff {row['eps_sig_diff_lag1']:.2f}  iso {row['eps_sig_iso_lag1']:.2f}"
              f"  ref {row['eps_sig_ref_lag1']:.2f}  raw {row['eps_sig_raw_lag1']:.2f}"
              f"  (2D)^¼ {row['eps_blind_theory']:.2f}", flush=True)
    table = pd.DataFrame(rows).set_index("session")
    table.to_csv(args.out_dir / "session_signal.csv")
    (args.out_dir / "session_signal.json").write_text(json.dumps(spectra, indent=1))
    return table


# ----------------------------------------------------------------------------- stage 2
def training_info(report_path: Path, arm: str) -> Dict[str, Any]:
    """Training metric, budget and realised sizes of one run."""
    if arm == "CLEAN":
        return {"metric": None, "eps_train": 0.0}
    if arm.startswith("PGD_L2_EPS"):
        eps = float(arm[len("PGD_L2_EPS"):])
        return {"metric": "raw", "eps_train": eps, "raw_realized": eps}  # the L2 projection is active
    if not report_path.is_file():
        warnings.warn(f"No training report {report_path}; run skipped in the budget tests.")
        return {"metric": None, "eps_train": float("nan")}
    report = json.loads(report_path.read_text())
    settings = report["settings"]
    key = (settings.get("adv_dprime_residuals"), settings.get("adv_dprime_spatial"),
           settings.get("adv_dprime_temporal"))
    metric = {value: name for name, value in METRICS.items()}.get(key, "other")
    pool = settings.get("adv_dprime_cov_pool")
    if pool:
        metric = "pool" if metric == "sp" and list(pool) == ["isotropic"] else "other"
    training = report.get("training") or {}
    realized = training.get("realized") or {}

    def median(name):
        return (realized.get(name) or {}).get("median")

    eps = training.get("final_epsilon")
    if eps is None:
        eps = (report.get("epsilon") or {}).get("eps_star")
    return {"metric": metric, "eps_train": float(eps), "d_train_realized": median("dprime_training"),
            "ref_realized": median("dprime_reference"), "raw_realized": median("raw_l2"),
            "loss_gain": median("loss_gain"), "eps_source": (report.get("epsilon") or {}).get("source")}


def parse_runs(results_dir: Path, sessions: Sequence[str]) -> pd.DataFrame:
    rows = []
    for source, mode in SOURCES:
        for session in sessions:
            for seed_dir in sorted((results_dir / source / session).glob("seed*")):
                for path in sorted((seed_dir / "robust_eval").glob("*.json")):
                    evaluation = json.loads(path.read_text())
                    arm = path.stem
                    row = {"mode": mode, "source": source, "session": session,
                           "seed": int(seed_dir.name[4:]), "arm": arm, "short": SHORT.get(arm, arm),
                           "clean_r2": evaluation["clean_r2"],
                           "ref_half": evaluation["reference"]["eps_at_half_r2"],
                           "iso_half": evaluation["isotropic"]["eps_at_half_r2"]}
                    row.update(training_info(seed_dir / "reports" / f"{arm}.json", arm))
                    rows.append(row)
    runs = pd.DataFrame(rows)
    if runs.empty:
        raise FileNotFoundError(f"No robust_eval results under {results_dir}.")
    duplicated = runs.duplicated(["mode", "session", "seed", "arm"], keep=False)
    if duplicated.any():
        raise ValueError("Duplicate runs:\n" + runs[duplicated].to_string())
    for column in ("loss_gain", "ref_realized", "raw_realized", "d_train_realized"):
        if column not in runs:
            runs[column] = np.nan
    return runs


def add_predictors(runs: pd.DataFrame, sig: pd.DataFrame, lags: Sequence[int]) -> pd.DataFrame:
    """ε_train / ε_sig(M_train) and the baselines, plus PGD6-paired outcomes and failure flags."""
    runs = runs[runs.session.isin(sig.index)].copy()
    s = sig.loc[runs.session]
    blind = s.eps_blind_theory.values
    nominal = np.where(runs.metric == "raw", runs.eps_train / s.sigma_bar.values, runs.eps_train)
    runs["x_nominal"] = np.where(runs.metric.isna(), np.nan, nominal)
    runs["x_blind"] = runs.x_nominal / blind
    runs["x_ref_size"] = runs.ref_realized.astype(float)
    runs["x_raw_size"] = runs.raw_realized.astype(float)
    for suffix in [f"lag{lag}" for lag in lags] + ["pairs"]:
        x = np.full(len(runs), np.nan)
        for metric in list(METRICS) + ["raw"]:
            column = f"eps_sig_{metric}_{suffix}"
            if column in sig:
                rows = (runs.metric == metric).values
                x[rows] = runs.eps_train.values[rows] / s[column].values[rows]
        pool = (runs.metric == "pool").values
        if f"eps_sig_sp_{suffix}" in sig and f"eps_sig_iso_{suffix}" in sig:
            # Each step uses one of the two metrics; the geometric mean of the two ratios.
            x[pool] = runs.eps_train.values[pool] / np.sqrt(
                s[f"eps_sig_sp_{suffix}"].values[pool] * s[f"eps_sig_iso_{suffix}"].values[pool])
        runs[f"x_signal_{suffix}"] = x
    base = runs[runs.short == "PGD6"].set_index(["mode", "session", "seed"])
    key = pd.MultiIndex.from_frame(runs[["mode", "session", "seed"]])
    runs["d_clean_vs_pgd6"] = runs.clean_r2.values - base.clean_r2.reindex(key).values
    runs["ref_ratio_vs_pgd6"] = runs.ref_half.values / base.ref_half.reindex(key).values
    trained = runs.short != "CLEAN"
    runs["failed_decoder"] = trained & (runs.ref_half <= CLIPPED_HALF)
    runs["failed_train"] = np.where(runs.loss_gain.notna(), runs.loss_gain >= SATURATED_GAIN, np.nan)
    return runs


# ----------------------------------------------------------------------------- statistics
def loso_isotonic(x: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """In-sample and leave-one-session-out R² of an increasing isotonic fit of y on log x."""
    from sklearn.isotonic import IsotonicRegression

    lx = np.log(x)
    pred = np.empty_like(y)
    for group in np.unique(groups):
        test = groups == group
        fit = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(lx[~test], y[~test])
        pred[test] = fit.predict(lx[test])
    sst = float(np.sum((y - y.mean())**2))
    full = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(lx, y)
    r2_in = 1.0 - float(np.sum((y - full.predict(lx))**2)) / sst
    r2_loso = 1.0 - float(np.sum((y - pred)**2)) / sst
    sse = {g: float(np.sum((y[groups == g] - pred[groups == g])**2)) for g in np.unique(groups)}
    return r2_in, r2_loso, sse, full


def auc(failed: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    failed = np.asarray(failed, dtype=bool)
    if failed.all() or not failed.any():
        return float("nan")
    return float(roc_auc_score(failed, score))


def youden(x: np.ndarray, failed: np.ndarray) -> Tuple[Optional[float], float]:
    """Threshold c* ("fail when x ≥ c*") maximising TPR − FPR; c* is the geometric midpoint of the
    neighbouring values. Returns (c*, J)."""
    failed = np.asarray(failed, dtype=bool)
    values = np.unique(x)
    if failed.all() or not failed.any() or len(values) < 2:
        return None, float("nan")
    best_j, best = -np.inf, None
    for lower, upper in zip(values[:-1], values[1:]):
        predicted = x >= upper
        j = (predicted & failed).sum() / failed.sum() - (predicted & ~failed).sum() / (~failed).sum()
        if j > best_j:
            best_j, best = j, math.sqrt(lower * upper)
    return best, float(best_j)


def loso_threshold(x: np.ndarray, failed: np.ndarray, groups: np.ndarray):
    """Accuracy of c* fitted on the other sessions, and the range of the fitted c*."""
    correct, used, thresholds = 0, 0, []
    for group in np.unique(groups):
        test = groups == group
        threshold, _ = youden(x[~test], failed[~test])
        if threshold is None:
            continue
        thresholds.append(threshold)
        correct += int(np.sum((x[test] >= threshold) == failed[test]))
        used += int(test.sum())
    if not used:
        return float("nan"), (float("nan"), float("nan"))
    return correct / used, (min(thresholds), max(thresholds))


def spearman(a, b) -> Tuple[float, float]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 4 or np.ptp(a[keep]) == 0 or np.ptp(b[keep]) == 0:
        return float("nan"), float("nan")
    result = stats.spearmanr(a[keep], b[keep])
    return float(result[0]), float(result[1])


# ----------------------------------------------------------------------------- tests
def test_a(runs: pd.DataFrame, lags: Sequence[int]):
    """Loss gain against each normalisation of the training budget (POOL excluded: a mixture)."""
    signal = [f"x_signal_lag{lag}" for lag in lags]
    predictors = list(A_PREDICTORS) + signal
    rows, session_rows, fits = [], [], {}
    for mode in MODES:
        d = runs[(runs["mode"] == mode) & runs.loss_gain.notna() & runs.metric.isin(list(METRICS))]
        d = d.dropna(subset=predictors)
        d = d[(d[predictors] > 0).all(axis=1)]
        if d.session.nunique() < 3:
            continue
        sets = [(p, d) for p in predictors]
        pairs = d.dropna(subset=["x_signal_pairs"])
        pairs = pairs[pairs.x_signal_pairs > 0]
        if pairs.session.nunique() >= 3:
            sets.append(("x_signal_pairs", pairs))  # white metrics only (no NA runs)
        for predictor, data in sets:
            r2_in, r2_loso, sse, fit = loso_isotonic(data[predictor].values, data.loss_gain.values,
                                                     data.session.values)
            rows.append({"mode": mode, "predictor": predictor, "n_runs": len(data),
                         "n_sessions": data.session.nunique(), "r2_insample": r2_in,
                         "r2_loso": r2_loso})
            session_rows += [{"mode": mode, "predictor": predictor, "session": k, "sse_loso": v}
                             for k, v in sse.items()]
            fits[(mode, predictor)] = fit
    table, per_session = pd.DataFrame(rows), pd.DataFrame(session_rows)
    comparisons = []
    for mode in MODES:
        sub = per_session[per_session["mode"] == mode]
        if sub.empty:
            continue
        pivot = sub.pivot(index="session", columns="predictor", values="sse_loso")
        for base in A_PREDICTORS:
            better = pivot[base] - pivot["x_signal_lag1"]  # > 0: the signal ratio predicts better
            p = stats.wilcoxon(better).pvalue if np.any(better != 0) else 1.0
            comparisons.append({"mode": mode, "baseline": base,
                                "sessions_signal_better": int((better > 0).sum()),
                                "n_sessions": int(better.notna().sum()), "wilcoxon_p": float(p)})
    return table, per_session, pd.DataFrame(comparisons), fits


def partial_spearman(x, y, control) -> float:
    """Spearman correlation of x and y after removing the (rank-)linear effect of ``control``."""
    ranks = [stats.rankdata(np.asarray(v, dtype=float)) for v in (x, y, control)]
    design = np.column_stack([np.ones(len(ranks[2])), ranks[2]])
    residuals = [r - design @ np.linalg.lstsq(design, r, rcond=None)[0] for r in ranks[:2]]
    if min(np.std(residuals[0]), np.std(residuals[1])) == 0:
        return float("nan")
    return float(np.corrcoef(residuals[0], residuals[1])[0, 1])


def test_a_within_arm(runs: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """At one nominal budget the nominal ε explains nothing across sessions; does ε_sig?

    Per arm, over sessions (loss gain averaged over seeds), the Spearman ρ between the loss gain
    and each session-level predictor, signed so that a positive ρ is the direction expected for a
    riskier session. Fewer channels alone already predict a larger gain, so the partial ρ of
    ε_train / ε_sig given the number of channels says whether ε_sig adds to that.
    """
    rows = []
    for mode in MODES:
        d = runs[(runs["mode"] == mode) & runs.loss_gain.notna()]
        for arm, g in d.groupby("short"):
            s = g.groupby("session")[["loss_gain", "x_signal_lag1", "x_signal_lag2", "x_blind"]].mean()
            s = s.join(sig[["kappa_whole", "noise_sd_median", "n_ch"]])
            if len(s) < 6:
                continue
            entry = {"mode": mode, "arm": arm, "n_sessions": len(s), "n_runs": len(g)}
            for name, values in (("x_signal_lag1", s.x_signal_lag1), ("x_signal_lag2", s.x_signal_lag2),
                                 ("x_blind", s.x_blind), ("kappa_whole", s.kappa_whole),
                                 ("low_noise_sd", -s.noise_sd_median), ("few_channels", -s.n_ch)):
                entry[f"rho_{name}"], entry[f"p_{name}"] = spearman(values, s.loss_gain)
            entry["partial_rho_x_signal_lag1_given_n_ch"] = partial_spearman(s.x_signal_lag1,
                                                                             s.loss_gain, s.n_ch)
            rows.append(entry)
    return pd.DataFrame(rows)


def test_b(runs: pd.DataFrame, sig: pd.DataFrame, lags: Sequence[int]):
    """Over-budget failures against ε_train / ε_sig and the baselines."""
    signal = [f"x_signal_lag{lag}" for lag in lags]
    descriptors = {"kappa_whole": 1, "noise_sd_median": -1, "n_ch": -1, "eps_blind_theory": -1}

    # B-i: AUTO (time-only) — the blind budget over-shot in M-RT3 and J-CO0.
    auto = runs[(runs["mode"] == "time_only") & (runs.short == "AUTO")].set_index("session")
    auto_rows = []
    if not auto.empty:
        auto = auto.join(sig[["eps_sig_sp_lag1", "kappa_whole", "noise_sd_median", "n_ch",
                              "eps_blind_theory"]])
        auto["rank_x"] = auto.x_signal_lag1.rank(ascending=False).astype(int)
        auto_rows = auto[["seed", "eps_train", "eps_sig_sp_lag1", "x_signal_lag1", "loss_gain",
                          "failed_train", "failed_decoder", "clean_r2", "d_clean_vs_pgd6",
                          "ref_half", "rank_x"]].copy()
        failed = auto.failed_train.astype(bool).values
        scores = {"x_signal_lag1": auto.x_signal_lag1.values, "eps_blind (budget)": auto.eps_train.values}
        scores.update({k: sign * auto[k].values for k, sign in descriptors.items()})
        auto_rows.attrs["auc"] = {k: auc(failed, v) for k, v in scores.items()}
        k = int(failed.sum())
        top = set(auto.sort_values("x_signal_lag1", ascending=False).index[:k])
        auto_rows.attrs["failures_are_top_x"] = bool(k and top == set(auto.index[failed]))
        auto_rows.attrs["chance_p"] = 1.0 / math.comb(len(auto), k) if k else float("nan")

    # B-ii: supervised brittleness per session and arm.
    sup_rows, sup_summary = [], []
    sup = runs[runs["mode"] == "supervised"]
    for arm in ("SP8", "REG8", "DIFF8", "POOL", "ISO8", "PGD6", "PGD3"):
        d = sup[sup.short == arm]
        if d.empty:
            continue
        g = d.groupby("session").agg(n_brittle=("failed_decoder", "sum"), seeds=("seed", "size"),
                                     x_signal_lag1=("x_signal_lag1", "mean"),
                                     ref_half=("ref_half", "mean"))
        g = g.join(sig[list(descriptors)])
        for session, r in g.iterrows():
            sup_rows.append({"arm": arm, "session": session, **r.to_dict()})
        rho, p = spearman(g.x_signal_lag1, g.n_brittle)
        majority = (g.n_brittle >= 2).values
        entry = {"arm": arm, "sessions_brittle_majority": int(majority.sum()),
                 "spearman_x_vs_brittle": rho, "p": p, "auc_x": auc(majority, g.x_signal_lag1.values),
                 "x_min": g.x_signal_lag1.min(), "x_max": g.x_signal_lag1.max()}
        for name, sign in descriptors.items():
            entry[f"spearman_{name}"] = spearman(sign * g[name], g.n_brittle)[0]
        sup_summary.append(entry)
    both = pd.DataFrame(sup_rows)
    if not both.empty:
        sub = both[both.arm.isin(["SP8", "REG8"])]
        rho, p = spearman(sub.x_signal_lag1, sub.n_brittle)
        sup_summary.append({"arm": "SP8+REG8", "spearman_x_vs_brittle": rho, "p": p,
                            "auc_x": auc((sub.n_brittle >= 2).values, sub.x_signal_lag1.values),
                            "sessions_brittle_majority": int((sub.n_brittle >= 2).sum())})

    # B-iii: all trained runs pooled within a mode.
    pooled, thresholds = [], {}
    for mode in MODES:
        for outcome, subset in (("failed_train", runs.loss_gain.notna()),
                                ("failed_decoder", runs.short != "CLEAN")):
            d = runs[(runs["mode"] == mode) & subset & runs.metric.notna()]
            if d.empty:
                continue
            failed = d[outcome].astype(bool).values
            for predictor in signal + ["x_signal_pairs"] + list(A_PREDICTORS):
                v = d[predictor].values.astype(float)
                keep = np.isfinite(v) & (v > 0)
                entry = {"mode": mode, "outcome": outcome, "predictor": predictor,
                         "n_runs": int(keep.sum()), "n_failed": int(failed[keep].sum()),
                         "auc": auc(failed[keep], v[keep])}
                if predictor in signal or predictor == "x_nominal":
                    c, j = youden(v[keep], failed[keep])
                    acc, (lo, hi) = loso_threshold(v[keep], failed[keep], d.session.values[keep])
                    entry.update({"c_star": c, "youden_j": j, "loso_accuracy": acc,
                                  "loso_c_min": lo, "loso_c_max": hi})
                    if predictor == "x_signal_lag1":
                        thresholds[(mode, outcome)] = c
                pooled.append(entry)
    return auto_rows, pd.DataFrame(sup_rows), pd.DataFrame(sup_summary), pd.DataFrame(pooled), \
        thresholds


def test_c(runs: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """Robustness reached (seed-mean ε½ per session) against the signal separation."""
    descriptors = ("n_ch", "noise_sd_median", "kappa_whole", "eps_blind_theory", "psi_sum")
    rows = []
    for mode in MODES:
        for arm in ("CLEAN", "PGD3", "PGD6", "ISO8", "SP8", "DIFF8", "POOL", "REG8"):
            d = runs[(runs["mode"] == mode) & (runs.short == arm)]
            if d.session.nunique() < 5:
                continue
            g = d.groupby("session")[["ref_half", "iso_half"]].mean().join(sig)
            rho_ref, p_ref = spearman(g.ref_half, g.eps_sig_ref_lag1)
            rho_iso, p_iso = spearman(g.iso_half, g.eps_sig_iso_lag1)
            entry = {"mode": mode, "arm": arm, "n_sessions": len(g),
                     "spearman_ref_vs_eps_sig_ref": rho_ref, "p_ref": p_ref,
                     "spearman_iso_vs_eps_sig_iso": rho_iso, "p_iso": p_iso,
                     "efficiency_ref_median": float(np.median(g.ref_half / g.eps_sig_ref_lag1)),
                     "efficiency_iso_median": float(np.median(g.iso_half / g.eps_sig_iso_lag1))}
            for name in descriptors:
                entry[f"spearman_ref_vs_{name}"] = spearman(g.ref_half, g[name])[0]
            rows.append(entry)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 6.5,
                         "savefig.dpi": 200})
    return plt


def _log_x(ax) -> None:
    """Log x axis with plain tick labels (1, 2, 5 per decade)."""
    from matplotlib import ticker
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def figure_a(runs, table, fits, path):
    plt = _style()
    panels = [("x_nominal", "nominal ε (d′, own metric)"), ("x_ref_size", "realised reference d′"),
              ("x_signal_lag1", "ε_train / ε_sig(M_train)")]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey="row")
    for i, mode in enumerate(MODES):
        d = runs[(runs["mode"] == mode) & runs.loss_gain.notna()]
        for j, (predictor, label) in enumerate(panels):
            ax = axes[i, j]
            for metric, g in d.groupby("metric"):
                if metric not in PALETTE:
                    continue
                hollow = metric == "pool"
                ax.scatter(g[predictor], g.loss_gain, s=16, marker=MARKERS[metric],
                           facecolor="white" if hollow else PALETTE[metric],
                           edgecolor=PALETTE[metric], lw=0.8, alpha=0.85, label=LABELS[metric])
            auto = d[d.short == "AUTO"]
            ax.scatter(auto[predictor], auto.loss_gain, s=46, facecolor="none", edgecolor="#0b0b0b",
                       lw=0.8, label="AUTO (ringed)")
            fit = fits.get((mode, predictor))
            if fit is not None:
                xs = np.linspace(fit.X_min_, fit.X_max_, 200)
                ax.plot(np.exp(xs), fit.predict(xs), color="#52514e", lw=1.2)
            row = table[(table["mode"] == mode) & (table.predictor == predictor)]
            r2 = f"LOSO R² = {row.r2_loso.iloc[0]:.2f}" if len(row) else ""
            ax.set_title(f"{mode} · {label}\n{r2}", fontsize=7.5, loc="left")
            _log_x(ax)
            ax.axhline(SATURATED_GAIN, color="#c3c2b7", lw=0.8, ls="--")
            if j == 0:
                ax.set_ylabel("training loss gain (median)")
                ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_b(runs, thresholds, path):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, mode in zip(axes, MODES):
        d = runs[(runs["mode"] == mode) & runs.metric.notna() & (runs.x_signal_lag1 > 0)]
        arms = [a for a in ("PGD3", "PGD6", "PGD12", "ISO8", "ISO16", "SP4", "SP8", "SP16", "NA4",
                            "AUTO", "DIFF8", "POOL", "REG8") if a in set(d.short)]
        for k, arm in enumerate(arms):
            g = d[d.short == arm]
            failed = g.failed_decoder.astype(bool) | (g.failed_train == 1)
            jitter = (np.random.default_rng(k).random(len(g)) - 0.5) * 0.3
            ax.scatter(g.x_signal_lag1[~failed], k + jitter[~failed.values], s=14, facecolor="white",
                       edgecolor="#898781", lw=0.8)
            ax.scatter(g.x_signal_lag1[failed], k + jitter[failed.values], s=16, color="#0b0b0b")
        for outcome, style in (("failed_train", "--"), ("failed_decoder", ":")):
            c = thresholds.get((mode, outcome))
            if c:
                ax.axvline(c, color="#2a78d6", ls=style, lw=1, label=f"c* {outcome} = {c:.2f}")
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels(arms)
        _log_x(ax)
        ax.set_xlabel("ε_train / ε_sig(M_train)   (filled: failed — loss gain ≥ 2 or ε½ clipped)")
        ax.set_title(mode, loc="left")
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_c(runs, sig, table, path):
    plt = _style()
    colors = {"CLEAN": "#b4b2aa", "PGD6": "#3d3c39", "SP8": "#2a78d6", "DIFF8": "#1baf7a",
              "POOL": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, mode in zip(axes, MODES):
        for arm, color in colors.items():
            d = runs[(runs["mode"] == mode) & (runs.short == arm)]
            if d.empty:
                continue
            g = d.groupby("session").ref_half.mean().to_frame().join(sig.eps_sig_ref_lag1)
            row = table[(table["mode"] == mode) & (table.arm == arm)]
            rho = row.spearman_ref_vs_eps_sig_ref.iloc[0] if len(row) else float("nan")
            ax.scatter(g.eps_sig_ref_lag1, g.ref_half, s=20, color=color, label=f"{arm} (ρ = {rho:.2f})")
        ax.axhline(1.0, color="#c3c2b7", lw=0.8, ls="--")
        ax.set_xlabel("ε_sig in the reference metric (signal separation, d′)")
        ax.set_title(mode, loc="left")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("reference ε½ (seed mean)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- summary
def _fmt(value, spec=".2f"):
    try:
        return "nan" if value is None or not np.isfinite(value) else format(value, spec)
    except TypeError:
        return str(value)


GUIDE = """How to read (the hypothesis is supported if):
  A   x_signal_* has a higher LOSO R² than x_nominal and x_blind (the a-priori baselines), and the
      within-arm ρ of x_signal is positive: at one nominal ε, sessions with a smaller ε_sig get a
      larger loss gain. Fewer channels alone already predict a larger gain (x_blind does exactly
      that), so the partial ρ given the channel count shows whether ε_sig adds information.
      x_ref_size / x_raw_size are realised sizes, known only after training: a reference, not a
      competitor for an a-priori rule.
  B-i the sessions where AUTO saturated (loss gain ≥ 2) have the largest ε_blind / ε_sig.
  B-ii supervised SP8 / REG8 are brittle where x = 8 / ε_sig is large (positive ρ, AUC > 0.5),
      and DIFF8 / POOL stay below that range of x.
  B-iii one threshold c* on x_signal separates failed from safe runs, stably across sessions
      (LOSO accuracy, narrow c* range); in supervised CEBRA ε½ is also clipped for weak models
      in hard sessions (e.g. PGD6 in T-RT1), which this test cannot separate from over-budget.
  C   positive ρ: decoders reach a larger ε½ where the signal separation is larger.
"""


def write_summary(path, sig, a_table, a_comp, a_within, auto, sup_summary, pooled, thresholds,
                  c_table):
    out = ["SIGNAL-REFERENCED TRAINING BUDGET — RETROSPECTIVE TESTS (no training)", "", GUIDE]
    out.append("Stage 1: signal separation ε_sig (lag-1, d′ in each training metric; raw in counts); "
               "eps_nn_note_sp: the note's ε_NN (noise-dominated), sqrt_2D: its noise floor")
    columns = ["n_ch", "sqrt_2D", "eps_nn_note_sp", "eps_blind_theory", "eps_sig_sp_lag1",
               "eps_sig_diff_lag1", "eps_sig_reg_lag1",
               "eps_sig_iso_lag1", "eps_sig_ref_lag1", "eps_sig_raw_lag1", "eps_sig_diff_pairs",
               "psi_sum", "n_modes_window_detectable", "kappa_whole"]
    columns = [c for c in columns if c in sig]
    out.append(sig[columns].round(2).to_string())
    out += ["", "Test A: training loss gain against each budget normalisation (LOSO isotonic R²)"]
    if not a_table.empty:
        out.append(a_table.round(3).to_string(index=False))
        out.append("")
        out.append("Per-session LOSO error: sessions where x_signal_lag1 predicts better than the baseline")
        out.append(a_comp.round(4).to_string(index=False))
    out += ["", "Test A (within arm, over sessions): Spearman ρ of the seed-mean loss gain with each "
            "session-level predictor; last column: partial ρ of ε_train/ε_sig given the channel count"]
    if not a_within.empty:
        columns = (["mode", "arm", "n_sessions"] + [c for c in a_within.columns if c.startswith("rho_")]
                   + ["partial_rho_x_signal_lag1_given_n_ch"])
        out.append(a_within[columns].round(3).to_string(index=False))
    out += ["", "Test B-i: AUTO (time-only); failures should have the largest ε_blind / ε_sig"]
    if isinstance(auto, pd.DataFrame) and not auto.empty:
        out.append(auto.round(3).to_string())
        out.append("AUC (loss-gain saturation) per predictor: " +
                   ", ".join(f"{k} {_fmt(v)}" for k, v in auto.attrs.get("auc", {}).items()))
        out.append(f"failing sessions are the top-k of ε_blind/ε_sig: "
                   f"{auto.attrs.get('failures_are_top_x')} (chance {_fmt(auto.attrs.get('chance_p'), '.3f')})")
    out += ["", "Test B-ii: supervised brittleness (ε½ clipped at 0.5) per arm across sessions"]
    if not sup_summary.empty:
        out.append(sup_summary.round(3).to_string(index=False))
    out += ["", "Test B-iii: all runs pooled (AUC of each predictor; c*: fail when x ≥ c*)"]
    if not pooled.empty:
        out.append(pooled.round(3).to_string(index=False))
    out += ["", "Test C: reference ε½ (seed mean per session) against the signal separation"]
    if not c_table.empty:
        out.append(c_table.round(3).to_string(index=False))
    out += ["", "Preview — budget at the fitted failure boundary, c*·ε_sig(M), per session"]
    for (mode, outcome), c in thresholds.items():
        if c is None:
            continue
        boundary = pd.DataFrame({m: c * sig[f"eps_sig_{m}_lag1"] for m in ("sp", "diff", "reg", "iso")
                                 if f"eps_sig_{m}_lag1" in sig})
        boundary["eps_blind_theory"] = sig.eps_blind_theory
        out.append(f"{mode}, {outcome}: c* = {c:.3f}")
        out.append(boundary.round(2).to_string())
    Path(path).write_text("\n".join(out) + "\n")


# ----------------------------------------------------------------------------- self-test
def self_test(noise) -> None:
    """Synthetic x = s + n (slow AR(1) signal, white noise): ε_sig must match its true value."""
    rng = np.random.default_rng(0)
    n, channels, window_len, k, rho = 60000, 24, 36, 3, 0.98
    loading = rng.normal(size=(channels, k)) * 0.5
    z = np.zeros((n, k))
    shocks = rng.normal(size=(n, k)) * math.sqrt(1 - rho**2)
    for t in range(1, n):
        z[t] = rho * z[t - 1] + shocks[t]
    noise_sd = rng.uniform(0.6, 1.6, channels)
    x = z @ loading.T + rng.normal(size=(n, channels)) * noise_sd + 5.0
    signal_cov = loading @ loading.T  # Var(z) = 1
    row, _ = session_statistics(noise, x, None, window_len, (1, 2), 3000, 0)
    worst = 0.0
    for name, (residuals, estimator, temporal) in METRICS.items():
        if residuals == "regression":
            continue
        model, _ = noise.fit_noise_model(x, None, window_len=window_len, residual_mode=residuals,
                                         spatial_estimator=estimator, temporal_method=temporal,
                                         kronecker_channels=0)
        spatial_inv = np.linalg.inv(model.spatial_cov)
        temporal_inv = np.linalg.inv(model.temporal_cov)
        lags = np.abs(np.subtract.outer(np.arange(window_len), np.arange(window_len)))
        truth = math.sqrt(2.0 * np.trace(spatial_inv @ signal_cov) * np.sum(temporal_inv * rho**lags))
        estimate = row[f"eps_sig_{name}_lag1"]
        worst = max(worst, abs(estimate / truth - 1.0))
        print(f"  {name:4s}: ε_sig estimate {estimate:7.3f}  truth {truth:7.3f}  "
              f"pairs {_fmt(row.get(f'eps_sig_{name}_pairs'), '.3f')}")
    print(f"self-test: largest relative error {worst:.3%} (lag-1 bias ≈ 1 − ρ = {1 - rho:.0%})")
    if worst > 0.06:
        raise SystemExit("self-test FAILED")
    print("self-test passed")


# ----------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fork-dir", type=Path, default=Path("Acorn-dprime"))
    p.add_argument("--data-dir", type=Path, default=Path("/data/hossein/mm_project/perich_data_valid_final_raw"))
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Folder with DPRIME_EXP5, DPRIME_EXP6 and DPRIME_SUPERVISED_PERICH_EXP8")
    p.add_argument("--out-dir", type=Path, default=Path("SIGNAL_BUDGET_TESTS"))
    p.add_argument("--sessions", nargs="+", default=list(SESSIONS))
    p.add_argument("--window-len", type=int, default=36, help="Receptive field T of offset36 models")
    p.add_argument("--lags", nargs="+", type=int, default=[1, 2, 3],
                   help="Lags used as the signal proxy (1 = difference residuals; 2, 3: robustness)")
    p.add_argument("--pairs", type=int, default=4000, help="Window pairs of the empirical check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-stage1", action="store_true", help="Reuse <out-dir>/session_signal.csv")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--self-test", action="store_true", help="Synthetic check of ε_sig and exit")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if 1 not in args.lags:
        raise SystemExit("--lags must include 1 (the primary estimate)")
    args.lags = sorted(set(args.lags))
    noise = load_noise_module(args.fork_dir)
    if args.self_test:
        self_test(noise)
        return 0
    if args.results_dir is None:
        raise SystemExit("--results-dir is required")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_stage1:
        sig = pd.read_csv(args.out_dir / "session_signal.csv").set_index("session")
    else:
        sig = stage1(args, noise)

    runs = add_predictors(parse_runs(args.results_dir, args.sessions), sig, args.lags)
    runs.to_csv(args.out_dir / "runs.csv", index=False)
    print(f"{len(runs)} evaluated models; {int(runs.loss_gain.notna().sum())} with training sizes")

    a_table, a_sessions, a_comp, fits = test_a(runs, args.lags)
    a_within = test_a_within_arm(runs, sig)
    a_table.to_csv(args.out_dir / "testA_collapse.csv", index=False)
    a_sessions.to_csv(args.out_dir / "testA_sessions.csv", index=False)
    a_comp.to_csv(args.out_dir / "testA_comparisons.csv", index=False)
    a_within.to_csv(args.out_dir / "testA_within_arm.csv", index=False)
    auto, sup_rows, sup_summary, pooled, thresholds = test_b(runs, sig, args.lags)
    if isinstance(auto, pd.DataFrame):
        auto.to_csv(args.out_dir / "testB_auto.csv")
    sup_rows.to_csv(args.out_dir / "testB_supervised.csv", index=False)
    sup_summary.to_csv(args.out_dir / "testB_supervised_summary.csv", index=False)
    pooled.to_csv(args.out_dir / "testB_pooled.csv", index=False)
    c_table = test_c(runs, sig)
    c_table.to_csv(args.out_dir / "testC_margin.csv", index=False)
    write_summary(args.out_dir / "SUMMARY.txt", sig, a_table, a_comp, a_within, auto, sup_summary,
                  pooled, thresholds, c_table)
    if not args.no_figures:
        for name, draw in (("fig_testA.png", lambda p: figure_a(runs, a_table, fits, p)),
                           ("fig_testB.png", lambda p: figure_b(runs, thresholds, p)),
                           ("fig_testC.png", lambda p: figure_c(runs, sig, c_table, p))):
            try:
                draw(args.out_dir / name)
            except Exception as error:  # a figure must not cost the tables
                warnings.warn(f"{name} failed: {error!r}")
    print((args.out_dir / "SUMMARY.txt").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

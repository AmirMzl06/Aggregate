"""Exp16 (analysis only; CPU; no training): the alpha-quantile overlap bound on the Exp14/15 runs.

Keep next to dprime_exp9.py, half_rule.py and the Acorn-dprime fork. The fork is only used for its
noise model and estimators; nothing is trained.

Theory (the reformulation after the 2026-11-07 critique). Let D = ||s_a - s_b|| be the distance
between the task signals of a reference window a and a sampled negative window b, in the d' units
of the training metric (noise-whitened). The eps-balls around s_a and s_b intersect if and only if
D <= 2 eps. Then one input lies within eps of both windows, so no deterministic encoder can satisfy
both robust contrastive comparisons of that pair. Hence, for any distribution of D (the definition
of a quantile):

    eps <= Q_alpha(D) / 2   =>   P(a sampled negative pair overlaps) <= alpha.

In units of the RMS separation eps_task (the fork's eps_beh / eps_sig, the reference of every
budget so far): x_alpha = Q_alpha(D) / (2 eps_task), with eps_task^2 = E[D^2]. x_alpha depends only
on the SHAPE of the distribution of D. Which overlap level alpha leads to a collapse or to a loss of
robustness is not given by the bound; this script measures it on the runs that exist. It is
retrospective: nothing here is pre-registered.

Shapes of D per session (training split; DIFF metric: difference residuals, Ledoit-Wolf, white):
  emp_beh      empirical. The behavioural signal is the fork's cross-fitted behaviour prediction
               (calibration.behaviour_prediction; uses the labels: the supervised task). Negatives
               are drawn uniformly in time, as CEBRA draws them.
  emp_beh_far  the same, only temporally non-overlapping pairs (|t_a - t_b| >= T).
  gauss_beh    Gaussian with the window covariance of the same prediction (checks the Gaussian shape).
  gauss_sig    Gaussian with the window covariance of the slow signal: the lag covariances of the
               recording (lag k >= 1 is signal for temporally white noise; lag 0 = sym lag 1, as in
               calibration.eps_signal), restricted to the modes above the sampling-noise edge (the
               criterion of calibration.signal_projector). Label-free: the time-only task.
  A Gaussian window signal with eigenvalues lambda gives D^2 = 2 sum lambda_i xi_i^2 (simulated).

Tests (DIFF-metric arms only; failure = refined reference eps1/2 < 0.5, as in Exp11-15):
  T1  are the failures only where z = x / x_alpha > 1 (x = eps / eps_task of the run)?
  T2  does z rank the failures better than x (AUC)? I.e. does a session's own distance distribution
      explain which sessions break early (C-RT4 at x 0.384)?
  T3  kappa relative to the session's AUTO run, in bins of z.
The time-only runs are also scored with x_beh / x_alpha(emp_beh) (uses labels, not label-free), to see
whether the behavioural or the whole slow signal sets the time-only limit.

Usage (CPU only; the per-session shapes are cached in --quantiles-csv, so a re-run only redoes the
tests; --all-sessions adds every session of --data-dir to the overview):
  python -u overlap_quantiles.py \\
      --sup-root /data/hossein/mm_project/EXP15/PERICH_SUP \\
      --time-root /data/hossein/mm_project/EXP15/PERICH_TIME \\
      --exp14-csv /path/to/EXP14_ALL_RESULTS.csv > overlap_report.txt
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import dprime_exp9 as runner  # noqa: E402  data loading, noise models, atomic writes
import half_rule as hr  # noqa: E402  the Exp15 records and the DIFF arm

T = runner.WINDOW
ALPHAS = (0.01, 0.05, 0.10, 0.50)
SHAPES = ("emp_beh", "emp_beh_far", "gauss_beh", "gauss_sig")
MODE_SHAPES = {"supervised": ("emp_beh", "emp_beh_far", "gauss_beh"), "time_only": ("gauss_sig",)}
EXP14_DIFF_ARMS = ("MADRY_SPDIFF_D8", "MADRY_SPDIFF_AUTOBEH", "MADRY_SPDIFF_AUTOSIG")
GAUSS_DRAWS = 100_000
BEH_CUMULATIVE = 0.999   # gauss_beh: spatial modes up to this share of the prediction's variance
Z_BINS = (0.0, 0.8, 1.0, 1.25, 1.6, np.inf)


def _key(alpha: float) -> str:
    return f"x{int(round(100 * alpha)):02d}"


# ============================================================================ shapes of D
def pair_d2(z: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 1000) -> np.ndarray:
    """Squared distances between the T-bin windows starting at a and at b of z (n, C)."""
    idx = np.arange(T)
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        diff = z[a[i:i + chunk, None] + idx] - z[b[i:i + chunk, None] + idx]
        out[i:i + chunk] = np.einsum("mtc,mtc->m", diff, diff)
    return out


def empirical_shape(z: np.ndarray, rng, pairs: int, far: bool) -> Dict[str, float]:
    """x_alpha of the empirical pair distances (negatives uniform in time; far: no overlap)."""
    windows = len(z) - T + 1
    a = rng.integers(0, windows, size=4 * pairs)
    b = rng.integers(0, windows, size=4 * pairs)
    keep = (np.abs(a - b) >= T) if far else (a != b)
    a, b = a[keep][:pairs], b[keep][:pairs]
    d2 = pair_d2(z, a, b)
    mean = float(d2.mean())
    out = {_key(alpha): 0.5 * math.sqrt(float(np.quantile(d2, alpha)) / mean) for alpha in ALPHAS}
    out["k_eff"] = 2 * mean**2 / float(d2.var()) if d2.var() > 0 else math.nan  # chi2 moment match
    out["pairs"] = int(len(d2))
    return out


def lag_covariances(z: np.ndarray, first: int) -> List[np.ndarray]:
    """Cov(z_{t+k}, z_t) for k = first ... T-1 (z centred)."""
    n = len(z)
    return [z[k:].T @ z[:n - k] / (n - k) for k in range(first, T)]


def window_covariance(blocks: List[np.ndarray]) -> np.ndarray:
    """(rT, rT) covariance of a T-bin window (time-major) from Gamma_0 ... Gamma_{T-1}."""
    r = blocks[0].shape[0]
    k = np.empty((T * r, T * r))
    for i in range(T):
        for j in range(T):
            k[i * r:(i + 1) * r, j * r:(j + 1) * r] = blocks[i - j] if i >= j else blocks[j - i].T
    return 0.5 * (k + k.T)


def gaussian_shape(cov: np.ndarray, rng, draws: int = GAUSS_DRAWS) -> Dict[str, float]:
    """x_alpha of D for two independent Gaussian windows with covariance cov (D^2 = 2 sum l xi^2)."""
    lam = np.clip(np.linalg.eigvalsh(cov), 0, None)
    if lam.max() <= 0:
        return {**{_key(a): math.nan for a in ALPHAS}, "k_eff": math.nan, "modes": 0}
    lam = lam[lam > 1e-9 * lam.max()]
    w = lam / lam.sum()
    samples = np.empty(draws)
    step = max(1, 2_000_000 // len(w))
    for i in range(0, draws, step):
        m = min(step, draws - i)
        samples[i:i + m] = (rng.standard_normal((m, len(w)))**2) @ w  # D^2 / E[D^2]
    out = {_key(alpha): 0.5 * math.sqrt(float(np.quantile(samples, alpha))) for alpha in ALPHAS}
    out["k_eff"] = 1.0 / float((w**2).sum())
    return out


def behaviour_blocks(zp: np.ndarray) -> List[np.ndarray]:
    """Lag covariances of the whitened prediction in its leading spatial modes."""
    c0 = zp.T @ zp / len(zp)
    psi, vectors = np.linalg.eigh(0.5 * (c0 + c0.T))
    order = np.argsort(psi)[::-1]
    psi, vectors = np.clip(psi[order], 0, None), vectors[:, order]
    count = int(np.searchsorted(np.cumsum(psi) / max(psi.sum(), 1e-300), BEH_CUMULATIVE) + 1)
    zr = zp @ vectors[:, :min(count, len(psi))]
    return lag_covariances(zr, 0)


def slow_signal_blocks(zx: np.ndarray) -> List[np.ndarray]:
    """Lag covariances of the slow signal of the whitened recording, in the signal modes of
    calibration.signal_projector (psi > max(1/T, 1.2 sqrt(2C/n))); lag 0 = sym lag 1."""
    from cebra.dprime import calibration
    n, channels = zx.shape
    g1 = zx[1:].T @ zx[:-1] / (n - 1)
    slow = 0.5 * (g1 + g1.T)
    psi, vectors = np.linalg.eigh(slow)
    order = np.argsort(psi)[::-1]
    psi, vectors = psi[order], vectors[:, order]
    threshold = max(1.0 / T, calibration._NOISE_EDGE_MARGIN * math.sqrt(2.0 * channels / n))
    count = min(max(int(np.sum(psi > threshold)), 1), channels)
    u = vectors[:, :count]
    zr = zx @ u
    blocks = lag_covariances(zr, 1)
    return [0.5 * (blocks[0] + blocks[0].T)] + blocks  # Gamma_0 := sym Gamma_1


def session_shapes(x: np.ndarray, y: np.ndarray, seed: int, pairs: int) -> Dict[str, Any]:
    """Separations and the four shapes of D of one training recording (x: counts, y: labels)."""
    from cebra.dprime import calibration, noise
    exp = hr.half_arm("supervised")  # the DIFF metric of every Exp14/15 arm analysed here
    spatial, temporal = runner._metric_members(exp, x, None, noise)[0]
    if not np.allclose(temporal, np.eye(T)):
        raise RuntimeError("Expected the white temporal factor of the DIFF metric")
    ws = calibration._symmetric_power(np.asarray(spatial, dtype=np.float64), -0.5)
    xd = np.asarray(x, dtype=np.float64)
    prediction, model = calibration.behaviour_prediction(
        xd, y, lags=runner.BEH_LAGS, features=runner.BEH_FEATURES, seed=0)
    eps_beh = calibration.eps_behaviour(xd, prediction, spatial, temporal)
    eps_sig = calibration.eps_signal(xd, spatial, temporal, runner.SIG_LAG)
    zx = (xd - xd.mean(0)) @ ws
    zp = (prediction - prediction.mean(0)) @ ws
    rng = np.random.default_rng(seed)
    out: Dict[str, Any] = dict(n_train=int(len(x)), neurons=int(x.shape[1]),
                               behaviour_r2=float(model["r2"]), eps_beh=eps_beh, eps_sig=eps_sig,
                               rho=eps_beh / eps_sig)
    shapes = {"emp_beh": empirical_shape(zp, rng, pairs, far=False),
              "emp_beh_far": empirical_shape(zp, rng, pairs, far=True),
              "gauss_beh": gaussian_shape(window_covariance(behaviour_blocks(zp)), rng)}
    slow = slow_signal_blocks(zx)
    shapes["gauss_sig"] = {**gaussian_shape(window_covariance(slow), rng),
                           "modes": int(slow[0].shape[0])}
    for shape, values in shapes.items():
        out.update({f"{shape}_{k}": v for k, v in values.items()})
    return out


# ============================================================================ runs
def _refined_half(d, half_col: str, refined_col: str, status_col: str):
    import pandas as pd
    value = (pd.to_numeric(d[refined_col], errors="coerce") if refined_col in d
             else pd.Series(np.nan, index=d.index))
    value = value.where(value.notna(), pd.to_numeric(d[half_col], errors="coerce"))
    status = d[status_col].fillna("") if status_col in d else pd.Series("", index=d.index)
    value = value.where(status != "below_floor", hr.REFINE_FLOOR)
    return value.where(status != "above_grid", hr.REFINE_TOP)


def load_runs(args):
    """DIFF-metric runs of Exp15 (AUTO, HALF, ladder) and Exp14 (DIFF8, AUTO)."""
    import pandas as pd
    parts = []
    for root in (args.sup_root, args.time_root):
        if root is None or not (Path(root) / hr.STATE_FILE).is_file():
            if root is not None:
                print(f"(no {hr.STATE_FILE} in {root}: skipped)")
            continue
        mode = json.loads((Path(root) / hr.STATE_FILE).read_text(encoding="utf-8"))["mode"]
        d = hr.load_runs(Path(root))
        if d is not None:
            parts.append(d.assign(source="Exp15", mode=mode)[
                ["source", "mode", "session", "seed", "experiment", "x", "half", "kappa", "fail"]])
    if args.exp14_csv is not None:
        d = pd.read_csv(args.exp14_csv, low_memory=False)
        d = d[(d.status == "success") & d.experiment.isin(EXP14_DIFF_ARMS)].copy()
        d["half"] = _refined_half(d, "ref_half", "ref_half_refined", "ref_status")
        d["kappa"] = d.half / pd.to_numeric(d.ref_eps_beh, errors="coerce")
        d["fail"] = d.half < hr.FAIL_REF_HALF
        d["x"] = np.where(d["mode"] == "supervised", d.prereg_x_beh, d.prereg_x_sig).astype(float)
        parts.append(d.assign(source="Exp14")[
            ["source", "mode", "session", "seed", "experiment", "x", "half", "kappa", "fail"]])
    if not parts:
        return None
    runs = pd.concat(parts, ignore_index=True)
    runs = runs[runs.x.notna() & runs.half.notna()].copy()
    auto = runs[runs.experiment.isin(hr.AUTO.values())]
    base = auto.groupby(["source", "mode", "session"]).kappa.median().rename("kappa_auto")
    runs = runs.merge(base.reset_index(), on=["source", "mode", "session"], how="left")
    runs["rel_kappa"] = runs.kappa / runs.kappa_auto
    return runs


# ============================================================================ shapes cache
def compute_shapes(args, sessions: List[str]):
    import pandas as pd
    path = args.quantiles_csv
    table = pd.read_csv(path) if path.is_file() and not args.recompute else None
    done = set() if table is None else set(table.session)
    rows = [] if table is None else table.to_dict("records")
    todo = [s for s in sessions if s not in done]
    if todo:
        print(f"computing the distance shapes of {len(todo)} sessions (cached in {path})", flush=True)
    for session in todo:
        try:
            x, _, y, _ = runner.load_data(args.data_dir / f"{session}.npz")
            row = {"session": session, **session_shapes(x, y, args.seed, args.pairs)}
        except Exception as exc:  # an unusable session: recorded, skipped by the tests
            row = {"session": session, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        if "error" in row:
            print(f"  {session}: {row['error']}", flush=True)
        else:
            print(f"  {session}: n {row['n_train']} C {row['neurons']} rho {row['rho']:.2f} | x05 "
                  + " ".join(f"{s} {row[f'{s}_x05']:.3f}" for s in SHAPES)
                  + f" | k_eff beh {row['emp_beh_k_eff']:.1f} sig {row['gauss_sig_k_eff']:.1f}",
                  flush=True)
        stream = io.StringIO()
        pd.DataFrame(rows).to_csv(stream, index=False)
        runner.atomic_write_text(path, stream.getvalue())
    table = pd.DataFrame(rows)
    if "error" in table:
        table = table[table.error.isna()]
    return table[table.session.isin(sessions)].copy()


# ============================================================================ tests
def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score of a failure > score of a non-failure) + 0.5 P(tie)."""
    import pandas as pd
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    ranks = pd.Series(np.concatenate([neg, pos])).rank().to_numpy()
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def overview(shapes) -> None:
    print("\n0  distance shapes per session: x_alpha = Q_alpha(D) / (2 eps_task) "
          "(= half the alpha-quantile of the pair distance in units of its RMS)")
    cols = ["n_train", "neurons", "rho"]
    for shape in SHAPES:
        cols += [f"{shape}_{_key(a)}" for a in ALPHAS[:2]]
    cols += ["emp_beh_x50", "emp_beh_k_eff", "gauss_beh_k_eff", "gauss_sig_k_eff", "gauss_sig_modes"]
    print(shapes.set_index("session")[cols].round(3).to_string())
    print("\n   summary over sessions: median [min, max]")
    for shape in SHAPES:
        parts = []
        for alpha in ALPHAS:
            v = shapes[f"{shape}_{_key(alpha)}"]
            parts.append(f"{_key(alpha)} {v.median():.3f} [{v.min():.3f}, {v.max():.3f}]")
        print(f"   {shape:12s} " + "  ".join(parts))
    for a, b in (("emp_beh_x05", "gauss_beh_x05"), ("emp_beh_x05", "emp_beh_far_x05"),
                 ("emp_beh_x05", "gauss_sig_x05"), ("emp_beh_x05", "n_train"),
                 ("gauss_sig_x05", "n_train")):
        r = shapes[[a, b]].corr(method="spearman").iloc[0, 1]
        print(f"   Spearman({a}, {b}) = {r:+.2f}")


def mode_tests(runs, shapes, mode: str) -> None:
    import pandas as pd
    d = runs[runs["mode"] == mode].merge(shapes, on="session", how="inner")
    if d.empty:
        return
    print(f"\n==================== {mode}: {len(d)} DIFF-metric runs "
          f"({d.groupby('source').size().to_dict()}), {d.session.nunique()} sessions, "
          f"{int(d.fail.sum())} failures")
    predictors = [(f"{shape} {_key(alpha)}", d.x / d[f"{shape}_{_key(alpha)}"])
                  for shape in MODE_SHAPES[mode] for alpha in ALPHAS]
    if mode == "time_only":  # the behavioural part of the slow signal (uses labels)
        x_beh = d.x * d.eps_sig / d.eps_beh
        predictors += [(f"x_beh / emp_beh {_key(alpha)}", x_beh / d[f"emp_beh_{_key(alpha)}"])
                       for alpha in ALPHAS]
    print(f"1  T1 failures by side of z = 1 and T2 AUC (failure ranking); baseline AUC of x "
          f"{auc(d.x, d.fail):.3f}" + (f", of x_beh {auc(d.x * d.eps_sig / d.eps_beh, d.fail):.3f}"
                                         if mode == "time_only" else ""))
    print(f"   {'predictor':28s} {'fails z<=1':>12s} {'fails z>1':>12s} {'AUC(z)':>7s}")
    for label, z in predictors:
        inside = z <= 1
        print(f"   {label:28s} {int(d.fail[inside].sum()):>5d}/{int(inside.sum()):<6d} "
              f"{int(d.fail[~inside].sum()):>5d}/{int((~inside).sum()):<6d} {auc(z, d.fail):7.3f}")
    print("\n2  T3 kappa relative to the session's AUTO run, by z (alpha 5 %)")
    for shape in MODE_SHAPES[mode]:
        z = d.x / d[f"{shape}_x05"]
        bins = pd.cut(z, Z_BINS, right=True)
        t = d.assign(z=z, bin=bins).groupby("bin", observed=True).agg(
            runs=("fail", "size"), fails=("fail", "sum"), x=("x", "median"),
            rel_kappa=("rel_kappa", "median"))
        print(f"   {shape}:")
        print("   " + t.round(3).to_string().replace("\n", "\n   "))
    failed = d[d.fail]
    if len(failed):
        cols = ["source", "session", "seed", "experiment", "x"] + [
            f"{s}_x05" for s in MODE_SHAPES[mode]] + (["rho"] if mode == "time_only" else [])
        print("\n3  the failed runs (x and the 5 % overlap budget of their session)")
        print(failed[cols].sort_values("x").round(3).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=runner.DEFAULT_DATA_DIR)
    p.add_argument("--fork-dir", type=Path, default=runner.DEFAULT_FORK_DIR)
    p.add_argument("--sup-root", type=Path, default=ROOT / hr.OUTPUT["supervised"],
                   help="Exp15 supervised folder (HALF_RULE.json + runs)")
    p.add_argument("--time-root", type=Path, default=ROOT / hr.OUTPUT["time_only"],
                   help="Exp15 time-only folder")
    p.add_argument("--exp14-csv", type=Path, default=None, help="EXP14_ALL_RESULTS.csv (optional)")
    p.add_argument("--quantiles-csv", type=Path, default=ROOT / "OVERLAP_QUANTILES.csv",
                   help="Per-session shapes (cache; delete or --recompute to redo)")
    p.add_argument("--sessions", nargs="+", default=None, help="Only these sessions")
    p.add_argument("--all-sessions", action="store_true",
                   help="Also every session of --data-dir (overview only)")
    p.add_argument("--pairs", type=int, default=20000, help="Window pairs per empirical shape")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recompute", action="store_true")
    return p


def main() -> int:
    import pandas as pd
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 300)
    args = build_parser().parse_args()
    runner.load_fork(args.fork_dir.resolve())
    runs = load_runs(args)
    sessions = set(args.sessions or [])
    if not args.sessions and runs is not None:
        sessions |= set(runs.session)
    if args.all_sessions:
        sessions |= {f.stem for f in args.data_dir.glob("*.npz")}
    if not sessions:
        print("No sessions: give the Exp15 folders / --exp14-csv, --sessions or --all-sessions")
        return 1
    shapes = compute_shapes(args, sorted(sessions))
    overview(shapes)
    if runs is not None:
        for mode in ("supervised", "time_only"):
            mode_tests(runs, shapes, mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Synthetic ground-truth benchmark of noise-calibrated adversarial training (Acorn-dprime v5).

Keep next to Acorn-dprime and dprime_exp9.py: it reuses the runner's arms, encoder training and
decoder, so every model is trained exactly as in Exp11-14. Nothing in the fork changes.

Why synthetic: on recordings the noise is only estimated. Here it is KNOWN, so robustness and
detectability are measured in the ideal observer's own units, and the theory of the note (d' =
||M^{-1/2} delta|| of the noise; training budgets referenced to the behavioural separation) is
checked against exact quantities.

Data-generating process (one session = one anisotropy level alpha and one replicate draw):
  behaviour  z_t  `latent_dim` smooth latents (white noise through a Gaussian kernel of width
                  `smooth` bins, unit variance); the labels of supervised CEBRA and decoders are z.
  signal     s_t  c * a_i * tanh(w_i . z_t + b_i) for neuron i (random unit w_i, lognormal a_i).
  noise      n_t  N(0, Sigma(alpha)), white in time. Sigma(0) = I: isotropic, d' is plain L2.
                  Sigma(alpha) has heterogeneous variances (log-sd alpha*hetero) and a rank-k shared
                  component (alpha*shared of the variance); the mean variance is 1 at every alpha.
  x_t = s_t + n_t. The amplitude c sets the behavioural separation in the TRUE noise metric,
  eps_beh_true = sqrt(2 T tr(Sigma^-1 Cov(s))), to `target_eps_beh` (default 25, the Perich
  DIFF-metric median): the ideal observer's task is equally hard at every alpha and only the
  noise geometry changes. All random draws except Sigma's shape are shared across alpha (common
  random numbers), so the alpha levels of a replicate are paired.

Every model is evaluated in three metrics (decoder R2 under d'-APGD; eps1/2 = the budget that
halves the clean R2, measured by bisection, --eval-refine):
  true       the ideal observer's metric Sigma (x) I: the primary endpoint. AUC* = Phi(eps1/2 /
             sqrt 2) is the ideal observer's detection AUC for the least detectable attack that
             halves decoding: the higher, the more detectable every damaging attack is;
  reference  Ledoit-Wolf x Welch of the whole training residuals, as on Perich (kappa);
  isotropic  sigma-scaled L2 (the trade-off).

Arms (names of dprime_exp9.catalogue(); 7 per mode):
  CLEAN; PGD_L2_EPS6 (ACORN as published, raw L2); MADRY_ISO_D8 (L2-APGD at d' 8 in noise-SD
  units: the isotropic twin of DIFF8); MADRY_SP_D8 (whole-residual metric); MADRY_SPDIFF_D8
  (noise-aware); MADRY_SPDIFF_AUTOBEH (supervised) / MADRY_SPDIFF_AUTOSIG (time-only), the
  automatic budgets; MADRY_SPDIFF_GAUSS (noise augmentation at the natural noise scale).

Pre-registered in <output-root>/PREDICTIONS.json before any training (never overwritten):
  H1 every adversarially trained arm (PGD6, ISO8, DIFF8, AUTO) is more robust than CLEAN (true);
  H2 DIFF8 / ISO8 (same budget and optimiser; noise-shaped vs isotropic) is 1 at alpha = 0
     (|log ratio| < 0.1), above 1 at the largest alpha, and grows with alpha;
  H3 the trade-off: DIFF8 / ISO8 <= 1 in the isotropic metric for alpha > 0;
  H4 whole-residual brittleness (supervised): SP8 below DIFF8 (true metric);
  H5 the adversary, not noise augmentation: AUTO above GAUSS (true metric);
  H6 (exploratory) the reference-metric kappa of each arm within +-30 % of its Perich value;
  budget suite: robustness peaks at x = eps / eps_beh(M_train) in [0.25, 0.45] and collapses
     (below half the peak) for x >= 0.5.
  H7 (isosweep, registered 2026-11-03 after the main results, see below) even the best isotropic
     budget of each session (PGD L2 1/2/3/6 and ISO d' 1/2/4/8, picked post hoc on the true metric:
     an oracle that favours the isotropic arms) stays below AUTO at alpha >= 0.5 (CI above 1), is
     equal to it at alpha = 0 (|log ratio| < 0.1), and the best isotropic budget shrinks as alpha
     grows. Motivation: in the main suite PGD6 and ISO8 lost clean R2 at alpha = 1 (0.80 and 0.90
     vs 0.93), i.e. their budget was too large in the low-noise directions; the question is whether
     a smaller isotropic budget closes the gap (calibration) or not (geometry).

Suites:
  pilot   alpha 0 and 1; replicate 0; seed 42; both modes: 28 jobs.
  main    alpha 0, 0.25, 0.5, 1; replicates 0-3; seed 42; both modes: 224 jobs.
  budget  alpha 1; replicates 0-3; seed 42; supervised DIFF budgets 4/8/16/32, SP 4/8/16: 28 jobs.
  isosweep  alpha 0, 0.5, 1; replicates 0-3; seed 42; both modes; the isotropic budgets PGD L2
          1/2/3 and ISO d' 1/2/4 (PGD6 and ISO8 come from main): 144 jobs. Run it into the same
          --output-root as main, so the sweep is paired with the main arms.

Usage (4 GPUs; re-running skips finished jobs):
  python -u synthetic_benchmark.py --suite pilot --dry-run     # sessions and checks, no training
  python -u synthetic_benchmark.py --suite pilot --gpu-workers 4
  python -u synthetic_benchmark.py --suite main --gpu-workers 4
  python -u synthetic_benchmark.py --suite budget --gpu-workers 4
  python -u synthetic_benchmark.py --suite isosweep --gpu-workers 4 --output-root <main root>
  python synthetic_benchmark.py --suite main --summarize > synth_main_report.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import dprime_exp9 as runner  # noqa: E402  arms, training and decoder of Exp11-14

import torch  # noqa: E402

WINDOW = runner.WINDOW
MAIN_ARMS = {
    "supervised": ("CLEAN", "PGD_L2_EPS6", "MADRY_ISO_D8", "MADRY_SP_D8", "MADRY_SPDIFF_D8",
                   "MADRY_SPDIFF_AUTOBEH", "MADRY_SPDIFF_GAUSS"),
    "time_only": ("CLEAN", "PGD_L2_EPS6", "MADRY_ISO_D8", "MADRY_SP_D8", "MADRY_SPDIFF_D8",
                  "MADRY_SPDIFF_AUTOSIG", "MADRY_SPDIFF_GAUSS"),
}
BUDGET_ARMS = {
    "supervised": ("MADRY_SPDIFF_D4", "MADRY_SPDIFF_D8", "MADRY_SPDIFF_D16", "MADRY_SPDIFF_D32",
                   "MADRY_SP_D4", "MADRY_SP_D8", "MADRY_SP_D16"),
}
# The isotropic geometry at several budgets: raw L2 PGD (ACORN as practised) and ISO d' (L2-APGD in
# noise-SD units, the same optimiser as the d' arms). PGD6 and ISO8 are part of the main suite.
ISO_NEW = ("PGD_L2_EPS1", "PGD_L2_EPS2", "PGD_L2_EPS3", "MADRY_ISO_D1", "MADRY_ISO_D2",
           "MADRY_ISO_D4")
ISO_SWEEP = ISO_NEW + ("PGD_L2_EPS6", "MADRY_ISO_D8")
SUITE_HYPOTHESES = {
    "isosweep": "H7: even the best isotropic budget per session (PGD L2 1/2/3/6, ISO d' 1/2/4/8; "
                "oracle picked on the true metric) stays below AUTO at alpha >= 0.5 (CI above 1), "
                "equals it at alpha = 0 (|log ratio| < 0.1), and the best isotropic budget "
                "shrinks as alpha grows.",
}
SUITES = {
    "pilot": dict(alphas=(0.0, 1.0), reps=(0,), seeds=(42,), modes=("supervised", "time_only"),
                  arms=MAIN_ARMS),
    "main": dict(alphas=(0.0, 0.25, 0.5, 1.0), reps=(0, 1, 2, 3), seeds=(42,),
                 modes=("supervised", "time_only"), arms=MAIN_ARMS),
    "budget": dict(alphas=(1.0,), reps=(0, 1, 2, 3), seeds=(42,), modes=("supervised",),
                   arms=BUDGET_ARMS),
    "isosweep": dict(alphas=(0.0, 0.5, 1.0), reps=(0, 1, 2, 3), seeds=(42,),
                     modes=("supervised", "time_only"), arms={m: ISO_NEW for m in MAIN_ARMS}),
}
# Reference-metric kappa of the 10 Perich sessions of Exp11-13 (median over sessions), frozen.
PERICH_KAPPA = {
    "supervised": {"CLEAN": 0.080, "PGD_L2_EPS6": 0.124, "MADRY_SPDIFF_D8": 0.151,
                   "MADRY_SPDIFF_AUTOBEH": 0.150, "MADRY_SPDIFF_GAUSS": 0.103},
    "time_only": {"CLEAN": 0.035, "PGD_L2_EPS6": 0.116, "MADRY_SPDIFF_D8": 0.139,
                  "MADRY_SPDIFF_AUTOSIG": 0.133, "MADRY_SPDIFF_GAUSS": 0.106},
}
AUTO = {"supervised": "MADRY_SPDIFF_AUTOBEH", "time_only": "MADRY_SPDIFF_AUTOSIG"}
FAIL_HALF = 0.5  # true-metric eps1/2 below half a JND: halved by an imperceptible perturbation


# ============================================================================ data generation
@dataclass(frozen=True)
class DGP:
    channels: int = 50
    latent_dim: int = 3
    n_train: int = 15000
    n_valid: int = 4000
    gap: int = 200
    smooth: float = 8.0
    hetero: float = 1.0
    shared: float = 0.5
    shared_rank: int = 2
    target_eps_beh: float = 25.0
    tuning_bias: float = 0.5
    gain_sd: float = 0.3
    base_seed: int = 20261102


def session_name(alpha: float, rep: int) -> str:
    return f"A{alpha:g}".replace(".", "p") + f"_R{rep}"


def _streams(dgp: DGP, rep: int):
    """Independent generators of one replicate, shared by every alpha (common random numbers)."""
    seq = np.random.SeedSequence([dgp.base_seed, rep])
    return [np.random.default_rng(s) for s in seq.spawn(4)]  # latents, tuning, geometry, noise


def smooth_latents(n: int, d: int, width: float, rng) -> np.ndarray:
    radius = int(math.ceil(4 * width))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / width) ** 2)
    kernel /= kernel.sum()
    white = rng.normal(size=(n + 2 * radius, d))
    z = np.stack([np.convolve(white[:, j], kernel, mode="valid") for j in range(d)], axis=1)
    return (z - z.mean(0)) / z.std(0)


def noise_covariance(dgp: DGP, alpha: float, rng) -> np.ndarray:
    """Sigma(alpha): heterogeneous variances and a rank-k shared part; mean variance 1."""
    c = dgp.channels
    g = rng.normal(size=c)
    g = (g - g.mean()) / g.std()
    u, _ = np.linalg.qr(rng.normal(size=(c, dgp.shared_rank)))
    var = np.exp(alpha * dgp.hetero * g)
    shared = alpha * dgp.shared
    corr = (1.0 - shared) * np.eye(c) + shared * (c / dgp.shared_rank) * (u @ u.T)
    cov = np.sqrt(var)[:, None] * corr * np.sqrt(var)[None, :]
    return cov * c / np.trace(cov)


def separation(signal: np.ndarray, spatial_cov: np.ndarray) -> float:
    """sqrt(2 T tr(Sigma^-1 Cov(s))): the behavioural separation in a temporally white metric."""
    cov = np.cov(signal, rowvar=False)
    return math.sqrt(2 * WINDOW * float(np.trace(np.linalg.solve(spatial_cov, cov))))


def generate_session(dgp: DGP, alpha: float, rep: int) -> Dict[str, Any]:
    r_lat, r_tun, r_geo, r_noise = _streams(dgp, rep)
    n = dgp.n_train + dgp.gap + dgp.n_valid
    z = smooth_latents(n, dgp.latent_dim, dgp.smooth, r_lat)
    w = r_tun.normal(size=(dgp.channels, dgp.latent_dim))
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    b = r_tun.uniform(-dgp.tuning_bias, dgp.tuning_bias, dgp.channels)
    a = np.exp(r_tun.normal(0.0, dgp.gain_sd, dgp.channels))
    unit_signal = a * np.tanh(z @ w.T + b)
    cov = noise_covariance(dgp, alpha, r_geo)
    amplitude = dgp.target_eps_beh / separation(unit_signal, cov)
    signal = amplitude * unit_signal
    noise = r_noise.normal(size=(n, dgp.channels)) @ np.linalg.cholesky(cov).T
    x = signal + noise
    train = slice(0, dgp.n_train)
    valid = slice(dgp.n_train + dgp.gap, n)
    return dict(train_data=x[train].astype(np.float32), valid_data=x[valid].astype(np.float32),
                train_label=z[train].astype(np.float32), valid_label=z[valid].astype(np.float32),
                train_signal=signal[train].astype(np.float32), true_spatial_cov=cov,
                amplitude=amplitude, eps_beh_true=separation(signal, cov))


def ensure_sessions(dgp: DGP, alphas, reps, root: Path) -> Dict[str, Dict[str, Any]]:
    """Generate (or check) every session file; the DGP is a pure function of its parameters."""
    folder = root / "sessions"
    folder.mkdir(parents=True, exist_ok=True)
    info = {}
    for alpha in alphas:
        for rep in reps:
            name = session_name(alpha, rep)
            path, meta_path = folder / f"{name}.npz", folder / f"{name}.json"
            meta = dict(session=name, alpha=alpha, rep=rep, dgp=asdict(dgp))
            if path.is_file() and meta_path.is_file():
                stored = json.loads(meta_path.read_text(encoding="utf-8"))
                if stored["dgp"] != meta["dgp"] or stored["alpha"] != alpha:
                    raise RuntimeError(f"{path} was generated with other parameters; use a new "
                                       "--output-root")
                info[name] = stored
                continue
            data = generate_session(dgp, alpha, rep)
            arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
            with open(path, "wb") as f:
                np.savez(f, **arrays)
            meta.update(amplitude=float(data["amplitude"]), eps_beh_true=float(data["eps_beh_true"]),
                        sha256=runner.file_sha256(path))
            runner.atomic_write_json(meta_path, meta)
            info[name] = meta
    return info


def load_session(root: Path, name: str):
    with np.load(root / "sessions" / f"{name}.npz", allow_pickle=False) as f:
        arrays = tuple(np.ascontiguousarray(f[k]) for k in
                       ("train_data", "valid_data", "train_label", "valid_label"))
        return arrays, np.asarray(f["true_spatial_cov"]), np.asarray(f["train_signal"])


# ============================================================================ arms
def arm(name: str) -> "runner.Experiment":
    """A catalogue arm, or MADRY_SPDIFF_D<eps> / MADRY_SP_D<eps> / MADRY_ISO_D<eps> with another
    fixed budget, or PGD_L2_EPS<eps> (raw L2 PGD, as the catalogue's PGD arms)."""
    by_name = {e.name: e for e in runner.catalogue()}
    if name in by_name:
        return by_name[name]
    if name.startswith("PGD_L2_EPS"):
        eps = float(name[len("PGD_L2_EPS"):])
        return runner.Experiment(name, {"training_mode": "adversarial",
                                        "adv_epsilon_mode": "constant", "attack_norm": "l2",
                                        "adv_epsilon": eps, "adv_alpha": eps / 5.0})
    for prefix, base in (("MADRY_SPDIFF_D", "MADRY_SPDIFF_D8"), ("MADRY_SP_D", "MADRY_SP_D8"),
                         ("MADRY_ISO_D", "MADRY_ISO_D8")):
        if name.startswith(prefix):
            eps = float(name[len(prefix):])
            template = by_name[base]
            return runner.Experiment(name, {**template.overrides, "adv_dprime_epsilon": eps},
                                     template.supervised_only)
    raise ValueError(f"Unknown arm {name!r}")


@dataclass(frozen=True)
class Job:
    session: str
    mode: str
    seed: int
    arm: str

    @property
    def ident(self) -> str:
        return f"{self.session}__{self.mode}__s{self.seed}__{self.arm}"


def plan_jobs(args) -> List[Job]:
    suite = SUITES[args.suite]
    jobs = []
    for alpha in args.alphas:
        for rep in args.reps:
            for mode in args.modes:
                arms = [a for a in (args.arms or suite["arms"][mode])
                        if mode == "supervised" or not arm(a).supervised_only]
                for seed in args.seeds:
                    jobs += [Job(session_name(alpha, rep), mode, seed, a) for a in arms]
    return jobs


# ============================================================================ checks and predictions
def session_checks(root: Path, name: str) -> Dict[str, Any]:
    """Estimated vs true metrics and separations of one session (no training; cached)."""
    path = root / "sessions" / f"{name}.checks.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    from cebra.dprime import calibration, noise
    (x, _, y, _), true_cov, signal = load_session(root, name)
    x, y = x.astype(np.float64), y.astype(np.float64)
    metrics = {"true": (true_cov, np.eye(WINDOW))}
    for key, (residuals, spatial, temporal) in {
            "diff": ("difference", "ledoit_wolf", "white"), "sp": ("whole", "ledoit_wolf", "white"),
            "iso": ("whole", "isotropic", "white"), "ref": ("whole", "ledoit_wolf", "welch")}.items():
        model, _ = noise.fit_noise_model(x, None, window_len=WINDOW, residual_mode=residuals,
                                         spatial_estimator=spatial, temporal_method=temporal,
                                         kronecker_channels=0)
        metrics[key] = (model.spatial_cov, model.temporal_cov)
    prediction, model = calibration.behaviour_prediction(x, y)
    out: Dict[str, Any] = {"session": name, "behaviour_r2": model["r2"]}
    for key, (spatial, temporal) in metrics.items():
        out[f"eps_beh_{key}"] = calibration.eps_behaviour(x, prediction, spatial, temporal)
        out[f"eps_beh_exact_{key}"] = calibration.eps_behaviour(signal, signal, spatial, temporal)
    # Spatial distortion of the estimated metrics: d'_est / d'_true of random channel patterns.
    v = np.random.default_rng(0).normal(size=(2000, x.shape[1]))
    true_norm = np.einsum("ij,ij->i", v, np.linalg.solve(true_cov, v.T).T)
    for key in ("diff", "sp", "iso"):
        est = np.einsum("ij,ij->i", v, np.linalg.solve(metrics[key][0], v.T).T)
        q = np.quantile(np.sqrt(est / true_norm), [0.05, 0.5, 0.95])
        out[f"distortion_{key}"] = [float(t) for t in q]
    runner.atomic_write_json(path, json.loads(json.dumps(out)))
    return out


def write_predictions(args, root: Path, jobs: List[Job], sessions: Dict[str, Any]) -> None:
    """Pre-register budgets, x_beh and the hypotheses; rows are only ever added."""
    from cebra.dprime import calibration, noise
    path = root / "PREDICTIONS.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "created_utc": runner.now_utc(), "hypotheses": __doc__.split("Pre-registered")[1]
        .split("Suites:")[0].strip(), "fail_half": FAIL_HALF, "perich_kappa": PERICH_KAPPA,
        "rows": [], "registrations": []}
    done = {(r["session"], r["mode"], r["arm"]) for r in stored["rows"]}
    todo = sorted({(j.session, j.mode, j.arm) for j in jobs} - done)
    if not todo:
        return
    cache: Dict[str, Any] = {}
    for session, mode, name in todo:
        if session not in cache:
            (x, _, y, _), _, signal = load_session(root, session)
            x, y = x.astype(np.float64), y.astype(np.float64)
            prediction, _ = calibration.behaviour_prediction(x, y)
            cache[session] = (x, y, signal, prediction)
        x, y, signal, prediction = cache[session]
        exp = arm(name)
        members = runner._metric_members(exp, x, y, noise)
        row = dict(session=session, mode=mode, arm=name, budget=None, x_beh=None, x_beh_exact=None,
                   kappa_perich=PERICH_KAPPA[mode].get(name),
                   eps_beh_ref=sessions[session]["checks"]["eps_beh_ref"])
        if members is not None:
            eps_sig = [calibration.eps_signal(x, a, b, runner.SIG_LAG) for a, b in members]
            eps_beh = [calibration.eps_behaviour(x, prediction, a, b) for a, b in members]
            exact = [calibration.eps_behaviour(signal, signal, a, b) for a, b in members]
            blind = calibration.eps_energy_detector(x.shape[1], WINDOW)
            budgets = runner._rule_budgets(exp, eps_sig, eps_beh, blind, x.shape[1])
            if budgets is not None:
                row.update(budget=budgets[0],
                           x_beh=float(np.exp(np.mean(np.log(np.divide(budgets, eps_beh))))),
                           x_beh_exact=float(np.exp(np.mean(np.log(np.divide(budgets, exact))))))
        if row["kappa_perich"] is not None:
            row["predicted_ref_half"] = row["kappa_perich"] * row["eps_beh_ref"]
        stored["rows"].append(row)
    stored["registrations"].append({
        "registered_utc": runner.now_utc(), "pairs": len(todo), "suite": args.suite,
        "suite_hypotheses": SUITE_HYPOTHESES.get(args.suite),
        "fork_fingerprints": runner._fingerprints(args.fork_dir, runner.FORK_EVAL_FILES)})
    runner.atomic_write_json(path, json.loads(json.dumps(stored)))
    print(f"Pre-registered {len(todo)} (session, mode, arm) predictions -> {path}", flush=True)


# ============================================================================ one job
def _ideal_metric_evaluation(evaluation, noise, true_cov, call):
    """Run evaluate_decoding with its "reference" metric replaced by the TRUE noise metric."""
    original = evaluation.common_metrics

    def ideal(x_train, window_len, labels=None, residuals="whole", device=None):
        models = original(x_train, window_len, labels, residuals, device)
        models["reference"] = noise.NoiseModel(true_cov, np.eye(window_len), device=device)
        return models

    evaluation.common_metrics = ideal
    try:
        return call()
    finally:
        evaluation.common_metrics = original


def _half(entry: Dict[str, Any]) -> Optional[float]:
    """Refined eps1/2; a censored search reports its bound (below the floor: the smallest budget
    tried, so it counts as a failure; above the grid: the largest budget)."""
    refined = entry.get("half_r2_refined")
    if refined is None:
        return entry.get("eps_at_half_r2")
    if refined["status"] == "below_floor":
        return refined["upper_bound"]
    if refined["status"] == "above_grid":
        return max(p["epsilon"] for p in entry["robust_r2"])
    return refined["epsilon"]


def run_job(args, job: Job) -> Dict[str, Any]:
    root = args.output_root
    exp = arm(job.arm)
    cebra = runner.load_fork(args.fork_dir, exp, needs_eval_options=True)
    from cebra.dprime import evaluation, noise
    arrays, true_cov, _ = load_session(root, job.session)
    x_train, x_valid, y_train, y_valid = arrays
    meta_session = json.loads((root / "sessions" / f"{job.session}.json").read_text())
    plan = runner.Plan(args.suite, job.mode, (exp,), (job.session,), (job.seed,),
                       args.max_iterations, None, None, root)
    run_root = root / "runs" / job.session / job.mode / f"seed{job.seed}"
    rjob = runner.Job(job.session, job.seed, exp, run_root)
    train_args = SimpleNamespace(batch_size=args.batch_size, device=args.device,
                                 epsilon=runner.PGD_EPS, mlp_epochs=args.mlp_epochs)
    if all(p.is_file() for p in runner._checkpoint_paths(run_root, exp.name)):
        encoder, decoder, meta, report = runner._restore_train(cebra, rjob, plan, arrays,
                                                               args.max_iterations)
    else:
        encoder, decoder, meta, report = runner._new_train(cebra, rjob, plan, arrays,
                                                           args.max_iterations, train_args)
    kwargs = dict(epsilons=args.eval_epsilons, steps=args.eval_steps,
                  batch_size=args.eval_batch_size, contrastive=False, residuals="whole",
                  labels_train=None, seed=job.seed + 200_000, refine=args.eval_refine,
                  behaviour_train=y_train, margins=False)
    if args.eval_restarts > 1:
        kwargs["restarts"] = args.eval_restarts
    est = evaluation.evaluate_decoding(encoder, decoder, x_train, x_valid, y_valid,
                                       metrics=("reference", "isotropic"), **kwargs)
    true = _ideal_metric_evaluation(evaluation, noise, true_cov, lambda: evaluation.evaluate_decoding(
        encoder, decoder, x_train, x_valid, y_valid, metrics=("reference",), **kwargs))
    for robust in (est, true):
        if abs(float(robust["clean_r2"]) - float(meta["clean_r2"])) > 5e-4:
            raise RuntimeError("Clean R2 of the evaluation differs from training: alignment error")
    folder = run_root / "robust_eval"
    runner.atomic_write_json(folder / f"{exp.name}_estimated.json", est)
    runner.atomic_write_json(folder / f"{exp.name}_true.json", true)
    epsilon = (report or {}).get("epsilon", {}) or {}
    realized = ((report or {}).get("training", {}) or {}).get("realized") or {}
    half_true = _half(true["reference"])
    record = dict(
        session=job.session, alpha=meta_session["alpha"], rep=meta_session["rep"],
        mode=job.mode, seed=job.seed, arm=job.arm, clean_r2=float(meta["clean_r2"]),
        eps_beh_true=meta_session["eps_beh_true"],
        true_half=half_true, true_status=true["reference"].get("half_r2_refined", {}).get("status"),
        true_half_grid=true["reference"]["eps_at_half_r2"],
        true_kappa_exact=(half_true / meta_session["eps_beh_true"]) if half_true else None,
        true_eps_beh_est=true["reference"].get("eps_beh"),
        true_natural_noise_r2=true["reference"]["natural_noise_r2"],
        auc_ideal=(0.5 * (1 + math.erf(half_true / 2.0))) if half_true is not None else None,
        ref_half=_half(est["reference"]), ref_status=est["reference"].get("half_r2_refined", {})
        .get("status"), ref_eps_beh=est["reference"].get("eps_beh"),
        ref_kappa=est["reference"].get("kappa"),
        iso_half=_half(est["isotropic"]), iso_status=est["isotropic"].get("half_r2_refined", {})
        .get("status"), iso_kappa=est["isotropic"].get("kappa"),
        eps_star=epsilon.get("eps_star"), eps_source=epsilon.get("source"),
        eps_binding=epsilon.get("binding"), x_train=epsilon.get("x_train"),
        loss_gain_median=(realized.get("loss_gain") or {}).get("median"),
        raw_l2_median=(realized.get("raw_l2") or {}).get("median"),
        true_curve=[[p["epsilon"], p["r2"]] for p in true["reference"]["robust_r2"]])
    return record


def worker(args) -> int:
    job = Job(**json.loads(args._job))
    begin = time.monotonic()
    base = dict(job=job.ident, session=job.session, mode=job.mode, seed=job.seed, arm=job.arm,
                status="error", started_utc=runner.now_utc(), error="")
    try:
        torch.set_num_threads(args.cpu_threads)
        base.update(run_job(args, job), status="success")
        print(f"{job.ident}: clean R2 {base['clean_r2']:.4f} true eps1/2 {base['true_half']}",
              flush=True)
    except BaseException as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        base["elapsed_seconds"] = round(time.monotonic() - begin, 1)
        base["finished_utc"] = runner.now_utc()
        runner.atomic_write_json(args.output_root / "records" / f"{job.ident}.json",
                                 json.loads(json.dumps(base)))
    return 0 if base["status"] == "success" else 1


# ============================================================================ coordinator
def worker_command(args, job: Job) -> List[str]:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker",
           "--_job", json.dumps(job.__dict__), "--suite", args.suite,
           "--output-root", str(args.output_root), "--fork-dir", str(args.fork_dir),
           "--max-iterations", str(args.max_iterations), "--batch-size", str(args.batch_size),
           "--mlp-epochs", str(args.mlp_epochs), "--device", args.device,
           "--cpu-threads", str(args.cpu_threads), "--eval-steps", str(args.eval_steps),
           "--eval-batch-size", str(args.eval_batch_size), "--eval-refine", str(args.eval_refine),
           "--eval-restarts", str(args.eval_restarts),
           "--eval-epsilons", *[str(e) for e in args.eval_epsilons]]
    return cmd


def finished(root: Path, job: Job) -> bool:
    path = root / "records" / f"{job.ident}.json"
    return path.is_file() and json.loads(path.read_text()).get("status") == "success"


def orchestrate(args, jobs: List[Job]) -> int:
    pending = [j for j in jobs if not finished(args.output_root, j)]
    print(f"{len(pending)}/{len(jobs)} jobs to run on {max(args.gpu_workers, 1)} worker(s); "
          f"output={args.output_root}", flush=True)
    if not pending:
        return 0
    tokens = runner.gpu_tokens(args.gpu_workers) if args.gpu_workers > 0 else [""]
    todo: "queue.Queue[Job]" = queue.Queue()
    for job in pending:
        todo.put(job)
    lock, failures = threading.Lock(), []
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)

    def loop(slot: int, token: str):
        while True:
            try:
                job = todo.get_nowait()
            except queue.Empty:
                return
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=token)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env[key] = str(args.cpu_threads)
            log = args.output_root / "logs" / f"{job.ident}.log"
            with lock:
                print(f"[{slot}] START {job.ident}", flush=True)
            with log.open("w") as f:
                result = subprocess.run(worker_command(args, job), env=env, stdout=f,
                                        stderr=subprocess.STDOUT, check=False)
            ok = result.returncode == 0 and finished(args.output_root, job)
            with lock:
                if not ok:
                    failures.append(job.ident)
                print(f"[{slot}] {'DONE' if ok else 'FAILED'} {job.ident}"
                      + ("" if ok else f" (see {log})"), flush=True)

    threads = [threading.Thread(target=loop, args=(i, t)) for i, t in enumerate(tokens)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"Finished: {len(pending) - len(failures)}/{len(pending)} successful", flush=True)
    return 1 if failures else 0


# ============================================================================ summary
def _session_ratio(d, mode: str, a: str, b: str, column: str):
    """Paired log ratios a / b per session (seeds averaged) and their bootstrap CI."""
    g = d[d["mode"] == mode]
    piv = g.pivot_table(index=["session", "seed"], columns="arm", values=column)
    if a not in piv or b not in piv:
        return None
    lr = np.log(piv[a] / piv[b]).replace([np.inf, -np.inf], np.nan).dropna()
    if lr.empty:
        return None
    per_session = lr.groupby(level=0).mean()
    rng = np.random.default_rng(2026)
    v = per_session.to_numpy()
    boot = v[rng.integers(0, len(v), size=(20000, len(v)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(ratio=float(np.exp(v.mean())), lo=float(np.exp(lo)), hi=float(np.exp(hi)),
                sessions_up=int((v > 0).sum()), sessions=len(v))


def _verdict(r, expect_up: bool = True) -> str:
    if r is None:
        return "no data"
    up = r["lo"] > 1 and r["sessions_up"] >= 0.75 * r["sessions"]
    down = r["hi"] < 1 and r["sessions_up"] <= 0.25 * r["sessions"]
    if (up and expect_up) or (down and not expect_up):
        return "SUPPORTED"
    if (down and expect_up) or (up and not expect_up):
        return "CONTRADICTED"
    return "inconclusive"


def summarize(args) -> int:
    import pandas as pd
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 400)
    root = args.output_root
    rows = [json.loads(p.read_text()) for p in sorted((root / "records").glob("*.json"))]
    if not rows:
        print(f"No records in {root}")
        return 1
    d = pd.DataFrame(rows)
    errors = d[d.status != "success"]
    d = d[d.status == "success"].copy()
    d.drop(columns=["true_curve"], errors="ignore").to_csv(root / "SYNTH_RUNS.csv", index=False)
    print(f"{root}: {len(d)} successful runs, {len(errors)} errors; sessions "
          f"{d.session.nunique()}, seeds {sorted(d.seed.unique())}")
    checks = []
    for name in sorted(d.session.unique()):
        c = session_checks(root, name)
        meta = json.loads((root / "sessions" / f"{name}.json").read_text())
        checks.append(dict(session=name, alpha=meta["alpha"], amplitude=meta["amplitude"],
                           eps_beh_true=meta["eps_beh_true"], beh_r2=c["behaviour_r2"],
                           eps_beh_diff=c["eps_beh_diff"], eps_beh_sp=c["eps_beh_sp"],
                           eps_beh_ref=c["eps_beh_ref"], eps_beh_true_est=c["eps_beh_true"],
                           dist_diff=c["distortion_diff"][1], dist_sp=c["distortion_sp"][1],
                           dist_iso=c["distortion_iso"][1]))
    print("\n0  sessions: separations (estimated in each metric; true = exact) and the median "
          "d'_est / d'_true of random patterns")
    print(pd.DataFrame(checks).round(3).to_string(index=False))
    d["fail"] = d.true_half < FAIL_HALF
    shorts = {"CLEAN": "CLEAN", "PGD_L2_EPS6": "PGD6", "MADRY_ISO_D8": "ISO8", "MADRY_SP_D8": "SP8",
              "MADRY_SPDIFF_D8": "DIFF8", "MADRY_SPDIFF_AUTOBEH": "AUTO_BEH",
              "MADRY_SPDIFF_AUTOSIG": "AUTO_SIG", "MADRY_SPDIFF_GAUSS": "GAUSS"}
    verdicts = []
    for mode in ("supervised", "time_only"):
        g = d[d["mode"] == mode]
        if g.empty:
            continue
        print(f"\n==================== {mode}")
        table = g.groupby(["alpha", "arm"]).agg(
            runs=("clean_r2", "size"), clean_r2=("clean_r2", "mean"),
            true_half=("true_half", "median"), auc_ideal=("auc_ideal", "median"),
            kappa_true=("true_kappa_exact", "median"), ref_half=("ref_half", "median"),
            ref_kappa=("ref_kappa", "median"), iso_half=("iso_half", "median"),
            fails=("fail", "sum"), budget=("eps_star", "median"),
            loss_gain=("loss_gain_median", "median"))
        print("1  per arm (true = ideal-observer metric; auc_ideal = Phi(true eps1/2 / sqrt 2))")
        print(table.round(3).to_string())
        comparisons = [("MADRY_SPDIFF_D8", "MADRY_ISO_D8", "true_half"),
                       ("MADRY_SPDIFF_D8", "MADRY_ISO_D8", "iso_half"),
                       (AUTO[mode], "PGD_L2_EPS6", "true_half"),
                       (AUTO[mode], "MADRY_ISO_D8", "true_half"),
                       (AUTO[mode], "MADRY_SPDIFF_GAUSS", "true_half"),
                       ("MADRY_SP_D8", "MADRY_SPDIFF_D8", "true_half")]
        comparisons += [(a, "CLEAN", "true_half") for a in
                        ("PGD_L2_EPS6", "MADRY_ISO_D8", "MADRY_SPDIFF_D8", AUTO[mode])]
        print("\n2  paired ratios (sessions = independent data draws; CI = session bootstrap)")
        for alpha in sorted(g.alpha.unique()):
            ga = g[g.alpha == alpha]
            for a, b, column in comparisons:
                r = _session_ratio(ga, mode, a, b, column)
                if r is not None:
                    print(f"  alpha {alpha:<4g} {shorts.get(a, a):>8} / {shorts.get(b, b):<6} "
                          f"{column:9s} {r['ratio']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] "
                          f"sessions up {r['sessions_up']}/{r['sessions']}")
        alphas = sorted(g.alpha.unique())
        for arm_name in ("PGD_L2_EPS6", "MADRY_ISO_D8", "MADRY_SPDIFF_D8", AUTO[mode]):
            r = _session_ratio(g, mode, arm_name, "CLEAN", "true_half")
            verdicts.append((f"H1 {mode} {shorts[arm_name]} > CLEAN (all alpha)", _verdict(r), r))
        r0 = _session_ratio(g[g.alpha == alphas[0]], mode, "MADRY_SPDIFF_D8", "MADRY_ISO_D8",
                            "true_half")
        band = (math.exp(-0.1), math.exp(0.1))
        if alphas[0] != 0:
            control = "no alpha 0"
        elif r0 is None:
            control = "no data"
        elif band[0] < r0["ratio"] < band[1]:
            control = "SUPPORTED"
        elif r0["lo"] > band[1] or r0["hi"] < band[0]:
            control = "CONTRADICTED"
        else:
            control = "inconclusive"
        verdicts.append((f"H2 {mode} DIFF8/ISO8 = 1 at alpha 0 (control)", control, r0))
        if len(alphas) > 1:
            r1 = _session_ratio(g[g.alpha == alphas[-1]], mode, "MADRY_SPDIFF_D8", "MADRY_ISO_D8",
                                "true_half")
            verdicts.append((f"H2 {mode} DIFF8/ISO8 > 1 at alpha {alphas[-1]:g}", _verdict(r1), r1))
            trend = [_session_ratio(g[g.alpha == a], mode, "MADRY_SPDIFF_D8", "MADRY_ISO_D8",
                                    "true_half") for a in alphas]
            if all(t is not None for t in trend):
                ratios = [t["ratio"] for t in trend]
                rising = all(b >= 0.98 * a for a, b in zip(ratios, ratios[1:])) and ratios[-1] > ratios[0]
                verdicts.append((f"H2 {mode} DIFF8/ISO8 grows with alpha: "
                                 + ", ".join(f"{a:g}: {r:.3f}" for a, r in zip(alphas, ratios)),
                                 "SUPPORTED" if rising else "not supported", None))
            r3 = _session_ratio(g[g.alpha > 0], mode, "MADRY_SPDIFF_D8", "MADRY_ISO_D8", "iso_half")
            verdicts.append((f"H3 {mode} DIFF8/ISO8 <= 1 in the isotropic metric (alpha > 0)",
                             _verdict(r3, expect_up=False), r3))
        if mode == "supervised":
            r4 = _session_ratio(g, mode, "MADRY_SP_D8", "MADRY_SPDIFF_D8", "true_half")
            verdicts.append(("H4 supervised SP8 < DIFF8", _verdict(r4, expect_up=False), r4))
        r5 = _session_ratio(g, mode, AUTO[mode], "MADRY_SPDIFF_GAUSS", "true_half")
        verdicts.append((f"H5 {mode} AUTO > GAUSS", _verdict(r5), r5))
        for arm_name, kappa in PERICH_KAPPA[mode].items():
            observed = g[g.arm == arm_name].ref_kappa.median()
            if np.isfinite(observed):
                ok = abs(observed / kappa - 1) <= 0.3
                verdicts.append((f"H6 {mode} {shorts[arm_name]} ref kappa {observed:.3f} vs Perich "
                                 f"{kappa:.3f}", "SUPPORTED" if ok else "not supported", None))
    budget = d[d.arm.str.match(r"MADRY_(SPDIFF|SP)_D\d") & (d["mode"] == "supervised")]
    if budget.arm.nunique() > 2 and (root / "PREDICTIONS.json").is_file():
        pred = pd.DataFrame(json.loads((root / "PREDICTIONS.json").read_text())["rows"])
        b = budget.merge(pred[["session", "mode", "arm", "x_beh", "x_beh_exact"]],
                         on=["session", "mode", "arm"], how="left")
        b["metric"] = np.where(b.arm.str.startswith("MADRY_SPDIFF"), "DIFF", "SP")
        peak = b.groupby(["session", "seed"]).true_half.transform("max")
        b["rel"] = b.true_half / peak
        print("\n3  budget: true eps1/2 relative to the session's best budget vs x_beh")
        print(b.groupby(["metric", "arm"]).agg(x_beh=("x_beh", "median"),
                                               x_exact=("x_beh_exact", "median"),
                                               true_half=("true_half", "median"),
                                               rel=("rel", "median"), clean_r2=("clean_r2", "mean"),
                                               fails=("fail", "sum")).round(3).to_string())
        best = b.loc[b.groupby(["session", "seed"]).true_half.idxmax()]
        print(f"  best x_beh per session: {best.x_beh.round(3).tolist()} (predicted in [0.25, 0.45])")
        beyond = b[b.x_beh >= 0.5]
        if len(beyond):
            print(f"  x_beh >= 0.5: {int((beyond.rel < 0.5).sum())}/{len(beyond)} runs below half "
                  "the peak (predicted: all)")
    iso_sweep_section(d, verdicts)
    if verdicts:
        print("\n4  pre-registered hypotheses")
        for label, verdict, r in verdicts:
            detail = (f"  ratio {r['ratio']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}], sessions "
                      f"{r['sessions_up']}/{r['sessions']}") if r else ""
            print(f"  {verdict:>13}  {label}{detail}")
    if len(errors):
        print("\nerrors:")
        for _, r in errors.iterrows():
            print(f"  {r.job}: {str(r.error)[:200]}")
    return 0


def _budget_of(name: str) -> float:
    return float(name.replace("PGD_L2_EPS", "").replace("MADRY_ISO_D", ""))


def iso_sweep_section(d, verdicts) -> None:
    """Section 5: every isotropic budget, the best one per session (an oracle) vs DIFF8 / AUTO."""
    import pandas as pd
    if d[d.arm.isin(ISO_NEW)].empty:
        return
    print("\n5  isotropic budget sweep (PGD = raw L2 PGD, ISO = L2-APGD in noise-SD units); "
          "ORACLE_ISO = the best isotropic arm of each session, picked on the true metric")
    for mode in ("supervised", "time_only"):
        g = d[d["mode"] == mode]
        iso = g[g.arm.isin(ISO_SWEEP)]
        if iso.arm.nunique() <= 2:
            continue
        keep = g[g.arm.isin(ISO_SWEEP + ("MADRY_SPDIFF_D8", AUTO[mode]))]
        for column, label in (("true_half", "true eps1/2 (median)"), ("clean_r2", "clean R2 (mean)"),
                              ("iso_half", "isotropic-metric eps1/2 (median)")):
            table = keep.pivot_table(index="arm", columns="alpha", values=column,
                                     aggfunc="median" if column != "clean_r2" else "mean")
            print(f"\n  {mode}: {label}")
            print(table.round(3).to_string())
        best = iso.loc[iso.groupby(["alpha", "session", "seed"]).true_half.idxmax()].copy()
        print(f"\n  {mode}: best isotropic arm per session:")
        for alpha, arms in best.groupby("alpha").arm:
            print(f"    alpha {alpha:g}: {arms.value_counts().to_dict()}")
        best["arm"] = "ORACLE_ISO"
        both = pd.concat([g, best], ignore_index=True)
        winners = {}
        for alpha in sorted(best.alpha.unique()):
            ga = both[both.alpha == alpha]
            fixed = iso[iso.alpha == alpha].groupby("arm").true_half.median().idxmax()
            winners[alpha] = _budget_of(fixed)
            for a in (AUTO[mode], "MADRY_SPDIFF_D8"):
                for b in ("ORACLE_ISO", fixed):
                    for column in ("true_half", "iso_half", "clean_r2"):
                        r = (_session_ratio(ga, mode, a, b, column) if column != "clean_r2"
                             else None)
                        if column == "clean_r2":
                            piv = ga.pivot_table(index=["session", "seed"], columns="arm",
                                                 values="clean_r2")
                            if a in piv and b in piv:
                                print(f"    alpha {alpha:<4g} {a:>21} - {b:<21} clean R2 "
                                      f"{(piv[a] - piv[b]).mean():+.4f}")
                        elif r is not None:
                            print(f"    alpha {alpha:<4g} {a:>21} / {b:<21} {column:9s} "
                                  f"{r['ratio']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] sessions up "
                                  f"{r['sessions_up']}/{r['sessions']}")
            r = _session_ratio(ga, mode, AUTO[mode], "ORACLE_ISO", "true_half")
            if r is None:
                verdict = "no data"
            elif alpha == 0:
                verdict = ("SUPPORTED" if abs(math.log(r["ratio"])) < 0.1 else
                           "CONTRADICTED" if (r["lo"] > math.exp(0.1) or r["hi"] < math.exp(-0.1))
                           else "inconclusive")
            else:
                verdict = _verdict(r)
            relation = "=" if alpha == 0 else ">"
            verdicts.append((f"H7 {mode} AUTO {relation} best isotropic budget (oracle) at alpha "
                             f"{alpha:g}", verdict, r))
        alphas = sorted(winners)
        if len(alphas) > 1:
            shrinks = winners[alphas[-1]] < winners[alphas[0]]
            verdicts.append((f"H7 {mode} best fixed isotropic budget shrinks with alpha: "
                             + ", ".join(f"{a:g}: {winners[a]:g}" for a in alphas),
                             "SUPPORTED" if shrinks else "not supported", None))


# ============================================================================ CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", choices=tuple(SUITES), default="pilot")
    p.add_argument("--alphas", nargs="+", type=float, default=None)
    p.add_argument("--reps", nargs="+", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--modes", nargs="+", choices=("supervised", "time_only"), default=None)
    p.add_argument("--arms", nargs="+", default=None, help="Overrides the suite's arms")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--fork-dir", type=Path, default=runner.DEFAULT_FORK_DIR)
    for field_name, value in asdict(DGP()).items():
        p.add_argument("--" + field_name.replace("_", "-"), type=type(value), default=value,
                       help=f"DGP (default {value})")
    p.add_argument("--max-iterations", type=int, default=runner.MAX_ITER)
    p.add_argument("--batch-size", type=int, default=runner.BATCH_SIZE)
    p.add_argument("--mlp-epochs", type=int, default=runner.MLP_EPOCHS)
    p.add_argument("--device", default=runner.DEVICE)
    p.add_argument("--eval-epsilons", nargs="+", type=float,
                   default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--eval-refine", type=int, default=4)
    p.add_argument("--eval-restarts", type=int, default=1)
    p.add_argument("--gpu-workers", type=int, default=2, help="0: one CPU worker (smoke tests)")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--dry-run", action="store_true", help="Sessions, checks and predictions only")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_job", default=None, help=argparse.SUPPRESS)
    return p


def main() -> int:
    args = build_parser().parse_args()
    suite = SUITES[args.suite]
    args.alphas = tuple(args.alphas) if args.alphas else suite["alphas"]
    args.reps = tuple(args.reps) if args.reps else suite["reps"]
    args.seeds = tuple(args.seeds) if args.seeds else suite["seeds"]
    args.modes = tuple(args.modes) if args.modes else suite["modes"]
    args.output_root = (args.output_root or ROOT / f"SYNTH_{args.suite.upper()}").resolve()
    args.fork_dir = args.fork_dir.resolve()
    if args._worker:
        return worker(args)
    if args.summarize:
        runner.load_fork(args.fork_dir)
        return summarize(args)
    dgp = DGP(**{k: getattr(args, k) for k in asdict(DGP())})
    jobs = plan_jobs(args)
    for job in jobs:
        arm(job.arm)  # unknown arms fail before anything is written
    runner.load_fork(args.fork_dir)
    sessions = ensure_sessions(dgp, args.alphas, args.reps, args.output_root)
    for name, meta in sessions.items():
        meta["checks"] = session_checks(args.output_root, name)
        c = meta["checks"]
        print(f"{name}: amplitude {meta['amplitude']:.3f}  eps_beh true {meta['eps_beh_true']:.2f}"
              f" | estimated: true-metric {c['eps_beh_true']:.2f} DIFF {c['eps_beh_diff']:.2f} "
              f"SP {c['eps_beh_sp']:.2f} ref {c['eps_beh_ref']:.2f} | d'_est/d'_true median DIFF "
              f"{c['distortion_diff'][1]:.3f} SP {c['distortion_sp'][1]:.3f} ISO "
              f"{c['distortion_iso'][1]:.3f} | behaviour R2 {c['behaviour_r2']:.3f}", flush=True)
    write_predictions(args, args.output_root, jobs, sessions)
    print(f"{len(jobs)} jobs: {len(sessions)} sessions x {len(args.modes)} modes x "
          f"{len(args.seeds)} seeds x arms", flush=True)
    if args.dry_run:
        return 0
    code = orchestrate(args, jobs)
    summarize(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())







# #!/usr/bin/env python3
# """Synthetic ground-truth benchmark of noise-calibrated adversarial training (Acorn-dprime v5).

# Keep next to Acorn-dprime and dprime_exp9.py: it reuses the runner's arms, encoder training and
# decoder, so every model is trained exactly as in Exp11-14. Nothing in the fork changes.

# Why synthetic: on recordings the noise is only estimated. Here it is KNOWN, so robustness and
# detectability are measured in the ideal observer's own units, and the theory of the note (d' =
# ||M^{-1/2} delta|| of the noise; training budgets referenced to the behavioural separation) is
# checked against exact quantities.

# Data-generating process (one session = one anisotropy level alpha and one replicate draw):
#   behaviour  z_t  `latent_dim` smooth latents (white noise through a Gaussian kernel of width
#                   `smooth` bins, unit variance); the labels of supervised CEBRA and decoders are z.
#   signal     s_t  c * a_i * tanh(w_i . z_t + b_i) for neuron i (random unit w_i, lognormal a_i).
#   noise      n_t  N(0, Sigma(alpha)), white in time. Sigma(0) = I: isotropic, d' is plain L2.
#                   Sigma(alpha) has heterogeneous variances (log-sd alpha*hetero) and a rank-k shared
#                   component (alpha*shared of the variance); the mean variance is 1 at every alpha.
#   x_t = s_t + n_t. The amplitude c sets the behavioural separation in the TRUE noise metric,
#   eps_beh_true = sqrt(2 T tr(Sigma^-1 Cov(s))), to `target_eps_beh` (default 25, the Perich
#   DIFF-metric median): the ideal observer's task is equally hard at every alpha and only the
#   noise geometry changes. All random draws except Sigma's shape are shared across alpha (common
#   random numbers), so the alpha levels of a replicate are paired.

# Every model is evaluated in three metrics (decoder R2 under d'-APGD; eps1/2 = the budget that
# halves the clean R2, measured by bisection, --eval-refine):
#   true       the ideal observer's metric Sigma (x) I: the primary endpoint. AUC* = Phi(eps1/2 /
#              sqrt 2) is the ideal observer's detection AUC for the least detectable attack that
#              halves decoding: the higher, the more detectable every damaging attack is;
#   reference  Ledoit-Wolf x Welch of the whole training residuals, as on Perich (kappa);
#   isotropic  sigma-scaled L2 (the trade-off).

# Arms (names of dprime_exp9.catalogue(); 7 per mode):
#   CLEAN; PGD_L2_EPS6 (ACORN as published, raw L2); MADRY_ISO_D8 (L2-APGD at d' 8 in noise-SD
#   units: the isotropic twin of DIFF8); MADRY_SP_D8 (whole-residual metric); MADRY_SPDIFF_D8
#   (noise-aware); MADRY_SPDIFF_AUTOBEH (supervised) / MADRY_SPDIFF_AUTOSIG (time-only), the
#   automatic budgets; MADRY_SPDIFF_GAUSS (noise augmentation at the natural noise scale).

# Pre-registered in <output-root>/PREDICTIONS.json before any training (never overwritten):
#   H1 every adversarially trained arm (PGD6, ISO8, DIFF8, AUTO) is more robust than CLEAN (true);
#   H2 DIFF8 / ISO8 (same budget and optimiser; noise-shaped vs isotropic) is 1 at alpha = 0
#      (|log ratio| < 0.1), above 1 at the largest alpha, and grows with alpha;
#   H3 the trade-off: DIFF8 / ISO8 <= 1 in the isotropic metric for alpha > 0;
#   H4 whole-residual brittleness (supervised): SP8 below DIFF8 (true metric);
#   H5 the adversary, not noise augmentation: AUTO above GAUSS (true metric);
#   H6 (exploratory) the reference-metric kappa of each arm within +-30 % of its Perich value;
#   budget suite: robustness peaks at x = eps / eps_beh(M_train) in [0.25, 0.45] and collapses
#      (below half the peak) for x >= 0.5.

# Suites:
#   pilot   alpha 0 and 1; replicate 0; seed 42; both modes: 28 jobs.
#   main    alpha 0, 0.25, 0.5, 1; replicates 0-3; seed 42; both modes: 224 jobs.
#   budget  alpha 1; replicates 0-3; seed 42; supervised DIFF budgets 4/8/16/32, SP 4/8/16: 28 jobs.

# Usage (4 GPUs; re-running skips finished jobs):
#   python -u synthetic_benchmark.py --suite pilot --dry-run     # sessions and checks, no training
#   python -u synthetic_benchmark.py --suite pilot --gpu-workers 4
#   python -u synthetic_benchmark.py --suite main --gpu-workers 4
#   python -u synthetic_benchmark.py --suite budget --gpu-workers 4
#   python synthetic_benchmark.py --suite main --summarize > synth_main_report.txt
# """

# from __future__ import annotations

# import argparse
# import json
# import math
# import os
# import queue
# import subprocess
# import sys
# import threading
# import time
# import traceback
# from dataclasses import asdict, dataclass
# from pathlib import Path
# from types import SimpleNamespace
# from typing import Any, Dict, List, Optional

# import numpy as np

# ROOT = Path(__file__).resolve().parent
# sys.path.insert(0, str(ROOT))
# import dprime_exp9 as runner  # noqa: E402  arms, training and decoder of Exp11-14

# import torch  # noqa: E402

# WINDOW = runner.WINDOW
# MAIN_ARMS = {
#     "supervised": ("CLEAN", "PGD_L2_EPS6", "MADRY_ISO_D8", "MADRY_SP_D8", "MADRY_SPDIFF_D8",
#                    "MADRY_SPDIFF_AUTOBEH", "MADRY_SPDIFF_GAUSS"),
#     "time_only": ("CLEAN", "PGD_L2_EPS6", "MADRY_ISO_D8", "MADRY_SP_D8", "MADRY_SPDIFF_D8",
#                   "MADRY_SPDIFF_AUTOSIG", "MADRY_SPDIFF_GAUSS"),
# }
# BUDGET_ARMS = {
#     "supervised": ("MADRY_SPDIFF_D4", "MADRY_SPDIFF_D8", "MADRY_SPDIFF_D16", "MADRY_SPDIFF_D32",
#                    "MADRY_SP_D4", "MADRY_SP_D8", "MADRY_SP_D16"),
# }
# SUITES = {
#     "pilot": dict(alphas=(0.0, 1.0), reps=(0,), seeds=(42,), modes=("supervised", "time_only"),
#                   arms=MAIN_ARMS),
#     "main": dict(alphas=(0.0, 0.25, 0.5, 1.0), reps=(0, 1, 2, 3), seeds=(42,),
#                  modes=("supervised", "time_only"), arms=MAIN_ARMS),
#     "budget": dict(alphas=(1.0,), reps=(0, 1, 2, 3), seeds=(42,), modes=("supervised",),
#                    arms=BUDGET_ARMS),
# }
# # Reference-metric kappa of the 10 Perich sessions of Exp11-13 (median over sessions), frozen.
# PERICH_KAPPA = {
#     "supervised": {"CLEAN": 0.080, "PGD_L2_EPS6": 0.124, "MADRY_SPDIFF_D8": 0.151,
#                    "MADRY_SPDIFF_AUTOBEH": 0.150, "MADRY_SPDIFF_GAUSS": 0.103},
#     "time_only": {"CLEAN": 0.035, "PGD_L2_EPS6": 0.116, "MADRY_SPDIFF_D8": 0.139,
#                   "MADRY_SPDIFF_AUTOSIG": 0.133, "MADRY_SPDIFF_GAUSS": 0.106},
# }
# AUTO = {"supervised": "MADRY_SPDIFF_AUTOBEH", "time_only": "MADRY_SPDIFF_AUTOSIG"}
# FAIL_HALF = 0.5  # true-metric eps1/2 below half a JND: halved by an imperceptible perturbation


# # ============================================================================ data generation
# @dataclass(frozen=True)
# class DGP:
#     channels: int = 50
#     latent_dim: int = 3
#     n_train: int = 15000
#     n_valid: int = 4000
#     gap: int = 200
#     smooth: float = 8.0
#     hetero: float = 1.0
#     shared: float = 0.5
#     shared_rank: int = 2
#     target_eps_beh: float = 25.0
#     tuning_bias: float = 0.5
#     gain_sd: float = 0.3
#     base_seed: int = 20261102


# def session_name(alpha: float, rep: int) -> str:
#     return f"A{alpha:g}".replace(".", "p") + f"_R{rep}"


# def _streams(dgp: DGP, rep: int):
#     """Independent generators of one replicate, shared by every alpha (common random numbers)."""
#     seq = np.random.SeedSequence([dgp.base_seed, rep])
#     return [np.random.default_rng(s) for s in seq.spawn(4)]  # latents, tuning, geometry, noise


# def smooth_latents(n: int, d: int, width: float, rng) -> np.ndarray:
#     radius = int(math.ceil(4 * width))
#     kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / width) ** 2)
#     kernel /= kernel.sum()
#     white = rng.normal(size=(n + 2 * radius, d))
#     z = np.stack([np.convolve(white[:, j], kernel, mode="valid") for j in range(d)], axis=1)
#     return (z - z.mean(0)) / z.std(0)


# def noise_covariance(dgp: DGP, alpha: float, rng) -> np.ndarray:
#     """Sigma(alpha): heterogeneous variances and a rank-k shared part; mean variance 1."""
#     c = dgp.channels
#     g = rng.normal(size=c)
#     g = (g - g.mean()) / g.std()
#     u, _ = np.linalg.qr(rng.normal(size=(c, dgp.shared_rank)))
#     var = np.exp(alpha * dgp.hetero * g)
#     shared = alpha * dgp.shared
#     corr = (1.0 - shared) * np.eye(c) + shared * (c / dgp.shared_rank) * (u @ u.T)
#     cov = np.sqrt(var)[:, None] * corr * np.sqrt(var)[None, :]
#     return cov * c / np.trace(cov)


# def separation(signal: np.ndarray, spatial_cov: np.ndarray) -> float:
#     """sqrt(2 T tr(Sigma^-1 Cov(s))): the behavioural separation in a temporally white metric."""
#     cov = np.cov(signal, rowvar=False)
#     return math.sqrt(2 * WINDOW * float(np.trace(np.linalg.solve(spatial_cov, cov))))


# def generate_session(dgp: DGP, alpha: float, rep: int) -> Dict[str, Any]:
#     r_lat, r_tun, r_geo, r_noise = _streams(dgp, rep)
#     n = dgp.n_train + dgp.gap + dgp.n_valid
#     z = smooth_latents(n, dgp.latent_dim, dgp.smooth, r_lat)
#     w = r_tun.normal(size=(dgp.channels, dgp.latent_dim))
#     w /= np.linalg.norm(w, axis=1, keepdims=True)
#     b = r_tun.uniform(-dgp.tuning_bias, dgp.tuning_bias, dgp.channels)
#     a = np.exp(r_tun.normal(0.0, dgp.gain_sd, dgp.channels))
#     unit_signal = a * np.tanh(z @ w.T + b)
#     cov = noise_covariance(dgp, alpha, r_geo)
#     amplitude = dgp.target_eps_beh / separation(unit_signal, cov)
#     signal = amplitude * unit_signal
#     noise = r_noise.normal(size=(n, dgp.channels)) @ np.linalg.cholesky(cov).T
#     x = signal + noise
#     train = slice(0, dgp.n_train)
#     valid = slice(dgp.n_train + dgp.gap, n)
#     return dict(train_data=x[train].astype(np.float32), valid_data=x[valid].astype(np.float32),
#                 train_label=z[train].astype(np.float32), valid_label=z[valid].astype(np.float32),
#                 train_signal=signal[train].astype(np.float32), true_spatial_cov=cov,
#                 amplitude=amplitude, eps_beh_true=separation(signal, cov))


# def ensure_sessions(dgp: DGP, alphas, reps, root: Path) -> Dict[str, Dict[str, Any]]:
#     """Generate (or check) every session file; the DGP is a pure function of its parameters."""
#     folder = root / "sessions"
#     folder.mkdir(parents=True, exist_ok=True)
#     info = {}
#     for alpha in alphas:
#         for rep in reps:
#             name = session_name(alpha, rep)
#             path, meta_path = folder / f"{name}.npz", folder / f"{name}.json"
#             meta = dict(session=name, alpha=alpha, rep=rep, dgp=asdict(dgp))
#             if path.is_file() and meta_path.is_file():
#                 stored = json.loads(meta_path.read_text(encoding="utf-8"))
#                 if stored["dgp"] != meta["dgp"] or stored["alpha"] != alpha:
#                     raise RuntimeError(f"{path} was generated with other parameters; use a new "
#                                        "--output-root")
#                 info[name] = stored
#                 continue
#             data = generate_session(dgp, alpha, rep)
#             arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
#             with open(path, "wb") as f:
#                 np.savez(f, **arrays)
#             meta.update(amplitude=float(data["amplitude"]), eps_beh_true=float(data["eps_beh_true"]),
#                         sha256=runner.file_sha256(path))
#             runner.atomic_write_json(meta_path, meta)
#             info[name] = meta
#     return info


# def load_session(root: Path, name: str):
#     with np.load(root / "sessions" / f"{name}.npz", allow_pickle=False) as f:
#         arrays = tuple(np.ascontiguousarray(f[k]) for k in
#                        ("train_data", "valid_data", "train_label", "valid_label"))
#         return arrays, np.asarray(f["true_spatial_cov"]), np.asarray(f["train_signal"])


# # ============================================================================ arms
# def arm(name: str) -> "runner.Experiment":
#     """A catalogue arm, or MADRY_SPDIFF_D<eps> / MADRY_SP_D<eps> with another fixed budget."""
#     by_name = {e.name: e for e in runner.catalogue()}
#     if name in by_name:
#         return by_name[name]
#     for prefix, base in (("MADRY_SPDIFF_D", "MADRY_SPDIFF_D8"), ("MADRY_SP_D", "MADRY_SP_D8")):
#         if name.startswith(prefix):
#             eps = float(name[len(prefix):])
#             template = by_name[base]
#             return runner.Experiment(name, {**template.overrides, "adv_dprime_epsilon": eps},
#                                      template.supervised_only)
#     raise ValueError(f"Unknown arm {name!r}")


# @dataclass(frozen=True)
# class Job:
#     session: str
#     mode: str
#     seed: int
#     arm: str

#     @property
#     def ident(self) -> str:
#         return f"{self.session}__{self.mode}__s{self.seed}__{self.arm}"


# def plan_jobs(args) -> List[Job]:
#     suite = SUITES[args.suite]
#     jobs = []
#     for alpha in args.alphas:
#         for rep in args.reps:
#             for mode in args.modes:
#                 arms = [a for a in (args.arms or suite["arms"][mode])
#                         if mode == "supervised" or not arm(a).supervised_only]
#                 for seed in args.seeds:
#                     jobs += [Job(session_name(alpha, rep), mode, seed, a) for a in arms]
#     return jobs


# # ============================================================================ checks and predictions
# def session_checks(root: Path, name: str) -> Dict[str, Any]:
#     """Estimated vs true metrics and separations of one session (no training; cached)."""
#     path = root / "sessions" / f"{name}.checks.json"
#     if path.is_file():
#         return json.loads(path.read_text(encoding="utf-8"))
#     from cebra.dprime import calibration, noise
#     (x, _, y, _), true_cov, signal = load_session(root, name)
#     x, y = x.astype(np.float64), y.astype(np.float64)
#     metrics = {"true": (true_cov, np.eye(WINDOW))}
#     for key, (residuals, spatial, temporal) in {
#             "diff": ("difference", "ledoit_wolf", "white"), "sp": ("whole", "ledoit_wolf", "white"),
#             "iso": ("whole", "isotropic", "white"), "ref": ("whole", "ledoit_wolf", "welch")}.items():
#         model, _ = noise.fit_noise_model(x, None, window_len=WINDOW, residual_mode=residuals,
#                                          spatial_estimator=spatial, temporal_method=temporal,
#                                          kronecker_channels=0)
#         metrics[key] = (model.spatial_cov, model.temporal_cov)
#     prediction, model = calibration.behaviour_prediction(x, y)
#     out: Dict[str, Any] = {"session": name, "behaviour_r2": model["r2"]}
#     for key, (spatial, temporal) in metrics.items():
#         out[f"eps_beh_{key}"] = calibration.eps_behaviour(x, prediction, spatial, temporal)
#         out[f"eps_beh_exact_{key}"] = calibration.eps_behaviour(signal, signal, spatial, temporal)
#     # Spatial distortion of the estimated metrics: d'_est / d'_true of random channel patterns.
#     v = np.random.default_rng(0).normal(size=(2000, x.shape[1]))
#     true_norm = np.einsum("ij,ij->i", v, np.linalg.solve(true_cov, v.T).T)
#     for key in ("diff", "sp", "iso"):
#         est = np.einsum("ij,ij->i", v, np.linalg.solve(metrics[key][0], v.T).T)
#         q = np.quantile(np.sqrt(est / true_norm), [0.05, 0.5, 0.95])
#         out[f"distortion_{key}"] = [float(t) for t in q]
#     runner.atomic_write_json(path, json.loads(json.dumps(out)))
#     return out


# def write_predictions(args, root: Path, jobs: List[Job], sessions: Dict[str, Any]) -> None:
#     """Pre-register budgets, x_beh and the hypotheses; rows are only ever added."""
#     from cebra.dprime import calibration, noise
#     path = root / "PREDICTIONS.json"
#     stored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
#         "created_utc": runner.now_utc(), "hypotheses": __doc__.split("Pre-registered")[1]
#         .split("Suites:")[0].strip(), "fail_half": FAIL_HALF, "perich_kappa": PERICH_KAPPA,
#         "rows": [], "registrations": []}
#     done = {(r["session"], r["mode"], r["arm"]) for r in stored["rows"]}
#     todo = sorted({(j.session, j.mode, j.arm) for j in jobs} - done)
#     if not todo:
#         return
#     cache: Dict[str, Any] = {}
#     for session, mode, name in todo:
#         if session not in cache:
#             (x, _, y, _), _, signal = load_session(root, session)
#             x, y = x.astype(np.float64), y.astype(np.float64)
#             prediction, _ = calibration.behaviour_prediction(x, y)
#             cache[session] = (x, y, signal, prediction)
#         x, y, signal, prediction = cache[session]
#         exp = arm(name)
#         members = runner._metric_members(exp, x, y, noise)
#         row = dict(session=session, mode=mode, arm=name, budget=None, x_beh=None, x_beh_exact=None,
#                    kappa_perich=PERICH_KAPPA[mode].get(name),
#                    eps_beh_ref=sessions[session]["checks"]["eps_beh_ref"])
#         if members is not None:
#             eps_sig = [calibration.eps_signal(x, a, b, runner.SIG_LAG) for a, b in members]
#             eps_beh = [calibration.eps_behaviour(x, prediction, a, b) for a, b in members]
#             exact = [calibration.eps_behaviour(signal, signal, a, b) for a, b in members]
#             blind = calibration.eps_energy_detector(x.shape[1], WINDOW)
#             budgets = runner._rule_budgets(exp, eps_sig, eps_beh, blind, x.shape[1])
#             if budgets is not None:
#                 row.update(budget=budgets[0],
#                            x_beh=float(np.exp(np.mean(np.log(np.divide(budgets, eps_beh))))),
#                            x_beh_exact=float(np.exp(np.mean(np.log(np.divide(budgets, exact))))))
#         if row["kappa_perich"] is not None:
#             row["predicted_ref_half"] = row["kappa_perich"] * row["eps_beh_ref"]
#         stored["rows"].append(row)
#     stored["registrations"].append({
#         "registered_utc": runner.now_utc(), "pairs": len(todo), "suite": args.suite,
#         "fork_fingerprints": runner._fingerprints(args.fork_dir, runner.FORK_EVAL_FILES)})
#     runner.atomic_write_json(path, json.loads(json.dumps(stored)))
#     print(f"Pre-registered {len(todo)} (session, mode, arm) predictions -> {path}", flush=True)


# # ============================================================================ one job
# def _ideal_metric_evaluation(evaluation, noise, true_cov, call):
#     """Run evaluate_decoding with its "reference" metric replaced by the TRUE noise metric."""
#     original = evaluation.common_metrics

#     def ideal(x_train, window_len, labels=None, residuals="whole", device=None):
#         models = original(x_train, window_len, labels, residuals, device)
#         models["reference"] = noise.NoiseModel(true_cov, np.eye(window_len), device=device)
#         return models

#     evaluation.common_metrics = ideal
#     try:
#         return call()
#     finally:
#         evaluation.common_metrics = original


# def _half(entry: Dict[str, Any]) -> Optional[float]:
#     """Refined eps1/2; a censored search reports its bound (below the floor: the smallest budget
#     tried, so it counts as a failure; above the grid: the largest budget)."""
#     refined = entry.get("half_r2_refined")
#     if refined is None:
#         return entry.get("eps_at_half_r2")
#     if refined["status"] == "below_floor":
#         return refined["upper_bound"]
#     if refined["status"] == "above_grid":
#         return max(p["epsilon"] for p in entry["robust_r2"])
#     return refined["epsilon"]


# def run_job(args, job: Job) -> Dict[str, Any]:
#     root = args.output_root
#     exp = arm(job.arm)
#     cebra = runner.load_fork(args.fork_dir, exp, needs_eval_options=True)
#     from cebra.dprime import evaluation, noise
#     arrays, true_cov, _ = load_session(root, job.session)
#     x_train, x_valid, y_train, y_valid = arrays
#     meta_session = json.loads((root / "sessions" / f"{job.session}.json").read_text())
#     plan = runner.Plan(args.suite, job.mode, (exp,), (job.session,), (job.seed,),
#                        args.max_iterations, None, None, root)
#     run_root = root / "runs" / job.session / job.mode / f"seed{job.seed}"
#     rjob = runner.Job(job.session, job.seed, exp, run_root)
#     train_args = SimpleNamespace(batch_size=args.batch_size, device=args.device,
#                                  epsilon=runner.PGD_EPS, mlp_epochs=args.mlp_epochs)
#     if all(p.is_file() for p in runner._checkpoint_paths(run_root, exp.name)):
#         encoder, decoder, meta, report = runner._restore_train(cebra, rjob, plan, arrays,
#                                                                args.max_iterations)
#     else:
#         encoder, decoder, meta, report = runner._new_train(cebra, rjob, plan, arrays,
#                                                            args.max_iterations, train_args)
#     kwargs = dict(epsilons=args.eval_epsilons, steps=args.eval_steps,
#                   batch_size=args.eval_batch_size, contrastive=False, residuals="whole",
#                   labels_train=None, seed=job.seed + 200_000, refine=args.eval_refine,
#                   behaviour_train=y_train, margins=False)
#     if args.eval_restarts > 1:
#         kwargs["restarts"] = args.eval_restarts
#     est = evaluation.evaluate_decoding(encoder, decoder, x_train, x_valid, y_valid,
#                                        metrics=("reference", "isotropic"), **kwargs)
#     true = _ideal_metric_evaluation(evaluation, noise, true_cov, lambda: evaluation.evaluate_decoding(
#         encoder, decoder, x_train, x_valid, y_valid, metrics=("reference",), **kwargs))
#     for robust in (est, true):
#         if abs(float(robust["clean_r2"]) - float(meta["clean_r2"])) > 5e-4:
#             raise RuntimeError("Clean R2 of the evaluation differs from training: alignment error")
#     folder = run_root / "robust_eval"
#     runner.atomic_write_json(folder / f"{exp.name}_estimated.json", est)
#     runner.atomic_write_json(folder / f"{exp.name}_true.json", true)
#     epsilon = (report or {}).get("epsilon", {}) or {}
#     realized = ((report or {}).get("training", {}) or {}).get("realized") or {}
#     half_true = _half(true["reference"])
#     record = dict(
#         session=job.session, alpha=meta_session["alpha"], rep=meta_session["rep"],
#         mode=job.mode, seed=job.seed, arm=job.arm, clean_r2=float(meta["clean_r2"]),
#         eps_beh_true=meta_session["eps_beh_true"],
#         true_half=half_true, true_status=true["reference"].get("half_r2_refined", {}).get("status"),
#         true_half_grid=true["reference"]["eps_at_half_r2"],
#         true_kappa_exact=(half_true / meta_session["eps_beh_true"]) if half_true else None,
#         true_eps_beh_est=true["reference"].get("eps_beh"),
#         true_natural_noise_r2=true["reference"]["natural_noise_r2"],
#         auc_ideal=(0.5 * (1 + math.erf(half_true / 2.0))) if half_true is not None else None,
#         ref_half=_half(est["reference"]), ref_status=est["reference"].get("half_r2_refined", {})
#         .get("status"), ref_eps_beh=est["reference"].get("eps_beh"),
#         ref_kappa=est["reference"].get("kappa"),
#         iso_half=_half(est["isotropic"]), iso_status=est["isotropic"].get("half_r2_refined", {})
#         .get("status"), iso_kappa=est["isotropic"].get("kappa"),
#         eps_star=epsilon.get("eps_star"), eps_source=epsilon.get("source"),
#         eps_binding=epsilon.get("binding"), x_train=epsilon.get("x_train"),
#         loss_gain_median=(realized.get("loss_gain") or {}).get("median"),
#         raw_l2_median=(realized.get("raw_l2") or {}).get("median"),
#         true_curve=[[p["epsilon"], p["r2"]] for p in true["reference"]["robust_r2"]])
#     return record


# def worker(args) -> int:
#     job = Job(**json.loads(args._job))
#     begin = time.monotonic()
#     base = dict(job=job.ident, session=job.session, mode=job.mode, seed=job.seed, arm=job.arm,
#                 status="error", started_utc=runner.now_utc(), error="")
#     try:
#         torch.set_num_threads(args.cpu_threads)
#         base.update(run_job(args, job), status="success")
#         print(f"{job.ident}: clean R2 {base['clean_r2']:.4f} true eps1/2 {base['true_half']}",
#               flush=True)
#     except BaseException as exc:
#         base["error"] = f"{type(exc).__name__}: {exc}"
#         traceback.print_exc()
#     finally:
#         base["elapsed_seconds"] = round(time.monotonic() - begin, 1)
#         base["finished_utc"] = runner.now_utc()
#         runner.atomic_write_json(args.output_root / "records" / f"{job.ident}.json",
#                                  json.loads(json.dumps(base)))
#     return 0 if base["status"] == "success" else 1


# # ============================================================================ coordinator
# def worker_command(args, job: Job) -> List[str]:
#     cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker",
#            "--_job", json.dumps(job.__dict__), "--suite", args.suite,
#            "--output-root", str(args.output_root), "--fork-dir", str(args.fork_dir),
#            "--max-iterations", str(args.max_iterations), "--batch-size", str(args.batch_size),
#            "--mlp-epochs", str(args.mlp_epochs), "--device", args.device,
#            "--cpu-threads", str(args.cpu_threads), "--eval-steps", str(args.eval_steps),
#            "--eval-batch-size", str(args.eval_batch_size), "--eval-refine", str(args.eval_refine),
#            "--eval-restarts", str(args.eval_restarts),
#            "--eval-epsilons", *[str(e) for e in args.eval_epsilons]]
#     return cmd


# def finished(root: Path, job: Job) -> bool:
#     path = root / "records" / f"{job.ident}.json"
#     return path.is_file() and json.loads(path.read_text()).get("status") == "success"


# def orchestrate(args, jobs: List[Job]) -> int:
#     pending = [j for j in jobs if not finished(args.output_root, j)]
#     print(f"{len(pending)}/{len(jobs)} jobs to run on {max(args.gpu_workers, 1)} worker(s); "
#           f"output={args.output_root}", flush=True)
#     if not pending:
#         return 0
#     tokens = runner.gpu_tokens(args.gpu_workers) if args.gpu_workers > 0 else [""]
#     todo: "queue.Queue[Job]" = queue.Queue()
#     for job in pending:
#         todo.put(job)
#     lock, failures = threading.Lock(), []
#     (args.output_root / "logs").mkdir(parents=True, exist_ok=True)

#     def loop(slot: int, token: str):
#         while True:
#             try:
#                 job = todo.get_nowait()
#             except queue.Empty:
#                 return
#             env = dict(os.environ, CUDA_VISIBLE_DEVICES=token)
#             for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
#                 env[key] = str(args.cpu_threads)
#             log = args.output_root / "logs" / f"{job.ident}.log"
#             with lock:
#                 print(f"[{slot}] START {job.ident}", flush=True)
#             with log.open("w") as f:
#                 result = subprocess.run(worker_command(args, job), env=env, stdout=f,
#                                         stderr=subprocess.STDOUT, check=False)
#             ok = result.returncode == 0 and finished(args.output_root, job)
#             with lock:
#                 if not ok:
#                     failures.append(job.ident)
#                 print(f"[{slot}] {'DONE' if ok else 'FAILED'} {job.ident}"
#                       + ("" if ok else f" (see {log})"), flush=True)

#     threads = [threading.Thread(target=loop, args=(i, t)) for i, t in enumerate(tokens)]
#     for t in threads:
#         t.start()
#     for t in threads:
#         t.join()
#     print(f"Finished: {len(pending) - len(failures)}/{len(pending)} successful", flush=True)
#     return 1 if failures else 0


# # ============================================================================ summary
# def _session_ratio(d, mode: str, a: str, b: str, column: str):
#     """Paired log ratios a / b per session (seeds averaged) and their bootstrap CI."""
#     g = d[d["mode"] == mode]
#     piv = g.pivot_table(index=["session", "seed"], columns="arm", values=column)
#     if a not in piv or b not in piv:
#         return None
#     lr = np.log(piv[a] / piv[b]).replace([np.inf, -np.inf], np.nan).dropna()
#     if lr.empty:
#         return None
#     per_session = lr.groupby(level=0).mean()
#     rng = np.random.default_rng(2026)
#     v = per_session.to_numpy()
#     boot = v[rng.integers(0, len(v), size=(20000, len(v)))].mean(axis=1)
#     lo, hi = np.percentile(boot, [2.5, 97.5])
#     return dict(ratio=float(np.exp(v.mean())), lo=float(np.exp(lo)), hi=float(np.exp(hi)),
#                 sessions_up=int((v > 0).sum()), sessions=len(v))


# def _verdict(r, expect_up: bool = True) -> str:
#     if r is None:
#         return "no data"
#     up = r["lo"] > 1 and r["sessions_up"] >= 0.75 * r["sessions"]
#     down = r["hi"] < 1 and r["sessions_up"] <= 0.25 * r["sessions"]
#     if (up and expect_up) or (down and not expect_up):
#         return "SUPPORTED"
#     if (down and expect_up) or (up and not expect_up):
#         return "CONTRADICTED"
#     return "inconclusive"


# def summarize(args) -> int:
#     import pandas as pd
#     pd.set_option("display.width", 220)
#     pd.set_option("display.max_columns", 30)
#     pd.set_option("display.max_rows", 400)
#     root = args.output_root
#     rows = [json.loads(p.read_text()) for p in sorted((root / "records").glob("*.json"))]
#     if not rows:
#         print(f"No records in {root}")
#         return 1
#     d = pd.DataFrame(rows)
#     errors = d[d.status != "success"]
#     d = d[d.status == "success"].copy()
#     d.drop(columns=["true_curve"], errors="ignore").to_csv(root / "SYNTH_RUNS.csv", index=False)
#     print(f"{root}: {len(d)} successful runs, {len(errors)} errors; sessions "
#           f"{d.session.nunique()}, seeds {sorted(d.seed.unique())}")
#     checks = []
#     for name in sorted(d.session.unique()):
#         c = session_checks(root, name)
#         meta = json.loads((root / "sessions" / f"{name}.json").read_text())
#         checks.append(dict(session=name, alpha=meta["alpha"], amplitude=meta["amplitude"],
#                            eps_beh_true=meta["eps_beh_true"], beh_r2=c["behaviour_r2"],
#                            eps_beh_diff=c["eps_beh_diff"], eps_beh_sp=c["eps_beh_sp"],
#                            eps_beh_ref=c["eps_beh_ref"], eps_beh_true_est=c["eps_beh_true"],
#                            dist_diff=c["distortion_diff"][1], dist_sp=c["distortion_sp"][1],
#                            dist_iso=c["distortion_iso"][1]))
#     print("\n0  sessions: separations (estimated in each metric; true = exact) and the median "
#           "d'_est / d'_true of random patterns")
#     print(pd.DataFrame(checks).round(3).to_string(index=False))
#     d["fail"] = d.true_half < FAIL_HALF
#     shorts = {"CLEAN": "CLEAN", "PGD_L2_EPS6": "PGD6", "MADRY_ISO_D8": "ISO8", "MADRY_SP_D8": "SP8",
#               "MADRY_SPDIFF_D8": "DIFF8", "MADRY_SPDIFF_AUTOBEH": "AUTO_BEH",
#               "MADRY_SPDIFF_AUTOSIG": "AUTO_SIG", "MADRY_SPDIFF_GAUSS": "GAUSS"}
#     verdicts = []
#     for mode in ("supervised", "time_only"):
#         g = d[d["mode"] == mode]
#         if g.empty:
#             continue
#         print(f"\n==================== {mode}")
#         table = g.groupby(["alpha", "arm"]).agg(
#             runs=("clean_r2", "size"), clean_r2=("clean_r2", "mean"),
#             true_half=("true_half", "median"), auc_ideal=("auc_ideal", "median"),
#             kappa_true=("true_kappa_exact", "median"), ref_half=("ref_half", "median"),
#             ref_kappa=("ref_kappa", "median"), iso_half=("iso_half", "median"),
#             fails=("fail", "sum"), budget=("eps_star", "median"),
#             loss_gain=("loss_gain_median", "median"))
#         print("1  per arm (true = ideal-observer metric; auc_ideal = Phi(true eps1/2 / sqrt 2))")
#         print(table.round(3).to_string())
#         comparisons = [("MADRY_SPDIFF_D8", "MADRY_ISO_D8", "true_half"),
#                        ("MADRY_SPDIFF_D8", "MADRY_ISO_D8", "iso_half"),
#                        (AUTO[mode], "PGD_L2_EPS6", "true_half"),
#                        (AUTO[mode], "MADRY_ISO_D8", "true_half"),
#                        (AUTO[mode], "MADRY_SPDIFF_GAUSS", "true_half"),
#                        ("MADRY_SP_D8", "MADRY_SPDIFF_D8", "true_half")]
#         comparisons += [(a, "CLEAN", "true_half") for a in
#                         ("PGD_L2_EPS6", "MADRY_ISO_D8", "MADRY_SPDIFF_D8", AUTO[mode])]
#         print("\n2  paired ratios (sessions = independent data draws; CI = session bootstrap)")
#         for alpha in sorted(g.alpha.unique()):
#             ga = g[g.alpha == alpha]
#             for a, b, column in comparisons:
#                 r = _session_ratio(ga, mode, a, b, column)
#                 if r is not None:
#                     print(f"  alpha {alpha:<4g} {shorts.get(a, a):>8} / {shorts.get(b, b):<6} "
#                           f"{column:9s} {r['ratio']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] "
#                           f"sessions up {r['sessions_up']}/{r['sessions']}")
#         alphas = sorted(g.alpha.unique())
#         for arm_name in ("PGD_L2_EPS6", "MADRY_ISO_D8", "MADRY_SPDIFF_D8", AUTO[mode]):
#             r = _session_ratio(g, mode, arm_name, "CLEAN", "true_half")
#             verdicts.append((f"H1 {mode} {shorts[arm_name]} > CLEAN (all alpha)", _verdict(r), r))
#         r0 = _session_ratio(g[g.alpha == alphas[0]], mode, "MADRY_SPDIFF_D8", "MADRY_ISO_D8",
#                             "true_half")
#         band = (math.exp(-0.1), math.exp(0.1))
#         if alphas[0] != 0:
#             control = "no alpha 0"
#         elif r0 is None:
#             control = "no data"
#         elif band[0] < r0["ratio"] < band[1]:
#             control = "SUPPORTED"
#         elif r0["lo"] > band[1] or r0["hi"] < band[0]:
#             control = "CONTRADICTED"
#         else:
#             control = "inconclusive"
#         verdicts.append((f"H2 {mode} DIFF8/ISO8 = 1 at alpha 0 (control)", control, r0))
#         if len(alphas) > 1:
#             r1 = _session_ratio(g[g.alpha == alphas[-1]], mode, "MADRY_SPDIFF_D8", "MADRY_ISO_D8",
#                                 "true_half")
#             verdicts.append((f"H2 {mode} DIFF8/ISO8 > 1 at alpha {alphas[-1]:g}", _verdict(r1), r1))
#             trend = [_session_ratio(g[g.alpha == a], mode, "MADRY_SPDIFF_D8", "MADRY_ISO_D8",
#                                     "true_half") for a in alphas]
#             if all(t is not None for t in trend):
#                 ratios = [t["ratio"] for t in trend]
#                 rising = all(b >= 0.98 * a for a, b in zip(ratios, ratios[1:])) and ratios[-1] > ratios[0]
#                 verdicts.append((f"H2 {mode} DIFF8/ISO8 grows with alpha: "
#                                  + ", ".join(f"{a:g}: {r:.3f}" for a, r in zip(alphas, ratios)),
#                                  "SUPPORTED" if rising else "not supported", None))
#             r3 = _session_ratio(g[g.alpha > 0], mode, "MADRY_SPDIFF_D8", "MADRY_ISO_D8", "iso_half")
#             verdicts.append((f"H3 {mode} DIFF8/ISO8 <= 1 in the isotropic metric (alpha > 0)",
#                              _verdict(r3, expect_up=False), r3))
#         if mode == "supervised":
#             r4 = _session_ratio(g, mode, "MADRY_SP_D8", "MADRY_SPDIFF_D8", "true_half")
#             verdicts.append(("H4 supervised SP8 < DIFF8", _verdict(r4, expect_up=False), r4))
#         r5 = _session_ratio(g, mode, AUTO[mode], "MADRY_SPDIFF_GAUSS", "true_half")
#         verdicts.append((f"H5 {mode} AUTO > GAUSS", _verdict(r5), r5))
#         for arm_name, kappa in PERICH_KAPPA[mode].items():
#             observed = g[g.arm == arm_name].ref_kappa.median()
#             if np.isfinite(observed):
#                 ok = abs(observed / kappa - 1) <= 0.3
#                 verdicts.append((f"H6 {mode} {shorts[arm_name]} ref kappa {observed:.3f} vs Perich "
#                                  f"{kappa:.3f}", "SUPPORTED" if ok else "not supported", None))
#     budget = d[d.arm.str.match(r"MADRY_(SPDIFF|SP)_D\d") & (d["mode"] == "supervised")]
#     if budget.arm.nunique() > 2 and (root / "PREDICTIONS.json").is_file():
#         pred = pd.DataFrame(json.loads((root / "PREDICTIONS.json").read_text())["rows"])
#         b = budget.merge(pred[["session", "mode", "arm", "x_beh", "x_beh_exact"]],
#                          on=["session", "mode", "arm"], how="left")
#         b["metric"] = np.where(b.arm.str.startswith("MADRY_SPDIFF"), "DIFF", "SP")
#         peak = b.groupby(["session", "seed"]).true_half.transform("max")
#         b["rel"] = b.true_half / peak
#         print("\n3  budget: true eps1/2 relative to the session's best budget vs x_beh")
#         print(b.groupby(["metric", "arm"]).agg(x_beh=("x_beh", "median"),
#                                                x_exact=("x_beh_exact", "median"),
#                                                true_half=("true_half", "median"),
#                                                rel=("rel", "median"), clean_r2=("clean_r2", "mean"),
#                                                fails=("fail", "sum")).round(3).to_string())
#         best = b.loc[b.groupby(["session", "seed"]).true_half.idxmax()]
#         print(f"  best x_beh per session: {best.x_beh.round(3).tolist()} (predicted in [0.25, 0.45])")
#         beyond = b[b.x_beh >= 0.5]
#         if len(beyond):
#             print(f"  x_beh >= 0.5: {int((beyond.rel < 0.5).sum())}/{len(beyond)} runs below half "
#                   "the peak (predicted: all)")
#     if verdicts:
#         print("\n4  pre-registered hypotheses")
#         for label, verdict, r in verdicts:
#             detail = (f"  ratio {r['ratio']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}], sessions "
#                       f"{r['sessions_up']}/{r['sessions']}") if r else ""
#             print(f"  {verdict:>13}  {label}{detail}")
#     if len(errors):
#         print("\nerrors:")
#         for _, r in errors.iterrows():
#             print(f"  {r.job}: {str(r.error)[:200]}")
#     return 0


# # ============================================================================ CLI
# def build_parser() -> argparse.ArgumentParser:
#     p = argparse.ArgumentParser(description=__doc__,
#                                 formatter_class=argparse.RawDescriptionHelpFormatter)
#     p.add_argument("--suite", choices=tuple(SUITES), default="pilot")
#     p.add_argument("--alphas", nargs="+", type=float, default=None)
#     p.add_argument("--reps", nargs="+", type=int, default=None)
#     p.add_argument("--seeds", nargs="+", type=int, default=None)
#     p.add_argument("--modes", nargs="+", choices=("supervised", "time_only"), default=None)
#     p.add_argument("--arms", nargs="+", default=None, help="Overrides the suite's arms")
#     p.add_argument("--output-root", type=Path, default=None)
#     p.add_argument("--fork-dir", type=Path, default=runner.DEFAULT_FORK_DIR)
#     for field_name, value in asdict(DGP()).items():
#         p.add_argument("--" + field_name.replace("_", "-"), type=type(value), default=value,
#                        help=f"DGP (default {value})")
#     p.add_argument("--max-iterations", type=int, default=runner.MAX_ITER)
#     p.add_argument("--batch-size", type=int, default=runner.BATCH_SIZE)
#     p.add_argument("--mlp-epochs", type=int, default=runner.MLP_EPOCHS)
#     p.add_argument("--device", default=runner.DEVICE)
#     p.add_argument("--eval-epsilons", nargs="+", type=float,
#                    default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
#     p.add_argument("--eval-steps", type=int, default=20)
#     p.add_argument("--eval-batch-size", type=int, default=512)
#     p.add_argument("--eval-refine", type=int, default=4)
#     p.add_argument("--eval-restarts", type=int, default=1)
#     p.add_argument("--gpu-workers", type=int, default=2, help="0: one CPU worker (smoke tests)")
#     p.add_argument("--cpu-threads", type=int, default=2)
#     p.add_argument("--dry-run", action="store_true", help="Sessions, checks and predictions only")
#     p.add_argument("--summarize", action="store_true")
#     p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
#     p.add_argument("--_job", default=None, help=argparse.SUPPRESS)
#     return p


# def main() -> int:
#     args = build_parser().parse_args()
#     suite = SUITES[args.suite]
#     args.alphas = tuple(args.alphas) if args.alphas else suite["alphas"]
#     args.reps = tuple(args.reps) if args.reps else suite["reps"]
#     args.seeds = tuple(args.seeds) if args.seeds else suite["seeds"]
#     args.modes = tuple(args.modes) if args.modes else suite["modes"]
#     args.output_root = (args.output_root or ROOT / f"SYNTH_{args.suite.upper()}").resolve()
#     args.fork_dir = args.fork_dir.resolve()
#     if args._worker:
#         return worker(args)
#     if args.summarize:
#         runner.load_fork(args.fork_dir)
#         return summarize(args)
#     dgp = DGP(**{k: getattr(args, k) for k in asdict(DGP())})
#     jobs = plan_jobs(args)
#     for job in jobs:
#         arm(job.arm)  # unknown arms fail before anything is written
#     runner.load_fork(args.fork_dir)
#     sessions = ensure_sessions(dgp, args.alphas, args.reps, args.output_root)
#     for name, meta in sessions.items():
#         meta["checks"] = session_checks(args.output_root, name)
#         c = meta["checks"]
#         print(f"{name}: amplitude {meta['amplitude']:.3f}  eps_beh true {meta['eps_beh_true']:.2f}"
#               f" | estimated: true-metric {c['eps_beh_true']:.2f} DIFF {c['eps_beh_diff']:.2f} "
#               f"SP {c['eps_beh_sp']:.2f} ref {c['eps_beh_ref']:.2f} | d'_est/d'_true median DIFF "
#               f"{c['distortion_diff'][1]:.3f} SP {c['distortion_sp'][1]:.3f} ISO "
#               f"{c['distortion_iso'][1]:.3f} | behaviour R2 {c['behaviour_r2']:.3f}", flush=True)
#     write_predictions(args, args.output_root, jobs, sessions)
#     print(f"{len(jobs)} jobs: {len(sessions)} sessions x {len(args.modes)} modes x "
#           f"{len(args.seeds)} seeds x arms", flush=True)
#     if args.dry_run:
#         return 0
#     code = orchestrate(args, jobs)
#     summarize(args)
#     return code


# if __name__ == "__main__":
#     raise SystemExit(main())

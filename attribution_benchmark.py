#!/usr/bin/env python3
"""Attribution benchmark: do the models learn WHICH neurons carry WHICH latent? (xCEBRA Fig. 5)

Keep next to Acorn-dprime, dprime_exp9.py and synthetic_benchmark.py; it reuses their arms,
training, robustness evaluation and job queue, and adds the latent-neuron attribution AUROC of
Schneider et al., "Time-series attribution maps with regularized contrastive learning" (xCEBRA,
Figure 5 / Appendix B.1). Nothing in the fork changes, and synthetic_benchmark.py is only imported
(a suite of it can keep running meanwhile; this one writes to its own --output-root).

Data (xCEBRA Fig. 5 with the noise of synthetic_benchmark.py):
  latents   z = [z1 (d1 = 3), z2 (d2 = 3)]: Brownian motion in the box [-1, 1]^6 (step SD 0.1),
            the labels of supervised CEBRA and of the decoders;
  neurons   x1 = tanh(g1 W1 z1) (25 neurons: z1 ONLY), x2 = tanh(g2 W2 z) (25 neurons: all of z),
            W with orthonormal columns, pre-activation SD 0.8; ground truth gt[i, j] = 1 iff neuron i
            depends on latent j (the x1 x z2 block is 0);
  noise     N(0, Sigma(alpha)) exactly as synthetic_benchmark.py (alpha 0: isotropic); the signal
            amplitude sets the behavioural separation in the true noise metric to 25.

Attribution of a trained model (on held-out data; the encoder's window is 36 bins, so its
Jacobian is summed over the lags, J(t) = sum_tau df(t)/dx(t + tau), the response to a
perturbation held over the window). The embedding has 64 dimensions and the recording 50
neurons, so J itself is tall and its pseudo-inverse unique; as in xCEBRA, whose encoder outputs
the d latents, the Jacobian is taken in latent coordinates: z ~ f W (linear readout fitted on one
half of the validation split), J_z = W^T J (d x C). On the other half, averaged |.| over time:
  inv        xCEBRA's inverted neuron gradient: pinv(J_z) (Euclidean pseudo-inverse, C x d);
  inv_noise  the same inverse in the NOISE metric: S pinv(J_z S), S = Sigma^1/2 with Sigma the
             Ledoit-Wolf covariance of the difference residuals (label-free, as DIFF training);
  inv_true   the same with the true Sigma (oracle);
  fwd        the forward neuron gradient J_z^T.
Why two inverses: for an encoder with the minimum Euclidean-norm Jacobian (what xCEBRA's
||J||_F penalty and isotropic adversarial training favour) pinv(J) A recovers dx/dz exactly; for
an encoder that whitens the noise (the BLUE decoder, what noise-aware training favours) only the
noise-metric inverse does, as soon as the noise is anisotropic. Exact surrogates of both encoders
are scored in every session (the checks), so the claim is verified before any model is trained.

Scores: AUROC of the z-scored map against gt (xCEBRA's metric), AUROC with every latent column
z-scored, and AUPRC. Every model also gets the robustness numbers of synthetic_benchmark.py.

Pre-registered in <output-root>/ATTRIBUTION_PREDICTIONS.json before any training:
  HA1 every adversarially trained arm (PGD6, ISO8, DIFF8, AUTO) has a higher AUROC than CLEAN
      (CEBRA) with the inverse matched to its geometry (inv for PGD6/ISO8, inv_true for DIFF8/AUTO;
      CLEAN with the better of the two);
  HA2 with the noise-metric inverse (inv_true), DIFF8 > ISO8 and AUTO > PGD6 at alpha > 0, and the
      Euclidean and noise-metric inverses agree at alpha = 0 (|AUROC difference| < 0.01);
  HA3 geometry matching at alpha > 0: inv_true > inv for DIFF8 and AUTO; inv >= inv_true for ISO8
      and PGD6;
  checks: exact surrogates score 1 with their own inverse (E with inv, W with inv_true) and < 1
      crossed at alpha > 0. inv_noise is the practical version of inv_true (Sigma estimated from
      the data); its surrogate ceiling is reported (0.92 for W at alpha 1 in the dry run: the
      Brownian latents leak into the difference residuals) and it is analysed as secondary.

Suites:
  pilot  alpha 0 and 1; replicate 0; seed 42; both modes: 28 jobs.
  main   alpha 0 and 1; replicates 0-3; seed 42; both modes: 112 jobs (--alphas 0 0.5 1: 168).
  sweep  the budgets: raw L2 PGD and noise-aware DIFF at 0.1/0.3/0.5/0.7/0.9, L-infinity PGD at
         0.003/0.01/0.03/0.1/0.3/0.5/0.7/0.9; alpha 0 and 1; replicates 0-3; both modes: 288 jobs.
         Run it into the --output-root of main: the summary pairs it with CLEAN, PGD6, DIFF8, AUTO.
         HA4 / HA5 (registered with the suite, after the main results): the best PGD budget of
         each session (oracle) stays below AUTO at alpha 1 and within 0.03 of it at alpha 0;
         DIFF beats raw L2 PGD at every matched budget at alpha 1.
Arms: those of synthetic_benchmark.py (CLEAN, PGD6, ISO8, SP8, DIFF8, AUTO, GAUSS).

Usage (4 GPUs; re-running skips finished jobs):
  python -u attribution_benchmark.py --suite pilot --dry-run    # sessions and checks, no training
  python -u attribution_benchmark.py --suite pilot --gpu-workers 4
  python -u attribution_benchmark.py --suite main --gpu-workers 4
  python attribution_benchmark.py --suite main --summarize > attribution_report.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import dprime_exp9 as runner  # noqa: E402
import synthetic_benchmark as sb  # noqa: E402

import torch  # noqa: E402

# Budget sweep: the same numeric budgets for raw L2 PGD and the noise-aware DIFF metric, and
# L-infinity PGD (ACORN's default norm). An L-infinity budget is per input entry: its ball reaches
# eps * sqrt(C T) = 42 eps in L2 over a 50-neuron x 36-bin window, so it also gets the small
# budgets 0.003-0.03 (L2 radius 0.13-1.3); otherwise it would only be tested where it is large.
SWEEP_EPS = (0.1, 0.3, 0.5, 0.7, 0.9)
SWEEP_ARMS = (tuple(f"PGD_L2_EPS{e:g}" for e in SWEEP_EPS)
              + tuple(f"PGD_LINF_EPS{e:g}" for e in (0.003, 0.01, 0.03) + SWEEP_EPS)
              + tuple(f"MADRY_SPDIFF_D{e:g}" for e in SWEEP_EPS))
SUITES = {
    "pilot": dict(alphas=(0.0, 1.0), reps=(0,), seeds=(42,), modes=("supervised", "time_only"),
                  arms=None),
    "main": dict(alphas=(0.0, 1.0), reps=(0, 1, 2, 3), seeds=(42,),
                 modes=("supervised", "time_only"), arms=None),
    "sweep": dict(alphas=(0.0, 1.0), reps=(0, 1, 2, 3), seeds=(42,),
                  modes=("supervised", "time_only"), arms=SWEEP_ARMS),
}
SUITE_HYPOTHESES = {
    "sweep": "HA4: the best PGD of each session (L2 0.1-0.9 and 6, L-infinity 0.003-0.9; picked "
             "post hoc on its AUROC with the better of the two inverses, an oracle) stays below "
             "AUTO at alpha 1 (CI above 0) and within 0.03 of it at alpha 0. HA5: at alpha 1, "
             "DIFF at budget eps > raw L2 PGD at the same eps for every eps in 0.1-0.9.",
}
ESTIMATORS = ("inv", "inv_noise", "inv_true", "fwd")
ISOTROPIC = ("PGD_L2_EPS6", "MADRY_ISO_D8")
NOISE_AWARE = ("MADRY_SPDIFF_D8",)  # + the mode's AUTO arm
SHORT = {"CLEAN": "CLEAN", "PGD_L2_EPS6": "PGD6", "MADRY_ISO_D8": "ISO8", "MADRY_SP_D8": "SP8",
         "MADRY_SPDIFF_D8": "DIFF8", "MADRY_SPDIFF_AUTOBEH": "AUTO_BEH",
         "MADRY_SPDIFF_AUTOSIG": "AUTO_SIG", "MADRY_SPDIFF_GAUSS": "GAUSS"}


# ============================================================================ data generation
@dataclass(frozen=True)
class AttrDGP:
    n1: int = 25
    n2: int = 25
    d1: int = 3
    d2: int = 3
    n_train: int = 15000
    n_valid: int = 4000
    gap: int = 200
    brownian_sigma: float = 0.1
    pre_sd: float = 0.8
    hetero: float = 1.0
    shared: float = 0.5
    shared_rank: int = 2
    target_eps_beh: float = 25.0
    base_seed: int = 20261103


def session_name(alpha: float, rep: int) -> str:
    return "X" + sb.session_name(alpha, rep)


def brownian_box(n: int, d: int, sigma: float, rng) -> np.ndarray:
    """z_1 ~ U[-1,1]^d; z_t = clip(z_{t-1} + sigma N(0, I), -1, 1) (xCEBRA Appendix B.1)."""
    z = np.empty((n, d))
    z[0] = rng.uniform(-1.0, 1.0, size=d)
    steps = sigma * rng.normal(size=(n, d))
    for t in range(1, n):
        z[t] = np.clip(z[t - 1] + steps[t], -1.0, 1.0)
    return z


def _orthonormal(rows: int, cols: int, rng) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(rows, cols)))
    return q[:, :cols]


def generator_jacobian(z: np.ndarray, gen: Dict[str, np.ndarray]) -> np.ndarray:
    """dx/dz of the noise-free signal at every z: [S, C, d]."""
    w1, w2, (g1, g2), amp = gen["W1"], gen["W2"], gen["gains"], float(gen["amplitude"])
    d1 = w1.shape[1]
    s1 = 1.0 - np.tanh(g1 * z[:, :d1] @ w1.T) ** 2
    s2 = 1.0 - np.tanh(g2 * z @ w2.T) ** 2
    jac = np.zeros((len(z), w1.shape[0] + w2.shape[0], z.shape[1]))
    jac[:, :w1.shape[0], :d1] = amp * g1 * s1[:, :, None] * w1[None]
    jac[:, w1.shape[0]:, :] = amp * g2 * s2[:, :, None] * w2[None]
    return jac


def generate_session(dgp: AttrDGP, alpha: float, rep: int) -> Dict[str, Any]:
    seq = np.random.SeedSequence([dgp.base_seed, rep])
    r_lat, r_tun, r_geo, r_noise = (np.random.default_rng(s) for s in seq.spawn(4))
    d, c = dgp.d1 + dgp.d2, dgp.n1 + dgp.n2
    n = dgp.n_train + dgp.gap + dgp.n_valid
    z = brownian_box(n, d, dgp.brownian_sigma, r_lat)
    w1, w2 = _orthonormal(dgp.n1, dgp.d1, r_tun), _orthonormal(dgp.n2, d, r_tun)
    g1 = dgp.pre_sd / float(np.std(z[:, :dgp.d1] @ w1.T))
    g2 = dgp.pre_sd / float(np.std(z @ w2.T))
    unit = np.concatenate([np.tanh(g1 * z[:, :dgp.d1] @ w1.T), np.tanh(g2 * z @ w2.T)], axis=1)
    geometry = SimpleNamespace(channels=c, hetero=dgp.hetero, shared=dgp.shared,
                               shared_rank=dgp.shared_rank)
    cov = sb.noise_covariance(geometry, alpha, r_geo)
    amplitude = dgp.target_eps_beh / sb.separation(unit, cov)
    signal = amplitude * unit
    x = signal + r_noise.normal(size=(n, c)) @ np.linalg.cholesky(cov).T
    gt = np.zeros((c, d), dtype=np.int8)
    gt[:dgp.n1, :dgp.d1] = 1
    gt[dgp.n1:, :] = 1
    train, valid = slice(0, dgp.n_train), slice(dgp.n_train + dgp.gap, n)
    return dict(train_data=x[train].astype(np.float32), valid_data=x[valid].astype(np.float32),
                train_label=z[train].astype(np.float32), valid_label=z[valid].astype(np.float32),
                train_signal=signal[train].astype(np.float32), true_spatial_cov=cov, gt=gt,
                W1=w1, W2=w2, gains=np.array([g1, g2]), amplitude=np.array(amplitude),
                eps_beh_true=np.array(sb.separation(signal, cov)))


def ensure_sessions(dgp: AttrDGP, alphas, reps, root: Path) -> Dict[str, Dict[str, Any]]:
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
            with open(path, "wb") as f:
                np.savez(f, **data)
            meta.update(amplitude=float(data["amplitude"]), eps_beh_true=float(data["eps_beh_true"]),
                        sha256=runner.file_sha256(path))
            runner.atomic_write_json(meta_path, meta)
            info[name] = meta
    return info


def load_attribution_inputs(root: Path, name: str) -> Dict[str, np.ndarray]:
    with np.load(root / "sessions" / f"{name}.npz", allow_pickle=False) as f:
        out = {k: np.asarray(f[k]) for k in ("gt", "W1", "W2", "gains", "amplitude",
                                             "true_spatial_cov", "valid_label")}
    with np.load(root / "sessions" / f"{name}.attr.npz", allow_pickle=False) as f:
        out["sigma_diff"] = np.asarray(f["sigma_diff"])
    return out


# ============================================================================ attribution maps
def _sqrtm(cov: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(cov)
    return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T


def attribution_maps(jac_z: np.ndarray, sigma_diff: np.ndarray,
                     sigma_true: np.ndarray) -> Dict[str, np.ndarray]:
    """[S, d, C] Jacobians of the decoded latents -> {estimator: [C, d] mean |map|}."""
    maps = {"inv": np.abs(np.linalg.pinv(jac_z, rcond=1e-8)).mean(0)}
    for key, cov in (("inv_noise", sigma_diff), ("inv_true", sigma_true)):
        root = _sqrtm(cov)
        maps[key] = np.abs(root[None] @ np.linalg.pinv(jac_z @ root[None], rcond=1e-8)).mean(0)
    maps["fwd"] = np.abs(np.swapaxes(jac_z, 1, 2)).mean(0)
    return maps


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    s, y = np.asarray(scores, float).ravel(), np.asarray(labels).ravel().astype(bool)
    import pandas as pd
    ranks = pd.Series(s).rank().to_numpy()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    s, y = np.asarray(scores, float).ravel(), np.asarray(labels).ravel().astype(bool)
    y = y[np.argsort(-s, kind="mergesort")]
    tp = np.cumsum(y)
    precision, recall = tp / np.arange(1, len(y) + 1), tp / y.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def binary_scores(m: np.ndarray, gt: np.ndarray, prefix: str) -> Dict[str, float]:
    glob = (m - m.mean()) / (m.std() + 1e-30)
    col = (m - m.mean(0, keepdims=True)) / (m.std(0, keepdims=True) + 1e-30)
    return {f"{prefix}_auroc": _auroc(glob, gt), f"{prefix}_auroc_col": _auroc(col, gt),
            f"{prefix}_auprc": _auprc(glob, gt)}


def surrogate_checks(inputs: Dict[str, np.ndarray], points: int = 400) -> Dict[str, float]:
    """Exact surrogate encoders scored like the models: min-norm (E) and noise-whitened (W)."""
    z = inputs["valid_label"].astype(np.float64)
    z = z[np.linspace(0, len(z) - 1, points).round().astype(int)]
    jac_g = generator_jacobian(z, inputs)
    inv_cov = np.linalg.inv(inputs["true_spatial_cov"])
    jac_e = np.linalg.pinv(jac_g)                                       # z_hat with min-norm J
    jac_w = np.linalg.solve(np.swapaxes(jac_g, 1, 2) @ inv_cov[None] @ jac_g,
                            np.swapaxes(jac_g, 1, 2) @ inv_cov[None])  # noise-whitened (BLUE)
    out = {}
    for name, jac in (("E", jac_e), ("W", jac_w)):
        maps = attribution_maps(jac, inputs["sigma_diff"], inputs["true_spatial_cov"])
        for key in ("inv", "inv_noise", "inv_true"):
            out[f"surrogate_{name}_{key}"] = binary_scores(maps[key], inputs["gt"], "x")["x_auroc"]
    return out


def model_attribution(encoder, x_valid: np.ndarray, inputs: Dict[str, np.ndarray],
                      points: int) -> Dict[str, Any]:
    """Attribution AUROCs of a fitted encoder on the second half of the validation split."""
    from cebra.dprime import evaluation
    z = inputs["valid_label"].astype(np.float64)
    n, half = len(x_valid), len(x_valid) // 2
    f = np.asarray(encoder.transform(x_valid), dtype=np.float64)
    f_map, z_map = f[:half], z[:half]
    f_mean, z_mean = f_map.mean(0), z_map.mean(0)
    fc, zc = f_map - f_mean, z_map - z_mean
    a = np.linalg.lstsq(zc, fc, rcond=None)[0].T                      # f ~ A z   [O, d]
    map_r2 = 1.0 - float(((fc - zc @ a.T) ** 2).sum() / (fc ** 2).sum())
    gram = fc.T @ fc
    w = np.linalg.solve(gram + 1e-3 * np.trace(gram) / len(gram) * np.eye(len(gram)), fc.T @ zc)
    pred = (f[half:] - f_mean) @ w + z_mean                             # z ~ f W   [O, d]
    lin_r2 = float(np.mean(1.0 - ((z[half:] - pred) ** 2).sum(0)
                           / ((z[half:] - z[half:].mean(0)) ** 2).sum(0)))
    model, left, right = evaluation._encoder_model(encoder)
    device = torch.device(encoder.device_)
    windows = evaluation._Windows(np.asarray(x_valid, dtype=np.float32), left, right, device)
    index = np.unique(np.linspace(half, n - 1, points).round().astype(int))
    training = model.training
    model.eval()
    jacobians = []
    try:
        for chunk in np.array_split(index, max(1, len(index) // 128)):
            x = windows(torch.as_tensor(chunk, device=device)).requires_grad_(True)
            out = evaluation._embed(model, x)
            rows = []
            for o in range(out.shape[1]):
                grad, = torch.autograd.grad(out[:, o].sum(), x, retain_graph=o < out.shape[1] - 1)
                rows.append(grad.sum(-1))                               # summed over the lags
            jacobians.append(torch.stack(rows, 1).double().cpu().numpy())
    finally:
        model.train(training)
    jac_z = np.einsum("od,soc->sdc", w, np.concatenate(jacobians))     # [S, d, C]
    maps = attribution_maps(jac_z, inputs["sigma_diff"], inputs["true_spatial_cov"])
    record: Dict[str, Any] = dict(attr_points=len(jac_z), attr_map_r2=map_r2,
                                  attr_linear_r2=lin_r2)
    for key, m in maps.items():
        record.update(binary_scores(m, inputs["gt"], key))
    sv = np.linalg.svd(jac_z, compute_uv=False)
    record["attr_sv_ratio"] = float(np.median(sv[:, -1] / sv[:, 0]))
    return record


# ============================================================================ checks and predictions
def attribution_checks(root: Path, name: str) -> Dict[str, Any]:
    """Noise covariance of the difference residuals (for inv_noise) and the surrogate scores."""
    path = root / "sessions" / f"{name}.attr.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    from cebra.dprime import noise
    (x, _, _, _), _, _ = sb.load_session(root, name)
    model, _ = noise.fit_noise_model(x.astype(np.float64), None, window_len=sb.WINDOW,
                                     residual_mode="difference", spatial_estimator="ledoit_wolf",
                                     temporal_method="white", kronecker_channels=0)
    with open(root / "sessions" / f"{name}.attr.npz", "wb") as f:
        np.savez(f, sigma_diff=model.spatial_cov)
    out = surrogate_checks(load_attribution_inputs(root, name))
    runner.atomic_write_json(path, out)
    return out


HYPOTHESES = __doc__.split("Pre-registered in")[1].split("Suites:")[0].strip()


def write_predictions(args, jobs: List["sb.Job"], checks: Dict[str, Any]) -> None:
    path = args.output_root / "ATTRIBUTION_PREDICTIONS.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "created_utc": runner.now_utc(), "hypotheses": HYPOTHESES, "jobs": [],
        "session_checks": {}, "registrations": []}
    new = sorted({j.ident for j in jobs} - set(stored["jobs"]))
    if not new:
        return
    stored["jobs"] += new
    stored["session_checks"].update(checks)
    stored["registrations"].append({
        "registered_utc": runner.now_utc(), "jobs": len(new), "suite": args.suite,
        "suite_hypotheses": SUITE_HYPOTHESES.get(args.suite),
        "fork_fingerprints": runner._fingerprints(args.fork_dir, runner.FORK_EVAL_FILES)})
    runner.atomic_write_json(path, json.loads(json.dumps(stored)))
    print(f"Pre-registered {len(new)} jobs -> {path}", flush=True)


# ============================================================================ worker and queue
def worker(args) -> int:
    job = sb.Job(**json.loads(args._job))
    begin = time.monotonic()
    base = dict(job=job.ident, session=job.session, mode=job.mode, seed=job.seed, arm=job.arm,
                status="error", started_utc=runner.now_utc(), error="")
    try:
        torch.set_num_threads(args.cpu_threads)
        record = sb.run_job(args, job)  # training (or the saved model) and robustness
        exp = sb.arm(job.arm)
        cebra = runner.load_fork(args.fork_dir, exp, needs_eval_options=True)
        arrays, _, _ = sb.load_session(args.output_root, job.session)
        plan = runner.Plan(args.suite, job.mode, (exp,), (job.session,), (job.seed,),
                           args.max_iterations, None, None, args.output_root)
        run_root = args.output_root / "runs" / job.session / job.mode / f"seed{job.seed}"
        encoder, _, _, _ = runner._restore_train(cebra, runner.Job(job.session, job.seed, exp,
                                                                   run_root), plan, arrays,
                                                 args.max_iterations)
        record.update(model_attribution(encoder, arrays[1],
                                        load_attribution_inputs(args.output_root, job.session),
                                        args.attr_points))
        base.update(record, status="success")
        print(f"{job.ident}: AUROC inv {base['inv_auroc']:.3f} inv_noise "
              f"{base['inv_noise_auroc']:.3f} (map R2 {base['attr_map_r2']:.3f})", flush=True)
    except BaseException as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        base["elapsed_seconds"] = round(time.monotonic() - begin, 1)
        base["finished_utc"] = runner.now_utc()
        runner.atomic_write_json(args.output_root / "records" / f"{job.ident}.json",
                                 json.loads(json.dumps(base)))
    return 0 if base["status"] == "success" else 1


def worker_command(args, job) -> List[str]:
    return [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker",
            "--_job", json.dumps(job.__dict__), "--suite", args.suite,
            "--output-root", str(args.output_root), "--fork-dir", str(args.fork_dir),
            "--max-iterations", str(args.max_iterations), "--batch-size", str(args.batch_size),
            "--mlp-epochs", str(args.mlp_epochs), "--device", args.device,
            "--cpu-threads", str(args.cpu_threads), "--eval-steps", str(args.eval_steps),
            "--eval-batch-size", str(args.eval_batch_size), "--eval-refine", str(args.eval_refine),
            "--eval-restarts", str(args.eval_restarts), "--attr-points", str(args.attr_points),
            "--eval-epsilons", *[str(e) for e in args.eval_epsilons]]


# ============================================================================ summary
def _session_diff(g, a: str, b: str, column: str, column_b: str = None):
    """Paired difference a - b (same session and seed; per session means; session bootstrap)."""
    if column_b is None:
        piv = g.pivot_table(index=["session", "seed"], columns="arm", values=column)
        if a not in piv or b not in piv:
            return None
        diff = (piv[a] - piv[b]).dropna()
    else:  # two estimators of the same arm
        rows = g[g.arm == a].set_index(["session", "seed"])
        diff = (rows[column] - rows[column_b]).dropna()
    if diff.empty:
        return None
    v = diff.groupby(level=0).mean().to_numpy()
    boot = v[np.random.default_rng(2026).integers(0, len(v), size=(20000, len(v)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(diff=float(v.mean()), lo=float(lo), hi=float(hi), up=int((v > 0).sum()),
                n=len(v))


def _verdict(r) -> str:
    if r is None:
        return "no data"
    if r["lo"] > 0 and r["up"] >= 0.75 * r["n"]:
        return "SUPPORTED"
    if r["hi"] < 0 and r["up"] <= 0.25 * r["n"]:
        return "CONTRADICTED"
    return "inconclusive"


def summarize(args) -> int:
    import pandas as pd
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 400)
    root = args.output_root
    rows = [json.loads(p.read_text()) for p in sorted((root / "records").glob("*.json"))]
    if not rows:
        print(f"No records in {root}")
        return 1
    d = pd.DataFrame(rows)
    errors = d[d.status != "success"]
    d = d[d.status == "success"].copy()
    d.drop(columns=["true_curve"], errors="ignore").to_csv(root / "ATTRIBUTION_RUNS.csv",
                                                            index=False)
    print(f"{root}: {len(d)} successful runs, {len(errors)} errors; sessions {d.session.nunique()}")
    checks = []
    for name in sorted(d.session.unique()):
        meta = json.loads((root / "sessions" / f"{name}.json").read_text())
        checks.append(dict(session=name, alpha=meta["alpha"], **attribution_checks(root, name)))
    print("\n0  exact surrogate encoders (E: minimum Euclidean norm, W: noise-whitened) scored "
          "with each inverse: AUROC (expected 1 for E-inv and W-inv_true; W-inv_noise is the "
          "ceiling of the practical inverse)")
    print(pd.DataFrame(checks).round(3).to_string(index=False))
    verdicts = []
    for mode in ("supervised", "time_only"):
        g = d[d["mode"] == mode]
        if g.empty:
            continue
        auto = sb.AUTO[mode]
        print(f"\n==================== {mode}")
        table = g.groupby(["alpha", "arm"]).agg(
            runs=("clean_r2", "size"), clean_r2=("clean_r2", "mean"),
            map_r2=("attr_map_r2", "mean"), lin_r2=("attr_linear_r2", "mean"),
            inv=("inv_auroc", "mean"), inv_noise=("inv_noise_auroc", "mean"),
            inv_true=("inv_true_auroc", "mean"), fwd=("fwd_auroc", "mean"),
            inv_col=("inv_auroc_col", "mean"), inv_noise_col=("inv_noise_auroc_col", "mean"),
            auprc_inv=("inv_auprc", "mean"), auprc_noise=("inv_noise_auprc", "mean"),
            true_half=("true_half", "median"))
        print("1  attribution AUROC per estimator (mean over sessions), with the linear "
              "identifiability (map / readout R2) and the robustness (true eps1/2)")
        print(table.round(3).to_string())
        print("\n2  paired AUROC differences (sessions = data draws; CI = session bootstrap)")
        for alpha in sorted(g.alpha.unique()):
            ga = g[g.alpha == alpha]
            for est in ("inv", "inv_true", "inv_noise"):
                col = f"{est}_auroc"
                for a, b in [(x, "CLEAN") for x in ISOTROPIC + NOISE_AWARE + (auto,)] + [
                        ("MADRY_SPDIFF_D8", "MADRY_ISO_D8"), (auto, "PGD_L2_EPS6"),
                        (auto, "MADRY_SPDIFF_GAUSS")]:
                    r = _session_diff(ga, a, b, col)
                    if r:
                        print(f"  alpha {alpha:<4g} {est:9s} {SHORT[a]:>8} - {SHORT[b]:<6} "
                              f"{r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] sessions up "
                              f"{r['up']}/{r['n']}")
            for a in ISOTROPIC + NOISE_AWARE + (auto, "CLEAN"):
                for est in ("inv_true", "inv_noise"):
                    r = _session_diff(ga, a, None, f"{est}_auroc", "inv_auroc")
                    if r:
                        print(f"  alpha {alpha:<4g} {SHORT[a]:>8}: {est} - inv {r['diff']:+.4f} "
                              f"[{r['lo']:+.4f}, {r['hi']:+.4f}] sessions up {r['up']}/{r['n']}")
        # HA1: matched inverse vs CLEAN (CLEAN read with the better of the two inverses).
        g = g.copy()
        g["matched"] = np.where(g.arm.isin(NOISE_AWARE + (auto,)), g.inv_true_auroc,
                                np.where(g.arm.isin(ISOTROPIC), g.inv_auroc,
                                         np.maximum(g.inv_auroc, g.inv_true_auroc)))
        for a in ISOTROPIC + NOISE_AWARE + (auto,):
            r = _session_diff(g, a, "CLEAN", "matched")
            verdicts.append((f"HA1 {mode} {SHORT[a]} > CLEAN (matched inverse, all alpha)",
                             _verdict(r), r))
        alphas = sorted(g.alpha.unique())
        for alpha in [x for x in alphas if x > 0]:
            ga = g[g.alpha == alpha]
            for a, b in (("MADRY_SPDIFF_D8", "MADRY_ISO_D8"), (auto, "PGD_L2_EPS6")):
                for est in ("inv_true", "inv_noise"):
                    r = _session_diff(ga, a, b, f"{est}_auroc")
                    verdicts.append((f"HA2 {mode} alpha {alpha:g} {SHORT[a]} > {SHORT[b]} ({est}"
                                     + ("" if est == "inv_true" else ", secondary") + ")",
                                     _verdict(r), r))
            for a in NOISE_AWARE + (auto,):
                r = _session_diff(ga, a, None, "inv_true_auroc", "inv_auroc")
                verdicts.append((f"HA3 {mode} alpha {alpha:g} {SHORT[a]}: inv_true > inv",
                                 _verdict(r), r))
            for a in ISOTROPIC:
                r = _session_diff(ga, a, None, "inv_auroc", "inv_true_auroc")
                ok = r is not None and r["diff"] >= 0
                verdicts.append((f"HA3 {mode} alpha {alpha:g} {SHORT[a]}: inv >= inv_true",
                                 "SUPPORTED" if ok else ("no data" if r is None else
                                                         "not supported"), r))
        if 0.0 in alphas:
            g0 = g[g.alpha == 0.0]
            gap = (g0.inv_true_auroc - g0.inv_auroc).abs().max()
            verdicts.append((f"HA2 {mode} alpha 0: the two inverses agree (max |diff| "
                             f"{gap:.4f})", "SUPPORTED" if gap < 0.01 else "not supported", None))
    sweep_section(d, verdicts)
    if verdicts:
        print("\n3  pre-registered hypotheses")
        for label, verdict, r in verdicts:
            detail = (f"  diff {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}], sessions "
                      f"{r['up']}/{r['n']}") if r else ""
            print(f"  {verdict:>13}  {label}{detail}")
    if len(errors):
        print("\nerrors:")
        for _, r in errors.iterrows():
            print(f"  {r.job}: {str(r.error)[:200]}")
    return 0


def _family(name: str):
    """(family, budget) of a budget-sweep arm, or None."""
    import re
    for pattern, family in ((r"^PGD_L2_EPS([0-9.]+)$", "PGD-L2"), (r"^PGD_LINF_EPS([0-9.]+)$",
                            "PGD-Linf"), (r"^MADRY_SPDIFF_D([0-9.]+)$", "DIFF"),
                            (r"^MADRY_ISO_D([0-9.]+)$", "ISO")):
        m = re.match(pattern, name)
        if m:
            return family, float(m.group(1))
    return None


def sweep_section(d, verdicts) -> None:
    """Section 4: AUROC against the budget for every family; the oracle PGD against AUTO."""
    import pandas as pd
    if d[d.arm.isin(SWEEP_ARMS)].empty:
        return
    d = d.copy()
    d["best_inverse"] = np.maximum(d.inv_auroc, d.inv_true_auroc)
    d["matched_inverse"] = np.where(d.arm.str.startswith("MADRY_SPDIFF"), d.inv_true_auroc,
                                    d.inv_auroc)
    print("\n4  budget sweep: AUROC against the training budget (PGD-L2 / PGD-Linf: raw input "
          "units, Linf per entry; DIFF: d' in the noise metric); 'best' = the better of inv and "
          "inv_true for every run, 'matched' = inv for PGD, inv_true for DIFF")
    for mode in ("supervised", "time_only"):
        g = d[d["mode"] == mode]
        fam = [(_family(a) or (None, None)) for a in g.arm]
        g = g.assign(family=[f for f, _ in fam], budget_value=[b for _, b in fam])
        sweep = g[g.family.notna()]
        if sweep.empty:
            continue
        auto = sb.AUTO[mode]
        for column in ("best_inverse", "matched_inverse", "clean_r2"):
            table = sweep.pivot_table(index=["alpha", "family"], columns="budget_value",
                                      values=column, aggfunc="mean")
            print(f"\n  {mode}: {column} (mean over sessions)")
            print(table.round(3).to_string())
        pgd = g[g.arm.str.startswith("PGD_")]
        best = pgd.loc[pgd.groupby(["alpha", "session", "seed"]).best_inverse.idxmax()].copy()
        print(f"\n  {mode}: best PGD arm per session (oracle on the AUROC):")
        for alpha, arms in best.groupby("alpha").arm:
            print(f"    alpha {alpha:g}: {arms.value_counts().to_dict()}")
        best["arm"] = "ORACLE_PGD"
        diffs = g[g.family == "DIFF"]
        best_diff = diffs.loc[diffs.groupby(["alpha", "session", "seed"]).best_inverse.idxmax()].copy()
        best_diff["arm"] = "ORACLE_DIFF"
        both = pd.concat([g, best, best_diff], ignore_index=True)
        for alpha in sorted(best.alpha.unique()):
            ga = both[both.alpha == alpha]
            for a in (auto, "MADRY_SPDIFF_D8", "ORACLE_DIFF"):
                r = _session_diff(ga, a, "ORACLE_PGD", "best_inverse")
                if r:
                    print(f"    alpha {alpha:<4g} {SHORT.get(a, a):>11} - ORACLE_PGD  "
                          f"{r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] sessions up "
                          f"{r['up']}/{r['n']}")
            for eps in SWEEP_EPS:
                a, b = f"MADRY_SPDIFF_D{eps:g}", f"PGD_L2_EPS{eps:g}"
                r = _session_diff(ga, a, b, "matched_inverse")
                if r:
                    print(f"    alpha {alpha:<4g} DIFF{eps:g} - PGD-L2 {eps:g} (matched)  "
                          f"{r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] sessions up "
                          f"{r['up']}/{r['n']}")
                    if alpha > 0:
                        verdicts.append((f"HA5 {mode} alpha {alpha:g} DIFF{eps:g} > PGD-L2 {eps:g}",
                                         _verdict(r), r))
            r = _session_diff(ga, auto, "ORACLE_PGD", "best_inverse")
            if alpha == 0:
                verdict = ("no data" if r is None else
                           "SUPPORTED" if abs(r["diff"]) < 0.03 else "not supported")
                label = f"HA4 {mode} alpha 0: AUTO within 0.03 of the best PGD (oracle)"
            else:
                verdict, label = _verdict(r), f"HA4 {mode} alpha {alpha:g}: AUTO > best PGD (oracle)"
            verdicts.append((label, verdict, r))


# ============================================================================ CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", choices=tuple(SUITES), default="pilot")
    p.add_argument("--alphas", nargs="+", type=float, default=None)
    p.add_argument("--reps", nargs="+", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--modes", nargs="+", choices=("supervised", "time_only"), default=None)
    p.add_argument("--arms", nargs="+", default=None, help="Overrides the arms")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--fork-dir", type=Path, default=runner.DEFAULT_FORK_DIR)
    for field_name, value in asdict(AttrDGP()).items():
        p.add_argument("--" + field_name.replace("_", "-"), type=type(value), default=value,
                       help=f"DGP (default {value})")
    p.add_argument("--attr-points", type=int, default=1000,
                   help="Held-out time points of the attribution maps")
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
    p.add_argument("--dry-run", action="store_true", help="Sessions and checks only")
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
    args.output_root = (args.output_root or ROOT / "SYNTH_ATTRIBUTION").resolve()
    args.fork_dir = args.fork_dir.resolve()
    if args._worker:
        return worker(args)
    runner.load_fork(args.fork_dir)
    if args.summarize:
        return summarize(args)
    dgp = AttrDGP(**{k: getattr(args, k) for k in asdict(AttrDGP())})
    jobs = []
    for alpha in args.alphas:
        for rep in args.reps:
            for mode in args.modes:
                arms = [a for a in (args.arms or suite["arms"] or sb.MAIN_ARMS[mode])
                        if mode == "supervised" or not sb.arm(a).supervised_only]
                jobs += [sb.Job(session_name(alpha, rep), mode, s, a) for s in args.seeds
                         for a in arms]
    sessions = ensure_sessions(dgp, args.alphas, args.reps, args.output_root)
    checks = {}
    for name, meta in sessions.items():
        c = sb.session_checks(args.output_root, name)
        checks[name] = attribution_checks(args.output_root, name)
        s = checks[name]
        print(f"{name}: eps_beh true {meta['eps_beh_true']:.2f} (DIFF est {c['eps_beh_diff']:.2f}) "
              f"| surrogate AUROC E (min-norm): inv {s['surrogate_E_inv']:.3f} inv_true "
              f"{s['surrogate_E_inv_true']:.3f} | W (whitened): inv {s['surrogate_W_inv']:.3f} "
              f"inv_true {s['surrogate_W_inv_true']:.3f} inv_noise {s['surrogate_W_inv_noise']:.3f}",
              flush=True)
    write_predictions(args, jobs, checks)
    print(f"{len(jobs)} jobs: {len(sessions)} sessions x {len(args.modes)} modes x arms",
          flush=True)
    if args.dry_run:
        return 0
    sb.worker_command = lambda a, j: worker_command(a, j)  # the queue runs this file's worker
    code = sb.orchestrate(args, jobs)
    summarize(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())



# #!/usr/bin/env python3
# """Attribution benchmark: do the models learn WHICH neurons carry WHICH latent? (xCEBRA Fig. 5)

# Keep next to Acorn-dprime, dprime_exp9.py and synthetic_benchmark.py; it reuses their arms,
# training, robustness evaluation and job queue, and adds the latent-neuron attribution AUROC of
# Schneider et al., "Time-series attribution maps with regularized contrastive learning" (xCEBRA,
# Figure 5 / Appendix B.1). Nothing in the fork changes, and synthetic_benchmark.py is only imported
# (a suite of it can keep running meanwhile; this one writes to its own --output-root).

# Data (xCEBRA Fig. 5 with the noise of synthetic_benchmark.py):
#   latents   z = [z1 (d1 = 3), z2 (d2 = 3)]: Brownian motion in the box [-1, 1]^6 (step SD 0.1),
#             the labels of supervised CEBRA and of the decoders;
#   neurons   x1 = tanh(g1 W1 z1) (25 neurons: z1 ONLY), x2 = tanh(g2 W2 z) (25 neurons: all of z),
#             W with orthonormal columns, pre-activation SD 0.8; ground truth gt[i, j] = 1 iff neuron i
#             depends on latent j (the x1 x z2 block is 0);
#   noise     N(0, Sigma(alpha)) exactly as synthetic_benchmark.py (alpha 0: isotropic); the signal
#             amplitude sets the behavioural separation in the true noise metric to 25.

# Attribution of a trained model (on held-out data; the encoder's window is 36 bins, so its
# Jacobian is summed over the lags, J(t) = sum_tau df(t)/dx(t + tau), the response to a
# perturbation held over the window). The embedding has 64 dimensions and the recording 50
# neurons, so J itself is tall and its pseudo-inverse unique; as in xCEBRA, whose encoder outputs
# the d latents, the Jacobian is taken in latent coordinates: z ~ f W (linear readout fitted on one
# half of the validation split), J_z = W^T J (d x C). On the other half, averaged |.| over time:
#   inv        xCEBRA's inverted neuron gradient: pinv(J_z) (Euclidean pseudo-inverse, C x d);
#   inv_noise  the same inverse in the NOISE metric: S pinv(J_z S), S = Sigma^1/2 with Sigma the
#              Ledoit-Wolf covariance of the difference residuals (label-free, as DIFF training);
#   inv_true   the same with the true Sigma (oracle);
#   fwd        the forward neuron gradient J_z^T.
# Why two inverses: for an encoder with the minimum Euclidean-norm Jacobian (what xCEBRA's
# ||J||_F penalty and isotropic adversarial training favour) pinv(J) A recovers dx/dz exactly; for
# an encoder that whitens the noise (the BLUE decoder, what noise-aware training favours) only the
# noise-metric inverse does, as soon as the noise is anisotropic. Exact surrogates of both encoders
# are scored in every session (the checks), so the claim is verified before any model is trained.

# Scores: AUROC of the z-scored map against gt (xCEBRA's metric), AUROC with every latent column
# z-scored, and AUPRC. Every model also gets the robustness numbers of synthetic_benchmark.py.

# Pre-registered in <output-root>/ATTRIBUTION_PREDICTIONS.json before any training:
#   HA1 every adversarially trained arm (PGD6, ISO8, DIFF8, AUTO) has a higher AUROC than CLEAN
#       (CEBRA) with the inverse matched to its geometry (inv for PGD6/ISO8, inv_true for DIFF8/AUTO;
#       CLEAN with the better of the two);
#   HA2 with the noise-metric inverse (inv_true), DIFF8 > ISO8 and AUTO > PGD6 at alpha > 0, and the
#       Euclidean and noise-metric inverses agree at alpha = 0 (|AUROC difference| < 0.01);
#   HA3 geometry matching at alpha > 0: inv_true > inv for DIFF8 and AUTO; inv >= inv_true for ISO8
#       and PGD6;
#   checks: exact surrogates score 1 with their own inverse (E with inv, W with inv_true) and < 1
#       crossed at alpha > 0. inv_noise is the practical version of inv_true (Sigma estimated from
#       the data); its surrogate ceiling is reported (0.92 for W at alpha 1 in the dry run: the
#       Brownian latents leak into the difference residuals) and it is analysed as secondary.

# Suites:
#   pilot  alpha 0 and 1; replicate 0; seed 42; both modes: 28 jobs.
#   main   alpha 0 and 1; replicates 0-3; seed 42; both modes: 112 jobs (--alphas 0 0.5 1: 168).
# Arms: those of synthetic_benchmark.py (CLEAN, PGD6, ISO8, SP8, DIFF8, AUTO, GAUSS).

# Usage (4 GPUs; re-running skips finished jobs):
#   python -u attribution_benchmark.py --suite pilot --dry-run    # sessions and checks, no training
#   python -u attribution_benchmark.py --suite pilot --gpu-workers 4
#   python -u attribution_benchmark.py --suite main --gpu-workers 4
#   python attribution_benchmark.py --suite main --summarize > attribution_report.txt
# """

# from __future__ import annotations

# import argparse
# import json
# import sys
# import time
# import traceback
# from dataclasses import asdict, dataclass
# from pathlib import Path
# from types import SimpleNamespace
# from typing import Any, Dict, List

# import numpy as np

# ROOT = Path(__file__).resolve().parent
# sys.path.insert(0, str(ROOT))
# import dprime_exp9 as runner  # noqa: E402
# import synthetic_benchmark as sb  # noqa: E402

# import torch  # noqa: E402

# SUITES = {
#     "pilot": dict(alphas=(0.0, 1.0), reps=(0,), seeds=(42,), modes=("supervised", "time_only")),
#     "main": dict(alphas=(0.0, 1.0), reps=(0, 1, 2, 3), seeds=(42,),
#                  modes=("supervised", "time_only")),
# }
# ESTIMATORS = ("inv", "inv_noise", "inv_true", "fwd")
# ISOTROPIC = ("PGD_L2_EPS6", "MADRY_ISO_D8")
# NOISE_AWARE = ("MADRY_SPDIFF_D8",)  # + the mode's AUTO arm
# SHORT = {"CLEAN": "CLEAN", "PGD_L2_EPS6": "PGD6", "MADRY_ISO_D8": "ISO8", "MADRY_SP_D8": "SP8",
#          "MADRY_SPDIFF_D8": "DIFF8", "MADRY_SPDIFF_AUTOBEH": "AUTO_BEH",
#          "MADRY_SPDIFF_AUTOSIG": "AUTO_SIG", "MADRY_SPDIFF_GAUSS": "GAUSS"}


# # ============================================================================ data generation
# @dataclass(frozen=True)
# class AttrDGP:
#     n1: int = 25
#     n2: int = 25
#     d1: int = 3
#     d2: int = 3
#     n_train: int = 15000
#     n_valid: int = 4000
#     gap: int = 200
#     brownian_sigma: float = 0.1
#     pre_sd: float = 0.8
#     hetero: float = 1.0
#     shared: float = 0.5
#     shared_rank: int = 2
#     target_eps_beh: float = 25.0
#     base_seed: int = 20261103


# def session_name(alpha: float, rep: int) -> str:
#     return "X" + sb.session_name(alpha, rep)


# def brownian_box(n: int, d: int, sigma: float, rng) -> np.ndarray:
#     """z_1 ~ U[-1,1]^d; z_t = clip(z_{t-1} + sigma N(0, I), -1, 1) (xCEBRA Appendix B.1)."""
#     z = np.empty((n, d))
#     z[0] = rng.uniform(-1.0, 1.0, size=d)
#     steps = sigma * rng.normal(size=(n, d))
#     for t in range(1, n):
#         z[t] = np.clip(z[t - 1] + steps[t], -1.0, 1.0)
#     return z


# def _orthonormal(rows: int, cols: int, rng) -> np.ndarray:
#     q, _ = np.linalg.qr(rng.normal(size=(rows, cols)))
#     return q[:, :cols]


# def generator_jacobian(z: np.ndarray, gen: Dict[str, np.ndarray]) -> np.ndarray:
#     """dx/dz of the noise-free signal at every z: [S, C, d]."""
#     w1, w2, (g1, g2), amp = gen["W1"], gen["W2"], gen["gains"], float(gen["amplitude"])
#     d1 = w1.shape[1]
#     s1 = 1.0 - np.tanh(g1 * z[:, :d1] @ w1.T) ** 2
#     s2 = 1.0 - np.tanh(g2 * z @ w2.T) ** 2
#     jac = np.zeros((len(z), w1.shape[0] + w2.shape[0], z.shape[1]))
#     jac[:, :w1.shape[0], :d1] = amp * g1 * s1[:, :, None] * w1[None]
#     jac[:, w1.shape[0]:, :] = amp * g2 * s2[:, :, None] * w2[None]
#     return jac


# def generate_session(dgp: AttrDGP, alpha: float, rep: int) -> Dict[str, Any]:
#     seq = np.random.SeedSequence([dgp.base_seed, rep])
#     r_lat, r_tun, r_geo, r_noise = (np.random.default_rng(s) for s in seq.spawn(4))
#     d, c = dgp.d1 + dgp.d2, dgp.n1 + dgp.n2
#     n = dgp.n_train + dgp.gap + dgp.n_valid
#     z = brownian_box(n, d, dgp.brownian_sigma, r_lat)
#     w1, w2 = _orthonormal(dgp.n1, dgp.d1, r_tun), _orthonormal(dgp.n2, d, r_tun)
#     g1 = dgp.pre_sd / float(np.std(z[:, :dgp.d1] @ w1.T))
#     g2 = dgp.pre_sd / float(np.std(z @ w2.T))
#     unit = np.concatenate([np.tanh(g1 * z[:, :dgp.d1] @ w1.T), np.tanh(g2 * z @ w2.T)], axis=1)
#     geometry = SimpleNamespace(channels=c, hetero=dgp.hetero, shared=dgp.shared,
#                                shared_rank=dgp.shared_rank)
#     cov = sb.noise_covariance(geometry, alpha, r_geo)
#     amplitude = dgp.target_eps_beh / sb.separation(unit, cov)
#     signal = amplitude * unit
#     x = signal + r_noise.normal(size=(n, c)) @ np.linalg.cholesky(cov).T
#     gt = np.zeros((c, d), dtype=np.int8)
#     gt[:dgp.n1, :dgp.d1] = 1
#     gt[dgp.n1:, :] = 1
#     train, valid = slice(0, dgp.n_train), slice(dgp.n_train + dgp.gap, n)
#     return dict(train_data=x[train].astype(np.float32), valid_data=x[valid].astype(np.float32),
#                 train_label=z[train].astype(np.float32), valid_label=z[valid].astype(np.float32),
#                 train_signal=signal[train].astype(np.float32), true_spatial_cov=cov, gt=gt,
#                 W1=w1, W2=w2, gains=np.array([g1, g2]), amplitude=np.array(amplitude),
#                 eps_beh_true=np.array(sb.separation(signal, cov)))


# def ensure_sessions(dgp: AttrDGP, alphas, reps, root: Path) -> Dict[str, Dict[str, Any]]:
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
#             with open(path, "wb") as f:
#                 np.savez(f, **data)
#             meta.update(amplitude=float(data["amplitude"]), eps_beh_true=float(data["eps_beh_true"]),
#                         sha256=runner.file_sha256(path))
#             runner.atomic_write_json(meta_path, meta)
#             info[name] = meta
#     return info


# def load_attribution_inputs(root: Path, name: str) -> Dict[str, np.ndarray]:
#     with np.load(root / "sessions" / f"{name}.npz", allow_pickle=False) as f:
#         out = {k: np.asarray(f[k]) for k in ("gt", "W1", "W2", "gains", "amplitude",
#                                              "true_spatial_cov", "valid_label")}
#     with np.load(root / "sessions" / f"{name}.attr.npz", allow_pickle=False) as f:
#         out["sigma_diff"] = np.asarray(f["sigma_diff"])
#     return out


# # ============================================================================ attribution maps
# def _sqrtm(cov: np.ndarray) -> np.ndarray:
#     w, v = np.linalg.eigh(cov)
#     return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T


# def attribution_maps(jac_z: np.ndarray, sigma_diff: np.ndarray,
#                      sigma_true: np.ndarray) -> Dict[str, np.ndarray]:
#     """[S, d, C] Jacobians of the decoded latents -> {estimator: [C, d] mean |map|}."""
#     maps = {"inv": np.abs(np.linalg.pinv(jac_z, rcond=1e-8)).mean(0)}
#     for key, cov in (("inv_noise", sigma_diff), ("inv_true", sigma_true)):
#         root = _sqrtm(cov)
#         maps[key] = np.abs(root[None] @ np.linalg.pinv(jac_z @ root[None], rcond=1e-8)).mean(0)
#     maps["fwd"] = np.abs(np.swapaxes(jac_z, 1, 2)).mean(0)
#     return maps


# def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
#     s, y = np.asarray(scores, float).ravel(), np.asarray(labels).ravel().astype(bool)
#     import pandas as pd
#     ranks = pd.Series(s).rank().to_numpy()
#     n_pos, n_neg = int(y.sum()), int((~y).sum())
#     return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
#     s, y = np.asarray(scores, float).ravel(), np.asarray(labels).ravel().astype(bool)
#     y = y[np.argsort(-s, kind="mergesort")]
#     tp = np.cumsum(y)
#     precision, recall = tp / np.arange(1, len(y) + 1), tp / y.sum()
#     return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


# def binary_scores(m: np.ndarray, gt: np.ndarray, prefix: str) -> Dict[str, float]:
#     glob = (m - m.mean()) / (m.std() + 1e-30)
#     col = (m - m.mean(0, keepdims=True)) / (m.std(0, keepdims=True) + 1e-30)
#     return {f"{prefix}_auroc": _auroc(glob, gt), f"{prefix}_auroc_col": _auroc(col, gt),
#             f"{prefix}_auprc": _auprc(glob, gt)}


# def surrogate_checks(inputs: Dict[str, np.ndarray], points: int = 400) -> Dict[str, float]:
#     """Exact surrogate encoders scored like the models: min-norm (E) and noise-whitened (W)."""
#     z = inputs["valid_label"].astype(np.float64)
#     z = z[np.linspace(0, len(z) - 1, points).round().astype(int)]
#     jac_g = generator_jacobian(z, inputs)
#     inv_cov = np.linalg.inv(inputs["true_spatial_cov"])
#     jac_e = np.linalg.pinv(jac_g)                                       # z_hat with min-norm J
#     jac_w = np.linalg.solve(np.swapaxes(jac_g, 1, 2) @ inv_cov[None] @ jac_g,
#                             np.swapaxes(jac_g, 1, 2) @ inv_cov[None])  # noise-whitened (BLUE)
#     out = {}
#     for name, jac in (("E", jac_e), ("W", jac_w)):
#         maps = attribution_maps(jac, inputs["sigma_diff"], inputs["true_spatial_cov"])
#         for key in ("inv", "inv_noise", "inv_true"):
#             out[f"surrogate_{name}_{key}"] = binary_scores(maps[key], inputs["gt"], "x")["x_auroc"]
#     return out


# def model_attribution(encoder, x_valid: np.ndarray, inputs: Dict[str, np.ndarray],
#                       points: int) -> Dict[str, Any]:
#     """Attribution AUROCs of a fitted encoder on the second half of the validation split."""
#     from cebra.dprime import evaluation
#     z = inputs["valid_label"].astype(np.float64)
#     n, half = len(x_valid), len(x_valid) // 2
#     f = np.asarray(encoder.transform(x_valid), dtype=np.float64)
#     f_map, z_map = f[:half], z[:half]
#     f_mean, z_mean = f_map.mean(0), z_map.mean(0)
#     fc, zc = f_map - f_mean, z_map - z_mean
#     a = np.linalg.lstsq(zc, fc, rcond=None)[0].T                      # f ~ A z   [O, d]
#     map_r2 = 1.0 - float(((fc - zc @ a.T) ** 2).sum() / (fc ** 2).sum())
#     gram = fc.T @ fc
#     w = np.linalg.solve(gram + 1e-3 * np.trace(gram) / len(gram) * np.eye(len(gram)), fc.T @ zc)
#     pred = (f[half:] - f_mean) @ w + z_mean                             # z ~ f W   [O, d]
#     lin_r2 = float(np.mean(1.0 - ((z[half:] - pred) ** 2).sum(0)
#                            / ((z[half:] - z[half:].mean(0)) ** 2).sum(0)))
#     model, left, right = evaluation._encoder_model(encoder)
#     device = torch.device(encoder.device_)
#     windows = evaluation._Windows(np.asarray(x_valid, dtype=np.float32), left, right, device)
#     index = np.unique(np.linspace(half, n - 1, points).round().astype(int))
#     training = model.training
#     model.eval()
#     jacobians = []
#     try:
#         for chunk in np.array_split(index, max(1, len(index) // 128)):
#             x = windows(torch.as_tensor(chunk, device=device)).requires_grad_(True)
#             out = evaluation._embed(model, x)
#             rows = []
#             for o in range(out.shape[1]):
#                 grad, = torch.autograd.grad(out[:, o].sum(), x, retain_graph=o < out.shape[1] - 1)
#                 rows.append(grad.sum(-1))                               # summed over the lags
#             jacobians.append(torch.stack(rows, 1).double().cpu().numpy())
#     finally:
#         model.train(training)
#     jac_z = np.einsum("od,soc->sdc", w, np.concatenate(jacobians))     # [S, d, C]
#     maps = attribution_maps(jac_z, inputs["sigma_diff"], inputs["true_spatial_cov"])
#     record: Dict[str, Any] = dict(attr_points=len(jac_z), attr_map_r2=map_r2,
#                                   attr_linear_r2=lin_r2)
#     for key, m in maps.items():
#         record.update(binary_scores(m, inputs["gt"], key))
#     sv = np.linalg.svd(jac_z, compute_uv=False)
#     record["attr_sv_ratio"] = float(np.median(sv[:, -1] / sv[:, 0]))
#     return record


# # ============================================================================ checks and predictions
# def attribution_checks(root: Path, name: str) -> Dict[str, Any]:
#     """Noise covariance of the difference residuals (for inv_noise) and the surrogate scores."""
#     path = root / "sessions" / f"{name}.attr.json"
#     if path.is_file():
#         return json.loads(path.read_text(encoding="utf-8"))
#     from cebra.dprime import noise
#     (x, _, _, _), _, _ = sb.load_session(root, name)
#     model, _ = noise.fit_noise_model(x.astype(np.float64), None, window_len=sb.WINDOW,
#                                      residual_mode="difference", spatial_estimator="ledoit_wolf",
#                                      temporal_method="white", kronecker_channels=0)
#     with open(root / "sessions" / f"{name}.attr.npz", "wb") as f:
#         np.savez(f, sigma_diff=model.spatial_cov)
#     out = surrogate_checks(load_attribution_inputs(root, name))
#     runner.atomic_write_json(path, out)
#     return out


# HYPOTHESES = __doc__.split("Pre-registered in")[1].split("Suites:")[0].strip()


# def write_predictions(args, jobs: List["sb.Job"], checks: Dict[str, Any]) -> None:
#     path = args.output_root / "ATTRIBUTION_PREDICTIONS.json"
#     stored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
#         "created_utc": runner.now_utc(), "hypotheses": HYPOTHESES, "jobs": [],
#         "session_checks": {}, "registrations": []}
#     new = sorted({j.ident for j in jobs} - set(stored["jobs"]))
#     if not new:
#         return
#     stored["jobs"] += new
#     stored["session_checks"].update(checks)
#     stored["registrations"].append({
#         "registered_utc": runner.now_utc(), "jobs": len(new), "suite": args.suite,
#         "fork_fingerprints": runner._fingerprints(args.fork_dir, runner.FORK_EVAL_FILES)})
#     runner.atomic_write_json(path, json.loads(json.dumps(stored)))
#     print(f"Pre-registered {len(new)} jobs -> {path}", flush=True)


# # ============================================================================ worker and queue
# def worker(args) -> int:
#     job = sb.Job(**json.loads(args._job))
#     begin = time.monotonic()
#     base = dict(job=job.ident, session=job.session, mode=job.mode, seed=job.seed, arm=job.arm,
#                 status="error", started_utc=runner.now_utc(), error="")
#     try:
#         torch.set_num_threads(args.cpu_threads)
#         record = sb.run_job(args, job)  # training (or the saved model) and robustness
#         exp = sb.arm(job.arm)
#         cebra = runner.load_fork(args.fork_dir, exp, needs_eval_options=True)
#         arrays, _, _ = sb.load_session(args.output_root, job.session)
#         plan = runner.Plan(args.suite, job.mode, (exp,), (job.session,), (job.seed,),
#                            args.max_iterations, None, None, args.output_root)
#         run_root = args.output_root / "runs" / job.session / job.mode / f"seed{job.seed}"
#         encoder, _, _, _ = runner._restore_train(cebra, runner.Job(job.session, job.seed, exp,
#                                                                    run_root), plan, arrays,
#                                                  args.max_iterations)
#         record.update(model_attribution(encoder, arrays[1],
#                                         load_attribution_inputs(args.output_root, job.session),
#                                         args.attr_points))
#         base.update(record, status="success")
#         print(f"{job.ident}: AUROC inv {base['inv_auroc']:.3f} inv_noise "
#               f"{base['inv_noise_auroc']:.3f} (map R2 {base['attr_map_r2']:.3f})", flush=True)
#     except BaseException as exc:
#         base["error"] = f"{type(exc).__name__}: {exc}"
#         traceback.print_exc()
#     finally:
#         base["elapsed_seconds"] = round(time.monotonic() - begin, 1)
#         base["finished_utc"] = runner.now_utc()
#         runner.atomic_write_json(args.output_root / "records" / f"{job.ident}.json",
#                                  json.loads(json.dumps(base)))
#     return 0 if base["status"] == "success" else 1


# def worker_command(args, job) -> List[str]:
#     return [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker",
#             "--_job", json.dumps(job.__dict__), "--suite", args.suite,
#             "--output-root", str(args.output_root), "--fork-dir", str(args.fork_dir),
#             "--max-iterations", str(args.max_iterations), "--batch-size", str(args.batch_size),
#             "--mlp-epochs", str(args.mlp_epochs), "--device", args.device,
#             "--cpu-threads", str(args.cpu_threads), "--eval-steps", str(args.eval_steps),
#             "--eval-batch-size", str(args.eval_batch_size), "--eval-refine", str(args.eval_refine),
#             "--eval-restarts", str(args.eval_restarts), "--attr-points", str(args.attr_points),
#             "--eval-epsilons", *[str(e) for e in args.eval_epsilons]]


# # ============================================================================ summary
# def _session_diff(g, a: str, b: str, column: str, column_b: str = None):
#     """Paired difference a - b (same session and seed; per session means; session bootstrap)."""
#     if column_b is None:
#         piv = g.pivot_table(index=["session", "seed"], columns="arm", values=column)
#         if a not in piv or b not in piv:
#             return None
#         diff = (piv[a] - piv[b]).dropna()
#     else:  # two estimators of the same arm
#         rows = g[g.arm == a].set_index(["session", "seed"])
#         diff = (rows[column] - rows[column_b]).dropna()
#     if diff.empty:
#         return None
#     v = diff.groupby(level=0).mean().to_numpy()
#     boot = v[np.random.default_rng(2026).integers(0, len(v), size=(20000, len(v)))].mean(axis=1)
#     lo, hi = np.percentile(boot, [2.5, 97.5])
#     return dict(diff=float(v.mean()), lo=float(lo), hi=float(hi), up=int((v > 0).sum()),
#                 n=len(v))


# def _verdict(r) -> str:
#     if r is None:
#         return "no data"
#     if r["lo"] > 0 and r["up"] >= 0.75 * r["n"]:
#         return "SUPPORTED"
#     if r["hi"] < 0 and r["up"] <= 0.25 * r["n"]:
#         return "CONTRADICTED"
#     return "inconclusive"


# def summarize(args) -> int:
#     import pandas as pd
#     pd.set_option("display.width", 230)
#     pd.set_option("display.max_columns", 40)
#     pd.set_option("display.max_rows", 400)
#     root = args.output_root
#     rows = [json.loads(p.read_text()) for p in sorted((root / "records").glob("*.json"))]
#     if not rows:
#         print(f"No records in {root}")
#         return 1
#     d = pd.DataFrame(rows)
#     errors = d[d.status != "success"]
#     d = d[d.status == "success"].copy()
#     d.drop(columns=["true_curve"], errors="ignore").to_csv(root / "ATTRIBUTION_RUNS.csv",
#                                                             index=False)
#     print(f"{root}: {len(d)} successful runs, {len(errors)} errors; sessions {d.session.nunique()}")
#     checks = []
#     for name in sorted(d.session.unique()):
#         meta = json.loads((root / "sessions" / f"{name}.json").read_text())
#         checks.append(dict(session=name, alpha=meta["alpha"], **attribution_checks(root, name)))
#     print("\n0  exact surrogate encoders (E: minimum Euclidean norm, W: noise-whitened) scored "
#           "with each inverse: AUROC (expected 1 for E-inv and W-inv_true; W-inv_noise is the "
#           "ceiling of the practical inverse)")
#     print(pd.DataFrame(checks).round(3).to_string(index=False))
#     verdicts = []
#     for mode in ("supervised", "time_only"):
#         g = d[d["mode"] == mode]
#         if g.empty:
#             continue
#         auto = sb.AUTO[mode]
#         print(f"\n==================== {mode}")
#         table = g.groupby(["alpha", "arm"]).agg(
#             runs=("clean_r2", "size"), clean_r2=("clean_r2", "mean"),
#             map_r2=("attr_map_r2", "mean"), lin_r2=("attr_linear_r2", "mean"),
#             inv=("inv_auroc", "mean"), inv_noise=("inv_noise_auroc", "mean"),
#             inv_true=("inv_true_auroc", "mean"), fwd=("fwd_auroc", "mean"),
#             inv_col=("inv_auroc_col", "mean"), inv_noise_col=("inv_noise_auroc_col", "mean"),
#             auprc_inv=("inv_auprc", "mean"), auprc_noise=("inv_noise_auprc", "mean"),
#             true_half=("true_half", "median"))
#         print("1  attribution AUROC per estimator (mean over sessions), with the linear "
#               "identifiability (map / readout R2) and the robustness (true eps1/2)")
#         print(table.round(3).to_string())
#         print("\n2  paired AUROC differences (sessions = data draws; CI = session bootstrap)")
#         for alpha in sorted(g.alpha.unique()):
#             ga = g[g.alpha == alpha]
#             for est in ("inv", "inv_true", "inv_noise"):
#                 col = f"{est}_auroc"
#                 for a, b in [(x, "CLEAN") for x in ISOTROPIC + NOISE_AWARE + (auto,)] + [
#                         ("MADRY_SPDIFF_D8", "MADRY_ISO_D8"), (auto, "PGD_L2_EPS6"),
#                         (auto, "MADRY_SPDIFF_GAUSS")]:
#                     r = _session_diff(ga, a, b, col)
#                     if r:
#                         print(f"  alpha {alpha:<4g} {est:9s} {SHORT[a]:>8} - {SHORT[b]:<6} "
#                               f"{r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] sessions up "
#                               f"{r['up']}/{r['n']}")
#             for a in ISOTROPIC + NOISE_AWARE + (auto, "CLEAN"):
#                 for est in ("inv_true", "inv_noise"):
#                     r = _session_diff(ga, a, None, f"{est}_auroc", "inv_auroc")
#                     if r:
#                         print(f"  alpha {alpha:<4g} {SHORT[a]:>8}: {est} - inv {r['diff']:+.4f} "
#                               f"[{r['lo']:+.4f}, {r['hi']:+.4f}] sessions up {r['up']}/{r['n']}")
#         # HA1: matched inverse vs CLEAN (CLEAN read with the better of the two inverses).
#         g = g.copy()
#         g["matched"] = np.where(g.arm.isin(NOISE_AWARE + (auto,)), g.inv_true_auroc,
#                                 np.where(g.arm.isin(ISOTROPIC), g.inv_auroc,
#                                          np.maximum(g.inv_auroc, g.inv_true_auroc)))
#         for a in ISOTROPIC + NOISE_AWARE + (auto,):
#             r = _session_diff(g, a, "CLEAN", "matched")
#             verdicts.append((f"HA1 {mode} {SHORT[a]} > CLEAN (matched inverse, all alpha)",
#                              _verdict(r), r))
#         alphas = sorted(g.alpha.unique())
#         for alpha in [x for x in alphas if x > 0]:
#             ga = g[g.alpha == alpha]
#             for a, b in (("MADRY_SPDIFF_D8", "MADRY_ISO_D8"), (auto, "PGD_L2_EPS6")):
#                 for est in ("inv_true", "inv_noise"):
#                     r = _session_diff(ga, a, b, f"{est}_auroc")
#                     verdicts.append((f"HA2 {mode} alpha {alpha:g} {SHORT[a]} > {SHORT[b]} ({est}"
#                                      + ("" if est == "inv_true" else ", secondary") + ")",
#                                      _verdict(r), r))
#             for a in NOISE_AWARE + (auto,):
#                 r = _session_diff(ga, a, None, "inv_true_auroc", "inv_auroc")
#                 verdicts.append((f"HA3 {mode} alpha {alpha:g} {SHORT[a]}: inv_true > inv",
#                                  _verdict(r), r))
#             for a in ISOTROPIC:
#                 r = _session_diff(ga, a, None, "inv_auroc", "inv_true_auroc")
#                 ok = r is not None and r["diff"] >= 0
#                 verdicts.append((f"HA3 {mode} alpha {alpha:g} {SHORT[a]}: inv >= inv_true",
#                                  "SUPPORTED" if ok else ("no data" if r is None else
#                                                          "not supported"), r))
#         if 0.0 in alphas:
#             g0 = g[g.alpha == 0.0]
#             gap = (g0.inv_true_auroc - g0.inv_auroc).abs().max()
#             verdicts.append((f"HA2 {mode} alpha 0: the two inverses agree (max |diff| "
#                              f"{gap:.4f})", "SUPPORTED" if gap < 0.01 else "not supported", None))
#     if verdicts:
#         print("\n3  pre-registered hypotheses")
#         for label, verdict, r in verdicts:
#             detail = (f"  diff {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}], sessions "
#                       f"{r['up']}/{r['n']}") if r else ""
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
#     p.add_argument("--arms", nargs="+", default=None, help="Overrides the arms")
#     p.add_argument("--output-root", type=Path, default=None)
#     p.add_argument("--fork-dir", type=Path, default=runner.DEFAULT_FORK_DIR)
#     for field_name, value in asdict(AttrDGP()).items():
#         p.add_argument("--" + field_name.replace("_", "-"), type=type(value), default=value,
#                        help=f"DGP (default {value})")
#     p.add_argument("--attr-points", type=int, default=1000,
#                    help="Held-out time points of the attribution maps")
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
#     p.add_argument("--dry-run", action="store_true", help="Sessions and checks only")
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
#     args.output_root = (args.output_root or ROOT / "SYNTH_ATTRIBUTION").resolve()
#     args.fork_dir = args.fork_dir.resolve()
#     if args._worker:
#         return worker(args)
#     runner.load_fork(args.fork_dir)
#     if args.summarize:
#         return summarize(args)
#     dgp = AttrDGP(**{k: getattr(args, k) for k in asdict(AttrDGP())})
#     jobs = []
#     for alpha in args.alphas:
#         for rep in args.reps:
#             for mode in args.modes:
#                 arms = [a for a in (args.arms or sb.MAIN_ARMS[mode])
#                         if mode == "supervised" or not sb.arm(a).supervised_only]
#                 jobs += [sb.Job(session_name(alpha, rep), mode, s, a) for s in args.seeds
#                          for a in arms]
#     sessions = ensure_sessions(dgp, args.alphas, args.reps, args.output_root)
#     checks = {}
#     for name, meta in sessions.items():
#         c = sb.session_checks(args.output_root, name)
#         checks[name] = attribution_checks(args.output_root, name)
#         s = checks[name]
#         print(f"{name}: eps_beh true {meta['eps_beh_true']:.2f} (DIFF est {c['eps_beh_diff']:.2f}) "
#               f"| surrogate AUROC E (min-norm): inv {s['surrogate_E_inv']:.3f} inv_true "
#               f"{s['surrogate_E_inv_true']:.3f} | W (whitened): inv {s['surrogate_W_inv']:.3f} "
#               f"inv_true {s['surrogate_W_inv_true']:.3f} inv_noise {s['surrogate_W_inv_noise']:.3f}",
#               flush=True)
#     write_predictions(args, jobs, checks)
#     print(f"{len(jobs)} jobs: {len(sessions)} sessions x {len(args.modes)} modes x arms",
#           flush=True)
#     if args.dry_run:
#         return 0
#     sb.worker_command = lambda a, j: worker_command(a, j)  # the queue runs this file's worker
#     code = sb.orchestrate(args, jobs)
#     summarize(args)
#     return code


# if __name__ == "__main__":
#     raise SystemExit(main())

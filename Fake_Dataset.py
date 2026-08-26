#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#  Fig-5-style synthetic attribution benchmark
#
#  Reproduces the data-generating process of
#    Schneider, Gonzalez Laiz, Filippova, Frey, Mathis,
#    "Time-series attribution maps with regularized contrastive learning",
#    AISTATS 2025, arXiv:2502.12977 -- Figure 5 / Appendix B.1
#  and asks whether PGD adversarial training acts as an IMPLICIT substitute
#  for the paper's explicit Jacobian-Frobenius regularizer.
#
#  ARMS (6):
#    cebra       1 clean update / iter                    lambda = 0
#    cebra_2x    2 clean updates / iter (SAME batch)      lambda = 0   <- control
#    xcebra      1 clean update / iter                    lambda = 0.1
#    xcebra_2x   2 clean updates / iter (SAME batch)      lambda = 0.1 <- control
#    acorn       clean update THEN adversarial update     lambda = 0
#    acorn_xreg  clean update THEN adversarial update     lambda = 0.1
#
#  WHY THE *_2x CONTROLS EXIST.
#    cebra/solver/base.py::Solver.step runs `self.optimizer.step()`
#    unconditionally, and the adversarial branch then runs a SECOND
#    `self.optimizer.step()` on the same batch.  That is the algorithm as
#    written -- not a bug to be silently removed -- but it means the
#    adversarial arm receives 2x the parameter updates.  Removing the double
#    update would no longer measure the method; keeping it silently would
#    confound "adversarial" with "twice the optimisation".  So the double
#    update is kept EXACTLY as in the fork, and a clean arm with the same
#    doubled update count is added.  The pre-registered primary comparison is
#    acorn vs cebra_2x (compute matched); acorn vs cebra is reported as the
#    "as-published" secondary.
#
#  WHAT IS TAKEN FROM THE FORK (unmodified):
#    cebra.models.init(...)      -- the encoder architectures
#    cebra.models.criterions.*   -- InfoNCE with learnable temperature
#    cebra.attribution.init(...) -- the "neuron gradient" Jacobian J_f
#
#  WHAT IS IMPLEMENTED HERE:
#    * lambda * ||J_f(x)||_F^2   -- paper Eq. 10/15, Hoffman et al. (2019)
#                                   random-projection estimator (+ exact mode
#                                   + an unbiasedness self-test)
#    * lambda schedule           -- 0 for 2500 steps, linear ramp over the next
#                                   2500, constant to 20000
#    * PGD-linf and PGD-l2       -- LITERAL transcription of Solver.step,
#                                   including _l2_normalize / _rand_radius_like
#                                   / _proj_l2_ball, the uniform eps-cube init,
#                                   the per-inner-step re-encoding of
#                                   positive/negative, the raw (non-Madry)
#                                   alpha, and the ABSENCE of any clamp to the
#                                   valid data range
#    * the alignment metric + its proof-carrying validation test
#
#  METRIC (and why it is the only one with a valid ground truth):
#    `gt` is the support of the GENERATOR Jacobian dx/dz, which is unique.
#    dz/dx is NOT unique for an over-determined system (50 neurons, d
#    latents): the encoder may legitimately use x1 to cancel z1's contribution
#    to x2, so a nonzero dz2/dx1 is compatible with a perfect encoder.
#    Forward-direction scores therefore have no well-defined target.
#
#    Let  Q = pinv(J_f) in R^{CxO}  (the paper's "inverted neuron gradient")
#    and  A in R^{OxD}  the linear map f ~ A z fitted on a held-out segment.
#    Score  R = Q A in R^{CxD}  estimates dx_i/dz_d, and is EXACTLY invariant
#    to the linear indeterminacy of contrastive learning: if f' = M f then
#    Q' = Q M^{-1} and A' = M A, so R' = Q M^{-1} M A = R.
#    validate_alignment_metric() checks this end-to-end on a known-perfect
#    analytic encoder and aborts the run if auROC < SURROGATE_AUROC_MIN.
# ==============================================================================

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------------------------

DRY_RUN = True                # <<< set False for the real run
EXPERIMENT_SCALE = "pilot"    # "dry" | "pilot" | "paper_final" | "paper_table1"

# --- architecture -------------------------------------------------------------
# "euclidean_linear": offset1-model-mse (normalize=False) + Euclidean InfoNCE.
#     Theory-matched: Brownian positives are conditionally Gaussian, so the
#     Euclidean similarity is the identifiable choice (Zimmermann et al. 2021),
#     and the paper's encoder likewise ends in a linear / scaled-tanh head, not
#     an L2 normalisation.  J_f is full rank -> no pinv rank truncation.
# "cosine_sphere":    offset1-model (normalize=True) + cosine InfoNCE.
#     Reproduces the production ACORN setting.  J_f = (1/|u|)(I - f f^T) A has
#     rank <= OUTPUT_DIM - 1, so OUTPUT_DIM must exceed d and one singular
#     value must be truncated.  Report as an ablation.
ARCH_VARIANT = "euclidean_linear"

_ARCH = {
    "euclidean_linear": dict(model="offset1-model-mse", normalize=False,
                             criterion="euclidean", out_extra=0, pinv_drop=0),
    "cosine_sphere":    dict(model="offset1-model",     normalize=True,
                             criterion="cosine",        out_extra=1, pinv_drop=1),
}[ARCH_VARIANT]

MODEL_NAME = _ARCH["model"]
CRITERION  = _ARCH["criterion"]
OUT_EXTRA  = _ARCH["out_extra"]     # OUTPUT_DIM = d + OUT_EXTRA
PINV_DROP  = _ARCH["pinv_drop"]     # singular values discarded in the pinv
NUM_UNITS  = 128

# --- data generating process (Figure 5 / Appendix B.1) ------------------------
# "We sample 10 different datasets with 100,000 samples, each with a different
#  mixing function g.  All latents [...] lie within the box [-1,1]^D.  We
#  sample z1 from a uniform distribution over [-1,1]^D.  The following time
#  steps are generated by Brownian motion, z_t = N_[-1,1](z_{t-1}, sigma^2 I)
#  where N_[-1,1] is a truncated normal distribution clipped to the bounds."
D_LATENT_LIST   = [6]        # Table 1 averages over d = 4..9
N1, N2          = 25, 25     # "g1 [...] outputs 25 neurons [...] g2 [...] 25"
D_OBS           = N1 + N2    # -> x is (T, 50)
BROWNIAN_SIGMA  = 0.10
TARGET_PRE_SD   = 0.80       # keeps tanh off its saturated tails
SIGMA_OBS_SWEEP = [0.00]     # paper adds no observation noise
TIME_OFFSET     = 1

# Left OFF by default: tanh output already lives in [-1,1] with roughly zero
# mean, so ADV_EPSILON below is in exactly the same units as the advisor's runs
# on raw data.  Turning it on rescales the eps-ball and breaks that comparison.
STANDARDIZE = False

# --- optimisation (Appendix B.1) ---------------------------------------------
# "We train on batches with 5,000 samples each.  The first 2,500 training steps
#  minimize the InfoNCE or supervised loss with lambda = 0; we then ramp up
#  lambda to its maximum value over the following 2,500 steps, and continue to
#  train until 20,000 total steps."
LEARNING_RATE    = 3e-4
MIN_TEMPERATURE  = 0.05
INIT_TEMPERATURE = 1.0

# --- the regularizer (paper Eq. 10 / 15) -------------------------------------
LAMBDA_MAX     = 0.10      # "Regularization: Off (lambda = 0), On (lambda = 0.1)"
JREG_ESTIMATOR = "proj"    # "proj" (Hoffman et al. 2019) | "exact"
JREG_NPROJ     = 1         # Hoffman et al. recommend 1
JREG_SUBBATCH  = 512       # unbiased sub-sample of the batch for the penalty
JREG_AT        = "clean"   # "clean" | "adv" -- "clean" keeps the penalty
                           # identical across rows of the design
JREG_ON        = "both"    # "both" | "clean_only" | "adv_only": which of the
                           # two updates in a doubled scheme carries the penalty

# --- PGD: LITERAL transcription of cebra/solver/base.py::Solver.step ---------
ATTACK_NORM     = "linf"   # "linf" | "l2"  (solver's `attack_norm`)
ADV_EPSILON     = 0.05     # solver default `adv_epsilon`
ADV_ALPHA_RULE  = "fork"   # "fork" -> ADV_ALPHA_RAW (the solver default)
                           # "madry" -> 2.5*eps/steps  (ablation only)
ADV_ALPHA_RAW   = 0.01     # solver default `adv_alpha`
ADV_STEPS       = 10       # solver default `adv_steps`
ADV_CLIP_RANGE  = False    # the fork does NOT clamp to a valid data range;
                           # only the eps-ball projection is applied
ADV_CACHE_POS_NEG = False  # False = re-encode positive/negative at every inner
                           # step, exactly as _inference does.  True is
                           # mathematically identical (parameters are frozen
                           # during the attack) and ~2x faster.
ADV_EPSILON_SWEEP = [0.05]         # e.g. [0.025, 0.05, 0.10, 0.20]

# --- attribution -------------------------------------------------------------
# The library supplies J_f ("neuron gradient").  The pseudo-inverse is taken
# here so the rank truncation is under our control -- with
# ARCH_VARIANT="cosine_sphere" an untruncated pinv amplifies the structurally
# null direction and destroys the map.  A library-provided jf-inv, if present,
# is cross-checked and reported but not used for the primary number.
ATTR_METHOD_CANDIDATES = ["jacobian-based-batched", "jacobian-based",
                          "neuron-gradient"]
ATTR_NUM_BATCHES = 32

# --- quality gates -----------------------------------------------------------
# "We compute the R2 for predicting the auxiliary variable c from the feature
#  space after a linear regression, and ensure that this metric is close to
#  100% for both our baseline and contrastive learning models to remove
#  performance as a potential confounder."
R2_MIN              = 0.95     # per-arm floor on mean R2(z <- f)
R2_PARITY_MAX       = 0.05     # max spread of R2 across arms within a seed
SURROGATE_AUROC_MIN = 0.995
JREG_SELFTEST_RTOL  = 0.15

# --- arms --------------------------------------------------------------------
# scheme: "clean_single" | "clean_double" | "fork_double" | "adv_single"
ALL_ARMS = {
    "cebra":       dict(scheme="clean_single", lam=0.0),
    "cebra_2x":    dict(scheme="clean_double", lam=0.0),
    "xcebra":      dict(scheme="clean_single", lam=LAMBDA_MAX),
    "xcebra_2x":   dict(scheme="clean_double", lam=LAMBDA_MAX),
    "acorn":       dict(scheme="fork_double",  lam=0.0),
    "acorn_xreg":  dict(scheme="fork_double",  lam=LAMBDA_MAX),
    # ablation: the adversarial update WITHOUT the preceding clean update
    "acorn_1x":    dict(scheme="adv_single",   lam=0.0),
}
ARM_SUBSET = ["cebra", "cebra_2x", "xcebra", "xcebra_2x", "acorn", "acorn_xreg"]
ARMS = {k: ALL_ARMS[k] for k in ARM_SUBSET}

# --- reporting ---------------------------------------------------------------
PRIMARY_METRIC     = "auroc_global"
PRIMARY_COMPARISON = ("acorn", "cebra_2x")      # compute-matched
SECONDARY_COMPARISONS = [("acorn", "cebra"),          # the as-published claim
                         ("cebra_2x", "cebra"),       # size of the confound
                         ("xcebra", "cebra"),         # the paper's own claim
                         ("acorn", "xcebra_2x"),
                         ("acorn_xreg", "acorn")]
N_BOOTSTRAP = 1000     # "95% CI obtained through bootstrapping (n=1,000)"
RESULT_CSV  = "attribution_benchmark_results.csv"
SEED0       = 1234

# --- scale presets -----------------------------------------------------------
_SCALES = {
    "dry":         dict(n_seeds=2,  T=8_000,   iters=300,    batch=256,
                        map_pts=1500, attr_pts=1500, d_list=[6]),
    "pilot":       dict(n_seeds=3,  T=40_000,  iters=4_000,  batch=1024,
                        map_pts=4000, attr_pts=4000, d_list=[6]),
    "paper_final": dict(n_seeds=10, T=100_000, iters=20_000, batch=5_000,
                        map_pts=8000, attr_pts=8000, d_list=[6]),
    "paper_table1":dict(n_seeds=10, T=100_000, iters=20_000, batch=5_000,
                        map_pts=8000, attr_pts=8000, d_list=[4, 5, 6, 7, 8, 9]),
}
if DRY_RUN:
    EXPERIMENT_SCALE = "dry"
_S = _SCALES[EXPERIMENT_SCALE]
N_SEEDS, T_SAMPLES, MAX_ITER = _S["n_seeds"], _S["T"], _S["iters"]
BATCH_SIZE, MAP_POINTS, ATTR_POINTS = _S["batch"], _S["map_pts"], _S["attr_pts"]
if EXPERIMENT_SCALE == "paper_table1":
    D_LATENT_LIST = _S["d_list"]

# the lambda schedule keeps the paper's 2500/2500-of-20000 shape at any length
LAMBDA_WARMUP = max(1, int(round(MAX_ITER * 2500 / 20000)))
LAMBDA_RAMP   = max(1, int(round(MAX_ITER * 2500 / 20000)))

# ------------------------------------------------------------------------------
# 2. ENVIRONMENT
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

try:
    from scipy import stats as _sps
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    warnings.warn("scipy missing: Wilcoxon replaced by a sign test.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_cebra():
    try:
        import cebra
        import cebra.models
        import cebra.models.criterions as _crit
    except Exception as exc:
        raise SystemExit("cannot import cebra -- run from an environment where "
                         f"your fork is on sys.path.  error: {exc}")
    return cebra, _crit


CEBRA, CRIT = _load_cebra()


def _banner():
    print("=" * 78)
    print(f" scale={EXPERIMENT_SCALE}  arch={ARCH_VARIANT}  model={MODEL_NAME}")
    print(f" seeds={N_SEEDS}  T={T_SAMPLES}  iters={MAX_ITER}  batch={BATCH_SIZE}")
    print(f" d_list={D_LATENT_LIST}  device={DEVICE}  standardize={STANDARDIZE}")
    print(f" lambda_max={LAMBDA_MAX} (warmup {LAMBDA_WARMUP} / ramp {LAMBDA_RAMP})"
          f"  jreg={JREG_ESTIMATOR}/{JREG_NPROJ} sub={JREG_SUBBATCH}"
          f" at={JREG_AT} on={JREG_ON}")
    print(f" attack={ATTACK_NORM} eps={ADV_EPSILON} alpha={adv_alpha():.4g}"
          f" ({ADV_ALPHA_RULE}) steps={ADV_STEPS} clip={ADV_CLIP_RANGE}"
          f" cache_pos_neg={ADV_CACHE_POS_NEG}")
    print(f" arms={list(ARMS)}")
    print(f" PRIMARY: {PRIMARY_COMPARISON[0]} vs {PRIMARY_COMPARISON[1]}"
          f"  on {PRIMARY_METRIC}")
    print(f" total model fits = "
          f"{N_SEEDS * len(D_LATENT_LIST) * len(ARMS) * len(SIGMA_OBS_SWEEP)}")
    print("=" * 78)


# ------------------------------------------------------------------------------
# 3. DATA GENERATING PROCESS
# ------------------------------------------------------------------------------

def brownian_motion_box(T: int, d: int, sigma: float, rng) -> np.ndarray:
    """z_1 ~ U[-1,1]^d ; z_t ~ N_[-1,1](z_{t-1}, sigma^2 I), TRUNCATED (clipped).

    The paper clips to the box rather than reflecting off it; clipping leaves a
    little probability mass on the faces of the cube, reflecting does not."""
    z = np.empty((T, d), dtype=np.float64)
    z[0] = rng.uniform(-1.0, 1.0, size=d)
    for t in range(1, T):
        z[t] = np.clip(z[t - 1] + sigma * rng.normal(size=d), -1.0, 1.0)
    return z


def orthonormal_columns(rows: int, cols: int, rng) -> np.ndarray:
    """Dense [rows, cols] with orthonormal columns (reduced QR).

    Orthonormality keeps the mixing well-conditioned so the generator Jacobian
    has no near-zero entries that would blur the ground truth; the Gaussian
    seed makes it dense almost surely, so every declared edge in `gt` is real."""
    assert rows >= cols
    q, _ = np.linalg.qr(rng.normal(size=(rows, cols)))
    return q[:, :cols]


@dataclass
class Mixer:
    """x1 = tanh(g1 W1 z[:d1]) (N1 neurons) ; x2 = tanh(g2 W2 z) (N2 neurons).

    Figure 5: "z1 is connected to both x1 and x2, while z2 is connected only to
    x2 [...] g1 takes 3 (d1) latent variables as input and outputs 25 neurons
    (n1), whereas g2 takes 6 (d1+d2) latent variables as input and outputs 25
    neurons (n2)."  """
    W1: np.ndarray            # [N1, d1]
    W2: np.ndarray            # [N2, d]
    d1: int
    gain1: float = 1.0
    gain2: float = 1.0

    def pre(self, z):
        return (self.gain1 * z[:, :self.d1] @ self.W1.T,
                self.gain2 * z @ self.W2.T)

    def __call__(self, z):
        p1, p2 = self.pre(z)
        return np.concatenate([np.tanh(p1), np.tanh(p2)], axis=1)

    def generator_jacobian(self, z) -> np.ndarray:
        """dx/dz, [S, C, D].  Unique -- this is what `gt` is derived from."""
        S, d = z.shape
        p1, p2 = self.pre(z)
        s1 = 1.0 - np.tanh(p1) ** 2
        s2 = 1.0 - np.tanh(p2) ** 2
        n1 = self.W1.shape[0]
        J = np.zeros((S, n1 + self.W2.shape[0], d))
        J[:, :n1, :self.d1] = s1[:, :, None] * (self.gain1 * self.W1)[None]
        J[:, n1:, :]        = s2[:, :, None] * (self.gain2 * self.W2)[None]
        return J


def generate_dataset(d: int, sigma_obs: float, data_seed: int) -> dict:
    rng = np.random.default_rng(data_seed)
    d1 = d // 2 + d % 2
    z = brownian_motion_box(T_SAMPLES, d, BROWNIAN_SIGMA, rng)
    W1 = orthonormal_columns(N1, d1, rng)
    W2 = orthonormal_columns(N2, d, rng)
    mix = Mixer(W1=W1, W2=W2, d1=d1)
    sub = z[rng.choice(T_SAMPLES, size=min(5000, T_SAMPLES), replace=False)]
    mix.gain1 = TARGET_PRE_SD / max(float(np.std(sub[:, :d1] @ W1.T)), 1e-12)
    mix.gain2 = TARGET_PRE_SD / max(float(np.std(sub @ W2.T)), 1e-12)
    x = mix(z)
    if sigma_obs > 0:
        x = x + sigma_obs * rng.normal(size=x.shape)
    return dict(z=z, x=x, mix=mix, d=d, d1=d1, d2=d - d1, seed=data_seed)


def ground_truth(d: int, d1: int) -> np.ndarray:
    """gt[C, D] = 1 iff neuron i is generated from latent j.  Support of dx/dz."""
    gt = np.zeros((D_OBS, d), dtype=np.int8)
    gt[:N1, :d1] = 1        # x1 sees z1 only
    gt[N1:, :] = 1          # x2 sees z1 and z2
    return gt


def split_data(data: dict) -> dict:
    """Disjoint train / map-fit / attribution segments."""
    T = T_SAMPLES
    assert MAP_POINTS + ATTR_POINTS < T // 2, "held-out segments too large"
    train_end = T - (MAP_POINTS + ATTR_POINTS) - TIME_OFFSET - 1
    idx_train = np.arange(0, train_end)
    idx_map   = np.arange(train_end, train_end + MAP_POINTS)
    idx_attr  = np.arange(train_end + MAP_POINTS,
                          train_end + MAP_POINTS + ATTR_POINTS)
    if STANDARDIZE:
        mu = data["x"][idx_train].mean(0, keepdims=True)
        sd = data["x"][idx_train].std(0, keepdims=True) + 1e-8
    else:
        mu = np.zeros((1, D_OBS))
        sd = np.ones((1, D_OBS))
    xs = (data["x"] - mu) / sd     # diagonal rescale: zeros of dx/dz unchanged
    return dict(xs=xs, mu=mu, sd=sd, idx_train=idx_train,
                idx_map=idx_map, idx_attr=idx_attr)


# ------------------------------------------------------------------------------
# 4. METRIC MACHINERY
# ------------------------------------------------------------------------------

def squeeze_to_3d(a: np.ndarray, name: str) -> np.ndarray:
    """Collapse trailing singleton axes ONLY.  Never average over a real axis:
    a signed mean over a lag axis cancels opposite-sign entries and turns a
    genuine edge into a zero."""
    a = np.asarray(a)
    while a.ndim > 3:
        if a.shape[-1] == 1:
            a = a[..., 0]
        else:
            raise ValueError(f"{name}: non-singleton extra axis {a.shape}")
    if a.ndim != 3:
        raise ValueError(f"{name}: expected 3 dims, got {a.shape}")
    return a


def canonicalize_jf(jf: np.ndarray, out_dim: int, name: str) -> np.ndarray:
    """-> [S, O, C]."""
    jf = squeeze_to_3d(jf, name)
    if jf.shape[1] == out_dim and jf.shape[2] == D_OBS:
        return jf
    if jf.shape[2] == out_dim and jf.shape[1] == D_OBS:
        return np.swapaxes(jf, 1, 2)
    raise ValueError(f"{name}: cannot orient {jf.shape} to [S,{out_dim},{D_OBS}]")


def truncated_pinv(jf: np.ndarray, drop: int) -> np.ndarray:
    """Batched pinv of J_f with rank truncation.  [S,O,C] -> [S,C,O].

    With an L2-normalised head, J_f = (1/|u|)(I - f f^T) A is rank deficient by
    exactly one; keeping that direction lets the pinv divide by a numerically
    zero singular value and blow up an arbitrary direction."""
    U, S, Vh = np.linalg.svd(jf, full_matrices=False)
    S = S.copy()
    if drop > 0:
        S[:, -drop:] = 0.0
    inv = np.zeros_like(S)
    good = S > np.maximum(1e-10 * S[:, :1], 1e-30)
    inv[good] = 1.0 / S[good]
    return np.einsum("ski,sk,sok->sio", Vh, inv, U)      # V diag(1/s) U^T


def fit_linear_maps(f: np.ndarray, z: np.ndarray, ridge: float = 1e-6):
    """A_z2e [O,D] with f ~ A z ; B_e2z [O,D] with z ~ f B ; R2 per latent dim.

    A_z2e is what the alignment needs (df/dz).  B_e2z gives the paper's "R2 for
    predicting the auxiliary variable from the feature space"."""
    f = np.asarray(f, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    fc = f - f.mean(0, keepdims=True)
    zc = z - z.mean(0, keepdims=True)
    O, D = f.shape[1], z.shape[1]
    G = zc.T @ zc
    G = G + ridge * np.trace(G) / max(D, 1) * np.eye(D)
    A_z2e = np.linalg.solve(G, zc.T @ fc).T                  # [O, D]
    H = fc.T @ fc
    H = H + ridge * np.trace(H) / max(O, 1) * np.eye(O)
    B_e2z = np.linalg.solve(H, fc.T @ zc)                    # [O, D]
    ss_res = ((zc - fc @ B_e2z) ** 2).sum(0)
    r2 = 1.0 - ss_res / ((zc ** 2).sum(0) + 1e-30)
    return A_z2e, B_e2z, r2


def align_to_latents(Q: np.ndarray, A_z2e: np.ndarray) -> np.ndarray:
    """R[s,i,d] = dx_i/dz_d = sum_o Q[s,i,o] A_z2e[o,d].  [S,C,O] -> [S,C,D].
    Exactly invariant to f -> M f: Q -> Q M^{-1}, A -> M A."""
    return np.einsum("sio,od->sid", Q, A_z2e)


def aggregate_map(R: np.ndarray) -> np.ndarray:
    """[S,C,D] -> [C,D].  ABSOLUTE value BEFORE the mean: a signed average over
    timepoints cancels a real edge whose sign flips along the trajectory."""
    return np.abs(R).mean(axis=0)


def score_variants(M: np.ndarray) -> Dict[str, np.ndarray]:
    """Global z-score (primary; monotone, hence auROC-equivalent to raw) plus a
    per-latent-column normalisation, which is NOT auROC-equivalent."""
    return {"global": (M - M.mean()) / (M.std() + 1e-30),
            "colnorm": (M - M.mean(0, keepdims=True)) /
                       (M.std(0, keepdims=True) + 1e-30)}


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(s)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    if y.sum() == 0:
        return float("nan")
    y = y[np.argsort(-s, kind="mergesort")]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def binary_scores(M: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    out = {}
    for key, s in score_variants(M).items():
        out[f"auroc_{key}"] = _auc(s, gt)
        out[f"auprc_{key}"] = _auprc(s, gt)
    return out


# ------------------------------------------------------------------------------
# 5. VALIDATION OF THE METRIC ITSELF
# ------------------------------------------------------------------------------

def validate_alignment_metric(data: dict, sp: dict, gt: np.ndarray, seed: int):
    """End-to-end test of the PRIMARY scoring path on a known-perfect encoder.

    Builds the exact composition of the true inverse mixing with a random
    linear reparametrisation, plus whatever head ARCH_VARIANT prescribes.  If
    the pipeline is correct auROC must be ~1.  Failure here means the METRIC is
    broken, not the model, so the run aborts rather than emitting numbers
    nobody can interpret."""
    rng = np.random.default_rng(seed)
    d = data["d"]
    O = d + OUT_EXTRA
    z = data["z"][sp["idx_attr"]]                          # [S, d]
    Jg = data["mix"].generator_jacobian(z) / sp["sd"][0][None, :, None]
    Jzx = np.linalg.pinv(Jg)                               # [S, d, C] = dz/dx
    A = rng.normal(size=(O, d))                            # the indeterminacy
    if _ARCH["normalize"]:
        u = z @ A.T
        nrm = np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
        f = u / nrm
        I = np.eye(O)
        dfdz = (I[None] - f[:, :, None] * f[:, None, :]) @ A[None] / nrm[:, :, None]
    else:
        f = z @ A.T
        dfdz = np.broadcast_to(A[None], (len(z), O, d))
    jf = dfdz @ Jzx                                        # [S, O, C] = df/dx
    A_z2e, _, r2 = fit_linear_maps(f, z)
    Q = truncated_pinv(jf, PINV_DROP)
    res = binary_scores(aggregate_map(align_to_latents(Q, A_z2e)), gt)
    print(f"    [validate] surrogate auROC={res['auroc_global']:.4f} "
          f"auPRC={res['auprc_global']:.4f} R2min={r2.min():.4f}")
    if not (res["auroc_global"] >= SURROGATE_AUROC_MIN):
        raise RuntimeError(
            f"alignment metric failed its own sanity check "
            f"(auROC={res['auroc_global']:.4f} < {SURROGATE_AUROC_MIN}). "
            f"Do not trust any model comparison until this passes.")
    return res


# ------------------------------------------------------------------------------
# 6. MODEL, CRITERION, SAMPLER
# ------------------------------------------------------------------------------

def build_model(d: int) -> Tuple[nn.Module, nn.Module]:
    model = CEBRA.models.init(MODEL_NAME, D_OBS, NUM_UNITS,
                              d + OUT_EXTRA).to(DEVICE)
    if len(model.get_offset()) != 1:
        raise RuntimeError(
            f"{MODEL_NAME} has receptive field {len(model.get_offset())}; this "
            f"benchmark assumes an instantaneous (offset-1) encoder so that "
            f"J_f is a plain [O, C] matrix per timepoint.")
    if bool(getattr(model, "normalize", None)) != _ARCH["normalize"]:
        raise RuntimeError(
            f"{MODEL_NAME}.normalize={getattr(model, 'normalize', None)} but "
            f"ARCH_VARIANT={ARCH_VARIANT} expects {_ARCH['normalize']}")
    if CRITERION == "cosine":
        crit = CRIT.LearnableCosineInfoNCE(temperature=INIT_TEMPERATURE,
                                           min_temperature=MIN_TEMPERATURE)
    else:
        crit = CRIT.LearnableEuclideanInfoNCE(temperature=INIT_TEMPERATURE,
                                              min_temperature=MIN_TEMPERATURE)
    return model, crit.to(DEVICE)


def flat(e: torch.Tensor) -> torch.Tensor:
    return e if e.dim() == 2 else e.reshape(e.shape[0], -1)


class TimeSampler:
    """CEBRA's `time` conditional.  Returns 3D [B, C, 1] tensors, the same shape
    cebra.data loaders hand to Solver.step, so the attack code below operates on
    exactly the tensor layout the fork perturbs."""

    def __init__(self, X: torch.Tensor, idx_train: np.ndarray, gen):
        self.X = X                                    # [T, C]
        self.base = int(idx_train[0])
        self.hi = len(idx_train) - TIME_OFFSET - 1
        self.gen = gen

    def __call__(self, batch: int):
        i = torch.randint(0, self.hi, (batch,), generator=self.gen,
                          device=self.X.device) + self.base
        j = torch.randint(0, self.hi, (batch,), generator=self.gen,
                          device=self.X.device) + self.base
        return (self.X[i].unsqueeze(-1),
                self.X[i + TIME_OFFSET].unsqueeze(-1),
                self.X[j].unsqueeze(-1))


# ------------------------------------------------------------------------------
# 7. THE JACOBIAN-FROBENIUS REGULARIZER  (paper Eq. 10 / 15)
# ------------------------------------------------------------------------------

def jacobian_frobenius_sq(model: nn.Module, x: torch.Tensor,
                          estimator: str = "proj", n_proj: int = 1
                          ) -> torch.Tensor:
    """E_x ||J_f(x)||_F^2, differentiable w.r.t. the model parameters.

    exact: O backward passes, sum_o ||d f_o / d x||^2.
    proj : Hoffman et al. (2019).  For v uniform on the unit sphere in R^O,
           E_v ||v^T J||^2 = tr(J J^T)/O = ||J||_F^2 / O, so
           (O / n_proj) * sum_mu ||d (v_mu . f) / d x||^2 is unbiased.  One
           projection replaces O backward passes -- this is what makes the
           penalty affordable at batch 5000."""
    x = x.detach().requires_grad_(True)
    out = flat(model(x))
    O = out.shape[1]
    total = 0.0
    if estimator == "exact":
        for o in range(O):
            g, = torch.autograd.grad(out[:, o].sum(), x,
                                     create_graph=True, retain_graph=True)
            total = total + (g ** 2).flatten(1).sum(1)
        return total.mean()
    for _ in range(n_proj):
        v = torch.randn(out.shape, device=out.device, dtype=out.dtype)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
        g, = torch.autograd.grad((out * v).sum(), x,
                                 create_graph=True, retain_graph=True)
        total = total + (g ** 2).flatten(1).sum(1)
    return (O / n_proj) * total.mean()


_JREG_TESTED = {"done": False}


def selftest_jreg(model: nn.Module, x: torch.Tensor):
    """The projection estimator is only useful if it is unbiased for THIS
    model.  Compare against the exact value on a small batch."""
    if _JREG_TESTED["done"]:
        return
    xs = x[:64]
    exact = float(jacobian_frobenius_sq(model, xs, "exact"))
    est = float(np.mean([float(jacobian_frobenius_sq(model, xs, "proj", 8))
                         for _ in range(24)]))
    rel = abs(est - exact) / max(exact, 1e-12)
    print(f"    [validate] ||J||_F^2 exact={exact:.5f} proj={est:.5f} "
          f"rel.err={rel:.3f}")
    if rel > JREG_SELFTEST_RTOL:
        raise RuntimeError(
            f"Hoffman projection estimator biased by {rel:.1%} -- refusing to "
            f"run a regularized arm on an untrustworthy penalty.")
    _JREG_TESTED["done"] = True


def lambda_at(it: int, lam_max: float) -> float:
    """0 for LAMBDA_WARMUP iterations, then a linear ramp over LAMBDA_RAMP.

    "The first 2,500 training steps minimize the InfoNCE [...] loss with
     lambda = 0; we then ramp up lambda to its maximum value over the following
     2,500 steps, and continue to train until 20,000 total steps."

    The counter advances per ITERATION, not per optimizer step, so lambda
    reaches its maximum at the same point in the run for single- and
    double-update arms."""
    if lam_max <= 0.0:
        return 0.0
    if it < LAMBDA_WARMUP:
        return 0.0
    return lam_max * min(1.0, max(0.0, (it - LAMBDA_WARMUP) / float(LAMBDA_RAMP)))


# ==============================================================================
# 8. THE ATTACK -- literal transcription of cebra/solver/base.py::Solver.step
# ==============================================================================
# The three helpers below are copied verbatim from the fork so that the l2
# branch is bit-for-bit the same geometry.

def _l2_normalize(t: torch.Tensor, eps: float = 1e-12):
    """Per-sample L2 normalisation (zero vectors stay zero)."""
    flat_ = t.reshape(t.size(0), -1)
    norm = flat_.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
    return t / norm.view(-1, *([1] * (t.dim() - 1)))


def _rand_radius_like(t: torch.Tensor):
    """U(0,1) radius shaped like t but broadcastable (B,1,1,...)."""
    return torch.rand([t.size(0)] + [1] * (t.dim() - 1), device=t.device)


def _proj_l2_ball(adv: torch.Tensor, orig: torch.Tensor, epsilon: float):
    """Project adv back to the closed L2 ball of radius eps around orig."""
    delta = adv - orig
    flat_ = delta.reshape(delta.size(0), -1)
    norm = flat_.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    factor = torch.where(norm > epsilon, norm / epsilon, torch.ones_like(norm))
    delta = delta / factor.view(-1, *([1] * (delta.dim() - 1)))
    return orig + delta


def adv_alpha() -> float:
    if ADV_ALPHA_RULE == "madry":
        return 2.5 * ADV_EPSILON / max(ADV_STEPS, 1)
    return ADV_ALPHA_RAW


def pgd_attack(model, crit, x_ref, x_pos, x_neg, lo, hi) -> torch.Tensor:
    """Maximise the InfoNCE loss w.r.t. the REFERENCE inputs only.

    Faithful to Solver.step:
      * only batch.reference is perturbed; positive and negative stay clean
      * linf init:  reference + U(-eps, +eps) over the whole eps-cube
      * l2   init:  reference + _l2_normalize(randn) * U(0,1) * eps
      * linf step:  x_adv += alpha * grad.sign(), then
                    torch.max(torch.min(x_adv, ref+eps), ref-eps)
      * l2   step:  x_adv += alpha * _l2_normalize(grad), then _proj_l2_ball
      * the gradient comes from torch.autograd.grad(loss, x_adv), so model
        parameters are never touched by the attack
      * NO clamp to a valid data range -- the fork applies none (ADV_CLIP_RANGE
        exposes it as an off-by-default option)
      * positive/negative are re-encoded at every inner step because
        _inference(adv_batch) re-runs the whole batch.  Parameters are frozen
        during the attack so caching them is mathematically identical;
        ADV_CACHE_POS_NEG=True enables that ~2x speedup.
    """
    eps, alpha = ADV_EPSILON, adv_alpha()

    if ATTACK_NORM == "linf":
        perturb = torch.empty_like(x_ref).uniform_(-eps, eps)
        x_adv = (x_ref + perturb).clone().detach()
    elif ATTACK_NORM == "l2":
        noise = _l2_normalize(torch.randn_like(x_ref))
        noise = noise * _rand_radius_like(x_ref) * eps
        x_adv = (x_ref + noise).clone().detach()
    else:
        raise ValueError(f"unknown ATTACK_NORM={ATTACK_NORM}")
    if ADV_CLIP_RANGE:
        x_adv = x_adv.clamp(lo, hi)
    x_adv.requires_grad_(True)

    e_pos = e_neg = None
    if ADV_CACHE_POS_NEG:
        with torch.no_grad():
            e_pos = flat(model(x_pos)).detach()
            e_neg = flat(model(x_neg)).detach()

    for _ in range(ADV_STEPS):
        with torch.enable_grad():
            ep = e_pos if e_pos is not None else flat(model(x_pos))
            en = e_neg if e_neg is not None else flat(model(x_neg))
            loss, _, _ = crit(flat(model(x_adv)), ep, en)
        grad_x, = torch.autograd.grad(loss, x_adv, retain_graph=False,
                                      create_graph=False)
        with torch.no_grad():
            if ATTACK_NORM == "linf":
                x_adv = x_adv + alpha * grad_x.sign()
                x_adv = torch.max(torch.min(x_adv, x_ref + eps), x_ref - eps)
            else:
                x_adv = x_adv + alpha * _l2_normalize(grad_x)
                x_adv = _proj_l2_ball(x_adv, x_ref, eps)
            if ADV_CLIP_RANGE:
                x_adv = x_adv.clamp(lo, hi)
        x_adv = x_adv.detach().requires_grad_(True)

    return x_adv.detach()


# ------------------------------------------------------------------------------
# 9. TRAINING
# ------------------------------------------------------------------------------

def _update(model, crit, opt, x_ref, x_pos, x_neg, lam, x_jreg, gen):
    """One optimizer.step(), optionally carrying the lambda ||J||_F^2 penalty."""
    opt.zero_grad(set_to_none=True)
    loss, align, uniform = crit(flat(model(x_ref)), flat(model(x_pos)),
                                flat(model(x_neg)))
    total = loss
    jval = 0.0
    if lam > 0.0:
        xj = x_jreg
        if JREG_SUBBATCH and JREG_SUBBATCH < xj.shape[0]:
            sel = torch.randint(0, xj.shape[0], (JREG_SUBBATCH,),
                                generator=gen, device=xj.device)
            xj = xj[sel]
        jf2 = jacobian_frobenius_sq(model, xj, JREG_ESTIMATOR, JREG_NPROJ)
        total = loss + lam * jf2
        jval = float(jf2)
    total.backward()
    opt.step()
    return float(loss), jval


def expected_steps(scheme: str) -> int:
    return MAX_ITER * (1 if scheme in ("clean_single", "adv_single") else 2)


def train_arm(arm: str, data: dict, sp: dict, seed: int, verbose_every: int = 0):
    cfg = ARMS[arm]
    scheme, lam_max = cfg["scheme"], cfg["lam"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    model, crit = build_model(data["d"])
    opt = torch.optim.Adam(list(model.parameters()) + list(crit.parameters()),
                           lr=LEARNING_RATE)
    X = torch.from_numpy(sp["xs"].astype(np.float32)).to(DEVICE)
    sampler = TimeSampler(X, sp["idx_train"], gen)
    lo = float(X[sp["idx_train"]].min())
    hi = float(X[sp["idx_train"]].max())

    if lam_max > 0:
        selftest_jreg(model, X[sp["idx_train"][:64]].unsqueeze(-1))

    def lam_for(which: str, it: int) -> float:
        if JREG_ON == "clean_only" and which != "clean":
            return 0.0
        if JREG_ON == "adv_only" and which != "adv":
            return 0.0
        return lambda_at(it, lam_max)

    n_steps, hist, t0 = 0, [], time.time()
    for it in range(MAX_ITER):
        x_ref, x_pos, x_neg = sampler(BATCH_SIZE)
        nce_c = nce_a = float("nan")
        j_c = j_a = 0.0

        # ---- update 1: the fork's UNCONDITIONAL clean update -----------------
        if scheme in ("clean_single", "clean_double", "fork_double"):
            nce_c, j_c = _update(model, crit, opt, x_ref, x_pos, x_neg,
                                 lam_for("clean", it), x_ref, gen)
            n_steps += 1

        # ---- update 2 -------------------------------------------------------
        if scheme in ("fork_double", "adv_single"):
            # NOTE: the attack is built AFTER the clean step, i.e. with the
            # already-updated parameters -- exactly as in Solver.step, where
            # the adversarial branch follows self.optimizer.step().
            x_adv = pgd_attack(model, crit, x_ref, x_pos, x_neg, lo, hi)
            xj = x_ref if JREG_AT == "clean" else x_adv
            nce_a, j_a = _update(model, crit, opt, x_adv, x_pos, x_neg,
                                 lam_for("adv", it), xj, gen)
            n_steps += 1
        elif scheme == "clean_double":
            # the compute-matched control: a SECOND clean update on the SAME
            # batch, so `acorn` vs `cebra_2x` differs only by the attack
            nce_a, j_a = _update(model, crit, opt, x_ref, x_pos, x_neg,
                                 lam_for("adv", it), x_ref, gen)
            n_steps += 1

        hist.append((nce_c, nce_a, max(j_c, j_a), lambda_at(it, lam_max)))
        if verbose_every and (it % verbose_every == 0 or it == MAX_ITER - 1):
            print(f"      [{arm:10s}] it={it:6d} nce_clean={nce_c:8.4f} "
                  f"nce_2nd={nce_a:8.4f} |J|^2={max(j_c, j_a):9.4f} "
                  f"lam={lambda_at(it, lam_max):.3f} tau={crit.temperature:.4f}")

    exp = expected_steps(scheme)
    assert n_steps == exp, f"{arm}: {n_steps} steps, expected {exp}"
    model.eval()
    tail = hist[-50:]
    return dict(model=model, crit=crit, n_steps=n_steps,
                seconds=time.time() - t0,
                final_nce=float(np.nanmean([h[1] if not math.isnan(h[1])
                                            else h[0] for h in tail])),
                final_jfro=float(np.mean([h[2] for h in tail])),
                temperature=float(crit.temperature))


@torch.no_grad()
def embed(model, xs: np.ndarray, idx: np.ndarray, chunk: int = 4096) -> np.ndarray:
    out = []
    for s in range(0, len(idx), chunk):
        xb = torch.from_numpy(xs[idx[s:s + chunk]].astype(np.float32)).to(DEVICE)
        out.append(flat(model(xb.unsqueeze(-1))).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64)


# ------------------------------------------------------------------------------
# 10. ATTRIBUTION (library-provided Jacobian; the pinv is taken here)
# ------------------------------------------------------------------------------

_ATTR_REPORTED = {"done": False}


def _pick(dct: dict, keys: Sequence[str]):
    for k in keys:
        for kk in dct:
            if str(kk).lower().replace("_", "-") == k:
                return dct[kk]
    return None


def _to_np(v):
    if v is None:
        return None
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def library_jacobian(model, xs, idx, out_dim):
    """Call cebra.attribution for J_f (the paper's "neuron gradient").

    Registry names and return signatures differ between xCEBRA revisions, so
    the available options are printed once and several call shapes are tried.
    Only J_f is needed; the "inverted neuron gradient" is formed here with
    truncated_pinv so the rank truncation stays explicit."""
    import cebra.attribution
    x = torch.from_numpy(xs[idx].astype(np.float32)).to(DEVICE).unsqueeze(-1)
    if not _ATTR_REPORTED["done"]:
        try:
            print("    [attr] registry options:", cebra.attribution.get_options())
        except Exception as exc:
            print("    [attr] get_options() unavailable:", exc)
    errors = []
    for name in ATTR_METHOD_CANDIDATES:
        for kw in (dict(model=model, input_data=x, output_dimension=out_dim,
                        num_batches=ATTR_NUM_BATCHES),
                   dict(model=model, input_data=x, output_dimension=out_dim),
                   dict(model=model, input_data=x)):
            try:
                res = cebra.attribution.init(name, **kw).compute_attribution_map()
            except Exception as exc:
                errors.append(f"{name}{list(kw)}: {type(exc).__name__}: {exc}")
                continue
            jf = jfinv = None
            if isinstance(res, dict):
                if not _ATTR_REPORTED["done"]:
                    print(f"    [attr] '{name}' -> dict keys: "
                          f"{sorted(map(str, res.keys()))}")
                jf = _pick(res, ["jf", "jacobian", "neuron-gradient"])
                jfinv = _pick(res, ["jf-inv", "jf-inv-svd", "jf-inv-lsq",
                                    "inverted-neuron-gradient"])
            elif isinstance(res, (tuple, list)) and len(res) == 2:
                jf, jfinv = res
            else:
                jf = res
            jf = _to_np(jf)
            if jf is None:
                errors.append(f"{name}: no J_f in {type(res)}")
                continue
            _ATTR_REPORTED["done"] = True
            return canonicalize_jf(jf, out_dim, f"jf[{name}]"), _to_np(jfinv)
    raise RuntimeError("cebra.attribution could not be called. Tried:\n  " +
                       "\n  ".join(errors))


# ------------------------------------------------------------------------------
# 11. PER-SEED RUNNER
# ------------------------------------------------------------------------------

def run_seed(d: int, sigma_obs: float, seed_i: int) -> List[dict]:
    seed = SEED0 + 977 * seed_i
    print(f"\n--- d={d} sigma_obs={sigma_obs} seed={seed} "
          f"({seed_i + 1}/{N_SEEDS}) ---")
    data = generate_dataset(d, sigma_obs, seed)
    gt = ground_truth(d, data["d1"])
    sp = split_data(data)
    print(f"    gt density={gt.mean():.3f}  d1={data['d1']} d2={data['d2']}")
    validate_alignment_metric(data, sp, gt, seed)

    rows = []
    for arm in ARMS:
        tr = train_arm(arm, data, sp, seed, verbose_every=max(1, MAX_ITER // 4))
        f_map = embed(tr["model"], sp["xs"], sp["idx_map"])
        A_z2e, _, r2 = fit_linear_maps(f_map, data["z"][sp["idx_map"]])
        jf, jfinv_lib = library_jacobian(tr["model"], sp["xs"], sp["idx_attr"],
                                          d + OUT_EXTRA)
        jf = jf.astype(np.float64)
        Q = truncated_pinv(jf, PINV_DROP)
        M_inv = aggregate_map(align_to_latents(Q, A_z2e))
        res = binary_scores(M_inv, gt)
        # the un-inverted "neuron gradient", for the paper's own comparison
        res_fwd = {f"{k}_jf": v for k, v in
                   binary_scores(aggregate_map(np.swapaxes(jf, 1, 2)), gt).items()}
        xcheck = float("nan")
        if jfinv_lib is not None:
            try:
                Ql = np.swapaxes(canonicalize_jf(jfinv_lib, d + OUT_EXTRA,
                                                 "jfinv_lib"), 1, 2)
                Ml = aggregate_map(align_to_latents(Ql.astype(np.float64), A_z2e))
                xcheck = float(np.corrcoef(Ml.ravel(), M_inv.ravel())[0, 1])
            except Exception as exc:
                print(f"    [attr] library jf-inv cross-check skipped: {exc}")
        sing = np.linalg.svd(jf, compute_uv=False)
        row = dict(arm=arm, scheme=ARMS[arm]["scheme"], lam=ARMS[arm]["lam"],
                   seed=seed, d=d, sigma_obs=sigma_obs,
                   r2_mean=float(r2.mean()), r2_min=float(r2.min()),
                   final_nce=tr["final_nce"], final_jfro=tr["final_jfro"],
                   temperature=tr["temperature"], seconds=tr["seconds"],
                   n_steps=tr["n_steps"],
                   sv_ratio=float(np.median(sing[:, -1] / (sing[:, 0] + 1e-30))),
                   libinv_corr=xcheck, **res, **res_fwd)
        rows.append(row)
        print(f"    [{arm:10s}] auROC={row['auroc_global']:.4f} "
              f"auPRC={row['auprc_global']:.4f} "
              f"(jf-only={row['auroc_global_jf']:.4f}) "
              f"R2={row['r2_mean']:.4f} steps={row['n_steps']} "
              f"{row['seconds']:.0f}s")

    r2s = [r["r2_mean"] for r in rows]
    ok = min(r2s) >= R2_MIN and (max(r2s) - min(r2s)) <= R2_PARITY_MAX
    if not ok:
        print(f"    !! seed EXCLUDED: R2 range [{min(r2s):.3f}, {max(r2s):.3f}] "
              f"violates R2_MIN={R2_MIN} / parity={R2_PARITY_MAX}.  Comparing "
              f"attribution across arms of unequal representation quality "
              f"measures representation quality, not attribution.")
    for r in rows:
        r["included"] = bool(ok)
    return rows


# ------------------------------------------------------------------------------
# 12. STATISTICS
# ------------------------------------------------------------------------------

def paired_stats(a, b, rng) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    dif = a[m] - b[m]
    n = len(dif)
    out = dict(n=n, mean_diff=float(dif.mean()) if n else float("nan"),
               n_wins=int((dif > 0).sum()))
    if n < 2:
        out.update(p=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                   test="n/a")
        return out
    if HAVE_SCIPY and np.any(dif != 0):
        out["p"], out["test"] = float(_sps.wilcoxon(dif).pvalue), "wilcoxon"
    else:
        nz, k = int((dif != 0).sum()), int((dif > 0).sum())
        out["p"] = float(_sps.binomtest(k, nz, 0.5).pvalue) if (
            HAVE_SCIPY and nz) else float("nan")
        out["test"] = "sign"
    boot = np.array([rng.choice(dif, size=n, replace=True).mean()
                     for _ in range(N_BOOTSTRAP)])
    out["ci_lo"], out["ci_hi"] = [float(v) for v in np.percentile(boot, [2.5, 97.5])]
    return out


def report(rows: List[dict]):
    rng = np.random.default_rng(0)
    inc = [r for r in rows if r["included"]]
    n_arms = max(1, len(ARMS))
    print("\n" + "=" * 78)
    print(f" RESULTS  included seeds: {len(inc)//n_arms} of {len(rows)//n_arms}")
    print(f" pre-registered primary metric     : {PRIMARY_METRIC}")
    print(f" pre-registered primary comparison : "
          f"{PRIMARY_COMPARISON[0]} vs {PRIMARY_COMPARISON[1]} (compute matched)")
    print("=" * 78)

    print(f"\n{'arm':<12}{'steps':>7}{'auROC':>19}{'auPRC':>19}{'R2':>9}"
          f"{'|J|_F^2':>11}")
    for arm in ARMS:
        sel = [r for r in inc if r["arm"] == arm]
        if not sel:
            print(f"{arm:<12}{'--':>7}")
            continue
        v = np.array([r[PRIMARY_METRIC] for r in sel])
        p = np.array([r["auprc_global"] for r in sel])
        q = np.array([r["r2_mean"] for r in sel])
        j = np.array([r["final_jfro"] for r in sel])
        sd = lambda t: t.std(ddof=1) if len(t) > 1 else 0.0
        print(f"{arm:<12}{sel[0]['n_steps']:>7}"
              f"{v.mean():>11.4f} +-{sd(v):.4f}"
              f"{p.mean():>11.4f} +-{sd(p):.4f}"
              f"{q.mean():>9.4f}{j.mean():>11.3f}")

    def get(arm):
        return np.array([r[PRIMARY_METRIC] for r in
                         sorted([x for x in inc if x["arm"] == arm],
                                key=lambda z: (z["d"], z["sigma_obs"], z["seed"]))])

    print("\npaired comparisons on", PRIMARY_METRIC)
    for tag, (x, y) in ([("PRIMARY", PRIMARY_COMPARISON)] +
                        [("secondary", c) for c in SECONDARY_COMPARISONS]):
        if x not in ARMS or y not in ARMS:
            continue
        s = paired_stats(get(x), get(y), rng)
        print(f"  [{tag:9s}] {x:>11s} - {y:<11s} diff={s['mean_diff']:+.4f} "
              f"95%CI[{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}] p={s['p']:.4g} "
              f"({s['test']}) wins {s['n_wins']}/{s['n']}")

    print("""
How to read this
  cebra_2x - cebra        the size of the doubled-update confound alone.  If
                          this is large, every previously reported acorn-vs-
                          cebra number was partly measuring optimisation
                          budget, not adversarial training.
  xcebra - cebra          the paper's own claim, reproduced on your data.  If
                          this is not positive, something upstream is wrong
                          and the rest of the table is uninterpretable.
  acorn - cebra_2x        PRIMARY.  Positive => PGD itself buys identifiable
                          attribution.  First-order, linf adversarial training
                          penalises eps*||grad_x L||_1, the dual-norm analogue
                          of the paper's Frobenius penalty, so this is the
                          mechanistically expected direction.
  acorn - xcebra_2x       does the implicit penalty match the explicit one?
  acorn_xreg - acorn      do the two mechanisms stack, or are they redundant?

Secondary p-values are uncorrected for multiplicity; only the PRIMARY line
supports a confirmatory claim.""")

    if rows:
        keys = list(rows[0].keys())
        with open(RESULT_CSV, "w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        print(f"\nwrote {RESULT_CSV}  ({len(rows)} rows)")


# ------------------------------------------------------------------------------
# 13. MAIN
# ------------------------------------------------------------------------------

def main():
    _banner()
    rows: List[dict] = []
    for d in D_LATENT_LIST:
        for sigma_obs in SIGMA_OBS_SWEEP:
            for i in range(N_SEEDS):
                rows.extend(run_seed(d, sigma_obs, i))
    report(rows)


if __name__ == "__main__":
    main()

###########################
###########################
###########################
###########################
###########################
###########################
###########################
###########################
######################################################
#################################################################################
############################################################################################################
#######################################################################################################################################
##################################################################################################################################################################

# # Lorenzo
# # import os
# # import gc
# # import random
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import pandas as pd

# # from sklearn.metrics import roc_auc_score
# # from utils.min_distance import min_l2_distance
# # from utils.constants import CEBRA_DIR

# # import sys
# # sys.path.insert(0, str(CEBRA_DIR))
# # import cebra
# # from cebra import CEBRA


# # # ============================================================
# # # 1) Synthetic Data Config & Generation
# # # ============================================================
# # T = 100_000
# # D1 = 3  # Lorenz system latents
# # D2 = 3  # Lorenz system latents
# # D_LATENT = D1 + D2  # 6

# # N1 = 25
# # N2 = 25
# # D_OBS = N1 + N2     # 4

# # N_MLP_LAYERS = 4
# # SIGMA_EPS = 0.03

# # # Output dimension must match latent dimension
# # OUTPUT_DIM = D_LATENT  # 6
# # BATCH_SIZE = 2048
# # MAX_ITER = 2500
# # adv_epsilon_default = 0.5

# # ATTR_BATCH_SIZE = 128

# # OUT_DIR = "outputs"
# # os.makedirs(OUT_DIR, exist_ok=True)

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # RANDOM_SEED = 88
# # np.random.seed(RANDOM_SEED)
# # torch.manual_seed(RANDOM_SEED)
# # random.seed(RANDOM_SEED)


# # def make_mlp(in_dim, out_dim, n_layers=4, seed=0):
# #     torch.manual_seed(seed)
# #     layers = []
# #     d_in = in_dim
# #     hidden = in_dim * 10

# #     for i in range(n_layers - 1):
# #         d_h = in_dim * 30 if i < n_layers - 2 else hidden
# #         lin = nn.Linear(d_in, d_h)
# #         nn.init.orthogonal_(lin.weight)
# #         nn.init.zeros_(lin.bias)
# #         layers += [lin, nn.GELU()]
# #         d_in = d_h

# #     lin = nn.Linear(d_in, out_dim)
# #     nn.init.orthogonal_(lin.weight)
# #     nn.init.zeros_(lin.bias)
# #     layers.append(lin)

# #     mlp = nn.Sequential(*layers)
# #     for p in mlp.parameters():
# #         p.requires_grad_(False)
# #     return mlp.eval()


# # def brownian_motion_box(T, d, sigma=0.03, seed=0):
# #     rng = np.random.default_rng(seed)
# #     x = np.zeros((T, d), dtype=np.float32)
# #     x[0] = rng.uniform(-1.0, 1.0, size=d).astype(np.float32)

# #     for t in range(T - 1):
# #         step = rng.normal(loc=0.0, scale=sigma, size=d).astype(np.float32)
# #         x[t + 1] = np.clip(x[t] + step, -1.0, 1.0)
# #     return x


# # def lorenz_system(T, dt=0.01, seed=0):
# #     """
# #     Generates T steps of the Lorenz attractor.
# #     Standard parameters: sigma=10, rho=28, beta=8/3
# #     """
# #     rng = np.random.default_rng(seed)

# #     sigma = 10.0
# #     rho = 28.0
# #     beta = 8.0 / 3.0

# #     xyz = np.zeros((T, 3), dtype=np.float32)

# #     # seed-dependent initial condition so z1 and z2 are different
# #     xyz[0] = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)

# #     for t in range(T - 1):
# #         x, y, z = xyz[t]
# #         dx = sigma * (y - x) * dt
# #         dy = (x * (rho - z) - y) * dt
# #         dz = (x * y - beta * z) * dt
# #         xyz[t + 1] = [x + dx, y + dy, z + dz]

# #     # Min-Max Normalize to [-1, 1]
# #     xyz_min = xyz.min(axis=0, keepdims=True)
# #     xyz_max = xyz.max(axis=0, keepdims=True)
# #     xyz_norm = 2.0 * (xyz - xyz_min) / (xyz_max - xyz_min + 1e-8) - 1.0

# #     return xyz_norm.astype(np.float32)


# # def make_binary_ground_truth(D1, D2, N1, N2):
# #     """
# #     Ground truth:
# #     - first D1 latents connect to all neurons
# #     - last D2 latents connect only to x2 block (last N2 neurons)
# #     shape = [D_LATENT, D_OBS] -> [6, 4]
# #     """
# #     gt = np.zeros((D1 + D2, N1 + N2), dtype=bool)
# #     gt[:D1, :] = True
# #     gt[D1:, N1:] = True
# #     return gt


# # def generate_synthetic_data(T=T, seed=42):
# #     # z1 uses Lorenz System (3 dimensions)
# #     z1 = lorenz_system(T, dt=0.01, seed=seed)

# #     # z2 also uses Lorenz System (3 dimensions)
# #     z2 = lorenz_system(T, dt=0.01, seed=seed + 1)

# #     g1 = make_mlp(D1, N1, n_layers=N_MLP_LAYERS, seed=seed + 10)
# #     g2 = make_mlp(D1 + D2, N2, n_layers=N_MLP_LAYERS, seed=seed + 20)

# #     z1_t = torch.tensor(z1, dtype=torch.float32)
# #     z2_t = torch.tensor(z2, dtype=torch.float32)

# #     with torch.no_grad():
# #         x1 = g1(z1_t).cpu().numpy()
# #         x2 = g2(torch.cat([z1_t, z2_t], dim=1)).cpu().numpy()

# #     x = np.concatenate([x1, x2], axis=1).astype(np.float32)
# #     latent = np.concatenate([z1, z2], axis=1).astype(np.float32)

# #     gt_bool = make_binary_ground_truth(D1, D2, N1, N2)
# #     gt_attr = gt_bool.astype(np.float32)

# #     return x, latent, gt_attr, gt_bool


# # # ============================================================
# # # 2) Utils
# # # ============================================================
# # def cleanup_cuda(*objs):
# #     for obj in objs:
# #         try:
# #             del obj
# #         except Exception:
# #             pass
# #     gc.collect()
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()
# #         torch.cuda.ipc_collect()


# # def reduce_attr_map(arr):
# #     """
# #     Convert attribution output to 2D map [output_dim, input_dim]
# #     if it has sample dimension.
# #     """
# #     arr = np.asarray(arr)
# #     if arr.ndim == 3:
# #         return np.abs(arr).mean(axis=0)
# #     if arr.ndim == 2:
# #         return np.abs(arr)
# #     if arr.ndim == 1:
# #         return np.abs(arr)[None, :]
# #     raise ValueError(f"Unsupported attribution shape: {arr.shape}")


# # def compute_auroc(attr_map_2d, gt_bool):
# #     y_true = gt_bool.ravel().astype(int)
# #     y_score = np.asarray(attr_map_2d, dtype=np.float64).ravel()

# #     if y_true.shape[0] != y_score.shape[0]:
# #         raise ValueError(
# #             f"Shape mismatch: y_true has {y_true.shape[0]} elements, "
# #             f"y_score has {y_score.shape[0]} elements, "
# #             f"attr_map shape={attr_map_2d.shape}, gt shape={gt_bool.shape}"
# #         )

# #     if len(np.unique(y_true)) < 2:
# #         return float("nan")

# #     return float(roc_auc_score(y_true, y_score))


# # # ============================================================
# # # 3) Data Generation
# # # ============================================================
# # print("Generating synthetic dataset...")
# # x_np, y_np, gt_attr, gt_attr_bool = generate_synthetic_data(T=T, seed=42)

# # print("x shape:", x_np.shape)             # (T, 4)
# # print("y shape:", y_np.shape)             # (T, 6)
# # print("gt_attr_bool shape:", gt_attr_bool.shape)  # (6, 4)

# # split_idx = int(0.8 * len(x_np))
# # train_data = x_np[:split_idx].astype(np.float32)
# # train_continuous_label = y_np[:split_idx].astype(np.float32)

# # # For later use in CEBRA setup
# # adv_epsilon = float(min_l2_distance(train_data)) / 2.0
# # adv_epsilon = max(adv_epsilon, 1e-6)

# # # ============================================================
# # # 4) Train + Attribution
# # # ============================================================
# # rows = []
# # all_results = {}

# # for adv in [False, True]:
# #     cleanup_cuda()

# #     model_name = "ACORN" if adv else "CEBRA"
# #     training_mode = "adversarial" if adv else "clean"

# #     print("\n" + "=" * 70)
# #     print(f"Training {model_name} ({training_mode})")
# #     print("=" * 70)

# #     model = CEBRA(
# #         batch_size=BATCH_SIZE,
# #         temperature=0.4,
# #         model_architecture="offset36-model-more-dropout",
# #         time_offsets=4,
# #         max_iterations=MAX_ITER,
# #         output_dimension=OUTPUT_DIM,
# #         verbose=True,
# #         training_mode=training_mode,
# #         adv_alpha=adv_epsilon / 5,
# #         adv_epsilon=adv_epsilon,
# #         adv_steps=10,
# #         attack_norm="linf",
# #         num_hidden_units=32,
# #     )

# #     model.fit(train_data, train_continuous_label)

# #     save_path = os.path.join(OUT_DIR, f"{model_name}_synthetic.pth")
# #     model.save(save_path)
# #     print("Saved model to:", save_path)

# #     trained_model = model.solver_.model.to(device)

# #     input_tensor = torch.from_numpy(train_data).float().to(device).requires_grad_(True)
# #     output_dim = int(getattr(trained_model, "num_output", OUTPUT_DIM))

# #     method = cebra.attribution.init(
# #         name="jacobian-based-batched",
# #         model=trained_model,
# #         input_data=input_tensor,
# #         output_dimension=output_dim,
# #     )

# #     result = method.compute_attribution_map(batch_size=min(128, len(train_data)))
# #     print("Attribution keys:", list(result.keys()))

# #     # Reduce to 2D maps
# #     jc_map = reduce_attr_map(result["jf"])                      # expected shape: (6, 4)
# #     jc_inv_map = reduce_attr_map(result["jf-inv-svd"])          # expected shape: (4, 6)
# #     jc_invconv_map = reduce_attr_map(result["jf-convabs-inv-svd"])  # expected shape: (4, 6)

# #     # AUROC scores
# #     auc_jc = compute_auroc(jc_map, gt_attr_bool)
# #     auc_jc_inv = compute_auroc(jc_inv_map.T, gt_attr_bool)
# #     auc_jc_invconv = compute_auroc(jc_invconv_map.T, gt_attr_bool)

# #     print(f"** {model_name} jc AUROC:        {auc_jc:.4f} **")
# #     print(f"** {model_name} jc_inv AUROC:    {auc_jc_inv:.4f} **")
# #     print(f"** {model_name} jc_invconv AUROC:{auc_jc_invconv:.4f} **")

# #     all_results[model_name] = {
# #         "jc": jc_map,
# #         "jc_inv": jc_inv_map,
# #         "jc_invconv": jc_invconv_map,
# #         "auc_jc": auc_jc,
# #         "auc_jc_inv": auc_jc_inv,
# #         "auc_jc_invconv": auc_jc_invconv,
# #     }

# #     rows.extend([
# #         {"model": model_name, "metric": "jc", "auroc": auc_jc},
# #         {"model": model_name, "metric": "jc_inv", "auroc": auc_jc_inv},
# #         {"model": model_name, "metric": "jc_invconv", "auroc": auc_jc_invconv},
# #     ])

# #     cleanup_cuda(method, trained_model, input_tensor, model)

# # # ============================================================
# # # 5) Summary
# # # ============================================================
# # print("\n" + "=" * 80)
# # print(" SUMMARY OF EXPERIMENT RESULTS ".center(80, "="))
# # print("=" * 80)
# # for model_name, res in all_results.items():
# #     print(
# #         f" Model: {model_name:<6} | "
# #         f"jc={res['auc_jc']:.4f} | "
# #         f"jc_inv={res['auc_jc_inv']:.4f} | "
# #         f"jc_invconv={res['auc_jc_invconv']:.4f}"
# #     )
# # print("=" * 80)

# # # ============================================================
# # # 6) Save CSV
# # # ============================================================
# # results_df = pd.DataFrame(rows)
# # csv_path = os.path.join(OUT_DIR, "synthetic_auroc_results.csv")
# # results_df.to_csv(csv_path, index=False)
# # print(f"Saved AUROC results to: {csv_path}")

# # print("Done.")

# #Brownian
# import os
# import gc
# import random
# import copy
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt

# from sklearn.metrics import roc_auc_score, average_precision_score

# from utils.min_distance import min_l2_distance
# from utils.constants import CEBRA_DIR

# import sys
# if "cebra" in sys.modules:
#     del sys.modules["cebra"]
# sys.path.insert(0, str(CEBRA_DIR))

# import cebra
# from cebra import CEBRA


# # ============================================================
# # 1) Global Config
# # ============================================================
# T = 100_000

# # Synthetic latent blocks
# D1 = 3
# D2 = 3

# # Observed neurons/features
# N1 = 25
# N2 = 25

# # Generator
# N_MLP_LAYERS = 4
# SIGMA_EPS_DEFAULT = 0.03

# # Training
# BATCH_SIZE = 2048
# MAX_ITER = 25000
# ATTR_BATCH_SIZE = 128

# # Model / run
# OUTPUT_DIM = D1 + D2
# OUT_DIR = "outputs"
# IMG_DIR = "images"
# os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(IMG_DIR, exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Keep one seed by default; add more if you want averaging
# SEEDS = [38,226,1,36,989,26,84,66,27,81,49]

# DATASET_CFG = {
#     "name": "FIG5_SINGLE",
#     "D1": D1,
#     "D2": D2,
#     "N1": N1,
#     "N2": N2,
#     "sigma_eps": SIGMA_EPS_DEFAULT,
# }


# # ============================================================
# # 2) Reproducibility
# # ============================================================
# def set_all_seeds(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     try:
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#     except Exception:
#         pass


# # ============================================================
# # 3) Synthetic Data Generation
# # ============================================================
# class ScaledTanh(nn.Module):
#     def __init__(self, scale=1.0):
#         super().__init__()
#         self.scale = float(scale)

#     def forward(self, x):
#         return self.scale * torch.tanh(x)


# def make_mlp(in_dim, out_dim, n_layers=4, seed=0):
#     """
#     Small random mixing network.
#     """
#     torch.manual_seed(seed)

#     layers = []
#     d_in = in_dim
#     hidden = max(64, 8 * max(in_dim, out_dim))

#     for _ in range(n_layers - 1):
#         lin = nn.Linear(d_in, hidden)
#         nn.init.orthogonal_(lin.weight)
#         nn.init.zeros_(lin.bias)
#         layers += [lin, nn.GELU()]
#         d_in = hidden

#     lin = nn.Linear(d_in, out_dim)
#     nn.init.orthogonal_(lin.weight)
#     nn.init.zeros_(lin.bias)
#     layers += [lin, ScaledTanh(scale=1.0)]

#     mlp = nn.Sequential(*layers).to(device).eval()
#     for p in mlp.parameters():
#         p.requires_grad_(False)
#     return mlp


# def brownian_motion_box(T, d, sigma=0.03, seed=0):
#     """
#     Brownian motion in [-1, 1]^d with rejection to keep it inside the box.
#     """
#     rng = np.random.default_rng(seed)
#     z = np.empty((T, d), dtype=np.float32)
#     z[0] = rng.uniform(-1.0, 1.0, size=d).astype(np.float32)

#     for t in range(1, T):
#         prev = z[t - 1].copy()
#         nxt = prev + rng.normal(0.0, sigma, size=d).astype(np.float32)

#         mask = (nxt < -1.0) | (nxt > 1.0)
#         while np.any(mask):
#             nxt[mask] = prev[mask] + rng.normal(0.0, sigma, size=mask.sum()).astype(np.float32)
#             mask = (nxt < -1.0) | (nxt > 1.0)

#         z[t] = nxt

#     return z


# def make_binary_ground_truth(D1, D2, N1, N2):
#     """
#     Ground truth attribution map:
#       rows = latent dimensions [z1, z2]
#       cols = observed neurons/features [x1, x2]

#     z1 -> x1 and x2
#     z2 -> x2 only
#     """
#     gt = np.zeros((D1 + D2, N1 + N2), dtype=bool)
#     gt[:D1, :] = True
#     gt[D1:, N1:] = True
#     return gt


# def generate_synthetic_data(cfg, seed=42):
#     D1 = int(cfg["D1"])
#     D2 = int(cfg["D2"])
#     N1 = int(cfg["N1"])
#     N2 = int(cfg["N2"])
#     sigma_eps = float(cfg.get("sigma_eps", SIGMA_EPS_DEFAULT))

#     z1 = brownian_motion_box(T, D1, sigma=sigma_eps, seed=seed)
#     z2 = brownian_motion_box(T, D2, sigma=sigma_eps, seed=seed + 1)

#     g1 = make_mlp(D1, N1, n_layers=N_MLP_LAYERS, seed=seed + 10)
#     g2 = make_mlp(D1 + D2, N2, n_layers=N_MLP_LAYERS, seed=seed + 20)

#     z1_t = torch.tensor(z1, dtype=torch.float32, device=device)
#     z2_t = torch.tensor(z2, dtype=torch.float32, device=device)

#     with torch.no_grad():
#         x1 = g1(z1_t).cpu().numpy()
#         x2 = g2(torch.cat([z1_t, z2_t], dim=1)).cpu().numpy()

#     x = np.concatenate([x1, x2], axis=1).astype(np.float32)
#     latent = np.concatenate([z1, z2], axis=1).astype(np.float32)

#     gt_bool = make_binary_ground_truth(D1, D2, N1, N2)
#     gt_attr = gt_bool.astype(np.float32)

#     return x, latent, gt_attr, gt_bool


# # ============================================================
# # 4) Utils
# # ============================================================
# def cleanup_cuda(*objs):
#     for obj in objs:
#         try:
#             del obj
#         except Exception:
#             pass
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()


# def reduce_attr_map(arr):
#     """
#     Convert attribution output to 2D map [output_dim, input_dim]
#     if it has sample dimension.
#     Works for torch.Tensor or numpy.ndarray.
#     """
#     if torch.is_tensor(arr):
#         arr = arr.detach().cpu().numpy()
#     else:
#         arr = np.asarray(arr)

#     arr = np.abs(arr)

#     if arr.ndim == 3:
#         # [samples, latent, features] -> average over samples
#         return arr.mean(axis=0).astype(np.float32)
#     if arr.ndim == 2:
#         return arr.astype(np.float32)
#     if arr.ndim == 1:
#         return np.abs(arr)[None, :].astype(np.float32)

#     raise ValueError(f"Unsupported attribution shape: {arr.shape}")


# def align_attr_to_gt(attr_map_2d, gt_bool):
#     if attr_map_2d.shape == gt_bool.shape:
#         return attr_map_2d
#     if attr_map_2d.T.shape == gt_bool.shape:
#         return attr_map_2d.T
#     raise ValueError(
#         f"Cannot align attribution map shape {attr_map_2d.shape} to ground truth shape {gt_bool.shape}"
#     )


# def compute_auroc(attr_map_2d, gt_bool):
#     y_true = gt_bool.ravel().astype(int)
#     y_score = np.asarray(attr_map_2d, dtype=np.float64).ravel()

#     if y_true.shape[0] != y_score.shape[0]:
#         raise ValueError(
#             f"Shape mismatch: y_true has {y_true.shape[0]} elements, "
#             f"y_score has {y_score.shape[0]} elements, "
#             f"attr_map shape={attr_map_2d.shape}, gt shape={gt_bool.shape}"
#         )

#     if len(np.unique(y_true)) < 2:
#         return float("nan"), float("nan")

#     auroc = float(roc_auc_score(y_true, y_score))
#     auprc = float(average_precision_score(y_true, y_score))
#     return auroc, auprc


# def infer_adv_epsilon(train_x_np: np.ndarray) -> float:
#     try:
#         x_t = torch.tensor(train_x_np, dtype=torch.float32)
#         eps = float(min_l2_distance(x_t)) / 2.0
#         return max(eps, 1e-6)
#     except Exception:
#         return max(float(np.std(train_x_np)) * 0.05, 1e-6)


# def save_heatmap(mat, path, title):
#     fig, ax = plt.subplots(figsize=(10, 5))
#     im = ax.imshow(mat, aspect="auto", cmap="cividis")
#     ax.set_title(title)
#     ax.set_xlabel("Observed feature")
#     ax.set_ylabel("Latent dimension")
#     fig.colorbar(im, ax=ax, shrink=0.9)
#     fig.tight_layout()
#     fig.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close(fig)


# def subsample_for_attribution(x_np, max_points=10000):
#     if len(x_np) <= max_points:
#         return x_np
#     idx = np.linspace(0, len(x_np) - 1, max_points).astype(int)
#     return x_np[idx]


# # ============================================================
# # 5) Model / Attribution
# # ============================================================
# def build_model(adv: bool, adv_epsilon: float):
#     return CEBRA(
#         batch_size=BATCH_SIZE,
#         temperature=0.4,
#         model_architecture="offset36-model-more-dropout",
#         time_offsets=4,
#         max_iterations=MAX_ITER,
#         output_dimension=OUTPUT_DIM,
#         verbose=True,
#         training_mode="adversarial" if adv else "clean",
#         adv_alpha=(5 / 5.0) if adv else 0.0,
#         adv_epsilon=5 if adv else 0.0,
#         adv_steps=10 if adv else 0,
#         attack_norm="linf",
#         num_hidden_units=32,
#         device="cuda_if_available",
#     )


# def train_and_score_one_run(
#     cfg,
#     seed,
#     train_x_np,
#     z1_train_np,
#     z2_train_np,
#     gt_bool,
#     adv: bool,
# ):
#     set_all_seeds(seed)

#     model_name = "ACORN" if adv else "CEBRA"
#     adv_epsilon = infer_adv_epsilon(train_x_np) if adv else 0.0

#     model = build_model(adv=adv, adv_epsilon=adv_epsilon)
#     model.fit(
#         train_x_np.astype(np.float32),
#         z1_train_np.astype(np.float32),
#         z2_train_np.astype(np.float32),
#     )

#     save_path = os.path.join(OUT_DIR, f"{cfg['name']}_seed{seed}_{model_name}.pth")
#     try:
#         model.save(save_path)
#         print("Saved model to:", save_path)
#     except Exception as e:
#         print("Could not save model:", e)

#     trained_model = model.solver_.model.to(device)
#     if hasattr(trained_model, "split_outputs"):
#         trained_model.split_outputs = False
#     trained_model.eval()

#     attr_x = subsample_for_attribution(train_x_np, max_points=10000)
#     input_tensor = torch.from_numpy(attr_x.astype(np.float32)).to(device)
#     input_tensor.requires_grad_(True)

#     method = cebra.attribution.init(
#         name="jacobian-based-batched",
#         model=trained_model,
#         input_data=input_tensor,
#         output_dimension=int(getattr(trained_model, "num_output", OUTPUT_DIM)),
#     )

#     batch_size = min(ATTR_BATCH_SIZE, len(attr_x))
#     result = method.compute_attribution_map(batch_size=batch_size)
#     print("Attribution keys:", list(result.keys()))

#     jf_key = "jf"
#     if "jf-inv-svd" in result:
#         jfinv_key = "jf-inv-svd"
#     elif "jf-inv" in result:
#         jfinv_key = "jf-inv"
#     else:
#         raise KeyError(f"No inverse attribution key found. Available: {list(result.keys())}")

#     jf_raw = reduce_attr_map(result[jf_key])
#     jfinv_raw = reduce_attr_map(result[jfinv_key])

#     jf_map = align_attr_to_gt(jf_raw, gt_bool)
#     jfinv_map = align_attr_to_gt(jfinv_raw, gt_bool)

#     auroc_jf, auprc_jf = compute_auroc(jf_map, gt_bool)
#     auroc_jfinv, auprc_jfinv = compute_auroc(jfinv_map, gt_bool)

#     print(f"[{cfg['name']}] seed={seed} | {model_name} | JF     AUROC: {auroc_jf:.4f} | AUPRC: {auprc_jf:.4f}")
#     print(f"[{cfg['name']}] seed={seed} | {model_name} | JF-INV AUROC: {auroc_jfinv:.4f} | AUPRC: {auprc_jfinv:.4f}")

#     # Save results / plots
#     run_tag = f"{cfg['name']}_seed{seed}_{model_name}"
#     np.savez_compressed(
#         os.path.join(OUT_DIR, f"{run_tag}_attrs.npz"),
#         jf=jf_map.astype(np.float32),
#         jfinv=jfinv_map.astype(np.float32),
#         gt=gt_bool.astype(np.uint8),
#         auroc_jf=np.array([auroc_jf], dtype=np.float32),
#         auprc_jf=np.array([auprc_jf], dtype=np.float32),
#         auroc_jfinv=np.array([auroc_jfinv], dtype=np.float32),
#         auprc_jfinv=np.array([auprc_jfinv], dtype=np.float32),
#     )

#     save_heatmap(jf_map, os.path.join(IMG_DIR, f"{run_tag}_JF.png"), f"{run_tag} | JF")
#     save_heatmap(jfinv_map, os.path.join(IMG_DIR, f"{run_tag}_JF_INV.png"), f"{run_tag} | JF-INV")
#     save_heatmap(gt_bool.astype(np.float32), os.path.join(IMG_DIR, f"{run_tag}_GT.png"), f"{run_tag} | GT")

#     cleanup_cuda(method, trained_model, input_tensor, model)

#     return {
#         "setup": cfg["name"],
#         "seed": seed,
#         "D1": int(cfg["D1"]),
#         "D2": int(cfg["D2"]),
#         "N1": int(cfg["N1"]),
#         "N2": int(cfg["N2"]),
#         "D_LATENT": int(cfg["D1"] + cfg["D2"]),
#         "D_OBS": int(cfg["N1"] + cfg["N2"]),
#         "model": model_name,
#         "training_mode": "adversarial" if adv else "clean",
#         "auroc_jf": auroc_jf,
#         "auprc_jf": auprc_jf,
#         "auroc_jfinv": auroc_jfinv,
#         "auprc_jfinv": auprc_jfinv,
#     }


# # ============================================================
# # 6) Main
# # ============================================================
# def main():
#     all_rows = []

#     print("\n" + "#" * 90)
#     print(
#         f"SETUP: {DATASET_CFG['name']} | "
#         f"D1={DATASET_CFG['D1']} D2={DATASET_CFG['D2']} | "
#         f"N1={DATASET_CFG['N1']} N2={DATASET_CFG['N2']} | "
#         f"sigma={DATASET_CFG['sigma_eps']}"
#     )
#     print("#" * 90)

#     # Generate one synthetic dataset per seed for reproducibility
#     for seed in SEEDS:
#         set_all_seeds(seed)
#         x_np, latent_np, gt_attr, gt_bool = generate_synthetic_data(DATASET_CFG, seed=seed)

#         print("x shape:", x_np.shape)
#         print("latent shape:", latent_np.shape)
#         print("gt_attr shape:", gt_attr.shape)

#         split_idx = int(0.8 * len(x_np))
#         train_x = x_np[:split_idx].astype(np.float32)
#         train_latent = latent_np[:split_idx].astype(np.float32)

#         z1_train = train_latent[:, :DATASET_CFG["D1"]]
#         z2_train = train_latent[:, DATASET_CFG["D1"]:]

#         # Save one copy of the synthetic ground truth maps for inspection
#         save_heatmap(gt_bool.astype(np.float32), os.path.join(IMG_DIR, f"{DATASET_CFG['name']}_seed{seed}_GT_ONLY.png"), "GT only")

#         adv_epsilon = infer_adv_epsilon(train_x)
#         print("adv_epsilon:", adv_epsilon)

#         for adv in [False, True]:
#             cleanup_cuda()
#             mode_name = "adversarial" if adv else "clean"
#             print("\n" + "=" * 70)
#             print(f"Training {DATASET_CFG['name']} | seed={seed} | mode={mode_name}")
#             print("=" * 70)

#             row = train_and_score_one_run(
#                 cfg=DATASET_CFG,
#                 seed=seed,
#                 train_x_np=train_x,
#                 z1_train_np=z1_train,
#                 z2_train_np=z2_train,
#                 gt_bool=gt_bool,
#                 adv=adv,
#             )
#             all_rows.append(row)

#     results_df = pd.DataFrame(all_rows)

#     detailed_csv = os.path.join(OUT_DIR, "synthetic_auroc_detailed.csv")
#     results_df.to_csv(detailed_csv, index=False)
#     print(f"\nSaved detailed results to: {detailed_csv}")

#     summary_df = (
#         results_df
#         .groupby(["setup", "model", "training_mode", "D1", "D2", "N1", "N2", "D_LATENT", "D_OBS"], as_index=False)
#         .agg(
#             auroc_jf_mean=("auroc_jf", "mean"),
#             auroc_jf_std=("auroc_jf", "std"),
#             auprc_jf_mean=("auprc_jf", "mean"),
#             auprc_jf_std=("auprc_jf", "std"),
#             auroc_jfinv_mean=("auroc_jfinv", "mean"),
#             auroc_jfinv_std=("auroc_jfinv", "std"),
#             auprc_jfinv_mean=("auprc_jfinv", "mean"),
#             auprc_jfinv_std=("auprc_jfinv", "std"),
#             n_runs=("seed", "count"),
#         )
#     )

#     summary_csv = os.path.join(OUT_DIR, "synthetic_auroc_summary.csv")
#     summary_df.to_csv(summary_csv, index=False)
#         # ============================================================
#     # Save seed-wise + mean ± std results
#     # ============================================================

#     fake_result_dir = "Fake_dataset_result"
#     os.makedirs(fake_result_dir, exist_ok=True)

#     seed_rows = []

#     # individual seed results
#     for _, row in results_df.iterrows():
#         seed_rows.append({
#             "type": "seed",
#             "setup": row["setup"],
#             "model": row["model"],
#             "training_mode": row["training_mode"],
#             "seed": row["seed"],
#             "AUROC_JF": row["auroc_jf"],
#             "AUROC_JFINV": row["auroc_jfinv"],
#             "AUPRC_JF": row["auprc_jf"],
#             "AUPRC_JFINV": row["auprc_jfinv"],
#         })


#     # mean ± std (paper format)
#     for _, row in summary_df.iterrows():
#         seed_rows.append({
#             "type": "mean_std",
#             "setup": row["setup"],
#             "model": row["model"],
#             "training_mode": row["training_mode"],
#             "seed": "all",

#             "AUROC_JF":
#                 f"{row['auroc_jf_mean']:.4f} ± {row['auroc_jf_std']:.4f}",

#             "AUROC_JFINV":
#                 f"{row['auroc_jfinv_mean']:.4f} ± {row['auroc_jfinv_std']:.4f}",

#             "AUPRC_JF":
#                 f"{row['auprc_jf_mean']:.4f} ± {row['auprc_jf_std']:.4f}",

#             "AUPRC_JFINV":
#                 f"{row['auprc_jfinv_mean']:.4f} ± {row['auprc_jfinv_std']:.4f}",
#         })


#     seed_summary_df = pd.DataFrame(seed_rows)

#     seed_summary_csv = os.path.join(
#         fake_result_dir,
#         "Fake_dataset_all_seeds_results.csv"
#     )

#     seed_summary_df.to_csv(seed_summary_csv, index=False)

#     print(f"Saved seed results to: {seed_summary_csv}")
#     print(f"Saved summary results to: {summary_csv}")

#     print("\n" + "=" * 120)
#     print(" SUMMARY ".center(120, "="))
#     print("=" * 120)
#     print(summary_df.to_string(index=False))
#     print("=" * 120)

#     print("Done.")


# if __name__ == "__main__":
#     main()




# #$#$#$#$#$#$#$#$#

# #$#$#$#$#$#$#$#$#

# #$#$#$#$#$#$#$#$#
# # import os
# # import gc
# # import random
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import pandas as pd

# # from sklearn.metrics import roc_auc_score
# # from utils.min_distance import min_l2_distance
# # from utils.constants import CEBRA_DIR

# # import sys
# # sys.path.insert(0, str(CEBRA_DIR))
# # import cebra
# # from cebra import CEBRA


# # # ============================================================
# # # 1) Global Config
# # # ============================================================
# # T = 100_000
# # N_MLP_LAYERS = 4
# # SIGMA_EPS_DEFAULT = 0.03

# # BATCH_SIZE = 2048
# # MAX_ITER = 15000
# # ATTR_BATCH_SIZE = 128

# # OUT_DIR = "outputs"
# # os.makedirs(OUT_DIR, exist_ok=True)

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # SEEDS = [38]

# # # One dataset only
# # DATASET_CFG = {
# #     "name": "FIG5_SINGLE",
# #     "D1": 3,
# #     "D2": 3,
# #     "N1": 25,
# #     "N2": 25,
# #     "sigma_eps": SIGMA_EPS_DEFAULT,
# # }


# # # ============================================================
# # # 2) Reproducibility
# # # ============================================================
# # def set_all_seeds(seed: int) -> None:
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed(seed)
# #         torch.cuda.manual_seed_all(seed)
# #     try:
# #         torch.backends.cudnn.deterministic = True
# #         torch.backends.cudnn.benchmark = False
# #     except Exception:
# #         pass


# # # ============================================================
# # # 3) Synthetic Data Generation
# # # ============================================================
# # class ScaledTanh(nn.Module):
# #     def __init__(self, scale=1.0):
# #         super().__init__()
# #         self.scale = float(scale)

# #     def forward(self, x):
# #         return self.scale * torch.tanh(x)


# # def make_mlp(in_dim, out_dim, n_layers=4, seed=0):
# #     """
# #     3 hidden layers with GELU, then one output layer + scaled tanh.
# #     IMPORTANT: move the model to the same device as the inputs.
# #     """
# #     torch.manual_seed(seed)

# #     layers = []
# #     d_in = in_dim
# #     hidden = max(64, 8 * max(in_dim, out_dim))

# #     for _ in range(n_layers - 1):
# #         lin = nn.Linear(d_in, hidden)
# #         nn.init.orthogonal_(lin.weight)
# #         nn.init.zeros_(lin.bias)
# #         layers += [lin, nn.GELU()]
# #         d_in = hidden

# #     lin = nn.Linear(d_in, out_dim)
# #     nn.init.orthogonal_(lin.weight)
# #     nn.init.zeros_(lin.bias)
# #     layers += [lin, ScaledTanh(scale=1.0)]

# #     mlp = nn.Sequential(*layers).to(device).eval()
# #     for p in mlp.parameters():
# #         p.requires_grad_(False)
# #     return mlp


# # def brownian_motion_box(T, d, sigma=0.03, seed=0):
# #     """
# #     Brownian motion in [-1, 1]^d.
# #     Uses rejection sampling so the latent always stays in the box.
# #     """
# #     rng = np.random.default_rng(seed)
# #     z = np.empty((T, d), dtype=np.float32)
# #     z[0] = rng.uniform(-1.0, 1.0, size=d).astype(np.float32)

# #     for t in range(1, T):
# #         prev = z[t - 1].copy()
# #         nxt = prev + rng.normal(0.0, sigma, size=d).astype(np.float32)

# #         mask = (nxt < -1.0) | (nxt > 1.0)
# #         while np.any(mask):
# #             nxt[mask] = prev[mask] + rng.normal(0.0, sigma, size=mask.sum()).astype(np.float32)
# #             mask = (nxt < -1.0) | (nxt > 1.0)

# #         z[t] = nxt

# #     return z


# # def make_binary_ground_truth(D1, D2, N1, N2):
# #     """
# #     Ground truth map:
# #       rows = latent variables [z1, z2]
# #       cols = observed neurons [x1, x2]

# #     z1 -> x1 and x2
# #     z2 -> x2 only
# #     """
# #     gt = np.zeros((D1 + D2, N1 + N2), dtype=bool)
# #     gt[:D1, :] = True
# #     gt[D1:, N1:] = True
# #     return gt


# # def generate_synthetic_data(cfg, seed=42):
# #     D1 = int(cfg["D1"])
# #     D2 = int(cfg["D2"])
# #     N1 = int(cfg["N1"])
# #     N2 = int(cfg["N2"])
# #     sigma_eps = float(cfg.get("sigma_eps", SIGMA_EPS_DEFAULT))

# #     z1 = brownian_motion_box(T, D1, sigma=sigma_eps, seed=seed)
# #     z2 = brownian_motion_box(T, D2, sigma=sigma_eps, seed=seed + 1)

# #     # mixing functions (new random mixing per seed)
# #     g1 = make_mlp(D1, N1, n_layers=N_MLP_LAYERS, seed=seed + 10)
# #     g2 = make_mlp(D1 + D2, N2, n_layers=N_MLP_LAYERS, seed=seed + 20)

# #     z1_t = torch.tensor(z1, dtype=torch.float32, device=device)
# #     z2_t = torch.tensor(z2, dtype=torch.float32, device=device)

# #     with torch.no_grad():
# #         x1 = g1(z1_t).cpu().numpy()
# #         x2 = g2(torch.cat([z1_t, z2_t], dim=1)).cpu().numpy()

# #     x = np.concatenate([x1, x2], axis=1).astype(np.float32)
# #     latent = np.concatenate([z1, z2], axis=1).astype(np.float32)

# #     gt_bool = make_binary_ground_truth(D1, D2, N1, N2)
# #     gt_attr = gt_bool.astype(np.float32)

# #     return x, latent, gt_attr, gt_bool


# # # ============================================================
# # # 4) Utils
# # # ============================================================
# # def cleanup_cuda(*objs):
# #     for obj in objs:
# #         try:
# #             del obj
# #         except Exception:
# #             pass
# #     gc.collect()
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()
# #         torch.cuda.ipc_collect()


# # def reduce_attr_map(arr):
# #     """
# #     Convert attribution output to 2D map [output_dim, input_dim]
# #     if it has sample dimension.
# #     """
# #     if torch.is_tensor(arr):
# #         arr = arr.detach().cpu().numpy()
# #     else:
# #         arr = np.asarray(arr)

# #     if arr.ndim == 3:
# #         return np.abs(arr).mean(axis=0)
# #     if arr.ndim == 2:
# #         return np.abs(arr)
# #     if arr.ndim == 1:
# #         return np.abs(arr)[None, :]
# #     raise ValueError(f"Unsupported attribution shape: {arr.shape}")


# # def align_attr_to_gt(attr_map_2d, gt_bool):
# #     if attr_map_2d.shape == gt_bool.shape:
# #         return attr_map_2d
# #     if attr_map_2d.T.shape == gt_bool.shape:
# #         return attr_map_2d.T
# #     raise ValueError(
# #         f"Cannot align attribution map shape {attr_map_2d.shape} "
# #         f"to ground truth shape {gt_bool.shape}"
# #     )


# # def compute_auroc(attr_map_2d, gt_bool):
# #     y_true = gt_bool.ravel().astype(int)
# #     y_score = np.asarray(attr_map_2d, dtype=np.float64).ravel()

# #     if y_true.shape[0] != y_score.shape[0]:
# #         raise ValueError(
# #             f"Shape mismatch: y_true has {y_true.shape[0]} elements, "
# #             f"y_score has {y_score.shape[0]} elements, "
# #             f"attr_map shape={attr_map_2d.shape}, gt shape={gt_bool.shape}"
# #         )

# #     if len(np.unique(y_true)) < 2:
# #         return float("nan")

# #     return float(roc_auc_score(y_true, y_score))


# # # ============================================================
# # # 5) Training + Attribution
# # # ============================================================
# # def run_one_model(
# #     cfg,
# #     seed,
# #     train_data,
# #     train_continuous_label,
# #     gt_attr_bool,
# #     training_mode,
# #     adv_epsilon,
# # ):
# #     D1 = int(cfg["D1"])
# #     D2 = int(cfg["D2"])
# #     N1 = int(cfg["N1"])
# #     N2 = int(cfg["N2"])
# #     D_LATENT = D1 + D2

# #     model_name = "ACORN" if training_mode == "adversarial" else "CEBRA"

# #     set_all_seeds(seed)

# #     model = CEBRA(
# #         batch_size=BATCH_SIZE,
# #         temperature=0.4,
# #         model_architecture="offset36-model-more-dropout",
# #         time_offsets=4,
# #         max_iterations=MAX_ITER,
# #         output_dimension=D_LATENT,
# #         verbose=True,
# #         training_mode=training_mode,
# #         adv_alpha=0.1 / 5,
# #         adv_epsilon=0.1,
# #         adv_steps=10,
# #         attack_norm="linf",   # keep your own setting
# #         num_hidden_units=32,
# #     )

# #     model.fit(train_data, train_continuous_label)

# #     save_path = os.path.join(OUT_DIR, f"{cfg['name']}_seed{seed}_{model_name}.pth")
# #     try:
# #         model.save(save_path)
# #         print("Saved model to:", save_path)
# #     except Exception as e:
# #         print("Could not save model:", e)

# #     trained_model = model.solver_.model.to(device)

# #     input_tensor = torch.from_numpy(train_data).float().to(device).requires_grad_(True)

# #     method = cebra.attribution.init(
# #         name="jacobian-based-batched",
# #         model=trained_model,
# #         input_data=input_tensor,
# #         output_dimension=D_LATENT,
# #     )

# #     result = method.compute_attribution_map(batch_size=min(ATTR_BATCH_SIZE, len(train_data)))
# #     print("Attribution keys:", list(result.keys()))

# #     jac_raw = reduce_attr_map(result["jf"])
# #     jac_inv_raw = reduce_attr_map(result["jf-inv-svd"])

# #     jac_map = align_attr_to_gt(jac_raw, gt_attr_bool)
# #     jac_inv_map = align_attr_to_gt(jac_inv_raw, gt_attr_bool)

# #     auc_jac = compute_auroc(jac_map, gt_attr_bool)
# #     auc_jac_inv = compute_auroc(jac_inv_map, gt_attr_bool)

# #     print(f"** {cfg['name']} | seed={seed} | {model_name} jac AUROC:     {auc_jac:.4f} **")
# #     print(f"** {cfg['name']} | seed={seed} | {model_name} jac_inv AUROC: {auc_jac_inv:.4f} **")

# #     cleanup_cuda(method, trained_model, input_tensor, model)

# #     return {
# #         "setup": cfg["name"],
# #         "seed": seed,
# #         "D1": D1,
# #         "D2": D2,
# #         "N1": N1,
# #         "N2": N2,
# #         "D_LATENT": D_LATENT,
# #         "D_OBS": N1 + N2,
# #         "model": model_name,
# #         "training_mode": training_mode,
# #         "jac_auc": auc_jac,
# #         "jac_inv_auc": auc_jac_inv,
# #     }


# # # ============================================================
# # # 6) Main
# # # ============================================================
# # all_rows = []

# # print("\n" + "#" * 90)
# # print(f"SETUP: {DATASET_CFG['name']} | D1={DATASET_CFG['D1']} D2={DATASET_CFG['D2']} | N1={DATASET_CFG['N1']} N2={DATASET_CFG['N2']} | sigma={DATASET_CFG['sigma_eps']}")
# # print("#" * 90)

# # set_all_seeds(42)
# # x_np, y_np, gt_attr, gt_attr_bool = generate_synthetic_data(DATASET_CFG, seed=42)

# # print("x shape:", x_np.shape)
# # print("y shape:", y_np.shape)
# # print("gt_attr_bool shape:", gt_attr_bool.shape)

# # split_idx = int(0.8 * len(x_np))
# # train_data = x_np[:split_idx].astype(np.float32)
# # train_continuous_label = y_np[:split_idx].astype(np.float32)

# # adv_epsilon = float(min_l2_distance(train_data)) / 2.0
# # adv_epsilon = max(adv_epsilon, 1e-6)
# # print("adv_epsilon:", adv_epsilon)

# # for seed in SEEDS:
# #     for training_mode in ["clean", "adversarial"]:
# #         cleanup_cuda()

# #         print("\n" + "=" * 70)
# #         print(f"Training {DATASET_CFG['name']} | seed={seed} | mode={training_mode}")
# #         print("=" * 70)

# #         row = run_one_model(
# #             cfg=DATASET_CFG,
# #             seed=seed,
# #             train_data=train_data,
# #             train_continuous_label=train_continuous_label,
# #             gt_attr_bool=gt_attr_bool,
# #             training_mode=training_mode,
# #             adv_epsilon=adv_epsilon,
# #         )
# #         all_rows.append(row)

# # results_df = pd.DataFrame(all_rows)

# # detailed_csv = os.path.join(OUT_DIR, "synthetic_auroc_detailed.csv")
# # results_df.to_csv(detailed_csv, index=False)
# # print(f"\nSaved detailed results to: {detailed_csv}")

# # summary_df = (
# #     results_df
# #     .groupby(["setup", "model", "training_mode", "D1", "D2", "N1", "N2", "D_LATENT", "D_OBS"], as_index=False)
# #     .agg(
# #         jac_mean=("jac_auc", "mean"),
# #         jac_std=("jac_auc", "std"),
# #         jac_inv_mean=("jac_inv_auc", "mean"),
# #         jac_inv_std=("jac_inv_auc", "std"),
# #         n_runs=("seed", "count"),
# #     )
# # )

# # summary_csv = os.path.join(OUT_DIR, "synthetic_auroc_summary.csv")
# # summary_df.to_csv(summary_csv, index=False)
# # print(f"Saved summary results to: {summary_csv}")

# # print("\n" + "=" * 120)
# # print(" SUMMARY ".center(120, "="))
# # print("=" * 120)
# # print(summary_df.to_string(index=False))
# # print("=" * 120)

# # print("Done.")

# # import os
# # import gc
# # import random
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import pandas as pd

# # from sklearn.metrics import roc_auc_score
# # from utils.min_distance import min_l2_distance
# # from utils.constants import CEBRA_DIR

# # import sys
# # sys.path.insert(0, str(CEBRA_DIR))
# # import cebra
# # from cebra import CEBRA


# # # ============================================================
# # # 1) Synthetic Data Config & Generation
# # # ============================================================
# # T = 100_000
# # D1 = 3
# # D2 = 3
# # D_LATENT = D1 + D2

# # N1 = 2
# # N2 = 2
# # D_OBS = N1 + N2

# # N_MLP_LAYERS = 4
# # SIGMA_EPS = 0.03

# # OUTPUT_DIM = D_LATENT
# # BATCH_SIZE = 2048
# # MAX_ITER = 2500
# # adv_epsilon_default = 0.5

# # ATTR_BATCH_SIZE = 128

# # OUT_DIR = "outputs"
# # os.makedirs(OUT_DIR, exist_ok=True)

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # RANDOM_SEED = 42
# # np.random.seed(RANDOM_SEED)
# # torch.manual_seed(RANDOM_SEED)
# # random.seed(RANDOM_SEED)


# # def make_mlp(in_dim, out_dim, n_layers=4, seed=0):
# #     torch.manual_seed(seed)
# #     layers = []
# #     d_in = in_dim
# #     hidden = in_dim * 10

# #     for i in range(n_layers - 1):
# #         d_h = in_dim * 30 if i < n_layers - 2 else hidden
# #         lin = nn.Linear(d_in, d_h)
# #         nn.init.orthogonal_(lin.weight)
# #         nn.init.zeros_(lin.bias)
# #         layers += [lin, nn.GELU()]
# #         d_in = d_h

# #     lin = nn.Linear(d_in, out_dim)
# #     nn.init.orthogonal_(lin.weight)
# #     nn.init.zeros_(lin.bias)
# #     layers.append(lin)

# #     mlp = nn.Sequential(*layers)
# #     for p in mlp.parameters():
# #         p.requires_grad_(False)
# #     return mlp.eval()


# # def brownian_motion_box(T, d, sigma=0.03, seed=0):
# #     rng = np.random.default_rng(seed)
# #     x = np.zeros((T, d), dtype=np.float32)
# #     x[0] = rng.uniform(-1.0, 1.0, size=d).astype(np.float32)

# #     for t in range(T - 1):
# #         step = rng.normal(loc=0.0, scale=sigma, size=d).astype(np.float32)
# #         x[t + 1] = np.clip(x[t] + step, -1.0, 1.0)
# #     return x


# # def make_binary_ground_truth(D1, D2, N1, N2):
# #     """
# #     Ground truth:
# #     - first D1 latents connect to all neurons
# #     - last D2 latents connect only to x2 block (last N2 neurons)
# #     shape = [D_LATENT, D_OBS]
# #     """
# #     gt = np.zeros((D1 + D2, N1 + N2), dtype=bool)
# #     gt[:D1, :] = True
# #     gt[D1:, N1:] = True
# #     return gt


# # def generate_synthetic_data(T=T, seed=42):
# #     z1 = brownian_motion_box(T, D1, sigma=SIGMA_EPS, seed=seed)
# #     z2 = brownian_motion_box(T, D2, sigma=SIGMA_EPS, seed=seed + 1)

# #     g1 = make_mlp(D1, N1, n_layers=N_MLP_LAYERS, seed=seed + 10)
# #     g2 = make_mlp(D1 + D2, N2, n_layers=N_MLP_LAYERS, seed=seed + 20)

# #     z1_t = torch.tensor(z1, dtype=torch.float32)
# #     z2_t = torch.tensor(z2, dtype=torch.float32)

# #     with torch.no_grad():
# #         x1 = g1(z1_t).cpu().numpy()
# #         x2 = g2(torch.cat([z1_t, z2_t], dim=1)).cpu().numpy()

# #     x = np.concatenate([x1, x2], axis=1).astype(np.float32)
# #     latent = np.concatenate([z1, z2], axis=1).astype(np.float32)

# #     gt_bool = make_binary_ground_truth(D1, D2, N1, N2)
# #     gt_attr = gt_bool.astype(np.float32)

# #     return x, latent, gt_attr, gt_bool


# # # ============================================================
# # # 2) Utils
# # # ============================================================
# # def cleanup_cuda(*objs):
# #     for obj in objs:
# #         try:
# #             del obj
# #         except Exception:
# #             pass
# #     gc.collect()
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()
# #         torch.cuda.ipc_collect()


# # def reduce_attr_map(arr):
# #     """
# #     Convert attribution output to 2D map [output_dim, input_dim]
# #     if it has sample dimension.
# #     """
# #     arr = np.asarray(arr)
# #     if arr.ndim == 3:
# #         return np.abs(arr).mean(axis=0)
# #     if arr.ndim == 2:
# #         return np.abs(arr)
# #     if arr.ndim == 1:
# #         return np.abs(arr)[None, :]
# #     raise ValueError(f"Unsupported attribution shape: {arr.shape}")


# # def compute_auroc(attr_map_2d, gt_bool):
# #     y_true = gt_bool.ravel().astype(int)
# #     y_score = np.asarray(attr_map_2d, dtype=np.float64).ravel()

# #     if y_true.shape[0] != y_score.shape[0]:
# #         raise ValueError(
# #             f"Shape mismatch: y_true has {y_true.shape[0]} elements, "
# #             f"y_score has {y_score.shape[0]} elements, "
# #             f"attr_map shape={attr_map_2d.shape}, gt shape={gt_bool.shape}"
# #         )

# #     if len(np.unique(y_true)) < 2:
# #         return float("nan")

# #     return float(roc_auc_score(y_true, y_score))


# # # ============================================================
# # # 3) Data Generation
# # # ============================================================
# # print("Generating synthetic dataset...")
# # x_np, y_np, gt_attr, gt_attr_bool = generate_synthetic_data(T=T, seed=42)

# # print("x shape:", x_np.shape)  # (T, 6)
# # print("y shape:", y_np.shape)  # (T, 4)
# # print("gt_attr_bool shape:", gt_attr_bool.shape)  # (4, 6)

# # split_idx = int(0.8 * len(x_np))
# # train_data = x_np[:split_idx].astype(np.float32)
# # train_continuous_label = y_np[:split_idx].astype(np.float32)

# # # For later use in CEBRA setup
# # adv_epsilon = float(min_l2_distance(train_data)) / 2.0
# # adv_epsilon = max(adv_epsilon, 1e-6)

# # # ============================================================
# # # 4) Train + Attribution
# # # ============================================================
# # rows = []
# # all_results = {}

# # for adv in [False, True]:
# #     cleanup_cuda()

# #     model_name = "ACORN" if adv else "CEBRA"
# #     training_mode = "adversarial" if adv else "clean"

# #     print("\n" + "=" * 70)
# #     print(f"Training {model_name} ({training_mode})")
# #     print("=" * 70)

# #     model = CEBRA(
# #         batch_size=BATCH_SIZE,
# #         temperature=0.4,
# #         model_architecture="offset36-model-more-dropout",
# #         time_offsets=4,
# #         max_iterations=MAX_ITER,
# #         output_dimension=OUTPUT_DIM,
# #         verbose=True,
# #         training_mode=training_mode,
# #         adv_alpha=adv_epsilon / 5,
# #         adv_epsilon=adv_epsilon,
# #         adv_steps=10,
# #         attack_norm="linf",
# #         num_hidden_units=32,
# #     )

# #     model.fit(train_data, train_continuous_label)

# #     save_path = os.path.join(OUT_DIR, f"{model_name}_synthetic.pth")
# #     model.save(save_path)
# #     print("Saved model to:", save_path)

# #     trained_model = model.solver_.model.to(device)

# #     # Use full training data for attribution; batch_size smaller to avoid errors
# #     input_tensor = torch.from_numpy(train_data).float().to(device).requires_grad_(True)

# #     output_dim = int(getattr(trained_model, "num_output", OUTPUT_DIM))

# #     method = cebra.attribution.init(
# #         name="jacobian-based-batched",
# #         model=trained_model,
# #         input_data=input_tensor,
# #         output_dimension=output_dim,
# #     )

# #     result = method.compute_attribution_map(batch_size=min(128, len(train_data)))
# #     print("Attribution keys:", list(result.keys()))

# #     # Reduce to 2D maps
# #     jc_map = reduce_attr_map(result["jf"])
# #     jc_inv_map = reduce_attr_map(result["jf-inv-svd"])
# #     jc_invconv_map = reduce_attr_map(result["jf-convabs-inv-svd"])

# #     # AUROC scores
# #     auc_jc = compute_auroc(jc_map, gt_attr_bool)
# #     auc_jc_inv = compute_auroc(jc_inv_map, gt_attr_bool)
# #     auc_jc_invconv = compute_auroc(jc_invconv_map, gt_attr_bool)

# #     print(f"** {model_name} jc AUROC:        {auc_jc:.4f} **")
# #     print(f"** {model_name} jc_inv AUROC:    {auc_jc_inv:.4f} **")
# #     print(f"** {model_name} jc_invconv AUROC:{auc_jc_invconv:.4f} **")

# #     all_results[model_name] = {
# #         "jc": jc_map,
# #         "jc_inv": jc_inv_map,
# #         "jc_invconv": jc_invconv_map,
# #         "auc_jc": auc_jc,
# #         "auc_jc_inv": auc_jc_inv,
# #         "auc_jc_invconv": auc_jc_invconv,
# #     }

# #     rows.extend([
# #         {"model": model_name, "metric": "jc", "auroc": auc_jc},
# #         {"model": model_name, "metric": "jc_inv", "auroc": auc_jc_inv},
# #         {"model": model_name, "metric": "jc_invconv", "auroc": auc_jc_invconv},
# #     ])

# #     cleanup_cuda(method, trained_model, input_tensor, model)

# # # ============================================================
# # # 5) Summary
# # # ============================================================
# # print("\n" + "=" * 80)
# # print(" SUMMARY OF EXPERIMENT RESULTS ".center(80, "="))
# # print("=" * 80)
# # for model_name, res in all_results.items():
# #     print(
# #         f" Model: {model_name:<6} | "
# #         f"jc={res['auc_jc']:.4f} | "
# #         f"jc_inv={res['auc_jc_inv']:.4f} | "
# #         f"jc_invconv={res['auc_jc_invconv']:.4f}"
# #     )
# # print("=" * 80)

# # # ============================================================
# # # 6) Save CSV
# # # ============================================================
# # results_df = pd.DataFrame(rows)
# # csv_path = os.path.join(OUT_DIR, "synthetic_auroc_results.csv")
# # results_df.to_csv(csv_path, index=False)
# # print(f"Saved AUROC results to: {csv_path}")

# # print("Done.")

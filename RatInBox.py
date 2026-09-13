# =============================================================================
#  xCEBRA / RatInABox  —  CLEAN vs ACORN, eps x norm SWEEP
#  Official cebra 0.6.0a1 multiobjective API (no fork).
#
#  MODE="calibrate"  ~6 min. Trains ONE clean model for CAL_STEPS, then measures
#                    for every (norm, eps) what fraction of the model's LEARNED
#                    MARGIN the attack destroys, on a FIXED set of batches.
#  MODE="sweep"      Full run. Baselines (eps-independent) trained ONCE per seed;
#                    only noise/acorn arms are swept. Resumes from results.csv.
#
#  WHY THE FIRST RUN FAILED: eps=0.03 linf gave gap=+0.046 on clean=13.054.
#  FixedCosineInfoNCE at tau=1 has cosine logits in [-1,1], so each objective's
#  loss travels only from log(2500)=7.82 to ~5.82. Chance = 2*log(B) = 15.65, so
#  margin = 2.59 nats and the attack erased 1.8% of it. Target destroy_frac
#  20-50%.
# =============================================================================

import os, sys, copy, json, time, math, pickle, itertools, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path

# --- force the ORIGINAL cebra --------------------------------------------------
CEBRA_DIR = Path(__file__).resolve().parent / "CEBRA-original"
for _m in list(sys.modules):
    if _m == "cebra" or _m.startswith("cebra."):
        del sys.modules[_m]
sys.path.insert(0, str(CEBRA_DIR))

import cebra
from cebra.data import DatasetxCEBRA, ContrastiveMultiObjectiveLoader, TensorDataset
from cebra.solver import MultiObjectiveConfig
from cebra.solver.schedulers import LinearRampUp

print("cebra loaded from:", cebra.__file__)


# ============================================================== CONFIG
MODE        = "calibrate"          # "calibrate" first, then "sweep"

DATA_FILE   = "cynthi_neurons90.p"
DATA_URL    = ("https://zenodo.org/records/15267195/files/"
               "cynthi_neurons90_gridbase0.5_gridmodules3_grid_head_direction_place_speed"
               "_duration2000_noise0.25_bs100_seed231209234.p?download=1")
OUT_DIR     = "xcebra_ratinabox_sweep"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# --- model / training (official demo notebook values) -------------------------
NUM_STEPS   = 25000
BATCH_SIZE  = 2500
N_LATENTS   = 14
NUM_UNITS   = 256
MODEL_ARCH  = "offset10-model"
LR          = 3e-4
TAU         = 1.0
BEHAVIOR_RANGE = (0, 4)
TIME_RANGE     = (0, N_LATENTS)
TIME_DELTA     = 1
TIME_OFFSET    = 10
RENORMALIZE    = True
LAMBDA_MAX     = 0.1

# --- attack -------------------------------------------------------------------
ADV_STEPS     = 10
ALPHA_RULE    = 0.2                # alpha = ALPHA_RULE * eps
ATTACK_OBJ    = "infonce"          # "infonce" | "vat"
CLAMP_TO_DATA = True
CLAMP_RANGE   = (0.0, 1.0)         # data verified to live exactly in [0,1]
JREG_AT       = "clean"            # penalty only on the clean update

# eps semantics (both RELATIVE so the two norms are comparable):
#   linf : eps = EPS_REL * (hi - lo)
#   l2   : eps = EPS_REL * mean_t ||x_t||_2
CAL_STEPS      = 4000              # clean pre-training for calibration
CAL_BATCHES    = 12                # probe batches per (norm, eps) cell
CAL_GRID       = [0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
CAL_NORMS      = ["linf", "l2"]

# --- the sweep itself (fill in after reading the calibration table) -----------
SWEEP_NORMS   = ["linf", "l2"]
SWEEP_EPS     = [0.05, 0.10, 0.20, 0.35]
SWEEP_ARMS    = ["noise", "acorn", "acorn_xreg"]
BASELINE_ARMS = ["cebra", "cebra_2x", "xcebra", "xcebra_2x"]

SEEDS         = [0, 1, 2]
ATTACK_LOG_EVERY = 250
ATTR_MAX_SAMPLES = None
os.makedirs(OUT_DIR, exist_ok=True)

BASE_CFG = {
    "cebra":      dict(lam=0.0,        adv=None,    double=False),
    "cebra_2x":   dict(lam=0.0,        adv=None,    double=True ),
    "xcebra":     dict(lam=LAMBDA_MAX, adv=None,    double=False),
    "xcebra_2x":  dict(lam=LAMBDA_MAX, adv=None,    double=True ),
    "noise":      dict(lam=0.0,        adv="noise", double=True ),
    "acorn":      dict(lam=0.0,        adv="pgd",   double=True ),
    "acorn_xreg": dict(lam=LAMBDA_MAX, adv="pgd",   double=True ),
}


# ============================================================== DATA
def get_data():
    if not os.path.exists(DATA_FILE):
        import requests
        print("downloading dataset ...")
        with open(DATA_FILE, "wb") as f:
            f.write(requests.get(DATA_URL).content)
    with open(DATA_FILE, "rb") as f:
        d = pickle.load(f)
    return (d,
            torch.FloatTensor(d["spikes"]).float(),
            torch.FloatTensor(d["position"]).float())


def compute_eps(norm, eps_rel, lo, hi, ts_norm):
    return eps_rel * (hi - lo) if norm == "linf" else eps_rel * ts_norm


# ============================================================== GROUND TRUTH
def build_ground_truth(n_neurons, split="notebook"):
    """Cell-type gt from the notebook. `cells` has NO 'speed' entries, so 10-11
    of the 14 rows are all-False. Kept only for continuity with the published
    numbers; `auroc_rownorm` on it was at chance for every arm."""
    cells = np.array(list(itertools.chain.from_iterable(
        [["position"] * 100, ["hd"] * 100, ["position"] * 100, ["grid"] * 60])))
    if len(cells) != n_neurons:
        warnings.warn("cell-type list length mismatch -- label gt disabled.")
        return None, cells
    n_beh = 3 if split == "notebook" else BEHAVIOR_RANGE[1]
    latents = [["position", "grid"]] * n_beh + [["speed"]] * (N_LATENTS - n_beh)
    gt = np.zeros((len(latents), len(cells)), dtype=bool)
    for i, lat in enumerate(latents):
        for j, ct in enumerate(cells):
            gt[i, j] = ct in lat
    return gt, cells


def empirical_tuning(neural, position, n_pos_bins=12, n_1d_bins=16):
    """Decoder-free, notebook-free gt: how strongly each neuron is actually
    tuned to position / speed / movement direction, as eta^2 (fraction of the
    neuron's variance explained by binning the variable). speed and hd are
    DERIVED from position, so no extra data is needed."""
    X = neural.numpy().astype(np.float64)
    P = position.numpy().astype(np.float64)
    vel   = np.diff(P, axis=0, prepend=P[:1])
    speed = np.linalg.norm(vel, axis=1)
    hd    = np.arctan2(vel[:, 1], vel[:, 0])

    def digi(v, nb):
        q = np.quantile(v, np.linspace(0, 1, nb + 1)[1:-1])
        return np.clip(np.digitize(v, q), 0, nb - 1).astype(np.int64)

    def eta2(idx, n_groups):
        tot = X.var(axis=0) + 1e-12
        cnt = np.bincount(idx, minlength=n_groups)
        sums = np.zeros((n_groups, X.shape[1]))
        np.add.at(sums, idx, X)
        means = sums / np.maximum(cnt, 1)[:, None]
        within = ((X - means[idx]) ** 2).mean(axis=0)
        return np.clip(1.0 - within / tot, 0.0, 1.0)

    bx, by = digi(P[:, 0], n_pos_bins), digi(P[:, 1], n_pos_bins)
    return {
        "position": eta2(bx * n_pos_bins + by, n_pos_bins ** 2),
        "speed":    eta2(digi(speed, n_1d_bins), n_1d_bins),
        "hd":       eta2(digi(hd,    n_1d_bins), n_1d_bins),
    }


# ============================================================== METRICS
def _avg_rank(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = _avg_rank(np.asarray(a, float)), _avg_rank(np.asarray(b, float))
    return float(np.corrcoef(ra, rb)[0, 1])


def auroc(scores, labels):
    s = np.asarray(scores, float).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _avg_rank(s)
    return (r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auroc_variants(M, gt):
    M = np.asarray(M, float)
    return {"global":  auroc(M, gt),
            "colnorm": auroc(M / (M.mean(axis=0, keepdims=True) + 1e-12), gt),
            "rownorm": auroc(M / (M.mean(axis=1, keepdims=True) + 1e-12), gt)}


def spectral_stats(J):
    s = np.linalg.svd(np.asarray(J, float), compute_uv=False)
    return dict(sv_max=float(s[0]), sv_min=float(s[-1]),
                sv_ratio=float(s[-1] / (s[0] + 1e-30)),
                fro=float(np.sqrt((s ** 2).sum())))


def sparsity_stats(J):
    s = np.clip(np.asarray(J, float).mean(axis=0), 0, None)
    tot = s.sum() + 1e-30
    p = s / tot
    pr = (s.sum() ** 2) / ((s ** 2).sum() + 1e-30)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    srt = np.sort(s); cum = np.cumsum(srt) / tot
    return dict(participation_ratio=float(pr), entropy=ent,
                gini=float(1 - 2 * np.trapz(cum, dx=1.0 / len(srt))),
                n_neurons=int(len(s)))


# ============================================================== ATTACK
def _l2_norm_per_timestep(t):
    """L2 across the CHANNEL axis only, per time-step. batch.reference is
    (B, C, T) for CEBRA conv models, so dim=1 is the neuron axis."""
    return t.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)


def _replace_reference(batch, new_ref):
    """copy.copy keeps `positive` (a LIST, one tensor per objective), `negative`
    and any index fields untouched; only the reference stream is swapped."""
    nb = copy.copy(batch)
    try:
        nb.reference = new_ref
    except Exception:
        object.__setattr__(nb, "reference", new_ref)
    return nb


def _batch_to(batch, device):
    """Move a multiobjective Batch to `device`. `positive` is a LIST."""
    nb = copy.copy(batch)
    vals = (("reference", batch.reference.to(device, non_blocking=True)),
            ("negative",  batch.negative.to(device, non_blocking=True)),
            ("positive",  [p.to(device, non_blocking=True)
                           for p in batch.positive]))
    for name, val in vals:
        try:
            setattr(nb, name, val)
        except Exception:
            object.__setattr__(nb, name, val)
    return nb


def make_adversarial_solver(solver, cfg, atk):
    """Rebind solver.__class__ to a subclass of whatever cebra.solver.init built.
    MultiCriterion and _inference are untouched; only `step`, through super().

    cfg/atk are read off `self` (not captured) so calibration can mutate
    solver._atk["eps"] between probes without rebuilding anything."""
    Base = type(solver)

    class AdversarialSolver(Base):

        def _adv_loss(self, x_adv, pos_out, neg_out):
            """The solver's OWN criterion, with pos/neg embeddings held fixed.
            Mirrors ContrastiveMultiobjectiveSolverxCEBRA._inference exactly."""
            ref_out = self.model(x_adv)
            preds = [cebra.data.Batch(reference=ref_out[i],
                                      positive=pos_out[i],
                                      negative=neg_out[i])
                     for i in range(len(pos_out))]
            return sum(self.criterion(preds))

        def _make_adv_batch(self, batch, record=False):
            a, c = self._atk, self._adv_cfg
            eps, alpha, steps = a["eps"], a["alpha"], a["steps"]
            norm, obj         = a["norm"], a["objective"]
            clamp, lo, hi     = a["clamp"], a["lo"], a["hi"]
            mode              = c["adv"]

            ref = batch.reference.detach()

            if norm == "linf":
                delta = torch.empty_like(ref).uniform_(-eps, eps)
            else:
                v = torch.randn_like(ref)
                delta = eps * v / _l2_norm_per_timestep(v)

            def _finish(d):
                x = ref + d
                if clamp:
                    x = torch.clamp(x, lo, hi)
                return _replace_reference(batch, x.detach())

            n_obj = len(batch.positive)
            with torch.no_grad():
                neg_all = self.model(batch.negative)
                neg_out = [neg_all[i] for i in range(n_obj)]
                pos_out = [self.model(p)[i] for i, p in enumerate(batch.positive)]
                if obj == "vat":
                    ref_out0 = [t.detach() for t in self.model(ref)]
                clean_loss = (float(self._adv_loss(ref, pos_out, neg_out))
                              if (record and obj != "vat") else None)

            if mode == "noise" or steps == 0:
                adv_batch = _finish(delta)
            else:
                delta.requires_grad_(True)
                for _ in range(steps):
                    x = torch.clamp(ref + delta, lo, hi) if clamp else ref + delta
                    if obj == "vat":
                        out = self.model(x)
                        loss = sum(((out[i] - ref_out0[i]) ** 2).sum(1).mean()
                                   for i in range(len(out)))
                    else:
                        loss = self._adv_loss(x, pos_out, neg_out)
                    g, = torch.autograd.grad(loss, delta)
                    with torch.no_grad():
                        if norm == "linf":
                            delta.add_(alpha * g.sign()).clamp_(-eps, eps)
                        else:
                            delta.add_(alpha * g / _l2_norm_per_timestep(g))
                            delta.mul_(torch.clamp(
                                eps / _l2_norm_per_timestep(delta), max=1.0))
                adv_batch = _finish(delta.detach())

            # --- the diagnostic that decides whether eps is usable ------------
            if record and clean_loss is not None:
                with torch.no_grad():
                    adv_loss = float(self._adv_loss(
                        adv_batch.reference, pos_out, neg_out))
                chance = n_obj * math.log(ref.shape[0])
                margin = max(chance - clean_loss, 1e-6)
                self._adv_log.append((clean_loss, adv_loss, margin))
            return adv_batch

        def step(self, *args, **kwargs):
            batch = args[0]
            self._adv_n = getattr(self, "_adv_n", 0) + 1
            rec = (self._adv_n % ATTACK_LOG_EVERY == 0)
            c = self._adv_cfg

            if not c["double"]:
                if c["adv"] is None:
                    return super().step(*args, **kwargs)
                return super().step(self._make_adv_batch(batch, rec),
                                    *args[1:], **kwargs)

            super().step(*args, **kwargs)                    # update 1: clean
            if c["adv"] is None:
                return super().step(*args, **kwargs)         # *_2x controls

            kw2 = dict(kwargs)
            if JREG_AT == "clean":
                kw2["weights_regularizer"] = None            # penalty at x_clean only
            return super().step(self._make_adv_batch(batch, rec),
                                *args[1:], **kw2)

    solver.__class__ = AdversarialSolver
    solver._adv_cfg, solver._atk = cfg, atk
    solver._adv_log, solver._adv_n = [], 0
    return solver


# ============================================================== BUILD
def build(seed, neural, position, num_steps):
    torch.manual_seed(seed); np.random.seed(seed)
    data = DatasetxCEBRA(neural, position=position)
    loader = ContrastiveMultiObjectiveLoader(dataset=data, num_steps=num_steps,
                                            batch_size=BATCH_SIZE).to(DEVICE)
    cfgm = MultiObjectiveConfig(loader)
    cfgm.set_slice(*BEHAVIOR_RANGE)
    cfgm.set_loss("FixedCosineInfoNCE", temperature=TAU)
    cfgm.set_distribution("time_delta", time_delta=TIME_DELTA, label_name="position")
    cfgm.push()
    cfgm.set_slice(*TIME_RANGE)
    cfgm.set_loss("FixedCosineInfoNCE", temperature=TAU)
    cfgm.set_distribution("time", time_offset=TIME_OFFSET)
    cfgm.push()
    cfgm.finalize()

    model = cebra.models.init(name=MODEL_ARCH, num_neurons=data.neural.shape[1],
                              num_units=NUM_UNITS, num_output=N_LATENTS).to(DEVICE)
    data.configure_for(model)
    opt = torch.optim.Adam(list(model.parameters()) +
                           list(cfgm.criterion.parameters()), lr=LR, weight_decay=0)
    solver = cebra.solver.init(name="multiobjective-solver", model=model,
                               feature_ranges=cfgm.feature_ranges,
                               regularizer=cebra.models.jacobian_regularizer.JacobianReg(),
                               renormalize=RENORMALIZE, use_sam=False,
                               criterion=cfgm.criterion, optimizer=opt,
                               tqdm_on=True).to(DEVICE)
    return solver, loader


def train_arm(arm_key, base, seed, neural, position, atk):
    cfg = BASE_CFG[base]
    solver, loader = build(seed, neural, position, NUM_STEPS)
    sched = LinearRampUp(n_splits=2, step_to_switch_on_reg=NUM_STEPS // 4,
                         step_to_switch_off_reg=NUM_STEPS // 2,
                         start_weight=0.0, end_weight=cfg["lam"])
    if cfg["adv"] is not None or cfg["double"]:
        solver = make_adversarial_solver(solver, cfg, atk)

    print(f"\n=== {arm_key} seed={seed} lam={cfg['lam']} adv={cfg['adv']} "
          f"double={cfg['double']} norm={atk['norm']} eps={atk['eps']:.4f} "
          f"alpha={atk['alpha']:.4f} jreg_at={JREG_AT} dev={DEVICE}")
    t0 = time.time()
    solver.fit(loader=loader, valid_loader=None, log_frequency=None,
               scheduler_regularizer=sched, scheduler_loss=None)
    meta = dict(wall_s=time.time() - t0, norm=atk["norm"],
                eps_rel=atk["eps_rel"], eps=atk["eps"], alpha=atk["alpha"],
                base=base)
    log = getattr(solver, "_adv_log", [])
    if log:
        cl = np.array([x[0] for x in log]); ad = np.array([x[1] for x in log])
        mg = np.array([x[2] for x in log])
        meta.update(adv_loss_clean=float(cl.mean()), adv_loss_adv=float(ad.mean()),
                    adv_loss_gap=float((ad - cl).mean()),
                    margin=float(mg.mean()),
                    destroy_frac=float(((ad - cl) / mg).mean()))
        print(f"    attack reach: clean={cl.mean():.4f} adv={ad.mean():.4f} "
              f"gap={np.mean(ad-cl):+.4f} nats  margin={mg.mean():.3f}  "
              f"DESTROYED={100*np.mean((ad-cl)/mg):.1f}%  (n={len(log)})")
    return solver, meta


# ============================================================== EVAL
def _set_split(model, flag):
    if hasattr(model, "set_split_outputs"):
        model.set_split_outputs(flag)
    else:
        model.split_outputs = flag


def compute_embedding(solver, neural):
    d = TensorDataset(neural, continuous=torch.zeros(len(neural), 1))
    d.configure_for(solver.model)
    x = d[torch.arange(len(d))]
    _set_split(solver.model, False)
    with torch.no_grad():
        return solver.model(x.to(DEVICE)).detach().cpu()


def decoding_scores(embedding, position):
    """`residual` [4:14] is the block that reproduces the notebook's "time" KNN
    (0.67 vs 0.69); the full (0,14) range is a superset of the behaviour block
    so it is forced to beat it and cannot be what the notebook plots."""
    Xb = embedding[:, slice(*BEHAVIOR_RANGE)].numpy()
    Xt = embedding[:, slice(*TIME_RANGE)].numpy()
    Xr = embedding[:, BEHAVIOR_RANGE[1]:N_LATENTS].numpy()
    y  = position[:len(embedding)].numpy()
    out, tscv = {}, TimeSeriesSplit(n_splits=5)
    for tag, X in (("behavior", Xb), ("time", Xt), ("residual", Xr)):
        out[f"R2_{tag}"] = LinearRegression().fit(X, y).score(X, y)
        out[f"KNN_{tag}"] = float(np.mean(
            [KNeighborsRegressor().fit(X[tr], y[tr]).score(X[va], y[va])
             for tr, va in tscv.split(X)]))
    return out


def attribution(solver, neural):
    model = solver.model.to(DEVICE)
    _set_split(model, False)
    x = neural if ATTR_MAX_SAMPLES is None else neural[:ATTR_MAX_SAMPLES]
    x = x.clone().to(DEVICE).requires_grad_(True)   # proven to work in run 1
    m = cebra.attribution.init(name="jacobian-based", model=model,
                               input_data=x, output_dimension=model.num_output)
    return m, m.compute_attribution_map()


def evaluate(arm_key, seed, solver, neural, position, gts, tuning, meta):
    row = dict(arm=arm_key, seed=seed, **meta)
    row.update(decoding_scores(compute_embedding(solver, neural), position))
    torch.save(solver.model.state_dict(),
               os.path.join(OUT_DIR, f"model_{arm_key}_s{seed}.pt"))

    method, result = attribution(solver, neural)
    maps = {name: np.abs(result[key]).mean(0)
            for key, name in (("jf", "jf"), ("jf-inv-svd", "jfinv"),
                              ("jf-convabs-inv-svd", "jfconvabsinv"))
            if key in result}
    np.savez_compressed(os.path.join(OUT_DIR, f"maps_{arm_key}_s{seed}.npz"), **maps)

    if "jf" in maps:
        row.update({f"sv_{k}": v for k, v in spectral_stats(maps["jf"]).items()})
        row.update({f"sp_{k}": v for k, v in sparsity_stats(maps["jf"]).items()})
        if "jfinv" in maps:
            # with a flat singular spectrum J_f and pinv(J_f) rank the same
            # neurons -> "you can skip the pinv". Direct corollary of Sigma vs
            # Sigma^-1. Expect HIGH for acorn, LOW for clean.
            row["rank_corr_jf_jfinv"] = spearman(maps["jf"].ravel(),
                                                 maps["jfinv"].ravel())

    # --- cell-type gt (kept for continuity; rownorm is at chance on it) -------
    for gt_name, gt in gts.items():
        if gt is None:
            continue
        for mname, M in maps.items():
            if M.shape != gt.shape:
                continue
            try:
                row[f"auc_{mname}_{gt_name}"] = float(
                    method.compute_attribution_score(M, gt))
            except Exception:
                row[f"auc_{mname}_{gt_name}"] = float("nan")
            for vn, v in auroc_variants(M, gt).items():
                row[f"auroc_{vn}_{mname}_{gt_name}"] = v

    # --- empirical tuning gt: the honest neuron-axis test --------------------
    eta_p, eta_h, eta_s = tuning["position"], tuning["hd"], tuning["speed"]
    pos_top = eta_p >= np.quantile(eta_p, 0.70)
    for mname, M in maps.items():
        colb = M[slice(*BEHAVIOR_RANGE), :].mean(axis=0)   # behaviour-latent score
        row[f"tune_sp_pos_{mname}"]   = spearman(colb, eta_p)
        row[f"tune_sp_hd_{mname}"]    = spearman(colb, eta_h)
        row[f"tune_sp_speed_{mname}"] = spearman(colb, eta_s)
        row[f"tune_auroc_{mname}"]    = auroc(colb, pos_top)
        row[f"tune_sel_{mname}"] = (row[f"tune_sp_pos_{mname}"]
                                    - row[f"tune_sp_hd_{mname}"])

    for mname, M in maps.items():
        plt.figure(figsize=(9, 3.2))
        plt.matshow(M, aspect="auto", fignum=0)
        plt.colorbar(); plt.title(f"{arm_key} (seed {seed}) — {mname}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"map_{arm_key}_s{seed}_{mname}.png"), dpi=140)
        plt.close()
    return row


# ============================================================== CALIBRATE
def cache_probe_batches(neural, position, n):
    """Pull n batches ONCE and park them on CPU, so every (norm, eps) cell sees
    the SAME batches and the destroy_frac differences across the grid are purely
    the attack. Building a throwaway solver here is wasteful, but it is the only
    tested way to get a correctly configured ContrastiveMultiObjectiveLoader."""
    _, loader = build(999, neural, position, n)
    out = []
    for batch in loader:
        out.append(_batch_to(batch, "cpu"))
        if len(out) >= n:
            break
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print(f"cached {len(out)} probe batches "
          f"(reference {tuple(out[0].reference.shape)}, "
          f"{len(out[0].positive)} objectives)")
    return out


def calibrate(neural, position, lo, hi, ts_norm):
    """Train ONE clean model, then measure destroy_frac over the (norm, eps)
    grid on a FIXED set of batches. No adversarial training happens."""
    print(f"\n### CALIBRATION: {CAL_STEPS} clean steps, then probe "
          f"{len(CAL_NORMS)}x{len(CAL_GRID)} cells x {CAL_BATCHES} batches")
    solver, loader = build(0, neural, position, CAL_STEPS)
    sched = LinearRampUp(n_splits=2, step_to_switch_on_reg=CAL_STEPS // 4,
                         step_to_switch_off_reg=CAL_STEPS // 2,
                         start_weight=0.0, end_weight=0.0)
    solver.fit(loader=loader, valid_loader=None, log_frequency=None,
               scheduler_regularizer=sched, scheduler_loss=None)

    atk = dict(eps=0.0, alpha=0.0, steps=ADV_STEPS, lo=lo, hi=hi,
               clamp=CLAMP_TO_DATA, objective=ATTACK_OBJ, norm="linf",
               eps_rel=0.0)
    solver = make_adversarial_solver(solver, BASE_CFG["acorn"], atk)
    solver.model.eval()

    probe_batches = cache_probe_batches(neural, position, CAL_BATCHES)

    print(f"\n{'norm':>5} {'eps_rel':>8} {'eps':>9} {'clean':>8} {'adv':>8} "
          f"{'gap':>8} {'margin':>7} {'DESTROYED':>10} {'hitwall':>8}")
    print("-" * 80)
    table = []
    for norm in CAL_NORMS:
        for er in CAL_GRID:
            eps = compute_eps(norm, er, lo, hi, ts_norm)
            solver._atk.update(norm=norm, eps=eps, alpha=ALPHA_RULE * eps,
                               eps_rel=er)
            solver._adv_log = []
            wall = []
            for b_cpu in probe_batches:
                batch = _batch_to(b_cpu, DEVICE)
                adv = solver._make_adv_batch(batch, record=True)
                # how much of the perturbation the [lo,hi] clamp ate. Once this
                # is large, raising eps buys nothing -- that is the linf ceiling.
                with torch.no_grad():
                    d = adv.reference - batch.reference
                    if norm == "linf":
                        wall.append(float((d.abs() < 0.999 * eps).float().mean()))
                    else:
                        wall.append(float((_l2_norm_per_timestep(d)
                                           < 0.999 * eps).float().mean()))
                del batch, adv
            cl = np.array([x[0] for x in solver._adv_log])
            ad = np.array([x[1] for x in solver._adv_log])
            mg = np.array([x[2] for x in solver._adv_log])
            df = float(np.mean((ad - cl) / mg))
            hw = float(np.mean(wall))
            table.append(dict(norm=norm, eps_rel=er, eps=eps,
                              clean=float(cl.mean()), adv=float(ad.mean()),
                              gap=float(np.mean(ad - cl)),
                              margin=float(mg.mean()), destroy_frac=df,
                              clipped_frac=hw))
            print(f"{norm:>5} {er:>8.3f} {eps:>9.4f} {cl.mean():>8.4f} "
                  f"{ad.mean():>8.4f} {np.mean(ad-cl):>+8.4f} {mg.mean():>7.3f} "
                  f"{100*df:>9.1f}% {100*hw:>7.1f}%")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    import pandas as pd
    pd.DataFrame(table).to_csv(os.path.join(OUT_DIR, "calibration.csv"),
                               index=False)
    print("\nPick eps_rel values whose DESTROYED lands in 20-50%. Below ~10% the\n"
          "attack is cosmetic; above ~60% you train on noise. If DESTROYED stops\n"
          "rising while `hitwall` climbs, you have hit the clamp ceiling and a\n"
          "larger eps is pointless. Then set MODE='sweep' and SWEEP_EPS.")


# ============================================================== MAIN
def main():
    _, neural, position = get_data()
    lo, hi = CLAMP_RANGE if CLAMP_RANGE is not None else (float(neural.min()),
                                                          float(neural.max()))
    ts_norm = float(neural.norm(p=2, dim=1).mean())      # mean per-timestep L2
    print(f"neural {tuple(neural.shape)}  position {tuple(position.shape)}  "
          f"std={float(neural.std()):.4f}  range=[{float(neural.min())},"
          f"{float(neural.max())}]  mean||x_t||2={ts_norm:.3f}")

    if MODE == "calibrate":
        calibrate(neural, position, lo, hi, ts_norm)
        return

    tuning = empirical_tuning(neural, position)
    for k, v in tuning.items():
        print(f"  empirical tuning eta2[{k}]: mean={v.mean():.4f} "
              f"p90={np.quantile(v,0.9):.4f}")
    np.savez_compressed(os.path.join(OUT_DIR, "empirical_tuning.npz"), **tuning)

    gt_nb, _ = build_ground_truth(neural.shape[1], "notebook")
    gt_md, _ = build_ground_truth(neural.shape[1], "model")
    gts = {"gtNB": gt_nb, "gtMODEL": gt_md}

    # --- job list: baselines once, sweep arms per (norm, eps) ----------------
    jobs = []
    for seed in SEEDS:
        for b in BASELINE_ARMS:
            jobs.append((b, b, seed, "linf", 0.0))
        for norm in SWEEP_NORMS:
            for er in SWEEP_EPS:
                for b in SWEEP_ARMS:
                    jobs.append((f"{b}_{norm}_e{er:g}", b, seed, norm, er))

    import pandas as pd
    csv_path = os.path.join(OUT_DIR, "results.csv")
    rows, done = [], set()
    if os.path.exists(csv_path):                          # resume
        prev = pd.read_csv(csv_path)
        rows = prev.to_dict("records")
        done = set(zip(prev["arm"], prev["seed"]))
        print(f"resuming: {len(done)} arm/seed pairs already finished")

    for arm_key, base, seed, norm, er in jobs:
        if (arm_key, seed) in done:
            continue
        eps = compute_eps(norm, er, lo, hi, ts_norm)
        atk = dict(eps=eps, alpha=ALPHA_RULE * eps, steps=ADV_STEPS,
                   lo=lo, hi=hi, clamp=CLAMP_TO_DATA, objective=ATTACK_OBJ,
                   norm=norm, eps_rel=er)
        solver, meta = train_arm(arm_key, base, seed, neural, position, atk)
        row = evaluate(arm_key, seed, solver, neural, position, gts, tuning, meta)
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in row.items()}, indent=1))
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        del solver
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print("\n================ SUMMARY (mean over seeds) ================")
    key_cols = ["destroy_frac", "sv_sv_ratio", "rank_corr_jf_jfinv",
                "sp_gini", "sp_participation_ratio",
                "tune_sp_pos_jf", "tune_sel_jf", "tune_auroc_jf",
                "auroc_rownorm_jf_gtMODEL", "auroc_global_jf_gtMODEL",
                "R2_behavior", "KNN_behavior", "KNN_residual"]
    print(df.groupby("arm")[[c for c in key_cols if c in df.columns]]
            .mean().to_string())

    # --- the money plot: metric vs eps, one line per norm -------------------
    plots = [("destroy_frac", "fraction of learned margin destroyed"),
             ("sv_sv_ratio", "sv_min/sv_max  (spectrum flatness)"),
             ("rank_corr_jf_jfinv", "rank corr  J_f vs pinv(J_f)"),
             ("tune_sel_jf", "position-vs-hd tuning selectivity"),
             ("KNN_behavior", "behaviour decoding (KNN)")]
    fig, axes = plt.subplots(len(plots), 1, figsize=(7, 3.0 * len(plots)),
                             squeeze=False, sharex=True)
    for ax, (col, title) in zip(axes[:, 0], plots):
        if col not in df.columns:
            continue
        for base in SWEEP_ARMS:
            for norm in SWEEP_NORMS:
                sub = df[df["arm"].str.startswith(f"{base}_{norm}_e")]
                if not len(sub):
                    continue
                g = sub.groupby("eps_rel")[col].agg(["mean", "std"])
                ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0),
                            marker="o", capsize=3, label=f"{base} {norm}")
        for b in BASELINE_ARMS:
            sub = df[df["arm"] == b]
            if len(sub) and col in sub:
                ax.axhline(sub[col].mean(), ls="--", lw=1, alpha=0.6,
                           label=f"{b} (baseline)")
        ax.set_ylabel(col); ax.set_title(title, fontsize=10)
        ax.set_xscale("log"); ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("eps_rel")
    axes[0, 0].legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sweep_curves.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

# # =============================================================================
# #  xCEBRA / RatInABox  —  CLEAN vs ACORN (manual PGD adversarial training)
# #  Built on the OFFICIAL cebra 0.6.0a1 multiobjective API (no fork required).
# #
# #  pip install --pre "cebra[integrations]==0.6.0a1"
# #  pip install --upgrade numpy==1.26 ratinabox
# #
# #  WHAT CHANGED vs the crashing/behaviour-only version
# #  ---------------------------------------------------
# #  The multiobjective Batch is NOT a list of Batch objects. It is ONE Batch with
# #      batch.reference : Tensor (B, C, T)        <- shared by all objectives
# #      batch.negative  : Tensor (B, C, T)        <- shared by all objectives
# #      batch.positive  : LIST of Tensors         <- one per objective  (the bug)
# #  and self.model(x) returns a TUPLE of per-objective heads that are already
# #  sliced and renormalized (see ContrastiveMultiobjectiveSolverxCEBRA._inference).
# #
# #  Consequences, all fixed here:
# #   * `batch.positive.detach()` -> AttributeError ('list' has no attribute detach)
# #   * indexing `batch.positive[0]` "fixes" the crash but silently attacks ONLY
# #     objective 0 (behaviour). The time objective spans all 14 latents and is
# #     where the Jacobian flattening comes from, so that variant is a different
# #     experiment and will look like a null result.
# #   * the hand-written cosine_infonce / embed_objective / derive_blocks stack is
# #     now deleted: the attack calls self.criterion() directly, so the ascent
# #     direction is byte-exact w.r.t. the training objective.
# #   * positives/negatives do not depend on delta -> embedded ONCE, not 10x.
# #     PGD now costs 1 forward+backward per step instead of (1 + n_obj + 1).
# # =============================================================================

# import os, sys, copy, json, time, pickle, itertools, warnings
# import numpy as np
# import torch
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.model_selection import TimeSeriesSplit

# from pathlib import Path

# # --- force the ORIGINAL cebra (the attack is reimplemented here, so the fork
# #     must not leak in through a stale sys.modules entry) -----------------------
# CEBRA_DIR = Path(__file__).resolve().parent / "CEBRA-original"
# for module_name in list(sys.modules):
#     if module_name == "cebra" or module_name.startswith("cebra."):
#         del sys.modules[module_name]
# sys.path.insert(0, str(CEBRA_DIR))

# import cebra
# from cebra.data import DatasetxCEBRA, ContrastiveMultiObjectiveLoader, TensorDataset
# from cebra.solver import MultiObjectiveConfig
# from cebra.solver.schedulers import LinearRampUp

# print("cebra loaded from:", cebra.__file__)


# # ----------------------------------------------------------------------------- CONFIG
# DATA_FILE   = "cynthi_neurons90.p"
# DATA_URL    = ("https://zenodo.org/records/15267195/files/"
#                "cynthi_neurons90_gridbase0.5_gridmodules3_grid_head_direction_place_speed"
#                "_duration2000_noise0.25_bs100_seed231209234.p?download=1")

# OUT_DIR     = "xcebra_ratinabox_acorn_REAL"
# DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# # --- model / training (matches the official demo notebook) --------------------
# NUM_STEPS   = 25000
# BATCH_SIZE  = 2500
# N_LATENTS   = 14
# NUM_UNITS   = 256
# MODEL_ARCH  = "offset10-model"
# LR          = 3e-4
# TAU         = 1.0
# BEHAVIOR_RANGE = (0, 4)
# TIME_RANGE     = (0, N_LATENTS)
# TIME_DELTA     = 1
# TIME_OFFSET    = 10
# RENORMALIZE    = True

# LAMBDA_MAX  = 0.1                 # xCEBRA Jacobian-reg end weight (notebook value)

# # --- attack -------------------------------------------------------------------
# ADV_NORM      = "linf"            # "linf" | "l2"
# EPS_MODE      = "range"           # "range" -> eps = EPS_REL * (hi - lo)
#                                   # "std"   -> eps = EPS_REL * std(neural)
# EPS_REL       = 0.03
# ADV_STEPS     = 10
# ALPHA_RULE    = 0.2               # alpha = ALPHA_RULE * eps
# ATTACK_OBJ    = "infonce"         # "infonce" (ascend the TRAINING loss) | "vat"

# # This dataset's neural array is normalized rates living exactly in [0, 1]
# # (verified at runtime: range=[0.0, 1.0]), so the advisor's hard clamp(0,1) is
# # the correct projection here. Set CLAMP_RANGE=None to clamp to the empirical
# # [min, max] instead.
# CLAMP_TO_DATA = True
# CLAMP_RANGE   = (0.0, 1.0)

# # Where the Jacobian penalty is evaluated on the doubled update:
# #   "clean" -> only on update 1 (matches the synthetic ACORN benchmark)
# #   "adv"   -> on both updates (what cebra's compute_regularizer does by default,
# #              since it regularizes at batch.reference == x_adv)
# JREG_AT       = "clean"

# ATTACK_LOG_EVERY = 250            # every N solver steps, record clean vs adv loss

# SEEDS         = [0,3]
# ARMS_TO_RUN = ["cebra", "cebra_2x", "xcebra", "xcebra_2x",
#                "noise_2x", "acorn", "acorn_xreg"]
# ARMS = {
#     "cebra":      dict(lam=0.0,        adv=None,    double=False),
#     "cebra_2x":   dict(lam=0.0,        adv=None,    double=True ),  # doubled-step control
#     "xcebra":     dict(lam=LAMBDA_MAX, adv=None,    double=False),  # <- the notebook
#     "xcebra_2x":  dict(lam=LAMBDA_MAX, adv=None,    double=True ),
#     "noise_2x":   dict(lam=0.0,        adv="noise", double=True ),  # eps-ball init, 0 ascent
#     "acorn":      dict(lam=0.0,        adv="pgd",   double=True ),
#     "acorn_xreg": dict(lam=LAMBDA_MAX, adv="pgd",   double=True ),
# }

# ATTR_MAX_SAMPLES = None           # e.g. 10000 if the Jacobian eval OOMs
# os.makedirs(OUT_DIR, exist_ok=True)


# # ----------------------------------------------------------------------------- DATA
# def get_data():
#     if not os.path.exists(DATA_FILE):
#         import requests
#         print("downloading dataset ...")
#         r = requests.get(DATA_URL)
#         with open(DATA_FILE, "wb") as f:
#             f.write(r.content)
#     with open(DATA_FILE, "rb") as f:
#         d = pickle.load(f)
#     neural   = torch.FloatTensor(d["spikes"]).float()
#     position = torch.FloatTensor(d["position"]).float()
#     return d, neural, position


# # ----------------------------------------------------------------------------- GROUND TRUTH
# def build_ground_truth(n_neurons, split="notebook"):
#     """
#     split="notebook" -> 3 position+grid latents, 11 speed latents  (as in the demo)
#     split="model"    -> 4 position+grid latents, 10 speed latents  (matches (0,4)/(4,14))

#     NOTE: `cells` contains NO 'speed' cells at all, so every speed row is all-False
#     and contributes nothing but negatives. Both variants are scored so the
#     3-vs-4 latent off-by-one in the notebook cannot silently drive a conclusion.
#     """
#     cells = np.array(list(itertools.chain.from_iterable(
#         [["position"] * 100, ["hd"] * 100, ["position"] * 100, ["grid"] * 60])))
#     if len(cells) != n_neurons:
#         warnings.warn(f"cell-type list has {len(cells)} entries but data has "
#                       f"{n_neurons} neurons -- ground truth disabled.")
#         return None, cells
#     n_beh = 3 if split == "notebook" else BEHAVIOR_RANGE[1]
#     latents = [["position", "grid"]] * n_beh + [["speed"]] * (N_LATENTS - n_beh)
#     gt = np.zeros((len(latents), len(cells)), dtype=bool)
#     for i, lat in enumerate(latents):
#         for j, ct in enumerate(cells):
#             gt[i, j] = ct in lat
#     return gt, cells


# # ----------------------------------------------------------------------------- METRICS
# def _avg_rank(x):
#     order = np.argsort(x, kind="mergesort")
#     ranks = np.empty(len(x), dtype=float)
#     ranks[order] = np.arange(1, len(x) + 1, dtype=float)
#     xs = x[order]
#     i = 0
#     while i < len(xs):                                   # average ties
#         j = i
#         while j + 1 < len(xs) and xs[j + 1] == xs[i]:
#             j += 1
#         if j > i:
#             ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
#         i = j + 1
#     return ranks


# def auroc(scores, labels):
#     s = np.asarray(scores, dtype=float).ravel()
#     y = np.asarray(labels).ravel().astype(bool)
#     n_pos, n_neg = y.sum(), (~y).sum()
#     if n_pos == 0 or n_neg == 0:
#         return float("nan")
#     r = _avg_rank(s)
#     return (r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# def auroc_variants(M, gt):
#     """global / colnorm / rownorm.  colnorm and rownorm remove a marginal that can
#     saturate the pooled AUROC without any real localisation."""
#     M = np.asarray(M, dtype=float)
#     out = {"global": auroc(M, gt)}
#     out["colnorm"] = auroc(M / (M.mean(axis=0, keepdims=True) + 1e-12), gt)
#     out["rownorm"] = auroc(M / (M.mean(axis=1, keepdims=True) + 1e-12), gt)
#     return out


# # ----------------------------------------------------------------------------- ATTACK HELPERS
# def _l2_norm_per_timestep(t):
#     """L2 norm across the CHANNEL/neuron axis only, computed independently per
#     time-step. batch.reference is (B, C, T) for CEBRA conv models, so norming
#     over dim=1 reproduces the advisor's per-timestep epsilon-ball exactly
#     (his code permutes to (B,T,C) and norms over dim=-1; same thing)."""
#     return t.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)


# def _replace_reference(batch, new_reference):
#     """copy.copy keeps `positive` (a LIST) and `negative` and any index fields
#     untouched; only the reference stream is swapped, exactly like the fork."""
#     nb = copy.copy(batch)
#     try:
#         nb.reference = new_reference
#     except Exception:
#         object.__setattr__(nb, "reference", new_reference)
#     return nb


# # ----------------------------------------------------------------------------- SOLVER PATCH
# def make_adversarial_solver(solver, cfg, atk):
#     """Rebind solver.__class__ to a subclass of whatever cebra.solver.init built.
#     MultiCriterion and _inference are never touched -- only `step`, via super()."""
#     Base = type(solver)

#     class AdversarialSolver(Base):

#         # -- the solver's own loss, with pos/neg embeddings held fixed ----------
#         def _adv_loss(self, x_adv, pos_out, neg_out):
#             ref_out = self.model(x_adv)
#             predictions = [cebra.data.Batch(reference=ref_out[i],
#                                             positive=pos_out[i],
#                                             negative=neg_out[i])
#                            for i in range(len(pos_out))]
#             return sum(self.criterion(predictions))

#         def _make_adv_batch(self, batch, record=False):
#             eps, alpha, steps = atk["eps"], atk["alpha"], atk["steps"]
#             norm, obj         = atk["norm"], atk["objective"]
#             clamp, lo, hi     = atk["clamp"], atk["lo"], atk["hi"]
#             mode              = cfg["adv"]

#             ref = batch.reference.detach()

#             if norm == "linf":
#                 delta = torch.empty_like(ref).uniform_(-eps, eps)
#             else:
#                 v = torch.randn_like(ref)
#                 delta = eps * v / _l2_norm_per_timestep(v)

#             def _finish(d):
#                 x = ref + d
#                 if clamp:
#                     x = torch.clamp(x, lo, hi)
#                 return _replace_reference(batch, x.detach())

#             if mode == "noise" or steps == 0:
#                 return _finish(delta)

#             # positives / negatives are constant w.r.t. delta -> embed ONCE.
#             # _inference forwards each positive separately and keeps head i,
#             # so that indexing is reproduced exactly.
#             n_obj = len(batch.positive)
#             with torch.no_grad():
#                 neg_all = self.model(batch.negative)
#                 neg_out = [neg_all[i] for i in range(n_obj)]
#                 pos_out = [self.model(p)[i] for i, p in enumerate(batch.positive)]
#                 if obj == "vat":
#                     ref_out0 = [t.detach() for t in self.model(ref)]
#                 clean_loss = (float(self._adv_loss(ref, pos_out, neg_out))
#                               if (record and obj != "vat") else None)

#             delta.requires_grad_(True)
#             for _ in range(steps):
#                 x = torch.clamp(ref + delta, lo, hi) if clamp else ref + delta
#                 if obj == "vat":
#                     out = self.model(x)
#                     loss = sum(((out[i] - ref_out0[i]) ** 2).sum(1).mean()
#                                for i in range(len(out)))
#                 else:
#                     loss = self._adv_loss(x, pos_out, neg_out)
#                 g, = torch.autograd.grad(loss, delta)
#                 with torch.no_grad():
#                     if norm == "linf":
#                         delta.add_(alpha * g.sign()).clamp_(-eps, eps)
#                     else:
#                         delta.add_(alpha * g / _l2_norm_per_timestep(g))
#                         delta.mul_(torch.clamp(
#                             eps / _l2_norm_per_timestep(delta), max=1.0))

#             adv_batch = _finish(delta.detach())

#             # --- diagnostic: is eps actually doing anything? -------------------
#             if record and clean_loss is not None:
#                 with torch.no_grad():
#                     adv_loss = float(self._adv_loss(
#                         adv_batch.reference, pos_out, neg_out))
#                 self._adv_log.append((clean_loss, adv_loss))
#             return adv_batch

#         def step(self, *args, **kwargs):
#             batch = args[0]
#             self._adv_n = getattr(self, "_adv_n", 0) + 1
#             rec = (self._adv_n % ATTACK_LOG_EVERY == 0)

#             if not cfg["double"]:
#                 if cfg["adv"] is None:
#                     return super().step(*args, **kwargs)
#                 return super().step(self._make_adv_batch(batch, rec),
#                                     *args[1:], **kwargs)

#             # --- update 1: unconditional clean step (fork-faithful) ------------
#             super().step(*args, **kwargs)

#             # --- update 2 ------------------------------------------------------
#             if cfg["adv"] is None:
#                 return super().step(*args, **kwargs)        # cebra_2x / xcebra_2x

#             kw2 = dict(kwargs)
#             if JREG_AT == "clean":
#                 kw2["weights_regularizer"] = None           # penalty only at x_clean
#             # attack is built with the ALREADY-UPDATED parameters
#             return super().step(self._make_adv_batch(batch, rec),
#                                 *args[1:], **kw2)

#     solver.__class__ = AdversarialSolver
#     solver._adv_log = []
#     solver._adv_n = 0
#     return solver


# # ----------------------------------------------------------------------------- TRAIN ONE ARM
# def train_arm(arm_name, seed, neural, position):
#     cfg = ARMS[arm_name]
#     torch.manual_seed(seed); np.random.seed(seed)

#     data = DatasetxCEBRA(neural, position=position)
#     loader = ContrastiveMultiObjectiveLoader(dataset=data,
#                                              num_steps=NUM_STEPS,
#                                              batch_size=BATCH_SIZE).to(DEVICE)

#     config = MultiObjectiveConfig(loader)
#     config.set_slice(*BEHAVIOR_RANGE)
#     config.set_loss("FixedCosineInfoNCE", temperature=TAU)
#     config.set_distribution("time_delta", time_delta=TIME_DELTA, label_name="position")
#     config.push()

#     config.set_slice(*TIME_RANGE)
#     config.set_loss("FixedCosineInfoNCE", temperature=TAU)
#     config.set_distribution("time", time_offset=TIME_OFFSET)
#     config.push()
#     config.finalize()

#     criterion      = config.criterion
#     feature_ranges = config.feature_ranges

#     model = cebra.models.init(name=MODEL_ARCH,
#                               num_neurons=data.neural.shape[1],
#                               num_units=NUM_UNITS,
#                               num_output=N_LATENTS).to(DEVICE)
#     data.configure_for(model)

#     opt = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()),
#                            lr=LR, weight_decay=0)

#     solver = cebra.solver.init(name="multiobjective-solver",
#                                model=model,
#                                feature_ranges=feature_ranges,
#                                regularizer=cebra.models.jacobian_regularizer.JacobianReg(),
#                                renormalize=RENORMALIZE,
#                                use_sam=False,
#                                criterion=criterion,
#                                optimizer=opt,
#                                tqdm_on=True).to(DEVICE)

#     # lam = 0 arms: same scheduler object, both weights zero -> penalty never fires.
#     sched = LinearRampUp(n_splits=2,
#                          step_to_switch_on_reg=NUM_STEPS // 4,
#                          step_to_switch_off_reg=NUM_STEPS // 2,
#                          start_weight=0.0,
#                          end_weight=cfg["lam"])

#     if CLAMP_RANGE is not None:
#         lo, hi = CLAMP_RANGE
#     else:
#         lo, hi = float(neural.min()), float(neural.max())

#     eps   = (EPS_REL * (hi - lo)) if EPS_MODE == "range" else (EPS_REL * float(neural.std()))
#     alpha = ALPHA_RULE * eps

#     atk = dict(eps=eps, alpha=alpha, steps=ADV_STEPS,
#                lo=lo, hi=hi, clamp=CLAMP_TO_DATA,
#                objective=ATTACK_OBJ, norm=ADV_NORM)

#     if cfg["adv"] is not None or cfg["double"]:
#         solver = make_adversarial_solver(solver, cfg, atk)

#     print(f"\n=== arm={arm_name} seed={seed} lam={cfg['lam']} adv={cfg['adv']} "
#           f"double={cfg['double']} eps={eps:.4f} alpha={alpha:.4f} "
#           f"clamp={'[%.3f,%.3f]' % (lo, hi) if CLAMP_TO_DATA else 'off'} "
#           f"jreg_at={JREG_AT} dev={DEVICE}")

#     t0 = time.time()
#     solver.fit(loader=loader, valid_loader=None, log_frequency=None,
#                scheduler_regularizer=sched, scheduler_loss=None)
#     wall = time.time() - t0

#     meta = dict(wall_s=wall, eps=eps, alpha=alpha)
#     log = getattr(solver, "_adv_log", [])
#     if log:
#         cl = np.array([a for a, _ in log]); ad = np.array([b for _, b in log])
#         meta["adv_loss_clean"] = float(cl.mean())
#         meta["adv_loss_adv"]   = float(ad.mean())
#         meta["adv_loss_gap"]   = float((ad - cl).mean())
#         print(f"    attack reach: clean={cl.mean():.4f}  adv={ad.mean():.4f}  "
#               f"gap={np.mean(ad - cl):+.4f} nats  (n={len(log)} probes)")
#     return solver, meta


# # ----------------------------------------------------------------------------- EVAL
# def _set_split(model, flag):
#     if hasattr(model, "set_split_outputs"):
#         model.set_split_outputs(flag)
#     else:
#         model.split_outputs = flag


# def compute_embedding(solver, neural):
#     d = TensorDataset(neural, continuous=torch.zeros(len(neural), 1))
#     d.configure_for(solver.model)
#     x = d[torch.arange(len(d))]
#     _set_split(solver.model, False)
#     with torch.no_grad():
#         return solver.model(x.to(DEVICE)).detach().cpu()


# def decoding_scores(embedding, position):
#     X_beh  = embedding[:, slice(*BEHAVIOR_RANGE)].numpy()        # block [0:4]
#     X_time = embedding[:, slice(*TIME_RANGE)].numpy()            # FULL (0,14) == the time objective
#     X_res  = embedding[:, BEHAVIOR_RANGE[1]:N_LATENTS].numpy()   # residual block [4:14]
#     y = position[:len(embedding)].numpy()

#     out = {}
#     tscv = TimeSeriesSplit(n_splits=5)
#     for tag, X in (("behavior", X_beh), ("time", X_time), ("residual", X_res)):
#         out[f"R2_{tag}"] = LinearRegression().fit(X, y).score(X, y)
#         out[f"KNN_{tag}"] = float(np.mean(
#             [KNeighborsRegressor().fit(X[tr], y[tr]).score(X[va], y[va])
#              for tr, va in tscv.split(X)]))
#     return out

# def attribution(solver, neural):
#     model = solver.model.to(DEVICE)
#     _set_split(model, False)
#     x = neural if ATTR_MAX_SAMPLES is None else neural[:ATTR_MAX_SAMPLES]
#     x = x.clone().to(DEVICE).requires_grad_(True)
#     method = cebra.attribution.init(name="jacobian-based",
#                                     model=model,
#                                     input_data=x,
#                                     output_dimension=model.num_output)
#     return method, method.compute_attribution_map()


# def spectral_stats(jf_mean):
#     s = np.linalg.svd(np.asarray(jf_mean, dtype=float), compute_uv=False)
#     return dict(sv_max=float(s[0]), sv_min=float(s[-1]),
#                 sv_ratio=float(s[-1] / (s[0] + 1e-30)),
#                 fro=float(np.sqrt((s ** 2).sum())))


# def sparsity_stats(jf_mean):
#     """scale-invariant -- the heatmap colourbar is NOT comparable across arms."""
#     s = np.clip(np.asarray(jf_mean, dtype=float).mean(axis=0), 0, None)
#     tot = s.sum() + 1e-30
#     p = s / tot
#     pr = (s.sum() ** 2) / ((s ** 2).sum() + 1e-30)          # effective #neurons
#     ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
#     srt = np.sort(s); cum = np.cumsum(srt) / tot
#     gini = float(1 - 2 * np.trapz(cum, dx=1.0 / len(srt)))
#     return dict(participation_ratio=float(pr), entropy=ent, gini=gini,
#                 n_neurons=int(len(s)))


# def evaluate(arm, seed, solver, neural, position, gts, meta):
#     row = dict(arm=arm, seed=seed, **meta)
#     emb = compute_embedding(solver, neural)
#     row.update(decoding_scores(emb, position))

#     torch.save(solver.model.state_dict(),
#                os.path.join(OUT_DIR, f"model_{arm}_s{seed}.pt"))

#     method, result = attribution(solver, neural)
#     maps = {}
#     for key, name in (("jf", "jf"),
#                       ("jf-inv-svd", "jfinv"),
#                       ("jf-convabs-inv-svd", "jfconvabsinv")):
#         if key in result:
#             maps[name] = np.abs(result[key]).mean(0)
#     np.savez_compressed(os.path.join(OUT_DIR, f"maps_{arm}_s{seed}.npz"), **maps)

#     if "jf" in maps:
#         row.update({f"sv_{k}": v for k, v in spectral_stats(maps["jf"]).items()})
#         row.update({f"sp_{k}": v for k, v in sparsity_stats(maps["jf"]).items()})
#         # free win: with a flat singular spectrum, J_f and pinv(J_f) rank the
#         # same neurons -> you can skip the pinv. Expect HIGH for acorn, LOW for cebra.
#         if "jfinv" in maps:
#             a = _avg_rank(maps["jf"].ravel()); b = _avg_rank(maps["jfinv"].ravel())
#             row["rank_corr_jf_jfinv"] = float(np.corrcoef(a, b)[0, 1])

#     for gt_name, gt in gts.items():
#         if gt is None:
#             continue
#         for mname, M in maps.items():
#             if M.shape != gt.shape:
#                 continue
#             try:
#                 row[f"auc_{mname}_{gt_name}"] = float(
#                     method.compute_attribution_score(M, gt))
#             except Exception:
#                 row[f"auc_{mname}_{gt_name}"] = float("nan")
#             for vname, v in auroc_variants(M, gt).items():
#                 row[f"auroc_{vname}_{mname}_{gt_name}"] = v

#     for mname, M in maps.items():
#         plt.figure(figsize=(9, 3.2))
#         plt.matshow(M, aspect="auto", fignum=0)
#         plt.colorbar(); plt.title(f"{arm} (seed {seed}) — {mname}")
#         plt.tight_layout()
#         plt.savefig(os.path.join(OUT_DIR, f"map_{arm}_s{seed}_{mname}.png"), dpi=140)
#         plt.close()
#     return row


# # ----------------------------------------------------------------------------- MAIN
# def main():
#     _, neural, position = get_data()
#     print(f"neural {tuple(neural.shape)}  position {tuple(position.shape)}  "
#           f"std={float(neural.std()):.4f}  "
#           f"range=[{float(neural.min())},{float(neural.max())}]")

#     gt_nb, _ = build_ground_truth(neural.shape[1], "notebook")
#     gt_md, _ = build_ground_truth(neural.shape[1], "model")
#     gts = {"gtNB": gt_nb, "gtMODEL": gt_md}

#     import pandas as pd
#     rows = []
#     for seed in SEEDS:
#         for arm in ARMS_TO_RUN:
#             solver, meta = train_arm(arm, seed, neural, position)
#             row = evaluate(arm, seed, solver, neural, position, gts, meta)
#             rows.append(row)
#             print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
#                               for k, v in row.items()}, indent=1))
#             pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)
#             del solver
#             torch.cuda.empty_cache()

#     df = pd.DataFrame(rows)
#     print("\n================ SUMMARY ================")
#     cols = [c for c in df.columns
#             if c.startswith(("auc_", "auroc_global_", "auroc_colnorm_", "sv_sv_ratio",
#                              "sp_participation", "sp_gini", "rank_corr_",
#                              "adv_loss_gap", "R2_", "KNN_"))]
#     print(df.groupby("arm")[cols].mean().T.to_string())

#     # shared-colourbar comparison of the jf maps (the honest version of the plot)
#     done = [a for a in ARMS_TO_RUN
#             if os.path.exists(os.path.join(OUT_DIR, f"maps_{a}_s{SEEDS[0]}.npz"))]
#     mats = {a: np.load(os.path.join(OUT_DIR, f"maps_{a}_s{SEEDS[0]}.npz"))["jf"]
#             for a in done}
#     fig, axes = plt.subplots(len(done), 1, figsize=(10, 2.6 * len(done)), squeeze=False)
#     vmax = max(m.max() for m in mats.values())
#     for ax, a in zip(axes[:, 0], done):
#         im = ax.imshow(mats[a], aspect="auto", vmin=0, vmax=vmax)
#         ax.set_title(f"{a}  (shared scale)")
#     fig.colorbar(im, ax=axes[:, 0].tolist())
#     plt.savefig(os.path.join(OUT_DIR, "jf_shared_scale.png"), dpi=150)
#     plt.close()


# if __name__ == "__main__":
#     main()






# # # =============================================================================
# # #  xCEBRA / RatInABox  —  CLEAN vs ACORN (manual PGD adversarial training)
# # #  Built on the OFFICIAL cebra 0.6.0a1 multiobjective API (no fork required).
# # #
# # #  pip install --pre "cebra[integrations]==0.6.0a1"
# # #  pip install --upgrade numpy==1.26 ratinabox
# # #
# # #  Fixes applied vs. the previous version (see inline "FIX" comments):
# # #   1) CRITICAL: build_adversarial's infonce branch referenced `self._inference`/
# # #      `self.criterion`, but build_adversarial is a free function (no `self`
# # #      exists) -> NameError on every PGD/infonce attack. Reverted to the
# # #      already-correct, already-written cosine_infonce(...) path.
# # #   2) L2 attack ball now matches the professor's Solver: the L2 norm is
# # #      computed PER TIME-STEP (across channels only), not globally over the
# # #      whole flattened (channel x time) sample.
# # #   3) CLAMP_TO_DATA now applies consistently to both "pgd" and "noise"
# # #      attack modes (previously "noise" always clamped, "pgd" only clamped
# # #      conditionally). Added CLAMP_RANGE so you can reproduce the professor's
# # #      hard-coded clamp(0,1) exactly if you want bit-for-bit parity, or leave
# # #      it at None to clamp to this dataset's own [min,max] instead.
# # # =============================================================================

# # import os, sys, copy, json, time, pickle, itertools, warnings
# # import numpy as np
# # import torch
# # import torch.nn.functional as F
# # import matplotlib
# # matplotlib.use("Agg")
# # import matplotlib.pyplot as plt

# # from sklearn.linear_model import LinearRegression
# # from sklearn.neighbors import KNeighborsRegressor
# # from sklearn.model_selection import TimeSeriesSplit

# # import sys
# # from pathlib import Path

# # CEBRA_DIR = Path(__file__).resolve().parent / "CEBRA-original"

# # for module_name in list(sys.modules):
# #     if module_name == "cebra" or module_name.startswith("cebra."):
# #         del sys.modules[module_name]

# # sys.path.insert(0, str(CEBRA_DIR))

# # import cebra
# # from cebra.data import DatasetxCEBRA, ContrastiveMultiObjectiveLoader, TensorDataset
# # from cebra.solver import MultiObjectiveConfig
# # from cebra.solver.schedulers import LinearRampUp

# # # ----------------------------------------------------------------------------- CONFIG
# # DATA_FILE   = "cynthi_neurons90.p"
# # DATA_URL    = ("https://zenodo.org/records/15267195/files/"
# #                "cynthi_neurons90_gridbase0.5_gridmodules3_grid_head_direction_place_speed"
# #                "_duration2000_noise0.25_bs100_seed231209234.p?download=1")

# # OUT_DIR     = "xcebra_ratinabox_acorn"
# # DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# # # --- verified against the official notebook (cebra.ai/docs/demo_notebooks/
# # #     Demo_xCEBRA_RatInABox.html) -- every value below matches it exactly ---
# # NUM_STEPS   = 25000
# # BATCH_SIZE  = 2500
# # N_LATENTS   = 14
# # NUM_UNITS   = 256
# # MODEL_ARCH  = "offset10-model"
# # LR          = 3e-4
# # TAU         = 1.0                       # FixedCosineInfoNCE temperature (both objectives)
# # BEHAVIOR_RANGE = (0, 4)
# # TIME_RANGE     = (0, N_LATENTS)
# # TIME_DELTA     = 1
# # TIME_OFFSET    = 10
# # RENORMALIZE    = True

# # LAMBDA_MAX  = 0.1                       # xCEBRA Jacobian-reg end weight (notebook value)

# # # --- attack ---
# # ADV_NORM      = "linf"                  # "linf" | "l2"
# # EPS_REL       = 0.05                    # eps = EPS_REL * std(neural).  -> printed at runtime
# # ADV_STEPS     = 10
# # ALPHA_RULE    = 0.2                     # alpha = ALPHA_RULE * eps  (saturates the ball)
# # ATTACK_OBJ    = "infonce"               # "infonce" (ascend the training loss) | "vat"

# # # FIX 3: CLAMP_TO_DATA now gates BOTH the "pgd" and "noise" attack modes
# # # consistently. CLAMP_RANGE=None -> clamp to [neural.min(), neural.max()];
# # # set CLAMP_RANGE=(0.0, 1.0) to exactly reproduce the professor's
# # # MultiobjectiveSolver, which unconditionally hard-clamps x_adv to [0,1]
# # # (only appropriate if your data actually lives in that range -- spike
# # # counts / RatInABox rates generally don't, so the default here is off).
# # CLAMP_TO_DATA = False
# # CLAMP_RANGE   = None                    # e.g. (0.0, 1.0) for exact professor parity

# # SEEDS         = [0]                     # add 1,2,... once the runtime is acceptable
# # ARMS_TO_RUN   = ["cebra", "cebra_2x", "xcebra","acorn"] #"cebra", "cebra_2x", "xcebra",
# # # full menu: cebra | cebra_2x | xcebra | xcebra_2x | noise_2x | acorn | acorn_xreg

# # ARMS = {
# #     "cebra":      dict(lam=0.0,        adv=None,    double=False),
# #     "cebra_2x":   dict(lam=0.0,        adv=None,    double=True ),  # the doubled-step control
# #     "xcebra":     dict(lam=LAMBDA_MAX, adv=None,    double=False),  # <- the notebook
# #     "xcebra_2x":  dict(lam=LAMBDA_MAX, adv=None,    double=True ),
# #     "noise_2x":   dict(lam=0.0,        adv="noise", double=True ),  # eps-ball init, 0 ascent steps
# #     "acorn":      dict(lam=0.0,        adv="pgd",   double=True ),
# #     "acorn_xreg": dict(lam=LAMBDA_MAX, adv="pgd",   double=True ),
# # }

# # ATTR_MAX_SAMPLES = None                 # e.g. 20000 if the Jacobian eval runs out of memory
# # os.makedirs(OUT_DIR, exist_ok=True)


# # # ----------------------------------------------------------------------------- DATA
# # def get_data():
# #     if not os.path.exists(DATA_FILE):
# #         import requests
# #         print("downloading dataset ...")
# #         r = requests.get(DATA_URL)
# #         with open(DATA_FILE, "wb") as f:
# #             f.write(r.content)
# #     with open(DATA_FILE, "rb") as f:
# #         d = pickle.load(f)
# #     neural   = torch.FloatTensor(d["spikes"]).float()
# #     position = torch.FloatTensor(d["position"]).float()
# #     return d, neural, position


# # # ----------------------------------------------------------------------------- GROUND TRUTH
# # def build_ground_truth(n_neurons, split="notebook"):
# #     """
# #     `split="notebook"`  -> 3 position+grid latents, 11 speed latents  (as in the demo)
# #     `split="model"`     -> 4 position+grid latents, 10 speed latents  (matches (0,4)/(4,14))
# #     NOTE: `cells` contains no 'speed' cells at all, so every speed row is all-False.
# #     """
# #     cells = np.array(list(itertools.chain.from_iterable(
# #         [["position"] * 100, ["hd"] * 100, ["position"] * 100, ["grid"] * 60])))
# #     if len(cells) != n_neurons:
# #         warnings.warn(f"cell-type list has {len(cells)} entries but data has {n_neurons} "
# #                       f"neurons -- ground truth disabled.")
# #         return None, cells
# #     n_beh = 3 if split == "notebook" else BEHAVIOR_RANGE[1]
# #     latents = [["position", "grid"]] * n_beh + [["speed"]] * (N_LATENTS - n_beh)
# #     gt = np.zeros((len(latents), len(cells)), dtype=bool)
# #     for i, lat in enumerate(latents):
# #         for j, ct in enumerate(cells):
# #             gt[i, j] = ct in lat
# #     return gt, cells


# # # ----------------------------------------------------------------------------- METRICS
# # def _avg_rank(x):
# #     order = np.argsort(x, kind="mergesort")
# #     ranks = np.empty(len(x), dtype=float)
# #     ranks[order] = np.arange(1, len(x) + 1, dtype=float)
# #     xs = x[order]
# #     i = 0
# #     while i < len(xs):                                   # average ties
# #         j = i
# #         while j + 1 < len(xs) and xs[j + 1] == xs[i]:
# #             j += 1
# #         if j > i:
# #             ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
# #         i = j + 1
# #     return ranks

# # def auroc(scores, labels):
# #     s = np.asarray(scores, dtype=float).ravel()
# #     y = np.asarray(labels).ravel().astype(bool)
# #     n_pos, n_neg = y.sum(), (~y).sum()
# #     if n_pos == 0 or n_neg == 0:
# #         return float("nan")
# #     r = _avg_rank(s)
# #     return (r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

# # def auroc_variants(M, gt):
# #     """global / colnorm / rownorm -- colnorm and rownorm remove a marginal that can
# #     saturate the pooled AUROC without any real localisation."""
# #     M = np.asarray(M, dtype=float)
# #     out = {"global": auroc(M, gt)}
# #     col = M / (M.mean(axis=0, keepdims=True) + 1e-12)
# #     row = M / (M.mean(axis=1, keepdims=True) + 1e-12)
# #     out["colnorm"] = auroc(col, gt)
# #     out["rownorm"] = auroc(row, gt)
# #     return out


# # # ----------------------------------------------------------------------------- ATTACK
# # def derive_blocks(ranges, renormalize):
# #     """[(0,4),(0,14)] --> disjoint blocks [0:4],[4:14] plus, per objective, which blocks
# #     it spans. This reproduces the solver's `renormalize` bookkeeping exactly."""
# #     bounds = sorted({0} | {a for a, _ in ranges} | {b for _, b in ranges})
# #     blocks = [slice(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
# #     per_obj = []
# #     for a, b in ranges:
# #         if renormalize:
# #             per_obj.append([k for k, bl in enumerate(blocks) if bl.start >= a and bl.stop <= b])
# #         else:
# #             per_obj.append(None)            # use the raw slice
# #     return blocks, per_obj

# # def embed_objective(z, obj_idx, blocks, per_obj, ranges):
# #     if per_obj[obj_idx] is None:
# #         a, b = ranges[obj_idx]
# #         return z[:, a:b]
# #     parts = [F.normalize(z[:, blocks[k]], dim=1) for k in per_obj[obj_idx]]
# #     return torch.cat(parts, dim=1)

# # def cosine_infonce(ref, pos, neg, tau=TAU):
# #     ref = F.normalize(ref, dim=1)
# #     pos = F.normalize(pos, dim=1)
# #     neg = F.normalize(neg, dim=1)
# #     pos_sim = (ref * pos).sum(1) / tau                       # (B,)
# #     neg_sim = ref @ neg.t() / tau                            # (B,B)
# #     return (-pos_sim).mean() + torch.logsumexp(neg_sim, dim=1).mean()

# # def as_batch_list(batch):
# #     if hasattr(batch, "reference"):
# #         return [batch], True
# #     return list(batch), False

# # def rebuild_batch(batch, new_refs, was_single):
# #     # FIX (cosmetic): the previous `zip(*as_batch_list(batch)[0:1] and (...))`
# #     # always simplified to `zip(batches, new_refs)` anyway -- written out
# #     # directly here, same behavior, just readable.
# #     batches, _ = as_batch_list(batch)
# #     out = []
# #     for b, r in zip(batches, new_refs):
# #         nb = copy.copy(b)
# #         try:
# #             nb.reference = r
# #         except Exception:
# #             object.__setattr__(nb, "reference", r)
# #         out.append(nb)
# #     return out[0] if was_single else (type(batch)(out) if isinstance(batch, tuple) else out)

# # def _forward_full(model, x):
# #     prev = getattr(model, "split_outputs", None)
# #     if prev is not None:
# #         model.split_outputs = False
# #     z = model(x)
# #     if isinstance(z, (list, tuple)):
# #         z = torch.cat(list(z), dim=1)
# #     if prev is not None:
# #         model.split_outputs = prev
# #     return z

# # def _l2_norm_per_timestep(t, keepdim=True):
# #     """FIX 2: L2 norm across the CHANNEL axis (dim=1) only, computed
# #     independently per time-step. batch.reference has shape (B, C, T) here
# #     (CEBRA conv models take channel-first input), so norming over dim=1 is
# #     exactly what the professor's code does when it permutes to (B,T,C) and
# #     norms over dim=-1 -- same per-timestep epsilon-ball, just without the
# #     extra permute."""
# #     return t.norm(p=2, dim=1, keepdim=keepdim).clamp(min=1e-12)

# # def build_adversarial(model, batch, eps, alpha, steps, lo, hi,
# #                       blocks, per_obj, ranges, mode="pgd", objective=ATTACK_OBJ,
# #                       norm=ADV_NORM, clamp_to_data=False):
# #     """PGD on the reference stream only -- mirrors the fork, where the adversarial
# #     example replaces x_ref and (x_pos, x_neg) stay clean."""
# #     batches, was_single = as_batch_list(batch)
# #     refs = [b.reference.detach() for b in batches]

# #     shared = all(torch.equal(refs[0], r) for r in refs[1:]) if len(refs) > 1 else True
# #     n_delta = 1 if shared else len(refs)

# #     def _init():
# #         if norm == "linf":
# #             return [torch.empty_like(refs[i if not shared else 0]).uniform_(-eps, eps)
# #                     for i in range(n_delta)]
# #         d = []
# #         for i in range(n_delta):
# #             v = torch.randn_like(refs[i if not shared else 0])
# #             v = v / _l2_norm_per_timestep(v)                       # FIX 2
# #             d.append(v * eps)
# #         return d

# #     deltas = [d.requires_grad_(True) for d in _init()]

# #     # FIX 3: "noise" mode now respects clamp_to_data the same way "pgd" does,
# #     # instead of always clamping regardless of the flag.
# #     if mode == "noise" or steps == 0:
# #         with torch.no_grad():
# #             new_refs = [
# #                 (torch.clamp(refs[i] + deltas[0 if shared else i], lo, hi)
# #                  if clamp_to_data else (refs[i] + deltas[0 if shared else i])).detach()
# #                 for i in range(len(refs))
# #             ]
# #         return rebuild_batch(batch, new_refs, was_single)

# #     # positives / negatives do not depend on delta -> embed them once
# #     with torch.no_grad():
# #         # z_pos = [_forward_full(model, b.positive.detach()) for b in batches]
# #         # z_neg = [_forward_full(model, b.negative.detach()) for b in batches]
# #         z_pos = [_forward_full(model, b.positive[0].detach() if isinstance(b.positive, (list, tuple)) else b.positive.detach()) for b in batches]
# #         z_neg = [_forward_full(model, torch.cat([v.detach() for v in b.negative], dim=0) if isinstance(b.negative, (list, tuple)) else b.negative.detach()) for b in batches]
# #         if objective == "vat":
# #             z_ref0 = [_forward_full(model, r) for r in refs]

# #     for _ in range(steps):
# #         loss = 0.0
# #         for i, b in enumerate(batches):
# #             d = deltas[0 if shared else i]
# #             x = torch.clamp(refs[i] + d, lo, hi) if clamp_to_data else refs[i] + d
# #             z = _forward_full(model, x)
# #             if objective == "vat":
# #                 loss = loss + ((z - z_ref0[i]) ** 2).sum(1).mean()
# #             else:
# #                 # FIX 1 (CRITICAL): this used to call self._inference(...) /
# #                 # self.criterion(...) -- but build_adversarial is a free
# #                 # function, `self` doesn't exist here, so this raised
# #                 # NameError on every call. cosine_infonce + embed_objective
# #                 # is the correct free-function equivalent (same math as the
# #                 # solver's own criterion, applied to this objective's slice).
# #                 loss = loss + cosine_infonce(
# #                     embed_objective(z,        i, blocks, per_obj, ranges),
# #                     embed_objective(z_pos[i], i, blocks, per_obj, ranges),
# #                     embed_objective(z_neg[i], i, blocks, per_obj, ranges))
# #         grads = torch.autograd.grad(loss, deltas)
# #         with torch.no_grad():
# #             for d, g in zip(deltas, grads):
# #                 if norm == "linf":
# #                     d.add_(alpha * g.sign())
# #                     d.clamp_(-eps, eps)
# #                 else:
# #                     gn = _l2_norm_per_timestep(g)                  # FIX 2
# #                     d.add_(alpha * g / gn)
# #                     dn = _l2_norm_per_timestep(d)                  # FIX 2
# #                     d.mul_(torch.clamp(eps / dn, max=1.0))

# #     with torch.no_grad():
# #         new_refs = [
# #             (torch.clamp(refs[i] + deltas[0 if shared else i], lo, hi)
# #              if clamp_to_data else (refs[i] + deltas[0 if shared else i])).detach()
# #             for i in range(len(refs))
# #         ]
# #     return rebuild_batch(batch, new_refs, was_single)


# # # ----------------------------------------------------------------------------- SOLVER PATCH
# # def make_adversarial_solver(solver, cfg, attack_kwargs):
# #     """Rebind solver.__class__ to a subclass of whatever cebra.solver.init produced.
# #     We never touch MultiCriterion or _inference -- only `step`, through super()."""
# #     Base = type(solver)

# #     class AdversarialSolver(Base):
# #         def step(self, *args, **kwargs):
# #             c = self._adv_cfg
# #             batch = args[0]

# #             def _adv(b):
# #                 return build_adversarial(self.model, b, mode=c["adv"], **self._attack_kwargs)

# #             if not c["double"]:
# #                 if c["adv"] is None:
# #                     return super().step(*args, **kwargs)
# #                 return super().step(_adv(batch), *args[1:], **kwargs)

# #             # --- update 1: unconditional clean step (fork-faithful) ---------------
# #             stats = super().step(*args, **kwargs)
# #             # --- update 2: built with the ALREADY-UPDATED parameters --------------
# #             if c["adv"] is None:
# #                 return super().step(*args, **kwargs)          # cebra_2x / xcebra_2x
# #             return super().step(_adv(batch), *args[1:], **kwargs)

# #     solver.__class__ = AdversarialSolver
# #     solver._adv_cfg = cfg
# #     solver._attack_kwargs = attack_kwargs
# #     return solver


# # # ----------------------------------------------------------------------------- TRAIN ONE ARM
# # def train_arm(arm_name, seed, neural, position):
# #     cfg = ARMS[arm_name]
# #     torch.manual_seed(seed); np.random.seed(seed)

# #     data = DatasetxCEBRA(neural, position=position)
# #     loader = ContrastiveMultiObjectiveLoader(dataset=data,
# #                                              num_steps=NUM_STEPS,
# #                                              batch_size=BATCH_SIZE).to(DEVICE)

# #     config = MultiObjectiveConfig(loader)
# #     config.set_slice(*BEHAVIOR_RANGE)
# #     config.set_loss("FixedCosineInfoNCE", temperature=TAU)
# #     config.set_distribution("time_delta", time_delta=TIME_DELTA, label_name="position")
# #     config.push()

# #     config.set_slice(*TIME_RANGE)
# #     config.set_loss("FixedCosineInfoNCE", temperature=TAU)
# #     config.set_distribution("time", time_offset=TIME_OFFSET)
# #     config.push()
# #     config.finalize()

# #     criterion      = config.criterion
# #     feature_ranges = config.feature_ranges

# #     model = cebra.models.init(name=MODEL_ARCH,
# #                               num_neurons=data.neural.shape[1],
# #                               num_units=NUM_UNITS,
# #                               num_output=N_LATENTS).to(DEVICE)
# #     data.configure_for(model)

# #     opt = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()),
# #                            lr=LR, weight_decay=0)

# #     solver = cebra.solver.init(name="multiobjective-solver",
# #                                model=model,
# #                                feature_ranges=feature_ranges,
# #                                regularizer=cebra.models.jacobian_regularizer.JacobianReg(),
# #                                renormalize=RENORMALIZE,
# #                                use_sam=False,
# #                                criterion=criterion,
# #                                optimizer=opt,
# #                                tqdm_on=True).to(DEVICE)

# #     # lambda = 0 arms: same scheduler object, both weights zero -> penalty never fires.
# #     sched = LinearRampUp(n_splits=2,
# #                          step_to_switch_on_reg=NUM_STEPS // 4,
# #                          step_to_switch_off_reg=NUM_STEPS // 2,
# #                          start_weight=0.0,
# #                          end_weight=cfg["lam"])

# #     ranges = [tuple(BEHAVIOR_RANGE), tuple(TIME_RANGE)]
# #     blocks, per_obj = derive_blocks(ranges, RENORMALIZE)

# #     eps   = EPS_REL * float(neural.std())
# #     alpha = ALPHA_RULE * eps

# #     # FIX 3: lo/hi now come from CLAMP_RANGE if you set one (e.g. (0.0, 1.0)
# #     # for exact professor parity), otherwise from this dataset's own range.
# #     if CLAMP_RANGE is not None:
# #         lo, hi = CLAMP_RANGE
# #     else:
# #         lo, hi = float(neural.min()), float(neural.max())

# #     attack_kwargs = dict(eps=eps, alpha=alpha, steps=ADV_STEPS,
# #                          lo=lo, hi=hi,
# #                          blocks=blocks, per_obj=per_obj, ranges=ranges,
# #                          objective=ATTACK_OBJ, norm=ADV_NORM,
# #                          clamp_to_data=CLAMP_TO_DATA)

# #     if cfg["adv"] is not None or cfg["double"]:
# #         solver = make_adversarial_solver(solver, cfg, attack_kwargs)

# #     print(f"\n=== arm={arm_name} seed={seed} lam={cfg['lam']} adv={cfg['adv']} "
# #           f"double={cfg['double']} eps={eps:.4f} alpha={alpha:.4f} "
# #           f"clamp={'[%.3f,%.3f]' % (lo, hi) if CLAMP_TO_DATA else 'off'} dev={DEVICE}")
# #     t0 = time.time()
# #     solver.fit(loader=loader, valid_loader=None, log_frequency=None,
# #                scheduler_regularizer=sched, scheduler_loss=None)
# #     wall = time.time() - t0
# #     return solver, dict(wall_s=wall, eps=eps, alpha=alpha)


# # # ----------------------------------------------------------------------------- EVAL
# # def compute_embedding(solver, neural):
# #     d = TensorDataset(neural, continuous=torch.zeros(len(neural)))
# #     d.configure_for(solver.model)
# #     d = d[torch.arange(len(d))]
# #     solver.model.split_outputs = False
# #     with torch.no_grad():
# #         return solver.model(d.to(DEVICE)).detach().cpu()

# # def decoding_scores(embedding, position):
# #     X_b = embedding[:, slice(*BEHAVIOR_RANGE)].numpy()
# #     X_t = embedding[:, BEHAVIOR_RANGE[1]:N_LATENTS].numpy()
# #     y   = position[:len(embedding)].numpy()
# #     out = {}
# #     out["R2_behavior"] = LinearRegression().fit(X_b, y).score(X_b, y)
# #     out["R2_time"]     = LinearRegression().fit(X_t, y).score(X_t, y)
# #     tscv = TimeSeriesSplit(n_splits=5)
# #     for tag, X in (("behavior", X_b), ("time", X_t)):
# #         sc = []
# #         for tr, va in tscv.split(X):
# #             sc.append(KNeighborsRegressor().fit(X[tr], y[tr]).score(X[va], y[va]))
# #         out[f"KNN_{tag}"] = float(np.mean(sc))
# #     return out

# # def attribution(solver, neural):
# #     model = solver.model.to(DEVICE)
# #     model.split_outputs = False
# #     x = neural if ATTR_MAX_SAMPLES is None else neural[:ATTR_MAX_SAMPLES]
# #     x = x.clone().requires_grad_(True)
# #     method = cebra.attribution.init(name="jacobian-based",
# #                                     model=model,
# #                                     input_data=x,
# #                                     output_dimension=model.num_output)
# #     return method, method.compute_attribution_map()

# # def spectral_stats(jf_mean):
# #     s = np.linalg.svd(np.asarray(jf_mean, dtype=float), compute_uv=False)
# #     return dict(sv_max=float(s[0]), sv_min=float(s[-1]),
# #                 sv_ratio=float(s[-1] / (s[0] + 1e-30)),
# #                 fro=float(np.sqrt((s ** 2).sum())))

# # def sparsity_stats(jf_mean):
# #     """scale-invariant -- the heatmap's colourbar is NOT comparable across arms."""
# #     s = np.asarray(jf_mean, dtype=float).mean(axis=0)
# #     s = np.clip(s, 0, None)
# #     tot = s.sum() + 1e-30
# #     p = s / tot
# #     pr = (s.sum() ** 2) / ((s ** 2).sum() + 1e-30)          # effective #neurons
# #     ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
# #     srt = np.sort(s); cum = np.cumsum(srt) / tot
# #     gini = float(1 - 2 * np.trapz(cum, dx=1.0 / len(srt)))
# #     return dict(participation_ratio=float(pr), entropy=ent, gini=gini,
# #                 n_neurons=int(len(s)))


# # def evaluate(arm, seed, solver, neural, position, gts, meta):
# #     row = dict(arm=arm, seed=seed, **meta)
# #     emb = compute_embedding(solver, neural)
# #     row.update(decoding_scores(emb, position))

# #     method, result = attribution(solver, neural)
# #     maps = {}
# #     for key, name in (("jf", "jf"),
# #                       ("jf-inv-svd", "jfinv"),
# #                       ("jf-convabs-inv-svd", "jfconvabsinv")):
# #         if key in result:
# #             maps[name] = np.abs(result[key]).mean(0)

# #     np.savez_compressed(os.path.join(OUT_DIR, f"maps_{arm}_s{seed}.npz"), **maps)

# #     row.update({f"sv_{k}": v for k, v in spectral_stats(maps["jf"]).items()})
# #     row.update({f"sp_{k}": v for k, v in sparsity_stats(maps["jf"]).items()})

# #     for gt_name, gt in gts.items():
# #         if gt is None:
# #             continue
# #         for mname, M in maps.items():
# #             if M.shape != gt.shape:
# #                 continue
# #             try:
# #                 row[f"auc_{mname}_{gt_name}"] = float(
# #                     method.compute_attribution_score(M, gt))
# #             except Exception as e:
# #                 row[f"auc_{mname}_{gt_name}"] = float("nan")
# #             for vname, v in auroc_variants(M, gt).items():
# #                 row[f"auroc_{vname}_{mname}_{gt_name}"] = v

# #     # heatmaps on a SHARED colour scale across arms is handled in the summary plot;
# #     # here we just dump the per-arm figure for quick inspection.
# #     for mname, M in maps.items():
# #         plt.figure(figsize=(9, 3.2))
# #         plt.matshow(M, aspect="auto", fignum=0)
# #         plt.colorbar(); plt.title(f"{arm} (seed {seed}) — {mname}")
# #         plt.tight_layout()
# #         plt.savefig(os.path.join(OUT_DIR, f"map_{arm}_s{seed}_{mname}.png"), dpi=140)
# #         plt.close()
# #     return row


# # # ----------------------------------------------------------------------------- MAIN
# # def main():
# #     _, neural, position = get_data()
# #     print(f"neural {tuple(neural.shape)}  position {tuple(position.shape)}  "
# #           f"std={float(neural.std()):.4f}  range=[{float(neural.min())},{float(neural.max())}]")

# #     gt_nb, _ = build_ground_truth(neural.shape[1], "notebook")
# #     gt_md, _ = build_ground_truth(neural.shape[1], "model")
# #     gts = {"gtNB": gt_nb, "gtMODEL": gt_md}

# #     rows = []
# #     for seed in SEEDS:
# #         for arm in ARMS_TO_RUN:
# #             solver, meta = train_arm(arm, seed, neural, position)
# #             row = evaluate(arm, seed, solver, neural, position, gts, meta)
# #             rows.append(row)
# #             print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
# #                               for k, v in row.items()}, indent=1))
# #             import pandas as pd
# #             pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)

# #     import pandas as pd
# #     df = pd.DataFrame(rows)
# #     print("\n================ SUMMARY ================")
# #     cols = [c for c in df.columns
# #             if c.startswith(("auc_", "auroc_global_", "sv_sv_ratio",
# #                              "sp_participation", "R2_", "KNN_"))]
# #     print(df.groupby("arm")[cols].mean().T.to_string())

# #     # shared-colourbar comparison of the jf maps (the honest version of the plot)
# #     fig, axes = plt.subplots(len(ARMS_TO_RUN), 1,
# #                              figsize=(10, 2.6 * len(ARMS_TO_RUN)), squeeze=False)
# #     mats = {a: np.load(os.path.join(OUT_DIR, f"maps_{a}_s{SEEDS[0]}.npz"))["jf"]
# #             for a in ARMS_TO_RUN}
# #     vmax = max(m.max() for m in mats.values())
# #     for ax, a in zip(axes[:, 0], ARMS_TO_RUN):
# #         im = ax.imshow(mats[a], aspect="auto", vmin=0, vmax=vmax)
# #         ax.set_title(f"{a}  (shared scale)")
# #     fig.colorbar(im, ax=axes[:, 0].tolist())
# #     plt.savefig(os.path.join(OUT_DIR, "jf_shared_scale.png"), dpi=150)
# #     plt.close()


# # if __name__ == "__main__":
# #     main()

# # # # =============================================================================
# # # #  xCEBRA / RatInABox  —  CLEAN vs ACORN (manual PGD adversarial training)
# # # #  Built on the OFFICIAL cebra 0.6.0a1 multiobjective API (no fork required).
# # # #
# # # #  pip install --pre "cebra[integrations]==0.6.0a1"
# # # #  pip install --upgrade numpy==1.26 ratinabox
# # # #
# # # #  Fixes applied vs. the previous version (see inline "FIX" comments):
# # # #   1) CRITICAL: build_adversarial's infonce branch referenced `self._inference`/
# # # #      `self.criterion`, but build_adversarial is a free function (no `self`
# # # #      exists) -> NameError on every PGD/infonce attack. Reverted to the
# # # #      already-correct, already-written cosine_infonce(...) path.
# # # #   2) L2 attack ball now matches the professor's Solver: the L2 norm is
# # # #      computed PER TIME-STEP (across channels only), not globally over the
# # # #      whole flattened (channel x time) sample.
# # # #   3) CLAMP_TO_DATA now applies consistently to both "pgd" and "noise"
# # # #      attack modes (previously "noise" always clamped, "pgd" only clamped
# # # #      conditionally). Added CLAMP_RANGE so you can reproduce the professor's
# # # #      hard-coded clamp(0,1) exactly if you want bit-for-bit parity, or leave
# # # #      it at None to clamp to this dataset's own [min,max] instead.
# # # # =============================================================================

# # # import os, sys, copy, json, time, pickle, itertools, warnings
# # # import numpy as np
# # # import torch
# # # import torch.nn.functional as F
# # # import matplotlib
# # # matplotlib.use("Agg")
# # # import matplotlib.pyplot as plt

# # # from sklearn.linear_model import LinearRegression
# # # from sklearn.neighbors import KNeighborsRegressor
# # # from sklearn.model_selection import TimeSeriesSplit
# # # import sys
# # # from pathlib import Path

# # # CEBRA_DIR = Path(__file__).resolve().parent / "CEBRA-original"

# # # for module_name in list(sys.modules):
# # #     if module_name == "cebra" or module_name.startswith("cebra."):
# # #         del sys.modules[module_name]

# # # sys.path.insert(0, str(CEBRA_DIR))

# # # import cebra
# # # from cebra.data import DatasetxCEBRA, ContrastiveMultiObjectiveLoader, TensorDataset
# # # from cebra.solver import MultiObjectiveConfig
# # # from cebra.solver.schedulers import LinearRampUp

# # # # ----------------------------------------------------------------------------- CONFIG
# # # DATA_FILE   = "cynthi_neurons90.p"
# # # DATA_URL    = ("https://zenodo.org/records/15267195/files/"
# # #                "cynthi_neurons90_gridbase0.5_gridmodules3_grid_head_direction_place_speed"
# # #                "_duration2000_noise0.25_bs100_seed231209234.p?download=1")

# # # OUT_DIR     = "xcebra_ratinabox_acorn"
# # # DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# # # # --- verified against the official notebook (cebra.ai/docs/demo_notebooks/
# # # #     Demo_xCEBRA_RatInABox.html) -- every value below matches it exactly ---
# # # NUM_STEPS   = 20
# # # BATCH_SIZE  = 2500
# # # N_LATENTS   = 14
# # # NUM_UNITS   = 256
# # # MODEL_ARCH  = "offset10-model"
# # # LR          = 3e-4
# # # TAU         = 1.0                       # FixedCosineInfoNCE temperature (both objectives)
# # # BEHAVIOR_RANGE = (0, 4)
# # # TIME_RANGE     = (0, N_LATENTS)
# # # TIME_DELTA     = 1
# # # TIME_OFFSET    = 10
# # # RENORMALIZE    = True

# # # LAMBDA_MAX  = 0.1                       # xCEBRA Jacobian-reg end weight (notebook value)

# # # # --- attack ---
# # # ADV_NORM      = "linf"                  # "linf" | "l2"
# # # EPS_REL       = 0.05                    # eps = EPS_REL * std(neural).  -> printed at runtime
# # # ADV_STEPS     = 10
# # # ALPHA_RULE    = 0.2                     # alpha = ALPHA_RULE * eps  (saturates the ball)
# # # ATTACK_OBJ    = "infonce"               # "infonce" (ascend the training loss) | "vat"

# # # # FIX 3: CLAMP_TO_DATA now gates BOTH the "pgd" and "noise" attack modes
# # # # consistently. CLAMP_RANGE=None -> clamp to [neural.min(), neural.max()];
# # # # set CLAMP_RANGE=(0.0, 1.0) to exactly reproduce the professor's
# # # # MultiobjectiveSolver, which unconditionally hard-clamps x_adv to [0,1]
# # # # (only appropriate if your data actually lives in that range -- spike
# # # # counts / RatInABox rates generally don't, so the default here is off).
# # # CLAMP_TO_DATA = False
# # # CLAMP_RANGE   = None                    # e.g. (0.0, 1.0) for exact professor parity

# # # SEEDS         = [0]                     # add 1,2,... once the runtime is acceptable
# # # ARMS_TO_RUN   = ["acorn"] #"cebra", "cebra_2x", "xcebra",
# # # # full menu: cebra | cebra_2x | xcebra | xcebra_2x | noise_2x | acorn | acorn_xreg

# # # ARMS = {
# # #     "cebra":      dict(lam=0.0,        adv=None,    double=False),
# # #     "cebra_2x":   dict(lam=0.0,        adv=None,    double=True ),  # the doubled-step control
# # #     "xcebra":     dict(lam=LAMBDA_MAX, adv=None,    double=False),  # <- the notebook
# # #     "xcebra_2x":  dict(lam=LAMBDA_MAX, adv=None,    double=True ),
# # #     "noise_2x":   dict(lam=0.0,        adv="noise", double=True ),  # eps-ball init, 0 ascent steps
# # #     "acorn":      dict(lam=0.0,        adv="pgd",   double=True ),
# # #     "acorn_xreg": dict(lam=LAMBDA_MAX, adv="pgd",   double=True ),
# # # }

# # # ATTR_MAX_SAMPLES = None                 # e.g. 20000 if the Jacobian eval runs out of memory
# # # os.makedirs(OUT_DIR, exist_ok=True)


# # # # ----------------------------------------------------------------------------- DATA
# # # def get_data():
# # #     if not os.path.exists(DATA_FILE):
# # #         import requests
# # #         print("downloading dataset ...")
# # #         r = requests.get(DATA_URL)
# # #         with open(DATA_FILE, "wb") as f:
# # #             f.write(r.content)
# # #     with open(DATA_FILE, "rb") as f:
# # #         d = pickle.load(f)
# # #     neural   = torch.FloatTensor(d["spikes"]).float()
# # #     position = torch.FloatTensor(d["position"]).float()
# # #     return d, neural, position


# # # # ----------------------------------------------------------------------------- GROUND TRUTH
# # # def build_ground_truth(n_neurons, split="notebook"):
# # #     """
# # #     `split="notebook"`  -> 3 position+grid latents, 11 speed latents  (as in the demo)
# # #     `split="model"`     -> 4 position+grid latents, 10 speed latents  (matches (0,4)/(4,14))
# # #     NOTE: `cells` contains no 'speed' cells at all, so every speed row is all-False.
# # #     """
# # #     cells = np.array(list(itertools.chain.from_iterable(
# # #         [["position"] * 100, ["hd"] * 100, ["position"] * 100, ["grid"] * 60])))
# # #     if len(cells) != n_neurons:
# # #         warnings.warn(f"cell-type list has {len(cells)} entries but data has {n_neurons} "
# # #                       f"neurons -- ground truth disabled.")
# # #         return None, cells
# # #     n_beh = 3 if split == "notebook" else BEHAVIOR_RANGE[1]
# # #     latents = [["position", "grid"]] * n_beh + [["speed"]] * (N_LATENTS - n_beh)
# # #     gt = np.zeros((len(latents), len(cells)), dtype=bool)
# # #     for i, lat in enumerate(latents):
# # #         for j, ct in enumerate(cells):
# # #             gt[i, j] = ct in lat
# # #     return gt, cells


# # # # ----------------------------------------------------------------------------- METRICS
# # # def _avg_rank(x):
# # #     order = np.argsort(x, kind="mergesort")
# # #     ranks = np.empty(len(x), dtype=float)
# # #     ranks[order] = np.arange(1, len(x) + 1, dtype=float)
# # #     xs = x[order]
# # #     i = 0
# # #     while i < len(xs):                                   # average ties
# # #         j = i
# # #         while j + 1 < len(xs) and xs[j + 1] == xs[i]:
# # #             j += 1
# # #         if j > i:
# # #             ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
# # #         i = j + 1
# # #     return ranks

# # # def auroc(scores, labels):
# # #     s = np.asarray(scores, dtype=float).ravel()
# # #     y = np.asarray(labels).ravel().astype(bool)
# # #     n_pos, n_neg = y.sum(), (~y).sum()
# # #     if n_pos == 0 or n_neg == 0:
# # #         return float("nan")
# # #     r = _avg_rank(s)
# # #     return (r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

# # # def auroc_variants(M, gt):
# # #     """global / colnorm / rownorm -- colnorm and rownorm remove a marginal that can
# # #     saturate the pooled AUROC without any real localisation."""
# # #     M = np.asarray(M, dtype=float)
# # #     out = {"global": auroc(M, gt)}
# # #     col = M / (M.mean(axis=0, keepdims=True) + 1e-12)
# # #     row = M / (M.mean(axis=1, keepdims=True) + 1e-12)
# # #     out["colnorm"] = auroc(col, gt)
# # #     out["rownorm"] = auroc(row, gt)
# # #     return out


# # # # ----------------------------------------------------------------------------- ATTACK
# # # def derive_blocks(ranges, renormalize):
# # #     """[(0,4),(0,14)] --> disjoint blocks [0:4],[4:14] plus, per objective, which blocks
# # #     it spans. This reproduces the solver's `renormalize` bookkeeping exactly."""
# # #     bounds = sorted({0} | {a for a, _ in ranges} | {b for _, b in ranges})
# # #     blocks = [slice(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
# # #     per_obj = []
# # #     for a, b in ranges:
# # #         if renormalize:
# # #             per_obj.append([k for k, bl in enumerate(blocks) if bl.start >= a and bl.stop <= b])
# # #         else:
# # #             per_obj.append(None)            # use the raw slice
# # #     return blocks, per_obj

# # # def embed_objective(z, obj_idx, blocks, per_obj, ranges):
# # #     if per_obj[obj_idx] is None:
# # #         a, b = ranges[obj_idx]
# # #         return z[:, a:b]
# # #     parts = [F.normalize(z[:, blocks[k]], dim=1) for k in per_obj[obj_idx]]
# # #     return torch.cat(parts, dim=1)

# # # def cosine_infonce(ref, pos, neg, tau=TAU):
# # #     ref = F.normalize(ref, dim=1)
# # #     pos = F.normalize(pos, dim=1)
# # #     neg = F.normalize(neg, dim=1)
# # #     pos_sim = (ref * pos).sum(1) / tau                       # (B,)
# # #     neg_sim = ref @ neg.t() / tau                            # (B,B)
# # #     return (-pos_sim).mean() + torch.logsumexp(neg_sim, dim=1).mean()

# # # def as_batch_list(batch):
# # #     if hasattr(batch, "reference"):
# # #         return [batch], True
# # #     return list(batch), False

# # # def rebuild_batch(batch, new_refs, was_single):
# # #     # FIX (cosmetic): the previous `zip(*as_batch_list(batch)[0:1] and (...))`
# # #     # always simplified to `zip(batches, new_refs)` anyway -- written out
# # #     # directly here, same behavior, just readable.
# # #     batches, _ = as_batch_list(batch)
# # #     out = []
# # #     for b, r in zip(batches, new_refs):
# # #         nb = copy.copy(b)
# # #         try:
# # #             nb.reference = r
# # #         except Exception:
# # #             object.__setattr__(nb, "reference", r)
# # #         out.append(nb)
# # #     return out[0] if was_single else (type(batch)(out) if isinstance(batch, tuple) else out)

# # # def _forward_full(model, x):
# # #     prev = getattr(model, "split_outputs", None)
# # #     if prev is not None:
# # #         model.split_outputs = False
# # #     z = model(x)
# # #     if isinstance(z, (list, tuple)):
# # #         z = torch.cat(list(z), dim=1)
# # #     if prev is not None:
# # #         model.split_outputs = prev
# # #     return z

# # # def _l2_norm_per_timestep(t, keepdim=True):
# # #     """FIX 2: L2 norm across the CHANNEL axis (dim=1) only, computed
# # #     independently per time-step. batch.reference has shape (B, C, T) here
# # #     (CEBRA conv models take channel-first input), so norming over dim=1 is
# # #     exactly what the professor's code does when it permutes to (B,T,C) and
# # #     norms over dim=-1 -- same per-timestep epsilon-ball, just without the
# # #     extra permute."""
# # #     return t.norm(p=2, dim=1, keepdim=keepdim).clamp(min=1e-12)

# # # def build_adversarial(model, batch, eps, alpha, steps, lo, hi,
# # #                       blocks, per_obj, ranges, mode="pgd", objective=ATTACK_OBJ,
# # #                       norm=ADV_NORM, clamp_to_data=False):
# # #     """PGD on the reference stream only -- mirrors the fork, where the adversarial
# # #     example replaces x_ref and (x_pos, x_neg) stay clean."""
# # #     batches, was_single = as_batch_list(batch)
# # #     refs = [b.reference.detach() for b in batches]

# # #     shared = all(torch.equal(refs[0], r) for r in refs[1:]) if len(refs) > 1 else True
# # #     n_delta = 1 if shared else len(refs)

# # #     def _init():
# # #         if norm == "linf":
# # #             return [torch.empty_like(refs[i if not shared else 0]).uniform_(-eps, eps)
# # #                     for i in range(n_delta)]
# # #         d = []
# # #         for i in range(n_delta):
# # #             v = torch.randn_like(refs[i if not shared else 0])
# # #             v = v / _l2_norm_per_timestep(v)                       # FIX 2
# # #             d.append(v * eps)
# # #         return d

# # #     deltas = [d.requires_grad_(True) for d in _init()]

# # #     # FIX 3: "noise" mode now respects clamp_to_data the same way "pgd" does,
# # #     # instead of always clamping regardless of the flag.
# # #     if mode == "noise" or steps == 0:
# # #         with torch.no_grad():
# # #             new_refs = [
# # #                 (torch.clamp(refs[i] + deltas[0 if shared else i], lo, hi)
# # #                  if clamp_to_data else (refs[i] + deltas[0 if shared else i])).detach()
# # #                 for i in range(len(refs))
# # #             ]
# # #         return rebuild_batch(batch, new_refs, was_single)

# # #     # # positives / negatives do not depend on delta -> embed them once
# # #     # with torch.no_grad():
# # #     #     z_pos = [_forward_full(model, b.positive.detach()) for b in batches]
# # #     #     z_neg = [_forward_full(model, b.negative.detach()) for b in batches]
# # #     #     if objective == "vat":
# # #     #         z_ref0 = [_forward_full(model, r) for r in refs]
# # #     # positives / negatives do not depend on delta -> embed them once
# # #     with torch.no_grad():
# # #         z_pos = []
# # #         z_neg = []
# # #         for b in batches:
# # #             if isinstance(b.positive, (list, tuple)):
# # #                 z_pos.append([_forward_full(model, x.detach()) for x in b.positive])
# # #                 z_neg.append([_forward_full(model, x.detach()) for x in b.negative])
# # #             else:
# # #                 z_pos.append(_forward_full(model, b.positive.detach()))
# # #                 z_neg.append(_forward_full(model, b.negative.detach()))
# # #         if objective == "vat":
# # #             z_ref0 = [_forward_full(model, r) for r in refs]
    
# # #     for _ in range(steps):
# # #         loss = 0.0
# # #         for i, b in enumerate(batches):
# # #             d = deltas[0 if shared else i]
# # #             x = torch.clamp(refs[i] + d, lo, hi) if clamp_to_data else refs[i] + d
# # #             z = _forward_full(model, x)
# # #             if objective == "vat":
# # #                 loss = loss + ((z - z_ref0[i]) ** 2).sum(1).mean()
# # #             else:
# # #                 # FIX 1 (CRITICAL): this used to call self._inference(...) /
# # #                 # self.criterion(...) -- but build_adversarial is a free
# # #                 # function, `self` doesn't exist here, so this raised
# # #                 # NameError on every call. cosine_infonce + embed_objective
# # #                 # is the correct free-function equivalent (same math as the
# # #                 # solver's own criterion, applied to this objective's slice).
# # #                 # loss = loss + cosine_infonce(
# # #                 #     embed_objective(z,        i, blocks, per_obj, ranges),
# # #                 #     embed_objective(z_pos[i], i, blocks, per_obj, ranges),
# # #                 #     embed_objective(z_neg[i], i, blocks, per_obj, ranges))
# # #                 for obj_idx in range(len(ranges)):
# # #                     if isinstance(z_pos[i], (list, tuple)):
# # #                         pos = z_pos[i][obj_idx]
# # #                         neg = z_neg[i][obj_idx]
# # #                     else:
# # #                         pos = z_pos[i]
# # #                         neg = z_neg[i]
# # #                     loss = loss + cosine_infonce(
# # #                         embed_objective(z, obj_idx, blocks, per_obj, ranges),
# # #                         embed_objective(pos, obj_idx, blocks, per_obj, ranges),
# # #                         embed_objective(neg, obj_idx, blocks, per_obj, ranges)
# # #                     )
# # #         grads = torch.autograd.grad(loss, deltas)
# # #         with torch.no_grad():
# # #             for d, g in zip(deltas, grads):
# # #                 if norm == "linf":
# # #                     d.add_(alpha * g.sign())
# # #                     d.clamp_(-eps, eps)
# # #                 else:
# # #                     gn = _l2_norm_per_timestep(g)                  # FIX 2
# # #                     d.add_(alpha * g / gn)
# # #                     dn = _l2_norm_per_timestep(d)                  # FIX 2
# # #                     d.mul_(torch.clamp(eps / dn, max=1.0))

# # #     with torch.no_grad():
# # #         new_refs = [
# # #             (torch.clamp(refs[i] + deltas[0 if shared else i], lo, hi)
# # #              if clamp_to_data else (refs[i] + deltas[0 if shared else i])).detach()
# # #             for i in range(len(refs))
# # #         ]
# # #     return rebuild_batch(batch, new_refs, was_single)


# # # # ----------------------------------------------------------------------------- SOLVER PATCH
# # # def make_adversarial_solver(solver, cfg, attack_kwargs):
# # #     """Rebind solver.__class__ to a subclass of whatever cebra.solver.init produced.
# # #     We never touch MultiCriterion or _inference -- only `step`, through super()."""
# # #     Base = type(solver)

# # #     class AdversarialSolver(Base):
# # #         def step(self, *args, **kwargs):
# # #             c = self._adv_cfg
# # #             batch = args[0]
# # #             print("DEBUG TYPE:", type(batch))
# # #             print("DEBUG LEN:", len(batch) if hasattr(batch, "__len__") else "no len")
        
# # #             if hasattr(batch, "__len__"):
# # #                 for i,b in enumerate(batch):
# # #                     print(i,type(b.reference),b.reference.shape)

# # #             def _adv(b):
# # #                 return build_adversarial(self.model, b, mode=c["adv"], **self._attack_kwargs)

# # #             if not c["double"]:
# # #                 if c["adv"] is None:
# # #                     return super().step(*args, **kwargs)
# # #                 return super().step(_adv(batch), *args[1:], **kwargs)

# # #             # --- update 1: unconditional clean step (fork-faithful) ---------------
# # #             stats = super().step(*args, **kwargs)
# # #             # --- update 2: built with the ALREADY-UPDATED parameters --------------
# # #             if c["adv"] is None:
# # #                 return super().step(*args, **kwargs)          # cebra_2x / xcebra_2x
# # #             return super().step(_adv(batch), *args[1:], **kwargs)

# # #     solver.__class__ = AdversarialSolver
# # #     solver._adv_cfg = cfg
# # #     solver._attack_kwargs = attack_kwargs
# # #     return solver


# # # # ----------------------------------------------------------------------------- TRAIN ONE ARM
# # # def train_arm(arm_name, seed, neural, position):
# # #     cfg = ARMS[arm_name]
# # #     torch.manual_seed(seed); np.random.seed(seed)

# # #     data = DatasetxCEBRA(neural, position=position)
# # #     loader = ContrastiveMultiObjectiveLoader(dataset=data,
# # #                                              num_steps=NUM_STEPS,
# # #                                              batch_size=BATCH_SIZE).to(DEVICE)

# # #     config = MultiObjectiveConfig(loader)
# # #     config.set_slice(*BEHAVIOR_RANGE)
# # #     config.set_loss("FixedCosineInfoNCE", temperature=TAU)
# # #     config.set_distribution("time_delta", time_delta=TIME_DELTA, label_name="position")
# # #     config.push()

# # #     config.set_slice(*TIME_RANGE)
# # #     config.set_loss("FixedCosineInfoNCE", temperature=TAU)
# # #     config.set_distribution("time", time_offset=TIME_OFFSET)
# # #     config.push()
# # #     config.finalize()

# # #     criterion      = config.criterion
# # #     feature_ranges = config.feature_ranges

# # #     model = cebra.models.init(name=MODEL_ARCH,
# # #                               num_neurons=data.neural.shape[1],
# # #                               num_units=NUM_UNITS,
# # #                               num_output=N_LATENTS).to(DEVICE)
# # #     data.configure_for(model)

# # #     opt = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()),
# # #                            lr=LR, weight_decay=0)

# # #     solver = cebra.solver.init(name="multiobjective-solver",
# # #                                model=model,
# # #                                feature_ranges=feature_ranges,
# # #                                regularizer=cebra.models.jacobian_regularizer.JacobianReg(),
# # #                                renormalize=RENORMALIZE,
# # #                                use_sam=False,
# # #                                criterion=criterion,
# # #                                optimizer=opt,
# # #                                tqdm_on=True).to(DEVICE)

# # #     # lambda = 0 arms: same scheduler object, both weights zero -> penalty never fires.
# # #     sched = LinearRampUp(n_splits=2,
# # #                          step_to_switch_on_reg=NUM_STEPS // 4,
# # #                          step_to_switch_off_reg=NUM_STEPS // 2,
# # #                          start_weight=0.0,
# # #                          end_weight=cfg["lam"])

# # #     ranges = [tuple(BEHAVIOR_RANGE), tuple(TIME_RANGE)]
# # #     blocks, per_obj = derive_blocks(ranges, RENORMALIZE)

# # #     eps   = EPS_REL * float(neural.std())
# # #     alpha = ALPHA_RULE * eps

# # #     # FIX 3: lo/hi now come from CLAMP_RANGE if you set one (e.g. (0.0, 1.0)
# # #     # for exact professor parity), otherwise from this dataset's own range.
# # #     if CLAMP_RANGE is not None:
# # #         lo, hi = CLAMP_RANGE
# # #     else:
# # #         lo, hi = float(neural.min()), float(neural.max())

# # #     attack_kwargs = dict(eps=eps, alpha=alpha, steps=ADV_STEPS,
# # #                          lo=lo, hi=hi,
# # #                          blocks=blocks, per_obj=per_obj, ranges=ranges,
# # #                          objective=ATTACK_OBJ, norm=ADV_NORM,
# # #                          clamp_to_data=CLAMP_TO_DATA)

# # #     if cfg["adv"] is not None or cfg["double"]:
# # #         solver = make_adversarial_solver(solver, cfg, attack_kwargs)

# # #     print(f"\n=== arm={arm_name} seed={seed} lam={cfg['lam']} adv={cfg['adv']} "
# # #           f"double={cfg['double']} eps={eps:.4f} alpha={alpha:.4f} "
# # #           f"clamp={'[%.3f,%.3f]' % (lo, hi) if CLAMP_TO_DATA else 'off'} dev={DEVICE}")
# # #     t0 = time.time()
# # #     solver.fit(loader=loader, valid_loader=None, log_frequency=None,
# # #                scheduler_regularizer=sched, scheduler_loss=None)
# # #     wall = time.time() - t0
# # #     return solver, dict(wall_s=wall, eps=eps, alpha=alpha)


# # # # ----------------------------------------------------------------------------- EVAL
# # # def compute_embedding(solver, neural):
# # #     d = TensorDataset(neural, continuous=torch.zeros(len(neural)))
# # #     d.configure_for(solver.model)
# # #     d = d[torch.arange(len(d))]
# # #     solver.model.split_outputs = False
# # #     with torch.no_grad():
# # #         return solver.model(d.to(DEVICE)).detach().cpu()

# # # def decoding_scores(embedding, position):
# # #     X_b = embedding[:, slice(*BEHAVIOR_RANGE)].numpy()
# # #     X_t = embedding[:, BEHAVIOR_RANGE[1]:N_LATENTS].numpy()
# # #     y   = position[:len(embedding)].numpy()
# # #     out = {}
# # #     out["R2_behavior"] = LinearRegression().fit(X_b, y).score(X_b, y)
# # #     out["R2_time"]     = LinearRegression().fit(X_t, y).score(X_t, y)
# # #     tscv = TimeSeriesSplit(n_splits=5)
# # #     for tag, X in (("behavior", X_b), ("time", X_t)):
# # #         sc = []
# # #         for tr, va in tscv.split(X):
# # #             sc.append(KNeighborsRegressor().fit(X[tr], y[tr]).score(X[va], y[va]))
# # #         out[f"KNN_{tag}"] = float(np.mean(sc))
# # #     return out

# # # def attribution(solver, neural):
# # #     model = solver.model.to(DEVICE)
# # #     model.split_outputs = False
# # #     x = neural if ATTR_MAX_SAMPLES is None else neural[:ATTR_MAX_SAMPLES]
# # #     x = x.clone().requires_grad_(True)
# # #     method = cebra.attribution.init(name="jacobian-based",
# # #                                     model=model,
# # #                                     input_data=x,
# # #                                     output_dimension=model.num_output)
# # #     return method, method.compute_attribution_map()

# # # def spectral_stats(jf_mean):
# # #     s = np.linalg.svd(np.asarray(jf_mean, dtype=float), compute_uv=False)
# # #     return dict(sv_max=float(s[0]), sv_min=float(s[-1]),
# # #                 sv_ratio=float(s[-1] / (s[0] + 1e-30)),
# # #                 fro=float(np.sqrt((s ** 2).sum())))

# # # def sparsity_stats(jf_mean):
# # #     """scale-invariant -- the heatmap's colourbar is NOT comparable across arms."""
# # #     s = np.asarray(jf_mean, dtype=float).mean(axis=0)
# # #     s = np.clip(s, 0, None)
# # #     tot = s.sum() + 1e-30
# # #     p = s / tot
# # #     pr = (s.sum() ** 2) / ((s ** 2).sum() + 1e-30)          # effective #neurons
# # #     ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
# # #     srt = np.sort(s); cum = np.cumsum(srt) / tot
# # #     gini = float(1 - 2 * np.trapz(cum, dx=1.0 / len(srt)))
# # #     return dict(participation_ratio=float(pr), entropy=ent, gini=gini,
# # #                 n_neurons=int(len(s)))


# # # def evaluate(arm, seed, solver, neural, position, gts, meta):
# # #     row = dict(arm=arm, seed=seed, **meta)
# # #     emb = compute_embedding(solver, neural)
# # #     row.update(decoding_scores(emb, position))

# # #     method, result = attribution(solver, neural)
# # #     maps = {}
# # #     for key, name in (("jf", "jf"),
# # #                       ("jf-inv-svd", "jfinv"),
# # #                       ("jf-convabs-inv-svd", "jfconvabsinv")):
# # #         if key in result:
# # #             maps[name] = np.abs(result[key]).mean(0)

# # #     np.savez_compressed(os.path.join(OUT_DIR, f"maps_{arm}_s{seed}.npz"), **maps)

# # #     row.update({f"sv_{k}": v for k, v in spectral_stats(maps["jf"]).items()})
# # #     row.update({f"sp_{k}": v for k, v in sparsity_stats(maps["jf"]).items()})

# # #     for gt_name, gt in gts.items():
# # #         if gt is None:
# # #             continue
# # #         for mname, M in maps.items():
# # #             if M.shape != gt.shape:
# # #                 continue
# # #             try:
# # #                 row[f"auc_{mname}_{gt_name}"] = float(
# # #                     method.compute_attribution_score(M, gt))
# # #             except Exception as e:
# # #                 row[f"auc_{mname}_{gt_name}"] = float("nan")
# # #             for vname, v in auroc_variants(M, gt).items():
# # #                 row[f"auroc_{vname}_{mname}_{gt_name}"] = v

# # #     # heatmaps on a SHARED colour scale across arms is handled in the summary plot;
# # #     # here we just dump the per-arm figure for quick inspection.
# # #     for mname, M in maps.items():
# # #         plt.figure(figsize=(9, 3.2))
# # #         plt.matshow(M, aspect="auto", fignum=0)
# # #         plt.colorbar(); plt.title(f"{arm} (seed {seed}) — {mname}")
# # #         plt.tight_layout()
# # #         plt.savefig(os.path.join(OUT_DIR, f"map_{arm}_s{seed}_{mname}.png"), dpi=140)
# # #         plt.close()
# # #     return row


# # # # ----------------------------------------------------------------------------- MAIN
# # # def main():
# # #     _, neural, position = get_data()
# # #     print(f"neural {tuple(neural.shape)}  position {tuple(position.shape)}  "
# # #           f"std={float(neural.std()):.4f}  range=[{float(neural.min())},{float(neural.max())}]")

# # #     gt_nb, _ = build_ground_truth(neural.shape[1], "notebook")
# # #     gt_md, _ = build_ground_truth(neural.shape[1], "model")
# # #     gts = {"gtNB": gt_nb, "gtMODEL": gt_md}

# # #     rows = []
# # #     for seed in SEEDS:
# # #         for arm in ARMS_TO_RUN:
# # #             solver, meta = train_arm(arm, seed, neural, position)
# # #             row = evaluate(arm, seed, solver, neural, position, gts, meta)
# # #             rows.append(row)
# # #             print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
# # #                               for k, v in row.items()}, indent=1))
# # #             import pandas as pd
# # #             pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)

# # #     import pandas as pd
# # #     df = pd.DataFrame(rows)
# # #     print("\n================ SUMMARY ================")
# # #     cols = [c for c in df.columns
# # #             if c.startswith(("auc_", "auroc_global_", "sv_sv_ratio",
# # #                              "sp_participation", "R2_", "KNN_"))]
# # #     print(df.groupby("arm")[cols].mean().T.to_string())

# # #     # shared-colourbar comparison of the jf maps (the honest version of the plot)
# # #     fig, axes = plt.subplots(len(ARMS_TO_RUN), 1,
# # #                              figsize=(10, 2.6 * len(ARMS_TO_RUN)), squeeze=False)
# # #     mats = {a: np.load(os.path.join(OUT_DIR, f"maps_{a}_s{SEEDS[0]}.npz"))["jf"]
# # #             for a in ARMS_TO_RUN}
# # #     vmax = max(m.max() for m in mats.values())
# # #     for ax, a in zip(axes[:, 0], ARMS_TO_RUN):
# # #         im = ax.imshow(mats[a], aspect="auto", vmin=0, vmax=vmax)
# # #         ax.set_title(f"{a}  (shared scale)")
# # #     fig.colorbar(im, ax=axes[:, 0].tolist())
# # #     plt.savefig(os.path.join(OUT_DIR, "jf_shared_scale.png"), dpi=150)
# # #     plt.close()


# # # if __name__ == "__main__":
# # #     main()

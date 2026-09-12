# =============================================================================
#  xCEBRA / RatInABox  —  CLEAN vs ACORN (manual PGD adversarial training)
#  Built on the OFFICIAL cebra 0.6.0a1 multiobjective API (no fork required).
#
#  pip install --pre "cebra[integrations]==0.6.0a1"
#  pip install --upgrade numpy==1.26 ratinabox
#
#  Fixes applied vs. the previous version (see inline "FIX" comments):
#   1) CRITICAL: build_adversarial's infonce branch referenced `self._inference`/
#      `self.criterion`, but build_adversarial is a free function (no `self`
#      exists) -> NameError on every PGD/infonce attack. Reverted to the
#      already-correct, already-written cosine_infonce(...) path.
#   2) L2 attack ball now matches the professor's Solver: the L2 norm is
#      computed PER TIME-STEP (across channels only), not globally over the
#      whole flattened (channel x time) sample.
#   3) CLAMP_TO_DATA now applies consistently to both "pgd" and "noise"
#      attack modes (previously "noise" always clamped, "pgd" only clamped
#      conditionally). Added CLAMP_RANGE so you can reproduce the professor's
#      hard-coded clamp(0,1) exactly if you want bit-for-bit parity, or leave
#      it at None to clamp to this dataset's own [min,max] instead.
# =============================================================================

import os, sys, copy, json, time, pickle, itertools, warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit
import sys
from pathlib import Path

CEBRA_DIR = Path(__file__).resolve().parent / "CEBRA-original"

for module_name in list(sys.modules):
    if module_name == "cebra" or module_name.startswith("cebra."):
        del sys.modules[module_name]

sys.path.insert(0, str(CEBRA_DIR))

import cebra
from cebra.data import DatasetxCEBRA, ContrastiveMultiObjectiveLoader, TensorDataset
from cebra.solver import MultiObjectiveConfig
from cebra.solver.schedulers import LinearRampUp

# ----------------------------------------------------------------------------- CONFIG
DATA_FILE   = "cynthi_neurons90.p"
DATA_URL    = ("https://zenodo.org/records/15267195/files/"
               "cynthi_neurons90_gridbase0.5_gridmodules3_grid_head_direction_place_speed"
               "_duration2000_noise0.25_bs100_seed231209234.p?download=1")

OUT_DIR     = "xcebra_ratinabox_acorn"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# --- verified against the official notebook (cebra.ai/docs/demo_notebooks/
#     Demo_xCEBRA_RatInABox.html) -- every value below matches it exactly ---
NUM_STEPS   = 20
BATCH_SIZE  = 2500
N_LATENTS   = 14
NUM_UNITS   = 256
MODEL_ARCH  = "offset10-model"
LR          = 3e-4
TAU         = 1.0                       # FixedCosineInfoNCE temperature (both objectives)
BEHAVIOR_RANGE = (0, 4)
TIME_RANGE     = (0, N_LATENTS)
TIME_DELTA     = 1
TIME_OFFSET    = 10
RENORMALIZE    = True

LAMBDA_MAX  = 0.1                       # xCEBRA Jacobian-reg end weight (notebook value)

# --- attack ---
ADV_NORM      = "linf"                  # "linf" | "l2"
EPS_REL       = 0.05                    # eps = EPS_REL * std(neural).  -> printed at runtime
ADV_STEPS     = 10
ALPHA_RULE    = 0.2                     # alpha = ALPHA_RULE * eps  (saturates the ball)
ATTACK_OBJ    = "infonce"               # "infonce" (ascend the training loss) | "vat"

# FIX 3: CLAMP_TO_DATA now gates BOTH the "pgd" and "noise" attack modes
# consistently. CLAMP_RANGE=None -> clamp to [neural.min(), neural.max()];
# set CLAMP_RANGE=(0.0, 1.0) to exactly reproduce the professor's
# MultiobjectiveSolver, which unconditionally hard-clamps x_adv to [0,1]
# (only appropriate if your data actually lives in that range -- spike
# counts / RatInABox rates generally don't, so the default here is off).
CLAMP_TO_DATA = False
CLAMP_RANGE   = None                    # e.g. (0.0, 1.0) for exact professor parity

SEEDS         = [0]                     # add 1,2,... once the runtime is acceptable
ARMS_TO_RUN   = ["acorn"] #"cebra", "cebra_2x", "xcebra",
# full menu: cebra | cebra_2x | xcebra | xcebra_2x | noise_2x | acorn | acorn_xreg

ARMS = {
    "cebra":      dict(lam=0.0,        adv=None,    double=False),
    "cebra_2x":   dict(lam=0.0,        adv=None,    double=True ),  # the doubled-step control
    "xcebra":     dict(lam=LAMBDA_MAX, adv=None,    double=False),  # <- the notebook
    "xcebra_2x":  dict(lam=LAMBDA_MAX, adv=None,    double=True ),
    "noise_2x":   dict(lam=0.0,        adv="noise", double=True ),  # eps-ball init, 0 ascent steps
    "acorn":      dict(lam=0.0,        adv="pgd",   double=True ),
    "acorn_xreg": dict(lam=LAMBDA_MAX, adv="pgd",   double=True ),
}

ATTR_MAX_SAMPLES = None                 # e.g. 20000 if the Jacobian eval runs out of memory
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------- DATA
def get_data():
    if not os.path.exists(DATA_FILE):
        import requests
        print("downloading dataset ...")
        r = requests.get(DATA_URL)
        with open(DATA_FILE, "wb") as f:
            f.write(r.content)
    with open(DATA_FILE, "rb") as f:
        d = pickle.load(f)
    neural   = torch.FloatTensor(d["spikes"]).float()
    position = torch.FloatTensor(d["position"]).float()
    return d, neural, position


# ----------------------------------------------------------------------------- GROUND TRUTH
def build_ground_truth(n_neurons, split="notebook"):
    """
    `split="notebook"`  -> 3 position+grid latents, 11 speed latents  (as in the demo)
    `split="model"`     -> 4 position+grid latents, 10 speed latents  (matches (0,4)/(4,14))
    NOTE: `cells` contains no 'speed' cells at all, so every speed row is all-False.
    """
    cells = np.array(list(itertools.chain.from_iterable(
        [["position"] * 100, ["hd"] * 100, ["position"] * 100, ["grid"] * 60])))
    if len(cells) != n_neurons:
        warnings.warn(f"cell-type list has {len(cells)} entries but data has {n_neurons} "
                      f"neurons -- ground truth disabled.")
        return None, cells
    n_beh = 3 if split == "notebook" else BEHAVIOR_RANGE[1]
    latents = [["position", "grid"]] * n_beh + [["speed"]] * (N_LATENTS - n_beh)
    gt = np.zeros((len(latents), len(cells)), dtype=bool)
    for i, lat in enumerate(latents):
        for j, ct in enumerate(cells):
            gt[i, j] = ct in lat
    return gt, cells


# ----------------------------------------------------------------------------- METRICS
def _avg_rank(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):                                   # average ties
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks

def auroc(scores, labels):
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _avg_rank(s)
    return (r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def auroc_variants(M, gt):
    """global / colnorm / rownorm -- colnorm and rownorm remove a marginal that can
    saturate the pooled AUROC without any real localisation."""
    M = np.asarray(M, dtype=float)
    out = {"global": auroc(M, gt)}
    col = M / (M.mean(axis=0, keepdims=True) + 1e-12)
    row = M / (M.mean(axis=1, keepdims=True) + 1e-12)
    out["colnorm"] = auroc(col, gt)
    out["rownorm"] = auroc(row, gt)
    return out


# ----------------------------------------------------------------------------- ATTACK
def derive_blocks(ranges, renormalize):
    """[(0,4),(0,14)] --> disjoint blocks [0:4],[4:14] plus, per objective, which blocks
    it spans. This reproduces the solver's `renormalize` bookkeeping exactly."""
    bounds = sorted({0} | {a for a, _ in ranges} | {b for _, b in ranges})
    blocks = [slice(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    per_obj = []
    for a, b in ranges:
        if renormalize:
            per_obj.append([k for k, bl in enumerate(blocks) if bl.start >= a and bl.stop <= b])
        else:
            per_obj.append(None)            # use the raw slice
    return blocks, per_obj

def embed_objective(z, obj_idx, blocks, per_obj, ranges):
    if per_obj[obj_idx] is None:
        a, b = ranges[obj_idx]
        return z[:, a:b]
    parts = [F.normalize(z[:, blocks[k]], dim=1) for k in per_obj[obj_idx]]
    return torch.cat(parts, dim=1)

def cosine_infonce(ref, pos, neg, tau=TAU):
    ref = F.normalize(ref, dim=1)
    pos = F.normalize(pos, dim=1)
    neg = F.normalize(neg, dim=1)
    pos_sim = (ref * pos).sum(1) / tau                       # (B,)
    neg_sim = ref @ neg.t() / tau                            # (B,B)
    return (-pos_sim).mean() + torch.logsumexp(neg_sim, dim=1).mean()

def as_batch_list(batch):
    if hasattr(batch, "reference"):
        return [batch], True
    return list(batch), False

def rebuild_batch(batch, new_refs, was_single):
    # FIX (cosmetic): the previous `zip(*as_batch_list(batch)[0:1] and (...))`
    # always simplified to `zip(batches, new_refs)` anyway -- written out
    # directly here, same behavior, just readable.
    batches, _ = as_batch_list(batch)
    out = []
    for b, r in zip(batches, new_refs):
        nb = copy.copy(b)
        try:
            nb.reference = r
        except Exception:
            object.__setattr__(nb, "reference", r)
        out.append(nb)
    return out[0] if was_single else (type(batch)(out) if isinstance(batch, tuple) else out)

def _forward_full(model, x):
    prev = getattr(model, "split_outputs", None)
    if prev is not None:
        model.split_outputs = False
    z = model(x)
    if isinstance(z, (list, tuple)):
        z = torch.cat(list(z), dim=1)
    if prev is not None:
        model.split_outputs = prev
    return z

def _l2_norm_per_timestep(t, keepdim=True):
    """FIX 2: L2 norm across the CHANNEL axis (dim=1) only, computed
    independently per time-step. batch.reference has shape (B, C, T) here
    (CEBRA conv models take channel-first input), so norming over dim=1 is
    exactly what the professor's code does when it permutes to (B,T,C) and
    norms over dim=-1 -- same per-timestep epsilon-ball, just without the
    extra permute."""
    return t.norm(p=2, dim=1, keepdim=keepdim).clamp(min=1e-12)

def build_adversarial(model, batch, eps, alpha, steps, lo, hi,
                      blocks, per_obj, ranges, mode="pgd", objective=ATTACK_OBJ,
                      norm=ADV_NORM, clamp_to_data=False):
    """PGD on the reference stream only -- mirrors the fork, where the adversarial
    example replaces x_ref and (x_pos, x_neg) stay clean."""
    batches, was_single = as_batch_list(batch)
    refs = [b.reference.detach() for b in batches]

    shared = all(torch.equal(refs[0], r) for r in refs[1:]) if len(refs) > 1 else True
    n_delta = 1 if shared else len(refs)

    def _init():
        if norm == "linf":
            return [torch.empty_like(refs[i if not shared else 0]).uniform_(-eps, eps)
                    for i in range(n_delta)]
        d = []
        for i in range(n_delta):
            v = torch.randn_like(refs[i if not shared else 0])
            v = v / _l2_norm_per_timestep(v)                       # FIX 2
            d.append(v * eps)
        return d

    deltas = [d.requires_grad_(True) for d in _init()]

    # FIX 3: "noise" mode now respects clamp_to_data the same way "pgd" does,
    # instead of always clamping regardless of the flag.
    if mode == "noise" or steps == 0:
        with torch.no_grad():
            new_refs = [
                (torch.clamp(refs[i] + deltas[0 if shared else i], lo, hi)
                 if clamp_to_data else (refs[i] + deltas[0 if shared else i])).detach()
                for i in range(len(refs))
            ]
        return rebuild_batch(batch, new_refs, was_single)

    # # positives / negatives do not depend on delta -> embed them once
    # with torch.no_grad():
    #     z_pos = [_forward_full(model, b.positive.detach()) for b in batches]
    #     z_neg = [_forward_full(model, b.negative.detach()) for b in batches]
    #     if objective == "vat":
    #         z_ref0 = [_forward_full(model, r) for r in refs]
    # positives / negatives do not depend on delta -> embed them once
    with torch.no_grad():
        z_pos = []
        z_neg = []
        for b in batches:
            if isinstance(b.positive, (list, tuple)):
                z_pos.append([_forward_full(model, x.detach()) for x in b.positive])
                z_neg.append([_forward_full(model, x.detach()) for x in b.negative])
            else:
                z_pos.append(_forward_full(model, b.positive.detach()))
                z_neg.append(_forward_full(model, b.negative.detach()))
        if objective == "vat":
            z_ref0 = [_forward_full(model, r) for r in refs]
    
    for _ in range(steps):
        loss = 0.0
        for i, b in enumerate(batches):
            d = deltas[0 if shared else i]
            x = torch.clamp(refs[i] + d, lo, hi) if clamp_to_data else refs[i] + d
            z = _forward_full(model, x)
            if objective == "vat":
                loss = loss + ((z - z_ref0[i]) ** 2).sum(1).mean()
            else:
                # FIX 1 (CRITICAL): this used to call self._inference(...) /
                # self.criterion(...) -- but build_adversarial is a free
                # function, `self` doesn't exist here, so this raised
                # NameError on every call. cosine_infonce + embed_objective
                # is the correct free-function equivalent (same math as the
                # solver's own criterion, applied to this objective's slice).
                loss = loss + cosine_infonce(
                    embed_objective(z,        i, blocks, per_obj, ranges),
                    embed_objective(z_pos[i], i, blocks, per_obj, ranges),
                    embed_objective(z_neg[i], i, blocks, per_obj, ranges))
        grads = torch.autograd.grad(loss, deltas)
        with torch.no_grad():
            for d, g in zip(deltas, grads):
                if norm == "linf":
                    d.add_(alpha * g.sign())
                    d.clamp_(-eps, eps)
                else:
                    gn = _l2_norm_per_timestep(g)                  # FIX 2
                    d.add_(alpha * g / gn)
                    dn = _l2_norm_per_timestep(d)                  # FIX 2
                    d.mul_(torch.clamp(eps / dn, max=1.0))

    with torch.no_grad():
        new_refs = [
            (torch.clamp(refs[i] + deltas[0 if shared else i], lo, hi)
             if clamp_to_data else (refs[i] + deltas[0 if shared else i])).detach()
            for i in range(len(refs))
        ]
    return rebuild_batch(batch, new_refs, was_single)


# ----------------------------------------------------------------------------- SOLVER PATCH
def make_adversarial_solver(solver, cfg, attack_kwargs):
    """Rebind solver.__class__ to a subclass of whatever cebra.solver.init produced.
    We never touch MultiCriterion or _inference -- only `step`, through super()."""
    Base = type(solver)

    class AdversarialSolver(Base):
        def step(self, *args, **kwargs):
            c = self._adv_cfg
            batch = args[0]
            print("DEBUG TYPE:", type(batch))
            print("DEBUG LEN:", len(batch) if hasattr(batch, "__len__") else "no len")
        
            if hasattr(batch, "__len__"):
                for i,b in enumerate(batch):
                    print(i,type(b.reference),b.reference.shape)

            def _adv(b):
                return build_adversarial(self.model, b, mode=c["adv"], **self._attack_kwargs)

            if not c["double"]:
                if c["adv"] is None:
                    return super().step(*args, **kwargs)
                return super().step(_adv(batch), *args[1:], **kwargs)

            # --- update 1: unconditional clean step (fork-faithful) ---------------
            stats = super().step(*args, **kwargs)
            # --- update 2: built with the ALREADY-UPDATED parameters --------------
            if c["adv"] is None:
                return super().step(*args, **kwargs)          # cebra_2x / xcebra_2x
            return super().step(_adv(batch), *args[1:], **kwargs)

    solver.__class__ = AdversarialSolver
    solver._adv_cfg = cfg
    solver._attack_kwargs = attack_kwargs
    return solver


# ----------------------------------------------------------------------------- TRAIN ONE ARM
def train_arm(arm_name, seed, neural, position):
    cfg = ARMS[arm_name]
    torch.manual_seed(seed); np.random.seed(seed)

    data = DatasetxCEBRA(neural, position=position)
    loader = ContrastiveMultiObjectiveLoader(dataset=data,
                                             num_steps=NUM_STEPS,
                                             batch_size=BATCH_SIZE).to(DEVICE)

    config = MultiObjectiveConfig(loader)
    config.set_slice(*BEHAVIOR_RANGE)
    config.set_loss("FixedCosineInfoNCE", temperature=TAU)
    config.set_distribution("time_delta", time_delta=TIME_DELTA, label_name="position")
    config.push()

    config.set_slice(*TIME_RANGE)
    config.set_loss("FixedCosineInfoNCE", temperature=TAU)
    config.set_distribution("time", time_offset=TIME_OFFSET)
    config.push()
    config.finalize()

    criterion      = config.criterion
    feature_ranges = config.feature_ranges

    model = cebra.models.init(name=MODEL_ARCH,
                              num_neurons=data.neural.shape[1],
                              num_units=NUM_UNITS,
                              num_output=N_LATENTS).to(DEVICE)
    data.configure_for(model)

    opt = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()),
                           lr=LR, weight_decay=0)

    solver = cebra.solver.init(name="multiobjective-solver",
                               model=model,
                               feature_ranges=feature_ranges,
                               regularizer=cebra.models.jacobian_regularizer.JacobianReg(),
                               renormalize=RENORMALIZE,
                               use_sam=False,
                               criterion=criterion,
                               optimizer=opt,
                               tqdm_on=True).to(DEVICE)

    # lambda = 0 arms: same scheduler object, both weights zero -> penalty never fires.
    sched = LinearRampUp(n_splits=2,
                         step_to_switch_on_reg=NUM_STEPS // 4,
                         step_to_switch_off_reg=NUM_STEPS // 2,
                         start_weight=0.0,
                         end_weight=cfg["lam"])

    ranges = [tuple(BEHAVIOR_RANGE), tuple(TIME_RANGE)]
    blocks, per_obj = derive_blocks(ranges, RENORMALIZE)

    eps   = EPS_REL * float(neural.std())
    alpha = ALPHA_RULE * eps

    # FIX 3: lo/hi now come from CLAMP_RANGE if you set one (e.g. (0.0, 1.0)
    # for exact professor parity), otherwise from this dataset's own range.
    if CLAMP_RANGE is not None:
        lo, hi = CLAMP_RANGE
    else:
        lo, hi = float(neural.min()), float(neural.max())

    attack_kwargs = dict(eps=eps, alpha=alpha, steps=ADV_STEPS,
                         lo=lo, hi=hi,
                         blocks=blocks, per_obj=per_obj, ranges=ranges,
                         objective=ATTACK_OBJ, norm=ADV_NORM,
                         clamp_to_data=CLAMP_TO_DATA)

    if cfg["adv"] is not None or cfg["double"]:
        solver = make_adversarial_solver(solver, cfg, attack_kwargs)

    print(f"\n=== arm={arm_name} seed={seed} lam={cfg['lam']} adv={cfg['adv']} "
          f"double={cfg['double']} eps={eps:.4f} alpha={alpha:.4f} "
          f"clamp={'[%.3f,%.3f]' % (lo, hi) if CLAMP_TO_DATA else 'off'} dev={DEVICE}")
    t0 = time.time()
    solver.fit(loader=loader, valid_loader=None, log_frequency=None,
               scheduler_regularizer=sched, scheduler_loss=None)
    wall = time.time() - t0
    return solver, dict(wall_s=wall, eps=eps, alpha=alpha)


# ----------------------------------------------------------------------------- EVAL
def compute_embedding(solver, neural):
    d = TensorDataset(neural, continuous=torch.zeros(len(neural)))
    d.configure_for(solver.model)
    d = d[torch.arange(len(d))]
    solver.model.split_outputs = False
    with torch.no_grad():
        return solver.model(d.to(DEVICE)).detach().cpu()

def decoding_scores(embedding, position):
    X_b = embedding[:, slice(*BEHAVIOR_RANGE)].numpy()
    X_t = embedding[:, BEHAVIOR_RANGE[1]:N_LATENTS].numpy()
    y   = position[:len(embedding)].numpy()
    out = {}
    out["R2_behavior"] = LinearRegression().fit(X_b, y).score(X_b, y)
    out["R2_time"]     = LinearRegression().fit(X_t, y).score(X_t, y)
    tscv = TimeSeriesSplit(n_splits=5)
    for tag, X in (("behavior", X_b), ("time", X_t)):
        sc = []
        for tr, va in tscv.split(X):
            sc.append(KNeighborsRegressor().fit(X[tr], y[tr]).score(X[va], y[va]))
        out[f"KNN_{tag}"] = float(np.mean(sc))
    return out

def attribution(solver, neural):
    model = solver.model.to(DEVICE)
    model.split_outputs = False
    x = neural if ATTR_MAX_SAMPLES is None else neural[:ATTR_MAX_SAMPLES]
    x = x.clone().requires_grad_(True)
    method = cebra.attribution.init(name="jacobian-based",
                                    model=model,
                                    input_data=x,
                                    output_dimension=model.num_output)
    return method, method.compute_attribution_map()

def spectral_stats(jf_mean):
    s = np.linalg.svd(np.asarray(jf_mean, dtype=float), compute_uv=False)
    return dict(sv_max=float(s[0]), sv_min=float(s[-1]),
                sv_ratio=float(s[-1] / (s[0] + 1e-30)),
                fro=float(np.sqrt((s ** 2).sum())))

def sparsity_stats(jf_mean):
    """scale-invariant -- the heatmap's colourbar is NOT comparable across arms."""
    s = np.asarray(jf_mean, dtype=float).mean(axis=0)
    s = np.clip(s, 0, None)
    tot = s.sum() + 1e-30
    p = s / tot
    pr = (s.sum() ** 2) / ((s ** 2).sum() + 1e-30)          # effective #neurons
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    srt = np.sort(s); cum = np.cumsum(srt) / tot
    gini = float(1 - 2 * np.trapz(cum, dx=1.0 / len(srt)))
    return dict(participation_ratio=float(pr), entropy=ent, gini=gini,
                n_neurons=int(len(s)))


def evaluate(arm, seed, solver, neural, position, gts, meta):
    row = dict(arm=arm, seed=seed, **meta)
    emb = compute_embedding(solver, neural)
    row.update(decoding_scores(emb, position))

    method, result = attribution(solver, neural)
    maps = {}
    for key, name in (("jf", "jf"),
                      ("jf-inv-svd", "jfinv"),
                      ("jf-convabs-inv-svd", "jfconvabsinv")):
        if key in result:
            maps[name] = np.abs(result[key]).mean(0)

    np.savez_compressed(os.path.join(OUT_DIR, f"maps_{arm}_s{seed}.npz"), **maps)

    row.update({f"sv_{k}": v for k, v in spectral_stats(maps["jf"]).items()})
    row.update({f"sp_{k}": v for k, v in sparsity_stats(maps["jf"]).items()})

    for gt_name, gt in gts.items():
        if gt is None:
            continue
        for mname, M in maps.items():
            if M.shape != gt.shape:
                continue
            try:
                row[f"auc_{mname}_{gt_name}"] = float(
                    method.compute_attribution_score(M, gt))
            except Exception as e:
                row[f"auc_{mname}_{gt_name}"] = float("nan")
            for vname, v in auroc_variants(M, gt).items():
                row[f"auroc_{vname}_{mname}_{gt_name}"] = v

    # heatmaps on a SHARED colour scale across arms is handled in the summary plot;
    # here we just dump the per-arm figure for quick inspection.
    for mname, M in maps.items():
        plt.figure(figsize=(9, 3.2))
        plt.matshow(M, aspect="auto", fignum=0)
        plt.colorbar(); plt.title(f"{arm} (seed {seed}) — {mname}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"map_{arm}_s{seed}_{mname}.png"), dpi=140)
        plt.close()
    return row


# ----------------------------------------------------------------------------- MAIN
def main():
    _, neural, position = get_data()
    print(f"neural {tuple(neural.shape)}  position {tuple(position.shape)}  "
          f"std={float(neural.std()):.4f}  range=[{float(neural.min())},{float(neural.max())}]")

    gt_nb, _ = build_ground_truth(neural.shape[1], "notebook")
    gt_md, _ = build_ground_truth(neural.shape[1], "model")
    gts = {"gtNB": gt_nb, "gtMODEL": gt_md}

    rows = []
    for seed in SEEDS:
        for arm in ARMS_TO_RUN:
            solver, meta = train_arm(arm, seed, neural, position)
            row = evaluate(arm, seed, solver, neural, position, gts, meta)
            rows.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in row.items()}, indent=1))
            import pandas as pd
            pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)

    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n================ SUMMARY ================")
    cols = [c for c in df.columns
            if c.startswith(("auc_", "auroc_global_", "sv_sv_ratio",
                             "sp_participation", "R2_", "KNN_"))]
    print(df.groupby("arm")[cols].mean().T.to_string())

    # shared-colourbar comparison of the jf maps (the honest version of the plot)
    fig, axes = plt.subplots(len(ARMS_TO_RUN), 1,
                             figsize=(10, 2.6 * len(ARMS_TO_RUN)), squeeze=False)
    mats = {a: np.load(os.path.join(OUT_DIR, f"maps_{a}_s{SEEDS[0]}.npz"))["jf"]
            for a in ARMS_TO_RUN}
    vmax = max(m.max() for m in mats.values())
    for ax, a in zip(axes[:, 0], ARMS_TO_RUN):
        im = ax.imshow(mats[a], aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(f"{a}  (shared scale)")
    fig.colorbar(im, ax=axes[:, 0].tolist())
    plt.savefig(os.path.join(OUT_DIR, "jf_shared_scale.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

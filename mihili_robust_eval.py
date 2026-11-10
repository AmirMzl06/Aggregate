"""Robustness of the saved Exp17 / Exp18 Mihili CO models, without retraining any encoder.

Exp17/18 reported clean cross-day R² only. This script attacks every saved model end to end
(encoder -> GRU decoder -> R²) with two per-bin threat models, the conventions used in training:

* ``iso``:   ``||delta_t||_2 <= eps`` in every 50-ms bin (the geometry of standard PGD).
* ``noise``: ``||Sigma^(-1/2) delta_t||_2 <= eps`` in every bin (the noise-aware geometry), with
  ``Sigma`` = Ledoit-Wolf covariance of Day-0 training bin differences ``x[t+1] - x[t]``, times
  ``--diff-scale`` (0.5: the covariance of the noise if it is white in time).

``Sigma`` comes from Day-0 training data only and is the same for every arm, so every model is
measured with the same ruler. Per geometry the script reports ``eps_half``: the smallest budget
at which R² falls to half of the model's own clean R², on the Day-0 holdout (``D0``) and on the
mean over the target days 1..N-1 (``X``). This is the same metric as on Perich (ε½). It also
reports the same quantity under random perturbations of the same size.

Decoders: monkey_gru keeps per-session decoders in a plain dict, so ``checkpoint.pt`` normally
holds only the encoder. Unless the checkpoint contains decoder weights, the script refits the
decoder with the original protocol on Day-0 training embeddings:
- GRU 512x2;
- adversarial decoder training at eps 0.1;
- early stopping;
- the run's seed.
It caches the decoder in ``<out>/decoders/`` and prints the refit clean R² next to the reported one.

Server usage. SRC is the Exp17 runner source, the folder with ``models.py``, ``data.py`` and
``utils/``.

    python mihili_robust_eval.py --inspect
    python mihili_robust_eval.py --source-dir SRC --include 's42-(pgd|noise-0p1)$'     # pilot
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i python mihili_robust_eval.py --source-dir SRC --shard $i/4 &
    done; wait
    python mihili_robust_eval.py --summarize

Exp18 (raw counts): add ``--no-zscore --runs-root <EXP18 runs> --out EXP18_ROBUST``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DEFAULT_RUNS = ("/data/hossein/mm_project/EXP17_MIHILI_CO/runs",
                "/data/hossein/mm_project/EXP17_FINE_SWEEP/runs")
DEFAULT_DATA_DIR = "/data/hossein/data"
DATASET = "Mihili_CO_2014_raw"
GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0)
GEOMETRIES = ("iso", "noise")
DEC_WINDOW = 256  # MonkeyDecoder window: R² is scored on non-overlapping windows of this length
TEST_FRACTION = 0.2  # last 20 % of every day (train_test_split(shuffle=False))
LOW_VAR = 1e-3  # channels with a smaller training std are zeroed (monkey_gru z-scoring)
TRAIN_DAY = 0
R2_FLOOR = -10.0  # stored R² is clipped here; non-finite predictions count as the floor
UNDEFINED = "clean R2 <= 0"  # eps_half status when the model cannot decode even without attack
ENC_PREFIX = "monkey_encoder."
SESSION_RE = re.compile(r"^monkey_encoder\.cebra\.session_layer\.session_dict\.ds_(.+)\.weight$")
DEC_ANCHOR = "gru.weight_ih_l0"
META_RE = re.compile(r"eps|adv|noise|auto|geom|mode|arm|method|zscore|seed|sigma|cov", re.I)


# ----------------------------------------------------------------------------- runs


@dataclass
class Run:
    name: str
    dir: Path
    path: Path
    seed: int | None
    family: str
    eps: float | None
    arm: str


def parse_run_name(name: str):
    """``exp17-co-s42-noise-0p1`` -> seed 42, family ``noise``, eps 0.1, arm ``noise-0.1``."""
    seed_re = re.compile(r"(?:^|[-_])s(\d+)(?=[-_]|$)")
    match = seed_re.search(name)
    seed = int(match.group(1)) if match else None
    tokens = [t for t in re.split(r"[-_]", seed_re.sub("", name)) if t]
    tokens = [t for t in tokens
              if not re.fullmatch(r"exp\d+\w*|co|rt|mihili|fs|fine|sweep|finesweep", t, re.I)]
    eps, family = None, []
    for token in tokens:
        number = re.fullmatch(r"(\d+)p(\d+)", token)
        if number:
            eps = float(f"{number.group(1)}.{number.group(2)}")
        elif re.fullmatch(r"\d+(\.\d+)?", token):
            eps = float(token)
        else:
            family.append(token.lower())
    family = "-".join(family) or "unknown"
    arm = family if eps is None else f"{family}-{eps:g}"
    return seed, family, eps, arm


def find_runs(roots, include=None, exclude=None):
    runs, seen = [], {}
    for root in roots:
        if not root.is_dir():
            print(f"[runs] missing folder {root}")
            continue
        for path in sorted(root.glob("*/checkpoint.pt")):
            name = path.parent.name
            if include and not re.search(include, name):
                continue
            if exclude and re.search(exclude, name):
                continue
            if name in seen:
                print(f"[runs] {name} appears twice; keeping {seen[name]}")
                continue
            seen[name] = path.parent
            seed, family, eps, arm = parse_run_name(name)
            runs.append(Run(name, path.parent, path, seed, family, eps, arm))
    return runs


def _json_files(run_dir):
    paths = [run_dir / "metrics.json"]
    paths += sorted(p for p in run_dir.glob("*.json") if p.name != "metrics.json")
    out = []
    for path in paths:
        if path.exists():
            try:
                out.append((path.name, json.loads(path.read_text())))
            except (OSError, ValueError):
                pass
    return out


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        children = obj.values()
    elif isinstance(obj, list):
        children = obj
    else:
        return None
    for child in children:
        found = _find_key(child, key)
        if found is not None:
            return found
    return None


def find_reported(run_dir):
    """Clean R² the training run reported (monkey_gru keys r2_holdout / r2_other_days)."""
    for name, obj in _json_files(run_dir):
        other = _find_key(obj, "r2_other_days")
        if isinstance(other, dict) and other:
            days = {int(k): float(v) for k, v in other.items() if v is not None}
            hold = _find_key(obj, "r2_holdout")
            return {"file": name, "D0": None if hold is None else float(hold), "days": days,
                    "X": float(np.mean(list(days.values())))}
    return None


def run_metadata(run_dir):
    meta = {}
    for _, obj in _json_files(run_dir):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if META_RE.search(key) and (value is None or isinstance(value, (str, int, float, bool))):
                    meta.setdefault(key, value)
    return meta


# ----------------------------------------------------------------------------- checkpoints


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only
        return torch.load(path, map_location="cpu")


def load_checkpoint(path):
    ckpt = _torch_load(path)
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        state, ckpt = ckpt, {"model": ckpt}
    else:
        raise ValueError(f"{path}: no 'model' state dict")
    state = {("monkey_encoder.cebra." + k[len("encoder."):] if k.startswith("encoder.") else k): v
             for k, v in state.items()}
    return ckpt, state


def session_info(ckpt, state, wanted=None):
    keys = [m.group(1) for m in (SESSION_RE.match(k) for k in state) if m]
    if wanted:
        key = wanted
    else:
        listed = [k for k in (ckpt.get("datasets") or []) if k in keys]
        key = listed[0] if listed else (keys[0] if len(keys) == 1 else None)
    if key not in keys:
        raise ValueError(f"session key {key!r} not among {keys}; pass --session-key")
    w_in = state[f"monkey_encoder.cebra.session_layer.session_dict.ds_{key}.weight"]
    w_out = state[f"monkey_encoder.cebra.last_layer_multi.session_dict.ds_{key}.weight"]
    smooth = ckpt.get("use_smooth")
    if isinstance(smooth, dict) and key in smooth:
        use_smooth = bool(smooth[key])
    else:
        use_smooth = any(k.startswith(f"monkey_encoder.cebra.smoothers.ds_{key}.") for k in state)
    labels = ckpt.get("n_labels")
    n_labels = int(labels[key]) if isinstance(labels, dict) and key in labels else None
    return {"key": key, "neurons": int(w_in.shape[1]), "hidden": int(w_in.shape[0]),
            "out": int(w_out.shape[0]), "use_smooth": use_smooth, "n_labels": n_labels}


def find_decoder_state(ckpt, state, key):
    """Decoder weights saved with the encoder, if any: (state with gru./linear. keys, where)."""

    def anchors(d):
        return [k for k in d if isinstance(k, str) and k.endswith(DEC_ANCHOR)]

    def strip(d):
        found = anchors(d)
        anchor = next((a for a in found if key in a), found[0])
        prefix = anchor[: -len(DEC_ANCHOR)]
        return {k[len(prefix):]: v for k, v in d.items()
                if isinstance(k, str) and k.startswith(prefix) and torch.is_tensor(v)}

    rest = {k: v for k, v in state.items() if not k.startswith(ENC_PREFIX)}
    if anchors(rest):
        return strip(rest), "checkpoint:model"
    for name, obj in ckpt.items():
        if name == "model" or not isinstance(obj, dict):
            continue
        if anchors(obj):
            return strip(obj), f"checkpoint:{name}"
        for sub_name, sub in obj.items():
            if isinstance(sub, dict) and anchors(sub) and (sub_name == key or len(obj) == 1):
                return strip(sub), f"checkpoint:{name}.{sub_name}"
    return None, None


def decoder_hparams(dstate):
    return {"hidden": int(dstate["gru.weight_hh_l0"].shape[1]),
            "layers": sum(1 for k in dstate if re.fullmatch(r"gru\.weight_ih_l\d+", k)),
            "bidir": any(k.endswith("_reverse") for k in dstate),
            "n_labels": int(dstate["linear.weight"].shape[0]),
            "ceb_out": int(dstate["gru.weight_ih_l0"].shape[1])}


# ----------------------------------------------------------------------------- runner source


def _optional(module, attr):
    try:
        return getattr(importlib.import_module(module), attr)
    except Exception as exc:  # noqa: BLE001 - any import problem means "not available"
        print(f"[source] {module}.{attr} not available ({type(exc).__name__}: {exc})")
        return None


class Source:
    """The monkey_gru-derived runner code the checkpoints were trained with."""

    def __init__(self, source_dir, workdir=None):
        self.dir = Path(source_dir).resolve()
        if not (self.dir / "models.py").exists():
            raise SystemExit(f"{self.dir} has no models.py: --source-dir must be the runner folder")
        sys.path.insert(0, str(self.dir))
        os.chdir(Path(workdir).resolve() if workdir else self.dir)  # loader/xds use relative paths
        models = importlib.import_module("models")
        self.Multi = models.MonkeyEncoderDecoderMulti
        self.Decoder = models.MonkeyDecoder
        self.train_session_decoder = _optional("embedding_decoder", "train_session_decoder")
        self.load_day_split = _optional("data", "load_day_split")

    def loader(self, data_dir=None):
        dataset_loader = importlib.import_module("utils.dataset_loader")
        if data_dir is None:
            data_dir = _optional("utils.constants", "DATA_DIR") or DEFAULT_DATA_DIR
        return dataset_loader.DatasetLoader(data_root_dir=str(data_dir))


# ----------------------------------------------------------------------------- data


@dataclass
class Day:
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor


def zscore_pair(train, test):
    """monkey_gru ``_zscore_spikes`` with statistics of ``train``."""
    mean = train.mean(0, keepdim=True)
    std = train.std(0, keepdim=True, unbiased=False)
    low = (std < LOW_VAR).squeeze(0)
    out = []
    for x in (train, test):
        x = (x - mean) / (std + 1e-8)
        x[:, low] = 0.0
        out.append(x)
    return out


def split_day(src, loader, dataset, day, zscore, xds_smooth):
    if src.load_day_split is not None:
        try:
            tr_x, te_x, tr_y, te_y = src.load_day_split(loader, dataset, day,
                                                        xds_smooth=xds_smooth, zscore=zscore)
            return Day(train_x=torch.as_tensor(tr_x).float(), train_y=torch.as_tensor(tr_y).float(),
                       test_x=torch.as_tensor(te_x).float(), test_y=torch.as_tensor(te_y).float())
        except TypeError as exc:
            print(f"[data] load_day_split signature differs ({exc}); splitting here")
    from sklearn.model_selection import train_test_split
    spikes, labels = loader.load_dataset_day(day, dataset, xds_smooth=xds_smooth)
    spikes = torch.as_tensor(np.asarray(spikes)).float()
    labels = torch.as_tensor(np.asarray(labels)).float()
    tr_x, te_x, tr_y, te_y = train_test_split(spikes, labels, test_size=TEST_FRACTION, shuffle=False)
    if zscore:
        tr_x, te_x = zscore_pair(tr_x, te_x)
    return Day(train_x=tr_x, train_y=tr_y, test_x=te_x, test_y=te_y)


def load_days(src, args):
    loader = src.loader(args.data_dir)
    n_days = args.num_days
    if n_days is None:
        try:
            n_days = int(loader.get_dataset_num_days(args.dataset))
        except Exception:  # noqa: BLE001
            n_days = 11
    days = {}
    for day in range(n_days):
        days[day] = split_day(src, loader, args.dataset, day, not args.no_zscore, args.xds_smooth)
        d = days[day]
        print(f"[data] day {day:2d}: train {tuple(d.train_x.shape)}  test {tuple(d.test_x.shape)}"
              f"  labels {d.train_y.shape[1]}")
    return days


def reference_operators(day0, args, device):
    """{"iso": None, "noise": Sigma^(1/2)} and a description of Sigma."""
    if args.sigma_file:
        cov = np.asarray(np.load(args.sigma_file), dtype=np.float64)
        origin = f"file {args.sigma_file}"
    else:
        from sklearn.covariance import LedoitWolf
        diffs = np.diff(day0.train_x.numpy().astype(np.float64), axis=0)
        fit = LedoitWolf().fit(diffs)
        cov = fit.covariance_ * args.diff_scale
        origin = f"Ledoit-Wolf(day-0 train diffs) x {args.diff_scale:g}, shrinkage {fit.shrinkage_:.3f}"
    cov = 0.5 * (cov + cov.T)
    eig, vec = np.linalg.eigh(cov)
    eig = np.clip(eig, eig.max() * 1e-10, None)
    root = (vec * np.sqrt(eig)) @ vec.T
    info = {"origin": origin, "dim": int(cov.shape[0]), "mean_var": float(eig.mean()),
            "participation": float(eig.sum() ** 2 / (eig ** 2).sum()),
            "condition": float(eig.max() / eig.min()), "zscore": not args.no_zscore}
    print(f"[sigma] {origin}: mean variance {info['mean_var']:.4f}, participation ratio "
          f"{info['participation']:.1f} of {info['dim']}, condition {info['condition']:.3g}")
    np.save(args.out / "reference_sigma.npy", cov)
    return {"iso": None, "noise": torch.as_tensor(root, dtype=torch.float32, device=device)}, info


# ----------------------------------------------------------------------------- models


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(src, info, state, device):
    model = src.Multi(ceb_hidden=info["hidden"], ceb_out=info["out"])
    model.add_session(info["key"], info["neurons"], info["n_labels"], use_smooth=info["use_smooth"])
    encoder = {k: v for k, v in state.items() if k.startswith(ENC_PREFIX)}
    result = torch.nn.Module.load_state_dict(model, encoder, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"encoder weights do not match the model: missing {result.missing_keys[:5]}, "
                           f"unexpected {result.unexpected_keys[:5]}")
    return model.to(device).eval()


def decoder_from_state(src, dstate, device):
    hp = decoder_hparams(dstate)
    decoder = src.Decoder(ceb_out=hp["ceb_out"], hidden=hp["hidden"], layers=hp["layers"],
                          dropout=0.4, bidir=hp["bidir"], n_labels=hp["n_labels"],
                          device=device, verbose=False)
    decoder.load_state_dict(dstate, strict=True)
    return decoder.to(device).eval()


def refit_decoder(src, model, info, day0, seed, args, device):
    """The original post-hoc decoder fit (embedding_decoder.train_session_decoder)."""
    kwargs = {"seed": seed, "use_val_checkpoint": not args.no_decoder_val,
              "adv_dec": not args.no_adv_dec, "adv_dec_eps": args.adv_dec_eps}
    fit = src.train_session_decoder
    if fit is not None and set(kwargs) | {"big_dec"} <= set(inspect.signature(fit).parameters):
        seed_all(seed)
        return fit(model, info["key"], day0.train_x, day0.train_y, info["n_labels"], device,
                   args.decoder_steps, big_dec=args.big_dec, **kwargs)
    print("[decoder] train_session_decoder unavailable or different; using MonkeyDecoder.fit")
    with torch.no_grad():
        emb = model.transform(day0.train_x.to(device), info["key"]).float()
    hp = ({"hidden": 1024, "layers": 5, "bidir": True} if args.big_dec
          else {"hidden": 512, "layers": 2, "bidir": False})
    seed_all(seed)
    decoder = src.Decoder(ceb_out=emb.shape[1], n_labels=info["n_labels"], dropout=0.4,
                          n_train_steps=args.decoder_steps, device=device, verbose=False, **hp)
    decoder.to(device).fit(emb, day0.train_y.to(device), **kwargs)
    return decoder


def get_decoder(src, model, ckpt, state, info, day0, run, args, device):
    dstate, where = find_decoder_state(ckpt, state, info["key"])
    if dstate is not None:
        return decoder_from_state(src, dstate, device), {"source": where}
    cache = args.out / "decoders" / f"{run.name}.pt"
    if cache.exists() and not args.refit_decoders:
        blob = _torch_load(cache)
        return decoder_from_state(src, blob["state"], device), {**blob["meta"], "source": "cache"}
    seed = run.seed if run.seed is not None else 42
    start = time.time()
    decoder = refit_decoder(src, model, info, day0, seed, args, device)
    meta = {"source": "refit", "seed": seed, "steps": args.decoder_steps,
            "val_checkpoint": not args.no_decoder_val, "adv_dec": not args.no_adv_dec,
            "adv_dec_eps": args.adv_dec_eps, "best_epoch": getattr(decoder, "best_epoch_", None),
            "best_val_r2": _finite(getattr(decoder, "best_val_r2_", None)),
            "minutes": round((time.time() - start) / 60, 2)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.detach().cpu() for k, v in decoder.state_dict().items()},
                "meta": meta}, cache)
    print(f"[decoder] refit in {meta['minutes']} min (best epoch {meta['best_epoch']})")
    return decoder.eval(), meta


class Pipeline:
    """Encoder trajectory -> non-overlapping decoder windows -> R² (as MonkeyDecoder.score).

    ``r2_many`` scores several recordings (Day-0 holdout and the target days) in one decoder
    call; their perturbations stay separate, so attacking them together is the same as
    attacking each one on its own.
    """

    def __init__(self, model, key, decoder, window=DEC_WINDOW):
        self.model, self.key, self.decoder, self.window = model, key, decoder, window

    def r2_many(self, xs, ys):
        windows, widths, counts = [], [], []
        for x in xs:
            emb = self.model.transform(x, self.key).float()
            width = min(self.window, emb.shape[0])
            count = emb.shape[0] // width
            win = emb[: count * width].reshape(count, width, emb.shape[1])
            if width < self.window:
                win = torch.nn.functional.pad(win, (0, 0, 0, self.window - width))
            windows.append(win)
            widths.append(width)
            counts.append(count)
        lengths = torch.tensor([w for w, c in zip(widths, counts) for _ in range(c)], dtype=torch.long)
        # cuDNN cannot backpropagate through a GRU in eval mode; the native kernel can
        with torch.backends.cudnn.flags(enabled=False):
            out = self.decoder(torch.cat(windows), lengths)
        scores, start = [], 0
        for y, width, count in zip(ys, widths, counts):
            pred = out[start: start + count, :width].reshape(count * width, -1)
            start += count
            y = y[: count * width]
            sse = ((pred - y) ** 2).sum(0)
            sst = ((y - y.mean(0)) ** 2).sum(0).clamp_min(1e-12)
            scores.append(1.0 - (sse / sst).mean())
        return torch.stack(scores)

    def r2(self, x, y):
        return self.r2_many([x], [y])[0]


# ----------------------------------------------------------------------------- attacks


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _clip_r2(value):
    value = float(value)
    return max(value, R2_FLOOR) if math.isfinite(value) else R2_FLOOR


def project(z, eps):
    """Per-bin projection onto ``||z_t||_2 <= eps``."""
    norm = z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return z * torch.clamp(eps / norm, max=1.0)


def delta(z, root):
    return z if root is None else z @ root  # root is symmetric


def random_ball(shape, eps, gen, device):
    u = torch.randn(shape, generator=gen, device=device)
    u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return u * torch.rand((shape[0], 1), generator=gen, device=device) * eps


def pgd_many(pipe, xs, ys, root, eps, steps, restarts, gen, warms=None):
    """Per recording, the lowest R² found by per-bin L2 PGD in z.

    ``delta_t = root z_t`` with ``||z_t|| <= eps``. ``warms`` are the perturbations found at a
    smaller budget. They are evaluated first, so curves along an increasing grid never go up.
    Returns (R² list, z list).
    """
    count = len(xs)
    starts = [] if warms is None else [[project(w.to(x.device), eps) for w, x in zip(warms, xs)]]
    for _ in range(max(restarts, int(warms is None))):
        starts.append([random_ball(x.shape, eps, gen, x.device) for x in xs])
    alpha = 2.5 * eps / max(steps, 1)
    best, best_z = [math.inf] * count, [None] * count
    for zs in starts:
        for k in range(steps + 1):
            last = k == steps
            zs = [z.detach().requires_grad_(not last) for z in zs]
            with torch.set_grad_enabled(not last):
                r2 = pipe.r2_many([x + delta(z, root) for x, z in zip(xs, zs)], ys)
            for i, value in enumerate(r2.detach().cpu().tolist()):
                value = _clip_r2(value)
                if best_z[i] is None or value < best[i]:
                    best[i], best_z[i] = value, zs[i].detach().clone()
            if last:
                break
            grads = torch.autograd.grad(r2.sum(), zs)
            with torch.no_grad():
                zs = [project(z - alpha * torch.nan_to_num(g)
                              / torch.nan_to_num(g).norm(dim=1, keepdim=True).clamp_min(1e-12), eps)
                      for z, g in zip(zs, grads)]
    return best, best_z


def random_many(pipe, xs, ys, root, eps, draws, gen):
    """Mean R² per recording under random perturbations with ``||z_t|| = eps`` in every bin."""
    total = np.zeros(len(xs))
    with torch.no_grad():
        for _ in range(draws):
            noisy = []
            for x in xs:
                u = torch.randn(x.shape, generator=gen, device=x.device)
                noisy.append(x + delta(u / u.norm(dim=1, keepdim=True).clamp_min(1e-12) * eps, root))
            total += np.array([_clip_r2(v) for v in pipe.r2_many(noisy, ys).cpu().tolist()])
    return (total / draws).tolist()


def half_from_curve(grid, curve, clean):
    """Log-interpolated budget where ``curve`` crosses ``clean / 2`` (None: not reached/undefined)."""
    if clean is None or not clean > 0:
        return None
    half = 0.5 * clean
    prev_eps, prev_r2 = None, clean
    for eps, r2 in zip(grid, curve):
        if r2 is None:  # grid stopped early: every recording was already far below half
            r2 = R2_FLOOR
        if r2 <= half:
            if prev_eps is None:
                return float(eps)
            t = (prev_r2 - half) / max(prev_r2 - r2, 1e-12)
            return float(math.exp(math.log(prev_eps) + t * (math.log(eps) - math.log(prev_eps))))
        prev_eps, prev_r2 = eps, r2
    return None


def half_entry(grid, curve, clean):
    """``half_from_curve`` with the status used for refined values."""
    if clean is None or not clean > 0:
        return {"value": None, "status": UNDEFINED}
    value = half_from_curve(grid, curve, clean)
    if value is None:
        return {"value": grid[-1], "status": f">{grid[-1]:g}"}
    if curve[0] is not None and curve[0] <= 0.5 * clean:
        return {"value": grid[0], "status": f"<{grid[0]:g}"}
    return {"value": value, "status": "ok"}


def attack_geometry(pipe, sets, targets, root, args, gen):
    """Adversarial and random R² curves on the grid, then refined eps_half for D0 and X."""
    names = list(sets)
    xs = [sets[n][0] for n in names]
    ys = [sets[n][1] for n in names]
    grid = list(args.grid)
    with torch.no_grad():
        clean = dict(zip(names, (_clip_r2(v) for v in pipe.r2_many(xs, ys).cpu().tolist())))
    adv = {n: [] for n in names}
    rnd = {n: [] for n in names}
    found = []  # per computed grid point: best z per recording (CPU), warm starts for refinement
    warms, stopped = None, False
    for eps in grid:
        if stopped:
            for n in names:
                adv[n].append(None)
                rnd[n].append(None)
            continue
        values, zs = pgd_many(pipe, xs, ys, root, eps, args.steps, args.restarts, gen, warms)
        noisy = (random_many(pipe, xs, ys, root, eps, args.random_draws, gen)
                 if args.random_draws else [None] * len(names))
        for n, value, noise_value in zip(names, values, noisy):
            adv[n].append(value)
            rnd[n].append(noise_value)
        warms = zs
        found.append([z.cpu() for z in zs])
        stopped = all(v < args.stop_r2 for v in values)

    def pooled(curves, group):
        out = []
        for i in range(len(grid)):
            values = [curves[n][i] for n in group]
            out.append(None if any(v is None for v in values) else float(np.mean(values)))
        return out

    def refine(group, curve, clean_value):
        if not clean_value > 0:
            return {"value": None, "status": UNDEFINED}
        half = 0.5 * clean_value
        idx = next((i for i, v in enumerate(curve) if v is None or v <= half), None)
        if idx is None:
            return {"value": grid[-1], "status": f">{grid[-1]:g}"}
        rows = [names.index(n) for n in group]
        gx, gy = [xs[i] for i in rows], [ys[i] for i in rows]

        def attack(eps, starts):
            values, zs = pgd_many(pipe, gx, gy, root, eps, args.steps, args.restarts, gen, starts)
            return float(np.mean(values)), zs

        hi = grid[idx]
        if idx == 0:
            lo = hi / 4.0
            value, starts = attack(lo, None)
            if value <= half:
                return {"value": lo, "status": f"<{lo:g}"}
        else:
            lo, starts = grid[idx - 1], [found[idx - 1][i] for i in rows]
        for _ in range(args.refine):
            mid = math.sqrt(lo * hi)
            value, zs = attack(mid, starts)
            if value <= half:
                hi = mid
            else:
                lo, starts = mid, zs
        return {"value": math.sqrt(lo * hi), "status": "ok"}

    clean_x = float(np.mean([clean[n] for n in targets]))
    adv_x, rnd_x = pooled(adv, targets), pooled(rnd, targets)
    result = {
        "clean": clean,
        "adv": adv,
        "rand": rnd,
        "adv_X": adv_x,
        "rand_X": rnd_x,
        "eps_half": {"D0": refine(["D0"], adv["D0"], clean["D0"]),
                     "X": refine(targets, adv_x, clean_x)},
        "eps_half_days": {n: half_from_curve(grid, adv[n], clean[n]) for n in targets},
    }
    if args.random_draws:
        result["rand_eps_half"] = {"D0": half_entry(grid, rnd["D0"], clean["D0"]),
                                   "X": half_entry(grid, rnd_x, clean_x)}
    return result


def stable_seed(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------------------- one run


def evaluate_run(run, src, days, roots, sigma_info, args, device):
    start = time.time()
    ckpt, state = load_checkpoint(run.path)
    info = session_info(ckpt, state, args.session_key)
    if info["n_labels"] is None:
        info["n_labels"] = int(days[TRAIN_DAY].train_y.shape[1])
    model = build_model(src, info, state, device)
    decoder, dec_meta = get_decoder(src, model, ckpt, state, info, days[TRAIN_DAY], run, args, device)
    pipe = Pipeline(model, info["key"], decoder)
    sets = {"D0": (days[TRAIN_DAY].test_x.to(device), days[TRAIN_DAY].test_y.to(device))}
    for day in sorted(days):
        if day != TRAIN_DAY:
            sets[f"D{day}"] = (days[day].test_x.to(device), days[day].test_y.to(device))
    targets = [n for n in sets if n != "D0"]

    with torch.no_grad():
        values = pipe.r2_many([v[0] for v in sets.values()], [v[1] for v in sets.values()])
        clean = dict(zip(sets, (_clip_r2(v) for v in values.cpu().tolist())))
        try:  # the pipeline must reproduce MonkeyDecoder.score on clean data
            emb = model.transform(sets["D0"][0], info["key"]).float()
            score = _finite(decoder.score(emb, sets["D0"][1], device))
        except Exception:  # noqa: BLE001
            score = None
    if score is not None and abs(score - clean["D0"]) > 1e-3:
        print(f"[warn] pipeline R² {clean['D0']:.4f} != MonkeyDecoder.score {score:.4f} on D0")
    clean_x = float(np.mean([clean[n] for n in targets]))
    reported = find_reported(run.dir)
    rep = (f"(reported D0 {reported['D0']:.4f}, D1-{len(targets)} {reported['X']:.4f})"
           if reported and reported["D0"] is not None else "(no reported R² found)")
    print(f"[clean] {run.name}: D0 {clean['D0']:.4f}, D1-{len(targets)} {clean_x:.4f} {rep}")

    result = {
        "name": run.name, "run_dir": str(run.dir), "arm": run.arm, "family": run.family,
        "eps_train": run.eps, "seed": run.seed, "session": info, "decoder": dec_meta,
        "meta": run_metadata(run.dir), "reported": reported, "sigma": sigma_info,
        "grid": list(args.grid), "clean": clean, "clean_X": clean_x, "score_check_D0": score,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "geometries": {},
    }
    for geom in GEOMETRIES:
        t0 = time.time()
        gen = torch.Generator(device=device)
        gen.manual_seed(stable_seed(run.name, geom))
        res = attack_geometry(pipe, sets, targets, roots[geom], args, gen)
        res["minutes"] = round((time.time() - t0) / 60, 2)
        result["geometries"][geom] = res
        eh = res["eps_half"]
        print(f"[{geom:5s}] eps_half D0 {_fmt_half(eh['D0'])}, D1-{len(targets)} {_fmt_half(eh['X'])}"
              f"  ({res['minutes']} min)")
    result["minutes"] = round((time.time() - start) / 60, 2)
    path = args.out / "runs" / f"{run.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(path)
    return result


def _fmt_half(entry):
    if entry is None or entry.get("value") is None:
        return "n/a" if entry is None else entry.get("status", "n/a")
    return f"{entry['value']:.3g}" if entry["status"] == "ok" else entry["status"]


# ----------------------------------------------------------------------------- summary


def _geomean(values):
    values = [v for v in values if v is not None and v > 0]
    return float(np.exp(np.mean(np.log(values)))) if values else None


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _num(value, digits=3):
    return "-" if value is None else f"{value:.{digits}g}" if abs(value) >= 1e-3 else f"{value:.1e}"


def run_rows(results):
    rows = []
    for r in results:
        rep = r.get("reported") or {}
        row = {"name": r["name"], "arm": r["arm"], "family": r["family"], "eps_train": r["eps_train"],
               "seed": r["seed"], "decoder": (r.get("decoder") or {}).get("source"),
               "clean_D0": r["clean"]["D0"], "clean_X": r["clean_X"],
               "reported_D0": rep.get("D0"), "reported_X": rep.get("X")}
        for geom in GEOMETRIES:
            g = r["geometries"].get(geom)
            if not g:
                continue
            for where in ("D0", "X"):
                entry = g["eps_half"][where]
                row[f"{geom}_half_{where}"] = entry["value"]
                row[f"{geom}_status_{where}"] = entry["status"]
                rand = (g.get("rand_eps_half") or {}).get(where)
                row[f"{geom}_rand_half_{where}"] = rand.get("value") if isinstance(rand, dict) else rand
                row[f"{geom}_rand_status_{where}"] = rand.get("status") if isinstance(rand, dict) else None
            days = [v for v in g["eps_half_days"].values() if v is not None]
            row[f"{geom}_half_day_median"] = float(np.median(days)) if days else None
        rows.append(row)
    return rows


def curve_rows(results):
    rows = []
    for r in results:
        for geom, g in r["geometries"].items():
            for kind in ("adv", "rand"):
                curves = dict(g[kind])
                curves["X"] = g[f"{kind}_X"]
                for where, curve in curves.items():
                    clean = r["clean_X"] if where == "X" else r["clean"][where]
                    for eps, value in [(0.0, clean)] + list(zip(r["grid"], curve)):
                        rows.append({"name": r["name"], "arm": r["arm"], "seed": r["seed"],
                                     "geometry": geom, "kind": kind, "set": where, "eps": eps,
                                     "r2": value})
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _arm_order(arm_rows):
    return sorted(arm_rows, key=lambda a: (a["family"], -1 if a["eps_train"] is None else a["eps_train"]))


def summarize(out, reference="pgd"):
    files = sorted((out / "runs").glob("*.json"))
    results = [json.loads(f.read_text()) for f in files]
    if not results:
        print(f"[summary] no results in {out / 'runs'}")
        return
    rows = run_rows(results)
    _write_csv(out / "summary_runs.csv", rows)
    _write_csv(out / "curves.csv", curve_rows(results))

    metrics = [f"{g}_half_{w}" for g in GEOMETRIES for w in ("D0", "X")]
    rand_metrics = [f"{g}_rand_half_X" for g in GEOMETRIES]
    by_arm = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    arms = []
    for arm, group in by_arm.items():
        entry = {"arm": arm, "family": group[0]["family"], "eps_train": group[0]["eps_train"],
                 "n": len(group), "seeds": sorted(r["seed"] for r in group if r["seed"] is not None),
                 "clean_D0": _mean([r["clean_D0"] for r in group]),
                 "clean_X": _mean([r["clean_X"] for r in group]),
                 "reported_X": _mean([r["reported_X"] for r in group])}
        for m in metrics + rand_metrics:
            entry[m] = _geomean([r.get(m) for r in group])
            geom, _, where = m.partition("_half_")
            statuses = [r.get(f"{geom}_status_{where}") for r in group]
            entry[f"{m}_undefined"] = sum(s == UNDEFINED for s in statuses)
            entry[f"{m}_censored"] = sum(s not in ("ok", None, UNDEFINED) for s in statuses)
        arms.append(entry)
    arms = _arm_order(arms)

    refs = [a for a in arms if reference in a["family"]]
    ref = refs[0] if refs else None
    if ref:
        ref_rows = {r["seed"]: r for r in by_arm[ref["arm"]]}
        for entry in arms:
            for m in metrics:
                ratios = [r[m] / ref_rows[r["seed"]][m] for r in by_arm[entry["arm"]]
                          if r["seed"] in ref_rows and r.get(m) and ref_rows[r["seed"]].get(m)]
                entry[f"ratio_{m}"] = _geomean(ratios)
            prefs = []
            for r in by_arm[entry["arm"]]:
                base = ref_rows.get(r["seed"])
                if base and all(v for v in (r.get("noise_half_X"), r.get("iso_half_X"),
                                            base.get("noise_half_X"), base.get("iso_half_X"))):
                    prefs.append((r["noise_half_X"] / r["iso_half_X"])
                                 / (base["noise_half_X"] / base["iso_half_X"]))
            entry["noise_over_iso_vs_ref"] = _geomean(prefs)
    for m in ("noise_half_X", "iso_half_X"):
        for entry in arms:
            entry[f"frontier_{m}"] = bool(entry["clean_X"] is not None and entry[m] is not None and not any(
                other is not entry and other["clean_X"] is not None and other[m] is not None
                and other["clean_X"] >= entry["clean_X"] and other[m] >= entry[m]
                and (other["clean_X"] > entry["clean_X"] or other[m] > entry[m]) for other in arms))
    _write_csv(out / "summary_arms.csv", arms)

    n_days = len(results[0]["clean"]) - 1
    print(f"\n=== Robustness of {len(rows)} runs ({len(arms)} arms); X = mean over D1-{n_days} ===")
    print("eps_half: budget (per bin) where R² falls to half of the model's own clean R²; "
          "geometric mean over seeds; '*' = some seeds censored at the grid edge (value = edge); "
          "n/a = clean R² <= 0; rand = random perturbations of the same size.")
    head = (f"{'arm':22s} {'n':>2s} {'clean D0':>8s} {'clean X':>8s} {'(rep X)':>8s} | "
            f"{'noise D0':>8s} {'noise X':>8s} | {'iso D0':>8s} {'iso X':>8s} | "
            f"{'rand nX':>8s} {'rand iX':>8s}")
    print(head)
    print("-" * len(head))
    for a in arms:
        def cell(m):
            if a.get(m) is None:
                return "n/a" if a.get(f"{m}_undefined") else "-"
            return _num(a[m]) + ("*" if a.get(f"{m}_censored") else "")
        print(f"{a['arm']:22s} {a['n']:2d} {_num(a['clean_D0']):>8s} {_num(a['clean_X']):>8s} "
              f"{_num(a['reported_X']):>8s} | {cell('noise_half_D0'):>8s} {cell('noise_half_X'):>8s} | "
              f"{cell('iso_half_D0'):>8s} {cell('iso_half_X'):>8s} | "
              f"{cell('noise_rand_half_X'):>8s} {cell('iso_rand_half_X'):>8s}")
    if ref:
        print(f"\nPaired ratio vs {ref['arm']} (same seed, geometric mean); "
              "pref = (noise/iso of the arm) / (noise/iso of the reference):")
        print(f"{'arm':22s} {'noise D0':>9s} {'noise X':>9s} {'iso D0':>9s} {'iso X':>9s} {'pref':>7s}")
        for a in arms:
            cells = [f"x{_num(a.get('ratio_' + m))}" if a.get("ratio_" + m) else "-" for m in
                     ("noise_half_D0", "noise_half_X", "iso_half_D0", "iso_half_X")]
            pref = a.get("noise_over_iso_vs_ref")
            print(f"{a['arm']:22s} " + " ".join(f"{c:>9s}" for c in cells)
                  + f" {('x' + _num(pref)) if pref else '-':>7s}")
    else:
        print(f"\n(no arm family contains {reference!r}; ratios skipped)")
    for m, label in (("noise_half_X", "noise-shaped"), ("iso_half_X", "isotropic")):
        front = [a["arm"] for a in arms if a.get(f"frontier_{m}")]
        print(f"Pareto frontier (clean X vs {label} eps_half X): {', '.join(front) or '-'}")
    sources = Counter(r["decoder"] for r in rows)
    print(f"Decoders: {dict(sources)}.  Files: {out / 'summary_runs.csv'}, {out / 'summary_arms.csv'}, "
          f"{out / 'curves.csv'}")


# ----------------------------------------------------------------------------- inspect / main


def inspect_runs(runs, limit):
    print(f"{len(runs)} runs found:")
    for run in runs:
        print(f"  {run.name:40s} seed {run.seed}  arm {run.arm}")
    for run in runs[:limit]:
        print(f"\n--- {run.name} ({run.path})")
        ckpt, state = load_checkpoint(run.path)
        print("  top-level:", {k: type(v).__name__ for k, v in ckpt.items()})
        print("  scalars:", {k: v for k, v in ckpt.items() if isinstance(v, (str, int, float, list))})
        prefixes = Counter(".".join(k.split(".")[:3]) for k in state)
        print(f"  {len(state)} tensors:", dict(prefixes))
        try:
            print("  session:", session_info(ckpt, state))
        except Exception as exc:  # noqa: BLE001
            print("  session: ?", exc)
        dstate, where = find_decoder_state(ckpt, state, (ckpt.get("datasets") or [""])[0])
        print("  decoder:", f"{where} {decoder_hparams(dstate)}" if dstate else
              "not in checkpoint -> will be refit with the original protocol")
        print("  reported:", find_reported(run.dir))
        print("  metadata:", run_metadata(run.dir))
        print("  files:", sorted(p.name for p in run.dir.iterdir())[:25])


def parse_shard(text):
    k, n = (int(v) for v in text.split("/"))
    if not 0 <= k < n:
        raise SystemExit(f"bad --shard {text}")
    return k, n


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-dir", help="Exp17 runner source (models.py, data.py, utils/)")
    p.add_argument("--workdir", help="working directory for the data loader (default: --source-dir)")
    p.add_argument("--runs-root", nargs="+", default=list(DEFAULT_RUNS))
    p.add_argument("--out", default="EXP17_ROBUST")
    p.add_argument("--data-dir", help="dataset root (default: utils.constants.DATA_DIR of the source)")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--num-days", type=int)
    p.add_argument("--session-key")
    p.add_argument("--no-zscore", action="store_true", help="Exp18: raw counts, no per-day z-score")
    p.add_argument("--xds-smooth", action="store_true", help="xds smoothing (Exp17/18 used none)")
    p.add_argument("--include", help="regex on run names")
    p.add_argument("--exclude", help="regex on run names")
    p.add_argument("--shard", default="0/1", help="k/n: evaluate every n-th run starting at k")
    p.add_argument("--grid", type=float, nargs="+", default=list(GRID), help="per-bin budgets")
    p.add_argument("--steps", type=int, default=20, help="PGD steps per start")
    p.add_argument("--restarts", type=int, default=1, help="random starts (plus a warm start)")
    p.add_argument("--refine", type=int, default=4, help="bisection steps for eps_half")
    p.add_argument("--random-draws", type=int, default=2, help="0 disables random-noise curves")
    p.add_argument("--stop-r2", type=float, default=-0.5,
                   help="stop the grid once every set is below this R²")
    p.add_argument("--diff-scale", type=float, default=0.5, help="Sigma = scale x Cov(x[t+1]-x[t])")
    p.add_argument("--sigma-file", help=".npy covariance to use instead (e.g. the training one)")
    p.add_argument("--decoder-steps", type=int, default=30000)
    p.add_argument("--adv-dec-eps", type=float, default=0.1)
    p.add_argument("--no-adv-dec", action="store_true")
    p.add_argument("--no-decoder-val", action="store_true")
    p.add_argument("--big-dec", action="store_true")
    p.add_argument("--refit-decoders", action="store_true", help="ignore cached decoders")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--reference-arm", default="pgd", help="family substring of the reference arm")
    p.add_argument("--inspect", action="store_true", help="describe checkpoints and exit")
    p.add_argument("--inspect-limit", type=int, default=2)
    p.add_argument("--summarize", action="store_true", help="only aggregate existing results")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.out = Path(args.out).resolve()
    for name in ("source_dir", "workdir", "data_dir", "sigma_file"):  # Source() changes the cwd
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    args.grid = sorted(set(args.grid))
    roots = [Path(r).resolve() for r in args.runs_root]
    if args.summarize:
        summarize(args.out, args.reference_arm)
        return
    runs = find_runs(roots, args.include, args.exclude)
    if args.inspect:
        inspect_runs(runs, args.inspect_limit)
        return
    if not runs:
        raise SystemExit(f"no */checkpoint.pt under {[str(r) for r in roots]}")
    if not args.source_dir:
        raise SystemExit("--source-dir is required (the runner folder with models.py)")
    k, n = parse_shard(args.shard)
    todo = [r for r in runs[k::n] if args.overwrite or not (args.out / "runs" / f"{r.name}.json").exists()]
    print(f"[runs] {len(runs)} found, shard {k}/{n}: {len(todo)} to evaluate -> {args.out}")
    if not todo:
        summarize(args.out, args.reference_arm)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    src = Source(args.source_dir, args.workdir)
    device = torch.device(args.device)
    days = load_days(src, args)
    operators, sigma_info = reference_operators(days[TRAIN_DAY], args, device)
    failed = []
    for i, run in enumerate(todo, 1):
        print(f"\n=== [{i}/{len(todo)}] {run.name} (arm {run.arm}, seed {run.seed})")
        try:
            evaluate_run(run, src, days, operators, sigma_info, args, device)
        except Exception:  # noqa: BLE001 - keep going, record the failure
            failed.append(run.name)
            message = traceback.format_exc()
            print(message)
            (args.out / "failed").mkdir(exist_ok=True)
            (args.out / "failed" / f"{run.name}.txt").write_text(message)
    if failed:
        print(f"[runs] failed: {failed} (tracebacks in {args.out / 'failed'})")
    summarize(args.out, args.reference_arm)


if __name__ == "__main__":
    main()

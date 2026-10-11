"""Train the nested-subset information-ranking encoder on one Perich session, then evaluate it.

Training (no behavior labels):
  every step: a random reference population B (|B| = min(b_max, round(b_frac * N))), A = the rest;
  G random time points t; per sample a random order of A whose prefixes give nested real subsets at
  the levels |A| * (1, 1/2, 1/4, 1/8); the other neurons of A become decoys (each one's own activity
  from a distant random time), so every input has |A| tokens with unchanged per-neuron statistics
  (the "resize" of the paper: the encoder cannot read the level from the input size).
      z_l = f(level-l input),   M_l = I_G(z_l ; activity of B at t)   (Gaussian MI on the batch)
      loss = -mean_l M_l + rank_weight * mean_{i<j} max(0, M_j - M_i + margin)

Evaluation (labels only here): ridge probe from z(t) to behavior, alpha picked on the last 20% of
train, R^2 on valid. Reported for the random-init encoder (step 0), during training and at the end,
together with the ridge baseline on the raw (smoothed) neurons.

    python train_rank_ssl.py --session C-CO12 --out_dir rank_runs/C-CO12/main
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from rank_ssl_model import NeuronSetEncoder, effective_rank, gaussian_mi, rank_loss

PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")


def get_parser():
    p = argparse.ArgumentParser(description="Nested-subset information ranking (self-supervised)")
    # data
    p.add_argument("--session", type=str, default="C-CO12")
    p.add_argument("--npz", type=str, default="", help="overrides --session")
    p.add_argument("--data_dir", type=str, default=str(PERICH_DATA_DIR))
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--min_valid_std", type=float, default=0.2,
                   help="drop neurons whose valid std is below this fraction of their train std")
    p.add_argument("--b_sigma", type=float, default=2.0, help="smoothing (bins) of the reference activity")
    p.add_argument("--input_sigma", type=float, default=0.0, help="smoothing (bins) of the encoder input")
    p.add_argument("--detrend_sigma", type=float, default=0.0,
                   help="if > 0, subtract a Gaussian-smoothed (this many bins) slow component everywhere")
    # nested subsets / reference population
    p.add_argument("--window", type=int, default=36, help="bins per encoder window (36 = CEBRA offset36)")
    p.add_argument("--b_max", type=int, default=32)
    p.add_argument("--b_frac", type=float, default=1 / 3)
    p.add_argument("--fixed_b", action="store_true", help="one B for the whole training (Exp. 1 control)")
    p.add_argument("--levels", type=str, default="1,0.5,0.25,0.125", help="fractions of A, largest first")
    p.add_argument("--k_min", type=int, default=4)
    p.add_argument("--decoy", type=str, default="time", choices=["time", "zero", "drop"],
                   help="time: activity of the same neuron from another time (default); zero: constant "
                        "input (size visible); drop: smaller sets, no decoys (the original proposal)")
    p.add_argument("--decoy_gap", type=int, default=360, help="min |t' - t| in bins for decoy activity")
    p.add_argument("--decoy_max_gap", type=int, default=0, help="if > 0, max |t' - t| (keeps slow drift similar)")
    # model
    p.add_argument("--d_tok", type=int, default=64)
    p.add_argument("--conv_channels", type=int, default=16)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_queries", type=int, default=4)
    p.add_argument("--d_out", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    # loss
    p.add_argument("--mi_levels", type=str, default="all", choices=["all", "full"],
                   help="maximize the MI of every level (proposal) or only of the full level")
    p.add_argument("--rank_weight", type=float, default=1.0, help="0 = no ranking (Exp. 3 control)")
    p.add_argument("--margin", type=float, default=0.05, help="ranking margin in nats")
    p.add_argument("--rank_order", type=str, default="correct", choices=["correct", "shuffled"],
                   help="shuffled: random order of the levels in the ranking term (Exp. 5 control)")
    p.add_argument("--mi_eps", type=float, default=1e-2, help="shrinkage of the correlation matrices (1e-3 + lr 1e-3 + bf16 collapsed)")
    # optimization
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=1024, help="time points per step (MI is estimated on them)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=250)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    # evaluation
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_subsets", type=int, default=4,
                   help="also evaluate z averaged over this many random |A|-sized neuron subsets")
    p.add_argument("--mlp_probe", action="store_true", help="also a fixed 2-hidden-layer MLP probe at the end")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no_amp", action="store_true")
    return p


# =====================================================================
# data
# =====================================================================
def load_data(npz_path):
    # the user's loader, with the file path as an argument
    if not npz_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = [np.asarray(data[key], dtype=np.float32)
                  for key in ("train_data", "valid_data", "train_label", "valid_label")]
    x_train, x_valid, y_train, y_valid = arrays
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_valid.ndim == 1:
        y_valid = y_valid[:, None]
    for name, values in zip(("X_train", "X_valid", "Y_train", "Y_valid"), (x_train, x_valid, y_train, y_valid)):
        if values.ndim != 2 or min(values.shape) == 0:
            raise ValueError(f"{name} must be a nonempty 2D array; got {values.shape}.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
    if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
        raise ValueError("feature/label lengths differ")
    if x_train.shape[1] != x_valid.shape[1] or y_train.shape[1] != y_valid.shape[1]:
        raise ValueError("train/valid dimensions differ")
    return tuple(np.ascontiguousarray(a) for a in (x_train, x_valid, y_train, y_valid))


def _zscore(tr, va):
    mu, sd = tr.mean(0), tr.std(0) + 1e-8
    return (tr - mu) / sd, (va - mu) / sd


def _pad(x, half, window):
    """Replicate-pad in time so that padded[t : t + window] is the window centered on original t."""
    return torch.cat([x[:1].expand(half, -1), x, x[-1:].expand(window - half, -1)], 0).contiguous()


class SessionData:
    """One session on the GPU: encoder input, reference activity, behavior; window gathering and sampling."""

    def __init__(self, args, device):
        npz = Path(args.npz) if args.npz else Path(args.data_dir) / f"{args.session}.npz"
        self.name = npz.stem
        x_tr, x_va, y_tr, y_va = (a.astype(np.float64) for a in load_data(npz))

        # neurons: same rule as mi_subset_analysis.py, on the reference-smoothed activity
        s_tr = gaussian_filter1d(x_tr, args.b_sigma, axis=0) if args.b_sigma > 0 else x_tr
        s_va = gaussian_filter1d(x_va, args.b_sigma, axis=0) if args.b_sigma > 0 else x_va
        sd_tr = s_tr.std(0)
        keep = (sd_tr > 1e-6) & (s_va.std(0) > args.min_valid_std * np.maximum(sd_tr, 1e-6))
        keep_y = y_va.std(0) > 1e-6
        self.n_dropped, self.n_dropped_y = int((~keep).sum()), int((~keep_y).sum())

        def detrend(a):
            return a - gaussian_filter1d(a, args.detrend_sigma, axis=0) if args.detrend_sigma > 0 else a

        # encoder input: binned counts (optionally smoothed), z-scored with train statistics
        in_tr, in_va = x_tr[:, keep], x_va[:, keep]
        if args.input_sigma > 0:
            in_tr = gaussian_filter1d(in_tr, args.input_sigma, axis=0)
            in_va = gaussian_filter1d(in_va, args.input_sigma, axis=0)
        in_tr, in_va = _zscore(detrend(in_tr), detrend(in_va))
        # reference activity (MI target) and raw-neuron baseline features: smoothed, z-scored
        ref_tr, ref_va = _zscore(detrend(s_tr[:, keep]), detrend(s_va[:, keep]))
        y_tr, y_va = _zscore(y_tr[:, keep_y], y_va[:, keep_y])

        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        self.window, self.half = args.window, args.window // 2
        self.xp_tr = _pad(t(in_tr), self.half, self.window)
        self.xp_va = _pad(t(in_va), self.half, self.window)
        self.ref_tr, self.ref_va = t(ref_tr), t(ref_va)
        self.y_tr, self.y_va = t(y_tr), t(y_va)
        self.T_tr, self.T_va = len(in_tr), len(in_va)
        self.n_neurons = int(keep.sum())
        self.device = device
        self.offsets = torch.arange(self.window, device=device)

    def windows(self, xp, times, neurons):
        """times, neurons: same-shape int tensors (centers in original time, neuron ids) -> (..., W)."""
        tidx = times[..., None] + self.offsets
        return xp[tidx, neurons[..., None].expand_as(tidx)]

    def decoy_times(self, ts, shape, gap, max_gap, gen):
        T, dev = self.T_tr, self.device
        if max_gap <= 0:  # anywhere in train, at least `gap` away (circular)
            off = torch.randint(gap, T - gap + 1, shape, device=dev, generator=gen)
            return (ts[:, None] + off) % T
        mag = torch.randint(gap, max_gap + 1, shape, device=dev, generator=gen)
        sign = torch.randint(0, 2, shape, device=dev, generator=gen) * 2 - 1
        td = ts[:, None] + sign * mag
        bad = (td < 0) | (td >= T)
        return torch.where(bad, ts[:, None] - sign * mag, td).clamp(0, T - 1)

    def train_batch(self, n, b_idx, a_idx, ks, decoy, gap, max_gap, gen):
        """Returns (inputs per level, neuron ids per level, reference activity (n, |B|), real-count mask info)."""
        dev, n_a = self.device, len(a_idx)
        ts = torch.randint(0, self.T_tr, (n,), device=dev, generator=gen)
        order = torch.argsort(torch.rand(n, n_a, device=dev, generator=gen), dim=1)
        neurons = a_idx[order]  # (n, |A|): per-sample random order of A; prefixes = nested real subsets
        target = self.ref_tr[ts][:, b_idx]
        if decoy == "drop":
            xs = [self.windows(self.xp_tr, ts[:, None].expand(n, k), neurons[:, :k]) for k in ks]
            return xs, [neurons[:, :k] for k in ks], target, ts
        real = self.windows(self.xp_tr, ts[:, None].expand(n, n_a), neurons)
        if decoy == "time":
            fake = self.windows(self.xp_tr, self.decoy_times(ts, (n, n_a), gap, max_gap, gen), neurons)
        else:  # zero = the z-scored mean, a constant window
            fake = torch.zeros_like(real)
        pos = torch.arange(n_a, device=dev)
        xs = [torch.where((pos < k)[None, :, None], real, fake) for k in ks]
        return xs, [neurons] * len(ks), target, ts

    @torch.no_grad()
    def embed(self, model, split, neuron_sets, amp):
        """z(t) for every time point of a split, averaged over the given neuron sets."""
        xp, T = (self.xp_tr, self.T_tr) if split == "train" else (self.xp_va, self.T_va)
        total = None
        for nset in neuron_sets:
            n = len(nset)
            chunk = max(64, int(4e6 // (n * self.window)))
            zs = []
            for t0 in range(0, T, chunk):
                ts = torch.arange(t0, min(T, t0 + chunk), device=self.device)
                ids = nset[None].expand(len(ts), n)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    zs.append(model(self.windows(xp, ts[:, None].expand(len(ts), n), ids), ids).float())
            z = torch.cat(zs)
            total = z if total is None else total + z
        return total / len(neuron_sets)


# =====================================================================
# probes
# =====================================================================
def _r2(y, p):
    return 1.0 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0).clamp_min(1e-12)


def ridge_r2(f_tr, y_tr, f_va, y_va, alphas=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    """Ridge probe; alpha chosen on the last 20% of train, refit on all of train, R^2 per dim on valid."""
    f_tr, y_tr, f_va, y_va = (a.double() for a in (f_tr, y_tr, f_va, y_va))
    mu, sd = f_tr.mean(0), f_tr.std(0).clamp_min(1e-6)
    f_tr, f_va = (f_tr - mu) / sd, (f_va - mu) / sd
    eye = torch.eye(f_tr.shape[1], dtype=f_tr.dtype, device=f_tr.device)

    def fit_predict(ftr, ytr, fte, a):
        fm, ym = ftr.mean(0), ytr.mean(0)
        c = ftr - fm
        w = torch.linalg.solve(c.T @ c + a * len(c) * eye, c.T @ (ytr - ym))
        return (fte - fm) @ w + ym

    n_in = int(0.8 * len(f_tr))
    scores = [float(_r2(y_tr[n_in:], fit_predict(f_tr[:n_in], y_tr[:n_in], f_tr[n_in:], a)).mean()) for a in alphas]
    alpha = alphas[int(np.argmax(scores))]
    dims = _r2(y_va, fit_predict(f_tr, y_tr, f_va, alpha))
    return float(dims.mean()), [round(float(d), 4) for d in dims], alpha


def mlp_r2(f_tr, y_tr, f_va, y_va, seed=0, hidden=256, epochs=200, patience=15):
    """Fixed 2-hidden-layer MLP probe, early stopping on the last 20% of train, R^2 on valid."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    mu, sd = f_tr.mean(0), f_tr.std(0).clamp_min(1e-6)
    f_tr, f_va = (f_tr - mu) / sd, (f_va - mu) / sd
    n_in = int(0.8 * len(f_tr))
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(f_tr.shape[1], hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, y_tr.shape[1])).to(f_tr.device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(n_in, generator=g).split(512):
            idx = idx.to(f_tr.device)
            loss = ((net(f_tr[idx]) - y_tr[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(((net(f_tr[n_in:]) - y_tr[n_in:]) ** 2).mean())
        if val < best - 1e-5:
            best, best_state, bad = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        dims = _r2(y_va.double(), net(f_va).double())
    return float(dims.mean()), [round(float(d), 4) for d in dims]


def evaluate(model, data, eval_sets, amp, mlp=False, seed=0):
    model.eval()
    out = {}
    for name, sets in eval_sets.items():
        z_tr, z_va = data.embed(model, "train", sets, amp), data.embed(model, "valid", sets, amp)
        out[f"r2_{name}"], out[f"r2_{name}_dims"], out[f"alpha_{name}"] = ridge_r2(z_tr, data.y_tr, z_va, data.y_va)
        if mlp:
            out[f"mlp_r2_{name}"], out[f"mlp_r2_{name}_dims"] = mlp_r2(z_tr, data.y_tr, z_va, data.y_va, seed)
    model.train()
    return out


# =====================================================================
# training
# =====================================================================
def main(argv=None):
    args = get_parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    amp = device.startswith("cuda") and not args.no_amp
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = SessionData(args, device)
    out_dir = args.out_dir or os.path.join("rank_runs", data.name, "default")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    n = data.n_neurons
    n_b = min(args.b_max, max(2, int(round(args.b_frac * n))))
    n_a = n - n_b
    ks = sorted({max(args.k_min, min(n_a, int(round(float(fr) * n_a)))) for fr in args.levels.split(",")}, reverse=True)
    if len(ks) < 2:
        raise ValueError(f"need at least 2 distinct levels, got {ks} for |A| = {n_a}")
    gap = min(args.decoy_gap, data.T_tr // 4)
    print(f"{data.name}: {n} neurons ({data.n_dropped} dropped), behavior dims {data.y_tr.shape[1]} "
          f"({data.n_dropped_y} constant dropped), train {data.T_tr} / valid {data.T_va} bins")
    print(f"|B| = {n_b}, |A| = {n_a}, levels (real neurons) = {ks}, decoy = {args.decoy}, gap = {gap}, "
          f"rank_weight = {args.rank_weight}, order = {args.rank_order}, mi_levels = {args.mi_levels}, "
          f"fixed_b = {args.fixed_b}")

    gen = torch.Generator(device=device).manual_seed(args.seed)
    all_ids = torch.arange(n, device=device)
    fixed_b = torch.randperm(n, device=device, generator=gen)[:n_b] if args.fixed_b else None
    model = NeuronSetEncoder(n, args.window, args.d_tok, args.conv_channels, args.n_layers, args.n_heads,
                             args.n_queries, args.d_out, args.dropout).to(device)
    print(f"encoder: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}k parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(args.warmup, 1)) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / max(args.steps, 1)))))

    # evaluation neuron sets: all neurons, and z averaged over random |A|-sized subsets
    eval_pool = all_ids if fixed_b is None else all_ids[~torch.isin(all_ids, fixed_b)]
    eval_sets = {"all": [eval_pool]}
    if args.eval_subsets > 0 and len(eval_pool) > n_a:
        g_eval = torch.Generator(device=device).manual_seed(args.seed + 1)
        eval_sets["subset"] = [eval_pool[torch.randperm(len(eval_pool), device=device, generator=g_eval)[:n_a]]
                               for _ in range(args.eval_subsets)]

    raw_r2, raw_dims, _ = ridge_r2(data.ref_tr, data.y_tr, data.ref_va, data.y_va)
    init = evaluate(model, data, eval_sets, amp)
    print(f"[baseline] raw neurons ridge R2 = {raw_r2:.4f} | random-init encoder R2 = {init['r2_all']:.4f}"
          + (f" (subset avg {init['r2_subset']:.4f})" if "r2_subset" in init else ""))
    history = {"log": [], "eval": [{"step": 0, **init}]}

    n_lv = len(ks)
    acc, t0 = {}, time.time()
    recent = []
    for step in range(1, args.steps + 1):
        b_idx = fixed_b if fixed_b is not None else torch.randperm(n, device=device, generator=gen)[:n_b]
        a_idx = all_ids[~torch.isin(all_ids, b_idx)]
        xs, ids, target, _ = data.train_batch(args.batch, b_idx, a_idx, ks, args.decoy, gap,
                                              args.decoy_max_gap, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            if args.decoy == "drop":  # different set sizes: one forward per level
                zs = [model(x, i) for x, i in zip(xs, ids)]
            else:  # same size: all levels in one forward
                zs = model(torch.cat(xs), ids[0].repeat(n_lv, 1)).split(args.batch)
        m = torch.stack([gaussian_mi(z.float(), target, args.mi_eps) for z in zs])
        loss_mi = -(m.mean() if args.mi_levels == "all" else m[0])
        order = torch.randperm(n_lv, device=device, generator=gen) if args.rank_order == "shuffled" else None
        l_rank, viol = rank_loss(m, args.margin, order)
        loss = loss_mi + args.rank_weight * l_rank
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, skipped")
            opt.zero_grad(set_to_none=True)
            sched.step()
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        # natural order of the levels (independent of the shuffled control): fraction of wrong pairs
        _, nat_viol = rank_loss(m.detach(), 0.0)
        stats = {"loss": float(loss), "loss_rank": float(l_rank), "viol": float(nat_viol)}
        stats.update({f"M{k}": float(v) for k, v in zip(ks, m.detach())})
        recent.append(stats)
        recent = recent[-200:]
        for k, v in stats.items():
            acc[k] = acc.get(k, 0.0) + v
        acc["_n"] = acc.get("_n", 0) + 1
        if step % args.log_every == 0:
            cnt = acc.pop("_n")
            row = {k: v / cnt for k, v in acc.items()}
            row.update(step=step, eff_rank=effective_rank(zs[0].float()), lr=sched.get_last_lr()[0])
            history["log"].append(row)
            mis = " ".join(f"M{k}={row[f'M{k}']:.3f}" for k in ks)
            print(f"step {step:6d} loss {row['loss']:+.4f} rank {row['loss_rank']:.4f} wrong-order "
                  f"{100 * row['viol']:.1f}% | {mis} | eff_rank {row['eff_rank']:.1f} ({time.time() - t0:.0f}s)")
            acc = {}
        if step % args.eval_every == 0 and step < args.steps:
            ev = evaluate(model, data, eval_sets, amp)
            history["eval"].append({"step": step, **ev})
            print(f"[eval] step {step}: R2 = {ev['r2_all']:.4f}"
                  + (f" (subset avg {ev['r2_subset']:.4f})" if "r2_subset" in ev else ""))

    final = evaluate(model, data, eval_sets, amp, mlp=args.mlp_probe, seed=args.seed)
    history["eval"].append({"step": args.steps, **final})
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    last = {k: float(np.mean([r[k] for r in recent])) for k in recent[0]} if recent else {}
    summary = {
        "session": data.name, "n_neurons": n, "n_b": n_b, "n_a": n_a, "levels": ks,
        "eval_neurons": "all" if fixed_b is None else "A (fixed B excluded)",
        "raw_r2": raw_r2, "raw_r2_dims": raw_dims,
        "init_r2_all": init["r2_all"], "init_r2_subset": init.get("r2_subset"),
        "final_r2_all": final["r2_all"], "final_r2_subset": final.get("r2_subset"),
        "final_r2_all_dims": final["r2_all_dims"],
        "final_mlp_r2_all": final.get("mlp_r2_all"),
        "train_last200": last, "train_seconds": time.time() - t0, "args": vars(args),
    }
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    with open(os.path.join(out_dir, "final.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\n" + "=" * 70)
    print(f"{data.name} | ridge R2 on valid (behavior dims averaged)")
    print(f"  raw neurons (smoothed)         : {raw_r2:.4f}")
    print(f"  random-init encoder            : {init['r2_all']:.4f}")
    print(f"  trained encoder, all neurons   : {final['r2_all']:.4f}")
    if "r2_subset" in final:
        print(f"  trained encoder, subset avg    : {final['r2_subset']:.4f}")
    if args.mlp_probe:
        print(f"  trained encoder, MLP probe     : {final['mlp_r2_all']:.4f}")
    if last:
        print("  last 200 steps: " + " ".join(f"M{k}={last[f'M{k}']:.3f}" for k in ks)
              + f" | wrong-order pairs {100 * last['viol']:.1f}%")
    print(f"saved {out_dir}/final.json, history.json, model.pt")
    return summary


if __name__ == "__main__":
    main()

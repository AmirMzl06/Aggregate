"""Data-only sanity check for the nested-subset ranking idea (no network, no training).

For random splits of the neurons into a reference population B (n_b neurons) and the rest A,
and nested subsets A_k (k = 5, 10, ..., 80 neurons) of A, it measures on held-out data:

    I(A_k ; B)          Gaussian (log-det) mutual information, cross-validated
    I(A_k ; behavior)   same estimator with the behavior labels as target
    R2(A_k -> behavior) ridge decoding R2 (mean over label dims)

and answers:
  1. is I(A_k; B) monotonic in k, and are the gaps between levels larger than the estimation noise?
  2. does I(A_k; B) track I(A_k; behavior)?  Across k, and more importantly, for a FIXED B, across
     different random subsets of A of the SAME size (if yes, B is a good label-free proxy for
     "behaviorally informative"; B must be fixed, otherwise a B that takes the informative neurons
     raises I(A;B) while lowering I(A;behavior) and the comparison is confounded).

Cross-validated Gaussian MI: fit a ridge map A_k -> target on the train split, then on the valid split
    I = 1/2 [ log det Cov(target) - log det Cov(target - prediction) ]
(predictive information under a linear-Gaussian model; it does not grow just because k grows).

    python mi_subset_analysis.py --session C-CO12
    python mi_subset_analysis.py --npz /path/to/file.npz --n_b 16 --ks 5,10,20,40,60,70,80 --b_draws 5 --reps 20
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")


def load_data(npz_path):
    # the user's loader, with the file path as an argument
    if not npz_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = [
            np.asarray(data[key], dtype=np.float32)
            for key in ("train_data", "valid_data", "train_label", "valid_label")
        ]
    x_train, x_valid, y_train, y_valid = arrays
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_valid.ndim == 1:
        y_valid = y_valid[:, None]
    for name, values in zip(
        ("X_train", "X_valid", "Y_train", "Y_valid"),
        (x_train, x_valid, y_train, y_valid),
    ):
        if values.ndim != 2 or min(values.shape) == 0:
            raise ValueError(f"{name} must be a nonempty 2D array; got {values.shape}.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
    if len(x_train) != len(y_train):
        raise ValueError(f"Training feature/label lengths differ: {len(x_train)} vs {len(y_train)}.")
    if len(x_valid) != len(y_valid):
        raise ValueError(f"Validation feature/label lengths differ: {len(x_valid)} vs {len(y_valid)}.")
    if x_train.shape[1] != x_valid.shape[1]:
        raise ValueError("Training and validation neuron counts do not match.")
    if y_train.shape[1] != y_valid.shape[1]:
        raise ValueError("Training and validation label dimensions do not match.")
    return tuple(np.ascontiguousarray(a) for a in (x_train, x_valid, y_train, y_valid))


def preprocess(x_tr, x_va, y_tr, y_va, sigma, lag, min_valid_std):
    """Gaussian smoothing over time, drop silent neurons, z-score with train statistics, apply lag.

    A neuron is dropped if it is silent in train, or if (after z-scoring with train statistics) its
    std in valid is below min_valid_std: a neuron that (almost) stops firing in valid has ~zero target
    variance there and makes log det Cov(B) blow up to a huge negative number.
    """
    if sigma > 0:
        x_tr = gaussian_filter1d(x_tr, sigma, axis=0)
        x_va = gaussian_filter1d(x_va, sigma, axis=0)
    sd_tr = x_tr.std(0)
    keep = (sd_tr > 1e-6) & (x_va.std(0) > min_valid_std * np.maximum(sd_tr, 1e-6))
    dropped = int((~keep).sum())
    x_tr, x_va = x_tr[:, keep], x_va[:, keep]
    mu, sd = x_tr.mean(0), x_tr.std(0)
    x_tr, x_va = (x_tr - mu) / sd, (x_va - mu) / sd
    keep_y = y_va.std(0) > 1e-6  # constant behavior dims carry no information
    y_tr, y_va = y_tr[:, keep_y], y_va[:, keep_y]
    ymu, ysd = y_tr.mean(0), y_tr.std(0) + 1e-8
    y_tr, y_va = (y_tr - ymu) / ysd, (y_va - ymu) / ysd
    if lag > 0:  # neural activity at t predicts behavior at t + lag
        x_tr, y_tr, x_va, y_va = x_tr[:-lag], y_tr[lag:], x_va[:-lag], y_va[lag:]
    return (x_tr.astype(np.float64), x_va.astype(np.float64), y_tr.astype(np.float64), y_va.astype(np.float64),
            int(keep.sum()), dropped, int((~keep_y).sum()))


def logdet_cov(z, eps=1e-6):
    c = np.cov(z, rowvar=False).reshape(z.shape[1], z.shape[1])
    return np.linalg.slogdet(c + eps * np.eye(len(c)))[1]


def cv_gaussian_mi_and_r2(x_tr, t_tr, x_va, t_va, alpha):
    """Ridge fit on train, Gaussian MI (nats) and mean R2 on valid."""
    xtx = x_tr.T @ x_tr
    w = np.linalg.solve(xtx + alpha * len(x_tr) * np.eye(len(xtx)), x_tr.T @ t_tr)
    resid = t_va - x_va @ w
    mi = 0.5 * (logdet_cov(t_va) - logdet_cov(resid))
    r2 = 1.0 - (resid ** 2).sum(0) / ((t_va - t_va.mean(0)) ** 2).sum(0)
    return float(mi), float(r2.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", type=str, default="C-CO12")
    p.add_argument("--npz", type=str, default="", help="overrides --session")
    p.add_argument("--n_b", type=int, default=16, help="neurons in the reference population B")
    p.add_argument("--ks", type=str, default="5,10,20,40,60,70,80", help="nested subset sizes of A")
    p.add_argument("--b_draws", type=int, default=5, help="random reference populations B")
    p.add_argument("--reps", type=int, default=20, help="random orders of A per B (nested prefixes)")
    p.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing in bins (0 = off)")
    p.add_argument("--lag", type=int, default=0, help="behavior lag in bins (neural leads behavior)")
    p.add_argument("--alpha", type=float, default=1e-3, help="ridge strength (relative to n samples)")
    p.add_argument("--min_valid_std", type=float, default=0.2,
                   help="drop neurons whose valid std is below this fraction of their train std")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="mi_analysis")
    args = p.parse_args()

    npz = Path(args.npz) if args.npz else PERICH_DATA_DIR / f"{args.session}.npz"
    x_tr, x_va, y_tr, y_va = load_data(npz)
    print(f"{npz.name}: train {x_tr.shape}, valid {x_va.shape}, behavior dims {y_tr.shape[1]}")
    x_tr, x_va, y_tr, y_va, n, dropped, dropped_y = preprocess(x_tr, x_va, y_tr, y_va, args.sigma, args.lag,
                                                                args.min_valid_std)
    print(f"dropped {dropped} neurons (silent in train or nearly silent in valid), "
          f"{dropped_y} constant behavior dims")
    ks = [k for k in (int(s) for s in args.ks.split(",")) if k <= n - args.n_b]
    print(f"{n} active neurons -> |B| = {args.n_b}, |A| = {n - args.n_b}, subset sizes {ks}")

    rng = np.random.default_rng(args.seed)
    shape = (args.b_draws, args.reps, len(ks))
    mi_b, mi_y, r2_y = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for d in range(args.b_draws):
        perm = rng.permutation(n)
        b, a_all = perm[:args.n_b], perm[args.n_b:]
        for r in range(args.reps):
            a = rng.permutation(a_all)  # A in a random order, prefixes are nested
            for j, k in enumerate(ks):
                idx = a[:k]
                mi_b[d, r, j], _ = cv_gaussian_mi_and_r2(x_tr[:, idx], x_tr[:, b], x_va[:, idx], x_va[:, b], args.alpha)
                mi_y[d, r, j], r2_y[d, r, j] = cv_gaussian_mi_and_r2(x_tr[:, idx], y_tr, x_va[:, idx], y_va, args.alpha)
    flat = lambda v: v.reshape(-1, len(ks))  # (all draws, k)

    # ---------------- report ----------------
    print("\n  k   I(A_k;B) nats      I(A_k;behav) nats   R2(A_k->behav)")
    for j, k in enumerate(ks):
        fb, fy, fr = flat(mi_b)[:, j], flat(mi_y)[:, j], flat(r2_y)[:, j]
        print(f"{k:4d}   {fb.mean():.4f} ± {fb.std():.4f}   {fy.mean():.4f} ± {fy.std():.4f}    "
              f"{fr.mean():.4f} ± {fr.std():.4f}")

    print("\nQ1: neighbouring levels (same draw, nested subsets)")
    print("  pair      mean gap I(.;B)   gap / std   ordered correctly")
    for j in range(len(ks) - 1):
        gap = flat(mi_b)[:, j + 1] - flat(mi_b)[:, j]
        sd = gap.std() + 1e-12
        print(f"  {ks[j]:>3d}->{ks[j + 1]:<3d}   {gap.mean():+.4f}          {gap.mean() / sd:5.2f}       "
              f"{100 * (gap > 0).mean():5.1f}% of draws")

    print("\nQ2: does information about B track information about behavior?")
    rho_all = spearmanr(mi_b.ravel(), mi_y.ravel()).correlation
    print(f"  across all points:                          Spearman rho = {rho_all:+.3f}   (mostly driven by k)")
    within = []
    for j, k in enumerate(ks):
        if k >= n - args.n_b:  # A_k is all of A: no variation for a fixed B
            continue
        rhos = [spearmanr(mi_b[d, :, j], mi_y[d, :, j]).correlation for d in range(args.b_draws)]
        within.append(float(np.nanmean(rhos)))
        print(f"  fixed B, same size k = {k:3d}, across subsets: Spearman rho = {within[-1]:+.3f} "
              f"(min {np.nanmin(rhos):+.2f}, max {np.nanmax(rhos):+.2f} over B draws)")
    print(f"  mean within-size rho = {np.nanmean(within):+.3f}   "
          f"(> 0: subsets that know more about B also know more about behavior)")

    os.makedirs(args.out_dir, exist_ok=True)
    res = {"session": npz.stem, "n_neurons": n, "n_b": args.n_b, "ks": ks, "args": vars(args),
           "mi_b": mi_b.tolist(), "mi_behavior": mi_y.tolist(), "r2_behavior": r2_y.tolist(),
           "spearman_all": rho_all, "spearman_within_k_fixed_b": within}
    out = os.path.join(args.out_dir, f"{npz.stem}_nb{args.n_b}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nsaved {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for arr, name, a in ((mi_b, "I(A_k; B) [nats]", ax[0]), (mi_y, "I(A_k; behavior) [nats]", ax[1])):
            a.errorbar(ks, flat(arr).mean(0), yerr=flat(arr).std(0), marker="o", capsize=3)
            a.set_xlabel("k (neurons in A_k)")
            a.set_ylabel(name)
        sc = ax[2].scatter(mi_b.ravel(), mi_y.ravel(), c=np.tile(ks, args.b_draws * args.reps), cmap="viridis", s=12)
        ax[2].set_xlabel("I(A_k; B)")
        ax[2].set_ylabel("I(A_k; behavior)")
        fig.colorbar(sc, ax=ax[2], label="k")
        fig.suptitle(f"{npz.stem}: |B| = {args.n_b}, {args.b_draws} B draws x {args.reps} orders of A, cross-validated")
        fig.tight_layout()
        png = out.replace(".json", ".png")
        fig.savefig(png, dpi=150)
        print(f"saved {png}")
    except ImportError:
        print("matplotlib not available, skipping the plot")


if __name__ == "__main__":
    main()

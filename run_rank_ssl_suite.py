"""Run the nested-subset ranking method and its controls on several Perich sessions, then summarize.

    nohup python run_rank_ssl_suite.py --gpus 0,1 > rank_suite.log 2>&1 &
    python run_rank_ssl_suite.py --summary

Each (session, condition) is one process of train_rank_ssl.py (one per GPU at a time); finished runs
(final.json present) are skipped, so rerunning the same command continues an interrupted suite.
Results: <root>/<session>/<condition>/final.json ; table: <root>/summary.csv
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time

# (name, flags for train_rank_ssl.py, what it tests)
CONDITIONS = [
    ("main",          "",                          "ranking + MI of every level, decoys = other times (proposed)"),
    ("no_rank",       "--rank_weight 0",           "Exp. 3: same MI terms, no ranking"),
    ("shuffled_rank", "--rank_order shuffled",     "Exp. 5: ranking with random level order"),
    ("mi_full",       "--mi_levels full",          "MI only on the full level, other levels only ranked"),
    ("decoy_zero",    "--decoy zero",              "no 'resize': missing neurons are constant (size visible)"),
    ("decoy_drop",    "--decoy drop",              "no 'resize': smaller sets (original proposal)"),
    ("drop_no_rank",  "--decoy drop --rank_weight 0", "smaller sets, no ranking (2x2 with main / no_rank / decoy_drop)"),
    ("fixed_b",       "--fixed_b",                 "Exp. 1: one fixed reference population B"),
]
# one or two sessions per monkey / task, small and large (J has < 40 neurons and is left out)
DEFAULT_SESSIONS = "C-CO12,C-CO41,C-RT2,M-CO1,M-RT0,T-CO4,T-RT3"


def parse():
    p = argparse.ArgumentParser(description="Nested-subset ranking suite")
    p.add_argument("--data_dir", type=str, default="/data/hossein/mm_project/perich_data_valid_final_raw/")
    p.add_argument("--sessions", type=str, default=DEFAULT_SESSIONS, help="comma separated, or 'all'")
    p.add_argument("--min_neurons", type=int, default=50, help="with --sessions all: skip smaller sessions")
    p.add_argument("--conditions", type=str, default="", help="comma separated subset of the conditions")
    p.add_argument("--root", type=str, default="rank_runs")
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--extra", type=str, default="", help="flags appended to every run, e.g. '--steps 3000'")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def sessions_of(args):
    if args.sessions != "all":
        return [s for s in args.sessions.split(",") if s]
    import numpy as np
    out = []
    for f in sorted(os.listdir(args.data_dir)):
        if f.endswith(".npz"):
            with np.load(os.path.join(args.data_dir, f)) as d:
                if d["train_data"].shape[1] >= args.min_neurons:
                    out.append(f[:-4])
    return out


def conditions_of(args):
    if not args.conditions:
        return CONDITIONS
    names = set(args.conditions.split(","))
    unknown = names - {c[0] for c in CONDITIONS}
    if unknown:
        sys.exit(f"unknown conditions {sorted(unknown)}; available: {[c[0] for c in CONDITIONS]}")
    return [c for c in CONDITIONS if c[0] in names]


def out_dir(args, session, cond):
    return os.path.join(args.root, session, cond)


def done(args, session, cond):
    return os.path.exists(os.path.join(out_dir(args, session, cond), "final.json"))


def command(args, session, cond, flags):
    here = os.path.dirname(os.path.abspath(__file__))
    return ([sys.executable, os.path.join(here, "train_rank_ssl.py"), "--data_dir", args.data_dir,
             "--session", session, "--out_dir", out_dir(args, session, cond)]
            + shlex.split(flags) + shlex.split(args.extra))


def summary(args):
    sessions, conds = sessions_of(args), conditions_of(args)
    res = {}
    for s in sessions:
        for c, _, _ in conds:
            f = os.path.join(out_dir(args, s, c), "final.json")
            if os.path.exists(f):
                with open(f) as fh:
                    res[(s, c)] = json.load(fh)
    if not res:
        print(f"no finished runs in {args.root}/")
        return
    print("\nRidge R2 on valid (mean over behavior dims) -- higher is better")
    head = f"{'condition':<15s}" + "".join(f"{s:>10s}" for s in sessions) + f"{'mean':>9s}{'Δinit':>8s}"
    print(head)
    rows = []

    def line(name, getter):
        vals = [getter(s) for s in sessions]
        ok = [v for v in vals if v is not None]
        mean = sum(ok) / len(ok) if ok else float("nan")
        return name, vals, mean

    first = {s: next((res[(s, c)] for c, _, _ in conds if (s, c) in res), None) for s in sessions}
    rows.append(line("raw neurons", lambda s: first[s]["raw_r2"] if first[s] else None) + (None,))
    rows.append(line("raw window", lambda s: first[s].get("raw_window_r2") if first[s] else None) + (None,))
    rows.append(line("random init", lambda s: first[s]["init_r2_all"] if first[s] else None) + (None,))
    for c, _, _ in conds:
        name, vals, mean = line(c, lambda s, c=c: res[(s, c)]["final_r2_all"] if (s, c) in res else None)
        gains = [res[(s, c)]["final_r2_all"] - res[(s, c)]["init_r2_all"] for s in sessions if (s, c) in res]
        rows.append((name, vals, mean, sum(gains) / len(gains) if gains else None))
    for name, vals, mean, gain in rows:
        print(f"{name:<15s}" + "".join(f"{v:10.4f}" if v is not None else f"{'-':>10s}" for v in vals)
              + f"{mean:9.4f}" + (f"{gain:+8.4f}" if gain is not None else f"{'':>8s}"))

    print("\nsubset-averaged embedding (|A|-sized random neuron sets), and training diagnostics (mean over sessions)")
    print(f"{'condition':<15s}{'R2 subset':>10s}{'wrong-order %':>15s}{'shifted-B':>11s}{'perm-B':>8s}"
          f"  MI per level (nats, last 200 steps)")
    for c, _, _ in conds:
        runs = [res[(s, c)] for s in sessions if (s, c) in res]
        if not runs:
            continue
        sub = [r["final_r2_subset"] for r in runs if r.get("final_r2_subset") is not None]
        viol = [r["train_last200"].get("viol") for r in runs if r.get("train_last200")]
        mis = [[v for k, v in r["train_last200"].items() if k.startswith("M") and k[1:].isdigit()]
               for r in runs if r.get("train_last200")]
        shift = [r["train_last200"].get("M_shift") for r in runs if r.get("train_last200", {}).get("M_shift") is not None]
        perm = [r["train_last200"].get("M_perm") for r in runs if r.get("train_last200", {}).get("M_perm") is not None]
        n_lv = min((len(m) for m in mis), default=0)
        mi_mean = [sum(m[i] for m in mis) / len(mis) for i in range(n_lv)]
        print(f"{c:<15s}" + (f"{sum(sub) / len(sub):10.4f}" if sub else f"{'-':>10s}")
              + (f"{100 * sum(viol) / len(viol):15.1f}" if viol else f"{'-':>15s}")
              + (f"{sum(shift) / len(shift):11.3f}" if shift else f"{'-':>11s}")
              + (f"{sum(perm) / len(perm):8.3f}" if perm else f"{'-':>8s}")
              + "  " + " ".join(f"{v:.3f}" for v in mi_mean))

    path = os.path.join(args.root, "summary.csv")
    with open(path, "w") as f:
        f.write("condition," + ",".join(sessions) + ",mean,gain_vs_init\n")
        for name, vals, mean, gain in rows:
            f.write(name + "," + ",".join("" if v is None else f"{v:.6f}" for v in vals)
                    + f",{mean:.6f}," + ("" if gain is None else f"{gain:.6f}") + "\n")
    print(f"\nsaved {path}")
    print("conditions: " + "; ".join(f"{c}: {d}" for c, _, d in conds))


def main():
    args = parse()
    os.makedirs(args.root, exist_ok=True)
    if args.summary:
        summary(args)
        return
    jobs = [(s, c, fl, d) for s in sessions_of(args) for c, fl, d in conditions_of(args)]
    skipped = [f"{s}/{c}" for s, c, _, _ in jobs if done(args, s, c)]
    queue = [j for j in jobs if not done(args, j[0], j[1])]
    if skipped:
        print(f"already finished, skipping {len(skipped)}: {', '.join(skipped)}")
    gpus = [g for g in args.gpus.split(",") if g.strip()]
    if args.dry_run:
        for i, (s, c, fl, d) in enumerate(queue):
            print(f"CUDA_VISIBLE_DEVICES={gpus[i % len(gpus)]} {shlex.join(command(args, s, c, fl))}  # {d}")
        return
    running, failed = {}, []
    try:
        while queue or running:
            for g in gpus:
                if g not in running and queue:
                    s, c, fl, d = queue.pop(0)
                    os.makedirs(out_dir(args, s, c), exist_ok=True)
                    log = open(os.path.join(out_dir(args, s, c), "train.log"), "a")
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONUNBUFFERED="1")
                    running[g] = (s, c, subprocess.Popen(command(args, s, c, fl), stdout=log,
                                                         stderr=subprocess.STDOUT, env=env), log)
                    print(f"[{time.strftime('%H:%M:%S')}] start {s}/{c} on GPU {g}", flush=True)
            for g, (s, c, proc, log) in list(running.items()):
                if proc.poll() is not None:
                    log.close()
                    del running[g]
                    ok = proc.returncode == 0 and done(args, s, c)
                    if not ok:
                        failed.append(f"{s}/{c}")
                    print(f"[{time.strftime('%H:%M:%S')}] {'done' if ok else 'FAILED'} {s}/{c}", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        for s, c, proc, log in running.values():
            proc.terminate()
            log.close()
        print("interrupted; rerun the same command to continue")
        raise
    if failed:
        print(f"failed: {', '.join(failed)} (see <root>/<session>/<condition>/train.log)")
    summary(args)


if __name__ == "__main__":
    main()

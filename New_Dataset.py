from __future__ import annotations

import os
import re
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.io import loadmat


# ============================================================
# CONFIG
# ============================================================
ROOT = Path("./data/RecogMemory")

SESSION = "P10HMH_092206"
BLOCK = "NO"

SORTED_DIR = ROOT / "Data" / "sorted" / SESSION / BLOCK
EVENTS_DIR = ROOT / "Data" / "events" / SESSION / BLOCK
CODE_DIR = ROOT / "Code"

MAX_NEURONS_TO_PRINT = 8
MAX_EVENT_ROWS_TO_PRINT = 40
MAX_CODE_MATCHES_PER_FILE = 12


# ============================================================
# HELPERS
# ============================================================
def banner(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def public_mat_keys(d: dict) -> list[str]:
    return [k for k in d.keys() if not k.startswith("__")]


def describe_array(name: str, arr) -> None:
    a = np.asarray(arr)
    print(f"{name}:")
    print(f"  dtype={a.dtype}")
    print(f"  shape={a.shape}")

    if a.size == 0:
        print("  EMPTY")
        return

    if np.issubdtype(a.dtype, np.number):
        flat = a.astype(np.float64, copy=False).ravel()
        finite = flat[np.isfinite(flat)]
        if finite.size:
            print(f"  min={finite.min():.6g}")
            print(f"  max={finite.max():.6g}")
            print(f"  mean={finite.mean():.6g}")


def load_mat(path: Path) -> dict:
    return loadmat(path, squeeze_me=True, struct_as_record=False)


def choose_spike_matrix(mat: dict):
    """Prefer `spikes`; otherwise return the largest numeric 2D array."""
    if "spikes" in mat:
        return np.asarray(mat["spikes"]), "spikes"

    candidates = []
    for k in public_mat_keys(mat):
        v = np.asarray(mat[k])
        if np.issubdtype(v.dtype, np.number) and v.ndim >= 1:
            candidates.append((v.size, k, v))

    if not candidates:
        return None, None

    _, key, arr = max(candidates, key=lambda x: x[0])
    return np.asarray(arr), key


def choose_events_matrix(mat: dict):
    if "events" in mat:
        return np.asarray(mat["events"]), "events"

    candidates = []
    for k in public_mat_keys(mat):
        v = np.asarray(mat[k])
        if (
            np.issubdtype(v.dtype, np.number)
            and v.ndim == 2
            and v.shape[1] >= 2
        ):
            candidates.append((v.size, k, v))

    if not candidates:
        return None, None

    _, key, arr = max(candidates, key=lambda x: x[0])
    return np.asarray(arr), key


def safe_unique_counts(x):
    vals, cnts = np.unique(x, return_counts=True)
    order = np.argsort(vals)
    return list(zip(vals[order], cnts[order]))


def find_timestamp_like_column(arr: np.ndarray):
    """
    Heuristic only:
    for a numeric 2D array, choose the column with largest dynamic range / magnitude.
    We print all column summaries anyway.
    """
    if arr.ndim != 2:
        return None

    scores = []
    for j in range(arr.shape[1]):
        c = np.asarray(arr[:, j], dtype=np.float64)
        c = c[np.isfinite(c)]
        if c.size == 0:
            scores.append((-np.inf, j))
            continue
        dynamic = float(c.max() - c.min())
        magnitude = float(np.median(np.abs(c)))
        scores.append((np.log10(dynamic + 1.0) + np.log10(magnitude + 1.0), j))

    return max(scores)[1]


def inspect_mat_file(path: Path) -> None:
    mat = load_mat(path)
    print(f"\nFILE: {path}")
    print("keys:", public_mat_keys(mat))

    for k in public_mat_keys(mat):
        v = mat[k]
        try:
            describe_array(k, v)
        except Exception as exc:
            print(f"{k}: could not summarize ({exc})")


# ============================================================
# MAIN INSPECTION
# ============================================================
def main():
    banner("PATH CHECK")

    print("ROOT      :", ROOT)
    print("SORTED_DIR:", SORTED_DIR)
    print("EVENTS_DIR:", EVENTS_DIR)
    print("CODE_DIR  :", CODE_DIR)

    for p in [ROOT, SORTED_DIR, EVENTS_DIR, CODE_DIR]:
        print(f"exists({p}) = {p.exists()}")

    if not SORTED_DIR.exists():
        raise FileNotFoundError(SORTED_DIR)
    if not EVENTS_DIR.exists():
        raise FileNotFoundError(EVENTS_DIR)

    banner("DIRECTORY CONTENTS")

    sorted_files = sorted(SORTED_DIR.iterdir())
    event_files = sorted(EVENTS_DIR.iterdir())

    print("\nSorted/neuron directory:")
    for p in sorted_files:
        print(" ", p.name)

    print("\nEvents directory:")
    for p in event_files:
        print(" ", p.name)

    # --------------------------------------------------------
    # Neurons
    # --------------------------------------------------------
    banner("NEURON FILES")

    neuron_files = sorted(SORTED_DIR.glob("A*_cells.mat"))
    print("Number of A*_cells.mat files:", len(neuron_files))

    spike_ranges = []
    spike_counts = []
    inferred_spike_col = None

    for i, path in enumerate(neuron_files):
        mat = load_mat(path)
        spikes, key = choose_spike_matrix(mat)

        if spikes is None:
            print(f"{path.name}: no numeric spike-like array found")
            continue

        spikes = np.asarray(spikes)

        if spikes.ndim == 1:
            spikes = spikes[:, None]

        ts_col = find_timestamp_like_column(spikes)
        if inferred_spike_col is None:
            inferred_spike_col = ts_col

        if i < MAX_NEURONS_TO_PRINT:
            print(f"\n{path.name}")
            print("  selected key:", key)
            print("  shape:", spikes.shape)
            print("  inferred timestamp-like column:", ts_col)

            if spikes.ndim == 2:
                for j in range(spikes.shape[1]):
                    c = spikes[:, j]
                    if np.issubdtype(c.dtype, np.number):
                        c = c.astype(np.float64)
                        print(
                            f"  col {j}: "
                            f"min={np.nanmin(c):.6g}, "
                            f"max={np.nanmax(c):.6g}, "
                            f"median={np.nanmedian(c):.6g}, "
                            f"unique~={len(np.unique(c[:min(len(c), 5000)]))}"
                        )

                print("  first rows:")
                print(spikes[:5])

        if ts_col is not None and spikes.ndim == 2 and len(spikes):
            ts = spikes[:, ts_col].astype(np.float64)
            finite = ts[np.isfinite(ts)]
            if finite.size:
                spike_ranges.append((path.name, finite.min(), finite.max()))
                spike_counts.append((path.name, len(finite)))

    if spike_ranges:
        global_spike_min = min(x[1] for x in spike_ranges)
        global_spike_max = max(x[2] for x in spike_ranges)

        print("\nGlobal spike timestamp-like range across neurons:")
        print("  min:", global_spike_min)
        print("  max:", global_spike_max)
        print("  span:", global_spike_max - global_spike_min)

        print("\nSpike counts:")
        counts = np.array([x[1] for x in spike_counts], dtype=float)
        print("  min neuron spikes :", int(counts.min()))
        print("  median             :", float(np.median(counts)))
        print("  max                :", int(counts.max()))

    # --------------------------------------------------------
    # eventsRaw.mat
    # --------------------------------------------------------
    banner("EVENTS")

    events_path = EVENTS_DIR / "eventsRaw.mat"
    if not events_path.exists():
        raise FileNotFoundError(events_path)

    emat = load_mat(events_path)
    events, events_key = choose_events_matrix(emat)

    print("eventsRaw keys:", public_mat_keys(emat))
    print("selected key:", events_key)

    if events is None:
        raise RuntimeError("Could not locate numeric event matrix.")

    events = np.asarray(events)

    if events.ndim == 1:
        events = events[:, None]

    print("events shape:", events.shape)
    print("\nFirst event rows:")
    print(events[:MAX_EVENT_ROWS_TO_PRINT])

    event_ts_col = find_timestamp_like_column(events)
    print("\nInferred event timestamp-like column:", event_ts_col)

    for j in range(events.shape[1]):
        c = events[:, j]
        if np.issubdtype(c.dtype, np.number):
            c = c.astype(np.float64)
            print(
                f"event col {j}: "
                f"min={np.nanmin(c):.6g}, "
                f"max={np.nanmax(c):.6g}, "
                f"median={np.nanmedian(c):.6g}, "
                f"unique={len(np.unique(c))}"
            )

    # In this release the second column is commonly event code and third experiment ID.
    if events.shape[1] >= 2:
        print("\nUnique values/counts in event column 1:")
        for val, cnt in safe_unique_counts(events[:, 1]):
            print(f"  {val}: {cnt}")

    if events.shape[1] >= 3:
        print("\nUnique values/counts in event column 2:")
        for val, cnt in safe_unique_counts(events[:, 2]):
            print(f"  {val}: {cnt}")

    event_ts = None
    if event_ts_col is not None:
        event_ts = events[:, event_ts_col].astype(np.float64)
        event_ts = event_ts[np.isfinite(event_ts)]

        print("\nEvent timestamp-like range:")
        print("  min:", event_ts.min())
        print("  max:", event_ts.max())
        print("  span:", event_ts.max() - event_ts.min())

    # --------------------------------------------------------
    # Compare clocks
    # --------------------------------------------------------
    banner("SPIKE vs EVENT CLOCK DIAGNOSTICS")

    if spike_ranges and event_ts is not None and event_ts.size:
        smin = min(x[1] for x in spike_ranges)
        smax = max(x[2] for x in spike_ranges)
        emin = event_ts.min()
        emax = event_ts.max()

        sspan = smax - smin
        espan = emax - emin

        print(f"spike range: [{smin:.12g}, {smax:.12g}]")
        print(f"event range: [{emin:.12g}, {emax:.12g}]")
        print(f"spike span : {sspan:.12g}")
        print(f"event span : {espan:.12g}")

        if espan != 0:
            print(f"span ratio spike/event = {sspan/espan:.12g}")

        if emin != 0:
            print(f"first-value ratio spike/event = {smin/emin:.12g}")

        print(f"simple offset (spike_min - event_min) = {smin-emin:.12g}")

        print("\nInterpretation:")
        print(
            "  If the spans are similar but absolute ranges differ, an offset may be involved.\n"
            "  If the span ratio is near a simple constant (e.g. 1e3, 1e4, 1e6), a unit/clock-scale\n"
            "  conversion may be involved. Do NOT apply either automatically; use the release code."
        )

    # --------------------------------------------------------
    # brainArea.mat
    # --------------------------------------------------------
    banner("BRAIN AREA METADATA")

    brain_path = EVENTS_DIR / "brainArea.mat"

    if brain_path.exists():
        inspect_mat_file(brain_path)
    else:
        print("brainArea.mat not found.")

    # --------------------------------------------------------
    # Text logs in this block
    # --------------------------------------------------------
    banner("TEXT LOG FILES")

    txt_files = sorted(EVENTS_DIR.glob("*.txt"))
    print("TXT files:", [p.name for p in txt_files])

    for p in txt_files:
        print(f"\n--- {p.name}: first 20 non-empty lines ---")
        try:
            lines = p.read_text(errors="replace").splitlines()
        except Exception as exc:
            print("Could not read:", exc)
            continue

        shown = 0
        for line in lines:
            if line.strip():
                print(line[:240])
                shown += 1
                if shown >= 20:
                    break

    # --------------------------------------------------------
    # Search the LOCAL OFFICIAL MATLAB release code
    # --------------------------------------------------------
    banner("SEARCHING OFFICIAL LOCAL MATLAB CODE")

    search_terms = [
        "eventsRaw",
        "getTimestampsOfTrials",
        "spikes",
        "timestamp",
        "stimulus",
        "STIMULUS_ON",
        "QUESTION_ON",
        "determineRecogState",
        "newold",
    ]

    matlab_files = sorted(CODE_DIR.rglob("*.m"))
    print("MATLAB files found:", len(matlab_files))

    matched_files = 0

    for path in matlab_files:
        try:
            lines = path.read_text(errors="replace").splitlines()
        except Exception:
            continue

        hits = []
        for lineno, line in enumerate(lines, start=1):
            low = line.lower()
            if any(term.lower() in low for term in search_terms):
                hits.append((lineno, line.rstrip()))

        if hits:
            matched_files += 1
            print(f"\nFILE: {path.relative_to(ROOT)}")
            for lineno, line in hits[:MAX_CODE_MATCHES_PER_FILE]:
                print(f"  L{lineno}: {line[:260]}")

    print("\nMatched MATLAB files:", matched_files)

    banner("DONE")
    print(
        "Send the complete terminal output of this script back.\n"
        "Especially important sections:\n"
        "  - NEURON FILES\n"
        "  - EVENTS\n"
        "  - SPIKE vs EVENT CLOCK DIAGNOSTICS\n"
        "  - SEARCHING OFFICIAL LOCAL MATLAB CODE\n"
    )


if __name__ == "__main__":
    main()

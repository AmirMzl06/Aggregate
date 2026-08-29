import h5py
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

NWB_PATH = Path(
    "data/Area2_Bump/"
    "sub-Han_desc-train_behavior+ecephys.nwb"
)


# ============================================================
# HELPERS
# ============================================================

def decode_value(x):
    """Decode bytes/string-like NWB values for display."""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")

    if isinstance(x, np.bytes_):
        return x.astype(str)

    return x


def get_timeseries_bin_ms(group):
    """
    Infer sampling/bin size of an NWB TimeSeries.

    NWB usually stores:
        starting_time.attrs['rate']

    or explicit:
        timestamps
    """

    # --------------------------------------------------------
    # Explicit timestamps
    # --------------------------------------------------------
    if "timestamps" in group:

        timestamps = group["timestamps"]

        if len(timestamps) >= 2:
            # Don't load the whole timestamps array
            n = min(len(timestamps), 1000)
            ts = np.asarray(timestamps[:n], dtype=float)

            dt = np.nanmedian(np.diff(ts))

            if np.isfinite(dt) and dt > 0:
                return dt * 1000.0

    # --------------------------------------------------------
    # starting_time + rate
    # --------------------------------------------------------
    if "starting_time" in group:

        starting_time = group["starting_time"]

        if "rate" in starting_time.attrs:
            rate = float(starting_time.attrs["rate"])

            if rate > 0:
                return 1000.0 / rate

    # --------------------------------------------------------
    # Occasionally rate may appear as an attribute elsewhere
    # --------------------------------------------------------
    if "rate" in group.attrs:
        rate = float(group.attrs["rate"])

        if rate > 0:
            return 1000.0 / rate

    return None


def find_data_groups(h5file):
    """
    Find HDF5 groups that look like NWB TimeSeries:
    group/data
    """

    found = []

    def visitor(name, obj):

        if not isinstance(obj, h5py.Group):
            return

        if "data" not in obj:
            return

        data = obj["data"]

        if not isinstance(data, h5py.Dataset):
            return

        found.append(
            {
                "path": name,
                "shape": data.shape,
                "dtype": str(data.dtype),
                "bin_ms": get_timeseries_bin_ms(obj),
            }
        )

    h5file.visititems(visitor)

    return found


def find_spikes_counts(h5file):
    """
    Locate NeuroTask spikes_counts TimeSeries.
    """

    candidates = []

    def visitor(name, obj):

        if not isinstance(obj, h5py.Group):
            return

        if name.split("/")[-1] != "spikes_counts":
            return

        if "data" not in obj:
            return

        candidates.append(name)

    h5file.visititems(visitor)

    if len(candidates) == 0:
        raise RuntimeError(
            "Could not find an NWB group named 'spikes_counts'."
        )

    print("\nspikes_counts candidates:")

    for path in candidates:
        print("  ", path)

    # Prefer processing/spikes
    for path in candidates:
        if "processing/spikes" in path:
            return path

    return candidates[0]


def print_sample(dataset, n=5):
    """
    Print a tiny sample without loading the complete dataset.
    """

    if dataset.ndim == 0:
        try:
            print("      sample:", decode_value(dataset[()]))
        except Exception:
            pass
        return

    if dataset.shape[0] == 0:
        return

    n = min(n, dataset.shape[0])

    try:
        arr = dataset[:n]

        if arr.ndim == 1:
            arr = [
                decode_value(x)
                for x in arr
            ]

        print("      sample:", arr)

    except Exception as exc:
        print("      sample could not be read:", exc)


# ============================================================
# MAIN
# ============================================================

print("=" * 100)
print("AREA2_BUMP DIRECT NWB INSPECTION")
print("=" * 100)

print("\nFile:")
print(NWB_PATH)

if not NWB_PATH.exists():
    raise FileNotFoundError(NWB_PATH)

print(
    f"Size: {NWB_PATH.stat().st_size / 1024**3:.3f} GB"
)


with h5py.File(NWB_PATH, "r") as f:

    # ========================================================
    # TOP LEVEL
    # ========================================================

    print("\n" + "=" * 100)
    print("TOP-LEVEL NWB GROUPS")
    print("=" * 100)

    for name in f.keys():
        print(name)

    # ========================================================
    # FIND SPIKES
    # ========================================================

    spikes_path = find_spikes_counts(f)
    spikes_group = f[spikes_path]
    spikes_ds = spikes_group["data"]

    print("\n" + "=" * 100)
    print("NEURAL DATA")
    print("=" * 100)

    print("path       :", spikes_path)
    print("shape      :", spikes_ds.shape)
    print("dtype      :", spikes_ds.dtype)

    bin_ms = get_timeseries_bin_ms(
        spikes_group
    )

    print("bin size   :", bin_ms, "ms")

    if spikes_ds.ndim != 2:
        raise RuntimeError(
            f"Expected spikes_counts to be 2D, "
            f"got shape {spikes_ds.shape}"
        )

    n_time = spikes_ds.shape[0]
    n_neurons = spikes_ds.shape[1]

    print("time bins  :", n_time)
    print("neurons    :", n_neurons)

    if bin_ms is not None:
        duration_sec = (
            n_time * bin_ms / 1000.0
        )

        print(
            "duration   :",
            f"{duration_sec:.2f} sec"
        )

        print(
            "duration   :",
            f"{duration_sec / 60:.2f} min"
        )

    # Tiny neural sample
    sample_rows = min(5, n_time)
    sample_neurons = min(10, n_neurons)

    print(
        f"\nFirst {sample_rows} bins × "
        f"first {sample_neurons} neurons:"
    )

    print(
        spikes_ds[
            :sample_rows,
            :sample_neurons
        ]
    )

    # ========================================================
    # FIND ALL TIMESERIES-LIKE DATA
    # ========================================================

    series = find_data_groups(f)

    print("\n" + "=" * 100)
    print("ALL DATA/TIMESERIES GROUPS")
    print("=" * 100)

    for item in series:

        print(
            f"{item['path']:70s} "
            f"shape={str(item['shape']):20s} "
            f"dtype={item['dtype']:12s} "
            f"bin_ms={item['bin_ms']}"
        )

    # ========================================================
    # FIND SIGNALS ALIGNED EXACTLY TO NEURAL TIMELINE
    # ========================================================

    aligned = []

    for item in series:

        path = item["path"]

        if path == spikes_path:
            continue

        shape = item["shape"]

        if len(shape) == 0:
            continue

        if shape[0] == n_time:
            aligned.append(item)

    print("\n" + "=" * 100)
    print("SIGNALS ALIGNED WITH NEURAL TIMELINE")
    print("=" * 100)

    print(
        "Number of aligned signals:",
        len(aligned)
    )

    for item in aligned:

        print("\n" + "-" * 100)

        print("path :", item["path"])
        print("shape:", item["shape"])
        print("dtype:", item["dtype"])
        print("bin  :", item["bin_ms"], "ms")

        ds = f[item["path"]]["data"]

        print_sample(ds, n=5)

    # ========================================================
    # INTERVAL TABLES / TRIALS
    # ========================================================

    print("\n" + "=" * 100)
    print("INTERVALS / TRIAL TABLES")
    print("=" * 100)

    if "intervals" not in f:

        print("No /intervals group found.")

    else:

        intervals = f["intervals"]

        for table_name in intervals.keys():

            table = intervals[table_name]

            print("\nTABLE:", table_name)

            if isinstance(table, h5py.Group):

                for col in table.keys():

                    obj = table[col]

                    if isinstance(
                        obj,
                        h5py.Dataset
                    ):

                        print(
                            f"  {col:30s} "
                            f"shape={str(obj.shape):15s} "
                            f"dtype={obj.dtype}"
                        )

                # Trial count
                if "id" in table:
                    print(
                        "  rows:",
                        len(table["id"])
                    )

                elif "start_time" in table:
                    print(
                        "  rows:",
                        len(table["start_time"])
                    )

    # ========================================================
    # UNITS TABLE
    # ========================================================

    print("\n" + "=" * 100)
    print("UNITS TABLE")
    print("=" * 100)

    if "units" in f:

        units = f["units"]

        if "id" in units:
            print(
                "units/id rows:",
                len(units["id"])
            )

        print("columns:")

        for col in units.keys():

            obj = units[col]

            if isinstance(
                obj,
                h5py.Dataset
            ):

                print(
                    f"  {col:30s} "
                    f"shape={str(obj.shape):20s} "
                    f"dtype={obj.dtype}"
                )

    else:
        print(
            "No top-level /units table."
        )


print("\n" + "=" * 100)
print("DONE")
print("=" * 100)

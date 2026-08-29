import h5py
import numpy as np
from pathlib import Path


NWB_PATH = Path(
    "data/Area2_Bump/"
    "sub-Han_desc-train_behavior+ecephys.nwb"
)


def get_rate_and_dt(group):
    """
    Return (rate_hz, dt_ms) if sampling info exists.
    """

    # Explicit timestamps
    if "timestamps" in group:
        ts = group["timestamps"]

        if len(ts) >= 2:
            n = min(1000, len(ts))
            x = np.asarray(ts[:n], dtype=float)

            dt = np.nanmedian(np.diff(x))

            if np.isfinite(dt) and dt > 0:
                return 1.0 / dt, dt * 1000.0

    # NWB starting_time + rate
    if "starting_time" in group:
        st = group["starting_time"]

        if "rate" in st.attrs:
            rate = float(st.attrs["rate"])

            if rate > 0:
                return rate, 1000.0 / rate

    return None, None


def inspect_timeseries(root, root_name):
    """
    Find groups containing a `data` dataset.
    """

    print("\n" + "=" * 100)
    print(root_name)
    print("=" * 100)

    found = []

    def visitor(name, obj):

        if not isinstance(obj, h5py.Group):
            return

        if "data" not in obj:
            return

        ds = obj["data"]

        if not isinstance(ds, h5py.Dataset):
            return

        rate, dt_ms = get_rate_and_dt(obj)

        full_path = f"{root.name}/{name}"

        found.append(
            (
                full_path,
                ds.shape,
                ds.dtype,
                rate,
                dt_ms,
            )
        )

    root.visititems(visitor)

    for path, shape, dtype, rate, dt_ms in found:

        print("\nPATH:")
        print(" ", path)

        print(" shape :", shape)
        print(" dtype :", dtype)

        if rate is not None:
            print(f" rate  : {rate:.6f} Hz")
            print(f" dt    : {dt_ms:.6f} ms")

    return found


print("=" * 100)
print("AREA2_BUMP NWB INSPECTOR")
print("=" * 100)

print("File:", NWB_PATH)
print(
    "Size:",
    f"{NWB_PATH.stat().st_size / 1024**3:.3f} GB"
)


with h5py.File(NWB_PATH, "r") as f:

    # ========================================================
    # UNITS
    # ========================================================

    print("\n" + "=" * 100)
    print("UNITS / SPIKES")
    print("=" * 100)

    if "units" not in f:

        print("No /units table found.")

    else:

        units = f["units"]

        print("\nColumns:")

        for key in units.keys():

            obj = units[key]

            if isinstance(obj, h5py.Dataset):

                print(
                    f"{key:35s}",
                    f"shape={str(obj.shape):20s}",
                    f"dtype={obj.dtype}"
                )

        # ----------------------------------------------------
        # Number of units
        # ----------------------------------------------------

        if "id" in units:

            n_units = len(units["id"])

            print("\nNumber of units:", n_units)

            print(
                "Unit IDs:",
                units["id"][:min(20, n_units)]
            )

        # ----------------------------------------------------
        # Standard NWB ragged spike times
        # ----------------------------------------------------

        if (
            "spike_times" in units
            and
            "spike_times_index" in units
        ):

            spike_times = units["spike_times"]
            spike_index = units["spike_times_index"][:]

            print("\nSpike times found.")

            print(
                "Total spikes:",
                len(spike_times)
            )

            print(
                "spike_times shape:",
                spike_times.shape
            )

            print(
                "spike_times_index shape:",
                spike_index.shape
            )

            # number of spikes per neuron
            previous = 0
            counts = []

            first_spike = np.inf
            last_spike = -np.inf

            for end in spike_index:

                end = int(end)

                counts.append(
                    end - previous
                )

                if end > previous:

                    unit_spikes = spike_times[
                        previous:end
                    ]

                    first_spike = min(
                        first_spike,
                        float(unit_spikes[0])
                    )

                    last_spike = max(
                        last_spike,
                        float(unit_spikes[-1])
                    )

                previous = end

            counts = np.asarray(counts)

            print(
                "\nSpikes per unit:"
            )

            print(
                " min   :",
                counts.min()
            )

            print(
                " mean  :",
                counts.mean()
            )

            print(
                " median:",
                np.median(counts)
            )

            print(
                " max   :",
                counts.max()
            )

            print(
                "\nSpike time range:"
            )

            print(
                " first:",
                first_spike,
                "sec"
            )

            print(
                " last :",
                last_spike,
                "sec"
            )

            print(
                " duration:",
                (last_spike - first_spike) / 60,
                "min"
            )

    # ========================================================
    # PROCESSING
    # ========================================================

    if "processing" in f:

        processing_series = inspect_timeseries(
            f["processing"],
            "PROCESSING TIMESERIES"
        )

    # ========================================================
    # ACQUISITION
    # ========================================================

    if "acquisition" in f:

        acquisition_series = inspect_timeseries(
            f["acquisition"],
            "ACQUISITION TIMESERIES"
        )

    # ========================================================
    # TRIAL / INTERVAL TABLES
    # ========================================================

    print("\n" + "=" * 100)
    print("INTERVAL TABLES")
    print("=" * 100)

    if "intervals" in f:

        intervals = f["intervals"]

        for table_name in intervals.keys():

            table = intervals[table_name]

            print("\nTABLE:", table_name)

            if "id" in table:
                print(
                    "rows:",
                    len(table["id"])
                )

            elif "start_time" in table:
                print(
                    "rows:",
                    len(table["start_time"])
                )

            print("columns:")

            for key in table.keys():

                obj = table[key]

                if isinstance(
                    obj,
                    h5py.Dataset
                ):

                    print(
                        f"  {key:35s}"
                        f" shape={str(obj.shape):20s}"
                        f" dtype={obj.dtype}"
                    )


print("\n" + "=" * 100)
print("DONE")
print("=" * 100)

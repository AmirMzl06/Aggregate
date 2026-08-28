import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

SESSION_ID = 1104058216
NWB_PATH = f"data/AllenVBN/ecephys_sessions/ecephys_session_{SESSION_ID}.nwb"
UNITS_CSV = "data/units.csv"
SCORE_CSV = f"AllenVBN_Jacobian_Plots_{SESSION_ID}/Neuron_Jacobian_Scores_{SESSION_ID}.csv"
OUT = f"AllenVBN_Jacobian_Plots_{SESSION_ID}"
os.makedirs(OUT, exist_ok=True)

PRESENCE_RATIO_MIN = 0.90
ISI_VIOLATIONS_MAX = 0.50
AMPLITUDE_CUTOFF_MAX = 0.10
TOP_K_LABEL = 10
DPI = 350

def load_scores():
    print("\n" + "=" * 100)
    print("LOADING JACOBIAN SCORES")
    print("=" * 100)
    scores = pd.read_csv(SCORE_CSV)
    required = ["neuron_index", "clean_jf_score", "acorn_jf_score", "clean_rank", "acorn_rank"]
    for col in required:
        if col not in scores.columns:
            raise RuntimeError(f"Missing column in score CSV: {col}")
    scores = scores.sort_values("neuron_index").reset_index(drop=True)
    print("Score rows:", len(scores))
    print("\nColumns:")
    print(scores.columns.tolist())
    print("\nCLEAN score range:")
    print(scores["clean_jf_score"].min(), "->", scores["clean_jf_score"].max())
    print("\nACORN score range:")
    print(scores["acorn_jf_score"].min(), "->", scores["acorn_jf_score"].max())
    return scores

def build_neuron_metadata(scores):
    print("\n" + "=" * 100)
    print("RECONSTRUCTING EXACT QC NEURON ORDER")
    print("=" * 100)
    units = pd.read_csv(UNITS_CSV)
    session_units = units[units["ecephys_session_id"] == SESSION_ID].copy()
    print("Total session units:", len(session_units))
    qc = session_units[(session_units["presence_ratio"] >= PRESENCE_RATIO_MIN) & (session_units["isi_violations"] <= ISI_VIOLATIONS_MAX) & (session_units["amplitude_cutoff"] <= AMPLITUDE_CUTOFF_MAX)].copy()
    print("QC units:", len(qc))
    qc_id_set = set(qc["unit_id"].astype(np.int64).tolist())
    with h5py.File(NWB_PATH, "r") as f:
        nwb_unit_ids = np.asarray(f["units/id"][:]).astype(np.int64)
    ordered_qc_unit_ids = np.array([unit_id for unit_id in nwb_unit_ids if int(unit_id) in qc_id_set], dtype=np.int64)
    print("QC units reconstructed in X order:", len(ordered_qc_unit_ids))
    if len(ordered_qc_unit_ids) != len(scores):
        raise RuntimeError(f"Mismatch: {len(ordered_qc_unit_ids)} QC units but {len(scores)} score rows.")
    mapping = pd.DataFrame({"neuron_index": np.arange(len(ordered_qc_unit_ids), dtype=np.int64), "unit_id": ordered_qc_unit_ids})
    meta = qc.set_index("unit_id").loc[ordered_qc_unit_ids].reset_index()
    meta.insert(0, "neuron_index", np.arange(len(meta), dtype=np.int64))
    merged = pd.merge(meta, scores, on="neuron_index", how="inner", validate="one_to_one")
    print("\nMerged rows:", len(merged))
    coordinate_cols = ["anterior_posterior_ccf_coordinate", "dorsal_ventral_ccf_coordinate", "left_right_ccf_coordinate"]
    for col in coordinate_cols:
        if col not in merged.columns:
            raise RuntimeError(f"Missing anatomical coordinate: {col}")
    n_before = len(merged)
    merged = merged.dropna(subset=coordinate_cols).copy()
    print("Units with valid CCF coordinates:", len(merged), "/", n_before)
    print("\nREGION COUNTS:")
    print(merged["structure_acronym"].value_counts().to_string())
    return merged

def compute_global_marker_sizes(merged):
    clean = merged["clean_jf_score"].to_numpy(dtype=float)
    acorn = merged["acorn_jf_score"].to_numpy(dtype=float)
    all_scores = np.concatenate([clean, acorn])
    positive = all_scores[all_scores > 0]
    if len(positive) == 0:
        raise RuntimeError("All Jacobian scores are zero.")
    global_min = float(positive.min())
    global_max = float(positive.max())
    log_min = np.log10(global_min)
    log_max = np.log10(global_max)
    def score_to_size(score):
        score = np.asarray(score, dtype=float)
        score = np.maximum(score, global_min)
        log_score = np.log10(score)
        if log_max == log_min:
            normalized = np.ones_like(log_score)
        else:
            normalized = (log_score - log_min) / (log_max - log_min)
        size = 15.0 + 180.0 * normalized
        return size
    return score_to_size, global_min, global_max

def configure_3d_axes(ax, merged):
    x = merged["left_right_ccf_coordinate"].to_numpy()
    y = merged["anterior_posterior_ccf_coordinate"].to_numpy()
    z = merged["dorsal_ventral_ccf_coordinate"].to_numpy()
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    z_min, z_max = float(np.nanmin(z)), float(np.nanmax(z))
    dx = max(x_max - x_min, 1)
    dy = max(y_max - y_min, 1)
    dz = max(z_max - z_min, 1)
    ax.set_xlim(x_min - 0.05 * dx, x_max + 0.05 * dx)
    ax.set_ylim(y_min - 0.05 * dy, y_max + 0.05 * dy)
    ax.set_zlim(z_min - 0.05 * dz, z_max + 0.05 * dz)
    ax.set_xlabel("Left–Right CCF coordinate", fontsize=11, labelpad=12)
    ax.set_ylabel("Anterior–Posterior CCF coordinate", fontsize=11, labelpad=12)
    ax.set_zlabel("Dorsal–Ventral CCF coordinate", fontsize=11, labelpad=12)
    ax.invert_zaxis()
    try:
        ax.set_box_aspect((dx, dy, dz))
    except Exception:
        pass
    ax.view_init(elev=24, azim=-58)

def plot_brain_map(merged, score_column, rank_column, model_name, output_filename, score_to_size, global_vmin, global_vmax):
    print("\n" + "=" * 100)
    print(f"PLOTTING {model_name} BRAIN MAP")
    print("=" * 100)
    df = merged.copy()
    x = df["left_right_ccf_coordinate"].to_numpy(dtype=float)
    y = df["anterior_posterior_ccf_coordinate"].to_numpy(dtype=float)
    z = df["dorsal_ventral_ccf_coordinate"].to_numpy(dtype=float)
    scores = df[score_column].to_numpy(dtype=float)
    sizes = score_to_size(scores)
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(x, y, z, c=scores, s=sizes, cmap="viridis", norm=LogNorm(vmin=global_vmin, vmax=global_vmax), alpha=0.82, edgecolors="none", depthshade=True)
    top = df.sort_values(rank_column).head(TOP_K_LABEL).copy()
    top_x = top["left_right_ccf_coordinate"].to_numpy(dtype=float)
    top_y = top["anterior_posterior_ccf_coordinate"].to_numpy(dtype=float)
    top_z = top["dorsal_ventral_ccf_coordinate"].to_numpy(dtype=float)
    top_scores = top[score_column].to_numpy(dtype=float)
    top_sizes = score_to_size(top_scores) * 1.6
    ax.scatter(top_x, top_y, top_z, c=top_scores, s=top_sizes, cmap="viridis", norm=LogNorm(vmin=global_vmin, vmax=global_vmax), edgecolors="black", linewidths=1.4, alpha=1.0, depthshade=False)
    for _, row in top.iterrows():
        ax.text(row["left_right_ccf_coordinate"], row["anterior_posterior_ccf_coordinate"], row["dorsal_ventral_ccf_coordinate"], f" {int(row['neuron_index'])}", fontsize=9, fontweight="bold")
    configure_3d_axes(ax, merged)
    ax.set_title(f"{model_name}\nSpatial distribution of Forward-Jacobian neuron sensitivity\nAllen VBN session {SESSION_ID}", fontsize=16, pad=24)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.68, pad=0.08)
    cbar.set_label("Mean absolute Forward-Jacobian score\n(logarithmic color scale)", fontsize=11)
    fig.text(0.5, 0.025, f"All points = QC neurons with valid CCF coordinates. Point color and size represent {model_name} Jacobian sensitivity. Top-{TOP_K_LABEL} neurons are outlined and labeled by zero-based neuron index.", ha="center", fontsize=10)
    path = os.path.join(OUT, output_filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved:")
    print(path)
    print(f"\nTop-{TOP_K_LABEL} {model_name}:")
    print(top[["neuron_index", "unit_id", "structure_acronym", score_column, rank_column, "anterior_posterior_ccf_coordinate", "dorsal_ventral_ccf_coordinate", "left_right_ccf_coordinate"]].to_string(index=False))

def main():
    print("\n" + "=" * 100)
    print("ALLEN VBN 3D BRAIN JACOBIAN MAPS")
    print("=" * 100)
    print("Session:", SESSION_ID)
    scores = load_scores()
    merged = build_neuron_metadata(scores)
    score_to_size, global_vmin, global_vmax = compute_global_marker_sizes(merged)
    print("\n" + "=" * 100)
    print("GLOBAL JACOBIAN SCALE")
    print("=" * 100)
    print("vmin:", global_vmin)
    print("vmax:", global_vmax)
    print("\nIMPORTANT:")
    print("CLEAN and ACORN use the SAME color normalization.")
    print("Color scale is logarithmic.")
    plot_brain_map(merged, score_column="clean_jf_score", rank_column="clean_rank", model_name="CEBRA CLEAN", output_filename=f"BrainMap_CLEAN_{SESSION_ID}.png", score_to_size=score_to_size, global_vmin=global_vmin, global_vmax=global_vmax)
    plot_brain_map(merged, score_column="acorn_jf_score", rank_column="acorn_rank", model_name="ACORN", output_filename=f"BrainMap_ACORN_{SESSION_ID}.png", score_to_size=score_to_size, global_vmin=global_vmin, global_vmax=global_vmax)
    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print("\nSaved brain maps:")
    print(os.path.join(OUT, f"BrainMap_CLEAN_{SESSION_ID}.png"))
    print(os.path.join(OUT, f"BrainMap_ACORN_{SESSION_ID}.png"))

if __name__ == "__main__":
    main()

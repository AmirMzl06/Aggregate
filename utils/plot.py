import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_multiple_embeddings_interactive(
    embeddings,
    embedding_labels=None,
    idx_order=(0, 1, 2),
    markersize=8,        # 🔹 much bigger markers
    alpha=0.6,
    cmap="Viridis",
    titles=None,
    figsize=(2400, 1200),
    showlegend=False,
    discrete=False,
    margin=0.05,         # fraction of range for padding
    title_font_size=28,  # 🔹 big titles
):
    """
    Plot multiple embeddings in a 2x4 grid with larger points and bigger titles.
    """
    import torch
    embeddings = [e.detach().cpu().numpy() if isinstance(e, torch.Tensor) else e
                  for e in embeddings]

    if embedding_labels is None:
        embedding_labels = ["grey"] * len(embeddings)
    else:
        embedding_labels = [
            l.detach().cpu().numpy() if isinstance(l, torch.Tensor) else l
            for l in embedding_labels
        ]

    fig = make_subplots(
        rows=1,
        cols=4,
        specs=[[{"type": "scatter3d"}] * 4],
        subplot_titles=titles if titles is not None else ["" for _ in embeddings],
        horizontal_spacing=0.005,
        vertical_spacing=0.005
    )

    for i, (embedding, labels) in enumerate(zip(embeddings, embedding_labels)):
        row = 1
        col = i % 4 + 1
        idx1, idx2, idx3 = idx_order

        fig.add_trace(
            go.Scatter3d(
                x=embedding[:, idx1],
                y=embedding[:, idx2],
                z=embedding[:, idx3],
                mode="markers",
                marker=dict(
                    size=markersize,  # 🔹 bigger points
                    opacity=alpha,
                    color=labels,
                    colorscale=cmap,
                ),
                name=f"Embedding {i+1}"
            ),
            row=row, col=col
        )

        # Auto-zoom each subplot
        x_min, x_max = embedding[:, idx1].min(), embedding[:, idx1].max()
        y_min, y_max = embedding[:, idx2].min(), embedding[:, idx2].max()
        z_min, z_max = embedding[:, idx3].min(), embedding[:, idx3].max()

        x_pad = (x_max - x_min) * margin
        y_pad = (y_max - y_min) * margin
        z_pad = (z_max - z_min) * margin * 1.6

        scene_name = f"scene{i+1}"
        fig.update_layout(
            **{
                scene_name: dict(
                    xaxis=dict(range=[x_min - x_pad, x_max + x_pad], visible=False),
                    yaxis=dict(range=[y_min - y_pad, y_max + y_pad], visible=False),
                    zaxis=dict(range=[z_min - z_pad, z_max + z_pad], visible=False),
                    aspectmode="cube",
                )
            }
        )

    # 🔹 Update annotation font size for subplot titles
    if titles is not None:
        for ann in fig['layout']['annotations']:
            ann['font'] = dict(size=title_font_size)

    fig.update_layout(
        height=figsize[1],
        width=figsize[0],
        margin=dict(l=5, r=5, t=150, b=5),  # 🔹 increase top margin
        showlegend=showlegend if discrete else False,
    )

    return fig


# === plotting helper ===
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

def plot_multiple_embeddings_interactive4(
    embeddings,
    embedding_labels=None,
    idx_order=(0, 1, 2),
    markersize=8,
    alpha=0.6,
    cmap="Viridis",
    titles=None,
    figsize=(2400, 1200),
    showlegend=False,
    discrete=False,
    margin=0.05,
    title_font_size=44,     # 🔹 larger subplot titles
    ncols=4,                # 🔹 4 columns -> rows inferred
    row_group_labels=None,  # 🔹 {row_index_1_based: "Label", ...}
    group_label_font_size=34,
    hover_font_size=20,     # 🔹 hover label font
):
    """
    Plot multiple 3D embeddings in an auto-arranged grid (default ncols=4).
    Supports row-group labels (e.g., Clean vs Adversarial).
    """
    import torch

    # Convert tensors → numpy
    embeddings = [
        e.detach().cpu().numpy() if isinstance(e, torch.Tensor) else np.asarray(e)
        for e in embeddings
    ]

    if embedding_labels is None:
        embedding_labels = ["grey"] * len(embeddings)
    else:
        embedding_labels = [
            (l.detach().cpu().numpy() if isinstance(l, torch.Tensor) else l)
            for l in embedding_labels
        ]

    N = len(embeddings)
    ncols = max(1, int(ncols))
    nrows = math.ceil(N / ncols)

    if titles is None:
        titles = ["" for _ in range(N)]

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=[[{"type": "scatter3d"} for _ in range(ncols)] for _ in range(nrows)],
        subplot_titles=titles,
        horizontal_spacing=0.005,
        vertical_spacing=0.005,
    )

    idx1, idx2, idx3 = idx_order

    for i, (embedding, labels) in enumerate(zip(embeddings, embedding_labels)):
        row = i // ncols + 1
        col = i % ncols + 1

        fig.add_trace(
            go.Scatter3d(
                x=embedding[:, idx1],
                y=embedding[:, idx2],
                z=embedding[:, idx3],
                mode="markers",
                marker=dict(
                    size=markersize,
                    opacity=alpha,
                    color=labels,
                    colorscale=cmap,
                ),
                name=f"Embedding {i+1}",
                showlegend=showlegend if discrete else False,
            ),
            row=row, col=col
        )

        # Auto-zoom per subplot
        x_min, x_max = float(embedding[:, idx1].min()), float(embedding[:, idx1].max())
        y_min, y_max = float(embedding[:, idx2].min()), float(embedding[:, idx2].max())
        z_min, z_max = float(embedding[:, idx3].min()), float(embedding[:, idx3].max())

        x_pad = (x_max - x_min) * margin
        y_pad = (y_max - y_min) * margin
        z_pad = (z_max - z_min) * margin * 1.6

        scene_name = f"scene{i+1}"
        fig.update_layout(
            **{
                scene_name: dict(
                    xaxis=dict(range=[x_min - x_pad, x_max + x_pad], visible=False),
                    yaxis=dict(range=[y_min - y_pad, y_max + y_pad], visible=False),
                    zaxis=dict(range=[z_min - z_pad, z_max + z_pad], visible=False),
                    aspectmode="cube",
                )
            }
        )

    # 🔹 Bigger subplot title fonts (Plotly 5.x-safe)
    if fig.layout.annotations:
        fig.update_annotations(font_size=title_font_size)

    # 🔹 Optional row-group labels (e.g., Clean at row 1, Adversarial at row 3)
    if row_group_labels:
        # y position per row in paper coords: top row near 1.0, bottom row near 0.0
        for row_idx, label in row_group_labels.items():
            y = 1.0 - ((row_idx - 0.5) / nrows)
            fig.add_annotation(
                xref="paper", yref="paper",
                x=-0.02, y=y,
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(size=group_label_font_size),
                xanchor="right", yanchor="middle"
            )

    fig.update_layout(
        height=figsize[1],
        width=figsize[0],
        margin=dict(l=90, r=20, t=150, b=40),
        hoverlabel=dict(font_size=hover_font_size),  # 🔹 bigger hover text
    )

    return fig

"""CEBRA InfoNCE vs. CEBRA-MSE with the same Offset36 convolutional backbone.

Designed to be imported from a dataset-specific runner.

Expected inputs
---------------
X : numpy.ndarray or torch.Tensor, shape (T, N)
    T time bins, N neural features/channels.
y : numpy.ndarray or torch.Tensor, shape (T, K) or (T,)
    Continuous behavioral labels aligned to time bins.

The InfoNCE branch uses y only to SAMPLE positives (CEBRA supervised/time_delta).
The MSE branch uses y directly as the regression target.

This file intentionally contains no dataset-specific loading or splitting logic.
"""

from __future__ import annotations

import copy
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Load the exact local CEBRA fork.
# Expected layout:
#
#   your_project/
#   ├── cebra_vs_mse.py
#   └── CEBRA-original/
#       └── cebra/
#           └── __init__.py
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CEBRA_DIR = ROOT / "CEBRA-original"


def load_cebra_fork():
    """Import CEBRA only from the sibling CEBRA-original fork.

    This deliberately removes any previously imported ``cebra`` modules,
    prepends the local fork to ``sys.path``, and verifies that Python actually
    resolved CEBRA from that directory.
    """
    if not (CEBRA_DIR / "cebra" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"Expected CEBRA fork at {CEBRA_DIR}. "
            "Put CEBRA-original next to cebra_vs_mse.py."
        )

    # Important in notebooks / reused interpreters: do not keep another CEBRA
    # installation cached in sys.modules.
    for name in list(sys.modules):
        if name == "cebra" or name.startswith("cebra."):
            del sys.modules[name]

    fork_path = str(CEBRA_DIR)
    if fork_path in sys.path:
        sys.path.remove(fork_path)
    sys.path.insert(0, fork_path)

    import cebra as _cebra

    imported_from = Path(_cebra.__file__).resolve()
    if not imported_from.is_relative_to(CEBRA_DIR.resolve()):
        raise RuntimeError(f"Wrong CEBRA imported: {imported_from}")

    # Same fork check used in your existing scripts. It is not required by the
    # plain InfoNCE-vs-MSE trainer itself, but it guarantees that this is the
    # matching adversarial-capable fork rather than a pip-installed CEBRA.
    parameters = inspect.signature(_cebra.CEBRA.__init__).parameters
    required = ("adv_negative_grad_scale",)
    missing = [name for name in required if name not in parameters]
    if missing:
        raise RuntimeError(
            f"This fork lacks {missing}. Use the matching CEBRA-original fork."
        )

    print("Using CEBRA fork:", imported_from, flush=True)
    return _cebra


cebra = load_cebra_fork()

# Import submodules only AFTER selecting the local fork above.
from cebra.integrations.sklearn.dataset import SklearnDataset


@dataclass
class CompareConfig:
    # Same backbone for both objectives.
    model_name: str = "offset36-model-more-dropout"
    hidden_dim: int = 256
    embedding_dim: int = 48
    normalize_embedding: bool = True

    # Optimization.
    lr: float = 3e-4
    weight_decay: float = 0.0
    batch_size: int = 2048
    steps: int = 5000

    # CEBRA / InfoNCE.
    temperature: float = 0.4
    time_offset: int = 4
    conditional: str = "time_delta"  # label-conditioned CEBRA positives

    # MSE head.
    mse_head_hidden: Optional[int] = None  # None -> a single Linear layer

    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LabelStats:
    mean: torch.Tensor
    std: torch.Tensor

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean


def _as_float_tensor(x, device: str) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.float().to(device)


def standardize_labels(y, device: str = "cpu") -> tuple[torch.Tensor, LabelStats]:
    """Standardize each continuous target dimension.

    Use the SAME standardized labels for InfoNCE sampling and MSE training if you
    want the comparison to be as controlled as possible.
    """
    y = _as_float_tensor(y, device)
    if y.ndim == 1:
        y = y[:, None]
    mean = y.mean(dim=0, keepdim=True)
    std = y.std(dim=0, keepdim=True).clamp_min(1e-8)
    stats = LabelStats(mean=mean, std=std)
    return stats.normalize(y), stats


def build_offset36_encoder(
    num_neurons: int,
    cfg: CompareConfig,
) -> nn.Module:
    """Build the same Offset36 convolutional encoder used by the vendored CEBRA.

    Architecture in acorn-main's vendored CEBRA:
      Conv1d(N -> hidden, k=2)
      Dropout1d + GELU
      16 residual Conv1d(hidden -> hidden, k=3) blocks
      Conv1d(hidden -> embedding_dim, k=3)
      optional L2 normalization

    Receptive field: 36 time bins (Offset(18, 18)).
    """
    return cebra.models.init(
        cfg.model_name,
        num_neurons=num_neurons,
        num_units=cfg.hidden_dim,
        num_output=cfg.embedding_dim,
        normalize=cfg.normalize_embedding,
    )


class CEBRAMSE(nn.Module):
    """Same CEBRA convolutional encoder + a small regression head.

    MSE gradients flow through BOTH the head and the encoder, so this is not a
    frozen-CEBRA decoder. It is an end-to-end MSE-trained version of the same
    convolutional backbone.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        target_dim: int,
        head_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = encoder
        if head_hidden is None:
            self.head = nn.Linear(embedding_dim, target_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(embedding_dim, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, target_dim),
            )

    def forward_windows(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """windows: (B, N, 36) -> (prediction, embedding)."""
        z = self.encoder(windows)
        if z.ndim != 2:
            raise RuntimeError(f"Expected window embeddings (B,D), got {tuple(z.shape)}")
        pred = self.head(z)
        return pred, z

    def forward_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (T,N) -> predictions and embeddings aligned to all T bins."""
        z = encode_full_sequence(self.encoder, x)  # (T,D)
        return self.head(z), z


def build_paired_models(
    num_neurons: int,
    target_dim: int,
    cfg: CompareConfig,
) -> tuple[nn.Module, CEBRAMSE]:
    """Create InfoNCE and MSE branches from IDENTICAL initial encoder weights."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    info_encoder = build_offset36_encoder(num_neurons, cfg)
    mse_encoder = copy.deepcopy(info_encoder)
    mse_model = CEBRAMSE(
        mse_encoder,
        embedding_dim=cfg.embedding_dim,
        target_dim=target_dim,
        head_hidden=cfg.mse_head_hidden,
    )
    return info_encoder.to(cfg.device), mse_model.to(cfg.device)


def _build_cebra_dataset(X, y, encoder: nn.Module, device: str) -> SklearnDataset:
    X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
    y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    if y_np.ndim == 1:
        y_np = y_np[:, None]

    dataset = SklearnDataset(
        X_np.astype(np.float32),
        (y_np.astype(np.float32),),
        device=device,
    )
    dataset.offset = encoder.get_offset()
    dataset.trial_ids = None
    return dataset


def train_infonce(
    encoder: nn.Module,
    X,
    y,
    cfg: CompareConfig,
    *,
    verbose_every: int = 200,
) -> list[float]:
    """Train CEBRA with label-conditioned InfoNCE.

    y is NOT regressed directly. It defines the continuous index used by CEBRA's
    positive-pair sampler. With conditional='time_delta', positives are sampled
    from the behavioral geometry induced by changes over cfg.time_offset bins.
    """
    encoder.train()
    dataset = _build_cebra_dataset(X, y, encoder, cfg.device)

    loader = cebra.data.ContinuousDataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_steps=cfg.steps,
        conditional=cfg.conditional,
        time_offset=cfg.time_offset,
    )
    criterion = cebra.models.FixedCosineInfoNCE(cfg.temperature).to(cfg.device)
    optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    history: list[float] = []
    for step, batch in enumerate(loader):
        batch.to(cfg.device)
        optimizer.zero_grad(set_to_none=True)

        z_ref = encoder(batch.reference)
        z_pos = encoder(batch.positive)
        z_neg = encoder(batch.negative)
        loss, _, _ = criterion(z_ref, z_pos, z_neg)

        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))

        if verbose_every and (step % verbose_every == 0 or step == cfg.steps - 1):
            print(f"[InfoNCE] step={step:5d}/{cfg.steps} loss={history[-1]:.6f}")

    return history


def train_mse(
    model: CEBRAMSE,
    X,
    y,
    cfg: CompareConfig,
    *,
    verbose_every: int = 200,
) -> list[float]:
    """Train the SAME convolutional backbone end-to-end using only MSE.

    There is no positive/negative sampling and no InfoNCE in this branch.
    Each batch samples valid time-bin centers, extracts the exact 36-bin window
    expected by Offset36, predicts y_t, and backpropagates MSE through the head
    and the encoder.
    """
    model.train()
    X_t = _as_float_tensor(X, cfg.device)
    y_t = _as_float_tensor(y, cfg.device)
    if y_t.ndim == 1:
        y_t = y_t[:, None]

    dataset = _build_cebra_dataset(X_t, y_t, model.encoder, cfg.device)
    offset = model.encoder.get_offset()
    T = X_t.shape[0]

    # Valid center indices so the 36-bin window never needs edge clamping.
    low = int(offset.left)
    high_exclusive = int(T - offset.right + 1)
    if high_exclusive <= low:
        raise ValueError(
            f"Sequence too short for Offset36: T={T}, offset=({offset.left},{offset.right})"
        )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    g = torch.Generator(device=cfg.device)
    g.manual_seed(cfg.seed)

    history: list[float] = []
    for step in range(cfg.steps):
        idx = torch.randint(
            low=low,
            high=high_exclusive,
            size=(cfg.batch_size,),
            generator=g,
            device=cfg.device,
        )
        windows = dataset[idx]          # (B, N, 36)
        targets = y_t[idx]              # (B, K)

        optimizer.zero_grad(set_to_none=True)
        pred, _ = model.forward_windows(windows)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()

        history.append(float(loss.detach().cpu()))
        if verbose_every and (step % verbose_every == 0 or step == cfg.steps - 1):
            print(f"[MSE]     step={step:5d}/{cfg.steps} loss={history[-1]:.6f}")

    return history


@torch.no_grad()
def encode_full_sequence(encoder: nn.Module, X) -> torch.Tensor:
    """Encode every time bin while preserving sequence length.

    Replicate padding matches the pattern used by acorn-main's MonkeyEncoder:
      left = offset.left
      right padding = offset.right - 1
    """
    device = next(encoder.parameters()).device
    x = _as_float_tensor(X, str(device))
    if x.ndim != 2:
        raise ValueError(f"Expected X shape (T,N), got {tuple(x.shape)}")

    offset = encoder.get_offset()
    x = x.T.unsqueeze(0)  # (1,N,T)
    x = F.pad(x, (offset.left, offset.right - 1), mode="replicate")
    z = encoder(x)

    if z.ndim == 2:       # only happens for a one-step output
        z = z.unsqueeze(-1)
    z = z.squeeze(0).T    # (T,D)
    return z


class FrozenProbe(nn.Module):
    """A small MLP decoder for comparing frozen representations fairly."""

    def __init__(self, embedding_dim: int, target_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def fit_frozen_probe(
    encoder: nn.Module,
    X,
    y,
    *,
    embedding_dim: int,
    hidden_dim: int = 64,
    lr: float = 3e-4,
    steps: int = 1000,
    batch_size: int = 2048,
    seed: int = 0,
) -> FrozenProbe:
    """Freeze an encoder and fit the SAME MLP probe on top of its embeddings.

    This is useful if the scientific question is specifically representation
    quality: train one probe on InfoNCE embeddings and an identical fresh probe
    on MSE embeddings.
    """
    device = next(encoder.parameters()).device
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        z = encode_full_sequence(encoder, X).detach()
    y_t = _as_float_tensor(y, str(device))
    if y_t.ndim == 1:
        y_t = y_t[:, None]

    probe = FrozenProbe(embedding_dim, y_t.shape[1], hidden_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    mse = nn.MSELoss()
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    for _ in range(steps):
        idx = torch.randint(0, z.shape[0], (batch_size,), generator=g, device=device)
        loss = mse(probe(z[idx]), y_t[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return probe


@torch.no_grad()
def predict_with_probe(encoder: nn.Module, probe: nn.Module, X) -> torch.Tensor:
    encoder.eval()
    probe.eval()
    return probe(encode_full_sequence(encoder, X))


@torch.no_grad()
def r2_per_dimension(y_true, y_pred) -> torch.Tensor:
    """R^2 for each target dimension."""
    yt = torch.as_tensor(y_true, dtype=torch.float32, device=y_pred.device)
    if yt.ndim == 1:
        yt = yt[:, None]
    ss_res = ((yt - y_pred) ** 2).sum(dim=0)
    ss_tot = ((yt - yt.mean(dim=0, keepdim=True)) ** 2).sum(dim=0).clamp_min(1e-12)
    return 1.0 - ss_res / ss_tot

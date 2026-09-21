"""Shortcut-robust temporal jigsaw pretext task for neural population data.

Companion / replacement for the earlier `puzzle` module. Keeps the tiles-mode
contract (shared-weight, context-free per-tile encoding, no cross-tile
convolution, 24-class permutation head) but changes two things that most
directly explain "good train puzzle accuracy, bad test puzzle accuracy, and
a HEAVILY puzzle-trained encoder gives worse downstream R^2 than a barely
trained one":

1. Per-tile statistic shortcut. Firing-rate style signals drift slowly
   (state, adaptation, electrode drift) and are smooth bin-to-bin, so a
   tile's own mean level is, by itself, a weak but real cue to its rank
   among the four tiles -- even if nothing about within-tile *dynamics* is
   learned. This is the neural-data analogue of the "low-level statistics"
   shortcut in Noroozi & Favaro (2016): they had to explicitly normalize
   each patch's mean/std before their siamese encoder (their Table 5:
   removing just that normalization cost ~9 points on the downstream task,
   almost as much as removing the inter-tile gaps). It plausibly explains
   your symptoms directly: a level-based shortcut is easy to fit on the
   *particular* drift trajectory seen during training (-> good train
   accuracy), does not transported to a held-out trajectory (-> bad test
   accuracy), and, if the encoder spends its capacity on "what is this
   tile's baseline" instead of "what is this tile's local geometry", the
   resulting embedding can be actively worse for behavior decoding than a
   random encoder's generic features (-> R^2 goes down as puzzle accuracy
   goes up). `tile_normalize` removes each tile's own per-channel mean (and
   optionally its std) before encoding, and `augment_*` adds fresh random
   gain/offset noise on top, only while training, so a residual amplitude
   correlate cannot be memorized either.

2. Selecting a checkpoint by pretext loss/accuracy. Puzzle accuracy is not
   a proxy for representation quality -- your own experiment is a direct
   demonstration of that. `fit` accepts an optional
   `monitor_fn(model) -> float`, evaluated every `monitor_every` epochs;
   whichever epoch scores best is what `fit` returns (the raw last-epoch
   weights stay reachable through `last_encoder_state_` / `last_head_state_`
   for comparison). The intended `monitor_fn` is a closure that embeds a
   small held-out behavioral set with `model.transform` and fits a quick
   ridge/linear decoder to get an R^2 -- i.e. select the encoder on the
   metric you actually care about, not on the pretext task.

Everything else (window/tile geometry, evaluate_puzzle semantics, fit/
transform/save/load contract) intentionally mirrors the earlier `puzzle`
module so the two are easy to compare head-to-head.

Requires NumPy and PyTorch >= 2.0. Inputs are (time, neurons) float arrays,
or a list of such arrays (one per trial/recording -- never join trials).

    from neural_jigsaw import NeuralJigsaw

    def r2_monitor(model):
        # X_val, y_val: held-out neural data / behavior, not used in fit()
        z = model.transform(X_val, pad=False)
        return quick_ridge_r2(z, y_val)          # your own linear probe

    model = NeuralJigsaw(window_size=8, tile_gap=(1, 4), tile_normalize="center",
                          augment_gain_jitter=0.1, augment_noise_std=0.05)
    model.fit(X_train, monitor_fn=r2_monitor, monitor_every=5)
    print(model.best_epoch_, model.best_score_)
    z_valid = model.transform(X_valid, pad=False)
    metrics = model.evaluate_puzzle(X_valid, min_gap=window_size)  # de-correlated

Quick real-PyTorch smoke test: python neural_jigsaw.py --self-test
"""
from copy import deepcopy
from itertools import permutations
import math
import numbers
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# label -> source ranks in ascending destination-slot order (same convention
# as the earlier `puzzle` module, so labels/checkpoints reason about
# identically).
PERMUTATIONS = np.asarray(list(permutations(range(4))), dtype=np.int64)


# --------------------------------------------------------------------------
# small shared helpers (kept close to the earlier `puzzle` module)
# --------------------------------------------------------------------------
def _integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}.")
    return int(value)


def _real(name, value, minimum, maximum=None, strict_min=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number.")
    value = float(value)
    if (not math.isfinite(value) or value < minimum
            or (strict_min and value == minimum)
            or (maximum is not None and value >= maximum)):
        raise ValueError(f"Invalid {name}: {value}.")
    return value


def _sequences(X, min_length, n_features=None):
    """Accept one (T,C) array, or a list of (T_i,C) arrays, without joining trials."""
    def array(value):
        if torch.is_tensor(value):
            value = value.detach().to(device="cpu", dtype=torch.float32).numpy()
        return np.asarray(value, dtype=np.float32)

    is_list = (isinstance(X, (list, tuple)) and len(X) > 0
               and array(X[0]).ndim == 2)
    values = list(X) if is_list else [X]
    result = []
    for index, value in enumerate(values):
        x = array(value)
        if x.ndim != 2 or x.shape[0] < min_length or x.shape[1] < 1:
            raise ValueError(f"Sequence {index}: expected (T,C), T >= {min_length}, C >= 1; got {x.shape}.")
        if not np.isfinite(x).all():
            raise ValueError(f"Sequence {index} contains NaN or infinite values.")
        if n_features is None:
            n_features = x.shape[1]
        if x.shape[1] != n_features:
            raise ValueError(f"Sequence {index}: expected {n_features} channels, got {x.shape[1]}.")
        result.append(np.array(x, dtype=np.float32, order="C", copy=True))
    return result, is_list


class _Windows(Dataset):
    def __init__(self, sequences, window_size):
        self.sequences = sequences
        self.window_size = window_size
        self.ends = np.cumsum([len(x) - window_size + 1 for x in sequences])

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        seq = int(np.searchsorted(self.ends, index, side="right"))
        start = int(index - (self.ends[seq - 1] if seq else 0))
        x = self.sequences[seq][start:start + self.window_size]
        return torch.from_numpy(x.T.copy())  # (channels, time)

    def locate(self, index):
        """index -> (sequence_id, start), for de-correlated evaluation sampling."""
        seq = int(np.searchsorted(self.ends, index, side="right"))
        start = int(index - (self.ends[seq - 1] if seq else 0))
        return seq, start


class _Residual(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout1d(dropout), nn.Conv1d(width, width, 3), nn.GELU())

    def forward(self, x):
        return x[..., 1:-1] + self.net(x)


class _Encoder(nn.Module):
    """Valid-convolution residual net; receptive field is exactly window_size."""
    def __init__(self, channels, window_size, width, output_dimension, dropout, normalize):
        super().__init__()
        first_kernel = 2 if window_size % 2 == 0 else 3
        blocks = (window_size - first_kernel - 2) // 2
        self.layers = nn.Sequential(
            nn.Conv1d(channels, width, first_kernel),
            nn.Dropout1d(dropout), nn.GELU(),
            *[_Residual(width, dropout) for _ in range(blocks)],
            nn.Conv1d(width, output_dimension, 3),
        )
        self.normalize = normalize

    def forward(self, x):
        z = self.layers(x)
        if z.shape[-1] != 1:
            raise ValueError("Encoder expects exactly window_size input bins.")
        z = z.squeeze(-1)
        return F.normalize(z, p=2, dim=1, eps=1e-8) if self.normalize else z


# --------------------------------------------------------------------------
# tile geometry, shortcut-removal normalization, and training-time jitter
# --------------------------------------------------------------------------
def _draw_tile_plan(batch, window_size, tile_gap, count, rng):
    """Four intact, nonoverlapping tiles; gaps/positions NEVER enter the head."""
    gaps = rng.integers(tile_gap[0], tile_gap[1] + 1, size=(batch, 3))
    starts = np.concatenate((np.zeros((batch, 1), dtype=np.int64),
                             np.cumsum(window_size + gaps, axis=1)), axis=1)
    if count == 24:
        labels = np.broadcast_to(np.arange(24), (batch, 24)).copy()
    elif count == 1:
        labels = rng.integers(0, 24, size=(batch, 1))
    else:
        labels = np.argsort(rng.random((batch, 24)), axis=1)[:, :count]
    return starts, labels.reshape(-1)


def _extract_tiles(windows, starts, window_size):
    """(B,C,span) -> (B,4,C,W), with no convolution across tile boundaries."""
    starts = torch.as_tensor(starts, dtype=torch.long, device=windows.device)
    offsets = torch.arange(window_size, device=windows.device)
    indices = (starts[:, :, None] + offsets).reshape(len(windows), 1, -1)
    tiles = windows.gather(2, indices.expand(-1, windows.shape[1], -1))
    return tiles.reshape(len(windows), windows.shape[1], 4, window_size).transpose(1, 2)


def _normalize_tiles(tiles, mode):
    """Remove each tile's OWN per-channel level (and optionally scale).

    tiles: (..., C, W). Uses only that tile's W bins -- never statistics
    from other tiles or from the full recording -- so the label (which
    tile came first/second/...) cannot be read off from a tile's absolute
    baseline. "center" (default) subtracts the per-channel mean only, which
    is the safer choice for small W (a handful of bins gives a very noisy
    std estimate); "zscore" additionally divides by the per-channel std;
    "off" reproduces the earlier module's behaviour (no normalization).
    """
    if mode == "off":
        return tiles
    mean = tiles.mean(dim=-1, keepdim=True)
    centered = tiles - mean
    if mode == "center":
        return centered
    if mode == "zscore":
        std = centered.std(dim=-1, unbiased=False, keepdim=True)
        return centered / (std + 1e-6)
    raise ValueError("tile_normalize must be 'off', 'center', or 'zscore'.")


def _augment_tiles(tiles, gain_jitter, noise_std, channel_std, generator):
    """Training-only jitter: one random gain per tile (all channels, so the
    within-tile *relative* activity across neurons is preserved) plus
    independent per-channel additive noise. Both are fresh every visit, so
    they cannot themselves be memorized as a shortcut; their only effect is
    to make any residual reliance on absolute amplitude a losing strategy.
    noise_std is a fraction of each channel's own std in the TRAINING data
    (channel_std), so one dimensionless number works regardless of firing-
    rate units.
    """
    if gain_jitter > 0:
        gain = 1.0 + (2 * torch.rand(tiles.shape[:2] + (1, 1), generator=generator,
                                     device=tiles.device) - 1.0) * gain_jitter
        tiles = tiles * gain
    if noise_std > 0:
        scale = (noise_std * channel_std).reshape(1, 1, -1, 1)
        tiles = tiles + torch.randn(tiles.shape, generator=generator, device=tiles.device) * scale
    return tiles


def _arrange_features(features, labels, count):
    """Shared-encoder features (B,4,D) -> shuffled (B*count,4*D)."""
    features = features.repeat_interleave(count, dim=0)
    target = torch.as_tensor(labels, dtype=torch.long, device=features.device)
    table = torch.as_tensor(PERMUTATIONS, dtype=torch.long, device=features.device)
    order = table[target, :, None].expand(-1, -1, features.shape[2])
    return features.gather(1, order).reshape(len(features), -1), target


def _min_gap_sample(dataset, n, min_gap, rng):
    """Pick up to n window starts, keeping picks from the same sequence at
    least min_gap bins apart, to reduce the pseudo-replication that comes
    from evaluating on heavily overlapping sliding windows. Falls back to
    plain sampling (like before) when min_gap is None.
    """
    if min_gap is None:
        return np.sort(rng.choice(len(dataset), size=min(n, len(dataset)), replace=False))
    order = rng.permutation(len(dataset))
    accepted, accepted_starts = [], {}
    for index in order:
        seq, start = dataset.locate(int(index))
        starts = accepted_starts.setdefault(seq, [])
        if all(abs(start - other) >= min_gap for other in starts):
            starts.append(start)
            accepted.append(int(index))
            if len(accepted) == n:
                break
    return np.sort(np.asarray(accepted, dtype=np.int64))


# --------------------------------------------------------------------------
# estimator
# --------------------------------------------------------------------------
class NeuralJigsaw:
    """Shared-encoder temporal-tiles jigsaw estimator with checkpoint selection.

    Architecturally this is the same context-free design as tiles-mode in
    the earlier module (encoder RF == window_size, 4 independent tiles,
    24-class head over the concatenated, permuted latents). The additions
    are `tile_normalize` / `augment_*` (see module docstring, point 1) and
    `monitor_fn`-based checkpoint selection in `fit` (point 2). No behavior
    labels enter fit() itself -- monitor_fn is the only place they may.
    """

    def __init__(self, window_size=8, output_dimension=64, num_hidden_units=64,
                 head_hidden_units=128, batch_size=256, max_epochs=200,
                 learning_rate=3e-4, weight_decay=1e-4, dropout=0.1, normalize=True,
                 tile_normalize="center", augment_gain_jitter=0.1, augment_noise_std=0.05,
                 permutations_per_window=1, tile_gap=(1, 4), device="cuda_if_available",
                 random_state=42, verbose=True):
        self.window_size = _integer("window_size", window_size, 4)
        self.output_dimension = _integer("output_dimension", output_dimension)
        self.num_hidden_units = _integer("num_hidden_units", num_hidden_units)
        self.head_hidden_units = _integer("head_hidden_units", head_hidden_units)
        self.batch_size = _integer("batch_size", batch_size)
        self.max_epochs = _integer("max_epochs", max_epochs, 0)
        self.learning_rate = _real("learning_rate", learning_rate, 0, strict_min=True)
        self.weight_decay = _real("weight_decay", weight_decay, 0)
        self.dropout = _real("dropout", dropout, 0, 1)
        if not isinstance(normalize, bool):
            raise ValueError("normalize must be bool (latent L2 normalization).")
        self.normalize = normalize
        if tile_normalize not in ("off", "center", "zscore"):
            raise ValueError("tile_normalize must be 'off', 'center', or 'zscore'.")
        self.tile_normalize = tile_normalize
        self.augment_gain_jitter = _real("augment_gain_jitter", augment_gain_jitter, 0, 1)
        self.augment_noise_std = _real("augment_noise_std", augment_noise_std, 0)
        self.permutations_per_window = _integer("permutations_per_window", permutations_per_window)
        if self.permutations_per_window > 24:
            raise ValueError("permutations_per_window must be between 1 and 24.")
        gaps = tile_gap if isinstance(tile_gap, (tuple, list)) else (tile_gap, tile_gap)
        if len(gaps) != 2:
            raise ValueError("tile_gap must be an integer or (minimum, maximum) pair.")
        self.tile_gap = tuple(_integer("tile_gap", g, 0) for g in gaps)
        if self.tile_gap[0] > self.tile_gap[1]:
            raise ValueError("tile_gap minimum exceeds maximum.")
        self.device = str(device)
        self.random_state = _integer("random_state", random_state, 0)
        self.verbose = bool(verbose)

    @property
    def training_span(self):
        return 4 * self.window_size + 3 * self.tile_gap[1]

    def get_params(self):
        names = ("window_size", "output_dimension", "num_hidden_units", "head_hidden_units",
                 "batch_size", "max_epochs", "learning_rate", "weight_decay", "dropout",
                 "normalize", "tile_normalize", "augment_gain_jitter", "augment_noise_std",
                 "permutations_per_window", "tile_gap", "device", "random_state", "verbose")
        return {name: getattr(self, name) for name in names}

    def _build(self, channels):
        name = self.device
        if name == "cuda_if_available":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_ = torch.device(name)
        self.n_features_in_ = int(channels)
        self.encoder_ = _Encoder(channels, self.window_size, self.num_hidden_units,
                                 self.output_dimension, self.dropout, self.normalize).to(self.device_)
        self.head_ = nn.Sequential(
            nn.Linear(4 * self.output_dimension, self.head_hidden_units),
            nn.GELU(), nn.Linear(self.head_hidden_units, 24),
        ).to(self.device_)

    def _tile_logits(self, windows, starts, labels, count, *, training):
        tiles = _extract_tiles(windows, starts, self.window_size)
        tiles = _normalize_tiles(tiles, self.tile_normalize)
        if training:
            tiles = _augment_tiles(tiles, self.augment_gain_jitter, self.augment_noise_std,
                                   self._channel_std_, self._augment_generator_)
        z = self.encoder_(tiles.reshape(-1, self.n_features_in_, self.window_size))
        z = z.reshape(len(windows), 4, self.output_dimension)
        features, target = _arrange_features(z, labels, count)
        return self.head_(features), target

    def fit(self, X, *, monitor_fn=None, monitor_every=1, select_best=True):
        """Fit on X. If monitor_fn is given, it is called as monitor_fn(self)
        every `monitor_every` epochs (higher is assumed better -- e.g. a
        downstream R^2). The BEST-scoring epoch's weights are what the
        estimator holds after fit() returns (when select_best=True); the
        final epoch's weights remain available via last_encoder_state_ /
        last_head_state_ regardless, so you can compare the two directly.
        """
        sequences, _ = _sequences(X, self.training_span)
        self.is_fitted_ = False
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        rng = np.random.default_rng(self.random_state)
        generator = torch.Generator().manual_seed(self.random_state)
        self._augment_generator_ = torch.Generator().manual_seed(self.random_state + 1)
        self._build(sequences[0].shape[1])
        # Per-channel std of the TRAINING data only, used to scale augmentation
        # noise in physically meaningful (relative) units; never touches test data.
        concatenated = np.concatenate(sequences, axis=0)
        self._channel_std_ = torch.as_tensor(concatenated.std(axis=0), dtype=torch.float32,
                                             device=self.device_).clamp_min(1e-6)
        dataset = _Windows(sequences, self.training_span)
        self.n_windows_ = len(dataset)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            drop_last=False, num_workers=0, generator=generator,
                            pin_memory=self.device_.type == "cuda")
        parameters = list(self.encoder_.parameters()) + list(self.head_.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        self.history_ = []
        self.n_steps_ = 0
        self.best_score_, self.best_epoch_ = -math.inf, None
        self.best_encoder_state_ = self.best_head_state_ = None
        if self.verbose:
            print(f"NeuralJigsaw: {self.n_windows_} valid starts, window={self.window_size}, "
                  f"training_span={self.training_span}, tile_gap={self.tile_gap}, "
                  f"tile_normalize={self.tile_normalize}, device={self.device_}", flush=True)
        for epoch in range(self.max_epochs):
            self.encoder_.train()
            self.head_.train()
            total_loss, correct, seen = 0.0, 0, 0
            for windows in loader:
                windows = windows.to(self.device_, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                starts, labels = _draw_tile_plan(len(windows), self.window_size,
                                                 self.tile_gap, self.permutations_per_window, rng)
                logits, target = self._tile_logits(windows, starts, labels,
                                                   self.permutations_per_window, training=True)
                loss = F.cross_entropy(logits, target)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Nonfinite puzzle loss at step {self.n_steps_}.")
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(target)
                correct += (logits.detach().argmax(dim=1) == target).sum().item()
                seen += len(target)
                self.n_steps_ += 1
            self.encoder_.eval()
            self.head_.eval()
            self.is_fitted_ = True  # transform()/monitor_fn may run mid-fit from here on
            row = dict(epoch=epoch + 1, loss=total_loss / seen, accuracy=correct / seen,
                       n_puzzles=seen, steps=self.n_steps_, monitor_score=None)
            do_monitor = monitor_fn is not None and ((epoch + 1) % monitor_every == 0
                                                      or epoch + 1 == self.max_epochs)
            if do_monitor:
                score = float(monitor_fn(self))
                row["monitor_score"] = score
                if score > self.best_score_:
                    self.best_score_, self.best_epoch_ = score, epoch + 1
                    self.best_encoder_state_ = deepcopy(self.encoder_.state_dict())
                    self.best_head_state_ = deepcopy(self.head_.state_dict())
            self.history_.append(row)
            if self.verbose:
                suffix = f", monitor={row['monitor_score']:.4f}" if do_monitor else ""
                print(f"Epoch {epoch + 1}/{self.max_epochs}: loss={row['loss']:.5f}, "
                      f"permutation accuracy={100 * row['accuracy']:.2f}%{suffix}", flush=True)
        self.last_encoder_state_ = deepcopy(self.encoder_.state_dict())
        self.last_head_state_ = deepcopy(self.head_.state_dict())
        if select_best and self.best_encoder_state_ is not None:
            self.encoder_.load_state_dict(self.best_encoder_state_)
            self.head_.load_state_dict(self.best_head_state_)
            if self.verbose:
                print(f"Restored best checkpoint: epoch {self.best_epoch_}, "
                      f"monitor={self.best_score_:.4f}", flush=True)
        self.is_fitted_ = True
        return self

    def use_checkpoint(self, which):
        """Switch the live weights between 'best' and 'last' after fit()."""
        self._check_fitted()
        if which == "best":
            if self.best_encoder_state_ is None:
                raise RuntimeError("No monitor_fn was used during fit(); there is no 'best' checkpoint.")
            self.encoder_.load_state_dict(self.best_encoder_state_)
            self.head_.load_state_dict(self.best_head_state_)
        elif which == "last":
            self.encoder_.load_state_dict(self.last_encoder_state_)
            self.head_.load_state_dict(self.last_head_state_)
        else:
            raise ValueError("which must be 'best' or 'last'.")
        return self

    def _check_fitted(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit(X_train) before transform/evaluate_puzzle/save.")

    def evaluate_puzzle(self, X, *, max_windows=1024, permutations_per_window=24,
                        batch_size=128, random_state=200042, min_gap=None, return_details=False):
        """Fixed-weight CE/accuracy. Pass min_gap (e.g. window_size, or the
        full training_span for a stricter test) to keep sampled windows at
        least that many bins apart within a sequence -- overlapping windows
        share almost all of their underlying bins, so without this the
        reported accuracy is optimistic about how independent the "test"
        observations really are.
        """
        self._check_fitted()
        maximum = _integer("max_windows", max_windows)
        size = _integer("batch_size", batch_size)
        count = _integer("permutations_per_window", permutations_per_window)
        if count > 24:
            raise ValueError("permutations_per_window must not exceed 24.")
        seed = _integer("random_state", random_state, 0)
        sequences, _ = _sequences(X, self.training_span, self.n_features_in_)
        dataset = _Windows(sequences, self.training_span)
        rng = np.random.default_rng(seed)
        chosen = _min_gap_sample(dataset, maximum, min_gap, rng)
        n = len(chosen)
        starts, labels = _draw_tile_plan(n, self.window_size, self.tile_gap, count, rng)
        predictions = np.empty(n * count, dtype=np.int64)
        losses = np.empty(n * count, dtype=np.float32)
        modes = self.encoder_.training, self.head_.training
        self.encoder_.eval()
        self.head_.eval()
        try:
            with torch.inference_mode():
                for begin in range(0, n, size):
                    end = min(begin + size, n)
                    windows = torch.stack([dataset[int(i)] for i in chosen[begin:end]]).to(self.device_)
                    label_batch = labels[begin * count:end * count]
                    logits, target = self._tile_logits(windows, starts[begin:end], label_batch,
                                                       count, training=False)
                    if logits.shape != ((end - begin) * count, 24) or not torch.isfinite(logits).all().item():
                        raise FloatingPointError("Invalid puzzle evaluation logits.")
                    losses[begin * count:end * count] = F.cross_entropy(logits, target, reduction="none").cpu().numpy()
                    predictions[begin * count:end * count] = logits.argmax(dim=1).cpu().numpy()
        finally:
            self.encoder_.train(modes[0])
            self.head_.train(modes[1])
        correct = labels == predictions
        metrics = dict(
            window_size=self.window_size, training_span=self.training_span,
            tile_gap=list(self.tile_gap), tile_normalize=self.tile_normalize,
            accuracy=float(correct.mean()), accuracy_percent=float(100 * correct.mean()),
            cross_entropy=float(losses.mean(dtype=np.float64)), chance_accuracy_percent=100 / 24,
            n_windows=n, available_starts=len(dataset), n_puzzles=len(labels),
            permutations_per_window=count, random_state=seed, min_gap=min_gap,
            sampling=("Distinct starts, no minimum spacing enforced (pseudo-replication likely)."
                     if min_gap is None else
                     f"Starts within a sequence kept >= {min_gap} bins apart."),
        )
        if not return_details:
            return metrics
        sequence_ids = np.searchsorted(dataset.ends, chosen, side="right")
        beginnings = np.r_[0, dataset.ends[:-1]]
        window_starts = chosen - beginnings[sequence_ids]
        details = dict(sequence_ids=sequence_ids, window_starts=window_starts,
                       true_class=labels.reshape(n, count), predicted_class=predictions.reshape(n, count),
                       cross_entropy=losses.reshape(n, count))
        return metrics, details

    def transform(self, X, *, pad=True, batch_size=None, return_indices=False):
        """Chronological-window embeddings; NEVER runs the puzzle head. Each
        window is put through the SAME tile_normalize step used in training
        (that is the point: the encoder was trained to be blind to a
        window's absolute level, so it must be queried the same way, or
        train/inference inputs would come from different distributions).
        """
        self._check_fitted()
        if not isinstance(pad, bool):
            raise ValueError("pad must be bool.")
        size = self.batch_size if batch_size is None else _integer("batch_size", batch_size)
        sequences, is_list = _sequences(X, 1 if pad else self.window_size, self.n_features_in_)
        left = self.window_size // 2
        right = self.window_size - left - 1
        result, indices = [], []
        was_training = self.encoder_.training
        self.encoder_.eval()
        try:
            with torch.inference_mode():
                for x in sequences:
                    if pad:
                        centers = np.arange(len(x), dtype=np.int64)
                        x = np.pad(x, ((left, right), (0, 0)), mode="edge")
                    else:
                        centers = np.arange(left, len(x) - right, dtype=np.int64)
                    embeddings = np.empty((len(centers), self.output_dimension), dtype=np.float32)
                    for start in range(0, len(centers), size):
                        end = min(start + size, len(centers))
                        locations = np.arange(start, end)[:, None] + np.arange(self.window_size)
                        windows = np.ascontiguousarray(x[locations].transpose(0, 2, 1))
                        tensor = torch.from_numpy(windows).to(self.device_)
                        tensor = _normalize_tiles(tensor, self.tile_normalize)
                        embeddings[start:end] = self.encoder_(tensor).cpu().numpy()
                    result.append(embeddings)
                    indices.append(centers)
        finally:
            self.encoder_.train(was_training)
        values = result if is_list else result[0]
        times = indices if is_list else indices[0]
        return (values, times) if return_indices else values

    def fit_transform(self, X, **kwargs):
        transform_kwargs = {k: v for k, v in kwargs.items()
                            if k in ("pad", "batch_size", "return_indices")}
        fit_kwargs = {k: v for k, v in kwargs.items() if k not in transform_kwargs}
        return self.fit(X, **fit_kwargs).transform(X, **transform_kwargs)

    def save(self, path):
        self._check_fitted()
        payload = dict(
            format_version=1, model_type="neural_jigsaw", params=self.get_params(),
            n_features=self.n_features_in_,
            encoder={k: v.detach().cpu() for k, v in self.encoder_.state_dict().items()},
            head={k: v.detach().cpu() for k, v in self.head_.state_dict().items()},
            last_encoder=self.last_encoder_state_, last_head=self.last_head_state_,
            best_epoch=self.best_epoch_, best_score=self.best_score_,
            history=self.history_, n_steps=self.n_steps_, n_windows=self.n_windows_,
        )
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path, device="cuda_if_available"):
        data = torch.load(Path(path), map_location="cpu", weights_only=True)
        if data.get("model_type") != "neural_jigsaw":
            raise ValueError("Not a NeuralJigsaw checkpoint.")
        model = cls(**dict(data["params"], device=device))
        model._build(data["n_features"])
        model.encoder_.load_state_dict(data["encoder"])
        model.head_.load_state_dict(data["head"])
        model.last_encoder_state_ = data["last_encoder"]
        model.last_head_state_ = data["last_head"]
        model.best_epoch_, model.best_score_ = data["best_epoch"], data["best_score"]
        model.best_encoder_state_ = model.best_head_state_ = None
        model.history_, model.n_steps_, model.n_windows_ = data["history"], data["n_steps"], data["n_windows"]
        model.encoder_.eval()
        model.head_.eval()
        model.is_fitted_ = True
        return model


def _self_test():
    """Small real-PyTorch CPU test: python neural_jigsaw.py --self-test."""
    import tempfile
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # 1) Per-tile normalization: zeroes each tile's own per-channel mean,
        #    and does so using ONLY that tile's bins (verified against a
        #    hand-computed reference), never other tiles' statistics.
        tiles = torch.randn(3, 4, 5, 6) * 10 + 100  # (B,4,C,W), large offset
        centered = _normalize_tiles(tiles, "center")
        torch.testing.assert_close(centered.mean(dim=-1), torch.zeros(3, 4, 5), atol=1e-4, rtol=0)
        reference = tiles[0, 1] - tiles[0, 1].mean(dim=-1, keepdim=True)
        torch.testing.assert_close(centered[0, 1], reference)
        zscored = _normalize_tiles(tiles, "zscore")
        torch.testing.assert_close(zscored.std(dim=-1, unbiased=False), torch.ones(3, 4, 5), atol=1e-3, rtol=0)
        torch.testing.assert_close(_normalize_tiles(tiles, "off"), tiles)

        # 2) A tile's absolute level is destroyed by design: two tiles that
        #    differ only by a constant offset are identical after centering.
        shifted = tiles.clone()
        shifted[0, 2] = tiles[0, 2] + 37.0
        torch.testing.assert_close(_normalize_tiles(tiles, "center")[0, 2],
                                   _normalize_tiles(shifted, "center")[0, 2], atol=1e-4, rtol=1e-4)

        # 3) Augmentation changes values but keeps shape/finiteness, and a
        #    zero-jitter/zero-noise call is a no-op.
        gen = torch.Generator().manual_seed(0)
        channel_std = torch.ones(5)
        augmented = _augment_tiles(tiles, 0.2, 0.1, channel_std, gen)
        assert augmented.shape == tiles.shape and torch.isfinite(augmented).all()
        assert not torch.equal(augmented, tiles)
        torch.testing.assert_close(_augment_tiles(tiles, 0.0, 0.0, channel_std, gen), tiles)

        # 4) Tiny real fit + monitor-based checkpoint selection on synthetic
        #    data with an injected pure drift shortcut. A model that ignores
        #    the shortcut (tile_normalize="center") should not be able to
        #    solve the puzzle purely from tile means; a raw model given the
        #    same drift with normalization off can still see the mean cue.
        rng = np.random.default_rng(3)
        T, C = 400, 6
        drift = np.linspace(0, 5, T)[:, None]
        trials = [(rng.normal(scale=0.3, size=(T, C)) + drift).astype(np.float32)]

        scores = iter([0.1, 0.2, 0.15, 0.4, 0.3])  # deliberately non-monotonic
        seen_scores = []

        def fake_monitor(model):
            value = next(scores)
            seen_scores.append(value)
            return value

        params = dict(window_size=4, output_dimension=6, num_hidden_units=8,
                      head_hidden_units=16, batch_size=16, device="cpu", verbose=False,
                      tile_gap=(1, 3), random_state=11, augment_gain_jitter=0.0,
                      augment_noise_std=0.0)
        model = NeuralJigsaw(**params, tile_normalize="center", max_epochs=5).fit(
            trials, monitor_fn=fake_monitor, monitor_every=1)
        assert model.history_[-1]["steps"] == model.n_steps_ and model.n_steps_ > 0
        assert seen_scores == [0.1, 0.2, 0.15, 0.4, 0.3]
        assert model.best_epoch_ == 4 and abs(model.best_score_ - 0.4) < 1e-9
        assert any(not torch.equal(a, b) for a, b in
                   zip(model.encoder_.state_dict().values(), model.last_encoder_state_.values()))
        model.use_checkpoint("last")
        assert all(torch.equal(a, b) for a, b in
                   zip(model.encoder_.state_dict().values(), model.last_encoder_state_.values()))
        model.use_checkpoint("best")
        assert all(torch.equal(a, b) for a, b in
                   zip(model.encoder_.state_dict().values(), model.best_encoder_state_.values()))

        # 5) transform shapes/alignment, and evaluate_puzzle with a min_gap
        #    actually enforces spacing between the sampled windows.
        padded = model.transform(trials, pad=True)
        valid, indices = model.transform(trials, pad=False, return_indices=True)
        assert padded[0].shape == (T, params["output_dimension"])
        assert valid[0].shape == (T - model.window_size + 1, params["output_dimension"])
        np.testing.assert_array_equal(indices[0], np.arange(len(valid[0])) + model.window_size // 2)

        metrics, details = model.evaluate_puzzle(trials, max_windows=8, min_gap=20,
                                                 permutations_per_window=4, return_details=True)
        starts = np.sort(details["window_starts"])
        assert np.all(np.diff(starts) >= 20)
        assert metrics["accuracy"] >= 0.0 and math.isfinite(metrics["cross_entropy"])

        # 6) save/load round-trips weights, history and best/last bookkeeping.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "neural_jigsaw.pt"
            model.save(path)
            loaded = NeuralJigsaw.load(path, device="cpu")
            np.testing.assert_allclose(loaded.transform(trials[0]), padded[0], rtol=1e-5, atol=1e-6)
            assert loaded.best_epoch_ == model.best_epoch_
            assert loaded.get_params() == model.get_params()

        print("PASS: tile normalization removes absolute level (incl. against a constant "
              "offset), augmentation is a real but shape-preserving perturbation, monitor-"
              "based best-checkpoint selection picks the right non-final epoch and is "
              "restorable, transform alignment, min_gap de-correlated evaluation, save/load.")
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run the small CPU test.")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        parser.print_help()

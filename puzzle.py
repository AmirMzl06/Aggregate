"""Temporal permutation representation learning, independent of CEBRA.

Requires NumPy and PyTorch >= 2.0. Input arrays use shape (time, channels).

    from puzzle import puzzle
    model = puzzle(window_size=10, delta=1, output_dimension=64,
                   num_hidden_units=64, batch_size=256, max_epochs=30)
    model.fit(X_train)
    Z_train = model.transform(X_train)  # (len(X_train), 64)
    Z_valid = model.transform(X_valid)

Objective: sample four slots [s, s+d, s+2d, s+3d], permute their VALUES,
and classify the permutation among 24 classes. The full window enters the
encoder. The head receives the latent plus the selected-slot mask; the encoder
does not receive that mask. Only cross-entropy is optimized.

One epoch visits EVERY valid sliding window once in shuffled order. Each visit
samples a start slot and, by default, one permutation. Set
permutations_per_window=24 to use all 24 permutations of that selected quartet.
delta is one positive integer (default 1), or a tuple sampled uniformly per
window. This does NOT enumerate all deltas or all start slots on every visit.

The encoder is a valid-convolution residual network with receptive field
exactly window_size. For window_size=10 and dropout=0 it has the offset10
layout: Conv2 -> three residual Conv3 blocks -> Conv3 -> optional L2 norm.
There is no smoothing, temporal averaging, input normalization or train padding.

transform uses chronological, unmodified windows. With pad=True (default),
edge padding is applied separately to each sequence, and output row t aligns
with input row t. For W=10 the context is t-5 ... t+4; this is NOT causal.
Use pad=False, return_indices=True for valid interior windows without padding.
Split train/validation/test BEFORE fit. A single 2D array is treated as one
continuous recording; pass a list of trial arrays to prevent crossing trials.

Representation usefulness must be evaluated with a separate behavioral decoder.
Permutation accuracy alone is not a measure of behavioral decoding quality.

Quick environment check: python puzzle.py --self-test
"normalize" controls only the latent L2 norm, not the raw input values.
The fit loop uses float32 without mixed precision. Seeding does not promise
bitwise reproducibility across devices or PyTorch versions.
"""
from itertools import permutations
import math
import numbers
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


# label -> source ranks in ascending destination-slot order.
# E.g. permutation [2,0,3,1] writes [v2,v0,v3,v1] into the four selected slots.
PERMUTATIONS = np.asarray(list(permutations(range(4))), dtype=np.int64)


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
        # Own the storage: training must never modify the caller's data.
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


class _Residual(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout1d(dropout), nn.Conv1d(width, width, 3), nn.GELU())

    def forward(self, x):
        return x[..., 1:-1] + self.net(x)


class _Encoder(nn.Module):
    def __init__(self, channels, window_size, width, output_dimension, dropout, normalize):
        super().__init__()
        # Sum(kernel-1) = W-1, hence W input bins -> exactly one output bin.
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


def _draw_plan(batch, window_size, deltas, count, rng):
    """Sample one quartet per window; choose count distinct permutation labels."""
    delta = np.asarray(deltas, dtype=np.int64)[rng.integers(len(deltas), size=batch)]
    start = rng.integers(0, window_size - 3 * delta)
    slots = start[:, None] + delta[:, None] * np.arange(4)
    if count == 24:
        labels = np.broadcast_to(np.arange(24), (batch, 24)).copy()
    elif count == 1:
        labels = rng.integers(0, 24, size=(batch, 1))
    else:
        labels = np.argsort(rng.random((batch, 24)), axis=1)[:, :count]
    return np.repeat(slots, count, axis=0), labels.reshape(-1)


def _make_puzzles(windows, slots, labels, count):
    """Only selected slots change. Every channel uses the SAME time permutation."""
    x = windows.repeat_interleave(count, dim=0)
    slots = torch.as_tensor(slots, dtype=torch.long, device=x.device)
    labels = torch.as_tensor(labels, dtype=torch.long, device=x.device)
    table = torch.as_tensor(PERMUTATIONS, dtype=torch.long, device=x.device)
    sources = slots.gather(1, table[labels])
    shape = (-1, x.shape[1], -1)
    values = x.gather(2, sources[:, None, :].expand(*shape))
    x.scatter_(2, slots[:, None, :].expand(*shape), values)
    mask = x.new_zeros((len(x), x.shape[2]))
    mask.scatter_(1, slots, 1.0)
    return x, mask, labels


class puzzle:
    """Unsupervised temporal puzzle estimator with fit/transform interface.

    delta=1 uses one-bin spacing. delta=(1,2,3) samples one of those spacings
    per window with equal probability; compute per epoch stays the same.
    permutations_per_window counts puzzle copies: batch_size=256 and count=24
    sends 6144 windows through the encoder in one optimizer step.

    fit always initializes new weights. It never uses behavioral labels.
    history_ contains epoch, loss, accuracy, and cumulative optimizer steps.
    encoder_ and head_ expose the trained PyTorch modules for later fine-tuning.
    """
    def __init__(self, window_size=10, delta=1, output_dimension=64,
                 num_hidden_units=64, head_hidden_units=128, batch_size=256,
                 max_epochs=30, learning_rate=3e-4, weight_decay=0.0,
                 dropout=0.0, normalize=True, permutations_per_window=1,
                 device="cuda_if_available", random_state=42, verbose=True):
        self.window_size = _integer("window_size", window_size, 4)
        self.output_dimension = _integer("output_dimension", output_dimension)
        self.num_hidden_units = _integer("num_hidden_units", num_hidden_units)
        self.head_hidden_units = _integer("head_hidden_units", head_hidden_units)
        self.batch_size = _integer("batch_size", batch_size)
        self.max_epochs = _integer("max_epochs", max_epochs)
        self.learning_rate = _real("learning_rate", learning_rate, 0, strict_min=True)
        self.weight_decay = _real("weight_decay", weight_decay, 0)
        self.dropout = _real("dropout", dropout, 0, 1)
        if not isinstance(normalize, bool):
            raise ValueError("normalize must be bool (latent L2 normalization).")
        self.normalize = normalize
        values = delta if isinstance(delta, (tuple, list)) else (delta,)
        self.delta = tuple(_integer("delta", d) for d in values)
        if (not self.delta or len(set(self.delta)) != len(self.delta)
                or any(3 * d >= self.window_size for d in self.delta)):
            raise ValueError("Deltas must be unique and satisfy 3*delta < window_size.")
        self.permutations_per_window = _integer("permutations_per_window", permutations_per_window)
        if self.permutations_per_window > 24:
            raise ValueError("permutations_per_window must be between 1 and 24.")
        self.device = str(device)
        self.random_state = _integer("random_state", random_state, 0)
        self.verbose = bool(verbose)

    def get_params(self):
        names = ("window_size", "delta", "output_dimension", "num_hidden_units",
                 "head_hidden_units", "batch_size", "max_epochs", "learning_rate",
                 "weight_decay", "dropout", "normalize", "permutations_per_window",
                 "device", "random_state", "verbose")
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
            nn.Linear(self.output_dimension + self.window_size, self.head_hidden_units),
            nn.GELU(), nn.Linear(self.head_hidden_units, 24),
        ).to(self.device_)

    def fit(self, X):
        sequences, _ = _sequences(X, self.window_size)
        self.is_fitted_ = False
        # Seed BEFORE initializing the encoder and head.
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        rng = np.random.default_rng(self.random_state)
        generator = torch.Generator().manual_seed(self.random_state)
        self._build(sequences[0].shape[1])
        dataset = _Windows(sequences, self.window_size)
        self.n_windows_ = len(dataset)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            drop_last=False, num_workers=0, generator=generator,
                            pin_memory=self.device_.type == "cuda")
        parameters = list(self.encoder_.parameters()) + list(self.head_.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        self.history_ = []
        self.n_steps_ = 0
        if self.verbose:
            print(f"Puzzle: {self.n_windows_} valid windows, delta={self.delta}, "
                  f"{self.permutations_per_window} permutation(s)/window, device={self.device_}", flush=True)
        for epoch in range(self.max_epochs):
            self.encoder_.train()
            self.head_.train()
            total_loss, correct, seen = 0.0, 0, 0
            for windows in loader:
                slots, labels = _draw_plan(len(windows), self.window_size, self.delta,
                                            self.permutations_per_window, rng)
                windows = windows.to(self.device_, non_blocking=True)
                inputs, mask, target = _make_puzzles(windows, slots, labels,
                                                     self.permutations_per_window)
                optimizer.zero_grad(set_to_none=True)
                z = self.encoder_(inputs)
                logits = self.head_(torch.cat((z, mask), dim=1))
                loss = F.cross_entropy(logits, target)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Nonfinite puzzle loss at step {self.n_steps_}.")
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(target)
                correct += (logits.detach().argmax(dim=1) == target).sum().item()
                seen += len(target)
                self.n_steps_ += 1
            row = dict(epoch=epoch + 1, loss=total_loss / seen,
                       accuracy=correct / seen, n_puzzles=seen, steps=self.n_steps_)
            self.history_.append(row)
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_epochs}: loss={row['loss']:.5f}, "
                      f"permutation accuracy={100 * row['accuracy']:.2f}%", flush=True)
        self.encoder_.eval()
        self.head_.eval()
        self.is_fitted_ = True
        return self

    def _check_fitted(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit(X_train) before transform/save.")

    def transform(self, X, *, pad=True, batch_size=None, return_indices=False):
        """Return chronological-window embeddings; NEVER run the puzzle head.

        A list of trials returns a list of embeddings (and index arrays).
        With pad=False, row j describes the window X[j:j+W] and aligns with
        X[j + W//2]. return_indices exposes those indices for label alignment.
        With pad=True, one embedding per original input row is returned.
        No normalization statistics are fitted on either train or test data.
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
                        embeddings[start:end] = self.encoder_(tensor).cpu().numpy()
                    result.append(embeddings)
                    indices.append(centers)
        finally:
            self.encoder_.train(was_training)
        values = result if is_list else result[0]
        times = indices if is_list else indices[0]
        return (values, times) if return_indices else values

    def fit_transform(self, X, **transform_kwargs):
        return self.fit(X).transform(X, **transform_kwargs)

    def save(self, path):
        """Save fitted weights/config; does not save data or optimizer state."""
        self._check_fitted()
        payload = dict(
            format_version=1, params=self.get_params(), n_features=self.n_features_in_,
            encoder={k: v.detach().cpu() for k, v in self.encoder_.state_dict().items()},
            head={k: v.detach().cpu() for k, v in self.head_.state_dict().items()},
            history=self.history_, n_steps=self.n_steps_, n_windows=self.n_windows_,
        )
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path, device="cuda_if_available"):
        data = torch.load(Path(path), map_location="cpu", weights_only=True)
        if data.get("format_version") != 1:
            raise ValueError("Unsupported puzzle checkpoint format.")
        parameters = dict(data["params"], device=device)
        model = cls(**parameters)
        model._build(data["n_features"])
        model.encoder_.load_state_dict(data["encoder"])
        model.head_.load_state_dict(data["head"])
        model.history_ = data["history"]
        model.n_steps_ = data["n_steps"]
        model.n_windows_ = data["n_windows"]
        model.encoder_.eval()
        model.head_.eval()
        model.is_fitted_ = True
        return model


Puzzle = puzzle


def _self_test():
    """Small real-PyTorch CPU test: python puzzle.py --self-test."""
    import tempfile

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Exhaustive permutation semantics, preserved unselected slots/channels.
        x = torch.arange(30, dtype=torch.float32).reshape(1, 3, 10)
        original = x.clone()
        slots = np.tile([0, 2, 4, 6], (24, 1))
        xp, masks, labels = _make_puzzles(x, slots, np.arange(24), 24)
        for label, perm in enumerate(PERMUTATIONS):
            expected = x[0].clone()
            expected[:, slots[label]] = x[0][:, slots[label][perm]]
            torch.testing.assert_close(xp[label], expected)
        assert torch.equal(x, original)
        assert (masks.sum(dim=1) == 4).all() and labels.unique().numel() == 24

        # Exact receptive field, including odd window sizes.
        for width in (4, 5, 10, 36):
            encoder = _Encoder(3, width, 8, 6, 0.0, True)
            assert encoder(torch.randn(2, 3, width)).shape == (2, 6)

        rng = np.random.default_rng(8)
        trials = [rng.normal(size=(29, 3)).astype(np.float32),
                  rng.normal(size=(23, 3)).astype(np.float32)]
        copies = [x.copy() for x in trials]
        model = puzzle(window_size=10, delta=(1, 2, 3), output_dimension=6,
                       num_hidden_units=8, head_hidden_units=16, batch_size=8,
                       max_epochs=2, device="cpu", verbose=False)
        assert model.fit(trials) is model
        assert model.n_windows_ == 34 and model.n_steps_ == 10
        assert all(row['n_puzzles'] == 34 for row in model.history_)
        assert any(p.grad is not None and p.grad.abs().sum().item() > 0
                   for p in model.encoder_.parameters())
        assert all(np.array_equal(a, b) for a, b in zip(trials, copies))
        padded = model.transform(trials)
        valid, indices = model.transform(trials, pad=False, return_indices=True)
        for seq, zpad, zvalid, times in zip(trials, padded, valid, indices):
            assert zpad.shape == (len(seq), 6)
            assert zvalid.shape == (len(seq) - 9, 6)
            np.testing.assert_array_equal(times, np.arange(5, len(seq) - 4))
            np.testing.assert_allclose(zpad[times], zvalid, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(zpad, model.transform(seq, batch_size=3),
                                       rtol=1e-5, atol=1e-6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'puzzle.pt'
            model.save(path)
            loaded = puzzle.load(path, device="cpu")
            np.testing.assert_allclose(loaded.transform(trials[0]), padded[0],
                                       rtol=1e-5, atol=1e-6)
        print("PASS: permutations, encoder shapes, CPU fit/autograd, trial boundaries, "
              "time alignment, chunking, input immutability and save/load.")
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

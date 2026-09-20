"""Single-window temporal position puzzle, independent of CEBRA.

    from puzzle_positions import puzzle
    model = puzzle(window_size=4, max_epochs=1000)
    model.fit(X_train)  # (time, neurons), or a list of separate trials
    Z, times = model.transform(X_valid, pad=False, return_indices=True)
    metrics = model.evaluate_puzzle(X_valid)
    shuffled = X_valid[:4][[2, 0, 3, 1]].T  # (neurons, 4)
    ordered, positions = model.reorder(shuffled, return_positions=True)

One WHOLE four-bin window enters one valid-convolution encoder. The head emits
(B,4,4) logits: row i describes the original chronological position of INPUT
bin i. Loss = mean of four 4-class cross-entropies, NOT a 24-class CE. Training
uses fresh permutations on every visit; no behavior labels enter this loss.
There are no tiles, gaps, multiple encoder branches, input normalization,
smoothing or train padding. The encoder is the same four-bin Conv2 -> GELU ->
Conv3 layout as the previous implementation; latent L2 normalization is optional.

For inference, choose the highest-scoring bijective assignment among all 24
permutations, then use its INVERSE to gather the input columns into order. No
neural values are synthesized or averaged. Hard assignment is only for metrics
and inference; CE trains encoder and head directly, without detach or argmax.
The predicted order is always valid but may be WRONG. Identical bins make some
position labels observationally indistinguishable.

fit accepts (T,N) arrays; reorder defaults to (N,4) or (B,N,4). Pass
layout='time_first' for (4,N) or (B,4,N). Reorder preserves the numeric input
values/dtype; neural network inference uses float32. transform uses natural,
unshuffled windows and NEVER sorts them before embedding. pad=False aligns a
window starting at j with label j+2. pad=True uses left=2/right=1 edge padding,
separately for each sequence. This is not a causal representation.

A single 2D array is treated as a continuous recording; lists preserve trial
boundaries. fit always resets weights. max_epochs=0 gives a random encoder
control. Save/load stores weights/config, not optimizer state for resumption.
These checkpoints are incompatible with earlier window/tiles checkpoints.

Chance reference: position accuracy 25%, exact whole-puzzle accuracy 1/24,
uniform position CE log(4) = 1.386294. Behavior usefulness still requires a
separate frozen-encoder decoding experiment.

Dependencies: numpy, torch>=2.0. Run: python puzzle_positions.py --self-test
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

# Each row maps shuffled INPUT slot -> original chronological position.
# [2,0,3,1] means shuffled=[x2,x0,x3,x1]; restore indices=argsort(row)=[1,3,0,2].
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
        return torch.from_numpy(x.T.copy())


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


class _PositionHead(nn.Module):
    def __init__(self, dimension, hidden):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
                                    nn.Linear(hidden, 16))

    def forward(self, z):
        return self.layers(z).reshape(-1, 4, 4)


def _draw_permutation_ids(batch, count, rng):
    if count == 24:
        ids = np.broadcast_to(np.arange(24), (batch, 24)).copy()
    elif count == 1:
        ids = rng.integers(0, 24, size=(batch, 1))
    else:
        ids = np.argsort(rng.random((batch, 24)), axis=1)[:, :count]
    return ids.reshape(-1)


def _shuffle_windows(windows, permutation_ids, count):
    """(B,N,4) -> (B*count,N,4), targets (B*count,4), same shuffle for ALL neurons."""
    x = windows.repeat_interleave(count, dim=0)
    ids = torch.as_tensor(permutation_ids, dtype=torch.long, device=x.device)
    table = torch.as_tensor(PERMUTATIONS, dtype=torch.long, device=x.device)
    target_positions = table[ids]
    shuffled = x.gather(2, target_positions[:, None, :].expand(-1, x.shape[1], -1))
    return shuffled, target_positions


def _valid_assignment(logits):
    """Select a one-to-one input-slot -> true-position mapping.

    Sum logits[i,p[i]] for every valid permutation p. This has the same argmax
    as summing row log-softmax values because the row constants cancel.
    No Hungarian dependency is needed for four bins. Ties use lexicographic order.
    """
    if logits.ndim != 3 or logits.shape[1:] != (4, 4):
        raise ValueError("Position logits must have shape (B,4,4).")
    table = torch.as_tensor(PERMUTATIONS, dtype=torch.long, device=logits.device)
    indices = table[None, :, :, None].expand(len(logits), -1, -1, 1)
    candidates = logits[:, None, :, :].expand(-1, 24, -1, -1)
    scores = candidates.gather(3, indices).squeeze(-1).sum(dim=2)
    return table[scores.argmax(dim=1)]


def _restore_windows(shuffled, positions):
    """Apply the INVERSE of an input-slot -> original-position assignment."""
    order = positions.argsort(dim=1)
    return shuffled.gather(2, order[:, None, :].expand(-1, shuffled.shape[1], -1))


class puzzle:
    """fit/transform/reorder estimator for exactly four time bins.

    Encoder input is (B,neurons,4), latent is (B,D), head logits are (B,4,4).
    An epoch visits every valid stride-one window once, with fresh shuffles.
    permutations_per_window multiplies training batch size/encoder compute.
    Public reorder/predict/__call__ return reordered neural vectors, while
    transform returns latent features for behavioral decoding.
    """
    def __init__(self, window_size=4, output_dimension=64, num_hidden_units=64,
                 head_hidden_units=128, batch_size=2048, max_epochs=1000,
                 learning_rate=3e-4, weight_decay=0.0, dropout=0.1, normalize=True,
                 permutations_per_window=1, device="cuda_if_available",
                 random_state=42, verbose=True, log_every=1):
        self.window_size = _integer("window_size", window_size)
        if self.window_size != 4:
            raise ValueError("This position-puzzle architecture requires window_size=4.")
        self.output_dimension = _integer("output_dimension", output_dimension)
        self.num_hidden_units = _integer("num_hidden_units", num_hidden_units)
        self.head_hidden_units = _integer("head_hidden_units", head_hidden_units)
        self.batch_size = _integer("batch_size", batch_size)
        self.max_epochs = _integer("max_epochs", max_epochs, 0)
        self.learning_rate = _real("learning_rate", learning_rate, 0, strict_min=True)
        self.weight_decay = _real("weight_decay", weight_decay, 0)
        self.dropout = _real("dropout", dropout, 0, 1)
        if not isinstance(normalize, bool):
            raise ValueError("normalize must be bool (latent L2 normalization only).")
        self.normalize = normalize
        self.permutations_per_window = _integer("permutations_per_window", permutations_per_window)
        if self.permutations_per_window > 24:
            raise ValueError("permutations_per_window must be in [1,24].")
        self.device = str(device)
        self.random_state = _integer("random_state", random_state, 0)
        self.verbose = bool(verbose)
        self.log_every = _integer("log_every", log_every)

    def get_params(self):
        names = ("window_size", "output_dimension", "num_hidden_units", "head_hidden_units",
                 "batch_size", "max_epochs", "learning_rate", "weight_decay", "dropout",
                 "normalize", "permutations_per_window", "device", "random_state", "verbose",
                 "log_every")
        return {name: getattr(self, name) for name in names}

    def _build(self, channels):
        name = self.device
        if name == "cuda_if_available":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_ = torch.device(name)
        self.n_features_in_ = int(channels)
        self.encoder_ = _Encoder(channels, 4, self.num_hidden_units, self.output_dimension,
                                 self.dropout, self.normalize).to(self.device_)
        self.head_ = _PositionHead(self.output_dimension, self.head_hidden_units).to(self.device_)

    def _logits(self, x):
        return self.head_(self.encoder_(x))

    def _check_fitted(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit(X_train) first; max_epochs=0 initializes a random model.")

    def fit(self, X):
        sequences, _ = _sequences(X, 4)
        self.is_fitted_ = False
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self._build(sequences[0].shape[1])
        rng = np.random.default_rng(self.random_state)
        generator = torch.Generator().manual_seed(self.random_state)
        dataset = _Windows(sequences, 4)
        self.n_windows_ = len(dataset)
        self.history_, self.n_steps_ = [], 0
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            drop_last=False, num_workers=0, generator=generator,
                            pin_memory=self.device_.type == "cuda")
        optimizer = torch.optim.Adam(list(self.encoder_.parameters()) + list(self.head_.parameters()),
                                      lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.verbose:
            print(f"Position puzzle: {self.n_windows_} windows, epochs={self.max_epochs}, "
                  f"batch={self.batch_size}, permutations/window={self.permutations_per_window}, "
                  f"head=4x4, device={self.device_}", flush=True)
            if self.max_epochs == 0:
                print("Random encoder control: zero optimizer updates.", flush=True)
        for epoch in range(self.max_epochs):
            self.encoder_.train()
            self.head_.train()
            total_loss, correct_bins, correct_puzzles, seen = 0.0, 0, 0, 0
            for windows in loader:
                ids = _draw_permutation_ids(len(windows), self.permutations_per_window, rng)
                windows = windows.to(self.device_, non_blocking=True)
                inputs, targets = _shuffle_windows(windows, ids, self.permutations_per_window)
                optimizer.zero_grad(set_to_none=True)
                logits = self._logits(inputs)
                # Flatten only the sample/input-slot dimensions. Four position classes.
                loss = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Nonfinite position CE at optimizer step {self.n_steps_}.")
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    predicted = _valid_assignment(logits.detach())
                    matches = predicted == targets
                    correct_bins += matches.sum().item()
                    correct_puzzles += matches.all(dim=1).sum().item()
                total_loss += loss.item() * len(targets)
                seen += len(targets)
                self.n_steps_ += 1
            row = dict(epoch=epoch + 1, position_cross_entropy=total_loss / seen,
                       position_accuracy=correct_bins / (4 * seen),
                       exact_accuracy=correct_puzzles / seen, n_puzzles=seen, steps=self.n_steps_)
            self.history_.append(row)
            if self.verbose and ((epoch + 1) % self.log_every == 0 or epoch + 1 == self.max_epochs):
                print(f"Epoch {epoch + 1}/{self.max_epochs}: position CE={row['position_cross_entropy']:.5f}, "
                      f"bin accuracy={100 * row['position_accuracy']:.2f}%, "
                      f"exact puzzle accuracy={100 * row['exact_accuracy']:.2f}%", flush=True)
        self.encoder_.eval()
        self.head_.eval()
        self.is_fitted_ = True
        return self

    def _input_windows(self, windows, layout):
        if layout not in ("neurons_first", "time_first"):
            raise ValueError("layout must be 'neurons_first' or 'time_first'.")
        if torch.is_tensor(windows):
            windows = windows.detach().cpu().numpy()
        original = np.asarray(windows)
        single = original.ndim == 2
        if original.ndim not in (2, 3):
            raise ValueError("Expected a single window or a batch of windows, not a recording.")
        data = original[None] if single else original
        if layout == "time_first":
            data = data.transpose(0, 2, 1)
        if (len(data) == 0 or data.shape[1:] != (self.n_features_in_, 4)
                or not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data)
                or not np.isfinite(data).all()):
            raise ValueError(f"Expected finite real windows with {self.n_features_in_} neurons and 4 bins.")
        network_input = np.asarray(data, dtype=np.float32)
        if not np.isfinite(network_input).all():
            raise ValueError("Window values overflow float32.")
        return data, network_input, single

    def predict_positions(self, windows, *, layout="neurons_first", batch_size=None):
        """Return zero-based input-slot -> predicted original-position assignments.

        Every returned row is a permutation of [0,1,2,3]. For [x2,x0,x3,x1],
        the correct target is [2,0,3,1], NOT its inverse.
        """
        self._check_fitted()
        _, inputs, single = self._input_windows(windows, layout)
        size = self.batch_size if batch_size is None else _integer("batch_size", batch_size)
        result = np.empty((len(inputs), 4), dtype=np.int64)
        modes = self.encoder_.training, self.head_.training
        self.encoder_.eval()
        self.head_.eval()
        try:
            with torch.inference_mode():
                for begin in range(0, len(inputs), size):
                    end = min(begin + size, len(inputs))
                    x = torch.from_numpy(np.ascontiguousarray(inputs[begin:end])).to(self.device_)
                    logits = self._logits(x)
                    if not torch.isfinite(logits).all().item():
                        raise FloatingPointError("Nonfinite position logits.")
                    result[begin:end] = _valid_assignment(logits).cpu().numpy()
        finally:
            self.encoder_.train(modes[0])
            self.head_.train(modes[1])
        return result[0] if single else result

    def reorder(self, windows, *, layout="neurons_first", batch_size=None, return_positions=False):
        """Return original neural vectors in PREDICTED chronological order.

        Shape/layout/dtype are preserved. No values are generated, smoothed or
        normalized in this output. Prediction errors can still produce wrong order.
        Returns NumPy arrays, including when the input is a torch Tensor.
        """
        self._check_fitted()
        original, _, single = self._input_windows(windows, layout)
        positions = self.predict_positions(windows, layout=layout, batch_size=batch_size)
        batch_positions = positions[None] if single else positions
        order = np.argsort(batch_positions, axis=1)
        ordered = np.take_along_axis(original, order[:, None, :], axis=2)
        if layout == "time_first":
            ordered = ordered.transpose(0, 2, 1)
        if single:
            ordered = ordered[0]
        return (ordered, positions) if return_positions else ordered

    predict = reorder
    __call__ = reorder

    def evaluate_puzzle(self, X, *, max_windows=1024, permutations_per_window=24,
                        batch_size=1024, random_state=200042, return_details=False):
        """Fixed-weight evaluation on unpadded four-bin windows.

        Sample distinct starts across sequences, use all 24 shuffles by default.
        batch_size is the maximum number of SHUFFLED windows per forward pass.
        Sampling is fixed before batching, independent of batch size. Overlapping
        windows are correlated observations. No behavior labels are used.
        """
        self._check_fitted()
        maximum = _integer("max_windows", max_windows)
        count = _integer("permutations_per_window", permutations_per_window)
        size = _integer("batch_size", batch_size)
        seed = _integer("random_state", random_state, 0)
        if count > 24:
            raise ValueError("permutations_per_window must be <=24.")
        sequences, _ = _sequences(X, 4, self.n_features_in_)
        dataset = _Windows(sequences, 4)
        rng = np.random.default_rng(seed)
        n = min(maximum, len(dataset))
        chosen = np.sort(rng.choice(len(dataset), size=n, replace=False))
        ids = _draw_permutation_ids(n, count, rng)
        targets = PERMUTATIONS[ids]
        predictions = np.empty_like(targets)
        raw_predictions = np.empty_like(targets)
        losses = np.empty(len(ids), dtype=np.float32)
        repeated = np.repeat(chosen, count)
        modes = self.encoder_.training, self.head_.training
        self.encoder_.eval()
        self.head_.eval()
        try:
            with torch.inference_mode():
                for begin in range(0, len(ids), size):
                    end = min(begin + size, len(ids))
                    windows = torch.stack([dataset[int(i)] for i in repeated[begin:end]]).to(self.device_)
                    shuffled, target = _shuffle_windows(windows, ids[begin:end], 1)
                    logits = self._logits(shuffled)
                    if logits.shape != (end - begin, 4, 4) or not torch.isfinite(logits).all().item():
                        raise FloatingPointError("Invalid position-puzzle evaluation logits.")
                    ce = F.cross_entropy(logits.reshape(-1, 4), target.reshape(-1), reduction="none")
                    losses[begin:end] = ce.reshape(-1, 4).mean(dim=1).cpu().numpy()
                    predictions[begin:end] = _valid_assignment(logits).cpu().numpy()
                    raw_predictions[begin:end] = logits.argmax(dim=2).cpu().numpy()
        finally:
            self.encoder_.train(modes[0])
            self.head_.train(modes[1])
        matches = predictions == targets
        exact = matches.all(axis=1)
        raw_valid = np.all(np.sort(raw_predictions, axis=1) == np.arange(4), axis=1)
        metrics = dict(
            objective="mean of four 4-class cross-entropies", window_size=4,
            position_cross_entropy=float(losses.mean(dtype=np.float64)),
            position_accuracy_percent=float(100 * matches.mean()),
            exact_accuracy_percent=float(100 * exact.mean()),
            unconstrained_position_accuracy_percent=float(100 * (raw_predictions == targets).mean()),
            unconstrained_invalid_assignment_percent=float(100 * (~raw_valid).mean()),
            chance_position_accuracy_percent=25.0, chance_exact_accuracy_percent=100 / 24,
            uniform_position_cross_entropy=float(np.log(4)),
            n_windows=n, available_windows=len(dataset), n_puzzles=len(ids),
            permutations_per_window=count, random_state=seed,
            per_input_slot_accuracy_percent=(100 * matches.mean(axis=0)).tolist(),
            permutation_counts=np.bincount(ids, minlength=24).tolist(),
            per_permutation_exact_accuracy_percent=[float(100 * exact[ids == k].mean())
                                                     if np.any(ids == k) else None for k in range(24)],
        )
        if not return_details:
            return metrics
        sequence_ids = np.searchsorted(dataset.ends, chosen, side="right")
        window_starts = chosen - np.r_[0, dataset.ends[:-1]][sequence_ids]
        details = dict(sequence_ids=sequence_ids, window_starts=window_starts,
                       permutation_ids=ids.reshape(n, count),
                       true_positions=targets.reshape(n, count, 4),
                       predicted_positions=predictions.reshape(n, count, 4),
                       raw_argmax_positions=raw_predictions.reshape(n, count, 4),
                       position_cross_entropy=losses.reshape(n, count),
                       permutation_table=PERMUTATIONS.copy())
        return metrics, details


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
        self._check_fitted()
        payload = dict(
            model_type="four_bin_position_puzzle", format_version=1,
            params=self.get_params(), n_features=self.n_features_in_,
            encoder={k: v.detach().cpu() for k, v in self.encoder_.state_dict().items()},
            head={k: v.detach().cpu() for k, v in self.head_.state_dict().items()},
            history=self.history_, n_steps=self.n_steps_, n_windows=self.n_windows_,
        )
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path, device="cuda_if_available"):
        data = torch.load(Path(path), map_location="cpu", weights_only=True)
        if data.get("model_type") != "four_bin_position_puzzle" or data.get("format_version") != 1:
            raise ValueError("Expected a position-puzzle checkpoint; old window/tiles checkpoints cannot be used.")
        model = cls(**dict(data["params"], device=device))
        model._build(data["n_features"])
        model.encoder_.load_state_dict(data["encoder"])
        model.head_.load_state_dict(data["head"])
        model.history_, model.n_steps_, model.n_windows_ = data["history"], data["n_steps"], data["n_windows"]
        model.encoder_.eval()
        model.head_.eval()
        model.is_fitted_ = True
        return model


Puzzle = puzzle


def _self_test():
    """Real-PyTorch CPU checks of shuffle/inverse, gradients, inference and IO."""
    import tempfile

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Exhaust all 24 permutations for two differently valued windows.
        original = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        copy = original.clone()
        ids = np.tile(np.arange(24), 2)
        shuffled, targets = _shuffle_windows(original, ids, 24)
        for b in range(2):
            for k, perm in enumerate(PERMUTATIONS):
                torch.testing.assert_close(shuffled[b * 24 + k], original[b][:, perm])
                assert targets[b * 24 + k].tolist() == perm.tolist()
        torch.testing.assert_close(original, copy, rtol=0, atol=0)
        oracle = torch.full((48, 4, 4), -7.0)
        oracle.scatter_(2, targets[:, :, None], 7.0)
        prediction = _valid_assignment(oracle)
        assert torch.equal(prediction, targets)
        torch.testing.assert_close(_restore_windows(shuffled, prediction), original.repeat_interleave(24, dim=0),
                                   rtol=0, atol=0)
        assert targets[13].tolist() == [2, 0, 3, 1]
        assert prediction[13].argsort().tolist() == [1, 3, 0, 2]

        # Assignment must solve the global constraint, even when all argmaxes collide.
        collision = torch.tensor([[[10., 9., 0., 0.], [10., 0., 8., 0.],
                                    [10., 0., 0., 7.], [20., 0., 0., 0.]]])
        assert collision.argmax(dim=2).tolist() == [[0, 0, 0, 0]]
        assert _valid_assignment(collision).tolist() == [[1, 2, 3, 0]]
        assert _valid_assignment(torch.zeros(1, 4, 4)).tolist() == [[0, 1, 2, 3]]

        # CE class dimension is the LAST axis; uniform loss is log(4), not log(24).
        zero_logits = torch.zeros(48, 4, 4, requires_grad=True)
        loss = F.cross_entropy(zero_logits.reshape(-1, 4), targets.reshape(-1))
        torch.testing.assert_close(loss, torch.tensor(math.log(4)))
        loss.backward()
        assert zero_logits.grad is not None and zero_logits.grad.abs().sum() > 0

        # Tiny real fit across separate trials; no cross-trial windows or data mutation.
        rng = np.random.default_rng(10)
        trials = [rng.normal(size=(13, 3)).astype(np.float32),
                  rng.normal(size=(17, 3)).astype(np.float32)]
        copies = [a.copy() for a in trials]
        params = dict(window_size=4, output_dimension=6, num_hidden_units=8,
                      head_hidden_units=12, batch_size=8, device="cpu", random_state=5, verbose=False)
        frozen = puzzle(**params, max_epochs=0).fit(trials)
        model = puzzle(**params, max_epochs=2).fit(trials)
        assert frozen.n_steps_ == 0 and frozen.history_ == []
        assert model.n_windows_ == 24 and model.n_steps_ == 6
        assert all(row['n_puzzles'] == 24 for row in model.history_)
        for trained_part, random_part in ((model.encoder_, frozen.encoder_), (model.head_, frozen.head_)):
            assert any(not torch.equal(a, b) for a, b in zip(trained_part.parameters(), random_part.parameters()))
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trained_part.parameters())
        assert all(np.array_equal(a, b) for a, b in zip(trials, copies))
        model.encoder_.train()
        model.head_.eval()
        torch_state = torch.random.get_rng_state().clone()
        m1, d1 = model.evaluate_puzzle(trials, max_windows=7, batch_size=1, return_details=True)
        m2, d2 = model.evaluate_puzzle(trials, max_windows=7, batch_size=31, return_details=True)
        assert model.encoder_.training and not model.head_.training
        assert torch.equal(torch_state, torch.random.get_rng_state())
        assert m1['n_puzzles'] == 7 * 24 and m1['permutation_counts'] == [7] * 24
        np.testing.assert_allclose(m1['position_cross_entropy'], m2['position_cross_entropy'], atol=1e-6)
        for key in ('window_starts', 'sequence_ids', 'true_positions', 'predicted_positions'):
            np.testing.assert_array_equal(d1[key], d2[key])
        assert np.all(np.sort(d1['predicted_positions'], axis=-1) == np.arange(4))
        for seq, start in zip(d1['sequence_ids'], d1['window_starts']):
            assert 0 <= start and start + 4 <= len(trials[int(seq)])

        padded = model.transform(trials, pad=True)
        valid, indices = model.transform(trials, pad=False, return_indices=True)
        for seq, zp, zv, times in zip(trials, padded, valid, indices):
            assert zp.shape == (len(seq), 6) and zv.shape == (len(seq) - 3, 6)
            np.testing.assert_array_equal(times, np.arange(2, len(seq) - 1))
            np.testing.assert_allclose(zp[times], zv, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(zp, model.transform(seq, batch_size=3), rtol=1e-5, atol=1e-6)

        # Public reorder API preserves float64 values, shape and channels exactly.
        windows = np.stack([trials[0][i:i + 4].T for i in range(5)]).astype(np.float64)
        windows += 1e-10
        before = windows.copy()
        ordered, positions = model.reorder(windows, return_positions=True)
        assert ordered.dtype == windows.dtype and ordered.shape == windows.shape
        for b in range(len(windows)):
            np.testing.assert_array_equal(ordered[b], windows[b][:, np.argsort(positions[b])])
        np.testing.assert_array_equal(windows, before)
        time_first = model.reorder(windows.transpose(0, 2, 1), layout="time_first")
        np.testing.assert_array_equal(time_first, ordered.transpose(0, 2, 1))
        np.testing.assert_array_equal(model(windows[0]), ordered[0])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'position_puzzle.pt'
            model.save(path)
            loaded = puzzle.load(path, device="cpu")
            np.testing.assert_allclose(loaded.transform(trials[0]), padded[0], rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(loaded.reorder(windows), ordered)
            assert loaded.get_params() == model.get_params()
        print("PASS: 24 shuffles/inverses, constrained assignment, four-class CE, real CPU "
              "fit/autograd, frozen control, trial boundaries, eval batching, label alignment, "
              "value-preserving reorder and checkpoint save/load.")
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        parser.print_help()

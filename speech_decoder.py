# CER & PER
import os
import sys
import math
import time
import numbers
import pickle
from typing import List, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from g2p_en import G2p
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import trange
from edit_distance import SequenceMatcher

from utils.constants import CEBRA_DIR
from utils.load_model_states import save_checkpoint, load_checkpoint

sys.path.insert(0, str(CEBRA_DIR))
# from cebra.models import (
#     Offset36Dropoutv2, Offset10Model, Offset36Dropoutv2BN,
#     Offset10ModelBN, Offset36Dropoutv205,
# )
from cebra.models import (
    Offset36Dropoutv2,
    Offset10Model,
)
import matplotlib.pyplot as plt
import cebra.attribution

# =====================================================================
# Charset -- verbatim from your professor's code (index 0 = CTC blank)
# =====================================================================
CHARS = [
    '>', ',', '?', '~', "'",
    'a', 'b', 'c', 'd', 'e', 'f', 'g',
    'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z',
]
BLANK_TOKEN = "<BLANK>"


class Charset:
    def __init__(self, symbols: List[str]):
        self.idx2sym = [BLANK_TOKEN] + symbols
        self.sym2idx = {s: i + 1 for i, s in enumerate(symbols)}
        self.sym2idx[BLANK_TOKEN] = 0

    @property
    def num_classes(self) -> int:
        return len(self.idx2sym)

    def text_to_int(self, text: str) -> List[int]:
        return [self.sym2idx[ch] for ch in text if ch in self.sym2idx]

    def int_to_text(self, ids: List[int]) -> str:
        return "".join(self.idx2sym[i] for i in ids if i != 0)

_g2p = G2p()

def text_to_phonemes(text: str) -> List[str]:
    text = text.replace(">", " ")
    cleaned = "".join(ch for ch in text if ch.isalpha() or ch in " '")
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return []
    phones = _g2p(cleaned)
    return [p for p in phones if p.strip() != ""]

charset = Charset(CHARS)


def text_to_char_ids(text: str) -> List[int]:
    """lower-case, map space -> '>' (see ASSUMPTION at top of file), drop
    anything not in the charset."""
    text = text.lower().replace(" ", ">")
    return charset.text_to_int(text)


# =====================================================================
# Config
# =====================================================================
# DEFAULT_ARGS = dict(
#     datasetPath="./data/competitionData/competitionData/train",
#     testDatasetPath="./data/competitionData/competitionData/competitionHoldOut",
#     out_dir="./outputs/ctc_char_run",
#     seed=42,

#     area_6v_channels=128,   # neural_dim = 2 * this (spikePow_6v + tx1_6v)
#     max_files=None,         # cap number of session-day .mat files, for a quick test run
#     # test_size=0.15,

#     # Encoder_Decoder / CEBRA
#     ceb_out=32,
#     kernel=8,
#     stride=4,
#     hidden=256,
#     layers=2,
#     dropout=0.4,
#     bidir=True,
#     cebra_unfolder=False,
#     gru=True,
#     gauss_in=True,
#     no_rnn=False,
#     ceb_bn=False,
#     cebra_window_10=True,   # True -> Offset10Model (window 10); False -> Offset36Dropoutv2

#     # optimization
#     batchSize=16,
#     lrStart=3e-4,
#     lrEnd=3e-5,
#     nBatch=50000,#epoch
#     l2_decay=1e-5,
#     temperature=0.1,
#     whiteNoiseSD=0.0,
#     constantOffsetSD=0.0,

#     # InfoNCE positive/negative sampling (see get_batch)
#     cont_batch=512,
#     offset=4,
#     sample_single=False,
#     random_dir=False,
#     random_offset=False,
#     all_ref=False,
#     lambda_contrastive=1.0,   # weight on the CEBRA contrastive term, professor's code uses 1.0

#     # adversarial training (optional, matches professor's PGD-on-input scheme)
#     adv=False,
#     adv_eps=5,
#     adv_norm="linf",
#     adv_steps=10,

#     eval_every=150,
# )

DEFAULT_ARGS = dict(
    # ============================================================
    # DATA
    # ============================================================
    datasetPath="./data/competitionData/competitionData/train",
    testDatasetPath="./data/competitionData/competitionData/competitionHoldOut",
    out_dir="./outputs/ctc_char_run",
    seed=0,

    area_6v_channels=128,
    max_files=None,

    # ============================================================
    # ENCODER / DECODER
    # matching professor setup
    # ============================================================
    ceb_out=32,
    kernel=32,
    stride=4,
    hidden=1024,
    layers=5,
    dropout=0.4,
    bidir=True,
    cebra_unfolder=False,
    gru=True,
    gauss_in=True,
    no_rnn=False,
    ceb_bn=False,
    cebra_window_10=True,

    # ============================================================
    # OPTIMIZATION
    # ============================================================
    batchSize=16,
    lrStart=3e-4,
    lrEnd=3e-5,
    nBatch=2000,
    l2_decay=1e-5,
    temperature=0.1,

    # ============================================================
    # INPUT AUGMENTATION
    # ============================================================
    whiteNoiseSD=0.8,
    constantOffsetSD=0.2,

    # ============================================================
    # INFONCE / CEBRA SAMPLING
    # ============================================================
    cont_batch=512,
    offset=4,
    sample_single=False,
    random_dir=True,
    random_offset=True,
    all_ref=False,
    lambda_contrastive=1.0,

    # ============================================================
    # ADVERSARIAL TRAINING
    # ============================================================
    adv=True,
    adv_eps=0.8,
    adv_norm="l2",
    adv_steps=10,

    # ============================================================
    # EVALUATION
    # ============================================================
    eval_every=150,
)



def cleanup(*objs):
    for o in objs:
        del o
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# =====================================================================
# Raw .mat loading (competitionData format) -- extends the single-trial
# attribution script's feature extraction to *all* trials in a session,
# with per-block z-scoring, exactly like that script did.
# =====================================================================
def cell_len(mat_cell):
    arr = np.asarray(mat_cell)
    return int(max(arr.shape))


def get_cell(mat_cell, idx):
    arr = np.asarray(mat_cell)
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            return arr[0, idx]
        elif arr.shape[1] == 1:
            return arr[idx, 0]
    return arr.flatten()[idx]


def decode_sentence_text(data, trial_idx):
    raw = np.asarray(get_cell(data["sentenceText"], trial_idx))
    try:
        chars = [chr(int(c)) for c in raw.flatten() if int(c) != 0]
        return "".join(chars).strip()
    except Exception:
        return str(raw)


def extract_features(data, trial_idx, area_6v_channels):
    spikePow_trial = np.asarray(get_cell(data["spikePow"], trial_idx), dtype=np.float32)
    tx1_trial = np.asarray(get_cell(data["tx1"], trial_idx), dtype=np.float32)
    spikePow_6v = spikePow_trial[:, :area_6v_channels]
    tx1_6v = tx1_trial[:, :area_6v_channels]
    return np.concatenate([spikePow_6v, tx1_6v], axis=1)


def load_session(mat_path, area_6v_channels):
    """One .mat file = one recording day. Returns list of (X [T,F], text)."""
    data = sio.loadmat(mat_path)
    n_trials = cell_len(data["spikePow"])
    block_ids = np.array([int(np.squeeze(get_cell(data["blockIdx"], i))) for i in range(n_trials)])
    feats_raw = [extract_features(data, i, area_6v_channels) for i in range(n_trials)]

    trials = [None] * n_trials
    for block in np.unique(block_ids):
        idx_in_block = np.where(block_ids == block)[0]
        all_feats = np.concatenate([feats_raw[i] for i in idx_in_block], axis=0)
        mu = all_feats.mean(axis=0, keepdims=True).astype(np.float32)
        sigma = (all_feats.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
        for i in idx_in_block:
            X = ((feats_raw[i] - mu) / sigma).astype(np.float32)
            text = decode_sentence_text(data, i)
            trials[i] = (X, text)
    return trials


def load_all_sessions(data_dir, area_6v_channels=128, max_files=None):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".mat"))
    if max_files is not None:
        files = files[:max_files]

    samples = []  # (X, text, day_idx)
    dropped_empty = 0
    for day_idx, fname in enumerate(files):
        path = os.path.join(data_dir, fname)
        print(f"loading day {day_idx}: {fname}")
        trials = load_session(path, area_6v_channels)
        for X, text in trials:
            ids = text_to_char_ids(text)
            if len(ids) == 0:
                dropped_empty += 1
                continue
            samples.append((X, text, day_idx))

    print(f"total usable trials: {len(samples)} across {len(files)} days "
          f"({dropped_empty} trials dropped: empty transcript after charset filtering)")
    return samples, files


# =====================================================================
# Dataset / collate (character ids pre-tokenized once at construction)
# =====================================================================
class BrainToTextCharDataset(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, str, int]]):
        self.items = []
        for X, text, day_idx in samples:
            ids = text_to_char_ids(text)
            x = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(ids, dtype=torch.long)
            self.items.append((
                x, y,
                torch.tensor(x.shape[0], dtype=torch.long),
                torch.tensor(y.shape[0], dtype=torch.long),
                day_idx,
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def ctc_collate(batch):
    """Pads x with its last frame (matches the professor's convention) and
    pads y with 0 (=blank id, ignored via target_lengths)."""
    xs, ys, input_lengths, target_lengths, sessions = zip(*batch)
    B = len(xs)
    feat_dim = xs[0].shape[-1]

    input_lengths = torch.stack(input_lengths)
    target_lengths = torch.stack(target_lengths)
    T_max = int(input_lengths.max())

    x_pad = torch.zeros(B, T_max, feat_dim, dtype=torch.float32)
    for i, x in enumerate(xs):
        T = x.shape[0]
        x_pad[i, :T] = x
        if T < T_max:
            x_pad[i, T:] = x[-1:]

    max_target_len = int(target_lengths.max())
    targets_padded = torch.zeros(B, max_target_len, dtype=torch.long)
    for i, y in enumerate(ys):
        L = y.shape[0]
        targets_padded[i, :L] = y

    sessions_t = torch.tensor(sessions, dtype=torch.int32)
    return x_pad, targets_padded, input_lengths, target_lengths, sessions_t


def get_dataset_loaders(datasetPath, testDatasetPath, batch_size, area_6v_channels, max_files, seed):
    train_samples, train_files = load_all_sessions(datasetPath, area_6v_channels, max_files)
    test_samples, test_files = load_all_sessions(testDatasetPath, area_6v_channels, max_files)

    if len(test_samples) == 0:
        raise RuntimeError(
            f"No usable trials in {testDatasetPath} after charset filtering. "
            f"competitionHoldOut sentence labels may be withheld for the competition -- "
            f"open one .mat file and check data['sentenceText'] directly before trusting this path."
        )

    train_ds = BrainToTextCharDataset(train_samples)
    test_ds = BrainToTextCharDataset(test_samples)
    print(f"train trials: {len(train_ds)} (from {datasetPath})")
    print(f"test trials: {len(test_ds)} (from {testDatasetPath})")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=ctc_collate, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=ctc_collate,
    )
    return train_loader, test_loader, test_samples, test_files

# =====================================================================
# Model pieces -- copied from your professor's code (GaussianSmoothing,
# Unfolder, Encoder_Decoder), only the CEBRA import path was adjusted to
# use CEBRA_DIR like the rest of your codebase.
# =====================================================================
class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super().__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing="ij"
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (1 / (std * math.sqrt(2 * math.pi))
                       * torch.exp(-(((mgrid - mean) / std) ** 2) / 2))
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(f"Only 1,2,3 dims supported, got {dim}")

    def forward(self, input):
        input = torch.permute(input, (0, 2, 1))
        input = self.conv(input, weight=self.weight, groups=self.groups, padding="same")
        input = torch.permute(input, (0, 2, 1))
        return input


class Unfolder(nn.Module):
    def __init__(self, kernel, stride):
        super().__init__()
        self.unfolder = torch.nn.Unfold((kernel, 1), dilation=1, padding=0, stride=stride)
        self.kernel = kernel
        self.stride = stride

    def forward(self, x, lengths):
        x = torch.permute(
            self.unfolder(torch.unsqueeze(torch.permute(x, (0, 2, 1)), 3)),
            (0, 2, 1),
        )
        lengths = ((lengths - self.kernel) / self.stride).to(torch.int32)
        return x, lengths


class Encoder_Decoder(nn.Module):
    """Input: (B,T,F) -> CTC logits (B,T,C) (permuted to (T,B,C) before CTCLoss)."""

    def __init__(self, neural_dim, cebra_out_dim, kernel, stride, num_classes,
                 rnn_hidden, rnn_layers, rnn_dr=0.4, rnn_bidir=True,
                 cebra_unfolder=False, gru=False, smooth_width=2.0, gauss_in=True,
                 no_rnn=False, cebra_window_10=False, cebra_bn=False):
        super().__init__()

        # def init_cebra(in_features):
        #     if cebra_window_10:
        #         self.left_of = 5
        #         ceb_model = Offset10ModelBN if cebra_bn else Offset10Model
        #     else:
        #         self.left_of = 18
        #         ceb_model = Offset36Dropoutv2
        #     return ceb_model(in_features, 256, cebra_out_dim)
         
        def init_cebra(in_features):
            if cebra_window_10:
                self.left_of = 5
                ceb_model = Offset10Model
            else:
                self.left_of = 18
                ceb_model = Offset36Dropoutv2
      
            return ceb_model(in_features, 256, cebra_out_dim)
           
        current_dim = neural_dim
        self.cebra_unfolder = cebra_unfolder
        self.smoother = GaussianSmoothing(neural_dim, 20, smooth_width, dim=1) if gauss_in else nn.Identity()

        if cebra_unfolder:
            self.cebra = init_cebra(current_dim)
            current_dim = cebra_out_dim

        self.unfolder = Unfolder(kernel, stride)
        current_dim *= kernel

        if not cebra_unfolder:
            self.cebra = init_cebra(current_dim)
            current_dim = cebra_out_dim

        if not no_rnn:
            rnn_cls = nn.GRU if gru else nn.LSTM
            self.rnn = rnn_cls(current_dim, rnn_hidden, rnn_layers, batch_first=True,
                                bidirectional=rnn_bidir, dropout=rnn_dr)
            current_dim = rnn_hidden * (2 if rnn_bidir else 1)
        else:
            self.rnn = lambda x: (x, None)

        self.final_decoder = nn.Linear(current_dim, num_classes)

    def _apply_cebra(self, x, lengths):
        x = x.permute(0, 2, 1)
        x = F.pad(x, (self.left_of, self.left_of - 1), mode="replicate")
        x = self.cebra(x).permute(0, 2, 1)
        self.embeddings = x
        self.emb_lengths = lengths
        return x

    def get_cebra_embs(self):
        return self.embeddings, self.emb_lengths

    def forward(self, x, lengths):
        x = self.smoother(x)
        if self.cebra_unfolder:
            x = self._apply_cebra(x, lengths)
        x, lengths = self.unfolder(x, lengths)
        if not self.cebra_unfolder:
            x = self._apply_cebra(x, lengths)
        x, _ = self.rnn(x)
        x = self.final_decoder(x)
        return x, lengths

class CebraFromRawInput(nn.Module):
    """Wraps just smoother -> unfolder -> cebra, so attribution sees CEBRA's
    own output as a function of the RAW neural input, not the CTC logits."""
    def __init__(self, encoder_decoder):
        super().__init__()
        self.ed = encoder_decoder

    def forward(self, x):
        ed = self.ed
        lengths = torch.tensor([x.shape[1]] * x.shape[0], device=x.device)
        h = ed.smoother(x)
        if ed.cebra_unfolder:
            h = ed._apply_cebra(h, lengths)
            h, lengths = ed.unfolder(h, lengths)
        else:
            h, lengths = ed.unfolder(h, lengths)
            h = ed._apply_cebra(h, lengths)
        return h

def reduce_attr_map(arr):
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    arr = np.abs(np.asarray(arr))
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    elif arr.ndim == 1:
        arr = arr[None, :]
    return arr.astype(np.float32)


def save_heatmap(arr, path, title, feature_boundary=None):
    plt.figure(figsize=(10, 6))
    plt.imshow(arr, aspect="auto", cmap="viridis")
    plt.colorbar(label="absolute attribution")
    if feature_boundary is not None:
        plt.axvline(feature_boundary, color="white", linestyle="--", linewidth=1)
    plt.xlabel("Neural feature / channel (spikePow 6v | tx1 6v)")
    plt.ylabel("Latent dimension")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved:", path)

def run_attribution(model, raw_X, area_6v_channels, ceb_out_dim, out_dir, device, tag):
    m = model.module if isinstance(model, torch.nn.DataParallel) else model
    wrapper = CebraFromRawInput(m).to(device)
    wrapper.eval()

    x_tensor = torch.tensor(raw_X, dtype=torch.float32, device=device).unsqueeze(0)
    x_tensor.requires_grad_(True)

    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=wrapper,
        input_data=x_tensor,
        output_dimension=ceb_out_dim,
    )
    
    result = method.compute_attribution_map(batch_size=x_tensor.shape[0])
    
    jf = result["jf"]
    jf_inv = result.get("jf-inv-svd", result.get("jf-inv-lsq", result.get("jf-inv")))

    jf_matrix = reduce_attr_map(jf)
    jf_inv_matrix = reduce_attr_map(jf_inv)

    torch.save(jf, os.path.join(out_dir, f"{tag}_jf.pt"))
    torch.save(jf_inv, os.path.join(out_dir, f"{tag}_jf_inv.pt"))
    save_heatmap(jf_matrix, os.path.join(out_dir, f"{tag}_jf.png"),
                 f"{tag} - Jacobian", feature_boundary=area_6v_channels)
    save_heatmap(jf_inv_matrix, os.path.join(out_dir, f"{tag}_jf_inv.png"),
                 f"{tag} - Inverse Jacobian", feature_boundary=area_6v_channels)

    cleanup(wrapper, x_tensor, method, result)


   
# =====================================================================
# InfoNCE + positive/negative sampler -- copied from professor's code
# =====================================================================
@torch.jit.script
def dot_similarity(ref: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
    pos_dist = torch.einsum("ni,ni->n", ref, pos)
    neg_dist = torch.einsum("ni,mi->nm", ref, neg)
    return pos_dist, neg_dist


@torch.jit.script
def infonce(pos_dist: torch.Tensor, neg_dist: torch.Tensor):
    with torch.no_grad():
        c, _ = neg_dist.max(dim=1, keepdim=True)
    c = c.detach()
    pos_dist = pos_dist - c.squeeze(1)
    neg_dist = neg_dist - c
    align = (-pos_dist).mean()
    uniform = torch.logsumexp(neg_dist, dim=1).mean()
    c_mean = c.mean()
    return align + uniform, align - c_mean, uniform + c_mean


class InfoNCE(nn.Module):
    def __init__(self, temp) -> None:
        super().__init__()
        self.temperature = temp

    def _distance(self, ref, pos, neg):
        pos_dist, neg_dist = dot_similarity(ref, pos, neg)
        return pos_dist / self.temperature, neg_dist / self.temperature

    def forward(self, ref, pos, neg):
        pos_dist, neg_dist = self._distance(ref, pos, neg)
        return infonce(pos_dist, neg_dist)


def get_batch(x, x_len, batch_size, offset, single_sequence=False,
              random_offset=False, random_dir=False, all_ref=False):
    B, T, F_ = x.shape
    device = x.device

    if all_ref:
        time_range = torch.arange(T, device=device).unsqueeze(0)
        valid_mask = time_range < x_len.unsqueeze(1)
        ref_batch_idx, ref_time_idx = torch.where(valid_mask)
        if single_sequence:
            chosen_batch = torch.randint(0, B, (1,)).item()
            mask = ref_batch_idx == chosen_batch
            ref_batch_idx, ref_time_idx = ref_batch_idx[mask], ref_time_idx[mask]
    else:
        if not single_sequence:
            ref_batch_idx = torch.randint(0, B, (batch_size,), device=device)
            max_times = torch.clamp(x_len[ref_batch_idx] - offset - 1, min=1)
            ref_time_idx = torch.randint(0 if not random_dir else offset,
                                          torch.max(max_times).item(), (batch_size,), device=device)
            ref_time_idx = torch.min(ref_time_idx, max_times - 1)
        else:
            pos_batch = torch.randint(0, B, (1,), device=device).item()
            ref_batch_idx = torch.full((batch_size,), pos_batch, device=device)
            max_times = torch.clamp(x_len[pos_batch] - offset - 1, min=1)
            ref_time_idx = torch.randint(0 if not random_dir else offset,
                                          max_times.item(), (batch_size,), device=device)

    if random_offset:
        add_offset = torch.randint(1, offset + 1, (len(ref_batch_idx),), device=device, dtype=torch.long)
    else:
        add_offset = torch.full((len(ref_batch_idx),), offset, device=device, dtype=torch.long)
    if random_dir:
        dir_val = torch.randint(0, 2, add_offset.shape, device=device) * 2 - 1
        add_offset = add_offset * dir_val

    pos_time_idx = ref_time_idx + add_offset
    max_valid = x_len[ref_batch_idx] - 1
    min_val = torch.zeros_like(max_valid)
    pos_time_idx = torch.clamp(pos_time_idx, min=min_val, max=max_valid)

    if single_sequence:
        neg_batch_idx = torch.randint(0, B, (len(ref_batch_idx),), device=device)
        if B > 1:
            mask = neg_batch_idx == ref_batch_idx[0]
            neg_batch_idx[mask] = (neg_batch_idx[mask] + 1) % B
    else:
        neg_batch_idx = torch.randint(0, B, (len(ref_batch_idx),), device=device)

    neg_max_times = x_len[neg_batch_idx]
    neg_time_idx = torch.randint(0, torch.max(neg_max_times).item(), (len(ref_batch_idx),), device=device)
    neg_time_idx = torch.min(neg_time_idx, neg_max_times - 1)

    reference = x[ref_batch_idx, ref_time_idx]
    positive = x[ref_batch_idx, pos_time_idx]
    negative = x[neg_batch_idx, neg_time_idx]
    return (reference, positive, negative, ref_batch_idx, ref_time_idx,
            pos_time_idx, neg_batch_idx, neg_time_idx)


# =====================================================================
# Training loop -- adapted from professor's train_model(args)
# =====================================================================
def evaluate_metrics(model, loader, ctc_criterion, device, compute_per=False):
    model.eval()
    allLoss = []
    total_char_edit, total_char_len = 0, 0
    total_phone_edit, total_phone_len = 0, 0
    with torch.no_grad():
        for Xv, yv, Xv_len, yv_len, _ in loader:
            Xv, yv, Xv_len, yv_len = Xv.to(device), yv.to(device), Xv_len.to(device), yv_len.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                pred_v, lengths_v = model(Xv, Xv_len)
                loss_v = torch.sum(ctc_criterion(
                    torch.permute(pred_v.log_softmax(2), [1, 0, 2]), yv, lengths_v, yv_len))
            allLoss.append(loss_v.cpu().item())

            for i in range(pred_v.shape[0]):
                # decoded_ids = torch.argmax(pred_v[i, :lengths_v[i], :], dim=-1)
                # decoded_ids = torch.unique_consecutive(decoded_ids)
                # decoded_ids = [c for c in decoded_ids.cpu().numpy().tolist() if c != 0]
                raw_ids = torch.argmax(pred_v[i, :lengths_v[i], :], dim=-1).cpu().numpy().tolist()
                collapsed = []
                for c in raw_ids:
                    if not collapsed or c != collapsed[-1]:
                        collapsed.append(c)
                
                decoded_ids = [c for c in collapsed if c != 0]
                true_ids = yv[i][:yv_len[i]].cpu().numpy().tolist()
        
                matcher = SequenceMatcher(a=true_ids, b=decoded_ids)
                total_char_edit += matcher.distance()
                total_char_len += len(true_ids)

                if compute_per:
                    pred_text = charset.int_to_text(decoded_ids)
                    true_text = charset.int_to_text(true_ids)
                    pred_phones = text_to_phonemes(pred_text)
                    true_phones = text_to_phonemes(true_text)
                    if len(true_phones) > 0:
                        pm = SequenceMatcher(a=true_phones, b=pred_phones)
                        total_phone_edit += pm.distance()
                        total_phone_len += len(true_phones)

    avgLoss = float(np.sum(allLoss) / max(len(loader), 1))
    cer = total_char_edit / max(total_char_len, 1)
    per = (total_phone_edit / total_phone_len) if (compute_per and total_phone_len > 0) else None
    return avgLoss, cer, per
    

def train_model(args: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args["out_dir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    checkpoint_address = os.path.join(args["out_dir"], "checkpoint.pt")

    neural_dim = 2 * args["area_6v_channels"]
    num_classes = charset.num_classes  # <- explicit, character-level (NOT the 41/32 hardcode)
    print(f"neural_dim={neural_dim} | num_classes={num_classes} (charset, incl. blank)")

    model = Encoder_Decoder(
        neural_dim, args["ceb_out"], args["kernel"], args["stride"], num_classes,
        args["hidden"], args["layers"], args["dropout"], args["bidir"],
        args["cebra_unfolder"], args["gru"], smooth_width=2.0,
        gauss_in=args["gauss_in"], no_rnn=args["no_rnn"],
        cebra_bn=args["ceb_bn"], cebra_window_10=args["cebra_window_10"],
    ).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    with open(os.path.join(args["out_dir"], "args"), "wb") as f:
        pickle.dump(args, f)

    train_loader, test_loader, test_samples, test_files = get_dataset_loaders(
        args["datasetPath"], args["testDatasetPath"], args["batchSize"],
        args["area_6v_channels"], args["max_files"], args["seed"],
    )
    
    criterion = InfoNCE(args["temperature"])
    ctc_criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lrStart"],
                                  betas=(0.9, 0.999), eps=0.1, weight_decay=args["l2_decay"])
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nBatch"],
    )

    so_far_batch = load_checkpoint(checkpoint_address, model, optimizer, scheduler)
    print("resuming from batch:", so_far_batch)

    inf_losses = 0
    testLoss, testCER = [], []
    train_iter = iter(train_loader)

    for batch in trange(args["nBatch"]):
        model.train()
        try:
            X, y, X_len, y_len, dayIdx = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X, y, X_len, y_len, dayIdx = next(train_iter)

        X, y, X_len, y_len, dayIdx = (X.to(device), y.to(device), X_len.to(device),
                                       y_len.to(device), dayIdx.to(device))
        if batch < so_far_batch:
            continue

        if args["whiteNoiseSD"] > 0:
            X = X + torch.randn(X.shape, device=device) * args["whiteNoiseSD"]
        if args["constantOffsetSD"] > 0:
            X = X + torch.randn([X.shape[0], 1, X.shape[2]], device=device) * args["constantOffsetSD"]

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            pred, lengths = model(X, X_len)
            m = model.module if isinstance(model, torch.nn.DataParallel) else model
            embeddings, emb_lengths = m.get_cebra_embs()

            ctc_loss = torch.sum(ctc_criterion(
                torch.permute(pred.log_softmax(2), [1, 0, 2]), y, lengths, y_len))

            (reference, positive, negative, ref_b, ref_t, pos_t, neg_b, neg_t) = get_batch(
                embeddings, emb_lengths, args["cont_batch"], args["offset"],
                args["sample_single"], args["random_offset"], args["random_dir"], args["all_ref"],
            )
            loss_contrastive = criterion(reference, positive, negative)[0]
            loss = args["lambda_contrastive"] * loss_contrastive + ctc_loss

        optimizer.zero_grad()
        if not torch.isfinite(loss):
            inf_losses += 1
            if inf_losses > 10:
                break
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # ---------------- optional adversarial (PGD on raw input X) ----------------
        if args["adv"]:
            epsilon, steps, alpha = args["adv_eps"], args["adv_steps"], args["adv_eps"] / 5.0
            X_adv = X.detach().clone()
            if args["adv_norm"] == "linf":
                X_adv = X_adv + torch.empty_like(X_adv).uniform_(-epsilon, epsilon)
            else:  # l2
                noise = torch.randn_like(X_adv)
                noise_norm = noise.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                noise = noise / noise_norm
                noise = noise * (torch.rand((noise.shape[0], noise.shape[1], 1), device=device) * epsilon)
                X_adv = X_adv + noise

            for _ in range(steps):
                X_adv = X_adv.detach().requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    pred_adv, lengths_adv = model(X_adv, X_len)
                    m = model.module if isinstance(model, torch.nn.DataParallel) else model
                    emb_adv, emb_len_adv = m.get_cebra_embs()
                    ctc_loss_adv = torch.sum(ctc_criterion(
                        torch.permute(pred_adv.log_softmax(2), [1, 0, 2]), y, lengths_adv, y_len))
                    ref = emb_adv[ref_b, ref_t]
                    pos = emb_adv[ref_b, pos_t].detach()
                    neg = emb_adv[neg_b, neg_t].detach()
                    loss_adv = args["lambda_contrastive"] * criterion(ref, pos, neg)[0] + ctc_loss_adv

                grad = torch.autograd.grad(loss_adv, X_adv, only_inputs=True)[0]
                with torch.no_grad():
                    if args["adv_norm"] == "linf":
                        X_adv = X_adv + alpha * grad.sign()
                        delta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
                        X_adv = X + delta
                    else:
                        grad_norm = grad.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                        X_adv = (X_adv + alpha * (grad / grad_norm)).detach()
                        delta = X_adv - X
                        delta_norm = delta.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                        scale = torch.clamp(epsilon / delta_norm, max=1.0)
                        X_adv = (X + delta * scale).detach()

            X_adv = X_adv.detach()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                pred_adv, lengths_adv = model(X_adv, X_len)
                m = model.module if isinstance(model, torch.nn.DataParallel) else model
                emb_adv, emb_len_adv = m.get_cebra_embs()
                ctc_loss_adv = torch.sum(ctc_criterion(
                    torch.permute(pred_adv.log_softmax(2), [1, 0, 2]), y, lengths_adv, y_len))
                ref = emb_adv[ref_b, ref_t]
                pos = emb_adv[ref_b, pos_t]
                neg = emb_adv[neg_b, neg_t]
                loss_adv = args["lambda_contrastive"] * criterion(ref, pos, neg)[0] + ctc_loss_adv

            optimizer.zero_grad()
            if torch.isfinite(loss_adv):
                loss_adv.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        scheduler.step()

        # ---------------- periodic validation (CER) ----------------
        if batch % args["eval_every"] == 0:
            model.eval()
            with torch.no_grad():
                allLoss, total_edit_distance, total_seq_length = [], 0, 0
                avgLoss, cer, _ = evaluate_metrics(model, test_loader, ctc_criterion, device, compute_per=False)
                print(f"batch {batch} | val ctc loss: {avgLoss:.4f} | CER: {cer:.4f} | train ctc: {ctc_loss.item():.4f} | train cont: {loss_contrastive.item():.4f}")
                # avgLoss = float(np.sum(allLoss) / max(len(test_loader), 1))
                # cer = total_edit_distance / max(total_seq_length, 1)
                # print(f"batch {batch} | val ctc loss: {avgLoss:.4f} | CER: {cer:.4f} "
                #       f"| train ctc: {ctc_loss.item():.4f} | train cont: {loss_contrastive.item():.4f}")

            state_dict = (model.module if isinstance(model, torch.nn.DataParallel) else model).state_dict()
            torch.save(state_dict, os.path.join(args["out_dir"], "modelWeights"))
            save_checkpoint(checkpoint_address, model, optimizer, scheduler, batch)
            

            testLoss.append(avgLoss)
            testCER.append(cer)
            with open(os.path.join(args["out_dir"], "trainingStats"), "wb") as f:
                pickle.dump({"testLoss": np.array(testLoss), "testCER": np.array(testCER)}, f)

    # print("DONE")
    # raw_X, _, _ = test_samples[0]
    # run_attribution(model, raw_X, args["area_6v_channels"], args["ceb_out"],
    #                  args["out_dir"], device, tag="CEBRA_trial0")
    # return model
    print("DONE")

    day_to_trial = {}
    for X, text, day_idx in test_samples:
        if day_idx not in day_to_trial:
            day_to_trial[day_idx] = X

    print(
        f"\nrunning attribution for {len(day_to_trial)} day(s) present in the test split "
        f"(out of {len(test_files)} total day files)"
    )
    
    
    final_loss, final_cer, final_per = evaluate_metrics(model, test_loader, ctc_criterion, device, compute_per=True)
    print("\n" + "=" * 60)
    print(f"FINAL TEST RESULTS ({args['testDatasetPath']})")
    print(f"  CTC loss: {final_loss:.4f}")
    print(f"  CER (character error rate): {final_cer:.4f}")
    print(f"  PER (phoneme error rate, via g2p_en): {final_per:.4f}" if final_per is not None else "  PER: N/A (no valid phoneme sequences)")
    print("=" * 60)

    for day_idx in sorted(day_to_trial.keys()):
        day_name = os.path.splitext(test_files[day_idx])[0]
        tag = f"CEBRA_day{day_idx}_{day_name}"
        run_attribution(
            model, day_to_trial[day_idx], args["area_6v_channels"], args["ceb_out"],
            args["out_dir"], device, tag=tag,
        )

    return model


if __name__ == "__main__":
    train_model(DEFAULT_ARGS)



# #only CER
# import os
# import sys
# import math
# import time
# import numbers
# import pickle
# from typing import List, Tuple

# import numpy as np
# import scipy.io as sio
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from tqdm import trange
# from edit_distance import SequenceMatcher

# from utils.constants import CEBRA_DIR
# from utils.load_model_states import save_checkpoint, load_checkpoint

# sys.path.insert(0, str(CEBRA_DIR))
# # from cebra.models import (
# #     Offset36Dropoutv2, Offset10Model, Offset36Dropoutv2BN,
# #     Offset10ModelBN, Offset36Dropoutv205,
# # )
# from cebra.models import (
#     Offset36Dropoutv2,
#     Offset10Model,
# )
# import matplotlib.pyplot as plt
# import cebra.attribution

# # =====================================================================
# # Charset -- verbatim from your professor's code (index 0 = CTC blank)
# # =====================================================================
# CHARS = [
#     '>', ',', '?', '~', "'",
#     'a', 'b', 'c', 'd', 'e', 'f', 'g',
#     'h', 'i', 'j', 'k', 'l', 'm', 'n',
#     'o', 'p', 'q', 'r', 's', 't',
#     'u', 'v', 'w', 'x', 'y', 'z',
# ]
# BLANK_TOKEN = "<BLANK>"


# class Charset:
#     def __init__(self, symbols: List[str]):
#         self.idx2sym = [BLANK_TOKEN] + symbols
#         self.sym2idx = {s: i + 1 for i, s in enumerate(symbols)}
#         self.sym2idx[BLANK_TOKEN] = 0

#     @property
#     def num_classes(self) -> int:
#         return len(self.idx2sym)

#     def text_to_int(self, text: str) -> List[int]:
#         return [self.sym2idx[ch] for ch in text if ch in self.sym2idx]

#     def int_to_text(self, ids: List[int]) -> str:
#         return "".join(self.idx2sym[i] for i in ids if i != 0)


# charset = Charset(CHARS)


# def text_to_char_ids(text: str) -> List[int]:
#     """lower-case, map space -> '>' (see ASSUMPTION at top of file), drop
#     anything not in the charset."""
#     text = text.lower().replace(" ", ">")
#     return charset.text_to_int(text)


# # =====================================================================
# # Config
# # =====================================================================
# DEFAULT_ARGS = dict(
#     datasetPath="./data/competitionData/competitionData/train",
#     testDatasetPath="./data/competitionData/competitionData/competitionHoldOut",
#     out_dir="./outputs/ctc_char_run",
#     seed=42,

#     area_6v_channels=128,   # neural_dim = 2 * this (spikePow_6v + tx1_6v)
#     max_files=None,         # cap number of session-day .mat files, for a quick test run
#     # test_size=0.15,

#     # Encoder_Decoder / CEBRA
#     ceb_out=32,
#     kernel=8,
#     stride=4,
#     hidden=256,
#     layers=2,
#     dropout=0.4,
#     bidir=True,
#     cebra_unfolder=False,
#     gru=True,
#     gauss_in=True,
#     no_rnn=False,
#     ceb_bn=False,
#     cebra_window_10=True,   # True -> Offset10Model (window 10); False -> Offset36Dropoutv2

#     # optimization
#     batchSize=16,
#     lrStart=3e-4,
#     lrEnd=3e-5,
#     nBatch=5,#epoch
#     l2_decay=1e-5,
#     temperature=0.1,
#     whiteNoiseSD=0.0,
#     constantOffsetSD=0.0,

#     # InfoNCE positive/negative sampling (see get_batch)
#     cont_batch=512,
#     offset=10,
#     sample_single=False,
#     random_dir=False,
#     random_offset=False,
#     all_ref=False,
#     lambda_contrastive=1.0,   # weight on the CEBRA contrastive term, professor's code uses 1.0

#     # adversarial training (optional, matches professor's PGD-on-input scheme)
#     adv=True,
#     adv_eps=5,
#     adv_norm="linf",
#     adv_steps=10,

#     eval_every=150,
# )


# def cleanup(*objs):
#     for o in objs:
#         del o
#     import gc
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()


# # =====================================================================
# # Raw .mat loading (competitionData format) -- extends the single-trial
# # attribution script's feature extraction to *all* trials in a session,
# # with per-block z-scoring, exactly like that script did.
# # =====================================================================
# def cell_len(mat_cell):
#     arr = np.asarray(mat_cell)
#     return int(max(arr.shape))


# def get_cell(mat_cell, idx):
#     arr = np.asarray(mat_cell)
#     if arr.ndim == 2:
#         if arr.shape[0] == 1:
#             return arr[0, idx]
#         elif arr.shape[1] == 1:
#             return arr[idx, 0]
#     return arr.flatten()[idx]


# def decode_sentence_text(data, trial_idx):
#     raw = np.asarray(get_cell(data["sentenceText"], trial_idx))
#     try:
#         chars = [chr(int(c)) for c in raw.flatten() if int(c) != 0]
#         return "".join(chars).strip()
#     except Exception:
#         return str(raw)


# def extract_features(data, trial_idx, area_6v_channels):
#     spikePow_trial = np.asarray(get_cell(data["spikePow"], trial_idx), dtype=np.float32)
#     tx1_trial = np.asarray(get_cell(data["tx1"], trial_idx), dtype=np.float32)
#     spikePow_6v = spikePow_trial[:, :area_6v_channels]
#     tx1_6v = tx1_trial[:, :area_6v_channels]
#     return np.concatenate([spikePow_6v, tx1_6v], axis=1)


# def load_session(mat_path, area_6v_channels):
#     """One .mat file = one recording day. Returns list of (X [T,F], text)."""
#     data = sio.loadmat(mat_path)
#     n_trials = cell_len(data["spikePow"])
#     block_ids = np.array([int(np.squeeze(get_cell(data["blockIdx"], i))) for i in range(n_trials)])
#     feats_raw = [extract_features(data, i, area_6v_channels) for i in range(n_trials)]

#     trials = [None] * n_trials
#     for block in np.unique(block_ids):
#         idx_in_block = np.where(block_ids == block)[0]
#         all_feats = np.concatenate([feats_raw[i] for i in idx_in_block], axis=0)
#         mu = all_feats.mean(axis=0, keepdims=True).astype(np.float32)
#         sigma = (all_feats.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
#         for i in idx_in_block:
#             X = ((feats_raw[i] - mu) / sigma).astype(np.float32)
#             text = decode_sentence_text(data, i)
#             trials[i] = (X, text)
#     return trials


# def load_all_sessions(data_dir, area_6v_channels=128, max_files=None):
#     files = sorted(f for f in os.listdir(data_dir) if f.endswith(".mat"))
#     if max_files is not None:
#         files = files[:max_files]

#     samples = []  # (X, text, day_idx)
#     dropped_empty = 0
#     for day_idx, fname in enumerate(files):
#         path = os.path.join(data_dir, fname)
#         print(f"loading day {day_idx}: {fname}")
#         trials = load_session(path, area_6v_channels)
#         for X, text in trials:
#             ids = text_to_char_ids(text)
#             if len(ids) == 0:
#                 dropped_empty += 1
#                 continue
#             samples.append((X, text, day_idx))

#     print(f"total usable trials: {len(samples)} across {len(files)} days "
#           f"({dropped_empty} trials dropped: empty transcript after charset filtering)")
#     return samples, files


# # =====================================================================
# # Dataset / collate (character ids pre-tokenized once at construction)
# # =====================================================================
# class BrainToTextCharDataset(Dataset):
#     def __init__(self, samples: List[Tuple[np.ndarray, str, int]]):
#         self.items = []
#         for X, text, day_idx in samples:
#             ids = text_to_char_ids(text)
#             x = torch.tensor(X, dtype=torch.float32)
#             y = torch.tensor(ids, dtype=torch.long)
#             self.items.append((
#                 x, y,
#                 torch.tensor(x.shape[0], dtype=torch.long),
#                 torch.tensor(y.shape[0], dtype=torch.long),
#                 day_idx,
#             ))

#     def __len__(self):
#         return len(self.items)

#     def __getitem__(self, idx):
#         return self.items[idx]


# def ctc_collate(batch):
#     """Pads x with its last frame (matches the professor's convention) and
#     pads y with 0 (=blank id, ignored via target_lengths)."""
#     xs, ys, input_lengths, target_lengths, sessions = zip(*batch)
#     B = len(xs)
#     feat_dim = xs[0].shape[-1]

#     input_lengths = torch.stack(input_lengths)
#     target_lengths = torch.stack(target_lengths)
#     T_max = int(input_lengths.max())

#     x_pad = torch.zeros(B, T_max, feat_dim, dtype=torch.float32)
#     for i, x in enumerate(xs):
#         T = x.shape[0]
#         x_pad[i, :T] = x
#         if T < T_max:
#             x_pad[i, T:] = x[-1:]

#     max_target_len = int(target_lengths.max())
#     targets_padded = torch.zeros(B, max_target_len, dtype=torch.long)
#     for i, y in enumerate(ys):
#         L = y.shape[0]
#         targets_padded[i, :L] = y

#     sessions_t = torch.tensor(sessions, dtype=torch.int32)
#     return x_pad, targets_padded, input_lengths, target_lengths, sessions_t


# def get_dataset_loaders(datasetPath, testDatasetPath, batch_size, area_6v_channels, max_files, seed):
#     train_samples, train_files = load_all_sessions(datasetPath, area_6v_channels, max_files)
#     test_samples, test_files = load_all_sessions(testDatasetPath, area_6v_channels, max_files)

#     if len(test_samples) == 0:
#         raise RuntimeError(
#             f"No usable trials in {testDatasetPath} after charset filtering. "
#             f"competitionHoldOut sentence labels may be withheld for the competition -- "
#             f"open one .mat file and check data['sentenceText'] directly before trusting this path."
#         )

#     train_ds = BrainToTextCharDataset(train_samples)
#     test_ds = BrainToTextCharDataset(test_samples)
#     print(f"train trials: {len(train_ds)} (from {datasetPath})")
#     print(f"test trials: {len(test_ds)} (from {testDatasetPath})")

#     train_loader = DataLoader(
#         train_ds, batch_size=batch_size, shuffle=True,
#         num_workers=4, pin_memory=True, collate_fn=ctc_collate, persistent_workers=True,
#     )
#     test_loader = DataLoader(
#         test_ds, batch_size=batch_size, shuffle=False,
#         num_workers=0, pin_memory=True, collate_fn=ctc_collate,
#     )
#     return train_loader, test_loader, test_samples, test_files

# # =====================================================================
# # Model pieces -- copied from your professor's code (GaussianSmoothing,
# # Unfolder, Encoder_Decoder), only the CEBRA import path was adjusted to
# # use CEBRA_DIR like the rest of your codebase.
# # =====================================================================
# class GaussianSmoothing(nn.Module):
#     def __init__(self, channels, kernel_size, sigma, dim=2):
#         super().__init__()
#         if isinstance(kernel_size, numbers.Number):
#             kernel_size = [kernel_size] * dim
#         if isinstance(sigma, numbers.Number):
#             sigma = [sigma] * dim

#         kernel = 1
#         meshgrids = torch.meshgrid(
#             [torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing="ij"
#         )
#         for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
#             mean = (size - 1) / 2
#             kernel *= (1 / (std * math.sqrt(2 * math.pi))
#                        * torch.exp(-(((mgrid - mean) / std) ** 2) / 2))
#         kernel = kernel / torch.sum(kernel)
#         kernel = kernel.view(1, 1, *kernel.size())
#         kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

#         self.register_buffer("weight", kernel)
#         self.groups = channels
#         if dim == 1:
#             self.conv = F.conv1d
#         elif dim == 2:
#             self.conv = F.conv2d
#         elif dim == 3:
#             self.conv = F.conv3d
#         else:
#             raise RuntimeError(f"Only 1,2,3 dims supported, got {dim}")

#     def forward(self, input):
#         input = torch.permute(input, (0, 2, 1))
#         input = self.conv(input, weight=self.weight, groups=self.groups, padding="same")
#         input = torch.permute(input, (0, 2, 1))
#         return input


# class Unfolder(nn.Module):
#     def __init__(self, kernel, stride):
#         super().__init__()
#         self.unfolder = torch.nn.Unfold((kernel, 1), dilation=1, padding=0, stride=stride)
#         self.kernel = kernel
#         self.stride = stride

#     def forward(self, x, lengths):
#         x = torch.permute(
#             self.unfolder(torch.unsqueeze(torch.permute(x, (0, 2, 1)), 3)),
#             (0, 2, 1),
#         )
#         lengths = ((lengths - self.kernel) / self.stride).to(torch.int32)
#         return x, lengths


# class Encoder_Decoder(nn.Module):
#     """Input: (B,T,F) -> CTC logits (B,T,C) (permuted to (T,B,C) before CTCLoss)."""

#     def __init__(self, neural_dim, cebra_out_dim, kernel, stride, num_classes,
#                  rnn_hidden, rnn_layers, rnn_dr=0.4, rnn_bidir=True,
#                  cebra_unfolder=False, gru=False, smooth_width=2.0, gauss_in=True,
#                  no_rnn=False, cebra_window_10=False, cebra_bn=False):
#         super().__init__()

#         # def init_cebra(in_features):
#         #     if cebra_window_10:
#         #         self.left_of = 5
#         #         ceb_model = Offset10ModelBN if cebra_bn else Offset10Model
#         #     else:
#         #         self.left_of = 18
#         #         ceb_model = Offset36Dropoutv2
#         #     return ceb_model(in_features, 256, cebra_out_dim)
         
#         def init_cebra(in_features):
#             if cebra_window_10:
#                 self.left_of = 5
#                 ceb_model = Offset10Model
#             else:
#                 self.left_of = 18
#                 ceb_model = Offset36Dropoutv2
      
#             return ceb_model(in_features, 256, cebra_out_dim)
           
#         current_dim = neural_dim
#         self.cebra_unfolder = cebra_unfolder
#         self.smoother = GaussianSmoothing(neural_dim, 20, smooth_width, dim=1) if gauss_in else nn.Identity()

#         if cebra_unfolder:
#             self.cebra = init_cebra(current_dim)
#             current_dim = cebra_out_dim

#         self.unfolder = Unfolder(kernel, stride)
#         current_dim *= kernel

#         if not cebra_unfolder:
#             self.cebra = init_cebra(current_dim)
#             current_dim = cebra_out_dim

#         if not no_rnn:
#             rnn_cls = nn.GRU if gru else nn.LSTM
#             self.rnn = rnn_cls(current_dim, rnn_hidden, rnn_layers, batch_first=True,
#                                 bidirectional=rnn_bidir, dropout=rnn_dr)
#             current_dim = rnn_hidden * (2 if rnn_bidir else 1)
#         else:
#             self.rnn = lambda x: (x, None)

#         self.final_decoder = nn.Linear(current_dim, num_classes)

#     def _apply_cebra(self, x, lengths):
#         x = x.permute(0, 2, 1)
#         x = F.pad(x, (self.left_of, self.left_of - 1), mode="replicate")
#         x = self.cebra(x).permute(0, 2, 1)
#         self.embeddings = x
#         self.emb_lengths = lengths
#         return x

#     def get_cebra_embs(self):
#         return self.embeddings, self.emb_lengths

#     def forward(self, x, lengths):
#         x = self.smoother(x)
#         if self.cebra_unfolder:
#             x = self._apply_cebra(x, lengths)
#         x, lengths = self.unfolder(x, lengths)
#         if not self.cebra_unfolder:
#             x = self._apply_cebra(x, lengths)
#         x, _ = self.rnn(x)
#         x = self.final_decoder(x)
#         return x, lengths

# class CebraFromRawInput(nn.Module):
#     """Wraps just smoother -> unfolder -> cebra, so attribution sees CEBRA's
#     own output as a function of the RAW neural input, not the CTC logits."""
#     def __init__(self, encoder_decoder):
#         super().__init__()
#         self.ed = encoder_decoder

#     def forward(self, x):
#         ed = self.ed
#         lengths = torch.tensor([x.shape[1]] * x.shape[0], device=x.device)
#         h = ed.smoother(x)
#         if ed.cebra_unfolder:
#             h = ed._apply_cebra(h, lengths)
#             h, lengths = ed.unfolder(h, lengths)
#         else:
#             h, lengths = ed.unfolder(h, lengths)
#             h = ed._apply_cebra(h, lengths)
#         return h

# def reduce_attr_map(arr):
#     if torch.is_tensor(arr):
#         arr = arr.detach().cpu().numpy()
#     arr = np.abs(np.asarray(arr))
#     if arr.ndim == 3:
#         arr = arr.mean(axis=0)
#     elif arr.ndim == 1:
#         arr = arr[None, :]
#     return arr.astype(np.float32)


# def save_heatmap(arr, path, title, feature_boundary=None):
#     plt.figure(figsize=(10, 6))
#     plt.imshow(arr, aspect="auto", cmap="viridis")
#     plt.colorbar(label="absolute attribution")
#     if feature_boundary is not None:
#         plt.axvline(feature_boundary, color="white", linestyle="--", linewidth=1)
#     plt.xlabel("Neural feature / channel (spikePow 6v | tx1 6v)")
#     plt.ylabel("Latent dimension")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print("saved:", path)

# def run_attribution(model, raw_X, area_6v_channels, ceb_out_dim, out_dir, device, tag):
#     m = model.module if isinstance(model, torch.nn.DataParallel) else model
#     wrapper = CebraFromRawInput(m).to(device)
#     wrapper.eval()

#     x_tensor = torch.tensor(raw_X, dtype=torch.float32, device=device).unsqueeze(0)
#     x_tensor.requires_grad_(True)

#     method = cebra.attribution.init(
#         name="jacobian-based-batched",
#         model=wrapper,
#         input_data=x_tensor,
#         output_dimension=ceb_out_dim,
#     )
    
#     result = method.compute_attribution_map(batch_size=x_tensor.shape[0])
    
#     jf = result["jf"]
#     jf_inv = result.get("jf-inv-svd", result.get("jf-inv-lsq", result.get("jf-inv")))

#     jf_matrix = reduce_attr_map(jf)
#     jf_inv_matrix = reduce_attr_map(jf_inv)

#     torch.save(jf, os.path.join(out_dir, f"{tag}_jf.pt"))
#     torch.save(jf_inv, os.path.join(out_dir, f"{tag}_jf_inv.pt"))
#     save_heatmap(jf_matrix, os.path.join(out_dir, f"{tag}_jf.png"),
#                  f"{tag} - Jacobian", feature_boundary=area_6v_channels)
#     save_heatmap(jf_inv_matrix, os.path.join(out_dir, f"{tag}_jf_inv.png"),
#                  f"{tag} - Inverse Jacobian", feature_boundary=area_6v_channels)

#     cleanup(wrapper, x_tensor, method, result)


   
# # =====================================================================
# # InfoNCE + positive/negative sampler -- copied from professor's code
# # =====================================================================
# @torch.jit.script
# def dot_similarity(ref: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
#     pos_dist = torch.einsum("ni,ni->n", ref, pos)
#     neg_dist = torch.einsum("ni,mi->nm", ref, neg)
#     return pos_dist, neg_dist


# @torch.jit.script
# def infonce(pos_dist: torch.Tensor, neg_dist: torch.Tensor):
#     with torch.no_grad():
#         c, _ = neg_dist.max(dim=1, keepdim=True)
#     c = c.detach()
#     pos_dist = pos_dist - c.squeeze(1)
#     neg_dist = neg_dist - c
#     align = (-pos_dist).mean()
#     uniform = torch.logsumexp(neg_dist, dim=1).mean()
#     c_mean = c.mean()
#     return align + uniform, align - c_mean, uniform + c_mean


# class InfoNCE(nn.Module):
#     def __init__(self, temp) -> None:
#         super().__init__()
#         self.temperature = temp

#     def _distance(self, ref, pos, neg):
#         pos_dist, neg_dist = dot_similarity(ref, pos, neg)
#         return pos_dist / self.temperature, neg_dist / self.temperature

#     def forward(self, ref, pos, neg):
#         pos_dist, neg_dist = self._distance(ref, pos, neg)
#         return infonce(pos_dist, neg_dist)


# def get_batch(x, x_len, batch_size, offset, single_sequence=False,
#               random_offset=False, random_dir=False, all_ref=False):
#     B, T, F_ = x.shape
#     device = x.device

#     if all_ref:
#         time_range = torch.arange(T, device=device).unsqueeze(0)
#         valid_mask = time_range < x_len.unsqueeze(1)
#         ref_batch_idx, ref_time_idx = torch.where(valid_mask)
#         if single_sequence:
#             chosen_batch = torch.randint(0, B, (1,)).item()
#             mask = ref_batch_idx == chosen_batch
#             ref_batch_idx, ref_time_idx = ref_batch_idx[mask], ref_time_idx[mask]
#     else:
#         if not single_sequence:
#             ref_batch_idx = torch.randint(0, B, (batch_size,), device=device)
#             max_times = torch.clamp(x_len[ref_batch_idx] - offset - 1, min=1)
#             ref_time_idx = torch.randint(0 if not random_dir else offset,
#                                           torch.max(max_times).item(), (batch_size,), device=device)
#             ref_time_idx = torch.min(ref_time_idx, max_times - 1)
#         else:
#             pos_batch = torch.randint(0, B, (1,), device=device).item()
#             ref_batch_idx = torch.full((batch_size,), pos_batch, device=device)
#             max_times = torch.clamp(x_len[pos_batch] - offset - 1, min=1)
#             ref_time_idx = torch.randint(0 if not random_dir else offset,
#                                           max_times.item(), (batch_size,), device=device)

#     if random_offset:
#         add_offset = torch.randint(1, offset + 1, (len(ref_batch_idx),), device=device, dtype=torch.long)
#     else:
#         add_offset = torch.full((len(ref_batch_idx),), offset, device=device, dtype=torch.long)
#     if random_dir:
#         dir_val = torch.randint(0, 2, add_offset.shape, device=device) * 2 - 1
#         add_offset = add_offset * dir_val

#     pos_time_idx = ref_time_idx + add_offset
#     max_valid = x_len[ref_batch_idx] - 1
#     min_val = torch.zeros_like(max_valid)
#     pos_time_idx = torch.clamp(pos_time_idx, min=min_val, max=max_valid)

#     if single_sequence:
#         neg_batch_idx = torch.randint(0, B, (len(ref_batch_idx),), device=device)
#         if B > 1:
#             mask = neg_batch_idx == ref_batch_idx[0]
#             neg_batch_idx[mask] = (neg_batch_idx[mask] + 1) % B
#     else:
#         neg_batch_idx = torch.randint(0, B, (len(ref_batch_idx),), device=device)

#     neg_max_times = x_len[neg_batch_idx]
#     neg_time_idx = torch.randint(0, torch.max(neg_max_times).item(), (len(ref_batch_idx),), device=device)
#     neg_time_idx = torch.min(neg_time_idx, neg_max_times - 1)

#     reference = x[ref_batch_idx, ref_time_idx]
#     positive = x[ref_batch_idx, pos_time_idx]
#     negative = x[neg_batch_idx, neg_time_idx]
#     return (reference, positive, negative, ref_batch_idx, ref_time_idx,
#             pos_time_idx, neg_batch_idx, neg_time_idx)


# # =====================================================================
# # Training loop -- adapted from professor's train_model(args)
# # =====================================================================
# def evaluate_cer(model, loader, ctc_criterion, device):
#     model.eval()
#     allLoss, total_edit_distance, total_seq_length = [], 0, 0
#     with torch.no_grad():
#         for Xv, yv, Xv_len, yv_len, _ in loader:
#             Xv, yv, Xv_len, yv_len = Xv.to(device), yv.to(device), Xv_len.to(device), yv_len.to(device)
#             with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
#                 pred_v, lengths_v = model(Xv, Xv_len)
#                 loss_v = torch.sum(ctc_criterion(
#                     torch.permute(pred_v.log_softmax(2), [1, 0, 2]), yv, lengths_v, yv_len))
#             allLoss.append(loss_v.cpu().item())
#             for i in range(pred_v.shape[0]):
#                 decoded = torch.argmax(pred_v[i, :lengths_v[i], :], dim=-1)
#                 decoded = torch.unique_consecutive(decoded)
#                 decoded = np.array([c for c in decoded.cpu().numpy() if c != 0])
#                 true_seq = yv[i][:yv_len[i]].cpu().numpy()
#                 matcher = SequenceMatcher(a=true_seq.tolist(), b=decoded.tolist())
#                 total_edit_distance += matcher.distance()
#                 total_seq_length += len(true_seq)
#     avgLoss = float(np.sum(allLoss) / max(len(loader), 1))
#     cer = total_edit_distance / max(total_seq_length, 1)
#     return avgLoss, cer
    
# def train_model(args: dict):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     os.makedirs(args["out_dir"], exist_ok=True)
#     torch.manual_seed(args["seed"])
#     np.random.seed(args["seed"])

#     checkpoint_address = os.path.join(args["out_dir"], "checkpoint.pt")

#     neural_dim = 2 * args["area_6v_channels"]
#     num_classes = charset.num_classes  # <- explicit, character-level (NOT the 41/32 hardcode)
#     print(f"neural_dim={neural_dim} | num_classes={num_classes} (charset, incl. blank)")

#     model = Encoder_Decoder(
#         neural_dim, args["ceb_out"], args["kernel"], args["stride"], num_classes,
#         args["hidden"], args["layers"], args["dropout"], args["bidir"],
#         args["cebra_unfolder"], args["gru"], smooth_width=2.0,
#         gauss_in=args["gauss_in"], no_rnn=args["no_rnn"],
#         cebra_bn=args["ceb_bn"], cebra_window_10=args["cebra_window_10"],
#     ).to(device)
#     if torch.cuda.device_count() > 1:
#         print(f"Using {torch.cuda.device_count()} GPUs")
#         model = torch.nn.DataParallel(model)

#     with open(os.path.join(args["out_dir"], "args"), "wb") as f:
#         pickle.dump(args, f)

#     train_loader, test_loader, test_samples, test_files = get_dataset_loaders(
#         args["datasetPath"], args["testDatasetPath"], args["batchSize"],
#         args["area_6v_channels"], args["max_files"], args["seed"],
#     )
    
#     criterion = InfoNCE(args["temperature"])
#     ctc_criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args["lrStart"],
#                                   betas=(0.9, 0.999), eps=0.1, weight_decay=args["l2_decay"])
#     scheduler = torch.optim.lr_scheduler.LinearLR(
#         optimizer, start_factor=1.0, end_factor=args["lrEnd"] / args["lrStart"],
#         total_iters=args["nBatch"],
#     )

#     so_far_batch = load_checkpoint(checkpoint_address, model, optimizer, scheduler)
#     print("resuming from batch:", so_far_batch)

#     inf_losses = 0
#     testLoss, testCER = [], []
#     train_iter = iter(train_loader)

#     for batch in trange(args["nBatch"]):
#         model.train()
#         try:
#             X, y, X_len, y_len, dayIdx = next(train_iter)
#         except StopIteration:
#             train_iter = iter(train_loader)
#             X, y, X_len, y_len, dayIdx = next(train_iter)

#         X, y, X_len, y_len, dayIdx = (X.to(device), y.to(device), X_len.to(device),
#                                        y_len.to(device), dayIdx.to(device))
#         if batch < so_far_batch:
#             continue

#         if args["whiteNoiseSD"] > 0:
#             X = X + torch.randn(X.shape, device=device) * args["whiteNoiseSD"]
#         if args["constantOffsetSD"] > 0:
#             X = X + torch.randn([X.shape[0], 1, X.shape[2]], device=device) * args["constantOffsetSD"]

#         with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
#             pred, lengths = model(X, X_len)
#             m = model.module if isinstance(model, torch.nn.DataParallel) else model
#             embeddings, emb_lengths = m.get_cebra_embs()

#             ctc_loss = torch.sum(ctc_criterion(
#                 torch.permute(pred.log_softmax(2), [1, 0, 2]), y, lengths, y_len))

#             (reference, positive, negative, ref_b, ref_t, pos_t, neg_b, neg_t) = get_batch(
#                 embeddings, emb_lengths, args["cont_batch"], args["offset"],
#                 args["sample_single"], args["random_offset"], args["random_dir"], args["all_ref"],
#             )
#             loss_contrastive = criterion(reference, positive, negative)[0]
#             loss = args["lambda_contrastive"] * loss_contrastive + ctc_loss

#         optimizer.zero_grad()
#         if not torch.isfinite(loss):
#             inf_losses += 1
#             if inf_losses > 10:
#                 break
#             continue
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#         optimizer.step()

#         # ---------------- optional adversarial (PGD on raw input X) ----------------
#         if args["adv"]:
#             epsilon, steps, alpha = args["adv_eps"], args["adv_steps"], args["adv_eps"] / 5.0
#             X_adv = X.detach().clone()
#             if args["adv_norm"] == "linf":
#                 X_adv = X_adv + torch.empty_like(X_adv).uniform_(-epsilon, epsilon)
#             else:  # l2
#                 noise = torch.randn_like(X_adv)
#                 noise_norm = noise.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
#                 noise = noise / noise_norm
#                 noise = noise * (torch.rand((noise.shape[0], noise.shape[1], 1), device=device) * epsilon)
#                 X_adv = X_adv + noise

#             for _ in range(steps):
#                 X_adv = X_adv.detach().requires_grad_(True)
#                 with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
#                     pred_adv, lengths_adv = model(X_adv, X_len)
#                     m = model.module if isinstance(model, torch.nn.DataParallel) else model
#                     emb_adv, emb_len_adv = m.get_cebra_embs()
#                     ctc_loss_adv = torch.sum(ctc_criterion(
#                         torch.permute(pred_adv.log_softmax(2), [1, 0, 2]), y, lengths_adv, y_len))
#                     ref = emb_adv[ref_b, ref_t]
#                     pos = emb_adv[ref_b, pos_t].detach()
#                     neg = emb_adv[neg_b, neg_t].detach()
#                     loss_adv = args["lambda_contrastive"] * criterion(ref, pos, neg)[0] + ctc_loss_adv

#                 grad = torch.autograd.grad(loss_adv, X_adv, only_inputs=True)[0]
#                 with torch.no_grad():
#                     if args["adv_norm"] == "linf":
#                         X_adv = X_adv + alpha * grad.sign()
#                         delta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
#                         X_adv = X + delta
#                     else:
#                         grad_norm = grad.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
#                         X_adv = (X_adv + alpha * (grad / grad_norm)).detach()
#                         delta = X_adv - X
#                         delta_norm = delta.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
#                         scale = torch.clamp(epsilon / delta_norm, max=1.0)
#                         X_adv = (X + delta * scale).detach()

#             X_adv = X_adv.detach()
#             with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
#                 pred_adv, lengths_adv = model(X_adv, X_len)
#                 m = model.module if isinstance(model, torch.nn.DataParallel) else model
#                 emb_adv, emb_len_adv = m.get_cebra_embs()
#                 ctc_loss_adv = torch.sum(ctc_criterion(
#                     torch.permute(pred_adv.log_softmax(2), [1, 0, 2]), y, lengths_adv, y_len))
#                 ref = emb_adv[ref_b, ref_t]
#                 pos = emb_adv[ref_b, pos_t]
#                 neg = emb_adv[neg_b, neg_t]
#                 loss_adv = args["lambda_contrastive"] * criterion(ref, pos, neg)[0] + ctc_loss_adv

#             optimizer.zero_grad()
#             if torch.isfinite(loss_adv):
#                 loss_adv.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#                 optimizer.step()

#         scheduler.step()

#         # ---------------- periodic validation (CER) ----------------
#         if batch % args["eval_every"] == 0:
#             model.eval()
#             with torch.no_grad():
#                 allLoss, total_edit_distance, total_seq_length = [], 0, 0
#                 avgLoss, cer = evaluate_cer(model, test_loader, ctc_criterion, device)
#                 print(f"batch {batch} | val ctc loss: {avgLoss:.4f} | CER: {cer:.4f} "f"| train ctc: {ctc_loss.item():.4f} | train cont: {loss_contrastive.item():.4f}")

#                 # avgLoss = float(np.sum(allLoss) / max(len(test_loader), 1))
#                 # cer = total_edit_distance / max(total_seq_length, 1)
#                 # print(f"batch {batch} | val ctc loss: {avgLoss:.4f} | CER: {cer:.4f} "
#                 #       f"| train ctc: {ctc_loss.item():.4f} | train cont: {loss_contrastive.item():.4f}")

#             state_dict = (model.module if isinstance(model, torch.nn.DataParallel) else model).state_dict()
#             torch.save(state_dict, os.path.join(args["out_dir"], "modelWeights"))
#             save_checkpoint(checkpoint_address, model, optimizer, scheduler, batch)
            

#             testLoss.append(avgLoss)
#             testCER.append(cer)
#             with open(os.path.join(args["out_dir"], "trainingStats"), "wb") as f:
#                 pickle.dump({"testLoss": np.array(testLoss), "testCER": np.array(testCER)}, f)

#     # print("DONE")
#     # raw_X, _, _ = test_samples[0]
#     # run_attribution(model, raw_X, args["area_6v_channels"], args["ceb_out"],
#     #                  args["out_dir"], device, tag="CEBRA_trial0")
#     # return model
#     print("DONE")

#     day_to_trial = {}
#     for X, text, day_idx in test_samples:
#         if day_idx not in day_to_trial:
#             day_to_trial[day_idx] = X

#     print(
#         f"\nrunning attribution for {len(day_to_trial)} day(s) present in the test split "
#         f"(out of {len(test_files)} total day files)"
#     )
    
    
#     final_loss, final_cer = evaluate_cer(model, test_loader, ctc_criterion, device)
#     print("\n" + "=" * 60)
#     print(f"FINAL TEST RESULTS ({args['testDatasetPath']})")
#     print(f"  CTC loss: {final_loss:.4f}")
#     print(f"  CER (character error rate): {final_cer:.4f}")
#     print("=" * 60)

#     for day_idx in sorted(day_to_trial.keys()):
#         day_name = os.path.splitext(test_files[day_idx])[0]
#         tag = f"CEBRA_day{day_idx}_{day_name}"
#         run_attribution(
#             model, day_to_trial[day_idx], args["area_6v_channels"], args["ceb_out"],
#             args["out_dir"], device, tag=tag,
#         )

#     return model


# if __name__ == "__main__":
#     train_model(DEFAULT_ARGS)




# import os
# import sys
# import math
# import time
# import numbers
# import pickle
# from typing import List, Tuple

# import numpy as np
# import scipy.io as sio
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from tqdm import trange
# from edit_distance import SequenceMatcher

# from utils.constants import CEBRA_DIR
# from utils.load_model_states import save_checkpoint, load_checkpoint

# sys.path.insert(0, str(CEBRA_DIR))
# # from cebra.models import (
# #     Offset36Dropoutv2, Offset10Model, Offset36Dropoutv2BN,
# #     Offset10ModelBN, Offset36Dropoutv205,
# # )
# from cebra.models import (
#     Offset36Dropoutv2,
#     Offset10Model,
# )
# import matplotlib.pyplot as plt
# import cebra.attribution

# # =====================================================================
# # Charset -- verbatim from your professor's code (index 0 = CTC blank)
# # =====================================================================
# CHARS = [
#     '>', ',', '?', '~', "'",
#     'a', 'b', 'c', 'd', 'e', 'f', 'g',
#     'h', 'i', 'j', 'k', 'l', 'm', 'n',
#     'o', 'p', 'q', 'r', 's', 't',
#     'u', 'v', 'w', 'x', 'y', 'z',
# ]
# BLANK_TOKEN = "<BLANK>"


# class Charset:
#     def __init__(self, symbols: List[str]):
#         self.idx2sym = [BLANK_TOKEN] + symbols
#         self.sym2idx = {s: i + 1 for i, s in enumerate(symbols)}
#         self.sym2idx[BLANK_TOKEN] = 0

#     @property
#     def num_classes(self) -> int:
#         return len(self.idx2sym)

#     def text_to_int(self, text: str) -> List[int]:
#         return [self.sym2idx[ch] for ch in text if ch in self.sym2idx]

#     def int_to_text(self, ids: List[int]) -> str:
#         return "".join(self.idx2sym[i] for i in ids if i != 0)


# charset = Charset(CHARS)


# def text_to_char_ids(text: str) -> List[int]:
#     """lower-case, map space -> '>' (see ASSUMPTION at top of file), drop
#     anything not in the charset."""
#     text = text.lower().replace(" ", ">")
#     return charset.text_to_int(text)


# # =====================================================================
# # Config
# # =====================================================================
# DEFAULT_ARGS = dict(
#     datasetPath="./data/competitionData/competitionData/train",
#     out_dir="./outputs/ctc_char_run",
#     seed=42,

#     area_6v_channels=128,   # neural_dim = 2 * this (spikePow_6v + tx1_6v)
#     max_files=None,         # cap number of session-day .mat files, for a quick test run
#     test_size=0.15,

#     # Encoder_Decoder / CEBRA
#     ceb_out=32,
#     kernel=8,
#     stride=4,
#     hidden=256,
#     layers=2,
#     dropout=0.4,
#     bidir=True,
#     cebra_unfolder=False,
#     gru=True,
#     gauss_in=True,
#     no_rnn=False,
#     ceb_bn=False,
#     cebra_window_10=True,   # True -> Offset10Model (window 10); False -> Offset36Dropoutv2

#     # optimization
#     batchSize=16,
#     lrStart=3e-4,
#     lrEnd=3e-5,
#     nBatch=50000,#epoch
#     l2_decay=1e-5,
#     temperature=0.1,
#     whiteNoiseSD=0.0,
#     constantOffsetSD=0.0,

#     # InfoNCE positive/negative sampling (see get_batch)
#     cont_batch=512,
#     offset=10,
#     sample_single=False,
#     random_dir=False,
#     random_offset=False,
#     all_ref=False,
#     lambda_contrastive=1.0,   # weight on the CEBRA contrastive term, professor's code uses 1.0

#     # adversarial training (optional, matches professor's PGD-on-input scheme)
#     adv=True,
#     adv_eps=5,
#     adv_norm="linf",
#     adv_steps=10,

#     eval_every=150,
# )


# def cleanup(*objs):
#     for o in objs:
#         del o
#     import gc
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()


# # =====================================================================
# # Raw .mat loading (competitionData format) -- extends the single-trial
# # attribution script's feature extraction to *all* trials in a session,
# # with per-block z-scoring, exactly like that script did.
# # =====================================================================
# def cell_len(mat_cell):
#     arr = np.asarray(mat_cell)
#     return int(max(arr.shape))


# def get_cell(mat_cell, idx):
#     arr = np.asarray(mat_cell)
#     if arr.ndim == 2:
#         if arr.shape[0] == 1:
#             return arr[0, idx]
#         elif arr.shape[1] == 1:
#             return arr[idx, 0]
#     return arr.flatten()[idx]


# def decode_sentence_text(data, trial_idx):
#     raw = np.asarray(get_cell(data["sentenceText"], trial_idx))
#     try:
#         chars = [chr(int(c)) for c in raw.flatten() if int(c) != 0]
#         return "".join(chars).strip()
#     except Exception:
#         return str(raw)


# def extract_features(data, trial_idx, area_6v_channels):
#     spikePow_trial = np.asarray(get_cell(data["spikePow"], trial_idx), dtype=np.float32)
#     tx1_trial = np.asarray(get_cell(data["tx1"], trial_idx), dtype=np.float32)
#     spikePow_6v = spikePow_trial[:, :area_6v_channels]
#     tx1_6v = tx1_trial[:, :area_6v_channels]
#     return np.concatenate([spikePow_6v, tx1_6v], axis=1)


# def load_session(mat_path, area_6v_channels):
#     """One .mat file = one recording day. Returns list of (X [T,F], text)."""
#     data = sio.loadmat(mat_path)
#     n_trials = cell_len(data["spikePow"])
#     block_ids = np.array([int(np.squeeze(get_cell(data["blockIdx"], i))) for i in range(n_trials)])
#     feats_raw = [extract_features(data, i, area_6v_channels) for i in range(n_trials)]

#     trials = [None] * n_trials
#     for block in np.unique(block_ids):
#         idx_in_block = np.where(block_ids == block)[0]
#         all_feats = np.concatenate([feats_raw[i] for i in idx_in_block], axis=0)
#         mu = all_feats.mean(axis=0, keepdims=True).astype(np.float32)
#         sigma = (all_feats.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
#         for i in idx_in_block:
#             X = ((feats_raw[i] - mu) / sigma).astype(np.float32)
#             text = decode_sentence_text(data, i)
#             trials[i] = (X, text)
#     return trials


# def load_all_sessions(data_dir, area_6v_channels=128, max_files=None):
#     files = sorted(f for f in os.listdir(data_dir) if f.endswith(".mat"))
#     if max_files is not None:
#         files = files[:max_files]

#     samples = []  # (X, text, day_idx)
#     dropped_empty = 0
#     for day_idx, fname in enumerate(files):
#         path = os.path.join(data_dir, fname)
#         print(f"loading day {day_idx}: {fname}")
#         trials = load_session(path, area_6v_channels)
#         for X, text in trials:
#             ids = text_to_char_ids(text)
#             if len(ids) == 0:
#                 dropped_empty += 1
#                 continue
#             samples.append((X, text, day_idx))

#     print(f"total usable trials: {len(samples)} across {len(files)} days "
#           f"({dropped_empty} trials dropped: empty transcript after charset filtering)")
#     return samples, files


# # =====================================================================
# # Dataset / collate (character ids pre-tokenized once at construction)
# # =====================================================================
# class BrainToTextCharDataset(Dataset):
#     def __init__(self, samples: List[Tuple[np.ndarray, str, int]]):
#         self.items = []
#         for X, text, day_idx in samples:
#             ids = text_to_char_ids(text)
#             x = torch.tensor(X, dtype=torch.float32)
#             y = torch.tensor(ids, dtype=torch.long)
#             self.items.append((
#                 x, y,
#                 torch.tensor(x.shape[0], dtype=torch.long),
#                 torch.tensor(y.shape[0], dtype=torch.long),
#                 day_idx,
#             ))

#     def __len__(self):
#         return len(self.items)

#     def __getitem__(self, idx):
#         return self.items[idx]


# def ctc_collate(batch):
#     """Pads x with its last frame (matches the professor's convention) and
#     pads y with 0 (=blank id, ignored via target_lengths)."""
#     xs, ys, input_lengths, target_lengths, sessions = zip(*batch)
#     B = len(xs)
#     feat_dim = xs[0].shape[-1]

#     input_lengths = torch.stack(input_lengths)
#     target_lengths = torch.stack(target_lengths)
#     T_max = int(input_lengths.max())

#     x_pad = torch.zeros(B, T_max, feat_dim, dtype=torch.float32)
#     for i, x in enumerate(xs):
#         T = x.shape[0]
#         x_pad[i, :T] = x
#         if T < T_max:
#             x_pad[i, T:] = x[-1:]

#     max_target_len = int(target_lengths.max())
#     targets_padded = torch.zeros(B, max_target_len, dtype=torch.long)
#     for i, y in enumerate(ys):
#         L = y.shape[0]
#         targets_padded[i, :L] = y

#     sessions_t = torch.tensor(sessions, dtype=torch.int32)
#     return x_pad, targets_padded, input_lengths, target_lengths, sessions_t


# def get_dataset_loaders(datasetPath, batch_size, area_6v_channels, max_files, test_size, seed):
#     samples, files = load_all_sessions(datasetPath, area_6v_channels, max_files)
#     train_samples, test_samples = train_test_split(
#         samples, test_size=test_size, random_state=seed, shuffle=True
#     )
#     train_ds = BrainToTextCharDataset(train_samples)
#     test_ds = BrainToTextCharDataset(test_samples)
#     print(f"train trials: {len(train_ds)} | test trials: {len(test_ds)}")

#     train_loader = DataLoader(
#         train_ds, batch_size=batch_size, shuffle=True,
#         num_workers=4, pin_memory=True, collate_fn=ctc_collate, persistent_workers=True,
#     )
#     test_loader = DataLoader(
#         test_ds, batch_size=batch_size, shuffle=False,
#         num_workers=0, pin_memory=True, collate_fn=ctc_collate,
#     )
#     return train_loader, test_loader, test_samples, , files

# # =====================================================================
# # Model pieces -- copied from your professor's code (GaussianSmoothing,
# # Unfolder, Encoder_Decoder), only the CEBRA import path was adjusted to
# # use CEBRA_DIR like the rest of your codebase.
# # =====================================================================
# class GaussianSmoothing(nn.Module):
#     def __init__(self, channels, kernel_size, sigma, dim=2):
#         super().__init__()
#         if isinstance(kernel_size, numbers.Number):
#             kernel_size = [kernel_size] * dim
#         if isinstance(sigma, numbers.Number):
#             sigma = [sigma] * dim

#         kernel = 1
#         meshgrids = torch.meshgrid(
#             [torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing="ij"
#         )
#         for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
#             mean = (size - 1) / 2
#             kernel *= (1 / (std * math.sqrt(2 * math.pi))
#                        * torch.exp(-(((mgrid - mean) / std) ** 2) / 2))
#         kernel = kernel / torch.sum(kernel)
#         kernel = kernel.view(1, 1, *kernel.size())
#         kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

#         self.register_buffer("weight", kernel)
#         self.groups = channels
#         if dim == 1:
#             self.conv = F.conv1d
#         elif dim == 2:
#             self.conv = F.conv2d
#         elif dim == 3:
#             self.conv = F.conv3d
#         else:
#             raise RuntimeError(f"Only 1,2,3 dims supported, got {dim}")

#     def forward(self, input):
#         input = torch.permute(input, (0, 2, 1))
#         input = self.conv(input, weight=self.weight, groups=self.groups, padding="same")
#         input = torch.permute(input, (0, 2, 1))
#         return input


# class Unfolder(nn.Module):
#     def __init__(self, kernel, stride):
#         super().__init__()
#         self.unfolder = torch.nn.Unfold((kernel, 1), dilation=1, padding=0, stride=stride)
#         self.kernel = kernel
#         self.stride = stride

#     def forward(self, x, lengths):
#         x = torch.permute(
#             self.unfolder(torch.unsqueeze(torch.permute(x, (0, 2, 1)), 3)),
#             (0, 2, 1),
#         )
#         lengths = ((lengths - self.kernel) / self.stride).to(torch.int32)
#         return x, lengths


# class Encoder_Decoder(nn.Module):
#     """Input: (B,T,F) -> CTC logits (B,T,C) (permuted to (T,B,C) before CTCLoss)."""

#     def __init__(self, neural_dim, cebra_out_dim, kernel, stride, num_classes,
#                  rnn_hidden, rnn_layers, rnn_dr=0.4, rnn_bidir=True,
#                  cebra_unfolder=False, gru=False, smooth_width=2.0, gauss_in=True,
#                  no_rnn=False, cebra_window_10=False, cebra_bn=False):
#         super().__init__()

#         # def init_cebra(in_features):
#         #     if cebra_window_10:
#         #         self.left_of = 5
#         #         ceb_model = Offset10ModelBN if cebra_bn else Offset10Model
#         #     else:
#         #         self.left_of = 18
#         #         ceb_model = Offset36Dropoutv2
#         #     return ceb_model(in_features, 256, cebra_out_dim)
         
#         def init_cebra(in_features):
#             if cebra_window_10:
#                 self.left_of = 5
#                 ceb_model = Offset10Model
#             else:
#                 self.left_of = 18
#                 ceb_model = Offset36Dropoutv2
      
#             return ceb_model(in_features, 256, cebra_out_dim)
           
#         current_dim = neural_dim
#         self.cebra_unfolder = cebra_unfolder
#         self.smoother = GaussianSmoothing(neural_dim, 20, smooth_width, dim=1) if gauss_in else nn.Identity()

#         if cebra_unfolder:
#             self.cebra = init_cebra(current_dim)
#             current_dim = cebra_out_dim

#         self.unfolder = Unfolder(kernel, stride)
#         current_dim *= kernel

#         if not cebra_unfolder:
#             self.cebra = init_cebra(current_dim)
#             current_dim = cebra_out_dim

#         if not no_rnn:
#             rnn_cls = nn.GRU if gru else nn.LSTM
#             self.rnn = rnn_cls(current_dim, rnn_hidden, rnn_layers, batch_first=True,
#                                 bidirectional=rnn_bidir, dropout=rnn_dr)
#             current_dim = rnn_hidden * (2 if rnn_bidir else 1)
#         else:
#             self.rnn = lambda x: (x, None)

#         self.final_decoder = nn.Linear(current_dim, num_classes)

#     def _apply_cebra(self, x, lengths):
#         x = x.permute(0, 2, 1)
#         x = F.pad(x, (self.left_of, self.left_of - 1), mode="replicate")
#         x = self.cebra(x).permute(0, 2, 1)
#         self.embeddings = x
#         self.emb_lengths = lengths
#         return x

#     def get_cebra_embs(self):
#         return self.embeddings, self.emb_lengths

#     def forward(self, x, lengths):
#         x = self.smoother(x)
#         if self.cebra_unfolder:
#             x = self._apply_cebra(x, lengths)
#         x, lengths = self.unfolder(x, lengths)
#         if not self.cebra_unfolder:
#             x = self._apply_cebra(x, lengths)
#         x, _ = self.rnn(x)
#         x = self.final_decoder(x)
#         return x, lengths

# class CebraFromRawInput(nn.Module):
#     """Wraps just smoother -> unfolder -> cebra, so attribution sees CEBRA's
#     own output as a function of the RAW neural input, not the CTC logits."""
#     def __init__(self, encoder_decoder):
#         super().__init__()
#         self.ed = encoder_decoder

#     def forward(self, x):
#         ed = self.ed
#         lengths = torch.tensor([x.shape[1]] * x.shape[0], device=x.device)
#         h = ed.smoother(x)
#         if ed.cebra_unfolder:
#             h = ed._apply_cebra(h, lengths)
#             h, lengths = ed.unfolder(h, lengths)
#         else:
#             h, lengths = ed.unfolder(h, lengths)
#             h = ed._apply_cebra(h, lengths)
#         return h

# def reduce_attr_map(arr):
#     if torch.is_tensor(arr):
#         arr = arr.detach().cpu().numpy()
#     arr = np.abs(np.asarray(arr))
#     if arr.ndim == 3:
#         arr = arr.mean(axis=0)
#     elif arr.ndim == 1:
#         arr = arr[None, :]
#     return arr.astype(np.float32)


# def save_heatmap(arr, path, title, feature_boundary=None):
#     plt.figure(figsize=(10, 6))
#     plt.imshow(arr, aspect="auto", cmap="viridis")
#     plt.colorbar(label="absolute attribution")
#     if feature_boundary is not None:
#         plt.axvline(feature_boundary, color="white", linestyle="--", linewidth=1)
#     plt.xlabel("Neural feature / channel (spikePow 6v | tx1 6v)")
#     plt.ylabel("Latent dimension")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print("saved:", path)

# def run_attribution(model, raw_X, area_6v_channels, ceb_out_dim, out_dir, device, tag):
#     m = model.module if isinstance(model, torch.nn.DataParallel) else model
#     wrapper = CebraFromRawInput(m).to(device)
#     wrapper.eval()

#     x_tensor = torch.tensor(raw_X, dtype=torch.float32, device=device).unsqueeze(0)
#     x_tensor.requires_grad_(True)

#     method = cebra.attribution.init(
#         name="jacobian-based-batched",
#         model=wrapper,
#         input_data=x_tensor,
#         output_dimension=ceb_out_dim,
#     )
    
#     result = method.compute_attribution_map(batch_size=x_tensor.shape[0])
    
#     jf = result["jf"]
#     jf_inv = result.get("jf-inv-svd", result.get("jf-inv-lsq", result.get("jf-inv")))

#     jf_matrix = reduce_attr_map(jf)
#     jf_inv_matrix = reduce_attr_map(jf_inv)

#     torch.save(jf, os.path.join(out_dir, f"{tag}_jf.pt"))
#     torch.save(jf_inv, os.path.join(out_dir, f"{tag}_jf_inv.pt"))
#     save_heatmap(jf_matrix, os.path.join(out_dir, f"{tag}_jf.png"),
#                  f"{tag} - Jacobian", feature_boundary=area_6v_channels)
#     save_heatmap(jf_inv_matrix, os.path.join(out_dir, f"{tag}_jf_inv.png"),
#                  f"{tag} - Inverse Jacobian", feature_boundary=area_6v_channels)

#     cleanup(wrapper, x_tensor, method, result)


   
# # =====================================================================
# # InfoNCE + positive/negative sampler -- copied from professor's code
# # =====================================================================
# @torch.jit.script
# def dot_similarity(ref: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
#     pos_dist = torch.einsum("ni,ni->n", ref, pos)
#     neg_dist = torch.einsum("ni,mi->nm", ref, neg)
#     return pos_dist, neg_dist


# @torch.jit.script
# def infonce(pos_dist: torch.Tensor, neg_dist: torch.Tensor):
#     with torch.no_grad():
#         c, _ = neg_dist.max(dim=1, keepdim=True)
#     c = c.detach()
#     pos_dist = pos_dist - c.squeeze(1)
#     neg_dist = neg_dist - c
#     align = (-pos_dist).mean()
#     uniform = torch.logsumexp(neg_dist, dim=1).mean()
#     c_mean = c.mean()
#     return align + uniform, align - c_mean, uniform + c_mean


# class InfoNCE(nn.Module):
#     def __init__(self, temp) -> None:
#         super().__init__()
#         self.temperature = temp

#     def _distance(self, ref, pos, neg):
#         pos_dist, neg_dist = dot_similarity(ref, pos, neg)
#         return pos_dist / self.temperature, neg_dist / self.temperature

#     def forward(self, ref, pos, neg):
#         pos_dist, neg_dist = self._distance(ref, pos, neg)
#         return infonce(pos_dist, neg_dist)


# def get_batch(x, x_len, batch_size, offset, single_sequence=False,
#               random_offset=False, random_dir=False, all_ref=False):
#     B, T, F_ = x.shape
#     device = x.device

#     if all_ref:
#         time_range = torch.arange(T, device=device).unsqueeze(0)
#         valid_mask = time_range < x_len.unsqueeze(1)
#         ref_batch_idx, ref_time_idx = torch.where(valid_mask)
#         if single_sequence:
#             chosen_batch = torch.randint(0, B, (1,)).item()
#             mask = ref_batch_idx == chosen_batch
#             ref_batch_idx, ref_time_idx = ref_batch_idx[mask], ref_time_idx[mask]
#     else:
#         if not single_sequence:
#             ref_batch_idx = torch.randint(0, B, (batch_size,), device=device)
#             max_times = torch.clamp(x_len[ref_batch_idx] - offset - 1, min=1)
#             ref_time_idx = torch.randint(0 if not random_dir else offset,
#                                           torch.max(max_times).item(), (batch_size,), device=device)
#             ref_time_idx = torch.min(ref_time_idx, max_times - 1)
#         else:
#             pos_batch = torch.randint(0, B, (1,), device=device).item()
#             ref_batch_idx = torch.full((batch_size,), pos_batch, device=device)
#             max_times = torch.clamp(x_len[pos_batch] - offset - 1, min=1)
#             ref_time_idx = torch.randint(0 if not random_dir else offset,
#                                           max_times.item(), (batch_size,), device=device)

#     if random_offset:
#         add_offset = torch.randint(1, offset + 1, (len(ref_batch_idx),), device=device, dtype=torch.long)
#     else:
#         add_offset = torch.full((len(ref_batch_idx),), offset, device=device, dtype=torch.long)
#     if random_dir:
#         dir_val = torch.randint(0, 2, add_offset.shape, device=device) * 2 - 1
#         add_offset = add_offset * dir_val

#     pos_time_idx = ref_time_idx + add_offset
#     max_valid = x_len[ref_batch_idx] - 1
#     min_val = torch.zeros_like(max_valid)
#     pos_time_idx = torch.clamp(pos_time_idx, min=min_val, max=max_valid)

#     if single_sequence:
#         neg_batch_idx = torch.randint(0, B, (len(ref_batch_idx),), device=device)
#         if B > 1:
#             mask = neg_batch_idx == ref_batch_idx[0]
#             neg_batch_idx[mask] = (neg_batch_idx[mask] + 1) % B
#     else:
#         neg_batch_idx = torch.randint(0, B, (len(ref_batch_idx),), device=device)

#     neg_max_times = x_len[neg_batch_idx]
#     neg_time_idx = torch.randint(0, torch.max(neg_max_times).item(), (len(ref_batch_idx),), device=device)
#     neg_time_idx = torch.min(neg_time_idx, neg_max_times - 1)

#     reference = x[ref_batch_idx, ref_time_idx]
#     positive = x[ref_batch_idx, pos_time_idx]
#     negative = x[neg_batch_idx, neg_time_idx]
#     return (reference, positive, negative, ref_batch_idx, ref_time_idx,
#             pos_time_idx, neg_batch_idx, neg_time_idx)


# # =====================================================================
# # Training loop -- adapted from professor's train_model(args)
# # =====================================================================
# def train_model(args: dict):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     os.makedirs(args["out_dir"], exist_ok=True)
#     torch.manual_seed(args["seed"])
#     np.random.seed(args["seed"])

#     checkpoint_address = os.path.join(args["out_dir"], "checkpoint.pt")

#     neural_dim = 2 * args["area_6v_channels"]
#     num_classes = charset.num_classes  # <- explicit, character-level (NOT the 41/32 hardcode)
#     print(f"neural_dim={neural_dim} | num_classes={num_classes} (charset, incl. blank)")

#     model = Encoder_Decoder(
#         neural_dim, args["ceb_out"], args["kernel"], args["stride"], num_classes,
#         args["hidden"], args["layers"], args["dropout"], args["bidir"],
#         args["cebra_unfolder"], args["gru"], smooth_width=2.0,
#         gauss_in=args["gauss_in"], no_rnn=args["no_rnn"],
#         cebra_bn=args["ceb_bn"], cebra_window_10=args["cebra_window_10"],
#     ).to(device)
#     if torch.cuda.device_count() > 1:
#         print(f"Using {torch.cuda.device_count()} GPUs")
#         model = torch.nn.DataParallel(model)

#     with open(os.path.join(args["out_dir"], "args"), "wb") as f:
#         pickle.dump(args, f)

#     # train_loader, test_loader, test_samples = get_dataset_loaders(
#     #     args["datasetPath"], args["batchSize"], args["area_6v_channels"],
#     #     args["max_files"], args["test_size"], args["seed"],
#     # )
#     train_loader, test_loader, test_samples, files = get_dataset_loaders(
#         args["datasetPath"], args["batchSize"], args["area_6v_channels"],
#         args["max_files"], args["test_size"], args["seed"],
#     )

#     criterion = InfoNCE(args["temperature"])
#     ctc_criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args["lrStart"],
#                                   betas=(0.9, 0.999), eps=0.1, weight_decay=args["l2_decay"])
#     scheduler = torch.optim.lr_scheduler.LinearLR(
#         optimizer, start_factor=1.0, end_factor=args["lrEnd"] / args["lrStart"],
#         total_iters=args["nBatch"],
#     )

#     so_far_batch = load_checkpoint(checkpoint_address, model, optimizer, scheduler)
#     print("resuming from batch:", so_far_batch)

#     inf_losses = 0
#     testLoss, testCER = [], []
#     train_iter = iter(train_loader)

#     for batch in trange(args["nBatch"]):
#         model.train()
#         try:
#             X, y, X_len, y_len, dayIdx = next(train_iter)
#         except StopIteration:
#             train_iter = iter(train_loader)
#             X, y, X_len, y_len, dayIdx = next(train_iter)

#         X, y, X_len, y_len, dayIdx = (X.to(device), y.to(device), X_len.to(device),
#                                        y_len.to(device), dayIdx.to(device))
#         if batch < so_far_batch:
#             continue

#         if args["whiteNoiseSD"] > 0:
#             X = X + torch.randn(X.shape, device=device) * args["whiteNoiseSD"]
#         if args["constantOffsetSD"] > 0:
#             X = X + torch.randn([X.shape[0], 1, X.shape[2]], device=device) * args["constantOffsetSD"]

#         with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
#             pred, lengths = model(X, X_len)
#             m = model.module if isinstance(model, torch.nn.DataParallel) else model
#             embeddings, emb_lengths = m.get_cebra_embs()

#             ctc_loss = torch.sum(ctc_criterion(
#                 torch.permute(pred.log_softmax(2), [1, 0, 2]), y, lengths, y_len))

#             (reference, positive, negative, ref_b, ref_t, pos_t, neg_b, neg_t) = get_batch(
#                 embeddings, emb_lengths, args["cont_batch"], args["offset"],
#                 args["sample_single"], args["random_offset"], args["random_dir"], args["all_ref"],
#             )
#             loss_contrastive = criterion(reference, positive, negative)[0]
#             loss = args["lambda_contrastive"] * loss_contrastive + ctc_loss

#         optimizer.zero_grad()
#         if not torch.isfinite(loss):
#             inf_losses += 1
#             if inf_losses > 10:
#                 break
#             continue
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#         optimizer.step()

#         # ---------------- optional adversarial (PGD on raw input X) ----------------
#         if args["adv"]:
#             epsilon, steps, alpha = args["adv_eps"], args["adv_steps"], args["adv_eps"] / 5.0
#             X_adv = X.detach().clone()
#             if args["adv_norm"] == "linf":
#                 X_adv = X_adv + torch.empty_like(X_adv).uniform_(-epsilon, epsilon)
#             else:  # l2
#                 noise = torch.randn_like(X_adv)
#                 noise_norm = noise.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
#                 noise = noise / noise_norm
#                 noise = noise * (torch.rand((noise.shape[0], noise.shape[1], 1), device=device) * epsilon)
#                 X_adv = X_adv + noise

#             for _ in range(steps):
#                 X_adv = X_adv.detach().requires_grad_(True)
#                 with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
#                     pred_adv, lengths_adv = model(X_adv, X_len)
#                     m = model.module if isinstance(model, torch.nn.DataParallel) else model
#                     emb_adv, emb_len_adv = m.get_cebra_embs()
#                     ctc_loss_adv = torch.sum(ctc_criterion(
#                         torch.permute(pred_adv.log_softmax(2), [1, 0, 2]), y, lengths_adv, y_len))
#                     ref = emb_adv[ref_b, ref_t]
#                     pos = emb_adv[ref_b, pos_t].detach()
#                     neg = emb_adv[neg_b, neg_t].detach()
#                     loss_adv = args["lambda_contrastive"] * criterion(ref, pos, neg)[0] + ctc_loss_adv

#                 grad = torch.autograd.grad(loss_adv, X_adv, only_inputs=True)[0]
#                 with torch.no_grad():
#                     if args["adv_norm"] == "linf":
#                         X_adv = X_adv + alpha * grad.sign()
#                         delta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
#                         X_adv = X + delta
#                     else:
#                         grad_norm = grad.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
#                         X_adv = (X_adv + alpha * (grad / grad_norm)).detach()
#                         delta = X_adv - X
#                         delta_norm = delta.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
#                         scale = torch.clamp(epsilon / delta_norm, max=1.0)
#                         X_adv = (X + delta * scale).detach()

#             X_adv = X_adv.detach()
#             with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
#                 pred_adv, lengths_adv = model(X_adv, X_len)
#                 m = model.module if isinstance(model, torch.nn.DataParallel) else model
#                 emb_adv, emb_len_adv = m.get_cebra_embs()
#                 ctc_loss_adv = torch.sum(ctc_criterion(
#                     torch.permute(pred_adv.log_softmax(2), [1, 0, 2]), y, lengths_adv, y_len))
#                 ref = emb_adv[ref_b, ref_t]
#                 pos = emb_adv[ref_b, pos_t]
#                 neg = emb_adv[neg_b, neg_t]
#                 loss_adv = args["lambda_contrastive"] * criterion(ref, pos, neg)[0] + ctc_loss_adv

#             optimizer.zero_grad()
#             if torch.isfinite(loss_adv):
#                 loss_adv.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#                 optimizer.step()

#         scheduler.step()

#         # ---------------- periodic validation (CER) ----------------
#         if batch % args["eval_every"] == 0:
#             model.eval()
#             with torch.no_grad():
#                 allLoss, total_edit_distance, total_seq_length = [], 0, 0
#                 for Xv, yv, Xv_len, yv_len, _ in test_loader:
#                     Xv, yv, Xv_len, yv_len = Xv.to(device), yv.to(device), Xv_len.to(device), yv_len.to(device)
#                     with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
#                         pred_v, lengths_v = model(Xv, Xv_len)
#                         loss_v = torch.sum(ctc_criterion(
#                             torch.permute(pred_v.log_softmax(2), [1, 0, 2]), yv, lengths_v, yv_len))
#                     allLoss.append(loss_v.cpu().item())

#                     for i in range(pred_v.shape[0]):
#                         decoded = torch.argmax(pred_v[i, :lengths_v[i], :], dim=-1)
#                         decoded = torch.unique_consecutive(decoded)
#                         decoded = np.array([c for c in decoded.cpu().numpy() if c != 0])
#                         true_seq = yv[i][:yv_len[i]].cpu().numpy()
#                         matcher = SequenceMatcher(a=true_seq.tolist(), b=decoded.tolist())
#                         total_edit_distance += matcher.distance()
#                         total_seq_length += len(true_seq)

#                 avgLoss = float(np.sum(allLoss) / max(len(test_loader), 1))
#                 cer = total_edit_distance / max(total_seq_length, 1)
#                 print(f"batch {batch} | val ctc loss: {avgLoss:.4f} | CER: {cer:.4f} "
#                       f"| train ctc: {ctc_loss.item():.4f} | train cont: {loss_contrastive.item():.4f}")

#             state_dict = (model.module if isinstance(model, torch.nn.DataParallel) else model).state_dict()
#             torch.save(state_dict, os.path.join(args["out_dir"], "modelWeights"))
#             save_checkpoint(checkpoint_address, model, optimizer, scheduler, batch)

#             testLoss.append(avgLoss)
#             testCER.append(cer)
#             with open(os.path.join(args["out_dir"], "trainingStats"), "wb") as f:
#                 pickle.dump({"testLoss": np.array(testLoss), "testCER": np.array(testCER)}, f)

#     # print("DONE")
#     # raw_X, _, _ = test_samples[0]
#     # run_attribution(model, raw_X, args["area_6v_channels"], args["ceb_out"],
#     #                  args["out_dir"], device, tag="CEBRA_trial0")
#     # return model
#     print("DONE")

#     day_to_trial = {}
#     for X, text, day_idx in test_samples:
#         if day_idx not in day_to_trial:
#             day_to_trial[day_idx] = X

#     print(f"\nrunning attribution for {len(day_to_trial)} day(s) present in the test split "
#           f"(out of {len(files)} total day files)")

#     for day_idx in sorted(day_to_trial.keys()):
#         day_name = os.path.splitext(files[day_idx])[0]
#         tag = f"CEBRA_day{day_idx}_{day_name}"
#         run_attribution(
#             model, day_to_trial[day_idx], args["area_6v_channels"], args["ceb_out"],
#             args["out_dir"], device, tag=tag,
#         )

#     return model


# if __name__ == "__main__":
#     train_model(DEFAULT_ARGS)

   
# # """
# # Character-level CTC training on the Nature (2023) "high-performance speech
# # neuroprosthesis" competitionData format (spikePow + tx1, area 6v, blockIdx,
# # sentenceText), following your professor's Encoder_Decoder / InfoNCE /
# # adversarial-training pattern.

# # Two things this script does differently from what you had, and why:

# # 1) The single-trial CEBRA/ACORN attribution script is a *different, unrelated*
# #    pipeline (self-supervised, single-trial, no text, no CTC) -- it was never
# #    meant to "be" the decoder. This file is the actual decoder.

# # 2) The professor's own `train_model(args)` hardcodes
# #        num_classes = 41 if is_speech else 32
# #    so the speech branch always assumes PHONEME targets (41 classes), and
# #    `Charset` (32 classes incl. blank) is only wired to the handwriting/NLP
# #    branch. To get CHARACTER-level CTC on the speech data, `num_classes` below
# #    is taken explicitly from `charset.num_classes`, and a brand-new data
# #    loader (`load_all_sessions` / `BrainToTextCharDataset`) reads the raw
# #    competitionData .mat files directly (the professor's loaders all expect
# #    an already-pickled, pre-tokenized dataset we don't have).

# # ASSUMPTION (please confirm with your professor): `CHARS` has no literal space
# # character, so '>' is treated as the word-boundary/space token below
# # (`text_to_char_ids`). If '>' means something else in their scheme, change
# # that one function.
# # """

# # import os
# # import sys
# # import math
# # import time
# # import numbers
# # import pickle
# # from typing import List, Tuple

# # import numpy as np
# # import scipy.io as sio
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.utils.data import Dataset, DataLoader
# # from sklearn.model_selection import train_test_split
# # from tqdm import trange
# # from edit_distance import SequenceMatcher

# # from utils.constants import CEBRA_DIR
# # from utils.load_model_states import save_checkpoint, load_checkpoint

# # sys.path.insert(0, str(CEBRA_DIR))
# # # from cebra.models import (
# # #     Offset36Dropoutv2, Offset10Model, Offset36Dropoutv2BN,
# # #     Offset10ModelBN, Offset36Dropoutv205,
# # # )
# # from cebra.models import (
# #     Offset36Dropoutv2,
# #     Offset10Model,
# # )


# # # =====================================================================
# # # Charset -- verbatim from your professor's code (index 0 = CTC blank)
# # # =====================================================================
# # CHARS = [
# #     '>', ',', '?', '~', "'",
# #     'a', 'b', 'c', 'd', 'e', 'f', 'g',
# #     'h', 'i', 'j', 'k', 'l', 'm', 'n',
# #     'o', 'p', 'q', 'r', 's', 't',
# #     'u', 'v', 'w', 'x', 'y', 'z',
# # ]
# # BLANK_TOKEN = "<BLANK>"


# # class Charset:
# #     def __init__(self, symbols: List[str]):
# #         self.idx2sym = [BLANK_TOKEN] + symbols
# #         self.sym2idx = {s: i + 1 for i, s in enumerate(symbols)}
# #         self.sym2idx[BLANK_TOKEN] = 0

# #     @property
# #     def num_classes(self) -> int:
# #         return len(self.idx2sym)

# #     def text_to_int(self, text: str) -> List[int]:
# #         return [self.sym2idx[ch] for ch in text if ch in self.sym2idx]

# #     def int_to_text(self, ids: List[int]) -> str:
# #         return "".join(self.idx2sym[i] for i in ids if i != 0)


# # charset = Charset(CHARS)


# # def text_to_char_ids(text: str) -> List[int]:
# #     """lower-case, map space -> '>' (see ASSUMPTION at top of file), drop
# #     anything not in the charset."""
# #     text = text.lower().replace(" ", ">")
# #     return charset.text_to_int(text)


# # # =====================================================================
# # # Config
# # # =====================================================================
# # DEFAULT_ARGS = dict(
# #     datasetPath="./data/competitionData/competitionData/train",
# #     out_dir="./outputs/ctc_char_run",
# #     seed=42,

# #     area_6v_channels=128,   # neural_dim = 2 * this (spikePow_6v + tx1_6v)
# #     max_files=None,         # cap number of session-day .mat files, for a quick test run
# #     test_size=0.15,

# #     # Encoder_Decoder / CEBRA
# #     ceb_out=32,
# #     kernel=8,
# #     stride=4,
# #     hidden=256,
# #     layers=2,
# #     dropout=0.4,
# #     bidir=True,
# #     cebra_unfolder=False,
# #     gru=True,
# #     gauss_in=True,
# #     no_rnn=False,
# #     ceb_bn=False,
# #     cebra_window_10=True,   # True -> Offset10Model (window 10); False -> Offset36Dropoutv2

# #     # optimization
# #     batchSize=16,
# #     lrStart=3e-4,
# #     lrEnd=3e-5,
# #     nBatch=50000,#epoch
# #     l2_decay=1e-5,
# #     temperature=0.1,
# #     whiteNoiseSD=0.0,
# #     constantOffsetSD=0.0,

# #     # InfoNCE positive/negative sampling (see get_batch)
# #     cont_batch=512,
# #     offset=10,
# #     sample_single=False,
# #     random_dir=False,
# #     random_offset=False,
# #     all_ref=False,
# #     lambda_contrastive=1.0,   # weight on the CEBRA contrastive term, professor's code uses 1.0

# #     # adversarial training (optional, matches professor's PGD-on-input scheme)
# #     adv=True,
# #     adv_eps=5,
# #     adv_norm="linf",
# #     adv_steps=10,

# #     eval_every=150,
# # )


# # def cleanup(*objs):
# #     for o in objs:
# #         del o
# #     import gc
# #     gc.collect()
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()
# #         torch.cuda.ipc_collect()


# # # =====================================================================
# # # Raw .mat loading (competitionData format) -- extends the single-trial
# # # attribution script's feature extraction to *all* trials in a session,
# # # with per-block z-scoring, exactly like that script did.
# # # =====================================================================
# # def cell_len(mat_cell):
# #     arr = np.asarray(mat_cell)
# #     return int(max(arr.shape))


# # def get_cell(mat_cell, idx):
# #     arr = np.asarray(mat_cell)
# #     if arr.ndim == 2:
# #         if arr.shape[0] == 1:
# #             return arr[0, idx]
# #         elif arr.shape[1] == 1:
# #             return arr[idx, 0]
# #     return arr.flatten()[idx]


# # def decode_sentence_text(data, trial_idx):
# #     raw = np.asarray(get_cell(data["sentenceText"], trial_idx))
# #     try:
# #         chars = [chr(int(c)) for c in raw.flatten() if int(c) != 0]
# #         return "".join(chars).strip()
# #     except Exception:
# #         return str(raw)


# # def extract_features(data, trial_idx, area_6v_channels):
# #     spikePow_trial = np.asarray(get_cell(data["spikePow"], trial_idx), dtype=np.float32)
# #     tx1_trial = np.asarray(get_cell(data["tx1"], trial_idx), dtype=np.float32)
# #     spikePow_6v = spikePow_trial[:, :area_6v_channels]
# #     tx1_6v = tx1_trial[:, :area_6v_channels]
# #     return np.concatenate([spikePow_6v, tx1_6v], axis=1)


# # def load_session(mat_path, area_6v_channels):
# #     """One .mat file = one recording day. Returns list of (X [T,F], text)."""
# #     data = sio.loadmat(mat_path)
# #     n_trials = cell_len(data["spikePow"])
# #     block_ids = np.array([int(np.squeeze(get_cell(data["blockIdx"], i))) for i in range(n_trials)])
# #     feats_raw = [extract_features(data, i, area_6v_channels) for i in range(n_trials)]

# #     trials = [None] * n_trials
# #     for block in np.unique(block_ids):
# #         idx_in_block = np.where(block_ids == block)[0]
# #         all_feats = np.concatenate([feats_raw[i] for i in idx_in_block], axis=0)
# #         mu = all_feats.mean(axis=0, keepdims=True).astype(np.float32)
# #         sigma = (all_feats.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
# #         for i in idx_in_block:
# #             X = ((feats_raw[i] - mu) / sigma).astype(np.float32)
# #             text = decode_sentence_text(data, i)
# #             trials[i] = (X, text)
# #     return trials


# # def load_all_sessions(data_dir, area_6v_channels=128, max_files=None):
# #     files = sorted(f for f in os.listdir(data_dir) if f.endswith(".mat"))
# #     if max_files is not None:
# #         files = files[:max_files]

# #     samples = []  # (X, text, day_idx)
# #     dropped_empty = 0
# #     for day_idx, fname in enumerate(files):
# #         path = os.path.join(data_dir, fname)
# #         print(f"loading day {day_idx}: {fname}")
# #         trials = load_session(path, area_6v_channels)
# #         for X, text in trials:
# #             ids = text_to_char_ids(text)
# #             if len(ids) == 0:
# #                 dropped_empty += 1
# #                 continue
# #             samples.append((X, text, day_idx))

# #     print(f"total usable trials: {len(samples)} across {len(files)} days "
# #           f"({dropped_empty} trials dropped: empty transcript after charset filtering)")
# #     return samples, files


# # # =====================================================================
# # # Dataset / collate (character ids pre-tokenized once at construction)
# # # =====================================================================
# # class BrainToTextCharDataset(Dataset):
# #     def __init__(self, samples: List[Tuple[np.ndarray, str, int]]):
# #         self.items = []
# #         for X, text, day_idx in samples:
# #             ids = text_to_char_ids(text)
# #             x = torch.tensor(X, dtype=torch.float32)
# #             y = torch.tensor(ids, dtype=torch.long)
# #             self.items.append((
# #                 x, y,
# #                 torch.tensor(x.shape[0], dtype=torch.long),
# #                 torch.tensor(y.shape[0], dtype=torch.long),
# #                 day_idx,
# #             ))

# #     def __len__(self):
# #         return len(self.items)

# #     def __getitem__(self, idx):
# #         return self.items[idx]


# # def ctc_collate(batch):
# #     """Pads x with its last frame (matches the professor's convention) and
# #     pads y with 0 (=blank id, ignored via target_lengths)."""
# #     xs, ys, input_lengths, target_lengths, sessions = zip(*batch)
# #     B = len(xs)
# #     feat_dim = xs[0].shape[-1]

# #     input_lengths = torch.stack(input_lengths)
# #     target_lengths = torch.stack(target_lengths)
# #     T_max = int(input_lengths.max())

# #     x_pad = torch.zeros(B, T_max, feat_dim, dtype=torch.float32)
# #     for i, x in enumerate(xs):
# #         T = x.shape[0]
# #         x_pad[i, :T] = x
# #         if T < T_max:
# #             x_pad[i, T:] = x[-1:]

# #     max_target_len = int(target_lengths.max())
# #     targets_padded = torch.zeros(B, max_target_len, dtype=torch.long)
# #     for i, y in enumerate(ys):
# #         L = y.shape[0]
# #         targets_padded[i, :L] = y

# #     sessions_t = torch.tensor(sessions, dtype=torch.int32)
# #     return x_pad, targets_padded, input_lengths, target_lengths, sessions_t


# # def get_dataset_loaders(datasetPath, batch_size, area_6v_channels, max_files, test_size, seed):
# #     samples, files = load_all_sessions(datasetPath, area_6v_channels, max_files)
# #     train_samples, test_samples = train_test_split(
# #         samples, test_size=test_size, random_state=seed, shuffle=True
# #     )
# #     train_ds = BrainToTextCharDataset(train_samples)
# #     test_ds = BrainToTextCharDataset(test_samples)
# #     print(f"train trials: {len(train_ds)} | test trials: {len(test_ds)}")

# #     train_loader = DataLoader(
# #         train_ds, batch_size=batch_size, shuffle=True,
# #         num_workers=4, pin_memory=True, collate_fn=ctc_collate, persistent_workers=True,
# #     )
# #     test_loader = DataLoader(
# #         test_ds, batch_size=batch_size, shuffle=False,
# #         num_workers=0, pin_memory=True, collate_fn=ctc_collate,
# #     )
# #     return train_loader, test_loader


# # # =====================================================================
# # # Model pieces -- copied from your professor's code (GaussianSmoothing,
# # # Unfolder, Encoder_Decoder), only the CEBRA import path was adjusted to
# # # use CEBRA_DIR like the rest of your codebase.
# # # =====================================================================
# # class GaussianSmoothing(nn.Module):
# #     def __init__(self, channels, kernel_size, sigma, dim=2):
# #         super().__init__()
# #         if isinstance(kernel_size, numbers.Number):
# #             kernel_size = [kernel_size] * dim
# #         if isinstance(sigma, numbers.Number):
# #             sigma = [sigma] * dim

# #         kernel = 1
# #         meshgrids = torch.meshgrid(
# #             [torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing="ij"
# #         )
# #         for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
# #             mean = (size - 1) / 2
# #             kernel *= (1 / (std * math.sqrt(2 * math.pi))
# #                        * torch.exp(-(((mgrid - mean) / std) ** 2) / 2))
# #         kernel = kernel / torch.sum(kernel)
# #         kernel = kernel.view(1, 1, *kernel.size())
# #         kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

# #         self.register_buffer("weight", kernel)
# #         self.groups = channels
# #         if dim == 1:
# #             self.conv = F.conv1d
# #         elif dim == 2:
# #             self.conv = F.conv2d
# #         elif dim == 3:
# #             self.conv = F.conv3d
# #         else:
# #             raise RuntimeError(f"Only 1,2,3 dims supported, got {dim}")

# #     def forward(self, input):
# #         input = torch.permute(input, (0, 2, 1))
# #         input = self.conv(input, weight=self.weight, groups=self.groups, padding="same")
# #         input = torch.permute(input, (0, 2, 1))
# #         return input


# # class Unfolder(nn.Module):
# #     def __init__(self, kernel, stride):
# #         super().__init__()
# #         self.unfolder = torch.nn.Unfold((kernel, 1), dilation=1, padding=0, stride=stride)
# #         self.kernel = kernel
# #         self.stride = stride

# #     def forward(self, x, lengths):
# #         x = torch.permute(
# #             self.unfolder(torch.unsqueeze(torch.permute(x, (0, 2, 1)), 3)),
# #             (0, 2, 1),
# #         )
# #         lengths = ((lengths - self.kernel) / self.stride).to(torch.int32)
# #         return x, lengths


# # class Encoder_Decoder(nn.Module):
# #     """Input: (B,T,F) -> CTC logits (B,T,C) (permuted to (T,B,C) before CTCLoss)."""

# #     def __init__(self, neural_dim, cebra_out_dim, kernel, stride, num_classes,
# #                  rnn_hidden, rnn_layers, rnn_dr=0.4, rnn_bidir=True,
# #                  cebra_unfolder=False, gru=False, smooth_width=2.0, gauss_in=True,
# #                  no_rnn=False, cebra_window_10=False, cebra_bn=False):
# #         super().__init__()

# #         # def init_cebra(in_features):
# #         #     if cebra_window_10:
# #         #         self.left_of = 5
# #         #         ceb_model = Offset10ModelBN if cebra_bn else Offset10Model
# #         #     else:
# #         #         self.left_of = 18
# #         #         ceb_model = Offset36Dropoutv2
# #         #     return ceb_model(in_features, 256, cebra_out_dim)
         
# #         def init_cebra(in_features):
# #             if cebra_window_10:
# #                 self.left_of = 5
# #                 ceb_model = Offset10Model
# #             else:
# #                 self.left_of = 18
# #                 ceb_model = Offset36Dropoutv2
      
# #             return ceb_model(in_features, 256, cebra_out_dim)
           
# #         current_dim = neural_dim
# #         self.cebra_unfolder = cebra_unfolder
# #         self.smoother = GaussianSmoothing(neural_dim, 20, smooth_width, dim=1) if gauss_in else nn.Identity()

# #         if cebra_unfolder:
# #             self.cebra = init_cebra(current_dim)
# #             current_dim = cebra_out_dim

# #         self.unfolder = Unfolder(kernel, stride)
# #         current_dim *= kernel

# #         if not cebra_unfolder:
# #             self.cebra = init_cebra(current_dim)
# #             current_dim = cebra_out_dim

# #         if not no_rnn:
# #             rnn_cls = nn.GRU if gru else nn.LSTM
# #             self.rnn = rnn_cls(current_dim, rnn_hidden, rnn_layers, batch_first=True,
# #                                 bidirectional=rnn_bidir, dropout=rnn_dr)
# #             current_dim = rnn_hidden * (2 if rnn_bidir else 1)
# #         else:
# #             self.rnn = lambda x: (x, None)

# #         self.final_decoder = nn.Linear(current_dim, num_classes)

# #     def _apply_cebra(self, x, lengths):
# #         x = x.permute(0, 2, 1)
# #         x = F.pad(x, (self.left_of, self.left_of - 1), mode="replicate")
# #         x = self.cebra(x).permute(0, 2, 1)
# #         self.embeddings = x
# #         self.emb_lengths = lengths
# #         return x

# #     def get_cebra_embs(self):
# #         return self.embeddings, self.emb_lengths

# #     def forward(self, x, lengths):
# #         x = self.smoother(x)
# #         if self.cebra_unfolder:
# #             x = self._apply_cebra(x, lengths)
# #         x, lengths = self.unfolder(x, lengths)
# #         if not self.cebra_unfolder:
# #             x = self._apply_cebra(x, lengths)
# #         x, _ = self.rnn(x)
# #         x = self.final_decoder(x)
# #         return x, lengths


# # # =====================================================================
# # # InfoNCE + positive/negative sampler -- copied from professor's code
# # # =====================================================================
# # @torch.jit.script
# # def dot_similarity(ref: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
# #     pos_dist = torch.einsum("ni,ni->n", ref, pos)
# #     neg_dist = torch.einsum("ni,mi->nm", ref, neg)
# #     return pos_dist, neg_dist


# # @torch.jit.script
# # def infonce(pos_dist: torch.Tensor, neg_dist: torch.Tensor):
# #     with torch.no_grad():
# #         c, _ = neg_dist.max(dim=1, keepdim=True)
# #     c = c.detach()
# #     pos_dist = pos_dist - c.squeeze(1)
# #     neg_dist = neg_dist - c
# #     align = (-pos_dist).mean()
# #     uniform = torch.logsumexp(neg_dist, dim=1).mean()
# #     c_mean = c.mean()
# #     return align + uniform, align - c_mean, uniform + c_mean


# # class InfoNCE(nn.Module):
# #     def __init__(self, temp) -> None:
# #         super().__init__()
# #         self.temperature = temp

# #     def _distance(self, ref, pos, neg):
# #         pos_dist, neg_dist = dot_similarity(ref, pos, neg)
# #         return pos_dist / self.temperature, neg_dist / self.temperature

# #     def forward(self, ref, pos, neg):
# #         pos_dist, neg_dist = self._distance(ref, pos, neg)
# #         return infonce(pos_dist, neg_dist)


# # def get_batch(x, x_len, batch_size, offset, single_sequence=False,
# #               random_offset=False, random_dir=False, all_ref=False):
# #     B, T, F_ = x.shape
# #     device = x.device

# #     if all_ref:
# #         time_range = torch.arange(T, device=device).unsqueeze(0)
# #         valid_mask = time_range < x_len.unsqueeze(1)
# #         ref_batch_idx, ref_time_idx = torch.where(valid_mask)
# #         if single_sequence:
# #             chosen_batch = torch.randint(0, B, (1,)).item()
# #             mask = ref_batch_idx == chosen_batch
# #             ref_batch_idx, ref_time_idx = ref_batch_idx[mask], ref_time_idx[mask]
# #     else:
# #         if not single_sequence:
# #             ref_batch_idx = torch.randint(0, B, (batch_size,), device=device)
# #             max_times = torch.clamp(x_len[ref_batch_idx] - offset - 1, min=1)
# #             ref_time_idx = torch.randint(0 if not random_dir else offset,
# #                                           torch.max(max_times).item(), (batch_size,), device=device)
# #             ref_time_idx = torch.min(ref_time_idx, max_times - 1)
# #         else:
# #             pos_batch = torch.randint(0, B, (1,), device=device).item()
# #             ref_batch_idx = torch.full((batch_size,), pos_batch, device=device)
# #             max_times = torch.clamp(x_len[pos_batch] - offset - 1, min=1)
# #             ref_time_idx = torch.randint(0 if not random_dir else offset,
# #                                           max_times.item(), (batch_size,), device=device)

# #     if random_offset:
# #         add_offset = torch.randint(1, offset + 1, (len(ref_batch_idx),), device=device, dtype=torch.long)
# #     else:
# #         add_offset = torch.full((len(ref_batch_idx),), offset, device=device, dtype=torch.long)
# #     if random_dir:
# #         dir_val = torch.randint(0, 2, add_offset.shape, device=device) * 2 - 1
# #         add_offset = add_offset * dir_val

# #     pos_time_idx = ref_time_idx + add_offset
# #     max_valid = x_len[ref_batch_idx] - 1
# #     min_val = torch.zeros_like(max_valid)
# #     pos_time_idx = torch.clamp(pos_time_idx, min=min_val, max=max_valid)

# #     if single_sequence:
# #         neg_batch_idx = torch.randint(0, B, (len(ref_batch_idx),), device=device)
# #         if B > 1:
# #             mask = neg_batch_idx == ref_batch_idx[0]
# #             neg_batch_idx[mask] = (neg_batch_idx[mask] + 1) % B
# #     else:
# #         neg_batch_idx = torch.randint(0, B, (len(ref_batch_idx),), device=device)

# #     neg_max_times = x_len[neg_batch_idx]
# #     neg_time_idx = torch.randint(0, torch.max(neg_max_times).item(), (len(ref_batch_idx),), device=device)
# #     neg_time_idx = torch.min(neg_time_idx, neg_max_times - 1)

# #     reference = x[ref_batch_idx, ref_time_idx]
# #     positive = x[ref_batch_idx, pos_time_idx]
# #     negative = x[neg_batch_idx, neg_time_idx]
# #     return (reference, positive, negative, ref_batch_idx, ref_time_idx,
# #             pos_time_idx, neg_batch_idx, neg_time_idx)


# # # =====================================================================
# # # Training loop -- adapted from professor's train_model(args)
# # # =====================================================================
# # def train_model(args: dict):
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     os.makedirs(args["out_dir"], exist_ok=True)
# #     torch.manual_seed(args["seed"])
# #     np.random.seed(args["seed"])

# #     checkpoint_address = os.path.join(args["out_dir"], "checkpoint.pt")

# #     neural_dim = 2 * args["area_6v_channels"]
# #     num_classes = charset.num_classes  # <- explicit, character-level (NOT the 41/32 hardcode)
# #     print(f"neural_dim={neural_dim} | num_classes={num_classes} (charset, incl. blank)")

# #     model = Encoder_Decoder(
# #         neural_dim, args["ceb_out"], args["kernel"], args["stride"], num_classes,
# #         args["hidden"], args["layers"], args["dropout"], args["bidir"],
# #         args["cebra_unfolder"], args["gru"], smooth_width=2.0,
# #         gauss_in=args["gauss_in"], no_rnn=args["no_rnn"],
# #         cebra_bn=args["ceb_bn"], cebra_window_10=args["cebra_window_10"],
# #     ).to(device)
# #     if torch.cuda.device_count() > 1:
# #         print(f"Using {torch.cuda.device_count()} GPUs")
# #         model = torch.nn.DataParallel(model)

# #     with open(os.path.join(args["out_dir"], "args"), "wb") as f:
# #         pickle.dump(args, f)

# #     train_loader, test_loader = get_dataset_loaders(
# #         args["datasetPath"], args["batchSize"], args["area_6v_channels"],
# #         args["max_files"], args["test_size"], args["seed"],
# #     )

# #     criterion = InfoNCE(args["temperature"])
# #     ctc_criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
# #     optimizer = torch.optim.Adam(model.parameters(), lr=args["lrStart"],
# #                                   betas=(0.9, 0.999), eps=0.1, weight_decay=args["l2_decay"])
# #     scheduler = torch.optim.lr_scheduler.LinearLR(
# #         optimizer, start_factor=1.0, end_factor=args["lrEnd"] / args["lrStart"],
# #         total_iters=args["nBatch"],
# #     )

# #     so_far_batch = load_checkpoint(checkpoint_address, model, optimizer, scheduler)
# #     print("resuming from batch:", so_far_batch)

# #     inf_losses = 0
# #     testLoss, testCER = [], []
# #     train_iter = iter(train_loader)

# #     for batch in trange(args["nBatch"]):
# #         model.train()
# #         try:
# #             X, y, X_len, y_len, dayIdx = next(train_iter)
# #         except StopIteration:
# #             train_iter = iter(train_loader)
# #             X, y, X_len, y_len, dayIdx = next(train_iter)

# #         X, y, X_len, y_len, dayIdx = (X.to(device), y.to(device), X_len.to(device),
# #                                        y_len.to(device), dayIdx.to(device))
# #         if batch < so_far_batch:
# #             continue

# #         if args["whiteNoiseSD"] > 0:
# #             X = X + torch.randn(X.shape, device=device) * args["whiteNoiseSD"]
# #         if args["constantOffsetSD"] > 0:
# #             X = X + torch.randn([X.shape[0], 1, X.shape[2]], device=device) * args["constantOffsetSD"]

# #         with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
# #             pred, lengths = model(X, X_len)
# #             m = model.module if isinstance(model, torch.nn.DataParallel) else model
# #             embeddings, emb_lengths = m.get_cebra_embs()

# #             ctc_loss = torch.sum(ctc_criterion(
# #                 torch.permute(pred.log_softmax(2), [1, 0, 2]), y, lengths, y_len))

# #             (reference, positive, negative, ref_b, ref_t, pos_t, neg_b, neg_t) = get_batch(
# #                 embeddings, emb_lengths, args["cont_batch"], args["offset"],
# #                 args["sample_single"], args["random_offset"], args["random_dir"], args["all_ref"],
# #             )
# #             loss_contrastive = criterion(reference, positive, negative)[0]
# #             loss = args["lambda_contrastive"] * loss_contrastive + ctc_loss

# #         optimizer.zero_grad()
# #         if not torch.isfinite(loss):
# #             inf_losses += 1
# #             if inf_losses > 10:
# #                 break
# #             continue
# #         loss.backward()
# #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
# #         optimizer.step()

# #         # ---------------- optional adversarial (PGD on raw input X) ----------------
# #         if args["adv"]:
# #             epsilon, steps, alpha = args["adv_eps"], args["adv_steps"], args["adv_eps"] / 5.0
# #             X_adv = X.detach().clone()
# #             if args["adv_norm"] == "linf":
# #                 X_adv = X_adv + torch.empty_like(X_adv).uniform_(-epsilon, epsilon)
# #             else:  # l2
# #                 noise = torch.randn_like(X_adv)
# #                 noise_norm = noise.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
# #                 noise = noise / noise_norm
# #                 noise = noise * (torch.rand((noise.shape[0], noise.shape[1], 1), device=device) * epsilon)
# #                 X_adv = X_adv + noise

# #             for _ in range(steps):
# #                 X_adv = X_adv.detach().requires_grad_(True)
# #                 with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
# #                     pred_adv, lengths_adv = model(X_adv, X_len)
# #                     m = model.module if isinstance(model, torch.nn.DataParallel) else model
# #                     emb_adv, emb_len_adv = m.get_cebra_embs()
# #                     ctc_loss_adv = torch.sum(ctc_criterion(
# #                         torch.permute(pred_adv.log_softmax(2), [1, 0, 2]), y, lengths_adv, y_len))
# #                     ref = emb_adv[ref_b, ref_t]
# #                     pos = emb_adv[ref_b, pos_t].detach()
# #                     neg = emb_adv[neg_b, neg_t].detach()
# #                     loss_adv = args["lambda_contrastive"] * criterion(ref, pos, neg)[0] + ctc_loss_adv

# #                 grad = torch.autograd.grad(loss_adv, X_adv, only_inputs=True)[0]
# #                 with torch.no_grad():
# #                     if args["adv_norm"] == "linf":
# #                         X_adv = X_adv + alpha * grad.sign()
# #                         delta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
# #                         X_adv = X + delta
# #                     else:
# #                         grad_norm = grad.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
# #                         X_adv = (X_adv + alpha * (grad / grad_norm)).detach()
# #                         delta = X_adv - X
# #                         delta_norm = delta.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
# #                         scale = torch.clamp(epsilon / delta_norm, max=1.0)
# #                         X_adv = (X + delta * scale).detach()

# #             X_adv = X_adv.detach()
# #             with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
# #                 pred_adv, lengths_adv = model(X_adv, X_len)
# #                 m = model.module if isinstance(model, torch.nn.DataParallel) else model
# #                 emb_adv, emb_len_adv = m.get_cebra_embs()
# #                 ctc_loss_adv = torch.sum(ctc_criterion(
# #                     torch.permute(pred_adv.log_softmax(2), [1, 0, 2]), y, lengths_adv, y_len))
# #                 ref = emb_adv[ref_b, ref_t]
# #                 pos = emb_adv[ref_b, pos_t]
# #                 neg = emb_adv[neg_b, neg_t]
# #                 loss_adv = args["lambda_contrastive"] * criterion(ref, pos, neg)[0] + ctc_loss_adv

# #             optimizer.zero_grad()
# #             if torch.isfinite(loss_adv):
# #                 loss_adv.backward()
# #                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
# #                 optimizer.step()

# #         scheduler.step()

# #         # ---------------- periodic validation (CER) ----------------
# #         if batch % args["eval_every"] == 0:
# #             model.eval()
# #             with torch.no_grad():
# #                 allLoss, total_edit_distance, total_seq_length = [], 0, 0
# #                 for Xv, yv, Xv_len, yv_len, _ in test_loader:
# #                     Xv, yv, Xv_len, yv_len = Xv.to(device), yv.to(device), Xv_len.to(device), yv_len.to(device)
# #                     with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
# #                         pred_v, lengths_v = model(Xv, Xv_len)
# #                         loss_v = torch.sum(ctc_criterion(
# #                             torch.permute(pred_v.log_softmax(2), [1, 0, 2]), yv, lengths_v, yv_len))
# #                     allLoss.append(loss_v.cpu().item())

# #                     for i in range(pred_v.shape[0]):
# #                         decoded = torch.argmax(pred_v[i, :lengths_v[i], :], dim=-1)
# #                         decoded = torch.unique_consecutive(decoded)
# #                         decoded = np.array([c for c in decoded.cpu().numpy() if c != 0])
# #                         true_seq = yv[i][:yv_len[i]].cpu().numpy()
# #                         matcher = SequenceMatcher(a=true_seq.tolist(), b=decoded.tolist())
# #                         total_edit_distance += matcher.distance()
# #                         total_seq_length += len(true_seq)

# #                 avgLoss = float(np.sum(allLoss) / max(len(test_loader), 1))
# #                 cer = total_edit_distance / max(total_seq_length, 1)
# #                 print(f"batch {batch} | val ctc loss: {avgLoss:.4f} | CER: {cer:.4f} "
# #                       f"| train ctc: {ctc_loss.item():.4f} | train cont: {loss_contrastive.item():.4f}")

# #             state_dict = (model.module if isinstance(model, torch.nn.DataParallel) else model).state_dict()
# #             torch.save(state_dict, os.path.join(args["out_dir"], "modelWeights"))
# #             save_checkpoint(checkpoint_address, model, optimizer, scheduler, batch)

# #             testLoss.append(avgLoss)
# #             testCER.append(cer)
# #             with open(os.path.join(args["out_dir"], "trainingStats"), "wb") as f:
# #                 pickle.dump({"testLoss": np.array(testLoss), "testCER": np.array(testCER)}, f)

# #     print("DONE")
# #     return model


# # if __name__ == "__main__":
# #     train_model(DEFAULT_ARGS)

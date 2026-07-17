from typing import Optional, Literal

import torch
from sklearn.model_selection import train_test_split

from utils.constants import CEBRA_DIR
import sys
sys.path.append(str(CEBRA_DIR))

from cebra import CEBRA
from cebra.data.helper import ensemble_embeddings
from utils.dataset_loader import DatasetLoader

dataset_loader = DatasetLoader()

DataNormalizationLiteral = Literal[
    "none", "z_score", "day_dist_align", "subtract_day_dist_align", "coral", "procrustes"]


def _covariance(X, eps=1e-5):
    # X: (T, N), rows are time bins (samples), columns are features (neurons)
    Xc = X - X.mean(dim=0, keepdim=True)
    # sample covariance (N x N)
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)
    # shrinkage for stability
    cov = cov + eps * torch.eye(cov.shape[0], device=X.device, dtype=X.dtype)
    return cov, Xc.mean(dim=0, keepdim=True)  # mean of Xc is ~0 but returned for completeness


def _symm_matrix_sqrt(A):
    # returns A^{1/2} and A^{-1/2} using eigen-decomposition
    # A must be symmetric PSD (we shrink above to help)
    eigvals, eigvecs = torch.linalg.eigh(A)  # eigvals ascending
    # clamp small/neg eigenvalues (numerical)
    eigvals_clamped = torch.clamp(eigvals, min=1e-12)
    A_half = (eigvecs * eigvals_clamped.sqrt()) @ eigvecs.T
    A_mhalf = (eigvecs * (1.0 / eigvals_clamped.sqrt())) @ eigvecs.T
    return A_half, A_mhalf


@torch.no_grad()
def coral_align(Xs, Xt, eps=1e-5, match_means=True):
    """
    Xs: (T_s, N) source tensor
    Xt: (T_t, N) target tensor
    Returns Xs_aligned: (T_s, N)
    """
    # means across time
    mu_s = Xs.mean(dim=0, keepdim=True)  # (1, N)
    mu_t = Xt.mean(dim=0, keepdim=True)  # (1, N)

    # centered copies
    Xs_c = Xs - mu_s
    Xt_c = Xt - mu_t

    # covariances
    Cs = (Xs_c.T @ Xs_c) / (Xs.shape[0] - 1) + eps * torch.eye(Xs.shape[1], device=Xs.device, dtype=Xs.dtype)
    Ct = (Xt_c.T @ Xt_c) / (Xt.shape[0] - 1) + eps * torch.eye(Xt.shape[1], device=Xt.device, dtype=Xt.dtype)

    # whitening and coloring
    Ct_half, Cs_mhalf = _symm_matrix_sqrt(Ct)[0], _symm_matrix_sqrt(Cs)[1]

    Xs_whitened = Xs_c @ Cs_mhalf  # remove source correlations
    Xs_colored = Xs_whitened @ Ct_half  # impose target correlations

    if match_means:
        Xs_aligned = Xs_colored + mu_t
    else:
        Xs_aligned = Xs_colored  # keep zero-mean if you plan to add means later

    return Xs_aligned


def normalize_to_target(x: torch.Tensor, dataset_name: str, cur_day: int,
                        data_normalization: DataNormalizationLiteral = 'none', target_day: int = 0,
                        verbose: bool = True) -> Optional[
    torch.Tensor]:
    if data_normalization == 'none':
        return x

    if data_normalization == 'z_score':
        target_day = cur_day

    tgt_day_data, tgt_day_label = dataset_loader.load_dataset_day(target_day, dataset_name)
    full_data_num_time_bins = tgt_day_data.shape[0]
    tgt_day_data, _, _, _ = train_test_split(tgt_day_data, tgt_day_label, test_size=0.20, shuffle=False)
    tgt_day_data = torch.from_numpy(tgt_day_data).float()

    tgt_day_mean = tgt_day_data.mean(dim=0, keepdim=True)
    tgt_day_std = tgt_day_data.std(dim=0, keepdim=True, unbiased=False)
    verbose and print("pre-normalization mean and std:", x.mean(), x.std())

    if data_normalization == 'subtract_day_dist_align':
        """
        every neuron is normalized to have zero mean and unit variance.
        for this purpose, mean and std of the train set of day-k is subtracted similarly to the tensor x.
        """
        # note that this is zero-shot, because we'll know about the missing neurons in advance!
        x_std_zero = (x.std(dim=0, keepdim=True, unbiased=False) < 1e-3).squeeze()
        x = (x - tgt_day_mean) / (tgt_day_std + 1e-8)
        # remove neurons with day0 std < 1e-2
        x[:, (tgt_day_std < 1e-3).squeeze()] = 0
        # remove missing neurons
        # note that we still remain zero-shot, because we'll know about the missing neurons in advance!
        x[:, x_std_zero] = 0
        return x
    elif data_normalization == 'day_dist_align':
        """
        every neuron is normalized to have mean and std equal to the same neuron, on day target_day.
        for this purpose, we calculate the diff of train set of day-k against train-set of day-0, and apply it to x.
        """
        if cur_day == target_day:
            return x
        x_train, x_train_label = dataset_loader.load_dataset_day(cur_day, dataset_name)
        x_train, _, _, _ = train_test_split(x_train, x_train_label, test_size=0.20, shuffle=False)
        x_train = torch.from_numpy(x_train).float()

        x_mean = x_train.mean(dim=0, keepdim=True)
        x_std = x_train.std(dim=0, keepdim=True, unbiased=False)
        std_diff = tgt_day_std / (x_std + 1e-8)
        x = (x - x_mean) * std_diff + tgt_day_mean
        # remove neurons with day0 std < 1e-2
        x[:, (tgt_day_std < 1e-3).squeeze()] = 0
        # remove neurons with x std < 1e-2
        x[:, (x_std < 1e-3).squeeze()] = 0
        return x
    elif data_normalization == 'z_score':
        """
        every neuron is normalized to have zero mean and unit variance.
        for this purpose, mean and std of the train set of day-k is subtracted from the tensor x.
        """
        assert target_day == cur_day, "z_score normalization is only supported for the same day."
        x = (x - tgt_day_mean) / (tgt_day_std + 1e-8)
        # remove neurons with x std < 1e-2
        x[:, (tgt_day_std < 1e-3).squeeze()] = 0
        return x
    elif data_normalization == 'coral':
        # note that this is zero-shot, because we'll know about the missing neurons in advance!
        if cur_day == target_day and x.shape[0] > full_data_num_time_bins // 2:
            print("x is from train/val set. coral alignment skipped.")
            return x
        # remove missing neurons
        x_std_zero = (x.std(dim=0, keepdim=True, unbiased=False) < 1e-3).squeeze()
        x[:, (tgt_day_std < 1e-3).squeeze()] = 0
        tgt_day_data[:, x_std_zero] = 0

        print("before shape:", x.shape)
        x = coral_align(x, tgt_day_data)
        print("after shape:", x.shape)

        # remove missing neurons
        x[:, (tgt_day_std < 1e-3).squeeze()] = 0
        x[:, x_std_zero] = 0

        return x
    elif data_normalization == "procrustes":
        x_std_zero = (x.std(dim=0, keepdim=True, unbiased=False) < 1e-3).squeeze()
        if cur_day == target_day and x.shape[0] > full_data_num_time_bins // 2:
            print("x is from train/val set. procrustes alignment skipped.")
            return x
        x = torch.from_numpy(ensemble_embeddings(embeddings=[tgt_day_data, x]))
        # remove neurons with x std < 1e-2
        x[:, (tgt_day_std < 1e-3).squeeze()] = 0
        # remove missing neurons
        x[:, x_std_zero] = 0
        return x

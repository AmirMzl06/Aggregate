import os

import numpy as np

from utils.cebra_decoder import TwoLayerMLP

import torch
import torch.nn as nn

import CEBRA.cebra.datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pgd_l2_attack_encoder(
        x: torch.Tensor,
        y: torch.Tensor,
        encoder: nn.Module,
        linear_model: nn.Module,
        epsilon: float = 0.05,  # Radius in L2 ball
        alpha: float = 0.01,  # Step size
        num_steps: int = 10,
):
    encoder = encoder.to(device)
    linear_model = linear_model.to(device)
    encoder.eval()
    linear_model.eval()

    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad = True
    x_orig = x.clone().detach().to(device)

    mse_loss = nn.MSELoss(reduction='mean')

    for _ in range(num_steps):
        if x_adv.grad is not None:
            x_adv.grad.zero_()

        # Forward pass
        with torch.enable_grad():
            # Repeat last dim to match your network input shape
            z = encoder(x_adv.unsqueeze(-1).repeat(1, 1, 36))
            predictions = linear_model(z)
            loss = mse_loss(predictions, y)

        # Backprop to get gradients
        loss.backward()

        with torch.no_grad():
            # 1) Take a step in the direction of normalized gradient
            grad = x_adv.grad
            grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1, keepdim=True)
            # Avoid division by zero
            grad_norm_clamped = torch.clamp(grad_norm, min=1e-8)

            grad_step = alpha * (grad / grad_norm_clamped.view(-1, 1))

            x_adv = x_adv + grad_step

            delta = x_adv - x_orig
            delta_norm = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1, keepdim=True)
            mask = (delta_norm > epsilon).float()

            delta = delta * (epsilon / delta_norm.clamp(min=1e-8)) * mask \
                    + delta * (1 - mask)

            x_adv = x_orig + delta

            x_adv.requires_grad = True

    return x_adv


def flips_generator(x: torch.Tensor):
    for i in range(x.shape[1]):
        perturbed = x.clone()
        perturbation = torch.zeros_like(x)
        perturbation[:, i] = x[:, i] * -1  # makes it all-zeros
        perturbed += perturbation
        yield perturbation, perturbed


def random_flips_generator(x: torch.Tensor, num_flips: int, flip_prob: float = 0.05):
    for _ in range(num_flips):
        perturbed = x.clone()
        perturbation = np.random.choice([-1, 0, 1], size=x.shape,
                                        p=[flip_prob / 2, 1 - flip_prob, flip_prob / 2])
        perturbation = torch.from_numpy(perturbation).float().to(device)
        perturbed += perturbation
        perturbed = torch.clamp(perturbed, min=0)
        yield perturbation, perturbed


def get_worst_flips_sorted(x: torch.Tensor, y: torch.Tensor, model: CEBRA, linear_model: TwoLayerMLP):
    flip_scores: list = []
    for perturbation, perturbed in flips_generator(x):
        score = linear_model.score(perturbed.cuda(), y, model)
        flip_scores.append((score, perturbation))
    flip_scores = sorted(flip_scores, key=lambda cell: cell[0])
    merged_perturbation = torch.zeros_like(x)
    num_neurons = x.shape[1]
    perturbed_scores = {}
    end = num_neurons
    considered_idx = {}
    considered_percentages = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    for p in considered_percentages:
        if round(p * end) not in considered_idx:
            considered_idx[round(p * end)] = p
    for i in range(0, end):
        merged_perturbation += flip_scores[i][1]
        if i in considered_idx:
            p = considered_idx[i]
            perturbed = x.clone() + merged_perturbation
            score = linear_model.score(perturbed.cuda(), y, model)
            perturbed_scores[p] = score
    return perturbed_scores


def get_random_flips_score(x: torch.Tensor, y: torch.Tensor, model: CEBRA, linear_model: TwoLayerMLP,
                           num_flips: int, flip_prob: float = 0.05):
    flips: list[tuple[tuple, torch.Tensor, torch.Tensor]] = []
    for perturbation, perturbed in random_flips_generator(x, num_flips, flip_prob):
        scores = linear_model.score(perturbed.cuda(), y, model)
        flips.append((scores, perturbation, perturbed))
    mean_scores: list = []
    for i in range(len(flips[0][0])):
        mean_scores.append(np.mean([flip[0][i] for flip in flips]))
    return mean_scores

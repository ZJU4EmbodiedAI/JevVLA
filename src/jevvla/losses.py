from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_preference_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2 or target.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("logits, target and mask must have shape [B,K]")
    valid = mask.bool()
    if not bool(valid.any(dim=1).all()):
        raise ValueError("every anchor needs a valid candidate")
    if not bool(torch.isfinite(target).all()) or not bool((target >= 0).all()):
        raise ValueError("target must be finite and nonnegative")
    if not bool((target.masked_fill(valid, 0) == 0).all()):
        raise ValueError("invalid candidates cannot carry target mass")
    if not bool(torch.allclose(target.sum(dim=1), torch.ones_like(target[:, 0]), atol=1e-5)):
        raise ValueError("target probabilities must sum to one")
    masked = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    return -(target * F.log_softmax(masked, dim=1)).sum(dim=1).mean()


def pairwise_outcome_loss(
    logits: torch.Tensor, outcome: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2 or outcome.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("logits, outcome and mask must have shape [B,K]")
    valid = mask.bool()
    better = (outcome[:, :, None] > outcome[:, None, :]) & valid[:, :, None] & valid[:, None, :]
    count = better.sum(dim=(1, 2))
    informative = count > 0
    safe_logits = logits.masked_fill(~valid, 0.0)
    if not bool(informative.any()):
        return safe_logits.sum() * 0.0
    penalty = F.softplus(-(safe_logits[:, :, None] - safe_logits[:, None, :]))
    per_anchor = (penalty * better).sum(dim=(1, 2)) / count.clamp_min(1)
    return per_anchor[informative].mean()


def gaussian_gradient_target(
    noisy_action: torch.Tensor, preferred_action: torch.Tensor,
    sigma: torch.Tensor | float,
) -> torch.Tensor:
    sigma = torch.as_tensor(sigma, dtype=noisy_action.dtype, device=noisy_action.device)
    if noisy_action.shape != preferred_action.shape or bool((sigma <= 0).any()):
        raise ValueError("invalid Gaussian target inputs")
    if sigma.ndim == 1:
        sigma = sigma[:, None]
    return (preferred_action - noisy_action) / sigma.square()


def gradient_matching_loss(
    predicted_score: torch.Tensor, target_score: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    coordinate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if predicted_score.shape != target_score.shape or predicted_score.ndim != 2:
        raise ValueError("score tensors must have shape [B,D]")
    error = (predicted_score - target_score).square()
    if coordinate_mask is not None:
        if coordinate_mask.shape not in (error.shape, error.shape[1:]):
            raise ValueError("coordinate mask shape mismatch")
        error = error * coordinate_mask
    per_sample = error.sum(dim=1)
    if sample_weight is None:
        return per_sample.mean()
    if sample_weight.shape != per_sample.shape or bool((sample_weight < 0).any()):
        raise ValueError("invalid sample weights")
    return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)


def denoising_score_matching_loss(
    predicted_score: torch.Tensor, noise: torch.Tensor,
    sigma: torch.Tensor | float, *, sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if predicted_score.shape != noise.shape or predicted_score.ndim != 2:
        raise ValueError("score and noise must have shape [B,D]")
    sigma = torch.as_tensor(sigma, dtype=predicted_score.dtype, device=predicted_score.device)
    if bool((sigma <= 0).any()):
        raise ValueError("sigma must be positive")
    if sigma.ndim == 1:
        sigma = sigma[:, None]
    per_sample = (sigma * predicted_score + noise).square().mean(dim=1)
    if sample_weight is None:
        return per_sample.mean()
    if sample_weight.shape != per_sample.shape or bool((sample_weight < 0).any()):
        raise ValueError("invalid sample weights")
    return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)


def mixture_gradient_target(
    noisy_action: torch.Tensor, preferred_actions: torch.Tensor,
    masses: torch.Tensor, sigma: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (noisy_action.ndim != 2 or preferred_actions.ndim != 3
            or preferred_actions.shape[0] != noisy_action.shape[0]
            or preferred_actions.shape[2] != noisy_action.shape[1]
            or masses.shape != preferred_actions.shape[:2]):
        raise ValueError("mixture tensors must have shapes [B,D], [B,K,D], [B,K]")
    sigma = torch.as_tensor(sigma, dtype=noisy_action.dtype, device=noisy_action.device)
    if sigma.ndim == 1:
        sigma = sigma[:, None]
    if bool((sigma <= 0).any()) or bool((masses < 0).any()) or not bool((masses.sum(dim=1) > 0).all()):
        raise ValueError("sigma and mixture masses must be valid")
    displacement = preferred_actions - noisy_action[:, None, :]
    variance = sigma.square()
    log_mass = torch.where(masses > 0, masses.clamp_min(torch.finfo(masses.dtype).tiny).log(),
                           torch.full_like(masses, -torch.inf))
    posterior = torch.softmax(log_mass - displacement.square().sum(dim=-1) / (2 * variance), dim=1)
    target = (posterior[:, :, None] * displacement).sum(dim=1) / variance
    return target, posterior


def noise_dsm(
    predicted_score: torch.Tensor, epsilon: torch.Tensor,
    sigma: torch.Tensor | float, weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return denoising_score_matching_loss(predicted_score, epsilon, sigma,
                                         sample_weight=weight)


def mixture_score(
    action: torch.Tensor, centers: torch.Tensor,
    weights: torch.Tensor, sigma: torch.Tensor | float,
) -> torch.Tensor:
    return mixture_gradient_target(action, centers, weights, sigma)[0]

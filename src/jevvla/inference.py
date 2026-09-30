from __future__ import annotations

from collections.abc import Callable

import torch

from .model import UnifiedQuadraticEnergy, bounded_action_features


def candidate_logits(
    model: UnifiedQuadraticEnergy,
    context: torch.Tensor,
    candidate: torch.Tensor,
    direct_index: torch.Tensor,
    *,
    prior: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    gain: float = 1.0,
    prior_scale: float = 1.0,
    direct_margin: float = 0.0,
) -> torch.Tensor:
    if prior is None:
        prior = candidate.new_zeros(candidate.shape[:2])
    if mask is None:
        mask = torch.ones(candidate.shape[:2], dtype=torch.bool, device=candidate.device)
    return model.logits(context, candidate, prior, mask, direct_index,
                        gain=gain, prior_scale=prior_scale, direct_margin=direct_margin)


def candidate_probabilities(
    model: UnifiedQuadraticEnergy,
    context: torch.Tensor,
    candidate: torch.Tensor,
    direct_index: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return candidate_logits(model, context, candidate, direct_index, **kwargs).softmax(dim=1)


def select_candidate(
    model: UnifiedQuadraticEnergy,
    context: torch.Tensor,
    candidate: torch.Tensor,
    direct_index: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    logits = candidate_logits(model, context, candidate, direct_index, **kwargs)
    choice = logits.argmax(dim=1)
    direct_score = logits.gather(1, direct_index.long()[:, None])[:, 0]
    return torch.where(direct_score == logits.max(dim=1).values, direct_index.long(), choice)


def correct_action(
    model: UnifiedQuadraticEnergy,
    context: torch.Tensor,
    action: torch.Tensor,
    *,
    gain: float,
    action_encoder: Callable[[torch.Tensor], torch.Tensor] = bounded_action_features,
    component_clip: float | None = None,
    lower_bound: torch.Tensor | float | None = None,
    upper_bound: torch.Tensor | float | None = None,
    coordinate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if action.ndim != 2 or action.shape[0] != context.shape[0] or gain < 0:
        raise ValueError("invalid action or correction gain")
    if component_clip is not None and component_clip <= 0:
        raise ValueError("component clip must be positive")
    with torch.enable_grad():
        working = action.detach().requires_grad_(True)
        encoded = action_encoder(working)
        score = model.raw_energy(context, encoded[:, None]).sum()
        direction = torch.autograd.grad(score, working)[0]
    correction = gain * direction
    if coordinate_mask is not None:
        correction = correction * coordinate_mask
    if component_clip is not None:
        correction = correction.clamp(-component_clip, component_clip)
    result = action + correction.detach()
    if lower_bound is not None:
        result = torch.maximum(result, torch.as_tensor(lower_bound, dtype=result.dtype, device=result.device))
    if upper_bound is not None:
        result = torch.minimum(result, torch.as_tensor(upper_bound, dtype=result.dtype, device=result.device))
    return result

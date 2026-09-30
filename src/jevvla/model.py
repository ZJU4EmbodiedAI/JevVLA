from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


SCHEMA = "jev_unified_quadratic_energy_v1"
CONTEXT_DIM = 128
CANDIDATE_DIM = 32


def pad_context(context: torch.Tensor) -> torch.Tensor:
    if context.ndim != 2 or context.shape[1] not in (32, CONTEXT_DIM):
        raise ValueError("context must have shape [B,32] or [B,128]")
    if context.shape[1] == CONTEXT_DIM:
        return context
    return F.pad(context, (0, CONTEXT_DIM - 32))


def bounded_action_features(action: torch.Tensor, *, scale: float = 4.0) -> torch.Tensor:
    if action.ndim != 2 or action.shape[1] != CANDIDATE_DIM:
        raise ValueError("native action must have shape [B,32]")
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError("action codec scale must be positive and finite")
    return scale * torch.tanh(action / scale)


class UnifiedQuadraticEnergy(nn.Module):
    def __init__(self, width: int = 128, rank: int = 8):
        super().__init__()
        if width < 8 or rank < 1 or rank > CANDIDATE_DIM:
            raise ValueError("invalid width or quadratic rank")
        self.width, self.rank = int(width), int(rank)
        self.context = nn.Sequential(nn.Linear(CONTEXT_DIM, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU())
        self.action = nn.Sequential(nn.Linear(CANDIDATE_DIM, width), nn.SiLU())
        self.residual = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(),
                                      nn.Linear(width, 1))
        self.linear = nn.Linear(width, CANDIDATE_DIM)
        self.precision = nn.Linear(width, CANDIDATE_DIM)
        self.low_rank_precision = nn.Linear(width, rank)
        self.low_rank = nn.Linear(CANDIDATE_DIM, rank, bias=False)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.precision.weight)
        nn.init.constant_(self.precision.bias, -4.0)
        nn.init.zeros_(self.low_rank_precision.weight)
        nn.init.constant_(self.low_rank_precision.bias, -4.0)

    def raw_energy(self, context: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        context = pad_context(context)
        if (candidate.ndim != 3 or candidate.shape[0] != context.shape[0]
                or candidate.shape[2] != CANDIDATE_DIM or candidate.shape[1] < 1):
            raise ValueError("candidate must have shape [B,K,32]")
        if context.device != candidate.device or context.dtype != candidate.dtype:
            raise ValueError("context/candidate device or dtype mismatch")
        hidden = self.context(context)
        linear = self.linear(hidden)
        precision = F.softplus(self.precision(hidden))
        rank_precision = F.softplus(self.low_rank_precision(hidden))
        projection = self.low_rank(candidate)
        quadratic = (linear[:, None] * candidate
                     - 0.5 * precision[:, None] * candidate.square()).sum(dim=-1)
        quadratic -= 0.5 * (rank_precision[:, None] * projection.square()).sum(dim=-1)
        shared = hidden[:, None, :].expand(-1, candidate.shape[1], -1)
        residual = self.residual(torch.cat((shared, self.action(candidate)), dim=-1))[:, :, 0]
        return quadratic + residual

    def score(self, context: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        return self.raw_energy(context, candidate)

    def energy(self, context: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        return -self.raw_energy(context, candidate)

    def forward(self, context: torch.Tensor, candidate: torch.Tensor,
                direct_index: torch.Tensor) -> torch.Tensor:
        if direct_index.shape != (len(context),):
            raise ValueError("Direct index shape mismatch")
        if torch.any(direct_index < 0) or torch.any(direct_index >= candidate.shape[1]):
            raise ValueError("Direct index outside candidate set")
        score = self.raw_energy(context, candidate)
        return score - score.gather(1, direct_index.long()[:, None])

    def continuous_score(self, context: torch.Tensor, action: torch.Tensor, *,
                         create_graph: bool | None = None, scale: float = 4.0) -> torch.Tensor:
        with torch.enable_grad():
            if not action.requires_grad:
                action = action.detach().requires_grad_(True)
            feature = bounded_action_features(action, scale=scale)
            score = self.raw_energy(context, feature[:, None]).sum()
            return torch.autograd.grad(
                score, action, create_graph=self.training if create_graph is None else create_graph)[0]

    def logits(self, context: torch.Tensor, candidate: torch.Tensor,
               prior: torch.Tensor, mask: torch.Tensor, direct_index: torch.Tensor,
               *, gain: float = 1.0, prior_scale: float = 1.0,
               direct_margin: float = 0.0) -> torch.Tensor:
        if (not math.isfinite(gain) or gain < 0 or not math.isfinite(prior_scale)
                or prior_scale <= 0 or not math.isfinite(direct_margin)
                or direct_margin < 0):
            raise ValueError("invalid global selection constants")
        if prior.shape != candidate.shape[:2] or mask.shape != prior.shape:
            raise ValueError("prior/mask geometry mismatch")
        direct_mask = mask.gather(1, direct_index.long()[:, None])
        if not bool(direct_mask.all()):
            raise ValueError("Direct candidate is masked")
        centered_prior = prior - prior.gather(1, direct_index.long()[:, None])
        result = centered_prior / prior_scale + gain * self(context, candidate, direct_index)
        result = result.masked_fill(~mask.bool(), torch.finfo(result.dtype).min)
        return result.scatter_add(1, direct_index.long()[:, None],
                                  torch.full_like(direct_mask, direct_margin, dtype=result.dtype))

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str = "cpu") -> "UnifiedQuadraticEnergy":
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("schema") != SCHEMA or payload.get("architecture") != cls.__name__:
            raise ValueError("unified quadratic checkpoint identity mismatch")
        model = cls(width=int(payload["width"]), rank=int(payload["rank"]))
        model.load_state_dict(payload["model"], strict=True)
        return model.to(device).eval()


JevVLA = UnifiedQuadraticEnergy

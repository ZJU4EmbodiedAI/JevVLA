from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch


class DiscreteAdapter:
    def __init__(self, model: torch.nn.Module, encoder: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]], *, gain: float = 1.0, prior_scale: float = 1.0, direct_margin: float = 0.0, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.encoder = encoder
        self.gain = float(gain)
        self.prior_scale = float(prior_scale)
        self.direct_margin = float(direct_margin)
        self.device = torch.device(device)

    def score(self, *args: Any, direct_index: int = 0, mask: np.ndarray | None = None, **kwargs: Any) -> np.ndarray:
        context, candidate, prior = self.encoder(*args, direct_index=direct_index, **kwargs)
        context = np.asarray(context, dtype=np.float32)
        candidate = np.asarray(candidate, dtype=np.float32)
        prior = np.asarray(prior, dtype=np.float32)
        if context.ndim == 1:
            context = context[None]
            candidate = candidate[None]
            prior = prior[None]
        if mask is None:
            mask = np.ones(prior.shape, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim == 1:
            mask = mask[None]
        if mask.shape != prior.shape:
            raise ValueError("mask shape mismatch")
        indices = np.full(len(context), direct_index, dtype=np.int64)
        with torch.no_grad():
            logits = self.model.logits(
                torch.as_tensor(context, device=self.device),
                torch.as_tensor(candidate, device=self.device),
                torch.as_tensor(prior, device=self.device),
                torch.as_tensor(mask, device=self.device),
                torch.as_tensor(indices, device=self.device),
                gain=self.gain,
                prior_scale=self.prior_scale,
                direct_margin=self.direct_margin,
            )
        return logits.cpu().numpy()

    def select(self, actions: Any, *args: Any, direct_index: int = 0, mask: np.ndarray | None = None, **kwargs: Any) -> tuple[Any, int, np.ndarray]:
        scores = self.score(*args, direct_index=direct_index, mask=mask, **kwargs)
        if scores.shape[0] != 1 or len(actions) != scores.shape[1]:
            raise ValueError("actions and scores must describe one candidate set")
        index = direct_index if scores[0, direct_index] == scores[0].max() else int(np.argmax(scores[0]))
        return actions[index], index, scores[0]


class CandidatePolicy:
    def __init__(self, propose: Callable[[Any], tuple[Any, tuple[Any, ...], dict[str, Any]]], adapter: DiscreteAdapter):
        self.propose = propose
        self.adapter = adapter

    def __call__(self, observation: Any) -> Any:
        actions, encoder_args, encoder_kwargs = self.propose(observation)
        return self.adapter.select(actions, *encoder_args, **encoder_kwargs)[0]

    def infer(self, observation: Any) -> dict[str, Any]:
        actions, encoder_args, encoder_kwargs = self.propose(observation)
        action, index, scores = self.adapter.select(actions, *encoder_args, **encoder_kwargs)
        return {"actions": action, "candidate_index": index, "scores": scores}

    def reset(self):
        if callable(getattr(self.propose, "reset", None)):
            self.propose.reset()

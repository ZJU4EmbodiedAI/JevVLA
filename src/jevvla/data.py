from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch


ROW_FIELDS = frozenset({
    "context", "candidate", "prior", "candidate_mask", "direct_index",
    "target", "terminal_success", "native_action", "action", "gradient_target",
    "centers", "mixture_weights", "sigma", "weight", "episode_id",
})


class TrainingData:
    def __init__(self, values: Mapping[str, np.ndarray | torch.Tensor]):
        self.values = {}
        for name, value in values.items():
            if name not in ROW_FIELDS and name != "coordinate_mask":
                continue
            array = np.asarray(value) if not isinstance(value, torch.Tensor) else value
            if name == "episode_id":
                self.episode_id = np.asarray(value).astype(str)
                continue
            dtype = torch.bool if name in {"candidate_mask", "coordinate_mask"} else (
                torch.long if name == "direct_index" else torch.float32
            )
            self.values[name] = torch.as_tensor(array, dtype=dtype).clone()
        self._validate()

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as archive:
            return cls({name: archive[name] for name in archive.files})

    def __len__(self):
        return len(self.values["context"])

    def batch(self, indices: torch.Tensor, device: torch.device):
        return {
            name: (value[indices.cpu()] if name in ROW_FIELDS else value).to(device)
            for name, value in self.values.items()
        }

    def _validate(self):
        if "context" not in self.values:
            raise ValueError("context is required")
        context = self.values["context"]
        if context.ndim != 2 or context.shape[1] not in (32, 128) or len(context) < 1:
            raise ValueError("context must have shape [N,32] or [N,128]")
        n = len(context)
        for name, value in self.values.items():
            if name in ROW_FIELDS and (value.ndim == 0 or len(value) != n):
                raise ValueError(f"{name} must have {n} rows")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains nonfinite values")
        if hasattr(self, "episode_id") and self.episode_id.shape != (n,):
            raise ValueError("episode_id must have shape [N]")
        if "candidate" in self.values:
            candidate = self.values["candidate"]
            if candidate.ndim != 3 or candidate.shape[2] != 32 or candidate.shape[1] < 1:
                raise ValueError("candidate must have shape [N,K,32]")
            shape = candidate.shape[:2]
            self.values.setdefault("candidate_mask", torch.ones(shape, dtype=torch.bool))
            self.values.setdefault("prior", torch.zeros(shape))
            self.values.setdefault("direct_index", torch.zeros(n, dtype=torch.long))
            for name in ("candidate_mask", "prior", "target", "terminal_success"):
                if name in self.values and self.values[name].shape != shape:
                    raise ValueError(f"{name} must have shape [N,K]")
            direct = self.values["direct_index"]
            if direct.shape != (n,) or bool(((direct < 0) | (direct >= shape[1])).any()):
                raise ValueError("direct_index is outside the candidate set")
            mask = self.values["candidate_mask"]
            if not bool(mask.gather(1, direct[:, None]).all()):
                raise ValueError("the Direct candidate must be valid")
            if "target" in self.values:
                target = self.values["target"]
                if bool((target < 0).any()) or bool((target[~mask] != 0).any()):
                    raise ValueError("target must be nonnegative and zero on invalid candidates")
                if not torch.allclose(target.sum(1), torch.ones(n), atol=1e-5):
                    raise ValueError("target rows must sum to one")
            if "terminal_success" in self.values:
                outcome = self.values["terminal_success"]
                if not bool(((outcome == 0) | (outcome == 1)).all()):
                    raise ValueError("terminal_success must contain binary outcomes")
            if "native_action" in self.values:
                native = self.values["native_action"]
                if native.ndim != 3 or native.shape[:2] != shape or not 1 <= native.shape[2] <= 32:
                    raise ValueError("native_action must have shape [N,K,D], D <= 32")
        for name in ("action", "gradient_target"):
            if name in self.values:
                value = self.values[name]
                if value.ndim != 2 or not 1 <= value.shape[1] <= 32:
                    raise ValueError(f"{name} must have shape [N,D], D <= 32")
        if "gradient_target" in self.values:
            if "action" not in self.values or self.values["gradient_target"].shape != self.values["action"].shape:
                raise ValueError("gradient_target must match action")
        if "centers" in self.values:
            centers = self.values["centers"]
            if centers.ndim != 3 or not 1 <= centers.shape[2] <= 32:
                raise ValueError("centers must have shape [N,K,D], D <= 32")
            if "mixture_weights" not in self.values or self.values["mixture_weights"].shape != centers.shape[:2]:
                raise ValueError("mixture_weights must match centers")
            weights = self.values["mixture_weights"]
            if bool((weights < 0).any()) or bool((weights.sum(1) <= 0).any()):
                raise ValueError("mixture_weights must have positive row mass")
        if "weight" in self.values:
            weights = self.values["weight"]
            if weights.shape != (n,) or bool((weights < 0).any()) or float(weights.sum()) <= 0:
                raise ValueError("weight must be a nonnegative [N] vector with positive mass")
        if "sigma" in self.values:
            sigma = self.values["sigma"]
            if sigma.shape != (n,) or bool((sigma <= 0).any()):
                raise ValueError("sigma must be a positive [N] vector")
        if "coordinate_mask" in self.values:
            mask = self.values["coordinate_mask"]
            if mask.ndim != 1 or not bool(mask.any()):
                raise ValueError("coordinate_mask must select at least one action coordinate")


def check_disjoint(train: TrainingData, validation: TrainingData):
    if hasattr(train, "episode_id") and hasattr(validation, "episode_id"):
        if set(train.episode_id) & set(validation.episode_id):
            raise ValueError("training and validation episodes overlap")

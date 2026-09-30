from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np


def encode_context(feature: np.ndarray, step: int, *, max_steps: int = 200) -> np.ndarray:
    feature = np.asarray(feature, np.float32)
    if feature.shape != (128,) or not np.isfinite(feature).all():
        raise ValueError("feature must be a finite 128-vector")
    if not 0 <= step < max_steps:
        raise ValueError("step outside episode horizon")
    indices = np.arange(128)
    bins = indices % 31
    signs = np.where((indices // 31) % 2 == 0, 1.0, -1.0)
    sums = np.bincount(bins, weights=feature.astype(np.float64) * signs, minlength=31)
    counts = np.bincount(bins, minlength=31)
    context = np.zeros(128, np.float32)
    context[:31] = np.tanh(sums / np.sqrt(counts))
    context[31] = step / max_steps
    return context


def encode_actions(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, np.float32)
    if actions.ndim != 2 or actions.shape[1] != 2 or not np.isfinite(actions).all():
        raise ValueError("actions must have shape [K,2]")
    encoded = np.zeros((len(actions), 32), np.float32)
    encoded[:, :2] = np.tanh(10.0 * actions)
    return encoded


def prepare_initial(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = ("context", "candidate", "native_action", "target", "task", "seed")
    if any(name not in arrays for name in required):
        raise ValueError("initial data is missing required fields")
    context = np.asarray(arrays["context"], np.float32)
    candidate = np.asarray(arrays["candidate"], np.float32)
    native = np.asarray(arrays["native_action"], np.float32)
    n = len(context)
    if (context.ndim != 2 or context.shape[1] not in (32, 128) or candidate.ndim != 3
            or candidate.shape[0] != n or candidate.shape[2] != 32
            or native.shape != (n, candidate.shape[1], 2)):
        raise ValueError("invalid initial context, candidate, or action geometry")
    if context.shape[1] == 32:
        context = np.pad(context, ((0, 0), (0, 96)))
    chosen = np.asarray(arrays.get("fit_mask", np.ones(n, bool)), bool)
    if chosen.shape != (n,):
        raise ValueError("initial fit mask must have one value per anchor")
    target = np.asarray(arrays["target"], np.float32)
    mask = np.asarray(arrays.get("candidate_mask", np.ones(candidate.shape[:2], bool)), bool)
    prior = np.asarray(arrays.get("prior", np.zeros(candidate.shape[:2], np.float32)), np.float32)
    direct = np.asarray(arrays.get("direct_index", np.zeros(n, np.int64)), np.int64)
    task = np.asarray(arrays["task"]).astype(str)
    seed = np.asarray(arrays["seed"])
    if (target.shape != candidate.shape[:2] or mask.shape != target.shape or prior.shape != target.shape
            or direct.shape != (n,) or task.shape != (n,) or seed.shape != (n,)):
        raise ValueError("invalid initial candidate metadata")
    return {
        "context": context[chosen], "candidate": candidate[chosen],
        "native_action": native[chosen], "target": target[chosen],
        "candidate_mask": mask[chosen], "prior": prior[chosen],
        "direct_index": direct[chosen],
        "episode_id": np.asarray([f"{t}:{s}" for t, s in zip(task[chosen], seed[chosen])]),
    }


def prepare_future(arrays: Mapping[str, np.ndarray], *, max_steps: int = 200) -> dict[str, np.ndarray]:
    required = ("frozen_BC_feature", "future_step", "native_expert_action",
                "branch_preference_weight", "task", "seed", "candidate_index")
    if any(name not in arrays for name in required):
        raise ValueError("future data is missing required fields")
    feature = np.asarray(arrays["frozen_BC_feature"], np.float32)
    step = np.asarray(arrays["future_step"])
    action = np.asarray(arrays["native_expert_action"], np.float32)
    mass = np.asarray(arrays["branch_preference_weight"], np.float32)
    task = np.asarray(arrays["task"]).astype(str)
    seed = np.asarray(arrays["seed"])
    candidate = np.asarray(arrays["candidate_index"])
    n = len(feature)
    if (feature.shape != (n, 128) or action.shape != (n, 2) or
            any(value.shape != (n,) for value in (step, mass, task, seed, candidate))):
        raise ValueError("invalid future row geometry")
    selected = np.asarray(arrays.get("future_mask", np.ones(n, bool)), bool).copy()
    selected &= np.asarray(arrays.get("fit_mask", np.ones(n, bool)), bool)
    if selected.shape != (n,) or not selected.any():
        raise ValueError("future mask must select at least one row")
    feature, step, action, mass = feature[selected], step[selected], action[selected], mass[selected]
    task, seed, candidate = task[selected], seed[selected], candidate[selected]
    if not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("branch preference weights must be nonnegative and finite")
    context = np.stack([encode_context(f, int(t), max_steps=max_steps)
                        for f, t in zip(feature, step)])
    episode = np.asarray([f"{t}:{s}" for t, s in zip(task, seed)])
    weights = np.zeros(len(episode), np.float32)
    for identity in np.unique(episode):
        group = np.flatnonzero(episode == identity)
        for branch in np.unique(candidate[group]):
            members = group[candidate[group] == branch]
            if not np.allclose(mass[members], mass[members[0]], rtol=1e-5, atol=1e-7):
                raise ValueError("branch preference differs within one executed branch")
            weights[members] = mass[members[0]] / len(members)
        total = weights[group].sum()
        if total <= 0:
            raise ValueError("episode has no positive branch preference mass")
        weights[group] /= total
    return {"context": context, "action": action, "weight": weights,
            "episode_id": episode}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


def _silu_prime(x: np.ndarray) -> np.ndarray:
    s = _sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


class LanguageTableAdapter:
    def __init__(self, state: Mapping[str, np.ndarray]):
        self.state = {name: np.asarray(value, np.float32) for name, value in state.items()}
        expected = {
            "context.0.weight": (128, 128), "context.0.bias": (128,),
            "context.2.weight": (128, 128), "context.2.bias": (128,),
            "action.0.weight": (128, 32), "action.0.bias": (128,),
            "residual.0.weight": (128, 256), "residual.0.bias": (128,),
            "residual.2.weight": (1, 128), "residual.2.bias": (1,),
            "linear.weight": (32, 128), "linear.bias": (32,),
            "precision.weight": (32, 128), "precision.bias": (32,),
            "low_rank_precision.weight": (8, 128), "low_rank_precision.bias": (8,),
            "low_rank.weight": (8, 32),
        }
        if {name: value.shape for name, value in self.state.items()} != expected:
            raise ValueError("expected width-128 rank-8 scalar energy weights")

    @classmethod
    def from_npz(cls, path: str | Path) -> LanguageTableAdapter:
        with np.load(path, allow_pickle=False) as archive:
            return cls({name: archive[name] for name in archive.files})

    def _affine(self, x: np.ndarray, name: str) -> np.ndarray:
        return x @ self.state[name + ".weight"].T + self.state[name + ".bias"]

    def _utility_and_gradient(self, context: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = _silu(self._affine(_silu(self._affine(context, "context.0")), "context.2"))
        linear = self._affine(hidden, "linear")
        precision = np.logaddexp(self._affine(hidden, "precision"), 0.0)
        rank_precision = np.logaddexp(self._affine(hidden, "low_rank_precision"), 0.0)
        projection = candidate @ self.state["low_rank.weight"].T
        utility = np.sum(candidate * linear - 0.5 * precision * candidate**2, axis=-1)
        utility -= 0.5 * np.sum(rank_precision * projection**2, axis=-1)
        action_pre = self._affine(candidate, "action.0")
        action_hidden = _silu(action_pre)
        residual_pre = self._affine(
            np.concatenate((np.broadcast_to(hidden, action_hidden.shape), action_hidden), axis=-1),
            "residual.0",
        )
        residual = self._affine(_silu(residual_pre), "residual.2")[..., 0]
        derivative = linear - precision * candidate
        derivative -= (rank_precision * projection) @ self.state["low_rank.weight"]
        back = self.state["residual.2.weight"][0] * _silu_prime(residual_pre)
        back = (back @ self.state["residual.0.weight"][:, 128:]) * _silu_prime(action_pre)
        derivative += back @ self.state["action.0.weight"]
        return utility + residual, derivative

    def utility(self, feature: np.ndarray, actions: np.ndarray, step: int, *, max_steps: int = 200) -> np.ndarray:
        context = encode_context(feature, step, max_steps=max_steps)
        values, _ = self._utility_and_gradient(context, encode_actions(actions))
        return np.asarray(values, np.float32)

    def gradient(self, feature: np.ndarray, action: np.ndarray, step: int, *, max_steps: int = 200) -> np.ndarray:
        native = np.asarray(action, np.float32)
        if native.shape != (2,):
            raise ValueError("action must be a 2-vector")
        context = encode_context(feature, step, max_steps=max_steps)
        candidate = encode_actions(native[None])
        _, derivative = self._utility_and_gradient(context, candidate)
        return np.asarray(derivative[0, :2] * 10.0 * (1.0 - candidate[0, :2] ** 2), np.float32)

    def refine(self, feature: np.ndarray, action: np.ndarray, step: int, *, gain: float = 0.0004,
               max_steps: int = 200, action_limit: float = 0.1) -> np.ndarray:
        native = np.asarray(action, np.float32)
        if not np.isfinite(gain) or gain < 0 or action_limit <= 0:
            raise ValueError("invalid gain or action limit")
        return np.clip(native + gain * self.gradient(feature, native, step, max_steps=max_steps),
                       -action_limit, action_limit).astype(native.dtype)

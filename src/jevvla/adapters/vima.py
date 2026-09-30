from __future__ import annotations

import numpy as np

from .discrete import CandidatePolicy, DiscreteAdapter


ACTION_SCALE = np.asarray([10, 10, 1, 1, 1, 1, 10, 10, 1, 1, 1, 1], dtype=np.float32)


def encode_observation(observation: np.ndarray, step_fraction: float = 0.0) -> np.ndarray:
    raw = np.asarray(observation, dtype=np.float32)
    if raw.shape != (768,) or not np.isfinite(raw).all() or not np.isfinite(step_fraction) or not 0 <= step_fraction <= 1:
        raise ValueError("invalid VIMA observation or progress")
    index = np.arange(raw.size)
    bins = index % 31
    signs = np.where((index // 31) % 2 == 0, 1.0, -1.0)
    sums = np.bincount(bins, weights=raw.astype(np.float64) * signs, minlength=31)
    counts = np.bincount(bins, minlength=31)
    context = np.empty(32, dtype=np.float32)
    context[:31] = np.tanh(sums / np.sqrt(np.maximum(counts, 1)))
    context[31] = step_fraction
    return context


def encode(observation: np.ndarray, actions: np.ndarray, prior: np.ndarray, *, step_fraction: float = 0.0, direct_index: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float32)
    prior = np.asarray(prior, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != len(ACTION_SCALE) or len(actions) < 1 or prior.shape != (len(actions),) or not 0 <= direct_index < len(actions):
        raise ValueError("invalid VIMA candidate geometry")
    if not np.isfinite(actions).all() or not np.isfinite(prior).all():
        raise ValueError("nonfinite VIMA candidates")
    context = encode_observation(observation, step_fraction)
    candidate = np.zeros((len(actions), 32), dtype=np.float32)
    candidate[:, :len(ACTION_SCALE)] = np.tanh(actions * ACTION_SCALE)
    return context, candidate, prior


def outcome_preference(components: np.ndarray, future_mask: np.ndarray, terminal_class: np.ndarray, suffix_success: np.ndarray, *, tau: float) -> np.ndarray:
    components = np.asarray(components, dtype=np.float32)
    future_mask = np.asarray(future_mask, dtype=bool)
    terminal_class = np.asarray(terminal_class)
    suffix_success = np.asarray(suffix_success, dtype=bool)
    if components.ndim != 3 or components.shape[2] != 4 or future_mask.shape != components.shape[:2] or terminal_class.shape != (len(components),) or suffix_success.shape != (len(components),) or not np.isfinite(components).all() or not np.isfinite(tau) or tau <= 0:
        raise ValueError("invalid VIMA preference evidence")
    target = np.zeros(len(components), dtype=np.float32)
    if suffix_success.any():
        selected = np.flatnonzero(suffix_success)
        if not future_mask[selected].any(axis=1).all():
            target[selected] = 1 / len(selected)
            return target
    else:
        selected = np.flatnonzero((terminal_class == 0) & future_mask.any(axis=1))
        if not len(selected):
            return target
    errors = np.asarray([(components[index].mean(axis=-1) * future_mask[index]).sum() / future_mask[index].sum() for index in selected])
    weights = np.exp(-(errors - errors.min()) / tau)
    target[selected] = weights / weights.sum()
    return target


def make_policy(model, *, propose=None, gain: float = 4.0, prior_scale: float = 20.0, direct_margin: float = 0.0, device: str = "cpu") -> DiscreteAdapter | CandidatePolicy:
    adapter = DiscreteAdapter(model, encode, gain=gain, prior_scale=prior_scale, direct_margin=direct_margin, device=device)
    return CandidatePolicy(propose, adapter) if propose is not None else adapter

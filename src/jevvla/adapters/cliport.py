from __future__ import annotations

import numpy as np

from .discrete import CandidatePolicy, DiscreteAdapter


def _basis(values: np.ndarray, dimensions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / max(len(values) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    components = eigenvectors[:, np.argsort(eigenvalues)[-dimensions:][::-1]].T.astype(np.float32)
    for axis in components:
        index = np.argmax(np.abs(axis))
        if axis[index] < 0:
            axis *= -1
    projected = centered @ components.T
    scale = np.maximum(projected.std(axis=0), 1e-3).astype(np.float32)
    return mean.astype(np.float32), components, scale


def fit_codec(visual: np.ndarray, language: np.ndarray) -> dict[str, np.ndarray]:
    visual = np.asarray(visual, dtype=np.float32)
    language = np.asarray(language, dtype=np.float32)
    if visual.ndim != 3 or visual.shape[1:] != (64, 1024) or language.shape != (len(visual), 384):
        raise ValueError("CLIPort frozen feature shape mismatch")
    if not np.isfinite(visual).all() or not np.isfinite(language).all():
        raise ValueError("nonfinite frozen features")
    vm, vc, vs = _basis(visual.mean(axis=1), 12)
    lm, lc, ls = _basis(language, 12)
    return {"visual_mean": vm, "visual_components": vc, "visual_scale": vs, "language_mean": lm, "language_components": lc, "language_scale": ls}


def encode(visual: np.ndarray, language: np.ndarray, state4: np.ndarray, progress: np.ndarray, actions: np.ndarray, codec: dict[str, np.ndarray], *, direct_index: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visual = np.asarray(visual, dtype=np.float32)
    language = np.asarray(language, dtype=np.float32)
    state4 = np.asarray(state4, dtype=np.float32)
    progress = np.asarray(progress, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if visual.shape != (64, 1024) or language.shape != (384,) or state4.shape != (4,) or progress.shape not in ((), (1,)) or actions.ndim != 2 or actions.shape[1] != 20 or len(actions) < 1:
        raise ValueError("CLIPort input shape mismatch")
    if not 0 <= direct_index < len(actions) or not all(np.isfinite(x).all() for x in (visual, language, state4, progress, actions)):
        raise ValueError("invalid CLIPort input")
    global_visual = (((visual[None].mean(axis=1) - codec["visual_mean"]) @ codec["visual_components"].T) / codec["visual_scale"])[0]
    lang = (((language[None] - codec["language_mean"]) @ codec["language_components"].T) / codec["language_scale"])[0]
    row = np.clip(np.rint(actions[:, 14] * 7).astype(int), 0, 7)
    col = np.clip(np.rint(actions[:, 15] * 7).astype(int), 0, 7)
    local = visual[row * 8 + col]
    local_visual = ((local - codec["visual_mean"]) @ codec["visual_components"][:8].T) / codec["visual_scale"][:8]
    context = np.concatenate((global_visual, lang, state4, progress.reshape(1), actions[:, :3].mean(axis=0))).astype(np.float32)
    candidate = np.tanh(np.concatenate((actions, local_visual, actions[:, :4] - actions[direct_index:direct_index + 1, :4]), axis=-1)).astype(np.float32)
    if context.shape != (32,) or candidate.shape != (len(actions), 32):
        raise ValueError("CLIPort encoded shape mismatch")
    return context, candidate, np.zeros(len(actions), dtype=np.float32)


def preference_target(terminal_success: np.ndarray, *, direct_index: int = 0) -> np.ndarray:
    value = np.asarray(terminal_success, dtype=np.float32)
    if value.ndim != 1 or len(value) < 1 or not 0 <= direct_index < len(value) or not np.isin(value, (0, 1)).all():
        raise ValueError("invalid terminal outcomes")
    target = value.copy()
    if target.sum() == 0:
        target[direct_index] = 1
    return target / target.sum()


def make_policy(model, codec: dict[str, np.ndarray], *, propose=None, gain: float = 1.0, direct_margin: float = 0.0, device: str = "cpu") -> DiscreteAdapter | CandidatePolicy:
    adapter = DiscreteAdapter(model, lambda visual, language, state4, progress, actions, **kw: encode(visual, language, state4, progress, actions, codec, **kw), gain=gain, direct_margin=direct_margin, device=device)
    return CandidatePolicy(propose, adapter) if propose is not None else adapter

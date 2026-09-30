from __future__ import annotations

from pathlib import Path

import numpy as np

from .discrete import CandidatePolicy, DiscreteAdapter


SHAPES = {"context_mean": (2048,), "context_basis": (2048, 31), "context_scale": (31,), "context_basis_extra": (2048, 96), "context_scale_extra": (96,), "action_mean": (56,), "action_basis": (56, 32), "action_scale": (32,)}


def _randomized_pca(data: np.ndarray, components: int, *, seed: int, scale_floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) <= components or not np.isfinite(values).all():
        raise ValueError("invalid fitted PCA input")
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = values - mean
    count = min(components + 10, min(values.shape))
    omega = np.random.Generator(np.random.PCG64(seed)).standard_normal((values.shape[1], count), dtype=np.float32)
    q, _ = np.linalg.qr(centered @ omega, mode="reduced")
    for _ in range(4):
        z, _ = np.linalg.qr(centered.T @ q, mode="reduced")
        q, _ = np.linalg.qr(centered @ z, mode="reduced")
    _, _, basis_t = np.linalg.svd(q.T @ centered, full_matrices=False)
    basis = basis_t[:components].T.astype(np.float32)
    sign = np.sign(basis[np.abs(basis).argmax(axis=0), np.arange(components)])
    basis *= np.where(sign == 0, 1, sign)[None]
    projection = centered @ basis
    scale = np.maximum(projection.std(axis=0, dtype=np.float64), scale_floor).astype(np.float32)
    return mean, basis, scale


def fit_codec(prefix: np.ndarray, chunks: np.ndarray, *, seed: int = 260926, extra_seed: int = 260928) -> dict[str, np.ndarray]:
    prefix = np.asarray(prefix, dtype=np.float32)
    chunks = np.asarray(chunks, dtype=np.float32)
    if prefix.ndim != 2 or prefix.shape[1] != 2048 or chunks.ndim != 4 or chunks.shape[0] != len(prefix) or chunks.shape[2:] != (8, 7) or chunks.shape[1] < 1 or not np.isfinite(prefix).all() or not np.isfinite(chunks).all():
        raise ValueError("invalid LIBERO training features or action chunks")
    context_mean, context_basis, context_scale = _randomized_pca(prefix, 31, seed=seed, scale_floor=0.05)
    action_mean, action_basis, action_scale = _randomized_pca(chunks.reshape(-1, 56), 32, seed=seed + 1, scale_floor=0.02)
    centered = prefix - context_mean
    residual = centered - (centered @ context_basis) @ context_basis.T
    _, context_basis_extra, context_scale_extra = _randomized_pca(residual, 96, seed=extra_seed, scale_floor=0.05)
    codec = {"context_mean": context_mean, "context_basis": context_basis, "context_scale": context_scale, "context_basis_extra": context_basis_extra, "context_scale_extra": context_scale_extra, "action_mean": action_mean, "action_basis": action_basis, "action_scale": action_scale}
    if np.max(np.abs(context_basis.T @ context_basis_extra)) > 1e-3:
        raise ValueError("extra LIBERO basis overlaps first context basis")
    return codec


def load_codec(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        codec = {key: np.asarray(archive[key], dtype=np.float32).copy() for key in archive.files}
    if set(codec) != set(SHAPES) or any(codec[key].shape != shape or not np.isfinite(codec[key]).all() for key, shape in SHAPES.items()) or any(np.any(codec[key] <= 0) for key in ("context_scale", "context_scale_extra", "action_scale")):
        raise ValueError("invalid LIBERO fitted codec")
    return codec


def encode_context(prefix: np.ndarray, codec: dict[str, np.ndarray]) -> np.ndarray:
    prefix = np.asarray(prefix, dtype=np.float32)
    if prefix.shape != (2048,) or not np.isfinite(prefix).all():
        raise ValueError("invalid frozen prefix")
    centered = prefix - codec["context_mean"]
    first = np.zeros(32, dtype=np.float32)
    first[:31] = np.tanh((centered @ codec["context_basis"]) / codec["context_scale"])
    extra = np.tanh((centered @ codec["context_basis_extra"]) / codec["context_scale_extra"])
    return np.concatenate((first, extra)).astype(np.float32)


def encode_candidates(chunks: np.ndarray, codec: dict[str, np.ndarray]) -> np.ndarray:
    chunks = np.asarray(chunks, dtype=np.float32)
    if chunks.ndim != 3 or chunks.shape[1:] != (8, 7) or len(chunks) < 1 or not np.isfinite(chunks).all():
        raise ValueError("LIBERO candidates must be normalized action chunks [K,8,7]")
    flat = chunks.reshape(len(chunks), 56)
    return np.tanh(((flat - codec["action_mean"]) @ codec["action_basis"]) / codec["action_scale"]).astype(np.float32)


def encode(prefix: np.ndarray, chunks: np.ndarray, codec: dict[str, np.ndarray], *, direct_index: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = encode_candidates(chunks, codec)
    if not 0 <= direct_index < len(candidates):
        raise ValueError("invalid Direct index")
    return encode_context(prefix, codec), candidates, np.zeros(len(candidates), dtype=np.float32)


def demonstration_preference(distance: np.ndarray, *, tau: float) -> np.ndarray:
    distance = np.asarray(distance, dtype=np.float32)
    if distance.ndim != 1 or not len(distance) or not np.isfinite(distance).all() or not np.isfinite(tau) or tau <= 0:
        raise ValueError("invalid demonstration distances")
    weight = np.exp(-(distance - distance.min()) / tau)
    return weight / weight.sum()


def make_policy(model, codec: dict[str, np.ndarray], *, propose=None, gain: float = 0.25, direct_margin: float = 0.005, device: str = "cpu") -> DiscreteAdapter | CandidatePolicy:
    adapter = DiscreteAdapter(model, lambda prefix, chunks, **kw: encode(prefix, chunks, codec, **kw), gain=gain, direct_margin=direct_margin, device=device)
    return CandidatePolicy(propose, adapter) if propose is not None else adapter

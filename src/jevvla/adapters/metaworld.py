from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np


def fit_context_codec(prefix: np.ndarray, *, seed: int = 260980) -> dict[str, np.ndarray]:
    prefix = np.asarray(prefix)
    if prefix.ndim != 2 or prefix.shape[1] != 2048 or len(prefix) < 96 or not np.isfinite(prefix).all():
        raise ValueError("expected at least 96 finite frozen prefix vectors of width 2048")
    sample = prefix[np.linspace(0, len(prefix) - 1, min(4096, len(prefix)), dtype=np.int64)].astype(np.float32)
    mean = sample.mean(axis=0)
    centered = sample - mean
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((2048, 112)).astype(np.float32)
    q, _ = np.linalg.qr(centered @ omega, mode="reduced")
    q, _ = np.linalg.qr(centered @ (centered.T @ q), mode="reduced")
    _, _, vt = np.linalg.svd(q.T @ centered, full_matrices=False)
    basis = vt[:96].T.astype(np.float32)
    projected = centered @ basis
    scale = np.maximum(projected.std(axis=0), 1e-5).astype(np.float32)
    return {"mean": mean.astype(np.float32), "basis": basis, "scale": scale}


def load_context_codec(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        codec = {key: np.asarray(archive[key], np.float32) for key in ("mean", "basis", "scale")}
    if (codec["mean"].shape != (2048,) or codec["basis"].shape != (2048, 96)
            or codec["scale"].shape != (96,) or any(not np.isfinite(x).all() for x in codec.values())
            or np.any(codec["scale"] <= 0)):
        raise ValueError("invalid MetaWorld context codec")
    return codec


def context128(prefix, time, codec: Mapping[str, object]):
    import jax.numpy as jnp

    if prefix.ndim != 2 or prefix.shape[1] != 2048 or time.shape != (len(prefix),):
        raise ValueError("invalid prefix or Euler time shape")
    padded = jnp.pad(jnp.asarray(prefix, jnp.float32), ((0, 0), (0, 29)))
    groups = padded.reshape(len(prefix), 67, 31)
    signs = jnp.where(jnp.arange(67) % 2 == 0, 1.0, -1.0)
    counts = jnp.concatenate((jnp.full((2,), 67.0), jnp.full((29,), 66.0)))
    signed = jnp.tanh(jnp.sum(groups * signs[None, :, None], axis=1) / jnp.sqrt(counts[None]))
    extended = jnp.tanh((prefix - codec["mean"]) @ codec["basis"] / codec["scale"])
    return jnp.concatenate((signed, time[:, None], extended), axis=1)


def branch_mass(relative_return: np.ndarray, success: np.ndarray, alignment: np.ndarray,
                valid: np.ndarray, *, temperature: float) -> np.ndarray:
    values = [np.asarray(x) for x in (relative_return, success, alignment, valid)]
    if any(x.shape != values[0].shape for x in values) or values[0].ndim != 2:
        raise ValueError("branch arrays must share shape [anchors,candidates]")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    valid = np.asarray(valid, bool)
    utility = values[0] + 2.0 * values[1] - values[2] / temperature
    utility = np.where(valid, utility, -1e9)
    mass = np.exp(np.clip(utility - utility.max(axis=1, keepdims=True), -40, 0)) * valid
    return np.asarray(mass / np.maximum(mass.sum(axis=1, keepdims=True), 1e-12), np.float32)


def local_score_target(executed: np.ndarray, expert_residual: np.ndarray,
                       latent: np.ndarray, time: np.ndarray, *, expert_shift: float = 0.25) -> np.ndarray:
    executed = np.asarray(executed, np.float32)
    residual = np.asarray(expert_residual, np.float32)
    latent = np.asarray(latent, np.float32)
    time = np.asarray(time, np.float32)
    if executed.shape != residual.shape or executed.shape != latent.shape or time.shape != executed.shape[:-1]:
        raise ValueError("incompatible executed, residual, latent, and time shapes")
    sigma = 0.2 + 0.8 * time
    return np.asarray((executed + expert_shift * residual - latent) / sigma[..., None] ** 2, np.float32)


def prepare_gradient_data(context_encoded: np.ndarray, executed: np.ndarray,
                          expert_residual: np.ndarray, latent: np.ndarray,
                          time: np.ndarray, weights: np.ndarray, *,
                          expert_shift: float = 0.25, supervised_start: int = 4) -> dict[str, np.ndarray]:
    latent = np.asarray(latent, np.float32)
    if latent.ndim < 2 or latent.shape[-1] != 32:
        raise ValueError("latent actions must end in 32 dimensions")
    leading = latent.shape[:-1]
    context = np.asarray(context_encoded, np.float32)
    executed = np.asarray(executed, np.float32)
    residual = np.asarray(expert_residual, np.float32)
    time = np.asarray(time, np.float32)
    weights = np.asarray(weights, np.float32)
    if context.shape[-1:] != (128,) or executed.shape[-1:] != (32,) or residual.shape != executed.shape:
        raise ValueError("context or executed action geometry is invalid")
    if not 0 <= supervised_start < 32 or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("invalid supervised coordinates or weights")

    def broadcast(value, trailing):
        if value.shape == leading + trailing:
            return value
        if value.shape == (leading[0],) + trailing:
            value = value.reshape((leading[0],) + (1,) * (len(leading) - 1) + trailing)
        elif len(leading) == 3 and value.shape == (leading[0], leading[1]) + trailing:
            value = value.reshape((leading[0], leading[1], 1) + trailing)
        elif len(leading) == 3 and value.shape == (leading[0], leading[2]) + trailing:
            value = value.reshape((leading[0], 1, leading[2]) + trailing)
        elif len(leading) == 3 and value.shape == (leading[1],) + trailing:
            value = value.reshape((1, leading[1], 1) + trailing)
        try:
            return np.broadcast_to(value, leading + trailing)
        except ValueError as error:
            raise ValueError("input does not broadcast to latent geometry") from error

    context = broadcast(context, (128,))
    executed = broadcast(executed, (32,))
    residual = broadcast(residual, (32,))
    time = broadcast(time, ())
    weights = broadcast(weights, ())
    target = local_score_target(executed, residual, latent, time, expert_shift=expert_shift)
    coordinate_mask = np.arange(32) >= supervised_start
    return {"context": context.reshape(-1, 128), "action": latent.reshape(-1, 32),
            "gradient_target": target.reshape(-1, 32), "weight": weights.reshape(-1),
            "coordinate_mask": coordinate_mask}


def guided_sample_actions(base, observation, noise, state: Mapping[str, object], codec: Mapping[str, object],
                          *, gain: float = 0.08, component_limit: float = 0.12, steps: int = 10,
                          execution_horizon: int = 8, physical_dim: int = 4):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib
    from openpi.models import pi0

    from .energy_jax import action_score

    if noise.ndim != 3 or noise.shape[1:] != (base.action_horizon, base.action_dim):
        raise ValueError("noise does not match base policy action geometry")
    if execution_horizon * physical_dim != 32 or execution_horizon > base.action_horizon:
        raise ValueError("executed physical action must encode to 32 dimensions")
    if steps < 1 or not np.isfinite(gain) or gain < 0 or not np.isfinite(component_limit) or component_limit <= 0:
        raise ValueError("invalid guidance settings")
    observation = model_lib.preprocess_observation(None, observation, train=False)
    tokens, mask, ar_mask = base.embed_prefix(observation)
    positions = jnp.cumsum(mask, axis=1) - 1
    (hidden, _), cache = base.PaliGemma.llm(
        [tokens, None], mask=pi0.make_attn_mask(mask, ar_mask), positions=positions)
    weight = mask.astype(jnp.float32)
    prefix = jnp.sum(hidden.astype(jnp.float32) * weight[..., None], axis=1)
    prefix = jax.lax.stop_gradient(prefix / jnp.maximum(jnp.sum(weight, axis=1, keepdims=True), 1.0))
    cache = jax.lax.stop_gradient(cache)
    dt = -1.0 / steps

    def step(carry, _):
        action, time = carry
        times = jnp.broadcast_to(time, (len(action),))
        suffix, suffix_mask, suffix_ar_mask, condition = base.embed_suffix(observation, action, times)
        attention = jnp.concatenate((
            jnp.broadcast_to(mask[:, None, :], (len(action), suffix.shape[1], mask.shape[1])),
            pi0.make_attn_mask(suffix_mask, suffix_ar_mask)), axis=-1)
        suffix_positions = jnp.sum(mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=1) - 1
        (_, suffix_hidden), _ = base.PaliGemma.llm(
            [None, suffix], mask=attention, positions=suffix_positions,
            kv_cache=cache, adarms_cond=[None, condition])
        velocity = base.action_out_proj(suffix_hidden[:, -base.action_horizon:])
        context = context128(prefix, times, codec)
        physical = action[:, :execution_horizon, :physical_dim]
        score = action_score(state, context, physical.reshape(len(action), 32))
        correction = -jnp.clip(gain * score.reshape(len(action), execution_horizon, physical_dim),
                               -component_limit, component_limit)
        correction = correction.at[:, 0].set(0.0)
        velocity = velocity.at[:, :execution_horizon, :physical_dim].add(correction)
        return (action + dt * velocity, time + dt), None

    (sampled, _), _ = jax.lax.scan(step, (noise, jnp.asarray(1.0, jnp.float32)), None, length=steps)
    return sampled

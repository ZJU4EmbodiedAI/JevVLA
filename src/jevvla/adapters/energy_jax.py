from __future__ import annotations

from collections.abc import Mapping


def utility(state: Mapping[str, object], context, action):
    import jax
    import jax.numpy as jnp

    if context.ndim != 2 or context.shape[1] != 128 or action.shape != (len(context), 32):
        raise ValueError("expected [B,128] context and [B,32] action")

    def dense(x, name):
        return (jnp.matmul(x, state[name + ".weight"].T, precision=jax.lax.Precision.HIGHEST)
                + state[name + ".bias"])

    feature = 4.0 * jnp.tanh(action / 4.0)
    hidden = jax.nn.silu(dense(context, "context.0"))
    hidden = jax.nn.silu(dense(hidden, "context.2"))
    action_hidden = jax.nn.silu(dense(feature, "action.0"))
    residual = dense(jax.nn.silu(dense(jnp.concatenate((hidden, action_hidden), axis=1),
                                       "residual.0")), "residual.2")[:, 0]
    linear = dense(hidden, "linear")
    precision = jax.nn.softplus(dense(hidden, "precision"))
    rank_precision = jax.nn.softplus(dense(hidden, "low_rank_precision"))
    projection = jnp.matmul(feature, state["low_rank.weight"].T,
                            precision=jax.lax.Precision.HIGHEST)
    return (jnp.sum(linear * feature - 0.5 * precision * feature**2, axis=1)
            - 0.5 * jnp.sum(rank_precision * projection**2, axis=1) + residual)


def action_score(state: Mapping[str, object], context, action):
    import jax
    import jax.numpy as jnp

    return jax.grad(lambda native: jnp.sum(utility(state, context, native)))(action)

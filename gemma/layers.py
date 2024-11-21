"""Base layers."""

from flax import linen as nn
import jax
import jax.numpy as jnp


class Einsum(nn.Module):
    """Einsum is a convenience module for parameterized tensor multiplication."""
    shape: tuple[int, ...]

    @nn.compact
    def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
        w = self.param('w', nn.initializers.normal(), self.shape)
        return jnp.einsum(eqn, x, w)


class RMSNorm(nn.Module):
    """RMSNorm layer."""
    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', nn.initializers.zeros_init(), (x.shape[-1]))
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

        # Jax.lax.rsqrt is used because it returns different floats than
        # jnp.reciprocal(jnp.sqrt(var + 1e-06))
        normed_inputs = x * jax.lax.rsqrt(var + 1e-06)

        # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale is
        # a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
        # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
        scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))
        normed_inputs = normed_inputs * (1 + scale)
        return normed_inputs

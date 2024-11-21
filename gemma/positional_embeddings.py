"""Utils for positional embeddings (including RoPE)."""

import jax
import jax.numpy as jnp

_MAX_WAVELENGTH = 10_000


def add_positional_embedding(
    inputs: jax.Array,
    positions: jax.Array,
    max_wavelength: int = _MAX_WAVELENGTH,
) -> jax.Array:
    """Adds positional embeddings to inputs.

    Let B denote batch size, L denote sequence length, N denote number of heads,
    and H denote head dimension. Note that H must be divisible by 2.

    Args:
      inputs: Array of shape [B, L, N, H].
      positions:  Array of shape [B, L].
      max_wavelength: The maximum wavelength.

    Returns:
      Array of shape [B, L, N, H].
    """
    head_dim = inputs.shape[-1]
    num_timescales = head_dim // 2
    log_timescale_increment = jnp.log(float(max_wavelength)) / jnp.maximum(
        jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1
    )
    inv_timescales = jnp.exp(
        jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment
    )
    scaled_time = (
        positions[..., jnp.newaxis] * inv_timescales[jnp.newaxis, jnp.newaxis, :]
    )
    scaled_time = scaled_time[..., jnp.newaxis, :]
    signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=-1)
    position_embedding = signal.astype(jnp.float32)
    return inputs + position_embedding


def apply_rope(
    inputs: jax.Array,
    positions: jax.Array,
    max_wavelength: int = _MAX_WAVELENGTH,
) -> jax.Array:
    """Applies RoPE.

    Let B denote batch size, L denote sequence length, N denote number of heads,
    and H denote head dimension. Note that H must be divisible by 2.

    Args:
      inputs: Array of shape [B, L, N, H].
      positions:  Array of shape [B, L].
      max_wavelength: The maximum wavelength.

    Returns:
      Array of shape [B, L, N, H].
    """
    head_dim = inputs.shape[-1]
    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = max_wavelength**fraction

    sinusoid_inp = positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)

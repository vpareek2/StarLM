"""Inference code for LLAMA 3.2"""

import jax
import jax.numpy as jnp
import math
import tiktoken

from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ModelParams:
    n_layers: int
    n_local_heads: int
    n_local_kv_heads: int
    head_dim: int
    max_seq_len: int
    rope_theta: float
    use_scaled_rope: bool

@dataclass
class LayerWeights:
    wq: jax.Array
    wk: jax.Array
    wv: jax.Array
    wo: jax.Array
    w1: jax.Array
    w2: jax.Array
    w3: jax.Array
    ffn_norm: jax.Array
    attention_norm: jax.Array

@dataclass
class TransformerWeights:
    tok_embeddings: jax.Array
    norm: jax.Array
    output: jax.Array
    layer_weights: List[LayerWeights]

params = {
    "dim": 2048,
    "n_layers": 16,
    "n_heads": 32,
    "n_kv_heads": 8,
    "vocab_size": 128256,
    "ffn_dim_multiplier": 1.5,
    "multiple_of": 256,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "use_scaled_rope": True,
    "max_seq_len": 4096
}

LLAMA_1B_PARAMS = ModelParams(
    n_layers=params["n_layers"],
    n_local_heads=params["n_heads"],
    n_local_kv_heads=params["n_kv_heads"],
    head_dim=params["dim"] // params["n_heads"],
    max_seq_len=params["max_seq_len"],
    rope_theta=params["rope_theta"],
    use_scaled_rope=params["use_scaled_rope"]
)

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)

def load_weights(ckpt_dir: Path, n_layers: int = 16):
    """Load model weights from files."""
    w = {}
    layer_weights = []
    try:
        device = jax.devices("gpu")[0]
    except RuntimeError:
        print("GPU not found. Using CPU instead.")
        device = jax.devices("cpu")[0]
    for file in ckpt_dir.glob("*.npy"):
        name = '.'.join(str(file).split('/')[-1].split('.')[:-1])
        weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
        w[name] = jax.device_put(weight, device)
    for i in range(n_layers):
        layer_weights.append(LayerWeights(
            wq=w[f'layers.{i}.attention.wq.weight'],
            wk=w[f'layers.{i}.attention.wk.weight'],
            wv=w[f'layers.{i}.attention.wv.weight'],
            wo=w[f'layers.{i}.attention.wo.weight'],
            w1=w[f'layers.{i}.feed_forward.w1.weight'],
            w2=w[f'layers.{i}.feed_forward.w2.weight'],
            w3=w[f'layers.{i}.feed_forward.w3.weight'],
            ffn_norm=w[f'layers.{i}.ffn_norm.weight'],
            attention_norm=w[f'layers.{i}.attention_norm.weight'],
        ))

    transformer_weights = TransformerWeights(
        tok_embeddings=w['tok_embeddings.weight'],
        norm=w['norm.weight'],
        output=w['output.weight'],
        layer_weights=layer_weights
    )

    return transformer_weights

class KVCache(NamedTuple):
    k: jax.Array
    v: jax.Array

    @classmethod
    def new(cls, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int) -> 'KVCache':
        """Initialize a KV cache."""
        return cls(
            k=jnp.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16),
            v=jnp.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16)
        )

    def update(self, xk: jax.Array, xv: jax.Array, layer_idx: int, cur_pos: int, n_rep: int):
        """Updates KVCache with new keys and values, handling special case for first position."""
        ck = jax.lax.dynamic_update_slice(self.k, jnp.bfloat16(xk[None, ...]), (layer_idx, 0, cur_pos, 0, 0))
        cv = jax.lax.dynamic_update_slice(self.v, jnp.bfloat16(xv[None, ...]), (layer_idx, 0, cur_pos, 0, 0))
        if cur_pos == 0:
            keys = jnp.repeat(xk, n_rep, axis=2)
            values = jnp.repeat(xv, n_rep, axis=2)
        else:
            keys = jnp.repeat(ck[layer_idx], n_rep, axis=2)
            values = jnp.repeat(cv[layer_idx], n_rep, axis=2)

        return keys, values, KVCache(k=ck, v=cv)

# @partial(jax.jit, static_argnames=("eps"))
def rms_norm(
    x: jax.Array,
    w: jax.Array,
    eps: float = 1e-6
) -> jax.Array:
    """Root mean square normalization."""
    return w * (x * jax.lax.rsqrt(jax.lax.pow(x, 2).mean(-1, keepdims=True) + eps))

# @partial(jax.jit, static_argnames=("dtype"))
def rope(
    xq: jax.Array,
    xk: jax.Array,
    freqs_cis: jax.Array,
    dtype: jnp.dtype = jnp.float32
) -> Tuple[jax.Array, jax.Array]:
    """Rotary Embeddings"""
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)
    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    xq_out = xq_ * freqs_cis[None, :, None, :]
    xk_out = xk_ * freqs_cis[None, :, None, :]
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(dtype), xk_out.astype(dtype)

# @partial(jax.jit, static_argnames=("model_params", "cur_pos", "layer_idx"))
def attention(
    x: jax.Array,
    layer_weights: LayerWeights,
    model_params: ModelParams,
    cur_pos: int,
    layer_idx: int,
    freqs_cis: jax.Array,
    kvcache: KVCache,
    attn_mask: Optional[jax.Array] = None
) -> Tuple[jax.Array, KVCache]:
    """Grouped Query Attention"""
    bsz, _, _ = x.shape
    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
    xq = jnp.einsum('...e,eh->...h', x, layer_weights.wq)
    xq = xq.reshape(bsz, -1, model_params.n_local_heads, model_params.head_dim)
    xk = jnp.einsum('...e,eh->...h', x, layer_weights.wk)
    xk = xk.reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xv = jnp.einsum('...e,eh->...h', x, layer_weights.wv)
    xv = xv.reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xq, xk = jax.vmap(lambda q, k: rope(q, k, freqs_cis=freqs_cis))(xq, xk)
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
    xq = jnp.transpose(xq, (0, 2, 1, 3))
    keys = jnp.transpose(keys, (0, 2, 3, 1))
    values = jnp.transpose(values, (0, 2, 1, 3))
    scores = jnp.einsum('bhsd,bhdk->bhsk', xq, keys) / jnp.sqrt(model_params.head_dim)
    scores = scores.astype(jnp.float32)
    if attn_mask is not None:
        scores = scores.at[..., :attn_mask.shape[-1]].add(attn_mask)
    mask = jnp.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
    padded_logits = jnp.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)
    scores = jax.nn.softmax(padded_logits, axis=-1).astype(x.dtype)
    output = jnp.einsum('bhsk,bhkd->bhsd', scores, values)
    output = jnp.transpose(output, (0, 2, 1, 3)).reshape(xq.shape[0], xq.shape[2], -1)
    out = jnp.einsum('...h,ho->...o', output, layer_weights.wo)

    return out, kvcache

#@partial(jax.jit)
def ffn(
    x: jax.Array,
    layer_weights: LayerWeights
) -> jax.Array:
    """Feed Forward Network"""
    return jnp.dot(jax.nn.silu(jnp.dot(x, layer_weights.w1.T)) * jnp.dot(x, layer_weights.w3.T), layer_weights.w2.T)

#@partial(jax.jit, static_argnames=("model_params", "cur_pos"))
def transformer(
    weights: TransformerWeights,
    model_params: ModelParams,
    tokens: jax.Array,
    cur_pos: int,
    freqs_cis: jax.Array,
    kvcache: KVCache,
    attn_mask: Optional[jax.Array] = None
) -> Tuple[jax.Array, KVCache]:
    """Forward pass through the transformer."""
    x = weights.tok_embeddings[tokens]
    for i in range(model_params.n_layers):
        norm_x = rms_norm(x, weights.layer_weights[i].attention_norm)
        attn, kvcache = attention(norm_x, weights.layer_weights[i], model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
        x = x + attn
        x = x + ffn(rms_norm(x, weights.layer_weights[i].ffn_norm), weights.layer_weights[i])
    logits = jnp.dot(rms_norm(x, weights.norm), weights.output.T)

    return logits, kvcache

def apply_scaling(freqs: jax.Array):
    """Applies dynamic frequency scaling based on wavelength ranges."""
    SCALE_FACTOR = 8
    LOW_FREQ_FACTOR = 1
    HIGH_FREQ_FACTOR = 4
    OLD_CONTEXT_LEN = 8192

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq):
        wavelen = 2 * math.pi / freq

        def scale_mid(_):
            smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
            return (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        return jax.lax.cond(
            wavelen < high_freq_wavelen,
            lambda _: freq,
            lambda _: jax.lax.cond(wavelen > low_freq_wavelen, lambda _: freq / SCALE_FACTOR, scale_mid, None),
            None
        )

    return jax.vmap(scale_freq)(freqs)


def build_attn_mask(seqlen: int, start_pos: int) -> jax.Array:
    """Generate the attention mask for a sequence of length."""
    mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
    if seqlen > 1:
        mask = jnp.full((seqlen, seqlen), float('-inf'))
        mask = jnp.triu(mask, k=1)
        mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)

    return mask

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    """Precompute freqs_cis for rotary embeddings."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)
    t = jnp.arange(end, dtype=dtype)
    freqs = jnp.outer(t, freqs)
    return jnp.exp(1j * freqs)

def main(weights_path: Path = Path('1B-Instruct-weights')):
    """Main execution block."""
    model_params = LLAMA_1B_PARAMS
    weights = load_weights(weights_path, model_params.n_layers)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    def generate(weights, model_params, tokens):
        """Generate text using supplied tokens."""
        gen_tokens = None
        cur_pos = 0
        tokens = jnp.array([tokens], jnp.int32)
        bsz, seqlen = tokens.shape
        attn_mask = build_attn_mask(seqlen, cur_pos)

        freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)

        kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)

        logits, kvcache = transformer(weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)

        next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
        gen_tokens = next_token
        print(tokenizer.decode([int(next_token.item())]), end='', flush=True)
        cur_pos = seqlen

        stop = jnp.array([128001, 128008, 128009])

        while cur_pos < 8192:
            cur_pos += 1
            logits, kvcache = transformer( weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
            next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
            gen_tokens = jnp.concatenate((gen_tokens, next_token))
            out_token = tokenizer.decode([int(next_token.item())])
            print(out_token, end='', flush=True)

            if jnp.isin(next_token, stop).any():
                break

        return cur_pos

    prompt = """

    """
    print(prompt)
    tokens = tokenizer.encode(prompt)

    gen_tokens = generate(weights, model_params, tokens)

if __name__ == "__main__":
    ckpt = Path("./1B-Instruct-weights")
    weights = load_weights(ckpt, ModelParams.n_layers)
    print("Transformer weights loaded successfully!")
    print(f"Token embeddings shape: {weights.tok_embeddings.shape}")
    print(f"Number of layers: {len(weights.layer_weights)}")
    main()

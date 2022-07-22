import jax
from jax import nn
from jax import jit, numpy as jnp

MASK_VALUE = -1e10

@jit
def attention(q, k, v, key_mask):
    dim, k_len = q.shape[-1], k.shape[-2]
    scale = 1 / jnp.sqrt(dim)

    q = q * scale
    sim = q @ k.transpose()

    sim = jnp.where(key_mask, sim, MASK_VALUE)

    attn = nn.softmax(sim, axis = -1)
    return attn @ v

@jit
def causal_attention(q, k, v):
    q_len, dim, k_len = *q.shape, k.shape[-2]
    scale = 1 / jnp.sqrt(dim)

    q = q * scale
    sim = q @ k.transpose()

    causal_mask = jnp.triu(jnp.ones((q_len, k_len)), k_len - q_len + 1)
    sim = jnp.where(causal_mask, MASK_VALUE, sim)

    attn = nn.softmax(sim, axis = -1)
    return attn @ v

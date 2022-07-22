import jax
from jax import nn
from jax import jit, numpy as jnp

@jit
def attention(q, k, v):
    dim = q.shape[-1]
    q = q / jnp.sqrt(dim)
    sim = q @ k.transpose()
    attn = nn.softmax(sim, axis = -1)
    return attn @ v

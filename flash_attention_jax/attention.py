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

# cosine sim attention

@jit
def l2norm(t, eps = 1e-6):
    norm = jnp.linalg.norm(t)
    return t / (norm + eps)

@jit
def cosine_sim_attention(q, k, v, scale = 16):
    q, k = map(l2norm, (q, k))
    sim = q @ k.transpose()
    sim = sim * scale
    attn = nn.softmax(sim, axis = -1)
    return attn @ v

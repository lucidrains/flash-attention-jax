import jax
from jax import nn
from jax import custom_vjp
from jax import numpy as jnp, lax

@custom_vjp
def flash_attention(q, k, v):
    sim = q @ k.transpose()
    sim = sim / jnp.sqrt(q.shape[-1])
    attn = nn.softmax(sim, axis = -1)
    out = attn @ v
    return out, q, k, v

def flash_attention_forward(q, k, v):
    out, q, k, v = flash_attention(q, k, v)
    return out, (q, k, v)

def flash_attention_backward(res, g):
    (q, k, v) = res
    return q, k, v

flash_attention.defvjp(flash_attention_forward, flash_attention_backward)

# flash cosine sim attention

def l2norm(t, eps = 1e-6):
    norm = jnp.linalg.norm(t)
    return t / (norm + eps)

@custom_vjp
def flash_cosine_sim_attention(q, k, v, scale = 16):
    q, k = map(l2norm, (q, k))

    sim = q @ k.transpose()
    sim = sim * scale
    attn = nn.softmax(sim, axis = -1)
    out = attn @ v
    return out, q, k, v

def flash_cosine_sim_attention_forward(q, k, v, scale = 16):
    out, q, k, v = flash_cosine_sim_attention(q, k, v, scale = scale)
    return out, (q, k, v)

def flash_cosine_sim_attention_backward(res, g):
    q, k, v = res
    return q, k, v, None

flash_cosine_sim_attention.defvjp(flash_cosine_sim_attention_forward, flash_cosine_sim_attention_backward)

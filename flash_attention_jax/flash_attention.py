import math
import jax
from functools import partial
from jax import nn
from jax import custom_vjp
from jax import numpy as jnp, lax, jit, grad

Q_CHUNK_SIZE = 1024

def _query_chunk_flash_attention(q, k, v):
    q_len, k_len, dim, v_dim = q.shape[-2], *k.shape, v.shape[-1]

    sim = q @ k.transpose()
    sim = sim / jnp.sqrt(dim)
    attn = nn.softmax(sim, axis = -1)
    out = attn @ v
    return out

@custom_vjp
def flash_attention(q, k, v):
    q_len, dim, v_dim = *q.shape, v.shape[-1]

    def chunk_scanner(chunk_idx, _):
        q_chunk = lax.dynamic_slice(q, (chunk_idx, 0), slice_sizes = (min(Q_CHUNK_SIZE, q_len), dim))
        return (chunk_idx + Q_CHUNK_SIZE, _query_chunk_flash_attention(q_chunk, k, v))

    _, res = lax.scan(chunk_scanner, init = 0, xs = None, length = math.ceil(q_len / Q_CHUNK_SIZE))
    out = res.reshape(q_len, v_dim)
    return out, q, k, v

@jit
def flash_attention_forward(q, k, v):
    out, q, k, v = flash_attention(q, k, v)
    return out, (q, k, v)

@jit
def flash_attention_backward(res, g):
    q, k, v = res

    dim = q.shape[-1]
    scale = 1 / jnp.sqrt(dim)

    sim = q @ k.transpose()
    sim = sim * scale
    attn = nn.softmax(sim, axis = -1)

    dv = attn.transpose() @ g
    dp = g @ v.transpose()

    dxhat = dp * attn
    ds = dxhat - attn * jnp.sum(dxhat, axis = 1, keepdims = True)
    ds = ds * scale

    dq = ds @ k
    dk = ds.transpose() @ q

    return dq, dk, dv

flash_attention.defvjp(flash_attention_forward, flash_attention_backward)

# flash cosine sim attention

COSINE_SIM_SCALE = 16

@jit
def l2norm(t, eps = 1e-6):
    norm = jnp.linalg.norm(t)
    return t / (norm + eps)

def _query_chunk_flash_cosine_sim_attention(q, k, v):
    q_len, k_len, dim, v_dim = q.shape[-2], *k.shape, v.shape[-1]

    sim = q @ k.transpose()
    sim = sim * COSINE_SIM_SCALE
    attn = nn.softmax(sim, axis = -1)
    out = attn @ v
    return out

def flash_cosine_sim_attention(q, k, v):
    q, k = map(l2norm, (q, k))
    return flash_cosine_sim_attention_post_l2norm(q, k, v)

@custom_vjp
def flash_cosine_sim_attention_post_l2norm(q, k, v):
    q_len, dim, v_dim = *q.shape, v.shape[-1]

    def chunk_scanner(chunk_idx, _):
        q_chunk = lax.dynamic_slice(q, (chunk_idx, 0), slice_sizes = (min(Q_CHUNK_SIZE, q_len), dim))
        return (chunk_idx + Q_CHUNK_SIZE, _query_chunk_flash_cosine_sim_attention(q_chunk, k, v))

    _, res = lax.scan(chunk_scanner, init = 0, xs = None, length = math.ceil(q_len / Q_CHUNK_SIZE))
    out = res.reshape(q_len, v_dim)
    return out, q, k, v

@jit
def flash_cosine_sim_attention_forward(q, k, v):
    out, q, k, v = flash_cosine_sim_attention(q, k, v)
    return out, (q, k, v)

@jit
def flash_cosine_sim_attention_backward(res, g):
    q, k, v = res

    sim = q @ k.transpose()
    sim = sim * COSINE_SIM_SCALE
    attn = nn.softmax(sim, axis = -1)

    dv = attn.transpose() @ g
    dp = g @ v.transpose()

    dxhat = dp * attn
    ds = dxhat - attn * jnp.sum(dxhat, axis = 1, keepdims = True)
    ds = ds * COSINE_SIM_SCALE

    dq = ds @ k
    dk = ds.transpose() @ q

    return dq, dk, dv

flash_cosine_sim_attention_post_l2norm.defvjp(flash_cosine_sim_attention_forward, flash_cosine_sim_attention_backward)

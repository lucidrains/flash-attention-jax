import math
import jax
from functools import partial
from jax import nn
from jax import custom_vjp
from jax import numpy as jnp, lax, jit, grad

# constants

Q_CHUNK_SIZE = 1024
K_CHUNK_SIZE = 2048

# flash attention

def _query_chunk_flash_attention(q, k, v):
    q_len, k_len, dim, v_dim = q.shape[-2], *k.shape, v.shape[-1]
    scale = 1 / jnp.sqrt(dim)

    def chunk_scanner(carries, _):
        chunk_idx, out, row_sum, row_max = carries
        k_chunk = lax.dynamic_slice(k, (chunk_idx, 0), slice_sizes=(K_CHUNK_SIZE, dim))
        v_chunk = lax.dynamic_slice(v, (chunk_idx, 0), slice_sizes=(K_CHUNK_SIZE, v_dim))

        attn_weights = (q @ k.transpose()) * scale

        block_row_max = jnp.max(attn_weights, axis = -1, keepdims = True)
        exp_weights = jnp.exp(attn_weights - block_row_max)
        block_row_sum = jnp.sum(exp_weights, axis = -1, keepdims = True)

        exp_values = exp_weights @ v

        new_row_max = jnp.maximum(block_row_max, row_max)

        exp_row_max_diff = jnp.exp(row_max - new_row_max)
        exp_block_row_max_diff = jnp.exp(block_row_max - new_row_max)

        new_row_sum = exp_row_max_diff * row_sum + exp_block_row_max_diff * block_row_sum

        out = (row_sum / new_row_sum) * exp_row_max_diff * out + \
              (exp_block_row_max_diff / new_row_sum) * exp_values

        return (chunk_idx + K_CHUNK_SIZE, out, new_row_sum, new_row_max), None

    out = jnp.zeros((q_len, dim))
    row_sum = jnp.zeros((q_len, 1))
    row_max = jnp.ones((q_len, 1)) * -1e6

    (_, out, row_sum, row_max), _ = lax.scan(chunk_scanner, init = (0, out, row_sum, row_max), xs = None, length = math.ceil(k_len / K_CHUNK_SIZE))
    return out.reshape(q_len, v_dim), row_sum.reshape(q_len), row_max.reshape(q_len)

@custom_vjp
def flash_attention(q, k, v):
    q_len, dim, v_dim = *q.shape, v.shape[-1]

    def chunk_scanner(chunk_idx, _):
        q_chunk = lax.dynamic_slice(q, (chunk_idx, 0), slice_sizes = (min(Q_CHUNK_SIZE, q_len), dim))
        return (chunk_idx + Q_CHUNK_SIZE, _query_chunk_flash_attention(q_chunk, k, v))

    _, (out, row_sum, row_max) = lax.scan(chunk_scanner, init = 0, xs = None, length = math.ceil(q_len / Q_CHUNK_SIZE))

    out = out.reshape(q_len, v_dim)
    row_sum = row_sum.reshape(q_len)
    row_max = row_max.reshape(q_len)

    return out, q, k, v, row_sum, row_max

@jit
def flash_attention_forward(q, k, v):
    out, q, k, v, row_sum, row_max = flash_attention(q, k, v)
    return out, (q, k, v, out, row_sum, row_max)

@jit
def flash_attention_backward(res, do):
    q, k, v, o, l, m = res

    dim = q.shape[-1]
    scale = 1 / jnp.sqrt(dim)

    sim = q @ k.transpose()
    sim = sim * scale
    p = nn.softmax(sim, axis = -1)

    dv = p.transpose() @ do
    dp = do @ v.transpose()

    D = jnp.sum(do * o, axis = -1, keepdims = True)
    ds = p * (dp - D)
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

    def chunk_scanner(carries, _):
        chunk_idx, out, row_sum = carries
        k_chunk = lax.dynamic_slice(k, (chunk_idx, 0), slice_sizes=(K_CHUNK_SIZE, dim))
        v_chunk = lax.dynamic_slice(v, (chunk_idx, 0), slice_sizes=(K_CHUNK_SIZE, v_dim))

        attn_weights = (q @ k.transpose()) * COSINE_SIM_SCALE
        exp_weights = jnp.exp(attn_weights - COSINE_SIM_SCALE)
        block_row_sum = jnp.sum(exp_weights, axis = -1, keepdims = True)

        exp_values = exp_weights @ v

        new_row_sum = row_sum + block_row_sum

        out = out + (exp_values / k_len)

        return (chunk_idx + K_CHUNK_SIZE, out, new_row_sum), None

    out = jnp.zeros((q_len, dim))
    row_sum = jnp.zeros((q_len, 1))

    (_, out, row_sum), _ = lax.scan(chunk_scanner, init = (0, out, row_sum), xs = None, length = math.ceil(k_len / K_CHUNK_SIZE))

    out = out * (k_len / row_sum)
    return out.reshape(q_len, v_dim), row_sum

def flash_cosine_sim_attention(q, k, v):
    q, k = map(l2norm, (q, k))
    return flash_cosine_sim_attention_post_l2norm(q, k, v)

@custom_vjp
def flash_cosine_sim_attention_post_l2norm(q, k, v):
    q_len, dim, v_dim = *q.shape, v.shape[-1]

    def chunk_scanner(chunk_idx, _):
        q_chunk = lax.dynamic_slice(q, (chunk_idx, 0), slice_sizes = (min(Q_CHUNK_SIZE, q_len), dim))
        return (chunk_idx + Q_CHUNK_SIZE, _query_chunk_flash_cosine_sim_attention(q_chunk, k, v))

    _, (out, row_sum) = lax.scan(chunk_scanner, init = 0, xs = None, length = math.ceil(q_len / Q_CHUNK_SIZE))

    out = out.reshape(q_len, v_dim)
    row_sum = row_sum.reshape(q_len)

    return out, q, k, v, row_sum

@jit
def flash_cosine_sim_attention_forward(q, k, v):
    out, q, k, v, row_sum = flash_cosine_sim_attention(q, k, v)
    return out, (q, k, v, out, row_sum)

@jit
def flash_cosine_sim_attention_backward(res, do):
    q, k, v, o, l = res

    sim = q @ k.transpose()
    sim = sim * COSINE_SIM_SCALE
    p = nn.softmax(sim, axis = -1)

    dv = p.transpose() @ do
    dp = do @ v.transpose()

    ds = p * (dp - jnp.sum(do * o, axis = -1, keepdims = True))
    ds = ds * COSINE_SIM_SCALE

    dq = ds @ k
    dk = ds.transpose() @ q

    return dq, dk, dv

flash_cosine_sim_attention_post_l2norm.defvjp(flash_cosine_sim_attention_forward, flash_cosine_sim_attention_backward)

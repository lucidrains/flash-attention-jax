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

def _query_chunk_flash_attention_backward(q, k, v, o, do, l, m):
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

@jit
def flash_attention_backward(res, do):
    q, k, v, o, l, m = res

    q_len, dim = q.shape

    dk = jnp.zeros_like(k)
    dv = jnp.zeros_like(v)

    def chunk_scanner(carries, _):
        chunk_idx, dk, dv = carries

        q_chunk = lax.dynamic_slice(q, (chunk_idx, 0), slice_sizes = (min(Q_CHUNK_SIZE, q_len), q.shape[-1]))
        o_chunk = lax.dynamic_slice(o, (chunk_idx, 0), slice_sizes = (min(Q_CHUNK_SIZE, q_len), o.shape[-1]))
        do_chunk = lax.dynamic_slice(do, (chunk_idx, 0), slice_sizes = (min(Q_CHUNK_SIZE, q_len), do.shape[-1]))

        dq_chunk, dk_chunk, dv_chunk = _query_chunk_flash_attention_backward(q_chunk, k, v, o_chunk, do_chunk, l, m)
        return (chunk_idx + Q_CHUNK_SIZE, dk + dk_chunk, dv + dv_chunk), dq_chunk

    (_, dk, dv), dq = lax.scan(chunk_scanner, init = (0, dk, dv), xs = None, length = math.ceil(q_len / Q_CHUNK_SIZE))

    dq = dq.reshape(q_len, dim)
    return dq, dk, dv

flash_attention.defvjp(flash_attention_forward, flash_attention_backward)

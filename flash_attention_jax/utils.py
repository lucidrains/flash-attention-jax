import jax
from functools import partial
import jax.numpy as jnp
from jax import random
from jax import value_and_grad

def value_and_grad_wrapper(fn, **kwargs):
    @partial(value_and_grad, **kwargs)
    def inner(*args, **kwargs):
        return jnp.sum(fn(*args, **kwargs))
    return inner

def diff(t1, t2):
    return jnp.max(jnp.abs(t1 - t2))

def value_and_grad_difference(
    fn1,
    fn2,
    seed = 42,
    q_seq_len = 1024,
    k_seq_len = 8192,
    dim = 512
):
    key = random.PRNGKey(seed)
    key1, key = random.split(key)
    key2, key = random.split(key)
    key3, key = random.split(key)

    q = random.normal(key1, (q_seq_len, dim))
    k = random.normal(key2, (k_seq_len, dim))
    v = random.normal(key3, (k_seq_len, dim))

    fn1_value_and_grad, fn2_value_and_grad = map(partial(value_and_grad_wrapper, argnums = (0, 1, 2)), (fn1, fn2))

    o1, grads1 = fn1_value_and_grad(q, k, v)
    o2, grads2 = fn2_value_and_grad(q, k, v)

    return diff(o1, o2), [diff(*args) for args in zip(grads1, grads2)]

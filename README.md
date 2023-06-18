<img src="./flash-attention.png" width="450px"></img>

## Flash Attention - Jax

Implementation of <a href="https://arxiv.org/abs/2205.14135">Flash Attention</a> in Jax. It will likely not be as performant as with the <a href="https://github.com/HazyResearch/flash-attention">official CUDA version</a>, given lack of ability for fine memory management. But just for educational purposes as well as to see how clever XLA compiler is (or is not).

## Install

```bash
$ pip install flash-attention-jax
```

## Usage

```python
from jax import random
from flash_attention_jax import flash_attention

rng_key = random.PRNGKey(42)

q = random.normal(rng_key, (1, 2, 131072, 512))  # (batch, heads, seq, dim)
k = random.normal(rng_key, (1, 2, 131072, 512))
v = random.normal(rng_key, (1, 2, 131072, 512))
mask = random.randint(rng_key, (1, 131072,), 0, 2) # (batch, seq)

out, _ = flash_attention(q, k, v, mask)

out.shape  # (1, 2, 131072, 512) - (batch, heads, seq, dim)
```

Quick sanity check


```python
from flash_attention_jax import plain_attention, flash_attention, value_and_grad_difference

diff, (dq_diff, dk_diff, dv_diff) = value_and_grad_difference(
    plain_attention,
    flash_attention,
    seed = 42
)

print('shows differences between normal and flash attention for output, dq, dk, dv')
print(f'o: {diff}')       # < 1e-4
print(f'dq: {dq_diff}')   # < 1e-6
print(f'dk: {dk_diff}')   # < 1e-6
print(f'dv: {dv_diff}')   # < 1e-6
```

Autoregressive Flash Attention - GPT-like decoder attention

```python
from jax import random
from flash_attention_jax import causal_flash_attention

rng_key = random.PRNGKey(42)

q = random.normal(rng_key, (131072, 512))
k = random.normal(rng_key, (131072, 512))
v = random.normal(rng_key, (131072, 512))

out, _ = causal_flash_attention(q, k, v)

out.shape  # (131072, 512)
```

## Todo

- [x] leading dimensions for causal flash attention variant

- [ ] figure out issue with jit and static argnums
- [ ] comment with references to paper algorithms and explanations
- [ ] make sure it can work one-headed key / values, as in PaLM

## Citations

```bibtex
@article{Dao2022FlashAttentionFA,
    title   = {FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
    author  = {Tri Dao and Daniel Y. Fu and Stefano Ermon and Atri Rudra and Christopher R'e},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2205.14135}
}
```

```bibtex
@article{Rabe2021SelfattentionDN,
    title   = {Self-attention Does Not Need O(n2) Memory},
    author  = {Markus N. Rabe and Charles Staats},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2112.05682}
}
```

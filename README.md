<img src="./flash-attention.png" width="450px"></img>

## Flash Attention - Jax (wip)

Implementation of <a href="https://arxiv.org/abs/2205.14135">Flash Attention</a> in Jax. It will likely not be as performant as with the <a href="https://github.com/HazyResearch/flash-attention">official CUDA version</a>, given lack of ability for fine memory management. But just for educational purposes as well as to see how clever XLA compiler is (or is not).

## Install

```bash
$ pip install flash-attention-jax
```

## Usage

```python
from jax import random
from flash_attention_jax import flash_attention

key = random.PRNGKey(42)
q = random.normal(key, (131072, 512))
k = random.normal(key, (131072, 512))
v = random.normal(key, (131072, 512))

out, _ = flash_attention(q, k, v)

out.shape  # (131072, 512)
```

## Using in Pytorch

You'll have to have both Jax and Pytorch installed

First install `jax2torch`

```bash
$ pip install jax2torch
```

Then

```python
import torch
from jax2torch import jax2torch
from flash_attention_jax import flash_attention

q = torch.randn(131072, 512).cuda()
k = torch.randn(131072, 512).cuda()
v = torch.randn(131072, 512).cuda()

torch_flash_attention = jax2torch(flash_attention)

out = torch_flash_attention(q, k, v)

out.shape # (131072, 512)
```

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

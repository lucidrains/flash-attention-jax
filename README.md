<img src="./flash-attention.png" width="450px"></img>

## Flash Attention - Jax (wip)

Implementation of <a href="https://arxiv.org/abs/2205.14135">Flash Attention</a> in Jax. It will likely not be as performant as with the <a href="https://github.com/HazyResearch/flash-attention">official CUDA version</a>, given lack of ability for fine memory management. But just for educational purposes as well as to see how clever XLA compiler is (or is not).

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

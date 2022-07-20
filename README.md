## Flash Attention - Jax (wip)

Implementation of <a href="https://arxiv.org/abs/2205.14135">Flash Attention</a> in Jax. It will likely not be as performant as with CUDA, given lack of ability for fine memory management. But just for educational purposes as well as to see how clever XLA compiler is (or is not).

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

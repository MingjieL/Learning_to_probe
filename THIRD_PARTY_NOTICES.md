# Third-party acknowledgements

## Rotlex

Learning to Probe's method and implementation draw on code from
[ustcllm/rotlex](https://github.com/ustcllm/rotlex). We acknowledge the original
authors' work on learned vector indexing and query-aware routing.
This release preserves the original upstream copyright notice,
`Copyright 2025 weiwch`, and its MIT license verbatim at
[third_party/rotlex/LICENSE](third_party/rotlex/LICENSE).
Learning to Probe additions are covered by the root [MIT license](LICENSE).

Upstream license source: https://github.com/ustcllm/rotlex/blob/main/LICENSE
(retrieved 2026-10-27).

Please also acknowledge the Rotlex paper when using the underlying work:

Wenqing Wei, Defu Lian, Qingshuai Feng, and Yongji Wu.
*Robust Tree-based Learned Vector Index with Query-aware Repartitioning*.
KDD 2025, pp. 3134–3143. https://doi.org/10.1145/3711896.3737112

```bibtex
@inproceedings{wei2025rotlex,
  author = {Wei, Wenqing and Lian, Defu and Feng, Qingshuai and Wu, Yongji},
  title = {Robust Tree-based Learned Vector Index with Query-aware Repartitioning},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  year = {2025},
  pages = {3134--3143},
  doi = {10.1145/3711896.3737112}
}
```

## Runtime dependencies

PyTorch, FAISS, NumPy, Pydantic, and optional testing/plotting packages are
external dependencies with their respective licenses. Their source trees and
binaries are not copied into this source release. The project's C++ extension
is rebuilt against the user's installed PyTorch.

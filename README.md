# Learning to Probe

Learned routing and adaptive probing for approximate nearest-neighbor vector search.
This release contains RAMPS router training, fixed-partition IVF retrieval,
Probability Gain-Cost prefix queries (L2PR), positive-marginal queries (L2PR+),
and the source for the native CPU kernels.

**Acknowledgement.** Our method and implementation draw on the code of
[Rotlex](https://github.com/ustcllm/rotlex). We thank its authors for making
their implementation available. The original copyright notice and MIT license
are preserved in [third_party/rotlex/LICENSE](third_party/rotlex/LICENSE).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and the
Rotlex paper reference.

The Python import name remains `ft_anns` so that configuration formats and
existing experiments remain compatible. The package distribution is named
`learning-to-probe`; both `learning-to-probe` and `ft-anns` run the CLI.

## Install

The verified environment is Linux x86-64, Python 3.11, PyTorch 2.5.1, FAISS 1.9.0,
NumPy 2.4.0, and Pydantic 2.13.4. A C++17 compiler with OpenMP support is needed
to build the included PyTorch extension. No GPU or dataset is needed for the smoke test.

In a new environment, install dependencies and then build against that same PyTorch:

```bash
python -m pip install 'torch==2.5.1' 'faiss-cpu==1.9.0' 'numpy>=2' 'pydantic>=2.11' 'setuptools>=77' wheel build pytest matplotlib
python -m pip install --no-build-isolation --no-deps -e .
python -m ft_anns --help
learning-to-probe --help
```

If the environment already has a working FAISS CPU/GPU installation, reuse it
and run the second command with `--no-deps`. The test record for this export
reuses installed dependencies and rebuilds this project's extension; it does
not verify downloading dependencies into a new environment. FAISS installation
options are documented [upstream](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

## Run a small CPU example

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python examples/smoke_cpu.py
```

This generates 256 synthetic 96-dimensional vectors, builds exact ground truth
and a four-bucket partition, trains a small RAMPS router for two epochs, reloads
its checkpoint, and runs Fixed Query, L2PR, and L2PR+ retrieval. Full probing
must match exact search. Outputs and temporary training files are cleaned up
automatically. Adaptive coefficients in this example are illustrative values;
production coefficients are fitted with the policy-training tool below.

Run the included synthetic regression tests with:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

## Prepare real datasets and train

Run all commands below from this directory after installing the package.
Datasets and generated artifacts are not bundled. The registry in
`src/ft_anns/datasets/registry.py` lists required filenames and vector formats.
Place files in `data/<dataset>/` or set the corresponding environment variable.

```bash
DEEP1M_DATA_DIR=/datasets/deep-1M bash tools/deep1m/run_matrix.sh
SIFT1M_DATA_DIR=/datasets/sift1M bash tools/sift1m/run_matrix.sh
SPACEV1M_DATA_DIR=/datasets/spacev-1M bash tools/spacev1m/run_matrix.sh
```

The 1M workflows prepare ground truth and workloads, train the RAMPS router,
and evaluate fixed probing widths. The released 1M configurations contain
the RAMPS main workflow. They default to CUDA for production training.
Set `PYTHON_BIN` if the active interpreter should be overridden.

The 100M workflows require both training and test queries with matching ground
truth; the repository does not contain those large input files:

```bash
DEEP100M_DATA_DIR=/datasets/deep-100M bash tools/deep100m/run_matrix.sh
SIFT100M_DATA_DIR=/datasets/sift-100M bash tools/sift100m/run_matrix.sh
SPACEV100M_DATA_DIR=/datasets/spacev-100M bash tools/spacev100m/run_matrix.sh
```

The 100M `run_bucket65536_matrix.sh` entry points select the larger partition.
See [docs/compact_partitions.md](docs/compact_partitions.md) for index storage.

## Fit and evaluate adaptive policies

The policy-fitting entry point is retained at its original path to preserve
experiment compatibility. After the matching router workflow completes:

```bash
python tools/prototypes/global_gain_cost_prefix/prototype.py all \
  --dataset deep1m --gain-mode probability --device cpu \
  --targets 0.50 0.60 0.70 0.80 0.90 0.95 0.98 0.99
```

Its `cache`, `fit`, and `evaluate` stages can also run separately. Use `--help`
for bucket counts, workload overrides, cache reuse, and training settings.
These stages run on real data and can take considerably longer than the smoke test.
The retained CPU tools expose L2PR and L2PR+ retrieval:

```bash
python tools/100m_cpu_gain_cost_search/search_sweep.py --help
python tools/100m_cpu_gain_cost_search/compare_probability_prefix_limits.py --help
python tools/prototypes/global_gain_cost_prefix/refit_probability_100m.py --help
```

Current RAMPS v5 checkpoints use raw-logit fixed routing and the Gain-Cost
policy workflow above. The source retains older calibration interfaces for
compatibility; those interfaces reject the new RAMPS checkpoint format.
Legacy `*-alpha*` configuration files are excluded from this release.

## Release contents and verification

`src/` holds the library and native sources; `configs/` holds current 1M/100M
workflows; `tools/` holds selected training, policy, and retrieval entry points;
`tests/` and `examples/` contain small synthetic checks. Raw datasets,
checkpoints, indexes, benchmark outputs, development notes, local Git history,
and compiled binaries are excluded.

[SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json) records the source commit and
file hashes, including selected current working-tree changes.
[verification/README.md](verification/README.md) records the checks actually run.
The source archive can be rebuilt with `python -m build --sdist --no-isolation`;
the manifest includes native headers, fixtures, configs, and selected tools.

## License

[MIT](LICENSE). Rotlex's original MIT notice is preserved separately.

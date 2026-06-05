# TinyTrainerCPU — Absolute Edition v5.0

A production-grade, CPU-first linear model trainer written in C11. TinyTrainerCPU is built for engineers who need to train large-scale linear regression models as fast as possible on commodity hardware, without a GPU and without a deep learning framework.

---

## What Is This?

TinyTrainerCPU is a single-purpose machine learning engine that trains **linear regression models** using mini-batch stochastic gradient descent. It is deliberately small in scope but extreme in performance: every layer of the stack — memory layout, SIMD dispatch, optimizer, parallelism, and numerical stability — has been carefully engineered and iteratively improved across five major versions.

The codebase is a fully self-contained C library with no external ML dependencies. It builds with GCC or Clang, links only against `libm` and optionally `libnuma`, and runs on any x86-64 or AArch64 machine.

---

## What Problem Does It Solve?

Training large linear models on CPUs is surprisingly hard to do well. The naive implementation leaves enormous performance on the table: denormal floats silently halve throughput, data layout mismatches break SIMD vectorization, Adam's running-variance accumulator is corrupted under lock-free parallelism, and datasets that exceed available RAM make training impossible without out-of-core support.

TinyTrainerCPU addresses all of these systematically. Its goals are:

- **Maximum FLOP utilization** on any x86 or ARM CPU, from a Raspberry Pi to a 96-core Sapphire Rapids server.
- **Correctness under parallelism** — Hogwild! lock-free SGD with proven fixes for the Adam variance underestimation bug and gradient-aggregation race conditions.
- **Generalizable models** — Lookahead optimization and Stochastic Weight Averaging (SWA) reduce the noise inherent in parallel training and find flatter, more transferable loss minima.
- **Scalability beyond RAM** — mmap-backed out-of-core dataset loading lets a 16 GB machine train on a 500 GB dataset transparently.
- **Portability** — a single cross-architecture SIMD abstraction layer means the same source code compiles and auto-tunes for AVX2, AVX-512, BF16, AMX, ARM NEON, and ARM SVE without per-file `#ifdef` clutter.

---

## Who Is This For?

TinyTrainerCPU is aimed at a specific, technically sophisticated audience:

**Systems engineers and ML infrastructure engineers** who operate in environments where GPU access is unavailable, expensive, or overkill — ad-click prediction pipelines, risk scoring, sensor telemetry regression, embedded inference — and who need to squeeze every cycle out of the CPU they have.

**Performance engineers and HPC practitioners** interested in applied SIMD programming, NUMA-aware memory allocation, and lock-free parallel training. The codebase is extensively commented and each optimization is tagged with an `OPT #N` or `IMP #N` reference that traces directly to the code where it is applied.

**Researchers benchmarking CPU-side linear model training** who need a reproducible, well-characterized baseline with multiple optimizer variants (SGD + momentum, Adam, Lookahead, SWA) and hardware paths (scalar, AVX2, AVX-512, BF16, AMX, NEON, SVE) all in one harness.

**Industrial ML practitioners working with sparse high-cardinality data** — the feature hashing support (MurmurHash3, configurable hash table size) and CSR sparse dataset format make TinyTrainerCPU directly applicable to CTR prediction, NLP bag-of-words, and recommendation features with tens of millions of unique dimensions, while keeping the weight vector small enough to stay in L3 cache.

---

## Architecture Overview

The codebase is organized as a set of focused header modules assembled by a central implementation file.

```
TinyTrainerCPU/
├── cpu_dispatch.h     # Runtime CPUID / AT_HWCAP detection; sets global kernel pointers
├── simd_compat.h      # Cross-architecture VF_* macro abstraction (x86 + ARM)
├── numa_alloc.h       # NUMA-local aligned allocation; HT-aware core mapping
├── sparse.h           # CSR sparse dataset; Feature Hashing; indirect prefetch
├── lookahead.h        # Lookahead Optimizer: slow/fast weight synchronization
├── swa.h              # Stochastic Weight Averaging accumulator
├── mmap_dataset.h     # mmap-backed out-of-core binary dataset loader
├── tiny_trainer.h     # Public API: Dataset, LinearModel, TrainConfig, TrainResult
├── tiny_trainer.c     # Core training loop, kernels, Adam, elastic net, shuffle
├── main.c             # Quick-start demo
├── benchmark.c        # Benchmark harness with multi-variant timing table
└── Makefile
```

### Data Layouts

The feature matrix is stored in either **Array-of-Structures (AoS)** or **Structure-of-Arrays (SoA)** layout, selected automatically at dataset allocation time based on feature count (threshold: 256 features). AoS vectorizes the dot product over features; SoA vectorizes the gradient accumulation over samples. Both paths have dedicated, branch-free kernel specializations.

### SIMD Dispatch

`cpu_dispatch.h` performs a one-time runtime capability probe at startup and selects the best available kernel. The detected level is stored in the global `g_cpu_level` and reported in `TrainResult`.

| Level | Hardware |
|---|---|
| `CPU_SCALAR` | Fallback (any architecture) |
| `CPU_AVX2` | Intel / AMD Haswell and later |
| `CPU_AVX512` | Intel Skylake-X, Ice Lake, AMD Zen 4 |
| `CPU_BF16` | Intel Cooper Lake / Ice Lake SP (VDPBF16PS) |
| `CPU_AMX` / `CPU_AMX_BF16` | Intel Sapphire Rapids (tile-based matrix ops) |
| `CPU_NEON` | All AArch64 (AWS Graviton, Apple Silicon, Ampere) |
| `CPU_SVE` | ARM SVE-capable silicon (Graviton3/4, Neoverse N2) |

SVE support is fully vector-length agnostic: `svcntw()` is called at runtime so the same binary runs correctly on 128-bit and 512-bit SVE implementations without recompilation.

### Optimizer Stack

Training is driven by `train()` in `tiny_trainer.c`. The optimizer stack in v5.0 is:

1. **Inner optimizer**: Adam (default) or SGD with momentum. Adam uses bias-corrected moment estimates and non-temporal SIMD stores for the weight update to avoid polluting L1/L2 cache.
2. **Elastic Net regularization**: L1 (soft-thresholding) and L2 (weight decay) applied after each weight update.
3. **Lookahead** (IMP #1): Every `k_steps` inner optimizer steps, slow weights are interpolated toward fast weights and fast weights are reset to the new slow-weight anchor. This global sync stabilizes Hogwild! training and reduces the need for tight critical sections.
4. **Stochastic Weight Averaging** (IMP #6): During the final `swa_frac` fraction of epochs (default 10%), a running average of the weight vector is maintained in double precision. The averaged weights are substituted into the model at the end of training, producing a flatter minimum with better generalization.

### Parallelism

Parallel training uses **Hogwild!** lock-free SGD via OpenMP. Each thread accumulates gradients into a thread-local buffer for `TINY_BLOCK_AGG` batches before entering a critical section to update the shared model (Bug Fix #4). This dramatically reduces contention compared to per-batch locking while preserving convergence properties.

A locked reduction path (`hogwild = 0`) is also available for exact reproducibility.

---

## Key Features by Version Milestone

All features from prior versions are fully preserved. Version 5.0 adds:

| ID | Category | Feature |
|---|---|---|
| IMP #1 | Algorithmic | Lookahead Optimizer (Zhang et al., 2019) — slow/fast weight sets |
| IMP #2 | Architecture | ARM NEON & SVE via `simd_compat.h` abstraction layer |
| IMP #3 | Memory | `mmap` out-of-core dataset loader with `MADV_SEQUENTIAL` / `MADV_WILLNEED` |
| IMP #4 | Data Layout | Feature Hashing via MurmurHash3 — fixed-size L3-resident weight table |
| IMP #5 | Hardware | Indirect prefetching for sparse CSR data (TINY_PREFETCH_DIST = 8) |
| IMP #6 | Numerical | Stochastic Weight Averaging over final training epochs |
| IMP #7 | Micro-opt | Function multiversioning — AoS/SoA/scalar kernels dispatched once, no `if` in inner loop |

Earlier milestones contributed:

| ID | Category | Feature |
|---|---|---|
| BUG #1 | Fix | Adam timestep race condition in Hogwild! |
| BUG #2 | Fix | Correct SoA index macro |
| BUG #3 | Fix | SoA SIMD vectorization correctness |
| BUG #4 | Fix | Adam `v_w` underestimation from gradient-aggregation race |
| OPT #16 | Performance | MXCSR DAZ+FTZ — prevents denormal float penalty on x86 |
| OPT #17 | Performance | Native VDPBF16PS BF16 dot product on Cooperlake/Ice Lake SP |
| OPT #18 | Performance | HT-aware NUMA thread mapping |
| OPT #19 | Performance | Non-temporal (streaming) stores in Adam weight update |

---

## Public API

```c
// Dataset management
Dataset       *ds_alloc(size_t n_samples, size_t n_features);
Dataset       *ds_alloc_soa(size_t n_samples, size_t n_features);
void           ds_normalize(Dataset *ds);
void           ds_free(Dataset *ds);

// Out-of-core I/O (mmap_dataset.h)
int            ds_mmap_write(const Dataset *ds, const char *path);
Dataset       *ds_mmap_load(const char *path, MmapHandle *handle_out);
void           ds_mmap_free(Dataset *ds, MmapHandle *handle);

// Sparse datasets (sparse.h)
SparseDataset *sparse_ds_alloc(size_t n_samples, size_t n_features, size_t nnz);
SparseDataset *sparse_ds_hashed_alloc(size_t n_samples, size_t n_features, size_t nnz);

// Model
LinearModel   *model_alloc(size_t n_features);
LinearModel   *model_alloc_hashed(void);     // Fixed TINY_HASH_SIZE weight table
float          model_predict(const LinearModel *m, const float *x);
void           model_free(LinearModel *m);

// Training
TrainConfig    train_config_default(void);
TrainResult    train(LinearModel *m, const Dataset *ds, TrainConfig cfg);
float          evaluate_mse(const LinearModel *m, const Dataset *ds);
```

`TrainResult` reports epochs run, final MSE, wall-clock time, the SIMD level used, and flags for SoA layout, BF16, Lookahead, and SWA — making it straightforward to log exactly what the training run did.

---

## Building

Requires GCC ≥ 9 or Clang ≥ 11, a C11-capable libc, and `libm`. OpenMP is strongly recommended. `libnuma` is optional.

```bash
# Standard build — AVX2 + OpenMP (most x86-64 machines)
make

# AVX-512 with native BF16 dot product (Intel Sapphire Rapids / Cooper Lake)
make bf16_vdp

# AVX-512 + NUMA-local allocation + HT-aware thread mapping
make avx512_numa

# ARM NEON — all AArch64 (Graviton, Apple Silicon, Ampere Altra)
make arm_neon

# ARM SVE — runtime-agnostic vector length (Graviton3/4, Neoverse N2)
make arm_sve

# mmap out-of-core dataset demo
make outofcore

# Scalar fallback — no SIMD (CI, embedded, or debugging)
make scalar

# Full benchmark suite across all available instruction sets
make bench
```

---

## Configuration Reference

`train_config_default()` provides sensible defaults. Key fields:

| Field | Default | Description |
|---|---|---|
| `lr` | 0.01 | Initial learning rate |
| `lr_decay` | 0.99 | Per-epoch learning rate decay multiplier |
| `use_adam` | 1 | Use Adam optimizer (0 = SGD + momentum) |
| `epochs` | 1000 | Maximum training epochs |
| `batch_size` | 256 | Samples per mini-batch |
| `hogwild` | 1 | Lock-free parallel training |
| `l1_lambda` | 0.0 | L1 regularization coefficient |
| `l2_lambda` | 1e-4 | L2 regularization coefficient |
| `use_lookahead` | 1 | Enable Lookahead Optimizer |
| `lookahead_k` | 5 | Inner steps between slow-weight syncs |
| `lookahead_alpha` | 0.5 | Slow-weight interpolation coefficient |
| `use_swa` | 1 | Enable Stochastic Weight Averaging |
| `swa_frac` | 0.10 | Fraction of epochs included in SWA average |
| `use_bf16` | 0 | Enable BF16 compute path (requires CPU_BF16 or higher) |
| `normalize` | 1 | Apply min-max feature normalization before training |
| `early_stop_eps` | 1e-7 | Stop when epoch-over-epoch MSE delta falls below this |

---

## Compile-Time Tunables

Override these via `-D` flags or by editing `tiny_trainer.h`:

| Macro | Default | Effect |
|---|---|---|
| `TINY_HASH_SIZE` | `1 << 20` | Feature hash table size (must be power of 2) |
| `TINY_PREFETCH_DIST` | 8 | Sparse indirect prefetch lookahead distance |
| `TINY_TILE_N` | 64 | Batch tiling depth for blocked GEMM kernels |
| `TINY_SOA_THRESHOLD` | 256 | Feature count above which SoA layout is auto-selected |
| `TINY_BLOCK_AGG` | 4 | Batches accumulated per thread before a critical section |
| `TINY_SWA_FRAC` | 0.10 | Default SWA epoch fraction |
| `TINY_USE_PREFETCH` | 1 | Enable `__builtin_prefetch` in dense kernels |

---

## Design Decisions

**Why a unified `simd_compat.h` instead of per-file `#ifdef`?**
All intrinsic calls route through `VF_*` macros. Adding a new ISA (e.g., RISC-V V extension) requires editing one file. The macro layer is resolved entirely at compile time with zero runtime overhead.

**Why does Lookahead reduce `omp critical` pressure?**
In vanilla Hogwild! + Adam, every `TINY_BLOCK_AGG` batches each thread enters a critical section. Lookahead's periodic slow-weight sync acts as its own global synchronization point, allowing threads to use a longer effective interval between critical sections — `k_steps × TINY_BLOCK_AGG` batches — at the cost of one extra weight-vector copy every `k_steps` inner steps.

**SWA and early stopping interaction:**
If early stopping fires before `swa_start_epoch`, no SWA snapshots are taken and the model is returned unmodified. This is intentional: convergence is already achieved, so averaging is unnecessary.

**Feature hashing collision rate:**
With `TINY_HASH_SIZE = 2^20` (~1M slots) and a typical 10M unique-feature dataset, the expected per-slot collision rate is approximately 0.5%. This matches the trade-off made by production systems such as Vowpal Wabbit and libFFM and is acceptable for most industrial sparse models.

---

## Limitations

- **Linear models only.** TinyTrainerCPU trains one layer: `ŷ = w·x + b`. It does not support neural networks, decision trees, or multi-class classification.
- **Regression loss only.** The objective is mean squared error. Classification or ranking losses are not implemented.
- **Linux-primary.** The `mmap` out-of-core loader, `MADV_WILLNEED`, `posix_fadvise`, and NUMA support are Linux-specific. The codebase compiles on macOS and Windows but falls back gracefully where OS APIs are absent.
- **No serialization for trained models.** The `ds_mmap_write` / `ds_mmap_load` functions serialize datasets, not trained model weights. Model persistence must be handled by the caller.

---

## License

See `LICENSE.txt` in the repository root.

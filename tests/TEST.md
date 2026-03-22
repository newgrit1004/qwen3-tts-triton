# Testing Philosophy

## 3-Tier Verification System

Inspired by Liger Kernel (LinkedIn) and adapted for TTS inference optimization.

| Tier | What it proves | How | Time |
|------|---------------|-----|------|
| **Tier 1** | Each kernel is numerically correct | atol/rtol vs PyTorch reference | ~2s |
| **Tier 2** | 28-layer model output is preserved | Multi-metric parity comparison | ~15s, GPU |
| **Tier 3** | End-to-end speech quality is equivalent | Independent distribution comparison | ~5-30min, GPU |

```bash
make test          # Tier 1
make test-parity   # Tier 2
make verify        # Tier 1 + 2
make verify-all    # Tier 1 + 2 + 3
```

## Tier 1: Kernel Correctness

Every Triton kernel is tested against a pure PyTorch reference implementation.
Tests live in `tests/kernels/` and follow a 5-stage pattern adapted from
[autokernel](https://github.com/RightNow-AI/autokernel):

### 1. Accuracy (smoke + shape sweep)

Compare kernel output to PyTorch reference with dtype-aware tolerances:

- `float32`: atol=1e-5, rtol=1e-5
- `float16`: atol=1e-3, rtol=1e-3
- `bfloat16`: atol=1e-2 to 5e-2, rtol=1e-2 to 5e-2 (7-bit mantissa)

Parametrized across multiple shapes and dtypes to cover realistic workloads.

### 2. Determinism

Each kernel is run 3 times with identical inputs. Outputs must be
**bitwise identical** (`torch.equal`). Non-deterministic kernels are bugs,
not features.

### 3. Numerical Stability

Adversarial inputs that stress floating-point edge cases:

- **Near-zero** (1e-7): tests epsilon handling in normalization
- **Large magnitude** (50% of dtype max): tests overflow protection
- **Mixed scale** (1000x and 0.001x in same tensor): tests dynamic range

All outputs must be free of NaN and Inf.

### 4. Edge Cases

- **2D inputs** `(N, H)` — kernels accept both 2D and 3D
- **Non-contiguous tensors** — `.contiguous()` path verification
- **Minimal dimensions** — seq_len=1, n_heads=1
- **Non-power-of-2** — odd sequence lengths (63, 127, 255)

### 5. Input Validation

For kernels with Python-level guards (e.g., SwiGLU):

- Shape mismatch raises `ValueError`
- CPU tensors raise `ValueError`

These tests do **not** require GPU.

## Tier 2: Model Parity — Why Multiple Metrics?

### Why Not Cosine Similarity Alone?

Cosine similarity is a natural first choice for comparing high-dimensional
vectors, but it has known blind spots. In particular, for vectors with low
variance around a constant (e.g., LayerNorm weights initialized near 1.0),
`cos_sim ≈ 1/(1+variance)` — always near 1.0 regardless of actual similarity.
This makes it unsuitable as a sole verification metric.

For our use case (comparing activation vectors), cosine similarity *is*
valid — activations have high variance and aren't near-constant. But even
here, relying on it **alone** leaves gaps:

1. **Scale-invariant**: Won't detect if activations are 10x too large
   (e.g., missing RMSNorm, wrong epsilon)
2. **Insensitive at high values**: The difference between 0.9999 and 0.999
   represents a 10x increase in angular error
3. **Industry consensus**: vLLM, TensorRT-LLM, and Liger Kernel all use
   multiple complementary metrics

### Tier 2 Metrics

| Metric | Threshold | Catches |
|--------|-----------|---------|
| Cosine similarity | > 0.95 | Directional alignment |
| Relative L2 | < 0.05 (5%) | Magnitude/scale drift |
| SNR (dB) | > 25 dB | Combined direction + magnitude |
| Max abs diff | Per-layer bounds | Extreme value divergence |

Thresholds are set at ~3-4x headroom over observed values.

**Informational metrics** (logged but not enforced):
Pearson correlation, RMSE, mean absolute difference.

### Sampled Layers

5 of 28 decoder layers are sampled: [0, 7, 14, 21, 27].
This covers early/mid/late layers while keeping memory usage reasonable.

## Tier 3: E2E Quality

Pair-level waveform comparison (PESQ/STOI/MCD) is inappropriate for
stochastic autoregressive TTS — the same model produces different
waveforms each run. Instead, we follow the vLLM/TensorRT-LLM pattern:

1. Each model generates **independently** on test sentences
2. Per-sample task metrics: CER (ASR accuracy), UTMOS (quality), speaker similarity
3. Compare **distributions** via mean delta + Mann-Whitney U test

See `docs/verification-tiers.md` for full threshold details.

## Test Organization

```
tests/
├── conftest.py                # device() fixture (shared)
├── kernels/                   # Tier 1: kernel correctness
│   ├── conftest.py            # hidden_size, eps, intermediate_size
│   ├── test_utils.py          # calculate_settings()
│   ├── test_rms_norm.py       # RMSNorm kernel
│   ├── test_rope.py           # M-RoPE kernel
│   ├── test_swiglu.py         # SwiGLU kernel
│   └── test_fused_norm.py     # Fused Add+RMSNorm kernel
└── test_model_parity.py       # Tier 2: model-level parity
```

## Principles

1. **Correctness first, performance second.** A fast wrong answer is worse
   than a slow right answer. Never merge a kernel that fails accuracy tests.

2. **Reference implementations are immutable.** The PyTorch reference in
   each test file is the ground truth. If results diverge, fix the kernel,
   not the reference.

3. **Multiple metrics, not just one.** No single number captures all
   failure modes. Use complementary metrics that cover each other's blind spots.

4. **Test what can go wrong, not just what should go right.** Determinism,
   numerical stability, and edge cases catch bugs that happy-path tests miss.

5. **Thresholds have headroom.** Set at 3-4x observed values to avoid
   flaky tests while catching genuine regressions.

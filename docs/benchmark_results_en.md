# Benchmark Results

> **v0.2.0** — 2026-04-13 | GPU: **NVIDIA RTX 5090** (Blackwell, sm_120, 32GB VRAM, CUDA 12.8)

## Speed Benchmark

Model load time includes weight loading + initialization (Faster/Hybrid includes CUDA graph capture).

| Runner | Load Time | Latency (ko) | Latency (en) | RTF (ko) | RTF (en) | vs Base | Peak VRAM |
|--------|-----------|-------------|-------------|---------|---------|---------|-----------|
| **Base** | 17.5s | 4,615 ms | 5,081 ms | 0.88x | 0.90x | 1.0x | 4.03 GB |
| **Base+TQ** | 8.3s | 9,030 ms | 5,745 ms | 0.82x | 0.79x | 0.7x | 4.07 GB |
| **Triton** | 7.9s | 4,130 ms | 4,462 ms | 1.00x | 1.00x | 1.1x | 4.03 GB |
| **Triton+TQ** | 7.4s | 8,045 ms | 5,877 ms | 0.93x | 0.88x | 0.7x | 4.09 GB |
| **Faster** | 9.2s | 1,136 ms | 1,265 ms | 3.49x | 3.52x | 4.0x | 4.28 GB |
| **Hybrid** | **6.0s** | **886 ms** | 1,042 ms | 4.20x | **4.26x** | **5.0x** | 4.32 GB |
| **Hybrid+TQ** | 6.5s | 944 ms | **1,032 ms** | **4.27x** | 4.25x | 4.9x | 4.33 GB |

- **RTF** (Real-Time Factor): audio_duration / generation_time. >1.0 = faster than real-time.
- All runners use the same model weights (~4GB VRAM). No extra VRAM from kernel fusion.
- Triton/Triton+TQ/Hybrid/Hybrid+TQ use the default partial patch range `[0, 24)`; the final 4 decoder layers stay in PyTorch for pronunciation stability.

## Kernel Micro-Benchmarks

| Kernel | PyTorch (µs) | Triton (µs) | Speedup | PT Mem (MB) | TR Mem (MB) |
|--------|-------------|------------|---------|------------|------------|
| RMSNorm | 40.91 | 7.42 | **5.51x** | 266.01 | 260.00 |
| SwiGLU | 19.43 | 16.00 | **1.21x** | 280.00 | 274.00 |
| M-RoPE | 367.90 | 37.29 | **9.86x** | 264.25 | 264.75 |
| FusedNorm+Residual | 40.57 | 9.03 | **4.50x** | 270.01 | 264.00 |

## 3-Tier Verification

### Tier 1: Kernel + CPU Regression Suite — PASS (197 tests)

Measured on RTX 5090 / WSL2 (Ubuntu 22.04) / PyTorch 2.10.0+cu128: **`make test` completes in ~48 s**.

| Scope | Tests | Status |
|--------|-------|--------|
| RMSNorm kernel | 25 | PASS |
| SwiGLU kernel | 21 | PASS |
| M-RoPE kernel | 18 | PASS |
| FusedNorm+Residual kernel | 14 | PASS |
| Fused dequant kernel | 22 | PASS |
| TurboQuant kernel | 56 | PASS |
| Kernel utils | 12 | PASS |
| Partial patching regressions | 18 | PASS |
| Release doc/Makefile/bench guards | 11 | PASS |
| **Total** | **197** | **PASS** |

### Tier 2: Model Parity — PASS (2-pair)

`make test-parity` completes in ~46 s on RTX 5090 / WSL2 (includes model loading).

Compares hidden states layer-by-layer. All pairs use the same model + kernel patching for a fair comparison.

**Pair A: Base vs Triton**

| Layer | Cosine Sim | Relative L2 | SNR (dB) |
|-------|-----------|------------|----------|
| L0 | 0.999995 | 0.0043 | 47.3 |
| L7 | 0.999977 | 0.0075 | 42.5 |
| L14 | 0.999852 | 0.0175 | 35.1 |
| L21 | 0.999177 | 0.0407 | 27.8 |
| L27 | 0.997900 | 0.0649 | 23.8 |
| **Output** | **0.997156** | - | - |

**Pair B: Faster vs Hybrid**

| Layer | Cosine Sim | Relative L2 | SNR (dB) |
|-------|-----------|------------|----------|
| L0 | 0.999995 | 0.0043 | 47.3 |
| L7 | 0.999977 | 0.0075 | 42.5 |
| L14 | 0.999852 | 0.0175 | 35.1 |
| L21 | 0.999177 | 0.0407 | 27.8 |
| L27 | 0.997900 | 0.0649 | 23.8 |
| **Output** | **0.997156** | - | - |

> Threshold: cosine sim > 0.95, relative L2 < 0.08, SNR > 22 dB. All layers pass with large margins.

### Tier 3: E2E Quality (full mode) — 4/6 optimized runners PASS

Official release quality numbers use full mode (3 runs/sentence), then compare each runner against Base.

| Runner | UTMOS | CER | Speaker Sim | Status |
|--------|-------|-----|-------------|--------|
| **Base** (ref) | 3.40 ± 0.78 | 0.04 ± 0.06 | - | ref |
| **Base+TQ** | 3.17 ± 0.81 | 0.42 ± 2.02 | 0.82 | **FAIL** |
| **Triton** | 3.40 ± 0.76 | 0.04 ± 0.07 | 0.85 | **PASS** |
| **Triton+TQ** | 3.04 ± 0.83 | 0.43 ± 1.49 | 0.83 | **FAIL** |
| **Faster** | 3.42 ± 0.75 | 0.04 ± 0.04 | 0.83 | **PASS** |
| **Hybrid** | 3.38 ± 0.78 | 0.04 ± 0.06 | 0.83 | **PASS** |
| **Hybrid+TQ** | 3.32 ± 0.78 | 0.05 ± 0.07 | 0.83 | **PASS** |

> Release caveats:
> `base+tq` fails the full release gate with `CER delta 0.3801 > 0.05` and `Mann-Whitney p=0.0340 < 0.05`.
> `triton+tq` fails the full release gate with `UTMOS delta 0.3565 > 0.3`, `CER delta 0.3865 > 0.05`, and `Mann-Whitney p=0.0015 < 0.05`.
> Fast mode remains useful as a smoke check, but full mode is the release authority.

**Thresholds**:
- |UTMOS delta| < 0.3
- UTMOS floor > 2.5 (both)
- |CER delta| < 0.05
- Speaker Sim > 0.75
- Mann-Whitney U p > 0.05 (full mode only)

## Commands

```bash
make bench          # Default suite (kernels + speed + fast quality + report)
make bench-speed    # E2E speed benchmark only
make bench-kernels  # Kernel micro-benchmarks only
make eval-fast      # Tier 3 quality evaluation (fast, ~15min)
make eval-full      # Tier 3 quality evaluation (full, ~80min)
make verify         # 3-Tier report from existing Tier 3 artifacts
make verify-all     # Run eval-full, then build the 3-Tier report
```

## Environment

- **OS**: Ubuntu 22.04 on WSL2 (Windows Subsystem for Linux 2)
- **Kernel**: Linux 5.15.167.4-microsoft-standard-WSL2
- **GPU**: NVIDIA RTX 5090 (32GB GDDR7, sm_120)
- **CUDA**: 12.8
- **PyTorch**: nightly (cu128)
- **Triton**: 3.2.0
- **Transformers**: 4.57.3
- **Python**: 3.12
- **Model**: Qwen3-TTS 1.7B (12Hz CustomVoice)

## Contributor Benchmark: RTX 4090

Contributed by **tantara**. These results are a separate community benchmark and
are not part of the official RTX 5090 release tables above.

### Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 4090 |
| PyTorch | 2.10.0+cu128 |
| Triton | 3.6.0 |
| CUDA | 12.8 |
| Driver | 550.144.03 |

### Kernel Micro-Benchmarks

`bf16`, `batch=1`, `seq_len=512`, `hidden=2048`

| Kernel | PyTorch (us) | Triton (us) | Speedup | Compile (s) |
|--------|--------------|-------------|---------|-------------|
| RMSNorm | 38.6 | 8.3 | 4.66x | 0.48 |
| SwiGLU | 32.1 | 26.6 | 1.21x | 0.13 |
| M-RoPE | 123.7 | 42.8 | 2.89x | 0.17 |
| Fused Norm+Residual | 42.1 | 12.1 | 3.49x | 0.16 |

### E2E Inference

`bf16`, 2 texts (`ko` + `en`), 3 warmup + 20 measured runs

| Mode | Latency (ko) | Latency (en) | RTF (ko) | RTF (en) | Peak VRAM |
|------|--------------|--------------|----------|----------|-----------|
| Base (PyTorch) | 2,546 ms | 2,878 ms | 1.59x | 1.60x | 4.01 GB |
| Compile (`torch.compile`) | 2,685 ms | 2,685 ms | 1.60x | 1.60x | 3.98 GB |
| Triton | 2,079 ms | 2,460 ms | 1.87x | 1.88x | 3.98 GB |
| Faster | 922 ms | 1,016 ms | 4.20x | 4.21x | 4.23 GB |
| Hybrid (Faster+Triton) | 770 ms | 881 ms | 5.16x | 5.19x | 4.27 GB |

### Notes

- Hybrid mode reaches `5.16x` RTF on Korean, well above real-time.
- `torch.compile` (`inductor`, `reduce-overhead`) shows no measurable E2E gain over eager Base; autoregressive decoding limits compiler upside here.
- Triton kernel fusion gives about `1.2x` over Base; the big wins come from CUDA graphs in Faster/Hybrid.
- Reproduced with `make bench-kernels` and `make bench-e2e`.
- `TorchCompileRunner` reference: <https://gist.github.com/tantara/b23b717c7bf252b7e897e1adb02c25b5>

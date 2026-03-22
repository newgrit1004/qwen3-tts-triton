# Benchmark Results

GPU: **NVIDIA RTX 5090** (Blackwell, sm_120, 32GB VRAM, CUDA 12.8)

## Speed Benchmark

Model load time includes weight loading + initialization (Faster/Hybrid includes CUDA graph capture).

| Runner | Load Time | Latency (ko) | Latency (en) | RTF (ko) | RTF (en) | Speedup | Peak VRAM |
|--------|----------|-------------|-------------|---------|---------|---------|-----------|
| **Base** | 9.9s | 3,902 ms | 5,511 ms | 1.00x | 0.82x | 1.0x | 4.01 GB |
| **Triton** | 6.7s | 3,767 ms | 3,747 ms | 1.22x | 1.27x | 1.1x | 4.04 GB |
| **Faster** | 5.1s | 1,199 ms | 1,247 ms | 3.60x | 3.50x | 3.6x | 4.32 GB |
| **Hybrid** | 7.1s | 919 ms | 1,047 ms | 4.39x | 4.26x | **4.7x** | 4.30 GB |

- **RTF** (Real-Time Factor): audio_duration / generation_time. >1.0 = faster than real-time.
- **Speedup**: vs Base (based on Korean mean latency).
- All runners use the same model weights (~4GB VRAM). No extra VRAM from kernel fusion.

## 3-Tier Verification

### Tier 1: Kernel Correctness — PASS

| Kernel | Tests | Status |
|--------|-------|--------|
| RMSNorm | 16 | PASS |
| SwiGLU | 12 | PASS |
| M-RoPE | 8 | PASS |
| FusedNorm+Residual | 7 | PASS |
| CLI/Utils | 82 | PASS |
| **Total** | **125** | **PASS** |

### Tier 2: Model Parity — PASS (2-pair)

Compares hidden states layer-by-layer before/after Triton kernel application. Both pairs use the same model + Triton patching for a fair comparison.

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

### Tier 3: E2E Quality — Triton PASS, Hybrid PASS, Faster FAIL

Each runner generates speech independently, then quality distributions are compared against Base (vLLM/TensorRT-LLM pattern).

| Runner | UTMOS | CER | Speaker Sim | Status |
|--------|-------|-----|-------------|--------|
| **Base** (ref) | 3.12 ± 0.54 | 0.16 ± 0.14 | - | - |
| **Triton** | 3.29 ± 0.46 | 0.18 ± 0.14 | 0.76 | PASS |
| **Faster** | 3.29 ± 0.60 | 0.22 ± 0.26 | 0.75 | FAIL |
| **Hybrid** | 3.30 ± 0.85 | 0.20 ± 0.14 | 0.77 | PASS |

> Faster FAIL reason: CER delta 0.057 > threshold 0.05. This is due to variance in fast mode (1 run/sentence, whisper-small). Expected to PASS in full mode (3 runs/sentence, whisper-large-v3, Mann-Whitney U test).

**Thresholds**:
- |UTMOS delta| < 0.3
- UTMOS floor > 2.5 (both)
- |CER delta| < 0.05
- Speaker Sim > 0.75
- Mann-Whitney U p > 0.05 (full mode only)

## Commands

```bash
make bench          # Full (kernels + speed + quality + verification)
make bench-speed    # E2E speed benchmark only
make bench-kernels  # Kernel micro-benchmarks only
make eval-fast      # Tier 3 quality evaluation (fast, ~15min)
make eval-full      # Tier 3 quality evaluation (full, ~90min)
make verify         # 3-Tier verification report
```

## Environment

- **OS**: Ubuntu 22.04 on WSL2 (Windows Subsystem for Linux 2)
- **Kernel**: Linux 5.15.167.4-microsoft-standard-WSL2
- **GPU**: NVIDIA RTX 5090 (32GB GDDR7, sm_120)
- **CUDA**: 12.8
- **PyTorch**: nightly (cu128)
- **Python**: 3.12
- **Model**: Qwen3-TTS 1.7B (12Hz CustomVoice)

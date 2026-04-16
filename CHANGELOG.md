# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-16

### Added

- **TurboQuant KV Cache**: calibration-free INT4/INT3 KV cache quantization based on PolarQuant (arXiv:2504.19874)
  - Lloyd-Max codebook + random orthogonal rotation + scalar quantization
  - Fused dequantization Triton kernel (CUDA launches per token: ~1792 → ~56)
  - 4-8x KV cache memory reduction (INT4)
  - Drop-in `TurboQuantKVCache` class replacing HuggingFace `DynamicCache`
- **Partial Patching**: layer-range selective Triton kernel patching
  - `patch_range=(0, 24)` — fuse first 24 layers with Triton, keep last 4 in PyTorch
  - Fine-grained speed/quality tradeoff control
  - Validation and logging of patched layer counts
- **7-Mode Inference**: expanded from 4 to 7 inference modes
  - Existing: `base`, `triton`, `faster`, `hybrid`
  - New: `base+tq`, `triton+tq`, `hybrid+tq` (TurboQuant variants)
  - `create_runner("hybrid+tq")` factory function for easy mode selection
  - `ALL_RUNNER_NAMES` constant listing all available modes
- 8 new benchmark scripts: partial patching sweep, KV memory analysis, throughput scaling, fixed/long sequence E2E, per-token latency analysis, tongue twister evaluation
- UI: Partial Patching visualization tab (`ui/tab_partial.py`)
- Tests: TurboQuant kernel tests (464 lines), Fused Dequant tests (303 lines), Partial Patching tests (212 lines)

### Changed

- **Evaluation tooling**: OpenAI Whisper → Cohere Transcribe + multilingual phoneme tools (pypinyin, g2p-en, g2pk)
- **Tier 2 verification**: simplified from 4-pair to 2-pair comparison (Base↔Triton, Faster↔Hybrid)
- Benchmark results refreshed for the 7-mode release tables: Hybrid 5.0x and Hybrid+TQ 4.9x faster than Base on RTX 5090

### Removed

- SageAttention dependency and integration
- OpenAI Whisper dependency (replaced by Cohere Transcribe)

## [0.1.0] - 2026-03-23

### Added

- **4 Fused Triton Kernels** for Qwen3-TTS 1.7B Talker:
  - **RMSNorm**: variance + normalize + scale in SRAM, single kernel launch
  - **SwiGLU**: fused silu(gate) * up, eliminates intermediate tensor
  - **M-RoPE**: 3D rotary position embedding (sections=[24,20,20]) in Triton
  - **Fused Norm+Residual**: residual add + RMSNorm in single kernel
- **4 Inference Modes**: Base, Triton, Faster, Hybrid
  - Monkey-patch drop-in via `apply_triton_kernels()`
  - Weight sharing — zero additional VRAM usage
- **3-Tier Verification** (Liger Kernel style):
  - Tier 1: kernel numerical accuracy (atol/rtol)
  - Tier 2: model parity (cosine similarity > 0.95 across layers)
  - Tier 3: E2E quality distribution comparison (UTMOS/CER/Speaker SIM)
- Streamlit dashboard: real-time inference comparison UI (4 tabs)
- Benchmarks: kernel microbenchmarks + E2E speed/quality measurement
- CLI entry point: `qwen3-tts` command
- PyPI packaging: hatchling build, Apache-2.0 license

[0.2.0]: https://github.com/newgrit1004/qwen3-tts-triton/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/newgrit1004/qwen3-tts-triton/releases/tag/v0.1.0

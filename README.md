# Qwen3-TTS-Triton

[![CI](https://github.com/newgrit1004/qwen3-tts-triton/actions/workflows/ci.yml/badge.svg)](https://github.com/newgrit1004/qwen3-tts-triton/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/qwen3-tts-triton)](https://pypi.org/project/qwen3-tts-triton/)
[![Python](https://img.shields.io/pypi/pyversions/qwen3-tts-triton)](https://pypi.org/project/qwen3-tts-triton/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Up to 5x faster Qwen3-TTS inference through Triton kernel fusion.**

[Korean (한국어)](README_ko.md) | [Benchmark Results](docs/benchmark_results_en.md)

> [!NOTE]
> This project has only been tested on **RTX 5090 (Blackwell, sm_120)** with **WSL2** (CUDA 12.8, PyTorch nightly cu128).
> Triton kernels are architecture-agnostic (no sm_120-specific code), so they should work on other NVIDIA GPUs (A100, H100, RTX 4090, etc.), but this has **not been verified**. If you test on a different GPU, please open an issue or PR with your results!

---

Qwen3-TTS-Triton replaces performance-critical operators in [Qwen3-TTS 1.7B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) with hand-written [Triton](https://github.com/triton-lang/triton) kernels. Inspired by [Liger Kernel](https://github.com/linkedin/Liger-Kernel) (LinkedIn), each kernel fuses multiple HBM round-trips into a single pass, reducing memory traffic without any additional VRAM usage.

It can also be combined with [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) (CUDA Graph + static KV-cache) as a **Hybrid** mode for maximum throughput.

### 💡 Why Triton?

- 🪶 **Lightweight & Portable** — No serving infrastructure needed. Just `pip install qwen3-tts-triton` and call `apply_triton_kernels()`. Works in standalone scripts, [ComfyUI nodes](https://github.com/newgrit1004/ComfyUI-Qwen3-TTS-Triton), Gradio apps, or any Python environment.
- 🎲 **Faster Iteration on Stochastic TTS** — Qwen3-TTS generates different output each run. For best results, generate multiple candidates and pick the best one. With Hybrid mode's **~5x speedup**, you can produce 5 candidates in the time it used to take for 1 — more takes, better results.

### ✨ Highlights

- ⚡ **4 Fused Triton Kernels** — RMSNorm, SwiGLU, M-RoPE, Norm+Residual
- 🎯 **4 Inference Modes** — Base, Triton, Faster, Hybrid
- 🔬 **3-Tier Verification** — Kernel correctness → Model parity → E2E quality distribution
- 💾 **Zero Extra VRAM** — Pure kernel fusion, no model changes
- 🔌 **Drop-in Patching** — Single `apply_triton_kernels()` call, weight sharing via monkey-patch
- 📊 **Streamlit Dashboard** — Side-by-side comparison UI with live metrics

## 📦 Install

**Requirements**: Python 3.12+, CUDA 12.8+, NVIDIA GPU (8GB+ VRAM). Tested on WSL2 (Windows Subsystem for Linux 2).

### From PyPI

```bash
# 1. Install PyTorch with CUDA support first
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Install qwen3-tts-triton
pip install qwen3-tts-triton
```

### From Source (development)

```bash
# Install UV (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/newgrit1004/qwen3-tts-triton.git
cd qwen3-tts-triton
make setup  # uv sync --all-extras --dev + pre-commit install + git config
```

> **UV handles virtual environments automatically** — no need to manually activate a venv.
> All commands use the `uv run` prefix (e.g., `uv run pytest`, `uv run python script.py`).
> PyTorch is installed from the [cu128 index](https://download.pytorch.org/whl/cu128) automatically via `pyproject.toml`.

#### Dependency Groups

```bash
uv sync                 # Core (triton, transformers, faster-qwen3-tts, streamlit, plotly)
uv sync --extra eval    # + Quality evaluation (whisper, jiwer, resemblyzer)
uv sync --extra dev     # + Dev tools (ruff, pytest, pre-commit)
uv sync --extra all     # Everything
```

## 🚀 Quick Start

> [!TIP]
> On first run, the model (~3.5GB) is automatically downloaded from HuggingFace.
> To download in advance: `huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`

### Triton Mode

```python
from qwen3_tts_triton import TritonRunner
import soundfile as sf

runner = TritonRunner()
runner.load_model()  # Downloads model on first run (~3.5GB)

result = runner.generate(
    text="Hello, this is optimized with Triton kernels.",
    language="English",
    speaker="vivian",
)

# Save audio
sf.write("output.wav", result["audio"], result["sample_rate"])
print(f"Generated in {result['time_s']:.2f}s, VRAM: {result['peak_vram_gb']:.2f}GB")

runner.unload_model()
```

### Hybrid Mode (Triton + CUDA Graph, ~5x faster)

```python
from qwen3_tts_triton import TritonFasterRunner
import soundfile as sf

runner = TritonFasterRunner()
runner.load_model()  # Triton patches applied before CUDA Graph capture

result = runner.generate(
    text="Hybrid mode: CUDA Graph + Triton fusion.",
    language="English",
    speaker="vivian",
)

sf.write("output.wav", result["audio"], result["sample_rate"])
runner.unload_model()
```

### 📊 Streamlit Dashboard

```bash
make ui  # http://localhost:8501
```

The dashboard provides:
- 🔄 Side-by-side inference comparison across all modes
- 📈 Live metrics (TTFA, RTF, Total Time, Peak VRAM)
- 📉 Plotly charts for visual comparison
- ✅ 3-Tier verification result cards

## 🎧 Audio Samples

Pre-generated samples comparing inference modes (custom voice + voice cloning).

| Mode | Directory |
|------|-----------|
| Base (PyTorch) | [`assets/audio_samples/base/`](assets/audio_samples/base/) |
| Triton | [`assets/audio_samples/triton/`](assets/audio_samples/triton/) |
| Faster (CUDA Graph) | [`assets/audio_samples/faster/`](assets/audio_samples/faster/) |
| Hybrid (Faster+Triton) | [`assets/audio_samples/hybrid/`](assets/audio_samples/hybrid/) |

Each directory contains custom voice samples (5 Korean + 5 English) and voice cloning samples using [LJSpeech reference audio](assets/reference_audio/) (Public Domain).

> Use `make ui` → **Audio Samples** tab for side-by-side playback and comparison.
> Regenerate: `make generate-samples` (GPU required).

## ⚡ Triton Kernels

All kernels target the **Qwen3-TTS Talker** (28-layer Transformer, hidden_size=2048, intermediate=6144).

| Kernel | What It Fuses | HBM Savings | File |
|--------|--------------|-------------|------|
| **RMSNorm** | variance + normalize + scale in SRAM | 4→1 round-trips | `kernels/rms_norm.py` |
| **SwiGLU** | `silu(gate) * up` — eliminates intermediate tensor | 3→1 round-trips | `kernels/swiglu.py` |
| **M-RoPE** | 3D positional encoding (sections=[24,20,20]) | In-place compute | `kernels/rope.py` |
| **Fused Norm+Residual** | `residual + x` then RMSNorm in one kernel | 2 kernels → 1 | `kernels/fused_norm_residual.py` |

### 🔌 How Patching Works

`apply_triton_kernels()` performs in-place monkey-patching:

1. **RMSNorm modules** → replaced with `TritonRMSNorm` (shares original weights, zero copy)
2. **MLP forward** → patched to use `triton_swiglu_forward` (fused gate+up projection)
3. **Decoder layer forward** → patched for fused residual addition + normalization

```python
from qwen3_tts_triton.models.patching import apply_triton_kernels

# Patches all 28 decoder layers in-place (patch counts logged via logging)
apply_triton_kernels(model)
```

<details>
<summary><b>Advanced: Manual Patching</b></summary>

If you want to apply Triton kernels to a model loaded outside the Runner API:

```python
from qwen_tts import Qwen3TTSModel
from qwen3_tts_triton.models.patching import apply_triton_kernels
import torch

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# Patch the internal nn.Module (not the wrapper)
apply_triton_kernels(model.model)

wavs, sr = model.generate_custom_voice(
    text="Hello, this is optimized with Triton kernels.",
    language="English",
    speaker="vivian",
)
```

For Hybrid mode with manual patching, use `find_patchable_model()` to resolve the internal module:

```python
from faster_qwen3_tts import FasterQwen3TTS
from qwen3_tts_triton.models.patching import apply_triton_kernels, find_patchable_model

model = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device="cuda"
)

# FasterQwen3TTS wraps multiple layers: model.model.model reaches the nn.Module
internal = find_patchable_model(model.model)
apply_triton_kernels(internal)
```

</details>

## 🔬 3-Tier Verification

Inspired by [Liger Kernel](https://github.com/linkedin/Liger-Kernel) and industry practices from [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang).

| Tier | What | Threshold | Time | Command |
|------|------|-----------|------|---------|
| **1. Kernel** | Per-kernel numerical correctness (atol/rtol) | bf16: 0.05, fp16: 1e-3 | ~5s | `make test` (125 tests) |
| **2. Model** | Layer-by-layer cosine similarity | > 0.95 at layers 0,7,14,21,27 | ~15s | `make test-parity` |
| **3. E2E** | Output quality distribution (UTMOS, CER, Speaker Sim) | See below | 5-30min | `make eval-fast` |

### Tier 3 Thresholds

Each model generates independently, then task-level metrics are compared via distribution analysis (not pair-level waveform comparison — stochastic TTS makes this unreliable).

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| UTMOS delta | \|mean\| < 0.3 | F5-TTS independent generation variance |
| UTMOS floor | Both > 2.5 | Absolute quality lower bound |
| CER delta | \|mean\| < 0.05 | SGLang 1-5% tolerance |
| Speaker Similarity | mean > 0.75 | Qwen3-TTS SIM > 0.79 |
| Mann-Whitney U | p > 0.05 (full mode) | Non-parametric distribution equivalence |

### Running Verification

```bash
make test          # Tier 1: 125 tests
make test-parity   # Tier 2: Model parity (GPU required)
make verify        # Tier 1 + 2
make eval-fast     # Tier 3: Fast (~5min, whisper-small, 1 run/utterance)
make eval-full     # Tier 3: Full (~30min, whisper-large-v3, 3 runs, Mann-Whitney)
make verify-all    # All 3 tiers
```

### 📋 Latest Results

✅ **Tier 1**: 125/125 PASS

✅ **Tier 2**: All layers > 0.95 cosine similarity

| Layer | Cosine Sim |
|-------|-----------|
| L0 | 0.999995 |
| L7 | 0.999977 |
| L14 | 0.999852 |
| L21 | 0.999177 |
| L27 | 0.997900 |
| Output | 0.997156 |

> FP accumulation naturally decreases similarity across 28 layers — this is expected behavior for fused kernels that change operation order.

## 📊 Benchmarks

<!-- BENCH:SUMMARY:START -->
> __Hybrid (Faster+Triton)__ achieves __4.7x__ faster inference than PyTorch baseline at equivalent VRAM on RTX 5090.
<!-- BENCH:SUMMARY:END -->

### 🏗️ Optimization Modes

```mermaid
graph TD
    A["Base (PyTorch eager)"] -->|"+Triton kernel fusion"| B["Triton (~1.1x)"]
    A -->|"+CUDA Graph + Static Cache"| C["Faster (~3.6x)"]
    C -->|"+Triton kernel fusion"| D["Hybrid (~4.7x)"]

    style D fill:#f96,stroke:#333,stroke-width:2px,color:#000
```

```bash
make bench-kernels  # Per-kernel micro-benchmarks (PyTorch vs Triton)
make bench-e2e      # End-to-end inference (all runners)
make bench          # Both
make profile        # torch.profiler trace
```

<details>
<summary><b>Hardware & Methodology</b></summary>

| Item | Spec |
|------|------|
| GPU | NVIDIA RTX 5090 (Blackwell, sm_120, 32GB) |
| CUDA | 12.8 |
| PyTorch | nightly (cu128) |
| Triton | 3.2.0 |
| Model | Qwen3-TTS-12Hz-1.7B (1.7B params) |
| OS | WSL2 (Linux 5.15) |
| Python | 3.12 |
| Dtype | bfloat16 |
| Batch Size | 1 |

**Kernel benchmarks**: `triton.testing.do_bench()`, batch=1, seq_len=512, hidden=2048.
**E2E benchmarks**: `torch.cuda.Event` timing, 3 warmup + 20 measured runs per text.
RTF (Real-Time Factor) = audio_duration / generation_time. RTF > 1 means faster-than-real-time.

</details>

### ⚡ Kernel Micro-Benchmarks

<!-- BENCH:KERNEL:START -->
> RTX 5090, bf16, batch=1, seq_len=512, hidden=2048. Run `make bench-kernels` to reproduce.

| Kernel | PyTorch (us) | Triton (us) | Speedup | Compile (s) | HBM Savings |
|--------|:------------:|:-----------:|:-------:|:-----------:|:-----------:|
| RMSNorm | 39.4 | **6.7** | **5.87x** | 0.61 | 4→1 trips |
| SwiGLU | 19.6 | **15.0** | **1.31x** | 0.00 | 3→1 trips |
| M-RoPE | 348.8 | **38.8** | **9.00x** | 0.00 | In-place |
| Fused Norm+Residual | 41.2 | **8.3** | **4.97x** | 0.00 | 2→1 kernels |
<!-- BENCH:KERNEL:END -->

### 🏎️ E2E Inference

<!-- BENCH:E2E:START -->
> RTX 5090, bf16, 2 texts (ko + en), 3 warmup + 20 runs each. Run `make bench-e2e` to reproduce.

| Mode | Load Time | Latency (ko) | Latency (en) | RTF (ko) | RTF (en) | Speedup | Peak VRAM |
|------|----------|:------------:|:------------:|:--------:|:--------:|:-------:|:---------:|
| Base (PyTorch) | 9.9s | 3,902 ms | 5,511 ms | 1.00x | 0.82x | 1.0x | 4.01 GB |
| Triton | 6.7s | 3,767 ms | 3,747 ms | 1.22x | 1.27x | 1.1x | 4.04 GB |
| Faster | 5.1s | 1,199 ms | 1,247 ms | 3.60x | 3.50x | 3.6x | 4.32 GB |
| __Hybrid (Faster+Triton)__ | 7.1s | **919 ms** | **1,047 ms** | **4.39x** | **4.26x** | **4.7x** | 4.30 GB |
<!-- BENCH:E2E:END -->

### 🎵 Audio Quality (Tier 3)

<!-- BENCH:QUALITY:START -->
Each runner generates speech independently, then quality distributions are compared against Base.

| Runner | UTMOS | CER | Speaker Sim | Status |
|--------|:-----:|:---:|:-----------:|:------:|
| **Base** (ref) | 3.12 ± 0.54 | 0.16 ± 0.14 | - | - |
| **Triton** | 3.29 ± 0.46 | 0.18 ± 0.14 | 0.76 | PASS |
| **Faster** | 3.29 ± 0.60 | 0.22 ± 0.26 | 0.75 | FAIL |
| **Hybrid** | 3.30 ± 0.85 | 0.20 ± 0.14 | 0.77 | PASS |

> Faster FAIL reason: CER delta 0.057 > threshold 0.05. This is due to variance in fast mode (1 run/sentence, whisper-small). Expected to PASS in full mode (3 runs/sentence, whisper-large-v3, Mann-Whitney U test).

Run `make eval-fast` to reproduce.
<!-- BENCH:QUALITY:END -->

> **Disclaimer**: Benchmarks measured on a single RTX 5090. Results vary with GPU model, driver version, system load, and input text length. Run `make bench` on your hardware for accurate numbers.

## 📁 Project Structure

```
qwen3-tts-triton/
├── src/
│   └── qwen3_tts_triton/           # PyPI package
│       ├── __init__.py              # Public API + __version__
│       ├── py.typed                 # PEP 561 type marker
│       ├── kernels/                 # Triton GPU kernels
│       │   ├── rms_norm.py          # Fused RMSNorm
│       │   ├── swiglu.py            # Fused SwiGLU
│       │   ├── rope.py              # Fused M-RoPE
│       │   └── fused_norm_residual.py # Fused Norm+Residual
│       └── models/                  # Model runners & patching
│           ├── patching.py          # Monkey-patch logic
│           ├── base_runner.py       # Standard PyTorch
│           ├── triton_runner.py     # Triton-optimized
│           ├── faster_runner.py     # faster-qwen3-tts wrapper
│           └── triton_faster_runner.py # Hybrid (faster + Triton)
├── tests/                           # Verification tests
│   ├── kernels/                     # Tier 1: Kernel correctness
│   └── test_model_parity.py         # Tier 2: Model parity
├── benchmark/                       # Benchmarking suite
├── ui/                              # Streamlit dashboard
├── docs/                            # Documentation
├── pyproject.toml                   # Project config (UV + hatchling)
├── uv.lock                          # Locked dependencies
└── Makefile                         # Development commands
```

## 🛠️ Development

```bash
make format      # Ruff formatting
make lint        # Ruff linting
make lint-fix    # Ruff auto-fix
make test        # pytest (Tier 1)
make test-cov    # pytest + coverage
make check       # lint + test
make pre-commit  # All pre-commit hooks
make clean       # Clear caches
```

### 🧠 Qwen3-TTS Talker Architecture

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-TTS-12Hz-1.7B-CustomVoice |
| Hidden Size | 2048 |
| Attention Heads | 16 (GQA, kv_heads=8) |
| Head Dim | 128 |
| Intermediate Size | 6144 |
| Layers | 28 |
| RMS Norm Eps | 1e-6 |
| Position Encoding | M-RoPE (sections=[24,20,20]) |
| Activation | SwiGLU |

## 🔄 Compatibility

### 🎤 Voice Modes by Runner

| Feature | Base | Triton | Faster | Hybrid |
|---------|:----:|:------:|:------:|:------:|
| Custom Voice | Yes | Yes | Yes | Yes |
| Voice Cloning | Yes | Yes | Yes | Yes |
| Voice Design | -- | -- | Yes | Yes |
| Streaming | -- | -- | Yes | Yes |
| Dynamic Shape | Yes | Yes | Yes | Yes |
| bfloat16 / float16 | Yes | Yes | Yes | Yes |

### 💻 Platform Support

| Platform | Supported |
|----------|-----------|
| Linux | Yes |
| Windows WSL2 | Yes |

## 🗺️ TODO

- [ ] Docker deployment
- [ ] [SageAttention](https://github.com/thu-ml/SageAttention) integration — low-bit attention for further speedup
- [ ] [ComfyUI-Qwen3-TTS-Triton](https://github.com/newgrit1004/ComfyUI-Qwen3-TTS-Triton) — ComfyUI custom node
- [ ] Multi-GPU architecture testing (A100, H100, RTX 4090, etc.)

## 📄 License

Apache-2.0

## 🙏 Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Base TTS model
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) — Triton kernel design patterns and verification methodology
- [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) — CUDA Graph optimization for Hybrid mode
- [Triton](https://github.com/triton-lang/triton) — GPU kernel compiler

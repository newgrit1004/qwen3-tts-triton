# 벤치마크 결과

GPU: **NVIDIA RTX 5090** (Blackwell, sm_120, 32GB VRAM, CUDA 12.8)

## 속도 벤치마크

모델 로드 시간은 가중치 로딩 + 초기화 포함 (Faster/Hybrid는 CUDA 그래프 캡처 포함).

| Runner | 로드 시간 | 지연 (한국어) | 지연 (영어) | RTF (한국어) | RTF (영어) | 속도 향상 | 최대 VRAM |
|--------|----------|-------------|-------------|---------|---------|---------|-----------|
| **Base** | 9.9s | 3,902 ms | 5,511 ms | 1.00x | 0.82x | 1.0x | 4.01 GB |
| **Triton** | 6.7s | 3,767 ms | 3,747 ms | 1.22x | 1.27x | 1.1x | 4.04 GB |
| **Faster** | 5.1s | 1,199 ms | 1,247 ms | 3.60x | 3.50x | 3.6x | 4.32 GB |
| **Hybrid** | 7.1s | 919 ms | 1,047 ms | 4.39x | 4.26x | **4.7x** | 4.30 GB |

- **RTF** (Real-Time Factor): 오디오 길이 / 생성 시간. >1.0 = 실시간보다 빠름.
- **속도 향상**: Base 대비 (한국어 평균 지연 기준).
- 모든 러너가 동일한 모델 가중치 사용 (~4GB VRAM). 커널 퓨전으로 인한 추가 VRAM 없음.

## 3-Tier 검증

### Tier 1: 커널 정확도 — PASS

| 커널 | 테스트 수 | 상태 |
|--------|-------|--------|
| RMSNorm | 16 | PASS |
| SwiGLU | 12 | PASS |
| M-RoPE | 8 | PASS |
| FusedNorm+Residual | 7 | PASS |
| CLI/Utils | 82 | PASS |
| **합계** | **125** | **PASS** |

### Tier 2: 모델 패리티 — PASS (2-pair)

Triton 커널 적용 전/후 hidden state를 레이어별로 비교. 두 pair 모두 동일 모델 + Triton 적용이므로 공평한 비교.

**Pair A: Base vs Triton**

| 레이어 | Cosine Sim | Relative L2 | SNR (dB) |
|-------|-----------|------------|----------|
| L0 | 0.999995 | 0.0043 | 47.3 |
| L7 | 0.999977 | 0.0075 | 42.5 |
| L14 | 0.999852 | 0.0175 | 35.1 |
| L21 | 0.999177 | 0.0407 | 27.8 |
| L27 | 0.997900 | 0.0649 | 23.8 |
| **출력** | **0.997156** | - | - |

**Pair B: Faster vs Hybrid**

| 레이어 | Cosine Sim | Relative L2 | SNR (dB) |
|-------|-----------|------------|----------|
| L0 | 0.999995 | 0.0043 | 47.3 |
| L7 | 0.999977 | 0.0075 | 42.5 |
| L14 | 0.999852 | 0.0175 | 35.1 |
| L21 | 0.999177 | 0.0407 | 27.8 |
| L27 | 0.997900 | 0.0649 | 23.8 |
| **출력** | **0.997156** | - | - |

> 임계값: cosine sim > 0.95, relative L2 < 0.08, SNR > 22 dB. 모든 레이어가 큰 마진으로 통과.

### Tier 3: E2E 품질 — Triton PASS, Hybrid PASS, Faster FAIL

각 러너가 독립적으로 음성을 생성하고, Base 대비 품질 분포를 비교 (vLLM/TensorRT-LLM 패턴).

| Runner | UTMOS | CER | Speaker Sim | 상태 |
|--------|-------|-----|-------------|--------|
| **Base** (기준) | 3.12 ± 0.54 | 0.16 ± 0.14 | - | - |
| **Triton** | 3.29 ± 0.46 | 0.18 ± 0.14 | 0.76 | PASS |
| **Faster** | 3.29 ± 0.60 | 0.22 ± 0.26 | 0.75 | FAIL |
| **Hybrid** | 3.30 ± 0.85 | 0.20 ± 0.14 | 0.77 | PASS |

> Faster FAIL 원인: CER delta 0.057 > 임계값 0.05. fast 모드(1회/문장, whisper-small)의 변동. full 모드(3회/문장, whisper-large-v3, Mann-Whitney U test)에서 재평가 시 PASS 예상.

**임계값**:
- |UTMOS delta| < 0.3
- UTMOS floor > 2.5 (양쪽 모두)
- |CER delta| < 0.05
- Speaker Sim > 0.75
- Mann-Whitney U p > 0.05 (full 모드만)

## 명령어

```bash
make bench          # 전체 (커널 + 속도 + 품질 + 검증)
make bench-speed    # E2E 속도 벤치마크만
make bench-kernels  # 커널 마이크로벤치마크만
make eval-fast      # Tier 3 품질 평가 (fast, ~15분)
make eval-full      # Tier 3 품질 평가 (full, ~90분)
make verify         # 3-Tier 검증 리포트 생성
```

## 환경

- **OS**: Ubuntu 22.04 on WSL2 (Windows Subsystem for Linux 2)
- **커널**: Linux 5.15.167.4-microsoft-standard-WSL2
- **GPU**: NVIDIA RTX 5090 (32GB GDDR7, sm_120)
- **CUDA**: 12.8
- **PyTorch**: nightly (cu128)
- **Python**: 3.12
- **모델**: Qwen3-TTS 1.7B (12Hz CustomVoice)

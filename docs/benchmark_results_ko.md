# 벤치마크 결과

> **v0.2.0** — 2026-04-13 | GPU: **NVIDIA RTX 5090** (Blackwell, sm_120, 32GB VRAM, CUDA 12.8)

## 속도 벤치마크

모델 로드 시간은 가중치 로딩 + 초기화 포함 (Faster/Hybrid는 CUDA 그래프 캡처 포함).

| Runner | 로드 시간 | 지연 (한국어) | 지연 (영어) | RTF (한국어) | RTF (영어) | Base 대비 | 최대 VRAM |
|--------|-----------|-------------|-------------|---------|---------|-----------|-----------|
| **Base** | 17.5s | 4,615 ms | 5,081 ms | 0.88x | 0.90x | 1.0x | 4.03 GB |
| **Base+TQ** | 8.3s | 9,030 ms | 5,745 ms | 0.82x | 0.79x | 0.7x | 4.07 GB |
| **Triton** | 7.9s | 4,130 ms | 4,462 ms | 1.00x | 1.00x | 1.1x | 4.03 GB |
| **Triton+TQ** | 7.4s | 8,045 ms | 5,877 ms | 0.93x | 0.88x | 0.7x | 4.09 GB |
| **Faster** | 9.2s | 1,136 ms | 1,265 ms | 3.49x | 3.52x | 4.0x | 4.28 GB |
| **Hybrid** | **6.0s** | **886 ms** | 1,042 ms | 4.20x | **4.26x** | **5.0x** | 4.32 GB |
| **Hybrid+TQ** | 6.5s | 944 ms | **1,032 ms** | **4.27x** | 4.25x | 4.9x | 4.33 GB |

- **RTF** (Real-Time Factor): 오디오 길이 / 생성 시간. >1.0 = 실시간보다 빠름.
- 모든 러너가 동일한 모델 가중치 사용 (~4GB VRAM). 커널 퓨전으로 인한 추가 VRAM 없음.
- Triton/Triton+TQ/Hybrid/Hybrid+TQ는 기본 partial patch range `[0, 24)` 설정을 사용하며, 마지막 4개 decoder 레이어는 발음 안정성을 위해 PyTorch로 남겨둡니다.

## 커널 마이크로벤치마크

| 커널 | PyTorch (µs) | Triton (µs) | 속도 향상 | PT 메모리 (MB) | TR 메모리 (MB) |
|--------|-------------|------------|---------|------------|------------|
| RMSNorm | 40.91 | 7.42 | **5.51x** | 266.01 | 260.00 |
| SwiGLU | 19.43 | 16.00 | **1.21x** | 280.00 | 274.00 |
| M-RoPE | 367.90 | 37.29 | **9.86x** | 264.25 | 264.75 |
| FusedNorm+Residual | 40.57 | 9.03 | **4.50x** | 270.01 | 264.00 |

## 3-Tier 검증

### Tier 1: 커널 + CPU 회귀 스위트 — PASS (197개 테스트)

RTX 5090 / WSL2 (Ubuntu 22.04) / PyTorch 2.10.0+cu128 기준: **`make test` 약 48초 소요**.

| 범위 | 테스트 수 | 상태 |
|--------|-------|--------|
| RMSNorm 커널 | 25 | PASS |
| SwiGLU 커널 | 21 | PASS |
| M-RoPE 커널 | 18 | PASS |
| FusedNorm+Residual 커널 | 14 | PASS |
| Fused dequant 커널 | 22 | PASS |
| TurboQuant 커널 | 56 | PASS |
| 커널 유틸리티 | 12 | PASS |
| Partial patching 회귀 테스트 | 18 | PASS |
| 릴리스 문서/Makefile/벤치 가드 | 11 | PASS |
| **합계** | **197** | **PASS** |

### Tier 2: 모델 패리티 — PASS (2-pair)

RTX 5090 / WSL2 기준 `make test-parity` 약 46초 소요 (모델 로딩 포함).

Triton 커널 적용 전/후 hidden state를 레이어별로 비교. 모든 pair가 동일 모델 + 커널 적용이므로 공평한 비교.

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

### Tier 3: E2E 품질 (full 모드) — 최적화 러너 6개 중 4개 PASS

공식 릴리스 품질 수치는 full 모드(문장당 3회 생성) 기준이며, 각 러너를 Base와 비교합니다.

| Runner | UTMOS | CER | Speaker Sim | 상태 |
|--------|-------|-----|-------------|--------|
| **Base** (기준) | 3.40 ± 0.78 | 0.04 ± 0.06 | - | 기준 |
| **Base+TQ** | 3.17 ± 0.81 | 0.42 ± 2.02 | 0.82 | **FAIL** |
| **Triton** | 3.40 ± 0.76 | 0.04 ± 0.07 | 0.85 | **PASS** |
| **Triton+TQ** | 3.04 ± 0.83 | 0.43 ± 1.49 | 0.83 | **FAIL** |
| **Faster** | 3.42 ± 0.75 | 0.04 ± 0.04 | 0.83 | **PASS** |
| **Hybrid** | 3.38 ± 0.78 | 0.04 ± 0.06 | 0.83 | **PASS** |
| **Hybrid+TQ** | 3.32 ± 0.78 | 0.05 ± 0.07 | 0.83 | **PASS** |

> 릴리스 주의사항:
> `base+tq`는 full 릴리스 게이트에서 `CER delta 0.3801 > 0.05`, `Mann-Whitney p=0.0340 < 0.05`로 실패합니다.
> `triton+tq`는 full 릴리스 게이트에서 `UTMOS delta 0.3565 > 0.3`, `CER delta 0.3865 > 0.05`, `Mann-Whitney p=0.0015 < 0.05`로 실패합니다.
> fast 모드는 스모크 체크에는 유용하지만, 릴리스 기준은 full 모드입니다.

**임계값**:
- |UTMOS delta| < 0.3
- UTMOS floor > 2.5 (양쪽 모두)
- |CER delta| < 0.05
- Speaker Sim > 0.75
- Mann-Whitney U p > 0.05 (full 모드만)

## 명령어

```bash
make bench          # 기본 세트 (커널 + 속도 + 빠른 품질 평가 + 리포트)
make bench-speed    # E2E 속도 벤치마크만
make bench-kernels  # 커널 마이크로벤치마크만
make eval-fast      # Tier 3 품질 평가 (fast, ~15분)
make eval-full      # Tier 3 품질 평가 (full, ~80분)
make verify         # 기존 Tier 3 결과를 포함한 3-Tier 리포트
make verify-all     # eval-full 실행 후 3-Tier 리포트 생성
```

## 환경

- **OS**: Ubuntu 22.04 on WSL2 (Windows Subsystem for Linux 2)
- **커널**: Linux 5.15.167.4-microsoft-standard-WSL2
- **GPU**: NVIDIA RTX 5090 (32GB GDDR7, sm_120)
- **CUDA**: 12.8
- **PyTorch**: nightly (cu128)
- **Triton**: 3.2.0
- **Transformers**: 4.57.3
- **Python**: 3.12
- **모델**: Qwen3-TTS 1.7B (12Hz CustomVoice)

## 기여자 벤치마크: RTX 4090

기여자 **tantara**가 제공한 별도 커뮤니티 벤치마크입니다. 위의 RTX 5090
공식 `v0.2.0` 릴리스 표와는 분리된 참고 결과입니다.

### 환경

| 구성 요소 | 값 |
|-----------|----|
| GPU | NVIDIA GeForce RTX 4090 |
| PyTorch | 2.10.0+cu128 |
| Triton | 3.6.0 |
| CUDA | 12.8 |
| Driver | 550.144.03 |

### 커널 마이크로벤치마크

`bf16`, `batch=1`, `seq_len=512`, `hidden=2048`

| 커널 | PyTorch (us) | Triton (us) | 속도 향상 | 컴파일 (s) |
|------|--------------|-------------|-----------|------------|
| RMSNorm | 38.6 | 8.3 | 4.66x | 0.48 |
| SwiGLU | 32.1 | 26.6 | 1.21x | 0.13 |
| M-RoPE | 123.7 | 42.8 | 2.89x | 0.17 |
| Fused Norm+Residual | 42.1 | 12.1 | 3.49x | 0.16 |

### E2E 추론

`bf16`, 2개 텍스트 (`ko` + `en`), warmup 3회 + 측정 20회

| 모드 | 지연 시간 (ko) | 지연 시간 (en) | RTF (ko) | RTF (en) | 최대 VRAM |
|------|----------------|----------------|----------|----------|-----------|
| Base (PyTorch) | 2,546 ms | 2,878 ms | 1.59x | 1.60x | 4.01 GB |
| Compile (`torch.compile`) | 2,685 ms | 2,685 ms | 1.60x | 1.60x | 3.98 GB |
| Triton | 2,079 ms | 2,460 ms | 1.87x | 1.88x | 3.98 GB |
| Faster | 922 ms | 1,016 ms | 4.20x | 4.21x | 4.23 GB |
| Hybrid (Faster+Triton) | 770 ms | 881 ms | 5.16x | 5.19x | 4.27 GB |

### 메모

- Hybrid 모드는 한국어 기준 `5.16x` RTF로 실시간을 충분히 넘깁니다.
- `torch.compile` (`inductor`, `reduce-overhead`)은 eager Base 대비 E2E 이득이 거의 없었습니다. 이 프로젝트에서는 autoregressive decoding 특성상 컴파일 이득이 제한적입니다.
- Triton 커널 퓨전만으로는 Base 대비 약 `1.2x` 수준이고, 큰 폭의 이득은 Faster/Hybrid의 CUDA Graph 경로에서 나옵니다.
- `make bench-kernels`, `make bench-e2e`로 재현했습니다.
- `TorchCompileRunner` 참고: <https://gist.github.com/tantara/b23b717c7bf252b7e897e1adb02c25b5>

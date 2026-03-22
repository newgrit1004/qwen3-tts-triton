# TTS 품질 평가 파이프라인

## 개요

Triton 커널 최적화 후 Base와 Triton 출력 간 **인지적 품질 동등성**을 자동으로 검증하는 파이프라인입니다.

IEEE 754 부동소수점 비결합성(FPNA)과 28 decoder layer 누적으로 인해, 같은 시드/greedy decoding에서도 토큰 분기가 발생합니다. 이는 **업계 표준에서 정상**이며, F5-TTS, CosyVoice, FishSpeech, Liger Kernel 등 모든 주요 프로젝트가 텐서 비교 대신 task-level 메트릭을 사용합니다.

---

## 변경 파일 요약

| 파일 | 유형 | 설명 |
|------|------|------|
| `benchmark/eval_config.py` | 신규 | 평가 설정, parity 임계값, 테스트 문장 (zh/ko 24개) |
| `benchmark/eval_quality.py` | 신규 | 6종 메트릭 품질 평가 파이프라인 (CLI 포함) |
| `benchmark/bench_e2e.py` | 수정 | CUDA event timing, warmup, 통계 리포트 추가 |
| `models/__init__.py` | 수정 | `get_runner_class()` 룩업 함수 추가 |
| `pyproject.toml` | 수정 | `eval` optional dependency 그룹 추가 |
| `Makefile` | 수정 | `eval-fast`, `eval-full`, `bench-speed` 타겟 추가 |

---

## 품질 메트릭

### Tier 1 — Parity (필수, 통과 기준)

| 메트릭 | 측정 대상 | 도구 | 임계값 |
|--------|----------|------|--------|
| **CER** | 발음/내용 정확도 | Whisper ASR + jiwer | delta < 0.5% |
| **UTMOS** | 음성 자연스러움 (MOS) | torch.hub SpeechMOS | delta < 0.15 |
| **Speaker Similarity** | 화자 음색 보존 | Resemblyzer cosine | > 0.90 |

### Tier 2 — Quality (보조, 심층 검증)

| 메트릭 | 도구 | 임계값 |
|--------|------|--------|
| **PESQ-WB** | pesq (16kHz) | > 3.5 |
| **STOI** | pystoi | > 0.95 |
| **MCD** | mel-cepstral-distance (DTW) | < 4.0 dB |

---

## 사용법

### 설치

```bash
uv sync --extra eval
```

### 평가 실행

```bash
# 빠른 CI 평가 (~5분, whisper-small, 언어별 5문장)
make eval-fast

# 전체 PR 게이트 평가 (~30분, whisper-large-v3, 전체 문장)
make eval-full

# 특정 WAV 쌍 비교
uv run python -m benchmark.eval_quality pair \
    --ref outputs/base.wav --opt outputs/triton.wav \
    --text "원본 텍스트" --asr-model small
```

### 벤치마크 (속도 측정)

```bash
# CUDA event timing + 통계 리포트 (mean/std/p50/p95/p99)
make bench-speed

# 커스텀 설정
uv run python -m benchmark.bench_e2e --warmup 5 --repeat 30 --output my_results.json
```

---

## bench_e2e.py 개선 사항

### 기존 문제점

- warmup 0회 → JIT 컴파일/GPU frequency ramp 미반영
- `time.perf_counter()` → CPU 시계 (GPU 비동기 미반영)
- 1회 측정 → 통계적 의미 없음

### 개선 내용

| 항목 | 기존 | 개선 |
|------|------|------|
| 타이밍 | `time.perf_counter()` | `torch.cuda.Event` (GPU 정밀 측정) |
| Warmup | 없음 | `--warmup N` (기본 3회) |
| 반복 | 1회 | `--repeat N` (기본 20회) |
| 통계 | 단일 값 | mean, std, min, max, p50, p95, p99 |
| 출력 | 고정 경로 | `--output` 옵션 |

---

## 결과 출력 형식

### 평가 결과 (JSON)

```json
{
  "ref_runner": "base",
  "opt_runner": "triton",
  "num_sentences": 10,
  "cer_delta_mean": 0.0023,
  "utmos_delta_mean": 0.08,
  "speaker_sim_mean": 0.95,
  "pesq_wb_mean": 3.8,
  "stoi_mean": 0.97,
  "mcd_db_mean": 2.1,
  "pass_rate": 1.0,
  "overall_verdict": "PASS"
}
```

### 벤치마크 결과 (테이블)

```
Runner          Lang  Mean(ms)    Std      P50      P95      P99    RTF   VRAM
Base            ko      1234.5   45.2   1230.1   1310.5   1345.2   8.50  12.30
Turbo           ko       890.3   32.1    885.7    950.2    970.1  11.80  12.30
```

---

## 아키텍처

```
benchmark/
├── eval_config.py      # 설정 + 테스트 문장
├── eval_quality.py     # 평가 파이프라인
│   ├── compute_cer()           # Whisper ASR → CER
│   ├── compute_utmos()         # torch.hub UTMOS
│   ├── compute_speaker_similarity()  # Resemblyzer
│   ├── compute_pesq()          # PESQ-WB
│   ├── compute_stoi()          # STOI
│   ├── compute_mcd()           # MCD (DTW)
│   ├── evaluate_pair()         # 단일 쌍 평가
│   └── evaluate_batch()        # 배치 평가 + 통계
└── bench_e2e.py        # E2E 벤치마크
    ├── CUDA Event Timing
    ├── Warmup + 반복 측정
    └── 통계 리포트 (p50/p95/p99)
```

---

## 업계 참조

| 프로젝트 | 검증 방식 | 메트릭 |
|----------|----------|--------|
| F5-TTS | Seed-TTS-Eval | WER + Speaker Sim |
| CosyVoice | 오프라인 vs 스트리밍 | WER + SS + NMOS |
| FishSpeech | Seed-TTS-Eval | CER + Speaker Distance |
| Liger Kernel | 3-tier 검증 | atol/rtol + loss curve |
| vLLM | perplexity 동등성 | PPL within 6% |

> 어떤 프로젝트도 텐서 수준 비교를 하지 않음 — 모두 task-level 메트릭 사용

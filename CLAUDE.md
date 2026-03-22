# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

Qwen3-TTS Triton 커널 최적화 프로젝트. Transformer 기반 TTS 모델(Qwen3-TTS 1.7B)의 추론 속도를 Triton 커널 퓨전으로 극대화. Liger Kernel(LinkedIn) 참조.

**GPU**: RTX 5090 (Blackwell, sm_120, 32GB VRAM, CUDA 12.8)
**제약**: PyTorch cu128 nightly, 추가 VRAM 사용 금지 (커널 퓨전만)

---

## Claude Code 확장 기능

이 프로젝트는 [everything-claude-code](https://github.com/affaan-m/everything-claude-code)의 철학을 적용하여 Claude Code와의 협업을 최적화했습니다.

### 디렉토리 구조

```
.claude/
├── agents/           # 전문화된 AI 에이전트
│   ├── planner.md          # 구현 계획 수립
│   ├── code-reviewer.md    # 자동화된 코드 리뷰
│   ├── code-quality-analyzer.md # 복잡도/데드코드 분석
│   ├── tdd-guide.md        # TDD 워크플로우 가이드
│   ├── architect.md        # 아키텍처 설계
│   ├── security-reviewer.md # 보안 취약점 검토
│   ├── refactor-cleaner.md # 코드 리팩토링
│   ├── build-error-resolver.md # 빌드 오류 해결
│   ├── e2e-runner.md       # E2E 테스트 실행기
│   └── doc-updater.md      # 문서 동기화
├── commands/         # 슬래시 명령어
│   ├── tdd.md              # /tdd - TDD 워크플로우
│   ├── code-review.md      # /code-review - 코드 리뷰
│   ├── plan.md             # /plan - 구현 계획
│   ├── refactor-clean.md   # /refactor-clean - 리팩토링
│   ├── security-check.md   # /security-check - 보안 검사
│   ├── build-fix.md        # /build-fix - 빌드 오류 수정
│   ├── e2e.md              # /e2e - E2E 테스트 생성
│   ├── learn.md            # /learn - 패턴 학습 추출
│   ├── complexity.md       # /complexity - 인지 복잡도 분석
│   ├── dead-code.md        # /dead-code - 데드 코드 탐지
│   ├── code-quality.md     # /code-quality - 통합 품질 분석
│   ├── ralph-loop.md       # /ralph-loop - 지속 실행 모드
│   └── cancel-ralph.md     # /cancel-ralph - 루프 취소
├── rules/            # 필수 규칙
│   ├── coding-style.md     # Python 코딩 스타일
│   ├── testing.md          # 테스트 규칙
│   ├── security.md         # 보안 규칙
│   ├── git-workflow.md     # Git 워크플로우
│   ├── performance.md      # 성능 가이드라인
│   └── ralph-loop.md       # Ralph Loop 규칙
├── skills/           # 도메인 지식
│   ├── python-patterns.md  # Python 패턴
│   ├── uv-ruff-ty-workflow.md # UV/Ruff/Ty 워크플로우
│   ├── project-structure.md   # 프로젝트 구조
│   ├── complexity-patterns.md # 복잡도 감소 패턴
│   ├── ralph-workflow.md   # Ralph 워크플로우
│   └── autonomous-agent.md # 자율 에이전트 워크플로우
├── hooks/            # 자동화 훅
│   ├── hooks.json          # 훅 설정
│   └── README.md           # 훅 사용법
└── mcp-configs/      # MCP 서버 설정
    ├── README.md           # MCP 사용 가이드
    ├── github.json         # GitHub 통합
    ├── database.json       # 데이터베이스 연결
    ├── filesystem.json     # 파일시스템 확장
    └── memory.json         # 영속적 메모리
```

### 사용 가능한 명령어

| 명령어 | 설명 |
|--------|------|
| `/plan <기능>` | 구현 전 계획 수립 |
| `/tdd <기능>` | TDD 워크플로우 시작 |
| `/code-review` | 변경사항 코드 리뷰 |
| `/refactor-clean` | 코드 품질 개선 |
| `/security-check` | 보안 취약점 검사 |
| `/build-fix` | 빌드 오류 진단 및 수정 |
| `/e2e <워크플로우>` | E2E 테스트 생성 및 실행 |
| `/learn` | 세션 중 발견된 패턴 추출 |
| `/complexity [경로]` | 인지 복잡도 분석 (complexipy) |
| `/dead-code [경로]` | 데드 코드 탐지 (skylos) |
| `/code-quality [경로]` | 통합 코드 품질 분석 |
| `/ralph-loop <작업>` | 완료까지 지속 실행 모드 |
| `/cancel-ralph` | Ralph Loop 취소 |
| `/autonomous-run <tasks-file>` | 다중 세션 자율 에이전트 실행 |

### 개발 워크플로우

```
1. /plan         → 계획 수립 및 승인
2. /tdd          → RED-GREEN-REFACTOR
3. /code-review  → 품질 검토
4. git commit    → pre-commit 자동 실행
```

### Ralph Loop 모드

복잡한 작업이나 "끝날 때까지 멈추지 마"라고 할 때 사용:

```
/ralph-loop <작업 설명>
```

- 모든 작업이 완료될 때까지 자동으로 계속
- 할 일 목록 자동 생성 및 추적
- 각 작업 완료 후 다음 작업으로 자동 진행
- 취소하려면 `/cancel-ralph` 또는 "중단" 입력

### 자율 에이전트 모드 (36시간+ 연속 실행)

컨텍스트 윈도우 한계를 넘는 장기 파이프라인에 사용:

```bash
# 태스크 파일로 실행
/autonomous-run tasks.json

# 또는 Python으로 직접 실행
python -m src.autonomous_agent.run --tasks-file tasks.json
```

**핵심 개념:**
- 세션 간 `progress.json`으로 상태 영속화
- 매 세션마다 Fresh 컨텍스트로 토큰 한계 우회
- 중단 후 같은 명령으로 재개 가능

| 모드 | 범위 | 사용 사례 |
|------|------|----------|
| Ralph Loop | 세션 내 | 복잡한 단일 작업 |
| Autonomous | 세션 간 | 장기 자동화 (36시간+) |

**설치:**
```bash
# Claude Code SDK 설치
uv add --optional agent claude-code-sdk

# OAuth 토큰 설정
export CLAUDE_CODE_OAUTH_TOKEN='your-token'
```

---

## 개발 환경 설정

### 초기 환경 구성
```bash
# UV 설치 (미설치 시)
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# 또는
pip install uv  # 모든 플랫폼

# 프로젝트 설정
make setup
```

`make setup` 명령어 수행 내역:
- UV를 사용한 모든 의존성 설치 (`uv sync --all-extras --dev`)
- Git commit 메시지 템플릿 설정
- Pre-commit hooks 설치

### UV 기반 워크플로우
- **가상환경 수동 활성화 불필요** - UV가 자동 처리
- 모든 명령어는 `uv run` 접두사 사용
- Lock 파일(`uv.lock`)로 재현 가능한 빌드 보장
- 빠른 병렬 의존성 해결

### 의존성 그룹
```bash
uv sync                 # 코어 + UI (torch, triton, transformers, streamlit, plotly, pynvml)
uv sync --extra dev     # + 개발 도구 (ruff, pytest, pre-commit)
uv sync --extra eval    # + 품질 평가 (jiwer, whisper, resemblyzer, scipy 등)
uv sync --extra faster  # + faster-qwen3-tts
uv sync --extra all     # 전부 (dev + faster + eval)
```

---

## 개발 명령어

### 코드 품질 관리
```bash
# Ruff 포맷팅 (Black + isort 대체)
make format DIR=<경로>  # 기본값: DIR=.

# Ruff 린팅 (flake8 + pylint 대체)
make lint               # 체크만 수행
make lint-fix           # 자동 수정 포함

# Ty 타입 체크 (mypy/pyright보다 10-20배 빠름)
make typecheck          # 전체 프로젝트 체크

# pytest 테스트 실행
make test               # Tier 1 커널 테스트 (43 tests)
make test-cov           # 커버리지 포함
make test-parity        # Tier 2 모델 패리티 테스트 (GPU, ~15초)
make verify             # Tier 1+2 통합 검증
make verify-all         # Tier 1+2+3 전체 검증

# 모든 pre-commit 체크 수동 실행
make pre-commit

# 전체 체크 (lint + typecheck + test)
make check

# 인지 복잡도 분석 (complexipy)
make complexity DIR=<경로>  # 기본값: DIR=.

# 데드 코드 탐지 (skylos)
make dead-code DIR=<경로>   # 기본값: DIR=.

# 통합 코드 품질 분석 (complexity + dead-code)
make code-quality
```

### 프로젝트 관리
```bash
# 의존성만 설치
make install            # uv sync

# 모든 의존성 업데이트
make update             # uv lock --upgrade

# 캐시 정리
make clean
```

### Git 워크플로우
```bash
git add .
git commit  # 템플릿 에디터 열림, pre-commit hooks 자동 실행
```

**Pre-commit 체크 순서**:
1. 대용량 파일 체크
2. JSON/YAML/TOML 문법 검증
3. 머지 충돌 체크
4. 파일 끝 줄바꿈 및 공백 문자 수정
5. **Ruff 린팅** (자동 수정 포함)
6. **Ruff 포맷팅**
7. **Ty 타입 체크**
8. **Complexipy 복잡도 체크** (임계값: 15)
9. **Skylos 데드 코드 체크** (신뢰도: 80%)

**Commit 메시지 형식** (.gitmessage.txt에서):
```
<type>: <Subject>

[body]

[footer]
```

타입: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- Subject: 50자 이하, 명령형, 대문자 시작, 마침표 없음
- Footer: 이슈 참조 선택사항 (예: "Resolves: #123")

---

## 도구 설정

### UV (패키지 매니저)
- **pyproject.toml** - PEP 621 준수 프로젝트 메타데이터
- **uv.lock** - 잠긴 의존성 버전 (poetry.lock이나 package-lock.json과 유사)
- requirements.txt 불필요 - 모든 의존성은 pyproject.toml에 정의
- 개발 의존성: `[project.optional-dependencies.dev]`

### Ruff (올인원 린터 & 포맷터)
- **대체 도구**: Black, isort, flake8, pylint
- 줄 길이: 88자
- 타겟: Python 3.12
- 기본 자동 수정 활성화
- 설정: pyproject.toml의 `[tool.ruff]` 및 `[tool.ruff.lint]`

**활성화된 규칙**:
- E, W: pycodestyle (PEP 8 준수)
- F: pyflakes (오류 감지)
- I: isort (import 정렬)

**포맷 설정**:
- 따옴표 스타일: 쌍따옴표
- 들여쓰기: 공백 4칸
- 줄 끝: 자동 감지

### Ty (타입 체커)
- **Rust 기반** - mypy/pyright보다 10-20배 빠름
- **Gradual guarantee** - 작동하는 코드에 오류 추가 안 함
- **현재 버전**: 0.0.1a23 (알파)
- 설정: pyproject.toml의 `[tool.ty]` (현재 기본값 사용)
- 명령어: `ty check` (전체 프로젝트 체크)
- **주의**: 알파/프리뷰 단계 - 빠른 변화와 breaking changes 예상

### Pytest
- Python 경로: src
- 테스트 경로: tests
- 커버리지 리포팅: term-missing + HTML
- 자동 커버리지 활성화

### Complexipy (인지 복잡도 분석기)
- **Rust 기반** - 빠른 실행
- 함수별 인지 복잡도 측정
- 임계값: 10(엄격), **15(기본)**, 20(느슨)
- 설정: pyproject.toml의 `[tool.complexipy]`
- 자세한 내용: `docs/code-quality-tools.md`

### Skylos (데드 코드 탐지기)
- 미사용 함수, 클래스, 변수 탐지
- 신뢰도 점수 제공 (0-100%)
- 프레임워크 인식 (Django, Flask, FastAPI)
- 보안 스캔 기능 (`--danger --secrets`)
- 자세한 내용: `docs/code-quality-tools.md`

---

## 코드 품질 기준

### 필수 규칙
| 규칙 | 제한 |
|------|------|
| 줄 길이 | 88자 |
| 함수 길이 | 50줄 |
| 파일 길이 | 500줄 |
| 중첩 깊이 | 4단계 |
| 인지 복잡도 | 15 이하 |
| 테스트 커버리지 | 80% 이상 |

### 금지 사항
- `print()` 문 (로깅 사용)
- `breakpoint()` (디버그 제거)
- 하드코딩된 비밀값
- 빈 `except:` 절
- `any` 타입 사용

---

## 예외 처리

### 인라인 예외

**Ruff:**
```python
from .base import *  # noqa: F403
```

**Ty:**
```python
def test(self, t: list[int]) -> str:  # type: ignore
    ...
```

### 설정 레벨 예외

**pyproject.toml에서:**
- `[tool.ruff.lint]` → `ignore = [...]`로 특정 규칙 건너뛰기
- 파일별 오버라이드는 `per-file-ignores` 사용

---

## CI/CD

### GitHub Actions
- CI: `.github/workflows/ci.yml` — Python 3.12/3.13, Ruff lint/format, Ty, pytest
- 배포: `.github/workflows/publish.yml` — `v*.*.*` 태그 시 PyPI Trusted Publishing
- 선택사항: Codecov 통합

### Pre-commit Hooks
- 매 커밋마다 실행
- 설정: `.pre-commit-config.yaml`
- 빠른 실행을 위해 ruff-pre-commit 사용

---

## 프로젝트 구조

**PyPI 패키지명**: `qwen3-tts-triton` / **Import명**: `qwen3_tts_triton`
**라이선스**: Apache-2.0 / **Python**: `>=3.12`

```
qwen3-tts-triton/
├── src/
│   └── qwen3_tts_triton/           # PyPI 패키지 (wheel 포함)
│       ├── __init__.py             # __version__ = "0.1.0"
│       ├── py.typed                # PEP 561 타입 마커
│       ├── kernels/                # Triton 커널 (핵심)
│       │   ├── __init__.py
│       │   ├── rms_norm.py         # Fused RMSNorm
│       │   ├── rope.py             # Fused M-RoPE
│       │   ├── swiglu.py           # Fused SwiGLU
│       │   ├── fused_norm_residual.py  # RMSNorm + Residual
│       │   └── utils.py            # 커널 유틸리티
│       └── models/                 # 모델 패칭/추론
│           ├── __init__.py         # get_runner_class()
│           ├── patching.py         # Monkey-patch + find_patchable_model
│           ├── base_runner.py      # 기본 Qwen3-TTS
│           ├── triton_runner.py    # Triton 최적화
│           ├── faster_runner.py    # faster-qwen3-tts
│           └── triton_faster_runner.py # Hybrid (faster + Triton)
├── tests/                          # 수치 무결성 검증 (wheel 미포함)
├── benchmark/                      # 벤치마크 (wheel 미포함)
├── ui/                             # Streamlit 대시보드 (wheel 미포함)
├── docs/                           # 문서
├── LICENSE                         # Apache-2.0
├── pyproject.toml                  # UV + hatchling 빌드
├── Makefile                        # 개발 명령어
└── .github/workflows/
    ├── ci.yml                      # CI (lint + test)
    └── publish.yml                 # PyPI 배포 (v*.*.* 태그)
```

**Import 경로**: `from qwen3_tts_triton.kernels import TritonRMSNorm`

---

## 핵심 아키텍처

### Qwen3-TTS 1.7B Talker (주 병목)
| 항목 | 값 |
|------|-----|
| hidden_size | 2048 |
| num_attention_heads | 16 (GQA: kv_heads=8) |
| head_dim | 128 |
| intermediate_size | 6144 |
| num_hidden_layers | 28 |
| rms_norm_eps | 1e-6 |
| rope | M-RoPE, sections=[24,20,20] |
| activation | SwiGLU (silu) |

### Triton 커널
1. **RMSNorm** - variance+normalize+scale을 SRAM에서 1회 처리
2. **SwiGLU** - silu(gate)*up 퓨전, 중간 텐서 제거
3. **M-RoPE** - 3차원 위치 인코딩 Triton 구현
4. **Fused Norm+Residual** - residual add + RMSNorm 퓨전

### 개발 명령어 (추가)
```bash
make bench-kernels  # 커널 마이크로벤치마크
make bench-e2e      # E2E 추론 벤치마크
make bench          # 전체 벤치마크
make profile        # torch.profiler 프로파일링
make ui             # Streamlit 비교 대시보드
```

---

## 3-Tier 검증 체계

Liger Kernel 스타일 3단계 검증. 자세한 내용: `docs/verification-tiers.md`

| Tier | 검증 대상 | 실행 | 소요 시간 |
|------|----------|------|----------|
| **Tier 1** | 커널 정확도 (atol/rtol) | `make test` | ~5초 |
| **Tier 2** | 모델 패리티 2-pair (Base↔Triton, Faster↔Hybrid) | `make verify` | ~30초, GPU |
| **Tier 3** | E2E 품질 multi-runner 분포 비교 (UTMOS/CER/SIM) | `make eval-fast` / `make eval-full` | ~15-90분, GPU |

### Tier 2 임계값 (2-pair 비교)
- **Pair A**: Base vs Triton (HF 모델 + Triton 커널)
- **Pair B**: Faster vs Hybrid (faster-qwen3-tts + Triton 커널)
- Hidden state cosine similarity > 0.95 (layer 0,7,14,21,27)
- Output cosine similarity > 0.95 (last_hidden_state)
- Max abs diff (보고용, 임계값 없음)

### Tier 3: Multi-Runner 독립 분포 비교 (vLLM/TensorRT-LLM 패턴)

Base를 기준으로 Triton/Faster/Hybrid 각각을 독립 생성 → 태스크 메트릭(CER/UTMOS) 분포 비교.
기존 pair-level 파형 비교(PESQ/STOI/MCD)는 stochastic TTS에 부적합하여 삭제.

| 메트릭 | 임계값 | 업계 근거 |
|--------|--------|----------|
| UTMOS delta | \|mean\| < 0.3 | F5-TTS 독립 생성 변동 |
| UTMOS floor | 양쪽 > 2.5 | 절대 품질 하한 |
| CER delta | \|mean\| < 0.05 | SGLang 1-5% 허용 |
| Speaker Sim | mean > 0.75 | Qwen3-TTS SIM > 0.79 |
| Mann-Whitney U | p > 0.05 (full) | 비모수 분포 동등성 |

### 명령어 실행
```bash
make test               # Tier 1 (기본)
make test-parity        # Tier 2 (GPU 필요)
make verify             # Tier 1+2 통합 검증
make verify-all         # Tier 1+2+3 전체 검증
make eval-fast          # Tier 3 빠른 평가 (Base vs all runners, whisper-small)
make eval-full          # Tier 3 전체 평가 (Base vs all runners, whisper-large-v3, Mann-Whitney)
make bench-kernels      # 커널별 speedup 확인
make bench-speed        # E2E 속도 벤치마크 (4 runners)
make bench              # 전체 (속도 + 품질 + 검증)
make lint
```

### 타입 체크 참고사항
- Ty는 초기 개발 단계 - breaking changes 예상
- Ty 문제 발생 시 일시적으로 pyright로 전환 가능
- Ty 버그 보고: https://github.com/astral-sh/ty

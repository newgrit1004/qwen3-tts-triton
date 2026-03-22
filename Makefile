DIR?=.
SERVICE_NAME=qwen3-tts-triton
VERSION=0.1

# 프로젝트 초기 설정
.PHONY: setup
setup:
	git config commit.template .gitmessage.txt
	uv sync --all-extras --dev
	uv run pre-commit install

# 코드 포맷팅
.PHONY: format
format:
	uv run ruff format ${DIR}

# 코드 린팅 체크
.PHONY: lint
lint:
	uv run ruff check ${DIR}

# 코드 린팅 + 자동 수정
.PHONY: lint-fix
lint-fix:
	uv run ruff check --fix ${DIR}

# 타입 체크
.PHONY: typecheck
typecheck:
	uv run ty check

# 테스트 실행
.PHONY: test
test:
	uv run pytest

# 테스트 + 커버리지 리포트
.PHONY: test-cov
test-cov:
	uv run pytest --cov

# 캐시 및 임시 파일 정리
.PHONY: clean
clean:
	rm -rf .ruff_cache .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# 의존성만 설치
.PHONY: install
install:
	uv sync

# 의존성 업데이트
.PHONY: update
update:
	uv lock --upgrade

# pre-commit 전체 실행
.PHONY: pre-commit
pre-commit:
	uv run pre-commit run --all-files

# === Triton 커널 벤치마크 ===

# 개별 커널 마이크로벤치마크
.PHONY: bench-kernels
bench-kernels:
	uv run python benchmark/bench_kernels.py

# E2E 속도 벤치마크 (4 runners × 2 texts, CUDA event timing)
.PHONY: bench-speed
bench-speed:
	uv run python benchmark/bench_e2e.py

# 전체 벤치마크 (속도 + 품질 + 검증 리포트)
.PHONY: bench
bench: bench-kernels bench-speed eval-fast verify

# torch.profiler 프로파일링
.PHONY: profile
profile:
	uv run python benchmark/profiler.py

# === Streamlit UI ===

# 음성 샘플 사전 생성 (4 modes x 10 custom + clone, GPU 필요)
.PHONY: generate-samples
generate-samples:
	uv run python scripts/generate_samples.py

# 비교 대시보드 실행
.PHONY: ui
ui:
	uv run streamlit run ui/app.py

# === 품질 평가 ===

# Tier 3 빠른 평가 (~5분, whisper-small, 1회/문장, 분포 비교)
.PHONY: eval-fast
eval-fast:
	uv run python -m benchmark.eval_quality --mode fast

# Tier 3 전체 평가 (~30분, whisper-large-v3, 3회/문장, Mann-Whitney)
.PHONY: eval-full
eval-full:
	uv run python -m benchmark.eval_quality --mode full

# bench-e2e는 bench-speed의 별칭 (하위호환)
.PHONY: bench-e2e
bench-e2e: bench-speed

# Tier 2: 모델 레벨 패리티 테스트 (GPU 필요, ~2분)
.PHONY: test-parity
test-parity:
	uv run pytest tests/test_model_parity.py -v --no-header

# === 3-Tier 검증 ===

# 3-Tier 전체 검증 실행 (Tier 1 + 2, Tier 3은 기존 결과 로드)
.PHONY: verify-all
verify-all:
	uv run python -m benchmark.run_verification

# Tier 1+2 실행 + Tier 3 기존 결과 로드
.PHONY: verify
verify:
	uv run python -m benchmark.run_verification

# 전체 체크 (lint + test)
.PHONY: check
check: lint test

# README 벤치마크 표 자동 업데이트
.PHONY: update-readme
update-readme:
	uv run python scripts/generate_bench_tables.py

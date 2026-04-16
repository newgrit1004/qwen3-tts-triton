DIR?=.
SERVICE_NAME=qwen3-tts-triton
VERSION=0.2.0

# Project initial setup
.PHONY: setup
setup:
	git config commit.template .gitmessage.txt
	uv sync --all-extras --dev
	uv run pre-commit install

# Code formatting
.PHONY: format
format:
	uv run ruff format ${DIR}

# Code linting check
.PHONY: lint
lint:
	uv run ruff check ${DIR}

# Code linting + auto-fix
.PHONY: lint-fix
lint-fix:
	uv run ruff check --fix ${DIR}

# Type checking
.PHONY: typecheck
typecheck:
	uv run ty check

# Run tests
.PHONY: test
test:
	uv run pytest

# Tests + coverage report
.PHONY: test-cov
test-cov:
	uv run pytest --cov

# Clean cache and temporary files
.PHONY: clean
clean:
	rm -rf .ruff_cache .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Install dependencies only
.PHONY: install
install:
	uv sync

# Update dependencies
.PHONY: update
update:
	uv lock --upgrade

# Run all pre-commit hooks
.PHONY: pre-commit
pre-commit:
	uv run pre-commit run --all-files

# === Triton Kernel Benchmarks ===

# Individual kernel microbenchmarks
.PHONY: bench-kernels
bench-kernels:
	uv run python benchmark/bench_kernels.py

# E2E speed benchmark (7 runners x 2 texts, CUDA event timing)
.PHONY: bench-speed
bench-speed:
	uv run python benchmark/bench_e2e.py

# KV cache memory measurement (CPU: theoretical, --device cuda: actual VRAM)
.PHONY: bench-kv-memory
bench-kv-memory:
	uv run python -m benchmark.bench_kv_memory

# KV cache memory measurement (GPU VRAM actual)
.PHONY: bench-kv-memory-gpu
bench-kv-memory-gpu:
	uv run python -m benchmark.bench_kv_memory --device cuda

# Phase 5: VRAM savings utilization experiment (context length scaling)
.PHONY: bench-throughput
bench-throughput:
	uv run python -m benchmark.bench_throughput_scaling

# E2E long sequence benchmark (TQ effect measurement, 4 texts x 5 runners)
.PHONY: bench-speed-long
bench-speed-long:
	uv run python -m benchmark.bench_e2e_long

# E2E fixed-token benchmark (per-token speed comparison, confound-free)
.PHONY: bench-speed-fixed
bench-speed-fixed:
	uv run python -m benchmark.bench_e2e_fixed

# Full benchmark (speed + quality + verification report)
.PHONY: bench
bench: bench-kernels bench-speed eval-fast verify

# torch.profiler profiling
.PHONY: profile
profile:
	uv run python benchmark/profiler.py

# === Streamlit UI ===

# Pre-generate audio samples (4 modes x 10 custom + clone, GPU required)
.PHONY: generate-samples
generate-samples:
	uv run python scripts/generate_samples.py

# Run comparison dashboard
.PHONY: ui
ui:
	uv run streamlit run ui/app.py

# === Quality Evaluation ===

# Tier 3 fast evaluation (~15min, Cohere Transcribe, 1 run/sentence, distribution comparison)
.PHONY: eval-fast
eval-fast:
	uv run python -m benchmark.eval_quality --mode fast

# Tier 3 full evaluation (~80min, Cohere Transcribe, 3 runs/sentence, Mann-Whitney)
.PHONY: eval-full
eval-full:
	uv run python -m benchmark.eval_quality --mode full

# === Tongue twister pronunciation stress test ===

# Tongue twister fast evaluation (~5min, 15 sentences, Cohere Transcribe)
.PHONY: eval-twister-fast
eval-twister-fast:
	uv run python -m benchmark.eval_tongue_twister --patch-range 0,24 --mode fast

# Tongue twister full evaluation (~50min, 45 sentences x 3 runs, Cohere Transcribe)
.PHONY: eval-twister-full
eval-twister-full:
	uv run python -m benchmark.eval_tongue_twister --patch-range 0,24 --mode full

# Tongue twister PER analysis (reuse existing WAVs)
.PHONY: analyze-per-twister
analyze-per-twister:
	uv run python -m benchmark.analyze_per \
		--eval-dir benchmark/output/eval_tongue_twister/twister_full \
		--sentence-set tongue_twister

# TQ quality evaluation PER analysis
.PHONY: analyze-per-tq
analyze-per-tq:
	uv run python -m benchmark.analyze_per \
		--eval-dir benchmark/output/eval/multi_full \
		--sentence-set standard

# bench-e2e is an alias for bench-speed (backward compat)
.PHONY: bench-e2e
bench-e2e: bench-speed

# Tier 2: model-level parity tests (GPU required, ~2min)
.PHONY: test-parity
test-parity:
	uv run pytest tests/test_model_parity.py -v --no-header

# === 3-Tier Verification ===

# Full 3-Tier verification (run Tier 3 full, then generate combined report)
.PHONY: verify-all
verify-all: eval-full verify

# Run Tier 1+2 + load existing Tier 3 results
.PHONY: verify
verify:
	uv run python -m benchmark.run_verification

# Full check (lint + test)
.PHONY: check
check: lint test

# Auto-update README benchmark tables
.PHONY: update-readme
update-readme:
	uv run python scripts/generate_bench_tables.py

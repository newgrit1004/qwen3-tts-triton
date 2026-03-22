# UV, Ruff, Ty 워크플로우

Astral의 현대적인 Python 도구 체인을 위한 도메인 지식입니다.

## UV - 패키지 매니저

### 핵심 개념
- **Rust 기반**: pip보다 10-100배 빠름
- **자동 venv**: 수동 활성화 불필요
- **Lock 파일**: `uv.lock`으로 재현 가능한 빌드

### 일반 명령어

```bash
# 프로젝트 설정
uv sync                     # 모든 의존성 설치
uv sync --all-extras --dev  # 모든 extras와 dev 의존성 포함 설치

# 의존성 추가
uv add httpx                # 프로덕션 의존성 추가
uv add --dev pytest         # 개발 의존성 추가
uv add "pydantic>=2.0"      # 버전 제약과 함께 추가

# 의존성 제거
uv remove httpx

# 의존성 업데이트
uv lock --upgrade           # lock 파일 업데이트
uv lock --upgrade-package httpx  # 특정 패키지 업데이트

# 명령어 실행
uv run python src/main.py   # 관리되는 venv로 실행
uv run pytest               # 테스트 실행
uv run ruff check src/      # 린팅 실행
```

### pyproject.toml 구조

```toml
[project]
name = "my-project"
version = "0.1.0"
description = "프로젝트 설명"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.25.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.7.0",
    "ty>=0.0.1a1",
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pre-commit>=4.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Ruff - 린터 & 포맷터

### 핵심 개념
- **대체**: Black, isort, flake8, pylint
- **Rust 기반**: Python 대안보다 10-100배 빠름
- **자동 수정**: 많은 이슈 자동으로 수정 가능

### 설정

```toml
[tool.ruff]
line-length = 88
target-version = "py312"
fix = true

[tool.ruff.lint]
select = [
    "E",    # pycodestyle 오류
    "W",    # pycodestyle 경고
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
    "SIM",  # flake8-simplify
]
ignore = []

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["S101"]  # 테스트에서 assert 허용

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "auto"
```

### 명령어

```bash
# 린팅
uv run ruff check src/          # 이슈 검사
uv run ruff check src/ --fix    # 이슈 자동 수정

# 포맷팅
uv run ruff format src/         # 코드 포맷
uv run ruff format src/ --check # 변경 없이 포맷 검사

# 조합
uv run ruff check src/ --fix && uv run ruff format src/
```

### 일반 규칙

| 규칙 | 설명 |
|------|------|
| E/W | pycodestyle (PEP 8) |
| F | pyflakes (오류) |
| I | isort (임포트 정렬) |
| B | flake8-bugbear (일반적인 버그) |
| UP | pyupgrade (현대 Python) |
| SIM | simplify (코드 단순화) |
| S | flake8-bandit (보안) |

### 인라인 무시

```python
# 라인에서 특정 규칙 무시
from module import *  # noqa: F403

# 여러 규칙 무시
value = eval(user_input)  # noqa: S307, B307

# 전체 파일 무시 (상단에)
# ruff: noqa: F401
```

## Ty - 타입 체커

### 핵심 개념
- **Rust 기반**: mypy보다 10-20배 빠름
- **점진적 타이핑**: 부분 타입 힌트와 작동
- **Astral 제작**: 통합 도구 체인의 일부

### 설정

```toml
[tool.ty]
# 현재 기본값 사용 (알파 버전)
# Ty가 성숙해지면 설정 옵션 확장 예정
```

### 명령어

```bash
# 전체 프로젝트 타입 검사
uv run ty check

# 특정 경로 타입 검사
uv run ty check src/
```

### 인라인 무시

```python
# 라인에서 타입 오류 무시
result = dynamic_function()  # type: ignore

# 특정 오류 무시
value: int = get_value()  # type: ignore[assignment]
```

### 타입 스텁 파일

타입 힌트가 없는 라이브러리용으로 `.pyi` 스텁 파일 생성:

```python
# src/vendor.pyi
def external_function(data: dict[str, str]) -> list[int]: ...

class ExternalClass:
    def method(self, value: int) -> str: ...
```

## Makefile 통합

```makefile
# 개발 명령어
install:
	uv sync --all-extras --dev

format:
	uv run ruff format src/ tests/

lint:
	uv run ruff check src/ tests/

lint-fix:
	uv run ruff check src/ tests/ --fix

typecheck:
	uv run ty check

test:
	uv run pytest

test-cov:
	uv run pytest --cov=src --cov-report=term-missing

# 통합 검사
check: lint typecheck test

# Pre-commit
pre-commit:
	uv run pre-commit run --all-files
```

## Pre-commit 통합

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: local
    hooks:
      - id: ty
        name: ty
        entry: uv run ty check
        language: system
        types: [python]
        pass_filenames: false
```

## 워크플로우 요약

### 일상 개발
```bash
# 1. 작업 시작
uv sync                    # 의존성 확인

# 2. 코드 작성
# ... 파일 편집 ...

# 3. 품질 검사
make lint-fix              # 린팅 이슈 수정
make format                # 코드 포맷
make typecheck             # 타입 검사

# 4. 테스트
make test                  # 테스트 실행

# 5. 커밋
git add .
git commit                 # pre-commit 훅 자동 실행
```

### 의존성 업데이트
```bash
# 특정 패키지 업데이트
uv add "httpx>=0.26"

# 모든 의존성 업데이트
make update                # uv lock --upgrade

# 아무것도 깨지지 않았는지 확인
make check                 # lint + typecheck + test
```

## 문제 해결

### UV 이슈
```bash
# 캐시 정리
uv cache clean

# 강제 재설치
rm -rf .venv && uv sync
```

### Ruff 충돌
```bash
# 어떤 규칙이 충돌하는지 확인
uv run ruff check src/ --show-fixes

# 규칙 설명 보기
uv run ruff rule E501
```

### Ty 제한사항 (알파)
- 일부 엣지 케이스가 아직 작동하지 않을 수 있음
- 필요시 pyright로 폴백:
  ```bash
  uv add --dev pyright
  uv run pyright src/
  ```

# Python 프로젝트 구조

Python 프로젝트 구성을 위한 도메인 지식입니다.

## 표준 구조

```
project-name/
├── .claude/                    # Claude Code 설정
│   ├── agents/                 # 에이전트 정의
│   ├── commands/               # 슬래시 명령어
│   ├── rules/                  # 프로젝트 규칙
│   ├── skills/                 # 도메인 지식
│   └── hooks/                  # 자동화 훅
├── .github/
│   ├── workflows/
│   │   └── ci.yml              # GitHub Actions
│   └── dependabot.yml          # 의존성 업데이트
├── src/                        # 소스 코드
│   ├── __init__.py
│   ├── main.py                 # 진입점
│   ├── config/                 # 설정
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── domain/                 # 비즈니스 로직
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── services.py
│   ├── adapters/               # 외부 인터페이스
│   │   ├── __init__.py
│   │   ├── api.py
│   │   └── database.py
│   └── utils/                  # 공유 유틸리티
│       ├── __init__.py
│       └── helpers.py
├── tests/                      # 테스트 디렉토리
│   ├── __init__.py
│   ├── conftest.py             # 공유 픽스처
│   ├── unit/
│   │   └── test_services.py
│   └── integration/
│       └── test_api.py
├── docs/                       # 문서
│   └── api.md
├── .env.example                # 환경 템플릿
├── .gitignore
├── .gitmessage.txt             # 커밋 템플릿
├── .pre-commit-config.yaml     # Pre-commit 훅
├── pyproject.toml              # 프로젝트 설정
├── uv.lock                     # 잠긴 의존성
├── Makefile                    # 개발 명령어
├── CLAUDE.md                   # Claude Code 지침
└── README.md                   # 프로젝트 문서
```

## 디렉토리 목적

### `src/` - 소스 코드

| 디렉토리 | 목적 |
|----------|------|
| `config/` | 설정 및 세팅 |
| `domain/` | 비즈니스 로직, 모델, 서비스 |
| `adapters/` | 외부 인터페이스 (API, DB 등) |
| `utils/` | 공유 헬퍼 함수 |

### `tests/` - 테스트 코드

| 디렉토리 | 목적 |
|----------|------|
| `unit/` | 단위 테스트 (격리됨, 빠름) |
| `integration/` | 통합 테스트 (실제 의존성) |
| `e2e/` | 엔드투엔드 테스트 (전체 시스템) |
| `conftest.py` | 공유 픽스처 |

### `.claude/` - Claude Code 설정

| 디렉토리 | 목적 |
|----------|------|
| `agents/` | 전문화된 AI 에이전트 |
| `commands/` | `/command` 정의 |
| `rules/` | 필수 가이드라인 |
| `skills/` | 도메인 지식 |
| `hooks/` | 이벤트 기반 자동화 |

## 설정 파일

### pyproject.toml
```toml
[project]
name = "project-name"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[project.optional-dependencies]
dev = ["ruff", "ty", "pytest", "pytest-cov", "pre-commit"]

[tool.ruff]
line-length = 88
target-version = "py312"

[tool.pytest.ini_options]
pythonpath = "src"
testpaths = ["tests"]
```

### .pre-commit-config.yaml
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-toml

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

### .gitignore
```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# 가상 환경
.venv/
venv/

# IDE
.idea/
.vscode/
*.swp

# 환경
.env
.env.local
*.local

# 커버리지
htmlcov/
.coverage
coverage.xml

# 캐시
.pytest_cache/
.ruff_cache/
.mypy_cache/
```

### Makefile
```makefile
.PHONY: install format lint lint-fix typecheck test test-cov clean

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
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html

check: lint typecheck test

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
```

## 명명 규칙

### 파일
| 유형 | 규칙 | 예시 |
|------|------|------|
| 모듈 | snake_case | `user_service.py` |
| 테스트 | test_ 접두사 | `test_user_service.py` |
| 설정 | 소문자 | `settings.py` |

### 코드
| 유형 | 규칙 | 예시 |
|------|------|------|
| 클래스 | PascalCase | `UserService` |
| 함수 | snake_case | `get_user_by_id` |
| 변수 | snake_case | `user_name` |
| 상수 | UPPER_SNAKE | `MAX_RETRIES` |

## 모듈 구성

### __init__.py 패턴
```python
# src/domain/__init__.py
from .models import User, Order
from .services import UserService, OrderService

__all__ = ["User", "Order", "UserService", "OrderService"]
```

### 순환 임포트 방지
```python
# 타입 힌트 전용으로 TYPE_CHECKING 사용
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .other_module import OtherClass

def function(obj: "OtherClass") -> None:
    ...
```

## 설정 패턴

### 환경 기반 설정
```python
# src/config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    debug: bool = False
    database_url: str
    api_key: str

    class Config:
        env_file = ".env"

settings = Settings()
```

### 사용법
```python
from src.config import settings

if settings.debug:
    print("디버그 모드 활성화됨")
```

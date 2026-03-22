# /build-fix 명령어

프로젝트의 빌드 오류를 진단하고 수정합니다.

## 사용법
```
/build-fix
/build-fix <특정 오류 메시지>
```

## 이 명령어의 기능

1. **호출**: `build-error-resolver` 에이전트
2. **분석**: 빌드 출력과 오류 메시지
3. **수정**: 의존성, import, 타입, 린트 오류

## 워크플로우

### 단계 1: 진단 실행

```bash
# 전체 체크 실행하여 이슈 식별
make check
```

### 단계 2: 오류 유형 식별

| 오류 유형 | 지표 |
|----------|------|
| 의존성 | `ModuleNotFoundError`, UV 해결 오류 |
| Import | `ImportError`, 순환 import |
| 타입 | Ty 오류 메시지 |
| 린트 | Ruff 오류 코드 (E, W, F) |

### 단계 3: 수정 적용

```bash
# 의존성 이슈
uv sync
uv add missing-package

# 린트 이슈
make lint-fix

# 타입 이슈
# Ty 출력 기반 수동 수정

# 포맷 이슈
make format
```

### 단계 4: 검증

```bash
make check  # 통과해야 함
```

## 일반적인 시나리오

### 시나리오 1: 누락된 의존성

```
오류: ModuleNotFoundError: No module named 'httpx'

수정:
uv add httpx
```

### 시나리오 2: 버전 충돌

```
오류: Could not resolve dependencies

수정:
uv lock --upgrade
# 또는 pyproject.toml에서 버전 제약 완화
```

### 시나리오 3: Pre-commit 실패

```
오류: pre-commit hook 'ruff' failed

수정:
make lint-fix
make format
```

### 시나리오 4: 타입 오류

```
오류: Ty: Incompatible return type

수정:
# 함수 시그니처 업데이트 또는 타입 캐스트 추가
```

## 빠른 참조

```bash
# 전체 진단
make check

# 개별 체크
make lint        # Ruff 린트
make lint-fix    # 자동 수정 린트
make format      # Ruff 포맷
make typecheck   # Ty 체크
make test        # pytest

# 의존성 관리
uv sync          # 의존성 설치
uv lock          # lock 파일 재생성
uv add package   # 새 의존성 추가
```

## 출력 형식

```markdown
## 빌드 수정 보고서

### 발견된 오류
| 유형 | 위치 | 메시지 |
|------|------|--------|
| 린트 | src/main.py:15 | E501: 줄이 너무 김 |
| 타입 | src/utils.py:30 | 호환되지 않는 반환 |

### 적용된 수정
1. `make lint-fix` - 3개 린트 이슈 자동 수정
2. `src/utils.py:30`에서 반환 타입 업데이트

### 검증
- `make check`: 통과
```

## 관련 명령어

- `/code-review` - 코드 품질 검토
- `/tdd` - 테스트 주도 개발
- `/refactor-clean` - 코드 정리

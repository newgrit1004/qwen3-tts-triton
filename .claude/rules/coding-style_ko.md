# Python 코딩 스타일 규칙

이 규칙들은 이 프로젝트의 모든 Python 코드에 반드시 따라야 합니다.

## 핵심 원칙

1. **가독성 우선**: 코드는 쓰여지는 것보다 읽히는 경우가 더 많음
2. **단순함**: 가장 간단한 작동 솔루션 선호
3. **DRY**: 반복하지 말 것 - 공통 로직 추출
4. **YAGNI**: 필요하지 않을 것이다 - 필요한 것만 구축

## 파일 구성

### 최대 크기
- **함수**: 최대 50줄
- **파일**: 최대 500줄 (절대 한계 800줄)
- **중첩**: 최대 4단계

### 디렉토리 구조
```
src/
├── __init__.py
├── main.py          # 진입점
├── config/          # 설정
├── domain/          # 비즈니스 로직
├── adapters/        # 외부 인터페이스
└── utils/           # 공유 유틸리티
```

## 명명 규칙

### 변수와 함수
```python
# snake_case 사용
user_name = "John"
def calculate_total(items: list) -> float: ...

# 설명적인 이름
# 나쁨
x = get_d()
# 좋음
user_data = get_user_details()
```

### 클래스
```python
# PascalCase 사용
class UserRepository:
    pass

class OrderProcessor:
    pass
```

### 상수
```python
# UPPER_SNAKE_CASE 사용
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT = 30
API_BASE_URL = "https://api.example.com"
```

## 타입 힌트 (필수)

모든 함수는 반드시 타입 힌트가 있어야 합니다.

```python
# 필수
def process_user(user_id: int, options: dict[str, Any] | None = None) -> User:
    ...

# 복잡한 타입의 경우 타입 별칭 사용
UserData = dict[str, str | int | list[str]]
def parse_user(data: UserData) -> User:
    ...
```

## 독스트링

### 공개 API (필수)
```python
def calculate_discount(price: Decimal, rate: float) -> Decimal:
    """할인된 가격을 계산합니다.

    Args:
        price: 원래 가격
        rate: 할인율 (0.0에서 1.0)

    Returns:
        할인된 가격

    Raises:
        ValueError: rate가 0과 1 사이가 아닌 경우
    """
    if not 0 <= rate <= 1:
        raise ValueError("Rate는 0과 1 사이여야 합니다")
    return price * Decimal(1 - rate)
```

### 내부 함수 (선택)
내부 함수의 경우, 함수 이름과 타입 힌트가 자명하면 독스트링은 선택사항입니다.

## 오류 처리

### 항상 특정 예외 잡기
```python
# 나쁨
try:
    result = process()
except:
    pass

# 좋음
try:
    result = process()
except ValidationError as e:
    logger.warning(f"검증 실패: {e}")
    raise
except DatabaseError as e:
    logger.error(f"데이터베이스 오류: {e}")
    raise ServiceError("처리 실패") from e
```

### 커스텀 예외 사용
```python
class DomainError(Exception):
    """도메인 오류의 기본 예외."""
    pass

class NotFoundError(DomainError):
    """리소스를 찾을 수 없음."""
    pass
```

## 임포트

Ruff (isort 규칙)에 의해 관리됩니다.

```python
# 표준 라이브러리
import os
from collections.abc import Callable
from pathlib import Path

# 서드파티
import httpx
from pydantic import BaseModel

# 로컬
from src.config import settings
from src.domain.models import User
```

## 코드 품질 체크리스트

커밋 전 확인:

- [ ] 함수 50줄 이하
- [ ] 파일 500줄 이하
- [ ] 중첩 깊이 <= 4
- [ ] 모든 공개 함수에 타입 힌트
- [ ] `print()` 또는 `breakpoint()` 문 없음
- [ ] 하드코딩된 값 없음 (상수 사용)
- [ ] 오류 처리가 특정적 (빈 `except:` 없음)
- [ ] 새 코드에 테스트 존재

## Ruff 설정

이 규칙들은 `pyproject.toml`의 Ruff에 의해 강제됩니다:

```toml
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
```

## 빠른 참조

| 규칙 | 제한 |
|------|------|
| 줄 길이 | 88자 |
| 함수 길이 | 50줄 |
| 파일 길이 | 500줄 |
| 중첩 깊이 | 4단계 |
| 테스트 커버리지 | 최소 80% |

# 테스트 규칙

이 규칙들은 이 프로젝트의 모든 테스트에 반드시 따라야 합니다.

## 커버리지 요구사항

| 코드 유형 | 최소 커버리지 |
|-----------|--------------|
| 일반 코드 | 80% |
| 핵심 비즈니스 로직 | 100% |
| 보안 관련 코드 | 100% |
| 데이터 검증 | 100% |

## 테스트 주도 개발 (TDD)

### 필수 워크플로우: RED-GREEN-REFACTOR

```
1. RED    → 실패하는 테스트 먼저 작성
2. GREEN  → 통과하는 최소한의 코드 작성
3. REFACTOR → 코드 품질 개선
```

### TDD 필수 적용 대상:
- 새 기능
- 새 함수/클래스
- 버그 수정 (재현 테스트 먼저 작성)
- 핵심 비즈니스 로직

## 테스트 구조

### 파일 구성
```
tests/
├── __init__.py
├── conftest.py          # 공유 픽스처
├── unit/
│   ├── __init__.py
│   ├── test_models.py
│   └── test_services.py
├── integration/
│   ├── __init__.py
│   └── test_api.py
└── e2e/                 # 엔드투엔드 테스트
    └── test_workflows.py
```

### 테스트 명명
```python
# 패턴: test_<무엇>_<조건>_<예상>
def test_calculate_total_empty_cart_returns_zero():
    ...

def test_authenticate_invalid_password_raises_error():
    ...
```

### 테스트 구조 (AAA 패턴)
```python
def test_user_creation():
    # 준비 (Arrange)
    user_data = {"name": "John", "email": "john@example.com"}

    # 실행 (Act)
    user = User.create(user_data)

    # 검증 (Assert)
    assert user.name == "John"
    assert user.email == "john@example.com"
```

## Pytest 모범 사례

### 픽스처 사용
```python
# conftest.py
import pytest

@pytest.fixture
def sample_user():
    return User(id=1, name="Test User")

@pytest.fixture
def mock_database(mocker):
    return mocker.patch("src.adapters.database.Database")

# test_user.py
def test_user_service(sample_user, mock_database):
    mock_database.get_user.return_value = sample_user
    result = user_service.get(1)
    assert result == sample_user
```

### 파라미터화된 테스트
```python
@pytest.mark.parametrize("input_value,expected", [
    (0, "zero"),
    (1, "one"),
    (-1, "negative"),
])
def test_number_to_word(input_value, expected):
    assert number_to_word(input_value) == expected
```

### 예외 테스트
```python
def test_invalid_input_raises_error():
    with pytest.raises(ValueError, match="입력이 비어있을 수 없습니다"):
        process_data("")
```

## 모킹 가이드라인

### 모킹할 때
- 외부 API
- 데이터베이스
- 파일 시스템 작업
- 시간 의존 작업
- 서드파티 서비스

### 모킹하지 말 때
- 순수 함수
- 단순 데이터 변환
- 테스트 중인 자신의 코드

### 모킹 예제
```python
from unittest.mock import MagicMock, patch

def test_api_client():
    with patch("src.adapters.api.httpx.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_client.return_value.get.return_value = mock_response

        result = api_client.fetch_data()

        assert result["status"] == "ok"
```

## 테스트 카테고리

### 단위 테스트
- 단일 함수/클래스 테스트
- 빠른 실행 (밀리초)
- 외부 의존성 없음
- 실행: `make test`

### 통합 테스트
- 컴포넌트 상호작용 테스트
- 실제 의존성 사용 가능
- 마커: `@pytest.mark.integration`
- 실행: `uv run pytest -m integration`

### 엔드투엔드 테스트
- 전체 워크플로우 테스트
- 전체 시스템 동작
- 마커: `@pytest.mark.e2e`
- 실행: `uv run pytest -m e2e`

## 테스트 명령어

```bash
# 모든 테스트 실행
make test

# 커버리지와 함께 실행
make test-cov

# 특정 파일 실행
uv run pytest tests/unit/test_models.py

# 특정 테스트 실행
uv run pytest -k "test_calculate_total"

# 상세 출력과 함께 실행
uv run pytest -v

# 단위 테스트만 실행
uv run pytest tests/unit/

# 마커된 테스트 실행
uv run pytest -m "not slow"
```

## 테스트 실패 시

1. **테스트를 통과하도록 수정하지 마세요** (테스트가 잘못된 경우 제외)
2. **대신 구현을 수정하세요**
3. 테스트가 잘못된 경우:
   - 이유 문서화
   - 리뷰 승인 받기
   - 그 후 테스트 수정

## 커밋 전 테스트 검증

커밋 전:
```bash
make test       # 모든 테스트 통과
make lint       # Ruff 통과
make typecheck  # Ty 통과
```

## 테스트 품질 체크리스트

- [ ] 테스트가 독립적 (공유 상태 없음)
- [ ] 테스트가 결정적 (매번 같은 결과)
- [ ] 테스트가 빠름 (단위 테스트 각 100ms 미만)
- [ ] 테스트에 설명적인 이름
- [ ] 엣지 케이스 커버됨
- [ ] 오류 조건 테스트됨
- [ ] 테스트에 `print()` 문 없음

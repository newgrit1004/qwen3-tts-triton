# /learn 명령어

현재 세션에서 발견된 패턴을 추출하고 문서화합니다.

## 사용법
```
/learn
/learn <특정 패턴 또는 개념>
```

## 이 명령어의 기능

1. **분석**: 현재 세션의 코드 변경과 토론
2. **추출**: 재사용 가능한 패턴과 학습 내용
3. **문서화**: 프로젝트 지식 베이스에 추가

## 사용 시점

- 복잡한 문제 해결 후
- 새로운 패턴 발견 시
- 까다로운 이슈 디버깅 후
- 새로운 솔루션 구현 시
- 긴 세션 종료 전

## 학습 카테고리

### 1. 코드 패턴

```python
# 패턴: 지수 백오프를 사용한 재시도
import time
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
        return wrapper
    return decorator
```

### 2. 버그 패턴

```markdown
## 버그: 순환 Import 오류

### 증상
ImportError: cannot import name 'X' from partially initialized module

### 근본 원인
두 모듈이 모듈 레벨에서 서로를 import

### 해결책
타입 힌트에 TYPE_CHECKING 사용:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .other_module import OtherClass
```

### 예방
- 설계에서 순환 의존성 피하기
- 의존성 주입 사용
- 공유 타입을 별도 모듈로 이동
```

### 3. 설정 패턴

```markdown
## 패턴: 환경 기반 설정

### 구현
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    debug: bool = False
    database_url: str

    class Config:
        env_file = ".env"
```

### 사용 시점
- 여러 배포 환경
- 민감한 설정 값
- 12-factor 앱 준수
```

### 4. 테스트 패턴

```markdown
## 패턴: 파라미터화된 픽스처

### 구현
```python
@pytest.fixture(params=["sqlite", "postgres"])
def database(request):
    if request.param == "sqlite":
        return SQLiteDatabase(":memory:")
    return PostgresDatabase(TEST_DB_URL)
```

### 장점
동일한 테스트를 여러 백엔드에 대해 실행
```

## 출력 형식

```markdown
## 세션 학습 보고서

### 세션 요약
- 소요 시간: 2시간
- 집중 영역: 사용자 인증 구현
- 수정된 파일: 5개

### 추출된 패턴

#### 1. JWT 토큰 검증 패턴
**카테고리**: 보안
**위치**: `src/auth/jwt.py`
**설명**: 적절한 오류 처리가 포함된 재사용 가능한 JWT 검증
**코드**:
```python
def validate_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthError("토큰 만료")
    except jwt.InvalidTokenError:
        raise AuthError("유효하지 않은 토큰")
```

#### 2. 데이터베이스 트랜잭션 컨텍스트 매니저
**카테고리**: 데이터베이스
**위치**: `src/db/transaction.py`
**설명**: 자동 롤백이 있는 안전한 트랜잭션 처리
**코드**:
```python
@contextmanager
def transaction(session):
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
```

### 해결된 버그

#### 버그: 캐시 업데이트의 경쟁 조건
**증상**: 간헐적으로 오래된 데이터 반환
**근본 원인**: 캐시 업데이트가 원자적이지 않음
**해결책**: 원자적 업데이트를 위해 Redis MULTI/EXEC 사용
**예방**: 공유 상태에는 항상 원자적 연산 사용

### 발견된 도구/기법

1. **UV cache clean**: 의존성 해결 문제 해결
2. **Ruff 규칙 조회**: `uv run ruff rule E501`로 규칙 설명
3. **Pytest --lf**: 마지막 실패한 테스트만 재실행

### 향후 권장사항

1. 모든 외부 API 호출에 재시도 로직 추가
2. 세션 저장소로 Redis 고려
3. 인증 이벤트에 구조화된 로깅 추가
```

## 워크플로우

### 단계 1: 학습 트리거

```
/learn
```

### 단계 2: 추출된 패턴 검토

Claude가 분석:
- 세션의 코드 변경
- 사용된 문제 해결 접근법
- 새로 발견된 기법

### 단계 3: 확인 및 저장

패턴이 문서화되는 위치:
- `LEARNINGS.md` (있는 경우)
- 세션 노트
- 관련 스킬 파일

## 통합

### 스킬과 함께

학습된 패턴으로 기존 스킬 강화:

```markdown
# .claude/skills/python-patterns.md

## 학습된 패턴

### 세션 2024-01-15: 재시도 패턴
[패턴 세부사항 자동 추가]
```

### 규칙과 함께

버그 패턴이 새 규칙이 될 수 있음:

```markdown
# .claude/rules/common-bugs.md

## 순환 Import 방지
[학습된 버그 패턴에서 규칙 추가]
```

## 빠른 참조

```bash
# 현재 세션에서 학습 추출
/learn

# 특정 패턴 추출
/learn "JWT 인증 흐름"

# 버그 패턴 추출
/learn "순환 import 수정"
```

## 관련 명령어

- `/plan` - 구현 계획
- `/code-review` - 코드 검토
- `/refactor-clean` - 코드 정리

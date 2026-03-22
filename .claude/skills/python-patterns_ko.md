# Python 패턴 & 모범 사례

현대적인 Python 개발을 위한 도메인 지식입니다.

## 핵심 패턴

### 1. 의존성 주입
```python
from typing import Protocol

class Repository(Protocol):
    def get(self, id: str) -> dict | None: ...
    def save(self, item: dict) -> None: ...

class UserService:
    def __init__(self, repository: Repository):
        self._repo = repository

    def get_user(self, user_id: str) -> dict | None:
        return self._repo.get(user_id)

# 사용법
service = UserService(repository=PostgresRepository())
```

### 2. 팩토리 패턴
```python
from typing import Callable

ProcessorFactory = dict[str, Callable[[], "Processor"]]

_processors: ProcessorFactory = {}

def register_processor(name: str):
    def decorator(cls):
        _processors[name] = cls
        return cls
    return decorator

def create_processor(name: str) -> "Processor":
    if name not in _processors:
        raise ValueError(f"알 수 없는 프로세서: {name}")
    return _processors[name]()

@register_processor("json")
class JsonProcessor:
    def process(self, data): ...

@register_processor("xml")
class XmlProcessor:
    def process(self, data): ...
```

### 3. 컨텍스트 매니저
```python
from contextlib import contextmanager
from typing import Generator

@contextmanager
def database_transaction(db: Database) -> Generator[Session, None, None]:
    session = db.create_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# 사용법
with database_transaction(db) as session:
    session.add(user)
```

### 4. 데코레이터 패턴
```python
import functools
import time
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")

def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
            raise RuntimeError("도달 불가")
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def fetch_data(url: str) -> dict:
    ...
```

## 데이터 검증

### Pydantic 모델
```python
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class User(BaseModel):
    id: int
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("이름은 비어있을 수 없습니다")
        return v.strip()

# 검증이 자동으로 수행됨
user = User(id=1, name="John", email="john@example.com")
```

### pydantic-settings로 설정
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    debug: bool = False
    database_url: str
    api_key: str
    max_connections: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

## 비동기 패턴

### 비동기 컨텍스트 매니저
```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def get_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    async with httpx.AsyncClient() as client:
        yield client

async def fetch_data():
    async with get_client() as client:
        response = await client.get("https://api.example.com")
        return response.json()
```

### 동시 요청
```python
import asyncio
import httpx

async def fetch_all(urls: list[str]) -> list[dict]:
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for response in responses:
            if isinstance(response, Exception):
                results.append({"error": str(response)})
            else:
                results.append(response.json())
        return results
```

## 오류 처리

### 커스텀 예외 계층
```python
class AppError(Exception):
    """기본 애플리케이션 오류."""
    def __init__(self, message: str, code: str = "UNKNOWN"):
        self.message = message
        self.code = code
        super().__init__(message)

class ValidationError(AppError):
    """입력 검증 실패."""
    def __init__(self, message: str, field: str | None = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field

class NotFoundError(AppError):
    """리소스를 찾을 수 없음."""
    def __init__(self, resource: str, id: str):
        super().__init__(f"id {id}인 {resource}를 찾을 수 없습니다", "NOT_FOUND")
        self.resource = resource
        self.id = id
```

### Result 패턴
```python
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")

@dataclass
class Ok(Generic[T]):
    value: T

@dataclass
class Err(Generic[E]):
    error: E

Result = Ok[T] | Err[E]

def divide(a: int, b: int) -> Result[float, str]:
    if b == 0:
        return Err("0으로 나눔")
    return Ok(a / b)

# 사용법
match divide(10, 2):
    case Ok(value):
        print(f"결과: {value}")
    case Err(error):
        print(f"오류: {error}")
```

## 타입 힌트

### 제네릭 타입
```python
from typing import TypeVar, Generic
from collections.abc import Sequence

T = TypeVar("T")

class Repository(Generic[T]):
    def get(self, id: str) -> T | None: ...
    def list(self) -> Sequence[T]: ...
    def save(self, item: T) -> T: ...

class UserRepository(Repository[User]):
    ...
```

### 프로토콜 (구조적 서브타이핑)
```python
from typing import Protocol

class Readable(Protocol):
    def read(self) -> str: ...

class Writable(Protocol):
    def write(self, data: str) -> None: ...

class ReadWritable(Readable, Writable, Protocol):
    pass

def copy_data(src: Readable, dst: Writable) -> None:
    data = src.read()
    dst.write(data)
```

## 테스트 패턴

### 픽스처
```python
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_repository():
    repo = MagicMock()
    repo.get.return_value = {"id": 1, "name": "Test"}
    return repo

@pytest.fixture
def service(mock_repository):
    return UserService(repository=mock_repository)

def test_get_user(service, mock_repository):
    result = service.get_user("1")
    mock_repository.get.assert_called_once_with("1")
    assert result["name"] == "Test"
```

### 파라미터화된 테스트
```python
@pytest.mark.parametrize("input_val,expected", [
    ("hello", "HELLO"),
    ("World", "WORLD"),
    ("", ""),
])
def test_uppercase(input_val, expected):
    assert uppercase(input_val) == expected
```

## 성능 패턴

### 캐싱
```python
from functools import cache, lru_cache

@cache
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

@lru_cache(maxsize=100)
def expensive_query(key: str) -> dict:
    ...
```

### 지연 로딩
```python
class LazyLoader:
    def __init__(self):
        self._data: dict | None = None

    @property
    def data(self) -> dict:
        if self._data is None:
            self._data = self._load_data()
        return self._data

    def _load_data(self) -> dict:
        # 비용이 큰 작업
        ...
```

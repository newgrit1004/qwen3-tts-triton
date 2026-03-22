# Python Patterns & Best Practices

Domain knowledge for modern Python development.

## Core Patterns

### 1. Dependency Injection
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

# Usage
service = UserService(repository=PostgresRepository())
```

### 2. Factory Pattern
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
        raise ValueError(f"Unknown processor: {name}")
    return _processors[name]()

@register_processor("json")
class JsonProcessor:
    def process(self, data): ...

@register_processor("xml")
class XmlProcessor:
    def process(self, data): ...
```

### 3. Context Manager
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

# Usage
with database_transaction(db) as session:
    session.add(user)
```

### 4. Decorator Pattern
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
            raise RuntimeError("Unreachable")
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def fetch_data(url: str) -> dict:
    ...
```

## Data Validation

### Pydantic Models
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
            raise ValueError("Name cannot be empty")
        return v.strip()

# Validation happens automatically
user = User(id=1, name="John", email="john@example.com")
```

### Settings with pydantic-settings
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

## Async Patterns

### Async Context Manager
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

### Concurrent Requests
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

## Error Handling

### Custom Exception Hierarchy
```python
class AppError(Exception):
    """Base application error."""
    def __init__(self, message: str, code: str = "UNKNOWN"):
        self.message = message
        self.code = code
        super().__init__(message)

class ValidationError(AppError):
    """Input validation failed."""
    def __init__(self, message: str, field: str | None = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field

class NotFoundError(AppError):
    """Resource not found."""
    def __init__(self, resource: str, id: str):
        super().__init__(f"{resource} with id {id} not found", "NOT_FOUND")
        self.resource = resource
        self.id = id
```

### Result Pattern
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
        return Err("Division by zero")
    return Ok(a / b)

# Usage
match divide(10, 2):
    case Ok(value):
        print(f"Result: {value}")
    case Err(error):
        print(f"Error: {error}")
```

## Type Hints

### Generic Types
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

### Protocol (Structural Subtyping)
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

## Testing Patterns

### Fixtures
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

### Parametrized Tests
```python
@pytest.mark.parametrize("input_val,expected", [
    ("hello", "HELLO"),
    ("World", "WORLD"),
    ("", ""),
])
def test_uppercase(input_val, expected):
    assert uppercase(input_val) == expected
```

## Performance Patterns

### Caching
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

### Lazy Loading
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
        # Expensive operation
        ...
```

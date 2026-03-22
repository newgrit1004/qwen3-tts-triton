# 성능 규칙

성능 좋은 Python 코드를 작성하기 위한 가이드라인입니다.

## 일반 원칙

1. **먼저 측정**: 프로파일링 없이 최적화하지 않기
2. **병목 최적화**: 80%의 문제를 일으키는 20%에 집중
3. **가독성 vs 성능**: 성능이 중요하지 않으면 가독성 선호

## 일반적인 성능 패턴

### 리스트 컴프리헨션이 루프보다 빠름
```python
# 느림
result = []
for item in items:
    if item.is_valid:
        result.append(item.value)

# 빠름
result = [item.value for item in items if item.is_valid]
```

### 대용량 데이터에는 제너레이터 표현식
```python
# 메모리 많이 사용: 전체 리스트 생성
total = sum([x * 2 for x in range(1000000)])

# 메모리 효율적: 제너레이터
total = sum(x * 2 for x in range(1000000))
```

### 내장 함수 사용
```python
# 느림
total = 0
for num in numbers:
    total += num

# 빠름
total = sum(numbers)

# 느림
exists = False
for item in items:
    if condition(item):
        exists = True
        break

# 빠름
exists = any(condition(item) for item in items)
```

### 딕셔너리 작업
```python
# 느림: 여러 번 조회
if key in d:
    value = d[key]
else:
    value = default

# 빠름: 한 번 조회
value = d.get(key, default)

# defaultdict 패턴용
from collections import defaultdict
counts = defaultdict(int)
for item in items:
    counts[item] += 1
```

### 문자열 연결
```python
# 느림: 루프에서 문자열 연결
result = ""
for item in items:
    result += str(item)

# 빠름: Join
result = "".join(str(item) for item in items)

# 빠름: 알려진 항목에 f-string
result = f"{item1}{item2}{item3}"
```

## I/O를 위한 Async/Await

### 동시 I/O 작업
```python
import asyncio
import httpx

# 느림: 순차적
async def fetch_all_slow(urls):
    results = []
    async with httpx.AsyncClient() as client:
        for url in urls:
            response = await client.get(url)
            results.append(response)
    return results

# 빠름: 동시
async def fetch_all_fast(urls):
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        return await asyncio.gather(*tasks)
```

## 캐싱

### functools.cache 사용
```python
from functools import cache, lru_cache

# 해시 가능한 인자를 가진 순수 함수용
@cache
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# 크기 제한
@lru_cache(maxsize=100)
def expensive_computation(arg: str) -> dict:
    ...
```

### 수동 캐싱
```python
_cache: dict[str, Result] = {}

def get_data(key: str) -> Result:
    if key not in _cache:
        _cache[key] = expensive_fetch(key)
    return _cache[key]
```

## 데이터베이스 최적화

### 배치 작업
```python
# 느림: 개별 삽입
for item in items:
    db.execute("INSERT INTO table VALUES (%s)", (item,))

# 빠름: 배치 삽입
db.executemany("INSERT INTO table VALUES (%s)", [(item,) for item in items])

# SQLAlchemy로
session.bulk_insert_mappings(Model, [item.dict() for item in items])
```

### 쿼리 최적화
```python
# 느림: N+1 쿼리
users = session.query(User).all()
for user in users:
    print(user.orders)  # 각 사용자마다 쿼리 발생

# 빠름: 즉시 로딩
users = session.query(User).options(joinedload(User.orders)).all()
```

## 프로파일링 도구

### CPU 프로파일링
```bash
# cProfile
uv run python -m cProfile -s cumtime src/main.py

# line_profiler (특정 함수용)
uv add --dev line_profiler
```

```python
# @profile 데코레이터 추가하고 kernprof로 실행
@profile
def slow_function():
    ...
```

### 메모리 프로파일링
```bash
uv add --dev memory_profiler

# 메모리 사용량 프로파일
uv run python -m memory_profiler src/main.py
```

### 타이밍
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.4f}초")

with timer("작업"):
    expensive_operation()
```

## 성능 체크리스트

최적화 전:
- [ ] 병목을 식별하기 위해 프로파일링 수행
- [ ] 기준 성능 측정
- [ ] 성능 요구사항 문서화

최적화 후:
- [ ] 개선 측정
- [ ] 테스트 여전히 통과
- [ ] 코드 여전히 가독성 있음
- [ ] 최적화 문서화

## 최적화할 때

| 상황 | 조치 |
|------|------|
| 프로덕션 핫 패스 | 최적화 |
| 일회성 스크립트 | 최적화하지 않음 |
| 가독성 영향 큼 | 트레이드오프 고려 |
| 측정된 병목 없음 | 최적화하지 않음 |

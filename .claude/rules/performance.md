# Performance Rules

Guidelines for writing performant Python code.

## General Principles

1. **Measure First**: Don't optimize without profiling
2. **Optimize Bottlenecks**: Focus on the 20% that causes 80% of issues
3. **Readability vs Performance**: Prefer readability unless performance is critical

## Common Performance Patterns

### List Comprehensions Over Loops
```python
# SLOW
result = []
for item in items:
    if item.is_valid:
        result.append(item.value)

# FAST
result = [item.value for item in items if item.is_valid]
```

### Generator Expressions for Large Data
```python
# MEMORY HEAVY: Creates full list
total = sum([x * 2 for x in range(1000000)])

# MEMORY EFFICIENT: Generator
total = sum(x * 2 for x in range(1000000))
```

### Use Built-in Functions
```python
# SLOW
total = 0
for num in numbers:
    total += num

# FAST
total = sum(numbers)

# SLOW
exists = False
for item in items:
    if condition(item):
        exists = True
        break

# FAST
exists = any(condition(item) for item in items)
```

### Dictionary Operations
```python
# SLOW: Multiple lookups
if key in d:
    value = d[key]
else:
    value = default

# FAST: Single lookup
value = d.get(key, default)

# For defaultdict patterns
from collections import defaultdict
counts = defaultdict(int)
for item in items:
    counts[item] += 1
```

### String Concatenation
```python
# SLOW: String concatenation in loop
result = ""
for item in items:
    result += str(item)

# FAST: Join
result = "".join(str(item) for item in items)

# FAST: f-strings for known items
result = f"{item1}{item2}{item3}"
```

## Async/Await for I/O

### Concurrent I/O Operations
```python
import asyncio
import httpx

# SLOW: Sequential
async def fetch_all_slow(urls):
    results = []
    async with httpx.AsyncClient() as client:
        for url in urls:
            response = await client.get(url)
            results.append(response)
    return results

# FAST: Concurrent
async def fetch_all_fast(urls):
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        return await asyncio.gather(*tasks)
```

## Caching

### Use functools.cache
```python
from functools import cache, lru_cache

# For pure functions with hashable args
@cache
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# With size limit
@lru_cache(maxsize=100)
def expensive_computation(arg: str) -> dict:
    ...
```

### Manual Caching
```python
_cache: dict[str, Result] = {}

def get_data(key: str) -> Result:
    if key not in _cache:
        _cache[key] = expensive_fetch(key)
    return _cache[key]
```

## Database Optimization

### Batch Operations
```python
# SLOW: Individual inserts
for item in items:
    db.execute("INSERT INTO table VALUES (%s)", (item,))

# FAST: Batch insert
db.executemany("INSERT INTO table VALUES (%s)", [(item,) for item in items])

# With SQLAlchemy
session.bulk_insert_mappings(Model, [item.dict() for item in items])
```

### Query Optimization
```python
# SLOW: N+1 queries
users = session.query(User).all()
for user in users:
    print(user.orders)  # Triggers query for each user

# FAST: Eager loading
users = session.query(User).options(joinedload(User.orders)).all()
```

## Profiling Tools

### CPU Profiling
```bash
# cProfile
uv run python -m cProfile -s cumtime src/main.py

# line_profiler (for specific functions)
uv add --dev line_profiler
```

```python
# Add @profile decorator and run with kernprof
@profile
def slow_function():
    ...
```

### Memory Profiling
```bash
uv add --dev memory_profiler

# Profile memory usage
uv run python -m memory_profiler src/main.py
```

### Timing
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.4f}s")

with timer("operation"):
    expensive_operation()
```

## Performance Checklist

Before optimization:
- [ ] Profiled to identify bottleneck
- [ ] Measured baseline performance
- [ ] Documented performance requirements

After optimization:
- [ ] Measured improvement
- [ ] Tests still pass
- [ ] Code is still readable
- [ ] Documented the optimization

## When to Optimize

| Situation | Action |
|-----------|--------|
| Hot path in production | Optimize |
| One-time script | Don't optimize |
| Readability impact is high | Consider trade-offs |
| No measured bottleneck | Don't optimize |

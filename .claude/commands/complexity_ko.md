# /complexity 명령어

complexipy를 사용하여 Python 함수의 인지 복잡도를 분석합니다.

## 사용법
```
/complexity [경로]
```

**인자:**
- `경로`: 분석할 디렉토리 또는 파일 (기본값: `src/`)

## 이 명령어의 역할

1. **실행**: 지정된 경로에서 complexipy 실행
2. **식별**: 복잡도 임계값을 초과하는 함수
3. **제안**: 각 함수에 대한 리팩토링 전략

## 복잡도 임계값

| 점수 | 수준 | 필요한 조치 |
|------|------|-------------|
| 1-10 | 낮음 | 허용 가능 |
| 11-15 | 중간 | 리팩토링 고려 |
| 16-20 | 높음 | 리팩토링 권장 |
| 21+ | 심각 | 반드시 리팩토링 |

## 실행 단계

### 단계 1: 복잡도 분석 실행
```bash
uv run complexipy [경로] --max-complexity-allowed 15
```

### 단계 2: 결과 분석
임계값을 초과하는 각 함수에 대해:
- 복잡도 원인 식별
- 구체적인 수정 방안 제안
- 복잡도 점수로 우선순위 지정

### 단계 3: 보고서 생성

```markdown
## 복잡도 분석 보고서

### 요약
- 분석한 파일: 10개
- 분석한 함수: 45개
- 임계값 초과 함수: 3개

### 높은 복잡도 함수

| 함수 | 파일:줄 | 점수 | 이슈 |
|------|---------|------|------|
| process_data | src/main.py:25 | 18 | 중첩 루프 + 조건문 |
| validate_input | src/utils.py:50 | 15 | 다수의 조기 반환 |
| transform_result | src/api.py:100 | 12 | 깊은 중첩 |

### 권장 수정 사항

#### 1. `process_data` (점수: 18)
**위치**: src/main.py:25
**이슈**:
- 3단계 중첩 루프
- 다수의 조건 분기

**제안**:
- 내부 루프를 별도 함수로 추출
- 조기 반환 패턴 사용
- 리스트 컴프리헨션 고려

#### 2. `validate_input` (점수: 15)
**위치**: src/utils.py:50
**제안**:
- 검증 라이브러리 사용 (pydantic)
- 검증 규칙을 별도 함수로 추출
```

## 인지 복잡도 요인

| 요인 | 영향 | 완화 방법 |
|------|------|-----------|
| 중첩 if/else | 레벨당 +1 | 조기 반환으로 평탄화 |
| 루프 (for/while) | 레벨당 +1 | 함수로 추출 |
| try/except | +1 | 핸들러 통합 |
| 논리 연산자 (and/or) | 연산자당 +1 | 불린 함수로 추출 |
| 재귀 | +1 | 반복으로 대체 고려 |

## 빠른 수정

### 패턴: 깊은 중첩 → 조기 반환
```python
# 이전 (복잡도: 5)
def process(data):
    if data:
        if data.is_valid:
            if data.has_items:
                return data.items
    return []

# 이후 (복잡도: 2)
def process(data):
    if not data:
        return []
    if not data.is_valid:
        return []
    if not data.has_items:
        return []
    return data.items
```

### 패턴: 복잡한 조건문 → 헬퍼 함수
```python
# 이전 (복잡도: 4)
if user.age >= 18 and user.is_verified and user.has_subscription:
    process(user)

# 이후 (복잡도: 1)
def is_eligible(user):
    return user.age >= 18 and user.is_verified and user.has_subscription

if is_eligible(user):
    process(user)
```

## 다른 도구와의 통합

`/complexity` 실행 후:

1. **이슈 수정**: 자동 수정 가능한 이슈는 `/refactor-clean` 사용
2. **변경사항 테스트**: 리팩토링 후 `make test` 실행
3. **검증**: `/complexity` 재실행하여 개선 확인

## 관련 명령어

- `/dead-code` - 제거할 사용하지 않는 코드 찾기
- `/code-quality` - 전체 품질 분석 실행
- `/refactor-clean` - 코드 이슈 자동 수정

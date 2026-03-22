# /refactor-clean 명령어

체계적으로 코드 품질을 개선합니다.

## 사용법
```
/refactor-clean [파일 또는 디렉토리]
```

## 이 명령어의 역할

1. **호출**: `refactor-cleaner` 에이전트
2. **분석**: 개선 기회를 위한 코드
3. **리팩토링**: 기능 유지하며 개선

## 분석 대상

- 50줄 초과 함수
- 500줄 초과 파일
- 4단계 초과 중첩
- 중복 코드
- 코드 스멜

## 리팩토링 프로세스

### 단계 1: 테스트 커버리지 확보
```bash
make test-cov
# 리팩토링 전 80%+ 커버리지 확인
```

### 단계 2: 이슈 식별
- 정적 분석 실행
- 복잡도 지표 확인
- 코드 스멜 찾기

### 단계 3: 점진적 리팩토링
- 한 번에 하나의 변경
- 각 변경 후 테스트 실행
- git 히스토리 유지

### 단계 4: 검증
```bash
make test      # 테스트 여전히 통과
make lint      # 새 경고 없음
make typecheck # 타입 유효
```

## 일반적인 리팩토링

### 함수 추출
```python
# 이전
def process(data):
    # 75줄의 코드
    ...

# 이후
def process(data):
    validated = validate_input(data)
    transformed = transform_data(validated)
    return format_output(transformed)
```

### 조기 반환
```python
# 이전
def check(user):
    if user:
        if user.active:
            if user.verified:
                return True
    return False

# 이후
def check(user):
    if not user:
        return False
    if not user.active:
        return False
    if not user.verified:
        return False
    return True
```

### 매직 넘버 대체
```python
# 이전
if age >= 18:
    discount = price * 0.1

# 이후
ADULT_AGE = 18
DISCOUNT_RATE = Decimal("0.10")

if age >= ADULT_AGE:
    discount = price * DISCOUNT_RATE
```

## 출력 보고서

```markdown
## 리팩토링 보고서

### 분석된 파일
- `src/processor.py` (450줄)

### 발견된 이슈
| 이슈 | 위치 | 조치 |
|------|------|------|
| 긴 함수 | process:45-120 | 3개 함수 추출 |
| 깊은 중첩 | validate:30-50 | 조기 반환 적용 |

### 수행된 변경
1. `process()`에서 `validate_input()` 추출
2. `process()`에서 `transform_data()` 추출
3. `validate()`에 조기 반환 패턴 적용

### 지표
| 지표 | 이전 | 이후 |
|------|------|------|
| 최대 함수 줄 | 75줄 | 20줄 |
| 최대 중첩 | 5 | 2 |
| 복잡도 | 15 | 5 |
```

## 검증

```bash
make test      # 모든 테스트 통과
make lint      # 깨끗함
make typecheck # 오류 없음
make test-cov  # 커버리지 유지
```

## 관련 명령어

- `/plan` - 대규모 리팩토링 전 계획
- `/tdd` - 리팩토링 전 테스트 추가
- `/code-review` - 리팩토링 후 리뷰

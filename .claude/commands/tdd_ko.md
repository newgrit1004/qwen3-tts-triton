# /tdd 명령어

테스트 주도 개발을 위한 TDD 워크플로우를 시작합니다.

## 사용법
```
/tdd <기능 설명>
```

## 이 명령어의 역할

1. **호출**: `tdd-guide` 에이전트
2. **시행**: RED-GREEN-REFACTOR 사이클
3. **보장**: 80%+ 테스트 커버리지

## 워크플로우

### 단계 1: 기능 이해
기능 요구사항을 분석하고 식별:
- 입/출력 명세
- 엣지 케이스
- 오류 조건

### 단계 2: RED 단계 - 실패하는 테스트 작성
```python
# tests/test_feature.py
def test_feature_basic_case():
    result = feature_function(valid_input)
    assert result == expected_output

def test_feature_edge_case():
    with pytest.raises(ValueError):
        feature_function(invalid_input)
```

### 단계 3: 테스트 실패 확인
```bash
make test
# 예상: FAILED
```

### 단계 4: GREEN 단계 - 최소 구현
```python
# src/feature.py
def feature_function(input_data):
    # 테스트를 통과하는 최소한의 코드
    return expected_output
```

### 단계 5: 테스트 통과 확인
```bash
make test
# 예상: PASSED
```

### 단계 6: REFACTOR 단계
- 코드 품질 개선
- 타입 힌트 추가
- 독스트링 추가
- 린팅 실행

```bash
make lint-fix
make format
make typecheck
```

### 단계 7: 반복
다음 테스트 케이스로 계속 진행.

## 커버리지 요구사항

| 코드 유형 | 커버리지 |
|-----------|----------|
| 일반 | 80% |
| 비즈니스 로직 | 100% |
| 보안 코드 | 100% |

## 검증 명령어

```bash
make test       # 테스트 실행
make test-cov   # 커버리지와 함께 실행
make lint       # 린팅 확인
make typecheck  # 타입 확인
```

## 예제 세션

```
사용자: /tdd 할인이 적용된 주문 총액 계산 함수 추가

Claude:
1. 먼저 실패하는 테스트를 작성합니다...
   [test_order.py 테스트 케이스 생성]

2. 테스트 실행 - 실패해야 합니다...
   [make test - 예상대로 FAILED]

3. 이제 최소 솔루션을 구현합니다...
   [order.py에 calculate_total 함수 생성]

4. 테스트 실행 - 통과해야 합니다...
   [make test - PASSED]

5. 리팩토링 및 타입 힌트 추가...
   [타입, 독스트링으로 코드 업데이트]

6. 최종 검증...
   [make lint, make typecheck, make test-cov]
```

## 관련 명령어

- `/plan` - 구현 전 계획
- `/code-review` - 구현 후 리뷰
- `/refactor-clean` - 코드 정리

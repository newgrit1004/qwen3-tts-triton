# /security-check 명령어

포괄적인 보안 분석을 수행합니다.

## 사용법
```
/security-check [파일 또는 디렉토리]
```

## 이 명령어의 역할

1. **호출**: `security-reviewer` 에이전트
2. **스캔**: 취약점을 위한 코드
3. **보고**: 수정 방법과 함께 보안 이슈

## 보안 검사

### 코드 분석
- 하드코딩된 자격증명
- SQL 인젝션 벡터
- 경로 순회 위험
- 명령 인젝션
- 안전하지 않은 역직렬화
- SSRF 취약점

### 의존성 분석
- 알려진 취약점 (CVE)
- 오래된 패키지
- 안전하지 않은 의존성

## 스캔 프로세스

### 단계 1: 정적 분석
```bash
uv run bandit -r src/
```

### 단계 2: 의존성 감사
```bash
uv run safety check
uv run pip-audit
```

### 단계 3: 비밀 탐지
```bash
uv run detect-secrets scan
```

## 출력 보고서

```markdown
## 보안 스캔 보고서

### 요약
- 스캔한 파일: 15개
- 심각: 0개
- 높음: 1개
- 중간: 2개
- 낮음: 3개

### 심각 발견사항
없음

### 높음 발견사항

#### [높음] SQL 인젝션 위험
**파일**: `src/db.py:45`
**코드**: `query = f"SELECT * FROM users WHERE id = {user_id}"`
**수정**: 파라미터화된 쿼리 사용
```python
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, (user_id,))
```

### 중간 발견사항
...

### 권장사항
1. pre-commit 훅에 Bandit 활성화
2. CI 파이프라인에 safety 검사 추가
3. 취약한 의존성 업데이트
```

## 심각도 수준

| 수준 | 설명 | 대응 |
|------|------|------|
| 심각 | 적극적 악용 가능 | 즉시 수정 |
| 높음 | 상당한 위험 | 커밋 전 수정 |
| 중간 | 보통 위험 | 곧 수정 |
| 낮음 | 경미한 위험 | 편할 때 수정 |

## 필요한 도구

```bash
# 보안 도구 설치
uv add --dev bandit safety pip-audit detect-secrets
```

## CI 통합

`.github/workflows/ci.yml`에 추가:
```yaml
- name: 보안 스캔
  run: |
    uv run bandit -r src/
    uv run safety check
```

## 관련 명령어

- `/code-review` - 일반 코드 리뷰
- `/plan` - 보안 개선 계획

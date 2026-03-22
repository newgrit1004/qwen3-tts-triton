# 보안 규칙

이 규칙들은 이 프로젝트의 모든 코드에 반드시 따라야 합니다.

## 커밋 전 보안 체크리스트

모든 커밋 전 확인:

- [ ] 하드코딩된 비밀 없음 (API 키, 비밀번호, 토큰)
- [ ] 모든 사용자 입력 검증됨
- [ ] SQL 쿼리는 파라미터화된 문 사용
- [ ] 파일 경로 정제됨 (경로 순회 없음)
- [ ] 오류 메시지에 민감 정보 노출 없음
- [ ] 의존성 최신 상태

## 심각: 하드코딩된 비밀 금지

### 절대 하지 마세요
```python
# 심각한 위반
API_KEY = "sk-1234567890abcdef"
DB_PASSWORD = "supersecret123"
AWS_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
```

### 항상 이렇게 하세요
```python
import os

# 환경 변수에서
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY 환경 변수가 필요합니다")

# pydantic-settings 사용
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str
    database_url: str

    class Config:
        env_file = ".env"

settings = Settings()
```

## 입력 검증

### 모든 외부 입력 검증
```python
from pydantic import BaseModel, Field, validator

class UserInput(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    age: int = Field(ge=0, le=150)

    @validator("username")
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError("사용자명은 영숫자여야 합니다")
        return v
```

## SQL 인젝션 방지

### 절대 하지 마세요
```python
# 심각한 취약점
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)
```

### 항상 이렇게 하세요
```python
# 파라미터화된 쿼리
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, (user_id,))

# ORM 사용 (SQLAlchemy)
user = session.query(User).filter(User.id == user_id).first()
```

## 경로 순회 방지

### 절대 하지 마세요
```python
# 취약점: 사용자가 모든 파일에 접근 가능
file_path = f"/data/{user_input}"
with open(file_path) as f:
    return f.read()
```

### 항상 이렇게 하세요
```python
from pathlib import Path

SAFE_BASE_DIR = Path("/data")

def safe_read_file(filename: str) -> str:
    # 절대 경로로 해석
    requested = (SAFE_BASE_DIR / filename).resolve()

    # 허용된 디렉토리 내인지 확인
    if not requested.is_relative_to(SAFE_BASE_DIR):
        raise ValueError("접근 거부: 잘못된 경로")

    if not requested.exists():
        raise FileNotFoundError("파일을 찾을 수 없습니다")

    return requested.read_text()
```

## 안전한 오류 처리

### 절대 하지 마세요
```python
# 취약점: 내부 세부사항 노출
try:
    result = database.query(sql)
except Exception as e:
    return f"데이터베이스 오류: {e}"  # SQL, 연결 정보 노출
```

### 항상 이렇게 하세요
```python
import logging

logger = logging.getLogger(__name__)

try:
    result = database.query(sql)
except DatabaseError as e:
    logger.error(f"데이터베이스 쿼리 실패: {e}", exc_info=True)
    raise ServiceError("요청 처리 중 오류가 발생했습니다")
```

## 의존성 보안

### 정기적인 취약점 검사
```bash
# 취약점 검사
uv add --dev safety
uv run safety check

# 대안: pip-audit
uv add --dev pip-audit
uv run pip-audit
```

### 의존성 최신 유지
```bash
# 모든 의존성 업데이트
make update

# 오래된 패키지 확인
uv pip list --outdated
```

## 로깅 보안

### 민감한 데이터 절대 로깅하지 않기
```python
# 위반
logger.info(f"사용자 {username}이 비밀번호 {password}로 로그인")
logger.debug(f"API 응답: {api_key}")
```

### 안전한 로깅
```python
# 올바름
logger.info(f"사용자 {username} 로그인 성공")
logger.debug("API 호출 완료", extra={"status": response.status_code})
```

## 환경 파일 규칙

### .env.example (이것은 커밋)
```env
# 필수 환경 변수
API_KEY=your_api_key_here
DATABASE_URL=postgresql://user:pass@localhost/db
SECRET_KEY=generate_a_secure_key
```

### .env (절대 커밋하지 마세요)
```env
API_KEY=sk-actual-production-key
DATABASE_URL=postgresql://prod:realpass@prod-db/production
SECRET_KEY=super-secret-production-key
```

### .gitignore (필수 항목)
```gitignore
# 환경 파일
.env
.env.local
.env.*.local

# 자격증명
*.pem
*.key
credentials.json
secrets/
*_secret*
*_credentials*
```

## 보안 이슈 대응 프로토콜

보안 취약점 발견 시:

1. **중단** - 모든 현재 작업 중단
2. **평가** - 심각도 평가 (심각/높음/중간/낮음)
3. **보고** - 정확한 위치와 함께 발견사항 보고
4. **수정** - 심각한 이슈 즉시 수정
5. **무효화** - 노출된 자격증명 무효화
6. **스캔** - 코드베이스에서 유사 이슈 스캔
7. **문서화** - 사건과 수정 내용 문서화

## 보안 도구

```bash
# 정적 보안 분석
uv add --dev bandit
uv run bandit -r src/

# 비밀 탐지
uv add --dev detect-secrets
uv run detect-secrets scan

# 의존성 취약점
uv run safety check
uv run pip-audit
```

## 보안 심각도 수준

| 수준 | 설명 | 대응 시간 |
|------|------|-----------|
| 심각 | 적극적 악용 가능 | 즉시 |
| 높음 | 상당한 위험 | 몇 시간 내 |
| 중간 | 보통 위험 | 며칠 내 |
| 낮음 | 경미한 위험 | 다음 릴리스 |

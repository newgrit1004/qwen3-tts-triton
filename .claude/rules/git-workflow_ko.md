# Git 워크플로우 규칙

이 규칙들은 이 프로젝트의 모든 Git 작업에 반드시 따라야 합니다.

## 커밋 메시지 형식

```
<type>: <subject>

[body]

[footer]
```

### 타입
| 타입 | 설명 |
|------|------|
| `feat` | 새 기능 |
| `fix` | 버그 수정 |
| `refactor` | 코드 리팩토링 |
| `docs` | 문서화 |
| `test` | 테스트 추가/업데이트 |
| `chore` | 유지보수 작업 |
| `perf` | 성능 개선 |
| `ci` | CI/CD 변경 |

### Subject 규칙
- 50자 이하
- 명령형 ("Add feature"가 아닌 "기능 추가")
- 끝에 마침표 없음
- 첫 글자 대문자

### 예시
```bash
# 좋음
feat: 사용자 인증 엔드포인트 추가
fix: 주문 처리의 경쟁 조건 해결
refactor: 검증 로직을 별도 모듈로 추출
docs: v2 엔드포인트 API 문서 업데이트

# 나쁨
added new feature
버그 수정.
REFACTOR: 코드 업데이트됨
```

## Pre-Commit 훅

모든 커밋은 pre-commit 검사를 통과해야 합니다:

1. **파일 검사**
   - 대용량 파일 없음 (> 500KB)
   - 유효한 JSON/YAML/TOML 문법
   - 머지 충돌 없음

2. **코드 품질**
   - Ruff 린팅 (자동 수정 활성화)
   - Ruff 포맷팅
   - Ty 타입 검사

### 수동 Pre-commit 실행
```bash
make pre-commit
```

## 브랜치 전략

### 메인 브랜치
- `main` - 프로덕션 준비된 코드
- `develop` - 통합 브랜치 (사용하는 경우)

### 기능 브랜치
```bash
# 패턴: <type>/<description>
feature/user-authentication
fix/order-validation-bug
refactor/database-layer
```

### 브랜치 워크플로우
```bash
# 기능 브랜치 생성
git checkout -b feature/new-feature

# 작업 및 커밋
git add .
git commit  # 커밋 템플릿 사용

# 푸시 및 PR 생성
git push -u origin feature/new-feature
```

## 풀 리퀘스트 프로세스

### PR 생성 전

1. **모든 검사 통과 확인**
   ```bash
   make lint        # Ruff 린트
   make format      # Ruff 포맷
   make typecheck   # Ty 타입 검사
   make test        # 모든 테스트 통과
   ```

2. **모든 변경사항 검토**
   ```bash
   git diff main...HEAD
   git log main...HEAD
   ```

### PR 설명 템플릿
```markdown
## 요약
변경사항에 대한 간략한 설명

## 변경 내용
- 변경 1
- 변경 2

## 테스트 계획
- [ ] 단위 테스트 추가/업데이트
- [ ] 통합 테스트 추가/업데이트
- [ ] 수동 테스트 완료

## 체크리스트
- [ ] 코드가 프로젝트 스타일 가이드라인 준수
- [ ] 로컬에서 테스트 통과
- [ ] 문서 업데이트됨 (필요한 경우)
- [ ] 보안 취약점 도입 없음
```

## 개발 워크플로우

### 1. 계획 단계
```bash
/plan  # 플래너 에이전트 호출
```

### 2. 테스트 주도 개발
```bash
/tdd  # RED-GREEN-REFACTOR 따르기
```

### 3. 코드 리뷰
```bash
/code-review  # 자동화된 리뷰 받기
```

### 4. 커밋
```bash
git add .
git commit  # 템플릿 에디터 열림, pre-commit 실행
```

### 5. 푸시
```bash
git push -u origin <branch>
```

## 커밋 체크리스트

커밋 전:

- [ ] 코드 컴파일/오류 없이 실행
- [ ] 모든 테스트 통과 (`make test`)
- [ ] Ruff 린트 통과 (`make lint`)
- [ ] Ty 타입체크 통과 (`make typecheck`)
- [ ] 디버그 문 없음 (`print()`, `breakpoint()`)
- [ ] 하드코딩된 비밀 없음
- [ ] 커밋 메시지 형식 준수

## Git 명령어 참조

```bash
# 상태 및 diff
git status
git diff
git diff --staged

# 스테이징
git add <file>
git add -p  # 대화형 스테이징

# 커밋
git commit  # 템플릿 열기
git commit -m "type: message"

# 브랜치
git branch -a
git checkout -b <branch>
git switch <branch>

# 동기화
git fetch origin
git pull origin main
git push -u origin <branch>

# 히스토리
git log --oneline -10
git log --graph --oneline

# 되돌리기
git restore <file>          # 변경 취소
git restore --staged <file> # 스테이징 해제
git reset --soft HEAD~1     # 마지막 커밋 취소 (변경 유지)
```

## 보호된 작업

### 추가 주의 필요
- `git push --force` - 대신 `--force-with-lease` 사용
- `git reset --hard` - 작업 손실 가능
- 공유 브랜치에서 `git rebase`

### main에서 절대 하지 말 것
```bash
# 하지 마세요
git push --force origin main
git reset --hard on main
git rebase main (다른 사람이 사용 중일 때)
```

## 충돌 해결

1. **최신 변경사항 가져오기**
   ```bash
   git fetch origin
   ```

2. **main에 리베이스**
   ```bash
   git rebase origin/main
   ```

3. **충돌 해결**
   - 충돌된 파일 편집
   - 충돌 마커 제거
   - 해결된 파일 스테이징

4. **리베이스 계속**
   ```bash
   git rebase --continue
   ```

5. **검증**
   ```bash
   make test
   make lint
   ```

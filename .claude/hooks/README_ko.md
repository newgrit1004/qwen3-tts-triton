# Claude Code 훅

이 디렉토리는 Claude Code 자동화를 위한 훅 설정을 포함합니다.

## 훅 유형

### PreToolUse
도구 실행 전에 실행됩니다. 할 수 있는 것:
- `allow` - 작업 허용
- `pause` - 확인 요청
- `block` - 작업 차단
- `warn` - 경고 표시 후 계속

### PostToolUse
도구 완료 후 실행됩니다. 할 수 있는 것:
- `run` - 명령어 실행
- `log` - 메시지 로깅
- `notify` - 알림 전송

### Stop
Claude Code 세션 종료 시 실행됩니다. 할 수 있는 것:
- `check` - 검증 검사 실행
- `remind` - 리마인더 표시

## 설정

### hooks.json 구조

```json
{
  "hooks": {
    "PreToolUse": [...],
    "PostToolUse": [...],
    "Stop": [...]
  },
  "settings": {
    "auto_format": true,
    ...
  }
}
```

### 매처

```json
{
  "matcher": {
    "tool": "Bash|Write|Edit|...",
    "command_pattern": "정규식 패턴",
    "file_pattern": "정규식 패턴"
  }
}
```

## 현재 훅

### PreToolUse
1. **UV 명령어 허용** - `uv run` 명령어 허용
2. **git push 확인** - 푸시 전 일시 중지
3. **마크다운 생성 경고** - 문서 생성 시 경고

### PostToolUse
1. **Python 자동 포맷** - Python 파일 변경 후 Ruff 실행
2. **PR 생성 로깅** - 리뷰 요청 리마인더

### Stop
1. **print() 검사** - 디버그 문 경고
2. **breakpoint() 검사** - 디버거 호출 경고
3. **비밀 검사** - 하드코딩된 자격증명 경고

## 사용법

이 훅을 적용하려면 Claude Code 설정에 설정을 복사:

```bash
# 전역 설정에 복사
cp hooks.json ~/.claude/settings.json
```

또는 프로젝트 레벨 설정에서 참조.

## 커스터마이징

특정 워크플로우 요구에 맞게 `hooks.json`을 편집하여 훅을 추가하거나 수정하세요.

# 자율 에이전트 워크플로우

다중 세션을 통한 장기 자율 태스크 실행에 대한 도메인 지식입니다.

## 핵심 개념

자율 에이전트 시스템은 상태를 영속화하면서 여러 세션을 실행하여 컨텍스트 윈도우 한계를 극복합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    세션 기반 실행                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   세션 1            세션 2            세션 3                      │
│   ┌────────┐        ┌────────┐        ┌────────┐               │
│   │ 태스크 1 │   →    │ 태스크 4 │   →    │ 태스크 7 │               │
│   │ 태스크 2 │        │ 태스크 5 │        │ 태스크 8 │               │
│   │ 태스크 3 │        │ 태스크 6 │        │ ...    │               │
│   └───┬────┘        └───┬────┘        └───┬────┘               │
│       │                 │                 │                     │
│       ▼                 ▼                 ▼                     │
│   ┌─────────────────────────────────────────────┐               │
│   │           progress.json (영속화)              │               │
│   │   - 완료된 태스크 목록                         │               │
│   │   - 다음 실행할 태스크                         │               │
│   │   - 메타데이터 (시작시간, 세션 수)             │               │
│   └─────────────────────────────────────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 작동 방식

### 1. 태스크 정의 (tasks.json)
```json
{
  "tasks": [
    {"id": "task_001", "description": "트렌드 주제 리서치", "agent": "research"},
    {"id": "task_002", "description": "스크립트 생성", "agent": "script"},
    {"id": "task_003", "description": "오디오 생성", "agent": "audio"}
  ]
}
```

### 2. 세션 실행 루프
```python
while not all_tasks_complete:
    session = new_claude_session()  # Fresh 컨텍스트
    load_progress()                 # 이미 완료된 것은?
    next_task = get_next_pending()

    execute_task(session, next_task)
    save_progress()                 # 상태 영속화

    session.close()
    wait(delay)                     # Rate limiting
```

### 3. 진행 상황 영속화
```json
{
  "created_at": "2025-01-20T22:00:00",
  "session_count": 5,
  "tasks": [...],
  "completed_tasks": ["task_001", "task_002"],
  "current_task_index": 2
}
```

## Ralph 모드와의 관계

| 측면 | Ralph 모드 | 자율 에이전트 |
|------|-----------|--------------|
| 범위 | **세션 내** | **세션 간** |
| 컨텍스트 | 단일 윈도우 | 다중 Fresh 컨텍스트 |
| 영속화 | 메모리 내 할 일 | 파일 기반 (JSON) |
| 지속 시간 한계 | 컨텍스트 윈도우 | **없음** (36시간+ 가능) |
| 사용 사례 | 복잡한 단일 작업 | 장기 자동화 파이프라인 |

### 상호보완적 사용
Ralph 모드는 각 자율 에이전트 세션 **내부에서** 작동합니다:

```
┌─────────────────────────────────────────┐
│ 자율 에이전트 세션                        │
│ ┌─────────────────────────────────────┐ │
│ │ Ralph 모드 (세션 내)                  │ │
│ │ - 태스크 분해                         │ │
│ │ - 할 일 추적                          │ │
│ │ - 완료 검증                           │ │
│ └─────────────────────────────────────┘ │
│ → progress.json에 저장                   │
│ → 다음 세션 시작                          │
└─────────────────────────────────────────┘
```

## 에이전트 실행

### 사전 요구사항
```bash
# Claude Code SDK 설치
pip install claude-code-sdk

# OAuth 토큰 설정
export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'
```

### 기본 실행
```bash
# 태스크 파일로 실행
python -m src.autonomous_agent.run \
    --project-dir . \
    --tasks-file tasks.json

# 제한된 반복으로 테스트
python -m src.autonomous_agent.run \
    --project-dir . \
    --tasks-file tasks.json \
    --max-iterations 3
```

### 옵션
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--project-dir` | `./project` | 작업 디렉토리 |
| `--tasks-file` | None | 태스크 정의 JSON |
| `--max-iterations` | 무제한 | 세션 한계 |
| `--model` | claude-sonnet-4-5 | 사용할 모델 |
| `--delay` | 3 | 세션 간 대기 시간(초) |

## 태스크 정의 형식

### 기본 구조
```json
{
  "project_name": "내 프로젝트",
  "config": {
    "model": "claude-sonnet-4-5-20250929"
  },
  "tasks": [
    {
      "id": "unique_task_id",
      "description": "수행할 작업",
      "agent": "agent_type",
      "status": "pending",
      "dependencies": ["other_task_id"]
    }
  ]
}
```

### 태스크 상태 값
- `pending` - 시작 안 됨
- `in_progress` - 현재 실행 중
- `complete` - 성공적으로 완료
- `failed` - 오류 발생

### 의존성
태스크는 다른 태스크에 의존할 수 있습니다:
```json
{
  "id": "visual_003",
  "description": "자막 생성",
  "dependencies": ["audio_002", "visual_002"]
}
```

## 보안

### 명령어 허용 목록
보안 모듈이 bash 명령어를 제한합니다:

```python
ALLOWED_COMMANDS = {
    "ls", "cat", "grep", "find",  # 읽기 전용
    "mkdir", "cp", "mv",          # 파일 작업
    "npm", "python", "uv",        # 패키지 매니저
    "git",                        # 버전 관리
}
```

### 위험 패턴 차단
```python
DANGEROUS_PATTERNS = [
    "sudo", "rm -rf /",
    "> /dev/", "nc -l",
]
```

## 모범 사례

### 1. 태스크 세분화
- 큰 태스크를 작고 검증 가능한 단위로 분해
- 각 태스크는 한 세션에서 완료 가능해야 함
- 명확한 완료 기준 포함

### 2. 오류 처리
```json
{
  "id": "task_005",
  "retry_count": 3,
  "on_failure": "skip"  // 또는 "stop", "retry"
}
```

### 3. 진행 상황 모니터링
```bash
# 실시간 진행 상황 확인
watch -n 5 cat progress.json | jq '.completed_tasks | length'
```

### 4. 중단 및 재개
- `Ctrl+C`로 정상 중단
- 같은 명령어로 마지막 체크포인트에서 재개

## 프로젝트 구조와의 통합

```
project/
├── src/
│   └── autonomous_agent/
│       ├── __init__.py
│       ├── run.py          # 진입점
│       ├── agent.py        # 세션 로직
│       ├── progress.py     # 상태 영속화
│       ├── client.py       # SDK 설정
│       └── security.py     # 명령어 허용 목록
├── prompts/
│   ├── system.md           # 에이전트 성격
│   └── task.md             # 태스크 실행 템플릿
├── tasks.json              # 태스크 정의
└── progress.json           # 런타임 상태 (자동 생성)
```

## 예시: YouTube Shorts 파이프라인

```json
{
  "tasks": [
    {"id": "research", "description": "주제 리서치"},
    {"id": "script", "description": "스크립트 생성"},
    {"id": "audio", "description": "TTS + BGM 믹싱"},
    {"id": "visual", "description": "이미지 생성 + 영상 합성"},
    {"id": "qa", "description": "품질 검증"},
    {"id": "publish", "description": "YouTube 업로드"}
  ]
}
```

이 파이프라인은 6단계 모두 완료될 때까지 여러 세션에 걸쳐 자동으로 실행됩니다.

# MCP 설정

향상된 Claude Code 통합을 위한 Model Context Protocol (MCP) 서버 설정입니다.

## 개요

MCP 서버는 외부 도구와 서비스에 대한 접근을 제공하여 Claude Code의 기능을 확장합니다. 이 디렉토리에는 일반적인 통합을 위한 샘플 설정이 포함되어 있습니다.

## 중요 가이드라인

[everything-claude-code](https://github.com/affaan-m/everything-claude-code)에서:

> "모든 MCP를 한 번에 활성화하지 마세요. 너무 많은 도구가 활성화되면 200k 컨텍스트 윈도우가 70k로 줄어들 수 있습니다."

**권장 한도:**
- 총 20-30개 설정된 MCP
- 프로젝트당 10개 미만 활성화
- 80개 미만의 활성 도구

## 설정 위치

MCP 서버는 `~/.claude.json`에서 설정됩니다:

```json
{
  "mcpServers": {
    "github": { ... },
    "database": { ... }
  }
}
```

## 사용 가능한 설정

### 1. GitHub (`github.json`)

GitHub 저장소, 이슈, PR에 접근합니다.

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "<your-token>"
      }
    }
  }
}
```

**기능:**
- 이슈 생성/읽기/업데이트
- PR 생성/리뷰
- 저장소 컨텐츠 접근
- 코드 검색

### 2. 데이터베이스 (`database.json`)

쿼리를 위한 직접 데이터베이스 접근.

```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost/db"
      }
    }
  }
}
```

**기능:**
- SQL 쿼리 실행
- 스키마 검사
- 데이터 읽기/쓰기

### 3. 파일시스템 (`filesystem.json`)

향상된 파일시스템 작업.

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
    }
  }
}
```

**기능:**
- 파일 읽기/쓰기
- 디렉토리 작업
- 파일 검색

### 4. 메모리 (`memory.json`)

세션 간 영속적 메모리.

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

**기능:**
- 키-값 쌍 저장
- 세션 간 영속화
- 지식 그래프 지원

## 설정 방법

### 단계 1: Node.js 설치

MCP 서버는 Node.js가 필요합니다:

```bash
# macOS
brew install node

# Ubuntu
sudo apt install nodejs npm

# 또는 nvm 사용
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install node
```

### 단계 2: 설정 생성

`~/.claude.json` 생성 또는 편집:

```bash
# 없으면 생성
touch ~/.claude.json

# 설정 추가
cat > ~/.claude.json << 'EOF'
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "<your-token>"
      }
    }
  }
}
EOF
```

### 단계 3: 환경 변수 설정

민감한 값에는 환경 변수 사용:

```bash
# .bashrc 또는 .zshrc
export GITHUB_PERSONAL_ACCESS_TOKEN="ghp_xxxxxxxxxxxx"
export DATABASE_URL="postgresql://..."
```

### 단계 4: Claude Code 재시작

설정 변경 후:

```bash
# 새 MCP를 로드하기 위해 Claude Code 재시작
```

## 보안 고려사항

1. **토큰을 커밋하지 않음**: 환경 변수 사용
2. **권한 제한**: 필요한 최소 범위 사용
3. **정기 감사**: 활성화된 MCP 검토
4. **토큰 갱신**: 주기적으로 토큰 변경

## 문제 해결

### MCP가 로드되지 않음

```bash
# npx 사용 가능 확인
which npx

# MCP 서버 수동 테스트
npx -y @modelcontextprotocol/server-github --help
```

### 컨텍스트 윈도우 문제

컨텍스트가 제한된 경우:
1. 사용하지 않는 MCP 비활성화
2. 활성 도구 수 줄이기
3. 프로젝트 특화 설정 사용

### 연결 오류

```bash
# 환경 변수 확인
echo $GITHUB_PERSONAL_ACCESS_TOKEN

# 네트워크 연결 확인
curl -I https://api.github.com
```

## 프로젝트 특화 설정

프로젝트 특화 MCP를 위해 프로젝트 루트에 `.claude.json` 생성:

```json
{
  "mcpServers": {
    "project-db": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "${PROJECT_DATABASE_URL}"
      }
    }
  }
}
```

## 참고 자료

- [MCP 문서](https://modelcontextprotocol.io/)
- [MCP 서버 목록](https://github.com/modelcontextprotocol/servers)
- [everything-claude-code](https://github.com/affaan-m/everything-claude-code)

# PyPI 배포 절차

## 사전 조건

- GitHub repository에 `pypi` environment 설정 완료 (Trusted Publishing / OIDC)
- 모든 변경사항 커밋 및 push 완료
- CI 통과 확인

## 1. 배포 전 로컬 검증

```bash
# 빌드
uv run python -m build

# wheel 내용 확인
unzip -l dist/qwen3_tts_triton-*.whl

# 깨끗한 환경에서 설치 테스트
uv venv /tmp/test-install && source /tmp/test-install/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install dist/qwen3_tts_triton-*.whl
python -c "from qwen3_tts_triton import __version__; print(__version__)"
tq3tts --help
deactivate && rm -rf /tmp/test-install
```

## 2. 버전 업데이트

두 곳의 버전이 일치해야 한다:

| 파일 | 위치 |
|------|------|
| `pyproject.toml` | `version = "X.Y.Z"` |
| `src/qwen3_tts_triton/__init__.py` | `__version__ = "X.Y.Z"` |

```bash
# 예: 0.1.0 → 0.2.0
sed -i 's/version = "0.1.0"/version = "0.2.0"/' pyproject.toml
sed -i 's/__version__ = "0.1.0"/__version__ = "0.2.0"/' src/qwen3_tts_triton/__init__.py
```

## 3. CHANGELOG 업데이트

`CHANGELOG.md`의 `[Unreleased]` 섹션을 버전 + 날짜로 변경:

```markdown
## [0.2.0] - 2025-XX-XX

### Added
- ...

### Changed
- ...
```

## 4. TestPyPI 테스트 배포 (권장)

RC 태그를 사용하면 `publish.yml`이 자동으로 TestPyPI에 배포한다:

```bash
git add -A
git commit -m "chore: Prepare v0.1.0 release"
git tag v0.1.0-rc1
git push origin claude_code  # 또는 main
git push origin v0.1.0-rc1
```

TestPyPI에서 설치 확인:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ qwen3-tts-triton
python -c "from qwen3_tts_triton import __version__; print(__version__)"
```

> `--extra-index-url`은 `triton`, `transformers` 등 의존성을 실제 PyPI에서 가져오기 위해 필요.

## 5. PyPI 정식 배포

RC 테스트 통과 후 정식 태그:

```bash
git tag v0.1.0
git push origin v0.1.0
```

`publish.yml` 자동 실행 내용:
1. 태그 버전 ↔ `pyproject.toml` 버전 일치 검증
2. `python -m build`로 wheel + sdist 빌드
3. wheel 설치 후 import 스모크 테스트
4. PyPI에 Trusted Publishing (OIDC)으로 배포

## 6. 배포 후 확인

```bash
# PyPI 페이지 확인
# https://pypi.org/project/qwen3-tts-triton/

# 깨끗한 환경에서 설치
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install qwen3-tts-triton
python -c "from qwen3_tts_triton import __version__; print(__version__)"
tq3tts --help
```

## publish.yml 동작 요약

| 태그 패턴 | 대상 | 예시 |
|-----------|------|------|
| `v*.*.*-rc*` | TestPyPI | `v0.1.0-rc1` |
| `v*.*.*` (rc 미포함) | PyPI | `v0.1.0` |

## 트러블슈팅

### "Tag vX.Y.Z does not match pyproject.toml version"
→ `pyproject.toml`의 `version`과 git 태그가 불일치. 둘을 맞춘 후 재태그.

### TestPyPI 배포 실패
→ GitHub repository Settings → Environments → `pypi`에서 TestPyPI Trusted Publisher도 설정 필요.

### pip install 시 CPU torch 설치됨
→ torch는 hard dependency가 아님. 사용자가 먼저 CUDA 버전을 설치해야 함. README 안내 참조.

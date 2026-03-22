# Docker Deployment Guide

Qwen3-TTS Triton 커널 최적화 프로젝트의 Docker 컨테이너 배포 가이드.

**Target**: RTX 5090 (Blackwell, sm_120, CUDA 12.8)

---

## 사전 요구사항

### 호스트 환경

| 요구사항 | 최소 버전 | 확인 명령 |
|----------|----------|----------|
| NVIDIA Driver | >= 570 | `nvidia-smi` |
| Docker Engine | >= 24.0 | `docker --version` |
| Docker Compose | v2.x+ | `docker compose version` |
| NVIDIA Container Toolkit | >= 1.14 | `nvidia-ctk --version` |
| GPU VRAM | >= 8GB (권장 16GB+) | `nvidia-smi` |

### NVIDIA Container Toolkit 설치

Docker 컨테이너에서 GPU를 사용하려면 NVIDIA Container Toolkit이 필요합니다.

```bash
# GPG 키 및 저장소 추가
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 설치
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Docker 런타임 설정
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 검증
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

---

## 빠른 시작

```bash
# 1. 이미지 빌드
docker compose -f docker/docker-compose.cu128.blackwell.yml build

# 2. 추론 실행
docker compose -f docker/docker-compose.cu128.blackwell.yml up inference

# 3. UI 대시보드 (http://localhost:8501)
docker compose -f docker/docker-compose.cu128.blackwell.yml up ui
```

---

## 파일 구조

```
qwen3-tts-triton/
├── docker/                             # Docker 배포 파일
│   ├── Dockerfile.cu128.blackwell      # Multi-stage Dockerfile
│   └── docker-compose.cu128.blackwell.yml  # 서비스 정의
└── .dockerignore                       # 빌드 컨텍스트 제외 목록
```

---

## Dockerfile 아키텍처

### Multi-Stage Build

```
┌──────────────────────────────────────────────────┐
│  Stage 1: builder                                │
│  nvidia/cuda:12.8.0-devel-ubuntu24.04            │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │ UV 패키지 매니저로 .venv 생성              │  │
│  │ pyproject.toml + uv.lock → 의존성 설치     │  │
│  │ torch/torchaudio: pytorch-cu128 인덱스     │  │
│  └────────────────────────────────────────────┘  │
└──────────────┬───────────────────────────────────┘
               │ COPY .venv + 소스코드
               ▼
┌──────────────────────────────────────────────────┐
│  Stage 2: runtime                                │
│  nvidia/cuda:12.8.0-devel-ubuntu24.04            │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │ .venv (빌더에서 복사)                      │  │
│  │ kernels/ models/ ui/ benchmark/ infer.py   │  │
│  │ UV 바이너리 없음, dev 헤더 없음            │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

### devel 이미지를 사용하는 이유

Triton은 **런타임에 GPU 커널을 JIT 컴파일**합니다. 이 과정에서 다음이 필요합니다:

- **ptxas**: PTX → SASS 어셈블러 (devel에만 포함)
- **libdevice.10.bc**: GPU 수학 함수 라이브러리 (devel에만 포함)

특히 Blackwell (sm_120)은 CUDA 12.8에서 새로 추가된 아키텍처이므로, 시스템 `ptxas`가 확실히 지원하는 `devel` 이미지가 안전합니다.

> Triton 3.3+는 자체 ptxas를 번들하지만, Blackwell 호환성을 위해 devel 이미지 사용을 권장합니다.

### 레이어 캐싱 전략

```dockerfile
# 1단계: 의존성만 설치 (pyproject.toml/uv.lock 변경시만 재빌드)
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev --extra ui

# 2단계: 소스 복사 및 프로젝트 설치 (코드 변경시만 재빌드)
COPY . .
RUN uv sync --locked --no-dev --no-editable --extra ui
```

소스 코드만 수정하면 1단계는 캐시에서 로드되어 빌드가 매우 빠릅니다.

---

## Docker Compose 서비스

### 서비스 구성

| 서비스 | GPU | 용도 | 포트 | 실행 방식 |
|--------|-----|------|------|----------|
| `inference` | 1 GPU | CLI 추론 | - | 기본 |
| `ui` | 1 GPU | Streamlit 대시보드 | 8501 | 기본 |
| `benchmark` | 1 GPU | E2E 벤치마크 | - | on-demand (profile) |

### 볼륨

| 볼륨 | 컨테이너 경로 | 용도 | 크기 |
|------|-------------|------|------|
| `hf_cache` | `/cache/huggingface` | HuggingFace 모델 가중치 | ~7GB |
| `triton_cache` | `/cache/triton` | Triton JIT 컴파일 캐시 | ~50MB |
| `outputs` | `/app/outputs` | 생성된 WAV 오디오 | 가변 |

> `hf_cache`를 named volume으로 사용하면 모델을 한 번만 다운로드합니다.
> `triton_cache`가 없으면 컨테이너 재시작마다 커널 재컴파일이 발생합니다 (~30초).

### 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `HF_TOKEN` | (없음) | HuggingFace 인증 토큰 (gated 모델용) |
| `UI_PORT` | `8501` | Streamlit UI 호스트 포트 |
| `TORCH_CUDA_ARCH_LIST` | `12.0` | CUDA 타겟 아키텍처 (Blackwell) |
| `TRITON_CACHE_DIR` | `/cache/triton` | Triton 커널 캐시 경로 |
| `HF_HOME` | `/cache/huggingface` | HuggingFace 캐시 경로 |

---

## 사용법

### 이미지 빌드

```bash
# 기본 빌드
docker compose -f docker/docker-compose.cu128.blackwell.yml build

# 캐시 없이 재빌드
docker compose -f docker/docker-compose.cu128.blackwell.yml build --no-cache

# 단독 docker build
docker build -f docker/Dockerfile.cu128.blackwell -t qwen3-tts-triton:cu128 .
```

### 추론

```bash
# 기본 모드 (Triton 최적화)
docker compose -f docker/docker-compose.cu128.blackwell.yml up inference

# 비교 모드 (Base vs Triton)
docker compose -f docker/docker-compose.cu128.blackwell.yml run --rm inference \
  python infer.py --mode compare-all

# 특정 텍스트로 추론
docker compose -f docker/docker-compose.cu128.blackwell.yml run --rm inference \
  python infer.py --mode triton --text "안녕하세요, 반갑습니다."

# 출력 파일 확인
docker compose -f docker/docker-compose.cu128.blackwell.yml run --rm inference \
  ls -la /app/outputs/
```

### UI 대시보드

```bash
# 시작 (http://localhost:8501)
docker compose -f docker/docker-compose.cu128.blackwell.yml up ui

# 커스텀 포트
UI_PORT=9000 docker compose -f docker/docker-compose.cu128.blackwell.yml up ui

# 백그라운드 실행
docker compose -f docker/docker-compose.cu128.blackwell.yml up -d ui
```

### 벤치마크

```bash
# E2E 벤치마크
docker compose -f docker/docker-compose.cu128.blackwell.yml --profile bench run --rm benchmark

# 결과는 ./benchmark/results/ 에 bind mount
ls benchmark/results/
```

### HuggingFace 인증

`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` 모델이 gated인 경우:

```bash
# 방법 1: 환경 변수
HF_TOKEN=hf_xxxxxxxxx docker compose -f docker/docker-compose.cu128.blackwell.yml up inference

# 방법 2: .env 파일
echo "HF_TOKEN=hf_xxxxxxxxx" > .env
docker compose -f docker/docker-compose.cu128.blackwell.yml up inference
```

---

## 모델 캐시 관리

### 사전 다운로드 (권장)

컨테이너 첫 실행 전에 모델을 미리 다운로드하면 시작 시간을 단축할 수 있습니다.

```bash
# 호스트에서 직접 다운로드
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

# Docker volume에 복사
docker run --rm \
  -v hf_cache:/cache \
  -e HF_HOME=/cache \
  python:3.12-slim \
  sh -c "pip install -q huggingface_hub && \
    python -c \"from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', cache_dir='/cache')\""
```

### 캐시 확인

```bash
# volume 크기 확인
docker volume inspect qwen3-tts-triton_hf_cache

# 캐시된 모델 목록
docker run --rm -v qwen3-tts-triton_hf_cache:/cache alpine ls -la /cache/hub/models--*/
```

### 캐시 초기화

```bash
# 모든 volume 제거
docker compose -f docker/docker-compose.cu128.blackwell.yml down -v

# 특정 volume만 제거
docker volume rm qwen3-tts-triton_hf_cache
```

---

## Triton 커널 캐시

Triton은 `@triton.jit` 커널을 첫 호출 시 JIT 컴파일하고, 결과를 캐시합니다.

### 캐시 동작

```
첫 번째 실행: Python → Triton IR → LLVM IR → PTX → SASS (~30초)
이후 실행:   캐시에서 SASS 바이너리 로드 (~0.1초)
```

### 환경 변수

| 변수 | 설명 |
|------|------|
| `TRITON_CACHE_DIR` | 캐시 저장 경로 (기본: `~/.triton/cache`) |
| `TRITON_HOME` | Triton 홈 디렉토리 (기본: `~/.triton`) |
| `TRITON_PTXAS_PATH` | ptxas 바이너리 경로 override |
| `TRITON_DUMP_DIR` | 디버깅용 IR/PTX 덤프 경로 |

### 캐시 무효화 조건

- Triton 버전 변경
- 커널 소스 코드 수정
- GPU 아키텍처 변경 (다른 GPU로 전환 시)

---

## 커스터마이징

### 의존성 그룹 변경

Dockerfile의 `uv sync` 명령에서 `--extra` 플래그를 수정합니다:

```dockerfile
# 기본 (core + ui)
RUN uv sync --locked --no-install-project --no-dev --extra ui

# faster-qwen3-tts 포함
RUN uv sync --locked --no-install-project --no-dev --extra ui --extra faster

# 평가 도구 포함
RUN uv sync --locked --no-install-project --no-dev --extra ui --extra eval

# 전체 (ui + faster + eval + dev)
RUN uv sync --locked --no-install-project --all-extras
```

### GPU 수 변경

`docker/docker-compose.cu128.blackwell.yml`에서:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all        # 모든 GPU 사용
          capabilities: [gpu]
```

또는 특정 GPU 지정:

```yaml
        - driver: nvidia
          device_ids: ["0"]  # GPU 0번만
          capabilities: [gpu]
```

### 호스트 디렉토리 바인드 마운트

개발 중 소스 코드를 실시간 반영하려면:

```yaml
volumes:
  - ./kernels:/app/kernels:ro
  - ./models:/app/models:ro
  - ./ui:/app/ui:ro
```

---

## 트러블슈팅

### GPU를 인식하지 못하는 경우

```bash
# 호스트에서 GPU 확인
nvidia-smi

# Docker에서 GPU 확인
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi

# NVIDIA Container Toolkit 재설정
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### `ptxas fatal: Value 'sm_120' is not defined`

Triton 또는 PyTorch 버전이 Blackwell을 지원하지 않습니다.

```bash
# 컨테이너 내 버전 확인
docker run --rm qwen3-tts-triton:cu128 python -c "
import torch, triton
print(f'PyTorch: {torch.__version__}')
print(f'Triton:  {triton.__version__}')
print(f'CUDA:    {torch.version.cuda}')
"
# PyTorch >= 2.7.0+cu128, Triton >= 3.3 필요
```

### 모델 다운로드 실패 (401/403)

```bash
# HF 토큰 설정 확인
HF_TOKEN=hf_xxx docker compose -f docker/docker-compose.cu128.blackwell.yml run --rm inference \
  python -c "from huggingface_hub import whoami; print(whoami())"
```

### Shared Memory 부족 (`Bus error`)

`shm_size`를 늘리거나 `ipc: host`를 사용합니다:

```yaml
services:
  inference:
    shm_size: "32g"
    # 또는
    ipc: host    # 호스트 IPC 네임스페이스 공유 (shm_size 무시됨)
```

### 이미지 크기 최적화

현재 devel 이미지는 크기가 큽니다 (~15GB+). 이미지 크기를 줄이려면:

1. `runtime` 이미지 시도 (Triton 번들 ptxas에 의존):
   ```dockerfile
   FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS runtime
   ```
2. Triton 캐시를 이미지에 bake-in하여 runtime에서 JIT 불필요하게 만들기
3. `.dockerignore` 확인하여 불필요한 파일 제외

---

## 기술 스택 참조

| 컴포넌트 | 버전 | 역할 |
|----------|------|------|
| CUDA | 12.8.0 | Blackwell sm_120 지원 (최소) |
| Ubuntu | 24.04 | Python 3.12 기본 내장 |
| Python | 3.12 | 프로젝트 요구사항 |
| PyTorch | >= 2.7.0+cu128 | Triton 3.3 번들 |
| Triton | >= 3.3 | Blackwell 아키텍처 JIT 지원 |
| UV | 0.6.x | 의존성 관리 (pyproject.toml + uv.lock) |
| NVIDIA Driver | >= 570 | CUDA 12.8 호스트 드라이버 |

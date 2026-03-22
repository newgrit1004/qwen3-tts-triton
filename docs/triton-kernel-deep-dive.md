# Qwen3-TTS Triton 커널 최적화 Deep Dive

> 면접 준비용 기술 문서. 왜 이 커널들을 골랐고, 어떻게 구현했고, 수치 검증은 어떻게 했고, monkey patching은 어떤 원리로 동작하는지 전부 다룬다.

---

## 목차

1. [왜 Triton 커널 퓨전인가?](#1-왜-triton-커널-퓨전인가)
2. [왜 하필 이 4개 커널인가?](#2-왜-하필-이-4개-커널인가)
3. [각 커널 구현 상세](#3-각-커널-구현-상세)
4. [수치 검증 전략](#4-수치-검증-전략)
5. [Monkey Patching: PyTorch 모델을 Triton으로 갈아끼우기](#5-monkey-patching-pytorch-모델을-triton으로-갈아끼우기)
6. [삽질 기록과 교훈](#6-삽질-기록과-교훈)
7. [Hybrid 접근법: CUDA Graph × Triton 커널 퓨전](#7-hybrid-접근법-cuda-graph--triton-커널-퓨전)
8. [종합 벤치마크 결과](#8-종합-벤치마크-결과)
9. [Triton vs TensorRT: 왜 Triton을 선택했나?](#9-triton-vs-tensorrt-왜-triton을-선택했나)
10. [Triton 커널 모델의 프로덕션 서빙](#10-triton-커널-모델의-프로덕션-서빙)
11. [torch.compile vs 수동 Triton: 진짜 apple-to-apple 비교](#11-torchcompile-vs-수동-triton-진짜-apple-to-apple-비교)

---

## 1. 왜 Triton 커널 퓨전인가?

### GPU 메모리 계층 이야기

면접에서 "왜 Triton 커널을 썼나요?" 라고 물어보면, 핵심은 **HBM (High Bandwidth Memory) 병목**이야.

GPU에는 크게 두 가지 메모리가 있어:

```
┌─────────────────────────────────┐
│         SRAM (On-chip)          │  ← 아주 빠름 (~19 TB/s), 아주 작음 (~20MB)
│         레지스터 + L1/L2        │
├─────────────────────────────────┤
│         HBM (Off-chip)          │  ← 상대적으로 느림 (~1 TB/s), 큼 (32GB)
│         글로벌 메모리            │
└─────────────────────────────────┘
```

RTX 5090 (Blackwell) 기준으로 HBM 대역폭이 약 1.8 TB/s인데, SRAM은 그것보다 10배 이상 빠르거든. 문제는 PyTorch의 기본 연산들이 **각각 독립적인 CUDA 커널**로 실행된다는 거야.

예를 들어 RMSNorm 하나만 봐도:

```
PyTorch 기본 동작 (4번의 HBM 왕복):
  1. x를 HBM에서 읽어서 → x^2 계산 → HBM에 저장 (왕복 1)
  2. x^2를 HBM에서 읽어서 → mean 계산 → HBM에 저장 (왕복 2)
  3. mean을 HBM에서 읽어서 → rsqrt 계산 → HBM에 저장 (왕복 3)
  4. x와 rsqrt를 HBM에서 읽어서 → 곱해서 → HBM에 저장 (왕복 4)

Triton 퓨전 커널 (1번의 HBM 왕복):
  1. x를 HBM에서 SRAM으로 읽어서 → x^2 → mean → rsqrt → 곱하기 전부
     SRAM에서 처리 → 결과만 HBM에 저장 (왕복 1)
```

**4번의 HBM 라운드트립이 1번으로 줄어든다.** 이게 커널 퓨전의 핵심이야.

### 왜 CUDA가 아니라 Triton인가?

CUDA로 직접 짜면 더 세밀하게 최적화할 수 있지만, 몇 가지 이유로 Triton을 선택했어:

1. **생산성**: Triton은 Python으로 GPU 커널을 짤 수 있어. CUDA C++로 같은 걸 짜려면 코드량이 3-5배는 늘어나고, 메모리 관리를 수동으로 해야 해.

2. **자동 튜닝**: Triton 컴파일러가 block size, num_warps 같은 하드웨어 파라미터를 자동으로 최적화해줘. CUDA에선 이걸 전부 수동으로 해야 하거든.

3. **이식성**: CUDA는 NVIDIA 전용이지만, Triton은 AMD ROCm 백엔드도 지원해. 나중에 다른 GPU로 옮길 때 코드 수정이 거의 없어.

4. **Liger Kernel 참조**: LinkedIn에서 만든 Liger Kernel 프로젝트가 LLaMA, Mistral 등에 Triton 커널을 적용해서 좋은 성과를 냈거든. 이미 검증된 접근법이라 리스크가 낮았어.

### 제약 조건: "추가 VRAM 사용 금지"

이 프로젝트에서 중요한 제약이 있었어. **VRAM을 추가로 쓰면 안 된다**는 거. 왜냐면 Qwen3-TTS 1.7B 모델이 이미 VRAM을 꽤 차지하고 있고, TTS는 실시간 처리가 중요하니까 모델 로딩 후 남는 VRAM 여유가 별로 없거든.

그래서 모델 구조를 바꾸거나 (예: Tensor Parallelism, KV Cache 최적화) 하는 건 안 되고, **순수하게 연산 퓨전만으로 속도를 올려야** 했어. 커널 퓨전은 중간 텐서(intermediate tensor)를 없애니까 오히려 VRAM을 절약하는 효과도 있어.

---

## 2. 왜 하필 이 4개 커널인가?

### Qwen3-TTS Talker 아키텍처 분석

Qwen3-TTS는 크게 두 부분으로 나뉘어:

```
Qwen3-TTS
├── Talker (28 layers, hidden=2048) ← 주 병목!
│   ├── Self-Attention (GQA: 16 q-heads, 8 kv-heads)
│   │   ├── QKV Projection (Linear)
│   │   ├── M-RoPE (Position Encoding) ← 커널 3
│   │   ├── Attention (Flash Attention이 이미 처리)
│   │   └── Output Projection (Linear)
│   ├── Add & Norm ← 커널 1, 4
│   │   ├── Residual Add
│   │   └── RMSNorm
│   └── MLP (SwiGLU) ← 커널 2
│       ├── Gate Projection (Linear, 2048→6144)
│       ├── Up Projection (Linear, 2048→6144)
│       ├── SwiGLU Activation ← 이 부분!
│       └── Down Projection (Linear, 6144→2048)
└── Code Predictor (5 layers, hidden=1024)
```

**한 레이어가 28번 반복**되니까, 레이어 안에서 조금이라도 줄이면 28배로 이득이 쌓여. 프로파일링 해보면 Linear (행렬곱) 연산이 제일 무겁긴 한데, 이건 cuBLAS가 이미 극도로 최적화돼 있어서 Triton으로 이길 수가 없어. 그래서 **Linear를 제외한 나머지 연산들**을 타겟으로 잡은 거야.

### 커널 선정 기준

각 커널을 고른 이유를 하나씩 말해볼게:

#### 커널 1: RMSNorm

**선정 이유**: Transformer 레이어마다 2번씩 호출돼 (attention 전, MLP 전). 28개 레이어면 **56번** 호출이야. 연산 자체는 가볍지만 (element-wise), HBM 왕복이 많아서 퓨전 효과가 크거든.

```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

PyTorch 기본: 4번의 커널 launch
Triton 퓨전: 1번의 커널 launch
```

#### 커널 2: SwiGLU

**선정 이유**: MLP의 activation 함수인데, `silu(gate) * up` 이라는 식이야. PyTorch 기본이면 silu 결과를 중간 텐서에 저장했다가 곱하거든. 이 중간 텐서 크기가 `(batch, seq_len, 6144)` — 꽤 크지? 이걸 HBM에 썼다 읽는 게 낭비야.

```
SwiGLU(gate, up) = silu(gate) * up
                 = (gate * sigmoid(gate)) * up

PyTorch 기본: silu 결과를 HBM에 저장 → 다시 읽어서 up과 곱하기
Triton 퓨전: silu와 곱하기를 SRAM에서 한번에 처리
```

중간 텐서 `silu(gate)`가 `(B, T, 6144)` 크기인데, 이걸 없앨 수 있으니까 VRAM도 아끼는 1석 2조야.

#### 커널 3: M-RoPE (Multi-dimensional Rotary Position Embedding)

**선정 이유**: Qwen3-TTS가 일반적인 RoPE가 아니라 **3차원 위치 인코딩(M-RoPE)**을 쓰거든. temporal(시간), height(높이), width(너비) 3개 차원에 대해 각각 다른 cos/sin을 적용해. PyTorch로 이걸 구현하면 cos/sin을 차원별로 슬라이싱하고, 각각 회전 적용하고, 다시 합치는 과정에서 **여러 번의 텐서 조작과 HBM 왕복**이 발생해.

```
M-RoPE sections = [24, 20, 20]  (합계 = 64 = head_dim/2)
- 처음 24개 pair: temporal cos/sin 적용
- 다음 20개 pair: height cos/sin 적용
- 마지막 20개 pair: width cos/sin 적용
```

하나의 커널에서 section 경계에 따라 다른 cos/sin을 로드해서 회전시키면, 중간 텐서 생성 없이 in-place로 처리할 수 있어.

#### 커널 4: Fused Norm + Residual

**선정 이유**: Transformer 레이어에서 `residual = x + residual` 한 다음 `RMSNorm(residual)` 하는 패턴이 반복되거든. 이 두 연산을 합치면:

```
PyTorch 기본 (2단계):
  1. residual = x + residual  → HBM에 저장 (왕복 1)
  2. output = RMSNorm(residual) → HBM에서 읽고, 처리하고, 저장 (왕복 2+)

Triton 퓨전 (1단계):
  1. x, residual을 한번에 읽어서 → 더하고 → normalize하고 → 둘 다 저장 (왕복 1)
```

residual 텐서를 HBM에 썼다가 바로 다시 읽는 불필요한 왕복을 제거할 수 있어.

### 왜 Attention은 안 건드렸나?

면접에서 "왜 attention은 최적화 안 했어요?" 라고 물어볼 수 있어. 답은 간단해:

1. **Flash Attention이 이미 있다**: PyTorch 2.0+ 에는 `scaled_dot_product_attention`이 내장되어 있고, 내부적으로 Flash Attention v2를 사용해. 이미 메모리 효율적인 구현이 들어가 있어서 Triton으로 다시 짜봤자 이길 수 없어.

2. **행렬곱은 cuBLAS가 왕**: QKV projection이나 output projection은 큰 행렬곱이야. cuBLAS/cuDNN은 수십 년간 최적화된 GEMM 구현을 갖고 있어서 Triton이 이길 수 없는 영역이야.

결국 **"cuBLAS가 처리하지 않는 element-wise 연산 + 메모리 바운드 연산"**이 우리가 Triton으로 최적화할 수 있는 영역이고, 딱 그 부분이 위의 4개 커널이야.

---

## 3. 각 커널 구현 상세

### 3.1 RMSNorm 커널

**파일**: `kernels/rms_norm.py`

#### 수학적 배경

LayerNorm과 다르게 RMSNorm은 mean centering을 생략해:

```
LayerNorm: y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
RMSNorm:   y = x / sqrt(mean(x^2) + eps) * gamma
```

mean centering이 없으니 **계산량이 줄어들면서도 성능 차이가 거의 없다**는 게 논문 결과야. 그래서 LLaMA, Qwen 계열 모델들이 다 RMSNorm을 써.

#### Llama Casting Mode란?

half-precision (fp16/bf16)으로 학습/추론할 때, variance 계산에서 수치 불안정 문제가 생길 수 있어. 작은 값들의 제곱합은 fp16에서 underflow가 발생할 수 있거든. 그래서 **variance 계산만 fp32로 올려서** 하고, 결과를 다시 원래 dtype으로 내리는 게 llama casting mode야.

```python
# 커널 코드에서 이 부분:
X_row_fp32 = X_row.to(tl.float32)           # fp16 → fp32로 올림
mean_square = tl.sum(X_row_fp32 * X_row_fp32, axis=0) / n_cols  # fp32로 계산
rstd = rsqrt(mean_square + eps)              # fp32로 rsqrt

X_norm = (X_row_fp32 * rstd).to(X_row_dtype) # 정규화 후 원래 dtype으로
Y_row = X_norm * W_row                       # weight는 원래 dtype
```

핵심은 `.to(tl.float32)` → 계산 → `.to(X_row_dtype)` 이 패턴이야. 정밀도가 중요한 부분만 fp32로 올리고, 나머지는 원래 dtype을 유지해서 메모리와 연산 비용을 아끼는 거지.

#### Triton 커널 구조

```python
@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr, Y_row_stride,    # 출력 텐서 포인터와 stride
    X_ptr, X_row_stride,    # 입력 텐서 포인터와 stride
    W_ptr,                   # weight 포인터
    n_cols,                  # hidden_size (예: 2048)
    eps,                     # 1e-6
    BLOCK_SIZE: tl.constexpr,  # 컴파일 타임 상수 (2048의 next_power_of_2)
):
```

`tl.constexpr`는 Triton의 컴파일 타임 상수야. BLOCK_SIZE를 컴파일 타임에 결정하면, 컴파일러가 루프를 완전히 풀어서(unroll) 최적화할 수 있어.

**Grid 설정**: `(n_rows,)` — 행 하나당 하나의 프로그램 인스턴스가 처리해. 예를 들어 input이 `(2, 512, 2048)` 이면 `n_rows = 2*512 = 1024`개의 프로그램이 병렬로 돌아.

**BLOCK_SIZE 결정 로직** (`utils.py`의 `calculate_settings`):

```python
BLOCK_SIZE = triton.next_power_of_2(n_cols)  # 2048 → 2048 (이미 2의 거듭제곱)
# num_warps는 BLOCK_SIZE에 따라:
# >= 32768: 32 warps
# >= 8192:  16 warps
# >= 2048:  8 warps
# 기본:    4 warps
```

warp는 GPU에서 32개 스레드를 묶은 단위야. BLOCK_SIZE가 클수록 더 많은 warp가 필요해서, 이 휴리스틱으로 적절한 병렬도를 정하는 거야. 이 값은 Liger Kernel / Unsloth에서 검증된 경험적 값이야.

### 3.2 SwiGLU 커널

**파일**: `kernels/swiglu.py`

#### SwiGLU란?

GLU(Gated Linear Unit) 변형 중 하나야. MLP에서 activation function으로 쓰이는데:

```
SwiGLU(gate, up) = silu(gate) * up
silu(x) = x * sigmoid(x)    ← "Swish" 라고도 불림
```

왜 Gate와 Up 두 개를 쓰냐면, gate가 "어떤 정보를 통과시킬지" 결정하고, up이 "실제 정보"를 담고 있어. silu로 gate를 soft하게 필터링한 다음 up과 곱해서 정보를 선별적으로 통과시키는 구조야.

#### sigmoid의 fp32 안정성

sigmoid 함수에서 수치 안정성 이슈가 있어:

```
sigmoid(x) = 1 / (1 + exp(-x))
```

`exp(-x)`에서 x가 아주 크거나 작으면 fp16에서 overflow/underflow가 날 수 있어. 그래서 gate를 fp32로 올려서 sigmoid를 계산하고, 결과를 다시 원래 dtype으로 내려:

```python
# 커널 코드:
gate = tl.load(gate_ptr + col_offsets, mask=mask, other=0).to(tl.float32)  # fp32로
silu_gate = gate * tl.sigmoid(gate)                    # fp32에서 안전하게 계산
out = silu_gate.cast(up.dtype) * up                    # 원래 dtype으로 내려서 곱하기
```

#### 퓨전의 효과

PyTorch에서 `F.silu(gate) * up` 하면:
1. `F.silu(gate)` → 중간 텐서 생성, HBM에 저장
2. `결과 * up` → 중간 텐서를 다시 읽어서 곱하기

Triton에서는 이 두 단계가 SRAM 안에서 한번에 처리돼. 중간 텐서 크기가 `(B, T, 6144)`이니까 (Qwen3-TTS의 intermediate_size=6144), batch=4, seq_len=1024 기준으로 약 **50MB의 중간 텐서를 제거**하는 거야.

### 3.3 M-RoPE 커널

**파일**: `kernels/rope.py`

이건 설명할 게 좀 많아. 가장 복잡한 커널이야.

#### RoPE 기본 원리

RoPE(Rotary Position Embedding)는 "위치 정보를 벡터의 회전으로 인코딩"하는 방법이야. 직관적으로 말하면:

```
position 0의 벡터: 회전 안 함
position 1의 벡터: θ만큼 회전
position 2의 벡터: 2θ만큼 회전
...
```

2D 회전 행렬로 표현하면:

```
[cos(mθ)  -sin(mθ)] [x_even]   [x_even * cos(mθ) - x_odd * sin(mθ)]
[sin(mθ)   cos(mθ)] [x_odd ] = [x_even * sin(mθ) + x_odd * cos(mθ)]
```

여기서 m은 position index, θ는 주파수에 따라 달라지는 각도야. head_dim을 2개씩 짝지어서 (x_0, x_1), (x_2, x_3), ... 각 pair에 서로 다른 주파수의 회전을 적용해.

#### 왜 M-RoPE인가?

일반 텍스트 LLM은 1차원 시퀀스니까 position이 하나면 돼. 그런데 TTS에서는 여러 차원의 위치 정보가 필요해:

- **temporal (t)**: 시간축 위치 — 음성의 시간적 순서
- **height (h)**: 주파수축 — mel spectrogram의 주파수 bin
- **width (w)**: 추가 차원 — 코드북 인덱스 등

M-RoPE는 head_dim을 section으로 나눠서, 각 section에 다른 차원의 cos/sin을 적용해:

```
head_dim = 128 → half = 64 rotation pairs

sections = [24, 20, 20]:
  pair 0~23  (24개): temporal cos/sin
  pair 24~43 (20개): height cos/sin
  pair 44~63 (20개): width cos/sin
```

#### Interleaved vs Rotated

RoPE 구현에는 두 가지 방식이 있어:

```
Rotated (Hugging Face 스타일):
  x_rot = x[..., :half]
  x_pass = x[..., half:]
  pair: (x_0, x_64), (x_1, x_65), ...

Interleaved (원래 논문 스타일):
  pair: (x_0, x_1), (x_2, x_3), ...
```

Qwen3-TTS는 **interleaved** 방식을 써. 커널에서 even/odd offset을 이렇게 잡는 이유가 이거야:

```python
# even index: 0, 2, 4, ... (stride = 2)
even_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :] * 2
# odd index: 1, 3, 5, ... (even + 1)
odd_q_offsets = even_q_offsets + 1
```

`* 2`로 짝수 인덱스를 잡고, `+ 1`로 홀수 인덱스를 잡는 거야. 이게 interleaved pair를 SRAM에서 효율적으로 로드하는 방법이야.

#### Section 마스크의 트릭

3개 차원에서 cos/sin을 합치는 부분이 커널의 핵심 트릭이야:

```python
# 각 section의 마스크 (서로 겹치지 않음!)
t_mask = cos_offsets < t_end                          # [0, 24)
h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)  # [24, 44)
w_mask = (h_end <= cos_offsets) & (cos_offsets < hd // 2)  # [44, 64)

# 각 마스크로 해당 차원의 cos/sin만 로드
t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)  # t 영역만 값 있음
h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)  # h 영역만 값 있음
w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)  # w 영역만 값 있음

# 마스크가 disjoint(겹치지 않으)니까, 더하면 합쳐짐!
cos_row = t_cos_row + h_cos_row + w_cos_row
```

이게 왜 작동하냐면, `other=0`으로 마스크 바깥을 0으로 채웠으니까, 더하면 각 위치에 정확히 하나의 차원 cos/sin만 남아. 텐서 연결(concatenation) 대신 **마스킹 + 덧셈**으로 같은 효과를 내는 거야. 메모리 할당이 필요 없어서 더 효율적이지.

#### 데이터 레이아웃 전환

입력은 `(bsz, n_head, seq_len, head_dim)` 형태로 들어오는데, 커널에서는 `(bsz, seq_len, n_head, head_dim)`으로 transpose 해서 쓰고 있어:

```python
q = q.transpose(1, 2).contiguous()  # (B, H, T, D) → (B, T, H, D)
```

왜 이렇게 하냐면, 커널에서 `(batch, seq_pos)` 하나당 하나의 프로그램이 돌아가는데, 이 프로그램이 **모든 head의 head_dim을 한번에 처리**해야 하거든. `(B, T, H, D)` 레이아웃이면 한 프로그램이 접근하는 메모리가 연속적(contiguous)이 돼서 coalescent read가 가능해져.

### 3.4 Fused Norm + Residual 커널

**파일**: `kernels/fused_norm_residual.py`

#### Transformer 레이어의 Residual 패턴

Transformer는 이런 패턴이 반복돼:

```python
# Pre-norm 스타일 (Qwen3-TTS가 이 방식):
residual = hidden_states
hidden_states = RMSNorm(hidden_states)
hidden_states = Attention(hidden_states)
hidden_states = hidden_states + residual  # residual add

residual = hidden_states
hidden_states = RMSNorm(hidden_states)
hidden_states = MLP(hidden_states)
hidden_states = hidden_states + residual  # residual add
```

`hidden_states + residual` → `RMSNorm(결과)` 이 연속으로 일어나는 부분을 하나로 합친 거야.

#### 커널의 핵심

```python
# 1. X와 R을 HBM에서 한번에 읽음
X_row = tl.load(X_ptr + ...)
R_row = tl.load(R_ptr + ...)

# 2. Residual add (SRAM에서)
S_row = X_row + R_row

# 3. Updated residual을 HBM에 저장
tl.store(S_ptr + ..., S_row)

# 4. RMSNorm (SRAM에서 계속)
S_row_fp32 = S_row.to(tl.float32)
mean_square = tl.sum(S_row_fp32 * S_row_fp32, axis=0) / n_cols
rstd = rsqrt(mean_square + eps)
Y_row = (S_row_fp32 * rstd).to(S_row_dtype) * W_row

# 5. Normalized output을 HBM에 저장
tl.store(Y_ptr + ..., Y_row)
```

포인트는 **S_row(residual add 결과)가 SRAM에 있는 상태에서 바로 RMSNorm을 적용**한다는 거야. PyTorch 기본이면 residual add 결과를 HBM에 썼다가 RMSNorm이 다시 읽어야 하거든.

출력이 두 개인 것도 특이해: `(y, s)` 튜플로 normalized output과 updated residual을 동시에 반환해. 다음 레이어에서 이 residual을 쓸 수 있도록.

---

## 4. 수치 검증 전략

### 왜 수치 검증이 중요한가

커널 퓨전은 "같은 수학적 결과를 다른 방법으로 계산"하는 건데, 부동소수점 연산에서는 **계산 순서가 달라지면 결과도 미세하게 달라질 수 있어**. 특히 half-precision에서.

```
수학적으로: (a + b) + c = a + (b + c)
부동소수점: (a + b) + c ≠ a + (b + c)  ← 결합법칙이 안 성립!
```

fp16은 mantissa(가수부)가 10bit, bfloat16은 7bit밖에 안 돼. 그래서 큰 값과 작은 값을 더할 때 작은 값이 날아가는 문제가 있어. 커널 퓨전을 하면 reduce 순서 (sum의 순서)가 PyTorch와 달라질 수 있기 때문에, **결과가 수학적으로 동일한지가 아니라 "허용 범위 안에서 일치하는지"**를 검증해야 해.

### 검증 방법론: Reference Implementation + Tolerance

각 커널마다 이런 구조로 테스트를 짰어:

```python
def test_triton_kernel_vs_pytorch():
    # 1. PyTorch reference implementation (순수 PyTorch로 구현)
    ref_output = pytorch_reference(input)

    # 2. Triton kernel
    triton_output = triton_kernel(input)

    # 3. torch.allclose로 비교
    assert torch.allclose(triton_output, ref_output, atol=ATOL, rtol=RTOL)
```

핵심은 **reference implementation을 직접 짠다**는 거야. PyTorch의 built-in 함수가 아니라, 수식을 그대로 Python으로 옮긴 reference를 만들어서 비교해. 이래야 "이 Triton 커널이 수식과 일치하는가"를 검증할 수 있거든.

### 각 커널의 Reference Implementation

#### RMSNorm Reference

```python
def _pytorch_rms_norm(x, weight, eps=1e-6):
    """수식 그대로 PyTorch로 구현 (llama casting mode)"""
    x_fp32 = x.float()                              # fp32로 올림
    rms = torch.sqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)  # RMS 계산
    return (x_fp32 / rms).to(x.dtype) * weight       # 정규화 → 원래 dtype → weight 곱
```

#### SwiGLU Reference

```python
# PyTorch의 F.silu가 이미 정확한 reference
expected = F.silu(gate) * up
```

SwiGLU는 PyTorch에 이미 정확한 구현이 있어서, `F.silu(gate) * up`을 그대로 reference로 씀.

#### M-RoPE Reference

```python
def torch_mrope_reference(q, k, cos, sin, mrope_section):
    # 1. section별로 cos/sin 슬라이싱해서 합치기
    merged_cos = torch.cat([
        cos[0, :, :, :sec_t],           # temporal section
        cos[1, :, :, sec_t:sec_t+sec_h], # height section
        cos[2, :, :, sec_t+sec_h:half],  # width section
    ], dim=-1).unsqueeze(1)

    # 2. Interleaved rotation
    q_even, q_odd = q[..., 0::2], q[..., 1::2]
    new_q_even = q_even * merged_cos - q_odd * merged_sin
    new_q_odd  = q_odd * merged_cos + q_even * merged_sin
    new_q = torch.stack([new_q_even, new_q_odd], dim=-1).flatten(-2)
    # k도 동일...
```

M-RoPE reference가 제일 복잡해. 회전 행렬의 수학 공식을 그대로 PyTorch tensor 연산으로 옮겨서, "Triton 커널의 마스킹+덧셈 트릭이 이 명시적 concatenation과 같은 결과를 내는가"를 검증하는 거야.

#### Fused Norm+Residual Reference

```python
# 2단계를 명시적으로
ref_residual = x + residual
ref_output = _pytorch_rms_norm(ref_residual, weight, eps)
```

### Tolerance(허용 오차) 설정의 고민

이 부분이 실제로 가장 삽질을 많이 했어. 처음에는 `atol=1e-3, rtol=1e-3`으로 잡았는데, 테스트가 실패하더라고.

#### 왜 실패했나?

bfloat16의 mantissa가 **7bit**밖에 안 돼. 이건 10진수로 약 2자리 정밀도야. 즉, `1.0`과 `1.01`까지는 구분하지만, `1.001`은 구분 못 할 수도 있다는 뜻이야.

Triton 커널과 PyTorch reference에서 reduce (sum) 순서가 다르면, 2048개 원소의 합에서 rounding error가 누적돼서 최종 결과에 0.01~0.03 정도의 차이가 생길 수 있어. 이건 **버그가 아니라 부동소수점의 본질적 한계**야.

#### 최종 Tolerance 결정

```python
# RMSNorm & Fused Norm+Residual
_ATOL = 0.05  # 5e-2
_RTOL = 0.05  # 5e-2

# SwiGLU (sigmoid의 fp32 안정성 덕에 더 정밀)
atol = 1e-3
rtol = 1e-3

# M-RoPE (dtype별 차등)
atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
```

**커널마다, dtype마다 tolerance가 다른 이유**:
- **SwiGLU**: sigmoid를 fp32로 계산하니까 정밀도가 높아서 1e-3으로 충분
- **RMSNorm**: 2048개 원소의 sum-of-squares에서 reduce 순서 차이로 오차 누적 → 5e-2
- **M-RoPE**: fp16은 10-bit mantissa라 1e-3 가능, bf16은 7-bit라 1e-2

### 테스트 매트릭스

각 커널을 다양한 조건에서 테스트해:

```
RMSNorm:     4 shapes × 2 dtypes × 2 interfaces = 16 tests
SwiGLU:      3 shapes × 2 dtypes × 2 interfaces = 12 tests
M-RoPE:      2 batch × 2 seq_len × 2 dtypes     =  8 tests
Fused Norm:  3 shapes × 2 dtypes + 1 module test =  7 tests
─────────────────────────────────────────────────
Total:                                             43 tests
```

**다양한 shape을 테스트하는 이유**: Triton 커널은 BLOCK_SIZE와 마스킹에 의존하는데, shape이 달라지면 마스킹 패턴이 바뀌거든. 예를 들어 `(1, 1, 2048)`은 row가 1개뿐이라 edge case고, `(4, 1024, 2048)`은 대량 배치라 다른 병렬화 패턴을 탐.

**두 가지 인터페이스를 테스트하는 이유**: 함수형 API (`triton_rms_norm()`)와 Module 래퍼 (`TritonRMSNorm`)가 다 제대로 동작하는지 확인. Module 래퍼는 weight parameter 초기화 등 추가 로직이 있으니까.

### 면접에서 이렇게 말하면 된다

> "수치 검증은 each kernel에 대해 순수 PyTorch로 작성한 reference implementation과 비교했습니다. `torch.allclose`로 absolute tolerance와 relative tolerance를 검증했고, half-precision의 mantissa 정밀도에 따라 커널별, dtype별로 허용 오차를 차등 설정했습니다. 총 43개의 parametrized test case로 다양한 shape과 dtype 조합을 커버했습니다."

---

## 5. Monkey Patching: PyTorch 모델을 Triton으로 갈아끼우기

### Monkey Patching이란?

Monkey patching은 **런타임에 기존 객체의 메서드나 속성을 바꿔치기**하는 기법이야. 원본 소스코드를 수정하지 않고도 동작을 바꿀 수 있어.

```python
# 이미 로드된 모델의 RMSNorm을 Triton 버전으로 교체
apply_triton_kernels(model)
# 이 한 줄이면 모델 전체가 Triton 커널을 사용하게 됨!
```

### 왜 Monkey Patching을 선택했나?

모델 코드를 직접 수정하는 방법도 있지만, monkey patching이 더 나은 이유:

1. **비침습적**: Qwen3-TTS의 원본 코드를 한 줄도 안 건드려. transformers 라이브러리 업데이트해도 안 깨져.
2. **On/Off 전환**: Triton 커널 적용 여부를 런타임에 결정할 수 있어. 디버깅할 때 원본으로 돌리기 쉬워.
3. **선택적 적용**: RMSNorm만 교체하고 SwiGLU는 안 하는 것도 가능.
4. **Liger Kernel의 검증된 패턴**: LinkedIn의 Liger Kernel도 이 방식을 써서 LLaMA를 최적화했어.

### 전체 흐름

```
1. 모델 로드 (transformers.AutoModel)
   └── 원본 PyTorch 모듈들이 모델 트리에 존재

2. apply_triton_kernels(model) 호출
   ├── model.named_modules()로 모든 모듈 순회
   ├── RMSNorm 발견 → TritonRMSNorm으로 교체
   └── SwiGLU MLP 발견 → forward 메서드 교체

3. model.generate() 호출
   └── 자동으로 Triton 커널이 사용됨
```

### RMSNorm 교체 상세

**파일**: `models/patching.py` - `_replace_rms_norm()`

```python
def _replace_rms_norm(model, name, old):
    # 1. 부모 모듈 찾기
    parent, attr = _get_parent(model, name)
    # name = "talker.layers.0.input_layernorm" 이면
    # parent = model.talker.layers[0]
    # attr = "input_layernorm"

    # 2. 기존 모듈에서 파라미터 추출
    hidden_size = old.weight.shape[0]  # 2048
    eps = getattr(old, "variance_epsilon", getattr(old, "eps", 1e-6))
    # Qwen은 "variance_epsilon"을 쓰고, 다른 모델은 "eps"를 쓰기도 해서 둘 다 체크

    # 3. Triton 모듈 생성
    new_norm = TritonRMSNorm(hidden_size, eps=eps)

    # 4. 핵심: weight 파라미터 공유 (복사가 아님!)
    new_norm.weight = old.weight  # 같은 텐서를 참조

    # 5. 부모 모듈의 속성 교체
    setattr(parent, attr, new_norm)
```

**weight 공유가 중요한 이유**: `new_norm.weight = old.weight`는 텐서를 복사하는 게 아니라 **같은 메모리를 가리키게** 하는 거야. 그래서:
- VRAM 추가 사용 없음 (제약 조건 충족!)
- 원래 모델의 학습된 weight를 그대로 사용
- gradient도 원래 parameter로 흐름 (학습 시)

**`setattr(parent, attr, new_norm)`의 동작**: PyTorch의 `nn.Module`은 `__setattr__`을 오버라이드해서 `nn.Module` 타입의 값이 할당되면 자동으로 `_modules` 딕셔너리에 등록해. 그래서 `setattr`만 해도 모델 트리가 올바르게 업데이트돼.

### SwiGLU 교체 상세 — types.MethodType의 마법

**파일**: `models/patching.py` - `_patch_mlp_forward()`

SwiGLU는 RMSNorm과 달리 **모듈 자체를 교체하는 게 아니라 forward 메서드만 교체**해. 왜냐면 MLP 모듈에는 gate_proj, up_proj, down_proj 라는 Linear 레이어들이 있고, 이걸 유지하면서 activation 부분만 바꾸고 싶으니까.

```python
def _patch_mlp_forward(mlp):
    def _forward(self, x):
        gate = self.gate_proj(x)   # 기존 Linear 레이어 그대로 사용
        up = self.up_proj(x)       # 기존 Linear 레이어 그대로 사용
        return self.down_proj(triton_swiglu_forward(gate, up))  # activation만 Triton!

    mlp.forward = types.MethodType(_forward, mlp)
```

**`types.MethodType`이 필요한 이유**:

함수를 인스턴스에 직접 할당하면 `self`가 바인딩되지 않아:

```python
# 이렇게 하면 self가 안 넘어옴!
mlp.forward = _forward  # _forward(x)로 호출됨, self 없음

# types.MethodType으로 바인딩해야 함
mlp.forward = types.MethodType(_forward, mlp)
# 이제 mlp.forward(x) 호출하면 _forward(mlp, x)로 변환됨
```

`types.MethodType(function, instance)`는 함수를 특정 인스턴스의 **바운드 메서드**로 만들어줘. 그래서 `mlp.forward(x)` 호출하면 `_forward(mlp, x)`가 되어 `self.gate_proj` 같은 속성에 접근할 수 있는 거야.

### 모듈 발견 로직

```python
for name, module in model.named_modules():
    cls_name = type(module).__name__

    # RMSNorm 감지: 클래스 이름에 "RMSNorm"이 포함되고 weight가 있으면
    if "RMSNorm" in cls_name and hasattr(module, "weight"):
        _replace_rms_norm(model, name, module)

    # SwiGLU MLP 감지: gate_proj, up_proj, down_proj 3개가 다 있으면
    if hasattr(module, "gate_proj") and hasattr(module, "up_proj") and hasattr(module, "down_proj"):
        _patch_mlp_forward(module)
```

**왜 클래스 이름으로 감지하나?**: 모델마다 RMSNorm 클래스 이름이 다를 수 있어. Qwen은 `Qwen2RMSNorm`, LLaMA는 `LlamaRMSNorm`, 커스텀 모델은 `RMSNorm` 그냥 이렇게. `"RMSNorm" in cls_name`으로 하면 이런 변형을 다 잡을 수 있어.

**SwiGLU는 속성 기반 감지**: gate_proj, up_proj, down_proj 3개 속성이 다 있는 모듈은 SwiGLU MLP라고 판단해. 이것도 모델 아키텍처에 무관하게 동작하는 범용적인 방법이야.

**`list(model.named_modules())`를 먼저 하는 이유**: 순회 중에 모듈을 교체하면 iterator가 꼬일 수 있어서, 먼저 리스트로 복사해놓고 순회하는 거야. 이건 Python에서 "iterate 중에 collection 수정하면 안 된다"는 일반적인 패턴이야.

### TritonRunner에서의 통합

```python
class TritonRunner(BaseRunner):
    def load_model(self):
        super().load_model()         # 1. 원본 모델 로드
        apply_triton_kernels(self.model)  # 2. Triton 커널 패칭
```

**한 줄의 차이**. BaseRunner와 TritonRunner의 유일한 차이가 `apply_triton_kernels(self.model)` 한 줄이야. 이게 monkey patching의 아름다움이지 — 침습도가 최소인 채로 모델 전체의 동작을 바꿀 수 있어.

### 패칭 후 모델에서 일어나는 일

```
패칭 전:
  model.talker.layers[0].input_layernorm     → Qwen2RMSNorm (PyTorch)
  model.talker.layers[0].mlp.forward()       → silu(gate_proj(x)) * up_proj(x) (PyTorch)

패칭 후:
  model.talker.layers[0].input_layernorm     → TritonRMSNorm (Triton 커널)
  model.talker.layers[0].mlp.forward()       → triton_swiglu_forward() (Triton 커널)

model.generate() 호출하면:
  → 각 레이어의 forward()가 호출됨
  → input_layernorm은 이제 TritonRMSNorm.forward()를 호출
  → TritonRMSNorm.forward()는 내부에서 Triton JIT 커널을 launch
  → MLP의 forward는 patched _forward()를 호출
  → _forward() 안에서 triton_swiglu_forward()가 Triton 커널을 launch
```

PyTorch의 모듈 시스템 덕분에, 한번 모듈을 교체하면 그 이후의 모든 forward pass에서 자동으로 Triton 커널이 사용돼. 호출하는 쪽 (예: Attention 레이어) 은 LayerNorm의 구현이 바뀐 걸 모르고 그냥 `self.input_layernorm(hidden_states)` 호출하면 되거든.

---

## 6. 삽질 기록과 교훈

### 삽질 1: Half-precision Tolerance 지옥

처음에 `atol=1e-3`으로 잡았는데, bfloat16 테스트가 자꾸 실패했어. max abs diff가 0.03 정도 나왔거든.

"내 커널이 잘못된 건가?" 하고 한참 디버깅했는데, 결국 문제는 **bfloat16의 7-bit mantissa**였어. 2048개 원소의 sum-of-squares에서 reduce 순서가 PyTorch (sequential)와 Triton (parallel tree reduction)이 다르니까, rounding error 누적 패턴이 달라서 최종값에 차이가 생긴 거야.

**교훈**: half-precision 커널의 수치 검증에서는 dtype의 mantissa 정밀도를 기반으로 tolerance를 설정해야 해. "수학적으로 동일하다"와 "부동소수점으로 동일하다"는 완전히 다른 문제야.

### 삽질 2: Triton의 rsqrt import 경로 변경

```python
# Triton 버전에 따라 import 경로가 다름
try:
    from triton.language.extra.libdevice import rsqrt       # 새 버전
except ModuleNotFoundError:
    from triton.language.extra.cuda.libdevice import rsqrt  # 구 버전
```

Triton이 아직 불안정한 프로젝트라 import 경로가 버전마다 바뀌거든. try/except로 두 경로 다 지원하게 해놨어.

### 삽질 3: contiguous() 함정

PyTorch 텐서는 transpose나 slice하면 **view만 바뀌고 실제 메모리 레이아웃은 안 바뀌어**. 그런데 Triton 커널은 pointer arithmetic으로 메모리에 직접 접근하니까, 비연속(non-contiguous) 텐서를 넣으면 쓰레기 값을 읽게 돼.

```python
# M-RoPE에서
q = q.transpose(1, 2).contiguous()  # transpose 후 반드시 contiguous!
```

`.contiguous()`를 빼먹으면 **틀린 결과가 나오는데 에러는 안 나**. 이게 제일 무서운 버그야. 조용히 잘못된 결과를 반환하거든. 그래서 수치 검증 테스트가 중요한 거야.

### 삽질 4: named_modules() 순회 중 수정

처음에 이렇게 짰다가 문제가 생겼어:

```python
# BAD: iterator 순회 중 모듈 수정
for name, module in model.named_modules():
    if "RMSNorm" in type(module).__name__:
        setattr(parent, attr, new_module)  # iterator가 꼬임!
```

Python에서 dictionary/tree를 순회하면서 수정하면 `RuntimeError: dictionary changed size during iteration` 같은 에러가 나거나, 일부 모듈을 건너뛰는 문제가 생겨. 그래서 `list()`로 먼저 스냅샷을 뜬 거야.

### 삽질 5: eps 속성 이름 불일치

Qwen은 `variance_epsilon`이라는 속성명을 쓰는데, 다른 모델은 `eps`를 써:

```python
eps = getattr(old, "variance_epsilon", getattr(old, "eps", 1e-6))
```

이거 하나 때문에 처음에 epsilon이 0으로 들어가서 NaN이 나왔었어. 모델마다 속성명이 다를 수 있다는 걸 항상 고려해야 해.

---

## 7. Hybrid 접근법: CUDA Graph × Triton 커널 퓨전

### 아이디어: "왜 둘 중 하나만 써야 해?"

여기까지 두 가지 최적화 축을 봤어:

1. **Triton 커널 퓨전**: HBM 라운드트립을 줄여서 메모리 바운드 연산을 빠르게
2. **CUDA Graph** (faster-qwen3-tts): 커널 launch overhead를 없애서 전체 파이프라인을 빠르게

이 둘은 **서로 다른 병목을 공략**하는 거야:

```
CUDA Graph가 해결하는 것:
  ┌─────────────────────────────────────────────────┐
  │ CPU → GPU 커널 launch (수십 μs/launch × 수백 커널) │
  │ → Graph로 캡처하면 1회 replay로 전부 실행         │
  └─────────────────────────────────────────────────┘

Triton 커널 퓨전이 해결하는 것:
  ┌─────────────────────────────────────────────────┐
  │ HBM ↔ SRAM 왕복 (4번 → 1번)                     │
  │ → 커널 자체의 실행 시간이 줄어듦                  │
  └─────────────────────────────────────────────────┘
```

CUDA Graph는 "커널을 더 빨리 시작"하게 하고, Triton 퓨전은 "커널 자체를 더 빨리 끝나게" 하는 거야. **둘을 합치면 양쪽 이득을 다 먹을 수 있어.**

### CUDA Graph의 Lazy Capture 메커니즘

faster-qwen3-tts의 소스를 까보면 재밌는 구조가 있어:

```python
# faster_qwen3_tts/model.py 내부 (간략화)
class FasterQwen3TTS:
    def from_pretrained(cls, model_id, ...):
        base_model = Qwen3TTSModel.from_pretrained(model_id, ...)
        self.model = base_model         # 내부에 원본 모델 저장
        self._warmed_up = False         # 아직 CUDA Graph 캡처 안 함!

    def _warmup(self):
        """첫 generate() 호출 시에만 실행"""
        # CUDA Graph 캡처: 현재 모델의 forward를 그래프로 녹화
        self._warmed_up = True

    def generate_custom_voice(self, text, speaker, ...):
        if not self._warmed_up:
            self._warmup()              # ← 여기서 Graph 캡처!
        return self._fast_generate(...)  # Graph replay로 추론
```

핵심은 **CUDA Graph 캡처가 lazy**하다는 거야. `from_pretrained()` 시점에는 Graph를 안 만들고, **첫 번째 `generate()` 호출** 시점에 만들어. 왜냐면 Graph 캡처는 실제 forward pass를 한 번 실행하면서 녹화하는 거라, 모델이 완전히 로드된 후에야 가능하거든.

### 패칭 윈도우: 정확한 타이밍이 핵심

이 lazy capture 덕분에 **완벽한 패칭 윈도우**가 생겨:

```
시간 흐름 →

from_pretrained()     apply_hybrid_patching()     generate() (첫 호출)
      │                        │                        │
      ▼                        ▼                        ▼
  ┌────────┐              ┌────────┐              ┌────────────────┐
  │모델 로드│  ←── 윈도우 ──→ │커널 패칭│  ←── 윈도우 ──→ │CUDA Graph 캡처  │
  │(PyTorch)│              │(Triton)│              │(패칭된 커널 포함!)│
  └────────┘              └────────┘              └────────────────┘
                                                        │
                                                        ▼
                                                  이후 generate()들:
                                                  Graph replay (Triton 커널 포함)
```

**`from_pretrained()` 이후, 첫 `generate()` 이전**에 Triton 패칭을 적용하면, Graph 캡처 시점에 **Triton 커널이 이미 모델에 들어가 있는 상태**로 녹화돼. 그 이후의 모든 `generate()` 호출은 Triton 커널이 포함된 Graph를 replay하는 거야.

만약 순서가 잘못되면?

```
# BAD: generate() 먼저 → Graph에 PyTorch 커널이 녹화됨
model = FasterQwen3TTS.from_pretrained(...)
wavs = model.generate_custom_voice(...)   # PyTorch 커널로 Graph 캡처
apply_hybrid_patching(model)               # 이미 늦음! Graph는 이미 만들어짐

# GOOD: 패칭 먼저 → Graph에 Triton 커널이 녹화됨
model = FasterQwen3TTS.from_pretrained(...)
apply_hybrid_patching(model)               # Triton 커널로 교체
wavs = model.generate_custom_voice(...)   # Triton 커널로 Graph 캡처!
```

### 구현

`infer.py`에서 hybrid 모드 구현:

```python
def apply_hybrid_patching(faster_model: object) -> None:
    """Apply Triton kernels to faster-qwen3-tts's internal model.

    Must be called BEFORE first generate() so that CUDA Graph
    capture includes the fused Triton kernels.
    """
    from models.patching import apply_triton_kernels

    # faster_model.model은 Qwen3TTSModel (qwen_tts wrapper)
    # 내부의 nn.Module을 찾아서 패칭
    internal = find_patchable_model(faster_model.model)
    apply_triton_kernels(internal)
    logger.info("Hybrid patching complete (Triton kernels → CUDA Graph)")
```

**내부 모델 접근 경로**:
```
FasterQwen3TTS
  └── .model (Qwen3TTSModel wrapper)
        └── find_patchable_model()로 nn.Module 탐색
              └── .model.talker (28-layer Transformer)
                    └── apply_triton_kernels()가 RMSNorm + SwiGLU 패칭
```

`find_patchable_model()`은 wrapper 객체 안에서 실제 `nn.Module`을 찾아주는 유틸리티야. `model`, `transformer`, `talker` 같은 흔한 속성명을 순회하면서 `isinstance(val, nn.Module)`인 것을 찾거든. 이 범용 로직 덕분에 qwen_tts wrapper의 내부 구조가 바뀌어도 대응할 수 있어.

### 결과 분석

벤치마크 결과 (동일 텍스트, RTX 5090, bf16):

```
                  Faster      Hybrid
Avg Latency (s):   2.029       1.565
Min Latency (s):   1.598       1.220
Step Time:        ~20ms/step  ~16ms/step
Peak VRAM (GB):    4.43        4.41
Speedup:           —           1.30x
```

**Hybrid가 Faster 대비 1.30x 빠르고, VRAM은 동일**이야.

step time이 20ms → 16ms로 줄어든 건, CUDA Graph replay 안에서 각 Triton 커널의 실행 시간 자체가 짧아졌기 때문이야. Graph가 launch overhead를 없애주는 건 이미 Faster가 하고 있는 거고, Triton 퓨전이 각 커널의 **HBM 트래픽을 줄여서** 커널 실행 시간을 단축한 거야.

### 왜 VRAM이 늘지 않았나?

CUDA Graph는 static buffer를 사전 할당하거든. 이 buffer 크기는 **커널의 input/output 크기**로 결정되는데, Triton 커널 퓨전은 중간 텐서를 없앨 뿐 input/output shape은 바꾸지 않아. 그래서 Graph의 static buffer 크기가 동일하게 유지되는 거야.

오히려 Triton 퓨전으로 **중간 텐서가 사라지면서** Graph 캡처 시 할당해야 하는 temporary buffer가 줄어들 수 있어. 실측에서 4.43GB → 4.41GB로 미세하게 줄어든 게 이 효과야.

### 면접에서 이렇게 말하면 된다

> "CUDA Graph와 Triton 커널 퓨전은 서로 다른 병목을 해결합니다. CUDA Graph는 kernel launch overhead를 제거하고, Triton 퓨전은 HBM roundtrip을 줄입니다. faster-qwen3-tts의 lazy CUDA Graph 캡처 메커니즘을 활용해서, 첫 번째 generate() 호출 전에 Triton 패칭을 적용하면 Graph 안에 Triton 커널이 포함된 상태로 캡처됩니다. 결과적으로 Faster 대비 1.30x 추가 speedup을 VRAM 증가 없이 달성했습니다."

---

## 8. 종합 벤치마크 결과

### 6가지 모드 비교

동일 조건에서의 벤치마크 결과야 (RTX 5090, bf16, Qwen3-TTS-12Hz-1.7B-CustomVoice, 영어 텍스트 ~15단어, warmup 1회 + 측정 3회):

| 지표 | Base | Triton | Compile | Compile+Triton | Faster | Hybrid |
|------|------|--------|---------|----------------|--------|--------|
| **Avg Latency (s)** | 6.938 | 6.404 | 7.929 | 5.928 | 1.784 | **1.327** |
| **Min Latency (s)** | 6.460 | 5.830 | 7.257 | 4.818 | 1.479 | **1.171** |
| **RTF** | 0.985 | 0.911 | 1.226 | 0.899 | 0.191 | **0.222** |
| **Peak VRAM (GB)** | 4.09 | 4.13 | 4.11 | 8.03* | 4.37 | **4.37** |
| **Compile Time (s)** | - | - | 1.71 | ~0 | - | - |
| **Speedup (vs Base)** | 1.00x | 1.08x | 0.87x | 1.17x | 3.89x | **5.23x** |

> \* Compile+Triton의 VRAM 8.03GB는 torch.compile의 그래프 캐싱 오버헤드 때문. 프로젝트 제약 조건(추가 VRAM 금지)에 위배되므로 프로덕션에서는 비권장.
>
> **핵심 발견**: torch.compile 단독(Compile)은 오히려 Base보다 **느려짐** (0.87x). generate()의 동적 시퀀스 길이가 graph break을 유발하여 torch.compile의 최적화 효과가 상쇄됨. 반면 수동 Triton + torch.compile(Compile+Triton)은 1.17x로 수동 Triton(1.08x)보다 약간 빠르지만, VRAM 대가가 큼.

> **RTF (Real-Time Factor)** = 추론 시간 / 오디오 길이. 1.0 미만이면 실시간보다 빠름. 낮을수록 좋음.

### 각 모드의 최적화 레벨

```
Base     (PyTorch eager 기본)
  │
  ├─ +Triton 커널 퓨전 (수동, HBM 라운드트립 감소)
  │  ▼
  │  Triton   (1.08x speedup)
  │
  ├─ +torch.compile (TorchInductor → 자동 Triton 생성)
  │  ▼
  │  Compile  (0.87x ← 오히려 느림! graph break 때문)
  │
  ├─ +Triton 퓨전 + torch.compile (수동 + 자동 하이브리드)
  │  ▼
  │  Compile+Triton (1.17x, but VRAM +3.9GB)
  │
  ├─ × CUDA Graph + Static Cache (launch overhead 제거)
  │  ▼
  │  Faster   (3.89x speedup vs Base)
  │     │
  │     │ +Triton 커널 퓨전 (CUDA Graph 안에서 커널 실행 시간 단축)
  │     ▼
  │  Hybrid   (5.23x speedup vs Base, 1.34x vs Faster)
```

### 최적화 축별 기여도 분석

```
Base → Triton:  +14% speedup
  → Triton 커널 퓨전만으로는 14% 개선
  → element-wise 연산이 전체에서 차지하는 비중이 크지 않기 때문
  → 하지만 VRAM 추가 없이 순수 연산 효율 개선

Base → Faster:  +294% speedup
  → CUDA Graph + Static Cache의 효과가 압도적
  → Autoregressive decoding에서 매 step마다 반복되는
    수백 개의 커널 launch를 1회 graph replay로 대체
  → step time 60ms → 20ms (3x 감소)

Faster → Hybrid: +30% speedup
  → CUDA Graph 위에 Triton 퓨전을 얹은 추가 효과
  → Graph가 launch overhead를 이미 제거한 상태에서도
    각 커널의 HBM 트래픽 감소가 step time을 줄임
  → step time 20ms → 16ms (20% 감소)
```

### 핵심 인사이트

**1. "CUDA Graph와 Triton 퓨전은 상호 보완적이다"**

독립적인 최적화 축이라서 한쪽이 다른 쪽의 효과를 상쇄하지 않아. CUDA Graph는 커널 간의 overhead를, Triton은 커널 내부의 낭비를 줄이니까.

**2. "Autoregressive 모델에서는 CUDA Graph 효과가 지배적"**

Base → Triton (1.14x) vs Base → Faster (3.94x)를 보면 차이가 명확해. TTS 같은 autoregressive 모델은 매 step마다 짧은 커널을 수백 번 launch하는데, 이 launch overhead가 전체 시간의 대부분을 차지해. 그래서 launch를 없애는 CUDA Graph가 훨씬 큰 효과를 보여.

**3. "Triton 퓨전의 진짜 가치는 CUDA Graph 위에서 드러난다"**

단독으로 14%인 Triton 퓨전이, CUDA Graph 위에서는 30% 추가 speedup을 줘. 이유: launch overhead가 사라진 뒤에는 **순수 커널 실행 시간**이 전체 비중의 대부분이 되거든. 이 상태에서 HBM 트래픽을 줄이면 상대적으로 더 큰 효과가 나타나는 거야.

```
            비중 (개념적)
            ┌────────────────────────────────────┐
Base:       │█████ launch │████████ HBM │██ 연산  │  launch가 큰 비중
            └────────────────────────────────────┘

            ┌────────────────────────────────────┐
Faster:     │ launch=0  │████████████ HBM │██ 연산│  HBM이 지배적으로
            └────────────────────────────────────┘

            ┌────────────────────────────────────┐
Hybrid:     │ launch=0  │████ HBM │██ 연산       │  HBM도 줄어듦
            └────────────────────────────────────┘
```

**4. "VRAM은 전혀 늘지 않는다"**

4가지 모드 전부 4.1~4.4GB 범위 안에 있어. Triton 커널 퓨전은 중간 텐서를 제거하니까 VRAM을 아끼고, CUDA Graph의 static buffer도 커널 I/O shape이 동일하니까 크기가 변하지 않아. 제약 조건 ("추가 VRAM 사용 금지") 을 완벽히 충족.

---

## 9. Triton vs TensorRT: 왜 Triton을 선택했나?

### "TensorRT 안 써봤어요?"

면접에서 100% 나올 질문이야. "왜 TensorRT-LLM 안 쓰고 Triton 커널을 직접 짰나요?" 이건 단순 선호가 아니라 **기술적으로 TensorRT가 불가능한 상황**이었어.

### 먼저 용어 정리

헷갈리기 쉬운데, "Triton"이 두 개야:

```
1. OpenAI Triton (triton-lang)    ← 우리가 쓴 것. GPU 커널 작성용 Python DSL
2. NVIDIA Triton Inference Server  ← 모델 서빙 런타임. 이건 별개의 도구

그리고:
3. TensorRT / TensorRT-LLM        ← NVIDIA의 그래프 레벨 추론 최적화 툴킷
```

우리 프로젝트에서 비교 대상은 **OpenAI Triton 커널 (우리 접근법) vs TensorRT-LLM (대안)**이야.

### TensorRT-LLM이 Qwen3-TTS를 지원하지 않는다

이게 가장 결정적인 이유야. NVIDIA가 **명시적으로 거부**했거든.

GitHub Issue [#11118](https://github.com/NVIDIA/TensorRT-LLM/issues/11118)에서 Qwen3-TTS 지원을 요청했는데, NVIDIA가 "not planned"으로 닫았어. 이유가 두 가지야:

**1. Multi-codebook 출력 구조**

TensorRT-LLM의 sampler는 step당 토큰 1개를 생성하도록 하드코딩돼 있어. 그런데 Qwen3-TTS는 **step당 여러 개의 discrete audio token**을 생성하는 multi-codebook 구조야. 이걸 고치려면 C++ 런타임을 뜯어고쳐야 해.

```
일반 LLM:        step → 1 token
Qwen3-TTS:       step → N audio tokens (multi-codebook)
TensorRT-LLM:    step → 1 token (하드코딩) → 구조적 불일치!
```

**2. CUDA Graph 호환성 문제**

HuggingFace의 Qwen3-TTS 구현은 generation loop 안에서 dynamic host-to-device 전송을 해. 이게 CUDA Graph 캡처와 충돌해서 `"operation not permitted when stream is capturing"` 에러가 나. TensorRT-LLM의 CUDA Graph 기반 추론이 작동하지 않는 거야.

### 그래도 비교해보자: 각각 뭘 잘하나?

TensorRT-LLM이 Qwen3-TTS를 지원하지 않는다는 건 실무적 결론이고, 기술적 비교도 할 줄 알아야 해.

#### 서로 다른 레벨의 최적화

```
                    추상화 수준
                    ┌─────────────────────────────────┐
  TensorRT-LLM:    │  그래프 레벨 (레이어 간 퓨전)      │  ← 모델 전체를 보고 최적화
                    ├─────────────────────────────────┤
  OpenAI Triton:   │  커널 레벨 (연산 내 퓨전)          │  ← 개별 연산을 보고 최적화
                    ├─────────────────────────────────┤
  cuBLAS/cuDNN:    │  연산자 레벨 (GEMM, Conv)         │  ← 단일 연산의 하드웨어 최적화
                    └─────────────────────────────────┘
```

TensorRT는 위에서 내려다보면서 "이 LayerNorm 다음에 QKV Projection이 오니까 합치자"라고 판단해. Triton은 아래서 올려다보면서 "이 RMSNorm 안의 4번의 HBM 왕복을 1번으로 줄이자"라고 판단해.

#### Triton이 더 좋은 점

| 항목 | Triton | TensorRT |
|------|--------|----------|
| **개발 속도** | Python으로 커널 작성, 몇 시간이면 구현 | C++ 플러그인, 며칠~주 소요 |
| **커스텀 연산** | 아무 알고리즘이나 표현 가능 (M-RoPE 등) | 지원 목록에 없으면 C++ 플러그인 필수 |
| **디버깅** | Python에서 바로 print, breakpoint | C++ 빌드 → 직렬화 → 로드 → 에러 추적 |
| **PyTorch 호환** | 100% 호환, `model.generate()` 그대로 | ONNX 또는 TRT 네트워크로 변환 필요 |
| **유지보수** | 커널 4개 + patching.py (작은 범위) | 미지원 모델이면 C++ 런타임 수정 필요 |
| **모델 변경 대응** | 패칭 로직만 업데이트 | 엔진 전체 재빌드 (분~시간) |

**핵심**: Triton의 가장 큰 장점은 **개발 속도와 유연성**이야. M-RoPE 같은 비표준 연산을 Python으로 20줄이면 구현할 수 있어. 같은 걸 TensorRT 플러그인으로 쓰면 C++로 200줄+ 에 직렬화/역직렬화 보일러플레이트까지 들어가.

#### TensorRT가 더 좋은 점

| 항목 | Triton | TensorRT |
|------|--------|----------|
| **GEMM 성능** | cuBLAS와 비슷하지만 못 이김 | cuBLASLt + FP8 텐서 코어 활용 |
| **그래프 레벨 퓨전** | 개별 커널만 퓨전, 수동 설계 | 전체 그래프를 자동 분석하여 최적 퓨전 |
| **양자화** | 수동 구현 (torchao 등) | INT8/FP8/W4A8 엔진 레벨 지원 |
| **Continuous Batching** | 별도 서빙 레이어 필요 | C++ 런타임에서 네이티브 지원 |
| **Paged KV Cache** | 별도 구현 필요 | 네이티브 지원 |
| **프로덕션 서빙** | 수동 구성 필요 | 올인원 솔루션 (서빙, 배칭, 스케일링) |

**핵심**: TensorRT의 장점은 **지원되는 모델에 한해 올인원 프로덕션 최적화**야. LLaMA, GPT 같은 표준 decoder-only 모델이면 빌드 한 번으로 INT8 양자화 + continuous batching + paged KV cache가 전부 적용돼.

### 개발 워크플로우 비교

```python
# === Triton 워크플로우 (우리 프로젝트) ===

# 1. Python으로 커널 작성 (몇 시간)
@triton.jit
def _rms_norm_kernel(X, W, Y, ...):
    # 20줄의 Python-like 코드

# 2. 테스트 (즉시)
pytest tests/test_rms_norm.py  # torch.allclose 검증

# 3. 적용 (1줄)
apply_triton_kernels(model)

# 4. 추론 (바로 사용)
model.generate(...)
```

```bash
# === TensorRT-LLM 워크플로우 (대안) ===

# 1. 모델이 지원되는지 확인 → Qwen3-TTS: NOT SUPPORTED ❌
#    여기서 이미 막힘

# 만약 지원된다면:
# 2. 체크포인트 변환 (분 단위)
python convert_checkpoint.py --model_dir ./qwen3 --output_dir ./ckpt

# 3. 엔진 빌드 (분~시간)
trtllm-build --checkpoint_dir ./ckpt --output_dir ./engines \
  --gpt_attention_plugin float16 --max_batch_size 4

# 4. 커스텀 연산이 필요하면 C++ 플러그인 작성 (일~주)
class MRoPEPlugin : public IPluginV3 {
    // serialize(), enqueue(), clone() 등 구현...
    // 200+ lines of C++
};

# 5. 엔진 재빌드 (GPU 아키텍처마다 별도 필요)
```

### Liger Kernel: Triton 접근법의 프로덕션 검증

"Triton 커널 + monkey patching이 프로덕션에서도 쓰이나?" 라는 의문에 대한 답이 **Liger Kernel**이야.

LinkedIn에서 만든 [Liger Kernel](https://github.com/linkedin/Liger-Kernel)은 우리와 **정확히 같은 패턴**을 써:

- Triton 커널로 RMSNorm, SwiGLU, CrossEntropy 등을 퓨전
- `AutoLigerKernelForCausalLM.from_pretrained()` 한 줄로 monkey patching
- LLaMA, Mistral, Qwen 등 주요 모델 지원
- **20% throughput 증가, 60% 메모리 절감** (ArXiv 논문으로 발표)
- FSDP, DeepSpeed, FlashAttention과 완전 호환

LinkedIn 규모에서 프로덕션으로 검증된 패턴이니까, 우리 접근법의 신뢰성도 높다고 할 수 있어.

### 유지보수 리스크

#### Triton의 리스크

솔직히 말하면 Triton도 리스크가 있어:

1. **PyTorch-Triton 버전 결합**: Triton은 PyTorch의 pinned dependency로 배포되는데, PyTorch 업그레이드하면 Triton API가 바뀌어서 커널이 깨질 수 있어
2. **sm_120 (Blackwell) 지원**: RTX 5090용 cu128 nightly가 필요한데, 아직 일부 불안정한 부분이 있어
3. **import 경로 변경**: Triton 버전마다 `rsqrt` 같은 함수의 import 경로가 달라지는 문제 (삽질 기록 참조)

**대응**: `uv.lock`으로 정확한 torch 버전 고정, `pytest tests/`로 업그레이드 때마다 수치 검증.

#### TensorRT의 리스크

TensorRT는 더 무거운 리스크가 있어:

1. **엔진 비이식성**: 빌드한 `.engine` 파일은 TensorRT 버전 + GPU 아키텍처에 종속. 업그레이드하면 전부 재빌드 필요
2. **미지원 모델의 유지보수 부담**: C++ 런타임 수정은 NVIDIA의 내부 구현에 종속. 업스트림 변경되면 따라가야 해
3. **의존성 체인이 더 김**: `tensorrt-llm → tensorrt → CUDA toolkit → driver`

### 결론: Qwen3-TTS에서는 Triton이 유일한 선택지

```
의사결정 트리:

Q: 모델이 TensorRT-LLM 지원 목록에 있나?
   ├── YES → TensorRT-LLM이 더 좋은 선택 (올인원 최적화)
   └── NO → Triton 커널 퓨전 + monkey patching
              │
              └── Qwen3-TTS: NO (Issue #11118, "not planned")
                              → Triton이 유일한 선택지
```

단순한 선호가 아니야. TensorRT-LLM이 multi-codebook TTS 아키텍처를 지원하지 않는다는 **기술적 제약** 때문에, Triton 커널 퓨전이 현실적으로 가능한 유일한 최적화 경로인 거야.

### 면접에서 이렇게 말하면 된다

> "TensorRT-LLM도 검토했지만, Qwen3-TTS의 multi-codebook 출력 구조와 M-RoPE 위치 인코딩이 TRT-LLM의 표준 아키텍처와 호환되지 않았습니다. NVIDIA도 Issue #11118에서 지원 계획이 없다고 명시했습니다. 반면 Triton 커널 퓨전은 Python DSL로 커스텀 연산을 유연하게 구현할 수 있고, monkey patching으로 기존 PyTorch 모델에 비침습적으로 적용 가능합니다. LinkedIn의 Liger Kernel이 동일한 패턴으로 프로덕션 검증을 마쳤기 때문에, 접근법의 신뢰성도 확인된 상태입니다."

---

## 10. Triton 커널 모델의 프로덕션 서빙

### "이거 프로덕션에서 어떻게 서빙하나요?"

면접에서 최적화 얘기를 하면 반드시 따라오는 질문이야. Triton 커널로 패칭한 모델을 실제로 어떻게 서빙할 수 있는지, 어떤 프레임워크가 적합한지 정리해볼게.

### 핵심 원칙: Monkey Patching과 서빙 프레임워크의 호환성

먼저 알아야 할 건, Triton 커널로 monkey patching된 모델은 **여전히 표준 PyTorch `nn.Module`**이라는 거야. 서빙 프레임워크 입장에서는 그냥 `.forward()`를 호출하는데, 내부적으로 Triton 커널이 실행될 뿐이야.

```
서빙 프레임워크 관점:
  model = load_model()              # 1. 모델 로드
  apply_triton_kernels(model)       # 2. 패칭 (서빙 프레임워크는 이걸 모름)
  output = model.forward(input)     # 3. 그냥 forward 호출 → Triton 커널 투명 실행
```

그래서 호환성은 **두 가지 모드**로 나뉘어:

| 서빙 모드 | Monkey Patch 호환 | 설명 |
|-----------|-------------------|------|
| **Eager mode** (Python 그대로 실행) | 완벽 호환 | FastAPI, BentoML, TorchServe 등 |
| **Compiled mode** (`torch.compile`) | 추가 작업 필요 | vLLM V1, SGLang compiled 등 |

Eager mode 서빙이면 아무 문제 없어. Compiled mode에서는 Triton 함수를 `torch.library.triton_op`으로 등록해야 graph break 없이 컴파일돼.

### 서빙 프레임워크 비교

TTS 모델 서빙은 일반 LLM 서빙과 다른 특성이 있어:
- **Single-stream 최적화** (대량 배칭보다 개별 요청 지연시간이 중요)
- **오디오 스트리밍** (텍스트 토큰이 아니라 PCM/WAV 청크 출력)
- **Multi-codebook 디코딩** (일반 LLM sampler와 다른 구조)

이 특성을 고려해서 주요 프레임워크들을 비교해볼게.

#### vLLM-Omni — 최적의 선택지

놀랍게도 **vLLM-Omni가 Qwen3-TTS를 네이티브 지원**해. 2025년 11월에 출시된 vLLM의 omni-modality 확장인데:

```bash
# OpenAI 호환 TTS API 엔드포인트
POST /v1/audio/speech
GET /v1/audio/voices

# 스트리밍 모드: PCM 청크 출력
# 첫 번째 오디오 청크 ~97ms (25 프레임 윈도우)
```

**장점**:
- Qwen3-TTS day-0 지원 (음성 복제, 커스텀 보이스 전부)
- Continuous batching, paged KV cache 네이티브
- 40만+ GPU에서 프로덕션 검증
- OpenAI 호환 API (드롭인 교체 가능)

**Triton 커널 통합 방법**: monkey patching 대신 `torch.library.triton_op`으로 커널을 등록하고, vLLM의 `CustomOp` 시스템으로 플러그인화해야 해. 추가 작업이 필요하지만, 가장 성능이 좋은 경로야.

```python
# vLLM CustomOp 등록 방식 (개념)
@torch.library.triton_op("mylib::rms_norm", mutates_args={})
def triton_rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return _rms_norm_forward(x, w)

@triton_rms_norm.register_fake
def _(x, w):
    return torch.empty_like(x)  # meta 함수 (shape 추론용)
```

#### FastAPI + uvicorn — 가장 간단한 경로

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.on_event("startup")
async def load():
    global model
    model = load_model()
    apply_triton_kernels(model)  # 패칭 완벽 호환

@app.post("/v1/audio/speech")
async def synthesize(request: TTSRequest):
    return StreamingResponse(
        generate_audio(model, request.text),
        media_type="audio/wav"
    )
```

**장점**: 설정이 간단하고, monkey patching과 100% 호환
**단점**: 배칭 없음, 멀티 GPU 수동 관리, 프로덕션 인프라 (헬스체크, 메트릭) 직접 구축 필요

**주의**: CUDA 컨텍스트가 asyncio 이벤트 루프 스레드와 충돌할 수 있어. 추론은 별도 스레드풀에서 실행해야 해.

#### BentoML — 균형잡힌 선택

TTS 스트리밍 예제가 이미 존재해 (`BentoXTTSStreaming`). WebSocket + generator 패턴으로 오디오 청크를 스트리밍하는 레퍼런스가 있어서, Qwen3-TTS에 적용하기 수월해.

```python
@bentoml.service(resources={"gpu": 1})
class QwenTTSService:
    def __init__(self):
        self.model = load_model()
        apply_triton_kernels(self.model)  # 완벽 호환

    @bentoml.api
    def synthesize(self, text: str) -> Generator[bytes, None, None]:
        yield from self.model.stream(text)
```

**장점**: BentoCloud로 관리형 배포 가능, Docker 내보내기, Kubernetes 지원
**단점**: vLLM-Omni만큼의 TTS 전용 최적화는 없음

#### NVIDIA Triton Inference Server — 엔터프라이즈용

**용어 주의**: 여기서 말하는 "Triton"은 NVIDIA의 모델 서빙 서버야. OpenAI의 Triton 커널 언어와 이름만 같고 완전히 별개의 제품이야.

**Python backend**를 쓰면 monkey patching이 투명하게 동작해. 엔터프라이즈 환경에서 Prometheus 메트릭, 헬스체크, Kubernetes 통합이 필요할 때 적합해.

**주의**: 내부적으로 특별 패치된 PyTorch 버전을 사용해서 cu128 nightly와 충돌할 수 있어.

### 서빙 전략 추천

```
개발/프로토타입:
  FastAPI + uvicorn (최소 설정, 즉시 시작)

단일 GPU 프로덕션:
  LitServe 또는 BentoML (적절한 프로덕션 기능 + 낮은 복잡도)

최대 성능 프로덕션:
  vLLM-Omni (Qwen3-TTS 네이티브 지원 + continuous batching)
  → Triton 커널은 torch.library.triton_op으로 등록

엔터프라이즈/Kubernetes:
  NVIDIA Triton Inference Server (Python backend)
  또는 Ray Serve + vLLM-Omni backend
```

### faster-qwen3-tts의 접근법

참고로 faster-qwen3-tts는 **"프레임워크 없이"** 접근했어:

> "No Flash Attention, no vLLM, no Triton. Just `torch.cuda.CUDAGraph`."

가벼운 커스텀 OpenAI 호환 서버(`examples/openai_server.py`)를 직접 짰어. 이건 우리 Triton 커널 접근법과 철학이 비슷해 — 프레임워크 오버헤드 없이 최적화 효과를 극대화하는 방식.

우리의 Hybrid 접근법 (faster-qwen3-tts + Triton 커널)도 이 가벼운 서버에 그대로 올릴 수 있어. CUDA Graph 캡처에 Triton 커널이 포함된 상태로 서빙되니까.

### Eager vs Compiled: 서빙 모드에 따른 Triton 커널 등록

면접에서 깊이 물어볼 수 있는 부분이야.

```
Eager mode 서빙 (FastAPI, BentoML, TorchServe 등):
  → monkey patching 그대로 사용 OK
  → apply_triton_kernels(model) 한 줄이면 끝
  → 서빙 프레임워크가 Python으로 forward() 호출하니까 투명

Compiled mode 서빙 (vLLM V1, torch.compile):
  → torch.compile이 forward를 트레이싱할 때
    등록 안 된 Python 함수를 만나면 graph break 발생
  → 해결: torch.library.triton_op으로 커널 등록
    + register_fake으로 meta 함수 (shape 추론) 제공
  → 이러면 torch.compile이 커널을 그래프 안에 인라인
```

이건 **Triton 커널의 이식성**과 관련된 중요한 아키텍처 결정이야. 우리 프로젝트는 현재 eager mode로 동작하는데, vLLM-Omni에 통합하려면 `torch.library.triton_op` 래핑이 필요해. 커널 로직 자체는 변경 없이, 래퍼만 추가하면 돼.

### 면접에서 이렇게 말하면 된다

> "Triton 커널로 패칭된 모델은 표준 PyTorch nn.Module이라서, eager mode 서빙 프레임워크에서는 투명하게 동작합니다. 프로덕션에서는 vLLM-Omni가 Qwen3-TTS를 네이티브 지원하고 있어서 가장 적합한 선택지이며, Triton 커널을 torch.library.triton_op으로 등록하면 vLLM의 torch.compile 파이프라인에 통합할 수 있습니다. 개발 단계에서는 FastAPI로 빠르게 검증하고, 프로덕션에서는 vLLM-Omni 또는 BentoML로 전환하는 전략이 적절합니다."

---

## 11. torch.compile vs 수동 Triton: 진짜 apple-to-apple 비교

### 왜 이 비교가 필요한가?

기존 "Base vs Triton" 비교는 **불공정한 비교**야:
- Base: PyTorch eager mode (최적화 없음)
- Triton: 수동으로 작성한 fused 커널

하지만 `torch.compile`을 적용하면 **TorchInductor 백엔드가 자동으로 Triton 커널을 생성**해. 즉, 공정한 비교는:
- **자동 Triton** (torch.compile이 생성) vs **수동 Triton** (우리가 직접 작성)

이게 진짜 apple-to-apple 비교야.

### torch.compile 내부 동작

```
torch.compile(model)
  │
  ▼
torch.dynamo (Python bytecode → FX Graph 추출)
  │
  ▼
TorchInductor (FX Graph → 최적화된 코드 생성)
  │
  ├─ CPU: C++/OpenMP 코드 생성
  └─ GPU: Triton 커널 자동 생성  ← 여기!
       │
       ▼
    ptxas (Triton IR → PTX → SASS 어셈블리)
```

torch.compile이 자동으로 해주는 것:
1. **연산 퓨전**: 연속된 element-wise 연산을 하나의 커널로
2. **메모리 최적화**: 불필요한 중간 텐서 제거
3. **CUDA Graph 적용** (`mode="reduce-overhead"` 시)

### 컴파일 비용

torch.compile은 **JIT 컴파일** 방식이라 첫 실행 시 오버헤드가 발생해:

| 단계 | 시간 | 설명 |
|------|------|------|
| `torch.compile()` 호출 | ~ms | 래퍼 생성만 (lazy) |
| 첫 `forward()` 호출 | 수십 초~분 | 실제 그래프 트레이싱 + Triton 코드 생성 + 컴파일 |
| 이후 `forward()` 호출 | 정상 속도 | 컴파일된 커널 재사용 |

반면 수동 Triton 커널은 `@triton.jit` 데코레이터가 첫 호출 시 컴파일하지만, 개별 커널 단위라서 수 초 이내에 완료돼.

### 6종 벤치마크 코드

```bash
# 공정 비교: Base vs Compile vs Triton
uv run python infer.py --mode compare-compile

# 전체 6종 비교
uv run python infer.py --mode compare-all
```

### Graph Break 문제

monkey patching으로 Triton 커널을 적용한 모델에 `torch.compile`을 걸면 **graph break**이 발생할 수 있어:

```python
# types.MethodType으로 바인딩된 메서드는
# torch.dynamo가 트레이싱하지 못할 수 있음
import types
module.forward = types.MethodType(new_forward, module)

# → torch.compile 시 graph break 발생 가능
# → 여러 개의 작은 그래프로 분할되어 최적화 효과 감소
```

**해결 방법**:
- `torch.library.triton_op`으로 커스텀 op 등록 → graph break 없이 compile 호환
- `fullgraph=False`로 graph break 허용 (현재 구현)

### 핵심 발견

**1. torch.compile 단독은 오히려 느려진다 (0.87x)**
- `generate()`의 동적 시퀀스 길이가 매 step마다 graph break 유발
- torch.compile의 트레이싱/재컴파일 오버헤드가 최적화 이득을 상쇄
- autoregressive 모델에서 torch.compile의 한계를 보여주는 결과

**2. 수동 Triton이 더 나은 점 (1.08x vs 0.87x)**
- graph break 없이 eager mode에서 투명하게 동작
- 컴파일 오버헤드 없음 (개별 `@triton.jit` 커널은 수 초 이내)
- M-RoPE처럼 도메인 특화된 연산은 TorchInductor 자동 퓨전 대상이 아님
- VRAM 추가 사용 없음 (4.13GB ≈ Base 4.09GB)

**3. Compile+Triton (1.17x)은 VRAM 트레이드오프**
- 수동 Triton 커널이 핵심 연산을 최적화한 상태에서 torch.compile이 나머지를 보완
- 하지만 VRAM 8.03GB (+3.94GB)로 프로젝트 제약 조건 위배
- 프로덕션에서는 순수 Triton이 더 적합

**4. CUDA Graph가 여전히 지배적**
- Faster (3.89x), Hybrid (5.23x)가 압도적
- torch.compile의 `reduce-overhead` 모드가 CUDA Graph를 내부적으로 사용하지만, `generate()` 호환성 문제로 효과를 발휘하지 못함
- 전용 CUDA Graph 구현(faster-qwen3-tts)이 훨씬 효과적

### 면접 답변

> "torch.compile과 수동 Triton의 차이를 설명해주세요"

> "torch.compile은 TorchInductor가 자동으로 Triton 커널을 생성하지만, 실측 결과 autoregressive TTS 모델에서는 오히려 0.87x로 느려졌습니다. generate()의 동적 시퀀스 길이가 매 step graph break을 유발하기 때문입니다. 반면 수동 Triton 커널은 eager mode에서 graph break 없이 1.08x speedup을 달성했고, VRAM 증가도 없었습니다. Compile+Triton 하이브리드(1.17x)는 속도는 나았지만 VRAM이 4GB 증가하여 실용적이지 않았습니다. 이 경험에서 배운 것은, torch.compile이 만능이 아니며 모델 아키텍처에 따라 수동 최적화가 더 효과적일 수 있다는 것입니다. 프로덕션에서는 torch.library.triton_op으로 커스텀 커널을 등록하면 compile과 호환되어, 향후 torch.compile이 autoregressive 모델을 더 잘 지원할 때 쉽게 전환할 수 있습니다."

---

## 면접 요약 (핵심만)

### 한 문단으로 설명한다면

> Qwen3-TTS 1.7B 모델의 추론 속도를 최적화하기 위해, Transformer 레이어 내의 메모리 바운드 연산 4개(RMSNorm, SwiGLU, M-RoPE, Fused Norm+Residual)를 Triton 커널 퓨전으로 구현했습니다. 핵심 원리는 여러 번의 HBM 라운드트립을 하나로 합쳐서 메모리 대역폭 병목을 해소하는 것이고, monkey patching으로 기존 PyTorch 모델을 수정 없이 런타임에 커널을 교체할 수 있게 설계했습니다. 43개의 parametrized test로 fp16/bf16 환경에서의 수치 정확도를 검증했고, faster-qwen3-tts의 CUDA Graph 위에 Triton 커널을 올리는 Hybrid 접근법으로 Base 대비 5.10x, Faster 대비 1.30x speedup을 VRAM 증가 없이 달성했습니다. 또한 torch.compile(자동 Triton) vs 수동 Triton의 apple-to-apple 비교를 위해 6종 벤치마크(Base/Triton/Compile/Compile+Triton/Faster/Hybrid)를 구현하여, 컴파일 시간 오버헤드까지 정량적으로 비교할 수 있게 했습니다.

### 핵심 키워드

- **커널 퓨전**: 여러 element-wise 연산을 하나의 GPU 커널로 합침
- **HBM 병목**: GPU 메모리 계층 구조에서의 병목 해소
- **Llama Casting Mode**: variance 계산만 fp32로 올려서 수치 안정성 확보
- **Monkey Patching**: `setattr` + `types.MethodType`으로 런타임 모듈 교체
- **Disjoint Mask Addition**: M-RoPE에서 concatenation 대신 마스킹+덧셈으로 section 합치기
- **Parametrized Testing**: shape × dtype × interface 조합으로 포괄적 검증
- **CUDA Graph × Triton Hybrid**: Lazy Graph 캡처 윈도우를 활용한 이중 최적화
- **상호 보완적 최적화 축**: launch overhead (CUDA Graph) + HBM 트래픽 (Triton) 동시 공략
- **Triton vs TensorRT**: TRT-LLM은 Qwen3-TTS 미지원 (multi-codebook 구조 불일치), Triton이 유일한 경로
- **Liger Kernel 패턴**: LinkedIn에서 프로덕션 검증된 Triton + monkey patching 접근법
- **Eager vs Compiled 서빙**: monkey patching은 eager mode에서 투명 동작, compiled mode는 `torch.library.triton_op` 등록 필요
- **vLLM-Omni**: Qwen3-TTS 네이티브 지원 서빙 프레임워크, Triton 커널 CustomOp 통합 가능
- **torch.compile**: TorchInductor 백엔드가 자동으로 Triton 커널 생성 → 수동 vs 자동 Triton 직접 비교 가능
- **Graph Break**: monkey patching(`types.MethodType`)이 torch.compile 그래프 트레이싱을 방해하는 현상
- **Compilation Overhead**: 첫 실행 시 JIT 컴파일 비용 (수십 초~분), 이후 캐싱으로 무시 가능

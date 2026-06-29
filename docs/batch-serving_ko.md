# 배치 서빙 (Batched Serving)

> **v0.3.0** | GPU: **NVIDIA RTX 5090** (Blackwell, sm_120, 32GB VRAM, CUDA 12.8)
>
> English: [batch-serving.md](batch-serving.md) (기준 문서)

배치 서빙은 한 러너가 스텝마다 여러 클립을 동시에 합성하여, 1.7B 가중치를 매 스텝
읽는 메모리 대역폭 비용을 배치 전체에 분산시킵니다. **throughput**(동시 요청) 레버이며,
단일 클립 **latency** 경로는 그대로 유지됩니다.

## 러너 이름이 아니라 capability

배치 서빙은 모든 canonical 러너의 메서드 `runner.generate_batch(texts, batch_size=B)`
로 노출됩니다. 공개 인터페이스는 v0.1.0/v0.2.0의 7-모드 축(`base`, `base+tq`, `triton`,
`triton+tq`, `faster`, `hybrid`, `hybrid+tq`)을 그대로 유지하며, `batched-*` 러너 이름은
없습니다. v0.2.0 대비 추가된 것은 `generate_batch` 하나뿐입니다.

```python
from qwen3_tts_triton import create_runner

runner = create_runner("hybrid")      # v0.2.0과 동일한 생성
runner.load_model()
out = runner.generate_batch(
    ["Hello.", "How are you today?", "Thanks!"],
    language="en", speaker="vivian", batch_size=32,
)
for clip in out["results"]:           # 원래 제출 순서 보존
    ...                               # clip = {audio, sample_rate, codec_steps, text}
```

## 두 엔진 패밀리

| 패밀리 | 러너 | 메커니즘 |
|---|---|---|
| **HF eager** | `base`, `triton` | `generate_custom_voice(text=[...])` — HuggingFace가 리스트를 left-padding 하고 per-seq EOS를 네이티브 처리하므로 **CUDA-graph fork 불필요**. `generate_batch`는 `BaseRunner`에 있고 `TritonRunner`가 상속. |
| **CUDA graph** | `faster`, `hybrid` | 배치 크기 `B`를 캡처된 CUDA 그래프에 굽습니다(`StaticCache(max_batch_size=B)`). `generate_batch`는 `FasterRunner`에 있고 `TritonFasterRunner`(hybrid)가 상속 → hybrid는 Triton 커널이 배치 그래프에 캡처되어 무료로 동작. |

4-way 표는 두 축으로 읽습니다. **엔진 패밀리**(HF eager vs CUDA graph)는 서빙
throughput 리더보드로, 디코드 엔진이 다른 것이지 배치만의 격리가 아닙니다. **같은 패밀리
델타**(`base`→`triton`, `faster`→`hybrid`)가 배치 하에서의 Triton 커널 기여분이며, 배치가
커질수록 스텝의 elementwise 비중이 줄어 작아집니다.

## 생성 머신

- **Length bucketing** — 가변 길이 입력을 길이로 정렬해 `≤ batch_size` 버킷으로 묶어,
  짧은 클립이 같은 배치의 긴 클립을 기다리며 idle 하지 않게 합니다. 토큰 단위 continuous
  batching은 여기서 도움이 안 됩니다(`B`가 CUDA 그래프의 정적 shape에 구워져 있어, 끝난
  row를 evict 해도 그래프 compute가 줄지 않음). bucketing이 정답입니다(greedy, 출력 불변).
- **Per-row 배치 샘플링** — row별 독립 stochastic 샘플링 + *per-sequence* repetition
  penalty(row가 갈라지면 단일 flat history는 틀림).
- **Per-sequence EOS** — 각 row가 자기 EOS에서 멈추고, 모든 row가 끝나거나
  `max_new_tokens`에 도달하면 배치 종료.

## 핵심 수치 (RTX 5090)

아래 모든 값은 커밋된 두 벤치(`make bench-batched-matrix`, `make bench-batched`)로
재현됩니다. 아티팩트는 `benchmark/results/batched_matrix.json`,
`benchmark/results/batched_runner.json`에 있습니다.

| 지표 | 값 |
|---|---|
| CUDA-graph throughput (B=16, ms/step ↓) | hybrid **35.2** · faster 40.1 |
| HF-eager throughput (B=16, ms/step ↓) | base 119.6 · triton 126.5 |
| Triton-in-graph 이득 (`faster`→`hybrid`) | 40.1 → 35.2 ms/step (**1.14×**) |
| Length bucketing (B=8) | RTF 9.4 → 10.2 (**1.09× wall**), 출력 불변 |
| Per-row 샘플링 | 동일 프롬프트 → row별 서로 다른 길이 (stochastic) |
| Tier 3 배치 패리티 | UTMOS / CER / speaker-sim 이 단일 클립과 분포 동등 (Mann-Whitney PASS) |

`ms_per_step = wall / max(codec_steps)`가 공정한 엔진 간 속도 지표입니다. RTF(= audio /
wall)는 greedy 디코드가 엔진마다 다른 길이로 갈라지므로 길이에 confound 됩니다. Triton
커널은 **CUDA 그래프에 캡처될 때만**(`faster`→`hybrid`, 1.14×) 이득을 냅니다. HF-eager
경로(`base`→`triton`)에서는 배치 시 스텝당 커널 런치 오버헤드가 미미한 elementwise
이득을 상쇄해 둘이 사실상 break-even입니다(HF-eager greedy는 노이즈가 크고 B=16에서
elementwise 비중이 무시할 수준).

## VRAM & per-sample 효율

`generate_batch`는 **total** peak VRAM을 **per-sample** 효율과 맞바꿉니다. 단일 클립
생성은 클립당 ~4 GB로 피크합니다(`e2e_benchmarks.json`). `B=16` 배치는 16개 시퀀스를
보유하므로 **total** 피크는 올라가지만, **per-sample** VRAM과 wall 시간은 크게
떨어집니다. 이것이 서빙 이점입니다 — 프로세스당이 아니라 *동시 요청당* VRAM.

| 러너 | ms/step | per-sample wall | total VRAM (B=16) | **per-sample VRAM** |
|---|---|---|---|---|
| base | 119.6 | 1.20 s | 10.05 GB | **0.63 GB** |
| triton | 126.5 | 1.36 s | 10.49 GB | **0.66 GB** |
| faster | 40.1 | 0.64 s | 8.08 GB | **0.51 GB** |
| hybrid | 35.2 | 0.36 s | 7.89 GB | **0.49 GB** |

단일 클립 ~4 GB/클립 대비 per-sample VRAM이 **~6–10×** 감소하고(hybrid 0.49 GB),
CUDA-graph 경로에서는 per-sample wall이 1초 미만으로 유지됩니다.
`make bench-batched-matrix`(`batched_matrix.json`)로 재현됩니다.

단일 클립 base latency(~5.0초/클립, README의 E2E 표 기준) 대비 hybrid의 0.36초
per-sample wall은 B=16에서 **per-sample throughput ~14×** 입니다 — ~5× 단일
클립(Hybrid vs Base)과 ~3× 배치 분산의 누적. 이는 단일 클립 latency가 아니라
throughput(동시 요청) 수치입니다.

## 실행

```bash
make bench-batched           # bucketing + per-row 샘플링 검증 (hybrid)
make bench-batched-matrix    # 4-way base/triton/faster/hybrid 표
```

결과는 `benchmark/results/batched_matrix.json`과 `benchmark/results/batched_runner.json`
에 기록됩니다(다른 모든 벤치 아티팩트와 같은 디렉토리). Streamlit 대시보드(`make ui`)는
Benchmarks 탭에서 4-way 표를 렌더링합니다.

## 품질 패리티 (Tier 3)

배치 생성은 단일 클립 생성과 품질이 동등합니다. `--batch-size` 플래그로 Tier 3 평가기를
배치 경로로 구동합니다(모든 러너에서 동작):

```bash
uv run python benchmark/eval_quality.py --mode full --runners hybrid --batch-size 32
```

full 모드 평가(3-run + Mann-Whitney)는 4개 canonical 러너 모두에서 단일 클립 vs 배치
생성의 UTMOS / CER / speaker-sim 분포 동등성을 확인합니다. 커밋된 실행 —
`benchmark/results/tier3_batched_full_multi.json`(base 기준 vs
`triton` / `faster` / `hybrid`, 36문장 × 3-run, **PASS**) — 은 CER / UTMOS가 단일
클립 full 매트릭스의 stochastic 노이즈 범위 안입니다(hybrid CER 0.042 vs 0.042;
faster 0.037 vs 0.039).

> **알려진 한계** — HF-eager 배치 경로에서 `+tq`(TurboQuant)(`base+tq` /
> `triton+tq`)는 현재 미지원입니다(KV-cache 마스크 off-by-one). 따라서 배치 증거는
> 4개 base 엔진 패밀리를 다룹니다. 단일 클립 `+tq`는 영향 없으며, 수정은 추후로
> 미룹니다.

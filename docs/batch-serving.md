# Batched Serving

> **v0.3.0** | GPU: **NVIDIA RTX 5090** (Blackwell, sm_120, 32GB VRAM, CUDA 12.8)
>
> 한국어: [batch-serving_ko.md](batch-serving_ko.md)

Batched serving lets one runner synthesise many clips per step, amortising the
memory-bandwidth cost of reading the 1.7B weights across the batch. It is the
**throughput** lever (concurrent requests); the single-clip **latency** path is
unchanged.

## A capability, not a new runner

Batched serving is exposed as a method — `runner.generate_batch(texts, batch_size=B)`
— on **every** canonical runner. The public interface stays the v0.1.0/v0.2.0
seven-mode axis (`base`, `base+tq`, `triton`, `triton+tq`, `faster`, `hybrid`,
`hybrid+tq`); there is no `batched-*` runner name. `generate_batch` is the only
addition over v0.2.0.

```python
from qwen3_tts_triton import create_runner

runner = create_runner("hybrid")      # same construction as v0.2.0
runner.load_model()
out = runner.generate_batch(
    ["Hello.", "How are you today?", "Thanks!"],
    language="en", speaker="vivian", batch_size=32,
)
for clip in out["results"]:           # original submission order preserved
    ...                               # clip = {audio, sample_rate, codec_steps, text}
```

## Two engine families

| Family | Runners | Mechanism |
|---|---|---|
| **HF eager** | `base`, `triton` | `generate_custom_voice(text=[...])` — HuggingFace left-pads the list and applies per-sequence EOS natively, so **no CUDA-graph fork** is needed. `generate_batch` lives on `BaseRunner` and is inherited by `TritonRunner`. |
| **CUDA graph** | `faster`, `hybrid` | The batch size `B` is baked into a captured CUDA graph (`StaticCache(max_batch_size=B)`). `generate_batch` lives on `FasterRunner` and is inherited by `TritonFasterRunner` (hybrid), so the hybrid path captures the Triton kernels into the batched graph for free. |

Read a 4-way table on two axes: the **engine family** (HF eager vs CUDA graph)
is a serving-throughput leaderboard — different decode engines, not an isolation
of batching alone; the **same-family delta** (`base`→`triton`, `faster`→`hybrid`)
is the Triton-kernel contribution under batch (it shrinks as batch grows because
the elementwise share of the step falls).

## Generation machinery

- **Length bucketing** — variable-length inputs are sorted by length and chunked
  into `≤ batch_size` buckets, so short clips don't idle while a long clip in the
  same batch keeps decoding. Per-token continuous batching does not help here:
  `B` is baked into the CUDA graph's static shapes, so evicting a finished row
  frees no graph compute. Bucketing is the right lever (greedy, output-invariant).
- **Per-row batch sampling** — independent stochastic sampling per row plus a
  *per-sequence* repetition penalty (a single flat history is wrong once rows
  diverge).
- **Per-sequence EOS** — each row stops at its own EOS; the batch ends when all
  rows finish or `max_new_tokens` is hit.

## Key numbers (RTX 5090)

Every value below is reproduced by the two committed benches
(`make bench-batched-matrix`, `make bench-batched`); the artifacts live in
`benchmark/results/batched_matrix.json` and `benchmark/results/batched_runner.json`.

| Metric | Value |
|---|---|
| CUDA-graph throughput (B=16, ms/step ↓) | hybrid **35.2** · faster 40.1 |
| HF-eager throughput (B=16, ms/step ↓) | base 119.6 · triton 126.5 |
| Triton-in-graph gain (`faster`→`hybrid`) | 40.1 → 35.2 ms/step (**1.14×**) |
| Length bucketing (B=8) | RTF 9.4 → 10.2 (**1.09× wall**), output-invariant |
| Per-row sampling | identical prompts → distinct per-row lengths (stochastic) |
| Tier 3 batched parity | UTMOS / CER / speaker-sim distribution-equivalent vs single-clip (Mann-Whitney PASS) |

`ms_per_step = wall / max(codec_steps)` is the fair cross-engine speed metric;
RTF (= audio / wall) is length-confounded because greedy decode diverges into
different sequence lengths across engines. Triton kernels pay off only once
**captured into a CUDA graph** (`faster`→`hybrid`, 1.14×): in the HF-eager path
(`base`→`triton`) per-step kernel-launch overhead at batch offsets the tiny
elementwise saving, so the two are effectively break-even (HF-eager greedy is
noisy and the elementwise share is negligible at B=16).

## Running it

```bash
make bench-batched           # bucketing + per-row sampling validation (hybrid)
make bench-batched-matrix    # 4-way base/triton/faster/hybrid table
```

Results are written to `benchmark/results/batched_matrix.json` and
`benchmark/results/batched_runner.json` (same directory as every other
benchmark artifact). The Streamlit dashboard (`make ui`) renders the 4-way table
under the Benchmarks tab.

## Quality parity (Tier 3)

Batched generation is quality-equivalent to single-clip generation. Drive the
Tier 3 evaluator through the batched path with the `--batch-size` flag (works for
any runner):

```bash
uv run python benchmark/eval_quality.py --mode full --runners hybrid --batch-size 32
```

Full-mode evaluation (3 runs + Mann-Whitney) confirms UTMOS / CER / speaker-sim
distribution equivalence between single-clip and batched generation for all four
canonical runners.

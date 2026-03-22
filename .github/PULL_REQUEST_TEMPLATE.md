## Summary

<!-- What does this PR do? Keep it to 1-3 sentences. -->

## Type of Change

<!-- Check all that apply. -->

- [ ] `feat`: New feature
- [ ] `fix`: Bug fix
- [ ] `perf`: Performance improvement
- [ ] `refactor`: Code refactoring (no functional change)
- [ ] `test`: Adding or updating tests
- [ ] `docs`: Documentation update
- [ ] `chore`: Maintenance (CI, dependencies, build)

## Changes

<!-- Bulleted list of specific changes. -->

-

## GPU Compatibility

<!-- IMPORTANT: This project uses Triton GPU kernels. Please specify which GPU(s) you tested on. -->

| Item | Value |
|------|-------|
| GPU tested on | <!-- e.g., RTX 5090 (Blackwell, sm_120) --> |
| CUDA version | <!-- e.g., 12.8 --> |
| PyTorch version | <!-- e.g., 2.8.0.dev+cu128 --> |
| Triton version | <!-- e.g., 3.3.1 --> |

<!-- If this PR adds/modifies Triton kernels, check the applicable SM architectures: -->

- [ ] Blackwell (sm_120) — RTX 5090/5080, B100/B200
- [ ] Hopper (sm_90) — H100/H200
- [ ] Ampere (sm_80/86) — A100/RTX 3090
- [ ] Ada Lovelace (sm_89) — RTX 4090/4080
- [ ] Architecture-independent (no kernel changes)

## Test Plan

<!-- How was this tested? Check all that apply. -->

- [ ] `make test` — Tier 1 kernel tests pass
- [ ] `make test-parity` — Tier 2 model parity tests pass (GPU required)
- [ ] `make eval-fast` — Tier 3 E2E quality evaluation pass (GPU required)
- [ ] `make lint` — Ruff lint passes
- [ ] `make typecheck` — Ty type check passes
- [ ] Manual testing (describe below)

<!-- If manual testing, describe what you tested: -->

## Performance Impact

<!-- If this PR affects inference performance, provide before/after numbers. -->
<!-- Use `make bench-kernels` or `make bench-e2e` to measure. -->
<!-- Delete this section if not applicable. -->

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Kernel latency (ms) | | | |
| E2E latency (ms) | | | |
| Peak VRAM (MB) | | | |

## Checklist

- [ ] Code follows project style guidelines (Ruff format + lint, `make check`)
- [ ] Tests added/updated for changes
- [ ] Documentation updated (if applicable)
- [ ] No hardcoded secrets or credentials
- [ ] Commit messages follow [conventional format](../.gitmessage.txt)

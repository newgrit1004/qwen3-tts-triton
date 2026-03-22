"""Evaluation configuration and test sentences for TTS quality assessment.

Defines distribution-level comparison thresholds (industry-standard),
ASR model settings, and test sentences for zh/ko languages.

Methodology: Independent benchmark comparison (vLLM/TensorRT-LLM pattern).
Each model generates independently, then distribution-level metrics are compared.
No pair-level waveform comparison (PESQ/STOI/MCD) — inappropriate for stochastic TTS.
"""

# --- Evaluation configuration ---
from typing import Any

EVAL_CONFIG: dict[str, Any] = {
    "warmup_runs": 3,
    "languages": ["zh", "ko"],
    # --- Tier 3: Quality Distribution (independent benchmark comparison) ---
    "tier3": {
        # Generation settings
        "runs_per_sentence_fast": 1,  # fast mode (CI)
        "runs_per_sentence_full": 3,  # full mode (F5-TTS style)
        # Per-metric thresholds (distribution-level comparison)
        "utmos_delta_max": 0.3,  # |mean(base) - mean(triton)| < 0.3
        "utmos_floor": 2.5,  # both models must score >= 2.5
        "cer_delta_max": 0.05,  # |mean(base) - mean(triton)| < 5%p
        "speaker_sim_min": 0.75,  # independent generation speaker sim
        # Statistical test (full mode only)
        "mann_whitney_alpha": 0.05,
        # ASR model sizes
        "asr_model_fast": "small",  # CI (~5 min)
        "asr_model_full": "large-v3",  # PR gate (~30 min)
    },
}

# --- Test sentences (language-stratified) ---
# Distribution: short (<50 chars) / medium (50-150) / long (>150)

EVAL_SENTENCES: dict[str, list[dict[str, str]]] = {
    "zh": [
        # Short (<50 chars)
        {"text": "你好，今天天气真好。", "language": "zh"},
        {"text": "欢迎使用语音合成系统。", "language": "zh"},
        {"text": "请问您需要什么帮助？", "language": "zh"},
        {"text": "谢谢你的支持。", "language": "zh"},
        {"text": "明天见，再见。", "language": "zh"},
        # Medium (50-150 chars)
        {
            "text": "人工智能技术正在快速发展，语音合成领域也取得了显著的进步。",
            "language": "zh",
        },
        {
            "text": "深度学习模型能够生成非常自然的语音，几乎与真人无法区分。",
            "language": "zh",
        },
        {
            "text": "这个系统使用了最新的Transformer架构，支持多语言语音合成。",
            "language": "zh",
        },
        {
            "text": "我们的目标是让每个人都能享受到高质量的语音合成服务。",
            "language": "zh",
        },
        {
            "text": "通过优化推理速度，我们可以实现实时语音生成。",
            "language": "zh",
        },
        # Long (>150 chars)
        {
            "text": (
                "随着大语言模型的快速发展，语音合成技术也迎来了新的突破。"
                "基于Transformer架构的端到端模型，不仅能够生成高保真的语音，"
                "还能够捕捉说话人的情感和语调变化，使合成语音更加自然流畅。"
            ),
            "language": "zh",
        },
        {
            "text": (
                "在实际应用中，语音合成系统需要在保证质量的同时，"
                "尽可能降低计算延迟。通过使用GPU加速和算子融合等优化技术，"
                "我们可以在不牺牲音质的前提下，大幅提升推理速度。"
            ),
            "language": "zh",
        },
    ],
    "ko": [
        # Short (<50 chars)
        {"text": "안녕하세요, 반갑습니다.", "language": "ko"},
        {"text": "오늘 날씨가 정말 좋네요.", "language": "ko"},
        {"text": "감사합니다, 좋은 하루 되세요.", "language": "ko"},
        {"text": "음성 합성 테스트입니다.", "language": "ko"},
        {"text": "다음에 또 만나요.", "language": "ko"},
        # Medium (50-150 chars)
        {
            "text": "인공지능 기술의 발전으로 음성 합성의 품질이 크게 향상되었습니다.",
            "language": "ko",
        },
        {
            "text": (
                "이 시스템은 최신 트랜스포머 아키텍처를 사용하여 "
                "자연스러운 음성을 생성합니다."
            ),
            "language": "ko",
        },
        {
            "text": (
                "딥러닝 모델을 활용한 음성 합성은 "
                "실시간 처리가 가능한 수준에 도달했습니다."
            ),
            "language": "ko",
        },
        {
            "text": "GPU 커널 최적화를 통해 추론 속도를 획기적으로 개선할 수 있습니다.",
            "language": "ko",
        },
        {
            "text": (
                "다국어 지원을 통해 한국어와 중국어 모두 높은 품질의 음성을 생성합니다."
            ),
            "language": "ko",
        },
        # Long (>150 chars)
        {
            "text": (
                "최근 대규모 언어 모델의 발전과 함께 음성 합성 기술도 "
                "눈부신 성장을 이루고 있습니다. 트랜스포머 기반의 "
                "엔드투엔드 모델은 고품질의 음성을 생성할 뿐만 아니라, "
                "화자의 감정과 억양까지 자연스럽게 표현할 수 있습니다."
            ),
            "language": "ko",
        },
        {
            "text": (
                "트리톤 커널 퓨전을 통한 최적화는 추론 속도를 크게 향상시킵니다. "
                "RMSNorm과 잔차 연결을 하나의 커널로 합치면 메모리 접근을 줄이고, "
                "SwiGLU 활성화 함수의 퓨전은 중간 텐서 할당을 제거합니다."
            ),
            "language": "ko",
        },
    ],
}

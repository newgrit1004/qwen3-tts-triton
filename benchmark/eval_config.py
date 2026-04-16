"""Evaluation configuration and test sentences for TTS quality assessment.

Defines distribution-level comparison thresholds (industry-standard),
ASR model settings, test sentences for zh/ko/en languages, and tongue
twister stress-test sentences for pronunciation robustness evaluation.

Methodology: Independent benchmark comparison (vLLM/TensorRT-LLM pattern).
Each model generates independently, then distribution-level metrics are compared.
No pair-level waveform comparison (PESQ/STOI/MCD) — inappropriate for stochastic TTS.
"""

# --- Evaluation configuration ---
from typing import Any

EVAL_CONFIG: dict[str, Any] = {
    "warmup_runs": 3,
    "languages": ["zh", "ko", "en"],
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
        "asr_model": "CohereLabs/cohere-transcribe-03-2026",
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
    "en": [
        # Short (<50 chars)
        {"text": "Hello, how are you today?", "language": "en"},
        {"text": "Welcome to the speech synthesis demo.", "language": "en"},
        {"text": "Thank you for your support.", "language": "en"},
        {"text": "See you tomorrow, goodbye.", "language": "en"},
        {"text": "This is a voice test.", "language": "en"},
        # Medium (50-150 chars)
        {
            "text": (
                "Artificial intelligence is transforming the way "
                "we interact with technology every day."
            ),
            "language": "en",
        },
        {
            "text": (
                "Deep learning models can generate incredibly natural speech "
                "that is almost indistinguishable from real humans."
            ),
            "language": "en",
        },
        {
            "text": (
                "This system uses the latest Transformer architecture "
                "to support multilingual speech synthesis."
            ),
            "language": "en",
        },
        {
            "text": (
                "Our goal is to make high quality speech synthesis "
                "accessible to everyone around the world."
            ),
            "language": "en",
        },
        {
            "text": (
                "By optimizing inference speed with GPU kernels, "
                "we can achieve real-time voice generation."
            ),
            "language": "en",
        },
        # Long (>150 chars)
        {
            "text": (
                "With the rapid advancement of large language models, "
                "speech synthesis technology has also reached new heights. "
                "End-to-end models based on Transformer architecture can "
                "generate high-fidelity speech while capturing the speaker's "
                "emotions and intonation naturally."
            ),
            "language": "en",
        },
        {
            "text": (
                "In practical applications, speech synthesis systems need to "
                "minimize computational latency while maintaining quality. "
                "Through GPU acceleration and operator fusion techniques, "
                "we can significantly improve inference speed without "
                "sacrificing audio quality."
            ),
            "language": "en",
        },
    ],
}

# --- Tongue twister sentences (pronunciation stress test) ---
# References: Seed-TTS hard set (ByteDance 2024), MaskGCT (ICLR 2025),
# EmergentTTS-Eval (NeurIPS 2025). 15 sentences per language.

TONGUE_TWISTER_SENTENCES: dict[str, list[dict[str, str]]] = {
    "zh": [
        # Tonal minimal pairs (si/shi distinction)
        {"text": "四是四，十是十，十四是十四，四十是四十。", "language": "zh"},
        # Repeated syllable clusters
        {
            "text": "吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。",
            "language": "zh",
        },
        # Plosive initials (b/p/b/p)
        {"text": "八百标兵奔北坡，炮兵并排北边跑。", "language": "zh"},
        # h/f/h alternation
        {
            "text": "黑化肥发灰会挥发，灰化肥挥发会发黑。",
            "language": "zh",
        },
        # l/lv/lv minimal trio
        {"text": "红鲤鱼与绿鲤鱼与驴。", "language": "zh"},
        # n/l nasal-lateral contrast
        {"text": "牛郎恋刘娘，刘娘念牛郎。", "language": "zh"},
        # b/d/ch length contrast
        {
            "text": "扁担长板凳宽，板凳没有扁担长，扁担没有板凳宽。",
            "language": "zh",
        },
        # g/k/g velar clusters
        {"text": "哥哥挎筐过宽沟，赶快过沟看怪狗。", "language": "zh"},
        # q/x/b/d rhythm pattern
        {
            "text": "天上七颗星，地上七块冰，树上七只鹰。",
            "language": "zh",
        },
        # f/h/f/h with tonal variation
        {
            "text": "粉红墙上画凤凰，凤凰画在粉红墙。",
            "language": "zh",
        },
        # All shi syllables (extreme tonal stress)
        {"text": "石室诗士施氏嗜狮誓食十狮。", "language": "zh"},
        # zhi/bu repetition
        {
            "text": "知之为知之，不知为不知，是知也。",
            "language": "zh",
        },
        # d/dao/dao tonal cascade
        {"text": "短刀断稻倒岛道。", "language": "zh"},
        # hu/zhu/lu animal enumeration
        {"text": "初入江湖，一日遇一虎一猪一鹿。", "language": "zh"},
        # h/f compound (variant of #4)
        {"text": "化肥会挥发，黑化肥发灰，灰化肥发黑。", "language": "zh"},
    ],
    "ko": [
        # ㄱ/ㄲ/ㅋ velar triple + ㅈ/ㅊ affricates
        {
            "text": (
                "간장 공장 공장장은 강 공장장이고 된장 공장 공장장은 공 공장장이다."
            ),
            "language": "ko",
        },
        # ㅊ/ㅅ/ㅆ sibilant + ㄹ lateral
        {
            "text": ("경찰청 철창살은 외철창살이고 검찰청 철창살은 쌍철창살이다."),
            "language": "ko",
        },
        # ㄲ/ㄱ tense-lax alternation
        {
            "text": "저기 저 콩깍지가 깐 콩깍지인가 안 깐 콩깍지인가.",
            "language": "ko",
        },
        # ㄱ/ㄹ/ㄴ nasal-liquid with vowel contrast
        {
            "text": (
                "내가 그린 기린 그림은 긴 기린 그림이고 "
                "네가 그린 기린 그림은 안 긴 기린 그림이다."
            ),
            "language": "ko",
        },
        # ㅊ/ㅋ aspirate cluster
        {
            "text": "칠월 칠석은 평창 친구 친정 칠촌 칠순 잔치 날.",
            "language": "ko",
        },
        # ㄱ/ㅂ/ㄱ plosive cluster
        {
            "text": ("고려고 교복은 고급 교복이고 고려고 교복은 고급 원단 교복이다."),
            "language": "ko",
        },
        # ㅂ/ㅃ/ㄲ tense consonant mix
        {
            "text": (
                "상표 붙인 큰 깡통은 된장 깡통이고 "
                "상표 안 붙인 큰 깡통은 간장 깡통이다."
            ),
            "language": "ko",
        },
        # ㄲ/ㄱ variant of #3
        {
            "text": "들의 콩깍지는 깐 콩깍지인가 안 깐 콩깍지인가.",
            "language": "ko",
        },
        # ㅅ/ㅆ/ㅊ sibilant-affricate
        {"text": "신진 샹숑 가수의 신춘 샹숑 쇼.", "language": "ko"},
        # ㄷ/ㅍ/ㅎ aspirate + plosive
        {
            "text": "청단풍잎 홍단풍잎 흑단풍잎 백단풍잎.",
            "language": "ko",
        },
        # ㅈ/ㅊ/ㅆ tense sibilant cluster
        {"text": "생쥐 철쥐 쇠철쥐 쌍철쥐.", "language": "ko"},
        # ㅎ/ㅋ/ㄱ aspirate-velar
        {
            "text": "서울특별시 특허허가과 허가과장 허과장.",
            "language": "ko",
        },
        # ㅈ/ㅊ/ㅅ affricate-sibilant
        {
            "text": "정책 실장은 정 실장이고 정치 실장도 정 실장이다.",
            "language": "ko",
        },
        # ㅇ/ㅈ/ㅇ vowel-onset repetition
        {
            "text": ("한양양장점 옆 한영양장점 한영양장점 옆은 한양양장점."),
            "language": "ko",
        },
        # ㄲ/ㅌ/ㅁ plosive-nasal mix
        {"text": "깐 토마토가 안 깐 토마토보다 낫다.", "language": "ko"},
    ],
    "en": [
        # Sibilant /s/sh/ stress
        {"text": "She sells seashells by the seashore.", "language": "en"},
        # Plosive /p/ cluster
        {
            "text": "Peter Piper picked a peck of pickled peppers.",
            "language": "en",
        },
        # /w/ch/ repetition
        {
            "text": (
                "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"
            ),
            "language": "en",
        },
        # /r/l/ liquid contrast
        {
            "text": "Red lorry, yellow lorry, red lorry, yellow lorry.",
            "language": "en",
        },
        # /s/ks/ consonant cluster
        {
            "text": "The sixth sick sheikh's sixth sheep's sick.",
            "language": "en",
        },
        # /n/y/k/ rhythm
        {
            "text": "Unique New York, you know you need unique New York.",
            "language": "en",
        },
        # /t/b/ minimal pair repetition
        {
            "text": "Toy boat, toy boat, toy boat, toy boat.",
            "language": "en",
        },
        # /r/s/w/ cluster
        {"text": "Irish wristwatch, Swiss wristwatch.", "language": "en"},
        # /w/ch/ alliteration
        {
            "text": "Which witch wished which wicked wish?",
            "language": "en",
        },
        # /p/k/ plosive
        {"text": "A proper copper coffee pot.", "language": "en"},
        # /f/d/b/ voiced-voiceless
        {
            "text": "Fred fed Ted bread and Ted fed Fred bread.",
            "language": "en",
        },
        # /s/l/ liquid-sibilant
        {
            "text": "Six slippery snails slid slowly seaward.",
            "language": "en",
        },
        # /th/ fricative cluster
        {
            "text": (
                "The thirty-three thieves thought that they "
                "thrilled the throne throughout Thursday."
            ),
            "language": "en",
        },
        # /k/n/ repetition
        {
            "text": "Can you can a can as a canner can can a can?",
            "language": "en",
        },
        # /b/t/ plosive alternation
        {
            "text": (
                "Betty Botter bought some butter but she said the butter's bitter."
            ),
            "language": "en",
        },
    ],
}

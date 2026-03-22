# Audio Samples

Pre-generated audio samples for comparing inference modes.

## Generation Parameters

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-TTS-12Hz-1.7B-CustomVoice |
| Speaker | sohee |
| Sample Rate | 24 kHz |
| Format | WAV (16-bit PCM, mono) |
| GPU | NVIDIA RTX 5090 (Blackwell, sm_120) |
| Dtype | bf16 |

## Directory Structure

```
assets/audio_samples/
├── metadata.json
├── README.md
├── base/                       # PyTorch baseline
│   ├── ko_01.wav ... ko_05.wav       # custom voice (Korean)
│   ├── en_01.wav ... en_05.wav       # custom voice (English)
│   ├── clone_ko_01.wav ...           # voice cloning (Korean)
│   └── clone_en_01.wav ...           # voice cloning (English)
├── triton/                     # Triton kernel fusion
├── faster/                     # faster-qwen3-tts (CUDA Graph)
└── hybrid/                     # Faster + Triton
```

## Generation

```bash
# Generate all samples (GPU required)
make generate-samples

# Generate specific modes only
uv run python scripts/generate_samples.py --modes base triton

# Skip voice cloning samples
uv run python scripts/generate_samples.py --skip-clone

# Custom speaker
uv run python scripts/generate_samples.py --speaker vivian
```

## Voice Cloning Reference

Voice cloning samples use the bundled LJSpeech reference audio (Public Domain):
- **File**: `assets/reference_audio/ljspeech_sample.wav`
- **Transcript**: "so subversive of meditation, so disturbing to the thoughts;"

## Sample Styles

Each mode generates 5 Korean + 5 English custom voice samples covering:
conversation, news, technical, emotional, descriptive.

Plus voice cloning samples (2 Korean + 2 English) per mode where supported.

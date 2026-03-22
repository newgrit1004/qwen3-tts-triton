# Reference Audio for Voice Cloning

## Included Files

| File | Source | License | Duration | Sample Rate |
|------|--------|---------|----------|-------------|
| `ljspeech_sample.wav` | [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/) (LJ007-0048) | Public Domain | ~4s | 22050 Hz |

## Transcript

> "so subversive of meditation, so disturbing to the thoughts;"

## Usage

### Python

```python
from qwen3_tts_triton.models.patching import apply_triton_kernels

wavs, sr = model.generate_voice_clone(
    text="Hello, this is a voice cloning test.",
    ref_audio_path="assets/reference_audio/ljspeech_sample.wav",
    ref_text="so subversive of meditation, so disturbing to the thoughts;",
)
```

## License

The LJSpeech dataset is in the **Public Domain**. No attribution required.
See: <https://keithito.com/LJ-Speech-Dataset/>

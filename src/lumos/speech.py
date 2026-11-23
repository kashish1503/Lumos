"""Speech-to-text with Whisper."""

import torch
from transformers import pipeline

ASR_ID = "openai/whisper-large-v3-turbo"


def load_asr(device=None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if "cuda" in str(device) else torch.float32
    return pipeline("automatic-speech-recognition", model=ASR_ID,
                    torch_dtype=dtype, device=device)


def transcribe(asr, audio_path, language="en"):
    if not audio_path:
        return ""
    kwargs = {"task": "transcribe"}
    if language:
        kwargs["language"] = language
    return asr(audio_path, generate_kwargs=kwargs)["text"].strip()

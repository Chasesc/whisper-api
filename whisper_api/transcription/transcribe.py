from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import whisper
from whisper_api.transcription.schemas import TranscriptionResult
from whisper_api.transcription.utils import ensure_filepath

if TYPE_CHECKING:
    from torch import device


@cache
def _load_model(model_name: str, device: device | str | None) -> whisper.Whisper:
    return whisper.load_model(model_name, device=device)


def preload_model(model_name: str, device: device | str | None = None) -> None:
    # Fill the cache so this work doesn't need to happen during the API call. Not required.
    _load_model(model_name, device)


def transcribe(
    audio: str | bytes,
    *,
    model_name: str = "base",
    device: device | str | None = None,
    word_timestamps: bool = False,
) -> TranscriptionResult:
    model = _load_model(model_name, device)

    with ensure_filepath(audio) as audio_file:
        transcription_input = whisper.load_audio(audio_file)

    return TranscriptionResult(
        model.transcribe(transcription_input, word_timestamps=word_timestamps)
    )

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, Final

import whisper

from whisper_api.transcription.schemas import TranscriptionResult
from whisper_api.transcription.utils import ensure_filepath

if TYPE_CHECKING:
    from numpy import floating, ndarray
    from torch import device

DEFAULT_MODEL: Final[str] = "base"


@cache
def _load_model(model_size: str, device: device | str | None) -> whisper.Whisper:
    return whisper.load_model(model_size, device=device)


def preload_model(
    model_size: str = DEFAULT_MODEL, device: device | str | None = None
) -> None:
    # Fill the cache so this work doesn't need to happen during the API call. Not required.
    _load_model(model_size, device)


def to_audio_buffer(audio: str | bytes) -> ndarray[floating[Any]]:
    with ensure_filepath(audio) as audio_file:
        return whisper.load_audio(audio_file)


def transcribe(
    audio_buffer: ndarray[floating[Any]],
    *,
    model_size: str = DEFAULT_MODEL,
    device: device | str | None = None,
    word_timestamps: bool = False,
) -> TranscriptionResult:
    model = _load_model(model_size, device)

    result_dict = model.transcribe(audio_buffer, word_timestamps=word_timestamps)
    return TranscriptionResult(
        text=result_dict["text"],
        segments=result_dict["segments"],
        language=result_dict["language"],
        model_size=model_size,
    )

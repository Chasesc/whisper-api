from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Final

import torch
import whisper

from whisper_api.transcription.schemas import TranscriptionResult
from whisper_api.transcription.utils import ensure_filepath

if TYPE_CHECKING:
    from numpy import ndarray

DEFAULT_MODEL: Final[str] = "base"


def _find_best_device() -> torch.device | None:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # https://github.com/openai/whisper/pull/382 - MPS is not fully supported by pytorch yet.
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     return torch.device("mps")

    return None


DEFAULT_DEVICE: torch.device | None = _find_best_device()


@cache
def _load_model(
    model_size: str, device: torch.device | str | None = DEFAULT_DEVICE
) -> whisper.Whisper:
    return whisper.load_model(model_size, device=device)


def preload_model(
    model_size: str = DEFAULT_MODEL, device: torch.device | str | None = DEFAULT_DEVICE
) -> None:
    # Fill the cache so this work doesn't need to happen during the API call. Not required.
    _load_model(model_size, device)


def to_audio_buffer(audio: str | bytes) -> ndarray:
    with ensure_filepath(audio) as audio_file:
        return whisper.load_audio(audio_file)


def transcribe(
    audio_buffer: ndarray,
    *,
    model_size: str = DEFAULT_MODEL,
    device: torch.device | str | None = DEFAULT_DEVICE,
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

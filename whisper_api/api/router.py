from typing import Annotated

from fastapi import APIRouter, File, HTTPException, status

from whisper_api.api.schemas import ModelSize, TranscriptionResult
from whisper_api.transcription import transcribe

router = APIRouter(prefix="/whisper", tags=["Whisper"])


@router.post("/transcribe", response_model=TranscriptionResult)
def transcribe_route(
    file: Annotated[bytes, File()],
    model_size: ModelSize = ModelSize.BASE,
    word_timestamps: bool = False,
) -> TranscriptionResult:
    try:
        audio_buffer = transcribe.to_audio_buffer(file)
    except RuntimeError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid audio file.")
    return transcribe.transcribe(
        audio_buffer, model_size=model_size.value, word_timestamps=word_timestamps
    )

import whisper

from whisper_api.transcription.schemas import TranscriptionResult
from whisper_api.transcription.utils import ensure_filepath

model = whisper.load_model("base")


def transcribe(audio: str | bytes) -> TranscriptionResult:
    with ensure_filepath(audio) as audio_file:
        transcription_input = whisper.load_audio(audio_file)

    return TranscriptionResult(model.transcribe(transcription_input))

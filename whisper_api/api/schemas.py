from enum import Enum, unique

from whisper_api.transcription.schemas import TranscriptionResult  # noqa


@unique
class ModelSize(Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

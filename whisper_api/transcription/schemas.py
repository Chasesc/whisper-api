from pydantic import BaseModel, Field


class TranscriptionWordResult(BaseModel):
    word: str
    start: float
    end: float
    probability: float


class TranscriptionSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speed_prop: float | None = None
    words: list[TranscriptionWordResult] = Field(
        default_factory=list,
        description="Per word timings. Only returned when word_timestamps is true.",
    )


class TranscriptionResult(BaseModel):
    text: str
    segments: list[TranscriptionSegment]
    language: str
    model_size: str

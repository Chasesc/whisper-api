from typing import TypedDict


class TranscriptionSegment(TypedDict):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speed_prop: float


class TranscriptionResult(TypedDict):
    text: str
    segments: list[TranscriptionSegment]
    language: str

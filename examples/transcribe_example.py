import sys
from io import BytesIO
from pprint import pprint

import requests

WHISPER_TRANSCRIBE_API_URL = "http://localhost:8000/whisper/transcribe"
AUDIO_URL = "https://upload.wikimedia.org/wikipedia/commons/a/a1/Audio_Sample_-_The_Quick_Brown_Fox_Jumps_Over_The_Lazy_Dog.ogg"

if __name__ == "__main__":
    audio_data_response = requests.get(AUDIO_URL)
    if not audio_data_response.ok:
        print("Something went wrong downloading the audio file.")
        sys.exit(1)

    transcription_response = requests.post(
        WHISPER_TRANSCRIBE_API_URL,
        params={"model_size": "tiny", "word_timestamps": False},
        files={
            "file": (
                AUDIO_URL.split("/")[-1],
                BytesIO(audio_data_response.content),
                "audio/mpeg",
            )
        },
    )
    if not transcription_response.ok:
        print("Something went wrong getting the transcription. Is the server running?")
        sys.exit(1)

    print(
        "Parsed transcription in",
        transcription_response.headers.get("x-response-time"),
        "seconds:",
    )

    transcription = transcription_response.json()
    pprint(transcription)

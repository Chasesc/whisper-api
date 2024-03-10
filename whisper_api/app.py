from fastapi import FastAPI

from whisper_api import middleware
from whisper_api.api.router import router as whisper_router
from whisper_api.transcription import transcribe

transcribe.preload_model()
app = FastAPI()
app.add_middleware(middleware.ProcessTimeMiddleware)

app.include_router(whisper_router)


@app.get("/health")
def health():
    return {"status": "OK"}

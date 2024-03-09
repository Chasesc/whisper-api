from fastapi import FastAPI

from whisper_api.api.router import router as whisper_router

app = FastAPI()

app.include_router(whisper_router)


@app.get("/health")
def health():
    return {"status": "OK"}

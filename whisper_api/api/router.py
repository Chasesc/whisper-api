from fastapi import APIRouter

router = APIRouter(prefix="/whisper", tags=["Whisper"])


@router.post("/transcribe")
async def transcribe():
    return {"text": "hello, world!"}

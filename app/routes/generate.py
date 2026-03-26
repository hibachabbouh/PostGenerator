from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=200, description="Caption topic")
    style: str = Field(..., min_length=2, max_length=50, description="Caption style")


class GenerateResponse(BaseModel):
    caption: str


@router.post("/generate", response_model=GenerateResponse)
def generate_caption_endpoint(payload: GenerateRequest) -> GenerateResponse:
    try:
        from app.services.rag_pipeline import generate_caption
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize RAG pipeline: {exc}",
        ) from exc

    try:
        caption = generate_caption(payload.topic, payload.style)
        return GenerateResponse(caption=caption)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Caption generation failed: {exc}",
        ) from exc

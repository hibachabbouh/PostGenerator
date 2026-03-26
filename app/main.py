from fastapi import FastAPI
from app.routes.generate import router as generate_router

app = FastAPI(
    title="PostGenerator API",
    version="1.0.0",
    description="API for generating Instagram captions with a RAG pipeline.",
)


@app.get("/health", tags=["health"])
def healthcheck() -> dict:
    return {"status": "ok"}


app.include_router(generate_router, prefix="/api", tags=["generate"])

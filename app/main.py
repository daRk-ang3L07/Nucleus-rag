from fastapi import FastAPI
from app.core.config import settings
from app.core.logger import get_logger

from app.api.endpoints.ingest import router as ingest_router
from app.api.endpoints.search import router as search_router
from app.api.endpoints.chat import router as chat_router
from app.api.endpoints.evaluate import router as eval_router
from fastapi.middleware.cors import CORSMiddleware

# Initialize logger
logger = get_logger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A robust RAG backend built for enterprise scale.",
    version=settings.VERSION,
)

# Enable CORS for frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(ingest_router, prefix=f"{settings.API_V1_STR}/ingest", tags=["Ingestion"])
app.include_router(search_router, prefix=f"{settings.API_V1_STR}/search", tags=["Search"])
app.include_router(chat_router, prefix=f"{settings.API_V1_STR}/chat", tags=["Chat"])
app.include_router(eval_router, prefix=f"{settings.API_V1_STR}/evaluate", tags=["Evaluation"])

@app.get("/", tags=["Home"])
async def root():
    """
    Root endpoint returning a welcome message.
    """
    return {
        "message": f"Welcome to the {settings.PROJECT_NAME}",
        "version": settings.VERSION,
        "docs_url": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Check the health status of the application.
    """
    logger.info("Health check endpoint accessed")
    return {
        "status": "ok",
        "project_name": settings.PROJECT_NAME,
        "version": settings.VERSION
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting {settings.PROJECT_NAME} on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

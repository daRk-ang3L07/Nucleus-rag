from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
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

# Serve Static Files (Frontend)
FRONTEND_PATH = os.path.join(os.getcwd(), "frontend/dist")

if os.path.exists(FRONTEND_PATH):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_PATH, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Allow API and Docs through, otherwise serve index.html
        if full_path.startswith(("api", "docs", "redoc", "openapi.json")):
             return None # Let FastAPI's internal router handle it
        
        # Check if the file exists physically
        file_path = os.path.join(FRONTEND_PATH, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
            
        return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))
else:
    @app.get("/", tags=["Home"])
    async def root():
        return {"message": "Backend is running. Frontend build not found.", "docs": "/docs"}

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
    logger.info(f"Starting {settings.PROJECT_NAME} on port 7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)

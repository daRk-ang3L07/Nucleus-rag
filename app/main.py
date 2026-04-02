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
    # Dynamic runtime environment variables for the frontend
    @app.get("/env.js")
    async def get_env_js():
        # Only expose the 'Public' anon and URL keys
        js_content = f"""
        window.VITE_SUPABASE_URL = "{os.getenv('VITE_SUPABASE_URL', '')}";
        window.VITE_SUPABASE_ANON_KEY = "{os.getenv('VITE_SUPABASE_ANON_KEY', '')}";
        """
        from fastapi import Response
        return Response(content=js_content, media_type="application/javascript")

    # Mount assets folder for JS/CSS
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_PATH, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa_backend(full_path: str):
        # 1. Skip API/Docs - let FastAPI routers handle these
        if full_path.startswith(("api", "docs", "redoc", "openapi.json")):
            from fastapi import HTTPException
            raise HTTPException(status_code=404)
        
        # 2. Try to serve exact file from dist (e.g. favicon.svg)
        file_path = os.path.join(FRONTEND_PATH, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
            
        # 3. Default to index.html for all other routes (SPA handling)
        return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))
else:
    logger.warning(f"Frontend dist not found at {FRONTEND_PATH}")
    @app.get("/")
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

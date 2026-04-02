import os
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Depends
from app.services.document_service import document_service
from app.services.vector_service import vector_service
from app.core.auth import get_current_user
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


async def _process_file_background(file_path: str, user_id: str):
    """Index a single file in the background, tagging chunks with user_id."""
    try:
        chunks = document_service.process_single_file(file_path)
        if chunks:
            vector_service.add_documents(chunks, user_id=user_id)
            logger.info(f"Background indexing complete for: {file_path}")
    except Exception as e:
        logger.error(f"Background indexing failed for {file_path}: {e}")


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

@router.get("/files", response_model=List[str])
async def list_files(user_id: str = Depends(get_current_user)):
    """List files belonging to the current user."""
    return document_service.get_all_files(user_id=user_id)


@router.post("/upload", response_model=dict)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    """Upload a file and index it under the current user's namespace."""
    if not file.filename.lower().endswith((".pdf", ".txt", ".js", ".ts", ".py")):
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, .js, .ts, and .py files are supported.")
    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File exceeds maximum size of {MAX_FILE_SIZE // (1024*1024)}MB.")
        file_path = document_service.save_file(content, file.filename, user_id=user_id)
        background_tasks.add_task(_process_file_background, file_path, user_id)
        return {
            "message": f"File '{file.filename}' uploaded. Indexing started.",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=dict)
async def ingest_documents(user_id: str = Depends(get_current_user)):
    """Trigger ingestion for the current user's documents."""
    try:
        chunks = document_service.load_and_split(user_id=user_id)
        if not chunks:
            return {"message": "No documents found to ingest."}
        vector_service.add_documents(chunks, user_id=user_id)
        return {"message": "Documents ingested.", "num_chunks": len(chunks), "status": "success"}
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/files/{filename}", response_model=dict)
async def delete_file(filename: str, user_id: str = Depends(get_current_user)):
    """Delete a file and its vectors for the current user only."""
    file_path = os.path.join(document_service.data_path, user_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    chunks_deleted = vector_service.delete_by_filename(filename, user_id=user_id)
    os.remove(file_path)
    logger.info(f"Deleted {file_path}")
    return {"message": f"'{filename}' deleted.", "chunks_removed": chunks_deleted, "status": "success"}

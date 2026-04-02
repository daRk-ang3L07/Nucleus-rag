from fastapi import APIRouter, HTTPException, Query
from app.services.vector_service import vector_service
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.get("/", response_model=dict)
async def search_documents(
    query: str = Query(..., description="The search query to find relevant documents"),
    k: int = Query(3, description="The number of results to return")
):
    """
    Perform a similarity search across the ingested documents.
    """
    try:
        logger.info(f"Search request received for query: {query}")
        results = vector_service.similarity_search(query, k=k)
        
        formatted_results = [
            {
                "content": result.page_content,
                "metadata": result.metadata
            }
            for result in results
        ]
        
        return {
            "query": query,
            "num_results": len(formatted_results),
            "results": formatted_results,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, HTTPException, Body, Depends
from app.services.llm_service import llm_service
from app.core.auth import get_current_user
from app.core.logger import get_logger
from app.core.rate_limit import check_rate_limit

router = APIRouter()
logger = get_logger(__name__)


@router.post("/", response_model=dict)
async def chat(
    question: str = Body(..., embed=True),
    user_id: str = Depends(get_current_user)
):
    """Chat with your documents. Results are isolated to the current user."""
    check_rate_limit(user_id)
    try:
        logger.info(f"Chat [{user_id}]: {question}")
        response = await llm_service.ask_question(question, user_id=user_id)
        return {
            "question": question,
            "answer": response["answer"],
            "sources": response["sources"],
            "expanded_queries": response.get("expanded_queries", []),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, HTTPException, Body
from app.services.llm_service import llm_service
from app.services.eval_service import eval_service
from app.core.logger import get_logger
from app.core.auth import get_current_user
from fastapi import Depends
from typing import List, Optional
from pydantic import BaseModel, Field

router = APIRouter()
logger = get_logger(__name__)

class EvalRequest(BaseModel):
    questions: Optional[List[str]] = Field(
        None, 
        description="A list of questions to evaluate. If empty, uses default benchmarks."
    )

@router.get("/questions", response_model=List[str])
async def get_eval_questions():
    """Load and return the questions from the master test bank."""
    import json
    import os
    file_path = "eval_questions.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            test_bank = json.load(f)
            return [item["question"] for item in test_bank]
    return ["What is the primary objective of this RAG system?"]

@router.post("/questions", response_model=dict)
async def add_eval_question(
    question: str = Body(..., embed=True)
):
    """Add a new question to the master evaluation test bank."""
    import json
    import os
    file_path = "eval_questions.json"
    test_bank = []
    
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            test_bank = json.load(f)
    
    # Check if question already exists
    if any(q["question"].lower() == question.lower() for q in test_bank):
        return {"message": "Question already exists.", "status": "info"}
    
    # Add new question with a placeholder answer (AI will generate the real one during benchmark)
    test_bank.append({
        "question": question,
        "answer": "Pending AI generation..."
    })
    
    with open(file_path, "w") as f:
        json.dump(test_bank, f, indent=4)
        
    return {"message": "Question added successfully!", "status": "success"}

@router.post("/", response_model=dict)
async def evaluate_rag(
    request: Optional[EvalRequest] = Body(None),
    user_id: str = Depends(get_current_user)
):
    """
    Scientifically evaluate the RAG system's accuracy and relevance.
    """
    import json
    import os

    # 1. Determine Questions to Audit
    test_questions = []
    if request and request.questions and len(request.questions) > 0:
        test_questions = request.questions
    else:
        # Load from the master test bank if no specific questions provided
        file_path = "eval_questions.json"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                test_bank = json.load(f)
                test_questions = [item["question"] for item in test_bank]
        
    # Final fallback if both are empty
    if not test_questions:
        test_questions = ["What is the primary objective of this RAG system?"]

    try:
        # 2. Prepare test data with full RAG trace and Synthetic Ground Truth
        test_data = []
        logger.info(f"Preparing audit data for {len(test_questions)} questions...")
        
        for q in test_questions:
            # Get live RAG response and context trace for the current user
            rag_response = await llm_service.ask_question(q, user_id=user_id)
            contexts = rag_response.get("context_trace", [])
            
            # Generate Synthetic Ground Truth using the "Elite Auditor" Analyst model
            ground_truth = await eval_service.generate_ground_truth(q, contexts)
            
            test_data.append({
                "question": q,
                "answer": rag_response["answer"],
                "contexts": contexts,
                "ground_truth": ground_truth
            })

        # 3. Perform Scientific Audit with Ragas Metrics
        results = await eval_service.evaluate_rag(test_data)
        return {
            "summary": results["summary"],
            "detailed_report": results["detailed_report"],
            "status": "success"
        }
    except Exception as e:
        logger.error(f"RAG Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

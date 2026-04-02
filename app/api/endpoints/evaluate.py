from fastapi import APIRouter, HTTPException, Body, Depends, BackgroundTasks
from app.services.llm_service import llm_service
from app.services.eval_service import eval_service
from app.core.logger import get_logger
from app.core.auth import get_current_user
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid
import json
import os
import asyncio

router = APIRouter()
logger = get_logger(__name__)

# Global In-Memory Task Cache (for single-container HF environment)
eval_tasks = {}

class EvalRequest(BaseModel):
    questions: Optional[List[str]] = Field(
        None, 
        description="A list of questions to evaluate. If empty, uses default benchmarks."
    )

@router.get("/questions", response_model=List[str])
async def get_eval_questions():
    """Load and return the questions from the master test bank."""
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
    file_path = "eval_questions.json"
    test_bank = []
    
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            test_bank = json.load(f)
    
    if any(q["question"].lower() == question.lower() for q in test_bank):
        return {"message": "Question already exists.", "status": "info"}
    
    test_bank.append({
        "question": question,
        "answer": "Pending AI generation..."
    })
    
    with open(file_path, "w") as f:
        json.dump(test_bank, f, indent=4)
        
    return {"message": "Question added successfully!", "status": "success"}

@router.get("/status/{task_id}")
async def get_eval_status(task_id: str):
    """Retrieve the progress or final report of an audit task."""
    task = eval_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.post("/", response_model=dict)
async def start_evaluate_rag(
    background_tasks: BackgroundTasks,
    request: Optional[EvalRequest] = Body(None),
    user_id: str = Depends(get_current_user)
):
    """
    Initiate a scientific audit in the background (prevents Timeout).
    """
    task_id = str(uuid.uuid4())
    eval_tasks[task_id] = {
        "status": "pending",
        "progress": "Initializing audit...",
        "result": None
    }
    
    background_tasks.add_task(run_audit_background, task_id, request, user_id)
    return {"task_id": task_id, "status": "accepted"}

async def run_audit_background(task_id: str, request: Optional[EvalRequest], user_id: str):
    """Heavy background logic for RAGAS Audit."""
    try:
        # 1. Determine Questions
        test_questions = []
        if request and request.questions and len(request.questions) > 0:
            test_questions = request.questions
        else:
            file_path = "eval_questions.json"
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    test_bank = json.load(f)
                    test_questions = [item["question"] for item in test_bank]
        
        if not test_questions:
            test_questions = ["What is the primary objective of this RAG system?"]
        
        test_questions = test_questions[:5] # Limit for speed
        eval_tasks[task_id]["progress"] = f"Generating responses for {len(test_questions)} samples..."
        
        # 2. Parallel Response Generation
        async def process_one_q(q):
            rag_response = await llm_service.ask_question(q, user_id=user_id)
            contexts = rag_response.get("context_trace", [])
            ground_truth = await eval_service.generate_ground_truth(q, contexts)
            return {
                "question": q,
                "answer": rag_response["answer"],
                "contexts": contexts,
                "ground_truth": ground_truth
            }

        test_data = await asyncio.gather(*[process_one_q(q) for q in test_questions])
        
        # 3. Scientific RAGAS Audit
        eval_tasks[task_id]["progress"] = "Calculating scientific benchmarks..."
        await asyncio.sleep(15)
        results = await eval_service.evaluate_rag(test_data)
        
        # 4. Finalise
        eval_tasks[task_id]["status"] = "completed"
        eval_tasks[task_id]["progress"] = "Audit complete!"
        eval_tasks[task_id]["result"] = results
        
    except Exception as e:
        logger.error(f"Background Audit failed: {e}")
        eval_tasks[task_id]["status"] = "failed"
        eval_tasks[task_id]["progress"] = f"Error: {str(e)}"

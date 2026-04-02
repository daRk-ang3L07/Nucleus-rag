from ragas import evaluate
from ragas.metrics import faithfulness, answer_similarity
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.core.config import settings
from app.core.logger import get_logger
from datasets import Dataset
from typing import List, Dict

logger = get_logger(__name__)

class SafeGemini(ChatGoogleGenerativeAI):
    """Safety wrapper for Gemini with Ragas."""
    def generate(self, messages, stop=None, callbacks=None, **kwargs):
        kwargs.pop("temperature", None)
        return super().generate(messages, stop=stop, callbacks=callbacks, **kwargs)

    async def agenerate(self, messages, stop=None, callbacks=None, **kwargs):
        kwargs.pop("temperature", None)
        return await super().agenerate(messages, stop=stop, callbacks=callbacks, **kwargs)

class EvalService:
    def __init__(self):
        # We use the user-confirmed working model: Gemini 2.5
        self.model_name = "gemini-2.5-flash"
        api_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
        
        self.gemini_llm = SafeGemini(
            model=self.model_name,
            google_api_key=api_key,
            temperature=0.1
        )
        
        self.gemini_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )

    async def generate_ground_truth(self, question: str, contexts: List[str]) -> str:
        """Generate a gold-standard answer using the 2.5 model."""
        context_text = "\n".join(contexts)
        prompt = f"Answer concisely based on this context:\nContext: {context_text}\nQuestion: {question}"
        try:
            res = await self.gemini_llm.ainvoke(prompt)
            return str(res.content)
        except Exception as e:
            logger.error(f"GT Generation failed: {e}")
            return "Answer based on provided document sections."

    async def evaluate_rag(self, test_data: List[Dict]) -> Dict:
        """Run a 'Quota-Friendly' evaluation using Gemini 2.5."""
        try:
            llm_wrapper = LangchainLLMWrapper(self.gemini_llm)
            emb_wrapper = LangchainEmbeddingsWrapper(self.gemini_embeddings)
            
            # Quota Strategy: 
            # 1. Faithfulness (Uses LLM - counts toward 20/day)
            # 2. Answer Similarity (Uses Embeddings - does NOT count toward 20/day)
            faithfulness.llm = llm_wrapper
            answer_similarity.embeddings = emb_wrapper
            
            dataset_dict = {
                "question": [str(item["question"]) for item in test_data],
                "answer": [str(item["answer"]) for item in test_data],
                "contexts": [[str(c) for c in item["contexts"]] for item in test_data],
                "ground_truth": [str(item.get("ground_truth", "The provided document context.")) for item in test_data]
            }
            dataset = Dataset.from_dict(dataset_dict)
            
            # max_workers=1 is essential for the limited Gemini 2.5 trial
            result = evaluate(
                dataset, 
                metrics=[faithfulness, answer_similarity],
                llm=llm_wrapper,
                embeddings=emb_wrapper,
                run_config=RunConfig(max_workers=1),
                raise_exceptions=False
            )
            
            return {
                "summary": {
                    "faithfulness": float(result.get("faithfulness", 0.0)),
                    "answer_similarity": float(result.get("answer_similarity", 0.0)),
                    "overall_score": (float(result.get("faithfulness", 0.0)) + float(result.get("answer_similarity", 0.0))) / 2
                }
            }
        except Exception as e:
            logger.error(f"Quota-limited Eval failed: {e}")
            return {"summary": {"error": f"Evaluation limit reached or failed: {str(e)}"}}

eval_service = EvalService()

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
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
        # Switch to the new Hugging Face Router (The old api-inference URL is retired)
        self.hf_model = "microsoft/Phi-3-mini-4k-instruct"
        router_url = f"https://router.huggingface.co/hf-inference/models/{self.hf_model}"
        
        # Configure the HF Engine (Primary for Audit)
        self.hf_llm = HuggingFaceEndpoint(
            endpoint_url=router_url,
            huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN,
            temperature=0.1,
            max_new_tokens=500
        )
        
        # Configure Gemini (Backup for Audit / Primary for Chat)
        self.gemini_llm = SafeGemini(
            model="gemini-2.5-flash",
            google_api_key=settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY,
            temperature=0.1
        )
        
        self.gemini_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
        )

    async def generate_ground_truth(self, question: str, contexts: List[str]) -> str:
        """Generate a gold-standard answer using the free Phi-3 model to save Gemini quota."""
        context_text = "\n".join(contexts)
        prompt = f"System: You are an expert data extractor. Answer concisely based on context.\nContext: {context_text}\nUser: {question}\nAnswer:"
        try:
            # We use the free HF model to save Gemini 20-req quota
            return self.hf_llm.invoke(prompt)
        except Exception as e:
            logger.error(f"HF GT Generation failed: {e}")
            # Fallback to Gemini 2.5 if HF is totally down
            res = await self.gemini_llm.ainvoke(prompt)
            return str(res.content)

    async def evaluate_rag(self, test_data: List[Dict]) -> Dict:
        """Run a slimmed-down evaluation using only free resources."""
        try:
            # Use Phi-3 as the main LLM to respect Gemini's 20-req limit
            llm_wrapper = LangchainLLMWrapper(self.hf_llm)
            emb_wrapper = LangchainEmbeddingsWrapper(self.gemini_embeddings)
            
            # Setup metrics for faithfulness and relevance
            faithfulness.llm = llm_wrapper
            answer_relevancy.llm = llm_wrapper
            answer_relevancy.embeddings = emb_wrapper
            
            dataset_dict = {
                "question": [str(item["question"]) for item in test_data],
                "answer": [str(item["answer"]) for item in test_data],
                "contexts": [[str(c) for c in item["contexts"]] for item in test_data],
                "ground_truth": [str(item.get("ground_truth", "Please answer based on documents.")) for item in test_data]
            }
            dataset = Dataset.from_dict(dataset_dict)
            
            # Run one-by-one to avoid rate limits
            result = evaluate(
                dataset, 
                metrics=[faithfulness, answer_relevancy],
                llm=llm_wrapper,
                embeddings=emb_wrapper,
                run_config=RunConfig(max_workers=1, timeout=300),
                raise_exceptions=False
            )
            
            summary = {
                "faithfulness": float(result.get("faithfulness", 0.0)),
                "answer_relevancy": float(result.get("answer_relevancy", 0.0))
            }
            summary["overall_score"] = sum(summary.values()) / max(len(summary), 1)
            
            return {"summary": summary}
        except Exception as e:
            logger.error(f"Audit failed: {e}")
            return {"summary": {"error": str(e)}}

eval_service = EvalService()

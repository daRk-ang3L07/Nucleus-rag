from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from app.core.config import settings
from app.core.logger import get_logger
import pandas as pd
from datasets import Dataset
from typing import List, Dict
from langchain_core.language_models.llms import LLM

logger = get_logger(__name__)

class HFRestLLM(LLM):
    """Custom LLM to bypass Hugging Face deprecation bugs."""
    hf_token: str
    model_name: str
    
    @property
    def _llm_type(self) -> str:
        return "hf_rest"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        import requests
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.hf_token}", 
            "Content-Type": "application/json"
        }
        
        models_to_try = [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "HuggingFaceH4/zephyr-7b-beta",
            "meta-llama/Llama-3.1-8B-Instruct",
            "microsoft/Phi-3.5-mini-instruct"
        ]
        
        last_error = None
        for model in models_to_try:
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500
                }
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
            except Exception as e:
                last_error = str(e)
                continue
                
        raise Exception(f"All Hugging Face fallback models failed. Last error: {last_error}")

class SafeGemini(ChatGoogleGenerativeAI):
    """Intercept kwargs injected by Ragas that crash the Gemini client."""
    def generate(self, messages, stop=None, callbacks=None, **kwargs):
        kwargs.pop("temperature", None)
        return super().generate(messages, stop=stop, callbacks=callbacks, **kwargs)

    async def agenerate(self, messages, stop=None, callbacks=None, **kwargs):
        kwargs.pop("temperature", None)
        return await super().agenerate(messages, stop=stop, callbacks=callbacks, **kwargs)

class EvalService:
    def __init__(self):
        # Primary Gemini setup with safety wrapper
        self.gemini_llm = SafeGemini(
            model=settings.GEMINI_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY,
            temperature=0,
            max_retries=0
        )
        self.gemini_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
        )
        
        # Secondary Hugging Face setup using our direct API bypass class
        self.hf_llm = HFRestLLM(
            hf_token=settings.HUGGINGFACEHUB_API_TOKEN,
            model_name=settings.HF_MODEL_NAME
        )
        
    def _setup_metrics(self, use_fallback=False):
        """Configure RAGAS metrics with the chosen LLM."""
        llm = self.hf_llm if use_fallback else self.gemini_llm
        # Note: We keep Gemini embeddings as they usually have higher limits than LLM
        # but the LLM is what usually triggers the 429 in RAGAS.
        wrapper = LangchainLLMWrapper(llm)
        faithfulness.llm = wrapper
        answer_relevancy.llm = wrapper
        answer_relevancy.embeddings = LangchainEmbeddingsWrapper(self.gemini_embeddings)
        return [faithfulness, answer_relevancy]

    async def generate_ground_truth(self, question: str, contexts: List[str]) -> str:
        """Use a high-performance LLM (Critic) to generate a gold-standard answer from the provided contexts."""
        prompt = f"""
        Given the following context snippets from a RAG system, write a definitive, accurate, and concise "Gold Standard" answer to the user question.
        
        Question: {question}
        
        Contexts:
        {chr(10).join(f"- {c}" for c in contexts)}
        
        Gold Standard Answer:"""
        
        try:
            # We prefer the stronger HF model for the "Gold Standard" generation
            return self.hf_llm._call(prompt)
        except Exception:
            # Fallback to Gemini if HF is down
            response = await self.gemini_llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)

    async def evaluate_rag(self, test_data: List[Dict], use_fallback=True):
        """
        Evaluate samples with automatic fallback support and synthetic ground truth.
        """
        logger.info(f"Starting RAGAS evaluation on {len(test_data)} samples.")
        
        try:
            # Setup wrappers
            llm_to_use = self.hf_llm if use_fallback else self.gemini_llm
            llm_wrapper = LangchainLLMWrapper(llm_to_use)
            emb_wrapper = LangchainEmbeddingsWrapper(self.gemini_embeddings)
            
            from ragas.metrics import answer_correctness, context_precision, context_recall
            
            # Explicitly configure ALL metrics to avoid OpenAI defaults
            for metric in [faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall]:
                metric.llm = llm_wrapper
                if hasattr(metric, 'embeddings'):
                    metric.embeddings = emb_wrapper
            
            advanced_metrics = [faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall]
            
            dataset_dict = {
                "question": [str(item["question"]) for item in test_data],
                "answer": [str(item["answer"]) for item in test_data],
                "contexts": [[str(c) for c in item["contexts"]] for item in test_data],
                "ground_truth": [str(item.get("ground_truth", "The AI should provide an accurate answer based on documents.")) for item in test_data]
            }

            from datasets import Features, Sequence, Value
            features = Features({
                "question": Value("string"),
                "answer": Value("string"),
                "contexts": Sequence(Value("string")),
                "ground_truth": Value("string")
            })
            
            dataset = Dataset.from_dict(dataset_dict, features=features)
            
            # Configure RAGAS to be "Gentle" and patient
            run_config = RunConfig(timeout=300, max_retries=5, max_workers=1)
            
            # Run evaluation with explicit LLM and Embeddings to override defaults
            result = evaluate(
                dataset, 
                metrics=advanced_metrics, 
                llm=llm_wrapper, 
                embeddings=emb_wrapper,
                run_config=run_config,
                raise_exceptions=False
            )
            
            # Convert to standard dictionary
            import json
            scores_json_str = result.to_pandas().to_json(orient="records")
            scores = json.loads(scores_json_str)
            
            # Convert primary metrics safely to floats
            summary = {
                "faithfulness": float(result.get("faithfulness", 0.0)) if result.get("faithfulness") is not None else 0.0,
                "answer_relevancy": float(result.get("answer_relevancy", 0.0)) if result.get("answer_relevancy") is not None else 0.0,
                "answer_correctness": float(result.get("answer_correctness", 0.0)) if result.get("answer_correctness") is not None else 0.0,
                "context_precision": float(result.get("context_precision", 0.0)) if result.get("context_precision") is not None else 0.0,
                "context_recall": float(result.get("context_recall", 0.0)) if result.get("context_recall") is not None else 0.0
            }
            # Clean up potential NaN values
            import math
            for k, v in summary.items():
                if math.isnan(v): summary[k] = 0.0

            summary["overall_score"] = sum(summary.values()) / max(len(summary), 1)
            
            return {
                "summary": summary,
                "detailed_report": scores
            }
        except Exception as e:
            # Handle 429 errors and same logic as before...
            error_str = str(e).lower()
            if ("429" in error_str or "quota" in error_str) and not use_fallback:
                return await self.evaluate_rag(test_data, use_fallback=True)
            logger.error(f"RAGAS evaluation failed: {e}")
            raise e

eval_service = EvalService()

from typing import List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
from app.services.vector_service import vector_service
from app.services.reranker_service import reranker_service
from app.core.logger import get_logger
import aiohttp

logger = get_logger(__name__)

class LLMService:
    def __init__(self):
        self.google_api_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
        self.hf_token = settings.HUGGINGFACEHUB_API_TOKEN
        
        # Use Gemini 2.5 as primary (User preferred)
        self.model_name = "gemini-2.5-flash"
        
        self.gemini_llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.google_api_key,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        
        self.template = """
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, say that you don't know. 
        Use three sentences maximum and keep the answer concise.

        Context: {context}

        Question: {question}

        Answer:"""
        self.prompt = ChatPromptTemplate.from_template(self.template)

    async def _hf_invoke(self, prompt: str) -> str:
        """Fallback to Qwen-7B on Hugging Face via Router API."""
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.hf_token}", 
            "Content-Type": "application/json"
        }
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 800,
            "temperature": 0.3
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        raise Exception(f"HF Error {response.status}: {error_text}")
        except Exception as e:
            logger.error(f"Hugging Face Fallback failed: {e}")
            raise e

    async def _safe_invoke(self, prompt: Any, is_fallback=False) -> str:
        """Safely invoke Gemini with automatic Qwen fallback on quota hit."""
        try:
            # If we are already in fallback mode, go straight to HF
            if is_fallback:
                logger.warning("Using Qwen-7B Fallback...")
                prompt_text = prompt.messages[0].content if hasattr(prompt, 'messages') else str(prompt)
                return await self._hf_invoke(prompt_text)
            
            # Normal Gemini path
            response = await self.gemini_llm.ainvoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
            
        except Exception as e:
            # If Gemini hits a quota limit (429), trigger the Qwen fallback
            if "429" in str(e) or "quota" in str(e).lower():
                logger.error(f"Gemini Limit Reached! Switching to Qwen-7B... ({e})")
                return await self._safe_invoke(prompt, is_fallback=True)
            
            logger.error(f"LLM Error: {e}")
            return f"Error: The AI service is currently overloaded. Please try again in 1 minute. ({str(e)})"

    async def _expand_query(self, query: str) -> List[str]:
        expansion_prompt = f"Generate 3 variations of this question to improve search: {query}\nProvide only the questions, one per line."
        logger.info(f"Expanding query: {query}")
        content = await self._safe_invoke(expansion_prompt)
        expanded_queries = [line.strip() for line in content.split("\n") if line.strip()]
        return list(set([query] + expanded_queries))

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    async def ask_question(self, question: str, user_id: str):
        logger.info(f"Expert RAG query: {question}")
        expanded_queries = await self._expand_query(question)
        
        all_docs = []
        for q in expanded_queries:
            all_docs.extend(vector_service.hybrid_search(q, k=5, user_id=user_id))
        
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        reranked_docs = reranker_service.rerank(question, unique_docs)
        final_docs = reranked_docs[:5]
        
        context_text = self._format_docs(final_docs)
        context_trace = [doc.page_content for doc in final_docs]
        
        formatted_prompt = self.prompt.format(context=context_text, question=question)
        answer = await self._safe_invoke(formatted_prompt)
        
        sources = [{"source": doc.metadata.get("source", "unknown"), "score": doc.metadata.get("rerank_score", 0)} for doc in final_docs[:1]]
        
        return {
            "answer": answer,
            "sources": sources,
            "context_trace": context_trace,
            "context": context_text,
            "expanded_queries": expanded_queries
        }

llm_service = LLMService()

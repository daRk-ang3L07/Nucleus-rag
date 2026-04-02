from typing import List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings
from app.services.vector_service import vector_service
from app.services.reranker_service import reranker_service
from app.core.logger import get_logger

logger = get_logger(__name__)

class LLMService:
    def __init__(self):
        # Configuration is loaded from settings
        self.google_api_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
        self.hf_token = settings.HUGGINGFACEHUB_API_TOKEN
        
        # Initialize Gemini as Primary
        self.gemini_llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL_NAME,
            google_api_key=self.google_api_key,
            temperature=0,
            max_retries=0,
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
        """Raw API call to bypass all HuggingFace library version bugs entirely."""
        import aiohttp
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.hf_token}", 
            "Content-Type": "application/json"
        }
        
        prompt_text = prompt.messages[0].content if hasattr(prompt, 'messages') else str(prompt)
        
        models_to_try = [
            "Qwen/Qwen2.5-72B-Instruct",
            "microsoft/Phi-3-mini-4k-instruct",
            "google/gemma-2b-it",
            "meta-llama/Llama-3.2-3B-Instruct"
        ]
        
        last_error = None
        async with aiohttp.ClientSession() as session:
            for model in models_to_try:
                try:
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt_text}],
                        "max_tokens": 500
                    }
                    async with session.post(url, headers=headers, json=payload, raise_for_status=False) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data["choices"][0]["message"]["content"]
                        else:
                            error_text = await response.text()
                            raise Exception(f"HTTP {response.status}: {error_text}")
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Fallback model {model} failed: {e}")
                    continue
                
        raise Exception(f"All Hugging Face fallback models failed. Last error: {last_error}")

    async def _safe_invoke(self, prompt: Any, use_fallback=False) -> str:
        """Safely invoke LLM with automatic fallback on quota errors."""
        try:
            if use_fallback:
                logger.warning(f"Using Backup HTTP LLM: {settings.HF_MODEL_NAME}")
                return await self._hf_invoke(prompt)
            else:
                # Gemini handles native async brilliantly
                response = await self.gemini_llm.ainvoke(prompt)
                return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                if not use_fallback:
                    logger.error(f"Gemini Quota Exceeded! Switching to Fallback... Error: {e}")
                    return await self._safe_invoke(prompt, use_fallback=True)
            logger.error(f"LLM Invocation Failed: {e}")
            raise e

    async def _expand_query(self, query: str) -> list[str]:
        """
        AI-driven Query Expansion: Generate 3 variations of the user query
        to improve retrieval coverage (Multi-Query Retrieval).
        """
        expansion_prompt = f"""
        You are an AI assistant. Your task is to generate 3 semantic variations of the user question
        to improve document retrieval. Provide only the questions, one per line.
        Original: {query}
        """
        logger.info(f"Expanding query: {query}")
        
        # Use safe invoke for fallback support
        content = await self._safe_invoke(expansion_prompt)
        expanded_queries = [line.strip() for line in content.split("\n") if line.strip()]
        
        # Prepend the original query to the list
        return list(set([query] + expanded_queries))

    def _format_docs(self, docs):
        """Helper to format documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    async def ask_question(self, question: str, user_id: str):
        """
        Perform an EXPERT RAG query with Filename Detection:
        1. Filename Extraction (Regex)
        2. Multi-Query Expansion
        3. Hybrid Search (Keyword + Semantic) with Filename Metadata Filter
        4. Grounded Generation
        """
        logger.info(f"Expert RAG query initiated: {question}")
        
        # 1. Filename Extraction (Detect mentions like rohan.pdf or fees.js)
        import re
        # Look for words ending in .pdf, .js, .txt, .py, .ts
        file_match = re.search(r'([\w\.-]+\.(pdf|txt|js|ts|py))', question.lower())
        detected_filename = file_match.group(1) if file_match else None
        
        if detected_filename:
            logger.info(f"Detected filename in query: {detected_filename}")

        # 2. Multi-Query Expansion
        expanded_queries = await self._expand_query(question)
        
        # 3. Hybrid Retrieval with Metadata Filter
        all_docs = []
        for q in expanded_queries:
            # Pass the detected_filename to the vector service for strict filtering
            all_docs.extend(vector_service.hybrid_search(q, k=5, user_id=user_id, filename=detected_filename))
        
        # De-duplicate docs by content
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        
        # 3. Reranking (Hugging Face Layer)
        reranked_docs = reranker_service.rerank(question, unique_docs)
        final_docs = reranked_docs[:5]
        
        # 4. Grounded Generation
        context_text = self._format_docs(final_docs)
        context_trace = [doc.page_content for doc in final_docs] # Structured for Ragas
        
        # Manually format the prompt to use _safe_invoke for generation fallback
        formatted_prompt = self.prompt.format(context=context_text, question=question)
        answer = await self._safe_invoke(formatted_prompt)
        
        # 5. Extract top 1 source and scores
        sources = []
        if final_docs:
            doc = final_docs[0]
            source = doc.metadata.get("source", "unknown")
            score = doc.metadata.get("rerank_score", 0)
            sources.append({"source": source, "score": score})
        
        return {
            "answer": answer,
            "sources": sources,
            "context_trace": context_trace, # Required for Ragas
            "context": context_text,
            "expanded_queries": expanded_queries
        }

llm_service = LLMService()

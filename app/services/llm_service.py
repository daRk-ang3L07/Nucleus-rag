from typing import List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
from app.services.vector_service import vector_service
from app.services.reranker_service import reranker_service
from app.core.logger import get_logger

logger = get_logger(__name__)

class LLMService:
    def __init__(self):
        self.google_api_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
        
        # Use Gemini 1.5 Flash (1,500 requests/day limit)
        # 2.5 Flash Lite is experimental and has a strict 20 req/day limit.
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

    async def _safe_invoke(self, prompt: Any) -> str:
        """Safely invoke Gemini with proper error handling."""
        try:
            response = await self.gemini_llm.ainvoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"Gemini Invocation Failed: {e}")
            return f"Error: The AI service is currently unavailable. ({str(e)})"

    async def _expand_query(self, query: str) -> List[str]:
        """Generate variations of the query to improve retrieval."""
        expansion_prompt = f"""
        You are an AI assistant. Generate 3 semantic variations of the user question
        to improve document retrieval. Provide only the questions, one per line.
        Original: {query}
        """
        logger.info(f"Expanding query: {query}")
        content = await self._safe_invoke(expansion_prompt)
        expanded_queries = [line.strip() for line in content.split("\n") if line.strip()]
        return list(set([query] + expanded_queries))

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    async def ask_question(self, question: str, user_id: str):
        """Perform a full RAG cycle."""
        logger.info(f"Expert RAG query initiated: {question}")
        
        # 1. Expand Query
        expanded_queries = await self._expand_query(question)
        
        # 2. Hybrid Retrieval
        all_docs = []
        for q in expanded_queries:
            all_docs.extend(vector_service.hybrid_search(q, k=5, user_id=user_id))
        
        # De-duplicate
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        
        # 3. Rerank
        reranked_docs = reranker_service.rerank(question, unique_docs)
        final_docs = reranked_docs[:5]
        
        # 4. Generate Answer
        context_text = self._format_docs(final_docs)
        context_trace = [doc.page_content for doc in final_docs]
        
        formatted_prompt = self.prompt.format(context=context_text, question=question)
        answer = await self._safe_invoke(formatted_prompt)
        
        sources = []
        if final_docs:
            doc = final_docs[0]
            sources.append({
                "source": doc.metadata.get("source", "unknown"),
                "score": doc.metadata.get("rerank_score", 0)
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "context_trace": context_trace,
            "context": context_text,
            "expanded_queries": expanded_queries
        }

llm_service = LLMService()

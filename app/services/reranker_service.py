from sentence_transformers import CrossEncoder
from app.core.logger import get_logger
from typing import List

logger = get_logger(__name__)

class RerankerService:
    def __init__(self):
        # Using a tiny but effective model for reranking
        # This will be downloaded automatically on the first run
        self.model_name = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
        try:
            logger.info(f"Loading Hugging Face Reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self.model = None

    def rerank(self, query: str, documents: List) -> List:
        """
        Precise Reranking: Recalculate scores for query-document pairs
        to find the absolute most relevant chunks.
        """
        if not self.model or not documents:
            return documents

        logger.info(f"Reranking {len(documents)} document chunks with Hugging Face.")
        
        # Prepare pairs for the Cross-Encoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Generate semantic scores
        scores = self.model.predict(pairs)
        
        # Attach scores to documents and sort
        for i, doc in enumerate(documents):
            doc.metadata["rerank_score"] = float(scores[i])
            
        # Sort documents by score in descending order
        reranked_docs = sorted(documents, key=lambda x: x.metadata["rerank_score"], reverse=True)
        
        return reranked_docs

reranker_service = RerankerService()

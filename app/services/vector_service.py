from typing import List, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class VectorService:
    def __init__(self):
        api_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
            task_type="retrieval_document"
        )
        self.vector_db: Optional[Chroma] = None
        self._bm25_retriever = None

    def _get_vector_db(self) -> Chroma:
        if self.vector_db is None:
            logger.info(f"Initializing ChromaDB at {settings.CHROMA_DB_PATH}")
            self.vector_db = Chroma(
                persist_directory=settings.CHROMA_DB_PATH,
                embedding_function=self.embeddings,
                collection_name="rag_collection"
            )
        return self.vector_db

    def add_documents(self, documents: List, user_id: str):
        """Add chunks to ChromaDB, tagging each with user_id metadata."""
        db = self._get_vector_db()
        # Inject user_id into every chunk's metadata
        for doc in documents:
            doc.metadata["user_id"] = user_id
        logger.info(f"Adding {len(documents)} chunks for user {user_id}")
        db.add_documents(documents)
        self._bm25_retriever = None  # reset BM25

    def _get_bm25_retriever(self, user_id: str) -> Optional[BM25Retriever]:
        """Lazy-initialize BM25 filtered to the current user's documents."""
        db = self._get_vector_db()
        all_data = db.get(where={"user_id": user_id})
        if not all_data["documents"]:
            return None
        from langchain_core.documents import Document
        docs = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(all_data["documents"], all_data["metadatas"])
        ]
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = 5
        return retriever

    def hybrid_search(self, query: str, user_id: str, k: int = 5, filename: Optional[str] = None) -> List:
        """Hybrid search scoped strictly to the current user."""
        logger.info(f"Hybrid search [{user_id}]: {query}")
        vector_docs = self.similarity_search(query, user_id=user_id, k=k, filename=filename)
        keyword_docs = []
        bm25 = self._get_bm25_retriever(user_id)
        if bm25:
            results = bm25.invoke(query)
            if filename:
                clean_name = filename.replace("data/", "")
                keyword_docs = [d for d in results if clean_name in d.metadata.get("source", "")]
            else:
                keyword_docs = results
        combined = vector_docs + keyword_docs
        unique = list({doc.page_content: doc for doc in combined}.values())
        logger.info(f"Hybrid search found {len(unique)} candidates")
        return unique

    def similarity_search(self, query: str, user_id: str, k: int = 3, filename: Optional[str] = None) -> List:
        """Semantic search always filtered by user_id."""
        db = self._get_vector_db()
        # Build filter: always include user_id
        chroma_filter: dict = {"user_id": user_id}
        if filename:
            file_path = filename if "data/" in filename else f"data/{filename}"
            chroma_filter = {"$and": [{"user_id": user_id}, {"source": file_path}]}
        results = db.similarity_search(query, k=k, filter=chroma_filter)
        return results

    def delete_by_filename(self, filename: str, user_id: str) -> int:
        """Delete chunks belonging to a specific user and file."""
        db = self._get_vector_db()
        source_path = f"data/{user_id}/{filename}"
        collection = db._collection
        results = collection.get(where={"$and": [{"user_id": user_id}, {"source": source_path}]})
        ids = results.get("ids", [])
        if ids:
            collection.delete(ids=ids)
            self._bm25_retriever = None
            logger.info(f"Deleted {len(ids)} chunks for {source_path}")
        return len(ids)


vector_service = VectorService()

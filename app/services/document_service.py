import os
from typing import List
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class DocumentService:
    def __init__(self, data_path: str = settings.DATA_DIR):
        self.data_path = data_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )

    def _user_path(self, user_id: str) -> str:
        """Returns and ensures the per-user data directory exists."""
        path = os.path.join(self.data_path, user_id)
        os.makedirs(path, exist_ok=True)
        return path

    def save_file(self, file_content: bytes, filename: str, user_id: str) -> str:
        """Save an uploaded file to data/{user_id}/filename."""
        user_dir = self._user_path(user_id)
        file_path = os.path.join(user_dir, filename)
        with open(file_path, "wb") as f:
            f.write(file_content)
        logger.info(f"File saved: {file_path}")
        return file_path

    def get_all_files(self, user_id: str) -> List[str]:
        """List files belonging to a specific user."""
        user_dir = os.path.join(self.data_path, user_id)
        if not os.path.exists(user_dir):
            return []
        return [f for f in os.listdir(user_dir) if os.path.isfile(os.path.join(user_dir, f))]

    def process_single_file(self, file_path: str) -> List:
        """Process a single file and return chunks."""
        loader = None
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith((".txt", ".js", ".ts", ".py")):
            loader = TextLoader(file_path)
        if loader:
            docs = loader.load()
            return self.text_splitter.split_documents(docs)
        return []

    def load_and_split(self, user_id: str) -> List:
        """Load and split all documents for a specific user."""
        user_dir = self._user_path(user_id)
        documents = []
        for filename in os.listdir(user_dir):
            file_path = os.path.join(user_dir, filename)
            if os.path.isfile(file_path):
                try:
                    chunks = self.process_single_file(file_path)
                    documents.extend(chunks)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        logger.info(f"Loaded {len(documents)} chunks for user {user_id}")
        return documents


document_service = DocumentService()

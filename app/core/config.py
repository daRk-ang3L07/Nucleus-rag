from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings for the Enterprise RAG Backend.
    Loads variables from the environment or a .env file.
    """
    PROJECT_NAME: str = "Enterprise RAG Backend"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"

    # Google Gemini Configuration
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash"
    
    # Hugging Face Configuration (Fallback)
    HUGGINGFACEHUB_API_TOKEN: Optional[str] = None
    HF_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.3"
    
    EMBEDDING_MODEL_NAME: str = "text-embedding-004"

    # Vector Database Configuration
    CHROMA_DB_PATH: str = "chroma_db"

    # Supabase Configuration
    SUPABASE_URL: Optional[str] = None
    SUPABASE_ANON_KEY: Optional[str] = None
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()

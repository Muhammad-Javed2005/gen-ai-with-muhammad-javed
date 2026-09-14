import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """
    Enterprise-grade settings management using Pydantic.
    Automatically reads from environment variables or .env file.
    """
    # Application Config

    APP_NAME: str = Field(default="Enterprise GenAI Service", description="Name of the application")
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")
    DEBUG: bool = Field(default=False, description="Debug mode flag")

    # LLM API Keys (Sensitive Data)

    OPENAI_API_KEY: str = Field(..., description="OpenAI API Key")
    MISTRAL_API_KEY: str = Field(default="", description="Mistral API Key")
    
    # Vector DB Config

    CHROMA_PERSIST_DIRECTORY: str = Field(default="./chroma_db", description="Path to Chroma vector store")
    
    # Security Config

    SECRET_KEY: str = Field(..., description="Secret key for JWT / Session encryption")
    MAX_TOKENS_LIMIT: int = Field(default=2000, description="Max token limit per request to prevent cost abuse")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" # Ignore extra env variables safely

# Singleton instance to import across the project


settings = Settings()
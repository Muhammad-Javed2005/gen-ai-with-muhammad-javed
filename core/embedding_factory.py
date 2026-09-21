from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings

class EmbeddingFactory:
    """
    Centralized factory to initialize and standardize embedding models 
    for Vector Stores across the application.
    """
    
    @staticmethod
    def get_openai_embeddings(model_name: str = "text-embedding-3-small"):
        """
        Returns OpenAI embedding model instance using validated API key.
        """
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API Key is missing for embeddings.")
            
        return OpenAIEmbeddings(
            model=model_name,
            api_key=settings.OPENAI_API_KEY
        )

    @staticmethod
    def get_huggingface_embeddings(model_name: str = "all-MiniLM-L-v2"):
        """
        Returns local HuggingFace embedding model instance (No API key needed).
        Great for cost saving and local privacy compliance.
        """
        return HuggingFaceEmbeddings(
            model_name=model_name
        )
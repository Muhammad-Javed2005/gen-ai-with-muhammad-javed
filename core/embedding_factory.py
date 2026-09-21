from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings

class EmbeddingFactory:
    """
    Centralized factory to initliaze and standerdize embedding models
    for vector Stores across the application.
    """

    @staticmethod
    def get_openai_embeddings(model_name : str = "text-embedding-3-small"):
        """
        Return OpenAI embedding Model instance using validated API key.
        """

        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is missing in enviornment settings!")

        return OpenAIEmbeddings(
            model = model_name, 
            api_key = settings.OPENAI_API_KEY
        )

    
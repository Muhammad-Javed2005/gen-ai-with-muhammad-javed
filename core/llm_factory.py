from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from config.settings import settings


class LLMFactory:
    """
    Enterprise LLM Factory: Centralized management for initializing 
    different LLM providers with built-in safety and configuration parameters.
    """


    @staticmethod
    def get_openai_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.2):
        """
        Initializes and returns an OpenAI chat model securely using Pydantic settings.
        """
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API Key is missing in environment settings!")
            
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=settings.OPENAI_API_KEY,
            max_tokens=settings.MAX_TOKENS_LIMIT
        )

    @staticmethod
    def get_mistral_llm(model_name: str = "mistral-large-latest", temperature: float = 0.2):
        """
        Initializes and returns a Mistral chat model as an alternative enterprise provider.
        """
        if not settings.MISTRAL_API_KEY:
            raise ValueError("Mistral API Key is missing in environment settings!")

        return ChatMistralAI(
            model=model_name,
            temperature=temperature,
            api_key=settings.MISTRAL_API_KEY
        )


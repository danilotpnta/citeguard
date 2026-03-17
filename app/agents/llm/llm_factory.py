import os
from langfuse.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class ModelFactory:
    """Factory for creating LLM clients (OpenAI SDK-compatible)"""

    @staticmethod
    def create_client(provider: str = "openai") -> OpenAI:
        """
        Create an OpenAI-compatible client for different providers

        Args:
            provider: 'openai' or 'groq'

        Returns:
            OpenAI client configured for the provider
        """
        if provider == "openai":
            return OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif provider == "groq":
            return OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.getenv("GROQ_API_KEY"),
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

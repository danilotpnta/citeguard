from pydantic import BaseModel
from omegaconf import DictConfig
from collections import defaultdict
from typing import Optional, Dict, Any, Tuple, Type

from langfuse import observe
from app.agents.llm.llm_factory import ModelFactory

from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMAgent:
    """LLM Agent with system prompt stored upfront"""

    def __init__(
        self,
        system_prompt: str,
        provider: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        output_schema: Optional[Type[BaseModel]] = None,
        agent_id: Optional[str] = None,
    ):
        """
        Initialize agent with system prompt

        Args:
            system_prompt: System message (stored for all runs)
            provider: 'openai' or 'groq' or 'gemini'
            model_name: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            output_schema: Optional Pydantic model for structured output
        """
        self.system_prompt = system_prompt
        self.client = ModelFactory.create_client(provider)
        self.model = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.output_schema = output_schema
        self.agent_id = agent_id
        self.execution_time = defaultdict(int)

        logger.info(
            f"Initialized {provider} agent with model: {model_name} and agent_id: {agent_id}"
        )

    @classmethod
    def from_config(
        cls,
        agent_id: str,
        system_prompt: str,
        config: DictConfig,
        output_schema: Optional[Type[BaseModel]] = None,
        max_tokens: int = 4000,
    ):
        """
        Create an LLMAgent from config using agent_id

        Args:
            agent_id: ID of the agent in config (e.g., 'math_teacher_agent')
            system_prompt: System message
            config: Agent-specific config dict
            output_schema: Optional Pydantic model for structured output
            max_tokens: Maximum tokens (defaults to 4000)

        Returns:
            LLMAgent instance
        """
        agent_config = config

        return cls(
            system_prompt=system_prompt,
            provider=agent_config.provider,
            model_name=agent_config.model,
            temperature=agent_config.temperature,
            max_tokens=agent_config.get("max_tokens", max_tokens),
            output_schema=output_schema,
            agent_id=agent_id,
        )

    @observe()
    def run(
        self,
        user_input: str,
        track_tokens: bool = True,
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """
        Execute the LLM agent with user input and return structured output.

        Args:
            user_input: The user message/prompt to send to the LLM
            track_tokens: Whether to return token usage statistics. Defaults to True.

        Returns:
            Tuple containing:
                - output: Either a validated Pydantic model instance (if output_schema
                is set) or raw string content (if no schema)
                - usage: Dictionary with token counts (input_tokens, output_tokens,
                total_tokens) if track_tokens=True, otherwise None
        """
        request_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.output_schema:
            # Check if provider supports structured outputs
            if hasattr(self.client, "beta"):  # OpenAI
                response = self.client.beta.chat.completions.parse(
                    **request_params,
                    response_format=self.output_schema,
                )
                output = response.choices[0].message.parsed
            else:  # Groq or other providers - fallback to json_object
                request_params["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**request_params)
                content = response.choices[0].message.content
                output = self.output_schema.model_validate_json(content)
        else:
            response = self.client.chat.completions.create(**request_params)
            output = response.choices[0].message.content

        usage = (
            {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            if track_tokens
            else None
        )

        return output, usage

    @observe()
    async def arun(
        self,
        user_input: str,
        track_tokens: bool = True,
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """
        Async version of run(). Execute the LLM agent asynchronously with user input.
        """
        request_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.output_schema:
            # Check if provider supports structured outputs
            if hasattr(self.client, "beta"):  # OpenAI
                response = await self.client.beta.chat.completions.parse(
                    **request_params,
                    response_format=self.output_schema,
                )
                output = response.choices[0].message.parsed
            else:  # Groq or other providers - fallback to json_object
                request_params["response_format"] = {"type": "json_object"}
                response = await self.client.chat.completions.create(**request_params)
                content = response.choices[0].message.content
                output = self.output_schema.model_validate_json(content)
        else:
            response = await self.client.chat.completions.create(**request_params)
            output = response.choices[0].message.content

        usage = (
            {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            if track_tokens
            else None
        )

        return output, usage

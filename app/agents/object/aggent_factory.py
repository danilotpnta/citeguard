from pydantic import BaseModel
from typing import Optional, Type

from app.core.config import config
from app.agents.llm.llm_executor import LLMAgent
from app.utils.agents.prompt_loader import load_prompt


class AgentFactory:
    """
    Creates agents with externally managed prompts.
    """

    def __init__(self):
        self.config = config
        self.load_prompt = load_prompt

    def get_agent(
        self,
        agent_id: str,
        workflow: str = "citeguard",
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> LLMAgent:
        """
        Get or create an agent with external prompts.

        Args:
            agent_id: Name of agent from config
            output_schema: Optional Pydantic model for structured output
        """

        agent_config = self.config.pipeline[workflow].nodes.agents[agent_id]
        system_prompt = self.load_prompt(agent_id)["system_prompt"]

        agent_executor = LLMAgent.from_config(
            agent_id=agent_id,
            system_prompt=system_prompt,
            config=agent_config,
            output_schema=output_schema,
        )

        return agent_executor


# Global instance
agent_factory = AgentFactory()

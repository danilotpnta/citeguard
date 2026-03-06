from typing import Optional
from ..misc.yaml import read_yaml
from pathlib import Path
from app.core.config import config


def load_prompt(
    agent_id: str,
    prompt_version: Optional[str] = None,
    workflow: str = "citeguard",
) -> dict:

    agent_config = config.pipeline[workflow].nodes.agents[agent_id]
    prompt_path = agent_config["prompt_path"]
    prompt_version = prompt_version or agent_config.get("prompt_version")
    prompt_file = Path(prompt_path) / f"{prompt_version}.yml"
    prompt_data = read_yaml(prompt_file)

    return prompt_data

from langfuse import observe
from pydantic import BaseModel, Field
from typing import List
from rich import print

from app.agents.object.aggent_factory import agent_factory
from app.utils.agents.prompt_loader import load_prompt


class OutputSchema(BaseModel):
    result: str = Field(description="The final output or answer produced by the agent.")
    reasoning: List[str] = Field(
        description="Step-by-step explanation of how the result was reached."
    )


@observe()
def run_agent(input_data: str) -> OutputSchema:
    AGENT_ID = "your_agent_id"

    prompts = load_prompt(AGENT_ID)
    user_prompt = prompts["user_prompt"].format(input_data=input_data)

    try:
        agent = agent_factory.get_agent(agent_id=AGENT_ID, output_schema=OutputSchema)
        response, _ = agent.run(user_prompt)
        return response
    except Exception as e:
        print(f"[bold red]Agent failed:[/bold red] {e}")


if __name__ == "__main__":
    sample_input = "Replace with your input."

    result = run_agent(sample_input)

    print("\n[bold green]Response successful![/bold green]")
    print(f"\nResult: {result.result}")
    for i, step in enumerate(result.reasoning):
        print(f"Step [{i}]: {step}")

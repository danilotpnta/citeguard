from langfuse import observe

from rich import print
from pydantic import Field
from pydantic import BaseModel
from typing import List

from app.agents.object.aggent_factory import agent_factory
from app.utils.agents.prompt_loader import load_prompt


class AnswerSchema(BaseModel):
    answer: str = Field(
        description="Give the final solution to the problem",
    )
    reasoning: List[str] = Field(
        description="Describe step by step how did you came up with the answer",
    )


@observe()
def answer_math_problem(math_problem: str):

    agent_id = "math_teacher_agent"
    extractor_webpages_prompt = load_prompt(agent_id)
    user_prompt = extractor_webpages_prompt["user_prompt"].format(
        input_data=math_problem,
    )
    try:
        agent = agent_factory.get_agent(
            agent_id=agent_id,
            output_schema=AnswerSchema,
        )
        response, _ = agent.run(user_prompt)
        return response

    except Exception as e:
        print(f"[bold red]Agent failed:[/bold red] {e}")


if __name__ == "__main__":
    math_problem = """
    A school is organizing a field trip. They rent buses that hold 48 students each.
    - There are 365 students going on the trip.
    - 8 teachers must also ride the buses.
    - Every bus must have at least one teacher on it.
    What is the minimum number of buses needed to transport everyone?
    """

    # Answer should be 8: https://chatgpt.com/s/t_69aae98e777881918159e00b2bcacc3c
    result = answer_math_problem(math_problem)

    print("\n[bold green]Response successful![/bold green]")
    print(f"\nResult: {result.answer}")
    for i, step in enumerate(result.reasoning):
        print(f"Step [{i}]: {step}")

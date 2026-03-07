from app.graph.state import WorkflowState

from app.agents.math_teacher import answer_math_problem
from rich import print
from app.core.logging import get_logger
from langfuse import observe

logger = get_logger(__name__)


@observe(name="answer_math_problem_node")
async def extract_ingredients_ai_node(state: WorkflowState) -> dict:

    math_problem = state["math_problem"]

    try:
        result = answer_math_problem(math_problem)
        print("\n[bold green]Response successful![/bold green]")
        print(f"\nResult: {result.answer}")
        for i, step in enumerate(result.reasoning):
            print(f"Step [{i}]: {step}")

        return {}

    except Exception as e:
        logger.error(f"AI failed: {e}", exc_info=True)
        raise ValueError(f"Could answer question: {e}")

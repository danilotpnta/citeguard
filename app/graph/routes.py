from typing import Literal
from app.graph.state import WorkflowState


def decide_content_type(state: WorkflowState) -> Literal["text", "file"]:
    """
    Called after: router_input_node
    Checks:       state["content_type"]
    Returns:
        "text" → goes to gather_all_info_node
        "file" → goes to parse_content_from_file_node
    """
    return "text" if state["content_type"] == "text" else "file"

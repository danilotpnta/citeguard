from typing import Literal
from app.graph.state import WorkflowState


def decide_content_type(state: WorkflowState) -> Literal["text", "file"]:
    """
    Called after: router_input_node
    Checks:       state["input_type"]
    Returns:
        "text" → goes to parse_content_from_file_node
        "file"  → goes to gather_all_info_node
    """
    return state["content_type"]


def decide_file_type(state: WorkflowState) -> Literal["pdf", "doc", "txt"]:
    """
    Called after: identify_url_type_node
    Checks:       state["is_youtube"]
    Returns:
        "pdf" → goes to extract_pdf_node
        "doc" → goes to extract_doc_node
        "txt" → goes to extract_txt_node
    """
    pass


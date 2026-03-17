from app.graph.state import WorkflowState

from app.core.logging import get_logger
from app.agents.tools.parser.extractor import ParserTool
from app.agents.reference_extractor import extract_references
from langfuse import observe

logger = get_logger(__name__)


@observe(name="parse_content_from_file_node")
async def parse_content_from_file_node(state: WorkflowState) -> dict:
    raw_input = state["raw_input"]
    content_type = state.get("content_type", "text/plain")

    text = ParserTool.parse(raw_input, content_type=content_type)
    # print("Content:\n")
    # print(text[:200])

    logger.info(
        "Parsed %s input (%d bytes) into %d characters of text",
        content_type,
        len(raw_input) if isinstance(raw_input, bytes) else len(raw_input.encode()),
        len(text),
    )

    return {"text": text}


@observe(name="extract_references_node")
async def extract_references_node(state: WorkflowState) -> dict:
    raw_text = state.get("text") or state["raw_input"]
    extracted_references = extract_references(raw_text=raw_text)

    return {"extracted_references": extracted_references}

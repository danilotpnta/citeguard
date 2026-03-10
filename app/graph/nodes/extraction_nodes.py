from app.graph.state import WorkflowState

from app.core.logging import get_logger
from app.agents.tools.parser.extractor import ParserTool
from langfuse import observe

logger = get_logger(__name__)


@observe(name="parse_content_from_file_node")
async def parse_content_from_file_node(state: WorkflowState) -> dict:
    raw_input = state["raw_input"]
    content_type = state.get("content_type", "text/plain")

    if content_type == "text/plain":
        text = ParserTool._parse_text(raw_input)
    elif content_type == "application/pdf":
        text = ParserTool._parse_pdf(raw_input)
    elif (
        content_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        text = ParserTool._parse_docx(raw_input)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    logger.info(
        "Parsed %s input (%d bytes) into %d characters of text",
        content_type,
        len(raw_input) if isinstance(raw_input, bytes) else len(raw_input.encode()),
        len(text),
    )

    return {"text": text}

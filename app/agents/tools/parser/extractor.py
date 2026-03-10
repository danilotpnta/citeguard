import io

from app.core.logging import get_logger
from langfuse import observe

logger = get_logger(__name__)


class ParserTool:

    @staticmethod
    @observe(name="_parse_text")
    def _parse_text(raw_input: str | bytes) -> str:
        """Handle plain text input."""
        if isinstance(raw_input, bytes):
            return raw_input.decode("utf-8", errors="ignore")
        return raw_input

    @staticmethod
    @observe(name="_parse_pdf")
    def _parse_pdf(raw_input: bytes) -> str:
        """Extract text from a PDF using PyMuPDF."""
        import fitz  # PyMuPDF

        if isinstance(raw_input, str):
            raw_input = raw_input.encode("utf-8")

        doc = fitz.open(stream=raw_input, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()

        return "\n".join(pages).strip()

    @staticmethod
    @observe(name="_parse_docx")
    def _parse_docx(raw_input: bytes) -> str:
        """Extract text from a DOCX using python-docx."""
        import docx

        if isinstance(raw_input, str):
            raw_input = raw_input.encode("utf-8")

        doc = docx.Document(io.BytesIO(raw_input))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        return "\n".join(paragraphs).strip()

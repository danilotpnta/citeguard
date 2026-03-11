import io
from langfuse import observe
from dotenv import load_dotenv
from app.core.logging import get_logger

load_dotenv()

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

        if not isinstance(raw_input, bytes):
            raise TypeError(f"_parse_pdf expects bytes, got {type(raw_input).__name__}")

        doc = fitz.open(stream=raw_input, filetype="pdf")
        try:
            pages = [page.get_text() for page in doc]
        finally:
            doc.close()

        return "\n".join(pages).strip()

    @staticmethod
    @observe(name="_parse_docx")
    def _parse_docx(raw_input: bytes) -> str:
        """Extract text from a DOCX using python-docx."""
        import docx

        if not isinstance(raw_input, bytes):
            raise TypeError(
                f"_parse_docx expects bytes, got {type(raw_input).__name__}"
            )

        doc = docx.Document(io.BytesIO(raw_input))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        return "\n".join(paragraphs).strip()

    @classmethod
    def parse(cls, raw_input: str | bytes, content_type: str) -> str:
        """Dispatch to the appropriate parser based on content_type."""
        if content_type == "text":
            return cls._parse_text(raw_input)
        elif content_type == "pdf":
            return cls._parse_pdf(raw_input)
        elif content_type == "docx":
            return cls._parse_docx(raw_input)
        else:
            raise ValueError(f"Unsupported content_type: {content_type!r}")


if __name__ == "__main__":
    import pathlib
    import textwrap

    def _section(title: str) -> None:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print("=" * 60)

    # --- _parse_text ---
    _section("_parse_text: str input")
    result = ParserTool._parse_text("Hello, world!")
    print(repr(result))

    _section("_parse_text: bytes input")
    result = ParserTool._parse_text(b"Hello from bytes \xc3\xa9")
    print(repr(result))

    # --- parse dispatcher ---
    _section("parse() dispatcher: text")
    result = ParserTool.parse("Dispatcher test", content_type="text")
    print(repr(result))

    # --- _parse_pdf (requires a real PDF file) ---
    sample_pdf = pathlib.Path("sample.pdf")
    print(sample_pdf)
    if sample_pdf.exists():
        _section("_parse_pdf: sample.pdf")
        result = ParserTool._parse_pdf(sample_pdf.read_bytes())
        print(textwrap.shorten(result, width=200))

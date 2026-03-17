from langfuse import observe

from rich import print
from typing import Optional, List
from pydantic import Field
from pydantic import BaseModel

from app.agents.object.aggent_factory import agent_factory
from app.utils.agents.prompt_loader import load_prompt
from app.models.schemas import ReferenceList


@observe()
def extract_references(raw_text: str):
    AGENT_ID = "extract_references_agent"
    extract_references_prompt = load_prompt(AGENT_ID)
    user_prompt = extract_references_prompt["user_prompt"].format(
        raw_text=raw_text,
    )
    try:
        agent = agent_factory.get_agent(
            agent_id=AGENT_ID,
            output_schema=ReferenceList,
        )
        response, _ = agent.run(user_prompt)

        return response

    except Exception as e:
        print(f"[bold red]Extraction failed:[/bold red] {e}")
        return ReferenceList()


if __name__ == "__main__":
    import pathlib
    from app.agents.tools.parser.extractor import ParserTool

    # sample_md = pathlib.Path("tests/assets/sample_references.md")
    # if sample_md.exists():
    #     raw_text = ParserTool._parse_text(sample_md.read_bytes())

    sample_pdf = pathlib.Path("tests/assets/sample_references_5.pdf")
    print(sample_pdf)
    if sample_pdf.exists():
        raw_text = ParserTool._parse_pdf(sample_pdf.read_bytes())

    references_found = extract_references(raw_text)

    if references_found.references:
        print(f"Found {len(references_found.references)} references")
        for ref in references_found.references:
            print(ref)

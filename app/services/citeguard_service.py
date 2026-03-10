from typing import Optional
from app.schemas.api import CreateRecipeResponse
from app.core.logging import get_logger

from app.graph.citeguard_graph import citeguard_graph
from langfuse import observe, propagate_attributes

logger = get_logger(__name__)


class CiteGuardService:

    @observe(name="verify_from_text")
    async def verify_from_text(
        self,
        raw_input: str,
        token_id: str,
        content_type: str = "text/plain",
        filename: str | None = None,
    ) -> CreateRecipeResponse:

        with propagate_attributes(
            user_id=token_id,
            metadata={
                "content_type": content_type,
            },
            tags=["copied_text"],
        ):
            result = await citeguard_graph.ainvoke(
                {
                    "raw_input": raw_input,
                    "content_type": content_type,
                    "filename": filename,
                }
            )
            return result

    @observe(name="verify_from_file")
    async def verify_from_file(
        self,
        raw_input: bytes,
        token_id: str,
        content_type: str,
        filename: str | None = None,
    ) -> CreateRecipeResponse:

        with propagate_attributes(
            user_id=token_id,
            metadata={
                "content_type": content_type,
            },
            tags=["uploaded_file"],
        ):
            result = await citeguard_graph.ainvoke(
                {
                    "raw_input": raw_input,
                    "content_type": content_type,
                    "filename": filename,
                }
            )
            return result

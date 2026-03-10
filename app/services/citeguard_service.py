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
        text: str,
        user_id: str,
    ) -> CreateRecipeResponse:


        with propagate_attributes(
            user_id=user_id,
            metadata={
                "text": text,
                "input_type": "text",
            },
            tags=["copied_text"],
        ):

            result = await citeguard_graph.ainvoke(
                {
                    "doc_text": text,
                    "input_type": "text",
                    "user_id": user_id,
                }
            )
            
            return result

    
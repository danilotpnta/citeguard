from pathlib import Path

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# Default killswitch file path. Can be overridden for testing.
KILLSWITCH_PATH = Path("/app/KILLSWITCH")


def set_killswitch_path(path: Path) -> None:
    global KILLSWITCH_PATH
    KILLSWITCH_PATH = path


class KillswitchMiddleware(BaseHTTPMiddleware):
    """
    If a KILLSWITCH file exists, reject all requests with 503.
    The /health endpoint is exempt so monitoring tools still work.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Always allow health checks through
        if request.url.path == "/health":
            return await call_next(request)

        if KILLSWITCH_PATH.exists():
            return Response(
                content='{"detail":"Service temporarily unavailable."}',
                status_code=503,
                media_type="application/json",
            )

        return await call_next(request)
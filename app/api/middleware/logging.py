import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.db.usage import log_request

logger = logging.getLogger("citeguard.access")


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request that has an authenticated token attached.
    The token is placed on request.state by get_current_token dependency.

    Also logs request duration to the Python logger for all requests.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000

        # Extract client IP (respects X-Forwarded-For from Cloudflare/proxy)
        ip = _get_client_ip(request)

        # Log to Python logger for all requests
        logger.info(
            "%s %s %s %d %.1fms",
            ip,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        # Log to database only if an authenticated token is present
        token = getattr(request.state, "token", None)
        if token is not None:
            # Estimate input size from content-length header
            input_size = int(request.headers.get("content-length", 0))

            try:
                await log_request(
                    token_id=token.token_id,
                    ip_address=ip,
                    method=request.method,
                    path=request.url.path,
                    input_size=input_size,
                    status_code=response.status_code,
                )
            except Exception:
                # Never let logging failures break the response
                logger.exception("Failed to log request to database")

        return response


def _get_client_ip(request: Request) -> str:
    """Extract client IP, checking proxy headers first."""
    # Cloudflare sets CF-Connecting-IP
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip

    # Standard proxy header
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()

    # Direct connection
    if request.client:
        return request.client.host

    return "unknown"
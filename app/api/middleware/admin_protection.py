from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


class AdminProtectionMiddleware(BaseHTTPMiddleware):
    """
    Hides the /admin routes from requests that came through a Cloudflare proxy.

    Cloudflare sets the CF-Connecting-IP header on every proxied request.
    Direct connections (localhost, Docker internal network, SSH tunnel) never
    carry this header.

    When a /admin request arrives with CF-Connecting-IP present, we return 404
    rather than 401/403 — the route appears not to exist to external callers.

    This middleware is a no-op when BLOCK_ADMIN_VIA_PROXY=false, which lets
    developers hit /admin directly in local dev without any Cloudflare setup.

    For deployments not behind Cloudflare:
    - The /admin routes are still protected by X-Admin-Key authentication.
    - Set BLOCK_ADMIN_VIA_PROXY=false to disable this middleware entirely.
    """

    def __init__(self, app, block_via_proxy: bool = True):
        super().__init__(app)
        self._block_via_proxy = block_via_proxy

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if (
            self._block_via_proxy
            and request.url.path.startswith("/admin")
            and request.headers.get("cf-connecting-ip")
        ):
            return Response(status_code=404)

        return await call_next(request)

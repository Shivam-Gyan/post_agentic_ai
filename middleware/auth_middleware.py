
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from controller.auth.jwt_verify_token import verify_token


class AuthMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Skip auth for public routes
        public_routes = ["/health", "/auth/register", "/auth/verify"]
        # Allow preflight CORS requests through without auth
        if request.method == "OPTIONS":
            return await call_next(request)

        if path in public_routes:
            return await call_next(request)
        
                # Allow streaming route with dynamic thread id
        if path.startswith("/generate"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")

        if not auth_header:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Missing token"}, status_code=401)

        try:
            scheme, token = auth_header.split()

            if scheme.lower() != "bearer":
                raise ValueError("Invalid auth scheme")

            payload = verify_token(token)

            request.state.user = payload

        except Exception:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Invalid token"}, status_code=401)

        response = await call_next(request)

        return response
"""Bound HTTP bodies before JSON parsing; correlate safe errors and structured logs."""

import uuid
from time import perf_counter

from app.api.errors import error_response
from app.core.logging import log_event, request_id


class BoundaryMiddleware:
    def __init__(self, app, max_body_bytes: int):
        self.app, self.max_body_bytes = app, max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        correlation = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = correlation
        token = request_id.set(correlation)
        start = perf_counter()
        status = 500
        started = False

        async def traced_send(message):
            nonlocal status, started
            if message["type"] == "http.response.start":
                started = True
                status = message["status"]
                headers = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() not in (b"x-request-id", b"x-process-time-ms")
                ]
                headers.extend(
                    [
                        (b"x-request-id", correlation.encode()),
                        (b"x-process-time-ms", f"{(perf_counter() - start) * 1000:.3f}".encode()),
                    ]
                )
                message = {**message, "headers": headers}
            await send(message)

        async def reject(code, message, http_status):
            await error_response(http_status, correlation, code, message)(
                scope, receive, traced_send
            )

        try:
            lengths = [v for k, v in scope["headers"] if k.lower() == b"content-length"]
            if lengths:
                if len(lengths) != 1 or not lengths[0].isdigit():
                    return await reject(
                        "invalid_content_length", "Invalid Content-Length header.", 400
                    )
                # Bound conversion too: Python rejects excessively long integer strings.
                normalized_length = lengths[0].lstrip(b"0") or b"0"
                if len(normalized_length) > 6 or int(normalized_length) > self.max_body_bytes:
                    return await reject(
                        "request_too_large", "Request body exceeds the configured byte limit.", 413
                    )
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                if message["type"] != "http.request":
                    continue
                body.extend(message.get("body", b""))
                if len(body) > self.max_body_bytes:
                    return await reject(
                        "request_too_large", "Request body exceeds the configured byte limit.", 413
                    )
                if not message.get("more_body", False):
                    break
            delivered = False

            async def buffered_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, buffered_receive, traced_send)
        except Exception as exc:
            log_event("request_failed", exception_type=type(exc).__name__)
            if started:
                raise
            await reject("internal_error", "An unexpected service error occurred.", 500)
        finally:
            route = scope.get("route")
            log_event(
                "http_request",
                method=scope.get("method"),
                route=getattr(route, "path", "unmatched"),
                status_code=status,
                duration_ms=round((perf_counter() - start) * 1000, 3),
            )
            request_id.reset(token)

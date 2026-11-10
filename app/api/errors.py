from fastapi.responses import JSONResponse


def error_response(
    status: int, request_id: str, code: str, message: str, details: list | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "request_id": request_id,
            "error": {"code": code, "message": message, "details": details or []},
        },
        headers={"X-Request-ID": request_id},
    )

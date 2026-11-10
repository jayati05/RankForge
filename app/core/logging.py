"""JSON events without raw queries, request bodies, paths or exception text."""

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone

request_id: ContextVar[str] = ContextVar("rankforge_request_id", default="")
logger = logging.getLogger("rankforge.service")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "event": record.getMessage(),
                "request_id": getattr(record, "correlation_id", "") or request_id.get(),
                **getattr(record, "fields", {}),
            },
            allow_nan=False,
        )


def configure_logging(level: str = "info") -> None:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False


def log_event(event: str, **fields) -> None:
    logger.info(event, extra={"fields": fields, "correlation_id": request_id.get()})

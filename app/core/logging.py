"""Structured (JSON) logging using only the standard library.

One JSON object per line is easy for machines to parse, which is what makes logs
searchable in CloudWatch Logs Insights later.
"""
import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Extra structured fields are passed as: logger.info("msg", extra={"fields": {...}})
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()  # avoid duplicate handlers if called twice
    root.addHandler(handler)
    root.setLevel(level.upper())

    # We log requests ourselves in the middleware; silence uvicorn's duplicate access log.
    logging.getLogger("uvicorn.access").disabled = True
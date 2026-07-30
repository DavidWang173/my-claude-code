"""Application logging with explicit secret redaction."""

from __future__ import annotations

import logging
from collections.abc import Iterable


class SecretRedactingFilter(logging.Filter):
    def __init__(self, secrets: Iterable[str | None] = ()) -> None:
        super().__init__()
        self._secrets = tuple(secret for secret in secrets if secret)

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for secret in self._secrets:
            message = message.replace(secret, "[REDACTED]")
        record.msg = message
        record.args = ()
        return True


def configure_logging(level: str, secrets: Iterable[str | None] = ()) -> None:
    """Configure the root logger once at the application boundary."""

    handler = logging.StreamHandler()
    handler.addFilter(SecretRedactingFilter(secrets))
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    # Application INFO logs are useful, but HTTP client and event-loop internals
    # make an interactive terminal noisy and can obscure streamed model output.
    for logger_name in ("asyncio", "httpcore", "httpx"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

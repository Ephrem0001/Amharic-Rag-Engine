from __future__ import annotations

import logging
import sys
from loguru import logger


class InterceptHandler(logging.Handler):
    """Redirect stdlib logging to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except Exception:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(log_level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stderr, level=log_level, enqueue=True, backtrace=False, diagnose=False)

    logging.basicConfig(handlers=[InterceptHandler()], level=log_level, force=True)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "sqlalchemy.engine"):
        logging.getLogger(name).handlers = [InterceptHandler()]
        logging.getLogger(name).propagate = False

"""
Logging setup for the Weather Station application.

Provides a pre-configured logger that writes to both a rotating file and stdout.
Import get_logger() from any module to obtain a named logger.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

import config


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given *name*, creating it on first call.

    The root *weather_station* logger is initialised once (file + stdout
    handlers with rotation).  Subsequent calls for the same *name* return the
    cached instance, so configuration is applied only once.

    Args:
        name: Dotted module path used as the logger name, e.g. ``"sensors.bme280"``.

    Returns:
        A :class:`logging.Logger` instance ready to use.
    """
    root_name = "weather_station"
    root_logger = logging.getLogger(root_name)

    if not root_logger.handlers:
        root_logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Rotating file handler
        file_handler = RotatingFileHandler(
            filename=config.LOG_FILE,
            maxBytes=config.LOG_MAX_BYTES,
            backupCount=config.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Stdout handler (INFO and above so the terminal stays readable)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter)
        root_logger.addHandler(stdout_handler)

    return logging.getLogger(f"{root_name}.{name}")

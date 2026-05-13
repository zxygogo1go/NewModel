"""Logging helpers."""

from __future__ import annotations

import logging


def get_logger(name: str = "neck_diffreg") -> logging.Logger:
    """Return a basic console logger."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)

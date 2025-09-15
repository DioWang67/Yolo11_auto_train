import logging
from pathlib import Path
from typing import Optional


def setup_module_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if log_file:
        log_path = Path(log_file)
        exists = any(
            isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_path
            for h in logger.handlers
        )
        if not exists:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(fh)

    root = logging.getLogger()
    if not root.handlers and not logger.handlers:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(sh)

    return logger


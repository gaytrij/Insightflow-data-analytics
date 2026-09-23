import logging
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Log directory
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Log file
LOG_FILE = LOG_DIR / "application.log"


def get_logger(name):
    """
    Create and return a configured application logger.
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers when Streamlit
    # reloads the application.
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger
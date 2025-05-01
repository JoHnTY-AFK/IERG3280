import os
import sys
# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import logging

def setup_logging(log_file="search_engine.log"):
    """Set up logging to track project progress."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger()

def log_message(message, level="info"):
    """Log a message to the log file."""
    logger = setup_logging()
    if level == "info":
        logger.info(message)
    elif level == "error":
        logger.error(message)
    elif level == "warning":
        logger.warning(message)

if __name__ == "__main__":
    log_message("Project initialized.", "info")
import logging
import sys

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a shared logger with a uniform format.
    
    Args:
        name (str): Name of the logger, typically __name__.
        
    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    # Set default level to INFO
    logger.setLevel(logging.INFO)
    
    # Avoid adding handlers multiple times if logger is reused
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # Custom format showing timestamp, level, filename, line number, and message
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

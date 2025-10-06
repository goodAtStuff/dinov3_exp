"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# Global logger registry
_loggers = {}


def setup_logger(name: str = 'dinov3_exp',
                level: int = logging.INFO,
                log_file: Optional[Path] = None,
                console: bool = True) -> logging.Logger:
    """
    Setup and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs to
        console: Whether to output to console
        
    Returns:
        Configured logger
    """
    # Check if logger already exists
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    _loggers[name] = logger
    
    return logger


def get_logger(name: str = 'dinov3_exp') -> logging.Logger:
    """
    Get or create a logger.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    return setup_logger(name)


def set_log_level(level: int, name: str = 'dinov3_exp') -> None:
    """
    Set the logging level for a logger.
    
    Args:
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        name: Logger name
    """
    logger = get_logger(name)
    logger.setLevel(level)
    
    for handler in logger.handlers:
        handler.setLevel(level)


def log_config(config: dict, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        logger: Logger to use (uses default if None)
    """
    if logger is None:
        logger = get_logger()
    
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")


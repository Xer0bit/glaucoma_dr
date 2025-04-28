import logging
import sys
import os

def setup_logging(log_level=logging.DEBUG):
    """Configure logging to both file and console with different levels"""
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Console handler - INFO level and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler - DEBUG level and above
    file_handler = logging.FileHandler('logs/debug.log')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Add both handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info("Logging setup complete - Console: INFO, File: DEBUG")
    
    return logger

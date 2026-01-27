# app/logs/logger.py
import logging
import os

# Log storage path
LOG_DIR = os.path.dirname(__file__)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Logger settings
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # Allow full log level

# console output handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# file save handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)

# Format settings
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Register handlers
if not root_logger.handlers:
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:  
    return logging.getLogger(name)
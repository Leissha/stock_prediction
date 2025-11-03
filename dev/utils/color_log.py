"""
Colored logging using loguru - just import logger directly for cleaner code.
Helper functions kept for backward compatibility.
"""
from loguru import logger

# Configure loguru with clean format and colors
logger.remove()  # Remove default handler
logger.add(
    lambda msg: print(msg, end="", flush=True),
    format="<level>{message}</level>",
    colorize=True,
    level="DEBUG"
)

# Helper functions for convenience (optional - can use logger directly)
cache = lambda msg: logger.info(f"{msg}")
data = lambda msg: logger.info(f"{msg}")
sentiment = lambda msg: logger.info(f"{msg}")
social = lambda msg: logger.info(f"{msg}")
info = logger.info
success = logger.success
warning = logger.warning
error = logger.error


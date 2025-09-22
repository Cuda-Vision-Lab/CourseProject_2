''' Wrapper fpr logging functionality'''

import logging
import traceback

def log_function(func):
    """
    Decorator for logging a method in case of raising an exception
    """
    def try_call_log(*args, **kwargs):
        """
        Calling the function but calling the logger in case an exception is raised
        """
        try:
            message = f"Calling: {func.__name__}..."
            logging.info(message)
            return func(*args, **kwargs)
        except Exception:
            message = traceback.format_exc()
            logging.error(message)
            exit(1)
    return try_call_log
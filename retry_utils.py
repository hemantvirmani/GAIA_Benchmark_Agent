"""Retry utilities with exponential backoff."""

import time
import requests
from typing import Callable, Any
from functools import wraps
import config


def retry_with_backoff(
    max_retries: int = config.MAX_RETRIES,
    initial_delay: float = config.INITIAL_RETRY_DELAY,
    backoff_factor: float = config.RETRY_BACKOFF_FACTOR,
    exceptions: tuple = (requests.RequestException,)
):
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed: {e}")
                        print(f"[RETRY] Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        print(f"[RETRY] All {max_retries} retries exhausted")

            # Re-raise the last exception if all retries failed
            raise last_exception

        return wrapper
    return decorator

import time
from typing import Dict, Tuple
from fastapi import HTTPException

# Store: user_id -> (window_start_time, request_count)
_rate_limits: Dict[str, Tuple[float, int]] = {}

# Configuration
MAX_REQUESTS = 50
WINDOW_SECONDS = 3600  # 1 hour

def check_rate_limit(user_id: str):
    """
    Dependency/Helper to enforce a fixed-window rate limit per user.
    Raises HTTPException 429 if the user exceeds the limit.
    """
    now = time.time()
    if user_id in _rate_limits:
        start_time, count = _rate_limits[user_id]
        if now - start_time < WINDOW_SECONDS:
            if count >= MAX_REQUESTS:
                raise HTTPException(
                    status_code=429, 
                    detail="Rate limit exceeded. Maximum 50 queries per hour."
                )
            _rate_limits[user_id] = (start_time, count + 1)
        else:
            # Window expired, reset
            _rate_limits[user_id] = (now, 1)
    else:
        # First request
        _rate_limits[user_id] = (now, 1)

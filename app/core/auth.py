from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> str:
    """
    FastAPI dependency — verifies the Supabase JWT token on every request.
    Returns the user_id (UUID string) of the authenticated user.
    Raises 401 if token is missing, expired, or invalid.
    """
    token = credentials.credentials
    try:
        from supabase import create_client
        # Use service_role key on backend so we can verify any user's token
        client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_ROLE_KEY
        )
        # Verify token and get user
        response = client.auth.get_user(token)
        if not response or not response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")
        user_id = response.user.id
        logger.info(f"Authenticated user: {user_id}")
        return user_id
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(status_code=401, detail="Could not validate credentials.")

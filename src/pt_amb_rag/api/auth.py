import os
import secrets

from fastapi import Header, HTTPException


async def require_token(authorization: str | None = Header(default=None)):
    expected = os.environ.get("API_TOKEN")
    if not expected:
        return  # dev mode: no token required

    scheme, _, provided = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not provided.strip():
        raise HTTPException(
            status_code=401,
            detail="Missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # compare_digest avoids timing leaks;
    # encode so non-ASCII input can't raise TypeError
    if not secrets.compare_digest(provided.strip().encode(), expected.encode()):
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

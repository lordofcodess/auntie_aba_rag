"""Authentication + anonymous-quota for the FastAPI app.

- Supabase issues JWTs. Modern projects use asymmetric signing (RS256/ES256)
  with a public JWKS endpoint; legacy projects use a shared HS256 secret. We
  support both.
- We verify the token in `get_current_user()` and surface a User object.
- `enforce_quota()` is a FastAPI dependency that:
    * lets authenticated users through unconditionally (per-user quotas live
      elsewhere — Phase 3),
    * rate-limits anonymous callers per-IP. After ANON_QUOTA messages, returns
      HTTP 402 with a clear message so the frontend can open the login modal.

Env vars:
    SUPABASE_URL          e.g. https://<project-ref>.supabase.co — used to
                          derive the JWKS endpoint for asymmetric verification.
                          Required for projects using the new asymmetric flow.
    SUPABASE_JWT_SECRET   Legacy HS256 signing secret (only used if the JWT's
                          alg header is HS256). Find under Dashboard → Project
                          Settings → JWT Keys → Legacy JWT Secret.
    SUPABASE_ISSUER       Optional. Defaults to <SUPABASE_URL>/auth/v1. If
                          neither is set, issuer verification is skipped.
    ANON_QUOTA            Optional. Default 3 — messages per IP before
                          anonymous users are blocked.
    ANON_QUOTA_WINDOW_S   Optional. Default 86400 — quota window in seconds.
    AUTH_QUOTA            Optional. Default 3 — messages per signed-in user
                          before they're rate-limited.
    AUTH_QUOTA_WINDOW_S   Optional. Default 86400 — quota window in seconds.

The anonymous quota uses an in-memory dict — fine for single-instance Modal
or local dev. Upgrade to Redis when you scale horizontally.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import jwt
from fastapi import Header, HTTPException, Request


# ─── Config ──────────────────────────────────────────────────────────────

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/") or None
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
SUPABASE_ISSUER = os.getenv("SUPABASE_ISSUER") or (
    f"{SUPABASE_URL}/auth/v1" if SUPABASE_URL else None
)
ANON_QUOTA = int(os.getenv("ANON_QUOTA", "3"))
ANON_QUOTA_WINDOW_S = int(os.getenv("ANON_QUOTA_WINDOW_S", str(60 * 60 * 24)))  # 24h
AUTH_QUOTA = int(os.getenv("AUTH_QUOTA", "3"))
AUTH_QUOTA_WINDOW_S = int(os.getenv("AUTH_QUOTA_WINDOW_S", str(60 * 60 * 24)))  # 24h


# ─── User type ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class User:
    id: str            # Supabase auth.user id (uuid)
    email: Optional[str]
    role: str          # 'authenticated' for signed-in users


# ─── JWT verification ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _jwks_client() -> Optional[jwt.PyJWKClient]:
    """Cached JWKS client. Re-fetches keys internally per its own lifespan."""
    if not SUPABASE_URL:
        return None
    url = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
    return jwt.PyJWKClient(url, cache_keys=True, lifespan=3600)


def _verify_token(token: str) -> User:
    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Malformed token: {e}")

    alg = header.get("alg", "")

    if alg == "HS256":
        if not SUPABASE_JWT_SECRET:
            raise HTTPException(
                status_code=500,
                detail="Token uses HS256 but SUPABASE_JWT_SECRET is not configured.",
            )
        key: object = SUPABASE_JWT_SECRET
    elif alg in ("RS256", "ES256"):
        client = _jwks_client()
        if not client:
            raise HTTPException(
                status_code=500,
                detail="Token uses asymmetric signing but SUPABASE_URL is not configured.",
            )
        try:
            key = client.get_signing_key_from_jwt(token).key
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Could not fetch signing key: {e}")
    else:
        raise HTTPException(status_code=401, detail=f"Unsupported token algorithm: {alg or 'none'}")

    try:
        payload = jwt.decode(
            token,
            key,
            algorithms=[alg],
            audience="authenticated",      # Supabase sets aud='authenticated' for signed-in users
            issuer=SUPABASE_ISSUER if SUPABASE_ISSUER else None,
            options={"verify_iss": bool(SUPABASE_ISSUER)},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired. Please sign in again.")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid auth token: {e}")
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Token missing subject claim.")
    return User(
        id=sub,
        email=payload.get("email"),
        role=payload.get("role", "authenticated"),
    )


def get_current_user(
    authorization: Optional[str] = Header(default=None),
) -> Optional[User]:
    """Returns the User if Authorization: Bearer <jwt> is valid; None otherwise.

    Does NOT raise for missing header — anonymous calls are allowed here and
    handled downstream by `enforce_quota`.
    """
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        return None
    return _verify_token(parts[1].strip())


# ─── Anonymous IP quota ──────────────────────────────────────────────────

_quota_lock = threading.Lock()
# ip -> list[float] of timestamps within the window
_quota_hits: dict[str, list[float]] = {}
# user_id -> list[float] of timestamps within the window
_auth_hits: dict[str, list[float]] = {}


def _client_ip(request: Request) -> str:
    # Honour standard proxy headers when behind Modal / Vercel / CloudFront
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    return request.client.host if request.client else "unknown"


def _record_anon_hit(ip: str) -> int:
    """Append a hit for ip; return the current count within the window."""
    now = time.time()
    cutoff = now - ANON_QUOTA_WINDOW_S
    with _quota_lock:
        hits = [t for t in _quota_hits.get(ip, []) if t >= cutoff]
        hits.append(now)
        _quota_hits[ip] = hits
        return len(hits)


def _peek_anon_count(ip: str) -> int:
    now = time.time()
    cutoff = now - ANON_QUOTA_WINDOW_S
    with _quota_lock:
        hits = [t for t in _quota_hits.get(ip, []) if t >= cutoff]
        _quota_hits[ip] = hits
        return len(hits)


def _record_auth_hit(user_id: str) -> int:
    now = time.time()
    cutoff = now - AUTH_QUOTA_WINDOW_S
    with _quota_lock:
        hits = [t for t in _auth_hits.get(user_id, []) if t >= cutoff]
        hits.append(now)
        _auth_hits[user_id] = hits
        return len(hits)


def _peek_auth_count(user_id: str) -> int:
    now = time.time()
    cutoff = now - AUTH_QUOTA_WINDOW_S
    with _quota_lock:
        hits = [t for t in _auth_hits.get(user_id, []) if t >= cutoff]
        _auth_hits[user_id] = hits
        return len(hits)


def enforce_quota(
    request: Request,
    user: Optional[User] = None,
) -> Optional[User]:
    """FastAPI dependency. Returns the resolved user (or None for anonymous).

    - Anonymous users: in-memory IP quota. Returns 402 once over the cap so the
      frontend knows to prompt for sign-in.
    - Authenticated users: in-memory per-user-id quota. Returns 429 once over
      the cap.
    """
    if user is not None:
        count = _record_auth_hit(user.id)
        if count > AUTH_QUOTA:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "user_quota_exceeded",
                    "message": (
                        f"You've used your {AUTH_QUOTA} messages for today. "
                        "Try again later."
                    ),
                    "limit": AUTH_QUOTA,
                    "used": count,
                },
            )
        return user
    ip = _client_ip(request)
    count = _record_anon_hit(ip)
    if count > ANON_QUOTA:
        raise HTTPException(
            status_code=402,
            detail={
                "code": "anonymous_quota_exceeded",
                "message": (
                    f"You've used your {ANON_QUOTA} free messages. "
                    "Sign in with Google to keep going — it's free."
                ),
                "limit": ANON_QUOTA,
                "used": count,
            },
        )
    return None


def quota_status(request: Request, user: Optional[User] = None) -> dict:
    """Read-only helper for the frontend so it can show 'X / 3 free' before a call."""
    if user is not None:
        used = _peek_auth_count(user.id)
        return {
            "authenticated": True,
            "limit": AUTH_QUOTA,
            "used": used,
            "remaining": max(0, AUTH_QUOTA - used),
        }
    ip = _client_ip(request)
    used = _peek_anon_count(ip)
    return {
        "authenticated": False,
        "limit": ANON_QUOTA,
        "used": used,
        "remaining": max(0, ANON_QUOTA - used),
    }

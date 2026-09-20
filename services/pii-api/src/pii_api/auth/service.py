"""Authentication: bootstrap, login, sessions.

The session token is generated here, hashed, and only the hash is stored. The
plaintext exists exactly once, in the Set-Cookie header on the login response.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Final

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from pii_api.auth.passwords import hash_password, verify_password
from pii_api.db.models import AppSession, AppUser
from pii_api.settings import Settings

__all__ = [
    "bootstrap_admin",
    "change_password",
    "create_session",
    "destroy_session",
    "login",
    "session_user",
]

logger: Final = structlog.get_logger(__name__)

_TOKEN_BYTES: Final = 32


def _token_hash(token: str) -> str:
    """SHA-256, not scrypt.

    A session token is 256 bits of CSPRNG output, not a human-chosen password,
    so there is nothing to brute-force and a slow KDF would only add latency to
    every authenticated request.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def bootstrap_admin(session: AsyncSession, settings: Settings) -> None:
    """Create the first admin from the environment, once.

    Idempotent, and deliberately does **not** reset the password of an existing
    account. A restart with the bootstrap variables still set must not silently
    revert a rotated admin password back to the one in the install guide.
    """
    email = settings.admin_email.strip().lower()
    existing = await session.scalar(select(AppUser).where(AppUser.email == email))
    if existing is not None:
        return

    session.add(
        AppUser(
            email=email,
            password_hash=hash_password(settings.admin_initial_password.get_secret_value()),
            # The whole point of the bootstrap flow: this account can log in
            # and can do nothing else until the password is changed.
            must_change_password=True,
        )
    )
    await session.commit()
    logger.info("auth.admin_bootstrapped", email=email, must_change_password=True)


async def login(session: AsyncSession, email: str, password: str) -> AppUser | None:
    """Verify credentials. Returns None for every kind of failure.

    One indistinguishable outcome for "no such user", "wrong password" and
    "disabled account", and the hash is computed even when the user does not
    exist, so response timing does not disclose which emails have accounts.
    """
    user = await session.scalar(select(AppUser).where(AppUser.email == email.strip().lower()))

    if user is None:
        # A real scrypt run against a throwaway hash, so a missing account
        # costs the same wall-clock time as a wrong password.
        verify_password(password, hash_password("timing-equalizer"))
        return None

    if not verify_password(password, user.password_hash) or user.disabled:
        return None

    user.last_login_at = datetime.now(UTC)
    await session.commit()
    return user


async def create_session(session: AsyncSession, user: AppUser, settings: Settings) -> str:
    token = secrets.token_urlsafe(_TOKEN_BYTES)
    session.add(
        AppSession(
            token_hash=_token_hash(token),
            user_id=user.id,
            expires_at=datetime.now(UTC) + timedelta(hours=settings.session_ttl_hours),
        )
    )
    await session.commit()
    return token


async def session_user(session: AsyncSession, token: str | None) -> AppUser | None:
    """Resolve a cookie value to a user, or None."""
    if not token:
        return None

    row = await session.scalar(
        select(AppSession).where(AppSession.token_hash == _token_hash(token))
    )
    if row is None:
        return None

    expires_at = row.expires_at
    if expires_at.tzinfo is None:  # SQLite round-trips naive datetimes
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at <= datetime.now(UTC):
        await session.delete(row)
        await session.commit()
        return None

    user = await session.get(AppUser, row.user_id)
    return None if user is None or user.disabled else user


async def destroy_session(session: AsyncSession, token: str | None) -> None:
    if not token:
        return
    await session.execute(delete(AppSession).where(AppSession.token_hash == _token_hash(token)))
    await session.commit()


async def change_password(session: AsyncSession, user: AppUser, new_password: str) -> None:
    """Rotate a password and clear the must-change flag.

    **Every** session for this user is destroyed, including the caller's. If
    the reason for the rotation was that someone else knew the old password,
    leaving their session alive would make the rotation ceremonial -- and
    there is no way to tell their session from this one.

    The caller is therefore responsible for issuing a fresh session
    afterwards; see the change-password route, which does exactly that. That
    also rotates the session token on a privilege change, which is the
    standard defence against session fixation.
    """
    user.password_hash = hash_password(new_password)
    user.must_change_password = False
    await session.execute(delete(AppSession).where(AppSession.user_id == user.id))
    await session.commit()
    logger.info("auth.password_changed", user_id=user.id)

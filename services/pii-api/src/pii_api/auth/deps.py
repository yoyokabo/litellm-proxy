"""Request dependencies: who is calling, and may they proceed.

One permission level in this build: authenticated or not. There is no role
check because there are no roles (see db/models.py). What *is* enforced is the
password-rotation gate from brief §10 -- the bootstrap admin can log in and
change its password, and nothing else, until it has.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from pii_api.auth.service import session_user
from pii_api.db.models import AppUser
from pii_api.settings import Settings

__all__ = ["CurrentUser", "DbSession", "RotatedUser", "get_db", "get_settings_dep"]


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


async def get_db(request: Request) -> AsyncSession:
    """One session per request, closed by the middleware that opened it."""
    return request.state.db  # type: ignore[no-any-return]


DbSession = Annotated[AsyncSession, Depends(get_db)]


async def current_user(request: Request, db: DbSession) -> AppUser:
    settings: Settings = request.app.state.settings
    user = await session_user(db, request.cookies.get(settings.cookie_name))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not authenticated")
    return user


CurrentUser = Annotated[AppUser, Depends(current_user)]


async def rotated_user(user: CurrentUser) -> AppUser:
    """A user who has rotated the bootstrap password.

    Every route except login, logout, /me and change-password depends on this.
    Brief §10: admin routes refuse to serve until the initial password is
    rotated. A 403 with a machine-readable code rather than a redirect, so the
    web app can route to the change-password screen and a script gets a reason
    it can branch on.
    """
    if user.must_change_password:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail={
                "code": "password_change_required",
                "message": (
                    "This account still uses its bootstrap password. Change it before "
                    "using the admin or chat routes."
                ),
            },
        )
    return user


RotatedUser = Annotated[AppUser, Depends(rotated_user)]

"""Authentication routes."""

from __future__ import annotations

from typing import Final

import structlog
from fastapi import APIRouter, HTTPException, Request, Response, status

from pii_api.api.schemas import LoginRequest, PasswordChangeRequest, UserView
from pii_api.auth.deps import CurrentUser, DbSession
from pii_api.auth.passwords import verify_password
from pii_api.auth.service import change_password, create_session, destroy_session, login

__all__ = ["router"]

logger: Final = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _set_cookie(response: Response, request: Request, token: str) -> None:
    settings = request.app.state.settings
    response.set_cookie(
        settings.cookie_name,
        token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        max_age=settings.session_ttl_hours * 3600,
        path="/",
    )


@router.post("/login", response_model=UserView)
async def login_route(
    payload: LoginRequest, request: Request, response: Response, db: DbSession
) -> UserView:
    user = await login(db, payload.email, payload.password)
    if user is None:
        # No detail about which half failed. The log line records the attempt
        # without the password and without confirming the address exists.
        logger.info("auth.login_failed")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")

    token = await create_session(db, user, request.app.state.settings)
    _set_cookie(response, request, token)
    logger.info("auth.login", user_id=user.id, must_change_password=user.must_change_password)
    return UserView(id=user.id, email=user.email, must_change_password=user.must_change_password)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout_route(request: Request, response: Response, db: DbSession) -> None:
    settings = request.app.state.settings
    await destroy_session(db, request.cookies.get(settings.cookie_name))
    response.delete_cookie(settings.cookie_name, path="/")


@router.get("/me", response_model=UserView)
async def me(user: CurrentUser) -> UserView:
    """Deliberately reachable before the password rotation.

    The web app calls this on load to decide which screen to show, and it
    cannot route someone to the change-password screen if the endpoint that
    reports must_change_password is itself gated on having changed it.
    """
    return UserView(id=user.id, email=user.email, must_change_password=user.must_change_password)


@router.post("/password", response_model=UserView)
async def change_password_route(
    payload: PasswordChangeRequest,
    user: CurrentUser,
    db: DbSession,
    request: Request,
    response: Response,
) -> UserView:
    """Rotate a password. Also ungates the account.

    Requires the current password even though the caller is already
    authenticated: a session cookie proves someone has the browser, not that
    they know the credential being replaced.
    """
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "current password is incorrect")
    if payload.new_password == payload.current_password:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "the new password must differ from the current one"
        )

    await change_password(db, user, payload.new_password)

    # change_password destroys every session for this user, this one included.
    # Issue a new one so the rotation does not log the person out halfway
    # through it -- and so the token itself rotates on a credential change.
    token = await create_session(db, user, request.app.state.settings)
    _set_cookie(response, request, token)
    return UserView(id=user.id, email=user.email, must_change_password=False)

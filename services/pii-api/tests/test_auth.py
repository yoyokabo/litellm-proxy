"""Auth, sessions, and the bootstrap rotation gate (brief §10)."""

from __future__ import annotations

from fastapi.testclient import TestClient
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, NEW_PASSWORD

from pii_api.auth.passwords import hash_password, needs_rehash, verify_password


def test_password_round_trip() -> None:
    encoded = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", encoded)
    assert not verify_password("wrong", encoded)
    assert not needs_rehash(encoded)


def test_a_corrupt_hash_fails_the_login_rather_than_raising() -> None:
    """A 500 here would tell an unauthenticated caller the account exists."""
    for broken in ("", "not-a-hash", "scrypt$bad$8$1$zz$zz", "bcrypt$1$2$3$4$5"):
        assert verify_password("anything", broken) is False


def test_bootstrap_admin_must_change_password(
    app_client: tuple[TestClient, object, object],
) -> None:
    client, _, _ = app_client
    response = client.post(
        "/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    )
    assert response.status_code == 200
    assert response.json()["must_change_password"] is True


def test_admin_routes_refuse_until_the_password_is_rotated(
    app_client: tuple[TestClient, object, object],
) -> None:
    """The gate brief §10 asks for, and the reason it is a code not a redirect."""
    client, _, _ = app_client
    client.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})

    blocked = client.get("/api/admin/events")
    assert blocked.status_code == 403
    assert blocked.json()["detail"]["code"] == "password_change_required"

    # /me stays reachable, or the web app cannot discover why it was refused.
    assert client.get("/api/auth/me").status_code == 200


def test_rotating_the_password_opens_the_gate_and_keeps_the_session(
    app_client: tuple[TestClient, object, object],
) -> None:
    client, _, _ = app_client
    client.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})

    rotated = client.post(
        "/api/auth/password",
        json={"current_password": ADMIN_PASSWORD, "new_password": NEW_PASSWORD},
    )
    assert rotated.status_code == 200
    assert rotated.json()["must_change_password"] is False

    # A fresh session cookie was issued, so the caller is not logged out by
    # the very act of fixing their password.
    assert client.get("/api/auth/me").status_code == 200


def test_rotation_requires_the_current_password(
    app_client: tuple[TestClient, object, object],
) -> None:
    """A session cookie proves browser access, not knowledge of the credential."""
    client, _, _ = app_client
    client.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})

    response = client.post(
        "/api/auth/password",
        json={"current_password": "not-the-password", "new_password": NEW_PASSWORD},
    )
    assert response.status_code == 403


def test_the_old_password_stops_working(app_client: tuple[TestClient, object, object]) -> None:
    client, _, _ = app_client
    client.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    client.post(
        "/api/auth/password",
        json={"current_password": ADMIN_PASSWORD, "new_password": NEW_PASSWORD},
    )
    client.post("/api/auth/logout")

    assert (
        client.post(
            "/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/auth/login", json={"email": ADMIN_EMAIL, "password": NEW_PASSWORD}
        ).status_code
        == 200
    )


def test_unknown_account_and_wrong_password_are_indistinguishable(
    app_client: tuple[TestClient, object, object],
) -> None:
    client, _, _ = app_client
    missing = client.post(
        "/api/auth/login", json={"email": "nobody@example.test", "password": "x" * 12}
    )
    wrong = client.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": "x" * 12})

    assert missing.status_code == wrong.status_code == 401
    assert missing.json() == wrong.json()


def test_logout_revokes_the_session(app_client: tuple[TestClient, object, object]) -> None:
    client, _, _ = app_client
    client.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401


def test_unauthenticated_requests_are_refused(
    app_client: tuple[TestClient, object, object],
) -> None:
    client, _, _ = app_client
    for path in ("/api/auth/me", "/api/admin/events", "/api/admin/timeline"):
        assert client.get(path).status_code == 401

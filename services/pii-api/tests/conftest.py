"""Shared fixtures.

The suite runs against SQLite with stubbed upstreams. That covers auth,
sessions, the rotation gate and the chat's control flow -- everything whose
correctness is this service's own logic.

The admin *queries* are deliberately not covered here: they use date_trunc,
FILTER (WHERE ...) and ILIKE, which SQLite does not have. Testing them against
a fake would prove the fake works, so they live in test_admin_queries.py and
skip unless PII_API_TEST_DATABASE_URL points at a real PostgreSQL -- the same
discipline pii-service uses.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("PII_API_ADMIN_INITIAL_PASSWORD", "bootstrap-password-1")
os.environ.setdefault("PII_API_LITELLM_MASTER_KEY", "sk-test-master")

ADMIN_EMAIL = "admin@example.test"
ADMIN_PASSWORD = "bootstrap-password-1"
NEW_PASSWORD = "a-much-longer-rotated-password"


class FakePiiService:
    """Stands in for pii-service. Records what it was asked to analyse."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = False
        self.blocked = False

    async def analyze(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        if self.fail:
            raise RuntimeError("pii-service is down")
        self.calls.append({"texts": texts, **kwargs})
        text = texts[0]
        nid = "28503122196078"
        masked = text.replace(nid, "<EG_NATIONAL_ID>")
        findings = []
        if nid in text:
            start = text.index(nid)
            findings.append(
                {
                    "text_index": 0,
                    "entity_type": "EG_NATIONAL_ID",
                    "category": "id",
                    "recognizer": "EgyptianNationalIdRecognizer",
                    "score": 1.0,
                    "action": "BLOCK" if self.blocked else "MASK",
                    "start": start,
                    "end": start + len(nid),
                    "value_len": len(nid),
                }
            )
        return {
            "request_id": kwargs.get("request_id", "r"),
            "texts": [masked],
            "findings": findings,
            "entity_counts": {"EG_NATIONAL_ID": 1} if findings else {},
            "blocked": self.blocked and bool(findings),
            "block_reason": {"EG_NATIONAL_ID": 1} if (self.blocked and findings) else None,
            "latency_ms": 3,
        }

    async def policy(self) -> dict[str, Any]:
        return {
            "version": 1,
            "entities": [
                {
                    "entity_type": "EG_NATIONAL_ID",
                    "category": "id",
                    "action": "MASK",
                    "tier": 1,
                    "score_threshold": 0.8,
                    "placeholder": "<EG_NATIONAL_ID>",
                }
            ],
        }

    async def aclose(self) -> None:
        return None


class FakeLiteLlm:
    """Stands in for the proxy. Captures the messages it was handed."""

    def __init__(self) -> None:
        self.sent: list[list[dict[str, str]]] = []
        self.keys_generated: list[str] = []

    async def generate_key(self, llm_user_id: str, alias: str) -> str:
        self.keys_generated.append(llm_user_id)
        return f"sk-virtual-{llm_user_id}"

    async def stream_chat(
        self, *, key: str, model: str, messages: list[dict[str, str]], end_user_id: str
    ) -> AsyncIterator[str]:
        self.sent.append(messages)
        for piece in ("Understood", ", ", "noted."):
            yield piece

    async def aclose(self) -> None:
        return None


@pytest.fixture
def app_client(tmp_path: Path) -> Iterator[tuple[TestClient, FakePiiService, FakeLiteLlm]]:
    from pii_api.db.models import Base
    from pii_api.db.session import Database
    from pii_api.main import create_app
    from pii_api.settings import Settings

    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'api.db'}",
        admin_email=ADMIN_EMAIL,
        admin_initial_password=ADMIN_PASSWORD,  # type: ignore[arg-type]
        litellm_master_key="sk-test-master",  # type: ignore[arg-type]
        cookie_secure=False,
    )
    database = Database(settings.database_url)

    import asyncio

    async def _create() -> None:
        async with database.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_create())

    app = create_app(settings, database=database)
    pii, litellm = FakePiiService(), FakeLiteLlm()
    app.state.pii = pii
    app.state.litellm = litellm

    with TestClient(app) as client:
        yield client, pii, litellm


@pytest.fixture
def logged_in(
    app_client: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> tuple[TestClient, FakePiiService, FakeLiteLlm]:
    """A client past both the login and the bootstrap-rotation gate."""
    client, pii, litellm = app_client
    client.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    client.post(
        "/api/auth/password",
        json={"current_password": ADMIN_PASSWORD, "new_password": NEW_PASSWORD},
    )
    return client, pii, litellm

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

        # Entity administration. `admin_token` empty models a deployment that
        # never set PII_API_PII_ADMIN_TOKEN; `admin_error` models pii-service
        # refusing a change.
        self.admin_token = "service-to-service-token"
        self.admin_error: tuple[int, Any] | None = None
        self.admin_calls: list[dict[str, Any]] = []
        self.overlay_entities: list[dict[str, Any]] = []

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

    # -- entity administration --------------------------------------------
    #
    # The real client forwards these to pii-service, which owns every rule
    # about what a valid overlay is. The fake therefore records what it was
    # asked and can be told to fail: what these tests check is this backend's
    # own contribution -- authorisation, attribution, and how an upstream
    # refusal reaches the operator.

    async def admin_policy(self) -> dict[str, Any]:
        self._require_admin()
        return {
            "entities": list(self.overlay_entities),
            "tier3_labels": {"internal project codename": "PROJECT_CODENAME"},
            "tier3_enabled": False,
            "realistic_replacement_entities": [],
            "warnings": [],
        }

    async def replacement_strategies(self) -> dict[str, Any]:
        self._require_admin()
        return {"strategies": [{"name": "placeholder", "example": "<PERSON>", "realistic": False}]}

    async def upsert_entity(
        self, entity_type: str, payload: dict[str, Any], *, acting_user: str
    ) -> dict[str, Any]:
        self._require_admin()
        self.admin_calls.append(
            {"method": "PUT", "entity_type": entity_type, "payload": payload, "by": acting_user}
        )
        self.overlay_entities.append({"entity_type": entity_type, **payload})
        return await self.admin_policy()

    async def delete_entity(self, entity_type: str, *, acting_user: str) -> dict[str, Any]:
        self._require_admin()
        self.admin_calls.append({"method": "DELETE", "entity_type": entity_type, "by": acting_user})
        self.overlay_entities = [
            e for e in self.overlay_entities if e["entity_type"] != entity_type
        ]
        return await self.admin_policy()

    def _require_admin(self) -> None:
        from pii_api.upstream import AdminDisabled, UpstreamError

        if not self.admin_token:
            raise AdminDisabled
        if self.admin_error is not None:
            raise UpstreamError(*self.admin_error)

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

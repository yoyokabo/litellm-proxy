"""API-level tests against the real FastAPI app.

The app is built with a SQLite-backed sink so the audit path runs for real
rather than being mocked out -- a test that stubs the sink would not notice the
service forgetting to record.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import anyio
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import create_async_engine

from conftest import CONFIG_DIR
from pii_service.audit.sink import AuditSink
from pii_service.db.models import Base, PiiEvent
from pii_service.main import create_app
from pii_service.settings import Settings
from pii_service.synthetic import (
    synthetic_mobile,
    synthetic_national_id,
    to_arabic_indic,
)

PEPPER = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def app_client(tmp_path: Path) -> Iterator[tuple[TestClient, object]]:
    """The real app, with a file-backed SQLite audit database.

    A file rather than ":memory:" so every connection -- the one that creates
    the tables and the sink's own -- sees the same database without pool
    gymnastics.
    """
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'audit.db'}"
    settings = Settings(
        audit_pepper=PEPPER,  # type: ignore[arg-type]
        config_dir=CONFIG_DIR,
        database_url=database_url,
        audit_wal_path=tmp_path / "wal",
        audit_flush_interval_seconds=0.05,
    )

    engine = create_async_engine(database_url)

    async def _create_tables() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    anyio.run(_create_tables)

    app = create_app(settings, sink=AuditSink(settings, engine=engine))
    with TestClient(app) as client:
        yield client, engine


def _analyze(client: TestClient, **kwargs: object) -> dict[str, object]:
    payload = {"request_id": "req-1", "texts": [], **kwargs}
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_analyze_masks_a_national_id(app_client: tuple[TestClient, object]) -> None:
    client, _ = app_client
    nid = synthetic_national_id(birth_date=date(1985, 3, 12), governorate_code="21", serial=4821)

    body = _analyze(client, texts=[f"my id is {nid}"])

    assert body["texts"] == ["my id is <EG_NATIONAL_ID>"]
    assert body["entity_counts"] == {"EG_NATIONAL_ID": 1}
    assert body["blocked"] is False


def test_analyze_handles_mixed_arabic_and_english(
    app_client: tuple[TestClient, object],
) -> None:
    client, _ = app_client
    nid = synthetic_national_id(birth_date=date(1990, 1, 1), governorate_code="01", serial=7)
    mobile = synthetic_mobile(prefix="011", rng=random.Random(3))

    body = _analyze(
        client,
        texts=[f"الرقم القومي {to_arabic_indic(nid)}", f"call {mobile} please"],
    )

    assert "<EG_NATIONAL_ID>" in body["texts"][0]
    assert "<EG_MOBILE>" in body["texts"][1]
    assert to_arabic_indic(nid) not in body["texts"][0]
    assert {f["text_index"] for f in body["findings"]} == {0, 1}


def test_response_order_and_length_match_the_request(
    app_client: tuple[TestClient, object],
) -> None:
    """The guardrail maps texts back positionally -- a reorder corrupts messages."""
    client, _ = app_client
    nid = synthetic_national_id(rng=random.Random(11))
    texts = ["nothing here", f"id {nid}", "", "also nothing"]

    body = _analyze(client, texts=texts)

    assert len(body["texts"]) == len(texts)
    assert body["texts"][0] == "nothing here"
    assert body["texts"][2] == ""
    assert body["texts"][3] == "also nothing"
    assert "<EG_NATIONAL_ID>" in body["texts"][1]


def test_findings_never_contain_the_matched_value(
    app_client: tuple[TestClient, object],
) -> None:
    client, _ = app_client
    nid = synthetic_national_id(rng=random.Random(13))
    response = client.post("/analyze", json={"request_id": "r", "texts": [f"id {nid}"]})
    body = response.json()

    serialized = response.text
    assert nid not in serialized
    for finding in body["findings"]:
        assert "value" not in finding
        # The preview is policy-governed and must not be the whole value.
        if finding["preview"]:
            assert finding["preview"] != nid
            assert "*" in finding["preview"]


def test_context_is_opt_in_and_off_by_default(
    app_client: tuple[TestClient, object],
) -> None:
    client, _ = app_client
    nid = synthetic_national_id(rng=random.Random(17))

    default = _analyze(client, texts=[f"my id is {nid} ok"])
    assert all(f["context"] is None for f in default["findings"])

    with_context = _analyze(
        client, texts=[f"my id is {nid} ok"], include_context=True, bypass_cache=True
    )
    finding = with_context["findings"][0]
    assert finding["context"] is not None
    assert nid in finding["context"]  # the caller asked to see their own text
    assert finding["context_offset"] is not None


def test_audit_rows_are_written(app_client: tuple[TestClient, object]) -> None:
    client, engine = app_client
    nid = synthetic_national_id(rng=random.Random(19))
    mobile = synthetic_mobile(rng=random.Random(19))

    _analyze(
        client,
        texts=[f"id {nid} phone {mobile}"],
        identity={"user_id": "eng-42", "team_id": "platform"},
        model="qwen3-30b",
        request_id="call-abc",
    )

    async def _read() -> list[PiiEvent]:
        # Give the background flusher a moment.
        await anyio.sleep(0.4)
        async with engine.connect() as connection:  # type: ignore[attr-defined]
            result = await connection.execute(select(PiiEvent))
            return list(result.mappings())

    rows = anyio.run(_read)

    assert len(rows) == 2
    assert {r["entity_type"] for r in rows} == {"EG_NATIONAL_ID", "EG_MOBILE"}
    for row in rows:
        assert row["user_id"] == "eng-42"
        assert row["team_id"] == "platform"
        assert row["request_id"] == "call-abc"
        assert row["model"] == "qwen3-30b"
        assert row["value_fp"] is not None
        assert row["action"] == "MASK"


def test_no_audit_row_contains_the_value(app_client: tuple[TestClient, object]) -> None:
    client, engine = app_client
    nid = synthetic_national_id(rng=random.Random(23))
    _analyze(client, texts=[f"id {nid}"], request_id="call-xyz")

    async def _read() -> list[dict[str, object]]:
        await anyio.sleep(0.4)
        async with engine.connect() as connection:  # type: ignore[attr-defined]
            result = await connection.execute(select(PiiEvent))
            return [dict(r) for r in result.mappings()]

    rows = anyio.run(_read)
    assert rows
    for row in rows:
        for value in row.values():
            assert nid not in str(value)


def test_cache_hit_is_reported_and_still_masks(
    app_client: tuple[TestClient, object],
) -> None:
    client, _ = app_client
    nid = synthetic_national_id(rng=random.Random(29))
    text = f"repeat id {nid}"

    first = _analyze(client, texts=[text])
    second = _analyze(client, texts=[text])

    assert first["cached"] is False
    assert second["cached"] is True
    assert second["texts"] == first["texts"]
    assert second["entity_counts"] == first["entity_counts"]


def test_cache_hit_still_writes_audit_rows(app_client: tuple[TestClient, object]) -> None:
    """A cached verdict must not mean a missing audit trail."""
    client, engine = app_client
    nid = synthetic_national_id(rng=random.Random(31))
    text = f"cached id {nid}"

    _analyze(client, texts=[text], request_id="first")
    _analyze(client, texts=[text], request_id="second")

    async def _count() -> int:
        await anyio.sleep(0.4)
        async with engine.connect() as connection:  # type: ignore[attr-defined]
            result = await connection.execute(select(func.count()).select_from(PiiEvent))
            return int(result.scalar_one())

    assert anyio.run(_count) == 2


def test_empty_texts_is_accepted(app_client: tuple[TestClient, object]) -> None:
    client, _ = app_client
    body = _analyze(client, texts=[])
    assert body["texts"] == []
    assert body["findings"] == []
    assert body["blocked"] is False


def test_unknown_field_is_rejected(app_client: tuple[TestClient, object]) -> None:
    client, _ = app_client
    response = client.post("/analyze", json={"request_id": "r", "texts": ["x"], "surprise": True})
    assert response.status_code == 422


def test_missing_request_id_is_rejected(app_client: tuple[TestClient, object]) -> None:
    client, _ = app_client
    assert client.post("/analyze", json={"texts": ["x"]}).status_code == 422


def test_livez_never_touches_the_database(app_client: tuple[TestClient, object]) -> None:
    client, _ = app_client
    assert client.get("/livez").json() == {"status": "ok"}


def test_health_reports_tiers_and_audit(app_client: tuple[TestClient, object]) -> None:
    client, _ = app_client
    body = client.get("/health").json()

    assert body["status"] in ("ok", "degraded")
    assert body["tiers"]["tier1_patterns"] is True
    assert body["tiers"]["tier2_arabic_ner"] is False
    assert body["tiers"]["tier3_gliner"] is False
    assert "submitted" in body["audit"]


def test_docs_are_not_exposed(app_client: tuple[TestClient, object]) -> None:
    client, _ = app_client
    assert client.get("/docs").status_code == 404


# ---------------------------------------------------------------------------
# /policy
# ---------------------------------------------------------------------------


def test_policy_lists_entities_with_their_categories(
    app_client: tuple[TestClient, object],
) -> None:
    """The admin timeline stacks by category and pii_events stores only a type.

    This endpoint is the mapping between them, and it exists so the web
    backend does not carry a second copy of entities.yaml.
    """
    client, _ = app_client
    response = client.get("/policy")
    assert response.status_code == 200
    body = response.json()

    assert body["version"] >= 1
    by_type = {entity["entity_type"]: entity for entity in body["entities"]}

    assert by_type["EG_NATIONAL_ID"]["category"] == "id"
    assert by_type["EG_NATIONAL_ID"]["action"] == "MASK"
    assert by_type["EG_NATIONAL_ID"]["placeholder"] == "<EG_NATIONAL_ID>"
    # ALLOW is a real action the UI must render differently from MASK.
    assert by_type["AR_ORG"]["action"] == "ALLOW"


def test_policy_covers_every_category_the_ui_colours(
    app_client: tuple[TestClient, object],
) -> None:
    """Brief §9 gives one colour per category; an unmapped one renders grey."""
    client, _ = app_client
    categories = {entity["category"] for entity in client.get("/policy").json()["entities"]}
    assert {"id", "person", "contact", "location", "finance"} <= categories

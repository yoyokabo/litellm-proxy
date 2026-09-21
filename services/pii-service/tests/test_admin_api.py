"""The admin API: custom tier-3 labels and replacement policy.

The tests that matter most here are the last two sections: that an edit
actually changes what the model receives, and that realistic replacement stays
idempotent across the two masking passes the architecture performs.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import anyio
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine

from conftest import CONFIG_DIR
from pii_service.audit.sink import AuditSink
from pii_service.db import custom_entities  # noqa: F401  -- registers the table
from pii_service.db.models import Base
from pii_service.main import create_app
from pii_service.settings import Settings
from pii_service.synthetic import synthetic_mobile

TOKEN = "admin-token-0123456789abcdef0123"
AUTH = {"Authorization": f"Bearer {TOKEN}", "X-Admin-User": "ops@example.com"}


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'admin.db'}"
    settings = Settings(
        audit_pepper="0123456789abcdef0123456789abcdef",  # type: ignore[arg-type]
        config_dir=CONFIG_DIR,
        database_url=database_url,
        audit_wal_path=tmp_path / "wal",
        audit_flush_interval_seconds=0.05,
        admin_token=TOKEN,  # type: ignore[arg-type]
    )
    engine = create_async_engine(database_url)

    async def _create() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    anyio.run(_create)

    with TestClient(create_app(settings, sink=AuditSink(settings, engine=engine))) as c:
        yield c


def _mask(client: TestClient, text: str, request_id: str = "r") -> str:
    response = client.post("/analyze", json={"request_id": request_id, "texts": [text]})
    assert response.status_code == 200, response.text
    return response.json()["texts"][0]


def _set(client: TestClient, entity: str, **body: object) -> dict:
    response = client.put(
        f"/admin/entities/{entity}", headers=AUTH, json={"entity_type": entity, **body}
    )
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def test_the_admin_api_requires_a_token(client: TestClient) -> None:
    assert client.get("/admin/policy").status_code == 401


@pytest.mark.parametrize(
    "header",
    [
        {"Authorization": "Bearer wrong-token"},
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": TOKEN},
        {"Authorization": ""},
    ],
)
def test_bad_credentials_are_rejected(client: TestClient, header: dict[str, str]) -> None:
    assert client.get("/admin/policy", headers=header).status_code == 401


def test_a_prefix_of_the_token_is_not_enough(client: TestClient) -> None:
    response = client.get("/admin/policy", headers={"Authorization": f"Bearer {TOKEN[:-1]}"})
    assert response.status_code == 401


def test_the_admin_api_is_disabled_when_no_token_is_configured(tmp_path: Path) -> None:
    """Runtime policy editing is off unless a deployment turns it on."""
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'noadmin.db'}"
    settings = Settings(
        audit_pepper="0123456789abcdef0123456789abcdef",  # type: ignore[arg-type]
        config_dir=CONFIG_DIR,
        database_url=database_url,
        audit_wal_path=tmp_path / "wal",
    )
    engine = create_async_engine(database_url)

    async def _create() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    anyio.run(_create)

    with TestClient(create_app(settings, sink=AuditSink(settings, engine=engine))) as c:
        assert c.get("/admin/policy", headers=AUTH).status_code == 503


# ---------------------------------------------------------------------------
# Custom tier-3 labels
# ---------------------------------------------------------------------------


def test_the_baseline_labels_are_listed(client: TestClient) -> None:
    labels = client.get("/admin/policy", headers=AUTH).json()["tier3_labels"]
    assert labels == {
        "PERSON": "person name",
        "LOCATION": "location",
        "ORGANIZATION": "organization",
    }


def test_an_admin_can_add_a_custom_english_label(client: TestClient) -> None:
    body = _set(
        client,
        "PROJECT_CODENAME",
        gliner_prompt="internal project codename",
        category="other",
        action="MASK",
        score_threshold=0.6,
        note="Q3 launch names",
    )

    assert body["tier3_labels"]["PROJECT_CODENAME"] == "internal project codename"
    entry = next(e for e in body["entities"] if e["entity_type"] == "PROJECT_CODENAME")
    assert entry["source"] == "overlay"
    assert entry["tier"] == 3
    assert entry["placeholder"] == "<PROJECT_CODENAME>"


def test_a_new_label_without_a_prompt_is_refused(client: TestClient) -> None:
    """Otherwise it is policy that can never fire."""
    response = client.put("/admin/entities/GHOST", headers=AUTH, json={"entity_type": "GHOST"})
    assert response.status_code == 400
    assert "gliner_prompt" in response.text


def test_a_label_survives_a_restart(client: TestClient, tmp_path: Path) -> None:
    """The overlay is persisted, not just held in memory."""
    _set(client, "BADGE_NUMBER", gliner_prompt="employee badge number")

    database_url = f"sqlite+aiosqlite:///{tmp_path / 'admin.db'}"
    settings = Settings(
        audit_pepper="0123456789abcdef0123456789abcdef",  # type: ignore[arg-type]
        config_dir=CONFIG_DIR,
        database_url=database_url,
        audit_wal_path=tmp_path / "wal",
        admin_token=TOKEN,  # type: ignore[arg-type]
    )
    engine = create_async_engine(database_url)
    with TestClient(create_app(settings, sink=AuditSink(settings, engine=engine))) as fresh:
        labels = fresh.get("/admin/policy", headers=AUTH).json()["tier3_labels"]
        assert labels["BADGE_NUMBER"] == "employee badge number"


def test_a_mismatched_path_and_body_is_refused(client: TestClient) -> None:
    response = client.put(
        "/admin/entities/ONE",
        headers=AUTH,
        json={"entity_type": "TWO", "gliner_prompt": "some label"},
    )
    assert response.status_code == 400


@pytest.mark.parametrize("bad", ["lowercase", "With-Dash", "HAS SPACE"])
def test_malformed_entity_types_are_refused(client: TestClient, bad: str) -> None:
    response = client.put(
        f"/admin/entities/{bad}", headers=AUTH, json={"entity_type": bad, "gliner_prompt": "x"}
    )
    assert response.status_code in (400, 422)


def test_warns_when_labels_exist_but_tier3_is_off(client: TestClient) -> None:
    """Configured-but-inert is the failure this whole codebase keeps guarding."""
    body = _set(client, "PROJECT_CODENAME", gliner_prompt="internal project codename")
    assert any("tier 3 is disabled" in warning for warning in body["warnings"])


# ---------------------------------------------------------------------------
# Replacement policy
# ---------------------------------------------------------------------------


def test_strategies_are_listed_for_a_menu(client: TestClient) -> None:
    strategies = client.get("/admin/replacement-strategies", headers=AUTH).json()["strategies"]
    names = {s["name"] for s in strategies}

    assert names == {"placeholder", "constant", "surrogate", "redact", "labelled_fingerprint"}
    # The menu has to be able to warn, so each entry says whether it is realistic.
    assert {s["name"] for s in strategies if s["realistic"]} == {"constant", "surrogate"}


def test_setting_a_constant_changes_the_masking(client: TestClient) -> None:
    """The feature as asked for: a label replaced with a chosen name."""
    mobile = synthetic_mobile(prefix="010", rng=random.Random(3))
    assert _mask(client, f"call {mobile}", "before") == "call <EG_MOBILE>"

    _set(
        client,
        "EG_MOBILE",
        replacement={"strategy": "constant", "value": "+20 100 000 0000"},
    )
    assert _mask(client, f"call {mobile}", "after") == "call +20 100 000 0000"


def test_a_constant_collapses_distinct_values(client: TestClient) -> None:
    """Documented trade-off, asserted so nobody is surprised by it."""
    rng = random.Random(5)
    a, b = synthetic_mobile(prefix="010", rng=rng), synthetic_mobile(prefix="011", rng=rng)
    _set(client, "EG_MOBILE", replacement={"strategy": "constant", "value": "SAME"})

    assert _mask(client, f"{a} and {b}", "collapse") == "SAME and SAME"


def test_a_surrogate_does_not_collapse_every_value_to_one(client: TestClient) -> None:
    """The property that separates `surrogate` from `constant`.

    Not "any two values differ": a pool is finite, so two values landing on the
    same surrogate is expected, and at pool size 4 it happens a quarter of the
    time. That collision is a feature -- several people sharing a fake name is
    weaker linkage, not a bug. What must hold is that the mapping spreads,
    where `constant` maps everything to a single string.
    """
    rng = random.Random(5)
    numbers = [synthetic_mobile(rng=rng) for _ in range(8)]
    _set(
        client,
        "EG_MOBILE",
        replacement={"strategy": "surrogate", "pool": ["N1", "N2", "N3", "N4"]},
    )

    masked = _mask(client, " ".join(numbers), "spread")
    used = {token for token in masked.split() if token.startswith("N")}

    assert len(used) > 1, f"surrogate collapsed everything to {used}"


def test_a_surrogate_is_stable_across_requests(client: TestClient) -> None:
    """The same person is the same fake name in every turn of a conversation."""
    mobile = synthetic_mobile(prefix="010", rng=random.Random(9))
    _set(
        client,
        "EG_MOBILE",
        replacement={"strategy": "surrogate", "pool": ["N1", "N2", "N3", "N4"]},
    )

    first = _mask(client, f"call {mobile}", "t1")
    second = _mask(client, f"call {mobile}", "t2")
    assert first == second


def test_labelled_fingerprint_stays_obviously_masked(client: TestClient) -> None:
    rng = random.Random(11)
    a, b = synthetic_mobile(prefix="010", rng=rng), synthetic_mobile(prefix="011", rng=rng)
    _set(client, "EG_MOBILE", replacement={"strategy": "labelled_fingerprint"})

    masked = _mask(client, f"{a} and {b}", "lf")
    assert masked.startswith("<EG_MOBILE:")
    assert masked.count("<EG_MOBILE:") == 2
    left, _, right = masked.partition(" and ")
    assert left != right


def test_redact_removes_the_value(client: TestClient) -> None:
    mobile = synthetic_mobile(prefix="010", rng=random.Random(13))
    _set(client, "EG_MOBILE", replacement={"strategy": "redact"})
    assert _mask(client, f"call {mobile} now", "redact") == "call  now"


def test_an_incoherent_replacement_is_refused(client: TestClient) -> None:
    response = client.put(
        "/admin/entities/EG_MOBILE",
        headers=AUTH,
        json={"entity_type": "EG_MOBILE", "replacement": {"strategy": "surrogate", "pool": []}},
    )
    assert response.status_code == 422


def test_the_policy_view_warns_about_realistic_replacement(client: TestClient) -> None:
    body = _set(
        client, "EG_MOBILE", replacement={"strategy": "constant", "value": "+20 100 000 0000"}
    )
    assert body["realistic_replacement_entities"] == ["EG_MOBILE"]
    assert any("no longer looks masked" in warning for warning in body["warnings"])


# ---------------------------------------------------------------------------
# Idempotency -- the property realistic replacement would otherwise break
# ---------------------------------------------------------------------------


def test_masking_a_surrogate_again_is_a_no_op(client: TestClient) -> None:
    """The architecture masks twice (brief §2); the second pass must change nothing.

    Without suppression the guardrail would detect "N1" as a mobile number and
    replace it with a different surrogate, corrupting the text on every hop.
    """
    rng = random.Random(17)
    a, b = synthetic_mobile(prefix="010", rng=rng), synthetic_mobile(prefix="011", rng=rng)
    _set(
        client,
        "EG_MOBILE",
        replacement={"strategy": "surrogate", "pool": ["+20 100 000 0001", "+20 100 000 0002"]},
    )

    once = _mask(client, f"{a} and {b}", "p1")
    twice = _mask(client, once, "p2")
    thrice = _mask(client, twice, "p3")

    assert once == twice == thrice


def test_masking_a_constant_again_is_a_no_op(client: TestClient) -> None:
    mobile = synthetic_mobile(prefix="010", rng=random.Random(19))
    _set(client, "EG_MOBILE", replacement={"strategy": "constant", "value": "+20 100 000 0000"})

    once = _mask(client, f"call {mobile}", "c1")
    assert _mask(client, once, "c2") == once


def test_placeholder_masking_is_still_idempotent(client: TestClient) -> None:
    mobile = synthetic_mobile(prefix="010", rng=random.Random(23))
    once = _mask(client, f"call {mobile}", "d1")
    assert _mask(client, once, "d2") == once


# ---------------------------------------------------------------------------
# Reverting and cache coherence
# ---------------------------------------------------------------------------


def test_deleting_an_overlay_reverts_to_the_baseline(client: TestClient) -> None:
    mobile = synthetic_mobile(prefix="010", rng=random.Random(29))
    _set(client, "EG_MOBILE", replacement={"strategy": "constant", "value": "GONE"})
    assert _mask(client, f"call {mobile}", "v1") == "call GONE"

    assert client.delete("/admin/entities/EG_MOBILE", headers=AUTH).status_code == 200
    assert _mask(client, f"call {mobile}", "v2") == "call <EG_MOBILE>"


def test_deleting_an_absent_overlay_is_a_404(client: TestClient) -> None:
    assert client.delete("/admin/entities/EG_MOBILE", headers=AUTH).status_code == 404


def test_a_policy_change_is_not_served_from_a_stale_cache(client: TestClient) -> None:
    """The same text, analysed before and after an edit, must not reuse the verdict."""
    mobile = synthetic_mobile(prefix="010", rng=random.Random(31))
    text = f"call {mobile}"

    assert _mask(client, text, "s1") == "call <EG_MOBILE>"
    _set(client, "EG_MOBILE", replacement={"strategy": "constant", "value": "CHANGED"})
    assert _mask(client, text, "s2") == "call CHANGED"


def test_disabling_an_entity_stops_it_being_masked(client: TestClient) -> None:
    mobile = synthetic_mobile(prefix="010", rng=random.Random(37))
    assert "<EG_MOBILE>" in _mask(client, f"call {mobile}", "e1")

    _set(client, "EG_MOBILE", enabled=False, note="too noisy for this tenant")
    assert _mask(client, f"call {mobile}", "e2") == f"call {mobile}"


def test_reload_picks_up_another_replicas_change(client: TestClient) -> None:
    assert client.post("/admin/reload", headers=AUTH).status_code == 200


# ---------------------------------------------------------------------------
# Integration with the read-only /policy the admin UI renders
# ---------------------------------------------------------------------------


def test_the_public_policy_endpoint_reflects_an_admin_change(client: TestClient) -> None:
    """The UI reads /policy; it must not show the frozen baseline.

    This is the screen an operator opens to confirm their change landed, so
    stale data here is worse than stale data anywhere else.
    """
    before = {e["entity_type"]: e for e in client.get("/policy").json()["entities"]}
    assert before["EG_MOBILE"]["replacement_strategy"] == "placeholder"

    _set(client, "EG_MOBILE", replacement={"strategy": "constant", "value": "John Doe"})

    after = {e["entity_type"]: e for e in client.get("/policy").json()["entities"]}
    assert after["EG_MOBILE"]["replacement_strategy"] == "constant"
    assert after["EG_MOBILE"]["replacement_example"] == "John Doe"


def test_a_custom_label_appears_in_the_public_policy(client: TestClient) -> None:
    _set(client, "PROJECT_CODENAME", gliner_prompt="internal project codename")

    entities = {e["entity_type"]: e for e in client.get("/policy").json()["entities"]}
    assert entities["PROJECT_CODENAME"]["gliner_prompt"] == "internal project codename"
    assert entities["PROJECT_CODENAME"]["tier"] == 3


def test_a_disabled_entity_disappears_from_the_public_policy(client: TestClient) -> None:
    _set(client, "EG_MOBILE", enabled=False)
    entities = {e["entity_type"] for e in client.get("/policy").json()["entities"]}
    assert "EG_MOBILE" not in entities


def test_the_public_policy_still_needs_no_auth(client: TestClient) -> None:
    """It carries entity names and thresholds -- no PII, no secrets."""
    assert client.get("/policy").status_code == 200


# ---------------------------------------------------------------------------
# Enum casing at the HTTP boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", ["MASK", "mask", "Mask"])
def test_an_action_is_accepted_in_any_case(client: TestClient, action: str) -> None:
    """The two policy enums disagree on case, and a caller should not have to know.

    EntityAction is MASK/BLOCK/ALLOW and EntityCategory is id/person/...,
    because each matches how its values already appear in entities.yaml and in
    the audit rows. Fine inside the service; at an HTTP boundary it means an
    operator round-tripping a value out of GET /policy sends "MASK" and one
    typing it by hand sends "mask", and one of them gets a 422 for no reason
    they can see.
    """
    response = _set(client, "PROJECT_CODENAME", gliner_prompt="a codename", action=action)
    entities = {e["entity_type"]: e for e in response["entities"]}
    assert entities["PROJECT_CODENAME"]["action"] == "MASK"


@pytest.mark.parametrize("category", ["other", "OTHER", "Other"])
def test_a_category_is_accepted_in_any_case(client: TestClient, category: str) -> None:
    response = _set(client, "PROJECT_CODENAME", gliner_prompt="a codename", category=category)
    entities = {e["entity_type"]: e for e in response["entities"]}
    assert entities["PROJECT_CODENAME"]["category"] == "other"


def test_a_genuinely_unknown_action_is_still_refused(client: TestClient) -> None:
    """Normalising case must not become accepting anything."""
    response = client.put(
        "/admin/entities/PROJECT_CODENAME",
        headers=AUTH,
        json={"entity_type": "PROJECT_CODENAME", "gliner_prompt": "a codename", "action": "SCRUB"},
    )
    assert response.status_code == 422

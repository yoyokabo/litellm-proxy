"""The entity-policy routes behind the admin menu's Entities screen.

What these cover is this backend's own contribution, which is narrow on
purpose: pii-service owns the policy rules, so there is nothing here that
re-checks them. What is here is the part no other service can do --

* **authorisation**: an anonymous caller and a caller still on the bootstrap
  password both get nothing;
* **attribution**: the operator's email reaches pii-service, so a policy change
  names a person and not a shared token;
* **failure translation**: a refusal upstream reaches the operator with its
  reason, and a deployment that set no token gets told that rather than a 401;
* **the category cache**, which would otherwise render a brand-new label in the
  "other" colour until the next restart.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, FakeLiteLlm, FakePiiService

CODENAME = {
    "entity_type": "PROJECT_CODENAME",
    "gliner_prompt": "internal project codename",
    "category": "other",
    "action": "MASK",
    "replacement": {"strategy": "placeholder"},
}


# ---------------------------------------------------------------------------
# Authorisation
# ---------------------------------------------------------------------------


def test_listing_entities_needs_a_session(
    app_client: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    client, _, _ = app_client
    assert client.get("/api/admin/entities").status_code == 401


def test_changing_an_entity_needs_a_session(
    app_client: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    client, pii, _ = app_client
    response = client.put("/api/admin/entities/PROJECT_CODENAME", json=CODENAME)
    assert response.status_code == 401
    # And nothing reached pii-service. An unauthenticated request that still
    # changed the policy would be the whole failure.
    assert pii.admin_calls == []


def test_the_bootstrap_password_gate_covers_entity_administration(
    app_client: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    """Brief §10: nothing admin works until the bootstrap password is rotated.

    Changing what the gateway masks is the most consequential thing in this
    app, so it is the last place that gate should have a hole.
    """
    client, pii, _ = app_client
    client.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})

    listed = client.get("/api/admin/entities")
    changed = client.put("/api/admin/entities/PROJECT_CODENAME", json=CODENAME)

    assert listed.status_code == 403
    assert listed.json()["detail"]["code"] == "password_change_required"
    assert changed.status_code == 403
    assert pii.admin_calls == []


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_an_operator_can_read_the_effective_policy(
    logged_in: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    client, _, _ = logged_in
    body = client.get("/api/admin/entities").json()
    assert "entities" in body
    assert body["tier3_labels"] == {"internal project codename": "PROJECT_CODENAME"}


def test_the_strategy_list_is_served_for_the_dropdown(
    logged_in: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    client, _, _ = logged_in
    body = client.get("/api/admin/replacement-strategies").json()
    assert [s["name"] for s in body["strategies"]] == ["placeholder"]


def test_adding_a_custom_label_reaches_pii_service_unchanged(
    logged_in: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    """The body is forwarded as received.

    Deliberate: pii-service validates it. If this backend started reshaping
    payloads, the two services' ideas of a valid overlay would drift, and the
    error messages an operator reads would stop matching what they sent.
    """
    client, pii, _ = logged_in
    response = client.put("/api/admin/entities/PROJECT_CODENAME", json=CODENAME)

    assert response.status_code == 200
    assert pii.admin_calls[-1]["payload"] == CODENAME
    assert pii.admin_calls[-1]["entity_type"] == "PROJECT_CODENAME"


def test_a_policy_change_is_attributed_to_the_person_who_made_it(
    logged_in: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    """The reason this backend proxies at all.

    pii-service's admin API takes one shared bearer token and cannot tell who
    is behind it. This service can, and forwards it, so custom_entities
    .updated_by names a person.
    """
    client, pii, _ = logged_in
    client.put("/api/admin/entities/PROJECT_CODENAME", json=CODENAME)
    client.delete("/api/admin/entities/PROJECT_CODENAME")

    assert [call["by"] for call in pii.admin_calls] == [ADMIN_EMAIL, ADMIN_EMAIL]


def test_deleting_an_entity_reverts_it(
    logged_in: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    client, pii, _ = logged_in
    client.put("/api/admin/entities/PROJECT_CODENAME", json=CODENAME)
    body = client.delete("/api/admin/entities/PROJECT_CODENAME").json()

    assert pii.admin_calls[-1]["method"] == "DELETE"
    assert body["entities"] == []


# ---------------------------------------------------------------------------
# Failures reach the operator intact
# ---------------------------------------------------------------------------


def test_an_upstream_refusal_keeps_its_status_and_its_reason(
    logged_in: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    """A 400 upstream must not become a 502 here.

    "NEW_LABEL is not in the baseline policy, so it needs a gliner_prompt" is
    the single most useful sentence this screen can show, and it is written in
    pii-service. Flattening it would leave the operator with "request failed".
    """
    client, pii, _ = logged_in
    pii.admin_error = (400, "NEW_LABEL is not in the baseline policy, so it needs a gliner_prompt")

    response = client.put("/api/admin/entities/NEW_LABEL", json={"entity_type": "NEW_LABEL"})

    assert response.status_code == 400
    assert "gliner_prompt" in response.json()["detail"]


def test_a_validation_error_upstream_arrives_as_one(
    logged_in: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    client, pii, _ = logged_in
    pii.admin_error = (422, "pool must not be empty for the surrogate strategy")
    response = client.put("/api/admin/entities/PERSON", json={"entity_type": "PERSON"})
    assert response.status_code == 422


def test_a_deployment_without_the_token_is_told_so(
    logged_in: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    """503 with an explanation, not 401.

    A 401 would send an operator hunting for a login problem when what is
    actually wrong is an unset environment variable in their compose file.
    """
    client, pii, _ = logged_in
    pii.admin_token = ""

    response = client.get("/api/admin/entities")

    assert response.status_code == 503
    assert "PII_API_PII_ADMIN_TOKEN" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Path safety and cache invalidation
# ---------------------------------------------------------------------------


def test_an_entity_type_cannot_escape_its_path_segment(
    logged_in: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    """The entity type is interpolated into an upstream URL.

    Without the shape check, "../reload" would name a different admin route.
    Rejected before any upstream call, so a refusal is not merely an upstream
    404 that happens to be safe today.
    """
    client, pii, _ = logged_in

    for bad in ("..", "lowercase", "WITH SPACE", "WITH-DASH", "9LEADING_DIGIT"):
        response = client.delete(f"/api/admin/entities/{bad}")
        assert response.status_code in (400, 404, 405), bad

    assert pii.admin_calls == []


def test_a_new_label_does_not_render_in_the_wrong_colour(
    logged_in: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    """Adding an entity drops the cached entity_type -> category map.

    That map is cached for the process lifetime because it normally changes
    only when someone edits entities.yaml and restarts. This screen is the
    case that assumption misses.
    """
    client, _, _ = logged_in
    app: Any = client.app
    app.state.category_map = {"EG_NATIONAL_ID": "id"}

    client.put("/api/admin/entities/PROJECT_CODENAME", json=CODENAME)

    assert app.state.category_map is None

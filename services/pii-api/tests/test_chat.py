"""The chat path: analyse first, forward only what survived.

The load-bearing test in this file is
``test_the_original_text_never_reaches_the_model``. Everything else about the
chat could work and that one property could still be broken, and if it were,
the product would be a PII leak with a reassuring chip above it.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from tests.conftest import FakeLiteLlm, FakePiiService

NID = "28503122196078"


def _events(raw: str) -> list[tuple[str, dict]]:
    """Parse the SSE stream into (event, payload) pairs."""
    parsed: list[tuple[str, dict]] = []
    for block in raw.split("\n\n"):
        name = payload = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                payload = json.loads(line.removeprefix("data: "))
        if name is not None:
            parsed.append((name, payload or {}))
    return parsed


def _send(client: TestClient, content: str) -> list[tuple[str, dict]]:
    with client.stream(
        "POST", "/api/chat/stream", json={"messages": [{"role": "user", "content": content}]}
    ) as response:
        assert response.status_code == 200
        return _events("".join(response.iter_text()))


def test_the_original_text_never_reaches_the_model(
    logged_in: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    client, _, litellm = logged_in
    _send(client, f"my national id is {NID} please remember it")

    assert litellm.sent, "nothing was forwarded"
    forwarded = json.dumps(litellm.sent[-1])
    assert NID not in forwarded
    assert "<EG_NATIONAL_ID>" in forwarded


def test_the_analysis_arrives_before_any_token(
    logged_in: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    """The feedback chip must render before the reply starts arriving."""
    client, _, _ = logged_in
    events = _send(client, f"id {NID}")

    names = [name for name, _ in events]
    assert names[0] == "analysis"
    assert "delta" in names
    assert names[-1] == "done"


def test_the_analysis_describes_what_was_filtered(
    logged_in: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    client, _, _ = logged_in
    events = dict(_send(client, f"id {NID} ok"))
    analysis = events["analysis"]

    assert analysis["counts"] == {"EG_NATIONAL_ID": 1}
    span = analysis["spans"][0]
    assert span["entity_type"] == "EG_NATIONAL_ID"
    assert span["placeholder"] == "<EG_NATIONAL_ID>"
    # The user's own value, going back to the user's own browser (brief §3).
    assert span["text"] == NID
    # Offsets must select exactly that text in the original string.
    assert f"id {NID} ok"[span["start"] : span["end"]] == NID


def test_a_blocked_message_is_not_forwarded(
    logged_in: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    client, pii, litellm = logged_in
    pii.blocked = True

    events = dict(_send(client, f"id {NID}"))

    assert events["analysis"]["blocked"] is True
    assert events["done"] == {"blocked": True}
    assert not litellm.sent, "a blocked message reached the model"


def test_detection_failure_fails_closed(
    logged_in: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    """Opposite default to the proxy guardrail, on purpose.

    The guardrail fails open so a detection outage cannot take the gateway
    down for everyone. This endpoint serves one person who is being told
    their message was screened -- sending it unscreened while implying
    otherwise is worse than an error.
    """
    client, pii, litellm = logged_in
    pii.fail = True

    events = dict(_send(client, f"id {NID}"))

    assert "error" in events
    assert not litellm.sent, "the message was forwarded despite detection failing"


def test_each_user_gets_their_own_virtual_key(
    logged_in: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    """Identity on the audit row comes from this key and nowhere else."""
    client, _, litellm = logged_in
    _send(client, "hello")
    _send(client, "hello again")

    # Minted once, then reused.
    assert litellm.keys_generated == ["app-user-1"]


def test_analyze_only_endpoint_does_not_send(
    logged_in: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    """Pre-send detection is detection, not sending."""
    client, _, litellm = logged_in
    response = client.post(
        "/api/chat/analyze", json={"messages": [{"role": "user", "content": f"id {NID}"}]}
    )

    assert response.status_code == 200
    assert response.json()["analysis"]["counts"] == {"EG_NATIONAL_ID": 1}
    assert not litellm.sent


def test_chat_requires_authentication(
    app_client: tuple[TestClient, FakePiiService, FakeLiteLlm],
) -> None:
    client, _, _ = app_client
    assert (
        client.post("/api/chat/stream", json={"messages": [{"role": "user", "content": "x"}]})
    ).status_code == 401

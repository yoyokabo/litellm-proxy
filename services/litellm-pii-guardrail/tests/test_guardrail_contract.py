"""Contract tests for the LiteLLM guardrail adapter.

Two kinds of test here, and the second kind is the point.

The unit tests check what the guardrail does to ``inputs``. The integration
tests at the bottom drive litellm's *real* ``OpenAIChatCompletionsHandler``
over a real request dict and assert on the messages that come out the other
side. Those are what actually prove the masking reaches the model, because the
handler's mapping rule is subtle enough that a guardrail can look completely
correct in isolation and still be a no-op in production.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from litellm.llms.openai.chat.guardrail_translation.handler import (
    OpenAIChatCompletionsHandler,
)
from pii_guardrail import ArabicPIIGuardrail, PiiBlockedError

SERVICE = "http://pii-service:8090"
NID = "28503122148219"
MASKED = "<EG_NATIONAL_ID>"


def _guardrail(**kwargs: Any) -> ArabicPIIGuardrail:
    return ArabicPIIGuardrail(
        service_url=SERVICE, guardrail_name="pii-ar", event_hook="pre_call", **kwargs
    )


def _service_reply(texts: list[str], **extra: Any) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "request_id": "r",
            "texts": texts,
            "findings": [],
            "entity_counts": extra.pop("entity_counts", {"EG_NATIONAL_ID": 1}),
            "blocked": False,
            "block_reason": None,
            "lang": "en",
            "latency_ms": 3,
            "cached": False,
            **extra,
        },
    )


def _request_data(**extra: Any) -> dict[str, Any]:
    return {
        "litellm_call_id": "call-123",
        "model": "qwen3-30b",
        "user": "end-user-7",
        "metadata": {
            "user_api_key_user_id": "eng-42",
            "user_api_key_team_id": "platform",
            "user_api_key_alias": "laptop",
            "user_api_key_hash": "sk-hash",
        },
        **extra,
    }


# ---------------------------------------------------------------------------
# Masking through `texts`
# ---------------------------------------------------------------------------


@respx.mock
async def test_texts_are_masked_in_place() -> None:
    respx.post(f"{SERVICE}/analyze").mock(return_value=_service_reply([f"id {MASKED}"]))

    texts = [f"id {NID}"]
    inputs: dict[str, Any] = {"texts": texts}
    result = await _guardrail().apply_guardrail(inputs, _request_data(), "request")

    assert texts[0] == f"id {MASKED}"
    assert result is inputs
    assert result["texts"] is texts  # same list object, edited in place


@respx.mock
async def test_structured_messages_is_never_rebound() -> None:
    """The load-bearing assertion of this whole file.

    litellm's handler takes the structured-messages branch when the returned
    ``structured_messages`` is a *different object* from the one it passed in,
    and in that branch it ignores ``texts`` entirely. Rebinding it here would
    silently discard every text edit above.
    """
    respx.post(f"{SERVICE}/analyze").mock(return_value=_service_reply([f"id {MASKED}"]))

    structured = [{"role": "user", "content": f"id {NID}"}]
    inputs: dict[str, Any] = {"texts": [f"id {NID}"], "structured_messages": structured}

    result = await _guardrail().apply_guardrail(inputs, _request_data(), "request")

    assert result["structured_messages"] is structured


@respx.mock
async def test_no_texts_and_no_tool_calls_is_a_noop() -> None:
    route = respx.post(f"{SERVICE}/analyze").mock(return_value=_service_reply([]))
    inputs: dict[str, Any] = {"texts": []}

    await _guardrail().apply_guardrail(inputs, _request_data(), "request")

    assert not route.called  # no pointless round trip


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------


@respx.mock
async def test_tool_call_arguments_are_scanned_and_masked_dict_format() -> None:
    masked_args = json.dumps({"q": f"lookup {MASKED}"})
    respx.post(f"{SERVICE}/analyze").mock(return_value=_service_reply([masked_args]))

    tool_calls = [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "search", "arguments": json.dumps({"q": f"lookup {NID}"})},
        }
    ]
    inputs: dict[str, Any] = {"texts": [], "tool_calls": tool_calls}

    await _guardrail().apply_guardrail(inputs, _request_data(), "request")

    assert tool_calls[0]["function"]["arguments"] == masked_args
    assert NID not in json.dumps(tool_calls)


@respx.mock
async def test_tool_call_arguments_are_masked_object_format() -> None:
    """The TypedDict allows pydantic tool-call objects, not only dicts."""
    from litellm.types.utils import ChatCompletionMessageToolCall, Function

    masked_args = json.dumps({"q": MASKED})
    respx.post(f"{SERVICE}/analyze").mock(return_value=_service_reply([masked_args]))

    tool_call = ChatCompletionMessageToolCall(
        function=Function(name="search", arguments=json.dumps({"q": NID})), id="c1"
    )
    inputs: dict[str, Any] = {"texts": [], "tool_calls": [tool_call]}

    await _guardrail().apply_guardrail(inputs, _request_data(), "request")

    assert tool_call.function.arguments == masked_args


@respx.mock
async def test_texts_and_tool_calls_are_sent_in_one_call_and_mapped_back_by_position() -> None:
    respx.post(f"{SERVICE}/analyze").mock(
        return_value=_service_reply(["masked-text", "masked-args"])
    )

    texts = ["text-one"]
    tool_calls = [{"function": {"name": "f", "arguments": "args-one"}}]
    inputs: dict[str, Any] = {"texts": texts, "tool_calls": tool_calls}

    await _guardrail().apply_guardrail(inputs, _request_data(), "request")

    assert texts[0] == "masked-text"
    assert tool_calls[0]["function"]["arguments"] == "masked-args"


@respx.mock
async def test_fields_label_each_slot_as_content_or_tool_call_args() -> None:
    """The audit row's `field` column comes from here and nowhere else.

    pii-service reads `texts` and `fields` positionally, so this is the one
    thing that can make a drill-down say "tool call arguments" rather than
    leaving the column null.
    """
    route = respx.post(f"{SERVICE}/analyze").mock(return_value=_service_reply(["a", "b", "c"]))

    inputs: dict[str, Any] = {
        "texts": ["one", "two"],
        "tool_calls": [{"function": {"name": "f", "arguments": "three"}}],
    }
    await _guardrail().apply_guardrail(inputs, _request_data(), "request")

    sent = json.loads(route.calls[0].request.content)
    assert sent["texts"] == ["one", "two", "three"]
    assert sent["fields"] == ["content", "content", "tool_call.args"]


@respx.mock
async def test_fields_stay_parallel_to_the_texts_actually_sent() -> None:
    """Cached texts are dropped from the request; `fields` must drop with them.

    If the two lists ever drift, every audit row after the first cache hit is
    labelled with another slot's field -- silently, and in the audit trail.
    """
    guardrail = _guardrail()

    first = respx.post(f"{SERVICE}/analyze").mock(return_value=_service_reply(["cached-masked"]))
    await guardrail.apply_guardrail({"texts": ["cached"]}, _request_data(), "request")
    assert first.called

    route = respx.post(f"{SERVICE}/analyze").mock(return_value=_service_reply(["args-masked"]))
    inputs: dict[str, Any] = {
        "texts": ["cached"],
        "tool_calls": [{"function": {"name": "f", "arguments": "fresh-args"}}],
    }
    await guardrail.apply_guardrail(inputs, _request_data(), "request")

    sent = json.loads(route.calls[-1].request.content)
    assert sent["texts"] == ["fresh-args"]
    assert sent["fields"] == ["tool_call.args"]


# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------


@respx.mock
async def test_block_raises_with_counts_and_never_the_value() -> None:
    respx.post(f"{SERVICE}/analyze").mock(
        return_value=_service_reply([f"id {NID}"], blocked=True, block_reason={"EG_NATIONAL_ID": 2})
    )

    with pytest.raises(PiiBlockedError) as caught:
        await _guardrail().apply_guardrail({"texts": [f"id {NID}"]}, _request_data(), "request")

    message = str(caught.value)
    assert "EG_NATIONAL_ID" in message
    assert "2" in message
    # This message reaches the client and the logs.
    assert NID not in message
    assert caught.value.counts == {"EG_NATIONAL_ID": 2}


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


@respx.mock
async def test_service_down_fails_open_by_default() -> None:
    respx.post(f"{SERVICE}/analyze").mock(side_effect=httpx.ConnectError("refused"))

    texts = [f"id {NID}"]
    await _guardrail().apply_guardrail({"texts": texts}, _request_data(), "request")

    # Unmasked, but the gateway is still serving 100+ engineers.
    assert texts[0] == f"id {NID}"


@respx.mock
async def test_service_down_fails_closed_when_configured() -> None:
    respx.post(f"{SERVICE}/analyze").mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(httpx.ConnectError):
        await _guardrail(fail_closed=True).apply_guardrail(
            {"texts": [f"id {NID}"]}, _request_data(), "request"
        )


@respx.mock
async def test_service_error_status_fails_open() -> None:
    respx.post(f"{SERVICE}/analyze").mock(return_value=httpx.Response(500, text="boom"))

    texts = [f"id {NID}"]
    await _guardrail().apply_guardrail({"texts": texts}, _request_data(), "request")
    assert texts[0] == f"id {NID}"


# ---------------------------------------------------------------------------
# Identity and payload hygiene
# ---------------------------------------------------------------------------


@respx.mock
async def test_identity_is_forwarded_from_request_metadata() -> None:
    route = respx.post(f"{SERVICE}/analyze").mock(return_value=_service_reply(["x"]))

    await _guardrail().apply_guardrail({"texts": ["y"]}, _request_data(), "request")

    sent = json.loads(route.calls[0].request.content)
    assert sent["request_id"] == "call-123"
    assert sent["identity"] == {
        "user_id": "eng-42",
        "team_id": "platform",
        "key_alias": "laptop",
        "key_hash": "sk-hash",
        "end_user_id": "end-user-7",
    }


@respx.mock
async def test_missing_identity_degrades_rather_than_failing() -> None:
    route = respx.post(f"{SERVICE}/analyze").mock(return_value=_service_reply(["x"]))

    await _guardrail().apply_guardrail({"texts": ["y"]}, {"litellm_call_id": "c"}, "request")

    sent = json.loads(route.calls[0].request.content)
    assert sent["identity"]["user_id"] is None


@respx.mock
async def test_request_data_is_never_sent_to_the_service() -> None:
    """The exact shape that caused the credential-leak incident."""
    route = respx.post(f"{SERVICE}/analyze").mock(return_value=_service_reply(["x"]))
    request_data = _request_data(api_key="sk-super-secret", messages=[{"content": "secret"}])

    await _guardrail().apply_guardrail({"texts": ["y"]}, request_data, "request")

    body = route.calls[0].request.content.decode()
    assert "sk-super-secret" not in body
    # An allow-list, not a deny-list: a new key has to be added here
    # deliberately, which is the review step this test exists to force.
    # `fields` is safe by construction -- its only values are the two
    # literals "content" and "tool_call.args".
    assert set(json.loads(body)) == {"texts", "fields", "request_id", "identity", "model"}


@respx.mock
async def test_returns_the_inputs_object_not_a_large_dict() -> None:
    respx.post(f"{SERVICE}/analyze").mock(return_value=_service_reply(["x"]))
    inputs: dict[str, Any] = {"texts": ["y"]}

    result = await _guardrail().apply_guardrail(inputs, _request_data(), "request")

    assert result is inputs
    assert "messages" not in result
    assert "api_key" not in result


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


@respx.mock
async def test_repeated_text_is_served_from_cache() -> None:
    route = respx.post(f"{SERVICE}/analyze").mock(return_value=_service_reply([MASKED]))
    guardrail = _guardrail()

    for _ in range(4):
        await guardrail.apply_guardrail({"texts": [NID]}, _request_data(), "request")

    assert route.call_count == 1


@respx.mock
async def test_only_uncached_texts_are_sent() -> None:
    guardrail = _guardrail()
    first = respx.post(f"{SERVICE}/analyze").mock(return_value=_service_reply(["m-a"]))
    await guardrail.apply_guardrail({"texts": ["a"]}, _request_data(), "request")

    respx.reset()
    second = respx.post(f"{SERVICE}/analyze").mock(return_value=_service_reply(["m-b"]))
    texts = ["a", "b"]
    await guardrail.apply_guardrail({"texts": texts}, _request_data(), "request")

    assert json.loads(second.calls[0].request.content)["texts"] == ["b"]
    assert texts == ["m-a", "m-b"]
    assert first.called


# ---------------------------------------------------------------------------
# Integration: litellm's real handler
# ---------------------------------------------------------------------------


@respx.mock
async def test_masking_reaches_request_messages_through_the_real_handler() -> None:
    """End to end through litellm's own translation handler.

    This is the test that would catch a regression to the structured-messages
    branch: it asserts on ``data["messages"]`` after the handler has run, which
    is what the model actually receives.
    """
    # The handler sends the system message too, so the reply must carry three
    # entries. The guardrail zips them strictly, which is what turned an
    # earlier version of this mock into a loud error instead of a silent
    # off-by-one that would have masked the wrong messages.
    respx.post(f"{SERVICE}/analyze").mock(
        return_value=_service_reply(["be helpful", f"my id is {MASKED}", f"and again {MASKED}"])
    )

    data: dict[str, Any] = {
        "model": "qwen3-30b",
        "litellm_call_id": "call-abc",
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": f"my id is {NID}"},
            {"role": "user", "content": f"and again {NID}"},
        ],
    }

    await OpenAIChatCompletionsHandler().process_input_messages(
        data=data, guardrail_to_apply=_guardrail()
    )

    rendered = json.dumps(data["messages"], ensure_ascii=False)
    assert NID not in rendered
    assert data["messages"][1]["content"] == f"my id is {MASKED}"
    assert data["messages"][2]["content"] == f"and again {MASKED}"


@respx.mock
async def test_tool_call_masking_reaches_request_messages_through_the_real_handler() -> None:
    masked_args = json.dumps({"q": MASKED})
    respx.post(f"{SERVICE}/analyze").mock(
        return_value=_service_reply(["look this up", masked_args])
    )

    data: dict[str, Any] = {
        "model": "qwen3-30b",
        "litellm_call_id": "call-def",
        "messages": [
            {"role": "user", "content": "look this up"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"q": NID}),
                        },
                    }
                ],
            },
        ],
    }

    await OpenAIChatCompletionsHandler().process_input_messages(
        data=data, guardrail_to_apply=_guardrail()
    )

    rendered = json.dumps(data["messages"], ensure_ascii=False)
    assert NID not in rendered
    assert data["messages"][1]["tool_calls"][0]["function"]["arguments"] == masked_args


@respx.mock
async def test_handler_leaves_messages_untouched_when_the_service_is_down() -> None:
    respx.post(f"{SERVICE}/analyze").mock(side_effect=httpx.ConnectError("refused"))

    data: dict[str, Any] = {
        "model": "qwen3-30b",
        "litellm_call_id": "call-ghi",
        "messages": [{"role": "user", "content": f"my id is {NID}"}],
    }

    await OpenAIChatCompletionsHandler().process_input_messages(
        data=data, guardrail_to_apply=_guardrail()
    )

    assert data["messages"][0]["content"] == f"my id is {NID}"

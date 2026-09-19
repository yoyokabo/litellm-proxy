"""Arabic PII guardrail for LiteLLM.

A thin adapter. It calls pii-service over HTTP and maps the result back onto the
request. No models, no policy, no database access -- all of that lives in
pii-service, which is what makes this drop-in.

-----------------------------------------------------------------------------
IMPORTANT -- how the result is mapped back, verified against litellm 1.101.0
-----------------------------------------------------------------------------
``litellm/llms/openai/chat/guardrail_translation/handler.py`` applies our return
value like this (paraphrased):

    original = inputs.get("structured_messages")
    out = await guardrail.apply_guardrail(inputs=inputs, ...)
    sm = out.get("structured_messages")
    if sm is not None and sm is not original:
        data["messages"] = merge(sm)          # <-- structured path
    else:
        apply(out["texts"]); apply(out["tool_calls"])   # <-- texts path

It is an if/**else**, keyed on the *identity* of ``structured_messages``.

So the two mapping strategies are mutually exclusive. If we replace
``structured_messages`` with a new list *and* edit ``texts`` in place, the
handler takes the structured branch and our text edits are silently discarded --
the request goes to the model unmasked, with no error anywhere.

We therefore use the texts path and never rebind ``structured_messages``.
Editing ``texts`` and ``tool_calls`` in place keeps the object identity of
``structured_messages`` unchanged, which is exactly what keeps the else branch
live. Do not "fix" this by also returning structured_messages.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Literal

import httpx
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

DEFAULT_TIMEOUT = 4.0
CACHE_ENTRIES = 1024


class PiiBlockedError(Exception):
    """Raised to block a request. Carries entity types and counts, never values."""

    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts
        super().__init__(
            f"Request blocked by the pii-ar guardrail. Blocked entities: {sorted(counts.items())}."
        )


class ArabicPIIGuardrail(CustomGuardrail):
    def __init__(
        self,
        service_url: str = "http://pii-service:8090",
        timeout: float = DEFAULT_TIMEOUT,
        fail_closed: bool = False,
        **kwargs: Any,
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout
        # Default fail-open: pii-service being down must not take the gateway
        # down with it. Deployments that would rather refuse than pass PII
        # through unmasked set fail_closed: true in the guardrail config.
        self.fail_closed = fail_closed
        self._client = httpx.AsyncClient(timeout=timeout)
        self._cache: OrderedDict[str, tuple[str, dict[str, int]]] = OrderedDict()
        super().__init__(**kwargs)

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        texts = inputs.get("texts") or []
        tool_calls = inputs.get("tool_calls") or []

        # Tool-call arguments are scanned explicitly. Presidio's own hook reads
        # only content strings and {"text": ...} items, so arguments in
        # conversation history went unscanned on both wire formats; we do not
        # assume the installed version is patched.
        slots = [(texts, i, t) for i, t in enumerate(texts)]
        slots += [(tool_calls, i, a) for i, tc in enumerate(tool_calls) if (a := _args(tc))]
        if not slots:
            return inputs

        payload = [text for _, _, text in slots]
        masked, counts, blocked = await self._analyze(payload, request_data, inputs)

        if blocked:
            raise PiiBlockedError(blocked)

        for (container, index, _), new_text in zip(slots, masked, strict=True):
            if container is texts:
                texts[index] = new_text
            else:
                _set_args(container[index], new_text)

        # Counts and status only into LiteLLM's logging path. Rich detail goes
        # to our own sink; anything resembling the request payload here is the
        # shape that caused the credential-leak incident.
        if counts:
            self._log_counts(request_data, counts)
        return inputs

    async def _analyze(
        self, texts: list[str], request_data: dict, inputs: GenericGuardrailAPIInputs
    ) -> tuple[list[str], dict[str, int], dict[str, int]]:
        pending = [t for t in texts if self._key(t) not in self._cache]
        if pending:
            try:
                response = await self._client.post(
                    f"{self.service_url}/analyze",
                    json={
                        "texts": pending,
                        "request_id": str(request_data.get("litellm_call_id") or "unknown"),
                        "identity": _identity(request_data),
                        "model": inputs.get("model") or request_data.get("model"),
                    },
                )
                response.raise_for_status()
                body = response.json()
            except Exception:
                if self.fail_closed:
                    raise
                return texts, {}, {}
            for original, result in zip(pending, body["texts"], strict=True):
                self._remember(original, result, body.get("entity_counts") or {})
            if body.get("blocked"):
                return texts, {}, body.get("block_reason") or {}

        masked, counts = [], {}
        for text in texts:
            result, text_counts = self._cache[self._key(text)]
            self._cache.move_to_end(self._key(text))
            masked.append(result)
            for entity, count in text_counts.items():
                counts[entity] = counts.get(entity, 0) + count
        return masked, counts, {}

    def _log_counts(self, request_data: dict, counts: dict[str, int]) -> None:
        # Premium-gated on some builds and a silent no-op on a community
        # licence. Our audit trail does not depend on it, so a failure here is
        # never allowed to fail the request.
        with contextlib.suppress(Exception):
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response={"masked": counts},
                request_data=request_data,
                guardrail_status="success",
                masked_entity_count=counts,
                guardrail_provider="pii-ar",
            )

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _remember(self, original: str, masked: str, counts: dict[str, int]) -> None:
        self._cache[self._key(original)] = (masked, counts if original != masked else {})
        self._cache.move_to_end(self._key(original))
        while len(self._cache) > CACHE_ENTRIES:
            self._cache.popitem(last=False)


def _function(tool_call: object) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get("function")
    return getattr(tool_call, "function", None)


def _args(tool_call: object) -> str | None:
    """Read a tool call's arguments across both wire formats LiteLLM may pass."""
    function = _function(tool_call)
    if function is None:
        return None
    value = (
        function.get("arguments")
        if isinstance(function, dict)
        else getattr(function, "arguments", None)
    )
    return value if isinstance(value, str) and value else None


def _set_args(tool_call: object, value: str) -> None:
    function = _function(tool_call)
    if isinstance(function, dict):
        function["arguments"] = value
    elif function is not None:
        function.arguments = value


def _identity(request_data: dict) -> dict[str, str | None]:
    metadata = request_data.get("metadata") or request_data.get("litellm_metadata") or {}
    return {
        "user_id": metadata.get("user_api_key_user_id"),
        "team_id": metadata.get("user_api_key_team_id"),
        "key_alias": metadata.get("user_api_key_alias"),
        "key_hash": metadata.get("user_api_key_hash"),
        "end_user_id": request_data.get("user"),
    }

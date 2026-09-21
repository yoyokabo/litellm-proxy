"""Clients for the two services this backend talks to.

pii-service for detection, LiteLLM for completions. Both are held as
long-lived httpx clients on app state -- a new connection per chat message
would add a TCP and TLS handshake to a path that is already paying detection
latency.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Final

import httpx
import structlog

from pii_api.settings import Settings

__all__ = ["AdminDisabled", "LiteLlmClient", "PiiServiceClient", "UpstreamError"]

logger: Final = structlog.get_logger(__name__)


class PiiServiceClient:
    """Detection. The chat backend calls this directly (brief §2).

    Not through the proxy guardrail: the chat needs rich synchronous feedback
    about what was filtered, and extracting that from LiteLLM response headers
    is fragile and breaks under streaming. The guardrail still runs on the
    proxy for every other client, and detection is idempotent, so the masked
    text this produces costs one cheap second pass there and finds nothing.
    """

    def __init__(self, settings: Settings) -> None:
        self._base = settings.pii_service_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=30.0)
        self._admin_token = settings.pii_admin_token.get_secret_value().strip()

    @property
    def admin_enabled(self) -> bool:
        return bool(self._admin_token)

    async def analyze(
        self,
        texts: list[str],
        *,
        request_id: str,
        user_id: str | None,
        end_user_id: str | None,
        model: str | None,
        include_context: bool = True,
        message_indices: list[int] | None = None,
        message_roles: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "texts": texts,
            "request_id": request_id,
            "identity": {"user_id": user_id, "end_user_id": end_user_id},
            "model": model,
            # The chat is the one caller that turns this on: it renders the
            # user's own spans back to them, which brief §3 says is not a
            # disclosure. It is never persisted.
            "include_context": include_context,
            "fields": ["content"] * len(texts),
        }
        if message_indices is not None:
            payload["message_indices"] = message_indices
        if message_roles is not None:
            payload["message_roles"] = message_roles

        response = await self._client.post(f"{self._base}/analyze", json=payload)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    async def policy(self) -> dict[str, Any]:
        response = await self._client.get(f"{self._base}/policy")
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    # -- entity policy administration -------------------------------------
    #
    # The calls behind the admin menu's Entities screen. Each one is a thin
    # forward: pii-service owns the policy and every rule about what a valid
    # overlay is, so validating here would be a second copy of those rules
    # drifting from the first. What this layer adds is *who* -- the caller's
    # identity, which pii-service records in custom_entities.updated_by.

    def _admin_headers(self, acting_user: str | None) -> dict[str, str]:
        if not self._admin_token:
            raise AdminDisabled
        headers = {"Authorization": f"Bearer {self._admin_token}"}
        if acting_user:
            headers["X-Admin-User"] = acting_user
        return headers

    async def _admin_request(
        self,
        method: str,
        path: str,
        *,
        acting_user: str | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self._client.request(
            method,
            f"{self._base}{path}",
            headers=self._admin_headers(acting_user),
            json=json_body,
        )
        if response.status_code >= 400:
            # pii-service's refusals are the useful ones -- "needs a
            # gliner_prompt, otherwise nothing would ever detect it" is exactly
            # what the operator needs to read. Carry the status and the detail
            # through rather than flattening everything to a 502.
            detail: Any = response.text
            try:
                body = response.json()
                detail = body.get("detail", body) if isinstance(body, dict) else body
            except ValueError:
                pass
            raise UpstreamError(response.status_code, detail)
        return response.json()  # type: ignore[no-any-return]

    async def admin_policy(self) -> dict[str, Any]:
        return await self._admin_request("GET", "/admin/policy")

    async def replacement_strategies(self) -> dict[str, Any]:
        return await self._admin_request("GET", "/admin/replacement-strategies")

    async def upsert_entity(
        self, entity_type: str, payload: dict[str, Any], *, acting_user: str
    ) -> dict[str, Any]:
        return await self._admin_request(
            "PUT", f"/admin/entities/{entity_type}", acting_user=acting_user, json_body=payload
        )

    async def delete_entity(self, entity_type: str, *, acting_user: str) -> dict[str, Any]:
        return await self._admin_request(
            "DELETE", f"/admin/entities/{entity_type}", acting_user=acting_user
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class AdminDisabled(RuntimeError):
    """PII_API_PII_ADMIN_TOKEN is unset, so the entity-policy routes are off."""


class LiteLlmClient:
    """The proxy. Used for completions and for minting one virtual key per user."""

    def __init__(self, settings: Settings) -> None:
        self._base = settings.litellm_url.rstrip("/")
        self._master_key = settings.litellm_master_key.get_secret_value()
        self._client = httpx.AsyncClient(timeout=settings.upstream_timeout_seconds)

    def _admin_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._master_key}", "Content-Type": "application/json"}

    async def generate_key(self, llm_user_id: str, alias: str) -> str:
        """Mint a virtual key bound to a user id.

        This is what makes the audit trail answer "who". LiteLLM copies the
        key's user_id into request metadata, the guardrail forwards it, and it
        lands in pii_events.user_id. A shared key would audit every chat
        message to the same identity.
        """
        response = await self._client.post(
            f"{self._base}/key/generate",
            headers=self._admin_headers(),
            json={"user_id": llm_user_id, "key_alias": alias},
        )
        response.raise_for_status()
        return str(response.json()["key"])

    async def stream_chat(
        self, *, key: str, model: str, messages: list[dict[str, str]], end_user_id: str
    ) -> AsyncIterator[str]:
        """Stream a completion, yielding content deltas.

        Consumes LiteLLM's SSE and yields plain text. Translating here rather
        than proxying the raw stream keeps OpenAI's chunk format out of the
        browser, and lets the endpoint interleave its own events (the
        filtering feedback) into one stream the client reads linearly.
        """
        payload = {
            "model": model,
            "messages": messages,
            "user": end_user_id,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        async with self._client.stream(
            "POST", f"{self._base}/v1/chat/completions", headers=headers, json=payload
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")
                # Truncated and logged without the request: an upstream error
                # body can echo the prompt back, and this line reaches stdout.
                logger.warning("chat.upstream_error", status=response.status_code)
                raise UpstreamError(response.status_code, body[:500])

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    yield str(delta)

    async def aclose(self) -> None:
        await self._client.aclose()


class UpstreamError(RuntimeError):
    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        super().__init__(f"upstream returned {status}")

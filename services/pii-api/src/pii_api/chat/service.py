"""The chat path: analyse, then forward what survived.

The order is the whole design (brief §2). The backend analyses first so it can
tell the person what was filtered, then forwards *already-masked* text to
LiteLLM. The proxy guardrail still runs there as enforcement -- detection is
idempotent, so it re-scans masked text, finds nothing, and costs one cheap
pass. One audit writer either way.

What this module must never do is send the original text upstream after
deciding to mask it.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any, Final

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pii_api.api.schemas import ChatAnalysis, ChatSpan
from pii_api.db.models import AppUser, UserLlmKey
from pii_api.upstream import LiteLlmClient, PiiServiceClient

__all__ = ["analyze_message", "ensure_virtual_key", "stream_reply"]

logger: Final = structlog.get_logger(__name__)


async def ensure_virtual_key(
    session: AsyncSession, user: AppUser, litellm: LiteLlmClient
) -> UserLlmKey:
    """Get or mint this user's LiteLLM key.

    Created lazily on first message rather than at signup, so a proxy that is
    down cannot block account creation -- and so an account that never chats
    never consumes a key.
    """
    existing = await session.scalar(select(UserLlmKey).where(UserLlmKey.user_id == user.id))
    if existing is not None:
        return existing

    llm_user_id = f"app-user-{user.id}"
    key = await litellm.generate_key(llm_user_id, alias=f"pii-web-{user.id}")
    record = UserLlmKey(user_id=user.id, llm_user_id=llm_user_id, virtual_key=key)
    session.add(record)
    await session.commit()
    logger.info("chat.virtual_key_minted", user_id=user.id, llm_user_id=llm_user_id)
    return record


def _to_analysis(
    result: dict[str, Any], original: str, placeholders: dict[str, str]
) -> ChatAnalysis:
    spans: list[ChatSpan] = []
    for finding in result.get("findings") or []:
        # Only spans in the message being sent (text_index 0).
        if finding.get("text_index", 0) != 0:
            continue
        start, end = finding["start"], finding["end"]
        spans.append(
            ChatSpan(
                entity_type=finding["entity_type"],
                category=finding.get("category") or "other",
                action=finding.get("action") or "MASK",
                start=start,
                end=end,
                # Sliced from the caller's own string rather than taken from
                # the response, so the offsets and the text cannot disagree.
                text=original[start:end],
                placeholder=placeholders.get(finding["entity_type"], f"<{finding['entity_type']}>"),
                score=finding.get("score") or 0.0,
            )
        )

    return ChatAnalysis(
        spans=sorted(spans, key=lambda span: span.start),
        masked_text=(result.get("texts") or [original])[0],
        blocked=bool(result.get("blocked")),
        block_reason=result.get("block_reason"),
        counts=result.get("entity_counts") or {},
        latency_ms=int(result.get("latency_ms") or 0),
    )


async def analyze_message(
    text: str,
    *,
    user: AppUser,
    llm_user_id: str,
    pii: PiiServiceClient,
    placeholders: dict[str, str],
    request_id: str | None = None,
) -> ChatAnalysis:
    result = await pii.analyze(
        [text],
        request_id=request_id or f"chat-{uuid.uuid4()}",
        user_id=llm_user_id,
        end_user_id=f"app-user-{user.id}",
        model=None,
        include_context=True,
        message_indices=[0],
        message_roles=["user"],
    )
    return _to_analysis(result, text, placeholders)


async def stream_reply(
    *,
    history: list[dict[str, str]],
    masked_text: str,
    key: UserLlmKey,
    model: str,
    litellm: LiteLlmClient,
) -> AsyncIterator[str]:
    """Send the masked message plus history and stream the reply.

    ``masked_text`` replaces the last user message. The caller has already
    decided the request is not blocked; this function never sees the original.
    """
    messages = [*history[:-1], {"role": "user", "content": masked_text}]
    async for delta in litellm.stream_chat(
        key=key.virtual_key,
        model=model,
        messages=messages,
        end_user_id=key.llm_user_id,
    ):
        yield delta

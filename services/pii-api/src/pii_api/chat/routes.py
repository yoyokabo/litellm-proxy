"""Chat routes.

One streaming endpoint and one analyse-only endpoint.

The stream carries more than tokens. Before a single token exists, the client
already needs to know what was filtered out of the message it just sent, so
the first event on the wire is the analysis. That is why this is a custom SSE
stream rather than a pass-through of LiteLLM's: the feedback and the reply
arrive in one ordered channel the client reads linearly.

Event types, all JSON after ``data:``:

    analysis  what was detected, the masked text, and whether it was blocked
    delta     one chunk of assistant text
    error     an upstream failure, already stripped of any prompt echo
    done      end of turn
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any, Final

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from pii_api.api.schemas import AnalyzeOnlyResponse, ChatRequest
from pii_api.auth.deps import DbSession, RotatedUser
from pii_api.chat.service import analyze_message, ensure_virtual_key, stream_reply
from pii_api.upstream import UpstreamError

__all__ = ["router"]

logger: Final = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _placeholders(request: Request) -> dict[str, str]:
    cached = getattr(request.app.state, "placeholder_map", None)
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    try:
        policy = await request.app.state.pii.policy()
        mapping = {e["entity_type"]: e["placeholder"] for e in policy["entities"]}
    except Exception:
        logger.warning("chat.policy_unavailable")
        return {}
    request.app.state.placeholder_map = mapping
    return mapping


@router.post("/analyze", response_model=AnalyzeOnlyResponse)
async def analyze_only(
    payload: ChatRequest, request: Request, db: DbSession, user: RotatedUser
) -> AnalyzeOnlyResponse:
    """Pre-send detection for the composer.

    Debounced by the client and **off by default** per brief §3: typing into a
    box that fires a detection request on every keystroke is a different
    product, and a noisier one. It writes an audit row like any other
    detection, which is correct -- the text was analysed, and pretending
    otherwise would put a hole in the trail.
    """
    key = await ensure_virtual_key(db, user, request.app.state.litellm)
    analysis = await analyze_message(
        payload.messages[-1].content,
        user=user,
        llm_user_id=key.llm_user_id,
        pii=request.app.state.pii,
        placeholders=await _placeholders(request),
    )
    return AnalyzeOnlyResponse(analysis=analysis)


@router.post("/stream")
async def stream(
    payload: ChatRequest, request: Request, db: DbSession, user: RotatedUser
) -> StreamingResponse:
    """Analyse, report, then forward the masked text and stream the reply."""
    settings = request.app.state.settings
    key = await ensure_virtual_key(db, user, request.app.state.litellm)
    placeholders = await _placeholders(request)
    request_id = f"chat-{uuid.uuid4()}"
    original = payload.messages[-1].content
    model = payload.model or settings.chat_model

    async def events() -> AsyncIterator[str]:
        try:
            analysis = await analyze_message(
                original,
                user=user,
                llm_user_id=key.llm_user_id,
                pii=request.app.state.pii,
                placeholders=placeholders,
                request_id=request_id,
            )
        except Exception:
            logger.exception("chat.analyze_failed", request_id=request_id)
            # Fail closed here, unlike the proxy guardrail. The guardrail
            # defaults open so a detection outage cannot take the whole
            # gateway down for every engineer; this endpoint serves one
            # person who is about to be told their message was screened, and
            # sending it unscreened while implying otherwise is worse than
            # an error message.
            yield _sse("error", {"message": "Detection is unavailable; the message was not sent."})
            yield _sse("done", {})
            return

        yield _sse("analysis", analysis.model_dump(mode="json"))

        if analysis.blocked:
            # Nothing is forwarded. The composer keeps the text.
            logger.info("chat.blocked", request_id=request_id, counts=analysis.block_reason or {})
            yield _sse("done", {"blocked": True})
            return

        history = [message.model_dump() for message in payload.messages]
        try:
            async for delta in stream_reply(
                history=history,
                masked_text=analysis.masked_text,
                key=key,
                model=model,
                litellm=request.app.state.litellm,
            ):
                yield _sse("delta", {"text": delta})
        except UpstreamError as exc:
            logger.warning("chat.upstream_failed", status=exc.status, request_id=request_id)
            yield _sse("error", {"message": f"The model returned an error ({exc.status})."})
        except Exception:
            logger.exception("chat.stream_failed", request_id=request_id)
            yield _sse("error", {"message": "The reply stream failed."})

        logger.info(
            "chat.completed",
            request_id=request_id,
            user_id=key.llm_user_id,
            counts=analysis.counts,
            latency_ms=analysis.latency_ms,
        )
        yield _sse("done", {})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            # nginx sits in front of this in the real deployment and will
            # otherwise buffer the whole stream into one response.
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

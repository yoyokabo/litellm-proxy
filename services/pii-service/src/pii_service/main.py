"""FastAPI application.

Startup order matters and is deliberate:

1. Settings -- which validates the audit pepper and **raises** if it is unset,
   empty, the placeholder, or too short. This happens before anything binds a
   port, so a misconfigured deployment fails to start rather than quietly
   writing worthless fingerprints.
2. Policy -- validated and cross-checked, so a typo in a YAML file is a
   startup error rather than an entity that silently stopped masking.
3. Analyzer -- builds blank spaCy pipelines and compiles every pattern. No
   network access, no model download.
4. Audit sink -- starts the background flusher, which first replays anything
   the WAL is holding from a previous run.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Final

import structlog
from fastapi import FastAPI

from pii_service.api.admin import router as admin_router
from pii_service.api.routes import router
from pii_service.audit.sink import AuditSink
from pii_service.detect.cache import DetectionCache
from pii_service.detect.registry import build_analyzer
from pii_service.detect.router import PiiRouter
from pii_service.logging_config import configure_logging
from pii_service.policy.loader import PolicyBundle, load_policy_bundle
from pii_service.policy.store import PolicyStore
from pii_service.service import PiiService
from pii_service.settings import Settings, get_settings

__all__ = ["create_app"]

VERSION: Final = "0.1.0"

logger: Final = structlog.get_logger(__name__)


def _tier_factories(
    settings: Settings, policy: PolicyBundle
) -> tuple[object | None, object | None]:
    """Resolve the optional tiers, importing them only if enabled.

    A deployment running tier 1 only must not need onnxruntime or gliner
    installed, so these imports live here rather than at module scope. A
    missing wheel for an *enabled* tier is a hard failure -- silently running
    without a tier the operator switched on would reduce detection coverage
    with nothing in the logs to say so.
    """
    tier2 = tier3 = None

    if settings.enable_tier2_arabic_ner:
        from pii_service.detect.tier2_arabic_ner import build_tier2_factory

        tier2 = build_tier2_factory(settings)

    if settings.enable_tier3_gliner:
        from pii_service.detect.tier3_gliner import build_tier3_factory

        # Conditioned on the policy's labels, so administrator-added tier-3
        # labels are passed to GLiNER2 alongside the baseline ones.
        tier3 = build_tier3_factory(settings, policy=policy)

    return tier2, tier3


def _register_tier3_reload(policy_store: PolicyStore, analyzer: object) -> None:
    """Re-prompt every loaded GLiNER recognizer when the policy changes."""
    recognizers = [
        recognizer
        for recognizer in analyzer.registry.recognizers  # type: ignore[attr-defined]
        if hasattr(recognizer, "set_prompts")
    ]
    if not recognizers:
        return

    def reprompt(policy: PolicyBundle) -> None:
        for recognizer in recognizers:
            recognizer.set_prompts(policy.gliner_prompts)

    policy_store.add_listener(reprompt)


def create_app(settings: Settings | None = None, *, sink: AuditSink | None = None) -> FastAPI:
    """Build the application.

    ``sink`` is injectable so tests can supply an engine of their own; in
    production it is always the one built from settings.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    policy = load_policy_bundle(settings.config_dir)
    tier2_factory, tier3_factory = _tier_factories(settings, policy)

    analyzer = build_analyzer(
        policy,
        languages=settings.languages,
        tier2_factory=tier2_factory,  # type: ignore[arg-type]
        tier3_factory=tier3_factory,  # type: ignore[arg-type]
    )
    detection_router = PiiRouter(analyzer, policy)
    sink = sink or AuditSink(settings)
    cache = DetectionCache(policy, max_entries=settings.detection_cache_size)
    policy_store = PolicyStore(policy, sink._engine)
    service = PiiService(
        settings=settings,
        policy=policy,
        router=detection_router,
        sink=sink,
        cache=cache,
        policy_store=policy_store,
    )

    # An administrator adding a tier-3 label changes an argument to GLiNER2,
    # not its weights, so the recognizers are re-prompted in place rather than
    # the analyzer being rebuilt and the model reloaded.
    _register_tier3_reload(policy_store, analyzer)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await sink.start()
        # Pick up administrator changes made before this process started, or by
        # another replica while it was down.
        await policy_store.refresh()
        logger.info(
            "pii-service.started",
            version=VERSION,
            languages=list(settings.languages),
            entities=len(policy.entities),
            tier2=settings.enable_tier2_arabic_ner,
            tier3=settings.enable_tier3_gliner,
            tier3_labels=len(policy_store.current.gliner_prompts),
            admin_api=settings.admin_api_enabled,
        )
        try:
            yield
        finally:
            await sink.stop()
            logger.info("pii-service.stopped", audit=sink.health())

    app = FastAPI(
        title="Arabic PII detection and audit service",
        version=VERSION,
        lifespan=lifespan,
        # No interactive docs in a government deployment by default; the schema
        # is served so clients can generate against it.
        docs_url=None,
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.policy = policy
    app.state.policy_store = policy_store
    app.state.admin_token = settings.admin_token.get_secret_value().strip()
    app.state.service = service
    app.state.sink = sink
    app.state.tiers = {
        "tier1_patterns": True,
        "tier2_arabic_ner": settings.enable_tier2_arabic_ner,
        "tier3_gliner": settings.enable_tier3_gliner,
    }
    app.include_router(router)
    app.include_router(admin_router)
    return app


def get_app() -> FastAPI:
    """ASGI factory, used as ``pii_service.main:get_app`` with uvicorn --factory.

    A factory rather than a module-level ``app``: constructing the application
    validates the pepper and the policy and raises on failure, and an import
    that can raise makes every tool that merely imports this module (a test
    collector, a CLI) fail in a confusing place.
    """
    return create_app()

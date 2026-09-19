# pii-service runtime image.
#
# Two properties this Dockerfile exists to guarantee:
#
# 1. **No PyTorch, no transformers.** Tier 2 runs ONNX through onnxruntime and
#    tier 3 is off by default. A multi-gigabyte torch install in a CPU-only
#    inference image is a packaging bug, and in an air-gapped delivery it is
#    the difference between a `docker load` tarball someone can carry and one
#    they cannot.
#
# 2. **No network access at runtime, including the first request.** spaCy's
#    blank pipelines ship inside the wheel, and detect/nlp_engine.py never
#    calls spacy.cli.download(). Nothing here lazily fetches a model.
#
# Phase 4 adds a builder stage that downloads and ONNX-converts the Arabic NER
# weights and copies only the artifacts into the runtime stage. The seam is
# already here: MODEL_DIR is a volume today and a COPY then.

FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv from PyPI rather than `COPY --from=ghcr.io/astral-sh/uv`. One registry
# instead of two: an air-gapped build already needs a PyPI mirror, and adding a
# container registry to that list is a second thing to mirror and a second
# thing to be blocked by.
RUN pip install --no-cache-dir "uv>=0.5,<1.0"

WORKDIR /build
COPY pyproject.toml ./
COPY src ./src

# Resolve into a self-contained venv so the runtime stage needs no compiler.
RUN uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache .

# ---------------------------------------------------------------------------

FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PII_CONFIG_DIR=/app/config \
    PII_AUDIT_WAL_PATH=/var/lib/pii-service/wal

# curl for the container healthcheck only. Everything else stays out.
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin pii

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=pii:pii src ./src
COPY --chown=pii:pii config ./config
COPY --chown=pii:pii alembic.ini ./alembic.ini

# The WAL is the fail-open spill target. It must be writable by the service
# user or the audit trail is lost precisely when the database is already down.
RUN mkdir -p /var/lib/pii-service/wal && chown -R pii:pii /var/lib/pii-service

USER pii
EXPOSE 8090

# /livez never touches the database, so a database outage does not restart a
# service that is still masking correctly and spilling to the WAL.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8090/livez || exit 1

CMD ["uvicorn", "pii_service.main:get_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8090", "--no-server-header"]

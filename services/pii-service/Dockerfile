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

# No apt, and no compiler. Every runtime dependency resolves to a wheel
# (presidio, spaCy, pydantic, psycopg[binary], onnxruntime), so the build needs
# a PyPI mirror and nothing else. An air-gapped site that has to mirror Debian's
# archives as well to build this image is a site that will not build it.

# Building behind an internal mirror and/or a TLS-intercepting proxy, which is
# the normal case at the sites this ships to:
#
#   docker build \
#     --secret id=pip_ca,src=/etc/ssl/certs/corporate-ca.pem \
#     --build-arg PIP_INDEX_URL=https://nexus.internal/repository/pypi/simple \
#     .
#
# Both are optional. With no secret and no build arg the build uses PyPI and
# the image's own trust store, so nothing here is required in a normal network.
ARG PIP_INDEX_URL=""
ARG PIP_TRUSTED_HOST=""
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

# uv from PyPI rather than `COPY --from=ghcr.io/astral-sh/uv`. One registry
# instead of two: an air-gapped build already needs a PyPI mirror, and adding a
# container registry to that list is a second thing to mirror and a second
# thing to be blocked by.
RUN --mount=type=secret,id=pip_ca \
    if [ -s /run/secrets/pip_ca ]; then \
        export PIP_CERT=/run/secrets/pip_ca SSL_CERT_FILE=/run/secrets/pip_ca; \
    fi; \
    pip install --no-cache-dir "uv>=0.5,<1.0"

WORKDIR /build
COPY pyproject.toml ./
COPY src ./src

# The [ner] extra -- onnxruntime and the Rust tokenizers binding -- is part of
# the image, not an optional add-on, because compose exposes
# PII_ENABLE_TIER2_ARABIC_NER as a runtime switch. An image where flipping that
# flag crashes on a missing wheel makes the flag a lie, and the failure would
# land at startup on the air-gapped host that cannot pip install anything.
#
# It costs about 60 MB and does NOT weaken the guarantee above: onnxruntime and
# tokenizers are self-contained wheels and pull in neither torch nor
# transformers. tests/test_deployment.py asserts both halves of that -- the
# extra is present, the deep-learning frameworks are not.
#
# Tier 3 (gliner) is deliberately NOT installed: it is off by default pending a
# latency decision, and it does depend on torch.
# Resolve into a self-contained venv so the runtime stage needs no compiler.
RUN --mount=type=secret,id=pip_ca \
    if [ -s /run/secrets/pip_ca ]; then \
        export SSL_CERT_FILE=/run/secrets/pip_ca UV_NATIVE_TLS=1; \
    fi; \
    uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache ".[ner]"

# ---------------------------------------------------------------------------

FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PII_CONFIG_DIR=/app/config \
    PII_AUDIT_WAL_PATH=/var/lib/pii-service/wal

# useradd ships in the base image; nothing here needs apt.
RUN useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin pii

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
#
# Python rather than curl: the interpreter is already here, and installing curl
# would mean an apt step, which means mirroring Debian's archives for an
# air-gapped build. One less thing in the image and one less thing to mirror.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8090/livez', timeout=3)"]

CMD ["uvicorn", "pii_service.main:get_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8090", "--no-server-header"]

# pii-api runtime image.
#
# No models and no detection here: this service calls pii-service over HTTP and
# LiteLLM over HTTP, and does nothing else that needs a wheel with a compiler
# in it. Same two-stage shape as pii-service, same no-apt rule, so an
# air-gapped build needs a PyPI mirror and nothing further.

FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

ARG PIP_INDEX_URL=""
ARG PIP_TRUSTED_HOST=""
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

RUN --mount=type=secret,id=pip_ca \
    if [ -s /run/secrets/pip_ca ]; then \
        export PIP_CERT=/run/secrets/pip_ca SSL_CERT_FILE=/run/secrets/pip_ca; \
    fi; \
    pip install --no-cache-dir "uv>=0.5,<1.0"

WORKDIR /build
COPY pyproject.toml ./
COPY src ./src

RUN --mount=type=secret,id=pip_ca \
    if [ -s /run/secrets/pip_ca ]; then \
        export SSL_CERT_FILE=/run/secrets/pip_ca UV_NATIVE_TLS=1; \
    fi; \
    uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache .

# ---------------------------------------------------------------------------

FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN useradd --system --uid 10002 --create-home --shell /usr/sbin/nologin api

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=api:api src ./src
COPY --chown=api:api alembic.ini ./alembic.ini

USER api
EXPOSE 8080

# /livez, not /healthz: healthz reaches pii-service, and an orchestrator must
# not restart a healthy web backend because a dependency is briefly down.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/livez', timeout=3)"]

CMD ["uvicorn", "pii_api.main:get_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8080", "--no-server-header"]

"""Deployment invariants: docker-compose.yml, the Dockerfile, and .env.example.

These need no Docker daemon. They exist because the failures they catch are
quiet ones -- a compose file that publishes the audit database to the host, an
image that quietly acquires PyTorch, or an .env.example placeholder that drifts
away from the constant the service compares it against, silently disarming the
pepper check.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
DOCKERFILE = REPO_ROOT / "services" / "pii-service" / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / "services" / "pii-service" / ".dockerignore"


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dockerfile_instructions() -> str:
    """The Dockerfile with comments stripped.

    The comments explain at length why PyTorch and spacy.cli.download are
    absent, so a naive substring scan over the whole file finds the words it is
    looking for in the very prose asserting they do not appear.
    """
    lines = [
        line
        for line in DOCKERFILE.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compose: the services the brief's done-criterion 1 names
# ---------------------------------------------------------------------------


def test_the_stack_defines_service_and_database(compose: dict[str, Any]) -> None:
    assert {"pii-service", "pii-db"} <= set(compose["services"])


def test_it_joins_the_existing_litellm_network_as_external(compose: dict[str, Any]) -> None:
    """External, because this attaches to a running deployment it does not own."""
    litellm = compose["networks"]["litellm"]
    assert litellm["external"] is True
    assert "litellm" in compose["services"]["pii-service"]["networks"]


def test_the_database_is_separate_from_litellms(compose: dict[str, Any]) -> None:
    """The brief forbids touching LiteLLM's Postgres."""
    database = compose["services"]["pii-db"]
    assert database["image"].startswith("postgres:16")
    # Its own volume, and not on the litellm network at all.
    assert database["networks"] == ["pii-internal"]


def test_the_audit_database_is_not_published_to_the_host(compose: dict[str, Any]) -> None:
    """An audit DB on a host port is one somebody queries from their laptop."""
    assert "ports" not in compose["services"]["pii-db"]


def test_the_service_is_not_published_to_the_host_by_default(compose: dict[str, Any]) -> None:
    assert "ports" not in compose["services"]["pii-service"]


def test_migrations_run_before_the_service_starts(compose: dict[str, Any]) -> None:
    depends = compose["services"]["pii-service"]["depends_on"]
    assert depends["pii-migrate"]["condition"] == "service_completed_successfully"
    assert depends["pii-db"]["condition"] == "service_healthy"


def test_the_database_has_a_healthcheck(compose: dict[str, Any]) -> None:
    assert "healthcheck" in compose["services"]["pii-db"]


@pytest.mark.parametrize("variable", ["PII_AUDIT_PEPPER", "PII_DB_PASSWORD"])
def test_required_secrets_fail_the_run_when_unset(compose: dict[str, Any], variable: str) -> None:
    """${VAR:?msg} turns a missing secret into a refusal, not a default."""
    rendered = COMPOSE_FILE.read_text(encoding="utf-8")
    assert re.search(rf"\$\{{{variable}:\?", rendered), (
        f"{variable} must use ${{{variable}:?message}} so compose refuses to start without it"
    )


def test_policy_is_mounted_read_only(compose: dict[str, Any]) -> None:
    """Config is policy; the service must not be able to rewrite it."""
    mounts = compose["services"]["pii-service"]["volumes"]
    config = [m for m in mounts if "/app/config" in m]
    assert config and all(m.endswith(":ro") for m in config)


def test_the_wal_is_a_named_volume(compose: dict[str, Any]) -> None:
    """Losing the spill means losing audit records the DB had not accepted."""
    mounts = compose["services"]["pii-service"]["volumes"]
    assert any(m.startswith("pii-wal:") for m in mounts)
    assert "pii-wal" in compose["volumes"]


def test_the_database_volume_is_named(compose: dict[str, Any]) -> None:
    assert "pii-db-data" in compose["volumes"]


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------


def test_the_image_never_installs_pytorch_or_transformers(
    dockerfile_instructions: str,
) -> None:
    """A CPU-only inference image carrying torch is a packaging bug (brief §4)."""
    lowered = dockerfile_instructions.lower()
    for forbidden in ("torch", "transformers", "cuda", "nvidia"):
        assert forbidden not in lowered, f"{forbidden} must not appear in the Dockerfile"


def test_the_build_is_multi_stage(dockerfile: str) -> None:
    assert len(re.findall(r"^FROM ", dockerfile, re.MULTILINE)) >= 2


def test_the_service_runs_as_a_non_root_user(dockerfile: str) -> None:
    users = re.findall(r"^USER\s+(\S+)", dockerfile, re.MULTILINE)
    assert users, "the image must declare a USER"
    assert users[-1] != "root"


def test_the_healthcheck_does_not_touch_the_database(dockerfile: str) -> None:
    """/livez, not /health: a DB outage must not restart a masking service."""
    healthcheck = re.search(r"HEALTHCHECK.*?CMD\s+(.+)", dockerfile, re.DOTALL)
    assert healthcheck is not None
    assert "/livez" in healthcheck.group(1)
    assert "/health" not in healthcheck.group(1).replace("/livez", "")


def test_the_wal_directory_is_writable_by_the_service_user(dockerfile: str) -> None:
    assert "chown -R pii:pii /var/lib/pii-service" in dockerfile


def test_no_spacy_model_download_at_build_or_run(dockerfile_instructions: str) -> None:
    """Air-gapped: nothing may reach for a model."""
    assert "spacy download" not in dockerfile_instructions
    assert "spacy.cli.download" not in dockerfile_instructions


def test_the_dockerignore_excludes_the_local_venv_and_env() -> None:
    """Otherwise a 480MB venv is uploaded and a stale .env can be baked in."""
    ignored = DOCKERIGNORE.read_text(encoding="utf-8").split()
    assert ".venv/" in ignored
    assert ".env" in ignored


# ---------------------------------------------------------------------------
# .env.example
# ---------------------------------------------------------------------------


def test_the_example_pepper_matches_the_constant_the_service_rejects() -> None:
    """If these drift apart, the placeholder check silently stops working.

    The service refuses to start on EXAMPLE_PEPPER. Should someone edit
    .env.example without editing settings.py, a deployment that never changed
    the placeholder would start happily with a pepper published in this repo.
    """
    from pii_service.settings import EXAMPLE_PEPPER

    content = ENV_EXAMPLE.read_text(encoding="utf-8")
    match = re.search(r"^PII_AUDIT_PEPPER=(.+)$", content, re.MULTILINE)

    assert match is not None, ".env.example must define PII_AUDIT_PEPPER"
    assert match.group(1).strip() == EXAMPLE_PEPPER


def test_the_example_env_holds_no_real_looking_secret() -> None:
    content = ENV_EXAMPLE.read_text(encoding="utf-8")
    for line in content.splitlines():
        if "PASSWORD" in line or "PEPPER" in line:
            _, _, value = line.partition("=")
            if value.strip():
                assert "CHANGE_ME" in value, f"placeholder expected in: {line}"


def test_tiers_two_and_three_ship_disabled() -> None:
    content = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "PII_ENABLE_TIER2_ARABIC_NER=false" in content
    assert "PII_ENABLE_TIER3_GLINER=false" in content


def test_every_compose_variable_is_documented_in_the_example() -> None:
    """A variable compose reads but .env.example never mentions is a trap."""
    rendered = COMPOSE_FILE.read_text(encoding="utf-8")
    referenced = {
        name
        for name in re.findall(r"\$\{([A-Z0-9_]+)[:?\-}]", rendered)
        if name.startswith("PII_") or name == "LITELLM_NETWORK"
    }
    documented = set(re.findall(r"^([A-Z0-9_]+)=", ENV_EXAMPLE.read_text(encoding="utf-8"), re.M))

    undocumented = referenced - documented
    assert not undocumented, f"not documented in .env.example: {sorted(undocumented)}"

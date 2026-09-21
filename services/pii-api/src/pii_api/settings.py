"""Configuration for the web backend.

Two settings here are load-bearing and validated at startup rather than on
first use: ``admin_initial_password`` and ``litellm_master_key``. A service
that starts without them appears healthy and then fails the first time someone
tries to log in or send a message, which is the worst moment to discover a
deployment mistake.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Final, Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["EXAMPLE_ADMIN_PASSWORD", "MIN_ADMIN_PASSWORD_LENGTH", "Settings", "get_settings"]

# The literal shipped in .env.example. Same reasoning as the audit pepper in
# pii-service: a value every reader of this repository knows is not a secret,
# and here it is the bootstrap credential for the admin account.
EXAMPLE_ADMIN_PASSWORD: Final = "CHANGE_ME__initial_admin_password"  # noqa: S105

MIN_ADMIN_PASSWORD_LENGTH: Final = 12


class ConfigurationError(RuntimeError):
    """A required setting is missing or unusable. The service must not start."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PII_API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"  # noqa: S104 -- containerized; the network is the boundary
    port: int = 8080
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # -- storage -----------------------------------------------------------
    # The same database pii-service writes audit rows to. This service owns
    # app_users / app_sessions / span_flags and reads pii_events; it never
    # writes pii_events, because pii-service is the single audit writer
    # (brief §3) and two writers is how audit trails start disagreeing.
    database_url: str = "postgresql+psycopg://pii:pii@pii-db:5432/pii"

    # -- upstreams ---------------------------------------------------------
    pii_service_url: str = "http://pii-service:8090"
    litellm_url: str = "http://litellm:4000"
    litellm_master_key: SecretStr = SecretStr("")
    # Model alias as it appears in the proxy's config.yaml.
    chat_model: str = "local-llama"
    upstream_timeout_seconds: Annotated[float, Field(gt=0)] = 120.0

    # -- admin bootstrap (brief §10) --------------------------------------
    admin_email: str = "admin@example.com"
    admin_initial_password: SecretStr = SecretStr("")

    # -- sessions ----------------------------------------------------------
    session_ttl_hours: Annotated[int, Field(ge=1)] = 12
    # Off for local http development, on everywhere else. A session cookie
    # without Secure is a session cookie that rides a downgraded request.
    cookie_secure: bool = True
    cookie_name: str = "pii_session"

    @model_validator(mode="after")
    def _validate(self) -> Self:
        password = self.admin_initial_password.get_secret_value().strip()

        if not password:
            raise ConfigurationError(
                "PII_API_ADMIN_INITIAL_PASSWORD is unset. It bootstraps the first admin "
                "account, which is flagged must_change_password and cannot use the admin "
                "routes until it is rotated. Generate one with `openssl rand -base64 24`."
            )
        if password == EXAMPLE_ADMIN_PASSWORD:
            raise ConfigurationError(
                "PII_API_ADMIN_INITIAL_PASSWORD is still the placeholder from .env.example. "
                "Every reader of this repository knows it, and it is the credential for an "
                "account that can read the whole audit trail."
            )
        if len(password) < MIN_ADMIN_PASSWORD_LENGTH:
            raise ConfigurationError(
                f"PII_API_ADMIN_INITIAL_PASSWORD must be at least "
                f"{MIN_ADMIN_PASSWORD_LENGTH} characters; got {len(password)}."
            )
        if not self.litellm_master_key.get_secret_value().strip():
            raise ConfigurationError(
                "PII_API_LITELLM_MASTER_KEY is unset. It is needed to mint one virtual key "
                "per application user, which is what puts a real user id on every audit row "
                "the chat produces."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

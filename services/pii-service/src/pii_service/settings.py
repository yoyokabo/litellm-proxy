"""Service configuration.

The load-bearing part of this module is ``audit_pepper``. Everything else is
ordinary configuration; the pepper is the difference between an audit database
that answers investigative questions and one that is itself the breach.

Why a pepper and not a hash (brief §6): the Egyptian national ID space is
trivially enumerable. Fourteen digits, but the first seven are a century digit
and a birth date, the next two a governorate from a list of 27, and the last a
check digit -- so the real search space is about 10^4 per (birth date,
governorate) pair. A dump of SHA-256(nid) can be inverted exhaustively on a
laptop in minutes. Storing plain hashes is therefore equivalent to storing the
IDs, and a "we only store hashes" claim in a government security review would
be false.

HMAC with a key held outside the database breaks that: an attacker with the
table and without the pepper has nothing to enumerate against.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__ = ["EXAMPLE_PEPPER", "MIN_PEPPER_LENGTH", "Settings", "get_settings"]

# The literal value shipped in .env.example. A deployment that never changed it
# has a pepper every reader of this repository knows, which is no pepper.
EXAMPLE_PEPPER: Final = "CHANGE_ME__openssl_rand_hex_32"

# 32 hex characters is 128 bits. Below that the pepper is brute-forceable
# against a known (value, fingerprint) pair, which an insider can always
# manufacture by sending themselves one request.
MIN_PEPPER_LENGTH: Final = 32


class PepperError(RuntimeError):
    """The audit pepper is missing or unusable. The service must not start."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PII_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- service -----------------------------------------------------------
    host: str = "0.0.0.0"  # noqa: S104 -- containerized; the network is the boundary
    port: int = 8090
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    config_dir: Path = Path("config")

    # -- audit -------------------------------------------------------------
    audit_pepper: SecretStr = SecretStr("")
    database_url: str = "postgresql+psycopg://pii:pii@pii-db:5432/pii"

    # The audit write must never sit in the request path (brief §8).
    audit_queue_max: Annotated[int, Field(ge=1)] = 10_000
    audit_batch_size: Annotated[int, Field(ge=1)] = 200
    audit_flush_interval_seconds: Annotated[float, Field(gt=0)] = 1.0

    # Fail open: if the DB is unreachable, spill here and keep serving. A
    # guardrail that fails requests because its audit log is down is a
    # self-inflicted outage.
    audit_wal_path: Path = Path("/var/lib/pii-service/wal")
    audit_wal_enabled: bool = True

    # -- detection ---------------------------------------------------------
    # NoDecode is required, not stylistic. Without it pydantic-settings tries
    # to JSON-decode any complex-typed value coming from the environment, so
    # PII_LANGUAGES=en,ar raises SettingsError before the validator below ever
    # runs -- a crash that appears only once the variable is actually set,
    # which in practice means only inside the container.
    languages: Annotated[tuple[str, ...], NoDecode] = ("en", "ar")
    enable_tier2_arabic_ner: bool = False
    enable_tier3_gliner: bool = False
    tier2_model_dir: Path | None = None
    detection_cache_size: Annotated[int, Field(ge=0)] = 2048

    # -- admin API ---------------------------------------------------------
    # Bearer token for /admin. Empty disables those routes entirely, which is
    # the default: a deployment that has not deliberately turned on runtime
    # policy editing should not expose it.
    #
    # A placeholder for the phase-2 role auth the brief specifies
    # (ADMIN_EMAIL / ADMIN_INITIAL_PASSWORD, admin and auditor roles). When
    # that lands this goes away.
    admin_token: SecretStr = SecretStr("")

    # -- break-glass (brief §6) -------------------------------------------
    # Defined in the data model, unassigned and disabled in phase 1.
    enable_reveal_endpoint: bool = False

    @field_validator("languages", mode="before")
    @classmethod
    def _split_languages(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @model_validator(mode="after")
    def _validate_pepper(self) -> Self:
        """Refuse to start on an unset, empty, example or too-short pepper.

        This is deliberately a hard failure rather than a warning. A service
        that starts with a weak pepper writes fingerprints that are worthless
        but indistinguishable from good ones, and nobody discovers that until
        the table is already full of them -- at which point the fix requires
        re-peppering history that no longer has the values to re-derive from.
        """
        pepper = self.audit_pepper.get_secret_value()

        if not pepper.strip():
            raise PepperError(
                "PII_AUDIT_PEPPER is unset or empty. Generate one with "
                "`openssl rand -hex 32` and set it in the environment. "
                "See the README section 'Generating the audit pepper'."
            )
        if pepper.strip() == EXAMPLE_PEPPER:
            raise PepperError(
                "PII_AUDIT_PEPPER is still the placeholder from .env.example. "
                "Every reader of this repository knows that value, so "
                "fingerprints computed with it are reversible by enumeration. "
                "Generate a real one with `openssl rand -hex 32`."
            )
        if len(pepper.strip()) < MIN_PEPPER_LENGTH:
            raise PepperError(
                f"PII_AUDIT_PEPPER must be at least {MIN_PEPPER_LENGTH} characters "
                f"({MIN_PEPPER_LENGTH * 4} bits); got {len(pepper.strip())}. "
                "Generate one with `openssl rand -hex 32`."
            )
        return self

    @property
    def admin_api_enabled(self) -> bool:
        return bool(self.admin_token.get_secret_value().strip())

    @property
    def pepper_bytes(self) -> bytes:
        return self.audit_pepper.get_secret_value().strip().encode("utf-8")

    @property
    def enable_reveal(self) -> bool:
        """Break-glass reveal stays off in phase 1 regardless of configuration."""
        return False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Cached so the pepper is validated exactly once."""
    return Settings()

"""Audit path: fingerprints, records, the WAL, and the sink.

The sink runs against an in-memory SQLite async engine. That is not Postgres,
but the properties under test here -- batching, never raising into the caller,
spilling on failure, replaying on startup -- are database-independent, and
testing them without a container means they are actually tested.
"""

from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from pii_service.audit.fingerprint import FINGERPRINT_HEX_LENGTH, canonicalize, fingerprint
from pii_service.audit.records import AuditRecord, RequestContext
from pii_service.audit.sink import AuditSink
from pii_service.audit.wal import AuditWal
from pii_service.db.models import Base, PiiEvent
from pii_service.detect.router import DetectedSpan
from pii_service.policy.models import EntityAction, EntityCategory
from pii_service.settings import EXAMPLE_PEPPER, Settings
from pii_service.spans import PreparedSpan
from pii_service.synthetic import (
    synthetic_mobile,
    synthetic_national_id,
    to_arabic_indic,
)

PEPPER = b"0123456789abcdef0123456789abcdef"


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def test_fingerprint_is_sixteen_hex_characters() -> None:
    value = fingerprint("EG_NATIONAL_ID", "28503122148219", PEPPER)
    assert value is not None
    assert len(value) == FINGERPRINT_HEX_LENGTH
    assert all(ch in "0123456789abcdef" for ch in value)


def test_same_id_in_arabic_and_ascii_digits_collides() -> None:
    """The whole investigative pivot depends on this."""
    nid = synthetic_national_id(rng=random.Random(2))
    assert fingerprint("EG_NATIONAL_ID", nid, PEPPER) == fingerprint(
        "EG_NATIONAL_ID", to_arabic_indic(nid), PEPPER
    )


def test_mobile_collides_across_local_international_and_spacing() -> None:
    mobile = synthetic_mobile(prefix="010", rng=random.Random(3))
    forms = [
        mobile,
        f"+20{mobile[1:]}",
        f"0020{mobile[1:]}",
        f"{mobile[:3]} {mobile[3:7]} {mobile[7:]}",
        f"{mobile[:3]}-{mobile[3:7]}-{mobile[7:]}",
        to_arabic_indic(mobile),
    ]
    fingerprints = {fingerprint("EG_MOBILE", form, PEPPER) for form in forms}
    assert len(fingerprints) == 1


def test_name_collides_across_diacritics_and_orthography() -> None:
    forms = ["محمد", "مُحَمَّد", "محمـــد"]
    assert len({fingerprint("AR_PERSON", form, PEPPER) for form in forms}) == 1


def test_different_values_do_not_collide() -> None:
    a = fingerprint("EG_NATIONAL_ID", "28503122148219", PEPPER)
    b = fingerprint("EG_NATIONAL_ID", "28503122148210", PEPPER)
    assert a != b


def test_different_peppers_give_different_fingerprints() -> None:
    """A pepper rotation must not leave old fingerprints correlatable."""
    a = fingerprint("EG_NATIONAL_ID", "28503122148219", PEPPER)
    b = fingerprint("EG_NATIONAL_ID", "28503122148219", b"a-completely-different-pepper!!!")
    assert a != b


def test_empty_pepper_is_refused() -> None:
    """An unkeyed fingerprint over an enumerable space is the value itself."""
    with pytest.raises(ValueError, match="empty pepper"):
        fingerprint("EG_NATIONAL_ID", "28503122148219", b"")


def test_value_that_canonicalizes_away_has_no_fingerprint() -> None:
    assert fingerprint("EG_NATIONAL_ID", "   ", PEPPER) is None
    assert fingerprint("AR_PERSON", "!!!", PEPPER) is None


def test_canonicalize_strips_to_digits_for_structured_ids() -> None:
    assert canonicalize("EG_NATIONAL_ID", "2850-3122-148219") == "28503122148219"
    assert canonicalize("CREDIT_CARD", "4111 1111 1111 1111") == "4111111111111111"


def test_fingerprint_does_not_contain_the_value() -> None:
    nid = "28503122148219"
    value = fingerprint("EG_NATIONAL_ID", nid, PEPPER)
    assert value is not None
    assert nid not in value
    for window in range(4, len(nid)):
        assert nid[:window] not in value


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _span(entity_type: str = "EG_NATIONAL_ID", value: str = "28503122148219") -> DetectedSpan:
    return DetectedSpan(
        entity_type=entity_type,
        recognizer="EgyptianNationalIdRecognizer",
        score=1.0,
        start=10,
        end=10 + len(value),
        value=value,
        action=EntityAction.MASK,
        category=EntityCategory.ID,
        lang="ar",
    )


def _prepared(
    policy: object,
    entity_type: str = "EG_NATIONAL_ID",
    value: str = "28503122148219",
) -> PreparedSpan:
    return PreparedSpan.from_detected(
        _span(entity_type, value),
        text_index=0,
        policy=policy,
        pepper=PEPPER,  # type: ignore[arg-type]
    )


def test_record_has_no_value_field(policy: object) -> None:
    record = AuditRecord.from_prepared(_prepared(policy), RequestContext(request_id="r1"))
    row = record.to_row()

    assert "value" not in row
    assert "28503122148219" not in record.to_json()


def test_record_carries_fingerprint_preview_and_length(policy: object) -> None:
    record = AuditRecord.from_prepared(
        _prepared(policy), RequestContext(request_id="r1", user_id="u1")
    )

    assert record.value_fp == fingerprint("EG_NATIONAL_ID", "28503122148219", PEPPER)
    assert record.preview == "2850******8219"
    assert record.value_len == 14
    assert record.user_id == "u1"


def test_person_record_has_no_preview(policy: object) -> None:
    record = AuditRecord.from_prepared(
        _prepared(policy, "AR_PERSON", "محمد علي"), RequestContext(request_id="r1")
    )
    assert record.preview is None
    assert record.value_fp is not None  # still correlatable


def test_record_round_trips_through_json(policy: object) -> None:
    original = AuditRecord.from_prepared(
        _prepared(policy),
        RequestContext(request_id="r1", user_id="u1", model="qwen3"),
        latency_ms=12,
    )
    restored = AuditRecord.from_json(original.to_json())
    assert restored == original


# ---------------------------------------------------------------------------
# WAL
# ---------------------------------------------------------------------------


def _records(count: int, policy: object) -> list[AuditRecord]:
    return [
        AuditRecord.from_prepared(_prepared(policy), RequestContext(request_id=f"r{i}"))
        for i in range(count)
    ]


def test_wal_append_and_drain(tmp_path: Path, policy: object) -> None:
    wal = AuditWal(tmp_path)
    written = wal.append(_records(5, policy))
    assert written == 5
    assert wal.pending_count() == 5

    drained = [(segment, records) for segment, records in wal.drain()]
    assert sum(len(records) for _, records in drained) == 5


def test_wal_discard_removes_the_segment(tmp_path: Path, policy: object) -> None:
    wal = AuditWal(tmp_path)
    wal.append(_records(3, policy))
    for segment, _ in list(wal.drain()):
        wal.discard(segment)
    assert wal.pending_count() == 0


def test_wal_survives_a_torn_line(tmp_path: Path, policy: object) -> None:
    """A partial write at the tail must not strand the records behind it."""
    wal = AuditWal(tmp_path)
    wal.append(_records(3, policy))
    segment = wal.segments()[0]
    with segment.open("a", encoding="utf-8") as handle:
        handle.write('{"request_id": "torn", "entity_ty')

    drained = [records for _, records in wal.drain()]
    assert sum(len(r) for r in drained) == 3


def test_wal_rolls_segments_and_drops_the_oldest(tmp_path: Path, policy: object) -> None:
    wal = AuditWal(tmp_path, max_segment_bytes=512, max_segments=2)
    for _ in range(40):
        wal.append(_records(5, policy))

    assert len(wal.segments()) <= 2
    assert wal.dropped_segments > 0, "bound was never exercised"


def test_wal_contains_no_pii(tmp_path: Path, policy: object) -> None:
    nid = synthetic_national_id(rng=random.Random(8))
    wal = AuditWal(tmp_path)
    wal.append(
        [AuditRecord.from_prepared(_prepared(policy, value=nid), RequestContext(request_id="r"))]
    )
    contents = "".join(p.read_text(encoding="utf-8") for p in wal.segments())
    assert nid not in contents


def test_wal_empty_directory_is_harmless(tmp_path: Path) -> None:
    wal = AuditWal(tmp_path / "missing")
    assert wal.segments() == []
    assert wal.pending_count() == 0
    assert list(wal.drain()) == []


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "audit_pepper": PEPPER.decode(),
        "database_url": "sqlite+aiosqlite:///:memory:",
        "audit_wal_path": tmp_path / "wal",
        "audit_batch_size": 50,
        "audit_flush_interval_seconds": 0.05,
    }
    return Settings(**{**defaults, **overrides})  # type: ignore[arg-type]


async def _sqlite_engine():  # type: ignore[no-untyped-def]
    # StaticPool is required, not a tuning choice: every connection to
    # ":memory:" otherwise opens a *separate* empty database, so the sink's
    # connection would never see the tables created here.
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine


async def _row_count(engine: object) -> int:
    async with engine.connect() as connection:  # type: ignore[attr-defined]
        result = await connection.execute(select(func.count()).select_from(PiiEvent))
        return int(result.scalar_one())


async def test_sink_writes_records_to_the_database(tmp_path: Path, policy: object) -> None:
    engine = await _sqlite_engine()
    sink = AuditSink(_settings(tmp_path), engine=engine)
    await sink.start()

    sink.submit(_records(7, policy))
    await asyncio.sleep(0.3)
    await sink.stop()

    assert sink.stats.written == 7


async def test_submit_never_blocks_or_raises(tmp_path: Path, policy: object) -> None:
    """The request path must be immune to anything the audit layer does."""
    engine = await _sqlite_engine()
    sink = AuditSink(_settings(tmp_path, audit_queue_max=2), engine=engine)
    # Deliberately not started: nothing is draining the queue.
    sink.submit(_records(50, policy))

    assert sink.stats.submitted == 50
    assert sink.stats.spilled >= 48  # overflow went to the WAL, not to an exception


async def test_unreachable_database_spills_instead_of_failing(
    tmp_path: Path, policy: object
) -> None:
    engine = create_async_engine("postgresql+psycopg://nobody@127.0.0.1:1/nothing")
    sink = AuditSink(_settings(tmp_path), engine=engine)
    await sink.start()

    sink.submit(_records(4, policy))
    await asyncio.sleep(0.4)
    await sink.stop()

    assert sink.stats.written == 0
    assert sink.stats.spilled == 4
    assert sink.stats.flush_failures >= 1


async def test_spilled_records_replay_on_next_startup(tmp_path: Path, policy: object) -> None:
    settings = _settings(tmp_path)

    # First run: database down, records spill.
    down = create_async_engine("postgresql+psycopg://nobody@127.0.0.1:1/nothing")
    first = AuditSink(settings, engine=down)
    await first.start()
    first.submit(_records(6, policy))
    await asyncio.sleep(0.4)
    await first.stop()
    assert first.stats.spilled == 6

    # Second run: database back, WAL is replayed.
    engine = await _sqlite_engine()
    second = AuditSink(settings, engine=engine)
    await second.start()
    await asyncio.sleep(0.3)

    assert second.stats.replayed == 6
    assert await _row_count(engine) == 6
    await second.stop()


async def test_failed_replay_keeps_the_segment(tmp_path: Path, policy: object) -> None:
    """A segment must not be deleted before its insert is committed."""
    settings = _settings(tmp_path)
    wal = AuditWal(settings.audit_wal_path)
    wal.append(_records(3, policy))

    down = create_async_engine("postgresql+psycopg://nobody@127.0.0.1:1/nothing")
    sink = AuditSink(settings, engine=down)
    await sink.start()
    await asyncio.sleep(0.2)
    await sink.stop()

    assert wal.pending_count() >= 3


async def test_stop_drains_outstanding_records(tmp_path: Path, policy: object) -> None:
    engine = await _sqlite_engine()
    sink = AuditSink(_settings(tmp_path, audit_flush_interval_seconds=5.0), engine=engine)
    await sink.start()
    sink.submit(_records(3, policy))
    await sink.stop()

    assert sink.stats.written + sink.stats.spilled == 3


async def test_health_reports_queue_and_wal_state(tmp_path: Path, policy: object) -> None:
    engine = await _sqlite_engine()
    sink = AuditSink(_settings(tmp_path), engine=engine)
    health = sink.health()

    assert set(health) >= {"submitted", "written", "spilled", "queue_depth", "wal_segments"}
    await sink.stop()


async def test_submitting_nothing_is_a_noop(tmp_path: Path) -> None:
    engine = await _sqlite_engine()
    sink = AuditSink(_settings(tmp_path), engine=engine)
    sink.submit([])
    assert sink.stats.submitted == 0
    await sink.stop()


# ---------------------------------------------------------------------------
# Pepper enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pepper", ["", "   ", EXAMPLE_PEPPER, f"  {EXAMPLE_PEPPER}  ", "short"])
def test_service_refuses_to_start_without_a_real_pepper(pepper: str) -> None:
    with pytest.raises(Exception, match="PII_AUDIT_PEPPER"):
        Settings(audit_pepper=pepper)  # type: ignore[arg-type]


def test_a_real_pepper_is_accepted() -> None:
    settings = Settings(audit_pepper="a" * 32)  # type: ignore[arg-type]
    assert settings.pepper_bytes == b"a" * 32


def test_pepper_is_not_in_the_settings_repr() -> None:
    settings = Settings(audit_pepper="s3cr3t" + "x" * 30)  # type: ignore[arg-type]
    assert "s3cr3t" not in repr(settings)
    assert "s3cr3t" not in str(settings)


def test_break_glass_reveal_is_off_in_phase_one() -> None:
    settings = Settings(audit_pepper="a" * 32, enable_reveal_endpoint=True)  # type: ignore[arg-type]
    assert settings.enable_reveal is False


def test_record_timestamp_is_timezone_aware(policy: object) -> None:
    record = AuditRecord.from_prepared(_prepared(policy), RequestContext(request_id="r"))
    assert record.ts.tzinfo is not None
    assert record.ts.astimezone(UTC) <= datetime.now(UTC)


# ---------------------------------------------------------------------------
# Settings parsed from the environment
# ---------------------------------------------------------------------------


def test_languages_parses_from_a_comma_separated_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The container sets PII_LANGUAGES=en,ar, and that must not crash.

    pydantic-settings JSON-decodes complex-typed values coming from the
    environment before any validator runs, so without NoDecode this raises
    SettingsError. It is invisible in tests that construct Settings with
    keyword arguments -- it only appears once the variable is really set, which
    in practice means only inside the container.
    """
    monkeypatch.setenv("PII_AUDIT_PEPPER", "a" * 32)
    monkeypatch.setenv("PII_LANGUAGES", "en,ar")

    assert Settings().languages == ("en", "ar")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("ar", ("ar",)), ("en, ar", ("en", "ar")), ("en , ar , fr", ("en", "ar", "fr"))],
)
def test_languages_tolerates_spacing(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: tuple[str, ...]
) -> None:
    monkeypatch.setenv("PII_AUDIT_PEPPER", "a" * 32)
    monkeypatch.setenv("PII_LANGUAGES", raw)
    assert Settings().languages == expected


def test_every_env_var_the_compose_file_sets_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct Settings exactly as docker-compose.yml would.

    Catches the whole class of bug above: a field whose type cannot be parsed
    from the string form the compose file actually supplies.
    """
    for name, value in {
        "PII_AUDIT_PEPPER": "b" * 32,
        "PII_DATABASE_URL": "postgresql+psycopg://pii:pw@pii-db:5432/pii",
        "PII_LOG_LEVEL": "INFO",
        "PII_LANGUAGES": "en,ar",
        "PII_ENABLE_TIER2_ARABIC_NER": "false",
        "PII_ENABLE_TIER3_GLINER": "false",
        "PII_TIER2_MODEL_DIR": "/models/arabic-ner",
        "PII_DETECTION_CACHE_SIZE": "2048",
        "PII_CONFIG_DIR": "/app/config",
        "PII_AUDIT_WAL_PATH": "/var/lib/pii-service/wal",
    }.items():
        monkeypatch.setenv(name, value)

    settings = Settings()
    assert settings.languages == ("en", "ar")
    assert settings.detection_cache_size == 2048
    assert settings.enable_tier2_arabic_ner is False
    assert str(settings.config_dir) == "/app/config"

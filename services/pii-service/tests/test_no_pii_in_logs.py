"""The brief's §12 requirement: no matched value may ever reach a log line.

Two complementary approaches, because neither alone is enough.

``test_no_fixture_pii_appears_in_captured_logs`` is the grep the brief asks
for: it drives a realistic workload through the whole stack with known
synthetic values, captures everything written to the logging system, and fails
if any of those values appears. That catches accidents anywhere in the call
graph, including inside libraries.

The rest are targeted tests on the mechanisms that are supposed to make such an
accident impossible in the first place -- the redacting processor, the
redacting ``__repr__``, the value-free record. Those give a readable failure
that says *which* guarantee broke, where the grep only says "something did".
"""

from __future__ import annotations

import logging
import random
from datetime import date
from pathlib import Path

import pytest
import structlog

from pii_service.audit.records import AuditRecord, RequestContext
from pii_service.detect.router import PiiRouter
from pii_service.logging_config import REDACTED, configure_logging, redact_text_keys
from pii_service.spans import PreparedSpan
from pii_service.synthetic import (
    synthetic_iban,
    synthetic_mobile,
    synthetic_national_id,
    synthetic_tax_id,
    to_arabic_indic,
)

PEPPER = b"0123456789abcdef0123456789abcdef"


@pytest.fixture
def known_pii() -> dict[str, str]:
    """Distinctive synthetic values to grep for."""
    rng = random.Random(777)
    nid = synthetic_national_id(birth_date=date(1985, 3, 12), governorate_code="21", serial=4821)
    return {
        "nid": nid,
        "nid_arabic": to_arabic_indic(nid),
        "mobile": synthetic_mobile(prefix="010", rng=rng),
        "iban": synthetic_iban(rng=rng),
        "tax": synthetic_tax_id(rng=rng),
    }


def test_no_fixture_pii_appears_in_captured_logs(
    router: PiiRouter,
    policy: object,
    known_pii: dict[str, str],
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run a realistic workload at DEBUG and grep everything it emitted."""
    configure_logging("DEBUG")

    texts = [
        f"my national id is {known_pii['nid']}",
        f"الرقم القومي {known_pii['nid_arabic']} والموبايل {known_pii['mobile']}",
        f"iban {known_pii['iban']} and الرقم الضريبي {known_pii['tax']}",
        "nothing sensitive at all",
    ]

    with caplog.at_level(logging.DEBUG):
        for index, text in enumerate(texts):
            outcome = router.analyze(text)

            # Exercise the paths most likely to leak: logging the objects.
            logging.getLogger("workload").debug("analyzed %s", outcome.spans)
            logging.getLogger("workload").debug("outcome repr %r", outcome.spans)

            for span in outcome.spans:
                prepared = PreparedSpan.from_detected(
                    span,
                    text_index=index,
                    policy=policy,
                    pepper=PEPPER,  # type: ignore[arg-type]
                )
                record = AuditRecord.from_prepared(
                    prepared, RequestContext(request_id=f"req-{index}")
                )
                logging.getLogger("workload").debug("record %s", record)
                logging.getLogger("workload").debug("json %s", record.to_json())

    emitted = "\n".join(
        [caplog.text, *(str(r.getMessage()) for r in caplog.records), capsys.readouterr().out]
    )

    for name, value in known_pii.items():
        assert value not in emitted, f"{name} leaked into the logs"

    # Sanity: the workload really did run and really did find things.
    assert "EG_NATIONAL_ID" in emitted


def test_detected_span_repr_is_redacted(router: PiiRouter, known_pii: dict[str, str]) -> None:
    outcome = router.analyze(f"id {known_pii['nid']}")
    assert outcome.spans

    for span in outcome.spans:
        for rendering in (repr(span), str(span), format(span), f"{span}", f"{span!r}"):
            assert known_pii["nid"] not in rendering
        assert "<redacted>" in repr(span)


def test_prepared_span_has_no_value_attribute(
    router: PiiRouter, policy: object, known_pii: dict[str, str]
) -> None:
    outcome = router.analyze(f"id {known_pii['nid']}")
    prepared = PreparedSpan.from_detected(
        outcome.spans[0],
        text_index=0,
        policy=policy,
        pepper=PEPPER,  # type: ignore[arg-type]
    )

    assert not hasattr(prepared, "value")
    assert known_pii["nid"] not in repr(prepared)


def test_audit_record_json_never_contains_a_value(
    router: PiiRouter, policy: object, known_pii: dict[str, str]
) -> None:
    for text in (f"id {known_pii['nid']}", f"موبايل {known_pii['mobile']}"):
        outcome = router.analyze(text)
        for span in outcome.spans:
            prepared = PreparedSpan.from_detected(
                span,
                text_index=0,
                policy=policy,
                pepper=PEPPER,  # type: ignore[arg-type]
            )
            record = AuditRecord.from_prepared(prepared, RequestContext(request_id="r"))
            serialized = record.to_json()

            assert known_pii["nid"] not in serialized
            assert known_pii["mobile"] not in serialized
            assert not hasattr(record, "value")

            # The repr is what ends up in a log line, and it withholds even
            # the policy-approved preview: the audit table has access controls,
            # a shipped log stream does not.
            rendered = repr(record)
            assert "preview=<redacted>" in rendered
            assert "value_fp=<redacted>" in rendered
            if prepared.preview:
                assert prepared.preview not in rendered


@pytest.mark.parametrize(
    "key",
    ["text", "texts", "value", "content", "prompt", "preview", "context", "pepper", "api_key"],
)
def test_redacting_processor_drops_prompt_carrying_keys(key: str) -> None:
    event = redact_text_keys(None, "info", {"event": "test", key: "28503122148219"})
    assert event[key] == REDACTED


def test_redacting_processor_keeps_safe_keys() -> None:
    event = redact_text_keys(
        None,
        "info",
        {"event": "analyze", "entity_counts": {"EG_NATIONAL_ID": 1}, "user_id": "eng-42"},
    )
    assert event["entity_counts"] == {"EG_NATIONAL_ID": 1}
    assert event["user_id"] == "eng-42"


def test_redacting_processor_truncates_exception_messages() -> None:
    event = redact_text_keys(None, "error", {"event": "failed", "message": "x" * 5000})
    assert len(event["message"]) <= 500


def test_service_log_line_carries_counts_not_text(
    tmp_path: Path, known_pii: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    """The /analyze log line must be counts and status only."""
    configure_logging("INFO")
    logger = structlog.get_logger("probe")

    logger.info(
        "analyze",
        request_id="r",
        entity_counts={"EG_NATIONAL_ID": 1},
        blocked=False,
        latency_ms=4,
        # A careless caller adding the text must not defeat the guarantee.
        text=f"id {known_pii['nid']}",
    )

    out = capsys.readouterr().out
    assert known_pii["nid"] not in out
    assert "EG_NATIONAL_ID" in out

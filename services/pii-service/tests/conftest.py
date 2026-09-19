"""Shared fixtures.

The analyzer is session-scoped: building it constructs spaCy pipelines and
compiles every recognizer's patterns, which is far too slow to repeat per test.
It is treated as read-only by everything that uses it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Final

import pytest

from pii_service.detect.registry import build_analyzer
from pii_service.detect.router import PiiRouter
from pii_service.policy.loader import PolicyBundle, load_policy_bundle

CONFIG_DIR: Final = Path(__file__).resolve().parents[1] / "config"

# Pinned so that "birth date is not in the future" and the 120-year rule stay
# deterministic. A test suite that changes behaviour on 1 January is a bad one.
REFERENCE_DATE: Final = date(2026, 9, 19)


@pytest.fixture(scope="session")
def policy() -> PolicyBundle:
    return load_policy_bundle(CONFIG_DIR)


@pytest.fixture(scope="session")
def router(policy: PolicyBundle) -> PiiRouter:
    analyzer = build_analyzer(policy, today=REFERENCE_DATE)
    return PiiRouter(analyzer, policy)

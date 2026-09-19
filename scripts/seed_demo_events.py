#!/usr/bin/env python3
"""Seed pii_events with synthetic audit rows.

For developing the phase-2 admin UI against realistic data without needing
traffic, and for demonstrating the investigative pivot the fingerprint exists
to support.

The shape of the generated data is the point. It contains:

* a broad base of ordinary activity -- many users, one or two findings each;
* one user who pasted a customer list, so a single request carries dozens of
  distinct fingerprints;
* one value that forty different users each sent once.

Those last two are the cases the timeline has to tell apart, and they look
identical in a plain event count. Only the fingerprint distinguishes them,
which is the argument for the pepper in one screenshot.

    python scripts/seed_demo_events.py --events 5000

Every value is synthetic. The script refuses to run against a database that
already has rows unless --append is passed, so it cannot quietly contaminate
a real audit trail.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "services" / "pii-service" / "src"))

from pii_service.audit.fingerprint import fingerprint  # noqa: E402
from pii_service.db.models import PiiEvent  # noqa: E402
from pii_service.policy.loader import load_policy_bundle  # noqa: E402
from pii_service.settings import Settings  # noqa: E402
from pii_service.synthetic import (  # noqa: E402
    synthetic_iban,
    synthetic_mobile,
    synthetic_national_id,
    synthetic_tax_id,
)
from sqlalchemy import func, insert, select  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

CONFIG_DIR = REPO_ROOT / "services" / "pii-service" / "config"

USERS = [f"eng-{n:02d}" for n in range(1, 41)]
TEAMS = ["platform", "data", "support", "research"]
MODELS = ["qwen3-30b-a3b", "qwen3-30b-a3b-fp8", "llama-gguf-7b"]
ROLES = ["user", "assistant", "system"]
FIELDS = ["content", "tool_call.args", "system"]


def _value_for(entity_type: str, rng: random.Random) -> str:
    match entity_type:
        case "EG_NATIONAL_ID":
            return synthetic_national_id(rng=rng)
        case "EG_MOBILE":
            return synthetic_mobile(rng=rng)
        case "EG_IBAN":
            return synthetic_iban(rng=rng)
        case "EG_TAX_ID":
            return synthetic_tax_id(rng=rng)
        case "AR_PERSON":
            first = rng.choice(["محمد", "أحمد", "فاطمة", "مريم", "يوسف", "سارة"])
            last = rng.choice(["علي", "حسن", "إبراهيم", "عبد الله", "منصور"])
            return f"{first} {last}"
        case "EG_ADDRESS":
            return f"{rng.randint(1, 90)} شارع {rng.choice(['الهرم', 'التحرير'])}"
        case _:
            return f"user{rng.randint(1000, 9999)}@example.com"


ENTITY_WEIGHTS = {
    "EG_NATIONAL_ID": 28,
    "EG_MOBILE": 24,
    "AR_PERSON": 18,
    "EMAIL_ADDRESS": 12,
    "EG_ADDRESS": 8,
    "EG_IBAN": 5,
    "EG_TAX_ID": 3,
    "CREDIT_CARD": 2,
}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=5000)
    parser.add_argument("--days", type=int, default=14, help="spread over this many days")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--append", action="store_true", help="allow seeding a non-empty table")
    parser.add_argument("--database-url", help="overrides PII_DATABASE_URL")
    arguments = parser.parse_args()

    settings = Settings()
    policy = load_policy_bundle(CONFIG_DIR)
    pepper = settings.pepper_bytes
    rng = random.Random(arguments.seed)

    engine = create_async_engine(arguments.database_url or settings.database_url)

    async with engine.connect() as connection:
        existing = int(
            (await connection.execute(select(func.count()).select_from(PiiEvent))).scalar_one()
        )
    if existing and not arguments.append:
        print(
            f"refusing to seed: pii_events already has {existing} rows. "
            "Pass --append if you are sure.",
            file=sys.stderr,
        )
        return 1

    rows = _build_rows(arguments, policy, pepper, rng)

    async with engine.begin() as connection:
        for start in range(0, len(rows), 500):
            await connection.execute(insert(PiiEvent), rows[start : start + 500])
    await engine.dispose()

    print(f"inserted {len(rows)} synthetic events over {arguments.days} days")
    print("includes: one bulk-paste request, and one value shared by 40 users")
    return 0


def _build_rows(
    arguments: argparse.Namespace, policy: object, pepper: bytes, rng: random.Random
) -> list[dict[str, object]]:
    now = datetime.now(UTC)
    entity_types = list(ENTITY_WEIGHTS)
    weights = list(ENTITY_WEIGHTS.values())
    rows: list[dict[str, object]] = []

    def row(
        *, entity_type: str, value: str, user: str, when: datetime, request_id: str
    ) -> dict[str, object]:
        action = policy.entities[entity_type].action if entity_type in policy.entities else "MASK"  # type: ignore[attr-defined]
        start = rng.randint(0, 400)
        return {
            "ts": when,
            "request_id": request_id,
            "user_id": user,
            "team_id": rng.choice(TEAMS),
            "key_alias": f"{user}-cli",
            "key_hash": f"sk-{abs(hash(user)) % 10**12:012d}",
            "end_user_id": None,
            "entity_type": entity_type,
            "recognizer": f"{entity_type.title().replace('_', '')}Recognizer",
            "score": round(rng.uniform(0.62, 1.0), 2),
            "action": str(action),
            "span_start": start,
            "span_end": start + len(value),
            "value_len": len(value),
            "message_index": rng.randint(0, 12),
            "message_role": rng.choice(ROLES),
            "field": rng.choice(FIELDS),
            "value_fp": fingerprint(entity_type, value, pepper),
            "preview": policy.render_preview(entity_type, value),  # type: ignore[attr-defined]
            "model": rng.choice(MODELS),
            "lang": rng.choice(["ar", "en"]),
            "latency_ms": rng.randint(2, 45),
        }

    # Background activity, weighted towards working hours.
    for index in range(arguments.events):
        when = now - timedelta(
            days=rng.uniform(0, arguments.days),
            hours=rng.triangular(0, 23, 14),
        )
        entity_type = rng.choices(entity_types, weights=weights, k=1)[0]
        rows.append(
            row(
                entity_type=entity_type,
                value=_value_for(entity_type, rng),
                user=rng.choice(USERS),
                when=when,
                request_id=f"call-{index:06d}",
            )
        )

    # One person pasting a customer list: many distinct values, one request.
    bulk_when = now - timedelta(days=2, hours=3)
    for index in range(60):
        rows.append(
            row(
                entity_type="EG_NATIONAL_ID",
                value=synthetic_national_id(rng=rng),
                user="eng-17",
                when=bulk_when + timedelta(seconds=index // 20),
                request_id=f"call-bulk-{index // 20:02d}",
            )
        )

    # One value, forty people, one occurrence each -- the same event count as
    # the case above, and a completely different story.
    shared = synthetic_national_id(rng=rng)
    for index, user in enumerate(USERS):
        rows.append(
            row(
                entity_type="EG_NATIONAL_ID",
                value=shared,
                user=user,
                when=now - timedelta(days=rng.uniform(0, arguments.days)),
                request_id=f"call-shared-{index:03d}",
            )
        )

    rows.sort(key=lambda r: r["ts"])  # type: ignore[arg-type,return-value]
    return rows


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

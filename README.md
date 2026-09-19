# Arabic PII Guardrail & Audit Platform

PII detection, masking and audit for an existing on-premises LiteLLM
deployment. Arabic-first, CPU-only, and built to run in air-gapped
environments.

The full design is in [`PII_GUARDRAIL_BRIEF.md`](PII_GUARDRAIL_BRIEF.md). This
README covers running what exists today.

**Status: phase 1.** `pii-service`, the LiteLLM guardrail plugin, the audit
schema and compose wiring. No UI — phases 2–5 are sketched in the brief and
deliberately not built.

---

## What it does

```
                  ┌──────────────┐
  web chat  ─────▶│   pii-api    │──▶ pii-service ──▶ audit DB
   (phase 3)      │  (backend)   │         ▲
                  └──────┬───────┘         │
                         │                 │ HTTP
                         ▼                 │
                  LiteLLM proxy ──▶ guardrail plugin
                         │
                         ▼
                    vLLM / Qwen3
```

Every request through the LiteLLM proxy is scanned before it reaches the model.
Detected PII is replaced with a placeholder, and one row per detected span is
written to a separate audit database — **without the matched value**.

## The two rules everything else follows from

**1. The audit database never contains PII.** There is no `value` column in
`pii_events` and there will never be one. Storing matched values would build a
searchable PII database next to the LLM gateway, which is a worse asset to lose
than the original leak. What replaces the value is a pepper-keyed HMAC
fingerprint, plus an optional partial preview governed by
[`preview_policy.yaml`](services/pii-service/config/preview_policy.yaml).

**2. Findings are reported against original offsets.** Detection runs on
normalized text — `\d{14}` cannot match `٢٨٥٠٣١٢٢١٤٨٢١٩` — and every offset is
mapped back through an offset map before anything is reported or spliced. A
normalizer that returns a bare string is a bug; see
[`normalize.py`](services/pii-service/src/pii_service/detect/normalize.py).

---

## Quick start

### 1. Generating the audit pepper

**Do this first. The service refuses to start without it.**

```bash
openssl rand -hex 32
```

Put the result in `.env` as `PII_AUDIT_PEPPER`.

The pepper keys the HMAC that turns a matched value into a correlation
fingerprint. It is not optional and a placeholder will not do, because the
Egyptian national ID space is small enough to enumerate exhaustively — a
century digit, a birth date, a governorate from a list of 27, and a check digit
leave roughly 10⁴ possibilities per (birth date, governorate) pair. A dump of
`SHA-256(nid)` can therefore be inverted on a laptop, which makes plain hashing
equivalent to storing the IDs. HMAC with a key held *outside* the database
breaks that: an attacker with the table and without the pepper has nothing to
enumerate against.

The service validates this at startup and **raises** if the pepper is unset,
empty, shorter than 32 characters, or still the value from `.env.example`. That
is a hard failure on purpose: a service that starts with a weak pepper writes
fingerprints that are worthless but indistinguishable from good ones, and by
the time anyone notices, the values needed to recompute them are long gone.

Keep it out of the audit database — a secrets manager, a Docker secret, or an
environment variable injected at deploy time. Storing it beside the data it
protects defeats the exercise.

> **Rotating the pepper is a one-way door.** Existing fingerprints stop matching
> new ones and cannot be recomputed, because the values were never stored.
> Rotate only on suspected compromise, and expect correlation history to reset.

### 2. Configure

```bash
cp .env.example .env
$EDITOR .env          # set PII_AUDIT_PEPPER, PII_DB_PASSWORD, LITELLM_NETWORK
```

`LITELLM_NETWORK` must name the docker network your existing LiteLLM stack runs
on — usually `<compose-project>_default`. Find it with `docker network ls`.

### 3. Bring it up

```bash
docker compose up -d
```

This starts `pii-db` (Postgres 16, **separate** from LiteLLM's), runs the
Alembic migration, and starts `pii-service` on the `litellm` network so the
guardrail can reach it at `http://pii-service:8090`.

```bash
curl -s localhost:8090/livez        # if you uncommented the port mapping
docker compose logs -f pii-service
```

### 4. Mount the guardrail

See [`services/litellm-pii-guardrail/README.md`](services/litellm-pii-guardrail/README.md)
for the volume mount and the `config.yaml` block. Then restart LiteLLM and send
a request containing a national ID; it should come back masked, with a row in
`pii_events`.

---

## Detection

Three tiers, descending reliability. Presidio is the orchestration layer, not
the detector.

| Tier | What | Status |
|---|---|---|
| 1 | Deterministic recognizers: Egyptian national ID, mobile, IBAN, tax ID, passport, plus email and card | **On**, always |
| 2 | Arabic NER (ONNX int8) + Egyptian address gazetteer | Written, **off** — needs a model artifact |
| 3 | GLiNER2 for Latin script | Written, **off** — needs a latency decision |

### Tier 1 scoring

A national ID scores **1.0** with a verified check digit and **0.85** without.
The checksum is a score booster, not a gate: published implementations of the
Egyptian NID checksum genuinely disagree, and a wrong gate means missed PII.
The same logic applies to governorate codes — the issued-code table names the
governorate and raises confidence, but the accepted range stays the wider
01–35/88, so an unissued in-range code is still reported.

Tax ID and passport are shape-only. Their base scores sit deliberately *below*
their thresholds, so they report only when an Arabic or English context term
(`الرقم الضريبي`, `جواز سفر`, `tax id`, …) appears nearby. Nine digits is far
too common a shape to report unconditionally.

### Tier 2 model artifact

Tier 2 needs a directory containing `model.onnx`, `tokenizer.json` and
`labels.json`, pointed at by `PII_TIER2_MODEL_DIR`. Enabling tier 2 without one
raises at startup rather than silently running with reduced coverage.

> **Open decision.** *Which CAMeLBERT variant to quantize has not been decided.*
> The brief names this as one of the things to flag rather than guess, because
> the choice materially changes Arabic name recall. `bert-base-arabic-camelbert-mix-ner`
> is the usual default; a dialect-specific variant may do better on Egyptian
> text. Decide, export to ONNX int8 in a builder stage, and record the choice
> here.

### Tier 3 latency

Off by default, and that is a measured decision rather than a precaution. The
vendor reports 50–200 ms/document on 8–16 core CPU; an independent benchmark
measured ~2.3 s per chat-sized message on an M1. That is more than an order of
magnitude, and the guardrail runs `pre_call`, so whatever it costs is added to
every request through the gateway.

```bash
python scripts/benchmark.py --iterations 500
```

Run that on hardware resembling production before anyone designs around a
number.

---

## Configuration is policy, not code

Three YAML files under `services/pii-service/config/`, mounted read-only into
the container and validated at startup. Compliance changes an action or a
threshold and restarts the service; nobody ships a release for it.

| File | Owns |
|---|---|
| `entities.yaml` | What is detected, at what confidence, and whether it is `MASK`, `BLOCK` or `ALLOW` |
| `preview_policy.yaml` | How much of a value may reach the audit DB. Default: none |
| `gazetteer_eg.yaml` | Governorates, address markers, localities, Arabic context terms |

Validation uses `extra="forbid"`, so a misspelled key is a startup error rather
than a setting that silently reads as a default. A preview rule naming an
entity `entities.yaml` does not define is also an error — that combination
almost always means a rename landed in one file and not the other.

---

## Operations

### Health

| Endpoint | Purpose |
|---|---|
| `/livez` | Liveness. Never touches the database |
| `/health` | Audit queue depth, WAL state, cache hit rate, tier status |

`/health` reports `degraded`, not unhealthy, when the database is unreachable.
The service is still masking correctly and spilling to the WAL, and an
orchestrator that restarted it for that would turn a logging outage into a
gateway outage.

### When the audit database goes down

The write path fails open. Records spill to a size-bounded write-ahead log
(`/var/lib/pii-service/wal`, a named volume) and replay automatically on the
next startup; a segment is deleted only after its insert commits. Watch
`spilled` and `wal_dropped_segments` in `/health` — the second means the WAL hit
its size bound and the oldest audit records were discarded, which is a loud
error in the logs and should page someone.

### Migrations

```bash
docker compose run --rm pii-migrate                              # upgrade head
docker compose run --rm pii-migrate alembic downgrade -1         # roll back
```

### Demo data

```bash
python scripts/seed_demo_events.py --events 5000
```

Generates synthetic audit rows shaped to exercise the investigative pivot: one
user who pasted a customer list (66 distinct fingerprints, one request) and one
value forty different users each sent once. Those look identical in an event
count and completely different through the fingerprint — which is the argument
for the pepper, in one screenshot.

---

## Development

```bash
cd services/pii-service
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q
```

```bash
cd services/litellm-pii-guardrail
uv venv .venv
uv pip install --python .venv/bin/python litellm==1.101.0 pytest pytest-asyncio pytest-timeout respx httpx
.venv/bin/python -m pytest -q
```

### The tests that matter most

Two suites are load-bearing, per the brief:

- **`tests/test_normalize.py`** — the offset map. Arabic-Indic digits mixed with
  ASCII, diacritics inside a matched name, RTL/LTR mixed sentences, spans that
  start inside a stripped character run, ligature expansion, and map invariants
  over randomized mixed-script input.
- **`tests/test_guardrail_contract.py`** — the LiteLLM contract, including two
  tests that drive litellm's real translation handler and assert on the
  messages the model would receive.

**`tests/test_no_pii_in_logs.py`** greps the whole suite's log output for known
fixture values. All PII fixtures are generated by `pii_service.synthetic` and
never pasted — a real national ID in a test file is one in git history and in
every CI log too.

### Why no PyTorch

The runtime image has neither PyTorch nor transformers. Tier 2 runs ONNX
through `onnxruntime`; model conversion belongs in a builder stage. In an
air-gapped delivery this is the difference between a `docker load` tarball
someone can carry in and one they cannot.

spaCy is present but only ever builds `spacy.blank()` pipelines — a tokenizer
and nothing else. Presidio's stock engine calls `spacy.cli.download()` for any
missing model, which with no internet is a crash on startup rather than a slow
path.

---

## Repository layout

```
.
├── docker-compose.yml          # pii-service + pii-db, joins the external litellm network
├── .env.example
├── PII_GUARDRAIL_BRIEF.md      # the authoritative design
├── services/
│   ├── pii-service/            # detection and audit authority (FastAPI, CPU only)
│   │   ├── src/pii_service/
│   │   │   ├── detect/         # normalize, tiers 1-3, router, cache, masking
│   │   │   ├── policy/         # YAML models and loader
│   │   │   ├── audit/          # fingerprint, records, sink, WAL
│   │   │   ├── db/             # models + alembic
│   │   │   └── api/            # routes and schemas
│   │   ├── config/             # entities, preview policy, gazetteer
│   │   └── tests/
│   └── litellm-pii-guardrail/  # the ~130-line CustomGuardrail adapter
└── scripts/
    ├── benchmark.py            # p50/p95 per tier on CPU
    └── seed_demo_events.py
```

## Not built yet

Phases 2–5 from the brief: the admin timeline and drill-down, the chat UI, the
air-gapped packaging tarball, and the false-positive-driven eval loop. The
interfaces here are shaped so they can be added without rework — findings carry
enough context for RTL-safe span rendering, the break-glass reveal role is
defined in the data model with the endpoint disabled, and the service API is
generic enough that the chat backend and the proxy guardrail share it.

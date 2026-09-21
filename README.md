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
docker compose up -d --wait
```

This starts `pii-db` (Postgres 16, **separate** from LiteLLM's), runs the
Alembic migration, and starts `pii-service` on the `litellm` network so the
guardrail can reach it at `http://pii-service:8090`.

Building behind an internal PyPI mirror or a TLS-intercepting proxy — the
normal case at the sites this ships to:

```bash
PIP_CA_BUNDLE=/etc/ssl/certs/corporate-ca.pem docker compose build
docker compose build --build-arg PIP_INDEX_URL=https://nexus.internal/repository/pypi/simple
```

Both are optional and change nothing on an ordinary network. The build needs a
PyPI mirror and **no container registry beyond the two base images** and no
Debian archive: there is no `apt` step, and the healthcheck uses the Python
already in the image rather than pulling in `curl`.

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

#### Choosing the CAMeLBERT variant

**Open decision — decide it by running the comparison, not by reading a
benchmark.** CAMeL-Lab publishes two NER checkpoints:

| Variant | Pre-training | Why it might win |
|---|---|---|
| `bert-base-arabic-camelbert-mix-ner` | MSA + dialectal + classical | Prompts from this gateway are Egyptian dialect, not newswire |
| `bert-base-arabic-camelbert-msa-ner` | MSA only | Usually stronger on formal register, which some traffic is |

The choice materially changes Arabic name recall, and a missed name is a leak,
so it is settled by measurement:

```bash
# Build-time only, on a machine with internet. Needs torch; the runtime image
# never does.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers "optimum[onnxruntime]"

python scripts/export_camelbert_onnx.py --all      # both variants, fp32 + int8
python scripts/eval_arabic_ner.py --all --json ner-verdict.json
```

That prints a ranked table and a verdict naming the directory to put in
`PII_TIER2_MODEL_DIR`. Exporting **both precisions** is deliberate: dynamic
int8 usually costs little accuracy, but "usually" is not a measurement, and
here a point of coverage is a leak rather than a rounding error.

**How it ranks.** Primary metric is **coverage** — the share of gold spans
overlapped by *any* prediction — per brief §5, not exact-match F1. A sloppy
boundary that still covers the name is a pass; a missed span is a leak. Type
confusion does not count against coverage either, because a PER predicted as
LOC is still masked.

Coverage alone would rate a model that masks everything as perfect, so there is
a gate: **coverage ≥ 80%** and **≤ 0.10 false positives per entity-free
sentence**. A variant failing either cannot win, whatever its other numbers.
Ties break on false positives, then p95 latency.

The report also breaks coverage down **by register**, with Egyptian first.
That column is the one that matters: a variant that wins on the MSA average
while missing dialect names is the wrong choice for this deployment, and a
single headline number would hide exactly that.

**The gold set** is [`services/pii-service/eval/arabic_ner_gold.yaml`](services/pii-service/eval/arabic_ner_gold.yaml)
— 62 sentences, 91 spans, weighted towards Egyptian dialect and mixed
Arabic/Latin technical text, with 12 entity-free sentences carrying the
false-positive measurement and hard cases where a name collides with an
ordinary word (`نور` the name vs `نور` the noun). Entities are authored as
inline `[surface](TYPE)` markup and offsets are derived from it — nobody
hand-counts a character offset in Arabic, which is the same discipline the
detector itself is built on.

Extend it. It is the input to phase 5's eval loop, and the false-positive flags
from the admin UI are meant to flow back into it.

> **Not yet run here.** This sandbox has no route to HuggingFace (and pulling
> CAMeLBERT from an unofficial mirror into a PII detector is a supply-chain
> risk not worth taking), so the two variants have not been measured. The
> harness itself is tested and CI exercises it end to end against deterministic
> stand-in ONNX models. Record the result below when you run it.
>
> | Variant | Coverage | Egyptian | FP/clean | p95 | Chosen |
> |---|---|---|---|---|---|
> | camelbert-mix-int8 | — | — | — | — | |
> | camelbert-msa-int8 | — | — | — | — | |
> | camelbert-mix-fp32 | — | — | — | — | |
> | camelbert-msa-fp32 | — | — | — | — | |

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

## Custom labels and replacement policy

Two things an administrator can change at runtime, through `/admin`, without a
release: **what tier 3 looks for**, and **what a masked span is replaced with**.

The admin *menu* is phase 2 (brief §10) and is not built. These are the
endpoints it will call, and they work with curl today.

> **Auth is a placeholder.** `/admin` takes a bearer token from
> `PII_ADMIN_TOKEN`, compared in constant time, and is **disabled when that is
> unset**. The brief specifies phase-2 role auth in detail (`ADMIN_EMAIL` /
> `ADMIN_INITIAL_PASSWORD`, `admin` and `auditor`, `must_change_password`);
> building it now would mean guessing at a design already committed to. The
> token is all-or-nothing — there is no read-only auditor yet — so pass
> `X-Admin-User` to record who made each change in `custom_entities.updated_by`.

### Adding a tier-3 label

GLiNER2 is schema-conditioned: labels travel in the forward pass, so a new
entity type is a prompt, not a retrain.

```bash
curl -X PUT localhost:8090/admin/entities/PROJECT_CODENAME \
  -H "Authorization: Bearer $PII_ADMIN_TOKEN" \
  -H "X-Admin-User: ops@example.com" \
  -H 'content-type: application/json' \
  -d '{
        "entity_type": "PROJECT_CODENAME",
        "gliner_prompt": "internal project codename",
        "category": "other", "action": "MASK", "score_threshold": 0.6,
        "note": "Q3 launch names must not reach the model"
      }'
```

`gliner_prompt` is a natural-language phrase, not an identifier — *"employee
badge number"*, not `EMPLOYEE_BADGE`. Wording changes recall, so treat editing
one as a model change.

The change takes effect on the next request: the recognizers are **re-prompted
in place** rather than the analyzer rebuilt, so adding a label costs nothing
and does not reload the model. A new label needs a prompt — an entity nothing
can detect is policy that silently does nothing, and the API refuses it.

### Choosing what a span is replaced with

| Strategy | Output | Realistic? |
|---|---|---|
| `placeholder` | `<PERSON>` | no — the default |
| `constant` | `John Doe` | **yes** |
| `surrogate` | `John Doe` / `Jane Roe` / … stable per value | **yes** |
| `redact` | *(removed)* | no |
| `labelled_fingerprint` | `<PERSON:9f75871d>` | no |

```bash
curl -X PUT localhost:8090/admin/entities/PERSON \
  -H "Authorization: Bearer $PII_ADMIN_TOKEN" -H 'content-type: application/json' \
  -d '{"entity_type": "PERSON",
       "replacement": {"strategy": "constant", "value": "John Doe"}}'
```

The baseline lives in
[`replacement_policy.yaml`](services/pii-service/config/replacement_policy.yaml)
under change control; the admin overlay layers on top and is stored in
`custom_entities`. `DELETE /admin/entities/PERSON` reverts to the baseline.

### Read this before switching anything to `constant` or `surrogate`

**The masked prompt stops looking masked.** A human reading "John Doe emailed
us" cannot tell it was redacted and may treat it as fact. `/admin/policy`
returns a warning listing every entity in this state, and so does the response
to the edit that caused it.

**Prefer `surrogate` over `constant`.** A constant collapses distinct values:
"Ahmed emailed Sara about Omar" becomes "John Doe emailed John Doe about John
Doe" — a false prompt, and a model asked to summarise it will answer about one
person. `surrogate` picks from a pool keyed by the value's HMAC fingerprint, so
two people stay two people and the same person is the same fake name in every
turn, with no mapping stored anywhere. Pool collisions are expected and are a
feature: several people sharing a fake name is weaker linkage, not a bug.

**Masking stays idempotent, at a price.** The architecture masks twice — the
chat backend, then the proxy guardrail — and relies on the second pass finding
nothing (brief §2). `<PERSON>` is not a name so it never did; "John Doe" is.
Every string a rule can emit is registered as its vocabulary, and a detected
value already in that vocabulary is suppressed rather than masked again, so
`mask(mask(x)) == mask(x)`.

The price: **a person genuinely named "John Doe" is never masked as a PERSON.**
That is inherent to replacing PII with text that looks like PII, and it is why
the shipped default is an unambiguous placeholder. Choose a vocabulary unlikely
to collide with the people in your data.

**`labelled_fingerprint` is the middle path** if you want distinct values to
stay distinct without the prompt looking real: `<PERSON:9f75871d>` correlates
against `pii_events.value_fp` for an auditor while staying obviously masked.

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

### Running against a real PostgreSQL

Most of the suite uses SQLite, which is fine for the sink's logic but cannot
tell you whether the *deployment* is right. The migration, the `DESC` indexes,
`BIGSERIAL` and psycopg's type handling are only honest against Postgres:

```bash
PII_TEST_DATABASE_URL=postgresql+psycopg://pii@127.0.0.1:5432/postgres \
    .venv/bin/python -m pytest tests/test_postgres_integration.py -q
```

Those tests create a throwaway database, migrate it with Alembic, and drop it
afterwards. One of them runs `alembic check`, which fails if the SQLAlchemy
model has drifted from the migration — the cheapest way to catch a model edit
that never got a migration. They skip silently when the variable is unset.

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

**`tests/test_deployment.py`** asserts the deployment invariants that fail
quietly: the audit database is not published to a host port, the image never
acquires PyTorch, the healthcheck uses `/livez` rather than `/health`, and the
`.env.example` placeholder still matches the constant `settings.py` rejects —
if those two drift apart, the pepper guard silently stops working.

### CI

`.github/workflows/ci.yml` runs three jobs: lint/types/unit plus the PostgreSQL
integration tests against a `postgres:16-alpine` service container; the
guardrail contract tests; and a `compose` job that actually builds the image,
runs `docker compose up --wait`, asserts the migration reached head, posts a
national ID through `/analyze` and checks both that the response is masked and
that an audit row landed without the value in it.

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

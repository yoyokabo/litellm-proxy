# Arabic PII Guardrail & Audit Platform — Implementation Brief

You are implementing a PII detection, masking and audit system that plugs into an
**existing** LiteLLM proxy deployment. Read this whole brief before writing code.

The architecture below is already decided. Do not re-open these decisions; if you
believe one is wrong, say so and stop rather than silently substituting your own.

---

## 1. Context

An on-premises LLM stack already exists and is serving 100+ engineers:

- vLLM serving Qwen3-30B-A3B (FP8) on a GPU host
- llama.cpp for GGUF models
- **LiteLLM proxy** with model aliasing — this is the integration point
- nginx reverse proxy
- Langfuse for observability
- LiteLLM has its own Postgres. **Do not touch it.**

Deployments include air-gapped government environments. Assume no internet at
runtime and often none at install time beyond a `docker load`.

## 2. What we are building

Four new services plus a database, joining the existing LiteLLM Docker network.

```
                  ┌──────────────┐
  web chat  ─────▶│   pii-api    │──▶ pii-service ──▶ audit DB
                  │  (backend)   │         ▲
                  └──────┬───────┘         │
                         │                 │ HTTP
                         ▼                 │
                  LiteLLM proxy ──▶ guardrail plugin
                         │
                         ▼
                    vLLM / Qwen3
```

**`pii-service`** — FastAPI. The single detection and audit authority. Owns
Presidio, the models, the normalizer, fingerprinting and all writes to the audit
DB. Stateless apart from its DB connection. CPU only.

**`litellm-pii-guardrail`** — A thin `CustomGuardrail` subclass mounted into the
existing LiteLLM container. It calls `pii-service` over HTTP and maps the result
back onto the request. **No models, no policy, no DB access in this file.** It
should be under ~100 lines. This is what makes the system drop-in.

**`pii-api`** — Backend for the web app. Auth, sessions, chat proxying with SSE,
admin queries. Maps each app user to a dedicated LiteLLM virtual key.

**`pii-web`** — Chat UI + admin observability UI.

**`pii-db`** — Postgres 16, separate instance from LiteLLM's.

### Why the chat backend calls `pii-service` directly

The chat needs rich synchronous feedback about what was filtered. Extracting that
from LiteLLM response headers is fragile and breaks under streaming. So the chat
backend analyses first, renders feedback, then forwards already-masked text to
LiteLLM. The guardrail still runs on the proxy as enforcement for every other
client (SDKs, Claude Code, scripts). Detection is idempotent — re-running on
masked text finds nothing and costs one cheap pass. One audit writer either way.

---

## 3. Phase 1 — the only thing in scope right now

**Deliver `pii-service` + the guardrail plugin + the DB schema + compose wiring.
No UI.**

Done criteria:

1. `docker compose up` brings up `pii-service` and `pii-db` and joins the
   existing external `litellm` network.
2. A `POST /analyze` call returns findings and anonymized text for mixed
   Arabic/English input.
3. A request through the *existing* LiteLLM proxy carrying an Egyptian national
   ID comes back masked, and a row lands in `pii_events` with the correct
   `user_api_key_user_id`.
4. `pytest` passes, including the Arabic normalization offset tests.
5. `scripts/benchmark.py` reports p50/p95 detection latency per recognizer tier
   on CPU.

Phases 2–5 (admin UI, chat, packaging, eval loop) are sketched in §10 so your
interfaces don't paint them into a corner. **Do not build them yet.**

---

## 4. Detection design

Three tiers, descending reliability. Presidio is the framework and orchestration
layer; it is **not** the detector.

### Tier 1 — deterministic recognizers (Presidio `PatternRecognizer` + validator)

Highest priority, near-zero false positives. Egyptian national ID belongs here,
never in a model.

Egyptian NID is exactly 14 digits:

| Position | Meaning |
|----------|---------|
| 1 | Century: `2` = 1900–1999, `3` = 2000–2099 |
| 2–7 | Birth date `YYMMDD` |
| 8–9 | Governorate code, `01`–`35` or `88` (abroad) |
| 10–13 | Serial; parity of digit 13 encodes gender |
| 14 | Check digit |

Validate structure *and* plausibility of the embedded date. Treat the check digit
as a score booster, not a hard gate — published checksum implementations disagree
and a wrong gate means missed PII. Score 1.0 with valid checksum, 0.85 without.

Also tier 1: Egyptian mobile numbers (`01[0125]` + 8 digits, with and without
`+20`), IBAN (`EG` + 27), tax ID, passport numbers.

Use Presidio `context` word lists to boost scores — include Arabic terms
(`الرقم القومي`, `رقم قومي`, `بطاقة`, `موبايل`, `عنوان`).

### Tier 2 — Arabic NER for names / orgs / locations

ONNX int8 quantized CAMeLBERT-family NER, `onnxruntime` only — **no PyTorch or
Transformers at inference time**. Wrap as a Presidio `EntityRecognizer` subclass.
Map `PER`→`AR_PERSON`, `LOC`→`AR_LOCATION`, `ORG`→`AR_ORG`.

Arabic addresses: do **not** rely on the model. Build a gazetteer + pattern
recognizer over the 27 governorates plus structural markers (`شارع`, `ش`,
`عمارة`, `الدور`, `شقة`, `مدينة`). Extensible via a YAML file, no retraining.

### Tier 3 — GLiNER2-PII for Latin-script text

Schema-conditioned: labels are passed at inference time in a single forward pass,
so adding an entity type is a config line, not a retrain. Supports EN/FR/ES/DE/
IT/PT/NL — **not Arabic**.

**Vendor latency claims are disputed.** The paper reports 50–200 ms/document on
8–16 core CPU; an independent benchmark measured ~2.3 s per chat-sized message on
an M1. Measure it yourself in `scripts/benchmark.py` before anyone designs around
a number. If it lands near the high end, put it behind a feature flag and default
it off.

### Routing

Cheap language/script detection (Unicode block ratio is sufficient) routes Arabic
spans to tier 2 and Latin to tier 3. Tier 1 always runs on everything.

---

## 5. The Arabic normalization trap — get this right first

Before any numeric regex runs, normalize:

- Arabic-Indic digits `٠١٢٣٤٥٦٧٨٩` → ASCII
- Extended Arabic-Indic (Persian) `۰۱۲۳۴۵۶۷۸۹` → ASCII
- Strip tatweel `ـ` and diacritics before name matching
- Normalize alef variants (`أإآ` → `ا`) and `ة`/`ه`, `ى`/`ي` for gazetteer lookup

**Every normalization must maintain an offset map back to the original string.**
Findings are reported against original offsets, always. Anonymization splices the
original text. A normalizer that returns a bare string is a bug — the signature is
`normalize(text) -> (normalized: str, offset_map: list[int])`.

Write the offset tests before the normalizer. Cover: Arabic-Indic digits mixed
with ASCII, diacritics inside a matched name, RTL/LTR mixed sentences, and a span
that starts inside a stripped character run.

---

## 6. Audit model — read this twice

**Never store PII values in the audit database.** Storing them builds a
searchable PII database next to the LLM gateway, which is a worse asset to lose
than the original leak. For government clients that is an incident, not a finding.

LiteLLM learned this the hard way: a custom guardrail returning the full request
payload got plaintext credentials written into spend logs and OTel traces. Their
fix classifies `guardrail_request`, `guardrail_response`, `match_details` and
`classification` as prompt-carrying and redacts them, keeping only name,
provider, mode, status, timings and masked-entity counts. Adopt that partition.

### Schema

```sql
CREATE TABLE pii_events (
  id              BIGSERIAL PRIMARY KEY,
  ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
  request_id      TEXT NOT NULL,          -- litellm_call_id
  -- identity
  user_id         TEXT,                   -- user_api_key_user_id
  team_id         TEXT,                   -- user_api_key_team_id
  key_alias       TEXT,
  key_hash        TEXT,
  end_user_id     TEXT,                   -- request body "user" field
  -- what
  entity_type     TEXT NOT NULL,
  recognizer      TEXT NOT NULL,
  score           REAL,
  action          TEXT NOT NULL,          -- MASK | BLOCK | ALLOW
  -- where, without the value
  span_start      INT,
  span_end        INT,
  value_len       INT,
  message_index   INT,
  message_role    TEXT,
  field           TEXT,                   -- content | tool_call.args | system
  -- correlation, without the value
  value_fp        TEXT,                   -- HMAC-SHA256(normalized, pepper)[:16]
  preview         TEXT,                   -- '2850******5974' or NULL
  -- context
  model           TEXT,
  lang            TEXT,
  latency_ms      INT
);

CREATE INDEX ON pii_events (user_id, ts DESC);
CREATE INDEX ON pii_events (value_fp);
CREATE INDEX ON pii_events (entity_type, ts DESC);
```

Use Alembic for migrations.

### `value_fp`

HMAC-SHA256 with a **server-side pepper held outside the database**, truncated to
16 hex chars. Plain hashing is useless here — the Egyptian NID space is trivially
enumerable, so a table dump under SHA-256 alone is equivalent to storing the IDs.

The pepper is what makes the admin view investigative rather than a counter: it
answers "is one person pasting a customer list, or did 40 people each paste one
record" without storing a digit.

**The service must refuse to start if the pepper is unset, empty, or matches the
value in `.env.example`.** Document generation in the README.

### `preview`

Per-entity-type masking policy in **config, not code**, so compliance can change
it without a deploy. Structured IDs → first-4/last-4. Names, addresses → `NULL`.

### Break-glass

Define the role and the audited-reveal path in the data model, leave it
unassigned and the endpoint disabled in phase 1.

---

## 7. LiteLLM integration

Implement `apply_guardrail`, not the individual hooks. Signature:

```python
async def apply_guardrail(
    self,
    inputs: GenericGuardrailAPIInputs,
    request_data: dict,
    input_type: Literal["request", "response"],
    logging_obj: Optional["LiteLLMLoggingObj"] = None,
) -> GenericGuardrailAPIInputs:
```

Mode is `pre_call`. `during_call` runs in parallel with the LLM call and cannot
reliably mask — blocking only. We mask, so `pre_call`, and we pay full detection
latency per request.

### Rules

- **Mask by editing `inputs["texts"]` and `inputs["tool_calls"]` in place.**
  LiteLLM maps them back onto the request.
- **`structured_messages` is the exception — replace the list with a new object.**
  In-place edits are ignored.
- **Scan `inputs["tool_calls"]` explicitly.** There is a known gap where Presidio's
  own hook only reads content strings and `{"text": ...}` items, so tool-call
  arguments in conversation history were never scanned on either wire format.
  Do not assume the installed version is patched.
- **Block by raising an exception.** Message must be machine-readable: entity
  types and counts. **Never the matched values** — that message reaches the
  client and the logs.
- **Never return `request_data` or any large dict from the guardrail.** That is
  the exact shape that caused the credential-leak incident.
- Into LiteLLM's own logging path, pass **counts and status only**. Rich detail
  goes to our sink. Note `add_standard_logging_guardrail_information_to_request_data`
  is premium-gated and silently no-ops on a community licence — do not depend on it.

### Caching

The same PII reappears in every turn of a conversation. Cache findings keyed by
content hash. This is where most of the latency saving lives in agentic
workloads.

### Config block (goes in the existing LiteLLM config)

```yaml
guardrails:
  - guardrail_name: "pii-ar"
    litellm_params:
      guardrail: pii_guardrail.ArabicPIIGuardrail
      mode: "pre_call"
      service_url: http://pii-service:8090
      default_on: true
```

---

## 8. Audit write path

- **The write must not sit in the request path.** Queue and flush from a
  background task.
- **Fail open.** If the DB is unreachable, spill to a local WAL file and keep
  serving. A guardrail that fails requests because its audit log is down is a
  self-inflicted outage. Emit a loud metric.
- Batch inserts. One request produces many spans.

---

## 9. Design tokens (for phases 2–3; define now, use later)

Dark-first, light theme available. 4px base grid, 8px rhythm. Dense but not
cramped.

Typography: **Inter** (Latin UI), **IBM Plex Sans Arabic** (Arabic content — the
metrics match well enough that mixed lines don't jump), **JetBrains Mono** (IDs,
offsets, fingerprints, placeholders).

```css
/* neutrals — dark */
--bg-0: #0B0F14;   --bg-1: #121821;   --bg-2: #1A2230;
--border: #263141; --text-1: #E6EDF5; --text-2: #94A3B8; --text-3: #64748B;

/* accent — teal, not blue: blue stays reserved for "information" */
--accent: #14B8A6; --accent-hover: #2DD4BF; --accent-dim: #0F766E;

/* status */
--ok: #34D399; --masked: #FBBF24; --blocked: #F87171; --info: #60A5FA;

/* entity categories — deliberately separate from status colors, so a
   stacked timeline never reads as "lots of red = bad" */
--ent-id:       #A78BFA;  /* national ID, passport, tax ID */
--ent-person:   #38BDF8;
--ent-contact:  #2DD4BF;  /* phone, email */
--ent-location: #4ADE80;
--ent-finance:  #F0ABFC;  /* IBAN, cards */
--ent-other:    #94A3B8;
```

Light theme inverts neutrals, accent drops to `#0F766E`, entity palette darkens
one step. Never encode action by colour alone — masked vs blocked also differ by
shape (fill vs outline).

Stack: Tailwind + shadcn/ui, Recharts for the timeline (swap to visx if it gets
sluggish past a few hundred thousand events).

---

## 10. Later phases — for interface design only, do not build

**Phase 2 — admin view.** Timeline and drill-down only, no leaderboard.
Available to a default admin bootstrapped from `ADMIN_EMAIL` /
`ADMIN_INITIAL_PASSWORD`, flagged `must_change_password`, with admin routes
refusing to serve until rotated. Roles: `admin`, `auditor` (read-only).

Timeline: stacked area of events over time by entity category, brush-to-zoom
filtering a virtualized event table below, plus a thin separate lane for blocked
events so they don't vanish into the stack.

Drill-down: right-side panel (not a route change, so the timeline stays as
context). Per-span type, recognizer, score, offsets, preview, message index and
field. **Clicking a fingerprint pivots the timeline to every other occurrence of
that value — one click, this is the core investigative move.** Plus a
false-positive flag per span.

*RTL constraint that affects your span API now:* character offsets over Arabic
render in visually confusing order — a span at 10–24 does not appear "after" one
at 0–9. Span context must render in an isolated element (`dir="auto"`,
`unicode-bidi: isolate`) with numeric offsets shown separately. Return enough
context in the API for that.

**Phase 3 — chat.** Showing detected PII back to the person who typed it is not a
disclosure; they wrote it. So: masked path sends, renders their own bubble with
spans underlined in category colours, plus an expandable `3 items redacted` chip
showing type, matched text and placeholder. Blocked path does not send, stays in
the composer, spans highlighted in place, with a "remove flagged text" button.
Pre-send debounced detection is a per-user toggle, **off by default**.

**Phase 4 — packaging.** Bake model weights into the image (multi-stage: download
and ONNX-convert in the builder, copy artifacts into a slim runtime with no
PyTorch). Air-gapped installs get one `docker load` tarball, not a multi-gigabyte
download.

**Phase 5 — eval loop.** False-positive flags become a labelled Arabic PII eval
set. Optimise for **coverage** (share of gold spans overlapped by any prediction),
not exact-match F1 — a sloppy boundary that still covers the ID is a pass, a
missed span is a leak.

---

## 11. Repo layout

```
.
├── docker-compose.yml
├── .env.example
├── README.md
├── services/
│   ├── pii-service/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── src/pii_service/
│   │   │   ├── main.py            # FastAPI app
│   │   │   ├── api/               # routes, schemas
│   │   │   ├── detect/
│   │   │   │   ├── normalize.py   # + offset map
│   │   │   │   ├── registry.py    # Presidio analyzer assembly
│   │   │   │   ├── tier1_patterns.py
│   │   │   │   ├── tier2_arabic_ner.py
│   │   │   │   ├── tier3_gliner.py
│   │   │   │   └── router.py
│   │   │   ├── policy/            # entity actions, preview rules (YAML)
│   │   │   ├── audit/             # fingerprint, sink, WAL fallback
│   │   │   └── db/                # models, alembic
│   │   ├── config/
│   │   │   ├── entities.yaml
│   │   │   ├── gazetteer_eg.yaml
│   │   │   └── preview_policy.yaml
│   │   └── tests/
│   └── litellm-pii-guardrail/
│       ├── pii_guardrail.py
│       ├── tests/
│       └── README.md              # mount + config instructions
└── scripts/
    ├── benchmark.py
    └── seed_demo_events.py
```

---

## 12. Conventions

- Python 3.11+, `uv`, full type hints, `ruff` + `mypy` clean.
- Pydantic v2 for all API schemas.
- Structured JSON logging. **Logging statements must never include matched values.**
  Add a test that greps the log output of the full test suite for known fixture
  PII values and fails if any appear.
- All PII test fixtures are synthetic. Generate valid-structure Egyptian NIDs in
  a helper; never paste real ones.
- Config via env vars with a Pydantic `Settings` class.
- `pytest`, with the normalizer offset tests and the guardrail contract tests as
  the non-negotiable core.

---

## 13. How to start

1. Read this brief back to me as a short plan — the order you'll build phase 1 in
   and anything you think is underspecified. **Wait for confirmation before
   writing code.**
2. Then: repo skeleton, `normalize.py` with its tests first, then tier 1, then
   the schema and audit sink, then the guardrail plugin, then tiers 2 and 3,
   then compose and the benchmark script.

Flag rather than guess if you hit: the exact installed LiteLLM version's
`GenericGuardrailAPIInputs` shape, which CAMeLBERT variant to quantize, or
anything where a wrong assumption would silently reduce detection coverage.

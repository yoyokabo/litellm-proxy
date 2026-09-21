# Arabic PII Guardrail & Audit Platform

PII detection, masking and audit for an existing on-premises LiteLLM
deployment. Arabic-first, CPU-only, and built to run in air-gapped
environments.

The full design is in [`PII_GUARDRAIL_BRIEF.md`](PII_GUARDRAIL_BRIEF.md). This
README covers running what exists today.

**Status: phase 1 verified end to end, plus the phase 2/3 web app.**
`pii-service`, the LiteLLM guardrail plugin, the audit schema, compose wiring,
and now `pii-api` + `pii-web` — the admin timeline and the chat. All five phase-1
done criteria have been run against a live stack; see
[`scripts/verify_end_to_end.py`](scripts/verify_end_to_end.py) and the tier-2
result below.

Phases 4 (air-gapped tarball) and 5 (the eval loop) are not built, though the
admin view already records the false-positive flags phase 5 consumes.

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

### 5. Verify the whole path

```bash
python scripts/verify_end_to_end.py
```

This is done-criterion 3 as a runnable check, against a stack that is already
up. It mints a virtual key bound to a user id, sends a national ID, a mobile
and an IBAN through the proxy, and asserts that all three reached the provider
as placeholders; that tool-call arguments were masked too; that one audit row
per span landed carrying `user_api_key_user_id`; and — the assertion that
matters most — **that none of the values appear anywhere in `pii_events`**.
Everything else can pass on a system that also quietly writes the national ID
into a column, and that system is worse to operate than no audit trail at all.

Fixtures are generated by `pii_service.synthetic`, never pasted.

#### If you don't have a LiteLLM proxy to point it at

[`deploy/local-litellm/`](deploy/local-litellm/README.md) stands one up: the
pinned proxy image, its own Postgres, the guardrail mounted the way a site
would mount it, and Ollama on the host in place of vLLM. It is a test
harness, not product — the root compose file does not reference it.

---

## The web app

```bash
docker compose up -d --wait
open http://127.0.0.1:8099
```

Sign in with `PII_API_ADMIN_EMAIL` / `PII_API_ADMIN_INITIAL_PASSWORD`. The
console **refuses to serve the admin and chat routes until that password is
rotated** (brief §10) — the bootstrap credential is written in a config file
somewhere, so it is treated as one-time, not standing. The change-password
screen says so rather than just blocking, because a gate with no stated reason
reads as a bug and people work around bugs.

> **One permission level.** Every account that can log in has full visibility
> over the audit trail and the chat. Brief §10 sketches a read-only `auditor`
> role; it is deliberately **not** implemented, and there is no vestigial
> `role` column pretending otherwise. Adding it is one migration plus one
> dependency in `auth/deps.py`. Until then, account creation *is* the
> authorization model.

### Chat

Analyse first, then forward what survived. The backend calls `pii-service`
directly rather than reading LiteLLM response headers, because the chat needs
rich synchronous feedback and header-scraping breaks under streaming (brief
§2). The proxy guardrail still runs as enforcement; detection is idempotent, so
it re-scans already-masked text, finds nothing, and costs one cheap pass.

The reply streams over SSE, and **the first event on the wire is the analysis** —
the feedback chip renders before the first token exists.

Your own message keeps its original text with detected spans underlined in
category colours, plus an expandable `2 items redacted` chip showing type,
matched text and placeholder. That is not a disclosure: you typed it (brief §3).
A blocked message is not sent and **stays in the composer** with its spans
highlighted and a "remove flagged text" button, because the alternative is
asking someone to retype from memory the thing they were told not to send.

Pre-send detection is a per-user toggle, **off by default**. On, each pause in
typing is a real detection call and a real audit row — not a free preview.

### Admin

An event log and a drill-down panel. No leaderboard: brief §10 excludes it, and
"who leaked the most PII" is a scoreboard that changes behaviour without
improving it.

- **A virtualized event table.** One row per detected span, newest first; only
  the rows in view are mounted, because a busy gateway produces hundreds of
  thousands of spans and mounting that many rows locks the tab.
- **Drill-down is a right-side panel, not a route change**, so the list stays
  on screen as context.
- **Clicking a fingerprint pivots to every other occurrence of that value.**
  One click. This is the core investigative move, and it is the thing the
  pepper buys: the panel can say *"this value came from 3 different people
  across 7 requests"* while remaining unable to say what the value is.

> **No chart.** Brief §10 specifies a stacked timeline with brush-to-zoom above
> this table. It was built and then removed on request: the log is what people
> actually read, and the chart mostly competed with it for vertical space.
> `GET /api/admin/timeline` still serves the bucketed series, so restoring it
> is a component rather than an API change. Dropping Recharts also took the
> bundle from 555 kB to 167 kB.

Action is never encoded by colour alone — masked is filled, blocked is
outlined — so the distinction survives a colour-blind reader and a greyscale
screenshot, both of which happen to audit evidence.

RTL is handled where brief §10 says it must be: character offsets over Arabic
render in visually confusing order, so any fragment of user text sits in a
bidi-isolated element with `dir="auto"` and the offsets are reported separately
as numbers.

### Shape

```
browser ──▶ pii-web (nginx)  ──/api──▶ pii-api ──▶ pii-service ──▶ pii-db
                                          └──────▶ LiteLLM ──▶ model
```

`pii-api` **reads** `pii_events` and never writes it — pii-service stays the
single audit writer (brief §3), because an audit trail with two writers is one
where nobody can say which component produced a row. The two services share
`pii-db` but not a schema: each owns its tables and keeps its own Alembic
version table.

nginx serves the built SPA and proxies `/api`, so the app is same-origin and the
httpOnly session cookie needs no CORS relaxation.

---

## Detection

Three tiers, descending reliability. Presidio is the orchestration layer, not
the detector.

| Tier | What | Status |
|---|---|---|
| 1 | Deterministic recognizers: Egyptian national ID, mobile, IBAN, tax ID, passport, plus email and card | **On**, always |
| 2 | Arabic NER (ONNX int8) + Egyptian address gazetteer | **Measured and chosen**: `camelbert-msa-int8`. Off by default — set the two variables below |
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
# never does. The service package is needed too: every export ends by loading
# the artifact through the service's own recognizer, so a directory that cannot
# be loaded fails here rather than at startup on an air-gapped host.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers "optimum[onnxruntime]"
pip install -e "services/pii-service[ner]"

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

#### Result — `camelbert-msa-int8`

Measured 2026-09-20 on an Intel i7-10700K (8 cores, AVX2, no VNNI), against
all four artifacts. Gate: coverage ≥ 80%, ≤ 0.10 false positives per
entity-free sentence.

| Variant | Coverage | Egyptian | Mixed | MSA | FP/clean | p95 | Gate |
|---|---|---|---|---|---|---|---|
| **camelbert-msa-int8** | **80.2%** | **80.0%** | 64.7% | 89.7% | 0.00 | 22.6 ms | **pass** |
| camelbert-msa-fp32 | 78.0% | 77.8% | 58.8% | 89.7% | 0.00 | 38.2 ms | fail |
| camelbert-mix-fp32 | 74.7% | 73.3% | 52.9% | 89.7% | 0.00 | 37.4 ms | fail |
| camelbert-mix-int8 | 70.3% | 68.9% | 41.2% | 89.7% | 0.00 | 23.0 ms | fail |

```
PII_ENABLE_TIER2_ARABIC_NER=true
PII_TIER2_MODEL_DIR=/models/camelbert-msa-int8
```

**This contradicts the hypothesis in the table above**, and that is the reason
for running it rather than reasoning about it. `mix` was the expected winner —
dialectal pre-training, and this gateway's traffic is Egyptian, not newswire.
It lost, and it lost *on Egyptian specifically* (73.3% against 77.8%), which is
the column the argument for it was built on. The two variants are identical on
MSA (89.7%); the entire difference is dialect and mixed-script text.

Three caveats, all of which matter more than the headline:

- **80.2% clears an 80% gate by two spans out of 91.** That is not a margin. A
  different sample of the same traffic could put it on either side, so treat
  this as "the least-bad variant that is not disqualified", not as a model that
  is comfortably good enough. The honest reading is that Arabic NER covers
  roughly four names in five, and tier 1 carries the entities where a miss is
  worst.
- **int8 scoring above fp32 is noise, not a finding.** The gap is ~2 spans.
  Quantization did not improve the model; it moved a couple of borderline
  decisions. What the measurement does establish is the thing worth
  establishing: int8 costs no meaningful coverage while roughly halving p95.
- **Coverage is not precision-weighted.** Precision is 100% on this gold set
  with zero false positives on the 12 entity-free sentences, which is the half
  of the gate that keeps the gateway usable.

**What coverage cannot see.** It scores the share of gold spans overlapped by
*any* prediction, so a span truncated by one character still passes. That hid a
real masking bug: CAMeLBERT's WordPiece tokenizer splits Arabic words, and
inconsistent BIO tags over the sub-words left `المعادي` masked as
`<AR_LOCATION>ي` — a character of the matched value surviving in the text
forwarded to the model — and `لمنى` masked as `<AR_LOCATION><AR_LOCATION>`.
`_tidy_spans` now snaps span edges out to word boundaries and merges what then
touches. Coverage did not move (80.2%); character recall went 69.5% → 69.7%,
which is the metric that sees boundary quality. If you add a metric to the
harness, make it that one.

Extend the gold set and re-run before treating any of this as settled. 91 spans
decides a variant; it does not characterize a deployment.

#### The quantization trap — read this before exporting

`--quantization-target` defaults to `avx2` for a reason that cost real time to
find. Quantizing for a CPU feature the runtime host does not have produces an
artifact that **loads cleanly, runs, reports no error, and predicts nothing**:
u8s8 matmuls saturate without VNNI, the logits collapse to roughly ±1, and
every token decodes as `O`.

The first export here used `avx512_vnni` on an AVX2-only CPU. Both int8
variants scored **0.0% coverage** — not "int8 costs a little accuracy", but
total Arabic name-detection loss, with a healthy service and a clean log.

```
camelbert-mix-int8    0.0%   ...   FAIL      <- before
camelbert-mix-int8   70.3%   ...   FAIL      <- after, same weights, avx2 target
```

Two things now stop that reaching a deployment. `avx2` is the default, and it
is correct on every x86-64 CPU including those that do have VNNI, at a small
throughput cost. And `export_camelbert_onnx.py` **fails** rather than warns
when its smoke sentence yields no entities, so an inert artifact cannot be
written successfully. Pick `avx512_vnni` only after checking
`grep -o avx512_vnni /proc/cpuinfo` on the *deployment* host, not the build one.

#### Tier 2 latency

Measured on the same machine, with `scripts/benchmark.py`:

| Corpus | tier 1 | tier 1 + 2 |
|---|---|---|
| Arabic, chat-sized (46 chars) | p50 0.30 ms / p95 0.56 ms | p50 20.9 ms / p95 33.2 ms |
| Mixed script (47 chars) | p50 0.30 ms / p95 0.34 ms | p50 8.1 ms / p95 18.0 ms |
| Latin, chat-sized | p50 0.18 ms / p95 0.50 ms | p50 0.22 ms / p95 0.54 ms |
| Arabic, long document (1.7 kB) | p50 4.4 ms | **p50 563 ms** |

Two things to take from this. Latin text is unaffected — the router keeps tier
2 off anything that is not an Arabic run, so an English-only workload pays
nothing. And a pasted Arabic *document* costs over half a second on `pre_call`,
which is the number to design around if this gateway sees long pastes; chat-
sized messages at ~33 ms p95 are not the problem.

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
│   ├── litellm-pii-guardrail/  # the ~150-line CustomGuardrail adapter
│   ├── pii-api/                # web backend: auth, admin queries, chat + SSE
│   └── pii-web/                # React + Tailwind UI
├── deploy/local-litellm/       # TEST HARNESS, not product: a stand-in proxy
│                               # so done-criterion 3 can be run on a laptop
└── scripts/
    ├── verify_end_to_end.py    # done-criterion 3, as a runnable check
    ├── benchmark.py            # p50/p95 per tier on CPU
    ├── export_camelbert_onnx.py# build-time ONNX export (the only place torch is allowed)
    ├── eval_arabic_ner.py      # variant comparison, coverage-first
    └── seed_demo_events.py
```

## Not built yet

Phase 4 (the air-gapped `docker load` tarball) and phase 5 (the eval loop that
folds false-positive flags back into the Arabic gold set). The admin view
already collects those flags, so phase 5 is a consumer of data that now exists
rather than a feature needing new plumbing.

Also deliberately absent, and worth saying out loud rather than discovering:

- **The read-only `auditor` role.** One permission level, by request. See the
  note under [The web app](#the-web-app).
- **Break-glass reveal.** The role is defined in the data model and the
  endpoint stays disabled, per brief §6.
- **Chat history persistence.** Conversations live in the browser tab and are
  gone on reload. Storing them would mean a database of prompts sitting beside
  the audit trail that was carefully designed not to be one — that is a
  decision to take deliberately, with a retention policy, not by default.

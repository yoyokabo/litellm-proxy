# Proof of concept — one command, one workstation

Everything in one compose file: the PII services, a LiteLLM proxy, and Ollama
serving a model. Nothing external is assumed except internet access.

```bash
./up.sh
```

Then open **http://127.0.0.1:8099** and sign in with the credentials the script
prints. It will make you change the password immediately — that one is written
in `.env`, so it is a one-time credential.

`./up.sh` is safe to re-run. It generates only the secrets still unset, and
skips downloads already done.

## What you get

| | |
|---|---|
| Chat | Masked before it reaches the model, with a chip showing what was replaced |
| Events | Every detected span, with the fingerprint pivot |
| Entities | Per-entity policy: threshold, replacement, and whether to mask at all |
| Tiers | **All three on** — patterns, Arabic NER, Latin-script NER |
| Model | `llama3.1` via Ollama, behind LiteLLM with the guardrail enforced `pre_call` |

## The first run is slow, and then it is not

Nothing is baked into the images. On first start:

- **images** — a few GB. The service image carries torch for tier 3 and the
  conversion toolchain for tier 2's artifact.
- **model artifacts** — ~1.4 GB. Tier 3's weights are downloaded; tier 2's are
  *converted* here, because there is no published ONNX build of CAMeLBERT.
- **the language model** — ~4.7 GB for `llama3.1`.

All three land in named volumes, so every later start skips them. `docker
compose down` keeps them; `down -v` throws them away.

Both artifacts are verified before the service is allowed to start — loaded the
way the service loads them, and checked against a sentence they must not be
silent on. A model quantized for the wrong CPU loads cleanly and predicts
nothing, and that has shipped here before.

## Turning masking off for an entity

Entities → pick one → **When this is found**:

- **Mask it** — the span is replaced before the model sees it
- **Detect only — do not mask** — still detected, still audited, text passes
  through untouched
- **Block the request** — refused; the caller gets types and counts, never
  values

"Detect only" is the one to reach for when a team says masking is breaking
their prompts: you keep the record of who sent what and stop rewriting the
text. It takes effect on the next request, with no restart.

## What this is not

A demo host, not a hardened one. Before this shape goes anywhere real:

- **The UI is served over http**, so the session cookie cannot be `Secure`.
  `PII_API_COOKIE_SECURE=false` is the first line to change.
- **Both databases share one password**, and the audit database is not
  separated from LiteLLM's the way the root compose file separates them.
- **One permission level.** Everyone who can sign in can read the whole audit
  trail.
- **Not load tested.** The latency figures in the root README are sequential
  and single-threaded. A gateway serving a team is a different measurement.
- **Secrets live in `.env` on disk**, not in a secrets manager.

For a real deployment use the root `docker-compose.yml`, which attaches to the
LiteLLM proxy you already run rather than standing up its own.

## Pointing it at something else

Edit `.env`:

```bash
POC_OLLAMA_MODEL=mistral     # any model ollama can pull
POC_PORT=9000                # if 8099 is taken
POC_BIND=0.0.0.0             # reachable from the LAN — see "What this is not"
```

To run a tier off, set `PII_ENABLE_TIER2_ARABIC_NER=false` or
`PII_ENABLE_TIER3_GLINER=false` and restart. The bootstrap still fetches both
artifacts; pass `--skip-tier3` to `pii_service.bootstrap` if you want to avoid
the 1.2 GB download entirely.

## Behind a corporate proxy

```bash
PIP_CA_BUNDLE=/etc/ssl/certs/corporate-ca.pem ./up.sh
```

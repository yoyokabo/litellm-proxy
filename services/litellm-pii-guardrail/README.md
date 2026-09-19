# `litellm-pii-guardrail`

A `CustomGuardrail` that calls `pii-service` over HTTP and maps the result back
onto the request. No models, no policy, no database access — that is what makes
it drop-in: mount one file, add one config block, restart LiteLLM.

## Mounting it

The plugin is a single module, `pii_guardrail.py`. Mount it into the *existing*
LiteLLM container and put its directory on `PYTHONPATH`.

```yaml
# In your existing LiteLLM docker-compose.yml
services:
  litellm:
    volumes:
      - ./services/litellm-pii-guardrail/pii_guardrail.py:/app/pii_guardrail.py:ro
    environment:
      PYTHONPATH: /app
    networks:
      - litellm        # the same network pii-service joins
```

`httpx` is the only dependency and LiteLLM already ships it.

## Configuring it

Add to your existing LiteLLM `config.yaml`:

```yaml
guardrails:
  - guardrail_name: "pii-ar"
    litellm_params:
      guardrail: pii_guardrail.ArabicPIIGuardrail
      mode: "pre_call"
      service_url: http://pii-service:8090
      default_on: true
```

### Why `pre_call`

`during_call` runs in parallel with the LLM call, so by the time it has a
verdict the prompt has already been sent — it can block, but it cannot mask. We
mask, so `pre_call`, and we pay full detection latency on every request. That
cost is measured in `scripts/benchmark.py`; tier 1 alone is well under a
millisecond for chat-sized messages.

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `service_url` | `http://pii-service:8090` | Where `pii-service` is reachable |
| `timeout` | `4.0` | Seconds before a detection call is abandoned |
| `fail_closed` | `false` | See below |

### Fail-open vs fail-closed

By default, if `pii-service` is unreachable the request **goes through
unmasked** and the guardrail gets out of the way. That is the deliberate
trade: a guardrail that fails requests because its own dependency is down turns
a detection outage into a gateway outage for every engineer on the platform.

Set `fail_closed: true` where passing PII through is worse than refusing
service. Know which one you have chosen — the default is not a safe choice in
every deployment, it is a safe choice in most of them.

## The one thing not to "fix"

This guardrail masks by editing `inputs["texts"]` and `inputs["tool_calls"]`
**in place**, and it never rebinds `inputs["structured_messages"]`.

That looks like an omission. It is not. In litellm 1.101.0,
`llms/openai/chat/guardrail_translation/handler.py` applies the return value
like this:

```python
original = inputs.get("structured_messages")
out = await guardrail.apply_guardrail(inputs=inputs, ...)
sm = out.get("structured_messages")
if sm is not None and sm is not original:
    data["messages"] = merge(sm)                      # structured branch
else:
    apply(out["texts"]); apply(out["tool_calls"])     # texts branch
```

It is an `if`/`else`, keyed on the *identity* of `structured_messages`. The two
strategies are mutually exclusive. If the guardrail both edits `texts` and
returns a new `structured_messages` list, the handler takes the structured
branch and every text edit is silently discarded — the request reaches the
model **unmasked, with no error anywhere**.

Editing in place is what preserves that object identity and keeps the `else`
branch live. `tests/test_guardrail_contract.py` asserts the identity is
preserved, and drives litellm's real `OpenAIChatCompletionsHandler` to check
that masking actually lands in `data["messages"]`.

If you upgrade LiteLLM, re-run these tests before trusting the guardrail. The
mapping rule is internal to LiteLLM and can change.

## Tool calls

Tool-call arguments are scanned explicitly. Presidio's own hook reads only
content strings and `{"text": ...}` items, so arguments carried in conversation
history went unscanned on both wire formats; this plugin does not assume the
installed version is patched.

The `tool_calls` field is a union type — LiteLLM may hand over plain dicts or
pydantic `ChatCompletionMessageToolCall` objects depending on the API surface.
Both are handled, and both are covered by tests.

## Blocking

A blocked request raises `PiiBlockedError`, whose message carries entity types
and counts and never a matched value:

```
Request blocked by the pii-ar guardrail. Blocked entities: [('EG_NATIONAL_ID', 2)].
```

That message reaches the client and the logs, which is exactly why it contains
no values. Nothing is blocked by default — every entity in `entities.yaml`
ships as `MASK` or `ALLOW`. Blocking is enabled by changing an action there, in
`pii-service`, not here.

## Caching

Findings are cached by SHA-256 of the text, bounded to 1024 entries. The same
system prompt and the same pasted record recur on every turn of a conversation,
so in agentic workloads most requests cost no round trip at all.

`pii-service` caches too. This cache saves the HTTP hop; that one saves the
detection.

## Running the tests

```bash
cd services/litellm-pii-guardrail
uv venv .venv && uv pip install --python .venv/bin/python \
    litellm==1.101.0 pytest pytest-asyncio pytest-timeout respx httpx
.venv/bin/python -m pytest -q
```

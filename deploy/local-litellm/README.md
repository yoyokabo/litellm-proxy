# `deploy/local-litellm` — a stand-in for the proxy we don't own

The brief describes a LiteLLM proxy that already exists and is serving 100+
engineers, and this repository attaches to it rather than owning it. That is
the right boundary, and it leaves one problem: **done-criterion 3 cannot be
checked without a proxy.**

> A request through the *existing* LiteLLM proxy carrying an Egyptian national
> ID comes back masked, and a row lands in `pii_events` with the correct
> `user_api_key_user_id`.

So this directory stands one up locally. None of it is product. Nothing here
ships to a site, and the root `docker-compose.yml` does not reference it.

## What differs from a real deployment, and what doesn't

| | Here | At a site |
|---|---|---|
| Model | Ollama `llama3.1:8b` on the host | vLLM serving Qwen3-30B-A3B on a GPU host |
| Proxy port | Published on `127.0.0.1:4000` | Behind nginx |
| Network | `litellm_default`, created by this compose file | Already exists; set `LITELLM_NETWORK` |

Everything the guardrail actually touches is the same: the read-only single-file
mount, `PYTHONPATH=/app`, the `guardrails:` block copied verbatim from
[`../../services/litellm-pii-guardrail/README.md`](../../services/litellm-pii-guardrail/README.md),
the shared docker network, and a virtual key carrying a `user_id` — which is
the only way `user_api_key_user_id` reaches an audit row at all.

The LiteLLM image is **pinned to `v1.101.0`**, the version the contract tests
drive. The guardrail depends on a mapping rule internal to LiteLLM, so the
version that runs and the version that is tested must not drift apart by
accident. `:main-latest` would let them.

## Two fixtures that are not product

**`echo-provider/`** — an OpenAI-compatible provider that returns the request's
messages verbatim as its completion. Done-criterion 3 says "comes back masked",
and the honest way to check that is to look at what the *provider* was handed.
Asking an 8B model to repeat a message back makes the model the witness, and a
masking regression would then look like the model paraphrasing — which is
indistinguishable from the model being a model. Use `--model local-llama` to
exercise the real path, and `echo` to assert on it.

**`ollama-relay/`** — Ollama listens on `127.0.0.1:11434`, which a container
cannot reach. The alternative is setting `OLLAMA_HOST=0.0.0.0` on the host's
service, which publishes a model server on every interface in order to solve a
container-networking problem. The relay is the smaller change: `network_mode:
host`, binds one Docker bridge address, forwards to loopback. A real deployment
reaches vLLM over the network and drops both of these.

## Running it

```bash
cp .env.example .env
sed -i "s/^LITELLM_MASTER_KEY=.*/LITELLM_MASTER_KEY=sk-$(openssl rand -hex 16)/" .env
sed -i "s/^LITELLM_SALT_KEY=.*/LITELLM_SALT_KEY=$(openssl rand -hex 32)/" .env

ollama serve &            # if it isn't already running
ollama pull llama3.1

docker compose up -d --wait
```

Then bring up `pii-service` from the repository root with
`LITELLM_NETWORK=litellm_default`, and run the checks:

```bash
cd ../..
docker compose up -d --wait
python scripts/verify_end_to_end.py                     # asserts on the wire
python scripts/verify_end_to_end.py --model local-llama # through the real model
```

### If `RELAY_BIND` is wrong for your host

The relay binds `172.17.0.1`, the usual `docker0` address, because that is what
`host.docker.internal` resolves to for containers on a user-defined bridge.
Check yours and override if it differs:

```bash
ip -4 addr show docker0 | grep inet
OLLAMA_RELAY_BIND=172.17.0.1 docker compose up -d --wait
```

## Tearing it down

```bash
docker compose down -v            # here
cd ../.. && docker compose down -v # the audit stack, including its data
```

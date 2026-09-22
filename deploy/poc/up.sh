#!/usr/bin/env bash
# Bring up the proof of concept on a workstation with internet access.
#
#   ./up.sh
#
# Generates any secret still at CHANGE_ME, builds the images, downloads the
# model artifacts, and waits until the whole stack is healthy. Safe to re-run:
# secrets already set are left alone, and downloads already done are skipped.
set -euo pipefail

cd "$(dirname "$0")"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

# --- secrets ---------------------------------------------------------------
# Generated here rather than shipped, because a secret in a repository is not
# one. The audit pepper especially: the service refuses to start on a known
# value, and fingerprints computed under one are reversible by enumeration.
if [ ! -f .env ]; then
    cp .env.example .env
    say "Created .env"
fi

fill() {
    local key="$1" value="$2"
    if grep -qE "^${key}=CHANGE_ME$" .env; then
        # A literal replacement, so a generated value containing / or & cannot
        # corrupt the file.
        python3 - "$key" "$value" <<'PY'
import pathlib, sys
key, value = sys.argv[1], sys.argv[2]
path = pathlib.Path(".env")
path.write_text("\n".join(
    f"{key}={value}" if line == f"{key}=CHANGE_ME" else line
    for line in path.read_text().splitlines()
) + "\n")
PY
        echo "  generated ${key}"
    fi
}

say "Secrets"
fill PII_AUDIT_PEPPER "$(openssl rand -hex 32)"
fill PII_DB_PASSWORD "$(openssl rand -hex 24)"
fill PII_ADMIN_TOKEN "$(openssl rand -hex 24)"
fill PII_API_ADMIN_INITIAL_PASSWORD "$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)"
fill LITELLM_MASTER_KEY "sk-$(openssl rand -hex 16)"
fill LITELLM_SALT_KEY "$(openssl rand -hex 32)"

# --- build -----------------------------------------------------------------
# The pii-service image carries torch for tier 3 and the conversion toolchain
# for tier 2's artifact, so this is a few gigabytes and a few minutes.
say "Building images (several GB, first run only)"
docker compose build

# --- models ----------------------------------------------------------------
# Separated from `up` so the download has somewhere to report to. Tier 2 is
# converted rather than downloaded -- there is no published ONNX build of
# CAMeLBERT -- and both are verified before the service is allowed to start.
say "Downloading and preparing models (~1.4 GB, first run only)"
docker compose run --rm pii-models

say "Pulling the language model"
docker compose up -d ollama
docker compose run --rm ollama-pull

# --- up --------------------------------------------------------------------
say "Starting the stack"
docker compose up -d --wait

# --- report ----------------------------------------------------------------
set -a; . ./.env; set +a
say "Ready"
cat <<EOF

  Open      http://${POC_BIND:-127.0.0.1}:${POC_PORT:-8099}

  Email     ${PII_API_ADMIN_EMAIL:-admin@example.com}
  Password  ${PII_API_ADMIN_INITIAL_PASSWORD}

  You will be asked to change that password on first sign-in. That is
  deliberate: it is written in .env, so it is a one-time credential.

  Tiers     1 (patterns) + 2 (Arabic NER) + 3 (Latin NER), all on
  Model     ${POC_OLLAMA_MODEL:-llama3.1} via LiteLLM, guardrail enforced pre_call

  Logs      docker compose logs -f pii-service
  Stop      docker compose down
  Wipe      docker compose down -v      (drops the audit DB and the models)

EOF

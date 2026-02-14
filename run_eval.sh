#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Agent-S + Bytebot OSWorld Evaluation Runner
# =============================================================================
# Usage:
#   ./run_eval.sh                     # run first 20 tests (default)
#   ./run_eval.sh -n 50               # run first 50 tests
#   ./run_eval.sh -n 0                # run ALL tests
#   ./run_eval.sh -d os               # run only 'os' domain
#   ./run_eval.sh -d chrome -n 10     # first 10 chrome tests
#   ./run_eval.sh -s                  # use small test set (65 examples)
#   ./run_eval.sh -t "Open Firefox"   # single interactive task (no eval)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Configuration (edit these) ───────────────────────────────────────────────

# Bytebot VM
BYTEBOT_URL="http://localhost:9990"
CONTAINER_NAME="bytebot-desktop"
SCREEN_W=1280
SCREEN_H=960

# Main LLM
PROVIDER="anthropic"
MODEL="claude-sonnet-4-5"
MODEL_URL=""           # set for Azure Foundry, leave empty for direct API
MODEL_API_KEY=""       # leave empty to use ANTHROPIC_API_KEY env var

# Grounding model (UI-TARS on vLLM)
GROUND_PROVIDER="vllm"
GROUND_URL="http://localhost:4244/v1"
GROUND_API_KEY="dummy"
GROUND_MODEL="ByteDance-Seed/UI-TARS-1.5-7B"
GROUND_W=1920
GROUND_H=1080

# OSWorld paths (relative to this script)
EVAL_EXAMPLES_DIR="${SCRIPT_DIR}/bytebot-azure/osworld-bench/OSWorld/evaluation_examples"
TEST_ALL_JSON="${SCRIPT_DIR}/Agent-S/evaluation_sets/test_all.json"
TEST_SMALL_JSON="${SCRIPT_DIR}/Agent-S/evaluation_sets/test_small_new.json"
RESULT_DIR="${SCRIPT_DIR}/results"

# Defaults
MAX_EXAMPLES=20
DOMAIN="all"
MAX_STEPS=15
USE_SMALL=false
TASK=""

# ── Parse arguments ──────────────────────────────────────────────────────────

usage() {
    sed -n '5,13p' "$0" | sed 's/^# //'
    exit 0
}

while getopts "n:d:t:sh" opt; do
    case $opt in
        n) MAX_EXAMPLES="$OPTARG" ;;
        d) DOMAIN="$OPTARG" ;;
        t) TASK="$OPTARG" ;;
        s) USE_SMALL=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

# ── Load .env if present ─────────────────────────────────────────────────────

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
fi

# Pick base_url from env if not set explicitly
if [[ -z "$MODEL_URL" && -n "${ANTHROPIC_BASE_URL:-}" ]]; then
    MODEL_URL="$ANTHROPIC_BASE_URL"
fi

# ── Preflight checks ────────────────────────────────────────────────────────

echo "=== Preflight checks ==="

# Check bytebot container
if ! curl -sf -o /dev/null --connect-timeout 3 -X POST "$BYTEBOT_URL/computer-use" \
    -H "Content-Type: application/json" -d '{"action":"screenshot"}' 2>/dev/null; then
    echo "ERROR: Cannot connect to bytebot at $BYTEBOT_URL"
    echo "Start it with:"
    echo "  cd ${SCRIPT_DIR}/bytebot/docker"
    echo "  docker compose -f docker-compose.development.yml up -d bytebot-desktop"
    exit 1
fi
echo "  bytebot VM .............. OK ($BYTEBOT_URL)"

# Check vLLM
if ! curl -sf -o /dev/null --connect-timeout 3 "$GROUND_URL/models" 2>/dev/null; then
    echo "WARNING: Cannot reach grounding model at $GROUND_URL"
    echo "  Agent will fail at the grounding step."
else
    echo "  grounding model ......... OK ($GROUND_URL)"
fi

echo "  provider ................. $PROVIDER"
echo "  model .................... $MODEL"
if [[ -n "$MODEL_URL" ]]; then
    echo "  model_url ................ $MODEL_URL"
fi
echo ""

# ── Run mode: single task ───────────────────────────────────────────────────

if [[ -n "$TASK" ]]; then
    echo "=== Running single task ==="
    echo "Task: $TASK"
    echo ""

    ARGS=(
        -m gui_agents.s3.cli_app_remote
        --bytebot_url "$BYTEBOT_URL"
        --screen_width "$SCREEN_W" --screen_height "$SCREEN_H"
        --container_name "$CONTAINER_NAME"
        --provider "$PROVIDER" --model "$MODEL"
        --ground_provider "$GROUND_PROVIDER"
        --ground_url "$GROUND_URL"
        --ground_api_key "$GROUND_API_KEY"
        --ground_model "$GROUND_MODEL"
        --grounding_width "$GROUND_W" --grounding_height "$GROUND_H"
        --task "$TASK"
    )
    [[ -n "$MODEL_URL" ]] && ARGS+=(--model_url "$MODEL_URL")
    [[ -n "$MODEL_API_KEY" ]] && ARGS+=(--model_api_key "$MODEL_API_KEY")

    cd "$SCRIPT_DIR"
    exec uv run python3 "${ARGS[@]}"
fi

# ── Run mode: OSWorld evaluation ─────────────────────────────────────────────

# Pick test set
if $USE_SMALL; then
    META_PATH="$TEST_SMALL_JSON"
else
    META_PATH="$TEST_ALL_JSON"
fi

if [[ ! -f "$META_PATH" ]]; then
    echo "ERROR: Test metadata not found: $META_PATH"
    exit 1
fi
if [[ ! -d "$EVAL_EXAMPLES_DIR/examples" ]]; then
    echo "ERROR: Evaluation examples not found: $EVAL_EXAMPLES_DIR/examples/"
    echo "Clone OSWorld and place examples there."
    exit 1
fi

# Count examples
TOTAL=$(python3 -c "
import json, sys
with open('$META_PATH') as f:
    d = json.load(f)
domains = ['$DOMAIN'] if '$DOMAIN' != 'all' else list(d.keys())
n = sum(len(d.get(dom, [])) for dom in domains)
print(n)
")

RUN_N="$MAX_EXAMPLES"
if [[ "$MAX_EXAMPLES" -eq 0 ]]; then
    RUN_N="$TOTAL"
fi

echo "=== OSWorld Evaluation ==="
echo "  test set ................. $(basename "$META_PATH") ($TOTAL total)"
echo "  domain ................... $DOMAIN"
echo "  running .................. $RUN_N examples"
echo "  max steps/example ........ $MAX_STEPS"
echo "  results .................. $RESULT_DIR"
echo ""

mkdir -p "$RESULT_DIR"
mkdir -p "${SCRIPT_DIR}/Agent-S/osworld_setup/s3/logs"

ARGS=(
    run_local.py
    --provider_name bytebot
    --bytebot_url "$BYTEBOT_URL"
    --container_name "$CONTAINER_NAME"
    --screen_width "$SCREEN_W" --screen_height "$SCREEN_H"
    --model_provider "$PROVIDER" --model "$MODEL"
    --ground_provider "$GROUND_PROVIDER"
    --ground_url "$GROUND_URL"
    --ground_api_key "$GROUND_API_KEY"
    --ground_model "$GROUND_MODEL"
    --grounding_width "$GROUND_W" --grounding_height "$GROUND_H"
    --test_all_meta_path "$META_PATH"
    --test_config_base_dir "$EVAL_EXAMPLES_DIR"
    --max_examples "$MAX_EXAMPLES"
    --max_steps "$MAX_STEPS"
    --domain "$DOMAIN"
    --result_dir "$RESULT_DIR"
)
[[ -n "$MODEL_URL" ]] && ARGS+=(--model_url "$MODEL_URL")
[[ -n "$MODEL_API_KEY" ]] && ARGS+=(--model_api_key "$MODEL_API_KEY")

cd "${SCRIPT_DIR}/Agent-S/osworld_setup/s3"
exec uv run python3 "${ARGS[@]}"

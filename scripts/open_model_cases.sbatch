#!/bin/bash
#SBATCH --job-name=open-model-cases
#SBATCH --output=slurm-%x-%j.log
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#
# The assistant's evaluation cases with an open-weight model in Claude's place:
# the same launcher, settings, skills and guard, with the client pointed at a
# server this job starts on its own node (run_cases.py --base-url,
# docs/agent/cases/README.md). Only the model changes. The server is Ollama for
# weights on our own GPUs, or a LiteLLM proxy for a model another service runs
# behind an OpenAI-compatible API, such as Blablador's (API_BASE set; see below).
# Given a claude- MODEL and no API_BASE, the job starts no server and runs the
# same cases against Anthropic, so that Claude's runs to compare with are made on
# the same kind of node.
#
# Once, before the first run: put Ollama 0.14 or later (the first to serve the
# Anthropic API) where OLLAMA names it, and pull the model into OLLAMA_MODELS, a
# folder in the group's /fast (the home folder's 100 GB is too small), on a node
# that reaches the internet:
#
#     OLLAMA_MODELS=<folder> OLLAMA_HOST=127.0.0.1:11999 "$OLLAMA" serve &
#     OLLAMA_HOST=127.0.0.1:11999 "$OLLAMA" pull <model>; kill %1
#
# A model the library has only as :cloud, such as GLM-5.3 on 5 November 2026,
# comes from a GGUF on Hugging Face instead (unsloth/GLM-5.3-GGUF; Q8_0 is 17
# files, 801 GB). Download its files into one folder, then, with the same server
# running, `"$OLLAMA" create <name> -f <Modelfile>` with this Modelfile:
#
#     FROM <the folder of .gguf files>
#     RENDERER glm-4.7
#     PARSER glm-4.7
#
# Ollama has no renderer of its own for GLM-5, and GLM-5.3's chat template uses
# GLM-4.7's markers; check 2 below tests the pair. The import copies the files
# into OLLAMA_MODELS, so it needs their size twice until the folder is deleted.
# The architecture, glm-dsa, is in the llama.cpp that Ollama 0.40.14 builds
# (b11963, read 7 November 2026). At each load Ollama hard-links a model's files
# under llama.cpp's own names into a folder in OLLAMA_MODELS/blobs.
#
# Then, at MDC, a model that needs a whole node (GLM-5.3 in 8 bits is ~750 GB of
# weights; the GPU flags are the site's):
#
#     sbatch -p gpu --gres=gpu:h200:8 --exclusive --mem=0 \
#         --export=ALL,MODEL=<model>,OLLAMA_MODELS=<folder> scripts/open_model_cases.sbatch
#
# a model that fits one card, such as gpt-oss:120b (65 GB) on an H100 (MDC
# refuses a job on its H100 nodes that asks for more than 6 hours, 5 November 2026):
#
#     sbatch -p gpu --gres=gpu:h100:1 --time=06:00:00 --mem=128G \
#         --export=ALL,MODEL=gpt-oss:120b scripts/open_model_cases.sbatch
#
# and Claude's arm, which needs no GPU but does need a node that reaches
# Anthropic, and costs model use on the account like any batch:
#
#     sbatch -p normal --mem=16G --export=ALL,MODEL=claude-opus-5-5 scripts/open_model_cases.sbatch
#
# A model behind an OpenAI-compatible API needs no GPU either. Once, install
# LiteLLM into a virtual environment of its own (not the pipeline's), pinned to a
# release at least two weeks old; 1.106.5 is the one read for this script.
# Never 1.82.7 or 1.82.8: those two PyPI releases stole credentials (24 March
# 2026), the second through litellm_init.pth, which every Python start runs.
#
#     python3 -m venv <folder>/litellm-venv
#     <folder>/litellm-venv/bin/pip install 'litellm[proxy]==1.106.5'
#
# Then put the service's key in a file only you can read, which keeps it out of
# the command line, its history and the job's environment (Slurm can store a
# job's environment with its accounting record), and submit, here for
# Blablador's GLM-5.3-Flash (a key is a Personal Access Token of Helmholtz
# Codebase with the read_user scope; the model IDs are in the service's /v1/models):
#
#     read -rs key && (umask 077; printf '%s\n' "$key" > ~/.blablador_key); unset key
#     sbatch -p normal --mem=16G --export=ALL,MODEL=glm-5.3-flash,\
#     API_BASE=https://api.blablador.fz-juelich.de/v1,\
#     API_MODEL=hosted_vllm/alias-glm-5.3-flash,API_KEY_FILE=$HOME/.blablador_key,\
#     LITELLM=<folder>/litellm-venv/bin/litellm scripts/open_model_cases.sbatch
#
# MODEL is the name the client asks for and the batch records; API_MODEL is the
# model as LiteLLM names it, provider first. hosted_vllm/ sends the client's
# requests to the service's chat/completions; openai/ would send them to the
# Responses API, which a vLLM service may not have (_RESPONSES_API_PROVIDERS in
# LiteLLM's llms/anthropic/pass_through/messages/handler.py, 1.106.5). The proxy
# listens on 127.0.0.1 with a key made for this job, so no other job on the node
# spends the service's key. It holds requests to REQUESTS_PER_MINUTE (9), since
# Blablador throttles a user over 10 a minute for up to an hour
# (scripts/litellm_pacer.py), and makes no retries of its own: the client's
# retries pass the same hold.
#
# Required: MODEL; for Ollama also OLLAMA_MODELS, and OLLAMA (the executable)
# unless ollama is on PATH; for a service API_BASE, API_MODEL, API_KEY_FILE (or
# API_KEY, the key itself) and LITELLM unless litellm is on PATH. Optional: CASES ("--all", or ids such as
# "E11 E12"), RUNS (3), CONTEXT (131072, the window in tokens the client
# compacts within, and Ollama's), TIMEOUT (seconds a turn of a run may take: 3600
# with a server, run_cases.py's 1200 for Claude), LOAD_TIMEOUT (seconds an
# Ollama model may take to load, 7200) and REQUESTS_PER_MINUTE.
#
# llama-server prints nothing while it reads a model's weights, and Ollama gives
# up on a load that prints nothing for OLLAMA_LOAD_TIMEOUT, 5 minutes by default
# (Ollama 0.40.14 and llama.cpp b11963, read 7 November 2026). With mmap, the
# default, llama.cpp reads every file whole (MAP_POPULATE) before it goes on.
# GLM-5.3, 801 GB from /fast, was stopped so on 7 November 2026. The job sets
# OLLAMA_LOAD_TIMEOUT to LOAD_TIMEOUT and prints how long the load took.
# LLAMA_ARG_LOAD_MODE=none in --export reads the weights without mmap instead:
# Ollama passes its environment on to llama-server.
#
# Before the batch, with a server, the job checks and stops at the first that
# fails:
# 1. the model is there: pulled into OLLAMA_MODELS (a GPU node may have no
#    internet to pull it), or listed by the service's /v1/models under the key
#    (which also shows the node reaches the service);
# 2. it answers a request that offers a tool with a call of that tool, through
#    the Anthropic API the client speaks (this request also loads an Ollama model);
# 3. for Ollama, it sits wholly on the GPUs: one partly on the CPUs answers so
#    slowly that the turn and time limits would grade the hardware;
# 4. one run of E11 through the client ends with exit 0 and a tool call, under
#    the model named here: the client got through to the server.
# While the batch runs, a server that stops stops it, since every run after
# would fail. The server's log and the checks go to OUT
# (outputs/open-model-cases/<job id>); the batches to outputs/eval-runs/ as
# always, each with grading-auto.md from scripts/grade_cases.py, and with a
# service, backend.txt (service, model, LiteLLM release, requests held).
#
# Disposable.

set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?not running under Slurm}
[[ -f "$REPO/scripts/run_cases.py" ]] \
    || { echo "submit from the root of a cryoet-pipeline clone, not $REPO" >&2; exit 1; }
cd "$REPO"

[[ -n "${MODEL:-}" ]] || { echo "MODEL is not set" >&2; exit 1; }
read -ra cases <<< "${CASES:---all}"
RUNS=${RUNS:-3}
CONTEXT=${CONTEXT:-131072}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-7200}
[[ "$LOAD_TIMEOUT" =~ ^[0-9]+$ ]] \
    || { echo "LOAD_TIMEOUT is in seconds, not '$LOAD_TIMEOUT'" >&2; exit 1; }
OUT=${OUT:-outputs/open-model-cases/$SLURM_JOB_ID}
mkdir -p "$OUT"

source scripts/activate_env.sh
command -v claude > /dev/null || { echo "Claude Code (claude) is not on PATH" >&2; exit 1; }
# The memory, since a model loaded with mmap is read into it first.
echo "$(hostname): $(nvidia-smi -L 2> /dev/null | wc -l) GPUs," \
     "$(free -g 2> /dev/null | awk '/^Mem:/ {print $2}') GB of memory; model $MODEL"

# The first batch folder run_cases.py made since this UTC time: it names them by
# the time they start, so names sort as times do.
utc_now() { date -u +%Y%m%d-%H%M%S; }
batch_since() {
    local name
    for name in $(ls outputs/eval-runs 2> /dev/null | sort); do
        [[ "$name" < "$1" ]] || { echo "outputs/eval-runs/$name"; return; }
    done
}

# 2. A tool call through the Anthropic API, from the server at $1 within $2
# seconds; the token the server wants, if any, is in ANTHROPIC_AUTH_TOKEN.
tool_check() {
    local status=0
    python3 - "$1" "$MODEL" "$2" > "$OUT/tool-check.json" <<'EOF' || status=$?
import json
import os
import sys
import urllib.error
import urllib.request

host, model, timeout = sys.argv[1:]
token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or "local"
body = {
    "model": model, "max_tokens": 2048,
    "tools": [{"name": "read_file", "description": "Read a text file and return what it says.",
               "input_schema": {"type": "object", "required": ["path"],
                                "properties": {"path": {"type": "string"}}}}],
    "messages": [{"role": "user", "content": "What does notes.txt say? Read it with the tool."}],
}
# Both headers, as the client sends the token as a bearer and Ollama ignores them.
request = urllib.request.Request(
    f"http://{host}/v1/messages", data=json.dumps(body).encode(),
    headers={"content-type": "application/json", "anthropic-version": "2023-06-01",
             "x-api-key": token, "authorization": f"Bearer {token}"})
# The first request loads an Ollama model, so its answer may come only after the load.
reason = None
try:
    reply = json.load(urllib.request.urlopen(request, timeout=int(timeout)))
except urllib.error.HTTPError as error:
    # The server's reason, such as a model that did not load, is in the body
    # (7 November 2026: GLM-5.3 on maxg29 gave only "HTTP Error 500").
    reason = f"the server answered HTTP {error.code}: {error.read().decode(errors='replace')}"
except (TimeoutError, urllib.error.URLError) as error:
    reason = f"no answer from the server: {error!r} (the check waits {timeout} s)"
if reason:
    print(reason)
    print(reason, file=sys.stderr)
    sys.exit(1)
print(json.dumps(reply, indent=2))
calls = [block for block in reply.get("content", []) if block.get("type") == "tool_use"]
sys.exit(0 if calls and calls[0].get("name") == "read_file"
         and calls[0].get("input", {}).get("path") == "notes.txt" else 1)
EOF
    ((status == 0)) || { echo "$MODEL did not answer with a call of the tool it was offered;" \
                              "see $OUT/tool-check.json and $server_log" >&2; exit 1; }
    echo "check 2: $MODEL answered with a call of read_file"
}

# 4. One run through the client.
check_run() {
    local since check status=0
    since=$(utc_now)
    python3 scripts/run_cases.py E11 --runs 1 "${options[@]}" || true
    check=$(batch_since "$since")
    [[ -n "$check" ]] || { echo "the check run made no batch folder" >&2; exit 1; }
    python3 - "$check" "$MODEL" > "$OUT/check-run.txt" <<'EOF' || status=$?
import json
import sys
from pathlib import Path

folder, model = sys.argv[1:]
[record] = json.loads((Path(folder) / "runs.json").read_text())
print(f"check 4: model {record.get('model')}, exit codes {record.get('exit_codes')}, "
      f"{record.get('calls')} tool calls, largest context {record.get('largest_context')}")
sys.exit(0 if record.get("model") == model and (record.get("exit_codes") or [None])[0] == 0
         and (record.get("calls") or 0) >= 1 and not record.get("unreachable") else 1)
EOF
    cat "$OUT/check-run.txt"
    ((status == 0)) || { echo "the check run did not get through; see $check" >&2; exit 1; }
}

server=
server_log=
backend=
options=(--model "$MODEL")
# A port of this job's own: another job may run a server on the same node.
port=$((20000 + SLURM_JOB_ID % 20000))
if [[ -n "${API_BASE:-}" ]]; then
    [[ -n "${API_MODEL:-}" ]] \
        || { echo "API_MODEL is not set: the model as LiteLLM names it, such as" \
                  "hosted_vllm/alias-glm-5.3-flash" >&2; exit 1; }
    if [[ -z "${API_KEY:-}" && -n "${API_KEY_FILE:-}" ]]; then
        [[ -r "$API_KEY_FILE" ]] || { echo "API_KEY_FILE is not a readable file: '$API_KEY_FILE'" >&2
                                      exit 1; }
        API_KEY=$(< "$API_KEY_FILE")
    fi
    [[ -n "${API_KEY:-}" ]] \
        || { echo "no key: name the file that holds the service's key in API_KEY_FILE" \
                  "(see this script's header)" >&2; exit 1; }
    LITELLM=${LITELLM:-$(command -v litellm || true)}
    [[ -x "$LITELLM" ]] || { echo "LITELLM is not an executable: '$LITELLM'" >&2; exit 1; }
    # Read from the files, since a compromised install runs at any Python start.
    venv=$(dirname "$(dirname "$LITELLM")")
    if [[ -n "$(find "$venv" -name litellm_init.pth 2> /dev/null | head -n 1)" ]]; then
        echo "$venv holds litellm_init.pth, the payload of LiteLLM 1.82.8: do not run it;" \
             "delete the environment and rotate the keys its machine held" >&2
        exit 1
    fi
    litellm_version=$(find "$venv" -maxdepth 5 -type d -name 'litellm-*.dist-info' 2> /dev/null \
                      | sed -E 's/.*litellm-(.*)\.dist-info$/\1/' | head -n 1)
    case "$litellm_version" in
        "") echo "no LiteLLM install found in $venv" >&2; exit 1 ;;
        1.82.7|1.82.8) echo "LiteLLM $litellm_version is a compromised release (see this" \
                            "script's header)" >&2; exit 1 ;;
    esac
    # The pacer writes each answer's prompt size to USAGE_LOG: through the proxy
    # the client records 0 input tokens (scripts/litellm_pacer.py).
    export API_KEY REQUESTS_PER_MINUTE=${REQUESTS_PER_MINUTE:-9} USAGE_LOG=$OUT/usage.jsonl

    # 1. Listed by the service under the key; also shows the node reaches it.
    status=0
    python3 - "$API_BASE" "${API_MODEL#*/}" > "$OUT/models.json" <<'EOF' || status=$?
import json
import os
import sys
import urllib.error
import urllib.request

base, model = sys.argv[1:]
# The key from the environment, not the command line, where ps would show it.
request = urllib.request.Request(f"{base.rstrip('/')}/models",
                                 headers={"authorization": f"Bearer {os.environ['API_KEY']}"})
try:
    listing = json.load(urllib.request.urlopen(request, timeout=60))
except urllib.error.HTTPError as error:
    sys.exit(f"{base} answered HTTP {error.code}: {error.read().decode(errors='replace')}")
except (TimeoutError, urllib.error.URLError) as error:
    sys.exit(f"no answer from {base}: {error!r}")
print(json.dumps(listing, indent=2))
ids = sorted(entry.get("id", "") for entry in listing.get("data", []))
if model not in ids:
    sys.exit(f"{base} does not list {model}; it lists: {', '.join(ids)}")
EOF
    ((status == 0)) || { echo "check 1 failed; see above and $OUT/models.json" >&2; exit 1; }
    echo "check 1: $API_BASE lists ${API_MODEL#*/}"

    # The proxy: LiteLLM imports the pacer from the config's folder.
    cp scripts/litellm_pacer.py "$OUT/"
    LITELLM_MASTER_KEY=sk-$(python3 -c 'import secrets; print(secrets.token_hex(16))')
    # The client's token (backend_env in run_cases.py reads it from this shell).
    export LITELLM_MASTER_KEY ANTHROPIC_AUTH_TOKEN=$LITELLM_MASTER_KEY
    # The model price list from the install, not from GitHub at each start.
    export LITELLM_LOCAL_MODEL_COST_MAP=True
    cat > "$OUT/litellm.yaml" <<EOF
model_list:
  - model_name: $MODEL
    litellm_params:
      model: $API_MODEL
      api_base: $API_BASE
      api_key: os.environ/API_KEY
      num_retries: 0
litellm_settings:
  callbacks: litellm_pacer.pacer
  drop_params: true
router_settings:
  num_retries: 0
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
EOF
    server_log=$OUT/litellm.log
    "$LITELLM" --config "$OUT/litellm.yaml" --host 127.0.0.1 --port "$port" > "$server_log" 2>&1 &
    server=$!
    trap 'kill "$server" 2> /dev/null || true' EXIT
    for _ in $(seq 60); do
        curl -sf "http://127.0.0.1:$port/health/liveliness" > /dev/null && break
        kill -0 "$server" 2> /dev/null \
            || { echo "LiteLLM stopped while starting; see $server_log" >&2; exit 1; }
        sleep 5
    done
    curl -sf "http://127.0.0.1:$port/health/liveliness" > /dev/null \
        || { echo "LiteLLM did not answer within 5 minutes; see $server_log" >&2; exit 1; }
    backend="service $API_BASE, model $API_MODEL, LiteLLM $litellm_version at 127.0.0.1:$port,"
    backend+=" at most $REQUESTS_PER_MINUTE requests a minute, window $CONTEXT tokens"
    echo "$backend"

    tool_check "127.0.0.1:$port" 900
    options+=(--base-url "http://127.0.0.1:$port" --context-tokens "$CONTEXT"
              --timeout "${TIMEOUT:-3600}")
    check_run
elif [[ "$MODEL" != claude-* ]]; then
    OLLAMA=${OLLAMA:-$(command -v ollama || true)}
    [[ -x "$OLLAMA" ]] || { echo "OLLAMA is not an executable: '$OLLAMA'" >&2; exit 1; }
    [[ -d "${OLLAMA_MODELS:-}" ]] \
        || { echo "OLLAMA_MODELS is not a folder: '${OLLAMA_MODELS:-}'" >&2; exit 1; }
    export OLLAMA_MODELS
    # "ollama version is X", or "Warning: client version is X" with no server.
    version=$("$OLLAMA" --version 2>&1 | grep -oE 'version is [0-9][0-9.]*' | tail -n 1 || true)
    version=${version#version is }
    if [[ "$(printf '%s\n' 0.14 "$version" | sort -V | head -n 1)" != 0.14 ]]; then
        echo "Ollama '$version' is older than 0.14, the first to serve the Anthropic API" >&2
        exit 1
    fi

    export OLLAMA_HOST=127.0.0.1:$port
    # One request at a time, each with the whole window, and the model kept loaded.
    export OLLAMA_CONTEXT_LENGTH=$CONTEXT OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=-1
    # The whole load, which prints nothing (see the header).
    export OLLAMA_LOAD_TIMEOUT=$LOAD_TIMEOUT
    server_log=$OUT/ollama.log
    "$OLLAMA" serve > "$server_log" 2>&1 &
    server=$!
    trap 'kill "$server" 2> /dev/null || true' EXIT
    for _ in $(seq 60); do
        curl -sf "http://$OLLAMA_HOST/api/version" > /dev/null && break
        kill -0 "$server" 2> /dev/null \
            || { echo "Ollama stopped while starting; see $server_log" >&2; exit 1; }
        sleep 5
    done
    curl -sf "http://$OLLAMA_HOST/api/version" > /dev/null \
        || { echo "Ollama did not answer within 5 minutes; see $server_log" >&2; exit 1; }
    echo "Ollama $version at $OLLAMA_HOST, window $CONTEXT tokens"

    # 1. Pulled.
    "$OLLAMA" show "$MODEL" > "$OUT/model.txt" 2>&1 \
        || { echo "$MODEL is not in $OLLAMA_MODELS; pull it first (see this script's header)" >&2
             exit 1; }

    # 2. The load, then up to 15 minutes for the answer.
    tool_check "$OLLAMA_HOST" "$((LOAD_TIMEOUT + 900))"
    grep -o 'llama-server started in [0-9.]* seconds' "$server_log" | tail -n 1 || true

    # 3. Wholly on the GPUs ("100% GPU", or "48%/52% CPU/GPU", in Ollama's ps).
    "$OLLAMA" ps > "$OUT/ps.txt" 2>&1 || true
    cat "$OUT/ps.txt"
    grep -q "100% GPU" "$OUT/ps.txt" \
        || { echo "$MODEL is not wholly on the GPUs (see $OUT/ps.txt): ask for more GPUs" \
                  "or pull a smaller quantization" >&2; exit 1; }

    options+=(--base-url "http://$OLLAMA_HOST" --context-tokens "$CONTEXT"
              --timeout "${TIMEOUT:-3600}")
    check_run
elif [[ -n "${TIMEOUT:-}" ]]; then
    options+=(--timeout "$TIMEOUT")
fi

since=$(utc_now)
python3 scripts/run_cases.py "${cases[@]}" --runs "$RUNS" "${options[@]}" &
runner=$!
while kill -0 "$runner" 2> /dev/null; do
    if [[ -n "$server" ]] && ! kill -0 "$server" 2> /dev/null; then
        echo "the server stopped during the batch; see $server_log" >&2
        kill "$runner" 2> /dev/null || true
        break
    fi
    sleep 30
done
status=0
wait "$runner" || status=$?

batch=$(batch_since "$since")
if [[ -n "$batch" ]]; then
    python3 scripts/grade_cases.py "$batch" || true
    if [[ -n "$backend" ]]; then
        # How often the pacer held a request, whether the service refused any, and
        # the largest prompt it answered (the checks' included), since the runs'
        # records have no largest context through the proxy.
        held=$(grep -c '^litellm_pacer: holding' "$server_log" || true)
        refused=$(grep -c 'RateLimitError' "$server_log" || true)
        largest=$(python3 -c 'import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
top = max((row.get("prompt_tokens") or 0 for row in rows), default=0)
print(f"{len(rows)} answers, largest prompt {top:,} tokens")' \
            "$USAGE_LOG" 2> /dev/null || echo "no answers logged")
        cp "$USAGE_LOG" "$batch/usage.jsonl" 2> /dev/null || true
        echo "$backend; $held requests held, $refused log lines with RateLimitError; $largest" \
            | tee "$batch/backend.txt"
    fi
    echo "$batch" > "$OUT/batch.txt"
    echo "batch: $batch"
fi
exit "$status"

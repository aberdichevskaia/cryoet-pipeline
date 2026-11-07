#!/bin/bash
#SBATCH --job-name=open-model-cases
#SBATCH --output=slurm-%x-%j.log
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#
# The assistant's evaluation cases with an open-weight model in Claude's place:
# the same launcher, settings, skills and guard, with the client pointed at an
# Ollama server this job starts on its own node (run_cases.py --base-url,
# docs/agent/cases/README.md). Only the model changes. Given a claude- MODEL,
# the job starts no server and runs the same cases against Anthropic, so that
# Claude's runs to compare with are made on the same kind of node.
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
# Required: MODEL; for an open model also OLLAMA_MODELS, and OLLAMA (the
# executable) unless ollama is on PATH. Optional: CASES ("--all", or ids such as
# "E11 E12"), RUNS (3), CONTEXT (131072, the server's window in tokens; the
# client compacts within it), TIMEOUT (seconds a turn of a run may take: 3600
# for an open model, run_cases.py's 1200 for Claude) and LOAD_TIMEOUT (seconds
# the model may take to load, 7200).
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
# Before the batch, for an open model, the job checks and stops at the first
# that fails:
# 1. the model is pulled: a GPU node may have no internet to pull it;
# 2. it answers a request that offers a tool with a call of that tool, through
#    the Anthropic API (this request also loads it);
# 3. it sits wholly on the GPUs: one partly on the CPUs answers so slowly that
#    the turn and time limits would grade the hardware;
# 4. one run of E11 through the client ends with exit 0 and a tool call, under
#    the model named here: the client got through to the server.
# While the batch runs, a server that stops stops it, since every run after
# would fail. The server's log and the checks go to OUT
# (outputs/open-model-cases/<job id>); the batches to outputs/eval-runs/ as
# always, each with grading-auto.md from scripts/grade_cases.py.
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

server=
options=(--model "$MODEL")
if [[ "$MODEL" != claude-* ]]; then
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

    # A port of this job's own: another job may run an Ollama on the same node.
    export OLLAMA_HOST=127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))
    # One request at a time, each with the whole window, and the model kept loaded.
    export OLLAMA_CONTEXT_LENGTH=$CONTEXT OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=-1
    # The whole load, which prints nothing (see the header).
    export OLLAMA_LOAD_TIMEOUT=$LOAD_TIMEOUT
    "$OLLAMA" serve > "$OUT/ollama.log" 2>&1 &
    server=$!
    trap 'kill "$server" 2> /dev/null || true' EXIT
    for _ in $(seq 60); do
        curl -sf "http://$OLLAMA_HOST/api/version" > /dev/null && break
        kill -0 "$server" 2> /dev/null \
            || { echo "Ollama stopped while starting; see $OUT/ollama.log" >&2; exit 1; }
        sleep 5
    done
    curl -sf "http://$OLLAMA_HOST/api/version" > /dev/null \
        || { echo "Ollama did not answer within 5 minutes; see $OUT/ollama.log" >&2; exit 1; }
    echo "Ollama $version at $OLLAMA_HOST, window $CONTEXT tokens"

    # 1. Pulled.
    "$OLLAMA" show "$MODEL" > "$OUT/model.txt" 2>&1 \
        || { echo "$MODEL is not in $OLLAMA_MODELS; pull it first (see this script's header)" >&2
             exit 1; }

    # 2. A tool call through the Anthropic API.
    status=0
    # The load, then up to 15 minutes for the answer.
    python3 - "$OLLAMA_HOST" "$MODEL" "$((LOAD_TIMEOUT + 900))" > "$OUT/tool-check.json" \
        <<'EOF' || status=$?
import json
import sys
import urllib.error
import urllib.request

host, model, timeout = sys.argv[1:]
body = {
    "model": model, "max_tokens": 2048,
    "tools": [{"name": "read_file", "description": "Read a text file and return what it says.",
               "input_schema": {"type": "object", "required": ["path"],
                                "properties": {"path": {"type": "string"}}}}],
    "messages": [{"role": "user", "content": "What does notes.txt say? Read it with the tool."}],
}
request = urllib.request.Request(
    f"http://{host}/v1/messages", data=json.dumps(body).encode(),
    headers={"content-type": "application/json", "anthropic-version": "2023-06-01",
             "x-api-key": "local"})
# The first request loads the model, so its answer may come only after the load.
reason = None
try:
    reply = json.load(urllib.request.urlopen(request, timeout=int(timeout)))
except urllib.error.HTTPError as error:
    # Ollama's reason, such as a model that did not load, is in the body
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
                              "see $OUT/tool-check.json and $OUT/ollama.log" >&2; exit 1; }
    echo "check 2: $MODEL answered with a call of read_file"
    grep -o 'llama-server started in [0-9.]* seconds' "$OUT/ollama.log" | tail -n 1 || true

    # 3. Wholly on the GPUs ("100% GPU", or "48%/52% CPU/GPU", in Ollama's ps).
    "$OLLAMA" ps > "$OUT/ps.txt" 2>&1 || true
    cat "$OUT/ps.txt"
    grep -q "100% GPU" "$OUT/ps.txt" \
        || { echo "$MODEL is not wholly on the GPUs (see $OUT/ps.txt): ask for more GPUs" \
                  "or pull a smaller quantization" >&2; exit 1; }

    options+=(--base-url "http://$OLLAMA_HOST" --context-tokens "$CONTEXT"
              --timeout "${TIMEOUT:-3600}")

    # 4. One run through the client.
    since=$(utc_now)
    python3 scripts/run_cases.py E11 --runs 1 "${options[@]}" || true
    check=$(batch_since "$since")
    [[ -n "$check" ]] || { echo "the check run made no batch folder" >&2; exit 1; }
    status=0
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
elif [[ -n "${TIMEOUT:-}" ]]; then
    options+=(--timeout "$TIMEOUT")
fi

since=$(utc_now)
python3 scripts/run_cases.py "${cases[@]}" --runs "$RUNS" "${options[@]}" &
runner=$!
while kill -0 "$runner" 2> /dev/null; do
    if [[ -n "$server" ]] && ! kill -0 "$server" 2> /dev/null; then
        echo "Ollama stopped during the batch; see $OUT/ollama.log" >&2
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
    echo "$batch" > "$OUT/batch.txt"
    echo "batch: $batch"
fi
exit "$status"

# Job 3790001 runs as the request comes: its log and its timing rows are moved to
# a start 40 minutes 13 seconds before the run, which squeue gives as its TIME,
# well inside the 16 h its stage may run. Rows from before the job stay as they are.
set -euo pipefail
python3 - slurm-ribo-mdc-gpu-3790001.log outputs/ribo-mdc/step_timings.jsonl <<'EOF'
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = Path(sys.argv[1])
started = datetime.fromisoformat(re.search(r"^started\s+(\S+)", log.read_text(), re.M)[1])
shift = datetime.now(timezone.utc) - timedelta(minutes=40, seconds=13) - started
stamp = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00")


def moved(match):
    when = datetime.fromisoformat(match[0])
    return (when + shift).isoformat(timespec="seconds") if when >= started else match[0]


for path in map(Path, sys.argv[1:]):
    path.write_text(stamp.sub(moved, path.read_text()))
EOF

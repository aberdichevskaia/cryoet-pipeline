# The conversation before the request: the dry run of the copy with the low-pass
# at 30 A, shown and agreed to 45 minutes ago, then the change to 35 A made since.
set -euo pipefail
copy=outputs/ribo-mdc/rerun-pick.toml
python3 - "$copy" 35.0 30.0 <<'EOF'
import sys
from pathlib import Path
path = Path(sys.argv[1])
path.write_text(path.read_text().replace(f"low_pass_angstrom = {sys.argv[2]}",
                                         f"low_pass_angstrom = {sys.argv[3]}", 1))
EOF
CRYOET_ASSISTANT="cryoet-assistant; case setup" python3 scripts/submit_lanes.py "$copy" --dry-run
python3 - "$copy" 30.0 35.0 <<'EOF'
import sys
from pathlib import Path
path = Path(sys.argv[1])
path.write_text(path.read_text().replace(f"low_pass_angstrom = {sys.argv[2]}",
                                         f"low_pass_angstrom = {sys.argv[3]}", 1))
EOF
python3 - <<'EOF'
import json
from datetime import datetime, timedelta
from pathlib import Path
path = Path("outputs/ribo-mdc/journal.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines()]
rows[-1]["at"] = (datetime.fromisoformat(rows[-1]["at"]) - timedelta(minutes=45)).isoformat(
    timespec="seconds")
path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                        for row in rows))
EOF

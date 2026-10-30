# Puts this sandbox's own folder, home and user where the collector left
# placeholders for the cluster's, and gives every file of the case the time it was
# last written there: git keeps no file times, and show_failure finds a step's
# logs by when they were written.
set -euo pipefail
python3 scripts/collect_case.py --restore "$(dirname "$0")"

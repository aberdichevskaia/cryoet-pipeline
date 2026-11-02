# Puts this sandbox's own folder, home and user where the collector left
# placeholders for the cluster's, and gives every file of the case the time it was
# last written there: git keeps no file times, and show_failure finds a step's
# logs by when they were written. Takes out the project's lane files the case does
# not hold, which were added after it ended.
set -euo pipefail
python3 scripts/collect_case.py --restore "$(dirname "$0")"

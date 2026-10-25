#!/bin/sh
# Delete quoting-journal runs (var/journal/run=*) that have not been written to in KEEP_DAYS days.
#
# Decided 2026-09-23: the journal is not kept -- nothing in the live engine reads a past run
# (strategy/journal.py only appends), and a 96-minute live session wrote 293 MB to the 32 GB EBS
# root. Only backtest/live_* post-mortems read it, and they must be run within the window.
#
# Age is the NEWEST file in the run, not the directory's own mtime, so a run still being written
# (a live session that started days ago would be unusual, but a directory mtime does not move when
# an existing file is appended to) is never removed from under the writer.
set -eu
KEEP_DAYS="${KEEP_DAYS:-3}"
DIR="${KALSHI_JOURNAL_DIR:-$(cd "$(dirname "$0")/.." && pwd)/var/journal}"
[ -d "$DIR" ] || exit 0
for run in "$DIR"/run=*; do
    [ -d "$run" ] || continue
    if [ -z "$(find "$run" -type f -mtime -"$KEEP_DAYS" -print -quit)" ]; then
        echo "$(date -Is) prune $(du -sh "$run" | cut -f1) $run"
        rm -rf -- "$run"
    fi
done

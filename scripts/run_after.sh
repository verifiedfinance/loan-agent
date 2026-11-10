#!/bin/bash
# run_after.sh <pid> <script>: start <script> once process <pid> has ended (successfully or not).
# Reads the process state from /proc, so a zombie or stopped parent is not mistaken for a running job.
cd "$(dirname "$0")/.."
pid=$1; next=$2
alive() { [ -r /proc/$pid/stat ] && ! awk '{exit !($3=="Z")}' /proc/$pid/stat; }
while alive; do sleep 60; done
echo "[$(date '+%F %T')] process $pid ended; starting $next"
exec "$next"

#!/bin/bash
# Start the schedule-reward run once the Mistral runner (PID $1) has ended, successfully or not.
# Uses the process state from /proc, not kill -0, so a zombie or stopped parent is not mistaken for a running job.
cd "$(dirname "$0")/.."
pid=$1
alive() { [ -r /proc/$pid/stat ] && ! awk '{exit !($3=="Z")}' /proc/$pid/stat; }
while alive; do sleep 60; done
echo "[$(date '+%F %T')] mistral runner $pid ended; starting schedule-reward run"
exec scripts/09_schedule_reward.sh

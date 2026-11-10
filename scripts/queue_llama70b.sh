#!/bin/bash
# Queue for the 70B reader: waits until the Llama schedule-reward rerun (10_llama_sched.sh) has finished or stopped,
# the 70B download has finished, and the GPU is free; then runs 12_llama70b_reader.sh, relaunching it up to
# 3 times if it exits without its done-marker. Waits use log markers, process state and GPU memory, never a bare PID
# (see the paused-queue zombie incident of 2026-11-04).
#   nohup scripts/queue_llama70b.sh > logs/queue_llama70b.log 2>&1 &
set -u; cd "$(dirname "$0")/.."
log() { echo "[$(date '+%F %T')] $*"; }
alive() {  # alive <pattern>: a matching process exists and is not a zombie
  pgrep -f "$1" | while read -r p; do ps -o stat= -p "$p" | grep -qv Z && echo y; done | grep -q y
}

log "waiting for the Llama rerun (LLAMA_SCHED_DONE, or 10_llama_sched.sh no longer running)"
until grep -q LLAMA_SCHED_DONE logs/llama_sched.log 2>/dev/null || ! alive "scripts/10_llama_sched.sh"; do sleep 60; done
grep -q LLAMA_SCHED_DONE logs/llama_sched.log && log "Llama rerun finished" || log "WARNING: Llama rerun stopped without LLAMA_SCHED_DONE"

log "waiting for the 70B download"
until grep -q DOWNLOAD_DONE logs/download_llama70b.log 2>/dev/null || ! alive "snapshot_download\('meta-llama/Llama-3.3-70B"; do sleep 60; done

log "waiting for >= 60 GB free on the GPU (it may be shared with another user)"
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 60000 ]; do sleep 60; done

for i in 1 2 3; do
  log "start 12_llama70b_reader.sh (launch $i/3)"
  bash scripts/12_llama70b_reader.sh >> logs/llama70b_reader.log 2>&1
  grep -q LLAMA70B_READER_DONE logs/llama70b_reader.log && { log "QUEUE_LLAMA70B_DONE"; exit 0; }
  log "runner exited without LLAMA70B_READER_DONE"; sleep 120
done
log "QUEUE_LLAMA70B_FAILED after 3 launches"

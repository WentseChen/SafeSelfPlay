#!/bin/bash
# Supervisor for the srtchain training chunks. Keeps the queue topped up so we grab
# whatever GPUs free up, and tracks cumulative training steps across chunks.
#
#   start:  setsid nohup bash /home/wentsec/chain_monitor.sh >/dev/null 2>&1 &
#   stop :  touch /home/wentsec/STOP_CHAIN
#   watch:  tail -f /home/wentsec/logs/chain_monitor.log
CHAIN_DIR=/project/flame/wentsec/srt_chain
LOG=/home/wentsec/logs/chain_monitor.log
STOP=/home/wentsec/STOP_CHAIN
SUBMITTED=$CHAIN_DIR/.submitted_count

TARGET_QUEUED=8      # keep this many chunks pending, so backfill always has candidates
MAX_SUBMITS=300      # lifetime cap: a runaway submitter would be worse than a slow one
STALL_LIMIT=3        # consecutive finished chunks that added 0 steps -> stop submitting
POLL=300

mkdir -p $(dirname $LOG) $CHAIN_DIR
[ -f $SUBMITTED ] || echo 0 > $SUBMITTED
say () { echo "$(date +'%m-%d %H:%M:%S') $*" >> $LOG; }

say "monitor start (target_queued=$TARGET_QUEUED cap=$MAX_SUBMITS)"
last_steps=$(cat $CHAIN_DIR/steps_done 2>/dev/null || echo 0)
last_done=$(ls /home/wentsec/logs/chain_*.out 2>/dev/null | wc -l)
stall=0

while [ ! -f $STOP ]; do
  n_all=$(squeue -u wentsec -h -n srtchain 2>/dev/null | wc -l)
  n_run=$(squeue -u wentsec -h -n srtchain -t R 2>/dev/null | wc -l)
  n_pd=$((n_all - n_run))
  steps=$(cat $CHAIN_DIR/steps_done 2>/dev/null || echo 0)
  subs=$(cat $SUBMITTED)

  # progress / stall detection: count finished chunk logs vs steps gained
  done_now=$(ls /home/wentsec/logs/chain_*.out 2>/dev/null | wc -l)
  if [ "$done_now" -gt "$last_done" ]; then
    if [ "$steps" -le "$last_steps" ]; then
      stall=$((stall + 1)); say "WARN chunk finished with no step gain (stall=$stall/$STALL_LIMIT)"
    else
      stall=0
    fi
    last_done=$done_now; last_steps=$steps
  fi

  if [ $stall -ge $STALL_LIMIT ]; then
    say "ALERT $STALL_LIMIT chunks in a row added no steps -- pausing submissions, investigate"
    sleep $POLL; continue
  fi

  if [ "$n_pd" -lt "$TARGET_QUEUED" ] && [ "$subs" -lt "$MAX_SUBMITS" ]; then
    need=$((TARGET_QUEUED - n_pd))
    for k in $(seq 1 $need); do
      [ "$(cat $SUBMITTED)" -ge "$MAX_SUBMITS" ] && break
      # alternate shapes/walltimes: more distinct shapes = more backfill chances
      case $(( (subs + k) % 4 )) in
        0) g=4; t=01:00:00 ;;
        1) g=8; t=01:00:00 ;;
        2) g=4; t=00:45:00 ;;
        3) g=4; t=02:00:00 ;;
      esac
      if sbatch --gres=gpu:$g -t $t /home/wentsec/run_chunk.sh >/dev/null 2>&1; then
        echo $(( $(cat $SUBMITTED) + 1 )) > $SUBMITTED
      fi
    done
    say "topped up: queued=$n_pd running=$n_run -> submitted $need (gpu mix), lifetime=$(cat $SUBMITTED)"
  fi

  node=$(squeue -u wentsec -h -n srtchain -t R -o '%N' 2>/dev/null | head -1)
  say "status steps=$steps running=$n_run(${node:-none}) queued=$n_pd lifetime_subs=$(cat $SUBMITTED)"
  sleep $POLL
done
say "monitor stopped (STOP flag)"

#!/bin/bash
# Replaces orchestrate_rest.sh: start Issues 5-7 per-seed stages now in new worktrees
# w5..w9 (they need only Issues 1-3, which are complete), while the Issue 4 workers in
# w0..w4 keep running. When both sets are done: merge Issue 4, merge Issues 5-7, then let
# the main run build Issues 4-7 tables and Issue 7's cross-seed aggregation.
set -u
MAIN=~/misbah-final-code
PAR=~/misbah-par
PY=$MAIN/.venv/bin/python
Q1=$MAIN/experiments/results/q1
COMMIT=$(git -C $MAIN rev-parse HEAD)
SEED_GROUPS=("42,7" "123,1" "2024,13" "21,99" "2025,314")  # the last group holds the last seed
N=${#SEED_GROUPS[@]}
log() { echo "[$(date -u +%F' '%T)] $*"; }

pids=()
for i in "${!SEED_GROUPS[@]}"; do
  W=$PAR/w$((i + N))
  [ -d $W ] || git -C $MAIN worktree add --detach $W $COMMIT >/dev/null
  rm -rf $W/experiments/results/q1 && cp -r $Q1 $W/experiments/results/q1
  cp $MAIN/workflow_config.json $W/
  ln -sf $MAIN/datasets/scada_dataset_V01.csv $W/datasets/scada_dataset_V01.csv
  (cd $W && PYTHONPATH=$W $PY -u $PAR/issue567_worker.py ${SEED_GROUPS[$i]} > $W/worker567.log 2>&1) &
  pids+=($!)
  log "Issues 5-7 worker w$((i + N)) started, seeds ${SEED_GROUPS[$i]}"
done
fail=0
for i in "${!pids[@]}"; do
  wait ${pids[$i]} || { log "Issues 5-7 worker w$((i + N)) FAILED (see $PAR/w$((i + N))/worker567.log)"; fail=1; }
done
[ $fail = 0 ] || exit 1
log "Issues 5-7 workers finished"

log "waiting for the Issue 4 workers"
while pgrep -u rana -f "issue4_worker.py" >/dev/null; do sleep 30; done
for i in $(seq 0 $((N - 1))); do
  grep -q "^WORKER_DONE" $PAR/w$i/worker.log || { log "Issue 4 worker w$i did not finish (see $PAR/w$i/worker.log)"; exit 1; }
done
log "Issue 4 workers finished"

$PY $PAR/issue4_merge.py $Q1 $(for i in $(seq 0 $((N - 1))); do echo $PAR/w$i/experiments/results/q1; done) || exit 1
log "Issue 4 merged"
LAST=$PAR/w$((2 * N - 1))
$PY $PAR/issue567_merge.py $Q1 $LAST/experiments/results/q1 \
    $(for i in $(seq $N $((2 * N - 2))); do echo $PAR/w$i/experiments/results/q1; done) || exit 1
log "Issues 5-7 merged; building tables and Issue 7's aggregation in the main run"
cd $MAIN
for issue in 4 5 6 7; do
  $PY -u -m src.experiments.workflow --config workflow_config.json --issue $issue 2>&1 | tee -a train.log
  [ ${PIPESTATUS[0]} = 0 ] || { log "Issue $issue failed"; exit 1; }
done
log "ALL ISSUES DONE"

#!/bin/bash
# Investigation (1): score-misranking diagnostic.
# H2 configuration (pT5 building sees isDup T5s and isDup pLS) + new ntuple branches
# pT5_score / t5_score (FP16-rounded, exactly the values the dedup kernels compare).
# Output analyzed directly (no NumDen).
# Launch: setsid nohup bash t5tc_expH2_score.sh > t5tc_expH2_score.log 2>&1 < /dev/null & disown

set -eo pipefail

STANDALONE=/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone
cd "$STANDALONE"

if ! mkdir "$STANDALONE/.expH2score.lock" 2>/dev/null; then
    echo "Another instance is running (lock exists). Exiting."
    exit 1
fi
trap 'rmdir "$STANDALONE/.expH2score.lock"' EXIT

set +u; source setup.sh; cmsenv; source setup.sh; set -u

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Applying H1 + H2 (pT5 building sees isDup T5s and pLS)"
python3 t5tc_patch.py apply H1
python3 t5tc_patch.py apply H2

log "=== BUILD ==="
lst_make_tracklooper -G 2>&1

log "=== RUN ==="
lst_cuda -i trackingNtuple-100.root --allobj --jet --idealpls -n 100 \
    -o LSTNtuple_expH2_score.root 2>&1

log "Restoring H1 and H2"
python3 t5tc_patch.py restore H1
python3 t5tc_patch.py restore H2

log "Experiment H2-score complete."

#!/bin/bash
# Experiment G: F1 (skip standalone-vs-pT5 dedup) + F2 (CrossCleanT5 vs-pT5 disabled)
# + G (RemoveDupQuintupletsBeforeTC: hit-overlap-only kill, no pure-geometric arms)
# Launch: setsid nohup bash t5tc_expG.sh > t5tc_expG.log 2>&1 < /dev/null & disown

set -eo pipefail

STANDALONE=/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone
cd "$STANDALONE"

# single-instance guard
if ! mkdir "$STANDALONE/.expG.lock" 2>/dev/null; then
    echo "Another instance is running (lock exists). Exiting."
    exit 1
fi
trap 'rmdir "$STANDALONE/.expG.lock"' EXIT

set +u; source setup.sh; cmsenv; source setup.sh; set -u

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Applying F1, F2, G"
python3 t5tc_patch.py apply F1
python3 t5tc_patch.py apply F2
python3 t5tc_patch.py apply G

log "=== BUILD ==="
lst_make_tracklooper -G 2>&1

log "=== RUN ==="
lst_cuda -i trackingNtuple-100.root --allobj --jet --idealpls -n 100 \
    -o LSTNtuple_expG.root 2>&1

log "=== NUMDEN ==="
createPerfNumDenHists \
    -i LSTNtuple_expG.root \
    -o LSTNumDen_expG.root -J 2>&1

log "Restoring F1, F2, G"
python3 t5tc_patch.py restore F1
python3 t5tc_patch.py restore F2
python3 t5tc_patch.py restore G

log "Experiment G complete."

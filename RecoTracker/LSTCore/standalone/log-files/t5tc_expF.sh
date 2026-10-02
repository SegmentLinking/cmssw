#!/bin/bash
# Experiment F: skip T5-vs-pT5 dedup (F1: Kernels.h && → ||, F2: CrossCleanT5 disabled)
# Launch: setsid nohup bash t5tc_expF.sh > t5tc_expF.log 2>&1 < /dev/null & disown

set -eo pipefail

STANDALONE=/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone
cd "$STANDALONE"
set +u; source setup.sh; cmsenv; source setup.sh; set -u

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Applying F1 (Kernels.h && -> ||) and F2 (CrossCleanT5 vs-pT5 disabled)"
python3 t5tc_patch.py apply F1
python3 t5tc_patch.py apply F2

log "=== BUILD ==="
lst_make_tracklooper -G 2>&1

log "=== RUN ==="
lst_cuda -i trackingNtuple-100.root --allobj --jet --idealpls -n 100 \
    -o LSTNtuple_noT5pT5dedup.root 2>&1

log "=== NUMDEN ==="
createPerfNumDenHists \
    -i LSTNtuple_noT5pT5dedup.root \
    -o LSTNumDen_noT5pT5dedup.root -J 2>&1

log "Restoring F1 and F2"
python3 t5tc_patch.py restore F1
python3 t5tc_patch.py restore F2

log "Experiment F complete."

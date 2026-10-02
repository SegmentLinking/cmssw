#!/bin/bash
# Diagnostic run: Experiment F state (F1+F2, no D) with new ntuple branches
# t5_isDupBits / t5_tightCutFlag / t5_partOfPT5 / pT5_isDupReco
# for exact gate attribution of jet-core T5->TC failures.
# Launch: setsid nohup bash t5tc_expF_diag.sh > t5tc_expF_diag.log 2>&1 < /dev/null & disown

set -eo pipefail

STANDALONE=/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone
cd "$STANDALONE"

# single-instance guard (Session 21 saw a mysterious double launch)
if ! mkdir "$STANDALONE/.expF_diag.lock" 2>/dev/null; then
    echo "Another instance is running (lock exists). Exiting."
    exit 1
fi
trap 'rmdir "$STANDALONE/.expF_diag.lock"' EXIT
set +u; source setup.sh; cmsenv; source setup.sh; set -u

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Applying F1 (Kernels.h && -> ||) and F2 (CrossCleanT5 vs-pT5 disabled)"
python3 t5tc_patch.py apply F1
python3 t5tc_patch.py apply F2

log "=== BUILD ==="
lst_make_tracklooper -G 2>&1

log "=== RUN ==="
lst_cuda -i trackingNtuple-100.root --allobj --jet --idealpls -n 100 \
    -o LSTNtuple_expF_diag.root 2>&1

log "Restoring F1 and F2"
python3 t5tc_patch.py restore F1
python3 t5tc_patch.py restore F2

log "Experiment F diagnostic run complete."

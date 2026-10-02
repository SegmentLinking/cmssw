#!/bin/bash
# Experiment H: pT5-building visibility of dedup-flagged objects.
#   H1      : pT5 pairing sees isDup T5s   (PixelQuintuplet.h:680 skip removed)
#   H1+H2   : ... and isDup pLS too        (PixelQuintuplet.h:666 skip removed)
# All other code at production state (no F/G patches).
# See PLAN_expH_pt5_visibility.md for motivation and expected outcomes.
# Launch: setsid nohup bash t5tc_expH.sh > t5tc_expH.log 2>&1 < /dev/null & disown

set -eo pipefail

STANDALONE=/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone
cd "$STANDALONE"

if ! mkdir "$STANDALONE/.expH.lock" 2>/dev/null; then
    echo "Another instance is running (lock exists). Exiting."
    exit 1
fi
trap 'rmdir "$STANDALONE/.expH.lock"' EXIT

set +u; source setup.sh; cmsenv; source setup.sh; set -u

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_variant() {
    local tag=$1
    log "=== BUILD ($tag) ==="
    lst_make_tracklooper -G 2>&1
    log "=== RUN ($tag) ==="
    lst_cuda -i trackingNtuple-100.root --allobj --jet --idealpls -n 100 \
        -o "LSTNtuple_${tag}.root" 2>&1
    log "=== NUMDEN ($tag) ==="
    createPerfNumDenHists -i "LSTNtuple_${tag}.root" -o "LSTNumDen_${tag}.root" -J 2>&1
}

# ---- variant 1: H1 only ----
log "Applying H1 (pT5 building sees isDup T5s)"
python3 t5tc_patch.py apply H1
run_variant expH1

# ---- variant 2: H1 + H2 ----
log "Applying H2 (pT5 building sees isDup pLS too)"
python3 t5tc_patch.py apply H2
run_variant expH2

log "Restoring H1 and H2"
python3 t5tc_patch.py restore H1
python3 t5tc_patch.py restore H2

log "Experiment H complete."

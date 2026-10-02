#!/bin/bash
# Experiment H2 + combined (outward+inward) pT5 dedup score.
# Compares against existing LSTNumDen_expH2.root (H2 + old outward-only dedup).
# Output: LSTNumDen_expH2_combinedDedup.root, then eff_fake_vs_deltaR_combinedDedup.png
# Launch: setsid nohup bash t5tc_expH2_combinedDedup.sh > t5tc_expH2_combinedDedup.log 2>&1 < /dev/null & disown

set -eo pipefail

STANDALONE=/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone
cd "$STANDALONE"

if ! mkdir "$STANDALONE/.expH2combinedDedup.lock" 2>/dev/null; then
    echo "Another instance is running (lock exists). Exiting."
    exit 1
fi
trap 'rmdir "$STANDALONE/.expH2combinedDedup.lock"' EXIT

set +u; source setup.sh; cmsenv; source setup.sh; set -u

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Applying H1 + H2 (pT5 building sees isDup T5s and pLS)"
python3 t5tc_patch.py apply H1
python3 t5tc_patch.py apply H2

log "=== BUILD (CUDA, combined dedup score already in kernel) ==="
lst_make_tracklooper -G 2>&1

log "=== RUN ==="
lst_cuda -i trackingNtuple-100.root --allobj --jet --idealpls -n 100 \
    -o LSTNtuple_expH2_combinedDedup.root 2>&1

log "Restoring H1 and H2"
python3 t5tc_patch.py restore H1
python3 t5tc_patch.py restore H2

log "=== NumDen ==="
createPerfNumDenHists -i LSTNtuple_expH2_combinedDedup.root \
    -o LSTNumDen_expH2_combinedDedup.root 2>&1

log "=== Plot ==="
python3 efficiency/python/lst_plot_expH_eff_fake_combined.py 2>&1

log "Experiment H2-combinedDedup complete."

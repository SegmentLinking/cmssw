#!/bin/bash
# T5-TC diagnostic experiments: A (noCC_T5), B (nopT5priority), D (notightcut), ALL
# Launched detached: setsid nohup bash t5tc_experiments.sh > t5tc_experiments.log 2>&1 < /dev/null & disown

set -eo pipefail

STANDALONE=/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone

cd "$STANDALONE"
set +u; source setup.sh; cmsenv; source setup.sh; set -u

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

patch() {
    local action=$1 exp=$2
    python3 "${STANDALONE}/t5tc_patch.py" "$action" "$exp"
}

run_experiment() {
    local tag=$1
    log "=== BUILD: $tag ==="
    lst_make_tracklooper -G 2>&1

    log "=== RUN: $tag ==="
    lst_cuda -i trackingNtuple-100.root --allobj --jet --idealpls -n 100 \
        -o "LSTNtuple_${tag}.root" 2>&1

    log "=== NUMDEN: $tag ==="
    createPerfNumDenHists \
        -i "LSTNtuple_${tag}.root" \
        -o "LSTNumDen_${tag}.root" -J 2>&1

    log "=== DONE: $tag ==="
}

# ---- Experiment A: disable CrossCleanT5 vs-pT5 isDup assignment ----
log "Starting experiment A (noCC_T5)"
patch apply A
run_experiment noCC_T5
patch restore A
log "Restored after experiment A"

# ---- Experiment B: remove isPT5_jx/ix priority kill ----
log "Starting experiment B (nopT5priority)"
patch apply B
run_experiment nopT5priority
patch restore B
log "Restored after experiment B"

# ---- Experiment D: disable tightCutFlag gate ----
log "Starting experiment D (notightcut)"
patch apply D
run_experiment notightcut
patch restore D
log "Restored after experiment D"

# ---- Experiment ALL: A + B + D combined ----
log "Starting experiment ALL (t5tc_all)"
patch apply A
patch apply B
patch apply D
run_experiment t5tc_all
patch restore A
patch restore B
patch restore D
log "Restored after experiment ALL"

log "All experiments complete."

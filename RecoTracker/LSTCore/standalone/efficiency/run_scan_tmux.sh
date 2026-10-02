#!/usr/bin/env bash
#
# run_scan_tmux.sh — env-robust wrapper so tune_dedup_cuts.sh can run detached
# from the terminal (survives terminal/SSH closure). Does the FULL CMS bootstrap
# itself so it works from a bare environment (e.g. a systemd --user service),
# not relying on the interactive `cmsenv` alias being defined.
#
# Preferred launch (persistent user service; survives logout when Linger=yes):
#   systemd-run --user --unit=dedupscan bash efficiency/run_scan_tmux.sh oat7
#   Monitor:  journalctl --user -u dedupscan -f     (banner/warnings)
#             tail -f efficiency/tune_dedup_log.txt  (results, one row per point)
#   Status:   systemctl --user status dedupscan
#   Stop:     systemctl --user stop dedupscan
#
# Also works inside tmux:
#   tmux new-session -d -s dedupscan 'bash efficiency/run_scan_tmux.sh oat7'
#
# NOTE: no `set -u` — the CMS setup scripts (thisrooutil.sh) read unbound vars
# like LD_LIBRARY_PATH and would abort under nounset in a bare service env.
SA=/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"   # pre-init so setup.sh can append

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$SA" || exit 1
source setup.sh
eval "$(scram runtime -sh)"    # equivalent to cmsenv
source setup.sh
export LD_LIBRARY_PATH="/mnt/data1/kk829/cuda_driver_libs_580:${LD_LIBRARY_PATH:-}"

STAGE="${1:-oat7}"
echo "=== run_scan: stage=$STAGE NEVENTS=${NEVENTS:-100} started $(date -Is) ==="
bash "$SA/efficiency/tune_dedup_cuts.sh" "$STAGE"
echo "=== run_scan: stage=$STAGE finished $(date -Is) ==="

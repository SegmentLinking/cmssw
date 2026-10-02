#!/bin/bash
# S48: S44 base with T5 AfterBuild dedup effectively off (LST_AB_NMATCHED=11 > 10 hits per T5),
# every other cut at master defaults. 100 events, real pLS, --jet, 4 streams, CUDA.
# Binary: sparse worktree /mnt/data1/kk829/s44_worktree at 7a424adcef4 (claude-edits-backup = S44 code;
# its fix switches default off). Outputs go to this (main) standalone's Ntuple-files/ and NumDen-files/.
MAIN=/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone
WT=/mnt/data1/kk829/s44_worktree/RecoTracker/LSTCore/standalone

cd "$MAIN" && source setup.sh > /dev/null 2>&1
eval "$(cd /mnt/data1/kk829/CMSSW_16_1_1/src && scram runtime -sh 2> /dev/null)"  # = cmsenv
source setup.sh > /dev/null 2>&1                   # main tools (createPerfNumDenHists, python)
cd "$WT" && source setup.sh > /dev/null 2>&1       # worktree LST libs first on LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/mnt/data1/kk829/cuda_driver_libs_580:${LD_LIBRARY_PATH:-}"  # CUDA driver workaround
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"                                # L40
cd "$MAIN"

TAG=s48_ABoff
NTUPLE=Ntuple-files/LSTNtuple_${TAG}_100evt.root
NUMDEN=NumDen-files/LSTNumDen_${TAG}_100evt.root
RLOG=efficiency/.s44_runlogs/${TAG}_100evt.log
echo "$(date -Is) START $TAG  lst_cuda=$(command -v lst_cuda)"
t0=$SECONDS
env -u LST_PT5_HIGHPT_GATE -u LST_PLS_DISTINCT_HITS -u LST_PT5_SCORE_MODE -u LST_EXTEND_DUPT5 \
  LST_AB_NMATCHED=11 \
  timeout 21600 "$WT/bin/lst_cuda" -i trackingNtuple-100.root --jet -n 100 -s 4 -o "$NTUPLE" > "$RLOG" 2>&1
rc=$?
echo "$(date -Is) lst_cuda rc=$rc wall=$((SECONDS - t0))s"
[ $rc -ne 0 ] && exit $rc
"$WT/efficiency/bin/createPerfNumDenHists" -i "$NTUPLE" -o "$NUMDEN" -J >> "$RLOG" 2>&1
metrics=$(python3 efficiency/python/s44_eval_fixes.py "$NTUPLE" 2>> "$RLOG")
echo "$(date -Is)  tag=$TAG  nevents=100  env=LST_AB_NMATCHED=11  $metrics  wall_s=$((SECONDS - t0))" \
  >> efficiency/s44_fixes_log.txt
echo "$(date -Is) DONE $metrics"

# Comparison plots vs the S44 base (default LST style; the duplicate rate is in the *zoom plot)
cd "$MAIN" && source setup.sh > /dev/null 2>&1  # main lst_plot_performance.py first on PATH
python3 "$MAIN/efficiency/python/lst_plot_performance.py" NumDen-files/LSTNumDen_s44_base_100evt.root "$NUMDEN" \
  -L S44_base,S44_ABoff -t s44base_vs_ABoff_100evt --compare -j -m {eff,fakerate,duplrate} -o TC -s base \
  -v deltaR > efficiency/.s48_aboff_plot.log 2>&1
echo "$(date -Is) PLOTS rc=$? -> performance/s44base_vs_ABoff_100evt_*/mtv/var/"

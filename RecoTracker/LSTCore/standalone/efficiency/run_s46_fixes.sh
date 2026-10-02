#!/usr/bin/env bash
# Session 46: Fix A (high-pT pT5 pairing fallback) and Fix B (CheckHitspLS distinct hits) re-ported onto the
# rebased code (new T4/T5 DNN + PR277). 1000 evt, real pLS, --jet, no --allobj, -s 4 (same as S44 stage 2 / S45).
# s46_base (no env) must reproduce s45_rebased exactly. Resumable: tags already in the log are skipped.
cd /mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone || exit 1
source setup.sh > /dev/null 2>&1
eval "$(cd /mnt/data1/kk829/CMSSW_16_1_1/src && scram runtime -sh 2> /dev/null)"
source setup.sh > /dev/null 2>&1
export LD_LIBRARY_PATH="/mnt/data1/kk829/cuda_driver_libs_580:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
LOG=efficiency/s44_fixes_log.txt
NEV=1000

run() {  # run <tag> [ENV=VAL ...]
  local T=s46_$1; shift
  local ENVS="$*"
  if grep -q " tag=$T  nevents=$NEV " $LOG 2> /dev/null; then echo "skip $T"; return; fi
  local N=Ntuple-files/LSTNtuple_${T}_${NEV}evt.root D=NumDen-files/LSTNumDen_${T}_${NEV}evt.root
  local R=efficiency/.s44_runlogs/${T}_${NEV}evt.log t0=$SECONDS
  echo "$(date -Is) START $T env=${ENVS:-none}"
  env $ENVS timeout 7200 lst_cuda -i trackingNtuple-1000.root --jet -n $NEV -s 4 -o $N > $R 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "$(date -Is)  tag=$T  nevents=$NEV  env=${ENVS:-none}  RUN_FAILED(rc=$rc) (see $R)" >> $LOG
    echo "$(date -Is) FAILED $T rc=$rc"; return
  fi
  createPerfNumDenHists -i $N -o $D -J >> $R 2>&1 || echo "NumDen failed" >> $R
  local M=$(python3 efficiency/python/s44_eval_fixes.py $N 2>> $R)
  echo "$(date -Is)  tag=$T  nevents=$NEV  env=${ENVS:-none}  $M  wall_s=$((SECONDS - t0))" | tee -a $LOG
}

run base
run fixB  LST_PLS_DISTINCT_HITS=1
run fixA  LST_PT5_HIGHPT_GATE=1
run fixAB LST_PLS_DISTINCT_HITS=1 LST_PT5_HIGHPT_GATE=1
# determinism check: identical binary + settings as s46_base
run base_rep1
run base_rep2
echo "$(date -Is) ALL DONE"

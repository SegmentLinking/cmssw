#!/usr/bin/env bash
# Session 47: S47-1 (LST_PT5_DEMOTE_SCORE: pT5 with rPhiChi2 score > X written as T5 TC) and S47-2
# (LST_PLS_T5EMBED_SCALE: scales the CrossCleanpLS pLS-T5 embedding cut). 1000 evt, real pLS, --jet, -s 4.
# Compare with s46_base / _rep1 / _rep2 (run-to-run noise ~±5 core tracks). Resumable: tags already in the log are skipped.
cd /mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone || exit 1
source setup.sh > /dev/null 2>&1
eval "$(cd /mnt/data1/kk829/CMSSW_16_1_1/src && scram runtime -sh 2> /dev/null)"
source setup.sh > /dev/null 2>&1
export LD_LIBRARY_PATH="/mnt/data1/kk829/cuda_driver_libs_580:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
LOG=efficiency/s44_fixes_log.txt
NEV=1000

run() {  # run <tag> [ENV=VAL ...]
  local T=s47_$1; shift
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

run base                                    # default-off sanity check: must sit within noise of s46_base reps
run demote50  LST_PT5_DEMOTE_SCORE=50
run demote100 LST_PT5_DEMOTE_SCORE=100
run demote200 LST_PT5_DEMOTE_SCORE=200
run embed0    LST_PLS_T5EMBED_SCALE=0
run embed05   LST_PLS_T5EMBED_SCALE=0.5
# CrossCleanT4 keeps the pT5 (>=2 shared hits) rule for demoted pT5s
run demote50_t4fix  LST_PT5_DEMOTE_SCORE=50
run demote100_t4fix LST_PT5_DEMOTE_SCORE=100
# S47-3 (LST_PT5_UNSTALE) and S47-4 (LST_PT5_DEDUP_KEY), alone and stacked on demote50 + T4 fix
run unstale             LST_PT5_UNSTALE=1
run dedupkey1           LST_PT5_DEDUP_KEY=1
run demote50_unstale    LST_PT5_DEMOTE_SCORE=50 LST_PT5_UNSTALE=1
run demote50_unstale_k1 LST_PT5_DEMOTE_SCORE=50 LST_PT5_UNSTALE=1 LST_PT5_DEDUP_KEY=1
echo "$(date -Is) ALL DONE"

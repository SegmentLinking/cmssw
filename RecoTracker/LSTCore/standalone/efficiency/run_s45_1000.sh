#!/usr/bin/env bash
# Session 45: 1000-event run of the rebased code (new T4/T5 DNN), same settings as S44 stage 2
# (real pLS, --jet, no --allobj, -s 4). Result line appended to efficiency/s44_fixes_log.txt.
cd /mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone || exit 1
source setup.sh > /dev/null 2>&1
eval "$(cd /mnt/data1/kk829/CMSSW_16_1_1/src && scram runtime -sh 2> /dev/null)"
source setup.sh > /dev/null 2>&1
export LD_LIBRARY_PATH="/mnt/data1/kk829/cuda_driver_libs_580:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
T=s45_rebased
N=Ntuple-files/LSTNtuple_${T}_1000evt.root
D=NumDen-files/LSTNumDen_${T}_1000evt.root
R=efficiency/.s44_runlogs/${T}_1000evt.log
LOG=efficiency/s44_fixes_log.txt
t0=$SECONDS
echo "$(date -Is) START lst_cuda"
timeout 43200 lst_cuda -i trackingNtuple-1000.root --jet -n 1000 -s 4 -o $N > $R 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "$(date -Is)  tag=$T  nevents=1000  env=none  RUN_FAILED(rc=$rc) (see $R)" >> $LOG
  echo "$(date -Is) FAILED rc=$rc"; exit 1
fi
echo "$(date -Is) lst_cuda done ($((SECONDS - t0)) s); NumDen"
createPerfNumDenHists -i $N -o $D -J >> $R 2>&1 || echo "NumDen failed" >> $R
M=$(python3 efficiency/python/s44_eval_fixes.py $N 2>> $R)
echo "$(date -Is)  tag=$T  nevents=1000  env=none  $M  wall_s=$((SECONDS - t0))" | tee -a $LOG
echo "$(date -Is) ALL DONE"

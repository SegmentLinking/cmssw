#!/usr/bin/env bash
#
# run_s44_fixes.sh — Session 44 validation runs for the LST-internal jet-core fixes.
#
# One binary (lst_make_tracklooper -G) carries every fix behind a runtime env switch,
# all defaulting to master behaviour:
#   Fix A  LST_PT5_HIGHPT_GATE=1     high-pT pT5 pairing fallback (pixel hits vs T5 circle)
#          LST_PT5_HIGHPT_MINPT      pLS pT threshold in GeV (default 50)
#          LST_PT5_HIGHPT_MAXRES     max RMS pixel-to-T5-circle residual in cm (default 0.03)
#   Fix B  LST_PLS_DISTINCT_HITS=1   CheckHitspLS counts each shared pixel hit once
#   Fix C  LST_PT5_SCORE_MODE=1      pT5 dedup ranks by K5 = (pT5 score + T5 score_rphisum) / T5 dnnScore
#   Fix D  LST_EXTEND_DUPT5=0        skip ExtendTrackCandidatesFromDupT5
#
# Real pixel seeds (no --idealpls), --jet. Stage 1: 100 events with --allobj, one run per fix
# plus A+C and all-combined. Stage 2: 1000 events (no --allobj), baseline vs all-combined.
# Each run: lst_cuda -> createPerfNumDenHists -J -> s44_eval_fixes.py -> one log line.
# Resumable: a tag already logged with metrics (or a timeout/failure) is skipped on relaunch.
#
# Launch detached (survives closing the Claude session / terminal):
#   cd <standalone> && setsid nohup bash efficiency/run_s44_fixes.sh > efficiency/.s44_runner.log 2>&1 < /dev/null &
# Monitor:
#   cat efficiency/s44_fixes_log.txt
#
STANDALONE="/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone"
cd "$STANDALONE" || { echo "cannot cd to $STANDALONE"; exit 1; }

# Environment: setup.sh + CMSSW runtime (cmsenv is an alias, unavailable in a detached shell).
# Done before `set -u`: setup.sh reads unset variables (e.g. LD_LIBRARY_PATH in thisrooutil.sh).
source setup.sh > /dev/null 2>&1
eval "$(cd /mnt/data1/kk829/CMSSW_16_1_1/src && scram runtime -sh 2> /dev/null)"
source setup.sh > /dev/null 2>&1
export LD_LIBRARY_PATH="/mnt/data1/kk829/cuda_driver_libs_580:${LD_LIBRARY_PATH:-}"  # CUDA driver workaround
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"                                # L40

set -u

for exe in lst_cuda createPerfNumDenHists python3; do
  command -v "$exe" > /dev/null 2>&1 || { echo "ERROR: $exe not on PATH"; exit 1; }
done

LOG="efficiency/s44_fixes_log.txt"
STREAMS=4
mkdir -p Ntuple-files NumDen-files efficiency/.s44_runlogs
[ -f "$LOG" ] || echo "# run_s44_fixes.sh — cols: timestamp tag nevents env metrics wall_s  (binary: $(git rev-parse --short HEAD) + uncommitted fix switches)" > "$LOG"

ENV_A="LST_PT5_HIGHPT_GATE=1"
ENV_B="LST_PLS_DISTINCT_HITS=1"
ENV_C="LST_PT5_SCORE_MODE=1"
ENV_D="LST_EXTEND_DUPT5=0"

already_done() {  # <tag> <nevents>
  grep -F " tag=$1 " "$LOG" 2> /dev/null | grep -F " nevents=$2 " | grep -qE "dr002_eff=|RUN_TIMEOUT|RUN_FAILED"
}

# run_one <tag> <nevents> <timeout_s> <allobj:0|1> [ENV=VAL ...]
run_one() {
  local tag="$1" nev="$2" tmo="$3" allobj="$4"
  shift 4
  if already_done "$tag" "$nev"; then
    echo "SKIP already-done: $tag nevents=$nev"
    return 0
  fi
  local input="trackingNtuple-100.root"
  [ "$nev" -gt 100 ] && input="trackingNtuple-1000.root"
  local ntuple="Ntuple-files/LSTNtuple_s44_${tag}_${nev}evt.root"
  local numden="NumDen-files/LSTNumDen_s44_${tag}_${nev}evt.root"
  local rlog="efficiency/.s44_runlogs/${tag}_${nev}evt.log"
  local envstr="${*:-none}"
  local extra=()
  [ "$allobj" = 1 ] && extra+=(--allobj)

  echo "$(date -Is) START $tag nevents=$nev env=$envstr"
  local t0=$SECONDS
  # env -u clears every fix switch first so only the listed ones apply.
  env -u LST_PT5_HIGHPT_GATE -u LST_PLS_DISTINCT_HITS -u LST_PT5_SCORE_MODE -u LST_EXTEND_DUPT5 "$@" \
    timeout "$tmo" lst_cuda -i "$input" --jet "${extra[@]}" -n "$nev" -s "$STREAMS" -o "$ntuple" > "$rlog" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    local reason="RUN_FAILED(rc=$rc)"
    [ $rc -eq 124 ] && reason="RUN_TIMEOUT_${tmo}s"
    echo "$(date -Is)  tag=$tag  nevents=$nev  env=$envstr  $reason (see $rlog)" >> "$LOG"
    return 1
  fi
  createPerfNumDenHists -i "$ntuple" -o "$numden" -J >> "$rlog" 2>&1 || echo "NumDen failed for $tag" >> "$rlog"
  local metrics
  metrics=$(python3 efficiency/python/s44_eval_fixes.py "$ntuple" 2>> "$rlog")
  echo "$(date -Is)  tag=$tag  nevents=$nev  env=$envstr  $metrics  wall_s=$((SECONDS - t0))" >> "$LOG"
}

# ── Stage 1: 100 events, --allobj, one fix at a time, then combinations ─────────
T100=7200   # 2 h per run (normal ~20 min; GPU is shared with other users)
run_one base    100 $T100 1
run_one fixB    100 $T100 1 $ENV_B
run_one fixC    100 $T100 1 $ENV_C
run_one fixD    100 $T100 1 $ENV_D
run_one fixA    100 $T100 1 $ENV_A
run_one fixAC   100 $T100 1 $ENV_A $ENV_C
run_one fixABCD 100 $T100 1 $ENV_A $ENV_B $ENV_C $ENV_D

# ── Stage 2: 1000 events, no --allobj (TC-level metrics), baseline vs all fixes ─
T1000=43200  # 12 h per run
run_one base    1000 $T1000 0
run_one fixABCD 1000 $T1000 0 $ENV_A $ENV_B $ENV_C $ENV_D

echo "$(date -Is)  ALL DONE" >> "$LOG"
echo "$(date -Is) ALL DONE"

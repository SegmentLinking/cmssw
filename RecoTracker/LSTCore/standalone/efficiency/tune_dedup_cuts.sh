#!/usr/bin/env bash
#
# tune_dedup_cuts.sh — automated cut-tuning sweep for the T5/pT5 dedup kernels.
#
# Each test: export the dedup-cut env vars -> run LST (ideal pLS, --debug so
# debug.root is overwritten in place, no rm needed) -> evaluate core TC efficiency
# + fake rate from debug.root -> append one line to the log. The binary is built
# ONCE; every combination is just a re-run with different env vars.
#
# Objective (Pareto study): map core TC efficiency (lowest-ΔR bin) vs fake rate as
# the cuts are loosened, then pick the operating knee. No hard guardrail.
#
# Scope of the 7-parameter study (Δη/Δφ gate params are pinned at default):
#   LST_AB_NMATCHED    (raise -> fewer T5 AfterBuild removals -> higher eff)
#   LST_BTC_NMATCHED   (raise -> fewer T5 BeforeTC removals)
#   LST_PT5_NMATCHED   (raise -> fewer pT5 removals)
#   LST_BTC_DR2TIGHT   (lower -> fewer BeforeTC removals)
#   LST_BTC_DNND2LOOSE (lower -> fewer BeforeTC removals)
#   LST_BTC_DR2LOOSE   (lower -> fewer BeforeTC removals)
#   LST_BTC_DNND2TIGHT (lower -> fewer BeforeTC removals)
#
# Usage:
#   bash tune_dedup_cuts.sh baseline   # 1 run at master defaults
#   bash tune_dedup_cuts.sh oat7       # Stage 1: one-at-a-time over the 7 in-scope params
#   bash tune_dedup_cuts.sh frontier   # Stage 2: graded combined loosenings + 2-D grid
#   bash tune_dedup_cuts.sh wiring     # 2-run sanity test that the env vars are live
#   NEVENTS=1000 bash tune_dedup_cuts.sh frontier   # finalists at higher stats
#
#   (legacy stages, still present: oat / grid — these also scan the Δη/Δφ gates)
#
# Monitor with:  tail -f efficiency/tune_dedup_log.txt
#
set -u

STAGE="${1:-oat7}"

STANDALONE="/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone"
cd "$STANDALONE" || { echo "cannot cd to $STANDALONE"; exit 1; }

# CUDA driver workaround (required before lst_cuda).
export LD_LIBRARY_PATH="/mnt/data1/kk829/cuda_driver_libs_580:${LD_LIBRARY_PATH:-}"

# The launcher is expected to have sourced setup.sh + cmsenv so lst_cuda/python3 are on PATH.
if ! command -v lst_cuda >/dev/null 2>&1; then
  echo "ERROR: lst_cuda not on PATH. Launch this script from a shell that has run:"
  echo "  source setup.sh && cmsenv && source setup.sh"
  exit 1
fi

INPUT="trackingNtuple-100.root"
NEVENTS="${NEVENTS:-100}"        # overridable from the environment (finalists: NEVENTS=1000)
STREAMS="${STREAMS:-4}"
EVAL="python3 efficiency/python/eval_core_eff_fakerate.py debug.root"
LOG="efficiency/tune_dedup_log.txt"

# ── master-default cut values (baseline) ─────────────────────────────────────
BASE_AB_DETA=0.1;  BASE_AB_DPHI=0.1;  BASE_AB_NM=7
BASE_PT5_DETA=0.2; BASE_PT5_DPHI=0.2; BASE_PT5_NM=7
BASE_BTC_NM=5
BASE_BTC_DR2TIGHT=0.001
BASE_BTC_DNND2LOOSE=1.0
BASE_BTC_DR2LOOSE=0.02
BASE_BTC_DNND2TIGHT=0.1

# ── level sets for the legacy gate sweeps (oat/grid) ─────────────────────────
LEV_AB_DETA=(0.05 0.1 0.15 0.2)
LEV_AB_DPHI=(0.05 0.1 0.15 0.2)
LEV_PT5_DETA=(0.1 0.2 0.3)
LEV_PT5_DPHI=(0.1 0.2 0.3)

# ── level sets for the 7-parameter study (oat7) ──────────────────────────────
# Arrays sorted TIGHTEST→LOOSEST so that, if a run times out, we already have
# the safe tightening direction covered. "Tight" = fewer objects survive dedup.
#
# nMatched: LOWER = tighter (easier to call duplicate = more killed).
LEV_AB_NM=(6 8 9)                            # default 7; 10 DROPPED — causes 17-h blowup
LEV_BTC_NM=(4 6 7 8 10)                      # default 5
LEV_PT5_NM=(9 11 13 14)                      # default 7; all looser, but pT5 pool is small
# BeforeTC dR²/DNN-d² limits: HIGHER = tighter (removes more pairs).
LEV_BTC_DR2TIGHT=(0.002 0.0005 0.0)          # default 0.001; tight(0.002)→loose(0.0)
LEV_BTC_DNND2LOOSE=(2.0 0.5 0.25)            # default 1.0;   tight(2.0)→loose(0.25)
LEV_BTC_DR2LOOSE=(0.04 0.01 0.005)           # default 0.02;  tight(0.04)→loose(0.005)
LEV_BTC_DNND2TIGHT=(0.2 0.05 0.025)          # default 0.1;   tight(0.2)→loose(0.025)

# Stage-2 2-D grid axes (edit after inspecting oat7 to target the two most
# efficiency-sensitive params). Default: DNN-loose limit × BeforeTC nMatched.
GRID_DNND2LOOSE=(0.25 0.5 1.0)
GRID_BTC_NM=(5 7 9)

# Legacy grid (nMatched thresholds only).
GRID_AB_NM=(7 8 9 10)
GRID_PT5_NM=(7 9 11 13 14)

if [ ! -f "$LOG" ]; then
  echo "# tune_dedup_cuts.sh log — cols: timestamp stage label params eval_fields wall_s" > "$LOG"
fi

# already_done <stage> <label> — true if a successful OR timed-out row for this
# stage+label at the current NEVENTS is already in the log. Skips both completed
# points (have core010_eff=) and points that previously timed out (RUN_TIMEOUT).
already_done() {
  local stage="$1" label="$2"
  grep -F "stage=$stage " "$LOG" 2>/dev/null \
    | grep -F " label=$label " \
    | grep -F "nevents=$NEVENTS " \
    | grep -qE "core010_eff=|RUN_TIMEOUT"
}

# run_one <stage> <label> <ab_deta> <ab_dphi> <ab_nm> <pt5_deta> <pt5_dphi> <pt5_nm> \
#         [btc_nm] [btc_dr2t] [btc_d2l] [btc_dr2l] [btc_d2t]
# The 5 BTC params are optional; they default to baseline so the legacy stages
# (which pass only 8 args) run with BeforeTC at master defaults.
run_one() {
  local stage="$1" label="$2"

  if already_done "$stage" "$label"; then
    echo "SKIP already-done: stage=$stage label=$label nevents=$NEVENTS" >&2
    return 0
  fi

  local ab_deta="$3" ab_dphi="$4" ab_nm="$5"
  local pt5_deta="$6" pt5_dphi="$7" pt5_nm="$8"
  local btc_nm="${9:-$BASE_BTC_NM}"
  local btc_dr2t="${10:-$BASE_BTC_DR2TIGHT}"
  local btc_d2l="${11:-$BASE_BTC_DNND2LOOSE}"
  local btc_dr2l="${12:-$BASE_BTC_DR2LOOSE}"
  local btc_d2t="${13:-$BASE_BTC_DNND2TIGHT}"

  export LST_AB_DETA="$ab_deta"   LST_AB_DPHI="$ab_dphi"   LST_AB_NMATCHED="$ab_nm"
  export LST_PT5_DETA="$pt5_deta" LST_PT5_DPHI="$pt5_dphi" LST_PT5_NMATCHED="$pt5_nm"
  export LST_BTC_NMATCHED="$btc_nm" LST_BTC_DR2TIGHT="$btc_dr2t" LST_BTC_DNND2LOOSE="$btc_d2l" \
         LST_BTC_DR2LOOSE="$btc_dr2l" LST_BTC_DNND2TIGHT="$btc_d2t"

  local params="AB_deta=$ab_deta AB_dphi=$ab_dphi AB_nm=$ab_nm PT5_deta=$pt5_deta PT5_dphi=$pt5_dphi PT5_nm=$pt5_nm BTC_nm=$btc_nm BTC_dr2t=$btc_dr2t BTC_d2l=$btc_d2l BTC_dr2l=$btc_dr2l BTC_d2t=$btc_d2t nevents=$NEVENTS"

  local t0=$SECONDS
  # --debug -> debug.root opened RECREATE (overwritten in place); no --allobj (TC branches suffice).
  # 2400s (40 min) timeout: ~3× the expected ~13 min/pt, catches blowups like AB_NM=10.
  timeout 2400 lst_cuda -i "$INPUT" --jet --idealpls -n "$NEVENTS" -s "$STREAMS" --debug \
        > "efficiency/.tune_lastrun.log" 2>&1
  local run_exit=$?
  if [ $run_exit -ne 0 ]; then
    local reason="RUN_FAILED"
    [ "$run_exit" -eq 124 ] && reason="RUN_TIMEOUT_2400s"
    echo "$(date -Is)  stage=$stage  label=$label  $params  $reason (see efficiency/.tune_lastrun.log)" >> "$LOG"
    return 1
  fi
  local metrics
  metrics=$($EVAL 2>/dev/null)
  local wall=$((SECONDS - t0))
  echo "$(date -Is)  stage=$stage  label=$label  $params  $metrics  wall_s=$wall" >> "$LOG"
}

# run7 <stage> <label> <ab_nm> <btc_nm> <pt5_nm> <btc_dr2t> <btc_d2l> <btc_dr2l> <btc_d2t>
# Convenience wrapper for the 7-parameter study: sets the 7 in-scope params and
# pins the 4 Δη/Δφ gate params to baseline.
run7() {
  run_one "$1" "$2" \
    "$BASE_AB_DETA" "$BASE_AB_DPHI" "$3" \
    "$BASE_PT5_DETA" "$BASE_PT5_DPHI" "$5" \
    "$4" "$6" "$7" "$8" "$9"
}

# Convenience: the 7 in-scope params at baseline, as positional args for run7.
BASE7=("$BASE_AB_NM" "$BASE_BTC_NM" "$BASE_PT5_NM" "$BASE_BTC_DR2TIGHT" "$BASE_BTC_DNND2LOOSE" "$BASE_BTC_DR2LOOSE" "$BASE_BTC_DNND2TIGHT")

case "$STAGE" in
  baseline)
    run7 baseline master "${BASE7[@]}"
    ;;

  wiring)
    # Sanity test: a moderate loosening must raise n_tc / core010_eff vs baseline,
    # proving the env vars reach the kernels. (Do NOT zero both DNN d² terms here —
    # that fully disables BeforeTC dedup and the run blows up; see frontier SAFETY.)
    run7 wiring baseline "${BASE7[@]}"
    run7 wiring loosened "$BASE_AB_NM" "$BASE_BTC_NM" "$BASE_PT5_NM" "$BASE_BTC_DR2TIGHT" 0.25 "$BASE_BTC_DR2LOOSE" "$BASE_BTC_DNND2TIGHT"
    echo "$(date -Is)  stage=wiring  DONE (compare the two rows: n_tc/core010_eff must move)" >> "$LOG"
    ;;

  oat7)
    # One-at-a-time over the 7 in-scope params; others pinned to baseline.
    # Loop order: BTC params first (operate on ~40k post-AB T5s — small pool, safe),
    # then PT5_NM (pT5 pool is tiny), then AB_NM last (highest blowup risk: controls
    # how many of the full 1.58M T5s survive AfterBuild into downstream O(N²) stages).
    # Within each loop, arrays are sorted tightest→loosest so safe points run first.
    run7 oat7 base "${BASE7[@]}"
    for v in "${LEV_BTC_NM[@]}";       do [ "$v" = "$BASE_BTC_NM" ]       || run7 oat7 "BTC_NM=$v"    "$BASE_AB_NM" "$v" "$BASE_PT5_NM" "$BASE_BTC_DR2TIGHT" "$BASE_BTC_DNND2LOOSE" "$BASE_BTC_DR2LOOSE" "$BASE_BTC_DNND2TIGHT"; done
    for v in "${LEV_BTC_DR2TIGHT[@]}"; do [ "$v" = "$BASE_BTC_DR2TIGHT" ] || run7 oat7 "BTC_DR2T=$v"  "$BASE_AB_NM" "$BASE_BTC_NM" "$BASE_PT5_NM" "$v" "$BASE_BTC_DNND2LOOSE" "$BASE_BTC_DR2LOOSE" "$BASE_BTC_DNND2TIGHT"; done
    for v in "${LEV_BTC_DNND2LOOSE[@]}"; do [ "$v" = "$BASE_BTC_DNND2LOOSE" ] || run7 oat7 "BTC_D2L=$v" "$BASE_AB_NM" "$BASE_BTC_NM" "$BASE_PT5_NM" "$BASE_BTC_DR2TIGHT" "$v" "$BASE_BTC_DR2LOOSE" "$BASE_BTC_DNND2TIGHT"; done
    for v in "${LEV_BTC_DR2LOOSE[@]}"; do [ "$v" = "$BASE_BTC_DR2LOOSE" ] || run7 oat7 "BTC_DR2L=$v"  "$BASE_AB_NM" "$BASE_BTC_NM" "$BASE_PT5_NM" "$BASE_BTC_DR2TIGHT" "$BASE_BTC_DNND2LOOSE" "$v" "$BASE_BTC_DNND2TIGHT"; done
    for v in "${LEV_BTC_DNND2TIGHT[@]}"; do [ "$v" = "$BASE_BTC_DNND2TIGHT" ] || run7 oat7 "BTC_D2T=$v" "$BASE_AB_NM" "$BASE_BTC_NM" "$BASE_PT5_NM" "$BASE_BTC_DR2TIGHT" "$BASE_BTC_DNND2LOOSE" "$BASE_BTC_DR2LOOSE" "$v"; done
    for v in "${LEV_PT5_NM[@]}";       do [ "$v" = "$BASE_PT5_NM" ]       || run7 oat7 "PT5_NM=$v"    "$BASE_AB_NM" "$BASE_BTC_NM" "$v" "$BASE_BTC_DR2TIGHT" "$BASE_BTC_DNND2LOOSE" "$BASE_BTC_DR2LOOSE" "$BASE_BTC_DNND2TIGHT"; done
    for v in "${LEV_AB_NM[@]}";        do [ "$v" = "$BASE_AB_NM" ]        || run7 oat7 "AB_NM=$v"     "$v" "$BASE_BTC_NM" "$BASE_PT5_NM" "$BASE_BTC_DR2TIGHT" "$BASE_BTC_DNND2LOOSE" "$BASE_BTC_DR2LOOSE" "$BASE_BTC_DNND2TIGHT"; done
    echo "$(date -Is)  stage=oat7  DONE" >> "$LOG"
    ;;

  frontier)
    # Graded combined loosenings (mild -> medium -> strong).
    # SAFETY: never set BOTH BTC_D2L and BTC_D2T to 0 — that disables ALL BeforeTC
    # removals, so every T5 survives into the O(N²) cross-clean/TC-build kernels
    # and the run blows up (observed: >40 min / effectively hung). Keep D2L > 0.
    run7 frontier base   "${BASE7[@]}"
    run7 frontier mild    8 6  9 0.0005 0.5  0.01  0.05
    run7 frontier medium  9 7 11 0.0    0.35 0.008 0.04
    run7 frontier strong 10 8 13 0.0    0.25 0.005 0.025
    # 2-D grid over the two params flagged by oat7 (default: DNND2LOOSE × BTC_NM).
    for d2l in "${GRID_DNND2LOOSE[@]}"; do
      for bnm in "${GRID_BTC_NM[@]}"; do
        run7 frontier "grid:D2L=$d2l,BTC_NM=$bnm" \
          "$BASE_AB_NM" "$bnm" "$BASE_PT5_NM" "$BASE_BTC_DR2TIGHT" "$d2l" "$BASE_BTC_DR2LOOSE" "$BASE_BTC_DNND2TIGHT"
      done
    done
    echo "$(date -Is)  stage=frontier  DONE" >> "$LOG"
    ;;

  # ── legacy stages (also scan the Δη/Δφ gate params) ────────────────────────
  oat)
    run_one oat base "$BASE_AB_DETA" "$BASE_AB_DPHI" "$BASE_AB_NM" "$BASE_PT5_DETA" "$BASE_PT5_DPHI" "$BASE_PT5_NM"
    for v in "${LEV_AB_DETA[@]}";  do [ "$v" = "$BASE_AB_DETA" ] || run_one oat "AB_DETA=$v"   "$v" "$BASE_AB_DPHI" "$BASE_AB_NM" "$BASE_PT5_DETA" "$BASE_PT5_DPHI" "$BASE_PT5_NM"; done
    for v in "${LEV_AB_DPHI[@]}";  do [ "$v" = "$BASE_AB_DPHI" ] || run_one oat "AB_DPHI=$v"   "$BASE_AB_DETA" "$v" "$BASE_AB_NM" "$BASE_PT5_DETA" "$BASE_PT5_DPHI" "$BASE_PT5_NM"; done
    for v in "${LEV_AB_NM[@]}";    do [ "$v" = "$BASE_AB_NM" ]   || run_one oat "AB_NM=$v"     "$BASE_AB_DETA" "$BASE_AB_DPHI" "$v" "$BASE_PT5_DETA" "$BASE_PT5_DPHI" "$BASE_PT5_NM"; done
    for v in "${LEV_PT5_DETA[@]}"; do [ "$v" = "$BASE_PT5_DETA" ] || run_one oat "PT5_DETA=$v" "$BASE_AB_DETA" "$BASE_AB_DPHI" "$BASE_AB_NM" "$v" "$BASE_PT5_DPHI" "$BASE_PT5_NM"; done
    for v in "${LEV_PT5_DPHI[@]}"; do [ "$v" = "$BASE_PT5_DPHI" ] || run_one oat "PT5_DPHI=$v" "$BASE_AB_DETA" "$BASE_AB_DPHI" "$BASE_AB_NM" "$BASE_PT5_DETA" "$v" "$BASE_PT5_NM"; done
    for v in "${LEV_PT5_NM[@]}";   do [ "$v" = "$BASE_PT5_NM" ]   || run_one oat "PT5_NM=$v"   "$BASE_AB_DETA" "$BASE_AB_DPHI" "$BASE_AB_NM" "$BASE_PT5_DETA" "$BASE_PT5_DPHI" "$v"; done
    echo "$(date -Is)  stage=oat  DONE" >> "$LOG"
    ;;

  grid)
    for a in "${GRID_AB_NM[@]}"; do
      for p in "${GRID_PT5_NM[@]}"; do
        run_one grid "AB_NM=$a,PT5_NM=$p" "$BASE_AB_DETA" "$BASE_AB_DPHI" "$a" "$BASE_PT5_DETA" "$BASE_PT5_DPHI" "$p"
      done
    done
    echo "$(date -Is)  stage=grid  DONE" >> "$LOG"
    ;;

  *)
    echo "unknown stage '$STAGE' (use: baseline | wiring | oat7 | frontier | oat | grid)"; exit 1 ;;
esac

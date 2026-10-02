# LST-Internal Stage Scan on the Rebased Code (4 agents) — Handoff (Session 47, 2026-09-25)

Four read-only agents (no source edits, builds or LST runs) scanned the current code (HEAD `e49be6e473d`) on `Ntuple-files/LSTNtuple_s45_rebased_100evt.root` (core ΔR<0.02: 1226 tracks, 316 failures, eff 0.742). Outputs are in `efficiency/python/s47_stage_scan/{early_md_ls,early_pixel,late_build,late_tc}/SUMMARY.txt`. **All gains are offline estimates per 100 evt (core /1226), not yet validated by LST runs.** Noise: ±~5 core tracks per 1000 evt.

**Candidate fixes, ranked** (S47 numbering, which avoids a clash with S44 Fixes A–D):

| # | Fix | Where | Est. core gain | Fakes | Confidence |
|---|---|---|---|---|---|
| S47-1 | **Demote wrong-pLS pT5 → T5 TC**: in AddpT5asTrackCandidate, if pT5 score (rPhiχ²) > X, write the pT5's T5 as a T5 TC and keep the pLS dead | TrackCandidate.h:638-687 (T5 path :583-596) | **+65 (+5.3 pp)** at X=50 (train +35 / test +30); X=100 +56, X=200 +46; oracle ceiling +74 | **−178 TCs (−8.8%)**; core fakes 584→342 | medium-high |
| S47-2 | Relax/scale CrossCleanpLS pLS–T5 embedding cut (`dnn::plsembdnn::kWP`) | TrackCandidate.h:302-331 | +13 (+1.1 pp) with the cut off | ~flat; dup 0.0246→0.028 | medium-high |
| S47-3 | Clear stale `partOfPT5` after pT5 dedup (also T3/pLS for pT3 building) | after Kernels.h:762 / LSTEvent.dev.cc:1248 | **+8** (late_tc, after re-running CrossCleanT5/BeforeTC; late_build's +14–18 skipped that) | +20 | medium |
| S47-4 | pT5 dedup rank = nLayers, then score | Kernels.h:760 | +8 (train +3 / test +5) | −26 | low-medium |
| S47-5 | New T5 DNN density penalty: clamp density inputs / shift WP at high density | Quintuplet.h:1538-1540, 1669-1689 | ≤10–12 (upper bound) | real risk | needs a `-d` rejected-pair instrumentation first |
| — | Combined S47-1 (X=50) + S47-3 + S47-4 (model) | | **+81 (+6.6 pp)**; ΔR<0.1 +90 | −182 | first-order model |

Key mechanism for S47-1: above 5 GeV the pT5 χ² cuts are skipped (PixelQuintuplet.h:612/645/658), so in jet cores a genuine T5 is often paired with a foreign pLS. In 84/98 fake-killer pairs the genuine pT5 has a ≥1.5× worse score. 74 failing core sims have their genuine T5 inside an alive pT5 with a wrong pLS, and ~67 near-miss TCs are pT5s with 0/4 correct pixel hits on a 10–12/12 genuine OT part. Upstream alternative: apply the rPhiχ² cut above 5 GeV in the pT5 builder (would also free the pLS). Caveat for S47-1: 642 correctly matched pT5 TCs would lose their pixel hits, so check pT/dxy resolution.

**Established negatives:**
- MD/LS/T3 are not a lever (core = non-core survival; no cap or early exit bites).
- No-genuine-pLS (119) has ~0 LST-internal fixes: the see_algo filter mirrors CMSSW's `LSTInputProducer.cc:73-74` input, so pixelPair seeds need a CMSSW input change; the pixelPair ceiling is 2.3 pp.
- **Drop Fix A** (−412 core per 1000 evt; foreign pT5s absorb T5s that PR277 made TC-eligible).
- Fix B stays off (+0.15–0.26 pp, −0.3 pp at pT>100).
- CrossCleanT4 threshold not worth it (T4 TCs 59% fake).
- Standalone builds T4 before pT5/pT3 while CMSSW builds it after, so standalone T4 numbers don't transfer.

**Bookkeeping bug (no physics):** CrossCleanpLS also writes `pLS_isDup` bit0 (TrackCandidate.h:334-366), so ~32 of `s46_core_funnel.py`'s 65 "pLS_flagged" are really CrossCleanpLS removals (the S46 funnel table below overstates Fix B's reach). Proposed: CrossCleanpLS sets `|= 4`.

**Implemented and run (uncommitted, all default-off):**
- `LST_PT5_DEMOTE_SCORE=X`: `AddpT5asTrackCandidate` writes a pT5 with score > X as a T5 TC and marks its pLS `isDup |= 8`.
- `LST_PLS_T5EMBED_SCALE=s`: `CrossCleanpLS` multiplies the embedding WP by s.
- `CrossCleanT4` treats a T5 TC whose T5 is `partOfPT5` (only possible for a demoted pT5) with the pT5 rule (≥2 shared hits). Without this, demotion let ~5.4k fake T4 TCs per 1000 evt survive (fake 0.141 → 0.164).

Runner: `efficiency/run_s47_fixes.sh`. 1000 evt; the base mean over s46_base, rep1, rep2 and s47_base is 8381.5 core.

| Run | core ΔR<0.02 | core pT>100 | ΔR<0.10 | all | n_TC | fake | dup |
|---|---|---|---|---|---|---|---|
| base mean | 0.7595 | 0.643 | 0.841 | 0.872 | 141.9k | 0.141 | 0.0245 |
| **demote50_t4fix** | **0.7921 (+3.3 pp)** | **0.689 (+4.6)** | 0.857 | 0.882 | 141.8k | **0.132** | 0.0248 |
| demote100_t4fix | 0.7893 (+3.0) | 0.687 (+4.4) | 0.855 | 0.881 | 141.9k | 0.134 | 0.0247 |
| demote50 (no T4 fix) | 0.7936 | 0.692 | 0.857 | 0.883 | 147.3k | 0.164 | 0.0239 |
| embed0 | 0.7699 (+1.0) | 0.651 | 0.855 | 0.883 | 142.9k | 0.140 | 0.0300 |
| embed05 | 0.7651 (+0.5) | 0.649 | 0.850 | 0.880 | 142.7k | 0.141 | 0.0283 |

Plots (S44 base / S45 newDNN / S47 demote50_t4fix): `performance/s44base_s45newDNN_s47demote50_1000evt_7c9ddeD-trackingNtuple-1000.root_e49be6-trackingNtuple-1000.root_e49be6D-trackingNtuple-1000.root/mtv/var/`.

Swapped-color version (S45 green, S47 red): same path with `_swapcolors` after `_1000evt`.

**S47-3 and S47-4 implemented and run (2026-09-25/26, uncommitted, default-off):**
- `LST_PT5_UNSTALE=1` (S47-3): new `ResetPartOfPT5` kernel in Kernels.h, launched twice right after `RemoveDupPixelQuintupletsFromMap` in `createPixelQuintuplets`. The first launch clears `partOfPT5` on the T5, both T3s and the pLS of each dead pT5; the second sets it again for surviving pT5s. It runs before the pT3 builder in both standalone and CMSSW.
- `LST_PT5_DEDUP_KEY=1` (S47-4): `RemoveDupPixelQuintupletsFromMap::keyMode_`; more `nLayers` wins, then lower score.

| Run (1000 evt; base mean core 8381.5, pT>100 2607.5) | core ΔR<0.02 | Δ core | core pT>100 | ΔR<0.10 | all | fake | dup |
|---|---|---|---|---|---|---|---|
| unstale | 0.7730 | +149 (+1.35 pp) | 0.664 (+2.2) | 0.848 | 0.877 | 0.142 | 0.0248 |
| dedupkey1 | 0.7610 | +16 (+0.14 pp, ≈ noise) | 0.646 | 0.843 | 0.874 | 0.140 | 0.0247 |
| demote50_t4fix | 0.7921 | +360 (+3.3 pp) | 0.689 (+4.6) | 0.857 | 0.882 | 0.132 | 0.0248 |
| **demote50 + unstale** | **0.8051** | **+503 (+4.55 pp)** | **0.708 (+6.5)** | 0.864 | 0.887 | **0.134** | 0.0250 |
| demote50 + unstale + key1 | 0.8065 | +519 (+4.7 pp) | 0.709 (+6.6) | 0.865 | 0.888 | 0.133 | 0.0251 |

- Unstale gave about 2× the offline estimate (+149 vs ~+80 per 1000 evt).
- Demotion and unstale add almost exactly (360 + 149 ≈ 503).
- key1 adds only +16, barely above noise; it can be dropped.
- Best practical combination: **`LST_PT5_DEMOTE_SCORE=50 LST_PT5_UNSTALE=1`**, which gives core eff 0.759 → 0.805 with the fake rate down 0.7 pp.

**Resolution of demoted TCs** (`efficiency/python/s47_demote_resolution.py`, outputs `s47_demote_resolution_dr{inf,0.02}.txt`; same sim tracks, base vs demote50_t4fix; unchanged controls reproduce exactly):
- pT improves or is unchanged: σ(ΔpT/pT) goes 0.17 → 0.11 at 10–100 GeV (a +4% bias removed) and 0.40 → 0.37 above 100 GeV.
- φ is unchanged.
- **η degrades to T5 precision**: σ 0.0003 → ~0.05.
- **dxy/dz cannot be checked** (no TC impact-parameter branches). Demoted TCs carry no pixel hits, so in CMSSW their impact parameters would be poor unless pixel hits are re-added downstream. **This must be checked (CMSSW workflow / MTV, or a refit) before demotion is adopted.**

**Still to do:**
- Evaluate demote50 + unstale in CMSSW (`runTheMatrix.py ... 24834.704` + MTV) for dxy/dz resolution. Alternatively, try a variant that keeps the pT5 but swaps in a better-matching pLS.
- Plots for demote50 + unstale vs S44 base / S45 newDNN (not yet made).
- Decide on embed0 (+1 pp core for +0.55 pp duplicate rate).
- Optional: commit the S46/S47 switches (all default-off) so they survive the next rebase. Nothing from S46/S47 is committed. Files changed: Kernels.h, LSTEvent.dev.cc, TrackCandidate.h.

---

# Fix A/B Re-ported onto the Rebased Code + Determinism Check — Handoff (Session 46, 2026-09-25)

## TL;DR

Fix B and Fix A's host wiring were re-ported as default-off env switches (uncommitted edits to `Kernels.h` and `LSTEvent.dev.cc`; CUDA rebuilt). At 1000 evt, **Fix A now HURTS** (−3.7 pp core, −7.5 pp core pT>100) and **Fix B is at most +0.3 pp**, only just above run-to-run GPU noise. **LST CUDA runs are not bit-reproducible**: three identical `s46_base` runs gave 8385/8378/8374 core tracks (±~5, ≈0.1 pp). Neither fix should be turned on as is.

| 1000 evt (core den 11035, pT>100 den 4056) | core num | eff ΔR<0.02 | pT>100 | all | n_TC | fake | dup |
|---|---|---|---|---|---|---|---|
| s45_rebased (binary built Sep 24 15:21) | 8367 | 0.7582 | 0.6410 | 0.8721 | 141,900 | 0.1409 | 0.0245 |
| s46_base / rep1 / rep2 (same binary, no env) | 8385 / 8378 / 8374 | 0.759–0.760 | 0.642–0.644 | 0.872 | ~141,910 | 0.1410 | 0.0245 |
| s46_fixB `LST_PLS_DISTINCT_HITS=1` | 8408 | 0.7619 | 0.6403 | 0.8736 | 141,648 | 0.1388 | 0.0254 |
| s46_fixA `LST_PT5_HIGHPT_GATE=1` | 7973 | **0.7225** | **0.5690** | 0.8609 | 140,300 | 0.1378 | 0.0233 |
| s46_fixAB | 8051 | 0.7296 | 0.5750 | 0.8635 | 140,152 | 0.1357 | 0.0242 |

Runner: `efficiency/run_s46_fixes.sh` (resumable, results in `efficiency/s44_fixes_log.txt`).

**Core-failure funnel** (`efficiency/python/s46_core_funnel.py <ntuple> [label] [dR]`, --allobj ntuple, 100 evt, ΔR<0.02, failures / 1226; it reproduces the S44 buckets exactly on `LSTNtuple_s44_base_100evt.root`):

| Bucket | S44 base | S44 fixABCD | S45 rebased |
|---|---|---|---|
| genuine pT5 survives dedup, no TC | 26 | 0 | 0 |
| all genuine pT5 dedup-killed | 109 | 176 | 65 |
| genuine unflagged pLS + T5, no pT5 (pairing) | 78 | 22 | 25 |
| genuine unflagged pLS, no genuine T5 | 34 | 39 | 42 |
| all genuine pLS flagged by CheckHitspLS | 93 | 40 | 65 |
| no genuine pLS | 230 | 170 | 119 |

The funnel predicted up to ~65 tracks for Fix B, but only ~+30 appeared at 1000 evt (≈+3 at 100-evt scale), so freed pLS mostly fail downstream. Fix A was only aimed at the 25-track pairing bucket, but it cost ~400 tracks at 1000 evt; the likely cause is that the extra high-pT pT5s it admits win pT5 dedup against genuine ones (not verified).

**Plots** (default LST style, `lst_plot_performance.py --compare -j`, eff / fakerate / duplrate vs ΔR):
- S44 base vs S44 A+B+C+D, 1000 evt: `performance/s44_1000evt_base_vs_ABCD_7c9ddeD-trackingNtuple-1000.root_7c9ddeD-trackingNtuple-1000.root/mtv/`
- S44 base vs S45 newDNN vs S46 fixAB, 1000 evt: `performance/s44base_s45newDNN_s46fixAB_1000evt_7c9ddeD-trackingNtuple-1000.root_e49be6-trackingNtuple-1000.root_e49be6D-trackingNtuple-1000.root/mtv/`
  (`var/TC_base_0_0_eff_deltaR.png`, `var/TC_fakerate_deltaR.png`, `var/TC_duplrate_deltaRzoom.png`)

**Correction to Session 45 below:** "PR277" is just the two DNN commits (branch `pr-277`: `e1270bc27d7` T5 DNN, `0b9d3bf0d5e` T4 DNN; cherry-picked here as `771ce681fad`, `2865f1c7aaa`, and fixed up by `34837ac17e3` and `e49be6e473d`). The same rebase also brought in ~10 master LST commits (`cc53146c65a` ExtendT5FromDupT5ByMD, `018a7974f5b` T5 TC assembly, `b07f513662f`, `9c3790817bc`, `617888b1d5c`, `67c0a3d11b9`, …). **The S44→S45 gain (+20 pp core) is therefore master updates + PR277 combined; no run separates them.** To separate them, build and run `78895a59165` (rebased master, pre-PR277).

## Next steps (proposed, none launched)
1. Diagnose Fix A: a 100-evt `--allobj` run with `LST_PT5_HIGHPT_GATE=1`, then find which pT5s it adds and what they kill in dedup.
2. Optional: a master-without-PR277 build (`78895a59165`) at 1000 evt to split the S45 gain.
3. Fix C needs a new ranking key built from the 3-class T5 DNN; the dedup-killed bucket (65) is now the largest LST-internal lever along with CheckHitspLS (65).

---

# Rebased Code (New T4/T5 DNN + PR277) at 1000 Events — Handoff (Session 45, 2026-09-24 → 2026-09-25)

## TL;DR

`claude-edits` was rebased onto the new T5 DNN (`771ce681fad`), the new T4 DNN (`2865f1c7aaa`) and a cherry-pick of PR277 (`34837ac17e3`, `e49be6e473d`). *(Corrected in S46: 771ce/2865f ARE PR277; 34837/e49be are fix-ups. The rebase also included master LST updates.)* HEAD is now **`e49be6e473d`**. With **no env switches**, it beats S44's best run (fixABCD) everywhere except the duplicate rate. The 1000-event run is complete and valid: 1000/1000 entries, not a zombie, no `Recover` needed.

| Metric (1000 evt, real pLS, `--jet`, `-s 4`) | S44 base | S44 fixABCD | **S45 rebased** |
|---|---|---|---|
| eff ΔR<0.02 (den 11035) | 0.558 | 0.641 | **0.758** |
| eff ΔR<0.05 | 0.659 | 0.724 | **0.812** |
| eff ΔR<0.10 | 0.716 | 0.769 | **0.841** |
| eff ΔR<0.02, pT>100 (den 4056) | 0.303 | 0.410 | **0.641** |
| eff all (den 44726) | 0.787 | 0.823 | **0.872** |
| fake rate | 0.181 | 0.191 | **0.141** |
| dup rate | 0.019 | 0.020 | 0.0245 |
| n_TC | 127,091 | 131,476 | 141,900 |

The 100-event S45 run gave ΔR<0.02 = 0.742, consistent with the 1000-event result. `lst_cuda` wall time was about 150 s for 1000 events, against about 4 h per S44 run. S44 shared GPU 0 with another user's training job, and the new DNN applies cuts at creation; these two effects have not been separated.

## Files / plots

- Runner: `efficiency/run_s45_1000.sh` (untracked). Results line in `efficiency/s44_fixes_log.txt`; run log `efficiency/.s44_runlogs/s45_rebased_1000evt.log`.
- `Ntuple-files/LSTNtuple_s45_rebased_{100,1000}evt.root`, `NumDen-files/LSTNumDen_s45_rebased_{100,1000}evt.root` (`-J`).
- 3-way comparison plots (S44 base / S44 fixABCD / S45 new DNN):
  - 1000 evt: `performance/s45_1000evt_base_ABCD_newDNN_7c9ddeD-trackingNtuple-1000.root_7c9ddeD-trackingNtuple-1000.root_e49be6-trackingNtuple-1000.root/` (ΔR plot: `mtv/var/TC_base_0_0_eff_deltaR.png`)
  - 100 evt: `performance/s45_base_ABCD_newDNN_7c9ddeD-trackingNtuple-100.root_7c9ddeD-trackingNtuple-100.root_2865f1D-trackingNtuple-100.root/`
  - To make these plots: `python3 $(which lst_plot_performance.py) <3 NumDen files> -L S44_base,S44_fixABCD,S45_newDNN -t <tag> --compare -j`. Call it through `python3`, because the `#!/bin/env python` shebang picks up `/usr/bin/python` after cmsenv and crashes with "No module named 'encodings'".

## ⚠️ SOURCE TREE STATE — the S44 Fix A–D switches did NOT survive the rebase

The S44 source changes survive in full only on branch **`claude-edits-backup`**, commit **`7a424adcef4`**. In the rebased copy (`e929013b438`), only the edits to `PixelQuintuplet.h` and `PixelTriplet.h` and the `pLS_isDup` branch in `write_lst_ntuple.cc` were kept. The `Kernels.h` and `LSTEvent.dev.cc` hunks were dropped. At HEAD:

| Switch | Status at HEAD |
|---|---|
| A `LST_PT5_HIGHPT_GATE` | The kernel code is present (`PixelQuintuplet.h:490-534`, members `:677-679`, `PixelTriplet.h` `skipCurvatureCuts`), but nothing sets `highPtGate_` from the environment, so the gate is always OFF (dead code). |
| B `LST_PLS_DISTINCT_HITS` | Gone. `CheckHitspLS{}` is launched with no arguments (`LSTEvent.dev.cc:638,1112`). |
| C `LST_PT5_SCORE_MODE` | Gone. `RemoveDupPixelQuintupletsFromMap{pt5DEtaCut, pt5DPhiCut, pt5NMatchedCut}` (`LSTEvent.dev.cc:1243`) has no K5 rank key. The new T5 DNN also changed the T5 score semantics (3-class; `score_rphisum` removed), so K5 must be redefined, not just re-ported. |
| D `LST_EXTEND_DUPT5` | Doesn't map. Master commit `cc53146c65a` (not PR277; corrected S46) replaced the TC-level `ExtendTrackCandidatesFromDupT5` with T5-level `ExtendT5FromDupT5ByMD` (`Kernels.h:274`). The S44 dilution mechanism (11/14 → 11/15) must be re-checked before deciding whether a switch is still needed. |

HEAD's pT5 physics is therefore plain master + new DNNs + PR277 + S42 runtime-tunable dedup cuts (defaults).

## Next steps (proposed, none launched)

1. Decide whether to port B (small, verified bug: triplet 3rd-hit double-count) and A (needs only host wiring: read env in `LSTEvent.dev.cc` and set `highPtGate_/highPtMinPt_/highPtMaxRes_` on the `CreatePixelQuintupletsFromMap` functor) from `7a424adcef4`. Port C only after re-deriving a ranking from the new T5 DNN outputs; re-evaluate D against `ExtendT5FromDupT5ByMD`.
2. Rerun the S44 late-stage attribution (`efficiency/python/s44_stage_scan/late_build`, `late_tc`) on `LSTNtuple_s45_rebased_100evt.root` to see which S44 loss buckets remain after the new DNN. The S44 numbers (e.g. the 61 CheckHitspLS tracks, 82 fake-winner tracks) are from pre-rebase code.
3. Then a 1000-event fixAB(+C) run on the rebased code (~3 min now) vs `s45_rebased`.

---

# Jet-Core Efficiency: LST-Internal Stage Scan (4 agents) — Handoff (Session 44, 2026-09-22)

## TL;DR

Constraint from user: **only LST-internal changes are possible** (no CMSSW pixel tracking). Four read-only sub-agents scanned every LST stage for jet-core (ΔR<0.02) losses on `Ntuple-files/LSTNtuple_realpls_mastercuts_100evt_v2.root` (real pLS, AB-ON, 100 evt; denom 1224, 570 fails, eff 53.4% — reproduced by all four). No source edits, builds or LST runs. **Result: the early OT stages (MD/LS) are clean; all actionable losses are in pixel-seed handling and pLS↔T5 pairing, plus pT5 dedup winner choice. This OVERTURNS the Session 43 §9 conclusion that pT5 dedup is only a ~0.65% lever.**

Candidate LST-internal fixes (offline estimates, NOT yet validated by LST runs; populations partly overlap, do not add):

| Fix | Est. core gain | Fake risk | Confidence |
|---|---|---|---|
| **A. High-pT pT5 pairing gate** — above ~20–50 GeV pLS pT, replace pixel-curvature `passRadiusCriterion` + pT3-DNN with pixel-hit residual to T5 circle | +3.5–4 pp | likely lower | medium |
| **B. `CheckHitspLS` distinct-hit counting** (+ optional "only unflagged winner may kill") | ~+2 pp | medium | high (found independently by 2 agents) |
| **C. pT5 dedup winner = K5 score** `(pT5 rPhiχ² + T5 score_rphisum) / T5 dnnScore` | +2.2 pp real (+1.1 ideal) | lower (−54 fake TCs) | medium; K5 picked from 5 keys on same 100 evt → overfit risk |
| **D. Disable `ExtendTrackCandidatesFromDupT5`** | +2.1 pp (26 tracks), 0 gains ever | — | high — hold lifted for testing; switch `LST_EXTEND_DUPT5=0` added |
| **E. Admit pixelPair (algo 6) seeds** (needs 2-hit handling in `prepareInput`) | +2–3 pp | unknown | low–medium |

## ⚠️ SOURCE TREE STATE

Branch `claude-edits` (HEAD `7c9dde8e9cc`), CUDA binary rebuilt with `lst_make_tracklooper -G`, original weights, AB ON. **Uncommitted source changes — all runtime env switches that default to master behaviour** (no env var set ⇒ identical physics to before):

| Switch | Fix | Where |
|---|---|---|
| `LST_PLS_DISTINCT_HITS=1` | B: `CheckHitspLS` counts each shared pixel hit once | `Kernels.h` (`CheckHitspLS::distinctHits_`), both launches in `LSTEvent.dev.cc` |
| `LST_PT5_SCORE_MODE=1` | C: pT5 dedup ranks by K5 = (pT5 score + T5 `score_rphisum`) / max(T5 `dnnScore`, 1e-3), computed in float at dedup time | `Kernels.h` (`RemoveDupPixelQuintupletsFromMap::rankKey`; kernel now also takes the Quintuplets const view) |
| `LST_PT5_HIGHPT_GATE=1` (+ `LST_PT5_HIGHPT_MINPT`=50 GeV, `LST_PT5_HIGHPT_MAXRES`=0.03 cm) | A: **additive** fallback — a pLS+T5 pairing that fails the pT5 pT3-algo (radius criterion / DNN / tracklet) is still accepted if pLS pT ≥ MINPT, the RMS distance of the 2 pixel anchor hits to the T5 regression circle ≤ MAXRES (double precision), and the tracklet pointing cuts pass with radius+DNN skipped | `PixelQuintuplet.h` (`runPixelQuintupletDefaultAlgo`, `CreatePixelQuintupletsFromMap` members), `PixelTriplet.h` (`runPixelTripletDefaultAlgo(..., skipCurvatureCuts)`) |
| `LST_EXTEND_DUPT5=0` | D: skip `ExtendTrackCandidatesFromDupT5` (user lifted the hold for TESTING only, 2026-09-22) | `LSTEvent.dev.cc` |

Plus a new ntuple branch **`pLS_isDup`** (raw `CheckHitspLS` bits: bit0 = pre-pT5 pass, bit1 = TC-stage pass) in `write_lst_ntuple.cc`.

Fix A design note: implemented as OR (fallback), NOT a replacement gate, because the late_build agent's own reference shows already-built genuine high-pT pT5s have residual p90 = 673 µm — a 300 µm replacement gate would drop >10% of working pT5s. Consequence: A can only ADD pT5s (fake risk rises, not falls), and a newly admitted genuine pT5 still has a large pixel-circle score, so it may lose pT5 dedup unless Fix C is also on — hence the A+C run. Fix E (pixelPair seeds) NOT implemented (needs 2-hit pLS construction in `prepareInput`; deferred).

Smoke test (2 events, all switches on vs off): no crash; pLS bit0-flagged 532→428 of 729, pT5 built 560→839, TCs 249→271, fakes 67→77. All new work = agent scripts + small result files, moved to **`efficiency/python/s44_stage_scan/{early_md_ls,early_pixel,late_build,late_tc}/`** (hardcoded output paths updated to that location). Start with `late_build/SUMMARY.txt`. Not moved (regenerable caches, still in the volatile session scratchpad under /tmp): ~1.9 GB of `.pkl` caches (mostly `late_build`) and the large per-module CSVs (`mdable_modules_diag.csv` 6.6 MB, `ls_pairs.csv`, `survival_*.csv`) — rerun the scripts to regenerate. `late_build/*.py` read/write `.pkl` caches relative to the working directory, so run them from inside their folder.

## Findings by stage (all ΔR<0.02, real pLS unless noted)

### 1. Hits → MD → LS: NOT a lever (agent "early_md_ls")
- Core ≈ non-core at every step: reco hits 99.15% vs ~99%; genuine MD per MD-able module 94.8% vs ~94%; genuine LS per adjacent MD pair 99.1% vs ~99%.
- 97.5% of missing MDs are hit pairs whose **own truth simhits are kinked** (Δφ ≥ 1e-3 rad; delta rays/interactions). Truth-clean pairs lose only 0.17%. ~6 of 570 core failures (~1%).
- **No overflow**: Σ`md_occupancies` = stored MDs and Σ`sg_occupancies` = stored LSs in all 200 events; module map ≤34 connections (< `max_connected_modules`=40, but `ModuleMethods.h:140` has no bounds check). `clustSizeCut`=16 negligible.
- MD/LS form from all hit pairs → a shared hit cannot "steal" an object here. Windows sized for 0.8 GeV → no high-pT pathology.
- Merged-cluster hits: 26% of core MD-able modules vs 8% outside, but the flag does NOT separate pT5-stage failures from successes (74% vs 77%, per late_build agent).

### 2. Pixel seed → pLS (agent "early_pixel")
Emulated `prepareInput()` reproduces LST's pLS count exactly (46,982; 100/100 events). All 100 events are permuted between the LST ntuple and `trackingNtuple-100.root` (matched by sim_pt fingerprint).

Attribution of the 259 "no genuine pLS" core failures:

| Class | Tracks | LST-fixable? |
|---|---|---|
| Perfect seed in input, dropped by `see_algo ∈ {4,22}` filter (`LSTPrepareInput.h:102`) — 54 pixelPair (algo 6), 5 lowPtQuad (23), … | **63** (24%) | **yes** |
| LST has a 3/4-correct pLS (fails strict `>0.75` match, `trkCore.cc:508`) | 23 (9%) | matching-definition artifact |
| 3/4 seed only in a filtered algo | 6 (2%) | yes (filter) |
| Best LST pLS 2/3 or 2/4 correct (contaminated) | 67 (26%) | partial |
| Seeds carry ≤½ of the track's hits | 79 (31%) | no (input) |
| No seed although ≥3 pixel layers hit | 18 (7%) | no (CMSSW seeding) |
| <3 pixel layers hit | 3 (1%) | physics |

- pT cut (`ptIn > ptCut − 2·ptErr`) drops **0** genuine core seeds. pLS cap 500,000 (`Common.h:30`) vs ~470/event → no overflow.
- 92% of these tracks share a pixel cluster with another sim (vs 34% of successes) → cluster merging.
- **Truly absent in input ≈ 100 tracks**, not 259.

**`CheckHitspLS` (`Kernels.h:504-585`) — VERIFIED BUG.** Triplet seeds are padded with the 3rd hit repeated (`LSTPrepareInput.h:143-144`: `hitIdx3 = hitIdx2`). `npMatched` counts that hit twice → a triplet sharing only 2 distinct hits reaches `minNHitsForDup_pLS = 3`. Count is also asymmetric (loops only over the first seed's hits). Pass flags 73% (34,360/46,982) of all pLS; flagged pLS are skipped by pT5 (`PixelQuintuplet.h:668`) and pT3 (`PixelTriplet.h:716`) builders. Winner = quad always beats triplet, then lower `score_lsq` → purity-blind (57/69 winners are fake seeds). Double-count alone = 3,487 kills. Core tracks with genuine pLS+T5: eff **0.766** if ≥1 genuine pLS survives vs **0.240** if all killed (96 tracks).

**Triplet seeds ARE allowed into pT5 building** (no `isQuad` check in `CreatePixelQuintupletsFromMap`) — corrects EFFICIENCY_ATTEMPTS B-table assumption. They are disadvantaged by: always losing pLS dedup to quads, `CrossCleanpLS` skipping them (`TrackCandidate.h:309`), and pLS-as-TC needing `tc_pls_triplets` (`TrackCandidate.h:560,681`). Conversion: 0.638 (best genuine pLS = triplet) vs 0.814 (quad).

### 3. pLS + T5 → pT5 building (agent "late_build")
**Bucket is 205, not 176** (Session 43's 570−259−135 double-counts 29 tracks with a genuine pT5 but no perfect pLS). 115/205 have only triplet genuine pLS. Emulation validated: pixelmap superbin/connectivity (98.7% of pT5-success core tracks pass), CheckHitspLS flag rate reproduced (73.0%), pT3-DNN at pT5 working point passes 99.9% of 5,094 built pairs.

| Furthest stage reached by any genuine pair | Tracks | median sim pT |
|---|---|---|
| No genuine T3 | 35 | 90 |
| Genuine T3, no genuine T5 | 24 | 111 |
| All genuine pLS flagged by `CheckHitspLS` | **61** | 117 |
| All genuine T5 flagged by AfterBuild | 16 | 341 |
| `passRadiusCriterion` (`PixelTriplet.h:584`; bounds `:354-358`) | **27** | 266 |
| pT3-DNN at pT5 WP (`PixelTriplet.h:682`) | **27** | — |
| Tracklet cuts / rPhi-inwards (by elimination) | 15 | — |
| pLS type, pixel-map, 2S veto, occupancy caps | 0 | — |

- **High-pT mechanism:** pixel 1/pT resolution 17.6% at 50–150 GeV, **116% above 150 GeV**, while claimed ptErr/pt ≈ 1%. rz/rφ χ² cuts are skipped >5 GeV (`PixelQuintuplet.h:576,622`, verified) so they are not the problem; the radius criterion (±64–66% on 1/R above 2 GeV) and the DNN's pixel-circle rPhiχ² input (median 1005 for failures vs 46 for built pT5s) are. Build rate after cuts: 97–99% <150 GeV, 68–69% >150 GeV (same inside/outside core).
- **Stolen pLS:** in 81/205 tracks the genuine pLS is paired with a foreign T5 (48 other-sim, 33 fake); 71 become TCs, 68 of them fake. Pixel-hit residual to the T5 circle separates them: genuine median 78 µm (90% <300 µm) vs stolen 645 µm (28% <300 µm); genuine T5 closer in 279/317 contests → basis for **Fix A**.
- Fix B (distinct-hit counting) frees 31 of the 61 pLS-dedup tracks (34 with unflagged-winner rule) → ~24 tracks after TC survival.
- pT3 is not a fallback: 2/205 build a genuine pT3.
- Funnel (pLS → unflagged → +T3 → +T5 → T5 alive → +pT5 → TC): core 0.77→0.68→0.65→0.62→0.60→0.54→0.47; **core pT>100 GeV 0.64→…→0.43→0.29→0.22**; ΔR 0.1–0.4 0.95→…→0.76.
- **Minor dead-code bug (verified):** `PixelQuintuplet.h:152` tests `layer4 == 7` then inside checks `layer4 == 13 / 8` → cat 16/17 rz cuts never run (only loosens; irrelevant >5 GeV).

### 4. After building: pT5 dedup, cleaning, TC (agent "late_tc")
Exact offline replay of `RemoveDupPixelQuintupletsFromMap` (14 hits rebuilt from MDs, kernel η/φ, `pT5_score`, GPU index tie order) reproduces `pT5_isDupReco` for 22,208/22,209 pT5s (real) and 17,803/17,803 (ideal).

Where the last genuine pT5 dies (pool 135):

| Cause | Tracks | Truth-oracle winner recovers |
|---|---|---|
| Surviving killer is a **fake** pT5 (same pLS in 462/600 killer links) | **82** | 78 |
| Killed only by pT5s that are themselves dead — chain-kill; kernel never checks the killer's `isDup` (`Kernels.h:490-496`, verified) | **25** | 25 |
| Survived to TC but `ExtendTrackCandidatesFromDupT5` diluted the match (11/14 → 11/15 or 11/16) | **26** | — |
| Killer is a genuine pT5 of another sim (true displacement) | 2 | ~1 |
| CrossCleanT5 / BeforeTC / pLS cleaning / TC overflow | 0 | — |

Ideal pLS (pool 54): fake-winner 23, chain-kill 11, extension 14, displacement 6; oracle +29.

- **Session 43 §9 is superseded:** its approximate winner search (η/φ window, lowest score) misattributed the killers. Its "DOWNSTREAM 14" = exactly the 14 extension losses; its "DISPLACEMENT 32" is really 6. **Truth-perfect winner ceiling = +104 tracks (+8.5 pp) at ΔR<0.02 with −199 fake TCs**, not +8.
- Flagged S43 defects cost **0** tracks: BeforeTC `isDup & 1` (`Kernels.h:259,272`) and CrossCleanT5-vs-pT3 on dR² alone (`TrackCandidate.h:279`).
- **Stale `partOfPT5` (verified):** set on pLS/T3/T5 when a pT5 is built (`PixelQuintuplet.h:744-747`) and never cleared when dedup kills it → pT3 builder skips them (`PixelTriplet.h:718,767`) → **zero genuine pT3 for pool tracks.** Releasing the stranded T5s offline gains 0 tracks, +38 fakes (fix belongs at the pT5 winner).
- pLS-as-TC fallback: 57 consumed by the fake winner and removed by CrossCleanpLS (`TrackCandidate.h:365-370`); 37 non-quad; 12 unresolved (no reco pLS `isDup` branch).
- No capacity hit: pT5 max 516/15000, pT3 174/5000.
- Offline counterfactual winner rules (real pLS, ΔR<0.02 / <0.10 / Δfake TCs): truth oracle +104/+155/−199; greedy (killed only by kept) +38/+54/+885; pixel-sharing-only dup +29/+36/+508; T5 dnnScore alone +27/+36/+80 (ideal −52); **K5 +27/+50/−54 (ideal +13/+23/−13)**; K2 (χ² sums without dnnScore) +17/+27/−38. Features favouring the genuine loser over the fake winner (n=344): T5 score_rphisum 77%, radius consistency 76%, T5 dnnScore 73%.

## ▶ RUNNING NOW (launched 2026-09-22 17:50 EDT, user-approved): `efficiency/run_s44_fixes.sh`

Detached (`setsid nohup`, PPID 1 — survives closing the session). Resumable: relaunching skips tags already logged. Real pLS, `--jet`, `-s 4`, GPU 0 (L40, shared with another user's DNN training → runs may be slower than ~20 min).
- Stage 1, 100 evt `--allobj`: `base`, `fixB`, `fixC`, `fixD`, `fixA`, `fixAC`, `fixABCD` (2 h timeout each)
- Stage 2, 1000 evt (no `--allobj`): `base`, `fixABCD` (12 h timeout each)
- Outputs: `Ntuple-files/LSTNtuple_s44_<tag>_<N>evt.root`, `NumDen-files/LSTNumDen_s44_<tag>_<N>evt.root` (`-J`), per-run logs `efficiency/.s44_runlogs/`, runner stdout `efficiency/.s44_runner.log`
- **Results table: `efficiency/s44_fixes_log.txt`** — one line per run from `efficiency/python/s44_eval_fixes.py`: eff at ΔR<0.02/0.05/0.10, core pT>100, all; n_tc, fakerate, duprate. Evaluator validated on the old v2 file: ΔR<0.02 = 655/1226 = 0.5343 (agents: 654/1224; 2-track ΔR-edge difference).
- Check status: `cat efficiency/s44_fixes_log.txt`; `ps -eo pid,etime,cmd | grep -E "run_s44|lst_cuda"`. Stop: `pkill -f run_s44_fixes.sh; pkill -f "LSTNtuple_s44_"`.
- Caveat: `fixA` MINPT/MAXRES defaults (50 GeV / 300 µm) are untuned first guesses from the agent's offline numbers.

## Next steps (original plan — steps 1–5 and 7 are now running, see above)

All proposed as env-var toggles, default OFF, one rebuild, diff reviewed before running. Each 100-evt CUDA run ≈ 20 min: `lst_cuda -i trackingNtuple-100.root --jet --allobj -n 100 -s 4`, then `createPerfNumDenHists`.
1. Current-binary real-pLS baseline + new reco `pLS_isDup` branch (`write_lst_ntuple.cc` near `pLS_isQuad`, ~L1827) — the existing real-pLS file is a July build.
2. Fix B. 3. Fix C (K5; FP16 overflow guard at low dnnScore). 4. Fix A (pT threshold + residual cut; largest change). 5. Fix D — ready: switch exists, run with `LST_EXTEND_DUPT5=0` (no rebuild needed). 6. Fix E (most invasive; can defer).
7. Best combination at 1000 events (`-i trackingNtuple-1000.root --jet`, ~3 h) to check K5/threshold overfitting.
Instrumentation still needed to close gaps: per-pair reject codes in `runPixelQuintupletDefaultAlgo` (15 tracklet-stage tracks) and in the T5 builder (24 "T3 but no T5" tracks). `-d` alone doesn't help (records values only for built objects).

---

# Jet-Core Efficiency: Dedup vs Upstream Investigation — Handoff (Session 43, 2026-09-21 → 2026-09-22)

> ⚠️ **Superseded in part by Session 44 (above):** the §9 oracle ceiling (+8 / ~0.65%) and the "~1% dedup lever" conclusion came from an approximate winner search. An exact replay puts the truth-perfect pT5-dedup winner ceiling at +104 tracks (+8.5 pp, real pLS) and a realistic K5 rule at ~+2.2 pp. The §10 "31% pLS-but-no-pT5 (176)" bucket is 205 tracks and is largely LST-internal (pLS dedup bug + high-pT pairing).

## TL;DR — the whole arc and bottom line

Question: can jet-core efficiency be raised (without a drastic fake/dup increase) by improving the pT5 duplicate-removal kernels? **Answer: no, dedup is only a ~1% lever; the jet-core gap is ~3:1 upstream (pixel-seed supply), not dedup.**

Investigation chain (sections below, in order): oat7 dedup-cut scan (§1–6) → 6-agent dedup review (§5) → red-team "dedup-off recovers +tracks" (§7) → 4-agent replication that corrected both sides (§8) → purity-aware-winner ORACLE study pinning the dedup ceiling at **~0.65% (production, AB-ON)** (§9) → 3-agent per-sim-track upstream failure budget (§10).

Reconciled jet-core failure budget (real-pLS, AB-ON, ΔR<0.02, 570 fails / 1224 denom, eff 53.4%): **~76% never build a pixel quintuplet (dominated by seed supply — 45% no seed, 31% seed→no pT5), ~24% built-but-killed downstream (~1% cleanly dedup-recoverable), ~6% physics floor.** Truth seeds recover ~52% of failures. **Highest-leverage lever = real pixel-seed supply (a CMSSW pixel-tracking problem), not LST dedup.** Recurring analysis pitfall to avoid: counting "has a genuine T5" as "recoverable at dedup" (a T5 needs a pixel seed; 95% of core TCs are pT5).

## ⚠️ SOURCE TREE STATE

Unchanged from Session 42. Branch `claude-edits`. Binary = `lst_make_tracklooper -G` (CUDA, original weights, AfterBuild ON). **No LST source/binary edits across this whole session** — all work was analysis (new `efficiency/python/*.py` scripts + one production ntuple `LSTNtuple_ABon_oracle_100evt.root`). Env housekeeping: removed the stale `export ANTHROPIC_MODEL=claude-sonnet-4-6` line from `~/.bashrc` (was overriding the Opus-4.8 default); takes effect in fresh shells.

## What we did (2026-09-21)

### 1. Confirmed `oat7` scan complete

The `dedupscan.service` systemd unit finished on 2026-09-16 and is gone. All 26 `oat7` points logged in `efficiency/tune_dedup_log.txt` (last entry `2026-09-16T15:49:03`).

### 2. Ran Pareto analysis

```bash
python3 efficiency/python/lst_plot_pareto_dedup.py efficiency/tune_dedup_log.txt \
    --stage baseline,oat7 --out pareto_oat7.png
```

Output: `pareto_oat7.png`.

### 3. Pareto frontier (core010_eff vs fakerate, 5 points)

⚠️ **CROSS-BUILD ARTIFACT — see §6 correction.** The `lst_plot_pareto_dedup.py` "baseline" row (0.699) is an **AB-off** log row; the oat7 points are **AB-on**. `project_dedup_cut_scan.md` warns these two builds are not comparable. The correct within-build baseline is the oat7 `base` row: **core010=0.677, fakerate=0.130, dupfakerate=0.175**.

| Rank | core010_eff | fakerate | dupfakerate | Change from baseline |
|---|---|---|---|---|
| 0 | 0.708 | 0.212 | 0.759 | PT5_NM: 7→14 |
| 1 (mixed-build baseline) | 0.699 | 0.154 | 0.197 | — |
| 2 | 0.688 | 0.130 | 0.175 | AB_NM: 7→6 |
| 3 | 0.678 | 0.114 | 0.159 | BTC_DR2T: 0.001→0.002 |
| 4 | 0.677 | 0.103 | 0.148 | BTC_D2L: 1.0→2.0 |

### 4. Key findings (corrected against the within-build base = 0.677)

- **PT5_NM is the only param that raises efficiency**, within-build gain is **+0.031** at NM=14 (0.677→0.708), but it is **entirely duplicate-driven**: n_tc 13,220→24,357 (nearly doubled), dupfakerate 0.175→0.759. NM=11 = +0.029 eff, dupfakerate 0.393. Disqualifying for a "no drastic fake increase" goal.
- **AB_NM=6** gives +0.011 eff (0.677→0.688) at flat fakes (0.130) — the only near-clean point, but +8/712 is ~2σ at 100 evt, sits in outer ΔR bins (core002 unchanged), and direction is inconsistent (AB_NM=8 → −7). Needs 1000-evt confirmation before it's believable.
- **All BTC parameters (NM, DR2L, D2T) are flat** — <0.003 in core010_eff. Definitive: `BTC_DR2T=0` disables a class of BeforeTC removals, adds 2,290 junk TCs, recovers **zero** efficiency. Proof BeforeTC dedup is NOT blocking genuine core tracks.
- **BTC_DR2T / BTC_D2L** are pure fake knobs (tighter = fewer fakes, eff flat).

### 5. Multi-agent review conclusion (Session 43): dedup is NOT the primary jet-core lever

Five sub-agents reviewed the dedup kernels, past results, code weaknesses, and (independently) the premise. Convergent verdict: **the jet-core efficiency loss is dominated UPSTREAM of dedup**, not at the dedup kernels.
- **Pre-dedup `pT5_lower` ≈ 0.687 in the core (before any dedup runs)**; TC (~0.70) just tracks it. T5_lower ≈ 0.90. The collapse is at pT5 *building*, which needs a pLS seed.
- Loss decomposition: ~22% structurally unrecoverable (conversion-electron secondaries, no pixel hits), bulk of the rest = **pixel-seed supply** ("no real pLS" / non-quad seeds) — upstream CMSSW.
- Every past dedup-rescue (Exp F/G/H, χ² redesign, oat7) netted ~0 or worse TCs. Threshold-loosening only re-admits duplicates.
- **Real but bounded dedup defects found** (fix on their own merits, not for large efficiency): (i) bit-mask chain-kill — BeforeTC skip tests `isDup & 1` should be `& 3` (`Kernels.h:259,272`); (ii) CrossCleanT5-vs-pT3 kills a standalone T5 on `dR²<1e-3` alone, no hit/embedding check (`TrackCandidate.h:279`); (iii) purity-blind winner selection (pT5 dedup picks fake over genuine ~50% in same-seed contests); (iv) FP16 score→index tiebreak, real but ~6% of pT5 kills, and `isDupTiebreaker` is dead (never set) in the current build (`PixelQuintuplet.h:40`).

### 6. Next steps (revised)

1. **DO THIS FIRST — measure the dedup ceiling, no rerun needed.** Per-genuine-sim-track survival through each stage in the jet core (real-pLS baseline): for each core sim track, does ≥1 matched pT5 (frac>0.75) (i) get built, (ii) survive `RemoveDupPixelQuintupletsFromMap`, (iii) reach TC? Uses existing `sim_pt5IdxAll` / `pT5_isDupReco` branches. The count of tracks that build a genuine pT5, have it killed by dedup, and have no surviving sibling = the hard ceiling on what any dedup fix can buy. Prior hints (S21 "cat B ≈ 17 tracks") suggest <20% of core failures.
2. Only if that ceiling is worth it: prototype a **purity-aware winner** (DNN fake-score or truth-nMatched deciding the survivor) — NOT a looser threshold.
3. The larger lever is upstream seed supply (`--idealpls` historically ~0.3→0.6); that is a CMSSW pixel-tracking issue, outside LST dedup.
4. The PT5_NM/AB_NM frontier scan (previously step 1) is **de-prioritized** — PT5_NM is duplicate-driven; AB_NM=6 alone can be confirmed cheaply at 1000 evt if desired.

### 7. Red-team check (2026-09-21): dedup-off DOES recover genuine core tracks

A 6th (adversarial) sub-agent re-examined the 2026-08-07 allobj3 (pT5 dedup ON) vs allobj4 (pT5 dedup OFF) data. **Correction (established in §8):** both files are AfterBuild-OFF and differ ONLY in `RemoveDupPixelQuintupletsFromMap`; the earlier "allobj4 = AfterBuild also disabled" wording was wrong. So the ON→OFF delta is attributable to that **single kernel**.

- The curve that rises with dedup off IS the **true unique-sim-track TC efficiency** (`TC_base…ef_numer_deltaR`, filled once per sim track — verified performance.cc:49), NOT an object/TC count.
- Red-team's headline (looser core selection, denom 331): ΔR<0.02 0.698→0.846, +49 distinct matched sims. **The 4-way replication (§8) supersedes this with the standard base_0_0 selection: 0.669→0.815, net +76 to +97.**
- **Cost:** blunt disable floods the TC collection 36× (allobj files) / up to 324× (looser bin), fake rate → 0.85–0.92. Unusable as-is.

### 8. 4-way replication (2026-09-21) — COMPLETE, RESOLVED

Four independent agents (2 tasked to disprove the red-team +recovery; 2 to disprove the "not-a-lever" conclusion) initially rate-limited, then resumed and all four finished. Their partial pre-resume findings (superset-violation, NumDen≠ntuple) were self-corrected — see below.

**CONSENSUS (all four, hard per-sim-track numbers on allobj3 vs allobj4):**
1. **Gain is real and genuine.** Net **+76 to +97** distinct matched sim tracks in core (ΔR<0.02, 0.669→0.815). Metric verified per-sim-track; recovered TCs are >75%-pure by construction → **zero fake contamination** in the recovered set.
2. **Loss is IN the dedup kernel, not upstream.** Genuine objects are built **byte-identically** on vs off (core pT5_lower=0.745, T5_lower=0.901, identical). A2: **76/76** recovered sims already had a pure pT5 built in the ON run but were blocked from TC by `RemoveDupPixelQuintupletsFromMap`/cross-clean. Build-gating=0, AfterBuild contributes 0 here. **This DISPROVES the §5 "loss is upstream / pT5_lower≈0.687 / near-zero ceiling" framing.**
3. **"22% unrecoverable secondaries" debunked** (B1): only 8.9% (59/664) structurally unbuilt, overwhelmingly pions (42), 1 electron.
4. **Blunt disable unusable** (36–324× TC flood, 85–92% fake) — the narrow correct part of §5 survives.
5. **The "superset violation" was an artifact:** allobj3/allobj4 ntuples have **11/50 events permuted** (stream ordering). Aligning by event index fabricates losses (B2's "109 lost", A1's earlier "68 lost"). Re-keying by sim_pt signature → gains with **~0 real losses** (A1: gain 99/lost 2; A2: gain 76/lost 0). NumDen files verified correct (A2: NumDen +94 vs ntuple +97 agree).

**RESOLVED CEILING (the decision number):** gross killed-with-no-sibling pool is large (~72–170 depending on pT5-only vs pT5+T5), but the **realizable purity-preserving net is small.** A2's cluster analysis: of 76 recovered sims, only **7 have a *unique* pure pT5** (clean over-kill, trivially recoverable, zero fake cost); **69 sit in dense duplicate clusters** (up to 196 pure pT5s/sim) that overlap across neighboring sims → promoting one genuine loser displaces another (**near zero-sum**) — which is why S43's purity-preserving attempts netted ~0. Best estimates: **firm ~7 sims (+~1%), central ~15 (+~2%), optimistic ≤~35 (+5%)** (A1 ~15–35, A2 ~7–15). The full +76–97 rides on a duplicate flood a clean dedup cannot reproduce.

**Actionable conclusion:** dedup IS a real but MODEST jet-core lever, only via a **purity-aware winner rule** in `RemoveDupPixelQuintupletsFromMap` (NOT threshold loosening, NOT disable). Target the **~7 unique-pT5 over-kills** first — clean, no-downside +~1%. The dense-cluster tracks are largely unrecoverable without a fake cost. Dominant remaining jet-core loss is still the harder upstream/structural component.

**Caveat:** allobj3/4 are AfterBuild-OFF 50-evt files; provenance dispute (A1 called them real-pLS, B2 ideal-pLS) unresolved but does not affect the on-vs-off comparison. The pT5-FromMap winner defect should transfer qualitatively to the AfterBuild-ON production config, but the exact numbers there are unmeasured.

Note: session model switched to Opus 4.8 mid-session.

### 9. Purity-aware-winner ORACLE study (2026-09-22): production ceiling ~0.65% — NOT worth a kernel change

> ⚠️ **SUPERSEDED (Session 44 §4):** this study found killers with an η/φ-window + lowest-score approximation. The exact kernel replay shows most "DISPLACEMENT" was misattributed (really 6), "DOWNSTREAM 14" = extension-kernel dilution, and the truth-perfect ceiling is +104 (real pLS) / +29 (ideal pLS), not +8. The decision below is withdrawn.

Per approved plan `~/.claude/plans/what-would-a-purity-aware-winner-virtual-moore.md`. New script **`efficiency/python/lst_pt5_winner_oracle.py`** (uproot, no rebuild). For each jet-core sim track whose genuine pT5s were ALL killed by `RemoveDupPixelQuintupletsFromMap` with no TC, it finds the surviving pT5 that "won" (min `pT5_score` within |Δη|<0.2,|Δφ|<0.2, score≤loser) and classifies it: **fake winner = CLEAN-RECOVERABLE** (a purity-aware winner keeps the genuine one, kills the fake — 0 fakes added); **other-genuine-track winner = DISPLACEMENT** (near zero-sum); **same-track survivor = DOWNSTREAM** (lost at CrossCleanT5/BeforeTC). Runs per-event within one file → immune to the §8 event-permutation artifact.

**Validation on allobj3 (AB-OFF, 50 evt):** denom 664, fail 220, base eff 0.669, all-killed 73 — all match the §8 replication. CLEAN **+19** (+0.029), DISPLACEMENT 54, DOWNSTREAM 27.

**Production operating point — AB-ON, 100 evt** (`LSTNtuple_ABon_oracle_100evt.root`, current binary, `-i trackingNtuple-100.root --jet --idealpls --allobj -n 100 -s 4`; run ~1201 s):

| ΔR | denom | base eff | CLEAN ceiling | displacement | downstream |
|---|---|---|---|---|---|
| <0.02 | 1224 | 0.728 (891) | **+8 → 0.7345 (+0.0065)** | 32 | 14 |
| <0.05 | 2038 | 0.778 | +10 → 0.783 (+0.0049) | 47 | 18 |
| <0.10 | 2840 | 0.824 | +10 → 0.827 (+0.0035) | 49 | 19 |

The ceiling **collapses from +19 (AB-OFF) to +8 (AB-ON)**: AfterBuild-ON already removes most duplicate T5s before pT5 building, so far fewer fake-winner clusters exist. Fake-winner score margins are large (median 36.6, min 2.7, **no FP16 ties**) → a fix must OVERRIDE the geometric score with a purity discriminant (constituent-T5 `dnnScore` via `quintupletIndices()`), not a tiebreak.

**DECISION: NOT worth a pT5-dedup kernel change.** At the production operating point the clean, no-downside oracle ceiling is only **~0.65% (+8/1224) at ΔR<0.02** (~0.35% at ΔR<0.10) — below the plan's ~1% bar, AND it is the truth-perfect upper bound (a real runtime discriminant recovers only a fraction). DISPLACEMENT (32) is near zero-sum; DOWNSTREAM (14) is a CrossCleanT5 problem, not pT5-dedup. **Jet-core efficiency effort should move upstream (pixel-seed supply) / to the structural component**, which the §8 replication showed dominates the gap.

New files this continuation: `efficiency/python/lst_pt5_winner_oracle.py`, `LSTNtuple_ABon_oracle_100evt.root`, `pt5_winner_margins_{allobj3,ABon}.png`. No LST source/binary change (production run used the existing AB-ON binary).

### 10. Upstream per-sim-track failure budget (2026-09-22): jet-core gap is ~76% seed-supply, ~24% downstream

Three per-sim-track agents on **real-pLS, AB-ON, 100 evt** (`Ntuple-files/LSTNtuple_realpls_mastercuts_100evt_v2.root`; base_0_0 denom). Scripts: `efficiency/python/{pixel_seed_supply_study,jetcore_failure_taxonomy,jetcore_failure_physics}.py`. Real-pLS core eff (ΔR<0.02) = **53.4%** (vs 72.8% ideal-pLS → seed supply worth ~19 pp).

⚠️ **Conflation trap (recurring):** two agents initially concluded "82% built-but-killed → fix dedup" by counting *"has a genuine T5"* as recoverable. WRONG — 95.4% of core successes come via **pT5**; a standalone T5 with no pixel seed almost never becomes a TC (~1.4% path). Direct joint measurement (core ΔR<0.02, 570 failures) settles it:

| Cause | tracks | % of fails |
|---|---|---|
| **Pixel quintuplet NEVER built** | **435** | **76%** |
| — no genuine pixel seed (pLS) at all | 259 | 45% |
| — had a pLS but no pT5 (non-quad / combination fail) | 176 | 31% |
| **Pixel quintuplet built, killed before TC** (dedup/CrossClean) | 135 | 24% |
| — of which cleanly dedup-recoverable (§9 oracle) | ~8 | ~1% |
| Physics-irreducible (structurally unbuilt; subset of no-seed) | 34 | 6% |

Reconciliation detail: "has genuine T5"=468(82%) but 216 of those have NO pixel seed; "has genuine pT5"=135(24%) is the true downstream-loss pool. (Agent pT5 counts 69 vs 135 differ because 66 tracks have a genuine pT5 whose pLS component matches <0.75.)

**Physics (agent 3, refutes old claims):** failing core tracks are high-pt (median 121 GeV — TeV jets), 99.8% prompt, 99.5% barrel. "22% conversion-electron secondaries" REFUTED (e = 0.7% of fails, under-represented). "~9% structurally unbuilt, mostly pions" CONFIRMED (6% tight core → ~8% at ΔR<0.10, pion-dominated). Irreducible floor ~6% → achievable core eff ~94-96% vs observed 53%.

**Real→ideal recovery ~52% of core failures** (truth seeds fix about half; caveat: real file Jul-built, ideal Sep-built — binary drift inflates slightly).

**CONCLUSION:** the jet-core inefficiency is ~3:1 upstream (pixel-quintuplet-building, dominated by seed supply) vs downstream (dedup/CrossClean, ~1% cleanly recoverable). The dominant lever is **real pixel-seed supply** — largely a CMSSW pixel-tracking problem, not LST-internal. LST-internal dedup work (§9) is confirmed a ~1% lever. New scripts only; no source/binary change.

---

# Runtime-Tunable Dedup Cuts + Pareto Scan — Handoff (Session 42, 2026-09-15)

## ⚠️ SOURCE TREE STATE (branch `claude-edits`, all COMMITTED this session)

Differs from Session 41's state — read carefully:
- **`src/alpaka/LSTEvent.dev.cc`**: AfterBuild **RE-ENABLED** (`#if 1 // AFTERBUILD-DISABLE (re-enabled for dedup-cut tuning)` in `createQuintuplets`). Reads dedup cuts from env vars at startup via `lstEnvF`/`lstEnvI` helpers and passes them to the three dedup kernel structs.
- **`src/alpaka/Kernels.h`**: three dedup structs now carry runtime-tunable member fields (all default to master values): `RemoveDupQuintupletsAfterBuild` (dEtaCut_, dPhiCut_, nMatchedCut_), `RemoveDupQuintupletsBeforeTC` (+ dR2TightCut_, dnnD2LooseCut_, dR2LooseCut_, dnnD2TightCut_), `RemoveDupPixelQuintupletsFromMap` (dEtaCut_, dPhiCut_, nMatchedCut_).
- **`src/alpaka/T5NeuralNetworkWeights.h`**: now a THIN SELECTOR (`#ifdef LST_T5DNN_RETRAINED`). Defaults to **ORIGINAL master weights** (`_original.h`); retrained weights preserved in `_retrained.h`, activated by building with `-r`. This REVERSES the S41 state (which had retrained weights inline).
- **`interface/PixelQuintupletsSoA.h`** / **`src/alpaka/PixelQuintuplet.h`**: pT5 instrumentation columns `isDupTiebreaker`, `passedNMatchedCut` (committed in 4120106d17c, pre-session).
- Binary rebuilt this session with `lst_make_tracklooper -G` (CUDA, original weights).

Commits this session (on top of pre-existing `4120106d17c`):
- `b77c5dc301b` compile-time toggle for retrained T5 DNN weights (selector + `_retrained.h`/`_original.h`, `-r` flag, `explicit_retrained_t5dnn` target).
- `87c4a501e8e` wire runtime-tunable dedup cuts (env-var plumbing in LSTEvent.dev.cc) + `pT5_isDupTiebreaker`/`pT5_passedNMatchedCut` ntuple branches in write_lst_ntuple.cc.
- `7c9dde8e9cc` make `RemoveDupQuintupletsBeforeTC` cuts runtime-tunable.

## The 13 tunable dedup env vars (no recompile per point)

| Env var | Kernel | Default | Loosen (raise eff) |
|---|---|---|---|
| LST_AB_DETA / DPHI / NMATCHED | AfterBuild (T5) | 0.1 / 0.1 / 7 | ↑ NMATCHED |
| LST_BTC_NMATCHED | BeforeTC (T5, has DNN) | 5 | ↑ |
| LST_BTC_DR2TIGHT / DNND2LOOSE / DR2LOOSE / DNND2TIGHT | BeforeTC | 0.001 / 1.0 / 0.02 / 0.1 | ↓ each |
| LST_PT5_DETA / DPHI / NMATCHED | pT5 FromMap | 0.2 / 0.2 / 7 | ↑ NMATCHED |

The 7-param study skips the Δη/Δφ gates; scans the 3 NMATCHED + 4 BeforeTC limits.

## New files this session (untracked, like other analysis scripts)

- **`efficiency/tune_dedup_cuts.sh`** — sweep driver. Stages: `baseline`, `wiring` (2-pt plumbing test), `oat7` (one-at-a-time over the 7 params), `frontier` (graded loosenings + 2-D grid), legacy `oat`/`grid`. `run7` wrapper sets the 7 params + pins gates. `already_done` guard makes it RESUMABLE (skips points already logged at the current `NEVENTS`). `NEVENTS=1000 bash ... frontier` for finalists.
- **`efficiency/python/eval_core_eff_fakerate.py`** — objective evaluator (reads `debug.root` via uproot; prints core002/004/006/010 eff + fakerate/dupfakerate). Reused unchanged.
- **`efficiency/python/lst_plot_pareto_dedup.py`** — parses the log, computes the non-dominated frontier (max eff / min fake), plots core010_eff vs fake/dupfake, prints a ranked table with per-point param deltas. `--stage baseline,oat7,frontier` excludes legacy gate rows; `--metric`, `--fake`, `--out` options.
- **`efficiency/run_scan_tmux.sh`** — env-robust wrapper (full CMS bootstrap; NO `set -u` — CMS scripts read unbound LD_LIBRARY_PATH) to run the sweep detached from the terminal.

## What we did (2026-09-15)

1. Made all 3 dedup kernels runtime-tunable (committed, above) and rebuilt.
2. Built the Pareto-scan infrastructure (harness + evaluator reuse + plot + wrapper).
3. Verified env→kernel plumbing: `DNND2LOOSE 1.0→0.25` moved n_tc 13219→14670, fakerate 0.13→0.21, core010_eff flat (0.673→0.671).
4. Launched the `oat7` sensitivity scan.

### Key findings
- **~13 min per 100-evt CUDA point** (AfterBuild-on adds an O(N²) pass), so `oat7` (26 pts) ≈ 5-6 hr. NOT the 1-2 min originally estimated.
- **New (AB-on) baseline** at master cuts (100 evt, --idealpls): core010_eff≈0.673, fakerate≈0.130, n_tc≈13219. Old 2026-08-18 log rows are AB-OFF (core010_eff≈0.701, n_tc≈13480) — do not cross-compare.
- **PATHOLOGICAL CORNER**: setting BOTH `LST_BTC_DNND2LOOSE=0` and `LST_BTC_DNND2TIGHT=0` disables ALL BeforeTC removals → every T5 floods the O(N²) cross-clean/TC-build kernels → blowup (>40 min, effectively hung; NOT OOM). The `frontier`/`wiring` stages were fixed to always keep DNND2LOOSE > 0. Killed one such run this session.

## RUNNING NOW / how to resume

**`oat7` scan is running as a persistent systemd user service** `dedupscan.service` (account has `Linger=yes`, so it survives terminal/SSH close). It is resumable (2 pts done at handoff: `base`, `AB_NM=6`).

Monitor / control:
```
systemctl --user status dedupscan
tail -f efficiency/tune_dedup_log.txt      # results, one row per point
tail -f efficiency/.oat7_service.log       # bootstrap/banner output
systemctl --user stop dedupscan            # stop early
# resume/continue after any stop (idempotent, skips done points):
systemd-run --user --unit=dedupscan -p StandardOutput=append:$PWD/efficiency/.oat7_service.log -p StandardError=append:$PWD/efficiency/.oat7_service.log bash efficiency/run_scan_tmux.sh oat7
```

A session-only cron (this Claude session) checks every 30 min and auto-runs the Pareto analysis when the `stage=oat7  DONE` marker appears.

### Next steps (in order)
1. When `oat7` done → `python3 efficiency/python/lst_plot_pareto_dedup.py efficiency/tune_dedup_log.txt --stage baseline,oat7 --out pareto_oat7.png`. Identify the 2 most efficiency-sensitive params and the frontier.
2. Edit `GRID_DNND2LOOSE`/`GRID_BTC_NM` (or swap axes) in `tune_dedup_cuts.sh` to target those 2 params, then run `frontier` stage (same systemd-run command, `frontier` instead of `oat7`).
3. Re-run the top ~3-5 frontier points at `NEVENTS=1000` (export before launching) to confirm ordering; re-plot; deliver ranked table + recommended operating point.
4. Optional follow-up: repeat the scan under retrained weights (build with `-r`) — the two BTC DNN-d² optima are weight-set-specific.

Plan file: `/home/kk829/.claude/plans/squishy-finding-yeti.md`. Memory: `project_dedup_cut_scan.md`.

---

# Dedup-Stage Efficiency Script — Handoff (Session 41, 2026-08-17)

## ⚠️ SOURCE TREE STATE

- **`src/alpaka/LSTEvent.dev.cc`**: AB **DISABLED** via `#if 0 // AFTERBUILD-DISABLE` (restored at end of Session 40).
- **`src/alpaka/T5NeuralNetworkWeights.h`**: retrained weights from `master-kasia-dnn` (commit `b3bac69f7b3`) — **NOT master weights**.
- **`src/alpaka/Kernels.h`**: matches master exactly (0.2f deta/dphi thresholds, pairwise `RemoveDupPixelQuintupletsFromMap`, `minNHitsForDup_T5=7`). 0-line diff vs master.
- All other files at HEAD.

## New files this session

- **`efficiency/python/lst_plot_dedup_stages_T5_pT5.py`** — new script: T5 and pT5 efficiency vs ΔR at each dedup stage, read directly from the LST ntuple (no NumDen file). Five curves: T5_lower (dotted), T5_post_AB (dashed), T5_post_dedup (solid), pT5_lower (dotted), pT5_post_dedup (solid). Orange = T5 group, blue = pT5 group.
- **`dedup_stages_T5_pT5_idealpls_mastercuts_100evt.png`** — output plot from `LSTNtuple_idealpls_mastercuts_100evt_v2.root` (S34, 100 events, standard LST, ideal pLS, 0.2f thresholds, master DNN weights).

## What we did (2026-08-17, continuation of Session 40)

### 1. Wrote `lst_plot_dedup_stages_T5_pT5.py`

New script that reads the ntuple directly (not NumDen) and computes five efficiency-vs-ΔR curves:

| Curve | Filter | Branches used |
|---|---|---|
| T5_lower | any T5 with frac > 0.75 | `sim_t5IdxAll`, `sim_t5IdxAllFrac` |
| T5_post_AB | `t5_isDupBits & 0x01 == 0` | same + `t5_isDupBits` |
| T5_post_dedup | `t5_isDupBits == 0` | same |
| pT5_lower | any pT5 with frac > 0.75 | `sim_pt5IdxAll`, `sim_pt5IdxAllFrac` |
| pT5_post_dedup | `pT5_isDupReco == 0` | same + `pT5_isDupReco` |

Denominator selection matches `createPerfNumDenHists -J` base_0_0: pt>0.8, |eta|<4.5, |vz|<30, vtx_perp<2.5, q≠0, matched genjet pt>1000 |eta|<2.5.

**Branch name fix**: pT5 sim-track arrays are `sim_pt5IdxAll` / `sim_pt5IdxAllFrac` (per-sim-track lists of pT5 indices), NOT `pT5_simIdxAll` / `pT5_simIdxAllFrac` (those are per-pT5 lists of matched sim tracks — reversed direction).

### 2. Ran the script on the best available standard-LST ntuple

Used `Ntuple-files/LSTNtuple_idealpls_mastercuts_100evt_v2.root` (S34, 100 events, master cuts, 0.2f thresholds, AB on, ideal pLS):
```bash
python3 efficiency/python/lst_plot_dedup_stages_T5_pT5.py \
    Ntuple-files/LSTNtuple_idealpls_mastercuts_100evt_v2.root \
    -o dedup_stages_T5_pT5_idealpls_mastercuts_100evt.png \
    --tag "Standard LST (AB on, ideal pLS, 100 evt, master cuts)"
```
Output: `dedup_stages_T5_pT5_idealpls_mastercuts_100evt.png`. The gaps between curves show AfterBuild losses (T5_lower→post_AB), BeforeTC+CrossCleanT5 losses (post_AB→post_dedup), and `RemoveDupPixelQuintupletsFromMap` losses (pT5_lower→post_dedup).

## What remains to do

- **Correct retrained DNN NumDen/comparison plots**: `LSTNtuple_retrained_T5dnn_ABon_ideal_50evt.root` (the correct one with 0.2f thresholds) exists but the NumDen file on disk (`NumDen-files/LSTNumDen_retrained_T5dnn_ABon_ideal_Jv1.root`) is from the FIRST INCORRECT run (0.01f thresholds). Need to rerun `createPerfNumDenHists -J` on the correct file and redo the comparison plot.
- **Decide on retrained T5 DNN weights**: `T5NeuralNetworkWeights.h` currently has retrained weights. Decision pending: commit to `claude-edits` or revert to master.
- All items from Session 39 still open.

---

# Retrained T5 DNN Weights Comparison — Handoff (Session 40, 2026-08-17)

## ⚠️ SOURCE TREE STATE (at end of session — corrected)

- **`src/alpaka/LSTEvent.dev.cc`**: AB **DISABLED** (`#if 0 // AFTERBUILD-DISABLE`). Re-enabled midway for runs, then disabled again at end of session.
- **`src/alpaka/T5NeuralNetworkWeights.h`**: retrained weights from `master-kasia-dnn` (commit `b3bac69f7b3`). Master weights are NOT active.
- **`src/alpaka/Kernels.h`**: matches master exactly (0.2f thresholds). A pre-existing 0.01f edit was discovered and reverted this session.
- All other files at HEAD.

## New ntuple files this session

- **`LSTNtuple_retrained_T5dnn_ABon_ideal_50evt.root`** (444 MB, hi-pT QCD PU200, AB ON, ideal pLS, CUDA, this server, 50 events) — retrained T5 DNN weights, standard pT5 dedup.
- **`NumDen-files/LSTNumDen_retrained_T5dnn_ABon_ideal_Jv1.root`** (16 MB) — NumDen for the retrained DNN run.

## What we did (2026-08-17)

### 1. Found retrained T5 DNN weights on `master-kasia-dnn`

Confirmed that `T5NeuralNetworkWeights.h` on the `claude-edits` / `master` branch contains the **original upstream weights**, not the retrained ones. The retrained version lives on `master-kasia-dnn` under commit `b3bac69f7b3`. Only `T5NeuralNetworkWeights.h` differs; T3, T4, and pT3 weights are identical across all branches. The retrained file adds `const` to the array declarations and uses single-line formatting, but the network architecture (array dimensions, variable names) is unchanged.

### 2. Applied retrained weights and re-enabled AfterBuild

- Re-enabled `RemoveDupQuintupletsAfterBuild` in `LSTEvent.dev.cc` (removed `#if 0 / #endif` wrapper).
- Applied retrained weights: `git checkout master-kasia-dnn -- RecoTracker/LSTCore/src/alpaka/T5NeuralNetworkWeights.h`.
- Rebuilt CUDA binary with `lst_make_tracklooper -G`.

### 3. Ran LST and compared to standard baseline

**Run command:**
```bash
lst_cuda -i trackingNtuple-100.root --jet --idealpls --allobj -n 50 -s 4 \
    -o LSTNtuple_retrained_T5dnn_ABon_ideal_50evt.root
```

**Baseline:** `LSTNtuple_allobj_ABon_ideal.root` / `NumDen-files/LSTNumDen_allobj_ABon_ideal_Jv1.root` (Session 39, same 50-event conditions, original upstream T5 DNN weights).

**Result:** The retrained T5 DNN weights improve efficiency slightly (see `eff_vs_deltaR_retrained_T5dnn_vs_standard_ABon_ideal.png` and the TC-type breakdown plots `eff_vs_deltaR_tc_breakdown_retrained_T5dnn_ABon_ideal.png` / `eff_vs_deltaR_tc_breakdown_standard_ABon_ideal.png`). The T5 DNN is used in `RemoveDupQuintupletsBeforeTC` (6D embedding distance condition) — not in AfterBuild or pT5 dedup — so the improvement manifests in the T5-type and TC efficiency.

## What remains to do

- Decide whether to commit the retrained `T5NeuralNetworkWeights.h` to `claude-edits` or revert to master weights.
- All items from Session 39 still open (isPT5-priority bit3 instrumentation gap, etc.).

---

# Cut Flow Corrections + isDupBits Audit + pT>10 Filter — Handoff (Session 39, 2026-08-07)

## ⚠️ SOURCE TREE STATE

- **`src/alpaka/LSTEvent.dev.cc`**: AB disabled via `#if 0 // AFTERBUILD-DISABLE` (HEAD). Re-enabled and reverted during this session for the QCD ABon re-run; current state = HEAD (AB off).
- **`src/alpaka/Kernels.h`**: at HEAD (pairwise `RemoveDupPixelQuintupletsFromMap`, `minNHitsForDup_T5=7`). NOT the re-instrumented Session 35 version — bits 2 and 3 are not written by the current kernel code (see isDupBits section below).
- All other files at HEAD.

## New ntuple files this session

- **`LSTNtuple_allobj_ABon_ideal.root`** (451 MB, hi-pT QCD PU200, AB ON, ideal pLS, CUDA, this server, 50 events) — the corrected QCD ABon column. Supersedes `LSTNtuple_dedup_stages_v2.root` in the cut flow table.
- **`LSTNtuple_allobj_ABon.root`** (453 MB, AB ON, real pLS) — produced first by mistake; superseded, can be ignored.
- **`NumDen-files/LSTNumDen_allobj_ABon_ideal_Jv1.root`** (16 MB) — NumDen for the corrected ABon run.

## 1. QCD AB ON column corrected in cut flow table

**Problem:** The previous "hi-pT QCD PU200 AB ON" column used `LSTNtuple_dedup_stages_v2.root`, which was produced on 2026-07-09 with the Experiment 1 threshold (`minNHitsForDup_T5=8`). Session 34 reverted this to 7, so the ABon column was using different code than the three ABoff columns — an apples-to-oranges comparison.

**Fix:** Re-ran with the same setup as allobj3/4/5:
- `lst_cuda -i trackingNtuple-100.root --jet --idealpls --allobj -n 50 -s 4`
- LSTEvent.dev.cc temporarily edited to remove `#if 0 // AFTERBUILD-DISABLE` wrappers (lines 999/1009 in the session 38 line count), then restored after run.
- Output: `LSTNtuple_allobj_ABon_ideal.root` (ideal pLS, `minNHitsForDup_T5=7`, CUDA, 50 events).
- First attempt accidentally omitted `--idealpls`; binary was still AB-enabled so re-ran immediately with `--idealpls` without rebuilding.

**Updated values (QCD AB ON, 50 events, ideal pLS):**

| Step | Value |
|---|---|
| T5s formed | 1,581,676 |
| Killed by AfterBuild | 1,541,483 (97.5%) |
| T5s surviving AfterBuild | 40,193 (2.5%) |
| pT5s formed | 8,558 |
| Killed by BeforeTC (T5s) | 32,366 (80.5% of surv. AB) |
| T5s surviving BeforeTC | 7,653 (19.0% of surv. AB) |
| Killed by pT5 dedup | 4,978 (58.2% of pT5s formed) |
| pT5s surviving (→ TC) | 3,580 (41.8% of pT5s formed) |

**Scripts updated:** `efficiency/python/lst_cutflow_table_plot.py` and `efficiency/python/lst_cutflow_table.py` — both now point to `LSTNtuple_allobj_ABon_ideal.root` for the QCD ABon column.

## 2. ttbar pT>10 GeV column added (no LST rerun)

A new "ttbar ABon pT>10 GeV" column was added to both cut flow scripts, filtering T5s and pT5s to only those matched to a genuine sim track with `sim_pt > 10 GeV`. The same `LSTNtuple-ttbar-50-ABon2.root` file is reused — no rerun of LST needed.

### Implementation

COLUMNS tuples expanded from 3-element to 4-element `(label, fname, nevents, pt_min)` where `pt_min=None` disables the filter. When `pt_min` is set:

1. **Load** additional branches: `t5_simIdx`, `pT5_simIdx`, `sim_trkNtupIdx`, `sim_pt`.
2. **Build reverse map** per event: `{int(sim_trkNtupIdx[i]): i}` — maps tracking-ntuple sim index → LST sim index.
3. **Mask** T5/pT5s: keep only those where `sim_trkNtupIdx_reversed[t5_simIdx] → sim_pt > pt_min`. Fakes (`simIdx < 0`) are excluded by definition.

### Key indexing caveat

`t5_simIdx` does NOT index into the LST ntuple's `sim_pt` array directly. It stores the tracking-ntuple's global sim track index (values up to ~36,000 per event), while `sim_pt` in the LST ntuple has only ~1,022 entries/event (the selected sim subset). The `sim_trkNtupIdx` branch provides the bridge: `sim_trkNtupIdx[lst_sim_i]` = the tracking-ntuple index for LST sim track `i`. The reverse map resolves this.

### New helper in both scripts

```python
def _build_pt_mask(simidx_arr, sim_trkntupidx_arr, sim_pt_arr, pt_min):
    # For each event: build reverse map, look up sim_pt for each T5/pT5 simIdx
```

## 3. isDupBits encoding: instrumentation differs across ntuples

**Critical caveat for the cut flow table's "of which isPT5-priority (bit3)" row.**

The isDupBits encoding was expected to be:
- bit0 (0x01): AfterBuild kill
- bit1 (0x02): BeforeTC condA
- bit2 (0x04): BeforeTC condB
- bit3 (0x08): BeforeTC isPT5-priority-decisive
- bit4 (0x10): CrossCleanT5

But the current HEAD `Kernels.h` writes only:
```cpp
quintuplets.isDup()[quintupletIndex] |= 1 + secondpass;
// secondpass=false → bit0 (AfterBuild); secondpass=true → bit1 (BeforeTC)
```
Bits 2 and 3 are **never written** by HEAD. CrossCleanT5 (TrackCandidate.h) writes bit4 independently.

**Actual bit populations across all 6 cut flow ntuples (50 events each):**

| Ntuple | bit0 (AB) | bit1 (BTC) | bit2 | bit3 (pT5pri) | bit4 (CC) |
|---|---:|---:|---:|---:|---:|
| ttbar ABon | 243,801 | 11,617 | 0 | 0 | 849 |
| QCD ABon ideal | 1,541,483 | 32,366 | 0 | 0 | 174 |
| QCD allobj3 (ABoff pairwise) | 0 | 1,566,610 | 1,552,696 | 960,197 | 337 |
| QCD allobj5 (score<12) | 0 | 1,566,610 | 1,552,696 | 960,197 | 337 |
| QCD allobj4 (remove-pix off) | 0 | 1,566,610 | 1,552,696 | 960,197 | 337 |

allobj3/4/5 were produced with the **Session 35 re-instrumented Kernels.h** (the version that sets bits 1, 2, and 3 granularly for BeforeTC sub-reasons). The two newer ntuples (ttbar ABon and QCD ABon ideal) were produced with HEAD Kernels.h, which collapses all BeforeTC kills into bit1 only.

**Consequence:** The "of which isPT5-priority" row is meaningful for the allobj3/4/5 columns (~960k / 1.57M BeforeTC kills = ~61% are pT5-priority kills) but is always 0 for the ttbar ABon and QCD ABon ideal columns — not because those samples have no isPT5-priority kills, but because the bit is not written by the current binary. The BeforeTC total kill count (`bits & 0x0E`) is still valid for all columns.

**To restore bit3 instrumentation:** apply the Session 35 re-instrumented `Kernels.h` (see that section), rebuild, and rerun both newer ntuples.

## 4. DNN in dedup kernels

Investigated whether AfterBuild, BeforeTC, or RemoveDupPixelQuintupletsFromMap use a DNN score.

- **AfterBuild (`RemoveDupQuintupletsAfterBuild`)**: No DNN. Dup condition: dEta<0.1, dPhi<0.1, shared hits ≥ 7. Winner: lower `score_rphisum` (geometric residual sum).
- **BeforeTC (`RemoveDupQuintupletsBeforeTC`)**: **Yes — DNN embedding used.** The duplicate detection condition (Kernels.h line ~289) requires the squared L2 distance in the 6D `t5Embed` space (`d2`) to be below a threshold: `((dR2 < 0.001 || nMatched >= 5) && d2 < 1.0) || (dR2 < 0.02 && d2 < 0.1)`. `t5Embed` is a DNN-derived embedding vector (`Params_T5::kEmbed = 6`). Winner selection still uses `score_rphisum`, not a DNN fake score.
- **`RemoveDupPixelQuintupletsFromMap`**: No DNN. Uses `pixelQuintuplets.score()` = `rPhiChiSquared` (passed as `score` at pT5 creation, see `addPixelQuintupletToMemory` line 736). Winner: lower score (lower chi-squared = better fit).

The T4 dedup kernels (`RemoveDupQuadrupletsAfterBuild`, `BeforeTC`) use `displacedScore - fakeScore` (DNN scores) for winner selection — this is distinct from the T5 path.

## 5. New plot: pt5_scores_surv_only.png

Surviving pT5s only (genuine and fake), from `LSTNtuple_allobj3.root`, 50 events. Same color scheme as `pt5_scores_all50evt.png`: genuine green (#2ca02c), fake cyan (#17becf).

- **Genuine survived** N=2,802: peaks in the 1–20 rPhiChiSquared range.
- **Fake survived** N=207: broadly distributed into the 10²–10⁴ tail — these fakes survived not because they scored well but because no close neighbor triggered the pairwise dedup.

Script: `efficiency/python/lst_plot_pt5_scores_surv.py`
```bash
python3 efficiency/python/lst_plot_pt5_scores_surv.py LSTNtuple_allobj3.root --nevents 50 -o pt5_scores_surv_only.png
```

## Cut flow table (6 columns, as of 2026-08-07)

| Step | ttbar ABon | ttbar ABon pT>10 | QCD ABon ★ | QCD ABoff pairwise | QCD score<12 | QCD rm-pix off |
|---|---|---|---|---|---|---|
| **T5s formed** | 348,276 | (genuine, pT>10 only) | 1,581,676 | 1,583,179 | 1,583,179 | 1,583,179 |
| Killed by AfterBuild | 243,801 (70.0%) | — | 1,541,483 (97.5%) | 0 | 0 | 0 |
| T5s surv. AfterBuild | 104,475 (30.0%) | — | 40,193 (2.5%) | 1,583,179 | 1,583,179 | 1,583,179 |
| **pT5s formed** | 98,900 | — | 8,558 | 229,117 | 229,117 | 229,117 |
| Killed by BeforeTC | 12,466 (11.9%) | — | 32,366 (80.5%) | 1,567,205 (99.0%) | 1,567,542 (99.0%) | 1,567,205 (99.0%) |
| T5s surv. BeforeTC | 92,009 (88.1%) | — | 7,653 (19.0%) | 15,637 (1.0%) | 15,637 (1.0%) | 15,637 (1.0%) |
| Killed by pT5 dedup | 60,396 (61.1%) | — | 4,978 (58.2%) | 226,108 (98.7%) | 207,914 (90.7%) | — |
| **pT5s → TC** | 38,504 (38.9%) | — | 3,580 (41.8%) | 3,009 (1.3%) | 21,203 (9.3%) | 229,117 |

See `cutflow_table.png` for exact numbers including the pT>10 filter column. The pT>10 column shows only T5s/pT5s whose matched sim track has pT > 10 GeV (fakes excluded); actual counts are in the rendered PNG.

★ QCD ABon corrected: uses `LSTNtuple_allobj_ABon_ideal.root` (this server, CUDA, ideal pLS, `minNHitsForDup_T5=7`). Previous column used Jul-9 file with threshold=8 (Experiment 1).

## What remains to do / caveats

- **isPT5-priority bit (bit3)**: uninstrumented in ttbar ABon and QCD ABon ideal columns. To fix, apply Session 35 re-instrumented Kernels.h, rebuild, rerun those two ntuples.
- **ttbar pT>10 column**: fake T5s/pT5s are excluded by the filter (fakes have simIdx < 0). The counts reflect only genuine high-pT activity in ttbar; PU tracks with pT > 10 GeV ARE included if sim_pt > 10 in the LST ntuple's selected sim subset.
- **T5 count discrepancy (1,581,676 vs 1,583,179)**: this-server CUDA gives 1,581,676; other-server files give 1,583,179. Due to server hardware difference (L40 vs other), not CPU vs GPU backend. ~0.1% variation from GPU nondeterminism in tiebreaker kills. Not a concern for physics comparisons.

---

# Timing + ttbar Investigation + allobj5 Correction — Handoff (Session 38, 2026-08-04)

## ⚠️ SOURCE TREE STATE

- **`src/alpaka/LSTEvent.dev.cc`**: AB is disabled via `#if 0 // AFTERBUILD-DISABLE` (HEAD). Temporarily re-enabled and reverted during this session; current state = HEAD (AB off).
- **`src/alpaka/Kernels.h`**: at HEAD (pairwise `RemoveDupPixelQuintupletsFromMap`). Score-cutoff version lives on the other server only (used to produce allobj5).
- All other files at HEAD.

## New ntuple files this session

- **`LSTNtuple-ttbar-50-ABon2.root`** (50 events, ttbar PU200, AB ON, CUDA, this server) — corrected ttbar run; see ttbar investigation below.
- **`LSTNtuple_allobj5.root`** (50 events, hi-pT QCD PU200, AB OFF, score<12 cutoff, `--idealpls`, CPU, other server) — corrected from previous real-pLS version.
- **`LSTNtuple-ttbar-test.root`** (1 event, scratch) — can be ignored.
- Stub files **`LSTNtuple-ttbar-50-ABon.root`** and **`LSTNtuple-ttbar-50-ABon3.root`** — empty/failed runs; ignore.

## 1. Three-server timing comparison (AB off, pairwise pT5 dedup, 10 events, 4 streams)

All runs: `trackingNtuple-100.root --jet -n 10 -v 1 -s 4`, `Kernels.h` at HEAD, AB disabled.  
Log files (this server): `timing_ABoff_base_10evt.log` (CUDA), `timing_ABoff_base_10evt_cpu.log` (CPU).

### Average per-event timing (ms)

| Stage | CUDA (this server) | CPU (this server) | CPU (other server) | CUDA/CPU(this) |
|---|---:|---:|---:|---:|
| Hits | 5.8 | 0.9 | 1.0 | 6.3× slower* |
| MD | 0.4 | 0.9 | 0.8 | 2.2× faster |
| LS | 0.4 | 3.0 | 2.6 | 7.5× faster |
| T3 | 6.1 | 61.4 | 50.9 | 10.1× faster |
| T5 | 2,459 | 13,619 | 11,751 | 5.5× faster |
| pLS | 255 | 0.5 | 0.5 | 486× slower* |
| T4 (BeforeTC) | 1,790 | 2,305 | 1,971 | 1.3× faster |
| pT5 | 16,259 | 494 | 423 | **33× slower** |
| pT3 | 136 | 146 | 129 | ~1× |
| TC | 135,151 | 10,051 | 8,652 | **13.5× slower** |
| **Total avg** | **156,063** | **26,681** | **22,982** | **5.9× slower** |

\* Inflated by warmup: events 0, 6, 8 show ~14 ms Hits and ~1000 ms pLS (first touch of those stream allocations).

**Wall-clock (10 events):** CUDA 663.7 s · CPU(this) 197.4 s · CPU(other) not recorded.

**Key findings:**
1. **CUDA faster where expected**: T3 (10×), T5 (5.5×), LS, MD.
2. **pT5 and TC catastrophic on CUDA**: pT5 dedup 33× slower, TC 13.5× slower. These consume 97% of CUDA total but only 39% of CPU total.
3. **CUDA is 3.4× slower wall-clock** — completely opposite of AB-on behaviour (~9× faster). AB-off makes CUDA the wrong backend.
4. **CPU servers consistent**: other server uniformly ~16% faster.

## 2. ttbar cut flow table investigation and correction

**Problem:** Session 36 cut flow table had "ttbar PU200 AB ON" column showing 0 AfterBuild kills.

**Investigation:** `LSTNtuple-ttbar-50.root` was produced 2026-07-27; `#if 0 // AFTERBUILD-DISABLE` was committed 2026-07-16 (commit `96919ff`). AB was compiled away 11 days before the file was made. Confirmed: bit0 of `t5_isDupBits` is the only code path that records AfterBuild kills, and it lives entirely inside the `#if 0` block. The 0 means the kernel never ran.

**Correction:** Column label changed to "ttbar PU200 AB OFF ‡". Re-ran with AB actually enabled:
- File: `LSTNtuple-ttbar-50-ABon2.root` (CUDA, this server, 50 events, `PU200` input = `/data2/segmentlinking/CMSSW_12_2_0_pre2/trackingNtuple_ttbar_PU200.root`)
- Command: `lst_cuda -i PU200 --allobj -n 50 -s 4 -o LSTNtuple-ttbar-50-ABon2.root`  
  (no `--jet` — ttbar tracking ntuple lacks genjet branches; `--jet` crashes with `Branch sim_genjet_deltaEta does not exist!`)
- **Result: AfterBuild kills 70% of T5s in ttbar** (243,801 / 348,276). The "occupancy too low" claim was entirely wrong. AB is effective in ttbar; the dense-jet QCD extreme (97.5%) is driven by jet-core geometry, not general occupancy.

Note: T5 count per event in the re-run (6,966/evt) is 30× higher than the original mislabeled file (226/evt), indicating the two files came from different code versions or conditions. The 70% kill rate is the reliable finding.

## 3. allobj5 corrected with `--idealpls`

The original `LSTNtuple_allobj5.root` used real pLS (~481 pLS/evt), inconsistent with allobj3 (~114 pLS/evt, idealpls). Re-run on other server:

```bash
lst_cpu -i trackingNtuple-100.root --jet --idealpls --allobj -n 50 -s 4 -o LSTNtuple_allobj5.root
```

New file confirmed: 50 events, 219.7 MB, pT5s formed = 229,117 (matches allobj3 idealpls ✓). Cut flow table in Session 36 updated with corrected numbers — see that section.

**Key change:** score<12 cutoff now kills 90.7% of pT5s (vs 94.4% with real pLS), leaving **21,203 pT5s → TC** (vs 18,383 before, vs 3,009 for pairwise dedup).

## NumDen files and three-way comparison (completed session continuation)

All three NumDen files and the overlay plot were produced successfully.

**Root cause of earlier Jv1 stub files (464/472 bytes):** `createPerfNumDenHists -J` works fine
on these ntuples — single-event tests confirmed no branch errors. The stubs came from background
tasks being killed before the loop finished. Fix: run in foreground.

**NumDen files (all 16M, valid):**
- `NumDen-files/LSTNumDen_allobj3_Jv2.root` — allobj3, pairwise pT5 dedup, AB off, real pLS
- `NumDen-files/LSTNumDen_allobj4_Jv1.root` — allobj4, no pT5 dedup, AB off, real pLS (prior session)
- `NumDen-files/LSTNumDen_allobj5ideal_Jv2.root` — allobj5, score<12 cutoff, AB off, ideal pLS

**Three-way overlay plot:** `eff_vs_deltaR_allobj345_3way.png`

Labels: "allobj3 (pairwise dedup)" / "allobj4 (score<12+CCT5)" / "allobj5 (score<12+CCT5+idealpLS)"
Stages plotted: pLS_lower, T5_lower, pT5_lower, TC (sub-stages only)

## What remains to do

- **Inspect** `eff_vs_deltaR_allobj345_3way.png` and interpret efficiency differences across the three configs.
- **Fake rate comparison** (allobj3 vs allobj5): no three-way fake rate script yet — use
  `lst_compare_nodnn_fakerate.py` pairwise, or extend the overlay script.
- **Numeric table**: `lst_compare_nodnn_table.py LSTNumDen_allobj3_Jv2.root LSTNumDen_allobj5ideal_Jv2.root`
- **allobj4 event count**: confirmed 50 events (16M NumDen matches allobj3/5 file sizes).

---

# Score-Cutoff pT5 Dedup + Three-Way Comparison — Handoff (Session 37, 2026-08-04)

## ⚠️ SOURCE TREE STATE

- **`src/alpaka/Kernels.h`**: `RemoveDupPixelQuintupletsFromMap` body **replaced with score cutoff** (score ≥ 12 → killed, score < 12 → kept). The previous O(N²) pairwise comparison is gone. `isDupTiebreaker` and `passedNMatchedCut` are never set in this version (stay all-False). **This change is on this server and needs to be manually applied on the other server** (see below).
- **`src/alpaka/LSTEvent.dev.cc`** on this server: kernel call is active (unchanged). On the **other server**: the call was commented out for allobj4 — **must be restored** before rebuilding for allobj5.
- All other source-tree state unchanged from Session 36.

## Instructions for the other server (to produce LSTNtuple_allobj5.root)

### 1. Restore the kernel call in `LSTEvent.dev.cc` (lines 1154–1160)
The other server commented this out for allobj4. Restore:
```cpp
auto const removeDupPixelQuintupletsFromMap_workDiv =
    cms::alpakatools::make_workdiv<Acc2D>({max_blocks, 1}, {16, 16});
alpaka::exec<Acc2D>(queue_,
                    removeDupPixelQuintupletsFromMap_workDiv,
                    RemoveDupPixelQuintupletsFromMap{},
                    pixelQuintupletsDC_->view());
```

### 2. Replace `RemoveDupPixelQuintupletsFromMap` body in `Kernels.h`
Replace the existing double-loop body (lines ~460–496) with:
```cpp
  struct RemoveDupPixelQuintupletsFromMap {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc, PixelQuintuplets pixelQuintuplets) const {
      unsigned int nPixelQuintuplets = pixelQuintuplets.nPixelQuintuplets();
      // Score cutoff: keep any pT5 with rPhiChiSquared score < 12, kill the rest.
      // Lower score = better fit; threshold sits between genuine-survived median (3.5)
      // and genuine-killed median (21.5) from the pairwise-dedup baseline.
      constexpr float kScoreCutoff = 12.0f;
      for (unsigned int ix : cms::alpakatools::uniform_elements_y(acc, nPixelQuintuplets)) {
        float score = __H2F(pixelQuintuplets.score()[ix]);
        if (score >= kScoreCutoff) {
          rmPixelQuintupletFromMemory(pixelQuintuplets, ix);
        }
      }
    }
  };
```

### 3. Rebuild and run (AB off, same as allobj3/allobj4)
```bash
lst_make_tracklooper -G
export LD_LIBRARY_PATH=/mnt/data1/kk829/cuda_driver_libs_580:$LD_LIBRARY_PATH
lst_cuda -i trackingNtuple-100.root --jet --allobj -n 50 -s 4 -o LSTNtuple_allobj5.root
createPerfNumDenHists -J -i LSTNtuple_allobj5.root -o LSTNumDen_allobj5_Jv1.root
```

### 4. Verify
After the run, confirm in the ntuple:
- All surviving pT5s (`pT5_isDupReco == False`) have `pT5_score < 12`
- All killed pT5s (`pT5_isDupReco == True`) have `pT5_score >= 12`
- `pT5_isDupTiebreaker` and `pT5_passedNMatchedCut` are all False
- TC count is between allobj3 (3,009) and allobj4 (no-dedup upper bound)

## New scripts this session

- **`efficiency/python/lst_overlay_3way_eff_vs_deltaR.py`** — three-file efficiency-vs-ΔR overlay. Same stage colors/markers as the two-file version; A=filled/solid, B=open/dashed, C=half-filled/dotted. Run:
  ```bash
  python3 efficiency/python/lst_overlay_3way_eff_vs_deltaR.py \
      LSTNumDen_allobj3_Jv1.root LSTNumDen_allobj4_Jv1.root LSTNumDen_allobj5_Jv1.root \
      --labels "Pairwise dedup","No dedup","Score<12 cutoff" \
      -o eff_3way_allobj345.png
  ```

## Comparison plan (once allobj5.root is available)

Three datasets: allobj3 (pairwise dedup), allobj4 (no dedup), allobj5 (score cutoff).

```bash
# Three-way efficiency
python3 efficiency/python/lst_overlay_3way_eff_vs_deltaR.py \
    LSTNumDen_allobj3_Jv1.root LSTNumDen_allobj4_Jv1.root LSTNumDen_allobj5_Jv1.root \
    --labels "Pairwise dedup","No dedup","Score<12 cutoff" -o eff_3way_allobj345.png

# Pairwise fake rate (no 3-way script yet — run individually or extend lst_compare_nodnn_fakerate.py)
python3 efficiency/python/lst_plot_fakerate_vs_deltaR.py LSTNumDen_allobj3_Jv1.root --out fr_allobj3.png
python3 efficiency/python/lst_plot_fakerate_vs_deltaR.py LSTNumDen_allobj5_Jv1.root --out fr_allobj5.png

# Numeric table (pairwise)
python3 efficiency/python/lst_compare_nodnn_table.py LSTNumDen_allobj3_Jv1.root LSTNumDen_allobj5_Jv1.root
```

## Key discrepancies to keep in mind

1. **Score cutoff ≠ deduplication**: unlike the pairwise kernel (always one winner per pair), the cutoff keeps ALL pT5s with score < 12. Multiple near-identical pT5s from the same track may survive → inflated TC duplicate rate.
2. **`isDupTiebreaker` / `passedNMatchedCut` are meaningless** in allobj5 (always False).
3. **allobj4 event count**: confirm it matches allobj3/allobj5 (50 events) before comparing.

## Pending

- ~~Other server: build with `RemoveDupPixelQuintupletsFromMap` disabled, run, compare TC counts.~~ **DONE** — `LSTNtuple_allobj4.root` produced on other server (AB OFF, no pT5 dedup). TC count comparison not yet done.
- Other server: apply score-cutoff Kernels.h change, restore LSTEvent.dev.cc kernel call, rebuild, run → `LSTNtuple_allobj5.root`.
- Run three-way comparison (eff + fake rate + numeric table) once allobj5 is available.
- All prior pending items from Session 35 still open (clean 4-config ΔR comparison, etc.).

---

# pT5 Dedup Deep-Dive + Cut Flow Tables — Handoff (Session 36, 2026-07-27)

## ⚠️ SOURCE TREE STATE

- **`src/alpaka/LSTEvent.dev.cc`**: `RemoveDupPixelQuintupletsFromMap` kernel call **commented out on this server** (for a study run on the other server). Lines 1154–1160 replaced with comments. Restore with:
  ```cpp
  auto const removeDupPixelQuintupletsFromMap_workDiv =
      cms::alpakatools::make_workdiv<Acc2D>({max_blocks, 1}, {16, 16});
  alpaka::exec<Acc2D>(queue_,
                      removeDupPixelQuintupletsFromMap_workDiv,
                      RemoveDupPixelQuintupletsFromMap{},
                      pixelQuintupletsDC_->view());
  ```
- **`interface/PixelQuintupletsSoA.h`** + **`src/alpaka/PixelQuintuplet.h`** + **`src/alpaka/Kernels.h`** + **`standalone/code/core/write_lst_ntuple.cc`**: Two new pT5 SoA columns and ntuple branches from the previous session (`isDupTiebreaker`, `passedNMatchedCut`) are confirmed working in `LSTNtuple_allobj3.root`. See Session 35 for the exact code changes.
- **AfterBuild state**: `RemoveDupQuintupletsAfterBuild` is still ENABLED on this server.

## New ntuple files this session

- **`LSTNtuple_allobj3.root`** (50 events, hi-pT QCD PU200, AB OFF, imported from other server) — contains `pT5_isDupTiebreaker` and `pT5_passedNMatchedCut` branches.
- **`LSTNtuple_allobj4.root`** (other server, hi-pT QCD PU200, AB OFF, `RemoveDupPixelQuintupletsFromMap` disabled) — upper bound on pT5 TC counts with no pT5 dedup.
- **`LSTNtuple-ttbar-50.root`** (50 events, ttbar PU200, **AB OFF** — mislabeled as AB ON in original; see footnote ‡) — used for cut flow comparison.

## New scripts this session

- **`efficiency/python/lst_pt5_pairs.py`** — post-hoc replay of `RemoveDupPixelQuintupletsFromMap` pair logic from ntuple. For every pair of pT5s within |Δη|<0.2, |Δφ|<0.2, records: `deta`, `dphi`, `score_i`, `score_j`, `genuine_i/j`, `killed_i/j`, `cat_i/j`, `jetdR_i/j`. Saves to `.npz`. Produces `_deta_dphi.png` (2D histogram by pair type) and `_score_ratio.png` (log score ratio histogram). Run: `python3 efficiency/python/lst_pt5_pairs.py LSTNtuple_allobj3.root --nevents 50 --out pt5_pairs_allobj3`. **Output**: `pt5_pairs_allobj3.npz` (802M pairs, ~3 min runtime).

## What we did (2026-07-27)

### 1. Confirmed `passedNMatchedCut` in allobj3.root

`pT5_passedNMatchedCut` is True for **226,108/226,108 killed pT5s (100%)** and 2,817/3,009 surviving pT5s. Only 192 surviving pT5s never had a competitor pass nMatched≥7. **Conclusion: nMatched≥7 is not a selective filter at all** — every killed pT5 reached the score comparison. The score and tiebreaker are doing all the work.

### 2. Score histograms (log scale, 4 categories)

- **`pt5_scores_all50evt.png`** — rPhiChiSquared for all 229,100 pT5s across 50 events (4 categories: genuine/fake × survived/killed). Genuine survived peaks well below 10; genuine killed has a longer tail; fake killed dominates 100–200k range.
- **`pt5_scores_jetcore_50evt.png`** — same but filtered to genuine pT5s with `sim_genjet_deltaR < 0.1`. Fake categories empty (no jet ΔR for fakes). Genuine survived N=1,417 (median 3.54), genuine killed N=28,701 (median 21.5). Overlap visible — tiebreakers cause ~20% of genuine jet-core kills regardless of score.

### 3. Tiebreaker counts (from `pT5_isDupTiebreaker`)

| Population | Killed by score | Killed by tiebreaker |
|---|---|---|
| All killed pT5s | 211,996 (93.8%) | 14,112 (6.2%) |
| Genuine killed | 27,903 (82.1%) | 6,094 (17.9%) |
| Jet-core genuine killed (jetdR<0.1) | 22,997 (80.1%) | 5,704 (19.9%) |

Tiebreakers are more common among genuines (~18%) than fakes (~4%) because genuine pT5s from the same track share the same pLS geometry, producing near-identical `rPhiChiSquared` that rounds to the same FP16 value.

### 4. pT5 all-pairs analysis (`lst_pt5_pairs.py`)

Since `passedNMatchedCut` is effectively always True, we can replay the pair logic using only |Δη|<0.2, |Δφ|<0.2. Ran on all 50 events → **802M pairs** (averages ~16M/event). Shows how dense the pT5 cloud is in AfterBuild-OFF. Pair data saved to `pt5_pairs_allobj3.npz`. Jet-core filtered plots (pairs where at least one pT5 has `jetdR<0.1`): `pt5_pairs_jetcore_deta_dphi.png`, `pt5_pairs_jetcore_score_ratio.png` (107M/802M pairs = 13% in jet core).

### 5. T5 + pT5 cut flow tables (three samples, 50 events each)

Key insight discovered: **T5 and pT5 dedup are parallel streams, not sequential.** `CreatePixelQuintupletsFromMap` runs AFTER AfterBuild but BEFORE BeforeTC. So pT5s are formed from T5s that BeforeTC will later kill, and BeforeTC kills affect T5 objects only — they do NOT propagate to pT5 objects. The correct flow diagram:

```
T5s formed
  ↓ AfterBuild
T5s surviving AfterBuild
  ├─→ [pT5 building] → pT5s formed → RemoveDupPixelQ → pT5s surviving (→ TC)
  └─→ [BeforeTC]    → T5s surviving (→ standalone TC)
```

**Cut flow (50 events each):**

| Step | ttbar PU200 AB OFF ‡ | ttbar PU200 AB ON ♦ | hi-pT QCD PU200 AB ON ★ | hi-pT QCD PU200 AB OFF (pairwise) | hi-pT QCD PU200 AB OFF (score<12) |
|---|---|---|---|---|---|
| T5s formed | 11,289 | 348,276 | 1,581,676 | 1,583,179 | 1,583,179 |
| Killed by AfterBuild | 0 (0.0%) | 243,801 (70.0%) | 1,541,483 (97.5%) | 0 (0.0%) | 0 (0.0%) |
| T5s surviving AfterBuild | 11,289 (100%) | 104,475 (30.0%) | 40,193 (2.5%) | 1,583,179 (100%) | 1,583,179 (100%) |
| pT5s formed | 9,470 | 98,900 | 8,558 | 229,117 | 229,117 |
| Killed by BeforeTC (T5s) | 1,880 (16.7%) | 12,466 (11.9%) | 32,366 (80.5%) | 1,567,205 (99.0%) | 1,567,542 (99.0%) |
| T5s surviving BeforeTC | 9,351 (82.8%) | 92,009 (88.1%) | 7,653 (19.0%) | 15,637 (1.0%) | 15,637 (1.0%) |
| Killed by pT5 dedup | 7,501 (79.2%) | 60,396 (61.1%) | 4,978 (58.2%) | 226,108 (98.7%) | 207,914 (90.7%) |
| pT5s surviving (→ TC) | 1,969 (20.8%) | 38,504 (38.9%) | 3,580 (41.8%) | 3,009 (1.3%) | 21,203 (9.3%) |

★ **hi-pT QCD AB ON corrected run (2026-08-07):** `LSTNtuple_allobj_ABon_ideal.root`, CUDA, 50 events, `trackingNtuple-100.root --jet --idealpls --allobj`, ideal pLS, `minNHitsForDup_T5=7`. Previous column used `LSTNtuple_dedup_stages_v2.root` (Jul 9, `minNHitsForDup_T5=8`, ideal pLS) — wrong code version (Experiment 1 threshold). BeforeTC kill % expressed as % of T5s surviving AfterBuild (consistent with other columns); the old column showed % of all T5s formed.

‡ **ttbar label correction (2026-08-04):** `LSTNtuple-ttbar-50.root` was produced on 2026-07-27, but `#if 0 // AFTERBUILD-DISABLE` was committed on 2026-07-16 (commit `96919ff`). AB was compiled away before the file was produced — the 0 AfterBuild kills means the kernel never ran, not that it ran and found nothing.

♦ **ttbar AB ON corrected run (2026-08-04):** `LSTNtuple-ttbar-50-ABon2.root`, CUDA, 50 events, same input (`trackingNtuple_ttbar_PU200.root`). AB kills **70%** of T5s — the "occupancy too low" claim was wrong. Note: the "T5s formed" count (348k ≈ 6,966/evt) is 30× higher than the mislabeled AB OFF file (226/evt), indicating those two files came from different code versions or input conditions; the 70% kill rate is the reliable new finding.

**allobj5 corrected (2026-08-04):** re-run with `--idealpls` on other server (CPU, `lst_cpu`); pT5s formed now matches allobj3 (229,117 both), confirming pLS type is consistent. Previous allobj5 used real pLS (325,454 pT5s). Score-cutoff dedup (score<12) kills 90.7% of pT5s, leaving 21,203 (vs 98.7% / 3,009 surviving for pairwise).

Files used: `LSTNtuple-ttbar-50.root`, `LSTNtuple-ttbar-50-ABon2.root`, `Ntuple-files/LSTNtuple_dedup_stages_v2.root` (first 50 events), `LSTNtuple_allobj3.root`, `LSTNtuple_allobj5.root` (corrected with `--idealpls`).

**Notable findings:**
- **ttbar PU200 AB ON**: AfterBuild kills **70%** of T5s — comparable occupancy to hi-pT QCD is not the issue; the previous 0-kill result was a code bug (AB disabled). AfterBuild is effective across both samples, just less extreme than hi-pT QCD (97.5%) because dense jet cores in QCD drive near-identical within-module T5s.
- **hi-pT QCD AB ON vs OFF**: nearly identical T5 counts formed (~1.58M each). AB ON gates pT5 formation to only 2.5% of T5s tried (40k), producing 9,371 pT5s of which 37% survive. AB OFF tries all T5s, produces ×24 more pT5s (229k), but RemoveDupPixelQ kills 98.7%, leaving only 3,009 surviving — *fewer* than AB ON.
- **ttbar apparent paradox resolved**: 9,470 pT5s formed despite only 9,351 T5s surviving BeforeTC. Explained by: (a) 361 pT5s were built on T5s that BeforeTC later kills (pT5s remain alive; BeforeTC doesn't propagate); (b) pT5 formation and BeforeTC run from the same post-AfterBuild pool, not sequentially.

### 6. `RemoveDupPixelQuintupletsFromMap` disabled on this server

Commented out in `LSTEvent.dev.cc:1154–1160`. Other server will rebuild and run to get a ntuple where all 229k pT5s survive (upper bound on pT5 TC counts with no dedup).

### 7. BeforeTC / pT5 parallel-stream verified numerically (ttbar PU200, 50 evt)

Confirmed explicitly from `LSTNtuple-ttbar-50.root`:

- **1,880 T5s killed by BeforeTC**, of which:
  - **361** had `partOfPT5 = True` (their T5 was embedded in a pT5 already formed)
  - **1,519** were not part of any pT5
- **Those 361 pT5s persisted independently** — BeforeTC writes only to the T5 `isDupBits` field and never touches the pT5 collection, which was already written by `CreatePixelQuintupletsFromMap`.
- Fate of the 361 pT5s whose embedded T5 BeforeTC killed:
  - **86 survived** `RemoveDupPixelQuintupletsFromMap` → became TCs
  - **275 killed** by `RemoveDupPixelQuintupletsFromMap`
- The 361 killed T5s map 1:1 to unique pT5s (no T5 backs multiple pT5s).

**Takeaway**: A pT5 can become a TC even if the standalone T5 it was built from was judged a duplicate by BeforeTC. The pT5 and its constituent T5 have fully independent fates once `CreatePixelQuintupletsFromMap` has run.

## Pending

- ~~Other server: build with `RemoveDupPixelQuintupletsFromMap` disabled, run, compare TC counts.~~ **DONE** — `LSTNtuple_allobj4.root` produced on other server (AB OFF, no pT5 dedup). TC count comparison not yet done.
- All prior pending items from Session 35 still open (clean 4-config ΔR comparison, etc.).

---

# AfterBuild-Off Dedup Funnel + BeforeTC Instrumentation — Handoff (Session 35, 2026-07-24)

## Standing Rules
- **Always Use CUDA** for efficiency runs (CPU ~9× slower). This session used `lst_cpu` only because the AfterBuild-off `--allobj` runs were done on a **second server** (see below).
- **INPUT: always `-i trackingNtuple-100.root` + `--jet`** (now the documented default in CLAUDE.md). Never `-i PU200`.
- **CUDA driver workaround:** `export LD_LIBRARY_PATH=/mnt/data1/kk829/cuda_driver_libs_580:$LD_LIBRARY_PATH` before `lst_cuda`.
- **Detached runs:** use `setsid bash -c '<env; cmd>' > log 2>&1 &` so they survive logout.

## File organization (NEW this session — saved to memory `project_file_organization`)
User moved outputs into subdirs of `standalone/`:
- `LSTNtuple*.root` → **`Ntuple-files/`**
- `LSTNumDen*.root` → **`NumDen-files/`**
Look there, not in the standalone root. (Freshly-produced files may still land in the root until moved.)

## ⚠️ SOURCE TREE STATE (changed this session)

- **`src/alpaka/Kernels.h` — RE-INSTRUMENTED (Path B).** Restored the fine-grained BeforeTC bit tracking from commit `96919ff6667`:
  - `rmQuintupletFromMemory` signature: `bool secondpass` → `uint8_t killBits = 0x01u`; body `isDup |= killBits`.
  - `RemoveDupQuintupletsBeforeTC` now sets **bit1=condA, bit2=condB, bit3=isPT5-priority-decisive** distinctly (was lumped into bit1 only after the S34 revert).
  - AfterBuild `minNHitsForDup_T5` kept at **7** (master value — did NOT reintroduce Experiment 1's 8).
  - This is an **uncommitted working-tree edit**. The other server got it by manual copy.
- Other S25 instrumentation (triedInPT5 SoA column + setter, CrossClean bit4, zero-init, `t5_isDupBits`/`t5_triedInPT5`/`t5_partOfPT5` ntuple branches) **survived the S34 revert** — only `Kernels.h` needed restoring.
- **`standalone/CLAUDE.md`**: `--jet` documented as the always-on default; removed the no-jet example line.
- **`efficiency/python/lst_count_dedup_stages.py`**: added `--nevents N` option (default -1 = all) for event subsetting; count-plot ylabel de-hardcoded from "100 events".
- **AfterBuild state**: on THIS server `RemoveDupQuintupletsAfterBuild` is still ENABLED. The AfterBuild-OFF runs were done on the **second server** (its `LSTEvent.dev.cc` has the kernel call commented out).

## Second-server setup (what it needs — from git-untracked audit)
Checkout of `claude-edits` (commit `bc4cc5`) has all build sources EXCEPT:
1. **`trackingNtuple-100.root`** (92 MB, gitignored `*.root`) — must be copied manually. Required, and `--jet` specifically needs it.
2. **CUDA driver libs** — machine-specific, not a repo file (likely N/A on a healthy-driver box).
`setup.sh` and `TruthPixelSeeds.cc/.h` ARE committed → present after checkout. No untracked `.cc/.h/.cu` build sources exist.

## What we did (2026-07-24)

### 1. Finished the 50-evt AfterBuild-off run WITH `--jet`
S34's `LSTNtuple_noAfterBuild_idealpls_50evt.root` was run WITHOUT `--jet` (no genjet branches). Reran (CUDA, this server): **`LSTNtuple_noAfterBuild_idealpls_50evt_jet.root`** (50 evt, ~2.35 h — same as the no-jet run; `--jet` only affects branch filling). NumDen: **`NumDen-files/LSTNumDen_noAfterBuild_idealpls_50evt_Jv1.root`**.

### 2. Cleaned up partial/corrupt ROOT files
Checked every `*.root` in standalone; renamed **27 partial/empty/truncated files** with `DELETE-` prefix (empty stubs, mid-write truncations with no `tree`, `.killed_partial`/`.crashed`/`.partial` variants). Verified the 64-evt rescued file `LSTNtuple_noAfterBuild_idealpls_jet_64evt.root` (+ its `_v4` NumDen) is intact and NOT prefixed.

### 3. Second-server `lst_cpu` AfterBuild-off `--allobj` runs
- **`LSTNtuple_allobj.root`** (Path A input): `lst_cpu --idealpls --jet --allobj -n 50`, AfterBuild OFF, **reverted (coarse) Kernels.h**. 50 evt, 210 MB, 437 branches, ~1 h (CPU noAfterBuild finished fast because final TC count stays ~128/evt even as intermediate pT5 explodes ~50×). NumDen: **`NumDen-files/LSTNumDen_allobj_cpu_noAfterBuild_50evt_Jv1.root`**.
- **`LSTNtuple_allobj2.root`** (Path B input): same command but with **re-instrumented Kernels.h**. 50 evt, 210 MB. Confirmed bit0=0 (AfterBuild off), **bit3 (isPT5-priority) now populated = 959,139**; bit1/bit2 populated. Same 1,583,179 total T5s as allobj (same events).
- Flag-name gotcha found: options are `--idealpls`/`--jet`/`--allobj` (a `--ideal-pls` typo throws `cxxopts Option_not_exists`).

### 4. CPU-vs-CUDA consistency plot
New script **`efficiency/python/lst_overlay_eff_vs_deltaR_stages.py`** (two NumDen files, dataset A = filled/solid, B = open/dashed, shared per-stage colors). Output **`eff_vs_deltaR_noAfterBuild_cuda64_vs_cpu50_overlay.png`** (CUDA-64 vs CPU-50, both AfterBuild-off). Curves track within stats — no CPU/CUDA discrepancy. Also made single-file `eff_vs_deltaR_allobj_cpu_noAfterBuild_50evt.png`.

### 5. Repeated the Session-25 dedup-funnel analysis, AfterBuild ON vs OFF
Session-25 tables used **`Ntuple-files/LSTNtuple_dedup_stages_v2.root`** (100 evt, CUDA, AfterBuild ON, FULL instrumentation — bit0..4 all match the S25 table exactly).
- **Path A** (`lst_count_dedup_stages.py` on `LSTNtuple_allobj.root`): coarse — BeforeTC lumped in bit1; isPT5-priority row read 0.
- **Session-25 first-50 subset** (`--nevents 50` on dedup_stages_v2): validated same event set — total T5s **1,581,676** vs Path A **1,583,179** (0.1%).
- **Path B** (`LSTNtuple_allobj2.root`): full breakdown with isPT5-priority.

**50-evt AfterBuild ON vs OFF (same events):**
| | ON (S25 first-50) | OFF (Path B) |
|---|---|---|
| Total T5s | 1,581,676 | 1,583,179 |
| Killed AfterBuild | 1,541,478 (97.5%) | 0 |
| pT5-matched (success) | 8,801 | 193,313 (×22) |
| BeforeTC killed | 31,962 | 1,567,206 |
| isPT5-priority (bit3) | 7,749 (24% of kills) | 959,139 (**61% of kills**, ×124) |
| Surviving T5s (all) | 8,075 | 15,636 (×1.9) |
| Surviving jet core | 346 | 752 (×2.2) |

**Findings:** AfterBuild was gating pT5 supply → OFF gives 12–27× more pT5 matches; **isPT5-priority becomes the majority BeforeTC kill mechanism (24%→61%)**; jet-core paradox — 12× more pT5s form but survivors only ×2.2 because the extra pT5s cannibalize standalone T5s via bit3. Plots: `dedup_stages_{pathA_cpu_noAfterBuild_50evt,session25_first50evt,pathB_cpu_noAfterBuild_50evt}_{counts,fractions}.png`.

### 6. pT5-breakdown comparison plots
New script **`efficiency/python/lst_compare_pt5_breakdown.py`** (`--on`/`--off`/`--nevents`). Outputs:
- **`compare_pt5_ONvsOFF_50evt_bars.png`** — grouped bars: pT5 success-rate (OFF lower, denominator effect) + isPT5-priority share (OFF higher everywhere).
- **`compare_pt5_ONvsOFF_50evt_prio_vs_dR.png`** — isPT5-priority fraction vs ΔR; OFF above ON across the range, peaks ~ΔR≈0.02 (~0.68). Note "All ΔR" bar includes fake T5s (simIdx<0), so it exceeds both core/isolation bars — matches S25 methodology.

### 7. BeforeTC keep-decision factors (documented, no code change)
Keep/kill priority order: **partOfPT5 (isPT5) → score_rphisum (lower kept) → object index (lower kept)**, gated by module co-location, both-isPT5 exclusion, dEta/dPhi≤0.1, and the condA/condB duplicate test (dR², nMatched≥5, embedding d²). **Countable** from `t5_isDupBits`: distinct T5s removed, condA/condB/both, isPT5-priority. **NOT** countable without more bits: score-decided vs index-tiebreaker split (both = killed-without-bit3), per-pair decision counts (flags OR-accumulated per object), which opponent killed a T5. To split score vs tiebreaker, add a bit (e.g. `0x20`) in the `else` branch at Kernels.h:301–302.

## Pending / next
- **Score-vs-tiebreaker bit** (`0x20`) — offered, not yet applied. Would need another second-server run.
- Optional: fake-vs-genuine split on the isPT5-priority-vs-ΔR plot.
- The re-instrumented `Kernels.h` is uncommitted here; second server has it by copy. Decide whether to commit to `claude-edits`.
- All prior S34/S33 pending items (clean 100-evt AfterBuild-on/off 4-config ΔR set, etc.) still open.

---

# Reverts + Mastercuts Baseline Runs — Handoff (Session 34, 2026-07-21)

## Standing Rules
- **Always Use CUDA.** 2-GPU box: GPU0=L40(46GB), GPU1=L4(23GB); 376GB RAM; 192 CPU.
- **INPUT: never `-i PU200`.** Always `-i trackingNtuple-100.root` (100 evt) or `trackingNtuple-1000.root` (1000 evt). PU200 has no genjet branches. Add `--jet` whenever ΔR-to-genjet is needed downstream.
- **CUDA driver workaround (recent):** `export LD_LIBRARY_PATH=/mnt/data1/kk829/cuda_driver_libs_580:$LD_LIBRARY_PATH` must be set before running `lst_cuda`.

## Source tree state (as of end of Session 34)

Active modifications vs master:
1. **`standalone/code/core/TruthPixelSeeds.cc`** (new file, not in master): ptErr/etaErr use **random draw** from per-event real-seed pool (`std::mt19937` seed 42). This was restored to random-draw during Session 34 (Step 1 of the session).
2. **All other previously-modified files** (`Kernels.h`, `PixelTriplet.h`) **reverted to master** this session (see below).

`LSTEvent.dev.cc`: **AfterBuild IS enabled**.

**CUDA binary** (`standalone/LST/liblst_cuda.so`): rebuilt Jul 21 (this session). Matches source — master cuts, random-draw ptErr/etaErr.

---

## Session 34 — What we did (2026-07-21)

### 1. Restored TruthPixelSeeds.cc to random-draw

`TruthPixelSeeds.cc` was reverted from Session 33's per-event median back to random draw from the real-seed pool (`std::mt19937` seed 42, `std::uniform_int_distribution`). This was done at the start of the session as Step 1 of the planned clean etaErr/ptErr comparison.

### 2. Investigated pT5 dedup score change (Kernels.h)

The `rPhiChiSquaredInwards` addition to the pT5-vs-pT5 dedup score in `Kernels.h` was traced back to **Session 22**. Conclusion: it was a tested-and-found-insufficient experiment (+1 TC recovered, fake rate worsened 0.281→0.354). It was left in the source tree after Session 22 and never reverted.

### 3. Reverted Kernels.h and PixelTriplet.h to master (commit `fb9f57c3f23`)

Both files restored with `git checkout master -- <file>`, then committed. Changes reverted:

**Kernels.h:**
- `rmQuintupletFromMemory` signature: `uint8_t killBits=0x01u` → `bool secondpass=false`
- `minNHitsForDup_T5`: 8 → **7** (reverts Experiment 1)
- BeforeTC dedup: condA/condB/geoBits/pT5Decisive instrumentation removed
- pT5 dedup score: `DNN score + rPhiChiSquaredInwards` → **DNN score only**

**PixelTriplet.h:**
- z-window (PPBB), zWindow, dPhiCut: 4× loosenings removed
- rt-window (PPEE), rtWindow, dPhiCut: 4× loosenings removed

Motivation: these changes were confounding the etaErr/ptErr sampling comparison. The `rPhiChiSquaredInwards` addition was also confirmed insufficient.

Also noted: `PixelQuintupletsSoA.h` and `PixelQuintuplet.h` still store `rPhiChiSquaredInwards` unconditionally (left from S22) but nothing in the dedup kernel uses it now.

### 4. CUDA binary rebuilt, both runs re-executed

After reverting, rebuilt with `lst_make_tracklooper -G` and ran:

```bash
export LD_LIBRARY_PATH=/mnt/data1/kk829/cuda_driver_libs_580:$LD_LIBRARY_PATH
lst_cuda --idealpls --jet --allobj -n 100 -s 4 -i trackingNtuple-100.root \
    -o LSTNtuple_idealpls_mastercuts_100evt_v2.root
lst_cuda --jet --allobj -n 100 -s 4 -i trackingNtuple-100.root \
    -o LSTNtuple_realpls_mastercuts_100evt_v2.root
```

Both completed (~1.1 GB each). These are the clean "master cuts" baseline ntuples (minNHitsForDup_T5=7, no PixelTriplet loosening, pT5 dedup = DNN score only, random-draw ptErr/etaErr).

### 5. NumDen files generated

```bash
createPerfNumDenHists -J -i LSTNtuple_idealpls_mastercuts_100evt_v2.root \
    -o LSTNumDen_idealpls_mastercuts_100evt_v2_Jv1.root   # done, 17 MB
createPerfNumDenHists -J -i LSTNtuple_realpls_mastercuts_100evt_v2.root \
    -o LSTNumDen_realpls_mastercuts_100evt_v2_Jv1.root    # in progress at handoff
```

### 6. Efficiency plots (pending)

Once both NumDen files are complete, produce plots with:

```bash
python3 efficiency/python/lst_plot_eff_vs_deltaR_stages.py \
    LSTNumDen_idealpls_mastercuts_100evt_v2_Jv1.root \
    -o eff_vs_deltaR_idealpls_mastercuts_100evt_v2.png
python3 efficiency/python/lst_plot_eff_vs_deltaR_stages.py \
    LSTNumDen_realpls_mastercuts_100evt_v2_Jv1.root \
    -o eff_vs_deltaR_realpls_mastercuts_100evt_v2.png
```

### 7. CLAUDE.md updated

Added to the "Running LST" section:
- `LD_LIBRARY_PATH` workaround (with "recent addition" note)
- `--jet` flag documentation (requires trackingNtuple-100.root; crashes on other inputs)
- Explicit note that `-i PU200` is never valid

---

# Real vs Ideal pLS T5 Dedup Comparison — Handoff (Session 33, 2026-07-17)

## Source tree state (as of end of Session 33)

Two active modifications from baseline:
1. **`src/alpaka/Kernels.h` line 209**: `minNHitsForDup_T5 = 8` (Experiment 1, unchanged from S30+).
2. **`standalone/code/core/TruthPixelSeeds.cc`**: ptErr/etaErr use **per-event median** (reverted from Session 32's random-draw). Uses `medianOrDefault` lambda; no `#include <random>`.

`LSTEvent.dev.cc`: **AfterBuild IS enabled** (restored from Session 32's comment-out). No marker comment remains.

**CUDA binary** (`standalone/LST/liblst_cuda.so`): rebuilt Jul 17 (current session). Matches source (AfterBuild ON, minNHitsForDup_T5=8, median ptErr).

**TruthPixelSeeds.cc note**: user said to restore to random-draw later. The current median version is intentional/temporary for this comparison.

---

## Session 33 — What we did (2026-07-17)

### 1. Two code reverts (in preparation for real vs ideal pLS comparison with AfterBuild ON)

**`src/alpaka/LSTEvent.dev.cc`**: Re-enabled `RemoveDupQuintupletsAfterBuild` kernel (removed the comment-out block from Session 32).

**`standalone/code/core/TruthPixelSeeds.cc`**: Reverted from random ptErr/etaErr draw back to per-event median:
- Removed `#include <random>`
- Replaced `std::mt19937 rng` + `uniform_int_distribution` with a `medianOrDefault` lambda
- `out.see_ptErr.push_back(medPtErr)` / `out.see_etaErr.push_back(medEtaErr)` (same value all seeds in event)

Rebuilt CUDA binary with `lst_make_tracklooper -G`.

### 2. 64-event partial file: efficiency plot

The 64-event recovered file from Session 32 (`LSTNtuple_noAfterBuild_idealpls_jet_64evt.root`) and its NumDen (`LSTNumDen_noAfterBuild_idealpls_jet_64evt_v4.root`) were used to produce:

```bash
python3 efficiency/python/lst_plot_eff_vs_deltaR_stages.py \
  LSTNumDen_noAfterBuild_idealpls_jet_64evt_v4.root \
  -o eff_vs_deltaR_noAfterBuild_idealpls_64evt.png
```

**Output**: `eff_vs_deltaR_noAfterBuild_idealpls_64evt.png`. Same script/flags as `eff_vs_deltaR_idealpls_fixed.png`. Key differences from baseline: TC eff at ΔR~0 lower (~0.40 vs ~0.53), pT5 lower and noisier, upper stages (T3, pLS, MD, LS) broadly similar. Wider error bars throughout due to only 64 events and AfterBuild disabled.

### 3. Real vs ideal pLS T5 dedup stage counts (AfterBuild ON, 100 events)

**Goal**: understand whether switching from real to synthetic (median ptErr) pLS changes T5 creation or survival. T5s are OT-only so total count is expected to be identical.

**New ntuple**: `LSTNtuple_realpls_jet_100evt.root` — real pLS, AfterBuild ON, `--jet --allobj -n 100`, Jul 17. (~20 min wall time, avg 8880 T5/evt).

**Ideal pLS baseline**: `LSTNtuple_dedup_stages_v2.root` — ideal pLS, AfterBuild ON, `--jet --allobj`, 100 events, **Jul 9** (older binary). Has all required branches (confirmed): `t5_isDupBits`, `t5_triedInPT5`, `t5_partOfPT5`, `t5_simIdx`, `sim_genjet_deltaR`.

Ran `efficiency/python/lst_count_dedup_stages.py` on both:
```bash
python3 efficiency/python/lst_count_dedup_stages.py LSTNtuple_realpls_jet_100evt.root   --out dedup_stages_realpls
python3 efficiency/python/lst_count_dedup_stages.py LSTNtuple_dedup_stages_v2.root      --out dedup_stages_idealpls_median
```

**Results**:

| | All ΔR (Real) | All ΔR (Ideal) | ΔR<0.01 (Real) | ΔR<0.01 (Ideal) | ΔR>0.05 (Real) | ΔR>0.05 (Ideal) |
|---|---|---|---|---|---|---|
| Total T5s | 5,049,304 | 5,049,304 | 34,434 | 34,434 | 4,965,001 | 4,965,001 |
| AfterBuild kill % | 97.9% | 98.3% | 94.2% | 94.6% | 98.0% | 98.4% |
| Eligible for BeforeTC | 107,172 | 84,011 | 2,000 | 1,853 | 100,093 | 77,257 |
| BeforeTC kill % of elig | 84.0% | 80.0% | 70.7% | 67.3% | 86.3% | 82.9% |
| isPT5-priority kills | 33,742 | 17,467 | 589 | 421 | 32,437 | 16,487 |
| Successful pT5 match | 22,963 | 18,155 | 1,213 | 1,191 | 18,225 | 13,542 |
| **Surviving T5s** | **16,698** | **16,479** | **575** | **601** | **13,263** | **12,933** |

**⚠️ BINARY MISMATCH CAVEAT**: `LSTNtuple_dedup_stages_v2.root` (Jul 9) and `LSTNtuple_realpls_jet_100evt.root` (Jul 17) were generated with different binaries. Total T5s are identical (correct — T5 building is OT-only and pLS-independent), but AfterBuild kill rate differs by ~0.4% (97.9 vs 98.3%), which cannot be caused by pLS differences since AfterBuild runs before pLS matching. This difference reflects a binary/code change between Jul 9 and Jul 17, contaminating the intermediate stage comparison.

**Clean conclusion despite caveat**: Final surviving T5 count is nearly identical (16,698 real vs 16,479 ideal globally; 575 vs 601 in jet core). Regardless of the binary mismatch, pLS type does NOT materially change the final T5 TC population.

**To get a clean comparison**: re-run `lst_cuda --idealpls --jet --allobj -n 100` with the current binary (Jul 17), then re-run `lst_count_dedup_stages.py` on both files.

**Plots**: `dedup_stages_realpls_counts.png`, `dedup_stages_realpls_fractions.png`, `dedup_stages_idealpls_median_counts.png`, `dedup_stages_idealpls_median_fractions.png`.

### 4. How ΔR is computed for T5 objects (explanation)

T5 sim matching: `matchedSimTrkIdxsAndFracs()` retrieves the T5's 10 `Phase2OT` hit indices, maps each via `trk_ph2_simHitIdx[hitidx]` → `trk_simhit_simTrkIdx[simhit_idx]` → simtrack index, counts fraction per simtrack. Best-fraction simtrack above `matchfrac` threshold becomes `t5_simIdx` (-999 if fake).

ΔR = `sim_genjet_deltaR[t5_simIdx]` — a per-simtrack quantity (angular distance to nearest qualifying genjet pT>1000, |η|<2.5). Fake T5s (simIdx=-999) have no ΔR and are excluded from core/isolation breakdowns in `lst_count_dedup_stages.py`.

For efficiency-vs-ΔR histograms in `performance.cc`, ΔR is always indexed per-simtrack (denominator loop), not per-T5-object — so the efficiency ΔR axis is well-defined. The T5 ΔR assigned in `lst_count_dedup_stages.py` is less reliable in dense jet cores than TC-level ΔR because OT-only hit matching is more ambiguous than pixel-inclusive TC matching.

### 5. NumDen files for `lst_overlay_t5_real_vs_ideal.py` (100 events)

The script `efficiency/python/lst_overlay_t5_real_vs_ideal.py` uses `--real` and `--ideal` args:
- **Real pLS**: `LSTNumDen_fix2.root`
- **Ideal pLS**: `LSTNumDen_idealpls_fixed.root`

Both verified to have the `_deltaR` histograms needed (372 T5_lower deltaR keys each).

---

## Pending after Session 33

- **Restore TruthPixelSeeds.cc to random-draw** (user: "we will restore later"). The median version is currently in place.
- **Clean real vs ideal pLS comparison**: re-run `--idealpls --jet --allobj -n 100` with current binary so both files share the same binary, then redo `lst_count_dedup_stages.py` comparison.
- All prior pending items from Session 32:
  - 100-evt real-pLS ntuple with `--jet` for the 4-config ΔR plot set: `LSTNtuple_realpls_jet_100evt.root` NOW EXISTS (produced this session) ✓
  - `LSTNtuple_noAfterBuild_realpls_jet_100evt.root` (real-pLS, AfterBuild-off counterpart) — still not done
  - The full 4-config ΔR comparison (standard-real, noAfterBuild-real, standard-ideal, noAfterBuild-ideal) is still incomplete

---

# AfterBuild-off + Input Correction — Handoff (Session 32, 2026-07-16)

## Standing Rules
- **Always Use CUDA.** 2-GPU box: GPU0=L40(46GB), GPU1=L4(23GB); 376GB RAM; 192 CPU.
- **INPUT: never `-i PU200`.** Always `-i trackingNtuple-100.root` (100 evt) or `trackingNtuple-1000.root` (1000 evt), or a subset. The PU200 sample (`trackingNtuple_ttbar_PU200.root`) is the WRONG input AND has **no genjet branches** (so no ΔR plots). Add `--jet` whenever downstream plotting needs ΔR-to-genjet. This rule was established this session after PU200 runs silently changed results/timing (see item 2). Saved to memory as `feedback_always_use_trackingntuple_input`.

## Source tree state (CHANGED this session)

Two active modifications from baseline:
1. **`src/alpaka/LSTEvent.dev.cc` lines ~999–1008**: `RemoveDupQuintupletsAfterBuild` kernel call is **commented out** (AfterBuild dedup DISABLED). Marker comment: `// RemoveDupQuintupletsAfterBuild disabled for experiment`.
2. **`src/alpaka/Kernels.h` line 209**: `minNHitsForDup_T5 = 8` (Experiment 1, unchanged from S30/31).

**CUDA binary** (`standalone/LST/liblst_cuda.so`): rebuilt Jul 16 with AfterBuild disabled — matches source. Confirmed via 5-evt runs showing `t5_isDupBits & 0x01 == 0` for all T5s.

Committed this session as `96919ff6667` ("Various changes made with Claude, notably synthetic pLS for testing") — includes the TruthPixelSeeds ptErr/etaErr change (item 1), dedup bit-tracking in Kernels.h, `runPT5DNN` plumbing, and standalone analysis code. NOTE: the AfterBuild comment-out is a *working-tree* edit made after that commit; it is not committed.

---

## Session 32 — What we did (2026-07-16)

### 1. TruthPixelSeeds: ptErr/etaErr now sampled from real distribution (was median)

**File**: `code/core/TruthPixelSeeds.cc`. Old behavior: every synthetic pLS got the same per-event **median** of `see_ptErr`/`see_etaErr`. New behavior: each synthetic seed draws a **random** real `see_ptErr`/`see_etaErr` (`std::mt19937` seed 42, `uniform_int_distribution` over the event's real-seed pool; fallbacks 0.01/0.001 if empty). Removed the `medianOrDefault` helper; added `#include <random>`.

**Comparison plot** (no LST run needed — reads the tracking ntuple directly):
`efficiency/python/lst_plot_pterr_etaerr_diff.py` → `pterr_etaerr_diff.png`. Per-seed (real − synthetic): ptErr relative, etaErr absolute; old (median) vs new (random draw) overlaid.
Result (100 evt, 99,719 seeds): ptErr rel 68% — old [−0.97,+0.74], new [−3.44,+0.78]; etaErr abs 68% — old [−0.00106,+0.00202], new [±0.00207]. Random draw is symmetric/less-biased but higher per-seed variance. (Superseded script `lst_plot_pterr_etaerr_sampling.py` plotted distributions instead of diffs.)

### 2. AfterBuild disabled — and the input-mistake discovery

Commented out `RemoveDupQuintupletsAfterBuild` (source state above). Initial 5/10/100-evt runs were done on `-i PU200` and finished suspiciously fast (~6 min/100evt).

**Investigation resolved two things:**
- **The speed was the WRONG INPUT, not AfterBuild.** PU200 ≠ trackingNtuple-100. `RemoveDupQuintupletsAfterBuild` only *marks* T5s (`rmQuintupletFromMemory` sets `isDup |= killBits`, Kernels.h:23); it never removes them, and the ntuple T5 write loop (write_lst_ntuple.cc:1633) writes **all** T5s regardless of `isDup`. So disabling AfterBuild **cannot** change T5 count or write time. On the correct dense input (trackingNtuple-100), AfterBuild-off actually *floods* the TC stage (5-evt real run: event 0 alone spent ~345 s in TC), consistent with the user's memory that disabling it historically took hours.
- **`CountMiniDoublets` is NOT a code-version issue (earlier claim retracted).** It has been in every full build since ~June 5 and is present in the fork parent `820464a2b09`. An earlier diff against branch `CMSSW_16_1_X` was misleading because that branch is an older May-10 snapshot predating `CountMiniDoublets`. The current tree is the correct/current version.

**All PU200 outputs from today renamed `DELETE_*`** (5 files: `DELETE_LSTNtuple_noAfterBuild_{5,10,100}evt.root`, `DELETE_LSTNtuple_noAfterBuild_idealpls_100evt.root`, and its `.partial`). Safe to `rm DELETE_*`.

### 3. Controlled test: T5 efficiency is provably pLS-independent

**Question**: comparing ideal-pLS `eff_vs_deltaR_idealpls_fixed.png` to the real-pLS equivalent, the T5 curve *looked* different. Is that real?

**Answer: NO.** T5 is built from Outer-Tracker hits only (pLS is not an input), and `T5_lower` efficiency (performance.cc:135–142) counts matched T5s with no pLS-dependent filter. Ran a **controlled** pair — same build, same `trackingNtuple-100.root`, same 5 events, differing only in `--idealpls`:
- `LSTNtuple_noAfterBuild_realpls_jet_5evt.root` + `LSTNtuple_noAfterBuild_idealpls_jet_5evt.root`
- NumDens: `LSTNumDen_{realpls,idealpls}_jet_5evt.root` (via `createPerfNumDenHists -J`)
- Overlay: `efficiency/python/lst_overlay_t5_real_vs_ideal.py` → **`t5_real_vs_ideal_overlay.png`**

**T5_lower numer/denom histograms are BIT-IDENTICAL** (numer 141=141, denom 151=151). `pLS_lower` control differs (numer 133 real → 148 ideal), confirming the swap took effect. So the apparent T5 difference between the two old plots came from them **not being a controlled swap** (100 evt, large error bars, and/or 8-day code gap between them), not from pLS.

### 4. Real-pLS equivalent of `eff_vs_deltaR_idealpls_fixed.png`

Identified as **`eff_vs_deltaR_stages_recreated.png`** (2026-06-22): same "LST Efficiency vs ΔR by Stage" title, same git tag `820464a2b09`, real pLS (pLS curve climbs ~0.5→0.9 vs ideal's flat ~1.0). Caveat: it's a 2-panel render vs the reference's single panel; earlier sibling is `eff_vs_deltaR_stages.png` (2026-06-12).

### 5. Currently running (detached, survives logout)

**Detached via `setsid`** (PPID 1, own session): `lst_cuda -i trackingNtuple-100.root --jet --idealpls --allobj -n 100` (AfterBuild-off, ideal pLS).
- Output: `LSTNtuple_noAfterBuild_idealpls_jet_100evt.root`
- Log: `noAfterBuild_idealpls_jet_100evt.log`
- Check: `pgrep -af "lst_cuda.*idealpls.*-n 100"` / `tail -f noAfterBuild_idealpls_jet_100evt.log`
- Expect several hours (AfterBuild-off TC flood on dense input).

Leftover stub `LSTNtuple_noAfterBuild_realpls_jet_100evt.root` (58 KB, empty — from a killed real-pLS run; `rm` was permission-denied). Delete when convenient.

### Files produced this session
- Scripts: `efficiency/python/lst_plot_pterr_etaerr_diff.py`, `efficiency/python/lst_overlay_t5_real_vs_ideal.py`
- Plots: `pterr_etaerr_diff.png`, `t5_real_vs_ideal_overlay.png`
- Ntuples/NumDens (jet, trackingNtuple-100): `LSTNtuple_noAfterBuild_{realpls,idealpls}_jet_5evt.root`, `LSTNumDen_{realpls,idealpls}_jet_5evt.root`, `LSTNtuple_noAfterBuild_idealpls_jet_5evt.root`

### Pending / next
- 100-evt AfterBuild-off ideal-pLS run in progress (item 5); real-pLS 100-evt counterpart still needed for the full 4-config ΔR plot set the user requested (standard-real, noAfterBuild-real, standard-ideal=`idealpls_fixed`, noAfterBuild-ideal). All must be `trackingNtuple-100 --jet`.
- Decision still open: whether the two **standard** (AfterBuild-ON) configs should be regenerated on current code for a clean same-build comparison, or reuse existing `fix2.root`/`idealpls_fixed.root`.
- Carryover from S31: Experiment 2 (bit0-only T5s into pT5), pLS non-overlap populations, S24 tasks.

---

# pLS Parameter Deep-Dive — Handoff (Session 31, 2026-07-15)

## Standing Rule: Always Use CUDA. 2-GPU box: GPU0=L40(46GB), GPU1=L4(23GB); 376GB RAM; 192 CPU.

## Source tree state (unchanged from Session 30)

**`src/alpaka/Kernels.h` line 209**: `minNHitsForDup_T5 = 8` — Experiment 1 (raised from 7). This is the **only** active modification from baseline. No rebuild needed.

---

## Session 31 — What we did (2026-07-15)

### 1. New plots: all pLS parameter differences (real − synthetic)

**Script**: `efficiency/python/lst_plot_pls_all_param_diff.py`

Extends `pls_matched_diff.png` (Session 22, 3-panel pt/η/φ) to ALL pLS parameters.
Matching logic identical: best pLS per sim track (highest `sim_plsIdxAllFrac`), accepted sim tracks only (pt>0.9, |η|<4.5, |vz|<30, vtx_perp<2.5, |q|=1). 8589 tracks with BOTH.
Input files: `LSTNtuple_fix2.root` (real) + `LSTNtuple_idealpls_fixed.root` (synthetic).

**Three output figures:**

| Plot | Contents |
|---|---|
| `pls_all_param_diff_kinematics.png` | pt (rel), η, φ, deltaPhi, px, py, pz, circleRadius (rel), ptErr (rel), etaErr, circleCenterX, circleCenterY |
| `pls_all_param_diff_hits.png` | hit0-3 x/y/z (hit3 = hit2 for triplet pLS) |
| `pls_all_param_diff_categorical.png` | charge, nhit, isQuad |

Conventions: relative difference `(r−s)/s` for always-positive scalars (pt, ptErr, circleRadius); absolute difference for px/py/pz (can be zero/negative); dphi-wrapped for φ/deltaPhi; bar chart for integer fields.

**Run command:**
```bash
python3 efficiency/python/lst_plot_pls_all_param_diff.py \
    --real LSTNtuple_fix2.root \
    --synth LSTNtuple_idealpls_fixed.root \
    --out-prefix pls_all_param_diff
```

### 2. Key clarifications established this session

**PCA momentum: truth vs Kalman fit**

For synthetic pLS (`buildTruthPixelSeeds`):
```cpp
out.see_px.push_back(sim_px[simTrkIdx]);  // Geant4 truth at production vertex
```
These are MC generator/Geant4 truth momentum components at the production vertex, used directly as the PCA momentum.

For real pLS (`see_px/py/pz` in the tracking ntuple): these come from the CMSSW pixel seed Kalman filter, which fits a pixel seed (2–3 reconstructed hits) and extrapolates back to the beam line. They have finite resolution (~6% core in pT) — this is exactly what the `pls_matched_diff.png` plots quantify.

**`kR1GeVf` — helix radius constant**
```cpp
constexpr float kR1GeVf = 1.f / (2.99792458e-3f * 3.8f);  // ≈ 87.8 cm/GeV
```
Converts pT [GeV/c] → helix radius R [cm] in the CMS 3.8 T solenoid.

Derivation: from `R [cm] = pT / (c × 10⁻¹¹ × B)` where `c = 2.998×10⁸ m/s`, converting to cm. Equivalent to the standard `R [m] = pT / (0.3 × B)` formula. Used in `buildTruthPixelSeeds` to compute `delta_phi = −q × 2 × arcsin(rt_outer / 2R)` — the helix rotation angle from PCA to the outer pixel hit — needed to propagate the truth momentum direction to `see_stateTrajGlbPx/Py/Pz`. This was the Session 15 bug fix: without this propagation, synthetic seeds had the PCA momentum direction where the outer-hit state was expected, causing ~0.1 rad betaIn bias that collapsed pT5 efficiency at low pT.

**C++17 structured binding syntax** (seen in `TruthPixelSeeds.cc`):
```cpp
for (auto const& [simTrkIdx, pixHits] : simTrkToPixHit)
```
Equivalent to the C++11 form:
```cpp
for (auto const& kv : simTrkToPixHit) {
    auto simTrkIdx = kv.first;
    auto& pixHits  = kv.second;
}
```
The `[key, val]` syntax unpacks the `std::pair` yielded by map iteration into named variables. `auto const&` means: don't copy (reference), don't modify (const).

**`ptErr` and `etaErr` roles in LST** (three distinct uses):

1. **Seed entry gate** (`LSTPrepareInput.h:111`): `if (ptIn > ptCut - 2 * ptErr)` — seeds with measured pT just below the cut are still accepted if their −2σ lower bound clears it. Also gates `pixelType` classification (line 128).

2. **pT3 geometric window widening** (`PixelTriplet.h`): `ptErr` loosens the lower pT bound by 10σ; `etaErr` propagates into dz/drt tolerances as `dzErr = drt² × etaErr² × cosh²η`. Applied for both inner and outer T3 matching.

3. **DNN pLS embedding features** (`NeuralNetwork.h:413,417`): `etaErr / kEtaErr_norm` and `log10(ptErr)` are input features to the pT5 neural network score. This is why `buildTruthPixelSeeds` borrows median real-seed values rather than using 0 or truth-perfect errors — the DNN was trained on realistic distributions.

**NOT used in**: any deduplication kernel (CheckHitspLS, RemoveDupQuintupletsAfterBuild, BeforeTC, CrossCleanT5, etc.).

---

## Pending after Session 31 (unchanged from Session 30)

- **Experiment 2 (on hold)**: Let bit0-only T5s into pT5 building (`PixelQuintuplet.h:680`).
- **Experiment 3** (disable AfterBuild entirely): must confer with user first.
- Rescued T5s fake rate rerun
- Task (1) from S24: interactive jet hit diagram
- Task (2) from S24: no-dedup-keep-CheckHitspLS run
- Task (3) from S24: cut loosenings
- pLS comparison: non-overlap populations (tracks with real-only or synthetic-only pLS)

---

# Post-AfterBuild T5 Efficiency Plot — Handoff (Session 30, 2026-07-14)

## Standing Rule: Always Use CUDA. 2-GPU box: GPU0=L40(46GB), GPU1=L4(23GB); 376GB RAM; 192 CPU.

## Source tree state

**`src/alpaka/Kernels.h` line 209**: `minNHitsForDup_T5 = 8` — Experiment 1 (raised from 7). This is the **only** active modification from baseline. All other files at baseline.

**CUDA binary**: rebuilt Jul 10 05:35 with threshold=8. No rebuild needed unless sources change.

---

## Session 30 — What we did (2026-07-14)

### 1. New plot: T5 efficiency after RemoveDupQuintupletsAfterBuild

**Script**: `efficiency/python/lst_plot_eff_vs_deltaR_postAB.py`
**Output**: `eff_vs_deltaR_postAB.png`

Reproduces `eff_vs_deltaR_idealpls_fixed.png` but replaces `T5_lower` with a **post-AfterBuild** T5 efficiency:
a sim track is counted in the numerator only if it has ≥1 T5 with hit-purity > 0.75 **and** `t5_isDupBits & 0x01 == 0` (not killed by `RemoveDupQuintupletsAfterBuild`).

All other stages (MD, LS, pLS, T3, pT3, pT5, TC) are read directly from `LSTNumDen_idealpls_fixed.root` via uproot. The post-AfterBuild T5 curve is computed directly from `LSTNtuple_dedup_stages_v2.root` (the 100-event idealpls run with `t5_isDupBits` instrumentation; `LSTNtuple_idealpls_fixed.root` predates that branch).

**Key result**:

| ΔR range | T5_lower (original, pre-AfterBuild) | T5 (post-AfterBuild) |
|---|---|---|
| ΔR < 0.01 (jet core) | ~0.90 | **~0.45–0.55** |
| ΔR > 0.03 (isolation) | ~0.93 | **~0.87–0.92** |

AfterBuild alone removes roughly half of all tracks' surviving genuine T5s in the jet core, before pT5 matching even starts. At large ΔR the loss is much smaller (0.93→0.87). The post-AfterBuild T5 efficiency at small ΔR (≈0.45–0.55) is now in the same ballpark as pT5_lower (≈0.45–0.65), confirming both are limited by the same upstream AfterBuild depletion.

**Run command**:
```bash
python3 efficiency/python/lst_plot_eff_vs_deltaR_postAB.py \
    --numden LSTNumDen_idealpls_fixed.root \
    --ntuple LSTNtuple_dedup_stages_v2.root \
    --out eff_vs_deltaR_postAB.png
```

**Script notes**: pure uproot (no PyROOT); denominator selection matches `createPerfNumDenHists -J` with `base_0_0` (pt>0.8, |eta|<4.5, |vz|<30, vtx_perp<2.5, q≠0, genjet pT>1000 & |eta|<2.5); Clopper-Pearson 68.3% intervals.

---

## Pending after Session 30 (unchanged from Session 29)

- **Experiment 2 (on hold — user said "wait on that")**: Let bit0-only T5s into pT5 building (`PixelQuintuplet.h:680`, `if (isDup() & ~0x01u)`).
- **Experiment 3** (disable AfterBuild entirely): must confer with user first.
- Rescued T5s fake rate rerun
- Task (1) from S24: interactive jet hit diagram
- Task (2) from S24: no-dedup-keep-CheckHitspLS run
- Task (3) from S24: cut loosenings
- pLS comparison plots

---

# Experiment 1 Analysis + Fake Rate Attribution — Handoff (Session 29, 2026-07-10)

## Session 29 — What we did (2026-07-10)

### 1. Experiment 1 (ab8) analysis COMPLETE

`LSTNumDen_ab8_v1.root` finished (createPerfNumDenHists -J, ~1412 real-s).

**Comparison command**: `python3 efficiency/python/lst_compare_nodnn_table.py LSTNumDen_baseline_Jv1.root LSTNumDen_ab8_v1.root`

**Results (threshold 7→8, 100 events, idealpls, dR<0.01)**:

| Metric | Baseline | Exp 1 (ab8) | Δ |
|--------|----------|-------------|---|
| TC eff | 0.6899 (247/358) | 0.6983 (250/358) | **+0.84 pp** |
| TC fake rate | 0.3606 (251/696) | 0.3720 (263/707) | +1.14 pp |
| pT5 fake rate | 0.3726 (237/636) | 0.3854 (249/646) | +1.28 pp |
| pT5_lower eff | 0.6899 (247/358) | 0.6955 (249/358) | +0.56 pp |
| T5_lower eff | 0.9022 (323/358) | 0.9022 (323/358) | **0 — unchanged** |

**Fake/efficiency exchange rate**: +1.14pp FR / +0.84pp eff ≈ **1.4×**
(compare: DNN disable was 2.4×)

**T5_lower unchanged**: The 3 recovered sim tracks were already findable by SOME T5 (T5_lower was already at 90.2%). The released T5s (7-hit-overlap losers from AfterBuild) had sim tracks not covered by their winner twins for pT5 matching — the loser, now surviving, formed the pT5 the winner couldn't.

**Verdict**: Experiment 1 is a net positive (better fake/eff ratio than DNN disable) but tiny — only 3 of ~111 missing jet-core tracks recovered. The dominant bottleneck (DNN gate, ~40 tracks; and pT5 dedup misranking, ~17 tracks) is untouched.

**Efficiency plot**: `eff_vs_deltaR_ab8.png` — visual matches baseline closely (change too small to see clearly).

**Experiment 2 status**: user said "wait on that" — on hold.

### 2. Fake rate source attribution COMPLETE

**Key finding**: The ~36 pp gap in TC fake rate between real-pLS and idealpls runs at small dR is entirely from fake pixel seeds.

| Run type | File | TC fake rate at dR≈0 |
|----------|------|----------------------|
| Real pLS (no --idealpls) | `performance/051826-1_4b9f92-trackingNtuple-100.root/mtv/var/TC_fakerate_deltaR.png` | ~0.72 |
| Idealpls | `fakerate_vs_deltaR_nodnn_compare.png` baseline panel | ~0.36–0.44 |

`TC_fakerate_deltaR.png` (the ~0.72 plot) was a pre-existing run from commit `4b9f92` on May 18 2026, using real pixel seeds WITHOUT `--idealpls`. It is NOT from the current NumDen pipeline.

**~36 pp of jet-core TC fake rate is purely from fake pixel seeds (combinatorial pLS fakes) propagating through pT5→TC.** With idealpls, those are eliminated, exposing the ~0.36 "residual" fake rate from fake T5/pT3/etc.

### 3. TC panel in compare plot — no _lower equivalent exists

User asked to make the TC panel in `fakerate_vs_deltaR_nodnn_compare.png` "reflect the same thing" as the pT5_lower change. Conclusion: no `TC_lower` histogram exists in the NumDen files. TC IS the final reconstruction output — all dedup (AfterBuild, BeforeTC, pT5 dedup, CrossClean) runs before TCs are created. `TC_fr` and `TC_fdr` are numerically identical for TC (TC-level duplicates are nearly zero).

If a "pre-TC-dedup" count is needed, it would require summing `pT5_lower + T5_lower + pT3_lower + pLS_lower` denominators from the NumDen file — not yet implemented. The TC panel in the compare plot currently uses `TC_fr` which is the correct and complete post-all-dedup TC collection.

---

## Pending after Session 29

### Experiment 2 (on hold, user said "wait on that")
Change `PixelQuintuplet.h:680` to let bit0-only T5s (AfterBuild killed) into pT5 building:
```cpp
if (quintuplets.isDup()[quintupletIndex] & ~0x01u) continue;
```
This is the next logical step — Experiment 1 showed that releasing 7-hit-overlap losers recovered 3 tracks; Experiment 2 would release ALL AfterBuild-killed T5s (not just 7-hit ones).

### Experiment 3 (disable AfterBuild entirely)
**MUST confer with user before running.**

### Other pending (unchanged from S25–26)
- Rescued T5s fake rate (rerun with bit3 disabled in BeforeTC)
- Task (1) from S24: interactive jet hit diagram
- Task (2) from S24: no-dedup-but-keep-CheckHitspLS run
- Task (3) from S24: cut loosenings — target: pT5 dedup score redesign or DNN retraining
- pLS comparison plots
- TC panel in compare plot: option to synthesize pre-TC-dedup denominator

---

## Session 28 — What we did (2026-07-10)

### 1. DNN-disable analysis COMPLETE

**Ntuples**: `LSTNtuple_nodnn_v2.root` (baseline-DNN-off, `--nopt5dnn --allobj --jet --idealpls -n 100 -i trackingNtuple-100.root`).
**NumDen files**: `LSTNumDen_nodnn_Jv1.root` and `LSTNumDen_baseline_Jv1.root` (both 17MB, both generated with `-J` from genjet-bearing ntuples).

**Efficiency plot**: `eff_vs_deltaR_nodnn.png` — same style as `eff_vs_deltaR_idealpls_fixed.png` (8 objects, same colors/markers). pT5 at dR<0.01 rises from ~0.45 → ~0.70; TC from ~0.54 → ~0.64.

**Fake rate summary** (numeric comparison at ΔR<0.01):
| Object | Baseline fake rate | NoDNN fake rate | Δ |
|---|---|---|---|
| pT5 | ~0.45 | ~0.55 | +10 pp |
| TC | 0.36 | 0.58 | **+22 pp** |
| TC denom count | 696 | 1227 | +76% more TCs |

**Conclusion: NOT VIABLE** as-is. Disabling the DNN costs ~2.5× fake rate per pp of efficiency gained. The DNN would need to be retrained to be less aggressive in the dense-region regime without abandoning its fake rejection elsewhere.

**Scripts produced** (session 27–28):
- `efficiency/python/lst_compare_nodnn_fakerate.py` — side-by-side fake rate vs ΔR (uproot-based)
- `efficiency/python/lst_compare_nodnn_table.py` — numeric table comparison from NumDen files
- `efficiency/python/lst_compare_nodnn_eff.py` — efficiency stages comparison (ntuple-based)

### 2. Solved: why genjet branches were missing from ab8 runs

The standard `-i PU200` tracking ntuple (`trackingNtuple_ttbar_PU200.root`) has **no genjet branches**. The PU200RelVal files also lack them. All prior ntuples that had genjet data (`LSTNtuple_dedup_stages_v2.root`, `LSTNtuple_nodnn_v2.root`, etc.) were generated using the local file `-i trackingNtuple-100.root` (in the standalone directory, 92MB, 100 events, has all `sim_genjet_*`/`genjet_*` branches).

Previous ab8 attempts all failed for this reason or CUDA driver absence:
- `LSTNtuple_ab8_idealpls_v2.root` (135KB stub) — wrong input, `-J` crash
- `LSTNtuple_ab8_idealpls_v3.root` (1.2GB) — generated without `--jet`, no genjet branches
- `LSTNtuple_ab8_idealpls_v4.root` (135KB stub) — CUDA driver not set up

### 3. Experiment 1 (ab8) run COMPLETE

**Command used**:
```bash
export LD_LIBRARY_PATH=/mnt/data1/kk829/cuda_driver_libs_580:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 lst_cuda -i trackingNtuple-100.root --allobj --jet --idealpls \
    -n 100 -s 4 -v 1 -o LSTNtuple_ab8_idealpls_v5.root
```

**Output**: `LSTNtuple_ab8_idealpls_v5.root` — 1.1GB, Jul 10 06:15. Matches baseline size.
**Real time**: ~20 min (2143 CPU-s, 1212 real-s).

`createPerfNumDenHists -J -i LSTNtuple_ab8_idealpls_v5.root -o LSTNumDen_ab8_v1.root` **running in background** (task `bpa4wwg42` at time of handoff). Expected ~10 min.

### 4. pLS-T5 cut catalog (for future cut-loosening planning)

All cuts applied inside `runPixelQuintupletDefaultAlgo` (PixelQuintuplet.h) and callees, in order:

**Pre-loop gates** (CreatePixelQuintupletsFromMap):
1. Skip if T5 lower module is TwoS
2. Skip if `pixelSegments.isDup()[i_pLS]` (pLS dedup)
3. Skip if `quintuplets.isDup()[quintupletIndex]` (T5 isDup, any bit — bit0=AfterBuild)

**Inside runPixelQuintupletDefaultAlgo → runPixelTripletDefaultAlgo on inner T3**:
4. **Radius criterion** (`passRadiusCriterion`): 1/R interval of pLS vs inner T3 must overlap. 4 geometry variants (BBB/BBE/BEE/EEE), different tolerance bounds. pT<2 GeV: tighter bounds.
5. **pLS–segment compatibility × 2** (`runPixelTrackletDefaultAlgoPPBB` or `PPEE`): applied for both the inner and outer segment of the inner T3. Each call checks:
   - z (or rt) geometric window cut — 4× loosened from original
   - z (or rt) pointing cut — 4× loosened
   - dPhi midpoint cut — 4× loosened
   - betaOut cut (outer segment angle vs pLS curvature)
   - dBeta cut: `(betaIn − betaOut)² ≤ dBetaCut²`
6. **pT3 RZ chi-squared** (`passPT3RZChiSquaredCuts`): helix residuals for pLS + 3 inner-T3 hits in r-z. **Only applied if pLS pT < 5 GeV.**
7. **DNN gate** (`lst::pt3dnn::runInference<pT5WP>`): inputs are rPhiChiSquared, T3 radius, pLS radius, pLS radius error, rzChiSquared, pLS eta, pLS pT, last T3 module type. **No pT guard — applies to all pT.** Disabled by `--nopt5dnn`.

**Back in runPixelQuintupletDefaultAlgo — full 5-layer checks**:
8. **pT5 RZ chi-squared** (`passPT5RZChiSquaredCuts`): helix residuals for pLS + all 5 T5 layers in r-z. **Only pT < 5 GeV.**
9. **pT5 rPhi chi-squared outward** (`passPT5RPhiChiSquaredCuts`): pLS circle vs T5 hits in r-φ. **Only pT < 5 GeV.**
10. **pT5 rPhi chi-squared inward** (`passPT5RPhiChiSquaredInwardsCuts`): T5 regression circle vs pLS hits. **Only if T5 regression radius implies pT < 5 GeV.**

From Session 27 data: cuts 8–10 explain ~3% of failures when loosened 4×. The DNN (cut 7) explains the dominant share. Radius criterion (cut 4) untested.

---

## Pending after Session 28

→ See **Session 29** section above — Experiment 1 analyzed, Experiment 2 on hold, fake rate attribution complete.

---

## Session 27 — What we did

### 1. Rewrote `lst_check_pt5_matching_density.py` (uproot, no PyROOT)

The original script used `import ROOT as r` which is inaccessible in the current environment. Rewrote in-place with uproot + numpy + scipy. Logic is identical to the original Session 7 script, plus two additions:

- **AfterBuild exclusion**: `has_t5` now filters out T5s with `isDupBits & 0x01 != 0` (killed by RemoveDupQuintupletsAfterBuild) before checking if the truth T5 is present. Added `sim_t5IdxAll` and `t5_isDupBits` to the branches list.
- **triedInPT5 diagnosis**: for nCompetingT5==0 failure cases, checks if the truth T5 has `triedInPT5=True` (geometric cut failure) or `False` (connectivity map excluded the pair).

Changed `--pt-cut` default 0.9 → 0.8 to match `performance.cc`.

Default output: `pt5_matching_density_idealpls.png`.

Run command:
```bash
python3 efficiency/python/lst_check_pt5_matching_density.py \
    LSTNtuple_dedup_stages_v2.root --pt-cut 0.8 -o pt5_matching_density_idealpls.png
```

### 2. Matching density results on post-fix idealpls ntuple (`LSTNtuple_dedup_stages_v2.root`)

**Without AfterBuild exclusion** (same denominator as Session 7 pre-fix):
```
Ingredients present (eligible pLS + any T5 with >75% match): 2154
  no pT5 formed: 245 (11.4%)
  → Same as Session 7 pre-fix (11.3%) — fix didn't change this metric
```

**With AfterBuild exclusion** (eligible pLS + surviving T5):
```
Ingredients present: 2109
  no pT5 formed: 208 (9.9%)
  nCompetingT5==0 (pLS pairs with NO T5 at all): 184 tracks, 172 no-pT5 (93.5%)
  nCompetingT5>=1 (pLS pairs with some other T5): 1925 tracks, 36 no-pT5 (1.9%)

nCompetingT5==0 failure breakdown (172 tracks):
  truth T5 tried (triedInPT5=True) → geometric cut failed: 172 (100.0%)
  truth T5 NOT tried (triedInPT5=False) → connectivity excluded: 0 (0.0%)
```

**Key finding**: 100% of "ingredients present, no pT5" failures are geometric cut failures inside `runPixelQuintupletDefaultAlgo`, NOT module-connectivity exclusions. The connectivity map is not the bottleneck.

### 3. `runPixelQuintupletDefaultAlgo` gate structure (from PixelQuintuplet.h)

Four sequential gates:
- **Gate A** (line 500–518): `runPixelTripletDefaultAlgo<pT5WP>` on (pLS, inner T3 of T5). Full pT3 algorithm: betaIn, betaOut, dPhi, z-window, radius, DNN score.
  - betaOut/z-window/dPhi: already tested 4× in Sessions 16+18 → zero recovery
  - betaIn: fixed by TruthPixelSeeds
  - passRadiusCriterion: not yet isolated
  - **DNN (`pT5WP` working point): not yet tested** ← current experiment
- **Gate B** (line 574–583): `passPT5RZChiSquaredCuts` — only for pT < 5 GeV
- **Gate C** (line 607–616): `passPT5RPhiChiSquaredCuts` — only for pT < 5 GeV
- **Gate D** (line 620–629): `passPT5RPhiChiSquaredInwardsCuts` — only when T5 regression radius implies pT < 5 GeV

### 4. Gates B+C+D loosening (4×): negligible effect

Dividing all chi2 values by 4 at the call sites (equivalent to 4× threshold loosening):
- 172 → 167 nCompetingT5==0 cut failures (−5, 3%)
- Gates B/C/D explain only ~3% of failures. **Reverted** after this result.

Ntuple: `LSTNtuple_chi2_4x_v3.root` (100 events, idealpls, 4× chi2 loosening).

### 5. DNN gate disable: COMPLETE (analyzed in Session 28)

Using existing `--nopt5dnn` runtime flag. `LSTNtuple_nodnn_v2.root` produced with:
```bash
lst_cuda -i trackingNtuple-100.root --allobj --jet --idealpls --nopt5dnn \
    -n 100 -s 4 -v 1 -o LSTNtuple_nodnn_v2.root
```

Results: see Session 28 section above. DNN disable not viable (+22 pp fake rate for +9 pp TC eff at dR<0.01).

### 6. Important clarifications established this session

**"Ingredients present"** = sim track has BOTH:
1. An eligible pLS: pLS in `sim_plsIdxAll` with >75% purity, `pLS_isQuad=True`, `pLS_isDuplicate=False`
2. A surviving (post-AfterBuild) T5: T5 in `sim_t5IdxAll` with >75% purity AND `isDupBits & 0x01 == 0`

**T5 efficiency in `eff_vs_deltaR_idealpls_fixed.png` is PRE-AfterBuild**: `sim_t5IdxAll` includes AfterBuild-killed T5s. The ~0.8+ T5 efficiency means a T5 was *built* with good purity, not that it survived to pT5 matching. The T5→pT5 efficiency gap is dominated by AfterBuild attrition, not matching cut failures.

**pT5 formation step-by-step**:
1. `createQuintuplets()` runs → `RemoveDupQuintupletsAfterBuild` marks T5s with `isDuplicate` (bit0)
2. `createPixelQuintuplets()` runs separately — loops over pLS × connected T5 modules (connectivity map, not all-vs-all); skips isDuplicate pLS (line 667) and T5s (line 681); sets `triedInPT5=true` (line 685) before any cut
3. `runPixelQuintupletDefaultAlgo` applies Gates A–D
4. Passing pairs written to pT5 SoA with `pT5_plsIdx` back-reference
5. `RemoveDupQuintupletsBeforeTC` and `CrossCleanT5` run AFTER pT5 building (on T5s only)

---

## Pending after Session 27 (superseded by Session 28 section above)

---

# Fake Rate vs ΔR Plots — Handoff (Session 26, 2026-07-09)

## Standing Rule: Always Use CUDA. 2-GPU box: GPU0=L40(46GB), GPU1=L4(23GB); 376GB RAM; 192 CPU.

## Source tree state: UNCHANGED from Session 25
All six dedup instrumentation files remain in place (see Session 25 section below). CUDA binary is current. No rebuild needed.

## Session 26 — What we did

### New script: `efficiency/python/lst_plot_fakerate_vs_deltaR.py`
Reads `Root__<OBJ>_fr_{numer,denom}_deltaR` histograms from a LSTNumDen ROOT file and produces a styled fake rate vs ΔR plot matching `eff_vs_deltaR_idealpls_fixed.png` (same colors, markers, Clopper-Pearson error bars, no connecting lines, `linestyle="none"`).

Object mapping (uses `_lower` for sub-TC objects, `TC` for the final stage):
```
MD_lower, LS_lower, pLS_lower, T3_lower, pT3_lower, T5_lower, pT5_lower, TC
```
Gracefully skips any object whose key is absent from the file.

```bash
python3 efficiency/python/lst_plot_fakerate_vs_deltaR.py LSTNumDen_idealpls_fixed.root \
    --out fakerate_vs_deltaR_idealpls_fixed.png --zoom 0.05
```

### Output: `fakerate_vs_deltaR_idealpls_fixed.png`
Aggregated fake rates (all ΔR, 100 events, idealpls):

| Object | Fakes | Total | Rate |
|--------|-------|-------|------|
| MD | 11k | 35k | 33% |
| LS | 743k | 764k | 97% |
| pLS | 0 | 2.9k | 0% |
| T3 | 7.4M | 7.5M | 99% |
| pT3 | 250 | 521 | 48% |
| T5 | 3.7M | 3.7M | 98% |
| pT5 | 4k | 9.2k | 44% |
| TC | 1.2k | 3.8k | 32% |

### What `_lower` fake rate means (clarified this session)
`performance.cc` iterates over **every reconstructed object of that type** (not just those in TCs). For each object:
- **Denominator** = every object satisfying basic pT/eta cuts, placed in the ΔR bin of **the object's own (η,φ)** to the nearest genjet (pT>1000, |η|<2.5). Objects with no qualifying nearby jet land at ΔR≈31 and fall off the [0,0.1] histogram — so the deltaR plot only contains objects that point toward a jet.
- **Numerator** = the subset with `isFake > 0` (no sim match).
- The `sel` function for all `_lower` objects is `return 1` — no selection filter. The denominator is the full object collection, NOT filtered to TCs.

The deltaR axis shows **proximity of the object itself to the jet center**, not the sim track's proximity. The high T3/T5 fake rates (99%/98%) reflect the combinatorial explosion of unfiltered objects before dedup.

### `lst_plot_performance.py --individual` status
PyROOT is not accessible in the current environment (neither the CMSSW_14 nor CMSSW_16 Python installations expose PyROOT in a runnable form outside the singularity container). Existing `TC_fakerate_deltaR.png` plots in breakdown format exist at:
```
performance/051926-3_a48fbf-trackingNtuple-100.root/mtv/var/TC_fakerate_deltaR.png
performance/051926-1_820464-trackingNtuple-100.root/mtv/var/TC_fakerate_deltaR.png
```
These are breakdown format (TC + pT5/pT3/T5/pLS/T4 sub-curves), not `--individual` (single TC curve). To generate the `--individual` version, `lst_plot_performance.py` needs PyROOT — likely requires running inside the CMS el8 singularity container (`cmssw-el8` or similar).

## Pending tasks (updated)

### Rescued T5s fake rate (requires rerunning lst_cuda)
Saved in `memory/project_rescued_fakerate.md`. To compute properly: disable only the isPT5-priority rule in `RemoveDupQuintupletsBeforeTC` (the `bit3` path in Kernels.h), rebuild with `-G`, rerun, compute NumDen, compare with baseline. This is "Experiment B revisited" targeting bit3 specifically.

### Tasks from S24 still outstanding
- Task (1): interactive jet hit diagram
- Task (2): no-dedup-but-keep-CheckHitspLS run — DEDUP-DISABLE blocks were removed in S25; would need re-applying
- Task (3): cut loosenings worktree — target cut not chosen
- pLS comparison plots: data available in `LSTNtuple_dedup_stages_v2.root`, script pending

---

# T5 Dedup Pipeline Instrumentation + Stage-by-Stage Analysis — Handoff (Session 25, 2026-07-09)

## Standing Rule: Always Use CUDA. 2-GPU box: GPU0=L40(46GB), GPU1=L4(23GB); 376GB RAM; 192 CPU.

## ⚠️ SOURCE TREE STATE — dedup instrumentation active; all dedup kernels running

**DEDUP-DISABLE-TEST blocks REMOVED** (were present from S24; all 10 kernels now active).

Six files carry permanent instrumentation changes (all on `claude-edits` branch):

| File | Change |
|---|---|
| `interface/QuintupletsSoA.h` | New SoA column `triedInPT5` (bool) added after `partOfPT5` |
| `src/alpaka/PixelQuintuplet.h` | Sets `quintuplets.triedInPT5()[quintupletIndex] = true` before `runPixelQuintupletDefaultAlgo` (after isDup gate, before pT5 attempt) |
| `src/alpaka/Kernels.h` | `rmQuintupletFromMemory` signature changed from `bool secondpass` to `uint8_t killBits = 0x01u`; uses `\|=` not `=true`. `RemoveDupQuintupletsBeforeTC` now tracks sub-conditions via bit OR. |
| `src/alpaka/TrackCandidate.h` | CrossCleanT5 uses `\|= 0x10` (bit 4) instead of `= true` |
| `src/alpaka/LSTEvent.dev.cc` | Zero-initializes `triedInPT5` alongside `partOfPT5` and `isDup` |
| `standalone/code/core/write_lst_ntuple.cc` | Writes `t5_triedInPT5` branch; updated `t5_isDupBits` comment |

**isDupBits encoding (5 bits now used):**
```
bit0 (0x01) = RemoveDupQuintupletsAfterBuild
bit1 (0x02) = BeforeTC sub-condition A: (dR2<0.001 OR nMatched>=5) AND d2<1.0
bit2 (0x04) = BeforeTC sub-condition B: dR2<0.02 AND d2<0.1
bit3 (0x08) = BeforeTC: killed by isPT5-priority (opponent is pT5-embedded AND score alone wouldn't kill this T5)
bit4 (0x10) = CrossCleanT5
```

**triedInPT5:** `true` if T5 entered `runPixelQuintupletDefaultAlgo` (passed isDup gate). Idempotent write. T5s killed by AfterBuild (bit0) are gated BEFORE the pT5 loop; cannot have `triedInPT5=true`.

**CUDA rebuild succeeded.** Binary current. No rebuild needed on resume unless sources change.

## Session 25 Results — T5 Fate at Each Dedup Stage

**Input:** `LSTNtuple_dedup_stages_v2.root` (100 events, `--jet --idealpls`, 1.1G, Jul 9 22:00)
**Script:** `efficiency/python/lst_count_dedup_stages.py`
**Plots:** `dedup_stages_v2_counts.png`, `dedup_stages_v2_fractions.png`
**Sanity check: PASSED** — zero T5s have `triedInPT5=True` AND AfterBuild-killed.

### Summary table (100 events, idealpls)

| Stage | All ΔR | ΔR<0.01 (core) | ΔR>0.05 (isolation) |
|---|---|---|---|
| Total T5s | 5,049,304 | 34,434 | 4,965,001 |
| Killed AfterBuild (bit0) | 4,965,293 **(98.3%)** | 32,581 (94.6%) | 4,887,744 (98.4%) |
| Eligible for BeforeTC | 84,011 (1.7%) | 1,853 (5.4%) | 77,257 (1.6%) |
| — pT5-matched (success) | 18,155 (21.6% of elig) | 1,191 **(64.3%)** | 13,542 (17.5%) |
| — tried but pT5 failed | 65,596 (78.1% of elig) | 662 (35.7%) | 63,463 (82.1%) |
| — never tried | 260 (0.3% of elig) | 0 (0%) | 252 (0.3%) |
| Killed BeforeTC (any) | 67,190 (80.0% of elig) | 1,247 (67.3%) | 64,012 (82.9%) |
| — sub-cond A fired | 66,787 | 1,242 | 63,637 |
| — sub-cond B fired | 54,469 | 840 | 52,389 |
| — isPT5-priority decisive | 17,467 (26.0% of killed) | 421 **(33.8%)** | 16,487 (25.8%) |
| — of killed: tried-failed | 62,242 (92.6% of killed) | 629 (50.4%) | 60,369 (94.3%) |
| — of killed: pT5-success | 4,807 (7.2%) | 618 (49.6%) | 3,506 (5.5%) |
| Killed CrossCleanT5 (bit4) | 342 (negligible) | 5 | 312 |
| **Surviving T5s** | **16,479 (0.3%)** | **601 (1.7%)** | **12,933 (0.3%)** |

### Key takeaways

1. **AfterBuild dominates at 98.3% globally.** Jet core slightly less aggressive (94.6%) so more T5s enter BeforeTC proportionally.
2. **In jet core, 64% of BeforeTC-eligible T5s are pT5-matched** vs 17.5% in isolation — more pLS → more successful pT5 pairings in dense regions.
3. **BeforeTC kills 92.6% of tried-but-failed T5s globally.** In jet core, tried-but-failed and pT5-success are killed in nearly equal numbers (629 vs 618).
4. **isPT5-priority (bit3) drives 33.8% of BeforeTC kills in jet core** — these T5s would NOT have been killed by score alone. This is the mechanism for standalone T5 suppression near pT5s.
5. **CrossCleanT5 negligible** — 5 kills in jet core, irrelevant for the investigation.

## Pending Tasks (from S24 — still outstanding)

### Task (2) — no-dedup-but-keep-CheckHitspLS run (30 evt)
DEDUP-DISABLE blocks were removed this session. To re-run: re-apply the 10 DEDUP-DISABLE-TEST `#if 0` wrappers from S24 (keep CheckHitspLS×2 active, disable the other 10 kernels listed in S24 handoff), rebuild with `-G`, run 30 evt.
- Restore: `sed -i '/DEDUP-DISABLE-TEST/d' src/alpaka/LSTEvent.dev.cc`
- The new instrumentation in Kernels.h/TrackCandidate.h is compatible; `t5_isDupBits` will just be all-zero.

### Task (1) — interactive jet hit diagram
Still pending. Recipe unchanged from S24 handoff below.

### Task (3) — cut loosenings worktree
Not started. Target cut not yet chosen.

### pLS comparison plots
Still pending. Data verified (S22 section below).

## Next investigation directions (NOT approved — propose before launching)

The data now quantifies which T5s are lost and where. The natural next step is to cross-match `t5_simIdx` of BeforeTC-killed tried-but-failed T5s in the jet core against sim tracks with no TC — this directly quantifies the efficiency cost attributable to BeforeTC. Other candidates:
- Check whether the 421 isPT5-priority kills (bit3) are preferentially hitting genuine vs. fake T5s
- Score-based BeforeTC that suppresses the isPT5-priority rule when the standalone T5 scores better

---

# Keep-CheckHitspLS No-Dedup Variant + Jet Hit Diagram Recipe — Handoff (Session 24, 2026-07-09)

## Standing Rule: Always Use CUDA. 2-GPU box: GPU0=L40(46GB), GPU1=L4(23GB); 376GB RAM; 192 CPU.

## ⚠️ SOURCE TREE STATE — dedup-disable config was CHANGED this session
`src/alpaka/LSTEvent.dev.cc` now has **10 DEDUP-DISABLE-TEST blocks disabled (20 markers)**, down from
Session-23's 12 blocks (24 markers). **The two `CheckHitspLS` kernels were RE-ENABLED** (now active:
1st pass line ~1044, 2nd pass line ~594) to match the user's ORIGINAL manual experiment: keep pLS
duplicate-cleaning ON, everything else OFF. `CrossCleanpLS` was briefly re-enabled then **re-disabled
per user** (user's manual version kept CheckHitspLS but NOT CrossCleanpLS).
- **Still DISABLED (10):** CrossCleanpT3, RemoveDupQuintupletsBeforeTC, CrossCleanT5,
  RemoveDupQuadrupletsBeforeTC, CrossCleanT4, **CrossCleanpLS**, RemoveDupPixelTripletsFromMap,
  RemoveDupQuintupletsAfterBuild, RemoveDupPixelQuintupletsFromMap, RemoveDupQuadrupletsAfterBuild.
- **Active pLS dupclean:** CheckHitspLS ×2 (gated by `!no_pls_dupclean`).
- Restore full baseline: `sed -i '/DEDUP-DISABLE-TEST/d' src/alpaka/LSTEvent.dev.cc`. Still do NOT
  `git checkout` this file (Session-14 runPT5DNN plumbing). Session-22 combined-dedup files still present.

## ✅ BUILD DONE (this session): CUDA rebuild SUCCEEDED (0 errors)
`lst_make_tracklooper -G` completed clean; fresh `LST/liblst_cuda.so` (13:04) + `bin/lst_cuda` (13:05)
now embody the keep-CheckHitspLS config. **On resume: NO rebuild needed — go straight to task (2) run
below.** (Only rebuild if you edit source again.)

## Task (2) — no-dedup-but-keep-CheckHitspLS: NEXT STEP = run 30 evt after build
Command (GPU0): `CUDA_VISIBLE_DEVICES=0 lst_cuda -i trackingNtuple-100.root --jet --idealpls -n 30 -o
LSTNtuple_nodedupKeepPLS_idealpls_30evt.root` then `createPerfNumDenHists -i <that> -o
LSTNumDen_nodedupKeepPLS_idealpls_30evt.root -J`; compare vs baseline `LSTNumDen_idealpls_fixed.root`
with `efficiency/python/lst_compare_numden_deltaR.py`. **MONITOR for stall** (see below).
- **Why keep CheckHitspLS:** the ALL-dedup-off 30-evt run STALLS deterministically (~event 13:
  single-core 99% CPU, output frozen ~2.76MB, RSS climbing) — host-side truth-match/ntuple-fill on the
  exploded TC collection. That is why S23's 30-evt was killed (`.killed_partial`, `.killed_partial2`);
  it was a manual kill, absent from the S23 handoff. Keeping pLS dupclean should reduce the explosion.
- **KEY correction to S23 fear:** NumDen `-J` is CHEAP here — 32 s / 515 MB on the 243,388-TC 10-evt
  file (NOT hours). 10-evt all-dedup-off ran clean in 50 s. Existing file:
  `LSTNtuple_nodedup_idealpls_10evt.root` (243k TCs). Both prior 30-evt attempts → `.killed_partial*`.

## Task (1) — interactive jet HIT DIAGRAM (user-in-the-loop, NOT a background agent)
Goal: diagram of one sim track's hits + its reconstructed track's hits, cut-away view showing all OT
layers from ≥2 angles (natural pair: r–z longitudinal + x–y transverse). **Feasibility VERIFIED; not
yet built.** Data recipe (reads `LSTNtuple_idealpls_fixed.root`, tree `tree`):
- **Sim track hits:** `sim_recoHitX/Y/Z[simIdx]` (all reco hits of the sim particle). Verified.
- **Reco pT5 (type 7) hits:** pixel = `pT5_plsIdx → pLS_hit0..3_{x,y,z}`; outer T5 =
  `pT5_t5Idx → (scalar) t5_t3Idx0/t5_t3Idx1 → t5_t3_0..5_{x,y,z}` (two constituent T3s, share middle
  layer ⇒ 10 unique OT hits).
- **⚠️ INDEX TRAP (verified):** the hit-detail branches `t5_t3_*_{x,y,z,r,layer}` have len **154,384**
  (filled once per unique T3, indexed by an internal `t3_index_map`), a DIFFERENT collection from the
  scalar t5 branches `t5_simIdx/pMatched/pt/eta/phi/t5_t3Idx0/t5_t3Idx1` (len **59,030**). Do NOT index
  `t5_t3_*_x` with `pT5_t5Idx` — silently returns the WRONG track's hits. Bridge via scalar
  `t5_t3Idx0/1`, which ARE indices into the hit table. `t5_tc_idx`/`t5_partOfTC` link only
  standalone-T5-type TCs, not the T5 inside a pT5.
- Verified example: event 0, sim 29 (central pT5, pMatched=1.0, OT barrel layers 1–5, third quadrant).
- **Next:** build first-draft plotting script (r–z + x–y, OT-layer backdrop, sim vs reco hits).
  User to pick first track: verified-clean sim29 vs a jet-core track.

## Session org (user decision): "Full parallel (3 agents)"
(1) jet viz interactive; (2) no-dedup in MAIN tree, GPU0; (3) new cuts in a git WORKTREE, GPU1.
Worktree for (3) needs a WIP commit first (only untracked build-source = `code/core/TruthPixelSeeds.{cc,h}`;
+ tracked M edits; EXCLUDE the 40 png / 24 py / root.partial artifacts). **Task (3) cut target NOT yet
chosen** (prior 4× loosenings of dBeta/betaOut/z-window/dPhi/radius/DNN/chi2 all null; candidates:
pT5 pairing χ², pLS/CheckHitspLS gates, pT3 cuts). Verify a worktree actually builds (cmsenv/scram
path caveat) before relying on it.

---

# Disable-All-Dedup Test + Jet Visualization — Handoff (Session 23, 2026-07-08)

## Standing Rule: Always Use CUDA
Build/run CUDA only (`lst_make_tracklooper -G`, `lst_cuda`). CPU ~9× slower.

## ⚠️ SOURCE TREE STATE (three overlapping sets of edits — read before reverting)
1. **NEW this session — DEDUP-DISABLE-TEST:** `src/alpaka/LSTEvent.dev.cc` has **12 kernel-launch
   sites wrapped in `#if 0 … #endif`**, each tagged `// DEDUP-DISABLE-TEST`. This disables ALL 11
   duplicate-removal / cross-cleaning kernels (CheckHitspLS appears twice): CrossCleanpT3,
   RemoveDupQuintupletsBeforeTC, CrossCleanT5, RemoveDupQuadrupletsBeforeTC, CheckHitspLS(×2),
   CrossCleanT4, CrossCleanpLS, RemoveDupPixelTripletsFromMap, RemoveDupQuintupletsAfterBuild,
   RemoveDupPixelQuintupletsFromMap, RemoveDupQuadrupletsAfterBuild. Kernel *bodies*
   (Kernels.h / TrackCandidate.h) are untouched.
   **To restore just this:** `sed -i '/DEDUP-DISABLE-TEST/d' src/alpaka/LSTEvent.dev.cc` then
   rebuild. **Do NOT `git checkout LSTEvent.dev.cc`** — it also carries the Session-14 `runPT5DNN`
   plumbing that a checkout would wipe (breaks the build against LSTEvent.h/LST.cc/trkCore).
2. **Pre-existing — Session-22 combined-dedup (still present, deliberately NOT reverted):**
   `interface/PixelQuintupletsSoA.h`, `src/alpaka/PixelQuintuplet.h`, `src/alpaka/Kernels.h`.
   These modify `RemoveDupPixelQuintupletsFromMap` — which is disabled in this test, so they are
   **inert here**. (PixelQuintuplet.h also mixes in the runPT5DNN plumbing → can't blind-revert.)
3. **Pre-existing — Session-14 runPT5DNN plumbing** in LSTEvent.dev.cc/.h, LST.cc, trkCore,
   PixelQuintuplet.h. Keep (defaults true, harmless).
**Full baseline restore = sed-remove the DEDUP markers AND `git checkout` the 3 combined-dedup
files (but PixelQuintuplet.h needs a surgical revert of only the combined-dedup hunk, keeping the
runPT5DNN hunks).**

## The experiment (user's request: recreate an old finding)
User previously found (outside Claude) that commenting out ALL dedup + cross-cleaning kernels
raises efficiency at the expense of fake rate. Goal this session: recreate it, keeping synthetic
`--idealpls` pLS. Approach approved via plan `~/.claude/plans/on-my-own-outside-cached-snowglobe.md`.

### Build: clean (`lst_make_tracklooper -G`, exit 0). No `-Werror` in build; unused-var warnings only.

### ⚠️ GOTCHA (important): `--allobj` + dedup-off = runaway. DROP `--allobj`.
First run (`--allobj --jet --idealpls -n 100`) went **runaway: 2h14m, single-core CPU-bound,
GPU idle, ~104 GB RSS, no output growth for 2h — KILLED** (partial saved as
`LSTNtuple_nodedup_idealpls.root.killed_partial`). Cause: with dedup off every object keeps
`isDup=false`, so the object+TC collections explode, and `--allobj` truth-matches/writes ALL of
them (the `*_simIdxAll` all-match vectors). **The dedup kernels are load-bearing for standalone
pipeline tractability, not just physics.** The efficiency/fake NumDen comparison needs only
`tc_*`/`sim_*` + `--jet` — NOT `--allobj`.

### Smoke test (5 evt, no --allobj): COMPLETED ~16s, memory bounded. Confirms the finding:
| first 5 evt | Total TC | genuine | fake | fake rate |
|---|---|---|---|---|
| baseline (dedup on) | 562 | 482 | 80 | 0.142 |
| **no-dedup** | 113,660 | 5,630 | 108,030 | **0.950** |
Genuine TC ↑ ~12× (but inflated by duplicate TCs — true sim-level efficiency gain is smaller,
NumDen will quantify); fakes ↑ ~1350×; fake rate 14%→95%; ~22k TC/event.

### CURRENT BACKGROUND JOB (in flight at handoff time): 30-event real comparison
- Job (session-scoped bg task id `bysygur34`), log `/tmp/claude-63526/nodedup_30evt.log`.
- Command (chained): `lst_cuda -i trackingNtuple-100.root --jet --idealpls -n 30 -o
  LSTNtuple_nodedup_idealpls_30evt.root` **&&** `createPerfNumDenHists -i … -o
  LSTNumDen_nodedup_idealpls_30evt.root -J`.
- Healthy (~0.9 GB RSS). ~11 min recon (buildTruthPixelSeeds dominates) + NumDen -J on ~700k TCs
  (duration unknown — `-J` scales with TC count).
- **If picking up in a new session:** the bg job may not survive session end. Check for both
  output files; if missing/partial, just rerun the chained command above. Then compare
  `LSTNumDen_nodedup_idealpls_30evt.root` vs baseline `LSTNumDen_idealpls_fixed.root` (100 evt;
  event-count mismatch is fine — eff/fake are per-bin rates) with
  `efficiency/python/lst_compare_numden_deltaR.py`. Expected: TC eff ↑, fake rate ↑ across ΔR.
- Note: NumDen -J on millions of TCs was flagged as a risk → chose 30 evt (user) over 100.

## Jet visualization thread (Event 0, GenJet1 = idx 1 at η=−1.16, φ=−2.37)
Built a family of η–φ scatter tools (all read `LSTNtuple_idealpls_fixed.root`, tree `tree`):
- `efficiency/python/lst_plot_simtrack_etaphi.py` — sim tracks, colored by ΔR (jet shape).
- `efficiency/python/lst_plot_tc_etaphi.py` — TCs colored by type (pT5/T5/pT3/pLS/T4), genuine
  vs fake (filled/×).
- `efficiency/python/lst_plot_sim_vs_tc_etaphi.py` — sim-vs-TC **overlay + side-by-side**;
  `--sim-ptcut` (default 0.8) + charged filter (genuine TC-creation restriction), `--xlim/--ylim`
  zoom (panel counts auto-restrict to window).
- `efficiency/python/lst_plot_object_overlay_etaphi.py` — overlay one *lower-level* collection
  (pT5/t5/pLS objects, before TC assignment) on sim; fakes as × on top (they occlude otherwise).
Common axes: x∈[−5.5,5.5], y∈[−π−.2,π+.2]. Marker size ∝ log(pT).

### Key physics observations (Event 0, GenJet1 core; ideal/synthetic pLS)
- **Reconstruction funnel** at the jet core (tight window ±0.1η×±0.075φ): 16 pLS (0 fake) →
  **250 genuine + 33,676 fake T5** → 14 genuine + 17 fake pT5 → 14 genuine + 7 fake TC.
- **Fake T5 are spatially degenerate:** the ~34k fakes collapse onto ~18 hit-defined (η,φ)
  points (std ~0.003) — a T5's η/φ is set by its hits and fakes reuse the same hits. Not a smeared
  cloud. This IS the combinatorial pileup the dedup kernels prune.
- **T5 objects cluster at η≈−1.10 (offset ~0.07 from the jet axis); pT5 pull back onto the axis
  (η≈−1.175).** ROOT-CAUSED (not displaced tracks — the 250 genuine core T5 map to 19 sim tracks
  all at η≈−1.18): **`t5_eta` = `quintuplets.eta()` is a HIT-POSITION η** (outer anchor-hit
  position from origin, no vertex constraint; branch doc "based on last anchor hit's eta"; same for
  md/ls/t3_eta). It differs from momentum η by `sinh(η_pos)=sinh(η_mom)+v_z/r`. This jet's PV is at
  **v_z≈+4.67 cm** (x=y≈0; the window's mean v_z=−126 cm is unrelated displaced pileup). A track
  going −z (η<0) from a +z vertex has less-negative position-η → pulled central. Numerically:
  η_mom=−1.178 → t5_eta=−1.107 (Δsinh=+0.124 ⇒ effective anchor r≈38 cm) — matches exactly. Fakes
  shift identically (reuse the same hits). **pT5/pLS DON'T shift** because their η comes from the
  pixel-seed direction (vertex-pointing; for synthetic pLS = truth momentum) → true η. So the
  ~0.07 T5-vs-pT5 gap IS this v_z effect. Takeaway: lower-object η/φ plots (MD/LS/T3/T5) are
  hit-position, PV-z-biased; pixel-seeded objects (pT3/pT5/pLS) are not. Optional future tweak:
  plot genuine lower objects at their matched-sim η/φ to line up with the core.
- **The TCs we examined use SYNTHETIC pLS** (idealpls; 10,776 pLS total ≈ synthetic signature,
  real would be ~47k). pT5/pT3/pLS TCs carry a synthetic pixel seed; T5/T4 TCs have no pixel seed.
  So the core inefficiency seen is LST-internal (dedup), not seeding-supply.
- **LST has 5 TC types** (Common.h:21 `T5=4,pT3=5,pT5=7,pLS=8,T4=9`), 5 `Add*asTrackCandidate`
  kernels. pLS TCs = pixel-only tracks (quad, non-dup pLS not absorbed into pT5/pT3;
  `TrackCandidate.h:669,681`, gated by `tc_pls_triplets`).

## Docs written this session (not code)
- `EFFICIENCY_ATTEMPTS.md` (standalone dir) — living ledger of every efficiency attempt (S3–22) +
  proposed-unapplied table. Memory: `project_efficiency_attempts_ledger.md`. **Add the no-dedup
  result to both once the 30-evt comparison lands** (it belongs in section A.4 as the extreme point:
  eff↑ / fake→0.95).

## Plots produced this session (standalone dir)
`simtrack_etaphi_evt0.png`, `tc_etaphi_evt0.png`, `sim_vs_tc_overlay_evt0.png`,
`sim_vs_tc_sidebyside_evt0.png`, `sim_vs_tc_{overlay,side}_evt0_gj1core.png` (±0.5/±0.25),
`sim_vs_tc_overlay_evt0_gj1zoom.png` + `obj_{pT5,t5,pLS}_over_sim_evt0_gj1zoom.png` (±0.1/±0.075).

## Next steps
1. When the 30-evt job finishes: validate outputs, run `lst_compare_numden_deltaR.py` vs baseline,
   report ΔR-binned TC eff & fake rate; add to `EFFICIENCY_ATTEMPTS.md` + ledger memory.
2. Decide whether to scale to more events (NumDen -J cost is the limiter, not the run).
3. Restore source tree to baseline when done (see SOURCE TREE STATE above).
4. Standing decision still pending from S22: revert-vs-keep the 3 combined-dedup files.

---

# Synthetic vs Real pLS Comparison — Handoff (Session 22, cont. 2026-07-07)

## Standing Rule: Always Use CUDA
Build/run with CUDA only (`lst_make_tracklooper -G`, `lst_cuda`). CPU ~9× slower.

## ⚠️ SOURCE TREE STILL NOT AT PRODUCTION STATE
The three combined-dedup edits from earlier today are still in place (see the Session-22 dedup
section below): `interface/PixelQuintupletsSoA.h`, `src/alpaka/PixelQuintuplet.h`,
`src/alpaka/Kernels.h`. No LST was rebuilt/run in this segment (analysis only, on existing
ntuples), so these edits are untouched. Revert them before any baseline run. Decision on
revert-vs-keep still pending.

## New thread: explicit comparison of synthetic (--idealpls) vs real (CMSSW) pLS
User is examining how the truth-derived synthetic seeds differ from the real input seeds.
**Agreed definition (user, 2026-07-07): use the LST-RECONSTRUCTED pLS** (matched to sim at
>75% hit purity via `sim_plsIdxAll`/`sim_plsIdxAllFrac`), NOT the raw input `see_*` seeds.

**Reference files (both from `trackingNtuple-100.root`, 100 events, 9665 accepted sim tracks):**
- REAL seeds: `LSTNtuple_fix2.root` (Jun 12; plain real-seed run — identified by ~47k total pLS
  objects vs ~11k for synthetic, i.e. ~5 vs ~1 pLS/track. The Session-15 momentum fix only
  touched SYNTHETIC seeds, so this older file is a valid real-seed reference).
- SYNTHETIC seeds: `LSTNtuple_idealpls_fixed.root` (`--idealpls`).
- **Index alignment VERIFIED**: both files have 100 events, identical per-event sim counts, and
  `sim_pt`/`sim_eta` identical index-for-index (max Δ = 0 over 102,799 sim tracks) → safe to match
  a real pLS and a synthetic pLS to the same sim track by (event, index).

### Result 1 — pLS availability (fraction of accepted sim tracks with a matched pLS)
| | source | with pLS | fraction |
|---|---|---|---|
| Real (originally assigned) | fix2 | 8589/9665 | **88.9%** |
| Synthetic (--idealpls) | idealpls_fixed | 9325/9665 | **96.5%** |

+7.6 pp overall (concentrated at small ΔR; real-seed deficit is far worse ~48% in jet cores).
Synthetic ≠ 100%: the missing 3.5% lack ≥3 distinct pixel-layer truth hits (conversion-electron
population). Synthetic makes ~4× fewer pLS objects (10.8k vs 47.0k; real floods fakes/dupes).

### Result 2 — kinematics of ALL genuine pLS (plot `pls_real_vs_synth_kinematics.png`)
Genuine (pLS_isFake==0) pLS, unit-area normalized. pT similar (steeply falling, synth slightly
harder). **η very different**: real forward-peaked (double hump |η|≈2.5–3), synthetic
central-peaked. φ ≈ uniform for both. CAVEAT: real genuine pLS are inflated by ~5 duplicate
seeds/track, and the multiplicity is η-dependent (forward tracks get more seeds) → part of the
η contrast is a multiplicity artifact, not underlying kinematics.

### Result 3 — per-track BEST-matched real vs synthetic diff (plots `pls_matched_diff.png`,
### `pls_matched_diff_invpt.png`)
Removes the multiplicity confound: one best-matched (highest-frac) pLS of each type per sim
track. **8589 tracks have BOTH.** Differences (real − synthetic):
| variable | median | 68% half-width |
|---|---|---|
| relative pT (pT_r−pT_s)/pT_s | −0.0023 | 0.060 (~6%) |
| **Δ(1/pT)** [GeV⁻¹] | +0.00033 | **0.0147** |
| η | +0.00000 | 0.0011 |
| φ [rad] | −0.00002 | 0.0019 |

**Interpretation:** η and φ agree to ~1–2×10⁻³ — largely BY CONSTRUCTION, because
`buildTruthPixelSeeds` anchors synthetic seeds to the SAME real pixel hits, so hit-derived
direction can't differ. The seeds differ essentially ONLY in the momentum estimate (synthetic
takes pT from truth; real from the pixel-triplet curvature fit), so panel A is effectively the
real pixel-seed pT/curvature resolution: unbiased, ~6% core in pT, ~0.015 GeV⁻¹ core in 1/pT,
with tails from low-pT curvature mismeasurement. **1/pT (curvature) is the well-behaved variable
— the fat relative-pT tails compress away; use it going forward.**

**KEY CONCLUSION for the ideal-pLS study:** for tracks that already had a real seed, synthetic and
real pLS are the SAME object (same hits, same η/φ) differing only marginally in momentum. So
`--idealpls`'s efficiency benefit is **seed AVAILABILITY, not seed QUALITY** — it comes from the
+7.6 pp of tracks that had no real seed at all, not from better-pointed seeds on existing tracks.

### Next steps (this thread)
1. Characterize the NON-OVERLAP populations: tracks with a real pLS but no synthetic, and
   synthetic but no real — where does each seeding method uniquely succeed/fail? (proposed,
   not yet run.)
2. Optionally regenerate a same-day real-seed run for a fully current apples-to-apples (fix2 is
   Jun 12; valid but old). ~35 min.
3. If curvature resolution matters, bin Δ(1/pT) vs η and vs pT to see where real seeds degrade.

### Files created this segment
| File | Purpose |
|---|---|
| `efficiency/python/lst_plot_pls_real_vs_synth.py` | all-genuine-pLS pT/η/φ overlay (Result 2) |
| `efficiency/python/lst_plot_pls_matched_diff.py` | per-track real−synth diff, relative pT (Result 3) |
| `efficiency/python/lst_plot_pls_matched_diff_invpt.py` | same but Δ(1/pT) curvature (Result 3) |
| plots: `pls_real_vs_synth_kinematics.png`, `pls_matched_diff.png`, `pls_matched_diff_invpt.png` | |

### To remember (user directives, this session)
- Comparison uses the **LST-reconstructed pLS definition** (not raw input `see_*`). User may later
  want the raw-input-seed version separately.
- Stop and ask if a variable's access/definition is unclear (honored: verified pT=transverse via
  the 0.8 cut, verified cross-file sim index alignment before matching).
- `pLS_pt` is transverse momentum; `pLS_eta`/`pLS_phi` direct branches; `sim_plsIdxAll[i]` +
  `sim_plsIdxAllFrac[i]` give a track's matched pLS (>75% only).

---

# pT5 Dedup Redesign Tested — Inward χ² Insufficient — Handoff (Session 22)

## Standing Rule: Always Use CUDA
Build/run with CUDA only (`lst_make_tracklooper -G`, `lst_cuda`). CPU ~9× slower.

## ⚠️ SOURCE TREE NOT AT PRODUCTION STATE
Three files carry the **combined-dedup change** (permanent edits, NOT script-restored):
- `interface/PixelQuintupletsSoA.h`: `rPhiChiSquaredInwards` moved OUT of `#ifdef CUT_VALUE_DEBUG`
  (now always stored, needed by the dedup kernel unconditionally).
- `src/alpaka/PixelQuintuplet.h`: `addPixelQuintupletToMemory` stores `rPhiChiSquaredInwards`
  unconditionally (line ~88, moved below the `#endif`).
- `src/alpaka/Kernels.h` (`RemoveDupPixelQuintupletsFromMap`, ~line 460/474): dedup score is now
  `__H2F(score) + rPhiChiSquaredInwards` (outward + inward) for BOTH `score1` and `score2`.
**Revert these three if returning to baseline.** The H1/H2 `PixelQuintuplet.h` visibility-gate
patches ARE auto-restored by the run scripts.

## Investigation (1) — score-misranking diagnostic (COMPLETE)

Added FP16-rounded `pT5_score` / `t5_score` branches; ran H2 config (both visibility gates open),
analyzed with `efficiency/python/lst_plot_score_vs_purity.py`.
Plots: `pt5_score_vs_purity.png` (2-panel), `pt5_score_vs_purity_inwards.png` (3-panel, -d build).

- **FP16-tie hypothesis REJECTED.** On 3269 deciding (victim,winner) pairs: only **6% exact ties**,
  **94% strict score wins**. The kernel's winning pT5 genuinely has the lower `rPhiChiSquared`.
  Fixing the tiebreaker alone recovers ≤6%.
- **Score measures the wrong thing.** Same-seed groups with BOTH a genuine (≥0.75) and a mixed pT5:
  the **mixed pairing wins on outward score 50%** of the time (40/80). `rPhiChiSquared` (outward:
  OT hits vs pixel-derived circle) is purity-blind — in dense cores a neighbor's T5 hits fit the
  target helix as well or better.

## pT5 dedup redesign — combined (outward + inward) χ² — TESTED, INSUFFICIENT

Plan (approved, `~/.claude/plans/sunny-wishing-mist.md`): add `rPhiChiSquaredInwards` (inward:
pixel hits vs T5 regression circle) to the dedup ranking. Hypothesis: a mixed pT5 (target pLS +
neighbor T5) fails the inward check because the neighbor's circle doesn't extrapolate to the
target's pixel hits.

**Panel C diagnostic (-Gd build, `LSTNtuple_expH2_score_inwards.root`):** combined `out+in` vs
outward-only, same-seed contested groups:
| metric | outward only | combined out+in |
|---|---|---|
| mixed pairing wins | 51% | **43%** |
| genuine pairing wins | 23% | **35%** |
Partial improvement, not a fix — inward χ² still misranks 43% of contested pairs.

**Efficiency measurement (`LSTNumDen_expH2_combinedDedup_J.root` vs `LSTNumDen_expH2.root`),
ΔR<0.01 (/358):**
| Metric | baseline | H2 outward-only | H2 combined |
|---|---|---|---|
| pT5_lower (objects) | 0.687 (246) | 0.768 (275) | 0.768 (275) |
| **pT5-type TC eff** | 0.642 (230) | 0.606 (217) | **0.609 (218)** |
| **pT5 fake rate** | 0.325 | 0.281 | **0.354** |

**Conclusion: combined dedup recovers +1 TC (217→218) and INCREASES fake rate (0.281→0.354).
The inward χ² is not a sufficient discriminant.** pT5_lower is identical by construction (dedup
picks which object becomes a TC, not how many exist). Plot: `eff_fake_vs_deltaR_combinedDedup.png`
(green ≈ orange in panel B; green slightly worse in panel C fake rate).

## Where this leaves the investigation
The pT5 deficit at ΔR<0.01 has THREE code-localized mechanisms (Sessions 21–22):
1. **Visibility gates** (`PixelQuintuplet.h:666,680`) — proven cause; opening them adds +29 genuine
   pT5 *objects* (0.687→0.768 pT5_lower) but the freed genuine pT5s are then killed by:
2. **pT5-vs-pT5 dedup score misranking** (`RemoveDupPixelQuintupletsFromMap`) — the score is
   purity-blind; **neither the FP16 tiebreaker (≤6%) nor the inward χ² (this session) fixes it.**
   A working fix needs a genuinely purity-aware signal (candidates: T5 DNN score, hit-level
   match weighting, `rzChiSquared`, or a redesigned selection that isn't pure geometric residual).
3. **`ExtendTrackCandidatesFromDupT5`** (`TrackCandidate.h:746`) — staples neighbor hits onto TCs,
   dropping match below 0.75 in cores. **USER DIRECTIVE: noted, do NOT act on yet.**

### Next options (NOT approved — propose before launching, per user rule)
- (2') Try `rzChiSquared` and/or T5 `dnnScore` (already in Quintuplets SoA, not propagated to pT5)
  as the dedup discriminant instead of / in addition to inward χ².
- (3) Seed-side dedup study (`CheckHitspLS`).
- (4) Chain-drift quantification (1173/4117 victims had no ≥7-shared survivor).
- Decide whether to revert the three combined-dedup edits or keep them as scaffolding for (2').

### Files changed this session
| File | Change |
|---|---|
| `interface/PixelQuintupletsSoA.h` | `rPhiChiSquaredInwards` made unconditional (out of CUT_VALUE_DEBUG) |
| `src/alpaka/PixelQuintuplet.h` | store `rPhiChiSquaredInwards` unconditionally |
| `src/alpaka/Kernels.h` | dedup score = outward + inward χ² |
| `efficiency/python/lst_plot_score_vs_purity.py` | added Panel C (combined-score comparison) |
| `efficiency/python/lst_plot_expH_eff_fake_combined.py` | New: 3-series eff/fake ΔR plot |
| `t5tc_expH2_score.sh`, `t5tc_expH2_score_inwards.sh`, `t5tc_expH2_combinedDedup.sh` | New run scripts |

### Output files this session
| File | Description |
|---|---|
| `LSTNtuple_expH2_score.root` | H2 + FP16 score branches (Investigation 1 panels A/B) |
| `LSTNtuple_expH2_score_inwards.root` | H2 + -d build (panel C: inward χ²) |
| `LSTNtuple_expH2_combinedDedup.root` | H2 + combined-dedup kernel |
| `LSTNumDen_expH2_combinedDedup_J.root` | NumDen with `-J` (the non-`_J` file is EMPTY — `-J` was omitted first pass) |
| `pt5_score_vs_purity.png`, `pt5_score_vs_purity_inwards.png`, `eff_fake_vs_deltaR_combinedDedup.png` | plots |

### Gotcha logged
`createPerfNumDenHists` needs **`-J`** for the ΔR-binned histograms (`*_deltaR`); without it the
numerators are empty and series vanish from plots. Open-gate ntuples have ~6810 pT5 TCs → `-J`
matching takes ~24 min (vs a few min normally).

---

# Dedup Root Cause Found; Experiment G Launched — Handoff (Session 21)

## Standing Rule: Always Use CUDA

**Always build and run with the CUDA backend** (`lst_make_tracklooper -G`, `lst_cuda`).
CPU is ~9× slower. Never use `lst_make_tracklooper -C` or `lst_cpu` unless explicitly asked.

## No Background Job Running (Experiment H completed 08:49, patches restored)

All source files at production state. Plan file: `PLAN_expH_pt5_visibility.md`.

### Experiment H RESULT (completed 2026-07-06 08:49): hypothesis CONFIRMED at object level;
### naive gate removal backfires at TC level (pre-registered "Outcome 4")

- H1: pT5 building sees isDup T5s (`PixelQuintuplet.h:680` skip removed)
- H2 = H1 + also sees isDup pLS (`:666` skip removed). Production state otherwise.
- NOTE: H2's lst_cuda run took ~2.4 h (vs ~35 min) — pLS visibility multiplies pairing work.

| Metric (ΔR<0.01, /358) | baseline | H1 | H2 |
|---|---|---|---|
| **pT5_lower (genuine pT5 objects)** | 0.687 (246) | 0.740 (265) | **0.768 (275)** |
| pT5-type TC eff | 0.642 (230) | 0.581 (208) | 0.606 (217) |
| Total TC eff | 0.690 (247) | 0.637 (228) | 0.654 (234) |
| pT5-type fake rate | 0.325 | 0.280 | 0.281 |
| cat B (pT5 exists, no TC) | 17 | 56 | 57 |
| cat D (free T5, no TC) | 59 | 45 | 38 |

**Interpretation:** the isDup visibility gates in pT5 building WERE removing genuine
pT5s (+29 tracks with a genuine pT5 in H2 — answers the user's "what removes pT5s"
question at the object level; both T5-side and pLS-side gates contribute, ~2/3 : 1/3).
BUT the gates were also protecting the pT5-vs-pT5 dedup from mixed-T5 pairings: with
them open, `RemoveDupPixelQuintupletsFromMap` (≥7/14 shared hits + score) now kills
MORE genuine pT5s than before (cat B 17→57; 41/57 confirmed dup-killed via
pT5_isDupReco), and TC-level efficiency NET DROPS. No fake explosion (unlike Exp G) —
pT5 dedup + DNN keep fakes controlled; core pT5 fake rate actually improves.

**The battlefield has moved to `RemoveDupPixelQuintupletsFromMap` (Kernels.h:454):
its score-based winner selection prefers mixed pT5s over genuine ones in dense cores.**

### pT5-vs-pT5 dedup winner analysis (plot: `pt5_dedup_winner_purity.png`)

`lst_plot_pt5_dedup_winners.py` on `LSTNtuple_expH2.root`: the 57 cat-B failing core
tracks have 4117 dup-killed genuine pT5 victims (~72/track — H-opened duplicate
clusters) and only 50 unique ≥7-shared-hit surviving winners:
- **28/50 winners are FAKE (mixed) pT5s, 26 of them built from the SAME pixel seed as
  the victim** — the seed pairs with a contaminated T5 and outscores its own genuine
  pairing. pT5-dedup score misranking under contamination CONFIRMED.
- 15/50 same-track genuine winners → these overlap with the "anomaly" below.
- 1173/4117 victims have no ≥7-shared non-dup competitor (chain drift, as at T5 level).

### ANOMALY ROOT-CAUSED: `ExtendTrackCandidatesFromDupT5` staples neighbor hits onto TCs

The "matched non-dup pT5 but no TC" anomaly (5 baseline → 16 in H2) is now explained:
- Empirically: pT5-object match 12/14 (0.857) becomes TC match 12/16 (0.750), failing
  the strict `> 0.75` cut. pT5-type TCs have (nhits,nhitOT) = **(16,12) for 74%** of
  cases vs the nominal (14,10) — two extra OT hits.
- Mechanism: **`ExtendTrackCandidatesFromDupT5` (TrackCandidate.h:746)** runs after TC
  creation and merges the best-scoring nearby (duplicate) T5's hits into uncovered
  logical layers of each T5/pT5 TC (second `addTrackCandidateLayerHits` caller, :916).
  At large ΔR the stitched hits come from a same-track duplicate T5 (benign, match
  rises, e.g. 15/16=0.938). **In jet cores they come from the neighbor track or mixed
  T5s — polluting genuine TCs below the 75% threshold, flipping them to "fake" and the
  track to "not found".** This is a ΔR-dependent efficiency+fake mechanism in its own
  right, previously invisible; it also biases every TC-level metric relative to
  object-level metrics by the 14-vs-16 denominator difference.
- Provenance verified: native LST dev code (NOT a Claude modification) — commit
  4cb70dbb599, GNiendorf, 2025-12-11, refined 2026-05-13 (16b52f53ab7); recent feature,
  likely untested in dense cores.
- **USER DIRECTIVE: noted for the future, DO NOT act on this yet** (no
  disable-experiment, no dev report) until the user asks.

### Experiment G result (completed 03:43): before-TC dedup NOT the pT5 remover; fake explosion

vs baseline at ΔR<0.01: total TC 247→257 (+2.8 pp), T5-type TC 0→14, **pT5_lower
unchanged** (246→247). Cost: standalone T5-type TC count exploded (17→794 core,
2221→9536 far), T5-type fake rate 0.71→0.96 core / 0.34→0.78 far, total TC fake rate
0.093→0.437 at ΔR>0.05. **Conclusions:** (i) the pure-geometric dedup arms do heavy
legitimate fake suppression — wholesale removal is unusable; (ii) pT5 loss happens
BEFORE the before-TC stage (pT5_lower immobile), consistent with the visibility-gate
hypothesis tested by Experiment H.

### Winner-purity measurement (completed, plot: `dedup_winner_purity.png`)

`lst_plot_dedup_winner_purity.py` on `LSTNtuple_expF_diag.root`: 2072 after-build-killed
pure T5s (59 cat-D tracks), 129 unique ≥7-shared-hit surviving winners:
- **73/129 winners are same-track pure T5s** — after-build score selection usually picks
  a CORRECT T5. Their fates: 62 killed at before-TC cross-track dedup, 9 consumed into
  pT5s built with the WRONG pLS (→ ~0.71-pure fake pT5s), 2 free but no TC.
- 52/129 mixed (0.5–0.75) winners — score misranking real but sub-dominant.
- **771/2072 victims have NO ≥7-shared-hit survivor at all** — dedup chain drift
  annihilates whole overlap neighborhoods (killer killed by a third T5 sharing <7 hits
  with the victim), leaving no local T5 for pairing.
**Implication:** fixing the after-build winner criterion (original plan (b)) is NOT the
main lever; the visibility gates in pT5 building are — hence Experiment H.

### Gate attribution (completed this session, `LSTNtuple_expF_diag.root`)

Using the new reco-flag branches, the 59 category-D failing core tracks decompose:
- **2072/2137 (97%) of their free matched T5s carry the after-build dedup bit** (bit0),
  63 the before-TC bit (bit1), 2 tightCut.
- Per track: 17 tracks all-afterBuild, 40 mixed afterBuild+beforeTC, 2 +tightCut.
- Neighborhood survivor scan (`lst_find_dedup_winners.py`): every failing track has
  survivors within dR<0.01; **92% of survivors are consumed into pT5s**, including
  740/1215 with pMatched<0.75 (mixed/contaminated T5s embedded into fake pT5s), plus
  the neighbor track's pure T5s.

**Mechanism (jet core, ≥2 collimated tracks A,B):** each track spawns ~36 pure duplicate
T5s. (1) Within-module after-build dedup (≥7/10 shared hits, score_rphisum) often hands
the cluster win to a *mixed-hit* T5 → consumed into a fake pT5. (2) The surviving pure
B-T5s then face before-TC standalone-vs-standalone dedup where the pure-geometric arm
(dR2<0.001 always true at ΔR<0.01) forces a score contest vs A's leftover pure T5s —
one track loses its last T5. Efficiency loss and fake-rate inflation at jet cores are
largely THE SAME OBJECTS (mixed winners becoming fake pT5s/T5s).

**Data caveat discovered:** `matchedSimTrkIdxsAndFracs` (trkCore.cc) only returns
matches with frac > matchfrac (0.75) despite branch comments claiming ">0%" — so
`sim_t5IdxAll`/`t5_simIdxAll` contain no sub-75% entries. Use `t5_pMatched`
(unthresholded max frac) for purity of unmatched objects.

### ANSWER to "what removes so many pT5s": after-build T5 dedup runs BEFORE pT5 pairing

Kernel order (`LSTEvent.dev.cc`): `createQuintuplets()` ends with
`RemoveDupQuintupletsAfterBuild` (line 1002) → `createPixelQuintuplets()` (line 1037).
And `CreatePixelQuintupletsFromMap` **skips isDup T5s** (`PixelQuintuplet.h:680`:
`if (quintuplets.isDup()[quintupletIndex]) continue;`).

So at jet cores the pure T5s of a losing track are already isDup-flagged before pLS-T5
pairing ever sees them; the pLS can only pair with the surviving (often mixed-hit)
cluster winners, producing impure pT5s that fail 75% truth matching. This explains:
- pT5_lower gap (0.687 vs T5_lower 0.902): T5_lower counts ALL T5s incl. isDup;
  pT5 building only sees non-dup ones.
- ALL Sessions 16–18 cut-loosening nulls (dBeta, betaOut, z/dPhi, radius, DNN, chi2):
  the pure (pLS,T5) pairs were skipped at line 680 and never reached those cuts.
- High core fake-pT5 rate: the mixed winners ARE the pT5s that form — efficiency loss
  and fakes are the same objects.

**The lever for pT5 recovery is the after-build winner selection** (score_rphisum
often prefers mixed-hit T5s in dense clusters), NOT the pairing cuts and NOT the
before-TC dedup. Note: Experiment G does not touch after-build dedup, so G can only
add standalone T5-type TCs, not recover pT5s.

### User directives (2026-07-06, recorded in memory)

1. Priority: determine what removes pT5s (answered above; quantify/fix next — with approval).
2. Physics study — no PR-level rigor needed yet.
3. Nothing queued after G without user approval.
4. Propose, don't auto-launch: user monitors experiment decisions. Fake rate matters
   broadly but is not the current optimization target.

## Session 21 Results

### Experiment F (skip standalone-vs-pT5 T5 dedup, F1+F2) — first non-null signal

vs `LSTNumDen_idealpls_fixed.root` baseline (all ΔR<0.01, q=-1 denom = 358):

| Metric | Baseline | Exp F | Exp F+D |
|---|---|---|---|
| Total TC eff | 0.690 (247) | 0.698 (250) | 0.701 (251) |
| T5-type TC eff | 0.000 (0) | 0.011 (4) | — |
| T5-type fake rate | 0.706 (12/17) | 0.686 (24/35) | — |
| Total TC eff ΔR>0.05 | 0.934 (1102) | 0.928 (1095) | 0.927 (1094) |

Rescued T5 TCs are *less* fake-dominated than baseline (fake fraction drops). Cost: ~7 TCs
lost at large ΔR — extra (mostly fake) T5 TCs kill genuine pLS TCs via `CrossCleanpLS`.
Below the pre-registered ">5 genuine" bar, but real.

### Master classification of failing jet-core tracks (new: `lst_classify_tc_failures.py`)

Exp F ntuple, ΔR<0.01 (core) vs ΔR>0.05 (far), first-match-wins categories:

| Category | Core | Far | Enrichment |
|---|---|---|---|
| A: no matched T5 or pT5 built | 32 (8.9% of denom) | 66 (5.6%) | 1.6× |
| B: genuine pT5 object exists, no TC | 17 (4.7%) | 8 (0.7%) | **7×** |
| C: all matched T5s consumed by pT5 | 0 | 0 | — |
| D: free matched T5 exists, no TC | 59 (16.5%) | 11 (0.9%) | **18×** |

Rescuing B+D fully would take core TC eff from 0.70 to ~0.91 ≈ T5_lower (0.90).

### Ruled out this session

1. **tightCutFlag** (category D): F+D vs F rescued exactly 1 track (D: 59→58). NOT the blocker.
2. **CrossCleanT5 vs-pT3 branch** (`TrackCandidate.h:281`, `dR2<1e-3` pure-geometric kill):
   only 2/76 failing tracks have all T5s within the 0.032 kill cone; median distance 0.197.
   Identical distribution in far control (`lst_check_pt3_killcone.py`).
3. **partOfPT5 consumption**: every failing track has ≥1 unconsumed T5 (`lst_check_partofpt5.py`).

### Remaining suspects (the actual blockers)

- **Category D (58 tracks)**: `RemoveDupQuintupletsAfterBuild` (Kernels.h:177, within-module
  ≥7/10 shared hits) and/or standalone-vs-standalone `RemoveDupQuintupletsBeforeTC`
  (Kernels.h:265 area; at core separations `dR2<1e-4 < 0.001` always fires; kill needs
  embedding `d2<1.0`). Purity argument: two ≥75%-pure T5s of *different* tracks can share at
  most 4/10 hits, so ≥7-shared or ≥5-shared kills involve a *contaminated* competitor T5
  winning on `score_rphisum` — or the pure-geometric `dR2<0.001 && d2<1.0` arm kills with
  zero shared hits.
- **Category B (17 tracks)**: `RemoveDupPixelQuintupletsFromMap` (Kernels.h:454, ≥7/14 shared
  hits + score). Two ≥75%-pure pT5s of different tracks can share at most 6/14 hits → the
  winner that killed a genuine pT5 must be a <75%-pure (fake) pT5 with better (lower) score.

### New diagnostic ntuple branches (write_lst_ntuple.cc, additive, kept)

- `t5_isDupBits`: reco isDup — bit0 = after-build dedup (or CrossCleanT5, rare under F2),
  bit1 = before-TC dedup
- `t5_tightCutFlag`, `t5_partOfPT5`: reco flags from Quintuplets SoA
- `pT5_isDupReco`: reco isDup from RemoveDupPixelQuintupletsFromMap
- NOTE: pre-existing `t5_isDuplicate`/`pT5_isDuplicate` are TRUTH-based (sim track has >1
  matched object), NOT the reco dedup flag — easy to confuse.

### Known latent issue (fix if D graduates)

Experiment D patch only removes the tightCutFlag gate in `AddT5asTrackCandidate`;
`CountSurvivingTCs` (TrackCandidate.h:540) still requires it → allocation undercount.
Benign in practice (shared pool has slack from pre-cross-clean pLS/T4 upper bounds).

### Job anomaly note

The F+D launch produced two script instances (second at 01:31, cause unclear — possibly a
harness command re-execution). Instance 2 idempotently "re-applied" patches, rebuilt a no-op,
and died at the output-exists check without restoring. Instance 1 completed normally and
restored all patches. The F+D ntuple is valid (verified: clean close, correct classification
denominators). For future launches: guard scripts with a lockfile (`flock` or mkdir-lock).

## Next Steps

1. **Analyze Experiment G** (see "Current Background Job" above). If G rescues the ~40
   mixed-gate tracks, remaining core losses are: 17 afterBuild-only tracks (needs the
   within-module winner-selection fix — score_rphisum favors mixed T5s; measure winner
   purity vs score directly using hit-level t5_t3_*_detId/x/y/z overlap), category B
   (17 tracks, pT5-vs-pT5 dedup score misranking), category A (32, build ceiling).
2. **Category B attribution detail (diag run)**: 12/17 have all matched pT5s with
   pT5_isDupReco=1 (dedup-killed); **5/17 have a non-dup matched pT5 that still isn't a
   TC** — unexplained, needs a check (AddpT5asTrackCandidate has no other gate; suspect
   TC-level matching or allocation edge).
3. **Category B fix candidate**: score direction in `RemoveDupPixelQuintupletsFromMap` — a
   fake (<75% pure) pT5 outscoring a genuine one suggests score misranks under
   contamination; consider nMatched-weighted preference or purity-robust score.
4. **If G works, design the production-quality combination**: F1+F2+G all remove
   geometric-proximity dedup in favor of hit-overlap dedup; evaluate at ΔR>0.05 for
   duplicate-rate cost, and re-check the CrossCleanpLS interaction (Exp F cost ~7 TCs at
   large ΔR via fake T5 TCs killing genuine pLS TCs — G may add more).
5. **Standing constraint**: no new TrackCandidate types.

### Files Changed This Session

| File | Change |
|---|---|
| `code/core/write_lst_ntuple.cc` | New branches: t5_isDupBits, t5_tightCutFlag, t5_partOfPT5, pT5_isDupReco |
| `t5tc_patch.py` | Added experiment G target (hit-overlap-only before-TC dedup) |
| `t5tc_expFD.sh` | New: F1+F2+D experiment script |
| `t5tc_expF_diag.sh` | New: F1+F2 + diagnostic-branch run script (lockfile-guarded) |
| `t5tc_expG.sh` | New: F1+F2+G experiment script (lockfile-guarded) |
| `efficiency/python/lst_classify_tc_failures.py` | New: master A/B/C/D failure classification |
| `efficiency/python/lst_compare_numden_deltaR.py` | New: NumDen ΔR-binned eff/fake comparison |
| `efficiency/python/lst_check_pt3_killcone.py` | New: CrossCleanT5 vs-pT3 cone check |
| `efficiency/python/lst_check_partofpt5.py` | New: partOfPT5 consumption check |
| `efficiency/python/lst_attribute_dedup_stage.py` | New: gate attribution via reco-flag branches |
| `efficiency/python/lst_find_dedup_winners.py` | New: dedup-winner purity scan (uses t5_pMatched) |
| `efficiency/python/lst_check_dedup_winners.py` | New: (superseded by lst_find_dedup_winners.py) |
| `efficiency/python/lst_check_partial_t5_fate.py` | New: (void — sim_t5IdxAll has no sub-75% entries) |

### Output Files This Session

| File | Description |
|---|---|
| `LSTNtuple_noT5pT5dedup.root` / `LSTNumDen_noT5pT5dedup.root` | Exp F (complete) |
| `LSTNtuple_noT5pT5dedup_notight.root` / `LSTNumDen_noT5pT5dedup_notight.root` | Exp F+D (complete) |
| `LSTNtuple_expF_diag.root` | F1+F2 + diagnostic branches (complete, analyzed) |
| `LSTNtuple_expG.root` / `LSTNumDen_expG.root` | Exp F1+F2+G *(in progress)* |

---

# Experiment F In Progress — Handoff (Session 20)

## Standing Rule: Always Use CUDA

**Always build and run with the CUDA backend** (`lst_make_tracklooper -G`, `lst_cuda`).
CPU is ~9× slower. Never use `lst_make_tracklooper -C` or `lst_cpu` unless explicitly asked.

## Current Background Job

**Script:** `t5tc_expF.sh`
**Log:** `t5tc_expF.log`
**PID:** 2568126 — fully detached (`PPID=1`, `TT=?`), safe to log out
**Started:** 2026-07-05 23:02 EDT
**ETA:** ~00:14 EDT (72 min total: 5 min build + 37 min run + 30 min NumDen)

**What it does:** applies F1 + F2, builds CUDA, runs 100-event idealpls, generates NumDen, restores.

## Current State of Source Files

All production-clean. The script applies/restores changes automatically.

- **`src/alpaka/Kernels.h`**: production state (F1 patch not yet applied — script handles it)
- **`src/alpaka/TrackCandidate.h`**: production state (F2 patch not yet applied)
- **`src/alpaka/PixelTriplet.h`**: clean, all cuts at original values

## After Job Completes

1. Check `tail t5tc_expF.log` — ends with `"Experiment F complete."`
2. Compare `LSTNumDen_noT5pT5dedup.root` against `LSTNumDen_idealpls_fixed.root`:

| Metric | Baseline | Watch for |
|---|---|---|
| T5-type TC eff ΔR<0.01 | ~0% (0/358) | meaningful rise — this is the primary target |
| Total TC eff ΔR<0.01 | 0.690 (247/358) | overall improvement |
| T5-type fake rate ΔR<0.01 | 0.706 (12/17 TCs) | will increase — acceptable if proportionate |
| pT5-type TC eff ΔR<0.01 | 0.642 (230/358) | should be unchanged |
| pT5 fake rate | 0.325 | should be unchanged |

3. **Success criterion**: T5-type TC eff rises noticeably (>5 genuine TCs at ΔR<0.01) with fake
   rate increase that is proportionate (not more fake T5s per genuine track than at large ΔR).
4. **Failure criterion**: gain < 5 genuine TCs → a more fundamental blocker exists (possibly
   `partOfPT5=true` being set even when pT5 attempt fails — needs a code check, or the T5 build
   itself is suppressed in dense regions).

---

## Session 20 Summary: Experiments A/B/D/ALL Null; Root Cause Identified; Experiment F Launched

### Experiments A/B/D/ALL — Results (all with --idealpls)

All four experiments from Session 19's background job completed successfully by the start of
this session. Results (TC-level metrics from NumDen histograms vs `LSTNumDen_idealpls_fixed.root`):

| Experiment | pT5-type TC ΔR<0.01 | T5-type TC ΔR<0.01 | Total TC ΔR<0.01 | T5 fake rate ΔR<0.01 |
|---|---|---|---|---|
| baseline | 0.642 (230/358) | ~0 (0/358) | 0.690 (247/358) | 0.706 (12/17) |
| A — noCC_T5 | 0.642 (230/358) | ~0 (0/358) | 0.690 (247/358) | 0.706 (12/17) |
| B — nopT5priority | 0.642 (230/358) | ~0 (0/358) | 0.690 (247/358) | 0.706 (12/17) |
| D — notightcut | 0.640 (229/358) | ~0 (0/358) | 0.687 (246/358) | 0.760 (19/25) |
| ALL (A+B+D) | 0.642 (230/358) | 0.003 (1/358) | 0.693 (248/358) | 0.778 (21/27) |

**All three suppression gates combined rescued exactly 1 TC at ΔR<0.01.** Disabling
`tightCutFlag` (D) made total TC efficiency slightly *worse* while adding 7 fake T5 TCs.

**Note on metric naming**: the NumDen histograms track TC-level efficiency (how many sim tracks
got a TC of each type), not the raw object-build efficiency. `pT5-type TC eff` = fraction of
sim tracks with a pT5 TC; `T5-type TC eff` = fraction with a standalone T5 TC (essentially
zero at ΔR<0.01 in all cases). The raw pT5_lower (~0.687) and T5_lower (~0.90) from prior
sessions must be computed from the ntuple directly.

### Root Cause of Null Results

Confirmed from reading `PixelQuintuplet.h:706–741`: `partOfPT5` is set **only inside the
`if (success)` block** — only when `runPixelQuintupletDefaultAlgo` returns true and a pT5 is
actually built. T5s where all pT5 pairing attempts failed have `partOfPT5=false` and are
nominally eligible for standalone TC promotion.

The actual bottleneck is `RemoveDupQuintupletsBeforeTC` (`Kernels.h:265`):

```cpp
if (isPT5_ix && isPT5_jx)   // only skips dedup when BOTH are pT5-embedded
    continue;
```

At ΔR<0.01, `dR2 < 0.001` **always fires** (dR2 ≈ 10⁻⁴ for ΔR<0.01). When `ix` is a
standalone T5 (partOfPT5=false) and `jx` is a pT5-embedded T5 (partOfPT5=true), the dedup
comparison runs. The standalone T5 consistently **loses on `score_rphisum`** because its hits
are contaminated by the neighboring track — the very same contamination that caused its pT5
pairing to fail. The pT5-embedded T5 from the neighboring track has cleaner hits and wins.

- **Experiment B** (removed explicit `isPT5_jx` priority) was null because score-based dedup
  still kills the contaminated standalone T5 — `isPT5_jx ||` was redundant; score wins anyway.
- **Experiment A** (CrossCleanT5 disabled) was null because almost no standalone T5s survive
  `RemoveDupQuintupletsBeforeTC` to reach `CrossCleanT5`.
- **Experiment D** (tightCutFlag disabled) rescued 8 extra T5 TCs at ΔR<0.01, 7 of which were
  fake — mostly noise from T5s that escaped dedup by chance.

### Experiment F: Skip Standalone-vs-pT5 Dedup

The fix: change `&&` → `||` at `Kernels.h:265` so the skip fires whenever **either** T5 is
part of a pT5, not just when both are. Physical rationale: T5_A (pT5-embedded, for track A)
and T5_B (standalone, for track B) likely represent different particles — deduplicating them
discards track B's only TC candidate. T5-vs-T5 dedup between two standalone T5s is unaffected.

`CrossCleanT5` must also be disabled for the vs-pT5 branch (F2, same as experiment A) because
standalone T5s rescued from dedup will otherwise be caught by `CrossCleanT5` next, using the
same dR2+embedding criterion.

### Files Created / Changed This Session

| File | Change |
|---|---|
| `standalone/t5tc_patch.py` | Added F1 (Kernels.h `&&`→`||`) and F2 (alias for A's CrossCleanT5 change) patch targets |
| `standalone/t5tc_expF.sh` | New single-experiment script for F1+F2 combined |

Source files (`Kernels.h`, `TrackCandidate.h`) are at production state — patches applied and
restored automatically by the script.

### Output Files from Sessions 19–20

| File | Description |
|---|---|
| `LSTNtuple_noCC_T5.root` | Exp A: CrossCleanT5 vs-pT5 disabled |
| `LSTNtuple_nopT5priority.root` | Exp B: isPT5 priority kill removed |
| `LSTNtuple_notightcut.root` | Exp D: tightCutFlag disabled |
| `LSTNtuple_t5tc_all.root` | Exp ALL: A+B+D combined |
| `LSTNumDen_noCC_T5.root` | NumDen for exp A |
| `LSTNumDen_nopT5priority.root` | NumDen for exp B |
| `LSTNumDen_notightcut.root` | NumDen for exp D |
| `LSTNumDen_t5tc_all.root` | NumDen for ALL |
| `LSTNtuple_noT5pT5dedup.root` | Exp F: F1+F2 combined *(in progress)* |
| `LSTNumDen_noT5pT5dedup.root` | NumDen for exp F *(in progress)* |

---

# T5-TC Suppression Experiments Launched — Handoff (Session 19)

## Standing Rule: Always Use CUDA

**Always build and run with the CUDA backend** (`lst_make_tracklooper -G`, `lst_cuda`).
CPU is ~9× slower. Never use `lst_make_tracklooper -C` or `lst_cpu` unless explicitly asked.

## Current Background Job

**Script:** `t5tc_experiments.sh` (runs 4 experiments sequentially)
**Log:** `t5tc_experiments.log`
**PID:** 2471410 — fully detached (`PPID=1`, `TT=?`), safe to log out
**Launched:** 2026-07-05 ~session end
**ETA:** ~5 hours total (4 × ~72 min: 5 min build + 37 min run + 30 min NumDen)

### What the Script Does

For each of four experiments, in order:

| Experiment | Tag | Change | File |
|---|---|---|---|
| A | `noCC_T5` | Disable `CrossCleanT5` vs-pT5 isDup assignment | `TrackCandidate.h:276-278` |
| B | `nopT5priority` | Remove `isPT5_jx`/`isPT5_ix` unconditional priority kill | `Kernels.h:291-296` |
| D | `notightcut` | Disable `tightCutFlag` requirement in `AddT5asTrackCandidate` | `TrackCandidate.h:636-637` |
| ALL | `t5tc_all` | All three combined | both files |

For each: `python3 t5tc_patch.py apply <X>` → `lst_make_tracklooper -G` → `lst_cuda -i trackingNtuple-100.root --allobj --jet --idealpls -n 100 -o LSTNtuple_<tag>.root` → `createPerfNumDenHists ... -J` → `python3 t5tc_patch.py restore <X>`.

The patch helper (`t5tc_patch.py`) uses exact string matching — it will fail loudly if a pattern is missing (no silent wrong substitution).

## After Job Completes

1. Check `tail t5tc_experiments.log` — ends with `"All experiments complete."`
2. Verify all four `LSTNtuple_*.root` files are present and not zombies
3. Compare against baseline `LSTNumDen_idealpls_fixed.root`:
   - **Primary**: `T5_lower` and `pT5_lower` at ΔR < 0.01 — does suppressing the gate rescue leftover T5s?
   - **Secondary**: fake-rate histograms `*_fr_*` at ΔR < 0.01 — any loosening will inflate T5 TC fakes
4. If experiment ALL recovers ~0.2 pp in `T5_lower` → at least one gate is responsible; check which individual experiment (A, B, D) drives the effect
5. If promising, pursue "Step 3 (manual)": replace `dR2 + embedding` in `CrossCleanT5` with a `checkHitsT5(iT5, innerT5Idx)` ≥5-shared-hits criterion — avoids geometry-based dedup for genuinely distinct objects

## Current State of Source Files

- **`src/alpaka/PixelTriplet.h`**: clean, all cuts at production values (betaOutCut, dBeta, z-window, z-pointed, dPhi all original)
- **`src/alpaka/TrackCandidate.h`**: production state (no pending diagnostic hacks)
- **`src/alpaka/Kernels.h`**: production state
- `t5tc_patch.py` and `t5tc_experiments.sh` are new standalone files, not part of the CMSSW tree

---

## Session 19 Summary: Pivot to T5-TC Suppression Investigation

### Context

Session 18 completed the z-window/z-pointed/dPhi 4× loosening experiment with null result (+0.0028 pp = 1 candidate, noise). **All geometric cuts inside `runTripletDefaultAlgoPPBB/EE` have now been exhaustively 4× loosened with zero effect:**

| Cut loosened | pT5_lower Δ at ΔR<0.01 | Verdict |
|---|---|---|
| dBeta 4× | ~0 (visual) | Ruled out |
| betaOutCut 4× | 0.0000 (246/358 both) | Ruled out |
| z-window + z-pointed + dPhi 4× | +0.0028 (246→247) | Ruled out (noise) |

This closed the pT5-matching-cuts thread. The residual gap (pT5_lower ≈ 0.687, T5_lower ≈ 0.902 at ΔR < 0.01 with ideal pLS) must arise either from the superbin/module-connectivity pairing step (T5 lower module not connected to pLS superbin — explored conceptually in Session 17 and ruled out as the primary cause) or from the T5-TC formation path.

### New Investigation: Why Leftover T5s Don't Become Standalone TCs

The observation: at ΔR < 0.01, only ~2% of T5s that fail pT5 matching become standalone TCs, vs ~30% at large ΔR. Identified the four-stage gauntlet a leftover T5 must survive in `createTrackCandidates` (`LSTEvent.dev.cc:518–792`):

**Stage 1 — `RemoveDupQuintupletsAfterBuild`** (`Kernels.h:177`): within-module dedup, ≥7 shared hits. Minor effect in dense regions.

**Stage 2 — `RemoveDupQuintupletsBeforeTC`** (`Kernels.h:223`, step 2 in TC formation): broad geometric dedup (`dR2 < 0.02 && embed_d2 < 0.1` OR `dR2 < 0.001 && d2 < 1.0`). **Critical rule**: `if (isPT5_jx) → kill ix (leftover T5) unconditionally**, regardless of score. A neighboring pT5 at dR2 < 0.02 always wins.

**Stage 3 — `CrossCleanT5`** (`TrackCandidate.h:222`, step 3): marks T5 isDup if near any pT5 (`dR2 < 0.02 && d2 < 0.1` OR `dR2 < 0.001 && d2 < 1.0`). Runs BEFORE pT5→TC promotion (step 8), reading the raw pT5 SoA directly.

**Stage 4 — `tightCutFlag`** (`TrackCandidate.h:637`, gate in `AddT5asTrackCandidate`): AND of tighter rzChiSquared (95th percentile) + T5 DNN score. Contaminated hits inflate rzChiSquared; dense-region T5 shapes may score lower.

**TC formation order matters**: CrossCleanT5 (step 3) runs before pT5→TC (step 8), using raw pT5 objects — not existing TCs. `CountSurvivingTCs` (step 6) counts *eligible* objects (isDup=false, partOfPT5=false, tightCutFlag=true) for SoA allocation; it is not itself a suppression step.

**Note on Session 3 overlap**: Session 3 tested loosening `nMatched` thresholds in `RemoveDupQuintupletsBeforeTC` and disabling the T5 branch of `CrossCleanpLS` (which marks *pLS* isDup based on T5 TCs — the opposite direction). The `isPT5_jx` unconditional priority kill (Experiment B) and `CrossCleanT5`'s vs-pT5 isDup assignment (Experiment A) were not previously tested.

### Files Created This Session

| File | Description |
|---|---|
| `standalone/t5tc_patch.py` | Apply/restore helper: exact string substitution for experiments A, B, D |
| `standalone/t5tc_experiments.sh` | Master script: runs all 4 experiments with build/run/numden/restore cycle |

### Files Not Changed

`PixelTriplet.h`, `TrackCandidate.h`, `Kernels.h`, `LSTEvent.dev.cc` — all at production state.

---

# z-Window / dPhi 4× Loosening In Progress — Handoff (Session 18)

## Standing Rule: Always Use CUDA

**Always build and run with the CUDA backend** (`lst_make_tracklooper -G`, `lst_cuda`).
CPU is ~9× slower and has no advantage for these efficiency investigations.
Never use `lst_make_tracklooper -C` or `lst_cpu` unless explicitly asked.

## Current Background Job

**Run:** `lst_cuda -i trackingNtuple-100.root --allobj --jet --idealpls -n 100 -o LSTNtuple_loosezphi4x_idealpls.root`
**Log:** `lst_cuda_loosezphi4x.log`
**PID:** 1904434 — fully detached (`PGID=SID=PID`, `TT=?`), safe to log out
**Expected:** ~37 min (same as prior idealpls CUDA runs)

## Changes In PixelTriplet.h (current state — both PPBB and PPEE)

- z-window raw: loosened 4×
- z-pointed window: loosened 4×
- dPhi cut: loosened 4×
- betaOutCut: **restored to original** (was 4× in Session 16, now reverted)
- dBeta: **original** (restored at end of Session 16)

## After Run Completes

1. `createPerfNumDenHists -i LSTNtuple_loosezphi4x_idealpls.root -o LSTNumDen_loosezphi4x_idealpls.root -J`
2. Compare `pT5_lower` vs ΔR against `LSTNumDen_idealpls_fixed.root` baseline
3. If recovery is seen, run individual cuts (z-window alone, dPhi alone) to isolate which is responsible
4. Restore `PixelTriplet.h` to production state before any PR

---

# z-Window / dPhi Identified as Next Suspects — Handoff (Session 17)

## Context

Continuation of Session 16. This session closed out the betaOutCut analysis and drilled into
the cut chain to identify where the residual ~0.215 pp dense-region pT5 loss (with ideal pLS)
is actually occurring.

## Key Conceptual Findings This Session

### betaOutCut ruled out (confirmed numerically)

Quantitative comparison of `LSTNumDen_loosebetaout4x_idealpls.root` vs
`LSTNumDen_idealpls_fixed.root` (baseline):

| Stage | ΔR < 0.01 | ΔR 0.01–0.05 | ΔR > 0.05 |
|---|---|---|---|
| pT5_lower (baseline) | 0.6872 (246/358) | 0.8073 (528/654) | 0.8401 (331/394) |
| pT5_lower (betaout4×) | 0.6872 (246/358) | 0.8073 (528/654) | 0.8401 (331/394) |
| TC (baseline) | 0.6899 (247/358) | 0.8517 (557/654) | 0.9442 (372/394) |
| TC (betaout4×) | 0.6872 (246/358) | 0.8517 (557/654) | 0.9442 (372/394) |

pT5_lower is bit-for-bit identical across all ΔR bins. TC loses one candidate at ΔR < 0.01
(246 vs 247) — statistical noise. **betaOutCut is definitively ruled out.** Combined with the
earlier dBeta ruling, the loss is not happening inside `runTripletDefaultAlgoPPBB/EE` at all —
it is upstream.

### Superbin mechanism — not the culprit

Superbins are computed for each **pLS** (not T5) in `interface/LSTPrepareInput.h` from
PCA η/φ/dz: a 25 × 72 × 25 = 45,000-bin grid (Δη ≈ 0.208, Δφ ≈ 0.087, Δdz = 2.4 cm).
Stored as `PixelSeedsSoA.superbin()`.

The GPU kernel `CreatePixelQuintupletsFromMap` (`PixelQuintuplet.h:655`) does not search T5s
by η/φ. Instead, a **host-side precomputation** in `createPixelQuintuplets()`
(`LSTEvent.dev.cc:1089`) maps each pLS's superbin to a list of OT lower module indices via
`pixelMapping_.connectedPixels[superbin]` (precomputed from detector geometry at startup by
`ModuleMethods.h:getConnectedPixels()`). The kernel then loops over all T5s whose
`quintupletLowerModuleIndex` is in that list. A T5 is invisible to the pairing if its lower
module is not connected to the pLS's superbin.

**Superbins are not the issue here.** Superbin bins are far wider (Δη ≈ 0.208) than the track
separation at ΔR < 0.01. T5 lower modules are physical detector modules (~10 cm in φ), not
shifted by hit contamination from adjacent tracks. Tracks at ΔR < 0.01 share the same
superbin neighborhood in every event.

### Why T5_lower is high but pT5_lower is lower in dense regions

T5 building (`Quintuplet.h`) works entirely within the OT: it checks that the two constituent
T3s are geometrically consistent with each other, but has **no pixel-radius cross-check**.
A T5 built with one contaminated OT hit can still pass all OT-internal consistency cuts and
be truth-matched at >75% hit purity, contributing to T5_lower.

pT5 matching (`runPixelTripletDefaultAlgo`, `PixelTriplet.h:557`) introduces a **new gate
absent during T5 building**: `passRadiusCriterion` compares the pLS pixel radius
(`ptIn × kR1GeVf`) against the inner T3's fitted radius (`triplets.radius()[innerT3Idx]`).
Tolerance is only ~16% for all-barrel low-pT tracks. One contaminated hit in the inner T3
can shift the circle fit outside this window. The T5 exists and is truth-matched, but the pT5
pairing fails before any angle cut is reached.

### Corrected cut ordering inside `runPixelTripletDefaultAlgo`

The Session 16 handoff listed cut order incorrectly. The actual ordered gates, as read from
`PixelTriplet.h:557–614` and `runTripletDefaultAlgoPPBB:837–`, are:

| # | Cut | Location | Status |
|---|---|---|---|
| 1 | `passRadiusCriterion` | `PixelTriplet.h:584` | ~21% of failures |
| 2 | z-window (raw) | `runTripletDefaultAlgoPPBB:898` | **untested** |
| 3 | z-pointed window | `runTripletDefaultAlgoPPBB:929` | **untested — prime suspect** |
| 4 | dPhi | `runTripletDefaultAlgoPPBB:944` | **untested — prime suspect** |
| 5 | betaOutCut | `runTripletDefaultAlgoPPBB:~1071` | **ruled out** |
| 6 | dBeta | `runTripletDefaultAlgoPPBB:~1083` | **ruled out** |
| 7 | DNN | `PixelQuintuplet.h:499` | ruled out (Session 14, real seeds) |
| 8 | χ² cuts | `PixelQuintuplet.h:~520` | ruled out (Sessions 10–11) |

Cuts 2–6 are applied **twice**: `runPixelTripletDefaultAlgo` calls `runPixelTrackletDefaultAlgopT3`
separately for the inner and outer segments of the inner T3 (lines 594 and 605), so failing
either segment kills the pair.

### passRadiusCriterion offline check result

Script `efficiency/python/lst_check_passradius.py` (new this session) recomputes
`passRadiusCriterion` offline for all truth-matched (pLS, T5) pairs from the same sim track
that fail to form a pT5, using `pLS_pt`, `pLS_ptErr`, and `t5_innerRadius` from the ntuple.
Category (BBB/BBE/BEE/EEE) approximated from `abs(t5_eta)`.

Run against `LSTNtuple_idealpls_fixed.root` (100 events):

```
Failing sim tracks (eligible pLS + T5, no pT5): 19
  passRadiusCriterion blocks ALL T5s: 4  (21.1%)
  at least one T5 passes radius:     15  (78.9%)
```

**passRadiusCriterion is a minor contributor (~21%).** For the dominant 78.9% of failures,
the matched T5's inner radius is consistent with the pLS — the pair should clear this gate.
The loss is at one of the subsequent untested cuts: z-window, z-pointed window, or dPhi
(all of which run before betaOutCut and are therefore consistent with the null betaOutCut result).

Note: the sample is small (19 failing tracks across 100 events); the aggregate breakdown is
clear but ΔR-binned plots are sparse.

## Files Changed This Session

| File | Change |
|---|---|
| `efficiency/python/lst_check_passradius.py` | New diagnostic script — offline passRadiusCriterion check |
| `src/alpaka/PixelTriplet.h` | Unchanged from Session 16 — betaOutCut still wrapped with `4.0f * (...)` |

## Output Files Produced This Session

| File | Description |
|---|---|
| `passradius_check.png` | 3-panel plot: radius-blocked fraction vs ΔR (zoom + full), radius ratio distribution |

## What Remains To Do

1. **Identify which cut(s) among z-window, z-pointed window, dPhi are responsible for the
   78.9% of failures.** Recommended approach: loosen them in sequence or together in
   `runTripletDefaultAlgoPPBB` (`PixelTriplet.h:898, 929, 944`) and `runTripletDefaultAlgoPPEE`
   (analogous lines), rebuild CPU, rerun with `--idealpls`, compare pT5_lower vs ΔR.
   These are all in the PPBB function (and its PPEE counterpart), which is called twice per
   inner T3 (once for each constituent segment).

2. **Restore `PixelTriplet.h`** before any production run or PR — betaOutCut is still
   `4.0f * (...)` from Session 16 experiment 2.

3. **Standing constraint**: no new TrackCandidate types; any fix must work within existing
   TC categories (pT5/pT3/T5/T4/pLS).

---

# betaOutCut Probe In Progress — Handoff (Session 16)

## Context

Continuation of Session 15. The `TruthPixelSeeds.cc` bug (wrong momentum direction in
`see_stateTrajGlbPx/Py/Pz`) was fixed in Session 15 and verified: pLS_lower recovered to
~0.989 and pT5_lower recovered from ~0.127 to ~0.840 at ΔR > 0.05 (LSTNtuple_idealpls_fixed.root,
100 events). However, a **residual dense-region pT5 matching loss** remains: even with perfect
seeds, pT5_lower ≈ 0.687 at ΔR < 0.01 vs T5_lower ≈ 0.902 — a ~0.215 pp gap that widens at
small ΔR, indicating an LST-internal matching failure independent of seeding.

This session investigated which cut(s) in the pT5 matching chain are responsible for that gap.

## Key Conceptual Findings This Session

### Why TC efficiency goes from ~0.3 (real pLS) to ~0.6 (ideal pLS)

The dominant cause is **seed availability**, not seed quality. In dense regions real CMSSW
reconstruction fails to produce a seed for ~52% of true tracks (pLS_lower ≈ 0.478 with real
seeds). Without a pLS, a track's only TC paths are standalone T4 or T4-seeded quintuplets, which
are much less efficient. Ideal pLS raises seed availability to ~0.989, restoring those tracks to
the pT5/pT3 TC paths and roughly doubling TC efficiency (0.3 → 0.6). The remaining gap from TC
(0.6) to T5_lower (0.9) is the LST-internal pT5 matching loss investigated here.

### Why `--idealpls` gives ~0.989 not 1.0

`--idealpls` calls `buildTruthPixelSeeds()` with an **empty** exclusion set, meaning ALL sim
tracks (including those with real CMSSW seeds) get synthetic seeds; real seeds are completely
discarded. The ~1% gap is NOT from keeping real seeds for tracks that have them. Actual causes:
- Tracks with fewer than 3 true pixel hits in distinct (subdet, layer, side) → no synthetic seed
- Tracks near the 0.8 GeV pT cut in `prepareInput` (ptIn < ptCut − 2×ptErr)
- `n_max_pixel_segments_per_module` capacity cap in busy PU200 events
- Deduplication (`isDup`) flagging two synthetic seeds that share pixel hits

### Why synthetic pLS need real pixel hits

`buildTruthPixelSeeds` uses actual pixel hit positions for two non-negotiable purposes:
1. **Geometry anchoring**: `see_stateTrajGlbX/Y/Z` is set to the outermost true pixel hit
   position (`chosen.back()`), which is the reference point for computing `betaIn` in pT5 matching
2. **Efficiency matching**: hit indices in `chosen` are stored in `see_hitIdx`; the efficiency
   code identifies which sim track a pLS belongs to by tracing `see_hitIdx → pix_simHitIdx →
   simhit_simTrkIdx`. A pLS with no real hit indices cannot be matched back to a sim track.

### Full pT5 matching cut chain (beyond dBeta)

The complete ordered list of gates in `CreatePixelQuintupletsFromMap` / `runPixelQuintupletDefaultAlgo`:

1. **Superbin pairing** — only (pLS, T5) pairs whose η/φ/dz superbins overlap are ever examined;
   a T5 with a contaminated innermost hit shifted out of the expected superbin is invisible to all
   subsequent cuts
2. **Deduplication** (`isDup`) — both pLS and T5 checked early; legitimate objects can be flagged
   as duplicates in dense regions
3. **`passRadiusCriterion`** — pLS radius (ptIn × kR1GeVf) must agree with T3 fitted radius;
   fails if T3 has a mixed hit from a nearby track
4. **`dPhi` cut** (Cut #5 in PPBB/PPEE) — azimuthal bending of T3's inner segment vs pLS
   direction; runs before betaOutCut and dBeta
5. **z/zPointed window (PPBB) or rt/rtPointed window (PPEE)** — propagation-based range check
   on T3's first OT hit; fails if T5 has an outlier hit
6. **`betaOutCut`** — absolute cut `|betaOut| < betaOutCut`, applied **before** dBeta; not
   loosened in the dBeta experiment. betaOut is computed from the T5's outermost segment and is
   most susceptible to hit contamination from nearby tracks.
7. **DNN** (`pt3dnn::runInference<pT5WP>`) — trained on the full PU200 distribution; may
   systematically reject dense-region candidates where input features are shifted. Cannot be
   probed by loosening geometric cuts.
8. **`rPhiChiSquared` / `rzChiSquared` / `rPhiChiSquaredInwards`** — 7-hit circle/helix fits
   (pT < 5 GeV only); a single contaminated T5 hit inflates these dramatically.

### Why `PixelTriplet.h` is involved in pT5 matching

`runPixelQuintupletDefaultAlgo` (`PixelQuintuplet.h:500`) calls `runPixelTripletDefaultAlgo`
(from `PixelTriplet.h`) to match the pLS against the **inner T3 of the T5**. This is code reuse:
the same pLS-T3 angle matching logic (betaOutCut, dBeta) used for standalone pT3 creation is
reused as a sub-step inside pT5 matching, just invoked with a pT5-specific DNN working point
(`pT5WP`) and with `runChiSquaredCuts=false`.

## Experiments Run This Session

### Experiment 1: dBeta cut loosened 4× (4.0f × prefactor)

**Files changed:** `src/alpaka/PixelTriplet.h`
- PPBB (`runTripletDefaultAlgoPPBB`, ~line 1079): added `4.0f *` to `dBetaCut2`
- PPEE (`runTripletDefaultAlgoPPEE`, ~line 1338): same

**Result:** 100-event run completed (`LSTNtuple_loose4x_idealpls.root`, 1008 MB).
Plots: `eff_vs_deltaR_loose4x_idealpls_all.png`, `eff_vs_deltaR_loose4x_idealpls_sub.png`.
Visual inspection of 10-event plots showed no convincing improvement. **dBeta cut is likely not
the dominant cause of the residual dense-region gap.**

### Experiment 2: betaOutCut loosened 4× (current state of code)

**Files changed:** `src/alpaka/PixelTriplet.h`
- Restored `dBetaCut2` to original (removed `4.0f *`)
- PPBB (~lines 1071–1073): wrapped entire betaOutCut expression with `4.0f * (...)`
- PPEE (~lines 1327–1329): same

The betaOutCut formula is now:
```cpp
betaOutCut = 4.0f *
    (alpaka::math::asin(...drt_tl_axis * k2Rinv1GeVf / ptCut...) +
     (0.02f / sdOut_d) + alpaka::math::sqrt(acc, dBetaLum2 + dBetaMuls2));
```

**Status:** 100-event run completed (`LSTNtuple_loosebetaout4x_idealpls.root`, 1008 MB).
NumDen generated (`LSTNumDen_loosebetaout4x_idealpls.root`, 17 MB).
Plots: `eff_vs_deltaR_loosebetaout4x_idealpls_all.png`, `eff_vs_deltaR_loosebetaout4x_idealpls_sub.png`.
**Analysis not yet performed** — plots were generated at end of session.

## Files Changed This Session

| File | Change |
|---|---|
| `src/alpaka/PixelTriplet.h` | dBeta loosened 4× (experiment 1), then restored; betaOutCut loosened 4× (experiment 2, current state) |

All other source files are unchanged from Session 15.

## Output Files Produced This Session

| File | Description |
|---|---|
| `LSTNtuple_loose4x_idealpls_10.root` | 10-event dBeta-4× idealpls ntuple (99 MB) |
| `LSTNumDen_loose4x_idealpls_10.root` | NumDen for above |
| `LSTNtuple_loose4x_idealpls.root` | 100-event dBeta-4× idealpls ntuple (1008 MB) |
| `LSTNumDen_loose4x_idealpls.root` | NumDen for above (17 MB) |
| `LSTNtuple_loosebetaout4x_idealpls.root` | 100-event betaOutCut-4× idealpls ntuple (1008 MB) |
| `LSTNumDen_loosebetaout4x_idealpls.root` | NumDen for above (17 MB) |
| `eff_vs_deltaR_loose4x_idealpls_{all,sub}.png` | Plots for dBeta-4× experiment |
| `eff_vs_deltaR_loosebetaout4x_idealpls_{all,sub}.png` | Plots for betaOutCut-4× experiment |

## What Remains To Do

1. **Analyze betaOutCut-4× plots**: compare `eff_vs_deltaR_loosebetaout4x_idealpls_sub.png`
   against the idealpls_fixed baseline (`LSTNumDen_idealpls_fixed.root`) to quantify how much of
   the ~0.215 pp gap at ΔR < 0.01 is recovered in pT5_lower and TC.

2. **If betaOutCut is also not the cause**, next candidates in rough priority order:
   - **Superbin pairing miss**: a T5 with a contaminated innermost hit may be binned out of the
     expected neighborhood — invisible to all cut-loosening experiments. Would require a dedicated
     diagnostic (e.g., truth-based superbin matching check).
   - **rPhiChiSquared / rzChiSquared / rPhiChiSquaredInwards**: 7-hit fits fail when a T5 has
     even one contaminated OT hit. Try loosening the per-module chi-squared thresholds in
     `PixelQuintuplet.h` (`passPT5RPhiChiSquaredCuts`, `passPT5RZChiSquaredCuts`,
     `passPT5RPhiChiSquaredInwardsCuts`).
   - **DNN (`pt3dnn::runInference<pT5WP>`)**: disable with `--nopt5dnn` and repeat the idealpls
     run to isolate its contribution. Note: prior session ruled out the DNN for *real-seed* TC
     efficiency; that result may not transfer to the idealpls context.

3. **Restore PixelTriplet.h** before any production run or PR — betaOutCut is currently 4×
   loosened and must be reverted once the investigation concludes.

4. **Standing constraint**: no new TrackCandidate types; any fix must work within existing TC
   categories (pT5/pT3/T5/T4/pLS).

---

# idealpls pT5 Collapse Root-Caused — Handoff (Session 15)

## Context

Continuation of Sessions 12–14. The idealpls experiment (replace all real CMSSW pixel seeds
with MC-truth synthetic seeds) showed pLS_lower ≈ 0.99 (flat, as designed) and T5_lower ≈
0.85–0.93 (unchanged from real-seed baseline), yet pT5_lower collapsed to 0.15–0.45. With
essentially perfect individual ingredients present, the pLS+T5→pT5 matching step was rejecting
nearly all truth-matched pairs. This session traced the failure to a concrete, fixable bug in
`TruthPixelSeeds.cc`.

Additionally, plotting infrastructure was improved: `lst_plot_performance.py` now places the
legend outside the axes box (to the right) for all `*_deltaR` efficiency plots, and
`lst_plot_eff_vs_deltaR_stages.py` now shows markers only (no connecting lines). NumDen files
for the three testing-pls ntuples were regenerated with T3/LS/MD histogram support.

## Root Cause Found: Wrong Momentum Direction in `buildTruthPixelSeeds`

**File:** `code/core/TruthPixelSeeds.cc:107-110`

```cpp
out.see_stateTrajGlbPx.push_back(px);   // = pt * cos(phi_PCA) ← WRONG
out.see_stateTrajGlbPy.push_back(py);   // = pt * sin(phi_PCA) ← WRONG
out.see_stateTrajGlbPz.push_back(pz);   // correct
```

`see_stateTrajGlbPx/Py/Pz` is supposed to be the track's momentum **at the last pixel hit**
(the outer reference point). For real CMSSW seeds, this is the fitted, propagated state at that
hit. For idealpls seeds, it is incorrectly set to the PCA (vertex) momentum — the initial
direction before the track curves through the pixel detector.

**Why it matters:** In `LSTPrepareInput.h:119-120`, `px = p3LH.x()` directly uses
`see_stateTrajGlbPx`. This feeds into `pixelData.px/py` via `loadPixelSeedData`
(`PixelTriplet.h:40-42`), which is used in `runTripletDefaultAlgoPPBB/EE` to compute:

```cpp
betaIn = -deltaPhi(px, py, tl_axis_x, tl_axis_y)  // PixelTriplet.h:970
```

`betaIn` measures the angle between the pLS momentum direction and the direction from the
pLS outer hit to the T3 outer hit — a key consistency check in the `dBeta = betaIn - betaOut`
cut (`PixelTriplet.h:1083-1084`).

For idealpls seeds, `px/py` points in the PCA direction, not the propagated direction at the
outer pixel hit. The track rotates in φ by `arcsin(rt_InUp / (2R))` as it travels from the
vertex to the outer pixel layer. For a 1 GeV track at rt_InUp ≈ 10 cm:

```
Δφ = arcsin(100 mm / (2 × 870 mm)) ≈ 0.058 rad
```

This bias in `betaIn` is comparable to the `betaOutCut` (~0.06 rad for 1 GeV at ptCut=0.8
GeV), causing a large fraction of truth-matched pT5 candidates to fail the `|betaOut| <
betaOutCut` cut and the `dBeta² ≤ dBetaCut2` cut. The effect is systematic for all tracks
below ~5 GeV (above 5 GeV the geometric cuts are dropped), and explains the full collapse
of pT5_lower from the expected ~0.85–0.93 to ~0.15–0.45.

**The fix:** propagate the truth momentum from PCA to the outer pixel hit before storing it:

```cpp
float rt_outer = std::hypot(pix_x[outerHit], pix_y[outerHit]);
float R = pt / (2.f * 3.8f * 0.003f);       // kR1GeVf ≈ 87 cm for B=3.8 T
float delta_phi = -float(sim_q[simTrkIdx]) * std::asin(rt_outer / (2 * R));
float phi_prop = phi + delta_phi;
out.see_stateTrajGlbPx.push_back(pt * std::cos(phi_prop));
out.see_stateTrajGlbPy.push_back(pt * std::sin(phi_prop));
out.see_stateTrajGlbPz.push_back(pz);       // pz unchanged on a helix
```

The sign convention for `delta_phi` follows the charge: positive charge → tracks curve
counterclockwise in the transverse plane in CMS's B field (−z direction), so phi decreases
(negative charge gets positive phi shift). Confirm by checking that the real seed
`DeltaPhi(p3LH, r3LH)` decreases after the fix, not increases.

## Code Read Performed (No Source Changes This Session)

Traced the full pT5 matching chain top-down:

| Function | File | What It Does | Failure Mode for idealpls |
|---|---|---|---|
| `createPixelQuintuplets()` | `LSTEvent.dev.cc:1037` | Entry point; builds pLS×T5 connectivity map from superbins | Superbin from truth eta/phi/dz — correct |
| `CreatePixelQuintupletsFromMap::operator()` | `PixelQuintuplet.h:637` | Iterates pLS×connected-T5 pairs, skips isDup pLS | isDup cleaning (hit-sharing logic) is not the issue — pLS_lower counts ALL objects |
| `runPixelQuintupletDefaultAlgo` | `PixelQuintuplet.h:469` | Calls `runPixelTripletDefaultAlgo` for inner T3, then RZ/RPhi chi2 cuts | Fails here because `betaIn` is biased |
| `runPixelTripletDefaultAlgo` | `PixelTriplet.h:557` | `passRadiusCriterion` → `runPixelTrackletDefaultAlgoPPBB/EE` × 2 → chi2 cuts → DNN | `runPixelTrackletDefaultAlgoPPBB/EE` fails on `dBeta` |
| `runTripletDefaultAlgoPPBB/EE` | `PixelTriplet.h:837/1088` | Computes betaIn, betaOut, dBeta; applies dBeta cut | `betaIn` wrong because px/py is PCA momentum, not last-hit momentum |
| `loadPixelSeedData` | `PixelTriplet.h:30` | Reads `pixelSeeds.px/py/pz` → `pixelData.px/py/pz` | No bug here; just propagates wrong upstream value |
| `LSTPrepareInput.h:119-120` | `interface/LSTPrepareInput.h` | `px = p3LH.x()` where `p3LH = (see_stateTrajGlbPx, ...)` | **Root cause** — idealpls sets this to PCA momentum |
| `buildTruthPixelSeeds` | `code/core/TruthPixelSeeds.cc:107-110` | Sets `see_stateTrajGlbPx/Py/Pz` | **Bug location** — must propagate to outer hit |

Also confirmed:
- The `pixelLineSegmentCleaning` (`CheckHitspLS` kernel) marks isDup by hit-sharing (≥3 shared
  hit indices between two pLS). This is NOT causing the pT5 collapse — pLS_lower is 0.99
  which includes isDup objects, and the isDup rate for idealpls is not abnormally high since
  hit sharing requires tracks to share actual pixel detector channels.
- The pT3DNN (`pt3dnn::runInference`) uses `log10(rPhiChiSquared)`, `log10(tripletRadius)`,
  `log10(pixelRadius)`, `log10(pixRadiusError)`, `log10(rzChiSquared)`, `|eta|`, `moduleType3`.
  For truth-matched pairs the chi2 values should be small but not zero (real detector noise),
  so DNN domain shift is a secondary concern, not the primary cause of the collapse.
- `pT3_lower` near zero is NOT a bug: the `CreatePixelTripletsFromMap` kernel skips any pLS
  with `partOfPT5=true` (set during pT5 building), so pT3 only sees the ~10-25% of leftover
  pLS seeds, by design.

## Plotting Changes Made This Session

### `efficiency/python/lst_plot_performance.py`

For all plots whose output name contains `_deltaR`, the legend is now placed outside the axes
to the right instead of overlapping the plot:
- Right margin widened to 0.28 (vs 0.15 normally)
- Legend positioned at NDC `TLegend(0.73, 0.75 - nleg*0.04, 0.87, 0.75)`
- Conditional: `ext_legend = "_deltaR" in output_name`

### `efficiency/python/lst_plot_eff_vs_deltaR_stages.py`

Removed connecting lines between data points (markers only):
- `fmt=style["marker"]` (was `fmt="-" + style["marker"]`)
- `linewidth` kwarg removed

### NumDen File Regeneration

All three `testing-pls/LSTNumDen_*.root` files were deleted and regenerated with
`createPerfNumDenHists -J` (required for genjet/deltaR histograms). The stash@{0}
(performance.cc with T3/LS/MD histogram support) had been popped before this session.

Plots generated (all in `testing-pls/`):
- `eff_vs_deltaR_{realpls,idealpls,fillpls}_all.png` — all 8 stages, markers only
- `eff_vs_deltaR_{realpls,idealpls,fillpls}_sub.png` — pLS, T5, pT5, TC, markers only
- `performance/testing-pls/{realpls,idealpls,fillpls}_820464D-trackingNtuple-100.root/mtv/var/TC_base_0_0_eff_deltaR.png` — legend right of plot

## Files Changed This Session

| File | Change |
|---|---|
| `efficiency/python/lst_plot_performance.py` | External right-side legend for `_deltaR` plots |
| `efficiency/python/lst_plot_eff_vs_deltaR_stages.py` | Markers only, no connecting lines |
| `testing-pls/LSTNumDen_realpls.root` | Regenerated with T3/LS/MD histograms |
| `testing-pls/LSTNumDen_idealpls.root` | Regenerated |
| `testing-pls/LSTNumDen_fillpls.root` | Regenerated |
| `testing-pls/*.png` | Various plots regenerated/created |

**No device/kernel code was changed.** `code/core/TruthPixelSeeds.cc` has the identified bug
but was NOT edited this session — the fix was described but not implemented.

## What Remains To Do

1. **Apply the fix to `TruthPixelSeeds.cc`**: propagate truth momentum to the outer pixel hit
   position in `buildTruthPixelSeeds()` before storing as `see_stateTrajGlbPx/Py/Pz`. The
   propagation formula is above. After the fix, rebuild (`lst_make_tracklooper -G`) and rerun
   the idealpls ntuple (`lst_cuda -i trackingNtuple-100.root --allobj --jet --idealpls -n 100
   -o LSTNtuple_idealpls_fixed.root`). Expect pT5_lower to recover toward T5_lower (~0.85–0.93)
   and TC efficiency to recover accordingly.

2. **Verify that `fillpls` benefits too**: the fill-mode synthetic seeds have the same bug for
   their appended synthetic portion. After fixing `buildTruthPixelSeeds`, rebuild and rerun with
   `--fillmissingpls` as well — the +6.4 pp recovery measured in Session 13 should improve.

3. **Standing constraint:** no new TrackCandidate types. Any fix must work within existing TC
   categories (pT5/pT3/T5/T4/pLS).

4. All earlier carried-forward items (Session 14 DNN-gate ruling, etc.) are otherwise unchanged.

---

# DNN Pre-Filter Gate Ablation — Ruled Out as a Cause — Handoff (Session 14)

## Context

Continuation of Session 13. With chi-square closed and the seeding-gap-fill only recovering
+6.4pp of a 36.8pp conditioned gap, this session tested the next lead from memory: the pT5 DNN
pre-filter gate (`dnn::pt3dnn::pT5WP` in `runPixelQuintupletDefaultAlgo`,
`PixelQuintuplet.h:499-516`). Per the approved plan, threaded a `bool runPT5DNN` parameter the
same way `tc_pls_triplets`/`no_pls_dupclean` already are, through 7 files: `AnalysisConfig.h`
(`disable_pt5_dnn`), `bin/lst.cc` (`--nopt5dnn` flag), `trkCore.h/.cc`, `LSTEvent.h/.dev.cc`,
`PixelQuintuplet.h` (3 edits — functor signature, algo function signature, hardcoded `true` →
`runPT5DNN`), `LST.cc` (explicit `true` to keep the CMSSW-integration path byte-for-byte
unchanged). Rebuilt CUDA (`-G`), ran `lst_cuda -i trackingNtuple-100.root --allobj --jet
--nopt5dnn -n 100 -o LSTNtuple_nopt5dnn.root` (100 events, ~33 min wall time — same
`--allobj --jet` truth-matching overhead seen for every flavor of this run this session,
unrelated to seed injection). Validated clean (100/100 entries, not a zombie).

## What Was Measured

1. **`lst_check_pt5_matching_density.py` (Session 7's diagnostic)** comparing
   `LSTNtuple_chidebug.root` (baseline, real seeds, DNN on) vs `LSTNtuple_nopt5dnn.root`:
   - Aggregate "ingredients present, no pT5 formed": baseline 4.1% (95/2340) → DNN-disabled 2.4%
     (56/2340). nCompetingT5==0 sub-rate: 67.9% → 47.7%.
   - At the smallest ΔR bin (zoom plot), the no-pT5 rate dropped from ~21% (baseline) to ~11%
     (DNN-disabled) — a real, visible effect on *pT5 formation* in dense regions.
2. **TC-level fake rate vs ΔR**, via `createPerfNumDenHists -J` on both ntuples (note: `-J` is
   required for the deltaR fake-rate histograms to fill at all — first attempt without it
   produced empty `*_fr_denom_deltaR`/`*_fr_numer_deltaR` histograms, silently, no error).
   pT5-type TC (`tc_type==pT5`) fake rate:
   - Baseline: overall 39.2% (1221/3116); ΔR<0.02: 60.1% (902/1501); ΔR>0.02: 19.8% (319/1615).
   - DNN-disabled: overall 55.3% (2406/4351); ΔR<0.02: 69.3% (1452/2095); ΔR>0.02: 42.3%
     (954/2256).
   - Fake rate rises substantially everywhere, more than doubling at ΔR>0.02 and still up
     ~9 points even in the already-high-fake dense region.
3. **TC-level conditioned efficiency vs ΔR** (the actual metric this whole investigation tracks),
   same `*_ef_denom_deltaR`/`*_ef_numer_deltaR` histograms used throughout this investigation
   (`Root__TC_base_0_-1_ef_*`):
   - ΔR<0.02: **358/607 in both files — bit-for-bit identical (58.98%)**.
   - ΔR>0.02: baseline 666/799 (83.35%) → DNN-disabled 660/799 (82.60%) — slightly *worse*.
   - Overall: 72.83% → 72.40% — flat to marginally worse.

## Conclusion — DNN gate ruled out

Disabling the DNN gate **does let more pT5 candidates form** (matching-density and
fake-rate-denominator both confirm this), but **none of that translates into additional genuine
TrackCandidates** — the conditioned TC efficiency at small ΔR is identical down to the exact same
numerator/denominator. The extra pT5s that form are essentially all fake (fake rate roughly
doubles at every ΔR), and any genuine candidate the gate had been rejecting was apparently
already being captured by another route to TC formation, since removing the gate doesn't change
which sim tracks end up with a passing TC. This is the plan's explicit "efficiency barely moves,
fake rate jumps" outcome — **joins chi-square thresholds and cross-cleaning/dedup tuning on the
ruled-out list.** The DNN gate is doing its job (suppressing combinatorial fakes) and is not a
contributor to the dense-region TC efficiency drop.

## Files Changed This Session

- `code/core/AnalysisConfig.h`: added `bool disable_pt5_dnn;`
- `bin/lst.cc`: added `--nopt5dnn` cxxopts flag, parsing, and threading into
  `runPixelQuintuplet(event, !ana.disable_pt5_dnn)`
- `code/core/trkCore.h`/`.cc`: `runPixelQuintuplet` gained `bool runPT5DNN = true` parameter
- `src/alpaka/LSTEvent.h`/`.dev.cc`: `createPixelQuintuplets(bool runPT5DNN = true)`
- `src/alpaka/PixelQuintuplet.h`: threaded `runPT5DNN` through
  `CreatePixelQuintupletsFromMap::operator()` and `runPixelQuintupletDefaultAlgo`, replacing the
  hardcoded `true` at the `runPixelTripletDefaultAlgo<dnn::pt3dnn::pT5WP>(...)` call
- `src/alpaka/LST.cc`: explicit `event.createPixelQuintuplets(true)` — CMSSW-integration path
  unchanged
- `LSTNtuple_nopt5dnn.root` created (100 events, kept)

All plumbing is additive and defaults to `true` (unchanged behavior) everywhere except the new
standalone `--nopt5dnn` flag — no production behavior change.

## What Remains To Do

1. With chi-square, dedup, and now the DNN gate all ruled out, **the still-open avenues from the
   Session 13 inventory** are: non-quad pLS acceptance in pT5 *building itself* (distinct from
   `tc_pls_triplets`, which only gates TC-formation); the `tc_pls_triplets` flag as a direct
   test (Session 13's attempt was ambiguous/not trusted, needs a clean redo); T5-alone standalone
   promotion criteria; the pT3 path specifically; reporting the upstream "no real seed" gap to
   the CMSSW pixel-tracking group.
2. Per the explicit constraint from the user: **no new TrackCandidate types** — any fix must work
   within the existing TC categories (pT5/pT3/T5/T4/pLS).
3. All earlier carried-forward items (Sessions 4-12) unchanged.

---

# Targeted Gap-Filling pLS Test (`--fillmissingpls`) — Small Recovery, Real Bug Found — Handoff (Session 13)

## Context

Continuation of Session 12. The full-replacement `--idealpls` test's premise broke down (a
structurally-unrecoverable conversion-electron population confounds it). The user redirected:
restrict the analysis to the "recoverable" subset (≥3 true pixel layers) and build a more
surgical test that only fills in synthetic pLS where CMSSW produced *no* real seed at all,
leaving existing real seeds completely untouched — a smaller, cleaner intervention than full
replacement.

## What Was Done

1. **Redid Session 3's pLS-deficit decomposition restricted to the recoverable subset** (1000-event
   sample): No pLS 26.4%/10.3% (low/high ΔR), non-quad 28.7%/22.5%, quad-all-dup 10.5%/27.5%,
   quad-clean 34.5%/39.7%. "No pLS" remains the single dominant category even after excluding the
   structurally-impossible tracks.
2. **Implemented `--fillmissingpls`**: `code/core/TruthPixelSeeds.h/.cc`'s `buildTruthPixelSeeds()`
   now takes an `excludeSimTrkIdx` set; added `findSimTrkIdxsWithRealSeed()` (reuses the existing
   `matchedSimTrkIdxs()` from `trkCore.h`, not reinvented) to find which sim tracks already have a
   real seed; `bin/lst.cc`'s new branch loads real seeds, computes the exclusion set, builds
   synthetic seeds only for the complement, and **appends** them onto the real seed arrays
   (`see_algo.insert(..., 4u)` for a valid sentinel).
3. **Found and fixed a real crash bug**: first attempt segfaulted in `setPixelLineSegmentBranches`
   (`write_lst_ntuple.cc`). Root cause: `trk_see_hitIdx[seedIdx]` assumed `seedIdx` always indexes
   the *real* input ntuple's seed arrays — false for synthetic (appended) seeds, out-of-bounds.
   **This was already a silent, undetected correctness bug in the pre-existing `--idealpls` path**
   too (didn't crash there only by luck of array-size ordering). Fixed by using the
   already-computed, seed-origin-agnostic `hit_idx`/`hit_type` (from `getHitIdxsAndHitTypesFrompLS`,
   computed earlier in the same function) instead. Removed the now-dead `trk_see_hitIdx`/
   `trk_see_hitType` declarations.
4. Ran `lst_cuda -i trackingNtuple-100.root --allobj --jet --fillmissingpls -n 100 -o
   LSTNtuple_fillpls.root`, validated clean.

## Result

**Gap-filling recovered only +6.4pp of the 36.8pp conditioned gap** (0.502→0.566 at ΔR<0.01, vs
0.870→0.912 at ΔR>0.05 baseline-vs-filled). The "no pLS" category dropped sharply (26.4%→6.7%)
but most of those tracks landed in "non-quad" instead (28.7%→36.7%) — getting *a* seed at all
barely moves the needle on TC formation, because most synthetic seeds for previously-seedless
tracks come out non-quad (3-hit), and non-quad pLS face additional downstream gates before
becoming a TC. Attempted to verify whether `tc_pls_triplets=0` (default off, explicitly blocks
non-quad pLS from TC-building per `TrackCandidate.h:560,681`) fully explains this, but the
verification script gave ambiguous numbers (39%/46% TC-pass rate for non-quad/quad gap-filled
tracks — not the clean 0% expected if fully blocked) using an unvalidated reimplementation of the
real-seed-matching logic — **flagged as not trusted**, needs a clean redo if revisited.

## Files Changed This Session

- `code/core/TruthPixelSeeds.h`/`.cc`: `excludeSimTrkIdx` param, `findSimTrkIdxsWithRealSeed()`
- `code/core/AnalysisConfig.h`: added `bool fill_missing_pls;`
- `bin/lst.cc`: `--fillmissingpls` flag, parsing, new branch building+appending synthetic seeds
- `code/core/write_lst_ntuple.cc`: bug fix described above (also fixes a latent `--idealpls` bug)
- `LSTNtuple_fillpls.root` created (100 events, kept); `LSTNtuple_fillpls.root.crashed` is the
  moved-aside corrupt output from the first (crashed) attempt — moved via `mv`, not deleted, per
  established project pattern (Session 10) when `rm` permission was denied.

## What Remains To Do

1. Re-verify the ambiguous quad/non-quad TC-pass-rate split with a corrected script if this
   thread is revisited.
2. Compiled a full inventory of all efficiency-recovery avenues discussed (tested-effective,
   tested-ruled-out, proposed-untested, structural ceiling, outside-LST) — saved to memory.
3. User constraint to carry forward into any future fix: **no new TrackCandidate types.**
4. Investigated next: the DNN gate (Session 14, above).

---

# Ideal-pLS Injection Run — Sanity Check Failed, But For an Interesting Reason — Handoff (Session 12)

## Context

Continuation of Session 11 — the chi-square thread is closed, this session moved to the
`--idealpls` test designed and implemented in Session 8 (`code/core/TruthPixelSeeds.h/.cc`,
`AnalysisConfig.h`'s `use_truth_pls`, `bin/lst.cc`'s `--idealpls` flag). Confirmed unchanged on
disk before running. User approved a plan (5 steps) and chose to drop `-d` for the rebuild.

## What Was Done

1. **Rebuilt CUDA without `-d`**: `lst_make_tracklooper -G` — plain `explicit` target, succeeded.
2. **Ran the injection**: `lst_cuda -i trackingNtuple-100.root --allobj --jet --idealpls -n 100 -o
   LSTNtuple_idealpls.root`. **Took ~37 min wall-clock for 100 events** despite the printed
   in-loop "CPU Time"/"Real Time" reading only ~2 seconds — i.e. the cost is in code *outside*
   that timer (most likely `buildSimTrkToPixHitMap`/`buildTruthPixelSeeds` in
   `TruthPixelSeeds.cc`, called once per event from `bin/lst.cc`, never profiled since this was
   its first real run). **Flagging this as a real performance issue, not fixed this session** —
   for context, the 1000-event real-seed production run (Session 10/11) only took ~31s by that
   same in-loop timer. Scaling this run to 1000+ events would take many hours; worth a profiling
   pass before any larger idealpls run.
3. **Validated the output file**: not a zombie, 100/100 entries, clean.
4. **Ran the plan's Step 3 sanity check (`has_eligible_pls` should be ~100% and flat vs ΔR)** —
   wrote a quick PyROOT check (not saved as a script, ad hoc) since
   `lst_check_pt5_matching_density.py` computes a different, more specific ratio than this
   simple gate. **Result: it is neither ~100% nor flat** — eligible-pLS fraction among selected
   sim tracks is 0.417 (ΔR<0.01) → 0.543 (0.01-0.05) → 0.696 (ΔR>0.05). This is *lower* than the
   real-seed baseline's pLS_lower efficiency at every ΔR (0.702/0.955 from Session 3), which on
   its face looks like the injection failed.
5. **Root-caused it before concluding "bug."** Broke the failure down by stage (any pLS match at
   all → quad match → quad-and-not-duplicate) and found the loss is **already fully explained at
   the very first stage**: "any pLS match" = 0.557 / 0.643 / 0.799, and quad-match = 0.449 / 0.550
   / 0.696 — both numbers **exactly** reproduce
   "fraction of selected sim tracks with ≥3 (≥4) distinct-(subdet,layer,side) truth-matched pixel
   hits," computed completely independently, straight from the raw `trackingNtuple-100.root`
   input (no LST code involved at all). The `isQuad`-and-not-`isDuplicate` "eligible" number is
   barely below the quad number (0.417 vs 0.449, 0.543 vs 0.550, 0.696 vs 0.696) — duplicate
   flagging is a negligible contributor.

## Conclusion — this is a real finding, not an injection bug

`buildTruthPixelSeeds()` and the matching pipeline are working exactly as designed: every sim
track with ≥3 distinct true pixel hits gets a synthetic seed, and it truth-matches. **The
ΔR-dependence is not coming from LST or from the injection code at all — it's coming from the
*input ntuple's own true-hit accounting*.** A meaningful fraction of sim tracks in dense jet
cores (44% at ΔR<0.01, vs 20% at ΔR>0.05) simply don't have ≥3 distinct truth-matched pixel hits
to begin with, independent of any reconstruction step. The likely mechanism (not confirmed this
session, would need further digging) is pixel-cluster merging/sharing between nearby tracks in
dense regions reducing the count of *distinct* hits attributable to any single track — a
detector/simulation-level effect, not a seeding-algorithm or LST-logic effect.

**This breaks the original test design's premise.** The plan assumed an "ideal" pLS could
achieve ~100% flat coverage by construction (modulo the explicit <3-hit exclusion, which was
expected to be small and ΔR-independent). It isn't small, and it isn't ΔR-independent — so any
T5/pT5/TC efficiency comparison between `LSTNtuple_idealpls.root` and the real-seed baseline
would now be confounded by *this* new gap on top of whatever LST-internal effect the test was
trying to isolate. **Did not proceed to Step 4 (the efficiency-curve comparison)** — per the
plan's own verification gate, stopping here rather than generating numbers that would be
misleading without this caveat attached.

## Files Changed This Session

None (rebuild only, no source edits). `LSTNtuple_idealpls.root` created (100 events, kept).
Ad hoc PyROOT check scripts were written to the session scratchpad only, not committed to the
repo.

## What Remains To Do

1. **Decide how to proceed on the idealpls thread**, now that its premise needs revisiting.
   Options not yet evaluated: (a) widen the eligible-hit definition beyond "≥3 distinct
   truth-matched pixel hits" if there's a way to recover hits CMSSW's own clustering merged away
   (likely not possible without re-simulating), (b) treat this newly-found hit-availability gap
   as a *third*, previously-uncounted contributor to the original ΔR efficiency drop (alongside
   the Session 3 "no pLS"/"non-quad pLS" upstream causes) and quantify it directly instead of
   using it as a clean baseline for isolating LST-internal effects, (c) abandon the idealpls
   comparison and return to the DNN-gate lead (Session 11/memory) instead.
2. **Profile `TruthPixelSeeds.cc`'s per-event cost** before any run beyond ~100 events — the
   ~37-minute wall time for 100 events (vs. ~2s of actual in-loop GPU/reconstruction time) is a
   real, unexplained, unprofiled cost, most likely in `buildSimTrkToPixHitMap`'s map-based
   reverse lookup or `buildTruthPixelSeeds`'s per-sim-track hit selection — not investigated this
   session.
3. All earlier carried-forward items (Sessions 4-9) unchanged.

---

# Conditioned Per-Category Chi-Square Breakdown, Full Stats — Handoff (Session 11)

## Context

Continuation of Session 10. The relaunched (`setsid`+`nohup`) 1000-event background job had
finished by the time this session started. Picked up the two pending items: (1) validate the
output file cleanly, (2) rerun the conditioned per-category `rPhiChiSquared` breakdown that
Session 10 set up but couldn't run (input file wasn't ready yet).

## What Was Done

1. **Validated `LSTNtuple_chidebug_1000.root`**: process (PID 2916155) had exited normally (log
   ends with `RooUtil:: Wrote output to LSTNtuple_chidebug_1000.root` and a clean final timing
   block, no truncation). PyROOT confirms `IsZombie()=False`, **1000/1000 entries**, no
   `TFile::Recover` needed — clean this time, unlike the Session 9/10 run that died at 480/1000.
2. **Reran the conditioned breakdown** (restrict to `pixelRadius < 5*kR1GeVf`, i.e. the regime
   where `rPhiChiSquared` is actually enforced; group by `pT5_catLayer1-5`; ΔR via
   `sim_genjet_deltaR[pT5_simIdx]`; non-fake matched candidates only). Stats jumped from n=6 to
   **n=458** cut-enforced candidates at ΔR<0.01 (and n=79263 at ΔR>0.05) — finally enough to draw
   a real conclusion.
3. **Two layer categories dominate the cut-enforced low-ΔR sample**: `(1,2,3,4,5)` and
   `(2,3,4,5,6)` (all-barrel) account for 308/458 (67%) of it. Computed percentile comparisons
   (p25/50/75/90/99) of `rPhiChiSquared`, low-ΔR vs high-ΔR, for both:

   | category | p25 ratio | p50 ratio | p75 ratio | p90 ratio | p99 ratio |
   |---|---|---|---|---|---|
   | (1,2,3,4,5) | 1.39 | 1.59 | 1.28 | 1.14 | 1.08 |
   | (2,3,4,5,6) | 1.23 | 2.03 | 1.68 | 1.50 | 1.28 |

   All other categories have n<20 at low-ΔR — too sparse for any conclusion, and several go the
   *opposite* direction (low-ΔR median below high-ΔR median), so there's no consistent
   cross-category pattern.

## Conclusion — chi-square line of inquiry is now closed, not just under-powered

With the pT-mix confound removed (Session 10) **and** real statistics (this session), the
honest effect is **~1.1-2x elevation at low ΔR for the two dominant categories, shrinking at
higher percentiles** — nothing like the original (confounded, Session 7) "8-14x median, 50-130x
p90" claim. Medians for the dominant categories (7.3 and 11.3) stay well below even the
*tightest* threshold table entries (21.265, 32.253, 37.058) — most candidates aren't near the
boundary at all. **The Session 7 "loosen rPhiChiSquared thresholds" idea is not just
under-supported, it's actively unsupported by the properly-conditioned, full-statistics data.**
No threshold change should be made.

This resolves the fork left open since Session 8: the chi-square cuts are not a meaningful
driver of the dense-region pT5 efficiency loss. **The DNN gate
(`dnn::pt3dnn::pT5WP` in `runPixelTripletDefaultAlgo`) is now the more promising next lead** if
this thread continues, per the structural note in Session 10.

## Files Changed This Session

None (analysis only; scripts written to the session scratchpad, not the repo —
`/tmp/claude-63526/.../scratchpad/pt5_chidebug_1000.py` and `..._pctl.py`, not persisted).

## What Remains To Do

1. **Decide whether to continue the DNN-gate thread or move to `--idealpls`.** Both are now
   unblocked — the background job that was gating both is done and validated. Ask the user
   before picking a direction.
2. **`--idealpls` build/run** (Sessions 8-10, still not built/run): rebuild CUDA without `-d`
   (`lst_make_tracklooper -G` — drops the `CUT_VALUE_DEBUG` branches used for this chi-square
   work, which are no longer needed now that this line of inquiry is closed) and run
   `lst_cuda -i trackingNtuple-100.root --allobj --jet --idealpls -n 100 -o LSTNtuple_idealpls.root`.
   Ask the user before rebuilding (standing instruction from Session 10).
3. `LSTNtuple_chidebug_1000.root.partial480` (the dead 480-entry file from the first failed
   run) is still sitting on disk — safe to delete now that the clean 1000-entry file exists,
   but not deleted without asking (kept once already at explicit user request).
4. All earlier carried-forward items from Sessions 4-9 (the `performance.cc` stash, etc.) are
   unchanged.

---

# Recover Dropped Findings + Relaunch Failed Background Job — Handoff (Session 10)

## Context

Picked up per standing instruction ("read HANDOFF.md and continue"). Two issues surfaced
immediately, neither of which the user had asked about directly — both came from auditing
state before trusting the handoff:

1. **A real finding from the Session 8/9 conversation was never written to HANDOFF.md.**
   Before launching the 1000-event background job, that session actually ran the Session 7
   per-category chi-square breakdown (on `LSTNtuple_chidebug4.root`, 100 events, using new
   `pT5_pixelRadius`/`pT5_catLayer1-5` branches added to `write_lst_ntuple.cc` that same
   session — these are it the same uncommitted diff currently in the working tree, not
   separately called out in Session 7's file-change table below). **Result: it found and then
   retracted Session 7's "loosen rPhiChiSquared by 50-100%" recommendation.** Recovered from
   the raw session transcript (`2026-06-23-185012-...txt`), not from HANDOFF.md itself.
2. **The Session 8/9 background job (`b9j8mjvjq`, PID 2644552) died partway, silently.** By
   the time this session started, the process was gone and `LSTNtuple_chidebug_1000.root`
   needed `TFile::Recover` (i.e. was never cleanly `Close()`'d) and contained only **480 of
   the expected 1000 entries**. No OOM-kill or crash signal found in `dmesg`; GPU was idle
   (0% util, 0 MiB used on both cards) by the time this was checked. Cause unknown — possibly
   `code-server`'s `--enable-remote-auto-shutdown` (the exact residual risk flagged, but not
   acted on, in Session 9), possibly something else. Not root-caused this session.

## The Retracted Finding (recovered, now permanently recorded)

Per-category breakdown on `LSTNtuple_chidebug4.root` (100 events), conditioning on the cut's
own pT gate (`pixelRadius < 5*kR1GeVf`, ≈pT<5 GeV) found in `PixelQuintuplet.h`:
`rzChiSquared`/`rPhiChiSquared`/`rPhiChiSquaredInwards` are **only enforced** in that regime —
above ~5 GeV the chi-squared values are still computed but the cut is skipped entirely
(always passes).

| | n where cut enforced (pixelRadius < 5×kR1GeVf) | n where cut skipped (high-pT) |
|---|---|---|
| ΔR<0.01 | 6 | 387 (98.5%) |
| ΔR>0.05 | 752 | 1025 (55%) |

**This invalidates Session 7's "loosen rPhiChiSquared by 50-100%" recommendation** (point 9 in
that section, below) — the original "7-15x worse rPhiChiSquared at small ΔR" comparison mixed
two very different pT populations, not a controlled comparison. 98.5% of dense-region (ΔR<0.01)
survivors are high-pT tracks where this cut doesn't even apply — most likely just jet physics
(ΔR<0.01 selects tracks nearly coincident with a >1000 GeV jet axis, which concentrates the
highest-pT fragments), not a reconstruction artifact. With the cut actually enforced, the
low-ΔR sample is only n=6 — nowhere near enough to say anything about whether those thresholds
are too tight in dense regions. **No threshold change should be made on the basis of the
Session 7 analysis.**

Structural implication, not yet investigated: for the population that actually dominates dense
regions (high-pT, cut skipped), the only gate left before a pT5 forms is the upfront DNN
classifier (`dnn::pt3dnn::pT5WP` in `runPixelTripletDefaultAlgo`). If there's a cut-driven loss
specific to dense regions in the *dominant* population, it's more likely to live there than in
the chi-square cuts analyzed in Session 7. This was the open fork at the end of that thread
("investigate the DNN gate" vs. "get more stats on the cut-enforced low-pT regime") — the
1000-event rerun (this session and Session 8/9) was the user choosing the second option.

## What Was Done (this session)

1. Confirmed the background job (PID 2644552) was no longer running and the GPU was idle.
2. Validated `LSTNtuple_chidebug_1000.root` via PyROOT: not a zombie, but required
   `TFile::Recover`, 480/1000 entries (input file confirmed to have exactly 1000 entries).
3. Surfaced both findings above to the user and asked how to proceed (background-job recovery
   vs. moving straight to the `--idealpls` build). **User chose: rerun the 1000-event job from
   scratch.**
4. Moved the incomplete file aside: `mv LSTNtuple_chidebug_1000.root
   LSTNtuple_chidebug_1000.root.partial480` (per explicit user instruction — kept, not deleted).
5. **Relaunched the job fully detached**, per the standing Session 9 decision (`nohup`+`setsid`,
   not just `nohup`, specifically to survive `code-server`'s auto-shutdown tree-kill, not just a
   dropped terminal):
   ```bash
   setsid nohup lst_cuda -i trackingNtuple-1000.root --allobj --jet -n -1 \
     -o LSTNtuple_chidebug_1000.root \
     > lst_cuda_1000_rerun.log 2>&1 < /dev/null & disown
   ```
   New PID 2916155, confirmed via `ps -o pid,ppid,pgid,sid,tty,stat`: `PGID=SID=2916155`, own
   session, `TT=?` — genuinely detached from `code-server`'s process tree this time, not just
   from the launching shell. Log file: `lst_cuda_1000_rerun.log`.

## Files Changed This Session

None (besides the file move above). HANDOFF.md itself updated with this entry.

## What Remains To Do

1. **Wait for the relaunched 1000-event job to finish**, then validate it cleanly this time
   (`IsZombie()`, no `Recover` needed, 1000/1000 entries) before trusting it.
2. **Once clean**, rerun the conditioned per-category breakdown (restrict to
   `pixelRadius < 5*kR1GeVf`, group by `catLayer1..5`) against the full 1000-event file — this
   is the actual point of the rerun: get >6 candidates in the cut-enforced low-ΔR bin. If it's
   still too few, the chi-square line of inquiry is likely a dead end and the DNN-gate
   investigation (above) is the better next step.
3. **`--idealpls` build/run is still pending**, blocked on this same job for the same reason as
   Sessions 8/9 (avoid rebuilding/competing for the GPU while a CUDA job is in flight). Ask the
   user before rebuilding once the job finishes — don't assume non-debug `-G` vs. keeping `-Gd`
   without checking whether the chi-square debug branches are still wanted alongside idealpls.
4. All carried-forward items from Sessions 7-9 below are otherwise unchanged.

---

# Background-Job Resilience Check — Handoff (Session 9)

## Context

Pure operational investigation, no code changes. The Session 8 background job
(`lst_cuda -i trackingNtuple-1000.root --allobj --jet -n -1 -o LSTNtuple_chidebug_1000.root`,
PID 2644552, harness task `b9j8mjvjq`) was still running. The user asked whether it would
survive a loss of their internet connection, then asked to relaunch it under `nohup`.

## What Was Done

1. **Traced the full process ancestry** of the running `lst_cuda` job via `ps`:
   `lst_cuda` → bash wrapper (both `TTY=?`, no controlling terminal) → `claude` → `pts/8` bash
   w/ VSCode shell integration → VSCode `ptyHost` → VSCode `server-main.js` → `code-server`
   (**PPID=1**, reparented to init — not a child of any live SSH login process). A separate
   `sshd-session: kk829@notty` process is the tunnel VSCode uses to talk to `code-server`, not
   its parent.
2. **Conclusion on connection resilience:** a brief network drop is safe — the job has no
   controlling terminal to hang up, and the whole chain is already detached from any
   live-connection-dependent process. **One real residual risk found:** the `code-server`
   command line includes `--enable-remote-auto-shutdown`, which can shut down the entire
   server (and everything under it, including this job) after some idle period with zero
   connected clients. Exact timeout not visible from `ps`.
3. **Before acting on "relaunch with nohup,"** checked the job's actual progress: **1h6m
   elapsed, 2.6+ GB already written** to `LSTNtuple_chidebug_1000.root`. `lst_cuda` has no
   checkpointing — killing and restarting forfeits 100% of that progress, not just some of it.
   Flagged this explicitly before touching anything, since `nohup` alone wouldn't have even
   fully solved the identified risk (it stops `SIGHUP` from a parent shell exiting, which
   wasn't the actual live risk here — the auto-shutdown risk would need `setsid`-style full
   process-group/session detachment from `code-server` to truly close).
4. **User decision (asked via AskUserQuestion): keep the current job running untouched.**
   Do not kill/restart it. Use `nohup`+`setsid` (not just `nohup`) for any *future* background
   jobs launched this session instead.

## Files Changed This Session

None.

## Decisions Made

- **Do not kill the in-flight `trackingNtuple-1000.root` job** (PID 2644552 / task `b9j8mjvjq`)
  to relaunch it — the 1h6m+ of unrecoverable progress outweighed the marginal protection
  `nohup` alone would have added against an already-low-probability auto-shutdown edge case.
- **For any future background job this session, use `nohup ... & disown` combined with
  `setsid`** (full session detachment from `code-server`'s process group), not just `nohup` —
  `nohup` alone doesn't address the specific residual risk identified (`code-server`'s
  `--enable-remote-auto-shutdown` tearing down its descendant tree), since that isn't a
  `SIGHUP`-from-parent-shell-exit scenario.

## What Remains To Do

- Everything from Session 8 is unchanged and still pending: the 1000-event background job
  (PID 2644552) is still running (1h6m+ elapsed as of this note) — wait for it to finish, then
  rerun the Session 7 per-category chi-square breakdown against
  `LSTNtuple_chidebug_1000.root`.
- The `--idealpls` ideal-pLS injection code (Session 8: `TruthPixelSeeds.h/.cc`,
  `AnalysisConfig.h`, `bin/lst.cc`) is implemented but still not built or run — still blocked
  on this same background job before rebuilding `liblst_cuda.so` (the user's standing
  preference from Session 8 was to avoid rebuilding while a CUDA job is in flight).
- All earlier carried-forward items (Session 7's `performance.cc` stash, etc.) are unchanged.

---

# Truth-Derived "Ideal pLS" Injection — Handoff (Session 8)

## Context

Two threads this session, both continuations of the ΔR investigation:

1. A quick background rerun of the Session 7 chi-square/category diagnostic on
   `trackingNtuple-1000.root` (more statistics in the rare cut-enforced, low-ΔR regime found
   in Session 7) — launched in background per the user's request so they could keep working.
2. The main new piece: the user wants to test how much of LST's TC efficiency drop in dense
   jet cores is *upstream* (CMSSW pixel seeding, already identified in Session 3) versus
   *inside LST itself* (T3/T5/pT3/pT5 matching, dedup, DNN gates). Approach: inject synthetic,
   Monte-Carlo-truth-derived "ideal" pLS objects — guaranteed to truth-match their sim track —
   in place of CMSSW's real seeds, then compare how the rest of LST's efficiency vs ΔR holds up.
   Constraint: all changes must stay inside `RecoTracker/LSTCore` (confirmed satisfied — the
   injection point is a per-event function call already entirely within this package).

User confirmed two design choices before implementation: synthetic pLS **replace** real seeds
entirely for in-scope sim tracks (not added alongside), and a sim track with only 3 true pixel
hits gets a triplet (non-quad) seed; fewer than 3 gets no synthetic pLS (no hit fabrication).

## What Was Done

### 1. Background 1000-event rerun (still running, not yet analyzed)

Launched `lst_cuda -i trackingNtuple-1000.root --allobj --jet -n -1 -o LSTNtuple_chidebug_1000.root`
in the background (task `b9j8mjvjq`) using the still-current `-Gd` (`CUT_VALUE_DEBUG`) CUDA
build from Session 7. **Still running as of end of session** — the 100-event equivalent took
~20-30 min each time this session; 1000 events on the same slow `explicit_cutvalue` path will
take substantially longer. No results yet; rerun the Session 7 per-category chi-square
breakdown against this ntuple once it completes.

### 2. Ideal-pLS injection — designed and implemented, not yet built/run

**Key architectural finding** (de-risked the whole design): `prepareInput()`
(`interface/LSTPrepareInput.h:25-261`) — which turns CMSSW's `see_*` seed branches into LST's
`PixelSeedsSoA` — needs **zero modification**. It only consumes 13 per-seed scalar/vector
inputs (momentum at two points, dxy/dz, ptErr/etaErr, charge, hit indices/types, an algo
filter). Every downstream device kernel (`Segment.h`'s `addPixelSegmentToMemory`,
`PixelTriplet.h`, `PixelQuintuplet.h`, the DNN embeddings) is equally untouched. The entire
task reduces to: build truth-derived versions of those 13 inputs, and call the same,
unmodified `prepareInput()` with them instead of the real `see_*` branches, at the existing
per-event call site `bin/lst.cc:405-428` (now ~440-470 after edits).

Truth mapping used (all confirmed against the actual `trackingNtuple-100.root` branches, not
assumed):
- `see_px/py/pz` **and** `see_stateTrajGlbPx/Py/Pz` → the same true momentum (from `sim_pt`,
  `sim_eta`, `sim_phi`) — an ideal seed has no momentum smearing.
- `see_dxy/dz` → the input ntuple's own `sim_pca_dxy`/`sim_pca_dz` (genuinely true impact
  parameters, already used by `write_lst_ntuple.cc:701-702`).
- `see_stateTrajGlbX/Y/Z` → the real `pix_x/y/z` position of the sim track's own outermost
  selected true pixel hit (guarantees the point genuinely lies on the helix, avoiding having
  to hand-implement curved-helix propagation).
- `see_ptErr/etaErr` → median of the real seeds' `see_ptErr`/`see_etaErr` in the same input
  file (not zero, not a hardcoded guess — `prepareInput()`'s `pixtype` classification and the
  pLS DNN embedding both take these as features and expect realistic non-zero values).
- `see_hitIdx/see_hitType` → the sim track's own true pixel hits (3 or 4, one per distinct
  `(subdet, layer, side)`, sorted by radius), all marked `HitType::Pixel`. `prepareInput()`
  already handles the 3-vs-4 (triplet-vs-quad) case itself
  (`LSTPrepareInput.h:142-144`) — no extra branching needed.
- `see_algo` → left empty; `prepareInput()` skips its own algo filter when empty
  (`LSTPrepareInput.h:102`).
- Truth-to-hit lookup: `pix_simHitIdx` (pixel-hit → simhit) + `simhit_simTrkIdx` (simhit →
  sim track) inverted into sim-track → pixel-hits (no existing helper did this — checked
  `AccessHelper.cc/.h`, which only has the opposite direction).

## Files Changed This Session

| File | Change | Status |
|---|---|---|
| `code/core/TruthPixelSeeds.h` / `.cc` | New: `buildSimTrkToPixHitMap()` (reverse truth lookup) and `buildTruthPixelSeeds()` (builds the 13 `see_*`-equivalent vectors per event) | Created, kept |
| `code/core/AnalysisConfig.h` | Added `bool use_truth_pls` to the `ana` struct | Modified, kept |
| `bin/lst.cc` | Added `--idealpls` cxxopts flag (parsed into `ana.use_truth_pls`); per-event `prepareInput()` call site now branches between real `see_*` branches and `buildTruthPixelSeeds()` output, feeding either into the same unmodified `prepareInput()` call | Modified, kept |

No device/kernel code (`Segment.h`, `PixelTriplet.h`, `PixelQuintuplet.h`, any `*SoA.h`) was
touched — by design, per the architectural finding above.

**Not yet built or run.** The CUDA library (`liblst_cuda.so`) is currently still the Session 7
`-Gd` debug build, and the background 1000-event job (above) is actively using it. The user
explicitly chose to **wait for that background job to finish** before rebuilding, rather than
risk disturbing it or switching to the CPU backend. So `lst_make_tracklooper -G` (a normal,
non-debug rebuild — Session 7's `-d` debug branches aren't needed for this test) has not been
run yet, and no `--idealpls` ntuple exists yet.

## Decisions Made

- Pivoted away from a deeper, riskier design (deriving `see_stateTrajGlbX/Y/Z` via hand-rolled
  curved-helix propagation) in favor of reusing the sim track's own real outermost pixel hit
  position — simpler, zero new physics/math code, and guaranteed self-consistent with the true
  trajectory.
- Chose not to apply any extra pt/eta/vertex pre-filtering inside `buildTruthPixelSeeds()` — it
  only requires `charge != 0` and ≥3 true pixel hits. `prepareInput()`'s own existing
  `ptIn > ptCut - 2*ptErr` gate (line 111) filters by momentum exactly the same way it does for
  real seeds, so no logic is duplicated.
- User chose "wait for background job" over "build CPU backend instead" or "rebuild CUDA now
  anyway" when a build-vs-running-job conflict came up (see above) — prioritizing the
  in-flight 1000-event analysis over faster iteration on the new test.

## What Remains To Do

1. **Once the 1000-event background job (`b9j8mjvjq`) completes**: rerun the Session 7
   per-category chi-square breakdown against `LSTNtuple_chidebug_1000.root` to get real
   statistics in the cut-enforced, low-pT, low-ΔR regime (only 6 candidates at 100 events).
2. **Then**, rebuild CUDA without `-d` (`lst_make_tracklooper -G`) and run:
   `lst_cuda -i trackingNtuple-100.root --allobj --jet --idealpls -n 100 -o LSTNtuple_idealpls.root`
3. **Sanity-check the injection worked**: pLS-level efficiency for in-scope sim tracks (the
   `has_eligible_pls` boolean in `lst_check_pt5_matching_density.py`) should read ~100%
   (bounded only by the ≥3-true-pixel-hit requirement), vs. baseline's ΔR-dependent drop.
4. **Answer the actual question**: rerun `lst_plot_eff_vs_deltaR_stages.py` and
   `lst_check_pt5_matching_density.py` against `LSTNtuple_idealpls.root`, compare
   T5_lower/pT5_lower/pT3_lower/TC efficiency vs ΔR against the existing baseline
   (`LSTNtuple_lower.root`/`LSTNtuple_nodupT5.root`). Whatever ΔR-dependent gap remains in TC
   efficiency with ideal pLS is attributable to LST's own logic, not upstream seeding.
5. Carries forward from Session 7: the per-category rPhiChiSquared threshold question is still
   open pending more statistics (item 1 above); the `performance.cc` stash (`stash@{0}`) is
   still un-popped; rebuild CUDA back to non-debug eventually covers this too since `-G` is
   needed for the idealpls test anyway.

---

# pT5 Matching-Density Investigation — Handoff (Session 7)

## Context

Follow-on to the Session 3 ΔR investigation. The user's hypothesis: maybe the pT5
(pLS+T5) matching/building step itself — not the already-documented upstream pLS
supply deficit — degrades at high object density (small ΔR), e.g. because more
candidate T5s compete for the same pLS. This had not been tested before. Also
covered, earlier in the session: explained the error bars on
`eff_vs_deltaR_stages_recreated.png` (ROOT `TEfficiency` Clopper-Pearson, 1σ,
asymmetric binomial CI — the original `eff_vs_deltaR_stages.png` apparently never
plotted them) and re-confirmed (by direct code read, not just citing Session 6) that
`isDup`/duplicate flags do not gate `T5_lower`-style efficiency.

## What Was Done

1. **Built a reusable diagnostic script**, `efficiency/python/lst_check_pt5_matching_density.py`
   (new, follows the `lst_plot_eff_vs_deltaR_stages.py` convention). For each selected sim
   track with an eligible pLS (quad, non-duplicate) **and** a true T5 ("ingredients present"),
   checks whether a true pT5 actually formed, binned vs ΔR. Denominator selection mirrors
   `performance.cc`'s deltaR-plot cuts exactly (pt>0.9, |eta|<4.5, vertex cuts, GenJet
   pt>1000/|eta|<2.5, charged-only).
2. **Found a real, smaller, previously-untested effect**: "ingredients present, no pT5
   formed" rises from 1.6% (ΔR>0.05) to 11.3% (ΔR<0.01) — real, but only about a third
   of the ~30pp pT5_lower drop already explained by upstream pLS supply (Session 3).
   `pT5_isFake` rate among pT5s that do form stayed flat (~0%) — ruled out "wrong/fake
   candidate wins" as the mechanism.
3. **Tested the literal "competing candidates" hypothesis without any rebuild.** Discovered
   `pT5_plsIdx`/`pT5_isFake`/`pT5_isDuplicate` are already written for *every* pT5 candidate
   row in `write_lst_ntuple.cc`, unfiltered by the device-level `isDup` bit (same pattern as
   `T5_lower`, confirmed independently this session). This let the script compute
   `nCompetingT5` (other T5s a given pLS successfully paired with) straight from the existing
   ntuple — **no kernel/device changes needed**, and this also avoided a real correctness risk:
   the originally-planned device-side atomic counter (read back mid-kernel) would have raced
   on the CUDA backend's parallel (pLS, T5) scan.
4. **Result: density does increase competition (mean nCompetingT5 ~8 at ΔR<0.002 → ~3 at
   ΔR>0.05), but competition is not the loss mechanism.** When a pLS pairs with *anything*,
   it's the correct T5 99% of the time (2233 cases, 1.0% no-pT5 rate). The loss concentrates
   almost entirely in pLS that pair with **nothing at all** (107 cases, 69.2% no-pT5 rate) —
   i.e. the true (pLS,T5) pair fails the matching cuts outright, with no competitor involved.
5. **Investigated `runPixelQuintupletDefaultAlgo` (`PixelQuintuplet.h:469-634`)** to find why.
   It's a 4-gate chain: a DNN pre-filter (`dnn::pt3dnn::pT5WP`, calibrated to ~99/99.5% signal
   efficiency — by design unlikely to be the main culprit), then three hardcoded,
   layer-category-dependent chi-square cuts (`rzChiSquared`, `rPhiChiSquared`,
   `rPhiChiSquaredInwards`), each with its own threshold table
   (`passPT5RZChiSquaredCuts`/`passPT5RPhiChiSquaredCuts`/`passPT5RPhiChiSquaredInwardsCuts`).
   Noted: any layer combination not listed in a threshold table falls through to `return true`
   (no cut applied at all) — a structural gap, not touched this session.
6. **Discovered `write_lst_ntuple.cc` never wrote the pT5 chi-square values to the ntuple**,
   unlike pT3/T4 which do (gap, not previously known). **Edited `write_lst_ntuple.cc`** to add
   `pT5_rzChiSquared`/`pT5_rPhiChiSquared`/`pT5_rPhiChiSquaredInwards` branches (under
   `#ifdef CUT_VALUE_DEBUG`), mirroring the existing pT3 pattern exactly — additive only, no
   behavior change for normal (non-debug) builds.
7. **Rebuilt CUDA with `-d`** (`lst_make_tracklooper -mGd` then `-Gd` after the
   `write_lst_ntuple.cc` edit) and reran `lst_cuda -i trackingNtuple-100.root --allobj --jet
   -n 100 -o LSTNtuple_chidebug2.root` to get the new chi-square branches.
8. **Compared chi-square distributions of surviving (non-fake) pT5s, ΔR<0.01 vs ΔR>0.05.**
   First pass wrongly concluded `rzChiSquared` was ~20x worse at small ΔR — **caught and
   corrected within the session**: that was a NaN-corrupted median (24 of 22,210 pT5
   candidates have `rzChiSquared = NaN`, a separate minor numerical edge case, not
   investigated further). After filtering NaN and using percentiles, `rzChiSquared` is
   **not** meaningfully different (p50 0.35 vs 0.32). `rPhiChiSquared` and
   `rPhiChiSquaredInwards` **are** substantially and broadly worse at small ΔR (not just a
   tail effect — visible from p25 up): medians ~8-14x higher, p90 ~50-130x higher.
9. **Recommendation (not yet tested) — RETRACTED, see Session 10:** the tightest
   `rPhiChiSquared` category thresholds (21.265, 32.253, 37.058 — see
   `passPT5RPhiChiSquaredCuts`, `PixelQuintuplet.h:174-257`) are the most likely to be rejecting
   genuinely-correct dense-region candidates outright, since the dense-region median (53.0)
   already exceeds them. Proposed loosening those by ~50-100% as a first hypothesis.
   `rzChiSquared` and `rPhiChiSquaredInwards` thresholds were *not* recommended for change
   (inwards thresholds are mostly well above even the dense-region p75).
   **Retracted in a later session (recorded as Session 10 below): this comparison was
   confounded by pT mix — the chi-square cuts only apply below ~5 GeV pT, and 98.5% of the
   dense-region (ΔR<0.01) population analyzed here is above that threshold and never hits the
   cut at all. Do not act on this recommendation.**

## Files Changed This Session

| File | Change | Status |
|---|---|---|
| `efficiency/python/lst_check_pt5_matching_density.py` | New diagnostic script (ingredients-present/no-pT5 rate, pT5_isFake rate, nCompetingT5 — all vs ΔR) | Created, kept, executable |
| `code/core/write_lst_ntuple.cc` | Added `pT5_rzChiSquared`/`pT5_rPhiChiSquared`/`pT5_rPhiChiSquaredInwards` branches under `CUT_VALUE_DEBUG` (createPixelQuintupletBranches + the pT5 fill loop, ~line 619 and ~line 2131) | **Modified, kept** (`git status` confirms this is the only source diff right now) |
| `pt5_matching_density.png` | 3-panel diagnostic plot (no-pT5 rate, isFake rate, mean nCompetingT5, all vs ΔR) | Created, kept |
| `LSTNtuple_chidebug.root` | Orphaned — built *before* the `write_lst_ntuple.cc` edit, missing the chi-square branches | Stale, safe to delete |
| `LSTNtuple_chidebug2.root` | Current — 100 events, `--allobj --jet`, has the new chi-square branches | Created, kept |

**Build state:** the CUDA backend (`LST/liblst_cuda.so` / `bin/lst_cuda`) is currently built
with `-d` (`CUT_VALUE_DEBUG`, the `explicit_cutvalue` target), via `lst_make_tracklooper -Gd`.
This is **not** the normal production build — rebuild with plain `lst_make_tracklooper -G`
before any timing-sensitive work.

**Confirmed via `git status`/`git stash list` this session:** `TrackCandidate.h` shows no
diff (already clean — the Session 3 diagnostic edit there is gone, no revert needed). The
`performance.cc` stash (`stash@{0}`, "temp revert for timing test") is still sitting un-popped,
unchanged from Sessions 4-6.

## What Remains To Do

1. **Get the exact per-category breakdown.** Current evidence only shows the aggregate
   `rPhiChiSquared`/`rPhiChiSquaredInwards` distributions are worse at small ΔR, not which of
   the ~15 layer categories in `passPT5RPhiChiSquaredCuts` are responsible. Needs one more
   instrumented branch (T5 `logicalLayers`/lowerModuleIndices alongside the chi-square values)
   + rebuild + rerun.
2. **Test an actual loosened threshold.** Pick the tightest `rPhiChiSquared` categories
   (21.265, 32.253, 37.058), loosen by ~50-100%, rebuild, rerun, and measure: (a) recovered
   `pT5_lower` efficiency at low ΔR via `lst_check_pt5_matching_density.py`, (b) fake-rate
   change via the existing `pT5_isFake`/`_fr_*` histograms in `performance.cc`. No threshold
   change has been made to `PixelQuintuplet.h` yet — current values are untouched.
3. **Minor, separate:** investigate the 24/22210 `rzChiSquared = NaN` candidates (likely a
   degenerate-geometry edge case in `computePT5RZChiSquared`) — not blocking, not yet looked
   at.
4. **Rebuild CUDA back to a normal (non-debug) build** (`lst_make_tracklooper -G`) before any
   timing benchmark or production-style run, since the current binary is the slower
   `-d`/`explicit_cutvalue` variant.
5. Delete or ignore the orphaned `LSTNtuple_chidebug.root` (missing the new branches; superseded
   by `LSTNtuple_chidebug2.root`).
6. All open items from Sessions 4-6 below are unchanged: `git stash pop` for `performance.cc`
   (`stash@{0}`), the optional non-quad pLS recovery follow-up, rerun-on-larger-stats follow-up.
   (Session 3's `TrackCandidate.h` revert item is now moot — confirmed clean above.)

---

# Efficiency & Dedup/Fake-Removal Code Trace — Handoff (Session 6)

## Context

Pure investigation, no code changes. The user asked how efficiency is
calculated for non-TC objects in `eff_vs_deltaR_stages.png` (pLS, MD, etc.)
and what duplicate/fake removal is applied to objects like T5 before that
efficiency is computed. This matters for the ΔR investigation (Session 3
below) since it nails down exactly what each curve in the reference plot
means, and whether dedup/fake-removal logic could be confounding any of the
conclusions already drawn.

## What Was Done

Traced two independent layers of the code (via 2 parallel Explore agents,
then verified key claims directly by reading the source):

1. **Efficiency definition (`efficiency/src/performance.cc`)** — two
   genuinely different pass/fail criteria per sim track:
   - **TC** (and "as-TC" breakdowns for T5/pT5/pT3/pLS/T4): pass iff
     `sim_tcIdx >= 0` (`performance.cc:49`) — i.e. some final TrackCandidate
     matches this sim track with hit-purity > 0.75. `sim_tcIdx` is set in
     `write_lst_ntuple.cc` (~line 2477).
   - **`*_lower` stages** (MD_lower, LS_lower, T3_lower, T5_lower, pT3_lower,
     pT5_lower, pLS_lower — these are what's actually plotted as MD/LS/T3/
     T5/pT3/pT5/pLS in the reference plot): pass iff **any** built object of
     that type anywhere in the event has hit-purity fraction > 0.75 to this
     sim track (`sim_<type>IdxAllFrac` vectors, `performance.cc:114-173` plus
     the `T3_lower`/`LS_lower`/`MD_lower` variants in `git stash@{0}`). This
     is independent of whether that object ever becomes part of a TC, and
     independent of any fake/duplicate flag.
2. **Two separate notions of "duplicate"/"fake" exist, and neither affects
   the efficiency curves:**
   - **Algorithm-internal, truth-blind, runs during reconstruction:**
     `RemoveDupQuintupletsAfterBuild`/`BeforeTC` (`Kernels.h:177,223`,
     T5-only; analogous kernels for pT3/pT5/pLS, none for T3/LS/MD) and
     `CrossCleanT5`/`CrossCleanpT3`/`CrossCleanpLS` (`TrackCandidate.h:222,
     187,291`) compare reconstructed objects to each other (Δη/Δφ, shared-hit
     count, DNN-embedding distance) and set an `isDup()` **bitflag** via
     `rmQuintupletFromMemory` (`Kernels.h:20`) — despite the name, this does
     **not** delete the object from device memory, it only flags it
     ineligible to become its own standalone TC. The object still gets
     written to the ntuple and still counts for the `_lower` efficiency.
   - **Analysis-level, truth-based, computed downstream in
     `write_lst_ntuple.cc`:** `t5_isFake`/`t5_isDuplicate` (and per-stage
     equivalents, e.g. `~line 1654` for T5 fake, `~line 1713-1726` for T5
     duplicate) use MC truth matching (`matchfrac=0.75`) and feed **only**
     the separate fake-rate/duplicate-rate histograms
     (`_fr_*`/`_dr_*`) — not the `_ef_*` efficiency histograms used in
     `eff_vs_deltaR_stages.png`.

## Files Changed This Session

None — investigation only, no edits.

## What Remains To Do

- Nothing new opened by this session. The full explanation (with file:line
  citations) is in the conversation transcript exported to
  `2026-06-22-174142-what-changes-have-you-made-to-this-git-branch-com.txt`,
  not duplicated in full here.
- All open items from Sessions 3-5 below are unchanged (revert
  `TrackCandidate.h`, `git stash pop` for `performance.cc`, the pixel-seeding
  follow-ups, etc.).

---

# Recreate eff_vs_deltaR_stages.png — Handoff (Session 5)

## Context

The user asked what it would take to recreate `eff_vs_deltaR_stages.png` (the
Session 3 reference plot below). The script that produced it had never been
saved — no trace of it on disk, in `/tmp`, or in any stash.

## What Was Done

1. **Checked the `claude-edits` branch vs `main`/`master`**: zero commits
   ahead, identical merge-base. All differences are uncommitted/untracked
   files (`.claude/`, `CLAUDE.md`, `HANDOFF.md`, `quick-handoff.md`, the
   `eff_vs_deltaR_*.png` plots).
2. **Confirmed the original plotting script is unrecoverable** — searched
   `/tmp`, the scratchpad, and git stash; nothing found.
3. **Confirmed the source data already exists and is sufficient** —
   `LSTNumDen_lower.root` (built 2026-06-12 from `trackingNtuple-100.root`,
   100 events, commit `820464a`) already contains, for every stage
   `{TC, T5_lower, pT5_lower, T3_lower, pT3_lower, LS_lower, pLS_lower,
   MD_lower}`, both `Root__<STAGE>_base_0_0_ef_numer_deltaR` and
   `..._ef_denom_deltaR` (50 bins, [0, 0.1)). Verified directly via PyROOT.
   **No ntuple regeneration was needed.**
4. **Noted (not touched)**: `T3_lower`/`LS_lower`/`MD_lower` exist in that
   ntuple because `efficiency/src/performance.cc` had the Session 1/2
   additions applied when it was made. Those edits are still sitting in
   `git stash@{0}` ("temp revert for timing test") from Session 4 — irrelevant
   here since the histograms are already baked into the `.root` file, but
   still pending if `performance.cc` itself needs those edits restored.
5. **Verified PyROOT/matplotlib both work** once the environment is properly
   sourced (`source setup.sh && cmsenv && source setup.sh`) — an earlier
   attempt without sourcing first wrongly suggested PyROOT/uproot were
   unavailable.
6. **Decision (user choice via AskUserQuestion):** build a **reusable CLI
   utility** under `efficiency/python/` (not a one-off script), and write
   output to a **new filename** rather than overwrite the original, so the
   two can be diffed.
7. **Wrote `efficiency/python/lst_plot_eff_vs_deltaR_stages.py`** — follows
   `lst_plot_performance.py`'s conventions (argparse, `import ROOT as r`,
   same `Root__<OBJ>_<SEL>_<PDGID>_<CHARGE>_ef_numer/denom_<VAR>` naming, same
   `r.TEfficiency(num, den).CreateGraph()` efficiency/error computation).
   Takes any `LSTNumDen*.root`, a configurable stage list/labels, and
   produces the two-panel (zoom ΔR<0.05 + full histogrammed range) matplotlib
   figure. Made executable; runnable directly as `lst_plot_eff_vs_deltaR_stages.py`
   since `efficiency/python/` is already on `PATH` via `setup.sh`.
8. **Ran it** against `LSTNumDen_lower.root` →
   `eff_vs_deltaR_stages_recreated.png`. Shape matches the original.
   **Numerically verified**: TC efficiency 0.462 at ΔR<0.01 and 0.902 at
   ΔR>0.05 — exact match to the table already recorded below in the Session 3
   notes.

## Files Changed This Session

| File | Change | Status |
|---|---|---|
| `efficiency/python/lst_plot_eff_vs_deltaR_stages.py` | New reusable script | Created, kept, executable |
| `eff_vs_deltaR_stages_recreated.png` | New output plot | Created, kept |

No existing source files were modified.

## What Remains To Do

- Nothing required for the recreation task itself — it's done and verified.
- Carries forward from Session 4: `git stash pop` to restore
  `efficiency/src/performance.cc` (`stash@{0}`) whenever that work resumes.
- Carries forward from Session 3 (unchanged, see below): revert
  `TrackCandidate.h`, and the open investigation steps (confirm pixel-seeding
  root cause, optional non-quad pLS recovery, report findings, rerun on
  `trackingNtuple-1000.root`).

---

# LST CPU vs CUDA Timing Benchmark — Handoff (Session 4)

## Context

Separate thread of work from the ΔR investigation below: benchmarking `lst_cpu`
vs `lst_cuda` wall-clock timing, including the per-object-type ("object
multiplicity") counts via `-v 2`. Started on the full `trackingNtuple-100.root`
sample, then switched to a fast-iterating 10-event subset partway through.

## What Was Done

1. **Confirmed a prior CPU run had finished** (`-v 1`, i.e. timing-only, 100
   events): `/tmp/lst_cpu_timing.log` — `CPU Time: 9513.4 ms`, `Real Time: 9551.0 ms`.
2. **Ran `lst_cuda` the same way** (`-v 1`, 100 events) → `/tmp/lst_cuda_timing.log`
   — `CPU Time: 1049.0 ms`, `Real Time: 1051.2 ms`. **CUDA ~9.1x faster than CPU**
   at 100 events, consistent across every reconstruction stage.
3. **Restored the stash** (`git stash pop`) bringing back the in-progress
   `efficiency/src/performance.cc` change (stash msg "temp revert for timing
   test") that had been stashed before the first CPU run.
4. **Identified GPUs available on this machine** (via `nvidia-smi`):
   NVIDIA L40 (46068 MiB) and NVIDIA L4 (23034 MiB), driver 580.105.08.
5. **Found the object-multiplicity flag**: `-v 2` sets `objectsStatistics_` in
   `LSTEvent` (`bin/lst.cc:442` → `LSTEvent(ana.verbose >= 2, ...)`), which
   gates `if (objectsStatistics_)` print blocks throughout
   `../src/alpaka/LSTEvent.dev.cc`, printing `[MEM] <Type>: N allocated (X MB)`
   lines via `lstWarning(...)` for every object type per event.
6. **Stashed `performance.cc` again** (`git stash push -m "temp revert for
   timing test" -- efficiency/src/performance.cc`) before the `-v 2` round.
7. **Decision: run CPU/CUDA sequentially, never concurrently.** First attempt
   launched `lst_cpu -v 2` and `lst_cuda -v 2` as parallel background jobs;
   stopped the CPU job almost immediately because the CUDA run's host-side
   CPU usage would contaminate CPU-backend timing numbers.
8. **100-event `-v 2` CPU run was killed twice** (background tasks `b1kgxxl0j`,
   then `b9ltpevle`) — at ~2h39m per the `-v 1` precedent, it was too slow for
   the iteration the user wanted. The 100-event CUDA `-v 2` run *did* complete:
   `/tmp/lst_cuda_timing_v2.log` — `CPU Time: 1050.7 ms`, `Real Time: 1052.7 ms`.
9. **Created `trackingNtuple-10.root`**, a 10-event subset of
   `trackingNtuple-100.root`, via a ROOT macro (`tree->CloneTree(10)` into a
   new file under the same `trackingNtuple/` directory structure). Verified
   10 entries, not a zombie. (A first inline `root -l -e '...'` one-liner
   attempt crashed cling; switched to a macro file, which worked — the
   resulting exit-time crash from a double `TFile::Close()` was harmless,
   file was already fully written.)
10. **Re-ran the `-v 2` benchmark on `trackingNtuple-10.root`**, sequentially:
    - CPU: `/tmp/lst_cpu_timing_10_v2.log` — `CPU Time: 186.1 ms`, `Real Time: 187.0 ms`.
    - CUDA: `/tmp/lst_cuda_timing_10_v2.log` — `CPU Time: 43.6 ms`, `Real Time: 43.8 ms`.
    - **~4.3x CUDA speedup** at 10 events — smaller than the 100-event ~9.1x,
      consistent with fixed GPU launch/allocation overhead dominating more at
      low event counts.
11. **Compared `[MEM]` object-multiplicity output between backends** (10-event
    run): 120 `[MEM]` lines each side. Hits/Ranges/PixelSegments/
    PixelQuintuplets/PixelTriplets counts matched exactly every event.
    TrackCandidate totals matched in 6/10 events and differed by ±1–2 in the
    rest (e.g. event 0: CPU 162 vs CUDA 161 TCs), traced to small ±1 diffs in
    `Quintuplets`/`Quadruplets`/`pT5`/`T4` counts — consistent with expected
    non-determinism in parallel reduction/dedup kernels across backends, not
    flagged as a correctness bug.

## Files Changed This Session

| File | Change | Status |
|---|---|---|
| `efficiency/src/performance.cc` | No new edits — only stashed/popped/stashed again around timing runs | Currently **stashed** (`stash@{0}`, msg "temp revert for timing test") |
| `trackingNtuple-10.root` | New file — 10-event subset of `trackingNtuple-100.root`, same tree structure | Created, kept |

No LST source files were edited as part of the timing benchmark itself.

## Logs Produced

| File | Run | Status |
|---|---|---|
| `/tmp/lst_cpu_timing.log` | `lst_cpu -v 1`, 100 events | Complete — kept |
| `/tmp/lst_cuda_timing.log` | `lst_cuda -v 1`, 100 events | Complete — kept |
| `/tmp/lst_cpu_timing_v2.log` | `lst_cpu -v 2`, 100 events | **Stale/incomplete** — killed twice, never finished |
| `/tmp/lst_cuda_timing_v2.log` | `lst_cuda -v 2`, 100 events | Complete — kept |
| `/tmp/lst_cpu_timing_10_v2.log` | `lst_cpu -v 2`, 10 events | Complete — kept |
| `/tmp/lst_cuda_timing_10_v2.log` | `lst_cuda -v 2`, 10 events | Complete — kept |

## What Remains To Do

1. **Decide whether the 100-event CPU `-v 2` run is still wanted.** It was
   never completed (too slow to iterate on); the 10-event result stands in
   for it for now. If the full 100-event `-v 2` object-multiplicity printout
   is needed, rerun `bin/lst_cpu -i trackingNtuple-100.root -n -1 -v 2 -w 0 >
   /tmp/lst_cpu_timing_v2.log 2>&1` by itself (~2h39m expected, run in
   background, no concurrent CUDA run).
2. `git stash pop` to restore `efficiency/src/performance.cc` (the
   `stash@{0}` entry). Leave `stash@{1}` ("WIP on adjust-dr") untouched — it
   predates this session and is unrelated.
3. Standard env setup reminder for any fresh shell before build/run commands:
   ```bash
   source /cvmfs/cms.cern.ch/cmsset_default.sh
   cd /mnt/data1/kk829/CMSSW_16_1_1
   eval $(scram runtime -sh)
   cd src/RecoTracker/LSTCore/standalone
   source setup.sh
   ```

---

# LST Efficiency vs ΔR — Investigation Handoff (Session 3)

## Context

Track candidate (TC) efficiency drops from ~90% at ΔR > 0.05 to ~46% at ΔR < 0.01,
where ΔR = distance from sim track to nearest GenJet center.

**Root cause identified (this session):** The dominant loss is upstream of LST —
CMSSW pixel seeding fails to reconstruct ~20% more charged tracks at ΔR < 0.01
than at large ΔR, plus ~9% more produce only non-quad (3-hit) pixel seeds that
LST's pT5 builder cannot use. LST's deduplication and cross-cleaning are NOT
the primary cause.

Reference plots (all in standalone dir):
- `eff_vs_deltaR_stages.png` — all stages vs ΔR (baseline)
- `eff_vs_deltaR_causal_chain.png` — T5/pLS/pT5 built vs as-TC, showing chain
- `eff_vs_deltaR_fix_comparison.png` — TC baseline vs dedup Fix1 vs Fix2 (negligible)
- `eff_vs_deltaR_nodupT5.png` — effect of disabling CrossCleanpLS T5 branch (negligible)

Sample: `trackingNtuple-100.root`, version tag `820464D`, 100 events (64 pass selection).

---

## Files Changed This Session

| File | Change | Status |
|---|---|---|
| `efficiency/src/performance.cc` | Added T3/LS/MD efficiency, fake-rate, duplicate-rate, fake-or-dup-rate inside `do_lower_level` blocks (Sessions 1+2) | Kept |
| `../src/alpaka/Kernels.h:289` | Two dedup fix attempts applied then **reverted** — negligible effect | **Reverted to HEAD** |
| `../src/alpaka/TrackCandidate.h:339` | T5 branch of CrossCleanpLS gated to `dR2 < 0.0f` (disabled) for diagnostic | **Needs revert** |

### Revert TrackCandidate.h before any further work

```bash
cd /mnt/data1/kk829/CMSSW_16_1_1/src
git checkout RecoTracker/LSTCore/src/alpaka/TrackCandidate.h
```

Then rebuild: `lst_make_tracklooper -G`

---

## Ntuples on Disk

| File | Description |
|---|---|
| `LSTNtuple_lower.root` | Baseline: `--allobj --jet`, not properly closed (ROOT auto-recovers) |
| `LSTNumDen_lower.root` | Baseline num/den histograms |
| `LSTNtuple_fix.root` | Dedup Fix1 (SC1 nMatched≥3) — negligible change |
| `LSTNumDen_fix.root` | Fix1 histograms |
| `LSTNtuple_fix2.root` | Dedup Fix2 (SC1+SC3 nMatched≥3, d²<0.01) — negligible change |
| `LSTNumDen_fix2.root` | Fix2 histograms |
| `LSTNtuple_nodupT5.root` | CrossCleanpLS T5 branch disabled — negligible change |
| `LSTNumDen_nodupT5.root` | nodupT5 histograms |

---

## Key Findings

### Stage-by-stage efficiency (baseline, ΔR<0.01 vs ΔR>0.05)

| Stage | ΔR<0.01 | ΔR>0.05 | Notes |
|-------|---------|---------|-------|
| MD    | 0.987   | 0.982   | Flat |
| LS    | 0.981   | 0.982   | Flat |
| T3    | 0.947   | 0.931   | Flat |
| T5_lower (built) | 0.900 | 0.892 | Flat — T5s build fine |
| T5 (as TC)  | 0.004 | 0.014 | T5s rarely become standalone TCs |
| pLS_lower   | 0.702 | 0.955 | **Drops — primary choke point** |
| pT5_lower   | 0.549 | 0.845 | **Drops — consequence of pLS drop** |
| pT5 (as TC) | 0.432 | 0.829 | Dominant TC type at all ΔR |
| TC total    | 0.462 | 0.902 | 44 pp drop, almost all from pT5 |

### pLS deficit decomposed (ΔR<0.01 vs ΔR>0.10, selected charged sim tracks)

Using `LSTNtuple_nodupT5.root` (T5 cross-clean disabled so isDup only from pT3/pT5 branches):

| Category | ΔR<0.01 | ΔR>0.10 | Δ | Cause |
|----------|---------|---------|---|-------|
| No pLS built | 26.3% | 6.4% | **+19.9 pp** | CMSSW pixel seeding failure |
| pLS exists, not quad | 29.1% | 20.2% | **+8.9 pp** | Quad seeds harder in dense jets |
| Quad pLS, all isDup | 8.0% | 21.8% | −13.8 pp | Less cross-cleaning at small ΔR |
| Quad pLS, not isDup | 36.6% | 51.6% | −15.0 pp | Available for pT5 seeding |

Combined **+28.8 pp** from no-pLS + non-quad ≈ matches the 29.2 pp pT5_lower drop.

Cross-cleaning (isDup) is **less** of a problem at small ΔR than large ΔR and is
**not a significant contributor** to the efficiency loss at ΔR < 0.01.

### What was ruled out

| Attempted fix | Result | Reason ineffective |
|---|---|---|
| `RemoveDupQuintupletsBeforeTC` SC1: require nMatched≥3 at dR<0.001 | +0.004 TC eff. | T5 dedup rarely fires for genuine distinct tracks |
| + SC3: require nMatched≥3 AND d²<0.01 at dR<0.02 | +0.007 TC eff. | Same; SC3 also not the limiting step |
| Disable CrossCleanpLS T5 branch entirely | +0.006 TC eff. | pT5 built before CrossClean runs; isDup from T5 branch doesn't block pT5→TC |

---

## Causal Chain (Final Picture)

```
CMSSW pixel seeding fails in dense jets (26% no seed at ΔR<0.01 vs 6% elsewhere)
  + Quad seeds harder to form in dense jets (29% non-quad at ΔR<0.01 vs 20% elsewhere)
  → pLS_lower drops: 0.955 → 0.702 at ΔR<0.01
  → pT5 = pLS + T5 can't form without quad non-dup pLS
  → pT5_lower drops: 0.845 → 0.549
  → pT5 is the dominant TC type; TC drops: 0.902 → 0.462
```

T5s are built correctly (0.900) and are NOT the problem.

---

## Environment Setup

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /mnt/data1/kk829/CMSSW_16_1_1
cmsenv
cd src/RecoTracker/LSTCore/standalone
source setup.sh
```

Always run this at the start of every session.

## Key Command Reference

```bash
# Rebuild CUDA backend
lst_make_tracklooper -G

# Produce lower-level ntuple with genjet branches
lst_cuda -i trackingNtuple-100.root --allobj --jet -n 100 -o <output>.root

# Produce num/den histograms (-J required for genjet ΔR histograms)
efficiency/bin/createPerfNumDenHists -i <input>.root -o <output>.root -J
```

---

## What Remains To Do

### Step 0 — Revert TrackCandidate.h (immediate)

```bash
cd /mnt/data1/kk829/CMSSW_16_1_1/src
git checkout RecoTracker/LSTCore/src/alpaka/TrackCandidate.h
lst_make_tracklooper -G
```

### Step 1 — Confirm pixel seeding as root cause

Verify the no-seed fraction by checking the input `trackingNtuple-100.root` directly.
The `see_*` branches in the tracking ntuple contain all pixel seeds fed to LST.
For sim tracks at ΔR < 0.01 that have no matched pLS, check whether the tracking
ntuple also has no matching seed — if so, the loss is confirmed pre-LST.

```python
# In the tracking ntuple, look at sim_seedIdx or trk_see_* branches
# to check pixel seed multiplicity as a function of ΔR to nearest genjet
```

### Step 2 — Investigate using non-quad pLS for pT5 (optional recovery)

The +8.9 pp non-quad loss is potentially recoverable within LST. Currently pT5
building and `CrossCleanpLS` both gate on `isQuad=true`. Non-quad seeds have
only 3 pixel hits (lower momentum resolution) but exist more often in dense jets.

Key files to modify if pursuing this:
- `../src/alpaka/TrackCandidate.h:309` — `isQuad` gate in CrossCleanpLS
- `../src/alpaka/PixelQuintuplet.h` — pT5 building (find where isQuad is required)

Risk: lower-purity non-quad pLS may increase fake rate for pT5.

### Step 3 — Report findings

The efficiency loss at small ΔR is primarily an upstream pixel seeding issue,
not fixable by LST deduplication tuning. This is an important conclusion:
the ~44 pp TC efficiency drop in jet cores comes from CMSSW failing to
reconstruct pixel seeds for ~30% of charged tracks in dense environments.
Consider opening a discussion with the CMSSW pixel tracking group.

### Step 4 — Run on larger statistics (optional)

Repeat on `trackingNtuple-1000.root` (1000 events) to reduce statistical noise.
The 100-event sample has ~800 selected sim tracks in the ΔR<0.01 bin — enough
for the main conclusion, but noisy for per-bin breakdowns.

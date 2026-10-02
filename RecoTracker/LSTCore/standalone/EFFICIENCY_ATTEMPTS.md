# LST ΔR Efficiency — Ledger of Attempts

**Living document.** Records every intervention tried to recover the TC-efficiency drop in
dense jet cores (ΔR = sim track to nearest GenJet center), plus proposed-but-unapplied changes.
Update the tables below whenever a new experiment is run or proposed. Companion to `HANDOFF.md`
(per-session narrative) and the memory `project-efficiency-attempts-ledger`.

_Last updated: 2026-09-22 (Sessions 3–22, 40, 42–44). See section C for the Session 44 LST-internal stage scan, which corrects several conclusions below._

## Reading the numbers — two caveats

1. **The baseline changed partway through.** Sessions 3–14 used the **real-seed** baseline
   (TC eff ≈ 0.462 at ΔR<0.01). Sessions 15–22 used the **fixed ideal-pLS** baseline
   (TC eff ≈ 0.690 / 247 tracks at ΔR<0.01). Numbers are only comparable **within the same
   baseline**.
2. All effects are quoted at **ΔR<0.01** (jet core, the problem region) unless noted.

---

## A. Applied changes / experiments

### A.1 Seeding interventions (upstream availability)

| Change | Motivation | Effect on efficiency |
|---|---|---|
| `--fillmissingpls` (S13) | Give a synthetic seed only to tracks CMSSW never seeded, to isolate the seed-availability gap | **+6.4 pp** (0.502→0.566); most rescued tracks landed in "non-quad" instead of forming a TC |
| `--idealpls` full synthetic replacement (S12/15) | Measure how much of the drop is upstream seeding vs. LST-internal | **~0.3→0.6 TC eff** — single biggest lever; established the drop is mostly **seed availability, not seed quality** |
| Momentum-direction bug fix in `TruthPixelSeeds.cc` (S15) | Synthetic seeds stored PCA momentum instead of last-pixel-hit momentum, biasing `betaIn` | **pT5_lower 0.127→0.840** at ΔR>0.05 — repaired the synthetic-seed *test harness*, not production |

### A.2 pT5-matching cut loosenings (ideal-pLS baseline, all ~4×)

| Change | Motivation | Effect |
|---|---|---|
| dBeta cut 4× (S16) | dBeta may reject contaminated dense-core pLS+T3 pairs | No convincing improvement — **ruled out** |
| betaOutCut 4× (S16–17) | betaOut is most susceptible to neighbor-hit contamination | pT5_lower **bit-for-bit identical**, TC −1 (noise) — **ruled out** |
| z-window + z-pointed + dPhi 4× (S18) | Last untested geometric gates in the T3 sub-algo | **+0.0028 pp** (246→247, noise) — **ruled out** |
| Loosen rPhiChiSquared thresholds 50–100% (proposed S7) | Dense-core median χ² exceeded the tightest thresholds | **Never tested — recommendation RETRACTED** (S10–11: confounded by pT mix; cut only applies <5 GeV but 98.5% of dense-core survivors are >5 GeV) |

### A.3 DNN gate

| Change | Motivation | Effect |
|---|---|---|
| Disable pT5 DNN pre-filter `--nopt5dnn` (S14) | The DNN may systematically reject dense-region candidates | **Zero efficiency gain** (358/607 identical), fake rate **~doubled** — **ruled out**; DNN is doing its job |
| Retrained T5 BeforeTC DNN weights (S40) | `master-kasia-dnn` branch holds retrained `T5NeuralNetworkWeights.h` (commit `b3bac69f7b3`); used in `RemoveDupQuintupletsBeforeTC` 6D embedding distance condition | **Slight efficiency improvement** vs upstream weights (AB ON, ideal pLS, 50-evt QCD; see `eff_vs_deltaR_retrained_T5dnn_vs_standard_ABon_ideal.png`). Affects T5-type and TC efficiency. Retrained weights currently active in working tree; decision pending whether to commit. |

### A.4 Dedup / cross-cleaning changes

| Change | Baseline | Motivation | Effect |
|---|---|---|---|
| Dedup Fix1: `RemoveDupQuintupletsBeforeTC` require nMatched≥3 (S3) | real | Stop T5 dedup killing genuine distinct tracks | **+0.004 TC eff** (negligible) |
| Dedup Fix2: + d²<0.01 at dR<0.02 (S3) | real | Same, tighter | **+0.007** (negligible) |
| Disable CrossCleanpLS T5 branch (S3) | real | pLS killed by nearby T5 TCs | **+0.006** (negligible) |
| Exp A: disable CrossCleanT5 vs-pT5 (S19–20) | ideal | Rescue standalone T5s near pT5s | **0** (230→230) |
| Exp B: remove isPT5 priority kill (S19–20) | ideal | Score-based dedup, not unconditional pT5 priority | **0** |
| Exp D: disable tightCutFlag (S19–20) | ideal | Tight χ²+DNN gate on standalone T5 promotion | **−1 TC** (+7 fakes) |
| Exp ALL (A+B+D) (S20) | ideal | All three combined | **+1 TC** (247→248) |
| Exp F: skip standalone-vs-pT5 dedup (`&&`→`||`, F1+F2) (S20–21) | ideal | A standalone T5 shouldn't be killed by a *different* track's pT5-embedded T5 | Total TC **+3** (247→250), T5-type 0→4; below the >5-genuine bar; cost ~7 TCs at large ΔR |
| Exp F+D (S21) | ideal | F plus tightCut off | Total TC **+4** (251) |
| Exp G: hit-overlap-only before-TC dedup (S21) | ideal | Replace geometric-proximity dedup with hit-sharing | Total TC **+10** (247→257) **but** pT5_lower unchanged and **fake rate exploded** (0.09→0.44 at ΔR>0.05) — **unusable** |
| Exp H1/H2: remove pT5-building visibility gates (S21) | ideal | pT5 builder skips isDup T5s/pLS, hiding genuine pT5s | pT5 *objects* **+29** (0.687→0.768) **but** pT5-type TC eff **dropped** (0.642→0.606) — freed objects then killed by pT5-vs-pT5 dedup |
| Combined outward+inward χ² pT5 dedup (S22) | ideal | Inward χ² might reject mixed (impure) pT5s that win on outward score | **+1 TC** (217→218), fake rate **worse** (0.281→0.354) — **insufficient** |

### Bottom line (applied)

- **The only large lever is seed availability** (ideal-pLS, ~+0.3 TC eff) — an *upstream CMSSW
  pixel-seeding* problem, not an LST fix.
- Every **cut-loosening** (dBeta, betaOut, z/dPhi, χ², DNN) was **null or ruled out**.
- Every **dedup change** either did nothing (≤+1 TC), added genuine objects the *next* dedup
  stage killed (Exp H), or recovered TCs only by exploding the fake rate (Exp G).
- Converged diagnosis: the pT5-vs-pT5 dedup **score is purity-blind**, and no purity-aware
  replacement tried yet (FP16 tiebreaker, inward χ²) has fixed it.

---

## B. Proposed changes — discussed, NOT yet applied

| Change | Motivation | Status / expected effect |
|---|---|---|
| Use T5 `dnnScore` as pT5 dedup discriminant | Inward χ² insufficient (still misranks 43%); need a genuinely purity-aware signal | **Offline-evaluated S44 (C.3):** `dnnScore` alone +27 core TCs real pLS but **−52 ideal**, +80 fakes — not recommended alone. Combined **K5** = (pT5 rPhiχ² + T5 score_rphisum)/T5 dnnScore: +27 real / +13 ideal, −54 fakes → leading candidate (Fix C). `dnnScore` reachable via `quintupletIndices()`. |
| Use T5 `rzChiSquared` as pT5 dedup discriminant | Same — alternative/additional purity-aware signal | **Untested.** Also in Quintuplets SoA, not propagated to pT5. |
| Redesigned pT5-dedup winner selection (per-seed keep pairing with most shared hits / purity-robust criterion), with visibility gates open | Score misranks under contamination; H showed genuine pT5 *objects* exist but lose dedup | **Untested (candidate FIX experiment).** Success = pT5-type TC eff follows pT5_lower recovery (0.768). Design depends on *why* score misranks. |
| Disable / limit `ExtendTrackCandidatesFromDupT5` (TrackCandidate.h:746) in dense cores | Staples neighbor/mixed-T5 hits onto genuine TCs → (16,12) vs (14,10) hits, match 12/14→12/16 fails >0.75 → TC flips to fake | **ON HOLD — USER DIRECTIVE: do NOT act yet.** Ready evidence: `pt5_dedup_winner_purity.png`, the 12/14→12/16 fingerprint. Provenance = native LST dev code (GNiendorf), not Claude's. **S44 measurement:** 26 core tracks lost at ΔR<0.02 (real pLS; 14 ideal), 0 ever gained; 45 genuine pT5 TCs + 17 genuine T5 TCs turned fake → est. +2.1 pp if disabled (Fix D). **Hold lifted for TESTING (user, 2026-09-22):** runtime switch `LST_EXTEND_DUPT5=0` skips the kernel (`LSTEvent.dev.cc`, default 1 = master behaviour). Not yet run. |
| Allow non-quad (3-hit) pLS into pT5 building itself | +8.9 pp of the ΔR<0.01 pLS deficit is non-quad seeds; within LST reach | **Premise CORRECTED S44:** there is **no** `isQuad` gate in `CreatePixelQuintupletsFromMap` — triplet pLS already enter pT5 building. Their disadvantage is losing `CheckHitspLS` to quads, `CrossCleanpLS` skipping them (`TrackCandidate.h:309`), and the TC gate (`:560,681`). Conversion 0.638 (triplet) vs 0.814 (quad). |
| Seed-side dedup study / relax `CheckHitspLS` | pLS dedup (≥3 shared pixel hits) may flag genuine seeds in cores (H2−H1 gap = 10 tracks) | **Studied offline S44 (C.2) → Fix B.** Verified double-count bug for triplet seeds; est. ~+2 pp. Reco pLS `isDup` branch still needed to validate the emulation. |
| Report upstream seed deficit to CMSSW pixel-tracking group | Dominant loss is CMSSW seeding failure (~26% no seed at ΔR<0.01), outside LST | **Not a code change / not done.** Now precisely characterized and ready to report. |

### Analysis-only next steps (not interventions)

- Characterize **non-overlap** real-vs-synthetic pLS populations (real-only vs synth-only tracks).
- **Chain-drift quantification**: tracks lost purely to dedup annihilation (T5: 771/2072
  victims; pT5: 1173/4117) — decides whether ordered/iterative dedup is needed.
- Bin Δ(1/pT) vs η and vs pT to map where real pixel-seed curvature resolution degrades.

### Standing constraints / open decisions

- **No new TrackCandidate types** (user constraint) — any fix must live within pT5/pT3/T5/T4/pLS.
- **Only LST-internal changes are possible** (user constraint, S44) — CMSSW pixel seeding is out of reach.
- **Source tree not at baseline**: three files carry the combined-dedup change (S22) —
  `interface/PixelQuintupletsSoA.h`, `src/alpaka/PixelQuintuplet.h`, `src/alpaka/Kernels.h`.
  Decision pending: revert to baseline, or keep as scaffolding for the dnnScore/rzChiSquared
  discriminant. _(Note S44: HANDOFF records Kernels.h/PixelTriplet.h reverted to master in S34;
  current tree = branch `claude-edits` with runtime-tunable dedup cuts, S42.)_

---

## C. Sessions 42–44 (2026-09-15 → 2026-09-22) — real-pLS, AB-ON, ΔR<0.02 unless noted

Baseline for C: `Ntuple-files/LSTNtuple_realpls_mastercuts_100evt_v2.root` — core denom 1224,
570 fails, TC eff **53.4%** (ideal pLS 72.8%). Genuine = match fraction >0.75.

### C.1 Applied / measured (S42–43)

| Change | Effect |
|---|---|
| `oat7` one-at-a-time dedup-cut scan, 26 points, 100 evt (S42–43) | Only `PT5_NM` raises core010 eff (+0.031 at NM=14) and it is **pure duplicates** (n_tc ×1.8, dupfakerate 0.175→0.759). `AB_NM=6` +0.011 at flat fakes (~2σ, unconfirmed). All BTC params flat; `BTC_DR2T=0` adds 2,290 junk TCs, 0 efficiency. |
| Disable `RemoveDupPixelQuintupletsFromMap` (allobj3 vs allobj4, AB-OFF, 50 evt; S43 4-agent replication) | Core eff 0.669→0.815 (+76–97 genuine sims) but TC flood 36–324×, fake 0.85–0.92 — **unusable**, but proves the loss is in the kernel. |
| Purity-aware-winner oracle, approximate killer search (S43 §9) | Reported ceiling +8 (0.65%). **SUPERSEDED by C.3** — approximation misattributed killers. |

### C.2 S44 stage scan — early stages (offline, read-only)

| Stage | Finding | Est. effect |
|---|---|---|
| Hits → MD → LS | Core ≈ non-core everywhere (MD 94.8% vs ~94%; LS 99.1% vs ~99%). Missing MDs = kinked truth hits (97.5%); truth-clean loss 0.17%. No overflow (stored = attempted in 200/200 events). | ~1% of fails — **not a lever** |
| Seed-algo filter `see_algo ∈ {4,22}` (`LSTPrepareInput.h:102`) | 63/259 no-pLS core fails have a perfect seed of another algo (54 pixelPair) | **Fix E**: +2–3 pp (needs 2-hit handling) |
| `CheckHitspLS` (`Kernels.h:504-585`) | **Bug:** triplet padding `hitIdx3 = hitIdx2` double-counts → 2 shared hits reach threshold 3; asymmetric count; flags 73% of pLS; quad-beats-triplet purity-blind | **Fix B**: ~+2 pp |
| Strict `>0.75` pLS match | 3/4-correct quads never count as genuine → 23+67 "no-pLS" tracks actually have a (partly) correct pLS | analysis artifact |
| pT cut, pLS capacity | 0 genuine core seeds lost | none |

### C.3 S44 stage scan — later stages (offline, read-only)

| Stage | Finding | Est. effect |
|---|---|---|
| pLS+T5 → pT5, high pT | 205 "pLS but no pT5" (not 176). Pixel 1/pT resolution 116% above 150 GeV → `passRadiusCriterion` (27) + pT3-DNN (27) + tracklet (15) reject genuine pairs; genuine pLS "stolen" by foreign T5 in 81 tracks (68 fake TCs). Pixel-hit residual to T5 circle separates genuine (78 µm) from stolen (645 µm). | **Fix A**: +3.5–4 pp, likely fewer fakes |
| pT5 dedup (exact replay, 22,208/22,209 reproduced) | Killer is fake in 82/135; chain-kill (killer already dead, no `isDup` check) 25; true displacement only 2. Truth ceiling **+104 (+8.5 pp)**, −199 fakes. | **Fix C** (K5 winner score): +2.2 pp real / +1.1 ideal, −54 fakes |
| `ExtendTrackCandidatesFromDupT5` | 26 tracks lost, 0 gained | **Fix D**: +2.1 pp — testable via `LST_EXTEND_DUPT5=0` |
| Stale `partOfPT5` after pT5 dedup | Blocks pT3 fallback (0 genuine pT3 for pool); releasing T5s offline: +0 tracks, +38 fakes | not a lever alone |
| BeforeTC `isDup & 1`; CrossCleanT5-vs-pT3 dR² | 0 tracks each | not levers |
| Greedy dedup; pixel-sharing-only dedup | +38 / +29 tracks but +885 / +508 fakes | not recommended alone |
| Occupancy caps (T3/T5/pT3/pT5/TC) | never hit | none |

Status (2026-09-22): Fixes A–D **implemented as default-off runtime switches**
(`LST_PT5_HIGHPT_GATE`, `LST_PLS_DISTINCT_HITS`, `LST_PT5_SCORE_MODE`, `LST_EXTEND_DUPT5=0`) and
**being validated** by `efficiency/run_s44_fixes.sh` (results: `efficiency/s44_fixes_log.txt`).
Fix A was implemented as an additive fallback, not a replacement gate (see HANDOFF Session 44).
Fix E deferred. Estimates overlap (A and C both target wrong pLS↔T5 pairings).

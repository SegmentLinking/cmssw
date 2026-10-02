# Plan (b) — Experiment H: pT5 visibility (let pT5 building see dedup-flagged objects)

Written 2026-07-06 ~04:00 EDT (Session 21). Author: Claude, per user request
("write up a plan for (b), save it, then execute").

## Motivation

The user's driving question: **what is removing so many pT5s at jet cores?**

Established so far (Sessions 15–21, all with `--idealpls`, 100-event PU200, ΔR<0.01,
q=−1 denominator = 358 tracks):

1. `pT5_lower` = 0.687 at the core vs `T5_lower` = 0.902 — the T5s exist, the pT5s don't.
2. Every pairing cut (dBeta, betaOut, z-window, dPhi, radius, DNN, χ²) was loosened 4×
   with **zero** effect (Sessions 16–18).
3. `CreatePixelQuintupletsFromMap` **skips duplicate-flagged objects**:
   - `PixelQuintuplet.h:680`: `if (quintuplets.isDup()[quintupletIndex]) continue;`
   - `PixelQuintuplet.h:666`: `if (pixelSegments.isDup()[i_pLS]) continue;`
4. `RemoveDupQuintupletsAfterBuild` runs BEFORE pT5 building and flags 97% of the
   failing tracks' truth-matched T5s (2072 victims across 59 tracks).
5. Tonight's winner-purity measurement (`dedup_winner_purity.png`) shows the after-build
   cluster winner is usually a same-track pure T5 (73/129), but those winners are then
   (i) killed by before-TC cross-track dedup (62), or (ii) consumed into pT5s built with
   the WRONG pLS (9) — and 771/2072 victims sit in fully-annihilated overlap
   neighborhoods (dedup chain drift), leaving NO local T5 for pairing at all.
6. Experiment G (hit-overlap-only before-TC dedup) rescued 10 TCs at the core but
   `pT5_lower` did not move (246→247) and standalone-T5 fakes exploded (T5-type TC count
   17→794 at core, fake rate 0.96) — confirming the pT5 loss happens BEFORE the
   before-TC stage, and that wholesale dedup removal is the wrong tool.

Hypothesis: **the pT5 deficit is caused by the isDup visibility gates in pT5 building,
not by the pairing cuts.** The pure T5 (or the pLS) is flagged as a duplicate by
upstream cleaning, so the genuine (pLS, T5) pair is never examined. If pairing is
allowed to see flagged objects, the genuine pT5s should form; the downstream pT5-vs-pT5
dedup (`RemoveDupPixelQuintupletsFromMap`, ≥7/14 shared hits + score) is the natural
place to absorb the resulting duplicates.

A secondary hypothesis (H2 vs H1 separation): the pLS-side skip matters too — in dense
cores the pLS duplicate-cleaning (`CheckHitspLS`, ≥3 shared pixel hits) may flag the
track's only seed, making a pT5 impossible regardless of T5 availability. (Session 15
dismissed pLS cleaning based on pLS_lower≈0.99, but that metric counts isDup objects —
the reco flag was never checked.)

## Experiment design

Two cumulative variants, run sequentially with automatic patch apply/restore:

- **H1** (`t5tc_patch.py` target `H1`): comment out the T5-isDup skip at
  `PixelQuintuplet.h:680`. pT5 pairing sees ALL T5s, including dedup-flagged ones.
- **H2** (`H1` + target `H2`): additionally comment out the pLS-isDup skip at `:666`.
  Pairing sees all (pLS, T5) combinations.

Everything else at production state (NO F1/F2/G — this isolates the visibility effect;
TC formation, cross-cleaning, and pT5 dedup all run unchanged).

For each variant: `lst_make_tracklooper -G` → `lst_cuda -i trackingNtuple-100.root
--allobj --jet --idealpls -n 100 -o LSTNtuple_expH{1,2}.root` →
`createPerfNumDenHists ... -J`. Outputs: `LSTNtuple_expH1.root` / `LSTNumDen_expH1.root`,
`LSTNtuple_expH2.root` / `LSTNumDen_expH2.root`. Log: `t5tc_expH.log`.
The diagnostic branches (t5_isDupBits, pT5_isDupReco, …) are compiled in and will be
present in both ntuples.

## Metrics to read (vs `LSTNumDen_idealpls_fixed.root` baseline)

1. **Primary: `pT5_lower` at ΔR<0.01** (baseline 0.687 = 246/358). This is the raw
   object question — did genuine pT5s form?
2. `pT5`-type TC eff at ΔR<0.01 (baseline 0.642) — do the new pT5s survive pT5-dedup
   and cross-cleaning to become TCs?
3. Failure re-classification (`lst_classify_tc_failures.py`): does category D
   (59 tracks) migrate into "pass" or into category B (pT5 exists, killed by pT5 dedup)?
4. pT5-type fake rate at all ΔR (baseline 0.325 core / 0.025 far) and pT5-vs-pT5 dedup
   load — cost side.
5. Runtime sanity: pairing now examines more combinations; watch wall time and
   "Pixel Quintuplet excess alert" capacity effects (n_max_pixel_quintuplets cap).

## Expected outcomes and interpretations

- **Outcome 1 — H1 recovers most of the pT5_lower gap (0.687 → ≥0.85):**
  The T5-side visibility gate is THE pT5 remover. Interpretation: after-build T5 dedup
  + the line-680 skip together deprive pairing of genuine T5s. Fix direction becomes
  clear and surgical: either pair-before-dedup, or exempt pairing from the isDup veto
  (as tested), and let pT5 dedup handle duplicates. Category D should shrink sharply;
  watch whether the recovered tracks' pT5s then die in pT5-vs-pT5 dedup (migration to
  category B), which would shift the fight there.

- **Outcome 2 — H1 flat, H2 recovers:** The pLS-side gate dominates: the track's seed
  is duplicate-flagged in dense cores (CheckHitspLS ≥3-shared-pixel-hits is too loose
  for collimated tracks). Interpretation: the pT5 remover is pixel-seed cleaning, not
  T5 dedup; fix direction moves to CheckHitspLS criteria (e.g., require ≥3 shared hits
  AND same charge/pt window, or hit-fraction-based), a much smaller and cheaper kernel
  to change.

- **Outcome 3 — both flat (pT5_lower ≈ 0.687 in H1 and H2):** The pair never forms even
  when both objects are visible → the removal happens inside `runPixelQuintupletDefaultAlgo`
  after all (contradicting the Sessions 16–18 loosening nulls — would imply the loosening
  experiments missed a gate, e.g., the superbin/module-connectivity pairing map, which
  would then become the prime suspect again), or the objects' kinematics in dense cores
  put them outside each other's superbin connection entirely. Next step would be a
  truth-based pairing-map check (is the matched T5's lower module in
  `connectedPixels[superbin(pLS)]`?).

- **Outcome 4 — recovery but fake pT5 rate explodes (à la Experiment G):** The visibility
  gates were doing real fake suppression. Interpretation: the fix cannot be a plain gate
  removal; it needs a qualifier (e.g., allow isDup T5s into pairing only if no non-dup
  T5 shares ≥7 hits with them, or only the after-build second-pass bit blocks pairing).
  Quantify which gate (T5 vs pLS side) drives the fakes via H1-vs-H2 difference.

Risks / limits: (i) combinatorial growth of pairing work — acceptable for a 100-event
study; (ii) `n_max_pixel_quintuplets` cap could silently truncate in busy events (check
totOccupancy vs cap in the log if numbers look odd); (iii) duplicate pT5s inflating the
pT5 dedup stage — measured by metric 4.

## Execution steps

1. Add patch targets H1, H2 to `t5tc_patch.py` (exact-string, loud failure).
2. Create `t5tc_expH.sh` (lockfile-guarded): H1 cycle, restore, H1+H2 cycle, restore.
3. Launch detached; watcher notifies on completion (~2.5 h total for both variants).
4. Analyze: comparison script + failure classification on both ntuples; write results
   into HANDOFF.md; produce a pT5_lower-focused plot if the numbers warrant it.

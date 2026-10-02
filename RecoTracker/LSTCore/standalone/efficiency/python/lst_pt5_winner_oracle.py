#!/usr/bin/env python3
"""pT5 purity-aware-winner ORACLE study.

On a dedup-ON ntuple, quantify how many jet-core genuine sim tracks are lost
specifically because RemoveDupPixelQuintupletsFromMap killed ALL their genuine
pT5s, and split those losses by WHY (what pT5 won the dedup contest):

  CLEAN-RECOVERABLE  the winner is a FAKE pT5  -> a purity-aware winner keeps the
                     genuine one, kills the fake instead: +1 track, 0 fakes added.
  DISPLACEMENT       the winner is a genuine pT5 of a DIFFERENT sim track -> keeping
                     the loser risks losing the winner's track (near zero-sum).
  DOWNSTREAM         a same-track genuine pT5 survived FromMap but the track still
                     has no TC -> lost later (CrossCleanT5 / BeforeTC), not a
                     FromMap-winner problem.

The CLEAN-RECOVERABLE count is the purity-aware-winner ceiling at flat fake rate.

Runs entirely within a single ntuple, per event (no cross-file alignment), so the
allobj3/allobj4 event-permutation artifact does not apply.

Usage: python3 lst_pt5_winner_oracle.py <ntuple.root> [--nevents N] [--out margins.png]

Winner replay follows lst_pt5_pairs.py: dedup competitors are pT5s within
|dEta|<0.2 and |dPhi|<0.2; the nMatched>=7 gate is not re-derivable from the
ntuple (no per-pT5 hit lists stored) but empirically almost always holds in-window.
"""
import argparse
import math
import sys
from collections import Counter

import numpy as np
import uproot

FRAC = 0.75
DETA = 0.2
DPHI = 0.2
DR_BINS = [0.02, 0.05, 0.10]  # cumulative core bins

BRANCHES = [
    "sim_q", "sim_pt", "sim_eta", "sim_vx", "sim_vy", "sim_vz",
    "sim_genjet_deltaR", "sim_genjet_idx", "genjet_pt", "genjet_eta",
    "sim_tcIdx", "sim_pt5IdxAll", "sim_pt5IdxAllFrac",
    "pT5_eta", "pT5_phi", "pT5_score", "pT5_isDupReco", "pT5_simIdx",
]


def dphi(a, b):
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def in_denominator(ev, i):
    if ev["sim_q"][i] == 0:
        return False
    if not (ev["sim_pt"][i] > 0.9 and abs(ev["sim_eta"][i]) < 4.5):
        return False
    if not (abs(ev["sim_vz"][i]) < 30 and
            math.sqrt(ev["sim_vx"][i] ** 2 + ev["sim_vy"][i] ** 2) < 2.5):
        return False
    ji = ev["sim_genjet_idx"][i]
    if ji < 0 or not (ev["genjet_pt"][ji] > 1000 and abs(ev["genjet_eta"][ji]) < 2.5):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fname")
    ap.add_argument("--nevents", type=int, default=-1)
    ap.add_argument("--out", default=None, help="optional score-margin PNG")
    args = ap.parse_args()

    t = uproot.open(args.fname)["tree"]
    n_entries = t.num_entries
    stop = n_entries if args.nevents < 0 else min(args.nevents, n_entries)

    # per-bin tallies
    denom = Counter()
    fail = Counter()                # no TC
    lower = Counter()               # >=1 genuine pT5 built
    lower_no_tc = Counter()         # pT5_lower AND no TC
    all_killed = Counter()          # all genuine pT5s killed by FromMap, no TC
    clean = Counter()               # CLEAN-RECOVERABLE (fake winner)
    displacement = Counter()
    downstream = Counter()
    unresolved = Counter()
    clean_margins = []              # score[g]-score[w] for clean-recoverable

    data = t.arrays(BRANCHES, entry_stop=stop, library="np")

    for e in range(stop):
        ev = {b: data[b][e] for b in BRANCHES}
        pt5_eta = np.asarray(ev["pT5_eta"], dtype=float)
        pt5_phi = np.asarray(ev["pT5_phi"], dtype=float)
        pt5_score = np.asarray(ev["pT5_score"], dtype=float)
        pt5_dup = np.asarray(ev["pT5_isDupReco"]).astype(bool)
        pt5_sim = np.asarray(ev["pT5_simIdx"], dtype=int)
        nsim = len(ev["sim_pt"])

        for i in range(nsim):
            if not in_denominator(ev, i):
                continue
            dr = ev["sim_genjet_deltaR"][i]
            bins_hit = [b for b in DR_BINS if dr < b]
            if not bins_hit:
                continue
            for b in bins_hit:
                denom[b] += 1

            has_tc = ev["sim_tcIdx"][i] >= 0
            if not has_tc:
                for b in bins_hit:
                    fail[b] += 1

            genuine = [int(idx) for idx, fr in
                       zip(ev["sim_pt5IdxAll"][i], ev["sim_pt5IdxAllFrac"][i])
                       if fr >= FRAC]
            if not genuine:
                continue
            for b in bins_hit:
                lower[b] += 1
            if has_tc:
                continue
            for b in bins_hit:
                lower_no_tc[b] += 1

            surviving_same = [g for g in genuine if not pt5_dup[g]]
            if surviving_same:
                # a same-track genuine pT5 survived FromMap but no TC -> downstream
                for b in bins_hit:
                    downstream[b] += 1
                continue

            # all genuine same-track pT5s were killed by FromMap
            for b in bins_hit:
                all_killed[b] += 1

            # classify by the winner each killed genuine pT5 lost to
            saw_fake_winner = False
            saw_other_winner = False
            best_margin = None
            for g in genuine:  # all are killed here
                e1, p1, s1 = pt5_eta[g], pt5_phi[g], pt5_score[g]
                win = -1
                win_score = None
                for w in range(len(pt5_score)):
                    if w == g or pt5_dup[w]:
                        continue  # only survivors are candidate winners
                    if abs(e1 - pt5_eta[w]) > DETA:
                        continue
                    if abs(dphi(p1, pt5_phi[w])) > DPHI:
                        continue
                    if pt5_score[w] > s1:
                        continue  # winner must not have a worse score
                    if win_score is None or pt5_score[w] < win_score:
                        win_score = pt5_score[w]
                        win = w
                if win < 0:
                    continue
                if pt5_sim[win] < 0:
                    saw_fake_winner = True
                    m = s1 - win_score
                    if best_margin is None or m < best_margin:
                        best_margin = m
                elif pt5_sim[win] != i:
                    saw_other_winner = True
                # pt5_sim[win] == i cannot happen: that winner survived and matches i,
                # so surviving_same would be non-empty (handled above).

            if saw_fake_winner:
                for b in bins_hit:
                    clean[b] += 1
                clean_margins.append(best_margin)
            elif saw_other_winner:
                for b in bins_hit:
                    displacement[b] += 1
            else:
                for b in bins_hit:
                    unresolved[b] += 1

    # ---------------------------------------------------------------- report
    print(f"file: {args.fname}   events: {stop}/{n_entries}")
    print(f"{'metric':<40}" + "".join(f"dR<{b:<8}" for b in DR_BINS))
    def row(name, c):
        print(f"{name:<40}" + "".join(f"{c[b]:<10}" for b in DR_BINS))
    row("denominator (sim tracks)", denom)
    row("failed (no TC)", fail)
    row("pT5_lower (>=1 genuine pT5 built)", lower)
    row("pT5_lower AND no TC", lower_no_tc)
    row("all genuine pT5 killed by FromMap", all_killed)
    print("  --- of the all-killed, winner class: ---")
    row("  CLEAN-RECOVERABLE (fake winner)", clean)
    row("  DISPLACEMENT (other-genuine winner)", displacement)
    row("  DOWNSTREAM (same-track survivor)", downstream)
    row("  unresolved (no in-window winner)", unresolved)
    print()
    for b in DR_BINS:
        d = denom[b]
        if d:
            base_eff = (d - fail[b]) / d
            ceil_eff = (d - fail[b] + clean[b]) / d
            print(f"dR<{b}: base core eff={base_eff:.4f} ({d-fail[b]}/{d}); "
                  f"CLEAN ceiling +{clean[b]} -> {ceil_eff:.4f} (+{ceil_eff-base_eff:.4f})")
    if clean_margins:
        v = sorted(m for m in clean_margins if m is not None)
        n = len(v)
        n_tie = sum(1 for m in v if m == 0.0)
        print(f"\nclean-recoverable score margins (score_g - score_w): "
              f"median {v[n//2]:.4g}, min {v[0]:.4g}, max {v[-1]:.4g}; "
              f"FP16-tie (margin==0): {n_tie}/{n}")

    if args.out and clean_margins:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        v = [m for m in clean_margins if m is not None]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(v, bins=40, color="#0072B2", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("score(genuine loser) - score(fake winner)")
        ax.set_ylabel("clean-recoverable pT5s")
        ax.set_title(f"pT5 dedup clean-recoverable margins ({args.fname.split('/')[-1]})")
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"plot written: {args.out}")


if __name__ == "__main__":
    main()

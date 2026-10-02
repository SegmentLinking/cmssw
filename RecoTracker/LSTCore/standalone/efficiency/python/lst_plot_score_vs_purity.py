#!/usr/bin/env python3
"""Investigation (1): WHY does the pT5-vs-pT5 dedup score prefer contaminated pT5s?

Uses pT5_score (FP16-rounded, exactly what RemoveDupPixelQuintupletsFromMap compares).
Kernel rule: for a >=7-shared-hit pair, the HIGHER score is removed; on an exact tie
the higher INDEX is removed (index = creation order, nondeterministic on GPU).

Measurements on cat-B failing jet-core tracks (genuine matched pT5 exists, no TC):
  A. For each (victim, winner) pair sharing >=7 hits: delta = score_winner - score_victim.
     By kernel logic delta <= 0 or exact tie. Report the TIE fraction (arbitrary winner)
     vs strict-win fraction (score genuinely prefers the winner).
  B. Same-seed score-vs-purity: for seeds (pLS) having both a genuine (>=0.75) and a
     mixed pT5, compare min score of each class: does the mixed pairing genuinely
     score better, and by how much?
  C. (requires -d build: pT5_rPhiChiSquaredInwards branch)
     Same-seed combined-score comparison: combined = rPhiChiSquared + rPhiChiSquaredInwards.
     Does the combined metric reduce the mixed-wins fraction vs outward-only (Panel B)?

Usage: python3 lst_plot_score_vs_purity.py [ntuple.root] [out.png]
"""
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ROOT

FNAME = sys.argv[1] if len(sys.argv) > 1 else "LSTNtuple_expH2_score.root"
OUT = sys.argv[2] if len(sys.argv) > 2 else "pt5_score_vs_purity.png"
FRAC = 0.75
MIN_SHARED = 7

f = ROOT.TFile(FNAME)
t = f.Get("tree")

pair_deltas = []       # score_winner - score_victim for deciding pairs
n_tie = 0
n_strict = 0
n_wrongway = 0         # winner score > victim (shouldn't happen if truly compared)
seed_gaps = []         # min(score genuine) - min(score mixed) per same-seed group
n_seed_mixed_wins = 0
n_seed_genuine_wins = 0
n_seed_tie = 0
# Panel C: combined (outward + inward) score comparison
seed_gaps_combined = []
n_seed_mixed_wins_c = 0
n_seed_genuine_wins_c = 0
n_seed_tie_c = 0
has_inwards = False  # set True if branch exists
n_catB = 0

has_inwards = t.GetBranch("pT5_rPhiChiSquaredInwards") is not None

for ievt, ev in enumerate(t):
    dr = ev.sim_genjet_deltaR
    jidx = ev.sim_genjet_idx
    gpt = ev.genjet_pt
    geta = ev.genjet_eta
    p_dup = ev.pT5_isDupReco
    p_fake = ev.pT5_isFake
    p_t5 = ev.pT5_t5Idx
    p_pls = ev.pT5_plsIdx
    p_score = ev.pT5_score
    p_inwards = list(ev.pT5_rPhiChiSquaredInwards) if has_inwards else None
    i3a = ev.t5_t3Idx0
    i3b = ev.t5_t3Idx1
    h_x = [getattr(ev, f"t3_hit_{k}_x") for k in range(6)]
    h_y = [getattr(ev, f"t3_hit_{k}_y") for k in range(6)]
    h_z = [getattr(ev, f"t3_hit_{k}_z") for k in range(6)]
    pls_h = [
        (getattr(ev, f"pLS_hit{k}_x"), getattr(ev, f"pLS_hit{k}_y"), getattr(ev, f"pLS_hit{k}_z"))
        for k in range(4)
    ]
    pls_nhit = ev.pLS_nhit

    def pt5_hitkeys(j):
        keys = set()
        t5j = p_t5[j]
        for i3 in (i3a[t5j], i3b[t5j]):
            for k in range(6):
                keys.add((round(h_x[k][i3], 3), round(h_y[k][i3], 3), round(h_z[k][i3], 3)))
        pls = p_pls[j]
        for k in range(min(4, int(pls_nhit[pls]))):
            x, y, z = pls_h[k]
            keys.add((round(x[pls], 3), round(y[pls], 3), round(z[pls], 3)))
        return keys

    # group pT5s by seed for panel B (built lazily per relevant seed)
    seed_groups = defaultdict(list)
    for j in range(len(p_dup)):
        seed_groups[p_pls[j]].append(j)

    victims = {}
    nsim = len(ev.sim_pt)
    for i in range(nsim):
        if ev.sim_q[i] != -1:
            continue
        if not (ev.sim_pt[i] > 0.9 and abs(ev.sim_eta[i]) < 4.5):
            continue
        if not (abs(ev.sim_vz[i]) < 30 and (ev.sim_vx[i] ** 2 + ev.sim_vy[i] ** 2) ** 0.5 < 2.5):
            continue
        ji = jidx[i]
        if ji < 0 or not (gpt[ji] > 1000 and abs(geta[ji]) < 2.5):
            continue
        if not (dr[i] < 0.01) or ev.sim_tcIdx[i] >= 0:
            continue
        pt5s = [
            idx for idx, fr in zip(ev.sim_pt5IdxAll[i], ev.sim_pt5IdxAllFrac[i]) if fr >= FRAC
        ]
        if not pt5s:
            continue
        n_catB += 1
        genuine_set = set(pt5s)
        for j in pt5s:
            if p_dup[j]:
                victims[j] = i
        # ---- B: same-seed genuine vs mixed comparison ----
        seeds = set(p_pls[j] for j in pt5s)
        for s in seeds:
            grp = seed_groups[s]
            g_scores = [p_score[j] for j in grp if j in genuine_set]
            m_scores = [p_score[j] for j in grp if j not in genuine_set and p_fake[j]]
            if not g_scores or not m_scores:
                continue
            gap = min(g_scores) - min(m_scores)
            seed_gaps.append(gap)
            if gap > 0:
                n_seed_mixed_wins += 1
            elif gap < 0:
                n_seed_genuine_wins += 1
            else:
                n_seed_tie += 1
            # Panel C: combined score
            if p_inwards is not None:
                g_combined = [p_score[j] + p_inwards[j] for j in grp if j in genuine_set]
                m_combined = [p_score[j] + p_inwards[j] for j in grp if j not in genuine_set and p_fake[j]]
                if g_combined and m_combined:
                    gap_c = min(g_combined) - min(m_combined)
                    seed_gaps_combined.append(gap_c)
                    if gap_c > 0:
                        n_seed_mixed_wins_c += 1
                    elif gap_c < 0:
                        n_seed_genuine_wins_c += 1
                    else:
                        n_seed_tie_c += 1

    if not victims:
        continue

    # ---- A: victim-winner deciding comparisons ----
    inv = defaultdict(list)
    vkeys = {}
    for j in victims:
        ks = pt5_hitkeys(j)
        vkeys[j] = ks
        for k in ks:
            inv[k].append(j)
    overlap = Counter()
    for j in range(len(p_dup)):
        if j in victims:
            continue
        for key in pt5_hitkeys(j):
            if key in inv:
                for v in inv[key]:
                    overlap[(v, j)] += 1
    for (v, j), shared in overlap.items():
        if shared < MIN_SHARED or p_dup[j]:
            continue
        d = p_score[j] - p_score[v]
        pair_deltas.append(d)
        if d == 0.0:
            n_tie += 1
        elif d < 0:
            n_strict += 1
        else:
            n_wrongway += 1

print(f"cat-B failing core tracks: {n_catB}")
tot = n_tie + n_strict + n_wrongway
print(f"deciding (victim, winner) pairs: {tot}")
if tot:
    print(f"  exact FP16 TIE (arbitrary, index-ordered winner): {n_tie} ({n_tie/tot:.1%})")
    print(f"  strict score win (winner < victim):               {n_strict} ({n_strict/tot:.1%})")
    print(f"  wrong-way (winner > victim, transitive kill):     {n_wrongway} ({n_wrongway/tot:.1%})")
print(f"\nsame-seed groups with BOTH genuine and mixed pT5s: {len(seed_gaps)}")
if seed_gaps:
    print(f"  [outward only]  mixed wins: {n_seed_mixed_wins}  genuine wins: {n_seed_genuine_wins}  tie: {n_seed_tie}")
if seed_gaps_combined:
    print(f"  [combined out+in] mixed wins: {n_seed_mixed_wins_c}  genuine wins: {n_seed_genuine_wins_c}  tie: {n_seed_tie_c}")

# ------------------------------------------------------------------ plotting
INK, MUTED, GRID = "#333333", "#767676", "#DDDDDD"
C_BLUE, C_ORANGE, C_GREEN, C_VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"

ncols = 3 if seed_gaps_combined else 2
fig_w = 18.5 if ncols == 3 else 12.5
fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 4.8))
axA, axB = axes[0], axes[1]
axC = axes[2] if ncols == 3 else None

fig.suptitle("Why the pT5 dedup keeps the wrong object: score comparison anatomy (H2)",
             fontsize=12.5, color=INK, y=0.98)

axA.hist(pair_deltas, bins=60, color=C_BLUE, edgecolor="white", linewidth=0.4)
axA.axvline(0, color=C_VERM, linestyle="--", linewidth=1.2)
axA.set_xlabel("score(winner) − score(victim)   [0 = exact FP16 tie]", fontsize=10, color=INK)
axA.set_ylabel("deciding pairs", fontsize=10, color=INK)
axA.set_title("A — winner-vs-victim score difference", fontsize=10.5, color=INK, loc="left")
if tot:
    axA.text(0.03, 0.95, f"tie: {n_tie/tot:.0%}   strict win: {n_strict/tot:.0%}",
             transform=axA.transAxes, fontsize=9.5, color=INK, va="top")

axB.hist(seed_gaps, bins=60, color=C_ORANGE, edgecolor="white", linewidth=0.4)
axB.axvline(0, color=C_VERM, linestyle="--", linewidth=1.2)
axB.set_xlabel("min score(genuine) − min score(mixed), same seed\n[> 0: mixed pairing wins; outward χ² only]",
               fontsize=10, color=INK)
axB.set_ylabel("seed groups", fontsize=10, color=INK)
axB.set_title("B — same-seed: genuine vs mixed (outward only)", fontsize=10.5, color=INK, loc="left")
if seed_gaps:
    ntot_b = n_seed_mixed_wins + n_seed_genuine_wins + n_seed_tie
    axB.text(0.03, 0.95,
             f"mixed wins: {n_seed_mixed_wins/ntot_b:.0%}   genuine: {n_seed_genuine_wins/ntot_b:.0%}",
             transform=axB.transAxes, fontsize=9.5, color=INK, va="top")

if axC is not None:
    axC.hist(seed_gaps_combined, bins=60, color=C_GREEN, edgecolor="white", linewidth=0.4)
    axC.axvline(0, color=C_VERM, linestyle="--", linewidth=1.2)
    axC.set_xlabel("min (out+in)(genuine) − min (out+in)(mixed), same seed\n[> 0: mixed pairing wins; combined χ²]",
                   fontsize=10, color=INK)
    axC.set_ylabel("seed groups", fontsize=10, color=INK)
    axC.set_title("C — same-seed: genuine vs mixed (out + inwards)", fontsize=10.5, color=INK, loc="left")
    if seed_gaps_combined:
        ntot_c = n_seed_mixed_wins_c + n_seed_genuine_wins_c + n_seed_tie_c
        axC.text(0.03, 0.95,
                 f"mixed wins: {n_seed_mixed_wins_c/ntot_c:.0%}   genuine: {n_seed_genuine_wins_c/ntot_c:.0%}",
                 transform=axC.transAxes, fontsize=9.5, color=INK, va="top")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=160)
print(f"\nplot written: {OUT}")

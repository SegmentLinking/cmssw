#!/usr/bin/env python3
"""After-build dedup winner purity, measured from hit overlap.

For each category-D failing jet-core track (dR<0.01, q=-1, ideal pLS, Exp-F state):
its 'victims' are truth-matched (>=75%) T5s carrying the after-build kill bit
(t5_isDupBits & 1). A 'competitor' is any other T5 sharing >= 7 of the victim's 10
hits (the RemoveDupQuintupletsAfterBuild criterion); a 'winner' is a competitor
whose after-build bit is NOT set — the surviving representation of those hits that
pT5 building actually saw.

Outputs a two-panel figure:
  A: distribution of winner purity (t5_pMatched, unthresholded)
  B: winner class vs eventual fate (consumed by pT5 / killed before-TC / survivor)

Usage: python3 lst_plot_dedup_winner_purity.py [ntuple.root] [out.png]
"""
import math
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ROOT

FNAME = sys.argv[1] if len(sys.argv) > 1 else "LSTNtuple_expF_diag.root"
OUT = sys.argv[2] if len(sys.argv) > 2 else "dedup_winner_purity.png"
FRAC = 0.75
MIN_SHARED = 7  # after-build dedup threshold

f = ROOT.TFile(FNAME)
t = f.Get("tree")

winner_pm_unique = []           # unique (event, T5) winner purities
winner_class_fate = Counter()   # (class, fate) counts, unique winners
victims_total = 0
victims_no_winner = 0
winners_same_module = 0
winners_diff_module = 0
n_catD = 0

for ievt, ev in enumerate(t):
    dr = ev.sim_genjet_deltaR
    jidx = ev.sim_genjet_idx
    gpt = ev.genjet_pt
    geta = ev.genjet_eta
    dupbits = ev.t5_isDupBits
    popt5 = ev.t5_partOfPT5
    ptc = ev.t5_partOfTC
    pm = ev.t5_pMatched
    i3a = ev.t5_t3Idx0
    i3b = ev.t5_t3Idx1
    h_det = [getattr(ev, f"t3_hit_{k}_detId") for k in range(6)]
    h_x = [getattr(ev, f"t3_hit_{k}_x") for k in range(6)]
    h_y = [getattr(ev, f"t3_hit_{k}_y") for k in range(6)]
    h_z = [getattr(ev, f"t3_hit_{k}_z") for k in range(6)]

    def t5_hitkeys(j):
        keys = set()
        for i3 in (i3a[j], i3b[j]):
            for k in range(6):
                keys.add((h_det[k][i3], round(h_x[k][i3], 3), round(h_y[k][i3], 3), round(h_z[k][i3], 3)))
        return keys

    # ---- find cat-D failing core tracks and their victims ----
    victims = {}          # t5 idx -> track idx
    track_matched = {}    # track idx -> set of matched T5s
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
        matched = [
            idx for idx, fr in zip(ev.sim_t5IdxAll[i], ev.sim_t5IdxAllFrac[i]) if fr >= FRAC
        ]
        pt5s = [
            idx for idx, fr in zip(ev.sim_pt5IdxAll[i], ev.sim_pt5IdxAllFrac[i]) if fr >= FRAC
        ]
        if pt5s or not matched:
            continue
        if not [j for j in matched if not popt5[j]]:
            continue
        n_catD += 1
        track_matched[i] = set(matched)
        for j in matched:
            if dupbits[j] & 1:  # after-build victims only
                victims[j] = i
    if not victims:
        continue

    # ---- inverted index over victim hits ----
    inv = defaultdict(list)
    vkeys = {}
    for j in victims:
        ks = t5_hitkeys(j)
        vkeys[j] = ks
        for k in ks:
            inv[k].append(j)

    # ---- scan all T5s for >=7-hit overlap with any victim ----
    overlap = Counter()  # (victim, j) -> shared hit count
    nt5 = len(pm)
    for j in range(nt5):
        if j in victims:
            continue
        for i3 in (i3a[j], i3b[j]):
            for k in range(6):
                key = (h_det[k][i3], round(h_x[k][i3], 3), round(h_y[k][i3], 3), round(h_z[k][i3], 3))
                if key in inv:
                    for v in inv[key]:
                        overlap[(v, j)] += 1
    # note: shared middle-MD hits counted once per (v, j) via set semantics? The two T3s
    # of j overlap in 2 hits; those keys are identical and get counted twice. Correct by
    # rebuilding with sets for candidates only:
    cand = defaultdict(set)
    for (v, j), n in overlap.items():
        if n >= MIN_SHARED - 3:  # loose pre-filter, exact check below
            cand[j].add(v)

    winners_seen = set()
    victim_has_winner = {v: False for v in victims}
    for j, vs in cand.items():
        jkeys = t5_hitkeys(j)
        for v in vs:
            shared = len(jkeys & vkeys[v])
            if shared < MIN_SHARED:
                continue
            if dupbits[j] & 1:
                continue  # competitor also after-build-killed; not a winner
            victim_has_winner[v] = True
            # same-lower-module proxy: innermost hit detId
            if min(jkeys)[0] == min(vkeys[v])[0]:
                winners_same_module += 1
            else:
                winners_diff_module += 1
            if j in winners_seen:
                continue
            winners_seen.add(j)
            pmj = pm[j]
            trk = victims[v]
            if j in track_matched.get(trk, set()):
                cls = "same-track pure"
            elif pmj >= FRAC:
                cls = "other-track pure"
            elif pmj >= 0.5:
                cls = "mixed (0.5–0.75)"
            else:
                cls = "low (<0.5)"
            if popt5[j]:
                fate = "consumed by pT5"
            elif dupbits[j] & 2:
                fate = "killed before-TC"
            else:
                fate = "survivor (TC)" if ptc[j] else "survivor (no TC)"
            winner_pm_unique.append(pmj)
            winner_class_fate[(cls, fate)] += 1

    victims_total += len(victims)
    victims_no_winner += sum(1 for v, has in victim_has_winner.items() if not has)

print(f"cat-D tracks: {n_catD}, victims (after-build-killed pure T5s): {victims_total}")
print(f"victims with NO surviving >= {MIN_SHARED}-hit competitor: {victims_no_winner}")
print(f"unique winners: {len(winner_pm_unique)}")
print(f"winner-victim pairs same innermost module: {winners_same_module}, diff: {winners_diff_module}")
for (cls, fate), n in winner_class_fate.most_common():
    print(f"  {cls:>18} | {fate:>18}: {n}")

# ------------------------------------------------------------------ plotting
INK = "#333333"
MUTED = "#767676"
GRID = "#DDDDDD"
C_BLUE = "#0072B2"    # Okabe-Ito, CVD-safe
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_VERM = "#D55E00"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.8))
fig.suptitle(
    "After-build T5 dedup: who wins the clusters of jet-core failing tracks?",
    fontsize=12.5, color=INK, y=0.98,
)

# Panel A — winner purity distribution
axA.hist(winner_pm_unique, bins=[x / 20 for x in range(21)], color=C_BLUE, edgecolor="white", linewidth=0.6)
axA.axvline(FRAC, color=C_VERM, linestyle="--", linewidth=1.4)
axA.text(FRAC - 0.02, axA.get_ylim()[1] * 0.97, "75% match threshold",
         rotation=90, va="top", ha="right", fontsize=8.5, color=C_VERM)
axA.set_xlabel("winner purity  (t5_pMatched, unthresholded)", fontsize=10, color=INK)
axA.set_ylabel("winners", fontsize=10, color=INK)
axA.set_title("A — purity of surviving >= 7-shared-hit competitors", fontsize=10.5, color=INK, loc="left")

# Panel B — winner class vs fate (stacked horizontal bars)
classes = ["same-track pure", "other-track pure", "mixed (0.5–0.75)", "low (<0.5)"]
fates = ["consumed by pT5", "killed before-TC", "survivor (TC)", "survivor (no TC)"]
fate_colors = {"consumed by pT5": C_ORANGE, "killed before-TC": C_BLUE,
               "survivor (TC)": C_GREEN, "survivor (no TC)": MUTED}
ypos = range(len(classes))
left = [0.0] * len(classes)
for fate in fates:
    vals = [winner_class_fate.get((cls, fate), 0) for cls in classes]
    axB.barh(ypos, vals, left=left, height=0.62, color=fate_colors[fate], label=fate)
    left = [a + b for a, b in zip(left, vals)]
for y, tot in zip(ypos, left):
    if tot:
        axB.text(tot + max(left) * 0.01, y, f"{int(tot)}", va="center", fontsize=9, color=INK)
axB.set_yticks(list(ypos))
axB.set_yticklabels(classes, fontsize=9.5, color=INK)
axB.invert_yaxis()
axB.set_xlabel("winners", fontsize=10, color=INK)
axB.set_title("B — winner class vs eventual fate", fontsize=10.5, color=INK, loc="left")
axB.legend(fontsize=8.5, frameon=False, loc="lower right")

for ax in (axA, axB):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x" if ax is axB else "y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=160)
print(f"\nplot written: {OUT}")

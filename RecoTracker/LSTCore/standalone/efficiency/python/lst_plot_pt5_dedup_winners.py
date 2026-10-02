#!/usr/bin/env python3
"""pT5-vs-pT5 dedup winner analysis (pT5-level analogue of dedup_winner_purity.png).

For each category-B failing jet-core track (dR<0.01, q=-1: genuine matched pT5 exists
but no TC), its 'victims' are matched (>=75%) pT5s with the reco duplicate flag
(pT5_isDupReco==1) set by RemoveDupPixelQuintupletsFromMap. A 'competitor' is any
other pT5 sharing >= 7 of the victim's 14 hits (the kernel's criterion); a 'winner'
is a competitor with no duplicate flag — i.e., a pT5 that became a TC in its place.

Classifies winners by truth class and by whether they were built from the SAME pixel
seed (pLS) as the victim, and records the winner's T5-part purity (t5_pMatched).

Usage: python3 lst_plot_pt5_dedup_winners.py [ntuple.root] [out.png]
"""
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ROOT

FNAME = sys.argv[1] if len(sys.argv) > 1 else "LSTNtuple_expH2.root"
OUT = sys.argv[2] if len(sys.argv) > 2 else "pt5_dedup_winner_purity.png"
FRAC = 0.75
MIN_SHARED = 7  # RemoveDupPixelQuintupletsFromMap threshold (of 14 hits)

f = ROOT.TFile(FNAME)
t = f.Get("tree")

n_catB = 0
victims_total = 0
victims_no_winner = 0
winner_records = []  # (class, same_seed, t5_part_purity) unique per (event, winner)

for ievt, ev in enumerate(t):
    dr = ev.sim_genjet_deltaR
    jidx = ev.sim_genjet_idx
    gpt = ev.genjet_pt
    geta = ev.genjet_eta
    p_dup = ev.pT5_isDupReco
    p_fake = ev.pT5_isFake
    p_t5 = ev.pT5_t5Idx
    p_pls = ev.pT5_plsIdx
    t5pm = ev.t5_pMatched
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

    # ---- cat-B failing core tracks and their dup-killed matched pT5s ----
    victims = {}        # pT5 idx -> sim track idx
    track_pt5s = {}     # sim track -> set of matched pT5s
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
        track_pt5s[i] = set(pt5s)
        for j in pt5s:
            if p_dup[j]:
                victims[j] = i
    if not victims:
        continue

    inv = defaultdict(list)
    vkeys = {}
    for j in victims:
        ks = pt5_hitkeys(j)
        vkeys[j] = ks
        for k in ks:
            inv[k].append(j)

    overlap = Counter()
    npt5 = len(p_dup)
    for j in range(npt5):
        if j in victims:
            continue
        for key in pt5_hitkeys(j):
            if key in inv:
                for v in inv[key]:
                    overlap[(v, j)] += 1

    winners_seen = set()
    victim_has_winner = {v: False for v in victims}
    for (v, j), shared in overlap.items():
        if shared < MIN_SHARED:
            continue
        if p_dup[j]:
            continue
        victim_has_winner[v] = True
        if j in winners_seen:
            continue
        winners_seen.add(j)
        trk = victims[v]
        if j in track_pt5s.get(trk, set()):
            cls = "same-track genuine"
        elif not p_fake[j]:
            cls = "other-track genuine"
        else:
            cls = "fake (mixed)"
        same_seed = p_pls[j] == p_pls[v]
        winner_records.append((cls, same_seed, t5pm[p_t5[j]]))

    victims_total += len(victims)
    victims_no_winner += sum(1 for has in victim_has_winner.values() if not has)

print(f"cat-B failing core tracks: {n_catB}")
print(f"victims (dup-killed genuine pT5s): {victims_total}")
print(f"victims with NO >= {MIN_SHARED}-shared-hit non-dup competitor: {victims_no_winner}")
print(f"unique winners: {len(winner_records)}")
cnt = Counter((c, s) for c, s, _ in winner_records)
for (c, s), n in cnt.most_common():
    print(f"  {c:>20} | {'same seed' if s else 'diff seed':>9}: {n}")

# ------------------------------------------------------------------ plotting
INK = "#333333"
MUTED = "#767676"
GRID = "#DDDDDD"
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_VERM = "#D55E00"

classes = ["same-track genuine", "other-track genuine", "fake (mixed)"]
cls_colors = {"same-track genuine": C_GREEN, "other-track genuine": C_BLUE, "fake (mixed)": C_ORANGE}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.8))
fig.suptitle(
    "pT5-vs-pT5 dedup (H2): who displaces the genuine pT5s of jet-core failing tracks?",
    fontsize=12.5, color=INK, y=0.98,
)

# Panel A — winner's T5-part purity, stacked by class
bins = [x / 20 for x in range(21)]
data = [[pm for c, s, pm in winner_records if c == cls] for cls in classes]
axA.hist(data, bins=bins, stacked=True,
         color=[cls_colors[c] for c in classes], label=classes,
         edgecolor="white", linewidth=0.5)
axA.axvline(FRAC, color=C_VERM, linestyle="--", linewidth=1.4)
axA.text(FRAC - 0.02, axA.get_ylim()[1] * 0.97, "75% match threshold",
         rotation=90, va="top", ha="right", fontsize=8.5, color=C_VERM)
axA.set_xlabel("winner's T5-part purity  (t5_pMatched of its T5)", fontsize=10, color=INK)
axA.set_ylabel("winners", fontsize=10, color=INK)
axA.set_title("A — purity of the T5 inside the winning pT5", fontsize=10.5, color=INK, loc="left")
axA.legend(fontsize=8.5, frameon=False, loc="upper left")

# Panel B — winner class vs seed identity
seed_labels = ["same seed as victim", "different seed"]
ypos = range(len(classes))
left = [0.0] * len(classes)
seed_colors = {True: C_VERM, False: MUTED}
for same_seed, lab in ((True, seed_labels[0]), (False, seed_labels[1])):
    vals = [cnt.get((cls, same_seed), 0) for cls in classes]
    axB.barh(ypos, vals, left=left, height=0.6, color=seed_colors[same_seed], label=lab)
    left = [a + b for a, b in zip(left, vals)]
for y, tot in zip(ypos, left):
    if tot:
        axB.text(tot + max(left) * 0.01, y, f"{int(tot)}", va="center", fontsize=9, color=INK)
axB.set_yticks(list(ypos))
axB.set_yticklabels(classes, fontsize=9.5, color=INK)
axB.invert_yaxis()
axB.set_xlabel("winners", fontsize=10, color=INK)
axB.set_title("B — winner class vs pixel-seed identity", fontsize=10.5, color=INK, loc="left")
axB.legend(fontsize=8.5, frameon=False, loc="lower right")

for ax in (axA, axB):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y" if ax is axA else "x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=160)
print(f"\nplot written: {OUT}")

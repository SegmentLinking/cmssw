#!/usr/bin/env python3
"""Re-generate sim_vs_tc_sidebyside with sim track 29 and its 3 nearest pT5-matched
neighbours (sim 30, 31, 32) highlighted.
"""
import math
import ROOT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TYPE_LABEL = {4: "T5", 5: "pT3", 7: "pT5", 8: "pLS", 9: "T4"}
TYPE_COLOR  = {4: "#1f77b4", 5: "#2ca02c", 7: "#d1495b", 8: "#ff8c00", 9: "#7b3fa0"}
DRCAP = 0.4

# Jet axis (genjet 1): eta=-1.1604, phi=-2.3667
# Cluster of highlighted tracks: eta≈-1.05, phi≈-2.29
# Centre halfway between jet axis and cluster so both are visible.
_JET_ETA, _JET_PHI = -1.1604, -2.3667
_CLU_ETA, _CLU_PHI = -1.048, -2.286
_CEN_ETA = (_JET_ETA + _CLU_ETA) / 2   # ≈ -1.104
_CEN_PHI = (_JET_PHI + _CLU_PHI) / 2   # ≈ -2.326
XLIM = (_CEN_ETA - 0.15, _CEN_ETA + 0.15)
YLIM = (_CEN_PHI - 0.075, _CEN_PHI + 0.075)

# Highlighted sim track and its nearest pT5-matched neighbours
HIGHLIGHT = {
    29: ("★ sim29 (26 GeV)", "#FF0000", 350),
    30: ("sim30 (57 GeV)",   "#FF6600", 250),
    31: ("sim31 (7.7 GeV)",  "#FFAA00", 200),
    32: ("sim32 (40 GeV)",   "#FFD700", 220),
}


def sizes(pt):
    return [12 + 55 * math.log10(max(p, 0.1) + 1.0) for p in pt]


def draw_jets(ax, jets):
    for je, jp in jets.values():
        ax.plot(je, jp, marker="*", markerfacecolor="none", markeredgecolor="k",
                markersize=24, markeredgewidth=1.3, linestyle="none", zorder=4)


def scatter_tcs(ax, eta, phi, sz, typ, fake):
    for ty in sorted(set(typ)):
        for is_fake, marker, face in ((0, "o", True), (1, "x", False)):
            idx = [i for i in range(len(eta)) if typ[i] == ty and int(fake[i]) == is_fake]
            if not idx:
                continue
            col = TYPE_COLOR.get(ty, "#555555")
            kw = dict(s=[sz[i] for i in idx], marker=marker, zorder=3, alpha=0.9)
            if face:
                kw.update(c=col, edgecolors="k", linewidths=0.3)
            else:
                kw.update(c=col, linewidths=1.3)
            ax.scatter([eta[i] for i in idx], [phi[i] for i in idx], **kw)


f = ROOT.TFile.Open("LSTNtuple_idealpls_fixed.root")
t = f.Get("tree")
t.GetEntry(0)

s_eta_all = list(t.sim_eta)
s_phi_all = list(t.sim_phi)
s_pt_all  = list(t.sim_pt)
s_dr_all  = list(t.sim_genjet_deltaR)
s_q_all   = list(t.sim_q)
c_eta = list(t.tc_eta); c_phi = list(t.tc_phi); c_pt = list(t.tc_pt)
c_typ = list(t.tc_type); c_fake = list(t.tc_isFake)

# GenJet axes
jets = {}
for i, g in enumerate(t.sim_genjet_idx):
    if g >= 0 and 0 <= s_dr_all[i] < 0.05 and g not in jets:
        jets[g] = (s_eta_all[i] - t.sim_genjet_deltaEta[i],
                   s_phi_all[i] - t.sim_genjet_deltaPhi[i])

# Filter sim tracks (same as main script: pt>0.8, charged)
keep = [i for i in range(len(s_eta_all)) if s_pt_all[i] > 0.8 and s_q_all[i] != 0]
s_eta = [s_eta_all[i] for i in keep]
s_phi = [s_phi_all[i] for i in keep]
s_pt  = [s_pt_all[i]  for i in keep]
s_dr  = [s_dr_all[i]  for i in keep]

# Track which kept-indices correspond to the originals we want to highlight
# keep[k] = original sim index
keep_to_highlight = {k: orig for k, orig in enumerate(keep) if orig in HIGHLIGHT}

s_sz, c_sz = sizes(s_pt), sizes(c_pt)
DRCAP = 0.4
cval = [min(d, DRCAP) if d >= 0 else DRCAP for d in s_dr]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(19, 7.5), sharex=True, sharey=True)

# ── Left panel: sim tracks coloured by ΔR, highlights on top ──────────────────
sc = axL.scatter(s_eta, s_phi, s=s_sz, c=cval, cmap="plasma_r", vmin=0, vmax=DRCAP,
                 alpha=0.85, edgecolors="k", linewidths=0.2, zorder=2)
draw_jets(axL, jets)
cb = fig.colorbar(sc, ax=axL, pad=0.01)
cb.set_label(rf"$\Delta R$ to nearest GenJet (capped at {DRCAP})")

# Draw highlight markers
for k, orig in keep_to_highlight.items():
    label, col, ms = HIGHLIGHT[orig]
    axL.scatter([s_eta[k]], [s_phi[k]], s=ms, c=col, edgecolors="black",
                linewidths=1.5, zorder=10, marker="*",
                label=label)
    # small annotation offset to avoid overlapping the star
    axL.annotate(f"  sim{orig}", (s_eta[k], s_phi[k]),
                 fontsize=8, fontweight="bold", color=col,
                 xytext=(5, 5), textcoords="offset points", zorder=11)

axL.legend(loc="upper right", fontsize=8, framealpha=0.9)
axL.set_title(f"Reconstructable sim tracks pT>0.8, charged (n={len(s_eta)})\n"
              "Stars = sim29 (red) + 3 nearest pT5-matched neighbours")

# ── Right panel: TrackCandidates ──────────────────────────────────────────────
draw_jets(axR, jets)
scatter_tcs(axR, c_eta, c_phi, c_sz, c_typ, c_fake)
# TC legend
counts = {ty: sum(1 for x in c_typ if x == ty) for ty in sorted(set(c_typ))}
for ty in sorted(set(c_typ)):
    axR.scatter([], [], c=TYPE_COLOR.get(ty, "#555555"), s=60, edgecolors="k",
                linewidths=0.3, label=f"{TYPE_LABEL.get(ty, ty)} (n={counts[ty]})")
axR.scatter([], [], c="#555555", marker="o", edgecolors="k", s=60, label="genuine (filled)")
axR.scatter([], [], c="#555555", marker="x", s=60, linewidths=1.2, label="fake (×)")
axR.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9, fontsize=9)
axR.set_title(f"TrackCandidates (n={len(c_eta)})")

for a in (axL, axR):
    a.set_xlim(*XLIM); a.set_ylim(*YLIM)
    a.set_xlabel(r"$\eta$"); a.grid(True, alpha=0.25)
axL.set_ylabel(r"$\phi$ [rad]")
fig.suptitle("Event 0: sim tracks vs. TrackCandidates — sim29 (26 GeV) and 3 nearest pT5-matched neighbours highlighted",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "sim_vs_tc_sidebyside_evt0_sim29highlight_zoom.png"
fig.savefig(out, dpi=130)
print(f"Wrote {out}")

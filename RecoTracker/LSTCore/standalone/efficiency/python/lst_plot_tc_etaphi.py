#!/usr/bin/env python3
"""Scatter reconstructed TrackCandidate positions in (eta, phi) for a single event.

Companion to lst_plot_simtrack_etaphi.py: instead of sim tracks, plots the LST TrackCandidates
(tc_eta/tc_phi), colored by TC type (pT5/T5/pT3/pLS/T4), with genuine vs. fake distinguished
(filled vs. open markers) and marker size scaled by pT. GenJet axes are reconstructed from the
sim_genjet_* branches of the same event and overlaid for context.
"""
import argparse
import math
import ROOT
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# LSTObjType enum (interface/Common.h)
TYPE_LABEL = {4: "T5", 5: "pT3", 7: "pT5", 8: "pLS", 9: "T4"}
TYPE_COLOR = {4: "#1f77b4", 5: "#2ca02c", 7: "#d1495b", 8: "#ff8c00", 9: "#7b3fa0"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", default="LSTNtuple_idealpls_fixed.root")
    ap.add_argument("-t", "--tree", default="tree")
    ap.add_argument("-e", "--event", type=int, default=0)
    ap.add_argument("-d", "--deltaR", type=float, default=0.05,
                    help="near-jet ΔR threshold used to locate GenJet axes")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    f = ROOT.TFile.Open(args.input)
    t = f.Get(args.tree)
    evt = args.event
    t.GetEntry(evt)

    eta = list(t.tc_eta)
    phi = list(t.tc_phi)
    pt = list(t.tc_pt)
    typ = list(t.tc_type)
    fake = list(t.tc_isFake)

    # Reconstruct GenJet axes from sim tracks near a jet (genjet = sim - delta).
    jets = {}
    for i, g in enumerate(t.sim_genjet_idx):
        dr = t.sim_genjet_deltaR[i]
        if g >= 0 and 0 <= dr < args.deltaR and g not in jets:
            jets[g] = (t.sim_eta[i] - t.sim_genjet_deltaEta[i],
                       t.sim_phi[i] - t.sim_genjet_deltaPhi[i])

    size = [12 + 55 * math.log10(max(p, 0.1) + 1.0) for p in pt]

    fig, ax = plt.subplots(figsize=(11, 7))
    # GenJet axes behind everything.
    for g, (je, jp) in jets.items():
        ax.plot(je, jp, marker="*", markerfacecolor="none", markeredgecolor="k",
                markersize=26, markeredgewidth=1.3, linestyle="none", zorder=1)
    # One scatter call per (type, genuine/fake) so the legend is readable.
    counts = {}
    for ty in sorted(set(typ)):
        counts[ty] = sum(1 for x in typ if x == ty)
        for is_fake, marker, face in ((0, "o", True), (1, "x", False)):
            idx = [i for i in range(len(eta)) if typ[i] == ty and int(fake[i]) == is_fake]
            if not idx:
                continue
            col = TYPE_COLOR.get(ty, "#555555")
            kw = dict(s=[size[i] for i in idx], marker=marker, zorder=2, alpha=0.85)
            if face:
                kw.update(c=col, edgecolors="k", linewidths=0.3)
            else:
                kw.update(c=col, linewidths=1.2)
            ax.scatter([eta[i] for i in idx], [phi[i] for i in idx], **kw)

    # Legend: type colors + genuine/fake marker convention.
    handles = []
    for ty in sorted(set(typ)):
        lab = f"{TYPE_LABEL.get(ty, ty)} (n={counts[ty]})"
        handles.append(ax.scatter([], [], c=TYPE_COLOR.get(ty, "#555555"), s=60,
                                   edgecolors="k", linewidths=0.3, label=lab))
    ax.scatter([], [], c="#555555", marker="o", edgecolors="k", s=60, label="genuine (filled)")
    ax.scatter([], [], c="#555555", marker="x", s=60, linewidths=1.2, label="fake (×)")
    if jets:
        ax.plot([], [], marker="*", markerfacecolor="none", markeredgecolor="k",
                markersize=13, linestyle="none", label="GenJet axis")

    n_fake = sum(1 for x in fake if int(x) == 1)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\phi$ [rad]")
    ax.set_title(f"TrackCandidates — event {evt} "
                 f"({len(eta)} TCs, {n_fake} fake). Marker size ∝ log(pT).")
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-math.pi - 0.2, math.pi + 0.2)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9, fontsize=9)
    fig.tight_layout()

    out = args.output or f"tc_etaphi_evt{evt}.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}  ({len(eta)} TCs: " +
          ", ".join(f"{TYPE_LABEL.get(k, k)}={v}" for k, v in sorted(counts.items())) + ")")


if __name__ == "__main__":
    main()

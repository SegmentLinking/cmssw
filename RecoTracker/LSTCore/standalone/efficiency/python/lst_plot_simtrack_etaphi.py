#!/usr/bin/env python3
"""Scatter sim-track positions in (eta, phi) for a single event, to examine jet shape.

Reads an LST output ntuple (per-sim-track sim_eta/sim_phi/sim_pt/sim_q + sim_genjet_*),
picks one event (default: the densest jet-core event), and draws an eta-vs-phi scatter with
marker size scaled by pT and near-jet tracks (sim_genjet_deltaR < deltaR-cut) highlighted.
GenJet axes are reconstructed from sim_genjet_deltaEta/deltaPhi and overlaid.
"""
import argparse
import math
import ROOT
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", default="LSTNtuple_idealpls_fixed.root")
    ap.add_argument("-t", "--tree", default="tree")
    ap.add_argument("-e", "--event", type=int, default=-1,
                    help="event index; -1 = auto-pick the densest jet-core event")
    ap.add_argument("-d", "--deltaR", type=float, default=0.05,
                    help="near-jet ΔR threshold for highlighting")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    f = ROOT.TFile.Open(args.input)
    t = f.Get(args.tree)
    n_events = t.GetEntries()

    # Auto-pick the event with the most near-jet sim tracks (a prominent jet core).
    if args.event < 0:
        best_evt, best_n = 0, -1
        for iev in range(n_events):
            t.GetEntry(iev)
            n_near = sum(1 for dr in t.sim_genjet_deltaR if dr >= 0 and dr < args.deltaR)
            if n_near > best_n:
                best_n, best_evt = n_near, iev
        evt = best_evt
        print(f"Auto-picked event {evt} ({best_n} sim tracks with ΔR < {args.deltaR})")
    else:
        evt = args.event

    t.GetEntry(evt)
    eta = list(t.sim_eta)
    phi = list(t.sim_phi)
    pt = list(t.sim_pt)
    dr = list(t.sim_genjet_deltaR)
    deta = list(t.sim_genjet_deltaEta)
    dphi = list(t.sim_genjet_deltaPhi)
    gidx = list(t.sim_genjet_idx)

    near = [(d >= 0 and d < args.deltaR) for d in dr]
    # Marker size grows with log(pT): visible but not saturated by the steeply falling spectrum.
    size = [12 + 55 * math.log10(max(p, 0.1) + 1.0) for p in pt]

    # Reconstruct GenJet axes: genjet = sim - delta (one entry per unique jet index near a jet).
    jets = {}
    for i, g in enumerate(gidx):
        if g >= 0 and near[i] and g not in jets:
            jets[g] = (eta[i] - deta[i], phi[i] - dphi[i])

    # Color tracks by ΔR to the nearest GenJet (capped) so jet cores glow and the halo fades —
    # this reveals the radial jet shape better than a binary near/far split.
    cap = 0.4
    cval = [min(d, cap) if d >= 0 else cap for d in dr]

    fig, ax = plt.subplots(figsize=(11, 7))
    # GenJet axes drawn first (behind tracks), hollow, so core tracks stay visible on top.
    for g, (je, jp) in jets.items():
        ax.plot(je, jp, marker="*", markerfacecolor="none", markeredgecolor="#1f77b4",
                markersize=26, markeredgewidth=1.6, linestyle="none", zorder=1)
    if jets:
        ax.plot([], [], marker="*", markerfacecolor="none", markeredgecolor="#1f77b4",
                markersize=16, markeredgewidth=1.4, linestyle="none", label="GenJet axis")
    sc = ax.scatter(eta, phi, s=size, c=cval, cmap="plasma_r", vmin=0.0, vmax=cap,
                    alpha=0.85, edgecolors="k", linewidths=0.25, zorder=2)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label(rf"$\Delta R$ to nearest GenJet (capped at {cap})")

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\phi$ [rad]")
    ax.set_title(f"Sim-track positions — event {evt} "
                 f"({len(eta)} tracks, {sum(near)} with ΔR < {args.deltaR}). Marker size ∝ log(pT).")
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-math.pi - 0.2, math.pi + 0.2)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()

    out = args.output or f"simtrack_etaphi_evt{evt}.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

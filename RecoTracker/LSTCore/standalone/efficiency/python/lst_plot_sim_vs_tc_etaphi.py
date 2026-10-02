#!/usr/bin/env python3
"""Compare sim-track and TrackCandidate positions in (eta, phi) for a single event.

Produces two figures on identical axes:
  1. overlay      — sim tracks (grey) with TCs (colored by type) drawn on top
  2. side-by-side — sim tracks (colored by ΔR to GenJet) | TCs (colored by type)

Companion to lst_plot_simtrack_etaphi.py and lst_plot_tc_etaphi.py; shares their conventions.
"""
import argparse
import math
import ROOT
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TYPE_LABEL = {4: "T5", 5: "pT3", 7: "pT5", 8: "pLS", 9: "T4"}
TYPE_COLOR = {4: "#1f77b4", 5: "#2ca02c", 7: "#d1495b", 8: "#ff8c00", 9: "#7b3fa0"}
XLIM = (-5.5, 5.5)
YLIM = (-math.pi - 0.2, math.pi + 0.2)
DRCAP = 0.4


def sizes(pt):
    return [12 + 55 * math.log10(max(p, 0.1) + 1.0) for p in pt]


def draw_jets(ax, jets, label=True):
    for je, jp in jets.values():
        ax.plot(je, jp, marker="*", markerfacecolor="none", markeredgecolor="k",
                markersize=24, markeredgewidth=1.3, linestyle="none", zorder=4)
    if label and jets:
        ax.plot([], [], marker="*", markerfacecolor="none", markeredgecolor="k",
                markersize=14, markeredgewidth=1.2, linestyle="none", label="GenJet axis")


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


def tc_legend(ax, typ, loc="center left", anchor=(1.01, 0.5)):
    counts = {ty: sum(1 for x in typ if x == ty) for ty in sorted(set(typ))}
    for ty in sorted(set(typ)):
        ax.scatter([], [], c=TYPE_COLOR.get(ty, "#555555"), s=60, edgecolors="k",
                   linewidths=0.3, label=f"{TYPE_LABEL.get(ty, ty)} (n={counts[ty]})")
    ax.scatter([], [], c="#555555", marker="o", edgecolors="k", s=60, label="genuine (filled)")
    ax.scatter([], [], c="#555555", marker="x", s=60, linewidths=1.2, label="fake (×)")
    ax.plot([], [], marker="*", markerfacecolor="none", markeredgecolor="k",
            markersize=13, linestyle="none", label="GenJet axis")
    ax.legend(loc=loc, bbox_to_anchor=anchor, framealpha=0.9, fontsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", default="LSTNtuple_idealpls_fixed.root")
    ap.add_argument("-t", "--tree", default="tree")
    ap.add_argument("-e", "--event", type=int, default=0)
    ap.add_argument("-d", "--deltaR", type=float, default=0.05)
    ap.add_argument("--sim-ptcut", type=float, default=0.8,
                    help="min sim-track pT [GeV]; matches the LST ptCut so the sim side shows "
                         "only tracks eligible to form a TC (set 0 to disable)")
    ap.add_argument("--sim-keep-neutral", action="store_true",
                    help="keep neutral sim tracks (default: require charge != 0, since neutrals "
                         "leave no tracker hits and cannot be reconstructed)")
    ap.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="eta axis range (zoom); default full acceptance")
    ap.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="phi axis range (zoom); default full 2π")
    ap.add_argument("--overlay-out", default=None)
    ap.add_argument("--side-out", default=None)
    args = ap.parse_args()

    xlim = tuple(args.xlim) if args.xlim else XLIM
    ylim = tuple(args.ylim) if args.ylim else YLIM
    zoomed = bool(args.xlim or args.ylim)

    f = ROOT.TFile.Open(args.input)
    t = f.Get(args.tree)
    evt = args.event
    t.GetEntry(evt)

    s_eta, s_phi, s_pt = list(t.sim_eta), list(t.sim_phi), list(t.sim_pt)
    s_dr, s_q = list(t.sim_genjet_deltaR), list(t.sim_q)
    c_eta, c_phi, c_pt = list(t.tc_eta), list(t.tc_phi), list(t.tc_pt)
    c_typ, c_fake = list(t.tc_type), list(t.tc_isFake)

    # Locate GenJet axes from the FULL sim collection (before any cut) so none are lost.
    jets = {}
    for i, g in enumerate(t.sim_genjet_idx):
        if g >= 0 and 0 <= s_dr[i] < args.deltaR and g not in jets:
            jets[g] = (s_eta[i] - t.sim_genjet_deltaEta[i], s_phi[i] - t.sim_genjet_deltaPhi[i])

    # Restrict sim side to tracks eligible to form a TC: pT > ptcut and (by default) charged.
    n_sim_all = len(s_eta)
    keep = [i for i in range(n_sim_all)
            if s_pt[i] > args.sim_ptcut and (args.sim_keep_neutral or s_q[i] != 0)]
    s_eta = [s_eta[i] for i in keep]
    s_phi = [s_phi[i] for i in keep]
    s_pt = [s_pt[i] for i in keep]
    s_dr = [s_dr[i] for i in keep]
    sim_cut_label = (f"pT>{args.sim_ptcut:g}"
                     + ("" if args.sim_keep_neutral else ", charged"))

    # Restrict both collections to the (possibly zoomed) axis window so panel counts reflect
    # what is actually shown — for a single-jet view this is the jet's own multiplicity.
    def _win(cols):
        e, p = cols[0], cols[1]
        idx = [i for i in range(len(e))
               if xlim[0] <= e[i] <= xlim[1] and ylim[0] <= p[i] <= ylim[1]]
        return [[c[i] for i in idx] for c in cols]

    s_eta, s_phi, s_pt, s_dr = _win([s_eta, s_phi, s_pt, s_dr])
    c_eta, c_phi, c_pt, c_typ, c_fake = _win([c_eta, c_phi, c_pt, c_typ, c_fake])

    s_sz, c_sz = sizes(s_pt), sizes(c_pt)
    n_fake = sum(1 for x in c_fake if int(x) == 1)

    # ---------- Figure 1: overlay ----------
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.scatter(s_eta, s_phi, s=s_sz, c="#c2c7cc", alpha=0.55, edgecolors="none",
               zorder=1, label=f"sim tracks [{sim_cut_label}] (n={len(s_eta)})")
    draw_jets(ax, jets)
    scatter_tcs(ax, c_eta, c_phi, c_sz, c_typ, c_fake)
    # Minimal TC-type key for the overlay.
    for ty in sorted(set(c_typ)):
        ax.scatter([], [], c=TYPE_COLOR.get(ty, "#555555"), s=60, edgecolors="k",
                   linewidths=0.3, label=f"TC {TYPE_LABEL.get(ty, ty)}")
    ax.scatter([], [], c="#555555", marker="x", s=60, linewidths=1.2, label="fake TC (×)")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\eta$"); ax.set_ylabel(r"$\phi$ [rad]")
    ax.set_title(f"Sim tracks + TrackCandidates — event {evt} "
                 f"({len(s_eta)} reconstructable sim [{sim_cut_label}], "
                 f"{len(c_eta)} TC, {n_fake} fake). Size ∝ log(pT).")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9, fontsize=9)
    fig.tight_layout()
    suffix = "_zoom" if zoomed else ""
    out1 = args.overlay_out or f"sim_vs_tc_overlay_evt{evt}{suffix}.png"
    fig.savefig(out1, dpi=130)
    print(f"Wrote {out1}")

    # ---------- Figure 2: side-by-side ----------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(19, 7.5), sharex=True, sharey=True)
    cval = [min(d, DRCAP) if d >= 0 else DRCAP for d in s_dr]
    sc = axL.scatter(s_eta, s_phi, s=s_sz, c=cval, cmap="plasma_r", vmin=0, vmax=DRCAP,
                     alpha=0.85, edgecolors="k", linewidths=0.2, zorder=2)
    draw_jets(axL, jets)
    cb = fig.colorbar(sc, ax=axL, pad=0.01)
    cb.set_label(rf"$\Delta R$ to nearest GenJet (capped at {DRCAP})")
    axL.set_title(f"Reconstructable sim tracks [{sim_cut_label}] (n={len(s_eta)})")

    draw_jets(axR, jets, label=False)
    scatter_tcs(axR, c_eta, c_phi, c_sz, c_typ, c_fake)
    tc_legend(axR, c_typ)
    axR.set_title(f"TrackCandidates (n={len(c_eta)}, {n_fake} fake)")

    for a in (axL, axR):
        a.set_xlim(*xlim); a.set_ylim(*ylim)
        a.set_xlabel(r"$\eta$"); a.grid(True, alpha=0.25)
    axL.set_ylabel(r"$\phi$ [rad]")
    fig.suptitle(f"Event {evt}: sim tracks vs. TrackCandidates. Marker size ∝ log(pT).",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out2 = args.side_out or f"sim_vs_tc_sidebyside_evt{evt}{suffix}.png"
    fig.savefig(out2, dpi=130)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()

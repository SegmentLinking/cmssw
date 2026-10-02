#!/usr/bin/env python3
"""Overlay one lower-level object collection (pT5 / T5 / pLS) on sim tracks in (eta, phi).

Unlike the TC plots, this shows the objects *as built* — the full collection written to the
ntuple, before dedup/cross-clean selects which become TrackCandidates. Genuine (isFake==0) and
fake (isFake==1) objects are distinguished; the fake combinatorial cloud (huge for T5) is drawn
faint so genuine objects stay visible. Intended for a tight single-jet zoom.
"""
import argparse
import math
import ROOT
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# branch prefix -> (display label, color)
OBJ = {"pT5": ("pT5", "#d1495b"), "t5": ("T5", "#1f77b4"), "pLS": ("pLS", "#ff8c00")}
YLIM = (-math.pi - 0.2, math.pi + 0.2)
XLIM = (-5.5, 5.5)


def sizes(pt):
    return [12 + 55 * math.log10(max(p, 0.1) + 1.0) for p in pt]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", default="LSTNtuple_idealpls_fixed.root")
    ap.add_argument("-t", "--tree", default="tree")
    ap.add_argument("-e", "--event", type=int, default=0)
    ap.add_argument("--object", required=True, choices=list(OBJ))
    ap.add_argument("--sim-ptcut", type=float, default=0.8)
    ap.add_argument("--sim-keep-neutral", action="store_true")
    ap.add_argument("-d", "--deltaR", type=float, default=0.05)
    ap.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    xlim = tuple(args.xlim) if args.xlim else XLIM
    ylim = tuple(args.ylim) if args.ylim else YLIM
    label, color = OBJ[args.object]

    f = ROOT.TFile.Open(args.input)
    t = f.Get(args.tree)
    evt = args.event
    t.GetEntry(evt)

    # GenJet axes from full sim collection.
    jets = {}
    for i, g in enumerate(t.sim_genjet_idx):
        dr = t.sim_genjet_deltaR[i]
        if g >= 0 and 0 <= dr < args.deltaR and g not in jets:
            jets[g] = (t.sim_eta[i] - t.sim_genjet_deltaEta[i],
                       t.sim_phi[i] - t.sim_genjet_deltaPhi[i])

    def in_win(e, p):
        return xlim[0] <= e <= xlim[1] and ylim[0] <= p <= ylim[1]

    # Reconstructable sim tracks in window.
    s_eta, s_phi, s_pt = [], [], []
    for i in range(len(t.sim_eta)):
        if t.sim_pt[i] > args.sim_ptcut and (args.sim_keep_neutral or t.sim_q[i] != 0) \
                and in_win(t.sim_eta[i], t.sim_phi[i]):
            s_eta.append(t.sim_eta[i]); s_phi.append(t.sim_phi[i]); s_pt.append(t.sim_pt[i])

    # Object collection in window, split genuine / fake.
    o_eta = getattr(t, f"{args.object}_eta")
    o_phi = getattr(t, f"{args.object}_phi")
    o_pt = getattr(t, f"{args.object}_pt")
    o_fake = getattr(t, f"{args.object}_isFake")
    gen = {"eta": [], "phi": [], "pt": []}
    fak = {"eta": [], "phi": [], "pt": []}
    for i in range(len(o_eta)):
        if not in_win(o_eta[i], o_phi[i]):
            continue
        d = gen if int(o_fake[i]) == 0 else fak
        d["eta"].append(o_eta[i]); d["phi"].append(o_phi[i]); d["pt"].append(o_pt[i])
    n_gen, n_fak = len(gen["eta"]), len(fak["eta"])

    fig, ax = plt.subplots(figsize=(11, 7))
    # sim tracks (background)
    ax.scatter(s_eta, s_phi, s=sizes(s_pt), c="#b8bdc2", alpha=0.55, edgecolors="none",
               zorder=1, label=f"reconstructable sim (n={len(s_eta)})")
    # fake objects (× markers, behind genuine)
    if n_fak:
        ax.scatter(fak["eta"], fak["phi"], s=34, c="k", alpha=0.5, marker="x",
                   linewidths=0.9, zorder=5, label=f"{label} fake (n={n_fak})")
    # genuine objects (filled; drawn under the fake × so both are visible where they overlap)
    if n_gen:
        ax.scatter(gen["eta"], gen["phi"], s=sizes(gen["pt"]), c=color, alpha=0.9,
                   edgecolors="k", linewidths=0.3, zorder=3,
                   label=f"{label} genuine (n={n_gen})")
    # jet axes
    for je, jp in jets.values():
        if in_win(je, jp):
            ax.plot(je, jp, marker="*", markerfacecolor="none", markeredgecolor="#1b9e77",
                    markersize=26, markeredgewidth=1.8, linestyle="none", zorder=6)
    ax.plot([], [], marker="*", markerfacecolor="none", markeredgecolor="#1b9e77",
            markersize=14, markeredgewidth=1.5, linestyle="none", label="GenJet axis")

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\eta$"); ax.set_ylabel(r"$\phi$ [rad]")
    ax.set_title(f"event {evt}: lower {label} objects over sim tracks "
                 f"({n_gen} genuine, {n_fak} fake in view). Size ∝ log(pT).")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9, fontsize=9)
    fig.tight_layout()

    out = args.output or f"obj_{args.object}_over_sim_evt{evt}.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}  ({label}: {n_gen} genuine, {n_fak} fake in view; "
          f"{len(s_eta)} sim)")


if __name__ == "__main__":
    main()

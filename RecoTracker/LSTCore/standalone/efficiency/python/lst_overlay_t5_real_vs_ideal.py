#!/bin/env python
"""
Overlay T5_lower efficiency vs deltaR for a real-pLS vs ideal-pLS run that differ
ONLY in --idealpls (same build, same input, same events). T5 is built from Outer
Tracker hits only, so T5_lower should be identical between the two. pLS_lower is
shown as a control: it MUST differ, confirming the pLS swap actually happened.
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ROOT as r
r.gROOT.SetBatch(True)

def eff_graph(infile, stage, sel="base", pdgid=0, charge=0):
    numer = infile.Get(f"Root__{stage}_{sel}_{pdgid}_{charge}_ef_numer_deltaR")
    denom = infile.Get(f"Root__{stage}_{sel}_{pdgid}_{charge}_ef_denom_deltaR")
    if not numer or not denom:
        return None
    teff = r.TEfficiency(numer, denom)
    g = teff.CreateGraph()
    n = g.GetN()
    x = [g.GetPointX(i) for i in range(n)]
    y = [g.GetPointY(i) for i in range(n)]
    eyl = [g.GetErrorYlow(i) for i in range(n)]
    eyh = [g.GetErrorYhigh(i) for i in range(n)]
    # raw integer counts for exact-equality check
    nb = numer.GetNbinsX()
    num_counts = [numer.GetBinContent(i) for i in range(1, nb + 1)]
    den_counts = [denom.GetBinContent(i) for i in range(1, nb + 1)]
    return x, y, [eyl, eyh], num_counts, den_counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="LSTNumDen_realpls_jet_5evt.root")
    ap.add_argument("--ideal", default="LSTNumDen_idealpls_jet_5evt.root")
    ap.add_argument("--zoom", type=float, default=0.05)
    ap.add_argument("-o", "--output", default="t5_real_vs_ideal_overlay.png")
    args = ap.parse_args()

    fr = r.TFile.Open(args.real)
    fi = r.TFile.Open(args.ideal)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, stage, title in [
        (axes[0], "T5_lower", "T5 (should be identical — OT-only, no pLS input)"),
        (axes[1], "pLS_lower", "pLS (control — must differ: real vs ideal seeds)"),
    ]:
        gr = eff_graph(fr, stage)
        gi = eff_graph(fi, stage)
        if gr:
            ax.errorbar(gr[0], gr[1], yerr=gr[2], fmt="o", color="#d62728",
                        markersize=5, capsize=0, label="real pLS")
        if gi:
            ax.errorbar(gi[0], gi[1], yerr=gi[2], fmt="s", color="#1f77b4",
                        markersize=4, capsize=0, label="ideal pLS", alpha=0.8)
        ax.set_xlim(0, args.zoom)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("ΔR (track — nearest GenJet)")
        ax.set_ylabel("Efficiency")
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")

        # numeric equality report
        if gr and gi:
            same_num = gr[3] == gi[3]
            same_den = gr[4] == gi[4]
            print(f"[{stage}] numer identical={same_num}  denom identical={same_den}"
                  f"  (sum numer real={sum(gr[3]):.0f} ideal={sum(gi[3]):.0f};"
                  f" denom real={sum(gr[4]):.0f} ideal={sum(gi[4]):.0f})")

    fig.suptitle("T5_lower efficiency vs ΔR: real vs ideal pLS (same build/input/events, 5 evt)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()

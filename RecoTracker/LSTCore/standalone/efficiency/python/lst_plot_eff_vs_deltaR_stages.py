#!/bin/env python

import argparse
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ROOT as r

r.gROOT.SetBatch(True)

# Stages listed from lowest-level to highest-level reconstruction object.
# Stages missing from the input file are silently skipped.
DEFAULT_STAGES = "MD_lower,LS_lower,pLS_lower,T3_lower,pT3_lower,T5_lower,pT5_lower,TC"
SUB_STAGES = "pLS_lower,T5_lower,pT5_lower,TC"

# Colors follow matplotlib's tab10 palette; TC is always black.
# Softer assignments that mirror the per-type breakdown in lst_plot_performance.py.
_STAGE_STYLE = {
    "TC":        dict(color="black",    marker="o",  lw=2.0, ms=4),
    "pT5_lower": dict(color="#1f77b4",  marker="s",  lw=1.2, ms=3),   # tab:blue
    "pT5":       dict(color="#1f77b4",  marker="s",  lw=1.2, ms=3),
    "T5_lower":  dict(color="#ff7f0e",  marker="^",  lw=1.2, ms=3),   # tab:orange
    "T5":        dict(color="#ff7f0e",  marker="^",  lw=1.2, ms=3),
    "pT3_lower": dict(color="#2ca02c",  marker="v",  lw=1.2, ms=3),   # tab:green
    "pT3":       dict(color="#2ca02c",  marker="v",  lw=1.2, ms=3),
    "T3_lower":  dict(color="#d62728",  marker="D",  lw=1.2, ms=3),   # tab:red
    "T3":        dict(color="#d62728",  marker="D",  lw=1.2, ms=3),
    "pLS_lower": dict(color="#9467bd",  marker="P",  lw=1.2, ms=3),   # tab:purple
    "pLS":       dict(color="#9467bd",  marker="P",  lw=1.2, ms=3),
    "LS_lower":  dict(color="#8c564b",  marker="X",  lw=1.2, ms=3),   # tab:brown
    "LS":        dict(color="#8c564b",  marker="X",  lw=1.2, ms=3),
    "MD_lower":  dict(color="#7f7f7f",  marker="*",  lw=1.0, ms=4),   # tab:gray
    "MD":        dict(color="#7f7f7f",  marker="*",  lw=1.0, ms=4),
}
_FALLBACK_COLORS = ["#17becf", "#bcbd22", "#e377c2", "#aec7e8"]

parser = argparse.ArgumentParser(
    description="Plot efficiency vs deltaR (track-to-nearest-GenJet), broken down by "
    "reconstruction stage. Reads the standard "
    "Root__<OBJ>_<SEL>_<PDGID>_<CHARGE>_ef_numer/denom_deltaR histograms produced by "
    "createPerfNumDenHists (see lst_plot_performance.py for the naming convention). "
    "Stages absent from the input file are silently skipped."
)
parser.add_argument("input", help="input LSTNumDen*.root file")
parser.add_argument("--stages", default=DEFAULT_STAGES, help="comma-separated stage/object names [DEFAULT={}]".format(DEFAULT_STAGES))
parser.add_argument("--sub", action="store_true", help="use the subset stages ({}) instead of the full default set".format(SUB_STAGES))
parser.add_argument("--labels", default=None, help="comma-separated legend labels, one per stage [DEFAULT: stage name with '_lower' suffix stripped]")
parser.add_argument("--selection", default="base", help="selection name component of the histogram [DEFAULT=base]")
parser.add_argument("--pdgid", type=int, default=0, help="pdgid component of the histogram, 0=inclusive [DEFAULT=0]")
parser.add_argument("--charge", type=int, default=0, help="charge component of the histogram, 0=inclusive [DEFAULT=0]")
parser.add_argument("--tag", default=None, help="run tag for the title [DEFAULT: short git hash of HEAD]")
parser.add_argument("--nevents", default=None, help="event count for the title [DEFAULT: parsed from input filename if it matches *NEVT<n>*, else blank]")
parser.add_argument("--zoom", type=float, default=0.05, help="deltaR upper limit for the plot [DEFAULT=0.05]")
parser.add_argument("-o", "--output", default="eff_vs_deltaR_stages_recreated.png", help="output image path [DEFAULT=eff_vs_deltaR_stages_recreated.png]")


def default_tag():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def default_nevents(input_path):
    m = re.search(r"NEVT(-?\d+)", os.path.basename(input_path))
    return m.group(1) if m else ""


def efficiency_graph(infile, stage, selection, pdgid, charge):
    numer = infile.Get("Root__{}_{}_{}_{}_ef_numer_deltaR".format(stage, selection, pdgid, charge))
    denom = infile.Get("Root__{}_{}_{}_{}_ef_denom_deltaR".format(stage, selection, pdgid, charge))
    if not numer or not denom:
        return None
    teff = r.TEfficiency(numer, denom)
    graph = teff.CreateGraph()
    n = graph.GetN()
    x = [graph.GetPointX(i) for i in range(n)]
    y = [graph.GetPointY(i) for i in range(n)]
    eyl = [graph.GetErrorYlow(i) for i in range(n)]
    eyh = [graph.GetErrorYhigh(i) for i in range(n)]
    return x, y, [eyl, eyh]


def main():
    args = parser.parse_args()

    stages = [s.strip() for s in (SUB_STAGES if args.sub else args.stages).split(",")]
    labels = [l.strip() for l in args.labels.split(",")] if args.labels else [re.sub(r"_lower$", "", s) for s in stages]
    if len(labels) != len(stages):
        parser.error("--labels must have the same number of entries as --stages")

    tag = args.tag if args.tag is not None else default_tag()
    nevents = args.nevents if args.nevents is not None else default_nevents(args.input)

    infile = r.TFile.Open(args.input)
    if not infile or infile.IsZombie():
        parser.error("Could not open input file: {}".format(args.input))

    fig, ax = plt.subplots(figsize=(7, 5))

    fallback_idx = 0
    plotted = 0
    for stage, label in zip(stages, labels):
        result = efficiency_graph(infile, stage, args.selection, args.pdgid, args.charge)
        if result is None:
            print("WARNING: skipping stage '{}' — histogram not found in {}".format(stage, args.input), file=sys.stderr)
            continue
        x, y, yerr = result
        style = _STAGE_STYLE.get(stage)
        if style is None:
            color = _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]
            fallback_idx += 1
            style = dict(color=color, marker="o", lw=1.2, ms=3)
        ax.errorbar(x, y, yerr=yerr, fmt=style["marker"],
                    color=style["color"],
                    markersize=style["ms"], capsize=0, label=label)
        plotted += 1

    if plotted == 0:
        print("ERROR: no stages found in input file", file=sys.stderr)
        sys.exit(1)

    ax.set_xlim(0, args.zoom)
    ax.set_xlabel("ΔR (track — nearest GenJet)")
    ax.set_ylabel("Efficiency")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    title = "LST Efficiency vs ΔR by Stage ({}{})".format(tag, ", {} events".format(nevents) if nevents else "")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print("Wrote {}".format(args.output))


if __name__ == "__main__":
    main()

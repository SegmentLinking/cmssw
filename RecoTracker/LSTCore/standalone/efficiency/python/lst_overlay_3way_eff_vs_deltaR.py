#!/bin/env python

# Overlay efficiency-vs-deltaR-by-stage curves from THREE LSTNumDen files on one axis.
# Reuses the per-stage color/marker convention from lst_overlay_eff_vs_deltaR_stages.py.
# The three datasets are distinguished by marker fill + line style:
#   dataset A -> filled markers,      solid line
#   dataset B -> open markers,        dashed line
#   dataset C -> half-filled markers, dotted line

import argparse
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import ROOT as r

r.gROOT.SetBatch(True)

DEFAULT_STAGES = "MD_lower,LS_lower,pLS_lower,T3_lower,pT3_lower,T5_lower,pT5_lower,TC"
SUB_STAGES = "pLS_lower,T5_lower,pT5_lower,TC"

_STAGE_STYLE = {
    "TC":        dict(color="black",    marker="o",  ms=4),
    "pT5_lower": dict(color="#1f77b4",  marker="s",  ms=3),
    "pT5":       dict(color="#1f77b4",  marker="s",  ms=3),
    "T5_lower":  dict(color="#ff7f0e",  marker="^",  ms=3),
    "T5":        dict(color="#ff7f0e",  marker="^",  ms=3),
    "pT3_lower": dict(color="#2ca02c",  marker="v",  ms=3),
    "pT3":       dict(color="#2ca02c",  marker="v",  ms=3),
    "T3_lower":  dict(color="#d62728",  marker="D",  ms=3),
    "T3":        dict(color="#d62728",  marker="D",  ms=3),
    "pLS_lower": dict(color="#9467bd",  marker="P",  ms=3),
    "pLS":       dict(color="#9467bd",  marker="P",  ms=3),
    "LS_lower":  dict(color="#8c564b",  marker="X",  ms=3),
    "LS":        dict(color="#8c564b",  marker="X",  ms=3),
    "MD_lower":  dict(color="#7f7f7f",  marker="*",  ms=4),
    "MD":        dict(color="#7f7f7f",  marker="*",  ms=4),
}
_FALLBACK_COLORS = ["#17becf", "#bcbd22", "#e377c2", "#aec7e8"]

_DATASET_KW = [
    dict(fillstyle="full",   linestyle="-"),
    dict(fillstyle="none",   linestyle="--"),
    dict(fillstyle="bottom", linestyle=":"),
]

parser = argparse.ArgumentParser(
    description="Overlay efficiency-vs-deltaR-by-stage from three LSTNumDen files. "
    "A = filled/solid, B = open/dashed, C = half-filled/dotted."
)
parser.add_argument("input_a", help="first LSTNumDen*.root file")
parser.add_argument("input_b", help="second LSTNumDen*.root file")
parser.add_argument("input_c", help="third LSTNumDen*.root file")
parser.add_argument("--labels", default="A,B,C", help="comma-separated dataset labels for A,B,C [DEFAULT=A,B,C]")
parser.add_argument("--stages", default=DEFAULT_STAGES, help="comma-separated stage/object names [DEFAULT={}]".format(DEFAULT_STAGES))
parser.add_argument("--sub", action="store_true", help="use the subset stages ({}) instead of the full default set".format(SUB_STAGES))
parser.add_argument("--selection", default="base", help="selection name component of the histogram [DEFAULT=base]")
parser.add_argument("--pdgid", type=int, default=0, help="pdgid component of the histogram, 0=inclusive [DEFAULT=0]")
parser.add_argument("--charge", type=int, default=0, help="charge component of the histogram, 0=inclusive [DEFAULT=0]")
parser.add_argument("--tag", default=None, help="run tag for the title [DEFAULT: short git hash of HEAD]")
parser.add_argument("--zoom", type=float, default=0.05, help="deltaR upper limit for the plot [DEFAULT=0.05]")
parser.add_argument("-o", "--output", default="eff_vs_deltaR_stages_3way.png", help="output image path")


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


def stage_style(stage, fallback):
    style = _STAGE_STYLE.get(stage)
    if style is None:
        color = _FALLBACK_COLORS[fallback[0] % len(_FALLBACK_COLORS)]
        fallback[0] += 1
        style = dict(color=color, marker="o", ms=3)
    return style


def main():
    args = parser.parse_args()

    stages = [s.strip() for s in (SUB_STAGES if args.sub else args.stages).split(",")]
    labels = [l.strip() for l in args.labels.split(",")]
    if len(labels) != 3:
        parser.error("--labels must have exactly three entries (for input_a,input_b,input_c)")
    tag = args.tag if args.tag is not None else default_tag()

    files = []
    for path in (args.input_a, args.input_b, args.input_c):
        f = r.TFile.Open(path)
        if not f or f.IsZombie():
            parser.error("Could not open input file: {}".format(path))
        files.append(f)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fallback = [0]
    for stage in stages:
        style = stage_style(stage, fallback)
        for infile, dkw in zip(files, _DATASET_KW):
            result = efficiency_graph(infile, stage, args.selection, args.pdgid, args.charge)
            if result is None:
                print("WARNING: skipping stage '{}' — not found in a file".format(stage), file=sys.stderr)
                continue
            x, y, yerr = result
            ax.errorbar(
                x, y, yerr=yerr, marker=style["marker"], color=style["color"],
                markersize=style["ms"], capsize=0, linestyle=dkw["linestyle"],
                lw=1.0, fillstyle=dkw["fillstyle"],
                markerfacecolor=(style["color"] if dkw["fillstyle"] == "full" else "none"),
            )

    ax.set_xlim(0, args.zoom)
    ax.set_xlabel("ΔR (track — nearest GenJet)")
    ax.set_ylabel("Efficiency")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Legend 1: stage colors/markers
    stage_handles = [
        Line2D([0], [0], color=stage_style(s, [0])["color"], marker=stage_style(s, [0])["marker"],
               linestyle="none", markersize=5, label=re.sub(r"_lower$", "", s))
        for s in stages
    ]
    leg1 = ax.legend(handles=stage_handles, fontsize=8, loc="lower right", title="Stage")
    ax.add_artist(leg1)

    # Legend 2: dataset line styles
    ds_handles = [
        Line2D([0], [0], color="gray", marker="o", linestyle="-",  markerfacecolor="gray",  markersize=5, label=labels[0]),
        Line2D([0], [0], color="gray", marker="o", linestyle="--", markerfacecolor="none",  markersize=5, label=labels[1]),
        Line2D([0], [0], color="gray", marker="o", linestyle=":",  markerfacecolor="none",  markersize=5, label=labels[2]),
    ]
    ax.legend(handles=ds_handles, fontsize=8, loc="upper right", title="Dataset")

    ax.set_title("LST Efficiency vs ΔR — {} / {} / {} ({})".format(labels[0], labels[1], labels[2], tag))
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print("Wrote {}".format(args.output))


if __name__ == "__main__":
    main()

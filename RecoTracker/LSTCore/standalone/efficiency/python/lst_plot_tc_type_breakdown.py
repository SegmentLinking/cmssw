#!/bin/env python
"""
Plot TC efficiency vs ΔR with optional overlay of TC subsets by object type.

TC is always shown. Each subset (pT5, T5, pT3, T4, pLS) shares the same
sim-track denominator as TC but counts only tracks whose best matched TC is
of the specified type. Histograms must be present in the NumDen file
(produced by createPerfNumDenHists with the -J flag).

Usage:
    python3 lst_plot_tc_type_breakdown.py LSTNumDen.root --pt5
    python3 lst_plot_tc_type_breakdown.py LSTNumDen.root --pt5 --t5 --pt3
    python3 lst_plot_tc_type_breakdown.py LSTNumDen.root --pt5 --t5 -o my_plot.png
"""

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

# CLI flag -> (histogram object-type prefix, display label)
_TYPES = {
    "pt5": ("pT5", "TC (pT5)"),
    "t5":  ("T5",  "TC (T5)"),
    "pt3": ("pT3", "TC (pT3)"),
    "t4":  ("T4",  "TC (T4)"),
    "pls": ("pLS", "TC (pLS)"),
}

# Colors/markers mirror the convention used in lst_plot_eff_vs_deltaR_stages.py
_TYPE_STYLE = {
    "pT5": dict(color="#1f77b4", marker="s", lw=1.5, ms=4),  # tab:blue
    "T5":  dict(color="#ff7f0e", marker="^", lw=1.5, ms=4),  # tab:orange
    "pT3": dict(color="#2ca02c", marker="v", lw=1.5, ms=4),  # tab:green
    "T4":  dict(color="#d62728", marker="D", lw=1.5, ms=4),  # tab:red
    "pLS": dict(color="#9467bd", marker="P", lw=1.5, ms=4),  # tab:purple
}
_TC_STYLE = dict(color="black", marker="o", lw=2.0, ms=5)


def _default_tag():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _default_nevents(path):
    m = re.search(r"NEVT(-?\d+)", os.path.basename(path))
    return m.group(1) if m else ""


def efficiency_graph(infile, prefix, selection, pdgid, charge):
    """Return (x, y, [eyl, eyh]) from numer/denom deltaR histograms, or None if missing."""
    numer = infile.Get("Root__{}_{}_{}_{}_ef_numer_deltaR".format(prefix, selection, pdgid, charge))
    denom = infile.Get("Root__{}_{}_{}_{}_ef_denom_deltaR".format(prefix, selection, pdgid, charge))
    if not numer or not denom:
        return None
    teff = r.TEfficiency(numer, denom)
    graph = teff.CreateGraph()
    n = graph.GetN()
    x   = [graph.GetPointX(i)     for i in range(n)]
    y   = [graph.GetPointY(i)     for i in range(n)]
    eyl = [graph.GetErrorYlow(i)  for i in range(n)]
    eyh = [graph.GetErrorYhigh(i) for i in range(n)]
    return x, y, [eyl, eyh]


parser = argparse.ArgumentParser(
    description="Plot TC efficiency vs ΔR (track-to-nearest-GenJet) with optional "
    "overlay of TC subsets broken down by object type. TC is always shown. "
    "Subsets share the same denominator as TC; the numerator counts only "
    "sim tracks whose matched TC is of the requested type.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("input", help="LSTNumDen*.root file produced by createPerfNumDenHists")
parser.add_argument("--pt5", action="store_true", help="overlay pT5-type TC subset")
parser.add_argument("--t5",  action="store_true", help="overlay T5-type TC subset")
parser.add_argument("--pt3", action="store_true", help="overlay pT3-type TC subset")
parser.add_argument("--t4",  action="store_true", help="overlay T4-type TC subset")
parser.add_argument("--pls", action="store_true", help="overlay pLS-type TC subset")
parser.add_argument("--selection", default="base",
                    help="selection name in histogram key")
parser.add_argument("--pdgid", type=int, default=0,
                    help="pdgid in histogram key (0=all)")
parser.add_argument("--charge", type=int, default=0,
                    help="charge in histogram key (0=both, 1=positive, -1=negative)")
parser.add_argument("--zoom", type=float, default=0.05,
                    help="ΔR upper limit for the x-axis")
parser.add_argument("--tag", default=None,
                    help="run tag for the plot title (default: git HEAD hash)")
parser.add_argument("--nevents", default=None,
                    help="event count for the plot title")
parser.add_argument("-o", "--output", default="eff_vs_deltaR_tc_breakdown.png",
                    help="output image path")


def main():
    args = parser.parse_args()

    requested = [flag for flag in ("pt5", "t5", "pt3", "t4", "pls") if getattr(args, flag)]
    if not requested:
        print(
            "Note: no subset flags given; plotting TC only. "
            "Add --pt5, --t5, --pt3, --t4, or --pls to overlay subsets.",
            file=sys.stderr,
        )

    tag     = args.tag     if args.tag     is not None else _default_tag()
    nevents = args.nevents if args.nevents is not None else _default_nevents(args.input)

    infile = r.TFile.Open(args.input)
    if not infile or infile.IsZombie():
        parser.error("Could not open: {}".format(args.input))

    fig, ax = plt.subplots(figsize=(7, 5))

    # Overall TC (always plotted)
    result = efficiency_graph(infile, "TC", args.selection, args.pdgid, args.charge)
    if result is None:
        print("ERROR: TC histograms not found in {}".format(args.input), file=sys.stderr)
        sys.exit(1)
    x, y, yerr = result
    ax.errorbar(
        x, y, yerr=yerr,
        fmt=_TC_STYLE["marker"], color=_TC_STYLE["color"], lw=_TC_STYLE["lw"],
        markersize=_TC_STYLE["ms"], capsize=0, label="TC (all types)", zorder=3,
    )

    # Requested subsets
    for flag in requested:
        prefix, label = _TYPES[flag]
        result = efficiency_graph(infile, prefix, args.selection, args.pdgid, args.charge)
        if result is None:
            print(
                "WARNING: {} histograms not found in {} — skipping".format(prefix, args.input),
                file=sys.stderr,
            )
            continue
        x, y, yerr = result
        st = _TYPE_STYLE[prefix]
        ax.errorbar(
            x, y, yerr=yerr,
            fmt=st["marker"], color=st["color"], lw=st["lw"],
            markersize=st["ms"], capsize=0, label=label,
        )

    ax.set_xlim(0, args.zoom)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("ΔR (track — nearest GenJet)")
    ax.set_ylabel("Efficiency")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    evt_str = ", {} events".format(nevents) if nevents else ""
    ax.set_title("TC Efficiency vs ΔR by Object Type ({}{})".format(tag, evt_str))
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print("Wrote {}".format(args.output))


if __name__ == "__main__":
    main()

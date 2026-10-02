#!/usr/bin/env python3
"""
lst_compare_pterr_sampling_eff.py

Compare efficiency vs ΔR for all reconstruction stages between two NumDen files.
Intended use: compare ideal-pLS runs with median vs random ptErr/etaErr sampling
in TruthPixelSeeds.cc.

Usage:
  python3 lst_compare_pterr_sampling_eff.py <file_a.root> <file_b.root> \
      [--label-a "Median etaErr"] [--label-b "Random etaErr"] \
      [--title "..."] [--out prefix] [--zoom 0.05]

Produces two plots:
  <prefix>_all.png    — all 8 stages
  <prefix>_sub.png    — pLS, T5, pT5 only
"""

import argparse

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import beta as beta_dist

ALL_STAGES = [
    ("MD_lower",  "#7f7f7f", "*", 4),
    ("LS_lower",  "#8c564b", "X", 3),
    ("pLS_lower", "#9467bd", "P", 3),
    ("T3_lower",  "#d62728", "D", 3),
    ("pT3_lower", "#2ca02c", "v", 3),
    ("T5_lower",  "#ff7f0e", "^", 3),
    ("pT5_lower", "#1f77b4", "s", 3),
    ("TC",        "black",   "o", 4),
]
SUB_STAGES = {"pLS_lower", "T5_lower", "pT5_lower"}

LABELS = {s: s.replace("_lower", "") for s, *_ in ALL_STAGES}


def clopper_pearson(k, n, cl=0.683):
    a = 1.0 - cl
    lo = np.where(k > 0, beta_dist.ppf(a / 2,     k,     n - k + 1), 0.0)
    hi = np.where(k < n, beta_dist.ppf(1 - a / 2, k + 1, n - k),     1.0)
    return lo, hi


def load_eff(path, selection="base", pdgid=0, charge=0):
    f = uproot.open(path)
    results = {}
    for stage, *_ in ALL_STAGES:
        nk = f"Root__{stage}_{selection}_{pdgid}_{charge}_ef_numer_deltaR"
        dk = f"Root__{stage}_{selection}_{pdgid}_{charge}_ef_denom_deltaR"
        if nk not in f or dk not in f:
            continue
        n_hist = f[nk]
        d_hist = f[dk]
        n_vals = n_hist.values()
        d_vals = d_hist.values()
        edges   = n_hist.axis().edges()
        centers = 0.5 * (edges[:-1] + edges[1:])
        eff = np.where(d_vals > 0, n_vals / d_vals, np.nan)
        lo, hi = clopper_pearson(n_vals, d_vals)
        results[stage] = (centers, eff, lo, hi)
    return results


def make_plot(data_a, data_b, label_a, label_b, title, out_path, zoom, subset=False):
    stages = [(s, c, m, ms) for s, c, m, ms in ALL_STAGES
              if (not subset or s in SUB_STAGES)]

    fig, ax = plt.subplots(figsize=(8, 5))

    stage_handles = []
    for stage, color, marker, ms in stages:
        label = LABELS[stage]
        plotted = False
        if stage in data_a:
            x, eff, lo, hi = data_a[stage]
            mask = x <= zoom
            h, = ax.plot(x[mask], eff[mask], marker=marker, color=color,
                         markersize=ms, linestyle="-", linewidth=1.2)
            ax.fill_between(x[mask], lo[mask], hi[mask], color=color, alpha=0.12)
            stage_handles.append((h, label))
            plotted = True
        if stage in data_b:
            x, eff, lo, hi = data_b[stage]
            mask = x <= zoom
            ax.plot(x[mask], eff[mask], marker=marker, color=color,
                    markersize=ms, linestyle="--", linewidth=1.2, alpha=0.75)
            ax.fill_between(x[mask], lo[mask], hi[mask], color=color, alpha=0.06)
            if not plotted:
                stage_handles.append((Line2D([0],[0],color=color,marker=marker,
                                             markersize=ms,linestyle="-"), label))

    # Style legend (solid vs dashed) — placed inside upper-left
    style_handles = [
        Line2D([0], [0], color="gray", linestyle="-",  linewidth=1.5, label=label_a),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5, label=label_b),
    ]
    style_leg = ax.legend(handles=style_handles, fontsize=8, loc="upper left")
    ax.add_artist(style_leg)

    # Stage legend — placed to the right of the axes
    ncol = 1 if subset else 2
    ax.legend(
        [h for h, _ in stage_handles],
        [l for _, l in stage_handles],
        fontsize=8, title="Stage", ncol=ncol,
        loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0.
    )

    ax.set_xlim(0, zoom)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("ΔR (track — nearest GenJet)")
    ax.set_ylabel("Efficiency")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_a", help="First NumDen file (solid lines)")
    parser.add_argument("file_b", help="Second NumDen file (dashed lines)")
    parser.add_argument("--label-a", default="Median etaErr / ptErr",
                        help="Legend label for file A")
    parser.add_argument("--label-b", default="Random etaErr / ptErr",
                        help="Legend label for file B")
    parser.add_argument("--title",
                        default="LST Efficiency vs ΔR — etaErr/ptErr sampling comparison\n"
                                "(100 events, ideal pLS, minNHitsForDup=8)",
                        help="Plot title")
    parser.add_argument("--out", default="eff_compare_etaerr_sampling",
                        help="Output path prefix (suffixes _all.png and _sub.png added)")
    parser.add_argument("--zoom", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Loading {args.file_a} ...")
    data_a = load_eff(args.file_a)
    print(f"Loading {args.file_b} ...")
    data_b = load_eff(args.file_b)

    make_plot(data_a, data_b, args.label_a, args.label_b, args.title,
              args.out + "_all.png", args.zoom, subset=False)
    make_plot(data_a, data_b, args.label_a, args.label_b, args.title,
              args.out + "_sub.png", args.zoom, subset=True)


if __name__ == "__main__":
    main()

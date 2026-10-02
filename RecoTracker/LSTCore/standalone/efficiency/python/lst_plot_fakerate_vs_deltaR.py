#!/usr/bin/env python3
"""
lst_plot_fakerate_vs_deltaR.py

Plots fake rate vs ΔR for all LST reconstruction stages, matching the style of
eff_vs_deltaR_idealpls_fixed.png.

Reads pre-computed histograms from a LSTNumDen ROOT file:
  Root__<OBJ>_fr_numer_deltaR  (fake reconstructed objects)
  Root__<OBJ>_fr_denom_deltaR  (all reconstructed objects)

Object hierarchy (uses _lower histograms for sub-TC objects):
  MD_lower → LS_lower → pLS_lower → T3_lower → pT3_lower → T5_lower → pT5_lower → TC

Usage:
  python3 lst_plot_fakerate_vs_deltaR.py <numden.root> [--out fakerate_vs_deltaR.png] [--zoom 0.05]
"""

import sys
import argparse
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

# Color/marker scheme matches lst_rescue_pt5priority.py / lst_plot_eff_vs_deltaR_stages.py
# (label, numden_key_suffix, color, marker, markersize)
OBJECTS = [
    ("MD",  "MD_lower",  "#7f7f7f", "*", 4),
    ("LS",  "LS_lower",  "#8c564b", "X", 3),
    ("pLS", "pLS_lower", "#9467bd", "P", 3),
    ("T3",  "T3_lower",  "#d62728", "D", 3),
    ("pT3", "pT3_lower", "#2ca02c", "v", 3),
    ("T5",  "T5_lower",  "#ff7f0e", "^", 3),
    ("pT5", "pT5_lower", "#1f77b4", "s", 3),
    ("TC",  "TC",        "black",   "o", 4),
]


def clopper_pearson(k, n, cl=0.683):
    alpha = 1.0 - cl
    lo = np.where(k > 0, beta_dist.ppf(alpha / 2,     k,     n - k + 1), 0.0)
    hi = np.where(k < n, beta_dist.ppf(1 - alpha / 2, k + 1, n - k),     1.0)
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("numden", help="LSTNumDen ROOT file")
    parser.add_argument("--out",  default="fakerate_vs_deltaR.png")
    parser.add_argument("--zoom", type=float, default=0.05)
    args = parser.parse_args()

    f = uproot.open(args.numden)
    all_keys = set(f.keys())

    fig, ax = plt.subplots(figsize=(7, 5))

    for label, key, color, marker, ms in OBJECTS:
        numer_key = f"Root__{key}_fr_numer_deltaR;1"
        denom_key = f"Root__{key}_fr_denom_deltaR;1"
        # uproot may or may not include the cycle suffix; try both
        if numer_key not in all_keys:
            numer_key = f"Root__{key}_fr_numer_deltaR"
            denom_key = f"Root__{key}_fr_denom_deltaR"
        if numer_key not in all_keys:
            print(f"  skip {label} ({key}) — key absent")
            continue

        numer   = f[numer_key].values().astype(float)
        denom   = f[denom_key].values().astype(float)
        edges   = f[numer_key].axis().edges()
        centers = 0.5 * (edges[:-1] + edges[1:])

        fr     = np.where(denom > 0, numer / denom, np.nan)
        lo, hi = clopper_pearson(numer, denom)
        yerr   = [np.where(np.isnan(fr), 0, fr - lo),
                  np.where(np.isnan(fr), 0, hi - fr)]

        ax.errorbar(centers, fr, yerr=yerr,
                    fmt=marker, color=color, markersize=ms, capsize=0,
                    linestyle="none", label=label)
        print(f"  plotted {label} ({key}): {int(np.nansum(numer))} fakes / {int(np.nansum(denom))} total")

    ax.set_xlim(0, args.zoom)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("ΔR (track — nearest GenJet)")
    ax.set_ylabel("Fake Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"LST Fake Rate vs ΔR — baseline\n({args.numden.split('/')[-1]})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()

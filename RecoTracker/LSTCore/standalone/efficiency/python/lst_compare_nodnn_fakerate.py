#!/usr/bin/env python3
"""
lst_compare_nodnn_fakerate.py

Compare fake rate vs ΔR for baseline vs DNN-disabled runs using pre-computed NumDen files.
Plots TC and pT5 fake rates side-by-side.

Usage:
  python3 lst_compare_nodnn_fakerate.py <baseline_numden.root> <nodnn_numden.root> [--out out.png] [--zoom 0.05]
"""

import argparse
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist


def clopper_pearson(k, n, cl=0.683):
    alpha = 1.0 - cl
    lo = np.where(k > 0, beta_dist.ppf(alpha / 2,     k,     n - k + 1), 0.0)
    hi = np.where(k < n, beta_dist.ppf(1 - alpha / 2, k + 1, n - k),     1.0)
    return lo, hi


def load_fr(f, obj):
    num = f[f"Root__{obj}_fr_numer_deltaR"].values().astype(float)
    den = f[f"Root__{obj}_fr_denom_deltaR"].values().astype(float)
    edges = f[f"Root__{obj}_fr_denom_deltaR"].axis().edges()
    centers = 0.5 * (edges[:-1] + edges[1:])
    fr = np.where(den > 0, num / den, np.nan)
    lo, hi = clopper_pearson(num, den)
    return centers, fr, lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", help="Baseline NumDen root file (DNN active)")
    parser.add_argument("nodnn",    help="DNN-disabled NumDen root file (--nopt5dnn)")
    parser.add_argument("--out",  default="fakerate_vs_deltaR_nodnn_compare.png")
    parser.add_argument("--zoom", type=float, default=0.05)
    args = parser.parse_args()

    fb = uproot.open(args.baseline)
    fn = uproot.open(args.nodnn)

    OBJECTS = [("TC", "TC"), ("pT5 (all objects)", "pT5_lower")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

    BASE_COLOR = "#1f77b4"
    NODNN_COLOR = "#e75480"

    for ax, (label, obj) in zip(axes, OBJECTS):
        dr_b, fr_b, lo_b, hi_b = load_fr(fb, obj)
        dr_n, fr_n, lo_n, hi_n = load_fr(fn, obj)

        mask_b = dr_b <= args.zoom
        mask_n = dr_n <= args.zoom

        ax.errorbar(dr_b[mask_b], fr_b[mask_b],
                    yerr=[fr_b[mask_b] - lo_b[mask_b], hi_b[mask_b] - fr_b[mask_b]],
                    fmt="o", color=BASE_COLOR, markersize=5,
                    capsize=0, linestyle="none", label="Baseline (DNN active)")

        ax.errorbar(dr_n[mask_n], fr_n[mask_n],
                    yerr=[fr_n[mask_n] - lo_n[mask_n], hi_n[mask_n] - fr_n[mask_n]],
                    fmt="s", color=NODNN_COLOR, markersize=5,
                    capsize=0, linestyle="none", label="DNN disabled (--nopt5dnn)")

        ax.set_xlabel("ΔR (track — nearest GenJet)")
        ax.set_ylabel("Fake Rate")
        ax.set_title(f"{label} Fake Rate vs ΔR")
        ax.set_xlim(0, args.zoom)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("DNN gate impact on fake rate (100 events, idealpls)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()

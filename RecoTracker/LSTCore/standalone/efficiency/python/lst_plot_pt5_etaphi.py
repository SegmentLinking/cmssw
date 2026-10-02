#!/usr/bin/env python3
"""
lst_plot_pt5_etaphi.py

Plots η-φ distributions of pT5s split into four categories:
  (1) genuine + survived   (2) fake + survived
  (3) genuine + killed     (4) fake + killed

genuine = pT5_simIdx >= 0   (matched sim track with >75% hit purity)
killed  = pT5_isDupReco == 1 (removed by RemoveDupPixelQuintupletsFromMap)

Usage:
  python3 lst_plot_pt5_etaphi.py <ntuple.root> [--nevents N] [--out FILE]
"""

import argparse
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def load(fname, nevents=-1):
    tree = uproot.open(fname)["tree"]
    branches = ["pT5_eta", "pT5_phi", "pT5_isDupReco", "pT5_simIdx"]
    entry_stop = None if nevents < 0 else nevents
    arrs = {b: tree[b].array(library="np", entry_stop=entry_stop) for b in branches}

    eta    = np.concatenate([arrs["pT5_eta"][i]      for i in range(len(arrs["pT5_eta"]))])
    phi    = np.concatenate([arrs["pT5_phi"][i]      for i in range(len(arrs["pT5_phi"]))])
    is_dup = np.concatenate([arrs["pT5_isDupReco"][i] for i in range(len(arrs["pT5_isDupReco"]))]).astype(bool)
    simidx = np.concatenate([arrs["pT5_simIdx"][i]   for i in range(len(arrs["pT5_simIdx"]))]).astype(np.int32)

    genuine = simidx >= 0
    survived = ~is_dup

    cats = {
        "Genuine, survived": ( genuine &  survived, "#2ca02c"),
        "Fake, survived":    (~genuine &  survived, "#98df8a"),
        "Genuine, killed":   ( genuine & ~survived, "#ff7f0e"),
        "Fake, killed":      (~genuine & ~survived, "#d62728"),
    }
    return eta, phi, cats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ntuple", help="LSTNtuple ROOT file")
    parser.add_argument("--nevents", type=int, default=-1,
                        help="First N events (default: all)")
    parser.add_argument("--out", default="pt5_etaphi.png", help="Output filename")
    parser.add_argument("--bins", type=int, default=100,
                        help="Number of bins per axis (default: 100)")
    args = parser.parse_args()

    print(f"Reading {args.ntuple} ...")
    eta, phi, cats = load(args.ntuple, nevents=args.nevents)

    eta_range = (-4.0, 4.0)
    phi_range = (-np.pi, np.pi)
    bins = (args.bins, args.bins)

    titles = list(cats.keys())
    masks  = [v[0] for v in cats.values()]

    for title, mask in zip(titles, masks):
        print(f"  {title}: {int(np.sum(mask)):,}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle("pT5 η–φ by category (AfterBuild OFF)", fontsize=13)

    for ax, title, mask in zip(axes.flat, titles, masks):
        eta_sel = eta[mask]
        phi_sel = phi[mask]
        n = int(np.sum(mask))

        if n == 0:
            ax.set_title(f"{title}\n(empty)")
            ax.set_xlabel("η")
            ax.set_ylabel("φ")
            continue

        h, xedge, yedge, img = ax.hist2d(
            eta_sel, phi_sel,
            bins=bins,
            range=[eta_range, phi_range],
            norm=LogNorm(),
            cmap="viridis",
        )
        fig.colorbar(img, ax=ax, label="pT5 count (log scale)")
        ax.set_title(f"{title}  (N={n:,})")
        ax.set_xlabel("η")
        ax.set_ylabel("φ")
        ax.set_xlim(eta_range)
        ax.set_ylim(phi_range)

    plt.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()

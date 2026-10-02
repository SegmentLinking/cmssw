#!/usr/bin/env python3
"""
lst_plot_pterr_etaerr_sampling.py

Compares old vs new synthetic pLS ptErr / etaErr behaviour directly from the
tracking ntuple, without running LST reconstruction.

Old: every synthetic seed in an event receives the same value = median(see_ptErr)
     / median(see_etaErr) for that event.
New: every synthetic seed receives a random draw from see_ptErr / see_etaErr
     for that event.

Three panels per variable (ptErr and etaErr):
  A — distribution of all real see_ptErr values (the pool)
  B — distribution of per-event medians (old synthetic value)
  C — overlay: real pool vs random-draw synthetic (new) — should be identical

Usage:
  python3 efficiency/python/lst_plot_pterr_etaerr_sampling.py \\
      [--input trackingNtuple-100.root] [--out pterr_etaerr_sampling.png]
"""

import argparse
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK   = "#333333"
MUTED = "#767676"
GRID  = "#DDDDDD"
C_REAL = "#0072B2"   # blue  — real seeds
C_OLD  = "#D55E00"   # red   — old synthetic (median)
C_NEW  = "#009E73"   # green — new synthetic (random draw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="trackingNtuple-100.root")
    parser.add_argument("--out",   default="pterr_etaerr_sampling.png")
    parser.add_argument("--seed",  type=int, default=42,
                        help="RNG seed matching TruthPixelSeeds.cc (default 42)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Reading {args.input} ...")
    f = uproot.open(args.input)
    tree = f["trackingNtuple/tree"]

    ptErr_all  = tree["see_ptErr"].array(library="np")
    etaErr_all = tree["see_etaErr"].array(library="np")

    n_events = len(ptErr_all)
    print(f"Events: {n_events}")

    # Per-event: collect real values, per-event medians, and one random draw per
    # real seed (mimicking: N synthetic seeds per event, each gets one random draw)
    real_ptErr  = []
    real_etaErr = []
    old_ptErr   = []   # one median per event, repeated len(event) times
    old_etaErr  = []
    new_ptErr   = []   # random draws, same count as real seeds
    new_etaErr  = []

    for ievt in range(n_events):
        pts  = ptErr_all[ievt].astype(float)
        etas = etaErr_all[ievt].astype(float)
        n    = len(pts)
        if n == 0:
            continue

        real_ptErr.extend(pts)
        real_etaErr.extend(etas)

        med_pt  = float(np.median(pts))
        med_eta = float(np.median(etas))
        old_ptErr.extend([med_pt]  * n)
        old_etaErr.extend([med_eta] * n)

        # Random draws with replacement — one per real seed, matching new code
        new_ptErr.extend(rng.choice(pts,  size=n, replace=True))
        new_etaErr.extend(rng.choice(etas, size=n, replace=True))

    real_ptErr  = np.array(real_ptErr)
    real_etaErr = np.array(real_etaErr)
    old_ptErr   = np.array(old_ptErr)
    old_etaErr  = np.array(old_etaErr)
    new_ptErr   = np.array(new_ptErr)
    new_etaErr  = np.array(new_etaErr)

    print(f"Total seeds: {len(real_ptErr)}")
    print(f"ptErr  — real median={np.median(real_ptErr):.4f}  "
          f"old median={np.median(old_ptErr):.4f}  "
          f"new median={np.median(new_ptErr):.4f}")
    print(f"etaErr — real median={np.median(real_etaErr):.5f}  "
          f"old median={np.median(old_etaErr):.5f}  "
          f"new median={np.median(new_etaErr):.5f}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(
        "Synthetic pLS ptErr / etaErr: old (per-event median) vs new (random draw from real)\n"
        f"Tracking ntuple: {args.input},  {n_events} events,  {len(real_ptErr):,} seeds",
        fontsize=11, color=INK
    )

    def style(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylabel("seeds", fontsize=8, color=INK)

    def hist(ax, data, color, label, bins, alpha=0.7, lw=0):
        ax.hist(data, bins=bins, color=color, alpha=alpha,
                linewidth=lw, label=label)

    for row, (var, real, old, new, unit, xmax) in enumerate([
        ("ptErr",  real_ptErr,  old_ptErr,  new_ptErr,  "GeV",  0.25),
        ("etaErr", real_etaErr, old_etaErr, new_etaErr, "",     0.006),
    ]):
        bins_full = np.linspace(0, xmax, 80)

        # Panel A: real distribution
        ax = axes[row, 0]
        hist(ax, np.clip(real, 0, xmax), C_REAL, "real seeds", bins_full)
        ax.set_title(f"A — real see_{var} distribution", fontsize=9, color=INK, loc="left")
        ax.set_xlabel(f"see_{var} [{unit}]" if unit else f"see_{var}", fontsize=8, color=INK)
        ax.axvline(np.median(real), color=INK, linestyle=":", linewidth=1.2,
                   label=f"median={np.median(real):.4g}")
        ax.legend(fontsize=7)
        style(ax)

        # Panel B: old synthetic (per-event medians, one per seed)
        ax = axes[row, 1]
        # Zoom in: per-event medians cluster tightly around the global median
        med_vals = np.array([float(np.median(ptErr_all[i])) if var == "ptErr"
                             else float(np.median(etaErr_all[i]))
                             for i in range(n_events) if len(ptErr_all[i]) > 0])
        bins_med = np.linspace(med_vals.min() * 0.8, med_vals.max() * 1.2, 50)
        hist(ax, med_vals, C_OLD, "per-event medians", bins_med)
        ax.set_title(f"B — old synthetic: per-event median of see_{var}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel(f"per-event median [{unit}]" if unit else "per-event median",
                      fontsize=8, color=INK)
        ax.axvline(np.median(real), color=INK, linestyle=":", linewidth=1.2,
                   label=f"global median={np.median(real):.4g}")
        ax.legend(fontsize=7)
        style(ax)

        # Panel C: overlay real vs new synthetic
        ax = axes[row, 2]
        hist(ax, np.clip(real, 0, xmax), C_REAL, "real seeds", bins_full, alpha=0.5)
        hist(ax, np.clip(new,  0, xmax), C_NEW,  "new synthetic (random draw)", bins_full, alpha=0.5)
        ax.set_title(f"C — real vs new synthetic see_{var}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel(f"see_{var} [{unit}]" if unit else f"see_{var}", fontsize=8, color=INK)
        ax.legend(fontsize=7)
        style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

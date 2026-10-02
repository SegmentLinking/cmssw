#!/usr/bin/env python3
"""
lst_plot_pterr_etaerr_diff.py

For every real seed in every event, computes the difference
    see_ptErr[i]  − synthetic_ptErr
    see_etaErr[i] − synthetic_etaErr

where synthetic_ptErr is:
  Old: median(see_ptErr in this event)            [one fixed value per event]
  New: a random draw from see_ptErr in this event [one independent draw per seed]

Plots the distribution of those differences side-by-side (old vs new),
matching the style of pls_all_param_diff_kinematics.png.

ptErr uses relative difference (r−s)/r; etaErr uses absolute (r−s).
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
C_OLD = "#D55E00"
C_NEW = "#009E73"


def rng99(a):
    finite = a[np.isfinite(a)]
    if len(finite) == 0:
        return 1.0
    return max(float(np.percentile(np.abs(finite), 99)), 1e-12)


def plot_panel(ax, old_vals, new_vals, xlabel, title):
    r = max(rng99(old_vals), rng99(new_vals))
    bins = np.linspace(-r, r, 81)

    ax.hist(np.clip(old_vals, bins[0], bins[-1]), bins=bins,
            color=C_OLD, alpha=0.6, label="old (per-event median)", linewidth=0)
    ax.hist(np.clip(new_vals, bins[0], bins[-1]), bins=bins,
            color=C_NEW, alpha=0.6, label="new (random draw)", linewidth=0)
    ax.axvline(0, color=INK, linestyle="--", linewidth=1.1)

    med_o, p16_o, p84_o = (np.median(old_vals),
                            np.percentile(old_vals, 16),
                            np.percentile(old_vals, 84))
    med_n, p16_n, p84_n = (np.median(new_vals),
                            np.percentile(new_vals, 16),
                            np.percentile(new_vals, 84))
    ax.axvline(med_o, color=C_OLD, linestyle=":", linewidth=1.1)
    ax.axvline(med_n, color=C_NEW, linestyle=":", linewidth=1.1)

    ax.text(0.03, 0.97,
            f"old  med={med_o:+.3g}  68%=[{p16_o:+.3g}, {p84_o:+.3g}]\n"
            f"new  med={med_n:+.3g}  68%=[{p16_n:+.3g}, {p84_n:+.3g}]",
            transform=ax.transAxes, fontsize=7, color=INK, va="top",
            fontfamily="monospace")

    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=INK)
    ax.set_ylabel("seeds", fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="trackingNtuple-100.root")
    parser.add_argument("--out",   default="pterr_etaerr_diff.png")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Reading {args.input} ...")
    tree = uproot.open(args.input)["trackingNtuple/tree"]
    ptErr_all  = tree["see_ptErr"].array(library="np")
    etaErr_all = tree["see_etaErr"].array(library="np")
    n_events   = len(ptErr_all)
    print(f"Events: {n_events}")

    old_pt_diff  = []
    new_pt_diff  = []
    old_eta_diff = []
    new_eta_diff = []

    for ievt in range(n_events):
        pt  = ptErr_all[ievt].astype(float)
        eta = etaErr_all[ievt].astype(float)
        n   = len(pt)
        if n == 0:
            continue

        med_pt  = np.median(pt)
        med_eta = np.median(eta)
        rand_pt  = rng.choice(pt,  size=n, replace=True)
        rand_eta = rng.choice(eta, size=n, replace=True)

        # ptErr: relative difference (r−s)/r
        old_pt_diff.append((pt - med_pt)  / pt)
        new_pt_diff.append((pt - rand_pt) / pt)

        # etaErr: absolute difference
        old_eta_diff.append(eta - med_eta)
        new_eta_diff.append(eta - rand_eta)

    old_pt_diff  = np.concatenate(old_pt_diff)
    new_pt_diff  = np.concatenate(new_pt_diff)
    old_eta_diff = np.concatenate(old_eta_diff)
    new_eta_diff = np.concatenate(new_eta_diff)

    n_seeds = len(old_pt_diff)
    print(f"Total seeds: {n_seeds:,}")
    print(f"ptErr  rel — old 68%: [{np.percentile(old_pt_diff,16):+.3f}, "
          f"{np.percentile(old_pt_diff,84):+.3f}]  "
          f"new 68%: [{np.percentile(new_pt_diff,16):+.3f}, "
          f"{np.percentile(new_pt_diff,84):+.3f}]")
    print(f"etaErr abs — old 68%: [{np.percentile(old_eta_diff,16):+.5f}, "
          f"{np.percentile(old_eta_diff,84):+.5f}]  "
          f"new 68%: [{np.percentile(new_eta_diff,16):+.5f}, "
          f"{np.percentile(new_eta_diff,84):+.5f}]")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Synthetic ptErr / etaErr: (real − synthetic) per seed\n"
        "old = per-event median,  new = random draw from event pool\n"
        f"{n_events} events, {n_seeds:,} seeds  ({args.input})",
        fontsize=10, color=INK
    )

    plot_panel(axes[0], old_pt_diff, new_pt_diff,
               "(see_ptErr_real − see_ptErr_synth) / see_ptErr_real",
               "A — ptErr (relative difference)")
    plot_panel(axes[1], old_eta_diff, new_eta_diff,
               "see_etaErr_real − see_etaErr_synth",
               "B — etaErr (absolute difference)")

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

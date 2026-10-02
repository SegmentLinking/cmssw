#!/usr/bin/env python3
"""
s44_plot_fixes_vs_dR.py

TC efficiency, fake rate and duplicate rate vs ΔR(track, closest gen jet) for the Session 44
fix-validation runs (run_s44_fixes.sh). Reads NumDen files (createPerfNumDenHists -J):
  eff : Root__TC_base_0_0_ef_{numer,denom}_deltaR   (sim ΔR, jet-selected denominator)
  fake: Root__TC_fr_{numer,denom}_deltaR            (TC ΔR to closest gen jet)
  dup : Root__TC_dr_{numer,denom}_deltaR
Histograms are 50 bins over [0, 0.1]. Two views: full ΔR < 0.1 (rebinned x2) and jet core
ΔR < 0.02 (native 0.002 bins). Upper panel = rate, lower panel = difference to base (pp).

Usage:
  python3 s44_plot_fixes_vs_dR.py <outdir>
"""

import os
import sys
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

NUMDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../NumDen-files")

# (tag, label, color, marker) — base is neutral; fixes take categorical slots in fixed order
CONFIGS = [
    ("base",    "base",      "#52514e", "o"),
    ("fixB",    "B",         "#2a78d6", "s"),
    ("fixC",    "C",         "#eb6834", "D"),
    ("fixD",    "D",         "#1baf7a", "^"),
    ("fixA",    "A",         "#eda100", "v"),
    ("fixAC",   "A+C",       "#e87ba4", "P"),
    ("fixABCD", "A+B+C+D",   "#4a3aa7", "X"),
]
SAMPLES = {100: [c[0] for c in CONFIGS], 1000: ["base", "fixABCD"]}

METRICS = [
    ("eff",  "TC_base_0_0_ef", "Efficiency",     "TC efficiency"),
    ("fake", "TC_fr",          "Fake rate",      "TC fake rate"),
    ("dup",  "TC_dr",          "Duplicate rate", "TC duplicate rate"),
]
VIEWS = [("dR010", 0.10, 2, "ΔR < 0.1"), ("dR002", 0.02, 1, "jet core, ΔR < 0.02")]


def clopper_pearson(k, n, cl=0.683):
    alpha = 1.0 - cl
    with np.errstate(invalid="ignore", divide="ignore"):
        lo = np.where(k > 0, beta_dist.ppf(alpha / 2, k, n - k + 1), 0.0)
        hi = np.where(k < n, beta_dist.ppf(1 - alpha / 2, k + 1, n - k), 1.0)
    return lo, hi


def load(tag, nevt, prefix, rebin, xmax):
    f = uproot.open(os.path.join(NUMDEN_DIR, f"LSTNumDen_s44_{tag}_{nevt}evt.root"))
    num = f[f"Root__{prefix}_numer_deltaR"].values()
    den = f[f"Root__{prefix}_denom_deltaR"].values()
    edges = f[f"Root__{prefix}_denom_deltaR"].axis().edges()
    num = num.reshape(-1, rebin).sum(1)
    den = den.reshape(-1, rebin).sum(1)
    edges = edges[::rebin]
    keep = edges[1:] <= xmax + 1e-9
    return num[keep], den[keep], edges[:len(num[keep]) + 1]


def plot(nevt, metric, view, outdir):
    mkey, prefix, ylabel, title = metric
    vkey, xmax, rebin, vlabel = view
    tags = SAMPLES[nevt]
    cfgs = [c for c in CONFIGS if c[0] in tags]

    fig, (ax, axd) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.06})
    base_r = None
    width = None
    for i, (tag, label, color, marker) in enumerate(cfgs):
        num, den, edges = load(tag, nevt, prefix, rebin, xmax)
        width = edges[1] - edges[0]
        centers = 0.5 * (edges[1:] + edges[:-1])
        # small horizontal offset so overlapping error bars stay readable
        x = centers + (i - (len(cfgs) - 1) / 2) * width * 0.07
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(den > 0, num / den, np.nan)
        lo, hi = clopper_pearson(num, den)
        tot = num.sum() / den.sum()
        lw = 2.2 if tag in ("base", "fixABCD") else 1.3
        ax.errorbar(x, r, yerr=[r - lo, hi - r], color=color, marker=marker, ms=6, lw=lw,
                    capsize=0, elinewidth=1, label=f"{label}  ({tot:.4f})" if mkey == "dup" else f"{label}  ({tot:.3f})")
        if tag == "base":
            base_r = r
        else:
            axd.plot(x, 100 * (r - base_r), color=color, marker=marker, ms=5, lw=lw)

    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} vs ΔR to closest gen jet — {vlabel}  ({nevt} events, real pLS)",
                 fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(title=f"config  (integrated over {vlabel.split(', ')[-1]})", fontsize=8.5,
              title_fontsize=8.5, frameon=False, ncol=2 if len(cfgs) > 3 else 1)
    axd.axhline(0, color="#52514e", lw=1)
    axd.set_ylabel("Δ vs base [pp]")
    axd.set_xlabel("ΔR(track, closest gen jet)")
    axd.grid(alpha=0.25, lw=0.6)
    axd.set_xlim(0, xmax)
    for a in (ax, axd):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    out = os.path.join(outdir, f"{mkey}_vs_{vkey}_{nevt}evt.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(out)


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    for nevt in SAMPLES:
        for metric in METRICS:
            for view in VIEWS:
                plot(nevt, metric, view, outdir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
lst_plot_pt5_scores_surv.py

Plot rPhiChiSquared score distribution for SURVIVING pT5s only,
split into genuine (simIdx >= 0) and fake (simIdx < 0).

Color scheme matches lst_plot_pt5_etaphi.py:
  Genuine survived: #2ca02c (green)
  Fake survived:    #98df8a (light green)

Usage:
  python3 lst_plot_pt5_scores_surv.py <ntuple.root> [--nevents N] [-o OUTPUT]
"""

import argparse
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(fname, nevents=-1):
    tree = uproot.open(fname)["tree"]
    stop = None if nevents < 0 else nevents
    score  = tree["pT5_score"].array(library="np",      entry_stop=stop)
    is_dup = tree["pT5_isDupReco"].array(library="np",  entry_stop=stop)
    simidx = tree["pT5_simIdx"].array(library="np",     entry_stop=stop)

    score_flat  = np.concatenate([ev.astype(np.float32) for ev in score])
    is_dup_flat = np.concatenate([ev.astype(bool)       for ev in is_dup])
    simidx_flat = np.concatenate([ev.astype(np.int32)   for ev in simidx])
    return score_flat, is_dup_flat, simidx_flat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ntuple", help="LSTNtuple ROOT file")
    ap.add_argument("--nevents", type=int, default=-1)
    ap.add_argument("-o", "--output", default="pt5_scores_surv.png")
    args = ap.parse_args()

    print(f"Reading {args.ntuple} ...")
    score, is_dup, simidx = load(args.ntuple, args.nevents)

    survived = ~is_dup
    genuine  = simidx >= 0

    gen_surv  = score[survived &  genuine]
    fake_surv = score[survived & ~genuine]

    print(f"Genuine survived: {len(gen_surv):,}")
    print(f"Fake    survived: {len(fake_surv):,}")

    # clip near-zero scores so log axis works
    eps = 1e-3
    gen_surv  = np.where(gen_surv  > 0, gen_surv,  eps)
    fake_surv = np.where(fake_surv > 0, fake_surv, eps)

    bins = np.logspace(np.log10(eps), np.log10(1e5), 80)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(gen_surv,  bins=bins, histtype="stepfilled", alpha=0.7,
            color="#2ca02c", label=f"Genuine, survived (N={len(gen_surv):,})")
    ax.hist(fake_surv, bins=bins, histtype="stepfilled", alpha=0.7,
            color="#17becf", label=f"Fake, survived (N={len(fake_surv):,})")

    ax.set_xscale("log")
    ax.set_xlabel("rPhiChiSquared score (log scale)")
    ax.set_ylabel("pT5 count")
    ax.set_title("rPhiChiSquared scores — surviving pT5s only, 50 events (AfterBuild OFF)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()

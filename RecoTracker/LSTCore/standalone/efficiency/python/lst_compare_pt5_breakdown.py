#!/usr/bin/env python3
"""
lst_compare_pt5_breakdown.py

Contrast the T5→pT5 breakdown between two LSTNtuple files that differ in the
RemoveDupQuintupletsAfterBuild stage (ON vs OFF), both carrying the full
t5_isDupBits instrumentation (bit1=condA, bit2=condB, bit3=isPT5-priority).

Produces:
  1. <out>_bars.png   — grouped bar chart of pT5-success rate and isPT5-priority
                        share of BeforeTC kills, per ΔR region, both configs.
  2. <out>_prio_vs_dR.png — isPT5-priority fraction of BeforeTC kills vs ΔR,
                        finely binned, both configs overlaid.

Usage:
  python3 lst_compare_pt5_breakdown.py --on <afterbuildON.root> --off <afterbuildOFF.root> \
      --labels "AfterBuild ON,AfterBuild OFF" --nevents 50 --out compare_pt5
"""

import argparse
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B_AFTER_BUILD = 0x01
B_BTC_A       = 0x02
B_BTC_B       = 0x04
B_BTC_PT5     = 0x08
B_BTC_ANY     = B_BTC_A | B_BTC_B


def load(fname, nevents):
    tree = uproot.open(fname)["tree"]
    stop = None if nevents is None or nevents < 0 else nevents
    br = ["t5_isDupBits", "t5_partOfPT5", "t5_triedInPT5", "t5_simIdx", "sim_genjet_deltaR"]
    return {b: tree[b].array(library="np", entry_stop=stop) for b in br}


def flatten(data):
    """Flatten per-event jagged arrays into flat T5 arrays with per-T5 ΔR."""
    bits, tried, pt5, dr = [], [], [], []
    n = len(data["t5_isDupBits"])
    for i in range(n):
        b = np.asarray(data["t5_isDupBits"][i], dtype=np.int32)
        t = np.asarray(data["t5_triedInPT5"][i], dtype=bool)
        p = np.asarray(data["t5_partOfPT5"][i], dtype=bool)
        s = np.asarray(data["t5_simIdx"][i], dtype=np.int32)
        drsim = np.asarray(data["sim_genjet_deltaR"][i], dtype=float)
        valid = s >= 0
        d = np.where(valid, drsim[np.clip(s, 0, len(drsim) - 1)], np.nan)
        bits.append(b); tried.append(t); pt5.append(p); dr.append(d)
    return (np.concatenate(bits), np.concatenate(tried),
            np.concatenate(pt5), np.concatenate(dr))


def region_stats(bits, tried, pt5, dr, mask):
    """pT5-success rate (of eligible) and isPT5-priority share (of BTC kills)."""
    elig = mask  # AfterBuild-off => all; for ON file bit0-killed excluded below
    elig = elig & ((bits & B_AFTER_BUILD) == 0)
    n_elig = int(np.sum(elig))
    n_pt5 = int(np.sum(elig & pt5))
    killed = mask & ((bits & B_BTC_ANY) > 0)
    n_killed = int(np.sum(killed))
    n_prio = int(np.sum(killed & ((bits & B_BTC_PT5) > 0)))
    return dict(
        pt5_rate=n_pt5 / max(n_elig, 1),
        prio_share=n_prio / max(n_killed, 1),
        n_elig=n_elig, n_pt5=n_pt5, n_killed=n_killed, n_prio=n_prio,
    )


def make_bars(stats_on, stats_off, labels, outfile):
    regions = ["All ΔR", "Jet core\n(ΔR<0.01)", "Isolation\n(ΔR>0.05)"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(regions))
    w = 0.35

    # Panel 1: pT5 success rate
    ax = axes[0]
    ax.bar(x - w/2, [s["pt5_rate"] for s in stats_on],  w, label=labels[0], color="#1f77b4")
    ax.bar(x + w/2, [s["pt5_rate"] for s in stats_off], w, label=labels[1], color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(regions)
    ax.set_ylabel("pT5-match success / eligible T5s")
    ax.set_title("pT5 formation success rate")
    ax.set_ylim(0, 1.0); ax.grid(True, axis="y", alpha=0.3); ax.legend()

    # Panel 2: isPT5-priority share of BeforeTC kills
    ax = axes[1]
    ax.bar(x - w/2, [s["prio_share"] for s in stats_on],  w, label=labels[0], color="#1f77b4")
    ax.bar(x + w/2, [s["prio_share"] for s in stats_off], w, label=labels[1], color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(regions)
    ax.set_ylabel("isPT5-priority kills / BeforeTC kills")
    ax.set_title("isPT5-priority share of BeforeTC kills")
    ax.set_ylim(0, 1.0); ax.grid(True, axis="y", alpha=0.3); ax.legend()

    fig.suptitle("T5→pT5 breakdown: {} vs {} (50 events)".format(*labels))
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved grouped bars → {outfile}")


def prio_vs_dr(bits, tried, pt5, dr, edges):
    """isPT5-priority fraction of BeforeTC kills, per ΔR bin."""
    killed = (bits & B_BTC_ANY) > 0
    prio = killed & ((bits & B_BTC_PT5) > 0)
    frac, err, centers = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inbin = (dr >= lo) & (dr < hi)
        nk = int(np.sum(inbin & killed))
        npr = int(np.sum(inbin & prio))
        centers.append(0.5 * (lo + hi))
        if nk > 0:
            f = npr / nk
            frac.append(f)
            err.append(np.sqrt(f * (1 - f) / nk))  # binomial
        else:
            frac.append(np.nan); err.append(0.0)
    return np.array(centers), np.array(frac), np.array(err)


def make_prio_vs_dr(on, off, labels, outfile, zoom=0.05):
    edges = np.linspace(0, zoom, 26)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (b, t, p, d), lab, col in [(on, labels[0], "#1f77b4"), (off, labels[1], "#ff7f0e")]:
        c, f, e = prio_vs_dr(*(b, t, p, d), edges)
        ax.errorbar(c, f, yerr=e, marker="o", ms=4, lw=1.2, capsize=0, color=col, label=lab)
    ax.set_xlim(0, zoom); ax.set_ylim(0, 1.05)
    ax.set_xlabel("ΔR (T5 — nearest GenJet)")
    ax.set_ylabel("isPT5-priority fraction of BeforeTC kills")
    ax.set_title("isPT5-priority dominance vs ΔR")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved isPT5-priority-vs-ΔR → {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", required=True, help="AfterBuild-ON ntuple")
    ap.add_argument("--off", required=True, help="AfterBuild-OFF ntuple")
    ap.add_argument("--labels", default="AfterBuild ON,AfterBuild OFF")
    ap.add_argument("--nevents", type=int, default=50)
    ap.add_argument("--out", default="compare_pt5")
    args = ap.parse_args()
    labels = [s.strip() for s in args.labels.split(",")]

    on_raw = flatten(load(args.on, args.nevents))
    off_raw = flatten(load(args.off, args.nevents))

    b_on, t_on, p_on, d_on = on_raw
    b_of, t_of, p_of, d_of = off_raw

    def masks(d):
        allm = np.ones(len(d), dtype=bool)
        core = d < 0.01
        iso = d > 0.05
        return allm, core, iso

    on_stats = [region_stats(b_on, t_on, p_on, d_on, m) for m in masks(d_on)]
    off_stats = [region_stats(b_of, t_of, p_of, d_of, m) for m in masks(d_of)]

    make_bars(on_stats, off_stats, labels, f"{args.out}_bars.png")
    make_prio_vs_dr(on_raw, off_raw, labels, f"{args.out}_prio_vs_dR.png")


if __name__ == "__main__":
    main()

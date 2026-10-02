#!/usr/bin/env python3
"""
lst_count_pt5_dedup.py

Counts pT5s surviving / killed by RemoveDupPixelQuintupletsFromMap,
broken down by genuine/fake and ΔR to nearest qualifying genjet.

RemoveDupPixelQuintupletsFromMap is the sole kernel that sets pT5 isDup=true.
No other kernel in the pipeline modifies pT5 isDup, so killed = dedup-killed
and survive = becomes a TC (via AddpT5asTrackCandidate).

Usage:
  python3 lst_count_pt5_dedup.py <ntuple.root> [--nevents N] [--out PREFIX]
"""

import argparse
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(fname, nevents=-1):
    tree = uproot.open(fname)["tree"]
    branches = ["pT5_isDupReco", "pT5_isFake", "pT5_simIdx"]
    has_jet = "sim_genjet_deltaR" in tree.keys()
    if has_jet:
        branches += ["sim_genjet_deltaR"]
    entry_stop = None if nevents < 0 else nevents
    data = {b: tree[b].array(library="np", entry_stop=entry_stop) for b in branches}
    data["has_jet"] = has_jet
    return data


def analyse_event(ievt, data, dr_lo=0.01, dr_hi=0.05):
    is_dup  = np.array(data["pT5_isDupReco"][ievt], dtype=bool)
    simidx  = np.array(data["pT5_simIdx"][ievt], dtype=np.int32)

    genuine = simidx >= 0  # has a sim match with >75% hit purity

    if data["has_jet"]:
        dr_sim = np.array(data["sim_genjet_deltaR"][ievt], dtype=float)
        pt5_dr = np.where(genuine, dr_sim[np.clip(simidx, 0, len(dr_sim) - 1)], np.inf)
        in_core = pt5_dr < dr_lo
        out_far = pt5_dr > dr_hi
    else:
        in_core = np.zeros(len(is_dup), dtype=bool)
        out_far = np.zeros(len(is_dup), dtype=bool)

    def counts(mask):
        m = mask
        total          = int(np.sum(m))
        killed         = int(np.sum(m & is_dup))
        survive        = int(np.sum(m & ~is_dup))
        killed_genuine = int(np.sum(m & is_dup & genuine))
        killed_fake    = int(np.sum(m & is_dup & ~genuine))
        surv_genuine   = int(np.sum(m & ~is_dup & genuine))
        surv_fake      = int(np.sum(m & ~is_dup & ~genuine))
        return dict(total=total,
                    killed=killed, killed_genuine=killed_genuine, killed_fake=killed_fake,
                    survive=survive, surv_genuine=surv_genuine, surv_fake=surv_fake)

    return (counts(np.ones(len(is_dup), dtype=bool)),
            counts(in_core),
            counts(out_far))


def sum_dicts(dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def print_table(label, c):
    tot = max(c["total"], 1)
    print(f"\n=== {label} ===")
    print(f"  Total pT5s:                  {c['total']:>10}")
    print(f"  Killed by dedup:             {c['killed']:>10}  ({100*c['killed']/tot:.1f}%)")
    print(f"    genuine killed:            {c['killed_genuine']:>10}")
    print(f"    fake killed:               {c['killed_fake']:>10}")
    print(f"  Surviving:                   {c['survive']:>10}  ({100*c['survive']/tot:.1f}%)")
    print(f"    genuine surviving:         {c['surv_genuine']:>10}")
    print(f"    fake surviving:            {c['surv_fake']:>10}")


def make_bar(c_all, c_core, c_far, outfile, fraction=False):
    cats   = ["All ΔR", "ΔR<0.01\n(jet core)", "ΔR>0.05\n(isolation)"]
    counts = [c_all, c_core, c_far]

    # bottom-to-top stacking order
    labels = ["Killed, genuine", "Killed, fake", "Survive, genuine", "Survive, fake"]
    colors = ["#ff7f0e", "#d62728", "#2ca02c", "#98df8a"]

    def seg(c):
        denom = max(c["total"], 1) if fraction else 1
        return [c["killed_genuine"] / denom, c["killed_fake"] / denom,
                c["surv_genuine"]   / denom, c["surv_fake"]   / denom]

    data_mat = np.array([seg(c) for c in counts], dtype=float)  # shape (3, 4)

    fig, ax = plt.subplots(figsize=(9, 6))
    bottoms = np.zeros(3)
    for lbl, col, vals in zip(labels, colors, data_mat.T):
        ax.bar(cats, vals, bottom=bottoms, color=col, label=lbl,
               edgecolor="white", linewidth=0.5)
        bottoms += vals

    if fraction:
        ax.set_ylabel("Fraction of total pT5s")
        ax.set_ylim(0, 1.05)
        ax.set_title("pT5 dedup survival — fractional breakdown")
    else:
        ax.set_ylabel("pT5 count (summed over events)")
        ax.set_title("pT5 dedup survival — counts")

    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved → {outfile}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ntuple", help="LSTNtuple ROOT file")
    parser.add_argument("--out", default="pt5_dedup", help="Output filename prefix")
    parser.add_argument("--nevents", type=int, default=-1,
                        help="Process only the first N events (default: all)")
    args = parser.parse_args()

    print(f"Reading {args.ntuple} ...")
    data = load(args.ntuple, nevents=args.nevents)
    n_events = len(data["pT5_isDupReco"])
    print(f"Events: {n_events}  |  has_jet={data['has_jet']}")

    all_c, core_c, far_c = [], [], []
    for ievt in range(n_events):
        ca, cc, cf = analyse_event(ievt, data)
        all_c.append(ca)
        core_c.append(cc)
        far_c.append(cf)

    c_all  = sum_dicts(all_c)
    c_core = sum_dicts(core_c)
    c_far  = sum_dicts(far_c)

    print_table("ALL ΔR", c_all)
    if data["has_jet"]:
        print_table("ΔR < 0.01 (jet core)", c_core)
        print_table("ΔR > 0.05 (isolation)", c_far)

    make_bar(c_all, c_core, c_far, f"{args.out}_counts.png",   fraction=False)
    make_bar(c_all, c_core, c_far, f"{args.out}_fractions.png", fraction=True)


if __name__ == "__main__":
    main()

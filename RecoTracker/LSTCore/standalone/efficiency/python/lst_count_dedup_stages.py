#!/usr/bin/env python3
"""
lst_count_dedup_stages.py

Counts T5s surviving at each dedup stage, broken down by pT5-attempt status,
and produces stacked-bar and ΔR-split plots.

isDupBits encoding:
  bit0 (0x01) = RemoveDupQuintupletsAfterBuild
  bit1 (0x02) = BeforeTC sub-condition A: (dR2<0.001 or nMatched>=5) and d2<1.0
  bit2 (0x04) = BeforeTC sub-condition B: dR2<0.02 and d2<0.1
  bit3 (0x08) = BeforeTC: killed by isPT5-priority (not score)
  bit4 (0x10) = CrossCleanT5

triedInPT5: 1 if T5 was attempted in pT5 building (isDup==0 at that point)
partOfPT5:  1 if T5 was successfully matched to a pT5

Usage:
  python3 lst_count_dedup_stages.py <ntuple.root> [--deltaR]
"""

import sys
import argparse
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── bit masks ─────────────────────────────────────────────────────────────────
B_AFTER_BUILD   = 0x01
B_BTC_A         = 0x02
B_BTC_B         = 0x04
B_BTC_PT5       = 0x08
B_CROSSCLEAN    = 0x10
B_BTC_ANY       = B_BTC_A | B_BTC_B       # any BeforeTC kill
B_ANY_DUP       = 0x1F                     # any kill


def load(fname, nevents=-1):
    tree = uproot.open(fname)["tree"]
    branches = ["t5_isDupBits", "t5_partOfPT5", "t5_triedInPT5",
                "t5_simIdx", "t5_isDuplicate"]
    # sim_genjet_deltaR for ΔR splitting (optional)
    has_jet = "sim_genjet_deltaR" in tree.keys()
    if has_jet:
        branches += ["sim_genjet_deltaR"]
    entry_stop = None if nevents is None or nevents < 0 else nevents
    data = {b: tree[b].array(library="np", entry_stop=entry_stop) for b in branches}
    data["has_jet"] = has_jet
    return data


def analyse_event(ievt, data, dr_lo=0.01, dr_hi=0.05):
    bits   = np.array(data["t5_isDupBits"][ievt], dtype=np.int32)
    tried  = np.array(data["t5_triedInPT5"][ievt], dtype=bool)
    pt5    = np.array(data["t5_partOfPT5"][ievt], dtype=bool)
    simidx = np.array(data["t5_simIdx"][ievt], dtype=np.int32)

    # ΔR lookup (per-sim-track → per-T5 via simidx)
    if data["has_jet"]:
        dr_sim = np.array(data["sim_genjet_deltaR"][ievt], dtype=float)
        # T5s with a valid sim match
        valid   = simidx >= 0
        t5_dr   = np.where(valid, dr_sim[np.clip(simidx, 0, len(dr_sim)-1)], np.inf)
        in_core = (t5_dr < dr_lo)
        out_far = (t5_dr > dr_hi)
    else:
        in_core = np.zeros(len(bits), dtype=bool)
        out_far = np.zeros(len(bits), dtype=bool)

    def counts(mask):
        m = mask
        total   = int(np.sum(m))
        ab      = int(np.sum(m & ((bits & B_AFTER_BUILD) > 0)))
        btc_a   = int(np.sum(m & ((bits & B_BTC_A) > 0)))
        btc_b   = int(np.sum(m & ((bits & B_BTC_B) > 0)))
        btc_pt5 = int(np.sum(m & ((bits & B_BTC_PT5) > 0)))
        btc_any = int(np.sum(m & ((bits & B_BTC_ANY) > 0)))
        cc      = int(np.sum(m & ((bits & B_CROSSCLEAN) > 0)))
        survive = int(np.sum(m & (bits == 0)))

        # pT5-attempt breakdown among those NOT killed by AfterBuild
        eligible = m & ((bits & B_AFTER_BUILD) == 0)
        n_elig   = int(np.sum(eligible))
        n_pt5ok  = int(np.sum(eligible & pt5))         # successful pT5 match
        n_tried  = int(np.sum(eligible & tried & ~pt5)) # tried but failed
        n_never  = int(np.sum(eligible & ~tried & ~pt5))# never tried

        # Of those killed by BeforeTC, how many are tried-but-failed?
        btc_killed = m & ((bits & B_BTC_ANY) > 0)
        btc_triedFailed = int(np.sum(btc_killed & tried & ~pt5))
        btc_pt5ok_lost  = int(np.sum(btc_killed & pt5))
        btc_never_lost  = int(np.sum(btc_killed & ~tried & ~pt5))

        return dict(total=total, ab=ab,
                    eligible=n_elig, pt5ok=n_pt5ok, tried_failed=n_tried, never_tried=n_never,
                    btc_any=btc_any, btc_a=btc_a, btc_b=btc_b, btc_pt5prio=btc_pt5,
                    btc_triedFailed=btc_triedFailed, btc_pt5ok_lost=btc_pt5ok_lost,
                    btc_never_lost=btc_never_lost,
                    cc=cc, survive=survive)

    all_mask  = np.ones(len(bits), dtype=bool)
    core_mask = in_core
    far_mask  = out_far

    return counts(all_mask), counts(core_mask), counts(far_mask)


def sum_dicts(dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def print_table(label, c):
    print(f"\n=== {label} ===")
    print(f"  Total T5s:                        {c['total']:>8}")
    print(f"  Killed by AfterBuild (bit0):      {c['ab']:>8}  ({100*c['ab']/max(c['total'],1):.1f}%)")
    print(f"  Eligible for BeforeTC:            {c['eligible']:>8}  ({100*c['eligible']/max(c['total'],1):.1f}%)")
    print(f"    → successful pT5 match:         {c['pt5ok']:>8}  ({100*c['pt5ok']/max(c['eligible'],1):.1f}% of elig)")
    print(f"    → tried but failed pT5:         {c['tried_failed']:>8}  ({100*c['tried_failed']/max(c['eligible'],1):.1f}% of elig)")
    print(f"    → never tried for pT5:          {c['never_tried']:>8}  ({100*c['never_tried']/max(c['eligible'],1):.1f}% of elig)")
    print(f"  Killed by BeforeTC (any):         {c['btc_any']:>8}  ({100*c['btc_any']/max(c['eligible'],1):.1f}% of elig)")
    print(f"    sub-cond A (tight dR/hits+emb): {c['btc_a']:>8}")
    print(f"    sub-cond B (broad dR+tight emb):{c['btc_b']:>8}")
    print(f"    killed by isPT5-priority:       {c['btc_pt5prio']:>8}  (not score)")
    print(f"    of killed: tried-but-failed:    {c['btc_triedFailed']:>8}")
    print(f"    of killed: successful pT5:      {c['btc_pt5ok_lost']:>8}")
    print(f"    of killed: never tried:         {c['btc_never_lost']:>8}")
    print(f"  Killed by CrossCleanT5 (bit4):    {c['cc']:>8}")
    print(f"  Surviving T5s (no dup bits):      {c['survive']:>8}  ({100*c['survive']/max(c['total'],1):.1f}%)")


def make_stacked_bar(c_all, c_core, c_far, outfile):
    cats   = ["All ΔR", "ΔR<0.01\n(jet core)", "ΔR>0.05\n(isolation)"]
    counts = [c_all, c_core, c_far]

    # Stacked segments among AfterBuild-eligible T5s only (order = bottom to top)
    labels  = ["BeforeTC kill (tried-failed)",
               "BeforeTC kill (pT5 success)", "BeforeTC kill (never tried)",
               "CrossCleanT5 kill", "Surviving"]
    colors  = ["#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#2ca02c"]

    def seg(c):
        return [c["btc_triedFailed"],
                c["btc_pt5ok_lost"],
                c["btc_never_lost"],
                c["cc"],
                c["survive"]]

    data_mat = np.array([seg(c) for c in counts], dtype=float)  # shape (3, 5)

    fig, ax = plt.subplots(figsize=(9, 6))
    bottoms = np.zeros(3)
    for lbl, col, vals in zip(labels, colors, data_mat.T):
        ax.bar(cats, vals, bottom=bottoms, color=col, label=lbl, edgecolor="white", linewidth=0.5)
        bottoms += vals

    ax.set_ylabel("T5 count (summed over events)")
    ax.set_title("T5 fate after AfterBuild — BeforeTC and CrossClean stages")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"\nSaved stacked bar → {outfile}")


def make_fraction_bar(c_all, c_core, c_far, outfile):
    """Same as above but normalised to eligible (post-AfterBuild) T5s."""
    cats   = ["All ΔR", "ΔR<0.01\n(jet core)", "ΔR>0.05\n(isolation)"]
    counts = [c_all, c_core, c_far]

    labels  = ["BeforeTC kill (tried-failed)",
               "BeforeTC kill (pT5 success)", "BeforeTC kill (never tried)",
               "CrossCleanT5 kill", "Surviving"]
    colors  = ["#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#2ca02c"]

    def seg(c):
        elig = max(c["eligible"], 1)
        return [c["btc_triedFailed"]/elig,
                c["btc_pt5ok_lost"]/elig,
                c["btc_never_lost"]/elig,
                c["cc"]/elig,
                c["survive"]/elig]

    data_mat = np.array([seg(c) for c in counts], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))
    bottoms = np.zeros(3)
    for lbl, col, vals in zip(labels, colors, data_mat.T):
        ax.bar(cats, vals, bottom=bottoms, color=col, label=lbl, edgecolor="white", linewidth=0.5)
        bottoms += vals

    ax.set_ylabel("Fraction of AfterBuild-eligible T5s")
    ax.set_ylim(0, 1.05)
    ax.set_title("T5 fate after AfterBuild — fractional breakdown")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved fraction bar → {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ntuple", help="LSTNtuple ROOT file")
    parser.add_argument("--out", default="dedup_stages", help="Output filename prefix")
    parser.add_argument("--nevents", type=int, default=-1,
                        help="Only process the first N events (default -1 = all)")
    args = parser.parse_args()

    print(f"Reading {args.ntuple} ...")
    data = load(args.ntuple, nevents=args.nevents)
    n_events = len(data["t5_isDupBits"])
    print(f"Events: {n_events}  |  has_jet={data['has_jet']}")

    all_c, core_c, far_c = [], [], []
    for ievt in range(n_events):
        ca, cc, cf = analyse_event(ievt, data)
        all_c.append(ca); core_c.append(cc); far_c.append(cf)

    c_all  = sum_dicts(all_c)
    c_core = sum_dicts(core_c)
    c_far  = sum_dicts(far_c)

    # Sanity check
    sanity_ok = True
    # no T5 should be triedInPT5=true AND killed by AfterBuild (bit0)
    for ievt in range(n_events):
        bits  = np.array(data["t5_isDupBits"][ievt], dtype=np.int32)
        tried = np.array(data["t5_triedInPT5"][ievt], dtype=bool)
        bad = np.sum(tried & ((bits & B_AFTER_BUILD) > 0))
        if bad > 0:
            print(f"  SANITY FAIL evt {ievt}: {bad} T5s triedInPT5=True AND AfterBuild-killed")
            sanity_ok = False
    if sanity_ok:
        print("\nSanity check PASSED: no T5 has triedInPT5=True with AfterBuild kill.")

    print_table("ALL ΔR", c_all)
    if data["has_jet"]:
        print_table("ΔR < 0.01 (jet core)", c_core)
        print_table("ΔR > 0.05 (isolation)", c_far)

    make_stacked_bar(c_all, c_core, c_far, f"{args.out}_counts.png")
    make_fraction_bar(c_all, c_core, c_far, f"{args.out}_fractions.png")


if __name__ == "__main__":
    main()

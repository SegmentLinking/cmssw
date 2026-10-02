#!/usr/bin/env python3
"""
lst_compare_nodnn_table.py

Print efficiency and fake rate table (by deltaR bin) comparing baseline vs DNN-disabled.
Uses uproot; mirrors the PyROOT lst_compare_numden_deltaR.py logic.

Usage:
  python3 lst_compare_nodnn_table.py <baseline_numden.root> <nodnn_numden.root>
"""

import sys
import numpy as np
import uproot

RANGES = [("dR<0.01", 0.0, 0.01), ("0.01-0.05", 0.01, 0.05), ("dR>0.05", 0.05, 999.0)]


def integrate(h, lo, hi):
    edges = h.axis().edges()
    vals  = h.values()
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (centers >= lo) & (centers < hi)
    return float(vals[mask].sum())


def ratio_table(f, obj, kind):
    if kind == "ef":
        num_key = f"Root__{obj}_base_0_-1_ef_numer_deltaR"
        den_key = f"Root__{obj}_base_0_-1_ef_denom_deltaR"
    else:
        num_key = f"Root__{obj}_fr_numer_deltaR"
        den_key = f"Root__{obj}_fr_denom_deltaR"
    try:
        num = f[num_key]
        den = f[den_key]
    except KeyError:
        return None
    out = []
    for label, lo, hi in RANGES:
        n = integrate(num, lo, hi)
        d = integrate(den, lo, hi)
        out.append((label, n, d, n / d if d else float("nan")))
    return out


def main():
    if len(sys.argv) < 3:
        print("Usage: lst_compare_nodnn_table.py <baseline.root> <nodnn.root>")
        sys.exit(1)

    baseline_path, nodnn_path = sys.argv[1], sys.argv[2]
    fb = uproot.open(baseline_path)
    fn = uproot.open(nodnn_path)

    print(f"baseline: {baseline_path}")
    print(f"nodnn:    {nodnn_path}\n")

    rows = [
        ("TC",       "ef", "Total TC efficiency"),
        ("pT5",      "ef", "pT5-type TC efficiency"),
        ("T5",       "ef", "T5-type TC efficiency"),
        ("pT5_lower","ef", "pT5_lower (raw obj) efficiency"),
        ("T5_lower", "ef", "T5_lower (raw obj) efficiency"),
        ("TC",       "fr", "Total TC fake rate"),
        ("pT5",      "fr", "pT5-type fake rate"),
        ("T5",       "fr", "T5-type fake rate"),
    ]

    for obj, kind, title in rows:
        tb = ratio_table(fb, obj, kind)
        tn = ratio_table(fn, obj, kind)
        print(f"== {title} ==")
        if tb is None or tn is None:
            print("   (histograms missing in one or both files)")
            continue
        for (lab, n0, d0, r0), (_, n1, d1, r1) in zip(tb, tn):
            print(
                f"  {lab:>12}: base {r0:.4f} ({n0:.0f}/{d0:.0f})"
                f"  ->  nodnn {r1:.4f} ({n1:.0f}/{d1:.0f})"
                f"   delta {r1 - r0:+.4f}"
            )
        print()


if __name__ == "__main__":
    main()

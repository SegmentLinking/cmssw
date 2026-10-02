#!/usr/bin/env python3
"""Compare Experiment F NumDen vs idealpls_fixed baseline, per Session 20 metric table."""
import sys
import ROOT

ROOT.gROOT.SetBatch(True)

BASE = "LSTNumDen_idealpls_fixed.root"
NEW = sys.argv[1] if len(sys.argv) > 1 else "LSTNumDen_noT5pT5dedup.root"

# ΔR bin edges: integrate histogram counts in ranges
RANGES = [("dR<0.01", 0.0, 0.01), ("0.01-0.05", 0.01, 0.05), ("dR>0.05", 0.05, 999.0)]


def integrate(h, lo, hi):
    tot = 0.0
    for b in range(0, h.GetNbinsX() + 2):  # include under/overflow
        c = h.GetBinCenter(b)
        if b == 0:
            c = h.GetXaxis().GetBinLowEdge(1) - 1e-9
        if b == h.GetNbinsX() + 1:
            c = h.GetXaxis().GetBinUpEdge(h.GetNbinsX()) + 1e-9
        if lo <= c < hi:
            tot += h.GetBinContent(b)
    return tot


def ratio_table(f, obj, kind):
    """kind: 'ef' (base_0_-1) or 'fr'."""
    if kind == "ef":
        num = f.Get(f"Root__{obj}_base_0_-1_ef_numer_deltaR")
        den = f.Get(f"Root__{obj}_base_0_-1_ef_denom_deltaR")
    else:
        num = f.Get(f"Root__{obj}_fr_numer_deltaR")
        den = f.Get(f"Root__{obj}_fr_denom_deltaR")
    if not num or not den:
        return None
    out = []
    for label, lo, hi in RANGES:
        n, d = integrate(num, lo, hi), integrate(den, lo, hi)
        out.append((label, n, d, n / d if d else float("nan")))
    return out


fb = ROOT.TFile(BASE)
fn = ROOT.TFile(NEW)

print(f"baseline: {BASE}")
print(f"new:      {NEW}\n")

for obj, kind, title in [
    ("TC", "ef", "Total TC efficiency"),
    ("pT5", "ef", "pT5-type TC efficiency"),
    ("T5", "ef", "T5-type TC efficiency"),
    ("pT5_lower", "ef", "pT5_lower (raw obj) efficiency"),
    ("T5_lower", "ef", "T5_lower (raw obj) efficiency"),
    ("TC", "fr", "Total TC fake rate"),
    ("pT5", "fr", "pT5-type fake rate"),
    ("T5", "fr", "T5-type fake rate"),
]:
    tb = ratio_table(fb, obj, kind)
    tn = ratio_table(fn, obj, kind)
    print(f"== {title} ==")
    if tb is None or tn is None:
        print("   (histograms missing)")
        continue
    for (lab, n0, d0, r0), (_, n1, d1, r1) in zip(tb, tn):
        print(
            f"  {lab:>10}: base {r0:.4f} ({n0:.0f}/{d0:.0f})  ->  "
            f"new {r1:.4f} ({n1:.0f}/{d1:.0f})   delta {r1 - r0:+.4f}"
        )
    print()

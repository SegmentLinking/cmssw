#!/usr/bin/env python3
"""
s47_demote_resolution.py

Track-parameter resolution of pT5s demoted to T5 TCs (LST_PT5_DEMOTE_SCORE), compared with the
same sim tracks in a base run. The events are stored in a different order in each run, so they are
matched by a sim_pt fingerprint. Sim indices within an event are identical (same input event).

For each denominator sim (s44_eval_fixes.py selection) matched (sim_tcIdx >= 0) in BOTH runs, the
residuals of its matched TC are grouped by (TC type in base -> TC type in test):
  pT5->T5 = demoted; pT5->pT5, T5->T5 = unchanged controls.
Reported per group and sim-pT bin: median and robust sigma (half the 16-84% width) of
(tc_pt - sim_pt)/sim_pt, the fraction with |relative pT residual| > 0.5, and robust sigmas of
d(eta) and d(phi). The ntuple has no TC dxy/dz, so those can't be checked.

Usage:
  python3 s47_demote_resolution.py <base LSTNtuple.root> <test LSTNtuple.root> [core_dR_max]
"""

import sys
import collections
import numpy as np
import uproot

TYPES = {4: "T5", 5: "pT3", 7: "pT5", 8: "pLS", 9: "T4"}
BR = ["sim_pt", "sim_eta", "sim_phi", "sim_q", "sim_vx", "sim_vy", "sim_vz", "sim_genjet_idx",
      "sim_genjet_deltaR", "genjet_pt", "genjet_eta", "sim_tcIdx", "tc_pt", "tc_eta", "tc_phi", "tc_type"]
PT_BINS = [(0.8, 10), (10, 100), (100, 1e9)]


def load(fn):
    t = uproot.open(fn)["tree"]
    have = set(t.keys())
    return t.arrays([b for b in BR if b in have], library="np")


def fingerprint(sim_pt):
    return tuple(np.round(np.sort(sim_pt.astype(float))[-8:], 3))


def denom(a, ie):
    pt = a["sim_pt"][ie].astype(float)
    gj = a["sim_genjet_idx"][ie].astype(np.int64)
    gpt, geta = a["genjet_pt"][ie].astype(float), a["genjet_eta"][ie].astype(float)
    gc = np.clip(gj, 0, max(len(gpt) - 1, 0))
    gpt_s = gpt[gc] if len(gpt) else np.zeros_like(pt)
    geta_s = geta[gc] if len(geta) else np.zeros_like(pt)
    return ((a["sim_q"][ie] != 0) & (pt > 0.8) & (np.abs(a["sim_eta"][ie]) < 4.5) &
            (np.abs(a["sim_vz"][ie]) < 30) &
            (np.hypot(a["sim_vx"][ie].astype(float), a["sim_vy"][ie].astype(float)) < 2.5) &
            (gj >= 0) & (gpt_s > 1000) & (np.abs(geta_s) < 2.5))


def robust_sigma(x):
    return 0.5 * (np.percentile(x, 84) - np.percentile(x, 16)) if len(x) > 5 else float("nan")


def main():
    base, test = load(sys.argv[1]), load(sys.argv[2])
    drmax = float(sys.argv[3]) if len(sys.argv) > 3 else np.inf
    fp_test = {fingerprint(test["sim_pt"][i]): i for i in range(len(test["sim_pt"]))}

    res = collections.defaultdict(list)  # (group, ptbin) -> list of (dpt_rel, deta, dphi)
    n_ev = 0
    for ib in range(len(base["sim_pt"])):
        it = fp_test.get(fingerprint(base["sim_pt"][ib]))
        if it is None:
            continue
        n_ev += 1
        sel = denom(base, ib) & (base["sim_genjet_deltaR"][ib].astype(float) < drmax)
        for s in np.nonzero(sel)[0]:
            tb, tt = int(base["sim_tcIdx"][ib][s]), int(test["sim_tcIdx"][it][s])
            if tb < 0 or tt < 0:
                continue
            group = f"{TYPES.get(int(base['tc_type'][ib][tb]), '?')}->{TYPES.get(int(test['tc_type'][it][tt]), '?')}"
            spt, seta, sphi = float(base["sim_pt"][ib][s]), float(base["sim_eta"][ib][s]), float(base["sim_phi"][ib][s])
            for name, arr, idx in (("base", base, (ib, tb)), ("test", test, (it, tt))):
                e, k = idx
                dpt = (float(arr["tc_pt"][e][k]) - spt) / spt
                deta = float(arr["tc_eta"][e][k]) - seta
                dphi = (float(arr["tc_phi"][e][k]) - sphi + np.pi) % (2 * np.pi) - np.pi
                for lo, hi in PT_BINS:
                    if lo < spt <= hi:
                        res[(group, name, f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}")].append((dpt, deta, dphi))

    print(f"matched events: {n_ev}/{len(base['sim_pt'])}   selection: denominator, dR<{drmax}")
    print(f"{'group':<10}{'run':<6}{'simpT':>8}{'N':>7}{'med dpt/pt':>12}{'sig dpt/pt':>12}{'|dpt/pt|>.5':>13}"
          f"{'sig deta':>11}{'sig dphi':>11}")
    for group in ["pT5->T5", "pT5->pT5", "T5->T5"]:
        for lo, hi in PT_BINS:
            b = f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}"
            for run in ("base", "test"):
                v = np.array(res.get((group, run, b), []))
                if len(v) == 0:
                    continue
                print(f"{group:<10}{run:<6}{b:>8}{len(v):>7}{np.median(v[:, 0]):>12.4f}{robust_sigma(v[:, 0]):>12.4f}"
                      f"{np.mean(np.abs(v[:, 0]) > 0.5):>13.3f}{robust_sigma(v[:, 1]):>11.5f}{robust_sigma(v[:, 2]):>11.5f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
s44_eval_fixes.py

Summary metrics for the Session 44 fix-validation runs (run_s44_fixes.sh). Reads ONE LST
ntuple and prints a single space-separated `key=value` line:

  - TC efficiency (sim_tcIdx >= 0, i.e. matched TC with frac > 0.75) for jet-core windows
    ΔR < 0.02 / 0.05 / 0.10, the core at sim pT > 100 GeV, and the full denominator
  - TC count, fake rate and duplicate rate over all TCs

Denominator selection is identical to eval_core_eff_fakerate.py (performance.cc base_0_0 with -J).

Usage:
  python3 s44_eval_fixes.py <LSTNtuple.root>
"""

import sys
import numpy as np
import uproot

PT_CUT     = 0.8
ETA_CUT    = 4.5
VTX_Z_MAX  = 30.0
VTX_R_MAX  = 2.5
GJ_PT_MIN  = 1000.0
GJ_ETA_MAX = 2.5

# (key, ΔR upper edge, min sim pT)
WINDOWS = [("dr002", 0.02, 0.0), ("dr005", 0.05, 0.0), ("dr010", 0.10, 0.0),
           ("dr002pt100", 0.02, 100.0), ("all", np.inf, 0.0)]


def main():
    tree = uproot.open(sys.argv[1])["tree"]
    data = tree.arrays(["sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz",
                        "sim_genjet_deltaR", "sim_genjet_idx", "genjet_pt", "genjet_eta",
                        "sim_tcIdx", "tc_isFake", "tc_isDuplicate"], library="np")

    num = {k: 0 for k, _, _ in WINDOWS}
    den = {k: 0 for k, _, _ in WINDOWS}
    n_tc = n_fake = n_dup = 0

    for ie in range(len(data["sim_pt"])):
        sim_pt = data["sim_pt"][ie].astype(float)
        sim_vx = data["sim_vx"][ie].astype(float)
        sim_vy = data["sim_vy"][ie].astype(float)
        sim_gj = data["sim_genjet_idx"][ie].astype(np.int32)
        gj_pt = data["genjet_pt"][ie].astype(float)
        gj_eta = data["genjet_eta"][ie].astype(float)
        gj_clip = np.clip(sim_gj, 0, max(len(gj_pt) - 1, 0))
        gj_pt_s = gj_pt[gj_clip] if len(gj_pt) else np.zeros_like(sim_pt)
        gj_eta_s = gj_eta[gj_clip] if len(gj_eta) else np.zeros_like(sim_pt)

        denom = ((data["sim_q"][ie] != 0) & (sim_pt > PT_CUT) &
                 (np.abs(data["sim_eta"][ie]) < ETA_CUT) &
                 (np.abs(data["sim_vz"][ie]) < VTX_Z_MAX) &
                 (np.sqrt(sim_vx**2 + sim_vy**2) < VTX_R_MAX) &
                 (sim_gj >= 0) & (gj_pt_s > GJ_PT_MIN) & (np.abs(gj_eta_s) < GJ_ETA_MAX))
        dr = data["sim_genjet_deltaR"][ie].astype(float)
        matched = data["sim_tcIdx"][ie].astype(np.int64) >= 0

        for k, hi, ptmin in WINDOWS:
            sel = denom & (dr >= 0) & (dr < hi) & (sim_pt > ptmin)
            den[k] += int(np.count_nonzero(sel))
            num[k] += int(np.count_nonzero(sel & matched))

        tc_fake = data["tc_isFake"][ie].astype(np.int32)
        n_tc += tc_fake.size
        n_fake += int(np.count_nonzero(tc_fake > 0))
        n_dup += int(np.count_nonzero(data["tc_isDuplicate"][ie].astype(np.int32) > 0))

    fields = []
    for k, _, _ in WINDOWS:
        eff = num[k] / den[k] if den[k] else float("nan")
        fields += [f"{k}_num={num[k]}", f"{k}_den={den[k]}", f"{k}_eff={eff:.4f}"]
    fields += [f"n_tc={n_tc}",
               f"fakerate={n_fake / n_tc if n_tc else float('nan'):.4f}",
               f"duprate={n_dup / n_tc if n_tc else float('nan'):.4f}"]
    print(" ".join(fields))


if __name__ == "__main__":
    main()

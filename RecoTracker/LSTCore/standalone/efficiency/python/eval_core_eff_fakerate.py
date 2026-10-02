#!/usr/bin/env python3
"""
eval_core_eff_fakerate.py

Fast objective-function evaluator for the dedup-cut tuning sweep. Reads ONE LST
ntuple and prints a single space-separated `key=value` line with:

  - core TC efficiency in the jet core, for two ΔR windows:
      [0, 0.002)  (the single lowest bin, as requested)
      [0, 0.01)   (wider jet-core window, used for ranking stability)
  - TC fake rate and duplicate-or-fake rate

All quantities are read directly from default ntuple branches (no NumDen file).
Denominator selection matches createPerfNumDenHists -J / performance.cc and
lst_plot_dedup_stages_T5_pT5.py exactly.

Usage:
  python3 eval_core_eff_fakerate.py debug.root [--nevents N]
"""

import argparse
import numpy as np
import uproot

# ── denominator selection (matches performance.cc base_0_0 with -J) ──────────
PT_CUT     = 0.8
ETA_CUT    = 4.5
VTX_Z_MAX  = 30.0
VTX_R_MAX  = 2.5
GJ_PT_MIN  = 1000.0
GJ_ETA_MAX = 2.5

WINDOWS = [("core002", 0.002), ("core004", 0.004), ("core006", 0.006), ("core010", 0.010)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ntuple")
    ap.add_argument("--nevents", type=int, default=None)
    args = ap.parse_args()

    tree = uproot.open(args.ntuple)["tree"]
    branches = [
        "sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz",
        "sim_genjet_deltaR", "sim_genjet_idx", "genjet_pt", "genjet_eta",
        "sim_tcIdx",
        "tc_isFake", "tc_isDuplicate",
    ]
    data = tree.arrays(branches, library="np")
    n_events = len(data["sim_pt"])
    if args.nevents is not None:
        n_events = min(n_events, args.nevents)

    num = {k: 0 for k, _ in WINDOWS}
    den = {k: 0 for k, _ in WINDOWS}
    n_tc = 0
    n_fake = 0
    n_dupfake = 0

    for ie in range(n_events):
        sim_pt  = data["sim_pt"][ie].astype(float)
        sim_eta = data["sim_eta"][ie].astype(float)
        sim_q   = data["sim_q"][ie].astype(np.int32)
        sim_vx  = data["sim_vx"][ie].astype(float)
        sim_vy  = data["sim_vy"][ie].astype(float)
        sim_vz  = data["sim_vz"][ie].astype(float)
        sim_dr  = data["sim_genjet_deltaR"][ie].astype(float)
        sim_gj  = data["sim_genjet_idx"][ie].astype(np.int32)
        gj_pt   = data["genjet_pt"][ie].astype(float)
        gj_eta  = data["genjet_eta"][ie].astype(float)
        sim_tc  = data["sim_tcIdx"][ie].astype(np.int64)

        vtx_perp = np.sqrt(sim_vx**2 + sim_vy**2)
        gj_clip  = np.clip(sim_gj, 0, max(len(gj_pt) - 1, 0))
        gj_pt_s  = gj_pt[gj_clip] if len(gj_pt) else np.zeros_like(sim_pt)
        gj_eta_s = gj_eta[gj_clip] if len(gj_eta) else np.zeros_like(sim_pt)

        denom = (
            (sim_q != 0) &
            (sim_pt > PT_CUT) &
            (np.abs(sim_eta) < ETA_CUT) &
            (np.abs(sim_vz) < VTX_Z_MAX) &
            (vtx_perp < VTX_R_MAX) &
            (sim_gj >= 0) &
            (gj_pt_s > GJ_PT_MIN) &
            (np.abs(gj_eta_s) < GJ_ETA_MAX)
        )

        for isim in np.where(denom)[0]:
            dr = float(sim_dr[isim])
            matched = int(sim_tc[isim]) >= 0
            for k, hi in WINDOWS:
                if 0.0 <= dr < hi:
                    den[k] += 1
                    if matched:
                        num[k] += 1

        tc_fake = data["tc_isFake"][ie].astype(np.int32)
        tc_dup  = data["tc_isDuplicate"][ie].astype(np.int32)
        n_tc      += tc_fake.size
        n_fake    += int(np.count_nonzero(tc_fake > 0))
        n_dupfake += int(np.count_nonzero((tc_fake > 0) | (tc_dup > 0)))

    def eff(k):
        return (num[k] / den[k]) if den[k] > 0 else float("nan")

    fakerate    = (n_fake / n_tc) if n_tc > 0 else float("nan")
    dupfakerate = (n_dupfake / n_tc) if n_tc > 0 else float("nan")

    fields = []
    for k, _ in WINDOWS:
        fields += [f"{k}_num={num[k]}", f"{k}_den={den[k]}", f"{k}_eff={eff(k):.4f}"]
    fields += [
        f"n_tc={n_tc}", f"n_fake={n_fake}", f"fakerate={fakerate:.4f}",
        f"n_dupfake={n_dupfake}", f"dupfakerate={dupfakerate:.4f}",
    ]
    print(" ".join(fields))


if __name__ == "__main__":
    main()

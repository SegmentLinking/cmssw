#!/usr/bin/env python3
"""Acceptance of genuine T3-pairs tried by CreateQuintuplets (Quintuplet.h:1950-1972) vs the local-density inputs
that the current T5 DNN receives (Quintuplet.h:1538-1540: f2 = #T3 leaving MD3, f3 = #T3 leaving MD1,
f4 = #MDs in the first module), core vs non-core, S45 (new DNN with density inputs) vs S44 (old DNN, no density inputs).
Density is recomputed from the ntuple (T3 by first MD; MDs per lower module = md_detId>>2 of the lower hit).
NB in S44 the T3 collection differs slightly (no widened pointing), so f2/f3 differ a bit between files.
Usage: t5pair_density.py <ntuple> <tag>"""
import sys, collections
import numpy as np
import uproot

fn, tag = sys.argv[1], sys.argv[2]
PT_CUT, ETA_CUT = 0.8, 4.5
BR = ["sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz", "sim_genjet_deltaR", "sim_genjet_idx", "genjet_pt",
      "genjet_eta", "sim_t3IdxAll", "sim_t3IdxAllFrac", "t3_lsIdx0", "t3_lsIdx1", "ls_mdIdx0", "ls_mdIdx1", "md_layer",
      "md_detId", "md_isPLS", "t5_t3Idx0", "t5_t3Idx1"]
rows = []
for A in uproot.open(fn)["tree"].iterate(BR, step_size=10, library="np"):
    for ie in range(len(A["sim_pt"])):
        pt = A["sim_pt"][ie].astype(float)
        gj = A["sim_genjet_idx"][ie].astype(np.int64)
        gpt, geta = A["genjet_pt"][ie].astype(float), A["genjet_eta"][ie].astype(float)
        gc = np.clip(gj, 0, max(len(gpt) - 1, 0))
        gpt_s = gpt[gc] if len(gpt) else np.zeros_like(pt)
        geta_s = geta[gc] if len(geta) else np.zeros_like(pt)
        dr = A["sim_genjet_deltaR"][ie].astype(float)
        sel = ((A["sim_q"][ie] != 0) & (pt > PT_CUT) & (np.abs(A["sim_eta"][ie]) < ETA_CUT) & (np.abs(A["sim_vz"][ie]) < 30) &
               (np.hypot(A["sim_vx"][ie].astype(float), A["sim_vy"][ie].astype(float)) < 2.5) & (gj >= 0) & (gpt_s > 1000) &
               (np.abs(geta_s) < 2.5))
        ls0 = A["ls_mdIdx0"][ie].astype(np.int64); ls1 = A["ls_mdIdx1"][ie].astype(np.int64)
        t0 = A["t3_lsIdx0"][ie].astype(np.int64); t1 = A["t3_lsIdx1"][ie].astype(np.int64)
        m0 = ls0[t0]; m1 = ls1[t0]; m2 = ls1[t1]
        nT3byMD = np.bincount(m0, minlength=len(A["md_layer"][ie]))
        mod = A["md_detId"][ie].astype(np.int64) >> 2
        u, inv, cnt = np.unique(mod, return_inverse=True, return_counts=True)
        nMDmod = cnt[inv]
        lay = A["md_layer"][ie]
        t5set = set(zip(A["t5_t3Idx0"][ie].tolist(), A["t5_t3Idx1"][ie].tolist()))
        for s in np.nonzero(sel)[0]:
            g = np.asarray(A["sim_t3IdxAll"][ie][s])[np.asarray(A["sim_t3IdxAllFrac"][ie][s]) >= 0.75].astype(np.int64)
            by0 = collections.defaultdict(list)
            for t in g:
                by0[int(m0[t])].append(int(t))
            for t in g:
                if lay[m0[t]] not in (1, 2, 7):
                    continue
                for u2 in by0.get(int(m2[t]), []):
                    rows.append((0 <= dr[s] < 0.02, pt[s], (int(t), u2) in t5set, nT3byMD[m2[t]], nT3byMD[m0[t]], nMDmod[m0[t]]))
R = np.array(rows, dtype=float)
core, acc, f2, f3, f4 = R[:, 0] > 0, R[:, 2] > 0, R[:, 3], R[:, 4], R[:, 5]
print(f"== {tag}: tried genuine T3 pairs {len(R)} (core {core.sum():.0f})")
for nm, f, bins in [("f2 #T3 leaving MD3", f2, [0, 5, 10, 20, 40, 80, 1e9]), ("f3 #T3 leaving MD1", f3, [0, 5, 10, 20, 40, 80, 1e9]),
                    ("f4 #MD in 1st module", f4, [0, 5, 10, 20, 40, 1e9])]:
    print(f"  {nm}: median core {np.median(f[core]):.0f}  non-core {np.median(f[~core]):.0f}")
    print(f"    {'bin':>12s} {'core n':>7s} {'core acc':>9s} {'nc n':>7s} {'nc acc':>7s}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mc = core & (f >= lo) & (f < hi); mn = ~core & (f >= lo) & (f < hi)
        print(f"    {f'[{lo:.0f},{hi:.0f})':>12s} {mc.sum():>7d} {acc[mc].mean() if mc.sum() else float('nan'):>9.3f} "
              f"{mn.sum():>7d} {acc[mn].mean() if mn.sum() else float('nan'):>7.3f}")

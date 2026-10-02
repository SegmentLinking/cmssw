#!/usr/bin/env python3
"""S44 (old) -> S45 (current) per-track differences at LS and T3 level (effect of b07f513662f origin-free LS residual
kLsLineResidCut=0.75, Segment.h, and the widened T3 pointing kT3PointingWiden=1.7, Triplet.h), and reverse class
transitions (S44 no genuine T5 -> S45 genuine T5)."""
import pickle, collections
import numpy as np
SCR = "/tmp/claude-63526/-mnt-data1-kk829-CMSSW-16-1-1-src-RecoTracker-LSTCore-standalone/dbd9af1c-87ff-4b4e-9196-c80ebb15a786/scratchpad/s47/early_md_ls/"
c5 = pickle.load(open(SCR + "stage_survival_s45.pkl", "rb")); c4 = pickle.load(open(SCR + "stage_survival_s44.pkl", "rb"))
key = lambda c: [(round(float(e), 3), round(float(p), 4), round(float(h), 4)) for e, p, h in zip(c["evt"], c["pt"], c["eta"])]
k4 = {k: i for i, k in enumerate(key(c4))}
j = np.array([k4[k] for k in key(c5)])
core = c5["core"].astype(bool)
for lab, m in [("core", core), ("non-core", ~core)]:
    d_ls = c5["n_lsp_ok"][m] - c4["n_lsp_ok"][j][m]
    d_t3 = c5["n_t3able_ok"][m] - c4["n_t3able_ok"][j][m]
    g5, g4 = c5["n_gt5"][m] > 0, c4["n_gt5"][j][m] > 0
    gt3_5, gt3_4 = c5["n_gt3"][m] > 0, c4["n_gt3"][j][m] > 0
    print(f"{lab}: tracks {m.sum()} | LS adjMD pairs built S44 {c4['n_lsp_ok'][j][m].sum()} -> S45 {c5['n_lsp_ok'][m].sum()} "
          f"(tracks losing >=1 LS {(d_ls < 0).sum()}, gaining {(d_ls > 0).sum()})")
    print(f"   T3 on T3-able genuine LS pairs S44 {c4['n_t3able_ok'][j][m].sum()} -> S45 {c5['n_t3able_ok'][m].sum()} "
          f"| tracks with >=1 genuine T3: S44 {gt3_4.sum()} S45 {gt3_5.sum()} (lost {(gt3_4 & ~gt3_5).sum()}, gained {(~gt3_4 & gt3_5).sum()})")
    print(f"   genuine T5: S44 {g4.sum()} S45 {g5.sum()} (lost {(g4 & ~g5).sum()}, gained {(~g4 & g5).sum()})")

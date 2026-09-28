#!/usr/bin/env python3
"""Attribute tracks without a genuine T5 to the earliest missing early stage (uses stage_survival pickle).
Classes (first that applies):
  A_fewMDlayers : < 5 logical layers with a genuine MD (a 5-MD T5 is geometrically impossible from genuine MDs)
  B_noGenLS     : >=5 genuine-MD layers but no genuine LS
  C_noGenT3     : genuine LS exist but no genuine T3 (split: T3-able LS pair existed -> T3 cut; else LS chain broken)
  D_noT3inReg   : genuine T3 only outside the T5 start region (BL1/BL2/EL1)
  E_noT5pair    : genuine T3 in region but no genuine outer T3 sharing its 3rd MD (pair never tried)
  F_triedReject : a genuine T3 pair was tried (CreateQuintuplets loop) but no T5 built -> dBeta1/dBeta2/T5-RZ/T5-DNN
Usage: no_t5_attrib.py <tag>"""
import sys, pickle, collections
import numpy as np
SCR = "/tmp/claude-63526/-mnt-data1-kk829-CMSSW-16-1-1-src-RecoTracker-LSTCore-standalone/dbd9af1c-87ff-4b4e-9196-c80ebb15a786/scratchpad/s47/early_md_ls/"
tag = sys.argv[1]
c = pickle.load(open(SCR + f"stage_survival_{tag}.pkl", "rb"))
n = len(c["pt"])


def cls(i):
    if c["n_gt5"][i] > 0:
        return "has_gT5"
    if c["nlay_md"][i] < 5:
        return "A_fewMDlayers"
    if c["n_gls"][i] == 0:
        return "B_noGenLS"
    if c["n_gt3"][i] == 0:
        return "C_noGenT3(T3cut)" if c["n_t3able"][i] > 0 else "C_noGenT3(noLSpair)"
    if c["n_gt3_region"][i] == 0:
        return "D_noT3inReg"
    if c["n_t5try"][i] == 0:
        return "E_noT5pair"
    return "F_triedReject"


K = np.array([cls(i) for i in range(n)])
core = c["core"].astype(bool)
order = ["A_fewMDlayers", "B_noGenLS", "C_noGenT3(noLSpair)", "C_noGenT3(T3cut)", "D_noT3inReg", "E_noT5pair",
         "F_triedReject", "has_gT5"]
print(f"== {tag}  class counts (tracks)")
print(f"{'class':22s}{'core':>7s}{'core%':>7s}{'noncore':>9s}{'nc%':>7s}{'coreFAIL':>9s}{'no_T5 bkt':>10s}{'coreFail_pT>50':>15s}")
for k in order:
    m = K == k
    print(f"{k:22s}{(m & core).sum():>7d}{100 * (m & core).sum() / core.sum():>6.1f}%{(m & ~core).sum():>9d}"
          f"{100 * (m & ~core).sum() / (~core).sum():>6.1f}%{(m & core & ~c['matched'].astype(bool)).sum():>9d}"
          f"{(m & core & (c['bucket'] == 'no_T5')).sum():>10d}{(m & core & ~c['matched'].astype(bool) & (c['pt'] > 50)).sum():>15d}")
# no_T5 bucket details
sel = core & (c["bucket"] == "no_T5")
print("\n-- core no_T5 bucket (", sel.sum(), "): other genuine objects")
for k in order:
    m = sel & (K == k)
    if m.sum():
        print(f"  {k:22s} n={m.sum():3d}  pT med {np.median(c['pt'][m]):7.1f}  |eta| med {np.median(np.abs(c['eta'][m])):.2f}"
              f"  nlayMD {collections.Counter(c['nlay_md'][m].tolist())}  gT4 {(c['n_gt4'][m] > 0).sum()} gpT3 {(c['n_gpt3'][m] > 0).sum()}"
              f"  n_t5try {c['n_t5try'][m].tolist() if k == 'F_triedReject' else ''}")
print("\n-- all core FAILURES without genuine T5, by bucket x class")
fm = core & ~c["matched"].astype(bool) & (K != "has_gT5")
cnt = collections.Counter(zip(c["bucket"][fm].tolist(), K[fm].tolist()))
for (b, k), v in sorted(cnt.items()):
    print(f"  {b:16s} {k:22s} {v}")

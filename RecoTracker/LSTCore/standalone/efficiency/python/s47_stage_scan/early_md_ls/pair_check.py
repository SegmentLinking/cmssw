#!/usr/bin/env python3
"""For S45 tracks that lost their genuine T5 relative to S44 (old code), find the S44 genuine T5's two T3s
(identified by their 3 MDs via anchor-hit coordinates) in the S45 ntuple:
  both T3s exist in S45 and form a tried pair, but no S45 T5 with that pair  -> rejected at T5 creation.
     dBeta selector and passT5RZConstraint pass thresholds are unchanged (7c9dde8e9cc vs HEAD) and the inputs
     (MD anchors, T3 radius/centre from the same 3 anchors) are identical, so the reject is the new T5 DNN
     (Quintuplet.h:1647-1673) [or computeDnnFeatures inputs], not dBeta/r-z.
  a T3 is missing in S45 -> T3-stage loss (T3 DNN / rz / pointing changed?) or its LS/MD missing.
Usage: pair_check.py   (reads the two pickles + both ntuples)"""
import pickle, collections
import numpy as np
import uproot

SCR = "/tmp/claude-63526/-mnt-data1-kk829-CMSSW-16-1-1-src-RecoTracker-LSTCore-standalone/dbd9af1c-87ff-4b4e-9196-c80ebb15a786/scratchpad/s47/early_md_ls/"
NT = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/"
F = {"s45": NT + "LSTNtuple_s45_rebased_100evt.root", "s44": NT + "LSTNtuple_s44_base_100evt.root"}
exec(open(__file__.replace("pair_check.py", "cross_s44_s45.py")).read().split("c5, K5, k5 =")[0].split('"""', 2)[2])
c5, K5, k5 = load("s45")
c4, K4, k4 = load("s44")
m4 = {k: i for i, k in enumerate(k4)}
targets = [i for i in range(len(k5)) if K5[i] != "has_gT5" and K4[m4[k5[i]]] == "has_gT5"]
print("targets (S45 no gT5, S44 gT5):", len(targets), " core:", int(c5["core"][targets].sum()))
BR = ["sim_pt", "sim_eta", "sim_t5IdxAll", "sim_t5IdxAllFrac", "sim_t3IdxAll", "sim_t3IdxAllFrac", "t5_t3Idx0", "t5_t3Idx1",
      "t3_lsIdx0", "t3_lsIdx1", "ls_mdIdx0", "ls_mdIdx1", "md_anchor_x", "md_anchor_y", "md_anchor_z", "md_detId"]
T = {k: uproot.open(v)["tree"].arrays(BR, library="np") for k, v in F.items()}
fp = {k: np.array([round(float(T[k]["sim_pt"][e][:3].sum()), 3) for e in range(len(T[k]["sim_pt"]))]) for k in T}


def mdkey(A, e, m):
    return (int(A["md_detId"][e][m]), round(float(A["md_anchor_x"][e][m]), 3), round(float(A["md_anchor_y"][e][m]), 3),
            round(float(A["md_anchor_z"][e][m]), 3))


def t3mds(A, e, t):
    l0, l1 = A["t3_lsIdx0"][e][t], A["t3_lsIdx1"][e][t]
    return (A["ls_mdIdx0"][e][l0], A["ls_mdIdx1"][e][l0], A["ls_mdIdx1"][e][l1])


res = collections.Counter()
det = []
cache = {}
for i in targets:
    evfp, pt, eta = k5[i]
    e5 = int(np.nonzero(fp["s45"] == evfp)[0][0]); e4 = int(np.nonzero(fp["s44"] == evfp)[0][0])
    A5, A4 = T["s45"], T["s44"]
    s5 = int(np.argmin(np.abs(A5["sim_pt"][e5].astype(float) - pt) + np.abs(A5["sim_eta"][e5].astype(float) - eta)))
    s4 = int(np.argmin(np.abs(A4["sim_pt"][e4].astype(float) - pt) + np.abs(A4["sim_eta"][e4].astype(float) - eta)))
    if e5 not in cache:
        t3k = {}
        for t in range(len(A5["t3_lsIdx0"][e5])):
            t3k[tuple(mdkey(A5, e5, m) for m in t3mds(A5, e5, t))] = t
        t5p = set(zip(A5["t5_t3Idx0"][e5].tolist(), A5["t5_t3Idx1"][e5].tolist()))
        mdk = {mdkey(A5, e5, m) for m in range(len(A5["md_detId"][e5]))}
        cache[e5] = (t3k, t5p, mdk)
    t3k, t5p, mdk = cache[e5]
    idx, fr = A4["sim_t5IdxAll"][e4][s4], A4["sim_t5IdxAllFrac"][e4][s4]
    outcomes = set()
    for t5 in np.asarray(idx)[np.asarray(fr) >= 0.75]:
        a, b = A4["t5_t3Idx0"][e4][t5], A4["t5_t3Idx1"][e4][t5]
        ka = tuple(mdkey(A4, e4, m) for m in t3mds(A4, e4, a)); kb = tuple(mdkey(A4, e4, m) for m in t3mds(A4, e4, b))
        ta, tb = t3k.get(ka), t3k.get(kb)
        if ta is not None and tb is not None:
            outcomes.add("bothT3_present_T5rejected" if (ta, tb) not in t5p else "T5_present?")
        else:
            miss = [k for k, t in ((ka, ta), (kb, tb)) if t is None]
            mdmiss = any(m not in mdk for k in miss for m in k)
            outcomes.add("T3_missing(MD missing)" if mdmiss else "T3_missing(MDs present)")
    # track-level: best outcome (if any S44 pair is re-found with both T3s, the loss is at T5 creation)
    o = ("bothT3_present_T5rejected" if "bothT3_present_T5rejected" in outcomes else
         "T3_missing(MDs present)" if "T3_missing(MDs present)" in outcomes else sorted(outcomes)[0] if outcomes else "none")
    res[(bool(c5["core"][i]), K5[i], c5["bucket"][i], o)] += 1
print(f"{'core':5s} {'S45 class':20s} {'bucket':16s} outcome")
for k, v in sorted(res.items()):
    print(f"{str(k[0]):5s} {k[1]:20s} {k[2]:16s} {k[3]:28s} {v}")

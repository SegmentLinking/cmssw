#!/usr/bin/env python3
"""Cross-map per-track early-stage classes between the S44 (old code) and S45 (current) ntuples.
Tracks are matched by (event fingerprint = sum of first 3 sim_pt, sim pt, sim eta).
Old code applied the (loose) T5 DNN first and then the same dBeta + T5 r-z cuts (Quintuplet.h @7c9dde8e9cc vs HEAD:
dBeta selector unchanged, passT5RZConstraint return thresholds unchanged, only tightCutFlag removed), and stored every
T5 that passed. So an S45 'F_triedReject' track that HAS a genuine T5 in S44 lost it to the new T5 DNN, not to dBeta/r-z."""
import pickle, collections, sys
import numpy as np


def cls(i):  # same as no_t5_attrib.py
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


SCR = "/tmp/claude-63526/-mnt-data1-kk829-CMSSW-16-1-1-src-RecoTracker-LSTCore-standalone/dbd9af1c-87ff-4b4e-9196-c80ebb15a786/scratchpad/s47/early_md_ls/"


def load(tag):
    c = pickle.load(open(SCR + f"stage_survival_{tag}.pkl", "rb"))
    globals()["c"] = c
    K = np.array([cls(i) for i in range(len(c["pt"]))])
    key = [(round(float(e), 3), round(float(p), 4), round(float(h), 4)) for e, p, h in zip(c["evt"], c["pt"], c["eta"])]
    return c, K, key


c5, K5, k5 = load("s45")
c4, K4, k4 = load("s44")
m4 = {k: i for i, k in enumerate(k4)}
print("matched tracks", sum(k in m4 for k in k5), "of", len(k5))
core = c5["core"].astype(bool)
for sub, lab in [(core, "core all"), (core & ~c5["matched"].astype(bool), "core FAIL"),
                 (core & (c5["bucket"] == "no_T5"), "core no_T5 bucket")]:
    t = collections.Counter()
    for i in np.nonzero(sub)[0]:
        j = m4.get(k5[i])
        t[(K5[i], K4[j] if j is not None else "unmatched")] += 1
    print(f"-- {lab}: (S45 class -> S44 class) for S45 tracks without genuine T5")
    for (a, b), v in sorted(t.items()):
        if a != "has_gT5":
            print(f"   {a:20s} -> {b:20s} {v}")

# S45 core failures that lost the genuine T5 vs S44: were they TC-matched in S44, and what S44 bucket?
lost = [i for i in np.nonzero(core & ~c5["matched"].astype(bool))[0] if K5[i] != "has_gT5" and K4[m4[k5[i]]] == "has_gT5"]
print("\n-- S45 core failures that had a genuine T5 in S44:", len(lost))
print("   S45 bucket:", dict(collections.Counter(c5["bucket"][lost].tolist())))
print("   S44 bucket:", dict(collections.Counter(c4["bucket"][[m4[k5[i]] for i in lost]].tolist())))
print("   pT quantiles (10/50/90):", np.round(np.quantile(c5["pt"][lost], [.1, .5, .9]), 1))

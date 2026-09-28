#!/usr/bin/env python3
"""Sanity check of the CC:T5_embed class: is there a T5 TC within dR2<0.02 of each such pLS (TrackCandidate.h:319-331)?
Which sim do those T5 TCs match, and does the victim have a pass1 near-tie (possible mis-emulation)?"""
import os, pickle, collections, numpy as np, uproot
SD = os.path.dirname(os.path.abspath(__file__))
BASE = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/"
C = pickle.load(open(os.path.join(SD, "tcstage.pkl"), "rb"))
L = uproot.open(BASE + "Ntuple-files/LSTNtuple_s45_rebased_100evt.root:tree").arrays(
    ["tc_type", "tc_t5Idx", "t5_eta", "t5_phi", "tc_simIdx", "tc_isFake", "pLS_eta", "pLS_phi", "pLS_pt", "sim_pt", "tc_simIdxAll", "tc_simIdxAllFrac"], library="np")
out = collections.Counter(); ex = []
for r in C:
    if r["is_tc"] or r["bit1"] or not (r["isdup"] & 1) or r["ov"]:
        continue
    le, p = r["le"], r["p"]
    e1, f1 = L["pLS_eta"][le][p], L["pLS_phi"][le][p]
    near = []
    for k in np.nonzero(L["tc_type"][le] == 4)[0]:
        t5 = L["tc_t5Idx"][le][k]
        de = e1 - L["t5_eta"][le][t5]; dp = (f1 - L["t5_phi"][le][t5] + np.pi) % (2 * np.pi) - np.pi
        if de * de + dp * dp < 0.02:
            sims_part = [(int(s), round(float(f), 2)) for s, f in zip(L["tc_simIdxAll"][le][k], L["tc_simIdxAllFrac"][le][k])]
            near.append((int(k), bool(L["tc_isFake"][le][k]), int(L["tc_simIdx"][le][k]), set(r["sims"]) & set(s for s, _ in sims_part)))
    key = ("no_T5TC_within_dR" if not near else "T5TC_near"), ("sim_noTC" if r["sims"] and not any(r["sim_hastc"]) else ("fake" if not r["sims"] else "sim_hasTC"))
    out[key] += 1
    if near and key[1] == "sim_noTC":
        out[("near_T5TC_matched_other_sim" if any(not f for _, f, _, _ in near) else "near_T5TC_all_fake", "")] += 1
        out[("near_T5TC_shares_victim_sim(partial)" if any(x for *_, x in near) else "near_T5TC_no_victim_sim", "")] += 1
        out[("nNearT5TC", min(len(near), 5))] += 1
for k, v in sorted(out.items(), key=str):
    print(k, v)

# dR2 separation: same-sim T5 TC (legit dup) vs nearest T5 TC for sim_noTC victims
import math
sim_tc = uproot.open(BASE + "Ntuple-files/LSTNtuple_s45_rebased_100evt.root:tree").arrays(["sim_tcIdxAll"], library="np")["sim_tcIdxAll"]
legit, victim = [], []
for r in C:
    if r["is_tc"] or r["bit1"] or not (r["isdup"] & 1) or r["ov"]:
        continue
    le, p = r["le"], r["p"]
    e1, f1 = L["pLS_eta"][le][p], L["pLS_phi"][le][p]
    best_same, best_any = 9, 9
    for k in np.nonzero(L["tc_type"][le] == 4)[0]:
        t5 = L["tc_t5Idx"][le][k]
        de = e1 - L["t5_eta"][le][t5]; dp = (f1 - L["t5_phi"][le][t5] + np.pi) % (2 * np.pi) - np.pi
        d2 = de * de + dp * dp
        best_any = min(best_any, d2)
        if int(L["tc_simIdx"][le][k]) in r["sims"]:
            best_same = min(best_same, d2)
    if r["sims"] and any(r["sim_hastc"]):
        if best_same < 9: legit.append(best_same)
    elif r["sims"]:
        victim.append(best_any)
legit, victim = np.array(legit), np.array(victim)
print("legit (same-sim T5 TC) n", len(legit), "dR2 quantiles", np.round(np.quantile(legit, [0.5, 0.9, 0.99, 1.0]), 6) if len(legit) else None)
print("victims (nearest T5 TC) n", len(victim), "dR2 quantiles", np.round(np.quantile(victim, [0, 0.1, 0.5]), 6))
for cut in [1e-4, 3e-4, 1e-3, 3e-3]:
    print(f"  embed flag only if dR2<{cut}: legit kept-flagged {np.sum(legit < cut)}/{len(legit)}, victims rescued {np.sum(victim >= cut)}/{len(victim)}")

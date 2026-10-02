#!/usr/bin/env python3
"""Counterfactual: pass1-alive TRIPLET pLS admitted as pLS TCs, but only through the same cleaning the quads get
(CrossCleanpLS hit-overlap/self/dR vs pT5/pT3 TCs, TrackCandidate.h:319-366; CheckHitspLS pass 2 vs quads & triplets
emulated with >=1 shared hit or dR2<1e-5). T5-embed cut: bounds (optimistic = never flags; pessimistic = flags if any
T5 TC within dR2<0.02). 100 evt, current code."""
import os, sys, collections, numpy as np, uproot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pls_common import *
BASE = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/"
lb = ["sim_q", "sim_pt", "sim_eta", "sim_vx", "sim_vy", "sim_vz", "sim_genjet_idx", "sim_genjet_deltaR", "genjet_pt",
      "genjet_eta", "sim_tcIdx", "pLS_isDup", "pLS_phi", "pLS_eta", "pLS_simIdxAll", "pLS_simIdxAllFrac", "pT5_plsIdx",
      "pT3_plsIdx", "tc_type", "tc_pt5Idx", "tc_pt3Idx", "tc_t5Idx", "t5_eta", "t5_phi"]
L = uproot.open(BASE + "Ntuple-files/LSTNtuple_s45_rebased_100evt.root:tree").arrays(lb, library="np")
ib = ["sim_pt", "sim_bunchCrossing", "sim_event", "see_algo", "see_hitIdx", "see_hitType", "see_stateTrajGlbPx",
      "see_stateTrajGlbPy", "see_stateTrajGlbPz", "see_stateTrajGlbX", "see_stateTrajGlbY", "see_stateTrajGlbZ",
      "see_px", "see_py", "see_pz", "see_dxy", "see_dz", "see_ptErr", "pix_simHitIdx", "ph2_simHitIdx", "simhit_simTrkIdx"]
I = uproot.open(BASE + "trackingNtuple-100.root:trackingNtuple/tree").arrays(ib, library="np")
l2i = event_map(L["sim_pt"], I["sim_pt"], I["sim_bunchCrossing"], I["sim_event"])
res = {m: collections.Counter() for m in ["opt", "pess"]}
newcore = {m: set() for m in res}; newden = {m: set() for m in res}; newhi = {m: set() for m in res}
for le in range(100):
    seeds = build_seeds({k: I[k][l2i[le]] for k in ib})
    li = [s for s, sd in enumerate(seeds) if sd["lst"]]
    ph = [seeds[s]["ph"] for s in li]; quad = np.array([seeds[s]["quad"] for s in li])
    score = np.array([seeds[s]["score"] for s in li], dtype=np.float32)
    eta = np.array([seeds[s]["eta"] for s in li]); phi = L["pLS_phi"][le].astype(float)
    pairs = pair_list(eta)
    d1, _ = checkhits_pass1(ph, quad, score, pairs, False)
    isdup = L["pLS_isDup"][le].astype(int)
    alive = ~d1
    # pass 2 including triplets: remove lower-priority (quad first, then score) of any alive pair sharing a hit
    d2 = np.zeros(len(li), bool)
    for i, j in pairs:
        if d1[i] or d1[j]:
            continue
        if any(h in ph[j] for h in ph[i]) or (eta[i] - eta[j]) ** 2 + ((phi[i] - phi[j] + np.pi) % (2 * np.pi) - np.pi) ** 2 < 1e-5:
            qd = int(quad[i]) - int(quad[j]); sd = score[i] - score[j]
            rm = j if (qd > 0 or (qd == 0 and sd < 0)) else i
            d2[rm] = True
    tctype = L["tc_type"][le]
    pix_tc = [int(L["pT5_plsIdx"][le][L["tc_pt5Idx"][le][k]]) for k in np.nonzero(tctype == 7)[0]] + \
             [int(L["pT3_plsIdx"][le][L["tc_pt3Idx"][le][k]]) for k in np.nonzero(tctype == 5)[0]]
    t5tc = [(L["t5_eta"][le][L["tc_t5Idx"][le][k]], L["t5_phi"][le][L["tc_t5Idx"][le][k]]) for k in np.nonzero(tctype == 4)[0]]
    stc = L["sim_tcIdx"][le]
    q = L["sim_q"][le]; spt = L["sim_pt"][le].astype(float)
    gj = L["sim_genjet_idx"][le].astype(int); gpt = L["genjet_pt"][le]; geta = L["genjet_eta"][le]
    gc = np.clip(gj, 0, max(len(gpt) - 1, 0))
    gpt_s = gpt[gc] if len(gpt) else np.zeros_like(spt); geta_s = geta[gc] if len(geta) else np.zeros_like(spt)
    dr = L["sim_genjet_deltaR"][le]
    den = ((q != 0) & (spt > 0.8) & (np.abs(L["sim_eta"][le]) < 4.5) & (np.abs(L["sim_vz"][le]) < 30) &
           (np.hypot(L["sim_vx"][le], L["sim_vy"][le]) < 2.5))
    core = den & (gj >= 0) & (gpt_s > 1000) & (np.abs(geta_s) < 2.5) & (dr >= 0) & (dr < 0.02)
    for p in range(len(li)):
        if quad[p] or d1[p] or d2[p]:
            continue
        if p in pix_tc:
            continue
        if any(len(set(ph[p]) & set(ph[x])) > 0 or (eta[p] - eta[x]) ** 2 + ((phi[p] - phi[x] + np.pi) % (2 * np.pi) - np.pi) ** 2 < 1e-6 for x in pix_tc):
            continue
        near = any((eta[p] - e) ** 2 + ((phi[p] - f + np.pi) % (2 * np.pi) - np.pi) ** 2 < 0.02 for e, f in t5tc)
        sims = [int(s) for s, f in zip(L["pLS_simIdxAll"][le][p], L["pLS_simIdxAllFrac"][le][p]) if f > 0.75]
        for m in res:
            if m == "pess" and near:
                continue
            res[m]["extraTC"] += 1
            if not sims: res[m]["fake"] += 1
            elif any(stc[s] >= 0 for s in sims): res[m]["dup(sim has TC)"] += 1
            else:
                for s in sims:
                    if den[s]: newden[m].add((le, s))
                    if core[s]: newcore[m].add((le, s))
                    if core[s] and spt[s] > 100: newhi[m].add((le, s))
import uproot as _u
_S = _u.open(BASE + "Ntuple-files/LSTNtuple_s45_rebased_100evt.root:tree").arrays(["sim_eta", "sim_pt"], library="np")
for m in res:
    ae = np.array([abs(_S["sim_eta"][le][s]) for le, s in newden[m]]); pt = np.array([_S["sim_pt"][le][s] for le, s in newden[m]])
    print(m, "new den sims |eta|>2.5:", int((ae > 2.5).sum()), " |eta|<2.5:", int((ae <= 2.5).sum()), " pT<2:", int((pt < 2).sum()))
    print(m, dict(res[m]), "new den sims", len(newden[m]), "new core sims", len(newcore[m]), "core pT>100", len(newhi[m]))

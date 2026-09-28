#!/usr/bin/env python3
"""
S47 early_pixel: TC-stage pLS losses on the 100-evt --allobj ntuple.
 - emulate CheckHitspLS pass 2 (Kernels.h:789-858, secondpass=true) -> validate vs pLS_isDup bit1
 - attribute CrossCleanpLS flags (TrackCandidate.h:280-373; value 1 without emulated pass1) to
   hit overlap with a pT5/pT3 TC's pLS, dR2<1e-6, or (residual) the T5 embedding cut
 - counterfactual: which extra pLS TCs appear under candidate rules, and are they new matches
   (efficiency), duplicates (sim already has a TC) or fakes. Global (all sims) and jet-core.
Read-only. Output: tcstage.pkl
"""
import os, sys, pickle, collections
import numpy as np
import uproot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pls_common import *

SD = os.path.dirname(os.path.abspath(__file__))
BASE = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/"
LSTF = BASE + "Ntuple-files/LSTNtuple_s45_rebased_100evt.root"
INF = BASE + "trackingNtuple-100.root"

lb = ["sim_q", "sim_pt", "sim_eta", "sim_vx", "sim_vy", "sim_vz", "sim_genjet_idx", "sim_genjet_deltaR",
      "genjet_pt", "genjet_eta", "sim_tcIdx", "sim_trkNtupIdx",
      "pLS_isDup", "pLS_pt", "pLS_eta", "pLS_phi", "pLS_simIdxAll", "pLS_simIdxAllFrac",
      "pT5_plsIdx", "pT3_plsIdx", "tc_type", "tc_plsIdx", "tc_pt5Idx", "tc_pt3Idx", "tc_isFake", "tc_simIdx"]
L = uproot.open(LSTF + ":tree").arrays(lb, library="np")
ib = ["sim_pt", "sim_bunchCrossing", "sim_event", "see_algo", "see_hitIdx", "see_hitType", "see_stateTrajGlbPx",
      "see_stateTrajGlbPy", "see_stateTrajGlbPz", "see_stateTrajGlbX", "see_stateTrajGlbY", "see_stateTrajGlbZ",
      "see_px", "see_py", "see_pz", "see_dxy", "see_dz", "see_ptErr", "pix_simHitIdx", "ph2_simHitIdx",
      "simhit_simTrkIdx"]
I = uproot.open(INF + ":trackingNtuple/tree").arrays(ib, library="np")
l2i = event_map(L["sim_pt"], I["sim_pt"], I["sim_bunchCrossing"], I["sim_event"])

G = collections.Counter()
cand = []  # per pLS candidate records for counterfactuals
for le in range(len(L["sim_pt"])):
    E = {k: I[k][l2i[le]] for k in ib}
    seeds = build_seeds(E)
    lst_idx = [s for s, sd in enumerate(seeds) if sd["lst"]]
    ph = [seeds[s]["ph"] for s in lst_idx]
    quad = np.array([seeds[s]["quad"] for s in lst_idx])
    score = np.array([seeds[s]["score"] for s in lst_idx], dtype=np.float32)
    eta = np.array([seeds[s]["eta"] for s in lst_idx]); phi = L["pLS_phi"][le].astype(np.float64)
    n = len(lst_idx)
    pairs = pair_list(eta)
    d1, _ = checkhits_pass1(ph, quad, score, pairs, False)
    # ---- pass 2 emulation ----
    d2 = np.zeros(n, bool)
    k2 = collections.defaultdict(list)
    for i, j in pairs:
        if not (quad[i] and quad[j]) or d1[i] or d1[j]:
            continue
        shared = any(h in ph[j] for h in ph[i])
        dph = (phi[i] - phi[j] + np.pi) % (2 * np.pi) - np.pi
        dr2 = (eta[i] - eta[j]) ** 2 + dph ** 2
        if shared or dr2 < 1e-5:
            sd = score[i] - score[j]
            rm = j if sd < 0 else i
            w = i if rm == j else j
            d2[rm] = True
            k2[rm].append((w, bool(shared), len(set(ph[i]) & set(ph[j]))))
    # pass 2 as NMS: only a still-unflagged higher-priority quad can remove (variant for Fix "P2-NMS")
    nb2 = collections.defaultdict(list)
    for i, j in pairs:
        if not (quad[i] and quad[j]) or d1[i] or d1[j]:
            continue
        shared = any(h in ph[j] for h in ph[i])
        dph = (phi[i] - phi[j] + np.pi) % (2 * np.pi) - np.pi
        if shared or (eta[i] - eta[j]) ** 2 + dph ** 2 < 1e-5:
            nb2[i].append(j); nb2[j].append(i)
    order2 = sorted([i for i in range(n) if quad[i] and not d1[i]], key=lambda i: (score[i], -i))
    pos2 = {i: k for k, i in enumerate(order2)}
    d2n = np.zeros(n, bool)
    for i in order2:
        for j in nb2[i]:
            if pos2[j] < pos2[i] and not d2n[j]:
                d2n[i] = True
                break
    G["emul_pass2_nms"] += int(d2n.sum())
    isdup = L["pLS_isDup"][le].astype(int)
    b1 = (isdup & 2) > 0
    G["bit1"] += int(b1.sum()); G["emul_pass2"] += int(d2.sum()); G["emul_pass2_and_bit1"] += int((d2 & b1).sum())
    # ---- TCs ----
    tctype = L["tc_type"][le].astype(int)
    pt5pls = L["pT5_plsIdx"][le].astype(int); pt3pls = L["pT3_plsIdx"][le].astype(int)
    tc_pix = []  # (tcidx, type, plsIdx)
    for k, ty in enumerate(tctype):
        if ty == 7:
            tc_pix.append((k, 7, int(pt5pls[L["tc_pt5Idx"][le][k]])))
        elif ty == 5:
            tc_pix.append((k, 5, int(pt3pls[L["tc_pt3Idx"][le][k]])))
    tc_isfake = L["tc_isFake"][le].astype(int); tc_sim = L["tc_simIdx"][le].astype(int)
    # sim info (LST sim index space)
    stc = L["sim_tcIdx"][le]
    q = L["sim_q"][le]; spt = L["sim_pt"][le].astype(float); seta = L["sim_eta"][le]
    gj = L["sim_genjet_idx"][le].astype(int); gpt = L["genjet_pt"][le]; geta = L["genjet_eta"][le]
    gc = np.clip(gj, 0, max(len(gpt) - 1, 0))
    gpt_s = gpt[gc] if len(gpt) else np.zeros_like(spt); geta_s = geta[gc] if len(geta) else np.zeros_like(spt)
    dr = L["sim_genjet_deltaR"][le]
    den = ((q != 0) & (spt > 0.8) & (np.abs(seta) < 4.5) & (np.abs(L["sim_vz"][le]) < 30) &
           (np.hypot(L["sim_vx"][le], L["sim_vy"][le]) < 2.5))
    core = den & (gj >= 0) & (gpt_s > 1000) & (np.abs(geta_s) < 2.5) & (dr >= 0) & (dr < 0.02)
    psim = L["pLS_simIdxAll"][le]; pfr = L["pLS_simIdxAllFrac"][le]
    for p in range(n):
        if not quad[p] or d1[p]:
            continue
        v = int(isdup[p])
        # CrossCleanpLS reasons (hit overlap / dR with pT5,pT3 TC pLS)
        ov = []
        for (k, ty, pl) in tc_pix:
            nsh = len(set(ph[p]) & set(ph[pl])) if pl != p else 99
            dph = (phi[p] - phi[pl] + np.pi) % (2 * np.pi) - np.pi
            dr2 = (eta[p] - eta[pl]) ** 2 + dph ** 2
            if nsh > 0 or dr2 < 1e-6:
                ov.append(dict(tc=k, ty=ty, pl=pl, nsh=nsh, dr2=float(dr2), fake=int(tc_isfake[k]), sim=int(tc_sim[k])))
        cc_flag = (v & 1) and not d1[p]
        sims = [int(s) for s, f in zip(psim[p], pfr[p]) if f > 0.75]
        rec = dict(le=le, p=p, isdup=v, pass2=bool(d2[p]), nms2=bool(d2n[p]), bit1=bool(v & 2), cc=bool(cc_flag), ov=ov,
                   k2=[dict(w=w, shared=sh, nd=nd, w_isdup=int(isdup[w]), w_sims=[int(s) for s, f in zip(psim[w], pfr[w]) if f > 0.75])
                       for (w, sh, nd) in k2[p]],
                   sims=sims, sim_hastc=[bool(stc[s] >= 0) for s in sims], sim_den=[bool(den[s]) for s in sims],
                   sim_core=[bool(core[s]) for s in sims], sim_pt=[float(spt[s]) for s in sims],
                   is_tc=(v == 0))
        cand.append(rec)
    sys.stdout.write(f"\rev {le}"); sys.stdout.flush()
print()
print(dict(G))
pickle.dump(cand, open(os.path.join(SD, "tcstage.pkl"), "wb"))
print("saved", len(cand))

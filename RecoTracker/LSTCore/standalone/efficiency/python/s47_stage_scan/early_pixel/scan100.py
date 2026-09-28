#!/usr/bin/env python3
"""
S47 early_pixel scan on the 100-evt --allobj ntuple (current code, no Fix B).
Per jet-core (dR<0.02) denominator track: funnel bucket (== s46_core_funnel.py), emulated
CheckHitspLS pass 1 (master / Fix B / NMS), CrossCleanpLS contamination of pLS_isDup bit0,
killer details, and input-seed attribution for tracks without a genuine pLS.
Output: core_rows.pkl (+ validation printout). Read-only.
"""
import os, sys, pickle, collections
import numpy as np
import uproot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pls_common import *

SD = os.path.dirname(os.path.abspath(__file__))
BASE = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/"
LSTF = sys.argv[1] if len(sys.argv) > 1 else BASE + "Ntuple-files/LSTNtuple_s45_rebased_100evt.root"
INF = BASE + "trackingNtuple-100.root"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(SD, "core_rows.pkl")
GEN = 0.75

lb = ["sim_q", "sim_pt", "sim_eta", "sim_vx", "sim_vy", "sim_vz", "sim_genjet_idx", "sim_genjet_deltaR",
      "genjet_pt", "genjet_eta", "sim_tcIdx", "sim_trkNtupIdx", "sim_tcIdxAll", "sim_tcIdxAllFrac"]
for o in ["pls", "t3", "t5", "pt3", "pt5", "t4"]:
    lb += [f"sim_{o}IdxAll", f"sim_{o}IdxAllFrac"]
lb += ["pLS_isDup", "pLS_isQuad", "pLS_pt", "pLS_eta", "pT5_plsIdx", "pT5_t5Idx", "pT5_isDupReco", "pT3_plsIdx",
       "tc_type", "tc_plsIdx", "tc_pt5Idx", "tc_pt3Idx", "tc_simIdx", "tc_isFake"]
print("reading LST ntuple...", flush=True)
L = uproot.open(LSTF + ":tree").arrays(lb, library="np")
nL = len(L["sim_pt"])

ib = ["sim_pt", "sim_bunchCrossing", "sim_event", "see_algo", "see_hitIdx", "see_hitType", "see_stateTrajGlbPx",
      "see_stateTrajGlbPy", "see_stateTrajGlbPz", "see_stateTrajGlbX", "see_stateTrajGlbY", "see_stateTrajGlbZ",
      "see_px", "see_py", "see_pz", "see_dxy", "see_dz", "see_ptErr", "pix_simHitIdx", "ph2_simHitIdx",
      "simhit_simTrkIdx", "sim_simHitIdx", "simhit_hitIdx", "simhit_hitType", "pix_layer", "pix_isBarrel"]
print("reading input ntuple...", flush=True)
I = uproot.open(INF + ":trackingNtuple/tree").arrays(ib, library="np")
l2i = event_map(L["sim_pt"], I["sim_pt"], I["sim_bunchCrossing"], I["sim_event"])
print(f"event map {len(l2i)}/{nL}", flush=True)

rows = []
glob = collections.Counter()
for le in range(nL):
    ie = l2i[le]
    E = {k: I[k][ie] for k in ib}
    seeds = build_seeds(E)
    lst_idx = [s for s, sd in enumerate(seeds) if sd["lst"]]
    pos = {s: k for k, s in enumerate(lst_idx)}
    nP = len(L["pLS_pt"][le])
    ok = nP == len(lst_idx) and np.allclose([seeds[s]["ptIn"] for s in lst_idx], L["pLS_pt"][le], rtol=1e-4)
    glob["ev_valid"] += ok
    if not ok:
        print("WARNING: pLS/seed mismatch in event", le, nP, len(lst_idx))
        continue
    ph = [seeds[s]["ph"] for s in lst_idx]
    quad = np.array([seeds[s]["quad"] for s in lst_idx])
    score = np.array([seeds[s]["score"] for s in lst_idx], dtype=np.float32)
    etas = np.array([seeds[s]["eta"] for s in lst_idx])
    pairs = pair_list(etas)
    d_m, k_m = checkhits_pass1(ph, quad, score, pairs, False)
    d_b, k_b = checkhits_pass1(ph, quad, score, pairs, True)
    d_n = checkhits_nms(ph, quad, score, pairs, False)
    d_nb = checkhits_nms(ph, quad, score, pairs, True)
    isdup = L["pLS_isDup"][le].astype(int)
    b0 = (isdup & 1).astype(bool)
    glob["pls"] += nP
    glob["emul_pass1"] += int(d_m.sum())
    glob["ntuple_bit0"] += int(b0.sum())
    glob["emul_not_in_bit0"] += int((d_m & ~b0).sum())
    glob["bit0_not_emul(CrossCleanpLS)"] += int((b0 & ~d_m).sum())
    glob["bit0_not_emul_trip"] += int((b0 & ~d_m & ~quad).sum())
    glob["bit1"] += int(((isdup & 2) > 0).sum())
    glob["fixB_pass1"] += int(d_b.sum())
    glob["nms_pass1"] += int(d_n.sum())
    glob["quad"] += int(quad.sum())
    # objects per pLS
    pt5pls = L["pT5_plsIdx"][le].astype(int); pt5dup = L["pT5_isDupReco"][le].astype(int)
    pt5t5 = L["pT5_t5Idx"][le].astype(int)
    pt3pls = L["pT3_plsIdx"][le].astype(int)
    tcpls = L["tc_plsIdx"][le].astype(int); tctype = L["tc_type"][le].astype(int)
    pls_in_tc = collections.Counter(tcpls[tcpls >= 0].tolist())
    # denominator
    q = L["sim_q"][le]; pt = L["sim_pt"][le].astype(float); eta = L["sim_eta"][le]
    gj = L["sim_genjet_idx"][le].astype(int); gpt = L["genjet_pt"][le]; geta = L["genjet_eta"][le]
    gc = np.clip(gj, 0, max(len(gpt) - 1, 0))
    gpt_s = gpt[gc] if len(gpt) else np.zeros_like(pt); geta_s = geta[gc] if len(geta) else np.zeros_like(pt)
    dr = L["sim_genjet_deltaR"][le]
    sel = ((q != 0) & (pt > 0.8) & (np.abs(eta) < 4.5) & (np.abs(L["sim_vz"][le]) < 30) &
           (np.hypot(L["sim_vx"][le], L["sim_vy"][le]) < 2.5) & (gj >= 0) & (gpt_s > 1000) & (np.abs(geta_s) < 2.5) &
           (dr >= 0) & (dr < 0.02))
    ntup = L["sim_trkNtupIdx"][le].astype(int)
    tcidx = L["sim_tcIdx"][le]
    sim2seeds = collections.defaultdict(list)
    for s, sd in enumerate(seeds):
        for t, f in sd["fr"].items():
            sim2seeds[t].append((s, f))
    for si in np.nonzero(sel)[0]:
        t = int(ntup[si])
        g = {}
        for o in ["pls", "t3", "t5", "pt3", "pt5", "t4"]:
            ii = np.asarray(L[f"sim_{o}IdxAll"][le][si]).astype(int); ff = np.asarray(L[f"sim_{o}IdxAllFrac"][le][si])
            g[o] = ii[ff >= GEN]
        gen = [int(p) for p in g["pls"]]
        emul_gen = sorted(pos[s] for s, f in sim2seeds.get(t, []) if f > GEN and s in pos)
        glob["gen_match_ok"] += (sorted(gen) == emul_gen)
        glob["gen_match_n"] += 1
        pls_ok = [p for p in gen if not (isdup[p] & 1)]
        if len(g["pt5"]):
            b = "pT5_survived" if np.any(L["pT5_isDupReco"][le][g["pt5"]] == 0) else "pT5_all_killed"
        elif not gen:
            b = "no_pLS"
        elif not pls_ok:
            b = "pLS_flagged"
        elif not len(g["t5"]):
            b = "no_T5"
        else:
            b = "pair_failed"
        fail = bool(tcidx[si] < 0)
        # genuine-pLS details
        gdet = []
        for p in gen:
            kl = []
            for (w, npm, nd) in k_m[p]:
                ws = lst_idx[w]
                wfr = seeds[ws]["fr"]
                ftr = wfr.get(t, 0.0)
                other = max([v for k2, v in wfr.items() if k2 != t], default=0.0)
                wcls = ("same_gen" if ftr > GEN else "other_gen" if other > GEN else
                        f"fake_{int(round(ftr * seeds[ws]['n']))}of{seeds[ws]['n']}_this")
                kl.append(dict(w=w, npm=npm, nd=nd, wquad=bool(quad[w]), wcls=wcls, wfr_this=ftr, wfr_other=other,
                               w_pass1=bool(d_m[w]), w_isdup=int(isdup[w]), w_npt5=int(np.sum(pt5pls == w)),
                               w_npt5surv=int(np.sum((pt5pls == w) & (pt5dup == 0))), w_npt3=int(np.sum(pt3pls == w)),
                               w_intc=int(pls_in_tc.get(w, 0)), w_score=float(score[w]), v_score=float(score[p])))
            gdet.append(dict(p=p, quad=bool(quad[p]), isdup=int(isdup[p]), pass1=bool(d_m[p]), fixB=bool(d_b[p]),
                             nms=bool(d_n[p]), nmsB=bool(d_nb[p]), killers=kl, npt5=int(np.sum(pt5pls == p)),
                             npt3=int(np.sum(pt3pls == p)), intc=int(pls_in_tc.get(p, 0)), pt=float(seeds[lst_idx[p]]["ptIn"])))
        # input seed attribution
        ss = sim2seeds.get(t, [])
        best_lst = max([f for s, f in ss if s in pos], default=0.0)
        best_lst_seed = [s for s, f in ss if s in pos and f == best_lst]
        best_lst_nhit = seeds[best_lst_seed[0]]["n"] if best_lst_seed else 0
        gen_in = [(s, f) for s, f in ss if f > GEN]
        # 3/4 LST pLS (frac==0.75) status
        p34 = [pos[s] for s, f in ss if s in pos and abs(f - 0.75) < 1e-6]
        rows.append(dict(le=le, ie=ie, si=int(si), t=t, pt=float(pt[si]), eta=float(eta[si]), dr=float(dr[si]),
                         fail=fail, bucket=b, n_gen=len(gen), gdet=gdet,
                         has={o: int(len(g[o])) for o in g},
                         best_lst=best_lst, best_lst_nhit=best_lst_nhit,
                         gen_in_algos=sorted(set(seeds[s]["algo"] for s, _ in gen_in)),
                         gen_in_lstalgo_ptfail=[s for s, _ in gen_in if seeds[s]["good_algo"] and not seeds[s]["good_pt"]],
                         gen_in_nraw={seeds[s]["algo"]: seeds[s]["nraw"] for s, _ in gen_in},
                         best_by_algo={al: max(f for s, f in ss if seeds[s]["algo"] == al)
                                       for al in set(seeds[s]["algo"] for s, _ in ss)},
                         n_in_seeds=len(ss),
                         p34=[dict(p=p, quad=bool(quad[p]), isdup=int(isdup[p]), pass1=bool(d_m[p]),
                                   npt5=int(np.sum(pt5pls == p)), npt5surv=int(np.sum((pt5pls == p) & (pt5dup == 0))),
                                   intc=int(pls_in_tc.get(p, 0))) for p in p34]))
    sys.stdout.write(f"\rev {le}"); sys.stdout.flush()
print()
for k, v in glob.items():
    print(f"{k:35s} {v}")
pickle.dump(dict(rows=rows, glob=dict(glob)), open(OUT, "wb"))
print("saved", len(rows), "core rows to", OUT)

#!/usr/bin/env python3
"""
S47 early_pixel, 1000 evt: per jet-core track, input-seed attribution + emulated CheckHitspLS pass 1
(master vs Fix B) from trackingNtuple-1000.root, joined with sim_tcIdx from the s46 base/rep1/rep2/fixB
LST ntuples (each mapped to input events by sim_pt fingerprint). Tests whether Fix B's realized gains
sit in the tracks it frees. Output: core1000.pkl in the scratch cache. Read-only.
"""
import os, sys, pickle, collections
import numpy as np
import uproot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pls_common import *

BASE = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/"
CACHE = "/tmp/claude-63526/-mnt-data1-kk829-CMSSW-16-1-1-src-RecoTracker-LSTCore-standalone/dbd9af1c-87ff-4b4e-9196-c80ebb15a786/scratchpad/s47/early_pixel/"
INF = BASE + "trackingNtuple-1000.root"
RUNS = {k: BASE + f"Ntuple-files/LSTNtuple_s46_{k}_1000evt.root" for k in ["base", "base_rep1", "base_rep2", "fixB"]}

lb = ["sim_q", "sim_pt", "sim_eta", "sim_vx", "sim_vy", "sim_vz", "sim_genjet_idx", "sim_genjet_deltaR",
      "genjet_pt", "genjet_eta", "sim_tcIdx", "sim_trkNtupIdx"]
Ls = {k: uproot.open(f + ":tree").arrays(lb, library="np") for k, f in RUNS.items()}
T = uproot.open(INF + ":trackingNtuple/tree")
H = T.arrays(["sim_pt", "sim_bunchCrossing", "sim_event"], library="np")
maps = {k: event_map(L["sim_pt"], H["sim_pt"], H["sim_bunchCrossing"], H["sim_event"]) for k, L in Ls.items()}
for k, m in maps.items():
    print(k, "mapped", len(m), "events", flush=True)
inv = {k: {ie: le for le, ie in m.items()} for k, m in maps.items()}

ib = ["see_algo", "see_hitIdx", "see_hitType", "see_stateTrajGlbPx", "see_stateTrajGlbPy", "see_stateTrajGlbPz",
      "see_stateTrajGlbX", "see_stateTrajGlbY", "see_stateTrajGlbZ", "see_px", "see_py", "see_pz", "see_dxy", "see_dz",
      "see_ptErr", "pix_simHitIdx", "ph2_simHitIdx", "simhit_simTrkIdx"]
rows = []
glob = collections.Counter()
ie0 = 0
for A in T.iterate(ib, step_size=50, library="np"):
    for k in range(len(A["see_algo"])):
        ie = ie0 + k
        if ie not in inv["base"]:
            continue
        E = {b: A[b][k] for b in ib}
        seeds = build_seeds(E)
        lst_idx = [s for s, sd in enumerate(seeds) if sd["lst"]]
        pos = {s: q for q, s in enumerate(lst_idx)}
        ph = [seeds[s]["ph"] for s in lst_idx]
        quad = np.array([seeds[s]["quad"] for s in lst_idx])
        score = np.array([seeds[s]["score"] for s in lst_idx], dtype=np.float32)
        pairs = pair_list([seeds[s]["eta"] for s in lst_idx])
        d_m, k_m = checkhits_pass1(ph, quad, score, pairs, False)
        d_b, _ = checkhits_pass1(ph, quad, score, pairs, True)
        glob["pls"] += len(lst_idx); glob["pass1_master"] += int(d_m.sum()); glob["pass1_fixB"] += int(d_b.sum())
        glob["freed_by_fixB"] += int((d_m & ~d_b).sum()); glob["newly_flagged_by_fixB"] += int((~d_m & d_b).sum())
        glob["freed_by_fixB_trip"] += int((d_m & ~d_b & ~quad).sum())
        sim2seeds = collections.defaultdict(list)
        for s, sd in enumerate(seeds):
            for t, f in sd["fr"].items():
                sim2seeds[t].append((s, f))
        L = Ls["base"]; le = inv["base"][ie]
        q = L["sim_q"][le]; pt = L["sim_pt"][le].astype(float); eta = L["sim_eta"][le]
        gj = L["sim_genjet_idx"][le].astype(int); gpt = L["genjet_pt"][le]; geta = L["genjet_eta"][le]
        gc = np.clip(gj, 0, max(len(gpt) - 1, 0))
        gpt_s = gpt[gc] if len(gpt) else np.zeros_like(pt); geta_s = geta[gc] if len(geta) else np.zeros_like(pt)
        dr = L["sim_genjet_deltaR"][le]
        sel = ((q != 0) & (pt > 0.8) & (np.abs(eta) < 4.5) & (np.abs(L["sim_vz"][le]) < 30) &
               (np.hypot(L["sim_vx"][le], L["sim_vy"][le]) < 2.5) & (gj >= 0) & (gpt_s > 1000) &
               (np.abs(geta_s) < 2.5) & (dr >= 0) & (dr < 0.02))
        ntup = L["sim_trkNtupIdx"][le].astype(int)
        for si in np.nonzero(sel)[0]:
            t = int(ntup[si])
            succ = {}
            for rk in RUNS:
                lr = inv[rk][ie]
                # same sim ordering across runs (same input event); check via trkNtupIdx
                assert int(Ls[rk]["sim_trkNtupIdx"][lr][si]) == t
                succ[rk] = bool(Ls[rk]["sim_tcIdx"][lr][si] >= 0)
            ss = sim2seeds.get(t, [])
            gen = [pos[s] for s, f in ss if f > 0.75 and s in pos]
            gen_in = [(s, f) for s, f in ss if f > 0.75]
            best_lst = max([f for s, f in ss if s in pos], default=0.0)
            kills = [(w, npm, nd, bool(quad[w]), seeds[lst_idx[w]]["fr"].get(t, 0.0)) for p in gen for (w, npm, nd) in k_m[p]]
            rows.append(dict(ie=ie, t=t, pt=float(pt[si]), succ=succ, n_gen=len(gen),
                             gen_quad=[bool(quad[p]) for p in gen], gen_m=[bool(d_m[p]) for p in gen],
                             gen_b=[bool(d_b[p]) for p in gen], kills=kills,
                             gen_in_algos=sorted(set(seeds[s]["algo"] for s, _ in gen_in)),
                             gen_in_lstalgo_ptfail=[s for s, _ in gen_in if seeds[s]["good_algo"] and not seeds[s]["good_pt"]],
                             best_lst=best_lst, n_in_seeds=len(ss),
                             best_by_algo={al: max(f for s, f in ss if seeds[s]["algo"] == al)
                                           for al in set(seeds[s]["algo"] for s, _ in ss)}))
    ie0 += len(A["see_algo"])
    print("events", ie0, flush=True)
print(dict(glob))
pickle.dump(dict(rows=rows, glob=dict(glob)), open(CACHE + "core1000.pkl", "wb"))
print("saved", len(rows))

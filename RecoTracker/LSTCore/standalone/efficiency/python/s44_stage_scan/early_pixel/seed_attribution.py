#!/usr/bin/env python3
"""
Early-pixel-side attribution for jet-core (dR<0.02) LST failures, real-pLS AB-ON 100 evt.

For every core failing sim track, classify the pixel-seed status using BOTH the LST output
ntuple (what LST built) and the INPUT trackingNtuple (what CMSSW supplied), and emulate
LST's prepareInput() seed filter + CheckHitspLS (first pass) exactly, to attribute losses to
 - absent in input vs. dropped by LST (algo filter / pT cut / pLS dedup)
 - matching-definition artifacts (3/4 seeds: frac 0.75 is NOT > 0.75 in matchedSimTrkIdxsAndFracs)

Read-only. Writes pickled per-track table + text summary to this directory.
"""
import uproot, awkward as ak, numpy as np, pickle, collections, sys, os

SD = os.path.dirname(os.path.abspath(__file__))
BASE = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/"
LSTF = BASE + "Ntuple-files/LSTNtuple_realpls_mastercuts_100evt_v2.root"
INF = BASE + "trackingNtuple-100.root"
PTCUT = 0.8
CORE = 0.02

# ---------------- LST output ----------------
lb = ["sim_q", "sim_pt", "sim_eta", "sim_phi", "sim_vx", "sim_vy", "sim_vz", "sim_genjet_idx", "sim_genjet_deltaR",
      "genjet_pt", "genjet_eta", "sim_tcIdx", "sim_trkNtupIdx", "sim_plsIdxAll", "sim_plsIdxAllFrac",
      "sim_pt5IdxAll", "sim_pt5IdxAllFrac", "sim_pt3IdxAll", "sim_pt3IdxAllFrac", "sim_t5IdxAllFrac", "sim_t3IdxAllFrac",
      "pLS_isQuad", "pLS_pt", "pLS_eta", "pT5_plsIdx", "pT5_isDupReco", "pT3_plsIdx", "tc_type", "tc_plsIdx",
      "tc_pt5Idx"]
L = uproot.open(LSTF + ":tree").arrays(lb, library="ak")
nL = len(L["sim_pt"])

# ---------------- input ----------------
ib = ["sim_pt", "sim_bunchCrossing", "sim_event", "sim_seedIdx", "sim_nPixel", "sim_nPixelLay", "sim_pdgId",
      "see_algo", "see_hitIdx", "see_hitType", "see_stateTrajGlbPx", "see_stateTrajGlbPy", "see_stateTrajGlbPz",
      "see_stateTrajGlbX", "see_stateTrajGlbY", "see_stateTrajGlbZ", "see_px", "see_py", "see_pz", "see_dxy", "see_dz",
      "see_ptErr", "pix_simHitIdx", "simhit_simTrkIdx", "pix_layer", "pix_isBarrel", "pix_chargeFraction",
      "sim_simHitIdx", "simhit_hitIdx", "simhit_hitType"]
I = uproot.open(INF + ":trackingNtuple/tree").arrays(ib, library="ak")
nI = len(I["sim_pt"])

# event map by accepted-sim sim_pt fingerprint
def fp(x):
    return np.asarray(x, dtype=np.float32).tobytes()
inmap = {}
for e in range(nI):
    acc = (np.asarray(I["sim_bunchCrossing"][e]) == 0) & (np.asarray(I["sim_event"][e]) == 0)
    inmap[fp(np.asarray(I["sim_pt"][e])[acc])] = e
l2i = {}
for e in range(nL):
    k = fp(L["sim_pt"][e])
    if k in inmap:
        l2i[e] = inmap[k]
print(f"LST->input event map: {len(l2i)}/{nL}; permuted: {sum(1 for a,b in l2i.items() if a!=b)}")


def dphi(a, b):
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def seed_table(ie):
    """Return per-seed arrays for input event ie + sim match info (LST convention)."""
    algo = np.asarray(I["see_algo"][ie])
    hidx = I["see_hitIdx"][ie]; htyp = I["see_hitType"][ie]
    pxl = np.asarray(I["see_stateTrajGlbPx"][ie]); pyl = np.asarray(I["see_stateTrajGlbPy"][ie]); pzl = np.asarray(I["see_stateTrajGlbPz"][ie])
    ptIn = np.hypot(pxl, pyl)
    etaLH = np.arcsinh(pzl / ptIn)
    ptErr = np.asarray(I["see_ptErr"][ie])
    pix_sh = I["pix_simHitIdx"][ie]; sh_trk = np.asarray(I["simhit_simTrkIdx"][ie])
    nS = len(algo)
    seeds = []
    for s in range(nS):
        hi = list(np.asarray(hidx[s])); ht = list(np.asarray(htyp[s]))
        # unique (idx,type) like matchedSimTrkIdxsAndFracs
        uniq = []
        for a, b in zip(hi, ht):
            if (a, b) not in uniq:
                uniq.append((a, b))
        cnt = collections.Counter()
        for a, b in uniq:
            if b == 0:
                sims = set(int(sh_trk[k]) for k in np.asarray(pix_sh[a]))
            else:
                sims = set()
            for t in sims:
                cnt[t] += 1
        n = len(uniq)
        fr = {t: c / n for t, c in cnt.items() if t >= 0}
        seeds.append(dict(algo=int(algo[s]), hits=hi, types=ht, n=n, fr=fr, ptIn=float(ptIn[s]), ptErr=float(ptErr[s]),
                          eta=float(etaLH[s]), phi=float(np.arctan2(pyl[s], pxl[s]))))
    return seeds


def lst_pass(sd):
    good_algo = sd["algo"] in (4, 22)
    good_pt = sd["ptIn"] > PTCUT - 2 * sd["ptErr"]
    return good_algo, good_pt


def pls_score(ie, s):
    """score_lsq exactly as AddPixelSegmentToEventKernel (slope from PCA eta, anchors r3PCA, r3LH)."""
    px = I["see_px"][ie][s]; py = I["see_py"][ie][s]; pz = I["see_pz"][ie][s]
    dxy = I["see_dxy"][ie][s]; dz = I["see_dz"][ie][s]
    pt = np.hypot(px, py); p = np.sqrt(pt * pt + pz * pz)
    vz = dz * pt * pt / p / p
    vx = -dxy * py / pt - px / p * pz / p * dz
    vy = dxy * px / pt - py / p * pz / p * dz
    eta = np.arcsinh(pz / pt)
    slope = np.sinh(np.float32(eta))
    intercept = vz - slope * np.hypot(vx, vy)
    X = I["see_stateTrajGlbX"][ie][s]; Y = I["see_stateTrajGlbY"][ie][s]; Z = I["see_stateTrajGlbZ"][ie][s]
    sc = (np.hypot(X, Y) * slope + intercept) - Z
    return float(sc * sc)


def emulate_checkhits(ie, seeds, lst_idx):
    """First pass of CheckHitspLS on the LST-accepted seeds (in LST order). Returns isDup array + killer lists."""
    n = len(lst_idx)
    quad = np.array([len(seeds[s]["hits"]) > 3 for s in lst_idx])
    eta = np.array([seeds[s]["eta"] for s in lst_idx])
    score = np.array([pls_score(ie, s) for s in lst_idx])
    ph = []
    for s in lst_idx:
        h = seeds[s]["hits"]
        h3 = h[3] if len(h) > 3 else h[2]
        ph.append((h[0], h[2], h[1], h3))  # pLSHitsIdxs order {inner anchor, outer anchor, inner outer, outer outer}
    isdup = np.zeros(n, bool)
    killers = collections.defaultdict(list)  # victim -> list of (winner, npMatched, nDistinctShared)
    order = np.argsort(eta)
    es = eta[order]
    for a in range(n):
        ix = order[a]
        # candidate partners within |deta|<=0.1
        hi = np.searchsorted(es, es[a] + 0.1, side="right")
        for b in range(a + 1, hi):
            jx = order[b]
            i, j = (ix, jx) if ix < jx else (jx, ix)  # kernel: ix < jx
            p1, p2 = ph[i], ph[j]
            npm = sum(1 for h in p1 if h in p2)
            if npm < 3:
                continue
            qd = int(quad[i]) - int(quad[j]); sd = score[i] - score[j]
            if qd > 0: rm = j
            elif qd < 0: rm = i
            elif sd < 0: rm = j
            elif sd > 0: rm = i
            else: rm = i
            win = j if rm == i else i
            isdup[rm] = True
            ndist = len(set(p1) & set(p2))
            killers[rm].append((win, npm, ndist))
    return isdup, killers, quad


# ---------------- main loop ----------------
rows = []
val = []
dedup_stats = collections.Counter()
for le in range(nL):
    ie = l2i[le]
    a = L
    gidx = np.asarray(a["sim_genjet_idx"][le])
    gpt = np.asarray(a["genjet_pt"][le]); geta = np.asarray(a["genjet_eta"][le])
    safe = np.where(gidx < 0, 0, gidx)
    gjpt = gpt[safe] if len(gpt) else np.zeros_like(gidx, dtype=float)
    gjeta = geta[safe] if len(geta) else np.zeros_like(gidx, dtype=float)
    q = np.asarray(a["sim_q"][le]); pt = np.asarray(a["sim_pt"][le]); eta = np.asarray(a["sim_eta"][le])
    vx = np.asarray(a["sim_vx"][le]); vy = np.asarray(a["sim_vy"][le]); vz = np.asarray(a["sim_vz"][le])
    dR = np.asarray(a["sim_genjet_deltaR"][le]); tc = np.asarray(a["sim_tcIdx"][le])
    den = (q != 0) & (pt > 0.9) & (abs(eta) < 4.5) & (abs(vz) < 30) & (np.hypot(vx, vy) < 2.5) & (gidx >= 0) & (gjpt > 1000) & (abs(gjeta) < 2.5)
    core = den & (dR < CORE)
    seeds = seed_table(ie)
    lst_idx = [s for s, sd in enumerate(seeds) if all(lst_pass(sd))]
    nPLS_ntuple = len(a["pLS_pt"][le])
    val.append((len(lst_idx), nPLS_ntuple))
    isdup, killers, quadarr = emulate_checkhits(ie, seeds, lst_idx)
    pos_in_lst = {s: k for k, s in enumerate(lst_idx)}
    pls_quad = np.asarray(a["pLS_isQuad"][le])
    ntup = np.asarray(a["sim_trkNtupIdx"][le])
    # per-sim seeds (input) with frac
    sim2seeds = collections.defaultdict(list)
    for s, sd in enumerate(seeds):
        for t, f in sd["fr"].items():
            sim2seeds[t].append((s, f))
    # pixel hits of each sim (reco pix hits with a simhit from this track)
    pix_sh = I["pix_simHitIdx"][ie]; sh_trk = np.asarray(I["simhit_simTrkIdx"][ie])
    pix_layer = np.asarray(I["pix_layer"][ie]); pix_barrel = np.asarray(I["pix_isBarrel"][ie])
    sim_simhit = I["sim_simHitIdx"][ie]; sh_hitIdx = I["simhit_hitIdx"][ie]; sh_hitType = I["simhit_hitType"][ie]
    sim_seedIdx = I["sim_seedIdx"][ie]
    # pT5 info
    pt5pls = np.asarray(a["pT5_plsIdx"][le]); pt5dup = np.asarray(a["pT5_isDupReco"][le])
    pt3pls = np.asarray(a["pT3_plsIdx"][le])
    for si in np.where(core)[0]:
        t = int(ntup[si])
        plsI = np.asarray(a["sim_plsIdxAll"][le][si]); plsF = np.asarray(a["sim_plsIdxAllFrac"][le][si])
        gen = plsI[(plsF >= 0.75) & (plsI >= 0)]
        has_pt5 = bool(np.any(np.asarray(a["sim_pt5IdxAllFrac"][le][si]) >= 0.75))
        has_pt3 = bool(np.any(np.asarray(a["sim_pt3IdxAllFrac"][le][si]) >= 0.75))
        # input seeds for this track
        ss = sim2seeds.get(t, [])
        best_all = max([f for _, f in ss], default=0.0)
        gen_in = [(s, f) for s, f in ss if f > 0.75]
        gen_in_lst = [(s, f) for s, f in gen_in if s in pos_in_lst]
        best_lst = max([f for s, f in ss if s in pos_in_lst], default=0.0)
        best_lst_n = None
        for s, f in ss:
            if s in pos_in_lst and f == best_lst:
                best_lst_n = (round(f * seeds[s]["n"]), seeds[s]["n"])
        # sim pixel content
        pixhits = set()
        for k in np.asarray(sim_simhit[t]):
            for hh, tt in zip(np.asarray(sh_hitIdx[k]), np.asarray(sh_hitType[k])):
                if int(tt) == 0 and int(hh) >= 0:
                    pixhits.add(int(hh))
        # reco pix hits carrying this sim (more robust: scan via simhits)
        layers = set((int(pix_barrel[h]), int(pix_layer[h])) for h in pixhits)
        shared = 0
        for h in pixhits:
            sims = set(int(sh_trk[k]) for k in np.asarray(pix_sh[h]))
            if len(sims - {t}) > 0:
                shared += 1
        # dedup emulation for genuine LST pLS
        gen_lst_pos = [pos_in_lst[s] for s, _ in gen_in_lst]
        all_gen_killed = len(gen_lst_pos) > 0 and all(isdup[p] for p in gen_lst_pos)
        killer_info = []
        if all_gen_killed:
            for p in gen_lst_pos:
                for (w, npm, nd) in killers[p]:
                    ws = lst_idx[w]
                    wfr = seeds[ws]["fr"]
                    wlabel = "same" if wfr.get(t, 0) > 0.75 else ("othergenuine" if any(v > 0.75 for v in wfr.values()) else "fake")
                    killer_info.append((wlabel, npm, nd, bool(quadarr[p]), bool(quadarr[w])))
        # usable seeds: LST pLS with frac>=0.5 to this sim (a pT5 from it + genuine T5 would be a genuine TC)
        usable = []
        for s, f in ss:
            if s in pos_in_lst and f >= 0.5 - 1e-6:
                p = pos_in_lst[s]
                kl = []
                for (w, npm, nd) in killers[p]:
                    kl.append((round(seeds[lst_idx[w]]["fr"].get(t, 0.0), 3),
                               any(v > 0.75 for v in seeds[lst_idx[w]]["fr"].values()), npm, nd, bool(quadarr[w]),
                               bool(isdup[w])))
                usable.append(dict(pos=p, frac=round(f, 3), n=seeds[s]["n"], quad=bool(quadarr[p]), isdup=bool(isdup[p]),
                                   killers=kl, n_pt5=int(np.sum(pt5pls == p)), n_pt5_surv=int(np.sum((pt5pls == p) & (pt5dup == 0))),
                                   n_pt3=int(np.sum(pt3pls == p)), algo=seeds[s]["algo"]))
        rows.append(dict(le=le, ie=ie, si=int(si), simIdx=t, pt=float(pt[si]), eta=float(eta[si]), dR=float(dR[si]),
                         pdg=int(I["sim_pdgId"][ie][t]), fail=bool(tc[si] < 0), has_gen_pls=len(gen) > 0,
                         gen_pls_quad=[bool(pls_quad[g]) for g in gen], has_pt5=has_pt5, has_pt3=has_pt3,
                         n_in_seeds_any=len(ss), best_all=best_all,
                         gen_in_algos=sorted(set(seeds[s]["algo"] for s, _ in gen_in)),
                         gen_in_pass=[lst_pass(seeds[s]) for s, _ in gen_in],
                         n_gen_in_lst=len(gen_in_lst), best_lst=best_lst, best_lst_n=best_lst_n,
                         best_by_algo={al: max([f for s, f in ss if seeds[s]["algo"] == al], default=0) for al in set(seeds[s]["algo"] for s, _ in ss)},
                         npixhits=len(pixhits), npixlayers=len(layers), nshared_pix=shared,
                         cmssw_seedIdx=list(np.asarray(sim_seedIdx[t])),
                         all_gen_killed=all_gen_killed, killer_info=killer_info,
                         gen_isdup=[bool(isdup[p]) for p in gen_lst_pos],
                         usable=usable, has_t5=bool(np.any(np.asarray(a["sim_t5IdxAllFrac"][le][si]) >= 0.75)),
                         has_t3=bool(np.any(np.asarray(a["sim_t3IdxAllFrac"][le][si]) >= 0.75))))
    # global dedup stats (all LST pLS)
    for p in range(len(lst_idx)):
        if isdup[p]:
            dedup_stats["killed"] += 1
            if all(nd < 3 for (_, _, nd) in killers[p]):
                dedup_stats["killed_only_by_doublecount"] += 1
            if not quadarr[p]:
                dedup_stats["killed_trip"] += 1
        dedup_stats["total"] += 1
        dedup_stats["quad" if quadarr[p] else "trip"] += 1
    sys.stdout.write(f"\rev {le}"); sys.stdout.flush()

print()
v = np.array(val)
print("validation: emulated #pLS == ntuple #pLS in", int(np.sum(v[:, 0] == v[:, 1])), "/", len(v), " sums", v.sum(0))
print("dedup stats (all LST pLS, 100 evt):", dict(dedup_stats))
pickle.dump(rows, open(os.path.join(SD, "core_rows.pkl"), "wb"))
print("saved", len(rows), "core rows")

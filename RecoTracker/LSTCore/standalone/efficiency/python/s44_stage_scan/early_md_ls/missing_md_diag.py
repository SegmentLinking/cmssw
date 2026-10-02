#!/usr/bin/env python3
"""Diagnose MD-able modules (track has reco hits on both sensors) that have NO genuine MD.

Joins the LST ntuple (survival CSV from md_ls_survival.py) with the INPUT trackingNtuple to get
per-hit cluster size, sim associations (merged clusters), positions. Events aligned by sim_pt
signature (NOT by index - stream permutation).
Usage: md_ls_survival csv must exist. python3 missing_md_diag.py <lst_ntuple> <survival_csv>
"""
import sys, uproot, awkward as ak, numpy as np, pandas as pd
from collections import defaultdict, Counter

LST, CSV = sys.argv[1], sys.argv[2]
OUT = '/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/efficiency/python/s44_stage_scan/early_md_ls/'
df = pd.read_csv(CSV, keep_default_na=False)
lt = uproot.open(LST)['tree']
la = lt.arrays(['sim_pt', 'sim_trkNtupIdx', 'md_detId', 'md_isPLS', 'sim_mdIdxAll', 'sim_mdIdxAllFrac',
                'md_simIdxAll'])
it = uproot.open('trackingNtuple-100.root')['trackingNtuple/tree']
ia = it.arrays(['sim_pt', 'sim_simHitIdx', 'simhit_hitIdx', 'simhit_detId', 'simhit_simTrkIdx', 'simhit_particle',
                'simhit_process', 'simhit_x', 'simhit_y', 'simhit_z', 'ph2_detId', 'ph2_x', 'ph2_y', 'ph2_z', 'ph2_clustSize', 'ph2_simHitIdx',
                'ph2_isLower', 'ph2_moduleType', 'ph2_subdet', 'ph2_layer', 'ph2_side'])

# align events
sig_in = {}
for j in range(len(ia)):
    p = ak.to_numpy(ia.sim_pt[j])
    sig_in[(len(p), round(float(p.max()), 3))] = j
evmap = {}
for e in range(len(la)):
    trk = ak.to_numpy(la.sim_trkNtupIdx[e]).astype(int)
    p = ak.to_numpy(la.sim_pt[e])
    for j in range(len(ia)):
        pin = ak.to_numpy(ia.sim_pt[j])
        if trk.max() < len(pin) and np.allclose(pin[trk], p, rtol=1e-5):
            evmap[e] = j
            break
print('aligned events', len(evmap), 'of', len(la), ' permuted:', sum(1 for k, v in evmap.items() if k != v))

recs = []
CACHE = {}
allrecs = []  # also MD-able modules WITH a genuine MD, for comparison
for r in df.itertuples():
    e = r.evt
    if e not in evmap:
        continue
    j = evmap[e]
    if j not in CACHE:
        CACHE.clear()
        ev = ia[j]
        CACHE[j] = dict(sim_simHitIdx=ak.to_list(ev.sim_simHitIdx), simhit_hitIdx=ak.to_list(ev.simhit_hitIdx),
                        simhit_detId=ak.to_numpy(ev.simhit_detId), simhit_simTrkIdx=ak.to_numpy(ev.simhit_simTrkIdx),
                        ph2_detId=ak.to_numpy(ev.ph2_detId), ph2_x=ak.to_numpy(ev.ph2_x), ph2_y=ak.to_numpy(ev.ph2_y),
                        ph2_z=ak.to_numpy(ev.ph2_z), ph2_clustSize=ak.to_numpy(ev.ph2_clustSize),
                        ph2_simHitIdx=ak.to_list(ev.ph2_simHitIdx), ph2_subdet=ak.to_numpy(ev.ph2_subdet),
                        ph2_layer=ak.to_numpy(ev.ph2_layer), ph2_side=ak.to_numpy(ev.ph2_side),
                        ph2_moduleType=ak.to_numpy(ev.ph2_moduleType), simhit_x=ak.to_numpy(ev.simhit_x), simhit_y=ak.to_numpy(ev.simhit_y), simhit_z=ak.to_numpy(ev.simhit_z), simhit_process=ak.to_numpy(ev.simhit_process), trk=ak.to_numpy(la.sim_trkNtupIdx[e]))
    C = CACHE[j]
    isim_in = int(C['trk'][r.isim])
    shs = C['sim_simHitIdx'][isim_in]
    # track reco hits per module
    mod_hits = defaultdict(list)
    mod_sh = defaultdict(list)
    for sh in shs:
        dsh = int(C['simhit_detId'][sh])
        if ((dsh >> 25) & 7) in (4, 5):
            mod_sh[dsh >> 2].append(sh)
        for h in C['simhit_hitIdx'][sh]:
            h = int(h)
            if h < 0 or h >= len(C['ph2_detId']):
                continue
            d = int(C['ph2_detId'][h])
            if int(C['simhit_detId'][sh]) != d:
                continue  # not an OT ph2 hit (IT index space)
            mod_hits[d >> 2].append(h)
    miss = set(int(x) for x in str(r.missmods).split(';') if x)
    for m, hs in mod_hits.items():
        hs = sorted(set(hs))
        sens = {int(C['ph2_detId'][h]) & 3 for h in hs}
        if sens != {1, 2}:
            continue
        cs = [int(C['ph2_clustSize'][h]) for h in hs]
        # merged: hit with simhits from >1 sim track
        nsimtrk = []
        for h in hs:
            st = {int(C['simhit_simTrkIdx'][s]) for s in C['ph2_simHitIdx'][h]}
            nsimtrk.append(len(st))
        lo = [h for h in hs if int(C['ph2_detId'][h]) & 3 == 1]
        up = [h for h in hs if int(C['ph2_detId'][h]) & 3 == 2]
        # naive geometry of best pair
        best = None
        for hl in lo:
            for hu in up:
                xl, yl, zl = (float(C['ph2_x'][hl]), float(C['ph2_y'][hl]), float(C['ph2_z'][hl]))
                xu, yu, zu = (float(C['ph2_x'][hu]), float(C['ph2_y'][hu]), float(C['ph2_z'][hu]))
                dphi = np.arctan2(xl * yu - xu * yl, xl * xu + yl * yu)
                dz = zl - zu
                drt = np.hypot(xl, yl) - np.hypot(xu, yu)
                if best is None or abs(dphi) < abs(best[0]):
                    best = (dphi, dz, drt, np.hypot(xl, yl), zl)
        h0 = hs[0]
        shl = mod_sh.get(m, [])
        nsh_lo = sum(1 for x in shl if int(C['simhit_detId'][x]) & 3 == 1)
        nsh_up = sum(1 for x in shl if int(C['simhit_detId'][x]) & 3 == 2)
        nonprim = sum(1 for x in shl if int(C['simhit_process'][x]) != 2)
        # max transverse residual of each reco hit to the nearest simhit of this track on same sensor
        res = []
        for h in hs:
            dd = int(C['ph2_detId'][h])
            cand = [x for x in shl if int(C['simhit_detId'][x]) == dd]
            if cand:
                res.append(min(np.hypot(C['ph2_x'][h] - C['simhit_x'][x], C['ph2_y'][h] - C['simhit_y'][x]) for x in cand))
        maxres = max(res) if res else -1
        rphi = []
        simdphi = None
        for h in hs:
            dd = int(C['ph2_detId'][h])
            cand = [x for x in shl if int(C['simhit_detId'][x]) == dd]
            if cand:
                ph_r = np.arctan2(C['ph2_y'][h], C['ph2_x'][h]); rr = np.hypot(C['ph2_x'][h], C['ph2_y'][h])
                rphi.append(min(rr * abs(np.angle(np.exp(1j * (ph_r - np.arctan2(C['simhit_y'][x], C['simhit_x'][x]))))) for x in cand))
        maxrphi = max(rphi) if rphi else -1
        slo = [x for x in shl if int(C['simhit_detId'][x]) & 3 == 1]
        sup = [x for x in shl if int(C['simhit_detId'][x]) & 3 == 2]
        if slo and sup:
            simdphi = min(abs(np.angle(np.exp(1j * (np.arctan2(C['simhit_y'][a1], C['simhit_x'][a1]) - np.arctan2(C['simhit_y'][b1], C['simhit_x'][b1]))))) for a1 in slo for b1 in sup)
        rec = dict(evt=e, isim=r.isim, pt=r.pt, dR=r.dR, matched=r.matched, mod=m, missing=m in miss,
                   subdet=int(C['ph2_subdet'][h0]), layer=int(C['ph2_layer'][h0]), side=int(C['ph2_side'][h0]),
                   modtype=int(C['ph2_moduleType'][h0]), nlo=len(lo), nup=len(up), maxcs=max(cs),
                   merged=max(nsimtrk) > 1, dphi=best[0], dz=best[1], drt=best[2], rt=best[3], z=best[4], nsh_lo=nsh_lo, nsh_up=nsh_up, nonprim=nonprim, maxres=maxres, maxrphi=maxrphi, simdphi=simdphi if simdphi is not None else -1)
        allrecs.append(rec)

A = pd.DataFrame(allrecs)
A.to_csv(OUT + 'mdable_modules_diag.csv', index=False)
print('MD-able modules (input-rebuilt):', len(A), ' missing genuine MD:', int(A.missing.sum()),
      f'({A.missing.mean():.4f})')
A['core'] = A.dR < 0.02
A['cs_gt16'] = A.maxcs > 16
A['multi'] = (A.nlo > 1) | (A.nup > 1)
for key in ['core', 'subdet', 'modtype', 'layer', 'side', 'cs_gt16', 'merged', 'multi']:
    g = A.groupby(key).missing.agg(['mean', 'sum', 'count'])
    print('\nmissing-MD rate by', key); print(g.to_string())
M = A[A.missing]
print('\nmissing MD modules: |dz| quantiles', np.quantile(np.abs(M.dz), [.5, .9, .99]).round(3),
      ' vs present', np.quantile(np.abs(A[~A.missing].dz), [.5, .9, .99]).round(3))
print('missing MD modules: |dphi| quantiles', np.quantile(np.abs(M.dphi), [.5, .9, .99]).round(5),
      ' vs present', np.quantile(np.abs(A[~A.missing].dphi), [.5, .9, .99]).round(5))
print('missing: max clust size dist', Counter(M.maxcs.clip(upper=20)).most_common(12))
print('present: max clust size dist', Counter(A[~A.missing].maxcs.clip(upper=20)).most_common(12))
# unexplained fraction: not cs>16
U = M[~M.cs_gt16]
print('\nmissing & clustsize<=16:', len(U), ' of which merged', int(U.merged.sum()), ' multi-hit', int(U.multi.sum()))
print(U[['subdet', 'layer', 'side', 'modtype', 'nlo', 'nup', 'maxcs', 'merged', 'dphi', 'dz', 'drt', 'rt', 'z', 'pt']]
      .head(40).to_string())

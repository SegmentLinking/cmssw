#!/usr/bin/env python3
"""Per-sim-track OT survival: reco hits -> genuine MD -> genuine LS, jet core vs outside.

Read-only on existing LST ntuples. Per denominator sim track (performance.cc base_0_0 +
jet selection, as in jetcore_failure_taxonomy.py):
  * "MD-able module": OT module (detId>>2) where the sim track has reco hits on BOTH sensors
    (detId&3 == 1 and 2).
  * genuine MD: MD whose sim_mdIdxAllFrac for this sim == 1.0 (both hits from the track).
  * MD efficiency per MD-able module = has genuine MD with detId>>2 == module.
  * LS: for consecutive (in radius) genuine MDs on ADJACENT logical layers
    (same subdet, |dlayer|==1, or barrel->endcap), is there an LS joining EXACTLY those
    two MDs (ls_mdIdx0/1)? Also: any genuine LS (frac>=0.75) at all.
Usage: python3 md_ls_survival.py <ntuple> [nevents]
"""
import sys, uproot, awkward as ak, numpy as np
from collections import defaultdict

F = sys.argv[1]
NEV = int(sys.argv[2]) if len(sys.argv) > 2 else None
t = uproot.open(F)['tree']
BR = ['sim_q', 'sim_pt', 'sim_eta', 'sim_vx', 'sim_vy', 'sim_vz', 'sim_genjet_idx', 'sim_genjet_deltaR',
      'genjet_pt', 'genjet_eta', 'sim_tcIdx', 'sim_recoHitDetId', 'sim_simHitDetId', 'sim_simHitLayer',
      'sim_mdIdxAll', 'sim_mdIdxAllFrac', 'sim_lsIdxAll', 'sim_lsIdxAllFrac', 'sim_plsIdxAllFrac',
      'sim_t3IdxAllFrac', 'sim_pt5IdxAllFrac',
      'md_detId', 'md_isPLS', 'md_layer', 'md_anchor_x', 'md_anchor_y', 'md_anchor_z',
      'ls_mdIdx0', 'ls_mdIdx1', 'ls_isPLS']
a = t.arrays(BR, entry_stop=NEV)


def otsub(d):
    return (d >> 25) & 7


def logical_layer(d):
    s = otsub(d)
    if s == 5:
        return (d >> 20) & 0xF
    if s == 4:
        return 6 + ((d >> 18) & 0xF)
    return 0


def adjacent(l1, l2):
    if l1 == 0 or l2 == 0:
        return False
    b1, b2 = l1 <= 6, l2 <= 6
    if b1 == b2:
        return abs(l1 - l2) == 1
    return True  # barrel<->endcap transition


rows = []
for e in range(len(a)):
    ev = a[e]
    gidx = ak.to_numpy(ev.sim_genjet_idx)
    gpt = ak.to_numpy(ev.genjet_pt)
    geta = ak.to_numpy(ev.genjet_eta)
    q = ak.to_numpy(ev.sim_q); pt = ak.to_numpy(ev.sim_pt); eta = ak.to_numpy(ev.sim_eta)
    vx = ak.to_numpy(ev.sim_vx); vy = ak.to_numpy(ev.sim_vy); vz = ak.to_numpy(ev.sim_vz)
    dR = ak.to_numpy(ev.sim_genjet_deltaR); tc = ak.to_numpy(ev.sim_tcIdx)
    safe = np.where(gidx < 0, 0, gidx)
    jpt = gpt[safe] if len(gpt) else np.zeros_like(pt)
    jeta = geta[safe] if len(geta) else np.zeros_like(pt)
    den = ((q != 0) & (pt > 0.9) & (np.abs(eta) < 4.5) & (np.abs(vz) < 30) & (np.hypot(vx, vy) < 2.5)
           & (gidx >= 0) & (jpt > 1000) & (np.abs(jeta) < 2.5))
    idx = np.nonzero(den)[0]
    if len(idx) == 0:
        continue
    md_det = ak.to_numpy(ev.md_detId); md_lay = ak.to_numpy(ev.md_layer)
    md_r = np.hypot(ak.to_numpy(ev.md_anchor_x), ak.to_numpy(ev.md_anchor_y))
    md_pls = ak.to_numpy(ev.md_isPLS)
    ls0 = ak.to_numpy(ev.ls_mdIdx0); ls1 = ak.to_numpy(ev.ls_mdIdx1)
    lsset = set(zip(ls0.tolist(), ls1.tolist()))
    for i in idx:
        rh = ak.to_numpy(ev.sim_recoHitDetId[i])
        sh = ak.to_numpy(ev.sim_simHitDetId[i])
        shl = ak.to_numpy(ev.sim_simHitLayer[i])
        # OT sim hit modules / sensors
        sh_ot = sh[shl > 0]
        sim_sens = defaultdict(set)
        for d in sh_ot:
            sim_sens[int(d) >> 2].add(int(d) & 3)
        reco_sens = defaultdict(set)
        for d in rh:
            if otsub(int(d)) in (4, 5):
                reco_sens[int(d) >> 2].add(int(d) & 3)
        sim_both = {m for m, s in sim_sens.items() if s >= {1, 2}}
        reco_both = {m for m, s in reco_sens.items() if s >= {1, 2}}
        reco_one = {m for m in sim_both if m not in reco_both}  # sim both sensors, reco missing >=1
        # genuine MDs
        mdi = ak.to_numpy(ev.sim_mdIdxAll[i]); mdf = ak.to_numpy(ev.sim_mdIdxAllFrac[i])
        gmd = [int(m) for m, f in zip(mdi, mdf) if f > 0.99 and not md_pls[m]]
        gmd_mods = {int(md_det[m]) >> 2 for m in gmd}
        n_mdable = len(reco_both)
        n_md_ok = len(reco_both & gmd_mods)
        missing_md_mods = reco_both - gmd_mods
        # genuine layers
        lay_mdable = {logical_layer(m << 2 | 1) for m in reco_both}
        lay_md = {int(md_lay[m]) for m in gmd}
        # LS test on radius-ordered genuine MDs (one per module: take all combos between adjacent layers)
        bylay = defaultdict(list)
        for m in gmd:
            bylay[int(md_lay[m])].append(m)
        lays = sorted(bylay, key=lambda L: np.mean([md_r[m] for m in bylay[L]]) if L <= 6 else 100 + L)
        n_pairs = n_pairs_ok = 0
        for L1, L2 in zip(lays[:-1], lays[1:]):
            if not adjacent(L1, L2):
                continue
            n_pairs += 1
            ok = any((m1, m2) in lsset for m1 in bylay[L1] for m2 in bylay[L2])
            n_pairs_ok += ok
        lsf = ak.to_numpy(ev.sim_lsIdxAllFrac[i])
        n_gls = int((lsf >= 0.75).sum())
        n_gls_full = int((lsf > 0.99).sum())
        plsf = ak.to_numpy(ev.sim_plsIdxAllFrac[i]); t3f = ak.to_numpy(ev.sim_t3IdxAllFrac[i])
        pt5f = ak.to_numpy(ev.sim_pt5IdxAllFrac[i])
        rows.append((e, i, pt[i], eta[i], dR[i], tc[i] >= 0, len(sim_both), n_mdable, len(reco_one),
                     n_md_ok, n_pairs, n_pairs_ok, n_gls, n_gls_full, len(lay_mdable), len(lay_md),
                     (plsf >= 0.75).any() if len(plsf) else False, (t3f >= 0.75).any() if len(t3f) else False,
                     (pt5f >= 0.75).any() if len(pt5f) else False,
                     ';'.join(str(m) for m in sorted(missing_md_mods))))

cols = ['evt', 'isim', 'pt', 'eta', 'dR', 'matched', 'n_sim2', 'n_mdable', 'n_recomiss', 'n_md_ok', 'n_lspairs',
        'n_lspairs_ok', 'n_gls', 'n_gls_full', 'nlay_mdable', 'nlay_md', 'hasPLS', 'hasT3', 'hasPT5', 'missmods']
import pandas as pd
df = pd.DataFrame(rows, columns=cols)
out = '/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/efficiency/python/s44_stage_scan/early_md_ls/'
tag = F.split('/')[-1].replace('.root', '')
df.to_csv(out + f'survival_{tag}.csv', index=False)


def summarize(sel, name):
    d = df[sel]
    n = len(d)
    if n == 0:
        return
    s2 = d.n_sim2.sum(); mda = d.n_mdable.sum(); mok = d.n_md_ok.sum()
    lp = d.n_lspairs.sum(); lpo = d.n_lspairs_ok.sum()
    print(f'{name:38s} N={n:5d} eff={d.matched.mean():.3f} | simhit-2sensor mods={s2:6d} '
          f'reco-both/sim-both={mda/s2 if s2 else 0:.4f} | MD/MD-able={mok/mda if mda else 0:.4f} '
          f'| tracks all-MD-able-have-MD={(d.n_md_ok == d.n_mdable).mean():.3f} '
          f'| LS/adjMDpair={lpo/lp if lp else 0:.4f} (pairs {lp}) | trk w/ >=1 missing LS={(d.n_lspairs_ok < d.n_lspairs).mean():.3f}'
          f' | noGenuineLS={(d.n_gls == 0).mean():.3f} | nlayMD>=3 {(d.nlay_md >= 3).mean():.3f}')


print('FILE', F, 'denominator tracks', len(df))
bins = [('core dR<0.02', df.dR < 0.02), ('0.02<=dR<0.10', (df.dR >= 0.02) & (df.dR < 0.10)),
        ('dR>=0.10 (in jet, outer)', df.dR >= 0.10)]
for nm, s in bins:
    summarize(s, nm)
print('--- pT>=50 GeV only (control for pT)')
for nm, s in bins:
    summarize(s & (df.pt >= 50), nm + ' pt>50')
print('--- core failures vs successes')
summarize((df.dR < 0.02) & (~df.matched), 'core FAIL')
summarize((df.dR < 0.02) & (df.matched), 'core OK')
print('--- core failures with pLS but no pT5 (the 31% class)')
summarize((df.dR < 0.02) & (~df.matched) & df.hasPLS & ~df.hasPT5, 'core FAIL pLS-noPT5')
print('--- core failures by nlay_md (genuine-MD logical layers)')
c = df[(df.dR < 0.02) & (~df.matched)]
print(c.nlay_md.value_counts().sort_index().to_dict())
print('core OK nlay_md', df[(df.dR < 0.02) & (df.matched)].nlay_md.value_counts().sort_index().to_dict())
print('core FAIL nlay_mdable', c.nlay_mdable.value_counts().sort_index().to_dict())

#!/usr/bin/env python3
"""For adjacent genuine-MD pairs of denominator sim tracks lacking an LS: in module map? which order? core vs not."""
import sys, pickle, uproot, awkward as ak, numpy as np, pandas as pd
from collections import defaultdict, Counter
F = sys.argv[1]
S = '/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/efficiency/python/s44_stage_scan/early_md_ls/'
M = pickle.load(open(S + 'modmap.pkl', 'rb'))
print('map key &3:', Counter(k & 3 for k in M))
MM = defaultdict(set)
for k, v in M.items():
    for x in v:
        MM[k >> 2].add(x >> 2)
df = pd.read_csv(S + 'survival_' + F.split('/')[-1].replace('.root', '') + '.csv', keep_default_na=False)
t = uproot.open(F)['tree']
a = t.arrays(['sim_mdIdxAll', 'sim_mdIdxAllFrac', 'md_detId', 'md_layer', 'md_isPLS', 'md_anchor_x', 'md_anchor_y',
              'md_anchor_z', 'ls_mdIdx0', 'ls_mdIdx1', 'sim_lsIdxAll', 'sim_lsIdxAllFrac'])
rows = []
for e, g in df.groupby('evt'):
    ev = a[e]
    det = ak.to_numpy(ev.md_detId); lay = ak.to_numpy(ev.md_layer); pls = ak.to_numpy(ev.md_isPLS)
    r = np.hypot(ak.to_numpy(ev.md_anchor_x), ak.to_numpy(ev.md_anchor_y)); z = ak.to_numpy(ev.md_anchor_z)
    lsset = set(zip(ak.to_numpy(ev.ls_mdIdx0).tolist(), ak.to_numpy(ev.ls_mdIdx1).tolist()))
    lsmods = set((int(det[i]) >> 2, int(det[j]) >> 2) for i, j in lsset)
    for rr in g.itertuples():
        mdi = ak.to_numpy(ev.sim_mdIdxAll[rr.isim]); mdf = ak.to_numpy(ev.sim_mdIdxAllFrac[rr.isim])
        gmd = [int(m) for m, f in zip(mdi, mdf) if f > 0.99 and not pls[m]]
        gmd.sort(key=lambda m: np.hypot(r[m], z[m]))
        for i in range(len(gmd)):
            for j in range(len(gmd)):
                m1, m2 = gmd[i], gmd[j]
                L1, L2 = int(lay[m1]), int(lay[m2])
                adj = (L1 <= 6 and L2 <= 6 and L2 == L1 + 1) or (L1 > 6 and L2 > 6 and L2 == L1 + 1) or (L1 <= 6 and L2 > 6)
                if not adj or np.hypot(r[m2], z[m2]) <= np.hypot(r[m1], z[m1]):
                    continue
                mod1, mod2 = int(det[m1]) >> 2, int(det[m2]) >> 2
                built = (m1, m2) in lsset
                rows.append(dict(evt=e, isim=rr.isim, pt=rr.pt, dR=rr.dR, matched=rr.matched, L1=L1, L2=L2,
                                 built=built, inmap=mod2 in MM.get(mod1, set()), anyls_mods=(mod1, mod2) in lsmods))
D = pd.DataFrame(rows)
D.to_csv(S + 'ls_pairs.csv', index=False)
D['core'] = D.dR < 0.02
print('adjacent genuine-MD pairs:', len(D), ' LS built:', D.built.mean().round(4))
print(D.pivot_table(index=['inmap'], columns='core', values='built', aggfunc=['mean', 'count']).round(4))
nb = D[~D.built]
print('not-built pairs: in map', int(nb.inmap.sum()), ' not in map', int((~nb.inmap).sum()))
print('not-in-map by layer transition:', Counter(zip(nb[~nb.inmap].L1, nb[~nb.inmap].L2)).most_common(10))
print('in-map-but-cut by layer transition:', Counter(zip(nb[nb.inmap].L1, nb[nb.inmap].L2)).most_common(10))
# per track: tracks where every pair between some adjacent layers is unbuilt
tk = D.groupby(['evt', 'isim', 'core', 'matched']).apply(lambda x: (~x.built).any()).reset_index(name='anymiss')
print(tk.groupby(['core', 'matched']).anymiss.agg(['mean', 'sum', 'count']))

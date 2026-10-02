#!/usr/bin/env python3
"""Per-sim-track FAILURE TAXONOMY for LST in the CENTER OF JETS.

For each genuine jet-core sim track that FAILS to reconstruct (sim_tcIdx<0),
find the FIRST tier of the build hierarchy where it lacks a GENUINE object
(match fraction >= 0.75). Distinguishes "object never built" (upstream) from
"built but killed" (dedup / cross-clean / BeforeTC).

Reads REAL-pLS, AfterBuild-ON, 100 evt ntuple. Does NOT modify LST source.
"""
import uproot, awkward as ak, numpy as np

F = 'Ntuple-files/LSTNtuple_realpls_mastercuts_100evt_v2.root'
GEN = 0.75  # genuine match-fraction threshold

BR = ['sim_q','sim_pt','sim_eta','sim_vx','sim_vy','sim_vz',
      'sim_genjet_idx','sim_genjet_deltaR','genjet_pt','genjet_eta','sim_tcIdx',
      'sim_t3IdxAllFrac','sim_t5IdxAllFrac','sim_plsIdxAllFrac',
      'sim_pt3IdxAllFrac','sim_pt5IdxAllFrac']

t = uproot.open(F)['tree']
a = t.arrays(BR)

# gather per-genjet quantities to per-sim-track via sim_genjet_idx (clamp<0 -> 0, masked later)
gidx = a['sim_genjet_idx']
safe = ak.where(gidx < 0, 0, gidx)
gjpt  = a['genjet_pt'][safe]
gjeta = a['genjet_eta'][safe]

def has_genuine(fracbranch):
    fr = a[fracbranch]
    return ak.fill_none(ak.max(fr, axis=-1), -1.0) >= GEN

hasT3  = has_genuine('sim_t3IdxAllFrac')
hasT5  = has_genuine('sim_t5IdxAllFrac')
hasPLS = has_genuine('sim_plsIdxAllFrac')
hasPT3 = has_genuine('sim_pt3IdxAllFrac')
hasPT5 = has_genuine('sim_pt5IdxAllFrac')

# denominator (performance.cc base_0_0)
vtxperp = np.sqrt(a['sim_vx']**2 + a['sim_vy']**2)
den = ((a['sim_q'] != 0) & (a['sim_pt'] > 0.9) & (abs(a['sim_eta']) < 4.5) &
       (abs(a['sim_vz']) < 30) & (vtxperp < 2.5) &
       (gidx >= 0) & (gjpt > 1000) & (abs(gjeta) < 2.5))

failed = den & (a['sim_tcIdx'] < 0)
success = den & (a['sim_tcIdx'] >= 0)
dR = a['sim_genjet_deltaR']

# flatten everything to 1D numpy aligned per-track across all events
def f1(x):
    return ak.to_numpy(ak.flatten(x))
F_hasT3, F_hasT5, F_hasPLS = f1(hasT3), f1(hasT5), f1(hasPLS)
F_hasPT3, F_hasPT5 = f1(hasPT3), f1(hasPT5)
F_dR = f1(dR)
F_den, F_failed, F_success = f1(den), f1(failed), f1(success)

# collect: apply a flat 1D boolean mask
def collect(mask):
    m = mask
    return dict(hasT3=F_hasT3[m], hasT5=F_hasT5[m], hasPLS=F_hasPLS[m],
                hasPT3=F_hasPT3[m], hasPT5=F_hasPT5[m], dR=F_dR[m])

def bucketize(d):
    """Return array of bucket labels for each failing track (first-match ladder)."""
    n = len(d['dR'])
    lab = np.empty(n, dtype=object)
    t3, t5, pls, pt3, pt5 = d['hasT3'], d['hasT5'], d['hasPLS'], d['hasPT3'], d['hasPT5']
    for i in range(n):
        if t5[i] or pt5[i] or pt3[i]:
            lab[i] = 'A: built-but-killed (T5/pT5/pT3 genuine, no TC)'
        elif t3[i] and pls[i]:
            lab[i] = 'B: T3+pLS present, no T5/pT3/pT5 (both chains stall)'
        elif t3[i] and not pls[i]:
            lab[i] = 'C: T3 only, no T5, no pLS (OT stall, no seed)'
        elif pls[i] and not t3[i]:
            lab[i] = 'D: pLS only, no T3 (pixel seed, no OT / no pT)'
        else:
            lab[i] = 'E: structurally unbuilt (no T3, no T5, no pLS, no pT3/pT5)'
    return lab

BUCKETS = ['A: built-but-killed (T5/pT5/pT3 genuine, no TC)',
           'B: T3+pLS present, no T5/pT3/pT5 (both chains stall)',
           'C: T3 only, no T5, no pLS (OT stall, no seed)',
           'D: pLS only, no T3 (pixel seed, no OT / no pT)',
           'E: structurally unbuilt (no T3, no T5, no pLS, no pT3/pT5)']

# strict structurally-unbuilt = no genuine object at ANY tier
def strict_unbuilt(d):
    return ~(d['hasT3'] | d['hasT5'] | d['hasPLS'] | d['hasPT3'] | d['hasPT5'])

CORE_BINS = [('dR<0.02', 0.02), ('dR<0.05', 0.05), ('dR<0.10', 0.10)]

df = collect(F_failed)
labels = bucketize(df)
su = strict_unbuilt(df)
ds = collect(F_success)

print('='*78)
print('LST JET-CORE FAILURE TAXONOMY   file:', F)
print('genuine threshold frac >=', GEN)
print('='*78)

for name, cut in CORE_BINS:
    sel = df['dR'] < cut
    ntot = int(sel.sum())
    print(f'\n### CORE {name}   (failing core tracks = {ntot}, core denominator = '
          f'{int((collect(F_den)["dR"] < cut).sum())})')
    if ntot == 0:
        continue
    for b in BUCKETS:
        c = int(((labels == b) & sel).sum())
        print(f'   {c:6d}  {100.0*c/ntot:6.2f}%   {b}')
    nsu = int((su & sel).sum())
    print(f'   ---- strict structurally-unbuilt (any-tier none): {nsu}  '
          f'{100.0*nsu/ntot:6.2f}% of core failures')

# bucket-A (built-but-killed) sub-breakdown: which terminal object survived to build
print('\n### Bucket A (built-but-killed) sub-breakdown by surviving genuine object')
for name, cut in CORE_BINS:
    sel = (labels == BUCKETS[0]) & (df['dR'] < cut)
    n = int(sel.sum())
    if not n: continue
    npt5 = int((df['hasPT5'] & sel).sum())
    nt5  = int((df['hasT5']  & sel).sum())
    npt3 = int((df['hasPT3'] & sel).sum())
    print(f'   {name}: A={n}   has genuine pT5={npt5} ({100.*npt5/n:.1f}%)  '
          f'T5={nt5} ({100.*nt5/n:.1f}%)  pT3={npt3} ({100.*npt3/n:.1f}%)')

# structurally-unbuilt as fraction of the CORE DENOMINATOR (prior work ~8.9%)
print('\n### structurally-unbuilt as %% of CORE DENOMINATOR (cf prior ~8.9%%)')
dd = collect(F_den)
sud = strict_unbuilt(dd)
fd = collect(F_failed)  # for consistency
for name, cut in CORE_BINS:
    denom = int((dd['dR'] < cut).sum())
    # structurally unbuilt among denominator (they are failures by construction)
    nsu = int((sud & (dd['dR'] < cut)).sum())
    print(f'   {name}: {nsu}/{denom} = {100.0*nsu/denom:6.2f}% of core denominator')

# successful core tracks: dominant path
print('\n### SUCCESSFUL core tracks: reconstruction path (priority pT5>pT3>standalone-T5)')
def path(d):
    n = len(d['dR']); lab = np.empty(n, dtype=object)
    for i in range(n):
        if d['hasPT5'][i]: lab[i] = 'pT5'
        elif d['hasPT3'][i]: lab[i] = 'pT3'
        elif d['hasT5'][i]: lab[i] = 'standalone T5-TC'
        else: lab[i] = 'other (pLS-only / <0.75)'
    return lab
plab = path(ds)
for name, cut in CORE_BINS:
    sel = ds['dR'] < cut
    ntot = int(sel.sum())
    print(f'   {name}: successful core tracks = {ntot}')
    for p in ['standalone T5-TC','pT5','pT3','other (pLS-only / <0.75)']:
        c = int(((plab == p) & sel).sum())
        if ntot: print(f'       {c:6d}  {100.0*c/ntot:6.2f}%   {p}')

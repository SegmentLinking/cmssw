#!/usr/bin/env python3
"""
Physics characterization of jet-core FAILING sim tracks in LST.

Estimates how much jet-core inefficiency is physics-irreducible vs algorithmic.
Input: REAL-pLS, AfterBuild-ON, 100 evt ntuple.

Denominator (matches performance.cc base_0_0):
  sim_q!=0, sim_pt>0.9, |sim_eta|<4.5, |sim_vz|<30, sqrt(vx^2+vy^2)<2.5,
  matched genjet (sim_genjet_idx>=0, genjet_pt>1000, |genjet_eta|<2.5).
Core = sim_genjet_deltaR < threshold (0.02 / 0.05 / 0.10, cumulative).
Failing = sim_tcIdx < 0 (no matched TrackCandidate).
Genuine object at a tier = any entry in sim_<tier>IdxAllFrac >= 0.75.
Structurally unbuilt = no genuine pLS AND no genuine T3 AND no genuine T5 AND no genuine pT5.
"""
import uproot, awkward as ak, numpy as np

F = 'Ntuple-files/LSTNtuple_realpls_mastercuts_100evt_v2.root'
FRAC = 0.75

t = uproot.open(F)['tree']
br = ['sim_pt','sim_eta','sim_q','sim_pdgId','sim_vx','sim_vy','sim_vz',
      'sim_tcIdx','sim_genjet_deltaR','sim_genjet_idx','genjet_pt','genjet_eta']
fracbr = ['sim_plsIdxAllFrac','sim_t3IdxAllFrac','sim_t5IdxAllFrac','sim_pt5IdxAllFrac']
d = t.arrays(br+fracbr)

# flatten per-sim-track (events -> tracks)
def flat(x): return ak.to_numpy(ak.flatten(x))
pt   = flat(d['sim_pt']);   eta = flat(d['sim_eta']); q = flat(d['sim_q'])
pdg  = flat(d['sim_pdgId']);vx  = flat(d['sim_vx']);  vy = flat(d['sim_vy']); vz = flat(d['sim_vz'])
tc   = flat(d['sim_tcIdx']);dR  = flat(d['sim_genjet_deltaR'])
gjidx= flat(d['sim_genjet_idx'])

# genjet_pt / eta indexed per event -> broadcast to sim tracks via gjidx
# safer: match genjet cut using per-track values; build a jagged genjet lookup
gjpt_ev  = d['genjet_pt']; gjeta_ev = d['genjet_eta']
sim_gjidx_ev = d['sim_genjet_idx']
# broadcast: for each sim track pick genjet_pt[idx] if idx>=0 else -1
def lookup(ev_vals, ev_idx):
    out=[]
    vals = ev_vals.tolist(); idxs = ev_idx.tolist()
    for vv,ii in zip(vals,idxs):
        for j in ii:
            if j is not None and j>=0 and j<len(vv): out.append(vv[j])
            else: out.append(-1.0)
    return np.array(out)
gjpt  = lookup(gjpt_ev,  sim_gjidx_ev)
gjeta = lookup(gjeta_ev, sim_gjidx_ev)

def has_genuine(fb):
    a = d[fb]                                  # events -> tracks -> objects
    mx = ak.fill_none(ak.max(a, axis=-1), -1.0)  # events -> tracks
    return ak.to_numpy(ak.flatten(mx)) >= FRAC
g_pls = has_genuine('sim_plsIdxAllFrac')
g_t3  = has_genuine('sim_t3IdxAllFrac')
g_t5  = has_genuine('sim_t5IdxAllFrac')
g_pt5 = has_genuine('sim_pt5IdxAllFrac')

vperp = np.sqrt(vx*vx+vy*vy)

# base denominator
base = (q!=0)&(pt>0.9)&(np.abs(eta)<4.5)&(np.abs(vz)<30)&(vperp<2.5)&\
       (gjidx>=0)&(gjpt>1000)&(np.abs(gjeta)<2.5)

def species(p):
    ap=np.abs(p)
    s=np.full(p.shape,'other',dtype=object)
    s[ap==211]='pi'; s[ap==321]='K'; s[ap==2212]='p'
    s[ap==11]='e'; s[ap==13]='mu'
    return s
sp = species(pdg)

structunbuilt = ~(g_pls|g_t3|g_t5|g_pt5)

def report_bin(name, coremask):
    dsel = base & coremask
    fail = dsel & (tc<0)
    succ = dsel & (tc>=0)
    nD=dsel.sum(); nF=fail.sum(); nS=succ.sum()
    print(f"\n{'='*70}\n{name}: denom={nD}  fail={nF}  succ={nS}  eff={nS/nD*100:.1f}%")
    if nF==0: return
    # species
    print(f"  {'species':8} {'fail':>6} {'fail%':>7} {'succ%':>7}")
    for k in ['pi','K','p','e','mu','other']:
        ff=(fail&(sp==k)).sum(); ss=(succ&(sp==k)).sum()
        print(f"  {k:8} {ff:6d} {ff/nF*100:6.1f}% {ss/max(nS,1)*100:6.1f}%")
    # pt spectrum
    fp=pt[fail]
    print(f"  pt<1.5: {(fp<1.5).sum()/nF*100:.1f}%   pt<2.0: {(fp<2.0).sum()/nF*100:.1f}%   pt<0.9-1.1:{((fp<1.1)).sum()/nF*100:.1f}%  median pt={np.median(fp):.2f}")
    for lo,hi in [(0.9,1.1),(1.1,1.5),(1.5,2.0),(2.0,5.0),(5.0,1e9)]:
        m=(fp>=lo)&(fp<hi); print(f"    pt[{lo},{hi}): {m.sum():5d} {m.sum()/nF*100:5.1f}%")
    # eta
    fe=np.abs(eta[fail])
    print(f"  |eta|: <1.5:{(fe<1.5).sum()/nF*100:.1f}%  1.5-2.5:{((fe>=1.5)&(fe<2.5)).sum()/nF*100:.1f}%  >2.5:{(fe>=2.5).sum()/nF*100:.1f}%")
    # vertex
    fperp=vperp[fail]; fvz=np.abs(vz[fail])
    print(f"  vtx r<0.1:{(fperp<0.1).sum()/nF*100:.1f}%  0.1-2.5(displaced):{((fperp>=0.1)&(fperp<2.5)).sum()/nF*100:.1f}%  median r={np.median(fperp):.4f}")
    # structurally unbuilt subset
    su = fail & structunbuilt
    nsu=su.sum()
    print(f"  STRUCTURALLY-UNBUILT (no genuine pLS/T3/T5/pT5): {nsu}  = {nsu/nF*100:.1f}% of fails, {nsu/nD*100:.1f}% of denom")
    if nsu>0:
        print(f"    species:", {k:int((su&(sp==k)).sum()) for k in ['pi','K','p','e','mu','other']})
        print(f"    pt<1.5: {(pt[su]<1.5).sum()/nsu*100:.1f}%  displaced(r>0.1):{(vperp[su]>=0.1).sum()/nsu*100:.1f}%  |eta|>1.5:{(np.abs(eta[su])>=1.5).sum()/nsu*100:.1f}%  median pt={np.median(pt[su]):.2f}")
    # partially built (has some tier but no TC) = candidate algorithmic
    pb = fail & ~structunbuilt
    print(f"  PARTIALLY-BUILT (>=1 genuine object but no TC): {pb.sum()}  = {pb.sum()/nF*100:.1f}% of fails")
    print(f"    of partial: has genuine T5:{(pb&g_t5).sum()}  pT5:{(pb&g_pt5).sum()}  T3:{(pb&g_t3).sum()}  pLS:{(pb&g_pls).sum()}")

for nm,thr in [('CORE dR<0.02',0.02),('CORE dR<0.05',0.05),('CORE dR<0.10',0.10),('ALL-in-jet dR<0.4',0.4)]:
    report_bin(nm, dR<thr)

"""Genuine pLS of bucket tracks that were paired with a FOREIGN T5 into a pT5: what is that T5, does the pT5 survive to TC?"""
import uproot, numpy as np, pickle, collections
from attrib3 import ev, rows, dup
F='/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_realpls_mastercuts_100evt_v2.root'
x=uproot.open(F)['tree'].arrays(['t5_simIdx','pT5_simIdx','pT5_isFake','tc_isFake','tc_simIdx','t5_pt'],library='np')
B=[r for r in rows if r['dR']<0.02 and r['tc']<0 and r['pls'] and not r['pt5']]
c=collections.Counter(); ex=[]
for r in B:
    e=ev[r['ev']]; ie=r['ev']
    pls=set(r['pls']); ids=[i for i,p in enumerate(e['pT5_plsIdx']) if p in pls]
    if not ids: continue
    c['tracks with genuine pLS in >=1 pT5']+=1
    t5sims=[x['t5_simIdx'][ie][e['pT5_t5Idx'][i]] for i in ids]
    surv=[i for i in ids if not e['pT5_isDupReco'][i]]
    intc=[i for i in ids if i in set(e['tc_pt5Idx'])]
    kind='T5 of another sim' if any(s>=0 and s!=r['sim'] for s in t5sims) else ('T5 is fake(-1)' if all(s<0 for s in t5sims) else 'T5 same sim (<0.75 T5)')
    c['  T5 partner: '+kind]+=1
    c['  a stolen pT5 survives pT5-dedup']+= bool(surv)
    c['  a stolen pT5 becomes a TC']+= bool(intc)
    # does this track have a genuine T5 at all?
    c['  track has genuine T5']+= bool(r['t5'])
    # is the foreign-T5 sim reconstructed by a TC?
    if intc:
        tcs=[list(e['tc_pt5Idx']).index(i) for i in intc]
        c['  stolen TC isFake']+= any(x['tc_isFake'][ie][t] for t in tcs)
for k,v in c.items(): print(f'{v:4d} {k}')

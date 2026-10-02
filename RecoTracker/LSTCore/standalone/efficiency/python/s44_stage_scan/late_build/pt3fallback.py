"""pT3 as fallback for core tracks with genuine pLS but no genuine pT5: emulate CreatePixelTripletsFromMap gates
(PixelTriplet.h:716-790): pLS isDup, pLS partOfPT5, T3 inner layer in {1,2}, T3 lower/middle PS, T3 partOfPT5,
radius criterion, pT3-DNN with pT3WP (tighter than pT5WP)."""
import numpy as np, collections, pickle
import pt3dnn as D
from attrib3 import ev, ev2, rows, plsinfo, connected, radius_pass, t3kind, kR1GeVf, dup
WP3=np.array([0.6288,0.8014,0.7218,0.743,0.7519,0.8633,0.6934,0.6983,0.6502,0.7037]); WPH3=0.657
class Fake: pass
def stage(ie,p,t):
    e=ev[ie]; e2=ev2[ie]
    if dup[ie][p]: return 1
    if p in set(e['pT5_plsIdx']): return 2
    L0=e['t3_hit_0_layer'][t]; L0b= L0 if L0<=6 else L0-6
    if L0b not in (1,2) or e['t3_hit_0_moduleType'][t]!=0 or e['t3_hit_2_moduleType'][t]!=0: return 3
    if not connected(plsinfo[ie][p],[int(e2['t3_hit_0_detId'][t]),int(e2['t3_hit_1_detId'][t])]): return 3
    if e['t3_partOfPT5'][t]: return 4
    if not radius_pass(e['pLS_pt'][p]*kR1GeVf,e['pLS_ptErr'][p]*kR1GeVf,e['t3_radius'][t],t3kind(e,t)): return 5
    # DNN with pT3WP: reuse evaluate() via a fake T5 index mapping
    e['_tmpT5']=None
    return 6
lab={1:'pLS isDup (all genuine pLS)',2:'genuine pLS consumed by a (non-genuine) pT5',3:'no genuine T3 in layer1/2 PS / not connected',
     4:'genuine T3 consumed by a pT5 (T3.partOfPT5)',5:'radius criterion',6:'reaches pT3-DNN'}
c=collections.Counter(); dnnres=collections.Counter()
B=[r for r in rows if r['dR']<0.02 and r['tc']<0 and r['pls'] and not r['pt5']]
for r in B:
    if r['pt3']: c['genuine pT3 BUILT']+=1; continue
    if not r['t3']: c['0 no genuine T3 at all']+=1; continue
    best=max((stage(r['ev'],p,t),p,t) for p in r['pls'] for t in r['t3'])
    c[lab[best[0]]]+=1
    if best[0]==6:
        # evaluate pT3 DNN score for best pairs (barrel only)
        e=ev[r['ev']]; sc=[]
        for p in r['pls']:
            for t in r['t3']:
                if stage(r['ev'],p,t)!=6: continue
                # temporarily map: find a T5 whose inner T3 is t is not needed; call internal pieces
                q=None
                idx=np.where(e['t5_t3Idx0']==t)[0]
                if len(idx): 
                    x=D.evaluate(r['ev'],p,idx[0])
                    if x:
                        thr=WPH3 if e['pLS_pt'][p]>5 else WP3[min(9,int(abs(e['pLS_eta'][p])/0.25))]
                        sc.append(x['score']>thr)
        dnnres['pass pT3WP' if any(sc) else ('fail pT3WP' if sc else 'not evaluable')]+=1
print('bucket N=',len(B))
for k,v in c.most_common(): print(f'  {v:4d}  {k}')
print('  DNN(pT3WP) for those reaching it:',dict(dnnres))

#!/usr/bin/env python3
"""Features distinguishing genuine pT5 loser (g) from its alive FAKE killer (w) in RemoveDupPixelQuintupletsFromMap.
Pairs from exact replay (all dR). Prints fraction of pairs where each feature favours g."""
import sys, collections, numpy as np, uproot
sys.path.insert(0, sys.argv[0].rsplit('/',1)[0])
from killed_pt5_attrib import hkey, dphi, GEN, DETA, DPHI, NM
FN=sys.argv[1]; NEV=int(sys.argv[2])
B=['pT5_plsIdx','pT5_t5Idx','pT5_score','pT5_isDupReco','pT5_isFake','pT5_simIdxAll','pT5_simIdxAllFrac','pT5_pt',
   'pLS_lsIdx','pLS_pt','pLS_eta','pLS_phi','ls_mdIdx0','ls_mdIdx1','md_anchor_x','md_anchor_y','md_anchor_z','md_other_x','md_other_y','md_other_z',
   't3_lsIdx0','t3_lsIdx1','t5_t3Idx0','t5_t3Idx1','t5_eta','t5_phi','t5_dnnScore','t5_score','t5_pt','t5_innerRadius','t5_outerRadius','t5_bridgeRadius',
   't5_simIdxAll','t5_simIdxAllFrac','md_layer','t5_isFake']
t=uproot.open(FN)['tree']; F=collections.defaultdict(list)
for s0 in range(0,NEV,10):
  A=t.arrays(B,entry_start=s0,entry_stop=min(NEV,s0+10))
  for ev in A:
    g=lambda k: np.asarray(ev[k])
    ax,ay,az,ox,oy,oz=[g(k) for k in ('md_anchor_x','md_anchor_y','md_anchor_z','md_other_x','md_other_y','md_other_z')]
    l0,l1,a0,a1,q0,q1=[g(k) for k in ('ls_mdIdx0','ls_mdIdx1','t3_lsIdx0','t3_lsIdx1','t5_t3Idx0','t5_t3Idx1')]
    mdl=g('md_layer'); pl=g('pLS_lsIdx')
    mdh=lambda m:[hkey(ax[m],ay[m],az[m]),hkey(ox[m],oy[m],oz[m])]
    def t5m(t5):
      ms=[]
      for t3 in (q0[t5],q1[t5]):
        for l in (a0[t3],a1[t3]):
          for m in (int(l0[l]),int(l1[l])):
            if m not in ms: ms.append(m)
      return ms
    pp=g('pT5_plsIdx').astype(int); p5=g('pT5_t5Idx').astype(int); sc=g('pT5_score'); dup=g('pT5_isDupReco').astype(bool); fk=g('pT5_isFake').astype(bool)
    n=len(sc); eta=g('t5_eta')[p5]; phi=g('t5_phi')[p5]
    mds=[t5m(p5[i]) for i in range(n)]
    allh=[mdh(int(l0[pl[pp[i]]]))+mdh(int(l1[pl[pp[i]]]))+sum((mdh(m) for m in mds[i]),[]) for i in range(n)]
    als=[set(h) for h in allh]
    dnn=g('t5_dnnScore')[p5]; ts=g('t5_score')[p5]; t5pt=g('t5_pt')[p5]; plspt=g('pLS_pt')[pp]
    ir=g('t5_innerRadius')[p5]; orr=g('t5_outerRadius')[p5]; br=g('t5_bridgeRadius')[p5]
    firstlayer=np.array([mdl[mds[i][0]] for i in range(n)]) if n else np.zeros(0)
    t5fake=g('t5_isFake')[p5]
    for i in range(n):
      if not dup[i] or fk[i]: continue
      for j in range(n):
        if j==i or dup[j] or not fk[j]: continue
        if abs(eta[i]-eta[j])>DETA or abs(dphi(phi[i],phi[j]))>DPHI: continue
        if sum(1 for h in allh[i] if h in als[j])<NM: continue
        if not (sc[i]>sc[j] or (sc[i]==sc[j] and i>j)): continue
        F['samepls'].append(pp[i]==pp[j])
        F['dnn_g_higher'].append(dnn[i]>dnn[j])
        F['t5score_g_lower'].append(ts[i]<ts[j])
        F['ptconsist_g_better'].append(abs(np.log(t5pt[i]/plspt[i]))<abs(np.log(t5pt[j]/plspt[j])))
        F['radconsist_g_better'].append(np.std(np.log([ir[i],orr[i],br[i]]))<np.std(np.log([ir[j],orr[j],br[j]])))
        F['g_starts_innermost'].append(firstlayer[i]<firstlayer[j])
        F['g_starts_same'].append(firstlayer[i]==firstlayer[j])
        F['w_T5_isFake'].append(bool(t5fake[j]))
        F['g_T5_isFake'].append(bool(t5fake[i]))
        F['score_ratio_w_over_g'].append(float(sc[j]/max(sc[i],1e-6)))
        F['dnn_g'].append(float(dnn[i])); F['dnn_w'].append(float(dnn[j]))
for k,v in F.items():
  v=np.array(v)
  if v.dtype==bool: print('%-22s %.3f (n=%d)'%(k,v.mean(),len(v)))
  else: print('%-22s median %.3f  q25 %.3f q75 %.3f'%(k,np.median(v),np.quantile(v,.25),np.quantile(v,.75)))

"""Per-sim-track funnel + attribution for jet-core tracks (real pLS, AB-ON, 100 evt).
Genuine = match fraction >= 0.75.  Keys per-event (no cross-file alignment)."""
import pickle, numpy as np, collections, json, sys
GEN=0.75
ev=pickle.load(open('ev.pkl','rb'))
rows=[]
for ie,e in enumerate(ev):
    ns=len(e['sim_pt'])
    vperp=np.hypot(e['sim_vx'],e['sim_vy'])
    gi=e['sim_genjet_idx']
    for s in range(ns):
        if gi[s]<0: continue
        if not (e['sim_q'][s]!=0 and e['sim_pt'][s]>0.9 and abs(e['sim_eta'][s])<4.5 and abs(e['sim_vz'][s])<30 and vperp[s]<2.5): continue
        if not (e['genjet_pt'][gi[s]]>1000 and abs(e['genjet_eta'][gi[s]])<2.5): continue
        def gen(k):
            idx=np.asarray(e['sim_%sIdxAll'%k][s]); fr=np.asarray(e['sim_%sIdxAllFrac'%k][s])
            return list(idx[fr>=GEN].astype(int))
        r=dict(ev=ie,sim=s,pt=float(e['sim_pt'][s]),eta=float(e['sim_eta'][s]),dR=float(e['sim_genjet_deltaR'][s]),
               tc=int(e['sim_tcIdx'][s]),pdg=int(e['sim_pdgId'][s]),
               md=gen('md'),ls=gen('ls'),t3=gen('t3'),t5=gen('t5'),pls=gen('pls'),pt3=gen('pt3'),pt5=gen('pt5'),t4=gen('t4'))
        rows.append(r)
pickle.dump(rows,open('rows.pkl','wb'))
core=[r for r in rows if r['dR']<0.02]
fail=[r for r in core if r['tc']<0]
b176=[r for r in fail if r['pls'] and not r['pt5']]
print('core denom',len(core),'fail',len(fail),'pls&!pt5',len(b176))

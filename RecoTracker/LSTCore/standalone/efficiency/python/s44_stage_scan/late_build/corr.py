import pickle,numpy as np,collections,csv
from attrib3 import *
M=collections.defaultdict(lambda:[0,0]); ptchk={}
for d in csv.DictReader(open('../early_md_ls/mdable_modules_diag.csv')):
    k=(int(d['evt']),int(d['isim'])); ptchk[k]=float(d['pt'])
    if d['subdet']=='5' and int(d['layer'])<=3:
        M[k][0]+=1; M[k][1]+= d['merged']=='True'
bad=sum(1 for (e_,s),pt in ptchk.items() if abs(ev[e_]['sim_pt'][s]-pt)>1e-3); print('csv rows keyed; pt mismatches vs my ev index:',bad,'of',len(ptchk))
def feats(r,b):
    e=ev[r['ev']]; st,p,q=b
    fr=dict(zip(e['sim_plsIdxAll'][r['sim']],e['sim_plsIdxAllFrac'][r['sim']]))
    f5=dict(zip(e['sim_t5IdxAll'][r['sim']],e['sim_t5IdxAllFrac'][r['sim']]))
    m=M.get((r['ev'],r['sim']),[0,0])
    return dict(plsfrac1=fr.get(p,0)>0.99,quad=bool(e['pLS_isQuad'][p]),t5frac1=f5.get(q,0)>0.99,merged=m[1]>0,pt=r['pt'])
ex=pickle.load(open('attrib3_0.02.pkl','rb'))  # from last run (PLSDUP=none)
S6=[feats(r,b) for r,b in ex[ST[7]]]
ctl=[]
for r in rows:
    if r['dR']<0.02 and r['tc']>=0 and r['pls'] and r['pt5']:
        k,b=classify(r)
        if k==ST[7] and r['pt']>100: ctl.append(feats(r,b))
for name,S in [('stage6 fails (pLS-dup ignored)',S6),('control successes simPt>100',ctl)]:
    print(name,'N=%d'%len(S),' '.join('%s=%.2f'%(k,np.mean([s[k] for s in S])) for k in ('plsfrac1','quad','t5frac1','merged')),'medpt=%.0f'%np.median([s['pt'] for s in S]))

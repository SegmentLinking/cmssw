"""Among core tracks where >=1 (genuine pLS, genuine T5) pair passes ALL emulated cuts INCLUDING current pLS isDup:
genuine-pT5 build rate split by whether such passing pairs include a quad pLS, and by simPt."""
import numpy as np, collections, sys
from attrib3 import *
CUT=float(sys.argv[1]) if len(sys.argv)>1 else 0.02
tab=collections.defaultdict(lambda:[0,0])
for r in rows:
    if not (r['dR']<CUT and r['pls'] and r['t5']): continue
    e=ev[r['ev']]; e2=ev2[r['ev']]
    ok=[(p,q) for p in r['pls'] for q in r['t5'] if pairstage(e,e2,r['ev'],p,q)==7]
    if not ok: continue
    hasq=any(e['pLS_isQuad'][p] for p,q in ok)
    ptb='<50' if r['pt']<50 else '50-150' if r['pt']<150 else '>150'
    k=('quad' if hasq else 'triplet-only',ptb); tab[k][0]+=1; tab[k][1]+=bool(r['pt5'])
for k in sorted(tab): print(k,'N=%d built genuine pT5=%d (%.0f%%)'%(tab[k][0],tab[k][1],100*tab[k][1]/tab[k][0]))

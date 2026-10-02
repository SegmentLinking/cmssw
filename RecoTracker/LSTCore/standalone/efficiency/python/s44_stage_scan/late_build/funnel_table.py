import numpy as np, pickle
from attrib3 import ev, rows, dup
def fun(S,lab):
    n=len(S)
    a=[r for r in S if r['pls']]
    a2=[r for r in a if any(not dup[r['ev']][p] for p in r['pls'])]
    b=[r for r in a2 if r['t3']]; c=[r for r in b if r['t5']]
    c2=[r for r in c if any((ev[r['ev']]['t5_isDupBits'][q]&1)==0 for q in r['t5'])]
    d=[r for r in c2 if r['pt5']]; t=[r for r in d if r['tc']>=0]
    pt3=[r for r in S if r['pt3']]
    print(f"{lab:26s} N={n:5d} | pLS {len(a)/n:.2f} | pLS!dup {len(a2)/n:.2f} | +T3 {len(b)/n:.2f} | +T5 {len(c)/n:.2f} | +T5 alive(AB) {len(c2)/n:.2f} | +pT5 {len(d)/n:.2f} | +TC {len(t)/n:.2f} || any genuine pT3 {len(pt3)/n:.3f}  TCeff {sum(r['tc']>=0 for r in S)/n:.3f}")
for lo,hi,l in [(0,0.02,'core dR<0.02'),(0.02,0.1,'0.02-0.1'),(0.1,0.4,'0.1-0.4')]:
    S=[r for r in rows if lo<=r['dR']<hi]; fun(S,l)
    fun([r for r in S if r['pt']>100],l+' pt>100')

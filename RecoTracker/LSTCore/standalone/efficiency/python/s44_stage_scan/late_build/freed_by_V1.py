"""Tracks whose genuine pLS are all killed by current CheckHitspLS pass-1 but freed by unique-hit counting (V1) / V3:
how many then pass radius + emulated pT3-DNN(pT5WP)?"""
import pickle, collections, numpy as np, sys
import attrib3 as A, pt3dnn as D
V=pickle.load(open('plsdup_variants.pkl','rb')); cur=pickle.load(open('plsdup.pkl','rb'))
B=[r for r in A.rows if r['dR']<0.02 and r['tc']<0 and r['pls'] and not r['pt5']]
for tag in ('V1','V3'):
    c=collections.Counter()
    for r in B:
        if not r['t5']: continue
        A.dup=cur; s0=max(A.pairstage(A.ev[r['ev']],A.ev2[r['ev']],r['ev'],p,q) for p in r['pls'] for q in r['t5'])
        if s0!=4: continue
        A.dup=V[tag]
        P=[(p,q) for p in r['pls'] for q in r['t5'] if A.pairstage(A.ev[r['ev']],A.ev2[r['ev']],r['ev'],p,q)==7]
        c['freed-from-isDup tracks']+=1
        if not P: c['  still fail (radius/AB/still dup)']+=1; continue
        res=[x for x in (D.evaluate(r['ev'],p,q) for p,q in P) if x]
        c['  pass radius']+=1
        c['  pass radius + DNN' if any(x['ok'] for x in res) else ('  DNN rejects' if res else '  not evaluable')]+=1
    print(tag,dict(c))

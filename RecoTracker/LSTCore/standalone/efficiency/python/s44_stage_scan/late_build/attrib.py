"""Attribution for jet-core failures with genuine pLS but no genuine pT5.
Real pLS, AB-ON, 100 evt.  Offline emulation of passRadiusCriterion (PixelTriplet.h:347-463)."""
import pickle, numpy as np, collections, sys
kR1GeVf = 1./(2.99792458e-3*3.8)
ev=pickle.load(open('ev.pkl','rb')); rows=pickle.load(open('rows.pkl','rb'))
CUT = float(sys.argv[1]) if len(sys.argv)>1 else 0.02

BOUNDS={'BBB':((0.15624,0.17235),(0.6588,0.6375)),
        'BBE':((0.45972,0.19644),(0.8557,0.6805)),
        'BEE':((1.59294,0.255181),(2.3548,2.2091)),
        'EEE':((1.7006,0.26367),(2.436,2.286))}
def radius_pass(pR, pRe, tR, kind):
    (tb,pb),(tbh,pbh)=BOUNDS[kind]
    if pR>2*kR1GeVf: tb,pb=tbh,pbh
    tmax=(1+tb)/tR; tmin=max((1-tb)/tR,0.)
    pmax=max((1+pb)/pR, 1./(pR-pRe)); pmin=min((1-pb)/pR, 1./(pR+pRe))
    if kind in('BEE','EEE'): pmin=max(pmin,0.)
    return (tmin<=pmin<tmax) or (pmin<tmin<pmax)
def t3kind(e,i):
    L=[e['t3_hit_%d_layer'%k][i] for k in (0,2,4)]
    E=[l>6 for l in L]
    if E[0]: return 'EEE'
    if E[1]: return 'BEE'
    if E[2]: return 'BBE'
    return 'BBB'

core=[r for r in rows if r['dR']<CUT]
fail=[r for r in core if r['tc']<0]
B=[r for r in fail if r['pls'] and not r['pt5']]
print(f'dR<{CUT}: core {len(core)} fail {len(fail)} bucket(pLS & no genuine pT5) {len(B)}')

cat=collections.Counter(); detail=collections.defaultdict(list)
for r in B:
    e=ev[r['ev']]
    pls=r['pls']; t5=r['t5']; t3=r['t3']
    quad=[i for i in pls if e['pLS_isQuad'][i]]
    # pLS used in any pT5?
    p5pls=set(e['pT5_plsIdx']); p5t5=set(e['pT5_t5Idx'])
    pls_used=[i for i in pls if i in p5pls]
    t5_alive=[i for i in t5 if (e['t5_isDupBits'][i]&1)==0]
    t5_tried=[i for i in t5_alive if e['t5_triedInPT5'][i]]
    t5_used=[i for i in t5 if e['t5_partOfPT5'][i]]
    # radius criterion for genuine pLS x genuine alive T5 (inner T3)
    rad=[]
    for p in pls:
        pR=e['pLS_pt'][p]*kR1GeVf; pRe=e['pLS_ptErr'][p]*kR1GeVf
        for q in t5_alive:
            it3=e['t5_t3Idx0'][q]
            rad.append(radius_pass(pR,pRe,e['t3_radius'][it3],t3kind(e,it3)))
    r.update(nquad=len(quad),pls_used=len(pls_used),t5_alive=len(t5_alive),t5_tried=len(t5_tried),t5_used=len(t5_used),
             rad_any=any(rad) if rad else None, pt3_any=bool(r['pt3']))
    if not t5:
        k='1 no genuine T5' + (' (has genuine T3)' if t3 else ' (no genuine T3)')
    elif not t5_alive:
        k='2 genuine T5 all killed by AfterBuild (bit0) before pT5 build'
    elif not t5_tried:
        k='3 alive genuine T5 never tried (module not connected to any pLS superbin / 2S inner)'
    elif rad and not any(rad):
        k='4 tried; ALL genuine (pLS,T5) pairs fail radius criterion'
    else:
        k='5 tried; radius ok for >=1 pair -> fails later pT3-algo/DNN/chi2 (or pLS isDup)'
    cat[k]+=1; detail[k].append(r)
for k in sorted(cat): print(f'{cat[k]:4d}  {k}')
print('\nsub-info per category: nquad>0, pls used in some pT5, genuine T5 used in some (wrong) pT5, genuine pT3 built; median sim pt')
for k in sorted(detail):
    d=detail[k]
    print(k[:2], 'n=%d quadpLS=%d plsUsed=%d t5Used=%d pt3=%d medpt=%.0f'%(len(d),sum(x['nquad']>0 for x in d),sum(x['pls_used']>0 for x in d),
          sum(x['t5_used']>0 for x in d),sum(x['pt3_any'] for x in d),np.median([x['pt'] for x in d])))
pickle.dump(B,open('bucket_%s.pkl'%CUT,'wb'))

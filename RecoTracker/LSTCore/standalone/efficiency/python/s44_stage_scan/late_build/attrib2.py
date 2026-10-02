"""Stage-by-stage attribution for genuine (pLS, T5) pairs of core tracks with genuine pLS but no genuine pT5.
Emulates exactly (from ntuple + trackingNtuple seeds + pixel map files):
  pixelType/superbin validity, pixel-map connectivity (LSTEvent.dev.cc:1118-1150, PixelQuintuplet.h:657-670),
  inner-module 2S veto (PixelQuintuplet.h:665), T5 AfterBuild isDup (682), radius criterion (PixelTriplet.h:584).
Remaining (not emulable): pLS first-pass isDup (Kernels.h:503), PPBB/PPEE tracklet cuts, pT3-DNN pT5WP, pT5 chi2."""
import pickle, numpy as np, collections, struct, sys, os
kR1GeVf = 1./(2.99792458e-3*3.8)
ev=pickle.load(open('ev.pkl','rb')); ev2=pickle.load(open('ev2.pkl','rb')); rows=pickle.load(open('rows.pkl','rb'))
plsinfo=pickle.load(open('plsinfo.pkl','rb'))
MD='/cvmfs/cms.cern.ch/el8_amd64_gcc13/cms/cmssw/CMSSW_16_1_1/external/el8_amd64_gcc13/data/RecoTracker/LSTCore/data/OT800_IT615_pt0.8/pixelmap/'
def loadmap(fn):
    d={}; b=open(fn,'rb').read(); o=0
    while o+8<=len(b):
        k,n=struct.unpack_from('II',b,o); o+=8
        d[k]=set(struct.unpack_from('%dI'%n,b,o)); o+=4*n
    return d
L=['_layer1_subdet5','_layer2_subdet5','_layer1_subdet4','_layer2_subdet4']
maps={0:[loadmap(MD+'pLS_map'+x+'.bin') for x in L],1:[loadmap(MD+'pLS_map_pos'+x+'.bin') for x in L],2:[loadmap(MD+'pLS_map_neg'+x+'.bin') for x in L]}
def connected(pi, detids):
    sb=pi['sb']; 
    if sb<0 or sb>=45000: return None
    key = sb+45000 if pi['ptype']==0 else sb
    s=set()
    for m in maps[pi['ptype']]: s|=m.get(key,set())
    return any(d in s for d in detids)
BOUNDS={'BBB':((0.15624,0.17235),(0.6588,0.6375)),'BBE':((0.45972,0.19644),(0.8557,0.6805)),
        'BEE':((1.59294,0.255181),(2.3548,2.2091)),'EEE':((1.7006,0.26367),(2.436,2.286))}
def radius_pass(pR,pRe,tR,kind):
    (tb,pb),(tbh,pbh)=BOUNDS[kind]
    if pR>2*kR1GeVf: tb,pb=tbh,pbh
    tmax=(1+tb)/tR; tmin=max((1-tb)/tR,0.)
    pmax=max((1+pb)/pR,1./(pR-pRe)); pmin=min((1-pb)/pR,1./(pR+pRe))
    if kind in('BEE','EEE'): pmin=max(pmin,0.)
    return (tmin<=pmin<tmax) or (pmin<tmin<pmax)
def t3kind(e,i):
    E=[e['t3_hit_%d_layer'%k][i]>6 for k in (0,2,4)]
    return 'EEE' if E[0] else 'BEE' if E[1] else 'BBE' if E[2] else 'BBB'

STAGES=['a: no genuine T5 (T3 exists)','a: no genuine T5 (no genuine T3)',
        'b: genuine T5s all AfterBuild-isDup',
        'c: genuine pLS invalid superbin (|eta_PCA|>2.6)',
        'd: alive genuine T5 inner module 2S',
        'e: pixel map: genuine T5 module NOT connected to genuine pLS superbin',
        'f: radius criterion fails for all connected pairs',
        'g: pair passes all emulated cuts -> pLS-isDup / tracklet(PPBB/PPEE) / pT3-DNN(pT5WP) / pT5 chi2']
def classify(r):
    e=ev[r['ev']]; e2=ev2[r['ev']]; pi_=plsinfo[r['ev']]
    if not r['t5']: return STAGES[0] if r['t3'] else STAGES[1], None
    alive=[q for q in r['t5'] if (e['t5_isDupBits'][q]&1)==0]
    if not alive: return STAGES[2], None
    pls=[p for p in r['pls'] if pi_[p] and 0<=pi_[p]['sb']<45000]
    if not pls: return STAGES[3], None
    alive_ps=[q for q in alive if e['t3_hit_0_moduleType'][e['t5_t3Idx0'][q]]==0]
    if not alive_ps: return STAGES[4], None
    conn=[(p,q) for p in pls for q in alive_ps if connected(pi_[p],[int(e2['t3_hit_0_detId'][e['t5_t3Idx0'][q]]),int(e2['t3_hit_1_detId'][e['t5_t3Idx0'][q]])])]
    if not conn: return STAGES[5], None
    rad=[(p,q) for p,q in conn if radius_pass(e['pLS_pt'][p]*kR1GeVf,e['pLS_ptErr'][p]*kR1GeVf,
                                            e['t3_radius'][e['t5_t3Idx0'][q]],t3kind(e,e['t5_t3Idx0'][q]))]
    if not rad: return STAGES[6], conn
    return STAGES[7], rad

if __name__=='__main__':
    CUT=float(sys.argv[1]) if len(sys.argv)>1 else 0.02
    for label,sel in [('FAIL: genuine pLS, no genuine pT5',lambda r:r['tc']<0 and r['pls'] and not r['pt5']),
                      ('SUCCESS-with-pT5 (reference)',lambda r:r['tc']>=0 and r['pls'] and r['pt5'])]:
        S=[r for r in rows if r['dR']<CUT and sel(r)]
        c=collections.Counter(); ex=collections.defaultdict(list)
        for r in S:
            k,pairs=classify(r); c[k]+=1; ex[k].append((r,pairs))
        print(f'\n=== dR<{CUT}  {label}: N={len(S)}')
        for k in STAGES:
            if c[k]: print(f'  {c[k]:4d} ({100*c[k]/len(S):5.1f}%)  {k}   median simpt={np.median([x[0]["pt"] for x in ex[k]]):.0f}')
        if label.startswith('FAIL'): pickle.dump(ex,open('attrib2_%s.pkl'%CUT,'wb'))

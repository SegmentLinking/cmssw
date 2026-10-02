"""Emulate pT3-DNN with pT5WP (NeuralNetwork.h:143-191, weights pT3NeuralNetworkWeights.h, WPs interface/alpaka/Common.h:103-120)
inside pT5 building (PixelQuintuplet.h:502) incl. rz chi2 (PixelTriplet.h:467) and rphi chi2 (PixelTriplet.h:257).
Validated on BUILT pT5s (must pass ~100%)."""
import re, struct, numpy as np, pickle, math, collections, warnings, uproot
warnings.filterwarnings('ignore')
from attrib3 import ev, ev2, rows, plsinfo, pairstage, kR1GeVf
W=open('/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/src/alpaka/pT3NeuralNetworkWeights.h').read()
def arr(name):
    m=re.search(name+r'\s*(\[[^=]*\])\s*=\s*\{(.*?)\};',W,re.S); v=np.array([float(x.rstrip('f')) for x in re.findall(r'-?\d+\.\d+(?:e-?\d+)?f?',m.group(2))],np.float32)
    dims=[int(d) for d in re.findall(r'\[(\d+)\]',m.group(1))]; return v.reshape(dims)
b1,w1,b2,w2,bo,wo=arr('bias_layer1'),arr('wgtT_layer1'),arr('bias_layer2'),arr('wgtT_layer2'),arr('bias_output_layer'),arr('wgtT_output_layer')
WP=np.array([0.1227,0.1901,0.218,0.3438,0.1011,0.1502,0.0391,0.0471,0.1444,0.1007]); WPH=0.1498
def dnn(rphi,tR,pR,pRe,rz,eta,pt,mt3):
    x=np.array([math.log10(rphi),math.log10(tR),math.log10(pR),math.log10(pRe),math.log10(rz),abs(eta)/2.5,float(mt3)],np.float32)
    h=np.maximum(b1+x@w1,0); h=np.maximum(b2+h@w2,0); o=(bo+h@wo)[0]; s=1/(1+math.exp(-o))
    thr=WPH if pt>5 else WP[9 if abs(eta)>2.5 else int(abs(eta)/0.25)]
    return s, s>thr
TG={}
b=open('/cvmfs/cms.cern.ch/el8_amd64_gcc13/cms/cmssw/CMSSW_16_1_1/external/el8_amd64_gcc13/data/RecoTracker/LSTCore/data/OT800_IT615_pt0.8/tilted_barrel_orientation.bin','rb').read()
for o in range(0,len(b)-11,12):
    d,rz_,dx=struct.unpack_from('Iff',b,o); TG[d]=(rz_,dx)
T=uproot.open('/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/trackingNtuple-100.root')['trackingNtuple/tree']
A=T.arrays(['sim_pt','see_stateTrajGlbX','see_stateTrajGlbY','see_stateTrajGlbZ'],library='np')
fp={np.asarray(A['sim_pt'][i],dtype=np.float32).tobytes():i for i in range(T.num_entries)}
TI=[fp[np.asarray(e['sim_pt'],dtype=np.float32).tobytes()] for e in ev]
inv1=0.01/0.009; inv2=0.15/0.009; k2=(2.99792458e-3*3.8)/2
def geom(e,e2,it3,k):
    det=int(e2['t3_hit_%d_detId'%k][it3]) if k in (0,1) else None
    return det
def evaluate(ie,p,q):
    e=ev[ie]; e2=ev2[ie]; it3=e['t5_t3Idx0'][q]
    L=[e['t3_hit_%d_layer'%k][it3] for k in (0,2,4)]
    if any(l>6 for l in L): return None  # barrel-only emulation
    xs=[e['t3_hit_%d_x'%k][it3] for k in (0,2,4)]; ys=[e['t3_hit_%d_y'%k][it3] for k in (0,2,4)]; zs=[e['t3_hit_%d_z'%k][it3] for k in (0,2,4)]
    mts=[e['t3_hit_%d_moduleType'%k][it3] for k in (0,2,4)]
    # detIds only stored for hit0/hit1 in ev2; get others from ntuple? use z-based tilt lookup via detId of hit0 only for module 0; others: approximate via TG side from detId unavailable -> load below
    dets=DETS[ie][it3]
    sides=[(d>>18)&3 for d in dets]; drdz=[TG.get(d,TG.get(d-1,TG.get(d+1,(0,0))))[0] for d in dets]
    # rphi chi2 with pixel circle
    g,f,R=e['pLS_circleCenterX'][p],e['pLS_circleCenterY'][p],e['pLS_circleRadius'][p]
    c=g*g+f*f-R*R; chi=0.
    for i in range(3):
        flat = (mts[i]==1) or sides[i]==3
        if mts[i]==1: d1=d2=1.
        elif sides[i]==3: d1=d2=inv1
        else: d1=inv1; d2=inv2*drdz[i]/math.sqrt(1+drdz[i]**2)
        x,y=xs[i],ys[i]
        if flat: xp,yp=x,y
        else:
            slope=TG.get(dets[i],(0,0))[1]; aas=abs(math.atan(slope)) if math.isfinite(slope) and slope!=123456789 else math.pi/2
            am = (math.pi/2-aas) if (x>0 and y>0) else (aas+math.pi/2) if (x<0 and y>0) else -(aas+math.pi/2) if (x<0 and y<0) else -(math.pi/2-aas)
            xp=x*math.cos(am)+y*math.sin(am); yp=y*math.cos(am)-x*math.sin(am)
        s2=4*((xp*d1)**2+(yp*d2)**2); r_=x*x+y*y-2*g*x-2*f*y+c; chi+=r_*r_/s2
    # rz chi2 (float64)
    s=plsinfo[ie][p]['seed']; ti=TI[ie]
    x1,y1,z1=A['see_stateTrajGlbX'][ti][s]/100,A['see_stateTrajGlbY'][ti][s]/100,A['see_stateTrajGlbZ'][ti][s]/100; r1=math.hypot(x1,y1)
    Px,Py,Pz=float(e['pLS_px'][p]),float(e['pLS_py'][p]),float(e['pLS_pz'][p]); a=-2*k2*100*e['pLS_charge'][p]; P=math.sqrt(Px*Px+Py*Py+Pz*Pz); rou=a/P; RM=0
    for i in range(3):
        zsi=zs[i]/100; rt=math.hypot(xs[i],ys[i])/100
        pA=r1*r1+2*(Px*Px+Py*Py)/(a*a)+2*(y1*Px-x1*Py)/a-rt*rt; pB=2*(x1*Px+y1*Py)/a; pC=2*(y1*Px-x1*Py)/a+2*(Px*Px+Py*Py)/(a*a)
        AA=pB*pB+pC*pC; B=2*pA*pB; C=pA*pA-pC*pC; D=math.sqrt(max(B*B-4*AA*C,0))
        zz=[math.asin(max(-1,min(1,sv)))/rou*Pz/P+z1 for sv in ((-B+D)/(2*AA),(-B-D)/(2*AA))]
        res=min(abs(z-zsi) for z in zz)*100
        e2_=0.15**2 if mts[i]==0 else 25.
        if mts[i]==0 and sides[i]!=3: e2_/=(1+drdz[i]**2)
        RM+=res*res/e2_
    rz=math.sqrt(0.2*RM)
    pR=e['pLS_pt'][p]*kR1GeVf; pRe=e['pLS_ptErr'][p]*kR1GeVf
    sc,ok=dnn(max(chi,1e-9),e['t3_radius'][it3],pR,max(pRe,1e-9),max(rz,1e-9),e['pLS_eta'][p],e['pLS_pt'][p],mts[2])
    return dict(score=sc,ok=ok,rphi=chi,rz=rz)
# detIds for all three anchor hits
F='/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_realpls_mastercuts_100evt_v2.root'
t=uproot.open(F)['tree']; dd=t.arrays(['t3_hit_0_detId','t3_hit_2_detId','t3_hit_4_detId'],library='np')
DETS=[np.stack([dd['t3_hit_%d_detId'%k][i].astype(np.int64) for k in (0,2,4)],1) for i in range(t.num_entries)]
if __name__=='__main__':
    # validation on built pT5s
    v=[]
    for ie,e in enumerate(ev):
        for p,q in list(zip(e['pT5_plsIdx'],e['pT5_t5Idx']))[:60]:
            r_=evaluate(ie,p,q)
            if r_: v.append(r_['ok'])
    print('VALIDATION built pT5 pairs (barrel T3): N=%d emulated DNN pass=%.3f'%(len(v),np.mean(v)))
    out=collections.defaultdict(list)
    for r in rows:
        if not(r['dR']<0.02 and r['pls'] and r['t5']): continue
        e=ev[r['ev']]
        P=[(p,q) for p in r['pls'] for q in r['t5'] if pairstage(e,ev2[r['ev']],r['ev'],p,q)==7]
        if not P: continue
        res=[x for x in (evaluate(r['ev'],p,q) for p,q in P) if x]
        if not res: continue
        g=('built' if r['pt5'] else 'NOTbuilt')+('_pt>150' if r['pt']>150 else '_pt<150')
        out[g].append((r,res))
    for g in sorted(out):
        n=len(out[g]); anyok=sum(any(x['ok'] for x in res) for r,res in out[g])
        print(f'{g:16s} tracks(barrel pairs)={n:4d}  >=1 pair passes emulated DNN: {anyok} ({100*anyok/n:.0f}%)')
    pickle.dump(dict(out),open('pt3dnn_out.pkl','wb'))

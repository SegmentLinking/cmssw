"""pLS pT resolution vs claimed ptErr (inputs to radius criterion + pT3/pT5 DNN), genuine pLS of denominator tracks."""
import pickle,numpy as np
ev=pickle.load(open('ev.pkl','rb')); rows=pickle.load(open('rows.pkl','rb'))
print('region   simPt     N   med(pLSpt/simpt)  IQR/2(1/pt rel)  med(ptErr/pt)  |pull|>3 frac   quad-only |pull|>3')
for reg,sel in [('core<0.02',lambda r:r['dR']<0.02),('0.1-0.4',lambda r:0.1<r['dR']<0.4)]:
    for lo,hi in [(1,10),(10,50),(50,150),(150,5000)]:
        R=[];P=[];E=[];Q=[]
        for r in rows:
            if not(sel(r) and lo<=r['pt']<hi): continue
            e=ev[r['ev']]
            for p in r['pls']:
                pt=e['pLS_pt'][p]; er=e['pLS_ptErr'][p]
                R.append(pt/r['pt']); E.append(er/pt); P.append((1/pt-1/r['pt'])/(er/pt**2)); Q.append(e['pLS_isQuad'][p])
        if not R: continue
        R=np.array(R);P=np.array(P);E=np.array(E);Q=np.array(Q,bool)
        inv=1/R; iqr=(np.percentile(inv,75)-np.percentile(inv,25))/2
        print(f'{reg:9s} [{lo},{hi})  {len(R):5d}   {np.median(R):.2f}   {iqr:.3f}   {np.median(E):.4f}   {np.mean(abs(P)>3):.2f}   {np.mean(abs(P[Q])>3):.2f}')

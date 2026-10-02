#!/usr/bin/env python3
"""Created genuine T5s: margin of the new T5 DNN score (1-P(fake), stored as t5_dnnScore, Quintuplet.h:1687) above its
creation WP (interface/alpaka/Common.h:97-99; pt bin = inner-T3-radius pT>5, eta bin = |eta| of first anchor/0.25,
NeuralNetwork.h:295-297). If jet-core genuine T5s pile up just above the WP (vs. isolated), the DNN is likely rejecting
a larger fraction of core genuine T5 candidates just below it. Only CREATED T5s are in the ntuple."""
import numpy as np, collections
from load47 import ev, rows
import uproot
kR1GeVf = 1. / (2.99792458e-3 * 3.8)
WP = np.array([[0.8838, 0.8933, 0.9270, 0.9192, 0.8223, 0.8524, 0.9099, 0.9383, 0.9640, 0.9608],
               [0.9761, 0.9744, 0.9855, 0.9765, 0.9327, 0.8803, 0.8803, 0.8978, 0.9049, 0.8822]])
t = uproot.open('/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_s45_rebased_100evt.root')['tree']
X = t.arrays(['t5_t3_0_eta', 'sim_genjet_deltaR', 'sim_genjet_idx', 'genjet_pt'], library='np')
M = collections.defaultdict(list)
for ie, e in enumerate(ev):
    dr = X['sim_genjet_deltaR'][ie]; gj = X['sim_genjet_idx'][ie]; gpt = X['genjet_pt'][ie]
    for q in range(len(e['t5_pt'])):
        s = e['t5_simIdx'][q]
        if s < 0:
            cat = 'fake/partial'
        else:
            j = gj[s]
            jetpt = gpt[j] if 0 <= j < len(gpt) else 0
            cat = 'genuine core(dR<0.02,jet>1TeV)' if (0 <= dr[s] < 0.02 and jetpt > 1000) else ('genuine dR>0.1 / no jet' if (dr[s] < 0 or dr[s] > 0.1) else 'genuine other')
        pb = int(e['t5_innerRadius'][q] / kR1GeVf > 5)
        eta = abs(X['t5_t3_0_eta'][ie][e['t5_t3Idx0'][q]])
        eb = 9 if eta > 2.5 else int(eta / 0.25)
        M[(cat, pb)].append(e['t5_dnnScore'][q] - WP[pb][eb])
print('margin = dnnScore - WP for CREATED T5s; frac within 0.01 / 0.03 of WP; median margin; N')
for k in sorted(M):
    m = np.array(M[k])
    print(f'  {k[0]:34s} ptbin {k[1]}: N={len(m):7d} frac<0.01 {np.mean(m < 0.01):.3f} frac<0.03 {np.mean(m < 0.03):.3f} median {np.median(m):.4f}  min {m.min():.4f}')

#!/usr/bin/env python3
"""Port of S44 pt3dnn.py to the S45 ntuple: emulate the pT3-DNN at the pT5 WP (PixelTriplet.h:688, NeuralNetwork.h pt3dnn,
WPs interface/alpaka/Common.h:111-114) with its rz-chi2 (computePT3RZChiSquared) and pixel-circle rphi-chi2
(computePT3RPhiChiSquared) inputs, for barrel inner T3s. Validate on built pT5s, then apply to 'pair_failed' stage-5 tracks.
Also reports pLS circleRadius (the pT5 chi2 cuts at PixelQuintuplet.h:604/633 apply only if circleRadius < 5 GeV)."""
import re, struct, math, collections, pickle, numpy as np, warnings
warnings.filterwarnings('ignore')
from load47 import ev, rows, plsinfo, SCR
from pair_attrib import pairstage, bucket, kR1GeVf
W = open('/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/src/alpaka/pT3NeuralNetworkWeights.h').read()


def arr(name):
    m = re.search(name + r'\s*(\[[^=]*\])\s*=\s*\{(.*?)\};', W, re.S)
    v = np.array([float(x.rstrip('f')) for x in re.findall(r'-?\d+\.\d+(?:e-?\d+)?f?', m.group(2))], np.float32)
    dims = [int(d) for d in re.findall(r'\[(\d+)\]', m.group(1))]
    return v.reshape(dims)


b1, w1, b2, w2, bo, wo = (arr(x) for x in ('bias_layer1', 'wgtT_layer1', 'bias_layer2', 'wgtT_layer2', 'bias_output_layer', 'wgtT_output_layer'))
WP = np.array([0.1227, 0.1901, 0.218, 0.3438, 0.1011, 0.1502, 0.0391, 0.0471, 0.1444, 0.1007]); WPH = 0.1498


def dnn(rphi, tR, pR, pRe, rz, eta, pt, mt3):
    x = np.array([math.log10(rphi), math.log10(tR), math.log10(pR), math.log10(pRe), math.log10(rz), abs(eta) / 2.5, float(mt3)], np.float32)
    h = np.maximum(b1 + x @ w1, 0); h = np.maximum(b2 + h @ w2, 0); o = (bo + h @ wo)[0]; s = 1 / (1 + math.exp(-o))
    thr = WPH if pt > 5 else WP[9 if abs(eta) > 2.5 else int(abs(eta) / 0.25)]
    return s, s > thr


TG = {}
b = open('/cvmfs/cms.cern.ch/el8_amd64_gcc13/cms/cmssw/CMSSW_16_1_1/external/el8_amd64_gcc13/data/RecoTracker/LSTCore/data/OT800_IT615_pt0.8/tilted_barrel_orientation.bin', 'rb').read()
for o in range(0, len(b) - 11, 12):
    d, rz_, dx = struct.unpack_from('Iff', b, o); TG[d] = (rz_, dx)
inv1 = 0.01 / 0.009; inv2 = 0.15 / 0.009; k2 = (2.99792458e-3 * 3.8) / 2


def evaluate(ie, p, q):
    e = ev[ie]; it3 = e['t5_t3Idx0'][q]
    if any(e['t3_hit_%d_layer' % k][it3] > 6 for k in (0, 2, 4)):
        return None
    xs = [e['t3_hit_%d_x' % k][it3] for k in (0, 2, 4)]; ys = [e['t3_hit_%d_y' % k][it3] for k in (0, 2, 4)]
    zs = [e['t3_hit_%d_z' % k][it3] for k in (0, 2, 4)]; mts = [e['t3_hit_%d_moduleType' % k][it3] for k in (0, 2, 4)]
    dets = [int(e['t3_hit_%d_detId' % k][it3]) for k in (0, 2, 4)]
    sides = [(d >> 18) & 3 for d in dets]; drdz = [TG.get(d, TG.get(d - 1, TG.get(d + 1, (0, 0))))[0] for d in dets]
    g, f, R = e['pLS_circleCenterX'][p], e['pLS_circleCenterY'][p], e['pLS_circleRadius'][p]
    c = g * g + f * f - R * R; chi = 0.
    for i in range(3):
        flat = (mts[i] == 1) or sides[i] == 3
        if mts[i] == 1: d1 = d2 = 1.
        elif sides[i] == 3: d1 = d2 = inv1
        else: d1 = inv1; d2 = inv2 * drdz[i] / math.sqrt(1 + drdz[i] ** 2)
        x, y = xs[i], ys[i]
        if flat: xp, yp = x, y
        else:
            slope = TG.get(dets[i], (0, 0))[1]
            aas = abs(math.atan(slope)) if math.isfinite(slope) and slope != 123456789 else math.pi / 2
            am = (math.pi / 2 - aas) if (x > 0 and y > 0) else (aas + math.pi / 2) if (x < 0 and y > 0) else -(aas + math.pi / 2) if (x < 0 and y < 0) else -(math.pi / 2 - aas)
            xp = x * math.cos(am) + y * math.sin(am); yp = y * math.cos(am) - x * math.sin(am)
        s2 = 4 * ((xp * d1) ** 2 + (yp * d2) ** 2); r_ = x * x + y * y - 2 * g * x - 2 * f * y + c; chi += r_ * r_ / s2
    pi = plsinfo[ie][p]
    x1, y1, z1 = pi['x1'] / 100, pi['y1'] / 100, pi['z1'] / 100; r1 = math.hypot(x1, y1)
    Px, Py, Pz = float(e['pLS_px'][p]), float(e['pLS_py'][p]), float(e['pLS_pz'][p]); a = -2 * k2 * 100 * e['pLS_charge'][p]
    P = math.sqrt(Px * Px + Py * Py + Pz * Pz); rou = a / P; RM = 0
    for i in range(3):
        zsi = zs[i] / 100; rt = math.hypot(xs[i], ys[i]) / 100
        pA = r1 * r1 + 2 * (Px * Px + Py * Py) / (a * a) + 2 * (y1 * Px - x1 * Py) / a - rt * rt; pB = 2 * (x1 * Px + y1 * Py) / a
        pC = 2 * (y1 * Px - x1 * Py) / a + 2 * (Px * Px + Py * Py) / (a * a)
        AA = pB * pB + pC * pC; B = 2 * pA * pB; C = pA * pA - pC * pC; D = math.sqrt(max(B * B - 4 * AA * C, 0))
        zz = [math.asin(max(-1, min(1, sv))) / rou * Pz / P + z1 for sv in ((-B + D) / (2 * AA), (-B - D) / (2 * AA))]
        res = min(abs(z - zsi) for z in zz) * 100
        e2_ = 0.15 ** 2 if mts[i] == 0 else 25.
        if mts[i] == 0 and sides[i] != 3: e2_ /= (1 + drdz[i] ** 2)
        RM += res * res / e2_
    rz = math.sqrt(0.2 * RM)
    pR = e['pLS_pt'][p] * kR1GeVf; pRe = e['pLS_ptErr'][p] * kR1GeVf
    sc, ok = dnn(max(chi, 1e-9), e['t3_radius'][it3], pR, max(pRe, 1e-9), max(rz, 1e-9), e['pLS_eta'][p], e['pLS_pt'][p], mts[2])
    return dict(score=sc, ok=ok, rphi=chi, rz=rz, circR_GeV=R / kR1GeVf)


if __name__ == '__main__':
    v = []; vhi = []
    for ie, e in enumerate(ev):
        for p, q in list(zip(e['pT5_plsIdx'], e['pT5_t5Idx']))[:80]:
            r_ = evaluate(ie, p, q)
            if r_:
                v.append(r_['ok'])
                if e['pLS_pt'][p] > 50: vhi.append(r_['ok'])
    print('VALIDATION built pT5 pairs (barrel inner T3): N=%d emulated DNN pass=%.3f ; pLS pT>50: N=%d pass=%.3f' % (len(v), np.mean(v), len(vhi), np.mean(vhi)))
    fails = [r for r in rows if r['tc'] < 0 and bucket(r) == 'pair_failed']
    c = collections.Counter()
    for r in fails:
        e = ev[r['ev']]
        pls = [p for p in r['pls'] if not (e['pLS_isDup'][p] & 1)]
        P5 = [(p, q) for p in pls for q in r['t5'] if pairstage(r['ev'], p, q) == 4]
        if not P5: continue
        res = [(p, q, evaluate(r['ev'], p, q)) for p, q in P5]
        res = [x for x in res if x[2]]
        if not res:
            c['stage5, no barrel pair (not emulated)'] += 1; continue
        anyok = any(x[2]['ok'] for x in res)
        c['stage5: DNN(pT5WP) fails all pairs' if not anyok else 'stage5: DNN passes >=1 pair -> tracklet PPBB or chi2 (circleR<5GeV)'] += 1
        for p, q, x in res:
            print(f"   simPt {r['pt']:6.1f} pLSpt {e['pLS_pt'][p]:8.1f} circR->pT {x['circR_GeV']:8.1f} rphiChi2 {x['rphi']:9.1f} rzChi2 {x['rz']:6.2f} dnn {x['score']:.3f} pass {int(x['ok'])}")
    for k, n in c.items(): print(f'{n:3d} {k}')

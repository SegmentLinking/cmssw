#!/usr/bin/env python3
"""Load the S45 rebased 100-evt --allobj ntuple (gate OFF) into per-event dicts + per-core-sim rows (genuine = frac>=0.75),
and match every pLS to its trackingNtuple seed (superbin, pixelType as LSTPrepareInput.h; S44 pls2seed.py logic).
Caches to the scratch dir. Import: from load47 import ev, rows, plsinfo, TI, TA"""
import os, pickle, math, numpy as np, uproot
SCR = "/tmp/claude-63526/-mnt-data1-kk829-CMSSW-16-1-1-src-RecoTracker-LSTCore-standalone/dbd9af1c-87ff-4b4e-9196-c80ebb15a786/scratchpad/s47/late_build/"
F = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_s45_rebased_100evt.root"
TN = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/trackingNtuple-100.root"
BR = ['sim_q', 'sim_pt', 'sim_eta', 'sim_phi', 'sim_vx', 'sim_vy', 'sim_vz', 'sim_genjet_idx', 'sim_genjet_deltaR',
      'genjet_pt', 'genjet_eta', 'sim_tcIdx', 'sim_T4_matched', 'sim_pT3_matched']
for o in ['pls', 't3', 't5', 'pt3', 'pt5', 't4', 'ls', 'md', 'tc']:
    BR += [f'sim_{o}IdxAll', f'sim_{o}IdxAllFrac']
BR += ['pLS_pt', 'pLS_ptErr', 'pLS_eta', 'pLS_phi', 'pLS_isQuad', 'pLS_isDup', 'pLS_charge', 'pLS_px', 'pLS_py', 'pLS_pz',
       'pLS_circleRadius', 'pLS_circleCenterX', 'pLS_circleCenterY', 'pLS_simIdx',
       'pLS_hit0_x', 'pLS_hit0_y', 'pLS_hit0_z', 'pLS_hit1_x', 'pLS_hit1_y', 'pLS_hit1_z',
       'pLS_hit2_x', 'pLS_hit2_y', 'pLS_hit2_z', 'pLS_hit3_x', 'pLS_hit3_y', 'pLS_hit3_z',
       't5_isDupBits', 't5_triedInPT5', 't5_partOfPT5', 't5_partOfTC', 't5_t3Idx0', 't5_t3Idx1', 't5_dnnScore',
       't5_pt', 't5_eta', 't5_phi', 't5_simIdx', 't5_innerRadius', 't5_outerRadius', 't5_bridgeRadius', 't5_nLayers',
       't3_radius', 't3_partOfPT5', 't3_partOfT5', 't3_partOfPT3', 't3_eta', 't3_phi', 't3_pt', 't3_simIdx', 't3_lsIdx0', 't3_lsIdx1',
       't3_hit_0_layer', 't3_hit_2_layer', 't3_hit_4_layer', 't3_hit_0_moduleType', 't3_hit_2_moduleType', 't3_hit_4_moduleType',
       't3_hit_0_detId', 't3_hit_1_detId', 't3_hit_2_detId', 't3_hit_4_detId',
       't3_hit_0_x', 't3_hit_0_y', 't3_hit_0_z', 't3_hit_2_x', 't3_hit_2_y', 't3_hit_2_z', 't3_hit_4_x', 't3_hit_4_y', 't3_hit_4_z',
       't3_fakeScore1' if False else 't3_pMatched',
       'ls_mdIdx0', 'ls_mdIdx1', 'md_layer', 'md_detId',
       'pT5_plsIdx', 'pT5_t5Idx', 'pT5_isDupReco', 'pT5_score', 'pT5_pt', 'pT5_simIdx',
       'pT3_plsIdx', 'pT3_t3Idx', 'pT3_score', 'pT3_pt', 'pT3_isDuplicate', 'pT3_simIdx',
       't4_isDup', 't4_partOfTC', 't4_simIdx', 't4_t3_idx0', 't4_t3_idx1', 't4_fakeScore', 't4_displacedScore', 't4_pt', 't4_eta',
       'tc_type', 'tc_pt5Idx', 'tc_pt3Idx', 'tc_t5Idx', 'tc_plsIdx', 'tc_t4Idx', 'tc_isFake', 'tc_simIdx', 'tc_pt', 'tc_eta', 'tc_phi']


def genuine(idx, frac):
    idx = np.asarray(idx); return [int(i) for i in idx[np.asarray(frac) >= 0.75]]


def build():
    t = uproot.open(F)["tree"]
    a = t.arrays(BR, library="np")
    ev = [{k: a[k][i] for k in BR} for i in range(t.num_entries)]
    rows = []
    for ie, e in enumerate(ev):
        pt = e['sim_pt'].astype(float); gj = e['sim_genjet_idx'].astype(np.int64)
        gpt, geta = e['genjet_pt'], e['genjet_eta']; gc = np.clip(gj, 0, max(len(gpt) - 1, 0))
        gp = gpt[gc] if len(gpt) else np.zeros_like(pt); ge = geta[gc] if len(geta) else np.zeros_like(pt)
        dr = e['sim_genjet_deltaR']
        sel = ((e['sim_q'] != 0) & (pt > 0.8) & (np.abs(e['sim_eta']) < 4.5) & (np.abs(e['sim_vz']) < 30) &
               (np.hypot(e['sim_vx'], e['sim_vy']) < 2.5) & (gj >= 0) & (gp > 1000) & (np.abs(ge) < 2.5) & (dr >= 0) & (dr < 0.02))
        for s in np.nonzero(sel)[0]:
            r = dict(ev=ie, sim=int(s), pt=float(pt[s]), eta=float(e['sim_eta'][s]), phi=float(e['sim_phi'][s]), dR=float(dr[s]),
                     tc=int(e['sim_tcIdx'][s]))
            for o in ['pls', 't3', 't5', 'pt3', 'pt5', 't4', 'ls', 'md']:
                r[o] = genuine(e[f'sim_{o}IdxAll'][s], e[f'sim_{o}IdxAllFrac'][s])
            rows.append(r)
    # pLS -> seed
    T = uproot.open(TN)['trackingNtuple/tree']
    SB = ['sim_pt', 'pix_x', 'pix_y', 'pix_z', 'see_px', 'see_py', 'see_pz', 'see_dz', 'see_algo', 'see_hitIdx', 'see_hitType',
          'see_stateTrajGlbPx', 'see_stateTrajGlbPy', 'see_stateTrajGlbX', 'see_stateTrajGlbY', 'see_stateTrajGlbZ']
    TA = T.arrays(SB, library='np')
    fp = {np.asarray(TA['sim_pt'][i], dtype=np.float32).tobytes(): i for i in range(T.num_entries)}
    TI = [fp[np.asarray(e['sim_pt'], dtype=np.float32).tobytes()] for e in ev]
    plsinfo = []; nf = nm = 0
    for ie, e in enumerate(ev):
        ti = TI[ie]; px, py, pz = TA['pix_x'][ti], TA['pix_y'][ti], TA['pix_z'][ti]
        key = {}
        for s in range(len(TA['see_px'][ti])):
            h = TA['see_hitIdx'][ti][s]; ht = TA['see_hitType'][ti][s]
            if len(h) < 3 or any(x != 0 for x in ht[:3]):
                continue
            k = tuple(np.round([px[h[0]], py[h[0]], pz[h[0]], px[h[1]], py[h[1]], pz[h[1]], px[h[2]], py[h[2]], pz[h[2]]], 3))
            key.setdefault(k, []).append(s)
        info = []
        for p in range(len(e['pLS_pt'])):
            k = tuple(np.round([e['pLS_hit0_x'][p], e['pLS_hit0_y'][p], e['pLS_hit0_z'][p], e['pLS_hit1_x'][p], e['pLS_hit1_y'][p],
                                e['pLS_hit1_z'][p], e['pLS_hit2_x'][p], e['pLS_hit2_y'][p], e['pLS_hit2_z'][p]], 3))
            best = None
            for s in key.get(k, []):
                ptIn = math.hypot(TA['see_stateTrajGlbPx'][ti][s], TA['see_stateTrajGlbPy'][ti][s])
                if abs(ptIn - e['pLS_pt'][p]) < 1e-3 * max(1, ptIn):
                    best = s; break
            if best is None:
                nm += 1; info.append(None); continue
            nf += 1; s = best
            P = np.array([TA['see_px'][ti][s], TA['see_py'][ti][s], TA['see_pz'][ti][s]])
            pt = math.hypot(P[0], P[1]); eta = math.asinh(P[2] / pt); phi = math.atan2(P[1], P[0]); dz = TA['see_dz'][ti][s]
            etabin = int((eta + 2.6) / ((2 * 2.6) / 25.)); phibin = int((phi + math.pi) / ((2 * math.pi) / 72.))
            dzbin = int((min(max(dz, -30), 30) + 30) / (2 * 30 / 25.))
            sb = int((25 * 72) * etabin + 25 * phibin + dzbin)
            pr = np.array([TA['see_stateTrajGlbPx'][ti][s], TA['see_stateTrajGlbPy'][ti][s]])
            r3 = np.array([TA['see_stateTrajGlbX'][ti][s], TA['see_stateTrajGlbY'][ti][s]])
            dphi = math.remainder(math.atan2(r3[1], r3[0]) - math.atan2(pr[1], pr[0]), 2 * math.pi)
            ptype = 0 if e['pLS_pt'][p] >= 2.0 else (1 if dphi >= 0 else 2)
            info.append(dict(seed=int(s), sb=sb, ptype=ptype,
                             x1=TA['see_stateTrajGlbX'][ti][s], y1=TA['see_stateTrajGlbY'][ti][s], z1=TA['see_stateTrajGlbZ'][ti][s]))
        plsinfo.append(info)
    print('pLS->seed matched', nf, 'missed', nm)
    return ev, rows, plsinfo


if os.path.exists(SCR + 'load47.pkl'):
    ev, rows, plsinfo = pickle.load(open(SCR + 'load47.pkl', 'rb'))
else:
    ev, rows, plsinfo = build()
    pickle.dump((ev, rows, plsinfo), open(SCR + 'load47.pkl', 'wb'), protocol=4)

if __name__ == '__main__':
    print('events', len(ev), 'core rows', len(rows), 'fails', sum(r['tc'] < 0 for r in rows))

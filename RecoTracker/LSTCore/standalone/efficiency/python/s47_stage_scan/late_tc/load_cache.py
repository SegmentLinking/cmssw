#!/usr/bin/env python3
"""Cache the branches needed by the S47 late_tc scans as per-event numpy dicts (pickle, one file per 10 evts).
READ-ONLY on the ntuple.  Usage: python3 load_cache.py [ntuple] [cachedir]"""
import sys, os, pickle
import numpy as np
import uproot

FN = sys.argv[1] if len(sys.argv) > 1 else \
    '/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_s45_rebased_100evt.root'
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    '/tmp/claude-63526/-mnt-data1-kk829-CMSSW-16-1-1-src-RecoTracker-LSTCore-standalone/dbd9af1c-87ff-4b4e-9196-c80ebb15a786/scratchpad/s47/late_tc/cache'

BR = ['sim_q', 'sim_pt', 'sim_eta', 'sim_phi', 'sim_vx', 'sim_vy', 'sim_vz', 'sim_genjet_idx', 'sim_genjet_deltaR',
      'genjet_pt', 'genjet_eta', 'sim_tcIdx', 'sim_tcIdxAll', 'sim_tcIdxAllFrac', 'sim_tcIdxBest', 'sim_tcIdxBestFrac']
for o in ['pls', 't3', 't5', 'pt3', 'pt5', 't4', 'md', 'ls']:
    BR += [f'sim_{o}IdxAll', f'sim_{o}IdxAllFrac']
BR += ['pT5_plsIdx', 'pT5_t5Idx', 'pT5_score', 'pT5_isDupReco', 'pT5_isDupTiebreaker', 'pT5_isFake',
       'pT5_simIdxAll', 'pT5_simIdxAllFrac', 'pT5_pt', 'pT5_eta', 'pT5_phi']
BR += ['t5_t3Idx0', 't5_t3Idx1', 't5_t3_idx0', 't5_t3_idx1', 't5_eta', 't5_phi', 't5_pt', 't5_dnnScore', 't5_isDupBits',
       't5_partOfPT5', 't5_partOfTC', 't5_tc_idx', 't5_hitIndices', 't5_nLayers', 't5_isFake', 't5_simIdxAll',
       't5_simIdxAllFrac', 't5_embed', 't5_moduleIdx', 't5_innerRadius', 't5_outerRadius', 't5_bridgeRadius',
       't5_t3_fakeScore1', 't5_t3_promptScore1', 't5_t3_displacedScore1', 't5_t3_fakeScore2', 't5_t3_promptScore2',
       't5_t3_displacedScore2', 't5_triedInPT5', 't5_logicalLayers']
BR += ['t3_lsIdx0', 't3_lsIdx1', 't3_partOfPT5', 't3_partOfPT3', 't3_partOfT5', 't3_isFake', 't3_simIdxAll',
       't3_simIdxAllFrac']
BR += ['ls_mdIdx0', 'ls_mdIdx1', 'md_anchor_x', 'md_anchor_y', 'md_anchor_z', 'md_other_x', 'md_other_y', 'md_other_z']
BR += ['pLS_lsIdx', 'pLS_isQuad', 'pLS_isDup', 'pLS_eta', 'pLS_phi', 'pLS_pt', 'pLS_isFake', 'pLS_simIdxAll',
       'pLS_simIdxAllFrac', 'pLS_circleCenterX', 'pLS_circleCenterY', 'pLS_circleRadius', 'pLS_charge',
       'pLS_hit0_x', 'pLS_hit0_y', 'pLS_hit0_z', 'pLS_hit1_x', 'pLS_hit1_y', 'pLS_hit1_z',
       'pLS_hit2_x', 'pLS_hit2_y', 'pLS_hit2_z', 'pLS_hit3_x', 'pLS_hit3_y', 'pLS_hit3_z']
BR += ['pT3_plsIdx', 'pT3_t3Idx', 'pT3_score', 'pT3_isFake', 'pT3_otHitIndices', 'pT3_simIdxAll', 'pT3_simIdxAllFrac',
       'pT3_eta', 'pT3_phi', 'pT3_pix_eta', 'pT3_pix_phi']
BR += ['t4_t3_idx0', 't4_t3_idx1', 't4_isDup', 't4_partOfTC', 't4_tc_idx', 't4_displacedScore', 't4_fakeScore',
       't4_eta', 't4_phi', 't4_isFake', 't4_simIdxAll', 't4_simIdxAllFrac', 't4_t3_0_moduleIdx']
BR += ['tc_type', 'tc_pt5Idx', 'tc_pt3Idx', 'tc_t5Idx', 'tc_plsIdx', 'tc_t4Idx', 'tc_simIdxAll', 'tc_simIdxAllFrac',
       'tc_isFake', 'tc_nhits', 'tc_nhitOT', 'tc_pt', 'tc_eta', 'tc_phi', 'tc_isDuplicate']


def conv(x):
    # jagged-of-jagged -> list of np arrays; flat -> np array
    try:
        return np.asarray(x) if not hasattr(x, '__len__') or len(x) == 0 or np.isscalar(x[0]) or \
            isinstance(x[0], (np.generic,)) else [np.asarray(v) for v in x]
    except Exception:
        return x


def main():
    os.makedirs(OUT, exist_ok=True)
    t = uproot.open(FN)['tree']
    n = t.num_entries
    for s in range(0, n, 10):
        A = t.arrays(BR, entry_start=s, entry_stop=min(n, s + 10), library='np')
        evs = []
        for ie in range(len(A['sim_pt'])):
            ev = {}
            for k in BR:
                v = A[k][ie]
                if hasattr(v, '__len__') and len(v) and hasattr(v[0], '__len__') and not isinstance(v[0], str):
                    ev[k] = [np.asarray(x) for x in v]
                else:
                    ev[k] = np.asarray(v)
            evs.append(ev)
        with open(f'{OUT}/ev_{s:04d}.pkl', 'wb') as f:
            pickle.dump(evs, f, protocol=4)
        print('cached', s, file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()

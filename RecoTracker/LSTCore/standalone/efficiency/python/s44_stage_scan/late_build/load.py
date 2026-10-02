"""Load real-pLS AB-ON 100-evt ntuple into a pickle of per-event dicts (numpy / lists)."""
import uproot, numpy as np, pickle, sys
F = '/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_realpls_mastercuts_100evt_v2.root'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'ev.pkl'
t = uproot.open(F)['tree']
BR = ['sim_q','sim_pt','sim_eta','sim_phi','sim_vx','sim_vy','sim_vz','sim_pdgId','sim_pca_dxy','sim_pca_dz',
      'sim_genjet_idx','sim_genjet_deltaR','genjet_pt','genjet_eta','sim_tcIdx',
      'sim_t3IdxAll','sim_t3IdxAllFrac','sim_t5IdxAll','sim_t5IdxAllFrac','sim_plsIdxAll','sim_plsIdxAllFrac',
      'sim_pt3IdxAll','sim_pt3IdxAllFrac','sim_pt5IdxAll','sim_pt5IdxAllFrac','sim_lsIdxAll','sim_lsIdxAllFrac',
      'sim_mdIdxAll','sim_mdIdxAllFrac','sim_t4IdxAll','sim_t4IdxAllFrac','sim_simHitLayer','sim_simHitDetId',
      'pLS_pt','pLS_ptErr','pLS_eta','pLS_etaErr','pLS_phi','pLS_isQuad','pLS_nhit','pLS_charge','pLS_px','pLS_py','pLS_pz',
      'pLS_circleRadius','pLS_circleCenterX','pLS_circleCenterY','pLS_deltaPhi',
      'pLS_hit0_x','pLS_hit0_y','pLS_hit0_z','pLS_hit1_x','pLS_hit1_y','pLS_hit1_z',
      'pLS_hit2_x','pLS_hit2_y','pLS_hit2_z','pLS_hit3_x','pLS_hit3_y','pLS_hit3_z',
      't5_isDupBits','t5_triedInPT5','t5_partOfPT5','t5_t3Idx0','t5_t3Idx1','t5_dnnScore','t5_innerRadius','t5_outerRadius',
      't5_bridgeRadius','t5_pt','t5_eta','t5_phi','t5_tightCutFlag','t5_partOfTC',
      't5_t3_0_layer','t5_t3_0_moduleType','t5_t3_2_layer','t5_t3_4_layer','t5_t3_0_r','t5_t3_0_z',
      't3_radius','t3_partOfPT5','t3_partOfT5','t3_partOfPT3','t3_betaIn','t3_eta','t3_phi','t3_pt',
      't3_hit_0_layer','t3_hit_2_layer','t3_hit_4_layer','t3_hit_0_moduleType','t3_hit_2_moduleType','t3_hit_4_moduleType',
      't3_hit_0_x','t3_hit_0_y','t3_hit_0_z','t3_hit_2_x','t3_hit_2_y','t3_hit_2_z','t3_hit_4_x','t3_hit_4_y','t3_hit_4_z',
      'pT5_plsIdx','pT5_t5Idx','pT5_isDupReco','pT5_score','pT5_pt',
      'pT3_plsIdx','pT3_t3Idx','pT3_score','pT3_pt','pT3_pixelRadius','pT3_tripletRadius',
      'tc_type','tc_pt5Idx','tc_pt3Idx','tc_t5Idx','tc_plsIdx',
      't3_occupancies','t5_occupancies','pT3_occupancies','pT5_occupancies','module_layers','module_subdets']
a = t.arrays(BR, library='np')
ev = [{k: a[k][i] for k in BR} for i in range(t.num_entries)]
pickle.dump(ev, open(OUT,'wb'), protocol=4)
print('ok', len(ev))

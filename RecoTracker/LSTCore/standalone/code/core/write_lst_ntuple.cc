#include "LSTEvent.h"
#include "Circle.h"

#include "write_lst_ntuple.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE::lst;

//________________________________________________________________________________________________________________________________
void createOutputBranches() {
    createSimTrackContainerBranches();
    createTrackCandidateBranches();
    
    if (ana.gnn_ntuple)
      createGnnNtupleBranches();
  
  #ifdef CUT_VALUE_DEBUG
    // Event-wide branches
    // ana.tx->createBranch<float>("evt_dummy");
  
    // Sim Track branches
    // NOTE: Must sync with main tc branch in length!!
    ana.tx->createBranch<std::vector<float>>("sim_dummy");
  
    // Track Candidate branches
    // NOTE: Must sync with main tc branch in length!!
    ana.tx->createBranch<std::vector<float>>("tc_dummy");
  
    createMiniDoubletBranches();
    createLineSegmentBranches();
    createTripletBranches();
    createQuintupletBranches();
    createPixelLineSegmentBranches();
    createPixelTripletBranches();
    createPixelQuintupletBranches();
  
    createOccupancyBranches();
  
    // DNN branches
    createT5DNNBranches();
    createT3DNNBranches();
  
  #endif
}

//________________________________________________________________________________________________________________________________
void fillOutputBranches(LSTEvent* event) {
    float matchfrac = 0.75;
    
    unsigned int n_accepted_simtrk = setSimTrackContainerBranches(event);
    
    if (ana.gnn_ntuple)
      setGnnNtupleBranches(event);
      
#ifdef CUT_VALUE_DEBUG
  setPixelQuintupletOutputBranches(event);
  setQuintupletOutputBranches(event);
  setPixelTripletOutputBranches(event);
  setpLSOutputBranches(event);
  
  setOccupancyBranches(event);

  setT3DNNBranches(event);
  setT5DNNBranches(event);
  setpT3DNNBranches(event);
#else
  setTrackCandidateBranches(event, n_accepted_simtrk, matchfrac);
#endif

  // Now actually fill the ttree
  ana.tx->fill();

  // Then clear the branches to default values (e.g. -999, or clear the vectors to empty vectors)
  ana.tx->clear();
}

//________________________________________________________________________________________________________________________________
void createT5DNNBranches() {
  // Common branches
  ana.tx->createBranch<std::vector<int>>("t5_t3_idx0");
  ana.tx->createBranch<std::vector<int>>("t5_t3_idx1");
  ana.tx->createBranch<std::vector<int>>("t5_tc_idx");
  ana.tx->createBranch<std::vector<int>>("t5_partOfTC");
  ana.tx->createBranch<std::vector<float>>("t5_t3_pt");
  ana.tx->createBranch<std::vector<float>>("t5_t3_eta");
  ana.tx->createBranch<std::vector<float>>("t5_t3_phi");
  ana.tx->createBranch<std::vector<float>>("t5_t3_fakeScore1");
  ana.tx->createBranch<std::vector<float>>("t5_t3_promptScore1");
  ana.tx->createBranch<std::vector<float>>("t5_t3_displacedScore1");
  ana.tx->createBranch<std::vector<float>>("t5_t3_fakeScore2");
  ana.tx->createBranch<std::vector<float>>("t5_t3_promptScore2");
  ana.tx->createBranch<std::vector<float>>("t5_t3_displacedScore2");

  // Hit-specific branches
  std::vector<std::string> hitIndices = {"0", "1", "2", "3", "4", "5"};
  std::vector<std::string> hitProperties = {"r", "x", "y", "z", "eta", "phi", "detId", "layer", "moduleType"};

  for (const auto& idx : hitIndices) {
    for (const auto& prop : hitProperties) {
      std::string branchName = "t5_t3_" + idx + "_" + prop;
      if (prop == "detId" || prop == "layer" || prop == "moduleType") {
        ana.tx->createBranch<std::vector<int>>(branchName);
      } else {
        ana.tx->createBranch<std::vector<float>>(branchName);
      }
    }
  }
}

//________________________________________________________________________________________________________________________________
void createT3DNNBranches() {
  // Common branches for T3 properties based on TripletsSoA fields
  ana.tx->createBranch<std::vector<float>>("t3_betaIn");
  ana.tx->createBranch<std::vector<float>>("t3_centerX");
  ana.tx->createBranch<std::vector<float>>("t3_centerY");
  ana.tx->createBranch<std::vector<float>>("t3_radius");
  ana.tx->createBranch<std::vector<bool>>("t3_partOfPT5");
  ana.tx->createBranch<std::vector<bool>>("t3_partOfT5");
  ana.tx->createBranch<std::vector<bool>>("t3_partOfPT3");
  ana.tx->createBranch<std::vector<float>>("t3_pMatched");
  ana.tx->createBranch<std::vector<float>>("t3_sim_vxy");
  ana.tx->createBranch<std::vector<float>>("t3_sim_vz");

  // Hit-specific branches (T3 has 4 hits from two segments)
  std::vector<std::string> hitIndices = {"0", "1", "2", "3", "4", "5"};
  std::vector<std::string> hitProperties = {"r", "x", "y", "z", "eta", "phi", "detId", "layer", "moduleType"};

  for (const auto& idx : hitIndices) {
    for (const auto& prop : hitProperties) {
      std::string branchName = "t3_hit_" + idx + "_" + prop;
      if (prop == "detId" || prop == "layer" || prop == "moduleType") {
        ana.tx->createBranch<std::vector<int>>(branchName);
      } else {
        ana.tx->createBranch<std::vector<float>>(branchName);
      }
    }
  }

  // Additional metadata branches
  ana.tx->createBranch<std::vector<int>>("t3_layer_binary");
  ana.tx->createBranch<std::vector<std::vector<int>>>("t3_matched_simIdx");
}

//________________________________________________________________________________________________________________________________
void createGnnNtupleBranches() {
  // Mini Doublets
  ana.tx->createBranch<std::vector<float>>("MD_pt");
  ana.tx->createBranch<std::vector<float>>("MD_eta");
  ana.tx->createBranch<std::vector<float>>("MD_phi");
  ana.tx->createBranch<std::vector<float>>("MD_dphichange");
  ana.tx->createBranch<std::vector<int>>("MD_isFake");
  ana.tx->createBranch<std::vector<int>>("MD_tpType");
  ana.tx->createBranch<std::vector<int>>("MD_detId");
  ana.tx->createBranch<std::vector<int>>("MD_layer");
  ana.tx->createBranch<std::vector<float>>("MD_0_r");
  ana.tx->createBranch<std::vector<float>>("MD_0_x");
  ana.tx->createBranch<std::vector<float>>("MD_0_y");
  ana.tx->createBranch<std::vector<float>>("MD_0_z");
  ana.tx->createBranch<std::vector<float>>("MD_1_r");
  ana.tx->createBranch<std::vector<float>>("MD_1_x");
  ana.tx->createBranch<std::vector<float>>("MD_1_y");
  ana.tx->createBranch<std::vector<float>>("MD_1_z");

  // Line Segments
  ana.tx->createBranch<std::vector<float>>("LS_pt");
  ana.tx->createBranch<std::vector<float>>("LS_eta");
  ana.tx->createBranch<std::vector<float>>("LS_phi");
  ana.tx->createBranch<std::vector<int>>("LS_isFake");
  ana.tx->createBranch<std::vector<int>>("LS_MD_idx0");
  ana.tx->createBranch<std::vector<int>>("LS_MD_idx1");
  ana.tx->createBranch<std::vector<float>>("LS_sim_pt");
  ana.tx->createBranch<std::vector<float>>("LS_sim_eta");
  ana.tx->createBranch<std::vector<float>>("LS_sim_phi");
  ana.tx->createBranch<std::vector<float>>("LS_sim_pca_dxy");
  ana.tx->createBranch<std::vector<float>>("LS_sim_pca_dz");
  ana.tx->createBranch<std::vector<int>>("LS_sim_q");
  ana.tx->createBranch<std::vector<int>>("LS_sim_pdgId");
  ana.tx->createBranch<std::vector<int>>("LS_sim_event");
  ana.tx->createBranch<std::vector<int>>("LS_sim_bx");
  ana.tx->createBranch<std::vector<float>>("LS_sim_vx");
  ana.tx->createBranch<std::vector<float>>("LS_sim_vy");
  ana.tx->createBranch<std::vector<float>>("LS_sim_vz");
  ana.tx->createBranch<std::vector<int>>("LS_isInTrueTC");

  // TC's LS
  ana.tx->createBranch<std::vector<std::vector<int>>>("tc_lsIdx");
}

//________________________________________________________________________________________________________________________________
void createSimTrackContainerBranches() {
  // Simulated Track Container
  //
  //  The container will hold per entry a simulated track in the event. Only the current bunch crossing, and
  //  primary vertex (hard-scattered) tracks will be saved to reduce the size of the output.
  //
  ana.tx->createBranch<std::vector<float>>("sim_pt");       // pt
  ana.tx->createBranch<std::vector<float>>("sim_eta");      // eta
  ana.tx->createBranch<std::vector<float>>("sim_phi");      // phi
  ana.tx->createBranch<std::vector<float>>("sim_pca_dxy");  // dxy of point of closest approach
  ana.tx->createBranch<std::vector<float>>("sim_pca_dz");   // dz of point of clossest approach
  ana.tx->createBranch<std::vector<int>>("sim_q");          // charge +1, -1, 0
  ana.tx->createBranch<std::vector<int>>("sim_pdgId");      // pdgId
  // production vertex x position (values are derived from simvtx_* and sim_parentVtxIdx branches in the tracking ntuple)
  ana.tx->createBranch<std::vector<float>>("sim_vx");
  // production vertex y position (values are derived from simvtx_* and sim_parentVtxIdx branches in the tracking ntuple)
  ana.tx->createBranch<std::vector<float>>("sim_vy");
  // production vertex z position (values are derived from simvtx_* and sim_parentVtxIdx branches in the tracking ntuple)
  ana.tx->createBranch<std::vector<float>>("sim_vz");
  // production vertex r (sqrt(x**2 + y**2)) position (values are derived from simvtx_* and sim_parentVtxIdx branches in the tracking ntuple)
  ana.tx->createBranch<std::vector<float>>("sim_vtxperp");
  // idx of sim_* in the tracking ntuple (N.B. this may be redundant)
  ana.tx->createBranch<std::vector<float>>("sim_trkNtupIdx");
  // idx to the best match (highest nhit match) tc_* container
  ana.tx->createBranch<std::vector<int>>("sim_tcIdxBest");
  // match fraction to the best match (highest nhit match) tc_* container
  ana.tx->createBranch<std::vector<float>>("sim_tcIdxBestFrac");
  // idx to the best match (highest nhit match and > 75%) tc_* container
  ana.tx->createBranch<std::vector<int>>("sim_tcIdx");
  // list of idx to any matches (> 0%) to tc_* container
  ana.tx->createBranch<std::vector<std::vector<int>>>("sim_tcIdxAll");
  // list of match fraction for each match (> 0%) to tc_* container
  ana.tx->createBranch<std::vector<std::vector<float>>>("sim_tcIdxAllFrac");
  // list of idx to matches (> 0%) to md_* container
  ana.tx->createBranch<std::vector<std::vector<int>>>("sim_mdIdxAll");
  // list of match fraction for each match (> 0%) to md_* container
  ana.tx->createBranch<std::vector<std::vector<float>>>("sim_mdIdxAllFrac");
  // list of idx to matches (> 0%) to ls_* container
  ana.tx->createBranch<std::vector<std::vector<int>>>("sim_lsIdxAll");
  // list of match fraction for each match (> 0%) to ls_* container
  ana.tx->createBranch<std::vector<std::vector<float>>>("sim_lsIdxAllFrac");
  // list of idx to matches (> 0%) to t3_* container
  ana.tx->createBranch<std::vector<std::vector<int>>>("sim_t3IdxAll");
  // list of match fraction for each match (> 0%) to t3_* container
  ana.tx->createBranch<std::vector<std::vector<float>>>("sim_t3IdxAllFrac");
  // list of idx to matches (> 0%) to t5_* container
  ana.tx->createBranch<std::vector<std::vector<int>>>("sim_t5IdxAll");
  // list of match fraction for each match (> 0%) to t5_* container
  ana.tx->createBranch<std::vector<std::vector<float>>>("sim_t5IdxAllFrac");
  // list of idx to matches (> 0%) to pls_* container
  ana.tx->createBranch<std::vector<std::vector<int>>>("sim_plsIdxAll");
  // list of match fraction for each match (> 0%) to pls_* container
  ana.tx->createBranch<std::vector<std::vector<float>>>("sim_plsIdxAllFrac");
  // list of idx to matches (> 0%) to pt3_* container
  ana.tx->createBranch<std::vector<std::vector<int>>>("sim_pt3IdxAll");
  // list of match fraction for each match (> 0%) to pt3_* container
  ana.tx->createBranch<std::vector<std::vector<float>>>("sim_pt3IdxAllFrac");
  // list of idx to matches (> 0%) to pt5_* container
  ana.tx->createBranch<std::vector<std::vector<int>>>("sim_pt5IdxAll");
  // list of match fraction for each match (> 0%) to pt5_* container
  ana.tx->createBranch<std::vector<std::vector<float>>>("sim_pt5IdxAllFrac");
  ana.tx->createBranch<std::vector<std::vector<float>>>("sim_simHitX");    // list of simhit's X positions
  ana.tx->createBranch<std::vector<std::vector<float>>>("sim_simHitY");    // list of simhit's Y positions
  ana.tx->createBranch<std::vector<std::vector<float>>>("sim_simHitZ");    // list of simhit's Z positions
  ana.tx->createBranch<std::vector<std::vector<int>>>("sim_simHitDetId");  // list of simhit's detId
  // list of simhit's layers (N.B. layer is numbered 1 2 3 4 5 6 for barrel, 7 8 9 10 11 for endcaps)
  ana.tx->createBranch<std::vector<std::vector<int>>>("sim_simHitLayer");
  // list of simhit's distance in xy-plane to the expected point based on simhit's z position and helix formed from pt,eta,phi,vx,vy,vz,q of the simulated track
  ana.tx->createBranch<std::vector<std::vector<float>>>("sim_simHitDistxyHelix");
  // length of 11 float numbers with min(simHitDistxyHelix) value for each layer. Useful for finding e.g. "sim tracks that traversed barrel detector entirelyand left a reasonable hit in layer 1 2 3 4 5 6 layers."
  ana.tx->createBranch<std::vector<std::vector<float>>>("sim_simHitLayerMinDistxyHelix");
  // length of 11 float numbers with min(simHitDistxyHelix) value for each layer. Useful for finding e.g. "sim tracks that traversed barrel detector entirelyand left a reasonable hit in layer 1 2 3 4 5 6 layers."
  ana.tx->createBranch<std::vector<std::vector<float>>>("sim_simHitLayerMinDistxyPrevHit");
  ana.tx->createBranch<std::vector<std::vector<float>>>("sim_recoHitX");    // list of recohit's X positions
  ana.tx->createBranch<std::vector<std::vector<float>>>("sim_recoHitY");    // list of recohit's Y positions
  ana.tx->createBranch<std::vector<std::vector<float>>>("sim_recoHitZ");    // list of recohit's Z positions
  ana.tx->createBranch<std::vector<std::vector<int>>>("sim_recoHitDetId");  // list of recohit's detId
}

//________________________________________________________________________________________________________________________________
void createTrackCandidateBranches() {
  // Track Candidates
  //
  //  The container will hold per entry a track candidate built by LST in the event.
  //
  ana.tx->createBranch<std::vector<float>>("tc_pt");    // pt
  ana.tx->createBranch<std::vector<float>>("tc_eta");   // eta
  ana.tx->createBranch<std::vector<float>>("tc_phi");   // phi
  ana.tx->createBranch<std::vector<int>>("tc_type");    // type = 7 (pT5), 5 (pT3), 4 (T5), 8 (pLS)
#ifdef CUT_VALUE_DEBUG
  ana.tx->createBranch<std::vector<int>>("tc_pt5Idx");  // index to the pt5_* if it is the said type, if not set to -999
  ana.tx->createBranch<std::vector<int>>("tc_pt3Idx");  // index to the pt3_* if it is the said type, if not set to -999
  ana.tx->createBranch<std::vector<int>>("tc_t5Idx");   // index to the t5_*  if it is the said type, if not set to -999
  ana.tx->createBranch<std::vector<int>>("tc_plsIdx");  // index to the pls_* if it is the said type, if not set to -999
#endif
  ana.tx->createBranch<std::vector<int>>("tc_isFake");  // 1 if tc is fake 0 other if not
  ana.tx->createBranch<std::vector<int>>("tc_isDuplicate");  // 1 if tc is duplicate 0 other if not
  ana.tx->createBranch<std::vector<int>>("tc_simIdx");  // idx of best matched (highest nhit and > 75%) simulated track
  // list of idx of all matched (> 0%) simulated track
  ana.tx->createBranch<std::vector<std::vector<int>>>("tc_simIdxAll");
  // list of idx of all matched (> 0%) simulated track
  ana.tx->createBranch<std::vector<std::vector<float>>>("tc_simIdxAllFrac");
}

//________________________________________________________________________________________________________________________________
void createMiniDoubletBranches() {
  // Mini-Doublets (i.e. Two reco hits paired in a single pT-module of Outer Tracker of CMS, a.k.a. MD)
  //
  //  The container will hold per entry a mini-doublet built by LST in the event.
  //
  ana.tx->createBranch<std::vector<float>>("md_pt");   // pt (computed based on delta phi change)
  ana.tx->createBranch<std::vector<float>>("md_eta");  // eta (computed based on anchor hit's eta)
  ana.tx->createBranch<std::vector<float>>("md_phi");  // phi (computed based on anchor hit's phi)
#ifdef CUT_VALUE_DEBUG
  ana.tx->createBranch<std::vector<float>>("md_dphi");
  ana.tx->createBranch<std::vector<float>>("md_dphichange");
  ana.tx->createBranch<std::vector<float>>("md_dz");
#endif
  ana.tx->createBranch<std::vector<float>>("md_anchor_x");  // anchor hit x
  ana.tx->createBranch<std::vector<float>>("md_anchor_y");  // anchor hit y
  ana.tx->createBranch<std::vector<float>>("md_anchor_z");  // anchor hit z
  ana.tx->createBranch<std::vector<float>>("md_other_x");   // other hit x
  ana.tx->createBranch<std::vector<float>>("md_other_y");   // other hit y
  ana.tx->createBranch<std::vector<float>>("md_other_z");   // other hit z
  // type of the module where the mini-doublet sit (type = 1 (PS), 0 (2S))
  ana.tx->createBranch<std::vector<int>>("md_type");
  // layer index of the module where the mini-doublet sit (layer = 1 2 3 4 5 6 (barrel) 7 8 9 10 11 (endcap))
  ana.tx->createBranch<std::vector<int>>("md_layer");
  // detId = detector unique ID that contains a lot of information that can be parsed later if needed
  ana.tx->createBranch<std::vector<int>>("md_detId");
  ana.tx->createBranch<std::vector<int>>("md_isFake");  // 1 if md is fake 0 other if not
  ana.tx->createBranch<std::vector<int>>("md_simIdx");  // idx of best matched (highest nhit and > 75%) simulated track
  // list of idx of all matched (> 0%) simulated track
  ana.tx->createBranch<std::vector<std::vector<int>>>("md_simIdxAll");
  // list of idx of all matched (> 0%) simulated track
  ana.tx->createBranch<std::vector<std::vector<float>>>("md_simIdxAllFrac");
}

//________________________________________________________________________________________________________________________________
void createLineSegmentBranches() {
  // Line Segments (i.e. Two mini-doublets, a.k.a. LS)
  //
  //  The container will hold per entry a line-segment built by LST in the event.
  //
  // pt (computed based on radius of the circle formed by three points: (origin), (anchor hit 1), (anchor hit 2))
  ana.tx->createBranch<std::vector<float>>("ls_pt");
  ana.tx->createBranch<std::vector<float>>("ls_eta");   // eta (computed based on last anchor hit's eta)
  ana.tx->createBranch<std::vector<float>>("ls_phi");   // phi (computed based on first anchor hit's phi)
  ana.tx->createBranch<std::vector<int>>("ls_mdIdx0");  // index to the first MD
  ana.tx->createBranch<std::vector<int>>("ls_mdIdx1");  // index to the second MD
  ana.tx->createBranch<std::vector<int>>("ls_isFake");  // 1 if md is fake 0 other if not
  ana.tx->createBranch<std::vector<int>>("ls_simIdx");  // idx of best matched (highest nhit and > 75%) simulated track
#ifdef CUT_VALUE_DEBUG
  ana.tx->createBranch<std::vector<float>>("ls_zLos");
  ana.tx->createBranch<std::vector<float>>("ls_zHis");
  ana.tx->createBranch<std::vector<float>>("ls_rtLos");
  ana.tx->createBranch<std::vector<float>>("ls_rtHis");
  ana.tx->createBranch<std::vector<float>>("ls_dPhis");
  ana.tx->createBranch<std::vector<float>>("ls_dPhiMins");
  ana.tx->createBranch<std::vector<float>>("ls_dPhiMaxs");
  ana.tx->createBranch<std::vector<float>>("ls_dPhiChanges");
  ana.tx->createBranch<std::vector<float>>("ls_dPhiChangeMins");
  ana.tx->createBranch<std::vector<float>>("ls_dPhiChangeMaxs");
  ana.tx->createBranch<std::vector<float>>("ls_dAlphaInners");
  ana.tx->createBranch<std::vector<float>>("ls_dAlphaOuters");
  ana.tx->createBranch<std::vector<float>>("ls_dAlphaInnerOuters");
#endif
  // list of idx of all matched (> 0%) simulated track
  ana.tx->createBranch<std::vector<std::vector<int>>>("ls_simIdxAll");
  // list of idx of all matched (> 0%) simulated track
  ana.tx->createBranch<std::vector<std::vector<float>>>("ls_simIdxAllFrac");
}

//________________________________________________________________________________________________________________________________
void createTripletBranches() {
  // Triplets (i.e. Three mini-doublets, a.k.a. T3)
  //
  //  The container will hold per entry a triplets built by LST in the event.
  //
  // pt (computed based on radius of the circle formed by three points: anchor hit 1, 2, 3
  ana.tx->createBranch<std::vector<float>>("t3_pt");
  ana.tx->createBranch<std::vector<float>>("t3_eta");        // eta (computed based on last anchor hit's eta)
  ana.tx->createBranch<std::vector<float>>("t3_phi");        // phi (computed based on first anchor hit's phi)
  ana.tx->createBranch<std::vector<int>>("t3_lsIdx0");       // index to the first LS
  ana.tx->createBranch<std::vector<int>>("t3_lsIdx1");       // index to the second LS
  ana.tx->createBranch<std::vector<int>>("t3_isFake");       // 1 if t3 is fake 0 other if not
  ana.tx->createBranch<std::vector<int>>("t3_isDuplicate");  // 1 if t3 is duplicate 0 other if not
  ana.tx->createBranch<std::vector<int>>("t3_simIdx");  // idx of best matched (highest nhit and > 75%) simulated track
  // list of idx of all matched (> 0%) simulated track
  ana.tx->createBranch<std::vector<std::vector<int>>>("t3_simIdxAll");
  // list of idx of all matched (> 0%) simulated track
  ana.tx->createBranch<std::vector<std::vector<float>>>("t3_simIdxAllFrac");
}

//________________________________________________________________________________________________________________________________
void createQuintupletBranches() {
  // Quintuplets (i.e. Five mini-doublets, a.k.a. T5)
  //
  //  The container will hold per entry a quintuplet built by LST in the event.
  //
  // pt (computed based on average of the 4 circles formed by, (1, 2, 3), (2, 3, 4), (3, 4, 5), (1, 3, 5)
  ana.tx->createBranch<std::vector<float>>("t5_pt");
  ana.tx->createBranch<std::vector<float>>("t5_eta");        // eta (computed based on last anchor hit's eta)
  ana.tx->createBranch<std::vector<float>>("t5_phi");        // phi (computed based on first anchor hit's phi)
  ana.tx->createBranch<std::vector<int>>("t5_t3Idx0");       // index of first T3
  ana.tx->createBranch<std::vector<int>>("t5_t3Idx1");       // index of second T3
  ana.tx->createBranch<std::vector<int>>("t5_isFake");       // 1 if t5 is fake 0 other if not
  ana.tx->createBranch<std::vector<int>>("t5_isDuplicate");  // 1 if t5 is duplicate 0 other if not
  ana.tx->createBranch<std::vector<int>>("t5_simIdx");  // idx of best matched (highest nhit and > 75%) simulated track
  // list of idx of all matched (> 0%) simulated track
  ana.tx->createBranch<std::vector<std::vector<int>>>("t5_simIdxAll");
  // list of idx of all matched (> 0%) simulated track
  ana.tx->createBranch<std::vector<std::vector<float>>>("t5_simIdxAllFrac");
}

//________________________________________________________________________________________________________________________________
void createPixelLineSegmentBranches() {
  // Pixel Line Segments (a.k.a pLS)
  //
  //  The container will hold per entry a pixel line segment (built by an external algo, e.g. patatrack) accepted by LST in the event.
  //
  // pt (taken from pt of the 3-vector from see_stateTrajGlbPx/Py/Pz)
  ana.tx->createBranch<std::vector<float>>("pLS_pt");
  // eta (taken from eta of the 3-vector from see_stateTrajGlbPx/Py/Pz)
  ana.tx->createBranch<std::vector<float>>("pLS_eta");
  // phi (taken from phi of the 3-vector from see_stateTrajGlbPx/Py/Pz)
  ana.tx->createBranch<std::vector<float>>("pLS_phi");
  ana.tx->createBranch<std::vector<int>>("pLS_nhit");         // Number of actual hit: 3 if triplet, 4 if quadruplet
  ana.tx->createBranch<std::vector<float>>("pLS_hit0_x");     // pLS's reco hit0 x
  ana.tx->createBranch<std::vector<float>>("pLS_hit0_y");     // pLS's reco hit0 y
  ana.tx->createBranch<std::vector<float>>("pLS_hit0_z");     // pLS's reco hit0 z
  ana.tx->createBranch<std::vector<float>>("pLS_hit1_x");     // pLS's reco hit1 x
  ana.tx->createBranch<std::vector<float>>("pLS_hit1_y");     // pLS's reco hit1 y
  ana.tx->createBranch<std::vector<float>>("pLS_hit1_z");     // pLS's reco hit1 z
  ana.tx->createBranch<std::vector<float>>("pLS_hit2_x");     // pLS's reco hit2 x
  ana.tx->createBranch<std::vector<float>>("pLS_hit2_y");     // pLS's reco hit2 y
  ana.tx->createBranch<std::vector<float>>("pLS_hit2_z");     // pLS's reco hit2 z
  ana.tx->createBranch<std::vector<float>>("pLS_hit3_x");     // pLS's reco hit3 x (if triplet, this is set to -999)
  ana.tx->createBranch<std::vector<float>>("pLS_hit3_y");     // pLS's reco hit3 y (if triplet, this is set to -999)
  ana.tx->createBranch<std::vector<float>>("pLS_hit3_z");     // pLS's reco hit3 z (if triplet, this is set to -999)
  ana.tx->createBranch<std::vector<int>>("pLS_isFake");       // 1 if pLS is fake 0 other if not
  ana.tx->createBranch<std::vector<int>>("pLS_isDuplicate");  // 1 if pLS is duplicate 0 other if not
  ana.tx->createBranch<std::vector<int>>("pLS_simIdx");  // idx of best matched (highest nhit and > 75%) simulated track
  // list of idx of all matched (> 0%) simulated track
  ana.tx->createBranch<std::vector<std::vector<int>>>("pLS_simIdxAll");
  // list of idx of all matched (> 0%) simulated track
  ana.tx->createBranch<std::vector<std::vector<float>>>("pLS_simIdxAllFrac");
}

//________________________________________________________________________________________________________________________________
void createPixelTripletBranches() {
  // pLS + T3 (i.e. an object where a pLS is linked with a T3, a.k.a. pT3)
  //
  //  The container will hold per entry a pT3 built by LST in the event.
  //
  ana.tx->createBranch<std::vector<float>>("pT3_pt");         // pt (taken from the pLS)
  ana.tx->createBranch<std::vector<float>>("pT3_eta");        // eta (taken from the pLS)
  ana.tx->createBranch<std::vector<float>>("pT3_phi");        // phi (taken from the pLS)
  ana.tx->createBranch<std::vector<int>>("pT3_plsIdx");       // idx to pLS
  ana.tx->createBranch<std::vector<int>>("pT3_t3Idx");        // idx to T3
  ana.tx->createBranch<std::vector<int>>("pT3_isFake");       // 1 if pT3 is fake 0 other if not
  ana.tx->createBranch<std::vector<int>>("pT3_isDuplicate");  // 1 if pT3 is duplicate 0 other if not
  ana.tx->createBranch<std::vector<int>>("pT3_simIdx");       // idx of best matched (highest nhit and > 75%) simulated track
  // list of idx of all matched (> 0%) simulated track
  ana.tx->createBranch<std::vector<std::vector<int>>>("pT3_simIdxAll");
  // list of idx of all matched (> 0%) simulated track
  ana.tx->createBranch<std::vector<std::vector<float>>>("pT3_simIdxAllFrac");
}

//________________________________________________________________________________________________________________________________
void createPixelQuintupletBranches() {
  // pLS + T5 (i.e. an object where a pLS is linked with a T5, a.k.a. pT5)
  //
  //  The container will hold per entry a pT5 built by LST in the event.
  //
  ana.tx->createBranch<std::vector<float>>("pT5_pt");         // pt (taken from the pLS)
  ana.tx->createBranch<std::vector<float>>("pT5_eta");        // eta (taken from the pLS)
  ana.tx->createBranch<std::vector<float>>("pT5_phi");        // phi (taken from the pLS)
  ana.tx->createBranch<std::vector<int>>("pT5_plsIdx");       // idx to pLS
  ana.tx->createBranch<std::vector<int>>("pT5_t5Idx");        // idx to T5
  ana.tx->createBranch<std::vector<int>>("pT5_isFake");       // 1 if pT5 is fake 0 other if not
  ana.tx->createBranch<std::vector<int>>("pT5_isDuplicate");  // 1 if pT5 is duplicate 0 other if not
  ana.tx->createBranch<std::vector<int>>("pT5_simIdx");  // idx of best matched (highest nhit and > 75%) simulated track
  // list of idx of all matched (> 0%) simulated track
  ana.tx->createBranch<std::vector<std::vector<int>>>("pT5_simIdxAll");
  // list of idx of all matched (> 0%) simulated track
  ana.tx->createBranch<std::vector<std::vector<float>>>("pT5_simIdxAllFrac");
}

//________________________________________________________________________________________________________________________________
void createOccupancyBranches() {
  ana.tx->createBranch<std::vector<int>>("module_layers");
  ana.tx->createBranch<std::vector<int>>("module_subdets");
  ana.tx->createBranch<std::vector<int>>("module_rings");
  ana.tx->createBranch<std::vector<int>>("module_rods");
  ana.tx->createBranch<std::vector<int>>("module_modules");
  ana.tx->createBranch<std::vector<bool>>("module_isTilted");
  ana.tx->createBranch<std::vector<float>>("module_eta");
  ana.tx->createBranch<std::vector<float>>("module_r");
  ana.tx->createBranch<std::vector<int>>("md_occupancies");
  ana.tx->createBranch<std::vector<int>>("sg_occupancies");
  ana.tx->createBranch<std::vector<int>>("t3_occupancies");
  ana.tx->createBranch<int>("tc_occupancies");
  ana.tx->createBranch<std::vector<int>>("t5_occupancies");
  ana.tx->createBranch<int>("pT3_occupancies");
  ana.tx->createBranch<int>("pT5_occupancies");
}

//________________________________________________________________________________________________________________________________
unsigned int setSimTrackContainerBranches(LSTEvent* event) {
    //--------------------------------------------
    //
    //
    // Sim Tracks
    //
    //
    //--------------------------------------------
    
    auto const& trk_sim_pt = trk.getVF("sim_pt");
    auto const& trk_sim_eta = trk.getVF("sim_eta");
    auto const& trk_sim_phi = trk.getVF("sim_phi");
    auto const& trk_sim_bunchCrossing = trk.getVI("sim_bunchCrossing");
    auto const& trk_sim_event = trk.getVI("sim_event");
    auto const& trk_sim_pca_dxy = trk.getVF("sim_pca_dxy");
    auto const& trk_sim_pca_dz = trk.getVF("sim_pca_dz");
    auto const& trk_sim_q = trk.getVI("sim_q");
    auto const& trk_sim_pdgId = trk.getVI("sim_pdgId");
    auto const& trk_sim_parentVtxIdx = trk.getVI("sim_parentVtxIdx");
    auto const& trk_simvtx_x = trk.getVF("simvtx_x");
    auto const& trk_simvtx_y = trk.getVF("simvtx_y");
    auto const& trk_simvtx_z = trk.getVF("simvtx_z");
    auto const& trk_sim_simHitIdx = trk.getVVI("sim_simHitIdx");
    auto const& trk_simhit_subdet = trk.getVUS("simhit_subdet");
    auto const& trk_simhit_layer = trk.getVUS("simhit_layer");
    auto const& trk_simhit_x = trk.getVF("simhit_x");
    auto const& trk_simhit_y = trk.getVF("simhit_y");
    auto const& trk_simhit_z = trk.getVF("simhit_z");
    auto const& trk_simhit_detId = trk.getVU("simhit_detId");
    auto const& trk_ph2_x = trk.getVF("ph2_x");
    auto const& trk_ph2_y = trk.getVF("ph2_y");
    auto const& trk_ph2_z = trk.getVF("ph2_z");
    auto const& trk_ph2_detId = trk.getVU("ph2_detId");
  
    // Total number of simulated tracks with the condition that the simulated track came from a particle produced in the hard scattering and from the current bunch-crossing)
    // "accepted" here would mean that in the tracking ntuple (sim_bunchCrossing == 0 and sim_event == 0)
    unsigned int n_accepted_simtrk = 0;
  
    // Looping over the simulated tracks in the tracking ntuple
    for (unsigned int isimtrk = 0; isimtrk < trk_sim_pt.size(); ++isimtrk) {
      // Skip out-of-time pileup
      if (trk_sim_bunchCrossing[isimtrk] != 0)
        continue;
  
      // Skip non-hard-scatter
      if (trk_sim_event[isimtrk] != 0)
        continue;
  
      // Now we have a list of "accepted" tracks (no condition on vtx_z/perp, nor pt, eta etc are applied yet)
  
      // Fill the branch with simulated tracks.
      // N.B. these simulated tracks are looser than MTV denominator
      ana.tx->pushbackToBranch<float>("sim_pt", trk_sim_pt[isimtrk]);
      ana.tx->pushbackToBranch<float>("sim_eta", trk_sim_eta[isimtrk]);
      ana.tx->pushbackToBranch<float>("sim_phi", trk_sim_eta[isimtrk]);
      ana.tx->pushbackToBranch<float>("sim_pca_dxy", trk_sim_pca_dxy[isimtrk]);
      ana.tx->pushbackToBranch<float>("sim_pca_dz", trk_sim_pca_dz[isimtrk]);
      ana.tx->pushbackToBranch<int>("sim_q", trk_sim_q[isimtrk]);
      ana.tx->pushbackToBranch<int>("sim_pdgId", trk_sim_pdgId[isimtrk]);
  
      // For vertex we need to look it up from simvtx info for the given simtrack
      // for each simulated track, there is an index that points to the production vertex
      int vtxidx = trk_sim_parentVtxIdx[isimtrk];
      ana.tx->pushbackToBranch<float>("sim_vx", trk_simvtx_x[vtxidx]);  // using the index we retrieve xyz position
      ana.tx->pushbackToBranch<float>("sim_vy", trk_simvtx_y[vtxidx]);
      ana.tx->pushbackToBranch<float>("sim_vz", trk_simvtx_z[vtxidx]);
      ana.tx->pushbackToBranch<float>(
          "sim_vtxperp",
          sqrt(trk_simvtx_x[vtxidx] * trk_simvtx_x[vtxidx] + trk_simvtx_y[vtxidx] * trk_simvtx_y[vtxidx]));
  
      // Retrieve some track parameter information so we can build a helix
      float pt = trk_sim_pt[isimtrk];
      float eta = trk_sim_eta[isimtrk];
      float phi = trk_sim_phi[isimtrk];
      float vx = trk_simvtx_x[vtxidx];
      float vy = trk_simvtx_y[vtxidx];
      float vz = trk_simvtx_z[vtxidx];
      float charge = trk_sim_q[isimtrk];
  
      // Build the helix model. This model is useful to compute some specific expected hits.
      lst_math::Helix helix(pt, eta, phi, vx, vy, vz, charge);
  
      // Information to keep track of so we can save to output
      std::vector<int> simHitLayer;
      std::vector<float> simHitDistxyHelix;
      std::vector<float> simHitX;
      std::vector<float> simHitY;
      std::vector<float> simHitZ;
      std::vector<int> simHitDetId;
      std::vector<float> recoHitX;
      std::vector<float> recoHitY;
      std::vector<float> recoHitZ;
      std::vector<int> recoHitDetId;
      std::vector<float> simHitLayerMinDistxyHelix(11, 999);
  
      std::vector<std::vector<int>> simHitIdxs(11);
      float k2Rinv1GeVf = (2.99792458e-3 * 3.8) / 2;
  
      // Loop over the simhits (truth hits)
      for (size_t isimhit = 0; isimhit < trk_sim_simHitIdx[isimtrk].size(); ++isimhit) {
        // Retrieve the actual index to the simhit_* container of the tracking ntuple
        int isimhitidx = trk_sim_simHitIdx[isimtrk][isimhit];
  
        // Following computes the distance of the simhit's actual positionin xy to the "expected" xy position based on simhit's z position.
        // i.e. Take simhit's z position -> plug them into helix parametric function to obtain the xy position for that given z.
        // Then compare the computed xy position from the helix to the simhit's actualy xy position.
        // This is a measure of "how off from the original trajectory the simhits are?"
        // For example, if the particle got deflected early on due to material, then the xy position distance would be large.
        float distxyconsistent = distxySimHitConsistentWithHelix(helix, isimhitidx, trk_simhit_x, trk_simhit_y, trk_simhit_z);
  
        // Also retrieve some basic information about the simhit's location (layers, isbarrel?, etc.)
        // subdet == 4 means endcap of the outer tracker, subdet == 5 means barrel of the outer tracker)
        int subdet = trk_simhit_subdet[isimhitidx];
        int is_endcap = subdet == 4;
  
        // Now compute "logical layer" index
        // N.B. if a hit is in the inner tracker, layer would be staying at layer = 0
        int layer = 0;
        if (subdet == 4 or subdet == 5)  // this is not an outer tracker hit
          // this accounting makes it so that you have layer 1 2 3 4 5 6 in the barrel, and 7 8 9 10 11 in the endcap. (becuase endcap is ph2_subdet == 4)
          layer = trk_simhit_layer[isimhitidx] + 6 * (is_endcap);
  
        // keep track of isimhits in each layers so we can compute mindistxy from previous hit in previous layer
        if (subdet == 4 or subdet == 5)
          simHitIdxs[layer - 1].push_back(isimhitidx);
  
        // For this hit, now we push back to the vector that we are keeping track of
        simHitLayer.push_back(layer);
        simHitDistxyHelix.push_back(distxyconsistent);
        simHitX.push_back(trk_simhit_x[isimhitidx]);
        simHitY.push_back(trk_simhit_y[isimhitidx]);
        simHitZ.push_back(trk_simhit_z[isimhitidx]);
        simHitDetId.push_back(trk_simhit_detId[isimhitidx]);
  
        // Also retrieve all the reco-hits matched to this simhit and also aggregate them
        for (size_t irecohit = 0; irecohit < trk_sim_simHitIdx[isimhitidx].size(); ++irecohit) {
          recoHitX.push_back(trk_ph2_x[trk_sim_simHitIdx[isimhitidx][irecohit]]);
          recoHitY.push_back(trk_ph2_y[trk_sim_simHitIdx[isimhitidx][irecohit]]);
          recoHitZ.push_back(trk_ph2_z[trk_sim_simHitIdx[isimhitidx][irecohit]]);
          recoHitDetId.push_back(trk_ph2_detId[trk_sim_simHitIdx[isimhitidx][irecohit]]);
        }
  
        // If the given simhit that we are dealing with is not in the outer tracker (i.e. layer == 0. see few lines above.)
        // then, skip this simhit and go to the next hit.
        if (layer == 0)
          continue;
  
        // If it is a outer tracker hit, then we keep track of out of the 11 layers, what is the minimum "DistxyHelix" (distance to the expected point in the helix in xy)
        // This variable will have a fixed 11 float numbers, and using this to restrict "at least one hit that is not too far from the expected helix" can be useful to select some interesting denominator tracks.
        if (distxyconsistent < simHitLayerMinDistxyHelix[layer - 1]) {
          simHitLayerMinDistxyHelix[layer - 1] = distxyconsistent;
        }
      }
  
      // TODO: should this be enabled or removed?
      std::vector<float> simHitLayerMinDistxyHelixPrevHit(11, 999);
      std::vector<float> simHitLayeriSimHitMinDixtxyHelixPrevHit(11, -999);
      // // The algorithm will be to start with the main helix from the sim information and get the isimhit with least distxy.
      // // Then, from that you find the min distxy and repeat
      // for (int ilogicallayer = 0; ilogicallayer < 11; ++ilogicallayer)
      // {
      //     int ilayer = ilogicallayer - 1;
      //     float prev_pt, prev_eta, prev_phi, prev_vx, prev_vy, prev_vz;
      //     if (ilayer == 0)
      //     {
      //         prev_pt = pt;
      //         prev_eta = eta;
      //         prev_phi = phi;
      //         prev_vx = vx;
      //         prev_vy = vy;
      //         prev_vz = vz;
      //     }
      //     else
      //     {
      //         int isimhitidx = simHitLayeriSimHitMinDixtxyHelixPrevHit[ilayer - 1];
      //         TVector3 pp(trk.simhit_px()[isimhitidx], trk.simhit_py()[isimhitidx], trk.simhit_pz()[isimhitidx]);
      //         prev_pt = pp.Pt();
      //         prev_eta = pp.Eta();
      //         prev_phi = pp.Phi();
      //         prev_vx = trk.simhit_x()[isimhitidx];
      //         prev_vy = trk.simhit_y()[isimhitidx];
      //         prev_vz = trk.simhit_z()[isimhitidx];
      //     }
      //     SDLMath::Helix prev_helix(prev_pt, prev_eta, prev_phi, prev_vx, prev_vy, prev_vz, charge);
      //     for (int isimhit = 0; isimhit < simHitIdxs[ilayer].size(); ++isimhit)
      //     {
      //         int isimhitidx = simHitIdxs[ilayer][isimhit];
      //         float distxyconsistent = distxySimHitConsistentWithHelix(prev_helix, isimhitidx);
      //         if (simHitLayerMinDistxyHelixPrevHit[ilayer] > distxyconsistent)
      //         {
      //             simHitLayerMinDistxyHelixPrevHit[ilayer] = distxyconsistent;
      //             simHitLayeriSimHitMinDixtxyHelixPrevHit[ilayer] = isimhitidx;
      //         }
      //     }
      // }
  
      // Now we fill the branch
      ana.tx->pushbackToBranch<std::vector<int>>("sim_simHitLayer", simHitLayer);
      ana.tx->pushbackToBranch<std::vector<float>>("sim_simHitDistxyHelix", simHitDistxyHelix);
      ana.tx->pushbackToBranch<std::vector<float>>("sim_simHitLayerMinDistxyHelix", simHitLayerMinDistxyHelix);
      ana.tx->pushbackToBranch<std::vector<float>>("sim_simHitLayerMinDistxyPrevHit", simHitLayerMinDistxyHelixPrevHit);
      ana.tx->pushbackToBranch<std::vector<float>>("sim_simHitX", simHitX);
      ana.tx->pushbackToBranch<std::vector<float>>("sim_simHitY", simHitY);
      ana.tx->pushbackToBranch<std::vector<float>>("sim_simHitZ", simHitZ);
      ana.tx->pushbackToBranch<std::vector<int>>("sim_simHitDetId", simHitDetId);
      ana.tx->pushbackToBranch<std::vector<float>>("sim_recoHitX", recoHitX);
      ana.tx->pushbackToBranch<std::vector<float>>("sim_recoHitY", recoHitY);
      ana.tx->pushbackToBranch<std::vector<float>>("sim_recoHitZ", recoHitZ);
      ana.tx->pushbackToBranch<std::vector<int>>("sim_recoHitDetId", recoHitDetId);
  
      // The trkNtupIdx is the idx in the trackingNtuple
      ana.tx->pushbackToBranch<float>("sim_trkNtupIdx", isimtrk);
  
      // Increase the counter for accepted simtrk
      n_accepted_simtrk++;
    }
    
    return n_accepted_simtrk;
}

//________________________________________________________________________________________________________________________________
void setTrackCandidateBranches(LSTEvent* event, unsigned int n_accepted_simtrk,
#ifdef CUT_VALUE_DEBUG
  std::map<unsigned int, unsigned int> t5_idx_map,
  std::map<unsigned int, unsigned int> pls_idx_map,
  std::map<unsigned int, unsigned int> pt3_idx_map,
  std::map<unsigned int, unsigned int> pt5_idx_map,
#endif
    float matchfrac) {
    //--------------------------------------------
    //
    //
    // Track Candidates
    //
    //
    //--------------------------------------------
  
    auto const& trk_sim_pt = trk.getVF("sim_pt");
    auto const& trk_ph2_x = trk.getVF("ph2_x");
    auto const& trk_ph2_y = trk.getVF("ph2_y");
    auto const& trk_ph2_z = trk.getVF("ph2_z");
    auto const& trk_simhit_simTrkIdx = trk.getVI("simhit_simTrkIdx");
    auto const& trk_ph2_simHitIdx = trk.getVVI("ph2_simHitIdx");
    auto const& trk_pix_simHitIdx = trk.getVVI("pix_simHitIdx");
    
    auto trackCandidates = event->getTrackCandidates();
    
    // Following are some vectors to keep track of the information to write to the ntuple
    // N.B. following two branches have a length for the entire sim track, but what actually will be written in sim_tcIdxAll branch is NOT that long
    // Later in the code, it will restrict to only the ones to write out.
    // The reason at this stage, the entire tcIdxAll is being tracked is to compute duplicate properly later on
    // When computing a duplicate object it is important to consider all simulated tracks including pileup tracks
    int n_total_simtrk = trk_sim_pt.size();
    std::vector<std::vector<int>> sim_tcIdxAll(n_total_simtrk);
    std::vector<std::vector<float>> sim_tcIdxAllFrac(n_total_simtrk);
    std::vector<std::vector<int>> tc_simIdxAll;
    std::vector<std::vector<float>> tc_simIdxAllFrac;
  
    // Number of total track candidates created in this event
    unsigned int nTrackCandidates = trackCandidates.nTrackCandidates();
  
    // Looping over each track candidate
    for (unsigned int tc_idx = 0; tc_idx < nTrackCandidates; tc_idx++) {
      // Compute reco quantities of track candidate based on final object
      int type, isFake;
      float pt, eta, phi;
      std::vector<int> simidx;        // list of all the matched sim idx
      std::vector<float> simidxfrac;  // list of match fraction for each matched sim idx
  
      // The following function reads off and computes the matched sim track indices
      std::tie(type, pt, eta, phi, isFake, simidx, simidxfrac) = parseTrackCandidateAllMatch(event, tc_idx, trk_ph2_x, trk_ph2_y, trk_ph2_z, trk_simhit_simTrkIdx, trk_ph2_simHitIdx, trk_pix_simHitIdx, matchfrac);
  
      // Fill some branches for this track candidate
      ana.tx->pushbackToBranch<float>("tc_pt", pt);
      ana.tx->pushbackToBranch<float>("tc_eta", eta);
      ana.tx->pushbackToBranch<float>("tc_phi", phi);
      ana.tx->pushbackToBranch<int>("tc_type", type);
#ifdef CUT_VALUE_DEBUG
      if (type == LSTObjType::pT5) {
        ana.tx->pushbackToBranch<int>("tc_pt5Idx", pt5_idx_map[trackCandidates.directObjectIndices()[tc_idx]]);
        ana.tx->pushbackToBranch<int>("tc_pt3Idx", -999);
        ana.tx->pushbackToBranch<int>("tc_t5Idx", -999);
        ana.tx->pushbackToBranch<int>("tc_plsIdx", -999);
      } else if (type == LSTObjType::pT3) {
        ana.tx->pushbackToBranch<int>("tc_pt5Idx", -999);
        ana.tx->pushbackToBranch<int>("tc_pt3Idx", pt3_idx_map[trackCandidates.directObjectIndices()[tc_idx]]);
        ana.tx->pushbackToBranch<int>("tc_t5Idx", -999);
        ana.tx->pushbackToBranch<int>("tc_plsIdx", -999);
      } else if (type == LSTObjType::T5) {
        ana.tx->pushbackToBranch<int>("tc_pt5Idx", -999);
        ana.tx->pushbackToBranch<int>("tc_pt3Idx", -999);
        ana.tx->pushbackToBranch<int>("tc_t5Idx", t5_idx_map[trackCandidates.directObjectIndices()[tc_idx]]);
        ana.tx->pushbackToBranch<int>("tc_plsIdx", -999);
      } else if (type == LSTObjType::pLS) {
        ana.tx->pushbackToBranch<int>("tc_pt5Idx", -999);
        ana.tx->pushbackToBranch<int>("tc_pt3Idx", -999);
        ana.tx->pushbackToBranch<int>("tc_t5Idx", -999);
        ana.tx->pushbackToBranch<int>("tc_plsIdx",
                                      pls_idx_map[ranges.segmentModuleIndices()[modules.nLowerModules()] +
                                                  trackCandidates.directObjectIndices()[tc_idx]]);
      }
#endif
  
      ana.tx->pushbackToBranch<int>("tc_isFake", isFake);
  
      // For this tc, keep track of all the simidx that are matched
      tc_simIdxAll.push_back(simidx);
      tc_simIdxAllFrac.push_back(simidxfrac);
  
      // The book keeping of opposite mapping is done here
      // For each matched sim idx, we go back and keep track of which tc it is matched to.
      // Loop over all the matched sim idx
      for (size_t is = 0; is < simidx.size(); ++is) {
        // For this matched sim index keep track (sim -> tc) mapping
        int sim_idx = simidx.at(is);
        float sim_idx_frac = simidxfrac.at(is);
        sim_tcIdxAll.at(sim_idx).push_back(tc_idx);
        sim_tcIdxAllFrac.at(sim_idx).push_back(sim_idx_frac);
      }
  
      // Also, among the simidx matches, find the best match (highest fractional match)
      // N.B. the simidx is already returned sorted by highest number of "nhits" match
      // So as it loops over, the condition will ensure that the highest fraction with highest nhits will be matched with the priority given to highest fraction
      int tc_simIdx = -999;
      float tc_simIdxBestFrac = 0;
      for (size_t isim = 0; isim < simidx.size(); ++isim) {
        int thisidx = simidx[isim];
        float thisfrac = simidxfrac[isim];
        if (thisfrac > tc_simIdxBestFrac and thisfrac > matchfrac) {
          tc_simIdxBestFrac = thisfrac;
          tc_simIdx = thisidx;
        }
      }
  
      // the best match index will then be saved here
      ana.tx->pushbackToBranch<int>("tc_simIdx", tc_simIdx);
    }
  
    // Now save the (tc -> simidx) mapping
    ana.tx->setBranch<std::vector<std::vector<int>>>("tc_simIdxAll", tc_simIdxAll);
    ana.tx->setBranch<std::vector<std::vector<float>>>("tc_simIdxAllFrac", tc_simIdxAllFrac);
  
    // Not all (sim->tcIdx) will be saved but only for the sim that is from hard scatter and current bunch crossing
    // So a restriction up to only "n_accepted_simtrk" done by chopping off the rest
    // N.B. the reason we can simply take the first "n_accepted_simtrk" is because the tracking ntuple is organized such that those sim tracks show up on the first "n_accepted_simtrk" of tracks.
    std::vector<std::vector<int>> sim_tcIdxAll_to_write;
    std::vector<std::vector<float>> sim_tcIdxAllFrac_to_write;
    std::copy(sim_tcIdxAll.begin(),
              sim_tcIdxAll.begin() + n_accepted_simtrk,
              std::back_inserter(
                  sim_tcIdxAll_to_write));  // this is where the vector is only copying the first "n_accepted_simtrk"
    std::copy(sim_tcIdxAllFrac.begin(),
              sim_tcIdxAllFrac.begin() + n_accepted_simtrk,
              std::back_inserter(sim_tcIdxAllFrac_to_write));  // ditto
    ana.tx->setBranch<std::vector<std::vector<int>>>("sim_tcIdxAll", sim_tcIdxAll_to_write);
    ana.tx->setBranch<std::vector<std::vector<float>>>("sim_tcIdxAllFrac", sim_tcIdxAllFrac_to_write);
  
    // Using the intermedaite variables to compute whether a given track candidate is a duplicate
    std::vector<int> tc_isDuplicate(tc_simIdxAll.size());
  
    // Loop over the track candidates
    for (unsigned int tc_idx = 0; tc_idx < tc_simIdxAll.size(); ++tc_idx) {
      bool isDuplicate = false;
      // Loop over the sim idx matched to this track candidate
      for (unsigned int isim = 0; isim < tc_simIdxAll[tc_idx].size(); ++isim) {
        int sim_idx = tc_simIdxAll[tc_idx][isim];
        int n_sim_matched = 0;
        for (size_t ism = 0; ism < sim_tcIdxAll.at(sim_idx).size(); ++ism) {
          if (sim_tcIdxAllFrac.at(sim_idx).at(ism) > matchfrac) {
            n_sim_matched += 1;
            if (n_sim_matched > 1) {
              isDuplicate = true;
              break;
            }
          }
        }
      }
      tc_isDuplicate[tc_idx] = isDuplicate;
    }
    ana.tx->setBranch<std::vector<int>>("tc_isDuplicate", tc_isDuplicate);
  
    // Similarly, the best match for the (sim -> tc is computed)
    // TODO: Is this redundant? I am not sure if it is guaranteed that sim_tcIdx will have same result with tc_simIdx.
    // I think it will be, but I have not rigorously checked. I only checked about first few thousands and it was all true. as long as tc->sim was pointing to a sim that is among the n_accepted.
    // For the most part I think this won't be a problem.
    for (size_t i = 0; i < sim_tcIdxAll_to_write.size(); ++i) {
      // bestmatch is not always the first one
      int bestmatch_idx = -999;
      float bestmatch_frac = -999;
      for (size_t jj = 0; jj < sim_tcIdxAll_to_write.at(i).size(); ++jj) {
        int idx = sim_tcIdxAll_to_write.at(i).at(jj);
        float frac = sim_tcIdxAllFrac_to_write.at(i).at(jj);
        if (bestmatch_frac < frac) {
          bestmatch_idx = idx;
          bestmatch_frac = frac;
        }
      }
      ana.tx->pushbackToBranch<int>("sim_tcIdxBest", bestmatch_idx);
      ana.tx->pushbackToBranch<float>("sim_tcIdxBestFrac", bestmatch_frac);
      if (bestmatch_frac > matchfrac)  // then this is a good match according to MTV
        ana.tx->pushbackToBranch<int>("sim_tcIdx", bestmatch_idx);
      else
        ana.tx->pushbackToBranch<int>("sim_tcIdx", -999);
    }
}

//________________________________________________________________________________________________________________________________
void setOccupancyBranches(LSTEvent* event) {
  auto modules = event->getModules<ModulesSoA>();
  auto miniDoublets = event->getMiniDoublets<MiniDoubletsOccupancySoA>();
  auto segments = event->getSegments<SegmentsOccupancySoA>();
  auto triplets = event->getTriplets<TripletsOccupancySoA>();
  auto quintuplets = event->getQuintuplets<QuintupletsOccupancySoA>();
  auto pixelQuintuplets = event->getPixelQuintuplets();
  auto pixelTriplets = event->getPixelTriplets();
  auto trackCandidates = event->getTrackCandidates();

  std::vector<int> moduleLayer;
  std::vector<int> moduleSubdet;
  std::vector<int> moduleRing;
  std::vector<int> moduleRod;
  std::vector<int> moduleModule;
  std::vector<float> moduleEta;
  std::vector<float> moduleR;
  std::vector<bool> moduleIsTilted;
  std::vector<int> trackCandidateOccupancy;
  std::vector<int> tripletOccupancy;
  std::vector<int> segmentOccupancy;
  std::vector<int> mdOccupancy;
  std::vector<int> quintupletOccupancy;

  for (unsigned int lowerIdx = 0; lowerIdx <= modules.nLowerModules(); lowerIdx++) {
    //layer = 0, subdet = 0 => pixel module
    moduleLayer.push_back(modules.layers()[lowerIdx]);
    moduleSubdet.push_back(modules.subdets()[lowerIdx]);
    moduleRing.push_back(modules.rings()[lowerIdx]);
    moduleRod.push_back(modules.rods()[lowerIdx]);
    moduleEta.push_back(modules.eta()[lowerIdx]);
    moduleR.push_back(modules.r()[lowerIdx]);
    bool isTilted = (modules.subdets()[lowerIdx] == 5 and modules.sides()[lowerIdx] != 3);
    moduleIsTilted.push_back(isTilted);
    moduleModule.push_back(modules.modules()[lowerIdx]);
    segmentOccupancy.push_back(segments.totOccupancySegments()[lowerIdx]);
    mdOccupancy.push_back(miniDoublets.totOccupancyMDs()[lowerIdx]);

    if (lowerIdx < modules.nLowerModules()) {
      quintupletOccupancy.push_back(quintuplets.totOccupancyQuintuplets()[lowerIdx]);
      tripletOccupancy.push_back(triplets.totOccupancyTriplets()[lowerIdx]);
    }
  }

  ana.tx->setBranch<std::vector<int>>("module_layers", moduleLayer);
  ana.tx->setBranch<std::vector<int>>("module_subdets", moduleSubdet);
  ana.tx->setBranch<std::vector<int>>("module_rings", moduleRing);
  ana.tx->setBranch<std::vector<int>>("module_rods", moduleRod);
  ana.tx->setBranch<std::vector<int>>("module_modules", moduleModule);
  ana.tx->setBranch<std::vector<bool>>("module_isTilted", moduleIsTilted);
  ana.tx->setBranch<std::vector<float>>("module_eta", moduleEta);
  ana.tx->setBranch<std::vector<float>>("module_r", moduleR);
  ana.tx->setBranch<std::vector<int>>("md_occupancies", mdOccupancy);
  ana.tx->setBranch<std::vector<int>>("sg_occupancies", segmentOccupancy);
  ana.tx->setBranch<std::vector<int>>("t3_occupancies", tripletOccupancy);
  ana.tx->setBranch<int>("tc_occupancies", trackCandidates.nTrackCandidates());
  ana.tx->setBranch<int>("pT3_occupancies", pixelTriplets.totOccupancyPixelTriplets());
  ana.tx->setBranch<std::vector<int>>("t5_occupancies", quintupletOccupancy);
  ana.tx->setBranch<int>("pT5_occupancies", pixelQuintuplets.totOccupancyPixelQuintuplets());
}

//________________________________________________________________________________________________________________________________
void setPixelQuintupletOutputBranches(LSTEvent* event) {
  // ============ pT5 =============
  auto const pixelQuintuplets = event->getPixelQuintuplets();
  auto const quintuplets = event->getQuintuplets<QuintupletsSoA>();
  auto const pixelSeeds = event->getInput<PixelSeedsSoA>();
  auto modules = event->getModules<ModulesSoA>();
  int n_accepted_simtrk = ana.tx->getBranch<std::vector<int>>("sim_TC_matched").size();

  unsigned int nPixelQuintuplets = pixelQuintuplets.nPixelQuintuplets();
  std::vector<int> sim_pT5_matched(n_accepted_simtrk);
  std::vector<std::vector<int>> pT5_matched_simIdx;

  auto const& trk_simhit_simTrkIdx = trk.getVI("simhit_simTrkIdx");
  auto const& trk_ph2_simHitIdx = trk.getVVI("ph2_simHitIdx");
  auto const& trk_pix_simHitIdx = trk.getVVI("pix_simHitIdx");

  for (unsigned int pT5 = 0; pT5 < nPixelQuintuplets; pT5++) {
    unsigned int T5Index = getT5FrompT5(event, pT5);
    unsigned int pLSIndex = getPixelLSFrompT5(event, pT5);
    float pt = (__H2F(quintuplets.innerRadius()[T5Index]) * k2Rinv1GeVf * 2 + pixelSeeds.ptIn()[pLSIndex]) / 2;
    float eta = pixelSeeds.eta()[pLSIndex];
    float phi = pixelSeeds.phi()[pLSIndex];

    std::vector<unsigned int> hit_idx = getHitIdxsFrompT5(event, pT5);
    std::vector<unsigned int> module_idx = getModuleIdxsFrompT5(event, pT5);
    std::vector<unsigned int> hit_type = getHitTypesFrompT5(event, pT5);

    int layer_binary = 1;
    int moduleType_binary = 0;
    for (size_t i = 0; i < module_idx.size(); i += 2) {
      layer_binary |= (1 << (modules.layers()[module_idx[i]] + 6 * (modules.subdets()[module_idx[i]] == 4)));
      moduleType_binary |= (modules.moduleType()[module_idx[i]] << i);
    }
    std::vector<int> simidx =
        matchedSimTrkIdxs(hit_idx, hit_type, trk_simhit_simTrkIdx, trk_ph2_simHitIdx, trk_pix_simHitIdx);
    ana.tx->pushbackToBranch<int>("pT5_isFake", static_cast<int>(simidx.size() == 0));
    ana.tx->pushbackToBranch<float>("pT5_pt", pt);
    ana.tx->pushbackToBranch<float>("pT5_eta", eta);
    ana.tx->pushbackToBranch<float>("pT5_phi", phi);
    ana.tx->pushbackToBranch<int>("pT5_layer_binary", layer_binary);
    ana.tx->pushbackToBranch<int>("pT5_moduleType_binary", moduleType_binary);
    ana.tx->pushbackToBranch<float>("pT5_rzChiSquared", pixelQuintuplets.rzChiSquared()[pT5]);

    pT5_matched_simIdx.push_back(simidx);

    // Loop over matched sim idx and increase counter of pT5_matched
    for (auto& idx : simidx) {
      // NOTE Important to note that the idx of the std::vector<> is same
      // as the tracking-ntuple's sim track idx ONLY because event==0 and bunchCrossing==0 condition is applied!!
      // Also do not try to access beyond the event and bunchCrossing
      if (idx < n_accepted_simtrk) {
        sim_pT5_matched.at(idx) += 1;
      }
    }
  }

  // Using the intermedaite variables to compute whether a given track candidate is a duplicate
  std::vector<int> pT5_isDuplicate(pT5_matched_simIdx.size());
  // Loop over the track candidates
  for (unsigned int i = 0; i < pT5_matched_simIdx.size(); ++i) {
    bool isDuplicate = false;
    // Loop over the sim idx matched to this track candidate
    for (unsigned int isim = 0; isim < pT5_matched_simIdx[i].size(); ++isim) {
      // Using the sim_pT5_matched to see whether this track candidate is matched to a sim track that is matched to more than one
      int simidx = pT5_matched_simIdx[i][isim];
      if (simidx < n_accepted_simtrk) {
        if (sim_pT5_matched[simidx] > 1) {
          isDuplicate = true;
        }
      }
    }
    pT5_isDuplicate[i] = isDuplicate;
  }

  // Now set the last remaining branches
  ana.tx->setBranch<std::vector<int>>("sim_pT5_matched", sim_pT5_matched);
  ana.tx->setBranch<std::vector<std::vector<int>>>("pT5_matched_simIdx", pT5_matched_simIdx);
  ana.tx->setBranch<std::vector<int>>("pT5_isDuplicate", pT5_isDuplicate);
}

//________________________________________________________________________________________________________________________________
void setQuintupletOutputBranches(LSTEvent* event) {
  auto const quintuplets = event->getQuintuplets<QuintupletsSoA>();
  auto const quintupletsOccupancy = event->getQuintuplets<QuintupletsOccupancySoA>();
  auto ranges = event->getRanges();
  auto modules = event->getModules<ModulesSoA>();
  int n_accepted_simtrk = ana.tx->getBranch<std::vector<int>>("sim_TC_matched").size();

  std::vector<int> sim_t5_matched(n_accepted_simtrk);
  std::vector<std::vector<int>> t5_matched_simIdx;

  auto const& trk_sim_parentVtxIdx = trk.getVI("sim_parentVtxIdx");
  auto const& trk_simvtx_x = trk.getVF("simvtx_x");
  auto const& trk_simvtx_y = trk.getVF("simvtx_y");
  auto const& trk_simvtx_z = trk.getVF("simvtx_z");
  auto const& trk_simhit_simTrkIdx = trk.getVI("simhit_simTrkIdx");
  auto const& trk_ph2_simHitIdx = trk.getVVI("ph2_simHitIdx");
  auto const& trk_pix_simHitIdx = trk.getVVI("pix_simHitIdx");

  for (unsigned int lowerModuleIdx = 0; lowerModuleIdx < modules.nLowerModules(); ++lowerModuleIdx) {
    int nQuintuplets = quintupletsOccupancy.nQuintuplets()[lowerModuleIdx];
    for (unsigned int idx = 0; idx < nQuintuplets; idx++) {
      unsigned int quintupletIndex = ranges.quintupletModuleIndices()[lowerModuleIdx] + idx;
      float pt = __H2F(quintuplets.innerRadius()[quintupletIndex]) * k2Rinv1GeVf * 2;
      float eta = __H2F(quintuplets.eta()[quintupletIndex]);
      float phi = __H2F(quintuplets.phi()[quintupletIndex]);

      std::vector<unsigned int> hit_idx = getHitIdxsFromT5(event, quintupletIndex);
      std::vector<unsigned int> hit_type = getHitTypesFromT5(event, quintupletIndex);
      std::vector<unsigned int> module_idx = getModuleIdxsFromT5(event, quintupletIndex);

      int layer_binary = 0;
      int moduleType_binary = 0;
      for (size_t i = 0; i < module_idx.size(); i += 2) {
        layer_binary |= (1 << (modules.layers()[module_idx[i]] + 6 * (modules.subdets()[module_idx[i]] == 4)));
        moduleType_binary |= (modules.moduleType()[module_idx[i]] << i);
      }

      float percent_matched;
      std::vector<int> simidx = matchedSimTrkIdxs(
          hit_idx, hit_type, trk_simhit_simTrkIdx, trk_ph2_simHitIdx, trk_pix_simHitIdx, false);

      ana.tx->pushbackToBranch<int>("t5_isFake", static_cast<int>(simidx.size() == 0));
      ana.tx->pushbackToBranch<float>("t5_pt", pt);
      ana.tx->pushbackToBranch<float>("t5_pMatched", percent_matched);
      ana.tx->pushbackToBranch<float>("t5_eta", eta);
      ana.tx->pushbackToBranch<float>("t5_phi", phi);
      ana.tx->pushbackToBranch<float>("t5_innerRadius", __H2F(quintuplets.innerRadius()[quintupletIndex]));
      ana.tx->pushbackToBranch<float>("t5_bridgeRadius", __H2F(quintuplets.bridgeRadius()[quintupletIndex]));
      ana.tx->pushbackToBranch<float>("t5_outerRadius", __H2F(quintuplets.outerRadius()[quintupletIndex]));
      ana.tx->pushbackToBranch<int>("t5_isDupAlgoFlag", quintuplets.isDup()[quintupletIndex]);
      ana.tx->pushbackToBranch<float>("t5_chiSquared", quintuplets.chiSquared()[quintupletIndex]);
      ana.tx->pushbackToBranch<float>("t5_rzChiSquared", quintuplets.rzChiSquared()[quintupletIndex]);
      ana.tx->pushbackToBranch<float>("t5_nonAnchorChiSquared", quintuplets.nonAnchorChiSquared()[quintupletIndex]);
      ana.tx->pushbackToBranch<float>("t5_dBeta1", quintuplets.dBeta1()[quintupletIndex]);
      ana.tx->pushbackToBranch<float>("t5_dBeta2", quintuplets.dBeta2()[quintupletIndex]);
      ana.tx->pushbackToBranch<int>("t5_layer_binary", layer_binary);
      ana.tx->pushbackToBranch<int>("t5_moduleType_binary", moduleType_binary);

      t5_matched_simIdx.push_back(simidx);

      for (auto& simtrk : simidx) {
        if (simtrk < n_accepted_simtrk) {
          sim_t5_matched.at(simtrk) += 1;
        }
      }

      // Avoid fakes when calculating the vertex distance, set default to 0.0.
      if (simidx.size() == 0) {
        ana.tx->pushbackToBranch<float>("t5_sim_vxy", 0.0);
        ana.tx->pushbackToBranch<float>("t5_sim_vz", 0.0);
        continue;
      }

      int vtxidx = trk_sim_parentVtxIdx[simidx[0]];
      float vtx_x = trk_simvtx_x[vtxidx];
      float vtx_y = trk_simvtx_y[vtxidx];
      float vtx_z = trk_simvtx_z[vtxidx];

      ana.tx->pushbackToBranch<float>("t5_sim_vxy", sqrt(vtx_x * vtx_x + vtx_y * vtx_y));
      ana.tx->pushbackToBranch<float>("t5_sim_vz", vtx_z);
    }
  }

  std::vector<int> t5_isDuplicate(t5_matched_simIdx.size());
  for (unsigned int i = 0; i < t5_matched_simIdx.size(); i++) {
    bool isDuplicate = false;
    for (unsigned int isim = 0; isim < t5_matched_simIdx[i].size(); isim++) {
      int simidx = t5_matched_simIdx[i][isim];
      if (simidx < n_accepted_simtrk) {
        if (sim_t5_matched[simidx] > 1) {
          isDuplicate = true;
        }
      }
    }
    t5_isDuplicate[i] = isDuplicate;
  }
  ana.tx->setBranch<std::vector<int>>("sim_T5_matched", sim_t5_matched);
  ana.tx->setBranch<std::vector<std::vector<int>>>("t5_matched_simIdx", t5_matched_simIdx);
  ana.tx->setBranch<std::vector<int>>("t5_isDuplicate", t5_isDuplicate);
}

//________________________________________________________________________________________________________________________________
void setPixelTripletOutputBranches(LSTEvent* event) {
  auto const pixelTriplets = event->getPixelTriplets();
  auto modules = event->getModules<ModulesSoA>();
  PixelSeedsConst pixelSeeds = event->getInput<PixelSeedsSoA>();
  int n_accepted_simtrk = ana.tx->getBranch<std::vector<int>>("sim_TC_matched").size();

  unsigned int nPixelTriplets = pixelTriplets.nPixelTriplets();
  std::vector<int> sim_pT3_matched(n_accepted_simtrk);
  std::vector<std::vector<int>> pT3_matched_simIdx;

  auto const& trk_simhit_simTrkIdx = trk.getVI("simhit_simTrkIdx");
  auto const& trk_ph2_simHitIdx = trk.getVVI("ph2_simHitIdx");
  auto const& trk_pix_simHitIdx = trk.getVVI("pix_simHitIdx");

  for (unsigned int pT3 = 0; pT3 < nPixelTriplets; pT3++) {
    unsigned int T3Index = getT3FrompT3(event, pT3);
    unsigned int pLSIndex = getPixelLSFrompT3(event, pT3);
    const float pt = pixelSeeds.ptIn()[pLSIndex];

    float eta = pixelSeeds.eta()[pLSIndex];
    float phi = pixelSeeds.phi()[pLSIndex];
    std::vector<unsigned int> hit_idx = getHitIdxsFrompT3(event, pT3);
    std::vector<unsigned int> hit_type = getHitTypesFrompT3(event, pT3);

    std::vector<int> simidx =
        matchedSimTrkIdxs(hit_idx, hit_type, trk_simhit_simTrkIdx, trk_ph2_simHitIdx, trk_pix_simHitIdx);
    std::vector<unsigned int> module_idx = getModuleIdxsFrompT3(event, pT3);
    int layer_binary = 1;
    int moduleType_binary = 0;
    for (size_t i = 0; i < module_idx.size(); i += 2) {
      layer_binary |= (1 << (modules.layers()[module_idx[i]] + 6 * (modules.subdets()[module_idx[i]] == 4)));
      moduleType_binary |= (modules.moduleType()[module_idx[i]] << i);
    }
    ana.tx->pushbackToBranch<int>("pT3_isFake", static_cast<int>(simidx.size() == 0));
    ana.tx->pushbackToBranch<float>("pT3_pt", pt);
    ana.tx->pushbackToBranch<float>("pT3_eta", eta);
    ana.tx->pushbackToBranch<float>("pT3_phi", phi);
    ana.tx->pushbackToBranch<int>("pT3_layer_binary", layer_binary);
    ana.tx->pushbackToBranch<int>("pT3_moduleType_binary", moduleType_binary);

    pT3_matched_simIdx.push_back(simidx);

    for (auto& idx : simidx) {
      if (idx < n_accepted_simtrk) {
        sim_pT3_matched.at(idx) += 1;
      }
    }
  }

  std::vector<int> pT3_isDuplicate(pT3_matched_simIdx.size());
  for (unsigned int i = 0; i < pT3_matched_simIdx.size(); i++) {
    bool isDuplicate = true;
    for (unsigned int isim = 0; isim < pT3_matched_simIdx[i].size(); isim++) {
      int simidx = pT3_matched_simIdx[i][isim];
      if (simidx < n_accepted_simtrk) {
        if (sim_pT3_matched[simidx] > 1) {
          isDuplicate = true;
        }
      }
    }
    pT3_isDuplicate[i] = isDuplicate;
  }
  ana.tx->setBranch<std::vector<int>>("sim_pT3_matched", sim_pT3_matched);
  ana.tx->setBranch<std::vector<std::vector<int>>>("pT3_matched_simIdx", pT3_matched_simIdx);
  ana.tx->setBranch<std::vector<int>>("pT3_isDuplicate", pT3_isDuplicate);
}

//________________________________________________________________________________________________________________________________
void fillpT3DNNBranches(LSTEvent* event, unsigned int iPT3) {
  // Retrieve the pT3 object from the PixelTriplets SoA.
  auto pixelTriplets = event->getPixelTriplets();

  float pixelRadius = pixelTriplets.pixelRadius()[iPT3];
  float pixelRadiusError = pixelTriplets.pixelRadiusError()[iPT3];
  float tripletRadius = pixelTriplets.tripletRadius()[iPT3];
  float phi = pixelTriplets.phi()[iPT3];          // from the T3
  float phi_pix = pixelTriplets.phi_pix()[iPT3];  // from the pLS
  float rPhiChiSquared = pixelTriplets.rPhiChiSquared()[iPT3];
  float rPhiChiSquaredInwards = pixelTriplets.rPhiChiSquaredInwards()[iPT3];
  float rzChiSquared = pixelTriplets.rzChiSquared()[iPT3];
  float pt = pixelTriplets.pt()[iPT3];
  float eta = pixelTriplets.eta()[iPT3];
  float eta_pix = pixelTriplets.eta_pix()[iPT3];  // eta from pLS
  float centerX = pixelTriplets.centerX()[iPT3];  // T3-based circle center x
  float centerY = pixelTriplets.centerY()[iPT3];  // T3-based circle center y

  ana.tx->pushbackToBranch<float>("pT3_rPhiChiSquared", rPhiChiSquared);
  ana.tx->pushbackToBranch<float>("pT3_rPhiChiSquaredInwards", rPhiChiSquaredInwards);
  ana.tx->pushbackToBranch<float>("pT3_rzChiSquared", rzChiSquared);
  ana.tx->pushbackToBranch<float>("pT3_pixelRadius", pixelRadius);
  ana.tx->pushbackToBranch<float>("pT3_pixelRadiusError", pixelRadiusError);
  ana.tx->pushbackToBranch<float>("pT3_tripletRadius", tripletRadius);
}

//________________________________________________________________________________________________________________________________
void fillT3DNNBranches(LSTEvent* event, unsigned int iT3) {
  auto hitsBase = event->getInput<HitsBaseSoA>();
  auto hitsExtended = event->getHits<HitsExtendedSoA>();
  auto modules = event->getModules<ModulesSoA>();

  std::vector<unsigned int> hitIdx = getHitsFromT3(event, iT3);
  std::vector<lst_math::Hit> hitObjects;

  auto const& trk_ph2_subdet = trk.getVUS("ph2_subdet");
  auto const& trk_ph2_layer = trk.getVUS("ph2_layer");
  auto const& trk_ph2_detId = trk.getVU("ph2_detId");

  for (int i = 0; i < hitIdx.size(); ++i) {
    unsigned int hit = hitIdx[i];
    float x = hitsBase.xs()[hit];
    float y = hitsBase.ys()[hit];
    float z = hitsBase.zs()[hit];
    lst_math::Hit hitObj(x, y, z);
    hitObjects.push_back(hitObj);

    std::string idx = std::to_string(i);
    ana.tx->pushbackToBranch<float>("t3_hit_" + idx + "_r", sqrt(x * x + y * y));
    ana.tx->pushbackToBranch<float>("t3_hit_" + idx + "_x", x);
    ana.tx->pushbackToBranch<float>("t3_hit_" + idx + "_y", y);
    ana.tx->pushbackToBranch<float>("t3_hit_" + idx + "_z", z);
    ana.tx->pushbackToBranch<float>("t3_hit_" + idx + "_eta", hitObj.eta());
    ana.tx->pushbackToBranch<float>("t3_hit_" + idx + "_phi", hitObj.phi());

    int subdet = trk_ph2_subdet[hitsBase.idxs()[hit]];
    int is_endcap = subdet == 4;
    int layer = trk_ph2_layer[hitsBase.idxs()[hit]] + 6 * is_endcap;
    int detId = trk_ph2_detId[hitsBase.idxs()[hit]];
    unsigned int module = hitsExtended.moduleIndices()[hit];

    ana.tx->pushbackToBranch<int>("t3_hit_" + idx + "_detId", detId);
    ana.tx->pushbackToBranch<int>("t3_hit_" + idx + "_layer", layer);
    ana.tx->pushbackToBranch<int>("t3_hit_" + idx + "_moduleType", modules.moduleType()[module]);
  }
}

//________________________________________________________________________________________________________________________________
void fillT5DNNBranches(LSTEvent* event, unsigned int iT3) {
  auto hitsBase = event->getInput<HitsBaseSoA>();
  auto hitsExtended = event->getHits<HitsExtendedSoA>();
  auto modules = event->getModules<ModulesSoA>();

  std::vector<unsigned int> hitIdx = getHitsFromT3(event, iT3);
  std::vector<lst_math::Hit> hitObjects(hitIdx.size());

  auto const& trk_ph2_subdet = trk.getVUS("ph2_subdet");
  auto const& trk_ph2_layer = trk.getVUS("ph2_layer");
  auto const& trk_ph2_detId = trk.getVU("ph2_detId");

  for (int i = 0; i < hitIdx.size(); ++i) {
    unsigned int hit = hitIdx[i];
    float x = hitsBase.xs()[hit];
    float y = hitsBase.ys()[hit];
    float z = hitsBase.zs()[hit];
    hitObjects[i] = lst_math::Hit(x, y, z);

    std::string idx = std::to_string(i);
    ana.tx->pushbackToBranch<float>("t5_t3_" + idx + "_r", sqrt(x * x + y * y));
    ana.tx->pushbackToBranch<float>("t5_t3_" + idx + "_x", x);
    ana.tx->pushbackToBranch<float>("t5_t3_" + idx + "_y", y);
    ana.tx->pushbackToBranch<float>("t5_t3_" + idx + "_z", z);
    ana.tx->pushbackToBranch<float>("t5_t3_" + idx + "_eta", hitObjects[i].eta());
    ana.tx->pushbackToBranch<float>("t5_t3_" + idx + "_phi", hitObjects[i].phi());

    int subdet = trk_ph2_subdet[hitsBase.idxs()[hit]];
    int is_endcap = subdet == 4;
    int layer = trk_ph2_layer[hitsBase.idxs()[hit]] + 6 * is_endcap;
    int detId = trk_ph2_detId[hitsBase.idxs()[hit]];
    unsigned int module = hitsExtended.moduleIndices()[hit];

    ana.tx->pushbackToBranch<int>("t5_t3_" + idx + "_detId", detId);
    ana.tx->pushbackToBranch<int>("t5_t3_" + idx + "_layer", layer);
    ana.tx->pushbackToBranch<int>("t5_t3_" + idx + "_moduleType", modules.moduleType()[module]);
  }

  float radius;
  auto const& devHost = cms::alpakatools::host();
  std::tie(radius, std::ignore, std::ignore) = computeRadiusFromThreeAnchorHits(devHost,
                                                                                hitObjects[0].x(),
                                                                                hitObjects[0].y(),
                                                                                hitObjects[1].x(),
                                                                                hitObjects[1].y(),
                                                                                hitObjects[2].x(),
                                                                                hitObjects[2].y());
  ana.tx->pushbackToBranch<float>("t5_t3_pt", k2Rinv1GeVf * 2 * radius);

  // Angles
  ana.tx->pushbackToBranch<float>("t5_t3_eta", hitObjects[2].eta());
  ana.tx->pushbackToBranch<float>("t5_t3_phi", hitObjects[0].phi());
}

//________________________________________________________________________________________________________________________________
void setpT3DNNBranches(LSTEvent* event) {
  auto pixelTriplets = event->getPixelTriplets();
  unsigned int nPT3 = pixelTriplets.nPixelTriplets();
  for (unsigned int iPT3 = 0; iPT3 < nPT3; ++iPT3) {
    fillpT3DNNBranches(event, iPT3);
  }
}

//________________________________________________________________________________________________________________________________
void setT3DNNBranches(LSTEvent* event) {
  auto const triplets = event->getTriplets<TripletsSoA>();
  auto const tripletsOccupancy = event->getTriplets<TripletsOccupancySoA>();
  auto modules = event->getModules<ModulesSoA>();
  auto ranges = event->getRanges();

  auto const& trk_sim_parentVtxIdx = trk.getVI("sim_parentVtxIdx");
  auto const& trk_simvtx_x = trk.getVF("simvtx_x");
  auto const& trk_simvtx_y = trk.getVF("simvtx_y");
  auto const& trk_simvtx_z = trk.getVF("simvtx_z");
  auto const& trk_simhit_simTrkIdx = trk.getVI("simhit_simTrkIdx");
  auto const& trk_ph2_simHitIdx = trk.getVVI("ph2_simHitIdx");
  auto const& trk_pix_simHitIdx = trk.getVVI("pix_simHitIdx");

  for (unsigned int lowerModuleIdx = 0; lowerModuleIdx < modules.nLowerModules(); ++lowerModuleIdx) {
    int nTriplets = tripletsOccupancy.nTriplets()[lowerModuleIdx];
    for (unsigned int idx = 0; idx < nTriplets; idx++) {
      unsigned int tripletIndex = ranges.tripletModuleIndices()[lowerModuleIdx] + idx;

      // Get hit indices and types
      std::vector<unsigned int> hit_idx = getHitsFromT3(event, tripletIndex);
      std::vector<unsigned int> hit_type = getHitTypesFromT3(event, tripletIndex);
      std::vector<unsigned int> module_idx = getModuleIdxsFromT3(event, tripletIndex);

      // Calculate layer binary representation
      int layer_binary = 0;
      for (size_t i = 0; i < module_idx.size(); i += 2) {
        layer_binary |= (1 << (modules.layers()[module_idx[i]] + 6 * (modules.subdets()[module_idx[i]] == 4)));
      }

      // Get matching information with percent matched
      float percent_matched;
      std::vector<int> simidx = matchedSimTrkIdxs(
          hit_idx, hit_type, trk_simhit_simTrkIdx, trk_ph2_simHitIdx, trk_pix_simHitIdx, false, &percent_matched);

      // Fill the branches with T3-specific data
      ana.tx->pushbackToBranch<float>("t3_betaIn", triplets.betaIn()[tripletIndex]);
      ana.tx->pushbackToBranch<float>("t3_centerX", triplets.centerX()[tripletIndex]);
      ana.tx->pushbackToBranch<float>("t3_centerY", triplets.centerY()[tripletIndex]);
      ana.tx->pushbackToBranch<float>("t3_radius", triplets.radius()[tripletIndex]);
      ana.tx->pushbackToBranch<bool>("t3_partOfPT5", triplets.partOfPT5()[tripletIndex]);
      ana.tx->pushbackToBranch<bool>("t3_partOfT5", triplets.partOfT5()[tripletIndex]);
      ana.tx->pushbackToBranch<bool>("t3_partOfPT3", triplets.partOfPT3()[tripletIndex]);
      ana.tx->pushbackToBranch<int>("t3_layer_binary", layer_binary);
      ana.tx->pushbackToBranch<std::vector<int>>("t3_matched_simIdx", simidx);
      ana.tx->pushbackToBranch<float>("t3_pMatched", percent_matched);

      // Add vertex information for matched sim tracks
      if (simidx.size() == 0) {
        // No matched sim track - set default values
        ana.tx->pushbackToBranch<float>("t3_sim_vxy", 0.0);
        ana.tx->pushbackToBranch<float>("t3_sim_vz", 0.0);
      } else {
        // Get vertex information from the first matched sim track
        int vtxidx = trk_sim_parentVtxIdx[simidx[0]];
        float vtx_x = trk_simvtx_x[vtxidx];
        float vtx_y = trk_simvtx_y[vtxidx];
        float vtx_z = trk_simvtx_z[vtxidx];

        // Calculate transverse distance from origin
        float vxy = sqrt(vtx_x * vtx_x + vtx_y * vtx_y);

        ana.tx->pushbackToBranch<float>("t3_sim_vxy", vxy);
        ana.tx->pushbackToBranch<float>("t3_sim_vz", vtx_z);
      }

      // Fill hit-specific information
      fillT3DNNBranches(event, tripletIndex);
    }
  }
}

//________________________________________________________________________________________________________________________________
void setT5DNNBranches(LSTEvent* event) {
  auto tripletsOcc = event->getTriplets<TripletsOccupancySoA>();
  auto tripletsSoA = event->getTriplets<TripletsSoA>();
  auto modules = event->getModules<ModulesSoA>();
  auto ranges = event->getRanges();
  auto const quintuplets = event->getQuintuplets<QuintupletsOccupancySoA>();
  auto trackCandidates = event->getTrackCandidates();

  std::unordered_set<unsigned int> allT3s;
  std::unordered_map<unsigned int, unsigned int> t3_index_map;

  for (unsigned int idx = 0; idx < modules.nLowerModules(); ++idx) {
    for (unsigned int jdx = 0; jdx < tripletsOcc.nTriplets()[idx]; ++jdx) {
      unsigned int t3Idx = ranges.tripletModuleIndices()[idx] + jdx;
      if (allT3s.insert(t3Idx).second) {
        t3_index_map[t3Idx] = allT3s.size() - 1;
        fillT5DNNBranches(event, t3Idx);
      }
    }
  }

  std::unordered_map<unsigned int, unsigned int> t5_tc_index_map;
  std::unordered_set<unsigned int> t5s_used_in_tc;

  for (unsigned int idx = 0; idx < trackCandidates.nTrackCandidates(); idx++) {
    if (trackCandidates.trackCandidateType()[idx] == LSTObjType::T5) {
      unsigned int objIdx = trackCandidates.directObjectIndices()[idx];
      t5s_used_in_tc.insert(objIdx);
      t5_tc_index_map[objIdx] = idx;
    }
  }

  for (unsigned int idx = 0; idx < modules.nLowerModules(); ++idx) {
    for (unsigned int jdx = 0; jdx < quintuplets.nQuintuplets()[idx]; ++jdx) {
      unsigned int t5Idx = ranges.quintupletModuleIndices()[idx] + jdx;
      std::vector<unsigned int> t3sIdx = getT3sFromT5(event, t5Idx);

      ana.tx->pushbackToBranch<int>("t5_t3_idx0", t3_index_map[t3sIdx[0]]);
      ana.tx->pushbackToBranch<int>("t5_t3_idx1", t3_index_map[t3sIdx[1]]);

      ana.tx->pushbackToBranch<float>("t5_t3_fakeScore1", tripletsSoA.fakeScore()[t3sIdx[0]]);
      ana.tx->pushbackToBranch<float>("t5_t3_promptScore1", tripletsSoA.promptScore()[t3sIdx[0]]);
      ana.tx->pushbackToBranch<float>("t5_t3_displacedScore1", tripletsSoA.displacedScore()[t3sIdx[0]]);
      ana.tx->pushbackToBranch<float>("t5_t3_fakeScore2", tripletsSoA.fakeScore()[t3sIdx[1]]);
      ana.tx->pushbackToBranch<float>("t5_t3_promptScore2", tripletsSoA.promptScore()[t3sIdx[1]]);
      ana.tx->pushbackToBranch<float>("t5_t3_displacedScore2", tripletsSoA.displacedScore()[t3sIdx[1]]);

      if (t5s_used_in_tc.find(t5Idx) != t5s_used_in_tc.end()) {
        ana.tx->pushbackToBranch<int>("t5_partOfTC", 1);
        ana.tx->pushbackToBranch<int>("t5_tc_idx", t5_tc_index_map[t5Idx]);
      } else {
        ana.tx->pushbackToBranch<int>("t5_partOfTC", 0);
        ana.tx->pushbackToBranch<int>("t5_tc_idx", -999);
      }
    }
  }
}

//________________________________________________________________________________________________________________________________
void setGnnNtupleBranches(LSTEvent* event) {
  // Get relevant information
  SegmentsOccupancyConst segmentsOccupancy = event->getSegments<SegmentsOccupancySoA>();
  MiniDoubletsOccupancyConst miniDoublets = event->getMiniDoublets<MiniDoubletsOccupancySoA>();
  auto hitsBase = event->getInput<HitsBaseSoA>();
  auto modules = event->getModules<ModulesSoA>();
  auto ranges = event->getRanges();
  auto const& trackCandidates = event->getTrackCandidates();

  std::set<unsigned int> mds_used_in_sg;
  std::map<unsigned int, unsigned int> md_index_map;
  std::map<unsigned int, unsigned int> sg_index_map;

  // Loop over modules (lower ones where the MDs are saved)
  unsigned int nTotalMD = 0;
  unsigned int nTotalLS = 0;
  for (unsigned int idx = 0; idx < modules.nLowerModules(); ++idx) {
    nTotalMD += miniDoublets.nMDs()[idx];
    nTotalLS += segmentsOccupancy.nSegments()[idx];
  }

  auto const& trk_simhit_simTrkIdx = trk.getVI("simhit_simTrkIdx");
  auto const& trk_ph2_simHitIdx = trk.getVVI("ph2_simHitIdx");
  auto const& trk_pix_simHitIdx = trk.getVVI("pix_simHitIdx");

  std::set<unsigned int> lss_used_in_true_tc;
  unsigned int nTrackCandidates = trackCandidates.nTrackCandidates();
  for (unsigned int idx = 0; idx < nTrackCandidates; idx++) {
    // Only consider true track candidates
    std::vector<unsigned int> hitidxs;
    std::vector<unsigned int> hittypes;
    std::tie(hitidxs, hittypes) = getHitIdxsAndHitTypesFromTC(event, idx);
    std::vector<int> simidxs =
        matchedSimTrkIdxs(hitidxs, hittypes, trk_simhit_simTrkIdx, trk_ph2_simHitIdx, trk_pix_simHitIdx);
    if (simidxs.size() == 0)
      continue;

    std::vector<unsigned int> LSs = getLSsFromTC(event, idx);
    for (auto& LS : LSs) {
      if (lss_used_in_true_tc.find(LS) == lss_used_in_true_tc.end()) {
        lss_used_in_true_tc.insert(LS);
      }
    }
  }

  std::cout << " lss_used_in_true_tc.size(): " << lss_used_in_true_tc.size() << std::endl;

  // std::cout <<  " nTotalMD: " << nTotalMD <<  std::endl;
  // std::cout <<  " nTotalLS: " << nTotalLS <<  std::endl;

  auto const& trk_sim_bunchCrossing = trk.getVI("sim_bunchCrossing");
  auto const& trk_sim_event = trk.getVI("sim_event");
  auto const& trk_sim_pt = trk.getVF("sim_pt");
  auto const& trk_sim_eta = trk.getVF("sim_eta");
  auto const& trk_sim_phi = trk.getVF("sim_phi");
  auto const& trk_sim_pca_dxy = trk.getVF("sim_pca_dxy");
  auto const& trk_sim_pca_dz = trk.getVF("sim_pca_dz");
  auto const& trk_sim_q = trk.getVI("sim_q");
  auto const& trk_sim_pdgId = trk.getVI("sim_pdgId");
  auto const& trk_sim_parentVtxIdx = trk.getVI("sim_parentVtxIdx");
  auto const& trk_simvtx_x = trk.getVF("simvtx_x");
  auto const& trk_simvtx_y = trk.getVF("simvtx_y");
  auto const& trk_simvtx_z = trk.getVF("simvtx_z");

  // Loop over modules (lower ones where the MDs are saved)
  for (unsigned int idx = 0; idx < modules.nLowerModules(); ++idx) {
    // // Loop over minidoublets
    // for (unsigned int jdx = 0; jdx < miniDoublets->nMDs[idx]; jdx++)
    // {
    //     // Get the actual index to the mini-doublet using ranges
    //     unsigned int mdIdx = ranges->miniDoubletModuleIndices[idx] + jdx;

    //     setGnnNtupleMiniDoublet(event, mdIdx);
    // }

    // Loop over segments
    for (unsigned int jdx = 0; jdx < segmentsOccupancy.nSegments()[idx]; jdx++) {
      // Get the actual index to the segments using ranges
      unsigned int sgIdx = ranges.segmentModuleIndices()[idx] + jdx;

      // Get the hit indices
      std::vector<unsigned int> MDs = getMDsFromLS(event, sgIdx);

      if (mds_used_in_sg.find(MDs[0]) == mds_used_in_sg.end()) {
        mds_used_in_sg.insert(MDs[0]);
        md_index_map[MDs[0]] = mds_used_in_sg.size() - 1;
        setGnnNtupleMiniDoublet(event,
                                MDs[0],
                                trk_sim_q,
                                trk_sim_pt,
                                trk_sim_eta,
                                trk_sim_bunchCrossing,
                                trk_sim_event,
                                trk_sim_parentVtxIdx,
                                trk_simvtx_x,
                                trk_simvtx_y,
                                trk_simvtx_z,
                                trk_simhit_simTrkIdx,
                                trk_ph2_simHitIdx,
                                trk_pix_simHitIdx);
      }

      if (mds_used_in_sg.find(MDs[1]) == mds_used_in_sg.end()) {
        mds_used_in_sg.insert(MDs[1]);
        md_index_map[MDs[1]] = mds_used_in_sg.size() - 1;
        setGnnNtupleMiniDoublet(event,
                                MDs[1],
                                trk_sim_q,
                                trk_sim_pt,
                                trk_sim_eta,
                                trk_sim_bunchCrossing,
                                trk_sim_event,
                                trk_sim_parentVtxIdx,
                                trk_simvtx_x,
                                trk_simvtx_y,
                                trk_simvtx_z,
                                trk_simhit_simTrkIdx,
                                trk_ph2_simHitIdx,
                                trk_pix_simHitIdx);
      }

      ana.tx->pushbackToBranch<int>("LS_MD_idx0", md_index_map[MDs[0]]);
      ana.tx->pushbackToBranch<int>("LS_MD_idx1", md_index_map[MDs[1]]);

      std::vector<unsigned int> hits = getHitsFromLS(event, sgIdx);

      // Computing line segment pt estimate (assuming beam spot is at zero)
      lst_math::Hit hitA(0, 0, 0);
      lst_math::Hit hitB(hitsBase.xs()[hits[0]], hitsBase.ys()[hits[0]], hitsBase.zs()[hits[0]]);
      lst_math::Hit hitC(hitsBase.xs()[hits[2]], hitsBase.ys()[hits[2]], hitsBase.zs()[hits[2]]);
      lst_math::Hit center = lst_math::getCenterFromThreePoints(hitA, hitB, hitC);
      float pt = lst_math::ptEstimateFromRadius(center.rt());
      float eta = hitC.eta();
      float phi = hitB.phi();

      ana.tx->pushbackToBranch<float>("LS_pt", pt);
      ana.tx->pushbackToBranch<float>("LS_eta", eta);
      ana.tx->pushbackToBranch<float>("LS_phi", phi);
      // ana.tx->pushbackToBranch<int>("LS_layer0", layer0);
      // ana.tx->pushbackToBranch<int>("LS_layer1", layer1);

      std::vector<unsigned int> hitidxs;
      std::vector<unsigned int> hittypes;
      std::tie(hitidxs, hittypes) = getHitIdxsAndHitTypesFromLS(event, sgIdx);
      std::vector<int> simidxs =
          matchedSimTrkIdxs(hitidxs, hittypes, trk_simhit_simTrkIdx, trk_ph2_simHitIdx, trk_pix_simHitIdx);

      ana.tx->pushbackToBranch<int>("LS_isFake", simidxs.size() == 0);
      ana.tx->pushbackToBranch<float>("LS_sim_pt", simidxs.size() > 0 ? trk_sim_pt[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<float>("LS_sim_eta", simidxs.size() > 0 ? trk_sim_eta[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<float>("LS_sim_phi", simidxs.size() > 0 ? trk_sim_phi[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<float>("LS_sim_pca_dxy", simidxs.size() > 0 ? trk_sim_pca_dxy[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<float>("LS_sim_pca_dz", simidxs.size() > 0 ? trk_sim_pca_dz[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<int>("LS_sim_q", simidxs.size() > 0 ? trk_sim_q[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<int>("LS_sim_event", simidxs.size() > 0 ? trk_sim_event[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<int>("LS_sim_bx", simidxs.size() > 0 ? trk_sim_bunchCrossing[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<int>("LS_sim_pdgId", simidxs.size() > 0 ? trk_sim_pdgId[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<float>("LS_sim_vx",
                                      simidxs.size() > 0 ? trk_simvtx_x[trk_sim_parentVtxIdx[simidxs[0]]] : -999);
      ana.tx->pushbackToBranch<float>("LS_sim_vy",
                                      simidxs.size() > 0 ? trk_simvtx_y[trk_sim_parentVtxIdx[simidxs[0]]] : -999);
      ana.tx->pushbackToBranch<float>("LS_sim_vz",
                                      simidxs.size() > 0 ? trk_simvtx_z[trk_sim_parentVtxIdx[simidxs[0]]] : -999);
      ana.tx->pushbackToBranch<int>("LS_isInTrueTC", lss_used_in_true_tc.find(sgIdx) != lss_used_in_true_tc.end());

      sg_index_map[sgIdx] = ana.tx->getBranch<std::vector<int>>("LS_isFake").size() - 1;

      // // T5 eta and phi are computed using outer and innermost hits
      // lst_math::Hit hitA(trk_ph2_x[anchitidx], trk_ph2_y[anchitidx], trk_ph2_z[anchitidx]);
      // const float phi = hitA.phi();
      // const float eta = hitA.eta();
    }
  }

  for (unsigned int idx = 0; idx < nTrackCandidates; idx++) {
    std::vector<unsigned int> LSs = getLSsFromTC(event, idx);
    std::vector<int> lsIdx;
    for (auto& LS : LSs) {
      lsIdx.push_back(sg_index_map[LS]);
    }
    ana.tx->pushbackToBranch<std::vector<int>>("tc_lsIdx", lsIdx);
  }

  std::cout << " mds_used_in_sg.size(): " << mds_used_in_sg.size() << std::endl;
}

//________________________________________________________________________________________________________________________________
void setGnnNtupleMiniDoublet(LSTEvent* event,
                             unsigned int MD,
                             std::vector<int> const& trk_sim_q,
                             std::vector<float> const& trk_sim_pt,
                             std::vector<float> const& trk_sim_eta,
                             std::vector<int> const& trk_sim_bunchCrossing,
                             std::vector<int> const& trk_sim_event,
                             std::vector<int> const& trk_sim_parentVtxIdx,
                             std::vector<float> const& trk_simvtx_x,
                             std::vector<float> const& trk_simvtx_y,
                             std::vector<float> const& trk_simvtx_z,
                             std::vector<int> const& trk_simhit_simTrkIdx,
                             std::vector<std::vector<int>> const& trk_ph2_simHitIdx,
                             std::vector<std::vector<int>> const& trk_pix_simHitIdx) {
  // Get relevant information
  MiniDoubletsConst miniDoublets = event->getMiniDoublets<MiniDoubletsSoA>();
  auto hitsBase = event->getInput<HitsBaseSoA>();

  // Get the hit indices
  unsigned int hit0 = miniDoublets.anchorHitIndices()[MD];
  unsigned int hit1 = miniDoublets.outerHitIndices()[MD];

  // Get the hit infos
  const float hit0_x = hitsBase.xs()[hit0];
  const float hit0_y = hitsBase.ys()[hit0];
  const float hit0_z = hitsBase.zs()[hit0];
  const float hit0_r = sqrt(hit0_x * hit0_x + hit0_y * hit0_y);
  const float hit1_x = hitsBase.xs()[hit1];
  const float hit1_y = hitsBase.ys()[hit1];
  const float hit1_z = hitsBase.zs()[hit1];
  const float hit1_r = sqrt(hit1_x * hit1_x + hit1_y * hit1_y);

  // Do sim matching
  std::vector<unsigned int> hit_idx = {hitsBase.idxs()[hit0], hitsBase.idxs()[hit1]};
  std::vector<unsigned int> hit_type = {4, 4};
  std::vector<int> simidxs =
      matchedSimTrkIdxs(hit_idx, hit_type, trk_simhit_simTrkIdx, trk_ph2_simHitIdx, trk_pix_simHitIdx);

  bool isFake = simidxs.size() == 0;
  int tp_type = getDenomSimTrkType(simidxs,
                                   trk_sim_q,
                                   trk_sim_pt,
                                   trk_sim_eta,
                                   trk_sim_bunchCrossing,
                                   trk_sim_event,
                                   trk_sim_parentVtxIdx,
                                   trk_simvtx_x,
                                   trk_simvtx_y,
                                   trk_simvtx_z);

  auto const& trk_ph2_subdet = trk.getVUS("ph2_subdet");
  auto const& trk_ph2_layer = trk.getVUS("ph2_layer");
  auto const& trk_ph2_detId = trk.getVU("ph2_detId");
  auto const& trk_ph2_x = trk.getVF("ph2_x");
  auto const& trk_ph2_y = trk.getVF("ph2_y");
  auto const& trk_ph2_z = trk.getVF("ph2_z");

  // Obtain where the actual hit is located in terms of their layer, module, rod, and ring number
  unsigned int anchitidx = hitsBase.idxs()[hit0];
  int subdet = trk_ph2_subdet[hitsBase.idxs()[anchitidx]];
  int is_endcap = subdet == 4;
  // this accounting makes it so that you have layer 1 2 3 4 5 6 in the barrel, and 7 8 9 10 11 in the endcap. (becuase endcap is ph2_subdet == 4)
  int layer = trk_ph2_layer[anchitidx] + 6 * (is_endcap);
  int detId = trk_ph2_detId[anchitidx];

  // Obtaining dPhiChange
  float dphichange = miniDoublets.dphichanges()[MD];

  // Computing pt
  float pt = hit0_r * k2Rinv1GeVf / sin(dphichange);

  // T5 eta and phi are computed using outer and innermost hits
  lst_math::Hit hitA(trk_ph2_x[anchitidx], trk_ph2_y[anchitidx], trk_ph2_z[anchitidx]);
  const float phi = hitA.phi();
  const float eta = hitA.eta();

  // Mini Doublets
  ana.tx->pushbackToBranch<float>("MD_pt", pt);
  ana.tx->pushbackToBranch<float>("MD_eta", eta);
  ana.tx->pushbackToBranch<float>("MD_phi", phi);
  ana.tx->pushbackToBranch<float>("MD_dphichange", dphichange);
  ana.tx->pushbackToBranch<int>("MD_isFake", isFake);
  ana.tx->pushbackToBranch<int>("MD_tpType", tp_type);
  ana.tx->pushbackToBranch<int>("MD_detId", detId);
  ana.tx->pushbackToBranch<int>("MD_layer", layer);
  ana.tx->pushbackToBranch<float>("MD_0_r", hit0_r);
  ana.tx->pushbackToBranch<float>("MD_0_x", hit0_x);
  ana.tx->pushbackToBranch<float>("MD_0_y", hit0_y);
  ana.tx->pushbackToBranch<float>("MD_0_z", hit0_z);
  ana.tx->pushbackToBranch<float>("MD_1_r", hit1_r);
  ana.tx->pushbackToBranch<float>("MD_1_x", hit1_x);
  ana.tx->pushbackToBranch<float>("MD_1_y", hit1_y);
  ana.tx->pushbackToBranch<float>("MD_1_z", hit1_z);
  // ana.tx->pushbackToBranch<int>("MD_sim_idx", simidxs.size() > 0 ? simidxs[0] : -999);
}

//________________________________________________________________________________________________________________________________
std::tuple<int, float, float, float, int, std::vector<int>> parseTrackCandidate(
    LSTEvent* event,
    unsigned int idx,
    std::vector<float> const& trk_ph2_x,
    std::vector<float> const& trk_ph2_y,
    std::vector<float> const& trk_ph2_z,
    std::vector<int> const& trk_simhit_simTrkIdx,
    std::vector<std::vector<int>> const& trk_ph2_simHitIdx,
    std::vector<std::vector<int>> const& trk_pix_simHitIdx,
    float matchfrac) {
  // Get the type of the track candidate
  auto const& trackCandidates = event->getTrackCandidates();
  short type = trackCandidates.trackCandidateType()[idx];

  // Compute pt eta phi and hit indices that will be used to figure out whether the TC matched
  float pt, eta, phi;
  std::vector<unsigned int> hit_idx, hit_type;
  switch (type) {
    case LSTObjType::pT5:
      std::tie(pt, eta, phi, hit_idx, hit_type) = parsepT5(event, idx);
      break;
    case LSTObjType::pT3:
      std::tie(pt, eta, phi, hit_idx, hit_type) = parsepT3(event, idx);
      break;
    case LSTObjType::T5:
      std::tie(pt, eta, phi, hit_idx, hit_type) = parseT5(event, idx, trk_ph2_x, trk_ph2_y, trk_ph2_z);
      break;
    case LSTObjType::pLS:
      std::tie(pt, eta, phi, hit_idx, hit_type) = parsepLS(event, idx);
      break;
  }

  // Perform matching
  std::vector<int> simidx =
      matchedSimTrkIdxs(hit_idx, hit_type, trk_simhit_simTrkIdx, trk_ph2_simHitIdx, trk_pix_simHitIdx, false, matchfrac);
  int isFake = simidx.size() == 0;

  return {type, pt, eta, phi, isFake, simidx};
}

//________________________________________________________________________________________________________________________________
std::tuple<int, float, float, float, int, std::vector<int>, std::vector<float>> parseTrackCandidateAllMatch(
    LSTEvent* event,
    unsigned int idx,
    std::vector<float> const& trk_ph2_x,
    std::vector<float> const& trk_ph2_y,
    std::vector<float> const& trk_ph2_z,
    std::vector<int> const& trk_simhit_simTrkIdx,
    std::vector<std::vector<int>> const& trk_ph2_simHitIdx,
    std::vector<std::vector<int>> const& trk_pix_simHitIdx,
    float matchfrac) {
  // Get the type of the track candidate
  auto const& trackCandidates = event->getTrackCandidates();
  short type = trackCandidates.trackCandidateType()[idx];

  // Compute pt eta phi and hit indices that will be used to figure out whether the TC matched
  float pt, eta, phi;
  std::vector<unsigned int> hit_idx, hit_type;
  switch (type) {
    case LSTObjType::pT5:
      std::tie(pt, eta, phi, hit_idx, hit_type) = parsepT5(event, idx);
      break;
    case LSTObjType::pT3:
      std::tie(pt, eta, phi, hit_idx, hit_type) = parsepT3(event, idx);
      break;
    case LSTObjType::T5:
      std::tie(pt, eta, phi, hit_idx, hit_type) = parseT5(event, idx, trk_ph2_x, trk_ph2_y, trk_ph2_z);
      break;
    case LSTObjType::pLS:
      std::tie(pt, eta, phi, hit_idx, hit_type) = parsepLS(event, idx);
      break;
  }

  // Perform matching
  std::vector<int> simidx;
  std::vector<float> simidxfrac;
  std::tie(simidx, simidxfrac) =
      matchedSimTrkIdxsAndFracs(hit_idx, hit_type, trk_simhit_simTrkIdx, trk_ph2_simHitIdx, trk_pix_simHitIdx, false, matchfrac);
  int isFake = simidx.size() == 0;

  return {type, pt, eta, phi, isFake, simidx, simidxfrac};
}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, std::vector<unsigned int>, std::vector<unsigned int>> parsepT5(LSTEvent* event,
                                                                                               unsigned int idx) {
  // Get relevant information
  auto const trackCandidates = event->getTrackCandidates();
  auto const quintuplets = event->getQuintuplets<QuintupletsSoA>();
  auto const pixelSeeds = event->getInput<PixelSeedsSoA>();

  //
  // pictorial representation of a pT5
  //
  // inner tracker        outer tracker
  // -------------  --------------------------
  // pLS            01    23    45    67    89   (anchor hit of a minidoublet is always the first of the pair)
  // ****           oo -- oo -- oo -- oo -- oo   pT5
  //                oo -- oo -- oo               first T3 of the T5
  //                            oo -- oo -- oo   second T3 of the T5
  unsigned int pT5 = trackCandidates.directObjectIndices()[idx];
  unsigned int pLS = getPixelLSFrompT5(event, pT5);
  unsigned int T5Index = getT5FrompT5(event, pT5);

  //=================================================================================
  // Some history and geometry lesson...
  // For a given T3, we compute two angles. (NOTE: This is a bit weird!)
  // Historically, T3 were created out of T4, which we used to build a long time ago.
  // So for the sake of argument let's discuss T4 first.
  // For a T4, we have 4 mini-doublets.
  // Therefore we have 4 "anchor hits".
  // Therefore we have 4 xyz points.
  //
  //
  //       *
  //       |\
    //       | \
    //       |1 \
    //       |   \
    //       |  * \
    //       |
  //       |
  //       |
  //       |
  //       |
  //       |  * /
  //       |   /
  //       |2 /
  //       | /
  //       |/
  //       *
  //
  //
  // Then from these 4 points, one can approximate a some sort of "best" fitted circle trajectory,
  // and obtain "tangential" angles from 1st and 4th hits.
  // See the carton below.
  // The "*" are the 4 physical hit points
  // angle 1 and 2 are the "tangential" angle for a "circle" from 4 * points.
  // Please note, that a straight line from first two * and the latter two * are NOT the
  // angle 1 and angle 2. (they were called "beta" angles)
  // But rather, a slightly larger angle.
  // Because 4 * points would be on a circle, and a tangential line on the circles
  // would deviate from the points on circles.
  //
  // In the early days of LST, there was an iterative algorithm (devised by Slava) to
  // obtain the angle beta1 and 2 _without_ actually performing a 4 point circle fit.
  // Hence, the beta1 and beta2 were quickly estimated without too many math operations
  // and afterwards (beta1-beta2) was computed to obtain what we call a "delta-beta" values.
  //
  // For a real track, the deltabeta ~ 0, for fakes, it'd have a flat distribution.
  //
  // However, after some time we abandonded the T4s, and moved to T3s.
  // In T3, however, now we have the following cartoon:
  //
  //       *
  //       |\
    //       | \
    //       |1 \
    //       |   \
    //       |  * X   (* here are "two" MDs but really just one)
  //       |   /
  //       |2 /
  //       | /
  //       |/
  //       *
  //
  // With the "four" *'s (really just "three") you can still run the iterative beta calculation,
  // which is what we still currently do, we still get two beta1 and beta2
  // But! high school geometry tells us that 3 points = ONLY 1 possible CIRCLE!
  // There is really nothing to "fit" here.
  // YET we still compute these in T3, out of legacy method of how we used to treat T4s.
  //
  // Hence, in the below code, "betaIn_in" and "betaOut_in" if we performed
  // a circle fit they would come out by definition identical values.
  // But due to our approximate iterative beta calculation method, they come out different values.
  // So if we are "cutting on" abs(deltaBeta) = abs(betaIn_in - betaOut_in) < threshold,
  // what does that even mean?
  //
  // Anyhow, as of now, we compute 2 beta's for T3s, and T5 has two T3s.
  // And from there we estimate the pt's and we compute pt_T5.

  // pixel pt
  const float pt_pLS = pixelSeeds.ptIn()[pLS];
  const float eta_pLS = pixelSeeds.eta()[pLS];
  const float phi_pLS = pixelSeeds.phi()[pLS];
  float pt_T5 = __H2F(quintuplets.innerRadius()[T5Index]) * 2 * k2Rinv1GeVf;
  const float pt = (pt_T5 + pt_pLS) / 2;

  // Form the hit idx/type std::vector
  std::vector<unsigned int> hit_idx = getHitIdxsFrompT5(event, pT5);
  std::vector<unsigned int> hit_type = getHitTypesFrompT5(event, pT5);

  return {pt, eta_pLS, phi_pLS, hit_idx, hit_type};
}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, std::vector<unsigned int>, std::vector<unsigned int>> parsepT3(LSTEvent* event,
                                                                                               unsigned int idx) {
  // Get relevant information
  auto const trackCandidates = event->getTrackCandidates();
  auto const triplets = event->getTriplets<TripletsSoA>();
  auto const pixelSeeds = event->getInput<PixelSeedsSoA>();

  //
  // pictorial representation of a pT3
  //
  // inner tracker        outer tracker
  // -------------  --------------------------
  // pLS            01    23    45               (anchor hit of a minidoublet is always the first of the pair)
  // ****           oo -- oo -- oo               pT3
  unsigned int pT3 = trackCandidates.directObjectIndices()[idx];
  unsigned int pLS = getPixelLSFrompT3(event, pT3);
  unsigned int T3 = getT3FrompT3(event, pT3);

  // pixel pt
  const float pt_pLS = pixelSeeds.ptIn()[pLS];
  const float eta_pLS = pixelSeeds.eta()[pLS];
  const float phi_pLS = pixelSeeds.phi()[pLS];
  float pt_T3 = triplets.radius()[T3] * 2 * k2Rinv1GeVf;

  // average pt
  const float pt = (pt_pLS + pt_T3) / 2;

  // Form the hit idx/type std::vector
  std::vector<unsigned int> hit_idx = getHitIdxsFrompT3(event, pT3);
  std::vector<unsigned int> hit_type = getHitTypesFrompT3(event, pT3);

  return {pt, eta_pLS, phi_pLS, hit_idx, hit_type};
}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, std::vector<unsigned int>, std::vector<unsigned int>> parseT5(
    LSTEvent* event,
    unsigned int idx,
    std::vector<float> const& trk_ph2_x,
    std::vector<float> const& trk_ph2_y,
    std::vector<float> const& trk_ph2_z) {
  auto const trackCandidates = event->getTrackCandidates();
  auto const quintuplets = event->getQuintuplets<QuintupletsSoA>();
  unsigned int T5 = trackCandidates.directObjectIndices()[idx];
  std::vector<unsigned int> hits = getHitsFromT5(event, T5);

  //
  // pictorial representation of a T5
  //
  // inner tracker        outer tracker
  // -------------  --------------------------
  //                01    23    45    67    89   (anchor hit of a minidoublet is always the first of the pair)
  //  (none)        oo -- oo -- oo -- oo -- oo   T5
  unsigned int Hit_0 = hits[0];
  unsigned int Hit_4 = hits[4];
  unsigned int Hit_8 = hits[8];

  // T5 radius is average of the inner and outer radius
  const float pt = __H2F(quintuplets.innerRadius()[T5]) * k2Rinv1GeVf * 2;

  // T5 eta and phi are computed using outer and innermost hits
  lst_math::Hit hitA(trk_ph2_x[Hit_0], trk_ph2_y[Hit_0], trk_ph2_z[Hit_0]);
  lst_math::Hit hitB(trk_ph2_x[Hit_8], trk_ph2_y[Hit_8], trk_ph2_z[Hit_8]);
  const float phi = hitA.phi();
  const float eta = hitB.eta();

  std::vector<unsigned int> hit_idx = getHitIdxsFromT5(event, T5);
  std::vector<unsigned int> hit_type = getHitTypesFromT5(event, T5);

  return {pt, eta, phi, hit_idx, hit_type};
}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, std::vector<unsigned int>, std::vector<unsigned int>> parsepLS(LSTEvent* event,
                                                                                               unsigned int idx) {
  auto const& trackCandidates = event->getTrackCandidates();
  auto pixelSeeds = event->getInput<PixelSeedsSoA>();

  // Getting pLS index
  unsigned int pLS = trackCandidates.directObjectIndices()[idx];

  // Getting pt eta and phi
  float pt = pixelSeeds.ptIn()[pLS];
  float eta = pixelSeeds.eta()[pLS];
  float phi = pixelSeeds.phi()[pLS];

  // Getting hit indices and types
  std::vector<unsigned int> hit_idx = getPixelHitIdxsFrompLS(event, pLS);
  std::vector<unsigned int> hit_type = getPixelHitTypesFrompLS(event, pLS);

  return {pt, eta, phi, hit_idx, hit_type};
}

void setpLSOutputBranches(LSTEvent* event) {
  auto const& pixelSegments = event->getPixelSegments();
  auto const& pixelSeeds = event->getInput<PixelSeedsSoA>();
  int n_accepted_simtrk = ana.tx->getBranch<std::vector<int>>("sim_TC_matched").size();

  auto const& trk_simhit_simTrkIdx = trk.getVI("simhit_simTrkIdx");
  auto const& trk_ph2_simHitIdx = trk.getVVI("ph2_simHitIdx");
  auto const& trk_pix_simHitIdx = trk.getVVI("pix_simHitIdx");

  unsigned int n_pLS = pixelSegments.metadata().size();
  std::vector<int> sim_pLS_matched(n_accepted_simtrk, 0);
  std::vector<std::vector<int>> pLS_matched_simIdx;

  for (unsigned int i_pLS = 0; i_pLS < n_pLS; ++i_pLS) {
    // Get pLS properties
    float pt = pixelSeeds.ptIn()[i_pLS];
    float px = pixelSeeds.px()[i_pLS];
    float py = pixelSeeds.py()[i_pLS];
    float pz = pixelSeeds.pz()[i_pLS];
    bool isQuad = static_cast<bool>(pixelSeeds.isQuad()[i_pLS]);
    float ptErr = pixelSeeds.ptErr()[i_pLS];
    float eta = pixelSeeds.eta()[i_pLS];
    float etaErr = pixelSeeds.etaErr()[i_pLS];
    float phi = pixelSeeds.phi()[i_pLS];
    float score = pixelSegments.score()[i_pLS];
    float centerX = pixelSegments.circleCenterX()[i_pLS];
    float centerY = pixelSegments.circleCenterY()[i_pLS];
    float radius = pixelSegments.circleRadius()[i_pLS];

    // Get hits from pLS
    std::vector<unsigned int> hit_idx = getPixelHitIdxsFrompLS(event, i_pLS);
    std::vector<unsigned int> hit_type = getPixelHitTypesFrompLS(event, i_pLS);

    // Match to sim tracks
    std::vector<int> simidx =
        matchedSimTrkIdxs(hit_idx, hit_type, trk_simhit_simTrkIdx, trk_ph2_simHitIdx, trk_pix_simHitIdx);
    bool isFake = simidx.empty();

    // Fill branches
    ana.tx->pushbackToBranch<float>("pLS_ptIn", pt);
    ana.tx->pushbackToBranch<float>("pLS_ptErr", ptErr);
    ana.tx->pushbackToBranch<float>("pLS_px", px);
    ana.tx->pushbackToBranch<float>("pLS_py", py);
    ana.tx->pushbackToBranch<float>("pLS_pz", pz);
    ana.tx->pushbackToBranch<float>("pLS_eta", eta);
    ana.tx->pushbackToBranch<float>("pLS_etaErr", etaErr);
    ana.tx->pushbackToBranch<float>("pLS_phi", phi);
    ana.tx->pushbackToBranch<float>("pLS_score", score);
    ana.tx->pushbackToBranch<float>("pLS_circleCenterX", centerX);
    ana.tx->pushbackToBranch<float>("pLS_circleCenterY", centerY);
    ana.tx->pushbackToBranch<float>("pLS_circleRadius", radius);
    ana.tx->pushbackToBranch<bool>("pLS_isQuad", isQuad);
    ana.tx->pushbackToBranch<int>("pLS_isFake", isFake);
    pLS_matched_simIdx.push_back(simidx);

    // Count matches
    for (auto& idx : simidx) {
      if (idx < n_accepted_simtrk) {
        sim_pLS_matched[idx]++;
      }
    }
  }

  std::vector<int> pLS_isDuplicate(pLS_matched_simIdx.size(), 0);
  for (size_t i = 0; i < pLS_matched_simIdx.size(); ++i) {
    for (int simidx : pLS_matched_simIdx[i]) {
      if (simidx < n_accepted_simtrk && sim_pLS_matched[simidx] > 1) {
        pLS_isDuplicate[i] = 1;
        break;
      }
    }
  }

  ana.tx->setBranch<std::vector<int>>("sim_pLS_matched", sim_pLS_matched);
  ana.tx->setBranch<std::vector<std::vector<int>>>("pLS_matched_simIdx", pLS_matched_simIdx);
  ana.tx->setBranch<std::vector<int>>("pLS_isDuplicate", pLS_isDuplicate);
}

//________________________________________________________________________________________________________________________________
void printHitMultiplicities(LSTEvent* event) {
  auto modules = event->getModules<ModulesSoA>();
  auto hitRanges = event->getHits<HitsRangesSoA>();

  int nHits = 0;
  for (unsigned int idx = 0; idx <= modules.nLowerModules();
       idx++)  // "<=" because cheating to include pixel track candidate lower module
  {
    nHits += hitRanges.hitRanges()[2 * idx][1] - hitRanges.hitRanges()[2 * idx][0] + 1;
    nHits += hitRanges.hitRanges()[2 * idx + 1][1] - hitRanges.hitRanges()[2 * idx + 1][0] + 1;
  }
  std::cout << " nHits: " << nHits << std::endl;
}

//________________________________________________________________________________________________________________________________
void printMiniDoubletMultiplicities(LSTEvent* event) {
  MiniDoubletsOccupancyConst miniDoublets = event->getMiniDoublets<MiniDoubletsOccupancySoA>();
  auto modules = event->getModules<ModulesSoA>();

  int nMiniDoublets = 0;
  int totOccupancyMiniDoublets = 0;
  for (unsigned int idx = 0; idx <= modules.nModules();
       idx++)  // "<=" because cheating to include pixel track candidate lower module
  {
    if (modules.isLower()[idx]) {
      nMiniDoublets += miniDoublets.nMDs()[idx];
      totOccupancyMiniDoublets += miniDoublets.totOccupancyMDs()[idx];
    }
  }
  std::cout << " nMiniDoublets: " << nMiniDoublets << std::endl;
  std::cout << " totOccupancyMiniDoublets (including trucated ones): " << totOccupancyMiniDoublets << std::endl;
}

//________________________________________________________________________________________________________________________________
void printAllObjects(LSTEvent* event) {
  printMDs(event);
  printLSs(event);
  printpLSs(event);
  printT3s(event);
}

//________________________________________________________________________________________________________________________________
void printMDs(LSTEvent* event) {
  MiniDoubletsConst miniDoublets = event->getMiniDoublets<MiniDoubletsSoA>();
  MiniDoubletsOccupancyConst miniDoubletsOccupancy = event->getMiniDoublets<MiniDoubletsOccupancySoA>();
  auto hitsBase = event->getInput<HitsBaseSoA>();
  auto modules = event->getModules<ModulesSoA>();
  auto ranges = event->getRanges();

  // Then obtain the lower module index
  for (unsigned int idx = 0; idx <= modules.nLowerModules(); ++idx) {
    for (unsigned int iMD = 0; iMD < miniDoubletsOccupancy.nMDs()[idx]; iMD++) {
      unsigned int mdIdx = ranges.miniDoubletModuleIndices()[idx] + iMD;
      unsigned int LowerHitIndex = miniDoublets.anchorHitIndices()[mdIdx];
      unsigned int UpperHitIndex = miniDoublets.outerHitIndices()[mdIdx];
      unsigned int hit0 = hitsBase.idxs()[LowerHitIndex];
      unsigned int hit1 = hitsBase.idxs()[UpperHitIndex];
      std::cout << "VALIDATION 'MD': "
                << "MD"
                << " hit0: " << hit0 << " hit1: " << hit1 << std::endl;
    }
  }
}

//________________________________________________________________________________________________________________________________
void printLSs(LSTEvent* event) {
  SegmentsConst segments = event->getSegments<SegmentsSoA>();
  SegmentsOccupancyConst segmentsOccupancy = event->getSegments<SegmentsOccupancySoA>();
  MiniDoubletsConst miniDoublets = event->getMiniDoublets<MiniDoubletsSoA>();
  auto hitsBase = event->getInput<HitsBaseSoA>();
  auto modules = event->getModules<ModulesSoA>();
  auto ranges = event->getRanges();

  int nSegments = 0;
  for (unsigned int i = 0; i < modules.nLowerModules(); ++i) {
    unsigned int idx = i;  //modules->lowerModuleIndices[i];
    nSegments += segmentsOccupancy.nSegments()[idx];
    for (unsigned int jdx = 0; jdx < segmentsOccupancy.nSegments()[idx]; jdx++) {
      unsigned int sgIdx = ranges.segmentModuleIndices()[idx] + jdx;
      unsigned int InnerMiniDoubletIndex = segments.mdIndices()[sgIdx][0];
      unsigned int OuterMiniDoubletIndex = segments.mdIndices()[sgIdx][1];
      unsigned int InnerMiniDoubletLowerHitIndex = miniDoublets.anchorHitIndices()[InnerMiniDoubletIndex];
      unsigned int InnerMiniDoubletUpperHitIndex = miniDoublets.outerHitIndices()[InnerMiniDoubletIndex];
      unsigned int OuterMiniDoubletLowerHitIndex = miniDoublets.anchorHitIndices()[OuterMiniDoubletIndex];
      unsigned int OuterMiniDoubletUpperHitIndex = miniDoublets.outerHitIndices()[OuterMiniDoubletIndex];
      unsigned int hit0 = hitsBase.idxs()[InnerMiniDoubletLowerHitIndex];
      unsigned int hit1 = hitsBase.idxs()[InnerMiniDoubletUpperHitIndex];
      unsigned int hit2 = hitsBase.idxs()[OuterMiniDoubletLowerHitIndex];
      unsigned int hit3 = hitsBase.idxs()[OuterMiniDoubletUpperHitIndex];
      std::cout << "VALIDATION 'LS': "
                << "LS"
                << " hit0: " << hit0 << " hit1: " << hit1 << " hit2: " << hit2 << " hit3: " << hit3 << std::endl;
    }
  }
  std::cout << "VALIDATION nSegments: " << nSegments << std::endl;
}

//________________________________________________________________________________________________________________________________
void printpLSs(LSTEvent* event) {
  SegmentsConst segments = event->getSegments<SegmentsSoA>();
  SegmentsOccupancyConst segmentsOccupancy = event->getSegments<SegmentsOccupancySoA>();
  MiniDoubletsConst miniDoublets = event->getMiniDoublets<MiniDoubletsSoA>();
  auto hitsBase = event->getInput<HitsBaseSoA>();
  auto modules = event->getModules<ModulesSoA>();
  auto ranges = event->getRanges();

  unsigned int i = modules.nLowerModules();
  unsigned int idx = i;  //modules->lowerModuleIndices[i];
  int npLS = segmentsOccupancy.nSegments()[idx];
  for (unsigned int jdx = 0; jdx < segmentsOccupancy.nSegments()[idx]; jdx++) {
    unsigned int sgIdx = ranges.segmentModuleIndices()[idx] + jdx;
    unsigned int InnerMiniDoubletIndex = segments.mdIndices()[sgIdx][0];
    unsigned int OuterMiniDoubletIndex = segments.mdIndices()[sgIdx][1];
    unsigned int InnerMiniDoubletLowerHitIndex = miniDoublets.anchorHitIndices()[InnerMiniDoubletIndex];
    unsigned int InnerMiniDoubletUpperHitIndex = miniDoublets.outerHitIndices()[InnerMiniDoubletIndex];
    unsigned int OuterMiniDoubletLowerHitIndex = miniDoublets.anchorHitIndices()[OuterMiniDoubletIndex];
    unsigned int OuterMiniDoubletUpperHitIndex = miniDoublets.outerHitIndices()[OuterMiniDoubletIndex];
    unsigned int hit0 = hitsBase.idxs()[InnerMiniDoubletLowerHitIndex];
    unsigned int hit1 = hitsBase.idxs()[InnerMiniDoubletUpperHitIndex];
    unsigned int hit2 = hitsBase.idxs()[OuterMiniDoubletLowerHitIndex];
    unsigned int hit3 = hitsBase.idxs()[OuterMiniDoubletUpperHitIndex];
    std::cout << "VALIDATION 'pLS': "
              << "pLS"
              << " hit0: " << hit0 << " hit1: " << hit1 << " hit2: " << hit2 << " hit3: " << hit3 << std::endl;
  }
  std::cout << "VALIDATION npLS: " << npLS << std::endl;
}

//________________________________________________________________________________________________________________________________
void printT3s(LSTEvent* event) {
  auto const triplets = event->getTriplets<TripletsSoA>();
  auto const tripletsOccupancy = event->getTriplets<TripletsOccupancySoA>();
  SegmentsConst segments = event->getSegments<SegmentsSoA>();
  MiniDoubletsConst miniDoublets = event->getMiniDoublets<MiniDoubletsSoA>();
  auto hitsBase = event->getInput<HitsBaseSoA>();
  auto modules = event->getModules<ModulesSoA>();
  int nTriplets = 0;
  for (unsigned int i = 0; i < modules.nLowerModules(); ++i) {
    // unsigned int idx = modules->lowerModuleIndices[i];
    nTriplets += tripletsOccupancy.nTriplets()[i];
    unsigned int idx = i;
    for (unsigned int jdx = 0; jdx < tripletsOccupancy.nTriplets()[idx]; jdx++) {
      unsigned int tpIdx = idx * 5000 + jdx;
      unsigned int InnerSegmentIndex = triplets.segmentIndices()[tpIdx][0];
      unsigned int OuterSegmentIndex = triplets.segmentIndices()[tpIdx][1];
      unsigned int InnerSegmentInnerMiniDoubletIndex = segments.mdIndices()[InnerSegmentIndex][0];
      unsigned int InnerSegmentOuterMiniDoubletIndex = segments.mdIndices()[InnerSegmentIndex][1];
      unsigned int OuterSegmentOuterMiniDoubletIndex = segments.mdIndices()[OuterSegmentIndex][1];

      unsigned int hit_idx0 = miniDoublets.anchorHitIndices()[InnerSegmentInnerMiniDoubletIndex];
      unsigned int hit_idx1 = miniDoublets.outerHitIndices()[InnerSegmentInnerMiniDoubletIndex];
      unsigned int hit_idx2 = miniDoublets.anchorHitIndices()[InnerSegmentOuterMiniDoubletIndex];
      unsigned int hit_idx3 = miniDoublets.outerHitIndices()[InnerSegmentOuterMiniDoubletIndex];
      unsigned int hit_idx4 = miniDoublets.anchorHitIndices()[OuterSegmentOuterMiniDoubletIndex];
      unsigned int hit_idx5 = miniDoublets.outerHitIndices()[OuterSegmentOuterMiniDoubletIndex];

      unsigned int hit0 = hitsBase.idxs()[hit_idx0];
      unsigned int hit1 = hitsBase.idxs()[hit_idx1];
      unsigned int hit2 = hitsBase.idxs()[hit_idx2];
      unsigned int hit3 = hitsBase.idxs()[hit_idx3];
      unsigned int hit4 = hitsBase.idxs()[hit_idx4];
      unsigned int hit5 = hitsBase.idxs()[hit_idx5];
      std::cout << "VALIDATION 'T3': "
                << "T3"
                << " hit0: " << hit0 << " hit1: " << hit1 << " hit2: " << hit2 << " hit3: " << hit3 << " hit4: " << hit4
                << " hit5: " << hit5 << std::endl;
    }
  }
  std::cout << "VALIDATION nTriplets: " << nTriplets << std::endl;
}

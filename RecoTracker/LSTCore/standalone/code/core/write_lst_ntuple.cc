#include "write_lst_ntuple.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

//________________________________________________________________________________________________________________________________
void createOutputBranches() {
  createRequiredOutputBranches();
  createOptionalOutputBranches();
}

//________________________________________________________________________________________________________________________________
void fillOutputBranches(lst::Event<Acc3D>* event) {
  setOutputBranches(event);
  setOptionalOutputBranches(event);
  if (ana.gnn_ntuple)
    setGnnNtupleBranches(event);

  // Now actually fill the ttree
  ana.tx->fill();

  // Then clear the branches to default values (e.g. -999, or clear the vectors to empty vectors)
  ana.tx->clear();
}

//________________________________________________________________________________________________________________________________
void createRequiredOutputBranches() {
  // Setup output TTree
  ana.tx->createBranch<std::vector<float>>("sim_pt");
  ana.tx->createBranch<std::vector<float>>("sim_eta");
  ana.tx->createBranch<std::vector<float>>("sim_phi");
  ana.tx->createBranch<std::vector<float>>("sim_pca_dxy");
  ana.tx->createBranch<std::vector<float>>("sim_pca_dz");
  ana.tx->createBranch<std::vector<int>>("sim_q");
  ana.tx->createBranch<std::vector<int>>("sim_event");
  ana.tx->createBranch<std::vector<int>>("sim_pdgId");
  ana.tx->createBranch<std::vector<float>>("sim_vx");
  ana.tx->createBranch<std::vector<float>>("sim_vy");
  ana.tx->createBranch<std::vector<float>>("sim_vz");
  ana.tx->createBranch<std::vector<float>>("sim_trkNtupIdx");
  ana.tx->createBranch<std::vector<int>>("sim_TC_matched");
  ana.tx->createBranch<std::vector<int>>("sim_TC_matched_mask");

  // Track candidates
  ana.tx->createBranch<std::vector<float>>("tc_pt");
  ana.tx->createBranch<std::vector<float>>("tc_eta");
  ana.tx->createBranch<std::vector<float>>("tc_phi");
  ana.tx->createBranch<std::vector<int>>("tc_type");
  ana.tx->createBranch<std::vector<int>>("tc_isFake");
  ana.tx->createBranch<std::vector<int>>("tc_isDuplicate");
  ana.tx->createBranch<std::vector<std::vector<int>>>("tc_matched_simIdx");
}

//________________________________________________________________________________________________________________________________
void createOptionalOutputBranches() {
#ifdef CUT_VALUE_DEBUG
  // Event-wide branches
  // ana.tx->createBranch<float>("evt_dummy");

  // Sim Track branches
  // NOTE: Must sync with main tc branch in length!!
  ana.tx->createBranch<std::vector<float>>("sim_dummy");

  // Track Candidate branches
  // NOTE: Must sync with main tc branch in length!!
  ana.tx->createBranch<std::vector<float>>("tc_dummy");

  // pT5 branches
  ana.tx->createBranch<std::vector<std::vector<int>>>("pT5_matched_simIdx");
  ana.tx->createBranch<std::vector<std::vector<int>>>("pT5_hitIdxs");
  ana.tx->createBranch<std::vector<int>>("sim_pT5_matched");
  ana.tx->createBranch<std::vector<float>>("pT5_pt");
  ana.tx->createBranch<std::vector<float>>("pT5_eta");
  ana.tx->createBranch<std::vector<float>>("pT5_phi");
  ana.tx->createBranch<std::vector<int>>("pT5_isFake");
  ana.tx->createBranch<std::vector<int>>("pT5_isDuplicate");
  ana.tx->createBranch<std::vector<int>>("pT5_score");
  ana.tx->createBranch<std::vector<int>>("pT5_layer_binary");
  ana.tx->createBranch<std::vector<int>>("pT5_moduleType_binary");
  ana.tx->createBranch<std::vector<float>>("pT5_matched_pt");
  ana.tx->createBranch<std::vector<float>>("pT5_rzChiSquared");
  ana.tx->createBranch<std::vector<float>>("pT5_rPhiChiSquared");
  ana.tx->createBranch<std::vector<float>>("pT5_rPhiChiSquaredInwards");

  // pT3 branches
  ana.tx->createBranch<std::vector<int>>("sim_pT3_matched");
  ana.tx->createBranch<std::vector<float>>("pT3_pt");
  ana.tx->createBranch<std::vector<int>>("pT3_isFake");
  ana.tx->createBranch<std::vector<int>>("pT3_isDuplicate");
  ana.tx->createBranch<std::vector<float>>("pT3_eta");
  ana.tx->createBranch<std::vector<float>>("pT3_phi");
  ana.tx->createBranch<std::vector<float>>("pT3_score");
  ana.tx->createBranch<std::vector<int>>("pT3_foundDuplicate");
  ana.tx->createBranch<std::vector<std::vector<int>>>("pT3_matched_simIdx");
  ana.tx->createBranch<std::vector<std::vector<int>>>("pT3_hitIdxs");
  ana.tx->createBranch<std::vector<float>>("pT3_pixelRadius");
  ana.tx->createBranch<std::vector<float>>("pT3_pixelRadiusError");
  ana.tx->createBranch<std::vector<std::vector<float>>>("pT3_matched_pt");
  ana.tx->createBranch<std::vector<float>>("pT3_tripletRadius");
  ana.tx->createBranch<std::vector<float>>("pT3_rPhiChiSquared");
  ana.tx->createBranch<std::vector<float>>("pT3_rPhiChiSquaredInwards");
  ana.tx->createBranch<std::vector<float>>("pT3_rzChiSquared");
  ana.tx->createBranch<std::vector<int>>("pT3_layer_binary");
  ana.tx->createBranch<std::vector<int>>("pT3_moduleType_binary");

  // PT2 branches
  ana.tx->createBranch<std::vector<int>>("sim_pT2_matched");
  ana.tx->createBranch<std::vector<int>>("pT2_isFake");
  ana.tx->createBranch<std::vector<int>>("pT2_isDup_reco_test");
  ana.tx->createBranch<std::vector<int>>("pT2_isDuplicate");
  ana.tx->createBranch<std::vector<float>>("pT2_pt");
  ana.tx->createBranch<std::vector<float>>("pT2_eta");
  ana.tx->createBranch<std::vector<float>>("pT2_phi");
  ana.tx->createBranch<std::vector<int>>("pT2_layer_binary");
  ana.tx->createBranch<std::vector<int>>("pT2_moduleType_binary");

  ana.tx->createBranch<std::vector<std::vector<int>>>("pT2_matched_simIdx");  
//  ana.tx->createBranch<std::vector<std::vector<int>>>("pT2_hitIdxs");
//  ana.tx->createBranch<std::vector<float>>("pT2_pixelRadius");
//  ana.tx->createBranch<std::vector<float>>("pT2_pixelRadiusError");
//  ana.tx->createBranch<std::vector<std::vector<float>>>("pT2_matched_pt");
//  ana.tx->createBranch<std::vector<float>>("pT2_rzChiSquared");

  // pLS branches
  ana.tx->createBranch<std::vector<int>>("sim_pLS_matched");
  ana.tx->createBranch<std::vector<std::vector<int>>>("sim_pLS_types");
  ana.tx->createBranch<std::vector<int>>("pLS_isFake");
  ana.tx->createBranch<std::vector<int>>("pLS_isDuplicate");
  ana.tx->createBranch<std::vector<float>>("pLS_pt");
  ana.tx->createBranch<std::vector<float>>("pLS_eta");
  ana.tx->createBranch<std::vector<float>>("pLS_phi");
  ana.tx->createBranch<std::vector<float>>("pLS_score");

  // T5 branches
  ana.tx->createBranch<std::vector<int>>("sim_T5_matched");
  ana.tx->createBranch<std::vector<int>>("t5_isFake");
  ana.tx->createBranch<std::vector<float>>("t5_sim_vxy");
  ana.tx->createBranch<std::vector<float>>("t5_sim_vz");
  ana.tx->createBranch<std::vector<int>>("t5_isDuplicate");
  ana.tx->createBranch<std::vector<int>>("t5_foundDuplicate");
  ana.tx->createBranch<std::vector<float>>("t5_pt");
  ana.tx->createBranch<std::vector<float>>("t5_pMatched");
  ana.tx->createBranch<std::vector<float>>("t5_eta");
  ana.tx->createBranch<std::vector<float>>("t5_phi");
  ana.tx->createBranch<std::vector<float>>("t5_score_rphisum");
  ana.tx->createBranch<std::vector<std::vector<int>>>("t5_hitIdxs");
  ana.tx->createBranch<std::vector<std::vector<int>>>("t5_matched_simIdx");
  ana.tx->createBranch<std::vector<int>>("t5_moduleType_binary");
  ana.tx->createBranch<std::vector<int>>("t5_layer_binary");
  ana.tx->createBranch<std::vector<float>>("t5_matched_pt");
  ana.tx->createBranch<std::vector<float>>("t5_innerRadius");
  ana.tx->createBranch<std::vector<float>>("t5_outerRadius");
  ana.tx->createBranch<std::vector<float>>("t5_bridgeRadius");
  ana.tx->createBranch<std::vector<float>>("t5_chiSquared");
  ana.tx->createBranch<std::vector<float>>("t5_rzChiSquared");
  ana.tx->createBranch<std::vector<float>>("t5_nonAnchorChiSquared");
  ana.tx->createBranch<std::vector<float>>("t5_dBeta1");
  ana.tx->createBranch<std::vector<float>>("t5_dBeta2");

  // Occupancy branches
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
  ana.tx->createBranch<int>("pT2_occupancies");
  ana.tx->createBranch<int>("pT3_occupancies");
  ana.tx->createBranch<int>("pT5_occupancies");

  // T5 DNN branches
  createT5DNNBranches();

#endif
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
void setOutputBranches(lst::Event<Acc3D>* event) {
  // ============ Sim tracks =============
  int n_accepted_simtrk = 0;
  for (unsigned int isimtrk = 0; isimtrk < trk.sim_pt().size(); ++isimtrk) {
    // Skip out-of-time pileup
    if (trk.sim_bunchCrossing()[isimtrk] != 0)
      continue;

    // Skip non-hard-scatter
    if (trk.sim_event()[isimtrk] != 0)
      continue;

    ana.tx->pushbackToBranch<float>("sim_pt", trk.sim_pt()[isimtrk]);
    ana.tx->pushbackToBranch<float>("sim_eta", trk.sim_eta()[isimtrk]);
    ana.tx->pushbackToBranch<float>("sim_phi", trk.sim_phi()[isimtrk]);
    ana.tx->pushbackToBranch<float>("sim_pca_dxy", trk.sim_pca_dxy()[isimtrk]);
    ana.tx->pushbackToBranch<float>("sim_pca_dz", trk.sim_pca_dz()[isimtrk]);
    ana.tx->pushbackToBranch<int>("sim_q", trk.sim_q()[isimtrk]);
    ana.tx->pushbackToBranch<int>("sim_event", trk.sim_event()[isimtrk]);
    ana.tx->pushbackToBranch<int>("sim_pdgId", trk.sim_pdgId()[isimtrk]);

    // For vertex we need to look it up from simvtx info
    int vtxidx = trk.sim_parentVtxIdx()[isimtrk];
    ana.tx->pushbackToBranch<float>("sim_vx", trk.simvtx_x()[vtxidx]);
    ana.tx->pushbackToBranch<float>("sim_vy", trk.simvtx_y()[vtxidx]);
    ana.tx->pushbackToBranch<float>("sim_vz", trk.simvtx_z()[vtxidx]);

    // The trkNtupIdx is the idx in the trackingNtuple
    ana.tx->pushbackToBranch<float>("sim_trkNtupIdx", isimtrk);

    // Increase the counter for accepted simtrk
    n_accepted_simtrk++;
  }

  // Intermediate variables to keep track of matched track candidates for a given sim track
  std::vector<int> sim_TC_matched(n_accepted_simtrk);
  std::vector<int> sim_TC_matched_mask(n_accepted_simtrk);
  std::vector<int> sim_TC_matched_for_duplicate(trk.sim_pt().size());

  // Intermediate variables to keep track of matched sim tracks for a given track candidate
  std::vector<std::vector<int>> tc_matched_simIdx;

  // ============ Track candidates =============
  lst::TrackCandidates const* trackCandidates = event->getTrackCandidates()->data();
  unsigned int nTrackCandidates = *trackCandidates->nTrackCandidates;
  for (unsigned int idx = 0; idx < nTrackCandidates; idx++) {
    // Compute reco quantities of track candidate based on final object
    int type, isFake;
    float pt, eta, phi;
    std::vector<int> simidx;
    std::tie(type, pt, eta, phi, isFake, simidx) = parseTrackCandidate(event, idx);
    ana.tx->pushbackToBranch<float>("tc_pt", pt);
    ana.tx->pushbackToBranch<float>("tc_eta", eta);
    ana.tx->pushbackToBranch<float>("tc_phi", phi);
    ana.tx->pushbackToBranch<int>("tc_type", type);
    ana.tx->pushbackToBranch<int>("tc_isFake", isFake);
    tc_matched_simIdx.push_back(simidx);

    // Loop over matched sim idx and increase counter of TC_matched
    for (auto& idx : simidx) {
      // NOTE Important to note that the idx of the std::vector<> is same
      // as the tracking-ntuple's sim track idx ONLY because event==0 and bunchCrossing==0 condition is applied!!
      // Also do not try to access beyond the event and bunchCrossing
      if (idx < n_accepted_simtrk) {
        sim_TC_matched.at(idx) += 1;
        sim_TC_matched_mask.at(idx) |= (1 << type);
      }
      sim_TC_matched_for_duplicate.at(idx) += 1;
    }
  }

  // Using the intermedaite variables to compute whether a given track candidate is a duplicate
  std::vector<int> tc_isDuplicate(tc_matched_simIdx.size());
  // Loop over the track candidates
  for (unsigned int i = 0; i < tc_matched_simIdx.size(); ++i) {
    bool isDuplicate = false;
    // Loop over the sim idx matched to this track candidate
    for (unsigned int isim = 0; isim < tc_matched_simIdx[i].size(); ++isim) {
      // Using the sim_TC_matched to see whether this track candidate is matched to a sim track that is matched to more than one
      int simidx = tc_matched_simIdx[i][isim];
      if (sim_TC_matched_for_duplicate[simidx] > 1) {
        isDuplicate = true;
      }
    }
    tc_isDuplicate[i] = isDuplicate;
  }

  // Now set the last remaining branches
  ana.tx->setBranch<std::vector<int>>("sim_TC_matched", sim_TC_matched);
  ana.tx->setBranch<std::vector<int>>("sim_TC_matched_mask", sim_TC_matched_mask);
  ana.tx->setBranch<std::vector<std::vector<int>>>("tc_matched_simIdx", tc_matched_simIdx);
  ana.tx->setBranch<std::vector<int>>("tc_isDuplicate", tc_isDuplicate);
}

//________________________________________________________________________________________________________________________________
void setOptionalOutputBranches(lst::Event<Acc3D>* event) {
#ifdef CUT_VALUE_DEBUG

  setPixelQuintupletOutputBranches(event);
  setQuintupletOutputBranches(event);
  setPixelTripletOutputBranches(event);
  setPT2OutputBranches(event);
  setOccupancyBranches(event);
  setT5DNNBranches(event);

#endif
}

//________________________________________________________________________________________________________________________________
void setOccupancyBranches(lst::Event<Acc3D>* event) {
  lst::Modules const& modulesInGPU = (*event->getModules()->data());
  lst::MiniDoublets const& mdsInGPU = (*event->getMiniDoublets()->data());
  lst::Segments const& segmentsInGPU = (*event->getSegments()->data());
  lst::Triplets const& tripletsInGPU = (*event->getTriplets()->data());
  lst::Quintuplets const& quintupletsInGPU = (*event->getQuintuplets()->data());
  lst::PixelQuintuplets const& pixelQuintupletsInGPU = (*event->getPixelQuintuplets()->data());
  lst::PixelTriplets const& pixelTripletsInGPU = (*event->getPixelTriplets()->data());
  lst::PT2s const& PT2sInGPU = (*event->getPT2s()->data());
  lst::TrackCandidates const& trackCandidatesInGPU = (*event->getTrackCandidates()->data());

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

  for (unsigned int lowerIdx = 0; lowerIdx <= *(modulesInGPU.nLowerModules); lowerIdx++) {
    //layer = 0, subdet = 0 => pixel module
    moduleLayer.push_back(modulesInGPU.layers[lowerIdx]);
    moduleSubdet.push_back(modulesInGPU.subdets[lowerIdx]);
    moduleRing.push_back(modulesInGPU.rings[lowerIdx]);
    moduleRod.push_back(modulesInGPU.rods[lowerIdx]);
    moduleEta.push_back(modulesInGPU.eta[lowerIdx]);
    moduleR.push_back(modulesInGPU.r[lowerIdx]);
    bool isTilted = (modulesInGPU.subdets[lowerIdx] == 5 and modulesInGPU.sides[lowerIdx] != 3);
    moduleIsTilted.push_back(isTilted);
    moduleModule.push_back(modulesInGPU.modules[lowerIdx]);
    segmentOccupancy.push_back(segmentsInGPU.totOccupancySegments[lowerIdx]);
    mdOccupancy.push_back(mdsInGPU.totOccupancyMDs[lowerIdx]);

    if (lowerIdx < *(modulesInGPU.nLowerModules)) {
      quintupletOccupancy.push_back(quintupletsInGPU.totOccupancyQuintuplets[lowerIdx]);
      tripletOccupancy.push_back(tripletsInGPU.totOccupancyTriplets[lowerIdx]);
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
  ana.tx->setBranch<int>("tc_occupancies", *(trackCandidatesInGPU.nTrackCandidates));
  ana.tx->setBranch<int>("pT2_occupancies", *(PT2sInGPU.totOccupancyPT2s));
  ana.tx->setBranch<int>("pT3_occupancies", *(pixelTripletsInGPU.totOccupancyPixelTriplets));
  ana.tx->setBranch<std::vector<int>>("t5_occupancies", quintupletOccupancy);
  ana.tx->setBranch<int>("pT5_occupancies", *(pixelQuintupletsInGPU.totOccupancyPixelQuintuplets));
}

//________________________________________________________________________________________________________________________________
void setPixelQuintupletOutputBranches(lst::Event<Acc3D>* event) {
  // ============ pT5 =============
  lst::PixelQuintuplets const* pixelQuintuplets = event->getPixelQuintuplets()->data();
  lst::Quintuplets const* quintuplets = event->getQuintuplets()->data();
  lst::Segments const* segments = event->getSegments()->data();
  lst::Modules const* modules = event->getModules()->data();
  int n_accepted_simtrk = ana.tx->getBranch<std::vector<int>>("sim_TC_matched").size();

  unsigned int nPixelQuintuplets =
      *pixelQuintuplets->nPixelQuintuplets;  // size of this nPixelTriplets array is 1 (NOTE: parallelism lost here.)
  std::vector<int> sim_pT5_matched(n_accepted_simtrk);
  std::vector<std::vector<int>> pT5_matched_simIdx;

  for (unsigned int pT5 = 0; pT5 < nPixelQuintuplets; pT5++) {
    unsigned int T5Index = getT5FrompT5(event, pT5);
    unsigned int pLSIndex = getPixelLSFrompT5(event, pT5);
    float pt = (__H2F(quintuplets->innerRadius[T5Index]) * lst::k2Rinv1GeVf * 2 + segments->ptIn[pLSIndex]) / 2;
    float eta = segments->eta[pLSIndex];
    float phi = segments->phi[pLSIndex];

    std::vector<unsigned int> hit_idx = getHitIdxsFrompT5(event, pT5);
    std::vector<unsigned int> module_idx = getModuleIdxsFrompT5(event, pT5);
    std::vector<unsigned int> hit_type = getHitTypesFrompT5(event, pT5);

    int layer_binary = 1;
    int moduleType_binary = 0;
    for (size_t i = 0; i < module_idx.size(); i += 2) {
      layer_binary |= (1 << (modules->layers[module_idx[i]] + 6 * (modules->subdets[module_idx[i]] == 4)));
      moduleType_binary |= (modules->moduleType[module_idx[i]] << i);
    }
    std::vector<int> simidx = matchedSimTrkIdxs(hit_idx, hit_type);
    ana.tx->pushbackToBranch<int>("pT5_isFake", static_cast<int>(simidx.size() == 0));
    ana.tx->pushbackToBranch<float>("pT5_pt", pt);
    ana.tx->pushbackToBranch<float>("pT5_eta", eta);
    ana.tx->pushbackToBranch<float>("pT5_phi", phi);
    ana.tx->pushbackToBranch<int>("pT5_layer_binary", layer_binary);
    ana.tx->pushbackToBranch<int>("pT5_moduleType_binary", moduleType_binary);
    ana.tx->pushbackToBranch<float>("pT5_rzChiSquared", pixelQuintuplets->rzChiSquared[pT5]);

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
void setQuintupletOutputBranches(lst::Event<Acc3D>* event) {
  lst::Quintuplets const* quintuplets = event->getQuintuplets()->data();
  lst::ObjectRanges const* ranges = event->getRanges()->data();
  lst::Modules const* modules = event->getModules()->data();
  int n_accepted_simtrk = ana.tx->getBranch<std::vector<int>>("sim_TC_matched").size();

  std::vector<int> sim_t5_matched(n_accepted_simtrk);
  std::vector<std::vector<int>> t5_matched_simIdx;

  for (unsigned int lowerModuleIdx = 0; lowerModuleIdx < *(modules->nLowerModules); ++lowerModuleIdx) {
    int nQuintuplets = quintuplets->nQuintuplets[lowerModuleIdx];
    for (unsigned int idx = 0; idx < nQuintuplets; idx++) {
      unsigned int quintupletIndex = ranges->quintupletModuleIndices[lowerModuleIdx] + idx;
      float pt = __H2F(quintuplets->innerRadius[quintupletIndex]) * lst::k2Rinv1GeVf * 2;
      float eta = __H2F(quintuplets->eta[quintupletIndex]);
      float phi = __H2F(quintuplets->phi[quintupletIndex]);

      std::vector<unsigned int> hit_idx = getHitIdxsFromT5(event, quintupletIndex);
      std::vector<unsigned int> hit_type = getHitTypesFromT5(event, quintupletIndex);
      std::vector<unsigned int> module_idx = getModuleIdxsFromT5(event, quintupletIndex);

      int layer_binary = 0;
      int moduleType_binary = 0;
      for (size_t i = 0; i < module_idx.size(); i += 2) {
        layer_binary |= (1 << (modules->layers[module_idx[i]] + 6 * (modules->subdets[module_idx[i]] == 4)));
        moduleType_binary |= (modules->moduleType[module_idx[i]] << i);
      }

      float percent_matched;
      std::vector<int> simidx = matchedSimTrkIdxs(hit_idx, hit_type, false, &percent_matched);

      ana.tx->pushbackToBranch<int>("t5_isFake", static_cast<int>(simidx.size() == 0));
      ana.tx->pushbackToBranch<float>("t5_pt", pt);
      ana.tx->pushbackToBranch<float>("t5_pMatched", percent_matched);
      ana.tx->pushbackToBranch<float>("t5_eta", eta);
      ana.tx->pushbackToBranch<float>("t5_phi", phi);
      ana.tx->pushbackToBranch<float>("t5_innerRadius", __H2F(quintuplets->innerRadius[quintupletIndex]));
      ana.tx->pushbackToBranch<float>("t5_bridgeRadius", __H2F(quintuplets->bridgeRadius[quintupletIndex]));
      ana.tx->pushbackToBranch<float>("t5_outerRadius", __H2F(quintuplets->outerRadius[quintupletIndex]));
      ana.tx->pushbackToBranch<float>("t5_chiSquared", quintuplets->chiSquared[quintupletIndex]);
      ana.tx->pushbackToBranch<float>("t5_rzChiSquared", quintuplets->rzChiSquared[quintupletIndex]);
      ana.tx->pushbackToBranch<float>("t5_nonAnchorChiSquared", quintuplets->nonAnchorChiSquared[quintupletIndex]);
      ana.tx->pushbackToBranch<float>("t5_dBeta1", quintuplets->dBeta1[quintupletIndex]);
      ana.tx->pushbackToBranch<float>("t5_dBeta2", quintuplets->dBeta2[quintupletIndex]);
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

      int vtxidx = trk.sim_parentVtxIdx()[simidx[0]];
      float vtx_x = trk.simvtx_x()[vtxidx];
      float vtx_y = trk.simvtx_y()[vtxidx];
      float vtx_z = trk.simvtx_z()[vtxidx];

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
void setPixelTripletOutputBranches(lst::Event<Acc3D>* event) {
  lst::PixelTriplets const* pixelTriplets = event->getPixelTriplets()->data();
  lst::Modules const* modules = event->getModules()->data();
  lst::Segments const* segments = event->getSegments()->data();
  int n_accepted_simtrk = ana.tx->getBranch<std::vector<int>>("sim_TC_matched").size();

  unsigned int nPixelTriplets = *pixelTriplets->nPixelTriplets;
  std::vector<int> sim_pT3_matched(n_accepted_simtrk);
  std::vector<std::vector<int>> pT3_matched_simIdx;

  for (unsigned int pT3 = 0; pT3 < nPixelTriplets; pT3++) {
    unsigned int T3Index = getT3FrompT3(event, pT3);
    unsigned int pLSIndex = getPixelLSFrompT3(event, pT3);
    const float pt = segments->ptIn[pLSIndex];

    float eta = segments->eta[pLSIndex];
    float phi = segments->phi[pLSIndex];
    std::vector<unsigned int> hit_idx = getHitIdxsFrompT3(event, pT3);
    std::vector<unsigned int> hit_type = getHitTypesFrompT3(event, pT3);

    std::vector<int> simidx = matchedSimTrkIdxs(hit_idx, hit_type);
    std::vector<unsigned int> module_idx = getModuleIdxsFrompT3(event, pT3);
    int layer_binary = 1;
    int moduleType_binary = 0;
    for (size_t i = 0; i < module_idx.size(); i += 2) {
      layer_binary |= (1 << (modules->layers[module_idx[i]] + 6 * (modules->subdets[module_idx[i]] == 4)));
      moduleType_binary |= (modules->moduleType[module_idx[i]] << i);
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
void setPT2OutputBranches(lst::Event<Acc3D>* event) {
  lst::PT2s const* PT2s = event->getPT2s()->data();
  lst::Modules const* modules = event->getModules()->data();
  lst::Segments const* segments = event->getSegments()->data();
  int n_accepted_simtrk = ana.tx->getBranch<std::vector<int>>("sim_TC_matched").size();

  unsigned int nPT2s = *PT2s->nPT2s;
  std::vector<int> sim_pT2_matched(n_accepted_simtrk);
  std::vector<std::vector<int>> pT2_matched_simIdx;

  for (unsigned int PT2 = 0; PT2 < nPT2s; PT2++) {
    //unsigned int LSIndex = getLSsFrompT2(event, PT2);
    unsigned int pLSIndex = getPixelLSFrompT2(event, PT2);
    const float pt = segments->ptIn[pLSIndex];

    float eta = segments->eta[pLSIndex];
    float phi = segments->phi[pLSIndex];
    std::vector<unsigned int> hit_idx = getHitIdxsFrompT2(event, PT2);
    std::vector<unsigned int> hit_type = getHitTypesFrompT2(event, PT2);

    std::vector<int> simidx = matchedSimTrkIdxs(hit_idx, hit_type);
    std::vector<unsigned int> module_idx = getModuleIdxsFrompT2(event, PT2);
    int layer_binary = 1;
    int moduleType_binary = 0;
    for (size_t i = 0; i < module_idx.size(); i += 2) {
      layer_binary |= (1 << (modules->layers[module_idx[i]] + 6 * (modules->subdets[module_idx[i]] == 4)));
      moduleType_binary |= (modules->moduleType[module_idx[i]] << i);
    }
    ana.tx->pushbackToBranch<int>("pT2_isFake", static_cast<int>(simidx.size() == 0));
    ana.tx->pushbackToBranch<float>("pT2_pt", pt);
    ana.tx->pushbackToBranch<float>("pT2_eta", eta);
    ana.tx->pushbackToBranch<float>("pT2_phi", phi);
    ana.tx->pushbackToBranch<int>("pT2_layer_binary", layer_binary);
    ana.tx->pushbackToBranch<int>("pT2_moduleType_binary", moduleType_binary);

    pT2_matched_simIdx.push_back(simidx);

    for (auto& idx : simidx) {
      if (idx < n_accepted_simtrk) {
        sim_pT2_matched.at(idx) += 1;
      }
    }
  }

  std::vector<int> pT2_isDuplicate(pT2_matched_simIdx.size());
  for (unsigned int i = 0; i < pT2_matched_simIdx.size(); i++) {
    bool isDuplicate = true;
    for (unsigned int isim = 0; isim < pT2_matched_simIdx[i].size(); isim++) {
      int simidx = pT2_matched_simIdx[i][isim];
      if (simidx < n_accepted_simtrk) {
        if (sim_pT2_matched[simidx] > 1) {
          isDuplicate = true;
        }
      }
    }
    pT2_isDuplicate[i] = isDuplicate;
  }
  ana.tx->setBranch<std::vector<int>>("sim_pT2_matched", sim_pT2_matched);
  ana.tx->setBranch<std::vector<std::vector<int>>>("pT2_matched_simIdx", pT2_matched_simIdx);
  ana.tx->setBranch<std::vector<int>>("pT2_isDuplicate", pT2_isDuplicate);
}

//________________________________________________________________________________________________________________________________
void fillT5DNNBranches(lst::Event<Acc3D>* event, unsigned int iT3) {
  lst::Hits const* hits = event->getHits()->data();
  lst::Modules const* modules = event->getModules()->data();

  std::vector<unsigned int> hitIdx = getHitsFromT3(event, iT3);
  std::vector<lst_math::Hit> hitObjects(hitIdx.size());

  for (int i = 0; i < hitIdx.size(); ++i) {
    unsigned int hit = hitIdx[i];
    float x = hits->xs[hit];
    float y = hits->ys[hit];
    float z = hits->zs[hit];
    hitObjects[i] = lst_math::Hit(x, y, z);

    std::string idx = std::to_string(i);
    ana.tx->pushbackToBranch<float>("t5_t3_" + idx + "_r", sqrt(x * x + y * y));
    ana.tx->pushbackToBranch<float>("t5_t3_" + idx + "_x", x);
    ana.tx->pushbackToBranch<float>("t5_t3_" + idx + "_y", y);
    ana.tx->pushbackToBranch<float>("t5_t3_" + idx + "_z", z);
    ana.tx->pushbackToBranch<float>("t5_t3_" + idx + "_eta", hitObjects[i].eta());
    ana.tx->pushbackToBranch<float>("t5_t3_" + idx + "_phi", hitObjects[i].phi());

    int subdet = trk.ph2_subdet()[hits->idxs[hit]];
    int is_endcap = subdet == 4;
    int layer = trk.ph2_layer()[hits->idxs[hit]] + 6 * is_endcap;
    int detId = trk.ph2_detId()[hits->idxs[hit]];
    unsigned int module = hits->moduleIndices[hit];

    ana.tx->pushbackToBranch<int>("t5_t3_" + idx + "_detId", detId);
    ana.tx->pushbackToBranch<int>("t5_t3_" + idx + "_layer", layer);
    ana.tx->pushbackToBranch<int>("t5_t3_" + idx + "_moduleType", modules->moduleType[module]);
  }

  float g, f;
  lst::DevHost const& devHost = cms::alpakatools::host();
  float radius = lst::computeRadiusFromThreeAnchorHits(devHost,
                                                       hitObjects[0].x(),
                                                       hitObjects[0].y(),
                                                       hitObjects[1].x(),
                                                       hitObjects[1].y(),
                                                       hitObjects[2].x(),
                                                       hitObjects[2].y(),
                                                       g,
                                                       f);
  ana.tx->pushbackToBranch<float>("t5_t3_pt", lst::k2Rinv1GeVf * 2 * radius);

  // Angles
  ana.tx->pushbackToBranch<float>("t5_t3_eta", hitObjects[2].eta());
  ana.tx->pushbackToBranch<float>("t5_t3_phi", hitObjects[0].phi());
}

//________________________________________________________________________________________________________________________________
void setT5DNNBranches(lst::Event<Acc3D>* event) {
  lst::Triplets const* triplets = event->getTriplets()->data();
  lst::Modules const* modules = event->getModules()->data();
  lst::ObjectRanges const* ranges = event->getRanges()->data();
  lst::Quintuplets const* quintuplets = event->getQuintuplets()->data();
  lst::TrackCandidates const* trackCandidates = event->getTrackCandidates()->data();

  std::unordered_set<unsigned int> allT3s;
  std::unordered_map<unsigned int, unsigned int> t3_index_map;

  for (unsigned int idx = 0; idx < *(modules->nLowerModules); ++idx) {
    for (unsigned int jdx = 0; jdx < triplets->nTriplets[idx]; ++jdx) {
      unsigned int t3Idx = ranges->tripletModuleIndices[idx] + jdx;
      if (allT3s.insert(t3Idx).second) {
        t3_index_map[t3Idx] = allT3s.size() - 1;
        fillT5DNNBranches(event, t3Idx);
      }
    }
  }

  std::unordered_map<unsigned int, unsigned int> t5_tc_index_map;
  std::unordered_set<unsigned int> t5s_used_in_tc;

  for (unsigned int idx = 0; idx < *(trackCandidates->nTrackCandidates); idx++) {
    if (trackCandidates->trackCandidateType[idx] == kT5) {
      unsigned int objIdx = trackCandidates->directObjectIndices[idx];
      t5s_used_in_tc.insert(objIdx);
      t5_tc_index_map[objIdx] = idx;
    }
  }

  for (unsigned int idx = 0; idx < *(modules->nLowerModules); ++idx) {
    for (unsigned int jdx = 0; jdx < quintuplets->nQuintuplets[idx]; ++jdx) {
      unsigned int t5Idx = ranges->quintupletModuleIndices[idx] + jdx;
      std::vector<unsigned int> t3sIdx = getT3sFromT5(event, t5Idx);

      ana.tx->pushbackToBranch<int>("t5_t3_idx0", t3_index_map[t3sIdx[0]]);
      ana.tx->pushbackToBranch<int>("t5_t3_idx1", t3_index_map[t3sIdx[1]]);

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
void setGnnNtupleBranches(lst::Event<Acc3D>* event) {
  // Get relevant information
  lst::Segments const* segments = event->getSegments()->data();
  lst::MiniDoublets const* miniDoublets = event->getMiniDoublets()->data();
  lst::Hits const* hitsEvt = event->getHits()->data();
  lst::Modules const* modules = event->getModules()->data();
  lst::ObjectRanges const* ranges = event->getRanges()->data();
  lst::TrackCandidates const* trackCandidates = event->getTrackCandidates()->data();

  std::set<unsigned int> mds_used_in_sg;
  std::map<unsigned int, unsigned int> md_index_map;
  std::map<unsigned int, unsigned int> sg_index_map;

  // Loop over modules (lower ones where the MDs are saved)
  unsigned int nTotalMD = 0;
  unsigned int nTotalLS = 0;
  for (unsigned int idx = 0; idx < *(modules->nLowerModules); ++idx) {
    nTotalMD += miniDoublets->nMDs[idx];
    nTotalLS += segments->nSegments[idx];
  }

  std::set<unsigned int> lss_used_in_true_tc;
  unsigned int nTrackCandidates = *trackCandidates->nTrackCandidates;
  for (unsigned int idx = 0; idx < nTrackCandidates; idx++) {
    // Only consider true track candidates
    std::vector<unsigned int> hitidxs;
    std::vector<unsigned int> hittypes;
    std::tie(hitidxs, hittypes) = getHitIdxsAndHitTypesFromTC(event, idx);
    std::vector<int> simidxs = matchedSimTrkIdxs(hitidxs, hittypes);
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

  // Loop over modules (lower ones where the MDs are saved)
  for (unsigned int idx = 0; idx < *(modules->nLowerModules); ++idx) {
    // // Loop over minidoublets
    // for (unsigned int jdx = 0; jdx < miniDoublets->nMDs[idx]; jdx++)
    // {
    //     // Get the actual index to the mini-doublet using ranges
    //     unsigned int mdIdx = ranges->miniDoubletModuleIndices[idx] + jdx;

    //     setGnnNtupleMiniDoublet(event, mdIdx);
    // }

    // Loop over segments
    for (unsigned int jdx = 0; jdx < segments->nSegments[idx]; jdx++) {
      // Get the actual index to the segments using ranges
      unsigned int sgIdx = ranges->segmentModuleIndices[idx] + jdx;

      // Get the hit indices
      std::vector<unsigned int> MDs = getMDsFromLS(event, sgIdx);

      if (mds_used_in_sg.find(MDs[0]) == mds_used_in_sg.end()) {
        mds_used_in_sg.insert(MDs[0]);
        md_index_map[MDs[0]] = mds_used_in_sg.size() - 1;
        setGnnNtupleMiniDoublet(event, MDs[0]);
      }

      if (mds_used_in_sg.find(MDs[1]) == mds_used_in_sg.end()) {
        mds_used_in_sg.insert(MDs[1]);
        md_index_map[MDs[1]] = mds_used_in_sg.size() - 1;
        setGnnNtupleMiniDoublet(event, MDs[1]);
      }

      ana.tx->pushbackToBranch<int>("LS_MD_idx0", md_index_map[MDs[0]]);
      ana.tx->pushbackToBranch<int>("LS_MD_idx1", md_index_map[MDs[1]]);

      std::vector<unsigned int> hits = getHitsFromLS(event, sgIdx);

      // Computing line segment pt estimate (assuming beam spot is at zero)
      lst_math::Hit hitA(0, 0, 0);
      lst_math::Hit hitB(hitsEvt->xs[hits[0]], hitsEvt->ys[hits[0]], hitsEvt->zs[hits[0]]);
      lst_math::Hit hitC(hitsEvt->xs[hits[2]], hitsEvt->ys[hits[2]], hitsEvt->zs[hits[2]]);
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
      std::vector<int> simidxs = matchedSimTrkIdxs(hitidxs, hittypes);

      ana.tx->pushbackToBranch<int>("LS_isFake", simidxs.size() == 0);
      ana.tx->pushbackToBranch<float>("LS_sim_pt", simidxs.size() > 0 ? trk.sim_pt()[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<float>("LS_sim_eta", simidxs.size() > 0 ? trk.sim_eta()[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<float>("LS_sim_phi", simidxs.size() > 0 ? trk.sim_phi()[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<float>("LS_sim_pca_dxy", simidxs.size() > 0 ? trk.sim_pca_dxy()[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<float>("LS_sim_pca_dz", simidxs.size() > 0 ? trk.sim_pca_dz()[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<int>("LS_sim_q", simidxs.size() > 0 ? trk.sim_q()[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<int>("LS_sim_event", simidxs.size() > 0 ? trk.sim_event()[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<int>("LS_sim_bx", simidxs.size() > 0 ? trk.sim_bunchCrossing()[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<int>("LS_sim_pdgId", simidxs.size() > 0 ? trk.sim_pdgId()[simidxs[0]] : -999);
      ana.tx->pushbackToBranch<float>("LS_sim_vx",
                                      simidxs.size() > 0 ? trk.simvtx_x()[trk.sim_parentVtxIdx()[simidxs[0]]] : -999);
      ana.tx->pushbackToBranch<float>("LS_sim_vy",
                                      simidxs.size() > 0 ? trk.simvtx_y()[trk.sim_parentVtxIdx()[simidxs[0]]] : -999);
      ana.tx->pushbackToBranch<float>("LS_sim_vz",
                                      simidxs.size() > 0 ? trk.simvtx_z()[trk.sim_parentVtxIdx()[simidxs[0]]] : -999);
      ana.tx->pushbackToBranch<int>("LS_isInTrueTC", lss_used_in_true_tc.find(sgIdx) != lss_used_in_true_tc.end());

      sg_index_map[sgIdx] = ana.tx->getBranch<std::vector<int>>("LS_isFake").size() - 1;

      // // T5 eta and phi are computed using outer and innermost hits
      // lst_math::Hit hitA(trk.ph2_x()[anchitidx], trk.ph2_y()[anchitidx], trk.ph2_z()[anchitidx]);
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
void setGnnNtupleMiniDoublet(lst::Event<Acc3D>* event, unsigned int MD) {
  // Get relevant information
  lst::MiniDoublets const* miniDoublets = event->getMiniDoublets()->data();
  lst::Hits const* hitsEvt = event->getHits()->data();

  // Get the hit indices
  unsigned int hit0 = miniDoublets->anchorHitIndices[MD];
  unsigned int hit1 = miniDoublets->outerHitIndices[MD];

  // Get the hit infos
  const float hit0_x = hitsEvt->xs[hit0];
  const float hit0_y = hitsEvt->ys[hit0];
  const float hit0_z = hitsEvt->zs[hit0];
  const float hit0_r = sqrt(hit0_x * hit0_x + hit0_y * hit0_y);
  const float hit1_x = hitsEvt->xs[hit1];
  const float hit1_y = hitsEvt->ys[hit1];
  const float hit1_z = hitsEvt->zs[hit1];
  const float hit1_r = sqrt(hit1_x * hit1_x + hit1_y * hit1_y);

  // Do sim matching
  std::vector<unsigned int> hit_idx = {hitsEvt->idxs[hit0], hitsEvt->idxs[hit1]};
  std::vector<unsigned int> hit_type = {4, 4};
  std::vector<int> simidxs = matchedSimTrkIdxs(hit_idx, hit_type);

  bool isFake = simidxs.size() == 0;
  int tp_type = getDenomSimTrkType(simidxs);

  // Obtain where the actual hit is located in terms of their layer, module, rod, and ring number
  unsigned int anchitidx = hitsEvt->idxs[hit0];
  int subdet = trk.ph2_subdet()[hitsEvt->idxs[anchitidx]];
  int is_endcap = subdet == 4;
  int layer =
      trk.ph2_layer()[anchitidx] +
      6 * (is_endcap);  // this accounting makes it so that you have layer 1 2 3 4 5 6 in the barrel, and 7 8 9 10 11 in the endcap. (becuase endcap is ph2_subdet == 4)
  int detId = trk.ph2_detId()[anchitidx];

  // Obtaining dPhiChange
  float dphichange = miniDoublets->dphichanges[MD];

  // Computing pt
  float pt = hit0_r * lst::k2Rinv1GeVf / sin(dphichange);

  // T5 eta and phi are computed using outer and innermost hits
  lst_math::Hit hitA(trk.ph2_x()[anchitidx], trk.ph2_y()[anchitidx], trk.ph2_z()[anchitidx]);
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
std::tuple<int, float, float, float, int, std::vector<int>> parseTrackCandidate(lst::Event<Acc3D>* event,
                                                                                unsigned int idx) {
  // Get the type of the track candidate
  lst::TrackCandidates const* trackCandidates = event->getTrackCandidates()->data();
  short type = trackCandidates->trackCandidateType[idx];

  enum { pT5 = 7, pT3 = 5, T5 = 4, pLS = 8, pT2 = 10 };

  // Compute pt eta phi and hit indices that will be used to figure out whether the TC matched
  float pt, eta, phi;
  std::vector<unsigned int> hit_idx, hit_type;
  switch (type) {
    case pT5:
      std::tie(pt, eta, phi, hit_idx, hit_type) = parsepT5(event, idx);
      break;
    case pT3:
      std::tie(pt, eta, phi, hit_idx, hit_type) = parsepT3(event, idx);
      break;
    case pT2:
      std::tie(pt, eta, phi, hit_idx, hit_type) = parsepT2(event, idx);
      break;
    case T5:
      std::tie(pt, eta, phi, hit_idx, hit_type) = parseT5(event, idx);
      break;
    case pLS:
      std::tie(pt, eta, phi, hit_idx, hit_type) = parsepLS(event, idx);
      break;
  }

  // Perform matching
  std::vector<int> simidx = matchedSimTrkIdxs(hit_idx, hit_type);
  int isFake = simidx.size() == 0;

  return {type, pt, eta, phi, isFake, simidx};
}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, std::vector<unsigned int>, std::vector<unsigned int>> parsepT5(lst::Event<Acc3D>* event,
                                                                                               unsigned int idx) {
  // Get relevant information
  lst::TrackCandidates const* trackCandidates = event->getTrackCandidates()->data();
  lst::Quintuplets const* quintuplets = event->getQuintuplets()->data();
  lst::Segments const* segments = event->getSegments()->data();

  //
  // pictorial representation of a pT5
  //
  // inner tracker        outer tracker
  // -------------  --------------------------
  // pLS            01    23    45    67    89   (anchor hit of a minidoublet is always the first of the pair)
  // ****           oo -- oo -- oo -- oo -- oo   pT5
  //                oo -- oo -- oo               first T3 of the T5
  //                            oo -- oo -- oo   second T3 of the T5
  unsigned int pT5 = trackCandidates->directObjectIndices[idx];
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
  const float pt_pLS = segments->ptIn[pLS];
  const float eta_pLS = segments->eta[pLS];
  const float phi_pLS = segments->phi[pLS];
  float pt_T5 = __H2F(quintuplets->innerRadius[T5Index]) * 2 * lst::k2Rinv1GeVf;
  const float pt = (pt_T5 + pt_pLS) / 2;

  // Form the hit idx/type std::vector
  std::vector<unsigned int> hit_idx = getHitIdxsFrompT5(event, pT5);
  std::vector<unsigned int> hit_type = getHitTypesFrompT5(event, pT5);

  return {pt, eta_pLS, phi_pLS, hit_idx, hit_type};
}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, std::vector<unsigned int>, std::vector<unsigned int>> parsepT3(lst::Event<Acc3D>* event,
                                                                                               unsigned int idx) {
  // Get relevant information
  lst::TrackCandidates const* trackCandidates = event->getTrackCandidates()->data();
  lst::Triplets const* triplets = event->getTriplets()->data();
  lst::Segments const* segments = event->getSegments()->data();

  //
  // pictorial representation of a pT3
  //
  // inner tracker        outer tracker
  // -------------  --------------------------
  // pLS            01    23    45               (anchor hit of a minidoublet is always the first of the pair)
  // ****           oo -- oo -- oo               pT3
  unsigned int pT3 = trackCandidates->directObjectIndices[idx];
  unsigned int pLS = getPixelLSFrompT3(event, pT3);
  unsigned int T3 = getT3FrompT3(event, pT3);

  // pixel pt
  const float pt_pLS = segments->ptIn[pLS];
  const float eta_pLS = segments->eta[pLS];
  const float phi_pLS = segments->phi[pLS];
  float pt_T3 = triplets->circleRadius[T3] * 2 * lst::k2Rinv1GeVf;

  // average pt
  const float pt = (pt_pLS + pt_T3) / 2;

  // Form the hit idx/type std::vector
  std::vector<unsigned int> hit_idx = getHitIdxsFrompT3(event, pT3);
  std::vector<unsigned int> hit_type = getHitTypesFrompT3(event, pT3);

  return {pt, eta_pLS, phi_pLS, hit_idx, hit_type};
}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, std::vector<unsigned int>, std::vector<unsigned int>> parsepT2(lst::Event<Acc3D>* event,
                                                                                               unsigned int idx) {
  // Get relevant information
  lst::TrackCandidates const* trackCandidates = event->getTrackCandidates()->data();
  lst::Segments const* segments = event->getSegments()->data();

  //
  // pictorial representation of a pT3
  //
  // inner tracker        outer tracker
  // -------------  --------------------------
  // pLS            01    23    45               (anchor hit of a minidoublet is always the first of the pair)
  // ****           oo -- oo -- oo               pT3
  unsigned int PT2 = trackCandidates->directObjectIndices[idx];
  unsigned int pLS = getPixelLSFrompT2(event, PT2);
  //unsigned int LS = getLSFromPT2(event, PT2);

  // pixel pt
  const float pt_pLS = segments->ptIn[pLS];
  const float eta_pLS = segments->eta[pLS];
  const float phi_pLS = segments->phi[pLS];
  // average pt
  const float pt = pt_pLS ;

  // Form the hit idx/type std::vector
  std::vector<unsigned int> hit_idx = getHitIdxsFrompT2(event, PT2);
  std::vector<unsigned int> hit_type = getHitTypesFrompT2(event, PT2);

  return {pt, eta_pLS, phi_pLS, hit_idx, hit_type};
}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, std::vector<unsigned int>, std::vector<unsigned int>> parseT5(lst::Event<Acc3D>* event,
                                                                                              unsigned int idx) {
  lst::TrackCandidates const* trackCandidates = event->getTrackCandidates()->data();
  lst::Quintuplets const* quintuplets = event->getQuintuplets()->data();
  unsigned int T5 = trackCandidates->directObjectIndices[idx];
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
  const float pt = quintuplets->innerRadius[T5] * lst::k2Rinv1GeVf * 2;

  // T5 eta and phi are computed using outer and innermost hits
  lst_math::Hit hitA(trk.ph2_x()[Hit_0], trk.ph2_y()[Hit_0], trk.ph2_z()[Hit_0]);
  lst_math::Hit hitB(trk.ph2_x()[Hit_8], trk.ph2_y()[Hit_8], trk.ph2_z()[Hit_8]);
  const float phi = hitA.phi();
  const float eta = hitB.eta();

  std::vector<unsigned int> hit_idx = getHitIdxsFromT5(event, T5);
  std::vector<unsigned int> hit_type = getHitTypesFromT5(event, T5);

  return {pt, eta, phi, hit_idx, hit_type};
}

//________________________________________________________________________________________________________________________________
std::tuple<float, float, float, std::vector<unsigned int>, std::vector<unsigned int>> parsepLS(lst::Event<Acc3D>* event,
                                                                                               unsigned int idx) {
  lst::TrackCandidates const* trackCandidates = event->getTrackCandidates()->data();
  lst::Segments const* segments = event->getSegments()->data();

  // Getting pLS index
  unsigned int pLS = trackCandidates->directObjectIndices[idx];

  // Getting pt eta and phi
  float pt = segments->ptIn[pLS];
  float eta = segments->eta[pLS];
  float phi = segments->phi[pLS];

  // Getting hit indices and types
  std::vector<unsigned int> hit_idx = getPixelHitIdxsFrompLS(event, pLS);
  std::vector<unsigned int> hit_type = getPixelHitTypesFrompLS(event, pLS);

  return {pt, eta, phi, hit_idx, hit_type};
}

//________________________________________________________________________________________________________________________________
void printHitMultiplicities(lst::Event<Acc3D>* event) {
  lst::Modules const* modules = event->getModules()->data();
  lst::ObjectRanges const* ranges = event->getRanges()->data();

  int nHits = 0;
  for (unsigned int idx = 0; idx <= *(modules->nLowerModules);
       idx++)  // "<=" because cheating to include pixel track candidate lower module
  {
    nHits += ranges->hitRanges[4 * idx + 1] - ranges->hitRanges[4 * idx] + 1;
    nHits += ranges->hitRanges[4 * idx + 3] - ranges->hitRanges[4 * idx + 2] + 1;
  }
  std::cout << " nHits: " << nHits << std::endl;
}

//________________________________________________________________________________________________________________________________
void printMiniDoubletMultiplicities(lst::Event<Acc3D>* event) {
  lst::MiniDoublets const* miniDoublets = event->getMiniDoublets()->data();
  lst::Modules const* modules = event->getModules()->data();

  int nMiniDoublets = 0;
  int totOccupancyMiniDoublets = 0;
  for (unsigned int idx = 0; idx <= *(modules->nModules);
       idx++)  // "<=" because cheating to include pixel track candidate lower module
  {
    if (modules->isLower[idx]) {
      nMiniDoublets += miniDoublets->nMDs[idx];
      totOccupancyMiniDoublets += miniDoublets->totOccupancyMDs[idx];
    }
  }
  std::cout << " nMiniDoublets: " << nMiniDoublets << std::endl;
  std::cout << " totOccupancyMiniDoublets (including trucated ones): " << totOccupancyMiniDoublets << std::endl;
}

//________________________________________________________________________________________________________________________________
void printAllObjects(lst::Event<Acc3D>* event) {
  printMDs(event);
  printLSs(event);
  printpLSs(event);
  printT3s(event);
}

//________________________________________________________________________________________________________________________________
void printMDs(lst::Event<Acc3D>* event) {
  lst::MiniDoublets const* miniDoublets = event->getMiniDoublets()->data();
  lst::Hits const* hitsEvt = event->getHits()->data();
  lst::Modules const* modules = event->getModules()->data();
  lst::ObjectRanges const* ranges = event->getRanges()->data();

  // Then obtain the lower module index
  for (unsigned int idx = 0; idx <= *(modules->nLowerModules); ++idx) {
    for (unsigned int iMD = 0; iMD < miniDoublets->nMDs[idx]; iMD++) {
      unsigned int mdIdx = ranges->miniDoubletModuleIndices[idx] + iMD;
      unsigned int LowerHitIndex = miniDoublets->anchorHitIndices[mdIdx];
      unsigned int UpperHitIndex = miniDoublets->outerHitIndices[mdIdx];
      unsigned int hit0 = hitsEvt->idxs[LowerHitIndex];
      unsigned int hit1 = hitsEvt->idxs[UpperHitIndex];
      std::cout << "VALIDATION 'MD': "
                << "MD"
                << " hit0: " << hit0 << " hit1: " << hit1 << std::endl;
    }
  }
}

//________________________________________________________________________________________________________________________________
void printLSs(lst::Event<Acc3D>* event) {
  lst::Segments const* segments = event->getSegments()->data();
  lst::MiniDoublets const* miniDoublets = event->getMiniDoublets()->data();
  lst::Hits const* hitsEvt = event->getHits()->data();
  lst::Modules const* modules = event->getModules()->data();
  lst::ObjectRanges const* ranges = event->getRanges()->data();

  int nSegments = 0;
  for (unsigned int i = 0; i < *(modules->nLowerModules); ++i) {
    unsigned int idx = i;  //modules->lowerModuleIndices[i];
    nSegments += segments->nSegments[idx];
    for (unsigned int jdx = 0; jdx < segments->nSegments[idx]; jdx++) {
      unsigned int sgIdx = ranges->segmentModuleIndices[idx] + jdx;
      unsigned int InnerMiniDoubletIndex = segments->mdIndices[2 * sgIdx];
      unsigned int OuterMiniDoubletIndex = segments->mdIndices[2 * sgIdx + 1];
      unsigned int InnerMiniDoubletLowerHitIndex = miniDoublets->anchorHitIndices[InnerMiniDoubletIndex];
      unsigned int InnerMiniDoubletUpperHitIndex = miniDoublets->outerHitIndices[InnerMiniDoubletIndex];
      unsigned int OuterMiniDoubletLowerHitIndex = miniDoublets->anchorHitIndices[OuterMiniDoubletIndex];
      unsigned int OuterMiniDoubletUpperHitIndex = miniDoublets->outerHitIndices[OuterMiniDoubletIndex];
      unsigned int hit0 = hitsEvt->idxs[InnerMiniDoubletLowerHitIndex];
      unsigned int hit1 = hitsEvt->idxs[InnerMiniDoubletUpperHitIndex];
      unsigned int hit2 = hitsEvt->idxs[OuterMiniDoubletLowerHitIndex];
      unsigned int hit3 = hitsEvt->idxs[OuterMiniDoubletUpperHitIndex];
      std::cout << "VALIDATION 'LS': "
                << "LS"
                << " hit0: " << hit0 << " hit1: " << hit1 << " hit2: " << hit2 << " hit3: " << hit3 << std::endl;
    }
  }
  std::cout << "VALIDATION nSegments: " << nSegments << std::endl;
}

//________________________________________________________________________________________________________________________________
void printpLSs(lst::Event<Acc3D>* event) {
  lst::Segments const* segments = event->getSegments()->data();
  lst::MiniDoublets const* miniDoublets = event->getMiniDoublets()->data();
  lst::Hits const* hitsEvt = event->getHits()->data();
  lst::Modules const* modules = event->getModules()->data();
  lst::ObjectRanges const* ranges = event->getRanges()->data();

  unsigned int i = *(modules->nLowerModules);
  unsigned int idx = i;  //modules->lowerModuleIndices[i];
  int npLS = segments->nSegments[idx];
  for (unsigned int jdx = 0; jdx < segments->nSegments[idx]; jdx++) {
    unsigned int sgIdx = ranges->segmentModuleIndices[idx] + jdx;
    unsigned int InnerMiniDoubletIndex = segments->mdIndices[2 * sgIdx];
    unsigned int OuterMiniDoubletIndex = segments->mdIndices[2 * sgIdx + 1];
    unsigned int InnerMiniDoubletLowerHitIndex = miniDoublets->anchorHitIndices[InnerMiniDoubletIndex];
    unsigned int InnerMiniDoubletUpperHitIndex = miniDoublets->outerHitIndices[InnerMiniDoubletIndex];
    unsigned int OuterMiniDoubletLowerHitIndex = miniDoublets->anchorHitIndices[OuterMiniDoubletIndex];
    unsigned int OuterMiniDoubletUpperHitIndex = miniDoublets->outerHitIndices[OuterMiniDoubletIndex];
    unsigned int hit0 = hitsEvt->idxs[InnerMiniDoubletLowerHitIndex];
    unsigned int hit1 = hitsEvt->idxs[InnerMiniDoubletUpperHitIndex];
    unsigned int hit2 = hitsEvt->idxs[OuterMiniDoubletLowerHitIndex];
    unsigned int hit3 = hitsEvt->idxs[OuterMiniDoubletUpperHitIndex];
    std::cout << "VALIDATION 'pLS': "
              << "pLS"
              << " hit0: " << hit0 << " hit1: " << hit1 << " hit2: " << hit2 << " hit3: " << hit3 << std::endl;
  }
  std::cout << "VALIDATION npLS: " << npLS << std::endl;
}

//________________________________________________________________________________________________________________________________
void printT3s(lst::Event<Acc3D>* event) {
  lst::Triplets const* triplets = event->getTriplets()->data();
  lst::Segments const* segments = event->getSegments()->data();
  lst::MiniDoublets const* miniDoublets = event->getMiniDoublets()->data();
  lst::Hits const* hitsEvt = event->getHits()->data();
  lst::Modules const* modules = event->getModules()->data();
  int nTriplets = 0;
  for (unsigned int i = 0; i < *(modules->nLowerModules); ++i) {
    // unsigned int idx = modules->lowerModuleIndices[i];
    nTriplets += triplets->nTriplets[i];
    unsigned int idx = i;
    for (unsigned int jdx = 0; jdx < triplets->nTriplets[idx]; jdx++) {
      unsigned int tpIdx = idx * 5000 + jdx;
      unsigned int InnerSegmentIndex = triplets->segmentIndices[2 * tpIdx];
      unsigned int OuterSegmentIndex = triplets->segmentIndices[2 * tpIdx + 1];
      unsigned int InnerSegmentInnerMiniDoubletIndex = segments->mdIndices[2 * InnerSegmentIndex];
      unsigned int InnerSegmentOuterMiniDoubletIndex = segments->mdIndices[2 * InnerSegmentIndex + 1];
      unsigned int OuterSegmentOuterMiniDoubletIndex = segments->mdIndices[2 * OuterSegmentIndex + 1];

      unsigned int hit_idx0 = miniDoublets->anchorHitIndices[InnerSegmentInnerMiniDoubletIndex];
      unsigned int hit_idx1 = miniDoublets->outerHitIndices[InnerSegmentInnerMiniDoubletIndex];
      unsigned int hit_idx2 = miniDoublets->anchorHitIndices[InnerSegmentOuterMiniDoubletIndex];
      unsigned int hit_idx3 = miniDoublets->outerHitIndices[InnerSegmentOuterMiniDoubletIndex];
      unsigned int hit_idx4 = miniDoublets->anchorHitIndices[OuterSegmentOuterMiniDoubletIndex];
      unsigned int hit_idx5 = miniDoublets->outerHitIndices[OuterSegmentOuterMiniDoubletIndex];

      unsigned int hit0 = hitsEvt->idxs[hit_idx0];
      unsigned int hit1 = hitsEvt->idxs[hit_idx1];
      unsigned int hit2 = hitsEvt->idxs[hit_idx2];
      unsigned int hit3 = hitsEvt->idxs[hit_idx3];
      unsigned int hit4 = hitsEvt->idxs[hit_idx4];
      unsigned int hit5 = hitsEvt->idxs[hit_idx5];
      std::cout << "VALIDATION 'T3': "
                << "T3"
                << " hit0: " << hit0 << " hit1: " << hit1 << " hit2: " << hit2 << " hit3: " << hit3 << " hit4: " << hit4
                << " hit5: " << hit5 << std::endl;
    }
  }
  std::cout << "VALIDATION nTriplets: " << nTriplets << std::endl;
}

//________________________________________________________________________________________________________________________________
void debugPrintOutlierMultiplicities(lst::Event<Acc3D>* event) {
  lst::TrackCandidates const* trackCandidates = event->getTrackCandidates()->data();
  lst::Triplets const* triplets = event->getTriplets()->data();
  lst::Segments const* segments = event->getSegments()->data();
  lst::MiniDoublets const* miniDoublets = event->getMiniDoublets()->data();
  lst::Modules const* modules = event->getModules()->data();
  lst::ObjectRanges const* ranges = event->getRanges()->data();
  //int nTrackCandidates = 0;
  for (unsigned int idx = 0; idx <= *(modules->nLowerModules); ++idx) {
    if (trackCandidates->nTrackCandidates[idx] > 50000) {
      std::cout << " modules->detIds[modules->lowerModuleIndices[idx]]: " << modules->detIds[idx] << std::endl;
      std::cout << " idx: " << idx
                << " trackCandidates->nTrackCandidates[idx]: " << trackCandidates->nTrackCandidates[idx] << std::endl;
      std::cout << " idx: " << idx << " triplets->nTriplets[idx]: " << triplets->nTriplets[idx] << std::endl;
      unsigned int i = idx;  //modules->lowerModuleIndices[idx];
      std::cout << " idx: " << idx << " i: " << i << " segments->nSegments[i]: " << segments->nSegments[i] << std::endl;
      int nMD = miniDoublets->nMDs[2 * idx] + miniDoublets->nMDs[2 * idx + 1];
      std::cout << " idx: " << idx << " nMD: " << nMD << std::endl;
      int nHits = 0;
      nHits += ranges->hitRanges[4 * idx + 1] - ranges->hitRanges[4 * idx] + 1;
      nHits += ranges->hitRanges[4 * idx + 3] - ranges->hitRanges[4 * idx + 2] + 1;
      std::cout << " idx: " << idx << " nHits: " << nHits << std::endl;
    }
  }
}

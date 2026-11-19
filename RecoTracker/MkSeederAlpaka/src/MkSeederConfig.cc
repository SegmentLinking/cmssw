#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederConfig.h"

namespace mkseeder {

  namespace {
    using Json = nlohmann::json;

    // the value of key in j, if present (a JSON number read as T, as nlohmann converts it)
    template <typename T>
    void read(const Json &j, const char *key, T &value) {
      if (j.contains(key))
        value = j.at(key).get<T>();
    }

    // a removed option: absent, or set to its "off" value (isSupported)
    template <typename T, typename F>
    void requireSupported(const Json &j, const char *key, F isSupported) {
      if (j.contains(key) && !isSupported(j.at(key).get<T>()))
        throw cms::Exception("Configuration") << "MkSeederConfig: option '" << key << "' = " << j.at(key).dump()
                                              << " is not supported (only its off value)";
    }
  }  // namespace

  MkSeederConfig readMkSeederConfig(const std::string &jsonFile) {
    std::ifstream file(jsonFile);
    if (!file)
      throw cms::Exception("Configuration") << "MkSeederConfig: cannot open '" << jsonFile << "'";
    Json j;
    try {
      j = Json::parse(file);
    } catch (const std::exception &e) {
      throw cms::Exception("Configuration") << "MkSeederConfig: cannot parse '" << jsonFile << "': " << e.what();
    }
    MkSeederConfig c;
    try {
      read(j, "pt_min", c.ptMin);
      read(j, "d0_max", c.d0Max);
      read(j, "zv", c.zVertexHalfWidth);
      read(j, "zv_outer", c.zVertexHalfWidthOuter);
      read(j, "zv_outer_rmin", c.zVertexOuterMinRadius);
      read(j, "zv_outer_cotmin", c.zVertexOuterMinAbsCot);
      read(j, "zv_outer_disk", c.zVertexOuterDiscOnly);
      read(j, "marg_b", c.doubletPhiMargin);
      read(j, "phi_c", c.phiWindowC);
      read(j, "q_c", c.qWindowC);
      read(j, "phi_d", c.phiWindowD);
      read(j, "q_d", c.qWindowD);
      read(j, "win_scale", c.windowScale);
      if (j.contains("patterns"))
        for (const Json &jp : j.at("patterns")) {
          MkSeederConfig::Pattern p;
          read(jp, "layers", p.layers);
          read(jp, "win", p.windows);
          read(jp, "bwin", p.dWindowSlopes);
          read(jp, "sref", p.referencePathLength);
          if (jp.contains("eta_win"))
            for (const Json &je : jp.at("eta_win")) {
              MkSeederConfig::EtaWindow w;
              read(je, "lo", w.absEtaLow);
              read(je, "hi", w.absEtaHigh);
              read(je, "aphi", w.phiWindowA);
              read(je, "bphi", w.phiWindowB);
              read(je, "aq", w.qWindowA);
              read(je, "bq", w.qWindowB);
              p.etaWindows.push_back(w);
            }
          c.patterns.push_back(p);
        }

      read(j, "max_holes", c.maxHoles);
      read(j, "max_holes_ot", c.maxHolesForOuterTracker);
      read(j, "hole_always", c.alwaysForwardMisses);
      read(j, "miss_gap_dz", c.missGapDz);
      read(j, "trip_net_c", c.stageCNetCuts);
      read(j, "any_combination", c.anyCombination);
      read(j, "start_holes", c.maxStartHoles);
      read(j, "lead_only", c.leadOnly);
      read(j, "inner_ot_only", c.innerOuterTrackerOnly);
      read(j, "start_gap", c.startGapMode);
      read(j, "crossing_margin", c.crossingMargin);
      read(j, "gap_map_margin", c.gapMapMargin);
      read(j, "chain_ot", c.chainOuterTrackerLayers);
      read(j, "ot_any", c.allowAnyOuterTracker);
      read(j, "ot_max_pix", c.maxPixelLayersForExtraOuterTracker);
      if (j.contains("prune"))
        for (const Json &jp : j.at("prune")) {
          std::vector<int> layers = jp.get<std::vector<int>>();
          if (layers.size() != 3 && layers.size() != 4)
            throw cms::Exception("Configuration") << "MkSeederConfig: prune: a combination of 3 or 4 layers";
          c.prunedCombinations.push_back(layers);
        }

      read(j, "fk_score", c.fakeScoreCut);
      read(j, "fk_score_fwd", c.fakeScoreCutForward);
      read(j, "fk_eta_fwd", c.forwardMinAbsEta);
      read(j, "fk_shape", c.useShapeCut);
      if (j.contains("shape_win"))
        for (const Json &js : j.at("shape_win")) {
          MkSeederConfig::ShapeWindow w;
          read(js, "layer", w.layer);
          read(js, "bin_width", w.binWidth);
          read(js, "lo", w.low);
          read(js, "hi", w.high);
          if (w.layer < 0 || w.layer > 3 || w.binWidth <= 0 || w.low.empty() || w.low.size() != w.high.size())
            throw cms::Exception("Configuration") << "MkSeederConfig: bad shape_win for layer " << w.layer;
          c.shapeWindows.push_back(w);
        }
      read(j, "fk_ot2", c.ot2WindowScale);
      read(j, "ot2_win", c.ot2Windows);
      read(j, "ot2_phimin", c.ot2PhiWindowMin);
      read(j, "fk_ot1", c.ot1WindowScale);
      read(j, "fk_ot1_eta", c.ot1MaxAbsEta);
      read(j, "fk_ot1_gate", c.ot1ConfirmationGate);
      read(j, "fk_ot2_sig", c.ot2ConfirmationSigma);
      read(j, "fk_ot2_gate", c.ot2ConfirmationGate);
      read(j, "fk_ot2_local", c.ot2LocalWindowScale);
      read(j, "fk_ot1_lsig", c.ot1LocalSigma);
      read(j, "fk_ot2_lsig", c.ot2LocalSigma);
      read(j, "fk_ot_lsx", c.otLocalSigmaX);
      read(j, "fk_ot1_lsy", c.ot1LocalSigmaY);
      read(j, "fk_ot2_lsy", c.ot2LocalSigmaY);
      read(j, "fk_ot2_lgate", c.ot2LocalGateSigma);
      read(j, "triplet_seeds", c.emitTriplets);
      read(j, "trip_cot_lo", c.tripletMinAbsCot);
      read(j, "trip_cot_hi", c.tripletMaxAbsCot);

      read(j, "dedup", c.quadMinSharedHitsToDrop);
      read(j, "dedup_inner", c.innermostFirst);
      read(j, "trip_clean_min", c.tripletMinSharedHitsToDrop);
      read(j, "trip_veto_quad", c.tripletVetoBySingleQuad);
      read(j, "trip_veto_fk", c.tripletVetoMaxQuadFakeScore);
      read(j, "trip_veto_tsc", c.tripletVetoMaxTripletScore);
      read(j, "min_pix", c.minPixelHits);
      read(j, "min_pix_ot", c.minPixelHitsOuterTrackerThird);
      read(j, "clf_lorentz", c.classifierLorentzSizeX);
      read(j, "d0_max_ot", c.d0MaxOuterTrackerThird);
      read(j, "trip_dnn", c.tripletMlpCut);
      read(j, "trip_dnn_ext", c.tripletMlpRescueOuterTrackerHits);
      read(j, "trip_dnn_ext_min", c.tripletMlpRescueMinScore);

      read(j, "dup_sig", c.dupMaxSigma);
      read(j, "dup_shared", c.dupMinSharedHits);
      read(j, "dup_shared_trip", c.dupMinSharedHitsTriplet);
      read(j, "dup_ns_trip", c.dupNoShareTriplets);
      read(j, "dup_dphi", c.dupMaxDPhi);
      read(j, "dup_dz", c.dupMaxDZ);
      read(j, "dup_hpt_pt", c.dupHighPtThreshold);
      read(j, "dup_hpt_dphi0", c.dupHighPtDPhi0);
      read(j, "dup_hpt_dphi1", c.dupHighPtDPhi1);
      read(j, "dup_hpt_dz", c.dupHighPtDZ);
      read(j, "dup_merge", c.dupMergeMode);
      read(j, "dup_merge_ot", c.dupMergeOuterTracker);
      read(j, "dup_merge_ot_sig", c.dupMergeOuterTrackerSigma);
      read(j, "dup_merge_ot_score", c.dupMergeOuterTrackerMaxScore);
      read(j, "dup_merge_ot_pass", c.dupMergeOuterTrackerPasses);
      read(j, "dup_noshare_eta", c.dupNoShareMinAbsEta);
      read(j, "dup_deta", c.dupNoShareMaxDEta);
      read(j, "dup_ns_sig", c.dupNoShareMaxSigma);
      read(j, "dup_ns_dphi", c.dupNoShareMaxDPhi);
      read(j, "dup_ns_dz", c.dupNoShareMaxDZ);
      read(j, "dup_ns_merge", c.dupNoShareMerge);
      read(j, "dup_ns_res", c.dupNoShareMaxResidual);

      read(j, "ps_mark", c.psMark.mode);
      read(j, "ps_pt", c.psMark.ptMin);
      read(j, "ps_d0", c.psMark.d0Max);
      read(j, "ps_w0", c.psMark.residualPad);
      read(j, "ps_hl", c.psMark.halfLength);
      read(j, "ps_dv", c.psMark.lengthPad);
      read(j, "ps_cnt_pt", c.psMark.countPtMin);
      read(j, "ps_mrg_n", c.psMark.mergedExcess);
      read(j, "ps_low_pt", c.psMark.lowPtMin);
      read(j, "ps_ot1", c.psRequireOT1);
      read(j, "ps_ot2", c.psRequireOT2);
      read(j, "ps_chain", c.psRequireChain);
      read(j, "ps_ext", c.psRequireExtension);
      read(j, "ps_curv", c.psCurvatureSigma);
      read(j, "ps_curv_chain", c.psCurvatureSigmaChain);
      read(j, "ps_curv_res", c.psCurvatureResolution);
      read(j, "ps_alt", c.psAlternatives);
      read(j, "ps_seed", c.psSeed);
      read(j, "ps_seed_pt", c.psSeedPtMin);
      read(j, "ps_seed_curv", c.psSeedSigma);
      read(j, "ps_seed_rng", c.psSeedRange);
      read(j, "ps_c2", c.psStageC2);

      read(j, "ext_ot", c.extendIntoOuterTracker);
      read(j, "ext_sig", c.extensionSigma);
      read(j, "ext_nlay", c.extensionNLayers);
      read(j, "ext_any", c.extensionAnyLayer);
      read(j, "ext_min_hits", c.extensionMinHits);
      read(j, "ext_local", c.extensionLocalWindows);
      read(j, "ext_wx", c.extensionWindowX);
      read(j, "ext_wy", c.extensionWindowY);
      read(j, "trip_keep_noot", c.keepTripletsWithoutOuterTracker);
      read(j, "ext_branch", c.extensionBranches);
      read(j, "ext_redo", c.extensionRedo);
      read(j, "ext_branch_maxpix", c.extensionBranchMaxPixelHits);
      read(j, "ext_min_ot", c.extensionMinOuterTrackerHits);
      read(j, "ext_min_ot_mk", c.extensionMinOuterTrackerHitsMarked);
      read(j, "ext_path_bonus", c.extensionPathBonus);

      read(j, "pix_fill", c.pixelFillMode);
      read(j, "pix_fill_ov", c.pixelFillOverlaps);
      read(j, "pix_fill_sig", c.pixelFillSigma);
      read(j, "pix_fill_sig_disc", c.pixelFillSigmaDisc);
      read(j, "pix_fill_w", c.pixelFillWeights);

      read(j, "mrg_tag", c.mergedTagMode);
      read(j, "mrg_q_mpv", c.mergedChargeMpv);
      read(j, "mrg_q", c.mergedChargeCut);
      read(j, "mrg_dsx", c.mergedSizeXCut);
      read(j, "mrg_dsy", c.mergedSizeYCut);
      read(j, "mrg_err", c.mergedErrorScale);
      read(j, "mrg_win", c.mergedWindowScale);
      read(j, "mrg_share", c.mergedNoShare);
      read(j, "mrg_shape", c.mergedShapeExempt);

      if (j.contains("fit")) {
        const Json &jf = j.at("fit");
        read(jf, "prior_sigma", c.fitPriorSigma);
        read(jf, "prior_scale", c.fitPriorScale);
        read(jf, "bl_err_rescale", c.fitBeamlineErrorRescale);
        // the fit of the working point: mode 1 (the Kalman fit over the hits), the position not taken from the
        // first hit, no beam-line state test
        requireSupported<int>(jf, "mode", [](int v) { return v == 1; });
        requireSupported<int>(jf, "pos_from_hit0", [](int v) { return v == 0; });
        requireSupported<int>(jf, "beamline_state", [](int v) { return v == 0; });
      }

      // the removed options: only their off values
      auto isZero = [](int v) { return v == 0; };
      auto isNotPositive = [](float v) { return v <= 0.f; };
      for (const char *key : {"d_mode",
                              "max_consec_holes",
                              "gap_holes",
                              "maybe_holes",
                              "trip_unavoid",
                              "dedup_helix",
                              "c_win_mode",
                              "ext_max_pix",
                              "dup_hpt_shared",
                              "dup_hpt_ns"})
        requireSupported<int>(j, key, isZero);
      for (const char *key : {"trip_max_score",
                              "dedup_res",
                              "tier_fk_score",
                              "tier_fk_score_fwd",
                              "c_asym_pt",
                              "c_pt_frac",
                              "c_d0_max",
                              "ext_sig3",
                              "dup_res"})
        requireSupported<float>(j, key, isNotPositive);
      requireSupported<float>(j, "ext_eta", [](float v) { return v >= 10.f; });
      requireSupported<float>(j, "ext_wide", [](float v) { return v <= 1.f; });
      // the second duplicate cleaning, after the pixel fill, is part of the seeder
      requireSupported<int>(j, "pix_fill_dedup", [](int v) { return v == 1; });
      if (!j.contains("pix_fill_dedup"))
        throw cms::Exception("Configuration") << "MkSeederConfig: pix_fill_dedup 1 is required";
    } catch (const nlohmann::json::exception &e) {  // try
      throw cms::Exception("Configuration") << "MkSeederConfig: bad value in '" << jsonFile << "': " << e.what();
    }
    if (c.dupMaxSigma <= 0.f)
      throw cms::Exception("Configuration") << "MkSeederConfig: dup_sig > 0 is required";
    if (c.pixelFillWeights.size() != 4 || c.extensionWindowX.size() != 2 || c.extensionWindowY.size() != 2)
      throw cms::Exception("Configuration") << "MkSeederConfig: pix_fill_w needs 4 values, ext_wx and ext_wy 2";
    if (c.psMark.mode == 0 && (c.psRequireOT1 || c.psRequireOT2 || c.psRequireChain || c.psRequireExtension ||
                               c.psCurvatureSigma > 0 || c.psCurvatureSigmaChain > 0 || c.psSeed || c.psStageC2))
      throw cms::Exception("Configuration") << "MkSeederConfig: the ps_* selections need ps_mark 1";
    if (c.psMark.mode != 0 && (c.psMark.ptMin <= 0.f || c.psMark.halfLength <= 0.f))
      throw cms::Exception("Configuration") << "MkSeederConfig: ps_pt and ps_hl must be > 0";
    return c;
  }  // readMkSeederConfig()

  MkSeederTripletMlpWeights readMkSeederTripletMlpWeights(const std::string &jsonFile) {
    using W = MkSeederTripletMlpWeights;
    std::ifstream file(jsonFile);
    if (!file)
      throw cms::Exception("Configuration") << "MkSeederTripletMlpWeights: cannot open '" << jsonFile << "'";
    W w;
    try {
      const Json j = Json::parse(file);
      if (j.at("nFeatures").get<int>() != W::kNFeatures || j.at("nHidden1").get<int>() != W::kNHidden1 ||
          j.at("nHidden2").get<int>() != W::kNHidden2)
        throw cms::Exception("Configuration")
            << "MkSeederTripletMlpWeights: '" << jsonFile << "' is not a " << W::kNFeatures << "-" << W::kNHidden1
            << "-" << W::kNHidden2 << "-1 network";
      // the numbers as the floats they are (a JSON number is read as a double, then converted)
      auto readArray = [&j, &jsonFile](const char *key, float *out, size_t n) {
        const std::vector<float> v = j.at(key).get<std::vector<float>>();
        if (v.size() != n)
          throw cms::Exception("Configuration") << "MkSeederTripletMlpWeights: '" << jsonFile << "': " << key << " has "
                                                << v.size() << " values, not " << n;
        std::copy(v.begin(), v.end(), out);
      };
      readArray("featureMean", w.featureMean, W::kNFeatures);
      readArray("featureScale", w.featureScale, W::kNFeatures);
      readArray("hidden1Weights", w.hidden1Weights, W::kNFeatures * W::kNHidden1);
      readArray("hidden1Bias", w.hidden1Bias, W::kNHidden1);
      readArray("hidden2Weights", w.hidden2Weights, W::kNHidden1 * W::kNHidden2);
      readArray("hidden2Bias", w.hidden2Bias, W::kNHidden2);
      readArray("outputWeights", w.outputWeights, W::kNHidden2);
      w.outputBias = j.at("outputBias").get<float>();
      if (j.contains("zFold"))
        w.zFold = j.at("zFold").get<float>();
      if (w.zFold != 0.f && w.zFold != 1.f && w.zFold != -1.f)
        throw cms::Exception("Configuration")
            << "MkSeederTripletMlpWeights: '" << jsonFile << "': zFold must be 0, 1 or -1";
    } catch (const nlohmann::json::exception &e) {
      throw cms::Exception("Configuration") << "MkSeederTripletMlpWeights: bad file '" << jsonFile << "': " << e.what();
    }
    return w;
  }

}  // namespace mkseeder

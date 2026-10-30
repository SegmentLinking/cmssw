#ifndef RecoTracker_MkFitAlpaka_interface_cands_DupCleanParams_h
#define RecoTracker_MkFitAlpaka_interface_cands_DupCleanParams_h

#include <cstdint>

#include "RecoTracker/MkFitAlpaka/interface/es/ESConfig.h"

namespace mkfitdev::clean {

  // Parameters of the duplicate cleaner: the IterationConfig dc_* values of the iteration (dc_fracSharedHits and
  // dc_drth_* from the iteration JSON, the pT-adaptive ones from MkFitIterationConfigESProducer) and the event's beam
  // spot.
  // pixelPriority: phase2:clean_duplicates_sharedhits_pixelpriority instead of
  // phase1:clean_duplicates_sharedhits_pixelseed; pixelLayers = TrackerInfo is_pixel() per layer (bit l of word
  // l / 64), zero for the pixelseed cleaner.
  // ptAdaptive: phase2:clean_duplicates_sharedhits_pixelpriority_ptadaptive (pixelPriority is set too).
  struct DupCleanParams {
    float fracSharedHits;
    float drthCentral;
    float drthObarrel;
    float drthForward;
    int pixelPriority = 0;
    uint64_t pixelLayers[4] = {0, 0, 0, 0};
    int ptAdaptive = 0;
    // MkFitCore compares only the pairs that share a hit: the pT-adaptive cleaner with every dc_drth_* <= 0
    int sharedHitPairsOnly = 0;
    float fracSharedHitsLowPt = 0.f;
    float invPtRampStart = 0.f;  // 1 / dc_lowPtRampStart
    float invPtRampEnd = 0.f;    // 1 / dc_lowPtRampEnd
    float lowPtMaxRelDiffInvPt = 0.f;
    bool lowPtRequireSameCharge = false;
    float lowPtMaxD0 = 0.f;
    float lowPtMaxDRTimesPt = 0.f;
    int minUniqueHitsToKeep = 0;
    float minPtUniqueHitsToKeep = 0.f;
    // EventOfHits beam spot (MkFitEventOfHitsProducer: mkfit::BeamSpot(bs.x0(), bs.y0(), ...)), set per event
    float bsX = 0.f;
    float bsY = 0.f;
  };

  // The cleaner of the iteration config and its parameters (as clean_duplicates_sharedhits_pixelseed_impl reads them).
  inline DupCleanParams makeDupCleanParams(ESConfig const& cfg) {
    DupCleanParams params{cfg.dc_fracSharedHits, cfg.dc_drth_central, cfg.dc_drth_obarrel, cfg.dc_drth_forward};
    const bool ptAdaptive = cfg.duplicate_cleaner == DuplicateCleaner::SharedHitsPixelPriorityPtAdaptive;
    if (ptAdaptive || cfg.duplicate_cleaner == DuplicateCleaner::SharedHitsPixelPriority) {
      params.pixelPriority = 1;
      for (int word = 0; word < 4; ++word)
        params.pixelLayers[word] = cfg.pixel_layer_mask[word];
    }
    if (ptAdaptive) {
      params.ptAdaptive = 1;
      params.sharedHitPairsOnly =
          cfg.dc_drth_central <= 0.f && cfg.dc_drth_obarrel <= 0.f && cfg.dc_drth_forward <= 0.f;
      params.fracSharedHitsLowPt = cfg.dc_fracSharedHitsLowPt;
      params.invPtRampStart = 1.f / cfg.dc_lowPtRampStart;
      params.invPtRampEnd = 1.f / cfg.dc_lowPtRampEnd;
      params.lowPtMaxRelDiffInvPt = cfg.dc_lowPtMaxRelDiffInvPt;
      params.lowPtRequireSameCharge = cfg.dc_lowPtRequireSameCharge;
      params.lowPtMaxD0 = cfg.dc_lowPtMaxD0;
      params.lowPtMaxDRTimesPt = cfg.dc_lowPtMaxDRTimesPt;
      params.minUniqueHitsToKeep = cfg.dc_minUniqueHitsToKeep;
      params.minPtUniqueHitsToKeep = cfg.dc_minPtUniqueHitsToKeep;
    }
    return params;
  }

}  // namespace mkfitdev::clean

#endif

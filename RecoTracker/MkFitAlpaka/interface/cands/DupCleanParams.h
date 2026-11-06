#ifndef RecoTracker_MkFitAlpaka_interface_cands_DupCleanParams_h
#define RecoTracker_MkFitAlpaka_interface_cands_DupCleanParams_h

#include <cstdint>

namespace mkfitdev::clean {

  // Parameters of the duplicate cleaner: the IterationConfig dc_* values of the iteration (from the iteration JSON).
  // pixelPriority: phase2:clean_duplicates_sharedhits_pixelpriority instead of
  // phase1:clean_duplicates_sharedhits_pixelseed; pixelLayers = TrackerInfo is_pixel() per layer (bit l of word l / 64),
  // zero for the pixelseed cleaner.
  struct DupCleanParams {
    float fracSharedHits;
    float drthCentral;
    float drthObarrel;
    float drthForward;
    int pixelPriority = 0;
    uint64_t pixelLayers[4] = {0, 0, 0, 0};
  };

}  // namespace mkfitdev::clean

#endif

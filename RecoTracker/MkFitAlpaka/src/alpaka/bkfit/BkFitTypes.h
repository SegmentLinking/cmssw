#ifndef RecoTracker_MkFitAlpaka_src_alpaka_bkfit_BkFitTypes_h
#define RecoTracker_MkFitAlpaka_src_alpaka_bkfit_BkFitTypes_h

// Backward-fit parameters shared by the launcher (BkFitLaunch.h) and the kernels.

#include <cstdint>

namespace mkfitdev::bkfit {

  // backward-fit outlier rejection (IterationConfig m_backward_fit_outlier_chi2 / max_outliers /
  // outlier_min_pt): chi2 <= 0 = off.
  struct OutlierParams {
    float chi2 = 0.f;
    int maxOutliers = 0;
    float minPt = 0.f;
  };

}  // namespace mkfitdev::bkfit

#endif

#ifndef RecoTracker_LSTCore_interface_OTHitCovariance_h
#define RecoTracker_LSTCore_interface_OTHitCovariance_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "DataFormats/TrackingRecHitSoA/interface/Phase2OTRecHitsSoA.h"

namespace lst {

  // the global covariance (xx, xy, yy, xz, yz, zz) of a hit with local variances varX and varY, uncorrelated, on a
  // module of frame `frame`
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void hitCovariance(SOAFrame<float> const& frame,
                                                         float varX,
                                                         float varY,
                                                         float cov[6]) {
    frame.toGlobal(varX, 0.f, varY, cov);
  }

  // What the global covariance of an OT hit of the LST input is computed from, where it is needed rather than stored
  // for every OT row: the OT rechits, whose row r is the OT row r of the LST input, and the module frames, one per
  // GeomDetUnit::index() (the rechits' detectorIndex). Views of device memory for a device computation.
  struct OTHitCovarianceInputs {
    ::reco::Phase2OTRecHitsConstView otRecHits;
    SOAFrame<float> const* frames;

    // the global covariance of the OT hit of row `row`, as the device LST input computes it for the fits of the
    // pixel tracks
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void covariance(uint32_t row, float cov[6]) const {
      hitCovariance(
          frames[otRecHits.detectorIndex()[row]], otRecHits.xerrLocal()[row], otRecHits.yerrLocal()[row], cov);
    }
  };

}  // namespace lst

#endif

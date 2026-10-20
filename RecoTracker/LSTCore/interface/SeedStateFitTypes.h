#ifndef RecoTracker_LSTCore_interface_SeedStateFitTypes_h
#define RecoTracker_LSTCore_interface_SeedStateFitTypes_h

#include <cstdint>

namespace lst {

  // One stored Broken Line Fit state at the outermost fitted node.
  struct BLFStateRow {
    float x, y, z;
    float phi, cotTheta, kappa;
    float covCircle[6];
    float covLine[3];
    int8_t charge;
  };

  // A hit as it enters the fit: global position (cm) and global covariance (cm^2) in the order of the Broken Line
  // Fit's hit errors: xx, xy, yy, xz, yz, zz.
  struct FitHit {
    float x, y, z;
    float cov[6];
  };

  // A fitted state as the seed-state product stores it.
  struct FittedState {
    ::lst::BLFStateRow row;
    uint32_t stateHit;    // the encoded hit the state sits on: lst::encodeStateHitOT or lst::encodeStateHitPixel
    uint8_t status;       // lst::SeedStateStatus
    uint8_t nFit;         // fitted hits, 0 if unfit
    uint8_t refFallback;  // the reference node of the outer chord is the first fitted node
  };

}  // namespace lst

#endif

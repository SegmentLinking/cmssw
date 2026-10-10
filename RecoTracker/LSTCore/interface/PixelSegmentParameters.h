#ifndef RecoTracker_LSTCore_interface_PixelSegmentParameters_h
#define RecoTracker_LSTCore_interface_PixelSegmentParameters_h

#include <algorithm>
#include <cmath>
#include <numbers>

#include "RecoTracker/LSTCore/interface/Common.h"

// How a pixel seed becomes a pLS of the LST input, shared by the host input (prepareInput) and the device input.
namespace lst {

  // the kinematics of a pixel seed that the pLS hits carry: the PCA position, the PCA momentum (pT, eta, phi), the
  // state position on the last hit, and the impact parameters
  struct PixelSegmentKinematics {
    float pcaX, pcaY, pcaZ;
    float ptPCA, etaPCA, phiPCA;
    float lastHitX, lastHitY, lastHitZ;
    float dxy, dz;
  };

  // pixel type of a seed with pT ptIn on its last hit; kInvalid: not a pLS
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE PixelType pixelSegmentType(float ptIn, float ptErr, float deltaPhi, float ptCut) {
    if (!(ptIn > ptCut - 2 * ptErr))
      return PixelType::kInvalid;
    if (ptIn >= 2.0f)
      return PixelType::kHighPt;
    return deltaPhi >= 0 ? PixelType::kLowPtPosCurv : PixelType::kLowPtNegCurv;
  }

  // the transverse and longitudinal impact parameters of the PCA (position, momentum, transverse momentum pt)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void pixelSegmentImpactParameters(float x,
                                                                        float y,
                                                                        float z,
                                                                        float px,
                                                                        float py,
                                                                        float pz,
                                                                        float pt,
                                                                        double beamSpotX,
                                                                        double beamSpotY,
                                                                        double beamSpotZ,
                                                                        float& dxy,
                                                                        float& dz) {
    dxy = (-(x - beamSpotX) * py + (y - beamSpotY) * px) / pt;
    dz = (z - beamSpotZ) - ((x - beamSpotX) * px + (y - beamSpotY) * py) / pt * (pz / pt);
  }

  // reco::TrackBase ptError and etaError from the perigee errors of (transverse curvature, theta)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void pixelSegmentMomentumErrors(double pt,
                                                                      double momentum,
                                                                      double pz,
                                                                      double charge,
                                                                      double curvatureVariance,
                                                                      double curvatureThetaCovariance,
                                                                      double thetaVariance,
                                                                      float& ptErr,
                                                                      float& etaErr) {
    double pt2 = pt * pt;
    double momentum2 = momentum * momentum;
    ptErr =
        std::sqrt(pt2 * momentum2 / (charge * charge) * curvatureVariance +
                  2.0 * std::sqrt(momentum2 * pt2) / charge * pz * curvatureThetaCovariance + pz * pz * thetaVariance);
    etaErr = std::sqrt(thetaVariance) * momentum / pt;
  }

  // the PCA position from the PCA momentum and the impact parameters
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void pixelSegmentPCAPosition(
      double px, double py, double pz, float dxy, float dz, float& pcaX, float& pcaY, float& pcaZ) {
    const float pt = std::sqrt(px * px + py * py);
    const float momentum = std::sqrt(px * px + py * py + pz * pz);
    pcaZ = dz * pt * pt / momentum / momentum;
    pcaX = -dxy * py / pt - px / momentum * pz / momentum * dz;
    pcaY = dxy * px / pt - py / momentum * pz / momentum * dz;
  }

  // the (eta, phi, dz) bin of the PCA that LST uses to pair pLS with outer-tracker segments
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int pixelSegmentSuperbin(float etaPCA, float phiPCA, float dz) {
    float neta = 25.;
    float nphi = 72.;
    float nz = 25.;
    int etabin = (etaPCA + 2.6) / ((2 * 2.6) / neta);
    int phibin = (phiPCA + std::numbers::pi_v<float>) / ((2. * std::numbers::pi_v<float>) / nphi);
    int dzbin = (std::clamp(dz, -30.f, 30.f) + 30) / (2 * 30 / nz);
    return (nz * nphi) * etabin + (nz)*phibin + dzbin;
  }

  // the seed hit stored in pLS hit slot `slot` of `nSlots`: the first hits, and the last hit in the last slot
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE unsigned int pixelSegmentHitOfSlot(unsigned int slot,
                                                                         unsigned int nSlots,
                                                                         unsigned int nHits) {
    return slot + 1 == nSlots ? nHits - 1 : slot;
  }

  // the position columns of pLS hit slot `slot`: the PCA position, the PCA momentum, the last-hit position, then
  // (last-hit x, dxy, dz) for the remaining slots
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void pixelSegmentHitColumns(
      PixelSegmentKinematics const& kinematics, unsigned int slot, float& x, float& y, float& z) {
    if (slot == 0) {
      x = kinematics.pcaX;
      y = kinematics.pcaY;
      z = kinematics.pcaZ;
    } else if (slot == 1) {
      x = kinematics.ptPCA;
      y = kinematics.etaPCA;
      z = kinematics.phiPCA;
    } else if (slot == 2) {
      x = kinematics.lastHitX;
      y = kinematics.lastHitY;
      z = kinematics.lastHitZ;
    } else {
      x = kinematics.lastHitX;
      y = kinematics.dxy;
      z = kinematics.dz;
    }
  }

}  // namespace lst

#endif

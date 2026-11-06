#ifndef RecoTracker_MkFitCore_interface_portable_TrackStateJacobians_h
#define RecoTracker_MkFitCore_interface_portable_TrackStateJacobians_h

#include "RecoTracker/MkFitCore/interface/portable/Macros.h"

namespace mkfit::portable::inline MKFIT_PORTABLE_NAMESPACE {

  // TrackState::jacobianCCSToCurvilinear: d(q/p, lambda, phi, xT, yT) / d(x, y, z, 1/pT, phi, theta) of the CCS state,
  // into any 6x6 matrix with operator()(row, col) that is zero on input
  template <typename Matrix>
  MKFIT_HOST_DEVICE inline void jacobianCCSToCurvilinear(
      float invpt, float cosP, float sinP, float cosT, float sinT, short charge, Matrix& jac) {
    jac(3, 0) = -sinP;
    jac(4, 0) = -cosP * cosT;
    jac(3, 1) = cosP;
    jac(4, 1) = -sinP * cosT;
    jac(4, 2) = sinT;
    jac(0, 3) = charge * sinT;
    jac(0, 5) = charge * cosT * invpt;
    jac(1, 5) = -1.f;
    jac(2, 4) = 1.f;
  }

}  // namespace mkfit::portable::inline MKFIT_PORTABLE_NAMESPACE

#endif

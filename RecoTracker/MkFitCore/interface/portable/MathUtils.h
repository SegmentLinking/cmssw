#ifndef RecoTracker_MkFitCore_interface_portable_MathUtils_h
#define RecoTracker_MkFitCore_interface_portable_MathUtils_h

#include <cmath>

#include "RecoTracker/MkFitCore/interface/portable/Macros.h"
#include "RecoTracker/MkFitCore/interface/portable/Matriplex/MatriplexVdt.h"

namespace mkfit {

  MKFIT_HOST_DEVICE inline float hipo(float x, float y) { return std::sqrt(x * x + y * y); }

  MKFIT_HOST_DEVICE inline float hipo_sqr(float x, float y) { return x * x + y * y; }

  MKFIT_HOST_DEVICE inline void sincos4(const float x, float& sin, float& cos) {
    // Had this writen with explicit division by factorial.
    // The *whole* fitting test ran like 2.5% slower on MIC, sigh.

    const float x2 = x * x;
    cos = 1.f - 0.5f * x2 + 0.04166667f * x2 * x2;
    sin = x - 0.16666667f * x * x2;
  }

  MKFIT_HOST_DEVICE inline float getPhi(float x, float y) { return Matriplex::vdt::fast_atan2f(y, x); }

  MKFIT_HOST_DEVICE inline float getTheta(float r, float z) { return Matriplex::vdt::fast_atan2f(r, z); }

}  // end namespace mkfit

#endif

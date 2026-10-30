#ifndef RecoTracker_MkFitAlpaka_interface_math_Config_h
#define RecoTracker_MkFitAlpaka_interface_math_Config_h

// Portable copy of the constants and inline functions of RecoTracker/MkFitCore/interface/Config.h
// that propagation, Kalman update, hit binning and building use.
// Host-side configuration (threading, registries, duplicate-removal globals) is not ported.

#include <alpaka/core/Common.hpp>

namespace mkfitdev {

  namespace Const {
    constexpr float PI = 3.14159265358979323846;
    constexpr float TwoPI = 6.28318530717958647692;
    constexpr float sol = 0.299792458;  // speed of light in m/ns
    constexpr float sol_over_100 = 0.299792458e-2;
  }  // namespace Const

  ALPAKA_FN_HOST_ACC inline float cdist(float a) { return a > Const::PI ? Const::TwoPI - a : a; }

  namespace Config {
    constexpr int nMaxTrkHits = 64;  // Used for array sizes in MkFitter/Finder, max hits in toy MC

    // Config for propagation
    constexpr int Niter = 5;
    constexpr bool useTrigApprox = true;
    // for prop to plane getS step
    constexpr int nSStepsInProp2Plane = 2;
    // MkFitCore: runtime globals (Config.cc, false) that MkFitGeometryESProducer sets to true for any Phase-2
    // geometry (no MKFIT_PHASE2CUSTOMFLAGS in CMSSW builds). The port only targets Phase-2: constexpr true.
    constexpr bool usePropToPlane = true;
    constexpr bool usePtMultScat = true;

    // Config for Bfield. MkFitCore: Bz = (b0 z^2 + b1 z + c1) (a r^2 + 1) with two sets of runtime constants
    // (Config.cc defaults below; MkFitGeometryESProducer sets them):
    // - mag_*: track building, the constants of CMSSW's ParabolicMf (portableParabolicMagneticField::Parameters);
    // - mag_fit_*: the final fit (PropagationFlags::use_fit_b_field), fitted to the CMS field map (bFieldParams).
    // The device code takes both from the ES (MaterialView::bField and bFieldFit, filled from mkfit::Config by
    // esFill); these constexpr values are only the defaults (= Config.cc).
    constexpr float Bfield = 3.8112;
    constexpr float mag_c1 = 3.8114;
    constexpr float mag_b0 = -3.94991e-06;
    constexpr float mag_b1 = 7.53701e-06;
    constexpr float mag_a = 2.43878e-11;
    constexpr float mag_fit_c1 = 3.81036;
    constexpr float mag_fit_b0 = -2.03767e-06;
    constexpr float mag_fit_b1 = 7.34495e-06;
    constexpr float mag_fit_a = 3.01291e-07;

    // sorting config (bonus,penalty)
    constexpr float validHitBonus_ = 4;
    constexpr float validHitSlope_ = 0.2;
    constexpr float overlapHitBonus_ = 0;  // set to negative for penalty
    constexpr float missingHitPenalty_ = 8;
    constexpr float tailMissingHitPenalty_ = 3;

    // config on duplicate removal (MkFitCore: extern const in Config.cc)
    constexpr float maxd1pt = 1.8;     //windows for hit
    constexpr float maxdphi = 0.37;    //and/or dr
    constexpr float maxdcth = 0.37;    //comparisons
    constexpr float maxcth_ob = 1.99;  //eta 1.44
    constexpr float maxcth_fw = 6.05;  //eta 2.5

    // One set of runtime constants of Config::bFieldFromZR: mkfit::Config::mag_* (building, the defaults here) or
    // mkfit::Config::mag_fit_* (the final fit, Config::bFieldFromZRFit).
    struct BFieldParams {
      float c1 = Config::mag_c1;
      float b0 = Config::mag_b0;
      float b1 = Config::mag_b1;
      float a = Config::mag_a;
    };

    // Config::bFieldFromZR with the ES constants: same operations and order
    ALPAKA_FN_HOST_ACC inline float bFieldFromZR(const BFieldParams& m, const float z, const float r) {
      return (m.b0 * z * z + m.b1 * z + m.c1) * (m.a * r * r + 1.f);
    }

  }  // namespace Config

}  // namespace mkfitdev

#endif

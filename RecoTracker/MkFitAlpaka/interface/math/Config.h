#ifndef RecoTracker_MkFitAlpaka_interface_math_Config_h
#define RecoTracker_MkFitAlpaka_interface_math_Config_h

// Portable copy of the constants and inline functions of RecoTracker/MkFitCore/interface/Config.h
// (CMSSW_20_1_0_pre2) that propagation, Kalman update, hit binning and building use.
// Host-side configuration (threading, registries, duplicate-removal globals) is not ported.

#include <alpaka/core/Common.hpp>

namespace mkfitdev {

  namespace Const {
    constexpr float PI = 3.14159265358979323846;
    constexpr float TwoPI = 6.28318530717958647692;
    constexpr float PIOver2 = Const::PI / 2.0f;
    constexpr float PIOver4 = Const::PI / 4.0f;
    constexpr float PI3Over4 = 3.0f * Const::PI / 4.0f;
    constexpr float InvPI = 1.0f / Const::PI;
    constexpr float sol = 0.299792458;  // speed of light in m/ns
    constexpr float sol_over_100 = 0.299792458e-2;
  }  // namespace Const

  ALPAKA_FN_HOST_ACC inline float cdist(float a) { return a > Const::PI ? Const::TwoPI - a : a; }

  namespace Config {
    constexpr int nMaxTrkHits = 64;  // Used for array sizes in MkFitter/Finder, max hits in toy MC

    // This will become layer dependent (in bits). To be consistent with min_dphi.
    constexpr int m_nphi = 256;

    // Config for propagation
    constexpr int Niter = 5;
    constexpr bool useTrigApprox = true;
    // for prop to plane getS step
    constexpr int nSStepsInProp2Plane = 2;
    // MkFitCore: runtime globals (Config.cc, false) that MkFitGeometryESProducer sets to true for any Phase-2
    // geometry (no MKFIT_PHASE2CUSTOMFLAGS in CMSSW builds). The port only targets Phase-2: constexpr true.
    constexpr bool usePropToPlane = true;
    constexpr bool usePtMultScat = true;

    // Config for Bfield. MkFitCore: Bz = (mag_b0 z^2 + mag_b1 z + mag_c1) (mag_a r^2 + 1) with runtime
    // constants (Config.cc defaults below) that MkFitGeometryESProducer sets from its bFieldParams. The device code
    // takes them from the ES (BFieldParams in MaterialView, filled from mkfit::Config::mag_* by esFill); these
    // constexpr values are only the defaults (= Config.cc) and the 2-argument bFieldFromZR below.
    constexpr float Bfield = 3.8112;
    constexpr float mag_c1 = 3.81036;
    constexpr float mag_b0 = -2.03767e-06;
    constexpr float mag_b1 = 7.34495e-06;
    constexpr float mag_a = 3.01291e-07;

    // Config for SelectHitIndices (MkFitCore default, CONFIG_PhiQArrays not defined)
    constexpr bool usePhiQArrays = true;

    // sorting config (bonus,penalty)
    constexpr float validHitBonus_ = 4;
    constexpr float validHitSlope_ = 0.2;
    constexpr float overlapHitBonus_ = 0;  // set to negative for penalty
    constexpr float missingHitPenalty_ = 8;
    constexpr float tailMissingHitPenalty_ = 3;

    // config on seed cleaning
    constexpr float track1GeVradius = 87.6;  // = 1/(c*B)
    constexpr float c_etamax_brl = 0.9;
    constexpr float c_dpt_common = 0.25;
    constexpr float c_dzmax_brl = 0.005;
    constexpr float c_drmax_brl = 0.010;
    constexpr float c_ptmin_hpt = 2.0;
    constexpr float c_dzmax_hpt = 0.010;
    constexpr float c_drmax_hpt = 0.010;
    constexpr float c_dzmax_els = 0.015;
    constexpr float c_drmax_els = 0.015;

    // config on duplicate removal (MkFitCore: extern const in Config.cc)
    constexpr bool useHitsForDuplicates = true;
    constexpr float maxdPt = 0.5;
    constexpr float maxdPhi = 0.25;
    constexpr float maxdEta = 0.05;
    constexpr float maxdR = 0.0025;
    constexpr float minFracHitsShared = 0.75;

    constexpr float maxd1pt = 1.8;     //windows for hit
    constexpr float maxdphi = 0.37;    //and/or dr
    constexpr float maxdcth = 0.37;    //comparisons
    constexpr float maxcth_ob = 1.99;  //eta 1.44
    constexpr float maxcth_fw = 6.05;  //eta 2.5

    // ================================================================

    ALPAKA_FN_HOST_ACC inline float bFieldFromZR(const float z, const float r) {
      return (Config::mag_b0 * z * z + Config::mag_b1 * z + Config::mag_c1) * (Config::mag_a * r * r + 1.f);
    }

    // The runtime constants of Config::bFieldFromZR (mkfit::Config::mag_c1, mag_b0, mag_b1, mag_a).
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

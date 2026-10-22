#ifndef RecoTracker_MkFitCore_interface_Config_h
#define RecoTracker_MkFitCore_interface_Config_h

namespace mkfit {

  namespace Const {
    constexpr float PI = 3.14159265358979323846;
    constexpr float TwoPI = 6.28318530717958647692;
    constexpr float PIOver2 = Const::PI / 2.0f;
    constexpr float PIOver4 = Const::PI / 4.0f;
    constexpr float PI3Over4 = 3.0f * Const::PI / 4.0f;
    constexpr float InvPI = 1.0f / Const::PI;
    constexpr float sol = 0.299792458;  // speed of light in m/ns
    constexpr float sol_over_100 = 0.299792458e-2;

    // NAN and silly track parameter tracking options
    constexpr bool nan_etc_sigs_enable = false;

    constexpr bool nan_n_silly_check_seeds = true;
    constexpr bool nan_n_silly_print_bad_seeds = false;
    constexpr bool nan_n_silly_fixup_bad_seeds = false;
    constexpr bool nan_n_silly_remove_bad_seeds = true;

    constexpr bool nan_n_silly_check_cands_every_layer = false;
    constexpr bool nan_n_silly_print_bad_cands_every_layer = false;
    constexpr bool nan_n_silly_fixup_bad_cands_every_layer = false;

    constexpr bool nan_n_silly_check_cands_pre_bkfit = true;
    constexpr bool nan_n_silly_check_cands_post_bkfit = true;
    constexpr bool nan_n_silly_print_bad_cands_bkfit = false;
  }  // namespace Const

  inline float cdist(float a) { return a > Const::PI ? Const::TwoPI - a : a; }

  //------------------------------------------------------------------------------

  namespace Config {
    // config for fitting
    constexpr int nLayers = 10;  // default/toy: 10; cms-like: 18 (barrel), 27 (endcap)

    // Layer constants for common barrel / endcap.
    // TrackerInfo more or less has all this information.
    constexpr int nMaxTrkHits = 64;  // Used for array sizes in MkFitter/Finder, max hits in toy MC
    constexpr int nAvgSimHits = 32;  // Used for reserve() calls for sim hits/states

    // This will become layer dependent (in bits). To be consistent with min_dphi.
    static constexpr int m_nphi = 256;

    // Config for propagation - could/should enter into PropagationFlags?!
    constexpr int Niter = 5;
    constexpr bool useTrigApprox = true;
    // for prop to plane getS step
    constexpr int nSStepsInProp2Plane = 2;
    // Move to Config.cc, make a command-line option in mkFit.cc to ease profiling comparisons.
    // If making this constexpr again, also fix ifs using it in MkBuilder.cc and MkFinder.cc.
    // constexpr bool usePropToPlane = true;
    // constexpr bool usePtMultScat = true;
    extern bool usePropToPlane;
    extern bool usePtMultScat;

    // Config for Bfield.
    // bFieldFromZR() below: Bz = (mag_b0 z^2 + mag_b1 z + mag_c1) (mag_a r^2 + 1), the same form and
    // units as CMSSW's ParabolicParametrizedMagneticField.  The defaults (Config.cc) are fitted to the CMS
    // field map over the tracker volume (0.03 % rms; the map is the same in Run 3 and Phase 2); in CMSSW,
    // MkFitGeometryESProducer sets them from its parameter bFieldParams.  The older constants, still those
    // of CMSSW's ParabolicMf, are low by 1.46 % on average against that map (up to 4.7 %).
    constexpr float Bfield = 3.8112;
    extern float mag_c1;
    extern float mag_b0;
    extern float mag_b1;
    extern float mag_a;

    // Refit only (MkBuilder::fit_tracks); set by MkFitGeometryESProducer, off by default here.
    // refitBFieldAtMid: sample B at the chord midpoint of each propagate-to-plane step instead of at its
    //   start, so that the outward and inward propagations are inverses of each other.
    // refitRadialFieldCorr: correct each step for the radial field component Br = -(r/2) dBz/dz, which
    //   the constant-Bz helix neglects, antisymmetrically (half at each end of the step).
    extern bool refitBFieldAtMid;
    extern bool refitRadialFieldCorr;
    // refitElossSignFromPass: sign of the energy loss from the pass (forward loses, backward gains) instead of
    //   from the sign of each step's path length, which is wrong on every step the refit takes backwards.
    extern bool refitElossSignFromPass;
    // refitBkwMsFixedMomentum: multiple-scattering noise of the backward pass at the momentum of its start
    //   state (the forward result), fixed per track, instead of at the running estimate.
    extern bool refitBkwMsFixedMomentum;
    // refitBkwSubSteps: number of sub-steps of each propagation of the backward pass (1 = one step).
    extern int refitBkwSubSteps;
    // refitMaterialPerModule: material of each crossed module from its own MediumProperties (ModuleInfo) instead
    //   of the (|z|, r) grid.
    extern bool refitMaterialPerModule;

    // Config for SelectHitIndices
    // Use extra arrays to store phi and q of hits.
    // MT: This would in principle allow fast selection of good hits, if
    // we had good error estimates and reasonable *minimal* phi and q windows.
    // Speed-wise, those arrays (filling AND access, about half each) cost 1.5%
    // and could help us reduce the number of hits we need to process with bigger
    // potential gains.
#ifdef CONFIG_PhiQArrays
    extern bool usePhiQArrays;
#else
    constexpr bool usePhiQArrays = true;
#endif

    // sorting config (bonus,penalty)
    constexpr float validHitBonus_ = 4;
    constexpr float validHitSlope_ = 0.2;
    constexpr float overlapHitBonus_ = 0;  // set to negative for penalty
    constexpr float missingHitPenalty_ = 8;
    constexpr float tailMissingHitPenalty_ = 3;

    // Threading
#if defined(MKFIT_STANDALONE)
    extern int numThreadsFinder;
    extern int numThreadsEvents;
    extern int numSeedsPerTask;
#else
    constexpr int numThreadsFinder = 1;
    constexpr int numThreadsEvents = 1;
    constexpr int numSeedsPerTask = 32;
#endif

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

    // config on duplicate removal
#if defined(MKFIT_STANDALONE)
    extern bool useHitsForDuplicates;
    extern bool removeDuplicates;
#else
    const bool useHitsForDuplicates = true;
#endif
    extern const float maxdPhi;
    extern const float maxdPt;
    extern const float maxdEta;
    extern const float minFracHitsShared;
    extern const float maxdR;

    // duplicate removal: tighter version
    extern const float maxd1pt;
    extern const float maxdphi;
    extern const float maxdcth;
    extern const float maxcth_ob;
    extern const float maxcth_fw;

    // ================================================================

    inline float bFieldFromZR(const float z, const float r) {
      return (Config::mag_b0 * z * z + Config::mag_b1 * z + Config::mag_c1) * (Config::mag_a * r * r + 1.f);
    }

  };  // namespace Config

  //------------------------------------------------------------------------------

}  // end namespace mkfit
#endif

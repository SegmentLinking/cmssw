#ifndef RecoTracker_MkFitAlpaka_interface_seeds_alpaka_LstSeedFit_h
#define RecoTracker_MkFitAlpaka_interface_seeds_alpaka_LstSeedFit_h

// the seed state of the LST T5/T4/pT3/pT5 seeds from a device mkFit Kalman
// fit of the seed's own hits, instead of the host CMSSW seed creator (LSTOutputConverter makeSeed: FastHelix from
// two hits + the origin, KF with PropagatorWithMaterial) + MkFitSeedConverter. pLS seeds keep the copied pixel state.
// SWITCH-GATED (build module parameter lstSeedFit, default false); a DEVIATION candidate.
// Exported entry point (src/alpaka/LstSeeds.dev.cc); callers never instantiate the kernel (one TU per component).

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESView.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/SeedSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"

namespace mkfitdev::lstseeds {

  // Per-event device counters (int32; the caller zeroes them).
  struct LstSeedFitCounters {
    int32_t nPixelOnly;     // seeds with no outer-tracker hit (pLS, or pT3/pT5 whose host makeSeed fell back): kept
    int32_t nFitted;        // seeds whose state was replaced by the device fit
    int32_t nTooFewHits;    // < 3 hits with an OT hit (never expected for LST seeds): kept
    int32_t nFailed;        // fit failed (propagation fail flag or non-finite state/chi2): host state kept
    int32_t nChargeFlip;    // fitted charge != the host seed's charge (diagnostic)
    int32_t nOverflowHits;  // more than kMaxSeedHits hits (never: the packer truncates and counts)
    int32_t nRejDPhi;       // host-structure mode: a step turned by more than MaxDPhi (part of nFailed)
    int32_t nRejBackward;   // host-structure mode: a step against the momentum (part of nFailed)
    int32_t nSafeRetry;     // host-structure mode: float failure with the exact prior, refitted with the capped prior
    int32_t nFallback;      // host-structure mode, dropFailed: a failed seed with a pixel track (pT3 / pT5) replaced by
                            // its pixel seed's hits and state (as LSTOutputConverter keeps the pixel seed)
    int32_t nFallbackFailed;  // ... and no pixel seed state for it (seed dropped)
  };

  // Per-seed outcome of the host-structure fit (written to `st` when given, for the validation dump).
  enum LstSeedFitStatus : int8_t {
    kStNotFitted = -1,  // pixel-only / too few hits / not host-structure mode
    kStOk = 0,
    kStRejDPhi = 1,
    kStRejBackward = 2,
    kStRejInvalid = 3,
    kStFallbackOk = 4,      // failed (any reason), replaced by its pixel seed
    kStFallbackFailed = 5,  // failed, no pixel seed state (dropped)
    kStDroppedOT = 6        // failed T5/T4 (OT-only) seed dropped (dropFailed; host: TC skipped)
  };

  // originPrior value selecting the host-creator structure: FastHelix through the region
  // origin, the creator's initial errors, ONE forward KF pass, the creator's rejections.
  constexpr int kHostCreator = 3;

  // Seed classes (from the hit content; written to `cls` when given, for the validation dump).
  enum LstSeedClass : int8_t { kClsPixelOnly = 0, kClsOTOnly = 1, kClsMixed = 2, kClsTooFew = 3 };

  // Tunables of the device seed fit.
  struct LstSeedFitConfig {
    float posVar = 1.0f;          // initial variance of x, y, z (cm^2) at the first hit
    float relInvPtErr = 1.0f;     // initial sigma(1/pT) relative to the 3-hit estimate
    float invPtErrFloor = 0.01f;  // plus this absolute floor (1/GeV)
    float angVar = 0.01f;         // initial variance of phi and theta (rad^2)
    float errScale = 1.0f;        // final errors scaled by this factor (1 = the KF errors as they are)
    int passes = 3;               // 3 = forward, backward, forward (errors x100 in between; validated); 1 = one pass
    // Origin prior as the host seed creator (SeedFromConsecutiveHitsCreator with a default GlobalTrackingRegion:
    // KF started at the beam line, transverse sigma 0.2 cm, z sigma 22.7 cm, unconstrained 1/pT and angles):
    // 0 = none, 1 = OT-only seeds (T5/T4), 2 = every fitted seed; 3 = kHostCreator (option b': the host creator's
    // structure, the 3-hit settings above are not used; passes selects the field model: 3 = the final fit's
    // refitKernelFlags (mid-point B + radial kick), 1 = B at the start of each step without the kick, as the
    // host's AnalyticalPropagator).
    int originPrior = 0;
    float originR2 = 0.04f;    // (0.2 cm)^2
    float originZ2 = 515.29f;  // (22.7 cm)^2
    // kHostCreator: GlobalTrackingRegion() ptMin, SeedCreatorPSet MinOneOverPtError, PropagatorWithMaterial MaxDPhi
    float hlPtMin = 1.0f;
    float hlMinOneOverPtErr = 1.0f;
    float hlMaxDPhi = 1.6f;
    // kHostCreator retry after a float failure: capped angle (rad^2) and yT (cm^2) prior variances
    float hlSafeAngVar = 0.01f;
    float hlSafeYTVar = 25.f;
    // kHostCreator: a step counts as 'against the momentum' only below -hlBackwardTol (cm) along the direction (the
    // strict test rejected host-accepted pT3/pT5 seeds whose consecutive hits sit on the same or nearly the
    // same plane, where the host propagator does not move: AnalyticalPropagator 'already on surface'); 0 = the earlier version
    float hlBackwardTol = 0.01f;
    // kHostCreator + dropFailed: a failed seed with a pixel track (pT3/pT5) takes its pixel seed's hits and state
    // instead of being dropped (LSTOutputConverter.cc 'seeds.empty() ? seed : seeds[0]': the host keeps the pixel seed
    // when the creator fails on the pT3/pT5 hits); false = drop
    bool hlPixelFallback = true;
    bool dropFailed = false;  // failed fits: false = keep the input state; true = NaN error (removed by the
                              // seed_post_cleaning), for host seeds that carry only a placeholder state
  };

  // the pLS seed state = the host pixel-seed creator of the menu
  // (SeedGeneratorFromProtoTracksEDProducer, useProtoTrackKinematics False, includeFourthHit True ->
  // SeedFromConsecutiveHitsCreator::makeSeed with GlobalTrackingRegion(pT of the proto track, its vertex, 0.2, 0.2))
  // emulated on the device with fitHostLike, then TSCBLBuilderNoMaterial of the state on the last hit (the LST host
  // input LSTInputProducer) with interface/math/PcaToBeamLine.h. One row per pixel track.
  constexpr int kMaxPixSeedHits =
      64;  // hits per pixel track (16 dropped ~5 tracks per ttbar event; more: no pLS, counted)
  struct PixSeedIn {
    ::mkfitdev::HitOnTrack hot[kMaxPixSeedHits];  // (mkFit layer, hit index) in the creator's order (radius)
    int32_t nh;                                   // < 2: not fitted (status kPixNotFitted)
    float vx, vy;                                 // region origin = the proto track's vertex (FastHelix vertex)
    float ptMin;                                  // region ptMin = the proto track's pT
  };
  enum PixSeedStatus : int32_t {
    kPixNotFitted = -1,
    kPixOk = 0,
    kPixRejDPhi = 1,      // creator failure: a step turned by more than MaxDPhi
    kPixRejBackward = 2,  // creator failure: a step against the momentum
    kPixRejInvalid = 3    // creator failure: invalid propagation / update / non-finite state
  };
  struct PixSeedOut {
    int32_t status;     // PixSeedStatus
    int32_t charge;     // fitted charge (kPixOk)
    int32_t pcaStatus;  // mkfitdev::pca::PcaStatus (0 ok, 1 invalid TSCBL on the host, 2 outside the field box)
    float par[6];       // KF state on the last hit, mkFit CCS (x, y, z, 1/pT, phi, theta)
    float err[21];      // its errors (packed lower triangle, TrackSoA order)
    float pcaX, pcaY, pcaZ, pcaPx, pcaPy, pcaPz;  // TSCBL state at the PCA (GlobalPoint / GlobalVector, float)
    float ptErr, etaErr;  // LSTInputProducer: reco::TrackBase ptError / etaError formulas on the TSCBL perigee error
  };
  struct PixSeedBeam {
    float x, y, z, dxdz, dydz;  // reco::BeamSpot position and slopes (TSCBLBuilderNoMaterial's beam line)
  };

  // the pixel seed of each seed row: row pixIdx[s] >= 0 of `states` (hltInputLSTDevice:pixelSeedStates, per pixel track
  // the device creator's seed hits and state, charge 0 = no state); pixIdx == nullptr: no pixel seeds
  struct PixelSeedRows {
    int32_t const* pixIdx = nullptr;
    ::mkfitdev::TrackSoAConstView states;
    int32_t nStates = 0;
  };

}  // namespace mkfitdev::lstseeds

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstseeds {

  using ::mkfitdev::lstseeds::LstSeedFitConfig;
  using ::mkfitdev::lstseeds::LstSeedFitCounters;

  // Replaces params / errors / charge of every seed row s < n with at least one outer-tracker hit by the device fit
  // of its hits (positions + errors from the event HitSoA: pixel rows first, strip rows from nPixel; module planes
  // from the ES). Rows without OT hits, and failed fits, keep their input state (dropFailed: failed fits are removed,
  // or with hlPixelFallback replaced by their pixel seed in `pix`). `cls` (optional, device, [n]):
  // the seed class per row; `st` (optional, device, [n]): LstSeedFitStatus per row. Asynchronous in `queue`.
  void fitLstSeeds(Queue& queue,
                   ::mkfitdev::ESView const& es,
                   ::mkfitdev::HitSoAConstView hits,
                   uint32_t nPixel,
                   ::mkfitdev::SeedSoAView seeds,
                   int32_t n,
                   LstSeedFitConfig const& cfg,
                   ::mkfitdev::lstseeds::PixelSeedRows const& pix,
                   LstSeedFitCounters* counters,
                   int8_t* cls = nullptr,
                   int8_t* st = nullptr);

  // host-creator fit (fitHostLike, cfg.originPrior is ignored: the creator structure always;
  // cfg.passes selects the field model as for the LST seeds) of every row r < n of `in`, then the TSCBL of the
  // state on the last hit. Asynchronous in `queue`; `out` [n] on the device.
  void fitPixelSeeds(Queue& queue,
                     ::mkfitdev::ESView const& es,
                     ::mkfitdev::HitSoAConstView hits,
                     ::mkfitdev::lstseeds::PixSeedIn const* in,
                     ::mkfitdev::lstseeds::PixSeedOut* out,
                     int32_t n,
                     LstSeedFitConfig const& cfg,
                     ::mkfitdev::lstseeds::PixSeedBeam const& beam);

  // the pLS seed state from the device: seed rows s < n with pixIdx[s] >= 0 (the edm pixel
  // track of a pLS, pT3 or pT5 seed; LSTOutputConverter: seed = *pixelSeed) take params / errors / charge from row
  // pixIdx[s] of `states` (hltInputLSTDevice:pixelSeedStates; charge 0 = no state: the row is left as it is).
  // fitLstSeeds then replaces the state of the rows with OT hits.
  void applyPixelSeedStates(Queue& queue,
                            ::mkfitdev::SeedSoAView seeds,
                            int32_t n,
                            int32_t const* pixIdx,
                            ::mkfitdev::TrackSoAConstView states,
                            int32_t nStates);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstseeds

#endif

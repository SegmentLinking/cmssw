#ifndef RecoTracker_MkFitAlpaka_src_alpaka_fit_FitTracks_h
#define RecoTracker_MkFitAlpaka_src_alpaka_fit_FitTracks_h

// Exported entry point of the device mkFit final fit (src/alpaka/FitTracks.dev.cc). Callers never instantiate the
// kernel themselves (one TU per component).

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESView.h"
#include "RecoTracker/MkFitAlpaka/interface/fit/CpeGeneric.h"
#include "RecoTracker/MkFitAlpaka/interface/fit/HitStateDev.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::fit {

  using ::mkfitdev::ESView;

  // Device-side counters of one event (all int32; the caller zeroes them).
  struct FitCounters {
    int32_t nRefit;             // refits after an outlier removal (MkFitCore "remap"), summed over the rounds
    int32_t nRemovedHits;       // outlier hits removed, summed over the rounds
    int32_t nNaN;               // fit directions with a non-finite chi2 sum (state kept, reFitOutputTracks)
    int32_t nHitCountMismatch;  // nFoundHits != number of hits with index >= 0 (the fit uses the latter)
    int32_t nOverflow;          // more than kMaxTrkHits hits to order (never expected)
    int32_t nFailed;            // fits flagged as failed (chi2 = -1): non-finite backward pass, or an outlier kept by
                                // the three-hit minimum with no other hit removed
  };

  // The outer-end outlier rule of the port (DEVIATION DEV-10; defaults = MkFitCore): nearEdgeHitsOuter > 1 tests the
  // nearEdgeHitsOuter hits nearest the outer end with the forward pass (chi2 > the end-hit cut), each only when the hits
  // before it lie on at least 3 other detector layers, and counts a hit only this window flags only in a round in
  // which no other hit of the track is flagged; outliersPerRound > 0 removes at most that many outliers per round
  // (largest scorer first). MkFitCore: the outermost hit only, every flagged hit of a round.
  struct FitOptions {
    int nearEdgeHitsOuter = 1;
    int outliersPerRound = 0;
  };

  // Fits rows [0, nTracks) of `tracks` in place (MkBuilder::fittracks): parameters, errors, chi2 at the innermost hit,
  // and the state at the outermost fitted hit (TrackSoA outer, the OuterHitState of the final fit); removed outliers
  // get the hit index Hit::fitOutlierIdx(index) and nFoundHits is decremented (Track::removeHit); a failed fit gets
  // chi2 = -1. Charge, score and status are not touched.
  // `hits` is the event HitSoA (pixel wrapper rows first, then strip rows starting at nPixel); `es` the device ES.
  // `counters` must point to zeroed device memory. Asynchronous in `queue`.
  // useCpe: apply the device PixelCPEGeneric with the track angles on every pixel hit (MkFitFitProducer behaviour);
  // `cpe` (device tables) and `clusters` ([nPixel], device) are then required. false = the no-CPE variant.
  // hitStates ([nTracks * kMaxTrkHits], device; nullptr = off): storeHitStates, the smoothed
  // local state at every HitOnTrack position, element t * kMaxTrkHits + position.
  // nTracksDev: if set, the row count is read on the device from
  // this address (e.g. the TrackSoA nTracks scalar of the device candCutSel) and nTracks is the CAPACITY (grid size,
  // scratch stride); rows [*nTracksDev, nTracks) are not touched. No host synchronization.
  void runFinalFit(Queue& queue,
                   ESView const& es,
                   ::mkfitdev::HitSoAConstView hits,
                   uint32_t nPixel,
                   ::mkfitdev::TrackSoAView tracks,
                   int nTracks,
                   FitCounters* counters,
                   bool useCpe = false,
                   ::mkfitdev::cpe::CpeTables cpe = {},
                   const ::mkfitdev::cpe::ClusterCpe* clusters = nullptr,
                   ::mkfitdev::fit::HitStateDev* hitStates = nullptr,
                   const int32_t* nTracksDev = nullptr,
                   FitOptions const& options = FitOptions{});

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::fit

#endif

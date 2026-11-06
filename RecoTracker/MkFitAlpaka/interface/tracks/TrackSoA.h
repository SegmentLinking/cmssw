#ifndef RecoTracker_MkFitAlpaka_interface_tracks_TrackSoA_h
#define RecoTracker_MkFitAlpaka_interface_tracks_TrackSoA_h

// Output track format of the device mkFit chain: what building (export of the best candidate), backward fit and the
// final fit write, and what the duplicate cleaner and the output converters read. One row = one mkfit::Track.
// Field meaning and units follow mkfit::Track (MkFitCore/interface/Track.h) exactly.

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "RecoTracker/MkFitAlpaka/interface/math/HitConstants.h"

namespace mkfitdev {

  // Config::nMaxTrkHits: hit-list capacity per track. Longer lists are truncated and counted (nOverflowHits).
  constexpr int kMaxTrkHits = 64;

  // Same bit layout and special values as mkfit::HitOnTrack (Hit.h): index < 0 is a hole/stop/edge code.
  struct HitOnTrack {
    int index : 24;
    int layer : 8;
  };
  static_assert(sizeof(HitOnTrack) == 4);

  // MkFitCore special hit indices (Hit.h): interface/math/HitConstants.h. nFoundHits counts index >= 0 and
  // kHitCCCFilterIdx, as Track::addHitIdx.

  // State as TrackState: parameters (x, y, z, 1/pT, phi, theta) and SMatrixSym66 errors in its packed
  // lower-triangle order, element (i, j) with i >= j at i * (i + 1) / 2 + j (= Matriplex Sym order).
  struct TrackParams {
    float v[6];
  };
  struct TrackErrors {
    float v[21];
  };
  struct TrackHits {
    HitOnTrack hot[kMaxTrkHits];
  };
  // mkfit::OuterHitState: the final fit's state at the outermost hit it used, in that module's local frame as
  // mkfit::HitStateOnTrack (par = q/p, dx/dz, dy/dz, x, y; err = lower triangle of the 5x5 covariance), the hit's
  // chi2 increment in the forward pass, its HitOnTrack position (hot, -1: none), the sign of the local z momentum and
  // kind (1 ForwardOnly, 2 BackwardOnly), and the module plane (ModuleInfo pos, xdir, zdir: the det surface's position
  // and rotation rows). Written by the final fit only; the output conversion navigates outward from it.
  struct TrackOuterState {
    float par[5];
    float err[15];
    float pos[3];
    float xdir[3];
    float zdir[3];
    float chi2;
    int16_t hot;
    int8_t pzSign;
    int8_t kind;
    int8_t valid;
  };

  GENERATE_SOA_LAYOUT(TrackLayout,
                      SOA_COLUMN(TrackParams, params),  // TrackState::parameters
                      SOA_COLUMN(TrackErrors, errors),  // TrackState::errors
                      SOA_COLUMN(int16_t, charge),      // TrackState::charge
                      SOA_COLUMN(float, chi2),          // TrackBase::chi2_
                      SOA_COLUMN(float, score),         // TrackBase::score_
                      SOA_COLUMN(int32_t, label),       // TrackBase::label_ (= seed index)
                      SOA_COLUMN(int16_t, nTotalHits),  // lastHitIdx_ + 1
                      SOA_COLUMN(int16_t, nFoundHits),  // nFoundHits_
                      // Track::Status fields that later stages or converters use
                      SOA_COLUMN(int8_t, nSeedHits),         // status_.n_seed_hits
                      SOA_COLUMN(int8_t, etaRegion),         // status_.eta_region
                      SOA_COLUMN(int8_t, algorithm),         // status_.algorithm
                      SOA_COLUMN(int8_t, nOverlaps),         // status_.n_overlaps (Track::nOverlapHits)
                      SOA_COLUMN(int8_t, duplicate),         // status_.duplicate (set by the duplicate cleaner)
                      SOA_COLUMN(TrackHits, hits),           // hitsOnTrk_[0 .. nTotalHits)
                      SOA_COLUMN(TrackOuterState, outer),    // final fit: OuterHitState
                      SOA_SCALAR(int32_t, nTracks),          // rows in use (<= capacity)
                      SOA_SCALAR(int32_t, nOverflowTracks),  // writers: tracks dropped because the SoA was full
                      SOA_SCALAR(int32_t, nOverflowHits))    // writers: tracks whose hit list exceeded kMaxTrkHits

  using TrackSoA = TrackLayout<>;
  using TrackSoAView = TrackSoA::View;
  using TrackSoAConstView = TrackSoA::ConstView;

}  // namespace mkfitdev

#endif

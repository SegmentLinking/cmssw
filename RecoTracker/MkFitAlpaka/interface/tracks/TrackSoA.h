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

  ALPAKA_FN_HOST_ACC constexpr inline int symIdx6(int i, int j) {
    return i >= j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
  }

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
                      SOA_SCALAR(int32_t, nTracks),          // rows in use (<= capacity)
                      SOA_SCALAR(int32_t, nOverflowTracks),  // writers: tracks dropped because the SoA was full
                      SOA_SCALAR(int32_t, nOverflowHits))    // writers: tracks whose hit list exceeded kMaxTrkHits

  using TrackSoA = TrackLayout<>;
  using TrackSoAView = TrackSoA::View;
  using TrackSoAConstView = TrackSoA::ConstView;

  // ---- helpers mirroring MkFitCore Track accessors (host and device) ----

  ALPAKA_FN_HOST_ACC inline float trkInvpT(TrackSoAConstView v, int i) { return v[i].params().v[3]; }
  ALPAKA_FN_HOST_ACC inline float trkMomPhi(TrackSoAConstView v, int i) { return v[i].params().v[4]; }
  ALPAKA_FN_HOST_ACC inline float trkTheta(TrackSoAConstView v, int i) { return v[i].params().v[5]; }

  // Track::nInsideMinusOneHits
  ALPAKA_FN_HOST_ACC inline int trkNInsideMinusOneHits(TrackSoAConstView v, int t) {
    int n = 0;
    bool insideValid = false;
    const auto& h = v[t].hits().hot;
    for (int i = v[t].nTotalHits() - 1; i >= 0; --i) {
      if (h[i].index >= 0)
        insideValid = true;
      if (insideValid && h[i].index == -1)
        ++n;
    }
    return n;
  }

  // Track::nTailMinusOneHits
  ALPAKA_FN_HOST_ACC inline int trkNTailMinusOneHits(TrackSoAConstView v, int t) {
    int n = 0;
    const auto& h = v[t].hits().hot;
    for (int i = v[t].nTotalHits() - 1; i >= 0; --i) {
      if (h[i].index >= 0)
        return n;
      if (h[i].index == -1)
        ++n;
    }
    return n;
  }

}  // namespace mkfitdev

#endif

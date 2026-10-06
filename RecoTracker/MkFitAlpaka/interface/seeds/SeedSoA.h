#ifndef RecoTracker_MkFitAlpaka_interface_seeds_SeedSoA_h
#define RecoTracker_MkFitAlpaka_interface_seeds_SeedSoA_h

// Device seed table of the mkFit LST step: one row per input seed (MkFitSeedWrapper::seeds(), in input order),
// plus the results of the device seed import (run_OneIteration: seed_post_cleaning, MkBuilder::import_seeds with
// the phase2:1 partitioner and the (phi, eta) binnor rank, EventOfCombCandidates::insertSeed).
//   input columns   params, errors, charge, label, chi2, status, nHits, hits, lastX/lastY/lastZ (position of the
//                   seed's last hit = MkFitCore eoh[layer].refHit(index), filled by the host packer)
//   import results  silly (seed_post_cleaning removes the row), region (TrackerInfo::EtaRegion), key (binnor
//                   masked key), cleanIdx (index in the cleaned seed vector = seed_idx / m_seed_origin_index),
//                   pos (import position = row of the CombCandidate, -1 if removed)
//   order           row p = input row of the seed imported at position p (p < nKept)
// Scalars: nSeeds (input rows), nKept (imported), regionEnd[r] (m_seedEtaSeparators after the cumulative sum),
//          minLastLayer/maxLastLayer[r] (m_seedMinLastLayer / m_seedMaxLastLayer, -1 if none),
//          nOverflowHits (seeds with more than kMaxSeedHits hits: their extra hits are lost, never silently).

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"

namespace mkfitdev {

  // Hit-list capacity of a seed row. LST seeds carry up to ~20 hits (pixel + both hits of every OT mini-doublet);
  // NOTE Track::Status::n_seed_hits has only 4 bits, so setNSeedHits() keeps nTotalHits & 15 for them
  // (reproduced: statusWithSeedHitsAndRegion / CandState::nSeedHits). Longer seeds are truncated and counted.
  constexpr int kMaxSeedHits = 32;
  // IterationConfig of the LST step: m_n_regions = 5 (TrackerInfo::EtaRegion).
  constexpr int kNSeedRegions = 5;

  struct SeedHits {
    HitOnTrack hot[kMaxSeedHits];
  };
  struct SeedRegionInts {
    int32_t v[kNSeedRegions];
  };

  GENERATE_SOA_LAYOUT(SeedLayout,
                      SOA_COLUMN(TrackParams, params),
                      SOA_COLUMN(TrackErrors, errors),
                      SOA_COLUMN(int16_t, charge),
                      SOA_COLUMN(int32_t, label),
                      SOA_COLUMN(float, chi2),
                      SOA_COLUMN(uint32_t, status),  // Track::Status bits (memcpy of getStatus())
                      SOA_COLUMN(int16_t, nHits),    // nTotalHits() (seeds carry no holes)
                      SOA_COLUMN(SeedHits, hits),
                      SOA_COLUMN(float, lastX),
                      SOA_COLUMN(float, lastY),
                      SOA_COLUMN(float, lastZ),
                      SOA_COLUMN(int8_t, silly),
                      SOA_COLUMN(int8_t, region),
                      SOA_COLUMN(uint32_t, key),
                      SOA_COLUMN(int32_t, cleanIdx),
                      SOA_COLUMN(int32_t, pos),
                      SOA_COLUMN(int32_t, order),
                      SOA_SCALAR(int32_t, nSeeds),
                      SOA_SCALAR(int32_t, nKept),
                      SOA_SCALAR(SeedRegionInts, regionEnd),
                      SOA_SCALAR(SeedRegionInts, minLastLayer),
                      SOA_SCALAR(SeedRegionInts, maxLastLayer),
                      SOA_SCALAR(int32_t, nOverflowHits))

  using SeedSoA = SeedLayout<>;
  using SeedSoAView = SeedSoA::View;
  using SeedSoAConstView = SeedSoA::ConstView;

  // Track::Status bit positions (GCC little-endian bit-field allocation of Track.h:198-246; checked against
  // MkFitCore on the host by the seed-import validation): n_seed_hits bits 21-24, eta_region bits 25-27.
  constexpr uint32_t kStatusNSeedHitsShift = 21, kStatusNSeedHitsMask = 0xFu;
  constexpr uint32_t kStatusEtaRegionShift = 25, kStatusEtaRegionMask = 0x7u;

  ALPAKA_FN_HOST_ACC inline uint32_t statusWithSeedHitsAndRegion(uint32_t st, int nSeedHits, int region) {
    st &= ~((kStatusNSeedHitsMask << kStatusNSeedHitsShift) | (kStatusEtaRegionMask << kStatusEtaRegionShift));
    st |= (uint32_t(nSeedHits) & kStatusNSeedHitsMask) << kStatusNSeedHitsShift;
    st |= (uint32_t(region) & kStatusEtaRegionMask) << kStatusEtaRegionShift;
    return st;
  }

  // MkFitCore phase2:1 partitioner layer limits (MkSeedPartitioners-phase2.cc:28-46), computed once from TrackerInfo.
  struct SeedPartitionLimits {
    float tecp1_rin, tecp1_rout, tecp1_zmin, tecp2_rin, tecp2_zmax;
    float tecn1_rin, tecn1_rout, tecn1_zmax, tecn2_rin, tecn2_zmin;
  };

}  // namespace mkfitdev

#endif

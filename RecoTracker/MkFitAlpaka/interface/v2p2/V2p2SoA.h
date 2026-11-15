#ifndef RecoTracker_MkFitAlpaka_interface_v2p2_V2p2SoA_h
#define RecoTracker_MkFitAlpaka_interface_v2p2_V2p2SoA_h

// Device storage of MkFinderV2p2 for one plan entry, all seeds at once (MkFinderV2p2Structures.h: CCandRep,
// CandIncoming, PassedHitMatch, CandNode). The candidate books (CombCandidate / TrackCand) and the HoT pool are the
// clone engine's (interface/cands/CandsSoA.h). Row conventions (s = seed row, ic = TrackCand index in the seed's
// current buffer = CandIncoming::m_origin_tcand_index, k = position, n = pool row):
//   V2p2SeedSoA       row s                                    per-seed state of this entry (with the seed's node
//                                                              list, CCandRep::m_nodes), event counters
//   V2p2CandSoA       row v2p2CandRow(s, ic)                   CandIncoming of TrackCand ic
//   V2p2LayerHitSoA   row v2p2LayerHitRow(s, ic, k, stride)    CandIncoming::m_layer_hits[k], path order;
//                                                              stride = V2p2Params::max_presel_hits
//   V2p2NodeSoA       row n                                    CandNode: the in-layer paths of every seed, a pool
//                                                              reset at each plan entry, rows in creation order
// MkFitCore keeps one arena per finder and frees a seed's nodes once it is selected; the device holds the nodes of
// all seeds of one entry. A pool row that does not fit is not stored and counted (nNodeOverflow).

#include <cmath>
#include <cstdint>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Params.h"

namespace mkfitdev {

  // ---- capacities, from the MkFitCore limits
  constexpr int kV2p2MaxPreselHits = V2p2Params::max_presel_hits_limit;       // CandIncoming::kPqCap
  constexpr int kV2p2MaxHitsPerLayer = V2p2Params::max_hits_per_layer_limit;  // in-layer path length, chain[] size

  // Nodes listed per seed at harvest (CCandRep::m_nodes): room for the depth-0 nodes of a full beam; a seed's later
  // nodes continue on its chain (nodeHead, nextInSeed)
  constexpr int kV2p2NodeListCap = kMaxCandsPerSeed * kV2p2MaxPreselHits;

  // Path key of a node: the m_layer_hits positions of its path, one digit per depth, most significant first.
  constexpr int kV2p2PathKeyBits = 4;
  static_assert(kV2p2MaxPreselHits <= (1 << kV2p2PathKeyBits), "a path position must fit in one key digit");
  static_assert(kV2p2MaxHitsPerLayer * kV2p2PathKeyBits <= std::numeric_limits<uint32_t>::digits,
                "a path key must fit in 32 bits");
  // the narrow columns below
  static_assert(kV2p2MaxPreselHits <= std::numeric_limits<int8_t>::max(), "hitPos and nLayerHits are int8_t");
  static_assert(kMaxCandsPerSeed <= std::numeric_limits<int8_t>::max(), "ic is int8_t");
  static_assert(hitpack::kDetIdBits <= std::numeric_limits<int16_t>::digits, "module (detIDinLayer) is int16_t");

  // mini_propagators::InitialState (State x .. fail_flag, then inv_pt, inv_k, theta)
  struct V2p2InitialState {
    float x, y, z;
    float px, py, pz;
    float dalpha;
    int32_t fail_flag;
    float inv_pt, inv_k;
    float theta;
  };

  // CandIncoming::PassedHitMatch (tr_hitmatch_id is trace only)
  struct V2p2PassedHitMatch {
    float ddphi;
    uint32_t hit_orig_index;
    uint32_t hit_index;  // internal index in the LayerOfHits
    int32_t layer;
    V2p2InitialState mixed_state;
  };

  // TrackState of a CandNode: parameters, covariance (21, CandState::err order) and charge
  struct V2p2TrackState {
    float par[6];
    float err[21];
    int32_t charge;
  };

  // The first kV2p2NodeListCap pool rows of a seed's nodes this entry, in no particular order
  struct V2p2NodeList {
    int32_t node[kV2p2NodeListCap];
  };

  // Event and entry counters of the v2p2 search (scalar of V2p2SeedSoA)
  struct V2p2Counters {
    int32_t nPoolRows;        // nodes created this entry (stored or not): the next pool row
    uint32_t nNodeOverflow;   // nodes lost because the pool was full (status kV2p2NodeOverflow)
    uint32_t nNanScore;       // selections that ranked a NaN score (status kV2p2NanScore)
    int32_t nActiveCands[2];  // V0's list of the entry's CandIncomings (activeRow), by step parity
  };

  GENERATE_SOA_LAYOUT(V2p2SeedSoALayout,
                      SOA_COLUMN(int32_t, nNodes),         // nodes of the seed stored this entry
                      SOA_COLUMN(V2p2NodeList, nodeList),  // its first kV2p2NodeListCap nodes
                      SOA_COLUMN(int32_t, nodeHead),       // its chain of the nodes past the list, -1 = none
                      SOA_COLUMN(int8_t, maxCands),        // CCandRep::m_max_cands
                      SOA_COLUMN(int8_t, inLayer),         // 1: the seed was pulled into this entry's layer
                      SOA_SCALAR(V2p2Counters, counters))

  GENERATE_SOA_LAYOUT(V2p2CandSoALayout,
                      SOA_COLUMN(int8_t, active),      // 1: a CandIncoming this entry
                      SOA_COLUMN(int8_t, wsr),         // m_wsr.m_wsr (WithinSensitiveRegion, es/ESView.h)
                      SOA_COLUMN(int8_t, inGap),       // m_wsr.m_in_gap
                      SOA_COLUMN(int8_t, nLayerHits),  // m_layer_hits.size()
                      SOA_COLUMN(float, logRho),       // m_log_rho
                      SOA_COLUMN(int32_t, activeRow))  // entry k < nActiveCands: the row of an active CandIncoming

  GENERATE_SOA_LAYOUT(V2p2LayerHitSoALayout, SOA_COLUMN(V2p2PassedHitMatch, hit))

  GENERATE_SOA_LAYOUT(V2p2NodeSoALayout,
                      SOA_COLUMN(int32_t, seed),          // seed row
                      SOA_COLUMN(int8_t, ic),             // m_cand->m_origin_tcand_index
                      SOA_COLUMN(int8_t, hitPos),         // m_hit_pos
                      SOA_COLUMN(int8_t, nHits),          // m_feat.n_hits
                      SOA_COLUMN(int32_t, parentNode),    // m_parent_idx (pool row), -1 = the CandIncoming
                      SOA_COLUMN(int32_t, nextInSeed),    // next node of the seed's chain, -1 = end
                      SOA_COLUMN(uint32_t, pathKey),      // v2p2PathKey
                      SOA_COLUMN(int32_t, hitIndex),      // m_hot.index
                      SOA_COLUMN(int16_t, hitLayer),      // m_hot.layer
                      SOA_COLUMN(int16_t, module),        // detIDinLayer of the hit (same-module veto)
                      SOA_COLUMN(float, chi2),            // m_chi2
                      SOA_COLUMN(float, chi2Sum),         // m_feat.chi2_sum
                      SOA_COLUMN(float, logDetVSum),      // m_feat.log_det_v_sum
                      SOA_COLUMN(V2p2TrackState, state))  // m_state

  using V2p2SeedSoA = V2p2SeedSoALayout<>;
  using V2p2CandSoA = V2p2CandSoALayout<>;
  using V2p2LayerHitSoA = V2p2LayerHitSoALayout<>;
  using V2p2NodeSoA = V2p2NodeSoALayout<>;

  // ---- row helpers
  ALPAKA_FN_HOST_ACC inline constexpr int v2p2CandRow(int seed, int ic) { return seed * kMaxCandsPerSeed + ic; }
  ALPAKA_FN_HOST_ACC inline constexpr int v2p2CandSeed(int candRow) { return candRow / kMaxCandsPerSeed; }
  ALPAKA_FN_HOST_ACC inline constexpr int v2p2CandIc(int candRow) { return candRow % kMaxCandsPerSeed; }
  ALPAKA_FN_HOST_ACC inline constexpr int v2p2LayerHitRow(int seed, int ic, int k, int stride) {
    return v2p2CandRow(seed, ic) * stride + k;
  }

  // Path key of a node at depth (= n_hits - 1) taking position hitPos, from its parent's key (0 for a root).
  ALPAKA_FN_HOST_ACC inline constexpr uint32_t v2p2PathKey(uint32_t parentKey, int depth, int hitPos) {
    return parentKey | (uint32_t(hitPos) << (kV2p2PathKeyBits * (kV2p2MaxHitsPerLayer - 1 - depth)));
  }

  // ---- the total order that replaces std::partial_sort's unspecified order of equal scores in
  // select_and_materialise: a competitor ranks before another if its score is larger, or, at equal score, if its
  // ordinal is smaller. The ordinal follows the order MkFitCore builds m_sel in: the hole of every CandIncoming (by
  // TrackCand index), then the in-layer paths (by depth, TrackCand index, path key), then the TrackCands that did not
  // enter the layer (by TrackCand index). A NaN score ranks after every number (counted in nNanScore).
  enum V2p2SelClass : uint64_t { kV2p2SelHole = 0, kV2p2SelNode = 1, kV2p2SelUntouched = 2 };

  // the ordinal's fields above the path key: ic, then nHits, then the class, one byte each
  constexpr int kV2p2SelIcShift = std::numeric_limits<uint32_t>::digits;
  constexpr int kV2p2SelHitsShift = kV2p2SelIcShift + std::numeric_limits<uint8_t>::digits;
  constexpr int kV2p2SelClassShift = kV2p2SelHitsShift + std::numeric_limits<uint8_t>::digits;
  static_assert(kMaxCandsPerSeed <= std::numeric_limits<uint8_t>::max(), "ic must fit in its ordinal field");
  static_assert(kV2p2MaxHitsPerLayer <= std::numeric_limits<uint8_t>::max(), "nHits must fit in its ordinal field");

  ALPAKA_FN_HOST_ACC inline constexpr uint64_t v2p2SelOrdinal(V2p2SelClass cls, int ic, int nHits, uint32_t pathKey) {
    return (uint64_t(cls) << kV2p2SelClassShift) | (uint64_t(nHits) << kV2p2SelHitsShift) |
           (uint64_t(ic) << kV2p2SelIcShift) | uint64_t(pathKey);
  }

  // a ranks before b
  ALPAKA_FN_HOST_ACC inline bool v2p2SelBefore(float scoreA, uint64_t ordA, float scoreB, uint64_t ordB) {
    const bool nanA = std::isnan(scoreA), nanB = std::isnan(scoreB);
    if (nanA != nanB)
      return nanB;
    if (!nanA && scoreA != scoreB)
      return scoreA > scoreB;
    return ordA < ordB;
  }

}  // namespace mkfitdev

#endif

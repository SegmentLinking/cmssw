#ifndef RecoTracker_MkFitAlpaka_interface_cands_CandsSoA_h
#define RecoTracker_MkFitAlpaka_interface_cands_CandsSoA_h

// Per-seed candidate storage of the clone engine (EventOfCombCandidates / CombCandidate), as fixed-capacity
// SoA collections. Row conventions (s = seed index in the event, ic = candidate, k = slot):
//   SeedCandsSoA      row s                                   per-seed scalars, best-short bookkeeping
//   CandSlotsSoA      row (s * kSlotsPerSeed + k)             candidate slots: two buffers of kMaxCandsPerSeed
//                                                             (k = buf * kMaxCandsPerSeed + ic) + 1 best-short slot
//                                                             (k = kBestShortSlot)
//   CandHotsSoA       row (s * hotsPerSeed + n)               HoT node pool (CombCandidate::m_hots)
//   CandOptionsSoA    row (s * kMaxOptsPerSeed + j)           per-seed option list (CandCloner::m_hits_to_add)
//   CandExtrasSoA     row (s * kMaxExtrasPerSeed + e)         held-back candidates (extra_cands)
//   CandUpdatesSoA    row (s * kMaxCandsPerSeed + u)          kalman update / overlap re-check lists (MkFitCore
//                                                             seed_cand_update_idx / seed_cand_overlap_idx)
// The SoA scalars hold the event-wide overflow counters: a fixed-capacity buffer never truncates silently.

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"

namespace mkfitdev {

  constexpr int kSlotsPerSeed = 2 * kMaxCandsPerSeed + 1;
  constexpr int kBestShortSlot = 2 * kMaxCandsPerSeed;

  GENERATE_SOA_LAYOUT(SeedCandsSoALayout,
                      SOA_COLUMN(int32_t, nCands),         // live candidates in buffer curBuf
                      SOA_COLUMN(int8_t, curBuf),          // 0/1: which half of the slot range is current
                      SOA_COLUMN(int8_t, state),           // SeedState
                      SOA_COLUMN(int16_t, pickupLayer),    // CombCandidate::m_pickup_layer
                      SOA_COLUMN(int16_t, layer),          // layer of the current plan step (set per step)
                      SOA_COLUMN(int16_t, region),         // eta region = index into per-region step tables
                      SOA_COLUMN(uint8_t, activeMask),     // K1: bit ic set = cand ic listed for this step
                      SOA_COLUMN(int8_t, nActive),         // K1: number of listed cands
                      SOA_COLUMN(int32_t, seedOriginIdx),  // index in the passed-in seed vector
                      SOA_COLUMN(int32_t, nHots),          // used HoT nodes
                      SOA_COLUMN(int32_t, nExtras),
                      SOA_COLUMN(int32_t, nUpdates),
                      SOA_COLUMN(int32_t, nOverlapUpdates),
                      SOA_COLUMN(CandBook, bestShort),     // m_best_short_cand bookkeeping; state in kBestShortSlot
                      SOA_COLUMN(int8_t, bestShortValid),  // m_best_short_cand.combCandidate() != nullptr
                      SOA_COLUMN(int16_t, lastHitIdxBeforeBkw),  // CombCandidate::m_lastHitIdx_before_bkwsearch
                      SOA_COLUMN(int16_t, nInsideMinusOneBeforeBkw),
                      SOA_COLUMN(int16_t, nTailMinusOneBeforeBkw),
                      SOA_COLUMN(uint32_t, overflowBits),  // CandOverflow bits set for this seed
                      SOA_COLUMN(int8_t, bkwRepacked),     // repackCandPostBkwSearch applied (engine post-filter)
                      // backward-search gate (EventOfCombCandidates m_pre_bkw_cands, m_bkw_min_pixel_layers)
                      SOA_COLUMN(CandBook, preBkwBook),
                      SOA_COLUMN(CandState, preBkwState),
                      SOA_COLUMN(int8_t, bkwMinPixLayers),
                      SOA_SCALAR(int32_t, hotsPerSeed),  // HoT pool capacity per seed (rows per seed)
                      SOA_SCALAR(uint32_t, nOverflowHots),
                      SOA_SCALAR(uint32_t, nOverflowOpts),
                      SOA_SCALAR(uint32_t, nOverflowExtras),
                      SOA_SCALAR(uint32_t, nRepackRepeat))  // post-filter called again on repacked rows

  GENERATE_SOA_LAYOUT(CandSlotsSoALayout, SOA_COLUMN(CandBook, book), SOA_COLUMN(CandState, state))

  GENERATE_SOA_LAYOUT(CandHotsSoALayout, SOA_COLUMN(HoTNode, node))

  GENERATE_SOA_LAYOUT(CandOptionsSoALayout, SOA_COLUMN(CandOption, opt))

  GENERATE_SOA_LAYOUT(CandExtrasSoALayout, SOA_COLUMN(CandExtra, extra))

  // outSrc: row (s, k) = parent slot (current buffer before K4) of K4's output candidate k,
  // -1 = no copy pending; K5 does the parent -> new-buffer state copy (K4 copy split).
  GENERATE_SOA_LAYOUT(CandUpdatesSoALayout,
                      SOA_COLUMN(CandUpdate, upd),
                      SOA_COLUMN(CandUpdate, ovl),
                      SOA_COLUMN(int8_t, outSrc))

  using SeedCandsSoA = SeedCandsSoALayout<>;
  using CandSlotsSoA = CandSlotsSoALayout<>;
  using CandHotsSoA = CandHotsSoALayout<>;
  using CandOptionsSoA = CandOptionsSoALayout<>;
  using CandExtrasSoA = CandExtrasSoALayout<>;
  using CandUpdatesSoA = CandUpdatesSoALayout<>;

  // Row helpers
  ALPAKA_FN_HOST_ACC inline constexpr int candSlotRow(int seed, int buf, int ic) {
    return seed * kSlotsPerSeed + buf * kMaxCandsPerSeed + ic;
  }
  ALPAKA_FN_HOST_ACC inline constexpr int bestShortRow(int seed) { return seed * kSlotsPerSeed + kBestShortSlot; }
  ALPAKA_FN_HOST_ACC inline constexpr int hotRow(int seed, int n, int hotsPerSeed) { return seed * hotsPerSeed + n; }
  ALPAKA_FN_HOST_ACC inline constexpr int optRow(int seed, int j) { return seed * kMaxOptsPerSeed + j; }
  // Option slot of candidate ic, hit slot ih (0..kMaxHitsPerCand-1) or ih = kMaxHitsPerCand for the invalid option.
  ALPAKA_FN_HOST_ACC inline constexpr int optSlot(int ic, int ih) { return ih * kMaxCandsPerSeed + ic; }
  ALPAKA_FN_HOST_ACC inline constexpr int extraRow(int seed, int e) { return seed * kMaxExtrasPerSeed + e; }
  ALPAKA_FN_HOST_ACC inline constexpr int updRow(int seed, int u) { return seed * kMaxCandsPerSeed + u; }

}  // namespace mkfitdev

#endif

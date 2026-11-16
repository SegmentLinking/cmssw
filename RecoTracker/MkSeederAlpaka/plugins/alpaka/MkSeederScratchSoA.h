#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederScratchSoA_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederScratchSoA_h

// The scratch of the device steps after the chain finder, one SoA (blocks) collection per step: the cleaning
// (quads and triplets), the duplicate cleanings, the pixel fill's split
// steps, the hit sharing of the classifier's features. The lists of items per hit, per bin and per owner seed are
// cms::alpakatools::OneToManyAssocRandomAccess (their storage: an offsets block with nOnes + 1 rows and the assoc
// object as its scalar, a content block).

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/OneToManyAssoc.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederEventHitsSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedsSoA.h"

#include "MkSeederRadixSort.h"

namespace mkseeder {

  // a list of uint32_t items (ranks, seeds) per one (a hit, a bin, an owner seed), sizes at run time
  using MkSeederItemLists = cms::alpakatools::
      OneToManyAssocRandomAccess<uint32_t, cms::alpakatools::kDynamicSize, cms::alpakatools::kDynamicSize>;
  // its offsets (nOnes + 1 rows) and the assoc object; its items
  GENERATE_SOA_LAYOUT(MkSeederListOffsetsLayout, SOA_COLUMN(uint32_t, offset), SOA_SCALAR(MkSeederItemLists, lists))
  GENERATE_SOA_LAYOUT(MkSeederListItemsLayout, SOA_COLUMN(uint32_t, item))

  // ---- the cleaning (MkSeederCleanKernels.h)
  // a quad's hits as their cluster-order positions, sorted (0xffffffff: none): read whole for the better-ranked
  // items in the rounds (one 16-byte load each)
  struct alignas(16) MkSeederSortedHits {
    uint32_t hit[4];
  };
  // One row per quad: by its position in the sorted order the quad's row (the chain finder's output is not copied
  // into that order), by finish index f (the position in the pattern order) its quad row and its rank, by rank r (the cleaning
  // order: the quads, then the triplets) its finish index, its sorted hits, its state (0 undecided, 1 kept, 2
  // dropped) and the dropped quads charged to it (n_amb)
  GENERATE_SOA_LAYOUT(MkSeederCleanItemsLayout,
                      SOA_COLUMN(uint32_t, quadOfSortedIndex),
                      SOA_COLUMN(uint32_t, quadOfFinishIndex),
                      SOA_COLUMN(uint32_t, rankOfFinishIndex),
                      SOA_COLUMN(uint32_t, finishIndexOfRank),
                      SOA_COLUMN(MkSeederSortedHits, sortedHits),
                      SOA_COLUMN(uint8_t, state),
                      SOA_COLUMN(int32_t, nAmbiguous))
  // One row per hit (cluster order): used by a kept quad
  GENERATE_SOA_LAYOUT(MkSeederHitUseLayout, SOA_COLUMN(uint8_t, usedByKeptQuad))
  // rows that only make an allocation larger: the sort's and the cleaning's scratch is at least as large as the
  // chain finder's candidate pool, so that the caching allocator hands it the pool's block (freed just before)
  GENERATE_SOA_LAYOUT(MkSeederPaddingLayout, SOA_COLUMN(uint8_t, byte))
  // the scratch of the quads' sort and of the cleaning, one allocation: the sort's keys and counts
  // (GlobalRadixSort), the items, the hits' use, per hit the items on it by rank (hitItems), the padding
  GENERATE_SOA_BLOCKS(MkSeederSortCleanScratchLayout,
                      SOA_BLOCK(keys0, MkSeederSortKeysLayout),
                      SOA_BLOCK(keys1, MkSeederSortKeysLayout),
                      SOA_BLOCK(counts, MkSeederSortCountsLayout),
                      SOA_BLOCK(items, MkSeederCleanItemsLayout),
                      SOA_BLOCK(hitUse, MkSeederHitUseLayout),
                      SOA_BLOCK(hitItemOffsets, MkSeederListOffsetsLayout),
                      SOA_BLOCK(hitItems, MkSeederListItemsLayout),
                      SOA_BLOCK(padding, MkSeederPaddingLayout))

  // a GPU's binning, one allocation: its input (the hits and the outer-tracker covariances in cluster order, the
  // blocks of MkSeederUnbinnedHitsSoA with the same layout: the host collection is copied into them) and the hits'
  // sort's keys and counts (GlobalRadixSort): one block of the caching allocator instead of two (the binning's
  // blocks are live with the event's hits, the binning's peak of device memory)
  GENERATE_SOA_BLOCKS(MkSeederBinningLayout,
                      SOA_BLOCK(hits, MkSeederHitsLayout),
                      SOA_BLOCK(otHits, MkSeederOTHitsLayout),
                      SOA_BLOCK(psStripHits, MkSeederPSStripHitsLayout),
                      SOA_BLOCK(psStripStarts, MkSeederPSStripStartsLayout),
                      SOA_BLOCK(keys0, MkSeederSortKeysLayout),
                      SOA_BLOCK(keys1, MkSeederSortKeysLayout),
                      SOA_BLOCK(counts, MkSeederSortCountsLayout))

  // ---- the duplicate cleanings (MkSeederDupKernels.h)
  // One row per input (call 1: per label of the cleaning's kept seeds; call 2: per post-step seed), and one more
  // (the scans' totals): dropped by the probe (call 1), the scans' inputs (an input that is a seed: a quad seed, a
  // triplet seed) and their exclusive scans (the new labels, the seeds' positions)
  GENERATE_SOA_LAYOUT(MkSeederDupInputsLayout,
                      SOA_COLUMN(uint8_t, probeDropped),
                      SOA_COLUMN(uint32_t, isLive),
                      SOA_COLUMN(uint32_t, isQuadSeed),
                      SOA_COLUMN(uint32_t, isTripletSeed),
                      SOA_COLUMN(uint32_t, newLabel),
                      SOA_COLUMN(uint32_t, quadSeedPosition),
                      SOA_COLUMN(uint32_t, tripletSeedPosition))
  // One row per seed (and one more, the scan's total): its input, its score, the rank maps, its state (0
  // undecided, 1 kept, 2 dropped), its owner (the kept seed that removed it, -1: none), its no-share bin (0xffffffff:
  // none), kept (the scan's input) and its position among the kept
  GENERATE_SOA_LAYOUT(MkSeederDupSeedsLayout,
                      SOA_COLUMN(uint32_t, input),
                      SOA_COLUMN(float, score),
                      SOA_COLUMN(uint32_t, seedOfRank),
                      SOA_COLUMN(uint32_t, rankOfSeed),
                      SOA_COLUMN(uint8_t, state),
                      SOA_COLUMN(int32_t, owner),
                      SOA_COLUMN(uint32_t, binKey),
                      SOA_COLUMN(uint32_t, isKept),
                      SOA_COLUMN(uint32_t, keptPosition))
  // the inputs; the seeds (their hits and helices, the merged hits to refit, the refits' states); per hit, per
  // no-share bin and per owner the seeds on it by rank
  GENERATE_SOA_BLOCKS(MkSeederDupScratchLayout,
                      SOA_BLOCK(inputs, MkSeederDupInputsLayout),
                      SOA_BLOCK(seeds, MkSeederDupSeedsLayout),
                      SOA_BLOCK(seedHits, MkSeederSeedHitsLayout),
                      SOA_BLOCK(helices, MkSeederSeedHelicesLayout),
                      SOA_BLOCK(mergedHits, MkSeederSeedHitsLayout),
                      SOA_BLOCK(refits, MkSeederSeedStatesLayout),
                      SOA_BLOCK(hitSeedOffsets, MkSeederListOffsetsLayout),
                      SOA_BLOCK(hitSeeds, MkSeederListItemsLayout),
                      SOA_BLOCK(binSeedOffsets, MkSeederListOffsetsLayout),
                      SOA_BLOCK(binSeeds, MkSeederListItemsLayout),
                      SOA_BLOCK(ownerSeedOffsets, MkSeederListOffsetsLayout),
                      SOA_BLOCK(ownerSeeds, MkSeederListItemsLayout))

  // ---- the classifier's hit sharing (MkSeederPcaKernels.h): per hit the final seeds on it
  GENERATE_SOA_BLOCKS(MkSeederShareScratchLayout,
                      SOA_BLOCK(hitSeedOffsets, MkSeederListOffsetsLayout),
                      SOA_BLOCK(hitSeeds, MkSeederListItemsLayout))

  using MkSeederListOffsetsSoA = MkSeederListOffsetsLayout<>;
  using MkSeederListItemsSoA = MkSeederListItemsLayout<>;
  using MkSeederCleanItemsSoA = MkSeederCleanItemsLayout<>;
  using MkSeederHitUseSoA = MkSeederHitUseLayout<>;
  using MkSeederPaddingSoA = MkSeederPaddingLayout<>;
  using MkSeederSortCleanScratchSoA = MkSeederSortCleanScratchLayout<>;
  using MkSeederBinningSoA = MkSeederBinningLayout<>;
  using MkSeederDupInputsSoA = MkSeederDupInputsLayout<>;
  using MkSeederDupSeedsSoA = MkSeederDupSeedsLayout<>;
  using MkSeederDupScratchSoA = MkSeederDupScratchLayout<>;
  using MkSeederShareScratchSoA = MkSeederShareScratchLayout<>;

}  // namespace mkseeder

#endif

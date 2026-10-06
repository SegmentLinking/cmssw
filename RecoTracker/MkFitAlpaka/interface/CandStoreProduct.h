#ifndef RecoTracker_MkFitAlpaka_interface_CandStoreProduct_h
#define RecoTracker_MkFitAlpaka_interface_CandStoreProduct_h

// Event product "candidate storage" of the clone engine (host side). One PortableCollection over a SoABlocks layout
// of the cands layouts (interface/cands/CandsSoA.h). Block views have exactly the standalone view types
// (SeedCandsSoA::View, CandSlotsSoA::View, ...), so the engine kernels and filterSeeds
// (interface/cands/alpaka/CandSeedOpsLaunch.h) take product.view().seeds() etc. unchanged.
// Block sizes for nSeeds seeds and hps HoT nodes per seed (use candStoreSizes()):
//   seeds nSeeds, slots nSeeds*kSlotsPerSeed, hots nSeeds*hps, opts nSeeds*kMaxOptsPerSeed,
//   extras nSeeds*kMaxExtrasPerSeed, upds nSeeds*kMaxCandsPerSeed.

#include <array>
#include <cstdint>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandsSoA.h"

namespace mkfitdev {
  GENERATE_SOA_BLOCKS(CandStoreBlocksLayout,
                      SOA_BLOCK(seeds, SeedCandsSoALayout),
                      SOA_BLOCK(slots, CandSlotsSoALayout),
                      SOA_BLOCK(hots, CandHotsSoALayout),
                      SOA_BLOCK(opts, CandOptionsSoALayout),
                      SOA_BLOCK(extras, CandExtrasSoALayout),
                      SOA_BLOCK(upds, CandUpdatesSoALayout))

  using CandStoreBlocks = CandStoreBlocksLayout<>;
  using CandStoreHostCollection = PortableHostCollection<CandStoreBlocks>;

  // Per-block sizes in block order, for the PortableCollection constructors taking std::array<int32_t, 6>.
  inline std::array<int32_t, 6> candStoreSizes(int32_t nSeeds, int32_t hotsPerSeed) {
    return {nSeeds,
            nSeeds * kSlotsPerSeed,
            nSeeds * hotsPerSeed,
            nSeeds * kMaxOptsPerSeed,
            nSeeds * kMaxExtrasPerSeed,
            nSeeds * kMaxCandsPerSeed};
  }
}  // namespace mkfitdev

#endif

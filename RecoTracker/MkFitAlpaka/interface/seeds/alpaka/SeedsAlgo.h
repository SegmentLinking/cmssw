#ifndef RecoTracker_MkFitAlpaka_interface_seeds_alpaka_SeedsAlgo_h
#define RecoTracker_MkFitAlpaka_interface_seeds_alpaka_SeedsAlgo_h

// Exported entry points of the device seed import / best-candidate export (src/alpaka/Seeds.dev.cc).
// Everything is enqueued on 'queue'; nothing synchronises.

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/DupCleanParams.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/SeedSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::seeds {

  // seed_post_cleaning + import_seeds (phase2:1, binnor rank, region-major) + insertSeed.
  // 'seeds' rows [0, nSeeds) hold the input (host packer, interface/seeds/SeedsHostPack.h); 'capacity' = its rows.
  // Writes the import results into 'seeds' and the initial CombCandidate rows [0, nKept) of the cands SoAs
  // (capacity >= capacity of 'seeds'; hotsPerSeed = HoT rows per seed of 'hots').
  // Last-hit positions from the device EventOfHits (HitSoA x/y/z columns and LayerSoA hitBase column, device memory).
  // All nullptr (default): the lastX/lastY/lastZ columns filled by the host packer are used.
  struct DeviceHitPositions {
    float const* x = nullptr;
    float const* y = nullptr;
    float const* z = nullptr;
    uint32_t const* layerHitBase = nullptr;
  };

  void importSeeds(Queue& queue,
                   ::mkfitdev::SeedSoAView seeds,
                   int32_t capacity,
                   ::mkfitdev::SeedPartitionLimits const& limits,
                   ::mkfitdev::SeedCandsSoA::View seedCands,
                   ::mkfitdev::CandSlotsSoA::View slots,
                   ::mkfitdev::CandHotsSoA::View hots,
                   int32_t hotsPerSeed,
                   DeviceHitPositions const& hitPos = DeviceHitPositions{});

  // export_best_comb_cands(out, remove_missing_hits = true): front candidate of every seed row
  // s < *nSeedsDev that has candidates -> TrackSoA, in seed order. Sets out.nTracks() (and overflow counters).
  // nSeedsDev: device pointer to the number of seed rows (e.g. &seeds.nKept()); capacity = rows of seedCands.
  // brokenChains (optional, device): incremented once per exported track whose HoT chain does not hold exactly
  // nFound valid hits along nFound + nMissing nodes (e.g. repackCandPostBkwSearch applied twice). Must stay 0.
  void exportBestCands(Queue& queue,
                       ::mkfitdev::SeedCandsSoA::ConstView seedCands,
                       ::mkfitdev::CandSlotsSoA::ConstView slots,
                       ::mkfitdev::CandHotsSoA::ConstView hots,
                       int32_t hotsPerSeed,
                       int32_t const* nSeedsDev,
                       int32_t capacity,
                       ::mkfitdev::TrackSoAView out,
                       uint32_t* brokenChains = nullptr);

  // Chain tail of the build module: the engine owns the post-backward-search filter, repack and compaction; this
  // function applies NO filter and NO repack:
  //   export_best_comb_cands(remove_missing_hits = true) -> clean_duplicates_sharedhits_pixelseed -> export_tracks.
  // nSeedsDev: device pointer to the survivor count of the engine's last compaction (the rows are in the result
  //   buffer). Rows s >= *nSeedsDev are never read, so stale rows of a capacity-wide buffer cannot be exported.
  //   capacity = rows of seedCands (>= *nSeedsDev), also the row count of 'exported' and 'out'.
  // 'exported' gets the export before the cleaner, 'out' the final tracks (MkFitProducer output).
  // dupClean: the duplicate-cleaner parameters of the iteration (used when removeDuplicates).
  // brokenChains: as in exportBestCands (optional; for the status product).
  void exportAndClean(Queue& queue,
                      ::mkfitdev::SeedCandsSoA::ConstView seedCands,
                      ::mkfitdev::CandSlotsSoA::ConstView slots,
                      ::mkfitdev::CandHotsSoA::ConstView hots,
                      int32_t hotsPerSeed,
                      int32_t const* nSeedsDev,
                      int32_t capacity,
                      bool removeDuplicates,
                      ::mkfitdev::clean::DupCleanParams const& dupClean,
                      ::mkfitdev::TrackSoAView exported,
                      ::mkfitdev::TrackSoAView out,
                      uint32_t* brokenChains = nullptr);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::seeds

#endif

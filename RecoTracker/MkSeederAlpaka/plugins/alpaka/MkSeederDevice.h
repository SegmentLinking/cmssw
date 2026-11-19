#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederDevice_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederDevice_h

#include <vector>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoTracker/MkSeederAlpaka/interface/alpaka/MkSeederEventDeviceCollection.h"

#include "MkSeederDeviceTypes.h"
#include "MkSeederRadixSort.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {
  struct CleanInputs;
}

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  // The chain finder of both z sides on the queue: per side stage b, then stage c and d per chain position
  // (binnedLayerIds: per side and chain position its layer if the seeder bins it and a candidate can reach it, else
  // -1, to skip the others). Its
  // candidate pool and its nQueues queues are allocated here: at least poolChunks chunks (a GPU: as many as fill the
  // caching allocator's bin of that size; poolChunks is set to them); the quads into quads (quadCapacity rows), the
  // counters (zeroed by the caller). Returns the bytes of the pool's allocation.
  size_t findQuads(Queue &queue,
                   mkseeder::ChainFinderInputs const (&views)[2],
                   std::vector<int> const (&binnedLayerIds)[2],
                   uint32_t nQueues,
                   uint32_t &poolChunks,
                   mkseeder::QuadsView quads,
                   uint32_t quadCapacity,
                   ::mkseeder::MkSeederFinderCounters *counters);

  // The OT extension and the pixel fill of the first *count (nullptr: n) of n seeds on the queue
  // (MkSeederPostKernels.h), in place; with fits (fit.on, the seeds' states) also the refit of the seeds whose hits
  // changed and the beam-line states (MkSeederFitKernels.h). A GPU extends with a block per seed and fills the pixel
  // layers in parallel; a CPU backend does both with a thread per seed. The scratch (for n seeds) is allocated here.
  void extendAndFillSeeds(Queue &queue,
                          mkseeder::ChainFinderInputs const &view,
                          mkseeder::PostInputs const &post,
                          mkseeder::FitInputs const &fit,
                          mkseeder::SeedHitsView seeds,
                          mkseeder::SeedStatesView states,
                          uint32_t n,
                          const uint32_t *count = nullptr);

  // The seed fits on the queue (MkSeederFit.dev.cc, MkSeederFitKernels.h): the fit of n kept quads and triplet seeds
  void fitSeedHits(Queue &queue,
                   mkseeder::FitInputs const &fit,
                   mkseeder::KeptSeedsConstView seeds,
                   mkseeder::SeedStatesView out,
                   uint32_t n,
                   const uint32_t *skip = nullptr);
  // skip[label] = 1 for the probe items (the first min(*nProbes, cap) rows of probe) that the probe's extension dropped:
  // fitSeedHits() need not fit them (the first duplicate cleaning drops their labels; fitOk 0, a zero state), and fits
  // the others as a compact list
  void markDroppedProbes(
      Queue &queue, mkseeder::SeedHitsConstView probe, const uint32_t *nProbes, uint32_t cap, uint32_t *skip);
  // the refit of the seeds whose hits changed (the input's hits in backup: the fallback) and the beam-line states of
  // the first *count (nullptr: n) of n seeds
  void refitSeeds(Queue &queue,
                  mkseeder::FitInputs const &fit,
                  mkseeder::SeedHitsView seeds,
                  mkseeder::SeedHitListView backup,
                  mkseeder::SeedStatesView states,
                  uint32_t n,
                  const uint32_t *count);
  // the refit (4 hits at least) of n merged seeds, with beamline also their beam-line states
  void fitMergedSeeds(Queue &queue,
                      mkseeder::FitInputs const &fit,
                      mkseeder::SeedHitsConstView seeds,
                      mkseeder::SeedStatesView out,
                      uint32_t n,
                      bool beamline);

  // The CPU backends' caches (MkSeederCpuCacheSoA): the global phi and r of the n hits (PostInputs::cpuHits), as
  // SeedHitAccessor::phi(), r() compute them
  void computeGlobalPolarCoordinates(Queue &queue,
                                     MkSeederEventHitsDeviceCollection const &hits,
                                     MkSeederCpuCacheDeviceCollection &cache,
                                     uint32_t n);

  // the frames of the nCov OT1-3 P hits (the layers with a covariance), as extensionLocalFrame() computes them
  // (PostInputs::cpuOTHitFrames), and on a CPU backend the layers' phi pads (PostInputs::cpuLayers)
  void computeOTModuleFrames(Queue &queue,
                             mkseeder::ChainFinderInputs const &view,
                             mkseeder::PostInputs const &post,
                             MkSeederCpuCacheDeviceCollection &cache,
                             uint32_t nCov);

  // The starts of a layer's nb bins (nb + 1 rows from firstBinStart) from the bins of its n hits from firstHit, in
  // bin order.
  void computeBinStarts(Queue &queue,
                        MkSeederEventHitsDeviceCollection &hits,
                        uint32_t firstHit,
                        uint32_t n,
                        uint32_t nb,
                        uint32_t firstBinStart);

  // The cleaning of the quads (clean.quads: clean.cap rows, the count on the device) in the fixed order of
  // their layers, then hits (their rows sorted) on the queue (MkSeederClean.h). The sort's and the cleaning's scratch is one
  // allocation, on a GPU padded into the caching allocator's bin of minBytes (the pool's, freed just before: the
  // allocator hands out the pool's block again).
  void sortAndCleanQuads(Queue &queue, mkseeder::CleanInputs const &clean, size_t minBytes);

  // The hits' bin order (MkSeederBinKernels.h): the bin-order hits, their N-bins (binIndex), the cluster
  // order's binOrderPosition and the clusters' hitPosition from the n unbinned hits (in, inOT; their number on the
  // device: count) and their keys (a stable sort on the device, sort: scratch for n keys); the layers' firstHit and
  // firstOTHit from the tables; the pixel hits' merged-cluster tags and the OT1-3 P hits' PS marks (tags)
  void scatterHitsToBins(Queue &queue,
                         mkseeder::HitsView in,
                         mkseeder::OTHitsView inOT,
                         const uint32_t *count,
                         GlobalRadixSort &sort,
                         uint32_t n,
                         MkSeederEventHitsDeviceCollection &hits,
                         MkSeederEventTablesDeviceCollection const &tables,
                         mkseeder::LayerGeometryView layerGeometry,
                         MkSeederClustersDeviceCollection &clusters,
                         mkseeder::HitTagInputs const &tags);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif

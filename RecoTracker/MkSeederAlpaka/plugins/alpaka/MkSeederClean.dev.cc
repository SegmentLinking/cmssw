#include <algorithm>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/OneToManyAssoc.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkSeederClean.h"
#include "MkSeederCleanKernels.h"
#include "MkSeederCleanStep.h"
#include "MkSeederDupKernels.h"
#include "MkSeederDevice.h"
#include "MkSeederPcaKernels.h"
#include "MkSeederRadixSort.h"
#include "MkSeederScratchSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  using namespace cms::alpakatools;
  // the threads of the one block that completes a cleaning's decisions after its rounds
  constexpr uint32_t kCompleteThreads = 256;

  namespace {
    using ListOffsetsView = ::mkseeder::MkSeederListOffsetsSoA::View;
    using ListItemsView = ::mkseeder::MkSeederListItemsSoA::View;
    using ::mkseeder::MkSeederItemLists;

    // the first n values of a device column set to the byte value
    template <typename T>
    void memsetColumn(Queue &queue, T *device, size_t n, uint8_t value) {
      auto view = make_device_view<T>(alpaka::getDev(queue), device, n);
      alpaka::memset(queue, view, value);
    }

    // the exclusive scan of the n values of in: out[0] = 0, out[i + 1] = in[0] + ... + in[i] (out[n]: the total)
    void exclusiveScan(Queue &queue, uint32_t *in, uint32_t n, uint32_t *out) {
      memsetColumn(queue, out, 1, 0);
      prefixScan<Acc1D>(queue, in, out + 1, n);
    }

    // the lists of an offsets block (its rows - 1 ones and the assoc object) and an items block (their capacity)
    MkSeederItemLists::View listsView(ListOffsetsView offsets, ListItemsView items) {
      return {&offsets.lists(),
              offsets.offset().data(),
              items.item().data(),
              uint32_t(offsets.metadata().size()),
              uint32_t(items.metadata().size())};
    }
    // the lists' storage set (one thread)
    class KernelInitLists {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc, MkSeederItemLists::View view) const {
        if (cms::alpakatools::once_per_grid(acc)) {
          view.assoc->initStorage(view);
        }
      }
    };
    // the lists set up and empty
    MkSeederItemLists *zeroLists(Queue &queue, MkSeederItemLists::View view) {
      memsetColumn(queue, view.offStorage, view.offSize, 0);
      alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(1u, 1u), KernelInitLists{}, view);
      return static_cast<MkSeederItemLists *>(view.assoc);
    }
    // after the counts: the offsets for the fill (the inclusive scan of the counts, in place)
    void finalizeLists(Queue &queue, MkSeederItemLists::View view) {
      prefixScan<Acc1D>(queue, view.offStorage, view.offStorage, view.offSize);
    }
    // after the fill: each list sorted (the fill's order is the atomics')
    void sortLists(Queue &queue, MkSeederItemLists *lists, uint32_t nOnes) {
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(divide_up_by(std::max(nOnes, 1u), 128u), 128u),
                          clean::KernelSortHitLists{},
                          lists,
                          nOnes);
    }
  }  // namespace

  // Cleaning of the quads and triplet seeds. Supported: the pattern order, the quads' cleaning by shared hits, the
  // triplet veto (not dedup_res, dedup_helix and the tier_fk_score tiers: the module throws for those on a GPU).
  // The orders come from the stable radix sort (MkSeederRadixSort.h): the pattern order (index f), the quads' cleaning
  // rank and the triplets' veto order (rank r: the quads first). A quad is kept iff no kept better-ranked quad shares
  // >= dedup hits with it; a triplet iff fewer than tripCleanMin of its hits are used by kept quads or kept
  // better-ranked triplets. The decisions are made in rounds (an item is decided once every better-ranked item it
  // depends on is) and completed in one block (the passes that decide what the rounds left); a dropped quad charges
  // the kept quad that the walk over the hit lists finds first. Hits are numbered by their cluster-order position
  // (MkSeederBinKernels.h), i.e. by layer, then by position in the layer's cluster list; per hit the list of the items
  // using it is sorted by rank after the fill.
  void cleanQuadsAndTriplets(Queue &queue, CleanInputs view, GlobalRadixSort &sort, CleanScratch const &scratch) {
    using namespace clean;
    CleanState v{view, CleanItemsView(scratch.items), HitUseView(scratch.hitUse), nullptr};
    const uint32_t cap = v.cap, nTot = v.nTot;
    if (cap == 0)
      return;
    const auto lists = listsView(scratch.hitItemOffsets, scratch.hitItems);
    v.hitItems = zeroLists(queue, lists);
    memsetColumn(queue, v.hitUse.usedByKeptQuad().data(), std::max(nTot, 1u), 0);

    const auto wn = make_workdiv<Acc1D>(divide_up_by(cap, 128u), 128u);
    // 1. the pattern order (one byte: np < 255); the position in it is the "finish index" f
    alpaka::exec<Acc1D>(queue, wn, KernelFillPatternKeys{}, v, sort.keys());
    const auto patternOrder = sort.sort(queue, v.count, {0});
    // 2. the ranks: the quads by the cleaning key, then the triplets by (tripletFailedMlp, score); ties by f
    alpaka::exec<Acc1D>(queue, wn, KernelFillRankKeys{}, v, patternOrder, sort.keys());
    const auto rankOrder = sort.sort(queue, v.count, {0, 1, 2, 3, 4, 8});
    alpaka::exec<Acc1D>(queue, wn, KernelFillRankFields{}, v, rankOrder);
    // 3. per hit its items by rank
    finalizeLists(queue, lists);
    alpaka::exec<Acc1D>(queue, wn, KernelFillHitLists{}, v);
    sortLists(queue, v.hitItems, nTot);
    // 4. the quads' cleaning (the rounds, then what they left in one block) and n_amb; 5. the triplet veto
    const auto w1 = make_workdiv<Acc1D>(1u, kCompleteThreads);
    if (v.parameters.dedup > 0) {
      for (int k = 0; k < v.parameters.nQuadRounds; ++k)
        alpaka::exec<Acc1D>(queue, wn, KernelCleanQuadsRound{}, v, k);
      alpaka::exec<Acc1D>(queue, w1, KernelCleanQuadsComplete{}, v, v.parameters.nQuadRounds);
      alpaka::exec<Acc1D>(queue, wn, KernelResolveAmbiguousQuads{}, v);
    }
    alpaka::exec<Acc1D>(queue, wn, KernelMarkQuadHitsUsed{}, v);
    for (int k = 0; k < v.parameters.nTripletRounds; ++k)
      alpaka::exec<Acc1D>(queue, wn, KernelCleanTripletsRound{}, v, k);
    alpaka::exec<Acc1D>(queue, w1, KernelCleanTripletsComplete{}, v, v.parameters.nTripletRounds);
    // 6. the kept seeds in finish-index order, their outputs
    alpaka::exec<Acc1D>(queue, wn, KernelFillKeptKeys{}, v, sort.keys());
    const auto keptOrder = sort.sort(queue, v.count, {0});
    alpaka::exec<Acc1D>(queue, wn, KernelBuildKeptSeeds{}, v, keptOrder);
  }

  void cleanDuplicateSeeds(Queue &queue, DupInputs view, FitInputs const &fit) {
    using namespace dup;
    DupState v{view, {}, {}, {}, {}, {}, {}, nullptr, nullptr, nullptr};
    const uint32_t nK = v.nK, nTot = v.nTot;
    if (nK == 0)
      return;
    constexpr uint32_t kSeedHitSlots = ::mkseeder::kMaxSeedHits;
    const uint32_t nB = v.nBins;
    // the state: per input and per seed (nK + 1 rows: the scans' totals), the seeds' hits, helices, merged hits and
    // refits (nK rows), the lists per hit, per no-share bin and per owner seed
    PortableCollection<::mkseeder::MkSeederDupScratchSoA> scratch(queue,
                                                                  int32_t(nK + 1),
                                                                  int32_t(nK + 1),
                                                                  int32_t(nK),
                                                                  int32_t(nK),
                                                                  int32_t(nK),
                                                                  int32_t(nK),
                                                                  int32_t(nTot + 1),
                                                                  int32_t(kSeedHitSlots * nK),
                                                                  int32_t(nB + 1),
                                                                  int32_t(nK),
                                                                  int32_t(nK + 1),
                                                                  int32_t(nK));
    GlobalRadixSort sort(queue, nK);
    auto sv = scratch.view();
    v.inputs = DupInputsView(sv.inputs()), v.seeds = DupSeedsView(sv.seeds());
    v.seedHits = SeedHitsView(sv.seedHits()), v.seedHelices = SeedHelicesView(sv.helices());
    v.merged = SeedHitsView(sv.mergedHits()), v.refits = SeedStatesView(sv.refits());
    const auto hitLists = listsView(sv.hitSeedOffsets(), sv.hitSeeds());
    const auto binLists = listsView(sv.binSeedOffsets(), sv.binSeeds());
    const auto ownerLists = listsView(sv.ownerSeedOffsets(), sv.ownerSeeds());
    v.hitSeeds = zeroLists(queue, hitLists);
    v.binSeeds = zeroLists(queue, binLists);
    v.ownerSeeds = zeroLists(queue, ownerLists);
    memsetColumn(queue, v.inputs.probeDropped().data(), nK, 0);
    const auto wk = make_workdiv<Acc1D>(divide_up_by(nK, 128u), 128u);
    if (v.parameters.pass == 1) {
      if (v.probeCap > 0)
        alpaka::exec<Acc1D>(
            queue, make_workdiv<Acc1D>(divide_up_by(v.probeCap, 128u), 128u), KernelDropProbeLabels{}, v);
      alpaka::exec<Acc1D>(queue, wk, KernelSetKeptSeedFlags{}, v);
      exclusiveScan(queue, v.inputs.isLive().data(), nK, v.inputs.newLabel().data());
      exclusiveScan(queue, v.inputs.isQuadSeed().data(), nK, v.inputs.quadSeedPosition().data());
      exclusiveScan(queue, v.inputs.isTripletSeed().data(), nK, v.inputs.tripletSeedPosition().data());
      alpaka::exec<Acc1D>(queue, wk, KernelBuildSeedItemsFromKept{}, v, sort.keys());
    } else {
      alpaka::exec<Acc1D>(queue, wk, KernelSetPostSeedFlags{}, v);
      exclusiveScan(queue, v.inputs.isQuadSeed().data(), nK, v.inputs.quadSeedPosition().data());
      alpaka::exec<Acc1D>(queue, wk, KernelBuildSeedItemsFromPost{}, v, sort.keys());
    }
    const auto rankOrder = sort.sort(queue, &v.counters->nSeeds, {0, 1, 2, 3, 8});
    alpaka::exec<Acc1D>(queue, wk, KernelDupFields{}, v, rankOrder);
    finalizeLists(queue, hitLists);
    finalizeLists(queue, binLists);
    alpaka::exec<Acc1D>(queue, wk, KernelDupFill{}, v);
    sortLists(queue, v.hitSeeds, nTot);
    sortLists(queue, v.binSeeds, nB);
    for (int k = 0; k < v.parameters.rounds; ++k)
      alpaka::exec<Acc1D>(queue, wk, KernelDupRound{}, v, k);
    alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(1u, kCompleteThreads), KernelDupComplete{}, v, v.parameters.rounds);
    if (v.parameters.merge) {
      alpaka::exec<Acc1D>(queue, wk, KernelOwnerCount{}, v, 0);
      finalizeLists(queue, ownerLists);
      alpaka::exec<Acc1D>(queue, wk, KernelOwnerCount{}, v, 1);
      sortLists(queue, v.ownerSeeds, nK);
      alpaka::exec<Acc1D>(queue, wk, KernelMergeOwnerHits{}, v);
      fitMergedSeeds(queue, fit, SeedHitsConstView(v.merged), v.refits, nK, v.parameters.pass != 1);
    } else
      memsetColumn(queue, v.merged.nHits().data(), nK, 0);
    alpaka::exec<Acc1D>(queue, wk, KernelFlagKeptSeeds{}, v);
    exclusiveScan(queue, v.seeds.isKept().data(), nK, v.seeds.keptPosition().data());
    alpaka::exec<Acc1D>(queue, wk, KernelDupOut{}, v);
  }

  void computeSeedHelices(Queue &queue,
                          SeedStatesConstView states,
                          bool checkFitOk,
                          uint32_t cap,
                          const uint32_t *count,
                          SeedHelicesView helices,
                          float bField) {
    if (cap == 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(cap, 128u), 128u),
                        clean::KernelComputeSeedHelices{},
                        states,
                        checkFitOk,
                        cap,
                        count,
                        helices,
                        bField);
  }

  void buildProbeSeeds(Queue &queue,
                       KeptSeedsConstView kept,
                       HitsView hits,
                       ::mkseeder::MkSeederCleanCounters *counters,
                       SeedHitsView probe,
                       uint32_t nKept) {
    if (nKept == 0)
      return;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(nKept, 128u), 128u),
                        clean::KernelBuildProbeItems{},
                        kept,
                        hits,
                        counters,
                        probe,
                        nKept);
  }

  void computeClassifierFeatures(Queue &queue, ClassifierFeatureInputs view) {
    pca::ClassifierFeatureState v{view, nullptr};
    if (v.cap == 0)
      return;
    // per hit the seeds on it: nTot + 1 offsets, up to ::mkseeder::kMaxSeedHits per seed
    PortableCollection<::mkseeder::MkSeederShareScratchSoA> scratch(
        queue, int32_t(v.nTot + 1), int32_t(size_t(::mkseeder::kMaxSeedHits) * v.cap));
    const auto lists = listsView(scratch.view().hitSeedOffsets(), scratch.view().hitSeeds());
    v.hitSeeds = zeroLists(queue, lists);
    const auto wk = make_workdiv<Acc1D>(divide_up_by(v.cap, 128u), 128u);
    alpaka::exec<Acc1D>(queue, wk, pca::KernelShareCount{}, v, 0);
    finalizeLists(queue, lists);
    alpaka::exec<Acc1D>(queue, wk, pca::KernelShareCount{}, v, 1);
    alpaka::exec<Acc1D>(
        queue, make_workdiv<Acc1D>(divide_up_by(v.cap, 64u), 64u), pca::KernelComputeClassifierFeatures{}, v);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

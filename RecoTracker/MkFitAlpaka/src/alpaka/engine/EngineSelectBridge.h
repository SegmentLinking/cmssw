#ifndef RecoTracker_MkFitAlpaka_src_alpaka_engine_EngineSelectBridge_h
#define RecoTracker_MkFitAlpaka_src_alpaka_engine_EngineSelectBridge_h

// Bridge between the engine (K1 / K3) and the select K2 (src/alpaka/select): entry points defined in
// src/alpaka/Cands.dev.cc. A src/ header because select's SoAs are src/ headers (the public CandsEngine.h must not
// include them). Producers of this package include it directly.
//   engineBuildSelectList  K1 -> K2: dense list of the listed candidates (seed_cand_idx order: seed-major, ic
//                          ascending): an exclusive prefix scan of seeds.nActive in one block + a fill kernel.
//   EngineSelectK2         an EngineSelectFn: list -> select::runSelectHits, which writes the engine rows directly.

#include <functional>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/alpaka/CandsEngine.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/select/SelectEntry.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/select/SelectSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  void engineBuildSelectList(Queue& queue, EngineBuffers& bufs, ::mkfitdev::SelListSoA::View list, int nSeeds);

  // K2 of one step for engineSearch. The caller owns the list / output collections (nSeeds * kMaxCandsPerSeed rows) and
  // the event inputs; nSeeds = capacity in seeds, the current seed count comes from engineSearch (nSeedsNow).
  struct EngineSelectK2 {
    ::mkfitdev::SelListSoA::View list;
    ::mkfitdev::PropStateSoA::View props;
    ::mkfitdev::SelHitsSoA::View sels;
    ::mkfitdev::ESView es;
    ::mkfitdev::LayerSoA::ConstView eohLayers;
    ::mkfitdev::BinnedHitSoA::ConstView eohBinned;
    ::mkfitdev::BinSoA::ConstView eohBins;
    ::mkfitdev::HitSoA::ConstView hits;
    int nSeeds;
    bool groupScan = true;  // GPU: K2 hit scan with kSelLanes lanes per candidate; false: one thread per candidate

    void operator()(Queue& queue, EngineBuffers& bufs, int /*step*/, int nSeedsNow) const {
      engineBuildSelectList(queue, bufs, list, nSeedsNow);
      select::runSelectHits(queue,
                            bufs.slots.const_view(),
                            list,
                            nSeedsNow * ::mkfitdev::kMaxCandsPerSeed,
                            es,
                            eohLayers,
                            eohBinned,
                            eohBins,
                            hits,
                            props,
                            sels,
                            bufs.step->sel.view().metadata().addressOf_prop(),
                            bufs.step->sel.view().metadata().addressOf_sel(),  // K2 writes the engine rows (no scatter)
                            groupScan);
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif

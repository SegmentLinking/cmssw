#ifndef RecoTracker_MkFitAlpaka_src_alpaka_select_SelectEntry_h
#define RecoTracker_MkFitAlpaka_src_alpaka_select_SelectEntry_h

// Host entry point of K2 (kernel instantiated once, in src/alpaka/Select.dev.cc).

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESView.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/EventOfHitsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/select/SelectSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::select {

  constexpr int kSelectBlockSize = 64;

  // nListMax: host upper bound of list.n() (grid size); the kernel processes list.n() rows.
  // engProp / engSel (optional): the clone engine's CandSelHitsSoA prop / sel columns; K2 then also writes the engine
  // rows selRow(seed, ic) of every listed candidate (no separate scatter kernel).
  // GPU backends: groupScan = true runs KernelSelectHitsGroup (kSelLanes lanes per candidate in the hit
  // scan); false = the thread-per-candidate KernelSelectHits (A/B). CPU backends always run KernelSelectHits.
  void runSelectHits(Queue& queue,
                     ::mkfitdev::CandSlotsSoA::ConstView slots,
                     ::mkfitdev::SelListSoA::ConstView list,
                     int nListMax,
                     ::mkfitdev::ESView const& es,
                     ::mkfitdev::LayerSoA::ConstView eohLayers,
                     ::mkfitdev::BinnedHitSoA::ConstView eohBinned,
                     ::mkfitdev::BinSoA::ConstView eohBins,
                     ::mkfitdev::HitSoA::ConstView hits,
                     ::mkfitdev::PropStateSoA::View props,
                     ::mkfitdev::SelHitsSoA::View sels,
                     ::mkfitdev::CandPropState* engProp = nullptr,
                     ::mkfitdev::CandSelHits* engSel = nullptr,
                     bool groupScan = true);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::select

#endif

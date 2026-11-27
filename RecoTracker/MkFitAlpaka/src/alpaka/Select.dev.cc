// K2 (hit selection) of the clone engine: the one translation unit that instantiates the select kernels
// (one TU per component). Kernel code in src/alpaka/select/.
#include "RecoTracker/MkFitAlpaka/src/alpaka/select/SelectEntry.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/select/SelectKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::select {

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
                     ::mkfitdev::CandPropState* engProp,
                     ::mkfitdev::CandSelHits* engSel,
                     bool groupScan) {
    if (nListMax <= 0)
      return;
    if constexpr (kNN == 1) {
      if (groupScan) {
        const auto wdPrep = cms::alpakatools::make_workdiv<Acc1D>(
            cms::alpakatools::divide_up_by(nListMax, kSelPrepBlock), kSelPrepBlock);
        alpaka::exec<Acc1D>(queue,
                            wdPrep,
                            KernelSelectPrep{},
                            slots,
                            list,
                            es,
                            eohLayers,
                            eohBinned,
                            eohBins,
                            props,
                            sels,
                            engProp,
                            engSel);
        const auto wdScan = cms::alpakatools::make_workdiv<Acc1D>(
            cms::alpakatools::divide_up_by(nListMax, kSelScanCands), kSelScanBlock);
        alpaka::exec<Acc1D>(queue,
                            wdScan,
                            KernelSelectScan{},
                            list,
                            es,
                            eohLayers,
                            eohBinned,
                            eohBins,
                            hits,
                            ::mkfitdev::PropStateSoA::ConstView{props},
                            sels,
                            engSel);
        return;
      }
    }
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(
        cms::alpakatools::divide_up_by(nListMax, kSelectBlockSize), kSelectBlockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        KernelSelectHits{},
                        slots,
                        list,
                        es,
                        eohLayers,
                        eohBinned,
                        eohBins,
                        hits,
                        props,
                        sels,
                        engProp,
                        engSel);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::select

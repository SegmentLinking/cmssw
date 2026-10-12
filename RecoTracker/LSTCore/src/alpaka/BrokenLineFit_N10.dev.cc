#include "BrokenLineFit.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  void launchBLFKernelN10(Queue& queue,
                          cms::alpakatools::WorkDiv<alpaka::Dim<Acc1D>> const& workDiv,
                          float bField,
                          TrackCandidatesBaseConst candsBase,
                          HitsBaseConst hitsBase,
#if LST_BLF_PIXEL_HITS
                          SeedHitsConst seedHits,
#endif
                          TrackCandidatesBLFFit fitResults) {
#if LST_BLF_PIXEL_HITS
    alpaka::exec<Acc1D>(queue, workDiv, Kernel_LSTBLFit<10>{}, bField, candsBase, hitsBase, seedHits, fitResults);
#else
    alpaka::exec<Acc1D>(queue, workDiv, Kernel_LSTBLFit<10>{}, bField, candsBase, hitsBase, fitResults);
#endif
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

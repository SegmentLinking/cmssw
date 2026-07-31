#include "BrokenLineFit.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  void launchBLFKernelN12(Queue& queue,
                           cms::alpakatools::WorkDiv<alpaka::Dim<Acc1D>> const& workDiv,
                           float bField,
                           TrackCandidatesBaseConst candsBase,
                           HitsBaseConst hitsBase,
                           TrackCandidatesBLFFit fitResults) {
    alpaka::exec<Acc1D>(queue, workDiv, Kernel_LSTBLFit<6>{}, bField, candsBase, hitsBase, fitResults);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

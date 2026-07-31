#include "BrokenLineFit.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  void launchBLFKernelN6(Queue&, cms::alpakatools::WorkDiv<alpaka::Dim<Acc1D>> const&, float, TrackCandidatesBaseConst, HitsBaseConst, TrackCandidatesBLFFit);
  void launchBLFKernelN8(Queue&, cms::alpakatools::WorkDiv<alpaka::Dim<Acc1D>> const&, float, TrackCandidatesBaseConst, HitsBaseConst, TrackCandidatesBLFFit);
  void launchBLFKernelN10(Queue&, cms::alpakatools::WorkDiv<alpaka::Dim<Acc1D>> const&, float, TrackCandidatesBaseConst, HitsBaseConst, TrackCandidatesBLFFit);
  void launchBLFKernelN12(Queue&, cms::alpakatools::WorkDiv<alpaka::Dim<Acc1D>> const&, float, TrackCandidatesBaseConst, HitsBaseConst, TrackCandidatesBLFFit);
  void launchBLFKernelN14(Queue&, cms::alpakatools::WorkDiv<alpaka::Dim<Acc1D>> const&, float, TrackCandidatesBaseConst, HitsBaseConst, TrackCandidatesBLFFit);

  void launchLSTBrokenLineKernels(Queue& queue,
                                   float bField,
                                   TrackCandidatesBaseConst candsBase,
                                   HitsBaseConst hitsBase,
                                   TrackCandidatesBLFFit fitResults,
                                   unsigned int nTrackCandidates) {
    if (nTrackCandidates == 0)
      return;

    constexpr uint32_t kBlockSize = 64;
    auto const workDiv =
        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nTrackCandidates, kBlockSize), kBlockSize);

    alpaka::exec<Acc1D>(queue, workDiv, Kernel_InitBLFFit{}, fitResults, nTrackCandidates);

    launchBLFKernelN6(queue, workDiv, bField, candsBase, hitsBase, fitResults);
    launchBLFKernelN8(queue, workDiv, bField, candsBase, hitsBase, fitResults);
    launchBLFKernelN10(queue, workDiv, bField, candsBase, hitsBase, fitResults);
    launchBLFKernelN12(queue, workDiv, bField, candsBase, hitsBase, fitResults);
    launchBLFKernelN14(queue, workDiv, bField, candsBase, hitsBase, fitResults);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

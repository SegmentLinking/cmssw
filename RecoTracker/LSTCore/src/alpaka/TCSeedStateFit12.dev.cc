// Compiled only in CMSSW, as the fit (TCSeedStateFit.h).
#ifndef LST_STANDALONE

#include "TCSeedStateFitKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst::tcSeedState {

  // the fits of 12 hits, in a compilation unit of their own
  template void fitCandidates<12>(Queue& queue, ::lst::HitsBaseConst hits, uint32_t nTCs, Scratch const& scratch);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst::tcSeedState

#endif  // LST_STANDALONE

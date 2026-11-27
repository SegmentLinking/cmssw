#ifndef RecoTracker_MkFitAlpaka_src_alpaka_prop_PropMatriplex_h
#define RecoTracker_MkFitAlpaka_src_alpaka_prop_PropMatriplex_h

// Matriplex types and portable math for the propagation: the mplex portable Matriplex (API and element
// order, templated on N) and its Config / Const / vdt / MathUtils, exported into
// ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev by MatriplexBackend.h (MPlexLV<N>, ..., kNN, vdt::, Config::, hipo, ...).

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatriplexBackend.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {
  using idx_t = ::mkfitdev::Matriplex::idx_t;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif

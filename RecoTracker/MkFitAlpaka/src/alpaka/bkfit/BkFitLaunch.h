#ifndef RecoTracker_MkFitAlpaka_src_alpaka_bkfit_BkFitLaunch_h
#define RecoTracker_MkFitAlpaka_src_alpaka_bkfit_BkFitLaunch_h

// Host entry points of the backward fit (MkBuilder::backwardFit with usePropToPlane, no PCA step).
// The kernels live in src/alpaka/bkfit/BkFitKernel.h and are instantiated once, in src/alpaka/BkFit.dev.cc.
//
// Contract with the clone engine (call order as runFunctions.cc):
//   ... findTracksCloneEngine; pre-bkfit filter; compactifyHitStorageForBestCand -> backwardFit -> beginBkwSearch ...
//   Input per seed s with nCands > 0 and no overflow bit: best candidate = slot (s, curBuf, 0) (state + book) and its
//   compacted HoT chain (book.lastHitIdx, nodes in the seed's pool). Output in place: state.par/err/charge,
//   book.chi2 = sum of the backward-fit hit chi2 (forward chi2 dropped) and book.score = the post-backward-fit scorer
//   (bkFitFitTracksProp2Plane copy-out). Only the outlier rejection changes the hit chain and the counters: a rejected
//   hit becomes a miss, then the hole and overlap counts are recounted.

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/alpaka/CandsEngine.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESConfig.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/bkfit/BkFitTypes.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/PropagationFlags.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  // Flags: prop::makePropagationFlags of backward_fit_pflags and the device material (src/alpaka/PropagationFlagsAdapter.h).

  // Production: in place on the clone-engine SoAs, one candidate per seed in [0, nSeeds); hits/modules from the
  // engine's hit inputs (forward or backward table: only x..e22, packed, nPixel, layers[].isPixel/moduleBegin and
  // modules are read).
  void backwardFit(Queue& queue,
                   ::mkfitdev::SeedCandsSoA::ConstView seeds,
                   ::mkfitdev::CandSlotsSoA::View slots,
                   ::mkfitdev::CandHotsSoA::View hots,
                   ::mkfitdev::EngineHitInputs const& hits,
                   ::mkfitdev::prop::PropagationFlags const& pflags,
                   int nSeeds,
                   ::mkfitdev::bkfit::OutlierParams const& outliers);

  // The clone engine's backward-fit hook (EngineBackwardFitFn, CandsEngine.h) calling backwardFit on its buffers.
  // outliers: backward-fit outlier rejection (default = off).
  inline EngineBackwardFitFn makeEngineBackwardFit(::mkfitdev::EngineHitInputs const& hits,
                                                   ::mkfitdev::prop::PropagationFlags const& pflags,
                                                   ::mkfitdev::bkfit::OutlierParams const& outliers = {}) {
    return [hits, pflags, outliers](Queue& queue, EngineBuffers& b, int nSeeds) {
      backwardFit(queue, b.seeds.const_view(), b.slots.view(), b.hots.view(), hits, pflags, nSeeds, outliers);
    };
  }

  // MkBuilder::fit_cands for candidates of findTracksStandardv2p2 (m_cands_from_v2p2): bkFitInputTracks +
  // MkFinder::bkFitFitTracksV2p2 (MkFinderV2p2-BkFit.cc), every hit of a physical layer in path order. Same
  // in-place contract as backwardFit.
  void backwardFitV2p2(Queue& queue,
                       ::mkfitdev::SeedCandsSoA::ConstView seeds,
                       ::mkfitdev::CandSlotsSoA::View slots,
                       ::mkfitdev::CandHotsSoA::View hots,
                       ::mkfitdev::EngineHitInputs const& hits,
                       ::mkfitdev::prop::PropagationFlags const& pflags,
                       int nSeeds,
                       ::mkfitdev::bkfit::OutlierParams const& outliers);

  // The v2p2 chain's backward-fit hook calling backwardFitV2p2 on its buffers.
  inline EngineBackwardFitFn makeV2p2BackwardFit(::mkfitdev::EngineHitInputs const& hits,
                                                 ::mkfitdev::prop::PropagationFlags const& pflags,
                                                 ::mkfitdev::bkfit::OutlierParams const& outliers = {}) {
    return [hits, pflags, outliers](Queue& queue, EngineBuffers& b, int nSeeds) {
      backwardFitV2p2(queue, b.seeds.const_view(), b.slots.view(), b.hots.view(), hits, pflags, nSeeds, outliers);
    };
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif

#ifndef RecoTracker_MkFitAlpaka_src_alpaka_bkfit_BkFitLaunch_h
#define RecoTracker_MkFitAlpaka_src_alpaka_bkfit_BkFitLaunch_h

// Host entry points of the backward fit (MkBuilder::backwardFit with usePropToPlane, no PCA step).
// The kernels live in src/alpaka/bkfit/BkFitKernel.h and are instantiated once, in src/alpaka/BkFit.dev.cc.
//
// Contract with the clone engine (call order as runFunctions.cc:78-87):
//   ... findTracksCloneEngine; pre-bkfit filter; compactifyHitStorageForBestCand -> backwardFit -> beginBkwSearch ...
//   Input per seed s with nCands > 0 and no overflow bit: best candidate = slot (s, curBuf, 0) (state + book) and its
//   compacted HoT chain (book.lastHitIdx, nodes in the seed's pool). Output in place: state.par/err/charge,
//   book.chi2 = sum of the backward-fit hit chi2 (forward chi2 dropped) and book.score = getScoreCand(book, pT)
//   (bkFitFitTracksProp2Plane copy-out). The hit chain
//   and every counter are left untouched.

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/alpaka/CandsEngine.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESConfig.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/bkfit/BkFitTypes.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/PropagationFlags.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  // Flags: prop::propagationFlags(cfg, deviceMaterial, PropStage::BackwardFit) (src/alpaka/PropagationFlagsAdapter.h).

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

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif

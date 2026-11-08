#include <cmath>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Phase2OTCAHitsKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTCAHits {

  using namespace cms::alpakatools;
  using ::phase2OTCAHits::BeamSpotPosition;
  using ::phase2OTCAHits::ModuleHits;

  namespace {
    // one group of threads per CA module, one thread per hit
    struct MakeCAHitsKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::reco::Phase2OTRecHitsConstView recHits,
                                    ModuleHits const* modules,
                                    uint32_t nModules,
                                    BeamSpotPosition beamSpot,
                                    uint32_t nPixelHits,
                                    uint16_t firstDetectorIndex,
                                    ::reco::TrackingRecHitView hits,
                                    ::reco::HitModuleSoAView hitModules) const {
        if (once_per_grid(acc)) {
          hitModules[nModules].moduleStart() = nPixelHits + hits.metadata().size();
          hits.offsetBPIX2() = 0;  // not used for the OT hits
        }
        for (uint32_t module : independent_groups(acc, nModules)) {
          const ModuleHits range = modules[module];
          if (once_per_block(acc))
            hitModules[module].moduleStart() = nPixelHits + range.firstHit;
          for (uint32_t hitInModule : independent_group_elements(acc, range.nHits)) {
            auto const recHit = recHits[range.firstRecHit + hitInModule];
            auto hit = hits[range.firstHit + hitInModule];
            hit.xLocal() = recHit.xLocal();
            hit.yLocal() = recHit.yLocal();
            hit.xerrLocal() = recHit.xerrLocal();
            hit.yerrLocal() = recHit.yerrLocal();
            const double globalX = recHit.xGlobal() - beamSpot.x;
            const double globalY = recHit.yGlobal() - beamSpot.y;
            const double globalZ = recHit.zGlobal() - beamSpot.z;
            hit.xGlobal() = globalX;
            hit.yGlobal() = globalY;
            hit.zGlobal() = globalZ;
            hit.rGlobal() = std::sqrt(globalX * globalX + globalY * globalY);
            hit.iphi() = unsafe_atan2s<7>(globalY, globalX);
            hit.chargeAndStatus() = SiPixelHitStatusAndCharge{};  // charge 0, every status bit 0
            hit.clusterSizeX() = -1;
            hit.clusterSizeY() = -1;
            hit.detectorIndex() = firstDetectorIndex + module;
          }
        }
      }
    };
  }  // namespace

  void makeCAHits(Queue& queue,
                  ::reco::Phase2OTRecHitsConstView recHits,
                  ModuleHits const* modules,
                  uint32_t nModules,
                  BeamSpotPosition beamSpot,
                  uint32_t nPixelHits,
                  uint16_t firstDetectorIndex,
                  ::reco::TrackingRecHitView hits,
                  ::reco::HitModuleSoAView hitModules) {
    // most modules have a few hits
    constexpr uint32_t threadsPerModule = 32;
    auto const workDiv = make_workdiv<Acc1D>(nModules, threadsPerModule);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        MakeCAHitsKernel{},
                        recHits,
                        modules,
                        nModules,
                        beamSpot,
                        nPixelHits,
                        firstDetectorIndex,
                        hits,
                        hitModules);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTCAHits

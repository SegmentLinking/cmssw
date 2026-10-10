#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEPosition.h"

#include "Phase2TrackerRecHitsKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2TrackerRecHits {

  using namespace cms::alpakatools;
  using ::phase2TrackerRecHits::ClusterData;
  using ::phase2TrackerRecHits::ModuleClusters;

  namespace {
    // one group of threads per module, one thread per cluster
    struct MakeRecHitsKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    Phase2StripCPEParamsSoA::ConstView params,
                                    ModuleClusters const* modules,
                                    uint32_t nModules,
                                    ClusterData const* clusters,
                                    ::reco::Phase2OTRecHitsView recHits) const {
        for (uint32_t module : independent_groups(acc, nModules)) {
          const ModuleClusters range = modules[module];
          auto const moduleParams = params[range.detectorIndex - params.firstModuleIndex()];
          for (uint32_t clusterInModule : independent_group_elements(acc, range.nClusters)) {
            const uint32_t index = range.firstCluster + clusterInModule;
            const ClusterData cluster = clusters[index];
            auto const local = phase2StripCPE::localPosition(
                moduleParams.position(), cluster.firstStrip, cluster.column, cluster.size);
            float globalX, globalY, globalZ;
            moduleParams.frame().toGlobal(local.x, local.y, globalX, globalY, globalZ);
            auto recHit = recHits[index];
            recHit.detId() = moduleParams.detId();
            recHit.detectorIndex() = range.detectorIndex;
            recHit.clusterSize() = cluster.size;
            recHit.xLocal() = local.x;
            recHit.yLocal() = local.y;
            recHit.xerrLocal() = moduleParams.xerrLocal();
            recHit.yerrLocal() = moduleParams.yerrLocal();
            recHit.xGlobal() = globalX;
            recHit.yGlobal() = globalY;
            recHit.zGlobal() = globalZ;
          }
        }
      }
    };
  }  // namespace

  void makeRecHits(Queue& queue,
                   Phase2StripCPEParamsSoA::ConstView params,
                   ModuleClusters const* modules,
                   uint32_t nModules,
                   ClusterData const* clusters,
                   ::reco::Phase2OTRecHitsView recHits) {
    if (nModules == 0)
      return;
    // most modules have a few clusters
    constexpr uint32_t threadsPerModule = 32;
    auto const workDiv = make_workdiv<Acc1D>(nModules, threadsPerModule);
    alpaka::exec<Acc1D>(queue, workDiv, MakeRecHitsKernel{}, params, modules, nModules, clusters, recHits);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2TrackerRecHits

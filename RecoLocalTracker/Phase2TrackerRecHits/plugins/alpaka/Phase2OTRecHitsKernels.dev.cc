#include <cstdint>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEPosition.h"

#include "Phase2OTRecHitsKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTRecHits {

  using namespace cms::alpakatools;
  using phase2OTHits::Cluster;

  static_assert(std::is_same_v<decltype(Cluster::module),
                               ::reco::Phase2OTRecHitsSoA::Metadata::ParametersTypeOf_moduleIndex::ValueType>,
                "the rechits store the module row of the cluster");

  namespace {
    // one thread per rechit: the Phase2StripCPE position on the module surface
    struct MakeRecHitsKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    Phase2OTModuleParamsSoA::ConstView modules,
                                    Cluster const* clusters,
                                    ::reco::Phase2OTRecHitsView recHits) const {
        for (uint32_t index : uniform_elements(acc, recHits.metadata().size())) {
          const Cluster cluster = clusters[index];
          auto const module = modules[cluster.module];
          auto const local =
              phase2StripCPE::localPosition(module.cpe(), cluster.firstStrip, cluster.column, cluster.size);
          float globalX, globalY, globalZ;
          module.frame().toGlobal(local.x, local.y, globalX, globalY, globalZ);
          auto recHit = recHits[index];
          recHit.xGlobal() = globalX;
          recHit.yGlobal() = globalY;
          recHit.zGlobal() = globalZ;
          recHit.moduleIndex() = cluster.module;
          recHit.clusterSize() = cluster.size;
        }
      }
    };
  }  // namespace

  void makeRecHits(Queue& queue,
                   Phase2OTModuleParamsSoA::ConstView modules,
                   Cluster const* clusters,
                   ::reco::Phase2OTRecHitsView recHits) {
    const uint32_t nRecHits = recHits.metadata().size();
    constexpr uint32_t threadsPerBlock = 256;
    auto const workDiv = make_workdiv<Acc1D>(divide_up_by(nRecHits, threadsPerBlock), threadsPerBlock);
    alpaka::exec<Acc1D>(queue, workDiv, MakeRecHitsKernel{}, modules, clusters, recHits);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTRecHits

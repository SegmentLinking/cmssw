#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkSeederClean.h"
#include "MkSeederPcaKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  namespace {
    // per seed of the first *count (at most cap) its hits' cluster indices
    struct KernelSeedClusterIndices {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                    mkseeder::SeedHitsConstView seeds,
                                    mkseeder::HitsView hits,
                                    const uint32_t *count,
                                    uint32_t cap,
                                    mkseeder::SeedClusterIndicesView out) const {
        const uint32_t n = alpaka::math::min(acc, *count, cap);
        for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
          const int nHits = seeds.nHits(i);
          for (int k = 0; k < nHits; ++k)
            out.hitClusterIndex(i)(k) = int32_t(hits.clusterIndex(seeds.hitRows(i)(k)));
        }
      }
    };
  }  // namespace

  void setSeedClusterIndices(Queue &queue,
                             mkseeder::SeedHitsConstView seeds,
                             mkseeder::HitsView hits,
                             const uint32_t *count,
                             uint32_t cap,
                             mkseeder::SeedClusterIndicesView out) {
    if (cap == 0)
      return;
    using namespace cms::alpakatools;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(cap, 128u), 128u),
                        KernelSeedClusterIndices{},
                        seeds,
                        hits,
                        count,
                        cap,
                        out);
  }

  void computeSeedPerigees(Queue &queue, mkseeder::PcaInputs const &v) {
    if (v.cap == 0)
      return;
    using namespace cms::alpakatools;
    alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(divide_up_by(v.cap, 64u), 64u), pca::KernelComputeSeedPerigees{}, v);
  }

  void computeSeedPixelStates(Queue &queue, mkseeder::PcaInputs const &v, mkseeder::SeedPixelStatesView out) {
    if (v.cap == 0)
      return;
    using namespace cms::alpakatools;
    alpaka::exec<Acc1D>(
        queue, make_workdiv<Acc1D>(divide_up_by(v.cap, 64u), 64u), pca::KernelComputeSeedPixelStates{}, v, out);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

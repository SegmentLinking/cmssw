#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkFitAlpakaSeedExportKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  namespace {
    struct KernelExportSeedRows {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::mkfitdev::SeedSoAConstView seeds,
                                    int32_t n,
                                    ::mkfitdev::TrackSoAView out) const {
        if (cms::alpakatools::once_per_grid(acc)) {
          out.nTracks() = n;
          out.nOverflowTracks() = 0;
          out.nOverflowHits() = 0;
        }
        for (int32_t s : cms::alpakatools::uniform_elements(acc, n)) {
          auto o = out[s];
          o.params() = seeds[s].params();
          o.errors() = seeds[s].errors();
          o.charge() = seeds[s].charge();
          o.chi2() = seeds[s].chi2();
          o.score() = 0.f;
          o.label() = seeds[s].label();
          const int nh = seeds[s].nHits() < ::mkfitdev::kMaxTrkHits ? seeds[s].nHits() : ::mkfitdev::kMaxTrkHits;
          int nFound = 0;
          for (int k = 0; k < nh; ++k) {
            o.hits().hot[k] = seeds[s].hits().hot[k];
            nFound += seeds[s].hits().hot[k].index >= 0;
          }
          o.nTotalHits() = int16_t(nh);
          o.nFoundHits() = int16_t(nFound);
          o.nSeedHits() = int8_t(nh);
          o.etaRegion() = 0;
          o.algorithm() = 0;
          o.nOverlaps() = 0;
          o.duplicate() = 0;
          o.outer().valid = 0;
        }
      }
    };
  }  // namespace

  void exportSeedRows(Queue& queue, ::mkfitdev::SeedSoAConstView seeds, int32_t n, ::mkfitdev::TrackSoAView out) {
    constexpr uint32_t kThreads = 128;
    const uint32_t blocks = cms::alpakatools::divide_up_by(static_cast<uint32_t>(n > 0 ? n : 1), kThreads);
    alpaka::exec<Acc1D>(
        queue, cms::alpakatools::make_workdiv<Acc1D>(blocks, kThreads), KernelExportSeedRows{}, seeds, n, out);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

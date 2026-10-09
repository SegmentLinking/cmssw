// Top-level translation unit of the LST seed fit: kernel in src/alpaka/seeds/LstSeedKernels.h.
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/alpaka/LstSeedFit.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/seeds/LstSeedKernels.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/seeds/PixSeedKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstseeds {

  void fitLstSeeds(Queue& queue,
                   ::mkfitdev::ESView const& es,
                   ::mkfitdev::HitSoAConstView hits,
                   uint32_t nPixel,
                   ::mkfitdev::SeedSoAView seeds,
                   int32_t n,
                   LstSeedFitConfig const& cfg,
                   ::mkfitdev::lstseeds::PixelSeedRows const& pix,
                   LstSeedFitCounters* counters,
                   int8_t* cls,
                   int8_t* st) {
    if (n <= 0)
      return;
#if !(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED))
    if (cfg.originPrior == kHostCreator) {
      // CPU: kNN-wide Matriplex fits of seeds with the same hit count (as the final fit)
      constexpr uint32_t kElements = 1024;
      const uint32_t groups = cms::alpakatools::divide_up_by(static_cast<uint32_t>(n), kElements);
      alpaka::exec<Acc1D>(queue,
                          cms::alpakatools::make_workdiv<Acc1D>(groups, kElements),
                          KernelLstSeedFitGrouped<kNN>{},
                          es,
                          hits,
                          nPixel,
                          seeds,
                          n,
                          cfg,
                          pix,
                          counters,
                          cls,
                          st);
      return;
    }
#endif
    // heavy kernel (Matriplex N = 1 Kalman chain): block <= 128; 64 as the final fit
    constexpr uint32_t kThreads = 64;
    const uint32_t blocks = cms::alpakatools::divide_up_by(static_cast<uint32_t>(n), kThreads);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(blocks, kThreads),
                        KernelLstSeedFit{},
                        es,
                        hits,
                        nPixel,
                        seeds,
                        n,
                        cfg,
                        pix,
                        counters,
                        cls,
                        st);
  }

  void fitPixelSeeds(Queue& queue,
                     ::mkfitdev::ESView const& es,
                     ::mkfitdev::HitSoAConstView hits,
                     ::mkfitdev::lstseeds::PixSeedIn const* in,
                     ::mkfitdev::lstseeds::PixSeedOut* out,
                     int32_t n,
                     LstSeedFitConfig const& cfg,
                     ::mkfitdev::lstseeds::PixSeedBeam const& beam) {
    if (n <= 0)
      return;
    constexpr uint32_t kThreads = 64;  // heavy kernels: block <= 128, 64 as KernelLstSeedFit
    const uint32_t blocks = cms::alpakatools::divide_up_by(static_cast<uint32_t>(n), kThreads);
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    alpaka::exec<Acc1D>(
        queue, cms::alpakatools::make_workdiv<Acc1D>(blocks, kThreads), KernelPixSeedFit{}, es, hits, in, out, n, cfg);
#else
    // CPU: kNN-wide Matriplex fits of pixel tracks with the same hit count (as the final fit)
    constexpr uint32_t kElements = 1024;
    const uint32_t groups = cms::alpakatools::divide_up_by(static_cast<uint32_t>(n), kElements);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(groups, kElements),
                        KernelPixSeedFitGrouped<kNN>{},
                        es,
                        hits,
                        in,
                        out,
                        n,
                        cfg);
#endif
    // the PCA in two steps through a queue-ordered scratch (5x5 curvilinear error per track): spill-free on sm_89
    auto cov = cms::alpakatools::make_device_buffer<double[]>(queue, n * kPixPcaCov);
    alpaka::exec<Acc1D>(
        queue, cms::alpakatools::make_workdiv<Acc1D>(blocks, kThreads), KernelPixSeedPca{}, out, cov.data(), n, beam);
    alpaka::exec<Acc1D>(
        queue, cms::alpakatools::make_workdiv<Acc1D>(blocks, kThreads), KernelPixSeedPerigee{}, out, cov.data(), n);
  }

  namespace {
    struct KernelApplyPixelSeedStates {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::mkfitdev::SeedSoAView seeds,
                                    int32_t n,
                                    int32_t const* pixIdx,
                                    ::mkfitdev::TrackSoAConstView states,
                                    int32_t nStates) const {
        for (int32_t s : cms::alpakatools::uniform_elements(acc, n)) {
          const int32_t e = pixIdx[s];
          if (e < 0 || e >= nStates || states[e].charge() == 0)
            continue;
          // DEVIATION DEV-7: the pLS seed state from the device creator emulation (replaces hltInitialStepSeeds)
          seeds[s].params() = states[e].params();
          seeds[s].errors() = states[e].errors();
          seeds[s].charge() = states[e].charge();
        }
      }
    };
  }  // namespace

  void applyPixelSeedStates(Queue& queue,
                            ::mkfitdev::SeedSoAView seeds,
                            int32_t n,
                            int32_t const* pixIdx,
                            ::mkfitdev::TrackSoAConstView states,
                            int32_t nStates) {
    if (n <= 0)
      return;
    constexpr uint32_t kThreads = 128;
    const uint32_t blocks = cms::alpakatools::divide_up_by(static_cast<uint32_t>(n), kThreads);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(blocks, kThreads),
                        KernelApplyPixelSeedStates{},
                        seeds,
                        n,
                        pixIdx,
                        states,
                        nStates);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstseeds

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkSeederDevicePixelSeeds.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  namespace {
    using SeedsConstView = ::mkseeder::MkSeederSeedsSoA::ConstView;
    using ScoresConstView = ::mkseeder::MkSeederSeedScoresSoA::ConstView;
    using PixelStatesConstView = ::mkseeder::MkSeederSeedPixelStatesSoA::ConstView;

    // a pLS needs three pixel hits (LSTInputProducer kMinPixelHitsForPLS)
    constexpr int kMinPixelHitsForPLS = 3;

    // 1 for a seed that becomes a row: selected by the classifier, with a perigee (a pixel track), with hits
    struct KernelRowFlags {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                    SeedsConstView seeds,
                                    const uint32_t *count,
                                    ScoresConstView scores,
                                    uint32_t cap,
                                    uint32_t *flags) const {
        const uint32_t n = alpaka::math::min(acc, *count, cap);
        for (uint32_t i : cms::alpakatools::uniform_elements(acc, cap))
          flags[i] = i < n && scores.isSelected(i) && seeds.perigee().status(i) != ::mkseeder::kPerigeeNone &&
                     seeds.hits().nHits(i) > 0;
      }
    };

    // seed i -> row pos[i] - 1 (pos: the inclusive scan of the flags)
    struct KernelFillRows {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                    SeedsConstView seeds,
                                    ScoresConstView scores,
                                    uint32_t cap,
                                    const uint32_t *flags,
                                    const uint32_t *pos,
                                    ::mkfitdev::TrackSoAView rows) const {
        if (cms::alpakatools::once_per_grid(acc))
          rows.nTracks() = int32_t(pos[cap - 1]);
        const auto hits = seeds.hits();
        const auto states = seeds.states();
        const auto clusters = seeds.clusterIndices();
        for (uint32_t i : cms::alpakatools::uniform_elements(acc, cap)) {
          if (!flags[i])
            continue;
          auto r = rows[pos[i] - 1];
          for (int d = 0; d < 6; ++d)
            r.params().v[d] = states.parameters(i)(d);
          for (int d = 0; d < 21; ++d)
            r.errors().v[d] = states.covariance(i)(d);
          r.charge() = int16_t(states.charge(i));
          r.chi2() = states.chi2(i);
          r.score() = scores.score(i);
          r.label() = int32_t(i);  // the seeder's row
          const int nh = hits.nHits(i);
          for (int k = 0; k < nh; ++k) {
            ::mkfitdev::HitOnTrack h;
            h.index = clusters.hitClusterIndex(i)(k);
            h.layer = hits.hitLayers(i)(k);
            r.hits().hot[k] = h;
          }
          r.nTotalHits() = int16_t(nh);
          r.nFoundHits() = int16_t(nh);
          r.nSeedHits() = int8_t(nh);
        }
      }
    };

    // the state the pLS starts from: the state at the last pixel hit (stateAtLastPixelHit), else none (charge 0)
    struct KernelFillPLSStates {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                    SeedsConstView seeds,
                                    PixelStatesConstView pixelStates,
                                    uint32_t cap,
                                    float maxPt,
                                    const uint32_t *flags,
                                    const uint32_t *pos,
                                    ::mkfitdev::TrackSoAView plsStates) const {
        if (cms::alpakatools::once_per_grid(acc))
          plsStates.nTracks() = int32_t(pos[cap - 1]);
        const auto states = seeds.states();
        for (uint32_t i : cms::alpakatools::uniform_elements(acc, cap)) {
          if (!flags[i] || pixelStates.status(i) != ::mkseeder::kPixelStateValid ||
              pixelStates.nPixelHits(i) < kMinPixelHitsForPLS)
            continue;
          // LSTInputProducer: the pT of the seed's state at its last hit
          const float pt = 1.f / states.parameters(i)(3);
          if (maxPt > 0.f && !(pt < maxPt))
            continue;
          auto r = plsStates[pos[i] - 1];
          for (int d = 0; d < 6; ++d)
            r.params().v[d] = pixelStates.parameters(i)(d);
          for (int d = 0; d < 21; ++d)
            r.errors().v[d] = pixelStates.covariance(i)(d);
          r.charge() = int16_t(pixelStates.charge(i));
          r.label() = int32_t(i);
        }
      }
    };
  }  // namespace

  void flagDevicePixelSeeds(Queue &queue,
                            ::mkseeder::MkSeederSeedsSoA::ConstView seeds,
                            const uint32_t *count,
                            ::mkseeder::MkSeederSeedScoresSoA::ConstView scores,
                            uint32_t cap,
                            uint32_t *flags,
                            uint32_t *pos) {
    if (cap == 0)
      return;
    using namespace cms::alpakatools;
    const auto wd = make_workdiv<Acc1D>(divide_up_by(cap, 128u), 128u);
    alpaka::exec<Acc1D>(queue, wd, KernelRowFlags{}, seeds, count, scores, cap, flags);
    prefixScan<Acc1D>(queue, flags, pos, cap);
  }

  void fillDevicePixelSeeds(Queue &queue,
                            ::mkseeder::MkSeederSeedsSoA::ConstView seeds,
                            ::mkseeder::MkSeederSeedScoresSoA::ConstView scores,
                            ::mkseeder::MkSeederSeedPixelStatesSoA::ConstView pixelStates,
                            uint32_t cap,
                            const uint32_t *flags,
                            const uint32_t *pos,
                            DevicePixelSeedsParameters const &p,
                            ::mkfitdev::TrackSoAView rows,
                            ::mkfitdev::TrackSoAView plsStates) {
    if (cap == 0)
      return;
    using namespace cms::alpakatools;
    const auto wd = make_workdiv<Acc1D>(divide_up_by(cap, 128u), 128u);
    alpaka::exec<Acc1D>(queue, wd, KernelFillRows{}, seeds, scores, cap, flags, pos, rows);
    if (p.stateAtLastPixelHit)
      alpaka::exec<Acc1D>(
          queue, wd, KernelFillPLSStates{}, seeds, pixelStates, cap, p.stateAtLastPixelHitMaxPt, flags, pos, plsStates);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

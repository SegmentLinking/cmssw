// Compiled only in CMSSW, as the fit (TCSeedStateFit.h).
#ifndef LST_STANDALONE

#include <Eigen/Core>  // before the SoA headers with Eigen columns

#include <algorithm>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/TCSeedStateFit.h"
#include "RecoTracker/LSTCore/interface/alpaka/TCSeedStateSelection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst::tcSeedState {

  namespace {

    // one thread per track candidate: prepareCandidate, then its entry in the list of its fit size or the counter of
    // why it is not fitted
    struct PrepareFitsKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::lst::LSTInputConstView input,
                                    ::lst::TrackCandidatesBaseConst candidates,
                                    uint32_t nTCs,
                                    FitInput* inputs,
                                    ::lst::FittedState* states,
                                    uint32_t* lists,
                                    Counters* counters) const {
        const InputHits inputHits{input.hits(), input.hitsIT(), input.pixelSeeds(), input.hits().nHitsOT()};
        for (uint32_t candidate : cms::alpakatools::uniform_elements(acc, nTCs)) {
          Collected collected;
          Selection selection;
          const Preparation preparation = prepareCandidate(inputHits,
                                                           candidates.hitIndices()[candidate],
                                                           candidates.trackCandidateType()[candidate],
                                                           collected,
                                                           selection,
                                                           inputs[candidate],
                                                           states[candidate]);
          if (preparation == Preparation::kPLS) {
            alpaka::atomicAdd(acc, &counters->nNotFittedPLS, 1u, alpaka::hierarchy::Blocks{});
          } else if (preparation == Preparation::kTooFewHits) {
            alpaka::atomicAdd(acc, &counters->nTooFewHits, 1u, alpaka::hierarchy::Blocks{});
          } else {
            const uint32_t list = fitList(selection.nFit);
            lists[list * nTCs + alpaka::atomicAdd(acc, &counters->listSize[list], 1u, alpaka::hierarchy::Blocks{})] =
                candidate;
          }
        }
      }
    };

    // One thread per track candidate: its row at its index, so that the rows do not depend on the order of the
    // lists. The per-event scalars once per grid.
    struct FillSeedStatesKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::lst::HitsBaseConst hits,
                                    ::lst::TrackCandidatesBaseConst candidates,
                                    uint32_t nTCs,
                                    ::lst::TCSeedStatesView seedStates,
                                    ::lst::FittedState const* states,
                                    Counters const* counters) const {
        auto state = seedStates.state();
        auto extra = seedStates.extra();
        if (cms::alpakatools::once_per_grid(acc)) {
          state.B() = hits.B();
          extra.nTCs() = nTCs;
          extra.nNotFittedPLS() = counters->nNotFittedPLS;
          extra.nTooFewHits() = counters->nTooFewHits;
          extra.nNonFinite() = counters->nNonFinite;
          extra.nChargeMismatch() = counters->nChargeMismatch;
          extra.nNonPositiveVariance() = counters->nNonPositiveVariance;
        }
        for (uint32_t candidate : cms::alpakatools::uniform_elements(acc, nTCs)) {
          ::lst::FittedState const& fitted = states[candidate];
          auto const& row = fitted.row;
          const ::lst::LSTObjType type = candidates.trackCandidateType()[candidate];
          auto out = state[candidate];
          out.type() = type;
          out.status() = fitted.status;
          out.x() = row.x;
          out.y() = row.y;
          out.z() = row.z;
          out.phi() = row.phi;
          out.cotTheta() = row.cotTheta;
          out.kappa() = row.kappa;
          for (int k = 0; k < 6; ++k)
            state.covCircle()[candidate][k] = row.covCircle[k];
          for (int k = 0; k < 3; ++k)
            state.covLine()[candidate][k] = row.covLine[k];
          out.charge() = row.charge;
          out.nFit() = fitted.nFit;
          out.refFallback() = fitted.refFallback;
          out.pixelSeedIndex() = rowPixelSeedIndex(type, candidates.pixelSeedIndex()[candidate]);
          out.stateHit() = fitted.stateHit;
        }
      }
    };

  }  // namespace

  void prepareFits(Queue& queue,
                   ::lst::LSTInputConstView input,
                   ::lst::TrackCandidatesBaseConst candidates,
                   uint32_t nTCs,
                   Scratch const& scratch) {
    if (nTCs == 0)
      return;
    // one track candidate per thread, with its collected hits in local memory
    constexpr uint32_t kThreads = 64;
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nTCs, kThreads), kThreads),
                        PrepareFitsKernel{},
                        input,
                        candidates,
                        nTCs,
                        scratch.inputs,
                        scratch.states,
                        scratch.lists,
                        scratch.counters);
  }

  void fillSeedStates(Queue& queue,
                      ::lst::HitsBaseConst hits,
                      ::lst::TrackCandidatesBaseConst candidates,
                      uint32_t nTCs,
                      ::lst::TCSeedStatesView seedStates,
                      Scratch const& scratch) {
    constexpr uint32_t kThreads = 256;
    // at least one block, for the per-event scalars
    alpaka::exec<Acc1D>(
        queue,
        cms::alpakatools::make_workdiv<Acc1D>(std::max(1u, cms::alpakatools::divide_up_by(nTCs, kThreads)), kThreads),
        FillSeedStatesKernel{},
        hits,
        candidates,
        nTCs,
        seedStates,
        scratch.states,
        scratch.counters);
  }

  void fitSeedStates(Queue& queue,
                     ::lst::LSTInputConstView input,
                     ::lst::TrackCandidatesBaseConst candidates,
                     uint32_t nTCs,
                     ::lst::TCSeedStatesView seedStates,
                     Scratch const& scratch) {
    auto counters = cms::alpakatools::make_device_view(queue, *scratch.counters);
    alpaka::memset(queue, counters, 0);
    prepareFits(queue, input, candidates, nTCs, scratch);
    fitCandidates<14>(queue, input.hits(), nTCs, scratch);
    fitCandidates<12>(queue, input.hits(), nTCs, scratch);
    fitCandidates<10>(queue, input.hits(), nTCs, scratch);
    fitCandidates<8>(queue, input.hits(), nTCs, scratch);
    fitCandidates<6>(queue, input.hits(), nTCs, scratch);
    fitCandidates<5>(queue, input.hits(), nTCs, scratch);
    static_assert(kNumFitSizes == 6 && kMaxFitHits == 14 && kMinFitHits == 5,
                  "one fitCandidates call and one compilation unit per fit size");
    fillSeedStates(queue, input.hits(), candidates, nTCs, seedStates, scratch);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst::tcSeedState

#endif  // LST_STANDALONE

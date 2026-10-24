#include <Eigen/Core>  // before the SoA headers with Eigen columns

#include <algorithm>
#include <cstdint>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/warpsize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"

#include "LSTInputDeviceKernels.h"
#include "LSTInputDeviceTrackFit.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice {

  using namespace cms::alpakatools;
  using ::lstInputDevice::fitList;

  namespace {

    // the OT module of the CA whose hit indices contain caIndex (caModuleStart is increasing)
    ALPAKA_FN_ACC inline uint32_t caOTModule(uint32_t const* caModuleStart, uint32_t nModules, uint32_t caIndex) {
      uint32_t low = 0, high = nModules;
      while (high - low > 1) {
        const uint32_t middle = (low + high) / 2;
        if (caModuleStart[middle] <= caIndex)
          low = middle;
        else
          high = middle;
      }
      return low;
    }

    // The hits of one pixel track for prepareTrack, in the pixel-track order (inside out): pixel rechits with the beam
    // spot added, or the OT rechits behind the CA's OT hits; their global covariances from the module frames.
    template <typename TrackHits>
    struct PixelTrackHits {
      TrackHits trackHits;
      uint32_t firstHit;
      ::reco::TrackingRecHitConstView pixelHits;
      uint32_t const* caModuleStart;
      uint32_t const* caRecHitStart;
      ::reco::Phase2OTRecHitsConstView otRecHits;
      ::lst::LSTSeedFitModulesConstView modules;
      Settings const& settings;

      ALPAKA_FN_ACC bool isOT(uint32_t k) const { return trackHits[firstHit + k].id() >= settings.nPixelHits; }

      ALPAKA_FN_ACC TrackHit hit(uint32_t k) const {
        const uint32_t caIndex = trackHits[firstHit + k].id();
        TrackHit result;
        float varX, varY;
        uint32_t module;
        if (caIndex < settings.nPixelHits) {
          auto const hit = pixelHits[caIndex];
          result.fit.x = hit.xGlobal() + settings.beamSpotX;
          result.fit.y = hit.yGlobal() + settings.beamSpotY;
          result.fit.z = hit.zGlobal() + settings.beamSpotZ;
          varX = hit.xerrLocal();
          varY = hit.yerrLocal();
          module = hit.detectorIndex();
          result.index = caIndex;
          result.detId = ::lst::kPixelModuleId;
          result.clusterSize = 1;
        } else {
          const uint32_t caModule = caOTModule(caModuleStart, settings.nCAOTModules, caIndex);
          const uint32_t row = caRecHitStart[caModule] + (caIndex - caModuleStart[caModule]);
          auto const hit = otRecHits[row];
          result.fit.x = hit.xGlobal();
          result.fit.y = hit.yGlobal();
          result.fit.z = hit.zGlobal();
          varX = hit.xerrLocal();
          varY = hit.yerrLocal();
          module = hit.detectorIndex();
          result.index = row;
          result.detId = hit.detId();
          result.clusterSize = hit.clusterSize();
        }
        hitCovariance(modules[module].frame(), varX, varY, result.fit.cov);
        return result;
      }
    };

    struct PrepareTracksKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::reco::TrackBlocksConstView tracks,
                                    uint32_t nTracks,
                                    int32_t const* edmIndex,
                                    ::reco::TrackingRecHitConstView pixelHits,
                                    uint32_t const* caModuleStart,
                                    uint32_t const* caRecHitStart,
                                    ::reco::Phase2OTRecHitsConstView otRecHits,
                                    ::lst::LSTSeedFitModulesConstView modules,
                                    Settings settings,
                                    FitInput* inputs,
                                    PixelSegmentCandidate* candidates,
                                    FittedState* states,
                                    FittedState* itStates,
                                    uint32_t* lists,
                                    Counters* counters) const {
        auto const trackView = tracks.tracks();
        auto const trackHits = tracks.trackHits();
        for (uint32_t track : uniform_elements(acc, nTracks)) {
          PixelSegmentCandidate& candidate = candidates[track];
          const uint32_t firstHit = track == 0 ? 0 : trackView[track - 1].hitOffsets();
          const uint32_t nHits = trackView[track].hitOffsets() - firstHit;
          if (edmIndex[track] < 0 || !isFitted(nHits)) {
            candidate.pixelType = ::lst::PixelType::kInvalid;
            candidate.nHits = 0;
            candidate.nITHits = 0;
            candidate.nSlots = 0;
            setUnfit(states[track], ::lst::kInvalidU32Idx);
            setUnfit(itStates[track], ::lst::kInvalidU32Idx);
            alpaka::atomicAdd(acc, &counters->nNotFitted, 1u, alpaka::hierarchy::Blocks{});
            continue;
          }
          const PixelTrackHits<decltype(trackHits)> hits{
              trackHits, firstHit, pixelHits, caModuleStart, caRecHitStart, otRecHits, modules, settings};
          const TrackPreparation preparation = prepareTrack(
              hits, nHits, settings.maxITHitsToDropOTHits, inputs[track], candidate, states[track], itStates[track]);
          const uint32_t list = fitList(preparation.nFit);
          lists[list * nTracks + alpaka::atomicAdd(acc, &counters->listSize[list], 1u, alpaka::hierarchy::Blocks{})] =
              track;
          if (preparation.itCandidate)
            alpaka::atomicAdd(acc, &counters->nITCandidates, 1u, alpaka::hierarchy::Blocks{});
          if (preparation.itFit) {
            const uint32_t itList = ::lstInputDevice::kITFitList;
            lists[itList * nTracks +
                  alpaka::atomicAdd(acc, &counters->listSize[itList], 1u, alpaka::hierarchy::Blocks{})] = track;
          } else if (preparation.itCandidate) {
            alpaka::atomicAdd(acc, &counters->nITTooFewHits, 1u, alpaka::hierarchy::Blocks{});
          }
        }
      }
    };

    struct MakePixelSegmentsKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    uint32_t nTracks,
                                    Settings settings,
                                    FittedState const* states,
                                    PixelSegmentCandidate* candidates,
                                    Counters* counters) const {
        for (uint32_t track : uniform_elements(acc, nTracks)) {
          FittedState const& state = states[track];
          if (state.status != static_cast<uint8_t>(::lst::SeedStateStatus::kOk))
            continue;
          PixelSegmentCandidate& candidate = candidates[track];
          makePixelSegment(state, settings, candidate);
          if (candidate.pcaStatus == ::lst::PixelSegmentClosestApproach::kNotConverged)
            alpaka::atomicAdd(acc, &counters->nPCANotConverged, 1u, alpaka::hierarchy::Blocks{});
          else if (candidate.pcaStatus == ::lst::PixelSegmentClosestApproach::kNonFinite)
            alpaka::atomicAdd(acc, &counters->nPCANonFinite, 1u, alpaka::hierarchy::Blocks{});
        }
      }
    };

    ALPAKA_FN_ACC inline uint32_t isSeed(FittedState const& state) {
      return state.status == static_cast<uint8_t>(::lst::SeedStateStatus::kOk);
    }
    ALPAKA_FN_ACC inline uint32_t isSegment(PixelSegmentCandidate const& candidate) {
      return candidate.pixelType != ::lst::PixelType::kInvalid;
    }
    ALPAKA_FN_ACC inline uint32_t segmentHits(PixelSegmentCandidate const& candidate) {
      return isSegment(candidate) ? candidate.nSlots : 0;
    }

    // exclusive prefix sums over the tracks, in one block: seeds, pLS and pLS hits
    struct IndexPixelSegmentsKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    FittedState const* states,
                                    PixelSegmentCandidate const* candidates,
                                    uint32_t nTracks,
                                    uint32_t* seedIndex,
                                    uint32_t* segmentIndex,
                                    uint32_t* hitOffset,
                                    Totals* totals) const {
        auto& workspace = alpaka::declareSharedVar<uint32_t[cms::alpakatools::warpSize], __COUNTER__>(acc);
        auto& carry = alpaka::declareSharedVar<uint32_t[3], __COUNTER__>(acc);
        uint32_t* const sums[3] = {seedIndex, segmentIndex, hitOffset};
        for (uint32_t track : independent_group_elements(acc, nTracks)) {
          seedIndex[track] = isSeed(states[track]);
          segmentIndex[track] = isSegment(candidates[track]);
          hitOffset[track] = segmentHits(candidates[track]);
        }
        if (cms::alpakatools::once_per_block(acc))
          carry[0] = carry[1] = carry[2] = 0;
        alpaka::syncBlockThreads(acc);
        // blockPrefixScan scans up to warpSize^2 elements at a time; with one thread per block, all at once
        const uint32_t chunk = requires_single_thread_per_block_v<Acc1D>
                                   ? nTracks
                                   : cms::alpakatools::warpSize * cms::alpakatools::warpSize;
        for (uint32_t first = 0; first < nTracks; first += chunk) {
          const uint32_t size = nTracks - first < chunk ? nTracks - first : chunk;
          for (int sum = 0; sum < 3; ++sum) {
            blockPrefixScan(acc, sums[sum] + first, static_cast<int32_t>(size), workspace);
            alpaka::syncBlockThreads(acc);
            for (uint32_t track : independent_group_elements(acc, size))
              sums[sum][first + track] += carry[sum];
            alpaka::syncBlockThreads(acc);
            if (cms::alpakatools::once_per_block(acc))
              carry[sum] = sums[sum][first + size - 1];
            alpaka::syncBlockThreads(acc);
          }
        }
        for (uint32_t track : independent_group_elements(acc, nTracks)) {
          seedIndex[track] -= isSeed(states[track]);
          segmentIndex[track] -= isSegment(candidates[track]);
          hitOffset[track] -= segmentHits(candidates[track]);
        }
        if (cms::alpakatools::once_per_block(acc))
          *totals = Totals{carry[0], carry[1], carry[2]};
      }
    };

    // the OT rows of the LST input: positions, detId and cluster size as the OT rechits,
    // and the global covariance from the module frames, computed as for the OT hits of the fits.
    struct FillOTHitsKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::lst::LSTInputView input,
                                    ::reco::Phase2OTRecHitsConstView otRecHits,
                                    uint32_t nOTHits,
                                    ::lst::LSTSeedFitModulesConstView modules,
                                    float bField) const {
        auto hits = input.hits();
        if (once_per_grid(acc)) {
          hits.nHitsOT() = nOTHits;
          hits.B() = bField;
        }
        for (uint32_t row : uniform_elements(acc, nOTHits)) {
          hits[row].xs() = otRecHits[row].xGlobal();
          hits[row].ys() = otRecHits[row].yGlobal();
          hits[row].zs() = otRecHits[row].zGlobal();
          hits[row].detid() = otRecHits[row].detId();
          hits[row].clustsize() = otRecHits[row].clusterSize();
          float cov[6];
          hitCovariance(modules[otRecHits[row].detectorIndex()].frame(),
                        otRecHits[row].xerrLocal(),
                        otRecHits[row].yerrLocal(),
                        cov);
          for (int k = 0; k < 6; ++k)
            hits.cov()[row][k] = cov[k];
        }
      }
    };

    // the pLS and their hits; the pLS-hit rows of the hits block carry packed kinematics, so their covariance is NaN,
    // and the hitsIT rows the slot hits with the values computed as for the fit.
    struct FillPixelSegmentsKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::lst::LSTInputView input,
                                    uint32_t nOTHits,
                                    PixelSegmentCandidate const* candidates,
                                    uint32_t nTracks,
                                    uint32_t const* seedIndex,
                                    uint32_t const* segmentIndex,
                                    uint32_t const* hitOffset) const {
        constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
        auto hits = input.hits();
        auto hitsIT = input.hitsIT();
        auto segments = input.pixelSeeds();
        for (uint32_t track : uniform_elements(acc, nTracks)) {
          PixelSegmentCandidate const& candidate = candidates[track];
          if (candidate.pixelType == ::lst::PixelType::kInvalid)
            continue;
          const uint32_t firstHit = nOTHits + hitOffset[track];
          for (uint32_t slot = 0; slot < candidate.nSlots; ++slot) {
            float x, y, z;
            ::lst::pixelSegmentHitColumns(candidate.kinematics, slot, x, y, z);
            auto hit = hits[firstHit + slot];
            hit.xs() = x;
            hit.ys() = y;
            hit.zs() = z;
            hit.detid() = candidate.slotDetId[slot];
            hit.clustsize() = candidate.slotClusterSize[slot];
            const uint32_t itRow = hitOffset[track] + slot;
            ::lstInputDevice::FitHit const& fitHit = candidate.slotHits[slot];
            hitsIT[itRow].idxs() = candidate.slotHitIndex[slot];
            hitsIT[itRow].xs() = fitHit.x;
            hitsIT[itRow].ys() = fitHit.y;
            hitsIT[itRow].zs() = fitHit.z;
            for (int k = 0; k < 6; ++k) {
              hits.cov()[firstHit + slot][k] = kNaN;
              hitsIT.cov()[itRow][k] = fitHit.cov[k];
            }
          }
          auto segment = segments[segmentIndex[track]];
          segment.firstHit() = firstHit;
          segment.nHits() = candidate.nHits;
          segment.hitDetBits() = candidate.hitDetBits;
          segment.deltaPhi() = candidate.deltaPhi;
          segment.seedIdx() = seedIndex[track];
          segment.charge() = candidate.charge;
          segment.superbin() = candidate.superbin;
          segment.pixelType() = static_cast<::lst::PixelType>(candidate.pixelType);
          segment.isQuad() = candidate.nHits > 3;
          segment.ptIn() = candidate.ptIn;
          segment.ptErr() = candidate.ptErr;
          segment.px() = candidate.px;
          segment.py() = candidate.py;
          segment.pz() = candidate.pz;
          segment.etaErr() = candidate.etaErr;
          segment.eta() = candidate.eta;
          segment.phi() = candidate.phi;
        }
      }
    };

    template <typename SeedStateView>
    ALPAKA_FN_ACC inline void fillSeedState(SeedStateView view,
                                            uint32_t row,
                                            FittedState const& state,
                                            uint32_t pixelSeedIndex) {
      auto const& fitted = state.row;
      auto out = view[row];
      out.type() = ::lst::LSTObjType::pLS;
      out.status() = state.status;
      out.x() = fitted.x;
      out.y() = fitted.y;
      out.z() = fitted.z;
      out.phi() = fitted.phi;
      out.cotTheta() = fitted.cotTheta;
      out.kappa() = fitted.kappa;
      for (int k = 0; k < 6; ++k)
        view.covCircle()[row][k] = fitted.covCircle[k];
      for (int k = 0; k < 3; ++k)
        view.covLine()[row][k] = fitted.covLine[k];
      out.charge() = fitted.charge;
      out.nFit() = state.nFit;
      out.refFallback() = state.refFallback;
      out.pixelSeedIndex() = pixelSeedIndex;
      out.stateHit() = state.stateHit;
    }

    struct FillSeedStatesKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::lst::PixelSeedStatesView seedStates,
                                    float bField,
                                    FittedState const* states,
                                    FittedState const* itStates,
                                    PixelSegmentCandidate const* candidates,
                                    uint32_t nTracks,
                                    uint32_t const* seedIndex,
                                    Counters const* counters) const {
        auto state = seedStates.state();
        auto itState = seedStates.itState();
        auto extra = seedStates.extra();
        if (once_per_grid(acc)) {
          state.B() = bField;
          itState.B() = bField;
          extra.nTracks() = nTracks;
          extra.nNotFitted() = counters->nNotFitted;
          extra.nNonFinite() = counters->nNonFinite;
          extra.nChargeMismatch() = counters->nChargeMismatch;
          extra.nNonPositiveVariance() = counters->nNonPositiveVariance;
          extra.nITCandidates() = counters->nITCandidates;
          extra.nITTooFewHits() = counters->nITTooFewHits;
          extra.nITNonFinite() = counters->nITNonFinite;
          extra.nITChargeMismatch() = counters->nITChargeMismatch;
          extra.nITNonPositiveVariance() = counters->nITNonPositiveVariance;
          extra.nPCANotConverged() = counters->nPCANotConverged;
          extra.nPCANonFinite() = counters->nPCANonFinite;
        }
        for (uint32_t track : uniform_elements(acc, nTracks)) {
          if (!isSeed(states[track]))
            continue;
          const uint32_t row = seedIndex[track];
          fillSeedState(state, row, states[track], row);
          fillSeedState(itState, row, itStates[track], row);
          extra[row].nITHits() = candidates[track].nITHits;
        }
      }
    };

  }  // namespace

  void prepareTracks(Queue& queue,
                     ::reco::TrackBlocksConstView tracks,
                     uint32_t nTracks,
                     int32_t const* edmIndex,
                     ::reco::TrackingRecHitConstView pixelHits,
                     uint32_t const* caModuleStart,
                     uint32_t const* caRecHitStart,
                     ::reco::Phase2OTRecHitsConstView otRecHits,
                     ::lst::LSTSeedFitModulesConstView modules,
                     Settings const& settings,
                     FitInput* inputs,
                     PixelSegmentCandidate* candidates,
                     FittedState* states,
                     FittedState* itStates,
                     uint32_t* lists,
                     Counters* counters) {
    if (nTracks == 0)
      return;
    constexpr uint32_t kThreads = 64;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(nTracks, kThreads), kThreads),
                        PrepareTracksKernel{},
                        tracks,
                        nTracks,
                        edmIndex,
                        pixelHits,
                        caModuleStart,
                        caRecHitStart,
                        otRecHits,
                        modules,
                        settings,
                        inputs,
                        candidates,
                        states,
                        itStates,
                        lists,
                        counters);
  }

  void fitAllTracks(Queue& queue,
                    uint32_t nTracks,
                    FitInput const* inputs,
                    float bField,
                    uint32_t const* lists,
                    Counters* counters,
                    FittedState* states,
                    FittedState* itStates) {
    fitInnerTrackerTracks(queue, nTracks, inputs, bField, lists, counters, itStates);
    fitTracks<4>(queue, nTracks, inputs, bField, lists, counters, states);
    fitTracks<6>(queue, nTracks, inputs, bField, lists, counters, states);
    fitTracks<8>(queue, nTracks, inputs, bField, lists, counters, states);
    fitTracks<10>(queue, nTracks, inputs, bField, lists, counters, states);
    fitTracks<12>(queue, nTracks, inputs, bField, lists, counters, states);
    fitTracks<14>(queue, nTracks, inputs, bField, lists, counters, states);
    static_assert(::lstInputDevice::kNumFitSizes == 6 && ::lstInputDevice::kMinFitHits == 4 &&
                      ::lstInputDevice::kMaxFitHits == 14,
                  "one fitTracks call and one compilation unit per fit size");
  }

  void makePixelSegments(Queue& queue,
                         uint32_t nTracks,
                         Settings const& settings,
                         FittedState const* states,
                         PixelSegmentCandidate* candidates,
                         Counters* counters) {
    if (nTracks == 0)
      return;
    // one track per thread: the closest approach keeps its 5x5 double matrices in registers and local memory
    constexpr uint32_t kThreads = 64;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(nTracks, kThreads), kThreads),
                        MakePixelSegmentsKernel{},
                        nTracks,
                        settings,
                        states,
                        candidates,
                        counters);
  }

  void indexPixelSegments(Queue& queue,
                          FittedState const* states,
                          PixelSegmentCandidate const* candidates,
                          uint32_t nTracks,
                          uint32_t* seedIndex,
                          uint32_t* segmentIndex,
                          uint32_t* hitOffset,
                          Totals* totals) {
    const uint32_t threads = requires_single_thread_per_block_v<Acc1D> ? 1 : 1024;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(1, threads),
                        IndexPixelSegmentsKernel{},
                        states,
                        candidates,
                        nTracks,
                        seedIndex,
                        segmentIndex,
                        hitOffset,
                        totals);
  }

  void fillLSTInput(Queue& queue,
                    ::lst::LSTInputView input,
                    ::reco::Phase2OTRecHitsConstView otRecHits,
                    uint32_t nOTHits,
                    ::lst::LSTSeedFitModulesConstView modules,
                    float bField,
                    PixelSegmentCandidate const* candidates,
                    uint32_t nTracks,
                    uint32_t const* seedIndex,
                    uint32_t const* segmentIndex,
                    uint32_t const* hitOffset) {
    constexpr uint32_t kThreads = 256;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(std::max(1u, divide_up_by(nOTHits, kThreads)), kThreads),
                        FillOTHitsKernel{},
                        input,
                        otRecHits,
                        nOTHits,
                        modules,
                        bField);
    if (nTracks > 0)
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(divide_up_by(nTracks, kThreads), kThreads),
                          FillPixelSegmentsKernel{},
                          input,
                          nOTHits,
                          candidates,
                          nTracks,
                          seedIndex,
                          segmentIndex,
                          hitOffset);
  }

  void fillSeedStates(Queue& queue,
                      ::lst::PixelSeedStatesView seedStates,
                      float bField,
                      FittedState const* states,
                      FittedState const* itStates,
                      PixelSegmentCandidate const* candidates,
                      uint32_t nTracks,
                      uint32_t const* seedIndex,
                      Counters const* counters) {
    constexpr uint32_t kThreads = 256;
    // at least one block, for the per-event scalars
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(std::max(1u, divide_up_by(nTracks, kThreads)), kThreads),
                        FillSeedStatesKernel{},
                        seedStates,
                        bField,
                        states,
                        itStates,
                        candidates,
                        nTracks,
                        seedIndex,
                        counters);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice

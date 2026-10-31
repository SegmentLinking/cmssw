#include <algorithm>
#include <cmath>
#include <numbers>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/warpsize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "MagneticField/ParametrizedEngine/interface/TkBfield.h"
#include "RecoTracker/MkFitCore/interface/portable/TrackStateJacobians.h"
#include "TrackingTools/PatternTools/interface/beamLineClosestApproach.h"
#include "TrackingTools/TrajectoryState/interface/curvilinear2PerigeeJacobian.h"

#include "LSTInputDeviceKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice {

  using namespace cms::alpakatools;
  using ::lstInputDevice::kMaxSeedHits;
  using ::lstInputDevice::kNoSeed;

  namespace {

    // a 6x6 matrix for the Jacobians, zero on construction
    struct Matrix6 {
      double element[6][6] = {};
      constexpr double& operator()(int row, int col) { return element[row][col]; }
    };

    // the curvilinear errors of the fitted state, as mkfit::TrackState::convertFromCCSToGlbCurvilinear
    ALPAKA_FN_ACC inline void curvilinearErrors(seedFromConsecutiveHits::State const& state, double errors[5][5]) {
      Matrix6 jacobian;
      mkfit::portable::jacobianCCSToCurvilinear(state.parameters[3],
                                                std::cos(state.parameters[4]),
                                                std::sin(state.parameters[4]),
                                                std::cos(state.parameters[5]),
                                                std::sin(state.parameters[5]),
                                                state.charge,
                                                jacobian);
      double ccs[6][6];
      for (int row = 0, packed = 0; row < 6; ++row)
        for (int col = 0; col <= row; ++col, ++packed)
          ccs[row][col] = ccs[col][row] = state.errors[packed];
      for (int row = 0; row < 5; ++row)
        for (int col = 0; col <= row; ++col) {
          double sum = 0.;
          for (int k = 0; k < 6; ++k)
            for (int l = 0; l < 6; ++l)
              sum += jacobian(row, k) * ccs[k][l] * jacobian(col, l);
          errors[row][col] = errors[col][row] = sum;
        }
    }

    // ptError and etaError of the state at the beam line, as LSTInputProducer: its perigee errors
    // (PerigeeConversions::ftsToPerigeeError) in the reco::TrackBase formulas
    ALPAKA_FN_ACC inline void perigeeErrors(beamLineClosestApproach::FreeState const& pca,
                                            magfieldparam::BCycl<float> const& field,
                                            float& ptError,
                                            float& etaError) {
      using curvilinear2PerigeeJacobian::Vector3;
      constexpr float kInverseGeV = 2.99792458e-3f;
      float fieldTesla[3];
      magfieldparam::inTeslaOAE(field, pca.position.x, pca.position.y, pca.position.z, fieldTesla);
      const Vector3 momentum(pca.momentum.x, pca.momentum.y, pca.momentum.z);
      const float pt = momentum.perp();
      curvilinear2PerigeeJacobian::Matrix5 jacobian;
      curvilinear2PerigeeJacobian::compute(
          momentum,
          Vector3(kInverseGeV * fieldTesla[0], kInverseGeV * fieldTesla[1], kInverseGeV * fieldTesla[2]),
          pca.charge / momentum.mag(),
          -kInverseGeV * pca.charge / pt * fieldTesla[2],
          [](double angle, double& sinAngle, double& cosAngle) {
            sinAngle = std::sin(angle);
            cosAngle = std::cos(angle);
          },
          jacobian);
      double perigee[2][2] = {};
      for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 2; ++col)
          for (int a = 0; a < 5; ++a)
            for (int b = 0; b < 5; ++b)
              perigee[row][col] += jacobian(row, a) * pca.error[a][b] * jacobian(col, b);
      ::lst::pixelSegmentMomentumErrors(
          pt, momentum.mag(), momentum.z(), pca.charge, perigee[0][0], perigee[0][1], perigee[1][1], ptError, etaError);
    }

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

    struct FitPixelTracksKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::reco::TrackBlocksConstView tracks,
                                    uint32_t nTracks,
                                    int32_t const* edmIndex,
                                    ::reco::TrackingRecHitConstView pixelHits,
                                    uint32_t const* caModuleStart,
                                    uint32_t const* caRecHitStart,
                                    ::reco::Phase2OTRecHitsConstView otRecHits,
                                    ::lst::LSTSeedFitModulesConstView modules,
                                    magfieldparam::BCycl<float> field,
                                    Settings settings,
                                    PixelSegmentCandidate* candidates) const {
        auto const trackView = tracks.tracks();
        auto const trackHits = tracks.trackHits();
        for (uint32_t track : uniform_elements(acc, nTracks)) {
          PixelSegmentCandidate& candidate = candidates[track];
          candidate.seedStatus = kNoSeed;
          candidate.pixelType = ::lst::PixelType::kInvalid;
          candidate.nSlots = 0;
          if (edmIndex[track] < 0)
            continue;
          const uint32_t firstHit = track == 0 ? 0 : trackView[track - 1].hitOffsets();
          const uint32_t nHits = trackView[track].hitOffsets() - firstHit;
          if (nHits < 3 || nHits > uint32_t(kMaxSeedHits))
            continue;

          // the hits in the pixel-track order (inside out): pixel rechits, or the OT rechits behind the CA's OT hits
          seedFromConsecutiveHits::Hit seedHits[kMaxSeedHits];
          uint32_t hitIndex[kMaxSeedHits], detId[kMaxSeedHits];
          uint16_t clusterSize[kMaxSeedHits];
          bool isOT[kMaxSeedHits];
          for (uint32_t iHit = 0; iHit < nHits; ++iHit) {
            const uint32_t caIndex = trackHits[firstHit + iHit].id();
            float x, y, z, varX, varY;
            uint32_t module;
            isOT[iHit] = caIndex >= settings.nPixelHits;
            if (!isOT[iHit]) {
              auto const hit = pixelHits[caIndex];
              x = hit.xGlobal() + settings.beamSpotX;
              y = hit.yGlobal() + settings.beamSpotY;
              z = hit.zGlobal() + settings.beamSpotZ;
              varX = hit.xerrLocal();
              varY = hit.yerrLocal();
              module = hit.detectorIndex();
              hitIndex[iHit] = caIndex;
              detId[iHit] = ::lst::kPixelModuleId;
              clusterSize[iHit] = 1;
            } else {
              const uint32_t caModule = caOTModule(caModuleStart, settings.nCAOTModules, caIndex);
              const uint32_t row = caRecHitStart[caModule] + (caIndex - caModuleStart[caModule]);
              auto const hit = otRecHits[row];
              x = hit.xGlobal();
              y = hit.yGlobal();
              z = hit.zGlobal();
              varX = hit.xerrLocal();
              varY = hit.yerrLocal();
              module = hit.detectorIndex();
              hitIndex[iHit] = row;
              detId[iHit] = hit.detId();
              clusterSize[iHit] = hit.clusterSize();
            }
            auto const frame = modules[module].frame();
            auto const& rotation = frame.rotation();
            float covariance[6];
            frame.toGlobal(varX, 0.f, varY, covariance);
            seedHits[iHit] = {x,
                              y,
                              z,
                              covariance[0],
                              covariance[1],
                              covariance[2],
                              covariance[3],
                              covariance[4],
                              covariance[5],
                              frame.x(),
                              frame.y(),
                              frame.z(),
                              rotation.zx(),
                              rotation.zy(),
                              rotation.zz(),
                              rotation.xx(),
                              rotation.xy(),
                              rotation.xz(),
                              modules[module].radLen(),
                              modules[module].xi()};
          }

          // the creator's region: the pixel track's vertex and pT (SeedGeneratorFromProtoTracksEDProducer)
          const auto trackState = trackView[track].state();
          const float phi0 = trackState(0), tip = trackState(1), qOverPt = trackState(2);
          const seedFromConsecutiveHits::Region region{settings.beamSpotX + tip * std::sin(phi0),
                                                       settings.beamSpotY - tip * std::cos(phi0),
                                                       1.f / std::abs(qOverPt),
                                                       settings.originRadius * settings.originRadius,
                                                       settings.originHalfLength * settings.originHalfLength};
          seedFromConsecutiveHits::State& state = candidate.state;
          const auto status = seedFromConsecutiveHits::fit(seedHits, nHits, region, settings.fit, state);
          candidate.seedStatus = static_cast<int8_t>(status);
          if (status != seedFromConsecutiveHits::Status::ok)
            continue;

          // LSTInputProducer: the state on the last hit, and its closest approach to the beam line
          const float invPt = state.parameters[3], phiLastHit = state.parameters[4], theta = state.parameters[5];
          const float pxLastHit = std::cos(phiLastHit) / invPt, pyLastHit = std::sin(phiLastHit) / invPt;
          const float pzLastHit = std::cos(theta) / std::sin(theta) / invPt;
          beamLineClosestApproach::FreeState lastHit{{state.parameters[0], state.parameters[1], state.parameters[2]},
                                                     {pxLastHit, pyLastHit, pzLastHit},
                                                     state.charge,
                                                     {}};
          curvilinearErrors(state, lastHit.error);
          float fieldTesla[3];
          magfieldparam::inTeslaOAE(field, lastHit.position.x, lastHit.position.y, lastHit.position.z, fieldTesla);
          const beamLineClosestApproach::BeamLine beamLine{{settings.beamSpotX, settings.beamSpotY, settings.beamSpotZ},
                                                           {settings.beamSlopeX, settings.beamSlopeY, 1.f}};
          beamLineClosestApproach::FreeState pca;
          const bool pcaValid = beamLineClosestApproach::stateNoMaterial(
              lastHit, {fieldTesla[0], fieldTesla[1], fieldTesla[2]}, beamLine, pca);
          float pxPCA = 0.f, pyPCA = 0.f, pzPCA = 0.f, dxy = 0.f, dz = 0.f, ptError = 0.f, etaError = 0.f;
          int charge = 0;
          if (pcaValid) {
            pxPCA = pca.momentum.x;
            pyPCA = pca.momentum.y;
            pzPCA = pca.momentum.z;
            ::lst::pixelSegmentImpactParameters(pca.position.x,
                                                pca.position.y,
                                                pca.position.z,
                                                pxPCA,
                                                pyPCA,
                                                pzPCA,
                                                std::sqrt(pxPCA * pxPCA + pyPCA * pyPCA),
                                                settings.beamSpotX,
                                                settings.beamSpotY,
                                                settings.beamSpotZ,
                                                dxy,
                                                dz);
            perigeeErrors(pca, field, ptError, etaError);
            charge = state.charge;
          }

          // the pLS
          const float ptIn = std::sqrt(pxLastHit * pxLastHit + pyLastHit * pyLastHit);
          const float phiIn = std::atan2(pyLastHit, pxLastHit);
          float deltaPhi = std::atan2(lastHit.position.y, lastHit.position.x) - phiIn;
          if (deltaPhi > std::numbers::pi_v<float>)
            deltaPhi -= 2.f * std::numbers::pi_v<float>;
          else if (deltaPhi <= -std::numbers::pi_v<float>)
            deltaPhi += 2.f * std::numbers::pi_v<float>;
          const auto pixelType = ::lst::pixelSegmentType(ptIn, ptError, deltaPhi, settings.ptCut);
          if (pixelType == ::lst::PixelType::kInvalid)
            continue;
          ::lst::PixelSegmentKinematics& kinematics = candidate.kinematics;
          ::lst::pixelSegmentPCAPosition(
              pxPCA, pyPCA, pzPCA, dxy, dz, kinematics.pcaX, kinematics.pcaY, kinematics.pcaZ);
          kinematics.ptPCA = std::sqrt(pxPCA * pxPCA + pyPCA * pyPCA);
          kinematics.etaPCA = kinematics.ptPCA > 0.f ? std::asinh(pzPCA / kinematics.ptPCA) : 0.f;
          kinematics.phiPCA = std::atan2(pyPCA, pxPCA);
          kinematics.lastHitX = lastHit.position.x;
          kinematics.lastHitY = lastHit.position.y;
          kinematics.lastHitZ = lastHit.position.z;
          kinematics.dxy = dxy;
          kinematics.dz = dz;
          candidate.pixelType = pixelType;
          candidate.charge = charge;
          candidate.nHits = nHits;
          candidate.superbin = ::lst::pixelSegmentSuperbin(kinematics.etaPCA, kinematics.phiPCA, dz);
          candidate.ptIn = ptIn;
          candidate.ptErr = ptError;
          candidate.px = pxLastHit;
          candidate.py = pyLastHit;
          candidate.pz = pzLastHit;
          candidate.etaErr = etaError;
          candidate.eta = std::asinh(pzLastHit / ptIn);
          candidate.phi = phiIn;
          candidate.deltaPhi = deltaPhi;
          candidate.nSlots = nHits < ::lst::kMaxPLSHitsInHitsSoA ? nHits : ::lst::kMaxPLSHitsInHitsSoA;
          for (uint32_t slot = 0; slot < candidate.nSlots; ++slot) {
            const uint32_t iHit = ::lst::pixelSegmentHitOfSlot(slot, candidate.nSlots, nHits);
            candidate.slotDetId[slot] = detId[iHit];
            candidate.slotClusterSize[slot] = clusterSize[iHit];
            candidate.slotHitIndex[slot] = hitIndex[iHit];
          }
          const uint32_t nBits = nHits < ::lst::kMaxPLSHitBitsInHitsSoA ? nHits : ::lst::kMaxPLSHitBitsInHitsSoA;
          uint8_t hitDetBits = 0;
          for (uint32_t bit = 0; bit < nBits; ++bit)
            hitDetBits |= isOT[::lst::pixelSegmentHitOfSlot(bit, nBits, nHits)] << bit;
          candidate.hitDetBits = hitDetBits;
        }
      }
    };

    // exclusive prefix sums over the tracks, in one block: seeds, pLS and pLS hits
    struct IndexPixelSegmentsKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
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
          PixelSegmentCandidate const& candidate = candidates[track];
          seedIndex[track] = candidate.seedStatus == static_cast<int8_t>(seedFromConsecutiveHits::Status::ok);
          segmentIndex[track] = candidate.pixelType != ::lst::PixelType::kInvalid;
          hitOffset[track] = candidate.nSlots;
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
        // inclusive to exclusive
        for (uint32_t track : independent_group_elements(acc, nTracks)) {
          PixelSegmentCandidate const& candidate = candidates[track];
          seedIndex[track] -= candidate.seedStatus == static_cast<int8_t>(seedFromConsecutiveHits::Status::ok);
          segmentIndex[track] -= candidate.pixelType != ::lst::PixelType::kInvalid;
          hitOffset[track] -= candidate.nSlots;
        }
        if (cms::alpakatools::once_per_block(acc))
          *totals = Totals{carry[0], carry[1], carry[2]};
      }
    };

    struct FillOTHitsKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::lst::LSTInputView input,
                                    ::reco::Phase2OTRecHitsConstView otRecHits,
                                    uint32_t nOTHits) const {
        auto hits = input.hits();
        if (once_per_grid(acc))
          hits.nHitsOT() = nOTHits;
        for (uint32_t row : uniform_elements(acc, nOTHits)) {
          hits[row].xs() = otRecHits[row].xGlobal();
          hits[row].ys() = otRecHits[row].yGlobal();
          hits[row].zs() = otRecHits[row].zGlobal();
          hits[row].detid() = otRecHits[row].detId();
          hits[row].clustsize() = otRecHits[row].clusterSize();
        }
      }
    };

    struct FillPixelSegmentsKernel {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::lst::LSTInputView input,
                                    uint32_t nOTHits,
                                    PixelSegmentCandidate const* candidates,
                                    uint32_t nTracks,
                                    uint32_t const* seedIndex,
                                    uint32_t const* segmentIndex,
                                    uint32_t const* hitOffset) const {
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
            hitsIT[hitOffset[track] + slot].idxs() = candidate.slotHitIndex[slot];
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

  }  // namespace

  void fitPixelTracks(Queue& queue,
                      ::reco::TrackBlocksConstView tracks,
                      uint32_t nTracks,
                      int32_t const* edmIndex,
                      ::reco::TrackingRecHitConstView pixelHits,
                      uint32_t const* caModuleStart,
                      uint32_t const* caRecHitStart,
                      ::reco::Phase2OTRecHitsConstView otRecHits,
                      ::lst::LSTSeedFitModulesConstView modules,
                      magfieldparam::BCycl<float> const& field,
                      Settings const& settings,
                      PixelSegmentCandidate* candidates) {
    if (nTracks == 0)
      return;
    // one track per thread: the fit keeps its hits and Matriplexes in registers and local memory
    constexpr uint32_t kThreads = 64;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(nTracks, kThreads), kThreads),
                        FitPixelTracksKernel{},
                        tracks,
                        nTracks,
                        edmIndex,
                        pixelHits,
                        caModuleStart,
                        caRecHitStart,
                        otRecHits,
                        modules,
                        field,
                        settings,
                        candidates);
  }

  void indexPixelSegments(Queue& queue,
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
                        nOTHits);
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

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lstInputDevice

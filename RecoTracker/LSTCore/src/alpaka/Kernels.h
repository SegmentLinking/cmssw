#ifndef RecoTracker_LSTCore_src_alpaka_Kernels_h
#define RecoTracker_LSTCore_src_alpaka_Kernels_h

#include <bit>

#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelQuintupletsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelTripletsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelSegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/QuintupletsSoA.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"
#include "RecoTracker/LSTCore/interface/QuadrupletsSoA.h"
#include "RecoTracker/LSTCore/interface/LSTInputSoA.h"

#include "EtaPhiGrid.h"
#include "PixelQuintupletAccessors.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmQuintupletFromMemory(Quintuplets quintuplets,
                                                             unsigned int quintupletIndex,
                                                             bool secondpass = false) {
    quintuplets.isDup()[quintupletIndex] |= 1 + secondpass;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmPixelTripletFromMemory(PixelTriplets pixelTriplets,
                                                               unsigned int pixelTripletIndex) {
    pixelTriplets.isDup()[pixelTripletIndex] = true;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmPixelQuintupletFromMemory(PixelQuintuplets pixelQuintuplets,
                                                                  unsigned int pixelQuintupletIndex) {
    pixelQuintuplets.isDup()[pixelQuintupletIndex] = true;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmPixelSegmentFromMemory(PixelSegments pixelSegments,
                                                               unsigned int pixelSegmentArrayIndex,
                                                               bool secondpass = false) {
    pixelSegments.isDup()[pixelSegmentArrayIndex] |= 1 + secondpass;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void rmQuadrupletFromMemory(Quadruplets quadruplets,
                                                             unsigned int quadrupletIndex,
                                                             bool secondpass = false) {
    quadruplets.isDup()[quadrupletIndex] |= 1 + secondpass;
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int checkHitsT5(unsigned int ix, unsigned int jx, QuintupletsConst quintuplets) {
    unsigned int hits1[Params_T5::kHits];
    unsigned int hits2[Params_T5::kHits];

    for (int i = 0; i < Params_T5::kHits; i++) {
      hits1[i] = quintuplets.hitIndices()[ix][i];
      hits2[i] = quintuplets.hitIndices()[jx][i];
    }

    int nMatched = 0;
    for (int i = 0; i < Params_T5::kHits; i++) {
      // Skip sentinel values from extended slots
      if (hits1[i] == lst::kTCEmptyHitIdx)
        continue;
      bool matched = false;
      for (int j = 0; j < Params_T5::kHits; j++) {
        if (hits2[j] == lst::kTCEmptyHitIdx)
          continue;
        if (hits1[i] == hits2[j]) {
          matched = true;
          break;
        }
      }
      if (matched) {
        nMatched++;
      }
    }
    return nMatched;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void checkHitspT3(unsigned int ix,
                                                   unsigned int jx,
                                                   PixelTripletsConst pixelTriplets,
                                                   int* matched) {
    int phits1[Params_pLS::kHits];
    int phits2[Params_pLS::kHits];

    for (int i = 0; i < Params_pLS::kHits; i++) {
      phits1[i] = pixelTriplets.hitIndices()[ix][i];
      phits2[i] = pixelTriplets.hitIndices()[jx][i];
    }

    int npMatched = 0;
    for (int i = 0; i < Params_pLS::kHits; i++) {
      bool pmatched = false;
      for (int j = 0; j < Params_pLS::kHits; j++) {
        if (phits1[i] == phits2[j]) {
          pmatched = true;
          break;
        }
      }
      if (pmatched) {
        npMatched++;
      }
    }

    int hits1[Params_T3::kHits];
    int hits2[Params_T3::kHits];

    for (int i = 0; i < Params_T3::kHits; i++) {
      hits1[i] = pixelTriplets.hitIndices()[ix][i + 4];  // Omitting the pLS hits
      hits2[i] = pixelTriplets.hitIndices()[jx][i + 4];  // Omitting the pLS hits
    }

    int nMatched = 0;
    for (int i = 0; i < Params_T3::kHits; i++) {
      bool tmatched = false;
      for (int j = 0; j < Params_T3::kHits; j++) {
        if (hits1[i] == hits2[j]) {
          tmatched = true;
          break;
        }
      }
      if (tmatched) {
        nMatched++;
      }
    }

    matched[0] = npMatched;
    matched[1] = nMatched;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int checkHitsT4(unsigned int ix, unsigned int jx, QuadrupletsConst quadruplets) {
    unsigned int hits1[Params_T4::kHits];
    unsigned int hits2[Params_T4::kHits];

    for (int i = 0; i < Params_T4::kHits; i++) {
      hits1[i] = quadruplets.hitIndices()[ix][i];
      hits2[i] = quadruplets.hitIndices()[jx][i];
    }

    int nMatched = 0;
    for (int i = 0; i < Params_T4::kHits; i++) {
      bool matched = false;
      for (int j = 0; j < Params_T4::kHits; j++) {
        if (hits1[i] == hits2[j]) {
          matched = true;
          break;
        }
      }
      if (matched) {
        nMatched++;
      }
    }
    return nMatched;
  };

  struct RemoveDupQuintupletsAfterBuild {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  Quintuplets quintuplets,
                                  QuintupletsOccupancyConst quintupletsOccupancy,
                                  ObjectRangesConst ranges) const {
      for (unsigned int lowmod : cms::alpakatools::uniform_elements_z(acc, modules.nLowerModules())) {
        unsigned int nQuintuplets_lowmod = quintupletsOccupancy.nQuintuplets()[lowmod];
        int quintupletModuleIndices_lowmod = ranges.quintupletModuleIndices()[lowmod];

        for (unsigned int ix1 : cms::alpakatools::uniform_elements_y(acc, nQuintuplets_lowmod)) {
          unsigned int ix = quintupletModuleIndices_lowmod + ix1;
          if (quintuplets.isDup()[ix])
            continue;
          float eta1 = __H2F(quintuplets.eta()[ix]);
          float phi1 = __H2F(quintuplets.phi()[ix]);
          float dnnScore1 = quintuplets.dnnScore()[ix];

          for (unsigned int jx1 : cms::alpakatools::uniform_elements_x(acc, ix1 + 1, nQuintuplets_lowmod)) {
            unsigned int jx = quintupletModuleIndices_lowmod + jx1;
            if (quintuplets.isDup()[jx])
              continue;

            float eta2 = __H2F(quintuplets.eta()[jx]);
            float phi2 = __H2F(quintuplets.phi()[jx]);
            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);

            if (dEta > 0.1f)
              continue;

            if (alpaka::math::abs(acc, dPhi) > 0.1f)
              continue;

            int nMatched = checkHitsT5(ix, jx, quintuplets);
            // Proportional sharing: at least 60% of the shorter track's hits.
            unsigned int nLayersIx = quintuplets.nLayers()[ix];
            unsigned int nLayersJx = quintuplets.nLayers()[jx];
            unsigned int nHitsIx = 2 * nLayersIx;
            unsigned int nHitsJx = 2 * nLayersJx;
            int minNHitsForDup = static_cast<int>(0.6f * (nHitsIx < nHitsJx ? nHitsIx : nHitsJx));
            if (nMatched >= minNHitsForDup) {
              // Tiebreak: longer track wins; otherwise the higher DNN score.
              if (nLayersIx > nLayersJx) {
                rmQuintupletFromMemory(quintuplets, jx);
              } else if (nLayersJx > nLayersIx) {
                rmQuintupletFromMemory(quintuplets, ix);
              } else if (dnnScore1 <= quintuplets.dnnScore()[jx]) {
                rmQuintupletFromMemory(quintuplets, ix);
              } else {
                rmQuintupletFromMemory(quintuplets, jx);
              }
            }
          }
        }
      }
    }
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void tryExtendT5(
      TAcc const& acc, Quintuplets quintuplets, unsigned int winnerIdx, unsigned int loserIdx, int loserSlot) {
    if (loserSlot < 0)
      return;

    unsigned int newSlot = alpaka::atomicAdd(acc, &quintuplets.nLayers()[winnerIdx], 1u, alpaka::hierarchy::Threads{});

    if (newSlot >= Params_T5::kLayers) {
      alpaka::atomicSub(acc, &quintuplets.nLayers()[winnerIdx], 1u, alpaka::hierarchy::Threads{});
      return;
    }

    quintuplets.logicalLayers()[winnerIdx][newSlot] = quintuplets.logicalLayers()[loserIdx][loserSlot];
    quintuplets.lowerModuleIndices()[winnerIdx][newSlot] = quintuplets.lowerModuleIndices()[loserIdx][loserSlot];
    quintuplets.hitIndices()[winnerIdx][2 * newSlot] = quintuplets.hitIndices()[loserIdx][2 * loserSlot];
    quintuplets.hitIndices()[winnerIdx][2 * newSlot + 1] = quintuplets.hitIndices()[loserIdx][2 * loserSlot + 1];
  }

  //kT5DuplicateMinSharedHits = 8 variant of what was previously ExtendT5FromDupT5
  //the initial variant preserves its logic re T5s both starting in B1 (not checked)
  struct ExtendT5FromDupT5ByMD {
    // Packed [score:32 | T5 index:28 | layer slot:4] for atomic best-per-OT-layer tracking.
    static constexpr int kPackedScoreShift = 32;
    static constexpr int kPackedIndexShift = 4;
    static constexpr unsigned int kPackedIndexMask = 0xFFFFFFF;
    static constexpr unsigned int kPackedSlotMask = 0xF;
    static constexpr int kT5DuplicateMinSharedHits = 8;  //can not change

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  Quintuplets quintuplets,
                                  QuintupletsOccupancyConst quintupletsOccupancy,
                                  QuintupletsRangesConst quintupletsRangesByMD0,
                                  QuintupletsByMDConst quintupletsByMD0,
                                  QuintupletsRangesConst quintupletsRangesByMD1,
                                  QuintupletsByMDConst quintupletsByMD1,
                                  TripletsConst triplets,
                                  SegmentsConst segments) const {
      // Best candidate per OT logical layer (1..11), packed score|index|slot.
      uint64_t* sharedBestPacked = alpaka::declareSharedVar<uint64_t[lst::kLogicalOTLayers], __COUNTER__>(acc);

      // One block per T5 in 1D; block index = ref T5 index.
      const unsigned int refT5Index = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];

      // Skip empty/unallocated T5 slots.
      if (quintuplets.nLayers()[refT5Index] == 0)
        return;

      // Initialize shared memory once per block.
      if (cms::alpakatools::once_per_block(acc)) {
        for (int logicalLayerBin = 0; logicalLayerBin < lst::kLogicalOTLayers; ++logicalLayerBin) {
          sharedBestPacked[logicalLayerBin] = 0;
        }
      }
      alpaka::syncBlockThreads(acc);

      const float baseEta = __H2F(quintuplets.eta()[refT5Index]);
      const float basePhi = __H2F(quintuplets.phi()[refT5Index]);
      const uint8_t refStartLogicalLayer = quintuplets.logicalLayers()[refT5Index][0];

      // Hoist ref data once: hit indices and embedding read every candidate iteration otherwise.
      float refEmbed[Params_T5::kEmbed];
      CMS_UNROLL_LOOP
      for (unsigned int e = 0; e < Params_T5::kEmbed; ++e)
        refEmbed[e] = quintuplets.t5Embed()[refT5Index][e];

      constexpr unsigned int kRefHits = 2 * Params_T5::kBaseLayers;
      unsigned int refHits[kRefHits];
      CMS_UNROLL_LOOP
      for (unsigned int h = 0; h < kRefHits; ++h)
        refHits[h] = quintuplets.hitIndices()[refT5Index][h];

      const auto threadIndexFlat = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc).x();
      const auto blockDimFlat = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc).x();

      const bool lookForwardOnly = (refStartLogicalLayer == 1);
      const auto refLS0Index = triplets.segmentIndices()[quintuplets.tripletIndices()[refT5Index][0]][0];
      const auto& mdIndices = segments.mdIndices();
      const auto refMD0Index = mdIndices[refLS0Index][0];
      const auto refMD1Index = mdIndices[refLS0Index][1];
      const auto refModuleIndex1 = quintuplets.lowerModuleIndices()[refT5Index][1];
      const bool refMD1HasT5s = quintupletsOccupancy.nQuintuplets()[refModuleIndex1];

      const auto refLS1Index = triplets.segmentIndices()[quintuplets.tripletIndices()[refT5Index][0]][1];
      const auto refLS3Index = triplets.segmentIndices()[quintuplets.tripletIndices()[refT5Index][1]][1];
      //MDs 1, 2, 3, 4; MD to LS: 0:0_0, 1:0_1/1_0, 2:1_1/2_0, 3:2_1/3_0, 4:3_1
      const uint32_t refMD1234Bar = (mdIndices[refLS1Index][0] & kT5ByMDBarCodeMask) |
                                    ((mdIndices[refLS1Index][1] & kT5ByMDBarCodeMask) << kT5ByMDBarOffset) |
                                    ((mdIndices[refLS3Index][0] & kT5ByMDBarCodeMask) << (kT5ByMDBarOffset * 2)) |
                                    ((mdIndices[refLS3Index][1] & kT5ByMDBarCodeMask) << (kT5ByMDBarOffset * 3));

      auto testT5 = [&](unsigned int testT5Index) {
        // Per-T5 eta/phi window.
        const float candidateEta = __H2F(quintuplets.eta()[testT5Index]);
        if (alpaka::math::abs(acc, baseEta - candidateEta) > 0.1f)
          return;

        const float candidatePhi = __H2F(quintuplets.phi()[testT5Index]);
        if (alpaka::math::abs(acc, cms::alpakatools::deltaPhi(acc, basePhi, candidatePhi)) > 0.1f)
          return;

        // Embedding distance against hoisted refEmbed.
        float embedDistance2 = 0.f;
        CMS_UNROLL_LOOP
        for (unsigned int embedIndex = 0; embedIndex < Params_T5::kEmbed; ++embedIndex) {
          const float diff = refEmbed[embedIndex] - quintuplets.t5Embed()[testT5Index][embedIndex];
          embedDistance2 += diff * diff;
        }
        if (embedDistance2 > 1.0f)
          return;

        int unmatchedLayerSlot = -1;
        // Hit matching against hoisted ref hits; record the candidate slot with no shared hit.
        int sharedHitCount = 0;
        CMS_UNROLL_LOOP
        for (unsigned int layerIndex = 0; layerIndex < Params_T5::kBaseLayers; ++layerIndex) {
          const unsigned int candidateHit0 = quintuplets.hitIndices()[testT5Index][2 * layerIndex + 0];
          const unsigned int candidateHit1 = quintuplets.hitIndices()[testT5Index][2 * layerIndex + 1];

          bool hit0InBase = false;
          bool hit1InBase = false;
          CMS_UNROLL_LOOP
          for (unsigned int baseHitIndex = 0; baseHitIndex < kRefHits; ++baseHitIndex) {
            const unsigned int baseHit = refHits[baseHitIndex];
            hit0InBase = hit0InBase || (candidateHit0 == baseHit);
            hit1InBase = hit1InBase || (candidateHit1 == baseHit);
          }

          sharedHitCount += int(hit0InBase) + int(hit1InBase);
          if (!hit0InBase && !hit1InBase)
            unmatchedLayerSlot = layerIndex;
        }

        if (sharedHitCount < kT5DuplicateMinSharedHits)
          return;
        if (unmatchedLayerSlot < 0)
          return;

        // Score = DNN output; layer bin = candidate's unmatched OT layer (1..11) - 1.
        const float candidateScore = quintuplets.dnnScore()[testT5Index];
        const uint8_t newLogicalLayer = quintuplets.logicalLayers()[testT5Index][unmatchedLayerSlot];
        const int logicalLayerBin = static_cast<int>(newLogicalLayer) - 1;

        uint64_t scoreBits = std::bit_cast<uint32_t>(candidateScore);
        uint64_t newPacked = (scoreBits << kPackedScoreShift) |
                             (static_cast<uint64_t>(testT5Index & kPackedIndexMask) << kPackedIndexShift) |
                             (unmatchedLayerSlot & kPackedSlotMask);

        // Atomic CAS into shared best-per-layer slot, retry until we win or are beaten.
        uint64_t oldPacked = sharedBestPacked[logicalLayerBin];
        while (true) {
          const float oldScore = std::bit_cast<float>(static_cast<uint32_t>(oldPacked >> kPackedScoreShift));
          if (candidateScore <= oldScore)
            break;

          uint64_t assumedOld = alpaka::atomicCas(
              acc, &sharedBestPacked[logicalLayerBin], oldPacked, newPacked, alpaka::hierarchy::Threads{});

          if (assumedOld == oldPacked) {
            break;
          } else {
            oldPacked = assumedOld;
          }
        }
      };  // testT5()

      constexpr uint32_t k3Mask = 0xFFFFFF;
      constexpr uint32_t k2Mask = 0xFFFF;
      if (refMD1HasT5s) {
        const auto testT5ByMDOffset = quintupletsRangesByMD0.offset()[refMD1Index];
        const auto testT5ByMDMax = quintupletsRangesByMD0.n()[refMD1Index];
        for (auto idx = threadIndexFlat; idx < testT5ByMDMax; idx += blockDimFlat) {
          const auto testT5ByMDIndex = testT5ByMDOffset + idx;
          const auto testT5Index = quintupletsByMD0.quintupletIndex()[testT5ByMDIndex];
          if (testT5Index == refT5Index)
            continue;

          const uint32_t refBar234 = refMD1234Bar >> kT5ByMDBarOffset;
          const uint32_t testBarFull = quintupletsByMD0.mdBarCode()[testT5ByMDIndex];
          //check ref234 with test 123x, x234, 1x34, and 13x4
          if ((testBarFull & k3Mask) == refBar234 || (testBarFull >> kT5ByMDBarOffset) == refBar234 ||
              ((testBarFull & kT5ByMDBarCodeMask) | ((testBarFull >> kT5ByMDBarOffset) & ~kT5ByMDBarCodeMask)) ==
                  refBar234 ||
              ((testBarFull & k2Mask) | ((testBarFull >> kT5ByMDBarOffset) & ~k2Mask)) == refBar234)
            testT5(testT5Index);
        }
      }
      if (not lookForwardOnly) {
        {
          const auto testT5ByMDOffset = quintupletsRangesByMD1.offset()[refMD0Index];
          const auto testT5ByMDMax = quintupletsRangesByMD1.n()[refMD0Index];
          for (auto idx = threadIndexFlat; idx < testT5ByMDMax; idx += blockDimFlat) {
            const auto testT5ByMDIndex = testT5ByMDOffset + idx;
            const auto testT5Index = quintupletsByMD1.quintupletIndex()[testT5ByMDIndex];
            if (testT5Index == refT5Index)
              continue;

            const uint32_t testBar234 = quintupletsByMD1.mdBarCode()[testT5ByMDIndex] >> kT5ByMDBarOffset;
            //check test234 with ref 123 and 234 (covers a gap in first logical layers)
            if (testBar234 == (refMD1234Bar & k3Mask) || (testBar234 == (refMD1234Bar >> kT5ByMDBarOffset)) ||
                testBar234 == ((refMD1234Bar & kT5ByMDBarCodeMask) |
                               ((refMD1234Bar >> kT5ByMDBarOffset) & ~kT5ByMDBarCodeMask)) ||
                testBar234 == ((refMD1234Bar & k2Mask) | ((refMD1234Bar >> kT5ByMDBarOffset) & ~k2Mask)))
              testT5(testT5Index);
          }
        }
        {  //should be with lookForwardOnly as well (but not covered in ExtendT5FromDupT5)
          const auto testT5ByMDOffset = quintupletsRangesByMD1.offset()[refMD1Index];
          const auto testT5ByMDMax = quintupletsRangesByMD1.n()[refMD1Index];
          for (auto idx = threadIndexFlat; idx < testT5ByMDMax; idx += blockDimFlat) {
            const auto testT5ByMDIndex = testT5ByMDOffset + idx;
            const auto testT5Index = quintupletsByMD1.quintupletIndex()[testT5ByMDIndex];
            if (testT5Index == refT5Index)
              continue;
            const auto testBar1234 = quintupletsByMD1.mdBarCode()[testT5ByMDIndex];
            if ((testBar1234 & kT5ByMDBarCodeMask) == refStartLogicalLayer)
              continue;

            const uint32_t testBar234 = testBar1234 >> kT5ByMDBarOffset;
            //check test234 with ref234 (the only option here)
            if (testBar234 == (refMD1234Bar >> kT5ByMDBarOffset))
              testT5(testT5Index);
          }
        }
      }

      alpaka::syncBlockThreads(acc);

      // One thread per block applies the per-layer winners.
      if (cms::alpakatools::once_per_block(acc)) {
        CMS_UNROLL_LOOP
        for (int logicalLayerBin = 0; logicalLayerBin < lst::kLogicalOTLayers; ++logicalLayerBin) {
          uint64_t bestPacked = sharedBestPacked[logicalLayerBin];
          if ((bestPacked >> kPackedScoreShift) == 0)
            continue;

          const int bestT5Index = static_cast<int>((bestPacked >> kPackedIndexShift) & kPackedIndexMask);
          const int bestT5LayerSlot = static_cast<int>(bestPacked & kPackedSlotMask);

          tryExtendT5(acc, quintuplets, refT5Index, bestT5Index, bestT5LayerSlot);
        }
      }
    }
  };

  // Eta-phi grid of the T5s alive before the TC stage. Cells are wider than the 0.1 dEta/dPhi window of
  // RemoveDupQuintupletsBeforeTC, so every pair inside the window is in the 3x3 neighbourhood of either T5.
  namespace t5DupGrid {
    constexpr int kNEta = 64;
    constexpr float kEtaMin = -4.f;
    constexpr float kInvEtaWidth = 8.f;  // 0.125 per cell
    constexpr int kNPhi = 50;            // 2 pi / 50 = 0.126 per cell
    constexpr int kNCells = kNEta * kNPhi;

    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE int cell(TAcc const& acc, float eta, float phi) {
      // Out-of-range values are clamped into the edge cells, which keeps the neighbourhood a superset.
      const float etaCell =
          alpaka::math::min(acc, alpaka::math::max(acc, (eta - kEtaMin) * kInvEtaWidth, 0.f), kNEta - 1.f);
      const float phiCell =
          alpaka::math::min(acc, alpaka::math::max(acc, (phi + kPi) * (kNPhi / (2.f * kPi)), 0.f), kNPhi - 1.f);
      return static_cast<int>(etaCell) * kNPhi + static_cast<int>(phiCell);
    }
  }  // namespace t5DupGrid

  struct CountT5DupGrid {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  QuintupletsConst quintuplets,
                                  QuintupletsOccupancyConst quintupletsOccupancy,
                                  ObjectRangesConst ranges,
                                  unsigned int* cellCount) const {
      for (unsigned int lowmodIdx : cms::alpakatools::uniform_elements_y(acc, ranges.nEligibleT5Modules())) {
        const uint16_t lowmod = ranges.indicesOfEligibleT5Modules()[lowmodIdx];
        const unsigned int nQuintuplets = quintupletsOccupancy.nQuintuplets()[lowmod];
        const unsigned int first = ranges.quintupletModuleIndices()[lowmod];
        for (unsigned int i : cms::alpakatools::uniform_elements_x(acc, nQuintuplets)) {
          const unsigned int ix = first + i;
          if (quintuplets.isDup()[ix] & 1)
            continue;
          const int cell = t5DupGrid::cell(acc, __H2F(quintuplets.eta()[ix]), __H2F(quintuplets.phi()[ix]));
          alpaka::atomicAdd(acc, &cellCount[cell], 1u, alpaka::hierarchy::Threads{});
        }
      }
    }
  };

  // Single block: cellStart = exclusive prefix sum of cellCount (nCells + 1 entries); cellCount becomes the fill cursor.
  struct ScanCellCounts {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  unsigned int* cellCount,
                                  unsigned int* cellStart,
                                  unsigned int nCells) const {
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));
      if constexpr (cms::alpakatools::requires_single_thread_per_block_v<Acc1D>) {
        unsigned int running = 0;
        for (unsigned int c = 0; c < nCells; ++c) {
          const unsigned int count = cellCount[c];
          cellStart[c] = running;
          cellCount[c] = running;
          running += count;
        }
        cellStart[nCells] = running;
      } else {
        constexpr unsigned int kMaxThreads = 1024;
        auto& partial = alpaka::declareSharedVar<unsigned int[kMaxThreads], __COUNTER__>(acc);
        auto& warpSums = alpaka::declareSharedVar<unsigned int[kMaxThreads / 16], __COUNTER__>(acc);
        const unsigned int nThreads = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
        const unsigned int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
        ALPAKA_ASSERT_ACC(nThreads <= kMaxThreads);
        const unsigned int chunk = cms::alpakatools::divide_up_by(nCells, nThreads);
        const unsigned int begin = cms::alpakatools::idx_min(tid * chunk, nCells);
        const unsigned int end = cms::alpakatools::idx_min(begin + chunk, nCells);
        unsigned int sum = 0;
        for (unsigned int c = begin; c < end; ++c)
          sum += cellCount[c];
        partial[tid] = sum;
        alpaka::syncBlockThreads(acc);
        cms::alpakatools::blockPrefixScan(acc, partial, static_cast<int32_t>(nThreads), warpSums);  // inclusive
        unsigned int running = partial[tid] - sum;
        for (unsigned int c = begin; c < end; ++c) {
          const unsigned int count = cellCount[c];
          cellStart[c] = running;
          cellCount[c] = running;
          running += count;
        }
        if (tid == nThreads - 1)
          cellStart[nCells] = partial[tid];
      }
    }
  };

  struct FillT5DupGrid {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  QuintupletsConst quintuplets,
                                  QuintupletsOccupancyConst quintupletsOccupancy,
                                  ObjectRangesConst ranges,
                                  unsigned int* cellCursor,
                                  unsigned int* cellEntries) const {
      for (unsigned int lowmodIdx : cms::alpakatools::uniform_elements_y(acc, ranges.nEligibleT5Modules())) {
        const uint16_t lowmod = ranges.indicesOfEligibleT5Modules()[lowmodIdx];
        const unsigned int nQuintuplets = quintupletsOccupancy.nQuintuplets()[lowmod];
        const unsigned int first = ranges.quintupletModuleIndices()[lowmod];
        for (unsigned int i : cms::alpakatools::uniform_elements_x(acc, nQuintuplets)) {
          const unsigned int ix = first + i;
          if (quintuplets.isDup()[ix] & 1)
            continue;
          const int cell = t5DupGrid::cell(acc, __H2F(quintuplets.eta()[ix]), __H2F(quintuplets.phi()[ix]));
          const unsigned int slot = alpaka::atomicAdd(acc, &cellCursor[cell], 1u, alpaka::hierarchy::Threads{});
          cellEntries[slot] = ix;
        }
      }
    }
  };

  // pT5s by the (eta, phi) of their pLS on the same cells, for CrossCleanpT3 (its window is far below a cell).
  struct CountPixelQuintupletSeedGrid {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  ObjectRangesConst ranges,
                                  PixelSeedsConst pixelSeeds,
                                  PixelQuintupletsConst pixelQuintuplets,
                                  unsigned int* cellCount) const {
      const unsigned int prefix = ranges.segmentModuleIndices()[modules.nLowerModules()];
      for (unsigned int i : cms::alpakatools::uniform_elements(acc, pixelQuintuplets.nPixelQuintuplets())) {
        const unsigned int pLS = pixelQuintuplets.pixelSegmentIndices()[i] - prefix;
        const int cell = t5DupGrid::cell(acc, pixelSeeds.eta()[pLS], pixelSeeds.phi()[pLS]);
        alpaka::atomicAdd(acc, &cellCount[cell], 1u, alpaka::hierarchy::Threads{});
      }
    }
  };

  struct FillPixelQuintupletSeedGrid {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  ObjectRangesConst ranges,
                                  PixelSeedsConst pixelSeeds,
                                  PixelQuintupletsConst pixelQuintuplets,
                                  unsigned int* cellCursor,
                                  unsigned int* cellEntries) const {
      const unsigned int prefix = ranges.segmentModuleIndices()[modules.nLowerModules()];
      for (unsigned int i : cms::alpakatools::uniform_elements(acc, pixelQuintuplets.nPixelQuintuplets())) {
        const unsigned int pLS = pixelQuintuplets.pixelSegmentIndices()[i] - prefix;
        const int cell = t5DupGrid::cell(acc, pixelSeeds.eta()[pLS], pixelSeeds.phi()[pLS]);
        const unsigned int slot = alpaka::atomicAdd(acc, &cellCursor[cell], 1u, alpaka::hierarchy::Threads{});
        cellEntries[slot] = i;
      }
    }
  };

  // Only bit 0 of isDup (set before this kernel) is read and only bit 1 is written, so the result does not depend
  // on the order in which pairs are visited; each unordered pair is tested once, from its lower index.
  struct RemoveDupQuintupletsBeforeTC {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  Quintuplets quintuplets,
                                  QuintupletsOccupancyConst quintupletsOccupancy,
                                  ObjectRangesConst ranges,
                                  unsigned int const* cellStart,
                                  unsigned int const* cellEntries) const {
      for (unsigned int lowmodIdx : cms::alpakatools::uniform_elements_z(acc, ranges.nEligibleT5Modules())) {
        const uint16_t lowmod = ranges.indicesOfEligibleT5Modules()[lowmodIdx];
        const unsigned int nQuintuplets = quintupletsOccupancy.nQuintuplets()[lowmod];
        const unsigned int first = ranges.quintupletModuleIndices()[lowmod];
        for (unsigned int i : cms::alpakatools::uniform_elements_y(acc, nQuintuplets)) {
          const unsigned int ix = first + i;
          if (quintuplets.isDup()[ix] & 1)
            continue;

          const bool isPT5_ix = quintuplets.partOfPT5()[ix];
          const float eta1 = __H2F(quintuplets.eta()[ix]);
          const float phi1 = __H2F(quintuplets.phi()[ix]);
          const float dnnScore1 = quintuplets.dnnScore()[ix];
          const int cell = t5DupGrid::cell(acc, eta1, phi1);
          const int etaBin = cell / t5DupGrid::kNPhi;
          const int phiBin = cell % t5DupGrid::kNPhi;

          for (int e = etaBin - 1; e <= etaBin + 1; ++e) {
            if (e < 0 || e >= t5DupGrid::kNEta)
              continue;
            for (int dp = -1; dp <= 1; ++dp) {
              const int neighbourCell = e * t5DupGrid::kNPhi + (phiBin + dp + t5DupGrid::kNPhi) % t5DupGrid::kNPhi;
              const unsigned int cellFirst = cellStart[neighbourCell];
              for (unsigned int k :
                   cms::alpakatools::uniform_elements_x(acc, cellStart[neighbourCell + 1] - cellFirst)) {
                const unsigned int jx = cellEntries[cellFirst + k];
                if (jx <= ix)
                  continue;

                const bool isPT5_jx = quintuplets.partOfPT5()[jx];
                if (isPT5_ix && isPT5_jx)
                  continue;

                const float eta2 = __H2F(quintuplets.eta()[jx]);
                const float dEta = alpaka::math::abs(acc, eta1 - eta2);
                if (dEta > 0.1f)
                  continue;

                const float phi2 = __H2F(quintuplets.phi()[jx]);
                const float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);
                if (alpaka::math::abs(acc, dPhi) > 0.1f)
                  continue;

                const int nMatched = checkHitsT5(ix, jx, quintuplets);

                float d2 = 0.f;
                CMS_UNROLL_LOOP
                for (unsigned int k2 = 0; k2 < Params_T5::kEmbed; ++k2) {
                  float diff = quintuplets.t5Embed()[ix][k2] - quintuplets.t5Embed()[jx][k2];
                  d2 += diff * diff;
                }

                // 99th percentile of true-dup d2 distribution measured on 100 PU200 events.
                constexpr float d2Thresh = 0.25f;
                constexpr int minNHitsForDup_T5 = 5;
                // Duplicate regardless of the embedding at this many shared hits.
                constexpr int nHitsForHardDup_T5 = 10;
                if ((nMatched >= minNHitsForDup_T5 && d2 < d2Thresh) || nMatched >= nHitsForHardDup_T5) {
                  const float dnnScore2 = quintuplets.dnnScore()[jx];
                  const bool ixLoses = (dnnScore1 < dnnScore2) || (dnnScore1 == dnnScore2 && ix < jx);
                  if (ixLoses)
                    rmQuintupletFromMemory(quintuplets, ix, true);
                  else
                    rmQuintupletFromMemory(quintuplets, jx, true);
                }
              }
            }
          }
        }
      }
    }
  };

  // One module per z block; its inner T4s are spread over the y blocks (no isDup read: order-independent).
  struct RemoveDupQuadrupletsAfterBuild {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  Quadruplets quadruplets,
                                  QuadrupletsOccupancyConst quadrupletsOccupancy,
                                  ObjectRangesConst ranges) const {
      for (auto iter : cms::alpakatools::uniform_elements_z(acc, ranges.nEligibleT4Modules())) {
        const uint16_t lowmod = ranges.indicesOfEligibleT4Modules()[iter];
        unsigned int nQuadruplets_lowmod = quadrupletsOccupancy.nQuadruplets()[lowmod];
        int quadrupletModuleIndices_lowmod = ranges.quadrupletModuleIndices()[lowmod];

        for (unsigned int ix1 : cms::alpakatools::uniform_elements_y(acc, nQuadruplets_lowmod)) {
          unsigned int ix = quadrupletModuleIndices_lowmod + ix1;
          const float eta1 = __H2F(quadruplets.eta()[ix]);
          const float phi1 = __H2F(quadruplets.phi()[ix]);
          const float score1 = quadruplets.displacedScore()[ix];

          for (unsigned int jx1 : cms::alpakatools::uniform_elements_x(acc, ix1 + 1, nQuadruplets_lowmod)) {
            unsigned int jx = quadrupletModuleIndices_lowmod + jx1;

            const float eta2 = __H2F(quadruplets.eta()[jx]);
            const float phi2 = __H2F(quadruplets.phi()[jx]);
            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);

            if (dEta > 0.1f)
              continue;

            if (alpaka::math::abs(acc, dPhi) > 0.1f)
              continue;

            const float score2 = quadruplets.displacedScore()[jx];

            int nMatched = checkHitsT4(ix, jx, quadruplets);
            const int minNHitsForDup_T4 = 5;
            if (nMatched >= minNHitsForDup_T4) {
              if (score1 >= score2) {
                rmQuadrupletFromMemory(quadruplets, jx);
              } else {
                rmQuadrupletFromMemory(quadruplets, ix);
              }
            }
          }
        }
      }
    }
  };

  // Same ordered pairs as a loop over T4 module pairs m1 <= m2 (module order); the loser gets isDup |= 2 and only
  // isDup & 1 (after-build dups) is read, so the result does not depend on the visiting order.
  // Needs the compact (dense, module-ordered) T4 layout: one T4 per y element, its partners spread over x.
  struct RemoveDupQuadrupletsBeforeTC {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  Quadruplets quadruplets,
                                  ObjectRangesConst ranges,
                                  const unsigned int nQuadruplets) const {
      for (unsigned int ix : cms::alpakatools::uniform_elements_y(acc, nQuadruplets)) {
        if ((quadruplets.isDup()[ix] & 1))
          continue;

        const unsigned int firstPartner = ranges.quadrupletModuleIndices()[quadruplets.lowerModuleIndices()[ix][0]];
        const float eta1 = __H2F(quadruplets.eta()[ix]);
        const float phi1 = __H2F(quadruplets.phi()[ix]);
        const float score1 = quadruplets.displacedScore()[ix];

        for (unsigned int jx : cms::alpakatools::uniform_elements_x(acc, firstPartner, nQuadruplets)) {
          if (ix == jx)
            continue;

          if ((quadruplets.isDup()[jx] & 1))
            continue;

          const float eta2 = __H2F(quadruplets.eta()[jx]);
          const float phi2 = __H2F(quadruplets.phi()[jx]);
          float dEta = alpaka::math::abs(acc, eta1 - eta2);
          float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);

          if (dEta > 0.1f)
            continue;

          if (alpaka::math::abs(acc, dPhi) > 0.1f)
            continue;

          const float score2 = quadruplets.displacedScore()[jx];

          int nMatched = checkHitsT4(ix, jx, quadruplets);
          const int minNHitsForDup_T4 = 4;
          if (nMatched >= minNHitsForDup_T4) {
            if (score1 > score2) {
              rmQuadrupletFromMemory(quadruplets, jx, true);
            } else if (score1 < score2) {
              rmQuadrupletFromMemory(quadruplets, ix, true);
            } else {
              rmQuadrupletFromMemory(quadruplets, (ix < jx ? ix : jx), true);
            }
          }
        }
      }
    }
  };

  // pT3s listed by their T3 hits, bucketed by hit index. A pT3 duplicate needs >= 5 shared hits of which at most 4
  // are pLS hits, so the two pT3s share a T3 hit and each finds the other in one of its six buckets.
  namespace pT3HitBuckets {
    constexpr unsigned int kNBuckets = 8192;
    constexpr unsigned int kT3HitOffset = Params_pLS::kHits;  // hits 4-9 of a pT3 are its T3 hits
    constexpr unsigned int kT3Hits = Params_T3::kHits;
    ALPAKA_FN_ACC ALPAKA_FN_INLINE unsigned int bucket(unsigned int hit) { return hit & (kNBuckets - 1); }
  }  // namespace pT3HitBuckets

  struct CountPixelTripletHitBuckets {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, PixelTripletsConst pixelTriplets, unsigned int* bucketCount) const {
      for (unsigned int ix : cms::alpakatools::uniform_elements(acc, pixelTriplets.nPixelTriplets())) {
        for (unsigned int i = 0; i < pT3HitBuckets::kT3Hits; ++i) {
          const unsigned int hitBucket =
              pT3HitBuckets::bucket(pixelTriplets.hitIndices()[ix][pT3HitBuckets::kT3HitOffset + i]);
          alpaka::atomicAdd(acc, &bucketCount[hitBucket], 1u, alpaka::hierarchy::Threads{});
        }
      }
    }
  };

  struct FillPixelTripletHitBuckets {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  PixelTripletsConst pixelTriplets,
                                  unsigned int* bucketCursor,
                                  unsigned int* bucketEntries) const {
      for (unsigned int ix : cms::alpakatools::uniform_elements(acc, pixelTriplets.nPixelTriplets())) {
        for (unsigned int i = 0; i < pT3HitBuckets::kT3Hits; ++i) {
          const unsigned int hitBucket =
              pT3HitBuckets::bucket(pixelTriplets.hitIndices()[ix][pT3HitBuckets::kT3HitOffset + i]);
          const unsigned int slot = alpaka::atomicAdd(acc, &bucketCursor[hitBucket], 1u, alpaka::hierarchy::Threads{});
          bucketEntries[slot] = ix;
        }
      }
    }
  };

  // isDup is written but never read here, so ix is removed iff some other pT3 beats it, in any visiting order.
  struct RemoveDupPixelTripletsFromMap {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  PixelTriplets pixelTriplets,
                                  unsigned int const* bucketStart,
                                  unsigned int const* bucketEntries) const {
      for (unsigned int ix : cms::alpakatools::uniform_elements_y(acc, pixelTriplets.nPixelTriplets())) {
        const auto layer_ix = pixelTriplets.logicalLayers()[ix][2];
        const float score_ix = __H2F(pixelTriplets.score()[ix]);
        bool removed = false;
        for (unsigned int i = 0; i < pT3HitBuckets::kT3Hits && !removed; ++i) {
          const unsigned int hitBucket =
              pT3HitBuckets::bucket(pixelTriplets.hitIndices()[ix][pT3HitBuckets::kT3HitOffset + i]);
          const unsigned int first = bucketStart[hitBucket];
          for (unsigned int k : cms::alpakatools::uniform_elements_x(acc, bucketStart[hitBucket + 1] - first)) {
            const unsigned int jx = bucketEntries[first + k];
            if (ix == jx)
              continue;
            // ix loses to jx: its T3 starts on a later logical layer, else the higher score, else the lower index.
            const auto layer_jx = pixelTriplets.logicalLayers()[jx][2];
            const float score_jx = __H2F(pixelTriplets.score()[jx]);
            const bool ixLoses = layer_jx < layer_ix || (layer_ix == layer_jx && score_ix > score_jx) ||
                                 (layer_ix == layer_jx && score_ix == score_jx && ix < jx);
            if (!ixLoses)
              continue;

            int nMatched[2];
            checkHitspT3(ix, jx, pixelTriplets, nMatched);
            const int minNHitsForDup_pT3 = 5;
            if ((nMatched[0] + nMatched[1]) >= minNHitsForDup_pT3) {
              rmPixelTripletFromMemory(pixelTriplets, ix);
              removed = true;
              break;
            }
          }
        }
      }
    }
  };

  // Grid fill for the pT5 duplicate removal: counts per cell (cellItems == nullptr), else scatter through the cursor.
  struct FillPixelQuintupletGrid {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  PixelQuintupletsConst pixelQuintuplets,
                                  EtaPhiGrid grid,
                                  unsigned int* __restrict__ cellCount,
                                  unsigned int* __restrict__ cellItems) const {
      for (unsigned int i : cms::alpakatools::uniform_elements(acc, pixelQuintuplets.nPixelQuintuplets())) {
        const int cell = grid.cell(acc, __H2F(pixelQuintuplets.eta()[i]), __H2F(pixelQuintuplets.phi()[i]));
        const unsigned int slot = alpaka::atomicAdd(acc, &cellCount[cell], 1u, alpaka::hierarchy::Blocks{});
        if (cellItems != nullptr)
          cellItems[slot] = i;
      }
    }
  };

  // True iff at least minShared of the nValid1 non-empty hits1 appear in hits2; returns as soon as that is known.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool pT5SharesAtLeast(unsigned int const (&hits1)[Params_pT5::kHits],
                                                       int nValid1,
                                                       unsigned int const (&hits2)[Params_pT5::kHits],
                                                       int minShared) {
    int nMatched = 0;
    int missesLeft = nValid1 - minShared;
    if (missesLeft < 0)
      return false;
    for (int i = 0; i < Params_pT5::kHits; i++) {
      if (hits1[i] == lst::kTCEmptyHitIdx)
        continue;
      bool matched = false;
      for (int j = 0; j < Params_pT5::kHits; j++) {
        if (hits1[i] == hits2[j]) {
          matched = true;
          break;
        }
      }
      if (matched) {
        if (++nMatched >= minShared)
          return true;
      } else if (--missesLeft < 0) {
        return false;
      }
    }
    return false;
  }

  // isDup is never read, so the decision for ix does not depend on the visiting order: the jx come from the 3x3 grid
  // cells around ix, a superset of the pT5s inside the 0.2 window.
  struct RemoveDupPixelQuintupletsFromMap {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  MiniDoubletsConst mds,
                                  SegmentsConst segments,
                                  QuintupletsConst quintuplets,
                                  PixelQuintuplets pixelQuintuplets,
                                  EtaPhiGrid grid,
                                  unsigned int const* __restrict__ cellStart,
                                  unsigned int const* __restrict__ cellItems) const {
      unsigned int nPixelQuintuplets = pixelQuintuplets.nPixelQuintuplets();
      for (unsigned int ix : cms::alpakatools::uniform_elements_y(acc, nPixelQuintuplets)) {
        float eta1 = __H2F(pixelQuintuplets.eta()[ix]);
        float phi1 = __H2F(pixelQuintuplets.phi()[ix]);
        float score1 = __H2F(pixelQuintuplets.score()[ix]);
        unsigned int hits1[Params_pT5::kHits];
        getPixelQuintupletHitIndices(mds, segments, quintuplets, pixelQuintuplets, ix, hits1);
        int nValid1 = 0;
        for (int i = 0; i < Params_pT5::kHits; i++)
          nValid1 += (hits1[i] != lst::kTCEmptyHitIdx);

        const int etaBin = grid.etaBin(acc, eta1);
        const int phiBin = grid.phiBin(acc, phi1);
        const int etaBinEnd = alpaka::math::min(acc, etaBin + 1, grid.nEta - 1);
        bool removed = false;
        for (int eBin = alpaka::math::max(acc, etaBin - 1, 0); eBin <= etaBinEnd && !removed; ++eBin) {
          for (int dPhiBin = -1; dPhiBin <= 1 && !removed; ++dPhiBin) {
            const int cell = grid.cell(eBin, grid.wrapPhiBin(phiBin + dPhiBin));
            for (unsigned int k : cms::alpakatools::uniform_elements_x(acc, cellStart[cell], cellStart[cell + 1])) {
              const unsigned int jx = cellItems[k];
              if (ix == jx)
                continue;

              float eta2 = __H2F(pixelQuintuplets.eta()[jx]);
              if (alpaka::math::abs(acc, eta1 - eta2) > 0.2f)
                continue;

              float phi2 = __H2F(pixelQuintuplets.phi()[jx]);
              if (alpaka::math::abs(acc, cms::alpakatools::deltaPhi(acc, phi1, phi2)) > 0.2f)
                continue;

              float score2 = __H2F(pixelQuintuplets.score()[jx]);
              if (!(score1 > score2 or ((score1 == score2) and (ix > jx))))
                continue;

              unsigned int hits2[Params_pT5::kHits];
              getPixelQuintupletHitIndices(mds, segments, quintuplets, pixelQuintuplets, jx, hits2);
              const int minNHitsForDup_pT5 = 7;
              if (pT5SharesAtLeast(hits1, nValid1, hits2, minNHitsForDup_pT5)) {
                rmPixelQuintupletFromMemory(pixelQuintuplets, ix);
                removed = true;
                break;
              }
            }
          }
        }
      }
    }
  };

  struct CheckHitspLS {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ModulesConst modules,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  PixelSeedsConst pixelSeeds,
                                  PixelSegments pixelSegments,
                                  bool secondpass) const {
      int pixelModuleIndex = modules.nLowerModules();
      unsigned int nPixelSegments = segmentsOccupancy.nSegments()[pixelModuleIndex];

      if (nPixelSegments > n_max_pixel_segments_per_module)
        nPixelSegments = n_max_pixel_segments_per_module;

      for (unsigned int ix : cms::alpakatools::uniform_elements_y(acc, nPixelSegments)) {
        if (secondpass && (!pixelSeeds.isQuad()[ix] || (pixelSegments.isDup()[ix] & 1)))
          continue;

        auto const& phits1 = pixelSegments.pLSHitsIdxs()[ix];
        float eta_pix1 = pixelSeeds.eta()[ix];
        float phi_pix1 = pixelSeeds.phi()[ix];

        for (unsigned int jx : cms::alpakatools::uniform_elements_x(acc, ix + 1, nPixelSegments)) {
          float eta_pix2 = pixelSeeds.eta()[jx];
          float phi_pix2 = pixelSeeds.phi()[jx];

          if (alpaka::math::abs(acc, eta_pix2 - eta_pix1) > 0.1f)
            continue;

          if (secondpass && (!pixelSeeds.isQuad()[jx] || (pixelSegments.isDup()[jx] & 1)))
            continue;

          int8_t quad_diff = pixelSeeds.isQuad()[ix] - pixelSeeds.isQuad()[jx];
          float score_diff = pixelSegments.score()[ix] - pixelSegments.score()[jx];
          // Always keep quads over trips. If they are the same, we want the object with better score
          int idxToRemove;
          if (quad_diff > 0)
            idxToRemove = jx;
          else if (quad_diff < 0)
            idxToRemove = ix;
          else if (score_diff < 0)
            idxToRemove = jx;
          else if (score_diff > 0)
            idxToRemove = ix;
          else
            idxToRemove = ix;

          auto const& phits2 = pixelSegments.pLSHitsIdxs()[jx];

          int npMatched = 0;
          for (int i = 0; i < Params_pLS::kHits; i++) {
            bool pmatched = false;
            for (int j = 0; j < Params_pLS::kHits; j++) {
              if (phits1[i] == phits2[j]) {
                pmatched = true;
                break;
              }
            }
            if (pmatched) {
              npMatched++;
              // Only one hit is enough
              if (secondpass)
                break;
            }
          }
          const int minNHitsForDup_pLS = 3;
          if (npMatched >= minNHitsForDup_pLS) {
            rmPixelSegmentFromMemory(pixelSegments, idxToRemove, secondpass);
          }
          if (secondpass) {
            float dEta = alpaka::math::abs(acc, eta_pix1 - eta_pix2);
            float dPhi = cms::alpakatools::deltaPhi(acc, phi_pix1, phi_pix2);

            float dR2 = dEta * dEta + dPhi * dPhi;
            if ((npMatched >= 1) || (dR2 < 1e-5f)) {
              rmPixelSegmentFromMemory(pixelSegments, idxToRemove, secondpass);
            }
          }
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif

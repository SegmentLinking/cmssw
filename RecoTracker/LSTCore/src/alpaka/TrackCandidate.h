#ifndef RecoTracker_LSTCore_src_alpaka_TrackCandidate_h
#define RecoTracker_LSTCore_src_alpaka_TrackCandidate_h

#include <bit>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"

#include "LSTEvent.h"
#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/HitsSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelQuintupletsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelSegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/PixelTripletsSoA.h"
#include "RecoTracker/LSTCore/interface/QuintupletsSoA.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"
#include "RecoTracker/LSTCore/interface/TrackCandidatesSoA.h"
#include "RecoTracker/LSTCore/interface/TripletsSoA.h"
#include "RecoTracker/LSTCore/interface/QuadrupletsSoA.h"

#include "EtaPhiGrid.h"
#include "NeuralNetwork.h"
#include "PixelQuintupletAccessors.h"
#include "Kernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addpLSTrackCandidateToMemory(TrackCandidatesBase& candsBase,
                                                                   TrackCandidatesExtended& candsExtended,
                                                                   unsigned int trackletIndex,
                                                                   unsigned int trackCandidateIndex,
                                                                   const Params_pLS::ArrayUxHits& hitIndices,
                                                                   int pixelSeedIndex) {
    candsBase.trackCandidateType()[trackCandidateIndex] = LSTObjType::pLS;
    candsExtended.directObjectIndices()[trackCandidateIndex] = trackletIndex;
    candsBase.pixelSeedIndex()[trackCandidateIndex] = pixelSeedIndex;

    candsExtended.objectIndices()[trackCandidateIndex][0] = trackletIndex;
    candsExtended.objectIndices()[trackCandidateIndex][1] = trackletIndex;

    // Initialize all slots to empty
    auto& tcHits = candsBase.hitIndices()[trackCandidateIndex];
    CMS_UNROLL_LOOP for (int layerSlot = 0; layerSlot < Params_TC::kLayers; ++layerSlot) {
      candsExtended.logicalLayers()[trackCandidateIndex][layerSlot] = 0;
      candsExtended.lowerModuleIndices()[trackCandidateIndex][layerSlot] = lst::kTCEmptyLowerModule;
      tcHits[layerSlot][0] = lst::kTCEmptyHitIdx;
      tcHits[layerSlot][1] = lst::kTCEmptyHitIdx;
    }

    // Order explanation in https://github.com/SegmentLinking/TrackLooper/issues/267
    tcHits[0][0] = hitIndices[0];
    tcHits[0][1] = hitIndices[2];
    tcHits[1][0] = hitIndices[1];
    tcHits[1][1] = hitIndices[3];
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTrackCandidateLayerHits(
      TrackCandidatesBase& candsBase,
      TrackCandidatesExtended& candsExtended,
      unsigned int trackCandidateIndex,
      int layerSlot,         // 0..12 (0/1 = pixel, 2..12 = OT logical layers 1..11)
      uint8_t logicalLayer,  // 0 for pixel, 1..11 for OT
      uint16_t lowerModule,
      unsigned int hitIndex0,
      unsigned int hitIndex1) {
    auto& tcHits = candsBase.hitIndices()[trackCandidateIndex];
    candsExtended.logicalLayers()[trackCandidateIndex][layerSlot] = logicalLayer;
    candsExtended.lowerModuleIndices()[trackCandidateIndex][layerSlot] = lowerModule;
    tcHits[layerSlot][0] = hitIndex0;
    tcHits[layerSlot][1] = hitIndex1;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTrackCandidateToMemory(TrackCandidatesBase& candsBase,
                                                                TrackCandidatesExtended& candsExtended,
                                                                LSTObjType trackCandidateType,
                                                                unsigned int innerTrackletIndex,
                                                                unsigned int outerTrackletIndex,
                                                                const uint8_t* logicalLayerIndices,
                                                                const uint16_t* lowerModuleIndices,
                                                                const unsigned int* hitIndices,
                                                                int pixelSeedIndex,
                                                                float centerX,
                                                                float centerY,
                                                                float radius,
                                                                unsigned int trackCandidateIndex,
                                                                unsigned int directObjectIndex) {
    candsBase.trackCandidateType()[trackCandidateIndex] = trackCandidateType;
    candsExtended.directObjectIndices()[trackCandidateIndex] = directObjectIndex;
    candsBase.pixelSeedIndex()[trackCandidateIndex] = pixelSeedIndex;

    candsExtended.objectIndices()[trackCandidateIndex][0] = innerTrackletIndex;
    candsExtended.objectIndices()[trackCandidateIndex][1] = outerTrackletIndex;

    // Initialize all slots to empty
    auto& tcHits = candsBase.hitIndices()[trackCandidateIndex];
    CMS_UNROLL_LOOP for (int layerSlot = 0; layerSlot < Params_TC::kLayers; ++layerSlot) {
      candsExtended.logicalLayers()[trackCandidateIndex][layerSlot] = 0;  // 0 is "pixel" when filled
      candsExtended.lowerModuleIndices()[trackCandidateIndex][layerSlot] = lst::kTCEmptyLowerModule;
      tcHits[layerSlot][0] = lst::kTCEmptyHitIdx;
      tcHits[layerSlot][1] = lst::kTCEmptyHitIdx;
    }

    // Configuration based on Type
    int nLayersToProcess = 0;
    int nPixelLayers = 0;

    if (trackCandidateType == LSTObjType::T5) {
      nLayersToProcess = Params_T5::kLayers;
    } else if (trackCandidateType == LSTObjType::pT5) {
      nLayersToProcess = Params_pT5::kLayers;
      nPixelLayers = Params_TC::kPixelLayerSlots;
    } else if (trackCandidateType == LSTObjType::T4) {
      nLayersToProcess = Params_T4::kLayers;
    } else if (trackCandidateType == LSTObjType::pT3) {
      nLayersToProcess = Params_pT3::kLayers;
      nPixelLayers = Params_TC::kPixelLayerSlots;
    }

    CMS_UNROLL_LOOP
    for (int i = 0; i < Params_TC::kLayers; ++i) {
      if (i >= nLayersToProcess)
        break;

      uint8_t logicalLayer = logicalLayerIndices[i];
      uint16_t lowerModule = lowerModuleIndices[i];
      unsigned int hit0 = hitIndices[2 * i];
      unsigned int hit1 = hitIndices[2 * i + 1];

      // Skip empty slots (sentinel values from extended T5/pT5 arrays)
      if (hit0 == lst::kTCEmptyHitIdx)
        continue;

      int layerSlot;

      if (i < nPixelLayers) {
        // Pixel layers occupy slots 0 and 1 strictly
        layerSlot = i;
        logicalLayer = 0;
      } else {
        // OT layers are mapped: (LogicalLayer - 1) + kPixelLayerSlots
        layerSlot = (logicalLayer - 1) + Params_TC::kPixelLayerSlots;
      }

      addTrackCandidateLayerHits(
          candsBase, candsExtended, trackCandidateIndex, layerSlot, logicalLayer, lowerModule, hit0, hit1);
    }

#ifdef CUT_VALUE_DEBUG
    candsExtended.centerX()[trackCandidateIndex] = __F2H(centerX);
    candsExtended.centerY()[trackCandidateIndex] = __F2H(centerY);
    candsExtended.radius()[trackCandidateIndex] = __F2H(radius);
#endif
  }

  struct CrossCleanpT3 {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ModulesConst modules,
                                  ObjectRangesConst ranges,
                                  PixelTriplets pixelTriplets,
                                  PixelSeedsConst pixelSeeds,
                                  PixelQuintupletsConst pixelQuintuplets,
                                  unsigned int const* cellStart,
                                  unsigned int const* cellEntries) const {
      const unsigned int prefix = ranges.segmentModuleIndices()[modules.nLowerModules()];
      for (unsigned int pixelTripletIndex : cms::alpakatools::uniform_elements_y(acc, pixelTriplets.nPixelTriplets())) {
        if (pixelTriplets.isDup()[pixelTripletIndex])
          continue;

        // Cross cleaning step: pT5s whose pLS points the same way, from the 3x3 cells around the pT3's pLS.
        float eta1 = __H2F(pixelTriplets.eta_pix()[pixelTripletIndex]);
        float phi1 = __H2F(pixelTriplets.phi_pix()[pixelTripletIndex]);
        const int cell = t5DupGrid::cell(acc, eta1, phi1);
        const int etaBin = cell / t5DupGrid::kNPhi;
        const int phiBin = cell % t5DupGrid::kNPhi;
        for (int e = etaBin - 1; e <= etaBin + 1; ++e) {
          if (e < 0 || e >= t5DupGrid::kNEta)
            continue;
          for (int dp = -1; dp <= 1; ++dp) {
            const int neighbourCell = e * t5DupGrid::kNPhi + (phiBin + dp + t5DupGrid::kNPhi) % t5DupGrid::kNPhi;
            const unsigned int first = cellStart[neighbourCell];
            for (unsigned int k : cms::alpakatools::uniform_elements_x(acc, cellStart[neighbourCell + 1] - first)) {
              unsigned int pLS_jx = pixelQuintuplets.pixelSegmentIndices()[cellEntries[first + k]];
              float eta2 = pixelSeeds.eta()[pLS_jx - prefix];
              float phi2 = pixelSeeds.phi()[pLS_jx - prefix];
              float dEta = alpaka::math::abs(acc, (eta1 - eta2));
              float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);

              float dR2 = dEta * dEta + dPhi * dPhi;
              if (dR2 < 1e-5f)
                pixelTriplets.isDup()[pixelTripletIndex] = true;
            }
          }
        }
      }
    }
  };

  // Grid fill for CrossCleanT5 over the promoted (!isDup) pixel objects: pT5 j as j, pT3 j as nPT5 + j. Counts per
  // cell (cellItems == nullptr), else scatter through the cursor.
  struct FillPixelObjectGrid {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  PixelQuintupletsConst pixelQuintuplets,
                                  PixelTripletsConst pixelTriplets,
                                  EtaPhiGrid grid,
                                  unsigned int* __restrict__ cellCount,
                                  unsigned int* __restrict__ cellItems) const {
      const unsigned int nPT5 = pixelQuintuplets.nPixelQuintuplets();
      for (unsigned int jx : cms::alpakatools::uniform_elements(acc, nPT5 + pixelTriplets.nPixelTriplets())) {
        const bool isPT5 = (jx < nPT5);
        const unsigned int ptidx = isPT5 ? jx : (jx - nPT5);
        if (isPT5 ? pixelQuintuplets.isDup()[ptidx] : pixelTriplets.isDup()[ptidx])
          continue;
        const float eta = __H2F(isPT5 ? pixelQuintuplets.eta()[ptidx] : pixelTriplets.eta()[ptidx]);
        const float phi = __H2F(isPT5 ? pixelQuintuplets.phi()[ptidx] : pixelTriplets.phi()[ptidx]);
        const int cell = grid.cell(acc, eta, phi);
        const unsigned int slot = alpaka::atomicAdd(acc, &cellCount[cell], 1u, alpaka::hierarchy::Blocks{});
        if (cellItems != nullptr)
          cellItems[slot] = jx;
      }
    }
  };

  // Only the T5's own isDup is written, so the decision does not depend on the visiting order: the pixel objects come
  // from the 3x3 grid cells around the T5, a superset of the promoted ones inside the 0.15 window.
  struct CrossCleanT5 {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  Quintuplets quintuplets,
                                  QuintupletsOccupancyConst quintupletsOccupancy,
                                  PixelQuintupletsConst pixelQuintuplets,
                                  PixelTripletsConst pixelTriplets,
                                  ObjectRangesConst ranges,
                                  EtaPhiGrid grid,
                                  unsigned int const* __restrict__ cellStart,
                                  unsigned int const* __restrict__ cellItems) const {
      for (int lowmod : cms::alpakatools::uniform_elements_z(acc, modules.nLowerModules())) {
        if (ranges.quintupletModuleIndices()[lowmod] == -1)
          continue;

        unsigned int nQuints = quintupletsOccupancy.nQuintuplets()[lowmod];
        for (unsigned int iOff : cms::alpakatools::uniform_elements_y(acc, nQuints)) {
          unsigned int iT5 = ranges.quintupletModuleIndices()[lowmod] + iOff;

          // skip already-dup or already in pT5
          if (quintuplets.isDup()[iT5] || quintuplets.partOfPT5()[iT5])
            continue;

          const unsigned int nPT5 = pixelQuintuplets.nPixelQuintuplets();

          float eta1 = __H2F(quintuplets.eta()[iT5]);
          float phi1 = __H2F(quintuplets.phi()[iT5]);

          // Pre-load T5 hits outside the jx loop.
          unsigned int iT5Hits[Params_T5::kHits];
          CMS_UNROLL_LOOP for (int i = 0; i < Params_T5::kHits; ++i) { iT5Hits[i] = quintuplets.hitIndices()[iT5][i]; }
          // A pixel object deletes a quintuplet only on shared outer-tracker hits.
          constexpr int otThresh = 4;

          // Cross-clean against both pT5s and pT3s (only promoted ones are in the grid)
          const int etaBin = grid.etaBin(acc, eta1);
          const int phiBin = grid.phiBin(acc, phi1);
          const int etaBinEnd = alpaka::math::min(acc, etaBin + 1, grid.nEta - 1);
          bool removed = false;
          for (int eBin = alpaka::math::max(acc, etaBin - 1, 0); eBin <= etaBinEnd && !removed; ++eBin) {
            for (int dPhiBin = -1; dPhiBin <= 1 && !removed; ++dPhiBin) {
              const int cell = grid.cell(eBin, grid.wrapPhiBin(phiBin + dPhiBin));
              for (unsigned int k : cms::alpakatools::uniform_elements_x(acc, cellStart[cell], cellStart[cell + 1])) {
                const unsigned int jx = cellItems[k];
                const bool isPT5 = (jx < nPT5);
                const unsigned int ptidx = isPT5 ? jx : (jx - nPT5);
                const float eta2 = __H2F(isPT5 ? pixelQuintuplets.eta()[ptidx] : pixelTriplets.eta()[ptidx]);
                const float phi2 = __H2F(isPT5 ? pixelQuintuplets.phi()[ptidx] : pixelTriplets.phi()[ptidx]);
                if (alpaka::math::abs(acc, eta1 - eta2) >= 0.15f ||
                    alpaka::math::abs(acc, cms::alpakatools::deltaPhi(acc, phi1, phi2)) >= 0.15f)
                  continue;

                // Shared outer-tracker hits: the pixel object's hits after its pLS slots (for a pT5, its T5's hits).
                unsigned int const* ptOTHits =
                    isPT5 ? quintuplets.hitIndices()[pixelQuintuplets.quintupletIndices()[ptidx]].data()
                          : pixelTriplets.hitIndices()[ptidx].data() + Params_pLS::kHits;
                const int nPtOTHits = isPT5 ? Params_T5::kHits : Params_pT3::kHits - Params_pLS::kHits;
                int nOTMatched = 0;
                for (int i = 0; i < Params_T5::kHits; ++i) {
                  const unsigned int hitI = iT5Hits[i];
                  if (hitI == lst::kTCEmptyHitIdx)
                    continue;
                  for (int j = 0; j < nPtOTHits; ++j) {
                    if (ptOTHits[j] == hitI) {
                      nOTMatched++;
                      break;
                    }
                  }
                }
                if (nOTMatched >= otThresh) {
                  quintuplets.isDup()[iT5] |= 4;
                  removed = true;
                  break;
                }
              }
            }
          }
        }
      }
    }
  };

  // CrossCleanpLS compares a pLS only with the TCs in its 3x3 neighbourhood of an (eta, phi) grid whose cells are
  // wider than its largest window (dR2 < 0.02 for T5s: |d eta|, |d phi| < 0.1415), and replaces its pixel-hit overlap
  // test with pT3/pT5 seeds by one bit per pixel-hit key, set for the hits of every pLS that seeds a pT3/pT5 TC.
  constexpr float kCrossCleanGridWindow = 0.15f;
  constexpr float kCrossCleanGridEtaMax = 3.f;

  // Dense key of a packed pLS hit index (bit 31 = OT hit): IT hits in [0, nKeysIT), OT hits after them.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE unsigned int pixelHitKey(unsigned int packedHit, unsigned int nKeysIT) {
    constexpr unsigned int kOTBit = 1u << 31;
    return (packedHit & kOTBit) ? nKeysIT + (packedHit & ~kOTBit) : packedHit;
  }

  // keyMax[0] = largest IT hit index, keyMax[1] = largest OT hit index, over the hits of all pLSs.
  struct PixelHitKeyMax {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  uint16_t nLowerModules,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  PixelSegmentsConst pixelSegments,
                                  unsigned int* keyMax) const {
      constexpr unsigned int kOTBit = 1u << 31;
      unsigned int maxIT = 0, maxOT = 0;
      unsigned int nPixels = segmentsOccupancy.nSegments()[nLowerModules];
      for (unsigned int i : cms::alpakatools::uniform_elements(acc, nPixels)) {
        for (int k = 0; k < Params_pLS::kHits; ++k) {
          const unsigned int hitIndex = pixelSegments.pLSHitsIdxs()[i][k];
          if (hitIndex & kOTBit)
            maxOT = (hitIndex & ~kOTBit) > maxOT ? (hitIndex & ~kOTBit) : maxOT;
          else
            maxIT = hitIndex > maxIT ? hitIndex : maxIT;
        }
      }
      alpaka::atomicMax(acc, &keyMax[0], maxIT, alpaka::hierarchy::Blocks{});
      alpaka::atomicMax(acc, &keyMax[1], maxOT, alpaka::hierarchy::Blocks{});
    }
  };

  // The (eta, phi) a pLS is compared with for a TC of this type; false for T4s, which CrossCleanpLS skips.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool crossCleanEtaPhi(LSTObjType type,
                                                       unsigned int innerTrackletIdx,
                                                       unsigned int prefix,
                                                       PixelTripletsConst pixelTriplets,
                                                       PixelSeedsConst pixelSeeds,
                                                       QuintupletsConst quintuplets,
                                                       float& eta,
                                                       float& phi) {
    if (type == LSTObjType::T5) {
      eta = __H2F(quintuplets.eta()[innerTrackletIdx]);
      phi = __H2F(quintuplets.phi()[innerTrackletIdx]);
    } else if (type == LSTObjType::pT3) {
      eta = __H2F(pixelTriplets.eta_pix()[innerTrackletIdx]);
      phi = __H2F(pixelTriplets.phi_pix()[innerTrackletIdx]);
    } else if (type == LSTObjType::pT5) {
      eta = pixelSeeds.eta()[innerTrackletIdx - prefix];
      phi = pixelSeeds.phi()[innerTrackletIdx - prefix];
    } else {
      return false;
    }
    return true;
  }

  // Per TC: count it in its grid cell and, for a pT3/pT5, mark the hit keys of its pLS.
  struct CountCrossCleanGrid {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  ObjectRangesConst ranges,
                                  PixelTripletsConst pixelTriplets,
                                  TrackCandidatesBaseConst candsBase,
                                  TrackCandidatesExtendedConst candsExtended,
                                  PixelSeedsConst pixelSeeds,
                                  PixelSegmentsConst pixelSegments,
                                  QuintupletsConst quintuplets,
                                  EtaPhiGrid grid,
                                  unsigned int nKeysIT,
                                  uint32_t* hitKeyBits,
                                  unsigned int* cellCount) const {
      unsigned int prefix = ranges.segmentModuleIndices()[modules.nLowerModules()];
      for (unsigned int tc : cms::alpakatools::uniform_elements(acc, candsBase.nTrackCandidates())) {
        LSTObjType type = candsBase.trackCandidateType()[tc];
        unsigned int innerTrackletIdx = candsExtended.objectIndices()[tc][0];
        float eta, phi;
        if (!crossCleanEtaPhi(type, innerTrackletIdx, prefix, pixelTriplets, pixelSeeds, quintuplets, eta, phi))
          continue;
        alpaka::atomicAdd(acc, &cellCount[grid.cell(acc, eta, phi)], 1u, alpaka::hierarchy::Blocks{});
        if (type == LSTObjType::T5)
          continue;
        unsigned int pLSIndex =
            type == LSTObjType::pT3 ? pixelTriplets.pixelSegmentIndices()[innerTrackletIdx] : innerTrackletIdx;
        for (int k = 0; k < Params_pLS::kHits; ++k) {
          const unsigned int key = pixelHitKey(pixelSegments.pLSHitsIdxs()[pLSIndex - prefix][k], nKeysIT);
          alpaka::atomicOr(acc, &hitKeyBits[key >> 5], 1u << (key & 31), alpaka::hierarchy::Blocks{});
        }
      }
    }
  };

  // Scatter the TCs into their cells (cellCursor = cellStart after EtaPhiGridPrefix).
  struct FillCrossCleanGrid {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  ObjectRangesConst ranges,
                                  PixelTripletsConst pixelTriplets,
                                  TrackCandidatesBaseConst candsBase,
                                  TrackCandidatesExtendedConst candsExtended,
                                  PixelSeedsConst pixelSeeds,
                                  QuintupletsConst quintuplets,
                                  EtaPhiGrid grid,
                                  unsigned int* cellCursor,
                                  unsigned int* cellTCs) const {
      unsigned int prefix = ranges.segmentModuleIndices()[modules.nLowerModules()];
      for (unsigned int tc : cms::alpakatools::uniform_elements(acc, candsBase.nTrackCandidates())) {
        float eta, phi;
        if (!crossCleanEtaPhi(candsBase.trackCandidateType()[tc],
                              candsExtended.objectIndices()[tc][0],
                              prefix,
                              pixelTriplets,
                              pixelSeeds,
                              quintuplets,
                              eta,
                              phi))
          continue;
        const unsigned int slot =
            alpaka::atomicAdd(acc, &cellCursor[grid.cell(acc, eta, phi)], 1u, alpaka::hierarchy::Blocks{});
        cellTCs[slot] = tc;
      }
    }
  };

  struct CrossCleanpLS {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ModulesConst modules,
                                  ObjectRangesConst ranges,
                                  PixelTripletsConst pixelTriplets,
                                  TrackCandidatesBaseConst candsBase,
                                  TrackCandidatesExtendedConst candsExtended,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  PixelSeedsConst pixelSeeds,
                                  PixelSegments pixelSegments,
                                  QuintupletsConst quintuplets,
                                  EtaPhiGrid grid,
                                  unsigned int nKeysIT,
                                  uint32_t const* hitKeyBits,
                                  unsigned int const* cellStart,
                                  unsigned int const* cellTCs) const {
      int pixelModuleIndex = modules.nLowerModules();
      unsigned int nPixels = segmentsOccupancy.nSegments()[pixelModuleIndex];
      unsigned int prefix = ranges.segmentModuleIndices()[pixelModuleIndex];
      for (unsigned int pixelArrayIndex : cms::alpakatools::uniform_elements(acc, nPixels)) {
        if (!pixelSeeds.isQuad()[pixelArrayIndex] || pixelSegments.isDup()[pixelArrayIndex])
          continue;

        // Shares a pixel hit with the pLS of a pT3/pT5 TC (or is one).
        bool isDup = false;
        for (int k = 0; k < Params_pLS::kHits; ++k) {
          const unsigned int key = pixelHitKey(pixelSegments.pLSHitsIdxs()[pixelArrayIndex][k], nKeysIT);
          if ((hitKeyBits[key >> 5] >> (key & 31)) & 1u)
            isDup = true;
        }
        if (isDup) {
          pixelSegments.isDup()[pixelArrayIndex] = true;
          continue;
        }

        float eta1 = pixelSeeds.eta()[pixelArrayIndex];
        float phi1 = pixelSeeds.phi()[pixelArrayIndex];

        // Store the pLS embedding outside the TC comparison loop.
        float plsEmbed[Params_pLS::kEmbed];
        CMS_UNROLL_LOOP for (unsigned k = 0; k < Params_pLS::kEmbed; ++k) {
          plsEmbed[k] = pixelSegments.plsEmbed()[pixelArrayIndex][k];
        }

        // Get pLS embedding eta bin and cut value for that bin.
        float absEta1 = alpaka::math::abs(acc, eta1);
        uint8_t bin_idx = (absEta1 > 2.5f) ? (dnn::kEtaBins - 1) : static_cast<uint8_t>(absEta1 / dnn::kEtaSize);
        const float threshold = dnn::plsembdnn::kWP[bin_idx];

        const int etaBin = grid.etaBin(acc, eta1);
        const int phiBin = grid.phiBin(acc, phi1);
        const int etaBinLast = etaBin + 1 < grid.nEta ? etaBin + 1 : grid.nEta - 1;
        for (int iEta = etaBin > 0 ? etaBin - 1 : 0; iEta <= etaBinLast && !isDup; ++iEta) {
          for (int dPhiBin = -1; dPhiBin <= 1 && !isDup; ++dPhiBin) {
            const int cell = grid.cell(iEta, grid.wrapPhiBin(phiBin + dPhiBin));
            for (unsigned int s = cellStart[cell]; s < cellStart[cell + 1] && !isDup; ++s) {
              unsigned int trackCandidateIndex = cellTCs[s];
              LSTObjType type = candsBase.trackCandidateType()[trackCandidateIndex];
              unsigned int innerTrackletIdx = candsExtended.objectIndices()[trackCandidateIndex][0];
              float eta2, phi2;
              crossCleanEtaPhi(type, innerTrackletIdx, prefix, pixelTriplets, pixelSeeds, quintuplets, eta2, phi2);
              float dEta = alpaka::math::abs(acc, eta1 - eta2);
              float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);
              float dR2 = dEta * dEta + dPhi * dPhi;
              if (type == LSTObjType::T5) {
                // Cut on pLS-T5 embed distance.
                if (dR2 < 0.02f) {
                  float d2 = 0.f;
                  CMS_UNROLL_LOOP for (unsigned k = 0; k < Params_pLS::kEmbed; ++k) {
                    const float diff = plsEmbed[k] - quintuplets.t5Embed()[innerTrackletIdx][k];
                    d2 += diff * diff;
                  }
                  // Compare squared embedding distance to the cut value for the eta bin.
                  isDup = d2 < threshold * threshold;
                }
              } else {
                isDup = dR2 < 0.000001f;
              }
            }
          }
        }
        if (isDup)
          pixelSegments.isDup()[pixelArrayIndex] = true;
      }
    }
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int nSharedHitsT4(unsigned int const* __restrict__ t4Hits,
                                                   unsigned int const* __restrict__ otherHits,
                                                   int nOtherHits) {
    // Every quadruplet hit slot is filled, so no empty-slot check.
    static_assert(Params_T4::kHits == 8);
    int nShared = 0;
    for (int i = 0; i < Params_T4::kHits; ++i) {
      for (int j = 0; j < nOtherHits; ++j) {
        if (otherHits[j] == t4Hits[i]) {
          nShared++;
          break;
        }
      }
    }
    return nShared;
  }

  // The hits CrossCleanT4 compares a T4 with: the T5 hits of a T5/pT5 TC, the T3 hits of a pT3 TC, none otherwise.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int crossCleanT4Hits(short type,
                                                      unsigned int outerTrackletIdx,
                                                      PixelTripletsConst pixelTriplets,
                                                      QuintupletsConst quintuplets,
                                                      unsigned int const*& hits) {
    if (type == LSTObjType::T5 || type == LSTObjType::pT5) {
      hits = quintuplets.hitIndices()[outerTrackletIdx].data();
      return Params_T5::kHits;
    }
    if (type == LSTObjType::pT3) {
      // Hits 4-9 of a pT3 are its T3's six hits.
      hits = pixelTriplets.hitIndices()[outerTrackletIdx].data() + Params_pLS::kHits;
      return Params_T3::kHits;
    }
    hits = nullptr;
    return 0;
  }

  // Track candidates listed by the hits CrossCleanT4 compares, bucketed by hit index. A T4 is removed only by a
  // candidate that shares at least two of its hits, so it finds every such candidate in the buckets of its own hits.
  namespace tcHitBuckets {
    constexpr unsigned int kNBuckets = 8192;
    ALPAKA_FN_ACC ALPAKA_FN_INLINE unsigned int bucket(unsigned int hit) { return hit & (kNBuckets - 1); }
  }  // namespace tcHitBuckets

  // Counts per bucket (bucketTCs == nullptr), else scatters (candidate, hit) through the cursor.
  struct FillTCHitBuckets {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  PixelTripletsConst pixelTriplets,
                                  QuintupletsConst quintuplets,
                                  TrackCandidatesBaseConst candsBase,
                                  TrackCandidatesExtendedConst candsExtended,
                                  unsigned int* __restrict__ bucketCount,
                                  unsigned int* __restrict__ bucketTCs,
                                  unsigned int* __restrict__ bucketHits) const {
      for (unsigned int tc : cms::alpakatools::uniform_elements(acc, candsBase.nTrackCandidates())) {
        unsigned int const* hits;
        const int nHits = crossCleanT4Hits(
            candsBase.trackCandidateType()[tc], candsExtended.objectIndices()[tc][1], pixelTriplets, quintuplets, hits);
        for (int iHit = 0; iHit < nHits; ++iHit) {
          const unsigned int hit = hits[iHit];
          if (hit == lst::kTCEmptyHitIdx)
            continue;
          const unsigned int slot =
              alpaka::atomicAdd(acc, &bucketCount[tcHitBuckets::bucket(hit)], 1u, alpaka::hierarchy::Blocks{});
          if (bucketTCs != nullptr) {
            bucketTCs[slot] = tc;
            bucketHits[slot] = hit;
          }
        }
      }
    }
  };

  // Per T4 (y, the dense T4 layout) and T4 hit (x): the candidates listed under that hit. isDup is only set, so the
  // visiting order does not matter.
  struct CrossCleanT4 {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  Quadruplets quadruplets,
                                  PixelTripletsConst pixelTriplets,
                                  QuintupletsConst quintuplets,
                                  TrackCandidatesBaseConst candsBase,
                                  TrackCandidatesExtendedConst candsExtended,
                                  unsigned int nQuadruplets,
                                  unsigned int const* __restrict__ bucketStart,
                                  unsigned int const* __restrict__ bucketTCs,
                                  unsigned int const* __restrict__ bucketHits) const {
      for (unsigned int iT4 : cms::alpakatools::uniform_elements_y(acc, nQuadruplets)) {
        // skip already-dup
        if (quadruplets.isDup()[iT4])
          continue;

        unsigned int const* t4Hits = quadruplets.hitIndices()[iT4].data();
        bool removed = false;
        for (unsigned int iHit : cms::alpakatools::uniform_elements_x(acc, Params_T4::kHits)) {
          const unsigned int hit = t4Hits[iHit];
          const unsigned int bucket = tcHitBuckets::bucket(hit);
          for (unsigned int slot = bucketStart[bucket]; slot < bucketStart[bucket + 1]; ++slot) {
            if (bucketHits[slot] != hit)
              continue;
            const unsigned int trackCandidateIndex = bucketTCs[slot];
            const short type = candsBase.trackCandidateType()[trackCandidateIndex];
            unsigned int const* otherHits;
            const int nOtherHits = crossCleanT4Hits(
                type, candsExtended.objectIndices()[trackCandidateIndex][1], pixelTriplets, quintuplets, otherHits);
            // Deleted when a promoted candidate owns three of its hits, or two for a pixel quintuplet or triplet.
            const int minShared = (type == LSTObjType::pT5 || type == LSTObjType::pT3) ? 2 : 3;
            if (nSharedHitsT4(t4Hits, otherHits, nOtherHits) >= minShared) {
              quadruplets.isDup()[iT4] = true;
              removed = true;
              break;
            }
          }
          if (removed)
            break;
        }
      }
    }
  };

  struct CountSurvivingTCs {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  uint16_t nLowerModules,
                                  PixelQuintupletsConst pixelQuintuplets,
                                  PixelTripletsConst pixelTriplets,
                                  QuintupletsConst quintuplets,
                                  QuintupletsOccupancyConst quintupletsOccupancy,
                                  QuadrupletsConst quadruplets,
                                  QuadrupletsOccupancyConst quadrupletsOccupancy,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  PixelSeedsConst pixelSeeds,
                                  PixelSegmentsConst pixelSegments,
                                  ObjectRangesConst ranges,
                                  unsigned int* nSurviving,
                                  bool tc_pls_triplets) const {
      // The overflow counters ride along in nSurviving[7..11] ([5..6] = PixelHitKeyMax), read back in one copy.
      if (cms::alpakatools::once_per_grid(acc)) {
        nSurviving[7] = ranges.nSegmentOverflows();
        nSurviving[8] = ranges.nTripletOverflows();
        nSurviving[9] = ranges.nQuintupletOverflows();
        nSurviving[10] = ranges.nT5byMDOverflows();
        nSurviving[11] = ranges.nQuintupletCapDrops();
      }
      // Count surviving pT5s
      unsigned int nPixelQuintuplets = pixelQuintuplets.nPixelQuintuplets();
      for (unsigned int i : cms::alpakatools::uniform_elements(acc, nPixelQuintuplets)) {
        if (!pixelQuintuplets.isDup()[i])
          alpaka::atomicAdd(acc, &nSurviving[0], 1u, alpaka::hierarchy::Threads{});
      }

      // Count surviving pT3s
      unsigned int nPixelTriplets = pixelTriplets.nPixelTriplets();
      for (unsigned int i : cms::alpakatools::uniform_elements(acc, nPixelTriplets)) {
        if (!pixelTriplets.isDup()[i])
          alpaka::atomicAdd(acc, &nSurviving[1], 1u, alpaka::hierarchy::Threads{});
      }

      // Count surviving T5s
      for (unsigned int idx : cms::alpakatools::uniform_elements(acc, (unsigned int)nLowerModules)) {
        if (ranges.quintupletModuleIndices()[idx] == -1)
          continue;
        unsigned int nQuints = quintupletsOccupancy.nQuintuplets()[idx];
        for (unsigned int jdx = 0; jdx < nQuints; ++jdx) {
          unsigned int quintupletIndex = ranges.quintupletModuleIndices()[idx] + jdx;
          if (!quintuplets.isDup()[quintupletIndex] && !quintuplets.partOfPT5()[quintupletIndex])
            alpaka::atomicAdd(acc, &nSurviving[2], 1u, alpaka::hierarchy::Threads{});
        }
      }

      // Count surviving T4s (upper bound - before CrossCleanT4)
      for (unsigned int idx : cms::alpakatools::uniform_elements(acc, (unsigned int)nLowerModules)) {
        if (ranges.quadrupletModuleIndices()[idx] == -1)
          continue;
        unsigned int nQuads = quadrupletsOccupancy.nQuadruplets()[idx];
        for (unsigned int jdx = 0; jdx < nQuads; ++jdx) {
          unsigned int quadrupletIndex = ranges.quadrupletModuleIndices()[idx] + jdx;
          if (!quadruplets.isDup()[quadrupletIndex])
            alpaka::atomicAdd(acc, &nSurviving[3], 1u, alpaka::hierarchy::Threads{});
        }
      }

      // Count surviving pLS (upper bound - before CrossCleanpLS)
      unsigned int nPixels = segmentsOccupancy.nSegments()[nLowerModules];
      for (unsigned int i : cms::alpakatools::uniform_elements(acc, nPixels)) {
        if ((tc_pls_triplets || pixelSeeds.isQuad()[i]) && !pixelSegments.isDup()[i])
          alpaka::atomicAdd(acc, &nSurviving[4], 1u, alpaka::hierarchy::Threads{});
      }
    }
  };

  struct AddpT3asTrackCandidates {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  uint16_t nLowerModules,
                                  PixelTripletsConst pixelTriplets,
                                  TrackCandidatesBase candsBase,
                                  TrackCandidatesExtended candsExtended,
                                  PixelSeedsConst pixelSeeds,
                                  ObjectRangesConst ranges,
                                  unsigned int nAllocated) const {
      unsigned int nPixelTriplets = pixelTriplets.nPixelTriplets();
      unsigned int pLS_offset = ranges.segmentModuleIndices()[nLowerModules];
      for (unsigned int pixelTripletIndex : cms::alpakatools::uniform_elements(acc, nPixelTriplets)) {
        if ((pixelTriplets.isDup()[pixelTripletIndex]))
          continue;

        unsigned int trackCandidateIdx =
            alpaka::atomicAdd(acc, &candsBase.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
        if (trackCandidateIdx >= nAllocated) {
#ifdef WARNINGS
          printf("Track Candidate excess alert! Type = pT3");
#endif
          alpaka::atomicSub(acc, &candsBase.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
          break;

        } else {
          alpaka::atomicAdd(acc, &candsExtended.nTrackCandidatespT3(), 1u, alpaka::hierarchy::Threads{});

          float radius = 0.5f * (__H2F(pixelTriplets.pixelRadius()[pixelTripletIndex]) +
                                 __H2F(pixelTriplets.tripletRadius()[pixelTripletIndex]));
          unsigned int pT3PixelIndex = pixelTriplets.pixelSegmentIndices()[pixelTripletIndex];
          addTrackCandidateToMemory(candsBase,
                                    candsExtended,
                                    LSTObjType::pT3,
                                    pixelTripletIndex,
                                    pixelTripletIndex,
                                    pixelTriplets.logicalLayers()[pixelTripletIndex].data(),
                                    pixelTriplets.lowerModuleIndices()[pixelTripletIndex].data(),
                                    pixelTriplets.hitIndices()[pixelTripletIndex].data(),
                                    pixelSeeds.seedIdx()[pT3PixelIndex - pLS_offset],
                                    __H2F(pixelTriplets.centerX()[pixelTripletIndex]),
                                    __H2F(pixelTriplets.centerY()[pixelTripletIndex]),
                                    radius,
                                    trackCandidateIdx,
                                    pixelTripletIndex);
        }
      }
    }
  };

  struct AddT5asTrackCandidate {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  uint16_t nLowerModules,
                                  QuintupletsConst quintuplets,
                                  QuintupletsOccupancyConst quintupletsOccupancy,
                                  TrackCandidatesBase candsBase,
                                  TrackCandidatesExtended candsExtended,
                                  ObjectRangesConst ranges,
                                  unsigned int nAllocated) const {
      for (int idx : cms::alpakatools::uniform_elements_y(acc, nLowerModules)) {
        if (ranges.quintupletModuleIndices()[idx] == -1)
          continue;

        unsigned int nQuints = quintupletsOccupancy.nQuintuplets()[idx];
        for (unsigned int jdx : cms::alpakatools::uniform_elements_x(acc, nQuints)) {
          unsigned int quintupletIndex = ranges.quintupletModuleIndices()[idx] + jdx;
          if (quintuplets.isDup()[quintupletIndex] or quintuplets.partOfPT5()[quintupletIndex])
            continue;

          unsigned int trackCandidateIdx =
              alpaka::atomicAdd(acc, &candsBase.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
          if (trackCandidateIdx >= nAllocated) {
#ifdef WARNINGS
            printf("Track Candidate excess alert! Type = T5");
#endif
            alpaka::atomicSub(acc, &candsBase.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
            break;
          } else {
            alpaka::atomicAdd(acc, &candsExtended.nTrackCandidatesT5(), 1u, alpaka::hierarchy::Threads{});
            addTrackCandidateToMemory(candsBase,
                                      candsExtended,
                                      LSTObjType::T5,
                                      quintupletIndex,
                                      quintupletIndex,
                                      quintuplets.logicalLayers()[quintupletIndex].data(),
                                      quintuplets.lowerModuleIndices()[quintupletIndex].data(),
                                      quintuplets.hitIndices()[quintupletIndex].data(),
                                      -1 /*no pixel seed index for T5s*/,
                                      quintuplets.regressionCenterX()[quintupletIndex],
                                      quintuplets.regressionCenterY()[quintupletIndex],
                                      quintuplets.regressionRadius()[quintupletIndex],
                                      trackCandidateIdx,
                                      quintupletIndex);
          }
        }
      }
    }
  };

  struct AddpLSasTrackCandidate {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  uint16_t nLowerModules,
                                  TrackCandidatesBase candsBase,
                                  TrackCandidatesExtended candsExtended,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  PixelSeedsConst pixelSeeds,
                                  PixelSegmentsConst pixelSegments,
                                  bool tc_pls_triplets,
                                  unsigned int nAllocated) const {
      unsigned int nPixels = segmentsOccupancy.nSegments()[nLowerModules];
      for (unsigned int pixelArrayIndex : cms::alpakatools::uniform_elements(acc, nPixels)) {
        if ((tc_pls_triplets ? 0 : !pixelSeeds.isQuad()[pixelArrayIndex]) || (pixelSegments.isDup()[pixelArrayIndex]))
          continue;

        unsigned int trackCandidateIdx =
            alpaka::atomicAdd(acc, &candsBase.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
        if (trackCandidateIdx >= nAllocated) {
#ifdef WARNINGS
          printf("Track Candidate excess alert! Type = pLS");
#endif
          alpaka::atomicSub(acc, &candsBase.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
          break;

        } else {
          alpaka::atomicAdd(acc, &candsExtended.nTrackCandidatespLS(), 1u, alpaka::hierarchy::Threads{});
          addpLSTrackCandidateToMemory(candsBase,
                                       candsExtended,
                                       pixelArrayIndex,
                                       trackCandidateIdx,
                                       pixelSegments.pLSHitsIdxs()[pixelArrayIndex],
                                       pixelSeeds.seedIdx()[pixelArrayIndex]);
        }
      }
    }
  };

  struct AddpT5asTrackCandidate {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  uint16_t nLowerModules,
                                  MiniDoubletsConst mds,
                                  SegmentsConst segments,
                                  QuintupletsConst quintuplets,
                                  PixelQuintupletsConst pixelQuintuplets,
                                  TrackCandidatesBase candsBase,
                                  TrackCandidatesExtended candsExtended,
                                  PixelSeedsConst pixelSeeds,
                                  ObjectRangesConst ranges,
                                  unsigned int nAllocated) const {
      int nPixelQuintuplets = pixelQuintuplets.nPixelQuintuplets();
      unsigned int pLS_offset = ranges.segmentModuleIndices()[nLowerModules];
      for (int pixelQuintupletIndex : cms::alpakatools::uniform_elements(acc, nPixelQuintuplets)) {
        if (pixelQuintuplets.isDup()[pixelQuintupletIndex])
          continue;

        unsigned int trackCandidateIdx =
            alpaka::atomicAdd(acc, &candsBase.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
        if (trackCandidateIdx >= nAllocated) {
#ifdef WARNINGS
          printf("Track Candidate excess alert! Type = pT5");
#endif
          alpaka::atomicSub(acc, &candsBase.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
          break;

        } else {
          alpaka::atomicAdd(acc, &candsExtended.nTrackCandidatespT5(), 1u, alpaka::hierarchy::Threads{});

          float radius = 0.5f * (__H2F(pixelQuintuplets.pixelRadius()[pixelQuintupletIndex]) +
                                 __H2F(pixelQuintuplets.quintupletRadius()[pixelQuintupletIndex]));
          unsigned int pT5PixelIndex = pixelQuintuplets.pixelSegmentIndices()[pixelQuintupletIndex];
          unsigned int pT5Hits[Params_pT5::kHits];
          getPixelQuintupletHitIndices(mds, segments, quintuplets, pixelQuintuplets, pixelQuintupletIndex, pT5Hits);
          addTrackCandidateToMemory(candsBase,
                                    candsExtended,
                                    LSTObjType::pT5,
                                    pT5PixelIndex,
                                    pixelQuintuplets.quintupletIndices()[pixelQuintupletIndex],
                                    pixelQuintuplets.logicalLayers()[pixelQuintupletIndex].data(),
                                    pixelQuintuplets.lowerModuleIndices()[pixelQuintupletIndex].data(),
                                    pT5Hits,
                                    pixelSeeds.seedIdx()[pT5PixelIndex - pLS_offset],
                                    __H2F(pixelQuintuplets.centerX()[pixelQuintupletIndex]),
                                    __H2F(pixelQuintuplets.centerY()[pixelQuintupletIndex]),
                                    radius,
                                    trackCandidateIdx,
                                    pixelQuintupletIndex);
        }
      }
    }
  };

  struct AddT4asTrackCandidate {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  uint16_t nLowerModules,
                                  Quadruplets quadruplets,
                                  QuadrupletsOccupancyConst quadrupletsOccupancy,
                                  TripletsConst triplets,
                                  TrackCandidatesBase candsBase,
                                  TrackCandidatesExtended candsExtended,
                                  ObjectRangesConst ranges,
                                  unsigned int nAllocated) const {
      for (int idx : cms::alpakatools::uniform_elements_y(acc, nLowerModules)) {
        if (ranges.quadrupletModuleIndices()[idx] == -1)
          continue;

        unsigned int nQuads = quadrupletsOccupancy.nQuadruplets()[idx];
        for (unsigned int jdx : cms::alpakatools::uniform_elements_x(acc, nQuads)) {
          unsigned int quadrupletIndex = ranges.quadrupletModuleIndices()[idx] + jdx;

          if (quadruplets.isDup()[quadrupletIndex])
            continue;

          unsigned int trackCandidateIdx =
              alpaka::atomicAdd(acc, &candsBase.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
          if (trackCandidateIdx >= nAllocated) {
#ifdef WARNINGS
            printf("Track Candidate excess alert! Type = T4");
#endif
            alpaka::atomicSub(acc, &candsBase.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
            break;
          } else {
            alpaka::atomicAdd(acc, &candsExtended.nTrackCandidatesT4(), 1u, alpaka::hierarchy::Threads{});
            addTrackCandidateToMemory(candsBase,
                                      candsExtended,
                                      LSTObjType::T4,
                                      quadrupletIndex,
                                      quadrupletIndex,
                                      quadruplets.logicalLayers()[quadrupletIndex].data(),
                                      quadruplets.lowerModuleIndices()[quadrupletIndex].data(),
                                      quadruplets.hitIndices()[quadrupletIndex].data(),
                                      -1 /*no pixel seed index for T4s*/,
                                      quadruplets.regressionCenterX()[quadrupletIndex],
                                      quadruplets.regressionCenterY()[quadrupletIndex],
                                      quadruplets.regressionRadius()[quadrupletIndex],
                                      trackCandidateIdx,
                                      quadrupletIndex);
          }
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(lst::TrackCandidatesBaseDeviceCollection, lst::TrackCandidatesBaseHostCollection);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(lst::TrackCandidatesExtendedDeviceCollection,
                                      lst::TrackCandidatesExtendedHostCollection);

#endif

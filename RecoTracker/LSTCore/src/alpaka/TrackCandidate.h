#ifndef RecoTracker_LSTCore_src_alpaka_TrackCandidate_h
#define RecoTracker_LSTCore_src_alpaka_TrackCandidate_h

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

#include "NeuralNetwork.h"

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

    // Order explanation in https://github.com/SegmentLinking/TrackLooper/issues/267
    auto& tcHits = candsBase.hitIndices()[trackCandidateIndex];
    tcHits[0][0] = hitIndices[0];
    tcHits[0][1] = hitIndices[2];
    tcHits[1][0] = hitIndices[1];
    tcHits[1][1] = hitIndices[3];
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void placeLayer(TrackCandidatesBase& candsBase,
                                                 TrackCandidatesExtended& candsExtended,
                                                 unsigned int trackCandidateIndex,
                                                 int layerSlot,  // 0..12 (0/1 = pixel, 2..12 = OT logical layers 1..11)
                                                 uint8_t logicalLayer,  // 0 for pixel, 1..11 for OT
                                                 uint16_t lowerModule,
                                                 unsigned int hitIndex0,
                                                 unsigned int hitIndex1) {
    if (candsExtended.lowerModuleIndices()[trackCandidateIndex][layerSlot] == ::lst::kTCEmptyLowerModule) {
      candsBase.nLayers()[trackCandidateIndex] += 1;
    }

    candsExtended.logicalLayers()[trackCandidateIndex][layerSlot] = logicalLayer;
    candsExtended.lowerModuleIndices()[trackCandidateIndex][layerSlot] = lowerModule;
    candsBase.hitIndices()[trackCandidateIndex][layerSlot][0] = hitIndex0;
    candsBase.hitIndices()[trackCandidateIndex][layerSlot][1] = hitIndex1;
    candsBase.nHits()[trackCandidateIndex] += 2;
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

    // Reset counters
    candsBase.nLayers()[trackCandidateIndex] = 0;
    candsBase.nHits()[trackCandidateIndex] = 0;

    // Mark all slots empty and clear hit indices
    CMS_UNROLL_LOOP for (int slot = 0; slot < Params_TC::kLayers; ++slot) {
      candsExtended.logicalLayers()[trackCandidateIndex][slot] = 0;  // 0 is "pixel" when filled
      candsExtended.lowerModuleIndices()[trackCandidateIndex][slot] = ::lst::kTCEmptyLowerModule;
      candsBase.hitIndices()[trackCandidateIndex][slot][0] = ::lst::kTCEmptyHitIdx;
      candsBase.hitIndices()[trackCandidateIndex][slot][1] = ::lst::kTCEmptyHitIdx;
    }

    // T5: 5 OT logical layers in [1..11], mapped to layer slots (logicalLayer - 1) + 2
    if (trackCandidateType == LSTObjType::T5) {
      CMS_UNROLL_LOOP for (int layerIndex = 0; layerIndex < Params_T5::kLayers; ++layerIndex) {
        const uint8_t logicalLayer = logicalLayerIndices[layerIndex];  // 1..11
        const int layerSlot = (logicalLayer - 1) + 2;                  // OT slots
        placeLayer(candsBase,
                   candsExtended,
                   trackCandidateIndex,
                   layerSlot,
                   logicalLayer,
                   lowerModuleIndices[layerIndex],
                   hitIndices[2 * layerIndex + 0],
                   hitIndices[2 * layerIndex + 1]);
      }
    } else if (trackCandidateType == LSTObjType::pT5) {
      // Pixel layers occupy slots 0 and 1 with logicalLayer = 0
      placeLayer(
          candsBase, candsExtended, trackCandidateIndex, 0, 0, lowerModuleIndices[0], hitIndices[0], hitIndices[1]);
      placeLayer(
          candsBase, candsExtended, trackCandidateIndex, 1, 0, lowerModuleIndices[1], hitIndices[2], hitIndices[3]);

      // pT5 then carries the T5's 5 OT layers starting at logicalLayerIndices[2]; hits start at index 4
      CMS_UNROLL_LOOP for (int layerIndex = 2; layerIndex < Params_pT5::kLayers; ++layerIndex) {
        const uint8_t logicalLayer = logicalLayerIndices[layerIndex];  // 1..11
        const int layerSlot = (logicalLayer - 1) + 2;                  // OT slots
        const int hitOffset = 4 + 2 * (layerIndex - 2);                // 10 OT hits after 4 pixel hits
        placeLayer(candsBase,
                   candsExtended,
                   trackCandidateIndex,
                   layerSlot,
                   logicalLayer,
                   lowerModuleIndices[layerIndex],
                   hitIndices[hitOffset + 0],
                   hitIndices[hitOffset + 1]);
      }
    } else {
      // Legacy contiguous writer for pT3
      const size_t layerCount = (trackCandidateType == LSTObjType::pT3) ? Params_pT3::kLayers : Params_pLS::kLayers;

      CMS_UNROLL_LOOP for (size_t layerIndex = 0; layerIndex < layerCount; ++layerIndex) {
        candsExtended.logicalLayers()[trackCandidateIndex][layerIndex] = logicalLayerIndices[layerIndex];
        candsExtended.lowerModuleIndices()[trackCandidateIndex][layerIndex] = lowerModuleIndices[layerIndex];
        candsBase.hitIndices()[trackCandidateIndex][layerIndex][0] = hitIndices[2 * layerIndex + 0];
        candsBase.hitIndices()[trackCandidateIndex][layerIndex][1] = hitIndices[2 * layerIndex + 1];
      }

      candsBase.nLayers()[trackCandidateIndex] = static_cast<uint8_t>(layerCount);
      candsBase.nHits()[trackCandidateIndex] = static_cast<uint8_t>(2 * layerCount);
    }

    candsExtended.centerX()[trackCandidateIndex] = __F2H(centerX);
    candsExtended.centerY()[trackCandidateIndex] = __F2H(centerY);
    candsExtended.radius()[trackCandidateIndex] = __F2H(radius);
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int checkPixelHits(
      unsigned int ix, unsigned int jx, MiniDoubletsConst mds, SegmentsConst segments, HitsBaseConst hitsBase) {
    int phits1[Params_pLS::kHits];
    int phits2[Params_pLS::kHits];

    phits1[0] = hitsBase.idxs()[mds.anchorHitIndices()[segments.mdIndices()[ix][0]]];
    phits1[1] = hitsBase.idxs()[mds.anchorHitIndices()[segments.mdIndices()[ix][1]]];
    phits1[2] = hitsBase.idxs()[mds.outerHitIndices()[segments.mdIndices()[ix][0]]];
    phits1[3] = hitsBase.idxs()[mds.outerHitIndices()[segments.mdIndices()[ix][1]]];

    phits2[0] = hitsBase.idxs()[mds.anchorHitIndices()[segments.mdIndices()[jx][0]]];
    phits2[1] = hitsBase.idxs()[mds.anchorHitIndices()[segments.mdIndices()[jx][1]]];
    phits2[2] = hitsBase.idxs()[mds.outerHitIndices()[segments.mdIndices()[jx][0]]];
    phits2[3] = hitsBase.idxs()[mds.outerHitIndices()[segments.mdIndices()[jx][1]]];

    int npMatched = 0;

    for (int i = 0; i < Params_pLS::kHits; i++) {
      bool pmatched = false;
      if (phits1[i] == -1)
        continue;

      for (int j = 0; j < Params_pLS::kHits; j++) {
        if (phits2[j] == -1)
          continue;

        if (phits1[i] == phits2[j]) {
          pmatched = true;
          break;
        }
      }
      if (pmatched)
        npMatched++;
    }
    return npMatched;
  }

  struct CrossCleanpT3 {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ModulesConst modules,
                                  ObjectRangesConst ranges,
                                  PixelTriplets pixelTriplets,
                                  PixelSeedsConst pixelSeeds,
                                  PixelQuintupletsConst pixelQuintuplets) const {
      unsigned int nPixelTriplets = pixelTriplets.nPixelTriplets();
      for (unsigned int pixelTripletIndex : cms::alpakatools::uniform_elements_y(acc, nPixelTriplets)) {
        if (pixelTriplets.isDup()[pixelTripletIndex])
          continue;

        // Cross cleaning step
        float eta1 = __H2F(pixelTriplets.eta_pix()[pixelTripletIndex]);
        float phi1 = __H2F(pixelTriplets.phi_pix()[pixelTripletIndex]);

        int pixelModuleIndex = modules.nLowerModules();
        unsigned int prefix = ranges.segmentModuleIndices()[pixelModuleIndex];

        unsigned int nPixelQuintuplets = pixelQuintuplets.nPixelQuintuplets();
        for (unsigned int pixelQuintupletIndex : cms::alpakatools::uniform_elements_x(acc, nPixelQuintuplets)) {
          unsigned int pLS_jx = pixelQuintuplets.pixelSegmentIndices()[pixelQuintupletIndex];
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
  };

  struct CrossCleanT5 {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  Quintuplets quintuplets,
                                  QuintupletsOccupancyConst quintupletsOccupancy,
                                  PixelQuintupletsConst pixelQuintuplets,
                                  PixelTripletsConst pixelTriplets,
                                  ObjectRangesConst ranges) const {
      for (int lowmod : cms::alpakatools::uniform_elements_z(acc, modules.nLowerModules())) {
        if (ranges.quintupletModuleIndices()[lowmod] == -1)
          continue;

        unsigned int nQuints = quintupletsOccupancy.nQuintuplets()[lowmod];
        for (unsigned int iOff : cms::alpakatools::uniform_elements_y(acc, nQuints)) {
          unsigned int iT5 = ranges.quintupletModuleIndices()[lowmod] + iOff;

          // skip already-dup or already in pT5
          if (quintuplets.isDup()[iT5] || quintuplets.partOfPT5()[iT5])
            continue;

          unsigned int loop_bound = pixelQuintuplets.nPixelQuintuplets() + pixelTriplets.nPixelTriplets();

          float eta1 = __H2F(quintuplets.eta()[iT5]);
          float phi1 = __H2F(quintuplets.phi()[iT5]);

          float iEmbedT5[Params_T5::kEmbed];
          CMS_UNROLL_LOOP for (unsigned k = 0; k < Params_T5::kEmbed; ++k) {
            iEmbedT5[k] = quintuplets.t5Embed()[iT5][k];
          }

          // Cross-clean against both pT5s and pT3s
          for (unsigned int jx : cms::alpakatools::uniform_elements_x(acc, loop_bound)) {
            float eta2, phi2;
            if (jx < pixelQuintuplets.nPixelQuintuplets()) {
              eta2 = __H2F(pixelQuintuplets.eta()[jx]);
              phi2 = __H2F(pixelQuintuplets.phi()[jx]);
            } else {
              unsigned int ptidx = jx - pixelQuintuplets.nPixelQuintuplets();
              eta2 = __H2F(pixelTriplets.eta()[ptidx]);
              phi2 = __H2F(pixelTriplets.phi()[ptidx]);
            }

            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);
            float dR2 = dEta * dEta + dPhi * dPhi;

            if (jx < pixelQuintuplets.nPixelQuintuplets()) {
              unsigned int jT5 = pixelQuintuplets.quintupletIndices()[jx];
              float d2 = 0.f;
              // Compute distance-squared between the two t5 embeddings.
              CMS_UNROLL_LOOP for (unsigned k = 0; k < Params_T5::kEmbed; ++k) {
                float df = iEmbedT5[k] - quintuplets.t5Embed()[jT5][k];
                d2 += df * df;
              }
              if ((dR2 < 0.02f && d2 < 0.1f) || (dR2 < 1e-3f && d2 < 1.0f)) {
                quintuplets.isDup()[iT5] = true;
              }
            } else if (dR2 < 1e-3f) {
              quintuplets.isDup()[iT5] = true;
            }

            if (quintuplets.isDup()[iT5])
              break;
          }
        }
      }
    }
  };

  struct CrossCleanpLS {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ModulesConst modules,
                                  ObjectRangesConst ranges,
                                  PixelTripletsConst pixelTriplets,
                                  TrackCandidatesBase candsBase,
                                  TrackCandidatesExtended candsExtended,
                                  SegmentsConst segments,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  PixelSeedsConst pixelSeeds,
                                  PixelSegments pixelSegments,
                                  MiniDoubletsConst mds,
                                  HitsBaseConst hitsBase,
                                  QuintupletsConst quintuplets) const {
      int pixelModuleIndex = modules.nLowerModules();
      unsigned int nPixels = segmentsOccupancy.nSegments()[pixelModuleIndex];
      for (unsigned int pixelArrayIndex : cms::alpakatools::uniform_elements_y(acc, nPixels)) {
        if (!pixelSeeds.isQuad()[pixelArrayIndex] || pixelSegments.isDup()[pixelArrayIndex])
          continue;

        float eta1 = pixelSeeds.eta()[pixelArrayIndex];
        float phi1 = pixelSeeds.phi()[pixelArrayIndex];
        unsigned int prefix = ranges.segmentModuleIndices()[pixelModuleIndex];

        // Store the pLS embedding outside the TC comparison loop.
        float plsEmbed[Params_pLS::kEmbed];
        CMS_UNROLL_LOOP for (unsigned k = 0; k < Params_pLS::kEmbed; ++k) {
          plsEmbed[k] = pixelSegments.plsEmbed()[pixelArrayIndex][k];
        }

        // Get pLS embedding eta bin and cut value for that bin.
        float absEta1 = alpaka::math::abs(acc, eta1);
        uint8_t bin_idx = (absEta1 > 2.5f) ? (dnn::kEtaBins - 1) : static_cast<uint8_t>(absEta1 / dnn::kEtaSize);
        const float threshold = dnn::plsembdnn::kWP[bin_idx];

        unsigned int nTrackCandidates = candsBase.nTrackCandidates();
        for (unsigned int trackCandidateIndex : cms::alpakatools::uniform_elements_x(acc, nTrackCandidates)) {
          LSTObjType type = candsBase.trackCandidateType()[trackCandidateIndex];
          unsigned int innerTrackletIdx = candsExtended.objectIndices()[trackCandidateIndex][0];
          if (type == LSTObjType::T5) {
            unsigned int quintupletIndex = innerTrackletIdx;  // T5 index
            float eta2 = __H2F(quintuplets.eta()[quintupletIndex]);
            float phi2 = __H2F(quintuplets.phi()[quintupletIndex]);
            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);
            float dR2 = dEta * dEta + dPhi * dPhi;
            // Cut on pLS-T5 embed distance.
            if (dR2 < 0.02f) {
              float d2 = 0.f;
              CMS_UNROLL_LOOP for (unsigned k = 0; k < Params_pLS::kEmbed; ++k) {
                const float diff = plsEmbed[k] - quintuplets.t5Embed()[quintupletIndex][k];
                d2 += diff * diff;
              }
              // Compare squared embedding distance to the cut value for the eta bin.
              if (d2 < threshold * threshold) {
                pixelSegments.isDup()[pixelArrayIndex] = true;
              }
            }
          }
          if (type == LSTObjType::pT3) {
            int pLSIndex = pixelTriplets.pixelSegmentIndices()[innerTrackletIdx];
            int npMatched = checkPixelHits(prefix + pixelArrayIndex, pLSIndex, mds, segments, hitsBase);
            if (npMatched > 0)
              pixelSegments.isDup()[pixelArrayIndex] = true;

            int pT3Index = innerTrackletIdx;
            float eta2 = __H2F(pixelTriplets.eta_pix()[pT3Index]);
            float phi2 = __H2F(pixelTriplets.phi_pix()[pT3Index]);
            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);

            float dR2 = dEta * dEta + dPhi * dPhi;
            if (dR2 < 0.000001f)
              pixelSegments.isDup()[pixelArrayIndex] = true;
          }
          if (type == LSTObjType::pT5) {
            unsigned int pLSIndex = innerTrackletIdx;
            int npMatched = checkPixelHits(prefix + pixelArrayIndex, pLSIndex, mds, segments, hitsBase);
            if (npMatched > 0) {
              pixelSegments.isDup()[pixelArrayIndex] = true;
            }

            float eta2 = pixelSeeds.eta()[pLSIndex - prefix];
            float phi2 = pixelSeeds.phi()[pLSIndex - prefix];
            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);

            float dR2 = dEta * dEta + dPhi * dPhi;
            if (dR2 < 0.000001f)
              pixelSegments.isDup()[pixelArrayIndex] = true;
          }
        }
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
                                  ObjectRangesConst ranges) const {
      // implementation is 1D with a single block
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      unsigned int nPixelTriplets = pixelTriplets.nPixelTriplets();
      unsigned int pLS_offset = ranges.segmentModuleIndices()[nLowerModules];
      for (unsigned int pixelTripletIndex : cms::alpakatools::uniform_elements(acc, nPixelTriplets)) {
        if ((pixelTriplets.isDup()[pixelTripletIndex]))
          continue;

        unsigned int trackCandidateIdx =
            alpaka::atomicAdd(acc, &candsBase.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
        if (trackCandidateIdx >= n_max_pixel_track_candidates)  // This is done before any non-pixel TCs are added
        {
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
                                  ObjectRangesConst ranges) const {
      for (int idx : cms::alpakatools::uniform_elements_y(acc, nLowerModules)) {
        if (ranges.quintupletModuleIndices()[idx] == -1)
          continue;

        unsigned int nQuints = quintupletsOccupancy.nQuintuplets()[idx];
        for (unsigned int jdx : cms::alpakatools::uniform_elements_x(acc, nQuints)) {
          unsigned int quintupletIndex = ranges.quintupletModuleIndices()[idx] + jdx;
          if (quintuplets.isDup()[quintupletIndex] or quintuplets.partOfPT5()[quintupletIndex])
            continue;
          if (!(quintuplets.tightCutFlag()[quintupletIndex]))
            continue;

          unsigned int trackCandidateIdx =
              alpaka::atomicAdd(acc, &candsBase.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
          if (trackCandidateIdx - candsExtended.nTrackCandidatespT5() - candsExtended.nTrackCandidatespT3() >=
              n_max_nonpixel_track_candidates)  // pT5 and pT3 TCs have been added, but not pLS TCs
          {
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
                                  bool tc_pls_triplets) const {
      unsigned int nPixels = segmentsOccupancy.nSegments()[nLowerModules];
      for (unsigned int pixelArrayIndex : cms::alpakatools::uniform_elements(acc, nPixels)) {
        if ((tc_pls_triplets ? 0 : !pixelSeeds.isQuad()[pixelArrayIndex]) || (pixelSegments.isDup()[pixelArrayIndex]))
          continue;

        unsigned int trackCandidateIdx =
            alpaka::atomicAdd(acc, &candsBase.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
        if (trackCandidateIdx - candsExtended.nTrackCandidatesT5() >=
            n_max_pixel_track_candidates)  // T5 TCs have already been added
        {
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

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void countSharedT5HitsAndFindUnmatched(QuintupletsConst quintuplets,
                                                                        unsigned int candidateQuintupletIndex,
                                                                        unsigned int baseQuintupletIndex,
                                                                        int& sharedHitCount,
                                                                        int& unmatchedLayerSlot) {
    sharedHitCount = 0;
    unmatchedLayerSlot = -1;

    CMS_UNROLL_LOOP
    for (int layerIndex = 0; layerIndex < Params_T5::kLayers; ++layerIndex) {
      const unsigned int candidateHit0 = quintuplets.hitIndices()[candidateQuintupletIndex][2 * layerIndex + 0];
      const unsigned int candidateHit1 = quintuplets.hitIndices()[candidateQuintupletIndex][2 * layerIndex + 1];

      bool hit0InBase = false;
      bool hit1InBase = false;

      CMS_UNROLL_LOOP
      for (int baseHitIndex = 0; baseHitIndex < Params_T5::kHits; ++baseHitIndex) {
        const unsigned int baseHit = quintuplets.hitIndices()[baseQuintupletIndex][baseHitIndex];
        if (candidateHit0 == baseHit)
          hit0InBase = true;
        if (candidateHit1 == baseHit)
          hit1InBase = true;
        if (hit0InBase && hit1InBase)
          break;
      }

      if (hit0InBase)
        ++sharedHitCount;
      if (hit1InBase)
        ++sharedHitCount;

      if (!hit0InBase && !hit1InBase) {
        unmatchedLayerSlot = layerIndex;
      }
    }
  }

  struct ExtendTrackCandidatesFromDupT5 {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  ObjectRangesConst ranges,
                                  QuintupletsConst quintuplets,
                                  QuintupletsOccupancyConst quintupletsOccupancy,
                                  TrackCandidatesBase candsBase,
                                  TrackCandidatesExtended candsExtended) const {
      // Shared memory: best candidate per logical OT layer (1..11)
      int* sharedBestScoreBits = alpaka::declareSharedVar<int[lst::kLogicalOtLayers], __COUNTER__>(acc);
      int* sharedBestQuintupletIndex = alpaka::declareSharedVar<int[lst::kLogicalOtLayers], __COUNTER__>(acc);
      int* sharedBestLayerSlotInQuintuplet = alpaka::declareSharedVar<int[lst::kLogicalOtLayers], __COUNTER__>(acc);

      const unsigned int trackCandidateIndex = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
      const unsigned int nTrackCandidates = candsBase.nTrackCandidates();

      if (trackCandidateIndex >= nTrackCandidates)
        return;

      // Initialize shared memory once per block
      if (cms::alpakatools::once_per_block(acc)) {
        for (int logicalLayerBin = 0; logicalLayerBin < lst::kLogicalOtLayers; ++logicalLayerBin) {
          sharedBestScoreBits[logicalLayerBin] = bitCastFloatToInt(0.f);  // Stores the highest DNN score.
          sharedBestQuintupletIndex[logicalLayerBin] = -1;
          sharedBestLayerSlotInQuintuplet[logicalLayerBin] = -1;
        }
      }
      alpaka::syncBlockThreads(acc);

      const LSTObjType trackCandidateType = candsBase.trackCandidateType()[trackCandidateIndex];
      if (!(trackCandidateType == LSTObjType::T5 || trackCandidateType == LSTObjType::pT5))
        return;

      // Base quintuplet (for T5: objectIndices[0], for pT5: objectIndices[1])
      const unsigned int baseQuintupletIndex = (trackCandidateType == LSTObjType::T5)
                                                   ? candsExtended.objectIndices()[trackCandidateIndex][0]
                                                   : candsExtended.objectIndices()[trackCandidateIndex][1];

      const float baseEta = __H2F(quintuplets.eta()[baseQuintupletIndex]);
      const float basePhi = __H2F(quintuplets.phi()[baseQuintupletIndex]);
      const uint8_t baseStartLogicalLayer = quintuplets.logicalLayers()[baseQuintupletIndex][0];

      // Module range to scan
      int lowerModuleBegin = 0;
      int lowerModuleEnd = static_cast<int>(modules.nLowerModules());

      // If starting at layer 1, restrict to the module of the second hit
      if (baseStartLogicalLayer == 1) {
        lowerModuleBegin = quintuplets.lowerModuleIndices()[baseQuintupletIndex][1];
        lowerModuleEnd = lowerModuleBegin + 1;
      }

      // Flatten 2D thread indices within the block (Y, X) into one index
      const int threadIndexFlat = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[1u] *
                                      alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[2u] +
                                  alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[2u];

      const int blockDimFlat = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[1u] *
                               alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[2u];

      // Scan over lower modules
      for (int lowerModuleIndex = lowerModuleBegin + threadIndexFlat; lowerModuleIndex < lowerModuleEnd;
           lowerModuleIndex += blockDimFlat) {
        const int firstQuintupletInModule = ranges.quintupletModuleIndices()[lowerModuleIndex];
        if (firstQuintupletInModule == -1)
          continue;

        const unsigned int nQuintupletsInModule = quintupletsOccupancy.nQuintuplets()[lowerModuleIndex];

        // Scan over quintuplets in this module
        for (unsigned int quintupletOffset = 0; quintupletOffset < nQuintupletsInModule; ++quintupletOffset) {
          const unsigned int candidateQuintupletIndex = firstQuintupletInModule + quintupletOffset;
          if (candidateQuintupletIndex == baseQuintupletIndex)
            continue;

          // Require different starting layer
          if (quintuplets.logicalLayers()[candidateQuintupletIndex][0] == baseStartLogicalLayer)
            continue;

          // Quick eta / phi window selection
          const float candidateEta = __H2F(quintuplets.eta()[candidateQuintupletIndex]);
          if (alpaka::math::abs(acc, baseEta - candidateEta) > 0.1f)
            continue;

          const float candidatePhi = __H2F(quintuplets.phi()[candidateQuintupletIndex]);
          if (alpaka::math::abs(acc, cms::alpakatools::deltaPhi(acc, basePhi, candidatePhi)) > 0.1f)
            continue;

          // Embedding distance
          float embedDistance2 = 0.f;
          CMS_UNROLL_LOOP
          for (unsigned int embedIndex = 0; embedIndex < Params_T5::kEmbed; ++embedIndex) {
            const float diff = quintuplets.t5Embed()[baseQuintupletIndex][embedIndex] -
                               quintuplets.t5Embed()[candidateQuintupletIndex][embedIndex];
            embedDistance2 += diff * diff;
          }
          if (embedDistance2 > 1.0f)
            continue;

          // Hit matching against base T5 hits
          int sharedHitCount = 0;
          int unmatchedLayerSlot = -1;

          countSharedT5HitsAndFindUnmatched(
              quintuplets, candidateQuintupletIndex, baseQuintupletIndex, sharedHitCount, unmatchedLayerSlot);

          if (sharedHitCount < lst::kT5DuplicateMinSharedHits)
            continue;
          if (unmatchedLayerSlot < 0)
            continue;

          // Candidate score is the T5 DNN output
          const float candidateScore = quintuplets.dnnScore()[candidateQuintupletIndex];

          // New OT logical layer from the candidate
          const uint8_t newLogicalLayer = quintuplets.logicalLayers()[candidateQuintupletIndex][unmatchedLayerSlot];
          const int logicalLayerBin = static_cast<int>(newLogicalLayer) - 1;  // 0..kLogicalOtLayers-1

          // Atomic CAS update on best score for this logical layer
          bool updateDone = false;
          while (!updateDone) {
            const int oldScoreBits = sharedBestScoreBits[logicalLayerBin];
            const float oldScore = bitCastIntToFloat(oldScoreBits);

            if (candidateScore <= oldScore) {
              updateDone = true;
            } else {
              const int newScoreBits = bitCastFloatToInt(candidateScore);
              const int assumedBits = alpaka::atomicCas(
                  acc, &sharedBestScoreBits[logicalLayerBin], oldScoreBits, newScoreBits, alpaka::hierarchy::Threads{});
              if (assumedBits == oldScoreBits) {
                sharedBestQuintupletIndex[logicalLayerBin] = static_cast<int>(candidateQuintupletIndex);
                sharedBestLayerSlotInQuintuplet[logicalLayerBin] = unmatchedLayerSlot;
                updateDone = true;
              }
            }
          }
        }
      }

      // One thread per block finalizes by actually extending the TC
      alpaka::syncBlockThreads(acc);

      if (cms::alpakatools::once_per_block(acc)) {
        CMS_UNROLL_LOOP
        for (int logicalLayerBin = 0; logicalLayerBin < lst::kLogicalOtLayers; ++logicalLayerBin) {
          const int bestQuintupletIndex = sharedBestQuintupletIndex[logicalLayerBin];
          if (bestQuintupletIndex < 0)
            continue;

          const int bestQuintupletLayerSlot = sharedBestLayerSlotInQuintuplet[logicalLayerBin];
          const uint8_t logicalLayer = static_cast<uint8_t>(logicalLayerBin + 1);  // 1..11
          const int layerSlot = (logicalLayer - 1) + 2;                            // OT slots for TC

          const uint16_t lowerModuleIndex =
              quintuplets.lowerModuleIndices()[bestQuintupletIndex][bestQuintupletLayerSlot];
          const unsigned int hitIndex0 = quintuplets.hitIndices()[bestQuintupletIndex][2 * bestQuintupletLayerSlot + 0];
          const unsigned int hitIndex1 = quintuplets.hitIndices()[bestQuintupletIndex][2 * bestQuintupletLayerSlot + 1];

          placeLayer(candsBase,
                     candsExtended,
                     trackCandidateIndex,
                     layerSlot,
                     logicalLayer,
                     lowerModuleIndex,
                     hitIndex0,
                     hitIndex1);
        }
      }
    }
  };

  struct AddpT5asTrackCandidate {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  uint16_t nLowerModules,
                                  PixelQuintupletsConst pixelQuintuplets,
                                  TrackCandidatesBase candsBase,
                                  TrackCandidatesExtended candsExtended,
                                  PixelSeedsConst pixelSeeds,
                                  ObjectRangesConst ranges) const {
      // implementation is 1D with a single block
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      int nPixelQuintuplets = pixelQuintuplets.nPixelQuintuplets();
      unsigned int pLS_offset = ranges.segmentModuleIndices()[nLowerModules];
      for (int pixelQuintupletIndex : cms::alpakatools::uniform_elements(acc, nPixelQuintuplets)) {
        if (pixelQuintuplets.isDup()[pixelQuintupletIndex])
          continue;

        unsigned int trackCandidateIdx =
            alpaka::atomicAdd(acc, &candsBase.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
        if (trackCandidateIdx >= n_max_pixel_track_candidates)  // No other TCs have been added yet
        {
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
          addTrackCandidateToMemory(candsBase,
                                    candsExtended,
                                    LSTObjType::pT5,
                                    pT5PixelIndex,
                                    pixelQuintuplets.quintupletIndices()[pixelQuintupletIndex],
                                    pixelQuintuplets.logicalLayers()[pixelQuintupletIndex].data(),
                                    pixelQuintuplets.lowerModuleIndices()[pixelQuintupletIndex].data(),
                                    pixelQuintuplets.hitIndices()[pixelQuintupletIndex].data(),
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
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(lst::TrackCandidatesBaseDeviceCollection, lst::TrackCandidatesBaseHostCollection);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(lst::TrackCandidatesExtendedDeviceCollection,
                                      lst::TrackCandidatesExtendedHostCollection);

#endif

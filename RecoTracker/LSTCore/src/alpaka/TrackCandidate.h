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
    candsBase.hitIndices()[trackCandidateIndex][0] = hitIndices[0];
    candsBase.hitIndices()[trackCandidateIndex][1] = hitIndices[2];
    candsBase.hitIndices()[trackCandidateIndex][2] = hitIndices[1];
    candsBase.hitIndices()[trackCandidateIndex][3] = hitIndices[3];
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void placeLayer(TrackCandidatesBase& base,
                                                 TrackCandidatesExtended& ext,
                                                 unsigned int tcIdx,
                                                 int slot,              // 0..12 (0/1 = pixel, 2..12 = OT L=1..11)
                                                 uint8_t logicalLayer,  // 0 for pixel, 1..11 for OT
                                                 uint16_t lowerMod,
                                                 unsigned int hit0,
                                                 unsigned int hit1) {
    if (ext.lowerModuleIndices()[tcIdx][slot] == ::lst::kTCEmptyLowerModule) {
      base.nLayers()[tcIdx] += 1;
    }
    ext.logicalLayers()[tcIdx][slot] = logicalLayer;
    ext.lowerModuleIndices()[tcIdx][slot] = lowerMod;
    base.hitIndices()[tcIdx][2 * slot + 0] = hit0;
    base.hitIndices()[tcIdx][2 * slot + 1] = hit1;
    base.nHits()[tcIdx] += 2;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTrackCandidateToMemoryByLayer(TrackCandidatesBase& candsBase,
                                                                       TrackCandidatesExtended& candsExtended,
                                                                       LSTObjType type,
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
    candsBase.trackCandidateType()[trackCandidateIndex] = type;
    candsExtended.directObjectIndices()[trackCandidateIndex] = directObjectIndex;
    candsBase.pixelSeedIndex()[trackCandidateIndex] = pixelSeedIndex;

    candsExtended.objectIndices()[trackCandidateIndex][0] = innerTrackletIndex;
    candsExtended.objectIndices()[trackCandidateIndex][1] = outerTrackletIndex;

    // Zero counts
    candsBase.nLayers()[trackCandidateIndex] = 0;
    candsBase.nHits()[trackCandidateIndex] = 0;

    // Mark all slots empty and zero the hits
    CMS_UNROLL_LOOP for (int s = 0; s < Params_TC::kLayers; ++s) {
      candsExtended.logicalLayers()[trackCandidateIndex][s] = 0;  // 0 means 'pixel' if filled; harmless for empty
      candsExtended.lowerModuleIndices()[trackCandidateIndex][s] = ::lst::kTCEmptyLowerModule;
    }
    CMS_UNROLL_LOOP for (int i = 0; i < Params_TC::kHits; ++i) { candsBase.hitIndices()[trackCandidateIndex][i] = 0u; }

    // --- Write layers by slot mapping ---
    if (type == LSTObjType::T5) {
      // T5 has 5 OT logical layers L in [1..11]; map to slots (L-1)+2 in [2..12]
      CMS_UNROLL_LOOP for (int i = 0; i < Params_T5::kLayers; ++i) {
        const uint8_t L = logicalLayerIndices[i];  // 1..11
        const int slot = (L - 1) + 2;              // OT slots
        placeLayer(candsBase,
                   candsExtended,
                   trackCandidateIndex,
                   slot,
                   L,
                   lowerModuleIndices[i],
                   hitIndices[2 * i + 0],
                   hitIndices[2 * i + 1]);
      }
    } else if (type == LSTObjType::pT5) {
      // Pixel layers occupy slots 0 and 1 with logicalLayer=0
      placeLayer(
          candsBase, candsExtended, trackCandidateIndex, 0, 0, lowerModuleIndices[0], hitIndices[0], hitIndices[1]);
      placeLayer(
          candsBase, candsExtended, trackCandidateIndex, 1, 0, lowerModuleIndices[1], hitIndices[2], hitIndices[3]);

      // pT5 then carries the T5's 5 OT layers starting at logicalLayerIndices[2]; hits start at index 4
      CMS_UNROLL_LOOP for (int i = 2; i < Params_pT5::kLayers; ++i) {
        const uint8_t L = logicalLayerIndices[i];  // 1..11
        const int slot = (L - 1) + 2;              // OT slots
        const int base = 4 + 2 * (i - 2);          // map into 10 OT hits that follow the 4 pixel hits
        placeLayer(candsBase,
                   candsExtended,
                   trackCandidateIndex,
                   slot,
                   L,
                   lowerModuleIndices[i],
                   hitIndices[base + 0],
                   hitIndices[base + 1]);
      }
    } else {
      // Legacy contiguous writer for other types (pT3, pLS).
      const size_t limits = (type == LSTObjType::pT3) ? Params_pT3::kLayers : Params_pLS::kLayers;

      // Fill the first 'limits' slots contiguously and leave the rest as empty sentinel.
      CMS_UNROLL_LOOP for (size_t i = 0; i < limits; ++i) {
        candsExtended.logicalLayers()[trackCandidateIndex][i] = logicalLayerIndices[i];
        candsExtended.lowerModuleIndices()[trackCandidateIndex][i] = lowerModuleIndices[i];
      }
      CMS_UNROLL_LOOP for (size_t i = 0; i < 2 * limits; ++i) {
        candsBase.hitIndices()[trackCandidateIndex][i] = hitIndices[i];
      }
      candsBase.nLayers()[trackCandidateIndex] = static_cast<uint8_t>(limits);
      candsBase.nHits()[trackCandidateIndex] = static_cast<uint8_t>(2 * limits);
    }

    candsExtended.centerX()[trackCandidateIndex] = __F2H(centerX);
    candsExtended.centerY()[trackCandidateIndex] = __F2H(centerY);
    candsExtended.radius()[trackCandidateIndex] = __F2H(radius);
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

    size_t limits = trackCandidateType == LSTObjType::pT5 ? Params_pT5::kLayers : Params_pT3::kLayers;

    //send the starting pointer to the logicalLayer and hitIndices
    for (size_t i = 0; i < limits; i++) {
      candsExtended.logicalLayers()[trackCandidateIndex][i] = logicalLayerIndices[i];
      candsExtended.lowerModuleIndices()[trackCandidateIndex][i] = lowerModuleIndices[i];
    }
    for (size_t i = 0; i < 2 * limits; i++) {
      candsBase.hitIndices()[trackCandidateIndex][i] = hitIndices[i];
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
            addTrackCandidateToMemoryByLayer(candsBase,
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

  struct ExtendTrackCandidatesFromDupT5 {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  ObjectRangesConst ranges,
                                  QuintupletsConst quintuplets,
                                  QuintupletsOccupancyConst quintupletsOcc,
                                  TrackCandidatesBase candsBase,
                                  TrackCandidatesExtended candsExtended) const {
      // Shared memory to store the best candidates per layer.
      int* sharedBestScore = alpaka::declareSharedVar<int[11], __COUNTER__>(acc);
      int* sharedBestQIdx = alpaka::declareSharedVar<int[11], __COUNTER__>(acc);
      int* sharedBestK = alpaka::declareSharedVar<int[11], __COUNTER__>(acc);

      // Helpers for bit casting between float and int using union (Safe against strict-aliasing)
      auto float_as_int = [](float f) -> int {
        union {
          float f_val;
          int i_val;
        } u;
        u.f_val = f;
        return u.i_val;
      };

      auto int_as_float = [](int i) -> float {
        union {
          int i_val;
          float f_val;
        } u;
        u.i_val = i;
        return u.f_val;
      };

      // 1. Identify which Track Candidate this Block is processing.
      const unsigned int tcIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
      const unsigned int nTC = candsBase.nTrackCandidates();

      if (tcIdx >= nTC)
        return;

      // Initialize shared memory (Thread 0 of the block does this)
      if (cms::alpakatools::once_per_block(acc)) {
        int minScoreBits = float_as_int(-1e30f);
        for (int L = 0; L < 11; ++L) {
          sharedBestScore[L] = minScoreBits;
          sharedBestQIdx[L] = -1;
          sharedBestK[L] = -1;
        }
      }
      alpaka::syncBlockThreads(acc);

      // Check TC Type constraints
      const LSTObjType type = candsBase.trackCandidateType()[tcIdx];
      if (!(type == LSTObjType::T5 || type == LSTObjType::pT5))
        return;

      // 2. Load TC-specific data
      const unsigned int baseT5 =
          (type == LSTObjType::T5) ? candsExtended.objectIndices()[tcIdx][0] : candsExtended.objectIndices()[tcIdx][1];

      const float etaBase = __H2F(quintuplets.eta()[baseT5]);
      const float phiBase = __H2F(quintuplets.phi()[baseT5]);
      const uint8_t baseStart = quintuplets.logicalLayers()[baseT5][0];

      // Determine Module Search Range
      int start_lowmod = 0;
      int end_lowmod = static_cast<int>(modules.nLowerModules());

      // Optimization: If Layer 1, restrict search to the module of the 2nd hit
      if (baseStart == 1) {
        start_lowmod = quintuplets.lowerModuleIndices()[baseT5][1];
        end_lowmod = start_lowmod + 1;
      }

      // 3. Parallel Scan over Modules and Quintuplets
      // Flatten block dimensions 1 (Y) and 2 (X) to create a strided loop.
      const int threadIdFlat = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[1u] *
                                   alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[2u] +
                               alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[2u];

      const int blockDimFlat = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[1u] *
                               alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[2u];

      for (int lowmod = start_lowmod + threadIdFlat; lowmod < end_lowmod; lowmod += blockDimFlat) {
        const int baseIdx = ranges.quintupletModuleIndices()[lowmod];
        if (baseIdx == -1)
          continue;

        const unsigned int nQuints = quintupletsOcc.nQuintuplets()[lowmod];

        // Inner Loop: Iterate over T5s in this module
        for (unsigned int off = 0; off < nQuints; ++off) {
          const unsigned int q = baseIdx + off;
          if (q == baseT5)
            continue;

          // -- Checks --
          if (quintuplets.logicalLayers()[q][0] == baseStart)
            continue;

          const float etaQ = __H2F(quintuplets.eta()[q]);
          if (alpaka::math::abs(acc, etaBase - etaQ) > 0.1f)
            continue;

          const float phiQ = __H2F(quintuplets.phi()[q]);
          if (alpaka::math::abs(acc, cms::alpakatools::deltaPhi(acc, phiBase, phiQ)) > 0.1f)
            continue;

          // Embed check
          float d2 = 0.f;
          CMS_UNROLL_LOOP
          for (unsigned int kk = 0; kk < Params_T5::kEmbed; ++kk) {
            float diff = quintuplets.t5Embed()[baseT5][kk] - quintuplets.t5Embed()[q][kk];
            d2 += diff * diff;
          }
          if (d2 > 1.0f)
            continue;

          // Hit Matching Logic
          int sharedCount = 0;
          int kstar = -1;

          // Brute force check against base T5 hits (10 hits)
          for (int k = 0; k < Params_T5::kLayers; ++k) {
            const unsigned int h0 = quintuplets.hitIndices()[q][2 * k + 0];
            const unsigned int h1 = quintuplets.hitIndices()[q][2 * k + 1];

            bool f0 = false, f1 = false;
            CMS_UNROLL_LOOP for (int i = 0; i < 10; ++i) {
              unsigned int bh = quintuplets.hitIndices()[baseT5][i];
              if (h0 == bh)
                f0 = true;
              if (h1 == bh)
                f1 = true;
            }

            if (f0)
              sharedCount++;
            if (f1)
              sharedCount++;

            if (!f0 && !f1) {
              kstar = k;
            }
          }

          if (sharedCount < 8)
            continue;
          if (kstar < 0)
            continue;

          const uint8_t Lnew = quintuplets.logicalLayers()[q][kstar];
          if (Lnew < 1 || Lnew > 11)
            continue;

          // Check if Lnew is already present in Base TC
          bool present = false;
          CMS_UNROLL_LOOP for (int i = 0; i < Params_T5::kLayers; ++i) {
            if (quintuplets.logicalLayers()[baseT5][i] == Lnew) {
              present = true;
              break;
            }
          }
          if (present)
            continue;

          // -- Atomic Update of Best Candidate --
          const float score = __H2F(quintuplets.dnnScore()[q]);
          int layerIdx = Lnew - 1;

          // Spin loop for float max atomic update
          bool done = false;
          while (!done) {
            int oldInt = sharedBestScore[layerIdx];
            float oldVal = int_as_float(oldInt);

            if (score <= oldVal) {
              done = true;
            } else {
              int newInt = float_as_int(score);
              // atomicCas on shared memory using Threads hierarchy
              int assumed =
                  alpaka::atomicCas(acc, &sharedBestScore[layerIdx], oldInt, newInt, alpaka::hierarchy::Threads{});
              if (assumed == oldInt) {
                // Success updating score, blindly update indices
                sharedBestQIdx[layerIdx] = (int)q;
                sharedBestK[layerIdx] = kstar;
                done = true;
              }
            }
          }
        }
      }

      // 4. Final Write Back (One thread per TC block)
      alpaka::syncBlockThreads(acc);

      if (cms::alpakatools::once_per_block(acc)) {
        CMS_UNROLL_LOOP for (int li = 0; li < 11; ++li) {
          const int q = sharedBestQIdx[li];
          if (q < 0)
            continue;

          const int k = sharedBestK[li];
          const uint8_t L = static_cast<uint8_t>(li + 1);
          const int slot = (L - 1) + 2;

          const uint16_t mod = quintuplets.lowerModuleIndices()[q][k];
          const unsigned int h0 = quintuplets.hitIndices()[q][2 * k + 0];
          const unsigned int h1 = quintuplets.hitIndices()[q][2 * k + 1];

          placeLayer(candsBase, candsExtended, tcIdx, slot, L, mod, h0, h1);
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
          addTrackCandidateToMemoryByLayer(candsBase,
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

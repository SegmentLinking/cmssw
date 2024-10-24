#ifndef RecoTracker_LSTCore_src_alpaka_TrackCandidate_h
#define RecoTracker_LSTCore_src_alpaka_TrackCandidate_h

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/Module.h"
#include "RecoTracker/LSTCore/interface/TrackCandidatesHostCollection.h"
#include "RecoTracker/LSTCore/interface/TrackCandidatesSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/TrackCandidatesDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"

#include "Triplet.h"
#include "Segment.h"
#include "MiniDoublet.h"
#include "PixelTriplet.h"
#include "Quintuplet.h"
#include "Hit.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addpLSTrackCandidateToMemory(TrackCandidates& cands,
                                                                   unsigned int trackletIndex,
                                                                   unsigned int trackCandidateIndex,
                                                                   uint4 hitIndices,
                                                                   int pixelSeedIndex) {
    cands.trackCandidateType()[trackCandidateIndex] = 8;  // type for pLS
    cands.directObjectIndices()[trackCandidateIndex] = trackletIndex;
    cands.pixelSeedIndex()[trackCandidateIndex] = pixelSeedIndex;

    cands.objectIndices()[trackCandidateIndex][0] = trackletIndex;
    cands.objectIndices()[trackCandidateIndex][1] = trackletIndex;

    cands.hitIndices()[trackCandidateIndex][0] =
        hitIndices.x;  // Order explanation in https://github.com/SegmentLinking/TrackLooper/issues/267
    cands.hitIndices()[trackCandidateIndex][1] = hitIndices.z;
    cands.hitIndices()[trackCandidateIndex][2] = hitIndices.y;
    cands.hitIndices()[trackCandidateIndex][3] = hitIndices.w;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTrackCandidateToMemory(TrackCandidates& cands,
                                                                short trackCandidateType,
                                                                unsigned int innerTrackletIndex,
                                                                unsigned int outerTrackletIndex,
                                                                uint8_t* logicalLayerIndices,
                                                                uint16_t* lowerModuleIndices,
                                                                unsigned int* hitIndices,
                                                                int pixelSeedIndex,
                                                                float centerX,
                                                                float centerY,
                                                                float radius,
                                                                unsigned int trackCandidateIndex,
                                                                unsigned int directObjectIndex) {
    cands.trackCandidateType()[trackCandidateIndex] = trackCandidateType;
    cands.directObjectIndices()[trackCandidateIndex] = directObjectIndex;
    cands.pixelSeedIndex()[trackCandidateIndex] = pixelSeedIndex;

    cands.objectIndices()[trackCandidateIndex][0] = innerTrackletIndex;
    cands.objectIndices()[trackCandidateIndex][1] = outerTrackletIndex;

    size_t limits = trackCandidateType == 7
                        ? Params_pT5::kLayers
                        : Params_pT3::kLayers;  // 7 means pT5, Params_pT3::kLayers = Params_T5::kLayers = 5

    //send the starting pointer to the logicalLayer and hitIndices
    for (size_t i = 0; i < limits; i++) {
      cands.logicalLayers()[trackCandidateIndex][i] = logicalLayerIndices[i];
      cands.lowerModuleIndices()[trackCandidateIndex][i] = lowerModuleIndices[i];
    }
    for (size_t i = 0; i < 2 * limits; i++) {
      cands.hitIndices()[trackCandidateIndex][i] = hitIndices[i];
    }
    cands.centerX()[trackCandidateIndex] = __F2H(centerX);
    cands.centerY()[trackCandidateIndex] = __F2H(centerY);
    cands.radius()[trackCandidateIndex] = __F2H(radius);
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int checkPixelHits(
      unsigned int ix, unsigned int jx, MiniDoubletsConst mds, SegmentsConst segments, HitsConst hits) {
    int phits1[Params_pLS::kHits];
    int phits2[Params_pLS::kHits];

    phits1[0] = hits.idxs()[mds.anchorHitIndices()[segments.mdIndices()[ix][0]]];
    phits1[1] = hits.idxs()[mds.anchorHitIndices()[segments.mdIndices()[ix][1]]];
    phits1[2] = hits.idxs()[mds.outerHitIndices()[segments.mdIndices()[ix][0]]];
    phits1[3] = hits.idxs()[mds.outerHitIndices()[segments.mdIndices()[ix][1]]];

    phits2[0] = hits.idxs()[mds.anchorHitIndices()[segments.mdIndices()[jx][0]]];
    phits2[1] = hits.idxs()[mds.anchorHitIndices()[segments.mdIndices()[jx][1]]];
    phits2[2] = hits.idxs()[mds.outerHitIndices()[segments.mdIndices()[jx][0]]];
    phits2[3] = hits.idxs()[mds.outerHitIndices()[segments.mdIndices()[jx][1]]];

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
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  Modules modulesInGPU,
                                  ObjectOccupancyConst objectOccupancy,
                                  PixelTriplets pixelTripletsInGPU,
                                  SegmentsPixelConst segmentsPixel,
                                  PixelQuintuplets pixelQuintupletsInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      unsigned int nPixelTriplets = *pixelTripletsInGPU.nPixelTriplets;
      for (unsigned int pixelTripletIndex = globalThreadIdx[2]; pixelTripletIndex < nPixelTriplets;
           pixelTripletIndex += gridThreadExtent[2]) {
        if (pixelTripletsInGPU.isDup[pixelTripletIndex])
          continue;

        // Cross cleaning step
        float eta1 = __H2F(pixelTripletsInGPU.eta_pix[pixelTripletIndex]);
        float phi1 = __H2F(pixelTripletsInGPU.phi_pix[pixelTripletIndex]);

        int pixelModuleIndex = *modulesInGPU.nLowerModules;
        unsigned int prefix = objectOccupancy.segmentModuleIndices()[pixelModuleIndex];

        unsigned int nPixelQuintuplets = *pixelQuintupletsInGPU.nPixelQuintuplets;
        for (unsigned int pixelQuintupletIndex = globalThreadIdx[1]; pixelQuintupletIndex < nPixelQuintuplets;
             pixelQuintupletIndex += gridThreadExtent[1]) {
          unsigned int pLS_jx = pixelQuintupletsInGPU.pixelIndices[pixelQuintupletIndex];
          float eta2 = segmentsPixel.eta()[pLS_jx - prefix];
          float phi2 = segmentsPixel.phi()[pLS_jx - prefix];
          float dEta = alpaka::math::abs(acc, (eta1 - eta2));
          float dPhi = calculate_dPhi(phi1, phi2);

          float dR2 = dEta * dEta + dPhi * dPhi;
          if (dR2 < 1e-5f)
            pixelTripletsInGPU.isDup[pixelTripletIndex] = true;
        }
      }
    }
  };

  struct CrossCleanT5 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  Modules modulesInGPU,
                                  Quintuplets quintupletsInGPU,
                                  PixelQuintuplets pixelQuintupletsInGPU,
                                  PixelTriplets pixelTripletsInGPU,
                                  ObjectOccupancyConst objectOccupancy) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (int innerInnerInnerLowerModuleArrayIndex = globalThreadIdx[0];
           innerInnerInnerLowerModuleArrayIndex < *(modulesInGPU.nLowerModules);
           innerInnerInnerLowerModuleArrayIndex += gridThreadExtent[0]) {
        if (objectOccupancy.quintupletModuleIndices()[innerInnerInnerLowerModuleArrayIndex] == -1)
          continue;

        unsigned int nQuints = quintupletsInGPU.nQuintuplets[innerInnerInnerLowerModuleArrayIndex];
        for (unsigned int innerObjectArrayIndex = globalThreadIdx[1]; innerObjectArrayIndex < nQuints;
             innerObjectArrayIndex += gridThreadExtent[1]) {
          unsigned int quintupletIndex =
              objectOccupancy.quintupletModuleIndices()[innerInnerInnerLowerModuleArrayIndex] + innerObjectArrayIndex;

          // Don't add duplicate T5s or T5s that are accounted in pT5s
          if (quintupletsInGPU.isDup[quintupletIndex] or quintupletsInGPU.partOfPT5[quintupletIndex])
            continue;
#ifdef Crossclean_T5
          unsigned int loop_bound = *pixelQuintupletsInGPU.nPixelQuintuplets + *pixelTripletsInGPU.nPixelTriplets;
          // Cross cleaning step
          float eta1 = __H2F(quintupletsInGPU.eta[quintupletIndex]);
          float phi1 = __H2F(quintupletsInGPU.phi[quintupletIndex]);

          for (unsigned int jx = globalThreadIdx[2]; jx < loop_bound; jx += gridThreadExtent[2]) {
            float eta2, phi2;
            if (jx < *pixelQuintupletsInGPU.nPixelQuintuplets) {
              eta2 = __H2F(pixelQuintupletsInGPU.eta[jx]);
              phi2 = __H2F(pixelQuintupletsInGPU.phi[jx]);
            } else {
              eta2 = __H2F(pixelTripletsInGPU.eta[jx - *pixelQuintupletsInGPU.nPixelQuintuplets]);
              phi2 = __H2F(pixelTripletsInGPU.phi[jx - *pixelQuintupletsInGPU.nPixelQuintuplets]);
            }

            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = calculate_dPhi(phi1, phi2);

            float dR2 = dEta * dEta + dPhi * dPhi;
            if (dR2 < 1e-3f)
              quintupletsInGPU.isDup[quintupletIndex] = true;
          }
#endif
        }
      }
    }
  };

  struct CrossCleanpLS {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  Modules modulesInGPU,
                                  ObjectOccupancyConst objectOccupancy,
                                  PixelTriplets pixelTripletsInGPU,
                                  TrackCandidates cands,
                                  SegmentsConst segments,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  SegmentsPixel segmentsPixel,
                                  MiniDoubletsConst mds,
                                  HitsConst hits,
                                  Quintuplets quintupletsInGPU) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      int pixelModuleIndex = *modulesInGPU.nLowerModules;
      unsigned int nPixels = segmentsOccupancy.nSegments()[pixelModuleIndex];
      for (unsigned int pixelArrayIndex = globalThreadIdx[2]; pixelArrayIndex < nPixels;
           pixelArrayIndex += gridThreadExtent[2]) {
        if (!segmentsPixel.isQuad()[pixelArrayIndex] || segmentsPixel.isDup()[pixelArrayIndex])
          continue;

        float eta1 = segmentsPixel.eta()[pixelArrayIndex];
        float phi1 = segmentsPixel.phi()[pixelArrayIndex];
        unsigned int prefix = objectOccupancy.segmentModuleIndices()[pixelModuleIndex];

        unsigned int nTrackCandidates = cands.nTrackCandidates();
        for (unsigned int trackCandidateIndex = globalThreadIdx[1]; trackCandidateIndex < nTrackCandidates;
             trackCandidateIndex += gridThreadExtent[1]) {
          short type = cands.trackCandidateType()[trackCandidateIndex];
          unsigned int innerTrackletIdx = cands.objectIndices()[trackCandidateIndex][0];
          if (type == 4)  // T5
          {
            unsigned int quintupletIndex = innerTrackletIdx;  // T5 index
            float eta2 = __H2F(quintupletsInGPU.eta[quintupletIndex]);
            float phi2 = __H2F(quintupletsInGPU.phi[quintupletIndex]);
            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = calculate_dPhi(phi1, phi2);

            float dR2 = dEta * dEta + dPhi * dPhi;
            if (dR2 < 1e-3f)
              segmentsPixel.isDup()[pixelArrayIndex] = true;
          }
          if (type == 5)  // pT3
          {
            int pLSIndex = pixelTripletsInGPU.pixelSegmentIndices[innerTrackletIdx];
            int npMatched = checkPixelHits(prefix + pixelArrayIndex, pLSIndex, mds, segments, hits);
            if (npMatched > 0)
              segmentsPixel.isDup()[pixelArrayIndex] = true;

            int pT3Index = innerTrackletIdx;
            float eta2 = __H2F(pixelTripletsInGPU.eta_pix[pT3Index]);
            float phi2 = __H2F(pixelTripletsInGPU.phi_pix[pT3Index]);
            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = calculate_dPhi(phi1, phi2);

            float dR2 = dEta * dEta + dPhi * dPhi;
            if (dR2 < 0.000001f)
              segmentsPixel.isDup()[pixelArrayIndex] = true;
          }
          if (type == 7)  // pT5
          {
            unsigned int pLSIndex = innerTrackletIdx;
            int npMatched = checkPixelHits(prefix + pixelArrayIndex, pLSIndex, mds, segments, hits);
            if (npMatched > 0) {
              segmentsPixel.isDup()[pixelArrayIndex] = true;
            }

            float eta2 = segmentsPixel.eta()[pLSIndex - prefix];
            float phi2 = segmentsPixel.phi()[pLSIndex - prefix];
            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = calculate_dPhi(phi1, phi2);

            float dR2 = dEta * dEta + dPhi * dPhi;
            if (dR2 < 0.000001f)
              segmentsPixel.isDup()[pixelArrayIndex] = true;
          }
        }
      }
    }
  };

  struct AddpT3asTrackCandidatesInGPU {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  uint16_t nLowerModules,
                                  PixelTriplets pixelTripletsInGPU,
                                  TrackCandidates cands,
                                  SegmentsPixelConst segmentsPixel,
                                  ObjectOccupancyConst objectOccupancy) const {
      // implementation is 1D with a single block
      static_assert(std::is_same_v<TAcc, ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>, "Should be Acc1D");
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      unsigned int nPixelTriplets = *pixelTripletsInGPU.nPixelTriplets;
      unsigned int pLS_offset = objectOccupancy.segmentModuleIndices()[nLowerModules];
      for (unsigned int pixelTripletIndex = globalThreadIdx[0]; pixelTripletIndex < nPixelTriplets;
           pixelTripletIndex += gridThreadExtent[0]) {
        if ((pixelTripletsInGPU.isDup[pixelTripletIndex]))
          continue;

        unsigned int trackCandidateIdx =
            alpaka::atomicAdd(acc, &cands.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
        if (trackCandidateIdx >= n_max_pixel_track_candidates)  // This is done before any non-pixel TCs are added
        {
#ifdef WARNINGS
          printf("Track Candidate excess alert! Type = pT3");
#endif
          alpaka::atomicSub(acc, &cands.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
          break;

        } else {
          alpaka::atomicAdd(acc, &cands.nTrackCandidatespT3(), 1u, alpaka::hierarchy::Threads{});

          float radius = 0.5f * (__H2F(pixelTripletsInGPU.pixelRadius[pixelTripletIndex]) +
                                 __H2F(pixelTripletsInGPU.tripletRadius[pixelTripletIndex]));
          unsigned int pT3PixelIndex = pixelTripletsInGPU.pixelSegmentIndices[pixelTripletIndex];
          addTrackCandidateToMemory(cands,
                                    5 /*track candidate type pT3=5*/,
                                    pixelTripletIndex,
                                    pixelTripletIndex,
                                    &pixelTripletsInGPU.logicalLayers[Params_pT3::kLayers * pixelTripletIndex],
                                    &pixelTripletsInGPU.lowerModuleIndices[Params_pT3::kLayers * pixelTripletIndex],
                                    &pixelTripletsInGPU.hitIndices[Params_pT3::kHits * pixelTripletIndex],
                                    segmentsPixel.seedIdx()[pT3PixelIndex - pLS_offset],
                                    __H2F(pixelTripletsInGPU.centerX[pixelTripletIndex]),
                                    __H2F(pixelTripletsInGPU.centerY[pixelTripletIndex]),
                                    radius,
                                    trackCandidateIdx,
                                    pixelTripletIndex);
        }
      }
    }
  };

  struct AddT5asTrackCandidateInGPU {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  uint16_t nLowerModules,
                                  Quintuplets quintupletsInGPU,
                                  TrackCandidates cands,
                                  ObjectOccupancyConst objectOccupancy) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (int idx = globalThreadIdx[1]; idx < nLowerModules; idx += gridThreadExtent[1]) {
        if (objectOccupancy.quintupletModuleIndices()[idx] == -1)
          continue;

        unsigned int nQuints = quintupletsInGPU.nQuintuplets[idx];
        for (unsigned int jdx = globalThreadIdx[2]; jdx < nQuints; jdx += gridThreadExtent[2]) {
          unsigned int quintupletIndex = objectOccupancy.quintupletModuleIndices()[idx] + jdx;
          if (quintupletsInGPU.isDup[quintupletIndex] or quintupletsInGPU.partOfPT5[quintupletIndex])
            continue;
          if (!(quintupletsInGPU.TightCutFlag[quintupletIndex]))
            continue;

          unsigned int trackCandidateIdx =
              alpaka::atomicAdd(acc, &cands.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
          if (trackCandidateIdx - cands.nTrackCandidatespT5() - cands.nTrackCandidatespT3() >=
              n_max_nonpixel_track_candidates)  // pT5 and pT3 TCs have been added, but not pLS TCs
          {
#ifdef WARNINGS
            printf("Track Candidate excess alert! Type = T5");
#endif
            alpaka::atomicSub(acc, &cands.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
            break;
          } else {
            alpaka::atomicAdd(acc, &cands.nTrackCandidatesT5(), 1u, alpaka::hierarchy::Threads{});
            addTrackCandidateToMemory(cands,
                                      4 /*track candidate type T5=4*/,
                                      quintupletIndex,
                                      quintupletIndex,
                                      &quintupletsInGPU.logicalLayers[Params_T5::kLayers * quintupletIndex],
                                      &quintupletsInGPU.lowerModuleIndices[Params_T5::kLayers * quintupletIndex],
                                      &quintupletsInGPU.hitIndices[Params_T5::kHits * quintupletIndex],
                                      -1 /*no pixel seed index for T5s*/,
                                      quintupletsInGPU.regressionG[quintupletIndex],
                                      quintupletsInGPU.regressionF[quintupletIndex],
                                      quintupletsInGPU.regressionRadius[quintupletIndex],
                                      trackCandidateIdx,
                                      quintupletIndex);
          }
        }
      }
    }
  };

  struct AddpLSasTrackCandidateInGPU {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  uint16_t nLowerModules,
                                  TrackCandidates cands,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  SegmentsPixelConst segmentsPixel,
                                  bool tc_pls_triplets) const {
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      unsigned int nPixels = segmentsOccupancy.nSegments()[nLowerModules];
      for (unsigned int pixelArrayIndex = globalThreadIdx[2]; pixelArrayIndex < nPixels;
           pixelArrayIndex += gridThreadExtent[2]) {
        if ((tc_pls_triplets ? 0 : !segmentsPixel.isQuad()[pixelArrayIndex]) ||
            (segmentsPixel.isDup()[pixelArrayIndex]))
          continue;

        unsigned int trackCandidateIdx =
            alpaka::atomicAdd(acc, &cands.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
        if (trackCandidateIdx - cands.nTrackCandidatesT5() >=
            n_max_pixel_track_candidates)  // T5 TCs have already been added
        {
#ifdef WARNINGS
          printf("Track Candidate excess alert! Type = pLS");
#endif
          alpaka::atomicSub(acc, &cands.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
          break;

        } else {
          alpaka::atomicAdd(acc, &cands.nTrackCandidatespLS(), 1u, alpaka::hierarchy::Threads{});
          addpLSTrackCandidateToMemory(cands,
                                       pixelArrayIndex,
                                       trackCandidateIdx,
                                       segmentsPixel.pLSHitsIdxs()[pixelArrayIndex],
                                       segmentsPixel.seedIdx()[pixelArrayIndex]);
        }
      }
    }
  };

  struct AddpT5asTrackCandidateInGPU {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  uint16_t nLowerModules,
                                  PixelQuintuplets pixelQuintupletsInGPU,
                                  TrackCandidates cands,
                                  SegmentsPixelConst segmentsPixel,
                                  ObjectOccupancyConst objectOccupancy) const {
      // implementation is 1D with a single block
      static_assert(std::is_same_v<TAcc, ALPAKA_ACCELERATOR_NAMESPACE::Acc1D>, "Should be Acc1D");
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      int nPixelQuintuplets = *pixelQuintupletsInGPU.nPixelQuintuplets;
      unsigned int pLS_offset = objectOccupancy.segmentModuleIndices()[nLowerModules];
      for (int pixelQuintupletIndex = globalThreadIdx[0]; pixelQuintupletIndex < nPixelQuintuplets;
           pixelQuintupletIndex += gridThreadExtent[0]) {
        if (pixelQuintupletsInGPU.isDup[pixelQuintupletIndex])
          continue;

        unsigned int trackCandidateIdx =
            alpaka::atomicAdd(acc, &cands.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
        if (trackCandidateIdx >= n_max_pixel_track_candidates)  // No other TCs have been added yet
        {
#ifdef WARNINGS
          printf("Track Candidate excess alert! Type = pT5");
#endif
          alpaka::atomicSub(acc, &cands.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
          break;

        } else {
          alpaka::atomicAdd(acc, &cands.nTrackCandidatespT5(), 1u, alpaka::hierarchy::Threads{});

          float radius = 0.5f * (__H2F(pixelQuintupletsInGPU.pixelRadius[pixelQuintupletIndex]) +
                                 __H2F(pixelQuintupletsInGPU.quintupletRadius[pixelQuintupletIndex]));
          unsigned int pT5PixelIndex = pixelQuintupletsInGPU.pixelIndices[pixelQuintupletIndex];
          addTrackCandidateToMemory(
              cands,
              7 /*track candidate type pT5=7*/,
              pT5PixelIndex,
              pixelQuintupletsInGPU.T5Indices[pixelQuintupletIndex],
              &pixelQuintupletsInGPU.logicalLayers[Params_pT5::kLayers * pixelQuintupletIndex],
              &pixelQuintupletsInGPU.lowerModuleIndices[Params_pT5::kLayers * pixelQuintupletIndex],
              &pixelQuintupletsInGPU.hitIndices[Params_pT5::kHits * pixelQuintupletIndex],
              segmentsPixel.seedIdx()[pT5PixelIndex - pLS_offset],
              __H2F(pixelQuintupletsInGPU.centerX[pixelQuintupletIndex]),
              __H2F(pixelQuintupletsInGPU.centerY[pixelQuintupletIndex]),
              radius,
              trackCandidateIdx,
              pixelQuintupletIndex);
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(lst::TrackCandidatesDeviceCollection, lst::TrackCandidatesHostCollection);

#endif

#ifndef RecoTracker_LSTCore_src_alpaka_PT2_h
#define RecoTracker_LSTCore_src_alpaka_PT2_h

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesSoA.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsSoA.h"
#include "RecoTracker/LSTCore/interface/SegmentsSoA.h"

namespace lst {
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addPT2ToMemory(MiniDoubletsConst mds,
                                                     SegmentsConst segments,
                                                     PT2s pT2s,
                                                     unsigned int pixelSegmentIndex,
                                                     unsigned int segmentIndex,
                                                     float pixelRadius,
                                                     float centerX,
                                                     float centerY,
                                                     float rzChiSquared,
                                                     unsigned int pT2Index,
                                                     float pt,
                                                     float eta,
                                                     float phi,
                                                     float eta_pix,
                                                     float phi_pix) {
    pT2s.pixelSegmentIndices()[pT2Index] = pixelSegmentIndex;
    pT2s.segmentIndices()[pT2Index] = segmentIndex;
    pT2s.pixelRadius()[pT2Index] = __F2H(pixelRadius);
    pT2s.pt()[pT2Index] = __F2H(pt);
    pT2s.eta()[pT2Index] = __F2H(eta);
    pT2s.phi()[pT2Index] = __F2H(phi);
    pT2s.eta_pix()[pT2Index] = __F2H(eta_pix);
    pT2s.phi_pix()[pT2Index] = __F2H(phi_pix);
    pT2s.isDup()[pT2Index] = false;

    pT2s.centerX()[pT2Index] = __F2H(centerX);
    pT2s.centerY()[pT2Index] = __F2H(centerY);
    pT2s.logicalLayers()[pT2Index][0] = 0;
    pT2s.logicalLayers()[pT2Index][1] = 0;
    pT2s.logicalLayers()[pT2Index][2] = segments.logicalLayers()[segmentIndex][0];
    pT2s.logicalLayers()[pT2Index][3] = segments.logicalLayers()[segmentIndex][1];

    pT2s.lowerModuleIndices()[pT2Index][0] = segments.innerLowerModuleIndices()[pixelSegmentIndex];
    pT2s.lowerModuleIndices()[pT2Index][1] = segments.outerLowerModuleIndices()[pixelSegmentIndex];
    pT2s.lowerModuleIndices()[pT2Index][2] = segments.innerLowerModuleIndices()[segmentIndex];
    pT2s.lowerModuleIndices()[pT2Index][3] = segments.outerLowerModuleIndices()[segmentIndex];

    unsigned int pixelInnerMD = segments.mdIndices()[pixelSegmentIndex][0];
    unsigned int pixelOuterMD = segments.mdIndices()[pixelSegmentIndex][1];

    pT2s.hitIndices()[pT2Index][0] = mds.anchorHitIndices()[pixelInnerMD];
    pT2s.hitIndices()[pT2Index][1] = mds.outerHitIndices()[pixelInnerMD];
    pT2s.hitIndices()[pT2Index][2] = mds.anchorHitIndices()[pixelOuterMD];
    pT2s.hitIndices()[pT2Index][3] = mds.outerHitIndices()[pixelOuterMD];

    pT2s.hitIndices()[pT2Index][4] = segments.innerMiniDoubletAnchorHitIndices()[segmentIndex];
    pT2s.hitIndices()[pT2Index][5] = segments.innerMiniDoubletOuterHitIndices()[segmentIndex];
    pT2s.hitIndices()[pT2Index][6] = segments.outerMiniDoubletAnchorHitIndices()[segmentIndex];
    pT2s.hitIndices()[pT2Index][7] = segments.outerMiniDoubletOuterHitIndices()[segmentIndex];
    pT2s.rzChiSquared()[pT2Index] = rzChiSquared;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runPT2DefaultAlgo(TAcc const& acc,
                                                        ModulesConst modules,
                                                        ObjectRangesConst ranges,
                                                        MiniDoubletsConst mds,
                                                        SegmentsConst segments,
                                                        unsigned int pixelSegmentIndex,
                                                        unsigned int segmentIndex,
                                                        float& pixelRadius,
                                                        float& centerX,
                                                        float& centerY,
                                                        float& rzChiSquared,
                                                        const float ptCut,
                                                        bool runChiSquaredCuts = true) {
    //run pT4 compatibility between the pixel segment and inner segment, and between the pixel and outer segment of the triplet
    //    uint16_t pixelModuleIndex = segments.innerLowerModuleIndices()[pixelSegmentIndex];

    //    uint16_t lowerModuleIndex = segments.innerLowerModuleIndices()[segmentIndex];
    //    uint16_t upperModuleIndex = segments.outerLowerModuleIndices()[segmentIndex];

    rzChiSquared = -1;
    centerX = 0;
    centerY = 0;
    return true;
  };

  struct CreatePT2sFromMap {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  ModulesConst modules,
                                  ModulesPixelConst modulesPixel,
                                  ObjectRangesConst ranges,
                                  MiniDoubletsConst mds,
                                  SegmentsConst segments,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  SegmentsPixel segmentsPixel,
                                  PT2s pT2s,
                                  unsigned int* connectedPixelSize,
                                  unsigned int* connectedPixelIndex,
                                  unsigned int nPixelSegments,
                                  const float ptCut) const {
      auto const globalBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
      auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
      auto const gridBlockExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc);
      auto const gridThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

      for (unsigned int i_pLS = globalThreadIdx[1]; i_pLS < nPixelSegments; i_pLS += gridThreadExtent[1]) {
        auto iLSModule_max = connectedPixelIndex[i_pLS] + connectedPixelSize[i_pLS];

        for (unsigned int iLSModule = connectedPixelIndex[i_pLS] + globalBlockIdx[0]; iLSModule < iLSModule_max;
             iLSModule += gridBlockExtent[0]) {
          uint16_t segmentLowerModuleIndex =
              modulesPixel.connectedPixels()
                  [iLSModule];  //connected pixels will have the appropriate lower module index by default!
#ifdef WARNINGS
          if (segmentLowerModuleIndex >= modules.nLowerModules()) {
            printf("segmentLowerModuleIndex %d >= modules.nLowerModules() %d \n",
                   segmentLowerModuleIndex,
                   modules.nLowerModules());
            continue;  //sanity check
          }
#endif
          //Removes 2S-2S :FIXME: filter these out in the pixel map
          if (modules.moduleType()[segmentLowerModuleIndex] == lst::TwoS)
            continue;

          uint16_t pixelModuleIndex = modules.nLowerModules();
          unsigned int nOuterSegments = segmentsOccupancy.nSegments()[segmentLowerModuleIndex];
          if (nOuterSegments == 0)
            continue;

          unsigned int pixelSegmentIndex = ranges.segmentModuleIndices()[pixelModuleIndex] + i_pLS;

          if (segmentsPixel.isDup()[i_pLS])
            continue;
          if (segmentsPixel.partOfPT5()[i_pLS])
            continue;  //don't make pT3s for those pixels that are part of pT5
          if (segmentsPixel.partOfPT3()[i_pLS])
            continue;

          short layer2_adjustment;
          if (modules.layers()[segmentLowerModuleIndex] == 1) {
            layer2_adjustment = 1;
          }  //get upper segment to be in second layer
          else if (modules.layers()[segmentLowerModuleIndex] == 2) {
            layer2_adjustment = 0;
          }  // get lower segment to be in second layer
          else {
            continue;
          }

          //fetch the segment
          for (unsigned int outerSegmentArrayIndex = globalThreadIdx[2]; outerSegmentArrayIndex < nOuterSegments;
               outerSegmentArrayIndex += gridThreadExtent[2]) {
            unsigned int outerSegmentIndex =
                ranges.segmentModuleIndices()[segmentLowerModuleIndex] + outerSegmentArrayIndex;
            if (modules.moduleType()[segments.outerLowerModuleIndices()[outerSegmentIndex]] == lst::TwoS)
              continue;  //REMOVES PS-2S

            if (segmentsPixel.partOfPT5()[outerSegmentIndex])
              continue;  //don't create pT2s for T2s accounted in pT5s

            if (segmentsPixel.partOfPT3()[outerSegmentIndex])
              continue;  //don't create pT2s for T2s accounted in pT3s

            float pixelRadius, rzChiSquared, centerX, centerY;
            bool success = runPT2DefaultAlgo(acc,
                                             modules,
                                             ranges,
                                             mds,
                                             segments,
                                             pixelSegmentIndex,
                                             outerSegmentIndex,
                                             pixelRadius,
                                             centerX,
                                             centerY,
                                             rzChiSquared,
                                             ptCut);

            if (success) {
              float phi = mds.anchorPhi()[segments.mdIndices()[outerSegmentIndex][layer2_adjustment]];
              float eta = mds.anchorEta()[segments.mdIndices()[outerSegmentIndex][layer2_adjustment]];
              float eta_pix = segmentsPixel.eta()[i_pLS];
              float phi_pix = segmentsPixel.phi()[i_pLS];
              float pt = segmentsPixel.ptIn()[i_pLS];
              unsigned int totOccupancyPT2s = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &pT2s.totOccupancyPT2s(), 1u);
              if (totOccupancyPT2s >= n_max_pt2s) {
#ifdef WARNINGS
                printf("Pixel Triplet excess alert!\n");
#endif
              } else {
                unsigned int pT2Index = alpaka::atomicOp<alpaka::AtomicAdd>(acc, &pT2s.nPT2s(), 1u);
                addPT2ToMemory(mds,
                               segments,
                               pT2s,
                               pixelSegmentIndex,
                               outerSegmentIndex,
                               pixelRadius,
                               centerX,
                               centerY,
                               rzChiSquared,
                               pT2Index,
                               pt,
                               eta,
                               phi,
                               eta_pix,
                               phi_pix);
              }
            }
          }  // for outerTripletArrayIndex
        }  // for iLSModule < iLSModule_max
      }  // for i_pLS
    }
  };

}  // namespace lst
#endif
#ifndef RecoTracker_LSTCore_src_alpaka_PT2_h
#define RecoTracker_LSTCore_src_alpaka_PT2_h

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/Module.h"

#include "Segment.h"
#include "MiniDoublet.h"
#include "Hit.h"
#include "ObjectRanges.h"

namespace lst {
  // One pixel segment, one outer tracker triplet!
  struct PT2s {
    unsigned int* pixelSegmentIndices;
    unsigned int* segmentIndices;
    unsigned int* nPT2s;
    unsigned int* totOccupancyPT2s;

    float* rzChiSquared;

    FPX* pixelRadius;
    FPX* pt;
    FPX* eta;
    FPX* phi;
    FPX* eta_pix;
    FPX* phi_pix;
    bool* isDup;
    bool* partOfPT3;
    bool* partOfPT5;

    uint8_t* logicalLayers;
    unsigned int* hitIndices;
    uint16_t* lowerModuleIndices;
    FPX* centerX;
    FPX* centerY;

    template <typename TBuff>
    void setData(TBuff& buf) {
      pixelSegmentIndices = alpaka::getPtrNative(buf.pixelSegmentIndices_buf);
      segmentIndices = alpaka::getPtrNative(buf.segmentIndices_buf);
      nPT2s = alpaka::getPtrNative(buf.nPT2s_buf);
      totOccupancyPT2s = alpaka::getPtrNative(buf.totOccupancyPT2s_buf);
      pixelRadius = alpaka::getPtrNative(buf.pixelRadius_buf);
      pt = alpaka::getPtrNative(buf.pt_buf);
      eta = alpaka::getPtrNative(buf.eta_buf);
      phi = alpaka::getPtrNative(buf.phi_buf);
      eta_pix = alpaka::getPtrNative(buf.eta_pix_buf);
      phi_pix = alpaka::getPtrNative(buf.phi_pix_buf);
      isDup = alpaka::getPtrNative(buf.isDup_buf);
      partOfPT3 = alpaka::getPtrNative(buf.partOfPT3_buf);
      partOfPT5 = alpaka::getPtrNative(buf.partOfPT5_buf);
      logicalLayers = alpaka::getPtrNative(buf.logicalLayers_buf);
      hitIndices = alpaka::getPtrNative(buf.hitIndices_buf);
      lowerModuleIndices = alpaka::getPtrNative(buf.lowerModuleIndices_buf);
      centerX = alpaka::getPtrNative(buf.centerX_buf);
      centerY = alpaka::getPtrNative(buf.centerY_buf);
      rzChiSquared = alpaka::getPtrNative(buf.rzChiSquared_buf);
    }
  };

  template <typename TDev>
  struct PT2sBuffer {
    Buf<TDev, unsigned int> pixelSegmentIndices_buf;
    Buf<TDev, unsigned int> segmentIndices_buf;
    Buf<TDev, unsigned int> nPT2s_buf;
    Buf<TDev, unsigned int> totOccupancyPT2s_buf;
    Buf<TDev, FPX> pixelRadius_buf;
    Buf<TDev, FPX> pt_buf;
    Buf<TDev, FPX> eta_buf;
    Buf<TDev, FPX> phi_buf;
    Buf<TDev, FPX> eta_pix_buf;
    Buf<TDev, FPX> phi_pix_buf;
    Buf<TDev, bool> isDup_buf;
    Buf<TDev, bool> partOfPT3_buf;
    Buf<TDev, bool> partOfPT5_buf;
    Buf<TDev, uint8_t> logicalLayers_buf;
    Buf<TDev, unsigned int> hitIndices_buf;
    Buf<TDev, uint16_t> lowerModuleIndices_buf;
    Buf<TDev, FPX> centerX_buf;
    Buf<TDev, FPX> centerY_buf;
    Buf<TDev, float> pixelRadiusError_buf;
    Buf<TDev, float> rzChiSquared_buf;

    PT2s data_;

    template <typename TQueue, typename TDevAcc>
    PT2sBuffer(unsigned int maxPT2s, TDevAcc const& devAccIn, TQueue& queue)
        : pixelSegmentIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPT2s, queue)),
          segmentIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPT2s, queue)),
          nPT2s_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          totOccupancyPT2s_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          pixelRadius_buf(allocBufWrapper<FPX>(devAccIn, maxPT2s, queue)),
          pt_buf(allocBufWrapper<FPX>(devAccIn, maxPT2s, queue)),
          eta_buf(allocBufWrapper<FPX>(devAccIn, maxPT2s, queue)),
          phi_buf(allocBufWrapper<FPX>(devAccIn, maxPT2s, queue)),
          eta_pix_buf(allocBufWrapper<FPX>(devAccIn, maxPT2s, queue)),
          phi_pix_buf(allocBufWrapper<FPX>(devAccIn, maxPT2s, queue)),
          isDup_buf(allocBufWrapper<bool>(devAccIn, maxPT2s, queue)),
          partOfPT3_buf(allocBufWrapper<bool>(devAccIn, maxPT2s, queue)),
          partOfPT5_buf(allocBufWrapper<bool>(devAccIn, maxPT2s, queue)),
          logicalLayers_buf(allocBufWrapper<uint8_t>(devAccIn, maxPT2s * Params_pT2::kLayers, queue)),
          hitIndices_buf(allocBufWrapper<unsigned int>(devAccIn, maxPT2s * Params_pT2::kHits, queue)),
          lowerModuleIndices_buf(allocBufWrapper<uint16_t>(devAccIn, maxPT2s * Params_pT2::kLayers, queue)),
          centerX_buf(allocBufWrapper<FPX>(devAccIn, maxPT2s, queue)),
          centerY_buf(allocBufWrapper<FPX>(devAccIn, maxPT2s, queue)),
          pixelRadiusError_buf(allocBufWrapper<float>(devAccIn, maxPT2s, queue)),
          rzChiSquared_buf(allocBufWrapper<float>(devAccIn, maxPT2s, queue)) {
      alpaka::memset(queue, nPT2s_buf, 0u);
      alpaka::memset(queue, totOccupancyPT2s_buf, 0u);
      alpaka::memset(queue, partOfPT3_buf, false);
      alpaka::memset(queue, partOfPT5_buf, false);
      alpaka::wait(queue);
    }

    inline PT2s const* data() const { return &data_; }
    inline void setData(PT2sBuffer& buf) { data_.setData(buf); }
  };

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addPT2ToMemory(lst::MiniDoublets const& mdsInGPU,
                                                              lst::Segments const& segmentsInGPU,
                                                              lst::PT2s& PT2sInGPU,
                                                              unsigned int pixelSegmentIndex,
                                                              unsigned int segmentIndex,
                                                              float pixelRadius,
                                                              float centerX,
                                                              float centerY,
                                                              float rzChiSquared,
                                                              unsigned int PT2Index,
                                                              float pt,
                                                              float eta,
                                                              float phi,
                                                              float eta_pix,
                                                              float phi_pix) {
    PT2sInGPU.pixelSegmentIndices[PT2Index] = pixelSegmentIndex;
    PT2sInGPU.segmentIndices[PT2Index] = segmentIndex;
    PT2sInGPU.pixelRadius[PT2Index] = __F2H(pixelRadius);
    PT2sInGPU.pt[PT2Index] = __F2H(pt);
    PT2sInGPU.eta[PT2Index] = __F2H(eta);
    PT2sInGPU.phi[PT2Index] = __F2H(phi);
    PT2sInGPU.eta_pix[PT2Index] = __F2H(eta_pix);
    PT2sInGPU.phi_pix[PT2Index] = __F2H(phi_pix);
    PT2sInGPU.isDup[PT2Index] = false;

    PT2sInGPU.centerX[PT2Index] = __F2H(centerX);
    PT2sInGPU.centerY[PT2Index] = __F2H(centerY);
    PT2sInGPU.logicalLayers[Params_pT2::kLayers * PT2Index] = 0;
    PT2sInGPU.logicalLayers[Params_pT2::kLayers * PT2Index + 1] = 0;
    PT2sInGPU.logicalLayers[Params_pT2::kLayers * PT2Index + 2] =
        segmentsInGPU.logicalLayers[segmentIndex * Params_LS::kLayers];
    PT2sInGPU.logicalLayers[Params_pT2::kLayers * PT2Index + 3] =
        segmentsInGPU.logicalLayers[segmentIndex * Params_LS::kLayers + 1];

    PT2sInGPU.lowerModuleIndices[Params_pT2::kLayers * PT2Index] =
        segmentsInGPU.innerLowerModuleIndices[pixelSegmentIndex];
    PT2sInGPU.lowerModuleIndices[Params_pT2::kLayers * PT2Index + 1] =
        segmentsInGPU.outerLowerModuleIndices[pixelSegmentIndex];
    PT2sInGPU.lowerModuleIndices[Params_pT2::kLayers * PT2Index + 2] =
        segmentsInGPU.innerLowerModuleIndices[Params_LS::kLayers * segmentIndex];
    PT2sInGPU.lowerModuleIndices[Params_pT2::kLayers * PT2Index + 3] =
        segmentsInGPU.outerLowerModuleIndices[Params_LS::kLayers * segmentIndex + 1];


    unsigned int pixelInnerMD = segmentsInGPU.mdIndices[2 * pixelSegmentIndex];
    unsigned int pixelOuterMD = segmentsInGPU.mdIndices[2 * pixelSegmentIndex + 1];

    PT2sInGPU.hitIndices[Params_pT2::kHits * PT2Index] = mdsInGPU.anchorHitIndices[pixelInnerMD];
    PT2sInGPU.hitIndices[Params_pT2::kHits * PT2Index + 1] = mdsInGPU.outerHitIndices[pixelInnerMD];
    PT2sInGPU.hitIndices[Params_pT2::kHits * PT2Index + 2] = mdsInGPU.anchorHitIndices[pixelOuterMD];
    PT2sInGPU.hitIndices[Params_pT2::kHits * PT2Index + 3] = mdsInGPU.outerHitIndices[pixelOuterMD];

    PT2sInGPU.hitIndices[Params_pT2::kHits * PT2Index + 4] =
        segmentsInGPU.innerMiniDoubletAnchorHitIndices[segmentIndex];
    PT2sInGPU.hitIndices[Params_pT2::kHits * PT2Index + 5] =
        segmentsInGPU.innerMiniDoubletOuterHitIndices[segmentIndex];
    PT2sInGPU.hitIndices[Params_pT2::kHits * PT2Index + 6] =
        segmentsInGPU.outerMiniDoubletAnchorHitIndices[segmentIndex];
    PT2sInGPU.hitIndices[Params_pT2::kHits * PT2Index + 7] =
        segmentsInGPU.outerMiniDoubletOuterHitIndices[segmentIndex];
    PT2sInGPU.rzChiSquared[PT2Index] = rzChiSquared;
  };

  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool runPT2DefaultAlgo(TAcc const& acc,
                                                                 lst::Modules const& modulesInGPU,
                                                                 lst::ObjectRanges const& rangesInGPU,
                                                                 lst::MiniDoublets const& mdsInGPU,
                                                                 lst::Segments const& segmentsInGPU,
                                                                 unsigned int pixelSegmentIndex,
                                                                 unsigned int segmentIndex,
                                                                 float& pixelRadius,
                                                                 float& centerX,
                                                                 float& centerY,
                                                                 float& rzChiSquared,
                                                                 const float ptCut,
                                                                 bool runChiSquaredCuts = true) {
    //run pT4 compatibility between the pixel segment and inner segment, and between the pixel and outer segment of the triplet
//    uint16_t pixelModuleIndex = segmentsInGPU.innerLowerModuleIndices[pixelSegmentIndex];

//    uint16_t lowerModuleIndex = segmentsInGPU.innerLowerModuleIndices[segmentIndex];
//    uint16_t upperModuleIndex = segmentsInGPU.outerLowerModuleIndices[segmentIndex];

    rzChiSquared = -1;
    centerX = 0;
    centerY = 0;
    return true;
  };

  struct createPT2sInGPUFromMapv2 {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  lst::Modules modulesInGPU,
                                  lst::ObjectRanges rangesInGPU,
                                  lst::MiniDoublets mdsInGPU,
                                  lst::Segments segmentsInGPU,
                                  lst::PT2s PT2sInGPU,
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
              modulesInGPU
                  .connectedPixels[iLSModule];  //connected pixels will have the appropriate lower module index by default!
#ifdef WARNINGS
          if (tripletLowerModuleIndex >= *modulesInGPU.nLowerModules) {
            printf("tripletLowerModuleIndex %d >= modulesInGPU.nLowerModules %d \n",
                   tripletLowerModuleIndex,
                   *modulesInGPU.nLowerModules);
            continue;  //sanity check
          }
#endif
          //Removes 2S-2S :FIXME: filter these out in the pixel map
          if (modulesInGPU.moduleType[segmentLowerModuleIndex] == lst::TwoS)
            continue;

          uint16_t pixelModuleIndex = *modulesInGPU.nLowerModules;
          unsigned int nOuterSegments = segmentsInGPU.nSegments[segmentLowerModuleIndex];
          if (nOuterSegments == 0)
            continue;

          unsigned int pixelSegmentIndex = rangesInGPU.segmentModuleIndices[pixelModuleIndex] + i_pLS;

          if (segmentsInGPU.isDup[i_pLS])
            continue;
          if (segmentsInGPU.partOfPT5[i_pLS])
            continue;  //don't make pT3s for those pixels that are part of pT5
          if (segmentsInGPU.partOfPT3[i_pLS])
            continue; 

          short layer2_adjustment;
          if (modulesInGPU.layers[segmentLowerModuleIndex] == 1) {
            layer2_adjustment = 1;
          }  //get upper segment to be in second layer
          else if (modulesInGPU.layers[segmentLowerModuleIndex] == 2) {
            layer2_adjustment = 0;
          }  // get lower segment to be in second layer
          else {
            continue;
          }

          //fetch the segment
          for (unsigned int outerSegmentArrayIndex = globalThreadIdx[2]; outerSegmentArrayIndex < nOuterSegments;
               outerSegmentArrayIndex += gridThreadExtent[2]) {
            unsigned int outerSegmentIndex =
                rangesInGPU.segmentModuleIndices[segmentLowerModuleIndex] + outerSegmentArrayIndex;
            if (modulesInGPU.moduleType[segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex]] == lst::TwoS)
              continue;  //REMOVES PS-2S

            if (segmentsInGPU.partOfPT5[outerSegmentIndex])
              continue;  //don't create pT2s for T2s accounted in pT5s

            if (segmentsInGPU.partOfPT3[outerSegmentIndex])
              continue; //don't create pT2s for T2s accounted in pT3s

            float pixelRadius, rzChiSquared, centerX, centerY;
            bool success = runPT2DefaultAlgo(acc,
                                                      modulesInGPU,
                                                      rangesInGPU,
                                                      mdsInGPU,
                                                      segmentsInGPU,
                                                      pixelSegmentIndex,
                                                      outerSegmentIndex,
                                                      pixelRadius,
                                                      centerX,
                                                      centerY,
                                                      rzChiSquared,
                                                      ptCut);

            if (success) {
              float phi =
                  mdsInGPU.anchorPhi[segmentsInGPU.mdIndices[outerSegmentIndex +
                                                             layer2_adjustment]];
              float eta =
                  mdsInGPU.anchorEta[segmentsInGPU.mdIndices[outerSegmentIndex +
                                                             layer2_adjustment]];
              float eta_pix = segmentsInGPU.eta[i_pLS];
              float phi_pix = segmentsInGPU.phi[i_pLS];
              float pt = segmentsInGPU.ptIn[i_pLS];
              unsigned int totOccupancyPT2s =
                  alpaka::atomicOp<alpaka::AtomicAdd>(acc, PT2sInGPU.totOccupancyPT2s, 1u);
              if (totOccupancyPT2s >= n_max_pixel_segments) {
#ifdef WARNINGS
                printf("Pixel Triplet excess alert!\n");
#endif
              } else {
                unsigned int PT2Index =
                    alpaka::atomicOp<alpaka::AtomicAdd>(acc, PT2sInGPU.nPT2s, 1u);
                addPT2ToMemory(mdsInGPU,
                                        segmentsInGPU,
                                        PT2sInGPU,
                                        pixelSegmentIndex,
                                        outerSegmentIndex,
                                        pixelRadius,
                                        centerX,
                                        centerY,
                                        rzChiSquared,
                                        PT2Index,
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

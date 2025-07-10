#ifndef RecoTracker_LSTCore_src_alpaka_TrackCandidate_h
#define RecoTracker_LSTCore_src_alpaka_TrackCandidate_h

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

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
#include "RecoTracker/LSTCore/interface/PixelQuadrupletsSoA.h"

#include "NeuralNetwork.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addpLSTrackCandidateToMemory(TrackCandidates& cands,
                                                                   unsigned int trackletIndex,
                                                                   unsigned int trackCandidateIndex,
                                                                   const Params_pLS::ArrayUxHits& hitIndices,
                                                                   int pixelSeedIndex) {
    cands.trackCandidateType()[trackCandidateIndex] = LSTObjType::pLS;
    cands.directObjectIndices()[trackCandidateIndex] = trackletIndex;
    cands.pixelSeedIndex()[trackCandidateIndex] = pixelSeedIndex;

    cands.objectIndices()[trackCandidateIndex][0] = trackletIndex;
    cands.objectIndices()[trackCandidateIndex][1] = trackletIndex;

    // Order explanation in https://github.com/SegmentLinking/TrackLooper/issues/267
    cands.hitIndices()[trackCandidateIndex][0] = hitIndices[0];
    cands.hitIndices()[trackCandidateIndex][1] = hitIndices[2];
    cands.hitIndices()[trackCandidateIndex][2] = hitIndices[1];
    cands.hitIndices()[trackCandidateIndex][3] = hitIndices[3];
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addTrackCandidateToMemory(TrackCandidates& cands,
                                                                short trackCandidateType,
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
    cands.trackCandidateType()[trackCandidateIndex] = trackCandidateType;
    cands.directObjectIndices()[trackCandidateIndex] = directObjectIndex;
    cands.pixelSeedIndex()[trackCandidateIndex] = pixelSeedIndex;

    cands.objectIndices()[trackCandidateIndex][0] = innerTrackletIndex;
    cands.objectIndices()[trackCandidateIndex][1] = outerTrackletIndex;

    size_t limits = trackCandidateType == LSTObjType::pT5 ? Params_pT5::kLayers : (trackCandidateType == LSTObjType::pT4 ? Params_pT4::kLayers : Params_pT3::kLayers);

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

  ALPAKA_FN_ACC ALPAKA_FN_INLINE void addT4TrackCandidateToMemory(TrackCandidates& cands,
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

    //send the starting pointer to the logicalLayer and hitIndices
    for (size_t i = 0; i < Params_T4::kLayers; i++) {
      cands.logicalLayers()[trackCandidateIndex][i] = logicalLayerIndices[i];
      cands.lowerModuleIndices()[trackCandidateIndex][i] = lowerModuleIndices[i];
    }
    for (size_t i = 0; i < 2 * Params_T4::kLayers; i++) {
      cands.hitIndices()[trackCandidateIndex][i] = hitIndices[i];
    }
    cands.centerX()[trackCandidateIndex] = __F2H(centerX);
    cands.centerY()[trackCandidateIndex] = __F2H(centerY);
    cands.radius()[trackCandidateIndex] = __F2H(radius);
  };

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
                                  PixelQuintupletsConst pixelQuintuplets
#ifdef USE_pT4
                                  ,PixelQuadrupletsConst pixelQuadruplets
#endif
                                ) const {
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
#ifdef USE_pT4
        unsigned int nPixelQuadruplets = pixelQuadruplets.nPixelQuadruplets();
        for (unsigned int pixelQuadrupletIndex : cms::alpakatools::uniform_elements_x(acc, nPixelQuadruplets)) {
          unsigned int pLS_jx = pixelQuadruplets.pixelSegmentIndices()[pixelQuadrupletIndex];
          float eta2 = pixelSeeds.eta()[pLS_jx - prefix];
          float phi2 = pixelSeeds.phi()[pLS_jx - prefix];
          float dEta = alpaka::math::abs(acc, (eta1 - eta2));
          float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);

          float dR2 = dEta * dEta + dPhi * dPhi;
          if (dR2 < 1e-5f)
            pixelTriplets.isDup()[pixelTripletIndex] = true;
        }
#endif
        
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
                                  ObjectRangesConst ranges
#ifdef USE_pT4                                  
                                  ,PixelQuadrupletsConst pixelQuadruplets
#endif                                
                                ) const {
      for (int lowmod : cms::alpakatools::uniform_elements_z(acc, modules.nLowerModules())) {
        if (ranges.quintupletModuleIndices()[lowmod] == -1)
          continue;

        unsigned int nQuints = quintupletsOccupancy.nQuintuplets()[lowmod];
        for (unsigned int iOff : cms::alpakatools::uniform_elements_y(acc, nQuints)) {
          unsigned int iT5 = ranges.quintupletModuleIndices()[lowmod] + iOff;

          // skip already-dup or already in pT5
          if (quintuplets.isDup()[iT5] || quintuplets.partOfPT5()[iT5]) 
            continue;

          float eta1 = __H2F(quintuplets.eta()[iT5]);
          float phi1 = __H2F(quintuplets.phi()[iT5]);

          float iEmbedT5[Params_T5::kEmbed];
          CMS_UNROLL_LOOP for (unsigned k = 0; k < Params_T5::kEmbed; ++k) {
            iEmbedT5[k] = quintuplets.t5Embed()[iT5][k];
          }
#ifndef USE_pT4
          unsigned int loop_bound = pixelQuintuplets.nPixelQuintuplets() + pixelTriplets.nPixelTriplets();
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
#endif
#ifdef USE_pT4
          unsigned int loop_bound = pixelQuintuplets.nPixelQuintuplets() + pixelTriplets.nPixelTriplets()+ pixelQuadruplets.nPixelQuadruplets();
          for (unsigned int jx : cms::alpakatools::uniform_elements_x(acc, loop_bound)) {
            float eta2, phi2;

            if (jx < pixelQuintuplets.nPixelQuintuplets()) {
              eta2 = __H2F(pixelQuintuplets.eta()[jx]);
              phi2 = __H2F(pixelQuintuplets.phi()[jx]);
            } else if (jx < pixelQuintuplets.nPixelQuintuplets() + pixelQuadruplets.nPixelQuadruplets()){
              eta2 = __H2F(pixelQuadruplets.eta()[jx - pixelQuintuplets.nPixelQuintuplets()]);
              phi2 = __H2F(pixelQuadruplets.phi()[jx - pixelQuintuplets.nPixelQuintuplets()]);
            } else {
              eta2 = __H2F(pixelTriplets.eta()[jx - (pixelQuintuplets.nPixelQuintuplets() + pixelQuadruplets.nPixelQuadruplets())]);
              phi2 = __H2F(pixelTriplets.phi()[jx - (pixelQuintuplets.nPixelQuintuplets() + pixelQuadruplets.nPixelQuadruplets())]);
            }
#endif
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
                                  TrackCandidates cands,
                                  SegmentsConst segments,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  PixelSeedsConst pixelSeeds,
                                  PixelSegments pixelSegments,
                                  MiniDoubletsConst mds,
                                  HitsBaseConst hitsBase,
                                  QuintupletsConst quintuplets
#ifdef USE_T4
                                  , QuadrupletsConst quadruplets
#endif                                
                                ) const {
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

        unsigned int nTrackCandidates = cands.nTrackCandidates();
        for (unsigned int trackCandidateIndex : cms::alpakatools::uniform_elements_x(acc, nTrackCandidates)) {
          short type = cands.trackCandidateType()[trackCandidateIndex];
          unsigned int innerTrackletIdx = cands.objectIndices()[trackCandidateIndex][0];
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
          if (type == LSTObjType::pT4) {
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
#ifdef USE_T4          
          if (type == LSTObjType::T4) {
            unsigned int quadrupletIndex = innerTrackletIdx;  // T4 index
            float eta2 = __H2F(quadruplets.eta()[quadrupletIndex]);
            float phi2 = __H2F(quadruplets.phi()[quadrupletIndex]);
            float dEta = alpaka::math::abs(acc, eta1 - eta2);
            float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);

            float dR2 = dEta * dEta + dPhi * dPhi;
            if (dR2 < 5e-3f)
              pixelSegments.isDup()[pixelArrayIndex] = true;
          }
#endif
        }
      }
    }
  };

  struct CrossCleanT4 {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  ModulesConst modules,
                                  Quadruplets quadruplets,
                                  QuadrupletsOccupancyConst quadrupletsOccupancy,
#ifdef USE_pT4                                  
                                  PixelQuadrupletsConst pixelQuadruplets,
#endif                                 
                                  PixelQuintupletsConst pixelQuintuplets,
                                  PixelTripletsConst pixelTriplets,
                                  QuintupletsConst quintuplets,
                                  TrackCandidates cands,
                                  MiniDoubletsConst mds,
                                  SegmentsConst segments,
                                  TripletsConst triplets,
                                  ObjectRangesConst ranges) const {
      for (int lowmod : cms::alpakatools::uniform_elements_z(acc, modules.nLowerModules())) {
        if (ranges.quadrupletModuleIndices()[lowmod] == -1)
          continue;                              

        unsigned int nQuads = quadrupletsOccupancy.nQuadruplets()[lowmod];
        for (unsigned int iOff : cms::alpakatools::uniform_elements_y(acc, nQuads)) {
          unsigned int iT4 = ranges.quadrupletModuleIndices()[lowmod] + iOff;

          // skip already-dup or already in pT4
          if (quadruplets.isDup()[iT4] or quadruplets.partOfPT4()[iT4])
            continue;

          // // Cross cleaning step
          float eta1 = __H2F(quadruplets.eta()[iT4]); 
          float phi1 = __H2F(quadruplets.phi()[iT4]);

          unsigned int nTrackCandidates = cands.nTrackCandidates();
          for (unsigned int trackCandidateIndex : cms::alpakatools::uniform_elements_x(acc, nTrackCandidates)) {
            short type = cands.trackCandidateType()[trackCandidateIndex];
            unsigned int outerTrackletIdx = cands.objectIndices()[trackCandidateIndex][1];
            if (type == LSTObjType::T5) {
              unsigned int quintupletIndex = outerTrackletIdx;  // T5 index
              uint16_t t5_lowerModIdx1 = quintuplets.lowerModuleIndices()[quintupletIndex][0];
              short layer2_adjustment =1;
              short layer3_adjustment;
              int layer = modules.layers()[t5_lowerModIdx1];
              if (layer == 1) {
                layer3_adjustment = 1;
              }  // third layer
              else {
                layer3_adjustment = 0;  //third layer
              }
              int innerTripletIndex = quintuplets.tripletIndices()[quintupletIndex][0];
              float phi2 =
                      mds.anchorPhi()[segments.mdIndices()[triplets.segmentIndices()[innerTripletIndex][layer3_adjustment]][layer2_adjustment]]; //layer 3
              float eta2 =
                      mds.anchorEta()[segments.mdIndices()[triplets.segmentIndices()[innerTripletIndex][layer3_adjustment]][layer2_adjustment]]; //layer 3
              float dEta = alpaka::math::abs(acc, eta1 - eta2);
              float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);
  
              float dR2 = dEta * dEta + dPhi * dPhi;
              if (dR2 < 1e-3f) {
                quadruplets.isDup()[iT4] = true;
              }
              
            }
            if (type == LSTObjType::pT3) { 
              int pT3Index = outerTrackletIdx;
              uint16_t pT3_lowerModIdx1 = pixelTriplets.lowerModuleIndices()[pT3Index][0];
              short layer2_adjustment = 1;
              short layer3_adjustment;
              int layer = modules.layers()[pT3_lowerModIdx1];
              if (layer == 1) {
                layer3_adjustment = 1;
              }  // third layer
              else {
                layer3_adjustment = 0;  //third layer
              }
              int innerTripletIndex = pixelTriplets.tripletIndices()[pT3Index];
              float phi2 =
                      mds.anchorPhi()[segments.mdIndices()[triplets.segmentIndices()[innerTripletIndex][layer3_adjustment]][layer2_adjustment]]; //layer 3
              float eta2 =
                      mds.anchorEta()[segments.mdIndices()[triplets.segmentIndices()[innerTripletIndex][layer3_adjustment]][layer2_adjustment]]; //layer 3
              float dEta = alpaka::math::abs(acc, eta1 - eta2);
              float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);
  
              float dR2 = dEta * dEta + dPhi * dPhi;
              if (dR2 < 1e-3f)
                quadruplets.isDup()[iT4] = true;
            }
            if (type == LSTObjType::pT5) { 
              unsigned int quintupletIndex = outerTrackletIdx;  // T5 index
              uint16_t t5_lowerModIdx1 = quintuplets.lowerModuleIndices()[quintupletIndex][0];
              short layer2_adjustment =1;
              short layer3_adjustment;
              int layer = modules.layers()[t5_lowerModIdx1];
              if (layer == 1) {
                layer3_adjustment = 1;
              }  // third layer
              else {
                layer3_adjustment = 0;  //third layer
              }
              int innerTripletIndex = quintuplets.tripletIndices()[quintupletIndex][0];
              float phi2 =
                      mds.anchorPhi()[segments.mdIndices()[triplets.segmentIndices()[innerTripletIndex][layer3_adjustment]][layer2_adjustment]]; //layer 3
              float eta2 =
                      mds.anchorEta()[segments.mdIndices()[triplets.segmentIndices()[innerTripletIndex][layer3_adjustment]][layer2_adjustment]]; //layer 3
              float dEta = alpaka::math::abs(acc, eta1 - eta2);
              float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);
  
              float dR2 = dEta * dEta + dPhi * dPhi;
              if (dR2 < 1e-3f) {
                quadruplets.isDup()[iT4] = true;
              }
            }
  #ifdef USE_pT4          
            if (type == LSTObjType::pT4) {
              unsigned int pT4Index = cands.directObjectIndices()[trackCandidateIndex]; 
              float eta2 = __H2F(quadruplets.eta()[pixelQuadruplets.quadrupletIndices()[pT4Index]]); //use T4 index of pT4 to get eta, phi in layer 3
              float phi2 = __H2F(quadruplets.phi()[pixelQuadruplets.quadrupletIndices()[pT4Index]]);
              float dEta = alpaka::math::abs(acc, eta1 - eta2);
              float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);
  
              float dR2 = dEta * dEta + dPhi * dPhi;
              if (dR2 < 1e-3f)
                quadruplets.isDup()[iT4] = true;
            }
#endif            
          }
        }
      }
    }
  };

  struct CrossCleanpT4 {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  ModulesConst modules,
                                  ObjectRangesConst ranges,
                                  PixelQuadruplets pixelQuadruplets,
                                  PixelSeedsConst pixelSeeds,
                                  PixelQuintupletsConst pixelQuintuplets) const {
      unsigned int nPixelQuadruplets = pixelQuadruplets.nPixelQuadruplets();
      for (unsigned int pixelQuadrupletIndex : cms::alpakatools::uniform_elements_y(acc, nPixelQuadruplets)) {
        if (pixelQuadruplets.isDup()[pixelQuadrupletIndex])
          continue;

        // Cross cleaning step
        unsigned int pLS_ix = pixelQuadruplets.pixelSegmentIndices()[pixelQuadrupletIndex];

        int pixelModuleIndex = modules.nLowerModules();
        unsigned int prefix = ranges.segmentModuleIndices()[pixelModuleIndex];

        float eta1 = pixelSeeds.eta()[pLS_ix - prefix];
        float phi1 = pixelSeeds.phi()[pLS_ix - prefix];

        unsigned int nPixelQuintuplets = pixelQuintuplets.nPixelQuintuplets();
        for (unsigned int pixelQuintupletIndex : cms::alpakatools::uniform_elements_x(acc, nPixelQuintuplets)) {
          unsigned int pLS_jx = pixelQuintuplets.pixelSegmentIndices()[pixelQuintupletIndex];
          float eta2 = pixelSeeds.eta()[pLS_jx - prefix];
          float phi2 = pixelSeeds.phi()[pLS_jx - prefix];
          float dEta = alpaka::math::abs(acc, (eta1 - eta2));
          float dPhi = cms::alpakatools::deltaPhi(acc, phi1, phi2);

          float dR2 = dEta * dEta + dPhi * dPhi;
          if (dR2 < 1e-5f)
            pixelQuadruplets.isDup()[pixelQuadrupletIndex] = true;
        }
      }
    }
  };

  struct AddpT3asTrackCandidates {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  uint16_t nLowerModules,
                                  PixelTripletsConst pixelTriplets,
                                  TrackCandidates cands,
                                  PixelSeedsConst pixelSeeds,
                                  ObjectRangesConst ranges) const {
      // implementation is 1D with a single block
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      unsigned int nPixelTriplets = pixelTriplets.nPixelTriplets();
      unsigned int pLS_offset = ranges.segmentModuleIndices()[nLowerModules];
      for (unsigned int pixelTripletIndex : cms::alpakatools::uniform_elements(acc, nPixelTriplets)) {
        if ((pixelTriplets.isDup()[pixelTripletIndex])) {
          continue; }

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

          float radius = 0.5f * (__H2F(pixelTriplets.pixelRadius()[pixelTripletIndex]) +
                                 __H2F(pixelTriplets.tripletRadius()[pixelTripletIndex]));
          unsigned int pT3PixelIndex = pixelTriplets.pixelSegmentIndices()[pixelTripletIndex];
          addTrackCandidateToMemory(cands,
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
                                  TrackCandidates cands,
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
              alpaka::atomicAdd(acc, &cands.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
          if (trackCandidateIdx - cands.nTrackCandidatespT5() - cands.nTrackCandidatespT3() - cands.nTrackCandidatespT4() >=
              n_max_nonpixel_track_candidates)  // pT5, pT4, and pT3 TCs have been added, but not pLS TCs
          {
#ifdef WARNINGS
            printf("Track Candidate excess alert! Type = T5");
#endif
            alpaka::atomicSub(acc, &cands.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
            break;
          } else {
            alpaka::atomicAdd(acc, &cands.nTrackCandidatesT5(), 1u, alpaka::hierarchy::Threads{});
            addTrackCandidateToMemory(cands,
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
                                  TrackCandidates cands,
                                  SegmentsOccupancyConst segmentsOccupancy,
                                  PixelSeedsConst pixelSeeds,
                                  PixelSegmentsConst pixelSegments,
                                  bool tc_pls_triplets) const {
      unsigned int nPixels = segmentsOccupancy.nSegments()[nLowerModules];
      for (unsigned int pixelArrayIndex : cms::alpakatools::uniform_elements(acc, nPixels)) {
        if ((tc_pls_triplets ? 0 : !pixelSeeds.isQuad()[pixelArrayIndex]) || (pixelSegments.isDup()[pixelArrayIndex]))
          continue;

        unsigned int trackCandidateIdx =
            alpaka::atomicAdd(acc, &cands.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
        if (trackCandidateIdx - cands.nTrackCandidatesT5() - cands.nTrackCandidatesT4() >=
            n_max_pixel_track_candidates)  // T5, T4 TCs have already been added
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
                                       pixelSegments.pLSHitsIdxs()[pixelArrayIndex],
                                       pixelSeeds.seedIdx()[pixelArrayIndex]);
        }
      }
    }
  };

  struct AddpT5asTrackCandidate {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  uint16_t nLowerModules,
                                  PixelQuintupletsConst pixelQuintuplets,
                                  TrackCandidates cands,
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

          float radius = 0.5f * (__H2F(pixelQuintuplets.pixelRadius()[pixelQuintupletIndex]) +
                                 __H2F(pixelQuintuplets.quintupletRadius()[pixelQuintupletIndex]));
          unsigned int pT5PixelIndex = pixelQuintuplets.pixelSegmentIndices()[pixelQuintupletIndex];
          addTrackCandidateToMemory(cands,
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

  struct AddT4asTrackCandidate {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  uint16_t nLowerModules,
                                  Quadruplets quadruplets,
                                  QuadrupletsOccupancyConst quadrupletsOccupancy,
                                  TripletsConst triplets,
                                  TrackCandidates cands,
                                  ObjectRangesConst ranges) const {
                              
      for (int idx : cms::alpakatools::uniform_elements_y(acc, nLowerModules)) {
        if (ranges.quadrupletModuleIndices()[idx] == -1)
          continue;

        unsigned int nQuads = quadrupletsOccupancy.nQuadruplets()[idx];
        for (unsigned int jdx : cms::alpakatools::uniform_elements_x(acc, nQuads)) {
          unsigned int quadrupletIndex = ranges.quadrupletModuleIndices()[idx] + jdx;

          if (!(quadruplets.tightCutFlag()[quadrupletIndex]))
            continue;
          if (!(quadruplets.tightDNNFlag()[quadrupletIndex]))
            continue;
          if (quadruplets.isDup()[quadrupletIndex] or quadruplets.partOfPT4()[quadrupletIndex])
            continue;

          unsigned int trackCandidateIdx =
              alpaka::atomicAdd(acc, &cands.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
          if (trackCandidateIdx - cands.nTrackCandidatespT5() - cands.nTrackCandidatespT4() - cands.nTrackCandidatespT3() 
              - cands.nTrackCandidatesT5() >= n_max_nonpixel_track_candidates)  // pT5, pT4, pT3, T5 TCs have been added, but not pLS TCs
          {
#ifdef WARNINGS
            printf("Track Candidate excess alert! Type = T4");
#endif
            alpaka::atomicSub(acc, &cands.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
            break;
          } else {
            alpaka::atomicAdd(acc, &cands.nTrackCandidatesT4(), 1u, alpaka::hierarchy::Threads{});
            unsigned int innerTripletIndex = quadruplets.tripletIndices()[quadrupletIndex][0];
            addT4TrackCandidateToMemory(cands,
                                      LSTObjType::T4,
                                      quadrupletIndex,
                                      quadrupletIndex,
                                      quadruplets.logicalLayers()[quadrupletIndex].data(),
                                      quadruplets.lowerModuleIndices()[quadrupletIndex].data(),
                                      quadruplets.hitIndices()[quadrupletIndex].data(),
                                      -1 /*no pixel seed index for T4s*/,
                                      triplets.centerX()[innerTripletIndex],
                                      triplets.centerY()[innerTripletIndex],
                                      quadruplets.innerRadius()[quadrupletIndex],
                                      trackCandidateIdx,
                                      quadrupletIndex);
            quadruplets.partOfTC()[quadrupletIndex] = true;
          }
        }
      }
    }
  };

  struct AddpT4asTrackCandidate {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  uint16_t nLowerModules,
                                  PixelQuadrupletsConst pixelQuadruplets,
                                  TrackCandidates cands,
                                  PixelSeedsConst pixelSeeds,
                                  ObjectRangesConst ranges,
                                  QuadrupletsConst quadruplets) const {
      // implementation is 1D with a single block
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));

      int nPixelQuadruplets = pixelQuadruplets.nPixelQuadruplets();
      unsigned int pLS_offset = ranges.segmentModuleIndices()[nLowerModules];
      for (int pixelQuadrupletIndex : cms::alpakatools::uniform_elements(acc, nPixelQuadruplets)) {
        if (pixelQuadruplets.isDup()[pixelQuadrupletIndex])
          continue;
        
        unsigned int trackCandidateIdx =
            alpaka::atomicAdd(acc, &cands.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
        if (trackCandidateIdx - cands.nTrackCandidatespT5() >= n_max_pixel_track_candidates)  
        {
#ifdef WARNINGS
          printf("Track Candidate excess alert! Type = pT4");
#endif
          alpaka::atomicSub(acc, &cands.nTrackCandidates(), 1u, alpaka::hierarchy::Threads{});
          break;

        } else {
          alpaka::atomicAdd(acc, &cands.nTrackCandidatespT4(), 1u, alpaka::hierarchy::Threads{});

          float radius = 0.5f * (__H2F(pixelQuadruplets.pixelRadius()[pixelQuadrupletIndex]) +
                                 __H2F(pixelQuadruplets.quadrupletRadius()[pixelQuadrupletIndex]));
          unsigned int pT4PixelIndex = pixelQuadruplets.pixelSegmentIndices()[pixelQuadrupletIndex];
          addTrackCandidateToMemory(
              cands,
              LSTObjType::pT4,
              pT4PixelIndex,
              pixelQuadruplets.quadrupletIndices()[pixelQuadrupletIndex],
              pixelQuadruplets.logicalLayers()[pixelQuadrupletIndex].data(),
              pixelQuadruplets.lowerModuleIndices()[pixelQuadrupletIndex].data(),
              pixelQuadruplets.hitIndices()[pixelQuadrupletIndex].data(),
              pixelSeeds.seedIdx()[pT4PixelIndex - pLS_offset],
              __H2F(pixelQuadruplets.centerX()[pixelQuadrupletIndex]),
              __H2F(pixelQuadruplets.centerY()[pixelQuadrupletIndex]),
              radius,
              trackCandidateIdx,
              pixelQuadrupletIndex);
        }
      }
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(lst::TrackCandidatesDeviceCollection, lst::TrackCandidatesHostCollection);

#endif

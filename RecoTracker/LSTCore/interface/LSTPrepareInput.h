#ifndef RecoTracker_LSTCore_interface_LSTPrepareInput_h
#define RecoTracker_LSTCore_interface_LSTPrepareInput_h

#include <algorithm>
#include <memory>
#include <Math/Vector3D.h>
#include <Math/VectorUtil.h>

#include "RecoTracker/LSTCore/interface/Common.h"
#include "RecoTracker/LSTCore/interface/LSTInputHostCollection.h"
#include "RecoTracker/LSTCore/interface/PixelSegmentParameters.h"

namespace lst {

  inline ROOT::Math::XYZVector calculateR3FromPCA(const ROOT::Math::XYZVector& p3, float dxy, float dz) {
    float vx, vy, vz;
    pixelSegmentPCAPosition(p3.x(), p3.y(), p3.z(), dxy, dz, vx, vy, vz);
    return {vx, vy, vz};
  }

  template <typename TQueue>
  inline LSTInputHostCollection prepareInput(std::vector<float> const& see_px,
                                             std::vector<float> const& see_py,
                                             std::vector<float> const& see_pz,
                                             std::vector<float> const& see_dxy,
                                             std::vector<float> const& see_dz,
                                             std::vector<float> const& see_ptErr,
                                             std::vector<float> const& see_etaErr,
                                             std::vector<float> const& see_stateTrajGlbX,
                                             std::vector<float> const& see_stateTrajGlbY,
                                             std::vector<float> const& see_stateTrajGlbZ,
                                             std::vector<float> const& see_stateTrajGlbPx,
                                             std::vector<float> const& see_stateTrajGlbPy,
                                             std::vector<float> const& see_stateTrajGlbPz,
                                             std::vector<int> const& see_q,
                                             std::vector<std::vector<int>> const& see_hitIdx,
                                             std::vector<std::vector<int>> const& see_hitType,
                                             std::vector<unsigned int> const& see_algo,
                                             std::vector<unsigned int> const& ph2_detId,
                                             std::vector<uint16_t> const& ph2_clustSize,
                                             std::vector<float> const& ph2_x,
                                             std::vector<float> const& ph2_y,
                                             std::vector<float> const& ph2_z,
                                             float const ptCut,
                                             TQueue const& queue) {
    std::vector<float> trkX;
    std::vector<float> trkY;
    std::vector<float> trkZ;
    std::vector<uint16_t> hitClustSize;
    std::vector<unsigned int> hitId;
    std::vector<unsigned int> hitIdxs;
    std::vector<unsigned int> firstHit_vec;
    std::vector<uint8_t> nHits_vec;
    std::vector<uint8_t> hitDetBits_vec;
    std::vector<float> deltaPhi_vec;
    std::vector<float> ptIn_vec;
    std::vector<float> ptErr_vec;
    std::vector<float> px_vec;
    std::vector<float> py_vec;
    std::vector<float> pz_vec;
    std::vector<float> eta_vec;
    std::vector<float> etaErr_vec;
    std::vector<float> phi_vec;
    std::vector<int> charge_vec;
    std::vector<unsigned int> seedIdx_vec;
    std::vector<int> superbin_vec;
    std::vector<PixelType> pixelType_vec;
    std::vector<char> isQuad_vec;

    const int hit_size = ph2_x.size();

    unsigned int count = 0;
    auto n_see = see_stateTrajGlbPx.size();
    px_vec.reserve(n_see);
    py_vec.reserve(n_see);
    pz_vec.reserve(n_see);
    firstHit_vec.reserve(n_see);
    nHits_vec.reserve(n_see);
    hitDetBits_vec.reserve(n_see);
    ptIn_vec.reserve(n_see);
    ptErr_vec.reserve(n_see);
    etaErr_vec.reserve(n_see);
    eta_vec.reserve(n_see);
    phi_vec.reserve(n_see);
    charge_vec.reserve(n_see);
    seedIdx_vec.reserve(n_see);
    deltaPhi_vec.reserve(n_see);
    trkX.reserve(4 * n_see);
    trkY.reserve(4 * n_see);
    trkZ.reserve(4 * n_see);
    hitClustSize.reserve(4 * n_see);
    hitId.reserve(4 * n_see);
    hitIdxs.reserve(hit_size + 4 * n_see);
    hitIdxs.resize(hit_size);

    std::iota(hitIdxs.begin(), hitIdxs.end(), 0);
    int nHitsOT = ph2_x.size();

    for (size_t iSeed = 0; iSeed < n_see; iSeed++) {
      // Only needed for standalone
      bool good_seed_type = see_algo.empty() || see_algo[iSeed] == 4 || see_algo[iSeed] == 22;
      if (!good_seed_type)
        continue;

      ROOT::Math::XYZVector p3LH(see_stateTrajGlbPx[iSeed], see_stateTrajGlbPy[iSeed], see_stateTrajGlbPz[iSeed]);
      float ptIn = p3LH.rho();
      float eta = p3LH.eta();
      float ptErr = see_ptErr[iSeed];

      if ((ptIn > ptCut - 2 * ptErr)) {
        ROOT::Math::XYZVector r3LH(see_stateTrajGlbX[iSeed], see_stateTrajGlbY[iSeed], see_stateTrajGlbZ[iSeed]);
        ROOT::Math::XYZVector p3PCA(see_px[iSeed], see_py[iSeed], see_pz[iSeed]);
        ROOT::Math::XYZVector r3PCA(calculateR3FromPCA(p3PCA, see_dxy[iSeed], see_dz[iSeed]));

        // The charge could be used directly in the line below
        float pixelSegmentDeltaPhiChange = ROOT::Math::VectorUtil::DeltaPhi(p3LH, r3LH);
        float etaErr = see_etaErr[iSeed];
        float px = p3LH.x();
        float py = p3LH.y();
        float pz = p3LH.z();

        int charge = see_q[iSeed];
        PixelType pixtype = pixelSegmentType(ptIn, ptErr, pixelSegmentDeltaPhiChange, ptCut);
        if (pixtype == PixelType::kInvalid)
          continue;

        firstHit_vec.push_back(hit_size + count);
        nHits_vec.push_back(see_hitIdx[iSeed].size());

        const unsigned int nHitsToSoA =
            std::min(static_cast<unsigned int>(see_hitIdx[iSeed].size()), kMaxPLSHitsInHitsSoA);
        assert(nHitsToSoA >= 3);  // need to rework the logic for less than 3 hits
        count += nHitsToSoA;

        auto const& hTypes = see_hitType[iSeed];
        auto constexpr intPixel = static_cast<int>(HitType::Pixel);
        auto const& hIdxs = see_hitIdx[iSeed];
        for (unsigned int iSH = 0; iSH < nHitsToSoA; iSH++) {
          auto iH = pixelSegmentHitOfSlot(iSH, nHitsToSoA, see_hitIdx[iSeed].size());
          hitId.push_back(hTypes[iH] == intPixel ? kPixelModuleId : ph2_detId[hIdxs[iH]]);
          hitClustSize.push_back(hTypes[iH] == intPixel ? 1 : ph2_clustSize[hIdxs[iH]]);
        }
        uint8_t hitDetBits = 0;
        uint8_t nToBits = std::min(kMaxPLSHitBitsInHitsSoA, static_cast<unsigned int>(see_hitIdx[iSeed].size()));
        for (int iSH = 0; iSH < nToBits; iSH++) {
          auto iH = pixelSegmentHitOfSlot(iSH, nToBits, see_hitIdx[iSeed].size());
          hitDetBits |= (hTypes[iH] != intPixel) << iSH;
        }
        hitDetBits_vec.push_back(hitDetBits);

        // eventually these trk[XYZ] should be moved to the PixelSeeds SoA
        float p3PCA_Eta = p3PCA.eta();
        float p3PCA_Phi = p3PCA.phi();
        const PixelSegmentKinematics kinematics{float(r3PCA.x()),
                                                float(r3PCA.y()),
                                                float(r3PCA.z()),
                                                float(p3PCA.rho()),
                                                p3PCA_Eta,
                                                p3PCA_Phi,
                                                float(r3LH.x()),
                                                float(r3LH.y()),
                                                float(r3LH.z()),
                                                see_dxy[iSeed],
                                                see_dz[iSeed]};
        for (unsigned int iSH = 0; iSH < nHitsToSoA; iSH++) {
          float x, y, z;
          pixelSegmentHitColumns(kinematics, iSH, x, y, z);
          trkX.push_back(x);
          trkY.push_back(y);
          trkZ.push_back(z);
        }
        assert(trkX.size() == count);

        px_vec.push_back(px);
        py_vec.push_back(py);
        pz_vec.push_back(pz);

        ptIn_vec.push_back(ptIn);
        ptErr_vec.push_back(ptErr);
        etaErr_vec.push_back(etaErr);
        eta_vec.push_back(eta);
        float phi = p3LH.phi();
        phi_vec.push_back(phi);
        charge_vec.push_back(charge);
        seedIdx_vec.push_back(iSeed);
        deltaPhi_vec.push_back(pixelSegmentDeltaPhiChange);

        hitIdxs.push_back(see_hitIdx[iSeed][0]);
        hitIdxs.push_back(see_hitIdx[iSeed][1]);
        hitIdxs.push_back(see_hitIdx[iSeed][2]);
        char isQuad = false;
        if (see_hitIdx[iSeed].size() > 3) {
          isQuad = true;
          hitIdxs.push_back(see_hitIdx[iSeed].back());
        }
        superbin_vec.push_back(pixelSegmentSuperbin(p3PCA_Eta, p3PCA_Phi, see_dz[iSeed]));
        pixelType_vec.push_back(pixtype);
        isQuad_vec.push_back(isQuad);
      }
    }

    // Build the SoAs
    int nHitsIT = trkX.size();
    int nPixelSeeds = ptIn_vec.size();
    if (static_cast<unsigned int>(nPixelSeeds) > n_max_pixel_segments_per_module) {
      nPixelSeeds = n_max_pixel_segments_per_module;
    }

    LSTInputHostCollection lstInputHC(queue, nHitsIT + nHitsOT, nPixelSeeds, nHitsIT);

    auto hits = lstInputHC.view().hits();
    hits.nHitsOT() = nHitsOT;
    std::copy_n(ph2_x.data(), nHitsOT, hits.xs().data());
    std::copy_n(ph2_y.data(), nHitsOT, hits.ys().data());
    std::copy_n(ph2_z.data(), nHitsOT, hits.zs().data());
    std::copy_n(ph2_detId.data(), nHitsOT, hits.detid().data());
    std::copy_n(ph2_clustSize.data(), nHitsOT, hits.clustsize().data());

    std::copy_n(trkX.data(), nHitsIT, hits.xs().data() + nHitsOT);
    std::copy_n(trkY.data(), nHitsIT, hits.ys().data() + nHitsOT);
    std::copy_n(trkZ.data(), nHitsIT, hits.zs().data() + nHitsOT);
    std::copy_n(hitId.data(), nHitsIT, hits.detid().data() + nHitsOT);
    std::copy_n(hitClustSize.data(), nHitsIT, hits.clustsize().data() + nHitsOT);

    // the OT part of hitIdxs is the identity (hitOrigIdx); only the pLS section is stored
    std::copy_n(hitIdxs.data() + nHitsOT, nHitsIT, lstInputHC.view().hitsIT().idxs().data());

    auto pixelSeeds = lstInputHC.view().pixelSeeds();
    std::copy_n(firstHit_vec.data(), nPixelSeeds, pixelSeeds.firstHit().data());
    std::copy_n(nHits_vec.data(), nPixelSeeds, pixelSeeds.nHits().data());
    std::copy_n(hitDetBits_vec.data(), nPixelSeeds, pixelSeeds.hitDetBits().data());
    std::copy_n(deltaPhi_vec.data(), nPixelSeeds, pixelSeeds.deltaPhi().data());
    std::copy_n(ptIn_vec.data(), nPixelSeeds, pixelSeeds.ptIn().data());
    std::copy_n(ptErr_vec.data(), nPixelSeeds, pixelSeeds.ptErr().data());
    std::copy_n(px_vec.data(), nPixelSeeds, pixelSeeds.px().data());
    std::copy_n(py_vec.data(), nPixelSeeds, pixelSeeds.py().data());
    std::copy_n(pz_vec.data(), nPixelSeeds, pixelSeeds.pz().data());
    std::copy_n(etaErr_vec.data(), nPixelSeeds, pixelSeeds.etaErr().data());
    std::copy_n(isQuad_vec.data(), nPixelSeeds, pixelSeeds.isQuad().data());
    std::copy_n(eta_vec.data(), nPixelSeeds, pixelSeeds.eta().data());
    std::copy_n(phi_vec.data(), nPixelSeeds, pixelSeeds.phi().data());
    std::copy_n(charge_vec.data(), nPixelSeeds, pixelSeeds.charge().data());
    std::copy_n(seedIdx_vec.data(), nPixelSeeds, pixelSeeds.seedIdx().data());
    std::copy_n(superbin_vec.data(), nPixelSeeds, pixelSeeds.superbin().data());
    std::copy_n(pixelType_vec.data(), nPixelSeeds, pixelSeeds.pixelType().data());

    return lstInputHC;
  }

}  // namespace lst

#endif

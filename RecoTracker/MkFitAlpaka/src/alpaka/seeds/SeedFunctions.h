#ifndef RecoTracker_MkFitAlpaka_src_alpaka_seeds_SeedFunctions_h
#define RecoTracker_MkFitAlpaka_src_alpaka_seeds_SeedFunctions_h

// Host+device transliterations of the MkFitCore seed-import arithmetic (CMSSW_20_1_0_pre2):
//   seedHasSillyValues   TrackBase::hasSillyValues (Track.cc:155-175) without dump/fix (Config.h:19-22)
//   seedMaxReachRadius   TrackBase::maxReachRadius (Track.cc:219-226)
//   seedZAtR             TrackBase::zAtR (Track.cc:228-275)
//   seedRegion           partitionSeeds1 = "phase2:1" (MkSeedPartitioners-phase2.cc:20-112)
//   seedBinnorKey        MkBuilder::import_seeds phi/eta binnor (MkBuilder.cc:248-256, binnor.h): masked B_pair key
// Same operations in the same order as MkFitCore.

#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"
#include "RecoTracker/MkFitAlpaka/interface/math/MathUtils.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/SeedSoA.h"

namespace mkfitdev::seeds {

  ALPAKA_FN_HOST_ACC inline bool seedHasSillyValues(const float* err21) {
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j <= i; ++j) {
        const float e = err21[i * (i + 1) / 2 + j];
        if ((i == j && e < 0) || !::mkfitdev::isFinite(e))  // edm::isFinite bit test
          return true;
      }
    return false;
  }

  struct SeedKin {
    float x, y, z, invpT, pT, px, py, pz;
    int charge;
  };

  ALPAKA_FN_HOST_ACC inline SeedKin seedKin(const float* p, int charge) {
    SeedKin k;
    k.x = p[0];
    k.y = p[1];
    k.z = p[2];
    k.invpT = p[3];
    k.pT = std::abs(1.f / p[3]);
    k.px = k.pT * std::cos(p[4]);
    k.py = k.pT * std::sin(p[4]);
    k.pz = k.pT / std::tan(p[5]);
    k.charge = charge;
    return k;
  }

  ALPAKA_FN_HOST_ACC inline float seedMaxReachRadius(SeedKin const& s) {
    const float k = ((s.charge < 0) ? 100.0f : -100.0f) / (Const::sol * Config::Bfield);
    const float abs_ooc_half = std::abs(k * s.pT);
    const float x_center = s.x - k * s.py;
    const float y_center = s.y + k * s.px;
    return hipo(x_center, y_center) + abs_ooc_half;
  }

  ALPAKA_FN_HOST_ACC inline float seedZAtR(SeedKin const& s, float R) {
    float xc = s.x;
    float yc = s.y;
    float pxc = s.px;
    float pyc = s.py;
    const float ipt = s.invpT;
    const float kinv = ((s.charge < 0) ? 0.01f : -0.01f) * Const::sol * Config::Bfield;
    const float k = 1.0f / kinv;
    const float c = 0.5f * kinv * ipt;
    const float ooc = 1.0f / c;
    const float lambda = s.pz * ipt;
    float D = 0;
    for (int i = 0; i < Config::Niter; ++i) {
      float r0 = hipo(xc, yc);
      float td = (R - r0) * c;
      float id = ooc * td * (1.0f + 0.16666666f * td * td);
      D += id;
      float cosa = std::cos(id * ipt * kinv);
      float sina = std::sin(id * ipt * kinv);
      xc += k * (pxc * sina - pyc * (1.0f - cosa));
      yc += k * (pyc * sina + pxc * (1.0f - cosa));
      const float pxo = pxc;
      pxc = pxc * cosa - pyc * sina;
      pyc = pyc * cosa + pxo * sina;
    }
    return s.z + lambda * D;
  }

  // TrackerInfo::EtaRegion: Reg_Endcap_Neg 0, Reg_Transition_Neg 1, Reg_Barrel 2, Reg_Transition_Pos 3, Reg_Endcap_Pos 4
  ALPAKA_FN_HOST_ACC inline int seedRegion(SeedKin const& S, SeedPartitionLimits const& L) {
    constexpr float tec_z_extra = 0.0f;
    const bool z_dir_pos = S.pz > 0;
    const float maxR = seedMaxReachRadius(S);
    if (z_dir_pos) {
      // barrel_pos_check(S, maxR, tecp2_rin, tecp2_zmax)
      const bool in_tec_as_brl = maxR > L.tecp2_rin && seedZAtR(S, L.tecp2_rin) < L.tecp2_zmax;
      if (!in_tec_as_brl)
        return 4;
      // endcap_pos_check(S, maxR, tecp1_rout, tecp1_rin, tecp1_zmin - tec_z_extra)
      const float zmin = L.tecp1_zmin - tec_z_extra;
      const bool in_tec =
          maxR > L.tecp1_rout ? seedZAtR(S, L.tecp1_rout) > zmin : (maxR > L.tecp1_rin && seedZAtR(S, maxR) > zmin);
      return in_tec ? 3 : 2;
    } else {
      const bool in_tec_as_brl = maxR > L.tecn2_rin && seedZAtR(S, L.tecn2_rin) > L.tecn2_zmin;
      if (!in_tec_as_brl)
        return 0;
      const float zmax = L.tecn1_zmax + tec_z_extra;
      const bool in_tec =
          maxR > L.tecn1_rout ? seedZAtR(S, L.tecn1_rout) < zmax : (maxR > L.tecn1_rin && seedZAtR(S, maxR) < zmax);
      return in_tec ? 1 : 2;
    }
  }

  // axis_pow2_u1<float, unsigned short, 10, 4>(-PI, PI) and axis<float, unsigned short, 8, 8>(-3, 3, 64):
  // register_entry_safe(phi, eta) -> B_pair(phiM, etaM) = etaM << 10 | phiM; c_A2_Mout_mask keeps all bits.
  constexpr int kSeedPhiMBits = 10;
  constexpr int kSeedEtaMBins = 64;

  ALPAKA_FN_HOST_ACC inline uint32_t seedBinnorKey(float phi, float eta) {
    const float phiMin = -Const::PI;
    const float phiFac = 1024u / (Const::PI - (-Const::PI));
    // float -> unsigned short of the floor, then & (2^10 - 1); the int path reproduces x86 for out-of-range values
    const int phiM = int(std::floor((phi - phiMin) * phiFac)) & ((1 << kSeedPhiMBits) - 1);
    const float etaMin = -3.0f, etaMax = 3.0f;
    const float etaFac = 64u / (etaMax - etaMin);
    const float etaLbhp = etaMax - 0.5 / etaFac;  // MkFitCore: R max - 0.5 / m_M_fac (double), stored as float
    int etaM;
    if (eta <= etaMin)
      etaM = 0;
    else if (eta >= etaLbhp)
      etaM = kSeedEtaMBins - 1;
    else
      etaM = int(std::floor((eta - etaMin) * etaFac));
    return (uint32_t(etaM & 0xffff) << kSeedPhiMBits) | uint32_t(phiM);
  }

  // Counting bins of the device import: region-major, then the key's eta bin, then its 4 high phi bits.
  // Bin order == (region, key) order, so a stable in-bin rank by (key, input row) gives MkFitCore's order.
  constexpr int kSeedBinsPerRegion = kSeedEtaMBins * 16;
  constexpr int kSeedNBins = kNSeedRegions * kSeedBinsPerRegion;
  ALPAKA_FN_HOST_ACC inline int seedCountBin(int region, uint32_t key) {
    return region * kSeedBinsPerRegion + int(key >> kSeedPhiMBits) * 16 + int((key & 1023u) >> 6);
  }

}  // namespace mkfitdev::seeds

#endif

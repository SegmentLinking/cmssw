#ifndef RecoTracker_MkFitAlpaka_interface_seeds_SeedsHostPack_h
#define RecoTracker_MkFitAlpaka_interface_seeds_SeedsHostPack_h

// HOST-ONLY: MkFitCore seeds (MkFitSeedWrapper::seeds(), the mkfit::TrackVec MkFitSeedConverter makes) -> device seed
// table rows, and the phase2:1 partitioner limits from the TrackerInfo. Do not include from device code.

#include <algorithm>
#include <cstring>

#include "RecoTracker/MkFitAlpaka/interface/seeds/SeedSoA.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

namespace mkfitdev {

  // MkSeedPartitioners-phase2.cc:10-46 (layer ids and merged mono/stereo limits)
  inline SeedPartitionLimits seedPartitionLimits(mkfit::TrackerInfo const& L) {
    constexpr int tecp1l_id = 28, tecp1u_id = 29, tecp2l_id = 30, tecp2u_id = 31;
    constexpr int tecn1l_id = 50, tecn1u_id = 51, tecn2l_id = 52, tecntu_id = 53;
    SeedPartitionLimits s;
    s.tecp1_rin = std::min(L[tecp1l_id].rin(), L[tecp1u_id].rin());
    s.tecp1_rout = std::max(L[tecp1l_id].rout(), L[tecp1u_id].rout());
    s.tecp1_zmin = std::min(L[tecp1l_id].zmin(), L[tecp1u_id].zmin());
    s.tecp2_rin = std::min(L[tecp2l_id].rin(), L[tecp2u_id].rin());
    s.tecp2_zmax = std::max(L[tecp2l_id].zmax(), L[tecp2u_id].zmax());
    s.tecn1_rin = std::min(L[tecn1l_id].rin(), L[tecn1u_id].rin());
    s.tecn1_rout = std::max(L[tecn1l_id].rout(), L[tecn1u_id].rout());
    s.tecn1_zmax = std::max(L[tecn1l_id].zmax(), L[tecn1u_id].zmax());
    s.tecn2_rin = std::min(L[tecn2l_id].rin(), L[tecntu_id].rin());
    s.tecn2_zmin = std::min(L[tecn2l_id].zmin(), L[tecntu_id].zmin());
    return s;
  }

  // Fills rows [0, n) and the scalars. lastHitPos(layer, index, float xyz[3]) must give the position of MkFitCore
  // eoh[layer].refHit(index) (the seed's last hit). Seeds with more than kMaxSeedHits hits are truncated and counted.
  template <typename LastHitPos>
  inline void packSeeds(mkfit::TrackVec const& in, LastHitPos&& lastHitPos, SeedSoAView v) {
    const int cap = v.metadata().size();
    const int n = std::min<int>(in.size(), cap);
    int ovf = 0;
    for (int i = 0; i < n; ++i) {
      const mkfit::Track& t = in[i];
      for (int k = 0; k < 6; ++k)
        v[i].params().v[k] = t.parameters()[k];
      std::memcpy(v[i].errors().v, t.errors().Array(), sizeof(float) * 21);
      v[i].charge() = t.charge();
      v[i].label() = t.label();
      v[i].chi2() = t.chi2();
      const auto st = t.getStatus();
      uint32_t stb;
      std::memcpy(&stb, &st, sizeof(stb));
      v[i].status() = stb;
      const int nTot = t.nTotalHits();
      const int nh = std::min(nTot, kMaxSeedHits);
      if (nTot > kMaxSeedHits)
        ++ovf;
      v[i].nHits() = nh;
      for (int h = 0; h < nh; ++h) {
        const mkfit::HitOnTrack hot = t.getHitOnTrack(h);
        v[i].hits().hot[h].index = hot.index;
        v[i].hits().hot[h].layer = hot.layer;
      }
      float xyz[3] = {0.f, 0.f, 0.f};
      if (nTot > 0) {
        const mkfit::HitOnTrack last = t.getLastHitOnTrack();
        lastHitPos(last.layer, last.index, xyz);
      }
      v[i].lastX() = xyz[0];
      v[i].lastY() = xyz[1];
      v[i].lastZ() = xyz[2];
    }
    v.nSeeds() = n;
    v.nKept() = 0;
    for (int r = 0; r < kNSeedRegions; ++r) {
      v.regionEnd().v[r] = 0;
      v.minLastLayer().v[r] = 9999;  // MkBuilder::begin_event initial values
      v.maxLastLayer().v[r] = 0;
    }
    v.nOverflowHits() = ovf + (int(in.size()) - n);
  }

}  // namespace mkfitdev

#endif

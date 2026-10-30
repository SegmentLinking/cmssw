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
  inline SeedPartitionLimits seedPartitionLimits(mkfit::TrackerInfo const& trackerInfo) {
    constexpr int tecp1l_id = 28, tecp1u_id = 29, tecp2l_id = 30, tecp2u_id = 31;
    constexpr int tecn1l_id = 50, tecn1u_id = 51, tecn2l_id = 52, tecntu_id = 53;
    SeedPartitionLimits limits;
    limits.tecp1_rin = std::min(trackerInfo[tecp1l_id].rin(), trackerInfo[tecp1u_id].rin());
    limits.tecp1_rout = std::max(trackerInfo[tecp1l_id].rout(), trackerInfo[tecp1u_id].rout());
    limits.tecp1_zmin = std::min(trackerInfo[tecp1l_id].zmin(), trackerInfo[tecp1u_id].zmin());
    limits.tecp2_rin = std::min(trackerInfo[tecp2l_id].rin(), trackerInfo[tecp2u_id].rin());
    limits.tecp2_zmax = std::max(trackerInfo[tecp2l_id].zmax(), trackerInfo[tecp2u_id].zmax());
    limits.tecn1_rin = std::min(trackerInfo[tecn1l_id].rin(), trackerInfo[tecn1u_id].rin());
    limits.tecn1_rout = std::max(trackerInfo[tecn1l_id].rout(), trackerInfo[tecn1u_id].rout());
    limits.tecn1_zmax = std::max(trackerInfo[tecn1l_id].zmax(), trackerInfo[tecn1u_id].zmax());
    limits.tecn2_rin = std::min(trackerInfo[tecn2l_id].rin(), trackerInfo[tecntu_id].rin());
    limits.tecn2_zmin = std::min(trackerInfo[tecn2l_id].zmin(), trackerInfo[tecntu_id].zmin());
    return limits;
  }

  // Fills rows [0, nSeeds) and the scalars. lastHitPos(layer, index, float xyz[3]) must give the position of MkFitCore
  // eoh[layer].refHit(index) (the seed's last hit). Seeds with more than kMaxSeedHits hits are truncated and counted.
  template <typename LastHitPos>
  inline void packSeeds(mkfit::TrackVec const& in, LastHitPos&& lastHitPos, SeedSoAView seeds) {
    const int cap = seeds.metadata().size();
    const int nSeeds = std::min<int>(in.size(), cap);
    int nOverflow = 0;
    for (int i = 0; i < nSeeds; ++i) {
      const mkfit::Track& track = in[i];
      for (int k = 0; k < 6; ++k)
        seeds[i].params().v[k] = track.parameters()[k];
      std::memcpy(seeds[i].errors().v, track.errors().Array(), sizeof(TrackErrors::v));
      seeds[i].charge() = track.charge();
      seeds[i].label() = track.label();
      seeds[i].chi2() = track.chi2();
      const auto status = track.getStatus();
      uint32_t statusBits;
      std::memcpy(&statusBits, &status, sizeof(statusBits));
      seeds[i].status() = statusBits;
      const int nTot = track.nTotalHits();
      const int nh = std::min(nTot, kMaxSeedHits);
      if (nTot > kMaxSeedHits)
        ++nOverflow;
      seeds[i].nHits() = nh;
      for (int ihit = 0; ihit < nh; ++ihit) {
        const mkfit::HitOnTrack hot = track.getHitOnTrack(ihit);
        seeds[i].hits().hot[ihit].index = hot.index;
        seeds[i].hits().hot[ihit].layer = hot.layer;
      }
      float xyz[3] = {0.f, 0.f, 0.f};
      if (nTot > 0) {
        const mkfit::HitOnTrack last = track.getLastHitOnTrack();
        lastHitPos(last.layer, last.index, xyz);
      }
      seeds[i].lastX() = xyz[0];
      seeds[i].lastY() = xyz[1];
      seeds[i].lastZ() = xyz[2];
    }
    seeds.nSeeds() = nSeeds;
    seeds.nKept() = 0;
    for (int region = 0; region < kNSeedRegions; ++region) {
      seeds.regionEnd().v[region] = 0;
      seeds.minLastLayer().v[region] = 9999;  // MkBuilder::begin_event initial values
      seeds.maxLastLayer().v[region] = 0;
    }
    seeds.nOverflowHits() = nOverflow + (int(in.size()) - nSeeds);
  }

}  // namespace mkfitdev

#endif

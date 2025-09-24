#ifndef RecoTracker_LSTCore_interface_LSTGeometry_PixelMapMethods_h
#define RecoTracker_LSTCore_interface_LSTGeometry_PixelMapMethods_h

#include <vector>
#include <tuple>
#include <unordered_map>
#include <boost/functional/hash.hpp>

#include "RecoTracker/LSTCore/interface/LSTGeometry/Common.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/Centroid.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/DetectorGeometry.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/Module.h"

namespace lst {

  using PtEtaPhiZChargeKey = std::tuple<unsigned int, unsigned int, unsigned int, unsigned int, int>;
  using LayerSubdetKey = std::tuple<unsigned int, unsigned int>;
  using LayerSubdetMap = std::unordered_map<LayerSubdetKey, std::vector<unsigned int>, boost::hash<LayerSubdetKey>>;
  using PtEtaPhiZChargeMap = std::unordered_map<PtEtaPhiZChargeKey, LayerSubdetMap, boost::hash<PtEtaPhiZChargeKey>>;

  PtEtaPhiZChargeMap computePixelMap(std::unordered_map<unsigned int, Centroid> const& centroids,
                                     DetectorGeometry const& det_geom) {
    constexpr unsigned int kNEta = 25;
    constexpr unsigned int kNPhi = 72;
    constexpr unsigned int kNZ = 25;
    constexpr double kPtBounds[] = {kPtThreshold, 2.0, 10'000.0};

    PtEtaPhiZChargeMap maps;

    // Initialize empty lists for the pixel map
    for (unsigned int ipt = 0; ipt < sizeof(kPtBounds) / sizeof(kPtBounds[0]); ipt++) {
      for (unsigned int ieta = 0; ieta < kNEta; ieta++) {
        for (unsigned int iphi = 0; iphi < kNPhi; iphi++) {
          for (unsigned int iz = 0; iz < kNZ; iz++) {
            maps[{ipt, ieta, iphi, iz, 1}] = LayerSubdetMap();
            maps[{ipt, ieta, iphi, iz, -1}] = LayerSubdetMap();
            auto& map_pos = maps.at({ipt, ieta, iphi, iz, 1});
            auto& map_neg = maps.at({ipt, ieta, iphi, iz, -1});
            for (unsigned int layer : {1, 2}) {
              for (unsigned int subdet : {4, 5}) {
                map_pos[{layer, subdet}] = {};
                map_neg[{layer, subdet}] = {};
              }
            }
          }
        }
      }
    }

    // Loop over the detids and for each detid compute which superbins it is connected to
    for (auto detId : det_geom.getDetIds()) {
      auto centroid = centroids.at(detId);

      // Parse the layer and subdet
      auto module = Module(detId, centroid.moduleType);
      auto layer = module.layer();
      if (layer > 2)
        continue;
      auto subdet = module.subdet();

      // Skip if the module is not PS module and is not lower module
      if (module.isLower() != 1 || module.moduleType() != 0)
        continue;
    }

    return maps;
  }

}  // namespace lst

#endif
#ifndef RecoTracker_LSTCore_interface_LSTGeometry_PixelMapMethods_h
#define RecoTracker_LSTCore_interface_LSTGeometry_PixelMapMethods_h

#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <boost/functional/hash.hpp>

#include "RecoTracker/LSTCore/interface/LSTGeometry/Common.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/Centroid.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/DetectorGeometry.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/Module.h"

namespace lstgeometry {

  using PtEtaPhiZChargeKey = std::tuple<unsigned int, unsigned int, unsigned int, unsigned int, int>;
  using LayerSubdetKey = std::tuple<unsigned int, unsigned int>;
  using PtEtaPhiZChargeMap = std::unordered_map<PtEtaPhiZChargeKey, std::unordered_set<unsigned int>, boost::hash<PtEtaPhiZChargeKey>>;
  using LayerSubdetMap = std::unordered_map<LayerSubdetKey, PtEtaPhiZChargeMap, boost::hash<LayerSubdetKey>>;
  using PixelMap = LayerSubdetMap;

  PixelMap computePixelMap(std::unordered_map<unsigned int, Centroid> const& centroids,
                                     DetectorGeometry const& det_geom) {

    LayerSubdetMap maps;

    // Initialize empty lists for the pixel map
    for (unsigned int layer : {1, 2}) {
      for (unsigned int subdet : {4, 5}) {
          maps[{layer, subdet}] = PtEtaPhiZChargeMap();
          auto& map = maps.at({layer, subdet});
            for (unsigned int ipt = 0; ipt < kPtBounds.size() - 1; ipt++) {
              for (unsigned int ieta = 0; ieta < kNEta; ieta++) {
                for (unsigned int iphi = 0; iphi < kNPhi; iphi++) {
                  for (unsigned int iz = 0; iz < kNZ; iz++) {
                    map.try_emplace({ipt, ieta, iphi, iz, 1});
                    map.try_emplace({ipt, ieta, iphi, iz, -1});
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

      // For this module, now compute which super bins they belong to
      // To compute which super bins it belongs to, one needs to provide at least pt and z window to compute compatible eta and phi range
      // So we have a loop in pt and Z
      for (unsigned int ipt = 0; ipt < kPtBounds.size() - 1; ipt++) {
        for (unsigned int iz = 0; iz < kNZ; iz++) {
          // The zmin, zmax of consideration
          double zmin = -30 + iz * (60. / kNZ);
          double zmax = -30 + (iz + 1) * (60. / kNZ);

          zmin -= 0.05;
          zmin += 0.05;

          // The ptmin, ptmax of consideration
          double pt_lo = kPtBounds[ipt];
          double pt_hi = kPtBounds[ipt + 1];

          auto [etamin, etamax] = det_geom.getCompatibleEtaRange(detId, zmin, zmax);

          etamin -= 0.05;
          etamax += 0.05;

          if (layer == 2 && subdet == 4) {
            if (etamax < 2.3)
              continue;
            if (etamin < 2.3)
              etamin = 2.3;
          }

          // Compute the indices of the compatible eta range
          unsigned int ietamin = static_cast<unsigned int>(std::max((etamin + 2.6) / (5.2 / kNEta), 0.0));
          unsigned int ietamax =
              static_cast<unsigned int>(std::min((etamax + 2.6) / (5.2 / kNEta), static_cast<double>(kNEta)));

          auto phi_ranges = det_geom.getCompatiblePhiRange(detId, pt_lo, pt_hi);

          int iphimin_pos = static_cast<int>((phi_ranges.first.first + std::numbers::pi_v<double>) /
                                             (2. * std::numbers::pi_v<double> / kNPhi));
          int iphimax_pos = static_cast<int>((phi_ranges.first.second + std::numbers::pi_v<double>) /
                                             (2. * std::numbers::pi_v<double> / kNPhi));
          int iphimin_neg = static_cast<int>((phi_ranges.second.first + std::numbers::pi_v<double>) /
                                             (2. * std::numbers::pi_v<double> / kNPhi));
          int iphimax_neg = static_cast<int>((phi_ranges.second.second + std::numbers::pi_v<double>) /
                                             (2. * std::numbers::pi_v<double> / kNPhi));

          unsigned int phibins_pos_start = iphimin_pos <= iphimax_pos ? iphimin_pos : 0;
          unsigned int phibins_pos_end = iphimin_pos <= iphimax_pos ? iphimax_pos : kNPhi;
          unsigned int phibins_neg_start = iphimin_neg <= iphimax_neg ? iphimin_neg : 0;
          unsigned int phibins_neg_end = iphimin_neg <= iphimax_neg ? iphimax_neg : kNPhi;

          for (unsigned int ieta = ietamin; ieta <= ietamax; ieta++) {
            for (unsigned int iphi = phibins_pos_start; iphi < phibins_pos_end; iphi++) {
              maps[{layer, subdet}][{ipt, ieta, iphi, iz, 1}].insert(detId);
            }
            for (unsigned int iphi = phibins_neg_start; iphi < phibins_neg_end; iphi++) {
              maps[{layer, subdet}][{ipt, ieta, iphi, iz, -1}].insert(detId);
            }
          }
        }
      }
    }

    return maps;
  }

}  // namespace lstgeometry

#endif
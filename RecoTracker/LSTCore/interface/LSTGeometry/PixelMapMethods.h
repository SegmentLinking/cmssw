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

  using LayerSubdetChargeKey = std::tuple<unsigned int, unsigned int, int>;
  using LayerSubdetChargeMap = std::unordered_map<LayerSubdetChargeKey,
                                                  std::vector<std::unordered_set<unsigned int>>,
                                                  boost::hash<LayerSubdetChargeKey>>;
  using PixelMap = LayerSubdetChargeMap;

  PixelMap computePixelMap(std::unordered_map<unsigned int, Centroid> const& centroids,
                           DetectorGeometry const& det_geom) {
    // Charge 0 is the union of charge 1 and charge -1
    PixelMap maps;

    std::size_t nSuperbin = (kPtBounds.size() - 1) * kNPhi * kNEta * kNZ;

    // Initialize empty lists for the pixel map
    for (unsigned int layer : {1, 2}) {
      for (unsigned int subdet : {4, 5}) {
        for (int charge : {-1, 0, 1}) {
          maps.try_emplace({layer, subdet, charge}, nSuperbin);
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
              static_cast<unsigned int>(std::min((etamax + 2.6) / (5.2 / kNEta), static_cast<double>(kNEta - 1)));

          auto phi_ranges = det_geom.getCompatiblePhiRange(detId, pt_lo, pt_hi);

          unsigned int iphimin_pos = static_cast<unsigned int>((phi_ranges.first.first + std::numbers::pi_v<double>) /
                                                               (2. * std::numbers::pi_v<double> / kNPhi));
          unsigned int iphimax_pos = static_cast<unsigned int>((phi_ranges.first.second + std::numbers::pi_v<double>) /
                                                               (2. * std::numbers::pi_v<double> / kNPhi));
          unsigned int iphimin_neg = static_cast<unsigned int>((phi_ranges.second.first + std::numbers::pi_v<double>) /
                                                               (2. * std::numbers::pi_v<double> / kNPhi));
          unsigned int iphimax_neg = static_cast<unsigned int>((phi_ranges.second.second + std::numbers::pi_v<double>) /
                                                               (2. * std::numbers::pi_v<double> / kNPhi));

          // <= to cover some inefficiencies
          for (unsigned int ieta = ietamin; ieta <= ietamax; ieta++) {
            // if the range is crossing the -pi v. pi boundary special care is needed
            if (iphimin_pos <= iphimax_pos) {
              for (unsigned int iphi = iphimin_pos; iphi < iphimax_pos; iphi++) {
                unsigned int isuperbin = (ipt * kNPhi * kNEta * kNZ) + (ieta * kNPhi * kNZ) + (iphi * kNZ) + iz;
                maps[{layer, subdet, 1}][isuperbin].insert(detId);
                maps[{layer, subdet, 0}][isuperbin].insert(detId);
              }
            } else {
              for (unsigned int iphi = 0; iphi < iphimax_pos; iphi++) {
                unsigned int isuperbin = (ipt * kNPhi * kNEta * kNZ) + (ieta * kNPhi * kNZ) + (iphi * kNZ) + iz;
                maps[{layer, subdet, 1}][isuperbin].insert(detId);
                maps[{layer, subdet, 0}][isuperbin].insert(detId);
              }
              for (unsigned int iphi = iphimin_pos; iphi < kNPhi; iphi++) {
                unsigned int isuperbin = (ipt * kNPhi * kNEta * kNZ) + (ieta * kNPhi * kNZ) + (iphi * kNZ) + iz;
                maps[{layer, subdet, 1}][isuperbin].insert(detId);
                maps[{layer, subdet, 0}][isuperbin].insert(detId);
              }
            }
            if (iphimin_neg <= iphimax_neg) {
              for (unsigned int iphi = iphimin_neg; iphi < iphimax_neg; iphi++) {
                unsigned int isuperbin = (ipt * kNPhi * kNEta * kNZ) + (ieta * kNPhi * kNZ) + (iphi * kNZ) + iz;
                maps[{layer, subdet, -1}][isuperbin].insert(detId);
                maps[{layer, subdet, 0}][isuperbin].insert(detId);
              }
            } else {
              for (unsigned int iphi = 0; iphi < iphimax_neg; iphi++) {
                unsigned int isuperbin = (ipt * kNPhi * kNEta * kNZ) + (ieta * kNPhi * kNZ) + (iphi * kNZ) + iz;
                maps[{layer, subdet, -1}][isuperbin].insert(detId);
                maps[{layer, subdet, 0}][isuperbin].insert(detId);
              }
              for (unsigned int iphi = iphimin_neg; iphi < kNPhi; iphi++) {
                unsigned int isuperbin = (ipt * kNPhi * kNEta * kNZ) + (ieta * kNPhi * kNZ) + (iphi * kNZ) + iz;
                maps[{layer, subdet, -1}][isuperbin].insert(detId);
                maps[{layer, subdet, 0}][isuperbin].insert(detId);
              }
            }
          }
        }
      }
    }

    return maps;
  }

}  // namespace lstgeometry

#endif
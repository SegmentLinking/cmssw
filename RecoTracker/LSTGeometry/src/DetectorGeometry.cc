#include "RecoTracker/LSTGeometry/interface/DetectorGeometry.h"

namespace lstgeometry {

  bool DetectorGeometry::isInEtaPhiBin(float eta, float phi, unsigned int eta_bin, unsigned int phi_bin) {
    float theta = 2. * std::atan(std::exp(-eta));

    if (eta_bin == 0) {
      if (theta > 3. * kEtaBinRad / 2.)
        return false;
    } else if (eta_bin == kNEtaBins - 1) {
      if (theta < (2 * (kNEtaBins - 1) - 1) * kEtaBinRad / 2.)
        return false;
    } else if (theta < (2 * eta_bin - 1) * kEtaBinRad / 2. || theta > (2 * (eta_bin + 1) + 1) * kEtaBinRad / 2.) {
      return false;
    }

    float pi = std::numbers::pi_v<float>;
    if (phi_bin == 0) {
      if (phi > -pi + kPhiBinWidth && phi < pi - kPhiBinWidth)
        return false;
    } else {
      if (phi < -pi + (phi_bin - 1) * kPhiBinWidth || phi > -pi + (phi_bin + 1) * kPhiBinWidth)
        return false;
    }

    return true;
  }

  std::pair<unsigned int, unsigned int> DetectorGeometry::getEtaPhiBins(float eta, float phi) {
    float theta = 2. * std::atan(std::exp(-eta));

    unsigned int eta_bin = 0;
    if (theta <= kEtaBinRad) {
      eta_bin = 0;
    } else if (theta >= (kNEtaBins - 1) * kEtaBinRad) {
      eta_bin = kNEtaBins - 1;
    } else {
      for (unsigned int i = 1; i < kNEtaBins - 1; i++) {
        if (theta >= i * kEtaBinRad && theta <= (i + 1) * kEtaBinRad) {
          eta_bin = i;
          break;
        }
      }
    }

    unsigned int phi_bin = 0;
    float pi = std::numbers::pi_v<float>;

    if (phi <= -pi + kPhiBinWidth / 2. || phi >= pi - kPhiBinWidth / 2.) {
      phi_bin = 0;
    } else {
      for (unsigned int i = 1; i < kNPhiBins; i++) {
        if (phi >= -pi + ((2 * i - 1) * kPhiBinWidth) / 2. && phi <= -pi + ((2 * i + 1) * kPhiBinWidth) / 2.) {
          phi_bin = i;
          break;
        }
      }
    }

    return std::make_pair(eta_bin, phi_bin);
  }

  DetectorGeometry::DetectorGeometry(Sensors sensors, std::vector<float> avg_radii, std::vector<float> avg_z)
      : sensors_(sensors), avg_radii_(avg_radii), avg_z_(avg_z) {}

  MatrixF4x3 const& DetectorGeometry::getCorners(unsigned int detId) const { return sensors_.at(detId).corners; }

  std::vector<unsigned int> DetectorGeometry::getDetIds(
      std::function<bool(const std::pair<const unsigned int, Sensor>&)> filter) const {
    std::vector<unsigned int> detIds;
    for (const auto& entry : sensors_) {
      if (filter(entry)) {
        detIds.push_back(entry.first);
      }
    }
    return detIds;
  }

  void DetectorGeometry::buildByLayer(Modules const& modules_info, Sensors const& sensors) {
    // Clear just in case they were already built
    barrel_lower_det_ids_.clear();
    endcap_lower_det_ids_.clear();

    // Initialize all vectors
    for (unsigned int etabin = 0; etabin < kNEtaBins; etabin++) {
      for (unsigned int phibin = 0; phibin < kNPhiBins; phibin++) {
        for (unsigned int layer = 1; layer < 7; layer++) {
          barrel_lower_det_ids_[{layer, etabin, phibin}] = {};
        }
        for (unsigned int layer = 1; layer < 6; layer++) {
          endcap_lower_det_ids_[{layer, etabin, phibin}] = {};
        }
      }
    }

    for (unsigned int layer = 1; layer < 7; layer++) {
      auto detids = getDetIds([&modules_info, &sensors, &layer](const auto& x) {
        auto& m = modules_info.at(sensors.at(x.first).moduleDetId);
        return m.subdet == 5 && m.layer == layer && m.isLower;
      });
      for (auto detid : detids) {
        auto corners = getCorners(detid);
        RowVectorF3 center = corners.colwise().mean();
        center /= 4.;
        auto etaphi = getEtaPhi(center(1), center(2), center(0));
        for (unsigned int etabin = 0; etabin < kNEtaBins; etabin++) {
          for (unsigned int phibin = 0; phibin < kNPhiBins; phibin++) {
            if (isInEtaPhiBin(etaphi.first, etaphi.second, etabin, phibin)) {
              barrel_lower_det_ids_[{layer, etabin, phibin}].push_back(detid);
            }
          }
        }
      }
    }
    for (unsigned int layer = 1; layer < 6; layer++) {
      auto detids = getDetIds([&modules_info, &sensors, &layer](const auto& x) {
        auto& m = modules_info.at(sensors.at(x.first).moduleDetId);
        return m.subdet == 4 && m.layer == layer && m.isLower;
      });
      for (auto detid : detids) {
        auto corners = getCorners(detid);
        RowVectorF3 center = corners.colwise().mean();
        center /= 4.;
        auto etaphi = getEtaPhi(center(1), center(2), center(0));
        for (unsigned int etabin = 0; etabin < kNEtaBins; etabin++) {
          for (unsigned int phibin = 0; phibin < kNPhiBins; phibin++) {
            if (isInEtaPhiBin(etaphi.first, etaphi.second, etabin, phibin)) {
              endcap_lower_det_ids_[{layer, etabin, phibin}].push_back(detid);
            }
          }
        }
      }
    }
  }

  std::vector<unsigned int> const& DetectorGeometry::getBarrelLayerDetIds(unsigned int layer,
                                                                          unsigned int etabin,
                                                                          unsigned int phibin) const {
    return barrel_lower_det_ids_.at({layer, etabin, phibin});
  }

  std::vector<unsigned int> const& DetectorGeometry::getEndcapLayerDetIds(unsigned int layer,
                                                                          unsigned int etabin,
                                                                          unsigned int phibin) const {
    return endcap_lower_det_ids_.at({layer, etabin, phibin});
  }

  float DetectorGeometry::getBarrelLayerAverageRadius(unsigned int layer) const { return avg_radii_[layer - 1]; }

  float DetectorGeometry::getEndcapLayerAverageAbsZ(unsigned int layer) const { return avg_z_[layer - 1]; }

  float DetectorGeometry::getMinR(unsigned int detId) const {
    auto const& corners = getCorners(detId);
    float minR = std::numeric_limits<float>::max();
    for (int i = 0; i < corners.rows(); i++) {
      float x = corners(i, 1);
      float y = corners(i, 2);
      minR = std::min(minR, std::sqrt(x * x + y * y));
    }
    return minR;
  }

  float DetectorGeometry::getMaxR(unsigned int detId) const {
    auto const& corners = getCorners(detId);
    float maxR = std::numeric_limits<float>::min();
    for (int i = 0; i < corners.rows(); i++) {
      float x = corners(i, 1);
      float y = corners(i, 2);
      maxR = std::max(maxR, std::sqrt(x * x + y * y));
    }
    return maxR;
  }

  float DetectorGeometry::getMinZ(unsigned int detId) const {
    auto const& corners = getCorners(detId);
    float minZ = std::numeric_limits<float>::max();
    for (int i = 0; i < corners.rows(); i++) {
      float z = corners(i, 0);
      minZ = std::min(minZ, z);
    }
    return minZ;
  }

  float DetectorGeometry::getMaxZ(unsigned int detId) const {
    auto const& corners = getCorners(detId);
    float maxZ = std::numeric_limits<float>::lowest();
    for (int i = 0; i < corners.rows(); i++) {
      float z = corners(i, 0);
      maxZ = std::max(maxZ, z);
    }
    return maxZ;
  }

  float DetectorGeometry::getMinPhi(unsigned int detId) const {
    auto const& corners = getCorners(detId);
    float minPhi = std::numeric_limits<float>::max();
    float minPosPhi = std::numeric_limits<float>::max();
    float minNegPhi = std::numeric_limits<float>::max();
    unsigned int nPos = 0;
    unsigned int nOverPi2 = 0;
    for (int i = 0; i < corners.rows(); i++) {
      float phi = phi_mpi_pi(std::numbers::pi_v<float> + std::atan2(-corners(i, 2), -corners(i, 1)));
      minPhi = std::min(minPhi, phi);
      if (phi > 0) {
        minPosPhi = std::min(minPosPhi, phi);
        nPos++;
      } else {
        minNegPhi = std::min(minNegPhi, phi);
      }
      if (std::fabs(phi) > std::numbers::pi_v<float> / 2.) {
        nOverPi2++;
      }
    }
    if (nPos == 4 || nPos == 0)
      return minPhi;
    if (nOverPi2 == 4)
      return minPosPhi;
    return minPhi;
  }

  float DetectorGeometry::getMaxPhi(unsigned int detId) const {
    auto const& corners = getCorners(detId);
    float maxPhi = std::numeric_limits<float>::lowest();
    float maxPosPhi = std::numeric_limits<float>::lowest();
    float maxNegPhi = std::numeric_limits<float>::lowest();
    unsigned int nPos = 0;
    unsigned int nOverPi2 = 0;
    for (int i = 0; i < corners.rows(); i++) {
      float phi = phi_mpi_pi(std::numbers::pi_v<float> + std::atan2(-corners(i, 2), -corners(i, 1)));
      maxPhi = std::max(maxPhi, phi);
      if (phi > 0) {
        maxPosPhi = std::max(maxPosPhi, phi);
        nPos++;
      } else {
        maxNegPhi = std::max(maxNegPhi, phi);
      }
      if (std::fabs(phi) > std::numbers::pi_v<float> / 2.) {
        nOverPi2++;
      }
    }
    if (nPos == 4 || nPos == 0)
      return maxPhi;
    if (nOverPi2 == 4)
      return maxNegPhi;
    return maxPhi;
  }

  std::pair<float, float> DetectorGeometry::getCompatibleEtaRange(unsigned int detId,
                                                                  float zmin_bound,
                                                                  float zmax_bound) const {
    float minr = getMinR(detId);
    float maxr = getMaxR(detId);
    float minz = getMinZ(detId);
    float maxz = getMaxZ(detId);
    float mineta = -std::log(std::tan(std::atan2(minz > 0 ? maxr : minr, minz - zmin_bound) / 2.));
    float maxeta = -std::log(std::tan(std::atan2(maxz > 0 ? minr : maxr, maxz - zmax_bound) / 2.));

    if (maxeta < mineta)
      std::swap(maxeta, mineta);
    return std::make_pair(mineta, maxeta);
  }

  std::pair<std::pair<float, float>, std::pair<float, float>> DetectorGeometry::getCompatiblePhiRange(
      unsigned int detId, float ptmin, float ptmax) const {
    float minr = getMinR(detId);
    float maxr = getMaxR(detId);
    float minphi = getMinPhi(detId);
    float maxphi = getMaxPhi(detId);
    float A = k2Rinv1GeVf * kB / 2.;
    float pos_q_phi_lo_bound = phi_mpi_pi(A * minr / ptmax + minphi);
    float pos_q_phi_hi_bound = phi_mpi_pi(A * maxr / ptmin + maxphi);
    float neg_q_phi_lo_bound = phi_mpi_pi(-A * maxr / ptmin + minphi);
    float neg_q_phi_hi_bound = phi_mpi_pi(-A * minr / ptmax + maxphi);
    return std::make_pair(std::make_pair(pos_q_phi_lo_bound, pos_q_phi_hi_bound),
                          std::make_pair(neg_q_phi_lo_bound, neg_q_phi_hi_bound));
  }
}  // namespace lstgeometry

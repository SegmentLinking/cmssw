#ifndef RecoTracker_LSTGeometry_interface_DetectorGeometry_h
#define RecoTracker_LSTGeometry_interface_DetectorGeometry_h

#include <algorithm>
#include <vector>
#include <functional>
#include <boost/functional/hash.hpp>

#include "RecoTracker/LSTGeometry/interface/Common.h"
#include "RecoTracker/LSTGeometry/interface/Module.h"
#include "RecoTracker/LSTGeometry/interface/Sensor.h"

namespace lstgeometry {

  using LayerEtaBinPhiBinKey = std::tuple<unsigned int, unsigned int, unsigned int>;

  class DetectorGeometry {
  private:
    Sensors sensors_;  // TODO: Refactor to avoid a copy
    std::vector<float> avg_radii_;
    std::vector<float> avg_z_;
    std::unordered_map<LayerEtaBinPhiBinKey, std::vector<unsigned int>, boost::hash<LayerEtaBinPhiBinKey>>
        barrel_lower_det_ids_;
    std::unordered_map<LayerEtaBinPhiBinKey, std::vector<unsigned int>, boost::hash<LayerEtaBinPhiBinKey>>
        endcap_lower_det_ids_;

  public:
    DetectorGeometry(Sensors sensors, std::vector<float> avg_radii, std::vector<float> avg_z);

    MatrixF4x3 const& getCorners(unsigned int detId) const;

    std::vector<unsigned int> getDetIds(std::function<bool(const std::pair<const unsigned int, Sensor>&)> filter =
                                            [](const auto&) { return true; }) const;

    void buildByLayer(Modules const& modules_info, Sensors const& sensors);

    std::vector<unsigned int> const& getBarrelLayerDetIds(unsigned int layer,
                                                          unsigned int etabin,
                                                          unsigned int phibin) const;

    std::vector<unsigned int> const& getEndcapLayerDetIds(unsigned int layer,
                                                          unsigned int etabin,
                                                          unsigned int phibin) const;

    float getBarrelLayerAverageRadius(unsigned int layer) const;

    float getEndcapLayerAverageAbsZ(unsigned int layer) const;

    float getMinR(unsigned int detId) const;

    float getMaxR(unsigned int detId) const;

    float getMinZ(unsigned int detId) const;

    float getMaxZ(unsigned int detId) const;

    float getMinPhi(unsigned int detId) const;

    float getMaxPhi(unsigned int detId) const;

    std::pair<float, float> getCompatibleEtaRange(unsigned int detId, float zmin_bound, float zmax_bound) const;

    std::pair<std::pair<float, float>, std::pair<float, float>> getCompatiblePhiRange(unsigned int detId,
                                                                                      float ptmin,
                                                                                      float ptmax) const;

    // We split modules into overlapping eta-phi bins so that it's easier to construct module maps
    // These values are just guesses and can be optimized later
    static constexpr unsigned int kNEtaBins = 4;
    static constexpr float kEtaBinRad = std::numbers::pi_v<float> / kNEtaBins;
    static constexpr unsigned int kNPhiBins = 6;
    static constexpr float kPhiBinWidth = 2 * std::numbers::pi_v<float> / kNPhiBins;

    static bool isInEtaPhiBin(float eta, float phi, unsigned int eta_bin, unsigned int phi_bin);
    static std::pair<unsigned int, unsigned int> getEtaPhiBins(float eta, float phi);
  };
}  // namespace lstgeometry

#endif

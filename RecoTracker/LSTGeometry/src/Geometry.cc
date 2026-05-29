#include <chrono>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/LSTGeometry/interface/Geometry.h"
#include "RecoTracker/LSTGeometry/interface/SensorBinning.h"
#include "RecoTracker/LSTGeometry/interface/Slope.h"

using namespace lstgeometry;

Geometry::Geometry(Sensors sensors,
                   std::array<float, kBarrelLayers> const &average_r_barrel,
                   std::array<float, kEndcapLayers> const &average_z_endcap,
                   float pt_cut) {
  const auto start = std::chrono::steady_clock::now();
  auto slopes = computeSlopes(sensors);
  const auto slopesDone = std::chrono::steady_clock::now();
  barrel_slopes = std::move(std::get<0>(slopes));
  endcap_slopes = std::move(std::get<1>(slopes));

  auto binned_detids = binDetIds(sensors);
  const auto binningDone = std::chrono::steady_clock::now();

  pixel_map = buildPixelMap(sensors, pt_cut);
  const auto pixelMapDone = std::chrono::steady_clock::now();

  module_map = buildModuleMap(sensors, binned_detids, average_r_barrel, average_z_endcap, pt_cut);
  const auto moduleMapDone = std::chrono::steady_clock::now();

  // Drop all the extra data that is no longer needed
  for (auto &[detId, sensor] : sensors) {
    sensor.extra.reset();
  }

  this->sensors = std::move(sensors);
  const auto done = std::chrono::steady_clock::now();

  edm::LogInfo("LSTGeometryESProducer")
      << "Temporary timing: lstgeometry::Geometry slopes "
      << std::chrono::duration<double, std::milli>(slopesDone - start).count() << " ms, binning "
      << std::chrono::duration<double, std::milli>(binningDone - slopesDone).count() << " ms, pixel map "
      << std::chrono::duration<double, std::milli>(pixelMapDone - binningDone).count() << " ms, module map "
      << std::chrono::duration<double, std::milli>(moduleMapDone - pixelMapDone).count() << " ms, cleanup "
      << std::chrono::duration<double, std::milli>(done - moduleMapDone).count() << " ms, total "
      << std::chrono::duration<double, std::milli>(done - start).count() << " ms";
}

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(lstgeometry::Geometry);

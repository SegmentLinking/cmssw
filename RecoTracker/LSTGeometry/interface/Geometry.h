#ifndef RecoTracker_LSTGeometry_interface_Geometry_h
#define RecoTracker_LSTGeometry_interface_Geometry_h

#include <memory>

#include "RecoTracker/LSTGeometry/interface/Slope.h"
#include "RecoTracker/LSTGeometry/interface/PixelMap.h"
#include "RecoTracker/LSTGeometry/interface/ModuleMap.h"
#include "RecoTracker/LSTGeometry/interface/Module.h"
#include "RecoTracker/LSTGeometry/interface/Sensor.h"

namespace lstgeometry {

  struct Geometry {
    std::shared_ptr<Sensors> sensors;
    Slopes barrel_slopes;
    Slopes endcap_slopes;
    PixelMap pixel_map;
    ModuleMap module_map;

    Geometry(Modules &modules,
             std::shared_ptr<Sensors> sensors,
             std::vector<float> const &average_r,
             std::vector<float> const &average_z,
             float ptCut);
  };

}  // namespace lstgeometry

#endif

#ifndef RecoLocalTracker_SiPixelRecHits_interface_pixelCPEforDeviceClusterParams_h
#define RecoLocalTracker_SiPixelRecHits_interface_pixelCPEforDeviceClusterParams_h

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDeviceTrackAngles.h"

namespace pixelCPEforDevice {

  // The inputs of hitParametersTrackAngles from a cluster (host): PixelCPEGenericBase::collect_edge_charges without
  // charge truncation, and the cluster charge
  inline GenericClusterParams genericClusterParams(SiPixelCluster const& cluster) {
    GenericClusterParams params{};
    params.minRow = cluster.minPixelRow();
    params.maxRow = cluster.maxPixelRow();
    params.minCol = cluster.minPixelCol();
    params.maxCol = cluster.maxPixelCol();
    for (int i = 0; i < cluster.size(); ++i) {
      auto const pixel = cluster.pixel(i);
      if (pixel.x == params.minRow)
        params.qFirstX += pixel.adc;
      if (pixel.x == params.maxRow)
        params.qLastX += pixel.adc;
      if (pixel.y == params.minCol)
        params.qFirstY += pixel.adc;
      if (pixel.y == params.maxCol)
        params.qLastY += pixel.adc;
    }
    params.charge = cluster.charge();
    return params;
  }

}  // namespace pixelCPEforDevice

#endif  // RecoLocalTracker_SiPixelRecHits_interface_pixelCPEforDeviceClusterParams_h

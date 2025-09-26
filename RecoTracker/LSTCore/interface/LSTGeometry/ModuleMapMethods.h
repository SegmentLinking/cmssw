#ifndef RecoTracker_LSTCore_interface_LSTGeometry_ModuleMapMethods_h
#define RecoTracker_LSTCore_interface_LSTGeometry_ModuleMapMethods_h

#include <cmath>
#include <vector>

#include "RecoTracker/LSTCore/interface/LSTGeometry/LSTMath.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/Centroid.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/Module.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/DetectorGeometry.h"

namespace lstgeometry {

  std::vector<unsigned int> getStraightLineConnections(unsigned int ref_detid,
                                                       std::unordered_map<unsigned int, Centroid> const& centroids,
                                                       DetectorGeometry const& det_geom) {
    auto centroid = centroids.at(ref_detid);

    double refphi = std::atan2(centroid.y, centroid.x);

    Module refmodule(ref_detid);

    unsigned short ref_layer = refmodule.layer();
    unsigned short ref_subdet = refmodule.subdet();

    auto const& tar_detids_to_be_considered =
        ref_subdet == 5 ? det_geom.getBarrelLayerDetIds(ref_layer + 1) : det_geom.getEndcapLayerDetIds(ref_layer + 1);

    std::vector<unsigned int> list_of_detids_etaphi_layer_tar;
    for (unsigned int tar_detid : tar_detids_to_be_considered) {
      if (moduleOverlapsInEtaPhi(det_geom.getCorners(ref_detid), det_geom.getCorners(tar_detid), refphi, 0) ||
          moduleOverlapsInEtaPhi(det_geom.getCorners(ref_detid), det_geom.getCorners(tar_detid), refphi, 10) ||
          moduleOverlapsInEtaPhi(det_geom.getCorners(ref_detid), det_geom.getCorners(tar_detid), refphi, -10))
        list_of_detids_etaphi_layer_tar.push_back(tar_detid);
    }

    // Consider barrel to endcap connections if the intersection area is > 0

    return list_of_detids_etaphi_layer_tar;
  }

}  // namespace lstgeometry

#endif
#ifndef RecoTracker_MkFitAlpaka_interface_navdev_NavFlatTables_h
#define RecoTracker_MkFitAlpaka_interface_navdev_NavFlatTables_h
// The flat layer tables of the portable compatibleDets search (interface/math/NavSearch.h) for
// every layer of the tracker, built once per TrackerRecoGeometryRecord IOV (MkFitAlpakaNavFlatTablesBuilder.h,
// MkFitAlpakaNavFlatTablesESProducer). Read by MkFitAlpakaOutputTrackConverter (navPortable).
#include <array>
#include <unordered_map>
#include <vector>

#include "RecoTracker/MkFitAlpaka/interface/math/NavSearch.h"

class DetLayer;
class GeomDet;

namespace mkfitdev::navdev {
  struct NavFlatTables {
    std::vector<NavDet> dets;
    std::vector<const GeomDet*> detPtr;  // flat det index -> GeomDet
    std::vector<int> idx;
    std::vector<NavRing> rings;
    std::vector<NavSubDisk> subDisks;
    std::vector<NavRod> rods;
    std::vector<NavBarrel> barrels;
    std::vector<float> zs;
    // per layer: {kind (0 ring layer, 1 pixel double disk, 2 barrel), first ring / sub-disk / barrel, count}; a layer
    // that is missing or has first < 0 is searched on the host
    std::unordered_map<const DetLayer*, std::array<int, 3>> layers;
    // the device navigation (MkFitAlpakaNavDeviceProducer): layers by their index in GeometricSearchTracker::allLayers()
    std::vector<const DetLayer*> allLayers;
    std::unordered_map<const DetLayer*, int> layerIndex;
    std::vector<std::array<int, 3>> layerFlat;  // per allLayers() index: as layers (first -1: host)
    std::vector<int> mkFitToLayer;              // mkFit layer number -> allLayers() index (-1: none)
    // allLayers indices in increasing DetLayer pointer order = the order of the std::set<const DetLayer*>
    // that SimpleNavigableLayer::compatibleLayers returns (the converter lists a device set in this order)
    std::vector<int> pointerOrder;
    int nLayers = 0, nUnsupported = 0;
    long long nonRect = 0;
    NavTables tables() const {
      return NavTables{dets.data(), idx.data(), rings.data(), subDisks.data(), rods.data(), barrels.data(), zs.data()};
    }
  };
}  // namespace mkfitdev::navdev

#endif

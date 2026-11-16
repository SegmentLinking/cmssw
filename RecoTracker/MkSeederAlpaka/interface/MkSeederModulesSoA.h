#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederModulesSoA_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederModulesSoA_h

#include <cstdint>

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkseeder {

  // The tracker modules, layer by layer (a layer's modules from MkSeederLayerGeometrySoA::firstModule, in the order of
  // mkfit::LayerInfo::module_info(), i.e. by Hit::detIDinLayer()): the plane of the seed fits (mkfit::ModuleInfo),
  // and for a pixel module its frame for the expected cluster sizes (the rotation's rows are the local x, y, z axes in
  // global coordinates, the pitches and the thickness); zeros for the other modules. With ps_mark (MkSeederPSMark.h),
  // an OT1-3 P module's S partner (its index in the S layer; -1: none, and every other module) and its frame (rows: u,
  // v, n of PSMarkLayer::frame)
  GENERATE_SOA_LAYOUT(MkSeederModulesLayout,
                      SOA_EIGEN_COLUMN(Eigen::Vector3f, position),
                      SOA_EIGEN_COLUMN(Eigen::Vector3f, zDirection),
                      SOA_EIGEN_COLUMN(Eigen::Vector3f, xDirection),
                      SOA_COLUMN(bool, isPixel),
                      SOA_EIGEN_COLUMN(Eigen::Matrix3f, rotation),
                      SOA_COLUMN(float, pitchX),
                      SOA_COLUMN(float, pitchY),
                      SOA_COLUMN(float, thickness),
                      SOA_COLUMN(int32_t, psPartner),
                      SOA_EIGEN_COLUMN(Eigen::Matrix3f, psFrame))

  // The material of the seed fits (mkfit::TrackerInfo's grid in |z| and r): one row per bin, row zBin * nBinsR +
  // rBin, with zBin = int(|z| * binsPerCmZ), rBin = int(r * binsPerCmR)
  GENERATE_SOA_LAYOUT(MkSeederMaterialLayout,
                      SOA_COLUMN(float, bbxi),
                      SOA_COLUMN(float, radiationLength),
                      SOA_SCALAR(int32_t, nBinsZ),
                      SOA_SCALAR(int32_t, nBinsR),
                      SOA_SCALAR(float, binsPerCmZ),
                      SOA_SCALAR(float, binsPerCmR))

  using MkSeederModulesSoA = MkSeederModulesLayout<>;
  using MkSeederMaterialSoA = MkSeederMaterialLayout<>;

}  // namespace mkseeder

#endif

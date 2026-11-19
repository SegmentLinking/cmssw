#ifndef RecoTracker_MkSeederAlpaka_interface_alpaka_MkSeederTablesDeviceCollection_h
#define RecoTracker_MkSeederAlpaka_interface_alpaka_MkSeederTablesDeviceCollection_h

// The device collections of the seeder's static tables (MkSeederESData)

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederTablesHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  using MkSeederLayerGeometryDeviceCollection = PortableCollection<::mkseeder::MkSeederLayerGeometrySoA>;
  using MkSeederChainDeviceCollection = PortableCollection<::mkseeder::MkSeederChainSoA>;
  using MkSeederShapeDeviceCollection = PortableCollection<::mkseeder::MkSeederShapeSoA>;
  using MkSeederGapMapDeviceCollection = PortableCollection<::mkseeder::MkSeederGapMapSoA>;
  using MkSeederModulesDeviceCollection = PortableCollection<::mkseeder::MkSeederModulesSoA>;
  using MkSeederMaterialDeviceCollection = PortableCollection<::mkseeder::MkSeederMaterialSoA>;
  using MkSeederPatternsDeviceCollection = PortableCollection<::mkseeder::MkSeederPatternsSoA>;
  using MkSeederTripletMlpWeightsDeviceObject = PortableObject<::mkseeder::MkSeederTripletMlpWeights>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif

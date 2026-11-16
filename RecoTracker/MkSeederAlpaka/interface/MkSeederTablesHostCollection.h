#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederTablesHostCollection_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederTablesHostCollection_h

// The host collections of the seeder's static tables (MkSeederESData)

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederChainSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederGapMapSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederLayerGeometrySoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederModulesSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederPatternsSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederShapeSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederTripletMlpWeights.h"

namespace mkseeder {

  using MkSeederLayerGeometryHostCollection = PortableHostCollection<MkSeederLayerGeometrySoA>;
  using MkSeederChainHostCollection = PortableHostCollection<MkSeederChainSoA>;
  using MkSeederShapeHostCollection = PortableHostCollection<MkSeederShapeSoA>;
  using MkSeederGapMapHostCollection = PortableHostCollection<MkSeederGapMapSoA>;
  using MkSeederModulesHostCollection = PortableHostCollection<MkSeederModulesSoA>;
  using MkSeederMaterialHostCollection = PortableHostCollection<MkSeederMaterialSoA>;
  using MkSeederPatternsHostCollection = PortableHostCollection<MkSeederPatternsSoA>;
  using MkSeederTripletMlpWeightsHostObject = PortableHostObject<MkSeederTripletMlpWeights>;

}  // namespace mkseeder

#endif

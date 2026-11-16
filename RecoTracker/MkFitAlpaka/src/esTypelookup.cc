#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESData.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESGeometryData.h"

TYPELOOKUP_DATA_REG(mkfitdev::ESData<alpaka_common::DevHost>);
TYPELOOKUP_DATA_REG(mkfitdev::ESGeometryData<alpaka_common::DevHost>);
#include "RecoTracker/MkFitAlpaka/interface/navdev/NavFlatTables.h"

TYPELOOKUP_DATA_REG(mkfitdev::navdev::NavFlatTables);

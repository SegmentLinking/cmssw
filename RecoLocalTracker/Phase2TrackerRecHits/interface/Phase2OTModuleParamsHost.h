#ifndef RecoLocalTracker_Phase2TrackerRecHits_interface_Phase2OTModuleParamsHost_h
#define RecoLocalTracker_Phase2TrackerRecHits_interface_Phase2OTModuleParamsHost_h

#include <cstdint>

#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEPosition.h"

// Every OT module, one row per module in GeomDetUnit::index() order, the row of a module being its
// GeomDetUnit::index() - firstModuleIndex: the Phase2StripCPE parameters, the module surface and the DetId
GENERATE_SOA_LAYOUT(Phase2OTModuleParamsLayout,
                    SOA_COLUMN(phase2StripCPE::ModuleParams, cpe),
                    SOA_COLUMN(float, xerrLocal),  // local position errors xx and yy (xy is zero)
                    SOA_COLUMN(float, yerrLocal),
                    SOA_COLUMN(SOAFrame<float>, frame),
                    SOA_COLUMN(uint32_t, detId),
                    SOA_SCALAR(uint32_t, firstModuleIndex))  // GeomDetUnit::index() of the first OT module

using Phase2OTModuleParamsSoA = Phase2OTModuleParamsLayout<>;
using Phase2OTModuleParamsHost = PortableHostCollection<Phase2OTModuleParamsSoA>;

#endif  // RecoLocalTracker_Phase2TrackerRecHits_interface_Phase2OTModuleParamsHost_h

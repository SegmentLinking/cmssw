#ifndef RecoLocalTracker_Phase2TrackerRecHits_interface_Phase2StripCPEParamsHost_h
#define RecoLocalTracker_Phase2TrackerRecHits_interface_Phase2StripCPEParamsHost_h

#include <cstdint>

#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEPosition.h"

// Phase2StripCPE for the device: one row per OT module, row = GeomDetUnit::index() - firstModuleIndex
GENERATE_SOA_LAYOUT(Phase2StripCPEParamsLayout,
                    SOA_COLUMN(phase2StripCPE::ModuleParams, position),
                    SOA_COLUMN(float, xerrLocal),  // local position errors xx and yy (xy is zero)
                    SOA_COLUMN(float, yerrLocal),
                    SOA_COLUMN(SOAFrame<float>, frame),  // module surface position and rotation
                    SOA_COLUMN(uint32_t, detId),
                    SOA_SCALAR(uint32_t, firstModuleIndex))

using Phase2StripCPEParamsSoA = Phase2StripCPEParamsLayout<>;
using Phase2StripCPEParamsHost = PortableHostCollection<Phase2StripCPEParamsSoA>;

#endif  // RecoLocalTracker_Phase2TrackerRecHits_interface_Phase2StripCPEParamsHost_h

#ifndef RecoTracker_LSTCore_interface_TrackCandidatesBLFFitDeviceCollection_h
#define RecoTracker_LSTCore_interface_TrackCandidatesBLFFitDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/TrackCandidatesSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using TrackCandidatesBLFFitDeviceCollection = PortableCollection<TrackCandidatesBLFFitSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif

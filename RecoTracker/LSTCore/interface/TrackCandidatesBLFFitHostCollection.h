#ifndef RecoTracker_LSTCore_interface_TrackCandidatesBLFFitHostCollection_h
#define RecoTracker_LSTCore_interface_TrackCandidatesBLFFitHostCollection_h

#include "RecoTracker/LSTCore/interface/TrackCandidatesSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using TrackCandidatesBLFFitHostCollection = PortableHostCollection<TrackCandidatesBLFFitSoA>;
}  // namespace lst
#endif

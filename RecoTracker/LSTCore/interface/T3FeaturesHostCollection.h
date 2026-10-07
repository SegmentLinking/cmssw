#ifndef RecoTracker_LSTCore_interface_T3FeaturesHostCollection_h
#define RecoTracker_LSTCore_interface_T3FeaturesHostCollection_h

#include "RecoTracker/LSTCore/interface/T3FeaturesSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using T3FeaturesHostCollection = PortableHostCollection<T3FeaturesSoA>;
}  // namespace lst
#endif

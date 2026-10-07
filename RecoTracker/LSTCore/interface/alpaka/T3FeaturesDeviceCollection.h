#ifndef RecoTracker_LSTCore_interface_alpaka_T3FeaturesDeviceCollection_h
#define RecoTracker_LSTCore_interface_alpaka_T3FeaturesDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/T3FeaturesSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using T3FeaturesDeviceCollection = PortableCollection<T3FeaturesSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif

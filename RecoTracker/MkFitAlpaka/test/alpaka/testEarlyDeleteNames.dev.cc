// Prints, per producer type, the class part of the branch name (edm::TypeID::friendlyClassName, as the framework
// registers the product) of the MkFitAlpaka products that RecoTracker/Configuration customiseEarlyDeleteForMkFit
// deletes early; testEarlyDeleteMkFitAlpaka.sh compares them with the names the customise writes.

#include <cstdio>
#include <string>

#include "DataFormats/Common/interface/DeviceProduct.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/EventOfHitsProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/TrackProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/othits/alpaka/OTRecHitDeviceCollection.h"

namespace {
  // the asynchronous backends put a device product into the event as edm::DeviceProduct<T>
  template <typename T>
  std::string branchClass() {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    return edm::TypeID(typeid(edm::DeviceProduct<T>)).friendlyClassName();
#else
    return edm::TypeID(typeid(T)).friendlyClassName();
#endif
  }
}  // namespace

int main() {
  std::printf("MkFitAlpakaEventOfHitsProducer@alpaka %s\n",
              branchClass<ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::EventOfHitsDeviceCollection>().c_str());
  std::printf("MkFitAlpakaBuildProducer@alpaka %s\n",
              branchClass<ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::TrackSoADeviceCollection>().c_str());
  std::printf("MkFitAlpakaOTRecHitsProducer@alpaka %s\n",
              branchClass<ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::OTRecHitDeviceCollection>().c_str());
  return 0;
}

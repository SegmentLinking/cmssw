#ifndef RecoTracker_MkSeederAlpaka_interface_alpaka_MkSeederESData_h
#define RecoTracker_MkSeederAlpaka_interface_alpaka_MkSeederESData_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederESData.h"
#include "RecoTracker/MkSeederAlpaka/interface/alpaka/MkSeederTablesDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  // the seeder's static tables in the memory of the backend's device
  using MkSeederESDataDevice = ::mkseeder::MkSeederESData<Device>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif

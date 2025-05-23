#ifndef RecoTracker_LSTCore_interface_LSTOutputHostCollection_h
#define RecoTracker_LSTCore_interface_LSTOutputHostCollection_h

#include "RecoTracker/LSTCore/interface/LSTOutputSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace lst {
  using LSTOutputHostCollection = PortableHostCollection<OutputSoA>;
}  // namespace lst

#endif

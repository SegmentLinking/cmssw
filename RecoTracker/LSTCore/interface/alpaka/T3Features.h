#ifndef RecoTracker_LSTCore_interface_alpaka_T3Features_h
#define RecoTracker_LSTCore_interface_alpaka_T3Features_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoTracker/LSTCore/interface/alpaka/LSTInputDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/MiniDoubletsDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/ObjectRangesDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/SegmentsDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/T3FeaturesDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/TripletsDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  // Builds the 49 transformer input features for every LST T3 with pt < maxT3Pt (see T3FeaturesSoA.h)
  T3FeaturesDeviceCollection makeT3Features(Queue& queue,
                                            LSTInputDeviceCollection const& lstInput,
                                            ObjectRangesDeviceCollection const& ranges,
                                            MiniDoubletsDeviceCollection const& miniDoublets,
                                            SegmentsDeviceCollection const& segments,
                                            TripletsDeviceCollection const& triplets,
                                            float maxT3Pt);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif

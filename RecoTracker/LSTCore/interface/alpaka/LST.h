#ifndef RecoTracker_LSTCore_interface_alpaka_LST_h
#define RecoTracker_LSTCore_interface_alpaka_LST_h

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/LSTESData.h"
#include "RecoTracker/LSTCore/interface/alpaka/LSTInputDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/MiniDoubletsDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/ObjectRangesDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/SegmentsDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/TrackCandidatesDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/TripletsDeviceCollection.h"

#include <cstdlib>
#include <numeric>
#include <alpaka/alpaka.hpp>

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  class LSTEvent;

  class LST {
  public:
    LST() = default;

    void run(Queue& queue,
             bool verbose,
             const float ptCut,
             const uint16_t clustSizeCut,
             LSTESData<Device> const* deviceESData,
             LSTInputDeviceCollection const* lstInputDC,
             bool no_pls_dupclean,
             bool tc_pls_triplets,
             bool reduce_mem_by_full_precompute);
    std::unique_ptr<TrackCandidatesBaseDeviceCollection> getTrackCandidates() {
      return std::move(trackCandidatesBaseDC_);
    }
    // T3-level intermediate collections (and the ranges/MDs/LSs they index into)
    std::unique_ptr<ObjectRangesDeviceCollection> getRanges() { return std::move(rangesDC_); }
    std::unique_ptr<MiniDoubletsDeviceCollection> getMiniDoublets() { return std::move(miniDoubletsDC_); }
    std::unique_ptr<SegmentsDeviceCollection> getSegments() { return std::move(segmentsDC_); }
    std::unique_ptr<TripletsDeviceCollection> getTriplets() { return std::move(tripletsDC_); }

  private:
    // Output collection
    std::unique_ptr<TrackCandidatesBaseDeviceCollection> trackCandidatesBaseDC_;
    std::unique_ptr<ObjectRangesDeviceCollection> rangesDC_;
    std::unique_ptr<MiniDoubletsDeviceCollection> miniDoubletsDC_;
    std::unique_ptr<SegmentsDeviceCollection> segmentsDC_;
    std::unique_ptr<TripletsDeviceCollection> tripletsDC_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif

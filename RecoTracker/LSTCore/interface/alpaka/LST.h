#ifndef RecoTracker_LSTCore_interface_alpaka_LST_h
#define RecoTracker_LSTCore_interface_alpaka_LST_h

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/LSTESData.h"
#include "RecoTracker/LSTCore/interface/alpaka/LSTInputDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/TrackCandidatesDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/TrackCandidatesBLFFitDeviceCollection.h"

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
             bool reduce_mem_by_full_precompute,
             const float bField);
    std::unique_ptr<TrackCandidatesBaseDeviceCollection> getTrackCandidates() {
      return std::move(trackCandidatesBaseDC_);
    }
    std::unique_ptr<TrackCandidatesBLFFitDeviceCollection> getTrackCandidatesBLFFit() {
      return std::move(trackCandidatesBLFFitDC_);
    }

  private:
    // Output collections
    std::unique_ptr<TrackCandidatesBaseDeviceCollection> trackCandidatesBaseDC_;
    std::unique_ptr<TrackCandidatesBLFFitDeviceCollection> trackCandidatesBLFFitDC_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif

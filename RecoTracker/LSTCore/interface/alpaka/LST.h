#ifndef RecoTracker_LSTCore_interface_alpaka_LST_h
#define RecoTracker_LSTCore_interface_alpaka_LST_h

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/LSTESData.h"
#include "RecoTracker/LSTCore/interface/alpaka/LSTInputDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/TrackCandidatesDeviceCollection.h"
#ifndef LST_STANDALONE
#include "RecoTracker/LSTCore/interface/alpaka/TCSeedStatesDeviceCollection.h"
#endif

#include <cstdlib>
#include <numeric>
#include <alpaka/alpaka.hpp>

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  class LSTEvent;

  class LST {
  public:
    LST() = default;

    // produceTCSeedStates: also compute the Broken Line Fit seed state of every track candidate (ignored in the
    // standalone build, which does not compute them). It requires an LST input filled on the device: the fit reads
    // hits.cov, the field scalar B and the hitsIT positions and covariances, which lst::prepareInput leaves unfilled.
    void run(Queue& queue,
             bool verbose,
             const float ptCut,
             const uint16_t clustSizeCut,
             LSTESData<Device> const* deviceESData,
             LSTInputDeviceCollection const* lstInputDC,
             bool no_pls_dupclean,
             bool tc_pls_triplets,
             bool produceTCSeedStates = false);
    std::unique_ptr<TrackCandidatesBaseDeviceCollection> getTrackCandidates() {
      return std::move(trackCandidatesBaseDC_);
    }
#ifndef LST_STANDALONE
    // the seed states of the track candidates, one row per track candidate; null unless run computed them
    std::unique_ptr<TCSeedStatesDeviceCollection> getTCSeedStates() { return std::move(tcSeedStatesDC_); }
#endif

  private:
    // Output collection
    std::unique_ptr<TrackCandidatesBaseDeviceCollection> trackCandidatesBaseDC_;
#ifndef LST_STANDALONE
    std::unique_ptr<TCSeedStatesDeviceCollection> tcSeedStatesDC_;
#endif
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif

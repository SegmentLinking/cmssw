#ifndef RecoTracker_LSTCore_interface_alpaka_LST_h
#define RecoTracker_LSTCore_interface_alpaka_LST_h

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/LSTESData.h"
#include "RecoTracker/LSTCore/interface/alpaka/LSTInputDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/LSTOutputHostCollection.h"

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
             LSTESData<Device> const* deviceESData,
             LSTInputDeviceCollection const* lstInputDC,
             bool no_pls_dupclean,
             bool tc_pls_triplets);
    std::unique_ptr<LSTOutputHostCollection> getOutput() { return std::move(outputHC_); }

  private:
    void makeOutput(LSTEvent& event, Queue& queue);

    // Output SoA
    std::unique_ptr<LSTOutputHostCollection> outputHC_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif

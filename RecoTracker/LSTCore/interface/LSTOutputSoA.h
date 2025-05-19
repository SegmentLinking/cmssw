#ifndef RecoTracker_LSTCore_interface_LSTOutputSoA_h
#define RecoTracker_LSTCore_interface_LSTOutputSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Portable/interface/PortableCollection.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {

  GENERATE_SOA_LAYOUT(OutputSoALayout,
                      SOA_COLUMN(ArrayUxMaxTrackHits, hitIdx),
                      SOA_COLUMN(unsigned int, nHits),
                      SOA_COLUMN(int, seedIdx),
                      SOA_COLUMN(short, trackCandidateType))

  using OutputSoA = OutputSoALayout<>;

  using Output = OutputSoA::View;
  using OutputConst = OutputSoA::ConstView;

}  // namespace lst

#endif

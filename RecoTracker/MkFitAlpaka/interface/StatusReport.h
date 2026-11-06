#ifndef RecoTracker_MkFitAlpaka_interface_StatusReport_h
#define RecoTracker_MkFitAlpaka_interface_StatusReport_h

// Host helpers for the per-event status product. Host only.
//   warnIfNotClean(status, label)   production: edm::LogWarning("MkFitAlpakaStatus") listing the non-zero counters
//   assertClean(status, label)      test / menu checks: throws cms::Exception("MkFitAlpakaStatus")

#include <sstream>
#include <string>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTracker/MkFitAlpaka/interface/StatusProduct.h"

namespace mkfitdev {

  inline bool isClean(MkFitStatus const& status) {
    for (int i = 0; i < kMaxStatusCounters; ++i)
      if (status.counter[i] != 0)
        return false;
    return true;
  }

  // "name=value name=value" of the non-zero counters
  inline std::string describeStatus(MkFitStatus const& status) {
    std::ostringstream out;
    for (int i = 0; i < kMaxStatusCounters; ++i)
      if (status.counter[i] != 0)
        out << (out.tellp() > 0 ? " " : "") << statusCounterName(i) << "[" << i << "]=" << status.counter[i];
    return out.str();
  }

  // returns true if clean
  inline bool warnIfNotClean(MkFitStatus const& status, std::string const& label) {
    if (isClean(status))
      return true;
    edm::LogWarning("MkFitAlpakaStatus") << label << ": device chain status not clean (output differs from MkFitCore "
                                         << "for a known reason): " << describeStatus(status);
    return false;
  }

  inline void assertClean(MkFitStatus const& status, std::string const& label) {
    if (!isClean(status))
      throw cms::Exception("MkFitAlpakaStatus")
          << label << ": device chain status not clean: " << describeStatus(status);
  }

}  // namespace mkfitdev

#endif

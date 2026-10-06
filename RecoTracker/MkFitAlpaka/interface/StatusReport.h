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

  inline bool isClean(MkFitStatus const& s) {
    for (int i = 0; i < kMaxStatusCounters; ++i)
      if (s.counter[i] != 0)
        return false;
    return true;
  }

  // "name=value name=value" of the non-zero counters
  inline std::string describeStatus(MkFitStatus const& s) {
    std::ostringstream o;
    for (int i = 0; i < kMaxStatusCounters; ++i)
      if (s.counter[i] != 0)
        o << (o.tellp() > 0 ? " " : "") << statusCounterName(i) << "[" << i << "]=" << s.counter[i];
    return o.str();
  }

  // returns true if clean
  inline bool warnIfNotClean(MkFitStatus const& s, std::string const& label) {
    if (isClean(s))
      return true;
    edm::LogWarning("MkFitAlpakaStatus") << label << ": device chain status not clean (output differs from MkFitCore "
                                         << "for a known reason): " << describeStatus(s);
    return false;
  }

  inline void assertClean(MkFitStatus const& s, std::string const& label) {
    if (!isClean(s))
      throw cms::Exception("MkFitAlpakaStatus") << label << ": device chain status not clean: " << describeStatus(s);
  }

}  // namespace mkfitdev

#endif

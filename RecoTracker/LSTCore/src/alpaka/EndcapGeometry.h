#ifndef EndcapGeometry_h
#define EndcapGeometry_h

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#else
#include "Constants.h"
#endif

namespace SDL {
  template <typename>
  class EndcapGeometry;
  template <>
  class EndcapGeometry<SDL::Dev> {
  private:
    std::map<unsigned int, float> dxdy_slope_;     // dx/dy slope
    std::map<unsigned int, float> centroid_phis_;  // centroid phi

  public:
    std::vector<unsigned int> geoMapDetId_buf;
    std::vector<float> geoMapPhi_buf;

    unsigned int nEndCapMap;

    EndcapGeometry() = default;
    EndcapGeometry(std::string filename);
    ~EndcapGeometry() = default;

    void load(std::string);
    void fillGeoMapArraysExplicit();
    float getdxdy_slope(unsigned int detid) const;
  };
}  // namespace SDL

#endif

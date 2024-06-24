#ifndef EndcapGeometryBuffers_h
#define EndcapGeometryBuffers_h

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"

namespace SDL {

  struct endcapGeom {
    const unsigned int* geoMapDetId;
    const float* geoMapPhi;

    template <typename TBuff>
    void setData(const TBuff& endcapgeombuf) {
      geoMapDetId = alpaka::getPtrNative(endcapgeombuf.geoMapDetId_buf);
      geoMapPhi = alpaka::getPtrNative(endcapgeombuf.geoMapPhi_buf);
    }
  };

  template <typename TDev>
  struct endcapGeometryBuffer : endcapGeom {
    Buf<TDev, unsigned int> geoMapDetId_buf;
    Buf<TDev, float> geoMapPhi_buf;

    endcapGeometryBuffer(TDev const& dev, unsigned int nEndCapMap)
        : geoMapDetId_buf(allocBufWrapper<unsigned int>(dev, nEndCapMap)),
          geoMapPhi_buf(allocBufWrapper<float>(dev, nEndCapMap)) {
      setData(*this);
    }

    template <typename TQueue, typename TDevSrc>
    inline void copyFromSrc(TQueue queue, const endcapGeometryBuffer<TDevSrc>& src) {
      alpaka::memcpy(queue, geoMapDetId_buf, src.geoMapDetId_buf);
      alpaka::memcpy(queue, geoMapPhi_buf, src.geoMapPhi_buf);
    }

    template <typename TQueue, typename TDevSrc>
    endcapGeometryBuffer(TQueue queue, const endcapGeometryBuffer<TDevSrc>& src, unsigned int nEndCapMap)
        : endcapGeometryBuffer(alpaka::getDev(queue), nEndCapMap) {
      copyFromSrc(queue, src);
    }

    inline SDL::endcapGeom const* data() const { return this; }
  };

}  // namespace SDL

#endif

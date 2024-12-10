#ifndef RecoTracker_LSTCore_interface_PT2sSoA_h
#define RecoTracker_LSTCore_interface_PT2sSoA_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {
  // One pixel segment, one outer tracker triplet!
  GENERATE_SOA_LAYOUT(PT2sSoALayout,
                      SOA_COLUMN(unsigned int, pixelSegmentIndices),
                      SOA_COLUMN(unsigned int, segmentIndices),
                      SOA_COLUMN(FPX, pixelRadius),
                      SOA_COLUMN(FPX, quintupletRadius),
                      SOA_COLUMN(FPX, pt),
                      SOA_COLUMN(FPX, eta),
                      SOA_COLUMN(FPX, phi),
                      SOA_COLUMN(FPX, eta_pix),
                      SOA_COLUMN(FPX, phi_pix),
                      SOA_COLUMN(bool, isDup),
                      SOA_COLUMN(bool, partOfPT3),
                      SOA_COLUMN(bool, partOfPT5),
                      SOA_COLUMN(Params_pT2::ArrayU8xLayers, logicalLayers),
                      SOA_COLUMN(Params_pT2::ArrayUxHits, hitIndices),
                      SOA_COLUMN(Params_pT2::ArrayU16xLayers, lowerModuleIndices),
                      SOA_COLUMN(FPX, centerX),
                      SOA_COLUMN(FPX, centerY),
                      SOA_COLUMN(float, pixelRadiusError),
                      SOA_COLUMN(float, rzChiSquared),
                      SOA_SCALAR(unsigned int, nPT2s),
                      SOA_SCALAR(unsigned int, totOccupancyPT2s));

  using PT2sSoA = PT2sSoALayout<>;
  using PT2s = PT2sSoA::View;
  using PT2sConst = PT2sSoA::ConstView;
}  // namespace lst
#endif
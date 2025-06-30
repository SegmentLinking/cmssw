#ifndef RecoTracker_LSTCore_interface_PixelQuadrupletsSoA_h
#define RecoTracker_LSTCore_interface_PixelQuadrupletsSoA_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {
  GENERATE_SOA_LAYOUT(PixelQuadrupletsSoALayout,
                      SOA_COLUMN(unsigned int, pixelSegmentIndices),
                      SOA_COLUMN(unsigned int, quadrupletIndices),
                      SOA_COLUMN(Params_pT4::ArrayU16xLayers, lowerModuleIndices),  // lower module index (OT part)
                      SOA_COLUMN(Params_pT4::ArrayU8xLayers, logicalLayers),        // layer ID
                      SOA_COLUMN(Params_pT4::ArrayUxHits, hitIndices),              // hit indices
                      SOA_COLUMN(float, rPhiChiSquared),                            // chi2 from pLS to T4
                      SOA_COLUMN(float, rPhiChiSquaredInwards),                     // chi2 from T4 to pLS
                      SOA_COLUMN(float, rzChiSquared),
                      SOA_COLUMN(FPX, pixelRadius),       // pLS pt converted
                      SOA_COLUMN(FPX, pixelRadiusError),
                      SOA_COLUMN(FPX, quadrupletRadius),  // T4 circle
                      SOA_COLUMN(FPX, eta),
                      SOA_COLUMN(FPX, phi),
                      SOA_COLUMN(float, pt),
                      SOA_COLUMN(FPX, score),    // used for ranking (in e.g. duplicate cleaning)
                      SOA_COLUMN(FPX, centerX),  // T3-based circle center x
                      SOA_COLUMN(FPX, centerY),  // T3-based circle center y
                      SOA_COLUMN(bool, isDup),
                      SOA_SCALAR(unsigned int, nPixelQuadruplets),
                      SOA_SCALAR(unsigned int, totOccupancyPixelQuadruplets));

  using PixelQuadrupletsSoA = PixelQuadrupletsSoALayout<>;
  using PixelQuadruplets = PixelQuadrupletsSoA::View;
  using PixelQuadrupletsConst = PixelQuadrupletsSoA::ConstView;
}  // namespace lst
#endif

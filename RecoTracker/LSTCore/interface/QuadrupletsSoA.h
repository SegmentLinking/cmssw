#ifndef RecoTracker_LSTCore_interface_QuadrupletsSoA_h
#define RecoTracker_LSTCore_interface_QuadrupletsSoA_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {
  GENERATE_SOA_LAYOUT(QuadrupletsSoALayout,
                      SOA_COLUMN(ArrayUx2, tripletIndices),                        // inner and outer triplet indices
                      SOA_COLUMN(Params_T4::ArrayU16xLayers, lowerModuleIndices),  // lower module index in each layer
                      SOA_COLUMN(Params_T4::ArrayU8xLayers, logicalLayers),        // layer ID
                      SOA_COLUMN(Params_T4::ArrayUxHits, hitIndices),              // hit indices
                      SOA_COLUMN(FPX, innerRadius),                                // inner triplet circle radius
                      SOA_COLUMN(FPX, outerRadius),                                // outer triplet radius
                      SOA_COLUMN(FPX, pt),
                      SOA_COLUMN(FPX, eta),
                      SOA_COLUMN(FPX, phi),
                      SOA_COLUMN(FPX, score_rphisum),  // r-phi based score
                      SOA_COLUMN(char, isDup),         // duplicate flag
                      SOA_COLUMN(bool, tightCutFlag),  // tight pass to be a TC
                      SOA_COLUMN(bool, partOfPT4),
                      SOA_COLUMN(bool, partOfTC),
                      SOA_COLUMN(float, regressionRadius),
                      SOA_COLUMN(float, nonAnchorRegressionRadius),
                      SOA_COLUMN(float, regressionG),
                      SOA_COLUMN(float, regressionF),
                      SOA_COLUMN(float, rzChiSquared),  // r-z only chi2
                      SOA_COLUMN(float, chiSquared),
                      SOA_COLUMN(float, nonAnchorChiSquared),
//mainly for debug, can probably be removed
                      SOA_COLUMN(float, promptscore_t4dnn),
                      SOA_COLUMN(float, displacedscore_t4dnn),
                      SOA_COLUMN(float, fakescore_t4dnn),
                      // SOA_COLUMN(float, uncertainty),
                      SOA_COLUMN(bool, tightDNNFlag),
                      SOA_COLUMN(int, layer),
//
                      SOA_COLUMN(float, dBeta));

  using QuadrupletsSoA = QuadrupletsSoALayout<>;
  using Quadruplets = QuadrupletsSoA::View;
  using QuadrupletsConst = QuadrupletsSoA::ConstView;

  GENERATE_SOA_LAYOUT(QuadrupletsOccupancySoALayout,
                      SOA_COLUMN(unsigned int, nQuadruplets),
                      SOA_COLUMN(unsigned int, totOccupancyQuadruplets));

  using QuadrupletsOccupancySoA = QuadrupletsOccupancySoALayout<>;
  using QuadrupletsOccupancy = QuadrupletsOccupancySoA::View;
  using QuadrupletsOccupancyConst = QuadrupletsOccupancySoA::ConstView;

}  // namespace lst
#endif

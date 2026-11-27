#ifndef RecoLocalTracker_SiPixelRecHits_interface_PixelGenErrorTablesSoA_h
#define RecoLocalTracker_SiPixelRecHits_interface_PixelGenErrorTablesSoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

// Data for generic CPE (with track angles) as used in pixelCPEforDeviceTrackAngles.h:
// GenError store index and local details per module

// Per module: its GenError store and the local magnetic field
GENERATE_SOA_LAYOUT(PixelGenErrorModuleSoALayout,
                    SOA_COLUMN(int32_t, genErrorStore),  // -1: no GenError for this module
                    SOA_COLUMN(float, localBx),          // tesla, at the module centre
                    SOA_COLUMN(float, localBz))

// Per GenError store: the SiPixelGenErrorHeader values that SiPixelGenError::qbin uses, and where its tables start in
// the pool
GENERATE_SOA_LAYOUT(PixelGenErrorStoreSoALayout,
                    SOA_COLUMN(int32_t, detectorType),  // SiPixelGenErrorHeader::Dtype
                    SOA_COLUMN(int32_t, nCotBetaY),     // NTy
                    SOA_COLUMN(int32_t, nCotBetaX),     // NTyx
                    SOA_COLUMN(int32_t, nCotAlphaX),    // NTxx
                    SOA_COLUMN(float, qscale),
                    SOA_COLUMN(float, fbin0),
                    SOA_COLUMN(float, fbin1),
                    SOA_COLUMN(float, fbin2),
                    SOA_COLUMN(float, cotAlpha0),          // enty[0].cotalpha
                    SOA_COLUMN(int32_t, cotBetaYOffset),   // into pool: cotbetaY[nCotBetaY]
                    SOA_COLUMN(int32_t, cotBetaXOffset),   // cotbetaX[nCotBetaX]
                    SOA_COLUMN(int32_t, cotAlphaXOffset),  // cotalphaX[nCotAlphaX]
                    SOA_COLUMN(int32_t, yEntryOffset),     // nCotBetaY x {qavg, syone, yrmsgen[4]}
                    SOA_COLUMN(int32_t, xEntryOffset),     // nCotBetaX x nCotAlphaX x xrmsgen[4]
                    SOA_COLUMN(int32_t, singleXOffset))    // nCotAlphaX x sxone of the first cotbetaX slice

// The GenError tables of all stores, one after the other
GENERATE_SOA_LAYOUT(PixelGenErrorPoolSoALayout, SOA_COLUMN(float, value))

// For the whole job: the cuts of the position formula and the errors of edge clusters, as configured for
// PixelCPEGeneric
GENERATE_SOA_LAYOUT(PixelGenErrorConfigSoALayout,
                    SOA_SCALAR(float, effChargeCutLowX),
                    SOA_SCALAR(float, effChargeCutLowY),
                    SOA_SCALAR(float, effChargeCutHighX),
                    SOA_SCALAR(float, effChargeCutHighY),
                    SOA_SCALAR(float, sizeCutX),
                    SOA_SCALAR(float, sizeCutY),
                    SOA_SCALAR(float, edgeClusterErrorX),  // microns
                    SOA_SCALAR(float, edgeClusterErrorY))

// All in one buffer
GENERATE_SOA_BLOCKS(PixelGenErrorTablesSoALayout,
                    SOA_BLOCK(modules, PixelGenErrorModuleSoALayout),
                    SOA_BLOCK(stores, PixelGenErrorStoreSoALayout),
                    SOA_BLOCK(pool, PixelGenErrorPoolSoALayout),
                    SOA_BLOCK(config, PixelGenErrorConfigSoALayout))

using PixelGenErrorTablesSoA = PixelGenErrorTablesSoALayout<>;
using PixelGenErrorTablesSoAView = PixelGenErrorTablesSoA::View;
using PixelGenErrorTablesSoAConstView = PixelGenErrorTablesSoA::ConstView;

namespace pixelGenErrorTables {
  constexpr int kYEntrySize = 6;  // qavg, syone, yrmsgen[4]
  constexpr int kXEntrySize = 4;  // xrmsgen[4]
}  // namespace pixelGenErrorTables

#endif  // RecoLocalTracker_SiPixelRecHits_interface_PixelGenErrorTablesSoA_h

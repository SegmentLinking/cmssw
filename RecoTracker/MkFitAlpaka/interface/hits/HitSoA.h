#ifndef RecoTracker_MkFitAlpaka_interface_hits_HitSoA_h
#define RecoTracker_MkFitAlpaka_interface_hits_HitSoA_h

#include <cstdint>
#include <limits>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkfitdev {

  // One row per input hit: the pixel MkFitHitWrapper hits first (rows [0, nPixel)), then the strip/OT wrapper hits
  // (rows [nPixel, nPixel + nStrip)). Position and error as mkfit::Hit; e.. is SMatrixSym33 in packed order
  // (00, 10, 11, 20, 21, 22). packed = Hit::PackedData bits (see hitpack below). The original hit index of a row
  // (its index in its own wrapper) is originalIndex(row, nPixel).
  GENERATE_SOA_LAYOUT(HitSoALayout,
                      SOA_COLUMN(float, x),
                      SOA_COLUMN(float, y),
                      SOA_COLUMN(float, z),
                      SOA_COLUMN(float, e00),
                      SOA_COLUMN(float, e10),
                      SOA_COLUMN(float, e11),
                      SOA_COLUMN(float, e20),
                      SOA_COLUMN(float, e21),
                      SOA_COLUMN(float, e22),
                      SOA_COLUMN(uint32_t, packed),
                      SOA_COLUMN(int8_t, layer),  // mkFit layer of the hit, -1 if not registered
                      SOA_SCALAR(uint32_t, nPixel),
                      SOA_SCALAR(uint32_t, nStrip))

  using HitSoA = HitSoALayout<>;
  using HitSoAView = HitSoA::View;
  using HitSoAConstView = HitSoA::ConstView;

  // number of mkFit layers the int8_t layer column can hold
  constexpr uint32_t kMaxLayers = std::numeric_limits<int8_t>::max();

  // MkFitCore original hit index (index in its own wrapper) of a HitSoA row
  constexpr uint32_t originalIndex(uint32_t row, uint32_t nPixel) { return row < nPixel ? row : row - nPixel; }

  // Hit::PackedData bit-field layout (gcc, LSB first): detid_in_layer 14, charge_pcm 8, span_rows 5, span_cols 5.
  namespace hitpack {
    constexpr uint32_t kDetIdBits = 14, kChargeBits = 8, kSpanBits = 5;
    constexpr uint32_t kChargeShift = kDetIdBits, kRowsShift = kDetIdBits + kChargeBits,
                       kColsShift = kDetIdBits + kChargeBits + kSpanBits;
    constexpr uint32_t pack(uint32_t detid, uint32_t chargeRaw, uint32_t rowsRaw, uint32_t colsRaw) {
      return (detid & ((1u << kDetIdBits) - 1)) | ((chargeRaw & ((1u << kChargeBits) - 1)) << kChargeShift) |
             ((rowsRaw & ((1u << kSpanBits) - 1)) << kRowsShift) | ((colsRaw & ((1u << kSpanBits) - 1)) << kColsShift);
    }
    constexpr uint32_t detIDinLayer(uint32_t p) { return p & ((1u << kDetIdBits) - 1); }
    constexpr uint32_t chargeRaw(uint32_t p) { return (p >> kChargeShift) & ((1u << kChargeBits) - 1); }
    // as Hit::chargePerCM(), spanRows(), spanCols()
    constexpr uint32_t chargePerCM(uint32_t p) { return chargeRaw(p) == 0 ? 0 : ((chargeRaw(p) - 1) << 3) + 1620; }
    constexpr uint32_t spanRows(uint32_t p) { return ((p >> kRowsShift) & ((1u << kSpanBits) - 1)) + 1; }
    constexpr uint32_t spanCols(uint32_t p) { return ((p >> kColsShift) & ((1u << kSpanBits) - 1)) + 1; }
  }  // namespace hitpack

}  // namespace mkfitdev

#endif

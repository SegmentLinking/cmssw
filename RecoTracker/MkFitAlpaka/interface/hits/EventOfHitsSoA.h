#ifndef RecoTracker_MkFitAlpaka_interface_hits_EventOfHitsSoA_h
#define RecoTracker_MkFitAlpaka_interface_hits_EventOfHitsSoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkfitdev {

  // LayerOfHits binning: binnor<unsigned int, axis_pow2_u1<float, ushort, 16, 8> (phi),
  // axis<float, ushort, 16, 8> (q = z in barrel, r in endcap), 18, 14>.
  constexpr uint32_t kPhiBitsM = 16, kPhiBitsN = 8, kQBitsM = 16, kQBitsN = 8;
  constexpr uint32_t kNPhiBins = 1u << kPhiBitsN;  // 256 normal phi bins
  constexpr uint32_t kPhiMaskM = (1u << kPhiBitsM) - 1, kPhiMaskN = (1u << kPhiBitsN) - 1;
  constexpr uint32_t kBinFirstBits = 18, kBinCountBits = 14;  // C_pair bit-fields
  constexpr uint32_t kBinFirstMask = (1u << kBinFirstBits) - 1, kBinCountMask = (1u << kBinCountBits) - 1;

  // Per layer (row = mkFit layer index). Axis constants are copied from the MkFitCore axis objects built on the host.
  GENERATE_SOA_LAYOUT(LayerSoALayout,
                      SOA_COLUMN(float, qRMin),         // axis_base::m_R_min
                      SOA_COLUMN(float, qRMax),         // m_R_max
                      SOA_COLUMN(float, qMFac),         // m_M_fac
                      SOA_COLUMN(float, qNFac),         // m_N_fac
                      SOA_COLUMN(float, qMLbhp),        // m_M_lbhp
                      SOA_COLUMN(float, qNLbhp),        // m_N_lbhp
                      SOA_COLUMN(uint16_t, qLastMBin),  // m_last_M_bin
                      SOA_COLUMN(uint16_t, qLastNBin),  // m_last_N_bin
                      SOA_COLUMN(uint32_t, nQBins),     // size_of_N() of the q axis
                      SOA_COLUMN(uint8_t, isBarrel),
                      SOA_COLUMN(uint8_t, isPixel),
                      SOA_COLUMN(uint32_t, binBegin),  // first bin-table row of the layer (nQBins * 256 rows)
                      SOA_COLUMN(uint32_t, hitBase),  // first HitSoA row of the layer's wrapper (0 pixel, nPixel strip)
                      SOA_COLUMN(uint32_t, hitBegin),  // first BinnedHitSoA row of the layer (device output)
                      SOA_COLUMN(uint32_t, nHits),     // registered hits of the layer (device output)
                      // phi axis, identical for all layers (axis_pow2_u1(-PI, PI))
                      SOA_SCALAR(float, phiRMin),
                      SOA_SCALAR(float, phiMFac),
                      SOA_SCALAR(float, phiNFac),
                      SOA_SCALAR(uint32_t, nBinsTotal),
                      // overflow counters (MkFitCore bit-fields would wrap silently): first >= 2^18, count >= 2^14
                      SOA_SCALAR(uint32_t, nOverflowFirst),
                      SOA_SCALAR(uint32_t, nOverflowCount))

  // One row per registered hit; layers concatenated in layer order, each layer in LayerOfHits internal order.
  // Row hitBegin[l] + i  <->  MkFitCore eoh[l] internal index i.
  GENERATE_SOA_LAYOUT(BinnedHitSoALayout,
                      SOA_COLUMN(uint32_t, rank),      // LayerOfHits::getOriginalHitIndex(i)
                      SOA_COLUMN(float, phi),          // LayerOfHits::hit_phi(i)
                      SOA_COLUMN(float, q),            // hit_q(i)
                      SOA_COLUMN(float, qHalfLength),  // hit_q_half_length(i)
                      SOA_COLUMN(float, qbar))         // hit_qbar(i)

  // One row per (layer, q bin, phi bin): row = binBegin[layer] + qBin * 256 + phiBin (m_bins / m_dead_bins order).
  GENERATE_SOA_LAYOUT(BinSoALayout,
                      SOA_COLUMN(uint32_t, content),  // C_pair: first (internal index in layer) | count << 18
                      SOA_COLUMN(uint8_t, dead))      // m_dead_bins

  using LayerSoA = LayerSoALayout<>;
  using BinnedHitSoA = BinnedHitSoALayout<>;
  using BinSoA = BinSoALayout<>;

  constexpr uint32_t binFirst(uint32_t content) { return content & kBinFirstMask; }
  constexpr uint32_t binCount(uint32_t content) { return content >> kBinFirstBits; }

  // Dead region of a layer, as mkfit::DeadRegion plus the layer index.
  struct DeadRegionDev {
    float phi1, phi2, q1, q2;
    int32_t layer;
  };

}  // namespace mkfitdev

#endif

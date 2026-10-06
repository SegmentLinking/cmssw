#ifndef RecoTracker_MkFitAlpaka_interface_hits_LayerOfHitsAccess_h
#define RecoTracker_MkFitAlpaka_interface_hits_LayerOfHitsAccess_h

// Device-side equivalent of the mkfit::LayerOfHits query API used by hit selection (MkFinder), on top of the
// device EventOfHits SoAs. Same bin arithmetic as MkFitCore (binnor.h axis classes).
//   internal index i (0 <= i < nHits())  <->  LayerOfHits internal index i
//   getOriginalHitIndex(i)                <->  getOriginalHitIndex(i) (index in the layer's MkFitHitWrapper)
//   hitRow(orig)                          :   HitSoA row of that hit (refHit(orig))

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "RecoTracker/MkFitAlpaka/interface/hits/EventOfHitsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitsMath.h"

namespace mkfitdev {

  struct LayerOfHitsAccess {
    using bin_index_t = uint16_t;

    LayerSoA::ConstView layers;
    BinnedHitSoA::ConstView binned;
    BinSoA::ConstView bins;
    int32_t layer;

    ALPAKA_FN_HOST_ACC uint32_t nHits() const { return layers[layer].nHits(); }
    ALPAKA_FN_HOST_ACC bool is_barrel() const { return layers[layer].isBarrel(); }
    ALPAKA_FN_HOST_ACC bool is_pixel() const { return layers[layer].isPixel(); }

    // axis_base::from_R_to_N_bin / from_R_to_N_bin_safe for q; axis_pow2_u1 versions for phi
    ALPAKA_FN_HOST_ACC bin_index_t qBin(float q) const {
      return hitsmath::toBin((q - layers[layer].qRMin()) * layers[layer].qNFac());
    }
    ALPAKA_FN_HOST_ACC bin_index_t qBinChecked(float q) const {
      return q <= layers[layer].qRMin() ? bin_index_t(0)
                                        : (q >= layers[layer].qNLbhp() ? layers[layer].qLastNBin() : qBin(q));
    }
    ALPAKA_FN_HOST_ACC bin_index_t phiBin(float phi) const {
      return hitsmath::toBin((phi - layers.phiRMin()) * layers.phiNFac());
    }
    ALPAKA_FN_HOST_ACC bin_index_t phiBinChecked(float phi) const { return phiBin(phi) & kPhiMaskN; }
    ALPAKA_FN_HOST_ACC bin_index_t phiMaskApply(bin_index_t in) const { return in & kPhiMaskN; }
    ALPAKA_FN_HOST_ACC uint32_t nQBins() const { return layers[layer].nQBins(); }

    // phiQBinContent(pi, qi): {first, count}
    ALPAKA_FN_HOST_ACC uint32_t binContent(bin_index_t pi, bin_index_t qi) const {
      return bins[layers[layer].binBegin() + uint32_t(qi) * kNPhiBins + pi].content();
    }
    ALPAKA_FN_HOST_ACC bool isBinDead(bin_index_t pi, bin_index_t qi) const {
      return bins[layers[layer].binBegin() + uint32_t(qi) * kNPhiBins + pi].dead() != 0;
    }

    // hit infos and index maps by internal index
    ALPAKA_FN_HOST_ACC uint32_t row(uint32_t i) const { return layers[layer].hitBegin() + i; }
    ALPAKA_FN_HOST_ACC float hit_phi(uint32_t i) const { return binned[row(i)].phi(); }
    ALPAKA_FN_HOST_ACC float hit_q(uint32_t i) const { return binned[row(i)].q(); }
    ALPAKA_FN_HOST_ACC float hit_q_half_length(uint32_t i) const { return binned[row(i)].qHalfLength(); }
    ALPAKA_FN_HOST_ACC float hit_qbar(uint32_t i) const { return binned[row(i)].qbar(); }
    ALPAKA_FN_HOST_ACC uint32_t getOriginalHitIndex(uint32_t i) const { return binned[row(i)].rank(); }
    ALPAKA_FN_HOST_ACC uint32_t hitRow(uint32_t orig) const { return layers[layer].hitBase() + orig; }
  };

}  // namespace mkfitdev

#endif

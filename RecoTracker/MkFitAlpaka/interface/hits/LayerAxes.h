#ifndef RecoTracker_MkFitAlpaka_interface_hits_LayerAxes_h
#define RecoTracker_MkFitAlpaka_interface_hits_LayerAxes_h

// Host-only: static layer table (q/phi axes) built with the MkFitCore binnor axis classes, so the constants are
// identical.

#include <vector>

#include "RecoTracker/MkFitCore/interface/binnor.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/EventOfHitsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"

namespace mkfitdev {

  using AxisPhi = mkfit::axis_pow2_u1<float, unsigned short, 16, 8>;  // mkfit::LayerOfHits::axis_phi_t
  using AxisQ = mkfit::axis<float, unsigned short, 16, 8>;            // mkfit::LayerOfHits::axis_eta_t

  // Per-layer q-axis inputs as LayerOfHits::Initializator computes them.
  struct LayerAxisInput {
    float qmin, qmax;
    unsigned int nq;
    bool isBarrel, isPixel;
  };

  // total bin-table rows of all layers
  inline uint32_t totalBins(const std::vector<LayerAxisInput>& in) {
    uint32_t nBins = 0;
    for (auto const& layer : in)
      nBins += layer.nq * kNPhiBins;
    return nBins;
  }

  // Fill the static part of the layer table and clear the per-event outputs (hitBegin, nHits, overflow counters).
  inline void fillLayers(const std::vector<LayerAxisInput>& in, uint32_t nPixelHits, LayerSoA::View view) {
    // same types as mkfit::LayerOfHits::axis_phi_t / axis_eta_t; -PI, PI = mkfit::Const::PI
    AxisPhi aphi(-Const::PI, Const::PI);
    view.phiRMin() = aphi.m_R_min;
    view.phiMFac() = aphi.m_M_fac;
    view.phiNFac() = aphi.m_N_fac;
    uint32_t binBegin = 0;
    for (size_t il = 0; il < in.size(); ++il) {
      AxisQ aq(in[il].qmin, in[il].qmax, in[il].nq);
      auto row = view[il];
      row.qRMin() = aq.m_R_min;
      row.qRMax() = aq.m_R_max;
      row.qMFac() = aq.m_M_fac;
      row.qNFac() = aq.m_N_fac;
      row.qMLbhp() = aq.m_M_lbhp;
      row.qNLbhp() = aq.m_N_lbhp;
      row.qLastMBin() = aq.m_last_M_bin;
      row.qLastNBin() = aq.m_last_N_bin;
      row.nQBins() = aq.size_of_N();
      row.isBarrel() = in[il].isBarrel;
      row.isPixel() = in[il].isPixel;
      row.binBegin() = binBegin;
      row.hitBase() = in[il].isPixel ? 0 : nPixelHits;
      row.hitBegin() = 0;
      row.nHits() = 0;
      binBegin += aq.size_of_N() * kNPhiBins;
    }
    view.nBinsTotal() = binBegin;
    view.nOverflowFirst() = 0;
    view.nOverflowCount() = 0;
  }

}  // namespace mkfitdev

#endif

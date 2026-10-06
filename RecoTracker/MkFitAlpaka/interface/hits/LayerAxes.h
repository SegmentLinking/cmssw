#ifndef RecoTracker_MkFitAlpaka_interface_hits_LayerAxes_h
#define RecoTracker_MkFitAlpaka_interface_hits_LayerAxes_h

// Host-only: static layer table (q/phi axes) built with the MkFitCore binnor axis classes, so the constants are
// identical.

#include <vector>

#include "RecoTracker/MkFitCore/interface/binnor.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/EventOfHitsHostCollections.h"
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

  inline uint32_t totalBins(const std::vector<LayerAxisInput>& in) {
    uint32_t n = 0;
    for (auto const& l : in)
      n += l.nq * kNPhiBins;
    return n;
  }

  // Fill the static part of the layer table (axes built with the MkFitCore axis classes, so the constants are
  // identical).
  inline void fillLayers(const std::vector<LayerAxisInput>& in, uint32_t nPixelHits, LayerSoA::View v) {
    // same types as mkfit::LayerOfHits::axis_phi_t / axis_eta_t; -PI, PI = mkfit::Const::PI
    AxisPhi aphi(-Const::PI, Const::PI);
    v.phiRMin() = aphi.m_R_min;
    v.phiMFac() = aphi.m_M_fac;
    v.phiNFac() = aphi.m_N_fac;
    uint32_t binBegin = 0;
    for (size_t il = 0; il < in.size(); ++il) {
      AxisQ aq(in[il].qmin, in[il].qmax, in[il].nq);
      auto r = v[il];
      r.qRMin() = aq.m_R_min;
      r.qRMax() = aq.m_R_max;
      r.qMFac() = aq.m_M_fac;
      r.qNFac() = aq.m_N_fac;
      r.qMLbhp() = aq.m_M_lbhp;
      r.qNLbhp() = aq.m_N_lbhp;
      r.qLastMBin() = aq.m_last_M_bin;
      r.qLastNBin() = aq.m_last_N_bin;
      r.nQBins() = aq.size_of_N();
      r.isBarrel() = in[il].isBarrel;
      r.isPixel() = in[il].isPixel;
      r.binBegin() = binBegin;
      r.hitBase() = in[il].isPixel ? 0 : nPixelHits;
      r.hitBegin() = 0;
      r.nHits() = 0;
      binBegin += aq.size_of_N() * kNPhiBins;
    }
    v.nBinsTotal() = binBegin;
    v.nOverflowFirst() = 0;
    v.nOverflowCount() = 0;
  }

}  // namespace mkfitdev

#endif

#ifndef RecoTracker_LSTCore_interface_T3FeaturesSoA_h
#define RecoTracker_LSTCore_interface_T3FeaturesSoA_h

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace lst {

  // Per-T3 input features of the transformer model, in the order of transformer-oc data/t3_processing.py:
  //   [0, 3)   T3: pt, eta, phi
  //   [3, 13)  LS0 then LS1: dPhi, dPhiChange, dAlphaInner, dAlphaOuter, dAlphaInnerOuter
  //   [13, 49) LS0.md0, LS0.md1, LS1.md0, LS1.md1: anchor x/y/z, other x/y/z, dphi, dphichange, dz
  // Values are raw (no log/min-max scaling). Rows are compact: only T3s with pt < maxT3Pt, ordered by
  // inner lower module and then by slot within the module (same order as the LST ntuple).
  namespace t3features {
    constexpr unsigned int kT3 = 0;
    constexpr unsigned int kNT3 = 3;
    constexpr unsigned int kLS = kT3 + kNT3;
    constexpr unsigned int kNLS = 5;
    constexpr unsigned int kMD = kLS + 2 * kNLS;
    constexpr unsigned int kNMD = 9;
    constexpr unsigned int kN = kMD + 4 * kNMD;
    static_assert(kN == 49);
  }  // namespace t3features

  using ArrayFxT3Features = edm::StdArray<float, t3features::kN>;

  GENERATE_SOA_LAYOUT(T3FeaturesSoALayout,
                      SOA_COLUMN(ArrayFxT3Features, features),  // row-major [nT3s, 49]
                      SOA_COLUMN(unsigned int, tripletIndex),   // index into the hltLST Triplets collection
                      SOA_SCALAR(unsigned int, nT3s))           // number of filled rows

  using T3FeaturesSoA = T3FeaturesSoALayout<>;
  using T3Features = T3FeaturesSoA::View;
  using T3FeaturesConst = T3FeaturesSoA::ConstView;

}  // namespace lst

#endif

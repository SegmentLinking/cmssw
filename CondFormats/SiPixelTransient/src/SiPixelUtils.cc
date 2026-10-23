#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "CondFormats/SiPixelTransient/interface/SiPixelUtils.h"
#else
#include "SiPixelUtils.h"
#endif

namespace siPixelUtils {

  float generic_position_formula(int size,
                                 int q_f,
                                 int q_l,
                                 float upper_edge_first_pix,
                                 float lower_edge_last_pix,
                                 float lorentz_shift,
                                 float theThickness,
                                 float cot_angle,
                                 float pitch,
                                 float pitchfraction_first,
                                 float pitchfraction_last,
                                 float eff_charge_cut_low,
                                 float eff_charge_cut_high,
                                 float size_cut) {
    return genericPositionFormula(size,
                                  q_f,
                                  q_l,
                                  upper_edge_first_pix,
                                  lower_edge_last_pix,
                                  lorentz_shift,
                                  theThickness,
                                  cot_angle,
                                  pitch,
                                  pitchfraction_first,
                                  pitchfraction_last,
                                  eff_charge_cut_low,
                                  eff_charge_cut_high,
                                  size_cut);
  }

}  // namespace siPixelUtils

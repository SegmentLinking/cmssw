#ifndef CondFormats_SiPixelTransient_SiPixelUtils_h
#define CondFormats_SiPixelTransient_SiPixelUtils_h

#include <cmath>

// constexpr functions are also callable on GPUs (CUDA relaxed constexpr, HIP)
namespace siPixelUtils {

  //-----------------------------------------------------------------------------
  //!  A generic version of the position formula.  Since it works for both
  //!  X and Y, in the interest of the simplicity of the code, all parameters
  //!  are passed by the caller.
  //-----------------------------------------------------------------------------
  constexpr float genericPositionFormula(int size,                    //!< Size of this projection.
                                         int q_f,                     //!< Charge in the first pixel.
                                         int q_l,                     //!< Charge in the last pixel.
                                         float upper_edge_first_pix,  //!< As the name says.
                                         float lower_edge_last_pix,   //!< As the name says.
                                         float lorentz_shift,         //!< L-shift at half thickness
                                         float theThickness,          //detector thickness
                                         float cot_angle,             //!< cot of alpha_ or beta_
                                         float pitch,                 //!< thePitchX or thePitchY
                                         float pitchfraction_first,
                                         float pitchfraction_last,
                                         float eff_charge_cut_low,   //!< Use edge if > w_eff  &&&
                                         float eff_charge_cut_high,  //!< Use edge if < w_eff  &&&
                                         float size_cut              //!< Use edge when size == cuts
  ) {
    float geom_center = 0.5f * (upper_edge_first_pix + lower_edge_last_pix);

    //--- The case of only one pixel in this projection is separate.  Note that
    //--- here first_pix == last_pix, so the average of the two is still the
    //--- center of the pixel.
    if (size == 1) {
      return geom_center;
    }

    //--- Width of the clusters minus the edge (first and last) pixels.
    //--- In the note, they are denoted x_F and x_L (and y_F and y_L)
    float w_inner = lower_edge_last_pix - upper_edge_first_pix;  // in cm

    //--- Predicted charge width from geometry
    float w_pred = theThickness * cot_angle  // geometric correction (in cm)
                   - lorentz_shift;          // (in cm) &&& check fpix!

    //--- Total length of the two edge pixels (first+last)
    float sum_of_edge = pitchfraction_first + pitchfraction_last;

    //--- The `effective' charge width -- particle's path in first and last pixels only
    float w_eff = std::abs(w_pred) - w_inner;

    //--- If the observed charge width is inconsistent with the expectations
    //--- based on the track, do *not* use w_pred-w_innner.  Instead, replace
    //--- it with an *average* effective charge width, which is the average
    //--- length of the edge pixels.
    //
    //  bool usedEdgeAlgo = false;
    if ((size >= size_cut) || ((w_eff / pitch < eff_charge_cut_low) | (w_eff / pitch > eff_charge_cut_high))) {
      w_eff = pitch * 0.5f * sum_of_edge;  // ave. length of edge pixels (first+last) (cm)
                                           //  usedEdgeAlgo = true;
    }

    //--- Finally, compute the position in this projection
    float q_diff = q_l - q_f;
    float q_sum = q_l + q_f;

    //--- Temporary fix for clusters with both first and last pixel with charge = 0
    if (q_sum == 0)
      q_sum = 1.0f;

    float hit_pos = geom_center + 0.5f * (q_diff / q_sum) * w_eff;

    return hit_pos;
  }

  // Host entry point of genericPositionFormula, kept out of line: inlining it into the callers could change how
  // the compiler contracts its multiply-adds, and with them the hit positions.
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
                                 float size_cut);

  //! Linear interpolation (1-r)*a + r*b, with the FMA contraction pinned
  //! Prevents 1ulp shifts due to compiler optimization
  //! https://github.com/cms-sw/cmssw/issues/48499
  constexpr float interpolate1d(float r, float a, float b) { return std::fma(1.f - r, a, r * b); }

  //! Bilinear interpolation, with the same contraction guarantee as interpolate1d()
  constexpr float interpolate2d(float ry, float rx, float a00, float a01, float a10, float a11) {
    return interpolate1d(ry, interpolate1d(rx, a00, a01), interpolate1d(rx, a10, a11));
  }

}  // namespace siPixelUtils

#endif

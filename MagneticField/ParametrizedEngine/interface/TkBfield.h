#ifndef MagneticField_ParametrizedEngine_interface_TkBfield_h
#define MagneticField_ParametrizedEngine_interface_TkBfield_h

/** \class magfieldparam::TkBfield
 *
 *
 *    B-field in Tracker volume - based on the TOSCA computation version 1103l
 *    (tuned on MTCC measured field (fall 2006))
 *    
 *     In:   x[3]: coordinates (m)
 *    Out:   B[3]: Bx,By,Bz    (T)    (getBxyz)
 *    Out:   B[3]: Br,Bf,Bz    (T)    (getBrfz)
 *
 *    Valid for r<1.15m and |z|<2.80m
 *
 *  \author V.Karimaki 080228, 080407
 *  new float version V.I. October 2012
 */

#include "MagneticField/ParametrizedEngine/interface/BCyl.h"
#include <cmath>
#include <string>

namespace magfieldparam {
  // TkBfield::getBxyz from a copy of TkBfield::bcycl(), e.g. in a GPU kernel
  constexpr void getBxyz(BCycl<float> const& bcyl, float const* __restrict__ x, float* __restrict__ Bxyz) {
    float br;
    float bz;
    float r2 = x[0] * x[0] + x[1] * x[1];
    bcyl(r2, x[2], br, bz);
    Bxyz[0] = br * x[0];
    Bxyz[1] = br * x[1];
    Bxyz[2] = bz;
  }

  // OAEParametrizedMagneticField (positions in cm) on top of TkBfield (in m)
  constexpr float kOAECmToM = 1. / 100;
  constexpr bool isDefinedOAE(float perp2, float z) { return perp2 < (115.f * 115.f) && std::abs(z) < 280.f; }

  // OAEParametrizedMagneticField::inTeslaUnchecked from a copy of TkBfield::bcycl(), e.g. in a GPU kernel
  constexpr void inTeslaOAE(BCycl<float> const& bcyl, float x, float y, float z, float* __restrict__ Bxyz) {
    float const xm[3] = {x * kOAECmToM, y * kOAECmToM, z * kOAECmToM};
    getBxyz(bcyl, xm, Bxyz);
  }

  class TkBfield {
  public:
    // Deprecated ctor, nominal field value specified with a string, eg. "3_8T"
    TkBfield(std::string T);

    // Ctor
    TkBfield(float fld = 3.8);

    /// B out in cartesian
    void getBxyz(float const* __restrict__ x, float* __restrict__ Bxyz) const;
    /// B out in cylindrical
    void getBrfz(float const* __restrict__ x, float* __restrict__ Brfz) const;

    /// the field evaluation; trivially copyable, e.g. to GPUs (see getBxyz and inTeslaOAE above)
    BCycl<float> const& bcycl() const { return bcyl; }

  private:
    BCycl<float> bcyl;
  };
}  // namespace magfieldparam

#endif

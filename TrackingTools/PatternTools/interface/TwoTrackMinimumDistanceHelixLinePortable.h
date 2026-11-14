#ifndef TrackingTools_PatternTools_interface_TwoTrackMinimumDistanceHelixLinePortable_h
#define TrackingTools_PatternTools_interface_TwoTrackMinimumDistanceHelixLinePortable_h

#include <cmath>

/*
 * The closest approach of a helix (charged track) and a straight line, as computed by
 * TwoTrackMinimumDistanceHelixLine: a Newton search for the zero of a function of the helix phase.
 * constexpr functions, usable on the host and on GPUs; TwoTrackMinimumDistanceHelixLine calls them.
 */
struct TwoTrackMinimumDistanceHelixLinePortable {
  struct Coefficients {
    // line point minus helix point, line direction and its squares
    double diffX, diffY, diffZ, px, py, pz, px2, py2, pz2;
    double baseFct, baseDer;
    // signed helix radius in the transverse plane, phase at the helix point, tan(lambda) of the helix
    double theh, phiH0, sinPhiH0, cosPhiH0, tanLambda;
    double aa, bb, cc, dd, ee, ff;
  };

  // posDiff: line point minus helix point; lineDir: line direction; helixP, helixPz, helixPhi: |p|, p_z and phi of
  // the helix momentum; bc2k = Bz * 2.99792458e-3 at the helix point. The float arguments are as the host's vectors.
  static constexpr Coefficients coefficients(float posDiffX,
                                             float posDiffY,
                                             float posDiffZ,
                                             float lineDirX,
                                             float lineDirY,
                                             float lineDirZ,
                                             double helixP,
                                             float helixPz,
                                             int charge,
                                             double bc2k,
                                             double helixPhi) {
    Coefficients coeffs{};
    coeffs.diffX = posDiffX;
    coeffs.diffY = posDiffY;
    coeffs.diffZ = posDiffZ;
    coeffs.px = lineDirX;
    coeffs.px2 = coeffs.px * coeffs.px;
    coeffs.py = lineDirY;
    coeffs.py2 = coeffs.py * coeffs.py;
    coeffs.pz = lineDirZ;
    coeffs.pz2 = coeffs.pz * coeffs.pz;
    coeffs.theh = -helixP / (charge * bc2k) * std::sqrt(1 - (((helixPz * helixPz) / (helixP * helixP))));
    coeffs.tanLambda = -helixPz / (charge * bc2k * coeffs.theh);
    coeffs.phiH0 = helixPhi;
    coeffs.sinPhiH0 = std::sin(coeffs.phiH0);
    coeffs.cosPhiH0 = std::cos(coeffs.phiH0);
    coeffs.aa = (coeffs.diffX + coeffs.theh * coeffs.sinPhiH0) * (coeffs.py2 + coeffs.pz2) -
                coeffs.px * (coeffs.py * coeffs.diffY + coeffs.pz * coeffs.diffZ);
    coeffs.bb = (coeffs.diffY - coeffs.theh * coeffs.cosPhiH0) * (coeffs.px2 + coeffs.pz2) -
                coeffs.py * (coeffs.px * coeffs.diffX + coeffs.pz * coeffs.diffZ);
    coeffs.cc = coeffs.pz * coeffs.theh * coeffs.tanLambda;
    coeffs.dd = coeffs.theh * coeffs.px * coeffs.py;
    coeffs.ee = coeffs.theh * (coeffs.px2 - coeffs.py2);
    coeffs.ff = (coeffs.px2 + coeffs.py2) * coeffs.theh * coeffs.tanLambda * coeffs.tanLambda;
    coeffs.baseFct = coeffs.tanLambda * (coeffs.diffZ * (coeffs.px2 + coeffs.py2) -
                                         coeffs.pz * (coeffs.px * coeffs.diffX + coeffs.py * coeffs.diffY));
    coeffs.baseDer = -coeffs.ff;
    return coeffs;
  }

  // the function of the helix phase phiH whose zero is the closest approach, and its derivative
  static constexpr void functionAndDerivative(Coefficients const& coeffs, double phiH, double& fct, double& derivative) {
    double sinPhiH = std::sin(phiH);
    double cosPhiH = std::cos(phiH);

    fct = coeffs.baseFct;
    fct -= coeffs.ff * (phiH - coeffs.phiH0);
    fct += cosPhiH * coeffs.aa;
    fct += sinPhiH * coeffs.bb;
    fct += coeffs.cc * (phiH - coeffs.phiH0) * (coeffs.px * cosPhiH + coeffs.py * sinPhiH);
    fct += coeffs.cc * (coeffs.px * (sinPhiH - coeffs.sinPhiH0) - coeffs.py * (cosPhiH - coeffs.cosPhiH0));
    fct += coeffs.dd * (sinPhiH * (sinPhiH - coeffs.sinPhiH0) - cosPhiH * (cosPhiH - coeffs.cosPhiH0));
    fct += coeffs.ee * cosPhiH * sinPhiH;

    derivative = coeffs.baseDer;
    derivative += -sinPhiH * coeffs.aa;
    derivative += cosPhiH * coeffs.bb;
    derivative += coeffs.cc * (phiH - coeffs.phiH0) * (coeffs.py * cosPhiH - coeffs.px * sinPhiH);
    derivative += 2 * coeffs.cc * (coeffs.px * cosPhiH + coeffs.py * sinPhiH);
    derivative += coeffs.dd * (4 * cosPhiH * sinPhiH - cosPhiH * coeffs.sinPhiH0 - sinPhiH * coeffs.cosPhiH0);
    derivative += coeffs.ee * (cosPhiH * cosPhiH - sinPhiH * sinPhiH);
  }

  // the search limits of TwoTrackMinimumDistanceHelixLine, and the tolerance TwoTrackMinimumDistance asks for
  static constexpr int maxIterations = 12;
  static constexpr float tolerance = 1.e-6f;

  // Newton search for the zero of functionAndDerivative, starting at phiH and kept within phiH0 +- pi; stops when the
  // step is below stepTolerance. phiH is the last phase; returns false if maxIterations steps did not converge.
  static constexpr bool newtonSearch(Coefficients const& coeffs, float stepTolerance, double& phiH) {
    const double phiMin = coeffs.phiH0 - M_PI, phiMax = coeffs.phiH0 + M_PI;
    for (int iteration = 1; iteration <= maxIterations; ++iteration) {
      double fct = 0, derivative = 0;
      functionAndDerivative(coeffs, phiH, fct, derivative);
      const double dPhiH = fct / derivative;
      phiH -= dPhiH;
      // a step out of the limits is moved back towards the previous phase
      if ((phiMin - phiH) * (phiH - phiMax) < 0.0)
        phiH += (dPhiH * 0.8);
      if (std::fabs(dPhiH) < stepTolerance)
        return true;
    }
    return false;
  }

  // point on the helix at phase phiH, relative to the helix point
  static constexpr void helixOffset(Coefficients const& coeffs, double phiH, double& dx, double& dy, double& dz) {
    dx = coeffs.theh * (std::sin(phiH) - coeffs.sinPhiH0);
    dy = coeffs.theh * (-std::cos(phiH) + coeffs.cosPhiH0);
    dz = coeffs.theh * (coeffs.tanLambda * (phiH - coeffs.phiH0));
  }

  // path length along the helix from the helix point to phase phiH
  static constexpr double helixPathLength(Coefficients const& coeffs, double phiH) {
    return (phiH - coeffs.phiH0) * (coeffs.theh * std::sqrt(1 + coeffs.tanLambda * coeffs.tanLambda));
  }
};

#endif  // TrackingTools_PatternTools_interface_TwoTrackMinimumDistanceHelixLinePortable_h

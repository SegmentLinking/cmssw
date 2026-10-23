#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/AnalyticalJacobians/interface/curvilinearJacobian.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include <vdt/vdtMath.h>

AnalyticalCurvilinearJacobian::AnalyticalCurvilinearJacobian(const GlobalTrajectoryParameters& globalParameters,
                                                             const GlobalPoint& x,
                                                             const GlobalVector& p,
                                                             const double& s)
    : theJacobian(AlgebraicMatrixID()) {
  //
  // helix: calculate full jacobian
  //
  if (s * s * fabs(globalParameters.transverseCurvature()) > 1.e-5) {
    // GlobalPoint xStart = globalParameters.position();
    // GlobalVector h  = globalParameters.magneticFieldInInverseGeV(xStart);
    GlobalVector h = globalParameters.magneticFieldInInverseGeV();
    computeFullJacobian(globalParameters, x, p, h, s);
  }
  //
  // straight line approximation, error in RPhi about 0.1um
  //
  else
    computeStraightLineJacobian(globalParameters, x, p, s);
  //dbg::dbg_trace(1,"ACJ1", globalParameters.vector(),x,p,s,theJacobian);
}

AnalyticalCurvilinearJacobian::AnalyticalCurvilinearJacobian(
    const GlobalTrajectoryParameters& globalParameters,
    const GlobalPoint& x,
    const GlobalVector& p,
    const GlobalVector& h,  // h is the magnetic Field in Inverse GeV
    const double& s)
    : theJacobian(AlgebraicMatrixID()) {
  //
  // helix: calculate full jacobian
  //
  if (s * s * fabs(globalParameters.transverseCurvature()) > 1.e-5)
    computeFullJacobian(globalParameters, x, p, h, s);
  //
  // straight line approximation, error in RPhi about 0.1um
  //
  else
    computeStraightLineJacobian(globalParameters, x, p, s);

  //dbg::dbg_trace(1,"ACJ2", globalParameters.vector(),x,p,s,theJacobian);
}

#if defined(USE_SSEVECT) && !defined(TRPRFN_SCALAR)
#include "AnalyticalCurvilinearJacobianSSE.icc"
#elif defined(USE_EXTVECT) && !defined(TRPRFN_SCALAR)
#include "AnalyticalCurvilinearJacobianEXT.icc"

#else

void AnalyticalCurvilinearJacobian::computeFullJacobian(const GlobalTrajectoryParameters& globalParameters,
                                                        const GlobalPoint& x,
                                                        const GlobalVector& p,
                                                        const GlobalVector& h,
                                                        const double& s) {
  auto toVector3 = [](auto const& vec) { return curvilinearJacobian::Vector3{vec.x(), vec.y(), vec.z()}; };
  double jacobian[5][5];
  curvilinearJacobian::fullJacobian(toVector3(globalParameters.position()),
                                    toVector3(globalParameters.momentum()),
                                    globalParameters.signedInverseMomentum(),
                                    toVector3(x),
                                    toVector3(p),
                                    toVector3(h),
                                    s,
                                    jacobian);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j)
      theJacobian(i, j) = jacobian[i][j];
}

#endif

void AnalyticalCurvilinearJacobian::computeInfinitesimalJacobian(const GlobalTrajectoryParameters& globalParameters,
                                                                 const GlobalPoint&,
                                                                 const GlobalVector& p,
                                                                 const GlobalVector& h,
                                                                 const double& s) {
  /*
   * origin  TRPROP
   *
   C *** ERROR PROPAGATION ALONG A PARTICLE TRAJECTORY IN A MAGNETIC FIELD
   C     ROUTINE ASSUMES THAT IN THE INTERVAL (X1,X2) THE QUANTITIES 1/P
   C     AND (HX,HY,HZ) ARE RATHER CONSTANT. DELTA(PHI) MUST NOT BE TOO LARGE
   C
   C     Authors: A. Haas and W. Wittek
   C
   
  */

  double qbp = globalParameters.signedInverseMomentum();
  double absS = s;

  // average momentum
  GlobalVector tn = (globalParameters.momentum() + p).unit();
  double sinl = tn.z();
  double cosl = std::sqrt(1. - sinl * sinl);
  double cosl1 = 1. / cosl;
  double tgl = sinl * cosl1;
  double sinp = tn.y() * cosl1;
  double cosp = tn.x() * cosl1;

  // define average magnetic field and gradient
  // at initial point - inlike TRPROP
  double b0 = h.x() * cosp + h.y() * sinp;
  double b2 = -h.x() * sinp + h.y() * cosp;
  double b3 = -b0 * sinl + h.z() * cosl;

  theJacobian = AlgebraicMatrixID();

  theJacobian(3, 2) = absS * cosl;
  theJacobian(4, 1) = absS;

  theJacobian(1, 0) = absS * b2;
  //if ( qbp<0) theJacobian(1,0) = -theJacobian(1,0);
  theJacobian(1, 2) = -b0 * (absS * qbp);
  theJacobian(1, 3) = b3 * (b2 * qbp * (absS * qbp));
  theJacobian(1, 4) = -b2 * (b2 * qbp * (absS * qbp));

  theJacobian(2, 0) = -absS * b3 * cosl1;
  // if ( qbp<0) theJacobian(2,0) = -theJacobian(2,0);
  theJacobian(2, 1) = b0 * (absS * qbp) * cosl1 * cosl1;
  theJacobian(2, 2) = 1. + tgl * b2 * (absS * qbp);
  theJacobian(2, 3) = -b3 * (b3 * qbp * (absS * qbp) * cosl1);
  theJacobian(2, 4) = b2 * (b3 * qbp * (absS * qbp) * cosl1);

  theJacobian(3, 4) = -b3 * tgl * (absS * qbp);
  theJacobian(4, 3) = b3 * tgl * (absS * qbp);
}

void AnalyticalCurvilinearJacobian::computeStraightLineJacobian(const GlobalTrajectoryParameters& globalParameters,
                                                                const GlobalPoint&,
                                                                const GlobalVector&,
                                                                const double& s) {
  //
  // matrix: elements =1 on diagonal and =0 are already set
  // in initialisation
  //
  theJacobian = AlgebraicMatrixID();
  GlobalVector p1 = globalParameters.momentum().unit();
  double cosl0 = p1.perp();
  theJacobian(3, 2) = cosl0 * s;
  theJacobian(4, 1) = s;
}

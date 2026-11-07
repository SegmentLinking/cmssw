#ifndef _Tracker_TwoTrackMinimumDistanceHelixLine_H_
#define _Tracker_TwoTrackMinimumDistanceHelixLine_H_

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "TrackingTools/PatternTools/interface/helixLineClosestApproach.h"
#include <utility>
/** \class TwoTrackMinimumDistanceHelixLine
 *  This is a helper class for TwoTrackMinimumDistance, for the
 *  case where one of the tracks is charged and the other not.
 *  No user should need direct access to this class.
 *  It implements a Newton method
 *  for finding the minimum distance between two tracks.
 */

class GlobalTrajectoryParameters;

class TwoTrackMinimumDistanceHelixLine {
public:
  TwoTrackMinimumDistanceHelixLine() : theH(nullptr), theL(nullptr), themaxiter(12), pointsUpdated(false) {}
  ~TwoTrackMinimumDistanceHelixLine() {}

  /**
   * Calculates the PCA between a charged particle (helix) and a neutral 
   * particle (line). The order of the trajectories (helix-line or line-helix)
   * is irrelevent, and will be conserved.
   */

  bool calculate(const GlobalTrajectoryParameters &,
                 const GlobalTrajectoryParameters &,
                 const float qual = .0001);  // retval=true? error occured.

  /**
   * Returns the PCA's on the two trajectories. The first point lies on the
   * first trajectory, the second point on the second trajectory.
   */

  std::pair<GlobalPoint, GlobalPoint> points() const;
  std::pair<double, double> pathLength() const;

  double firstAngle() const;
  double secondAngle() const;

private:
  const GlobalTrajectoryParameters *theH, *theL, *firstGTP, *secondGTP;
  GlobalVector posDiff;
  GlobalVector theLp;
  helixLineClosestApproach::Coefficients theCoeffs;
  double thePhiH;
  double Hn, Ln;

  int themaxiter;
  bool updateCoeffs();
  void finalPoints();
  bool oneIteration(double &thePhiH, double &fct, double &derivative) const;
  GlobalPoint helixPoint, linePoint;
  double tL, linePath, helixPath;
  bool pointsUpdated;
};
#endif

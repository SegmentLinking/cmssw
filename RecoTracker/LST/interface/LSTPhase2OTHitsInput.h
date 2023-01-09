#ifndef RecoTracker_LST_LSTPhase2OTHitsInput_h
#define RecoTracker_LST_LSTPhase2OTHitsInput_h

#include <memory>
#include <vector>

#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"

class LSTPhase2OTHitsInput {
public:
  LSTPhase2OTHitsInput() = default;
  ~LSTPhase2OTHitsInput() = default;

  void setLSTPhase2OTHitsTraits(std::vector<unsigned int> detId,
                                std::vector<float> x,
                                std::vector<float> y,
                                std::vector<float> z,
                                std::vector<TrackingRecHit const*> hits) {
    detId_ = detId;
    x_ = x;
    y_ = y;
    z_ = z;
    hits_ = hits;
  }

  std::vector<unsigned int> const& detId() const { return detId_; }
  std::vector<float> const& x() const { return x_; }
  std::vector<float> const& y() const { return y_; }
  std::vector<float> const& z() const { return z_; }
  std::vector<TrackingRecHit const*> const& hits() const { return hits_; }

private:
  std::vector<unsigned int> detId_;
  std::vector<float> x_;
  std::vector<float> y_;
  std::vector<float> z_;
  std::vector<TrackingRecHit const*> hits_;
};

#endif


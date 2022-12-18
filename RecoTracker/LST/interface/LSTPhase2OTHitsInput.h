#ifndef RecoTracker_LST_LSTPhase2OTHitsInput_h
#define RecoTracker_LST_LSTPhase2OTHitsInput_h

#include <memory>
#include <vector>

class LSTPhase2OTHitsInput {
public:
  LSTPhase2OTHitsInput() = default;
  ~LSTPhase2OTHitsInput() = default;

  void setLSTPhase2OTHitsTraits(std::vector<unsigned int> detId,
                                std::vector<float> x,
                                std::vector<float> y,
                                std::vector<float> z) {
    detId_ = detId;
    x_ = x;
    y_ = y;
    z_ = z;
  }

  std::vector<unsigned int> const& detId() const { return detId_; }
  std::vector<float> const& x() const { return x_; }
  std::vector<float> const& y() const { return y_; }
  std::vector<float> const& z() const { return z_; }

private:
  std::vector<unsigned int> detId_;
  std::vector<float> x_;
  std::vector<float> y_;
  std::vector<float> z_;
};

#endif


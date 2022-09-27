#ifndef RecoTracker_LST_LSTPhase2OTHitsInput_h
#define RecoTracker_LST_LSTPhase2OTHitsInput_h

#include <memory>
#include <vector>

class LSTPhase2OTHitsInput {
public:
  LSTPhase2OTHitsInput();
  ~LSTPhase2OTHitsInput();

  LSTPhase2OTHitsInput(LSTPhase2OTHitsInput const&) = delete;
  LSTPhase2OTHitsInput& operator=(LSTPhase2OTHitsInput const&) = delete;
  LSTPhase2OTHitsInput(LSTPhase2OTHitsInput&&);
  LSTPhase2OTHitsInput& operator=(LSTPhase2OTHitsInput&&);

  void setLSTPhase2OTHitsTraits(std::vector<unsigned int> ph2_detId,
                                std::vector<float> ph2_x,
                                std::vector<float> ph2_y,
                                std::vector<float> ph2_z) {
    ph2_detId_ = ph2_detId;
    ph2_x_ = ph2_x;
    ph2_y_ = ph2_y;
    ph2_z_ = ph2_z;
  }

  std::vector<unsigned int>& ph2_detId() { return ph2_detId_; }
  std::vector<unsigned int> const& ph2_detId() const { return ph2_detId_; }

  std::vector<float>& ph2_x() { return ph2_x_; }
  std::vector<float> const& ph2_x() const { return ph2_x_; }

  std::vector<float>& ph2_y() { return ph2_y_; }
  std::vector<float> const& ph2_y() const { return ph2_y_; }

  std::vector<float>& ph2_z() { return ph2_z_; }
  std::vector<float> const& ph2_z() const { return ph2_z_; }

private:
  std::vector<unsigned int> ph2_detId_;
  std::vector<float> ph2_x_;
  std::vector<float> ph2_y_;
  std::vector<float> ph2_z_;
};

#endif


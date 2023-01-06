#ifndef RecoTracker_LST_LSTOutput_h
#define RecoTracker_LST_LSTOutput_h

#include <memory>
#include <vector>

class LSTOutput {
public:
  LSTOutput() = default;
  ~LSTOutput() = default;

  void setLSTOutputTraits(std::vector<std::vector<unsigned int>> hitIdx,
                          std::vector<int> len,
                          std::vector<float> pt,
                          std::vector<float> eta,
                          std::vector<float> phi) {
    hitIdx_ = hitIdx;
    len_ = len;
    pt_ = pt;
    eta_ = eta;
    phi_ = phi;
  }

  std::vector<std::vector<unsigned int>> const& hitIdx() const { return hitIdx_; }
  std::vector<int> const& len() const { return len_; }
  std::vector<float> const& pt() const { return pt_; }
  std::vector<float> const& eta() const { return eta_; }
  std::vector<float> const& phi() const { return phi_; }

private:
  std::vector<std::vector<unsigned int>> hitIdx_;
  std::vector<int> len_;
  std::vector<float> pt_;
  std::vector<float> eta_;
  std::vector<float> phi_;
};

#endif


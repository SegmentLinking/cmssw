#ifndef RecoTracker_LST_LSTOutput_h
#define RecoTracker_LST_LSTOutput_h

#include <memory>
#include <vector>

class LSTOutput {
public:
  LSTOutput() = default;
  ~LSTOutput() = default;

  void setLSTOutputTraits(std::vector<std::vector<unsigned int>> hitIdx,
                          std::vector<unsigned int> len,
                          std::vector<float> pt,
                          std::vector<float> eta,
                          std::vector<float> phi,
                          std::vector<int> seedIdx,
                          std::vector<short> trackCandidateType) {
    hitIdx_ = hitIdx;
    len_ = len;
    pt_ = pt;
    eta_ = eta;
    phi_ = phi;
    seedIdx_ = seedIdx;
    trackCandidateType_ = trackCandidateType;
  }

  std::vector<std::vector<unsigned int>> const& hitIdx() const { return hitIdx_; }
  std::vector<unsigned int> const& len() const { return len_; }
  std::vector<float> const& pt() const { return pt_; }
  std::vector<float> const& eta() const { return eta_; }
  std::vector<float> const& phi() const { return phi_; }
  std::vector<int> const& seedIdx() const { return seedIdx_; }
  std::vector<short> const& trackCandidateType() const { return trackCandidateType_; }

private:
  std::vector<std::vector<unsigned int>> hitIdx_;
  std::vector<unsigned int> len_;
  std::vector<float> pt_;
  std::vector<float> eta_;
  std::vector<float> phi_;
  std::vector<int> seedIdx_;
  std::vector<short> trackCandidateType_;
};

#endif


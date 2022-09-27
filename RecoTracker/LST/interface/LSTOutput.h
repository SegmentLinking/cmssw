#ifndef RecoTracker_LST_LSTOutput_h
#define RecoTracker_LST_LSTOutput_h

#include <memory>
#include <vector>

class LSTOutput {
public:
  LSTOutput();
  ~LSTOutput();

  LSTOutput(LSTOutput const&) = delete;
  LSTOutput& operator=(LSTOutput const&) = delete;
  LSTOutput(LSTOutput&&);
  LSTOutput& operator=(LSTOutput&&);

  void setLSTOutputTraits(std::vector<unsigned int> lstTrackCandidate_hitIdx,
                          std::vector<unsigned int> lstTrackCandidate_len,
                          std::vector<float> lstTrackCandidate_pt,
                          std::vector<float> lstTrackCandidate_eta,
                          std::vector<float> lstTrackCandidate_phi) {
    lstTrackCandidate_hitIdx_ = lstTrackCandidate_hitIdx;
    lstTrackCandidate_len_ = lstTrackCandidate_len;
    lstTrackCandidate_pt_ = lstTrackCandidate_pt;
    lstTrackCandidate_eta_ = lstTrackCandidate_eta;
    lstTrackCandidate_phi_ = lstTrackCandidate_phi;
  }

  std::vector<unsigned int>& lstTrackCandidate_hitIdx() { return lstTrackCandidate_hitIdx_; }
  std::vector<unsigned int> const& lstTrackCandidate_hitIdx() const { return lstTrackCandidate_hitIdx_; }

  std::vector<unsigned int>& lstTrackCandidate_len() { return lstTrackCandidate_len_; }
  std::vector<unsigned int> const& lstTrackCandidate_len() const { return lstTrackCandidate_len_; }

  std::vector<float>& lstTrackCandidate_pt() { return lstTrackCandidate_pt_; }
  std::vector<float> const& lstTrackCandidate_pt() const { return lstTrackCandidate_pt_; }

  std::vector<float>& lstTrackCandidate_eta() { return lstTrackCandidate_eta_; }
  std::vector<float> const& lstTrackCandidate_eta() const { return lstTrackCandidate_eta_; }

  std::vector<float>& lstTrackCandidate_phi() { return lstTrackCandidate_phi_; }
  std::vector<float> const& lstTrackCandidate_phi() const { return lstTrackCandidate_phi_; }

private:
  std::vector<unsigned int> lstTrackCandidate_hitIdx_;
  std::vector<unsigned int> lstTrackCandidate_len_;
  std::vector<float> lstTrackCandidate_pt_;
  std::vector<float> lstTrackCandidate_eta_;
  std::vector<float> lstTrackCandidate_phi_;
};

#endif


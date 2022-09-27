#ifndef RecoTracker_LST_LSTPixelSeedInput_h
#define RecoTracker_LST_LSTPixelSeedInput_h

#include <memory>
#include <vector>

class LSTPixelSeedInput {
public:
  LSTPixelSeedInput();
  ~LSTPixelSeedInput();

  LSTPixelSeedInput(LSTPixelSeedInput const&) = delete;
  LSTPixelSeedInput& operator=(LSTPixelSeedInput const&) = delete;
  LSTPixelSeedInput(LSTPixelSeedInput&&);
  LSTPixelSeedInput& operator=(LSTPixelSeedInput&&);

  void setLSTPixelSeedTraits(std::vector<float> see_px,
                             std::vector<float> see_py,
                             std::vector<float> see_pz,
                             std::vector<float> see_dxy,
                             std::vector<float> see_dz,
                             std::vector<float> see_ptErr,
                             std::vector<float> see_etaErr,
                             std::vector<float> see_stateTrajGlbPx,
                             std::vector<float> see_stateTrajGlbPy,
                             std::vector<float> see_stateTrajGlbPz,
                             std::vector<int> see_q,
                             std::vector<unsigned int> see_algo,
                             std::vector<std::vector<int>> see_hitIdx) {
    see_px_ = see_px;
    see_py_ = see_py;
    see_pz_ = see_pz;
    see_dxy_ = see_dxy;
    see_dz_ = see_dz;
    see_ptErr_ = see_ptErr;
    see_etaErr_ = see_etaErr;
    see_stateTrajGlbPx_ = see_stateTrajGlbPx;
    see_stateTrajGlbPy_ = see_stateTrajGlbPy;
    see_stateTrajGlbPz_ = see_stateTrajGlbPz;
    see_q_ = see_q;
    see_algo_ = see_algo;
    see_hitIdx_ = see_hitIdx;
  }

  std::vector<float>& see_px() { return see_px_; }
  std::vector<float> const& see_px() const { return see_px_; }

  std::vector<float>& see_py() { return see_py_; }
  std::vector<float> const& see_py() const { return see_py_; }

  std::vector<float>& see_pz() { return see_pz_; }
  std::vector<float> const& see_pz() const { return see_pz_; }

  std::vector<float>& see_dxy() { return see_dxy_; }
  std::vector<float> const& see_dxy() const { return see_dxy_; }

  std::vector<float>& see_dz() { return see_dz_; }
  std::vector<float> const& see_dz() const { return see_dz_; }

  std::vector<float>& see_ptErr() { return see_ptErr_; }
  std::vector<float> const& see_ptErr() const { return see_ptErr_; }

  std::vector<float>& see_etaErr() { return see_etaErr_; }
  std::vector<float> const& see_etaErr() const { return see_etaErr_; }

  std::vector<float>& see_stateTrajGlbPx() { return see_stateTrajGlbPx_; }
  std::vector<float> const& see_stateTrajGlbPx() const { return see_stateTrajGlbPx_; }

  std::vector<float>& see_stateTrajGlbPy() { return see_stateTrajGlbPy_; }
  std::vector<float> const& see_stateTrajGlbPy() const { return see_stateTrajGlbPy_; }

  std::vector<float>& see_stateTrajGlbPz() { return see_stateTrajGlbPz_; }
  std::vector<float> const& see_stateTrajGlbPz() const { return see_stateTrajGlbPz_; }

  std::vector<int>& see_q() { return see_q_; }
  std::vector<int> const& see_q() const { return see_q_; }

  std::vector<unsigned int>& see_algo() { return see_algo_; }
  std::vector<unsigned int> const& see_algo() const { return see_algo_; }

  std::vector<std::vector<int>>& see_hitIdx() { return see_hitIdx_; }
  std::vector<std::vector<int>> const& see_hitIdx() const { return see_hitIdx_; }

private:
  std::vector<float> see_px_;
  std::vector<float> see_py_;
  std::vector<float> see_pz_;
  std::vector<float> see_dxy_;
  std::vector<float> see_dz_;
  std::vector<float> see_ptErr_;
  std::vector<float> see_etaErr_;
  std::vector<float> see_stateTrajGlbPx_;
  std::vector<float> see_stateTrajGlbPy_;
  std::vector<float> see_stateTrajGlbPz_;
  std::vector<int> see_q_;
  std::vector<unsigned int> see_algo_;
  std::vector<std::vector<int>> see_hitIdx_;
};

#endif


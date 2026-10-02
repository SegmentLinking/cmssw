// Pair loop for s48_dedup_threshold_hists.py (AfterBuild / pT5-FromMap replay).
// Build: g++ -O3 -march=native -shared -fPIC -o s48_pairmax.so s48_pairmax.cc
//
// Objects are pre-sorted by key (group * 100 + eta). For every pair inside the eta window of the
// same group that passes the kernel's |deta| <= dEtaCut and |dphi| <= dPhiCut (float32, as on the
// GPU), count shared hits (rows of H are sorted) and record, for the pair's loser, the maximum
// shared-hit count: out[loser] = max(out[loser], nMatched).
//   mode 0 (RemoveDupQuintupletsAfterBuild): ix = lower index; ix loses iff score[ix] >= score[jx]
//   mode 1 (RemoveDupPixelQuintupletsFromMap): i loses iff score[i] > score[j] or (== and i > j)
#include <cmath>
#include <cstdint>

static inline float deltaPhi(float a, float b) {
  float d = a - b;
  const float pi = 3.14159265358979323846f;
  if (d > pi) d -= 2.f * pi;
  else if (d < -pi) d += 2.f * pi;
  return d;
}

extern "C" void pair_max(int64_t n, const int64_t* order, const double* key, double width, const float* eta,
                         const float* phi, float dEtaCut, float dPhiCut, const float* score, const int32_t* H,
                         int K, int mode, int32_t* out) {
  for (int64_t si = 0; si < n; ++si) {
    const int64_t i = order[si];
    const int32_t* hi = H + i * K;
    for (int64_t sj = si + 1; sj < n && key[sj] - key[si] <= width; ++sj) {
      const int64_t j = order[sj];
      if (std::fabs(eta[i] - eta[j]) > dEtaCut) continue;
      if (std::fabs(deltaPhi(phi[i], phi[j])) > dPhiCut) continue;
      const int32_t* hj = H + j * K;
      int a = 0, b = 0, nm = 0;
      while (a < K && b < K) {
        if (hi[a] == hj[b]) { ++nm; ++a; ++b; }
        else if (hi[a] < hj[b]) ++a;
        else ++b;
      }
      const int64_t lo = i < j ? i : j, hi_ = i < j ? j : i;
      int64_t loser;
      if (mode == 0) loser = (score[lo] >= score[hi_]) ? lo : hi_;
      else loser = (score[lo] > score[hi_]) ? lo : hi_;  // tie -> higher index loses
      if (nm > out[loser]) out[loser] = nm;
    }
  }
}

#ifndef RecoTracker_MkFitAlpaka_interface_matriplex_Matrix_h
#define RecoTracker_MkFitAlpaka_interface_matriplex_Matrix_h

// Portable port of RecoTracker/MkFitCore/src/Matrix.h (CMSSW_20_1_0_pre2): the Matriplex typedefs, templated on
// N = tracks per Alpaka thread (MkFitCore: fixed NN = MPT_SIZE = 8 on x86-64-v3). hipo/sincos4 live in
// interface/math/MathUtils.h. Per-backend N: see MatriplexBackend.h (ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::kNN).

#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatriplexSym.h"
#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatrixSTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"
#include "RecoTracker/MkFitAlpaka/interface/math/MathUtils.h"

namespace mkfitdev {

  // NN of the default CMSSW x86-64-v3 build (MPT_SIZE for __AVX2__). Lane width on CPU backends.
  constexpr Matriplex::idx_t kNNCpu = 8;
  // one track per thread.
  constexpr Matriplex::idx_t kNNGpu = 1;

  constexpr Matriplex::idx_t LL = 6;  // Dimension of large/long  MPlex entities
  constexpr Matriplex::idx_t HH = 3;  // Dimension of small/short MPlex entities

  template <Matriplex::idx_t N>
  using MPlexLL = Matriplex::Matriplex<float, LL, LL, N>;
  template <Matriplex::idx_t N>
  using MPlexLV = Matriplex::Matriplex<float, LL, 1, N>;
  template <Matriplex::idx_t N>
  using MPlexLS = Matriplex::MatriplexSym<float, LL, N>;

  template <Matriplex::idx_t N>
  using MPlexHH = Matriplex::Matriplex<float, HH, HH, N>;
  template <Matriplex::idx_t N>
  using MPlexHV = Matriplex::Matriplex<float, HH, 1, N>;
  template <Matriplex::idx_t N>
  using MPlexHS = Matriplex::MatriplexSym<float, HH, N>;

  template <Matriplex::idx_t N>
  using MPlex5V = Matriplex::Matriplex<float, 5, 1, N>;
  template <Matriplex::idx_t N>
  using MPlex5S = Matriplex::MatriplexSym<float, 5, N>;

  template <Matriplex::idx_t N>
  using MPlex55 = Matriplex::Matriplex<float, 5, 5, N>;
  template <Matriplex::idx_t N>
  using MPlex56 = Matriplex::Matriplex<float, 5, 6, N>;
  template <Matriplex::idx_t N>
  using MPlex65 = Matriplex::Matriplex<float, 6, 5, N>;

  template <Matriplex::idx_t N>
  using MPlex22 = Matriplex::Matriplex<float, 2, 2, N>;
  template <Matriplex::idx_t N>
  using MPlex2V = Matriplex::Matriplex<float, 2, 1, N>;
  template <Matriplex::idx_t N>
  using MPlex2S = Matriplex::MatriplexSym<float, 2, N>;

  template <Matriplex::idx_t N>
  using MPlexLH = Matriplex::Matriplex<float, LL, HH, N>;
  template <Matriplex::idx_t N>
  using MPlexHL = Matriplex::Matriplex<float, HH, LL, N>;

  template <Matriplex::idx_t N>
  using MPlex52 = Matriplex::Matriplex<float, 5, 2, N>;
  template <Matriplex::idx_t N>
  using MPlexL2 = Matriplex::Matriplex<float, LL, 2, N>;
  template <Matriplex::idx_t N>
  using MPlexH2 = Matriplex::Matriplex<float, HH, 2, N>;
  template <Matriplex::idx_t N>
  using MPlex2H = Matriplex::Matriplex<float, 2, HH, N>;

  template <Matriplex::idx_t N>
  using MPlexQF = Matriplex::Matriplex<float, 1, 1, N>;
  template <Matriplex::idx_t N>
  using MPlexQI = Matriplex::Matriplex<int, 1, 1, N>;
  template <Matriplex::idx_t N>
  using MPlexQUI = Matriplex::Matriplex<unsigned int, 1, 1, N>;
  template <Matriplex::idx_t N>
  using MPlexQH = Matriplex::Matriplex<short, 1, 1, N>;
  template <Matriplex::idx_t N>
  using MPlexQUH = Matriplex::Matriplex<unsigned short, 1, 1, N>;

  template <Matriplex::idx_t N>
  using MPlexQB = Matriplex::Matriplex<bool, 1, 1, N>;

  // Short names at mkfitdev level (the class itself is mkfitdev::Matriplex::Matriplex, as MkFitCore
  // ::Matriplex::Matriplex).
  template <typename T, Matriplex::idx_t D1, Matriplex::idx_t D2, Matriplex::idx_t N>
  using MPlex = Matriplex::Matriplex<T, D1, D2, N>;
  template <typename T, Matriplex::idx_t D, Matriplex::idx_t N>
  using MPlexSym = Matriplex::MatriplexSym<T, D, N>;
  template <typename T, Matriplex::idx_t D, Matriplex::idx_t N>
  using MatriplexSym = Matriplex::MatriplexSym<T, D, N>;

}  // namespace mkfitdev

#endif

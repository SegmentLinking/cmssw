#ifndef RecoTracker_MkFitAlpaka_interface_matriplex_MatrixSTypes_h
#define RecoTracker_MkFitAlpaka_interface_matriplex_MatrixSTypes_h

// Portable, layout-compatible stand-ins for the ROOT SMatrix typedefs of RecoTracker/MkFitCore/interface/MatrixSTypes.h
// (CMSSW_20_1_0_pre2). ROOT SMatrix is not device code; these PODs keep exactly its storage order so that per-track
// states (6 parameters + 21 packed errors, as mkfit::TrackState) and hit errors can be stored in device memory and
// moved into a Matriplex with copyIn/copyOut (Matriplex and MatRepSym share the packed lower-triangle order):
//   SVector<T, D>          D elements
//   SMatrix<T, D1, D2>     row-major, element (i, j) at i * D2 + j      (ROOT MatRepStd)
//   SMatrixSym<T, D>       packed lower triangle, (i, j) at symOffset(i, j) = i(i+1)/2 + j for i >= j  (ROOT MatRepSym)
// Only storage and element access are provided (Array(), operator(), At, operator[], kRows/kCols/kSize) plus
// MkFitCore's diagonalOnly(); SMatrix algebra stays on the host (MkFitCore uses it there only).

#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatriplexCommon.h"

namespace mkfitdev {

  template <typename T, int D>
  struct SVector {
    static constexpr int kSize = D;
    T fArray[D];

    ALPAKA_FN_HOST_ACC T* Array() { return fArray; }
    ALPAKA_FN_HOST_ACC const T* Array() const { return fArray; }
    ALPAKA_FN_HOST_ACC T& operator[](int i) { return fArray[i]; }
    ALPAKA_FN_HOST_ACC const T& operator[](int i) const { return fArray[i]; }
    ALPAKA_FN_HOST_ACC T& operator()(int i) { return fArray[i]; }
    ALPAKA_FN_HOST_ACC const T& operator()(int i) const { return fArray[i]; }
    ALPAKA_FN_HOST_ACC T& At(int i) { return fArray[i]; }
    ALPAKA_FN_HOST_ACC const T& At(int i) const { return fArray[i]; }
  };

  template <typename T, int D1, int D2>
  struct SMatrix {
    static constexpr int kRows = D1;
    static constexpr int kCols = D2;
    static constexpr int kSize = D1 * D2;
    T fArray[kSize];

    ALPAKA_FN_HOST_ACC T* Array() { return fArray; }
    ALPAKA_FN_HOST_ACC const T* Array() const { return fArray; }
    ALPAKA_FN_HOST_ACC T& operator()(int i, int j) { return fArray[i * D2 + j]; }
    ALPAKA_FN_HOST_ACC const T& operator()(int i, int j) const { return fArray[i * D2 + j]; }
    ALPAKA_FN_HOST_ACC T& At(int i, int j) { return fArray[i * D2 + j]; }
    ALPAKA_FN_HOST_ACC const T& At(int i, int j) const { return fArray[i * D2 + j]; }
  };

  template <typename T, int D>
  struct SMatrixSym {
    static constexpr int kRows = D;
    static constexpr int kCols = D;
    static constexpr int kSize = D * (D + 1) / 2;
    T fArray[kSize];

    ALPAKA_FN_HOST_ACC T* Array() { return fArray; }
    ALPAKA_FN_HOST_ACC const T* Array() const { return fArray; }
    ALPAKA_FN_HOST_ACC T& operator()(int i, int j) { return fArray[Matriplex::symOffset(i, j)]; }
    ALPAKA_FN_HOST_ACC const T& operator()(int i, int j) const { return fArray[Matriplex::symOffset(i, j)]; }
    ALPAKA_FN_HOST_ACC T& At(int i, int j) { return fArray[Matriplex::symOffset(i, j)]; }
    ALPAKA_FN_HOST_ACC const T& At(int i, int j) const { return fArray[Matriplex::symOffset(i, j)]; }
  };

  typedef SMatrixSym<float, 6> SMatrixSym66;
  typedef SMatrix<float, 6, 6> SMatrix66;
  typedef SVector<float, 6> SVector6;

  typedef SMatrix<float, 3, 3> SMatrix33;
  typedef SMatrixSym<float, 3> SMatrixSym33;
  typedef SVector<float, 3> SVector3;

  typedef SMatrix<float, 2, 2> SMatrix22;
  typedef SMatrixSym<float, 2> SMatrixSym22;
  typedef SVector<float, 2> SVector2;

  typedef SMatrix<float, 3, 6> SMatrix36;
  typedef SMatrix<float, 6, 3> SMatrix63;

  typedef SMatrix<float, 2, 6> SMatrix26;
  typedef SMatrix<float, 6, 2> SMatrix62;

  // diagonalOnly: zero all off-diagonal elements (for a symmetric matrix each stored element once)
  template <typename Matrix>
  ALPAKA_FN_HOST_ACC inline void diagonalOnly(Matrix& m) {
    for (int r = 0; r < m.kRows; r++) {
      for (int c = 0; c < m.kCols; c++) {
        if (r != c)
          m(r, c) = 0.f;
      }
    }
  }

}  // namespace mkfitdev

#endif

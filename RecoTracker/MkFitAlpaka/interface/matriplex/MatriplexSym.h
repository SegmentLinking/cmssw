#ifndef RecoTracker_MkFitAlpaka_interface_matriplex_MatriplexSym_h
#define RecoTracker_MkFitAlpaka_interface_matriplex_MatriplexSym_h

// Portable (host + Alpaka device) port of the storage and element access of RecoTracker/MkFitCore/src/Matriplex/
// MatriplexSym.h: same packed lower-triangle layout fArray[symOffset(i, j) * N + n] (= gSymOffsets[D][i * D + j]).

#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatriplexCommon.h"

namespace mkfitdev {
  namespace Matriplex {

    template <typename T, idx_t D, idx_t N>
    class alignas(mplexAlign<T, N>()) MatriplexSym {
    public:
      typedef T value_type;

      static constexpr int kRows = D;
      static constexpr int kCols = D;
      static constexpr int kSize = (D + 1) * D / 2;
      static constexpr int kTotSize = N * kSize;

      T fArray[kTotSize];

      ALPAKA_FN_HOST_ACC MatriplexSym() {}
      ALPAKA_FN_HOST_ACC MatriplexSym(T v) { setVal(v); }

      ALPAKA_FN_HOST_ACC void setVal(T v) {
        for (idx_t i = 0; i < kTotSize; ++i) {
          fArray[i] = v;
        }
      }

      ALPAKA_FN_HOST_ACC void scale(T scale) {
        for (idx_t i = 0; i < kTotSize; ++i) {
          fArray[i] *= scale;
        }
      }

      ALPAKA_FN_HOST_ACC T operator[](idx_t xx) const { return fArray[xx]; }
      ALPAKA_FN_HOST_ACC T& operator[](idx_t xx) { return fArray[xx]; }

      ALPAKA_FN_HOST_ACC const T& constAt(idx_t n, idx_t i, idx_t j) const { return fArray[symOffset(i, j) * N + n]; }

      ALPAKA_FN_HOST_ACC T& At(idx_t n, idx_t i, idx_t j) { return fArray[symOffset(i, j) * N + n]; }

      ALPAKA_FN_HOST_ACC T& operator()(idx_t n, idx_t i, idx_t j) { return At(n, i, j); }
      ALPAKA_FN_HOST_ACC const T& operator()(idx_t n, idx_t i, idx_t j) const { return constAt(n, i, j); }

      ALPAKA_FN_HOST_ACC MatriplexSym& operator=(const MatriplexSym& m) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = m.fArray[i];
        return *this;
      }

      MatriplexSym(const MatriplexSym& m) = default;

      ALPAKA_FN_HOST_ACC void copySlot(idx_t n, const MatriplexSym& m) {
        for (idx_t i = n; i < kTotSize; i += N) {
          fArray[i] = m.fArray[i];
        }
      }

      ALPAKA_FN_HOST_ACC void copyIn(idx_t n, const T* arr) {
        for (idx_t i = n; i < kTotSize; i += N) {
          fArray[i] = *(arr++);
        }
      }

      ALPAKA_FN_HOST_ACC void copyOut(idx_t n, T* arr) const {
        for (idx_t i = n; i < kTotSize; i += N) {
          *(arr++) = fArray[i];
        }
      }
    };

  }  // namespace Matriplex
}  // namespace mkfitdev

#endif

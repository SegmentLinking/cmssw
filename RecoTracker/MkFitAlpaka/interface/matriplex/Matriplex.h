#ifndef RecoTracker_MkFitAlpaka_interface_matriplex_Matriplex_h
#define RecoTracker_MkFitAlpaka_interface_matriplex_Matriplex_h

// Portable (host + Alpaka device) port of the storage and element access of RecoTracker/MkFitCore/src/Matriplex/
// Matriplex.h: same element layout fArray[(i * D2 + j) * N + n]. The arithmetic is written out by the callers.

#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatriplexCommon.h"

namespace mkfitdev {
  namespace Matriplex {

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    class alignas(mplexAlign<T, N>()) Matriplex {
    public:
      typedef T value_type;

      static constexpr int kRows = D1;
      static constexpr int kCols = D2;
      static constexpr int kSize = D1 * D2;
      static constexpr int kTotSize = N * kSize;

      T fArray[kTotSize];

      ALPAKA_FN_HOST_ACC Matriplex() {}
      ALPAKA_FN_HOST_ACC Matriplex(T v) { setVal(v); }

      ALPAKA_FN_HOST_ACC void setVal(T v) {
        for (idx_t i = 0; i < kTotSize; ++i) {
          fArray[i] = v;
        }
      }

      ALPAKA_FN_HOST_ACC void add(const Matriplex& v) {
        for (idx_t i = 0; i < kTotSize; ++i) {
          fArray[i] += v.fArray[i];
        }
      }

      ALPAKA_FN_HOST_ACC T operator[](idx_t xx) const { return fArray[xx]; }
      ALPAKA_FN_HOST_ACC T& operator[](idx_t xx) { return fArray[xx]; }

      ALPAKA_FN_HOST_ACC const T& constAt(idx_t n, idx_t i, idx_t j) const { return fArray[(i * D2 + j) * N + n]; }

      ALPAKA_FN_HOST_ACC T& At(idx_t n, idx_t i, idx_t j) { return fArray[(i * D2 + j) * N + n]; }

      ALPAKA_FN_HOST_ACC T& operator()(idx_t n, idx_t i, idx_t j) { return fArray[(i * D2 + j) * N + n]; }
      ALPAKA_FN_HOST_ACC const T& operator()(idx_t n, idx_t i, idx_t j) const { return fArray[(i * D2 + j) * N + n]; }

      ALPAKA_FN_HOST_ACC Matriplex& operator=(T t) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = t;
        return *this;
      }

      ALPAKA_FN_HOST_ACC void copySlot(idx_t n, const Matriplex& m) {
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

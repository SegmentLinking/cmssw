#ifndef RecoTracker_MkFitAlpaka_interface_matriplex_MatriplexSym_h
#define RecoTracker_MkFitAlpaka_interface_matriplex_MatriplexSym_h

// Portable (host + Alpaka device) port of RecoTracker/MkFitCore/src/Matriplex/MatriplexSym.h (CMSSW_20_1_0_pre2).
// Same API, same packed lower-triangle layout fArray[symOffset(i, j) * N + n] (= gSymOffsets[D][i * D + j]),
// same arithmetic in the same order. See Matriplex.h for the list of differences.

#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatriplexCommon.h"
#include "RecoTracker/MkFitAlpaka/interface/matriplex/Matriplex.h"

//==============================================================================
// MatriplexSym
//==============================================================================

namespace mkfitdev {
  namespace Matriplex {

    //------------------------------------------------------------------------------

    template <typename T, idx_t D, idx_t N>
    class alignas(mplexAlign<T, N>()) MatriplexSym {
    public:
      typedef T value_type;

      /// no. of matrix rows
      static constexpr int kRows = D;
      /// no. of matrix columns
      static constexpr int kCols = D;
      /// no of elements: lower triangle
      static constexpr int kSize = (D + 1) * D / 2;
      /// size of the whole matriplex
      static constexpr int kTotSize = N * kSize;

      T fArray[kTotSize];

      ALPAKA_FN_HOST_ACC MatriplexSym() {}
      ALPAKA_FN_HOST_ACC MatriplexSym(T v) { setVal(v); }

      ALPAKA_FN_HOST_ACC idx_t plexSize() const { return N; }

      ALPAKA_FN_HOST_ACC void setVal(T v) {
        for (idx_t i = 0; i < kTotSize; ++i) {
          fArray[i] = v;
        }
      }

      ALPAKA_FN_HOST_ACC void add(const MatriplexSym& v) {
        for (idx_t i = 0; i < kTotSize; ++i) {
          fArray[i] += v.fArray[i];
        }
      }

      ALPAKA_FN_HOST_ACC void scale(T scale) {
        for (idx_t i = 0; i < kTotSize; ++i) {
          fArray[i] *= scale;
        }
      }

      ALPAKA_FN_HOST_ACC T operator[](idx_t xx) const { return fArray[xx]; }
      ALPAKA_FN_HOST_ACC T& operator[](idx_t xx) { return fArray[xx]; }

      // MkFitCore: gSymOffsets[D][i] for i = row * D + col.
      ALPAKA_FN_HOST_ACC idx_t off(idx_t i) const { return symOffset(i / D, i % D); }

      ALPAKA_FN_HOST_ACC const T& constAt(idx_t n, idx_t i, idx_t j) const { return fArray[symOffset(i, j) * N + n]; }

      ALPAKA_FN_HOST_ACC T& At(idx_t n, idx_t i, idx_t j) { return fArray[symOffset(i, j) * N + n]; }

      ALPAKA_FN_HOST_ACC T& operator()(idx_t n, idx_t i, idx_t j) { return At(n, i, j); }
      ALPAKA_FN_HOST_ACC const T& operator()(idx_t n, idx_t i, idx_t j) const { return constAt(n, i, j); }

      ALPAKA_FN_HOST_ACC MatriplexSym& operator=(const MatriplexSym& m) {
        // MkFitCore: memcpy
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

      ALPAKA_FN_HOST_ACC void copyIn(idx_t n, const MatriplexSym& m, idx_t in) {
        for (idx_t i = n; i < kTotSize; i += N, in += N) {
          fArray[i] = m[in];
        }
      }

      ALPAKA_FN_HOST_ACC void copy(idx_t n, idx_t in) {
        for (idx_t i = n; i < kTotSize; i += N, in += N) {
          fArray[i] = fArray[in];
        }
      }

      // MkFitCore generic (non-gather) slurpIn: element i  is arr[i + vi[j]] (vi in units of T).
      ALPAKA_FN_HOST_ACC void slurpIn(const T* arr, const int* vi, const int N_proc = N) {
        // Separate N_proc == N case (gains about 7% in fit test).
        if (N_proc == N) {
          for (int i = 0; i < kSize; ++i) {
            for (int j = 0; j < N; ++j) {
              fArray[i * N + j] = *(arr + i + vi[j]);
            }
          }
        } else {
          for (int i = 0; i < kSize; ++i) {
            for (int j = 0; j < N_proc; ++j) {
              fArray[i * N + j] = *(arr + i + vi[j]);
            }
          }
        }
      }

      ALPAKA_FN_HOST_ACC void copyOut(idx_t n, T* arr) const {
        for (idx_t i = n; i < kTotSize; i += N) {
          *(arr++) = fArray[i];
        }
      }

      ALPAKA_FN_HOST_ACC void setDiagonal3x3(idx_t n, T d) {
        T* p = fArray + n;

        p[0 * N] = d;
        p[1 * N] = 0;
        p[2 * N] = d;
        p[3 * N] = 0;
        p[4 * N] = 0;
        p[5 * N] = d;
      }

      ALPAKA_FN_HOST_ACC MatriplexSym& subtract(const MatriplexSym& a, const MatriplexSym& b) {
        // Does *this = a - b;

        MPLEX_SIMD
        for (idx_t i = 0; i < kTotSize; ++i) {
          fArray[i] = a.fArray[i] - b.fArray[i];
        }

        return *this;
      }

      // ==================================================================
      // Operations specific to Kalman fit in 6 parameter space
      // ==================================================================

      ALPAKA_FN_HOST_ACC void addNoiseIntoUpperLeft3x3(T noise) {
        T* p = fArray;

        MPLEX_SIMD
        for (idx_t n = 0; n < N; ++n) {
          p[0 * N + n] += noise;
          p[2 * N + n] += noise;
          p[5 * N + n] += noise;
        }
      }

      ALPAKA_FN_HOST_ACC void invertUpperLeft3x3() {
        typedef T TT;

        T* a = fArray;

        MPLEX_SIMD
        for (idx_t n = 0; n < N; ++n) {
          const TT c00 = a[2 * N + n] * a[5 * N + n] - a[4 * N + n] * a[4 * N + n];
          const TT c01 = a[4 * N + n] * a[3 * N + n] - a[1 * N + n] * a[5 * N + n];
          const TT c02 = a[1 * N + n] * a[4 * N + n] - a[2 * N + n] * a[3 * N + n];
          const TT c11 = a[5 * N + n] * a[0 * N + n] - a[3 * N + n] * a[3 * N + n];
          const TT c12 = a[3 * N + n] * a[1 * N + n] - a[4 * N + n] * a[0 * N + n];
          const TT c22 = a[0 * N + n] * a[2 * N + n] - a[1 * N + n] * a[1 * N + n];

          // Force determinant calculation in double precision.
          const double det = (double)a[0 * N + n] * c00 + (double)a[1 * N + n] * c01 + (double)a[3 * N + n] * c02;
          const TT s = TT(1) / det;

          a[0 * N + n] = s * c00;
          a[1 * N + n] = s * c01;
          a[2 * N + n] = s * c11;
          a[3 * N + n] = s * c02;
          a[4 * N + n] = s * c12;
          a[5 * N + n] = s * c22;
        }
      }

      ALPAKA_FN_HOST_ACC Matriplex<T, 1, 1, N> ReduceFixedIJ(idx_t i, idx_t j) const {
        Matriplex<T, 1, 1, N> t;
        for (idx_t n = 0; n < N; ++n) {
          t[n] = constAt(n, i, j);
        }
        return t;
      }
    };

    template <typename T, idx_t D, idx_t N>
    using MPlexSym = MatriplexSym<T, D, N>;

    //==============================================================================
    // Multiplications
    //==============================================================================

    template <typename T, idx_t D, idx_t N>
    struct SymMultiplyCls {
      static_assert(always_false_v<T>, "general symmetric multiplication not supported");
    };

    template <typename T, idx_t N>
    struct SymMultiplyCls<T, 3, N> {
      ALPAKA_FN_HOST_ACC static void multiply(const MPlexSym<T, 3, N>& A,
                                              const MPlexSym<T, 3, N>& B,
                                              MPlex<T, 3, 3, N>& C) {
        const T* a = A.fArray;
        const T* b = B.fArray;
        T* c = C.fArray;

        MPLEX_SIMD
        for (idx_t n = 0; n < N; ++n) {
#include "RecoTracker/MkFitAlpaka/interface/matriplex/std_sym_3x3.ah"
        }
      }
    };

    template <typename T, idx_t N>
    struct SymMultiplyCls<T, 6, N> {
      ALPAKA_FN_HOST_ACC static void multiply(const MPlexSym<float, 6, N>& A,
                                              const MPlexSym<float, 6, N>& B,
                                              MPlex<float, 6, 6, N>& C) {
        const T* a = A.fArray;
        const T* b = B.fArray;
        T* c = C.fArray;

        MPLEX_SIMD
        for (idx_t n = 0; n < N; ++n) {
#include "RecoTracker/MkFitAlpaka/interface/matriplex/std_sym_6x6.ah"
        }
      }
    };

    template <typename T, idx_t D, idx_t N>
    ALPAKA_FN_HOST_ACC void multiply(const MPlexSym<T, D, N>& A, const MPlexSym<T, D, N>& B, MPlex<T, D, D, N>& C) {
      SymMultiplyCls<T, D, N>::multiply(A, B, C);
    }

    //==============================================================================
    // Cramer inversion
    //==============================================================================

    template <typename T, idx_t D, idx_t N>
    struct CramerInverterSym {
      static_assert(always_false_v<T>, "general cramer inversion not supported");
    };

    template <typename T, idx_t N>
    struct CramerInverterSym<T, 2, N> {
      ALPAKA_FN_HOST_ACC static void invert(MPlexSym<T, 2, N>& A, double* determ = nullptr) {
        typedef T TT;

        T* a = A.fArray;

        MPLEX_SIMD
        for (idx_t n = 0; n < N; ++n) {
          // Force determinant calculation in double precision.
          const double det = (double)a[0 * N + n] * a[2 * N + n] - (double)a[1 * N + n] * a[1 * N + n];
          if (determ)
            determ[n] = det;

          const TT s = TT(1) / det;
          const TT tmp = s * a[2 * N + n];
          a[1 * N + n] *= -s;
          a[2 * N + n] = s * a[0 * N + n];
          a[0 * N + n] = tmp;
        }
      }
    };

    template <typename T, idx_t N>
    struct CramerInverterSym<T, 3, N> {
      ALPAKA_FN_HOST_ACC static void invert(MPlexSym<T, 3, N>& A, double* determ = nullptr) {
        typedef T TT;

        T* a = A.fArray;

        MPLEX_SIMD
        for (idx_t n = 0; n < N; ++n) {
          const TT c00 = a[2 * N + n] * a[5 * N + n] - a[4 * N + n] * a[4 * N + n];
          const TT c01 = a[4 * N + n] * a[3 * N + n] - a[1 * N + n] * a[5 * N + n];
          const TT c02 = a[1 * N + n] * a[4 * N + n] - a[2 * N + n] * a[3 * N + n];
          const TT c11 = a[5 * N + n] * a[0 * N + n] - a[3 * N + n] * a[3 * N + n];
          const TT c12 = a[3 * N + n] * a[1 * N + n] - a[4 * N + n] * a[0 * N + n];
          const TT c22 = a[0 * N + n] * a[2 * N + n] - a[1 * N + n] * a[1 * N + n];

          // Force determinant calculation in double precision.
          const double det = (double)a[0 * N + n] * c00 + (double)a[1 * N + n] * c01 + (double)a[3 * N + n] * c02;
          if (determ)
            determ[n] = det;

          const TT s = TT(1) / det;
          a[0 * N + n] = s * c00;
          a[1 * N + n] = s * c01;
          a[2 * N + n] = s * c11;
          a[3 * N + n] = s * c02;
          a[4 * N + n] = s * c12;
          a[5 * N + n] = s * c22;
        }
      }
    };

    template <typename T, idx_t D, idx_t N>
    ALPAKA_FN_HOST_ACC void invertCramerSym(MPlexSym<T, D, N>& A, double* determ = nullptr) {
      CramerInverterSym<T, D, N>::invert(A, determ);
    }

    //==============================================================================
    // Cholesky inversion
    //==============================================================================

    template <typename T, idx_t D, idx_t N>
    struct CholeskyInverterSym {
      static_assert(always_false_v<T>, "general cholesky inversion not supported");
    };

    template <typename T, idx_t N>
    struct CholeskyInverterSym<T, 3, N> {
      ALPAKA_FN_HOST_ACC static void invert(MPlexSym<T, 3, N>& A) {
        typedef T TT;

        T* a = A.fArray;

        MPLEX_SIMD
        for (idx_t n = 0; n < N; ++n) {
          TT l0 = std::sqrt(T(1) / a[0 * N + n]);
          TT l1 = a[1 * N + n] * l0;
          TT l2 = a[2 * N + n] - l1 * l1;
          l2 = std::sqrt(T(1) / l2);
          TT l3 = a[3 * N + n] * l0;
          TT l4 = (a[4 * N + n] - l1 * l3) * l2;
          TT l5 = a[5 * N + n] - (l3 * l3 + l4 * l4);
          l5 = std::sqrt(T(1) / l5);

          // decomposition done

          l3 = (l1 * l4 * l2 - l3) * l0 * l5;
          l1 = -l1 * l0 * l2;
          l4 = -l4 * l2 * l5;

          a[0 * N + n] = l3 * l3 + l1 * l1 + l0 * l0;
          a[1 * N + n] = l3 * l4 + l1 * l2;
          a[2 * N + n] = l4 * l4 + l2 * l2;
          a[3 * N + n] = l3 * l5;
          a[4 * N + n] = l4 * l5;
          a[5 * N + n] = l5 * l5;

          // m(2,x) are all zero if anything went wrong at l5.
          // all zero, if anything went wrong already for l0 or l2.
        }
      }
    };

    template <typename T, idx_t D, idx_t N>
    ALPAKA_FN_HOST_ACC void invertCholeskySym(MPlexSym<T, D, N>& A) {
      CholeskyInverterSym<T, D, N>::invert(A);
    }

  }  // namespace Matriplex
}  // namespace mkfitdev

#endif

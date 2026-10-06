#ifndef RecoTracker_MkFitAlpaka_interface_matriplex_Matriplex_h
#define RecoTracker_MkFitAlpaka_interface_matriplex_Matriplex_h

// Portable (host + Alpaka device) port of RecoTracker/MkFitCore/src/Matriplex/Matriplex.h (CMSSW_20_1_0_pre2).
// Same class, same API, same element layout fArray[(i * D2 + j) * N + n], same arithmetic in the same order.
// Differences: every function is ALPAKA_FN_HOST_ACC; no intrinsics (CMSSW builds do not enable them
// either); the general (unsupported D) cases are compile-time errors instead of exceptions; vdt functions are
// the portable mkfitdev::vdt transliterations (float only, as used by mkFit); slurpIn is the MkFitCore generic
// (non-gather) variant; alignment is the MkFitCore 32 bytes only when N * sizeof(T) fills whole vectors.

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatriplexCommon.h"
#include "RecoTracker/MkFitAlpaka/interface/math/vdtMath.h"

namespace mkfitdev {
  namespace Matriplex {

    template <typename>
    inline constexpr bool always_false_v = false;

    //------------------------------------------------------------------------------

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    class alignas(mplexAlign<T, N>()) Matriplex {
    public:
      typedef T value_type;

      /// return no. of matrix rows
      static constexpr int kRows = D1;
      /// return no. of matrix columns
      static constexpr int kCols = D2;
      /// return no of elements: rows*columns
      static constexpr int kSize = D1 * D2;
      /// size of the whole matriplex
      static constexpr int kTotSize = N * kSize;

      T fArray[kTotSize];

      ALPAKA_FN_HOST_ACC Matriplex() {}
      ALPAKA_FN_HOST_ACC Matriplex(T v) { setVal(v); }

      ALPAKA_FN_HOST_ACC idx_t plexSize() const { return N; }

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

      ALPAKA_FN_HOST_ACC void scale(T scale) {
        for (idx_t i = 0; i < kTotSize; ++i) {
          fArray[i] *= scale;
        }
      }

      ALPAKA_FN_HOST_ACC Matriplex& negate() {
        for (idx_t i = 0; i < kTotSize; ++i) {
          fArray[i] = -fArray[i];
        }
        return *this;
      }

      template <typename TT>
      ALPAKA_FN_HOST_ACC Matriplex& negate_if_ltz(const Matriplex<TT, D1, D2, N>& sign) {
        for (idx_t i = 0; i < kTotSize; ++i) {
          if (sign.fArray[i] < 0)
            fArray[i] = -fArray[i];
        }
        return *this;
      }

      ALPAKA_FN_HOST_ACC T operator[](idx_t xx) const { return fArray[xx]; }
      ALPAKA_FN_HOST_ACC T& operator[](idx_t xx) { return fArray[xx]; }

      ALPAKA_FN_HOST_ACC const T& constAt(idx_t n, idx_t i, idx_t j) const { return fArray[(i * D2 + j) * N + n]; }

      ALPAKA_FN_HOST_ACC T& At(idx_t n, idx_t i, idx_t j) { return fArray[(i * D2 + j) * N + n]; }

      ALPAKA_FN_HOST_ACC T& operator()(idx_t n, idx_t i, idx_t j) { return fArray[(i * D2 + j) * N + n]; }
      ALPAKA_FN_HOST_ACC const T& operator()(idx_t n, idx_t i, idx_t j) const { return fArray[(i * D2 + j) * N + n]; }

      // reduction operators

      using QReduced = Matriplex<T, 1, 1, N>;

      ALPAKA_FN_HOST_ACC QReduced ReduceFixedIJ(idx_t i, idx_t j) const {
        QReduced t;
        for (idx_t n = 0; n < N; ++n) {
          t[n] = constAt(n, i, j);
        }
        return t;
      }
      ALPAKA_FN_HOST_ACC QReduced rij(idx_t i, idx_t j) const { return ReduceFixedIJ(i, j); }
      ALPAKA_FN_HOST_ACC QReduced operator()(idx_t i, idx_t j) const { return ReduceFixedIJ(i, j); }

      struct QAssigner {
        Matriplex& m_matriplex;
        const int m_i, m_j;

        ALPAKA_FN_HOST_ACC QAssigner(Matriplex& m, int i, int j) : m_matriplex(m), m_i(i), m_j(j) {}
        ALPAKA_FN_HOST_ACC Matriplex& operator=(const QReduced& qvec) {
          for (idx_t n = 0; n < N; ++n) {
            m_matriplex(n, m_i, m_j) = qvec[n];
          }
          return m_matriplex;
        }
        ALPAKA_FN_HOST_ACC Matriplex& operator=(T qscalar) {
          for (idx_t n = 0; n < N; ++n) {
            m_matriplex(n, m_i, m_j) = qscalar;
          }
          return m_matriplex;
        }
      };

      ALPAKA_FN_HOST_ACC QAssigner AssignFixedIJ(idx_t i, idx_t j) { return QAssigner(*this, i, j); }
      ALPAKA_FN_HOST_ACC QAssigner aij(idx_t i, idx_t j) { return AssignFixedIJ(i, j); }

      // assignment operators

      ALPAKA_FN_HOST_ACC Matriplex& operator=(T t) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = t;
        return *this;
      }

      ALPAKA_FN_HOST_ACC Matriplex& operator+=(T t) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] += t;
        return *this;
      }

      ALPAKA_FN_HOST_ACC Matriplex& operator-=(T t) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] -= t;
        return *this;
      }

      ALPAKA_FN_HOST_ACC Matriplex& operator*=(T t) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] *= t;
        return *this;
      }

      ALPAKA_FN_HOST_ACC Matriplex& operator/=(T t) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] /= t;
        return *this;
      }

      ALPAKA_FN_HOST_ACC Matriplex& operator+=(const Matriplex& a) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] += a.fArray[i];
        return *this;
      }

      ALPAKA_FN_HOST_ACC Matriplex& operator-=(const Matriplex& a) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] -= a.fArray[i];
        return *this;
      }

      ALPAKA_FN_HOST_ACC Matriplex& operator*=(const Matriplex& a) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] *= a.fArray[i];
        return *this;
      }

      ALPAKA_FN_HOST_ACC Matriplex& operator/=(const Matriplex& a) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] /= a.fArray[i];
        return *this;
      }

      ALPAKA_FN_HOST_ACC Matriplex operator-() {
        Matriplex t;
        for (idx_t i = 0; i < kTotSize; ++i)
          t.fArray[i] = -fArray[i];
        return t;
      }

      ALPAKA_FN_HOST_ACC Matriplex& abs(const Matriplex& a) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = std::abs(a.fArray[i]);
        return *this;
      }
      ALPAKA_FN_HOST_ACC Matriplex& abs() {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = std::abs(fArray[i]);
        return *this;
      }

      ALPAKA_FN_HOST_ACC Matriplex& sqr(const Matriplex& a) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = a.fArray[i] * a.fArray[i];
        return *this;
      }
      ALPAKA_FN_HOST_ACC Matriplex& sqr() {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = fArray[i] * fArray[i];
        return *this;
      }

      //---------------------------------------------------------
      // transcendentals, std version

      ALPAKA_FN_HOST_ACC Matriplex& sqrt(const Matriplex& a) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = std::sqrt(a.fArray[i]);
        return *this;
      }
      ALPAKA_FN_HOST_ACC Matriplex& sqrt() {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = std::sqrt(fArray[i]);
        return *this;
      }

      ALPAKA_FN_HOST_ACC Matriplex& hypot(const Matriplex& a, const Matriplex& b) {
        for (idx_t i = 0; i < kTotSize; ++i) {
          fArray[i] = a.fArray[i] * a.fArray[i] + b.fArray[i] * b.fArray[i];
        }
        return sqrt();
      }

      ALPAKA_FN_HOST_ACC Matriplex& sin(const Matriplex& a) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = std::sin(a.fArray[i]);
        return *this;
      }
      ALPAKA_FN_HOST_ACC Matriplex& sin() {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = std::sin(fArray[i]);
        return *this;
      }

      ALPAKA_FN_HOST_ACC Matriplex& cos(const Matriplex& a) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = std::cos(a.fArray[i]);
        return *this;
      }
      ALPAKA_FN_HOST_ACC Matriplex& cos() {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = std::cos(fArray[i]);
        return *this;
      }

      ALPAKA_FN_HOST_ACC Matriplex& tan(const Matriplex& a) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = std::tan(a.fArray[i]);
        return *this;
      }
      ALPAKA_FN_HOST_ACC Matriplex& tan() {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = std::tan(fArray[i]);
        return *this;
      }

      ALPAKA_FN_HOST_ACC Matriplex& atan2(const Matriplex& y, const Matriplex& x) {
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = std::atan2(y.fArray[i], x.fArray[i]);
        return *this;
      }

      //---------------------------------------------------------
      // transcendentals, vdt version (MPLEX_VDT on, MPLEX_VDT_USE_STD off). Float only.

      ALPAKA_FN_HOST_ACC Matriplex& fast_isqrt(const Matriplex& a) {
        static_assert(std::is_same_v<T, float>, "mkfitdev vdt port is float only");
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = vdt::fast_isqrtf(a.fArray[i]);
        return *this;
      }
      ALPAKA_FN_HOST_ACC Matriplex& fast_isqrt() {
        static_assert(std::is_same_v<T, float>, "mkfitdev vdt port is float only");
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = vdt::fast_isqrtf(fArray[i]);
        return *this;
      }

      ALPAKA_FN_HOST_ACC Matriplex& fast_sin(const Matriplex& a) {
        static_assert(std::is_same_v<T, float>, "mkfitdev vdt port is float only");
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = vdt::fast_sinf(a.fArray[i]);
        return *this;
      }
      ALPAKA_FN_HOST_ACC Matriplex& fast_sin() {
        static_assert(std::is_same_v<T, float>, "mkfitdev vdt port is float only");
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = vdt::fast_sinf(fArray[i]);
        return *this;
      }

      ALPAKA_FN_HOST_ACC Matriplex& fast_cos(const Matriplex& a) {
        static_assert(std::is_same_v<T, float>, "mkfitdev vdt port is float only");
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = vdt::fast_cosf(a.fArray[i]);
        return *this;
      }
      ALPAKA_FN_HOST_ACC Matriplex& fast_cos() {
        static_assert(std::is_same_v<T, float>, "mkfitdev vdt port is float only");
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = vdt::fast_cosf(fArray[i]);
        return *this;
      }

      ALPAKA_FN_HOST_ACC void fast_sincos(Matriplex& s, Matriplex& c) const {
        static_assert(std::is_same_v<T, float>, "mkfitdev vdt port is float only");
        for (idx_t i = 0; i < kTotSize; ++i)
          vdt::fast_sincosf(fArray[i], s.fArray[i], c.fArray[i]);
      }

      ALPAKA_FN_HOST_ACC Matriplex& fast_tan(const Matriplex& a) {
        static_assert(std::is_same_v<T, float>, "mkfitdev vdt port is float only");
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = vdt::fast_tanf(a.fArray[i]);
        return *this;
      }
      ALPAKA_FN_HOST_ACC Matriplex& fast_tan() {
        static_assert(std::is_same_v<T, float>, "mkfitdev vdt port is float only");
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = vdt::fast_tanf(fArray[i]);
        return *this;
      }

      ALPAKA_FN_HOST_ACC Matriplex& fast_atan2(const Matriplex& y, const Matriplex& x) {
        static_assert(std::is_same_v<T, float>, "mkfitdev vdt port is float only");
        for (idx_t i = 0; i < kTotSize; ++i)
          fArray[i] = vdt::fast_atan2f(y.fArray[i], x.fArray[i]);
        return *this;
      }

      ALPAKA_FN_HOST_ACC void sincos4(Matriplex& s, Matriplex& c) const {
        for (idx_t i = 0; i < kTotSize; ++i)
          internal::sincos4(fArray[i], s.fArray[i], c.fArray[i]);
      }

      //---------------------------------------------------------

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

      ALPAKA_FN_HOST_ACC void copyIn(idx_t n, const Matriplex& m, idx_t in) {
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
    };

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    using MPlex = Matriplex<T, D1, D2, N>;

    //==============================================================================
    // Operators
    //==============================================================================

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> operator-(const MPlex<T, D1, D2, N>& a) {
      MPlex<T, D1, D2, N> t = a;
      t.negate();
      return t;
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> negate(const MPlex<T, D1, D2, N>& a) {
      MPlex<T, D1, D2, N> t = a;
      t.negate();
      return t;
    }

    template <typename T, typename TT, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> negate_if_ltz(const MPlex<T, D1, D2, N>& a,
                                                         const MPlex<TT, D1, D2, N>& sign) {
      MPlex<T, D1, D2, N> t = a;
      t.negate_if_ltz(sign);
      return t;
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> operator+(const MPlex<T, D1, D2, N>& a, const MPlex<T, D1, D2, N>& b) {
      MPlex<T, D1, D2, N> t = a;
      t += b;
      return t;
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> operator-(const MPlex<T, D1, D2, N>& a, const MPlex<T, D1, D2, N>& b) {
      MPlex<T, D1, D2, N> t = a;
      t -= b;
      return t;
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> operator*(const MPlex<T, D1, D2, N>& a, const MPlex<T, D1, D2, N>& b) {
      MPlex<T, D1, D2, N> t = a;
      t *= b;
      return t;
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> operator/(const MPlex<T, D1, D2, N>& a, const MPlex<T, D1, D2, N>& b) {
      MPlex<T, D1, D2, N> t = a;
      t /= b;
      return t;
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> operator+(const MPlex<T, D1, D2, N>& a, T b) {
      MPlex<T, D1, D2, N> t = a;
      t += b;
      return t;
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> operator-(const MPlex<T, D1, D2, N>& a, T b) {
      MPlex<T, D1, D2, N> t = a;
      t -= b;
      return t;
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> operator*(const MPlex<T, D1, D2, N>& a, T b) {
      MPlex<T, D1, D2, N> t = a;
      t *= b;
      return t;
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> operator/(const MPlex<T, D1, D2, N>& a, T b) {
      MPlex<T, D1, D2, N> t = a;
      t /= b;
      return t;
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> operator+(T a, const MPlex<T, D1, D2, N>& b) {
      MPlex<T, D1, D2, N> t = a;
      t += b;
      return t;
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> operator-(T a, const MPlex<T, D1, D2, N>& b) {
      MPlex<T, D1, D2, N> t = a;
      t -= b;
      return t;
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> operator*(T a, const MPlex<T, D1, D2, N>& b) {
      MPlex<T, D1, D2, N> t = a;
      t *= b;
      return t;
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> operator/(T a, const MPlex<T, D1, D2, N>& b) {
      MPlex<T, D1, D2, N> t = a;
      t /= b;
      return t;
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> abs(const MPlex<T, D1, D2, N>& a) {
      MPlex<T, D1, D2, N> t;
      return t.abs(a);
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> sqr(const MPlex<T, D1, D2, N>& a) {
      MPlex<T, D1, D2, N> t;
      return t.sqr(a);
    }

    //---------------------------------------------------------
    // transcendentals, std version

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> sqrt(const MPlex<T, D1, D2, N>& a) {
      MPlex<T, D1, D2, N> t;
      return t.sqrt(a);
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> hypot(const MPlex<T, D1, D2, N>& a, const MPlex<T, D1, D2, N>& b) {
      MPlex<T, D1, D2, N> t;
      return t.hypot(a, b);
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> sin(const MPlex<T, D1, D2, N>& a) {
      MPlex<T, D1, D2, N> t;
      return t.sin(a);
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> cos(const MPlex<T, D1, D2, N>& a) {
      MPlex<T, D1, D2, N> t;
      return t.cos(a);
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC void sincos(const MPlex<T, D1, D2, N>& a, MPlex<T, D1, D2, N>& s, MPlex<T, D1, D2, N>& c) {
      for (idx_t i = 0; i < a.kTotSize; ++i) {
        s.fArray[i] = std::sin(a.fArray[i]);
        c.fArray[i] = std::cos(a.fArray[i]);
      }
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> tan(const MPlex<T, D1, D2, N>& a) {
      MPlex<T, D1, D2, N> t;
      return t.tan(a);
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> atan2(const MPlex<T, D1, D2, N>& y, const MPlex<T, D1, D2, N>& x) {
      MPlex<T, D1, D2, N> t;
      return t.atan2(y, x);
    }

    //---------------------------------------------------------
    // transcendentals, vdt version

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> fast_isqrt(const MPlex<T, D1, D2, N>& a) {
      MPlex<T, D1, D2, N> t;
      return t.fast_isqrt(a);
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> fast_sin(const MPlex<T, D1, D2, N>& a) {
      MPlex<T, D1, D2, N> t;
      return t.fast_sin(a);
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> fast_cos(const MPlex<T, D1, D2, N>& a) {
      MPlex<T, D1, D2, N> t;
      return t.fast_cos(a);
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC void fast_sincos(const MPlex<T, D1, D2, N>& a, MPlex<T, D1, D2, N>& s, MPlex<T, D1, D2, N>& c) {
      a.fast_sincos(s, c);
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> fast_tan(const MPlex<T, D1, D2, N>& a) {
      MPlex<T, D1, D2, N> t;
      return t.fast_tan(a);
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> fast_atan2(const MPlex<T, D1, D2, N>& y, const MPlex<T, D1, D2, N>& x) {
      MPlex<T, D1, D2, N> t;
      return t.fast_atan2(y, x);
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC void sincos4(const MPlex<T, D1, D2, N>& a, MPlex<T, D1, D2, N>& s, MPlex<T, D1, D2, N>& c) {
      a.sincos4(s, c);
    }

    //---------------------------------------------------------

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC void min_max(const MPlex<T, D1, D2, N>& a,
                                    const MPlex<T, D1, D2, N>& b,
                                    MPlex<T, D1, D2, N>& min,
                                    MPlex<T, D1, D2, N>& max) {
      for (idx_t i = 0; i < a.kTotSize; ++i) {
        min.fArray[i] = std::min(a.fArray[i], b.fArray[i]);
        max.fArray[i] = std::max(a.fArray[i], b.fArray[i]);
      }
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> min(const MPlex<T, D1, D2, N>& a, const MPlex<T, D1, D2, N>& b) {
      MPlex<T, D1, D2, N> t;
      for (idx_t i = 0; i < a.kTotSize; ++i) {
        t.fArray[i] = std::min(a.fArray[i], b.fArray[i]);
      }
      return t;
    }

    template <typename T, idx_t D1, idx_t D2, idx_t N>
    ALPAKA_FN_HOST_ACC MPlex<T, D1, D2, N> max(const MPlex<T, D1, D2, N>& a, const MPlex<T, D1, D2, N>& b) {
      MPlex<T, D1, D2, N> t;
      for (idx_t i = 0; i < a.kTotSize; ++i) {
        t.fArray[i] = std::max(a.fArray[i], b.fArray[i]);
      }
      return t;
    }

    //==============================================================================
    // Multiplications
    //==============================================================================

    template <typename T, idx_t D1, idx_t D2, idx_t D3, idx_t N>
    ALPAKA_FN_HOST_ACC void multiplyGeneral(const MPlex<T, D1, D2, N>& A,
                                            const MPlex<T, D2, D3, N>& B,
                                            MPlex<T, D1, D3, N>& C) {
      for (idx_t i = 0; i < D1; ++i) {
        for (idx_t j = 0; j < D3; ++j) {
          const idx_t ijo = N * (i * D3 + j);

          MPLEX_SIMD
          for (idx_t n = 0; n < N; ++n) {
            C.fArray[ijo + n] = 0;
          }

          for (idx_t k = 0; k < D2; ++k) {
            const idx_t iko = N * (i * D2 + k);
            const idx_t kjo = N * (k * D3 + j);

            MPLEX_SIMD
            for (idx_t n = 0; n < N; ++n) {
              C.fArray[ijo + n] += A.fArray[iko + n] * B.fArray[kjo + n];
            }
          }
        }
      }
    }

    //------------------------------------------------------------------------------

    template <typename T, idx_t D, idx_t N>
    struct MultiplyCls {
      static_assert(always_false_v<T>, "general multiplication not supported, well, call multiplyGeneral()");
    };

    template <typename T, idx_t N>
    struct MultiplyCls<T, 3, N> {
      ALPAKA_FN_HOST_ACC static void multiply(const MPlex<T, 3, 3, N>& A,
                                              const MPlex<T, 3, 3, N>& B,
                                              MPlex<T, 3, 3, N>& C) {
        const T* a = A.fArray;
        const T* b = B.fArray;
        T* c = C.fArray;

        MPLEX_SIMD
        for (idx_t n = 0; n < N; ++n) {
          c[0 * N + n] = a[0 * N + n] * b[0 * N + n] + a[1 * N + n] * b[3 * N + n] + a[2 * N + n] * b[6 * N + n];
          c[1 * N + n] = a[0 * N + n] * b[1 * N + n] + a[1 * N + n] * b[4 * N + n] + a[2 * N + n] * b[7 * N + n];
          c[2 * N + n] = a[0 * N + n] * b[2 * N + n] + a[1 * N + n] * b[5 * N + n] + a[2 * N + n] * b[8 * N + n];
          c[3 * N + n] = a[3 * N + n] * b[0 * N + n] + a[4 * N + n] * b[3 * N + n] + a[5 * N + n] * b[6 * N + n];
          c[4 * N + n] = a[3 * N + n] * b[1 * N + n] + a[4 * N + n] * b[4 * N + n] + a[5 * N + n] * b[7 * N + n];
          c[5 * N + n] = a[3 * N + n] * b[2 * N + n] + a[4 * N + n] * b[5 * N + n] + a[5 * N + n] * b[8 * N + n];
          c[6 * N + n] = a[6 * N + n] * b[0 * N + n] + a[7 * N + n] * b[3 * N + n] + a[8 * N + n] * b[6 * N + n];
          c[7 * N + n] = a[6 * N + n] * b[1 * N + n] + a[7 * N + n] * b[4 * N + n] + a[8 * N + n] * b[7 * N + n];
          c[8 * N + n] = a[6 * N + n] * b[2 * N + n] + a[7 * N + n] * b[5 * N + n] + a[8 * N + n] * b[8 * N + n];
        }
      }
    };

    template <typename T, idx_t N>
    struct MultiplyCls<T, 6, N> {
      ALPAKA_FN_HOST_ACC static void multiply(const MPlex<T, 6, 6, N>& A,
                                              const MPlex<T, 6, 6, N>& B,
                                              MPlex<T, 6, 6, N>& C) {
        const T* a = A.fArray;
        const T* b = B.fArray;
        T* c = C.fArray;

        MPLEX_SIMD
        for (idx_t n = 0; n < N; ++n) {
          c[0 * N + n] = a[0 * N + n] * b[0 * N + n] + a[1 * N + n] * b[6 * N + n] + a[2 * N + n] * b[12 * N + n] +
                         a[3 * N + n] * b[18 * N + n] + a[4 * N + n] * b[24 * N + n] + a[5 * N + n] * b[30 * N + n];
          c[1 * N + n] = a[0 * N + n] * b[1 * N + n] + a[1 * N + n] * b[7 * N + n] + a[2 * N + n] * b[13 * N + n] +
                         a[3 * N + n] * b[19 * N + n] + a[4 * N + n] * b[25 * N + n] + a[5 * N + n] * b[31 * N + n];
          c[2 * N + n] = a[0 * N + n] * b[2 * N + n] + a[1 * N + n] * b[8 * N + n] + a[2 * N + n] * b[14 * N + n] +
                         a[3 * N + n] * b[20 * N + n] + a[4 * N + n] * b[26 * N + n] + a[5 * N + n] * b[32 * N + n];
          c[3 * N + n] = a[0 * N + n] * b[3 * N + n] + a[1 * N + n] * b[9 * N + n] + a[2 * N + n] * b[15 * N + n] +
                         a[3 * N + n] * b[21 * N + n] + a[4 * N + n] * b[27 * N + n] + a[5 * N + n] * b[33 * N + n];
          c[4 * N + n] = a[0 * N + n] * b[4 * N + n] + a[1 * N + n] * b[10 * N + n] + a[2 * N + n] * b[16 * N + n] +
                         a[3 * N + n] * b[22 * N + n] + a[4 * N + n] * b[28 * N + n] + a[5 * N + n] * b[34 * N + n];
          c[5 * N + n] = a[0 * N + n] * b[5 * N + n] + a[1 * N + n] * b[11 * N + n] + a[2 * N + n] * b[17 * N + n] +
                         a[3 * N + n] * b[23 * N + n] + a[4 * N + n] * b[29 * N + n] + a[5 * N + n] * b[35 * N + n];
          c[6 * N + n] = a[6 * N + n] * b[0 * N + n] + a[7 * N + n] * b[6 * N + n] + a[8 * N + n] * b[12 * N + n] +
                         a[9 * N + n] * b[18 * N + n] + a[10 * N + n] * b[24 * N + n] + a[11 * N + n] * b[30 * N + n];
          c[7 * N + n] = a[6 * N + n] * b[1 * N + n] + a[7 * N + n] * b[7 * N + n] + a[8 * N + n] * b[13 * N + n] +
                         a[9 * N + n] * b[19 * N + n] + a[10 * N + n] * b[25 * N + n] + a[11 * N + n] * b[31 * N + n];
          c[8 * N + n] = a[6 * N + n] * b[2 * N + n] + a[7 * N + n] * b[8 * N + n] + a[8 * N + n] * b[14 * N + n] +
                         a[9 * N + n] * b[20 * N + n] + a[10 * N + n] * b[26 * N + n] + a[11 * N + n] * b[32 * N + n];
          c[9 * N + n] = a[6 * N + n] * b[3 * N + n] + a[7 * N + n] * b[9 * N + n] + a[8 * N + n] * b[15 * N + n] +
                         a[9 * N + n] * b[21 * N + n] + a[10 * N + n] * b[27 * N + n] + a[11 * N + n] * b[33 * N + n];
          c[10 * N + n] = a[6 * N + n] * b[4 * N + n] + a[7 * N + n] * b[10 * N + n] + a[8 * N + n] * b[16 * N + n] +
                          a[9 * N + n] * b[22 * N + n] + a[10 * N + n] * b[28 * N + n] + a[11 * N + n] * b[34 * N + n];
          c[11 * N + n] = a[6 * N + n] * b[5 * N + n] + a[7 * N + n] * b[11 * N + n] + a[8 * N + n] * b[17 * N + n] +
                          a[9 * N + n] * b[23 * N + n] + a[10 * N + n] * b[29 * N + n] + a[11 * N + n] * b[35 * N + n];
          c[12 * N + n] = a[12 * N + n] * b[0 * N + n] + a[13 * N + n] * b[6 * N + n] + a[14 * N + n] * b[12 * N + n] +
                          a[15 * N + n] * b[18 * N + n] + a[16 * N + n] * b[24 * N + n] + a[17 * N + n] * b[30 * N + n];
          c[13 * N + n] = a[12 * N + n] * b[1 * N + n] + a[13 * N + n] * b[7 * N + n] + a[14 * N + n] * b[13 * N + n] +
                          a[15 * N + n] * b[19 * N + n] + a[16 * N + n] * b[25 * N + n] + a[17 * N + n] * b[31 * N + n];
          c[14 * N + n] = a[12 * N + n] * b[2 * N + n] + a[13 * N + n] * b[8 * N + n] + a[14 * N + n] * b[14 * N + n] +
                          a[15 * N + n] * b[20 * N + n] + a[16 * N + n] * b[26 * N + n] + a[17 * N + n] * b[32 * N + n];
          c[15 * N + n] = a[12 * N + n] * b[3 * N + n] + a[13 * N + n] * b[9 * N + n] + a[14 * N + n] * b[15 * N + n] +
                          a[15 * N + n] * b[21 * N + n] + a[16 * N + n] * b[27 * N + n] + a[17 * N + n] * b[33 * N + n];
          c[16 * N + n] = a[12 * N + n] * b[4 * N + n] + a[13 * N + n] * b[10 * N + n] + a[14 * N + n] * b[16 * N + n] +
                          a[15 * N + n] * b[22 * N + n] + a[16 * N + n] * b[28 * N + n] + a[17 * N + n] * b[34 * N + n];
          c[17 * N + n] = a[12 * N + n] * b[5 * N + n] + a[13 * N + n] * b[11 * N + n] + a[14 * N + n] * b[17 * N + n] +
                          a[15 * N + n] * b[23 * N + n] + a[16 * N + n] * b[29 * N + n] + a[17 * N + n] * b[35 * N + n];
          c[18 * N + n] = a[18 * N + n] * b[0 * N + n] + a[19 * N + n] * b[6 * N + n] + a[20 * N + n] * b[12 * N + n] +
                          a[21 * N + n] * b[18 * N + n] + a[22 * N + n] * b[24 * N + n] + a[23 * N + n] * b[30 * N + n];
          c[19 * N + n] = a[18 * N + n] * b[1 * N + n] + a[19 * N + n] * b[7 * N + n] + a[20 * N + n] * b[13 * N + n] +
                          a[21 * N + n] * b[19 * N + n] + a[22 * N + n] * b[25 * N + n] + a[23 * N + n] * b[31 * N + n];
          c[20 * N + n] = a[18 * N + n] * b[2 * N + n] + a[19 * N + n] * b[8 * N + n] + a[20 * N + n] * b[14 * N + n] +
                          a[21 * N + n] * b[20 * N + n] + a[22 * N + n] * b[26 * N + n] + a[23 * N + n] * b[32 * N + n];
          c[21 * N + n] = a[18 * N + n] * b[3 * N + n] + a[19 * N + n] * b[9 * N + n] + a[20 * N + n] * b[15 * N + n] +
                          a[21 * N + n] * b[21 * N + n] + a[22 * N + n] * b[27 * N + n] + a[23 * N + n] * b[33 * N + n];
          c[22 * N + n] = a[18 * N + n] * b[4 * N + n] + a[19 * N + n] * b[10 * N + n] + a[20 * N + n] * b[16 * N + n] +
                          a[21 * N + n] * b[22 * N + n] + a[22 * N + n] * b[28 * N + n] + a[23 * N + n] * b[34 * N + n];
          c[23 * N + n] = a[18 * N + n] * b[5 * N + n] + a[19 * N + n] * b[11 * N + n] + a[20 * N + n] * b[17 * N + n] +
                          a[21 * N + n] * b[23 * N + n] + a[22 * N + n] * b[29 * N + n] + a[23 * N + n] * b[35 * N + n];
          c[24 * N + n] = a[24 * N + n] * b[0 * N + n] + a[25 * N + n] * b[6 * N + n] + a[26 * N + n] * b[12 * N + n] +
                          a[27 * N + n] * b[18 * N + n] + a[28 * N + n] * b[24 * N + n] + a[29 * N + n] * b[30 * N + n];
          c[25 * N + n] = a[24 * N + n] * b[1 * N + n] + a[25 * N + n] * b[7 * N + n] + a[26 * N + n] * b[13 * N + n] +
                          a[27 * N + n] * b[19 * N + n] + a[28 * N + n] * b[25 * N + n] + a[29 * N + n] * b[31 * N + n];
          c[26 * N + n] = a[24 * N + n] * b[2 * N + n] + a[25 * N + n] * b[8 * N + n] + a[26 * N + n] * b[14 * N + n] +
                          a[27 * N + n] * b[20 * N + n] + a[28 * N + n] * b[26 * N + n] + a[29 * N + n] * b[32 * N + n];
          c[27 * N + n] = a[24 * N + n] * b[3 * N + n] + a[25 * N + n] * b[9 * N + n] + a[26 * N + n] * b[15 * N + n] +
                          a[27 * N + n] * b[21 * N + n] + a[28 * N + n] * b[27 * N + n] + a[29 * N + n] * b[33 * N + n];
          c[28 * N + n] = a[24 * N + n] * b[4 * N + n] + a[25 * N + n] * b[10 * N + n] + a[26 * N + n] * b[16 * N + n] +
                          a[27 * N + n] * b[22 * N + n] + a[28 * N + n] * b[28 * N + n] + a[29 * N + n] * b[34 * N + n];
          c[29 * N + n] = a[24 * N + n] * b[5 * N + n] + a[25 * N + n] * b[11 * N + n] + a[26 * N + n] * b[17 * N + n] +
                          a[27 * N + n] * b[23 * N + n] + a[28 * N + n] * b[29 * N + n] + a[29 * N + n] * b[35 * N + n];
          c[30 * N + n] = a[30 * N + n] * b[0 * N + n] + a[31 * N + n] * b[6 * N + n] + a[32 * N + n] * b[12 * N + n] +
                          a[33 * N + n] * b[18 * N + n] + a[34 * N + n] * b[24 * N + n] + a[35 * N + n] * b[30 * N + n];
          c[31 * N + n] = a[30 * N + n] * b[1 * N + n] + a[31 * N + n] * b[7 * N + n] + a[32 * N + n] * b[13 * N + n] +
                          a[33 * N + n] * b[19 * N + n] + a[34 * N + n] * b[25 * N + n] + a[35 * N + n] * b[31 * N + n];
          c[32 * N + n] = a[30 * N + n] * b[2 * N + n] + a[31 * N + n] * b[8 * N + n] + a[32 * N + n] * b[14 * N + n] +
                          a[33 * N + n] * b[20 * N + n] + a[34 * N + n] * b[26 * N + n] + a[35 * N + n] * b[32 * N + n];
          c[33 * N + n] = a[30 * N + n] * b[3 * N + n] + a[31 * N + n] * b[9 * N + n] + a[32 * N + n] * b[15 * N + n] +
                          a[33 * N + n] * b[21 * N + n] + a[34 * N + n] * b[27 * N + n] + a[35 * N + n] * b[33 * N + n];
          c[34 * N + n] = a[30 * N + n] * b[4 * N + n] + a[31 * N + n] * b[10 * N + n] + a[32 * N + n] * b[16 * N + n] +
                          a[33 * N + n] * b[22 * N + n] + a[34 * N + n] * b[28 * N + n] + a[35 * N + n] * b[34 * N + n];
          c[35 * N + n] = a[30 * N + n] * b[5 * N + n] + a[31 * N + n] * b[11 * N + n] + a[32 * N + n] * b[17 * N + n] +
                          a[33 * N + n] * b[23 * N + n] + a[34 * N + n] * b[29 * N + n] + a[35 * N + n] * b[35 * N + n];
        }
      }
    };

    template <typename T, idx_t D, idx_t N>
    ALPAKA_FN_HOST_ACC void multiply(const MPlex<T, D, D, N>& A, const MPlex<T, D, D, N>& B, MPlex<T, D, D, N>& C) {
      MultiplyCls<T, D, N>::multiply(A, B, C);
    }

    //==============================================================================
    // Cramer inversion
    //==============================================================================

    template <typename T, idx_t D, idx_t N>
    struct CramerInverter {
      static_assert(always_false_v<T>, "general cramer inversion not supported");
    };

    template <typename T, idx_t N>
    struct CramerInverter<T, 2, N> {
      ALPAKA_FN_HOST_ACC static void invert(MPlex<T, 2, 2, N>& A, double* determ = nullptr) {
        typedef T TT;

        T* a = A.fArray;

        MPLEX_SIMD
        for (idx_t n = 0; n < N; ++n) {
          // Force determinant calculation in double precision.
          const double det = (double)a[0 * N + n] * a[3 * N + n] - (double)a[2 * N + n] * a[1 * N + n];
          if (determ)
            determ[n] = det;

          const TT s = TT(1) / det;
          const TT tmp = s * a[3 * N + n];
          a[1 * N + n] *= -s;
          a[2 * N + n] *= -s;
          a[3 * N + n] = s * a[0 * N + n];
          a[0 * N + n] = tmp;
        }
      }
    };

    template <typename T, idx_t N>
    struct CramerInverter<T, 3, N> {
      ALPAKA_FN_HOST_ACC static void invert(MPlex<T, 3, 3, N>& A, double* determ = nullptr) {
        typedef T TT;

        T* a = A.fArray;

        MPLEX_SIMD
        for (idx_t n = 0; n < N; ++n) {
          const TT c00 = a[4 * N + n] * a[8 * N + n] - a[5 * N + n] * a[7 * N + n];
          const TT c01 = a[5 * N + n] * a[6 * N + n] - a[3 * N + n] * a[8 * N + n];
          const TT c02 = a[3 * N + n] * a[7 * N + n] - a[4 * N + n] * a[6 * N + n];
          const TT c10 = a[7 * N + n] * a[2 * N + n] - a[8 * N + n] * a[1 * N + n];
          const TT c11 = a[8 * N + n] * a[0 * N + n] - a[6 * N + n] * a[2 * N + n];
          const TT c12 = a[6 * N + n] * a[1 * N + n] - a[7 * N + n] * a[0 * N + n];
          const TT c20 = a[1 * N + n] * a[5 * N + n] - a[2 * N + n] * a[4 * N + n];
          const TT c21 = a[2 * N + n] * a[3 * N + n] - a[0 * N + n] * a[5 * N + n];
          const TT c22 = a[0 * N + n] * a[4 * N + n] - a[1 * N + n] * a[3 * N + n];

          // Force determinant calculation in double precision.
          const double det = (double)a[0 * N + n] * c00 + (double)a[1 * N + n] * c01 + (double)a[2 * N + n] * c02;
          if (determ)
            determ[n] = det;

          const TT s = TT(1) / det;
          a[0 * N + n] = s * c00;
          a[1 * N + n] = s * c10;
          a[2 * N + n] = s * c20;
          a[3 * N + n] = s * c01;
          a[4 * N + n] = s * c11;
          a[5 * N + n] = s * c21;
          a[6 * N + n] = s * c02;
          a[7 * N + n] = s * c12;
          a[8 * N + n] = s * c22;
        }
      }
    };

    template <typename T, idx_t D, idx_t N>
    ALPAKA_FN_HOST_ACC void invertCramer(MPlex<T, D, D, N>& A, double* determ = nullptr) {
      CramerInverter<T, D, N>::invert(A, determ);
    }

    //==============================================================================
    // Cholesky inversion
    //==============================================================================

    template <typename T, idx_t D, idx_t N>
    struct CholeskyInverter {
      static_assert(always_false_v<T>, "general cholesky inversion not supported");
    };

    template <typename T, idx_t N>
    struct CholeskyInverter<T, 3, N> {
      // Note: this only works on symmetric matrices.
      // Optimized version for positive definite matrices, no checks.
      ALPAKA_FN_HOST_ACC static void invert(MPlex<T, 3, 3, N>& A) {
        typedef T TT;

        T* a = A.fArray;

        MPLEX_SIMD
        for (idx_t n = 0; n < N; ++n) {
          TT l0 = std::sqrt(T(1) / a[0 * N + n]);
          TT l1 = a[3 * N + n] * l0;
          TT l2 = a[4 * N + n] - l1 * l1;
          l2 = std::sqrt(T(1) / l2);
          TT l3 = a[6 * N + n] * l0;
          TT l4 = (a[7 * N + n] - l1 * l3) * l2;
          TT l5 = a[8 * N + n] - (l3 * l3 + l4 * l4);
          l5 = std::sqrt(T(1) / l5);

          // decomposition done

          l3 = (l1 * l4 * l2 - l3) * l0 * l5;
          l1 = -l1 * l0 * l2;
          l4 = -l4 * l2 * l5;

          a[0 * N + n] = l3 * l3 + l1 * l1 + l0 * l0;
          a[1 * N + n] = a[3 * N + n] = l3 * l4 + l1 * l2;
          a[4 * N + n] = l4 * l4 + l2 * l2;
          a[2 * N + n] = a[6 * N + n] = l3 * l5;
          a[5 * N + n] = a[7 * N + n] = l4 * l5;
          a[8 * N + n] = l5 * l5;

          // m(2,x) are all zero if anything went wrong at l5.
          // all zero, if anything went wrong already for l0 or l2.
        }
      }
    };

    template <typename T, idx_t D, idx_t N>
    ALPAKA_FN_HOST_ACC void invertCholesky(MPlex<T, D, D, N>& A) {
      CholeskyInverter<T, D, N>::invert(A);
    }

  }  // namespace Matriplex
}  // namespace mkfitdev

#endif

#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederKernelsCpu_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederKernelsCpu_h

// The chain finder's kernels on the CPU backends (included by MkSeederKernels.h, after the helpers): one thread runs
// the whole grid there (MkSeederDevice.dev.cc launches one block), so the GPU's splits and passes only cost time.
// These paths keep the GPU kernels' arithmetic expression by expression and push in the same order; the differences
// (per-stage details are at the functions): stage c has one item per candidate instead of kRows items, and the misses
// are forwarded in the same loop; stages b, c and d test the cuts with a loose AVX2 prefilter before the exact test.
// Only the serial and TBB backends instantiate them (kCpuAcc), and the GPU kernels are unchanged.

#include <bit>
#include <cstring>
#include <type_traits>
#include <vector>

#include "MkSeederBackend.h"
#include "MkSeederStageCNet.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  // (kCpuAcc, kSerialAcc: MkSeederBackend.h; the serial backend stores the candidates' lines and crossed positions at
  // the push, CandidateLine)
  // stage b's gap-ray check in cpu::ScanStart
  template <typename TAcc>
#ifdef MKSEEDER_CPU_AVX2
  constexpr bool kScanGapRay = kCpuAcc<TAcc>;
#else
  constexpr bool kScanGapRay = false;
#endif

  namespace cpu {

    // nextCrossedPosition() with the chain positions' envelopes as arrays, 8 positions per step (AVX2); the same expressions as
    // state() / crossingClass(), so the same result
    struct Router {
      static constexpr int kMax = 128;
      int n = 0, ot_max_pix = 0;
      alignas(32) float pos[kMax], plo[kMax], phi[kMax], lo_in[kMax], hi_in[kMax], lo_out[kMax], hi_out[kMax];
      alignas(32) int ok[kMax], disc[kMax], pix[kMax], xot[kMax];
      bool use = false;  // false: nextCrossedPosition() as is (n > kMax or no AVX2)
      // n <= 32: the positions as bit masks (sweep() / nextCrossedPositionFromMasks() with a CandidateLine)
      bool use32 = false;
      uint32_t okMask = 0, pixMask = 0, xotMask = 0;
      ALPAKA_FN_HOST_ACC void init(const ChainFinderInputs &finder) {
        n = finder.chain.nPositions, ot_max_pix = finder.chain.maxPixelLayersForExtraOuterTracker;
#ifdef MKSEEDER_CPU_AVX2
        use = n <= kMax;
#endif
        if (!use)
          return;
        const int n8 = (n + 7) / 8 * 8;
        for (int q = 0; q < n8; ++q) {
          const bool in = q < n;
          const auto e = finder.chainPositions[in ? q : 0];
          pos[q] = e.position(), plo[q] = e.positionLow(), phi[q] = e.positionHigh();
          lo_in[q] = e.innerLow(), hi_in[q] = e.innerHigh(), lo_out[q] = e.outerLow(), hi_out[q] = e.outerHigh();
          ok[q] = in && e.envelopeValid() ? -1 : 0, disc[q] = e.isDisc() ? -1 : 0, pix[q] = e.isPixel() ? -1 : 0;
          xot[q] = e.isExtraOuterTracker();
        }
        use32 = n <= 32;
        for (int q = 0; q < n && use32; ++q)
          okMask |= (ok[q] ? 1u : 0u) << q, pixMask |= (pix[q] ? 1u : 0u) << q, xotMask |= (xot[q] ? 1u : 0u) << q;
      }
#ifdef MKSEEDER_CPU_AVX2
      // the states (0, 1, 2) of the 8 positions from q0
      ALPAKA_FN_HOST_ACC inline __m256i states8(
          int q0, __m256 vz0, __m256 vc, __m256 vic, __m256 vzr, __m256 vval) const {
        const __m256 vplo = _mm256_load_ps(plo + q0), vphi = _mm256_load_ps(phi + q0);
        const __m256 db = _mm256_castsi256_ps(_mm256_load_si256((const __m256i *)(disc + q0)));
        // barrel: z0 + c * plo, z0 + c * phi; disc: plo * ic + zr, phi * ic + zr
        // (state() is compiled with these multiply-adds contracted into FMA)
        const __m256 x1 = _mm256_blendv_ps(_mm256_fmadd_ps(vc, vplo, vz0), _mm256_fmadd_ps(vplo, vic, vzr), db);
        const __m256 x2 = _mm256_blendv_ps(_mm256_fmadd_ps(vc, vphi, vz0), _mm256_fmadd_ps(vphi, vic, vzr), db);
        const __m256 xl = _mm256_min_ps(x2, x1), xh = _mm256_max_ps(x2, x1);  // std::min / std::max (x1, x2)
        const __m256 o = _mm256_and_ps(_mm256_cmp_ps(xh, _mm256_load_ps(lo_out + q0), _CMP_GT_OQ),
                                       _mm256_cmp_ps(xl, _mm256_load_ps(hi_out + q0), _CMP_LT_OQ));
        const __m256 i = _mm256_and_ps(_mm256_cmp_ps(xl, _mm256_load_ps(lo_in + q0), _CMP_GT_OQ),
                                       _mm256_cmp_ps(xh, _mm256_load_ps(hi_in + q0), _CMP_LT_OQ));
        // a disc: valid = ic != 0 and pos * ic + zr > 0
        const __m256 dv = _mm256_and_ps(
            vval, _mm256_cmp_ps(_mm256_fmadd_ps(_mm256_load_ps(pos + q0), vic, vzr), _mm256_setzero_ps(), _CMP_GT_OQ));
        const __m256 keep = _mm256_or_ps(_mm256_xor_ps(db, _mm256_castsi256_ps(_mm256_set1_epi32(-1))), dv);
        const __m256i one = _mm256_set1_epi32(1);
        __m256i st = _mm256_add_epi32(_mm256_and_si256(_mm256_castps_si256(o), one),
                                      _mm256_and_si256(_mm256_castps_si256(i), one));
        return _mm256_and_si256(st, _mm256_castps_si256(keep));
      }
#endif
#ifdef MKSEEDER_CPU_AVX2
      // use32: the states of every position for the line (z0, c), as masks: crs state > 0, def state 2 (the
      // states state() gives, ok or not)
      template <typename TAcc>
      ALPAKA_FN_ACC inline void sweep(const TAcc &acc, float z0, float c, uint32_t &crs, uint32_t &def) const {
        const float cs = alpaka::math::abs(acc, c) >= 1e-9f ? c : 1.0f,
                    ic = alpaka::math::abs(acc, c) >= 1e-9f ? 1.0f / cs : 0.0f;
        const float zr = -z0 * ic;
        const __m256 vz0 = _mm256_set1_ps(z0), vc = _mm256_set1_ps(c), vic = _mm256_set1_ps(ic),
                     vzr = _mm256_set1_ps(zr);
        const __m256 vval = _mm256_castsi256_ps(_mm256_set1_epi32(ic != 0 ? -1 : 0));
        const __m256i zero = _mm256_setzero_si256(), two = _mm256_set1_epi32(2);
        uint32_t a = 0, d = 0;
        for (int q0 = 0; q0 < n; q0 += 8) {
          const __m256i s8 = states8(q0, vz0, vc, vic, vzr, vval);
          a |= (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(s8, zero))) << q0;
          d |= (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(s8, two))) << q0;
        }
        const uint32_t all = n >= 32 ? ~0u : (1u << n) - 1u;
        crs = a & all, def = d & all;
      }
#else
      template <typename TAcc>
      ALPAKA_FN_ACC inline void sweep(const TAcc &, float, float, uint32_t &crs, uint32_t &def) const {
        crs = def = 0;
      }
#endif
      // use: the state of position q alone, by the operations of states8 (the same bit as sweep()'s)
      template <typename TAcc>
      ALPAKA_FN_ACC inline int state1(const TAcc &acc, int q, float z0, float c) const {
        const float cs = alpaka::math::abs(acc, c) >= 1e-9f ? c : 1.0f,
                    ic = alpaka::math::abs(acc, c) >= 1e-9f ? 1.0f / cs : 0.0f;
        const float zr = -z0 * ic;
        const bool dq = disc[q] != 0;
        const float x1 = dq ? alpaka::math::fma(acc, plo[q], ic, zr) : alpaka::math::fma(acc, c, plo[q], z0);
        const float x2 = dq ? alpaka::math::fma(acc, phi[q], ic, zr) : alpaka::math::fma(acc, c, phi[q], z0);
        const float xl = x2 < x1 ? x2 : x1, xh = x2 > x1 ? x2 : x1;  // _mm256_min_ps / _mm256_max_ps (x2, x1)
        const int st = ((xh > lo_out[q]) & (xl < hi_out[q])) + ((xl > lo_in[q]) & (xh < hi_in[q]));
        return dq && !((ic != 0) & (alpaka::math::fma(acc, pos[q], ic, zr) > 0)) ? 0 : st;
      }
      // nextCrossedPosition() from the masks of the line: the first crossed position after p (a non-pixel one only with allow),
      // the outer-tracker positions of xot only if the line definitely crosses at most ot_max_pix pixel layers
      ALPAKA_FN_HOST_ACC inline int nextCrossedPositionFromMasks(int p,
                                                                 const CandidateLine &L,
                                                                 int allow,
                                                                 int &st) const {
        uint32_t m = L.crossedPositions & okMask & (allow ? ~0u : pixMask) & ~((2u << p) - 1u);
        if ((m & xotMask) && ot_max_pix > 0 && std::popcount(L.definitePositions & okMask & pixMask) > ot_max_pix)
          m &= ~xotMask;
        if (!m) {
          st = 0;
          return -1;
        }
        const int q = std::countr_zero(m);
        st = (L.definitePositions >> q) & 1u ? 2 : 1;
        return q;
      }
      template <typename TAcc, typename F>
      ALPAKA_FN_ACC inline int nextCrossedPosition(const TAcc &acc,
                                                   const ChainFinderInputs &finder,
                                                   int p,
                                                   float z0,
                                                   float c,
                                                   int allow,
                                                   int &st,
                                                   F &&scalar) const {
#ifdef MKSEEDER_CPU_AVX2
        if (use) {
          // state()'s disc terms, the same for every position
          const float cs = alpaka::math::abs(acc, c) >= 1e-9f ? c : 1.0f,
                      ic = alpaka::math::abs(acc, c) >= 1e-9f ? 1.0f / cs : 0.0f;
          const float zr = -z0 * ic;
          const __m256 vz0 = _mm256_set1_ps(z0), vc = _mm256_set1_ps(c), vic = _mm256_set1_ps(ic),
                       vzr = _mm256_set1_ps(zr);
          const __m256 vval = _mm256_castsi256_ps(_mm256_set1_epi32(ic != 0 ? -1 : 0));
          const __m256i vallow = _mm256_set1_epi32(allow ? -1 : 0), zero = _mm256_setzero_si256();
          alignas(32) int sv[8];
          int xok = -1;
          for (int q0 = (p + 1) & ~7; q0 < n; q0 += 8) {
            const __m256i s8 = states8(q0, vz0, vc, vic, vzr, vval);
            const __m256i ok8 =
                _mm256_and_si256(_mm256_load_si256((const __m256i *)(ok + q0)),
                                 _mm256_or_si256(_mm256_load_si256((const __m256i *)(pix + q0)), vallow));
            const __m256i hit = _mm256_and_si256(ok8, _mm256_cmpgt_epi32(s8, zero));
            unsigned int m = _mm256_movemask_ps(_mm256_castsi256_ps(hit));
            m &= q0 > p ? 0xffu : (0xffu << (p + 1 - q0)) & 0xffu;
            if (!m)
              continue;
            _mm256_store_si256((__m256i *)sv, s8);
            while (m) {
              const int k = std::countr_zero(m);
              m &= m - 1;
              const int q = q0 + k;
              if (xot[q] && ot_max_pix > 0) {
                if (xok < 0) {
                  int np = 0;
                  for (int r0 = 0; r0 < n; r0 += 8) {
                    const __m256i t8 = states8(r0, vz0, vc, vic, vzr, vval);
                    const __m256i two =
                        _mm256_and_si256(_mm256_and_si256(_mm256_load_si256((const __m256i *)(ok + r0)),
                                                          _mm256_load_si256((const __m256i *)(pix + r0))),
                                         _mm256_cmpeq_epi32(t8, _mm256_set1_epi32(2)));
                    np += std::popcount(unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(two))));
                  }
                  xok = np <= ot_max_pix;
                }
                if (!xok)
                  continue;
              }
              st = sv[k];
              return q;
            }
          }
          st = 0;
          return -1;
        }
#endif
        return scalar();
      }
    };  // struct Router
    struct NoRouter {
      static constexpr bool use32 = false;
      ALPAKA_FN_HOST_ACC void init(const ChainFinderInputs &) {}
      template <typename TAcc>
      ALPAKA_FN_ACC void sweep(const TAcc &, float, float, uint32_t &, uint32_t &) const {}
      template <typename TAcc>
      ALPAKA_FN_ACC int state1(const TAcc &acc, int, float, float) const {
        return 0;
      }
      ALPAKA_FN_HOST_ACC int nextCrossedPositionFromMasks(int, const CandidateLine &, int, int &) const { return -1; }
      template <typename TAcc, typename F>
      ALPAKA_FN_ACC int nextCrossedPosition(
          const TAcc &acc, const ChainFinderInputs &, int, float, float, int, int &, F &&scalar) const {
        return scalar();
      }
    };
    template <typename TAcc>
    using RouterOf = std::conditional_t<kCpuAcc<TAcc>, Router, NoRouter>;

    // push() / readCandidate() of the serial backend with the candidate's CandidateLine in Queues::line
    ALPAKA_FN_HOST_ACC inline void pushCandidateWithLine(const Queues &candidateQueues,
                                                         int queue,
                                                         const Candidate &c,
                                                         const CandidateLine &L) {
      const uint32_t s = candidateQueues.queues.count(queue)++;
      const uint32_t ci = s / Queues::kCandidateChunkSize, slot = s % Queues::kCandidateChunkSize;
      if (ci >= candidateQueues.maxChunksPerQueue) {
        ++candidateQueues.counters->nLost;
        return;
      }
      int &t = candidateQueues.chunkTable.chunk(queue * candidateQueues.maxChunksPerQueue + ci);
      if (slot == 0)
        t = takeChunkSerial(candidateQueues);
      const int cid = t;
      if (cid < 0) {
        ++candidateQueues.counters->nLost;
        return;
      }
      const uint32_t at = (uint32_t)cid * Queues::kCandidateChunkSize + slot;
      storeCandidate(candidateQueues.pool, at, c), storeCandidateLine(candidateQueues.lines, at, L);
    }
    ALPAKA_FN_HOST_ACC inline bool readCandidateWithLine(
        const Queues &candidateQueues, int queue, uint32_t i, Candidate &c, CandidateLine &L) {
      const uint32_t ci = i / Queues::kCandidateChunkSize;
      if (ci >= candidateQueues.maxChunksPerQueue)
        return false;
      const int cid = candidateQueues.chunkTable.chunk(queue * candidateQueues.maxChunksPerQueue + ci);
      if (cid < 0)
        return false;
      const uint32_t at = (uint32_t)cid * Queues::kCandidateChunkSize + i % Queues::kCandidateChunkSize;
      loadCandidate(candidateQueues.pool, at, c), loadCandidateLine(candidateQueues.lines, at, L);
      return true;
    }

    // scoreTriplet() with the MLP's hidden units as the SIMD axis: each unit's sum sees the scalar evaluateTripletMlp's sequence
    // ((B + x0 w0) + x1 w1) + ... (mul then add, as the compiled evaluateTripletMlp: no FMA there), with 4 (2) independent accumulators
    // instead of one dependent chain per unit: the same score
    template <typename TAcc>
    ALPAKA_FN_ACC inline float scoreTriplet(
        const TAcc &acc, const ChainFinderInputs &finder, uint32_t ia, uint32_t ib, uint32_t ic) {
#ifdef MKSEEDER_CPU_AVX2
      static_assert(kTripletMlpNHidden1 == 32 && kTripletMlpNHidden2 == 16, "scoreTriplet layout");
      const uint32_t ih[3] = {ia, ib, ic};
      float x[3], y[3], z[3], f[kTripletMlpNFeatures];
      int rows[3], cols[3];
      for (int i = 0; i < 3; ++i)
        x[i] = finder.hits.x(ih[i]), y[i] = finder.hits.y(ih[i]), z[i] = finder.hits.z(ih[i]),
        rows[i] = finder.hits.clusterRows(ih[i]), cols[i] = finder.hits.clusterCols(ih[i]);
      if (!mkseeder::computeTripletFeatures(acc,
                                            x,
                                            y,
                                            z,
                                            rows,
                                            cols,
                                            finder.chainEvent.beamSpotX,
                                            finder.chainEvent.beamSpotY,
                                            finder.chainEvent.beamSpotZ,
                                            f))
        return -1.f;
      const MkSeederTripletMlpWeights &weights = *finder.tripletMlpWeights;
      foldTripletFeatures(f, weights.zFold);
      const float *mean = weights.featureMean, *scale = weights.featureScale, *W1 = weights.hidden1Weights;
      const float *B1 = weights.hidden1Bias, *W2 = weights.hidden2Weights, *B2 = weights.hidden2Bias;
      const float *W3 = weights.outputWeights, *B3 = &weights.outputBias;
      // a product the compiler must not contract into an FMA with the following add (evaluateTripletMlp has none)
      auto mulnc = [](__m256 a, __m256 b) {
        __m256 p = _mm256_mul_ps(a, b);
        asm("" : "+x"(p));
        return p;
      };
      __m256 a0 = _mm256_loadu_ps(B1), a1 = _mm256_loadu_ps(B1 + 8), a2 = _mm256_loadu_ps(B1 + 16),
             a3 = _mm256_loadu_ps(B1 + 24);
      for (int i = 0; i < kTripletMlpNFeatures; ++i) {
        const __m256 xn = _mm256_set1_ps((f[i] - mean[i]) / scale[i]);
        const float *w = W1 + i * kTripletMlpNHidden1;
        a0 = _mm256_add_ps(a0, mulnc(xn, _mm256_loadu_ps(w)));
        a1 = _mm256_add_ps(a1, mulnc(xn, _mm256_loadu_ps(w + 8)));
        a2 = _mm256_add_ps(a2, mulnc(xn, _mm256_loadu_ps(w + 16)));
        a3 = _mm256_add_ps(a3, mulnc(xn, _mm256_loadu_ps(w + 24)));
      }
      // ReLU as s > 0 ? s : 0 (NaN and -0 give +0)
      const __m256 zero = _mm256_setzero_ps();
      alignas(32) float h1[kTripletMlpNHidden1];
      _mm256_store_ps(h1, _mm256_and_ps(a0, _mm256_cmp_ps(a0, zero, _CMP_GT_OQ)));
      _mm256_store_ps(h1 + 8, _mm256_and_ps(a1, _mm256_cmp_ps(a1, zero, _CMP_GT_OQ)));
      _mm256_store_ps(h1 + 16, _mm256_and_ps(a2, _mm256_cmp_ps(a2, zero, _CMP_GT_OQ)));
      _mm256_store_ps(h1 + 24, _mm256_and_ps(a3, _mm256_cmp_ps(a3, zero, _CMP_GT_OQ)));
      __m256 b0 = _mm256_loadu_ps(B2), b1 = _mm256_loadu_ps(B2 + 8);
      for (int j = 0; j < kTripletMlpNHidden1; ++j) {
        const __m256 hj = _mm256_set1_ps(h1[j]);
        const float *w = W2 + j * kTripletMlpNHidden2;
        b0 = _mm256_add_ps(b0, mulnc(hj, _mm256_loadu_ps(w)));
        b1 = _mm256_add_ps(b1, mulnc(hj, _mm256_loadu_ps(w + 8)));
      }
      alignas(32) float h2[kTripletMlpNHidden2];
      _mm256_store_ps(h2, _mm256_and_ps(b0, _mm256_cmp_ps(b0, zero, _CMP_GT_OQ)));
      _mm256_store_ps(h2 + 8, _mm256_and_ps(b1, _mm256_cmp_ps(b1, zero, _CMP_GT_OQ)));
      float o = B3[0];
      for (int k = 0; k < kTripletMlpNHidden2; ++k) {
        float pk = h2[k] * W3[k];
        asm("" : "+x"(pk));
        o += pk;
      }
      return 1.f / (1.f + alpaka::math::exp(acc, -o));
#else
      return mkseeder::scoreTriplet(acc, finder, ia, ib, ic);
#endif
    }

#ifdef MKSEEDER_CPU_AVX2
    // mkseeder::computeTripletFeatures of 4 triplets (lanes) with the hits' azimuths given (ChainFinderInputs::cpuHits.phiBeam): the
    // operations of the compiled scalar function (its FMA contractions; std::round replicated, log by the scalar
    // std::log per lane); ok[t] false for a degenerate triplet (f[t] not set)
    template <typename TAcc>
    ALPAKA_FN_ACC inline void computeTripletFeatures4(const TAcc &acc,
                                                      const ChainFinderInputs &finder,
                                                      const uint32_t (*ih)[3],
                                                      float (*f)[kTripletMlpNFeatures],
                                                      bool *ok) {
      alignas(16) float X[3][4], Y[3][4], Z[3][4], P[3][4];
      int rows[4][3], cols[4][3];
      for (int t = 0; t < 4; ++t)
        for (int i = 0; i < 3; ++i) {
          const uint32_t h = ih[t][i];
          X[i][t] = finder.hits.x(h), Y[i][t] = finder.hits.y(h), Z[i][t] = finder.hits.z(h),
          P[i][t] = finder.cpuHits.phiBeam(h);
          rows[t][i] = finder.hits.clusterRows(h), cols[t][i] = finder.hits.clusterCols(h);
        }
      const __m128 bx = _mm_set1_ps(finder.chainEvent.beamSpotX), by = _mm_set1_ps(finder.chainEvent.beamSpotY),
                   bz = _mm_set1_ps(finder.chainEvent.beamSpotZ);
      __m128 x[3], y[3], z[3], r[3], p[3];
      for (int i = 0; i < 3; ++i) {
        x[i] = _mm_sub_ps(_mm_load_ps(X[i]), bx), y[i] = _mm_sub_ps(_mm_load_ps(Y[i]), by),
        z[i] = _mm_sub_ps(_mm_load_ps(Z[i]), bz);
        r[i] = _mm_sqrt_ps(_mm_fmadd_ps(x[i], x[i], _mm_mul_ps(y[i], y[i])));
        p[i] = _mm_load_ps(P[i]);
      }
      const __m128 ax = x[0], ay = y[0], qx = x[1], qy = y[1], cx = x[2], cy = y[2];
      const __m128 absm = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)), zero = _mm_setzero_ps(),
                   one = _mm_set1_ps(1.0f);
      const __m128 qy_cy = _mm_sub_ps(qy, cy), cy_ay = _mm_sub_ps(cy, ay), ay_qy = _mm_sub_ps(ay, qy);
      __m128 d = _mm_fmadd_ps(cx, ay_qy, _mm_fmadd_ps(ax, qy_cy, _mm_mul_ps(qx, cy_ay)));
      d = _mm_add_ps(d, d);
      const __m128 dok = _mm_cmp_ps(_mm_and_ps(d, absm), _mm_set1_ps(1e-6f), _CMP_GT_OQ);
      const __m128 a2 = _mm_fmadd_ps(ax, ax, _mm_mul_ps(ay, ay)), q2 = _mm_fmadd_ps(qx, qx, _mm_mul_ps(qy, qy));
      const __m128 c2 = _mm_fmadd_ps(cx, cx, _mm_mul_ps(cy, cy));
      const __m128 ox = _mm_div_ps(_mm_fmadd_ps(ay_qy, c2, _mm_fmadd_ps(qy_cy, a2, _mm_mul_ps(q2, cy_ay))), d);
      const __m128 oy = _mm_div_ps(
          _mm_fmadd_ps(
              _mm_sub_ps(qx, ax), c2, _mm_fmadd_ps(_mm_sub_ps(cx, qx), a2, _mm_mul_ps(q2, _mm_sub_ps(ax, cx)))),
          d);
      const __m128 dxo = _mm_sub_ps(ax, ox), dyo = _mm_sub_ps(ay, oy);
      const __m128 rr = _mm_sqrt_ps(_mm_fmadd_ps(dxo, dxo, _mm_mul_ps(dyo, dyo)));
      const __m128 ro = _mm_sqrt_ps(_mm_fmadd_ps(ox, ox, _mm_mul_ps(oy, oy)));
      const __m128 kap = _mm_and_ps(dok, _mm_div_ps(one, rr)),
                   dca = _mm_and_ps(dok, _mm_and_ps(_mm_sub_ps(ro, rr), absm));
      const __m128 dr = _mm_sub_ps(r[2], r[0]), dz = _mm_sub_ps(z[2], z[0]);
      const __m128 bad = _mm_and_ps(_mm_cmp_ps(dr, zero, _CMP_EQ_OQ), _mm_cmp_ps(dz, zero, _CMP_EQ_OQ));
      const __m128 barrelish = _mm_cmp_ps(_mm_and_ps(dr, absm), _mm_set1_ps(1e-3f), _CMP_GT_OQ);
      const __m128 cot = _mm_blendv_ps(_mm_set1_ps(1e3f), _mm_div_ps(dz, dr), barrelish);
      const __m128 z0 = _mm_fnmadd_ps(cot, r[0], z[0]);
      const __m128 rz = _mm_sub_ps(z[1], _mm_fmadd_ps(_mm_sub_ps(r[1], r[0]), cot, z[0]));
      auto round4 = [&](__m128 a) {  // std::round
        const __m128 tr = _mm_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        const __m128 up = _mm_cmp_ps(_mm_and_ps(_mm_sub_ps(a, tr), absm), _mm_set1_ps(0.5f), _CMP_GE_OQ);
        return _mm_add_ps(tr, _mm_and_ps(up, _mm_or_ps(one, _mm_and_ps(a, _mm_set1_ps(-0.0f)))));
      };
      auto wrapr4 = [&](__m128 a) {  // wrapPhiRound (MkSeederPostMath.h)
        return _mm_fnmadd_ps(round4(_mm_mul_ps(a, _mm_set1_ps(mkseeder::kInvTwoPi))), _mm_set1_ps(mkseeder::kTwoPi), a);
      };
      const __m128 dp12 = _mm_and_ps(wrapr4(_mm_sub_ps(p[1], p[0])), absm),
                   dp23 = _mm_and_ps(wrapr4(_mm_sub_ps(p[2], p[1])), absm);
      auto clamp4 = [](__m128 a, float c) { return _mm_max_ps(_mm_min_ps(a, _mm_set1_ps(c)), _mm_set1_ps(-c)); };
      alignas(16) float o[13][4], dcas[4];
      for (int i = 0; i < 3; ++i)
        _mm_store_ps(o[i], r[i]), _mm_store_ps(o[3 + i], z[i]);
      _mm_store_ps(o[6], kap), _mm_store_ps(dcas, dca);
      _mm_store_ps(o[8], clamp4(z0, 30.f)), _mm_store_ps(o[9], clamp4(rz, 5.f)), _mm_store_ps(o[10], clamp4(cot, 20.f));
      _mm_store_ps(o[11], dp12), _mm_store_ps(o[12], dp23);
      const int badm = _mm_movemask_ps(bad);
      for (int t = 0; t < 4; ++t) {
        ok[t] = !((badm >> t) & 1);
        if (!ok[t])
          continue;
        float *g = f[t];
        for (int n = 0; n < 13; ++n)
          g[n] = o[n][t];
        g[7] = alpaka::math::log(acc, dcas[t] + 1e-4f);
        for (int i = 0; i < 3; ++i)
          g[13 + i] = (float)std::min(32, rows[t][i]), g[16 + i] = (float)std::min(32, cols[t][i]);
      }
    }
#endif

    // scoreTriplet() of k <= 4 triplets (ih[t]: a, b, c) in lockstep: the MLP's dependent add chains of several
    // triplets interleaved (layer 1 two triplets at a time, layer 2 four, layer 3 four scalar chains); per triplet the
    // same operations as scoreTriplet, so the same scores
    // okl != nullptr: out the MLP's logits (before the sigmoid), okl[t] 0 for a degenerate triplet (score -1)
    template <typename TAcc>
    ALPAKA_FN_ACC inline void scoreTriplets4(const TAcc &acc,
                                             const ChainFinderInputs &finder,
                                             int k,
                                             const uint32_t (*ih)[3],
                                             float *out,
                                             int *okl = nullptr) {
#ifdef MKSEEDER_CPU_AVX2
      static_assert(kTripletMlpNHidden1 == 32 && kTripletMlpNHidden2 == 16, "scoreTriplets4 layout");
      const MkSeederTripletMlpWeights &weights = *finder.tripletMlpWeights;
      const float *mean = weights.featureMean, *scale = weights.featureScale, *W1 = weights.hidden1Weights;
      const float *B1 = weights.hidden1Bias, *W2 = weights.hidden2Weights, *B2 = weights.hidden2Bias;
      const float *W3 = weights.outputWeights, *B3 = &weights.outputBias;
      float xn[4][kTripletMlpNFeatures];
      bool ok[4];
      // the features of the 4 triplets at once (unused lanes repeat the first triplet)
      float f4[4][kTripletMlpNFeatures];
      bool ok4[4] = {false, false, false, false};
      if (finder.hasPhiBeam) {
        uint32_t ih4[4][3];
        for (int t = 0; t < 4; ++t)
          for (int i = 0; i < 3; ++i)
            ih4[t][i] = ih[t < k ? t : 0][i];
        computeTripletFeatures4(acc, finder, ih4, f4, ok4);
      }
      for (int t = 0; t < 4; ++t) {
        ok[t] = false;
        if (t < k && finder.hasPhiBeam) {
          ok[t] = ok4[t];
          if (ok[t])
            foldTripletFeatures(f4[t], weights.zFold);
          if (ok[t])
            for (int i = 0; i < kTripletMlpNFeatures; ++i)
              xn[t][i] = (f4[t][i] - mean[i]) / scale[i];
        } else if (t < k) {
          float x[3], y[3], z[3], f[kTripletMlpNFeatures];
          int rows[3], cols[3];
          float ph[3];
          for (int i = 0; i < 3; ++i) {
            const uint32_t h = ih[t][i];
            x[i] = finder.hits.x(h), y[i] = finder.hits.y(h), z[i] = finder.hits.z(h),
            rows[i] = finder.hits.clusterRows(h), cols[i] = finder.hits.clusterCols(h);
            ph[i] = finder.hasPhiBeam ? finder.cpuHits.phiBeam(h) : 0.f;
          }
          ok[t] = mkseeder::computeTripletFeatures(acc,
                                                   x,
                                                   y,
                                                   z,
                                                   rows,
                                                   cols,
                                                   finder.chainEvent.beamSpotX,
                                                   finder.chainEvent.beamSpotY,
                                                   finder.chainEvent.beamSpotZ,
                                                   f,
                                                   finder.hasPhiBeam ? ph : nullptr);
          if (ok[t])
            foldTripletFeatures(f, weights.zFold);
          if (ok[t])
            for (int i = 0; i < kTripletMlpNFeatures; ++i)
              xn[t][i] = (f[i] - mean[i]) / scale[i];
        }
        if (!ok[t])
          for (int i = 0; i < kTripletMlpNFeatures; ++i)
            xn[t][i] = 0.f;
      }
      auto mulnc = [](__m256 a, __m256 b) {
        __m256 p = _mm256_mul_ps(a, b);
        asm("" : "+x"(p));
        return p;
      };
      const __m256 zero = _mm256_setzero_ps();
      auto relu = [&](__m256 a) { return _mm256_and_ps(a, _mm256_cmp_ps(a, zero, _CMP_GT_OQ)); };
      alignas(32) float h1[4][kTripletMlpNHidden1];
      for (int t0 = 0; t0 < k; t0 += 2) {
        __m256 a0 = _mm256_loadu_ps(B1), a1 = _mm256_loadu_ps(B1 + 8), a2 = _mm256_loadu_ps(B1 + 16),
               a3 = _mm256_loadu_ps(B1 + 24);
        __m256 c0 = a0, c1 = a1, c2 = a2, c3 = a3;
        for (int i = 0; i < kTripletMlpNFeatures; ++i) {
          const float *w = W1 + i * kTripletMlpNHidden1;
          const __m256 w0 = _mm256_loadu_ps(w), w1 = _mm256_loadu_ps(w + 8), w2 = _mm256_loadu_ps(w + 16),
                       w3 = _mm256_loadu_ps(w + 24);
          const __m256 xa = _mm256_set1_ps(xn[t0][i]), xb = _mm256_set1_ps(xn[t0 + 1][i]);
          a0 = _mm256_add_ps(a0, mulnc(xa, w0)), c0 = _mm256_add_ps(c0, mulnc(xb, w0));
          a1 = _mm256_add_ps(a1, mulnc(xa, w1)), c1 = _mm256_add_ps(c1, mulnc(xb, w1));
          a2 = _mm256_add_ps(a2, mulnc(xa, w2)), c2 = _mm256_add_ps(c2, mulnc(xb, w2));
          a3 = _mm256_add_ps(a3, mulnc(xa, w3)), c3 = _mm256_add_ps(c3, mulnc(xb, w3));
        }
        _mm256_store_ps(h1[t0], relu(a0)), _mm256_store_ps(h1[t0] + 8, relu(a1));
        _mm256_store_ps(h1[t0] + 16, relu(a2)), _mm256_store_ps(h1[t0] + 24, relu(a3));
        _mm256_store_ps(h1[t0 + 1], relu(c0)), _mm256_store_ps(h1[t0 + 1] + 8, relu(c1));
        _mm256_store_ps(h1[t0 + 1] + 16, relu(c2)), _mm256_store_ps(h1[t0 + 1] + 24, relu(c3));
      }
      for (int t = (k + 1) / 2 * 2; t < 4; ++t)
        for (int j = 0; j < kTripletMlpNHidden1; ++j)
          h1[t][j] = 0.f;
      __m256 b[4][2];
      for (int t = 0; t < 4; ++t)
        b[t][0] = _mm256_loadu_ps(B2), b[t][1] = _mm256_loadu_ps(B2 + 8);
      for (int j = 0; j < kTripletMlpNHidden1; ++j) {
        const float *w = W2 + j * kTripletMlpNHidden2;
        const __m256 w0 = _mm256_loadu_ps(w), w1 = _mm256_loadu_ps(w + 8);
        for (int t = 0; t < 4; ++t) {
          const __m256 hj = _mm256_set1_ps(h1[t][j]);
          b[t][0] = _mm256_add_ps(b[t][0], mulnc(hj, w0)), b[t][1] = _mm256_add_ps(b[t][1], mulnc(hj, w1));
        }
      }
      alignas(32) float h2[4][kTripletMlpNHidden2];
      for (int t = 0; t < 4; ++t)
        _mm256_store_ps(h2[t], relu(b[t][0])), _mm256_store_ps(h2[t] + 8, relu(b[t][1]));
      float o[4] = {B3[0], B3[0], B3[0], B3[0]};
      for (int kk = 0; kk < kTripletMlpNHidden2; ++kk)
        for (int t = 0; t < 4; ++t) {
          float pk = h2[t][kk] * W3[kk];
          asm("" : "+x"(pk));
          o[t] += pk;
        }
      if (okl) {
        for (int t = 0; t < k; ++t)
          out[t] = o[t], okl[t] = ok[t];
        return;
      }
      for (int t = 0; t < k; ++t)
        out[t] = ok[t] ? 1.f / (1.f + alpaka::math::exp(acc, -o[t])) : -1.f;
#else
      for (int t = 0; t < k; ++t)
        out[t] = mkseeder::scoreTriplet(acc, finder, ih[t][0], ih[t][1], ih[t][2]);
#endif
    }  // scoreTriplets4()

#ifdef MKSEEDER_CPU_AVX2
    // qBinRangePadded(L, qlo, qhi) and phiBinRangeAround(L, cm, pw) for nb windows (a multiple of 8 is computed), 8 at a time: the
    // operations of qBinClamped / phiBinRangeFromEdges / wrapPhiOnce (compare, select, floor, the conversion through a 32-bit int)
    ALPAKA_FN_HOST_ACC inline void binRanges8(const LayerAxes &L,
                                              uint32_t nb,
                                              const float *qlo_,
                                              const float *qhi_,
                                              const float *cm_,
                                              const float *pw_,
                                              int *qb_,
                                              int *qe_,
                                              int *pb_,
                                              int *pe_) {
      const __m256 qrmin = _mm256_set1_ps(L.qAxisMin()), qfac = _mm256_set1_ps(L.qAxisScale());
      const __m256 qlbhp = _mm256_set1_ps(L.qAxisLastBinHighEdge());
      const __m256 prmin = _mm256_set1_ps(L.phiAxisMin()), pfac = _mm256_set1_ps(L.phiAxisScale());
      const __m256i qlast = _mm256_set1_epi32(L.qLastBin()), m16 = _mm256_set1_epi32(0xffff),
                    mphi = _mm256_set1_epi32(::mkseeder::kNPhiBins - 1);
      const __m256i one = _mm256_set1_epi32(1);
      const __m256 eps = _mm256_set1_ps(1e-4f), wmax = _mm256_set1_ps(0.9f * kPi);
      const __m256 cpi = _mm256_set1_ps(kPi), c2pi = _mm256_set1_ps(2 * kPi), cmpi = _mm256_set1_ps(-kPi);
      auto qsafe = [&](__m256 r) {
        const __m256i f =
            _mm256_and_si256(_mm256_cvttps_epi32(_mm256_floor_ps(_mm256_mul_ps(_mm256_sub_ps(r, qrmin), qfac))), m16);
        const __m256 le = _mm256_cmp_ps(r, qrmin, _CMP_LE_OQ), ge = _mm256_cmp_ps(r, qlbhp, _CMP_GE_OQ);
        __m256i finder = _mm256_blendv_epi8(f, qlast, _mm256_castps_si256(ge));
        return _mm256_andnot_si256(_mm256_castps_si256(le), finder);  // r <= rmin ? 0 : r >= lbhp ? last : f
      };
      auto wrapl8 = [&](__m256 x) {
        const __m256 gt = _mm256_cmp_ps(x, cpi, _CMP_GT_OQ), lt = _mm256_cmp_ps(x, cmpi, _CMP_LT_OQ);
        return _mm256_blendv_ps(_mm256_blendv_ps(x, _mm256_add_ps(x, c2pi), lt), _mm256_sub_ps(x, c2pi), gt);
      };
      auto pbin = [&](__m256 x) {
        return _mm256_and_si256(_mm256_cvttps_epi32(_mm256_floor_ps(_mm256_mul_ps(_mm256_sub_ps(x, prmin), pfac))),
                                m16);
      };
      for (uint32_t j = 0; j < nb; j += 8) {
        const __m256 qlo = _mm256_sub_ps(_mm256_load_ps(qlo_ + j), eps),
                     qhi = _mm256_add_ps(_mm256_load_ps(qhi_ + j), eps);
        _mm256_store_si256((__m256i *)(qb_ + j), qsafe(qlo));
        _mm256_store_si256((__m256i *)(qe_ + j), _mm256_add_epi32(qsafe(qhi), one));
        const __m256 c = _mm256_load_ps(cm_ + j),
                     w = _mm256_min_ps(wmax, _mm256_load_ps(pw_ + j));  // std::min(w, 0.9 pi)
        const __m256 lo = wrapl8(_mm256_sub_ps(c, w)), hi = wrapl8(_mm256_add_ps(c, w));
        _mm256_store_si256((__m256i *)(pb_ + j), _mm256_and_si256(pbin(lo), mphi));
        _mm256_store_si256((__m256i *)(pe_ + j), _mm256_and_si256(_mm256_add_epi32(pbin(hi), one), mphi));
      }
    }
#endif

    // the serial backend's stage d: the per-candidate prologue of KernelExtendToQuads (the candidate, its helix, the window's
    // crossings, azimuths and sine, the pre-filter's mid point) for a block of candidates before their hit loops, so
    // that the independent prologues overlap; the same expressions as KernelExtendToQuads, which reads them from here
    constexpr uint32_t kDProBlock = 16;
    struct DPro {
      Candidate c;
      CandidateLine L;
      HelixF H;
      int got, ix;
      bool inner_ok, win, ok0, ok1, pre, mrg;
      float aphi, bphi, aq, bq, x0, y0, q0, s0, x1, y1, q1, s1, p0, p1, sw, qm;
      BinRange qd, pb;  // the window's bins (with MKSEEDER_CPU_AVX2)
    };

    template <typename TAcc>
    ALPAKA_FN_ACC inline void doubletPrologue(const TAcc &acc,
                                              const ChainFinderInputs &finder,
                                              const Queues &candidateQueues,
                                              int p,
                                              bool disc,
                                              float u0,
                                              float u1,
                                              uint32_t j0,
                                              uint32_t j1,
                                              DPro *P) {
      const MkSeederChainParameters &chainParameters = finder.chain;
      const int n = chainParameters.nPositions;
      // the candidates, their helices
      alignas(32) float in[8][kDProBlock];
      alignas(32) int mk[kDProBlock];
      for (uint32_t j = j0; j < j1; ++j) {
        DPro &d = P[j - j0];
        const uint32_t jj = j - j0;
        mk[jj] = 0;
        d.win = false;
        d.got = readCandidateWithLine(candidateQueues, q3(finder, p), j, d.c, d.L);
        if (!d.got)
          continue;
        const Candidate &c = d.c;
        const EventLayer La = finder.layers[finder.chainPositions[c.positions[0]].binnedLayerId()],
                         Lb = finder.layers[finder.chainPositions[c.positions[1]].binnedLayerId()],
                         Lc = finder.layers[finder.chainPositions[c.positions[2]].binnedLayerId()];
        const uint32_t ia = La.firstHit() + c.hits[0], ib = Lb.firstHit() + c.hits[1], ic = Lc.firstHit() + c.hits[2];
        const int ix =
            finder.windowTableByQuad[((c.positions[0] * n + c.positions[1]) * n + c.positions[2]) * n + p].windowTable();
        d.ix = ix;
        // a merged-cluster candidate among the triplet's hits: stage d's windows widened (mrg_win)
        d.mrg = chainParameters.mergedWindowScale > 0.f &&
                (::mkseeder::isStrongMergedCandidate(finder.hits.mergedTag(ia)) ||
                 ::mkseeder::isStrongMergedCandidate(finder.hits.mergedTag(ib)) ||
                 ::mkseeder::isStrongMergedCandidate(finder.hits.mergedTag(ic)));
        d.inner_ok = !chainParameters.innerOuterTrackerOnly || !c.innerHoles || !finder.chainPositions[p].isPixel();
        mk[jj] = searchesCombination(chainParameters, ix) && d.inner_ok ? -1 : 0;
        in[0][jj] = finder.hits.xBeam(ia), in[1][jj] = finder.hits.yBeam(ia), in[2][jj] = finder.hits.z(ia),
        in[3][jj] = finder.hits.xBeam(ib), in[4][jj] = finder.hits.yBeam(ib);
        in[5][jj] = finder.hits.xBeam(ic), in[6][jj] = finder.hits.yBeam(ic), in[7][jj] = finder.hits.z(ic);
      }
      const float um = 0.5f * (u0 + u1);
#ifdef MKSEEDER_CPU_AVX2
      {
        // the windows' q range and phi centre / half width per candidate (lanes without a window: 0), their bins
        alignas(32) float W_qlo[kDProBlock] = {}, W_qhi[kDProBlock] = {}, W_cm[kDProBlock] = {}, W_pw[kDProBlock] = {};
        // make for 8 candidates at a time, on a barrel layer also the three positionAtR (a disc: positionAtZ one by one)
        for (uint32_t jj = j1 - j0; jj < kDProBlock; ++jj) {
          mk[jj] = 0;
          for (int a = 0; a < 8; ++a)
            in[a][jj] = 0.f;
        }
        for (uint32_t b0 = 0; b0 < j1 - j0; b0 += 8) {
          const __m256 m = _mm256_castsi256_ps(_mm256_load_si256((const __m256i *)(mk + b0)));
          if (!_mm256_movemask_ps(m)) {
            for (uint32_t jj = b0; jj < b0 + 8 && jj < j1 - j0; ++jj)
              P[jj].H = HelixF{};
            continue;
          }
          avx2::Helix8 H;
          avx2::make8(H,
                      _mm256_load_ps(in[0] + b0),
                      _mm256_load_ps(in[1] + b0),
                      _mm256_load_ps(in[2] + b0),
                      _mm256_load_ps(in[3] + b0),
                      _mm256_load_ps(in[4] + b0),
                      _mm256_load_ps(in[5] + b0),
                      _mm256_load_ps(in[6] + b0),
                      _mm256_load_ps(in[7] + b0),
                      m);
          __m256 ok[3], wr[3], px[3], py[3], q[3], s3[3];
          const float rr[3] = {u0, u1, um};
          for (int t = 0; t < 3; ++t) {
            if (disc)
              avx2::positionAtZ8(H, _mm256_set1_ps(rr[t]), H.ok, ok[t], px[t], py[t], q[t], s3[t]), wr[t] = ok[t];
            else
              avx2::positionAtR8(H, _mm256_set1_ps(rr[t]), H.ok, ok[t], wr[t], px[t], py[t], q[t], s3[t]);
          }
          alignas(32) float hf[16][8], o[3][6][8];
          const __m256 *hv[16] = {&H.k,
                                  &H.tx,
                                  &H.ty,
                                  &H.cx,
                                  &H.cy,
                                  &H.cz,
                                  &H.cot,
                                  &H.ux,
                                  &H.uy,
                                  &H.im,
                                  &H.c2,
                                  &H.cn,
                                  &H.hk,
                                  &H.ahk,
                                  &H.sq,
                                  &H.ok};
          for (int f = 0; f < 16; ++f)
            _mm256_store_ps(hf[f], *hv[f]);
          // the window's azimuths: the crossings with the fix-ups below (a missing one takes the other's), atan2
          alignas(32) float pz[2][8];
          {
            const __m256 zero = _mm256_setzero_ps();
            const __m256 x0 = avx2::sel(wr[0], px[0], zero), y0 = avx2::sel(wr[0], py[0], zero);
            const __m256 x1 = avx2::sel(wr[1], px[1], zero), y1 = avx2::sel(wr[1], py[1], zero);
            const __m256 x0f = avx2::sel(ok[0], x0, x1), y0f = avx2::sel(ok[0], y0, y1);
            const __m256 x1f = avx2::sel(ok[1], x1, x0f), y1f = avx2::sel(ok[1], y1, y0f);
            const __m256 win = _mm256_or_ps(ok[0], ok[1]);
            _mm256_store_ps(pz[0], avx2::atan2f8(y0f, x0f, win));
            _mm256_store_ps(pz[1], avx2::atan2f8(y1f, x1f, win));
          }
          for (int t = 0; t < 3; ++t) {
            _mm256_store_ps(o[t][0], ok[t]), _mm256_store_ps(o[t][1], wr[t]), _mm256_store_ps(o[t][2], px[t]);
            _mm256_store_ps(o[t][3], py[t]), _mm256_store_ps(o[t][4], q[t]), _mm256_store_ps(o[t][5], s3[t]);
          }
          for (uint32_t l = 0; l < 8 && b0 + l < j1 - j0; ++l) {
            DPro &d = P[b0 + l];
            HelixF &Hs = d.H;
            Hs = HelixF{};
            if (!mk[b0 + l])
              continue;
            auto lb = [&](const float *a) { return std::bit_cast<uint32_t>(a[l]) != 0; };
            Hs.cx = hf[3][l], Hs.cy = hf[4][l], Hs.cz = hf[5][l];
            Hs.ok = lb(hf[15]);
            if (!Hs.ok)
              continue;
            Hs.k = hf[0][l], Hs.tx = hf[1][l], Hs.ty = hf[2][l], Hs.cot = hf[6][l], Hs.ux = hf[7][l], Hs.uy = hf[8][l];
            Hs.im = hf[9][l], Hs.c2 = hf[10][l], Hs.cn = hf[11][l], Hs.hk = hf[12][l], Hs.ahk = hf[13][l],
            Hs.sq = hf[14][l];
            const auto w = finder.windowTables[d.ix >= 0 ? d.ix : 0];
            float aphi = w.phiWindowA(), bphi = w.phiWindowB(), aq = w.qWindowA(), bq = w.qWindowB();
            if (w.nEtaWindows() > 0) {
              const float ae = alpaka::math::asinh(acc, alpaka::math::abs(acc, Hs.cot));
              for (int i = 0; i < w.nEtaWindows(); ++i) {
                const auto E = finder.etaWindows[w.firstEtaWindow() + i];
                if (ae >= E.absEtaLow() && ae < E.absEtaHigh()) {
                  aphi = E.phiWindowA(), bphi = E.phiWindowB(), aq = E.qWindowA(), bq = E.qWindowB();
                  break;
                }
              }
            }
            if (d.mrg) {
              const float s = chainParameters.mergedWindowScale;
              aphi *= s, bphi *= s, aq *= s, bq *= s;
            }
            d.aphi = aphi, d.bphi = bphi, d.aq = aq, d.bq = bq;
            float x0 = 0, y0 = 0, q0 = 0, s0 = -1, x1 = 0, y1 = 0, q1 = 0, s1 = -1;
            const bool ok0 = lb(o[0][0]), ok1 = lb(o[1][0]);
            if (lb(o[0][1]))
              x0 = o[0][2][l], y0 = o[0][3][l], q0 = o[0][4][l], s0 = o[0][5][l];
            if (lb(o[1][1]))
              x1 = o[1][2][l], y1 = o[1][3][l], q1 = o[1][4][l], s1 = o[1][5][l];
            d.ok0 = ok0, d.ok1 = ok1;
            if (!(ok0 || ok1))
              continue;
            if (!ok0)
              x0 = x1, y0 = y1, q0 = q1;
            if (!ok1)
              x1 = x0, y1 = y0, q1 = q0;
            d.win = true;
            d.x0 = x0, d.y0 = y0, d.q0 = q0, d.s0 = s0, d.x1 = x1, d.y1 = y1, d.q1 = q1, d.s1 = s1;
            d.p0 = pz[0][l], d.p1 = pz[1][l];
            const float pte = Hs.pt(acc), ipt = 1.0f / std::max(chainParameters.ptMin, pte);
            const float isr = w.referencePathLength() > 0 ? 1.0f / w.referencePathLength() : 0.0f;
            const float smax = std::max(s0, s1), lmax = isr > 0 && smax > 0 ? smax * isr : 1.0f;
            const float wpd = aphi + lmax * bphi * ipt, wqd = aq + lmax * bq * ipt;
            d.sw = alpaka::math::sin(acc, wpd);
            // KernelExtendToQuads's window, its bins below
            {
              const float dpp = wrapPhiFloor(acc, d.p1 - d.p0), dmid = d.p0 + 0.5f * dpp,
                          dhalf = 0.5f * alpaka::math::abs(acc, dpp);
              W_qlo[b0 + l] = std::min(q0, q1) - wqd, W_qhi[b0 + l] = std::max(q0, q1) + wqd;
              W_cm[b0 + l] = dmid, W_pw[b0 + l] = dhalf + wpd + 1e-6f;
            }
            d.pre = ok0 && ok1 && lb(o[2][0]);
            d.qm = lb(o[2][1]) ? o[2][4][l] : 0.f;
          }
        }
        {
          const LayerAxes targetLayer = finder.layerGeometry[finder.chainPositions[p].binnedLayerId()];
          alignas(32) int qb[kDProBlock], qe[kDProBlock], pb[kDProBlock], pe[kDProBlock];
          binRanges8(targetLayer, (j1 - j0 + 7) / 8 * 8, W_qlo, W_qhi, W_cm, W_pw, qb, qe, pb, pe);
          for (uint32_t jj = 0; jj < j1 - j0; ++jj)
            P[jj].qd = {(uint32_t)qb[jj], (uint32_t)qe[jj]}, P[jj].pb = {(uint32_t)pb[jj], (uint32_t)pe[jj]};
        }
        return;
      }  // #ifdef MKSEEDER_CPU_AVX2 block
#endif
      for (uint32_t j = j0; j < j1; ++j) {
        DPro &d = P[j - j0];
        if (!d.got)
          continue;
        HelixF &H = d.H;
        H = HelixF{};
        const uint32_t jj = j - j0;
        if (mk[jj])
          H.make(acc, in[0][jj], in[1][jj], in[2][jj], in[3][jj], in[4][jj], in[5][jj], in[6][jj], in[7][jj]);
        if (!(mk[jj] && H.ok))
          continue;
        const int ix = d.ix;
        const auto w = finder.windowTables[ix >= 0 ? ix : 0];
        float aphi = w.phiWindowA(), bphi = w.phiWindowB(), aq = w.qWindowA(), bq = w.qWindowB();
        if (w.nEtaWindows() > 0) {
          const float ae = alpaka::math::asinh(acc, alpaka::math::abs(acc, H.cot));
          for (int i = 0; i < w.nEtaWindows(); ++i) {
            const auto E = finder.etaWindows[w.firstEtaWindow() + i];
            if (ae >= E.absEtaLow() && ae < E.absEtaHigh()) {
              aphi = E.phiWindowA(), bphi = E.phiWindowB(), aq = E.qWindowA(), bq = E.qWindowB();
              break;
            }
          }
        }
        if (d.mrg) {
          const float s = chainParameters.mergedWindowScale;
          aphi *= s, bphi *= s, aq *= s, bq *= s;
        }
        d.aphi = aphi, d.bphi = bphi, d.aq = aq, d.bq = bq;
        float x0, y0, q0, s0 = -1, x1, y1, q1, s1 = -1;
        const bool ok0 = H.at(acc, disc, u0, x0, y0, q0, s0), ok1 = H.at(acc, disc, u1, x1, y1, q1, s1);
        d.ok0 = ok0, d.ok1 = ok1;
        if (!(ok0 || ok1))
          continue;
        if (!ok0)
          x0 = x1, y0 = y1, q0 = q1;
        if (!ok1)
          x1 = x0, y1 = y0, q1 = q0;
        d.win = true;
        d.x0 = x0, d.y0 = y0, d.q0 = q0, d.s0 = s0, d.x1 = x1, d.y1 = y1, d.q1 = q1, d.s1 = s1;
        d.p0 = libm::atan2(acc, y0, x0), d.p1 = libm::atan2(acc, y1, x1);
        const float pte = H.pt(acc), ipt = 1.0f / std::max(chainParameters.ptMin, pte);
        const float isr = w.referencePathLength() > 0 ? 1.0f / w.referencePathLength() : 0.0f;
        const float smax = std::max(s0, s1), lmax = isr > 0 && smax > 0 ? smax * isr : 1.0f;
        const float wpd = aphi + lmax * bphi * ipt;
        d.sw = alpaka::math::sin(acc, wpd);
        float xm, ym, qm = 0, sm = -1;
        d.pre = ok0 && ok1 && H.at(acc, disc, um, xm, ym, qm, sm);
        d.qm = qm;
      }
    }  // doubletPrologue()

    // the serial backend's stage d emits: the triplets that need the MLP wait, with every emit after them (in order),
    // until 4 are there; then their scores (scoreTriplets4) and the emits in their order, as KernelExtendToQuads's emitTripletSeed / take
    struct PendingEmits {
      static constexpr int kMax = 64, kBatch = 4;
      struct P {
        Quad o;
        uint32_t ih[3];
        int scored;
      };
      P e[kMax];
      int n = 0, ns = 0;
#ifdef MKSEEDER_CPU_AVX2
      // the score 1 / (1 + exp(-x)) is non-decreasing in the logit x over every float (checked on all of them with
      // this libm's expf), so score < t <=> x < lt(t) with lt(t) the smallest float logit whose score is >= t: the
      // gates compare the logits, the sigmoid is not evaluated
      static float sig(float x) { return 1.f / (1.f + std::exp(-x)); }
      static float logitThreshold(float t) {
        auto key = [](float x) {
          const auto u = std::bit_cast<int32_t>(x);
          return u < 0 ? -(int64_t)(u & 0x7fffffff) : (int64_t)u;
        };
        auto val = [](int64_t kk) {
          const uint32_t u = kk < 0 ? (uint32_t)(-kk) | 0x80000000u : (uint32_t)kk;
          return std::bit_cast<float>(u);
        };
        int64_t lo = key(-INFINITY), hi = key(INFINITY);  // the smallest key with sig >= t in [lo, hi], or none
        if (!(sig(val(hi)) >= t))
          return INFINITY;
        while (lo < hi) {
          const int64_t mid = lo + (hi - lo) / 2;
          if (sig(val(mid)) >= t)
            hi = mid;
          else
            lo = mid + 1;
        }
        return val(lo);
      }
      bool thr = false;
      float lt_dnn = 0, lt_ext = 0;
#endif
      template <typename TAcc>
      ALPAKA_FN_ACC void flush(const TAcc &acc, const ChainFinderInputs &finder, const Queues &candidateQueues) {
        uint32_t ih[kBatch][3];
        float ts[kBatch];
        int k = 0;
        for (int i = 0; i < n; ++i)
          if (e[i].scored)
            ih[k][0] = e[i].ih[0], ih[k][1] = e[i].ih[1], ih[k][2] = e[i].ih[2], ++k;
#ifdef MKSEEDER_CPU_AVX2
        int okl[kBatch];
        if (k) {
          if (!thr)
            lt_dnn = logitThreshold(finder.chain.tripletMlpCut),
            lt_ext = logitThreshold(finder.chain.tripletMlpRescueMinScore), thr = true;
          scoreTriplets4(acc, finder, k, ih, ts, okl);
        }
#else
        if (k)
          scoreTriplets4(acc, finder, k, ih, ts);
#endif
        int t = 0;
        for (int i = 0; i < n; ++i) {
          Quad &o = e[i].o;
          if (e[i].scored) {
#ifdef MKSEEDER_CPU_AVX2
            // a degenerate triplet: score -1
            const int tt = t++;
            const bool lo1 = okl[tt] ? ts[tt] < lt_dnn : -1.f < finder.chain.tripletMlpCut;
            const bool ge2 = okl[tt] ? ts[tt] >= lt_ext : -1.f >= finder.chain.tripletMlpRescueMinScore;
            if (lo1) {
              if (finder.chain.tripletMlpRescueOuterTrackerHits > 0 && ge2)
#else
            const float s = ts[t++];
            if (s < finder.chain.tripletMlpCut) {
              if (finder.chain.tripletMlpRescueOuterTrackerHits > 0 && s >= finder.chain.tripletMlpRescueMinScore)
#endif
                o.tripletFailedMlp = 1;  // trip_dnn_ext: kept for the extension's probe
              else
                continue;
            }
          }
          emit(acc, candidateQueues, o);
        }
        n = ns = 0;
      }
      // a triplet seed to score (o.fake = o.score = the candidate's score)
      template <typename TAcc>
      ALPAKA_FN_ACC void trip(const TAcc &acc,
                              const ChainFinderInputs &finder,
                              const Queues &candidateQueues,
                              const Quad &o,
                              uint32_t ia,
                              uint32_t ib,
                              uint32_t ic) {
        e[n].o = o, e[n].ih[0] = ia, e[n].ih[1] = ib, e[n].ih[2] = ic, e[n].scored = 1;
        ++n, ++ns;
        if (ns == kBatch || n == kMax)
          flush(acc, finder, candidateQueues);
      }
      template <typename TAcc>
      ALPAKA_FN_ACC void quad(const TAcc &acc,
                              const ChainFinderInputs &finder,
                              const Queues &candidateQueues,
                              const Quad &o) {
        if (n == 0) {
          emit(acc, candidateQueues, o);
          return;
        }
        e[n].o = o, e[n].scored = 0;
        if (++n == kMax)
          flush(acc, finder, candidateQueues);
      }
    };

#ifdef MKSEEDER_CPU_AVX2
    // 8 floats from p, unmasked also for the last k < 8 of a run: the hits' columns have kHitsPadding rows after the
    // last hit, so the load stays inside the buffer; the lanes beyond k are masked out of the result
    ALPAKA_FN_HOST_ACC inline __m256 load8(const float *p, uint32_t, __m256i) { return _mm256_loadu_ps(p); }
    ALPAKA_FN_HOST_ACC inline __m256i tailMask(uint32_t k) {
      return _mm256_cmpgt_epi32(_mm256_set1_epi32((int)std::min(8u, k)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    }
    // f(i) for each set bit of m (bit j: hit i0 + j)
    template <typename F>
    ALPAKA_FN_HOST_ACC inline void forEachSetBit(unsigned int m, uint32_t i0, F &&f) {
      while (m) {
        f(i0 + std::countr_zero(m));
        m &= m - 1;
      }
    }
#endif

    // Loose prefilters (AVX2, 8 hits at a time) before the exact scalar tests, with their constants built once per
    // candidate / item (the scans are called once per run). f(i) gets every hit that may pass; the exact test is f's.
    //
    // stage c: t > 1 and |q - (qa + dqab t)| <= qcw, t = (u - ua) idu; u, q: the layer's arrays at its own indices
    struct ScanTQ {
#ifdef MKSEEDER_CPU_AVX2
      __m256 vua, vidu, vqa, vdq, vone, vqc, vabs;
#endif
      ALPAKA_FN_HOST_ACC ScanTQ(float ua, float idu, float qa, float dqab, float qcw) {
#ifdef MKSEEDER_CPU_AVX2
        vua = _mm256_set1_ps(ua), vidu = _mm256_set1_ps(idu), vqa = _mm256_set1_ps(qa);
        vdq = _mm256_set1_ps(dqab), vone = _mm256_set1_ps(1 - 1e-6f);
        vqc = _mm256_set1_ps(qcw * (1 + 1e-4f) + 1e-3f);
        vabs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
#endif
      }
      template <typename F>
      ALPAKA_FN_HOST_ACC void operator()(const float *u, const float *q, uint32_t b, uint32_t e, F &&f) const {
#ifdef MKSEEDER_CPU_AVX2
        for (uint32_t i0 = b; i0 < e; i0 += 8) {
          const uint32_t k = e - i0;
          const __m256i msk = tailMask(k);
          const __m256 vu = load8(u + i0, k, msk), vq = load8(q + i0, k, msk);
          const __m256 t = _mm256_mul_ps(_mm256_sub_ps(vu, vua), vidu);
          const __m256 dq = _mm256_and_ps(_mm256_sub_ps(vq, _mm256_add_ps(vqa, _mm256_mul_ps(vdq, t))), vabs);
          const __m256 ok = _mm256_and_ps(_mm256_cmp_ps(t, vone, _CMP_GT_OQ), _mm256_cmp_ps(dq, vqc, _CMP_LE_OQ));
          // NaN lanes fail the compares and the exact test alike
          forEachSetBit(_mm256_movemask_ps(_mm256_and_ps(ok, _mm256_castsi256_ps(msk))), i0, f);
        }
#else
        for (uint32_t i = b; i < e; ++i)
          f(i);
#endif
      }
    };

    // stage b: the doublet cuts of KernelFindStartDoublets's testHitB with hit a (r, phi, z, 1 / r), exactly (its operations and FMA
    // contractions): dr > 0.1, |dphi| <= the window, the z0 range, the side and the start's cot range [clo, chi];
    // f(kb, pre) with kb layer B's own hit index and pre = dr, dphi, cot, z0 of the doublet (testHitB skips its cuts)
    struct ScanStart {
      const float *br, *bz, *bp, *bi;
#ifdef MKSEEDER_CPU_AVX2
      __m256 vra, vza, vpa, vinva, vinv2R, vd0, vmarg, vpi, vmpi, v2pi, vdrmin, vzlo, vzhi;
      __m256 vclo, vchi, vside, vabs;
      bool sideNeg;
      // the gap-ray check of a start with one skipped barrel layer (g_lid >= 0): the same operations as KernelFindStartDoublets's
      // (FMA where its scalar code is contracted), so the same decisions; testHitB then skips its own
      bool gray = false;
      __m256 vrmra, vpfac, vizc, vzg0, vpi2;
      __m256i vnzg;
      const float *gr1 = nullptr, *gr2 = nullptr;
      const unsigned char *gzg = nullptr, *gedge = nullptr;
#endif
      ScanStart() = default;
      ALPAKA_FN_HOST_ACC ScanStart(const ChainFinderInputs &finder,
                                   const EventLayer &B,
                                   float raf,
                                   float paf,
                                   float zaf,
                                   float inva,
                                   const MkSeederChainParameters &chainParameters,
                                   float clo,
                                   float chi,
                                   [[maybe_unused]] bool zOuter,
                                   int g_lid = -1)
          : br(finder.hits.r().data() + B.firstHit()),
            bz(finder.hits.z().data() + B.firstHit()),
            bp(finder.hits.phi().data() + B.firstHit()),
            bi(finder.hits.invR().data() + B.firstHit()) {
#ifdef MKSEEDER_CPU_AVX2
        if (g_lid >= 0) {
          constexpr int kGapRayNPhiBins = ::mkseeder::kGapRayNPhiBins;
          const auto G = finder.gapMapLayers[g_lid];
          gray = true;
          vrmra = _mm256_set1_ps(G.rMid() - raf), vpfac = _mm256_set1_ps(kGapRayNPhiBins / kTwoPi);
          vizc = _mm256_set1_ps(1.0f / 0.005f), vzg0 = _mm256_set1_ps(G.zGap0()), vpi2 = _mm256_set1_ps(kPi);
          vnzg = _mm256_set1_epi32(G.nZGaps());
          gr1 = finder.gapRays.rInner().data() + g_lid * kGapRayNPhiBins,
          gr2 = finder.gapRays.rOuter().data() + g_lid * kGapRayNPhiBins;
          gzg = finder.gapMapZGaps.inGap().data() + G.firstZGap(),
          gedge = finder.gapRays.nearEdge().data() + g_lid * kGapRayNPhiBins;
        }
        vra = _mm256_set1_ps(raf), vza = _mm256_set1_ps(zaf), vpa = _mm256_set1_ps(paf);
        vinva = _mm256_set1_ps(inva), vinv2R = _mm256_set1_ps(chainParameters.doubletInverseTwoRadius),
        vd0 = _mm256_set1_ps(chainParameters.doubletD0Max);
        vmarg = _mm256_set1_ps(chainParameters.doubletPhiMargin), vpi = _mm256_set1_ps(kPi),
        vmpi = _mm256_set1_ps(-kPi);
        v2pi = _mm256_set1_ps(kTwoPi), vdrmin = _mm256_set1_ps(0.1f);
        // zv_outer (KernelFindStartDoublets's gate)
        vzlo = _mm256_set1_ps(zOuter ? finder.chainEvent.zVertexLowOuter : finder.chainEvent.zVertexLow),
        vzhi = _mm256_set1_ps(zOuter ? finder.chainEvent.zVertexHighOuter : finder.chainEvent.zVertexHigh);
        vclo = _mm256_set1_ps(clo), vchi = _mm256_set1_ps(chi), vside = _mm256_set1_ps(chainParameters.side);
        vabs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        sideNeg = chainParameters.side < 0;
#endif
      }
      template <typename F>
      ALPAKA_FN_HOST_ACC void operator()(uint32_t b, uint32_t e, F &&f) const {
#ifdef MKSEEDER_CPU_AVX2
        const __m256 vzero = _mm256_setzero_ps();
        for (uint32_t i0 = b; i0 < e; i0 += 8) {
          const uint32_t k = e - i0;
          const __m256i msk = tailMask(k);
          const __m256 r = load8(br + i0, k, msk), z = load8(bz + i0, k, msk);
          const __m256 ph = load8(bp + i0, k, msk), ir = load8(bi + i0, k, msk);
          const __m256 dr = _mm256_sub_ps(r, vra), dz = _mm256_sub_ps(z, vza);
          __m256 dp = _mm256_sub_ps(ph, vpa);
          dp = _mm256_blendv_ps(dp, _mm256_sub_ps(dp, v2pi), _mm256_cmp_ps(dp, vpi, _CMP_GT_OQ));
          dp = _mm256_blendv_ps(dp, _mm256_add_ps(dp, v2pi), _mm256_cmp_ps(dp, vmpi, _CMP_LT_OQ));
          // the exact doublet cuts of KernelFindStartDoublets's testHitB (its FMA contractions): wd = (inva - invr) d0 + dr inv2R, then
          // + marg; num = zaf dr - raf dz
          const __m256 wd =
              _mm256_add_ps(_mm256_fmadd_ps(_mm256_sub_ps(vinva, ir), vd0, _mm256_mul_ps(dr, vinv2R)), vmarg);
          const __m256 num = _mm256_fnmadd_ps(vra, dz, _mm256_mul_ps(vza, dr));
          __m256 ok = _mm256_cmp_ps(dr, vdrmin, _CMP_GT_OQ);
          ok = _mm256_and_ps(ok, _mm256_cmp_ps(_mm256_and_ps(dp, vabs), wd, _CMP_LE_OQ));
          ok = _mm256_and_ps(ok, _mm256_cmp_ps(num, _mm256_mul_ps(vzlo, dr), _CMP_GE_OQ));
          ok = _mm256_and_ps(ok, _mm256_cmp_ps(num, _mm256_mul_ps(vzhi, dr), _CMP_LE_OQ));
          // the side: not (dz * side < 0 or (dz == 0 and side < 0))
          __m256 bad = _mm256_cmp_ps(_mm256_mul_ps(dz, vside), vzero, _CMP_LT_OQ);
          if (sideNeg)
            bad = _mm256_or_ps(bad, _mm256_cmp_ps(dz, vzero, _CMP_EQ_OQ));
          ok = _mm256_andnot_ps(bad, ok);
          const __m256 cot = _mm256_div_ps(dz, dr);
          ok = _mm256_and_ps(ok,
                             _mm256_and_ps(_mm256_cmp_ps(cot, vclo, _CMP_GE_OQ), _mm256_cmp_ps(cot, vchi, _CMP_LE_OQ)));
          ok = _mm256_and_ps(ok, _mm256_castsi256_ps(msk));
          if (gray && _mm256_movemask_ps(ok)) {
            // ph = paf + dp (r_mid - raf) / dr, wrapped once each way; pbn = (int)((ph + pi) * pfac), clamped
            __m256 ph = _mm256_add_ps(vpa, _mm256_div_ps(_mm256_mul_ps(dp, vrmra), dr));
            ph = _mm256_blendv_ps(ph, _mm256_sub_ps(ph, v2pi), _mm256_cmp_ps(ph, vpi, _CMP_GT_OQ));
            ph = _mm256_blendv_ps(ph, _mm256_add_ps(ph, v2pi), _mm256_cmp_ps(ph, vmpi, _CMP_LT_OQ));
            __m256i pbn = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_add_ps(ph, vpi2), vpfac));
            pbn = _mm256_min_epi32(_mm256_max_epi32(pbn, _mm256_setzero_si256()),
                                   _mm256_set1_epi32(::mkseeder::kGapRayNPhiBins - 1));
            // the lanes that fail the cuts can hold anything: gather only for the others
            const __m256i okm = _mm256_castps_si256(ok);
            pbn = _mm256_and_si256(pbn, okm);
            const __m256 r1 = _mm256_i32gather_ps(gr1, pbn, 4), r2 = _mm256_i32gather_ps(gr2, pbn, 4);
            // z = zaf + cot (r - raf) (contracted), k = (int)floor((z - zg0) * izc)
            const __m256 z1 = _mm256_fmadd_ps(cot, _mm256_sub_ps(r1, vra), vza),
                         z2 = _mm256_fmadd_ps(cot, _mm256_sub_ps(r2, vra), vza);
            const __m256i k1 = _mm256_cvttps_epi32(_mm256_floor_ps(_mm256_mul_ps(_mm256_sub_ps(z1, vzg0), vizc)));
            const __m256i k2 = _mm256_cvttps_epi32(_mm256_floor_ps(_mm256_mul_ps(_mm256_sub_ps(z2, vzg0), vizc)));
            const __m256i zero = _mm256_setzero_si256(), m1 = _mm256_set1_epi32(-1), b255 = _mm256_set1_epi32(0xff);
            // in range: 0 <= k < nzg (out of range counts as in a gap)
            const __m256i in1 = _mm256_andnot_si256(_mm256_cmpgt_epi32(zero, k1), _mm256_cmpgt_epi32(vnzg, k1));
            const __m256i in2 = _mm256_andnot_si256(_mm256_cmpgt_epi32(zero, k2), _mm256_cmpgt_epi32(vnzg, k2));
            const __m256i g1 = _mm256_and_si256(
                _mm256_mask_i32gather_epi32(zero, (const int *)gzg, k1, _mm256_and_si256(in1, okm), 1), b255);
            const __m256i g2 = _mm256_and_si256(
                _mm256_mask_i32gather_epi32(zero, (const int *)gzg, k2, _mm256_and_si256(in2, okm), 1), b255);
            const __m256i ed = _mm256_and_si256(_mm256_i32gather_epi32((const int *)gedge, pbn, 1), b255);
            const __m256i in1g =
                _mm256_or_si256(_mm256_xor_si256(in1, m1), _mm256_xor_si256(_mm256_cmpeq_epi32(g1, zero), m1));
            const __m256i in2g =
                _mm256_or_si256(_mm256_xor_si256(in2, m1), _mm256_xor_si256(_mm256_cmpeq_epi32(g2, zero), m1));
            const __m256i keep =
                _mm256_or_si256(_mm256_xor_si256(_mm256_cmpeq_epi32(ed, zero), m1), _mm256_or_si256(in1g, in2g));
            ok = _mm256_and_ps(ok, _mm256_castsi256_ps(keep));
          }
          // testHitB with the lanes' dr, dp, cot and z0 = zaf - cot raf (contracted)
          alignas(32) float pk[4][8];
          _mm256_store_ps(pk[0], dr), _mm256_store_ps(pk[1], dp), _mm256_store_ps(pk[2], cot);
          _mm256_store_ps(pk[3], _mm256_fnmadd_ps(cot, vra, vza));
          const unsigned int okm = _mm256_movemask_ps(ok);
          for (unsigned int m = okm; m; m &= m - 1) {
            const int l = std::countr_zero(m);
            const float pre[4] = {pk[0][l], pk[1][l], pk[2][l], pk[3][l]};
            f(i0 + l, pre);
          }
        }
#else
        for (uint32_t i = b; i < e; ++i)
          f(i, nullptr);
#endif
      }
    };  // struct ScanStart

    // stage d: the q pre-filter |q - (qm + t (qa1 + t qa2))| <= wq, t = (u - um) ihh
    struct ScanPre {
#ifdef MKSEEDER_CPU_AVX2
      __m256 vum, vihh, vqm, va1, va2, vw, vabs;
#endif
      ScanPre() = default;
      ALPAKA_FN_HOST_ACC ScanPre(float um, float ihh, float qm, float qa1, float qa2, float wq) {
#ifdef MKSEEDER_CPU_AVX2
        vum = _mm256_set1_ps(um), vihh = _mm256_set1_ps(ihh), vqm = _mm256_set1_ps(qm);
        va1 = _mm256_set1_ps(qa1), va2 = _mm256_set1_ps(qa2);
        vw = _mm256_set1_ps(wq * (1 + 1e-4f) + 1e-3f);
        vabs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
#endif
      }
      template <typename F>
      ALPAKA_FN_HOST_ACC void operator()(const float *u, const float *q, uint32_t b, uint32_t e, F &&f) const {
#ifdef MKSEEDER_CPU_AVX2
        for (uint32_t i0 = b; i0 < e; i0 += 8) {
          const uint32_t k = e - i0;
          const __m256i msk = tailMask(k);
          const __m256 vu = load8(u + i0, k, msk), vq = load8(q + i0, k, msk);
          const __m256 t = _mm256_mul_ps(_mm256_sub_ps(vu, vum), vihh);
          const __m256 pq = _mm256_add_ps(vqm, _mm256_mul_ps(t, _mm256_add_ps(va1, _mm256_mul_ps(t, va2))));
          const __m256 d = _mm256_and_ps(_mm256_sub_ps(vq, pq), vabs);
          // NaN lanes (d unordered) pass, for the exact test
          const __m256 ok = _mm256_cmp_ps(d, vw, _CMP_NGT_UQ);
          forEachSetBit(_mm256_movemask_ps(_mm256_and_ps(ok, _mm256_castsi256_ps(msk))), i0, f);
        }
#else
        for (uint32_t i = b; i < e; ++i)
          f(i);
#endif
      }
    };

    // stage c's window arithmetic for nb candidates, 8 at a time: the expressions
    // of the scalar stage c with the multiply-adds the compiler contracts into FMA (-ffp-contract=fast) as FMA
    // intrinsics, the rest as separate operations: out dp, idu, the q range (min, max of cq -+ q_c), the phi centre
    // and half width (cmid, chalf + wcphi + 1e-6), act = max(t0, t1) > 1
    template <typename TAcc>
    ALPAKA_FN_ACC inline void cwin8(const TAcc &acc,
                                    uint32_t nb,
                                    bool disc,
                                    float u0,
                                    float u1,
                                    float d0max,
                                    const float *ua_,
                                    const float *ub_,
                                    const float *qa_,
                                    const float *qb_,
                                    const float *pa_,
                                    const float *pb_,
                                    const float *ia_,
                                    const float *ib_,
                                    const float *pc_,
                                    const float *qc_,
                                    float *dp_,
                                    float *idu_,
                                    float *qlo_,
                                    float *qhi_,
                                    float *cm_,
                                    float *pw_,
                                    int *act_) {
#ifdef MKSEEDER_CPU_AVX2
      const __m256 vu0 = _mm256_set1_ps(u0), vu1 = _mm256_set1_ps(u1), vd0 = _mm256_set1_ps(d0max);
      const __m256 one = _mm256_set1_ps(1.0f), half = _mm256_set1_ps(0.5f), v2pi = _mm256_set1_ps(kTwoPi);
      const __m256 vi2pi = _mm256_set1_ps(kInvTwoPi), absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
      const __m256 eps = _mm256_set1_ps(1e-6f);
      // wrapPhiFloor(d) = d - kTwoPi * floor(d * kInvTwoPi + 0.5f)
      auto wrap8 = [&](__m256 d) {
        const __m256 fl = _mm256_floor_ps(_mm256_fmadd_ps(d, vi2pi, half));
        return _mm256_fnmadd_ps(v2pi, fl, d);
      };
      for (uint32_t j = 0; j < nb; j += 8) {
        const __m256 ua = _mm256_load_ps(ua_ + j), ub = _mm256_load_ps(ub_ + j), qa = _mm256_load_ps(qa_ + j);
        const __m256 qb = _mm256_load_ps(qb_ + j), pa = _mm256_load_ps(pa_ + j), pb = _mm256_load_ps(pb_ + j);
        const __m256 ia = _mm256_load_ps(ia_ + j), ib = _mm256_load_ps(ib_ + j), pc = _mm256_load_ps(pc_ + j);
        const __m256 qc = _mm256_load_ps(qc_ + j);
        const __m256 dp = wrap8(_mm256_sub_ps(pb, pa));
        const __m256 idu = _mm256_div_ps(one, _mm256_sub_ps(ub, ua));
        const __m256 t0 = _mm256_mul_ps(_mm256_sub_ps(vu0, ua), idu), t1 = _mm256_mul_ps(_mm256_sub_ps(vu1, ua), idu);
        const __m256 act = _mm256_cmp_ps(_mm256_max_ps(t1, t0), one, _CMP_GT_OQ);  // std::max(t0, t1) > 1
        const __m256 dq = _mm256_sub_ps(qb, qa);
        const __m256 cq0 = _mm256_fmadd_ps(dq, t0, qa), cq1 = _mm256_fmadd_ps(dq, t1, qa);
        const __m256 cp0 = _mm256_fmadd_ps(dp, t0, pa), cp1 = _mm256_fmadd_ps(dp, t1, pa);
        const __m256 far0 = _mm256_cmp_ps(_mm256_and_ps(t0, absm), _mm256_and_ps(t1, absm), _CMP_GT_OQ);
        const __m256 ufar = _mm256_blendv_ps(vu1, vu0, far0), tfar = _mm256_blendv_ps(t1, t0, far0);
        const __m256 rf0 = disc ? _mm256_fmadd_ps(dq, tfar, qa) : ufar;
        const __m256 rfar = _mm256_max_ps(rf0, half);  // std::max(0.5f, x)
        const __m256 in = _mm256_fmadd_ps(_mm256_sub_ps(ib, ia), tfar, ia);
        const __m256 wcphi = _mm256_fmadd_ps(vd0, _mm256_and_ps(_mm256_sub_ps(_mm256_div_ps(one, rfar), in), absm), pc);
        const __m256 qlo = _mm256_sub_ps(_mm256_min_ps(cq1, cq0), qc), qhi = _mm256_add_ps(_mm256_max_ps(cq1, cq0), qc);
        const __m256 dcp = wrap8(_mm256_sub_ps(cp1, cp0));
        const __m256 cm = _mm256_fmadd_ps(half, dcp, cp0);
        const __m256 pw = _mm256_add_ps(_mm256_fmadd_ps(half, _mm256_and_ps(dcp, absm), wcphi), eps);
        _mm256_store_ps(dp_ + j, dp), _mm256_store_ps(idu_ + j, idu), _mm256_store_ps(qlo_ + j, qlo);
        _mm256_store_ps(qhi_ + j, qhi), _mm256_store_ps(cm_ + j, cm), _mm256_store_ps(pw_ + j, pw);
        _mm256_store_si256((__m256i *)(act_ + j), _mm256_castps_si256(act));
      }
#else
      for (uint32_t j = 0; j < nb; ++j) {
        const float ua = ua_[j], ub = ub_[j], qa = qa_[j], qb = qb_[j], pa = pa_[j];
        const float dp = wrapPhiFloor(acc, pb_[j] - pa), idu = 1.0f / (ub - ua);
        const float t0_ = (u0 - ua) * idu, t1_ = (u1 - ua) * idu;
        act_[j] = std::max(t0_, t1_) > 1;
        const float cq0 = qa + (qb - qa) * t0_, cq1 = qa + (qb - qa) * t1_;
        const float cp0 = pa + dp * t0_, cp1 = pa + dp * t1_;
        const bool far0 = alpaka::math::abs(acc, t0_) > alpaka::math::abs(acc, t1_);
        const float ufar = far0 ? u0 : u1, tfar = far0 ? t0_ : t1_;
        const float rfar = std::max(0.5f, disc ? qa + (qb - qa) * tfar : ufar);
        const float wcphi = d0max * alpaka::math::abs(acc, 1 / rfar - (ia_[j] + (ib_[j] - ia_[j]) * tfar)) + pc_[j];
        qlo_[j] = std::min(cq0, cq1) - qc_[j], qhi_[j] = std::max(cq0, cq1) + qc_[j];
        const float dcp = wrapPhiFloor(acc, cp1 - cp0);
        cm_[j] = cp0 + 0.5f * dcp, pw_[j] = 0.5f * alpaka::math::abs(acc, dcp) + wcphi + 1e-6f;
        dp_[j] = dp, idu_[j] = idu;
      }
#endif
    }
    // the stage-c network (MkSeederStageCNet.h, trip_net_c) of n <= 8 found triplets (q[t].ih: the hit rows a, b, c):
    // lg[t] its logit, acot[t] its |cot theta| (the cut's |eta| bin), ok[t] false for a degenerate triplet (no logit: it
    // is kept). AVX2: the triplets as lanes (features and network); else the scalar functions.
#ifdef MKSEEDER_CPU_AVX2
    // log of 8 positive normal floats (the Cephes logf polynomial; within ~1e-7 relative of logf)
    inline __m256 stageCNetLog8(__m256 v) {
      const __m256i vi = _mm256_castps_si256(v);
      __m256 e = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_srli_epi32(vi, 23), _mm256_set1_epi32(126)));
      __m256 m = _mm256_castsi256_ps(
          _mm256_or_si256(_mm256_and_si256(vi, _mm256_set1_epi32(0x007fffff)), _mm256_set1_epi32(0x3f000000)));
      const __m256 lt = _mm256_cmp_ps(m, _mm256_set1_ps(0.707106781186547524f), _CMP_LT_OQ);
      e = _mm256_sub_ps(e, _mm256_and_ps(lt, _mm256_set1_ps(1.f)));
      const __m256 x = _mm256_sub_ps(_mm256_add_ps(m, _mm256_and_ps(lt, m)), _mm256_set1_ps(1.f));
      const __m256 z = _mm256_mul_ps(x, x);
      __m256 y = _mm256_set1_ps(7.0376836292E-2f);
      for (float c : {-1.1514610310E-1f,
                      1.1676998740E-1f,
                      -1.2420140846E-1f,
                      1.4249322787E-1f,
                      -1.6668057665E-1f,
                      2.0000714765E-1f,
                      -2.4999993993E-1f,
                      3.3333331174E-1f})
        y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(c));
      y = _mm256_mul_ps(_mm256_mul_ps(y, x), z);
      y = _mm256_fmadd_ps(e, _mm256_set1_ps(-2.12194440e-4f), y);
      y = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f), z, y);
      return _mm256_fmadd_ps(e, _mm256_set1_ps(0.693359375f), _mm256_add_ps(x, y));
    }
    // the features (computeTripletFeatures, with the hits' azimuths given) and the network of 8 triplets (lanes)
    template <typename TAcc>
    __attribute__((noinline)) void stageCNetLogits8(
        const TAcc &acc, const ChainFinderInputs &finder, const uint32_t (*ih)[3], float *lg, float *acot, bool *ok) {
      using namespace stagecnet;
      static_assert(kNIn == kTripletMlpNFeatures);
      alignas(32) float X[3][8], Y[3][8], Z[3][8], P[3][8], RW[3][8], CL[3][8];
      for (int t = 0; t < 8; ++t)
        for (int i = 0; i < 3; ++i) {
          const uint32_t h = ih[t][i];
          X[i][t] = finder.hits.x(h), Y[i][t] = finder.hits.y(h), Z[i][t] = finder.hits.z(h);
          P[i][t] = finder.cpuHits.phiBeam(h);
          RW[i][t] = float(std::min(32, int(finder.hits.clusterRows(h))));
          CL[i][t] = float(std::min(32, int(finder.hits.clusterCols(h))));
        }
      const __m256 bx = _mm256_set1_ps(finder.chainEvent.beamSpotX), by = _mm256_set1_ps(finder.chainEvent.beamSpotY),
                   bz = _mm256_set1_ps(finder.chainEvent.beamSpotZ);
      const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)), zero = _mm256_setzero_ps(),
                   one = _mm256_set1_ps(1.0f);
      __m256 x[kNIn], hx[3], hy[3];
      for (int i = 0; i < 3; ++i) {
        hx[i] = _mm256_sub_ps(_mm256_load_ps(X[i]), bx), hy[i] = _mm256_sub_ps(_mm256_load_ps(Y[i]), by);
        x[3 + i] = _mm256_sub_ps(_mm256_load_ps(Z[i]), bz);
        x[i] = _mm256_sqrt_ps(_mm256_fmadd_ps(hx[i], hx[i], _mm256_mul_ps(hy[i], hy[i])));
        x[13 + i] = _mm256_load_ps(RW[i]), x[16 + i] = _mm256_load_ps(CL[i]);
      }
      const __m256 ax = hx[0], ay = hy[0], qx = hx[1], qy = hy[1], cx = hx[2], cy = hy[2];
      const __m256 qy_cy = _mm256_sub_ps(qy, cy), cy_ay = _mm256_sub_ps(cy, ay), ay_qy = _mm256_sub_ps(ay, qy);
      __m256 d = _mm256_fmadd_ps(cx, ay_qy, _mm256_fmadd_ps(ax, qy_cy, _mm256_mul_ps(qx, cy_ay)));
      d = _mm256_add_ps(d, d);
      const __m256 dok = _mm256_cmp_ps(_mm256_and_ps(d, absm), _mm256_set1_ps(1e-6f), _CMP_GT_OQ);
      const __m256 ds = _mm256_blendv_ps(one, d, dok);
      const __m256 a2 = _mm256_fmadd_ps(ax, ax, _mm256_mul_ps(ay, ay)),
                   q2 = _mm256_fmadd_ps(qx, qx, _mm256_mul_ps(qy, qy)),
                   c2 = _mm256_fmadd_ps(cx, cx, _mm256_mul_ps(cy, cy));
      const __m256 ox =
          _mm256_div_ps(_mm256_fmadd_ps(ay_qy, c2, _mm256_fmadd_ps(qy_cy, a2, _mm256_mul_ps(q2, cy_ay))), ds);
      const __m256 oy = _mm256_div_ps(
          _mm256_fmadd_ps(_mm256_sub_ps(qx, ax),
                          c2,
                          _mm256_fmadd_ps(_mm256_sub_ps(cx, qx), a2, _mm256_mul_ps(q2, _mm256_sub_ps(ax, cx)))),
          ds);
      const __m256 dxo = _mm256_sub_ps(ax, ox), dyo = _mm256_sub_ps(ay, oy);
      const __m256 rr = _mm256_sqrt_ps(_mm256_fmadd_ps(dxo, dxo, _mm256_mul_ps(dyo, dyo)));
      const __m256 ro = _mm256_sqrt_ps(_mm256_fmadd_ps(ox, ox, _mm256_mul_ps(oy, oy)));
      x[6] = _mm256_and_ps(dok, _mm256_div_ps(one, _mm256_blendv_ps(one, rr, dok)));
      const __m256 dca = _mm256_and_ps(dok, _mm256_and_ps(_mm256_sub_ps(ro, rr), absm));
      x[7] = stageCNetLog8(_mm256_add_ps(dca, _mm256_set1_ps(1e-4f)));
      const __m256 dr = _mm256_sub_ps(x[2], x[0]), dz = _mm256_sub_ps(x[5], x[3]);
      const __m256 bad = _mm256_and_ps(_mm256_cmp_ps(dr, zero, _CMP_EQ_OQ), _mm256_cmp_ps(dz, zero, _CMP_EQ_OQ));
      const __m256 barrelish = _mm256_cmp_ps(_mm256_and_ps(dr, absm), _mm256_set1_ps(1e-3f), _CMP_GT_OQ);
      const __m256 cot =
          _mm256_blendv_ps(_mm256_set1_ps(1e3f), _mm256_div_ps(dz, _mm256_blendv_ps(one, dr, barrelish)), barrelish);
      auto clamp8 = [](__m256 a, float c) {
        return _mm256_max_ps(_mm256_min_ps(a, _mm256_set1_ps(c)), _mm256_set1_ps(-c));
      };
      x[8] = clamp8(_mm256_fnmadd_ps(cot, x[0], x[3]), 30.f);
      x[9] = clamp8(_mm256_sub_ps(x[4], _mm256_fmadd_ps(_mm256_sub_ps(x[1], x[0]), cot, x[3])), 5.f);
      x[10] = clamp8(cot, 20.f);
      auto wrapr8 = [&](__m256 a) {  // wrapPhiRound (MkSeederPostMath.h), std::round
        const __m256 u = _mm256_mul_ps(a, _mm256_set1_ps(mkseeder::kInvTwoPi));
        const __m256 tr = _mm256_round_ps(u, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        const __m256 up = _mm256_cmp_ps(_mm256_and_ps(_mm256_sub_ps(u, tr), absm), _mm256_set1_ps(0.5f), _CMP_GE_OQ);
        const __m256 rd =
            _mm256_add_ps(tr, _mm256_and_ps(up, _mm256_or_ps(one, _mm256_and_ps(u, _mm256_set1_ps(-0.0f)))));
        return _mm256_fnmadd_ps(rd, _mm256_set1_ps(mkseeder::kTwoPi), a);
      };
      const __m256 p0 = _mm256_load_ps(P[0]), p1 = _mm256_load_ps(P[1]), p2 = _mm256_load_ps(P[2]);
      x[11] = _mm256_and_ps(wrapr8(_mm256_sub_ps(p1, p0)), absm);
      x[12] = _mm256_and_ps(wrapr8(_mm256_sub_ps(p2, p1)), absm);
      _mm256_storeu_ps(acot, _mm256_and_ps(x[10], absm));
      // the z fold (foldTripletFeatures with fold 1): the signed features negated for cot theta < 0
      const __m256 neg = _mm256_and_ps(_mm256_cmp_ps(x[10], zero, _CMP_LT_OQ), _mm256_set1_ps(-0.0f));
      for (int i : {3, 4, 5, 8, 9, 10})
        x[i] = _mm256_xor_ps(x[i], neg);
      // the network: layer 1 in two blocks of 8 units (accumulators in registers), layer 2 over the stored layer 1
      alignas(32) float h1[kNHidden1][8];
      static_assert(kNHidden1 % 8 == 0 && kNHidden2 == 8);
      for (int j0 = 0; j0 < kNHidden1; j0 += 8) {
        __m256 a[8];
        for (int j = 0; j < 8; ++j)
          a[j] = _mm256_set1_ps(bias_layer1[j0 + j]);
        for (int i = 0; i < kNIn; ++i)
          for (int j = 0; j < 8; ++j)
            a[j] = _mm256_fmadd_ps(x[i], _mm256_set1_ps(wgtT_layer1[i][j0 + j]), a[j]);
        for (int j = 0; j < 8; ++j)
          _mm256_store_ps(h1[j0 + j], _mm256_max_ps(a[j], zero));
      }
      __m256 b[kNHidden2];
      for (int k = 0; k < kNHidden2; ++k)
        b[k] = _mm256_set1_ps(bias_layer2[k]);
      for (int j = 0; j < kNHidden1; ++j) {
        const __m256 hj = _mm256_load_ps(h1[j]);
        for (int k = 0; k < kNHidden2; ++k)
          b[k] = _mm256_fmadd_ps(hj, _mm256_set1_ps(wgtT_layer2[j][k]), b[k]);
      }
      __m256 o = _mm256_set1_ps(bias_layer3[0]);
      for (int k = 0; k < kNHidden2; ++k)
        o = _mm256_fmadd_ps(_mm256_max_ps(b[k], zero), _mm256_set1_ps(wgtT_layer3[k][0]), o);
      _mm256_storeu_ps(lg, o);
      const int badm = _mm256_movemask_ps(bad);
      for (int t = 0; t < 8; ++t)
        ok[t] = !((badm >> t) & 1);
    }
#endif
    template <typename TAcc, typename TPend>
    ALPAKA_FN_ACC inline void stageCNetLogits(
        const TAcc &acc, const ChainFinderInputs &finder, const TPend *q, int n, float *lg, float *acot, bool *ok) {
#ifdef MKSEEDER_CPU_AVX2
      if (finder.hasPhiBeam) {
        uint32_t ih[8][3];
        for (int t = 0; t < 8; ++t)
          for (int i = 0; i < 3; ++i)
            ih[t][i] = q[t < n ? t : 0].ih[i];
        float o[8], ac[8];
        bool ok8[8];
        stageCNetLogits8(acc, finder, ih, o, ac, ok8);
        for (int t = 0; t < n; ++t)
          lg[t] = o[t], acot[t] = ac[t], ok[t] = ok8[t];
        return;
      }
#endif
      for (int t = 0; t < n; ++t) {
        float x[3], y[3], z[3], f[kTripletMlpNFeatures];
        int rows[3], cols[3];
        for (int i = 0; i < 3; ++i) {
          const uint32_t h = q[t].ih[i];
          x[i] = finder.hits.x(h), y[i] = finder.hits.y(h), z[i] = finder.hits.z(h);
          rows[i] = finder.hits.clusterRows(h), cols[i] = finder.hits.clusterCols(h);
        }
        ok[t] = mkseeder::computeTripletFeatures(acc,
                                                 x,
                                                 y,
                                                 z,
                                                 rows,
                                                 cols,
                                                 finder.chainEvent.beamSpotX,
                                                 finder.chainEvent.beamSpotY,
                                                 finder.chainEvent.beamSpotZ,
                                                 f);
        acot[t] = alpaka::math::abs(acc, f[10]);
        lg[t] = ok[t] ? stageCNetLogit(acc, f) : 0.f;
      }
    }

    // KernelExtendToTriplets + KernelExtendToTripletsMiss at position p, in blocks of kBlk candidates: the
    // candidates and their hits a, b, the windows (cwin8) and bins, the runs, then the hit scans and the pushes in
    // candidate order
    template <typename TAcc>
    ALPAKA_FN_ACC inline void extendDoubletsToTriplets(const TAcc &acc,
                                                       const ChainFinderInputs &finder,
                                                       const Queues &candidateQueues,
                                                       int p) {
      constexpr uint32_t kBlk = 64;
      // the stage-c network (trip_net_c): the found triplets wait in batches of kStcBatch and are queued in their order
      // if they pass (stcFlush)
      constexpr int kStcBatch = 8;
      struct StcPend {
        Candidate t3;
        CandidateLine L3;
        int nq;
        uint32_t ih[3];
      };
      StcPend stcP[kStcBatch];
      int nStc = 0;
      struct CS {
        Candidate c;
        CandidateLine L;
        float cz0, ccot, ua, idu, qa, dqab, qcw, pa, dp, iA, dib, pcw, iqcw, fks_c, za, ra;
        int slo, shi, act;
        BinRange qc, pb;
        uint32_t r0, r1;
      };
      CS cs[kBlk];
      Router router;
      router.init(finder);
      // the serial backend: the candidates' lines and crossed positions from the pool (CandidateLine)
      const bool ml = kSerialAcc<TAcc> && candidateQueues.hasLines && router.use32;
      const bool stcOn = finder.chain.useStageCNet;
      auto stcFlush = [&]() {
        if (nStc == 0)
          return;
        float lg[kStcBatch], acot[kStcBatch];
        bool ok[kStcBatch];
        stageCNetLogits(acc, finder, stcP, nStc, lg, acot, ok);
        for (int t = 0; t < nStc; ++t) {
          if (ok[t] && lg[t] < stageCNetCut(finder.chain.stageCNetCut, acot[t]))
            continue;
          if (ml)
            pushCandidateWithLine(candidateQueues, q3(finder, stcP[t].nq), stcP[t].t3, stcP[t].L3);
          else
            push(acc, candidateQueues, q3(finder, stcP[t].nq), stcP[t].t3);
        }
        nStc = 0;
      };
      // the runs of a block: a fixed array, the vector only if a block has more
      constexpr uint32_t kMaxRuns = 16 * kBlk;
      BinRange runsBuf[kMaxRuns];
      std::vector<BinRange> runsVec;
      const MkSeederChainParameters &chainParameters = finder.chain;
      const int n = chainParameters.nPositions;
      const EventLayer targetLayer = finder.layers[finder.chainPositions[p].binnedLayerId()];
      const LayerAxes Tg = finder.layerGeometry[finder.chainPositions[p].binnedLayerId()];
      const bool disc = Tg.isDisc();
      const float u0 = targetLayer.qbarLo(), u1 = targetLayer.qbarHi();
      const int shT = shapeOfPosition(finder, p);
      const float d0c = stageCD0Max(finder, p);
      const int mpc = minPixelHitsOfThird(finder, p);  // the quads of the triplets with their third hit at p
      const float *uarr = (disc ? finder.hits.z() : finder.hits.r()).data(),
                  *qarr = (disc ? finder.hits.r() : finder.hits.z()).data();
      const float *hu = uarr + targetLayer.firstHit(), *hq = qarr + targetLayer.firstHit();
      [[maybe_unused]] const uint32_t np = ::mkseeder::kNPhiBins;
      [[maybe_unused]] const uint32_t *bst = finder.binStarts.start().data() + targetLayer.firstBinStart();
      const uint32_t nc = std::min(candidateQueues.queues.count(q2(finder, p)),
                                   candidateQueues.maxChunksPerQueue * Queues::kCandidateChunkSize);
      // one thread on the CPU backends; still split the items as the grid does
      for (uint32_t blk : cms::alpakatools::uniform_elements(acc, (nc + kBlk - 1) / kBlk)) {
        const uint32_t j0 = blk * kBlk, j1 = std::min(nc, j0 + kBlk);
        // pass 1: the candidates and their lines (scalar), the windows of 8 candidates at a time (AVX2: the same
        // operations as the scalar code, with FMA where the compiler contracts the scalar expressions), the bins
        alignas(32) float A_ua[kBlk], A_ub[kBlk], A_qa[kBlk], A_qb[kBlk], A_pa[kBlk], A_pb[kBlk], A_ia[kBlk],
            A_ib[kBlk];
        alignas(32) float A_pc[kBlk], A_qc[kBlk];
        alignas(32) float O_dp[kBlk], O_idu[kBlk], O_qlo[kBlk], O_qhi[kBlk], O_cm[kBlk], O_pw[kBlk];
        alignas(32) int O_act[kBlk];
        const uint32_t nb = j1 - j0;
        for (uint32_t j = j0; j < j1; ++j) {
          const uint32_t jj = j - j0;
          CS &s = cs[jj];
          s.act = 0;
          Candidate &c = s.c;
          // neutral inputs for the lanes that are not computed
          A_ua[jj] = 0.f, A_ub[jj] = 1.f, A_qa[jj] = A_qb[jj] = A_pa[jj] = A_pb[jj] = A_ia[jj] = A_ib[jj] = 0.f;
          A_pc[jj] = A_qc[jj] = 1.f;
          if (!(ml ? readCandidateWithLine(candidateQueues, q2(finder, p), j, c, s.L)
                   : readCandidate(candidateQueues, q2(finder, p), j, c))) {
            s.act = -1;  // lost: no miss either
            continue;
          }
          if (ml)
            s.cz0 = s.L.z0, s.ccot = s.L.cot;
          else
            line(acc, finder, c, s.cz0, s.ccot);
          const int ix = finder.windowTableByTriplet[(c.positions[0] * n + c.positions[1]) * n + p].windowTable();
          s.fks_c = chainParameters.fakeScoreCutForward > 0 &&
                            alpaka::math::abs(acc, s.ccot) >= chainParameters.forwardMinAbsCot
                        ? chainParameters.fakeScoreCutForward
                        : chainParameters.fakeScoreCut;
          if (!searchesCombination(chainParameters, ix))
            continue;
          const auto w = finder.windowTables[ix >= 0 ? ix : 0];
          const EventLayer La = finder.layers[finder.chainPositions[c.positions[0]].binnedLayerId()],
                           Lb = finder.layers[finder.chainPositions[c.positions[1]].binnedLayerId()];
          const uint32_t ia = La.firstHit() + c.hits[0], ib = Lb.firstHit() + c.hits[1];
          A_ua[jj] = uarr[ia], A_ub[jj] = uarr[ib], A_qa[jj] = qarr[ia], A_qb[jj] = qarr[ib];
          A_pa[jj] = finder.hits.phi(ia), A_pb[jj] = finder.hits.phi(ib), A_ia[jj] = finder.hits.invR(ia),
          A_ib[jj] = finder.hits.invR(ib);
          A_pc[jj] = w.phiWindowC(), A_qc[jj] = w.qWindowC();
          s.za = finder.hits.z(ia), s.ra = finder.hits.r(ia);
          s.act = 2;  // gathered
        }
        for (uint32_t jj = nb; jj < (nb + 7) / 8 * 8; ++jj)
          A_ua[jj] = 0.f, A_ub[jj] = 1.f, A_qa[jj] = A_qb[jj] = A_pa[jj] = A_pb[jj] = A_ia[jj] = A_ib[jj] = 0.f,
          A_pc[jj] = A_qc[jj] = 1.f;
        cwin8(acc,
              nb,
              disc,
              u0,
              u1,
              d0c,
              A_ua,
              A_ub,
              A_qa,
              A_qb,
              A_pa,
              A_pb,
              A_ia,
              A_ib,
              A_pc,
              A_qc,
              O_dp,
              O_idu,
              O_qlo,
              O_qhi,
              O_cm,
              O_pw,
              O_act);
#ifdef MKSEEDER_CPU_AVX2
        // the bins of the windows (qBinRangePadded, phiBinRangeAround) 8 candidates at a time: the same operations
        alignas(32) int B_qb[kBlk], B_qe[kBlk], B_pb[kBlk], B_pe[kBlk];
        binRanges8(Tg, nb, O_qlo, O_qhi, O_cm, O_pw, B_qb, B_qe, B_pb, B_pe);
#endif
        for (uint32_t jj = 0; jj < nb; ++jj) {
          CS &s = cs[jj];
          if (s.act != 2)
            continue;
          s.act = 0;
          if (!O_act[jj])
            continue;
          // all the rows of the GPU's kRows items, in their order
#ifdef MKSEEDER_CPU_AVX2
          s.qc = {(uint32_t)B_qb[jj], (uint32_t)B_qe[jj]};
          if (!(s.qc.first < s.qc.last))
            continue;
          s.pb = {(uint32_t)B_pb[jj], (uint32_t)B_pe[jj]};
#else
          s.qc = qBinRangePadded(acc, Tg, O_qlo[jj], O_qhi[jj]);
          if (!(s.qc.first < s.qc.last))
            continue;
          s.pb = phiBinRangeAround(acc, Tg, O_cm[jj], O_pw[jj]);
#endif
          const float idu = O_idu[jj];
          s.ua = A_ua[jj], s.idu = idu, s.qa = A_qa[jj], s.dqab = A_qb[jj] - A_qa[jj], s.qcw = A_qc[jj],
          s.pa = A_pa[jj];
          s.dp = O_dp[jj], s.iA = A_ia[jj], s.dib = A_ib[jj] - A_ia[jj], s.pcw = A_pc[jj], s.iqcw = 1.0f / A_qc[jj];
          s.slo = 0, s.shi = 1 << 30;
          if (shT >= 0) {
            const int band = shapeBand(finder, shT, shapeBin(finder, shT, alpaka::math::abs(acc, s.dqab * idu)));
            s.slo = finder.shapeBands[band].spanLow(), s.shi = finder.shapeBands[band].spanHigh();
          }
          s.act = 1;
        }
        // pass 2: the runs
        uint32_t nRuns = 0;
        bool inVec = false;
        runsVec.clear();
        for (uint32_t j = j0; j < j1; ++j) {
          CS &s = cs[j - j0];
          s.r0 = nRuns;
          if (s.act == 1)
            forEachBinRun(finder, targetLayer, s.pb, s.qc, [&](uint32_t b, uint32_t e) {
              if (!inVec && nRuns < kMaxRuns) {
                runsBuf[nRuns++] = {b, e};
                return;
              }
              if (!inVec)
                runsVec.assign(runsBuf, runsBuf + nRuns), inVec = true;
              runsVec.push_back({b, e});
              ++nRuns;
            });
          s.r1 = nRuns;
        }
        const BinRange *runs = inVec ? runsVec.data() : runsBuf;
        // pass 3: the hits, the triplets and the misses in candidate order
        for (uint32_t j = j0; j < j1; ++j) {
          const CS &s = cs[j - j0];
          if (s.act < 0)
            continue;
          const Candidate &c = s.c;
          bool found = false;
          if (s.act) {
            const float ua = s.ua, idu = s.idu, qa = s.qa, dqab = s.dqab, qcw = s.qcw, pa = s.pa, dp = s.dp;
            const float iA = s.iA, dib = s.dib, pcw = s.pcw, iqcw = s.iqcw, fks_c = s.fks_c, za = s.za, ra = s.ra;
            const int slo = s.slo, shi = s.shi;
            // a triplet with fewer than min_pix (min_pix_ot) pixel positions and no pixel position left on its line can
            // emit nothing (its quads and the triplet itself fail min_pix at the emit): not queued
            const bool lowPix = mpc > 0 && finder.chainPositions[c.positions[0]].isPixel() +
                                                   finder.chainPositions[c.positions[1]].isPixel() +
                                                   finder.chainPositions[p].isPixel() <
                                               mpc;
            const uint32_t pixLeft = router.okMask & router.pixMask & ~((2u << p) - 1u);
            auto hit = [&](uint32_t i) {
              const uint32_t ih = targetLayer.firstHit() + i;
              const float hu_ = uarr[ih], hq_ = qarr[ih];
              const float t = (hu_ - ua) * idu;
              const float dq = hq_ - (qa + dqab * t);
              if (!(t > 1) || !(alpaka::math::abs(acc, dq) <= qcw))
                return;
              const float dph = wrapPhiFloor(acc, wrapPhiFloor(acc, finder.hits.phi(ih) - pa) - dp * t);
              const float wph = d0c * alpaka::math::abs(acc, finder.hits.invR(ih) - (iA + dib * t)) + pcw;
              if (!(alpaka::math::abs(acc, dph) <= wph))
                return;
              float sc = 0;
              if (chainParameters.useFakeScore) {
                const float rq = dq * iqcw, rp = dph / wph;
                sc = rq * rq + rp * rp;
                if (sc >= fks_c || finder.hits.clusterCols(ih) < slo ||
                    (finder.hits.clusterCols(ih) > shi && !shapeExempt(finder, ih)))
                  return;
              }
              if ((finder.psMode & 4) && psUnmarked(finder, finder.chainPositions[p].layerId(), ih))
                return;
              if (psRejectStageC2(acc, finder, c, p, ih))
                return;
              found = true;
              const float zc = finder.hits.z(ih), rc = finder.hits.r(ih);
              const float cot = (zc - za) / (rc - ra), z0 = za - cot * ra;
              int st;
              const int allow = c.holes <= chainParameters.maxHolesForOuterTracker;
              CandidateLine L3;
              int nq;
              if (ml) {
                L3.z0 = z0, L3.cot = cot;
                router.sweep(acc, z0, cot, L3.crossedPositions, L3.definitePositions);
                if (lowPix && !(L3.crossedPositions & pixLeft))
                  return;
                nq = router.nextCrossedPositionFromMasks(p, L3, allow, st);
              } else
                nq = router.nextCrossedPosition(acc, finder, p, z0, cot, allow, st, [&] {
                  return mkseeder::nextCrossedPosition(acc, finder, p, z0, cot, allow, st);
                });
              if (nq < 0 || finder.chainPositions[nq].binnedLayerId() < 0)
                return;
              Candidate t3 = c;
              t3.hits[2] = i, t3.positions[2] = p, t3.score = sc, t3.stage = st;
              if (stcOn) {
                StcPend &q = stcP[nStc];
                q.t3 = t3, q.L3 = L3, q.nq = nq;
                q.ih[0] = finder.layers[finder.chainPositions[c.positions[0]].binnedLayerId()].firstHit() + c.hits[0];
                q.ih[1] = finder.layers[finder.chainPositions[c.positions[1]].binnedLayerId()].firstHit() + c.hits[1];
                q.ih[2] = ih;
                if (++nStc == kStcBatch)
                  stcFlush();
                return;
              }
              if (ml)
                pushCandidateWithLine(candidateQueues, q3(finder, nq), t3, L3);
              else
                push(acc, candidateQueues, q3(finder, nq), t3);
            };
            const ScanTQ scan(ua, idu, qa, dqab, qcw);
            for (uint32_t r = s.r0; r < s.r1; ++r)
              scan(hu, hq, runs[r].first, runs[r].last, hit);
          }
          // the miss (KernelExtendToTripletsMiss)
          if (found && !chainParameters.alwaysForwardMisses)
            continue;
          const int h = c.holes + (c.stage == 2);
          if (h > chainParameters.maxHoles)
            continue;
          int st;
          const int nq = ml ? router.nextCrossedPositionFromMasks(p, s.L, chainParameters.allowAnyOuterTracker, st)
                            : router.nextCrossedPosition(
                                  acc, finder, p, s.cz0, s.ccot, chainParameters.allowAnyOuterTracker, st, [&] {
                                    return mkseeder::nextCrossedPosition(
                                        acc, finder, p, s.cz0, s.ccot, chainParameters.allowAnyOuterTracker, st);
                                  });
          if (nq < 0 || finder.chainPositions[nq].binnedLayerId() < 0 || skipOuterTrackerMiss(finder, found, nq) ||
              skipMissOutsideGap(finder, c, p, s.cz0, s.ccot))
            continue;
          Candidate m = c;
          m.innerHoles += h > c.holes;
          m.holes = h, m.stage = st;
          if (ml)
            pushCandidateWithLine(candidateQueues, q2(finder, nq), m, s.L);
          else
            push(acc, candidateQueues, q2(finder, nq), m);
        }
      }  // for (blk : blocks of candidates)
      stcFlush();
    }  // extendDoubletsToTriplets()

  }  // namespace cpu

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif

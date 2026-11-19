#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederHelix8_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederHelix8_h

// Included by MkSeederKernels.h after HelixF: on the host with AVX2 (the CPU backends), HelixF's make / positionAtR and
// libm::glibc::atan2f for 8 lanes with the same values as the scalar functions (MKSEEDER_CPU_AVX2).

#include <bit>
#include "MkSeederBackend.h"

#ifdef MKSEEDER_CPU_AVX2

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {
  // HelixF::make / positionAtR / asinOx for 8 helices at a time: the operations of the compiled scalar functions, with
  // the multiply-adds the compiler contracts into FMA (-ffp-contract=fast) as FMA intrinsics in the same operand roles
  // and the rest separate, so the same values; asin for the lanes that need it by the scalar libm::glibc::asinf
  namespace avx2 {
    inline __m256 fma(__m256 a, __m256 b, __m256 c) { return _mm256_fmadd_ps(a, b, c); }
    inline __m256 fms(__m256 a, __m256 b, __m256 c) { return _mm256_fmsub_ps(a, b, c); }
    inline __m256 sel(__m256 m, __m256 a, __m256 b) { return _mm256_blendv_ps(b, a, m); }  // m ? a : b
    // asinOx(x) on the lanes of m
    inline __m256 asinOx8(__m256 x, __m256 m) {
      const __m256 one = _mm256_set1_ps(1.0f), x2 = _mm256_mul_ps(x, x);
      __m256 t = fma(_mm256_set1_ps(63.f / 2816), x2, _mm256_set1_ps(35.f / 1152));
      t = fma(t, x2, _mm256_set1_ps(5.f / 112));
      t = fma(t, x2, _mm256_set1_ps(3.f / 40));
      t = fma(t, x2, _mm256_set1_ps(1.f / 6));
      const __m256 poly = fma(x2, t, one);
      const __m256 mp = _mm256_cmp_ps(_mm256_set1_ps(0.04f), x2, _CMP_GT_OQ);  // x2 < 0.04f
      const __m256 m1 = _mm256_cmp_ps(x, one, _CMP_GE_OQ);
      __m256 r = sel(mp, poly, sel(m1, _mm256_set1_ps(kPi / 2), poly));
      unsigned int rest = _mm256_movemask_ps(_mm256_andnot_ps(_mm256_or_ps(mp, m1), m));
      if (rest) {
        alignas(32) float xs[8], rs[8];
        _mm256_store_ps(xs, x), _mm256_store_ps(rs, r);
        while (rest) {
          const int l = std::countr_zero(rest);
          rest &= rest - 1;
          rs[l] = libm::glibc::asinf(xs[l]) / xs[l];
        }
        r = _mm256_load_ps(rs);
      }
      return r;
    }
    struct Helix8 {
      __m256 k, tx, ty, cx, cy, cz, cot, ux, uy, im, c2, cn, hk, ahk, sq, ok;
    };
    // HelixF::make on the lanes of m (the others: anything)
    inline void make8(
        Helix8 &H, __m256 ax, __m256 ay, __m256 az, __m256 bx, __m256 by, __m256 cx, __m256 cy, __m256 cz, __m256 m) {
      const __m256 one = _mm256_set1_ps(1.0f), half = _mm256_set1_ps(0.5f), eps = _mm256_set1_ps(1e-6f);
      const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
      H.cx = cx, H.cy = cy, H.cz = cz;
      const __m256 vx0 = _mm256_sub_ps(bx, ax), vy0 = _mm256_sub_ps(by, ay), vx = _mm256_sub_ps(cx, bx),
                   vy = _mm256_sub_ps(cy, by);
      const __m256 lu = _mm256_sqrt_ps(fma(vx0, vx0, _mm256_mul_ps(vy0, vy0)));
      const __m256 lv = _mm256_sqrt_ps(fma(vy, vy, _mm256_mul_ps(vx, vx)));
      const __m256 wx = _mm256_sub_ps(cx, ax), wy = _mm256_sub_ps(cy, ay);
      const __m256 lw = _mm256_sqrt_ps(fma(wy, wy, _mm256_mul_ps(wx, wx)));
      H.ok = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(lu, eps, _CMP_GE_OQ), _mm256_cmp_ps(lv, eps, _CMP_GE_OQ)), m);
      __m256 num = fms(vx0, vy, _mm256_mul_ps(vy0, vx));
      num = _mm256_add_ps(num, num);
      const __m256 k = _mm256_div_ps(num, _mm256_mul_ps(_mm256_mul_ps(lu, lv), lw));
      H.k = k;
      const __m256 hk = _mm256_mul_ps(k, half);
      H.hk = hk;
      const __m256 sn = _mm256_min_ps(one, _mm256_max_ps(_mm256_set1_ps(-1.0f), _mm256_mul_ps(hk, lv)));
      const __m256 cs = _mm256_sqrt_ps(_mm256_fnmadd_ps(sn, sn, one));
      const __m256 ex = _mm256_div_ps(vx, lv), ey = _mm256_div_ps(vy, lv);
      const __m256 tx = fms(cs, ex, _mm256_mul_ps(ey, sn)), ty = fma(ex, sn, _mm256_mul_ps(ey, cs));
      H.tx = tx, H.ty = ty;
      const __m256 ahk = _mm256_mul_ps(_mm256_and_ps(k, absm), half);
      H.ahk = ahk;
      const __m256 xab = _mm256_min_ps(_mm256_mul_ps(ahk, lu), one), xbc = _mm256_min_ps(_mm256_mul_ps(ahk, lv), one);
      const __m256 sab = _mm256_mul_ps(lu, asinOx8(xab, H.ok));
      const __m256 den = fma(lv, asinOx8(xbc, H.ok), sab);
      const __m256 cot = _mm256_div_ps(_mm256_sub_ps(cz, az), den);
      H.cot = cot;
      const __m256 mx = fms(k, cx, ty), my = fma(cy, k, tx);
      const __m256 im = _mm256_div_ps(one, _mm256_sqrt_ps(fma(mx, mx, _mm256_mul_ps(my, my))));
      H.im = im, H.ux = _mm256_mul_ps(mx, im), H.uy = _mm256_mul_ps(my, im);
      H.c2 = fma(cy, cy, _mm256_mul_ps(cx, cx));
      H.cn = fma(cy, tx, _mm256_mul_ps(_mm256_xor_ps(cx, _mm256_set1_ps(-0.0f)), ty));
      H.sq = _mm256_sqrt_ps(fma(cot, cot, one));
    }
    // HelixF::positionAtR(r) on the lanes of m: ok, wrote (the outputs were written), px, py, q, s3
    inline void positionAtR8(
        const Helix8 &H, __m256 r, __m256 m, __m256 &ok, __m256 &wrote, __m256 &px, __m256 &py, __m256 &q, __m256 &s3) {
      const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
      const __m256 A = _mm256_mul_ps(fma(fma(r, r, H.c2), H.hk, H.cn), H.im);
      const __m256 g2 = fms(r, r, _mm256_mul_ps(A, A));
      const __m256 good = _mm256_andnot_ps(_mm256_cmp_ps(zero, g2, _CMP_GT_OQ), m);
      const __m256 g = _mm256_sqrt_ps(g2);
      const __m256 p1x = fms(H.ux, A, _mm256_mul_ps(H.uy, g)), p1y = fma(H.uy, A, _mm256_mul_ps(H.ux, g));
      const __m256 p2x = fma(H.ux, A, _mm256_mul_ps(H.uy, g)), p2y = fms(H.uy, A, _mm256_mul_ps(H.ux, g));
      const __m256 d1x = _mm256_sub_ps(p1x, H.cx), d2x = _mm256_sub_ps(p2x, H.cx);
      const __m256 d1y = _mm256_sub_ps(p1y, H.cy), d2y = _mm256_sub_ps(p2y, H.cy);
      const __m256 dot1 = fma(H.tx, d1x, _mm256_mul_ps(H.ty, d1y)), dot2 = fma(H.tx, d2x, _mm256_mul_ps(H.ty, d2y));
      const __m256 L2 = fma(d2x, d2x, _mm256_mul_ps(d2y, d2y)), L1 = fma(d1y, d1y, _mm256_mul_ps(d1x, d1x));
      const __m256 a1 = _mm256_cmp_ps(dot1, zero, _CMP_GT_OQ), a2 = _mm256_cmp_ps(dot2, zero, _CMP_GT_OQ);
      const __m256 s1 = _mm256_and_ps(a1,
                                      _mm256_or_ps(_mm256_xor_ps(a2, _mm256_castsi256_ps(_mm256_set1_epi32(-1))),
                                                   _mm256_cmp_ps(L1, L2, _CMP_LE_OQ)));
      wrote = _mm256_and_ps(good, _mm256_or_ps(a1, a2));
      px = sel(s1, p1x, p2x), py = sel(s1, p1y, p2y);
      const __m256 L = _mm256_sqrt_ps(sel(s1, L1, L2));
      const __m256 x = _mm256_min_ps(_mm256_mul_ps(L, H.ahk), one);
      const __m256 s = _mm256_mul_ps(L, asinOx8(x, wrote));
      q = fma(H.cot, s, H.cz);
      s3 = _mm256_mul_ps(s, H.sq);
      ok = _mm256_and_ps(wrote, _mm256_cmp_ps(s, zero, _CMP_GT_OQ));
    }
    inline __m256 nc8(__m256 p) {
      asm("" : "+x"(p));
      return p;
    }
    // libm::glibc::atan2f (MkSeederLibm.h) for 8 lanes: the same operations lane by lane; the special
    // arguments (NaN, inf, zeros, x = 1, |y / x| beyond 2^+-26) of the lanes in lanes by the scalar function
    inline __m256 atan2f8(__m256 y, __m256 x, __m256 lanes) {
      const __m256i hx = _mm256_castps_si256(x), hy = _mm256_castps_si256(y);
      const __m256i m7 = _mm256_set1_epi32(0x7fffffff);
      const __m256i ix = _mm256_and_si256(hx, m7), iy = _mm256_and_si256(hy, m7);
      const __m256i inf = _mm256_set1_epi32(0x7f800000);
      const __m256i k = _mm256_srai_epi32(_mm256_sub_epi32(iy, ix), 23);
      __m256i sp = _mm256_or_si256(_mm256_cmpgt_epi32(ix, _mm256_set1_epi32(0x7f7fffff)),
                                   _mm256_cmpgt_epi32(iy, _mm256_set1_epi32(0x7f7fffff)));
      sp = _mm256_or_si256(sp, _mm256_cmpeq_epi32(hx, _mm256_set1_epi32(0x3f800000)));
      sp = _mm256_or_si256(sp,
                           _mm256_or_si256(_mm256_cmpeq_epi32(ix, _mm256_setzero_si256()),
                                           _mm256_cmpeq_epi32(iy, _mm256_setzero_si256())));
      sp = _mm256_or_si256(sp, _mm256_cmpgt_epi32(k, _mm256_set1_epi32(26)));
      sp = _mm256_or_si256(sp,
                           _mm256_and_si256(_mm256_cmpgt_epi32(_mm256_setzero_si256(), hx),
                                            _mm256_cmpgt_epi32(_mm256_set1_epi32(-26), k)));
      (void)inf;
      const __m256 absm = _mm256_castsi256_ps(m7);
      const __m256 t = _mm256_and_ps(_mm256_div_ps(y, x), absm);
      const __m256i it = _mm256_castps_si256(t);
      const __m256 one = _mm256_set1_ps(1.0f), two = _mm256_set1_ps(2.0f), c15 = _mm256_set1_ps(1.5f);
      // the reductions: id -1 (t < 0.4375), 0 (< 0.6875), 1 (< 1.1875), 2 (< 2.4375), 3
      const __m256i lt0 = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x3ee00000), it);
      const __m256i lt1 = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x3f300000), it);
      const __m256i lt2 = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x3f980000), it);
      const __m256i lt3 = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x401c0000), it);
      auto selm = [](__m256i m, __m256 a, __m256 b) { return _mm256_blendv_ps(b, a, _mm256_castsi256_ps(m)); };
      // numerators / denominators: 0: (2t - 1) / (2 + t); 1: (t - 1) / (t + 1); 2: (t - 1.5) / (1 + 1.5t); 3: -1 / t
      const __m256 n0 = _mm256_sub_ps(nc8(_mm256_mul_ps(two, t)), one), d0 = _mm256_add_ps(two, t);
      const __m256 n1 = _mm256_sub_ps(t, one), d1 = _mm256_add_ps(t, one);
      const __m256 n2 = _mm256_sub_ps(t, c15), d2 = _mm256_add_ps(one, nc8(_mm256_mul_ps(c15, t)));
      const __m256 n3 = _mm256_set1_ps(-1.0f), d3 = t;
      const __m256 num = selm(lt1, n0, selm(lt2, n1, selm(lt3, n2, n3)));
      const __m256 den = selm(lt1, d0, selm(lt2, d1, selm(lt3, d2, d3)));
      const __m256 xr = selm(lt0, t, _mm256_div_ps(num, den));
      const __m256 z = nc8(_mm256_mul_ps(xr, xr)), w = nc8(_mm256_mul_ps(z, z));
      auto c = [](float v) { return _mm256_set1_ps(v); };
      auto mm = [](__m256 a, __m256 b) { return nc8(_mm256_mul_ps(a, b)); };
      const __m256 s1 =
          mm(z,
             _mm256_add_ps(
                 c(3.3333334327e-01f),
                 mm(w,
                    _mm256_add_ps(
                        c(1.4285714924e-01f),
                        mm(w,
                           _mm256_add_ps(
                               c(9.0908870101e-02f),
                               mm(w,
                                  _mm256_add_ps(
                                      c(6.6610731184e-02f),
                                      mm(w, _mm256_add_ps(c(4.9768779427e-02f), mm(w, c(1.6285819933e-02f))))))))))));
      const __m256 s2 = mm(
          w,
          _mm256_add_ps(
              c(-2.0000000298e-01f),
              mm(w,
                 _mm256_add_ps(
                     c(-1.1111110449e-01f),
                     mm(w,
                        _mm256_add_ps(c(-7.6918758452e-02f),
                                      mm(w, _mm256_add_ps(c(-5.8335702866e-02f), mm(w, c(-3.6531571299e-02f))))))))));
      const __m256 xs = mm(xr, _mm256_add_ps(s1, s2));
      const __m256 rneg = _mm256_sub_ps(xr, xs);  // id -1
      const __m256 hi = selm(lt1,
                             c(4.6364760399e-01f),
                             selm(lt2, c(7.8539812565e-01f), selm(lt3, c(9.8279368877e-01f), c(1.5707962513e+00f))));
      const __m256 lo = selm(lt1,
                             c(5.0121582440e-09f),
                             selm(lt2, c(3.7748947079e-08f), selm(lt3, c(3.4473217170e-08f), c(7.5497894159e-08f))));
      const __m256 rid = _mm256_sub_ps(hi, _mm256_sub_ps(_mm256_sub_ps(xs, lo), xr));
      __m256 a = selm(lt0, rneg, rid);
      // tiny (t < 2^-29): t; huge (t >= 2^25): atanhi[3] + atanlo[3]
      a = selm(_mm256_cmpgt_epi32(_mm256_set1_epi32(0x31000000), it), t, a);
      a = selm(_mm256_cmpgt_epi32(it, _mm256_set1_epi32(0x4bffffff)),
               _mm256_add_ps(c(1.5707962513e+00f), c(7.5497894159e-08f)),
               a);
      // the quadrant: m = sign(y) | 2 sign(x)
      const __m256 pi = c(3.1415927410e+00f), pi_lo = c(-8.7422776573e-08f);
      const __m256i yneg = _mm256_cmpgt_epi32(_mm256_setzero_si256(), hy),
                    xneg = _mm256_cmpgt_epi32(_mm256_setzero_si256(), hx);
      const __m256 q0 = selm(yneg, _mm256_xor_ps(a, c(-0.0f)), a);
      const __m256 q2 =
          selm(yneg, _mm256_sub_ps(_mm256_sub_ps(a, pi_lo), pi), _mm256_sub_ps(pi, _mm256_sub_ps(a, pi_lo)));
      __m256 r = selm(xneg, q2, q0);
      unsigned int ms = _mm256_movemask_ps(_mm256_and_ps(_mm256_castsi256_ps(sp), lanes));
      if (ms) {
        alignas(32) float ys[8], xs_[8], rs[8];
        _mm256_store_ps(ys, y), _mm256_store_ps(xs_, x), _mm256_store_ps(rs, r);
        while (ms) {
          const int l = std::countr_zero(ms);
          ms &= ms - 1;
          rs[l] = libm::glibc::atan2f(ys[l], xs_[l]);
        }
        r = _mm256_load_ps(rs);
      }
      return r;
    }
  }  // namespace avx2
  namespace avx2 {
    // HelixF::positionAtZ(u) on the lanes of m: ok (= the outputs were written), px, py, q, s3; sincCosSin's sin / cos for the
    // lanes beyond its polynomial by the scalar functions
    inline void positionAtZ8(
        const Helix8 &H, __m256 u, __m256 m, __m256 &ok, __m256 &px, __m256 &py, __m256 &q, __m256 &s3) {
      const __m256 one = _mm256_set1_ps(1.0f), absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
      const __m256 ds = _mm256_div_ps(_mm256_sub_ps(u, H.cz), H.cot);
      ok = _mm256_andnot_ps(_mm256_cmp_ps(_mm256_and_ps(H.cot, absm), _mm256_set1_ps(1e-9f), _CMP_LT_OQ), m);
      ok = _mm256_andnot_ps(_mm256_cmp_ps(ds, _mm256_setzero_ps(), _CMP_LE_OQ), ok);
      const __m256 h = _mm256_mul_ps(H.hk, ds), h2 = _mm256_mul_ps(h, h);
      __m256 ts = _mm256_fnmadd_ps(h2, _mm256_set1_ps(1.f / 5040), _mm256_set1_ps(1.f / 120));
      ts = _mm256_fnmadd_ps(h2, ts, _mm256_set1_ps(1.f / 6));
      __m256 sc = _mm256_fnmadd_ps(h2, ts, one);
      __m256 tc = _mm256_fnmadd_ps(h2, _mm256_set1_ps(1.f / 40320), _mm256_set1_ps(1.f / 720));
      tc = _mm256_fnmadd_ps(h2, tc, _mm256_set1_ps(1.f / 24));
      tc = _mm256_fnmadd_ps(h2, tc, _mm256_set1_ps(0.5f));
      __m256 ch = _mm256_fnmadd_ps(h2, tc, one), sh = _mm256_mul_ps(h, sc);
      unsigned int rest =
          _mm256_movemask_ps(_mm256_andnot_ps(_mm256_cmp_ps(h2, _mm256_set1_ps(0.04f), _CMP_LT_OQ), ok));
      if (rest) {
        alignas(32) float hs[8], scs[8], chs[8], shs[8];
        _mm256_store_ps(hs, h), _mm256_store_ps(scs, sc), _mm256_store_ps(chs, ch), _mm256_store_ps(shs, sh);
        for (; rest; rest &= rest - 1) {
          const int l = std::countr_zero(rest);
          const float s_ = std::sin(hs[l]), c_ = std::cos(hs[l]);
          shs[l] = s_, chs[l] = c_, scs[l] = s_ / hs[l];
        }
        sc = _mm256_load_ps(scs), ch = _mm256_load_ps(chs), sh = _mm256_load_ps(shs);
      }
      const __m256 f = _mm256_mul_ps(ds, sc);
      px = fma(f, fms(ch, H.tx, _mm256_mul_ps(sh, H.ty)), H.cx);
      py = fma(f, fma(sh, H.tx, _mm256_mul_ps(ch, H.ty)), H.cy);
      q = _mm256_sqrt_ps(fma(px, px, _mm256_mul_ps(py, py)));
      s3 = _mm256_mul_ps(ds, H.sq);
    }
    // one helix in every lane
    inline Helix8 bcast(const HelixF &h) {
      Helix8 H;
      H.k = _mm256_set1_ps(h.k), H.tx = _mm256_set1_ps(h.tx), H.ty = _mm256_set1_ps(h.ty), H.cx = _mm256_set1_ps(h.cx);
      H.cy = _mm256_set1_ps(h.cy), H.cz = _mm256_set1_ps(h.cz), H.cot = _mm256_set1_ps(h.cot),
      H.ux = _mm256_set1_ps(h.ux);
      H.uy = _mm256_set1_ps(h.uy), H.im = _mm256_set1_ps(h.im), H.c2 = _mm256_set1_ps(h.c2),
      H.cn = _mm256_set1_ps(h.cn);
      H.hk = _mm256_set1_ps(h.hk), H.ahk = _mm256_set1_ps(h.ahk), H.sq = _mm256_set1_ps(h.sq);
      H.ok = _mm256_castsi256_ps(_mm256_set1_epi32(h.ok ? -1 : 0));
      return H;
    }
    // wrapPhiFloor for 8 lanes (wrapPhiFloor is compiled with both multiply-adds contracted)
    inline __m256 wrapPhiFloor8(__m256 d) {
      const __m256 fl = _mm256_floor_ps(_mm256_fmadd_ps(d, _mm256_set1_ps(kInvTwoPi), _mm256_set1_ps(0.5f)));
      return _mm256_fnmadd_ps(_mm256_set1_ps(kTwoPi), fl, d);
    }
  }  // namespace avx2
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder
#endif

#endif

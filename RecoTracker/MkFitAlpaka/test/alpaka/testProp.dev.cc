// Prop-runs the Alpaka port of MkFitCore propagation / Kalman functions on the inputs
// written by testMkFitAlpakaPropMkFitCoreRef and compares with the MkFitCore outputs stored in the same file.
// One kernel per operation (so ptxas/cuobjdump report registers per function); N tracks per thread:
// N = 1 on GPU backends, 8 (NN for x86-64-v3) on CPU backends.
//
// Usage: testMkFitAlpakaProp<Backend> <mkfitcore_ref.bin>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/stringize.h"

#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/KalmanUtilsMPlex.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/PropagationMPlex.h"
#include "RecoTracker/MkFitAlpaka/test/propTestIO.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace ::mkfitdev::proptest;
namespace mkd = ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev;
using ::mkfitdev::prop::PropagationFlags;

// tracks per Alpaka thread: the mplex per-backend kNN (1 on GPU, 8 = NN on CPU)
constexpr int kN = mkd::kNN;

namespace {

  template <int N>
  struct Slots {
    mkd::MPlexLS<N> inErr, outErr;
    mkd::MPlexLV<N> inPar, outPar;
    mkd::MPlexQI<N> chg, fail, noMat;
    mkd::MPlexQF<N> msRad, msZ, chi2;
    mkd::MPlexHS<N> msErr;
    mkd::MPlexHV<N> msPar, pnt, nrm, dir;
  };

  template <int N>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void loadSlot(const Rec& r, Slots<N>& s, int n) {
    for (int i = 0; i < 6; ++i)
      s.inPar(n, i, 0) = r.inPar[i];
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j <= i; ++j)
        s.inErr(n, i, j) = r.inErr[symIdx(i, j)];
    s.chg(n, 0, 0) = r.chg;
    s.noMat(n, 0, 0) = r.noMatEff;
    s.msRad(n, 0, 0) = r.msRad;
    s.msZ(n, 0, 0) = r.msZ;
    for (int i = 0; i < 3; ++i) {
      s.msPar(n, i, 0) = r.msPar[i];
      s.pnt(n, i, 0) = r.plPnt[i];
      s.nrm(n, i, 0) = r.plNrm[i];
      s.dir(n, i, 0) = r.plDir[i];
      for (int j = 0; j <= i; ++j)
        s.msErr(n, i, j) = r.msErr[symIdx(i, j)];
    }
  }

  template <int N>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void storeSlot(Rec& r, const Slots<N>& s, int n) {
    for (int i = 0; i < 6; ++i)
      r.outPar[i] = s.outPar.constAt(n, i, 0);
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j <= i; ++j)
        r.outErr[symIdx(i, j)] = s.outErr.constAt(n, i, j);
    r.chi2 = s.chi2.constAt(n, 0, 0);
    r.fail = s.fail.constAt(n, 0, 0);
    r.chgOut = s.chg.constAt(n, 0, 0);
  }

  template <int N, int OP>
  struct PropKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, Rec* recs, int ntrk, PropagationFlags pf, int propToHit) const {
      const int ngroups = (ntrk + N - 1) / N;
      for (int g : cms::alpakatools::uniform_elements(acc, ngroups)) {
        const int base = g * N;
        const int N_proc = std::min(N, ntrk - base);
        Slots<N> s;
        for (int n = 0; n < N; ++n)
          loadSlot(recs[base + (n < N_proc ? n : 0)], s, n);
        s.outErr.setVal(0.f);
        s.outPar.setVal(0.f);
        s.chi2.setVal(0.f);
        s.fail.setVal(0);
        const bool p2h = propToHit;
        if constexpr (OP == kPropR) {
          mkd::propagateHelixToRMPlex(
              s.inErr, s.inPar, s.chg, s.msRad, s.outErr, s.outPar, s.fail, N_proc, pf, &s.noMat);
        } else if constexpr (OP == kPropZ) {
          mkd::propagateHelixToZMPlex(s.inErr, s.inPar, s.chg, s.msZ, s.outErr, s.outPar, s.fail, N_proc, pf, &s.noMat);
        } else if constexpr (OP == kPropPlane) {
          mkd::propagateHelixToPlaneMPlex(
              s.inErr, s.inPar, s.chg, s.pnt, s.nrm, s.outErr, s.outPar, s.fail, N_proc, pf, &s.noMat);
        } else if constexpr (OP == kChi2Plane) {
          mkd::kalmanPropagateAndComputeChi2Plane(
              s.inErr, s.inPar, s.chg, s.msErr, s.msPar, s.nrm, s.dir, s.pnt, s.chi2, s.outPar, s.fail, N_proc, pf, p2h);
        } else if constexpr (OP == kUpdPlane) {
          mkd::kalmanPropagateAndUpdatePlane(s.inErr,
                                             s.inPar,
                                             s.chg,
                                             s.msErr,
                                             s.msPar,
                                             s.nrm,
                                             s.dir,
                                             s.pnt,
                                             s.outErr,
                                             s.outPar,
                                             s.fail,
                                             N_proc,
                                             pf,
                                             p2h);
        } else if constexpr (OP == kUpdChi2Plane) {
          mkd::kalmanPropagateAndUpdateAndChi2Plane(s.inErr,
                                                    s.inPar,
                                                    s.chg,
                                                    s.msErr,
                                                    s.msPar,
                                                    s.nrm,
                                                    s.dir,
                                                    s.pnt,
                                                    s.outErr,
                                                    s.outPar,
                                                    s.fail,
                                                    s.chi2,
                                                    N_proc,
                                                    pf,
                                                    p2h,
                                                    &s.noMat);
        } else if constexpr (OP == kOpPlaneLocal) {
          mkd::kalmanOperationPlaneLocal(mkd::KFO_Calculate_Chi2 | mkd::KFO_Update_Params | mkd::KFO_Local_Cov,
                                         s.inErr,
                                         s.inPar,
                                         s.chg,
                                         s.msErr,
                                         s.msPar,
                                         s.nrm,
                                         s.dir,
                                         s.pnt,
                                         s.outErr,
                                         s.outPar,
                                         s.chi2,
                                         N_proc);
        }
        for (int n = 0; n < N_proc; ++n)
          storeSlot(recs[base + n], s, n);
      }
    }
  };

  template <int OP>
  void launch(Queue& queue, Rec* d_recs, int ntrk, PropagationFlags pf, int propToHit) {
    const int ngroups = (ntrk + kN - 1) / kN;
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    const int threads = 128;
#else
    const int threads = 1;
#endif
    const int blocks = cms::alpakatools::divide_up_by(ngroups, threads);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threads);
    alpaka::exec<Acc1D>(queue, workDiv, PropKernel<kN, OP>{}, d_recs, ntrk, pf, propToHit);
  }

  struct Stat {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    double q(double f) {
      if (v.empty())
        return 0;
      // host-side test statistics only (selection, no full sort)
      const size_t k = std::min<size_t>(v.size() - 1, size_t(f * v.size()));
      std::nth_element(v.begin(), v.begin() + k, v.end());
      return v[k];
    }
    double frac_below(double c) const {
      if (v.empty())
        return 1;
      size_t k = 0;
      for (double x : v)
        k += x <= c;
      return double(k) / v.size();
    }
  };

  double rel(double a, double b, double floor) { return std::abs(a - b) / std::max(std::abs(b), floor); }

  double dphi(double a, double b) {
    double d = a - b;
    while (d > M_PI)
      d -= 2 * M_PI;
    while (d < -M_PI)
      d += 2 * M_PI;
    return std::abs(d);
  }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s mkfitcore_ref.bin\n", argv[0]);
    return 1;
  }
  MaterialBlock mb;
  std::vector<OpBlock> ops;
  if (!readFile(argv[1], mb, ops)) {
    std::printf("cannot read %s\n", argv[1]);
    return 2;
  }

  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::printf("No devices available for the %s backend, skipping.\n", EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE));
    return 0;
  }
  auto const& device = devices[0];
  Queue queue(device);
  std::printf("backend %s, N (tracks per thread) = %d\n", EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE), kN);

  // material map on the device
  const int nmat = mb.nBinsZ * mb.nBinsR;
  auto h_bbxi = cms::alpakatools::make_host_buffer<float[]>(queue, nmat);
  auto h_radl = cms::alpakatools::make_host_buffer<float[]>(queue, nmat);
  std::copy(mb.bbxi.begin(), mb.bbxi.end(), h_bbxi.data());
  std::copy(mb.radl.begin(), mb.radl.end(), h_radl.data());
  auto d_bbxi = cms::alpakatools::make_device_buffer<float[]>(queue, nmat);
  auto d_radl = cms::alpakatools::make_device_buffer<float[]>(queue, nmat);
  alpaka::memcpy(queue, d_bbxi, h_bbxi);
  alpaka::memcpy(queue, d_radl, h_radl);
  ::mkfitdev::MaterialView mv{
      d_bbxi.data(), d_radl.data(), mb.nBinsZ, mb.nBinsR, mb.nBinsZ / mb.rngZ, mb.nBinsR / mb.rngR};

  // PROPTEST_GOT=<capture>: compare that capture's outputs (same inputs, e.g. MkFitCore built for x86-64-v2) with the
  // reference instead of the port's, with the same metrics.
  MaterialBlock gotMb;
  std::vector<OpBlock> gotOps;
  const char* gotFile = std::getenv("PROPTEST_GOT");
  if (gotFile && !readFile(gotFile, gotMb, gotOps)) {
    std::printf("cannot read %s\n", gotFile);
    return 2;
  }
  if (gotFile)
    std::printf("PROPTEST_GOT: comparing %s (not the port) with the reference\n", gotFile);
  int nBad = 0;
  int iBlock = -1;
  for (auto& b : ops) {
    ++iBlock;
    const int ntrk = b.recs.size();
    if (ntrk == 0)
      continue;
    auto h_recs = cms::alpakatools::make_host_buffer<Rec[]>(queue, ntrk);
    std::copy(b.recs.begin(), b.recs.end(), h_recs.data());
    auto d_recs = cms::alpakatools::make_device_buffer<Rec[]>(queue, ntrk);
    alpaka::memcpy(queue, d_recs, h_recs);
    PropagationFlags pf(b.pflags, mv);
    switch (b.op) {
      case kPropR:
        launch<kPropR>(queue, d_recs.data(), ntrk, pf, b.propToHit);
        break;
      case kPropZ:
        launch<kPropZ>(queue, d_recs.data(), ntrk, pf, b.propToHit);
        break;
      case kPropPlane:
        launch<kPropPlane>(queue, d_recs.data(), ntrk, pf, b.propToHit);
        break;
      case kChi2Plane:
        launch<kChi2Plane>(queue, d_recs.data(), ntrk, pf, b.propToHit);
        break;
      case kUpdPlane:
        launch<kUpdPlane>(queue, d_recs.data(), ntrk, pf, b.propToHit);
        break;
      case kUpdChi2Plane:
        launch<kUpdChi2Plane>(queue, d_recs.data(), ntrk, pf, b.propToHit);
        break;
      case kOpPlaneLocal:
        launch<kOpPlaneLocal>(queue, d_recs.data(), ntrk, pf, b.propToHit);
        break;
    }
    alpaka::memcpy(queue, h_recs, d_recs);
    alpaka::wait(queue);
    if (gotFile)
      for (int i = 0; i < ntrk; ++i)
        h_recs[i] = gotOps[iBlock].recs[i];

    // timing: kernel only (inputs re-copied each rep; the copy is outside the timed region)
    double nsPerSlot = 0;
    {
      const int reps = 5;
      double tot = 0;
      auto d_tmp = cms::alpakatools::make_device_buffer<Rec[]>(queue, ntrk);
      for (int r = 0; r < reps; ++r) {
        alpaka::memcpy(queue, d_tmp, d_recs);
        alpaka::wait(queue);
        const auto t0 = std::chrono::steady_clock::now();
        switch (b.op) {
          case kPropR:
            launch<kPropR>(queue, d_tmp.data(), ntrk, pf, b.propToHit);
            break;
          case kPropZ:
            launch<kPropZ>(queue, d_tmp.data(), ntrk, pf, b.propToHit);
            break;
          case kPropPlane:
            launch<kPropPlane>(queue, d_tmp.data(), ntrk, pf, b.propToHit);
            break;
          case kChi2Plane:
            launch<kChi2Plane>(queue, d_tmp.data(), ntrk, pf, b.propToHit);
            break;
          case kUpdPlane:
            launch<kUpdPlane>(queue, d_tmp.data(), ntrk, pf, b.propToHit);
            break;
          case kUpdChi2Plane:
            launch<kUpdChi2Plane>(queue, d_tmp.data(), ntrk, pf, b.propToHit);
            break;
          case kOpPlaneLocal:
            launch<kOpPlaneLocal>(queue, d_tmp.data(), ntrk, pf, b.propToHit);
            break;
        }
        alpaka::wait(queue);
        tot += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      }
      nsPerSlot = 1e9 * tot / reps / ntrk;
    }

    // compare
    const bool hasChi2 = (b.op == kChi2Plane || b.op == kUpdChi2Plane || b.op == kOpPlaneLocal);
    const bool hasErr = (b.op != kChi2Plane);
    Stat sPar, sErr, sErrF, sChi2, sPos;
    int nMkFitCoreNonPD = 0, nChi2FinMismatch = 0, nChi2BothNonFinite = 0, nOutlier = 0, nOutlierPath = 0;
    int failAgree = 0, nFailMkFitCore = 0, chgAgree = 0, nNonFinite = 0;
    for (int i = 0; i < ntrk; ++i) {
      const Rec& ref = b.recs[i];
      const Rec& got = h_recs[i];
      failAgree += (ref.fail != 0) == (got.fail != 0);
      nFailMkFitCore += ref.fail != 0;
      chgAgree += ref.chgOut == got.chgOut;
      bool finite = true;
      for (int k = 0; k < 6; ++k)
        finite = finite && std::isfinite(got.outPar[k]) == std::isfinite(ref.outPar[k]);
      if (!finite) {
        ++nNonFinite;
        continue;
      }
      // parameters: x,y,z relative to max(|ref|, 1 cm); 1/pT relative; phi, theta absolute (rad)
      double mpar = 0, mpos = 0;
      for (int k = 0; k < 3; ++k)
        mpos = std::max(mpos, rel(got.outPar[k], ref.outPar[k], 1.0));
      mpar = std::max(mpar, mpos);
      mpar = std::max(mpar, rel(got.outPar[3], ref.outPar[3], 1e-6));
      mpar = std::max(mpar, dphi(got.outPar[4], ref.outPar[4]));
      mpar = std::max(mpar, std::abs(double(got.outPar[5]) - ref.outPar[5]));
      sPar.add(mpar);
      sPos.add(mpos);
      double fdiff = 0;  // Frobenius relative difference of the errors (filled below when available)
      if (hasErr) {
        // errors: relative on the diagonal, off-diagonal relative to sqrt(Cii Cjj)
        double merr = 0;
        for (int r = 0; r < 6; ++r)
          for (int c = 0; c <= r; ++c) {
            const double sc = std::sqrt(std::abs(double(ref.outErr[symIdx(r, r)]) * double(ref.outErr[symIdx(c, c)])));
            merr = std::max(
                merr, std::abs(double(got.outErr[symIdx(r, c)]) - ref.outErr[symIdx(r, c)]) / std::max(sc, 1e-30));
          }
        sErr.add(merr);
        // Frobenius-norm relative difference of the full 6x6 (robust where MkFitCore itself loses
        // positive-definiteness)
        double dn = 0, rn = 0;
        bool npd = false;
        for (int r = 0; r < 6; ++r) {
          npd = npd || ref.outErr[symIdx(r, r)] < 0.f;
          for (int c = 0; c < 6; ++c) {
            const double d = double(got.outErr[symIdx(r, c)]) - ref.outErr[symIdx(r, c)];
            dn += d * d;
            rn += double(ref.outErr[symIdx(r, c)]) * ref.outErr[symIdx(r, c)];
          }
        }
        fdiff = std::sqrt(dn / std::max(rn, 1e-60));
        sErrF.add(fdiff);
        nMkFitCoreNonPD += npd;
        if (std::getenv("PROPTEST_DUMP") && merr > std::atof(std::getenv("PROPTEST_DUMP"))) {
          std::printf("   [dump] op %d trk %d merr %.3e chi2 ref %.6g got %.6g\n", b.op, i, merr, ref.chi2, got.chi2);
          std::printf("     in  par:");
          for (int k = 0; k < 6; ++k)
            std::printf(" %.6g", ref.inPar[k]);
          std::printf("\n     ref par:");
          for (int k = 0; k < 6; ++k)
            std::printf(" %.7g", ref.outPar[k]);
          std::printf("\n     got par:");
          for (int k = 0; k < 6; ++k)
            std::printf(" %.7g", got.outPar[k]);
          std::printf("\n     ref err:");
          for (int k = 0; k < 21; ++k)
            std::printf(" %.4g", ref.outErr[k]);
          std::printf("\n     got err:");
          for (int k = 0; k < 21; ++k)
            std::printf(" %.4g", got.outErr[k]);
          std::printf("\n     msErr:");
          for (int k = 0; k < 6; ++k)
            std::printf(" %.4g", ref.msErr[k]);
          std::printf("  nrm %.4f %.4f %.4f\n", ref.plNrm[0], ref.plNrm[1], ref.plNrm[2]);
        }
      }
      // outliers (param diff > 1e-4 or error diff > 1e-3): are they states that MkFitCore itself mishandles?
      if (mpar > 1e-4 || fdiff > 1e-3) {
        ++nOutlier;
        bool path = false;
        for (int r = 0; r < 6; ++r)
          path = path || ref.inErr[symIdx(r, r)] < 0.f || (hasErr && ref.outErr[symIdx(r, r)] < 0.f);
        path = path || (hasChi2 && !(std::abs(ref.chi2) < 1e3f));
        nOutlierPath += path;
      }
      if (hasChi2) {
        // MkFitCore evaluates slots whose hit was never loaded (MkFinder: hit_cnt >= XHitSize); those chi2 are
        // non-finite in MkFitCore and port alike and are counted, not compared
        const bool fr = std::isfinite(ref.chi2), fg = std::isfinite(got.chi2);
        if (fr && fg)
          sChi2.add(std::abs(double(got.chi2) - ref.chi2) / std::max(std::abs(double(ref.chi2)), 1e-3));
        else if (fr != fg)
          ++nChi2FinMismatch;
        else
          ++nChi2BothNonFinite;
      }
    }
    std::printf("\n== %s  (n=%d, pflags=%d, propToHit=%d)  port kernel %.2f ns/slot\n",
                opName(b.op),
                ntrk,
                b.pflags,
                b.propToHit,
                nsPerSlot);
    // FNV-1a over the port's output fields: run-to-run / device-to-device reproducibility check
    uint64_t hsh = 1469598103934665603ull;
    for (int i = 0; i < ntrk; ++i) {
      const Rec& got = h_recs[i];
      const unsigned char* q = reinterpret_cast<const unsigned char*>(&got.outPar[0]);
      const size_t nb = sizeof(Rec) - offsetof(Rec, outPar);
      for (size_t k = 0; k < nb; ++k)
        hsh = (hsh ^ q[k]) * 1099511628211ull;
    }
    std::printf("   port output hash %016llx\n", static_cast<unsigned long long>(hsh));
    std::printf("   fail flags agree: %d/%d (%.5f), MkFitCore fails %d; charge agree %d/%d; non-finite mismatch %d\n",
                failAgree,
                ntrk,
                double(failAgree) / ntrk,
                nFailMkFitCore,
                chgAgree,
                ntrk,
                nNonFinite);
    std::printf(
        "   params  max-diff per track: median %.2e  q99 %.2e  q999 %.2e  max %.2e  frac<=1e-5 %.5f  frac<=1e-4 %.5f\n",
        sPar.q(0.5),
        sPar.q(0.99),
        sPar.q(0.999),
        sPar.q(1.0),
        sPar.frac_below(1e-5),
        sPar.frac_below(1e-4));
    if (hasErr)
      std::printf(
          "   errors  max elem-diff/sqrt(CiiCjj): median %.2e  q99 %.2e  q999 %.2e  max %.2e  frac<=1e-4 %.5f  "
          "frac<=1e-3 %.5f\n",
          sErr.q(0.5),
          sErr.q(0.99),
          sErr.q(0.999),
          sErr.q(1.0),
          sErr.frac_below(1e-4),
          sErr.frac_below(1e-3));
    if (hasErr)
      std::printf(
          "   errors  |d|_F/|ref|_F:      median %.2e  q99 %.2e  q999 %.2e  max %.2e  frac<=1e-4 %.5f  frac<=1e-3 %.5f "
          " (MkFitCore non-PD outputs: %d)\n",
          sErrF.q(0.5),
          sErrF.q(0.99),
          sErrF.q(0.999),
          sErrF.q(1.0),
          sErrF.frac_below(1e-4),
          sErrF.frac_below(1e-3),
          nMkFitCoreNonPD);
    if (hasChi2)
      std::printf(
          "   chi2    rel-diff:           median %.2e  q99 %.2e  q999 %.2e  max %.2e  frac<=1e-4 %.5f  frac<=1e-3 "
          "%.5f\n",
          sChi2.q(0.5),
          sChi2.q(0.99),
          sChi2.q(0.999),
          sChi2.q(1.0),
          sChi2.frac_below(1e-4),
          sChi2.frac_below(1e-3));
    std::printf(
        "   outliers (param diff > 1e-4 or |dErr|_F/|Err|_F > 1e-3): %d (%.5f), of which MkFitCore-pathological "
        "(negative variance in MkFitCore in/out, or MkFitCore chi2 >= 1e3): %d\n",
        nOutlier,
        double(nOutlier) / ntrk,
        nOutlierPath);
    if (hasChi2)
      std::printf("   chi2    non-finite in both: %d, finiteness mismatch: %d\n", nChi2BothNonFinite, nChi2FinMismatch);
    if (double(failAgree) / ntrk < 0.999 || sPar.frac_below(1e-3) < 0.99 || nChi2FinMismatch > 0 || nNonFinite > 0)
      ++nBad;
  }
  std::printf("\n%s: %d operation(s) outside the loose agreement gate\n", nBad ? "FAIL" : "PASS", nBad);
  return nBad ? 3 : 0;
}

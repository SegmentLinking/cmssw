// Runs the Alpaka port of the MkFitCore propagation / Kalman functions and mini propagators on the inputs written by
// testMkFitAlpakaPropMkFitCoreRef and compares with the MkFitCore outputs stored in the same file.
// Usage: testMkFitAlpakaProp<Backend> <mkfitcore_ref.bin>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/KalmanUtilsMPlex.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/PropagationMPlex.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/select/MiniPropagators.h"
#include "RecoTracker/MkFitAlpaka/test/propTestIO.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace ::mkfitdev::proptest;
namespace mkd = ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev;
namespace mini = ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::mini_propagators;
using ::mkfitdev::prop::PropagationFlags;

constexpr int kN = mkd::kNN;

namespace {

  template <int N>
  struct Slots {
    mkd::MPlexLS<N> inErr, outErr;
    mkd::MPlexLV<N> inPar, outPar;
    mkd::MPlexQI<N> chg, fail, noMat;
    mkd::MPlexQF<N> msRad, msZ, chi2, modRadl, modBbxi, msRefP;
    mkd::MPlexHS<N> msErr;
    mkd::MPlexHV<N> msPar, pnt, nrm, dir;
  };

  template <int N>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void loadSlot(const Rec& rec, Slots<N>& slots, int slot) {
    for (int i = 0; i < 6; ++i) {
      slots.inPar(slot, i, 0) = rec.inPar[i];
      for (int j = 0; j <= i; ++j)
        slots.inErr(slot, i, j) = rec.inErr[symIdx(i, j)];
    }
    slots.chg(slot, 0, 0) = rec.chg;
    slots.noMat(slot, 0, 0) = 0;
    slots.msRad(slot, 0, 0) = rec.msRad;
    slots.msZ(slot, 0, 0) = rec.msZ;
    slots.modRadl(slot, 0, 0) = rec.modRadl;
    slots.modBbxi(slot, 0, 0) = rec.modBbxi;
    slots.msRefP(slot, 0, 0) = rec.msRefP;
    for (int i = 0; i < 3; ++i) {
      slots.msPar(slot, i, 0) = rec.msPar[i];
      slots.pnt(slot, i, 0) = rec.plPnt[i];
      slots.nrm(slot, i, 0) = rec.plNrm[i];
      slots.dir(slot, i, 0) = rec.plDir[i];
      for (int j = 0; j <= i; ++j)
        slots.msErr(slot, i, j) = rec.msErr[symIdx(i, j)];
    }
  }

  template <int N>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void storeSlot(Rec& rec, const Slots<N>& slots, int slot) {
    for (int i = 0; i < 6; ++i) {
      rec.outPar[i] = slots.outPar.constAt(slot, i, 0);
      for (int j = 0; j <= i; ++j)
        rec.outErr[symIdx(i, j)] = slots.outErr.constAt(slot, i, j);
    }
    rec.chi2 = slots.chi2.constAt(slot, 0, 0);
    rec.fail = slots.fail.constAt(slot, 0, 0);
    rec.chgOut = slots.chg.constAt(slot, 0, 0);
  }

  // the outputs of a mini-propagator operation: outPar = (x, y, z, px, py, pz), chi2 = dalpha
  template <int N>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void storeMini(
      Slots<N>& slots, int slot, float x, float y, float z, float px, float py, float pz, float dalpha, int fail) {
    const float out[6] = {x, y, z, px, py, pz};
    for (int i = 0; i < 6; ++i)
      slots.outPar(slot, i, 0) = out[i];
    slots.chi2(slot, 0, 0) = dalpha;
    slots.fail(slot, 0, 0) = fail;
  }

  template <int N, int OP>
  struct PropKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, Rec* recs, int ntrk, PropagationFlags pf, int propToHit) const {
      for (int group : cms::alpakatools::uniform_elements(acc, cms::alpakatools::divide_up_by(ntrk, N))) {
        const int base = group * N;
        const int N_proc = std::min(N, ntrk - base);
        Slots<N> slots;
        for (int n = 0; n < N; ++n)
          loadSlot(recs[base + (n < N_proc ? n : 0)], slots, n);
        slots.outErr.setVal(0.f);
        slots.outPar.setVal(0.f);
        slots.chi2.setVal(0.f);
        slots.fail.setVal(0);
        slots.noMat.setVal(OP == kPropPlaneNoMat ? 1 : 0);
        const bool p2h = propToHit;
        if constexpr (OP == kPropR) {
          mkd::propagateHelixToRMPlex(slots.inErr,
                                      slots.inPar,
                                      slots.chg,
                                      slots.msRad,
                                      slots.outErr,
                                      slots.outPar,
                                      slots.fail,
                                      N_proc,
                                      pf,
                                      &slots.noMat);
        } else if constexpr (OP == kPropZ) {
          mkd::propagateHelixToZMPlex(slots.inErr,
                                      slots.inPar,
                                      slots.chg,
                                      slots.msZ,
                                      slots.outErr,
                                      slots.outPar,
                                      slots.fail,
                                      N_proc,
                                      pf,
                                      &slots.noMat);
        } else if constexpr (OP == kPropPlane || OP == kPropPlaneNoMat) {
          mkd::propagateHelixToPlaneMPlex(slots.inErr,
                                          slots.inPar,
                                          slots.chg,
                                          slots.pnt,
                                          slots.nrm,
                                          slots.outErr,
                                          slots.outPar,
                                          slots.fail,
                                          N_proc,
                                          pf,
                                          &slots.noMat);
        } else if constexpr (OP == kUpdPlane || OP == kUpdPlaneNoProp) {
          mkd::kalmanPropagateAndUpdatePlane(slots.inErr,
                                             slots.inPar,
                                             slots.chg,
                                             slots.msErr,
                                             slots.msPar,
                                             slots.nrm,
                                             slots.dir,
                                             slots.pnt,
                                             slots.outErr,
                                             slots.outPar,
                                             slots.fail,
                                             N_proc,
                                             pf,
                                             p2h);
        } else if constexpr (OP == kChi2Plane) {
          mkd::kalmanPropagateAndComputeChi2Plane(slots.inErr,
                                                  slots.inPar,
                                                  slots.chg,
                                                  slots.msErr,
                                                  slots.msPar,
                                                  slots.nrm,
                                                  slots.dir,
                                                  slots.pnt,
                                                  slots.chi2,
                                                  slots.outPar,
                                                  slots.fail,
                                                  N_proc,
                                                  pf,
                                                  p2h);
        } else if constexpr (OP == kFitUpdFwd) {
          // the forward pass: no sub-steps, no fixed-momentum multiple scattering
          mkd::final_fit::propagate_update<N, mkd::NoCpe, false>(slots.inErr,
                                                                 slots.inPar,
                                                                 slots.chg,
                                                                 slots.msErr,
                                                                 slots.msPar,
                                                                 slots.nrm,
                                                                 slots.dir,
                                                                 slots.pnt,
                                                                 slots.outErr,
                                                                 slots.outPar,
                                                                 slots.fail,
                                                                 slots.chi2,
                                                                 N_proc,
                                                                 pf,
                                                                 1,
                                                                 &slots.noMat,
                                                                 mkd::NoCpe{},
                                                                 &slots.modRadl,
                                                                 &slots.modBbxi,
                                                                 nullptr);
        } else if constexpr (OP == kFitUpdBkw) {
          // the backward pass: sub-steps for the slots off their plane, multiple scattering at msRefP
          mkd::final_fit::propagate_update<N>(slots.inErr,
                                              slots.inPar,
                                              slots.chg,
                                              slots.msErr,
                                              slots.msPar,
                                              slots.nrm,
                                              slots.dir,
                                              slots.pnt,
                                              slots.outErr,
                                              slots.outPar,
                                              slots.fail,
                                              slots.chi2,
                                              N_proc,
                                              pf,
                                              kBkwSubSteps,
                                              &slots.noMat,
                                              mkd::NoCpe{},
                                              &slots.modRadl,
                                              &slots.modBbxi,
                                              &slots.msRefP);
        } else if constexpr (OP == kMiniPlexR || OP == kMiniPlexZ) {
          const mini::InitialStatePlex<N> isp(slots.inPar.fArray, slots.chg.fArray);
          mini::StatePlex<N> sp;
          if constexpr (OP == kMiniPlexR)
            isp.propagate_to_r(slots.msRad.fArray, sp);
          else
            isp.propagate_to_z(slots.msZ.fArray, sp);
          for (int n = 0; n < N; ++n)
            storeMini(slots, n, sp.x[n], sp.y[n], sp.z[n], sp.px[n], sp.py[n], sp.pz[n], sp.dalpha[n], sp.fail_flag[n]);
        } else if constexpr (OP == kMiniR || OP == kMiniPlane) {
          for (int n = 0; n < N_proc; ++n) {
            float par[6];
            for (int i = 0; i < 6; ++i)
              par[i] = slots.inPar(n, i, 0);
            const mini::InitialState is(par, slots.chg(n, 0, 0));
            mini::State st;
            if constexpr (OP == kMiniR) {
              is.propagate_to_r(slots.msRad(n, 0, 0), st);
            } else {
              const float pos[3] = {slots.pnt(n, 0, 0), slots.pnt(n, 1, 0), slots.pnt(n, 2, 0)};
              const float zdir[3] = {slots.nrm(n, 0, 0), slots.nrm(n, 1, 0), slots.nrm(n, 2, 0)};
              is.propagate_to_plane(pos, zdir, st);
            }
            storeMini(slots, n, st.x, st.y, st.z, st.px, st.py, st.pz, st.dalpha, st.fail_flag);
          }
        }
        for (int n = 0; n < N_proc; ++n)
          storeSlot(recs[base + n], slots, n);
      }
    }
  };

  template <int OP>
  void launch(Queue& queue, Rec* d_recs, int ntrk, PropagationFlags pf, int propToHit) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    const int threads = 128;
#else
    const int threads = 1;
#endif
    const int blocks = cms::alpakatools::divide_up_by(cms::alpakatools::divide_up_by(ntrk, kN), threads);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(blocks, threads),
                        PropKernel<kN, OP>{},
                        d_recs,
                        ntrk,
                        pf,
                        propToHit);
  }

  void run(int op, Queue& queue, Rec* d_recs, int ntrk, PropagationFlags pf, int propToHit) {
    switch (op) {
      case kPropR:
        return launch<kPropR>(queue, d_recs, ntrk, pf, propToHit);
      case kPropZ:
        return launch<kPropZ>(queue, d_recs, ntrk, pf, propToHit);
      case kPropPlane:
        return launch<kPropPlane>(queue, d_recs, ntrk, pf, propToHit);
      case kUpdPlane:
        return launch<kUpdPlane>(queue, d_recs, ntrk, pf, propToHit);
      case kFitUpdBkw:
        return launch<kFitUpdBkw>(queue, d_recs, ntrk, pf, propToHit);
      case kChi2Plane:
        return launch<kChi2Plane>(queue, d_recs, ntrk, pf, propToHit);
      case kUpdPlaneNoProp:
        return launch<kUpdPlaneNoProp>(queue, d_recs, ntrk, pf, propToHit);
      case kPropPlaneNoMat:
        return launch<kPropPlaneNoMat>(queue, d_recs, ntrk, pf, propToHit);
      case kFitUpdFwd:
        return launch<kFitUpdFwd>(queue, d_recs, ntrk, pf, propToHit);
      case kMiniPlexR:
        return launch<kMiniPlexR>(queue, d_recs, ntrk, pf, propToHit);
      case kMiniPlexZ:
        return launch<kMiniPlexZ>(queue, d_recs, ntrk, pf, propToHit);
      case kMiniR:
        return launch<kMiniR>(queue, d_recs, ntrk, pf, propToHit);
      case kMiniPlane:
        return launch<kMiniPlane>(queue, d_recs, ntrk, pf, propToHit);
    }
  }

  double relDiff(double val, double ref, double minScale) {
    return std::abs(val - ref) / std::max(std::abs(ref), minScale);
  }

  double absDeltaPhi(double phi1, double phi2) { return std::abs(std::remainder(phi1 - phi2, 2 * M_PI)); }

  // Agreement gate, met on every backend for every operation. Per track: parameter difference (x, y, z relative to
  // max(|ref|, 1 cm), 1/pT relative, phi and theta absolute; mini propagators: the momentum relative to the reference
  // |p|, the turning angle absolute), covariance difference (relative Frobenius norm) and chi2 difference (relative to
  // max(|ref|, kChi2MinScale)); per metric, its median and its 99% quantile are gated.
  // the propagation status must agree almost always: a near-boundary track may flip
  constexpr double kMinFailAgreement = 0.999;
  // the typical track agrees at the rounding level of a long chain of single-precision operations
  constexpr double kMaxMedianDiff = 1e-3;
  // the tail: near-singular updates amplify rounding, but a wrong term moves most tracks well past this
  constexpr double kMaxQ99Diff = 1e-2;
  constexpr double kChi2MinScale = 1e-3;

  // per-track differences of one metric; the quantiles sort a copy
  struct Distribution {
    std::vector<double> values;
    double quantile(double fraction) const {
      if (values.empty())
        return 0.;
      std::vector<double> sorted(values);
      std::sort(sorted.begin(), sorted.end());
      return sorted[std::min<std::size_t>(sorted.size() - 1, std::size_t(fraction * sorted.size()))];
    }
    bool inGate() const { return quantile(0.5) <= kMaxMedianDiff && quantile(0.99) <= kMaxQ99Diff; }
    void print(const char* name) const {
      std::printf("    %-7s q50 %.2e q90 %.2e q99 %.2e q99.9 %.2e max %.2e\n",
                  name,
                  quantile(0.5),
                  quantile(0.9),
                  quantile(0.99),
                  quantile(0.999),
                  quantile(1.));
    }
  };

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
  Queue queue(devices[0]);

  const int nmat = mb.nBinsZ * mb.nBinsR;
  auto d_bbxi = cms::alpakatools::make_device_buffer<float[]>(queue, nmat);
  auto d_radl = cms::alpakatools::make_device_buffer<float[]>(queue, nmat);
  alpaka::memcpy(queue, d_bbxi, cms::alpakatools::make_host_view(mb.bbxi.data(), nmat));
  alpaka::memcpy(queue, d_radl, cms::alpakatools::make_host_view(mb.radl.data(), nmat));
  // field constants: the MkFitCore defaults of mkfit::Config::mag_* (building) and mag_fit_* (final fit)
  const ::mkfitdev::MaterialView mv{d_bbxi.data(),
                                    d_radl.data(),
                                    mb.nBinsZ,
                                    mb.nBinsR,
                                    mb.nBinsZ / mb.rngZ,
                                    mb.nBinsR / mb.rngR,
                                    ::mkfitdev::Config::BFieldParams{},
                                    ::mkfitdev::Config::BFieldParams{::mkfitdev::Config::mag_fit_c1,
                                                                     ::mkfitdev::Config::mag_fit_b0,
                                                                     ::mkfitdev::Config::mag_fit_b1,
                                                                     ::mkfitdev::Config::mag_fit_a}};

  int nBad = 0;
  for (auto& block : ops) {
    const int ntrk = block.recs.size();
    if (ntrk == 0)
      continue;
    auto h_recs = cms::alpakatools::make_host_buffer<Rec[]>(queue, ntrk);
    std::copy(block.recs.begin(), block.recs.end(), h_recs.data());
    auto d_recs = cms::alpakatools::make_device_buffer<Rec[]>(queue, ntrk);
    alpaka::memcpy(queue, d_recs, h_recs);
    // final fit: its propagation flags (use_fit_b_field) plus every FinalFitFlags switch; the energy-loss sign of
    // the pass (eloss_sign_from_pass) is outward in the forward pass
    using namespace ::mkfitdev::prop;
    const bool fit = block.op == kFitUpdBkw || block.op == kFitUpdFwd;
    const int outward = block.op == kFitUpdFwd ? PF_eloss_outward : PF_none;
    const PropagationFlags pf =
        fit ? PropagationFlags(PF_use_param_b_field | PF_apply_material | PF_use_fit_b_field | PF_b_field_at_mid |
                                   PF_radial_field_corr | PF_eloss_by_pass | PF_final_fit | outward,
                               mv)
            : PropagationFlags(block.pflags, mv);
    run(block.op, queue, d_recs.data(), ntrk, pf, block.propToHit);
    alpaka::memcpy(queue, h_recs, d_recs);
    alpaka::wait(queue);

    int failAgree = 0, nNonFinite = 0, nChi2FinMismatch = 0, nBitwise = 0;
    Distribution parDiff, errDiff, chi2Diff;
    for (int i = 0; i < ntrk; ++i) {
      const Rec& ref = block.recs[i];
      const Rec& got = h_recs[i];
      failAgree += (ref.fail != 0) == (got.fail != 0);
      nBitwise += std::memcmp(ref.outPar, got.outPar, sizeof(ref.outPar)) == 0 &&
                  std::memcmp(ref.outErr, got.outErr, sizeof(ref.outErr)) == 0 &&
                  std::memcmp(&ref.chi2, &got.chi2, sizeof(ref.chi2)) == 0 && ref.fail == got.fail;
      bool finite = true;
      for (int k = 0; k < 6; ++k)
        finite = finite && std::isfinite(got.outPar[k]) == std::isfinite(ref.outPar[k]);
      if (!finite) {
        ++nNonFinite;
        continue;
      }
      double maxParDiff = 0.;
      for (int k = 0; k < 3; ++k)
        maxParDiff = std::max(maxParDiff, relDiff(got.outPar[k], ref.outPar[k], 1.0));
      if (isMini(block.op)) {
        const double pRef = std::sqrt(double(ref.outPar[3]) * ref.outPar[3] + double(ref.outPar[4]) * ref.outPar[4] +
                                      double(ref.outPar[5]) * ref.outPar[5]);
        for (int k = 3; k < 6; ++k)
          maxParDiff = std::max(maxParDiff, std::abs(double(got.outPar[k]) - ref.outPar[k]) / pRef);
        maxParDiff = std::max(maxParDiff, std::abs(double(got.chi2) - ref.chi2));
      } else {
        maxParDiff = std::max(maxParDiff, absDeltaPhi(got.outPar[4], ref.outPar[4]));
        maxParDiff = std::max(maxParDiff, std::abs(double(got.outPar[5]) - ref.outPar[5]));
        maxParDiff = std::max(maxParDiff, relDiff(got.outPar[3], ref.outPar[3], 1e-6));
      }
      parDiff.values.push_back(maxParDiff);
      double diffNorm2 = 0, refNorm2 = 0;
      for (int k = 0; k < 21; ++k) {
        const double diff = double(got.outErr[k]) - ref.outErr[k];
        diffNorm2 += diff * diff;
        refNorm2 += double(ref.outErr[k]) * ref.outErr[k];
      }
      if (refNorm2 > 0.)  // kalmanPropagateAndComputeChi2Plane returns no covariance
        errDiff.values.push_back(std::sqrt(diffNorm2 / refNorm2));
      if (hasChi2(block.op)) {
        const bool refFinite = std::isfinite(ref.chi2), gotFinite = std::isfinite(got.chi2);
        nChi2FinMismatch += refFinite != gotFinite;
        if (refFinite && gotFinite)
          chi2Diff.values.push_back(relDiff(got.chi2, ref.chi2, kChi2MinScale));
      }
    }
    const double fFail = double(failAgree) / ntrk;
    std::string outside;
    if (fFail < kMinFailAgreement)
      outside += " fail-agreement";
    if (nNonFinite > 0 || nChi2FinMismatch > 0)
      outside += " non-finite";
    if (!parDiff.inGate())
      outside += " params";
    if (!errDiff.inGate())
      outside += " errors";
    if (!chi2Diff.inGate())
      outside += " chi2";
    std::printf("%-42s n=%6d fail-agree %.5f bitwise %.5f non-finite %d%s%s\n",
                opName(block.op),
                ntrk,
                fFail,
                double(nBitwise) / ntrk,
                nNonFinite + nChi2FinMismatch,
                outside.empty() ? "" : "  OUTSIDE THE GATE:",
                outside.c_str());
    parDiff.print("params");
    errDiff.print("errors");
    if (hasChi2(block.op))
      chi2Diff.print("chi2");
    if (!outside.empty())
      ++nBad;
  }
  std::printf("%s %s: %d operation(s) outside the agreement gate\n",
              nBad ? "FAIL" : "PASS",
              EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE),
              nBad);
  return nBad ? 3 : 0;
}

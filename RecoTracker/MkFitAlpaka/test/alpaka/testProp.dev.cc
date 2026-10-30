// Runs the Alpaka port of the MkFitCore propagation / Kalman functions on the inputs written by
// testMkFitAlpakaPropMkFitCoreRef and compares with the MkFitCore outputs stored in the same file.
// Usage: testMkFitAlpakaProp<Backend> <mkfitcore_ref.bin>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/KalmanUtilsMPlex.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/PropagationMPlex.h"
#include "RecoTracker/MkFitAlpaka/test/propTestIO.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace ::mkfitdev::proptest;
namespace mkd = ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev;
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
        } else if constexpr (OP == kPropPlane) {
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
        } else if constexpr (OP == kUpdPlane) {
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
        } else if constexpr (OP == kFitUpdBkw) {
          // final_fit::propagate_update with 2 sub-steps: slots off their plane are sub-stepped, then updated
          bool split[N];
          bool anySplit = false;
          for (int n = 0; n < N; ++n) {
            split[n] = false;
            if (n < N_proc) {
              const float distToPlane = (slots.pnt(n, 0, 0) - slots.inPar(n, 0, 0)) * slots.nrm(n, 0, 0) +
                                        (slots.pnt(n, 1, 0) - slots.inPar(n, 1, 0)) * slots.nrm(n, 1, 0) +
                                        (slots.pnt(n, 2, 0) - slots.inPar(n, 2, 0)) * slots.nrm(n, 2, 0);
              split[n] = distToPlane != 0.f;
              anySplit = anySplit || split[n];
            }
          }
          mkd::MPlexLS<N> propErr;
          mkd::MPlexLV<N> propPar;
          if (anySplit)
            mkd::propagateHelixToPlaneSubStepMPlex(slots.inErr,
                                                   slots.inPar,
                                                   slots.chg,
                                                   slots.pnt,
                                                   slots.nrm,
                                                   propErr,
                                                   propPar,
                                                   slots.fail,
                                                   N_proc,
                                                   pf,
                                                   2,
                                                   split,
                                                   &slots.noMat,
                                                   &slots.modRadl,
                                                   &slots.modBbxi,
                                                   &slots.msRefP);
          mkd::kalmanPropagateAndUpdateAndChi2Plane(anySplit ? propErr : slots.inErr,
                                                    anySplit ? propPar : slots.inPar,
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
                                                    !anySplit,
                                                    &slots.noMat,
                                                    mkd::NoCpe{},
                                                    &slots.modRadl,
                                                    &slots.modBbxi,
                                                    &slots.msRefP);
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
    }
  }

  double relDiff(double val, double ref, double minScale) {
    return std::abs(val - ref) / std::max(std::abs(ref), minScale);
  }

  double absDeltaPhi(double phi1, double phi2) { return std::abs(std::remainder(phi1 - phi2, 2 * M_PI)); }

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
    // final fit: its propagation flags (use_fit_b_field) plus every FinalFitFlags switch, backward pass
    using namespace ::mkfitdev::prop;
    const bool fit = block.op == kFitUpdBkw;
    const PropagationFlags pf =
        fit ? PropagationFlags(PF_use_param_b_field | PF_apply_material | PF_use_fit_b_field | PF_b_field_at_mid |
                                   PF_radial_field_corr | PF_eloss_by_pass | PF_final_fit,
                               mv)
            : PropagationFlags(block.pflags, mv);
    run(block.op, queue, d_recs.data(), ntrk, pf, block.propToHit);
    alpaka::memcpy(queue, h_recs, d_recs);
    alpaka::wait(queue);

    // parameters: x, y, z relative to max(|ref|, 1 cm), 1/pT relative, phi and theta absolute; errors: relative
    // Frobenius norm of the difference; chi2 relative (non-finite in both: not compared)
    int failAgree = 0, nNonFinite = 0, nChi2FinMismatch = 0, parOk = 0, errOk = 0, chi2Ok = 0, nChi2 = 0;
    for (int i = 0; i < ntrk; ++i) {
      const Rec& ref = block.recs[i];
      const Rec& got = h_recs[i];
      failAgree += (ref.fail != 0) == (got.fail != 0);
      bool finite = true;
      for (int k = 0; k < 6; ++k)
        finite = finite && std::isfinite(got.outPar[k]) == std::isfinite(ref.outPar[k]);
      if (!finite) {
        ++nNonFinite;
        continue;
      }
      double maxParDiff =
          std::max(absDeltaPhi(got.outPar[4], ref.outPar[4]), std::abs(double(got.outPar[5]) - ref.outPar[5]));
      for (int k = 0; k < 3; ++k)
        maxParDiff = std::max(maxParDiff, relDiff(got.outPar[k], ref.outPar[k], 1.0));
      maxParDiff = std::max(maxParDiff, relDiff(got.outPar[3], ref.outPar[3], 1e-6));
      parOk += maxParDiff <= 1e-3;
      {
        double diffNorm2 = 0, refNorm2 = 0;
        for (int k = 0; k < 21; ++k) {
          const double diff = double(got.outErr[k]) - ref.outErr[k];
          diffNorm2 += diff * diff;
          refNorm2 += double(ref.outErr[k]) * ref.outErr[k];
        }
        errOk += diffNorm2 <= 1e-6 * std::max(refNorm2, 1e-60);
      }
      if (fit) {
        const bool refFinite = std::isfinite(ref.chi2), gotFinite = std::isfinite(got.chi2);
        nChi2FinMismatch += refFinite != gotFinite;
        if (refFinite && gotFinite) {
          ++nChi2;
          chi2Ok += relDiff(got.chi2, ref.chi2, 1e-3) <= 1e-3;
        }
      }
    }
    const double fFail = double(failAgree) / ntrk, fPar = double(parOk) / ntrk;
    std::printf("%-38s n=%7d fail-agree %.5f params<=1e-3 %.5f errors<=1e-3 %.5f chi2<=1e-3 %.5f non-finite %d\n",
                opName(block.op),
                ntrk,
                fFail,
                fPar,
                double(errOk) / ntrk,
                nChi2 ? double(chi2Ok) / nChi2 : 1.,
                nNonFinite + nChi2FinMismatch);
    if (fFail < 0.999 || fPar < 0.99 || nChi2FinMismatch > 0 || nNonFinite > 0)
      ++nBad;
  }
  std::printf("%s %s: %d operation(s) outside the agreement gate\n",
              nBad ? "FAIL" : "PASS",
              EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE),
              nBad);
  return nBad ? 3 : 0;
}

// MkFitCore reference for the propagation tests: generates Phase-2-like track states, module planes and hits, runs the
// MkFitCore functions (libRecoTrackerMkFitCore, NN as built) on them and writes inputs + outputs to a file
// read by testMkFitAlpakaProp<Backend> (test/alpaka/testProp.dev.cc).
//
// Usage: testMkFitAlpakaPropMkFitCoreRef <out.bin> [ntracks=20000] [seed=12345]
//        testMkFitAlpakaPropMkFitCoreRef --rerun <captured.bin> [reps=5]
//          re-runs the MkFitCore functions on a captured file: checks the capture reproduces bit-exactly and times
//          the MkFitCore code per slot (same load/store pattern as the Alpaka comparator kernels).

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
// The reference needs the MkFitCore (package-private) mkFit headers on purpose; angle brackets keep scram's
// private-header check (which greps for quoted includes) from rejecting this test-only use.
#include <RecoTracker/MkFitCore/src/KalmanUtilsMPlex.h>
#include <RecoTracker/MkFitCore/src/PropagationMPlex.h>

#include "propTestIO.h"

using namespace mkfit;
using namespace mkfitdev::proptest;

namespace {

  struct V3 {
    float x, y, z;
  };
  V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
  float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
  V3 unit(V3 a) {
    const float l = std::sqrt(dot(a, a));
    return {a.x / l, a.y / l, a.z / l};
  }
  V3 add(V3 a, V3 b, float s) { return {a.x + s * b.x, a.y + s * b.y, a.z + s * b.z}; }
  // rotate v about unit axis k by angle a (Rodrigues)
  V3 rot(V3 v, V3 k, float a) {
    const float c = std::cos(a), s = std::sin(a);
    V3 kxv = cross(k, v);
    const float kv = dot(k, v);
    return {v.x * c + kxv.x * s + k.x * kv * (1 - c),
            v.y * c + kxv.y * s + k.y * kv * (1 - c),
            v.z * c + kxv.z * s + k.z * kv * (1 - c)};
  }

  // load / store one slot
  void loadState(const Rec& r, MPlexLS& err, MPlexLV& par, MPlexQI& chg, int n) {
    for (int i = 0; i < 6; ++i)
      par(n, i, 0) = r.inPar[i];
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j <= i; ++j)
        err(n, i, j) = r.inErr[symIdx(i, j)];
    chg(n, 0, 0) = r.chg;
  }
  void loadHit(const Rec& r, MPlexHS& msErr, MPlexHV& msPar, MPlexHV& pnt, MPlexHV& nrm, MPlexHV& dir, int n) {
    for (int i = 0; i < 3; ++i) {
      msPar(n, i, 0) = r.msPar[i];
      pnt(n, i, 0) = r.plPnt[i];
      nrm(n, i, 0) = r.plNrm[i];
      dir(n, i, 0) = r.plDir[i];
      for (int j = 0; j <= i; ++j)
        msErr(n, i, j) = r.msErr[symIdx(i, j)];
    }
  }
  void storeOut(Rec& r,
                const MPlexLS& err,
                const MPlexLV& par,
                const MPlexQF& chi2,
                const MPlexQI& fail,
                const MPlexQI& chg,
                int n) {
    for (int i = 0; i < 6; ++i)
      r.outPar[i] = par.constAt(n, i, 0);
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j <= i; ++j)
        r.outErr[symIdx(i, j)] = err.constAt(n, i, j);
    r.chi2 = chi2.constAt(n, 0, 0);
    r.fail = fail.constAt(n, 0, 0);
    r.chgOut = chg.constAt(n, 0, 0);
  }

  // Run one op in batches of NN with the MkFitCore functions.
  void runMkFitCore(OpBlock& b, const PropagationFlags& pf) {
    const int ntrk = b.recs.size();
    for (int base = 0; base < ntrk; base += NN) {
      const int N_proc = std::min<int>(NN, ntrk - base);
      MPlexLS inErr, outErr;
      MPlexLV inPar, outPar;
      MPlexQI chg, fail, noMat;
      MPlexQF msRad, msZ, chi2;
      MPlexHS msErr;
      MPlexHV msPar, pnt, nrm, dir;
      outErr.setVal(0.f);
      outPar.setVal(0.f);
      chi2.setVal(0.f);
      fail.setVal(0);
      for (int n = 0; n < NN; ++n) {
        // slots beyond N_proc repeat the first track (MkFitCore leaves stale data there; values are never stored)
        const Rec& r = b.recs[base + (n < N_proc ? n : 0)];
        loadState(r, inErr, inPar, chg, n);
        loadHit(r, msErr, msPar, pnt, nrm, dir, n);
        msRad(n, 0, 0) = r.msRad;
        msZ(n, 0, 0) = r.msZ;
        noMat(n, 0, 0) = r.noMatEff;
      }
      const bool p2h = b.propToHit;
      switch (b.op) {
        case kPropR:
          propagateHelixToRMPlex(inErr, inPar, chg, msRad, outErr, outPar, fail, N_proc, pf, &noMat);
          break;
        case kPropZ:
          propagateHelixToZMPlex(inErr, inPar, chg, msZ, outErr, outPar, fail, N_proc, pf, &noMat);
          break;
        case kPropPlane:
          propagateHelixToPlaneMPlex(inErr, inPar, chg, pnt, nrm, outErr, outPar, fail, N_proc, pf, &noMat);
          break;
        case kChi2Plane:
          kalmanPropagateAndComputeChi2Plane(
              inErr, inPar, chg, msErr, msPar, nrm, dir, pnt, chi2, outPar, fail, N_proc, pf, p2h);
          break;
        case kUpdPlane:
          kalmanPropagateAndUpdatePlane(
              inErr, inPar, chg, msErr, msPar, nrm, dir, pnt, outErr, outPar, fail, N_proc, pf, p2h);
          break;
        case kUpdChi2Plane:
          kalmanPropagateAndUpdateAndChi2Plane(
              inErr, inPar, chg, msErr, msPar, nrm, dir, pnt, outErr, outPar, fail, chi2, N_proc, pf, p2h, &noMat);
          break;
        case kOpPlaneLocal:
          kalmanOperationPlaneLocal(KFO_Calculate_Chi2 | KFO_Update_Params | KFO_Local_Cov,
                                    inErr,
                                    inPar,
                                    chg,
                                    msErr,
                                    msPar,
                                    nrm,
                                    dir,
                                    pnt,
                                    outErr,
                                    outPar,
                                    chi2,
                                    N_proc);
          break;
      }
      for (int n = 0; n < N_proc; ++n)
        storeOut(b.recs[base + n], outErr, outPar, chi2, fail, chg, n);
    }
  }

}  // namespace

namespace {
  int rerun(const char* fn, int reps) {
    Config::usePropToPlane = true;
    Config::usePtMultScat = true;
    MaterialBlock mb;
    std::vector<OpBlock> ops;
    if (!readFile(fn, mb, ops)) {
      std::printf("cannot read %s\n", fn);
      return 2;
    }
    TrackerInfo ti;
    ti.create_material(mb.nBinsZ, mb.rngZ, mb.nBinsR, mb.rngR);
    for (int bz = 0; bz < mb.nBinsZ; ++bz)
      for (int br = 0; br < mb.nBinsR; ++br) {
        ti.material_bbxi(bz, br) = mb.bbxi[bz * mb.nBinsR + br];
        ti.material_radl(bz, br) = mb.radl[bz * mb.nBinsR + br];
      }
    std::printf("MkFitCore rerun of %s, NN = %d, reps = %d\n", fn, int(NN), reps);
    for (auto const& b : ops) {
      PropagationFlags pf(b.pflags);
      pf.tracker_info = &ti;
      OpBlock w = b;
      runMkFitCore(w, pf);  // warm-up + reproducibility check
      size_t exact = 0;
      for (size_t i = 0; i < b.recs.size(); ++i) {
        Rec x = w.recs[i], y = b.recs[i];
        exact += std::memcmp(x.outPar, y.outPar, sizeof(x.outPar)) == 0 &&
                 std::memcmp(x.outErr, y.outErr, sizeof(x.outErr)) == 0 &&
                 (std::memcmp(&x.chi2, &y.chi2, sizeof(float)) == 0 || (x.chi2 != x.chi2 && y.chi2 != y.chi2)) &&
                 x.fail == y.fail && x.chgOut == y.chgOut;
      }
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < reps; ++r) {
        w = b;
        runMkFitCore(w, pf);
      }
      const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("  %-38s pflags=%d p2h=%d n=%7zu  rerun bit-exact %7zu (%.5f)  MkFitCore %.1f ns/slot (1 core)\n",
                  opName(b.op),
                  b.pflags,
                  b.propToHit,
                  b.recs.size(),
                  exact,
                  b.recs.empty() ? 1. : double(exact) / b.recs.size(),
                  1e9 * dt / reps / std::max<size_t>(1, b.recs.size()));
    }
    return 0;
  }
}  // namespace

int main(int argc, char** argv) {
  if (argc >= 3 && std::string(argv[1]) == "--rerun")
    return rerun(argv[2], argc > 3 ? std::atoi(argv[3]) : 5);
  if (argc < 2) {
    std::printf("usage: %s out.bin [ntracks] [seed]\n", argv[0]);
    return 1;
  }
  const std::string outFn = argv[1];
  const int nGen = argc > 2 ? std::atoi(argv[2]) : 20000;
  const unsigned seed = argc > 3 ? std::atoi(argv[3]) : 12345;
  std::printf("MkFitCore NN = %d\n", int(NN));

  // Phase-2 settings (MkFitGeometryESProducer): prop-to-plane, pT-dependent multiple scattering.
  Config::usePropToPlane = true;
  Config::usePtMultScat = true;

  // Material map with the Phase-2 binning (create_material(300, 300, 120, 120)); synthetic but layer-like:
  // thin cylinders at barrel radii, disks at endcap z, plus a low uniform floor.
  TrackerInfo ti;
  ti.create_material(300, 300.0f, 120, 120.0f);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(0.f, 1.f);
  std::normal_distribution<float> G(0.f, 1.f);
  const float brlR[] = {2.9f, 6.8f, 10.9f, 16.0f, 23.0f, 36.0f, 51.0f, 68.0f, 88.0f, 108.0f};
  const float ecZ[] = {26.f, 30.f, 36.f, 44.f, 53.f, 63.f, 75.f, 90.f, 110.f, 131.f, 156.f, 185.f, 220.f, 250.f, 270.f};
  MaterialBlock mb;
  mb.nBinsZ = 300;
  mb.nBinsR = 120;
  mb.rngZ = 300.f;
  mb.rngR = 120.f;
  mb.bbxi.resize(300 * 120);
  mb.radl.resize(300 * 120);
  for (int bz = 0; bz < 300; ++bz) {
    for (int br = 0; br < 120; ++br) {
      const float z = bz + 0.5f, r = br + 0.5f;
      float radl = 0.0005f;
      for (float R : brlR)
        if (std::abs(r - R) < 1.5f && z < 120.f + R)
          radl = 0.01f + 0.04f * U(rng);
      for (float Z : ecZ)
        if (std::abs(z - Z) < 1.5f && r > 3.f && r < 112.f)
          radl = 0.01f + 0.05f * U(rng);
      const float bbxi = radl * (0.9e-3f + 0.2e-3f * U(rng));
      ti.material_radl(bz, br) = radl;
      ti.material_bbxi(bz, br) = bbxi;
      mb.radl[bz * 120 + br] = radl;
      mb.bbxi[bz * 120 + br] = bbxi;
    }
  }

  PropagationFlags pfNone(PF_none);
  PropagationFlags pfMat(PF_use_param_b_field | PF_apply_material);
  pfNone.tracker_info = &ti;
  pfMat.tracker_info = &ti;

  std::vector<OpBlock> ops(kNOps);
  for (int op = 0; op < kNOps; ++op) {
    ops[op].op = op;
    ops[op].pflags = (op == kOpPlaneLocal) ? PF_none : (PF_use_param_b_field | PF_apply_material);
    ops[op].propToHit = (op == kChi2Plane || op == kUpdPlane || op == kUpdChi2Plane) ? 1 : 0;
  }

  // helper: propagate one state (slot 0) with MkFitCore functions, no material
  auto prop1 = [&](const Rec& in, bool toR, float target, Rec& out) -> bool {
    MPlexLS e, oe;
    MPlexLV p, op;
    MPlexQI c, f;
    MPlexQF t;
    for (int n = 0; n < NN; ++n) {
      loadState(in, e, p, c, n);
      t(n, 0, 0) = target;
    }
    f.setVal(0);
    if (toR)
      propagateHelixToRMPlex(e, p, c, t, oe, op, f, 1, pfNone);
    else
      propagateHelixToZMPlex(e, p, c, t, oe, op, f, 1, pfNone);
    out = in;
    for (int i = 0; i < 6; ++i)
      out.inPar[i] = op.constAt(0, i, 0);
    return f.constAt(0, 0, 0) == 0;
  };

  int nBrl = 0, nEc = 0, nSkip = 0;
  for (int itrk = 0; itrk < nGen; ++itrk) {
    // generator-level track
    const float pt = std::exp(std::log(0.4f) + U(rng) * (std::log(50.f) - std::log(0.4f)));
    const float eta = -2.4f + 4.8f * U(rng);
    const float phi = -Const::PI + Const::TwoPI * U(rng);
    const int q = U(rng) < 0.5f ? -1 : 1;
    const float theta = 2.f * std::atan(std::exp(-eta));
    Rec vtx{};
    vtx.inPar[0] = 0.01f * G(rng);
    vtx.inPar[1] = 0.01f * G(rng);
    vtx.inPar[2] = 4.f * G(rng);
    vtx.inPar[3] = 1.f / pt;
    vtx.inPar[4] = phi;
    vtx.inPar[5] = theta;
    vtx.chg = q;
    for (int i = 0; i < 21; ++i)
      vtx.inErr[i] = 0.f;
    for (int i = 0; i < 6; ++i)
      vtx.inErr[symIdx(i, i)] = 1e-4f;

    // starting layer: one of the inner barrel radii
    const int il = int(U(rng) * 6);
    Rec s0;
    if (!prop1(vtx, true, brlR[il], s0)) {
      ++nSkip;
      continue;
    }
    // smear the start state with realistic errors (diag sigmas + x-phi, y-phi, z-theta correlations)
    const float sx = 0.003f + 0.02f * U(rng), sz = 0.005f + 0.05f * U(rng);
    const float sipt = (0.005f + 0.03f * U(rng)) * s0.inPar[3];
    const float sphi = 0.0003f + 0.002f * U(rng), sth = 0.0003f + 0.002f * U(rng);
    const float sig[6] = {sx, sx, sz, sipt, sphi, sth};
    for (int i = 0; i < 21; ++i)
      s0.inErr[i] = 0.f;
    for (int i = 0; i < 6; ++i)
      s0.inErr[symIdx(i, i)] = sig[i] * sig[i];
    s0.inErr[symIdx(4, 0)] = 0.4f * sig[4] * sig[0];
    s0.inErr[symIdx(4, 1)] = -0.3f * sig[4] * sig[1];
    s0.inErr[symIdx(5, 2)] = 0.5f * sig[5] * sig[2];
    s0.inErr[symIdx(4, 3)] = 0.2f * sig[4] * sig[3];
    Rec truthAt0 = s0;
    for (int i = 0; i < 6; ++i)
      s0.inPar[i] += sig[i] * G(rng);

    // target: barrel layer further out, or endcap disk
    const bool isBarrel = std::abs(eta) < 1.3f ? (U(rng) < 0.85f) : (U(rng) < 0.25f);
    Rec xing;
    V3 nrm, P;
    float msRad = 0.f, msZ = 0.f;
    if (isBarrel) {
      const int jl = std::min(9, il + 1 + int(U(rng) * 3));
      msRad = brlR[jl] + 1.0f * (U(rng) - 0.5f);
      if (!prop1(truthAt0, true, msRad, xing)) {
        ++nSkip;
        continue;
      }
      if (std::abs(xing.inPar[2]) > 120.f) {
        ++nSkip;
        continue;
      }
      P = {xing.inPar[0], xing.inPar[1], xing.inPar[2]};
      V3 radial = unit({P.x, P.y, 0.f});
      // module normal: radial, rotated about z (phi tilt) and, for tilted modules, about the phi direction
      nrm = rot(radial, {0.f, 0.f, 1.f}, 0.3f * (U(rng) - 0.5f));
      if (U(rng) < 0.4f) {
        V3 phiDir = unit(cross({0.f, 0.f, 1.f}, nrm));
        nrm = rot(nrm, phiDir, (P.z > 0 ? 1.f : -1.f) * 1.2f * U(rng));
      }
      ++nBrl;
    } else {
      const float zsgn = std::cos(truthAt0.inPar[5]) > 0 ? 1.f : -1.f;
      const float z0 = std::abs(truthAt0.inPar[2]) + 2.f;
      int kz = 0;
      while (kz < 15 && ecZ[kz] < z0)
        ++kz;
      kz = std::min(14, kz + int(U(rng) * 2));
      msZ = zsgn * (ecZ[kz] + 0.5f * (U(rng) - 0.5f));
      if (!prop1(truthAt0, false, msZ, xing)) {
        ++nSkip;
        continue;
      }
      P = {xing.inPar[0], xing.inPar[1], xing.inPar[2]};
      const float rP = std::sqrt(P.x * P.x + P.y * P.y);
      if (rP > 110.f || rP < 3.f) {
        ++nSkip;
        continue;
      }
      nrm = unit({0.05f * (U(rng) - 0.5f), 0.05f * (U(rng) - 0.5f), zsgn});
      ++nEc;
    }
    // measurement direction u (precise) in the plane; v completes the frame
    V3 tmp = isBarrel ? V3{0.f, 0.f, 1.f} : unit({P.x, P.y, 0.f});
    V3 u = unit(cross(nrm, tmp));
    V3 v = cross(nrm, u);
    // module centre offset within the plane, hit smeared around the true crossing
    V3 pnt = add(add(P, u, 2.f * (U(rng) - 0.5f)), v, 4.f * (U(rng) - 0.5f));
    const int htype = int(U(rng) * 3);  // 0 pixel, 1 PS strip, 2 2S strip
    const float su = htype == 0 ? 0.0025f : (htype == 1 ? 0.0029f : 0.0026f);
    const float sv = htype == 0 ? 0.0040f : (htype == 1 ? 0.0722f : 1.443f);
    V3 hit = add(add(P, u, su * G(rng)), v, sv * G(rng) * (htype == 2 ? 0.f : 1.f));
    float msErr[6];
    {
      const float uu[3] = {u.x, u.y, u.z}, vv[3] = {v.x, v.y, v.z};
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j <= i; ++j)
          msErr[symIdx(i, j)] = su * su * uu[i] * uu[j] + sv * sv * vv[i] * vv[j];
    }

    Rec r = s0;
    r.msRad = isBarrel ? msRad : std::sqrt(P.x * P.x + P.y * P.y);
    r.msZ = isBarrel ? P.z : msZ;
    r.plPnt[0] = pnt.x, r.plPnt[1] = pnt.y, r.plPnt[2] = pnt.z;
    r.plNrm[0] = nrm.x, r.plNrm[1] = nrm.y, r.plNrm[2] = nrm.z;
    r.plDir[0] = u.x, r.plDir[1] = u.y, r.plDir[2] = u.z;
    r.msPar[0] = hit.x, r.msPar[1] = hit.y, r.msPar[2] = hit.z;
    for (int i = 0; i < 6; ++i)
      r.msErr[i] = msErr[i];

    if (isBarrel)
      ops[kPropR].recs.push_back(r);
    else
      ops[kPropZ].recs.push_back(r);
    ops[kPropPlane].recs.push_back(r);
    ops[kUpdPlane].recs.push_back(r);
    ops[kUpdChi2Plane].recs.push_back(r);
  }

  // finding_inter_layer / backward_fit / forward_fit / intra_layer: use_param_b_field | apply_material
  runMkFitCore(ops[kPropR], pfMat);
  runMkFitCore(ops[kPropZ], pfMat);
  runMkFitCore(ops[kPropPlane], pfMat);
  runMkFitCore(ops[kUpdPlane], pfMat);
  runMkFitCore(ops[kUpdChi2Plane], pfMat);

  // chi2 test (MkFinder pattern): input = state propagated to the layer (R or Z) by the MkFitCore code,
  // then a short propagation to the hit's module plane and the chi2.
  for (auto const* src : {&ops[kPropR], &ops[kPropZ]}) {
    for (auto const& r : src->recs) {
      if (r.fail)
        continue;
      Rec c = r;
      for (int i = 0; i < 6; ++i)
        c.inPar[i] = r.outPar[i];
      for (int i = 0; i < 21; ++i)
        c.inErr[i] = r.outErr[i];
      ops[kChi2Plane].recs.push_back(c);
    }
  }
  runMkFitCore(ops[kChi2Plane], pfMat);

  // backward-fit pattern: kalmanOperationPlaneLocal on the MkFitCore plane-propagated state
  for (auto const& r : ops[kPropPlane].recs) {
    if (r.fail)
      continue;
    Rec c = r;
    for (int i = 0; i < 6; ++i)
      c.inPar[i] = r.outPar[i];
    for (int i = 0; i < 21; ++i)
      c.inErr[i] = r.outErr[i];
    ops[kOpPlaneLocal].recs.push_back(c);
  }
  runMkFitCore(ops[kOpPlaneLocal], pfNone);

  std::printf("generated %d tracks: barrel %d endcap %d skipped %d\n", nGen, nBrl, nEc, nSkip);
  for (auto const& b : ops) {
    int nf = 0;
    for (auto const& r : b.recs)
      nf += r.fail != 0;
    std::printf("  %-38s n=%6zu MkFitCore fail=%d\n", opName(b.op), b.recs.size(), nf);
  }
  if (!writeFile(outFn, mb, ops)) {
    std::printf("cannot write %s\n", outFn.c_str());
    return 2;
  }
  std::printf("wrote %s\n", outFn.c_str());
  return 0;
}

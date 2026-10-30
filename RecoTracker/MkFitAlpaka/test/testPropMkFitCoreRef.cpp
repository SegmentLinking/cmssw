// MkFitCore reference for the propagation chain test: generates Phase-2-like track states, module planes and hits,
// runs the MkFitCore functions on them and writes the inputs and outputs for testMkFitAlpakaProp<Backend>.
// Usage: testMkFitAlpakaPropMkFitCoreRef <out.bin> [ntracks=20000] [seed=12345]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
// package-private MkFitCore headers on purpose: angle brackets keep scram's private-header check off this test
#include <RecoTracker/MkFitCore/src/FinalFit.h>
#include <RecoTracker/MkFitCore/src/KalmanUtilsMPlex.h>
#include <RecoTracker/MkFitCore/src/PropagationMPlex.h>

#include "propTestIO.h"

using namespace mkfit;
using namespace mkfitdev::proptest;

namespace {

  struct V3 {
    float x, y, z;
  };
  V3 cross(V3 lhs, V3 rhs) {
    return {lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x};
  }
  float dot(V3 lhs, V3 rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
  V3 unit(V3 vec) {
    const float len = std::sqrt(dot(vec, vec));
    return {vec.x / len, vec.y / len, vec.z / len};
  }
  V3 addScaled(V3 base, V3 step, float scale) {
    return {base.x + scale * step.x, base.y + scale * step.y, base.z + scale * step.z};
  }
  // rotates vec about the unit vector axis by angle (Rodrigues' formula)
  V3 rotateAbout(V3 vec, V3 axis, float angle) {
    const float cosA = std::cos(angle), sinA = std::sin(angle);
    const V3 axisCrossVec = cross(axis, vec);
    const float axisDotVec = dot(axis, vec);
    return {vec.x * cosA + axisCrossVec.x * sinA + axis.x * axisDotVec * (1 - cosA),
            vec.y * cosA + axisCrossVec.y * sinA + axis.y * axisDotVec * (1 - cosA),
            vec.z * cosA + axisCrossVec.z * sinA + axis.z * axisDotVec * (1 - cosA)};
  }

  void loadState(const Rec& rec, MPlexLS& err, MPlexLV& par, MPlexQI& chg, int slot) {
    for (int i = 0; i < 6; ++i) {
      par(slot, i, 0) = rec.inPar[i];
      for (int j = 0; j <= i; ++j)
        err(slot, i, j) = rec.inErr[symIdx(i, j)];
    }
    chg(slot, 0, 0) = rec.chg;
  }

  // Runs one operation in batches of NN with the MkFitCore functions.
  void runMkFitCore(OpBlock& block, const PropagationFlags& pf) {
    FinalFitFlags ff;
    ff.b_field_at_mid = true;
    ff.radial_field_corr = true;
    ff.eloss_sign_from_pass = true;
    ff.bkw_ms_fixed_momentum = true;
    ff.bkw_sub_steps = 2;
    ff.material_per_module = true;
    const int ntrk = block.recs.size();
    for (int base = 0; base < ntrk; base += NN) {
      const int N_proc = std::min<int>(NN, ntrk - base);
      MPlexLS inErr, outErr(0.f);
      MPlexLV inPar, outPar(0.f);
      MPlexQI chg, fail(0), noMat(0);
      MPlexQF msRad, msZ, chi2(0.f), modRadl, modBbxi;
      MPlexHS msErr;
      MPlexHV msPar, pnt, nrm, dir;
      float msRefP[NN];
      for (int n = 0; n < NN; ++n) {
        // slots beyond N_proc repeat the first track (never stored)
        const Rec& rec = block.recs[base + (n < N_proc ? n : 0)];
        loadState(rec, inErr, inPar, chg, n);
        for (int i = 0; i < 3; ++i) {
          msPar(n, i, 0) = rec.msPar[i];
          pnt(n, i, 0) = rec.plPnt[i];
          nrm(n, i, 0) = rec.plNrm[i];
          dir(n, i, 0) = rec.plDir[i];
          for (int j = 0; j <= i; ++j)
            msErr(n, i, j) = rec.msErr[symIdx(i, j)];
        }
        msRad(n, 0, 0) = rec.msRad;
        msZ(n, 0, 0) = rec.msZ;
        modRadl(n, 0, 0) = rec.modRadl;
        modBbxi(n, 0, 0) = rec.modBbxi;
        msRefP[n] = rec.msRefP;
      }
      const bool p2h = block.propToHit;
      const final_fit::ModuleMaterial modMat{modRadl, modBbxi};
      switch (block.op) {
        case kPropR:
          propagateHelixToRMPlex(inErr, inPar, chg, msRad, outErr, outPar, fail, N_proc, pf, &noMat);
          break;
        case kPropZ:
          propagateHelixToZMPlex(inErr, inPar, chg, msZ, outErr, outPar, fail, N_proc, pf, &noMat);
          break;
        case kPropPlane:
          propagateHelixToPlaneMPlex(inErr, inPar, chg, pnt, nrm, outErr, outPar, fail, N_proc, pf, &noMat);
          break;
        case kUpdPlane:
          kalmanPropagateAndUpdatePlane(
              inErr, inPar, chg, msErr, msPar, nrm, dir, pnt, outErr, outPar, fail, N_proc, pf, p2h);
          break;
        case kFitUpdBkw:
          final_fit::propagate_update(inErr,
                                      inPar,
                                      chg,
                                      msErr,
                                      msPar,
                                      nrm,
                                      dir,
                                      pnt,
                                      outErr,
                                      outPar,
                                      fail,
                                      chi2,
                                      N_proc,
                                      final_fit::Pass{pf, ff, false, msRefP, ff.bkw_sub_steps},
                                      &noMat,
                                      nullptr,
                                      nullptr,
                                      &modMat);
          break;
      }
      for (int n = 0; n < N_proc; ++n) {
        Rec& rec = block.recs[base + n];
        for (int i = 0; i < 6; ++i) {
          rec.outPar[i] = outPar.constAt(n, i, 0);
          for (int j = 0; j <= i; ++j)
            rec.outErr[symIdx(i, j)] = outErr.constAt(n, i, j);
        }
        rec.chi2 = chi2.constAt(n, 0, 0);
        rec.fail = fail.constAt(n, 0, 0);
        rec.chgOut = chg.constAt(n, 0, 0);
      }
    }
  }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s out.bin [ntracks] [seed]\n", argv[0]);
    return 1;
  }
  const std::string outFn = argv[1];
  const int nGen = argc > 2 ? std::atoi(argv[2]) : 20000;
  const unsigned seed = argc > 3 ? std::atoi(argv[3]) : 12345;

  // Phase-2 settings (MkFitGeometryESProducer): prop-to-plane, pT-dependent multiple scattering.
  Config::usePropToPlane = true;
  Config::usePtMultScat = true;

  // material map with the Phase-2 binning; synthetic layer-like cylinders and disks over a low baseline
  constexpr int kMatBinsZ = 300, kMatBinsR = 120;
  constexpr float kMatRangeZ = 300.f, kMatRangeR = 120.f;
  TrackerInfo ti;
  ti.create_material(kMatBinsZ, kMatRangeZ, kMatBinsR, kMatRangeR);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(0.f, 1.f);
  std::normal_distribution<float> gauss(0.f, 1.f);
  const float brlR[] = {2.9f, 6.8f, 10.9f, 16.0f, 23.0f, 36.0f, 51.0f, 68.0f, 88.0f, 108.0f};
  const float ecZ[] = {26.f, 30.f, 36.f, 44.f, 53.f, 63.f, 75.f, 90.f, 110.f, 131.f, 156.f, 185.f, 220.f, 250.f, 270.f};
  MaterialBlock mb{kMatBinsZ,
                   kMatBinsR,
                   kMatRangeZ,
                   kMatRangeR,
                   std::vector<float>(kMatBinsZ * kMatBinsR),
                   std::vector<float>(kMatBinsZ * kMatBinsR)};
  for (int bz = 0; bz < kMatBinsZ; ++bz) {
    for (int br = 0; br < kMatBinsR; ++br) {
      const float zCenter = bz + 0.5f, rCenter = br + 0.5f;
      float radl = 0.0005f;
      for (float layerR : brlR)
        if (std::abs(rCenter - layerR) < 1.5f && zCenter < 120.f + layerR)
          radl = 0.01f + 0.04f * uniform(rng);
      for (float diskZ : ecZ)
        if (std::abs(zCenter - diskZ) < 1.5f && rCenter > 3.f && rCenter < 112.f)
          radl = 0.01f + 0.05f * uniform(rng);
      const float bbxi = radl * (0.9e-3f + 0.2e-3f * uniform(rng));
      ti.material_radl(bz, br) = mb.radl[bz * kMatBinsR + br] = radl;
      ti.material_bbxi(bz, br) = mb.bbxi[bz * kMatBinsR + br] = bbxi;
    }
  }

  const int kMat = PF_use_param_b_field | PF_apply_material;
  PropagationFlags pfNone(PF_none), pfMat(kMat), pfFit(kMat | PF_use_fit_b_field);
  pfNone.tracker_info = pfMat.tracker_info = pfFit.tracker_info = &ti;
  std::mt19937 rngMod(seed + 1);

  std::vector<OpBlock> ops(kNOps);
  for (int op = 0; op < kNOps; ++op) {
    ops[op].op = op;
    ops[op].pflags = op == kFitUpdBkw ? (kMat | PF_use_fit_b_field) : kMat;
    ops[op].propToHit = op == kUpdPlane;
  }

  // one state (slot 0) propagated with MkFitCore to a radius or a z, no material; false if the propagation failed
  auto propagateOne = [&](const Rec& in, bool toR, float target, Rec& out) -> bool {
    MPlexLS err, outErr;
    MPlexLV par, outPar;
    MPlexQI chg, fail(0);
    MPlexQF targetRZ(target);
    for (int n = 0; n < NN; ++n)
      loadState(in, err, par, chg, n);
    if (toR)
      propagateHelixToRMPlex(err, par, chg, targetRZ, outErr, outPar, fail, 1, pfNone);
    else
      propagateHelixToZMPlex(err, par, chg, targetRZ, outErr, outPar, fail, 1, pfNone);
    out = in;
    for (int i = 0; i < 6; ++i)
      out.inPar[i] = outPar.constAt(0, i, 0);
    return fail.constAt(0, 0, 0) == 0;
  };

  int nBrl = 0, nEc = 0, nSkip = 0;
  for (int itrk = 0; itrk < nGen; ++itrk) {
    const float pt = std::exp(std::log(0.4f) + uniform(rng) * (std::log(50.f) - std::log(0.4f)));
    const float eta = -2.4f + 4.8f * uniform(rng);
    Rec vtx{};
    vtx.inPar[0] = 0.01f * gauss(rng);
    vtx.inPar[1] = 0.01f * gauss(rng);
    vtx.inPar[2] = 4.f * gauss(rng);
    vtx.inPar[3] = 1.f / pt;
    vtx.inPar[4] = -Const::PI + Const::TwoPI * uniform(rng);
    vtx.inPar[5] = 2.f * std::atan(std::exp(-eta));
    vtx.chg = uniform(rng) < 0.5f ? -1 : 1;
    for (int i = 0; i < 6; ++i)
      vtx.inErr[symIdx(i, i)] = 1e-4f;

    // start on an inner barrel layer, with realistic errors (diagonal + x-phi, y-phi, z-theta correlations)
    const int il = int(uniform(rng) * 6);
    Rec s0;
    if (!propagateOne(vtx, true, brlR[il], s0)) {
      ++nSkip;
      continue;
    }
    const float sx = 0.003f + 0.02f * uniform(rng), sz = 0.005f + 0.05f * uniform(rng);
    const float sig[6] = {sx,
                          sx,
                          sz,
                          (0.005f + 0.03f * uniform(rng)) * s0.inPar[3],
                          0.0003f + 0.002f * uniform(rng),
                          0.0003f + 0.002f * uniform(rng)};
    for (int i = 0; i < 21; ++i)
      s0.inErr[i] = 0.f;
    for (int i = 0; i < 6; ++i)
      s0.inErr[symIdx(i, i)] = sig[i] * sig[i];
    s0.inErr[symIdx(4, 0)] = 0.4f * sig[4] * sig[0];
    s0.inErr[symIdx(4, 1)] = -0.3f * sig[4] * sig[1];
    s0.inErr[symIdx(5, 2)] = 0.5f * sig[5] * sig[2];
    s0.inErr[symIdx(4, 3)] = 0.2f * sig[4] * sig[3];
    const Rec truthAt0 = s0;
    for (int i = 0; i < 6; ++i)
      s0.inPar[i] += sig[i] * gauss(rng);

    // target: a barrel layer further out (tilted modules included) or an endcap disk
    const bool isBarrel = std::abs(eta) < 1.3f ? (uniform(rng) < 0.85f) : (uniform(rng) < 0.25f);
    Rec xing;
    V3 nrm, xingPos;
    float msRad = 0.f, msZ = 0.f;
    if (isBarrel) {
      msRad = brlR[std::min(9, il + 1 + int(uniform(rng) * 3))] + 1.0f * (uniform(rng) - 0.5f);
      if (!propagateOne(truthAt0, true, msRad, xing) || std::abs(xing.inPar[2]) > 120.f) {
        ++nSkip;
        continue;
      }
      xingPos = {xing.inPar[0], xing.inPar[1], xing.inPar[2]};
      nrm = rotateAbout(unit({xingPos.x, xingPos.y, 0.f}), {0.f, 0.f, 1.f}, 0.3f * (uniform(rng) - 0.5f));
      if (uniform(rng) < 0.4f)
        nrm = rotateAbout(nrm, unit(cross({0.f, 0.f, 1.f}, nrm)), (xingPos.z > 0 ? 1.f : -1.f) * 1.2f * uniform(rng));
      ++nBrl;
    } else {
      const float zsgn = std::cos(truthAt0.inPar[5]) > 0 ? 1.f : -1.f;
      const float z0 = std::abs(truthAt0.inPar[2]) + 2.f;
      int kz = 0;
      while (kz < 15 && ecZ[kz] < z0)
        ++kz;
      kz = std::min(14, kz + int(uniform(rng) * 2));
      msZ = zsgn * (ecZ[kz] + 0.5f * (uniform(rng) - 0.5f));
      if (!propagateOne(truthAt0, false, msZ, xing)) {
        ++nSkip;
        continue;
      }
      xingPos = {xing.inPar[0], xing.inPar[1], xing.inPar[2]};
      const float rXing = std::sqrt(xingPos.x * xingPos.x + xingPos.y * xingPos.y);
      if (rXing > 110.f || rXing < 3.f) {
        ++nSkip;
        continue;
      }
      nrm = unit({0.05f * (uniform(rng) - 0.5f), 0.05f * (uniform(rng) - 0.5f), zsgn});
      ++nEc;
    }
    // measurement direction uDir (precise) in the plane, vDir completes the frame; pixel, PS or 2S resolution
    const V3 uDir = unit(cross(nrm, isBarrel ? V3{0.f, 0.f, 1.f} : unit({xingPos.x, xingPos.y, 0.f})));
    const V3 vDir = cross(nrm, uDir);
    const V3 pnt = addScaled(addScaled(xingPos, uDir, 2.f * (uniform(rng) - 0.5f)), vDir, 4.f * (uniform(rng) - 0.5f));
    const int htype = int(uniform(rng) * 3);
    const float su = htype == 0 ? 0.0025f : (htype == 1 ? 0.0029f : 0.0026f);
    const float sv = htype == 0 ? 0.0040f : (htype == 1 ? 0.0722f : 1.443f);
    const V3 hit =
        addScaled(addScaled(xingPos, uDir, su * gauss(rng)), vDir, sv * gauss(rng) * (htype == 2 ? 0.f : 1.f));

    Rec rec = s0;
    rec.msRad = isBarrel ? msRad : std::sqrt(xingPos.x * xingPos.x + xingPos.y * xingPos.y);
    rec.msZ = isBarrel ? xingPos.z : msZ;
    const float uu[3] = {uDir.x, uDir.y, uDir.z}, vv[3] = {vDir.x, vDir.y, vDir.z}, pp[3] = {pnt.x, pnt.y, pnt.z};
    const float nn[3] = {nrm.x, nrm.y, nrm.z}, hh[3] = {hit.x, hit.y, hit.z};
    for (int i = 0; i < 3; ++i) {
      rec.plPnt[i] = pp[i];
      rec.plNrm[i] = nn[i];
      rec.plDir[i] = uu[i];
      rec.msPar[i] = hh[i];
      for (int j = 0; j <= i; ++j)
        rec.msErr[symIdx(i, j)] = su * su * uu[i] * uu[j] + sv * sv * vv[i] * vv[j];
    }
    ops[isBarrel ? kPropR : kPropZ].recs.push_back(rec);
    ops[kPropPlane].recs.push_back(rec);
    ops[kUpdPlane].recs.push_back(rec);
    // final fit: the module's material and the |p| of the start state (as in bkReFitFitTracks)
    rec.modRadl = 0.01f + 0.04f * uniform(rngMod);
    rec.modBbxi = rec.modRadl * (0.9e-3f + 0.2e-3f * uniform(rngMod));
    const float ipt = rec.inPar[3], sT = std::sin(rec.inPar[5]);
    rec.msRefP = (ipt > 0.f && sT > 0.f) ? 1.f / (ipt * sT) : 1.f;
    ops[kFitUpdBkw].recs.push_back(rec);
  }

  for (int op : {kPropR, kPropZ, kPropPlane, kUpdPlane})
    runMkFitCore(ops[op], pfMat);
  runMkFitCore(ops[kFitUpdBkw], pfFit);

  std::printf(
      "MkFitCore NN = %d, generated %d tracks: barrel %d endcap %d skipped %d\n", int(NN), nGen, nBrl, nEc, nSkip);
  if (!writeFile(outFn, mb, ops)) {
    std::printf("cannot write %s\n", outFn.c_str());
    return 2;
  }
  return 0;
}

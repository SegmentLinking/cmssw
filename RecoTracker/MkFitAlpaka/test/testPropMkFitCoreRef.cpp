// MkFitCore reference for the propagation chain test: generates Phase-2-like track states, module planes and hits,
// runs the MkFitCore functions on them and writes the inputs and outputs for testMkFitAlpakaProp<Backend>. Some
// low-pT tracks cross their module at the turning point, where the plane path solve has two comparable roots. The v2p2
// operations run the mini propagators, the Hermite cubic and KalmanOpArgs::do_kalman_stuff of MkFinderV2p2 as its
// search calls them.
// Usage: testMkFitAlpakaPropMkFitCoreRef <out.bin> [ntracks=20000] [seed=12345]

#include <algorithm>
#include <array>
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
#include <RecoTracker/MkFitCore/src/MiniPropagators.h>
#include <RecoTracker/MkFitCore/src/MkFinderV2p2Structures.h>
#include <RecoTracker/MkFitCore/src/PropagationMPlex.h>

#include "RecoTracker/MkFitAlpaka/interface/math/Config.h"

#include "propTestIO.h"

using namespace mkfit;
using namespace mkfitdev::proptest;
namespace mp = mkfit::mini_propagators;

// the port's copies of the mini-propagator constants (interface/math/Config.h)
static_assert(mkfitdev::mini_propagators::kReachMargin == mp::kReachMargin &&
              mkfitdev::mini_propagators::kReachTolerance == mp::kReachTolerance);

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

  // the outputs of a mini-propagator operation: outPar = (x, y, z, px, py, pz), chi2 = dalpha
  void storeMini(const mp::State& st, MPlexLV& outPar, MPlexQF& chi2, MPlexQI& fail, int slot) {
    const float out[6] = {st.x, st.y, st.z, st.px, st.py, st.pz};
    for (int i = 0; i < 6; ++i)
      outPar(slot, i, 0) = out[i];
    chi2(slot, 0, 0) = st.dalpha;
    fail(slot, 0, 0) = st.fail_flag;
  }

  TrackState trackState(const Rec& rec) {
    TrackState ts;
    for (int i = 0; i < 6; ++i) {
      ts.parameters[i] = rec.inPar[i];
      for (int j = 0; j <= i; ++j)
        ts.errors(i, j) = rec.inErr[symIdx(i, j)];
    }
    ts.charge = rec.chg;
    return ts;
  }

  // MkFinderV2p2::prop_to_layer_edges + MkBins::prop_to_limits_in_order for the NN records of a batch: the start state
  // with the field per lane (paramB) or uniform, then the near and the far bounding surface of the layer
  void layerEdges(const Rec* const* recs,
                  bool paramB,
                  mp::InitialStatePlex& isp,
                  mp::StatePlex& sp1,
                  mp::StatePlex& sp2,
                  int N_proc) {
    MPlexQF phi(0.0f), bf(Config::Bfield);
    MPlexQI chg(0);
    for (int i = 0; i < NN; ++i) {
      const TrackState ts = trackState(*recs[i]);
      isp.copyIn_partial_track_state(i, ts);
      phi[i] = ts.momPhi();
      chg[i] = ts.charge;
      if (paramB)
        bf[i] = Config::bFieldFromZR(ts.z(), std::hypot(ts.x(), ts.y()));
    }
    if (paramB) {
      isp.init_momentum_vec_and_k(phi, chg, bf);
    } else {
      isp.init_momentum_vec_and_k(phi, chg);
    }
    // the slots are copied through a one-slot plex, as the finder copies its candidates
    mp::InitialStatePlex isp1, isp2 = isp;
    for (int i = 0; i < NN; ++i) {
      isp1.copyIn(0, isp, i);
      isp2.copyIn(i, isp1, 0);
    }
    MPlexQF e1, e2;
    for (int i = 0; i < NN; ++i) {
      e1[i] = recs[i]->edge[0];
      e2[i] = recs[i]->edge[1];
    }
    // the records of a batch are all barrel or all endcap
    if (recs[0]->barrel) {
      isp2.propagate_to_r(mp::PA_Exact, e1, sp1, true, N_proc);
      isp2 = sp1;
      isp2.propagate_to_r(mp::PA_Exact, e2, sp2, true, N_proc);
    } else {
      isp2.propagate_to_z(mp::PA_Exact, e1, sp1, true, N_proc);
      isp2 = sp1;
      isp2.propagate_to_z(mp::PA_Exact, e2, sp2, true, N_proc);
    }
    isp = isp2;
  }

  // the Hermite cubic through the two crossings, copied slot by slot through a one-slot cubic as the finder does
  mp::Hermite3D hermite(const mp::StatePlex& sp1, const mp::StatePlex& sp2, const MPlexQF& inv_k) {
    mp::Hermite3D H, H1, h3d(0.0f);
    H.calculate_coeffs(sp1, sp2, inv_k);
    for (int i = 0; i < NN; ++i) {
      H1.copyIn(0, H, i);
      h3d.copyIn(i, H1, 0);
    }
    return h3d;
  }

  // both layer crossings of the v2p2 search fail (MkFinderV2p2::determine_wsr: WSR_Outside)
  bool missesLayer(const Rec& rec) {
    const Rec* recs[NN];
    for (int i = 0; i < NN; ++i)
      recs[i] = &rec;
    mp::InitialStatePlex isp;
    mp::StatePlex sp1, sp2;
    layerEdges(recs, true, isp, sp1, sp2, 1);
    return sp1.fail_flag[0] != 0 && sp2.fail_flag[0] != 0;
  }

  // Runs one operation in batches of NN with the MkFitCore functions.
  void runMkFitCore(OpBlock& block, const PropagationFlags& pf) {
    FinalFitFlags ff;
    ff.b_field_at_mid = true;
    ff.radial_field_corr = true;
    ff.eloss_sign_from_pass = true;
    ff.bkw_ms_fixed_momentum = true;
    ff.bkw_sub_steps = kBkwSubSteps;
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
      if (block.op == kPropPlaneNoMat)
        noMat.setVal(1);
      switch (block.op) {
        case kPropR:
          propagateHelixToRMPlex(inErr, inPar, chg, msRad, outErr, outPar, fail, N_proc, pf, &noMat);
          break;
        case kPropZ:
          propagateHelixToZMPlex(inErr, inPar, chg, msZ, outErr, outPar, fail, N_proc, pf, &noMat);
          break;
        case kPropPlane:
        case kPropPlaneNoMat:
          propagateHelixToPlaneMPlex(inErr, inPar, chg, pnt, nrm, outErr, outPar, fail, N_proc, pf, &noMat);
          break;
        case kUpdPlane:
        case kUpdPlaneNoProp:
          kalmanPropagateAndUpdatePlane(
              inErr, inPar, chg, msErr, msPar, nrm, dir, pnt, outErr, outPar, fail, N_proc, pf, p2h);
          break;
        case kChi2Plane:
          kalmanPropagateAndComputeChi2Plane(
              inErr, inPar, chg, msErr, msPar, nrm, dir, pnt, chi2, outPar, fail, N_proc, pf, p2h);
          break;
        case kFitUpdFwd:
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
                                      final_fit::Pass{pf, ff, true},
                                      &noMat,
                                      nullptr,
                                      nullptr,
                                      &modMat);
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
        case kMiniPlexR:
        case kMiniPlexZ: {
          const mp::InitialStatePlex isp(inPar, chg);
          mp::StatePlex sp;
          if (block.op == kMiniPlexR)
            isp.propagate_to_r(mp::PA_Exact, msRad, sp, true, N_proc);
          else
            isp.propagate_to_z(mp::PA_Exact, msZ, sp, true, N_proc);
          for (int n = 0; n < NN; ++n)
            storeMini(mp::State(sp, n), outPar, chi2, fail, n);
          break;
        }
        case kMiniR:
        case kMiniPlane:
          for (int n = 0; n < N_proc; ++n) {
            mp::InitialState is(inPar, chg, n);
            is.dalpha = 0.f;  // State(par, ti) leaves the turning angle indeterminate
            is.fail_flag = 0;
            mp::State st;
            if (block.op == kMiniR)
              is.propagate_to_r(mp::PA_Exact, msRad.constAt(n, 0, 0), st, true);
            else
              is.propagate_to_plane(mp::PA_Line,
                                    SVector3(pnt(n, 0, 0), pnt(n, 1, 0), pnt(n, 2, 0)),
                                    SVector3(nrm(n, 0, 0), nrm(n, 1, 0), nrm(n, 2, 0)),
                                    st,
                                    true);
            storeMini(st, outPar, chi2, fail, n);
          }
          break;
      }
      // v2p2: outputs that are not a track state go to outErr as a flat list
      MPlexQF detV(0.f), chi2Acc(0.f);
      const Rec* recs[NN];
      for (int n = 0; n < NN; ++n)
        recs[n] = &block.recs[base + (n < N_proc ? n : 0)];
      std::vector<std::array<float, 21>> extras(NN, std::array<float, 21>{});
      switch (block.op) {
        case kV2p2Edges:
        case kHermite:
        case kHermitePlane: {
          mp::InitialStatePlex isp;
          mp::StatePlex sp1, sp2;
          layerEdges(recs, block.op != kHermite, isp, sp1, sp2, N_proc);
          if (block.op == kV2p2Edges) {
            for (int n = 0; n < NN; ++n) {
              storeMini(mp::State(sp2, n), outPar, chi2, fail, n);
              fail(n, 0, 0) = 3 * sp1.fail_flag[n] + sp2.fail_flag[n];
              const float near[7] = {sp1.x[n], sp1.y[n], sp1.z[n], sp1.px[n], sp1.py[n], sp1.pz[n], sp1.dalpha[n]};
              std::copy(near, near + 7, extras[n].begin());
            }
            break;
          }
          const mp::Hermite3D h3d = hermite(sp1, sp2, isp.inv_k);
          if (block.op == kHermite) {
            MPlexQF t, x, y, z, dx, dy, dz, x3, y3, z3;
            for (int n = 0; n < NN; ++n)
              t[n] = recs[n]->hermT;
            h3d.evaluate(t, x, y, z, dx, dy, dz);
            h3d.evaluate(t, x3, y3, z3);
            for (int n = 0; n < NN; ++n) {
              storeMini(mp::State(x[n], y[n], z[n], dx[n], dy[n], dz[n], 0.f, 0), outPar, chi2, fail, n);
              for (int k = 0; k < 4; ++k) {
                extras[n][k] = h3d.m_Hx.constAt(n, k, 0);
                extras[n][4 + k] = h3d.m_Hy.constAt(n, k, 0);
                extras[n][8 + k] = h3d.m_Hz.constAt(n, k, 0);
              }
              extras[n][12] = h3d.m_Hderfac[n];
              extras[n][13] = x3[n];
              extras[n][14] = y3[n];
              extras[n][15] = z3[n];
            }
            break;
          }
          mp::Hermite3DOnPlane h3dop;
          h3dop.init_coeffs(h3d, pnt, nrm);
          const MPlexQF t0 = h3dop.m_T;
          h3dop.solve();
          MPlexQF d, dddt, d2;
          h3dop.evaluate(h3dop.m_T, d, dddt);
          h3dop.evaluate(h3dop.m_T, d2);
          mp::StatePlex h3_state;
          h3d.evaluate(h3dop.m_T, h3_state);
          for (int n = 0; n < NN; ++n) {
            storeMini(mp::State(h3_state, n), outPar, chi2, fail, n);
            chi2(n, 0, 0) = h3dop.m_T[n];
            for (int k = 0; k < 4; ++k)
              extras[n][k] = h3dop.m_D.constAt(n, k, 0);
            extras[n][4] = t0[n];
            extras[n][5] = d[n];
            extras[n][6] = dddt[n];
            extras[n][7] = d2[n];
          }
          break;
        }
        case kKalmanD0:
        case kKalmanD1: {
          const bool deep = block.op == kKalmanD1;
          PropagationConfig pc;
          pc.finding_inter_layer_pflags = pc.finding_intra_layer_pflags = pf;
          std::vector<KalmanOpArgs::ItemOut> out;
          KalmanOpArgs koa;
          koa.prop_config = &pc;
          koa.mp_out = &out;
          koa.m_solve_plane = deep;
          koa.m_chi2_trk_fac = kV2p2Chi2TrkFac;
          if (!deep) {
            // the search's crossing on the module plane, as MkFinderV2p2::preselect_hit_batch makes it
            mp::InitialStatePlex isp;
            mp::StatePlex sp1, sp2;
            layerEdges(recs, true, isp, sp1, sp2, N_proc);
            const mp::Hermite3D h3d = hermite(sp1, sp2, isp.inv_k);
            mp::Hermite3DOnPlane h3dop;
            h3dop.init_coeffs(h3d, pnt, nrm);
            h3dop.solve();
            mp::StatePlex h3_state;
            h3d.evaluate(h3dop.m_T, h3_state);
            for (int n = 0; n < N_proc; ++n) {
              h3_state.dalpha[n] = sp1.dalpha[n] + h3dop.m_T[n] * (sp2.dalpha[n] - sp1.dalpha[n]);
              const mp::InitialState is(h3_state, n, isp, n);
              const float xyz[10] = {is.x, is.y, is.z, is.px, is.py, is.pz, is.dalpha, is.inv_pt, is.inv_k, is.theta};
              std::copy(xyz, xyz + 10, block.recs[base + n].xyz);
            }
          }
          for (int n = 0; n < N_proc; ++n) {
            const Rec& rec = block.recs[base + n];
            koa.item_begin(nullptr, HitOnTrack(n, 0), deep ? n : -1, n);
            if (deep) {
              koa.load_state_err_chg(trackState(rec));
            } else {
              mp::InitialState is;
              is.x = rec.xyz[0];
              is.y = rec.xyz[1];
              is.z = rec.xyz[2];
              is.px = rec.xyz[3];
              is.py = rec.xyz[4];
              is.pz = rec.xyz[5];
              is.dalpha = rec.xyz[6];
              is.inv_pt = rec.xyz[7];
              is.inv_k = rec.xyz[8];
              is.theta = rec.xyz[9];
              koa.load_state_err_chg(is, trackState(rec));
            }
            for (int i = 0; i < 3; ++i) {
              koa.msPar(n, i, 0) = rec.msPar[i];
              koa.plPnt(n, i, 0) = rec.plPnt[i];
              koa.plNrm(n, i, 0) = rec.plNrm[i];
              koa.plDir(n, i, 0) = rec.plDir[i];
              for (int j = 0; j <= i; ++j)
                koa.msErr(n, i, j) = rec.msErr[symIdx(i, j)];
            }
            koa.item_finished();
          }
          // MkFinderV2p2::rk_flush
          if (!deep)
            koa.compute_pars();
          koa.do_kalman_stuff();
          for (const auto& o : out) {
            const int n = o.hit_pos;
            for (int i = 0; i < 6; ++i) {
              outPar(n, i, 0) = o.state.parameters[i];
              for (int j = 0; j <= i; ++j)
                outErr(n, i, j) = o.state.errors(i, j);
            }
            chi2(n, 0, 0) = o.chi2;
            detV(n, 0, 0) = o.det_v;
            chi2Acc(n, 0, 0) = o.chi2_acc;
            chg(n, 0, 0) = o.state.charge;
          }
          break;
        }
      }
      for (int n = 0; n < N_proc; ++n) {
        Rec& rec = block.recs[base + n];
        for (int i = 0; i < 6; ++i) {
          rec.outPar[i] = outPar.constAt(n, i, 0);
          for (int j = 0; j <= i; ++j)
            rec.outErr[symIdx(i, j)] = outErr.constAt(n, i, j);
        }
        if (block.op == kV2p2Edges || block.op == kHermite || block.op == kHermitePlane)
          std::copy(extras[n].begin(), extras[n].end(), rec.outErr);
        rec.chi2 = chi2.constAt(n, 0, 0);
        rec.fail = fail.constAt(n, 0, 0);
        rec.chgOut = chg.constAt(n, 0, 0);
        rec.detV = detV.constAt(n, 0, 0);
        rec.chi2Acc = chi2Acc.constAt(n, 0, 0);
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
  // the turning-point tracks draw from their own stream, so the other tracks stay the same
  std::mt19937 rngTurn(seed + 2);
  // and so do the v2p2 inputs
  std::mt19937 rngV2p2(seed + 3);
  // half thickness of the layer whose bounding surfaces the v2p2 search crosses
  constexpr float kLayerHalfThickness = 0.6f;
  // fraction of the low-pT barrel tracks that cross their module at the turning point
  constexpr float kTurningFraction = 0.1f;

  std::vector<OpBlock> ops(kNOps), opsEc(kNOps);
  for (int op = 0; op < kNOps; ++op) {
    ops[op].op = opsEc[op].op = op;
    ops[op].pflags = opsEc[op].pflags = (op == kFitUpdBkw || op == kFitUpdFwd) ? (kMat | PF_use_fit_b_field) : kMat;
    ops[op].propToHit = opsEc[op].propToHit = op == kUpdPlane || op == kChi2Plane;
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

  int nBrl = 0, nEc = 0, nSkip = 0, nTurn = 0, nMissV2p2 = 0;
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
    bool turning = false;
    if (isBarrel) {
      msRad = brlR[std::min(9, il + 1 + int(uniform(rng) * 3))] + 1.0f * (uniform(rng) - 0.5f);
      std::uniform_real_distribution<float> uniformTurn(0.f, 1.f);
      if (pt < 1.f && uniformTurn(rngTurn) < kTurningFraction) {
        // just inside the largest radius of the trajectory, d_c + R_c (mini propagators), where the momentum is
        // almost tangent to the layer and so almost in the module plane
        MPlexLS err;
        MPlexLV par;
        MPlexQI chg;
        loadState(truthAt0, err, par, chg, 0);
        mp::InitialState is(par, chg, 0);
        is.dalpha = 0.f;
        const float k = 1.f / is.inv_k;
        const float rMax = hipo(is.x - k * is.py, is.y + k * is.px) + std::abs(k) * hipo(is.px, is.py);
        const float rTurn = rMax - 0.05f - 0.5f * uniformTurn(rngTurn);
        mp::State st;
        if (rTurn > hipo(is.x, is.y) + 1.f && !is.propagate_to_r(mp::PA_Exact, rTurn, st, true)) {
          turning = true;
          msRad = rTurn;
          xing.inPar[0] = st.x;
          xing.inPar[1] = st.y;
          xing.inPar[2] = st.z;
        }
      }
      if ((!turning && !propagateOne(truthAt0, true, msRad, xing)) || std::abs(xing.inPar[2]) > 120.f) {
        ++nSkip;
        continue;
      }
      nTurn += turning;
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
    // the iterative propagation to a radius fails near the turning point: the turning tracks go to the planes only
    if (!turning)
      ops[isBarrel ? kPropR : kPropZ].recs.push_back(rec);
    for (int op : {kPropPlane, kUpdPlane, kChi2Plane, kUpdPlaneNoProp, kPropPlaneNoMat})
      ops[op].recs.push_back(rec);
    if (isBarrel) {
      for (int op : {kMiniPlexR, kMiniR, kMiniPlane})
        ops[op].recs.push_back(rec);
    } else {
      ops[kMiniPlexZ].recs.push_back(rec);
    }
    // v2p2: the bounding surfaces around the layer of the module, in search (outward) order; the barrel and the endcap
    // records go to separate operations so that a batch never mixes them
    rec.barrel = isBarrel;
    if (isBarrel) {
      rec.edge[0] = msRad - kLayerHalfThickness;
      rec.edge[1] = msRad + kLayerHalfThickness;
    } else {
      const float zsgn = msZ > 0.f ? 1.f : -1.f;
      rec.edge[0] = msZ - zsgn * kLayerHalfThickness;
      rec.edge[1] = msZ + zsgn * kLayerHalfThickness;
    }
    rec.hermT = std::uniform_real_distribution<float>(0.f, 1.f)(rngV2p2);
    for (int op : {kV2p2Edges, kKalmanD1})
      (isBarrel ? ops : opsEc)[op].recs.push_back(rec);
    // a track that crosses neither bounding surface is WSR_Outside (MkFinderV2p2::determine_wsr): no Hermite cubic
    if (!missesLayer(rec)) {
      for (int op : {kHermite, kHermitePlane, kKalmanD0})
        (isBarrel ? ops : opsEc)[op].recs.push_back(rec);
    } else {
      ++nMissV2p2;
    }

    // final fit: the module's material and the |p| of the start state (as in bkReFitFitTracks)
    rec.modRadl = 0.01f + 0.04f * uniform(rngMod);
    rec.modBbxi = rec.modRadl * (0.9e-3f + 0.2e-3f * uniform(rngMod));
    const float ipt = rec.inPar[3], sT = std::sin(rec.inPar[5]);
    rec.msRefP = (ipt > 0.f && sT > 0.f) ? 1.f / (ipt * sT) : 1.f;
    ops[kFitUpdBkw].recs.push_back(rec);
    rec.msRefP = 0.f;  // the forward pass has no fixed-momentum multiple scattering
    ops[kFitUpdFwd].recs.push_back(rec);
  }

  for (int op : {kPropR,
                 kPropZ,
                 kPropPlane,
                 kUpdPlane,
                 kChi2Plane,
                 kUpdPlaneNoProp,
                 kPropPlaneNoMat,
                 kMiniPlexR,
                 kMiniPlexZ,
                 kMiniR,
                 kMiniPlane})
    runMkFitCore(ops[op], pfMat);
  for (int op : {kFitUpdBkw, kFitUpdFwd})
    runMkFitCore(ops[op], pfFit);
  // v2p2: the finding flags inter- and intra-layer (both use_param_b_field | apply_material in Phase 2); the endcap
  // records as blocks of their own
  for (int op : {kV2p2Edges, kHermite, kHermitePlane, kKalmanD0, kKalmanD1}) {
    runMkFitCore(ops[op], pfMat);
    runMkFitCore(opsEc[op], pfMat);
    ops.push_back(std::move(opsEc[op]));
  }

  std::printf(
      "MkFitCore NN = %d, generated %d tracks: barrel %d (at the turning point %d) endcap %d skipped %d; v2p2: %d miss "
      "the layer\n",
      int(NN),
      nGen,
      nBrl,
      nTurn,
      nEc,
      nSkip,
      nMissV2p2);
  if (!writeFile(outFn, mb, ops)) {
    std::printf("cannot write %s\n", outFn.c_str());
    return 2;
  }
  return 0;
}

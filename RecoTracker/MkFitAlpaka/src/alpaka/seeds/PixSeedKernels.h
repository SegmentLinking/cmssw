#ifndef RecoTracker_MkFitAlpaka_src_alpaka_seeds_PixSeedKernels_h
#define RecoTracker_MkFitAlpaka_src_alpaka_seeds_PixSeedKernels_h

// the pLS seed state on the device.
//   KernelPixSeedFit : SeedFromConsecutiveHitsCreator::makeSeed of the menu's pixel-seed producer
//                      (SeedGeneratorFromProtoTracksEDProducer) = fitHostLike (LstSeedKernels.h, the DEV-2 host-creator
//                      emulation) with the proto track's region (its vertex, its pT, r 0.2 cm, half-length 0.2 cm);
//                      state on the last hit. One thread per pixel track, block 64 (as KernelLstSeedFit).
//   KernelPixSeedPca : LSTInputProducer's host part, step 1: TSCBLBuilderNoMaterial of that state (interface/math/
//                      PcaToBeamLine.h): PCA state + its 5x5 curvilinear error to scratch.
//   KernelPixSeedPerigee : step 2: PerigeeConversions::ftsToPerigeeError rows 0-1 and the reco::TrackBase ptError /
//                      etaError formulas. Separate kernels keep the double PCA out of the fit kernel's registers and
//                      the two PCA steps spill-free on sm_89.

#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/math/PcaToBeamLine.h"
#include "RecoTracker/MkFitAlpaka/interface/math/TkBfield.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/seeds/LstSeedKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstseeds {

  using ::mkfitdev::lstseeds::PixSeedBeam;
  using ::mkfitdev::lstseeds::PixSeedIn;
  using ::mkfitdev::lstseeds::PixSeedOut;

  // the pixel-seed state after its first fit (status st): the capped-prior retry, then the output row
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void pixSeedStore(Acc1D const& acc,
                                                   ESView const& es,
                                                   HitSoAConstView hits,
                                                   uint32_t nPixel,
                                                   const ::mkfitdev::HitOnTrack* hot,
                                                   const int nh,
                                                   const double vx,
                                                   const double vy,
                                                   LstSeedFitConfig const& cfg,
                                                   int st,
                                                   MPlexLV<1>& parH,
                                                   MPlexLS<1>& errH,
                                                   MPlexQI<1>& chgH,
                                                   PixSeedOut& o) {
    if (st == kHLRejInvalid)  // float failure of the exact prior: the capped prior (as KernelLstSeedFit)
      st = fitHostLike(acc, es, hits, nPixel, hot, nh, cfg, true, parH, errH, chgH, vx, vy);
    o.status = st;  // LstHostLikeStatus == PixSeedStatus for 0..3
    if (st != kHLOk)
      return;
    o.charge = chgH.At(0, 0, 0);
    for (int k = 0; k < 6; ++k)
      o.par[k] = parH.At(0, k, 0);
    errH.copyOut(0, o.err);
  }

  class KernelPixSeedFit {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ESView es,
                                  HitSoAConstView hits,
                                  PixSeedIn const* in,
                                  PixSeedOut* out,
                                  int32_t n,
                                  LstSeedFitConfig cfg) const {
      const uint32_t nPixel = hits.nPixel();
      for (int32_t t : cms::alpakatools::uniform_elements(acc, n)) {
        PixSeedOut& o = out[t];
        o.pcaStatus = ::mkfitdev::pca::kPcaFailed;
        const int nh = in[t].nh;
        if (nh < 2) {
          o.status = ::mkfitdev::lstseeds::kPixNotFitted;
          continue;
        }
        // GlobalTrackingRegion(ptMin = proto pT, origin = proto vertex, originRadius 0.2, originHalfLength 0.2)
        cfg.hlPtMin = in[t].ptMin;
        cfg.originR2 = 0.04f;
        cfg.originZ2 = 0.04f;
        MPlexLV<1> parH;
        MPlexLS<1> errH;
        MPlexQI<1> chgH;
        const double vx = in[t].vx, vy = in[t].vy;
        int st = fitHostLike(acc, es, hits, nPixel, in[t].hot, nh, cfg, false, parH, errH, chgH, vx, vy);
        pixSeedStore(acc, es, hits, nPixel, in[t].hot, nh, vx, vy, cfg, st, parH, errH, chgH, o);
      }
    }
  };

  constexpr int kPixPcaCov = 25;  // doubles of scratch per pixel track between the two PCA kernels

  class KernelPixSeedPca {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, PixSeedOut* out, double* cov, int32_t n, PixSeedBeam beam) const {
      namespace pca = ::mkfitdev::pca;
      for (int32_t t : cms::alpakatools::uniform_elements(acc, n)) {
        PixSeedOut& o = out[t];
        if (o.status != ::mkfitdev::lstseeds::kPixOk)
          continue;
        pca::TrackIn ti;
        pca::ccsToCurvilinear(o.par, o.err, o.charge, ti);
        const pca::BeamIn bl{pca::F3{beam.x, beam.y, beam.z}, pca::F3{beam.dxdz, beam.dydz, 1.f}};
        pca::PcaOut po;
        o.pcaStatus = pca::pcaFromFirstHitState(ti, bl, po);
        if (o.pcaStatus != pca::kPcaOk)
          continue;
        o.pcaX = po.x.x;
        o.pcaY = po.x.y;
        o.pcaZ = po.x.z;
        o.pcaPx = po.p.x;
        o.pcaPy = po.p.y;
        o.pcaPz = po.p.z;
        double* c = cov + kPixPcaCov * t;  // the full symmetric matrix, as similarity5 wrote it
        for (int a = 0; a < 5; ++a)
          for (int b = 0; b < 5; ++b)
            c[5 * a + b] = po.C[a][b];
      }
    }
  };

  class KernelPixSeedPerigee {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, PixSeedOut* out, double const* cov, int32_t n) const {
      namespace pca = ::mkfitdev::pca;
      for (int32_t t : cms::alpakatools::uniform_elements(acc, n)) {
        PixSeedOut& o = out[t];
        if (o.status != ::mkfitdev::lstseeds::kPixOk || o.pcaStatus != pca::kPcaOk)
          continue;
        double const* c = cov + kPixPcaCov * t;
        const pca::F3 poX{o.pcaX, o.pcaY, o.pcaZ}, poP{o.pcaPx, o.pcaPy, o.pcaPz};  // the PCA state (float)
        // PerigeeConversions::jacobianCurvilinear2Perigee rows 0 (transverse curvature) and 1 (theta) on the TSCBL
        // state; the field is the GlobalTrajectoryParameters' cached field at the PCA point, in 1/GeV
        float bxT, byT, bzT;
        ::mkfitdev::field::tkBfield(poX.x, poX.y, poX.z, bxT, byT, bzT);
        const double Bx = 2.99792458e-3f * bxT, By = 2.99792458e-3f * byT, Bz = 2.99792458e-3f * bzT;
        const double px = poP.x, py = poP.y, pz = poP.z;
        const double pt2 = px * px + py * py, pmag = std::sqrt(pt2 + pz * pz), pt = std::sqrt(pt2);
        const double Tx = px / pmag, Ty = py / pmag, Tz = pz / pmag;
        double Ux = -Ty, Uy = Tx;  // Z x T, unit
        {
          const double un = std::sqrt(Ux * Ux + Uy * Uy);
          Ux /= un;
          Uy /= un;
        }
        const double Vx = Ty * 0. - Tz * Uy, Vy = Tz * Ux - Tx * 0., Vz = Tx * Uy - Ty * Ux;  // T x U
        const double Ix = -px / pt, Iy = -py / pt;                                            // I = (-p_T).unit
        const double Bm = std::sqrt(Bx * Bx + By * By + Bz * Bz);
        const double Hx = Bx / Bm, Hy = By / Bm, Hz = Bz / Bm;
        const double HTx = Hy * Tz - Hz * Ty, HTy = Hz * Tx - Hx * Tz, HTz = Hx * Ty - Hy * Tx;
        const double alpha = std::sqrt(HTx * HTx + HTy * HTy + HTz * HTz);
        const double Nx = HTx / alpha, Ny = HTy / alpha, Nz = HTz / alpha;
        const double qbp = double(o.charge) / pmag;
        const double alphaQ = alpha * (-Bm * qbp);
        const double lambda = 0.5 * M_PI - std::atan2(pt, pz);
        double sinl, cosl;
        ::mkfitdev::vdt::fast_sincos(lambda, sinl, cosl);
        const double secl = 1. / cosl;
        const double ITI = 1. / (Tx * Ix + Ty * Iy);
        const double NV = Nx * Vx + Ny * Vy + Nz * Vz;
        const double UI = Ux * Ix + Uy * Iy, VI = Vx * Ix + Vy * Iy;
        double J0[5] = {0., 0., 0., 0., 0.}, J1[5] = {0., -1., 0., 0., 0.};
        const double tc = -2.99792458e-3 * o.charge / pt * bzT;  // GlobalTrajectoryParameters::transverseCurvature
        if (std::fabs(tc) < 1.e-10) {
          J0[0] = secl;
          J0[1] = sinl * secl * secl * std::fabs(qbp);
        } else {
          J0[0] = -Bz * secl;
          J0[1] = -Bz * sinl * secl * secl * qbp;
          J1[3] = alphaQ * NV * UI * ITI;
          J1[4] = alphaQ * NV * VI * ITI;
          J0[3] = -J0[1] * J1[3];
          J0[4] = -J0[1] * J1[4];
        }
        double E00 = 0., E01 = 0., E11 = 0.;
        for (int a = 0; a < 5; ++a)
          for (int b = 0; b < 5; ++b) {
            E00 += J0[a] * c[5 * a + b] * J0[b];
            E01 += J0[a] * c[5 * a + b] * J1[b];
            E11 += J1[a] * c[5 * a + b] * J1[b];
          }
        // LSTInputProducer: reco::TrackBase::ptError2 / etaError on (pt, p, pz, q) of the PCA momentum
        const double ptF = double(pca::perp(poP)), pF = double(pca::mag(poP)), pzF = double(poP.z);
        const double q = double(o.charge);
        const double v = ptF * ptF * pF * pF / (q * q) * E00 + 2.0 * std::sqrt(pF * pF * ptF * ptF) / q * pzF * E01 +
                         pzF * pzF * E11;
        o.ptErr = std::sqrt(v);
        o.etaErr = std::sqrt(E11) * pF / ptF;
      }
    }
  };

#if !(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED))
  // GlobalTrackingRegion of a pixel track (as KernelPixSeedFit): ptMin = proto pT, origin radius / half-length 0.2 cm
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void pixSeedRegion(LstSeedFitConfig& cfg, PixSeedIn const& in) {
    cfg.hlPtMin = in.ptMin;
    cfg.originR2 = 0.04f;
    cfg.originZ2 = 0.04f;
  }

  // CPU: the pixel tracks of a block grouped by hit count into N-wide fits (KernelPixSeedFit per track)
  template <idx_t N>
  class KernelPixSeedFitGrouped {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ESView es,
                                  HitSoAConstView hits,
                                  PixSeedIn const* in,
                                  PixSeedOut* out,
                                  int32_t n,
                                  LstSeedFitConfig cfg) const {
      const uint32_t nPixel = hits.nPixel();
      int32_t group[::mkfitdev::lstseeds::kMaxPixSeedHits + 1][N];
      int fill[::mkfitdev::lstseeds::kMaxPixSeedHits + 1] = {};
      auto fitGroup = [&](const int nh) {
        const int nl = fill[nh];
        const ::mkfitdev::HitOnTrack* hot[N];
        LstSeedFitConfig cfgs[N];
        double vx[N], vy[N];
        for (int i = 0; i < N; ++i) {
          const PixSeedIn& ti = in[group[nh][i < nl ? i : 0]];
          hot[i] = ti.hot;
          cfgs[i] = cfg;
          pixSeedRegion(cfgs[i], ti);
          vx[i] = ti.vx;
          vy[i] = ti.vy;
        }
        MPlexLV<N> parG;
        MPlexLS<N> errG;
        MPlexQI<N> chgG;
        int st[N];
        fitHostLikeGroup<N>(acc, es, hits, nPixel, hot, nh, nl, cfgs, vx, vy, parG, errG, chgG, st);
        for (int i = 0; i < nl; ++i) {
          MPlexLV<1> parH;
          MPlexLS<1> errH;
          MPlexQI<1> chgH;
          hostLikeSlot(parG, errG, chgG, i, parH, errH, chgH);
          const int32_t t = group[nh][i];
          pixSeedStore(acc, es, hits, nPixel, hot[i], nh, vx[i], vy[i], cfgs[i], st[i], parH, errH, chgH, out[t]);
        }
        fill[nh] = 0;
      };
      for (int32_t t : cms::alpakatools::uniform_elements(acc, n)) {
        PixSeedOut& o = out[t];
        o.pcaStatus = ::mkfitdev::pca::kPcaFailed;
        const int nh = in[t].nh;
        if (nh < 2) {
          o.status = ::mkfitdev::lstseeds::kPixNotFitted;
          continue;
        }
        if (hostLikeMissingHit(in[t].hot, nh)) {
          LstSeedFitConfig cfgT = cfg;
          pixSeedRegion(cfgT, in[t]);
          MPlexLV<1> parH;
          MPlexLS<1> errH;
          MPlexQI<1> chgH;
          const double vx = in[t].vx, vy = in[t].vy;
          const int st = fitHostLike(acc, es, hits, nPixel, in[t].hot, nh, cfgT, false, parH, errH, chgH, vx, vy);
          pixSeedStore(acc, es, hits, nPixel, in[t].hot, nh, vx, vy, cfgT, st, parH, errH, chgH, o);
          continue;
        }
        group[nh][fill[nh]] = t;
        if (++fill[nh] == N)
          fitGroup(nh);
      }
      for (int nh = 0; nh <= ::mkfitdev::lstseeds::kMaxPixSeedHits; ++nh)
        if (fill[nh] > 0)
          fitGroup(nh);
    }
  };
#endif

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstseeds

#endif

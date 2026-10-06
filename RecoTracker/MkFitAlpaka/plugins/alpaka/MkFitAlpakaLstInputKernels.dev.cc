// Kernels of the device LST input: see MkFitAlpakaLstInputKernels.h.
#include <Eigen/Core>  // before any SoA header (Eigen columns of TracksSoA)
#include <cmath>
#include <numbers>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/LSTCore/interface/Common.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/DeviceHitInput.h"
#include "RecoTracker/MkFitAlpaka/interface/math/TkBfield.h"

#include "MkFitAlpakaLstInputKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstin {

  using namespace cms::alpakatools;
  using ::mkfitdev::lstin::kMaxTrackHits;
  using ::mkfitdev::lstin::kNoKey;

  namespace {
    constexpr uint32_t kChunk = 256;  // tracks per scan chunk

    // ROOT::Math::VectorUtil::DeltaPhi(v1, v2) = phi(v2) - phi(v1), wrapped to (-pi, pi]
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float deltaPhiRoot(float phi1, float phi2) {
      float d = phi2 - phi1;
      if (d > float(M_PI))
        d -= float(2.0 * M_PI);
      else if (d <= -float(M_PI))
        d += float(2.0 * M_PI);
      return d;
    }

    // ROOT Eta_FromRhoZ for rho > 0
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float etaRoot(float rho, float z) {
      const float zs = z / rho;
      return std::log(zs + std::sqrt(zs * zs + 1.f));
    }

    struct KernelPLSCompute {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::reco::TrackBlocksConstView tracks,
                                    uint32_t maxTracks,
                                    PixelTrackHits hits,
                                    uint32_t const* pixKey,
                                    uint32_t const* otKey,
                                    uint32_t const* otDetId,
                                    uint16_t const* otClust,
                                    int32_t const* seedOfTrack,
                                    Params p,
                                    PLS* scratch,
                                    uint32_t* counts,
                                    ::mkfitdev::lstseeds::PixSeedOut const* kf) const {
        auto const tk = tracks.tracks();
        auto const th = tracks.trackHits();
        const int nT = tk.nTracks();
        if (once_per_grid(acc))
          counts[2] = nT;
        for (uint32_t t : uniform_elements(acc, maxTracks)) {
          PLS& o = scratch[t];
          o.pass = 0;
          o.nToSoA = 0;
          if (int(t) >= nT)
            continue;
          if (static_cast<int>(tk[t].quality()) < p.minQuality)
            continue;
          int32_t seedIdx = int32_t(t);
          if (p.nSeedMap > 0) {  // only tracks with a host seed (the converter needs it)
            seedIdx = t < p.nSeedMap ? seedOfTrack[t] : -1;
            if (seedIdx < 0)
              continue;
          }
          const uint32_t start = t == 0 ? 0 : tk[t - 1].hitOffsets();
          const uint32_t end = tk[t].hitOffsets();
          const uint32_t nh = end - start;
          if (nh > uint32_t(kMaxTrackHits) || nh < 3) {
            alpaka::atomicAdd(acc, &counts[3], 1u, alpaka::hierarchy::Blocks{});
            continue;
          }
          const auto st = tk[t].state();
          const auto cv = tk[t].covariance();
          const float phi0 = st(0), tip = st(1), qpt = st(2), cot = st(3), zip = st(4);
          if (!(std::isfinite(phi0) && std::isfinite(tip) && std::isfinite(qpt) && std::isfinite(cot) &&
                std::isfinite(zip) && qpt != 0.f)) {
            alpaka::atomicAdd(acc, &counts[5], 1u, alpaka::hierarchy::Blocks{});
            continue;
          }
          int q = qpt > 0.f ? 1 : -1;
          // pixKF: the device creator (fitPixelSeeds) decides; a failure gives no pLS (the host seeding: no seed, no pLS)
          if (kf != nullptr && kf[t].status != ::mkfitdev::lstseeds::kPixOk) {
            alpaka::atomicAdd(acc, &counts[6], 1u, alpaka::hierarchy::Blocks{});
            continue;
          }
          const float ptFit = 1.f / std::abs(qpt);  // Patatrack's pT (field at the origin): the radius is R = ptFit / k
          const float sphi = std::sin(phi0), cphi = std::cos(phi0);

          // hits in Patatrack order (inside-out; the host seed sorts by radius: checked by the comparator)
          uint32_t key[4] = {kNoKey, kNoKey, kNoKey, kNoKey};
          bool isOT[4] = {false, false, false, false};
          uint8_t bits = 0;
          const uint32_t nToBits = nh < ::lst::kMaxPLSHitBitsInHitsSoA ? nh : ::lst::kMaxPLSHitBitsInHitsSoA;
          float lx = 0.f, ly = 0.f, lz = 0.f;
          float bzSum = 0.f;
          for (uint32_t i = 0; i < nh; ++i) {
            const uint32_t id = th[start + i].id();
            if (p.ptFieldCorrection)
              bzSum += ::mkfitdev::field::tkBz(
                  hits[id].xGlobal() + p.bsx, hits[id].yGlobal() + p.bsy, hits[id].zGlobal() + p.bsz);
            const bool ot = id >= p.nPixelSoA;
            uint32_t k = kNoKey;
            if (!ot)
              k = id < p.nPixKeys ? pixKey[id] : kNoKey;
            else
              k = (id - p.nPixelSoA) < p.nOTSoA ? otKey[id - p.nPixelSoA] : kNoKey;
            if (k == kNoKey)
              alpaka::atomicAdd(acc, &counts[4], 1u, alpaka::hierarchy::Blocks{});
            // slot of this hit among (0, 1, 2, last); hits 0..2 and the last one
            const int slot = i < 3 ? int(i) : (i + 1 == nh ? 3 : -1);
            if (slot == 0) {
              key[0] = k;
              isOT[0] = ot;
            } else if (slot == 1) {
              key[1] = k;
              isOT[1] = ot;
            } else if (slot == 2) {
              key[2] = k;
              isOT[2] = ot;
            } else if (slot == 3) {
              key[3] = k;
              isOT[3] = ot;
            }
            // hitDetBits: bit iSH for iSH < nToBits, the last bit is the last hit
            if (i + 1 < nToBits)
              bits |= uint8_t(ot) << i;
            if (i + 1 == nh)
              bits |= uint8_t(ot) << (nToBits - 1);
            if (i + 1 == nh) {
              lx = hits[id].xGlobal() + p.bsx;
              ly = hits[id].yGlobal() + p.bsy;
              lz = hits[id].zGlobal() + p.bsz;
            }
          }
          if (nh == 3) {  // with 3 hits slot 2 is also the last; slot 3 unused
            key[3] = kNoKey;
          }
          // momentum scale: 1, or <Bz>_hits / Bz(0)
          const float pt = p.ptFieldCorrection ? ptFit * (bzSum / float(nh)) / p.bz0 : ptFit;

          // PCA (perigee of the Patatrack fit, beam-spot frame): position bs + (tip sin, -tip cos, zip)
          const float x0 = p.bsx + tip * sphi;
          const float y0 = p.bsy - tip * cphi;
          float px0 = pt * cphi, py0 = pt * sphi, pz0 = pt * cot;
          float dxy = -tip;
          float dz = zip;

          // momentum at the outermost hit: uniform-field helix (Patatrack's field) from the PCA
          const float R = ptFit / p.k;
          const float cx = x0 + q * R * sphi;
          const float cy = y0 - q * R * cphi;
          const float lxHit = lx, lyHit = ly, lzHit = lz;
          if (p.pseudoLH == 1 && kf == nullptr) {
            // r3LH = the helix point at the hit's transverse radius rL (circle-circle intersection nearest to the hit;
            // no intersection: the hit projected radially onto the helix circle), z from the helix arc length
            const float rL2 = lx * lx + ly * ly;
            const float d = std::sqrt(cx * cx + cy * cy);
            float hx = 0.f, hy = 0.f;
            const float a = d > 0.f ? (rL2 - R * R + d * d) / (2.f * d) : 0.f;
            const float h2 = rL2 - a * a;
            if (d > 0.f && h2 >= 0.f) {
              const float h = std::sqrt(h2), ux = cx / d, uy = cy / d;
              const float x1 = a * ux - h * uy, y1 = a * uy + h * ux;
              const float x2 = a * ux + h * uy, y2 = a * uy - h * ux;
              const bool first =
                  (x1 - lx) * (x1 - lx) + (y1 - ly) * (y1 - ly) <= (x2 - lx) * (x2 - lx) + (y2 - ly) * (y2 - ly);
              hx = first ? x1 : x2;
              hy = first ? y1 : y2;
            } else {
              const float ex = lx - cx, ey = ly - cy;
              const float en = std::sqrt(ex * ex + ey * ey);
              hx = cx + R * ex / en;
              hy = cy + R * ey / en;
            }
            // transverse arc from the PCA, positive along the momentum (q > 0 turns clockwise)
            const float ax = x0 - cx, ay = y0 - cy, bx = hx - cx, by = hy - cy;
            const float dphi = std::atan2(ax * by - ay * bx, ax * bx + ay * by);
            const float s = -float(q) * R * dphi;
            lz = p.bsz + zip + s * cot;
            lx = hx;
            ly = hy;
          }
          const float ddx = lx - cx, ddy = ly - cy;
          const float dn = std::sqrt(ddx * ddx + ddy * ddy);
          float pxL = pt * (q * ddy / dn);
          float pyL = pt * (-q * ddx / dn);
          float pzL = pz0;
          if (kf != nullptr) {
            // DEVIATION DEV-7: the LSTInputProducer quantities of the host seed
            // (hltInitialStepSeeds) from the device creator emulation: the KF state on the last hit
            // (r3LH, p3LH, charge) and its TSCBL (p3PCA, dxy, dz; ptErr / etaErr below)
            ::mkfitdev::lstseeds::PixSeedOut const& f = kf[t];
            q = f.charge;
            lx = f.par[0];
            ly = f.par[1];
            lz = f.par[2];
            const float iptL = f.par[3], phL = f.par[4], thL = f.par[5];
            pxL = std::cos(phL) / iptL;
            pyL = std::sin(phL) / iptL;
            pzL = std::cos(thL) / std::sin(thL) / iptL;
            if (f.pcaStatus == 0) {
              px0 = f.pcaPx;
              py0 = f.pcaPy;
              pz0 = f.pcaPz;
              const float ptP = std::sqrt(px0 * px0 + py0 * py0);
              dxy = (-(f.pcaX - p.bsx) * py0 + (f.pcaY - p.bsy) * px0) / ptP;
              dz = (f.pcaZ - p.bsz) - ((f.pcaX - p.bsx) * px0 + (f.pcaY - p.bsy) * py0) / ptP * (pz0 / ptP);
            } else {  // host: invalid TSCBL -> zero PCA fields and errors, charge 0 (kept)
              alpaka::atomicAdd(acc, &counts[7], 1u, alpaka::hierarchy::Blocks{});
              if (f.pcaStatus == 2)
                alpaka::atomicAdd(acc, &counts[10], 1u, alpaka::hierarchy::Blocks{});
              px0 = py0 = pz0 = dxy = dz = 0.f;
              q = 0;
            }
          }

          // PCA used for the pseudo-hits / dxy / dz / superbin (pcaAnchor 1: the helix through the outermost hit with
          // the tangent there, back to its point of closest approach to the beam line, uniform field)
          float xa = x0, ya = y0, za = p.bsz + zip, cpa = cphi, spa = sphi;
          if (p.pcaAnchor == 1 && kf == nullptr) {
            const float hx = lxHit - cx, hy = lyHit - cy;
            const float hn = std::sqrt(hx * hx + hy * hy);
            const float cxa = lxHit - R * hx / hn, cya = lyHit - R * hy / hn;  // centre: the circle through the hit
            const float bx = p.bsx - cxa, by = p.bsy - cya;
            const float bn = std::sqrt(bx * bx + by * by);
            const float ux = bx / bn, uy = by / bn;
            xa = cxa + R * ux;
            ya = cya + R * uy;
            cpa = float(q) * uy;  // tangent at the PCA (direction of motion)
            spa = -float(q) * ux;
            const float ex = lxHit - cxa, ey = lyHit - cya;
            const float dphiA = std::atan2(ux * ey - uy * ex, ux * ex + uy * ey);
            za = lzHit + float(q) * R * dphiA * cot;  // z at the hit minus (arc length s = -q R dphi) x cot
          }
          const float ptIn = std::sqrt(pxL * pxL + pyL * pyL);
          // perigee errors (rho, theta) of the Patatrack covariance, then the host formulas of LSTInputProducer
          const float s2 = 1.f / (1.f + cot * cot);
          const float varRho = p.k * p.k * cv(9);
          const float covRT = p.k * s2 * cv(10);
          const float varTh = s2 * s2 * cv(12);
          const float pmag = pt * std::sqrt(1.f + cot * cot);
          const float qf = float(q);
          float ptErr = std::sqrt(pt * pt * pmag * pmag / (qf * qf) * varRho +
                                  2.f * std::sqrt(pmag * pmag * pt * pt) / qf * pz0 * covRT + pz0 * pz0 * varTh);
          float etaErr = std::sqrt(varTh) * pmag / pt;
          if (kf != nullptr) {
            const bool pcaOk = kf[t].pcaStatus == 0;
            ptErr = pcaOk ? kf[t].ptErr : 0.f;
            etaErr = pcaOk ? kf[t].etaErr : 0.f;
          }

          if (!(ptIn > p.ptCut - 2 * ptErr))
            continue;
          const float phiL = std::atan2(pyL, pxL);
          const float dPhi = deltaPhiRoot(phiL, std::atan2(ly, lx));
          int8_t pixtype;
          if (ptIn >= 2.0f)
            pixtype = ::lst::PixelType::kHighPt;
          else if (ptIn >= (p.ptCut - 2 * ptErr) and ptIn < 2.0f)
            pixtype = dPhi >= 0 ? ::lst::PixelType::kLowPtPosCurv : ::lst::PixelType::kLowPtNegCurv;
          else
            continue;

          //::lst::calculateR3FromPCA
          const bool anchor = p.pcaAnchor == 1 && kf == nullptr;
          const float pxA = anchor ? pt * cpa : px0, pyA = anchor ? pt * spa : py0;
          const float dxyA =
              anchor ? (-(xa - p.bsx) * pyA + (ya - p.bsy) * pxA) / std::sqrt(pxA * pxA + pyA * pyA) : dxy;
          const float dzA = anchor ? za - p.bsz : dz;
          const float ptP = std::sqrt(pxA * pxA + pyA * pyA);
          const float pP = std::sqrt(pxA * pxA + pyA * pyA + pz0 * pz0);
          const float vz = dzA * ptP * ptP / pP / pP;
          const float vx = -dxyA * pyA / ptP - pxA / pP * pz0 / pP * dzA;
          const float vy = dxyA * pxA / ptP - pyA / pP * pz0 / pP * dzA;
          const float etaP = etaRoot(ptP, pz0);
          const float phiP = std::atan2(pyA, pxA);

          o.pass = 1;
          o.seedIdx = seedIdx;
          o.charge = q;
          o.nHits = nh;
          o.nToSoA = nh < ::lst::kMaxPLSHitsInHitsSoA ? nh : ::lst::kMaxPLSHitsInHitsSoA;
          o.hitDetBits = bits;
          o.isQuad = nh > 3;
          o.pixelType = pixtype;
          o.ptIn = ptIn;
          o.ptErr = ptErr;
          o.px = pxL;
          o.py = pyL;
          o.pz = pzL;
          o.etaErr = etaErr;
          o.eta = etaRoot(ptIn, pzL);
          o.phi = phiL;
          o.deltaPhi = dPhi;
          // pseudo-hits as ::lst::prepareInput
          o.x[0] = vx;
          o.y[0] = vy;
          o.z[0] = vz;
          o.x[1] = ptP;
          o.y[1] = etaP;
          o.z[1] = phiP;
          o.x[2] = lx;
          o.y[2] = ly;
          o.z[2] = lz;
          o.x[3] = lx;
          o.y[3] = dxyA;
          o.z[3] = dzA;
          // detid / cluster size of the hits stored: iSH -> iH (the last slot is the last hit)
          for (int s = 0; s < 4; ++s) {
            const int src = (s + 1 == int(o.nToSoA)) ? (nh > 3 ? 3 : 2) : s;
            const uint32_t k = key[src];
            const bool ot = isOT[src];
            o.detid[s] = ot ? (k != kNoKey ? otDetId[k] : 0u) : ::lst::kPixelModuleId;
            o.clust[s] = ot ? (k != kNoKey ? otClust[k] : uint16_t(0)) : uint16_t(1);
          }
          o.idx[0] = key[0];
          o.idx[1] = key[1];
          o.idx[2] = key[2];
          o.idx[3] = nh > 3 ? key[3] : kNoKey;
          // superbin, with the host's mixed float/double expressions
          const float neta = 25.f, nphi = 72.f, nz = 25.f;
          const int etabin = (etaP + 2.6) / ((2 * 2.6) / neta);
          const int phibin = (phiP + std::numbers::pi_v<float>) / ((2. * std::numbers::pi_v<float>) / nphi);
          const float dzc = dzA < -30.f ? -30.f : (dzA > 30.f ? 30.f : dzA);
          const int dzbin = (dzc + 30) / (2 * 30 / nz);
          o.superbin = (nz * nphi) * etabin + (nz)*phibin + dzbin;
        }
      }
    };

    // the fitPixelSeeds inputs per SoA track
    struct KernelPixSeedIn {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::reco::TrackBlocksConstView tracks,
                                    uint32_t maxTracks,
                                    PixelTrackHits hits,
                                    uint32_t const* pixKey,
                                    uint32_t const* otKey,
                                    int32_t const* seedOfTrack,
                                    Params p,
                                    ::mkfitdev::HitSoAConstView mk,
                                    ::mkfitdev::lstseeds::PixSeedIn* out,
                                    uint32_t* counts) const {
        constexpr int kMax = ::mkfitdev::lstseeds::kMaxPixSeedHits;
        auto const tk = tracks.tracks();
        auto const th = tracks.trackHits();
        const int nT = tk.nTracks();
        const uint32_t nPixel = mk.nPixel();
        const uint32_t nRows = mk.metadata().size();
        for (uint32_t t : uniform_elements(acc, maxTracks)) {
          ::mkfitdev::lstseeds::PixSeedIn& o = out[t];
          o.nh = 0;
          if (int(t) >= nT || static_cast<int>(tk[t].quality()) < p.minQuality)
            continue;
          if (p.nSeedMap > 0 && (t >= p.nSeedMap || seedOfTrack[t] < 0))
            continue;
          const uint32_t start = t == 0 ? 0 : tk[t - 1].hitOffsets();
          const uint32_t nh = tk[t].hitOffsets() - start;
          if (nh < 3 || nh > uint32_t(kMaxTrackHits))
            continue;  // no pLS anyway (KernelPLSCompute)
          if (nh > uint32_t(kMax)) {
            alpaka::atomicAdd(acc, &counts[8], 1u, alpaka::hierarchy::Blocks{});
            o.nh = 1;  // < 2: status kPixNotFitted -> no pLS
            continue;
          }
          float r2[kMax];
          bool bad = false;
          for (uint32_t i = 0; i < nh; ++i) {
            const uint32_t id = th[start + i].id();
            const bool ot = id >= p.nPixelSoA;
            uint32_t k = kNoKey;
            if (!ot)
              k = id < p.nPixKeys ? pixKey[id] : kNoKey;
            else
              k = (id - p.nPixelSoA) < p.nOTSoA ? otKey[id - p.nPixelSoA] : kNoKey;
            const uint32_t row = k == kNoKey ? nRows : (ot ? nPixel : 0u) + k;
            if (row >= nRows || mk[row].layer() < 0) {
              alpaka::atomicAdd(acc, &counts[row >= nRows ? 11 : 12], 1u, alpaka::hierarchy::Blocks{});
              bad = true;
              break;
            }
            const float x = mk[row].x(), y = mk[row].y();
            const float dx = x - (hits[id].xGlobal() + p.bsx), dy = y - (hits[id].yGlobal() + p.bsy),
                        dz = mk[row].z() - (hits[id].zGlobal() + p.bsz);
            if (dx * dx + dy * dy + dz * dz > 1e-6f)
              alpaka::atomicAdd(acc, &counts[9], 1u, alpaka::hierarchy::Blocks{});
            // HitLessByRadius (stable insertion of <= kMax hits in this thread; no general sort)
            const float rr = x * x + y * y;
            int j = int(i);
            while (j > 0 && r2[j - 1] > rr) {
              r2[j] = r2[j - 1];
              o.hot[j] = o.hot[j - 1];
              --j;
            }
            r2[j] = rr;
            o.hot[j].index = int(k);
            o.hot[j].layer = mk[row].layer();
          }
          if (bad) {
            o.nh = 1;
            continue;
          }
          // GlobalTrackingRegion(proto pT, proto vertex, 0.2, 0.2): the reco::Track of PixelTrackProducerFromSoAAlpaka
          // = Patatrack's perigee point (beam-spot frame) and pT
          const auto st = tk[t].state();
          const float phi0 = st(0), tip = st(1), qpt = st(2);
          o.vx = p.bsx + tip * std::sin(phi0);
          o.vy = p.bsy - tip * std::cos(phi0);
          o.ptMin = 1.f / std::abs(qpt);
          o.nh = int32_t(nh);
        }
      }
    };

    struct KernelPixSeedStates {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::mkfitdev::TrackSoAView states,
                                    int32_t nStates,
                                    uint32_t maxTracks,
                                    int32_t const* seedOfTrack,
                                    uint32_t nSeedMap,
                                    ::mkfitdev::lstseeds::PixSeedOut const* kf,
                                    ::mkfitdev::lstseeds::PixSeedIn const* in) const {
        for (uint32_t t : uniform_elements(acc, maxTracks)) {
          if (t >= nSeedMap)
            continue;
          const int32_t e = seedOfTrack[t];
          if (e < 0 || e >= nStates)
            continue;
          if (in != nullptr && in[t].nh >= 2) {
            // K1: the pixel seed hits (radius order) of every listed track, whatever its KF status
            const int nh = in[t].nh < ::mkfitdev::kMaxTrkHits ? in[t].nh : ::mkfitdev::kMaxTrkHits;
            for (int k = 0; k < nh; ++k)
              states[e].hits().hot[k] = in[t].hot[k];
            states[e].nTotalHits() = static_cast<int16_t>(nh);
          }
          if (kf[t].status != ::mkfitdev::lstseeds::kPixOk)
            continue;
          for (int k = 0; k < 6; ++k)
            states[e].params().v[k] = kf[t].par[k];
          for (int k = 0; k < 21; ++k)
            states[e].errors().v[k] = kf[t].err[k];
          states[e].charge() = static_cast<int16_t>(kf[t].charge);
        }
      }
    };

    // scan: per chunk of kChunk tracks the pLS and pseudo-hit counts, a serial scan over chunks, then per-track offsets
    struct KernelChunkCount {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    PLS const* scratch,
                                    uint32_t maxTracks,
                                    uint32_t nChunks,
                                    uint32_t* chunkP,
                                    uint32_t* chunkH) const {
        for (uint32_t c : uniform_elements(acc, nChunks)) {
          uint32_t np = 0, nhs = 0;
          const uint32_t e = (c + 1) * kChunk < maxTracks ? (c + 1) * kChunk : maxTracks;
          for (uint32_t t = c * kChunk; t < e; ++t)
            if (scratch[t].pass) {
              ++np;
              nhs += scratch[t].nToSoA;
            }
          chunkP[c] = np;
          chunkH[c] = nhs;
        }
      }
    };

    struct KernelChunkScan {
      ALPAKA_FN_ACC void operator()(
          Acc1D const& acc, uint32_t nChunks, uint32_t* chunkP, uint32_t* chunkH, uint32_t* counts) const {
        if (once_per_grid(acc)) {
          uint32_t sp = 0, sh = 0;
          for (uint32_t c = 0; c < nChunks; ++c) {
            const uint32_t a = chunkP[c], b = chunkH[c];
            chunkP[c] = sp;
            chunkH[c] = sh;
            sp += a;
            sh += b;
          }
          counts[0] = sp;
          counts[1] = sh;
        }
      }
    };

    struct KernelChunkOffsets {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    PLS const* scratch,
                                    uint32_t maxTracks,
                                    uint32_t nChunks,
                                    uint32_t const* chunkP,
                                    uint32_t const* chunkH,
                                    uint32_t* pIdx,
                                    uint32_t* hOff) const {
        for (uint32_t c : uniform_elements(acc, nChunks)) {
          uint32_t np = chunkP[c], nhs = chunkH[c];
          const uint32_t e = (c + 1) * kChunk < maxTracks ? (c + 1) * kChunk : maxTracks;
          for (uint32_t t = c * kChunk; t < e; ++t) {
            pIdx[t] = np;
            hOff[t] = nhs;
            if (scratch[t].pass) {
              ++np;
              nhs += scratch[t].nToSoA;
            }
          }
        }
      }
    };

    // OT hits from the device OT rechit SoA (row = cluster key = legacy rechit index)
    struct KernelFillOTSoA {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::lst::LSTInputView out,
                                    ::mkfitdev::OTRecHitSoA::ConstView ot,
                                    uint32_t nOT) const {
        auto hits = out.hits();
        if (once_per_grid(acc))
          hits.nHitsOT() = nOT;
        for (uint32_t i : uniform_elements(acc, nOT)) {
          hits[i].xs() = ot[i].gx();
          hits[i].ys() = ot[i].gy();
          hits[i].zs() = ot[i].gz();
          hits[i].detid() = ot[i].detId();
          hits[i].clustsize() = ot[i].clustSize();
        }
      }
    };

    struct KernelFillPLS {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::lst::LSTInputView out,
                                    uint32_t maxTracks,
                                    uint32_t nPLSCap,
                                    uint32_t nOT,
                                    PLS const* scratch,
                                    uint32_t const* pIdx,
                                    uint32_t const* hOff) const {
        auto hits = out.hits();
        auto ps = out.pixelSeeds();
        for (uint32_t t : uniform_elements(acc, maxTracks)) {
          PLS const& o = scratch[t];
          if (!o.pass)
            continue;
          const uint32_t h0 = nOT + hOff[t];
          for (uint32_t s = 0; s < o.nToSoA; ++s) {
            hits[h0 + s].xs() = o.x[s];
            hits[h0 + s].ys() = o.y[s];
            hits[h0 + s].zs() = o.z[s];
            hits[h0 + s].detid() = o.detid[s];
            hits[h0 + s].clustsize() = o.clust[s];
            // first three hit keys + the last one (3-hit seed: hit 2 is the last): pLS rows only
            out.hitsIT()[hOff[t] + s].idxs() = o.idx[s];
          }
          const uint32_t r = pIdx[t];
          if (r >= nPLSCap)
            continue;
          ps[r].firstHit() = h0;
          ps[r].nHits() = o.nHits;
          ps[r].hitDetBits() = o.hitDetBits;
          ps[r].deltaPhi() = o.deltaPhi;
          ps[r].seedIdx() = o.seedIdx;
          ps[r].charge() = o.charge;
          ps[r].superbin() = o.superbin;
          ps[r].pixelType() = static_cast<::lst::PixelType>(o.pixelType);
          ps[r].isQuad() = o.isQuad;
          ps[r].ptIn() = o.ptIn;
          ps[r].ptErr() = o.ptErr;
          ps[r].px() = o.px;
          ps[r].py() = o.py;
          ps[r].pz() = o.pz;
          ps[r].etaErr() = o.etaErr;
          ps[r].eta() = o.eta;
          ps[r].phi() = o.phi;
        }
      }
    };
  }  // namespace

  void launchPLS(Queue& queue,
                 ::reco::TrackBlocksConstView tracks,
                 uint32_t maxTracks,
                 PixelTrackHits hits,
                 uint32_t const* pixKey,
                 uint32_t const* otKey,
                 uint32_t const* otDetId,
                 uint16_t const* otClust,
                 int32_t const* seedOfTrack,
                 Params p,
                 PLS* scratch,
                 uint32_t* pIdx,
                 uint32_t* hOff,
                 uint32_t* counts,
                 ::mkfitdev::lstseeds::PixSeedOut const* kf) {
    if (maxTracks == 0)
      return;
    constexpr uint32_t kBlock = 128;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(maxTracks, kBlock), kBlock),
                        KernelPLSCompute{},
                        tracks,
                        maxTracks,
                        hits,
                        pixKey,
                        otKey,
                        otDetId,
                        otClust,
                        seedOfTrack,
                        p,
                        scratch,
                        counts,
                        kf);
    const uint32_t nChunks = divide_up_by(maxTracks, kChunk);
    // chunk sums: counts[kNCounts .. kNCounts + 2 nChunks) (the producer sizes the counts buffer)
    uint32_t* chunkP = counts + ::mkfitdev::lstin::kNCounts;
    uint32_t* chunkH = chunkP + nChunks;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(nChunks, 64u), 64u),
                        KernelChunkCount{},
                        scratch,
                        maxTracks,
                        nChunks,
                        chunkP,
                        chunkH);
    alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(1, 1), KernelChunkScan{}, nChunks, chunkP, chunkH, counts);
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(nChunks, 64u), 64u),
                        KernelChunkOffsets{},
                        scratch,
                        maxTracks,
                        nChunks,
                        chunkP,
                        chunkH,
                        pIdx,
                        hOff);
  }

  void launchPixSeedStates(Queue& queue,
                           ::mkfitdev::TrackSoAView states,
                           int32_t nStates,
                           uint32_t maxTracks,
                           int32_t const* seedOfTrack,
                           uint32_t nSeedMap,
                           ::mkfitdev::lstseeds::PixSeedOut const* kf,
                           ::mkfitdev::lstseeds::PixSeedIn const* in) {
    if (maxTracks == 0)
      return;
    constexpr uint32_t kBlock = 128;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(maxTracks, kBlock), kBlock),
                        KernelPixSeedStates{},
                        states,
                        nStates,
                        maxTracks,
                        seedOfTrack,
                        nSeedMap,
                        kf,
                        in);
  }

  void launchPixSeedIn(Queue& queue,
                       ::reco::TrackBlocksConstView tracks,
                       uint32_t maxTracks,
                       PixelTrackHits hits,
                       uint32_t const* pixKey,
                       uint32_t const* otKey,
                       int32_t const* seedOfTrack,
                       Params p,
                       ::mkfitdev::HitSoAConstView mkHits,
                       ::mkfitdev::lstseeds::PixSeedIn* out,
                       uint32_t* counts) {
    if (maxTracks == 0)
      return;
    constexpr uint32_t kBlock = 128;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(maxTracks, kBlock), kBlock),
                        KernelPixSeedIn{},
                        tracks,
                        maxTracks,
                        hits,
                        pixKey,
                        otKey,
                        seedOfTrack,
                        p,
                        mkHits,
                        out,
                        counts);
  }

  void launchFillSoA(Queue& queue,
                     ::lst::LSTInputView out,
                     uint32_t maxTracks,
                     uint32_t nPLSCap,
                     ::mkfitdev::OTRecHitSoA::ConstView ot,
                     Params p,
                     PLS const* scratch,
                     uint32_t const* pIdx,
                     uint32_t const* hOff) {
    constexpr uint32_t kBlock = 128;
    const uint32_t nOT = p.nOT;
    alpaka::exec<Acc1D>(
        queue, make_workdiv<Acc1D>(divide_up_by(nOT > 0 ? nOT : 1u, kBlock), kBlock), KernelFillOTSoA{}, out, ot, nOT);
    if (maxTracks > 0)
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(divide_up_by(maxTracks, kBlock), kBlock),
                          KernelFillPLS{},
                          out,
                          maxTracks,
                          nPLSCap,
                          nOT,
                          scratch,
                          pIdx,
                          hOff);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstin

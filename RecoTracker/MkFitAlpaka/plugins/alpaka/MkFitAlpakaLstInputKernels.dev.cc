// Kernels of the device LST input: see MkFitAlpakaLstInputKernels.h.
#include <Eigen/Core>  // before any SoA header (Eigen columns of TracksSoA)
#include <cmath>
#include <cstring>
#include <numbers>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/LSTCore/interface/Common.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/DeviceHitInput.h"
#include "RecoTracker/MkFitAlpaka/interface/math/PcaToBeamLine.h"
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

    // pixKF: the pLS of one pixel seed from the device creator's state f (status kPixOk): DEVIATION DEV-7, the
    // LSTInputProducer quantities of the host seed (hltInitialStepSeeds) from the KF state on the last hit (r3LH, p3LH,
    // charge) and its TSCBL (p3PCA, dxy, dz, ptErr, etaErr). key / isOT: the seed's hits 0, 1, 2 and the last one (slot
    // 3 = kNoKey for 3 hits), bits: hitDetBits. Returns false (o.pass untouched) below the pT cut.
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool plsFromKF(Acc1D const& acc,
                                                  Params const& p,
                                                  ::mkfitdev::lstseeds::PixSeedOut const& f,
                                                  int32_t seedIdx,
                                                  uint32_t nh,
                                                  uint32_t const (&key)[4],
                                                  bool const (&isOT)[4],
                                                  uint8_t bits,
                                                  uint32_t const* otDetId,
                                                  uint16_t const* otClust,
                                                  uint32_t* counts,
                                                  PLS& o) {
      int q = f.charge;
      const float lx = f.par[0], ly = f.par[1], lz = f.par[2];
      const float iptL = f.par[3], phL = f.par[4], thL = f.par[5];
      const float pxL = std::cos(phL) / iptL;
      const float pyL = std::sin(phL) / iptL;
      const float pzL = std::cos(thL) / std::sin(thL) / iptL;
      float px0, py0, pz0, dxy, dz;
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
      const float ptIn = std::sqrt(pxL * pxL + pyL * pyL);
      const bool pcaOk = f.pcaStatus == 0;
      const float ptErr = pcaOk ? f.ptErr : 0.f;
      const float etaErr = pcaOk ? f.etaErr : 0.f;

      if (!(ptIn > p.ptCut - 2 * ptErr))
        return false;
      const float phiL = std::atan2(pyL, pxL);
      const float dPhi = deltaPhiRoot(phiL, std::atan2(ly, lx));
      int8_t pixtype;
      if (ptIn >= 2.0f)
        pixtype = ::lst::PixelType::kHighPt;
      else if (ptIn >= (p.ptCut - 2 * ptErr) and ptIn < 2.0f)
        pixtype = dPhi >= 0 ? ::lst::PixelType::kLowPtPosCurv : ::lst::PixelType::kLowPtNegCurv;
      else
        return false;

      //::lst::calculateR3FromPCA
      const float pxA = px0, pyA = py0;
      const float dxyA = dxy;
      const float dzA = dz;
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
      return true;
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
          if (kf != nullptr) {  // pixKF: every pLS field from the device creator's state
            plsFromKF(acc, p, kf[t], seedIdx, nh, key, isOT, bits, otDetId, otClust, counts, o);
            continue;
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
          if (p.pseudoLH == 1) {
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
          // PCA used for the pseudo-hits / dxy / dz / superbin (pcaAnchor 1: the helix through the outermost hit with
          // the tangent there, back to its point of closest approach to the beam line, uniform field)
          float xa = x0, ya = y0, za = p.bsz + zip, cpa = cphi, spa = sphi;
          if (p.pcaAnchor == 1) {
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
          const bool anchor = p.pcaAnchor == 1;
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

    // generic pixel seeds (seed rows: TrackSoA, row = seed index): the state on the last hit as the creator's output
    // (charge 0 or a non-finite state = no seed; pcaPixelSeeds adds the TSCBL); with plsStates (nPls rows) the state
    // of row e there instead where its charge is not 0 (the state the pLS starts from, e.g. at the last pixel hit)
    struct KernelSeedRowsToKF {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::mkfitdev::TrackSoAConstView seeds,
                                    uint32_t n,
                                    ::mkfitdev::TrackSoAConstView plsStates,
                                    uint32_t nPls,
                                    ::mkfitdev::lstseeds::PixSeedOut* out) const {
        for (uint32_t e : uniform_elements(acc, n)) {
          ::mkfitdev::lstseeds::PixSeedOut& o = out[e];
          o.pcaStatus = ::mkfitdev::pca::kPcaFailed;
          bool ok = seeds[e].charge() != 0;
          const bool pls = ok && e < nPls && plsStates[e].charge() != 0;
          auto const& st = pls ? plsStates[e] : seeds[e];
          for (int k = 0; k < 6; ++k) {
            o.par[k] = st.params().v[k];
            ok = ok && std::isfinite(o.par[k]);
          }
          for (int k = 0; k < 21; ++k) {
            o.err[k] = st.errors().v[k];
            ok = ok && std::isfinite(o.err[k]);
          }
          o.charge = st.charge();
          o.status = ok ? ::mkfitdev::lstseeds::kPixOk : ::mkfitdev::lstseeds::kPixNotFitted;
        }
      }
    };

    // generic pixel seeds: the pLS of every seed row (the counterpart of KernelPLSCompute with pixKF); a hit is an OT
    // hit when its mkFit layer is not a pixel layer, its key the cluster key (the LST hit index)
    struct KernelPLSFromSeedRows {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::mkfitdev::TrackSoAConstView seeds,
                                    ::mkfitdev::LayerInfoSoA::ConstView layers,
                                    uint32_t n,
                                    ::mkfitdev::lstseeds::PixSeedOut const* kf,
                                    uint32_t const* otDetId,
                                    uint16_t const* otClust,
                                    Params p,
                                    PLS* scratch,
                                    uint32_t* counts) const {
        if (once_per_grid(acc))
          counts[2] = n;
        for (uint32_t e : uniform_elements(acc, n)) {
          PLS& o = scratch[e];
          o.pass = 0;
          o.nToSoA = 0;
          const int nTot = seeds[e].nTotalHits();
          if (seeds[e].charge() == 0 && nTot == 0)
            continue;  // not a seed
          uint32_t key[4] = {kNoKey, kNoKey, kNoKey, kNoKey};
          bool isOT[4] = {false, false, false, false};
          uint32_t nh = 0;
          for (int i = 0; i < nTot; ++i)
            nh += seeds[e].hits().hot[i].index >= 0;
          if (nh > uint32_t(kMaxTrackHits) || nh < 3) {
            alpaka::atomicAdd(acc, &counts[3], 1u, alpaka::hierarchy::Blocks{});
            continue;
          }
          if (kf[e].status != ::mkfitdev::lstseeds::kPixOk) {
            alpaka::atomicAdd(acc, &counts[6], 1u, alpaka::hierarchy::Blocks{});
            continue;
          }
          uint8_t bits = 0;
          const uint32_t nToBits = nh < ::lst::kMaxPLSHitBitsInHitsSoA ? nh : ::lst::kMaxPLSHitBitsInHitsSoA;
          uint32_t i = 0;
          for (int h = 0; h < nTot; ++h) {
            const auto hot = seeds[e].hits().hot[h];
            if (hot.index < 0)
              continue;
            const bool ot = !layers[hot.layer].is_pixel();
            const uint32_t k = uint32_t(hot.index);
            const int slot = i < 3 ? int(i) : (i + 1 == nh ? 3 : -1);
            if (slot >= 0) {
              key[slot] = k;
              isOT[slot] = ot;
            }
            if (i + 1 < nToBits)
              bits |= uint8_t(ot) << i;
            if (i + 1 == nh)
              bits |= uint8_t(ot) << (nToBits - 1);
            ++i;
          }
          if (nh == 3)
            key[3] = kNoKey;
          plsFromKF(acc, p, kf[e], int32_t(e), nh, key, isOT, bits, otDetId, otClust, counts, o);
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
        if (once_per_grid(acc))
          states.nTracks() = nStates;  // the pixel-track count (the seed hand-off's pixelSeedIndex range)
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
          if (kf == nullptr || kf[t].status != ::mkfitdev::lstseeds::kPixOk)
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

  namespace {
    // pLS index and pseudo-hit offset per scratch row (chunk counts, a serial scan over chunks, per-row offsets)
    void scanPLS(Queue& queue, uint32_t maxTracks, PLS* scratch, uint32_t* pIdx, uint32_t* hOff, uint32_t* counts) {
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
    scanPLS(queue, maxTracks, scratch, pIdx, hOff, counts);
  }

  void launchPixSeedStates(Queue& queue,
                           ::mkfitdev::TrackSoAView states,
                           int32_t nStates,
                           uint32_t maxTracks,
                           int32_t const* seedOfTrack,
                           uint32_t nSeedMap,
                           ::mkfitdev::lstseeds::PixSeedOut const* kf,
                           ::mkfitdev::lstseeds::PixSeedIn const* in) {
    constexpr uint32_t kBlock = 128;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(maxTracks > 0 ? maxTracks : 1u, kBlock), kBlock),
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

  void launchPLSFromSeedRows(Queue& queue,
                             ::mkfitdev::TrackSoAConstView seeds,
                             ::mkfitdev::TrackSoAConstView plsStates,
                             uint32_t nPls,
                             ::mkfitdev::LayerInfoSoA::ConstView layers,
                             uint32_t n,
                             uint32_t const* otDetId,
                             uint16_t const* otClust,
                             Params p,
                             ::mkfitdev::lstseeds::PixSeedBeam const& beam,
                             ::mkfitdev::lstseeds::PixSeedOut* kf,
                             PLS* scratch,
                             uint32_t* pIdx,
                             uint32_t* hOff,
                             uint32_t* counts) {
    if (n == 0)
      return;
    constexpr uint32_t kBlock = 128;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(n, kBlock), kBlock),
                        KernelSeedRowsToKF{},
                        seeds,
                        n,
                        plsStates,
                        nPls,
                        kf);
    ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstseeds::pcaPixelSeeds(queue, kf, int32_t(n), beam);
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(n, kBlock), kBlock),
                        KernelPLSFromSeedRows{},
                        seeds,
                        layers,
                        n,
                        kf,
                        otDetId,
                        otClust,
                        p,
                        scratch,
                        counts);
    scanPLS(queue, n, scratch, pIdx, hOff, counts);
  }

  namespace {
    // PixelTrackProducerFromSoAAlpaka's selection of SoA track t (the tracks it converts, minNumberOfHits <= 3): quality
    // >= minQuality and a finite converted track (its check: chi2 * ndof and the momentum, i.e. phi, q/pT != 0 and
    // cot(theta); the positions tip / zip do not enter it)
    template <typename TTk>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool legacyPixelTrack(TTk const& tk, uint32_t t, int minQuality) {
      if (static_cast<int>(tk[t].quality()) < minQuality)
        return false;
      const auto st = tk[t].state();
      const float phi0 = st(0), qpt = st(2), cot = st(3);
      return std::isfinite(tk[t].chi2()) && std::isfinite(phi0) && std::isfinite(qpt) && qpt != 0.f &&
             std::isfinite(cot);
    }

    // the legacy numbering is the rank in descending pT (the SoA pt column): a histogram on the pT bits (bucket order =
    // pT order), the bucket offsets, then the exact rank inside the bucket (ties by SoA index)
    constexpr uint32_t kPtBucketBits = 14;
    constexpr uint32_t kPtBuckets = 1u << kPtBucketBits;  // 1024 per octave, pT 0.25 .. 16384 GeV (clamped outside)
    constexpr uint32_t kPtBucketChunk = 256;
    constexpr uint32_t kPtBucketChunks = kPtBuckets / kPtBucketChunk;
    ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t ptBucketDesc(float pt) {
      // positive floats order as their bits; NaN / non-positive into the lowest-pT bucket
      constexpr uint32_t kBase = 0x3e800000u;  // 0.25f
      uint32_t bits;
      memcpy(&bits, &pt, sizeof(bits));
      uint32_t b = 0;
      if (pt > 0.f && bits > kBase)
        b = (bits - kBase) >> (23 - 10);
      b = b < kPtBuckets ? b : kPtBuckets - 1;
      return kPtBuckets - 1 - b;
    }

    struct KernelPixelTrackCount {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::reco::TrackBlocksConstView tracks,
                                    uint32_t capacity,
                                    int minQuality,
                                    uint32_t* counts) const {
        auto const tk = tracks.tracks();
        const uint32_t nT = uint32_t(tk.nTracks()) < capacity ? uint32_t(tk.nTracks()) : capacity;
        if (once_per_grid(acc))
          counts[0] = nT;
        for (uint32_t t : uniform_elements(acc, nT))
          if (legacyPixelTrack(tk, t, minQuality))
            alpaka::atomicAdd(acc, &counts[1], 1u, alpaka::hierarchy::Blocks{});
      }
    };

    struct KernelPtBucketCount {
      ALPAKA_FN_ACC void operator()(
          Acc1D const& acc, ::reco::TrackBlocksConstView tracks, uint32_t nT, int minQuality, uint32_t* cnt) const {
        auto const tk = tracks.tracks();
        for (uint32_t t : uniform_elements(acc, nT))
          if (legacyPixelTrack(tk, t, minQuality))
            alpaka::atomicAdd(acc, &cnt[ptBucketDesc(tk[t].pt())], 1u, alpaka::hierarchy::Blocks{});
      }
    };

    // bucket offsets: per chunk the sum, a serial scan over the chunks (and the total), per bucket the offset
    struct KernelPtBucketChunkSum {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, uint32_t const* cnt, uint32_t* chunk) const {
        for (uint32_t c : uniform_elements(acc, kPtBucketChunks)) {
          uint32_t n = 0;
          for (uint32_t b = c * kPtBucketChunk; b < (c + 1) * kPtBucketChunk; ++b)
            n += cnt[b];
          chunk[c] = n;
        }
      }
    };
    struct KernelPtBucketChunkScan {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, uint32_t* chunk, uint32_t* total) const {
        if (once_per_grid(acc)) {
          uint32_t run = 0;
          for (uint32_t c = 0; c < kPtBucketChunks; ++c) {
            const uint32_t n = chunk[c];
            chunk[c] = run;
            run += n;
          }
          *total = run;
        }
      }
    };
    struct KernelPtBucketOffsets {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc, uint32_t const* cnt, uint32_t const* chunk, uint32_t* off) const {
        for (uint32_t c : uniform_elements(acc, kPtBucketChunks)) {
          uint32_t run = chunk[c];
          for (uint32_t b = c * kPtBucketChunk; b < (c + 1) * kPtBucketChunk; ++b) {
            off[b] = run;
            run += cnt[b];
          }
        }
      }
    };
    struct KernelPtBucketFill {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::reco::TrackBlocksConstView tracks,
                                    uint32_t nT,
                                    int minQuality,
                                    uint32_t const* off,
                                    uint32_t* fill,
                                    int32_t* list) const {
        auto const tk = tracks.tracks();
        for (uint32_t t : uniform_elements(acc, nT)) {
          if (!legacyPixelTrack(tk, t, minQuality))
            continue;
          const uint32_t b = ptBucketDesc(tk[t].pt());
          list[off[b] + alpaka::atomicAdd(acc, &fill[b], 1u, alpaka::hierarchy::Blocks{})] = int32_t(t);
        }
      }
    };
    struct KernelPtRank {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::reco::TrackBlocksConstView tracks,
                                    uint32_t nT,
                                    int minQuality,
                                    uint32_t const* cnt,
                                    uint32_t const* off,
                                    int32_t const* list,
                                    int32_t* seedOf) const {
        auto const tk = tracks.tracks();
        for (uint32_t t : uniform_elements(acc, nT)) {
          if (!legacyPixelTrack(tk, t, minQuality))
            continue;
          const float pt = tk[t].pt();
          const uint32_t b = ptBucketDesc(pt);
          uint32_t rank = off[b];
          for (uint32_t k = off[b]; k < off[b] + cnt[b]; ++k) {
            const uint32_t j = uint32_t(list[k]);
            const float ptj = tk[j].pt();
            rank += ptj > pt || (ptj == pt && j < t);
          }
          seedOf[t] = int32_t(rank);
        }
      }
    };
  }  // namespace

  void launchPixelTrackCount(
      Queue& queue, ::reco::TrackBlocksConstView tracks, uint32_t capacity, int minQuality, uint32_t* counts) {
    constexpr uint32_t kBlock = 256;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(capacity > 0 ? capacity : 1u, kBlock), kBlock),
                        KernelPixelTrackCount{},
                        tracks,
                        capacity,
                        minQuality,
                        counts);
  }

  uint32_t pixelTrackIndexScratchSize(uint32_t nT) { return 3 * kPtBuckets + kPtBucketChunks + 1 + (nT > 0 ? nT : 1u); }

  void launchPixelTrackIndex(Queue& queue,
                             ::reco::TrackBlocksConstView tracks,
                             uint32_t nT,
                             int minQuality,
                             int32_t* seedOf,
                             uint32_t* scratch,
                             uint32_t* nSelected) {
    // scratch: cnt, fill (zeroed here), off [kPtBuckets] each, chunk sums [kPtBucketChunks], the total, the list [nT]
    uint32_t* cnt = scratch;
    uint32_t* fill = cnt + kPtBuckets;
    uint32_t* off = fill + kPtBuckets;
    uint32_t* chunk = off + kPtBuckets;
    int32_t* list = reinterpret_cast<int32_t*>(chunk + kPtBucketChunks + 1);
    auto counters = cms::alpakatools::make_device_view(queue, scratch, 2 * kPtBuckets);
    alpaka::memset(queue, counters, 0x00);
    auto index = cms::alpakatools::make_device_view(queue, seedOf, nT > 0 ? nT : 1u);
    alpaka::memset(queue, index, 0xff);  // -1
    constexpr uint32_t kBlock = 128;
    const uint32_t blocks = divide_up_by(nT > 0 ? nT : 1u, kBlock);
    alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(blocks, kBlock), KernelPtBucketCount{}, tracks, nT, minQuality, cnt);
    alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(1, kPtBucketChunks), KernelPtBucketChunkSum{}, cnt, chunk);
    alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(1, 1), KernelPtBucketChunkScan{}, chunk, nSelected);
    alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(1, kPtBucketChunks), KernelPtBucketOffsets{}, cnt, chunk, off);
    alpaka::exec<Acc1D>(
        queue, make_workdiv<Acc1D>(blocks, kBlock), KernelPtBucketFill{}, tracks, nT, minQuality, off, fill, list);
    alpaka::exec<Acc1D>(
        queue, make_workdiv<Acc1D>(blocks, kBlock), KernelPtRank{}, tracks, nT, minQuality, cnt, off, list, seedOf);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstin

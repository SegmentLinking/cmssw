#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "vdt/atan2.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederCandidatesSoA.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederTableBuilder.h"

// The static tables of the chain finder (the chains, the binned layers' axes, the gap map, the patterns), ported from
// mkFit's seeder configuration (SeedChain, SeedLayerOfHits, SensorGapMap) with the same arithmetic.

namespace mkseeder {

  namespace {

    constexpr float kPi = 3.14159265358979323846f;  // the binning's phi range [-kPi, kPi)

    // a pixel layer of the chains' crossing envelopes: barrel 0-3, the discs 16-27 (+z) and 38-49 (-z)
    bool isPixelEnvelopeLayer(int l) { return l <= 3 || (l >= 16 && l <= 27) || (l >= 38 && l <= 49); }

    // ---- the binning axes of a layer (mkFit's binnor axes: R = float, I = unsigned short; 16 fine bits; 11 bits of
    // phi bins, 8 bits of fine bins per q bin)
    constexpr unsigned int kFineBits = 16, kPhiBits = 11, kQFineShift = 8;
    struct Axis {
      float min, max, fineScale, scale, fineLastBinHighEdge, lastBinHighEdge;
      uint16_t lastFineBin, lastBin;
      Axis(float rMin, float rMax, unsigned int fineSize, unsigned int size)
          : min(rMin),
            max(rMax),
            fineScale(fineSize / (rMax - rMin)),
            scale(size / (rMax - rMin)),
            fineLastBinHighEdge(rMax - 0.5 / fineScale),
            lastBinHighEdge(rMax - 0.5 / scale),
            lastFineBin(fineSize - 1),
            lastBin(size - 1) {}
    };
    Axis phiAxis() { return Axis(-kPi, kPi, 1u << kFineBits, 1u << kPhiBits); }
    Axis qAxis(float lo, float hi, unsigned int nBins) { return Axis(lo, hi, nBins << kQFineShift, nBins); }
    unsigned int nQBinsOf(double lo, double hi, double bin) {
      return std::max(1u, (unsigned int)std::ceil((hi - lo) / bin));
    }

    // a binned layer: its axes from the layer's nominal extent
    struct BinnedLayer {
      int id;
      bool disc;
      double qbarLo, qbarHi, qLo, qHi;
      unsigned int nQ;
      Axis phi, q;
      BinnedLayer(int id_, const MkSeederTrackerLayer &li, double qBin)
          : id(id_),
            disc(!li.isBarrel),
            qbarLo(li.isBarrel ? li.rIn : li.zMin),
            qbarHi(li.isBarrel ? li.rOut : li.zMax),
            qLo(li.isBarrel ? li.zMin : li.rIn),
            qHi(li.isBarrel ? li.zMax : li.rOut),
            nQ(nQBinsOf(qLo, qHi, qBin)),
            phi(phiAxis()),
            q(qAxis((float)qLo, (float)qHi, nQ)) {}
    };

    // ---- the chains' crossing envelopes (geometry only): a layer is crossed DEFINITELY by an r-z line if the
    // crossing lies inside it by more than delta (cm: in z for a barrel layer, in r for a disc), MAYBE within delta of
    // an edge
    struct Envelope {
      int id;
      bool disc;
      double pos, posLow, posHigh;  // qbar: barrel r (mid, rin, rout); disc z (mid, zmin, zmax)
      double low, high;             // q range: barrel z; disc r
    };
    struct Envelopes {
      std::vector<Envelope> env;
      double delta = -1;
      // the pixel layers, plus the non-pixel layers the chain uses (e.g. OT1-P)
      void setup(const MkSeederTrackerDescription &tracker, const std::set<int> &extra) {
        env.clear();
        for (int l = 0; l < (int)tracker.layers.size(); ++l) {
          if (!isPixelEnvelopeLayer(l) && !extra.count(l))
            continue;
          const MkSeederTrackerLayer &li = tracker.layers[l];
          if (li.isBarrel)
            env.push_back({l, false, 0.5 * (li.rIn + li.rOut), li.rIn, li.rOut, li.zMin, li.zMax});
          else
            env.push_back({l, true, 0.5 * (li.zMin + li.zMax), li.zMin, li.zMax, li.rIn, li.rOut});
        }
      }
      // 0 no, 1 maybe, 2 definite: the line's q over the layer's whole qbar slab against the q range
      int cross(const Envelope &e, double z0, double cot) const {
        double x1, x2;
        if (!e.disc) {
          x1 = z0 + cot * e.posLow;
          x2 = z0 + cot * e.posHigh;
        } else {
          if (std::abs(cot) < 1e-9)
            return 0;
          const double s = (e.pos - z0) / cot;
          if (s <= 0)
            return 0;
          x1 = (e.posLow - z0) / cot;
          x2 = (e.posHigh - z0) / cot;
        }
        const double xl = std::min(x1, x2), xh = std::max(x1, x2);
        if (xl > e.low + delta && xh < e.high - delta)
          return 2;
        if (xh > e.low - delta && xl < e.high + delta)
          return 1;
        return 0;
      }
    };

    // ---- the seeding windows (one table per layer combination)
    struct EtaWindow {
      float lo, hi, aphi, bphi, aq, bq;
    };
    struct Windows {
      float ptMin = 0.9f, d0Max = 0.1f, zv = 25.0f, bsZ = 0.0f, margB = 0.002f;
      float zvScan = 25.0f;  // the z0 range of the start-pair scans: max(zv, zv_outer)
      float phiC = 0.004f, qC = 0.10f, phiD = 0.004f, qD = 0.10f;
      float bPhiD = 0, bQD = 0, sRef = 0;
      static constexpr int kMaxEtaWindows = 8;
      EtaWindow eta[kMaxEtaWindows];
      int nEta = 0;
      void addEtaWindow(const EtaWindow &w) {
        if (nEta < kMaxEtaWindows)
          eta[nEta++] = w;
      }
    };
    // a pattern's stage d windows
    struct DWindows {
      float aphi, bphi, aq, bq, sref = 0;
      std::vector<EtaWindow> eta;
      float qc = 0;  // the pattern's q_c (the cleaning score); 0: the chain's
    };

    // ---- one z side's chain
    struct ChainEnvelope {
      bool ok = false, disc = false, pix = false, xot = false;
      float pos = 0, plo = 0, phi = 0, loIn = 0, hiIn = 0, loOut = 0, hiOut = 0;
    };
    struct WindowTable {
      float phiC, qC, aphi, bphi, aq, bq, sref;
      int e0, ne;
    };
    struct Chain {
      // the configuration
      std::map<std::array<int, 3>, std::pair<float, float>> winC;
      std::map<std::array<int, 4>, DWindows> winD;
      std::set<std::array<int, 3>> prunedC;  // prune: combinations not searched
      std::set<std::array<int, 4>> prunedD;
      Windows P;
      int maxHoles = 1, alwaysForward = 0, startHoles = -1, leadOnly = 0, knownOnly = 1, maxHolesOT = 0;
      int innerOTOnly = 0, otAny = 0, otMaxPix = 0, startGap = 0;
      std::vector<int> chainOT;
      bool hasGapMap = false;
      int side = 1;
      // the chain
      std::vector<int> order;  // mkFit layer ids, crossing order
      std::vector<int> envOf;  // per position: the envelope's index
      std::vector<std::pair<int, int>> starts;
      std::vector<char> startGapOk;
      // the finder's tables
      int n = 0;
      std::vector<Windows> params;
      std::vector<int> idxC, idxD;
      std::vector<ChainEnvelope> env;
      std::vector<WindowTable> tables;
      std::vector<EtaWindow> etaWindows;
      std::vector<std::vector<std::pair<int, int>>> holePos;
      std::vector<std::pair<float, float>> startCot;
      std::vector<int> gapLid;

      // the order (B1..B4, the side's discs, OT1-P and the chain's OT layers if present) and the start pairs, from
      // lines of the beam region
      void setup(const Envelopes &o, int side_, const std::set<int> &have) {
        side = side_;
        order.clear();
        for (int l : {0, 1, 2, 3})
          order.push_back(l);
        for (int l = 16; l <= 27; ++l)
          order.push_back(side > 0 ? l : l + 22);
        if (have.count(4))
          order.push_back(4);
        for (int l : chainOT)
          if (l != 4 && have.count(l))
            order.push_back(l);
        envOf.assign(order.size(), -1);
        for (int p = 0; p < (int)order.size(); ++p)
          for (int e = 0; e < (int)o.env.size(); ++e)
            if (o.env[e].id == order[p])
              envOf[p] = e;
        std::set<std::pair<int, int>> sp, gp;
        for (double z0 = P.bsZ - P.zvScan; z0 <= P.bsZ + P.zvScan + 1e-9; z0 += 0.5)
          for (double eta = 0; eta < 4.2; eta += 0.002) {
            const double cot = side * std::sinh(eta);
            std::vector<int> seq;  // the pixel positions crossed (maybe or definite)
            for (int p = 0; p < (int)order.size(); ++p) {
              if (!isPixelEnvelopeLayer(order[p]) || envOf[p] < 0)
                continue;
              if (o.cross(o.env[envOf[p]], z0, cot))
                seq.push_back(p);
            }
            const int hs = startHoles < 0 ? maxHoles : startHoles;
            for (int i = 0; i <= hs && i < (int)seq.size(); ++i)
              for (int j = i + 1; j <= (leadOnly ? i + 1 : i + 1 + hs - i) && j < (int)seq.size(); ++j)
                sp.insert({seq[i], seq[j]});
            const int hg = innerOTOnly ? std::min(hs, maxHolesOT) : hs;
            if (startGap && leadOnly)
              for (int i = 0; i + 1 <= hg && i + 2 < (int)seq.size(); ++i)
                if (startGap == 1 || (startGap == 2 && order[seq[i + 2]] < 4) ||
                    (startGap == 3 && order[seq[i + 1]] < 4))
                  gp.insert({seq[i], seq[i + 2]});
          }
        sp.insert(gp.begin(), gp.end());
        starts.assign(sp.begin(), sp.end());
        startGapOk.assign(starts.size(), 0);
        for (size_t si = 0; si < starts.size(); ++si)
          startGapOk[si] = gp.count(starts[si]) > 0;
      }

      // the window tables per combination of positions: an index into params, -1 without a table; params[0] is P
      void buildParams() {
        const int n = order.size();
        params.assign(1, P);
        idxC.assign(n * n * n, -1);
        idxD.assign(n * n * n * n, -1);
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < n; ++j)
            for (int p = 0; p < n; ++p) {
              auto it = winC.find({order[i], order[j], order[p]});
              if (prunedC.count({order[i], order[j], order[p]}))
                idxC[(i * n + j) * n + p] = kPrunedCombination;
              else if (it != winC.end()) {
                Windows Q = P;
                Q.phiC = it->second.first, Q.qC = it->second.second;
                params.push_back(Q);
                idxC[(i * n + j) * n + p] = params.size() - 1;
              }
              for (int k = 0; k < n; ++k) {
                if (prunedD.count({order[i], order[j], order[k], order[p]})) {
                  idxD[((i * n + j) * n + k) * n + p] = kPrunedCombination;
                  continue;
                }
                auto jt = winD.find({order[i], order[j], order[k], order[p]});
                if (jt == winD.end())
                  continue;
                Windows Q = P;
                Q.phiD = jt->second.aphi, Q.bPhiD = jt->second.bphi, Q.qD = jt->second.aq, Q.bQD = jt->second.bq,
                Q.sRef = jt->second.sref;
                if (jt->second.qc > 0)
                  Q.qC = jt->second.qc;
                for (const auto &w : jt->second.eta)
                  Q.addEtaWindow(w);
                params.push_back(Q);
                idxD[((i * n + j) * n + k) * n + p] = params.size() - 1;
              }
            }
      }

      static int crossingClass(const ChainEnvelope &e, float x1, float x2) {
        const float xl = std::min(x1, x2), xh = std::max(x1, x2);
        return ((xh > e.loOut) & (xl < e.hiOut)) + ((xl > e.loIn) & (xh < e.hiIn));
      }
      static int crossingState(const ChainEnvelope &e, float z0, float cot) {
        if (!e.disc)
          return crossingClass(e, z0 + cot * e.plo, z0 + cot * e.phi);
        const float c = cot, cs = std::abs(c) >= 1e-9f ? c : 1.0f, ic = std::abs(c) >= 1e-9f ? 1.0f / cs : 0.0f;
        const float zr = -z0 * ic;
        const int valid = (ic != 0) & (e.pos * ic + zr > 0);
        return valid * crossingClass(e, e.plo * ic + zr, e.phi * ic + zr);
      }

      // the finder's tables: the envelopes in float, the window tables, the hole positions of the start pairs, the
      // gap map's layer of the gap starts, the start pairs' cot ranges
      void setupFinder(const Envelopes &o, const std::vector<bool> &gapMapLayer) {
        buildParams();
        n = order.size();
        const double dl = o.delta;
        env.assign(n, ChainEnvelope{});
        for (int p = 0; p < n; ++p) {
          ChainEnvelope &e = env[p];
          e.pix = isPixelEnvelopeLayer(order[p]);
          e.xot = !e.pix && order[p] != 4;
          if (envOf[p] < 0)
            continue;
          const Envelope &E = o.env[envOf[p]];
          e.ok = true, e.disc = E.disc, e.pos = E.pos, e.plo = E.posLow, e.phi = E.posHigh;
          e.loIn = E.low + dl, e.hiIn = E.high - dl, e.loOut = E.low - dl, e.hiOut = E.high + dl;
        }
        tables.clear();
        etaWindows.clear();
        for (const Windows &Q : params) {
          tables.push_back({Q.phiC, Q.qC, Q.phiD, Q.bPhiD, Q.qD, Q.bQD, Q.sRef, (int)etaWindows.size(), Q.nEta});
          for (int i = 0; i < Q.nEta; ++i)
            etaWindows.push_back(Q.eta[i]);
        }
        holePos.assign(starts.size(), {});
        for (size_t si = 0; si < starts.size(); ++si)
          for (int q = 0; q < starts[si].second; ++q)
            if (q != starts[si].first && env[q].pix && env[q].ok)
              holePos[si].push_back({q, q > starts[si].first});
        gapLid.assign(starts.size(), -1);
        if (hasGapMap)
          for (size_t si = 0; si < starts.size(); ++si) {
            if (!startGapOk[si])
              continue;
            int nl = 0, lid = -1;
            for (const auto &h : holePos[si]) {
              const int l = order[h.first];
              if (h.second && l >= 0 && l < (int)gapMapLayer.size() && gapMapLayer[l])
                ++nl, lid = l;
            }
            if (nl == 1)
              gapLid[si] = lid;
          }
        scanStarts();
      }

      void scanStarts() {
        const int ns = starts.size(), hs = startHoles < 0 ? maxHoles : startHoles;
        std::vector<float> elo(ns, 1e30f), ehi(ns, -1e30f);
        std::vector<int> st(n);
        int pOT = -1;  // OT1-P's position, if present
        for (int p = 0; p < n; ++p)
          if (order[p] == 4)
            pOT = p;
        const float zlo = P.bsZ - P.zvScan, zhi = P.bsZ + P.zvScan;
        constexpr int kNz = 100, kNe = 2500;
        constexpr float kDe = 0.002f, kWiden = 0.02f;
        for (int iz = 0; iz <= kNz; ++iz) {
          const float z0 = zlo + (zhi - zlo) * iz / kNz;
          for (int ie = 0; ie <= kNe; ++ie) {
            const float eta = ie * kDe, cot = side * std::sinh(eta);
            for (int p = 0; p < n; ++p)
              st[p] = env[p].ok ? crossingState(env[p], z0, cot) : 0;
            for (int si = 0; si < ns; ++si) {
              const int pa = starts[si].first, pb = starts[si].second;
              if (st[pa] == 0 || st[pb] == 0)
                continue;
              int holes = 0, bt = 0;
              for (const auto &h : holePos[si]) {
                const int d = st[h.first] == 2;
                holes += d, bt |= d & h.second;
              }
              if ((leadOnly && bt && !startGapOk[si]) || holes > hs)
                continue;
              if (bt && innerOTOnly && (pOT < 0 || st[pOT] == 0 || holes > maxHolesOT))
                continue;
              bool next = false;
              for (int q = pb + 1; q < n && !next; ++q)
                next = env[q].ok && env[q].pix && st[q] > 0;
              if (!next)
                continue;
              elo[si] = std::min(elo[si], eta), ehi[si] = std::max(ehi[si], eta);
            }
          }
        }
        startCot.assign(ns, {1e30f, -1e30f});
        for (int si = 0; si < ns; ++si) {
          if (elo[si] > ehi[si])
            continue;
          const float a = elo[si] - kWiden, b = ehi[si] + kWiden;
          const float slo = a <= 0 ? -1e30f : std::sinh(a), shi = b >= kNe * kDe ? 1e30f : std::sinh(b);
          startCot[si] = side > 0 ? std::make_pair(slo, shi) : std::make_pair(-shi, -slo);
        }
      }
    };  // struct Chain

    // ---- the gap map: where a flat barrel layer has no sensor (between the modules of a ladder along z, between the
    // ladders in phi, beyond the ends), from the modules' outlines
    constexpr float kGapMapDz = 0.005f;  // the z cell of the covered bitmap and of the layer's z gaps, cm
    struct GapLadder {
      float nx, ny;  // the plane's transverse unit normal: nx x + ny y = d
      float d;
      float ux, uy;          // the transverse unit x direction
      float u0, du;          // the modules' u centre and half width
      std::vector<float> z;  // the modules' z intervals, sorted
      float z0 = 0;
      int nz = 0;
      std::vector<unsigned long long> zbits;  // bit k: [z0 + k dz, z0 + (k + 1) dz) inside a module by > margin
      bool covered(float zz) const {
        const int k = (int)std::floor((zz - z0) * (1.0f / kGapMapDz));
        return k >= 0 && k < nz && ((zbits[k >> 6] >> (k & 63)) & 1ull);
      }
    };
    struct GapLayer {
      bool ok = false;
      float rMid = 0;
      float zg0 = 0;
      std::vector<unsigned char> zgap;
      std::vector<GapLadder> ladders;
      std::vector<int> start, lidx;  // the ladders by phi bin of the crossing at rMid
    };
    std::vector<GapLayer> buildGapMap(const MkSeederTrackerDescription &tracker, float margin) {
      constexpr int kNPhi = kGapMapNPhiBins;
      std::vector<GapLayer> layers(tracker.layers.size());
      for (int l = 0; l < (int)tracker.layers.size(); ++l) {
        const MkSeederTrackerLayer &L = tracker.layers[l];
        if (!L.isBarrel || L.nModules == 0)
          continue;
        const std::vector<MkSeederGapMapModule> &modules = tracker.modules[l];
        GapLayer &Y = layers[l];
        bool flat = true;
        double rsum = 0;
        for (int m = 0; m < L.nModules && flat; ++m) {
          const MkSeederGapMapModule &mi = modules[m];
          flat = std::abs(mi.zDirection[2]) < 1e-3f && std::abs(mi.xDirection[2]) < 1e-3f &&
                 std::abs(std::abs(mi.yDirectionZ) - 1) < 1e-3f && mi.isRectangle;
          if (!flat)
            break;
          const float nn = std::hypot(mi.zDirection[0], mi.zDirection[1]),
                      xn = std::hypot(mi.xDirection[0], mi.xDirection[1]);
          const float nx = mi.zDirection[0] / nn, ny = mi.zDirection[1] / nn, ux = mi.xDirection[0] / xn,
                      uy = mi.xDirection[1] / xn;
          const float d = nx * mi.position[0] + ny * mi.position[1], u0 = ux * mi.position[0] + uy * mi.position[1];
          rsum += std::hypot(mi.position[0], mi.position[1]);
          // the module's ladder: the same plane and u interval (to 10 um, and the direction to 1e-4)
          GapLadder *lad = nullptr;
          for (GapLadder &q : Y.ladders) {
            const float s = q.nx * nx + q.ny * ny;  // the normal may be flipped between modules
            const float dd = s > 0 ? d : -d, uu = (q.ux * ux + q.uy * uy) > 0 ? u0 : -u0;
            if (std::abs(std::abs(s) - 1) < 1e-4f && std::abs(q.d - dd) < 1e-3f && std::abs(q.u0 - uu) < 1e-3f &&
                std::abs(q.du - mi.halfWidth) < 1e-3f) {
              lad = &q;
              break;
            }
          }
          if (!lad) {
            Y.ladders.push_back({nx, ny, d, ux, uy, u0, mi.halfWidth, {}, 0, 0, {}});
            lad = &Y.ladders.back();
          }
          lad->z.push_back(mi.position[2] - mi.halfLength);
          lad->z.push_back(mi.position[2] + mi.halfLength);
        }
        if (!flat) {
          Y = GapLayer();
          continue;
        }
        Y.ok = true;
        Y.rMid = rsum / L.nModules;
        for (GapLadder &q : Y.ladders) {
          std::vector<std::pair<float, float>> iv;
          for (size_t k = 0; k < q.z.size(); k += 2)
            iv.push_back({q.z[k], q.z[k + 1]});
          std::sort(iv.begin(), iv.end());
          q.z.clear();
          for (auto &[lo, hi] : iv)
            q.z.push_back(lo), q.z.push_back(hi);
          q.z0 = std::floor((q.z.front() - 1) / kGapMapDz) * kGapMapDz;
          q.nz = (int)std::ceil((q.z.back() + 1 - q.z0) / kGapMapDz);
          q.zbits.assign((q.nz + 63) / 64, 0ull);
          for (int k = 0; k < q.nz; ++k) {
            const float c0 = q.z0 + k * kGapMapDz, c1 = c0 + kGapMapDz;
            for (size_t i = 0; i < q.z.size(); i += 2)
              if (c0 > q.z[i] + margin && c1 < q.z[i + 1] - margin) {
                q.zbits[k >> 6] |= 1ull << (k & 63);
                break;
              }
          }
        }
        // the phi bins a ladder can be crossed in at rMid: its edges' phi widened by 0.05 rad
        std::vector<std::vector<int>> bins(kNPhi);
        for (int k = 0; k < (int)Y.ladders.size(); ++k) {
          const GapLadder &q = Y.ladders[k];
          float p[2];
          for (int e = 0; e < 2; ++e) {
            const float u = q.u0 + (e ? q.du : -q.du);
            p[e] = std::atan2(q.ny * q.d + q.uy * u, q.nx * q.d + q.ux * u);
          }
          float lo = std::min(p[0], p[1]), hi = std::max(p[0], p[1]);
          if (hi - lo > M_PI)
            std::swap(lo, hi), hi += 2 * M_PI;
          lo -= 0.05f, hi += 0.05f;
          const int b0 = (int)std::floor((lo + M_PI) / (2 * M_PI) * kNPhi),
                    b1 = (int)std::floor((hi + M_PI) / (2 * M_PI) * kNPhi);
          for (int b = b0; b <= b1; ++b)
            bins[((b % kNPhi) + kNPhi) % kNPhi].push_back(k);
        }
        {
          float zlo = 1e30f, zhi = -1e30f;
          for (const GapLadder &q : Y.ladders)
            zlo = std::min(zlo, q.z0), zhi = std::max(zhi, q.z0 + q.nz * kGapMapDz);
          Y.zg0 = zlo;
          const int nz = (int)std::ceil((zhi - zlo) / kGapMapDz);
          Y.zgap.assign(nz, 0);
          for (int k = 0; k < nz; ++k) {
            const float zc = zlo + (k + 0.5f) * kGapMapDz;
            for (const GapLadder &q : Y.ladders)
              if (!q.covered(zc)) {
                Y.zgap[k] = 1;
                break;
              }
          }
        }
        Y.start.assign(kNPhi + 1, 0);
        for (int b = 0; b < kNPhi; ++b) {
          Y.start[b + 1] = Y.start[b] + bins[b].size();
          Y.lidx.insert(Y.lidx.end(), bins[b].begin(), bins[b].end());
        }
      }  // for (l : layers)
      return layers;
    }  // buildGapMap()

    // ---- the ES collections
    void fillLayerGeometry(MkSeederESDataHost &data,
                           const MkSeederTrackerDescription &tracker,
                           const std::map<int, BinnedLayer> &binned) {
      const int nLayers = tracker.layers.size();
      auto layers = std::make_shared<MkSeederLayerGeometryHostCollection>(cms::alpakatools::host(), nLayers);
      auto view = layers->view();
      int firstModule = 0;
      for (int l = 0; l < nLayers; ++l) {
        const MkSeederTrackerLayer &li = tracker.layers[l];
        auto row = view[l];
        row.isBinned() = false, row.isDisc() = false;
        row.nQBins() = 0, row.qLastBin() = 0;
        row.qAxisMin() = 0, row.qAxisScale() = 0, row.qAxisLastBinHighEdge() = 0;
        row.phiAxisMin() = 0, row.phiAxisScale() = 0;
        row.isPixel() = li.isPixel, row.isBarrel() = li.isBarrel;
        row.isPixelEnvelope() = isPixelEnvelopeLayer(l);
        row.rIn() = li.rIn, row.rOut() = li.rOut;
        row.zMin() = li.zMin, row.zMax() = li.zMax, row.zMean() = 0.5f * (li.zMin + li.zMax);
        row.firstModule() = firstModule;
        firstModule += li.nModules;
        const auto it = binned.find(l);
        if (it == binned.end())
          continue;
        const BinnedLayer &b = it->second;
        row.isBinned() = true, row.isDisc() = b.disc;
        row.nQBins() = b.nQ, row.qLastBin() = b.q.lastBin;
        row.qAxisMin() = b.q.min, row.qAxisScale() = b.q.scale;
        row.qAxisLastBinHighEdge() = b.q.lastBinHighEdge;
        row.phiAxisMin() = b.phi.min, row.phiAxisScale() = b.phi.scale;
      }
      data.layerGeometry = std::move(layers);
    }

    void fillChain(MkSeederESDataHost &data,
                   const Chain &chain,
                   const MkSeederConfig &config,
                   const std::map<int, BinnedLayer> &binned) {
      const int n = chain.n;
      const int nStarts = chain.starts.size();
      int nHoles = 0;
      for (int si = 0; si < nStarts; ++si)
        nHoles += chain.holePos[si].size();
      auto collection = std::make_shared<MkSeederChainHostCollection>(cms::alpakatools::host(),
                                                                      n,
                                                                      std::max(nStarts, 1),
                                                                      std::max(nHoles, 1),
                                                                      std::max<int>(chain.tables.size(), 1),
                                                                      std::max<int>(chain.etaWindows.size(), 1),
                                                                      std::max<int>(chain.idxC.size(), 1),
                                                                      std::max<int>(chain.idxD.size(), 1));
      auto view = collection->view();
      auto positions = view.positions();
      for (int p = 0; p < n; ++p) {
        const ChainEnvelope &envelope = chain.env[p];
        auto row = positions[p];
        row.layerId() = chain.order[p];
        row.binnedLayerId() = binned.count(chain.order[p]) ? chain.order[p] : -1;
        row.envelopeValid() = envelope.ok, row.isDisc() = envelope.disc, row.isPixel() = envelope.pix;
        row.isExtraOuterTracker() = envelope.xot;
        row.position() = envelope.pos, row.positionLow() = envelope.plo, row.positionHigh() = envelope.phi;
        row.innerLow() = envelope.loIn, row.innerHigh() = envelope.hiIn;
        row.outerLow() = envelope.loOut, row.outerHigh() = envelope.hiOut;
      }
      auto starts = view.starts();
      auto holes = view.holes();
      int hole = 0;
      for (int si = 0; si < nStarts; ++si) {
        auto row = starts[si];
        row.positionA() = chain.starts[si].first, row.positionB() = chain.starts[si].second;
        row.gapAllowed() = chain.startGapOk[si];
        row.gapLayer() = chain.gapLid[si];
        row.cotMin() = chain.startCot[si].first, row.cotMax() = chain.startCot[si].second;
        row.firstHole() = hole, row.nHoles() = chain.holePos[si].size();
        for (const auto &h : chain.holePos[si]) {
          holes[hole].position() = h.first, holes[hole].isBetweenStartHits() = h.second != 0;
          ++hole;
        }
      }
      auto windowTables = view.windowTables();
      for (size_t i = 0; i < chain.tables.size(); ++i) {
        const WindowTable &par = chain.tables[i];
        auto row = windowTables[i];
        row.phiWindowC() = par.phiC, row.qWindowC() = par.qC;
        row.phiWindowA() = par.aphi, row.phiWindowB() = par.bphi, row.qWindowA() = par.aq, row.qWindowB() = par.bq;
        row.referencePathLength() = par.sref, row.firstEtaWindow() = par.e0, row.nEtaWindows() = par.ne;
      }
      auto etaWindows = view.etaWindows();
      for (size_t i = 0; i < chain.etaWindows.size(); ++i) {
        const EtaWindow &eta = chain.etaWindows[i];
        auto row = etaWindows[i];
        row.absEtaLow() = eta.lo, row.absEtaHigh() = eta.hi;
        row.phiWindowA() = eta.aphi, row.phiWindowB() = eta.bphi, row.qWindowA() = eta.aq, row.qWindowB() = eta.bq;
      }
      for (size_t i = 0; i < chain.idxC.size(); ++i)
        view.windowTableByTriplet()[i].windowTable() = chain.idxC[i];
      for (size_t i = 0; i < chain.idxD.size(); ++i)
        view.windowTableByQuad()[i].windowTable() = chain.idxD[i];

      // the chain's constants
      const int side = chain.side > 0 ? 0 : 1;
      MkSeederChainParameters &C = data.chainParameters[side];
      C = MkSeederChainParameters{};
      // the candidates pack their chain positions, holes and inner holes in 6 bits each (MkSeederCandidatesSoA.h)
      if (n > kMaxCandidateChainPositions)
        throw cms::Exception("Configuration")
            << "MkSeederTableBuilder: " << n << " chain positions, at most " << kMaxCandidateChainPositions;
      const float fakeScoreCut = config.fakeScoreCut;
      const bool useShapeCut = config.useShapeCut != 0;
      C.nPositions = n, C.side = chain.side, C.nStartPairs = nStarts;
      C.maxHoles = chain.maxHoles, C.maxHolesForOuterTracker = chain.maxHolesOT;
      C.maxStartHoles = chain.startHoles < 0 ? chain.maxHoles : chain.startHoles;
      C.leadOnly = chain.leadOnly, C.innerOuterTrackerOnly = chain.innerOTOnly;
      C.knownCombinationsOnly = chain.knownOnly, C.alwaysForwardMisses = chain.alwaysForward;
      C.missGapDz = config.missGapDz;
      C.useStageCNet = 0;
      for (int b = 0; b < 5; ++b)
        C.stageCNetCut[b] = config.stageCNetCuts[b], C.useStageCNet |= config.stageCNetCuts[b] > kStageCNetOff;
      C.useGapMap = chain.hasGapMap;
      C.useFakeScore = fakeScoreCut > 0 || useShapeCut, C.useShapeCut = useShapeCut;
      C.emitTriplets = config.emitTriplets;
      C.fakeScoreCut = fakeScoreCut > 0 ? fakeScoreCut : 1e30f;
      C.fakeScoreCutForward = config.fakeScoreCutForward, C.forwardMinAbsCot = std::sinh(config.forwardMinAbsEta);
      C.tripletMinAbsCot = config.tripletMinAbsCot, C.tripletMaxAbsCot = config.tripletMaxAbsCot;
      C.d0Max = chain.P.d0Max, C.ptMin = chain.P.ptMin;
      C.d0MaxOuterTracker = config.d0MaxOuterTrackerThird > 0 ? config.d0MaxOuterTrackerThird : chain.P.d0Max;
      C.allowAnyOuterTracker = chain.otAny, C.maxPixelLayersForExtraOuterTracker = chain.otMaxPix;
      const Windows &params = chain.P;
      C.doubletInverseTwoRadius = 0.003f * 3.8f / (2.0f * params.ptMin);
      C.doubletD0Max = params.d0Max, C.doubletPhiMargin = params.margB;
      C.startPtMin = params.ptMin, C.startD0Max = params.d0Max, C.zVertexHalfWidth = params.zv;
      C.zVertexHalfWidthOuter = config.zVertexHalfWidthOuter, C.zVertexOuterMinRadius = config.zVertexOuterMinRadius;
      C.zVertexOuterMinAbsCot = config.zVertexOuterMinAbsCot, C.zVertexOuterDiscOnly = config.zVertexOuterDiscOnly;
      C.doubletMargin = params.margB;
      C.ot2LayerId = config.ot2WindowScale > 0 && binned.count(6) ? 6 : -1;
      C.ot1LayerId = config.ot1WindowScale > 0 && binned.count(4) ? 4 : -1;
      C.ot2WindowScale = config.ot2WindowScale;
      C.ot2PhiWindowA = config.ot2Windows[0], C.ot2PhiWindowB = config.ot2Windows[1];
      C.ot2QWindowA = config.ot2Windows[2], C.ot2QWindowB = config.ot2Windows[3];
      C.ot2PhiWindowMin = config.ot2PhiWindowMin;
      C.ot2ConfirmationSigma = config.ot2ConfirmationSigma, C.ot2ConfirmationGate = config.ot2ConfirmationGate;
      C.ot2LocalWindowScale = config.ot2LocalWindowScale;
      for (int k = 0; k < 2; ++k)
        C.ot2LocalWindowX[k] = config.extensionWindowX[k], C.ot2LocalWindowY[k] = config.extensionWindowY[k];
      C.ot1WindowScale = config.ot1WindowScale;
      C.ot1MaxAbsCot = config.ot1MaxAbsEta >= 10 ? 1e30f : std::sinh(config.ot1MaxAbsEta);
      C.ot1ConfirmationGate = config.ot1ConfirmationGate;
      C.ot1LocalSigma = config.ot1LocalSigma, C.ot2LocalSigma = config.ot2LocalSigma;
      for (int k = 0; k < 3; ++k)
        C.otLocalSigmaX[k] = k < int(config.otLocalSigmaX.size()) ? config.otLocalSigmaX[k] : 0.f;
      C.ot1LocalSigmaY = config.ot1LocalSigmaY, C.ot2LocalSigmaY = config.ot2LocalSigmaY;
      C.ot2LocalGateSigma = config.ot2LocalGateSigma;
      C.minPixelHits = config.minPixelHits, C.tripletMlpCut = config.tripletMlpCut;
      C.minPixelHitsOuterTrackerThird = config.minPixelHitsOuterTrackerThird;
      C.tripletMlpRescueOuterTrackerHits = config.tripletMlpRescueOuterTrackerHits;
      C.tripletMlpRescueMinScore = config.tripletMlpRescueMinScore;
      C.mergedWindowScale = config.mergedTagMode > 0 && config.mergedWindowScale > 1.f ? config.mergedWindowScale : 0.f;
      C.mergedShapeExempt = config.mergedTagMode > 0 ? config.mergedShapeExempt : 0;
      data.chains[side] = std::move(collection);
    }  // fillChain()

    // the shape bands of the barrel pixel layers
    void fillShapes(MkSeederESDataHost &data, const MkSeederConfig &config) {
      struct Shape {
        float invBinWidth = 0;
        std::vector<int> lo, hi;
      } shapes[kNShapeLayers];
      for (const MkSeederConfig::ShapeWindow &w : config.shapeWindows) {
        Shape &s = shapes[w.layer];
        s.invBinWidth = 1.0f / w.binWidth;
        s.lo = w.low, s.hi = w.high;
      }
      int nBands = 0;
      for (int l = 0; l < kNShapeLayers; ++l)
        nBands += shapes[l].lo.size();
      auto collection =
          std::make_shared<MkSeederShapeHostCollection>(cms::alpakatools::host(), kNShapeLayers, std::max(nBands, 1));
      auto view = collection->view();
      int band = 0;
      for (int l = 0; l < kNShapeLayers; ++l) {
        const Shape &shape = shapes[l];
        auto row = view.layers()[l];
        row.valid() = !shape.lo.empty(), row.firstBin() = band, row.nBins() = shape.lo.size();
        row.invBinWidth() = shape.invBinWidth;
        for (size_t b = 0; b < shape.lo.size(); ++b, ++band)
          view.bands()[band].spanLow() = shape.lo[b], view.bands()[band].spanHigh() = shape.hi[b];
      }
      data.shapes = std::move(collection);
    }

    // the gap map's layers 0-3 (the rays depend on the beam spot: per event, computeGapRays())
    void fillGapMap(MkSeederESDataHost &data, const std::vector<GapLayer> *gapMap, float margin) {
      struct LayerRow {
        bool valid = false;
        float rMid = 0, zGap0 = 0, margin = 0;
        int nZGaps = 0, firstZGap = 0, firstLadder = 0, nLadders = 0, firstPhiStart = 0, firstLadderIndex = 0;
      } layers[kGapMapNLayers];
      std::vector<const GapLadder *> ladders;
      std::vector<int> ladderFirstBit, phiStarts, ladderIndex;
      std::vector<unsigned long long> bits;
      std::vector<unsigned char> zGaps;
      if (gapMap)
        for (int lid = 0; lid < kGapMapNLayers && lid < (int)gapMap->size(); ++lid) {
          const GapLayer &Y = (*gapMap)[lid];
          if (!Y.ok)
            continue;
          LayerRow &layer = layers[lid];
          layer.valid = true, layer.rMid = Y.rMid, layer.zGap0 = Y.zg0, layer.margin = margin;
          layer.nZGaps = Y.zgap.size(), layer.firstZGap = zGaps.size();
          zGaps.insert(zGaps.end(), Y.zgap.begin(), Y.zgap.end());
          layer.firstLadder = ladders.size(), layer.nLadders = Y.ladders.size();
          for (const GapLadder &q : Y.ladders) {
            ladders.push_back(&q);
            ladderFirstBit.push_back(bits.size());
            bits.insert(bits.end(), q.zbits.begin(), q.zbits.end());
          }
          layer.firstPhiStart = phiStarts.size(), layer.firstLadderIndex = ladderIndex.size();
          phiStarts.insert(phiStarts.end(), Y.start.begin(), Y.start.end());
          ladderIndex.insert(ladderIndex.end(), Y.lidx.begin(), Y.lidx.end());
        }
      const int nLadders = ladders.size();
      auto collection = std::make_shared<MkSeederGapMapHostCollection>(cms::alpakatools::host(),
                                                                       kGapMapNLayers,
                                                                       std::max(nLadders, 1),
                                                                       std::max<int>(zGaps.size(), 1),
                                                                       std::max<int>(bits.size(), 1),
                                                                       std::max<int>(phiStarts.size(), 1),
                                                                       std::max<int>(ladderIndex.size(), 1));
      auto view = collection->view();
      for (int lid = 0; lid < kGapMapNLayers; ++lid) {
        const LayerRow &layer = layers[lid];
        auto row = view.layers()[lid];
        row.valid() = layer.valid, row.rMid() = layer.rMid, row.zGap0() = layer.zGap0, row.margin() = layer.margin;
        row.nZGaps() = layer.nZGaps, row.firstZGap() = layer.firstZGap;
        row.firstLadder() = layer.firstLadder, row.nLadders() = layer.nLadders;
        row.firstPhiStart() = layer.firstPhiStart, row.firstLadderIndex() = layer.firstLadderIndex;
      }
      for (int i = 0; i < nLadders; ++i) {
        const GapLadder &q = *ladders[i];
        auto row = view.ladders()[i];
        row.normalX() = q.nx, row.normalY() = q.ny, row.distance() = q.d;
        row.alongX() = q.ux, row.alongY() = q.uy, row.alongOffset() = q.u0, row.halfWidth() = q.du;
        row.zFirstBin() = q.z0, row.nZBins() = q.nz, row.firstBitWord() = ladderFirstBit[i];
      }
      for (size_t i = 0; i < zGaps.size(); ++i)
        view.zGaps()[i].inGap() = zGaps[i];
      for (size_t i = 0; i < bits.size(); ++i)
        view.bits()[i].word() = bits[i];
      for (size_t i = 0; i < phiStarts.size(); ++i)
        view.phiStarts()[i].firstEntry() = phiStarts[i];
      for (size_t i = 0; i < ladderIndex.size(); ++i)
        view.ladderIndex()[i].ladder() = ladderIndex[i];
      data.gapMap = std::move(collection);
    }

    // the patterns' positions, sorted by their packed layers (the cleaning's binary search)
    void fillPatterns(MkSeederESDataHost &data, const std::map<std::array<int, 4>, int> &patternIndex) {
      int nPositions = 0;
      for (const auto &kv : patternIndex)
        nPositions = std::max(nPositions, kv.second + 1);
      std::vector<std::pair<uint32_t, int>> table;
      for (const auto &[layers, index] : patternIndex) {
        uint32_t key = 0;
        for (int k = 0; k < 4; ++k)
          key = key << 8 | uint32_t(uint8_t(int16_t(layers[k]) + 1));
        table.push_back({key, index});
      }
      std::sort(table.begin(), table.end());
      auto collection =
          std::make_shared<MkSeederPatternsHostCollection>(cms::alpakatools::host(), std::max<int>(table.size(), 1));
      auto view = collection->view();
      for (size_t i = 0; i < table.size(); ++i)
        view[i].layerKey() = table[i].first, view[i].patternIndex() = table[i].second;
      view.nPatternPositions() = nPositions;
      data.nPatternPositions = nPositions, data.nPatternEntries = table.size();
      data.patterns = std::move(collection);
    }

  }  // namespace

  void buildMkSeederTables(const MkSeederConfig &config,
                           const MkSeederTrackerDescription &tracker,
                           double qBinScale,
                           MkSeederESDataHost &tables,
                           MkSeederHostTables &hostTables) {
    const int nLayers = tracker.layers.size();
    auto checkLayer = [nLayers](int l) {
      if (l < 0 || l >= nLayers)
        throw cms::Exception("Configuration") << "MkSeederTableBuilder: no mkFit layer " << l;
      return l;
    };
    // the patterns, each followed by its mirror if it has +z discs (the mirror keeps the windows); their order
    std::vector<MkSeederConfig::Pattern> patterns;
    for (const MkSeederConfig::Pattern &p : config.patterns) {
      patterns.push_back(p);
      MkSeederConfig::Pattern m = p;
      bool has = false;
      for (int &l : m.layers)
        if (l >= 16 && l <= 27)
          l += 22, has = true;
      if (has)
        patterns.push_back(m);
    }
    std::map<std::array<int, 4>, int> patternIndex;
    for (size_t i = 0; i < patterns.size(); ++i)
      patternIndex.insert({patterns[i].layers, (int)i});
    // the binned layers: the patterns', every one the chain can visit, OT1-P and OT2-P for the OT checks
    std::map<int, BinnedLayer> binned;
    auto addLayer = [&](int l, double qBin) {
      if (!binned.count(l))
        binned.emplace(l, BinnedLayer(l, tracker.layers[checkLayer(l)], qBin));
    };
    auto qBinOf = [&](int l) { return (tracker.layers[checkLayer(l)].isBarrel ? 2.0 : 1.0) * qBinScale; };
    for (const MkSeederConfig::Pattern &p : patterns)
      for (int l : p.layers)
        addLayer(l, qBinOf(l));
    std::set<int> have;
    for (const MkSeederConfig::Pattern &p : patterns)
      for (int l : p.layers)
        have.insert(l);
    for (int l : {0, 1, 2, 3})
      have.insert(l);
    for (int l = 16; l <= 27; ++l)
      have.insert(l), have.insert(l + 22);
    for (int l : config.chainOuterTrackerLayers)
      have.insert(l);
    for (int l : have)
      addLayer(l, qBinOf(l));
    if (config.ot1WindowScale > 0)
      addLayer(4, 2.0 * qBinScale);
    if (config.ot2WindowScale > 0)
      for (int l : {4, 6})
        addLayer(l, 2.0 * qBinScale);
    // the envelopes and the gap map
    Envelopes envelopes;
    envelopes.setup(tracker, have);
    envelopes.delta = config.crossingMargin;
    std::unique_ptr<std::vector<GapLayer>> gapMap;
    std::vector<bool> gapMapLayer(nLayers, false);
    if (config.gapMapMargin >= 0) {
      gapMap = std::make_unique<std::vector<GapLayer>>(buildGapMap(tracker, config.gapMapMargin));
      for (int l = 0; l < nLayers; ++l)
        gapMapLayer[l] = (*gapMap)[l].ok;
    }
    // the two chains: the parameters, the options, the window tables
    Windows P;
    P.ptMin = config.ptMin, P.d0Max = config.d0Max, P.zv = config.zVertexHalfWidth, P.margB = config.doubletPhiMargin;
    P.zvScan = std::max(config.zVertexHalfWidth, config.zVertexHalfWidthOuter);
    P.phiC = config.phiWindowC, P.qC = config.qWindowC, P.phiD = config.phiWindowD, P.qD = config.qWindowD;
    const double ws = config.windowScale;
    auto windowsOf = [&](const MkSeederConfig::Pattern &p) {
      Windows Q = P;
      if (p.windows[0] >= 0)
        Q.phiC = p.windows[0], Q.qC = p.windows[1], Q.phiD = p.windows[2], Q.qD = p.windows[3];
      Q.bPhiD = p.dWindowSlopes[0], Q.bQD = p.dWindowSlopes[1];
      Q.sRef = p.referencePathLength;
      for (const MkSeederConfig::EtaWindow &e : p.etaWindows) {
        EtaWindow w{e.absEtaLow, e.absEtaHigh, e.phiWindowA, e.phiWindowB, e.qWindowA, e.qWindowB};
        w.aphi *= ws, w.bphi *= ws, w.aq *= ws, w.bq *= ws;
        Q.addEtaWindow(w);
      }
      Q.phiC *= ws, Q.qC *= ws, Q.phiD *= ws, Q.qD *= ws;
      Q.bPhiD *= ws, Q.bQD *= ws;
      return Q;
    };
    Chain chains[2];
    for (int sd = 0; sd < 2; ++sd) {
      Chain &C = chains[sd];
      C.P = P;
      C.P.phiC *= ws, C.P.qC *= ws, C.P.phiD *= ws, C.P.qD *= ws;
      C.maxHoles = config.maxHoles;
      C.alwaysForward = config.alwaysForwardMisses;
      C.maxHolesOT = config.maxHolesForOuterTracker;
      C.knownOnly = !config.anyCombination;
      C.startHoles = config.maxStartHoles;
      C.leadOnly = config.leadOnly;
      C.innerOTOnly = config.innerOuterTrackerOnly;
      C.startGap = config.startGapMode;
      C.chainOT = config.chainOuterTrackerLayers, C.otAny = config.allowAnyOuterTracker;
      C.otMaxPix = config.maxPixelLayersForExtraOuterTracker;
      C.hasGapMap = gapMap != nullptr;
      C.setup(envelopes, sd == 0 ? 1 : -1, have);
    }
    for (const MkSeederConfig::Pattern &p : patterns) {
      const Windows Q = windowsOf(p);
      for (Chain &C : chains) {
        auto &wc = C.winC[{p.layers[0], p.layers[1], p.layers[2]}];
        wc.first = std::max(wc.first, Q.phiC), wc.second = std::max(wc.second, Q.qC);
        C.winD[p.layers] = {Q.phiD, Q.bPhiD, Q.qD, Q.bQD, Q.sRef, std::vector<EtaWindow>(Q.eta, Q.eta + Q.nEta), Q.qC};
      }
    }
    // the pruned combinations, each also mirrored if it has +z discs
    for (const std::vector<int> &q : config.prunedCombinations)
      for (int mirror = 0; mirror < 2; ++mirror) {
        std::vector<int> l = q;
        bool has = false;
        for (int &x : l)
          if (x >= 16 && x <= 27)
            x += 22 * mirror, has = true;
        if (mirror && !has)
          continue;
        for (Chain &C : chains) {
          if (l.size() == 3)
            C.prunedC.insert({l[0], l[1], l[2]});
          else
            C.prunedD.insert({l[0], l[1], l[2], l[3]});
        }
      }
    for (Chain &C : chains)
      C.setupFinder(envelopes, gapMapLayer);

    // the tables
    tables.nLayers = nLayers;
    fillLayerGeometry(tables, tracker, binned);
    for (const Chain &C : chains)
      fillChain(tables, C, config, binned);
    fillShapes(tables, config);
    fillGapMap(tables, gapMap.get(), config.gapMapMargin);
    fillPatterns(tables, patternIndex);

    // the host tables: the binned layers' axes for the hits' sort keys
    hostTables.binnedLayers.clear();
    for (const auto &[id, b] : binned) {
      MkSeederLayerBinning lb;
      lb.layerId = id, lb.isDisc = b.disc, lb.nQBins = b.nQ;
      lb.phiAxisMin = b.phi.min, lb.phiFineScale = b.phi.fineScale;
      lb.qAxisMin = b.q.min, lb.qFineScale = b.q.fineScale, lb.qFineLastBinHighEdge = b.q.fineLastBinHighEdge;
      lb.qLastFineBin = b.q.lastFineBin;
      lb.qbarLowNominal = b.qbarLo, lb.qbarHighNominal = b.qbarHi, lb.qLowNominal = b.qLo, lb.qHighNominal = b.qHi;
      hostTables.binnedLayers.push_back(lb);
    }
  }  // buildMkSeederTables()

  namespace {
    // a hit's beam-relative azimuth (mkFit's getPhi(), Hit::phi()) as one scalar out-of-line call: whether the
    // compiler inlines and vectorises it in the loop decides the last bit of phi, and so the binning and the seeds
    __attribute__((noinline)) float hitPhi(float x, float y) { return vdt::fast_atan2f(y, x); }
  }  // namespace

  void computeHitBinning(const MkSeederLayerBinning &layer,
                         const MkSeederBeamSpot &bs,
                         uint32_t n,
                         const float *gxs,
                         const float *gys,
                         const float *gzs,
                         const MkSeederHitBinningOutput &o,
                         double ext[4]) {
    // the fine bins (binnor's from_R_to_M_bin_safe of the axes: unsigned short bins, the phi bin wraps)
    using Bin = unsigned short;
    auto phiFineBin = [&layer](float r) -> unsigned int {
      return Bin(Bin(std::floor((r - layer.phiAxisMin) * layer.phiFineScale)) & Bin((1 << kFineBits) - 1));
    };
    auto qFineBin = [&layer](float r) -> unsigned int {
      return r <= layer.qAxisMin                 ? Bin(0)
             : (r >= layer.qFineLastBinHighEdge) ? layer.qLastFineBin
                                                 : Bin(std::floor((r - layer.qAxisMin) * layer.qFineScale));
    };
    // scalar cos / sin (no vector variants)
#pragma GCC novector
    for (uint32_t k = 0; k < n; ++k) {
      const float gx = gxs[k], gy = gys[k], gz = gzs[k];
      const float dz = gz - bs.z;
      const float x = gx - (bs.x + bs.dxdz * dz), y = gy - (bs.y + bs.dydz * dz);
      const float phi = hitPhi(x, y);
      const float r = std::sqrt(x * x + y * y);
      const unsigned int b1 = phiFineBin(phi);
      const unsigned int b2 = qFineBin(layer.isDisc ? r : gz);
      o.key[k] = ((b2 << kFineBits) | b1) & ~(((1u << kQFineShift) - 1) << kFineBits);
      o.phi[k] = phi, o.r[k] = r;
      o.invR[k] = 1.0f / r;
      const double u = layer.isDisc ? gz : r, v = layer.isDisc ? r : gz;
      ext[0] = std::min(ext[0], u), ext[1] = std::max(ext[1], u);
      ext[2] = std::min(ext[2], v), ext[3] = std::max(ext[3], v);
    }
    // x and y in loops of their own: cosf and sinf (one loop could let the compiler merge them into sincosf, whose
    // sin differs from sinf's in the last bit for a few hits)
#pragma GCC novector
    for (uint32_t k = 0; k < n; ++k)
      o.xBeam[k] = o.r[k] * std::cos(o.phi[k]);
#pragma GCC novector
    for (uint32_t k = 0; k < n; ++k)
      o.yBeam[k] = o.r[k] * std::sin(o.phi[k]);
  }

  void computeGapRays(const MkSeederGapMapSoA::ConstView &gapMap,
                      int lid,
                      float bx,
                      float by,
                      int nphi,
                      float tolU,
                      float *r1,
                      float *r2,
                      unsigned char *edge) {
    constexpr int kNPhi = kGapMapNPhiBins;
    const auto layer = gapMap.layers()[lid];
    const float rMid = layer.rMid();
    const auto ladders = gapMap.ladders();
    const auto phiStarts = gapMap.phiStarts();
    const auto ladderIndex = gapMap.ladderIndex();
    const int firstLadder = layer.firstLadder(), firstPhiStart = layer.firstPhiStart(),
              firstLadderIndex = layer.firstLadderIndex();
    for (int i = 0; i < nphi; ++i) {
      const float ph = -(float)M_PI + (i + 0.5f) * (2 * (float)M_PI / nphi);
      const float cx = std::cos(ph), cy = std::sin(ph);
      // the map's phi bin of the ray at rMid (global phi; the beam offset is far inside the 0.05 rad widening)
      int bin = (int)((std::atan2(by + rMid * cy, bx + rMid * cx) + (float)M_PI) * (kNPhi / (2 * (float)M_PI)));
      bin = std::min(std::max(bin, 0), kNPhi - 1);
      float lo = 1e30f, hi = -1e30f;
      bool nearEdge = false;
      for (int k = phiStarts[firstPhiStart + bin].firstEntry(); k < phiStarts[firstPhiStart + bin + 1].firstEntry();
           ++k) {
        const auto q = ladders[firstLadder + ladderIndex[firstLadderIndex + k].ladder()];
        const float qnx = q.normalX(), qny = q.normalY(), qd = q.distance(), qux = q.alongX(), quy = q.alongY(),
                    qu0 = q.alongOffset(), qdu = q.halfWidth();
        const float den = qnx * cx + qny * cy;
        if (std::abs(den) < 0.1f)
          continue;
        const float r = (qd - qnx * bx - qny * by) / den;
        if (r <= 0)
          continue;
        const float u = std::abs(qux * (bx + r * cx) + quy * (by + r * cy) - qu0);
        if (u > qdu + tolU)
          continue;
        if (u > qdu - tolU)
          nearEdge = true;
        lo = std::min(lo, r), hi = std::max(hi, r);
      }
      if (lo > hi)
        nearEdge = true, lo = hi = rMid;
      r1[i] = lo, r2[i] = hi, edge[i] = nearEdge;
    }
  }

}  // namespace mkseeder

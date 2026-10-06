// K1: device seed hand-off kernels.
#include <cstdio>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/LSTCore/interface/Common.h"

#include "MkFitAlpakaSeedHandoffK1Kernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::k1 {

  using namespace cms::alpakatools;

  namespace {
    // the working hit list of one TC (pixel seed list + LST OT hits); 64 pixel hits (kMaxTrkHits) + 22 OT slots
    constexpr int kListCap = 64;
    struct ListHit {
      ::mkfitdev::HitOnTrack hot;
      int32_t det;  // index in the det table
    };

    struct KernelK1Rows {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::lst::TrackCandidatesBaseConst tcs,
                                    uint32_t tcCapacity,
                                    ::mkfitdev::TrackSoAConstView states,
                                    ::mkfitdev::HitSoAConstView mk,
                                    DetInfo const* dets,
                                    int32_t const* layerBase,
                                    int8_t const* layerIsPixel,
                                    K1Config cfg,
                                    ::mkfitdev::TrackSoAView rows,
                                    RowInfo* info,
                                    HostReq* req,
                                    int32_t* nReq,
                                    int32_t* counters,
                                    int32_t* nTCOut) const {
        const uint32_t nTC = tcs.nTrackCandidates() < tcCapacity ? tcs.nTrackCandidates() : tcCapacity;
        if (cms::alpakatools::once_per_grid(acc))
          *nTCOut = int32_t(nTC);
        const uint32_t nPixel = mk.nPixel();
        const uint32_t nRows = mk.metadata().size();
        // det table index of the hit (layer, index); -1: no mkFit row / layer / module
        auto detOf = [&](int layer, int index) -> int32_t {
          if (layer < 0 || layer >= cfg.nLayers || index < 0)
            return -1;
          const uint32_t row = layerIsPixel[layer] ? uint32_t(index) : nPixel + uint32_t(index);
          if (row >= nRows || mk[row].layer() != layer)
            return -1;
          const int32_t d = layerBase[layer] + int32_t(::mkfitdev::hitpack::detIDinLayer(mk[row].packed()));
          return d < layerBase[layer + 1] ? d : -1;
        };
        for (uint32_t r : uniform_elements(acc, nTC)) {
          RowInfo& ri = info[r];
          ri.pixTrack = -1;
          ri.nTot = 0;
          ri.kind = kSkip;
          ri.deviceFit = 0;
          rows[r].nTotalHits() = 0;
          const auto type = tcs.trackCandidateType()[r];
          const bool isT5orT4 = type == ::lst::LSTObjType::T5 || type == ::lst::LSTObjType::T4;
          ListHit list[kListCap];
          int n = 0;
          int nPix = 0;
          int32_t e = -1;
          bool err = false;
          if (!isT5orT4) {
            e = int32_t(tcs.pixelSeedIndex()[r]);
            if (tcs.pixelSeedIndex()[r] >= uint32_t(cfg.nPixTracks))
              continue;  // as LSTOutputConverter: no pixel seed, the TC is skipped
            const int np = e < cfg.nStates ? states[e].nTotalHits() : 0;
            if (np < 2) {
              ri.kind = kError;  // no device pixel seed list for a referenced pixel track
              alpaka::atomicAdd(acc, &counters[1], 1, alpaka::hierarchy::Blocks{});
              continue;
            }
            const int nUse = (np > 3 && !cfg.includeFourthHit) ? 3 : np;
            for (int k = 0; k < nUse; ++k) {
              const auto h = states[e].hits().hot[k];
              const int32_t d = detOf(h.layer, h.index);
              err = err || d < 0;
              list[n++] = ListHit{h, d};
            }
            nPix = n;
          }
          if (err) {
            ri.kind = kError;
            alpaka::atomicAdd(acc, &counters[1], 1, alpaka::hierarchy::Blocks{});
            continue;
          }
          // a host request for this row (slot by atomic counter; the entry carries the row, so the order is irrelevant)
          auto request = [&](int kind, int32_t pe, ListHit const* hs, int nh) {
            const int32_t slot = alpaka::atomicAdd(acc, nReq, 1, alpaka::hierarchy::Blocks{});
            if (slot >= kMaxHostReq || nh > kReqHits) {
              alpaka::atomicAdd(acc, &counters[0], 1, alpaka::hierarchy::Blocks{});
              ri.kind = kError;
              return;
            }
            HostReq& q = req[slot];
            q.tc = int32_t(r);
            q.e = pe;
            q.kind = kind;
            q.nHot = nh;
            for (int k = 0; k < nh; ++k)
              q.hot[k] = hs[k].hot;
            ri.kind = int8_t(kind);
          };
          auto writeHits = [&](ListHit const* hs, int nh) {
            const int nw = nh < ::mkfitdev::kMaxSeedHits ? nh : ::mkfitdev::kMaxSeedHits;
            for (int k = 0; k < nw; ++k)
              rows[r].hits().hot[k] = hs[k].hot;
            rows[r].nTotalHits() = int16_t(nw);
            ri.nTot = int16_t(nh);
            if (nh > ::mkfitdev::kMaxSeedHits)
              alpaka::atomicAdd(acc, &counters[2], 1, alpaka::hierarchy::Blocks{});
          };
          // the pixel seed itself (pLS, or a pT3 / pT5 without a selected hit): the device pixel state, else the host
          auto pixelSeed = [&]() {
            writeHits(list, nPix);
            ri.pixTrack = e;
            if (states[e].charge() != 0) {
              rows[r].params() = states[e].params();
              rows[r].errors() = states[e].errors();
              rows[r].charge() = states[e].charge();
              ri.kind = kDevice;
            } else {
              request(kHostPixState, e, list + nPix - 1, 1);
            }
          };
          if (type == ::lst::LSTObjType::pLS) {
            int nIT = 0;
            for (int k = 0; k < nPix; ++k)
              nIT += !dets[list[k].det].isOT;
            const bool drop = cfg.dropOTHitsPurePLS && dets[list[nPix - 1].det].isOT && nIT <= cfg.maxITHitsToDrop;
            if (!drop) {
              pixelSeed();
              continue;
            }
            // the creator refit on the IT hits (host); the seed's hits are those IT hits in list order
            ListHit it[kReqHits];
            int ni = 0;
            for (int k = 0; k < nPix && ni < kReqHits; ++k)
              if (!dets[list[k].det].isOT)
                it[ni++] = list[k];
            writeHits(it, ni);
            request(kHostRefit, e, it, ni);
            continue;
          }
          // T5 / T4 / pT3 / pT5, hits in LSTOutputConverter's order (equal keys as DEVIATIONS DEV-8):
          //  - the pixel seed's IT hits, barrel then endcap, each in the list's radius order;
          //  - the OT hits (the pixel seed's, then the LST OT hits not already present = same OT cluster) placed in the
          //    slot of their module's LST logical layer (module table), slots in layer order (barrel 1-6, endcap 7-11);
          //  - within a slot by rank on (subdetector, span key), equal keys in arrival order.
          ListHit slotHits[kOTSlots][kSlotCapacity];
          int slotN[kOTSlots];
          for (int b = 0; b < kOTSlots; ++b)
            slotN[b] = 0;
          auto place = [&](ListHit const& h) {
            const int b = dets[h.det].slot;
            if (b <= 0 || b >= kOTSlots || slotN[b] >= kSlotCapacity) {
              err = true;
              return;
            }
            slotHits[b][slotN[b]++] = h;
          };
          auto isDupOT = [&](uint32_t key) {
            for (int b = 1; b < kOTSlots; ++b)
              for (int k = 0; k < slotN[b]; ++k)
                if (uint32_t(slotHits[b][k].hot.index) == key)
                  return true;
            return false;
          };
          {
            ListHit pix[kListCap];
            for (int k = 0; k < nPix; ++k)
              pix[k] = list[k];
            n = 0;
            for (int k = 0; k < nPix; ++k)
              if (!dets[pix[k].det].isOT && dets[pix[k].det].barrel)
                list[n++] = pix[k];
            for (int k = 0; k < nPix; ++k)
              if (!dets[pix[k].det].isOT && !dets[pix[k].det].barrel)
                list[n++] = pix[k];
            for (int k = 0; k < nPix && !err; ++k)
              if (dets[pix[k].det].isOT)
                place(pix[k]);
          }
          for (int slot = ::lst::Params_TC::kPixelLayerSlots; slot < ::lst::Params_TC::kLayers && !err; ++slot) {
            for (int hs = 0; hs < ::lst::Params_TC::kHitsPerLayer && !err; ++hs) {
              const uint32_t key = tcs.hitIndices()[r][slot][hs];
              if (key == ::lst::kTCEmptyHitIdx || isDupOT(key))
                continue;
              const uint32_t row = nPixel + key;
              const int layer = row < nRows ? mk[row].layer() : -1;
              const int32_t d = detOf(layer, int(key));
              if (d < 0) {
                err = true;
                break;
              }
              place(ListHit{::mkfitdev::HitOnTrack{int(key), layer}, d});
            }
          }
          for (int b = 1; b < kOTSlots && !err; ++b) {
            if (n + slotN[b] > kListCap) {
              err = true;
              break;
            }
            // DEVIATION DEV-8: equal keys in input order for any list length
            for (int k = 0; k < slotN[b]; ++k) {
              DetInfo const& dk = dets[slotHits[b][k].det];
              int rank = 0;
              for (int j = 0; j < slotN[b]; ++j) {
                DetInfo const& dj = dets[slotHits[b][j].det];
                const bool equal = dj.sub == dk.sub && dj.sortKey == dk.sortKey;
                rank += dj.sub < dk.sub || (dj.sub == dk.sub && dj.sortKey < dk.sortKey) || (equal && j < k);
              }
              list[n + rank] = slotHits[b][k];
            }
            n += slotN[b];
          }
          if (err) {
            ri.kind = kError;
            alpaka::atomicAdd(acc, &counters[1], 1, alpaka::hierarchy::Blocks{});
            continue;
          }
          // LSTOutputConverter's selections, in place (the selected hits keep their order)
          int ns = 0;
          int32_t firstLayer = -1;
          for (int k = 0; k < n; ++k) {
            DetInfo const& di = dets[list[k].det];
            if (type == ::lst::LSTObjType::T5 && ns < 2 && di.type != 1)
              continue;  // the first two should be P
            if (type == ::lst::LSTObjType::T4) {
              if (ns == 0)
                firstLayer = di.topoLayer;
              else if (di.topoLayer == firstLayer && di.type == 2)
                continue;
            }
            list[ns++] = list[k];
          }
          if (ns == 0) {
            if (!isT5orT4)
              pixelSeed();  // never reached with pixel hits (kept for LSTOutputConverter's structure)
            continue;
          }
          writeHits(list, ns);
          if (cfg.deviceFitStates && ns >= 2) {
            // the light seeds' zero state (charge 1, unit diagonal), replaced by the build module's seed fit; a pT3 /
            // pT5 whose fit fails takes its pixel seed (pixelTrackOfSeed)
            ri.pixTrack = e;
            for (int k = 0; k < 6; ++k)
              rows[r].params().v[k] = 0.f;
            for (int k = 0; k < 21; ++k)
              rows[r].errors().v[k] = 0.f;
            for (int k = 0; k < 6; ++k)
              rows[r].errors().v[::mkfitdev::symIdx6(k, k)] = 1.f;
            rows[r].charge() = 1;
            ri.kind = kDevice;
            ri.deviceFit |= 1;
            continue;
          }
          // light seed: the placeholder on the last hit (host: the OT CPE, or the pixel track's rechit)
          if (dets[list[ns - 1].det].isOT)
            request(kHostOTState, -1, list + ns - 1, 1);
          else
            request(kHostPixState, e, list + ns - 1, 1);
          ri.pixTrack = -1;
        }
      }
    };

    struct KernelK1Compact {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::mkfitdev::TrackSoAConstView rows,
                                    int32_t const* outIndex,
                                    uint32_t nTC,
                                    ::mkfitdev::TrackSoAView out) const {
        for (uint32_t r : uniform_elements(acc, nTC)) {
          const int32_t o = outIndex[r];
          if (o < 0)
            continue;
          out[o].params() = rows[r].params();
          out[o].errors() = rows[r].errors();
          out[o].charge() = rows[r].charge();
          out[o].chi2() = 0.f;
          out[o].score() = 0.f;
          out[o].label() = o;
          const int nh = rows[r].nTotalHits();
          out[o].nTotalHits() = int16_t(nh);
          out[o].nFoundHits() = int16_t(nh);
          for (int k = 0; k < nh; ++k)
            out[o].hits().hot[k] = rows[r].hits().hot[k];
        }
      }
    };

    struct KernelK1HostStates {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    HostState const* hs,
                                    int32_t n,
                                    int32_t nOverflowHits,
                                    int32_t nOut,
                                    ::mkfitdev::TrackSoAView out) const {
        if (cms::alpakatools::once_per_grid(acc)) {
          out.nTracks() = nOut;
          out.nOverflowTracks() = 0;
          out.nOverflowHits() = nOverflowHits;
        }
        for (int32_t i : uniform_elements(acc, n)) {
          const int32_t o = hs[i].row;
          for (int k = 0; k < 6; ++k)
            out[o].params().v[k] = hs[i].par[k];
          for (int k = 0; k < 21; ++k)
            out[o].errors().v[k] = hs[i].err[k];
          out[o].charge() = hs[i].charge;
        }
      }
    };

    struct KernelSeedsFromK1 {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::mkfitdev::TrackSoAConstView in,
                                    int32_t n,
                                    uint32_t status,
                                    ::mkfitdev::SeedSoAView v) const {
        if (cms::alpakatools::once_per_grid(acc)) {
          v.nSeeds() = n;
          v.nKept() = 0;
          for (int k = 0; k < ::mkfitdev::kNSeedRegions; ++k) {
            v.regionEnd().v[k] = 0;
            v.minLastLayer().v[k] = 9999;  // MkBuilder::begin_event initial values (as packSeeds)
            v.maxLastLayer().v[k] = 0;
          }
          v.nOverflowHits() = in.nOverflowHits();
        }
        for (int32_t i : uniform_elements(acc, n)) {
          v[i].params() = in[i].params();
          v[i].errors() = in[i].errors();
          v[i].charge() = in[i].charge();
          v[i].label() = in[i].label();
          v[i].chi2() = in[i].chi2();
          v[i].status() = status;
          const int nh = in[i].nTotalHits();
          v[i].nHits() = int16_t(nh);
          for (int k = 0; k < nh; ++k)
            v[i].hits().hot[k] = in[i].hits().hot[k];
          v[i].lastX() = 0.f;  // as packSeedsToDevice: importSeeds reads the last-hit position from the EventOfHits
          v[i].lastY() = 0.f;
          v[i].lastZ() = 0.f;
        }
      }
    };
  }  // namespace

  void launchK1Rows(Queue& queue,
                    ::lst::TrackCandidatesBaseConst tcs,
                    uint32_t tcCapacity,
                    ::mkfitdev::TrackSoAConstView states,
                    ::mkfitdev::HitSoAConstView mk,
                    DetInfo const* dets,
                    int32_t const* layerBase,
                    int8_t const* layerIsPixel,
                    K1Config cfg,
                    ::mkfitdev::TrackSoAView rows,
                    RowInfo* info,
                    HostReq* req,
                    int32_t* nReq,
                    int32_t* counters,
                    int32_t* nTCOut) {
    constexpr uint32_t kBlock = 64;  // a 96-entry local hit list per thread
    const uint32_t blocks = divide_up_by(tcCapacity > 0 ? tcCapacity : 1u, kBlock);
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocks, kBlock),
                        KernelK1Rows{},
                        tcs,
                        tcCapacity,
                        states,
                        mk,
                        dets,
                        layerBase,
                        layerIsPixel,
                        cfg,
                        rows,
                        info,
                        req,
                        nReq,
                        counters,
                        nTCOut);
  }

  void launchK1Compact(Queue& queue,
                       ::mkfitdev::TrackSoAConstView rows,
                       int32_t const* outIndex,
                       uint32_t nTC,
                       HostState const* hostStates,
                       int32_t nHostStates,
                       int32_t nOverflowHits,
                       int32_t nOut,
                       ::mkfitdev::TrackSoAView out) {
    constexpr uint32_t kBlock = 128;
    if (nTC > 0)
      alpaka::exec<Acc1D>(
          queue, make_workdiv<Acc1D>(divide_up_by(nTC, kBlock), kBlock), KernelK1Compact{}, rows, outIndex, nTC, out);
    const uint32_t nh = nHostStates > 0 ? uint32_t(nHostStates) : 1u;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(nh, kBlock), kBlock),
                        KernelK1HostStates{},
                        hostStates,
                        nHostStates,
                        nOverflowHits,
                        nOut,
                        out);
  }

  void launchSeedsFromK1(
      Queue& queue, ::mkfitdev::TrackSoAConstView in, int32_t n, uint32_t status, ::mkfitdev::SeedSoAView seeds) {
    constexpr uint32_t kBlock = 128;
    const uint32_t nb = n > 0 ? uint32_t(n) : 1u;
    alpaka::exec<Acc1D>(
        queue, make_workdiv<Acc1D>(divide_up_by(nb, kBlock), kBlock), KernelSeedsFromK1{}, in, n, status, seeds);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::k1

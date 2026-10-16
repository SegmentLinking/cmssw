// The converter's missing-hit navigation on the device.
// Kernel 1, one thread per track: the start state from the fit row (navdev::startFromFitRow), the
// compatible-layer sets of both directions (SimpleNavigationSchool::compatibleLayers on the static candidate lists)
// from the layers of the first and last valid hit of the row, and the compacted call list: one entry (the
// result row) per (track, direction, layer) the device searches, one list per layer kind. Kernel 2, one per kind, one
// thread per call: the compatibleDets search (NavSearch.h) of that layer, inlined, written by row (no order dependence).
// The host looks up its own (track, direction, layer) calls and searches every call the device did not compute; it takes
// the device set as its compatibleLayers list only where the header row says the set is exact and starts at the
// host's own start layer.
#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkFitAlpakaNavDeviceKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::navdevice {
  using namespace cms::alpakatools;
  namespace nd = ::mkfitdev::navdev;
  namespace nav = ::mkfitdev::nav;

  namespace {
    struct KernelNavStart {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::mkfitdev::TrackSoAConstView trk,
                                    int capacity,
                                    NavDeviceTables tables,
                                    nd::DetTestStart* __restrict__ starts,
                                    int* __restrict__ calls,
                                    int* __restrict__ nCalls,
                                    ::mkfitdev::NavResultSoAView out) const {
        const int n = trk.nTracks();
        const int nL = tables.nLayers;
        const int header = 2 * capacity * nL;  // the set header rows (NavResultSoA.h)
        for (int t : uniform_elements(acc, capacity)) {
          for (int r = 2 * t * nL; r < 2 * (t + 1) * nL; ++r)
            out[r].det() = ::mkfitdev::kNavResultNotComputed;
          out[header + 2 * t].det() = ::mkfitdev::kNavResultNotComputed;
          out[header + 2 * t + 1].det() = ::mkfitdev::kNavResultNotComputed;
          if (t >= n)
            continue;
          nd::DetTestStart s;
          if (!nd::startFromFitRow(trk[t].params().v, trk[t].errors().v, trk[t].charge(), s))
            continue;  // outside the closed-form field volume: the host navigates this track
          starts[t] = s;
          int first = -1, last = -1;
          const int nh = trk[t].nTotalHits();
          for (int i = 0; i < nh; ++i) {
            const auto hot = trk[t].hits().hot[i];
            if (hot.index < 0)
              continue;
            if (first < 0)
              first = hot.layer;
            last = hot.layer;
          }
          nav::TrackIn in;
          in.x = s.t.x.x, in.y = s.t.x.y, in.z = s.t.x.z;
          in.px = s.t.p.x, in.py = s.t.p.y, in.pz = s.t.p.z;
          in.charge = s.t.charge;
          in.bz = s.b.z;
          in.start[0] = first >= 0 && first < tables.nMkFit ? tables.mkFitToLayer[first] : -1;
          in.start[1] = last >= 0 && last < tables.nMkFit ? tables.mkFitToLayer[last] : -1;
          for (int d = 0; d < 2; ++d) {
            if (in.start[d] < 0)
              continue;
            bool marginal = false;
            const uint64_t set = nav::compatibleLayers(*tables.layers, in.start[d], in, d == 1, marginal);
            out[header + 2 * t + d].det() = marginal ? int32_t(::mkfitdev::kNavResultHost) : in.start[d];
            for (int l = 0; l < nL; ++l) {
              if (!((set >> l) & 1))
                continue;
              const int kind = tables.layerFlat[3 * l], first = tables.layerFlat[3 * l + 1];
              // the device call lists: 0 OT endcap, 1 barrel with stacked (OT) rods, 2 barrel with pixel rods. Not laid
              // out in the flat tables, or a pixel double disk (none in D121): the host searches it (marked, so that
              // every layer of the set has a row other than kNavResultNotComputed)
              if (first < 0 || (kind != 0 && kind != 2)) {
                out[(2 * t + d) * nL + l].det() = ::mkfitdev::kNavResultHost;
                continue;
              }
              const int list = kind == 0 ? 0 : (tables.flat.barrels[first].stacked ? 1 : 2);
              // at most one entry per (track, direction, layer) of the list's layers: the list can not overflow
              const int i = alpaka::atomicAdd(acc, nCalls + list, 1, alpaka::hierarchy::Blocks{});
              // constant indices: a runtime index into the kernel argument would copy it to the stack
              const int start =
                  list == 0 ? tables.listStart[0] : (list == 1 ? tables.listStart[1] : tables.listStart[2]);
              calls[size_t(2 * capacity) * start + i] = (2 * t + d) * nL + l;
            }
          }
        }
      }
    };

    // one thread per call of one list: memoLayerSearch (NavSearch.h), the only det-test site, inlined; the group-list
    // and det-test scratch is per thread (a thread runs its calls in turn). kStage: 0 the OT endcap list, 2 the pixel
    // barrel list, 1 / 3 the rods / the tilted rings of the OT barrel list (two kernels: one would spill on sm_89; the
    // rods part is kept per call in `parts` and combined with the rings part as barrelLayerSearch does)
    template <int kStage>
    struct KernelNavSearch {
      static constexpr int kList = kStage == 3 ? 1 : kStage;
      static constexpr int kKind = kStage == 0 ? 0 : (kStage == 1 ? 5 : (kStage == 2 ? 4 : 6));  // memoLayerSearch's
      // inlined into the kernel entry: an out-of-line body would spill its callee-saved registers
      ALPAKA_FN_ACC MKFITDEV_NAV_INLINE void operator()(Acc1D const& acc,
                                                        int capacity,
                                                        NavDeviceTables tables,
                                                        NavEstimator est,
                                                        nd::DetTestStart const* __restrict__ starts,
                                                        int const* __restrict__ calls,
                                                        int const* __restrict__ nCalls,
                                                        nd::NavGroups* __restrict__ arena,
                                                        nd::DetTestResult* __restrict__ memo,
                                                        int* __restrict__ memoDet,
                                                        NavPart* __restrict__ parts,
                                                        ::mkfitdev::NavResultSoAView out) const {
        const int nL = tables.nLayers;
        const int n = nCalls[kList];
        int const* list = calls + size_t(2 * capacity) * tables.listStart[kList];
        const size_t thread = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
        for (auto i : uniform_elements(acc, n)) {
          const int row = list[i];
          const int k = row / nL, l = row - k * nL;
          const int first = tables.layerFlat[3 * l + 1], count = tables.layerFlat[3 * l + 2];
          nd::NavCtx c{tables.flat,
                       starts + k / 2,
                       (k % 2) == 1,
                       est.maxSagitta,
                       est.minTolerance2,
                       est.nSigma,
                       est.maxDisplacement};
          c.arena = arena + thread * nd::kNavArena;
          c.memo = memo + thread * nd::kNavMemo;
          c.memoDet = memoDet + thread * nd::kNavMemo;
          nd::NavGroups res;
          const bool ok = nd::memoLayerSearch<kKind>(first, count, c, res, kStage == 3 ? parts[i].detTests : 0);
          int overflow = c.overflow | (ok ? 0 : int(nd::kNavUnsupported));
          int front = res.n == 0 ? int32_t(::mkfitdev::kNavResultEmpty) : res.g[0].first;
          if constexpr (kStage == 1) {
            parts[i] = NavPart{front, res.n, overflow, c.detTests};
            continue;
          }
          if constexpr (kStage == 3) {  // barrelLayerSearch: result = rods, then pushGroup of every rings group
            NavPart const p = parts[i];
            overflow |= p.overflow | (p.n + res.n > nd::kNavMaxGroups ? int(nd::kNavOvfGroups) : 0);
            if (p.n > 0)
              front = p.front;
          }
          out[row].det() = overflow != 0 ? int32_t(::mkfitdev::kNavResultHost) : front;
        }
      }
    };
  }  // namespace

  int navSearchThreads(int capacity) { return divide_up_by(2 * capacity, kNavSearchBlock) * kNavSearchBlock; }

  void launchNavDevice(Queue& queue,
                       ::mkfitdev::TrackSoAConstView trk,
                       int capacity,
                       NavDeviceTables const& tables,
                       NavEstimator const& est,
                       nd::DetTestStart* starts,
                       int* calls,
                       int* nCalls,
                       nd::NavGroups* arena,
                       nd::DetTestResult* memo,
                       int* memoDet,
                       NavPart* parts,
                       ::mkfitdev::NavResultSoAView out) {
    if (capacity <= 0)
      return;
    // double-precision kernels with few threads: small blocks spread them over the multiprocessors
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(divide_up_by(capacity, kNavSearchBlock), kNavSearchBlock),
                        KernelNavStart{},
                        trk,
                        capacity,
                        tables,
                        starts,
                        calls,
                        nCalls,
                        out);
    // the call counts stay on the device: each kind's grid covers 2 * capacity calls per sweep and strides over the rest
    const auto wd = make_workdiv<Acc1D>(divide_up_by(2 * capacity, kNavSearchBlock), kNavSearchBlock);
    // OT barrel rods, OT endcap, pixel barrel, then the OT barrel rings (they complete the rods' calls)
    alpaka::exec<Acc1D>(
        queue, wd, KernelNavSearch<1>{}, capacity, tables, est, starts, calls, nCalls, arena, memo, memoDet, parts, out);
    alpaka::exec<Acc1D>(
        queue, wd, KernelNavSearch<0>{}, capacity, tables, est, starts, calls, nCalls, arena, memo, memoDet, parts, out);
    alpaka::exec<Acc1D>(
        queue, wd, KernelNavSearch<2>{}, capacity, tables, est, starts, calls, nCalls, arena, memo, memoDet, parts, out);
    alpaka::exec<Acc1D>(
        queue, wd, KernelNavSearch<3>{}, capacity, tables, est, starts, calls, nCalls, arena, memo, memoDet, parts, out);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::navdevice

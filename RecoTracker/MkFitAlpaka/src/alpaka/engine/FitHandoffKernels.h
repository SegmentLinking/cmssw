#ifndef RecoTracker_MkFitAlpaka_src_alpaka_engine_FitHandoffKernels_h
#define RecoTracker_MkFitAlpaka_src_alpaka_engine_FitHandoffKernels_h

// Kernels of the building -> fit handoff (see FitHandoff.h). Instantiated only in src/alpaka/FitHandoff.dev.cc.

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelClusterSoA/interface/ClusteringConstants.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/math/MathUtils.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/engine/FitHandoff.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::handoff {

  using ::mkfitdev::TrackSoAConstView;
  using ::mkfitdev::TrackSoAView;

  // threads of the one-block index kernel on GPU backends (CPU backends: one thread walks all rows)
  constexpr int kSelThreads = 256;

  // MkFitFitProducer.cc:143-149, on the building output row (Track::pT() = |1 / invpT|, Track::momEta() =
  // getEta(theta) with the vdt approximations, Track::nTotalHits()).
  template <typename TAcc>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool passesCandCutSel(TAcc const& acc,
                                                       TrackSoAConstView in,
                                                       int i,
                                                       CandCutSel const& s) {
    if (!s.enabled)
      return true;
    const float momEta = ::mkfitdev::getEta(in[i].params().v[5]);
    const float minPtCutForCand =
        (s.minPtRelaxed > 0 && alpaka::math::abs(acc, momEta) > s.minAbsEtaRelaxed) ? s.minPtRelaxed : s.minPt;
    const float pT = alpaka::math::abs(acc, 1.f / in[i].params().v[3]);
    return !(pT < minPtCutForCand || in[i].nTotalHits() < s.minNHits);
  }

  // One block: thread k counts the passing rows of its contiguous chunk, thread 0 scans the counts (exclusive), each
  // thread writes the destination row of its rows (-1 = dropped). Input order is kept exactly.
  class KernelCandSelIndex {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc, TrackSoAConstView in, int capacity, CandCutSel sel, int32_t* dst, TrackSoAView out) const {
      auto& cnt = alpaka::declareSharedVar<int32_t[kSelThreads + 1], __COUNTER__>(acc);
      const int nThr = static_cast<int>(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
      const int tid = static_cast<int>(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      int nIn = in.nTracks();
      nIn = nIn < 0 ? 0 : (nIn > capacity ? capacity : nIn);
      const int chunk = (nIn + nThr - 1) / nThr;
      const int b = tid * chunk < nIn ? tid * chunk : nIn;
      const int e = b + chunk < nIn ? b + chunk : nIn;
      int c = 0;
      for (int i = b; i < e; ++i) {
        const bool ok = passesCandCutSel(acc, in, i, sel);
        dst[i] = ok ? 1 : -1;
        c += ok ? 1 : 0;
      }
      cnt[tid] = c;
      alpaka::syncBlockThreads(acc);
      if (tid == 0) {
        int s = 0;
        for (int k = 0; k < nThr; ++k) {
          const int x = cnt[k];
          cnt[k] = s;
          s += x;
        }
        out.nTracks() = s;
        out.nOverflowTracks() = in.nOverflowTracks();
        out.nOverflowHits() = in.nOverflowHits();
      }
      alpaka::syncBlockThreads(acc);
      int d = cnt[tid];
      for (int i = b; i < e; ++i)
        dst[i] = dst[i] > 0 ? d++ : -1;
    }
  };

  // Thread per input row: copy the kept rows to their destination (all columns of the row).
  class KernelCandSelCopy {
  public:
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc, TrackSoAConstView in, int capacity, const int32_t* dst, TrackSoAView out) const {
      int nIn = in.nTracks();
      nIn = nIn < 0 ? 0 : (nIn > capacity ? capacity : nIn);
      for (int32_t i : cms::alpakatools::uniform_elements(acc, nIn)) {
        const int d = dst[i];
        if (d < 0)
          continue;
        // element-wise copies (a whole-struct copy of the 256 B hit list spills registers)
        for (int k = 0; k < 6; ++k)
          out[d].params().v[k] = in[i].params().v[k];
        for (int k = 0; k < 21; ++k)
          out[d].errors().v[k] = in[i].errors().v[k];
        out[d].charge() = in[i].charge();
        out[d].chi2() = in[i].chi2();
        out[d].score() = in[i].score();
        out[d].label() = in[i].label();
        out[d].nTotalHits() = in[i].nTotalHits();
        out[d].nFoundHits() = in[i].nFoundHits();
        out[d].nSeedHits() = in[i].nSeedHits();
        out[d].etaRegion() = in[i].etaRegion();
        out[d].algorithm() = in[i].algorithm();
        out[d].nOverlaps() = in[i].nOverlaps();
        out[d].duplicate() = in[i].duplicate();
        // the hit list up to nTotalHits (every reader stops there; MkFitCore Track keeps only these)
        int nh = in[i].nTotalHits();
        nh = nh < 0 ? 0 : (nh > ::mkfitdev::kMaxTrkHits ? ::mkfitdev::kMaxTrkHits : nh);
        for (int h = 0; h < nh; ++h)
          out[d].hits().hot[h] = in[i].hits().hot[h];
      }
    }
  };

  // ---- device ClusterCpe (buildClusterCpe) ----
  constexpr int kAccretionMaxSize = 256;  // PixelClusterizerBase::AccretionCluster::MAXSIZE (later digis are dropped)
  constexpr int kMaxSpan = 255;           // SiPixelCluster::MAXSPAN (offsets and spans are capped)

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int32_t capSpan(int32_t v) { return v < kMaxSpan ? v : kMaxSpan; }

  // per SoA cluster, [cap] each
  struct ClusScratch {
    int32_t* cnt;  // legacy digis of the cluster (before the 256 cap)
    int32_t* xmin;
    int32_t* xmax;
    int32_t* ymin;
    int32_t* ymax;
    int32_t* charge;
    int32_t* qfX;
    int32_t* qlX;
    int32_t* qfY;
    int32_t* qlY;
    uint32_t cap;
  };

  // SiPixelDigisClustersFromSoAAlpaka::produce: the digis that enter a legacy cluster
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool legacyDigi(SiPixelDigisSoAConstView d, uint32_t i) {
    return d[i].rawIdArr() != 0 && d[i].adc() != 0 && d[i].moduleId() != ::pixelClustering::invalidModuleId &&
           d[i].clus() != ::pixelClustering::invalidClusterId && d[i].clus() >= 0;
  }

  // SoA cluster index of a legacy digi (cap = none: cluster id beyond the module's clusters or beyond the capacity)
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t digiCluster(SiPixelDigisSoAConstView d,
                                                      SiPixelClustersSoAConstView c,
                                                      uint32_t i,
                                                      uint32_t cap) {
    const uint32_t m = d[i].moduleId();
    const uint32_t b = c[m].clusModuleStart(), e = c[m + 1].clusModuleStart();
    const uint32_t g = b + static_cast<uint32_t>(d[i].clus());
    return (g < e && g < cap) ? g : cap;
  }

  class KernelClusInit {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, ClusScratch s) const {
      for (uint32_t g : cms::alpakatools::uniform_elements(acc, s.cap)) {
        s.cnt[g] = 0;
        s.xmin[g] = 0x7fffffff;
        s.xmax[g] = -1;
        s.ymin[g] = 0x7fffffff;
        s.ymax[g] = -1;
        s.charge[g] = 0;
        s.qfX[g] = 0;
        s.qlX[g] = 0;
        s.qfY[g] = 0;
        s.qlY[g] = 0;
      }
    }
  };

  class KernelClusCount {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SiPixelDigisSoAConstView d,
                                  uint32_t nDigis,
                                  SiPixelClustersSoAConstView c,
                                  ClusScratch s,
                                  ::mkfitdev::handoff::ClusterCpeCounters* cnt) const {
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, nDigis)) {
        if (!legacyDigi(d, i))
          continue;
        const uint32_t g = digiCluster(d, c, i, s.cap);
        if (g == s.cap) {
          alpaka::atomicAdd(acc, &cnt->nClusterOverflow, 1, alpaka::hierarchy::Blocks{});
          continue;
        }
        const int32_t before = alpaka::atomicAdd(acc, &s.cnt[g], 1, alpaka::hierarchy::Blocks{});
        if (before == kAccretionMaxSize)  // exactly once per big cluster
          alpaka::atomicAdd(acc, &cnt->nBigClusters, 1, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // AccretionCluster keeps the first 256 legacy digis of a cluster in digi order (the legacy converter walks the digis
  // in index order, a module's digis are contiguous). Only clusters with more than 256 digis need the rank.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool acceptedDigi(SiPixelDigisSoAConstView d, uint32_t i, int32_t cnt) {
    if (cnt <= kAccretionMaxSize)
      return true;
    const uint32_t raw = d[i].rawIdArr();
    const int32_t cl = d[i].clus();
    int rank = 0;
    for (int64_t j = int64_t(i) - 1; j >= 0; --j) {
      if (!legacyDigi(d, uint32_t(j)))
        continue;
      if (d[j].rawIdArr() != raw)
        break;
      if (d[j].clus() == cl && ++rank >= kAccretionMaxSize)
        return false;
    }
    return true;
  }

  class KernelClusMinMax {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SiPixelDigisSoAConstView d,
                                  uint32_t nDigis,
                                  SiPixelClustersSoAConstView c,
                                  ClusScratch s) const {
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, nDigis)) {
        if (!legacyDigi(d, i))
          continue;
        const uint32_t g = digiCluster(d, c, i, s.cap);
        if (g == s.cap || !acceptedDigi(d, i, s.cnt[g]))
          continue;
        const int32_t x = d[i].xx(), y = d[i].yy();
        alpaka::atomicMin(acc, &s.xmin[g], x, alpaka::hierarchy::Blocks{});
        alpaka::atomicMax(acc, &s.xmax[g], x, alpaka::hierarchy::Blocks{});
        alpaka::atomicMin(acc, &s.ymin[g], y, alpaka::hierarchy::Blocks{});
        alpaka::atomicMax(acc, &s.ymax[g], y, alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(acc, &s.charge[g], int32_t(d[i].adc()), alpaka::hierarchy::Blocks{});
      }
    }
  };

  // PixelCPEGenericBase::collect_edge_charges on the legacy SiPixelCluster: pixel x = minRow + min(x - xmin, 255),
  // maxRow = minRow + min(xmax - xmin, 255) (same for y).
  class KernelClusEdges {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SiPixelDigisSoAConstView d,
                                  uint32_t nDigis,
                                  SiPixelClustersSoAConstView c,
                                  ClusScratch s) const {
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, nDigis)) {
        if (!legacyDigi(d, i))
          continue;
        const uint32_t g = digiCluster(d, c, i, s.cap);
        if (g == s.cap || !acceptedDigi(d, i, s.cnt[g]))
          continue;
        const int32_t q = d[i].adc();
        const int32_t ox = capSpan(int32_t(d[i].xx()) - s.xmin[g]);
        const int32_t oy = capSpan(int32_t(d[i].yy()) - s.ymin[g]);
        const int32_t sx = capSpan(s.xmax[g] - s.xmin[g]);
        const int32_t sy = capSpan(s.ymax[g] - s.ymin[g]);
        if (ox == 0)
          alpaka::atomicAdd(acc, &s.qfX[g], q, alpaka::hierarchy::Blocks{});
        if (ox == sx)
          alpaka::atomicAdd(acc, &s.qlX[g], q, alpaka::hierarchy::Blocks{});
        if (oy == 0)
          alpaka::atomicAdd(acc, &s.qfY[g], q, alpaka::hierarchy::Blocks{});
        if (oy == sy)
          alpaka::atomicAdd(acc, &s.qlY[g], q, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // thread per mkFit pixel row: the legacy cluster key's SoA cluster -> ClusterCpe (host clusterCpe() field by field)
  class KernelClusGather {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SiPixelClustersSoAConstView c,
                                  ClusScratch s,
                                  const ::mkfitdev::handoff::ClusterRef* refs,
                                  uint32_t nRows,
                                  ::mkfitdev::cpe::ClusterCpe* out,
                                  ::mkfitdev::handoff::ClusterCpeCounters* cnt) const {
      for (uint32_t k : cms::alpakatools::uniform_elements(acc, nRows)) {
        ::mkfitdev::cpe::ClusterCpe r{};
        r.module = -1;
        const auto ref = refs[k];
        if (ref.module >= 0) {
          const uint32_t b = c[ref.module].clusModuleStart(), e = c[ref.module + 1].clusModuleStart();
          const uint32_t g = b + static_cast<uint32_t>(ref.ic);
          if (ref.ic < 0 || g >= e || g >= s.cap || s.cnt[g] == 0) {
            alpaka::atomicAdd(acc, &cnt->nRefOutOfRange, 1, alpaka::hierarchy::Blocks{});
          } else {
            r.module = ref.module;
            r.e.minRow = s.xmin[g];
            r.e.maxRow = s.xmin[g] + capSpan(s.xmax[g] - s.xmin[g]);
            r.e.minCol = s.ymin[g];
            r.e.maxCol = s.ymin[g] + capSpan(s.ymax[g] - s.ymin[g]);
            r.e.qfX = s.qfX[g];
            r.e.qlX = s.qlX[g];
            r.e.qfY = s.qfY[g];
            r.e.qlY = s.qlY[g];
            r.charge = static_cast<float>(s.charge[g]);
          }
        }
        out[k] = r;
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::handoff

#endif

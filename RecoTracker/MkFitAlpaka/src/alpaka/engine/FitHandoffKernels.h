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
                                                       CandCutSel const& cuts) {
    if (!cuts.enabled)
      return true;
    const float momEta = ::mkfitdev::getEta(in[i].params().v[5]);
    const float minPtCutForCand = (cuts.minPtRelaxed > 0 && alpaka::math::abs(acc, momEta) > cuts.minAbsEtaRelaxed)
                                      ? cuts.minPtRelaxed
                                      : cuts.minPt;
    const float pT = alpaka::math::abs(acc, 1.f / in[i].params().v[3]);
    return !(pT < minPtCutForCand || in[i].nTotalHits() < cuts.minNHits);
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
      const int chunk = cms::alpakatools::divide_up_by(nIn, nThr);
      const int begin = tid * chunk < nIn ? tid * chunk : nIn;
      const int end = begin + chunk < nIn ? begin + chunk : nIn;
      int nPass = 0;
      for (int i = begin; i < end; ++i) {
        const bool ok = passesCandCutSel(acc, in, i, sel);
        dst[i] = ok ? 1 : -1;
        nPass += ok ? 1 : 0;
      }
      cnt[tid] = nPass;
      alpaka::syncBlockThreads(acc);
      if (tid == 0) {
        int sum = 0;
        for (int k = 0; k < nThr; ++k) {
          const int count = cnt[k];
          cnt[k] = sum;
          sum += count;
        }
        out.nTracks() = sum;
        out.nOverflowTracks() = in.nOverflowTracks();
        out.nOverflowHits() = in.nOverflowHits();
      }
      alpaka::syncBlockThreads(acc);
      int next = cnt[tid];
      for (int i = begin; i < end; ++i)
        dst[i] = dst[i] > 0 ? next++ : -1;
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
        const int row = dst[i];
        if (row < 0)
          continue;
        // element-wise copies (a whole-struct copy of the 256 B hit list spills registers)
        for (int k = 0; k < 6; ++k)
          out[row].params().v[k] = in[i].params().v[k];
        for (int k = 0; k < 21; ++k)
          out[row].errors().v[k] = in[i].errors().v[k];
        out[row].charge() = in[i].charge();
        out[row].chi2() = in[i].chi2();
        out[row].score() = in[i].score();
        out[row].label() = in[i].label();
        out[row].nTotalHits() = in[i].nTotalHits();
        out[row].nFoundHits() = in[i].nFoundHits();
        out[row].nSeedHits() = in[i].nSeedHits();
        out[row].etaRegion() = in[i].etaRegion();
        out[row].algorithm() = in[i].algorithm();
        out[row].nOverlaps() = in[i].nOverlaps();
        out[row].duplicate() = in[i].duplicate();
        // the hit list up to nTotalHits (every reader stops there; MkFitCore Track keeps only these)
        int nh = in[i].nTotalHits();
        nh = nh < 0 ? 0 : (nh > ::mkfitdev::kMaxTrkHits ? ::mkfitdev::kMaxTrkHits : nh);
        for (int ih = 0; ih < nh; ++ih)
          out[row].hits().hot[ih] = in[i].hits().hot[ih];
      }
    }
  };

  // ---- device ClusterCpe (buildClusterCpe) ----
  constexpr int kAccretionMaxSize = 256;  // PixelClusterizerBase::AccretionCluster::MAXSIZE (later digis are dropped)
  constexpr int kMaxSpan = 255;           // SiPixelCluster::MAXSPAN (offsets and spans are capped)

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int32_t capSpan(int32_t span) { return span < kMaxSpan ? span : kMaxSpan; }

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
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool legacyDigi(SiPixelDigisSoAConstView digis, uint32_t i) {
    return digis[i].rawIdArr() != 0 && digis[i].adc() != 0 &&
           digis[i].moduleId() != ::pixelClustering::invalidModuleId &&
           digis[i].clus() != ::pixelClustering::invalidClusterId && digis[i].clus() >= 0;
  }

  // SoA cluster index of a legacy digi (cap = none: cluster id beyond the module's clusters or beyond the capacity)
  ALPAKA_FN_ACC ALPAKA_FN_INLINE uint32_t digiCluster(SiPixelDigisSoAConstView digis,
                                                      SiPixelClustersSoAConstView clusters,
                                                      uint32_t i,
                                                      uint32_t cap) {
    const uint32_t module = digis[i].moduleId();
    const uint32_t begin = clusters[module].clusModuleStart(), end = clusters[module + 1].clusModuleStart();
    const uint32_t cluster = begin + static_cast<uint32_t>(digis[i].clus());
    return (cluster < end && cluster < cap) ? cluster : cap;
  }

  class KernelClusInit {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, ClusScratch scratch) const {
      for (uint32_t cluster : cms::alpakatools::uniform_elements(acc, scratch.cap)) {
        scratch.cnt[cluster] = 0;
        scratch.xmin[cluster] = 0x7fffffff;
        scratch.xmax[cluster] = -1;
        scratch.ymin[cluster] = 0x7fffffff;
        scratch.ymax[cluster] = -1;
        scratch.charge[cluster] = 0;
        scratch.qfX[cluster] = 0;
        scratch.qlX[cluster] = 0;
        scratch.qfY[cluster] = 0;
        scratch.qlY[cluster] = 0;
      }
    }
  };

  class KernelClusCount {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SiPixelDigisSoAConstView digis,
                                  uint32_t nDigis,
                                  SiPixelClustersSoAConstView clusters,
                                  ClusScratch scratch,
                                  ::mkfitdev::handoff::ClusterCpeCounters* cnt) const {
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, nDigis)) {
        if (!legacyDigi(digis, i))
          continue;
        const uint32_t cluster = digiCluster(digis, clusters, i, scratch.cap);
        if (cluster == scratch.cap) {
          alpaka::atomicAdd(acc, &cnt->nClusterOverflow, 1, alpaka::hierarchy::Blocks{});
          continue;
        }
        alpaka::atomicAdd(acc, &scratch.cnt[cluster], 1, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // AccretionCluster keeps the first 256 legacy digis of a cluster in digi order (the legacy converter walks the digis
  // in index order, a module's digis are contiguous). Only clusters with more than 256 digis need the rank.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE bool acceptedDigi(SiPixelDigisSoAConstView digis, uint32_t i, int32_t cnt) {
    if (cnt <= kAccretionMaxSize)
      return true;
    const uint32_t raw = digis[i].rawIdArr();
    const int32_t cl = digis[i].clus();
    int rank = 0;
    for (int64_t j = int64_t(i) - 1; j >= 0; --j) {
      if (!legacyDigi(digis, uint32_t(j)))
        continue;
      if (digis[j].rawIdArr() != raw)
        break;
      if (digis[j].clus() == cl && ++rank >= kAccretionMaxSize)
        return false;
    }
    return true;
  }

  class KernelClusMinMax {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SiPixelDigisSoAConstView digis,
                                  uint32_t nDigis,
                                  SiPixelClustersSoAConstView clusters,
                                  ClusScratch scratch) const {
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, nDigis)) {
        if (!legacyDigi(digis, i))
          continue;
        const uint32_t cluster = digiCluster(digis, clusters, i, scratch.cap);
        if (cluster == scratch.cap || !acceptedDigi(digis, i, scratch.cnt[cluster]))
          continue;
        const int32_t row = digis[i].xx(), col = digis[i].yy();
        alpaka::atomicMin(acc, &scratch.xmin[cluster], row, alpaka::hierarchy::Blocks{});
        alpaka::atomicMax(acc, &scratch.xmax[cluster], row, alpaka::hierarchy::Blocks{});
        alpaka::atomicMin(acc, &scratch.ymin[cluster], col, alpaka::hierarchy::Blocks{});
        alpaka::atomicMax(acc, &scratch.ymax[cluster], col, alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(acc, &scratch.charge[cluster], int32_t(digis[i].adc()), alpaka::hierarchy::Blocks{});
      }
    }
  };

  // PixelCPEGenericBase::collect_edge_charges on the legacy SiPixelCluster: pixel x = minRow + min(x - xmin, 255),
  // maxRow = minRow + min(xmax - xmin, 255) (same for y).
  class KernelClusEdges {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SiPixelDigisSoAConstView digis,
                                  uint32_t nDigis,
                                  SiPixelClustersSoAConstView clusters,
                                  ClusScratch scratch) const {
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, nDigis)) {
        if (!legacyDigi(digis, i))
          continue;
        const uint32_t cluster = digiCluster(digis, clusters, i, scratch.cap);
        if (cluster == scratch.cap || !acceptedDigi(digis, i, scratch.cnt[cluster]))
          continue;
        const int32_t adc = digis[i].adc();
        const int32_t ox = capSpan(int32_t(digis[i].xx()) - scratch.xmin[cluster]);
        const int32_t oy = capSpan(int32_t(digis[i].yy()) - scratch.ymin[cluster]);
        const int32_t sx = capSpan(scratch.xmax[cluster] - scratch.xmin[cluster]);
        const int32_t sy = capSpan(scratch.ymax[cluster] - scratch.ymin[cluster]);
        if (ox == 0)
          alpaka::atomicAdd(acc, &scratch.qfX[cluster], adc, alpaka::hierarchy::Blocks{});
        if (ox == sx)
          alpaka::atomicAdd(acc, &scratch.qlX[cluster], adc, alpaka::hierarchy::Blocks{});
        if (oy == 0)
          alpaka::atomicAdd(acc, &scratch.qfY[cluster], adc, alpaka::hierarchy::Blocks{});
        if (oy == sy)
          alpaka::atomicAdd(acc, &scratch.qlY[cluster], adc, alpaka::hierarchy::Blocks{});
      }
    }
  };

  // thread per mkFit pixel row: the legacy cluster key's SoA cluster -> ClusterCpe (host clusterCpe() field by field)
  class KernelClusGather {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SiPixelClustersSoAConstView clusters,
                                  ClusScratch scratch,
                                  const ::mkfitdev::handoff::ClusterRef* refs,
                                  uint32_t nRows,
                                  ::mkfitdev::cpe::ClusterCpe* out,
                                  ::mkfitdev::handoff::ClusterCpeCounters* cnt) const {
      for (uint32_t k : cms::alpakatools::uniform_elements(acc, nRows)) {
        ::mkfitdev::cpe::ClusterCpe cpe{};
        cpe.module = -1;
        const auto ref = refs[k];
        if (ref.module >= 0) {
          const uint32_t begin = clusters[ref.module].clusModuleStart(),
                         end = clusters[ref.module + 1].clusModuleStart();
          const uint32_t cluster = begin + static_cast<uint32_t>(ref.clusterInModule);
          if (ref.clusterInModule < 0 || cluster >= end || cluster >= scratch.cap || scratch.cnt[cluster] == 0) {
            alpaka::atomicAdd(acc, &cnt->nRefOutOfRange, 1, alpaka::hierarchy::Blocks{});
          } else {
            cpe.module = ref.module;
            cpe.edges.minRow = scratch.xmin[cluster];
            cpe.edges.maxRow = scratch.xmin[cluster] + capSpan(scratch.xmax[cluster] - scratch.xmin[cluster]);
            cpe.edges.minCol = scratch.ymin[cluster];
            cpe.edges.maxCol = scratch.ymin[cluster] + capSpan(scratch.ymax[cluster] - scratch.ymin[cluster]);
            cpe.edges.qfX = scratch.qfX[cluster];
            cpe.edges.qlX = scratch.qlX[cluster];
            cpe.edges.qfY = scratch.qfY[cluster];
            cpe.edges.qlY = scratch.qlY[cluster];
            cpe.charge = static_cast<float>(scratch.charge[cluster]);
          }
        }
        out[k] = cpe;
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::handoff

#endif

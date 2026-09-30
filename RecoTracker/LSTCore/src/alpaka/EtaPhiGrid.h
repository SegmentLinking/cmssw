#ifndef RecoTracker_LSTCore_src_alpaka_EtaPhiGrid_h
#define RecoTracker_LSTCore_src_alpaka_EtaPhiGrid_h

#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/warpsize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  // Cells at least `window` wide (phi periodic, eta clamped into the edge cells): for finite eta and phi, the 3x3
  // neighbourhood of an object holds all its |dEta|, |dPhi| <= window partners. Filled by count -> ScanCellCounts -> fill.
  struct EtaPhiGrid {
    // |eta| beyond kEtaMax goes into the edge cells; it only sets the number of cells.
    static constexpr float kEtaMax = 3.f;
    static constexpr int kNeighbourCells = 9;

    int nEta;
    int nPhi;
    float etaMin;
    float invWidth;

    // nPhi = the largest number of cells of width >= window; the eta cells have the same width.
    static EtaPhiGrid make(float window) {
      // At least 3 phi cells, so that the 3x3 neighbourhood visits distinct cells.
      if (!(window > 0.f && 3.f * window <= 2.f * kPi))
        lstLogicError("EtaPhiGrid: the window must be in (0, 2 pi / 3]");
      EtaPhiGrid grid;
      grid.nPhi = static_cast<int>(2.f * kPi / window);
      const float width = 2.f * kPi / grid.nPhi;
      grid.invWidth = 1.f / width;
      grid.nEta = static_cast<int>(2.f * kEtaMax * grid.invWidth) + 1;
      grid.etaMin = -0.5f * grid.nEta * width;
      return grid;
    }

    ALPAKA_FN_HOST_ACC unsigned int nCells() const { return static_cast<unsigned int>(nEta * nPhi); }

    // alpaka min / max follow fmin / fmax, so a NaN eta goes to the first cell (no undefined conversion).
    template <typename TAcc>
    ALPAKA_FN_ACC int etaBin(TAcc const& acc, float eta) const {
      const float bin = alpaka::math::floor(acc, (eta - etaMin) * invWidth);
      return static_cast<int>(alpaka::math::min(acc, alpaka::math::max(acc, bin, 0.f), static_cast<float>(nEta - 1)));
    }

    template <typename TAcc>
    ALPAKA_FN_ACC int phiBin(TAcc const& acc, float phi) const {
      const float bin = alpaka::math::floor(acc, (phi + kPi) * invWidth);
      // A non-finite phi (NaN fails the comparison) goes to the first cell instead of an undefined conversion.
      if (!(alpaka::math::abs(acc, bin) < 2.f * nPhi))
        return 0;
      return wrapPhiBin(static_cast<int>(bin) % nPhi);
    }

    // Phi bin index modulo nPhi, for arguments in (-nPhi, 2 nPhi).
    ALPAKA_FN_HOST_ACC int wrapPhiBin(int bin) const { return bin < 0 ? bin + nPhi : (bin >= nPhi ? bin - nPhi : bin); }

    ALPAKA_FN_HOST_ACC int cell(int etaBin, int phiBin) const { return etaBin * nPhi + phiBin; }

    // Cell iNeighbour in [0, kNeighbourCells) of the 3x3 neighbourhood of (etaBin, phiBin), or -1 past an eta edge.
    ALPAKA_FN_HOST_ACC int neighbourCell(int etaBin, int phiBin, int iNeighbour) const {
      const int neighbourEtaBin = etaBin + iNeighbour / 3 - 1;
      if (neighbourEtaBin < 0 || neighbourEtaBin >= nEta)
        return -1;
      return cell(neighbourEtaBin, wrapPhiBin(phiBin + iNeighbour % 3 - 1));
    }

    template <typename TAcc>
    ALPAKA_FN_ACC int cell(TAcc const& acc, float eta, float phi) const {
      return cell(etaBin(acc, eta), phiBin(acc, phi));
    }
  };

  // One block of kBlockThreads: cellStart = exclusive prefix sum of cellCount plus the total at [nCells], and
  // cellCount becomes the fill cursor. Each thread sums a contiguous chunk of cells; a block scan gives their offsets.
  struct ScanCellCounts {
    // One chunk per thread, at most warpSize^2 for blockPrefixScan.
    static constexpr unsigned int kBlockThreads = 1024;
    static_assert(cms::alpakatools::warpSize == 1 ||
                      kBlockThreads <= cms::alpakatools::warpSize * cms::alpakatools::warpSize,
                  "blockPrefixScan takes at most warpSize^2 elements");

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  unsigned int nCells,
                                  unsigned int* __restrict__ cellCount,
                                  unsigned int* __restrict__ cellStart) const {
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));
      auto& chunkSums = alpaka::declareSharedVar<unsigned int[kBlockThreads], __COUNTER__>(acc);
      auto& scanWorkspace = alpaka::declareSharedVar<unsigned int[cms::alpakatools::warpSize], __COUNTER__>(acc);
      const unsigned int nThreads = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      const unsigned int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
      ALPAKA_ASSERT_ACC(nThreads <= kBlockThreads);

      const unsigned int chunk = cms::alpakatools::divide_up_by(nCells, nThreads);
      const unsigned int begin = cms::alpakatools::idx_min(tid * chunk, nCells);
      const unsigned int end = cms::alpakatools::idx_min(begin + chunk, nCells);
      unsigned int sum = 0;
      for (unsigned int c = begin; c < end; ++c)
        sum += cellCount[c];
      chunkSums[tid] = sum;
      alpaka::syncBlockThreads(acc);
      cms::alpakatools::blockPrefixScan(acc, chunkSums, static_cast<int32_t>(nThreads), scanWorkspace);  // inclusive
      if (tid == nThreads - 1)
        cellStart[nCells] = chunkSums[tid];
      unsigned int offset = chunkSums[tid] - sum;
      for (unsigned int c = begin; c < end; ++c) {
        const unsigned int count = cellCount[c];
        cellStart[c] = offset;
        cellCount[c] = offset;
        offset += count;
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif

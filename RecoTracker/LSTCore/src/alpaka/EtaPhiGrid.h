#ifndef RecoTracker_LSTCore_src_alpaka_EtaPhiGrid_h
#define RecoTracker_LSTCore_src_alpaka_EtaPhiGrid_h

#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  // Eta-phi cell grid for windowed pair searches (|dEta| <= window and |dPhi| <= window). Cells are at least
  // `window` wide in eta and in phi (phi periodic, eta clamped into the edge cells), so both objects of a pair
  // inside the window sit in the same or in adjacent cells: the 3x3 neighbourhood of an object holds a superset
  // of its window partners. Filled by count -> exclusive prefix -> scatter (no sort); the order inside a cell is
  // arbitrary, so it suits decisions that do not depend on the visiting order.
  // Usage: cellCount[nCells()] zeroed; count (atomicAdd on cellCount[cell]); EtaPhiGridPrefix (cellStart, and
  // cellCount becomes the scatter cursor); scatter (cellItems[atomicAdd(cellCount[cell])] = object); then scan
  // cellItems[cellStart[c] .. cellStart[c + 1]) for the neighbour cells c.
  struct EtaPhiGrid {
    int nEta;
    int nPhi;
    float etaMin;
    float invWidth;

    // nPhi = the largest number of cells of width >= window over 2 pi; the eta cells get the same width.
    // Eta beyond +-etaMax goes to the edge cells (still a superset; etaMax only sets the cell count).
    static EtaPhiGrid make(float window, float etaMax) {
      EtaPhiGrid grid;
      grid.nPhi = static_cast<int>(2.f * kPi / window);
      const float width = 2.f * kPi / grid.nPhi;
      grid.invWidth = 1.f / width;
      grid.nEta = static_cast<int>(2.f * etaMax * grid.invWidth) + 1;
      grid.etaMin = -0.5f * grid.nEta * width;
      return grid;
    }

    ALPAKA_FN_HOST_ACC int nCells() const { return nEta * nPhi; }

    template <typename TAcc>
    ALPAKA_FN_ACC int etaBin(TAcc const& acc, float eta) const {
      const float bin = alpaka::math::floor(acc, (eta - etaMin) * invWidth);
      return static_cast<int>(alpaka::math::min(acc, alpaka::math::max(acc, bin, 0.f), static_cast<float>(nEta - 1)));
    }

    template <typename TAcc>
    ALPAKA_FN_ACC int phiBin(TAcc const& acc, float phi) const {
      return wrapPhiBin(static_cast<int>(alpaka::math::floor(acc, (phi + kPi) * invWidth)) % nPhi);
    }

    // Phi bin index modulo nPhi, for arguments in (-nPhi, 2 nPhi).
    ALPAKA_FN_HOST_ACC int wrapPhiBin(int bin) const { return bin < 0 ? bin + nPhi : (bin >= nPhi ? bin - nPhi : bin); }

    ALPAKA_FN_HOST_ACC int cell(int etaBin, int phiBin) const { return etaBin * nPhi + phiBin; }

    template <typename TAcc>
    ALPAKA_FN_ACC int cell(TAcc const& acc, float eta, float phi) const {
      return cell(etaBin(acc, eta), phiBin(acc, phi));
    }
  };

  // Single block of <= 1024 threads (a multiple of the warp size): cellStart[c] = exclusive prefix of cellCount over
  // the cells, cellStart[nCells] = total; cellCount is overwritten with cellStart (the scatter cursor).
  struct EtaPhiGridPrefix {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  int nCells,
                                  unsigned int* __restrict__ cellCount,
                                  unsigned int* __restrict__ cellStart) const {
      ALPAKA_ASSERT_ACC((alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0] == 1));
      constexpr unsigned int kMaxThreads = 1024;
      auto& partial = alpaka::declareSharedVar<unsigned int[kMaxThreads], __COUNTER__>(acc);
      auto& ws = alpaka::declareSharedVar<unsigned int[32], __COUNTER__>(acc);
      const unsigned int nThreads = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      const unsigned int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
      ALPAKA_ASSERT_ACC(nThreads <= kMaxThreads);

      // Each thread sums a contiguous chunk of cells; an inclusive scan over the threads gives the chunk offsets.
      const unsigned int nGridCells = static_cast<unsigned int>(nCells);
      const unsigned int chunk = cms::alpakatools::divide_up_by(nGridCells, nThreads);
      const unsigned int begin = cms::alpakatools::idx_min(tid * chunk, nGridCells);
      const unsigned int end = cms::alpakatools::idx_min(begin + chunk, nGridCells);
      unsigned int sum = 0;
      for (unsigned int c = begin; c < end; ++c)
        sum += cellCount[c];
      partial[tid] = sum;
      alpaka::syncBlockThreads(acc);
      cms::alpakatools::blockPrefixScan(acc, partial, static_cast<int32_t>(nThreads), ws);
      if (tid == nThreads - 1)
        cellStart[nGridCells] = partial[tid];
      unsigned int offset = partial[tid] - sum;
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
